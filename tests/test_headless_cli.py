from __future__ import annotations

import json
import importlib.util
import os
import pathlib
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from ouroboros.gateway.tasks import (
    _compose_task_text,
    _resolve_workspace_root,
    api_task_artifact,
    api_task_events,
    api_task_get,
    api_tasks_create,
    api_tasks_list,
    iter_task_events,
)
from ouroboros.headless import (
    ARTIFACT_STATUS_FAILED,
    ARTIFACT_STATUS_FINALIZING,
    ARTIFACT_STATUS_READY,
    ARTIFACT_STATUS_READY_WITH_CHANGES,
    _incidental_lockfile_excludes,
    build_memory_export,
    build_workspace_patch,
    finalize_task_artifacts,
    prune_headless_task_drives,
    prune_task_drives,
    task_artifacts_dir,
    write_workspace_patch_artifacts,
)
from ouroboros.task_results import write_task_result
from ouroboros.tools.core import _repo_read
from ouroboros.tools.registry import ToolContext, ToolRegistry
from ouroboros.utils import utc_now_iso
from ouroboros.workspace_preflight import _infer_tools_from_manifests


def _init_repo_with_file(repo, name="tracked.txt", content="old\n"):
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    (repo / name).write_text(content, encoding="utf-8")
    subprocess.run(["git", "add", name], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "-c", "user.email=t@example.com", "-c", "user.name=T", "commit", "-m", "init"],
        cwd=repo,
        check=True,
        capture_output=True,
    )


def test_task_api_enqueue_workspace_creates_child_drive(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    subprocess.run(["git", "init"], cwd=workspace, check=True, capture_output=True)
    repo = tmp_path / "repo"
    repo.mkdir()
    data = tmp_path / "data"
    (data / "memory").mkdir(parents=True)
    (data / "memory" / "identity.md").write_text("seed identity", encoding="utf-8")

    captured = []
    bootstrapped = []

    def fake_enqueue(task):
        captured.append(dict(task))
        return task

    monkeypatch.setattr("supervisor.queue.enqueue_task", fake_enqueue)
    monkeypatch.setattr("supervisor.queue.persist_queue_snapshot", lambda reason="": None)
    monkeypatch.setattr("ouroboros.workspace_admission.bootstrap_process_path", lambda: bootstrapped.append(True) or [])

    app = Starlette(routes=[Route("/api/tasks", endpoint=api_tasks_create, methods=["POST"])])
    app.state.drive_root = data
    app.state.repo_dir = repo
    response = TestClient(app).post(
        "/api/tasks",
        json={
            "description": "fix it",
            "workspace_root": str(workspace),
            "memory_mode": "forked",
            "expected_output": "A workspace patch and concise handoff.",
            "constraints": "No network.",
            "allowed_resources": {"web": False, "network": False},
            "resource_policy": {
                "protected_artifacts": [
                    {
                        "id": "reference",
                        "role": "black_box_reference",
                        "paths": ["reference.bin"],
                        "allow": ["execute"],
                    }
                ]
            },
            "deadline_at": "2026-06-04T12:00:00Z",
            "service_teardown": "keep",
            "context_requires_self_body_docs": "false",
            "metadata": {
                "root_task_id": "forged-root",
                "parent_task_id": "forged-parent",
                "delegation_role": "root",
                "child_drive_root": "/tmp/forged-child",
            },
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["task_id"]
    assert bootstrapped
    assert captured and captured[0]["workspace_root"] == str(workspace.resolve(strict=False))
    assert captured[0]["deadline_at"] == "2026-06-04T12:00:00Z"
    assert captured[0]["metadata"]["service_teardown"] == "keep"
    assert captured[0]["allowed_resources"] == {"web": False, "network": False}
    assert captured[0]["context_requires_self_body_docs"] is False
    assert captured[0]["task_contract"]["expected_output"] == "A workspace patch and concise handoff."
    assert captured[0]["task_contract"]["constraints"] == "No network."
    assert captured[0]["task_contract"]["context_requires_self_body_docs"] is False
    assert captured[0]["task_contract"]["resource_policy"]["protected_artifacts"][0]["paths"] == ["reference.bin"]
    child_drive = captured[0]["drive_root"]
    assert child_drive
    assert (tmp_path / "data" / "task_results" / f"{payload['task_id']}.json").is_file()
    assert "seed identity" in (data / "state" / "headless_tasks" / payload["task_id"] / "data" / "memory" / "identity.md").read_text(encoding="utf-8")
    result = json.loads((data / "task_results" / f"{payload['task_id']}.json").read_text(encoding="utf-8"))
    assert result["artifact_status"] == "pending"
    assert captured[0]["root_task_id"] == payload["task_id"]
    assert captured[0]["parent_task_id"] is None
    assert captured[0]["delegation_role"] == "root"
    assert result["metadata"]["root_task_id"] == payload["task_id"]
    assert result["metadata"]["parent_task_id"] == ""
    assert result["metadata"]["delegation_role"] == "root"
    assert result["task_contract"]["deadline_at"] == "2026-06-04T12:00:00Z"
    assert result["task_contract"]["allowed_resources"] == {"web": False, "network": False}
    assert result["task_contract"]["resource_policy"]["protected_artifacts"][0]["id"] == "reference"
    assert result["metadata"]["child_drive_root"] == captured[0]["child_drive_root"]
    assert "/tmp/forged-child" not in json.dumps(result["metadata"])
    assert result["metadata"]["workspace_preflight"]["git"]["head"] == ""
    assert any(item["kind"] == "workspace_preflight" for item in result["artifacts"])
    assert "workspace_preflight:" in captured[0]["text"]
    assert "target workspace, not the Ouroboros system repo" in captured[0]["text"]


def test_api_tasks_create_requires_description_not_legacy_aliases(monkeypatch):
    captured = []
    monkeypatch.setattr("supervisor.queue.enqueue_task", lambda task: captured.append(task) or task)
    app = Starlette(routes=[Route("/api/tasks", endpoint=api_tasks_create, methods=["POST"])])
    client = TestClient(app)

    for payload in ({"text": "legacy task"}, {"prompt": "legacy task"}, {"description": ""}):
        response = client.post("/api/tasks", json=payload)
        assert response.status_code == 400, (payload, response.text)
        assert "description is required" in response.json().get("error", "")

    response = client.post("/api/tasks", json={"description": "x", "service_teardown": "detach"})
    assert response.status_code == 400
    assert "service_teardown" in response.json().get("error", "")

    assert captured == []


def test_api_tasks_create_rejects_internal_task_types(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    data = tmp_path / "data"
    (data / "memory").mkdir(parents=True)

    monkeypatch.setattr("supervisor.queue.enqueue_task", lambda task: task)
    monkeypatch.setattr("supervisor.queue.persist_queue_snapshot", lambda reason="": None)
    monkeypatch.setattr("ouroboros.workspace_admission.bootstrap_process_path", lambda: [])

    app = Starlette(routes=[Route("/api/tasks", endpoint=api_tasks_create, methods=["POST"])])
    app.state.drive_root = data
    app.state.repo_dir = repo
    client = TestClient(app)

    for internal_type in ("evolution", "review", "deep_self_review"):
        resp = client.post("/api/tasks", json={"description": "x", "type": internal_type})
        assert resp.status_code == 400, (internal_type, resp.text)
        assert "internal" in resp.json().get("error", "").lower()

    # A normal task type is still accepted.
    ok = client.post("/api/tasks", json={"description": "do normal work", "type": "task"})
    assert ok.status_code == 200, ok.text


def test_compose_task_text_extends_existing_headless_workspace_block(tmp_path):
    text = _compose_task_text(
        "fix\n\n[HEADLESS_WORKSPACE]\nexisting: yes\n[END_HEADLESS_WORKSPACE]",
        workspace_root=tmp_path,
        workspace_mode="external",
        memory_mode="empty",
        workspace_preflight={"error": "probe failed"},
        attachments=[],
    )

    assert text.count("[HEADLESS_WORKSPACE]") == 1
    assert "existing: yes" in text
    assert "preflight_error: probe failed" in text
    assert text.index("workspace_root:") < text.index("[END_HEADLESS_WORKSPACE]")


def test_task_api_rejects_unsafe_task_id_and_system_workspace(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    subprocess.run(["git", "init"], cwd=workspace, check=True, capture_output=True)
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setattr("supervisor.queue.enqueue_task", lambda task: task)
    monkeypatch.setattr("supervisor.queue.persist_queue_snapshot", lambda reason="": None)

    app = Starlette(routes=[Route("/api/tasks", endpoint=api_tasks_create, methods=["POST"])])
    app.state.drive_root = data
    app.state.repo_dir = repo
    client = TestClient(app)

    bad_id = client.post("/api/tasks", json={"description": "x", "task_id": "../settings", "workspace_root": str(workspace)})
    assert bad_id.status_code == 400
    assert not (data / "settings.json").exists()

    system_repo = client.post("/api/tasks", json={"description": "x", "workspace_root": str(repo)})
    assert system_repo.status_code == 400
    assert "system repo" in system_repo.json()["error"]

    bad_numbers = client.post("/api/tasks", json={"description": "x", "chat_id": "not-int", "workspace_root": str(workspace)})
    assert bad_numbers.status_code == 400
    bad_deadline = client.post("/api/tasks", json={"description": "x", "deadline_at": "not-a-date", "workspace_root": str(workspace)})
    assert bad_deadline.status_code == 400
    assert "deadline_at" in bad_deadline.json()["error"]
    naive_deadline = client.post("/api/tasks", json={"description": "x", "deadline_at": "2026-06-04T12:00:00", "workspace_root": str(workspace)})
    assert naive_deadline.status_code == 400
    assert "timezone" in naive_deadline.json()["error"]

    first = client.post("/api/tasks", json={"description": "x", "task_id": "fixed1", "workspace_root": str(workspace)})
    assert first.status_code == 200
    duplicate = client.post("/api/tasks", json={"description": "x", "task_id": "fixed1", "workspace_root": str(workspace)})
    assert duplicate.status_code == 409

    typed = client.post("/api/tasks", json={"description": "x", "type": "deep_self_review", "workspace_root": str(workspace)})
    assert typed.status_code == 400


def test_resolve_workspace_root_blocks_case_variant_control_plane(tmp_path):
    system_repo = tmp_path / "Ouroboros" / "repo"
    drive = tmp_path / "Ouroboros" / "data"
    workspace_repo_case = tmp_path / "ouroboros" / "repo"
    workspace_data_case = tmp_path / "ouroboros" / "data" / "workspace"
    for path in (system_repo, drive / "workspace"):
        path.mkdir(parents=True)

    with pytest.raises(ValueError, match="Ouroboros system repo"):
        _resolve_workspace_root(workspace_repo_case, system_repo_dir=system_repo, drive_root=drive)
    with pytest.raises(ValueError, match="Ouroboros data drive"):
        _resolve_workspace_root(workspace_data_case, system_repo_dir=system_repo, drive_root=drive)


def test_task_api_rejects_forged_subagent_without_child_drive_side_effect(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    subprocess.run(["git", "init"], cwd=workspace, check=True, capture_output=True)
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setattr("supervisor.queue.enqueue_task", lambda task: pytest.fail("forged subagent enqueued"))
    monkeypatch.setattr("supervisor.queue.persist_queue_snapshot", lambda reason="": None)

    app = Starlette(routes=[Route("/api/tasks", endpoint=api_tasks_create, methods=["POST"])])
    app.state.drive_root = data
    app.state.repo_dir = repo
    client = TestClient(app)

    top_level = client.post(
        "/api/tasks",
        json={"description": "x", "task_id": "forged1", "workspace_root": str(workspace), "delegation_role": "subagent"},
    )
    metadata = client.post(
        "/api/tasks",
        json={"description": "x", "task_id": "forged2", "workspace_root": str(workspace), "metadata": {"delegation_role": "subagent"}},
    )

    assert top_level.status_code == 400
    assert metadata.status_code == 400
    assert "internal schedule_subagent" in top_level.json()["error"]
    assert not (data / "state" / "headless_tasks" / "forged1").exists()
    assert not (data / "state" / "headless_tasks" / "forged2").exists()


def test_task_api_rejects_external_lineage_forgery(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    subprocess.run(["git", "init"], cwd=workspace, check=True, capture_output=True)
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setattr("supervisor.queue.enqueue_task", lambda task: pytest.fail("forged lineage enqueued"))
    monkeypatch.setattr("supervisor.queue.persist_queue_snapshot", lambda reason="": None)

    app = Starlette(routes=[Route("/api/tasks", endpoint=api_tasks_create, methods=["POST"])])
    app.state.drive_root = data
    app.state.repo_dir = repo

    response = TestClient(app).post(
        "/api/tasks",
        json={
            "description": "x",
            "workspace_root": str(workspace),
            "parent_task_id": "parent1",
            "root_task_id": "root1",
        },
    )

    assert response.status_code == 400
    assert "internal lineage fields" in response.json()["error"]
    assert not list((data / "task_results").glob("*.json"))


def test_task_api_preserves_top_level_actor_id_after_metadata_sanitization(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    subprocess.run(["git", "init"], cwd=workspace, check=True, capture_output=True)
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    data = tmp_path / "data"
    data.mkdir()
    captured = []
    monkeypatch.setattr("supervisor.queue.enqueue_task", lambda task: captured.append(dict(task)) or task)
    monkeypatch.setattr("supervisor.queue.persist_queue_snapshot", lambda reason="": None)

    app = Starlette(routes=[Route("/api/tasks", endpoint=api_tasks_create, methods=["POST"])])
    app.state.drive_root = data
    app.state.repo_dir = repo

    response = TestClient(app).post(
        "/api/tasks",
        json={
            "description": "x",
            "workspace_root": str(workspace),
            "memory_mode": "forked",
            "actor_id": "operator-1",
            "metadata": {"actor_id": "forged-metadata"},
        },
    )

    assert response.status_code == 200
    assert captured[0]["actor_id"] == "operator-1"
    result = json.loads((data / "task_results" / f"{response.json()['task_id']}.json").read_text(encoding="utf-8"))
    assert result["metadata"]["actor_id"] == "operator-1"
    assert "forged-metadata" not in json.dumps(result)


def test_task_event_replay_uses_existing_logs_and_result(tmp_path):
    data = tmp_path / "data"
    logs = data / "logs"
    logs.mkdir(parents=True)
    task_id = "abc123"
    (logs / "progress.jsonl").write_text(
        json.dumps({"ts": "2026-01-01T00:00:00Z", "task_id": task_id, "content": "working"}) + "\n",
        encoding="utf-8",
    )
    result_dir = data / "task_results"
    result_dir.mkdir()
    (result_dir / f"{task_id}.json").write_text(
        json.dumps({"task_id": task_id, "status": "completed", "result": "done", "ts": "2026-01-01T00:00:01Z"}),
        encoding="utf-8",
    )

    events = iter_task_events(data, task_id)

    assert [event["type"] for event in events] == ["progress", "task_result"]
    assert events[0]["seq"] == 1
    assert events[1]["data"]["result"] == "done"


def test_task_event_replay_parent_includes_child_lineage_events(tmp_path):
    data = tmp_path / "data"
    logs = data / "logs"
    logs.mkdir(parents=True)
    parent_id = "parent1"
    child_id = "child1"
    (logs / "progress.jsonl").write_text(
        "\n".join([
            json.dumps({"ts": "2026-01-01T00:00:00Z", "task_id": parent_id, "content": "parent"}),
            json.dumps({
                "ts": "2026-01-01T00:00:01Z",
                "task_id": child_id,
                "parent_task_id": parent_id,
                "root_task_id": parent_id,
                "delegation_role": "subagent",
                "subagent_task_id": child_id,
                "content": "child progress",
            }),
        ]) + "\n",
        encoding="utf-8",
    )
    write_task_result(
        data,
        parent_id,
        "running",
        result="parent pending",
        ts="2026-01-01T00:00:00Z",
    )
    write_task_result(
        data,
        child_id,
        "running",
        result="child pending",
        parent_task_id=parent_id,
        root_task_id=parent_id,
        delegation_role="subagent",
        ts="2026-01-01T00:00:01Z",
    )

    events = iter_task_events(data, parent_id)

    progress_events = [event for event in events if event["type"] == "progress"]
    assert [event["task_id"] for event in progress_events] == [parent_id, child_id]
    assert progress_events[1]["data"]["content"] == "child progress"


def test_logs_tail_parent_filter_includes_child_lineage_events(tmp_path):
    from ouroboros.gateway.logs import api_logs_tail

    data = tmp_path / "data"
    logs = data / "logs"
    logs.mkdir(parents=True)
    (logs / "progress.jsonl").write_text(
        "\n".join([
            json.dumps({"ts": "2026-01-01T00:00:00Z", "task_id": "parent1", "content": "parent"}),
            json.dumps({
                "ts": "2026-01-01T00:00:01Z",
                "task_id": "child1",
                "subagent_task_id": "child1",
                "parent_task_id": "parent1",
                "root_task_id": "parent1",
                "delegation_role": "subagent",
                "content": "child",
            }),
            json.dumps({"ts": "2026-01-01T00:00:02Z", "task_id": "other", "content": "other"}),
        ]) + "\n",
        encoding="utf-8",
    )
    app = Starlette(routes=[Route("/api/logs/{name}", endpoint=api_logs_tail, methods=["GET"])])
    app.state.drive_root = data

    response = TestClient(app).get("/api/logs/progress?task_id=parent1&limit=10")
    payload = response.json()

    assert response.status_code == 200
    assert [row["content"] for row in payload["entries"]] == ["parent", "child"]


def test_workspace_event_replay_suppresses_task_done_until_artifacts_terminal(tmp_path):
    data = tmp_path / "data"
    logs = data / "logs"
    logs.mkdir(parents=True)
    task_id = "abc123"
    (logs / "events.jsonl").write_text(
        json.dumps({"ts": "2026-01-01T00:00:01Z", "type": "task_done", "task_id": task_id}) + "\n",
        encoding="utf-8",
    )
    write_task_result(
        data,
        task_id,
        "completed",
        workspace_root=str(tmp_path / "workspace"),
        artifact_status="finalizing",
        child_status="completed",
    )

    events = iter_task_events(data, task_id)

    assert "task_done" not in [event["type"] for event in events]
    assert events[-1]["type"] == "task_result"


def test_effective_child_completion_waits_for_artifacts(tmp_path):
    data = tmp_path / "data"
    child = tmp_path / "child"
    for root in (data, child):
        (root / "task_results").mkdir(parents=True)
    write_task_result(
        data,
        "task-artifacts",
        "scheduled",
        child_drive_root=str(child),
        workspace_root=str(tmp_path / "workspace"),
        artifact_status="pending",
        result="queued",
    )
    write_task_result(
        child,
        "task-artifacts",
        "completed",
        result="done",
        ts="2026-01-01T00:00:02Z",
        outcome_axes={
            "lifecycle": {"status": "completed"},
            "artifacts": {"status": "not_applicable"},
        },
    )

    app = Starlette(routes=[Route("/api/tasks/{task_id}", endpoint=api_task_get, methods=["GET"])])
    app.state.drive_root = data
    payload = TestClient(app).get("/api/tasks/task-artifacts").json()

    assert payload["status"] == "running"
    assert payload["artifact_status"] == "finalizing"
    assert payload["child_status"] == "completed"
    assert payload["outcome_axes"]["lifecycle"]["status"] == "running"
    assert payload["outcome_axes"]["artifacts"]["status"] == "finalizing"

    write_task_result(data, "task-artifacts", "completed", artifact_status="ready", child_drive_root=str(child), workspace_root=str(tmp_path / "workspace"))
    payload = TestClient(app).get("/api/tasks/task-artifacts").json()
    assert payload["status"] == "completed"
    assert payload["artifact_status"] == "ready"


def test_public_task_result_strips_nested_legacy_result_status(tmp_path):
    data = tmp_path / "data"
    (data / "task_results").mkdir(parents=True)
    write_task_result(
        data,
        "legacy-loop",
        "completed",
        result="done",
        loop_outcome={"result_status": "failed", "compat_result_status": "failed", "reason_code": "legacy"},
        verification_ledger={
            "entries": [
                {"kind": "legacy", "result_status": "partial"},
                {"kind": "nested", "payload": {"compat_result_status": "infra_failed"}},
                {"kind": "list", "items": [{"result_status": "failed"}]},
            ],
        },
    )
    app = Starlette(routes=[Route("/api/tasks/{task_id}", endpoint=api_task_get, methods=["GET"])])
    app.state.drive_root = data

    payload = TestClient(app).get("/api/tasks/legacy-loop").json()

    assert "result_status" not in payload
    assert "result_status" not in payload["loop_outcome"]
    assert "compat_result_status" not in payload["loop_outcome"]
    rendered = json.dumps(payload)
    assert "result_status" not in rendered
    assert "compat_result_status" not in rendered


def test_effective_child_failure_waits_for_artifacts(tmp_path):
    data = tmp_path / "data"
    child = tmp_path / "child"
    for root in (data, child):
        (root / "task_results").mkdir(parents=True)
    write_task_result(
        data,
        "task-failed",
        "failed",
        child_drive_root=str(child),
        workspace_root=str(tmp_path / "workspace"),
        artifact_status="finalizing",
        child_status="failed",
        result="boom",
    )
    write_task_result(child, "task-failed", "failed", result="boom", ts="2026-01-01T00:00:02Z")

    app = Starlette(routes=[Route("/api/tasks/{task_id}", endpoint=api_task_get, methods=["GET"])])
    app.state.drive_root = data
    payload = TestClient(app).get("/api/tasks/task-failed").json()

    assert payload["status"] == "running"
    assert payload["artifact_status"] == "finalizing"
    assert payload["child_status"] == "failed"


def test_task_sse_emits_final_result_after_cursor_saw_scheduled_result(tmp_path):
    data = tmp_path / "data"
    (data / "task_results").mkdir(parents=True)
    task_id = "abc123"
    (data / "task_results" / f"{task_id}.json").write_text(
        json.dumps({"task_id": task_id, "status": "completed", "result": "done", "ts": "2026-01-01T00:00:01Z"}),
        encoding="utf-8",
    )
    app = Starlette(routes=[Route("/api/tasks/{task_id}/events", endpoint=api_task_events, methods=["GET"])])
    app.state.drive_root = data

    response = TestClient(app).get(f"/api/tasks/{task_id}/events?cursor=1&wait=0")

    assert response.status_code == 200
    assert '"type": "task_result"' in response.text
    assert '"status": "completed"' in response.text


def test_task_list_filters_on_effective_child_status(tmp_path):
    data = tmp_path / "data"
    child_running = tmp_path / "child-running"
    child_done = tmp_path / "child-done"
    for root in (data, child_running, child_done):
        (root / "task_results").mkdir(parents=True)

    write_task_result(data, "task-running", "scheduled", child_drive_root=str(child_running), result="queued")
    write_task_result(child_running, "task-running", "running", result="working", ts="2026-01-01T00:00:01Z")
    write_task_result(data, "task-done", "scheduled", child_drive_root=str(child_done), result="queued")
    write_task_result(child_done, "task-done", "completed", result="done", ts="2026-01-01T00:00:02Z")

    app = Starlette(routes=[Route("/api/tasks", endpoint=api_tasks_list, methods=["GET"])])
    app.state.drive_root = data
    client = TestClient(app)

    running = client.get("/api/tasks?status=running").json()["tasks"]
    completed = client.get("/api/tasks?status=completed").json()["tasks"]

    assert [task["task_id"] for task in running] == ["task-running"]
    assert running[0]["result"] == "working"
    assert [task["task_id"] for task in completed] == ["task-done"]
    assert completed[0]["result"] == "done"


@pytest.mark.parametrize("status", ["cancelled", "failed"])
def test_effective_task_result_preserves_parent_terminal_status(tmp_path, status):
    data = tmp_path / "data"
    child = tmp_path / "child"
    for root in (data, child):
        (root / "task_results").mkdir(parents=True)
    write_task_result(
        data,
        "task-terminal",
        status,
        child_drive_root=str(child),
        result="parent terminal",
        ts="2026-01-01T00:00:02Z",
    )
    write_task_result(
        child,
        "task-terminal",
        "running",
        result="child stale",
        ts="2026-01-01T00:00:03Z",
    )

    app = Starlette(routes=[Route("/api/tasks/{task_id}", endpoint=api_task_get, methods=["GET"])])
    app.state.drive_root = data

    payload = TestClient(app).get("/api/tasks/task-terminal").json()

    assert payload["status"] == status
    assert payload["result"] == "parent terminal"
    assert payload["ts"] == "2026-01-01T00:00:02Z"


def test_workspace_context_routes_repo_tools_and_blocks_self_commit(tmp_path):
    system_repo = tmp_path / "system"
    workspace = tmp_path / "workspace"
    data = tmp_path / "data"
    system_repo.mkdir()
    workspace.mkdir()
    data.mkdir()
    (system_repo / "README.md").write_text("system", encoding="utf-8")
    (workspace / "README.md").write_text("workspace", encoding="utf-8")
    (workspace / "BIBLE.md").write_text("external bible", encoding="utf-8")

    ctx = ToolContext(
        repo_dir=system_repo,
        drive_root=data,
        workspace_root=workspace,
        workspace_mode="external",
    )

    assert "workspace" in _repo_read(ctx, "README.md")
    registry = ToolRegistry(repo_dir=system_repo, drive_root=data)
    registry.set_context(ctx)
    assert "WORKSPACE_MODE_BLOCKED" in registry.execute("commit_reviewed", {"commit_message": "nope"})
    assert registry.get_schema_by_name("commit_reviewed") is None
    assert registry.get_schema_by_name("request_restart") is None
    assert "WORKSPACE_MODE_BLOCKED" in registry.execute("request_restart", {"reason": "nope"})
    assert "Written" in registry.execute("write_file", {"path": "BIBLE.md", "content": "external edit"})
    assert (workspace / "BIBLE.md").read_text(encoding="utf-8") == "external edit"
    replaced = registry.execute(
        "edit_text",
        {"path": "README.md", "old_str": "workspace", "new_str": "workspace edited"},
    )
    assert "Replaced" in replaced
    assert (workspace / "README.md").read_text(encoding="utf-8") == "workspace edited"


def test_workspace_run_shell_cwd_allows_scratch_blocks_runtime(tmp_path, monkeypatch):
    """External-workspace tasks may run from host scratch (a sibling checkout, a
    /tmp tree); only the Ouroboros runtime (system repo + data drive) stays
    off-limits as a working directory, and runtime writes remain blocked."""
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    # Pin $HOME outside tmp_path so the host-scratch cwd allowance holds on Windows
    # CI too (where pytest's tmp dir lives UNDER home and the data-parent-under-home
    # protection would otherwise block the sibling scratch cwd). See the same fixture
    # in test_external_workspace_access.py.
    fake_home = tmp_path / "_home"
    fake_home.mkdir()
    monkeypatch.setattr(pathlib.Path, "home", lambda: fake_home)
    system_repo = tmp_path / "system"
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    data = tmp_path / "data"
    for path in (system_repo, workspace, outside, data):
        path.mkdir()
    ctx = ToolContext(
        repo_dir=system_repo,
        drive_root=data,
        workspace_root=workspace,
        workspace_mode="external",
    )
    registry = ToolRegistry(repo_dir=system_repo, drive_root=data)
    registry.set_context(ctx)

    # Host scratch outside the declared workspace is now a legitimate cwd...
    scratch_cwd = registry.execute("run_command", {"cmd": ["pwd"], "cwd": str(outside)})
    assert "SHELL_CWD_BLOCKED" not in scratch_cwd
    # ...but the Ouroboros runtime (system repo + data drive) is never a cwd.
    runtime_repo_cwd = registry.execute("run_command", {"cmd": ["pwd"], "cwd": str(system_repo)})
    assert "SHELL_CWD_BLOCKED" in runtime_repo_cwd
    runtime_data_cwd = registry.execute("run_command", {"cmd": ["pwd"], "cwd": str(data)})
    assert "SHELL_CWD_BLOCKED" in runtime_data_cwd
    git_escape = registry.execute("run_command", {"cmd": ["git", "-C", str(system_repo), "status"]})
    assert "WORKSPACE_GIT_BLOCKED" in git_escape
    git_chain = registry.execute("run_command", {"cmd": ["sh", "-c", "true && git --version; echo git binary OK"]})
    assert "WORKSPACE_GIT_BLOCKED" not in git_chain
    outside_write = registry.execute("run_command", {"cmd": ["touch", str(system_repo / "README.md")]})
    assert "WORKSPACE_SHELL_BLOCKED" in outside_write
    embedded_outside_write = registry.execute(
        "run_command",
        {"cmd": ["python", "-c", "open('/tmp/ouroboros-outside.txt','w').write('x')"]},
    )
    assert "WORKSPACE_SHELL_BLOCKED" in embedded_outside_write


def test_workspace_shell_safe_stdio_redirects_are_not_write_like(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    system_repo = tmp_path / "system"
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    data = tmp_path / "data"
    for path in (system_repo, workspace, outside, data):
        path.mkdir()
    (outside / "visible.txt").write_text("ok\n", encoding="utf-8")
    ctx = ToolContext(repo_dir=system_repo, drive_root=data, workspace_root=workspace, workspace_mode="external")
    registry = ToolRegistry(repo_dir=system_repo, drive_root=data)
    registry.set_context(ctx)

    stderr_sink = registry.execute("run_command", {"cmd": f"find {outside} -maxdepth 1 2>/dev/null"})
    fd_dup = registry.execute("run_command", {"cmd": f"ls {outside} 2>&1 | head -n 1"})
    fd_close = registry.execute("run_command", {"cmd": f"find {outside} -maxdepth 1 2>&-"})
    real_redirect = registry.execute("run_command", {"cmd": f"echo x > {outside / 'out.txt'}"})

    assert "WORKSPACE_SHELL_BLOCKED" not in stderr_sink, stderr_sink
    assert "WORKSPACE_SHELL_BLOCKED" not in fd_dup, fd_dup
    assert "WORKSPACE_SHELL_BLOCKED" not in fd_close, fd_close
    assert "WORKSPACE_SHELL_BLOCKED" in real_redirect


def test_workspace_shell_blocks_windows_absolute_redirects_before_shell_execution(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    system_repo = tmp_path / "system"
    workspace = tmp_path / "workspace"
    data = tmp_path / "data"
    for path in (system_repo, workspace, data):
        path.mkdir()
    ctx = ToolContext(repo_dir=system_repo, drive_root=data, workspace_root=workspace, workspace_mode="external")
    registry = ToolRegistry(repo_dir=system_repo, drive_root=data)
    registry.set_context(ctx)

    drive_redirect = registry.execute("run_command", {"cmd": r"echo x > C:\ouroboros-outside\out.txt"})
    unc_redirect = registry.execute("run_command", {"cmd": r"echo x > \\server\share\out.txt"})

    assert "WORKSPACE_SHELL_BLOCKED" in drive_redirect
    assert "SHELL_SYNTAX_UNSUPPORTED" not in drive_redirect
    assert "WORKSPACE_SHELL_BLOCKED" in unc_redirect
    assert "SHELL_SYNTAX_UNSUPPORTED" not in unc_redirect


def test_workspace_shell_keeps_symlinked_workspace_absolute_paths_allowed(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    system_repo = tmp_path / "system"
    real_workspace = tmp_path / "real_workspace"
    workspace_link = tmp_path / "workspace_link"
    data = tmp_path / "data"
    for path in (system_repo, real_workspace, data):
        path.mkdir()
    try:
        workspace_link.symlink_to(real_workspace, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unavailable on this platform: {exc}")
    ctx = ToolContext(repo_dir=system_repo, drive_root=data, workspace_root=workspace_link, workspace_mode="external")
    registry = ToolRegistry(repo_dir=system_repo, drive_root=data)
    registry.set_context(ctx)

    target = workspace_link / "inside.txt"
    result = registry.execute("run_command", {"cmd": [sys.executable, "-c", f"open({str(target)!r}, 'w').write('ok')"]})

    assert "WORKSPACE_SHELL_BLOCKED" not in result, result
    assert (real_workspace / "inside.txt").exists()


def test_workspace_shell_blocks_nested_symlink_escape_absolute_path(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    system_repo = tmp_path / "system"
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    data = tmp_path / "data"
    for path in (system_repo, workspace, outside, data):
        path.mkdir()
    outlink = workspace / "outlink"
    outside_file = outside / "target.txt"
    outside_file.write_text("old\n", encoding="utf-8")
    filelink = workspace / "filelink"
    executable_name_link = workspace / "touch"
    try:
        outlink.symlink_to(outside, target_is_directory=True)
        filelink.symlink_to(outside_file)
        executable_name_link.symlink_to(outside_file)
    except OSError as exc:
        pytest.skip(f"symlink unavailable on this platform: {exc}")
    ctx = ToolContext(repo_dir=system_repo, drive_root=data, workspace_root=workspace, workspace_mode="external")
    registry = ToolRegistry(repo_dir=system_repo, drive_root=data)
    registry.set_context(ctx)

    result = registry.execute("run_command", {"cmd": f"touch {outlink / 'escaped.txt'}"})
    relative_result = registry.execute("run_command", {"cmd": "touch outlink/escaped-relative.txt"})
    bare_result = registry.execute("run_command", {"cmd": "touch outlink"})
    executable_name_result = registry.execute("run_command", {"cmd": ["touch", "touch"]})
    redirect_result = registry.execute("run_command", {"cmd": "echo changed > filelink"})
    compact_redirect_result = registry.execute("run_command", {"cmd": "echo changed >filelink"})
    shell_inline_result = registry.execute("run_command", {"cmd": ["sh", "-c", "echo changed > filelink"]})
    shell_inline_touch_result = registry.execute("run_command", {"cmd": ["sh", "-c", "touch filelink"]})
    bash_redirect_result = registry.execute("run_command", {"cmd": ["bash", "-c", "echo changed &> filelink"]})
    compact_bash_redirect_result = registry.execute("run_command", {"cmd": ["bash", "-c", "echo changed &>filelink"]})
    tee_result = registry.execute("run_command", {"cmd": "printf changed | tee filelink"})
    shell_inline_tee_result = registry.execute("run_command", {"cmd": ["sh", "-c", "printf changed | tee filelink"]})
    python_inline_result = registry.execute(
        "run_command",
        {"cmd": [sys.executable, "-c", "open('filelink', 'w').write('changed')"]},
    )
    python_versioned_result = registry.execute(
        "run_command",
        {"cmd": ["python3.12", "-c", "open('filelink', 'w').write('changed')"]},
    )
    node_script_result = registry.execute(
        "run_script",
        {
            "interpreter": "node",
            "script": "require('fs').writeFileSync('filelink', 'changed')",
        },
    )

    assert "WORKSPACE_SHELL_BLOCKED" in result
    assert "WORKSPACE_SHELL_BLOCKED" in relative_result
    assert "WORKSPACE_SHELL_BLOCKED" in bare_result
    assert "WORKSPACE_SHELL_BLOCKED" in executable_name_result
    assert "WORKSPACE_SHELL_BLOCKED" in redirect_result
    assert "WORKSPACE_SHELL_BLOCKED" in compact_redirect_result
    assert "WORKSPACE_SHELL_BLOCKED" in shell_inline_result
    assert "WORKSPACE_SHELL_BLOCKED" in shell_inline_touch_result
    assert "WORKSPACE_SHELL_BLOCKED" in bash_redirect_result
    assert "WORKSPACE_SHELL_BLOCKED" in compact_bash_redirect_result
    assert "WORKSPACE_SHELL_BLOCKED" in tee_result
    assert "WORKSPACE_SHELL_BLOCKED" in shell_inline_tee_result
    assert "WORKSPACE_SHELL_BLOCKED" in python_inline_result
    assert "WORKSPACE_SHELL_BLOCKED" in python_versioned_result
    assert "WORKSPACE_SHELL_BLOCKED" in node_script_result
    assert not (outside / "escaped.txt").exists()
    assert not (outside / "escaped-relative.txt").exists()
    assert outside_file.read_text(encoding="utf-8") == "old\n"


def test_external_workspace_shell_allows_task_local_git(tmp_path, monkeypatch):
    system_repo = tmp_path / "system"
    workspace = tmp_path / "workspace"
    data = tmp_path / "data"
    system_repo.mkdir()
    data.mkdir()
    _init_repo_with_file(workspace)
    ctx = ToolContext(repo_dir=system_repo, drive_root=data, workspace_root=workspace, workspace_mode="external")
    registry = ToolRegistry(repo_dir=system_repo, drive_root=data)
    registry.set_context(ctx)
    monkeypatch.setenv("OUROBOROS_TEST_RUNTIME_REPO", str(system_repo))

    allowed = [
        ["git", "for-each-ref", "--format=%(refname)"],
        ["git", "rev-list", "--count", "HEAD"],
        ["git", "show-ref", "--heads"],
        ["git", "branch", "--show-current"],
        ["git", "branch", "--list"],
        ["git", "branch", "--list", "ma*"],
        ["git", "branch", "-av"],
        ["git", "tag", "-l"],
        ["git", "tag", "--list", "v*"],
        ["git", "branch", "new-branch"],
        ["git", "branch", "-v", "new-branch"],
        ["git", "branch", "--verbose", "new-branch"],
        ["git", "branch", "-d", "main"],
        ["git", "tag", "v1"],
        ["git", "tag", "-a", "v1", "-m", "x"],
        ["git", "commit", "--allow-empty", "-m", "task-local commit"],
        ["sh", "-c", "git --version; echo git binary OK"],
    ]

    for cmd in allowed:
        assert registry._run_shell_safety_check({"cmd": cmd}, "advanced") is None, cmd

    for cmd in (
        ["git", "-C", str(system_repo), "status"],
        ["git", "--git-dir", str(system_repo / ".git"), "status"],
        # as_posix(): a POSIX shell (sh -c) uses forward slashes; a Windows
        # backslash literal would be eaten as shell escapes during parsing.
        ["sh", "-c", f"cd {system_repo.as_posix()} && git status"],
        ["sh", "-c", "git -C $OUROBOROS_TEST_RUNTIME_REPO status"],
    ):
        result = registry._run_shell_safety_check({"cmd": cmd}, "advanced")
        assert result and "WORKSPACE_GIT_BLOCKED" in result, (cmd, result)


def test_workspace_shell_git_ls_remote_requires_network_contract(tmp_path):
    system_repo = tmp_path / "system"
    workspace = tmp_path / "workspace"
    data = tmp_path / "data"
    system_repo.mkdir()
    data.mkdir()
    _init_repo_with_file(workspace)
    contract = {
        "allowed_resources": {"network": False},
        "resource_policy": {},
    }
    ctx = ToolContext(
        repo_dir=system_repo,
        drive_root=data,
        workspace_root=workspace,
        workspace_mode="external",
        task_contract=contract,
        task_metadata={"task_contract": contract},
    )
    registry = ToolRegistry(repo_dir=system_repo, drive_root=data)
    registry.set_context(ctx)

    for cmd in (
        ["git", "ls-remote", "origin"],
        ["git", "submodule", "update", "--init", "--recursive"],
    ):
        result = registry._run_shell_safety_check({"cmd": cmd}, "advanced")
        assert result and "RESOURCE_CONSTRAINT_BLOCKED" in result, (cmd, result)


def test_workspace_run_shell_allows_absolute_cwd_under_workspace_and_child_drive(tmp_path):
    system_repo = tmp_path / "system"
    workspace = tmp_path / "workspace"
    parent_data = tmp_path / "data"
    parent_task_dir = parent_data / "task_drives" / "task-workspace" / "scratch"
    child_drive = tmp_path / "child-data"
    child_dir = child_drive / "task_drives" / "task-workspace" / "scratch"
    child_control_dir = child_drive / "memory"
    for path in (system_repo, workspace, parent_data / "logs", parent_task_dir, child_dir, child_control_dir):
        path.mkdir(parents=True)
    ctx = ToolContext(
        repo_dir=system_repo,
        drive_root=parent_data,
        workspace_root=workspace,
        workspace_mode="external",
        task_id="task-workspace",
        task_metadata={"drive_root": str(child_drive), "budget_drive_root": str(parent_data)},
    )
    registry = ToolRegistry(repo_dir=system_repo, drive_root=parent_data)
    registry.set_context(ctx)

    def assert_python_cwd(path):
        output = registry.execute(
            "run_command",
            {"cmd": [sys.executable, "-c", "import os; print(os.getcwd())"], "cwd": str(path)},
        )
        assert "exit_code=0" in output
        cwd_output = output.rsplit("STDOUT:\n", 1)[-1].strip()
        assert pathlib.Path(cwd_output).resolve() == path.resolve()

    assert_python_cwd(workspace)
    assert_python_cwd(child_dir)
    child_control = registry.execute("run_command", {"cmd": ["pwd"], "cwd": str(child_control_dir)})
    assert "SHELL_CWD_BLOCKED" in child_control
    blocked = registry.execute("run_command", {"cmd": ["pwd"], "cwd": str(parent_data / "logs")})
    assert "SHELL_CWD_BLOCKED" in blocked
    git_escape = registry._run_shell_safety_check(
        {"cmd": ["git", "-C", "../other-repo", "status"], "cwd": str(child_dir)},
        "advanced",
    )
    assert "WORKSPACE_GIT_BLOCKED" in git_escape
    protected_escape = registry._run_shell_safety_check(
        {"cmd": ["touch", "../data/state/state.json"]},
        "pro",
    )
    assert "WORKSPACE_SHELL_BLOCKED" in protected_escape
    task_drive_write = registry.execute("run_command", {"cmd": ["touch", "output.txt"], "cwd": str(child_dir)})
    assert "WORKSPACE_SHELL_BLOCKED" not in task_drive_write
    assert (child_dir / "output.txt").is_file()
    parent_task_drive_write = registry.execute("run_command", {"cmd": ["touch", "output.txt"], "cwd": str(parent_task_dir)})
    assert "WORKSPACE_SHELL_BLOCKED" not in parent_task_drive_write
    assert (parent_task_dir / "output.txt").is_file()
    absolute_task_drive_file = parent_task_dir / "absolute-python.txt"
    absolute_task_drive_write = registry.execute(
        "run_command",
        {"cmd": [sys.executable, "-c", f"open({str(absolute_task_drive_file)!r}, 'w').write('ok')"]},
    )
    assert "WORKSPACE_SHELL_BLOCKED" not in absolute_task_drive_write
    assert absolute_task_drive_file.read_text(encoding="utf-8") == "ok"


def test_workspace_shell_allows_nested_relative_write_paths(tmp_path):
    system_repo = tmp_path / "system"
    workspace = tmp_path / "workspace"
    data = tmp_path / "data"
    for path in (system_repo, workspace, data):
        path.mkdir(parents=True)
    ctx = ToolContext(repo_dir=system_repo, drive_root=data, workspace_root=workspace, workspace_mode="external")
    registry = ToolRegistry(repo_dir=system_repo, drive_root=data)
    registry.set_context(ctx)

    assert registry._run_shell_safety_check({"cmd": ["touch", "subdir/file.txt"]}, "advanced") is None
    assert registry._run_shell_safety_check({"cmd": ["mkdir", "-p", "build/output"]}, "advanced") is None
    python_write = {"cmd": [sys.executable, "-c", "open('subdir/python.txt', 'w').write('ok')"]}
    assert registry._run_shell_safety_check(python_write, "advanced") is None


def test_workspace_shell_sudo_and_pro_passthrough_policy(tmp_path):
    system_repo = tmp_path / "system"
    workspace = tmp_path / "workspace"
    data = tmp_path / "data"
    for path in (system_repo, workspace, data):
        path.mkdir()
    ctx = ToolContext(repo_dir=system_repo, drive_root=data, workspace_root=workspace, workspace_mode="external")
    registry = ToolRegistry(repo_dir=system_repo, drive_root=data)
    registry.set_context(ctx)

    assert "SUDO_INTERACTIVE_BLOCKED" in registry._run_shell_safety_check({"cmd": ["sudo", "true"]}, "pro")
    assert "SUDO_INTERACTIVE_BLOCKED" in registry._run_shell_safety_check({"cmd": ["sh", "-c", "sudo true"]}, "pro")
    assert "SUDO_INTERACTIVE_BLOCKED" in registry._run_shell_safety_check({"cmd": ["sudo", "-S", "true"]}, "pro")
    assert "SUDO_INTERACTIVE_BLOCKED" in registry._run_shell_safety_check({"cmd": ["sudo", "-nS", "true"]}, "pro")
    assert "SUDO_INTERACTIVE_BLOCKED" in registry._run_shell_safety_check({"cmd": ["sudoedit", "/etc/hosts"]}, "pro")
    assert registry._run_shell_safety_check({"cmd": ["sudo", "-n", "python", "-S", "-c", "print(1)"]}, "pro") is None
    assert "SAFETY_VIOLATION" in registry._run_shell_safety_check({"cmd": ["sh", "-c", "gh\nrepo\ncreate x"]}, "pro")
    assert "SAFETY_VIOLATION" in registry._run_shell_safety_check({"cmd": ["sh", "-c", "gh\nauth\nlogin"]}, "pro")
    outside_write = {"cmd": ["python", "-c", "open('/tmp/ouroboros-pro.txt','w').write('x')"]}
    assert "WORKSPACE_SHELL_BLOCKED" in registry._run_shell_safety_check(outside_write, "advanced")
    assert registry._run_shell_safety_check(outside_write, "pro") is None


def test_workspace_preflight_infers_binaries_from_script_commands():
    tools = _infer_tools_from_manifests([
        {
            "type": "node",
            "scripts": ["test"],
            "script_commands": {"test": "vitest --run"},
        }
    ])
    assert "vitest" in tools
    assert "test" not in tools
    noisy = _infer_tools_from_manifests([
        {
            "type": "node",
            "scripts": ["build"],
            "script_commands": {"build": "NODE_ENV=production cd web && vite build"},
        }
    ])
    assert "NODE_ENV=production" not in noisy
    assert "cd" not in noisy
    assert "vite" in noisy


def test_workspace_patch_includes_tracked_and_untracked_files(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    (repo / "tracked.txt").write_text("old\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "-c", "user.email=t@example.com", "-c", "user.name=T", "commit", "-m", "init"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    (repo / "tracked.txt").write_text("new\n", encoding="utf-8")
    (repo / "new.txt").write_text("hello\n", encoding="utf-8")

    patch = build_workspace_patch(repo)

    assert "diff --git a/tracked.txt b/tracked.txt" in patch
    assert "+new" in patch
    assert "diff --git" in patch and "new.txt" in patch


def test_workspace_patch_lockfile_without_manifest_is_incidental_only_with_code_changes():
    assert _incidental_lockfile_excludes(["package-lock.json"]) == set()
    assert _incidental_lockfile_excludes(["package-lock.json", "package.json", "app.js"]) == set()
    assert _incidental_lockfile_excludes(["package-lock.json", "app.js"]) == {"package-lock.json"}
    assert _incidental_lockfile_excludes(["pkg/poetry.lock", "pkg/module.py"]) == {"pkg/poetry.lock"}


def test_workspace_patch_preserves_lockfile_when_other_changes_are_junk(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "-c", "user.email=t@example.com", "-c", "user.name=T", "commit", "-m", "init"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    (repo / "package-lock.json").write_text('{"lockfileVersion": 3}\n', encoding="utf-8")
    (repo / "dist").mkdir()
    (repo / "dist" / "out.txt").write_text("junk\n", encoding="utf-8")

    _artifacts, manifest = write_workspace_patch_artifacts(repo, tmp_path / "artifacts", task={})
    patch = (tmp_path / "artifacts" / "workspace.patch").read_text(encoding="utf-8")

    assert "package-lock.json" in patch
    assert "dist/out.txt" not in patch
    assert manifest["counts"]["untracked_included"] == 1
    assert manifest["counts"]["untracked_excluded"] == 1


def test_workspace_patch_excludes_binary_junk_and_oversize(tmp_path, monkeypatch):
    """T7 (v6.35.0): the real-usage workspace patch drops untracked build/runtime
    binaries, junk artifacts, and oversize blobs (recorded, not silently lost),
    while keeping real source additions."""
    import ouroboros.headless as headless

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    (repo / "seed.txt").write_text("seed\n", encoding="utf-8")
    subprocess.run(["git", "add", "seed.txt"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "-c", "user.email=t@example.com", "-c", "user.name=T", "commit", "-m", "init"],
        cwd=repo, check=True, capture_output=True,
    )
    # Untracked additions: a real source file (keep), a compiled binary (drop),
    # a redis dump + log junk (drop), and an oversize text file (drop).
    (repo / "fix.py").write_text("def fixed():\n    return 1\n", encoding="utf-8")
    (repo / "app").write_bytes(b"\x7fELF\x00\x01\x02\x03binary\x00blob")  # compiled binary
    (repo / "dump.rdb").write_bytes(b"REDIS\x00\x01")
    (repo / "run.log").write_text("noise\n", encoding="utf-8")
    (repo / "htmlcov").mkdir()
    (repo / "htmlcov" / "index.html").write_text("<html></html>\n", encoding="utf-8")  # top-level coverage junk
    monkeypatch.setattr(headless, "_PATCH_MAX_UNTRACKED_FILE_BYTES", 100)
    (repo / "big.txt").write_text("x" * 200, encoding="utf-8")  # 200 bytes > cap; small files pass size

    artifacts, manifest = write_workspace_patch_artifacts(repo, tmp_path / "artifacts", task={})

    assert manifest["exclude_rules_version"] == 2
    excluded = {item["path"]: item["reason"] for item in manifest["untracked_excluded"]}
    assert "binary file" in excluded.get("app", "")
    assert "binary file" in excluded.get("dump.rdb", "") or "junk artifact" in excluded.get("dump.rdb", "")
    assert "junk artifact" in excluded.get("run.log", "")
    assert "junk artifact" in excluded.get("htmlcov/index.html", "")  # top-level htmlcov excluded
    assert "size cap" in excluded.get("big.txt", "")
    assert "fix.py" in manifest["untracked_included"]
    patch = (tmp_path / "artifacts" / "workspace.patch").read_text(encoding="utf-8")
    assert "fix.py" in patch
    assert "diff --git a/app b/app" not in patch
    assert "dump.rdb" not in patch
    assert "run.log" not in patch
    assert "big.txt" not in patch


def test_workspace_patch_supports_unborn_git_worktree(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    (repo / "created.txt").write_text("hello\n", encoding="utf-8")

    artifacts, manifest = write_workspace_patch_artifacts(repo, tmp_path / "artifacts", task={})

    assert manifest["status"] == ARTIFACT_STATUS_READY_WITH_CHANGES
    assert manifest["base_is_empty_tree"] is True
    assert manifest["base_head"] == "(unborn)"
    assert manifest["current_head"] == "(unborn)"
    assert any(item["kind"] == "workspace_patch" for item in artifacts)
    patch = (tmp_path / "artifacts" / "workspace.patch").read_text(encoding="utf-8")
    assert "created.txt" in patch
    assert "+hello" in patch
    head = subprocess.run(["git", "rev-parse", "--verify", "HEAD"], cwd=repo, capture_output=True)
    assert head.returncode != 0


def test_workspace_patch_supports_unborn_sha256_git_worktree(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    init = subprocess.run(["git", "init", "--object-format=sha256"], cwd=repo, capture_output=True)
    if init.returncode != 0:
        pytest.skip("git does not support sha256 object-format")
    (repo / "created.txt").write_text("hello\n", encoding="utf-8")

    artifacts, manifest = write_workspace_patch_artifacts(repo, tmp_path / "artifacts", task={})

    assert manifest["status"] == ARTIFACT_STATUS_READY_WITH_CHANGES
    assert manifest["base_is_empty_tree"] is True
    assert len(manifest["base_ref"]) == 64
    assert any(item["kind"] == "workspace_patch" for item in artifacts)


def test_workspace_patch_allows_external_workspace_first_commit(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    (repo / "created.txt").write_text("hello\n", encoding="utf-8")
    subprocess.run(["git", "add", "created.txt"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "-c", "user.email=t@example.com", "-c", "user.name=T", "commit", "-m", "first"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    task = {"metadata": {"workspace_preflight": {"git": {"head": ""}}}}

    artifacts, manifest = write_workspace_patch_artifacts(repo, tmp_path / "artifacts", task=task)

    assert manifest["status"] == ARTIFACT_STATUS_READY_WITH_CHANGES
    assert manifest["errors"] == []
    assert any(item["kind"] == "workspace_patch" for item in artifacts)


def test_workspace_patch_fails_on_invalid_head_not_unborn(tmp_path):
    repo = tmp_path / "repo"
    _init_repo_with_file(repo)
    head_ref = subprocess.run(["git", "symbolic-ref", "--quiet", "HEAD"], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    ref_path = repo / ".git" / head_ref
    ref_path.unlink()

    artifacts, manifest = write_workspace_patch_artifacts(repo, tmp_path / "artifacts", task={})

    assert manifest["status"] == ARTIFACT_STATUS_FAILED
    assert any(error["type"] == "git_invalid_head" for error in manifest["errors"])
    assert not any(item["kind"] == "workspace_patch" for item in artifacts)


def test_workspace_patch_manifest_excludes_env_cache_dirs(tmp_path):
    repo = tmp_path / "repo"
    _init_repo_with_file(repo)
    (repo / "new.txt").write_text("hello\n", encoding="utf-8")
    (repo / "node_modules" / "pkg").mkdir(parents=True)
    (repo / "node_modules" / "pkg" / "index.js").write_text("generated\n", encoding="utf-8")
    artifact_dir = tmp_path / "artifacts"

    artifacts, manifest = write_workspace_patch_artifacts(repo, artifact_dir, task={})

    assert manifest["status"] == "ready_with_changes"
    assert "new.txt" in (artifact_dir / "workspace.patch").read_text(encoding="utf-8")
    assert "node_modules" not in (artifact_dir / "workspace.patch").read_text(encoding="utf-8")
    assert manifest["counts"]["untracked_excluded"] == 1
    assert any(item["kind"] == "workspace_patch_manifest" for item in artifacts)


def test_workspace_patch_fails_on_sensitive_untracked_file(tmp_path):
    repo = tmp_path / "repo"
    _init_repo_with_file(repo)
    (repo / ".npmrc").write_text("//registry.npmjs.org/:_authToken=secret\n", encoding="utf-8")

    artifacts, manifest = write_workspace_patch_artifacts(repo, tmp_path / "artifacts", task={})

    assert manifest["status"] == ARTIFACT_STATUS_FAILED
    assert manifest["errors"][0]["type"] == "sensitive_untracked_files"
    assert manifest["sensitive_blocked"][0]["path"] == ".npmrc"
    assert not any(item["kind"] == "workspace_patch" for item in artifacts)


def test_workspace_patch_fails_on_sensitive_untracked_file_inside_excluded_dir(tmp_path):
    repo = tmp_path / "repo"
    _init_repo_with_file(repo)
    secret = repo / "node_modules" / "pkg" / "service-account.json"
    secret.parent.mkdir(parents=True)
    secret.write_text("TOKEN=secret\n", encoding="utf-8")

    artifacts, manifest = write_workspace_patch_artifacts(repo, tmp_path / "artifacts", task={})

    assert manifest["status"] == ARTIFACT_STATUS_FAILED
    assert manifest["counts"]["sensitive_blocked"] == 1
    assert manifest["sensitive_blocked"][0]["path"] == "node_modules/pkg/service-account.json"
    assert not any(item["kind"] == "workspace_patch" for item in artifacts)


def test_workspace_patch_fails_on_common_credential_paths(tmp_path):
    repo = tmp_path / "repo"
    _init_repo_with_file(repo)
    (repo / "credentials").write_text("secret\n", encoding="utf-8")
    (repo / "prod.env").write_text("SECRET=1\n", encoding="utf-8")
    (repo / "settings.env.local").write_text("SECRET=1\n", encoding="utf-8")
    (repo / ".aws").mkdir()
    (repo / ".aws" / "credentials").write_text("secret\n", encoding="utf-8")

    artifacts, manifest = write_workspace_patch_artifacts(repo, tmp_path / "artifacts", task={})

    assert manifest["status"] == ARTIFACT_STATUS_FAILED
    assert {item["path"] for item in manifest["sensitive_blocked"]} == {
        "credentials",
        "prod.env",
        "settings.env.local",
        ".aws/credentials",
    }
    assert not any(item["kind"] == "workspace_patch" for item in artifacts)


def test_workspace_patch_allows_benign_tokenizer_json(tmp_path):
    repo = tmp_path / "repo"
    _init_repo_with_file(repo)
    (repo / "tokenizer.json").write_text("{}\n", encoding="utf-8")

    artifacts, manifest = write_workspace_patch_artifacts(repo, tmp_path / "artifacts", task={})

    assert manifest["status"] == ARTIFACT_STATUS_READY_WITH_CHANGES
    assert manifest["sensitive_blocked"] == []
    assert any(item["kind"] == "workspace_patch" for item in artifacts)


def test_failed_refinalization_drops_stale_workspace_patch_metadata(tmp_path):
    parent = tmp_path / "data"
    repo = tmp_path / "repo"
    parent.mkdir()
    _init_repo_with_file(repo)
    (repo / "tracked.txt").write_text("new\n", encoding="utf-8")
    task = {"id": "task-stale", "workspace_root": str(repo)}
    write_task_result(parent, "task-stale", "completed", workspace_root=str(repo), artifact_status="finalizing")
    finalize_task_artifacts(parent, task)
    result = json.loads((parent / "task_results" / "task-stale.json").read_text(encoding="utf-8"))
    assert any(item.get("kind") == "workspace_patch" for item in result["artifacts"])

    (repo / ".env").write_text("TOKEN=secret\n", encoding="utf-8")
    finalize_task_artifacts(parent, task)

    result = json.loads((parent / "task_results" / "task-stale.json").read_text(encoding="utf-8"))
    assert result["artifact_status"] == ARTIFACT_STATUS_FAILED
    assert not any(item.get("kind") == "workspace_patch" for item in result["artifacts"])


def test_workspace_patch_preserves_untracked_paths_with_whitespace(tmp_path):
    repo = tmp_path / "repo"
    _init_repo_with_file(repo)
    leading = repo / " leading.txt"
    nested = repo / "dir with space" / "file name.txt"
    leading.write_text("leading\n", encoding="utf-8")
    nested.parent.mkdir()
    nested.write_text("nested\n", encoding="utf-8")

    _artifacts, manifest = write_workspace_patch_artifacts(repo, tmp_path / "artifacts", task={})

    assert manifest["status"] == "ready_with_changes"
    assert " leading.txt" in manifest["untracked_included"]
    assert "dir with space/file name.txt" in manifest["untracked_included"]
    assert manifest["patch_size"] > 0


def test_finalize_workspace_patch_allows_external_workspace_head_changed(tmp_path):
    parent = tmp_path / "data"
    repo = tmp_path / "repo"
    parent.mkdir()
    _init_repo_with_file(repo)
    old_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    (repo / "tracked.txt").write_text("new\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "-c", "user.email=t@example.com", "-c", "user.name=T", "commit", "-m", "move"], cwd=repo, check=True, capture_output=True)
    task = {
        "id": "task-head",
        "workspace_root": str(repo),
        "metadata": {"workspace_preflight": {"git": {"head": old_head}}},
    }
    write_task_result(parent, "task-head", "completed", workspace_root=str(repo), artifact_status="finalizing")

    finalize_task_artifacts(parent, task)

    result = json.loads((parent / "task_results" / "task-head.json").read_text(encoding="utf-8"))
    assert result["artifact_status"] == ARTIFACT_STATUS_READY_WITH_CHANGES
    manifest = json.loads((task_artifacts_dir(parent, "task-head") / "workspace_patch.json").read_text(encoding="utf-8"))
    assert manifest["errors"] == []


def test_finalize_workspace_patch_exception_manifest_keeps_base_fields(tmp_path, monkeypatch):
    import ouroboros.headless as headless

    parent = tmp_path / "data"
    repo = tmp_path / "repo"
    parent.mkdir()
    _init_repo_with_file(repo)
    task = {"id": "task-exception", "workspace_root": str(repo)}
    write_task_result(parent, "task-exception", "completed", workspace_root=str(repo), artifact_status="finalizing")

    def boom(*_args, **_kwargs):
        raise RuntimeError("artifact failure")

    monkeypatch.setattr(headless, "write_workspace_patch_artifacts", boom)
    headless.finalize_task_artifacts(parent, task)

    result = json.loads((parent / "task_results" / "task-exception.json").read_text(encoding="utf-8"))
    manifest = json.loads((task_artifacts_dir(parent, "task-exception") / "workspace_patch.json").read_text(encoding="utf-8"))
    assert result["artifact_status"] == ARTIFACT_STATUS_FAILED
    assert manifest["status"] == ARTIFACT_STATUS_FAILED
    assert manifest["base_ref"] == ""
    assert manifest["base_head"] == ""
    assert manifest["base_is_empty_tree"] is False
    assert manifest["current_head"] == ""


def test_workspace_patch_uses_acting_base_sha_without_preflight_metadata(tmp_path):
    repo = tmp_path / "repo"
    _init_repo_with_file(repo)
    base_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    (repo / "tracked.txt").write_text("acting edit\n", encoding="utf-8")
    task = {
        "task_constraint": {
            "mode": "acting_subagent",
            "surface": "self_worktree",
            "base_sha": base_head,
        },
    }

    artifacts, manifest = write_workspace_patch_artifacts(repo, tmp_path / "artifacts", task=task)

    assert manifest["status"] == "ready_with_changes"
    assert manifest["base_ref"] == base_head
    assert manifest["base_head"] == base_head
    assert manifest["current_head"] == base_head
    assert any(item["kind"] == "workspace_patch" for item in artifacts)


def test_workspace_patch_fails_when_acting_base_sha_head_changed(tmp_path):
    repo = tmp_path / "repo"
    _init_repo_with_file(repo)
    base_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    (repo / "tracked.txt").write_text("committed by child\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "-c", "user.email=t@example.com", "-c", "user.name=T", "commit", "-m", "child commit"], cwd=repo, check=True, capture_output=True)
    moved_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    task = {
        "task_constraint": {
            "mode": "acting_subagent",
            "surface": "self_worktree",
            "base_sha": base_head,
        },
    }

    artifacts, manifest = write_workspace_patch_artifacts(repo, tmp_path / "artifacts", task=task)

    assert manifest["status"] == ARTIFACT_STATUS_FAILED
    assert manifest["base_ref"] == base_head
    assert manifest["errors"][-1]["type"] == "workspace_head_changed"
    assert manifest["errors"][-1]["expected_head"] == base_head
    assert manifest["errors"][-1]["current_head"] == moved_head
    assert not any(item["kind"] == "workspace_patch" for item in artifacts)


def test_effective_result_preserves_workspace_artifact_status_with_child_drive(tmp_path):
    from ouroboros.headless import copy_child_task_result
    from ouroboros.task_results import STATUS_COMPLETED
    from ouroboros.task_status import load_effective_task_result

    parent = tmp_path / "data"
    child = tmp_path / "child"
    repo = tmp_path / "repo"
    parent.mkdir()
    child.mkdir()
    _init_repo_with_file(repo)
    old_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    (repo / "tracked.txt").write_text("new\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "-c", "user.email=t@example.com", "-c", "user.name=T", "commit", "-m", "move"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    task_id = "patchfail"
    write_task_result(
        child,
        task_id,
        STATUS_COMPLETED,
        result="child done",
        artifact_status=ARTIFACT_STATUS_READY,
        artifact_bundle={"status": ARTIFACT_STATUS_READY, "artifacts": [], "errors": []},
        ts="2026-01-01T00:00:02Z",
    )
    ledger_path = parent / "task_results" / "artifacts" / task_id / "verification_ledger.json"
    ledger_path.parent.mkdir(parents=True)
    ledger_path.write_text(
        json.dumps({
            "schema_version": 2,
            "outcome_axes": {
                "artifacts": {"status": "finalizing"},
                "objective": {"status": "not_evaluated", "source": "none"},
            },
            "entries": [{"kind": "objective_outcome", "status": "not_evaluated"}],
        }),
        encoding="utf-8",
    )
    write_task_result(
        parent,
        task_id,
        STATUS_COMPLETED,
        result="child done",
        workspace_root=str(repo),
        child_drive_root=str(child),
        artifact_status="finalizing",
        artifacts=[{"kind": "verification_ledger", "name": "verification_ledger.json", "path": str(ledger_path)}],
        child_status=STATUS_COMPLETED,
    )

    finalize_task_artifacts(
        parent,
        {
            "id": task_id,
            "workspace_root": str(repo),
            "drive_root": str(child),
            "metadata": {"workspace_preflight": {"git": {"head": old_head}}},
        },
    )

    effective = load_effective_task_result(parent, task_id)
    assert effective["artifact_status"] == ARTIFACT_STATUS_READY_WITH_CHANGES
    assert not effective.get("artifact_error")
    assert effective["artifact_bundle"]["status"] == ARTIFACT_STATUS_READY_WITH_CHANGES
    refreshed_ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert refreshed_ledger["outcome_axes"]["artifacts"]["status"] == ARTIFACT_STATUS_READY_WITH_CHANGES

    copied = copy_child_task_result(parent, {"id": task_id, "workspace_root": str(repo), "drive_root": str(child)})
    assert copied is not None
    assert copied["artifact_status"] == ARTIFACT_STATUS_READY_WITH_CHANGES
    assert not copied.get("artifact_error")
    assert copied["artifact_bundle"]["status"] == ARTIFACT_STATUS_READY_WITH_CHANGES

    readonly_task_id = "readonlychild"
    write_task_result(
        child,
        readonly_task_id,
        STATUS_COMPLETED,
        result="readonly handoff",
        workspace_root=str(repo),
        workspace_mode="external",
        delegation_role="subagent",
        task_constraint={"mode": "local_readonly_subagent"},
    )
    copied_readonly = copy_child_task_result(
        parent,
        {
            "id": readonly_task_id,
            "workspace_root": str(repo),
            "drive_root": str(child),
            "delegation_role": "subagent",
            "task_constraint": {"mode": "local_readonly_subagent"},
        },
    )
    assert copied_readonly is not None
    assert copied_readonly.get("artifact_status", "") != "finalizing"
    assert "child_status" not in copied_readonly
    effective_readonly = load_effective_task_result(parent, readonly_task_id)
    assert effective_readonly["status"] == STATUS_COMPLETED
    assert effective_readonly["workspace_root"] == str(repo)


def test_finalize_task_artifacts_preserves_existing_artifact_axis_fields(tmp_path):
    from ouroboros.cli import _is_terminal_result
    from ouroboros.task_results import STATUS_COMPLETED, load_task_result

    parent = tmp_path / "data"
    repo = tmp_path / "repo"
    parent.mkdir()
    _init_repo_with_file(repo)
    (repo / "tracked.txt").write_text("new\n", encoding="utf-8")
    task_id = "axisfields"
    write_task_result(
        parent,
        task_id,
        STATUS_COMPLETED,
        workspace_root=str(repo),
        artifact_status=ARTIFACT_STATUS_FINALIZING,
        artifact_bundle={"schema_version": 1, "status": "pending", "artifacts": [], "errors": []},
        outcome_axes={
            "lifecycle": {"status": STATUS_COMPLETED},
            "artifacts": {
                "status": ARTIFACT_STATUS_FINALIZING,
                "diagnostics": {"existing": True},
                "error_count": 0,
            },
            "objective": {"status": "not_evaluated", "source": "none"},
        },
    )

    finalize_task_artifacts(parent, {"id": task_id, "workspace_root": str(repo)})

    result = load_task_result(parent, task_id)
    artifact_axis = result["outcome_axes"]["artifacts"]
    assert artifact_axis["status"] == result["artifact_bundle"]["status"]
    assert result["artifact_bundle"]["status"] == result["artifact_status"]
    assert result["artifact_bundle"]["status"] not in {"pending", "finalizing"}
    assert _is_terminal_result(result) is True
    assert artifact_axis["diagnostics"] == {"existing": True}
    assert artifact_axis["error_count"] == 0


def test_effective_result_preserves_workspace_patch_kind_with_child_drive(tmp_path):
    from ouroboros.artifacts import copy_file_to_task_artifacts
    from ouroboros.cli import _patch_from_result
    from ouroboros.task_results import STATUS_COMPLETED
    from ouroboros.task_status import load_effective_task_result

    parent = tmp_path / "data"
    child = tmp_path / "child"
    repo = tmp_path / "repo"
    parent.mkdir()
    child.mkdir()
    _init_repo_with_file(repo)
    (repo / "tracked.txt").write_text("new\n", encoding="utf-8")

    task_id = "patchkind"
    report = tmp_path / "report.html"
    report.write_text("<h1>done</h1>", encoding="utf-8")
    child_record = copy_file_to_task_artifacts(SimpleNamespace(drive_root=child, task_id=task_id), report, kind="user_file")
    assert child_record is not None
    write_task_result(
        child,
        task_id,
        STATUS_COMPLETED,
        result="child done",
        artifacts=[child_record],
        artifact_status=ARTIFACT_STATUS_READY,
        ts="2026-01-01T00:00:02Z",
    )
    write_task_result(
        parent,
        task_id,
        STATUS_COMPLETED,
        result="child done",
        workspace_root=str(repo),
        child_drive_root=str(child),
        artifacts=[child_record],
        artifact_status="finalizing",
        child_status=STATUS_COMPLETED,
    )

    finalize_task_artifacts(parent, {"id": task_id, "workspace_root": str(repo), "drive_root": str(child)})

    effective = load_effective_task_result(parent, task_id)
    patch_artifacts = [
        item
        for item in effective.get("artifacts") or []
        if isinstance(item, dict) and item.get("name") == "workspace.patch"
    ]
    assert patch_artifacts
    assert patch_artifacts[0]["kind"] == "workspace_patch"
    assert any(item.get("kind") == "user_file" for item in effective.get("artifacts") or [] if isinstance(item, dict))

    class FakeClient:
        def __init__(self):
            self.paths = []

        def get_bytes(self, path):
            self.paths.append(path)
            return b"diff --git a/tracked.txt b/tracked.txt\n"

    client = FakeClient()
    assert _patch_from_result(client, task_id, effective, strict=True).startswith("diff --git")
    assert client.paths == [f"/api/tasks/{task_id}/artifacts/workspace.patch"]


def test_task_artifact_endpoint_serves_only_declared_artifacts(tmp_path):
    data = tmp_path / "data"
    artifact_dir = task_artifacts_dir(data, "task-artifact")
    patch_path = artifact_dir / "workspace.patch"
    patch_path.write_text("diff --git a/a b/a\n", encoding="utf-8")
    write_task_result(
        data,
        "task-artifact",
        "completed",
        artifacts=[{"kind": "workspace_patch", "name": "workspace.patch", "path": str(patch_path), "size": patch_path.stat().st_size}],
        artifact_status="ready",
    )
    app = Starlette(routes=[Route("/api/tasks/{task_id}/artifacts/{name}", endpoint=api_task_artifact, methods=["GET"])])
    app.state.drive_root = data
    client = TestClient(app)

    assert client.get("/api/tasks/task-artifact/artifacts/workspace.patch").text.startswith("diff --git")
    assert client.get("/api/tasks/task-artifact/artifacts/missing.patch").status_code == 404
    assert client.get("/api/tasks/task-artifact/artifacts/bad%5Cname").status_code == 400


def test_task_artifact_endpoint_serves_manifest_artifact_after_status_repair(tmp_path):
    from ouroboros.artifacts import copy_file_to_task_artifacts

    data = tmp_path / "data"
    source_dir = tmp_path / "Desktop"
    source_dir.mkdir()
    source = source_dir / "report.html"
    source.write_text("<h1>ok</h1>", encoding="utf-8")
    copy_file_to_task_artifacts(SimpleNamespace(drive_root=data, task_id="orphaned"), source, kind="user_file")
    write_task_result(
        data,
        "orphaned",
        "running",
        result_status="infra_failed",
        reason_code="provider_failure",
        result="provider failed before normal finalization",
    )
    (data / "state").mkdir(parents=True, exist_ok=True)
    (data / "state" / "queue_snapshot.json").write_text('{"pending": [], "running": []}', encoding="utf-8")
    app = Starlette(routes=[Route("/api/tasks/{task_id}/artifacts/{name}", endpoint=api_task_artifact, methods=["GET"])])
    app.state.drive_root = data

    response = TestClient(app).get("/api/tasks/orphaned/artifacts/report.html")

    assert response.status_code == 200
    assert response.text == "<h1>ok</h1>"


def test_task_artifact_endpoint_rebases_child_drive_artifact_after_status_repair(tmp_path):
    from ouroboros.artifacts import collect_task_artifact_records, copy_file_to_task_artifacts

    data = tmp_path / "data"
    child = tmp_path / "child"
    source_dir = tmp_path / "Desktop"
    source_dir.mkdir()
    source = source_dir / "report.html"
    source.write_text("<h1>child</h1>", encoding="utf-8")
    copy_file_to_task_artifacts(SimpleNamespace(drive_root=child, task_id="childart"), source, kind="user_file")
    child_artifacts = collect_task_artifact_records(child, "childart")
    write_task_result(
        child,
        "childart",
        "completed",
        result="done",
        artifacts=child_artifacts,
        artifact_status="ready",
        ts="2026-01-01T00:00:02Z",
    )
    write_task_result(
        data,
        "childart",
        "running",
        child_drive_root=str(child),
        workspace_root=str(tmp_path / "workspace"),
        result_status="infra_failed",
        reason_code="provider_failure",
        result="provider failed before normal finalization",
    )
    (data / "state").mkdir(parents=True, exist_ok=True)
    (data / "state" / "queue_snapshot.json").write_text('{"pending": [], "running": []}', encoding="utf-8")
    app = Starlette(routes=[Route("/api/tasks/{task_id}/artifacts/{name}", endpoint=api_task_artifact, methods=["GET"])])
    app.state.drive_root = data

    response = TestClient(app).get("/api/tasks/childart/artifacts/report.html")

    parent_artifact = task_artifacts_dir(data, "childart", create=False) / "report.html"
    assert response.status_code == 200
    assert response.text == "<h1>child</h1>"
    assert parent_artifact.read_text(encoding="utf-8") == "<h1>child</h1>"


def test_task_artifact_endpoint_rejects_metadata_name_path_mismatch(tmp_path):
    data = tmp_path / "data"
    artifact_dir = task_artifacts_dir(data, "task-artifact")
    wrong_path = artifact_dir / "memory_export.json"
    wrong_path.write_text("{}", encoding="utf-8")
    write_task_result(
        data,
        "task-artifact",
        "completed",
        artifacts=[{"kind": "workspace_patch", "name": "workspace.patch", "path": str(wrong_path), "size": wrong_path.stat().st_size}],
        artifact_status="ready",
    )
    app = Starlette(routes=[Route("/api/tasks/{task_id}/artifacts/{name}", endpoint=api_task_artifact, methods=["GET"])])
    app.state.drive_root = data

    assert TestClient(app).get("/api/tasks/task-artifact/artifacts/workspace.patch").status_code == 500


def test_memory_export_includes_nested_memory_files(tmp_path):
    drive = tmp_path / "child"
    memory = drive / "memory"
    nested = memory / "knowledge" / "patterns"
    nested.mkdir(parents=True)
    (memory / "identity.md").write_text("id\n", encoding="utf-8")
    (nested / "cli.md").write_text("pattern\n", encoding="utf-8")

    export = build_memory_export(drive, {"id": "task-1", "memory_mode": "forked"})

    assert export["files"]["identity.md"] == "id\n"
    assert export["files"]["knowledge/patterns/cli.md"] == "pattern\n"


def test_startup_prune_removes_only_old_terminal_child_drives(tmp_path):
    data = tmp_path / "data"
    terminal_dir = data / "state" / "headless_tasks" / "oldterminal"
    pending_dir = data / "state" / "headless_tasks" / "oldpending"
    fresh_timestamp_dir = data / "state" / "headless_tasks" / "freshresult"
    terminal_drive = terminal_dir / "data"
    pending_drive = pending_dir / "data"
    fresh_timestamp_drive = fresh_timestamp_dir / "data"
    terminal_drive.mkdir(parents=True)
    pending_drive.mkdir(parents=True)
    fresh_timestamp_drive.mkdir(parents=True)

    now = time.time()
    old = now - (8 * 86400)
    old_iso = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(old))
    fresh_iso = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(now))
    write_task_result(data, "oldterminal", "completed", child_drive_root=str(terminal_drive), artifact_status="ready", result="done", ts=old_iso)
    write_task_result(data, "oldpending", "scheduled", child_drive_root=str(pending_drive), result="queued")
    write_task_result(data, "freshresult", "completed", child_drive_root=str(fresh_timestamp_drive), artifact_status="ready", result="done", ts=fresh_iso)
    os.utime(terminal_dir, (old, old))
    os.utime(pending_dir, (old, old))
    os.utime(fresh_timestamp_dir, (old, old))

    report = prune_headless_task_drives(data, retention_days=7, now=now)

    assert [item["task_id"] for item in report["pruned"]] == ["oldterminal"]
    assert not terminal_dir.exists()
    assert pending_dir.exists()
    assert fresh_timestamp_dir.exists()
    assert any(item["task_id"] == "oldpending" and item["reason"] == "parent_not_terminal" for item in report["skipped"])
    assert any(item["task_id"] == "freshresult" and item["reason"] == "younger_than_retention" for item in report["skipped"])


def test_startup_prune_uses_effective_terminal_status(tmp_path):
    data = tmp_path / "data"
    task_drive = data / "task_drives" / "stalerun"
    child_dir = data / "state" / "headless_tasks" / "stalechild"
    child_drive = child_dir / "data"
    task_drive.mkdir(parents=True)
    child_drive.mkdir(parents=True)
    (task_drive / "scratch.txt").write_text("scratch", encoding="utf-8")
    (child_drive / "scratch.txt").write_text("child", encoding="utf-8")
    (data / "state").mkdir(parents=True, exist_ok=True)
    (data / "state" / "queue_snapshot.json").write_text('{"pending": [], "running": []}', encoding="utf-8")

    now = time.time()
    old = now - (8 * 86400)
    old_iso = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(old))
    for task_id, extra in (
        ("stalerun", {}),
        ("stalechild", {"child_drive_root": str(child_drive)}),
    ):
        write_task_result(
            data,
            task_id,
            "running",
            result_status="infra_failed",
            reason_code="provider_failure",
            result="provider failed",
            ts=old_iso,
            **extra,
        )
    os.utime(task_drive, (old, old))
    os.utime(child_dir, (old, old))

    direct_report = prune_task_drives(data, retention_days=7, now=now)
    child_report = prune_headless_task_drives(data, retention_days=7, now=now)

    assert [item["task_id"] for item in direct_report["pruned"]] == ["stalerun"]
    assert [item["task_id"] for item in child_report["pruned"]] == ["stalechild"]
    assert not task_drive.exists()
    assert not child_dir.exists()


def test_startup_prune_removes_only_old_terminal_task_scratch(tmp_path):
    data = tmp_path / "data"
    old_terminal = data / "task_drives" / "oldterminal"
    old_pending = data / "task_drives" / "oldpending"
    fresh_terminal = data / "task_drives" / "freshterminal"
    for path in (old_terminal, old_pending, fresh_terminal):
        path.mkdir(parents=True)
        (path / "scratch.txt").write_text("scratch", encoding="utf-8")

    now = time.time()
    old = now - (8 * 86400)
    old_iso = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(old))
    fresh_iso = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(now))
    write_task_result(data, "oldterminal", "completed", result="done", ts=old_iso)
    write_task_result(data, "oldpending", "running", result="running")
    write_task_result(data, "freshterminal", "completed", result="done", ts=fresh_iso)
    os.utime(old_terminal, (old, old))
    os.utime(old_pending, (old, old))
    os.utime(fresh_terminal, (old, old))

    report = prune_task_drives(data, retention_days=7, now=now)

    assert [item["task_id"] for item in report["pruned"]] == ["oldterminal"]
    assert not old_terminal.exists()
    assert old_pending.exists()
    assert fresh_terminal.exists()
    assert any(item["task_id"] == "oldpending" and item["reason"] == "task_not_terminal" for item in report["skipped"])
    assert any(item["task_id"] == "freshterminal" and item["reason"] == "younger_than_retention" for item in report["skipped"])


def test_external_child_task_budget_uses_parent_drive_state(tmp_path, monkeypatch):
    from ouroboros.agent import Env, OuroborosAgent

    repo = tmp_path / "repo"
    parent = tmp_path / "parent-data"
    child = tmp_path / "child-data"
    for root in (repo, parent, child):
        root.mkdir()
    for drive in (parent, child):
        (drive / "state").mkdir()
        (drive / "logs").mkdir()
    (parent / "state" / "state.json").write_text('{"spent_usd": 9.0}\n', encoding="utf-8")
    (child / "state" / "state.json").write_text('{"spent_usd": 0.0}\n', encoding="utf-8")

    monkeypatch.setenv("TOTAL_BUDGET", "10")
    monkeypatch.setattr(OuroborosAgent, "_log_worker_boot_once", lambda self: None)
    monkeypatch.setattr("ouroboros.agent.build_llm_messages", lambda **kwargs: ([], {}))

    agent = OuroborosAgent(Env(repo_dir=repo, drive_root=child))
    ctx, _messages, cap_info = agent._prepare_task_context({
        "id": "budget-task",
        "type": "task",
        "text": "x",
        "budget_drive_root": str(parent),
    })

    assert cap_info["budget_remaining"] == 1.0
    assert ctx.task_metadata["budget_drive_root"] == str(parent)


def test_cli_patch_downloads_http_artifact():
    from ouroboros.cli import _patch_from_result

    class FakeClient:
        def __init__(self):
            self.paths = []

        def get_bytes(self, path):
            self.paths.append(path)
            return b"diff --git a/a b/a\n"

    client = FakeClient()
    result = {"artifact_status": "ready", "artifacts": [{"kind": "workspace_patch", "name": "workspace.patch"}]}

    assert _patch_from_result(client, "task-1", result, strict=True).startswith("diff --git")
    assert client.paths == ["/api/tasks/task-1/artifacts/workspace.patch"]


def test_cli_patch_falls_back_to_workspace_patch_name():
    from ouroboros.cli import _patch_from_result

    class FakeClient:
        def __init__(self):
            self.paths = []

        def get_bytes(self, path):
            self.paths.append(path)
            return b"diff --git a/a b/a\n"

    client = FakeClient()
    result = {"artifact_status": "ready", "artifacts": [{"kind": "task_artifact", "name": "workspace.patch"}]}

    assert _patch_from_result(client, "task-1", result, strict=True).startswith("diff --git")
    assert client.paths == ["/api/tasks/task-1/artifacts/workspace.patch"]


def test_cli_patch_strict_rejects_empty_artifact():
    from ouroboros.cli import PatchCLIError, _patch_from_result

    class FakeClient:
        def get_bytes(self, path):
            return b""

    result = {"artifact_status": "ready", "artifacts": [{"kind": "workspace_patch", "name": "workspace.patch"}]}
    with pytest.raises(PatchCLIError, match="empty"):
        _patch_from_result(FakeClient(), "task-1", result, strict=True)


def test_cli_terminal_success_uses_outcome_axes():
    from ouroboros.cli import _is_terminal_success

    base = {
        "status": "completed",
        "artifact_status": "ready",
        "outcome_axes": {
            "execution": {"status": "ok"},
            "objective": {"status": "not_evaluated"},
        },
    }
    assert _is_terminal_success(base) is True

    failed_objective = {
        **base,
        "outcome_axes": {
            "execution": {"status": "ok"},
            "objective": {"status": "fail", "source": "task_acceptance_review"},
        },
    }
    assert _is_terminal_success(failed_objective) is False

    degraded_execution = {
        **base,
        "outcome_axes": {
            "execution": {"status": "degraded"},
            "objective": {"status": "not_evaluated"},
        },
    }
    assert _is_terminal_success(degraded_execution) is False

    # Pin the documented best_effort contract: a forced-finalization best-effort
    # completion is NOT clean terminal success (CLI strict modes must not treat
    # it as a clean pass)...
    best_effort_execution = {
        **base,
        "outcome_axes": {
            "execution": {"status": "best_effort"},
            "objective": {"status": "not_evaluated"},
        },
    }
    assert _is_terminal_success(best_effort_execution) is False


def test_cli_has_no_file_or_review_commit_groups():
    from ouroboros.cli import build_parser

    parser = build_parser()
    assert parser.parse_args(["run", "hello"]).command == "run"
    with pytest.raises(SystemExit):
        parser.parse_args(["files"])
    with pytest.raises(SystemExit):
        parser.parse_args(["commit"])
    with pytest.raises(SystemExit):
        parser.parse_args(["review"])
    with pytest.raises(SystemExit):
        parser.parse_args(["skills", "review", "demo"])


def test_source_server_start_is_blocked_in_packaged_cli_env(monkeypatch):
    from ouroboros import cli

    monkeypatch.setenv("OUROBOROS_PACKAGED_CLI", "1")
    monkeypatch.setattr(cli.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("direct server start"))

    with pytest.raises(cli.CLIError, match="packaged CLI must launch the desktop app"):
        cli._start_local_server("http://127.0.0.1:8765")


def test_packaged_cli_run_start_scan_skips_timeout_value():
    from ouroboros.packaged_cli import _run_start_index

    assert _run_start_index(["run", "--timeout", "5", "--start", "hello"], 0) == 3


def test_cli_run_no_stream_waits_without_jsonl(monkeypatch, capsys):
    from ouroboros import cli

    class FakeClient:
        def request(self, method, path, body=None):
            assert method == "POST"
            assert path == "/api/tasks"
            return {"task_id": "abc123"}

    monkeypatch.setattr(cli, "_client", lambda args, start=False: FakeClient())
    monkeypatch.setattr(cli, "_wait_task", lambda client, task_id, timeout_sec: {"status": "completed", "result": "done"})

    assert cli.main(["run", "--no-stream", "hello"]) == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == "done"


def test_cli_run_timeout_waits_through_finalization_grace(monkeypatch):
    from ouroboros import cli
    from supervisor import queue

    captured = {}

    class FakeClient:
        def request(self, method, path, body=None):
            return {"task_id": "abc123"}

    def fake_wait(_client, _task_id, timeout_sec):
        captured["timeout_sec"] = timeout_sec
        return {"status": "completed", "result": "done"}

    monkeypatch.setenv("OUROBOROS_FINALIZATION_GRACE_SEC", "2")
    queue.init(pathlib.Path("/tmp/ouroboros-test-data"), 600, 1800)
    assert queue.FINALIZATION_GRACE_SEC == 2
    monkeypatch.setattr(cli, "_client", lambda args, start=False: FakeClient())
    monkeypatch.setattr(cli, "_wait_task", fake_wait)

    assert cli.main(["run", "--no-stream", "--timeout", "7", "hello"]) == 0
    assert captured["timeout_sec"] == 14.0
    monkeypatch.delenv("OUROBOROS_FINALIZATION_GRACE_SEC", raising=False)
    assert cli._deadline_wait_timeout(7) == 132.0


def test_cli_run_detach_prints_task_id_without_waiting(monkeypatch, capsys):
    from ouroboros import cli

    class FakeClient:
        def request(self, method, path, body=None):
            assert method == "POST"
            assert path == "/api/tasks"
            return {"task_id": "abc123"}

    monkeypatch.setattr(cli, "_client", lambda args, start=False: FakeClient())
    monkeypatch.setattr(cli, "_watch_task", lambda *args, **kwargs: pytest.fail("detach should not watch"))
    monkeypatch.setattr(cli, "_wait_task", lambda *args, **kwargs: pytest.fail("detach should not wait"))

    assert cli.main(["run", "--detach", "hello"]) == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == "abc123"


def test_cli_run_actor_id_is_sent_as_gateway_root_field(monkeypatch, capsys):
    from ouroboros import cli

    captured = {}

    class FakeClient:
        def request(self, method, path, body=None):
            captured["method"] = method
            captured["path"] = path
            captured["body"] = body
            return {"task_id": "abc123"}

    monkeypatch.setattr(cli, "_client", lambda args, start=False: FakeClient())
    monkeypatch.setattr(cli, "_watch_task", lambda *args, **kwargs: pytest.fail("detach should not watch"))

    assert cli.main(["run", "--detach", "--timeout", "7", "--actor-id", "operator-1", "hello"]) == 0
    assert captured["method"] == "POST"
    assert captured["path"] == "/api/tasks"
    assert captured["body"]["description"] == "hello"
    assert "text" not in captured["body"]
    assert "prompt" not in captured["body"]
    assert captured["body"]["actor_id"] == "operator-1"
    assert captured["body"]["timeout_sec"] == 7.0
    assert captured["body"]["source"] == "cli"
    assert captured["body"]["metadata"]["source"] == "cli"
    assert "actor_id" not in captured["body"]["metadata"]
    assert capsys.readouterr().out.strip() == "abc123"


def test_cli_run_disable_tools_sent_as_gateway_root_field(monkeypatch, capsys):
    from ouroboros import cli

    captured = {}

    class FakeClient:
        def request(self, method, path, body=None):
            captured["method"] = method
            captured["path"] = path
            captured["body"] = body
            return {"task_id": "abc123"}

    monkeypatch.setattr(cli, "_client", lambda args, start=False: FakeClient())
    monkeypatch.setattr(cli, "_watch_task", lambda *args, **kwargs: pytest.fail("detach should not watch"))

    assert cli.main([
        "run", "--detach",
        "--disable-tools", "web_search,browse_page",
        "--disable-tools", "claude_code_edit",
        "hello",
    ]) == 0
    assert captured["method"] == "POST"
    assert captured["path"] == "/api/tasks"
    assert captured["body"]["disabled_tools"] == ["web_search", "browse_page", "claude_code_edit"]
    assert capsys.readouterr().out.strip() == "abc123"


def test_cli_run_task_metadata_json_merges_but_cannot_forge_service_keys(monkeypatch, capsys):
    """--task-metadata-json (v6.56.0) merges user metadata (e.g. budget_profile)
    into body.metadata, but the host-owned delegation_role/source keys are spread
    last and can never be overridden by the user JSON (subagent forgery)."""
    from ouroboros import cli

    captured = {}

    class FakeClient:
        def request(self, method, path, body=None):
            captured["body"] = body
            return {"task_id": "abc123"}

    monkeypatch.setattr(cli, "_client", lambda args, start=False: FakeClient())
    monkeypatch.setattr(cli, "_watch_task", lambda *args, **kwargs: pytest.fail("detach should not watch"))

    payload = (
        '{"budget_profile": {"improvement_policy": "until_deadline", "cost_hard_stop_pct": 0},'
        ' "delegation_role": "subagent", "source": "forged"}'
    )
    assert cli.main(["run", "--detach", "--task-metadata-json", payload, "hello"]) == 0
    metadata = captured["body"]["metadata"]
    assert metadata["budget_profile"] == {
        "improvement_policy": "until_deadline",
        "cost_hard_stop_pct": 0,
    }
    assert metadata["delegation_role"] == "root"
    assert metadata["source"] == "cli"
    assert capsys.readouterr().out.strip() == "abc123"


def test_cli_run_task_metadata_json_rejects_invalid_payloads(monkeypatch):
    from ouroboros import cli

    monkeypatch.setattr(cli, "_client", lambda *args, **kwargs: pytest.fail("client should not be created"))

    for bad in ("not json", "[1, 2]"):
        args = SimpleNamespace(
            prompt=["hello"], delegation_role="root", task_metadata_json=bad,
        )
        with pytest.raises(cli.CLIError, match="task-metadata-json"):
            cli._run_command(args)


def test_cli_run_rejects_forged_subagent_role_before_request(monkeypatch):
    from ouroboros import cli

    monkeypatch.setattr(cli, "_client", lambda *args, **kwargs: pytest.fail("client should not be created"))

    args = SimpleNamespace(prompt=["hello"], delegation_role="subagent")
    with pytest.raises(cli.CLIError, match="internal schedule_subagent"):
        cli._run_command(args)


def test_cli_watch_caps_sse_wait_by_timeout(monkeypatch):
    from ouroboros import cli

    calls = []
    times = iter([100.0, 100.1, 100.2, 101.0])

    class FakeClient:
        def stream_sse(self, path, timeout=120.0):
            calls.append((path, timeout))
            return iter(())

    monkeypatch.setattr(cli.time, "time", lambda: next(times))
    monkeypatch.setattr(cli.time, "sleep", lambda _seconds: None)

    with pytest.raises(cli.TaskTimeoutCLIError):
        cli._watch_task(FakeClient(), "abc123", jsonl=False, quiet=True, timeout_sec=0.5)
    assert "wait=0" in calls[0][0]
    assert calls[0][1] <= 1.5


def test_cli_wait_task_caps_poll_request_by_timeout(monkeypatch):
    from ouroboros import cli

    calls = []
    times = iter([100.0, 100.1, 100.6])

    class FakeClient:
        timeout = 30.0

        def request(self, method, path, body=None, *, timeout=None):
            calls.append(timeout)
            raise cli.ConnectionCLIError("poll timed out")

    monkeypatch.setattr(cli.time, "time", lambda: next(times))

    with pytest.raises(cli.TaskTimeoutCLIError):
        cli._wait_task(FakeClient(), "abc123", timeout_sec=0.5)
    assert calls and calls[0] <= 0.5


def test_swebench_helper_records_cli_timeout_with_continue(tmp_path, monkeypatch):
    script_path = (
        pathlib.Path(__file__).resolve().parent.parent
        / "devtools"
        / "benchmarks"
        / "swe_bench"
        / "swebench_predictions.py"
    )
    spec = importlib.util.spec_from_file_location("swebench_predictions_test", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    rows_path = tmp_path / "rows.jsonl"
    output_path = tmp_path / "predictions.jsonl"
    logs_dir = tmp_path / "logs"
    rows_path.write_text(
        json.dumps({"instance_id": "inst1", "workspace_root": str(workspace), "problem_statement": "fix"}) + "\n",
        encoding="utf-8",
    )

    run_timeouts = []

    def fake_run(cmd, **kwargs):
        if cmd[:2] == ["git", "rev-parse"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="abc\n", stderr="")
        if cmd[:2] == ["git", "status"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        run_timeouts.append(kwargs.get("timeout"))
        raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout", 1), output="partial-out", stderr="partial-err")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(module, "get_finalization_grace_sec", lambda: 7)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "swebench_predictions.py",
            "--input",
            str(rows_path),
            "--output",
            str(output_path),
            "--timeout",
            "1",
            "--continue-on-error",
            "--logs-dir",
            str(logs_dir),
        ],
    )

    assert module.main() == 0
    errors = (tmp_path / "predictions.jsonl.errors.jsonl").read_text(encoding="utf-8")
    assert '"timeout": true' in errors
    assert run_timeouts == [68]
    assert (logs_dir / "inst1" / "ouroboros.stdout").read_text(encoding="utf-8") == "partial-out"
    assert (logs_dir / "inst1" / "ouroboros.stderr").read_text(encoding="utf-8") == "partial-err"


def test_terminal_bench_harbor_adapter_imports_without_harbor():
    script_path = (
        pathlib.Path(__file__).resolve().parent.parent
        / "devtools"
        / "benchmarks"
        / "terminal_bench"
        / "harbor_installed_agent.py"
    )
    spec = importlib.util.spec_from_file_location("terminal_bench_harbor_adapter_test", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.OuroborosTerminalBenchAgent.name() == "Ouroboros Installed"
    assert module._repo_root() == pathlib.Path(__file__).resolve().parent.parent


def test_queue_restore_accepts_headless_chat_zero(tmp_path, monkeypatch):
    import supervisor.queue as queue

    monkeypatch.setattr(queue, "PENDING", [])
    monkeypatch.setattr(queue, "RUNNING", {})
    monkeypatch.setattr(queue, "QUEUE_SEQ_COUNTER_REF", {"value": 0})
    monkeypatch.setattr(queue, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(queue, "QUEUE_SNAPSHOT_PATH", tmp_path / "queue_snapshot.json")
    monkeypatch.setattr(queue, "append_jsonl", lambda *args, **kwargs: None)
    monkeypatch.setattr(queue, "persist_queue_snapshot", lambda reason="": None)
    (tmp_path / "queue_snapshot.json").write_text(
        json.dumps({
            "ts": utc_now_iso(),
            "pending": [{"task": {"id": "headless1", "type": "task", "chat_id": 0, "text": "x"}}],
        }),
        encoding="utf-8",
    )

    assert queue.restore_pending_from_snapshot(max_age_sec=900) == 1
    assert queue.PENDING[0]["id"] == "headless1"
