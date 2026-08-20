"""Gateway task-creation API: admission, validation and lineage authority.

Split verbatim out of ``tests/test_headless_cli.py`` by theme. This module
owns ``POST /api/tasks`` behaviour — child-drive creation, reservation and
admission refusals, payload validation, and the forgery guards on task id,
workspace root, subagent role and lineage.
"""
from __future__ import annotations

import json
import subprocess

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from ouroboros.gateway.tasks import (
    _compose_task_text,
    _resolve_workspace_root,
    api_tasks_create,
)
from ouroboros.headless import (
    task_artifacts_dir,
)


from tests._headless_cli_shared import (  # noqa: F401  (autouse fixture applies on import)
    _managed_worker_pool_available,
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
    monkeypatch.setattr("supervisor.queue.persist_queue_snapshot", lambda reason="": True)
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


def test_task_api_admission_refusal_is_terminal_not_scheduled_phantom(tmp_path, monkeypatch):
    from ouroboros.task_results import STATUS_FAILED, load_task_result

    repo = tmp_path / "repo"
    repo.mkdir()
    data = tmp_path / "data"
    (data / "memory").mkdir(parents=True)
    persisted = []

    monkeypatch.setattr(
        "supervisor.queue.enqueue_task",
        lambda task: {
            **task,
            "_admission_blocked": "project_routing_fence",
            "_project_id": "closed-project",
            "_project_lifecycle": "deleting",
        },
    )
    monkeypatch.setattr(
        "supervisor.queue.persist_queue_snapshot",
        lambda reason="": persisted.append(reason),
    )

    app = Starlette(routes=[Route("/api/tasks", endpoint=api_tasks_create, methods=["POST"])])
    app.state.drive_root = data
    app.state.repo_dir = repo
    response = TestClient(app).post(
        "/api/tasks",
        json={
            "description": "must not run",
            "task_id": "blocked-root",
            "project_id": "closed-project",
        },
    )

    assert response.status_code == 409
    payload = response.json()
    assert payload["task_id"] == "blocked-root"
    assert payload["status"] == STATUS_FAILED
    assert payload["admission"]["reason_code"] == "project_routing_fence"
    assert payload["admission"]["project_lifecycle"] == "deleting"
    assert persisted == []
    result = load_task_result(data, "blocked-root")
    assert result["status"] == STATUS_FAILED
    assert result["reason_code"] == "project_routing_fence"
    assert result["admission_cleanup"] == {"child_drive_removed": True}
    assert not (data / "state" / "headless_tasks" / "blocked-root").exists()


def test_task_api_refuses_when_durable_queue_snapshot_fails(tmp_path, monkeypatch):
    import supervisor.queue as queue
    from ouroboros.task_results import STATUS_FAILED, load_task_result

    repo = tmp_path / "repo"
    repo.mkdir()
    data = tmp_path / "data"
    (data / "memory").mkdir(parents=True)
    pending = []
    monkeypatch.setattr(queue, "DRIVE_ROOT", data)
    monkeypatch.setattr(queue, "PENDING", pending)
    monkeypatch.setattr(queue, "RUNNING", {})
    calls = []

    def persist(reason=""):
        calls.append(reason)
        return reason == "api_task_create_rollback"

    monkeypatch.setattr(queue, "persist_queue_snapshot", persist)
    app = Starlette(routes=[Route("/api/tasks", endpoint=api_tasks_create, methods=["POST"])])
    app.state.drive_root = data
    app.state.repo_dir = repo
    response = TestClient(app).post(
        "/api/tasks",
        json={"description": "must be durable", "task_id": "snapshot-fail"},
    )

    assert response.status_code == 503
    assert response.json()["admission"]["reason_code"] == "queue_snapshot_persist_failed"
    assert pending == []
    assert calls == ["api_task_create", "api_task_create_rollback"]
    assert load_task_result(data, "snapshot-fail")["status"] == STATUS_FAILED
    assert not (data / "state" / "headless_tasks" / "snapshot-fail").exists()


def test_task_api_releases_reservation_when_payload_composition_fails(
    tmp_path, monkeypatch,
):
    import supervisor.queue as queue
    from ouroboros.gateway import tasks

    data = tmp_path / "data"
    repo = tmp_path / "repo"
    data.mkdir()
    repo.mkdir()
    task_id = "compose-failure"
    real_compose = tasks._compose_task_text
    monkeypatch.setattr(
        tasks,
        "_compose_task_text",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("compose failed")),
    )
    monkeypatch.setattr(queue, "enqueue_task", lambda task: task)
    monkeypatch.setattr(queue, "persist_queue_snapshot", lambda **_kwargs: True)
    app = Starlette(routes=[Route("/api/tasks", endpoint=api_tasks_create, methods=["POST"])])
    app.state.drive_root = data
    app.state.repo_dir = repo
    client = TestClient(app)

    failed = client.post(
        "/api/tasks", json={"task_id": task_id, "description": "compose me"}
    )
    assert failed.status_code == 503
    assert task_id not in queue.ADMISSION_RESERVATIONS
    assert not task_artifacts_dir(data, task_id, create=False).exists()

    monkeypatch.setattr(tasks, "_compose_task_text", real_compose)
    retried = client.post(
        "/api/tasks", json={"task_id": task_id, "description": "compose me"}
    )
    assert retried.status_code == 200, retried.text


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
    monkeypatch.setattr("supervisor.queue.persist_queue_snapshot", lambda reason="": True)
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
    monkeypatch.setattr("supervisor.queue.persist_queue_snapshot", lambda reason="": True)

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
    monkeypatch.setattr("supervisor.queue.persist_queue_snapshot", lambda reason="": True)

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
    monkeypatch.setattr("supervisor.queue.persist_queue_snapshot", lambda reason="": True)

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
    monkeypatch.setattr("supervisor.queue.persist_queue_snapshot", lambda reason="": True)

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
