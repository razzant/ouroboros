"""Task event replay, log tails and effective child status projection.

Split verbatim out of ``tests/test_headless_cli.py`` by theme. This module
owns what readers see after a task runs: event replay, lineage-filtered log
tails, SSE finalization order, task listing, and the effective-status/result
projection that waits for workspace artifacts.
"""
from __future__ import annotations

import json

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from ouroboros.gateway.tasks import (
    api_task_events,
    api_task_get,
    api_tasks_list,
    iter_task_events,
)
from ouroboros.task_results import write_task_result


from tests._headless_cli_shared import (  # noqa: F401  (autouse fixture applies on import)
    _managed_worker_pool_available,
)


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
