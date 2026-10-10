"""Admission evidence survives promotion, conversion and repeated recovery."""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

from ouroboros import projects_registry as registry
from ouroboros.task_results import load_task_result, write_task_result
from tests.test_schedule_occurrence import _row
from tests.test_swarm_host_admission import host  # noqa: F401

pytestmark = pytest.mark.serial


def convert(host, tid="converted"):  # noqa: F811
    from ouroboros.gateway.projects import api_project_from_task

    async def body():
        return {"task_id": tid, "id": "chosen", "name": "Chosen"}
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(drive_root=host.root)), json=body)
    response = host.run_async(api_project_from_task(request))
    return response, json.loads(response.body)


def admit(host, tmp_path, *, scoped=False, tid="converted"):  # noqa: F811
    from supervisor import queue

    folder = tmp_path / "prepared-folder"
    folder.mkdir(exist_ok=True)
    basis = registry.project_scope_admission(host.root, workspace_root=str(folder) if scoped else "")
    row = queue.enqueue_task({"id": tid, "type": "task", "text": "Synthetic work", "chat_id": 1,
                              "root_task_id": tid, "project_id": basis["project_id"],
                              "workspace_root": str(folder), "_project_admission": basis})
    assert not row.get("_admission_blocked")
    return row


@pytest.mark.parametrize("authority", ["malformed", "unreadable", "missing", "readable"])
def test_bare_promotion_checks_scope_before_drive_effects(host, tmp_path, monkeypatch, authority):  # noqa: F811
    from ouroboros import headless
    from supervisor.events_project_routing import _handle_promote_chat_to_task

    folder = tmp_path / "bare-workspace"
    folder.mkdir()
    path = registry._registry_path(host.root)
    if authority != "missing":
        registry.create_project(host.root, "unrelated")
    if authority == "malformed":
        path.write_text("{torn", encoding="utf-8")
    if authority == "unreadable":
        original_open = type(path).open
        def denied(self, *args, **kwargs):
            if self == path:
                raise PermissionError("synthetic authority failure")
            return original_open(self, *args, **kwargs)
        monkeypatch.setattr(type(path), "open", denied)
    drives = []
    prepare = headless.prepare_task_drive
    def captured(*args, **kwargs):
        drives.append(args)
        return prepare(*args, **kwargs)
    monkeypatch.setattr(headless, "prepare_task_drive", captured)
    result = _handle_promote_chat_to_task({"task_id": "bare", "routing_token": "ours",
        "objective": "Use this folder", "workspace_root": str(folder), "chat_id": 1}, host.ctx)
    if authority in {"malformed", "unreadable"}:
        assert result["reason"] == "project_routing_fence_lookup_failed"
        assert not host.pending and not drives
        assert load_task_result(host.root, "bare")["admission_outcome"] == "never_admitted"
    else:
        assert result["status"] == "scheduled" and len(drives) == 1
        [queued] = host.pending
        assert queued["project_id"].startswith("proj_")
        assert queued["_project_admission"]["project"] is None
        assert queued["workspace_root"] == str(folder)


def test_explicit_promotion_cannot_write_into_replacement_incarnation(host, tmp_path, monkeypatch):  # noqa: F811
    from ouroboros import workspace_admission
    from supervisor.events_project_routing import _handle_promote_chat_to_task

    original = registry.create_project(host.root, "target")
    folder = tmp_path / "explicit-folder"
    folder.mkdir()
    resolve = workspace_admission.resolve_room_workspace
    def replace(*args, **kwargs):
        resolved = resolve(*args, **kwargs)
        registry._registry_path(host.root).write_text('{"projects": []}', encoding="utf-8")
        registry.create_project(host.root, "target")
        return resolved
    monkeypatch.setattr(workspace_admission, "resolve_room_workspace", replace)
    result = _handle_promote_chat_to_task({"task_id": "replacement", "routing_token": "ours",
        "objective": "Use chosen folder", "project_id": "target", "workspace_root": str(folder),
        "chat_id": 1}, host.ctx)
    assert result["reason"] == "project_routing_fence_changed" and not host.pending
    replacement = registry.get_reserved_project(host.root, "target")
    assert replacement["routing_incarnation"] != original["routing_incarnation"]
    assert replacement["working_dir"] == ""


@pytest.mark.parametrize("scoped", [False, True])
@pytest.mark.parametrize("consumer", ["restore", "timeout"])
def test_ui_conversion_keeps_prepared_resource_through_recovery(host, tmp_path, monkeypatch, scoped, consumer):  # noqa: F811
    from supervisor import queue
    from tests.test_retry_project_binding import _patch_retry_input_handoff, _retry

    prepared = admit(host, tmp_path, scoped=scoped)
    write_task_result(host.root, "converted", "scheduled")
    folder = prepared["workspace_root"]
    response, answer = convert(host)
    assert response.status_code == 200
    chosen = answer["project"]
    assert prepared["_project_admission"] == registry.project_admission_basis("chosen", chosen, frozen=True)
    basis = copy.deepcopy(prepared["_project_admission"])
    registry.update_project(host.root, "chosen", working_dir=str(tmp_path / "new-default"))
    if consumer == "restore":
        assert queue.persist_queue_snapshot()
        host.pending.clear()
        assert queue.restore_pending_from_snapshot() == 1
    else:
        _patch_retry_input_handoff(monkeypatch)
        host.pending.clear()
        write_task_result(host.root, "converted", "running", result="working")
        requeued, attempt, _, suppression = _retry(SimpleNamespace(q=queue), prepared, "converted", "retry")
        assert requeued and attempt == 2 and not suppression
        assert registry.project_id_for_task(host.root, "retry") == "chosen"
    [recovered] = host.pending
    assert recovered["project_id"] == "chosen" and recovered["workspace_root"] == folder
    assert recovered["_project_admission"] == basis and not recovered.get("_project_admission_restore_hold")


@pytest.mark.parametrize("stale", [False, True])
def test_api_conversion_binding_overrides_old_result_scope_on_restore(host, tmp_path, monkeypatch, stale):  # noqa: F811
    from starlette.requests import Request
    from ouroboros.gateway.tasks import _create_task_from_body
    from supervisor import queue, workers
    from tests.test_project_hold_recovery import resume_after_app_stop, worker

    folder = tmp_path / "api-folder"
    folder.mkdir()
    request = Request({"type": "http", "app": SimpleNamespace(state=SimpleNamespace(
        drive_root=host.root, repo_dir=workers.REPO_DIR))})
    response = _create_task_from_body(request, {
        "task_id": "converted", "description": "Keep the original API work",
        "workspace_root": str(folder), "memory_mode": "empty"})
    assert response.status_code == 200, response.body
    original = load_task_result(host.root, "converted")
    assert original["project_id"].startswith("proj_")
    response, answer = convert(host)
    assert response.status_code == 200, answer
    assert registry.project_binding_for_task(host.root, "converted")["project_id"] == "chosen"
    assert load_task_result(host.root, "converted")["project_id"] == original["project_id"]
    prepared = copy.deepcopy(host.pending[0])
    assert queue.persist_queue_snapshot()
    if stale:
        snapshot = json.loads(queue.QUEUE_SNAPSHOT_PATH.read_text(encoding="utf-8"))
        snapshot["ts"] = "2000-01-01T00:00:00Z"
        queue.QUEUE_SNAPSHOT_PATH.write_text(json.dumps(snapshot), encoding="utf-8")
    host.pending.clear()
    assert queue.restore_pending_from_snapshot() == 1
    assert bool(host.pending[0].get("_project_admission_restore_hold")) is stale
    sent = worker(host, monkeypatch)
    workers.assign_tasks()
    assert not sent
    resume_after_app_stop(host, "converted")
    workers.assign_tasks()
    workers.assign_tasks()
    assert [row["id"] for row in sent] == ["converted"]
    for key in ("project_id", "_project_admission", "workspace_root", "drive_root", "text"):
        assert sent[0].get(key) == prepared.get(key)
    assert load_task_result(host.root, "converted")["status"] == "running"
    assert not host.attempts


@pytest.mark.parametrize("scoped", [False, True])
def test_in_task_conversion_uses_chosen_row_for_retry(host, tmp_path, scoped):  # noqa: F811
    from supervisor import queue
    from supervisor.worker_promotion import ensure_project_scope

    prepared = admit(host, tmp_path, scoped=scoped)
    host.pending.clear()
    host.running[prepared["id"]] = {"task": prepared}
    # The adopted-origin case is the legitimate derived-scope conversion path.
    if scoped:
        from ouroboros.project_dialogue import build_owner_message_ref
        ref = build_owner_message_ref(chat_id=1, client_message_id="origin", ts="2026-01-01T00:00:00Z", text="Work")
        prepared.update(origin_message_ref=ref, origin_message_text="Work")
        registry.create_project(host.root, "chosen")
        registry.bind_task_to_project(host.root, "sibling", "chosen", origin={"ref": ref, "text": "Work"})
    result = ensure_project_scope({"task_id": "converted", "project_id": "chosen"}, host.ctx)
    assert result["status"] == "delivered"
    chosen = registry.get_reserved_project(host.root, "chosen")
    assert prepared["_project_admission"] == registry.project_admission_basis("chosen", chosen, frozen=True)
    host.running.clear()
    retried = queue.enqueue_task(prepared)
    assert not retried.get("_admission_blocked") and retried["workspace_root"] == prepared["workspace_root"]


@pytest.mark.parametrize("carrier", ["absent", "main", "derived"])
def test_conversion_rollback_restores_scope_and_exact_carrier(host, tmp_path, monkeypatch, carrier):  # noqa: F811
    from supervisor import queue

    prepared = admit(host, tmp_path, scoped=carrier == "derived")
    if carrier == "absent":
        prepared.pop("_project_admission")
    before = copy.deepcopy(prepared)
    marked = []
    def refuse(*args, **kwargs):
        marked.append(copy.deepcopy(prepared))
        raise ValueError("synthetic refused bind")
    monkeypatch.setattr(registry, "bind_task_to_project", refuse)
    response, _ = convert(host)
    assert marked[0]["project_id"] == "chosen"
    assert marked[0]["_project_admission"]["project"]["id"] == "chosen"
    assert response.status_code >= 400 and prepared == before
    saved = json.loads(queue.QUEUE_SNAPSHOT_PATH.read_text(encoding="utf-8"))["pending"][0]["task"]
    assert saved["project_id"] == before["project_id"]
    assert ("_project_admission" in saved) == ("_project_admission" in before)
    assert saved.get("_project_admission") == before.get("_project_admission")


@pytest.mark.parametrize("timestamp", ["old", "invalid", "absent"])
@pytest.mark.parametrize("schedule", [False, True])
def test_project_hold_survives_early_restore_pruning(host, tmp_path, timestamp, schedule):  # noqa: F811
    from supervisor import queue

    registry.create_project(host.root, "chosen")
    if schedule:
        _row(SimpleNamespace(queue=queue), project_id="chosen", intent={"kind": "explicit_none"})
        queue.check_scheduled_tasks()
        [prepared] = host.pending
    else:
        prepared = queue.enqueue_task({"id": "held", "type": "task", "text": "Work",
                                      "project_id": "chosen", "chat_id": 1})
    assert queue.persist_queue_snapshot()
    registry._registry_path(host.root).write_text("{torn", encoding="utf-8")
    host.pending.clear()
    queue.restore_pending_from_snapshot()
    [held] = host.pending
    assert held["_project_admission_restore_hold"]
    if schedule:
        (host.root / "task_results" / (prepared["id"] + ".json")).write_text("{torn", encoding="utf-8")
    snapshot = json.loads(queue.QUEUE_SNAPSHOT_PATH.read_text(encoding="utf-8"))
    if timestamp == "absent":
        snapshot.pop("ts", None)
    else:
        snapshot["ts"] = "2000-01-01T00:00:00Z" if timestamp == "old" else "invalid"
    queue.QUEUE_SNAPSHOT_PATH.write_text(json.dumps(snapshot), encoding="utf-8")
    host.pending.clear()
    assert queue.restore_pending_from_snapshot() == 1
    [retained] = host.pending
    assert retained["id"] == prepared["id"]
    assert retained["_project_admission"] == held["_project_admission"]
    assert retained["_project_admission_restore_hold"] == held["_project_admission_restore_hold"]
    assert not retained.get("_terminalization_retry")
    assert queue.persist_queue_snapshot(reason="startup")
    assert json.loads(queue.QUEUE_SNAPSHOT_PATH.read_text(encoding="utf-8"))["pending"][0]["task"]["id"] == prepared["id"]


@pytest.mark.parametrize("present", [False, True])
def test_absent_and_historical_null_carriers_remain_distinct(host, present):  # noqa: F811
    from supervisor import queue

    registry.create_project(host.root, "chosen")
    raw = {"id": "legacy", "type": "task", "text": "Work", "chat_id": 1, "project_id": "chosen"}
    if present:
        raw["_project_admission"] = None
    host.pending.append(raw)
    assert queue.persist_queue_snapshot()
    saved = json.loads(queue.QUEUE_SNAPSHOT_PATH.read_text(encoding="utf-8"))["pending"][0]["task"]
    assert ("_project_admission" in saved) == present
    host.pending.clear()
    queue.restore_pending_from_snapshot()
    [restored] = host.pending
    if present:
        assert restored["_project_admission"] is None and restored["_project_admission_restore_hold"]
        assert not load_task_result(host.root, "legacy")
    else:
        # A legacy snapshot has no positive pre-handoff proof. Retention must
        # preserve absence, not capture today's identity to enable replay.
        assert "_project_admission" not in restored
        assert restored["_project_admission_restore_hold"]


def test_present_null_is_refused_before_current_identity_capture(host):  # noqa: F811
    from supervisor import queue

    registry.create_project(host.root, "chosen")
    denied = queue.enqueue_task({"id": "null", "type": "task", "chat_id": 1,
                                "project_id": "chosen", "_project_admission": None})
    assert denied["_admission_blocked"] == "project_routing_fence_lookup_failed"
    assert not host.pending


@pytest.mark.parametrize("ending", ["completed", "cancelled", "cancel_intent"])
def test_project_hold_keeps_terminal_and_cancel_authority_independent(host, tmp_path, ending):  # noqa: F811
    from ouroboros.cancel_intents import request_cancel
    from supervisor import queue

    row = admit(host, tmp_path)
    row["_project_admission_restore_hold"] = {"reason": "project_routing_fence_lookup_failed"}
    assert queue.persist_queue_snapshot()
    snapshot = json.loads(queue.QUEUE_SNAPSHOT_PATH.read_text(encoding="utf-8"))
    snapshot["ts"] = "2000-01-01T00:00:00Z"
    queue.QUEUE_SNAPSHOT_PATH.write_text(json.dumps(snapshot), encoding="utf-8")
    if ending == "cancel_intent":
        request_cancel(host.root, row["id"], reason="owner stopped")
    else:
        write_task_result(host.root, row["id"], ending, result="settled")
    host.pending.clear()
    queue.restore_pending_from_snapshot()
    assert not host.pending


def test_historical_null_does_not_acquire_a_replacement_room_identity(host):  # noqa: F811
    from supervisor import queue

    original = registry.create_project(host.root, "chosen")
    registry.bind_task_to_project(host.root, "legacy", "chosen", origin={"absent": "system"})
    registry._registry_path(host.root).write_text('{"projects": []}', encoding="utf-8")
    replacement = registry.create_project(host.root, "chosen")
    assert original["routing_incarnation"] != replacement["routing_incarnation"]
    host.pending.extend([
        {"id": "legacy", "type": "task", "chat_id": 1, "project_id": "chosen", "_project_admission": None},
        {"id": "healthy", "type": "task", "chat_id": 1},
    ])
    queue.persist_queue_snapshot()
    host.pending.clear()
    queue.restore_pending_from_snapshot()
    rows = {r["id"]: r for r in host.pending}
    assert set(rows) == {"legacy", "healthy"}
    assert rows["legacy"]["_project_admission"] is None
    assert "unknown historical identity" in rows["legacy"]["_project_admission_restore_hold"]["detail"]
    host.pending.clear()
    queue.restore_pending_from_snapshot()
    assert next(r for r in host.pending if r["id"] == "legacy")["_project_admission"] is None


def test_main_promotion_preserves_absence_in_queue_snapshot_and_result(host):  # noqa: F811
    from supervisor import queue
    from supervisor.events_project_routing import _handle_promote_chat_to_task

    result = _handle_promote_chat_to_task({"task_id": "main", "routing_token": "ours",
        "objective": "Ordinary Main work", "chat_id": 1}, host.ctx)
    assert result["status"] == "scheduled"
    assert "_project_admission" not in result
    assert "_project_admission" not in host.pending[0]
    assert "_project_admission" not in load_task_result(host.root, "main")
    assert "_project_admission" not in json.loads(queue.QUEUE_SNAPSHOT_PATH.read_text(encoding="utf-8"))["pending"][0]["task"]
