"""Accepted Project work: real promotion → restore → assignment and read models."""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

from ouroboros import projects_registry as registry
from ouroboros.task_results import load_task_result, write_task_result
from supervisor import queue, state, workers
from tests.test_swarm_host_admission import host  # noqa: F401

pytestmark = pytest.mark.serial


def accepted(host, tmp_path, tid="held"):  # noqa: F811
    from supervisor.events_project_routing import _handle_promote_chat_to_task

    folder = tmp_path / "prepared"
    folder.mkdir(exist_ok=True)
    if not registry.get_project(host.root, "target"):
        registry.create_project(host.root, "target", name="Original Project", working_dir=str(folder))
    answer = _handle_promote_chat_to_task({"task_id": tid, "routing_token": tid + "-token",
        "objective": "Continue the original work", "project_id": "target", "chat_id": 1}, host.ctx)
    assert answer["status"] == "scheduled"
    return next(row for row in host.pending if row["id"] == tid)


def restore_unreadable(host):  # noqa: F811
    path = registry._registry_path(host.root)
    original = path.read_bytes()
    assert queue.persist_queue_snapshot()
    path.write_text("{torn", encoding="utf-8")
    host.pending.clear()
    queue.restore_pending_from_snapshot()
    assert host.pending[0]["_project_admission_restore_hold"]
    assert queue.persist_queue_snapshot()
    return path, original


def resume_after_app_stop(host, *task_ids):  # noqa: F811
    """Owner-approved Quit/crash policy: verification healing alone cannot dispatch.

    Tests of subsequent admission explicitly Resume the named accepted tasks;
    the real selection still checks all independent no-dispatch/money guards.
    """
    from supervisor.events_budget import HOLD_SAVED_WORK, budget_hold_fact

    for task_id in task_ids:
        row = next(task for task in host.pending if task["id"] == task_id)
        assert budget_hold_fact(row)["reason"] == HOLD_SAVED_WORK
        answer = queue.resume_budget_paused_task(task_id)
        assert answer["ok"], answer


def worker(host, monkeypatch):  # noqa: F811
    sent = []
    monkeypatch.setattr(workers, "get_event_q", lambda: SimpleNamespace(put=lambda _event: None))
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_k: 5.0)
    workers.WORKERS[0] = SimpleNamespace(wid=0, busy_task_id=None, reaping=False,
        in_q=SimpleNamespace(put=lambda row: sent.append(copy.deepcopy(row))))
    return sent


def test_same_id_recovers_once_with_original_resources_and_visible_wait(host, tmp_path, monkeypatch):  # noqa: F811
    from ouroboros.gateway.state import _chat_activities_snapshot_safe
    from ouroboros.task_status import effective_task_result

    prepared = copy.deepcopy(accepted(host, tmp_path))
    path, original = restore_unreadable(host)
    sent = worker(host, monkeypatch)
    workers.assign_tasks()
    assert not sent and not host.attempts
    stored = load_task_result(host.root, "held")
    detail = effective_task_result(host.root, stored, materialize_artifacts=False)
    assert detail["status"] == "scheduled"
    assert detail["project_admission_hold"]["label"] == "Waiting for Project verification"
    availability = {}
    census = _chat_activities_snapshot_safe(host.root, availability=availability)
    assert next(row for row in census if row["activity_id"] == "held")["project_admission_hold"]
    for stamp in ("2000-01-01T00:00:00Z", "invalid", None):
        snap = json.loads(queue.QUEUE_SNAPSHOT_PATH.read_text(encoding="utf-8"))
        snap["ts"] = stamp
        queue.QUEUE_SNAPSHOT_PATH.write_text(json.dumps(snap), encoding="utf-8")
        host.pending.clear()
        assert queue.restore_pending_from_snapshot() == 1
        assert host.pending[0]["_project_admission"] == prepared["_project_admission"]
    path.write_bytes(original)
    workers.assign_tasks()
    assert not sent
    resume_after_app_stop(host, "held")
    workers.assign_tasks()
    workers.assign_tasks()
    assert [row["id"] for row in sent] == ["held"]
    assert sent[0]["workspace_root"] == prepared["workspace_root"]
    assert sent[0]["drive_root"] == prepared["drive_root"]
    assert sent[0]["_project_admission"] == prepared["_project_admission"]
    assert not sent[0].get("_project_admission_restore_hold") and not host.attempts
    assert not effective_task_result(host.root, load_task_result(host.root, "held"),
                                     materialize_artifacts=False)["project_admission_hold"]
    write_task_result(host.root, "held", "completed", result="Done.")  # outside the queue: key absent
    assert "project_admission_hold" not in effective_task_result(
        host.root, load_task_result(host.root, "held"), materialize_artifacts=False)


@pytest.mark.parametrize("guard", ["owner_hold", "budget_pause", "budget_root", "acceptance", "reservation",
                                  "running", "missing_result", "unreadable_result", "started", "deadline",
                                  "null", "malformed", "missing_workspace", "retargeted_workspace", "missing_drive"])
def test_recovery_preserves_independent_guards(host, tmp_path, monkeypatch, guard):  # noqa: F811
    accepted(host, tmp_path)
    path, original = restore_unreadable(host)
    sent = worker(host, monkeypatch)
    path.write_bytes(original)
    [held] = host.pending
    basis = copy.deepcopy(held["_project_admission"])
    if guard == "owner_hold":
        held["_owner_hold"] = {"reason": "owner paused"}
    elif guard == "budget_pause":
        held["_budget_pause"] = {"status": "paused_before_dispatch", "auto_resume": False}
    elif guard == "budget_root":
        queue.BUDGET_ROOT_FENCES["held"] = {"status": "active"}
    elif guard == "acceptance":
        queue.ACCEPTANCE_FENCES["held"] = {"status": "sealed"}
    elif guard == "reservation":
        queue.ADMISSION_RESERVATIONS["held"] = "foreign"
    elif guard == "running":
        write_task_result(host.root, "held", "running")
    elif guard in {"missing_result", "unreadable_result"}:
        result = host.root / "task_results/held.json"
        result.unlink() if guard == "missing_result" else result.write_text("{torn", encoding="utf-8")
    elif guard == "started":
        write_task_result(host.root, "held", "scheduled", started_at="2026-01-01T00:00:00Z")
    elif guard == "deadline":
        held["deadline_at"] = "2000-01-01T00:00:00Z"
    elif guard in {"null", "malformed"}:
        held["_project_admission"] = None if guard == "null" else ["raw"]
        basis = copy.deepcopy(held["_project_admission"])
    elif guard in {"missing_workspace", "retargeted_workspace"}:
        folder = tmp_path / "prepared"
        folder.rmdir()
        if guard == "retargeted_workspace":
            elsewhere = tmp_path / "elsewhere"
            elsewhere.mkdir()
            folder.symlink_to(elsewhere, target_is_directory=True)
    elif guard == "missing_drive":
        held["drive_root"] = str(tmp_path / "gone")
    workers.assign_tasks()
    assert not sent and not host.attempts
    assert host.pending[0]["_project_admission"] == basis
    if guard in {"owner_hold", "budget_pause", "budget_root"}:
        assert not host.pending[0].get("_project_admission_restore_hold")
    else:
        assert host.pending[0]["_project_admission_restore_hold"]


@pytest.mark.parametrize("change", ["delete", "remove", "incarnation"])
def test_real_authority_loss_never_redirects_and_failed_terminal_write_retains(host, tmp_path, monkeypatch, change):  # noqa: F811
    accepted(host, tmp_path)
    path, original = restore_unreadable(host)
    path.write_bytes(original)
    if change == "delete":
        registry.begin_project_deletion(host.root, "target")
    else:
        path.write_text('{"projects": []}', encoding="utf-8")
        if change == "incarnation":
            registry.create_project(host.root, "target", working_dir=str(tmp_path / "prepared"))
    sent = worker(host, monkeypatch)
    workers.assign_tasks()
    assert not sent and host.pending[0]["_terminalization_retry"]
    with monkeypatch.context() as patch:
        patch.setattr("ouroboros.task_results.write_task_result", lambda *_a, **_k: False)
        workers.assign_tasks()
    assert not sent and host.pending
    workers.assign_tasks()
    assert not sent and not host.pending
    result = load_task_result(host.root, "held")
    assert result["status"] == "failed" and result.get("admission_outcome") != "never_admitted"


def test_snapshot_clear_failure_retains_hold_then_revalidates(host, tmp_path, monkeypatch):  # noqa: F811
    accepted(host, tmp_path)
    path, original = restore_unreadable(host)
    path.write_bytes(original)
    sent = worker(host, monkeypatch)
    resume_after_app_stop(host, "held")
    hold = copy.deepcopy(host.pending[0]["_project_admission_restore_hold"])
    with monkeypatch.context() as patch:
        patch.setattr(queue, "persist_queue_snapshot", lambda **_k: False)
        workers.assign_tasks()
    assert not sent and host.pending[0]["_project_admission_restore_hold"] == hold
    workers.assign_tasks()
    assert [row["id"] for row in sent] == ["held"]


def test_batch_revalidation_reads_registry_once_and_healthy_sibling_progresses(host, tmp_path, monkeypatch):  # noqa: F811
    from ouroboros import project_admission

    accepted(host, tmp_path)
    accepted(host, tmp_path, "second")
    path, original = restore_unreadable(host)
    queue.enqueue_task({"id": "main", "type": "task", "chat_id": 1, "text": "Independent work"})
    sent = worker(host, monkeypatch)
    reads = []
    read = project_admission._strict_admission_snapshot
    def observed(*args, **kwargs):
        reads.append(queue._queue_lock._is_owned())
        return read(*args, **kwargs)
    monkeypatch.setattr(project_admission, "_strict_admission_snapshot", observed)
    workers.assign_tasks()
    assert reads == [True] and [row["id"] for row in sent] == ["main"]
    path.write_bytes(original)
    workers.WORKERS[0].busy_task_id = None
    resume_after_app_stop(host, "held", "second")
    workers.assign_tasks()
    assert reads == [True, True] and [row["id"] for row in sent] == ["main", "held"]


def test_held_row_recovers_beside_a_room_with_malformed_routing(host, tmp_path, monkeypatch):  # noqa: F811
    """R4: an unrelated room's malformed routing field does not block the held room's release."""
    prepared = copy.deepcopy(accepted(host, tmp_path))
    registry.create_project(host.root, "other", name="Other")
    path, original = restore_unreadable(host)
    data = json.loads(original)
    next(row for row in data["projects"] if row["id"] == "other")["routing_generation"] = "0"
    path.write_text(json.dumps(data), encoding="utf-8")
    sent = worker(host, monkeypatch)
    resume_after_app_stop(host, "held")
    workers.assign_tasks()
    workers.assign_tasks()
    assert [row["id"] for row in sent] == ["held"] and not host.pending
    assert sent[0]["_project_admission"] == prepared["_project_admission"]


def test_stop_during_project_hold_wins_recovery(host, tmp_path, monkeypatch):  # noqa: F811
    from ouroboros.cancel_intents import request_cancel

    accepted(host, tmp_path)
    path, original = restore_unreadable(host)
    request_cancel(host.root, "held", reason="owner stopped")
    path.write_bytes(original)
    sent = worker(host, monkeypatch)
    workers.assign_tasks()
    assert not sent and not host.pending
    assert load_task_result(host.root, "held")["status"] == "cancelled"


@pytest.mark.parametrize("stamp", ["2000-01-01T00:00:00Z", "invalid", None])
def test_first_restore_of_old_project_row_keeps_original_custody(host, tmp_path, monkeypatch, stamp):  # noqa: F811
    accepted(host, tmp_path)
    assert queue.persist_queue_snapshot()
    snap = json.loads(queue.QUEUE_SNAPSHOT_PATH.read_text(encoding="utf-8"))
    snap["ts"] = stamp
    queue.QUEUE_SNAPSHOT_PATH.write_text(json.dumps(snap), encoding="utf-8")
    host.pending.clear()
    assert queue.restore_pending_from_snapshot() == 1
    assert host.pending[0]["_project_admission_restore_hold"]
    sent = worker(host, monkeypatch)
    workers.assign_tasks()
    assert not sent
    resume_after_app_stop(host, "held")
    workers.assign_tasks()
    assert [task["id"] for task in sent] == ["held"]


@pytest.mark.parametrize("dispatch", [None, "none", "possible"])
def test_schedule_requires_its_positive_original_no_dispatch_receipt(host, tmp_path, monkeypatch, dispatch):  # noqa: F811
    task = accepted(host, tmp_path)
    task["metadata"] = {"schedule_occurrence": {"schedule_id": "schedule", "token": "original"}}
    if dispatch is not None:
        write_task_result(host.root, "held", "scheduled", schedule_admission={
            "token": "original", "schedule_id": "schedule", "dispatch": dispatch, "task": copy.deepcopy(task),
        })
    path, original = restore_unreadable(host)
    sent = worker(host, monkeypatch)
    path.write_bytes(original)
    if dispatch == "none":
        resume_after_app_stop(host, "held")
    workers.assign_tasks()
    if dispatch == "none":
        assert [row["id"] for row in sent] == ["held"]
        assert load_task_result(host.root, "held")["schedule_admission"]["dispatch"] == "possible"
    else:
        assert not sent and host.pending[0]["_project_admission_restore_hold"]


@pytest.mark.parametrize("field", ["acceptance_fences", "budget_root_fences"])
def test_invalid_independent_snapshot_fence_cannot_be_cleared_by_project_recovery(host, tmp_path, monkeypatch, field):  # noqa: F811
    accepted(host, tmp_path)
    path, original = restore_unreadable(host)
    snap = json.loads(queue.QUEUE_SNAPSHOT_PATH.read_text(encoding="utf-8"))
    snap[field] = ["invalid"]
    queue.QUEUE_SNAPSHOT_PATH.write_text(json.dumps(snap), encoding="utf-8")
    host.pending.clear()
    queue.restore_pending_from_snapshot()
    path.write_bytes(original)
    sent = worker(host, monkeypatch)
    workers.assign_tasks()
    assert not sent and host.pending[0]["_owner_hold"]


@pytest.mark.parametrize("frozen", [True, False])
@pytest.mark.parametrize("back", [True, False])
def test_room_rebind_and_back_cannot_redirect_original_preparation(host, tmp_path, monkeypatch, frozen, back):  # noqa: F811
    task = accepted(host, tmp_path)
    task["_project_admission"]["frozen"] = frozen
    prepared = copy.deepcopy(task)
    path, original = restore_unreadable(host)
    path.write_bytes(original)
    registry.update_project(host.root, "target", working_dir=str(tmp_path / "elsewhere"))
    if back:
        registry.update_project(host.root, "target", working_dir=str(tmp_path / "prepared"))
    sent = worker(host, monkeypatch)
    if frozen:
        resume_after_app_stop(host, "held")
    workers.assign_tasks()
    workers.assign_tasks()
    if frozen:
        assert [row["id"] for row in sent] == [prepared["id"]]
        for key in ("_project_admission", "workspace_root", "drive_root", "text"):
            assert sent[0][key] == prepared[key]
    else:
        assert not sent and load_task_result(host.root, prepared["id"])["status"] == "failed"


@pytest.mark.parametrize("dispatch", [None, "none", "possible"])
@pytest.mark.parametrize("readable", [False, True])
def test_held_deadline_keeps_custody_without_replay(host, tmp_path, monkeypatch, dispatch, readable):  # noqa: F811
    accepted(host, tmp_path)
    path, original = restore_unreadable(host)
    if readable:
        path.write_bytes(original)
    [held] = host.pending
    held["deadline_at"] = "2000-01-01T00:00:00Z"
    if dispatch is None:
        held.pop("admitted_dispatch", None)
    else:
        held["admitted_dispatch"] = dispatch
    sent = worker(host, monkeypatch)
    workers.assign_tasks()
    assert held["_terminalization_retry"]["trigger"] == "deadline"
    assert held["_terminalization_retry"]["reconcile_delegate_custody"] is True
    assert not sent and not host.attempts
    workers.assign_tasks()
    assert not host.pending and not sent
    stored = load_task_result(host.root, "held")
    assert stored["status"] == "failed" and stored.get("admission_outcome") != "never_admitted"
    assert "deadline" in stored["result"].lower()


def test_unchanged_hold_does_not_rewrite_snapshot_but_changed_reason_does(host, tmp_path, monkeypatch):  # noqa: F811
    accepted(host, tmp_path)
    restore_unreadable(host)
    sent = worker(host, monkeypatch)
    workers.assign_tasks()  # First failure projects its current cause.
    writes = []
    persist = queue.persist_queue_snapshot
    def observed(**kwargs):
        writes.append(kwargs)
        return persist(**kwargs)
    monkeypatch.setattr(queue, "persist_queue_snapshot", observed)
    workers.assign_tasks()
    workers.assign_tasks()
    assert not writes and not sent
    host.pending[0]["_project_admission"] = None
    workers.assign_tasks()
    assert len(writes) == 1
    assert "original" in host.pending[0]["_project_admission_restore_hold"]["detail"].lower()
    workers.assign_tasks()
    assert len(writes) == 1


def test_replaced_binding_cannot_authorize_old_snapshot(host, tmp_path, monkeypatch):  # noqa: F811
    accepted(host, tmp_path)
    path, original = restore_unreadable(host)
    path.write_bytes(original)
    registry.create_project(host.root, "new-room")
    # Bindings are immutable through the production writer. Inject replacement
    # authority to prove an old snapshot cannot silently overrule it.
    binding_path = registry._bindings_path(host.root)
    bindings = json.loads(binding_path.read_text(encoding="utf-8"))
    bindings["bindings"]["held"]["project_id"] = "new-room"
    binding_path.write_text(json.dumps(bindings), encoding="utf-8")
    sent = worker(host, monkeypatch)
    workers.assign_tasks()
    assert not sent and host.pending[0]["_terminalization_retry"]
    assert host.pending[0]["project_id"] == "target"


def test_unwritten_acceptance_cancellation_keeps_project_custody(host, tmp_path, monkeypatch):  # noqa: F811
    accepted(host, tmp_path)
    queue.transition_acceptance_fence(action="begin", token="a" * 32, root_task_id="held", task_id="held")
    assert queue.persist_queue_snapshot()
    host.pending.clear()
    with monkeypatch.context() as patch:
        patch.setattr("ouroboros.task_results.write_task_result", lambda *_a, **_k: False)
        queue.restore_pending_from_snapshot()
    assert host.pending[0]["_terminalization_retry"]["status"] == "cancelled"
    sent = worker(host, monkeypatch)
    workers.assign_tasks()
    assert not sent and not host.pending
    assert load_task_result(host.root, "held")["status"] == "cancelled"


def test_schedule_projection_cannot_clear_independent_owner_hold(host, tmp_path, monkeypatch):  # noqa: F811
    task = accepted(host, tmp_path)
    task["metadata"] = {"schedule_occurrence": {"schedule_id": "schedule", "token": "original"}}
    write_task_result(host.root, "held", "scheduled", schedule_admission={
        "token": "original", "schedule_id": "schedule", "dispatch": "none", "task": copy.deepcopy(task),
    })
    path, original = restore_unreadable(host)
    path.write_bytes(original)
    host.pending[0]["_owner_hold"] = {"reason": "independent owner pause"}
    sent = worker(host, monkeypatch)
    workers.assign_tasks()
    assert not sent and host.pending[0]["_owner_hold"] == {"reason": "independent owner pause"}
    assert not host.pending[0].get("_project_admission_restore_hold")
