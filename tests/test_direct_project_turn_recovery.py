"""A direct Project turn keeps the admission basis it consumed through stop, restore and Resume (#2547).

The contributor's report: a Project-room direct turn saved its loop state when
the application stopped and was restored into the pending queue; the owner's
Resume released the stop hold (``ok: true``), yet the task stayed pending under
``project_routing_fence_lookup_failed`` because neither its result nor its
restored row carried ``_project_admission``. Queued Project work carries that
basis from admission through snapshots; the direct lane bound the room but
recorded no basis, so restore could only ask for one that never existed.

These tests drive the real seams: the direct lane's admission, the agent's
RUNNING writer, the loop's working checkpoint, the stop's queue publication,
the next generation's queue init and restore, the owner's Resume and the
assigner's Project-hold revalidation. A genuinely missing basis keeps its hold.
"""
from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

from ouroboros.task_results import load_task_result
from tests._cancel_intents_shared import _CaptureQueue
from tests.test_direct_roots_adoption import _crash_without_projections, _install, _next_generation

PID = "room-2547"


def _admit_project_turn(tmp_path, monkeypatch):
    """The real direct lane admits one owner turn in a registered Project room."""
    from ouroboros.projects_registry import create_project
    from supervisor import message_bus, workers
    from supervisor import worker_chat_lane as lane

    project = create_project(tmp_path, PID, name="Room 2547")
    monkeypatch.setattr(workers, "REPO_DIR", tmp_path / "repo")
    monkeypatch.setattr(workers, "send_with_budget", lambda *a, **k: None)
    monkeypatch.setattr(message_bus, "get_bridge", lambda: SimpleNamespace(
        send_chat_action=lambda *a, **k: None, broadcast=lambda *a, **k: None))
    workers.open_repo_writer_admission()
    actor = SimpleNamespace(handle_task=lambda task: [], _event_queue=_CaptureQueue())
    origin = {"chat_id": project["chat_id"], "client_message_id": "msg-2547",
              "ts": "2026-10-31T12:00:00+00:00", "text_sha256": "0" * 64}
    admitted = lane._admit_chat_task(actor, project["chat_id"], "Continue the room's work", task_metadata={
        "project_id": PID, "client_message_id": "msg-2547", "origin_message_ref": origin,
        "origin_message_text": "Continue the room's work"})
    assert admitted is not None, "the direct lane admitted the Project turn"
    return project, admitted


def _run_until_saved(tmp_path, admitted):
    """The turn starts (the agent's RUNNING writer) and its loop saves one working round."""
    from ouroboros import working_checkpoint as wc
    from ouroboros.agent import OuroborosAgent
    from tests._budget_pause_exact_helpers import _loop_ctx

    task = admitted["task"]
    owner = SimpleNamespace(env=SimpleNamespace(drive_root=tmp_path, budget_drive_root=tmp_path),
                            _task_started_ts=time.time())
    OuroborosAgent._persist_running_record(owner, task)
    _ctx, limit = _loop_ctx(tmp_path, task["id"], direct=True, attempt=0)
    wc.save_round(limit, "pre_effect")
    return task


def _publish_stop(tmp_path, admitted):
    """The stopping process publishes its queue with the live direct turn on its roster."""
    from supervisor.queue_snapshot import persist_queue_snapshot

    task, actor = admitted["task"], admitted["agent"]
    actor.__dict__.update(_busy=True, _current_task_id=task["id"], _accepting_owner_messages=True,
                          _owner_message_admission_lock=threading.Lock(), _current_chat_id=task["chat_id"],
                          _current_task_metadata=dict(task.get("metadata") or {}),
                          _current_task_text=task["text"], _task_started_ts=time.time())
    assert persist_queue_snapshot(reason="pre_restart_exit") is True
    roster = (tmp_path / "state" / "direct_roots.json").read_text(encoding="utf-8")
    assert task["id"] in roster, "the stop named the live direct turn"
    admitted["registry"].unregister(task["id"])  # the process ends with its actors


def _restore_next_generation(tmp_path, monkeypatch, queue):
    _next_generation(monkeypatch, tmp_path)
    queue.PENDING.clear()
    queue.init(tmp_path)
    fenced: list = []
    queue.restore_pending_from_snapshot(terminalized=fenced)
    return fenced


def _resume_and_revalidate(queue, task_id):
    from supervisor.events_budget import budget_hold_fact
    from supervisor.task_admission import revalidate_project_holds

    outcome = queue.resume_budget_paused_task(task_id)
    assert outcome["ok"], outcome
    row = next(row for row in queue.PENDING if row["id"] == task_id)
    assert budget_hold_fact(row) is None, "Resume released the application-stop hold"
    with queue._queue_lock:
        revalidate_project_holds()
    return row


@pytest.mark.serial
@pytest.mark.parametrize("door", ["published_stop", "crash_without_projections"])
def test_a_direct_project_turn_continues_after_stop_restore_and_resume(tmp_path, monkeypatch, door):
    from ouroboros import cancel_intents as ci
    from ouroboros.project_admission import validate_project_admission
    from supervisor.task_admission import _working_resume_granted

    queue, _workers = _install(tmp_path, monkeypatch)
    project, admitted = _admit_project_turn(tmp_path, monkeypatch)
    task = _run_until_saved(tmp_path, admitted)
    tid = task["id"]

    stored = load_task_result(tmp_path, tid)
    assert stored["project_id"] == PID and stored["_is_direct_chat"] is True
    assert "_project_admission" in stored, "the canonical result keeps the basis the turn consumed"
    basis = validate_project_admission(stored["_project_admission"])
    assert basis["project_id"] == PID and basis["project"]["chat_id"] == project["chat_id"]
    assert not basis.get("legacy_basis")

    if door == "published_stop":
        _publish_stop(tmp_path, admitted)
        fenced = _restore_next_generation(tmp_path, monkeypatch, queue)
    else:
        admitted["registry"].unregister(tid)
        fenced = _crash_without_projections(tmp_path, monkeypatch, queue)

    assert fenced == [] and not ci.has_active_intent(tmp_path, tid, strict=True)
    row = next(row for row in queue.PENDING if row["id"] == tid)
    assert row["project_id"] == PID and row["_is_direct_chat"] is True
    assert row.get("_project_admission") == stored["_project_admission"], "the restored row keeps the basis"
    assert row["_working_recovery"]["source_ref"] and row["_attempt"] == 2

    row = _resume_and_revalidate(queue, tid)

    assert not row.get("_project_admission_restore_hold"), row.get("_project_admission_restore_hold")
    assert not row.get("_terminalization_retry")
    assert _working_resume_granted(row, tmp_path) is True, "the same saved continuation is the next attempt"
    from supervisor.worker_assignment import _candidate_ineligible

    with queue._queue_lock:
        assert _candidate_ineligible(queue, row, 5.0, {}) is False, "the assigner may launch the continuation"


@pytest.mark.serial
def test_a_direct_turn_without_a_recorded_basis_keeps_its_project_hold(tmp_path, monkeypatch):
    """A row from before the basis existed proves no original identity; Resume never synthesizes one."""
    from ouroboros import working_checkpoint as wc
    from ouroboros.projects_registry import bind_task_to_project, create_project
    from ouroboros.task_results import STATUS_RUNNING, write_task_result
    from ouroboros.task_status import execution_owner_record
    from tests._budget_pause_exact_helpers import _loop_ctx

    queue, _workers = _install(tmp_path, monkeypatch)
    project = create_project(tmp_path, PID, name="Room 2547")
    tid = "legacy-direct"
    bind_task_to_project(tmp_path, tid, PID, project["chat_id"], origin={"absent": "mid_task_no_origin"})
    write_task_result(tmp_path, tid, STATUS_RUNNING, chat_id=project["chat_id"], task_attempt=0,
                      _is_direct_chat=True, project_id=PID,
                      execution_owner=execution_owner_record(tmp_path, {"id": tid, "_attempt": 0}, "direct"))
    _ctx, limit = _loop_ctx(tmp_path, tid, direct=True, attempt=0)
    wc.save_round(limit, "pre_effect")

    _crash_without_projections(tmp_path, monkeypatch, queue)
    row = next(row for row in queue.PENDING if row["id"] == tid)
    assert "_project_admission" not in row

    row = _resume_and_revalidate(queue, tid)

    hold = row.get("_project_admission_restore_hold") or {}
    assert hold.get("reason") == "project_routing_fence_lookup_failed"
    assert "missing or invalid" in hold.get("detail", "")
    assert "_project_admission" not in row, "no basis is recaptured from today's registry"


@pytest.mark.serial
def test_a_parked_direct_project_turn_keeps_its_basis_through_the_queue_and_a_stop(tmp_path, monkeypatch):
    """The pause branch (31 Oct: Nova, Circles): the queue parks the turn's own record,
    which a later pooled attempt and the next restore read; its basis rides unchanged."""
    from supervisor.direct_pause import retain_direct_hold

    queue, workers = _install(tmp_path, monkeypatch)
    _project, admitted = _admit_project_turn(tmp_path, monkeypatch)
    task = _run_until_saved(tmp_path, admitted)
    basis = task["_project_admission"]
    retain_direct_hold(task, unsaved=True)  # parks parkable_direct_task(task) and publishes the queue
    admitted["registry"].unregister(task["id"])
    [parked] = [row for row in workers.PENDING if row["id"] == task["id"]]
    assert parked["_project_admission"] == basis

    _restore_next_generation(tmp_path, monkeypatch, queue)

    [row] = [row for row in queue.PENDING if row["id"] == task["id"]]
    assert row["_project_admission"] == basis, row.get("_project_admission")
    assert not row["_project_admission"].get("legacy_basis"), "restore never recaptures the basis"
