"""A conversation-lane turn saved by an application stop comes back once, on the folder it was admitted with.

Since #2654 an owner turn owns a RUNNING row in the conversation lane, so the stop keeps it there with
its saved work (``workers.kill_workers`` -> ``restart_retention.retain_saved_running``). Two regressions
of that move, both seen on the real keyless server (system_e2e S42):

1. Duplicate restore. The next boot's queue init also adopts the same id as an orphaned direct root
   (``direct_roots.adopt_orphaned_direct_results``) whenever the dead server's snapshot is older than
   the ownership observation window — any owner who quits and reopens later. Restore filtered those
   handed-over ids by the snapshot's PENDING rows only, so ``recover_saved_running`` built a successor
   from the RUNNING row AND one from the handover: two queue owners of one id, which the Project hold
   then refuses for both ("Another queue owner holds this task id"), so Resume answered ok over a turn
   that never ran.
2. Recaptured basis. The saved row keeps the carrier's first-step obligation (``_conversation_prepare``),
   so the resumed attempt re-ran ``prepare_conversation_task``, which captured a fresh room basis: a
   folderless turn moved into the folder a sibling gave the room meanwhile (#2547's guard regressed).
"""

from __future__ import annotations

import json
import queue as stdqueue
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from ouroboros import cancel_intents as ci
from ouroboros import working_checkpoint as wc
from ouroboros.task_results import STATUS_RUNNING, load_task_result, write_task_result
from ouroboros.utils import utc_now_iso
from supervisor import worker_chat_lane, workers
from supervisor.events_budget import budget_hold_fact
from tests._budget_pause_exact_helpers import _loop_ctx
from tests._shared import write_queue_document
from tests.test_update_abort_retention import _assert_returned, _Process, _system


TID = "lane-saved-turn"


def _next_generation(monkeypatch, drive):
    """The process booting after the stop: a new custody generation holding the root's admission."""
    from ouroboros import process_custody
    from ouroboros.server_entrypoint import admit_data_root

    monkeypatch.setattr(process_custody, "_SESSION_ID", "next-server-generation")
    admit_data_root(drive, wait_sec=0)  # released by conftest's autouse admission fixture


def _stopped_lane_turn(tmp_path, monkeypatch, *, snapshot_age_sec: float):
    """A lane turn whose attempt saved work, caught by an application stop ``snapshot_age_sec`` ago."""
    from ouroboros import working_checkpoint as wc
    from ouroboros.task_status import execution_owner_record
    from tests._budget_pause_exact_helpers import _install_queue, _loop_ctx
    from tests._cancel_intents_shared import _CaptureQueue

    queue, state, pool = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(pool, "get_event_q", lambda create=True: _CaptureQueue(), raising=False)
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_k: 5.0)
    owner = {**execution_owner_record(tmp_path, {"id": TID, "_attempt": 1}, "direct"),
             "lane": "conversation", "executor": "worker"}
    write_task_result(tmp_path, TID, STATUS_RUNNING, chat_id=1, task_attempt=1, _is_direct_chat=True,
                      execution_owner=owner)
    _ctx, limit = _loop_ctx(tmp_path, TID, direct=True, attempt=1)
    wc.save_round(limit, "pre_effect")
    task = {"id": TID, "type": "task", "chat_id": 1, "text": "write the note", "_is_direct_chat": True,
            "_attempt": 1, "root_task_id": TID, "depth": 0, "_execution_owner": owner,
            "_conversation_prepare": {"quiet": False}}
    stopped_at = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(time.time() - snapshot_age_sec))
    path = tmp_path / "state" / "queue_snapshot.json"
    write_queue_document(path, {"ts": stopped_at, "pending": [], "acceptance_fences": [], "budget_root_fences": [],
                                "running": [{"id": TID, "attempt": 1, "worker_id": 0, "task": task}]})
    monkeypatch.setattr(queue, "QUEUE_SNAPSHOT_PATH", path, raising=False)
    (tmp_path / "state" / "direct_roots.json").unlink(missing_ok=True)
    _next_generation(monkeypatch, tmp_path)
    queue.PENDING.clear()
    queue.init(tmp_path)
    fenced: list = []
    queue.restore_pending_from_snapshot(terminalized=fenced)
    return queue, pool, fenced


@pytest.mark.parametrize("age_sec", [60, 86_400])  # past the 10 s ownership window: a later reopen, a next day
def test_a_saved_lane_turn_restores_under_exactly_one_queue_row(tmp_path, monkeypatch, age_sec):
    from supervisor.events_budget import HOLD_SAVED_WORK, budget_hold_fact

    queue, pool, fenced = _stopped_lane_turn(tmp_path, monkeypatch, snapshot_age_sec=age_sec)

    # The premise: queue init's orphan sweep named the same turn as the snapshot's RUNNING row.
    assert TID in (queue.PRIOR_DIRECT_ROOTS.get("task_ids") or [])
    rows = [row for row in pool.PENDING if row.get("id") == TID]
    assert len(rows) == 1, [{k: row.get(k) for k in ("_attempt", "_working_recovery")} for row in rows]
    task = rows[0]
    assert task["_working_recovery"]["from_attempt"] == 1 and task["_attempt"] == 2
    hold = budget_hold_fact(task)
    assert hold and hold["reason"] == HOLD_SAVED_WORK and hold["door"] == "app_stop"
    assert fenced == [] and not ci.has_active_intent(tmp_path, TID, strict=True)
    # One owner of the id, so the owner's Resume releases it for dispatch.
    assert queue.resume_budget_paused_task(TID)["ok"]
    assert budget_hold_fact(task) is None and task["_working_recovery"]["source_ref"]
    assert [row.get("id") for row in pool.PENDING].count(TID) == 1


def _room(tmp_path, folder: str = ""):
    from ouroboros.project_admission import direct_turn_basis
    from ouroboros.projects_registry import create_project

    create_project(tmp_path, "s42-room", working_dir=folder)
    return direct_turn_basis(tmp_path, "s42-room")


def _resumed_seed(tmp_path, basis):
    task = worker_chat_lane._seed_task(1, "write the proof note", None, None, {"client_message_id": "cm-s42"})
    task.update(project_id="s42-room", _attempt=2, _conversation_prepare={"quiet": False},
                **({"_project_admission": basis} if basis is not None else {}))
    return task


def test_a_resumed_first_step_keeps_the_basis_the_turn_was_admitted_with(tmp_path, monkeypatch):
    from ouroboros.project_admission import admitted_folder
    from ouroboros.projects_registry import update_project

    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    admitted = _room(tmp_path)  # folderless when the turn was admitted
    assert admitted_folder(admitted, "s42-room") == ""
    sibling = tmp_path / "sibling-folder"
    sibling.mkdir()
    update_project(tmp_path, "s42-room", working_dir=str(sibling))  # a sibling's start gave the room a folder

    task = _resumed_seed(tmp_path, admitted)
    frames: list = []
    worker_chat_lane.prepare_conversation_task(SimpleNamespace(), task, tmp_path, broadcast=frames.append)

    assert admitted_folder(task["_project_admission"], "s42-room") == "", task["_project_admission"]
    assert task["_project_admission"] == admitted
    assert not [frame for frame in frames if frame.get("type") == "direct_turn_bound"], frames
    assert "_conversation_prepare" not in task


def test_a_first_step_without_a_carried_basis_still_binds_the_rooms_current_folder(tmp_path, monkeypatch):
    """Positive control: an admission (no basis yet) captures and broadcasts the room as it is now."""
    from ouroboros.project_admission import admitted_folder

    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    folder = tmp_path / "room-folder"
    folder.mkdir()
    _room(tmp_path, str(folder))

    task = _resumed_seed(tmp_path, None)
    frames: list = []
    worker_chat_lane.prepare_conversation_task(SimpleNamespace(), task, tmp_path, broadcast=frames.append)

    assert admitted_folder(task["_project_admission"], "s42-room") == str(folder)
    bound = [frame for frame in frames if frame.get("type") == "direct_turn_bound"]
    assert bound and bound[0]["facts"]["_project_admission"] == task["_project_admission"], frames


def test_a_basis_for_another_room_is_replaced_by_the_rooms_own(tmp_path, monkeypatch):
    """Placement moves a turn by replacing its room: a carried basis of another room never binds it."""
    from ouroboros.project_admission import admitted_folder

    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    folder = tmp_path / "room-folder"
    folder.mkdir()
    _room(tmp_path, str(folder))
    other = {"project_id": "another-room", "project": {"id": "another-room", "working_dir": ""}}

    task = _resumed_seed(tmp_path, other)
    worker_chat_lane.prepare_conversation_task(SimpleNamespace(), task, tmp_path)

    assert task["_project_admission"]["project_id"] == "s42-room"
    assert admitted_folder(task["_project_admission"], "s42-room") == str(folder)


def test_restored_saved_turn_rows_are_dated_by_the_stop(tmp_path):
    """Fixture honesty: the snapshot the restore test writes is older than the ownership window."""
    from ouroboros.queue_observation import load_queue_document
    from ouroboros.task_status import queue_snapshot_observation

    path = tmp_path / "state" / "queue_snapshot.json"
    path.parent.mkdir(parents=True)
    stopped_at = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(time.time() - 60))
    write_queue_document(path, {"ts": stopped_at, "pending": [], "running": []})
    assert queue_snapshot_observation(load_queue_document(tmp_path, fresh=True))["fresh"] is False
    write_queue_document(path, {"ts": utc_now_iso(), "pending": [], "running": []})
    assert queue_snapshot_observation(load_queue_document(tmp_path, fresh=True))["fresh"] is True


# ---------------------------------------------------------------------------
# #2555: a fresh Main conversation through lane -> pool -> saved stop / aborted Update
# ---------------------------------------------------------------------------

def _lane_carrier(system, wid=50):
    from ouroboros.task_carrier import LANE_CONVERSATION
    from tests.worker_pool_support import affirm_ready

    proc = _Process(710000 + wid)
    system.processes[proc.pid] = proc
    commands = stdqueue.Queue()
    slot = system.workers.Worker(wid, proc, commands, lane=LANE_CONVERSATION)
    affirm_ready(slot)
    system.workers.WORKERS[wid] = slot
    return slot, commands


def _persist_running(root, task, *, clock=True):
    """The body's actual start record (attempt and owner included), as its agent writes it."""
    from ouroboros.agent import OuroborosAgent

    OuroborosAgent._persist_running_record(SimpleNamespace(env=SimpleNamespace(drive_root=root, budget_drive_root=root),
                                                           _task_started_ts=time.time() if clock else None), task)


def _carried_turn(system, root, *, metadata=None, wake=False, patch=None, started=True):
    """A real Main (or wake) admission, published before its carrier's real first step."""
    from supervisor.conversation_lane import handle_direct_turn_bound
    from supervisor.worker_chat_lane import prepare_conversation_task

    slot, commands = _lane_carrier(system)
    before = set(system.workers.RUNNING)
    if wake:
        from ouroboros.consciousness_wake import wake_task_metadata

        _lane_carrier(system, wid=51)  # a wake never takes the owner's last idle carrier
        patch.setattr(system.state, "load_state", lambda: {"bg_consciousness_enabled": True})
        patch.setattr(system.state, "budget_remaining", lambda *_a, **_k: 5.0)
        receipt = system.workers.handle_wake_direct(1, "[Wake-up · test]", wake_task_metadata("reflex", "heartbeat"))
        assert receipt["admitted"], receipt
    else:
        system.workers.handle_chat_direct(1, "keep working", None, task_metadata=dict(metadata or {}))
    [task_id] = set(system.workers.RUNNING) - before
    published = {row["id"]: row for row in json.loads(system.queue.QUEUE_SNAPSHOT_PATH.read_text())["running"]}
    frames = []
    dispatched = commands.get_nowait()
    prepare_conversation_task(SimpleNamespace(), dispatched, root, broadcast=frames.append)
    for frame in frames:
        handle_direct_turn_bound(frame, SimpleNamespace(RUNNING=system.workers.RUNNING))
    if started:
        _persist_running(root, dispatched)
    return SimpleNamespace(id=task_id, slot=slot, task=dispatched, published=published[task_id]["task"])


def _save_work(root, turn, *, attempt=1):
    _ctx, limit = _loop_ctx(root, turn.id, direct=True, attempt=attempt)
    wc.save_round(limit, "pre_effect")


def _launched(system):
    sent = []
    for commands in system.launches:
        while not commands.empty():
            sent.append(commands.get_nowait())
    return sent


def _detach_to_pool(system, root, monkeypatch, turn):
    """The real ``continue_in_background``: arm, save, park the lane row, wake, launch; its pooled start record."""
    from ouroboros import budget_pause, model_detach
    from ouroboros.tools.control_mind import _continue_in_background
    from supervisor import message_bus
    from supervisor.sleep_wake import wake_ready_sleepers
    from tests._budget_pause_exact_helpers import _fast_hold, _quiet_external
    from tests.conversation_carrier_support import park_lane_pause

    monkeypatch.setattr(model_detach, "pool_available", lambda: {"available": True})
    monkeypatch.setattr(message_bus, "get_bridge", lambda: SimpleNamespace(push_log=lambda *_a: None))
    monkeypatch.setattr(system.state, "budget_remaining", lambda *_a, **_k: 5.0)
    _fast_hold(monkeypatch, budget_pause)
    _quiet_external(monkeypatch, budget_pause)
    ctx, limit = _loop_ctx(root, turn.id, direct=True)
    ctx.task_attempt = turn.task.get("_attempt")  # the agent's ``ctx.task_attempt``: None on a seed's first launch
    ctx.task_metadata = {"_execution_owner": turn.task["_execution_owner"]}
    limit.messages.append({"role": "tool", "tool_call_id": "call_b", "content": "done b"})
    assert json.loads(_continue_in_background(ctx, "keep working"))["status"] == "detach_armed"
    with pytest.raises(budget_pause.BudgetPauseRequested) as raised:
        budget_pause.enter_cold_sleep(limit)
    park_lane_pause(budget_pause.pause_event(turn.task, raised.value.pause), worker_id=turn.slot.wid)
    budget_pause.end_dispatch_fence(turn.id)
    [parked] = system.workers.PENDING
    assert parked["id"] == turn.id and parked["detached_by"] == "mind"
    assert any(row.get("ok") for row in wake_ready_sleepers(system.queue))
    system.workers.assign_tasks()
    [pooled] = _launched(system)
    assert pooled["id"] == turn.id and system.workers.RUNNING[turn.id]["worker_id"] != turn.slot.wid
    _persist_running(root, pooled)
    stored = load_task_result(root, turn.id)
    assert stored["execution_owner"]["kind"] == "pooled"
    # The pooled body consumes its single-use grant: its own ``resumed`` source, under the row's actual attempt.
    resumed_ctx, resumed = _loop_ctx(root, turn.id, direct=True)
    resumed_ctx.task_attempt, resumed_ctx.budget_pause_resume = pooled.get("_attempt"), pooled["_budget_pause_resume"]
    state = budget_pause.load_budget_pause(resumed_ctx)
    monkeypatch.setattr("ouroboros.owner_wait.rebind_restored_route", lambda *_a, **_k: (None, "max"))  # no model here
    budget_pause.resume_paused_loop(resumed.tools, state, [], {}, {}, set(), budget_remaining_usd=5.0)
    assert wc.save_round(resumed, "pre_effect")  # it keeps working on the pool
    return pooled


def _resume_and_launch(system, task_id):
    resumed = system.queue.resume_budget_paused_task(task_id)
    assert resumed["ok"], resumed
    system.workers.assign_tasks()
    return _launched(system)


def _boot_after_stop(system, monkeypatch, root):
    """The application stops (its teardown kill); the next generation restores the saved queue."""
    _stop(system)
    _boot(system, monkeypatch, root)


def _stop(system):
    system.workers.kill_workers(force=True, terminal_status="cancelled", result_reason="Server shutdown.",
                                stop_source="server_shutdown", retain_saved_work=True)


def _boot(system, monkeypatch, root):
    system.workers.PENDING[:] = []
    system.workers.RUNNING.clear()
    _next_generation(monkeypatch, root)
    system.queue.init(root)
    system.queue.restore_pending_from_snapshot()
    monkeypatch.setattr(system.state, "budget_remaining", lambda *_a, **_k: 5.0)
    assert system.workers.ensure_worker_pool_started(allow_disabled_restart=True)
    system.workers.open_repo_writer_admission()


@pytest.mark.parametrize("origin", ["owner", "wake"])
def test_a_fresh_main_turn_returns_from_an_aborted_update_once_on_its_own_scope_evidence(tmp_path, monkeypatch, origin):
    system = _system(tmp_path, monkeypatch)
    turn = _carried_turn(system, tmp_path, metadata={"client_message_id": "cm-main"}, wake=origin == "wake", patch=monkeypatch)
    assert turn.published["_project_scope_none"] is True, "published before any carrier holds the turn"
    stored = load_task_result(tmp_path, turn.id)
    # The real first launch: what task_admission._working_attempt_unstarted reads as 0 -> 1.
    assert stored["task_attempt"] == 0 and stored["execution_owner"]["kind"] == "direct"
    assert stored["execution_owner"]["task_attempt"] == 0 and "admitted_dispatch_attempt" not in stored
    _save_work(tmp_path, turn)
    saved = wc.read_working(tmp_path, turn.id, 1)[1]
    assert system.control._quiesce_repo_writers("auto_merge") == []
    system.control._respawn_workers_after_failed_update()
    source = _assert_returned(system, tmp_path, turn.id)["_working_recovery"]
    assert source["from_attempt"] == 1
    ctx, _ = _loop_ctx(tmp_path, turn.id, attempt=2)
    frozen = wc.load_recovery(ctx, source)
    assert {k: frozen[k] for k in ("usage", "round_idx", "started_at")} == {
        k: saved[k] for k in ("usage", "round_idx", "started_at")}
    system.workers.assign_tasks()
    [sent] = _launched(system)
    assert (sent["id"], sent["_attempt"], sent["_working_recovery"]) == (turn.id, 2, source)
    assert not sent.get("project_id") and "_project_admission" not in sent
    assert load_task_result(tmp_path, turn.id)["admitted_dispatch_attempt"] == 2
    system.workers.assign_tasks()
    assert not _launched(system) and not system.workers.PENDING


def test_a_main_turn_saved_by_a_quit_waits_for_resume_and_survives_another_stop_cycle(tmp_path, monkeypatch):
    from supervisor.events_budget import HOLD_SAVED_WORK

    system = _system(tmp_path, monkeypatch)
    turn = _carried_turn(system, tmp_path, metadata={"client_message_id": "cm-main"})
    _save_work(tmp_path, turn)
    _boot_after_stop(system, monkeypatch, tmp_path)
    [held] = system.workers.PENDING
    assert budget_hold_fact(held)["reason"] == HOLD_SAVED_WORK and held["_attempt"] == 2
    assert held["_project_scope_none"] is True
    for _ in range(2):
        system.workers.assign_tasks()
    assert not _launched(system), "only the owner's Resume releases a quit's saved work"
    [sent] = _resume_and_launch(system, turn.id)
    assert (sent["id"], sent["_attempt"]) == (turn.id, 2) and sent["_project_scope_none"] is True
    system.workers.assign_tasks()
    assert not _launched(system)
    # The second cycle: the resumed attempt runs and saves, then an Update aborts around it.
    _persist_running(tmp_path, sent)
    _save_work(tmp_path, turn, attempt=2)
    assert system.control._quiesce_repo_writers("auto_merge") == []
    system.control._respawn_workers_after_failed_update()
    [row] = system.workers.PENDING
    assert row["_attempt"] == 3 and row["_working_recovery"]["from_attempt"] == 2 and budget_hold_fact(row) is None
    system.workers.assign_tasks()
    [again] = _launched(system)
    assert (again["id"], again["_attempt"]) == (turn.id, 3)
    system.workers.assign_tasks()
    assert not _launched(system)


def test_an_acknowledged_restart_returns_a_main_turn_without_a_resume(tmp_path, monkeypatch):
    from supervisor.restart_retention import prepare_restart_returns
    from tests.test_restart_saved_work import _ack

    system = _system(tmp_path, monkeypatch)
    turn = _carried_turn(system, tmp_path, metadata={"client_message_id": "cm-main"})
    _save_work(tmp_path, turn)
    assert prepare_restart_returns(tmp_path, system.workers.RUNNING, system.workers.PENDING,
                                   transaction_id="tx-main") == {turn.id}
    _ack(tmp_path, "tx-main")
    _boot_after_stop(system, monkeypatch, tmp_path)
    [row] = system.workers.PENDING
    assert budget_hold_fact(row) is None and row["_working_recovery"]["cause"] == "restart"
    system.workers.assign_tasks()
    [sent] = _launched(system)
    assert (sent["id"], sent["_attempt"]) == (turn.id, 2)
    system.workers.assign_tasks()
    assert not _launched(system)


@pytest.mark.parametrize("door", ["aborted_update", "quit_restore"])
def test_a_main_turn_detached_to_the_pool_returns_once_after_a_stop(tmp_path, monkeypatch, door):
    system = _system(tmp_path, monkeypatch)
    turn = _carried_turn(system, tmp_path, metadata={"client_message_id": "cm-main"})
    pooled = _detach_to_pool(system, tmp_path, monkeypatch, turn)
    stored = load_task_result(tmp_path, turn.id)
    assert pooled["_project_scope_none"] is True and pooled["_attempt"] == 1, "the parked row keeps its attempt"
    assert stored["task_attempt"] == 1 and stored["execution_owner"]["task_attempt"] == 1
    assert stored["execution_owner"]["kind"] == "pooled" and stored["admitted_dispatch_attempt"] == 1
    if door == "aborted_update":
        assert system.control._quiesce_repo_writers("auto_merge") == []
        system.control._respawn_workers_after_failed_update()
        source = _assert_returned(system, tmp_path, turn.id)["_working_recovery"]
        system.workers.assign_tasks()
        sent = _launched(system)
    else:
        _boot_after_stop(system, monkeypatch, tmp_path)
        [held] = system.workers.PENDING
        assert budget_hold_fact(held) and held["_attempt"] == 2
        source = held["_working_recovery"]
        system.workers.assign_tasks()
        assert not _launched(system), "only the owner's Resume releases a quit's saved work"
        sent = _resume_and_launch(system, turn.id)
    assert [(row["id"], row["_attempt"], row["_working_recovery"]) for row in sent] == [(turn.id, 2, source)]
    system.workers.assign_tasks()
    assert not _launched(system)
    # The repeat cycle: the successor runs and saves, then another Update aborts around it.
    _persist_running(tmp_path, sent[0])
    _save_work(tmp_path, turn, attempt=2)
    assert system.control._quiesce_repo_writers("auto_merge") == []
    system.control._respawn_workers_after_failed_update()
    system.workers.assign_tasks()
    assert [(row["id"], row["_attempt"]) for row in _launched(system)] == [(turn.id, 3)]
    system.workers.assign_tasks()
    assert not _launched(system)


def test_a_historical_pooled_attempt_zero_stays_held_after_an_aborted_update(tmp_path, monkeypatch):
    """A row a past release recorded this way proves no attempt: zero is never normalized by owner kind."""
    system = _system(tmp_path, monkeypatch)
    turn = _carried_turn(system, tmp_path, metadata={"client_message_id": "cm-main"})
    _detach_to_pool(system, tmp_path, monkeypatch, turn)
    stored = load_task_result(tmp_path, turn.id)
    write_task_result(tmp_path, turn.id, "running", task_attempt=0,
                      execution_owner={**stored["execution_owner"], "task_attempt": 0})
    _aborted(system, tmp_path, turn)
    [held] = system.workers.PENDING
    assert held["_project_admission_restore_hold"]["reason"] == "project_dispatch_unconfirmed"


def _parked_before_its_body(system, root, monkeypatch, *, mark=None):
    """The carrier's real pre-body Pause event for a real Main seed, parked by the supervisor."""
    from ouroboros import budget_pause
    from ouroboros.owner_pause import read_fence
    from supervisor import message_bus
    from supervisor.owner_pause_control import request_owner_pause
    from tests.conversation_carrier_support import park_lane_pause

    monkeypatch.setattr(message_bus, "get_bridge", lambda: SimpleNamespace(push_log=lambda *_a: None))
    monkeypatch.setattr(system.state, "budget_remaining", lambda *_a, **_k: 5.0)
    turn = _carried_turn(system, root, metadata={"client_message_id": "cm-main"}, started=False)
    if mark:
        system.workers.RUNNING[turn.id]["task"]["admitted_dispatch"] = mark
    write_task_result(root, turn.id, "scheduled", chat_id=1, root_task_id=turn.id)
    assert request_owner_pause(turn.id, request_id="pause-main")["ok"]
    fence = read_fence(root, turn.id)  # the Pause this carrier's prestart wait observed
    owner = SimpleNamespace(executed_seconds=lambda: 0.0, prestart_pause_fence=fence, drive_root=root)
    error = SimpleNamespace(before_task_start=True, control_reason=budget_pause.RAIL_OWNER_PAUSE)
    event = budget_pause.prestart_pause_event(turn.task, error, owner)
    assert event is not None
    park_lane_pause(event, worker_id=turn.slot.wid)
    return turn, event


def test_a_main_turn_paused_before_its_body_keeps_its_attempt_through_resume_and_abort(tmp_path, monkeypatch):
    system = _system(tmp_path, monkeypatch)
    turn, _event = _parked_before_its_body(system, tmp_path, monkeypatch)
    [parked] = system.workers.PENDING
    assert parked["_project_scope_none"] is True and parked["_attempt"] == 1
    assert "admitted_dispatch" not in parked, "the lane put this turn on its carrier: never 'none'"
    system.workers.assign_tasks()
    assert not _launched(system), "the owner's Pause still holds it"
    [pooled] = _resume_and_launch(system, turn.id)
    _persist_running(tmp_path, pooled)
    assert load_task_result(tmp_path, turn.id)["task_attempt"] == 1
    _save_work(tmp_path, turn)
    assert system.control._quiesce_repo_writers("auto_merge") == []
    system.control._respawn_workers_after_failed_update()
    source = _assert_returned(system, tmp_path, turn.id)["_working_recovery"]
    system.workers.assign_tasks()
    assert [(row["id"], row["_attempt"], row["_working_recovery"]) for row in _launched(system)] == [(turn.id, 2, source)]
    system.workers.assign_tasks()
    assert not _launched(system)


def test_a_resumed_pre_body_turn_caught_by_a_quit_before_its_launch_waits_for_one_more_resume(tmp_path, monkeypatch):
    system = _system(tmp_path, monkeypatch)
    turn, _event = _parked_before_its_body(system, tmp_path, monkeypatch)
    assert system.queue.resume_budget_paused_task(turn.id)["ok"]
    _boot_after_stop(system, monkeypatch, tmp_path)
    [held] = system.workers.PENDING
    assert budget_hold_fact(held)["reason"] == "saved_work_hold" and "admitted_dispatch" not in held
    from supervisor.task_admission import PRESTART_PROOF, prestart_returned_unsent

    proof = prestart_returned_unsent(held)  # the quit hold replaced the result's limit and carries the proof
    assert proof["before_task_start"] is True and budget_hold_fact(held)[PRESTART_PROOF] == proof
    assert not held.get("_project_admission_restore_hold"), "its carrier's pre-body return is positive proof"
    system.workers.assign_tasks()
    assert not _launched(system), "a quit still needs the owner's Resume"
    assert [(row["id"], row["_attempt"]) for row in _resume_and_launch(system, turn.id)] == [(turn.id, 1)]
    system.workers.assign_tasks()
    assert not _launched(system)


@pytest.mark.parametrize("door", ["second_quit", "panic"])
def test_a_resumed_pre_body_turn_keeps_its_proof_through_another_stop(tmp_path, monkeypatch, door):
    system = _system(tmp_path, monkeypatch)
    turn, _event = _parked_before_its_body(system, tmp_path, monkeypatch)
    assert system.queue.resume_budget_paused_task(turn.id)["ok"]
    _boot_after_stop(system, monkeypatch, tmp_path)
    if door == "second_quit":  # Resume, then another quit before the launch: the carried proof still holds
        assert system.queue.resume_budget_paused_task(turn.id)["ok"]
    else:
        (tmp_path / "state" / "panic_stop.flag").write_text("panic", encoding="utf-8")
    _boot_after_stop(system, monkeypatch, tmp_path)
    [held] = system.workers.PENDING
    assert budget_hold_fact(held) and not held.get("_project_admission_restore_hold")
    system.workers.assign_tasks()
    assert not _launched(system), "every stop still needs the owner's Resume"
    if door == "panic":
        (tmp_path / "state" / "panic_stop.flag").unlink()
    assert [(row["id"], row["_attempt"]) for row in _resume_and_launch(system, turn.id)] == [(turn.id, 1)]
    system.workers.assign_tasks()
    assert not _launched(system)


@pytest.mark.parametrize("evidence", ["launch_mark", "body_start"])
def test_a_pre_body_return_proves_nothing_once_a_launch_or_body_may_have_followed(tmp_path, monkeypatch, evidence):
    from supervisor import task_admission

    system = _system(tmp_path, monkeypatch)
    turn, _event = _parked_before_its_body(system, tmp_path, monkeypatch)
    assert system.queue.resume_budget_paused_task(turn.id)["ok"]
    [row] = system.workers.PENDING
    if evidence == "launch_mark":  # the launch's own commit landed; the process died before the put
        assert task_admission.record_project_dispatch_possible(row)
    elif evidence == "body_start":
        _persist_running(tmp_path, {**row, "_execution_owner": {"kind": "pooled"}}, clock=False)  # its attempt alone
        write_task_result(tmp_path, turn.id, "scheduled")
    _boot_after_stop(system, monkeypatch, tmp_path)
    [held] = system.workers.PENDING
    assert held["_project_admission_restore_hold"] and "admitted_dispatch" not in held
    for _ in range(2):
        system.workers.assign_tasks()
    resumed = system.queue.resume_budget_paused_task(turn.id)
    system.workers.assign_tasks()
    assert not _launched(system), resumed
    [still] = system.workers.PENDING
    assert still["_project_admission_restore_hold"]["reason"] in {"project_dispatch_unconfirmed",
                                                                  "project_routing_fence_lookup_failed"}
    assert not task_admission.prestart_returned_unsent(still)


@pytest.mark.parametrize(("read_gap", "door", "resume_first"), [
    ("proof_reader", "quit", True),
    ("proof_reader", "owner_restart", False),
    ("store", "panic", True),
    ("store", "owner_restart", False),
])
def test_pre_body_readability_recovery_keeps_quit_authority_and_resumes_once(
        tmp_path, monkeypatch, resume_first, door, read_gap):
    from supervisor import task_admission
    from tests._store_shared import unreadable_result

    system = _system(tmp_path, monkeypatch)
    turn, _event = _parked_before_its_body(system, tmp_path, monkeypatch)
    assert system.queue.resume_budget_paused_task(turn.id)["ok"]
    original = load_task_result(tmp_path, turn.id)["resource_limit"]
    if door != "quit":
        flag = "panic_stop.flag" if door == "panic" else "owner_restart_no_resume.flag"
        (tmp_path / "state" / flag).write_text(door, encoding="utf-8")
    with monkeypatch.context() as patched:
        def unreadable(*_a, **_k):
            raise OSError("injected task store failure")

        if read_gap == "store":
            unreadable_result(patched, turn.id, root=tmp_path)
        else:
            patched.setattr(task_admission, "load_task_result", unreadable)
        _boot_after_stop(system, monkeypatch, tmp_path)
        for _ in range(2):
            system.workers.assign_tasks()
        [held] = system.workers.PENDING
        assert held["_project_admission_restore_hold"]
        assert not _launched(system)
        assert not system.queue.resume_budget_paused_task(turn.id)["ok"]
    assert load_task_result(tmp_path, turn.id)["resource_limit"] == original
    if not resume_first:
        system.workers.assign_tasks()
        [held] = system.workers.PENDING
        assert not held.get("_project_admission_restore_hold")
        assert budget_hold_fact(held), "read recovery cannot grant Resume after Quit"
        assert not _launched(system)
    assert [(row["id"], row["_attempt"]) for row in _resume_and_launch(system, turn.id)] == [(turn.id, 1)]
    system.workers.assign_tasks()
    assert not _launched(system)


@pytest.mark.parametrize("changed", ["launch_mark", "stop", "snapshot_failure"])
def test_recovered_pre_body_readability_does_not_override_other_evidence(tmp_path, monkeypatch, changed):
    from supervisor import task_admission
    from ouroboros.projects_registry import bind_task_to_project, create_project

    system = _system(tmp_path, monkeypatch)
    turn, _event = _parked_before_its_body(system, tmp_path, monkeypatch)
    assert system.queue.resume_budget_paused_task(turn.id)["ok"]
    with monkeypatch.context() as patched:
        def unreadable(*_a, **_k):
            raise OSError("injected task store failure")

        patched.setattr(task_admission, "load_task_result", unreadable)
        _boot_after_stop(system, monkeypatch, tmp_path)
    [held] = system.workers.PENDING
    if changed == "launch_mark":
        assert task_admission.record_project_dispatch_possible(held)
    elif changed == "body_start":
        _persist_running(tmp_path, {**held, "_execution_owner": {"kind": "pooled"}}, clock=False)
        write_task_result(tmp_path, turn.id, "scheduled")
    elif changed == "late_binding":
        room = create_project(tmp_path, "late-room")
        bind_task_to_project(tmp_path, turn.id, "late-room", room.get("chat_id"), origin={"absent": "mid_task_no_origin"})
    elif changed == "stop":
        ci.request_cancel(tmp_path, turn.id, reason="owner stop", source="owner", requested_by="owner")
    elif changed == "legacy":
        held.pop("_project_scope_none")
    with monkeypatch.context() as patched:
        if changed == "snapshot_failure":
            assert system.queue.resume_budget_paused_task(turn.id)["ok"]
            patched.setattr(system.queue, "publish_queue_snapshot", lambda **_k: False)
        system.workers.assign_tasks()
        system.queue.resume_budget_paused_task(turn.id)
        system.workers.assign_tasks()
        assert not _launched(system)
    if changed == "snapshot_failure":
        assert system.workers.PENDING[0]["_project_admission_restore_hold"]
        system.workers.assign_tasks()
        assert [(row["id"], row["_attempt"]) for row in _launched(system)] == [(turn.id, 1)]
        system.workers.assign_tasks()
        assert not _launched(system)


@pytest.mark.parametrize("unreadable", [False, True])
def test_a_planned_restart_returns_a_pre_body_queue_row_once(tmp_path, monkeypatch, unreadable):
    from supervisor import restart_retention, task_admission
    from tests.test_restart_saved_work import _ack

    system = _system(tmp_path, monkeypatch)
    turn, _event = _parked_before_its_body(system, tmp_path, monkeypatch)
    assert system.queue.resume_budget_paused_task(turn.id)["ok"]
    restart_retention.prepare_restart_returns(tmp_path, system.workers.RUNNING, system.workers.PENDING,
                                               transaction_id="tx-pre-body")
    _ack(tmp_path, "tx-pre-body")
    with monkeypatch.context() as patched:
        if unreadable:
            def failed_read(*_a, **_k):
                raise OSError("injected task store failure")

            patched.setattr(task_admission, "load_task_result", failed_read)
        _boot_after_stop(system, monkeypatch, tmp_path)
        if unreadable:
            system.workers.assign_tasks()
            assert not _launched(system)
            assert system.workers.PENDING[0]["_project_admission_restore_hold"]
    [row] = system.workers.PENDING
    assert budget_hold_fact(row) is None, "this acknowledged restart already granted return"
    system.workers.assign_tasks()
    assert [(row["id"], row["_attempt"]) for row in _launched(system)] == [(turn.id, 1)]
    system.workers.assign_tasks()
    assert not _launched(system)


def test_a_panic_or_stop_still_binds_a_turn_paused_before_its_body(tmp_path, monkeypatch):
    from ouroboros.cancel_intents import request_cancel

    system = _system(tmp_path, monkeypatch)
    turn, _event = _parked_before_its_body(system, tmp_path, monkeypatch)
    (tmp_path / "state" / "panic_stop.flag").write_text("panic", encoding="utf-8")
    _boot_after_stop(system, monkeypatch, tmp_path)
    system.workers.assign_tasks()
    assert not _launched(system) and [row["id"] for row in system.workers.PENDING] == [turn.id]
    (tmp_path / "state" / "panic_stop.flag").unlink()
    request_cancel(tmp_path, turn.id, reason="owner stop", source="owner", requested_by="owner")
    system.workers.assign_tasks()
    assert not _launched(system) and turn.id not in system.workers.RUNNING
    assert load_task_result(tmp_path, turn.id)["status"] in {"cancelled", "cancel_requested", "interrupted"}


def test_a_running_or_marked_turn_never_acquires_no_dispatch_from_a_park(tmp_path, monkeypatch):
    """Zero recorded calls is not "ran no body": only the carrier's checked pre-body event says that."""
    from supervisor import message_bus
    from tests.conversation_carrier_support import park_lane_pause

    system = _system(tmp_path, monkeypatch)
    monkeypatch.setattr(message_bus, "get_bridge", lambda: SimpleNamespace(push_log=lambda *_a: None))
    turn = _carried_turn(system, tmp_path, metadata={"client_message_id": "cm-main"})
    assert load_task_result(tmp_path, turn.id)["status"] == "running"
    park_lane_pause({"type": "budget_pause", "task_id": turn.id, "worker_id": turn.slot.wid, "resource_limit": {
        "status": "paused_before_dispatch", "scope": "global", "replay_safe": True, "physical_calls": 0}})
    [parked] = system.workers.PENDING
    assert parked["_project_scope_none"] is True and "admitted_dispatch" not in parked


def test_a_pre_body_event_never_replaces_an_existing_dispatch_mark(tmp_path, monkeypatch):
    system = _system(tmp_path, monkeypatch)
    _parked_before_its_body(system, tmp_path, monkeypatch, mark="possible")
    [parked] = system.workers.PENDING
    assert parked["admitted_dispatch"] == "possible"


def test_a_pre_body_event_for_a_later_attempt_is_refused_without_a_mark(tmp_path, monkeypatch):
    from ouroboros import budget_pause
    from ouroboros.owner_pause import read_fence
    from supervisor import message_bus
    from supervisor.owner_pause_control import request_owner_pause
    from tests.conversation_carrier_support import park_lane_pause

    system = _system(tmp_path, monkeypatch)
    monkeypatch.setattr(message_bus, "get_bridge", lambda: SimpleNamespace(push_log=lambda *_a: None))
    turn = _carried_turn(system, tmp_path, metadata={"client_message_id": "cm-main"}, started=False)
    write_task_result(tmp_path, turn.id, "scheduled", chat_id=1, root_task_id=turn.id)
    assert request_owner_pause(turn.id, request_id="pause-main")["ok"]
    owner = SimpleNamespace(executed_seconds=lambda: 0.0, prestart_pause_fence=read_fence(tmp_path, turn.id), drive_root=tmp_path)
    error = SimpleNamespace(before_task_start=True, control_reason=budget_pause.RAIL_OWNER_PAUSE)
    event = budget_pause.prestart_pause_event(turn.task, error, owner)
    system.workers.RUNNING[turn.id]["attempt"] = 2  # the row is already a successor: this event is not its evidence
    with pytest.raises(ValueError, match="no-body evidence"):
        park_lane_pause(event, worker_id=turn.slot.wid)
    assert turn.id in system.workers.RUNNING and "admitted_dispatch" not in system.workers.RUNNING[turn.id]["task"]


def _move_mid_run(system, root, monkeypatch, task_id, pid="moved-room"):
    from ouroboros.project_placement import move_task_to_project, placement_basis

    monkeypatch.setattr("supervisor.message_bus.get_bridge", lambda: SimpleNamespace(broadcast=lambda _event: None))
    move_task_to_project(root, task_id, pid, origin={"absent": "mid_task_no_origin"},
                         basis=placement_basis(root, task_id, pid), running=system.workers.RUNNING,
                         pending=system.workers.PENDING, initiated_by="owner")
    return pid


@pytest.mark.parametrize("door", ["aborted_update", "quit_restore"])
def test_a_main_turn_moved_mid_run_keeps_its_room_not_its_old_absence(tmp_path, monkeypatch, door):
    from ouroboros.projects_registry import get_project

    system = _system(tmp_path, monkeypatch)
    turn = _carried_turn(system, tmp_path, metadata={"client_message_id": "cm-main"})
    _save_work(tmp_path, turn)
    pid = _move_mid_run(system, tmp_path, monkeypatch, turn.id)
    assert system.workers.RUNNING[turn.id]["task"]["_project_scope_none"] is True, "RUNNING keeps its lease until it leaves"
    if door == "aborted_update":
        assert system.control._quiesce_repo_writers("auto_merge") == []
        system.control._respawn_workers_after_failed_update()
        system.workers.assign_tasks()
        sent = _launched(system)
    else:
        _boot_after_stop(system, monkeypatch, tmp_path)
        system.workers.assign_tasks()
        assert not _launched(system)
        sent = _resume_and_launch(system, turn.id)
    assert [row["id"] for row in sent] == [turn.id], system.workers.PENDING
    room = get_project(tmp_path, pid)
    assert sent[0]["project_id"] == pid and sent[0]["chat_id"] == room["chat_id"]
    assert sent[0]["_project_admission"]["project_id"] == pid and "_project_scope_none" not in sent[0]


@pytest.mark.parametrize("door", ["aborted_update", "quit_restore"])
def test_a_project_room_seed_keeps_its_bound_basis_and_never_claims_absence(tmp_path, monkeypatch, door):
    from ouroboros.projects_registry import create_project, project_binding_for_task

    system = _system(tmp_path, monkeypatch)
    folder = tmp_path / "room-folder"
    folder.mkdir()
    create_project(tmp_path, "room", working_dir=str(folder))
    turn = _carried_turn(system, tmp_path, metadata={"client_message_id": "cm-room", "project_id": "room"})
    assert "_project_scope_none" not in turn.published
    row = system.workers.RUNNING[turn.id]["task"]
    basis = dict(row["_project_admission"])
    assert basis["project_id"] == "room" and project_binding_for_task(tmp_path, turn.id)["project_id"] == "room"
    _save_work(tmp_path, turn)
    if door == "aborted_update":
        assert system.control._quiesce_repo_writers("auto_merge") == []
        system.control._respawn_workers_after_failed_update()
        system.workers.assign_tasks()
        sent = _launched(system)
    else:
        _boot_after_stop(system, monkeypatch, tmp_path)
        sent = _resume_and_launch(system, turn.id)
    assert [item["id"] for item in sent] == [turn.id], system.workers.PENDING
    assert sent[0]["project_id"] == "room" and sent[0]["_project_admission"]["project_id"] == "room"
    assert sent[0]["_project_admission"]["project"] == basis["project"] and "_project_scope_none" not in sent[0]


@pytest.mark.parametrize("carried", [
    {"project_id": "room"}, {"resource_intent": {"kind": "explicit_resource", "project_id": "room"}},
    {"parent_task_id": "pred"},
])
def test_a_seed_carrying_a_room_or_lineage_is_never_attested_unscoped(carried):
    from supervisor.worker_chat_lane import _seed_task

    assert "_project_scope_none" not in _seed_task(1, "work", None, None, {"client_message_id": "cm", **carried})
    assert _seed_task(1, "work", None, None, {"client_message_id": "cm"})["_project_scope_none"] is True
    assert _seed_task(1, "work", None, None, None)["_project_scope_none"] is True


def test_a_presence_seed_stays_unmarked():
    from ouroboros import presence_carrier
    from ouroboros.presence_events import PresenceTurnEvent

    event = PresenceTurnEvent("event1", "chat", "a", "r", "", "chat:a:r", {"platform_actor_id": "person"}, {}, {}, "hello")
    admission = SimpleNamespace(binding_id="binding", transport_skill="skill", allowance=None)
    task = presence_carrier.presence_seed_task(admission=admission, event=event, skill_name="skill", event_payload={},
                                               staged_files=(), gate_wait=False)
    assert "_project_scope_none" not in task


def test_a_direct_turn_bound_frame_replaces_stale_absence_with_the_true_basis():
    from supervisor.conversation_lane import handle_direct_turn_bound

    basis = {"project_id": "room", "project": None, "frozen": True}
    task = {"id": "t", "_project_scope_none": True}
    running = {"t": {"task": task, "attempt": 1}}
    handle_direct_turn_bound({"task_id": "t", "task_attempt": 1, "facts": {"project_id": "room"}},
                             SimpleNamespace(RUNNING=running))
    assert task["_project_scope_none"] is True, "no basis landed: nothing replaces the mark"
    handle_direct_turn_bound({"task_id": "t", "task_attempt": 1, "facts": {"_project_admission": basis, "project_id": "room"}},
                             SimpleNamespace(RUNNING=running))
    assert task["_project_admission"] == basis and task["project_id"] == "room" and "_project_scope_none" not in task


def _aborted(system, root, turn=None):
    assert system.control._quiesce_repo_writers("auto_merge") == []
    system.control._respawn_workers_after_failed_update()
    system.workers.assign_tasks()
    system.workers.assign_tasks()
    assert not _launched(system)
    assert not turn or turn.id not in system.workers.RUNNING


def test_a_legacy_main_row_without_the_witness_stays_held_with_its_old_refusal(tmp_path, monkeypatch):
    system = _system(tmp_path, monkeypatch)
    turn = _carried_turn(system, tmp_path, metadata={"client_message_id": "cm-main"})
    _save_work(tmp_path, turn)
    system.workers.RUNNING[turn.id]["task"].pop("_project_scope_none")  # a row admitted before this release
    _aborted(system, tmp_path, turn)
    [held] = system.workers.PENDING
    assert "_project_scope_none" not in held, "restore never backfills historical scope from absence"
    assert held["_project_admission_restore_hold"]["reason"] == "project_routing_fence_lookup_failed"
    assert "missing or invalid" in held["_project_admission_restore_hold"]["detail"]


@pytest.mark.parametrize("when", ["before_quiesce", "after_quiesce"])
def test_an_owner_stop_of_a_main_turn_is_never_returned(tmp_path, monkeypatch, when):
    from ouroboros.cancel_intents import request_cancel

    system = _system(tmp_path, monkeypatch)
    turn = _carried_turn(system, tmp_path, metadata={"client_message_id": "cm-main"})
    _save_work(tmp_path, turn)
    if when == "before_quiesce":
        request_cancel(tmp_path, turn.id, reason="owner stop", source="owner", requested_by="owner")
    assert system.control._quiesce_repo_writers("auto_merge") == []
    if when == "after_quiesce":
        request_cancel(tmp_path, turn.id, reason="owner stop", source="owner", requested_by="owner")
    system.control._respawn_workers_after_failed_update()
    system.workers.assign_tasks()
    assert not _launched(system) and turn.id not in system.workers.RUNNING
    assert load_task_result(tmp_path, turn.id)["status"] in {"cancelled", "interrupted"}


def test_a_panic_holds_a_main_turn_for_an_explicit_resume(tmp_path, monkeypatch):
    system = _system(tmp_path, monkeypatch)
    turn = _carried_turn(system, tmp_path, metadata={"client_message_id": "cm-main"})
    _save_work(tmp_path, turn)
    assert system.control._quiesce_repo_writers("auto_merge") == []
    monkeypatch.setattr("ouroboros.config.DATA_DIR", tmp_path)
    monkeypatch.setattr("ouroboros.server_control._restart_stop_requested", True)
    system.control._respawn_workers_after_failed_update()
    system.workers.assign_tasks()
    assert not _launched(system)
    [held] = system.workers.PENDING
    assert held["id"] == turn.id and budget_hold_fact(held)


@pytest.mark.parametrize("failure", ["corrupt_source", "missing_source", "unreadable_result"])
def test_unproven_source_or_state_keeps_a_main_turn_held(tmp_path, monkeypatch, failure):
    system = _system(tmp_path, monkeypatch)
    turn = _carried_turn(system, tmp_path, metadata={"client_message_id": "cm-main"})
    _save_work(tmp_path, turn)
    assert system.control._quiesce_repo_writers("auto_merge") == []
    checkpoint = wc.checkpoint_path(tmp_path, turn.id, 1)
    if failure == "corrupt_source":
        checkpoint.write_text("corrupt")
    elif failure == "missing_source":
        checkpoint.unlink()
    system.control._respawn_workers_after_failed_update()
    [row] = [item for item in system.workers.PENDING if item["id"] == turn.id]
    if failure == "mismatched_source":
        row["_working_recovery"] = {**row["_working_recovery"], "source_ref": "working-a1-s1-" + "0" * 64 + ".json"}
    elif failure == "unreadable_result":
        def unreadable(*_a, **_k):
            raise OSError("injected task store failure")

        monkeypatch.setattr("supervisor.task_admission.load_task_result", unreadable)
    system.workers.assign_tasks()
    system.workers.assign_tasks()
    assert not _launched(system) and turn.id not in system.workers.RUNNING
    [held] = [item for item in system.workers.PENDING if item["id"] == turn.id]
    assert held.get("_project_admission_restore_hold") or budget_hold_fact(held)
    assert not held.get("_terminalization_retry"), "unknown is custody, never a terminal"


def test_a_late_binding_without_a_placement_grant_never_launches_an_attested_main_turn(tmp_path, monkeypatch):
    from ouroboros.projects_registry import bind_task_to_project, create_project

    system = _system(tmp_path, monkeypatch)
    turn = _carried_turn(system, tmp_path, metadata={"client_message_id": "cm-main"})
    _save_work(tmp_path, turn)
    room = create_project(tmp_path, "late-room")
    bind_task_to_project(tmp_path, turn.id, "late-room", room.get("chat_id"), origin={"absent": "mid_task_no_origin"})
    _aborted(system, tmp_path, turn)
    assert not system.workers.PENDING, "the attested scope changed: the existing settlement owns it"
    assert load_task_result(tmp_path, turn.id)["status"] == "failed"


def test_a_lineage_seed_bound_to_a_room_is_held_uncertain_not_failed_as_a_false_absence(tmp_path, monkeypatch):
    from ouroboros.projects_registry import bind_task_to_project, create_project

    system = _system(tmp_path, monkeypatch)
    room = create_project(tmp_path, "lineage-room")
    bind_task_to_project(tmp_path, "pred", "lineage-room", room.get("chat_id"), origin={"absent": "mid_task_no_origin"})
    turn = _carried_turn(system, tmp_path, metadata={"client_message_id": "cm-main", "origin_task_id": "pred"})
    assert "_project_scope_none" not in turn.published
    _save_work(tmp_path, turn)
    _aborted(system, tmp_path, turn)
    [held] = system.workers.PENDING
    assert held["_project_admission_restore_hold"] and not held.get("_terminalization_retry")


def test_a_main_turn_held_across_a_second_quit_still_waits_for_one_resume(tmp_path, monkeypatch):
    system = _system(tmp_path, monkeypatch)
    turn = _carried_turn(system, tmp_path, metadata={"client_message_id": "cm-main"})
    _save_work(tmp_path, turn)
    _boot_after_stop(system, monkeypatch, tmp_path)
    source = system.workers.PENDING[0]["_working_recovery"]
    _boot_after_stop(system, monkeypatch, tmp_path)
    [held] = system.workers.PENDING
    assert budget_hold_fact(held) and held["_attempt"] == 2 and held["_working_recovery"] == source
    system.workers.assign_tasks()
    assert not _launched(system)
    assert [(row["id"], row["_attempt"]) for row in _resume_and_launch(system, turn.id)] == [(turn.id, 2)]
    system.workers.assign_tasks()
    assert not _launched(system)


# ---------------------------------------------------------------------------
# D5: a returning stop gives back an owner turn it cut before its first save (05.11, 88cb09a6)
# ---------------------------------------------------------------------------

def _assert_seed_runs_once(system, root, turn, cause):
    from ouroboros import body_candidate
    from supervisor.task_admission import SEED_RETURN

    [row] = [row for row in system.workers.PENDING if row["id"] == turn.id]
    assert row[SEED_RETURN] == {"source_task_id": turn.id, "from_attempt": 1, "cause": cause}
    assert row["_attempt"] == 2 and "_working_recovery" not in row and budget_hold_fact(row) is None
    assert row["text"] == "keep working" and row["_project_scope_none"] is True  # its own admitted input
    # Nothing of it took effect: no child was scheduled and no body candidate was opened (both are gated tools).
    assert [r["id"] for r in system.workers.PENDING] == [turn.id] and body_candidate.find(turn.id) is None
    system.workers.assign_tasks()
    [sent] = _launched(system)
    assert (sent["id"], sent["_attempt"]) == (turn.id, 2) and "_working_recovery" not in sent
    _persist_running(root, sent)  # its launch records attempt 2: the proof is spent
    from supervisor.task_admission import prestart_returned_unsent

    assert prestart_returned_unsent(sent) == {}
    system.workers.assign_tasks()
    assert not _launched(system)


def test_a_turn_cut_before_its_first_save_comes_back_once_from_its_own_input(tmp_path, monkeypatch):
    """05.11 (88cb09a6): the owner's Main message was cut 17 s into its preparation and lost: no working save
    existed, so the returning stop had nothing to give back. Every tool batch waits for its pre_effect save, so
    with the task's root present no save means nothing took effect: the return starts it again from its input."""
    from supervisor.restart_retention import prepare_restart_returns
    from tests.test_restart_saved_work import _ack

    system = _system(tmp_path, monkeypatch)
    turn = _carried_turn(system, tmp_path, metadata={"client_message_id": "cm-main"})
    assert not wc.checkpoint_path(tmp_path, turn.id, 1).is_file()
    assert prepare_restart_returns(tmp_path, system.workers.RUNNING, system.workers.PENDING,
                                   transaction_id="tx-seed") == {turn.id}
    _ack(tmp_path, "tx-seed")
    _boot_after_stop(system, monkeypatch, tmp_path)
    _assert_seed_runs_once(system, tmp_path, turn, "restart")


@pytest.mark.parametrize("outcome", ["aborted", "completed"])
def test_the_update_door_gives_back_a_turn_it_cut_in_preparation(tmp_path, monkeypatch, outcome):
    """88cb09a6 replayed on the managed Update door, whose census and stop are the ones that cut it."""
    from ouroboros import delegate_recovery as dr
    from tests.test_restart_saved_work import _ack

    system = _system(tmp_path, monkeypatch)
    turn = _carried_turn(system, tmp_path, metadata={"client_message_id": "cm-main"})
    assert system.control._quiesce_repo_writers("auto_merge") == []
    assert dr.active_restart_transaction(tmp_path)["return_ids"] == [turn.id]
    if outcome == "aborted":
        system.control._respawn_workers_after_failed_update()
        _assert_seed_runs_once(system, tmp_path, turn, "update_aborted")
    else:
        _ack(tmp_path, dr.active_restart_transaction(tmp_path)["transaction_id"])
        _boot_after_stop(system, monkeypatch, tmp_path)
        _assert_seed_runs_once(system, tmp_path, turn, "restart")


def test_an_unacknowledged_return_keeps_todays_interrupted_path(tmp_path, monkeypatch):
    """The exit was never acknowledged: the turn the door named is neither returned nor held, its boot fences it."""
    from supervisor.restart_retention import prepare_restart_returns

    system = _system(tmp_path, monkeypatch)
    turn = _carried_turn(system, tmp_path, metadata={"client_message_id": "cm-main"})
    assert prepare_restart_returns(tmp_path, system.workers.RUNNING, system.workers.PENDING,
                                   transaction_id="tx-lost") == {turn.id}
    _boot_after_stop(system, monkeypatch, tmp_path)
    assert turn.id not in [row["id"] for row in system.workers.PENDING] and turn.id not in system.workers.RUNNING
    assert ci.has_active_intent(tmp_path, turn.id, strict=True)


@pytest.mark.parametrize("fact", ["nothing", "saved", "answer_delivered", "no_root", "call", "wake", "later_attempt",
                                  "stopped", "model_called", "ledger_unreadable", "ledger_degraded"])
def test_only_a_turn_with_no_effect_is_given_back_from_its_input(tmp_path, monkeypatch, fact):
    from supervisor.task_admission import seed_returnable
    from supervisor.terminal_delivery import delivery_id_for, register_delivery

    system = _system(tmp_path, monkeypatch)
    turn = _carried_turn(system, tmp_path, metadata={"client_message_id": "cm-main"})
    task, attempt, root = dict(system.workers.RUNNING[turn.id]["task"]), 1, tmp_path
    if fact == "saved":  # the ordinary saved return takes it instead
        _save_work(tmp_path, turn)
    elif fact == "answer_delivered":  # a no-tool answer saved non-gating, then sent
        register_delivery(tmp_path, delivery_id_for(turn.id, "done"))
    elif fact == "no_root":  # nothing could have been saved, so no file proves nothing
        root = tmp_path / "missing-root"
    elif fact == "call":
        task["metadata"] = {**task.get("metadata", {}), "execution_transport": {"kind": "live_conversation"}}
    elif fact == "wake":
        from ouroboros.consciousness_authority import WAKE_CAUSE_OWN_CLOCK

        task["metadata"] = {**task.get("metadata", {}), "wake_cause": WAKE_CAUSE_OWN_CLOCK}
    elif fact == "later_attempt":
        attempt = 2
    elif fact == "stopped":
        ci.request_cancel(tmp_path, turn.id, reason="owner stop", source="test")
    elif fact in {"model_called", "ledger_unreadable", "ledger_degraded"}:  # the ledger's verdict, at the seam
        ledger = {"model_called": {"cost_accounting_status": "available", "total_rounds": 1},
                  "ledger_unreadable": {"cost_accounting_status": "unavailable", "total_rounds": None},
                  "ledger_degraded": {"cost_accounting_status": "available", "total_rounds": 0,
                                      "ledger_integrity_degraded": True}}[fact]
        monkeypatch.setattr(system.queue, "reconstruct_task_cost", lambda *_a, **_k: ledger)
    assert seed_returnable(root, task, attempt) is (fact == "nothing")


@pytest.mark.parametrize("door", ["acknowledged_restart", "owner_restart", "aborted_update"])
def test_a_project_room_turn_cut_before_its_first_save_comes_back_on_its_bound_basis(tmp_path, monkeypatch, door):
    """A room turn's seed carries the basis it was bound with; Project-hold revalidation admits it on its D5 proof."""
    from ouroboros.projects_registry import create_project
    from supervisor.restart_retention import prepare_restart_returns
    from supervisor.task_admission import SEED_RETURN
    from tests.test_restart_saved_work import _ack

    system = _system(tmp_path, monkeypatch)
    folder = tmp_path / "room-folder"
    folder.mkdir()
    create_project(tmp_path, "room", working_dir=str(folder))
    turn = _carried_turn(system, tmp_path, metadata={"client_message_id": "cm-room", "project_id": "room"})
    basis = dict(system.workers.RUNNING[turn.id]["task"]["_project_admission"])
    if door == "aborted_update":
        assert system.control._quiesce_repo_writers("auto_merge") == []
        system.control._respawn_workers_after_failed_update()
    else:  # the owner's own Restart door holds what it does not return (``hold_stopped_queue``)
        from ouroboros import delegate_recovery as dr

        tx = dr.prepare_owner_restart(tmp_path)["transaction_id"] if door == "owner_restart" else "tx-room"
        assert prepare_restart_returns(tmp_path, system.workers.RUNNING, system.workers.PENDING,
                                       transaction_id=tx) == {turn.id}
        _ack(tmp_path, tx)
        _stop(system)
        monkeypatch.setattr(dr, "_PROCESS_GENERATION", "the-next-generation")  # boot reads a prior generation's door
        _boot(system, monkeypatch, tmp_path)
    [row] = [row for row in system.workers.PENDING if row["id"] == turn.id]
    assert row[SEED_RETURN]["from_attempt"] == 1 and row["_attempt"] == 2
    system.workers.assign_tasks()
    [sent] = _launched(system)
    assert (sent["id"], sent["_attempt"]) == (turn.id, 2) and sent["project_id"] == "room"
    assert sent["_project_admission"]["project"] == basis["project"] and "_project_scope_none" not in sent
    system.workers.assign_tasks()
    assert not _launched(system)


@pytest.mark.parametrize("second", ["acknowledged_restart", "quit"])
def test_a_seed_caught_again_before_its_launch_stays_recoverable(tmp_path, monkeypatch, second):
    """The returned turn waits for a slot when a second stop comes: its published row keeps the proof, so an
    acknowledged return gives it back once and a Quit holds it for a Resume that launches it once."""
    from supervisor.events_budget import HOLD_SAVED_WORK
    from supervisor.restart_retention import prepare_restart_returns
    from supervisor.task_admission import SEED_RETURN
    from tests.test_restart_saved_work import _ack

    system = _system(tmp_path, monkeypatch)
    turn = _carried_turn(system, tmp_path, metadata={"client_message_id": "cm-main"})
    assert prepare_restart_returns(tmp_path, system.workers.RUNNING, system.workers.PENDING,
                                   transaction_id="tx-seed") == {turn.id}
    _ack(tmp_path, "tx-seed")
    _boot_after_stop(system, monkeypatch, tmp_path)
    assert system.queue.publish_queue_snapshot(reason="seed_waits") is not False
    [published] = json.loads(system.queue.QUEUE_SNAPSHOT_PATH.read_text())["pending"]
    assert published["task"][SEED_RETURN] == {"source_task_id": turn.id, "from_attempt": 1, "cause": "restart"}
    if second == "acknowledged_restart":
        assert prepare_restart_returns(tmp_path, system.workers.RUNNING, system.workers.PENDING,
                                       transaction_id="tx-second") == set()  # a queued row: named in queue_ids
        _ack(tmp_path, "tx-second")
    _boot_after_stop(system, monkeypatch, tmp_path)
    [row] = system.workers.PENDING
    assert row[SEED_RETURN]["from_attempt"] == 1 and row["_attempt"] == 2
    if second == "quit":
        assert budget_hold_fact(row)["reason"] == HOLD_SAVED_WORK
        system.workers.assign_tasks()
        assert not _launched(system), "only the owner's Resume releases a quit's saved work"
        sent = _resume_and_launch(system, turn.id)
    else:
        system.workers.assign_tasks()
        sent = _launched(system)
    assert [(item["id"], item["_attempt"]) for item in sent] == [(turn.id, 2)]
    system.workers.assign_tasks()
    assert not _launched(system)


def test_an_answer_sent_after_the_stops_mark_is_never_given_back(tmp_path, monkeypatch):
    """A degraded in-process executor is only asked to stop: an answer it delivers after the stop marked the
    turn is an effect, so the boot's re-check keeps today's path instead of answering again."""
    from supervisor.restart_retention import prepare_restart_returns
    from supervisor.task_admission import SEED_RETURN
    from supervisor.terminal_delivery import delivery_id_for, register_delivery
    from tests.test_restart_saved_work import _ack

    system = _system(tmp_path, monkeypatch)
    turn = _carried_turn(system, tmp_path, metadata={"client_message_id": "cm-main"})
    assert prepare_restart_returns(tmp_path, system.workers.RUNNING, system.workers.PENDING,
                                   transaction_id="tx-late") == {turn.id}
    _ack(tmp_path, "tx-late")
    _stop(system)
    [kept] = [row for row in json.loads(system.queue.QUEUE_SNAPSHOT_PATH.read_text())["running"] if row["id"] == turn.id]
    assert kept["task"][SEED_RETURN] == {"source_task_id": turn.id, "from_attempt": 1}  # the stop's own mark
    register_delivery(tmp_path, delivery_id_for(turn.id, "the answer"))
    _boot(system, monkeypatch, tmp_path)
    assert turn.id not in [row["id"] for row in system.workers.PENDING] and turn.id not in system.workers.RUNNING
    assert ci.has_active_intent(tmp_path, turn.id, strict=True)


def test_a_quit_without_a_returning_door_keeps_the_stops_own_terminal(tmp_path, monkeypatch):
    """Only a returning door's transaction names a seed: a Quit retains nothing unsaved and ends it as before."""
    from ouroboros.task_results import _TRULY_TERMINAL_STATUSES

    system = _system(tmp_path, monkeypatch)
    turn = _carried_turn(system, tmp_path, metadata={"client_message_id": "cm-main"})
    _stop(system)
    running = json.loads(system.queue.QUEUE_SNAPSHOT_PATH.read_text())["running"]
    assert turn.id not in [row["id"] for row in running]
    assert load_task_result(tmp_path, turn.id)["status"] in _TRULY_TERMINAL_STATUSES


@pytest.mark.parametrize("launch", ["none", "dispatch_mark", "recorded_start", "terminal"])
def test_a_seed_proof_holds_until_the_attempt_it_gave_back_launches(launch):
    """The proof rides the row by value; a dispatch mark or a recorded start of that attempt spends it."""
    from supervisor.task_admission import SEED_RETURN, prestart_returned_unsent

    row = {"id": "seed", "_is_direct_chat": True, "_attempt": 2,
           SEED_RETURN: {"source_task_id": "seed", "from_attempt": 1, "cause": "restart"}}
    stored = {"task_id": "seed", "status": "running", "task_attempt": 0,
              **({"admitted_dispatch_attempt": 2} if launch == "dispatch_mark" else {}),
              **({"task_attempt": 2} if launch == "recorded_start" else {}),
              **({"status": "completed"} if launch == "terminal" else {})}
    proof = prestart_returned_unsent(row, stored)
    assert proof == ({"seed_return": True, "from_attempt": 1, "cause": "restart"} if launch == "none" else {})


def test_a_source_lost_after_a_real_save_before_the_door_decides_is_never_a_seed(tmp_path, monkeypatch):
    """The working file can vanish before a stop looks (RV1 astra #2): its turn already called the model, which the
    usage ledger keeps apart from that file, so the door names nothing and the stop ends it as before."""
    from supervisor.restart_retention import prepare_restart_returns
    from tests.test_restart_saved_work import _ack

    system = _system(tmp_path, monkeypatch)
    turn = _carried_turn(system, tmp_path, metadata={"client_message_id": "cm-main"})
    _save_work(tmp_path, turn)  # a pre_effect save: the model answered with a tool batch
    monkeypatch.setattr(system.queue, "reconstruct_task_cost",
                        lambda *_a, **_k: {"cost_accounting_status": "available", "total_rounds": 1})
    wc.checkpoint_path(tmp_path, turn.id, 1).unlink()
    assert prepare_restart_returns(tmp_path, system.workers.RUNNING, system.workers.PENDING,
                                   transaction_id="tx-lost-source") == set()
    _ack(tmp_path, "tx-lost-source")
    _boot_after_stop(system, monkeypatch, tmp_path)
    assert turn.id not in [row["id"] for row in system.workers.PENDING] and turn.id not in system.workers.RUNNING


@pytest.mark.parametrize("state", ["none", "reserved", "released", "dispatched", "settled", "unresolved"])
def test_a_seed_reads_the_real_usage_ledger_of_its_own_task(tmp_path, monkeypatch, state):
    """The real ledger, not a stub: a call reserved or released unsent leaves the seed; one dispatched never does."""
    from ouroboros import usage_accounting as ua
    from ouroboros.usage_origin import UsageScope
    from supervisor.task_admission import seed_returnable

    monkeypatch.setenv("OUROBOROS_SETTINGS_PATH", str(tmp_path / "settings.json"))
    monkeypatch.setenv("TOTAL_BUDGET", "1000000")
    system = _system(tmp_path, monkeypatch)
    turn = _carried_turn(system, tmp_path, metadata={"client_message_id": "cm-main"})
    if state != "none":
        scope = UsageScope(drive_root=tmp_path, task_id=turn.id, root_task_id=turn.id, category="task", source="agent.task")
        with ua.usage_scope(scope):
            held = ua.reserve_attempt(ua.AttemptRequest(model="stub", provider="local", reservation_usd=0.01,
                                                        drive_root=tmp_path, task_id=turn.id, root_task_id=turn.id))
            if state == "released":
                ua.release_attempt(held, "not sent")
            elif state != "reserved":
                ua.mark_dispatched(held)
                if state == "settled":
                    ua.settle_attempt(held, {}, cost_usd=0.01, cost_final=True)
                elif state == "unresolved":
                    ua.mark_unresolved(held, "killed in flight")
    task = system.workers.RUNNING[turn.id]["task"]
    assert seed_returnable(tmp_path, task, 1) is (state in {"none", "reserved", "released"})


def test_a_returned_seed_at_an_exhausted_budget_is_paused_before_dispatch_not_failed(tmp_path, monkeypatch):
    """Its given-back attempt never put: a spent budget pauses it before dispatch, and one Resume launches it."""
    from supervisor import worker_assignment
    from supervisor.restart_retention import prepare_restart_returns
    from tests.test_restart_saved_work import _ack

    system = _system(tmp_path, monkeypatch)
    turn = _carried_turn(system, tmp_path, metadata={"client_message_id": "cm-main"})
    assert prepare_restart_returns(tmp_path, system.workers.RUNNING, system.workers.PENDING,
                                   transaction_id="tx-budget") == {turn.id}
    _ack(tmp_path, "tx-budget")
    _boot_after_stop(system, monkeypatch, tmp_path)
    with monkeypatch.context() as spent:
        spent.setattr(system.state, "budget_remaining", lambda *_a, **_k: 0.0)
        spent.setattr(worker_assignment, "needs_paid_admission", lambda _task: True)
        system.workers.assign_tasks()
    [row] = [row for row in system.workers.PENDING if row["id"] == turn.id]
    assert row["_budget_pause"]["replay_safe"] is True and load_task_result(tmp_path, turn.id)["status"] != "failed"
    assert not _launched(system)
    sent = _resume_and_launch(system, turn.id)
    assert [(item["id"], item["_attempt"]) for item in sent] == [(turn.id, 2)]


# ---------------------------------------------------------------------------
# #2870 / #2725: my human's returned turn keeps its conversation lane (owner choice of 9 November 2026: the
# conversation path survives every permitted return; a fenced, full or leased project pool never holds it)
# ---------------------------------------------------------------------------

def _owner_turn(system, root, monkeypatch, *, chat_id=1, bound=True, text="keep working"):
    """My human's own turn as the host admits it: the real intake (owner routing stamps its ingress), the lane's
    admission onto an idle carrier, the carrier's real first step (``bound``: its room frame folded into the RUNNING
    row, as the event loop does when no stop comes first) and the body's own start record."""
    from supervisor import message_bus
    from supervisor.conversation_lane import handle_direct_turn_bound
    from supervisor.worker_chat_lane import prepare_conversation_task
    from tests._main_followup import _Bridge, _say

    monkeypatch.setattr(message_bus, "log_chat", lambda *_a, **_k: None)
    slot, commands = _lane_carrier(system)
    ctx = SimpleNamespace(DRIVE_ROOT=root, RUNNING=system.workers.RUNNING, PENDING=system.workers.PENDING,
                          BUDGET_ROOT_FENCES={}, load_state=lambda: {"owner_id": 1, "owner_chat_id": 1},
                          update_state=lambda fn: fn({"owner_id": 1, "owner_chat_id": 1}),
                          consciousness=SimpleNamespace(), handle_chat_direct=system.workers.handle_chat_direct,
                          send_with_budget=lambda *a, **k: None)
    before = set(system.workers.RUNNING)
    _say(_Bridge(), ctx, text, "cm-" + text.replace(" ", "-"), chat_id=chat_id)
    [task_id] = set(system.workers.RUNNING) - before
    dispatched = commands.get_nowait()
    frames: list = []
    prepare_conversation_task(SimpleNamespace(), dispatched, root, broadcast=frames.append)
    if bound:
        for frame in frames:
            handle_direct_turn_bound(frame, SimpleNamespace(RUNNING=system.workers.RUNNING))
    _persist_running(root, dispatched)
    return SimpleNamespace(id=task_id, slot=slot, task=dispatched, frames=frames)


def _carried(system):
    """What the conversation lane's carriers were handed since the last look, as (slot, task)."""
    sent = []
    for slot in list(system.workers.WORKERS.values()):
        while getattr(slot, "lane", "") == "conversation" and not slot.in_q.empty():
            sent.append((slot.wid, slot.in_q.get_nowait()))
    return sent


def _on_lane(system, root, turn_id, attempt):
    """The returned turn holds a lane carrier as a lane row: its origin and carrier, its admission's client id and
    reply fact, no fresh identity, the attempt's dispatch mark committed before the put."""
    meta = system.workers.RUNNING[turn_id]
    owner = meta["task"]["_execution_owner"]
    assert (meta["lane"], meta["attempt"], meta["reply_expected"], meta["fresh_identity"]) == (
        "conversation", attempt, True, False), meta
    assert system.workers.WORKERS[meta["worker_id"]].lane == "conversation"
    assert (owner["kind"], owner["lane"], owner["executor"], owner["task_attempt"]) == (
        "direct", "conversation", "worker", attempt), owner
    assert load_task_result(root, turn_id)["admitted_dispatch_attempt"] == attempt
    return meta


def test_a_returned_owner_turn_takes_the_lanes_carrier_beside_a_full_project_pool(tmp_path, monkeypatch):
    """#2870: an acknowledged Restart returns background work and my human's Main turn; the one project slot
    (``MAX_WORKERS`` 1) goes to the background work, the turn to the lane's carrier in the same pass."""
    from supervisor.restart_retention import prepare_restart_returns
    from tests.test_restart_saved_work import _ack

    system = _system(tmp_path, monkeypatch)
    system.saved("bg")  # older background work holding the project slot
    turn = _owner_turn(system, tmp_path, monkeypatch)
    _save_work(tmp_path, turn)
    assert system.workers.RUNNING[turn.id]["task"]["metadata"]["origin_message_ref"], "the owner door stamped it"
    returns = prepare_restart_returns(tmp_path, system.workers.RUNNING, system.workers.PENDING, transaction_id="tx-a")
    assert returns == {"bg", turn.id}
    _ack(tmp_path, "tx-a")
    _boot_after_stop(system, monkeypatch, tmp_path)
    [restored] = [row for row in system.workers.PENDING if row["id"] == turn.id]
    assert restored["_execution_owner"]["lane"] == "conversation" and budget_hold_fact(restored) is None
    system.workers.assign_tasks()
    [(wid, sent)] = _carried(system)
    assert (sent["id"], sent["_attempt"]) == (turn.id, 2) and sent["_working_recovery"]["from_attempt"] == 1
    meta = _on_lane(system, tmp_path, turn.id, 2)
    assert meta["worker_id"] == wid and meta["client_message_id"] == "cm-keep-working"
    assert [row["id"] for row in _launched(system)] == ["bg"], "the project slot took only the background work"
    assert sum(1 for w in system.workers.WORKERS.values() if w.lane != "conversation") == 1, "the one project slot"
    system.workers.assign_tasks()
    assert not _carried(system) and not _launched(system) and not system.workers.PENDING


def _project_work(system, root, project_id, task_id="bg"):
    """Background work in a Project room as the host runs it: admitted with the room's basis, launched on a project
    slot (its dispatch marks), its body's start record and a saved working state: the room's lease holder."""
    from ouroboros.projects_registry import get_project, project_scope_admission

    admitted = system.queue.enqueue_task(
        {"id": task_id, "type": "task", "chat_id": get_project(root, project_id)["chat_id"], "text": "background",
         "root_task_id": task_id, "project_id": project_id},
        project_admission=project_scope_admission(root, project_id=project_id))
    assert not admitted.get("_admission_blocked") and "_project_scope_none" not in admitted, admitted
    _project_slot(system)
    system.workers.assign_tasks()
    [sent] = _launched(system)
    _persist_running(root, sent)
    _ctx, limit = _loop_ctx(root, task_id, attempt=1)
    wc.save_round(limit, "pre_effect")
    return sent


def _project_slot(system, wid=40):
    """A ready project-pool slot, its commands seen by ``_launched``."""
    from tests.worker_pool_support import affirm_ready

    proc = _Process(720000 + wid)
    system.processes[proc.pid] = proc
    commands = stdqueue.Queue()
    system.launches.append(commands)
    system.workers.WORKERS[wid] = affirm_ready(system.workers.Worker(wid, proc, commands))


def _registered_room(root, project_id="room"):
    from ouroboros.projects_registry import create_project, get_project

    folder = root / f"{project_id}-folder"
    folder.mkdir()
    create_project(root, project_id, working_dir=str(folder))
    return get_project(root, project_id)


def test_a_returned_room_turn_takes_a_carrier_while_its_rooms_lease_is_held(tmp_path, monkeypatch):
    """#2870: background work in the room holds the room's one-writer lease; my human's returned room turn holds
    none and continues on its lane in the same pass, the background work on its project slot."""
    from ouroboros.project_lease import running_project_ids
    from supervisor.restart_retention import prepare_restart_returns
    from tests.test_restart_saved_work import _ack

    system = _system(tmp_path, monkeypatch)
    room = _registered_room(tmp_path)
    turn = _owner_turn(system, tmp_path, monkeypatch, chat_id=room["chat_id"])  # first: the room's reader is no other work
    _project_work(system, tmp_path, "room")
    basis = system.workers.RUNNING[turn.id]["task"]["_project_admission"]
    _save_work(tmp_path, turn)
    assert running_project_ids(system.workers.RUNNING.values()) == {"room"}, "the lane turn holds no lease"
    assert prepare_restart_returns(tmp_path, system.workers.RUNNING, system.workers.PENDING,
                                   transaction_id="tx-lease") == {"bg", turn.id}
    _ack(tmp_path, "tx-lease")
    _boot_after_stop(system, monkeypatch, tmp_path)
    system.workers.assign_tasks()
    [(_wid, sent)] = _carried(system)
    assert (sent["id"], sent["_attempt"], sent["project_id"]) == (turn.id, 2, "room")
    assert sent["_project_admission"] == basis and "_project_scope_none" not in sent
    _on_lane(system, tmp_path, turn.id, 2)
    assert [row["id"] for row in _launched(system)] == ["bg"]
    assert running_project_ids(system.workers.RUNNING.values()) == {"room"}


def test_a_fenced_project_pool_never_holds_or_fails_a_returned_owner_turn(tmp_path, monkeypatch):
    """#2870, owner decision 1A: the readiness-exhaustion fence owns the project slots; my human's returned turn,
    still queued when it lands, stays queued and continues on its lane while the pool stays fenced."""
    from supervisor.restart_retention import prepare_restart_returns
    from tests.test_restart_saved_work import _ack

    system = _system(tmp_path, monkeypatch)
    system.saved("bg")
    turn = _owner_turn(system, tmp_path, monkeypatch)
    _save_work(tmp_path, turn)
    prepare_restart_returns(tmp_path, system.workers.RUNNING, system.workers.PENDING, transaction_id="tx-fence")
    _ack(tmp_path, "tx-fence")
    _boot_after_stop(system, monkeypatch, tmp_path)
    assert {row["id"] for row in system.workers.PENDING} == {"bg", turn.id}
    assert system.workers.kill_workers(disable_reason="worker_readiness_exhausted", terminal_status="failed",
                                       result_reason="Worker startup attempts were exhausted.",
                                       lanes=frozenset({"project"}))  # the fence's own door
    assert system.workers._WORKER_POOL_DISABLED_REASON == "worker_readiness_exhausted"
    assert [row["id"] for row in system.workers.PENDING] == [turn.id], "the fence settled only the pool's own row"
    system.workers.assign_tasks()
    [(_wid, sent)] = _carried(system)
    assert (sent["id"], sent["_attempt"]) == (turn.id, 2) and not _launched(system)
    _on_lane(system, tmp_path, turn.id, 2)
    assert system.workers._WORKER_POOL_DISABLED_REASON == "worker_readiness_exhausted"


def _door(system, monkeypatch, root, door):
    """One application stop and the next start: an acknowledged Restart returns the saved turn, a quit or a crash
    keeps it for my human's Resume (a reopen later too, when queue init also hands the id over as an orphaned direct
    root), an aborted Update returns it in this process."""
    from supervisor.restart_retention import prepare_restart_returns
    from tests.test_restart_saved_work import _ack

    if door == "acknowledged_restart":
        prepare_restart_returns(root, system.workers.RUNNING, system.workers.PENDING, transaction_id="tx-door")
        _ack(root, "tx-door")
        _boot_after_stop(system, monkeypatch, root)
    elif door == "quit":
        _boot_after_stop(system, monkeypatch, root)
    elif door == "quit_reopen_later":
        _stop(system)
        path = system.queue.QUEUE_SNAPSHOT_PATH
        snap = json.loads(path.read_text())
        snap["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(time.time() - 60))
        write_queue_document(path, snap)
        _boot(system, monkeypatch, root)
    else:
        assert system.control._quiesce_repo_writers("auto_merge") == []
        system.control._respawn_workers_after_failed_update()


@pytest.mark.parametrize("door", ["acknowledged_restart", "quit", "quit_reopen_later", "aborted_update"])
def test_a_room_turn_stopped_before_its_bound_frame_returns_on_the_basis_its_result_holds(tmp_path, monkeypatch, door):
    """#2725: the body recorded the room basis its carrier bound, the stop landed before the frame taught the
    RUNNING row: restore takes the basis from the same task's result of the same attempt (attempt 0 for a direct
    first launch, 1 on the row) under exactly one queue row, and the turn continues once on its lane."""
    system = _system(tmp_path, monkeypatch)
    room = _registered_room(tmp_path)
    turn = _owner_turn(system, tmp_path, monkeypatch, chat_id=room["chat_id"], bound=False)
    stored = load_task_result(tmp_path, turn.id)
    row = system.workers.RUNNING[turn.id]["task"]
    assert [frame["type"] for frame in turn.frames] == ["direct_turn_bound"] and "_project_admission" not in row
    assert (stored["task_attempt"], stored["execution_owner"]["kind"], row["project_id"]) == (0, "direct", "room")
    _save_work(tmp_path, turn)
    _door(system, monkeypatch, tmp_path, door)
    [restored] = [item for item in system.workers.PENDING if item["id"] == turn.id]
    assert restored["_project_admission"] == stored["_project_admission"], "the result's basis, not a lookup"
    system.workers.assign_tasks()
    sent = _carried(system)
    if door.startswith("quit"):
        assert not sent and budget_hold_fact(restored), "a quit keeps it for my human's Resume"
        assert system.queue.resume_budget_paused_task(turn.id)["ok"]
        system.workers.assign_tasks()
        sent = _carried(system)
    [(_wid, task)] = sent
    assert (task["id"], task["_attempt"], task["_project_admission"]) == (turn.id, 2, stored["_project_admission"])
    _on_lane(system, tmp_path, turn.id, 2)
    system.workers.assign_tasks()
    assert not _carried(system) and not _launched(system) and not system.workers.PENDING


def test_a_result_of_another_attempt_lends_its_basis_to_no_returned_row(tmp_path, monkeypatch):
    """#2725 control: a basis recorded by another attempt is no evidence for this row; it stays held, never run."""
    system = _system(tmp_path, monkeypatch)
    room = _registered_room(tmp_path)
    turn = _owner_turn(system, tmp_path, monkeypatch, chat_id=room["chat_id"], bound=False)
    stored = load_task_result(tmp_path, turn.id)
    write_task_result(tmp_path, turn.id, "running", task_attempt=3,  # constructed: a later attempt's start record
                      execution_owner={**stored["execution_owner"], "task_attempt": 3})
    assert load_task_result(tmp_path, turn.id)["_project_admission"] == stored["_project_admission"], "one to lend"
    _save_work(tmp_path, turn)
    _door(system, monkeypatch, tmp_path, "aborted_update")
    system.workers.assign_tasks()
    assert not _carried(system) and not _launched(system)
    [held] = system.workers.PENDING
    assert "_project_admission" not in held and held["_project_admission_restore_hold"]["reason"] in {
        "project_dispatch_unconfirmed", "project_routing_fence_lookup_failed"}, held["_project_admission_restore_hold"]


def test_a_returned_basis_fills_only_a_rooms_missing_basis_from_its_own_result(tmp_path, monkeypatch):
    """#2725 controls through a real return (an aborted Update) over produced turns whose results each name a basis
    the row must not take: a basis the row carries is never replaced, a Main row's attested absence takes none, a
    basis of another room lends none (that row stays held)."""
    system = _system(tmp_path, monkeypatch)
    main = _owner_turn(system, tmp_path, monkeypatch, text="main turn")
    room_a, room_b = (_registered_room(tmp_path, pid)["chat_id"] for pid in ("room-a", "room-b"))
    bound = _owner_turn(system, tmp_path, monkeypatch, chat_id=room_a, text="bound turn")
    unbound = _owner_turn(system, tmp_path, monkeypatch, chat_id=room_b, bound=False, text="unbound turn")
    own = json.loads(json.dumps(system.workers.RUNNING[bound.id]["task"]["_project_admission"]))
    room_a_basis = load_task_result(tmp_path, bound.id)["_project_admission"]
    assert room_a_basis["project_id"] == "room-a" and system.workers.RUNNING[main.id]["task"]["_project_scope_none"]
    # Constructed results: another basis for the bound row, a room's basis for Main and for the other room's turn.
    write_task_result(tmp_path, bound.id, "running", _project_admission={**room_a_basis, "constructed": "replaced"})
    for turn in (main, unbound):
        write_task_result(tmp_path, turn.id, "running", _project_admission=room_a_basis)
    for turn in (main, bound, unbound):
        _save_work(tmp_path, turn)
    _door(system, monkeypatch, tmp_path, "aborted_update")
    rows = {row["id"]: row for row in system.workers.PENDING}
    assert rows[bound.id]["_project_admission"] == own, "the row's own basis stays"
    assert rows[main.id]["_project_scope_none"] is True and "_project_admission" not in rows[main.id]
    assert "_project_admission" not in rows[unbound.id] and rows[unbound.id]["_project_admission_restore_hold"]


def _parked_mid_run(system, root, monkeypatch, turn, rail):
    """The running turn's own park at a round boundary, as its carrier's loop makes it, folded by the supervisor's
    park: my human's Pause (``owner``), the global budget rail (``budget``), its own cold sleep due in ten minutes
    (``sleep``, woken by ``_wake_due_sleep``) or the mind's ``continue_in_background`` (``detach``)."""
    from datetime import datetime, timezone

    from ouroboros import budget_pause, model_detach, model_sleep
    from ouroboros.tools.control_mind import _continue_in_background
    from supervisor import message_bus
    from supervisor.owner_pause_control import request_owner_pause
    from tests._budget_pause_exact_helpers import _fast_hold, _mock_pause_observation, _quiet_external
    from tests.conversation_carrier_support import park_lane_pause

    monkeypatch.setattr(model_detach, "pool_available", lambda: {"available": True})
    monkeypatch.setattr(message_bus, "get_bridge", lambda: SimpleNamespace(push_log=lambda *_a: None))
    monkeypatch.setattr(system.state, "budget_remaining", lambda *_a, **_k: 5.0)
    _fast_hold(monkeypatch, budget_pause)
    _quiet_external(monkeypatch, budget_pause)
    ctx, limit = _loop_ctx(root, turn.id, direct=True)
    ctx.task_attempt = turn.task.get("_attempt")
    ctx.task_metadata = {"_execution_owner": turn.task["_execution_owner"]}
    limit.messages.append({"role": "tool", "tool_call_id": "call_b", "content": "done b"})
    with pytest.raises(budget_pause.BudgetPauseRequested) as raised:
        if rail == "owner":
            assert request_owner_pause(turn.id, request_id="pause-mid-run")["ok"]
            budget_pause.enter_owner_pause(limit)
        elif rail == "budget":
            _mock_pause_observation(monkeypatch, budget_pause, [])
            budget_pause.request_pause(limit, rail=budget_pause.RAIL_GLOBAL_EXHAUSTED, scope="global",
                                       reason_text="money gone", root_task_id=turn.id)
        else:
            if rail == "detach":
                assert json.loads(_continue_in_background(ctx, "keep working"))["status"] == "detach_armed"
            else:
                due = datetime.fromtimestamp(time.time() + SLEEP_DUE_SEC, timezone.utc).isoformat()
                armed = model_sleep.request_sleep(ctx, {"wake_at": due, "senders": [], "tasks": [], "runs": [],
                                                        "any_mail": False}, model_sleep.MODE_COLD)
                assert armed["reason"] == "sleep_armed", armed
            budget_pause.enter_cold_sleep(limit)
    park_lane_pause(budget_pause.pause_event(turn.task, raised.value.pause), worker_id=turn.slot.wid)
    budget_pause.end_dispatch_fence(turn.id)
    [parked] = system.workers.PENDING
    return parked


SLEEP_DUE_SEC = 600.0  # far beyond any tick of the test: only ``_wake_due_sleep`` makes the sleep due


def _wake_due_sleep(system, monkeypatch):
    """The sleep's own time has come: one wake pass reads a clock past its ``wake_at`` (the real clock again after)."""
    from datetime import datetime, timedelta, timezone

    from supervisor.sleep_wake import wake_ready_sleepers

    later = datetime.now(timezone.utc) + timedelta(seconds=SLEEP_DUE_SEC + 1)
    with monkeypatch.context() as clock:
        clock.setattr("ouroboros.deadline_utils.utc_now", lambda: later)
        return wake_ready_sleepers(system.queue)


@pytest.mark.parametrize("rail", ["owner", "budget", "sleep", "detach"])
def test_a_paused_or_sleeping_owner_turn_returns_on_its_lane_and_a_detached_one_on_the_pool(tmp_path, monkeypatch,
                                                                                             rail):
    """Every permitted return of my human's turn keeps its lane: my human's Resume of a Pause or a budget pause, its
    own cold sleep waking. The turn the mind sent to the background continues on the pool, as it chose."""
    from supervisor.sleep_wake import wake_ready_sleepers

    system = _system(tmp_path, monkeypatch)
    turn = _owner_turn(system, tmp_path, monkeypatch)
    parked = _parked_mid_run(system, tmp_path, monkeypatch, turn, rail)
    assert parked["_execution_owner"]["lane"] == "conversation" and parked["_attempt"] == 1
    system.workers.assign_tasks()
    if rail == "detach":  # due at once: the same pass wakes it, onto a project slot
        assert not _carried(system) and [(row["id"], row["detached_by"]) for row in _launched(system)] == [
            (turn.id, "mind")]
        assert system.workers.RUNNING[turn.id]["task"]["_execution_owner"]["kind"] == "pooled"
        return
    assert not _carried(system) and not _launched(system), "nothing returns before its release"
    if rail in {"owner", "budget"}:
        assert system.queue.resume_budget_paused_task(turn.id)["ok"]
    else:
        assert not any(row.get("ok") for row in wake_ready_sleepers(system.queue)), "not before its time"
        assert any(row.get("ok") for row in _wake_due_sleep(system, monkeypatch))
    system.workers.assign_tasks()
    [(_wid, sent)] = _carried(system)
    assert (sent["id"], sent["_attempt"]) == (turn.id, 1) and sent["_budget_pause_resume"]
    _on_lane(system, tmp_path, turn.id, 1)
    assert not _launched(system)


def test_a_returned_wake_is_no_owner_turn_and_keeps_its_project_slot(tmp_path, monkeypatch):
    """A consciousness wake rides the conversation lane once admitted; returned by a stop it is queued work."""
    from supervisor.restart_retention import prepare_restart_returns
    from tests.test_restart_saved_work import _ack

    system = _system(tmp_path, monkeypatch)
    wake = _carried_turn(system, tmp_path, wake=True, patch=monkeypatch)
    _save_work(tmp_path, wake)
    assert prepare_restart_returns(tmp_path, system.workers.RUNNING, system.workers.PENDING,
                                   transaction_id="tx-wake") == {wake.id}
    _ack(tmp_path, "tx-wake")
    _boot_after_stop(system, monkeypatch, tmp_path)
    system.workers.assign_tasks()
    assert not _carried(system) and [row["id"] for row in _launched(system)] == [wake.id]
    assert system.workers.RUNNING[wake.id].get("lane") != "conversation"


@pytest.mark.parametrize("executor, returns", [("worker", True), ("inline_degraded", True), ("inline_call", False)])
def test_only_a_live_calls_executor_keeps_a_returned_owner_turn_off_the_lane(tmp_path, monkeypatch, executor, returns):
    """Which carrier record a return reads (constructed executor on a produced row): the degraded executor is the
    lane's disclosed fallback when no carrier could start, so its turn returns on a carrier; a live call
    (``inline_call``) runs in this process for its process-local port and never resumes on the lane, as
    ``conversation_lease.owner_conversation_key`` reads it."""
    from ouroboros.conversation_lease import owner_conversation_key
    from supervisor.conversation_lane import resumes_on_lane

    system = _system(tmp_path, monkeypatch)
    turn = _owner_turn(system, tmp_path, monkeypatch)
    row = dict(system.workers.RUNNING[turn.id]["task"])
    row["_execution_owner"] = {**row["_execution_owner"], "executor": executor}
    assert resumes_on_lane(row) is returns and bool(owner_conversation_key(row)) is returns


def test_a_held_return_keeps_its_lane_through_another_stop_before_its_dispatch(tmp_path, monkeypatch):
    """The saved queue carries the returned row's carrier facts: a quit held it, a second quit restores it from the
    PENDING snapshot, and my human's one Resume still sends it to the lane."""
    system = _system(tmp_path, monkeypatch)
    turn = _owner_turn(system, tmp_path, monkeypatch)
    _save_work(tmp_path, turn)
    _boot_after_stop(system, monkeypatch, tmp_path)
    _boot_after_stop(system, monkeypatch, tmp_path)
    saved = next(row["task"] for row in json.loads(system.queue.QUEUE_SNAPSHOT_PATH.read_text())["pending"]
                 if row["id"] == turn.id)
    assert saved["_execution_owner"]["lane"] == "conversation", "the published PENDING row names its carrier"
    [held] = system.workers.PENDING
    assert budget_hold_fact(held) and held["_attempt"] == 2
    assert system.queue.resume_budget_paused_task(turn.id)["ok"]
    system.workers.assign_tasks()
    [(_wid, sent)] = _carried(system)
    assert (sent["id"], sent["_attempt"]) == (turn.id, 2) and not _launched(system)
    _on_lane(system, tmp_path, turn.id, 2)


@pytest.mark.parametrize("record", ["own", "detached", "another_attempt"])
def test_a_waiting_turn_the_previous_release_saved_without_its_carrier_takes_its_own_record(
        tmp_path, monkeypatch, record):
    """The first Update onto #2870: the previous release's PENDING serialiser dropped ``_execution_owner`` and
    nothing else, so the Update's stop saves my human's held turn without it. The next boot takes the carrier from
    the same task's own result of the same attempt and the turn continues on its lane after Resume. A turn the mind
    detached, or a result of another attempt, lends nothing: the row returns as saved."""
    system = _system(tmp_path, monkeypatch)
    turn = _owner_turn(system, tmp_path, monkeypatch)
    if record == "detached":
        _detach_to_pool(system, tmp_path, monkeypatch, turn)
    else:
        _save_work(tmp_path, turn)
    _boot_after_stop(system, monkeypatch, tmp_path)  # a quit: the saved turn waits for my human's Resume
    stored = load_task_result(tmp_path, turn.id)
    if record == "another_attempt":
        write_task_result(tmp_path, turn.id, "running", task_attempt=3,  # constructed: a later attempt's start record
                          execution_owner={**stored["execution_owner"], "task_attempt": 3})
    _stop(system)  # the Update's stop
    path = system.queue.QUEUE_SNAPSHOT_PATH
    snap = json.loads(path.read_text())
    [saved] = [row["task"] for row in snap["pending"] if row["id"] == turn.id]
    assert saved.pop("_execution_owner")["kind"] == ("pooled" if record == "detached" else "direct")
    write_queue_document(path, snap)  # as the previous release's serialiser wrote it
    _boot(system, monkeypatch, tmp_path)
    [carrier] = [row.get("_execution_owner") for row in system.workers.PENDING if row["id"] == turn.id]
    assert carrier == (stored["execution_owner"] if record == "own" else None), "its own result's record, by value"
    resumed = system.queue.resume_budget_paused_task(turn.id)  # another attempt's record also refuses this Resume
    system.workers.assign_tasks()
    if record != "own":
        assert not _carried(system)
        return
    assert resumed["ok"]
    [(_wid, sent)] = _carried(system)
    assert (sent["id"], sent["_attempt"]) == (turn.id, 2) and not _launched(system)
    _on_lane(system, tmp_path, turn.id, 2)


def test_a_dispatched_return_stopped_before_its_first_save_carries_its_source_on_the_lane(tmp_path, monkeypatch):
    """The lane dispatch commits the attempt's mark before the put; a stop before that attempt saved anything
    carries the same frozen source one attempt on (``working_checkpoint._carry_recovery``), never a replay."""
    from supervisor.restart_retention import prepare_restart_returns
    from tests.test_restart_saved_work import _ack

    system = _system(tmp_path, monkeypatch)
    turn = _owner_turn(system, tmp_path, monkeypatch)
    _save_work(tmp_path, turn)
    prepare_restart_returns(tmp_path, system.workers.RUNNING, system.workers.PENDING, transaction_id="tx-1")
    _ack(tmp_path, "tx-1")
    _boot_after_stop(system, monkeypatch, tmp_path)
    system.workers.assign_tasks()
    [(_wid, first)] = _carried(system)
    _persist_running(tmp_path, first)  # the carrier's body started attempt 2; nothing saved yet
    _boot_after_stop(system, monkeypatch, tmp_path)
    [held] = system.workers.PENDING
    assert held["_attempt"] == 3 and held["_working_recovery"]["source_ref"] == first["_working_recovery"]["source_ref"]
    assert held["_working_recovery"]["continued_from_attempt"] == 2 and held["_execution_owner"]["lane"] == "conversation"
    assert system.queue.resume_budget_paused_task(turn.id)["ok"]
    system.workers.assign_tasks()
    [(_wid, second)] = _carried(system)
    assert (second["id"], second["_attempt"]) == (turn.id, 3) and not _launched(system)
    _on_lane(system, tmp_path, turn.id, 3)


@pytest.mark.parametrize("control", ["stop", "pause"])
def test_my_humans_stop_or_pause_after_a_return_still_holds_the_turn(tmp_path, monkeypatch, control):
    """The lane's carrier is no bypass of the common holds: a Stop settles the returned turn unrun, a Pause keeps
    it queued until Resume."""
    from ouroboros.cancel_intents import request_cancel
    from supervisor.owner_pause_control import request_owner_pause
    from supervisor.restart_retention import prepare_restart_returns
    from tests.test_restart_saved_work import _ack

    system = _system(tmp_path, monkeypatch)
    turn = _owner_turn(system, tmp_path, monkeypatch)
    _save_work(tmp_path, turn)
    prepare_restart_returns(tmp_path, system.workers.RUNNING, system.workers.PENDING, transaction_id="tx-ctl")
    _ack(tmp_path, "tx-ctl")
    _boot_after_stop(system, monkeypatch, tmp_path)
    if control == "stop":
        request_cancel(tmp_path, turn.id, reason="owner stop", source="owner", requested_by="owner")
    else:
        monkeypatch.setattr("supervisor.message_bus.get_bridge", lambda: SimpleNamespace(push_log=lambda *_a: None))
        assert request_owner_pause(turn.id, request_id="pause-after-return")["ok"]
    for _ in range(2):
        system.workers.assign_tasks()
    assert not _carried(system) and not _launched(system) and turn.id not in system.workers.RUNNING
    if control == "stop":
        assert not system.workers.PENDING and load_task_result(tmp_path, turn.id)["status"] == "cancelled"
        return
    assert [row["id"] for row in system.workers.PENDING] == [turn.id]
    assert system.queue.resume_budget_paused_task(turn.id)["ok"]
    system.workers.assign_tasks()
    [(_wid, sent)] = _carried(system)
    assert sent["id"] == turn.id
    _on_lane(system, tmp_path, turn.id, 2)


def test_a_returned_main_turn_moved_to_a_room_before_its_dispatch_continues_there_on_its_lane(tmp_path, monkeypatch):
    """Placement governs the queued row: the move readmits it into the room with the room's basis, and it is still
    my human's turn, carried by the lane."""
    from ouroboros.projects_registry import get_project
    from supervisor.restart_retention import prepare_restart_returns
    from tests.test_restart_saved_work import _ack

    system = _system(tmp_path, monkeypatch)
    turn = _owner_turn(system, tmp_path, monkeypatch)
    _save_work(tmp_path, turn)
    prepare_restart_returns(tmp_path, system.workers.RUNNING, system.workers.PENDING, transaction_id="tx-move")
    _ack(tmp_path, "tx-move")
    _boot_after_stop(system, monkeypatch, tmp_path)
    pid = _move_mid_run(system, tmp_path, monkeypatch, turn.id)
    system.workers.assign_tasks()
    [(_wid, sent)] = _carried(system)
    assert (sent["id"], sent["project_id"], sent["chat_id"]) == (turn.id, pid, get_project(tmp_path, pid)["chat_id"])
    assert sent["_project_admission"]["project_id"] == pid and "_project_scope_none" not in sent
    _on_lane(system, tmp_path, turn.id, 2)


@pytest.mark.parametrize("work", ["owner_turn", "detached"])
def test_a_saved_turn_moved_into_a_busy_room_continues_there_on_its_lane_and_pooled_work_waits(
        tmp_path, tmp_path_factory, monkeypatch, work):
    """A move does not make my human's returned turn a room writer: held by a quit, moved into a room whose
    background work runs (the room's one-writer lease) and Resumed, it is readmitted with the room's basis and folder
    and continues on its lane beside that work. The turn the mind sent to the pool waits for the lease, as pooled
    work does."""
    from ouroboros.project_lease import running_project_ids
    from ouroboros.projects_registry import create_project

    system = _system(tmp_path, monkeypatch)
    folder = tmp_path_factory.mktemp("busy-room-folder")  # a room folder outside the data drive
    create_project(tmp_path, "busy-room", working_dir=str(folder))
    turn = _owner_turn(system, tmp_path, monkeypatch)
    if work == "detached":
        _detach_to_pool(system, tmp_path, monkeypatch, turn)
    else:
        _save_work(tmp_path, turn)
    _boot_after_stop(system, monkeypatch, tmp_path)  # a quit: the saved turn waits for my human's Resume
    _project_work(system, tmp_path, "busy-room")
    _move_mid_run(system, tmp_path, monkeypatch, turn.id, pid="busy-room")
    assert system.queue.resume_budget_paused_task(turn.id)["ok"]
    system.workers.assign_tasks()
    assert running_project_ids(system.workers.RUNNING.values()) == {"busy-room"}, "the background work runs on"
    if work == "detached":
        assert not _carried(system) and not _launched(system)
        assert [row["id"] for row in system.workers.PENDING] == [turn.id]
        return
    [(_wid, sent)] = _carried(system)
    assert (sent["id"], sent["project_id"], sent["_project_admission"]["project_id"]) == (turn.id, "busy-room",
                                                                                         "busy-room")
    assert Path(sent["workspace_root"]).resolve() == folder.resolve() and not _launched(system)
    _on_lane(system, tmp_path, turn.id, 2)


class _Body:
    """The body runtime taking a cold sleeper's delegated run (``sleep_custody._runtime``): no engine here."""

    def ensure_runs(self, pairs, *, task_id=""):
        return True

    def identity(self):
        return {"runtime_id": "body", "active": True}


def _cold_writer(system, root, monkeypatch, project_id, task_id="bg"):
    """The room's background work in its own cold sleep while its delegated writer keeps writing: the body accepts
    the run (``sleep_custody.reconcile``), the park keeps the row queued with its receipt, the room's writer lane."""
    from ouroboros import budget_pause, delegate_custody, model_sleep, sleep_custody
    from tests._budget_pause_exact_helpers import _fast_hold
    from tests.conversation_carrier_support import park_lane_pause

    sent = _project_work(system, root, project_id, task_id)
    wid = system.workers.RUNNING[task_id]["worker_id"]
    assert delegate_custody.emit(root, delegate_custody.STARTED, {
        "run_id": "run-w", "invocation_id": "invoke-w", "task_id": task_id, "root_task_id": task_id,
        "route": "route", "source": "delegate_start", "access": "workspace_write"})
    body = _Body()
    monkeypatch.setattr(sleep_custody, "_runtime", lambda _root: body)
    _fast_hold(monkeypatch, budget_pause)
    monkeypatch.setattr(budget_pause, "_hold_control_reason",  # the host's pass, as the release waits
                        lambda _ctx: (sleep_custody.reconcile(system.queue, runtime=body), "")[1])
    ctx, limit = _loop_ctx(root, task_id)
    ctx.project_id, ctx.task_metadata = project_id, {"root_task_id": task_id, "project_id": project_id}
    limit.messages = [{"role": "user", "content": "background"}]
    assert model_sleep.request_sleep(ctx, model_sleep.selectors(ctx, runs=["run-w"]),
                                     model_sleep.MODE_COLD)["reason"] == "sleep_armed"
    with pytest.raises(budget_pause.BudgetPauseRequested) as raised:
        model_sleep.enter_cold_sleep(limit)
    park_lane_pause(budget_pause.pause_event(sent, raised.value.pause), worker_id=wid)
    budget_pause.end_dispatch_fence(task_id)
    assert task_id not in system.workers.RUNNING


@pytest.mark.parametrize("work", ["moved", "room_turn", "detached"])
def test_a_returned_turn_continues_beside_a_cold_room_writer_and_pooled_work_waits(
        tmp_path, tmp_path_factory, monkeypatch, work):
    """A cold writer's kept lane binds pooled work only (owner choice of 9 November 2026: the returned conversation is
    outside the project lease): my human's turn held by a quit, moved into that room (readmission: the room's basis
    and folder) or already in it, continues on its lane after Resume while the writer stays parked with its receipt.
    The turn the mind sent to the pool, moved there the same way, waits for the lane, as pooled work does."""
    from ouroboros.project_lease import lane_holders
    from ouroboros.projects_registry import create_project, get_project
    from ouroboros.sleep_custody import carrier_receipt

    system = _system(tmp_path, monkeypatch)
    folder = tmp_path_factory.mktemp("cold-room-folder")  # a room folder outside the data drive
    create_project(tmp_path, "cold-room", working_dir=str(folder))
    turn = _owner_turn(system, tmp_path, monkeypatch,
                       **({"chat_id": get_project(tmp_path, "cold-room")["chat_id"]} if work == "room_turn" else {}))
    if work == "detached":
        _detach_to_pool(system, tmp_path, monkeypatch, turn)
    else:
        _save_work(tmp_path, turn)
    _boot_after_stop(system, monkeypatch, tmp_path)  # a quit: the saved turn waits for my human's Resume
    _cold_writer(system, tmp_path, monkeypatch, "cold-room")
    if work != "room_turn":
        _move_mid_run(system, tmp_path, monkeypatch, turn.id, pid="cold-room")
    assert system.queue.resume_budget_paused_task(turn.id)["ok"]
    system.workers.assign_tasks()
    [writer] = [row for row in system.workers.PENDING if row["id"] == "bg"]
    assert carrier_receipt(writer)["retains_project_lane"]
    assert not [row for row in _launched(system) if "id" in row], "no work on a project slot (a probe is no launch)"
    assert lane_holders(system.workers.RUNNING.values(), system.workers.PENDING) == {"cold-room": {"bg"}}, (
        "the writer keeps its room's lane")
    if work == "detached":
        assert not _carried(system) and sorted(row["id"] for row in system.workers.PENDING) == sorted(["bg", turn.id])
        return
    [(_wid, sent)] = _carried(system)
    assert (sent["id"], sent["project_id"], sent["_project_admission"]["project_id"]) == (turn.id, "cold-room",
                                                                                         "cold-room")
    if work == "moved":  # readmission binds the room's folder; a room turn's carrier binds its basis at its first step
        assert Path(sent["workspace_root"]).resolve() == folder.resolve()
    _on_lane(system, tmp_path, turn.id, 2)


def test_a_returned_turn_on_its_carrier_reads_my_humans_next_words_under_a_fresh_intake(tmp_path, monkeypatch):
    """#2996 on a return: the carrier that takes the returned attempt opens that attempt's own intake (the dead
    attempt's fact admits nothing for it), and my human's next Main words reach the turn, not a second author."""
    import os

    from supervisor.restart_retention import prepare_restart_returns
    from tests._main_followup import _Bridge, _delivered, _say
    from tests.test_restart_saved_work import _ack

    system = _system(tmp_path, monkeypatch)
    turn = _owner_turn(system, tmp_path, monkeypatch)
    _save_work(tmp_path, turn)
    prepare_restart_returns(tmp_path, system.workers.RUNNING, system.workers.PENDING, transaction_id="tx-intake")
    _ack(tmp_path, "tx-intake")
    _boot_after_stop(system, monkeypatch, tmp_path)
    system.workers.assign_tasks()
    [(_wid, sent)] = _carried(system)
    intake = tmp_path / "memory" / "owner_mailbox" / f"{turn.id}.intake.json"
    assert json.loads(intake.read_text())["attempt"] == 1, "the dead attempt's fact, until a carrier takes the turn"
    carrier = worker_chat_lane.CarrierTurn(sent, tmp_path, stdqueue.Queue())
    assert carrier.intake == "owner:1" and json.loads(intake.read_text()) == {
        "accepting": True, "conversation_key": "owner:1", "pid": os.getpid(), "attempt": 2}
    bridge, before = _Bridge(), set(system.workers.RUNNING)
    ctx = SimpleNamespace(DRIVE_ROOT=tmp_path, RUNNING=system.workers.RUNNING, PENDING=system.workers.PENDING,
                          BUDGET_ROOT_FENCES={}, load_state=lambda: {"owner_id": 1, "owner_chat_id": 1},
                          update_state=lambda fn: fn({"owner_id": 1, "owner_chat_id": 1}),
                          consciousness=SimpleNamespace(), handle_chat_direct=system.workers.handle_chat_direct,
                          send_with_budget=lambda *a, **k: None)
    _say(bridge, ctx, "and the second thing", "cm-follow-up")
    (ack,) = _delivered(bridge, "cm-follow-up")
    assert ack["target"] == turn.id and set(system.workers.RUNNING) == before, "no second author"
    carrier.ended()
    assert not intake.exists(), "the attempt's end closes its own fact"


def test_returned_owner_turns_each_start_their_own_carrier_in_one_pass(tmp_path, monkeypatch):
    """The lane's capacity rule for returns: one carrier per returned turn still waiting, beside the spare, so two
    returned conversations continue together rather than one behind the other's carrier boot."""
    from supervisor.restart_retention import prepare_restart_returns
    from tests.test_restart_saved_work import _ack

    system = _system(tmp_path, monkeypatch)
    main = _owner_turn(system, tmp_path, monkeypatch, text="main turn")
    in_room = _owner_turn(system, tmp_path, monkeypatch, chat_id=_registered_room(tmp_path)["chat_id"], text="room turn")
    for turn in (main, in_room):
        _save_work(tmp_path, turn)
    prepare_restart_returns(tmp_path, system.workers.RUNNING, system.workers.PENDING, transaction_id="tx-two")
    _ack(tmp_path, "tx-two")
    _boot_after_stop(system, monkeypatch, tmp_path)
    assert [w.wid for w in system.workers.WORKERS.values() if w.lane == "conversation"] == [0], "the spare alone"
    system.workers.assign_tasks()  # the harness's tick: a pass, then another once the growth it started is ready
    sent = _carried(system)
    assert sorted(task["id"] for _wid, task in sent) == sorted([main.id, in_room.id])
    assert len({wid for wid, _task in sent}) == 2 and not _launched(system)
    for turn in (main, in_room):
        _on_lane(system, tmp_path, turn.id, 2)
    carriers = sorted(w.wid for w in system.workers.WORKERS.values() if w.lane == "conversation")
    system.workers.assign_tasks()  # the demand the returns counted is spent: the next tick grows nothing
    assert sorted(w.wid for w in system.workers.WORKERS.values() if w.lane == "conversation") == carriers
    idle = [w for w in system.workers.WORKERS.values() if w.lane == "conversation" and w.busy_task_id is None]
    assert len(idle) == 1, "and the reserved spare again"


def test_a_turn_given_back_before_its_first_save_restarts_from_its_input_on_its_lane(tmp_path, monkeypatch):
    """D5 under the lane rule: a returned turn cut before its first model call starts again from its own input,
    carried by the lane, its given-back attempt's mark committed before the put."""
    from supervisor.restart_retention import prepare_restart_returns
    from supervisor.task_admission import SEED_RETURN
    from tests.test_restart_saved_work import _ack

    system = _system(tmp_path, monkeypatch)
    turn = _owner_turn(system, tmp_path, monkeypatch)
    assert prepare_restart_returns(tmp_path, system.workers.RUNNING, system.workers.PENDING,
                                   transaction_id="tx-seed") == {turn.id}
    _ack(tmp_path, "tx-seed")
    _boot_after_stop(system, monkeypatch, tmp_path)
    [row] = system.workers.PENDING
    assert row[SEED_RETURN]["from_attempt"] == 1 and "_working_recovery" not in row
    system.workers.assign_tasks()
    [(_wid, sent)] = _carried(system)
    assert (sent["id"], sent["_attempt"], sent["text"]) == (turn.id, 2, "keep working") and not _launched(system)
    _on_lane(system, tmp_path, turn.id, 2)


def test_a_returned_owner_turn_still_meets_the_money_gate_before_its_lane(tmp_path, monkeypatch):
    """The lane's carrier buys no exception from money: at a reached limit the returned turn is held before dispatch
    with its saved work, the budget's return alone sends nothing, my human's Resume sends it to its lane."""
    from supervisor import worker_assignment
    from supervisor.restart_retention import prepare_restart_returns
    from tests.test_restart_saved_work import _ack

    system = _system(tmp_path, monkeypatch)
    turn = _owner_turn(system, tmp_path, monkeypatch)
    _save_work(tmp_path, turn)
    prepare_restart_returns(tmp_path, system.workers.RUNNING, system.workers.PENDING, transaction_id="tx-money")
    _ack(tmp_path, "tx-money")
    _boot_after_stop(system, monkeypatch, tmp_path)
    with monkeypatch.context() as spent:
        spent.setattr(system.state, "budget_remaining", lambda *_a, **_k: 0.0)
        spent.setattr(worker_assignment, "needs_paid_admission", lambda _task: True)
        system.workers.assign_tasks()
    [row] = system.workers.PENDING
    assert row["_budget_pause"]["continuation"] == "working" and not _carried(system) and not _launched(system)
    system.workers.assign_tasks()  # money is back (the harness's 5.0): no Resume, no dispatch
    assert not _carried(system) and not _launched(system)
    assert system.queue.resume_budget_paused_task(turn.id)["ok"]
    system.workers.assign_tasks()
    [(_wid, sent)] = _carried(system)
    assert (sent["id"], sent["_attempt"]) == (turn.id, 2) and not _launched(system)
    _on_lane(system, tmp_path, turn.id, 2)
