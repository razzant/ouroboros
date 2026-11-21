"""A moved Presence turn's Project lifecycle on the conversation-lane carriers (#2566).

Real producers and consumers: the owner's Pause door and the exact pause take a turn the owner moved to a
Project as text, and nothing else (``presence_carrier.moved_text_turn``); its park, through the supervisor's own
``budget_pause`` handler, keeps the turn's own built record in the queue, answers its waiting transport once that
custody is durable and leaves its carrier's slot held for the carrier's own settlement; a settlement frame acts
on its sender's own slot and row only, whatever executes the same id later; a Resume's grant waits while that
carrier holds; a holder that dies settles its waiter; the queue's custody is read before any start, loss stamp
or end, including before a restarted server restores its queue; saved work runs in the carrier that takes the
conversation before any restore or call. Its allowance: ``test_presence_moved_allowance``; custody no read confirms:
``test_presence_retained_custody``.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from tests._budget_pause_exact_helpers import _pause, _supervisor_ctx
from tests.test_presence_lane_rows import _admit_lane_row
from tests.test_presence_continuation import harness as harness
from tests.test_presence_project_activity import PROJECT, _live_turn, _move, root as root
from tests.test_presence_runner import _admission, _event

CEILING = 0.4  # what was left of a Place's consciousness allowance when the turn was admitted


def _seed_and_built(root):
    """The lane's seed of a Host turn and the turn's own built task, from the real producers."""
    from ouroboros.presence_carrier import presence_seed_task
    from ouroboros.presence_runner import _build_task
    from tests.test_conversation_lane_presence import _payload

    seed = presence_seed_task(admission=_admission(), event=_event(), skill_name="telegram-bot",
                              event_payload=_payload(_event()), staged_files=[], gate_wait=True)
    built = _build_task(_admission(), _event(), drive_root=root, staged_files=(), physical_task_id=seed["id"])
    return seed, built


def _slot(workers, wid, pid, task_id, lane="conversation"):
    slot = workers.Worker(wid, SimpleNamespace(pid=pid, is_alive=lambda: True), [], lane=lane)
    slot.busy_task_id = task_id
    workers.WORKERS[wid] = slot
    return slot


@pytest.fixture
def executions():
    """The Host's waiter of a turn (``PresenceTurnExecutions``); every one still live retires at teardown, so no
    later test's lookup of the same turn id finds it."""
    from ouroboros.presence_executions import PresenceTurnExecutions

    made = []

    def start(task_id):
        registry, released = PresenceTurnExecutions(), []
        execution, _started = registry.start_or_dispatch(
            task_id, reserve=lambda: True, release=lambda: released.append(1), dispatch=lambda _execution: None)
        made.append((registry, execution))
        return registry, execution, released

    yield start
    for registry, execution in made:
        registry.settle_from_frame(execution, error=RuntimeError("test teardown"))


def test_the_pause_door_and_the_exact_pause_take_a_moved_text_turn_only(root):
    from ouroboros.budget_pause import RAIL_GLOBAL_EXHAUSTED, RAIL_MODEL_SLEEP, RAIL_OWNER_PAUSE
    from ouroboros.owner_pause import fence_closed, read_fence
    from ouroboros.presence_carrier import moved_text_turn, saves_exact_pause
    from ouroboros.task_results import load_task_result
    from supervisor.owner_pause_control import request_owner_pause

    seen = {}

    def scenario(_author, task):
        tid = task["id"]
        _admit_lane_row(task)
        seen["unmoved"] = request_owner_pause(tid, request_id="pause-unmoved")
        meta = (load_task_result(root, tid) or {}).get("metadata")
        ctx = SimpleNamespace(task_id=tid, task_metadata=meta, drive_root=root, budget_drive_root=root)
        seen["unmoved_rail"] = saves_exact_pause(ctx, RAIL_OWNER_PAUSE)
        _move(root, tid, PROJECT)
        seen["rails"] = {rail: saves_exact_pause(ctx, rail)
                         for rail in (RAIL_OWNER_PAUSE, RAIL_GLOBAL_EXHAUSTED, RAIL_MODEL_SLEEP)}
        seen["call"] = moved_text_turn(root, tid, {**meta, "execution_transport": {"call_id": "call-1"}})
        seen["moved"] = request_owner_pause(tid, request_id="pause-moved")
        seen["fence"] = fence_closed(read_fence(root, tid))

    _live_turn(root, scenario)
    # Unmoved: the door and the rail keep today's refusal.
    assert (seen["unmoved"]["ok"], seen["unmoved"]["error"]) == (False, "presence_pause_unsupported")
    assert seen["unmoved_rail"] is False
    # Moved as text: the owner's Pause and the money rails save the exact pause; the model's own sleep and a live
    # call never do.
    assert seen["rails"] == {RAIL_OWNER_PAUSE: True, RAIL_GLOBAL_EXHAUSTED: True, RAIL_MODEL_SLEEP: False}
    assert seen["call"] is False
    assert seen["moved"]["ok"] is True and seen["fence"] is True


def test_a_park_keeps_the_turns_own_record_answers_its_waiter_and_leaves_its_carrier_slot(root, monkeypatch,
                                                                                     executions):
    from ouroboros import budget_pause as bp
    from ouroboros.presence_events import PresenceTurnError
    from ouroboros.task_store import patch_document
    from supervisor import queue as q, workers
    from supervisor.events import _handle_budget_pause

    seed, built = _seed_and_built(root)
    tid = built["id"]
    built.update(workspace_root=str(root / "place-folder"), workspace_mode="external")
    _admit_lane_row(built)  # the lane's RUNNING row, dispatched to its spare carrier
    carrier = next(slot for slot in workers.WORKERS.values() if slot.busy_task_id == tid)
    _ctx, _limit, pause = _pause(root, monkeypatch, task_id=tid)  # the author's real exact pause
    bp.end_dispatch_fence(tid)
    patch_document(root, tid, lambda current: {**current, "metadata": built["metadata"], "presence_continuation": {
        "initial": {"status": "continuing", "outcome": "deferred", "text": "", "output_ref": "", "work_ref": ""},
        "continuation_version": 1, "event_identity": built["metadata"]["presence_event_identity"]}})
    registry, execution, released = executions(tid)
    event = {key: value for key, value in bp.pause_event(built, pause).items() if key != "worker_id"}  # the carrier's
    _handle_budget_pause(event, _supervisor_ctx(root, workers, q, [], []))
    parked = next(task for task in workers.PENDING if task["id"] == tid)
    # The queue keeps the turn's own built record: its contract and ceiling, its folder as the explicit resource,
    # its source room; never the lane seed or its admission facts. Custody moved; the carrier keeps its slot.
    assert parked["task_contract"] == built["task_contract"] and parked["_budget_pause"]["exact_continuation"]
    assert "_presence_run" not in parked and "_execution_owner" not in parked
    assert parked["chat_id"] == built["chat_id"] and parked["_presence_turn"] is True
    assert parked["metadata"]["resource_intent"] == {"kind": "explicit_resource"}
    assert parked["metadata"]["presence_request_waits"] is False
    assert tid not in workers.RUNNING and carrier.busy_task_id == tid
    # Its waiting transport heard the queue's custody as soon as the park was on disk; nothing settled or returned.
    with pytest.raises(PresenceTurnError) as v0:
        execution.result.result(timeout=1)
    assert (v0.value.code, v0.value.disposition, getattr(v0.value, "custody", "")) == (
        "presence_turn_parked", "retained", "queued")
    assert execution.initial.result(timeout=1).status == "continuing"
    assert released == [] and tid in registry.live()
    assert seed["id"] == tid


def test_a_settlement_frame_acts_on_its_senders_own_slot_and_row_only(root, executions):
    """The interleavings a late frame meets: its holder died and the respawned slot (same wid) runs the same id's
    successor at the same attempt (an exact Resume keeps it), or that successor parked in turn; only the
    successor's own frames act."""
    from ouroboros.presence_carrier import handle_presence_turn_settled, settle_ended_turn
    from supervisor import workers

    _seed, built = _seed_and_built(root)
    tid = built["id"]
    registry, _execution_, released = executions(tid)
    ctx = SimpleNamespace(RUNNING=workers.RUNNING, WORKERS=workers.WORKERS, DRIVE_ROOT=root)

    def frame(pid, parked="pause-1"):
        return {"type": "presence_turn_settled", "turn_id": tid, "task_id": tid, "physical_id": tid, "result": None,
                "error": {"kind": "presence", "code": "presence_turn_parked", "disposition": "retained",
                          "turn_ref": tid, "custody": "queued"},
                "worker_id": 90, "pid": pid, "attempt": 1, "parked": parked}

    def bound():
        workers.RUNNING[tid] = {"task": {**built, "_attempt": 1}, "worker_id": 90, "attempt": 1}

    # The holder (pid 4242) died in its hold: its reaper settled the waiter and respawned slot 90 (pid 5151),
    # which now runs the Resume's successor of the same id at the same attempt.
    assert settle_ended_turn(tid) is True and released == [1] and tid not in registry.live()
    successor = _slot(workers, 90, 5151, tid, lane="project")
    bound()
    handle_presence_turn_settled(frame(4242), ctx)  # the dead holder's parked frame, read only now
    assert tid in workers.RUNNING and successor.busy_task_id == tid
    handle_presence_turn_settled(frame(4242, parked=""), ctx)  # even an unparked one names another process
    assert tid in workers.RUNNING and successor.busy_task_id == tid
    # The successor parked in turn (its row's custody moved to the queue): the old frame still frees nothing.
    workers.RUNNING.pop(tid)
    handle_presence_turn_settled(frame(4242), ctx)
    assert successor.busy_task_id == tid
    # The control: the successor's own parked frame frees its own slot.
    handle_presence_turn_settled(frame(5151), ctx)
    assert successor.busy_task_id is None
    # Dispatched again, its own unparked end pops the row it is bound to and frees its slot.
    successor.busy_task_id = tid
    bound()
    handle_presence_turn_settled(frame(5151, parked=""), ctx)
    assert tid not in workers.RUNNING and successor.busy_task_id is None


def test_a_resumes_grant_waits_while_the_parking_carrier_holds_the_same_id(root, monkeypatch):
    from supervisor import queue as q, worker_assignment as wa, workers

    _seed, built = _seed_and_built(root)
    tid = built["id"]
    candidate = {**built, "_attempt": 1, "_budget_pause_resume": {"grant_id": "grant-1", "pause_id": "pause-1"}}
    for name, value in (("_foreign_roots_refused", lambda _c: False), ("budget_hold_fact", lambda _c: None),
                        ("continuation_refusal", lambda _c: {})):
        monkeypatch.setattr(wa, name, value)
    monkeypatch.setattr("supervisor.consciousness_carriers.autonomous_work_enabled", lambda *_a: True)
    monkeypatch.setattr("supervisor.events_budget.budget_resume_dispatch_allowed", lambda *_a: True)
    monkeypatch.setattr("supervisor.task_lifecycle.root_cancellation_fenced", lambda *_a: False)
    monkeypatch.setattr("ouroboros.project_placement.refresh_pending_placement", lambda *_a, **_k: True)
    monkeypatch.setattr("ouroboros.project_lease.candidate_is_leasable", lambda *_a, **_k: True)
    monkeypatch.setattr(workers, "repo_writer_task_allowed", lambda _t: True)
    monkeypatch.setattr(workers, "_invalid_depth_deferred", lambda *_a: False)

    def ineligible(row=candidate, **kwargs):
        return wa._candidate_ineligible(q, row, 10.0, {}, **kwargs)

    holder = _slot(workers, 90, 4242, tid)
    assert ineligible() is True  # the carrier that parked it still holds its conversation and slot
    claimed = {**candidate, wa._LAUNCH_CLAIM: {"token": "claim-1"}}
    assert ineligible(claimed, own_claim="claim-1") is False  # a launch's revalidation of its own claimed slot
    holder.busy_task_id = None
    assert ineligible() is False  # its settlement freed the slot: the accepted grant dispatches now
    holder.busy_task_id = "ordinary-work"
    ordinary = {key: value for key, value in candidate.items() if key != "_presence_turn"}
    assert ineligible({**ordinary, "id": "ordinary-work"}) is False  # only a Presence turn waits on a busy slot


def test_a_holder_that_dies_in_its_hold_settles_its_waiter_and_its_slot_respawns(root, monkeypatch, executions):
    from supervisor import task_reaper, workers
    from supervisor.worker_health import recover_confirmed_dead_worker

    _seed, built = _seed_and_built(root)
    tid = built["id"]
    registry, _execution_, released = executions(tid)
    respawned = []
    dead = workers.Worker(90, SimpleNamespace(pid=4242, is_alive=lambda: False, exitcode=-9), [], lane="conversation")
    dead.busy_task_id, dead.reaping = tid, True
    workers.WORKERS[90] = dead
    monkeypatch.setattr("supervisor.worker_health._dead_job_is_current", lambda _job: True)
    monkeypatch.setattr(task_reaper, "_respawn_after_reap", lambda *a, **_k: respawned.append(a))
    monkeypatch.setattr(workers, "_reconcile_confirmed_dead_review_owner", lambda *_a: None)
    recover_confirmed_dead_worker({"kind": "confirmed_dead_worker", "worker_id": 90, "worker": dead, "task_id": tid,
                                   "meta": None, "task": {}, "attempt": 1, "exitcode": -9, "drive_root": str(root)})
    assert released == [1] and tid not in registry.live() and respawned


def _boot_window_row(root, tid, built, *, status="running", pause=None, move=True):
    """The row a moved turn's earlier attempt left, its controller gone (``move=False``: a turn never moved)."""
    from ouroboros.gateway.projects import _owner_task_origin
    from ouroboros.projects_registry import bind_task_to_project
    from ouroboros.task_results import write_task_result

    owner = {"kind": "presence", "data_root": str(root.resolve()), "task_id": tid, "task_attempt": 1,
             "controller": {"pid": 999999, "process_birth": "gone", "host": "this"}}
    write_task_result(root, tid, status, metadata=built["metadata"], chat_id=built["chat_id"], _is_direct_chat=True,
                      source="presence", execution_owner=owner, result="Task is running.",
                      **({"budget_pause": pause} if pause else {}))
    if move:
        from ouroboros.projects_registry import project_binding_for_task

        if not project_binding_for_task(root, tid):
            bind_task_to_project(root, tid, PROJECT, origin=_owner_task_origin(root, tid))


def _saved_queue(root, *, running=(), pending=()):
    (root / "state").mkdir(exist_ok=True)
    (root / "state" / "queue_snapshot.json").write_text(json.dumps(
        {"generation": "previous", "pending": list(pending), "running": list(running)}), encoding="utf-8")


def test_custody_is_read_before_any_start_or_loss_and_unknown_is_never_confirmed(root, monkeypatch):
    """The window between a restarted server's Host and its queue restore: the saved document names the turn and
    nothing ruled on it yet. The poll stays pending and a retry is retained as unknown custody, with no author and
    no loss stamp; once the restored queue keeps it, the answer is the queue's; a canonical terminal outranks every
    pause fact."""
    from ouroboros.conversation_lease import PresenceTurnLease
    from ouroboros.gateway.host_service_places import _presence_exception, _queue_custody_refusal
    from ouroboros.presence_events import PresenceTurnError
    from ouroboros.presence_recovery import (
        CUSTODY_QUEUED, CUSTODY_TERMINAL, CUSTODY_UNKNOWN, attempt_ended, queue_custody, retained_work_view,
    )
    from ouroboros.presence_runner import run_presence_turn
    from ouroboros.task_results import load_task_result
    from supervisor import queue_snapshot, workers

    seed, built = _seed_and_built(root)
    tid, identity = built["id"], built["metadata"]["presence_event_identity"]
    monkeypatch.setattr("ouroboros.review_operation.controller_state", lambda _controller: "dead")
    # A turn never moved keeps its own rules even in the boot window: no custody, its dead controller ends it.
    unmoved = tid + "-unmoved"
    _boot_window_row(root, unmoved, {**built, "id": unmoved}, move=False)
    _saved_queue(root, running=[{"id": unmoved, "task": {**seed, "id": unmoved}}])
    monkeypatch.setattr(queue_snapshot, "saved_queue_unowned", lambda: True)
    assert queue_custody(root, unmoved) == "" and attempt_ended(load_task_result(root, unmoved), root) is True
    _boot_window_row(root, tid, built)
    _saved_queue(root, running=[{"id": tid, "task": dict(seed)}])
    stored = load_task_result(root, tid)
    # The negative control: with the queue restored and not holding it, the dead controller ends the attempt.
    monkeypatch.setattr(queue_snapshot, "saved_queue_unowned", lambda: False)
    assert queue_custody(root, tid, stored) == "" and attempt_ended(stored, root) is True
    # The boot window: this server has not taken the saved queue yet.
    monkeypatch.setattr(queue_snapshot, "saved_queue_unowned", lambda: True)
    assert queue_custody(root, tid, stored) == CUSTODY_UNKNOWN and attempt_ended(stored, root) is False
    status, body = retained_work_view(stored, tid, root)
    assert (status, body["status"]) == (202, "pending")
    with pytest.raises(PresenceTurnError) as unknown:
        _queue_custody_refusal(root, tid, identity)
    response = _presence_exception(unknown.value, 409)
    answer = json.loads(response.body)
    assert response.status_code == 409 and (answer["code"], answer["disposition"], answer["custody"]) == (
        "presence_turn_custody_unknown", "retained", "unknown")
    bound = []
    monkeypatch.setattr(PresenceTurnLease, "bind_author", lambda self, author: bound.append(author))
    with pytest.raises(PresenceTurnError) as direct:
        run_presence_turn(admission=_admission(), event=_event(), repo_dir=root, drive_root=root,
                          agent_factory=lambda **_kw: pytest.fail("no author is built for kept work"))
    assert direct.value.code == "presence_turn_custody_unknown" and bound == []
    assert not load_task_result(root, tid).get("presence_recovery_recorded")  # no loss stamp, no room note
    # Restored: the live queue keeps its returned successor PENDING; the answer is the queue's, confirmed.
    monkeypatch.setattr(queue_snapshot, "saved_queue_unowned", lambda: False)
    workers.PENDING.append({**built, "_attempt": 2})
    assert queue_custody(root, tid, stored) == CUSTODY_QUEUED
    with pytest.raises(PresenceTurnError) as queued:
        _queue_custody_refusal(root, tid, identity)
    assert (queued.value.code, queued.value.custody) == ("presence_turn_parked", "queued")
    workers.PENDING.clear()
    # Saved work keeps custody by its own durable facts (a live exact pause, the queue's park status).
    _boot_window_row(root, tid, built, status="scheduled",
                     pause={"state": "paused", "pause_id": "p", "source_ref": "source", "task_attempt": 1})
    assert queue_custody(root, tid, load_task_result(root, tid)) == CUSTODY_QUEUED
    # A canonical terminal outranks every pause fact.
    _boot_window_row(root, tid, built, status="completed",
                     pause={"state": "paused", "pause_id": "p", "source_ref": "source", "task_attempt": 1})
    assert queue_custody(root, tid, load_task_result(root, tid)) == CUSTODY_TERMINAL


def test_a_stop_reaches_saved_work_the_queue_keeps_and_unknown_custody_is_retried(root, monkeypatch):
    from ouroboros.gateway.tasks import _record_stop
    from supervisor import queue_snapshot, workers

    seed, built = _seed_and_built(root)
    tid = built["id"]
    _boot_window_row(root, tid, built)  # the row its old attempt left, its controller gone
    monkeypatch.setattr("ouroboros.review_operation.controller_state", lambda _controller: "dead")
    monkeypatch.setattr(queue_snapshot, "saved_queue_unowned", lambda: False)
    assert _record_stop(tid, cascade=True, stop_action_id="stop-ended") is None  # no custody: it ended here
    workers.PENDING.append({**built, "_attempt": 2})  # a restart returned it: the queue keeps it
    assert _record_stop(tid, cascade=True, stop_action_id="stop-queued") is not None  # recorded, never a false 404
    workers.PENDING.clear()
    other = tid + "-b"
    _boot_window_row(root, other, {**built, "id": other})
    _saved_queue(root, pending=[{"id": other, "task": {**seed, "id": other}}])
    monkeypatch.setattr(queue_snapshot, "saved_queue_unowned", lambda: True)  # named, not yet restored
    with pytest.raises(OSError):  # the door's 503: retry, never "not found"
        _record_stop(other, cascade=True, stop_action_id="stop-unknown")


def test_saved_work_runs_in_its_carrier_after_taking_the_conversation_and_a_crash_retry_never_does(root, monkeypatch):
    from ouroboros import agent as agent_module
    from ouroboros import presence_carrier
    from ouroboros.presence_runner import _configured_gate, _read_previous_turn
    from ouroboros.task_results import load_task_result, write_task_result

    seed, built = _seed_and_built(root)
    tid, key = built["id"], _event().conversation_key
    write_task_result(root, tid, "scheduled", metadata=built["metadata"], chat_id=built["chat_id"],
                      _is_direct_chat=True, source="presence", task_contract=built["task_contract"],
                      result="Task paused.")
    _move(root, tid, PROJECT)
    seen = {"built": 0}

    class Author:
        def handle_task(self, task):
            seen["task"] = dict(task)
            ctx = SimpleNamespace(task_id=task["id"], task_metadata=task["metadata"],
                                  model_wait_context=SimpleNamespace(control_reason=lambda: seen.get("stop")))
            state = presence_carrier.reenter(ctx)  # the loop's first step, before any restore or call
            seen["held"] = bool(state and state.lease.held())
            seen["lost"] = getattr(ctx, "_presence_conversation_lost", "")
            if seen["lost"]:
                return [{"type": "task_done", "task_id": task["id"], "status": "cancelled"}]
            write_task_result(root, task["id"], "completed", metadata=task["metadata"], result="Done.",
                              terminal_origin="model_final")
            return [{"type": "presence_result", "outcome": "message", "text": "Done.", "work_ref": ""},
                    {"type": "task_done", "task_id": task["id"], "status": "completed"}]

    def make_agent(**_kwargs):
        seen["built"] += 1
        return Author()

    monkeypatch.setattr(agent_module, "make_agent", make_agent)
    frames = []
    out_q = SimpleNamespace(put=frames.append)
    # A crash retry is no owner act: retained unknown, no agent built, one settlement frame naming its sender. Its ended
    # terminal is its own row's (``test_a_refused_crash_continuation_...``), so it runs on an id of its own here.
    crashed = tid + "-crashed"
    write_task_result(root, crashed, "running", metadata=built["metadata"], chat_id=built["chat_id"],
                      _is_direct_chat=True, source="presence", task_contract=built["task_contract"],
                      result="Task is running.")
    _move(root, crashed, PROJECT)
    crash = {**built, "id": crashed, "_attempt": 2, "_working_recovery": {"cause": "worker_crash"}}
    refused = presence_carrier.carry_presence_turn(dict(crash), out_q, root, root, worker_id=91)
    assert [event["type"] for event in refused] == ["task_done"]
    assert seen["built"] == 0 and frames[-1]["error"]["code"] == "presence_attempt_outcome_unknown"
    assert frames[-1]["worker_id"] == 91 and frames[-1]["parked"] == ""
    # A Stop while it waits for its conversation: the turn ends on the zero-call rail, holding nothing.
    other = _configured_gate(root).acquire(key)
    seen["stop"] = "cancelled"
    resumed = {**built, "_attempt": 1, "_budget_pause_resume": {"grant_id": "g", "pause_id": "p"},
               "_execution_owner": {"kind": "pooled", "lane": "project"}}
    events = presence_carrier.carry_presence_turn(dict(resumed), out_q, root, root, worker_id=91)
    assert seen["lost"] == "cancelled" and seen["held"] is False and [e["status"] for e in events] == ["cancelled"]
    other.release()
    # The owner's Resume: the conversation is taken first, the saved record runs, its end writes the pointer under
    # that lock, and its own task_done goes back to the worker; no presence_result frame leaves the worker.
    seen.pop("stop")
    events = presence_carrier.carry_presence_turn(dict(resumed), out_q, root, root, worker_id=91)
    assert seen["held"] is True and seen["lost"] == ""
    assert seen["task"]["task_contract"] == built["task_contract"] and "_presence_run" not in seen["task"]
    assert [event["type"] for event in events] == ["task_done"]
    assert (_read_previous_turn(root, key) or {}).get("task_id") == tid
    assert load_task_result(root, tid)["status"] == "completed"
    assert seed["id"] == tid


def test_a_refused_crash_continuation_publishes_its_ended_terminal_before_its_attempt_is_released(root, monkeypatch):
    """The review's reproduction through assignment, refusal and settlement. Assignment stamped a pooled owner, and the
    refused attempt leaves RUNNING and its slot: its ended, unknown-outcome terminal is durable before that release, so
    nothing still claims queue custody, the Host poll answers the terminal and a resend replays nothing."""
    import os

    from ouroboros.gateway.host_service_places import _queue_custody_refusal
    from ouroboros.presence_carrier import carry_presence_turn, handle_presence_turn_settled
    from ouroboros.presence_events import PresenceTurnError
    from ouroboros.presence_recovery import CUSTODY_TERMINAL, attempt_ended, queue_custody, retained_work_view
    from ouroboros.presence_runner import presence_turn_replay
    from ouroboros.task_results import load_task_result
    from supervisor import events, queue as q, queue_snapshot, workers
    from supervisor.worker_assignment import _mirror_assigned_running_status

    monkeypatch.setattr(queue_snapshot, "saved_queue_unowned", lambda: False)
    monkeypatch.setattr("ouroboros.agent.make_agent", lambda **_k: pytest.fail("a refused continuation builds no agent"))
    _seed, built = _seed_and_built(root)
    tid, identity = built["id"], built["metadata"]["presence_event_identity"]
    _boot_window_row(root, tid, built)
    task = {**built, "drive_root": str(root), "_attempt": 2, "_working_recovery": {"cause": "worker_crash"},
            "metadata": {**built["metadata"], "presence_request_waits": False}}
    _mirror_assigned_running_status(task)
    assert load_task_result(root, tid)["execution_owner"]["kind"] == "pooled"
    slot = _slot(workers, 91, os.getpid(), tid, lane="project")
    workers.RUNNING[tid] = {"task": task, "worker_id": 91, "attempt": 2}
    frames, at_release = [], {}

    def put(frame):
        at_release.update(load_task_result(root, tid))  # what any reader finds the moment its attempt is released
        frames.append(frame)

    events_back = carry_presence_turn(task, SimpleNamespace(put=put), root, root, worker_id=91)
    assert at_release["status"] == "failed" and at_release["status_reconciled_from"] == "running"
    assert frames[-1]["error"]["code"] == "presence_attempt_outcome_unknown" and frames[-1]["parked"] == ""
    handle_presence_turn_settled(frames[-1], SimpleNamespace(RUNNING=workers.RUNNING, WORKERS=workers.WORKERS, DRIVE_ROOT=root))
    assert tid not in workers.RUNNING and slot.busy_task_id is None
    # The terminal goes through the ordinary dispatch: its card, fences and projections end like any task's.
    pushed = []
    [done] = [event for event in events_back if event["type"] == "task_done"]
    assert done["status"] == "failed" and done["task_id"] == tid
    events._handle_task_done({**done, "worker_id": 91}, _supervisor_ctx(root, workers, q, [], pushed))
    assert [row["type"] for row in pushed] == ["task_done"] and pushed[0]["status"] == "failed"
    stored = load_task_result(root, tid)
    assert stored["reason_code"] == "presence_attempt_lost" and stored.get("presence_recovery_recorded")
    assert stored["execution_owner"]["kind"] == "pooled" and queue_custody(root, tid, stored) == CUSTODY_TERMINAL
    assert attempt_ended(stored, root) is True
    status, body = retained_work_view(stored, tid, root)
    assert (status, body["status"], body["code"]) == (200, "failed", "presence_attempt_outcome_unknown")
    assert (body["outcome"], body["text"]) == ("silent", "")
    # Nothing is replayed: a resend hears the same retained answer, and no queue custody refuses it first.
    _queue_custody_refusal(root, tid, identity)
    with pytest.raises(PresenceTurnError) as resent:
        presence_turn_replay(root, tid, _event().conversation_key, identity, 0)
    assert (resent.value.code, resent.value.disposition) == ("presence_attempt_outcome_unknown", "retained")


def test_a_refused_attempt_whose_terminal_cannot_be_written_is_not_released(root, monkeypatch):
    """No terminal, no release: the failed write raises into the shared crash policy, which still holds the row and slot."""
    from ouroboros import task_results
    from ouroboros.presence_carrier import carry_presence_turn

    monkeypatch.setattr("ouroboros.agent.make_agent", lambda **_k: pytest.fail("a refused continuation builds no agent"))
    _seed, built = _seed_and_built(root)
    _boot_window_row(root, built["id"], built)

    def unwritable(*_a, **_k):
        raise OSError("store unavailable")

    monkeypatch.setattr(task_results, "write_task_result", unwritable)
    frames = []
    task = {**built, "_attempt": 2, "_working_recovery": {"cause": "worker_crash"}}
    with pytest.raises(OSError):
        carry_presence_turn(task, SimpleNamespace(put=frames.append), root, root, worker_id=91)
    assert frames == []


@pytest.mark.parametrize("own_terminal", [False, True], ids=["store_still_busy", "exception_terminal"])
def test_a_saved_turn_whose_row_cannot_be_read_ends_through_its_worker_failure_and_recovery(
        root, monkeypatch, own_terminal):
    """Review R2 F1 (``.review-evidence/2566-review2-astra/test_lifecycle_read_failure.py``), through
    ``run_worker_task``: a strict read of the saved turn's row that fails after assignment stamped its pooled owner
    settles nothing. It raises into the worker's exception rail, which writes the attempt's own terminal when the store
    answers again (``exception_terminal``); the RUNNING row and the slot stay the attempt's, the worker exits on the
    exception (``test_worker_main_exits_after_post_bootstrap_task_exception``) and the dead worker's recovery ends the
    attempt, so the Host's poll answers a terminal, never a 202 nothing will end."""
    import os
    import queue as stdqueue

    from ouroboros import presence_runner
    from ouroboros.presence_events import PresenceTurnError
    from ouroboros.presence_recovery import CUSTODY_TERMINAL, queue_custody, retained_work_view
    from ouroboros.store import StoreBusy
    from ouroboros.task_results import load_task_result
    from supervisor import queue_snapshot, task_reaper, workers
    from supervisor.worker_assignment import _mirror_assigned_running_status
    from supervisor.worker_health import recover_confirmed_dead_worker
    from supervisor.worker_process import run_worker_task

    monkeypatch.setattr(queue_snapshot, "saved_queue_unowned", lambda: False)
    monkeypatch.setattr("ouroboros.agent.make_agent", lambda **_k: pytest.fail("a failed read builds no agent"))
    _seed, built = _seed_and_built(root)
    tid = built["id"]
    _boot_window_row(root, tid, built)
    task = {**built, "drive_root": str(root), "_attempt": 2, "_working_recovery": {"cause": "owner_restart"},
            "metadata": {**built["metadata"], "presence_request_waits": False}}
    _mirror_assigned_running_status(task)
    assert load_task_result(root, tid)["execution_owner"]["kind"] == "pooled"
    slot = _slot(workers, 91, os.getpid(), tid, lane="project")
    meta = workers.RUNNING[tid] = {"task": task, "worker_id": 91, "attempt": 2}
    env = SimpleNamespace(drive_root=root, drive_path=lambda name: root / name)
    runner = SimpleNamespace(_event_queue=None, owner_wait_callback=None, **({"env": env} if own_terminal else {}))
    frames = []

    def unreadable(*args, **kwargs):
        assert args[1] == tid and kwargs.get("strict") is True
        raise StoreBusy("task store acquisition timed out", reason="contention")

    with monkeypatch.context() as outage, pytest.raises(PresenceTurnError) as raised:
        outage.setattr(presence_runner, "load_task_result", unreadable)
        run_worker_task(runner, task, wid=91, out_q=SimpleNamespace(put=frames.append), repo_dir=str(root),
                        drive_root=str(root))
    assert raised.value.code == "presence_result_unreadable"
    assert [frame for frame in frames if frame.get("type") == "presence_turn_settled"] == []
    assert workers.RUNNING[tid] is meta and slot.busy_task_id == tid
    after_failure = load_task_result(root, tid)
    assert after_failure["status"] == ("failed" if own_terminal else "running")
    # The exited worker's recovery: its own terminal published, else the crash policy's end of a second attempt.
    events, respawned = stdqueue.Queue(), []
    slot.proc = SimpleNamespace(pid=4242, is_alive=lambda: False, exitcode=0)
    monkeypatch.setattr(workers, "QUEUE_MAX_RETRIES", 1)
    monkeypatch.setattr(workers, "get_event_q", lambda *_a, **_k: events)
    monkeypatch.setattr(workers, "send_with_budget", lambda *_a, **_k: None)
    monkeypatch.setattr(workers, "_reconcile_confirmed_dead_review_owner", lambda *_a: None)
    monkeypatch.setattr(task_reaper, "_respawn_after_reap", lambda *a, **_k: respawned.append(a))
    recover_confirmed_dead_worker({"kind": "confirmed_dead_worker", "worker_id": 91, "worker": slot, "task_id": tid,
                                   "meta": meta, "task": dict(task), "attempt": 2, "exitcode": 0,
                                   "drive_root": str(root)})
    stored, published = load_task_result(root, tid), []
    while not events.empty():
        published.append(events.get_nowait())
    assert respawned and queue_custody(root, tid, stored) == CUSTODY_TERMINAL
    assert [event["status"] for event in published if event.get("type") == "task_done"] == ["failed"]
    status, body = retained_work_view(stored, tid, root)
    assert (status, body["status"], body["outcome"], body["text"]) == (200, "failed", "silent", "")
    if own_terminal:  # its own terminal is published like any task's; the done ingress releases the row
        assert stored["reason_code"] == "task_exception" and workers.RUNNING[tid] is meta
    else:
        assert stored["reason_code"] == "worker_crash_retry_exhausted" and stored["status_reconciled_from"] == "running"
        assert body["code"] == "presence_attempt_outcome_unknown" and tid not in workers.RUNNING


@pytest.mark.parametrize("grant", [
    {"_budget_pause_resume": {"grant_id": "g", "pause_id": "p"}}, {"_owner_wait_resume": {"restart_transaction_id": "tx"}},
    {"budget_resumed_at": "2026-11-03T00:00:00+00:00", "_attempt": 1},
], ids=["owner_resume", "restart_wait", "unstarted_resume"])
def test_a_continuation_the_owner_or_a_restart_authorized_still_carries_with_no_terminal(root, monkeypatch, grant):
    from ouroboros.presence_carrier import carry_presence_turn
    from ouroboros.task_results import load_task_result, write_task_result

    _seed, built = _seed_and_built(root)
    tid = built["id"]
    write_task_result(root, tid, "running", metadata=built["metadata"], chat_id=built["chat_id"], _is_direct_chat=True,
                      source="presence", task_contract=built["task_contract"], result="Task is running.")
    _move(root, tid, PROJECT)
    carried = []

    class Author:
        def handle_task(self, task):
            carried.append(task["id"])
            return []

    monkeypatch.setattr("ouroboros.agent.make_agent", lambda **_k: Author())
    frames = []
    assert carry_presence_turn({**built, "_attempt": 2, **grant}, SimpleNamespace(put=frames.append), root, root, worker_id=91) == []
    assert carried == [tid] and frames == []
    assert load_task_result(root, tid)["status"] == "running"


def _phase_frames(frames):
    return [frame for frame in frames if frame.get("type") == "direct_turn_phase"]


def _model_frames(turn, tid):
    """A turn's first Main call, started and seen going out: the loop sends these only after the turn's re-entry."""
    turn.put({"type": "main_llm_call_state", "task_id": tid, "llm_call_id": "c1", "phase": "started"})
    turn.put({"type": "main_llm_call_state", "task_id": tid, "llm_call_id": "c1", "phase": "observed",
              "model_activity": {"phase": "awaiting_response"}})


def _through_a_project_slot(root, task, frames, put=None):
    """The real worker body, its endpoint and start-phase observer, on project slot 91 (never that slot's own agent)."""
    from supervisor.worker_process import run_worker_task

    run_worker_task(SimpleNamespace(_event_queue=None, owner_wait_callback=None), task, wid=91,
                    out_q=SimpleNamespace(put=put or frames.append), repo_dir=str(root), drive_root=str(root))


@pytest.mark.parametrize("refusal", ["crash_retry", "unmoved", "no_row", "unreadable_row"])
def test_a_saved_turn_the_carrier_refuses_claims_no_start_phase(root, monkeypatch, refusal):
    """#2862, the refused half of the pair below: the same worker body and observer as an admitted turn. The carrier's
    own gate comes before any agent, so no context or model frame exists, and no ``started`` of the worker's stands in."""
    from ouroboros import presence_runner
    from ouroboros.presence_events import PresenceTurnError
    from ouroboros.store import StoreBusy

    monkeypatch.setattr("ouroboros.agent.make_agent", lambda **_k: pytest.fail("a refused turn builds no agent"))
    _seed, built = _seed_and_built(root)
    if refusal != "no_row":
        _boot_window_row(root, built["id"], built, move=refusal != "unmoved")
    task = {**built, "drive_root": str(root), "_attempt": 2, "_phase_generation": "gen-new",
            "_working_recovery": {"cause": "worker_crash" if refusal == "crash_retry" else "owner_restart"}}
    frames = []
    if refusal == "unreadable_row":
        def unreadable(*_args, **_kwargs):
            raise StoreBusy("task store acquisition timed out", reason="contention")

        with monkeypatch.context() as outage, pytest.raises(PresenceTurnError):
            outage.setattr(presence_runner, "load_task_result", unreadable)
            _through_a_project_slot(root, task, frames)
        assert frames == []
    else:
        _through_a_project_slot(root, task, frames)
        [settled] = [frame for frame in frames if frame.get("type") == "presence_turn_settled"]
        assert settled["error"]["code"] == "presence_attempt_outcome_unknown"
    assert _phase_frames(frames) == []


@pytest.mark.parametrize("prepare", [{"quiet": True, "reply_expected": False}, None], ids=["on_its_carrier", "on_a_slot"])
def test_a_host_turn_that_fails_its_readmission_claims_no_start_phase(root, monkeypatch, prepare):
    """The Host's turn re-admits from durable facts in its carrier; this root holds no binding, so it is refused there."""
    monkeypatch.setattr("ouroboros.agent.make_agent", lambda **_k: pytest.fail("a refused turn builds no agent"))
    seed, _built = _seed_and_built(root)
    frames = []
    _through_a_project_slot(root, {**seed, "drive_root": str(root), "_phase_generation": "gen-new",
                                   **({"_conversation_prepare": prepare} if prepare else {})}, frames)
    [settled] = [frame for frame in frames if frame.get("type") == "presence_turn_settled"]
    assert settled["error"] and settled["result"] is None
    assert _phase_frames(frames) == []


@pytest.mark.parametrize("grant", [
    {"_budget_pause_resume": {"grant_id": "g", "pause_id": "p"}}, {"_owner_wait_resume": {"restart_transaction_id": "tx"}},
    {"budget_resumed_at": "2026-11-03T00:00:00+00:00", "_attempt": 1},
], ids=["owner_resume", "restart_wait", "unstarted_resume"])
def test_a_readmitted_saved_turn_reports_its_own_start_phases_to_its_row_and_work_line(root, monkeypatch, grant):
    """#2862, the admitted half, on the order a real agent sends: its context is built before the loop's first step
    takes the conversation (``presence_carrier.reenter``), its model call after it. The first frame is that first
    context fact, each names this attempt, slot and dispatch, and each lands on the row the census words the work from."""
    from ouroboros import presence_carrier
    from ouroboros.task_results import write_task_result
    from supervisor import conversation_lane, workers
    from tests.test_presence_project_activity import _row, _state, _work

    _seed, built = _seed_and_built(root)
    tid = built["id"]
    write_task_result(root, tid, "running", metadata=built["metadata"], chat_id=built["chat_id"], _is_direct_chat=True,
                      source="presence", task_contract=built["task_contract"], result="Task is running.")
    _move(root, tid, PROJECT)
    task = {**built, "drive_root": str(root), "_attempt": 2, "_phase_generation": "gen-new", **grant}
    attempt = task["_attempt"]
    row = workers.RUNNING[tid] = {"task": task, "worker_id": 91, "attempt": attempt, "phase_generation": "gen-new"}
    supervisor = SimpleNamespace(RUNNING=workers.RUNNING)
    # The execution before this Resume kept the attempt and the slot; only the generation tells its late frame apart.
    conversation_lane.handle_direct_turn_phase({"task_id": tid, "task_attempt": attempt, "worker_id": 91, "execution_id": "old",
                                                "phase_generation": "gen-old", "phase": "dispatched"}, supervisor)
    assert "turn_phase" not in row
    frames, seen = [], {}

    def put(frame):
        frames.append(frame)
        if frame.get("type") == "direct_turn_phase":
            conversation_lane.handle_direct_turn_phase(frame, supervisor)

    def line():
        census, works, _complete = _state(root)
        work = _work(works, tid)
        return _row(census, tid).get("carrier"), row.get("turn_phase"), work["call"], [r["code"] for r in work["because"]]

    class Author:
        def __init__(self, turn):
            self.turn = turn

        def handle_task(self, task):
            seen["before_its_agent"] = _phase_frames(frames)
            for kind in ("task_started", "context_building_started"):
                self.turn.put({"type": "log_event", "data": {"type": kind, "task_id": tid}})
            seen["building"] = line()
            self.turn.put({"type": "log_event", "data": {"type": "context_building_finished", "task_id": tid}})
            ctx = SimpleNamespace(task_id=tid, task_metadata=task["metadata"],
                                  model_wait_context=SimpleNamespace(control_reason=lambda: None))
            state = presence_carrier.reenter(ctx)
            seen["held"] = bool(state and state.lease.held())
            _model_frames(self.turn, tid)
            write_task_result(root, tid, "completed", metadata=task["metadata"], result="Done.", terminal_origin="model_final")
            return [{"type": "presence_result", "outcome": "message", "text": "Done.", "work_ref": ""},
                    {"type": "task_done", "task_id": tid, "status": "completed"}]

    monkeypatch.setattr("ouroboros.agent.make_agent", lambda **kwargs: Author(kwargs["event_queue"]))
    _through_a_project_slot(root, task, frames, put=put)
    phases = _phase_frames(frames)
    assert seen["before_its_agent"] == [] and seen["held"] is True
    assert seen["building"] == ("worker", "context_building", "preparing", ["turn_context_building"])
    assert phases == [{"type": "direct_turn_phase", "task_id": tid, "task_attempt": attempt, "worker_id": 91,
                       "phase_generation": "gen-new", "execution_id": phases[0]["execution_id"], "phase": phase}
                      for phase in ("context_building", "context_built", "requested", "dispatched")]
    assert row["turn_phase"] == "dispatched" and row["turn_execution"] == phases[0]["execution_id"]


def test_a_host_turn_on_its_carrier_reports_its_start_phases_from_its_first_context_frame(root, monkeypatch):
    """The admitted half of a Host turn, whose carrier (``CarrierTurn``) publishes through the same producer."""
    from ouroboros import presence_admission
    from ouroboros.task_results import write_task_result

    monkeypatch.setattr(presence_admission, "admit_presence_turn", lambda **_kw: _admission())
    seed, _built = _seed_and_built(root)
    tid, frames, seen = seed["id"], [], {}

    class Author:
        owner_wait_callback = None

        def __init__(self, turn):
            self.turn = turn

        def handle_task(self, task):
            seen["before_its_agent"] = _phase_frames(frames)
            for kind in ("context_building_started", "context_building_finished"):
                self.turn.put({"type": "log_event", "data": {"type": kind, "task_id": tid}})
            _model_frames(self.turn, tid)
            write_task_result(root, tid, "completed", metadata=task["metadata"], result="Hi", terminal_origin="model_final")
            return [{"type": "presence_result", "outcome": "message", "text": "Hi", "work_ref": ""}]

    monkeypatch.setattr("ouroboros.agent.make_agent", lambda **kwargs: Author(kwargs["event_queue"]))
    _through_a_project_slot(root, {**seed, "drive_root": str(root), "_attempt": 3, "_phase_generation": "gen-fresh",
                                   "_conversation_prepare": {"quiet": True, "reply_expected": False}}, frames)
    phases = _phase_frames(frames)
    assert seen["before_its_agent"] == []
    assert [(p["phase"], p["task_attempt"], p["worker_id"], p["phase_generation"]) for p in phases] == [
        (phase, 3, 91, "gen-fresh") for phase in ("context_building", "context_built", "requested", "dispatched")]
    assert len({p["execution_id"] for p in phases}) == 1


def test_the_real_warm_pause_producer_leaves_a_paused_work_the_owner_can_resume(harness, monkeypatch):
    """The actual warm park (a held critic) under the normal carried census: Paused work offers Resume and Stop, never Pause,
    and its row still says it is the moved text work a Restart returns."""
    from ouroboros import owner_pause
    from ouroboros.task_results import load_task_result
    from tests._budget_pause_exact_helpers import _install_queue
    from tests.test_presence_project_activity import _actions, _row, _state, _work

    h, found = harness, {}
    _install_queue(h.data, monkeypatch)
    release = owner_pause.release_fence

    def observe_before_resume(root_drive, tid, **kwargs):
        stored = load_task_result(root_drive, tid)
        assert stored["owner_wait"]["reason"] == "owner_pause" and stored["owner_wait"]["state"] == "waiting"
        # The fixture's author executes inline; admit the lane row the normal carried path exposes.
        _admit_lane_row({**stored, "id": tid})
        census, works, complete = _state(root_drive)
        found.update(row=_row(census, tid), work=_work(works, tid), complete=complete)
        return release(root_drive, tid, **kwargs)

    monkeypatch.setattr(owner_pause, "release_fence", observe_before_resume)
    test_an_owner_pause_beside_an_open_critic_parks_warm_lends_the_place_and_resume_replays_the_one_review(h)
    assert found["complete"] is True
    assert found["row"]["phase"] == "budget_paused" and found["row"]["presence_pause"] is True
    assert found["work"]["activity"] == "paused"
    assert _actions(found["work"]) == ["resume", "stop"]


def test_a_saved_record_is_the_turns_own_and_never_its_seed(root):
    from ouroboros.presence_carrier import park_record, saved_turn_record
    from ouroboros.task_results import write_task_result

    seed, built = _seed_and_built(root)
    tid = built["id"]
    seed.update(_attempt=1, _budget_pause={"exact_continuation": True})
    carried = saved_turn_record(seed, root, carried={**built, "_execution_owner": {"kind": "presence"}})
    assert carried["task_contract"] == built["task_contract"] and carried["_budget_pause"] == seed["_budget_pause"]
    assert "_execution_owner" not in carried and carried["metadata"]["presence_request_waits"] is False
    # A record of another event is never taken; with no durable record either, the row keeps a typed refusal on its
    # own pause point, so the chooser never dispatches it (``worker_assignment._candidate_ineligible``).
    from ouroboros.work_stop import continuation_refusal

    foreign = {**built, "metadata": {**built["metadata"], "presence_event_identity": "another"}}
    assert saved_turn_record(seed, root, carried=foreign) is None
    held = park_record(seed, root, foreign)
    assert continuation_refusal(held).get("reason") == "presence_record_unavailable" and "task_contract" not in held
    # The durable row the turn's start wrote is the record when no event carries one.
    write_task_result(root, tid, "running", metadata=built["metadata"], chat_id=built["chat_id"],
                      task_contract=built["task_contract"], result="Task is running.")
    durable = saved_turn_record(seed, root)
    assert durable["task_contract"] == built["task_contract"] and "_presence_run" not in durable


def test_a_moved_turns_saved_state_keeps_its_coverage_and_handoff(root):
    from ouroboros.owner_wait import continuation_state
    from tests._budget_pause_exact_helpers import _loop_ctx

    ctx, limit = _loop_ctx(root, "covered-turn")
    ctx._presence_delivered = [{"source_event_id": "telegram:bot-1:43", "message": {"message_id": "43"}}]
    ctx._presence_covered = 1
    ctx._swarm_handoff_attempt = {"task_id": "child-1", "status": "scheduled"}
    state = continuation_state(ctx, limit.messages, {}, {}, 3, [], set())
    delivery = state["delivery"]
    assert delivery["_presence_delivered"] == ctx._presence_delivered and delivery["_presence_covered"] == 1
    assert delivery["_swarm_handoff_attempt"] == {"task_id": "child-1", "status": "scheduled"}


@pytest.mark.parametrize("rail", ["global_exhausted", "dispatch_refused", "control_authority_unreadable"])
def test_every_money_and_authority_rail_saves_a_moved_turn_exactly_with_no_generic_callback_unmoved_keeps_its_terminal(
        root, monkeypatch, rail):
    from ouroboros import budget_pause as bp
    from ouroboros.gateway.projects import _owner_task_origin
    from ouroboros.projects_registry import bind_task_to_project
    from ouroboros.task_results import write_task_result
    from tests._budget_pause_exact_helpers import _fast_hold, _loop_ctx, _quiet_external

    _seed, built = _seed_and_built(root)
    tid = built["id"]
    _fast_hold(monkeypatch, bp)
    _quiet_external(monkeypatch, bp)
    write_task_result(root, tid, "running", metadata=built["metadata"], chat_id=built["chat_id"], result="Task is running.")
    ctx, limit = _loop_ctx(root, tid)
    ctx.owner_wait_callback, ctx.task_metadata = None, built["metadata"]
    # Unmoved: the rail has no continuation owner, so the turn keeps today's terminal.
    assert bp.request_pause(limit, rail=rail, scope="task", reason_text="money gone") is None
    assert limit.accumulated_usage["exact_pause_unavailable"] == "no_continuation_owner"
    # Moved as text: the same rail saves its exact pause, with no generic owner wait granted.
    limit.accumulated_usage.pop("exact_pause_unavailable")
    bind_task_to_project(root, tid, PROJECT, origin=_owner_task_origin(root, tid))
    with pytest.raises(bp.BudgetPauseRequested) as saved:
        bp.request_pause(limit, rail=rail, scope="task", reason_text="money gone")
    bp.end_dispatch_fence(tid)
    assert saved.value.pause["rail"] == rail and saved.value.pause["source_ref"]
    assert ctx.owner_wait_callback is None


def test_an_owner_pause_beside_an_open_critic_parks_warm_lends_the_place_and_resume_replays_the_one_review(harness):
    """The real review dispatch loop with a held critic (GO: a scenario model, no paid canary). The owner's Pause detaches
    the moved author from the running wave; it parks warm through its qualified review wait with no dispatch fence, so its
    critic can still send (a cold pause's fence refuses that, the control); the Place is lent and a second event runs; the
    critic finishes; Resume retakes the conversation before any call and the same review replays once, unpaid."""
    from ouroboros import budget_pause as bp, owner_pause, review_pause
    from ouroboros.gateway.projects import _owner_task_origin
    from ouroboros.projects_registry import bind_task_to_project
    from ouroboros.task_results import load_task_result
    from ouroboros.usage_accounting import DispatchFenced, _check_dispatch_fences
    from ouroboros.usage_origin import UsageScope
    from tests.test_presence_continuation import ANSWER, RESUMED, _seen, deliver, event, quick_turn, review, wait_for

    h, again = harness, []
    h.addressed = None  # nothing but the Pause detaches the author from its running wave

    def step(messages):
        if not _seen(messages, RESUMED, role="user"):
            return review()
        if not again:  # the same stack asks for the same review once it holds the conversation again
            again.append(True)
            return review("review-again")
        return deliver()

    h.script = step
    execution = h.start(event())
    assert h.entered.wait(90), "the critic never started"
    tid = execution.turn_id
    bind_task_to_project(h.data, tid, "warm-room", origin=_owner_task_origin(h.data, tid))
    owner_pause.install_fence(h.data, tid, request_id="pause-warm")

    def parked_warm():
        wait = (load_task_result(h.data, tid) or {}).get("owner_wait") or {}
        return wait.get("state") == "waiting" and wait.get("reason") == "owner_pause"

    wait_for(parked_warm, what="the warm park")
    initial = execution.initial.result(timeout=60)
    assert (initial.status, initial.continuation_ref) == ("continuing", tid) and not execution.result.done()
    [operation] = review_pause.live_detached_operations(tid)
    assert review_pause.review_operation_state(operation["owner_id"]) == "running" and h.calls == 1
    # No dispatch fence on this path, so the critic's next send passes; a cold pause's fence refuses the same send.
    reviewer = UsageScope(drive_root=h.data, task_id=tid, root_task_id=tid)
    assert bp.dispatch_fenced(tid) is False
    _check_dispatch_fences(reviewer, h.data)
    bp.begin_dispatch_fence(tid)
    with pytest.raises(DispatchFenced):
        _check_dispatch_fences(reviewer, h.data)
    bp.end_dispatch_fence(tid)
    # The Place is lent: the conversation's next event runs while the owner's Pause holds this work.
    second = quick_turn(h)
    assert second.text == "Noted." and not execution.result.done() and h.calls == 1
    # The critic finishes under the Pause; nothing wakes the author but the owner's Resume.
    h.release.set()
    wait_for(lambda: not review_pause.live_detached_operations(tid), what="the critic's end")
    assert h.calls == 1 and not execution.result.done()
    owner_pause.release_fence(h.data, tid, reason="owner_resume")
    final = execution.result.result(timeout=90)
    assert (final.status, final.outcome, final.text) == ("completed", "message", ANSWER)
    assert len(h.reviews) == 1, "Resume replayed the one retained review, no second paid seat"


def test_a_carried_turns_first_attempt_proves_its_return_unstarted_only_for_its_exact_next_attempt():
    """A carried Presence turn's first launch stores attempt 0 where its checkpoint names 1, as a direct one does; a Project-
    bound return reads that shape at restore (``project_dispatch_unconfirmed`` otherwise). Its exact next attempt alone passes."""
    from supervisor.task_admission import _working_attempt_unstarted

    source = {"task_attempt": 1, "retained_work_dispatch_protocol": 1}

    def stored(kind, attempt=0, **extra):
        return {"task_id": "turn", "status": "running", "task_attempt": attempt,
                "execution_owner": {"kind": kind, "task_attempt": attempt}, **extra}

    for kind in ("presence", "direct"):
        assert _working_attempt_unstarted({"id": "turn", "_attempt": 2}, stored(kind), source) is True
        assert _working_attempt_unstarted({"id": "turn", "_attempt": 3}, stored(kind), source) is False  # not its next
        marked = stored(kind, admitted_dispatch_attempt=2)  # that attempt's own dispatch already landed
        assert _working_attempt_unstarted({"id": "turn", "_attempt": 2}, marked, source) is False
    assert _working_attempt_unstarted({"id": "turn", "_attempt": 2}, stored("pooled"), source) is False  # no such shape
    # Started under its returned attempt, a later return reads the ordinary rule.
    assert _working_attempt_unstarted({"id": "turn", "_attempt": 3}, stored("pooled", 2), source) is True


def test_recovery_turns_a_moved_turns_returned_seed_into_its_own_record_before_any_gate_reads_it(root, monkeypatch):
    """A lane seed carries a default contract of its own, so a contract proves nothing: run from it, a returned turn would
    lose its built ceiling, allowance, folder and origin. Recovery adopts the record its start wrote, or holds typed."""
    from ouroboros.gateway.projects import _owner_task_origin
    from ouroboros.presence_carrier import _saved_record, adopt_saved_records
    from ouroboros.projects_registry import bind_task_to_project
    from ouroboros.task_pacing import recorded_producer_ceiling
    from ouroboros.task_results import write_task_result
    from ouroboros.work_stop import continuation_refusal

    seed, built = _seed_and_built(root)
    tid = built["id"]
    folder = str(root / "pinned-folder")
    metadata = {**built["metadata"], "root_cost_ceiling_usd": CEILING, "focus": {"place_id": "place-1"}, "wake_cause": "focus",
                "resource_intent": {"kind": "explicit_resource"}}
    write_task_result(root, tid, "running", metadata=metadata, chat_id=built["chat_id"], workspace_root=folder,
                      workspace_mode="external", task_contract=built["task_contract"], result="Task is running.")
    bind_task_to_project(root, tid, PROJECT, origin=_owner_task_origin(root, tid))
    # The returned row: the seed, with the default contract a lane row carries and its recovery point.
    recovery = {"source_ref": {"path": "working"}, "from_attempt": 1, "source_task_id": tid}
    returned = {**seed, "task_contract": {"objective": "a lane's default contract"}, "_attempt": 2,
                "_working_recovery": recovery, "_presence_run": {"skill_name": "telegram-bot"}}
    assert _saved_record(returned) is False
    unmoved = {**seed, "id": tid + "-unmoved", "task_contract": {"objective": "default"}, "_attempt": 2}
    other = {"id": "ordinary", "type": "task", "metadata": {}, "task_contract": {"objective": "ordinary"}}
    untouched = [json.loads(json.dumps(row)) for row in (unmoved, other)]
    adopt_saved_records([returned, unmoved, other, "not-a-row"], root)
    assert [unmoved, other] == untouched, "only a moved text turn's row is adopted"
    assert _saved_record(returned) and returned["task_contract"] == built["task_contract"]
    assert recorded_producer_ceiling(returned) == CEILING and returned["metadata"]["focus"] == {"place_id": "place-1"}
    assert returned["workspace_root"] == folder and returned["metadata"]["resource_intent"] == {"kind": "explicit_resource"}
    assert returned["_working_recovery"] == recovery and returned["_attempt"] == 2 and "_presence_run" not in returned
    assert returned["chat_id"] == built["chat_id"]
    # Adopted once: a second recovery leaves it as it is.
    again = json.loads(json.dumps(returned))
    adopt_saved_records([returned], root)
    assert returned == again
    # No record of this event anywhere: held typed, never run from its seed.
    lost = tid + "-lost"
    bind_task_to_project(root, lost, PROJECT, origin=_owner_task_origin(root, tid))
    orphan = {**seed, "id": lost, "task_contract": {"objective": "default"}, "_attempt": 2, "_working_recovery": recovery}
    adopt_saved_records([orphan], root)
    assert continuation_refusal(orphan).get("reason") == "presence_record_unavailable"
    assert not _saved_record(orphan)


@pytest.mark.parametrize("by", ["owner", "mind"])
def test_a_restart_while_a_moved_turn_parks_warm_saves_its_pause_and_a_resume_replays_nothing(tmp_path, monkeypatch, by):
    """Real checkpoint, stop, snapshot, Resume and restore owners. The warm park becomes a saved pause; recovery gives the
    row its built record; a Resume re-sends nothing (the call sent before the Pause reads UNKNOWN), spend so far stays; only
    the owner's door moves the work off its allowance."""
    from ouroboros import budget_pause, owner_wait, task_pacing
    from ouroboros.control_actor import mind_control, owner_door
    from ouroboros.gateway.projects import _owner_task_origin
    from ouroboros.presence_carrier import _saved_record
    from ouroboros.presence_recovery import CUSTODY_QUEUED, queue_custody
    from ouroboros.projects_registry import bind_task_to_project
    from ouroboros.task_results import load_task_result, write_task_result
    from ouroboros.usage_accounting import usage_scope
    from ouroboros.usage_origin import task_usage_scope
    from tests._budget_pause_exact_helpers import _install_queue
    from tests.test_restart_retention import _pool_events
    from tests.test_warm_restart_retention import _handoff, _park, _stop_boot

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr("ouroboros.config.DATA_DIR", tmp_path)
    _pool_events(workers, monkeypatch)
    _seed, built = _seed_and_built(tmp_path)
    ctx, limit, wait, task = _park(tmp_path, workers, "owner_pause")
    tid = task["id"]
    metadata = {**built["metadata"], "root_cost_ceiling_usd": CEILING, "focus": {"place_id": "place-1"}}
    write_task_result(tmp_path, tid, "running", metadata=metadata, task_contract=built["task_contract"],
                      chat_id=built["chat_id"], source="presence", result="Task is running.")
    bind_task_to_project(tmp_path, tid, PROJECT, origin=_owner_task_origin(tmp_path, "no-captured-origin"))
    # The lane's row is the seed, with the default contract a lane row carries.
    task.update(type="presence", source="presence", _presence_turn=True, _is_direct_chat=True, chat_id=built["chat_id"],
                task_contract={"objective": "a lane's default contract"},
                metadata={key: built["metadata"][key] for key in ("presence", "presence_event_identity")})
    row = _stop_boot(queue, workers, _handoff(tmp_path, workers, wait))
    assert row["_budget_pause"]["exact_continuation"] is True, row
    without_lineage = lambda contract: {key: value for key, value in contract.items() if key != "lineage"}  # noqa: E731
    assert _saved_record(row) and row["metadata"]["focus"]  # lineage names this harness's fixed id, the rest is the turn's
    assert without_lineage(row["task_contract"]) == without_lineage(built["task_contract"])
    assert queue_custody(tmp_path, tid) == CUSTODY_QUEUED
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_k: 5.0)
    if by == "owner":
        with owner_door("api_task_resume"):
            outcome = queue.resume_budget_paused_task(tid)
    else:
        with mind_control("mind-1", "the owner asked me"):
            outcome = queue.resume_budget_paused_task(tid)
    assert outcome["ok"], outcome
    lifted = budget_pause.budget_pause_row(tmp_path, tid)["grant"].get("producer_allowance_lifted")
    assert bool(lifted) is (by == "owner")
    assert (row["metadata"].get("root_cost_ceiling_usd") is None) is (by == "owner")
    # The worker's restore of this row: nothing re-sent, spend kept, the allowance as the Resume decided.
    ctx.budget_pause_resume, ctx.task_metadata = row["_budget_pause_resume"], row["metadata"]
    saved = budget_pause.load_budget_pause(ctx)
    monkeypatch.setattr(owner_wait, "rebind_restored_route", lambda *_a, **_k: (None, "max"))
    messages, usage = [], {}
    scope = task_usage_scope(row, drive_root=tmp_path, root_task_id=tid, parent_task_id="", billing={"root_limit_usd": 5.0})
    with usage_scope(scope):
        budget_pause.resume_paused_loop(SimpleNamespace(_ctx=ctx), saved, messages, {}, usage, set(), budget_remaining_usd=5.0)
        authority = task_pacing.cost_stop_authority(ctx)
    assert authority == (task_pacing.COST_STOP_NONE if by == "owner" else task_pacing.COST_STOP_PRODUCER)
    assert scope.root_limit_usd == 5.0 and usage["cost"] == 1.25
    unknown = [m for m in messages if m.get("tool_call_id") == "call_b" and m.get("role") == "tool"]
    assert len(unknown) == 1 and "UNKNOWN" in unknown[0]["content"]
    assert load_task_result(tmp_path, tid)["status"] != "failed"


def test_a_saved_wait_restores_through_the_qualified_review_callback_and_no_generic_capability_opens(tmp_path, monkeypatch):
    from ouroboros import owner_wait
    from tests.test_continuation_tools import prepare

    tools, state, messages, trace, _enabled, _projections = prepare(tmp_path, monkeypatch)
    ctx, parked = tools._ctx, []
    ctx.owner_wait_callback, ctx.owner_wait_resume = None, dict(state, reason="review", review_binding="left-review")
    ctx.review_wait_callback = lambda *args: parked.append(args) or "review_settled"
    owner_wait.resume_native_loop(tools, state, messages, trace, {}, set())
    assert len(parked) == 1 and parked[0][0] is ctx and parked[0][2] is messages  # its own transcript takes the reentry note
    # With neither, the turn never parks generically: the wait has no owner and refuses.
    tools, state, messages, trace, _enabled, _projections = prepare(tmp_path / "second", monkeypatch)
    ctx = tools._ctx
    ctx.owner_wait_callback, ctx.review_wait_callback = None, None
    ctx.owner_wait_resume = dict(state, reason="review", review_binding="left-review")
    with pytest.raises(RuntimeError, match="no worker continuation owner"):
        owner_wait.resume_native_loop(tools, state, messages, trace, {}, set())


def test_a_rebuilt_record_keeps_the_attachment_authority_of_the_turn_it_saves(root, tmp_path):
    from ouroboros.gateway.projects import _owner_task_origin
    from ouroboros.presence_carrier import adopt_saved_records
    from ouroboros.presence_runner import _build_task
    from ouroboros.projects_registry import bind_task_to_project
    from ouroboros.task_results import write_task_result

    seed, _built = _seed_and_built(root)
    staged = tmp_path / "evidence.txt"
    staged.write_text("attached words", encoding="utf-8")
    built = _build_task(_admission(), _event(), drive_root=root, staged_files=(staged,), physical_task_id=seed["id"])
    manifest = built["task_contract"].get("attachment_manifest") or built["task_contract"].get("attachment_manifest_ref")
    assert manifest, "the built contract is where attachment authority lives"
    write_task_result(root, built["id"], "running", metadata=built["metadata"], chat_id=built["chat_id"],
                      task_contract=built["task_contract"], result="Task is running.")
    bind_task_to_project(root, built["id"], PROJECT, origin=_owner_task_origin(root, built["id"]))
    returned = {**seed, "task_contract": {"objective": "a lane's default contract"}, "_attempt": 2}
    adopt_saved_records([returned], root)
    assert returned["task_contract"] == built["task_contract"]


def test_a_parked_carrier_holds_until_its_own_effects_settled_through_an_early_resume_and_a_repeated_pause(root, monkeypatch):
    """The real hold over the durable pause row: an owner pause's sent work, the author's own tools and processes, and a
    money pause's own runs each keep the conversation and slot. A Resume granted meanwhile settles nothing; a newer Pause
    is not this one, yet its own effects still hold; a terminal (a Stop) ends the hold at once."""
    import threading
    import time

    from ouroboros import budget_pause as bp, presence_carrier, presence_continuation
    from ouroboros.owner_pause import SETTLEMENT_EXTERNAL_RUNNING
    from ouroboros.task_results import write_task_result

    _seed, built = _seed_and_built(root)
    tid = built["id"]
    monkeypatch.setattr(presence_carrier, "_HOLD_POLL_SEC", 0.01)
    monkeypatch.setattr(presence_carrier, "_RUNS_RECHECK_SEC", 0.01)
    blockers, runs_open = [], [False]
    monkeypatch.setattr(presence_continuation, "_custody_blockers", lambda *_a, **_k: list(blockers))
    monkeypatch.setattr(presence_carrier, "_task_runs_open", lambda *_a: runs_open[0])
    _ctx, _limit, pause = _pause(root, monkeypatch, task_id=tid)
    bp.end_dispatch_fence(tid)
    pause_id = pause["pause_id"]

    def row(**fields):
        bp.set_budget_pause(root, tid, {**bp.budget_pause_row(root, tid), **fields})

    def holding(reason="owner", settle_for=0.15):
        """A hold in its own thread; returns it after a moment in which it must not have returned on its own."""
        thread = threading.Thread(target=presence_carrier._hold_until_settled, args=(root, tid, pause_id, reason), daemon=True)
        thread.start()
        time.sleep(settle_for)
        return thread

    # The owner's Pause: its sent work keeps the hold, and an early Resume is no settlement of it.
    row(state=bp.STATE_PAUSED, settlement=SETTLEMENT_EXTERNAL_RUNNING)
    hold = holding()
    assert hold.is_alive(), "sent work still runs"
    row(state=bp.STATE_RESUME_GRANTED, grant={"grant_id": "early", "generation": 1})
    time.sleep(0.1)
    assert hold.is_alive(), "a Resume granted meanwhile settles nothing"
    # The sent work settled, the author's own tool has not: the conversation stays held.
    blockers.append({"kind": "tool"})
    row(settlement="")
    time.sleep(0.1)
    assert hold.is_alive(), "its own effects still run"
    # A repeated Pause is another pause; this hold no longer waits for that one's sent work, its own effects still count.
    row(pause_id="pause-newer", state=bp.STATE_PAUSED, settlement=SETTLEMENT_EXTERNAL_RUNNING)
    time.sleep(0.1)
    assert hold.is_alive(), "its own effects still hold across a repeated Pause"
    blockers.clear()
    hold.join(5)
    assert not hold.is_alive(), "nothing of its own runs: the hold ends"
    # A money pause: its own runs observed open keep the hold until they are stopped.
    row(pause_id=pause_id, state=bp.STATE_PAUSED, settlement="")
    runs_open[0] = True
    hold = holding("budget")
    assert hold.is_alive(), "a money pause waits for its own runs"
    runs_open[0] = False
    hold.join(5)
    assert not hold.is_alive()
    # A Stop's terminal ends every hold at once, whatever still reads open.
    blockers.append({"kind": "tool"})
    hold = holding()
    assert hold.is_alive()
    write_task_result(root, tid, "cancelled", result="Stopped.")
    hold.join(5)
    assert not hold.is_alive(), "a terminal ends the hold"
