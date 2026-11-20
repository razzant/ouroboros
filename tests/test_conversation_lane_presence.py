"""A Host-admitted Presence turn's body runs in a conversation-lane carrier (A7).

The Host keeps authentication, capture, dedup and its reservation; ``dispatch_presence_turn``
admits the body as a lane RUNNING row (origin ``presence``, carrier ``conversation``); the
carrier re-admits from durable facts, takes the gate itself, runs ``run_presence_turn`` and
reports its settlement as ONE frame the Host resolves its waiter from; errors travel by
value. An arrival for a carried author is accepted through one transaction under the
author's mailbox lock against its durable intake fact.
"""

import json
import os
import pathlib
import queue as stdqueue
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from ouroboros import presence_carrier, presence_runner
from ouroboros.presence_carrier import (
    PRESENCE_RUN_KEY, carry_presence_turn, dispatch_presence_turn, handle_presence_turn_settled, presence_seed_task,
)
from ouroboros.presence_events import PresenceTurnError
from ouroboros.presence_runner import PresenceTurnExecutions, PresenceTurnResult
from ouroboros.task_carrier import EXECUTOR_WORKER, LANE_CONVERSATION
from supervisor import conversation_lane, workers
from tests._presence_event import event_payload
from tests.test_owner_wait_pool import InputQueue, Process, pool as pool  # noqa: F401 -- fixture
from tests.test_presence_runner import _admission, _event
from tests.worker_pool_support import affirm_ready


_REAL_LANE_RECONCILE = conversation_lane.reconcile_conversation_lane


@pytest.fixture
def lane(pool, monkeypatch):
    monkeypatch.setattr(conversation_lane, "reconcile_conversation_lane", _REAL_LANE_RECONCILE)
    monkeypatch.setattr(conversation_lane, "_ARMED_TABLE", id(workers.WORKERS))
    monkeypatch.setattr(workers, "_get_chat_agent", lambda: pytest.fail("the server constructed an agent"))
    spare = workers.Worker(5, Process(5005), InputQueue(), lane=LANE_CONVERSATION)
    affirm_ready(spare)
    workers.WORKERS[5] = spare
    pool.spare = spare
    return pool


def test_the_host_dispatches_the_body_as_a_presence_lane_row(lane):
    admission, event = _admission(), _event()
    registry = PresenceTurnExecutions()
    turn_id = presence_runner.presence_turn_task_id(admission.binding_id, event.source_event_id)  # the Host's id
    execution, started = registry.start_or_dispatch(
        turn_id, identity="id-1", reserve=lambda: True, release=lambda: None,
        dispatch=lambda execution: dispatch_presence_turn(
            execution, drive_root=lane.root, skill_name="telegram-bot", admission=admission, event=event,
            event_payload=event_payload(event), staged_files=[], gate_wait=True))
    assert started and execution is registry.find(turn_id)
    rows = {tid: meta for tid, meta in workers.RUNNING.items() if meta.get("lane") == LANE_CONVERSATION}
    assert len(rows) == 1
    tid, meta = next(iter(rows.items()))
    task = meta["task"]
    assert task["_presence_turn"] and task["_execution_owner"]["kind"] == "presence"
    assert task["_execution_owner"]["lane"] == LANE_CONVERSATION and task["_execution_owner"]["executor"] == EXECUTOR_WORKER
    assert task["metadata"]["presence_request_waits"] is True
    dispatched = lane.spare.in_q.get_nowait()
    assert dispatched["id"] == tid and dispatched[PRESENCE_RUN_KEY]["skill_name"] == "telegram-bot"
    assert dispatched[PRESENCE_RUN_KEY]["event"]["source_event_id"] == event.source_event_id
    assert not workers.PENDING
    # The settlement frame resolves the waiter and releases the row and the slot.
    handle_presence_turn_settled({"turn_id": tid, "physical_id": tid, "error": None,
                                  "result": presence_carrier._result_payload(
                                      PresenceTurnResult(outcome="message", text="Hi", task_id=tid))},
                                 SimpleNamespace(RUNNING=workers.RUNNING, WORKERS=workers.WORKERS))
    assert execution.result.result(timeout=1).text == "Hi"
    assert tid not in workers.RUNNING and lane.spare.busy_task_id is None and registry.find(tid) is None


def test_the_carrier_runs_the_turn_under_the_gate_and_settles_by_frame(tmp_path, monkeypatch):
    repo, data = tmp_path / "repo", tmp_path / "data"
    repo.mkdir()
    data.mkdir()
    admission, event = _admission(), _event()
    from ouroboros import presence_admission
    monkeypatch.setattr(presence_admission, "admit_presence_turn", lambda **kw: admission)
    seen = {}

    class Agent:
        def __init__(self, **kwargs):
            self.owner_wait_callback = None

        def handle_task(self, task):
            seen.update(task)
            from ouroboros.task_results import write_task_result
            write_task_result(pathlib.Path(data), task["id"], "completed", metadata=task["metadata"], result="Hi")
            return [{"type": "presence_result", "outcome": "message", "text": "Hi", "work_ref": ""}]

    monkeypatch.setattr("ouroboros.agent.make_agent", lambda **kwargs: Agent(**kwargs))
    task = presence_seed_task(admission=admission, event=event, skill_name="telegram-bot",
                              event_payload=event_payload(event), staged_files=[], gate_wait=True)
    task["_execution_owner"] = {"kind": "presence", "lane": LANE_CONVERSATION, "executor": EXECUTOR_WORKER}
    out_q = stdqueue.Queue()
    carry_presence_turn(task, out_q, repo, data, owner_wait_callback="pooled-wait")
    frames = []
    while not out_q.empty():
        frames.append(out_q.get_nowait())
    settled = [f for f in frames if f["type"] == "presence_turn_settled"]
    assert len(settled) == 1 and settled[0]["error"] is None and settled[0]["result"]["text"] == "Hi"
    assert seen["_execution_owner"]["lane"] == LANE_CONVERSATION  # the carrier facts rode the presence record
    assert seen["metadata"]["presence_request_waits"] is True  # the Host request waits: the post-task rule's fact
    # The gate was taken and released by the carrier: a second turn in the conversation is admissible.
    gate = presence_runner._configured_gate(data)
    lease, busy = gate.try_admit(event.conversation_key)
    assert lease is not None and busy == ""
    lease.release()


def test_a_refusal_travels_by_value_to_the_waiter(lane):
    registry = PresenceTurnExecutions()
    execution, _started = registry.start_or_dispatch("presence-2", identity="id-2", reserve=lambda: True,
                                                      release=lambda: None, dispatch=lambda execution: None)
    error = PresenceTurnError("presence_capacity_full", "conversation", disposition="retry")
    error.retry_after = 1
    handle_presence_turn_settled({"turn_id": "presence-2", "physical_id": "presence-2", "result": None,
                                  "error": presence_carrier._error_payload(error)},
                                 SimpleNamespace(RUNNING=workers.RUNNING, WORKERS=workers.WORKERS))
    with pytest.raises(PresenceTurnError) as raised:
        execution.result.result(timeout=1)
    assert raised.value.code == "presence_capacity_full" and raised.value.retry_after == 1
    assert registry.find("presence-2") is None


def test_a_dispatch_that_raises_settles_the_waiter_and_reraises(lane):
    registry = PresenceTurnExecutions()

    def boom(execution):
        raise RuntimeError("no lane")

    with pytest.raises(RuntimeError):
        registry.start_or_dispatch("presence-3", identity="", reserve=lambda: True, release=lambda: None, dispatch=boom)
    assert registry.find("presence-3") is None


def test_a_remote_author_accepts_mail_only_while_its_durable_intake_is_open(tmp_path):
    from ouroboros.conversation_lease import (
        accept_author_mail, author_intake_accepting, close_author_intake, open_author_intake,
    )

    root = tmp_path
    (root / "memory" / "owner_mailbox").mkdir(parents=True)
    appended = []

    def append(admit=None):
        if admit is not None and not admit():
            return False
        appended.append(1)
        return True

    assert accept_author_mail(root, "author-1", "key-1", append) is False  # no fact: nobody reads
    open_author_intake(root, "author-1", "key-1")
    assert author_intake_accepting(root, "author-1", "key-1") and not author_intake_accepting(root, "author-1", "other")
    assert accept_author_mail(root, "author-1", "key-1", append) is True and appended == [1]
    assert accept_author_mail(root, "author-1", "other-key", append) is False
    close_author_intake(root, "author-1")
    assert accept_author_mail(root, "author-1", "key-1", append) is False and appended == [1]
    # The runner opens the fact before the body and closes it in its own finally once the body returned
    # (``presence_runner.run_presence_turn``), after the wait scope closed: never inside the agent's release.
    close_author_intake(root, "author-1")  # closing twice is quiet
    assert not author_intake_accepting(root, "author-1", "key-1")


_REAL_THREAD = threading.Thread  # the pool fixture stubs the lifecycle's threads; the inline carrier needs one


def _unarmed_host(pool, monkeypatch, *, server=True, stop_requested=None, expects_supervisor_pool=None):
    from supervisor import inline_carrier, message_bus, queue

    monkeypatch.setattr(conversation_lane, "_ARMED_TABLE", None)  # no generation is armed here
    monkeypatch.setattr(queue, "INITIALIZED", server)  # a Host serving alone owns no live queue
    monkeypatch.setattr(queue, "OWNER_PID", os.getpid())
    monkeypatch.setattr(conversation_lane, "_SPAWN_REFUSED", ["carrier_spawn_refused:OSError" if server else ""])
    assert queue.owns_live_queue(pool.root) is server
    monkeypatch.setattr(workers, "_get_chat_agent", lambda: pytest.fail("a Presence body needs no native agent"))
    monkeypatch.setattr(inline_carrier, "threading", SimpleNamespace(Thread=_REAL_THREAD))
    monkeypatch.setattr(inline_carrier, "_INLINE", type(inline_carrier._INLINE)(maxlen=50))
    monkeypatch.setattr(inline_carrier, "_DEGRADED_TOTAL", [0])
    monkeypatch.setattr(message_bus, "_BRIDGE", SimpleNamespace(send_chat_action=lambda *a, **k: None,
                                                                 broadcast=lambda m: None, push_log=lambda *a, **k: None))
    monkeypatch.setattr(workers, "send_with_budget", lambda *a, **k: None)
    monkeypatch.setattr(workers, "REPO_DIR", pool.root / "repo")
    bus = InputQueue()

    def event_queue(*, create=True):
        if server:
            return bus
        assert not create, "a standalone Host must not create an undrained event bus"
        raise RuntimeError("no supervisor event bus")

    monkeypatch.setattr(workers, "get_event_q", event_queue)
    monkeypatch.setattr(queue, "QUEUE_PUBLICATION", None)
    (pool.root / "logs").mkdir(parents=True, exist_ok=True)
    admission, event = _admission(), _event()
    from ouroboros import presence_admission
    monkeypatch.setattr(presence_admission, "admit_presence_turn", lambda **kw: admission)
    registry = PresenceTurnExecutions()
    turn_id = presence_runner.presence_turn_task_id(admission.binding_id, event.source_event_id)
    released = []

    def submit():
        return registry.start_or_dispatch(
            turn_id, identity="id-u", reserve=lambda: True, release=lambda: released.append(turn_id),
            dispatch=lambda execution: dispatch_presence_turn(
                execution, drive_root=pool.root, skill_name="telegram-bot", admission=admission, event=event,
                event_payload=event_payload(event), staged_files=[], gate_wait=True, stop_requested=stop_requested,
                expects_supervisor_pool=server if expects_supervisor_pool is None else expects_supervisor_pool))

    return SimpleNamespace(submit=submit, tid=turn_id, registry=registry, released=released)


class _DeferredThread:
    """Advance the inline body explicitly, including after a bounded join."""

    def __init__(self, *, target, **_kwargs):
        self.target, self.live = target, False

    def start(self):
        self.live = True

    def is_alive(self):
        return self.live

    def join(self, timeout=None):
        assert timeout is not None

    def run(self):
        assert self.live
        try:
            self.target()
        finally:
            self.live = False


@pytest.mark.parametrize("retrying", [False, True], ids=["first-attempt", "retry-after-reset"])
def test_pool_stop_of_inline_gate_waiter_survives_a_fresh_replay(pool, monkeypatch, retrying):
    from ouroboros.task_results import load_task_result, write_task_result

    host = _unarmed_host(pool, monkeypatch)
    # The fixture's ordinary process is unrelated. Only the thread belongs to this Stop.
    workers.WORKERS.clear()
    workers.RUNNING.clear()
    monkeypatch.setattr(conversation_lane, "_ARMED_TABLE", id(workers.WORKERS))
    gate = presence_runner._configured_gate(pool.root)
    monkeypatch.setattr(presence_carrier, "_configured_gate", lambda _root: gate)
    event = _event()
    identity = presence_runner.presence_event_identity(_admission().binding_id, event)
    if retrying:
        proof = {"kind": "first_round_engine_not_started", "task_id": host.tid,
                 "event_identity": identity, "reset_at": "2000-01-01T00:00:00+00:00"}
        write_task_result(pool.root, host.tid, "failed", reason_code="resource_refusal_no_resend",
                          metadata={"presence_event_identity": identity, "presence_retry_proof": proof})
    occupied, _ = gate.try_admit(event.conversation_key)
    assert occupied is not None
    waiting = threading.Event()
    acquire = gate.acquire_resources

    def wait_for_conversation(key, stopped=None):
        waiting.set()
        return acquire(key, stopped)

    monkeypatch.setattr(gate, "acquire_resources", wait_for_conversation)
    monkeypatch.setattr(workers, "_record_worker_processes", lambda: None)
    try:
        execution, _ = host.submit()
        assert waiting.wait(3)
        assert workers.kill_workers(terminal_status="cancelled", preserve_pending=True,
                                   reconcile_delegate_custody=False, reconcile_review_custody=False)
        with pytest.raises(PresenceTurnError) as stopped:
            execution.result.result(timeout=3)
        assert stopped.value.code == "presence_turn_cancelled"
    finally:
        occupied.release()
    assert host.tid not in workers.RUNNING and host.tid not in host.registry.live()
    physical_id = presence_runner._successor_id(host.tid) if retrying else host.tid
    stored = load_task_result(pool.root, physical_id, strict=True)
    assert stored["status"] == "cancelled" and stored["metadata"]["presence_event_identity"] == identity
    if retrying:
        old = load_task_result(pool.root, host.tid, strict=True)
        assert old["presence_retry_next"] == physical_id and old["status"] == "failed"
    with pytest.raises(PresenceTurnError) as retry:
        presence_runner.presence_turn_replay(pool.root, host.tid, event.conversation_key, identity)
    assert (retry.value.code, retry.value.disposition, retry.value.turn_ref) == (
        "presence_turn_cancelled", "retained", physical_id)


@pytest.mark.parametrize("completed", [True, False], ids=["natural-completion", "write-failure"])
def test_inline_gate_stop_settles_from_the_durable_row_and_recovers_without_a_tick(pool, monkeypatch, completed):
    from ouroboros import task_results

    _unarmed_host(pool, monkeypatch, server=False)
    admission, event = _admission(), _event()
    seed = presence_seed_task(admission=admission, event=event, skill_name=admission.transport_skill,
                              event_payload=event_payload(event), staged_files=[], gate_wait=True)
    tid, identity = seed["id"], seed["metadata"]["presence_event_identity"]
    seed["drive_root"] = str(pool.root)
    registry, released = PresenceTurnExecutions(), []
    execution, _ = registry.start_or_dispatch(tid, identity=identity, reserve=lambda: True,
                                               release=lambda: released.append(tid), dispatch=lambda _: None)
    meta = {"task": seed, "worker_id": -2, "_presence_execution": execution}
    workers.RUNNING[tid] = meta
    if completed:
        task_results.write_task_result(pool.root, tid, "completed", metadata=seed["metadata"],
                                       terminal_origin="model_final", result="Already answered")

    def unavailable(*args, **kwargs):
        raise OSError("temporary result write failure")

    with monkeypatch.context() as fault:
        fault.setattr(task_results, "write_task_result", unavailable)
        evt = {"error": {"code": "presence_turn_cancelled", "field": "conversation_key"}}
        presence_carrier.finish_inline_presence(meta, evt, workers)
    assert not registry.live() and released == [tid]
    if completed:
        assert execution.result.result(timeout=1).text == "Already answered"
    else:
        with pytest.raises(PresenceTurnError) as unknown:
            execution.result.result(timeout=1)
        assert (unknown.value.code, unknown.value.disposition) == ("presence_result_unreadable", "retry")
        assert workers.RUNNING[tid] is meta
        presence_carrier.retry_inline_gate_stop(pool.root, tid, identity)
        with pytest.raises(PresenceTurnError) as retained:
            presence_runner.presence_turn_replay(pool.root, tid, event.conversation_key, identity)
        assert (retained.value.code, retained.value.disposition) == ("presence_turn_cancelled", "retained")
    assert tid not in workers.RUNNING


@pytest.mark.parametrize("owner_pause", [True, False], ids=["owner-pause", "budget-park"])
def test_moved_turn_on_inline_fallback_keeps_its_running_pools_pause_owner(tmp_path, monkeypatch, owner_pause):
    from ouroboros import budget_pause as bp
    from ouroboros.gateway.projects import _owner_task_origin
    from ouroboros.model_wait import TaskModelWait
    from ouroboros.owner_pause import fence_closed, read_fence
    from ouroboros.projects_registry import bind_task_to_project
    from ouroboros.task_results import write_task_result
    from supervisor import inline_carrier
    from supervisor.owner_pause_control import request_owner_pause
    from tests._budget_pause_exact_helpers import _fast_hold, _install_queue, _loop_ctx, _quiet_external

    root = tmp_path
    _install_queue(root, monkeypatch)
    monkeypatch.setattr("ouroboros.config.DATA_DIR", root)
    monkeypatch.setattr("ouroboros.delegate_custody._CUSTODY", {})
    host = _unarmed_host(SimpleNamespace(root=root), monkeypatch)
    monkeypatch.setattr(conversation_lane, "_ARMED_TABLE", id(workers.WORKERS))
    monkeypatch.setattr(conversation_lane, "reconcile_conversation_lane", lambda: None)
    monkeypatch.setattr(inline_carrier, "threading", SimpleNamespace(Thread=_DeferredThread))
    _fast_hold(monkeypatch, bp)
    _quiet_external(monkeypatch, bp)
    task = presence_runner._build_task(_admission(), _event(), drive_root=root, staged_files=())
    tid, metadata = task["id"], task["metadata"]
    held = []

    def hold(_root, task_id, pause_id, _reason):
        pending, = workers.PENDING
        assert pending["id"] == task_id and pending["_budget_pause"]["checkpoint"]["pause_id"] == pause_id
        assert slot.busy_task_id == task_id
        held.append(task_id)

    monkeypatch.setattr(presence_carrier, "_hold_until_settled", hold)

    def run(*, admitted, event, event_sink, **kwargs):
        write_task_result(root, tid, "running", metadata=metadata, chat_id=task["chat_id"], source="presence")
        admitted.bind_author(tid)
        bind_task_to_project(root, tid, "activity-room", origin=_owner_task_origin(root, tid))
        ctx, limit = _loop_ctx(root, tid)
        ctx.owner_wait_callback, ctx.task_metadata = None, metadata
        ctx.model_wait_context = TaskModelWait(task=task, drive_root=root, event_queue=None, worker_slot_held=False)
        assert ctx.model_wait_context.inline_presence_stop is admitted.inline_stop is not None
        if owner_pause:
            assert request_owner_pause(tid, request_id="pause-inline")["ok"]
            assert fence_closed(read_fence(root, tid)) and presence_carrier.saves_exact_pause(ctx, bp.RAIL_OWNER_PAUSE)
            return PresenceTurnResult(outcome="silent", text="", task_id=tid)
        with pytest.raises(bp.BudgetPauseRequested) as saved:
            bp.request_pause(limit, rail=bp.RAIL_GLOBAL_EXHAUSTED, scope="task", reason_text="budget exhausted")
        bp.end_dispatch_fence(tid)
        parked = presence_carrier.park_moved_turn(
            root, task, [bp.pause_event(task, saved.value.pause)], agent=None, lease=admitted, event=event,
            identity=metadata["presence_event_identity"], sink=event_sink)
        assert parked is not None
        raise parked

    presence_carrier.set_presence_runner(root, run)
    try:
        execution, _ = host.submit()
        meta = workers.RUNNING[tid]
        assert conversation_lane.host_presence_row(meta) and meta["worker_id"] < -1
        slot = workers.WORKERS[meta["worker_id"]]
        slot.proc._thread.run()
        if owner_pause:
            assert execution.result.result(timeout=0).outcome == "silent"
        else:
            with pytest.raises(PresenceTurnError) as parked:
                execution.result.result(timeout=0)
            assert (parked.value.code, parked.value.custody) == ("presence_turn_parked", "queued")
            pending, = workers.PENDING
            assert pending["id"] == tid and pending["_budget_pause"]["exact_continuation"]
            assert bp.budget_pause_row(root, tid)["state"] == bp.STATE_PAUSED
            assert held == [tid]
        assert tid not in workers.RUNNING
    finally:
        presence_carrier.set_presence_runner(root, None)


def test_inline_presence_releases_its_waiter_and_slot_if_the_body_import_fails(pool, monkeypatch):
    import sys

    from ouroboros.presence_executions import PresenceTurnInterrupted
    from supervisor import inline_carrier
    from supervisor.active_activity import get_direct_activity_registry

    host = _unarmed_host(pool, monkeypatch, server=False)
    monkeypatch.setattr(inline_carrier, "threading", SimpleNamespace(Thread=_DeferredThread))
    execution, _ = host.submit()
    slot = workers.WORKERS[workers.RUNNING[host.tid]["worker_id"]]
    with monkeypatch.context() as missing:
        missing.setitem(sys.modules, "supervisor.worker_process", None)
        slot.proc._thread.run()
    with pytest.raises(PresenceTurnInterrupted):
        execution.result.result(timeout=0)
    assert slot.proc.exitcode == 1 and slot.wid not in workers.WORKERS and host.tid not in workers.RUNNING
    assert get_direct_activity_registry().get(host.tid) is None and host.released == [host.tid]


@pytest.mark.parametrize("startup", [False, True], ids=["ordinary_stop", "pool_startup"])
def test_pool_stop_takes_unbound_presence_reservations_before_join(pool, monkeypatch, startup):
    from ouroboros.presence_executions import PresenceTurnInterrupted
    from supervisor import inline_carrier, queue, worker_pool_lifecycle, worker_process

    host = _unarmed_host(pool, monkeypatch)
    monkeypatch.setattr(inline_carrier, "threading", SimpleNamespace(Thread=_DeferredThread))
    monkeypatch.setattr(workers, "kill_worker_tree", worker_pool_lifecycle.kill_worker_tree)
    monkeypatch.setattr(workers, "_record_worker_processes", lambda: None)
    monkeypatch.setattr(worker_process, "run_worker_task", lambda *a, out_q, **kw: out_q.put({
        "type": "presence_turn_settled", "turn_id": host.tid,
        "result": {"outcome": "silent", "text": "", "task_id": host.tid}}))
    publish, tear_down = queue.publish_queue_snapshot, workers._tear_down_slots
    reservation = []

    def check_before_join(slots):
        assert (workers.RUNNING.get(host.tid) is reservation[0]) is startup
        tear_down(slots)

    def stop_at_publication(**kwargs):
        published = publish(**kwargs)
        if kwargs["reason"] == "conversation_turn_admitted":
            reservation.append(workers.RUNNING[host.tid])
            assert workers.kill_workers(startup=startup, preserve_pending=True,
                                        reconcile_delegate_custody=False, reconcile_review_custody=False)
        return published

    monkeypatch.setattr(workers, "_tear_down_slots", check_before_join)
    monkeypatch.setattr(queue, "publish_queue_snapshot", stop_at_publication)
    if startup:
        execution, started = host.submit()
        assert started and workers.RUNNING[host.tid] is reservation[0]
        slot = workers.WORKERS[reservation[0]["worker_id"]]
        slot.proc._thread.run()
        assert execution.result.result(timeout=0).outcome == "silent"
    else:
        with pytest.raises(PresenceTurnInterrupted):
            host.submit()
    assert len(reservation) == 1 and host.released == [host.tid]
    assert host.tid not in workers.RUNNING and not workers.WORKERS


@pytest.mark.parametrize("dispatch", ["admission", "tick"])
def test_presence_after_refill_refusal_keeps_its_private_wait_and_slot_until_exit(pool, monkeypatch, dispatch):
    from ouroboros import owner_wait
    from supervisor import events_task_done, inline_carrier, queue, worker_pool_lifecycle

    host = _unarmed_host(pool, monkeypatch)
    monkeypatch.setattr(inline_carrier, "threading", SimpleNamespace(Thread=_DeferredThread))
    monkeypatch.setattr(conversation_lane, "_ARMED_TABLE", id(workers.WORKERS))
    refusal = conversation_lane.spawn_refusal()
    if dispatch == "tick":
        conversation_lane._SPAWN_REFUSED[0] = ""
    monkeypatch.setattr(workers, "kill_worker_tree", worker_pool_lifecycle.kill_worker_tree)
    monkeypatch.setattr(workers, "_record_worker_processes", lambda: None)
    waits = []

    def private_wait(lease, checkpoint):
        waits.append(checkpoint)
        assert workers.kill_workers(preserve_pending=True, reconcile_delegate_custody=False,
                                    reconcile_review_custody=False)
        assert slot.proc.stop_requested and lease.inline_stop.is_set()
        assert workers.WORKERS[slot.wid] is slot and workers.RUNNING[host.tid] is meta
        assert workers.last_worker_exit_census()["unconfirmed"] == [0]
        assert not presence_carrier.settle_ended_turn(host.tid)
        return "stopped"

    def runner(*, admitted, **_kwargs):
        assert admitted.inline_stop is not None and not admitted.inline_stop.is_set()
        initial = PresenceTurnResult(outcome="silent", text="", task_id=host.tid, status="continuing")
        admitted.execution.publish_initial(initial, return_capacity=False)
        assert execution.initial.result(timeout=0) == initial and not host.released
        assert admitted.review_wait(admitted, "private question") == "stopped"
        events_task_done._handle_task_done({"task_id": host.tid, "status": "completed"}, workers)
        assert workers.RUNNING[host.tid] is meta and not execution.result.done()
        return PresenceTurnResult(outcome="silent", text="", task_id=host.tid)

    monkeypatch.setattr(owner_wait, "direct_owner_wait", private_wait)
    presence_carrier.set_presence_runner(pool.root, runner)
    try:
        execution, started = host.submit()
        meta = workers.RUNNING[host.tid]
        if dispatch == "tick":
            assert meta["worker_id"] is None
            conversation_lane._SPAWN_REFUSED[0] = refusal
            with queue._queue_lock:
                assert conversation_lane.dispatch_awaiting_turns() == 1
        slot = workers.WORKERS[meta["worker_id"]]
        assert started and slot.wid < -1 and conversation_lane.host_presence_row(meta)
        assert host.submit() == (execution, False)
        slot.proc._thread.run()
        assert execution.result.result(timeout=0).outcome == "silent"
        assert waits == ["private question"] and host.released == [host.tid]
        assert slot.wid not in workers.WORKERS and host.tid not in workers.RUNNING
    finally:
        presence_carrier.set_presence_runner(pool.root, None)


def test_initial_thread_refusal_keeps_the_outcome_of_a_racing_lane_withdrawal(pool, monkeypatch):
    from supervisor import inline_carrier, queue

    host = _unarmed_host(pool, monkeypatch)
    monkeypatch.setattr(conversation_lane, "_ARMED_TABLE", id(workers.WORKERS))
    dispatch = conversation_lane.dispatch_admitted_turn

    def dispatch_and_tick(meta):
        try:
            dispatch(meta)
        except inline_carrier.InlineThreadNotStarted:
            with queue._queue_lock:  # the tick wins Q before the admitting Host retakes it
                conversation_lane.dispatch_awaiting_turns()
            raise

    monkeypatch.setattr(conversation_lane, "dispatch_admitted_turn", dispatch_and_tick)
    with patch.object(inline_carrier.InlineCarrier, "start", side_effect=RuntimeError("thread unavailable")) as start:
        with pytest.raises(presence_runner.PresenceTurnNotStarted):
            host.submit()
    assert start.call_count == 2 and host.released == [host.tid]
    assert host.tid not in workers.RUNNING and host.registry.find(host.tid) is None


def test_retried_presence_delivers_only_its_new_carriers_answer(pool, monkeypatch):
    from supervisor import inline_carrier, queue, worker_health, worker_process

    host = _unarmed_host(pool, monkeypatch)
    monkeypatch.setattr(inline_carrier, "threading", SimpleNamespace(Thread=_DeferredThread))
    monkeypatch.setattr(conversation_lane, "_ARMED_TABLE", id(workers.WORKERS))
    first = workers.Worker(5, Process(5005), InputQueue(), lane=LANE_CONVERSATION)
    affirm_ready(first)
    workers.WORKERS[first.wid] = first
    execution, started = host.submit()
    prior = workers.RUNNING[host.tid]
    assert started and prior["worker_id"] == first.wid
    assert "_presence_execution" not in first.in_q.get_nowait()

    first.proc.alive, first.proc.exitcode = False, 1
    job = {"worker": first, "worker_id": first.wid, "meta": prior, "task": prior["task"],
           "task_id": host.tid, "attempt": 1, "drive_root": str(pool.root)}
    next_task = {**prior["task"], "_attempt": 2}
    retry = worker_health._readmit_crash_retry(job, queue, next_task, 1)
    assert retry["attempt"] == 2 and retry["_presence_execution"] is execution
    assert "_presence_execution" not in retry["task"]
    workers.WORKERS.pop(first.wid)
    assert conversation_lane.publish_awaiting_rows()
    with patch.object(inline_carrier.InlineCarrier, "start", side_effect=RuntimeError("retry thread unavailable")) as start:
        with queue._queue_lock:
            assert conversation_lane.dispatch_awaiting_turns() == 0
    start.assert_called_once()
    assert not execution.result.done() and not host.released and workers.RUNNING[host.tid] is retry
    monkeypatch.setattr(worker_process, "run_worker_task", lambda *a, out_q, **kw: out_q.put({
        "type": "presence_turn_settled", "turn_id": host.tid,
        "result": {"outcome": "message", "text": "second", "task_id": host.tid}}))
    conversation_lane.dispatch_admitted_turn(retry)
    slot = workers.WORKERS[retry["worker_id"]]
    handle_presence_turn_settled({"turn_id": host.tid, "attempt": 1, "pid": first.proc.pid,
                                  "worker_id": first.wid, "result": {"outcome": "message", "text": "first",
                                                                   "task_id": host.tid}}, workers)
    assert not execution.result.done() and not host.released and workers.RUNNING[host.tid] is retry
    slot.proc._thread.run()
    assert execution.result.result(timeout=0).text == "second" and host.released == [host.tid]
    assert slot.wid not in workers.WORKERS and host.tid not in workers.RUNNING


@pytest.mark.parametrize("state", ["unreadable", "unrestored", "new_generation"])
def test_host_checks_readable_saved_queue_outside_lock_before_reserving(pool, monkeypatch, state):
    from supervisor import queue, queue_publication, queue_snapshot

    host = _unarmed_host(pool, monkeypatch)
    identity = [object()]
    before = dict(workers.RUNNING)
    reads = []

    def saved(*, require_readable):
        reads.append((require_readable, queue._queue_lock._is_owned()))
        if state == "unreadable":
            raise OSError("snapshot unavailable")
        if state == "new_generation":
            identity[0] = object()
        return state == "unrestored"

    monkeypatch.setattr(queue_publication, "publication_identity", lambda: identity[0])
    monkeypatch.setattr(queue_snapshot, "saved_queue_unowned", saved)
    monkeypatch.setattr(queue, "publish_queue_snapshot", lambda **kw: pytest.fail("nothing was admitted"))
    with pytest.raises(presence_runner.PresenceTurnNotStarted):
        host.submit()
    assert reads == [(True, False)] and host.released == [host.tid]
    assert workers.RUNNING == before and workers.WORKERS == {0: pool.original}


@pytest.mark.parametrize("new_generation", [False, True])
def test_failed_host_publication_withdraws_only_from_its_own_generation(pool, monkeypatch, new_generation):
    from supervisor import queue, queue_publication

    host = _unarmed_host(pool, monkeypatch)
    with queue._queue_lock:
        original = queue_publication.publication_identity()
    current, calls = [original], []

    def publish(*, reason, expected_identity):
        assert not queue._queue_lock._is_owned() and expected_identity == original
        calls.append(reason)
        if new_generation:
            current[0] = object()
        return False

    monkeypatch.setattr(queue_publication, "publication_identity", lambda: current[0])
    monkeypatch.setattr(queue, "publish_queue_snapshot", publish)
    with pytest.raises(presence_runner.PresenceTurnNotStarted):
        host.submit()
    assert calls == (["conversation_turn_admitted"] if new_generation
                     else ["conversation_turn_admitted", "presence_turn_not_started"])
    assert host.released == [host.tid] and workers.RUNNING == {"owner": pool.meta}


@pytest.mark.parametrize("server,fault", [(False, None), (True, None), (True, "admission"),
                                         (True, "withdrawal"), (True, "snapshot"), (True, "thread"),
                                         (True, "after_start")])
def test_inline_admission_failure_returns_its_reservation_and_the_same_event_runs_once(pool, monkeypatch, server, fault):
    from supervisor import inline_carrier, queue, queue_publication
    from supervisor.active_activity import get_direct_activity_registry

    host = _unarmed_host(pool, monkeypatch, server=server)
    if fault in {"admission", "withdrawal"}:
        workers.RUNNING.clear()  # even the first, partially published row can be withdrawn
    entered, finish = threading.Event(), threading.Event()
    ran = []

    def runner(**kwargs):
        ran.append(kwargs["carrier_owner"]["executor"])
        entered.set()
        assert finish.wait(10)
        return PresenceTurnResult(outcome="message", text="Hi", task_id=host.tid)

    presence_carrier.set_presence_runner(pool.root, runner)
    slot = None
    try:
        if fault == "after_start":
            start = inline_carrier.InlineCarrier.start

            def started_then_raised(carrier):
                start(carrier)
                raise RuntimeError("the thread is already running")

            monkeypatch.setattr(inline_carrier.InlineCarrier, "start", started_then_raised)
        elif fault:
            seen = []
            with monkeypatch.context() as failing:
                if fault == "thread":
                    def refuse_start(_carrier):
                        seen.append("thread")
                        raise RuntimeError("thread unavailable")

                    failing.setattr(inline_carrier.InlineCarrier, "start", refuse_start)
                if fault in {"admission", "withdrawal"}:
                    write_alive = queue_publication._write_alive

                    def refuse_certificate(path, data):
                        rows = json.loads(queue.QUEUE_SNAPSHOT_PATH.read_text())["running"]
                        if seen or any(row["id"] == host.tid for row in rows):
                            seen.append("certificate")
                            return False
                        return write_alive(path, data)

                    failing.setattr(queue_publication, "_write_alive", refuse_certificate)
                if fault in {"snapshot", "withdrawal"}:
                    write = queue.atomic_write_text

                    def refuse_snapshot(path, data):
                        if path == queue.QUEUE_SNAPSHOT_PATH:
                            admitting = any(row["id"] == host.tid for row in json.loads(data)["running"])
                            if (fault == "snapshot" and admitting) or (fault == "withdrawal" and seen):
                                seen.append("snapshot")
                                return False
                        return write(path, data)

                    failing.setattr(queue, "atomic_write_text", refuse_snapshot)
                with pytest.raises(presence_runner.PresenceTurnNotStarted) as refused:
                    host.submit()
                if fault == "thread":
                    assert str(refused.value) == "presence turn thread could not start"
                    assert isinstance(refused.value.__cause__, inline_carrier.InlineThreadNotStarted)
                    assert str(refused.value.__cause__.__cause__) == "thread unavailable"
                else:
                    assert str(refused.value) == "the presence admission could not be published"
            assert seen and not ran and host.released == [host.tid]
            assert host.registry.find(host.tid) is None and host.tid not in workers.RUNNING
            assert get_direct_activity_registry().get(host.tid) is None
            saved = json.loads(queue.QUEUE_SNAPSHOT_PATH.read_text())
            if fault == "withdrawal":
                from supervisor.queue_snapshot import saved_queue_unowned

                assert [r["id"] for r in saved["running"]] == [host.tid]  # only the uncertified first write landed
                assert not saved_queue_unowned() and not conversation_lane.awaiting_turns()
            else:
                assert {r["id"] for r in saved["running"]} == set(workers.RUNNING)
            assert not saved["pending"] and set(workers.WORKERS) == {0}
        execution, started = host.submit()
        assert started and entered.wait(5)
        slot = workers.WORKERS[workers.RUNNING[host.tid]["worker_id"]]
        if server:
            saved = json.loads(queue.QUEUE_SNAPSHOT_PATH.read_text())
            row = next(row for row in saved["running"] if row["id"] == host.tid)
            assert row["worker_id"] is None and slot.wid < -1
            assert row["task"]["_execution_owner"] == workers.RUNNING[host.tid]["task"]["_execution_owner"]
            assert row["task"]["_execution_owner"]["executor"] == "inline_degraded"
        assert host.submit() == (execution, False)
        finish.set()
        assert execution.result.result(timeout=5).text == "Hi"
        slot.proc.join(5)
        assert not slot.proc.is_alive() and ran == ["inline_degraded"]
        assert host.tid not in workers.RUNNING and host.registry.find(host.tid) is None
        assert len(host.released) == (1 if fault in {None, "after_start"} else 2)
        vitals = inline_carrier.inline_vitals()
        assert vitals["degraded_total"] == 1
        assert vitals["inline_executions"][-1]["reason"] == (
            "carrier_spawn_refused:OSError" if server else "pool_not_started")
    finally:
        finish.set()
        if slot:
            slot.proc.join(5)
        presence_carrier.set_presence_runner(pool.root, None)


@pytest.mark.parametrize("stopping", [False, True], ids=["serving", "shutting-down"])
def test_stop_captures_the_unstarted_turn_while_host_publication_is_blocked(pool, monkeypatch, stopping):
    from starlette.testclient import TestClient

    from ouroboros.cancel_intents import request_cancel
    from ouroboros.gateway import host_service
    from ouroboros.presence_executions import PresenceTurnInterrupted
    from ouroboros.task_results import load_task_result
    from supervisor import queue, queue_publication, task_lifecycle, worker_process

    host = _unarmed_host(pool, monkeypatch)
    monkeypatch.setattr(threading, "Thread", _REAL_THREAD)
    writing, unblock, captured = threading.Event(), threading.Event(), threading.Event()
    write_alive = queue_publication._write_alive
    take = conversation_lane.take_awaiting_turn
    ran = []

    def delay_host_certificate(path, data):
        rows = json.loads(queue.QUEUE_SNAPSHOT_PATH.read_text())["running"]
        # On the old path it is the *second* publication that first names the Host executor.
        if any(r["id"] == host.tid and r["task"]["_execution_owner"]["executor"] == "inline_degraded" for r in rows):
            writing.set()
            assert unblock.wait(20)
        return write_alive(path, data)

    def take_for_stop(tid):
        meta = take(tid)
        if tid == host.tid and meta is not None:
            captured.set()
        return meta

    monkeypatch.setattr(queue_publication, "_write_alive", delay_host_certificate)
    monkeypatch.setattr(conversation_lane, "take_awaiting_turn", take_for_stop)
    monkeypatch.setattr(worker_process, "run_worker_task", lambda *_a, **_kw: ran.append(host.tid))
    with ThreadPoolExecutor(max_workers=2) as requests:
        admission = requests.submit(host.submit)
        try:
            assert writing.wait(5)
            assert host.submit() == (host.registry.find(host.tid), False)
            request_cancel(pool.root, host.tid, reason="owner stop", source="owner", requested_by="owner")
            stop = requests.submit(task_lifecycle.cancel_task_custody, host.tid)
            assert captured.wait(5), "Stop must acquire Q and take the unstarted row before the disk is released"
        finally:
            unblock.set()
        with pytest.raises(PresenceTurnInterrupted):
            admission.result(timeout=10)
        assert stop.result(timeout=10) == task_lifecycle.CANCEL_CANCELLED
    stored = load_task_result(pool.root, host.tid)
    event = _event()
    assert stored["status"] == "cancelled"
    assert stored["metadata"]["presence_event_identity"] == presence_runner.presence_event_identity(
        _admission().binding_id, event)

    async def authenticated(*_args):
        return "telegram-bot", {}

    monkeypatch.setattr(host_service, "_authenticated", authenticated)
    with TestClient(host_service.create_host_service_app(pool.root, stop_requested=lambda: stopping)) as client:
        retry = client.post("/presence/turn", json={"event": event_payload(event)})
        if stopping:
            fresh = client.post("/presence/turn", json={"event": {**event_payload(event), "source_event_id": "next",
                                                                   "message": {"message_id": "43"}}})
            assert (fresh.status_code, fresh.json()["code"], fresh.json()["disposition"]) == (
                503, "presence_turn_not_started", "retry")
    assert (retry.status_code, retry.json()["code"], retry.json()["disposition"], retry.json()["turn_ref"]) == (
        409, "presence_turn_cancelled", "retained", host.tid)
    assert not ran and host.released == [host.tid] and host.registry.find(host.tid) is None
    assert host.tid not in workers.RUNNING and set(workers.WORKERS) == {0}


def test_a_shutdown_between_reservation_and_inline_bind_never_starts_the_body(pool, monkeypatch):
    from supervisor import inline_carrier, queue

    stopped = threading.Event()
    host = _unarmed_host(pool, monkeypatch, stop_requested=stopped.is_set)
    publish = queue.publish_queue_snapshot

    def publish_then_close(**kwargs):
        published = publish(**kwargs)
        if kwargs["reason"] == "conversation_turn_admitted":
            stopped.set()
        return published

    monkeypatch.setattr(queue, "publish_queue_snapshot", publish_then_close)
    monkeypatch.setattr(inline_carrier, "start_inline_turn", lambda *_a, **_kw: pytest.fail("late body bound"))
    with pytest.raises(presence_runner.PresenceTurnNotStarted, match="shutting down"):
        host.submit()
    assert host.tid not in workers.RUNNING and host.registry.find(host.tid) is None
    assert host.released == [host.tid]


@pytest.mark.parametrize("late", [False, True], ids=["after-publication", "waiting-for-carrier"])
def test_shutdown_before_pooled_presence_handoff_refuses_and_allows_one_retry(lane, monkeypatch, late):
    from starlette.testclient import TestClient

    from ouroboros.gateway import host_service
    from supervisor import inline_carrier, queue

    stopped, admitted = threading.Event(), threading.Event()
    host = _unarmed_host(lane, monkeypatch, stop_requested=stopped.is_set)
    monkeypatch.setattr(threading, "Thread", _REAL_THREAD)
    monkeypatch.setattr(conversation_lane, "_ARMED_TABLE", id(workers.WORKERS))
    monkeypatch.setattr(conversation_lane, "_SPAWN_REFUSED", [""])
    monkeypatch.setattr(conversation_lane, "replenish_after_claim", lambda: None)
    monkeypatch.setattr(inline_carrier, "start_inline_turn", lambda *_a, **_kw: pytest.fail("late inline body"))
    spare = lane.spare
    if late:
        spare.busy_task_id = "preceding-turn"

    async def authenticated(*_args):
        return "telegram-bot", {}

    def publish_then_close(**kwargs):
        result = publish(**kwargs)
        if kwargs["reason"] == "conversation_turn_admitted" and not admitted.is_set() and not late:
            stopped.set()
        return result

    def observe_dispatch(meta):
        try:
            dispatch(meta)
        finally:
            admitted.set()

    publish, dispatch = queue.publish_queue_snapshot, conversation_lane.dispatch_admitted_turn
    monkeypatch.setattr(host_service, "_authenticated", authenticated)
    monkeypatch.setattr(queue, "publish_queue_snapshot", publish_then_close)
    monkeypatch.setattr(conversation_lane, "dispatch_admitted_turn", observe_dispatch)
    app = host_service.create_host_service_app(lane.root, stop_requested=stopped.is_set)
    registry = app.state.host_service_context.presence_turns
    payload = {"event": event_payload(_event())}
    with TestClient(app) as client, ThreadPoolExecutor(max_workers=1) as requests:
        submitted = requests.submit(client.post, "/presence/turn", json=payload)
        try:
            assert admitted.wait(5)
            if late:
                assert workers.RUNNING[host.tid]["worker_id"] is None
                stopped.set()
                spare.busy_task_id = None
                assert conversation_lane.dispatch_awaiting_turns() == 0
            assert spare.busy_task_id is None and spare.in_q.empty()
            response = submitted.result(timeout=5)
            assert (response.status_code, response.json()["code"], response.json()["disposition"]) == (
                503, "presence_turn_not_started", "retry")
            assert host.tid not in workers.RUNNING and registry.find(host.tid) is None
            rows = json.loads(queue.QUEUE_SNAPSHOT_PATH.read_text())["running"]
            assert all(row["id"] != host.tid for row in rows)

            monkeypatch.setattr(queue, "publish_queue_snapshot", publish)
            admitted.clear()
            stopped.clear()
            retried = requests.submit(client.post, "/presence/turn", json=payload)
            assert admitted.wait(5)
            assert spare.in_q.get_nowait()["id"] == host.tid and spare.in_q.empty()

            stopped.set()
            assert conversation_lane.dispatch_awaiting_turns() == 0
            assert workers.RUNNING[host.tid]["worker_id"] == spare.wid
            assert not registry.find(host.tid).result.done()  # an assigned turn stays owned
            handle_presence_turn_settled({"turn_id": host.tid, "result": presence_carrier._result_payload(
                PresenceTurnResult(outcome="message", text="Hi", task_id=host.tid))}, workers)
            assert retried.result(timeout=5).status_code == 200
            assert spare.busy_task_id is None and spare.in_q.empty()
        finally:
            # On the base the fake carrier received the seed but cannot finish it.
            if execution := registry.find(host.tid):
                registry.settle_from_frame(execution, error=RuntimeError("test carrier has no body"))


@pytest.mark.parametrize("door", ["unpublished", "shutdown", "shutdown-unconfirmed", "startup-failed", "new-writer", "unconfirmed"])
def test_no_carrier_claim_is_needed_to_release_a_closed_presence_waiter(lane, monkeypatch, door):
    from supervisor import inline_carrier, queue, queue_publication
    from ouroboros.presence_executions import PresenceTurnInterrupted, PresenceTurnNotStarted

    stopped = threading.Event()
    host = _unarmed_host(lane, monkeypatch, stop_requested=stopped.is_set)
    monkeypatch.setattr(conversation_lane, "_ARMED_TABLE", id(workers.WORKERS))
    monkeypatch.setattr(conversation_lane, "_SPAWN_REFUSED", [""])
    monkeypatch.setattr(conversation_lane, "replenish_after_claim", lambda: None)
    monkeypatch.setattr(inline_carrier, "start_inline_turn", lambda *_a, **_kw: pytest.fail("late inline body"))
    lane.spare.busy_task_id = "earlier-turn"
    # No process operations, even on the bulk shutdown door.
    monkeypatch.setattr(workers, "_tear_down_slots", lambda slots: None)
    monkeypatch.setattr(workers, "_confirm_dead_slots", lambda slots: (set(), []))
    workers.RUNNING.pop("owner", None)
    workers.WORKERS.pop(0, None)
    publish = queue.publish_queue_snapshot
    writes = []

    def publication(**kwargs):
        writes.append(kwargs["reason"])
        if kwargs["reason"] == "conversation_turn_admitted" and door in {"unpublished", "shutdown"}:
            return False
        if kwargs["reason"] == "presence_turn_not_started" and door == "unconfirmed":
            return False
        if kwargs["reason"] == "server_shutdown" and door == "shutdown-unconfirmed":
            return None
        return publish(**kwargs)

    monkeypatch.setattr(queue, "publish_queue_snapshot", publication)
    execution, started = host.submit()
    assert started and not execution.result.done() and not host.released
    meta = workers.RUNNING[host.tid]
    assert meta["worker_id"] is None and lane.spare.in_q.empty()
    assert bool(meta.get("_unpublished")) == (door in {"unpublished", "shutdown"})
    lane.spare.busy_task_id = None
    if door == "unpublished":
        assert conversation_lane.dispatch_awaiting_turns() == 0  # still waits while open
        assert not execution.result.done()
    if door not in {"startup-failed", "new-writer"}:
        stopped.set()
    if door in {"shutdown", "shutdown-unconfirmed"}:
        settled = workers.kill_workers(stop_source="server_shutdown", preserve_pending=True,
                                       reconcile_delegate_custody=False, reconcile_review_custody=False)
        assert settled is (door == "shutdown")
        assert workers.last_worker_exit_census()["snapshot_ok"] is settled
        assert writes.count("server_shutdown") == 1 and "presence_turn_not_started" not in writes
    elif door == "startup-failed":
        conversation_lane.close_presence_after_failed_startup()
        assert not conversation_lane.armed()
    else:
        if door == "new-writer":
            monkeypatch.setattr(queue_publication, "_WRITER", queue_publication._TickWriter())
        assert conversation_lane.dispatch_awaiting_turns() == 0
    confirmed = door not in {"unconfirmed", "new-writer", "shutdown-unconfirmed"}
    error = PresenceTurnNotStarted if confirmed else PresenceTurnInterrupted
    for future in (execution.initial, execution.result):
        assert isinstance(future.exception(timeout=0), error)
    assert workers.RUNNING.get(host.tid) is None and host.registry.find(host.tid) is None
    assert lane.spare.in_q.empty() and host.released == [host.tid]
    if door == "new-writer":
        assert "presence_turn_not_started" not in writes
    rows = json.loads(queue.QUEUE_SNAPSHOT_PATH.read_text())["running"]
    assert any(row["id"] == host.tid for row in rows) is not confirmed


def _parked_inline_reservation(pool, monkeypatch):
    """An admitted, published Host-inline first message whose publisher waits between its
    admission and the bind, with the pool stop's process work stubbed out (no OS effects)."""
    from supervisor import inline_carrier, queue

    host = _unarmed_host(pool, monkeypatch)
    monkeypatch.setattr(conversation_lane, "replenish_after_claim", lambda: None)
    monkeypatch.setattr(inline_carrier, "start_inline_turn", lambda *_a, **_kw: pytest.fail("late body bound"))
    monkeypatch.setattr(workers, "_tear_down_slots", lambda slots: None)
    monkeypatch.setattr(workers, "_confirm_dead_slots", lambda slots: (set(), []))
    monkeypatch.setattr(workers, "RUNNING", {})
    parked, go = threading.Event(), threading.Event()
    publish = queue.publish_queue_snapshot

    def publication(**kwargs):
        published = publish(**kwargs)
        if kwargs["reason"] == "conversation_turn_admitted" and not parked.is_set():
            parked.set()
            assert go.wait(20)  # the publisher stays between its admission and the bind
        return published

    monkeypatch.setattr(queue, "publish_queue_snapshot", publication)
    return host, publish, parked, go


@pytest.mark.parametrize("confirmed", [True, False], ids=["published", "unconfirmed"])
def test_pool_stop_withdraws_an_unbound_inline_first_message_for_retry(pool, monkeypatch, confirmed):
    """An ordinary shutdown withdraws the admitted-but-unbound Host-inline first message: the
    waiter hears the retryable outcome and no terminal result burns the event."""
    from ouroboros.presence_executions import PresenceTurnInterrupted, PresenceTurnNotStarted
    from ouroboros.task_results import load_task_result
    from supervisor import inline_carrier, queue

    host, publish, parked, go = _parked_inline_reservation(pool, monkeypatch)
    if not confirmed:
        wrapper = queue.publish_queue_snapshot

        def unconfirmed_shutdown(**kwargs):
            if kwargs["reason"] == "server_shutdown":
                return None  # the closing snapshot cannot confirm the withdrawal
            return wrapper(**kwargs)

        monkeypatch.setattr(queue, "publish_queue_snapshot", unconfirmed_shutdown)
    with ThreadPoolExecutor(max_workers=1) as requests:
        admission = requests.submit(host.submit)
        assert parked.wait(5)
        execution = host.registry.find(host.tid)
        meta = workers.RUNNING[host.tid]
        assert execution is not None and meta["worker_id"] is None and meta.get("_inline_bind")

        settled = workers.kill_workers(terminal_status="cancelled", stop_source="server_shutdown",
                                        preserve_pending=True, reconcile_delegate_custody=False,
                                        reconcile_review_custody=False)
        assert settled is confirmed
        assert workers.RUNNING.get(host.tid) is None and host.released == [host.tid]
        assert not load_task_result(pool.root, host.tid, strict=True)  # no terminal burns the event
        error = PresenceTurnNotStarted if confirmed else PresenceTurnInterrupted
        assert isinstance(execution.result.exception(timeout=0), error)
        rows = json.loads(queue.QUEUE_SNAPSHOT_PATH.read_text())["running"]
        assert any(row["id"] == host.tid for row in rows) is not confirmed
        go.set()
        with pytest.raises(PresenceTurnInterrupted):
            admission.result(timeout=10)  # the fenced publisher finds its admission gone
        assert isinstance(execution.result.exception(timeout=0), error)  # its own settlement stands

        if confirmed:
            monkeypatch.setattr(queue, "publish_queue_snapshot", publish)
            bound = []

            def bind(bound_meta, **_kwargs):
                bound_meta["worker_id"] = -3
                bound_meta.pop("_inline_bind", None)
                bound.append(bound_meta["task"]["id"])

            monkeypatch.setattr(inline_carrier, "start_inline_turn", bind)
            retry, started = host.submit()  # the same event admits and binds afresh
            assert started and bound == [host.tid] and workers.RUNNING[host.tid]["worker_id"] == -3
            workers.RUNNING.pop(host.tid, None)
            host.registry.settle_from_frame(retry, error=PresenceTurnNotStarted("the test ends the turn"))


def test_pool_stop_leaves_an_explicitly_stopped_inline_first_message_to_its_stop(pool, monkeypatch):
    """A recorded owner Stop keeps its exact source and its event identity: the closing
    withdrawal never takes the row the Stop's custody owns."""
    from ouroboros.cancel_intents import request_cancel
    from ouroboros.presence_executions import PresenceTurnInterrupted
    from ouroboros.task_results import load_task_result

    host, _publish, parked, go = _parked_inline_reservation(pool, monkeypatch)
    with ThreadPoolExecutor(max_workers=1) as requests:
        admission = requests.submit(host.submit)
        assert parked.wait(5)
        execution = host.registry.find(host.tid)
        identity = workers.RUNNING[host.tid]["task"]["metadata"]["presence_event_identity"]
        request_cancel(pool.root, host.tid, reason="owner stop", source="owner", requested_by="owner")
        assert workers.kill_workers(terminal_status="cancelled", stop_source="server_shutdown",
                                    preserve_pending=True, reconcile_delegate_custody=False,
                                    reconcile_review_custody=False)
        stored = load_task_result(pool.root, host.tid, strict=True)
        assert (stored["status"], stored["cancel_origin"]["source"]) == ("cancelled", "owner")
        assert stored["metadata"]["presence_event_identity"] == identity  # a retry reads a retained refusal
        assert workers.RUNNING.get(host.tid) is None and not host.released and execution is not None
        go.set()
        with pytest.raises(PresenceTurnInterrupted):
            admission.result(timeout=10)
        assert host.released == [host.tid]


def test_a_plain_pool_kill_still_terminalises_an_unbound_inline_first_message(pool, monkeypatch):
    """Without an ordinary shutdown's typed source the full kill keeps its terminal: the
    withdrawal is the shutdown door's, never a weakened Panic."""
    from ouroboros.presence_executions import PresenceTurnInterrupted
    from ouroboros.task_results import load_task_result

    host, _publish, parked, go = _parked_inline_reservation(pool, monkeypatch)
    with ThreadPoolExecutor(max_workers=1) as requests:
        admission = requests.submit(host.submit)
        assert parked.wait(5)
        assert workers.kill_workers(preserve_pending=True, reconcile_delegate_custody=False,
                                    reconcile_review_custody=False)
        stored = load_task_result(pool.root, host.tid, strict=True)
        assert stored["status"] == "failed"  # the unstarted reservation ended with the pool, terminally
        assert workers.RUNNING.get(host.tid) is None
        go.set()
        with pytest.raises(PresenceTurnInterrupted):
            admission.result(timeout=10)


def test_pool_stop_removes_an_inline_reservation_whose_waiter_already_settled(pool, monkeypatch):
    """A row whose publisher already refused the fenced copy keeps only its removal: the
    stop takes it out of RUNNING without a settlement and without a spurious terminal."""
    from ouroboros.presence_executions import PresenceTurnInterrupted
    from ouroboros.task_results import load_task_result
    from supervisor import queue

    host, _publish, parked, go = _parked_inline_reservation(pool, monkeypatch)
    with ThreadPoolExecutor(max_workers=1) as requests:
        admission = requests.submit(host.submit)
        assert parked.wait(5)
        execution = host.registry.find(host.tid)
        assert execution is not None
        host.registry.settle_from_frame(execution, error=PresenceTurnInterrupted("the waiter heard its outcome"))
        assert host.released == [host.tid] and host.registry.find(host.tid) is None
        assert workers.kill_workers(terminal_status="cancelled", stop_source="server_shutdown",
                                    preserve_pending=True, reconcile_delegate_custody=False,
                                    reconcile_review_custody=False)
        assert workers.RUNNING.get(host.tid) is None
        assert not load_task_result(pool.root, host.tid, strict=True)  # no spurious terminal
        rows = json.loads(queue.QUEUE_SNAPSHOT_PATH.read_text())["running"]
        assert all(row["id"] != host.tid for row in rows)  # the stop's snapshot published the removal
        go.set()
        with pytest.raises(PresenceTurnInterrupted):
            admission.result(timeout=10)


def test_pool_stop_keeps_the_legacy_none_without_presence_withdrawals(pool, monkeypatch):
    from supervisor import queue

    monkeypatch.setattr(workers, "RUNNING", {})
    monkeypatch.setattr(workers, "_tear_down_slots", lambda slots: None)
    monkeypatch.setattr(workers, "_confirm_dead_slots", lambda slots: (set(), []))
    monkeypatch.setattr(queue, "publish_queue_snapshot", lambda *, reason: None)
    assert workers.kill_workers(stop_source="server_shutdown", preserve_pending=True,
                                reconcile_delegate_custody=False, reconcile_review_custody=False)
    assert workers.last_worker_exit_census()["snapshot_ok"] is True


@pytest.mark.parametrize("record", ["fresh", "legacy", "committed", "write-retry"])
def test_pool_stop_keeps_the_exact_source_of_an_explicitly_stopped_fresh_turn(lane, monkeypatch, record):
    from starlette.testclient import TestClient

    from ouroboros.cancel_intents import request_cancel
    from ouroboros.gateway import host_service
    from ouroboros.task_results import load_task_result, write_task_result
    from supervisor import inline_carrier
    from tests._store_shared import result_signature

    stopping = threading.Event()
    host = _unarmed_host(lane, monkeypatch, stop_requested=stopping.is_set)
    monkeypatch.setattr(threading, "Thread", _REAL_THREAD)
    monkeypatch.setattr(conversation_lane, "_ARMED_TABLE", id(workers.WORKERS))
    monkeypatch.setattr(conversation_lane, "_SPAWN_REFUSED", [""])
    monkeypatch.setattr(conversation_lane, "replenish_after_claim", lambda: None)
    monkeypatch.setattr(inline_carrier, "start_inline_turn", lambda *_a, **_kw: pytest.fail("turn must stay queued"))
    monkeypatch.setattr(workers, "_tear_down_slots", lambda slots: None)
    monkeypatch.setattr(workers, "_confirm_dead_slots", lambda slots: (set(), []))
    workers.RUNNING.pop("owner", None)
    workers.WORKERS.pop(0, None)
    lane.spare.busy_task_id = "earlier-turn"
    execution, started = host.submit()
    task = workers.RUNNING[host.tid]["task"]
    assert started and workers.RUNNING[host.tid]["worker_id"] is None and lane.spare.in_q.empty()
    identity = task["metadata"]["presence_event_identity"]
    if record == "legacy":
        del task["metadata"]["presence_event_identity"]
    request_cancel(lane.root, host.tid, reason="owner stop", source="owner", requested_by="owner")
    if record == "committed":
        write_task_result(lane.root, host.tid, "completed", result="already committed", metadata=task["metadata"])
        before = result_signature(lane.root, host.tid)
    stopping.set()
    stop_options = dict(terminal_status="cancelled", stop_source="server_shutdown", preserve_pending=True,
                        reconcile_delegate_custody=False, reconcile_review_custody=False)
    try:
        if record == "write-retry":
            def write_unavailable(*_a, **_kw):
                raise OSError("terminal store temporarily unavailable")

            with monkeypatch.context() as unavailable:
                unavailable.setattr(workers, "_write_failure_result", write_unavailable)
                workers.kill_workers(**stop_options)
            pending = next(row for row in workers.PENDING if row["id"] == host.tid)
            assert pending["_terminalization_retry"]["unstarted_presence"] is True
            assert not load_task_result(lane.root, host.tid, strict=True)
        assert workers.kill_workers(**stop_options)
        stored = load_task_result(lane.root, host.tid, strict=True)
        if record == "committed":
            assert result_signature(lane.root, host.tid) == before
            return
        assert stored["status"] == "cancelled" and stored["cancel_origin"]["source"] == "owner"
        if record == "legacy":
            assert "presence_event_identity" not in (stored.get("metadata") or {})
            return
        assert stored["source"] == "presence" and stored["chat_id"] == task["chat_id"]
        assert stored["metadata"]["presence_event_identity"] == identity

        async def authenticated(*_args):
            return "telegram-bot", {}

        monkeypatch.setattr(host_service, "_authenticated", authenticated)
        with TestClient(host_service.create_host_service_app(lane.root, stop_requested=stopping.is_set)) as client:
            retry = client.post("/presence/turn", json={"event": event_payload(_event())})
        assert (retry.status_code, retry.json()["code"], retry.json()["disposition"], retry.json()["turn_ref"]) == (
            409, "presence_turn_cancelled", "retained", host.tid)
    finally:
        if host.registry.find(host.tid) is execution:
            host.registry.settle_from_frame(execution, error=RuntimeError("test carrier has no body"))


def test_pool_expectation_is_resampled_before_a_no_pool_reservation(pool, monkeypatch):
    from supervisor import inline_carrier

    planned = threading.Event()
    host = _unarmed_host(pool, monkeypatch, server=False, expects_supervisor_pool=planned.is_set)
    seed = presence_carrier.presence_seed_task

    def after_settings_save(**kwargs):
        planned.set()
        return seed(**kwargs)

    monkeypatch.setattr(presence_carrier, "presence_seed_task", after_settings_save)
    monkeypatch.setattr(inline_carrier, "start_inline_turn", lambda *_a, **_kw: pytest.fail("startup ran inline"))
    with pytest.raises(presence_runner.PresenceTurnNotStarted, match="worker pool"):
        host.submit()
    assert host.tid not in workers.RUNNING and host.released == [host.tid]


@pytest.mark.parametrize("delayed", ["journal", "notice"])
def test_blocked_degraded_disclosure_allows_stop_and_the_thread_to_finish(pool, monkeypatch, delayed):
    from supervisor import inline_carrier, message_bus, queue, worker_process

    host = _unarmed_host(pool, monkeypatch)
    monkeypatch.setattr(threading, "Thread", _REAL_THREAD)
    monkeypatch.setattr(message_bus, "DATA_DIR", pool.root)
    blocked, unblock, finish = threading.Event(), threading.Event(), threading.Event()
    notices = []
    append_jsonl = inline_carrier.append_jsonl

    def delay(stage):
        if delayed == stage:
            blocked.set()
            assert unblock.wait(20)

    def journal(path, row):
        delay("journal")
        return append_jsonl(path, row)

    def notice(*args, **kwargs):
        delay("notice")
        notices.append(kwargs["task_id"])

    def body(agent, task, *, out_q, **_kwargs):
        assert finish.wait(20)
        assert out_q.stop_event.is_set()
        out_q.put({"type": "presence_turn_settled", "turn_id": task["id"],
                   "result": {"outcome": "silent", "text": "", "task_id": task["id"]}})

    monkeypatch.setattr(inline_carrier, "append_jsonl", journal)
    monkeypatch.setattr(workers, "send_with_budget", notice)
    monkeypatch.setattr(worker_process, "run_worker_task", body)
    with ThreadPoolExecutor(max_workers=1) as requests:
        admission = requests.submit(host.submit)
        slot = None
        try:
            assert blocked.wait(5)
            execution, started = host.submit()
            assert not started and not execution.result.done()
            assert queue._queue_lock.acquire(timeout=5), f"the delayed Host {delayed} held Q"
            try:
                slot = workers.WORKERS[workers.RUNNING[host.tid]["worker_id"]]
                assert slot.proc._ouroboros_stop_request()["confirmation"] == "unconfirmed"
            finally:
                queue._queue_lock.release()
            finish.set()
            assert execution.result.result(timeout=5).outcome == "silent"
            slot.proc.join(5)
            assert not slot.proc.is_alive() and slot.wid not in workers.WORKERS and host.tid not in workers.RUNNING
            assert not admission.done() and host.released == [host.tid]
        finally:
            finish.set()
            unblock.set()
            if slot is not None:
                slot.proc.join(5)
        assert admission.result(timeout=5) == (execution, True)
    lines = (pool.root / "logs" / "supervisor.jsonl").read_text().splitlines()
    rows = [json.loads(line) for line in lines]
    assert [r["task_id"] for r in rows if r.get("type") == "conversation_inline_executor"] == [host.tid]
    assert notices == [host.tid] and inline_carrier.inline_vitals()["degraded_total"] == 1


def test_withdrawing_a_partial_admission_keeps_a_foreign_snapshot(pool, monkeypatch):
    from supervisor import queue, queue_publication

    host = _unarmed_host(pool, monkeypatch)
    workers.RUNNING.clear()
    path = queue.QUEUE_SNAPSHOT_PATH
    foreign = '{"generation":"other","running":[],"pending":[{"id":"saved"}]}'

    def lost_certificate(_path, _data):
        assert [r["id"] for r in json.loads(path.read_text())["running"]] == [host.tid]
        path.write_text(foreign)
        return False

    with monkeypatch.context() as partial:
        partial.setattr(queue_publication, "_write_alive", lost_certificate)
        with pytest.raises(presence_runner.PresenceTurnNotStarted):
            host.submit()
    with pytest.raises(presence_runner.PresenceTurnNotStarted):
        host.submit()
    assert host.released == [host.tid, host.tid] and not workers.RUNNING and path.read_text() == foreign


def test_late_pool_start_and_timed_join_keep_the_live_host_turn(pool, monkeypatch):
    from ouroboros.task_results import load_task_result, write_task_result
    from supervisor import events_task_done, inline_carrier, worker_pool_lifecycle, worker_process

    host = _unarmed_host(pool, monkeypatch)
    entered, finish = threading.Event(), threading.Event()
    final_frame, exit_body = threading.Event(), threading.Event()
    stops = []

    def runner(**kwargs):
        write_task_result(pool.root, host.tid, "completed", result="kept answer")
        stops.append(kwargs["admitted"].inline_stop)
        entered.set()
        assert finish.wait(20)
        return PresenceTurnResult(outcome="message", text="kept answer", task_id=host.tid)

    presence_carrier.set_presence_runner(pool.root, runner)
    put = inline_carrier._InlineFrames.put

    def hold_final_frame(frames, item, *args, **kwargs):
        put(frames, item, *args, **kwargs)
        if isinstance(item, dict) and item.get("type") == "presence_turn_settled":
            final_frame.set()
            assert exit_body.wait(20)

    monkeypatch.setattr(inline_carrier._InlineFrames, "put", hold_final_frame)
    monkeypatch.setattr(workers, "kill_worker_tree", worker_pool_lifecycle.kill_worker_tree)  # fixture processes
    monkeypatch.setattr(workers, "_record_worker_processes", lambda: None)
    monkeypatch.setattr(workers.mp, "get_context", lambda _method: SimpleNamespace(Queue=InputQueue))
    monkeypatch.setattr(workers, "events_log_cursor", lambda: (0, 0, 0))
    monkeypatch.setattr(workers, "event_queue_generation", lambda: "test")
    monkeypatch.setattr(worker_process, "spawn_worker_process", lambda *_args: Process(900))
    stop_options = dict(preserve_pending=True, reconcile_delegate_custody=False, reconcile_review_custody=False)
    pool.original.proc.alive = False
    join = None
    try:
        execution, _ = host.submit()
        assert entered.wait(5)
        meta = workers.RUNNING[host.tid]
        slot = workers.WORKERS[meta["worker_id"]]
        join = slot.proc.join
        assert workers.kill_workers(startup=True, **stop_options)
        assert 0 not in workers.WORKERS and "owner" not in workers.RUNNING
        assert workers.worker_pool_admission_state()["disabled_reason"] == "no_workers"
        assert workers.ensure_worker_pool_started(1)
        assert workers.WORKERS[0].proc.pid == 900 and workers.WORKERS[slot.wid] is slot
        assert not stops[0].is_set() and slot.proc.is_alive()
        # A frame from a reused pool address cannot settle this live Host waiter.
        stale = {"turn_id": host.tid, "worker_id": 0, "pid": 900,
                 "result": {"outcome": "silent", "text": "", "task_id": host.tid}}
        presence_carrier.handle_presence_turn_initial(stale, workers)
        handle_presence_turn_settled(stale, workers)
        events_task_done._handle_task_done({"task_id": host.tid, "worker_id": 0, "status": "failed"}, workers)
        assert not execution.initial.done() and not execution.result.done() and workers.RUNNING[host.tid] is meta
        joins = []
        monkeypatch.setattr(slot.proc, "join", lambda timeout=None: joins.append(timeout))
        assert workers.kill_workers(disable_reason="managed_update", **stop_options)
        assert joins and all(timeout is not None for timeout in joins)
        assert stops[0].is_set() and slot.proc.is_alive() and workers.WORKERS[slot.wid] is slot
        assert workers.RUNNING[host.tid] is meta and host.submit() == (execution, False)
        assert not execution.result.done() and not host.released
        assert load_task_result(pool.root, host.tid)["status"] == "completed"
        assert workers.last_worker_exit_census()["unconfirmed"] == [0]
        assert not workers.ensure_worker_pool_started(1)
        assert workers.ensure_worker_pool_started(1, allow_disabled_restart=True)
        restored = workers.WORKERS[0]
        assert workers.WORKERS[slot.wid] is slot
        finish.set()
        assert final_frame.wait(5)
        assert not execution.result.done() and not host.released
        assert workers.WORKERS[slot.wid] is slot and workers.RUNNING[host.tid] is meta
        exit_body.set()
        assert execution.result.result(timeout=5).text == "kept answer"
        join(5)
        assert slot.wid not in workers.WORKERS and host.tid not in workers.RUNNING
        assert workers.WORKERS[0] is restored
        assert host.released == [host.tid]
    finally:
        finish.set()
        exit_body.set()
        if join:
            join(5)
        presence_carrier.set_presence_runner(pool.root, None)


@pytest.mark.parametrize("already_finished", [False, True])
def test_unarmed_cold_pause_has_a_durable_outcome_and_no_resume_debt(tmp_path, already_finished):
    from ouroboros.task_results import load_task_result, write_task_result

    admission, event = _admission(), _event()
    tid = presence_runner.presence_turn_task_id(admission.binding_id, event.source_event_id)
    identity = presence_runner.presence_event_identity(admission.binding_id, event)
    status = "completed" if already_finished else "running"
    write_task_result(tmp_path, tid, status, result="author's answer" if already_finished else "",
                      metadata={"presence_event_identity": identity})
    lease = SimpleNamespace(inline_stop=threading.Event())
    presence_carrier.park_moved_turn(tmp_path, {"id": tid}, [{"type": "budget_pause"}], lease=lease, agent=None,
                                    event=event, identity=identity, sink=lambda _: pytest.fail("queued a cold park"))
    row = load_task_result(tmp_path, tid)
    assert not row.get("budget_pause")
    if already_finished:
        assert (row["status"], row["result"]) == ("completed", "author's answer")
        result = presence_runner.finish_turn(tmp_path, {"chat_id": 1}, tid, identity, event, [],
                                             own_start=0, replay_completed=True)
        assert result.text == "author's answer"
    else:
        assert (row["status"], row["reason_code"]) == ("failed", "no_continuation_owner")
        with pytest.raises(PresenceTurnError) as retained:
            presence_runner.presence_turn_replay(tmp_path, tid, event.conversation_key, identity)
        assert (retained.value.code, retained.value.disposition) == ("presence_continuation_unavailable", "retained")
        from ouroboros.presence_recovery import retained_work_view

        status, view = retained_work_view(row, tid, tmp_path)
        assert (status, view["code"], view["reason_code"]) == (200, retained.value.code, "no_continuation_owner")


@pytest.mark.parametrize("saved", [False, True], ids=["empty_saved_queue", "saved_queue_not_restored"])
def test_a_server_before_its_pool_arms_refuses_the_turn_typed_and_touches_no_saved_queue(pool, monkeypatch, saved):
    """S9: in the server process (it owns the live queue of this root) a Presence event that arrives before the
    pool arms (the boot's window between ``queue.init`` and ``spawn_workers``, or after a whole-table stop) is not
    started: ``PresenceTurnNotStarted``, the typed 503 ``presence_turn_not_started`` retry the connector already
    handles (CREATING_SKILLS), never the degraded executor on a server thread. Nothing is admitted, so no one-row
    document is published over a saved queue this generation has not restored yet (#1952): the saved
    snapshot's bytes stay as they were."""
    import os

    from supervisor import queue
    from supervisor.inline_carrier import inline_vitals

    monkeypatch.setattr(conversation_lane, "_ARMED_TABLE", None)
    monkeypatch.setattr(queue, "INITIALIZED", True)
    monkeypatch.setattr(queue, "OWNER_PID", os.getpid())
    assert queue.owns_live_queue(pool.root)
    path = queue.QUEUE_SNAPSHOT_PATH
    if saved:  # the previous generation's queue, not restored by this one yet
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"ts": "2026-11-02T17:54:00+00:00", "pending": [{"id": "saved", "task": {"id": "saved"}}],
                                    "running": []}), encoding="utf-8")
    before = path.read_bytes() if saved else None
    degraded_before = inline_vitals()["degraded_total"]
    admission, event = _admission(), _event()
    from ouroboros import presence_admission
    monkeypatch.setattr(presence_admission, "admit_presence_turn", lambda **kw: admission)
    registry = PresenceTurnExecutions()
    turn_id = presence_runner.presence_turn_task_id(admission.binding_id, event.source_event_id)
    with pytest.raises(presence_runner.PresenceTurnNotStarted):
        dispatch_presence_turn(SimpleNamespace(turn_id=turn_id), drive_root=pool.root, skill_name="telegram-bot",
                               admission=admission, event=event, event_payload=event_payload(event), staged_files=[],
                               gate_wait=True)
    assert turn_id not in workers.RUNNING and registry.find(turn_id) is None
    assert inline_vitals()["degraded_total"] == degraded_before, "nothing ran on a server thread"
    if saved:
        assert path.read_bytes() == before, "the saved queue is untouched"
    else:
        assert not path.exists()


def test_a_server_bound_host_before_queue_init_refuses_typed_then_runs_the_same_event_once(pool, monkeypatch):
    """#3149: the embedded server's Host carries the boot factory's ``expects_supervisor_pool`` fact before it
    opens, so the pre-``queue.init`` boot window (``owns_live_queue`` still false) is the same typed refusal as
    the post-init unarmed one — before any admission, phase mark, queue persistence or body: no row, no
    pseudo-slot, no publication over the unrestored saved queue, the Host's reservation released. Once the
    pool arms, the same source event is admitted exactly once (one row, one carrier); a retry while live joins
    it, never a second body. The standalone Host without the fact keeps the degraded executor (the
    parametrized test above)."""
    from supervisor import queue
    from supervisor.inline_carrier import inline_vitals
    from tests.conversation_carrier_support import dispatched_tasks, install_lane_spare

    monkeypatch.setattr(conversation_lane, "_ARMED_TABLE", None)
    monkeypatch.setattr(conversation_lane, "_SPAWN_REFUSED", [""])
    monkeypatch.setattr(queue, "INITIALIZED", False)  # the boot window before queue.init
    monkeypatch.setattr(queue, "OWNER_PID", None)
    assert not queue.owns_live_queue(pool.root)
    path = queue.QUEUE_SNAPSHOT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"ts": "2026-11-17T00:00:00+00:00", "pending": [{"id": "saved", "task": {"id": "saved"}}],
                                "running": []}), encoding="utf-8")
    before = path.read_bytes()
    degraded_before = inline_vitals()["degraded_total"]
    admission, event = _admission(), _event()
    from ouroboros import presence_admission
    monkeypatch.setattr(presence_admission, "admit_presence_turn", lambda **kw: admission)
    registry = PresenceTurnExecutions()
    turn_id = presence_runner.presence_turn_task_id(admission.binding_id, event.source_event_id)
    released = []

    def dispatch(execution):
        dispatch_presence_turn(execution, drive_root=pool.root, skill_name="telegram-bot", admission=admission,
                               event=event, event_payload=event_payload(event), staged_files=[], gate_wait=True,
                               expects_supervisor_pool=True)

    with pytest.raises(presence_runner.PresenceTurnNotStarted):
        registry.start_or_dispatch(turn_id, identity="id-u", reserve=lambda: True,
                                   release=lambda: released.append(1), dispatch=dispatch)
    assert released == [1] and registry.find(turn_id) is None  # the reservation returned with the refusal
    assert turn_id not in workers.RUNNING
    assert all(getattr(slot, "executor", EXECUTOR_WORKER) == EXECUTOR_WORKER
               for slot in workers.WORKERS.values()), "no inline pseudo-slot on a server thread"
    assert inline_vitals()["degraded_total"] == degraded_before, "nothing ran on a server thread"
    assert path.read_bytes() == before, "the saved queue is untouched"

    # The supervisor generation completes: spawn arms the lane (``queue.init`` owns the queue from here on).
    spare = install_lane_spare(workers, 90)
    monkeypatch.setattr(conversation_lane, "_ARMED_TABLE", id(workers.WORKERS))
    assert conversation_lane.armed()
    execution, started = registry.start_or_dispatch(turn_id, identity="id-u", reserve=lambda: True,
                                                   release=lambda: None, dispatch=dispatch)
    assert started
    rows = {tid: meta for tid, meta in workers.RUNNING.items() if meta.get("lane") == LANE_CONVERSATION}
    assert list(rows) == [turn_id]  # exactly one body for the event
    assert rows[turn_id].get("worker_id") == spare.wid
    assert [task["id"] for task in dispatched_tasks(spare)] == [turn_id]
    joined, started_again = registry.start_or_dispatch(turn_id, identity="id-u", reserve=lambda: True,
                                                      release=lambda: None, dispatch=dispatch)
    assert joined is execution and not started_again  # a retry while live joins; no second row or body
    assert list(rows) == [turn_id] and dispatched_tasks(spare) == []


def test_an_owner_turns_intake_follows_its_own_loop_and_never_another_writers(tmp_path):
    """An owner turn's fact (``owner:<chat>``, its carrier's) moves with its loop's admission whatever the loop's
    metadata says; a fact another attempt or another process wrote stays as its writer left it."""
    import json

    from ouroboros.conversation_lease import (
        _intake_path,
        author_intake_accepting,
        follow_loop_admission,
        open_author_intake,
    )

    (tmp_path / "memory" / "owner_mailbox").mkdir(parents=True)
    loop = SimpleNamespace(task_id="main-turn", drive_root=tmp_path, task_metadata={}, task_attempt=None)
    assert open_author_intake(tmp_path, "main-turn", "owner:1", attempt=1)
    follow_loop_admission(loop, False)  # the final drain is about to read
    assert author_intake_accepting(tmp_path, "main-turn", "owner:1", attempt=1) is False
    follow_loop_admission(loop, True)  # a follow-up reopened its loop
    assert author_intake_accepting(tmp_path, "main-turn", "owner:1", attempt=1) is True
    follow_loop_admission(SimpleNamespace(**{**vars(loop), "task_attempt": 2}), False)  # not this attempt's loop
    assert author_intake_accepting(tmp_path, "main-turn", "owner:1", attempt=1) is True
    path = _intake_path(tmp_path, "main-turn")
    path.write_text(json.dumps({**json.loads(path.read_text()), "pid": 1}))
    follow_loop_admission(loop, False)  # another process's fact is its own
    assert json.loads(path.read_text())["accepting"] is True


def test_a_dead_carriers_intake_accepts_nothing_and_the_rows_end_closes_it(tmp_path):
    """A carrier that died before its release leaves ``accepting: true`` behind: the fact is its writer's, so a
    dead writer accepts nothing (an arrival would be swallowed), and the row's end in the server closes it."""
    import json
    import subprocess
    import sys

    from ouroboros.conversation_lease import author_intake_accepting, open_author_intake

    root = tmp_path
    (root / "memory" / "owner_mailbox").mkdir(parents=True)
    open_author_intake(root, "author-dead", "key")
    assert author_intake_accepting(root, "author-dead", "key")
    gone = subprocess.Popen([sys.executable, "-c", "pass"])
    gone.wait()  # reaped: its pid names no process now
    path = root / "memory" / "owner_mailbox" / "author-dead.intake.json"
    row = json.loads(path.read_text())
    row["pid"] = gone.pid
    path.write_text(json.dumps(row))
    assert not author_intake_accepting(root, "author-dead", "key"), "a dead writer reads no mail"
    open_author_intake(root, "author-live", "key")
    assert author_intake_accepting(root, "author-live", "key")
    handle_presence_turn_settled({"turn_id": "author-live", "physical_id": "author-live", "error": None, "result": None},
                                 SimpleNamespace(RUNNING={}, WORKERS={}, DRIVE_ROOT=root))
    assert not author_intake_accepting(root, "author-live", "key")
    assert not (root / "memory" / "owner_mailbox" / "author-live.intake.json").exists()


def test_the_intake_writers_liveness_is_the_platform_layers_probe(tmp_path, monkeypatch):
    """Review C3: the writer's liveness is ``platform_layer.pid_is_alive`` (OpenProcess on Windows), never a raw
    ``os.kill(pid, 0)``, which on Windows delivers CTRL_C_EVENT to the pid's console group and proves nothing."""
    import os

    from ouroboros import platform_layer
    from ouroboros.conversation_lease import author_intake_accepting, open_author_intake

    (tmp_path / "memory" / "owner_mailbox").mkdir(parents=True)
    open_author_intake(tmp_path, "author", "key")  # written by this (live) process
    probed = []
    monkeypatch.setattr(platform_layer, "pid_is_alive", lambda pid: probed.append(pid) or False)
    assert not author_intake_accepting(tmp_path, "author", "key")
    assert probed == [os.getpid()]
    monkeypatch.setattr(platform_layer, "pid_is_alive", lambda pid: True)
    assert author_intake_accepting(tmp_path, "author", "key")


def test_a_presence_turn_whose_carrier_ended_unreported_settles_at_its_published_terminal(tmp_path):
    """Review C2: a Stop's custody kill (or a crash, or a pool stop) ends a Presence carrier before its
    ``presence_turn_settled`` frame. The server's published terminal (the ``task_done`` dispatch, once the row left
    RUNNING) settles the Host waiter with a typed retry: its in-flight reservation returns, and a retry of the same
    event starts a fresh execution instead of joining the dead one forever. A turn its carrier already settled
    keeps its own answer: the terminal finds no live waiter to overwrite."""
    from ouroboros.presence_executions import PresenceTurnInterrupted
    from ouroboros.task_results import write_task_result
    from ouroboros.utils import append_jsonl
    from supervisor import events

    (tmp_path / "memory" / "owner_mailbox").mkdir(parents=True)
    metadata = {"source": "presence", "presence": {"binding_id": "b"}}
    ctx = SimpleNamespace(DRIVE_ROOT=tmp_path, RUNNING={}, WORKERS={}, append_jsonl=append_jsonl,
                          persist_queue_snapshot=lambda reason="": None)
    registry, released = PresenceTurnExecutions(), []

    def start(turn_id):
        return registry.start_or_dispatch(turn_id, identity="i", reserve=lambda: True,
                                          release=lambda: released.append(turn_id), dispatch=lambda execution: None)

    killed, started = start("pres-killed")
    assert started and registry.live() == ["pres-killed"]
    write_task_result(tmp_path, "pres-killed", "cancelled", result="Running task cancelled and worker terminated.",
                      metadata=metadata)  # the custody's terminal; the row already left RUNNING
    events._handle_task_done({"task_id": "pres-killed", "status": "cancelled", "task_type": "presence",
                              "chat_id": 5}, ctx)
    with pytest.raises(PresenceTurnInterrupted):
        killed.result.result(timeout=1)
    assert released == ["pres-killed"] and registry.live() == []
    retry, started_again = start("pres-killed")
    assert started_again and retry is not killed  # the resend is admitted afresh; its durable row answers it

    answered, _ = start("pres-answered")
    handle_presence_turn_settled({"turn_id": "pres-answered", "physical_id": "pres-answered", "error": None,
                                  "result": {"outcome": "message", "text": "hello", "task_id": "pres-answered"}}, ctx)
    write_task_result(tmp_path, "pres-answered", "completed", result="hello", metadata=metadata)
    events._handle_task_done({"task_id": "pres-answered", "status": "completed", "task_type": "presence",
                              "chat_id": 5}, ctx)
    assert answered.result.result(timeout=1).text == "hello" and released.count("pres-answered") == 1


def test_the_carrier_takes_the_hosts_admitted_signal_facts_by_value():
    """Review K2: the Host admits ``WAKE_FACTS`` plus the authenticated connector's declared signals (its live
    publication, in the server). The seed carries the facts the Host admitted, and the carrier's re-admission
    honours them by value, never its own process's registry; a fact nobody vouched for is still refused."""
    from dataclasses import replace

    from ouroboros.presence_events import presence_turn_event

    event = replace(_event(), facts=("mention", "signal.words"))
    payload = {"source_event_id": event.source_event_id, "provider": event.provider, "account_id": event.account_id,
               "conversation_id": event.conversation_id, "thread_id": event.thread_id,
               "conversation_key": event.conversation_key, "actor": dict(event.actor),
               "conversation": dict(event.conversation), "message": dict(event.message), "text": event.text,
               "facts": list(event.facts)}
    seed = presence_seed_task(admission=_admission(), event=event, skill_name="telegram-bot", event_payload=payload,
                              staged_files=[], gate_wait=False)
    vouched = seed[PRESENCE_RUN_KEY]["vouched_facts"]
    assert vouched == ["mention", "signal.words"]
    readmitted = presence_turn_event(seed[PRESENCE_RUN_KEY]["event"], _admission(), vouched_facts=vouched)
    assert readmitted.facts == ("mention", "signal.words")
    with pytest.raises(PresenceTurnError) as refused:
        presence_turn_event(seed[PRESENCE_RUN_KEY]["event"], _admission())
    assert refused.value.code == "presence_payload_invalid"


def test_the_carrier_takes_the_words_the_host_read_for_its_facts_by_value(tmp_path, monkeypatch):
    """A focus names its kinds in its connector's declared words (``focus.matched_labels``): the Host's door reads them
    onto its admission and the seed carries that map whole, by value, since attention also matches kinds the event
    never names (``from_bot`` from its author, ``any_message``). The carrier's re-admission, whose process has no
    registry, takes the map; its focus keeps the words of the kinds it matched, explicit or implicit, and no others."""
    from dataclasses import replace

    from ouroboros import presence_admission
    from ouroboros.places import Place, WakeItem
    from ouroboros.presence_context import focus_metadata

    place = Place(place_id="place-1", connector="telegram-bot", provider="telegram", account_id="bot-1",
                  conversation_id="room-1", thread_id="topic-1",
                  wakes=tuple(WakeItem(fact, "owner") for fact in ("signal.words", "from_bot", "any_message")))
    monkeypatch.setattr(presence_admission, "admit_presence_turn",  # its place, and no words of its own
                        lambda **_kw: replace(_admission(), place=place))
    seen = {}

    def module_runner(*, admission, event, **_kw):  # what the runner's focus reads (``presence_runner._build_task``)
        seen["words"], seen["focus"] = admission.fact_labels, focus_metadata(admission, event, "t", 1)["focus"]
        return PresenceTurnResult(outcome="silent", text="", task_id="t")

    monkeypatch.setattr(presence_carrier, "run_presence_turn", module_runner)
    words = {"signal.words": "Words you chose appear", "from_bot": "Written by a bot",
             "any_message": "Anything said here", "direct_message": "Mail to you"}
    declared = dict(words)
    event = replace(_event(), facts=("mention", "signal.words"), actor={**_event().actor, "is_bot": True})
    seed = presence_seed_task(admission=replace(_admission(), fact_labels=words), event=event,
                              skill_name="telegram-bot", event_payload={**event_payload(event), "facts": list(event.facts)},
                              staged_files=[], gate_wait=False)
    words.clear()  # the door's own map: the seed already holds its copy
    frames = stdqueue.Queue()
    presence_carrier.carry_presence_turn(json.loads(json.dumps(seed)), frames, tmp_path, tmp_path)
    assert frames.get_nowait()["error"] is None
    assert seen["focus"]["matched_wakes"] == ["signal.words", "from_bot", "any_message"], seen["focus"]
    assert seen["focus"].get("matched_labels") == {fact: declared[fact] for fact in seen["focus"]["matched_wakes"]}
    assert seen["words"] == seed[PRESENCE_RUN_KEY]["fact_labels"] == declared


def test_a_version_2_event_finding_the_gate_busy_is_refused_before_any_lane_admission(lane, monkeypatch):
    """Review M11: version 2 never waits in the gate. A busy gate is refused before the lane admits anything (no
    row, no barrier, no spare claimed, no carrier spawned) with base's answer: the busy resource named, retry
    after 1 s for the conversation and 5 s for the active-turn cap; a free gate admits the turn as usual."""
    from ouroboros.conversation_lease import PresenceTurnGate

    monkeypatch.setenv("OUROBOROS_PRESENCE_MAX_ACTIVE", "1")
    admission, event = _admission(), _event()
    other_process = PresenceTurnGate(1, state_root=lane.root / "state")  # another process's view of the same files

    def dispatch_v2():
        registry = PresenceTurnExecutions()
        turn_id = presence_runner.presence_turn_task_id(admission.binding_id, event.source_event_id)
        return registry.start_or_dispatch(
            turn_id, identity="id-v2", reserve=lambda: True, release=lambda: None,
            dispatch=lambda execution: dispatch_presence_turn(
                execution, drive_root=lane.root, skill_name="telegram-bot", admission=admission, event=event,
                event_payload=event_payload(event), staged_files=[], gate_wait=False))

    for held_key, busy, retry_after in ((event.conversation_key, "conversation", 1), ("chat:a:other:", "capacity", 5)):
        holder = other_process.acquire(held_key)
        try:
            with pytest.raises(PresenceTurnError) as refused:
                dispatch_v2()
        finally:
            holder.release()
        assert (refused.value.code, refused.value.field, refused.value.retry_after) == (
            "presence_capacity_full", busy, retry_after)
        assert not _lane_rows() and lane.spare.busy_task_id is None and lane.spare.in_q.empty()
    execution, started = dispatch_v2()
    assert started and len(_lane_rows()) == 1


def _lane_rows():
    return [meta for meta in workers.RUNNING.values() if meta.get("lane") == LANE_CONVERSATION]


def test_a_carried_presence_author_gets_no_generic_owner_wait_and_parks_its_review_through_the_carrier(
        tmp_path, monkeypatch):
    """Review M9: a Presence agent never holds the generic owner-wait capability (no sleep, detach, owner Pause or
    budget pause: those rails gate on it). The carrier's pooled wait reaches only the qualified review/private
    question park, through the turn's lease."""
    from ouroboros import agent as agent_module, presence_admission

    seen = {}
    monkeypatch.setattr(presence_admission, "admit_presence_turn", lambda **_kw: _admission())
    monkeypatch.setattr(agent_module, "make_agent", lambda **_kw: SimpleNamespace(owner_wait_callback=None))

    def module_runner(*, agent_factory, admitted, **_kw):  # the carrier's own runner path, with its factory
        agent = agent_factory(repo_dir=str(tmp_path), drive_root=str(tmp_path), event_queue=None)
        seen.update(generic=agent.owner_wait_callback, review=getattr(admitted, "review_wait", None))
        return PresenceTurnResult(outcome="silent", text="", task_id="t")

    monkeypatch.setattr(presence_carrier, "run_presence_turn", module_runner)
    pooled_wait = object()
    task = presence_seed_task(admission=_admission(), event=_event(), skill_name="telegram-bot",
                              event_payload=event_payload(_event()), staged_files=[], gate_wait=False)
    frames = stdqueue.Queue()
    presence_carrier.carry_presence_turn(task, frames, tmp_path, tmp_path, owner_wait_callback=pooled_wait)
    assert seen == {"generic": None, "review": pooled_wait}
    settled = [frame for frame in list(frames.queue) if frame.get("type") == "presence_turn_settled"]
    assert len(settled) == 1 and settled[0]["error"] is None


def test_a_durable_intake_follows_its_loops_admission_under_the_mail_lock(tmp_path, monkeypatch):
    """Review M10: the loop's admission transitions move a Presence author's durable intake (closed before the final
    drain, reopened for a follow-up); only the writer's own process moves its fact, a non-Presence turn has none,
    and open/close honour the mail lock (an unordered open is refused, a close never leaves the fact standing)."""
    import contextlib
    import json

    from ouroboros import task_custody
    from ouroboros.conversation_lease import (
        _intake_path, accept_author_mail, close_author_intake, follow_loop_admission, open_author_intake,
    )

    (tmp_path / "memory" / "owner_mailbox").mkdir(parents=True)
    appended = []

    def append(admit=None):
        if admit is not None and not admit():
            return False
        appended.append(1)
        return True

    author = SimpleNamespace(task_id="author-m10", drive_root=tmp_path, task_metadata={"presence": {"binding_id": "b"}})
    assert open_author_intake(tmp_path, "author-m10", "key") is True
    follow_loop_admission(author, False)  # the final drain is about to read
    assert accept_author_mail(tmp_path, "author-m10", "key", append) is False and appended == []
    follow_loop_admission(author, True)  # an owner follow-up reopened the loop
    assert accept_author_mail(tmp_path, "author-m10", "key", append) is True and appended == [1]
    follow_loop_admission(SimpleNamespace(task_id="author-m10", drive_root=tmp_path, task_metadata={}), False)
    assert accept_author_mail(tmp_path, "author-m10", "key", append) is True  # a non-Presence turn moves nothing
    path = _intake_path(tmp_path, "author-m10")
    path.write_text(json.dumps({**json.loads(path.read_text()), "pid": 1}))
    follow_loop_admission(author, False)
    assert json.loads(path.read_text())["accepting"] is True  # another process's fact is its own
    close_author_intake(tmp_path, "author-m10")

    monkeypatch.setattr(task_custody, "task_mail_lock", lambda *_a, **_k: contextlib.nullcontext(False))
    assert open_author_intake(tmp_path, "author-unordered", "key") is False
    assert not _intake_path(tmp_path, "author-unordered").exists()
    _intake_path(tmp_path, "author-held").write_text('{"accepting": true, "conversation_key": "key", "pid": 1}')
    assert close_author_intake(tmp_path, "author-held") is False and not _intake_path(tmp_path, "author-held").exists()
