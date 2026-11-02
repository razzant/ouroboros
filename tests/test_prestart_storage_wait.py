"""A new message whose execution provably has not started waits through storage contention and runs once
(owner 2026-10-21, A); its controls still end that wait, and nothing past it is replayed or held falsely.

Every test drives the real outer direct-turn lane (``worker_chat_lane._execute_chat_task``) or the real
agent ``handle_task`` against a real SQLite lock on temporary stores (``exclusive_hold``). A fake clock
stands in for long waits; no provider, server, port or live data is reached.

Each attempt on a held store still spends its whole busy wait before it reports contention: 10 s on the
current store (``store.STORE_WAIT_SEC``), 45 s on the money store (``USAGE_LOCK_TIMEOUT_SEC``). What the
lane does next does not depend on that length, so ``short_waits`` cuts both to ``SHORT_WAIT_SEC`` through
their call-time seams (``store.acquisition_budget``, the money store's read bound); production keeps its own.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from ouroboros import store, usage_store
from ouroboros.settings_integrity import task_settings_snapshot
from tests._budget_pause_exact_helpers import _install_queue
from tests._store_journal_shared import exclusive_hold

BINDING = {"billing_group_id": "S", "billing_group_limit_usd": 20.0,
           "billing_group_limit_source": "initial_task_admission", "billing_group_limit_revision": "cap-1"}
SHORT_WAIT_SEC = 0.05


@pytest.fixture(autouse=True)
def short_waits(monkeypatch):
    monkeypatch.setattr(usage_store, "USAGE_LOCK_TIMEOUT_SEC", SHORT_WAIT_SEC)
    with store.acquisition_budget(SHORT_WAIT_SEC):
        yield


@pytest.fixture
def lane(tmp_path, monkeypatch):
    """A direct turn of task ``S`` through the real lane, with the body observed and the clock faked."""
    import os

    from ouroboros import _usage_wait, queue_observation
    from supervisor import message_bus, worker_chat_lane
    from supervisor.active_activity import get_direct_activity_registry

    root = tmp_path / "data"
    (root / "state").mkdir(parents=True)
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(root))
    monkeypatch.setenv("OUROBOROS_SETTINGS_PATH", str(root / "settings.json"))
    _q, _state, workers = _install_queue(root, monkeypatch)
    world = SimpleNamespace(root=root, workers=workers, body=[], sleeps=[], errors=[], released=[], events=[],
                            clock=[0.0], on_sleep=lambda _n: None, blockers=[], body_error=None, replies=[])
    task = {"id": "S", "type": "task", "_is_direct_chat": True, "chat_id": 7, "_attempt": 1, "text": "keep this input"}
    world.task = task

    def scoped(received):
        world.body.append(received)
        assert not world.blockers or all(getattr(b, "closed", False) for b in world.blockers)
        if world.body_error is not None:
            raise world.body_error
        return []

    def sleep(_seconds):
        world.sleeps.append(world.clock[0])
        world.on_sleep(len(world.sleeps))

    monkeypatch.setattr(queue_observation, "_sleep", sleep)
    monkeypatch.setattr(_usage_wait, "time", SimpleNamespace(monotonic=lambda: world.clock[0]))
    monkeypatch.setattr("ouroboros.config.get_task_idle_timeout_sec", lambda: 900.0)
    monkeypatch.setattr("ouroboros.subagent_runtime.apply_task_start_settings_or_disclose",
                        lambda *_a, **_k: task_settings_snapshot({}, os.environ))
    monkeypatch.setattr("ouroboros.project_naming.spawn_turn_namer", lambda *_a, **_k: None)
    monkeypatch.setattr(workers, "get_event_q", lambda: SimpleNamespace(put=world.events.append))
    monkeypatch.setattr(message_bus, "get_bridge", lambda: SimpleNamespace(push_log=lambda _e: None,
                                                                            send_chat_action=lambda *_a, **_k: None))
    report_error = worker_chat_lane._report_direct_chat_error
    def observed_report(admitted, error):
        world.errors.append(error)
        report_error(admitted, error)
    monkeypatch.setattr(worker_chat_lane, "_report_direct_chat_error", observed_report)
    monkeypatch.setattr(workers, "send_with_budget", lambda *args, **kwargs: world.replies.append((args, kwargs)))
    host = _host(root, scoped)
    registry = get_direct_activity_registry()

    class Registry:  # the real registry; its unregister is observed
        def unregister(self, task_id):
            world.released.append(task_id)
            return registry.unregister(task_id)

    def run(agent=None):
        # As ``_admit_chat_task`` does: the turn is registered, "preparing", before it executes.
        registry.register("S", 7, client_message_id="m-1", kind="direct_chat", actor=agent or host,
                          reply_expected=True)
        return worker_chat_lane._execute_chat_task({
            "task": task, "agent": agent or host, "chat_id": 7, "kind": "chat", "client_msg_id": "m-1",
            "registry": Registry()})

    world.run = run
    yield world
    registry.unregister("S")
    for blocker in world.blockers:
        blocker.close()
    store.close_connections_under(root)


def _host(root, body):
    """The chat agent's surface the lane, the census and the owner controls read, with BODY as its body."""
    import queue as queue_mod
    import threading

    from ouroboros.agent import OuroborosAgent

    host = SimpleNamespace(env=SimpleNamespace(drive_root=root), _emit_live_log=lambda *_a, **_k: None,
                           _event_queue=None, _handle_task_scoped=body, _owner_message_admission_lock=threading.Lock(),
                           _busy=False, _accepting_owner_messages=False, _current_task_id=None, _current_chat_id=None,
                           _current_task_text="", _current_task_metadata={}, _last_activity_ts=None,
                           _incoming_messages=queue_mod.Queue())
    for name in ("handle_task", "_prepare_task_start", "_bind_turn", "_release_turn"):
        if hasattr(OuroborosAgent, name):
            setattr(host, name, getattr(OuroborosAgent, name).__get__(host))
    return host


def _hold(world, path):
    world.blockers.append(exclusive_hold(path))


def _release(world):
    for blocker in world.blockers:
        blocker.close()
        blocker.closed = True


def _accept_stop(world):
    """The owner's Stop as its door records it. Its acceptance reads the task's lineage, so while the
    store is held it is refused, honestly and with nothing recorded; it lands in a moment the store is
    free, and the turn's wait, which meets the held store again, still has to observe it."""
    from ouroboros.cancel_intents import CancelIntentLineageIndeterminate, active_intent, request_cancel

    with pytest.raises(CancelIntentLineageIndeterminate):
        request_cancel(world.root, "S", reason="owner stop", source="http_single")
    assert active_intent(world.root, "S") is None
    _release(world)
    request_cancel(world.root, "S", reason="owner stop", source="http_single")
    _hold(world, store.store_path(world.root))


@pytest.mark.parametrize("phase", ["room", "billing"])
def test_an_unstarted_message_waits_past_the_turn_window_and_then_runs_once(lane, monkeypatch, phase):
    """Contention met at the room read, or only at the billing read after it (the store goes busy once the
    room was read), outlasts the running turn's 900 s accounting window on the fake clock: the unstarted
    input keeps waiting at THAT read (no ``accounting_wait_expired``), visibly, and runs exactly once when
    the store is free; a read that already succeeded is not repeated, and nothing is held or reported."""
    from ouroboros import settings_integrity, usage_admission
    from ouroboros.task_results import write_task_result

    write_task_result(lane.root, "S", "requested", billing_group=BINDING)
    store.close_connections_under(lane.root)
    reads = {"room": 0, "billing": 0}

    def counted(stage, read):
        def call(*args, **kwargs):
            reads[stage] += 1
            result = read(*args, **kwargs)
            if stage == "room" and phase == "billing" and not lane.blockers:
                _hold(lane, store.store_path(lane.root))  # busy from the billing read on
            return result
        return call

    monkeypatch.setattr(settings_integrity, "actor_settings_snapshot",
                        counted("room", settings_integrity.actor_settings_snapshot))
    monkeypatch.setattr(usage_admission, "task_billing_fields", counted("billing", usage_admission.task_billing_fields))
    if phase == "room":
        _hold(lane, store.store_path(lane.root))

    def on_sleep(count):
        lane.clock[0] += 1001.0 if count == 1 else 1.0
        if count == 2:
            _release(lane)

    lane.on_sleep = on_sleep
    assert lane.run() is True, lane.errors
    assert lane.body == [lane.task] and lane.released == ["S"] and lane.errors == []
    assert len(lane.sleeps) == 2 and lane.clock[0] > 900.0
    assert reads == ({"room": 2, "billing": 1} if phase == "room" else {"room": 1, "billing": 2})
    assert not lane.workers.PENDING, "storage contention is never a saved-work hold"
    rows = [json.loads(line) for line in (lane.root / "logs" / "events.jsonl").read_text().splitlines()]
    waits = [row for row in rows if row.get("checkpoint_kind") == "usage_lock_wait"]
    assert [row["phase"] for row in waits] == ["entered", "ended"] and all(row["owner_visible"] for row in waits)


@pytest.mark.parametrize("mode", ["WAL", "PERSIST"])
@pytest.mark.parametrize("cached_metadata", [False, True])
def test_money_store_contention_in_the_billing_read_waits_instead_of_starting_unbound(
        lane, monkeypatch, mode, cached_metadata):
    """No pinned binding: the billing read reaches the money store. Its positive contention is typed
    (``BillingAuthorityBusy``) and waited out as a lock, sliced, instead of a 45 s blocking read that
    then let the turn start unbound; once free, the initial binding is pinned and the body runs once."""
    from ouroboros import usage_ledger
    from ouroboros.task_results import load_task_result
    from supervisor import state

    monkeypatch.setattr(state, "TOTAL_BUDGET_LIMIT", 0.0)  # isolate billing, after the outer money gate
    monkeypatch.setattr(store, "_journal_mode", lambda *_args: mode)
    usage_store.migrate_from_journal(lane.root)
    if cached_metadata:
        # Boot/revival can drop readiness while its metadata connection stays pooled.
        usage_store.migrate_from_journal(lane.root)
        assert str(lane.root / usage_store.STORE_REL) not in usage_store._READY
    else:
        store.close_connections_under(lane.root)
    _hold(lane, lane.root / usage_store.STORE_REL)
    lane.on_sleep = lambda count: count == 3 and _release(lane)
    assert lane.run() is True, lane.errors
    assert lane.body == [lane.task] and len(lane.sleeps) == 3
    assert load_task_result(lane.root, "S")["billing_group"]["billing_group_id"] == "S"
    busy = usage_ledger.BillingAuthorityBusy("S", rollback_confirmed=True)
    assert isinstance(busy, usage_ledger.BillingAuthorityUnavailable) and busy.kind == "unreadable"


@pytest.mark.parametrize("transaction_active", [False, True])
def test_metadata_contention_does_not_claim_an_active_transaction_was_rolled_back(
        tmp_path, monkeypatch, transaction_active):
    """The metadata read proves safe retry only when no transaction remains on its connection."""
    import contextlib
    import sqlite3

    from ouroboros.usage_ledger import UsageLockUnavailable

    def busy(_sql):
        raise sqlite3.OperationalError("database is locked")

    conn = SimpleNamespace(execute=busy, in_transaction=transaction_active)
    monkeypatch.setattr(store, "connection", lambda *_a, **_k: contextlib.nullcontext(conn))
    with pytest.raises(UsageLockUnavailable) as caught:
        usage_store._stored_meta(tmp_path / usage_store.STORE_REL, store.TIER_ENFORCED, 0.0)
    assert caught.value.reason == "contention"
    assert caught.value.rollback_confirmed is (not transaction_active)


@pytest.mark.parametrize("budget", [0.0, 20.0], ids=["uncapped", "configured"])
def test_first_money_read_waits_for_initialization_lock_before_body(lane, monkeypatch, budget):
    """A competing initializer owns the process lock; this caller has no transaction."""
    import threading

    from ouroboros.task_results import load_task_result
    from supervisor import state

    monkeypatch.setattr(state, "TOTAL_BUDGET_LIMIT", budget)
    usage_store.migrate_from_journal(lane.root)
    usage_store.forget(lane.root)
    lock = usage_store._process_lock(lane.root / usage_store.STORE_REL)
    held, release, released = threading.Event(), threading.Event(), threading.Event()

    def initializer():
        with lock:
            held.set()
            release.wait(60)  # longer than the existing 45-second initialization acquisition
        released.set()

    thread = threading.Thread(target=initializer)
    thread.start()
    assert held.wait(5)

    def unblock(count):
        assert count == 1 and not lane.body
        release.set()
        assert released.wait(5)

    lane.on_sleep = unblock
    try:
        assert lane.run() is True, lane.errors
        assert len(lane.sleeps) == 1 and lane.body == [lane.task] and not lane.errors
        assert load_task_result(lane.root, "S")["billing_group"]["billing_group_id"] == "S"
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive()


@pytest.mark.parametrize("control", ["stop", "deadline", "absolute_ceiling"])
def test_a_control_ends_the_wait_and_the_unstarted_turn_concludes_without_a_hold(lane, monkeypatch, control):
    """Stop, a reached deadline or the execution ceiling still end the wait. Nothing of the body ran, so
    the lane concludes Stop as stopped and bounds as failures: no pause record is read, no saved-work
    hold is invented and no second accounting window starts (the 2026-10-20 held messages)."""
    from ouroboros import budget_pause
    from ouroboros.llm_attempt import PhysicalDispatchInterrupted
    from ouroboros.task_results import write_task_result

    write_task_result(lane.root, "S", "requested", billing_group=BINDING)
    store.close_connections_under(lane.root)
    _hold(lane, store.store_path(lane.root))
    monkeypatch.setattr(budget_pause, "budget_pause_row", lambda *_a: pytest.fail("an unstarted turn has no pause"))

    def on_sleep(count):
        if count == 1:
            if control == "stop":
                _accept_stop(lane)
            elif control == "deadline":
                lane.task["deadline_at"] = "2000-01-01T00:00:00Z"
            else:
                lane.task["execution_timeout_sec"] = 0.000000001

    lane.on_sleep = on_sleep
    assert lane.run() is False
    assert lane.body == [] and lane.released == ["S"] and not lane.workers.PENDING
    assert len(lane.errors) == 1 and isinstance(lane.errors[0], PhysicalDispatchInterrupted)
    assert lane.errors[0].before_task_start is True
    assert lane.errors[0].control_reason == {"stop": "cancelled", "deadline": "deadline",
                                             "absolute_ceiling": "absolute_ceiling"}[control]
    assert len(lane.sleeps) == 1
    [(_args, reply)] = lane.replies
    assert reply["system_type"] == ("task_not_started" if control == "stop" else "task_error")
    assert reply["progress_meta"]["task_terminal_status"] == ("cancelled" if control == "stop" else "failed")
    if control == "stop":
        assert "stopped this chat turn before it started" in _args[1]
        assert "PhysicalDispatchInterrupted" not in _args[1]


@pytest.mark.parametrize("evidence", ["unreadable", "readable"])
def test_after_the_body_an_unreadable_pause_record_holds_as_unknown_never_as_saved_on_stop(lane, evidence):
    """The body ran, possibly with effects, then raised. A readable record without a pause of this attempt
    reports the failure; an unreadable one keeps the turn held for an explicit decision, classified as
    the unreadable authority it is (``work_stop``), with no replay grant and no new wait."""
    from ouroboros.task_results import write_task_result
    from ouroboros.work_stop import stop_presentation
    from supervisor.events_budget import budget_hold_fact

    write_task_result(lane.root, "S", "requested", billing_group=BINDING)
    store.close_connections_under(lane.root)
    error = OSError("after the body, possibly after effects")

    def body(received):
        lane.body.append(received)
        if evidence == "unreadable":  # the store goes busy before the lane reads the pause record
            store.close_connections_under(lane.root)
            _hold(lane, store.store_path(lane.root))
        raise error

    result = _run_with_body(lane, body)
    assert lane.body == [lane.task] and lane.released == ["S"] and lane.sleeps == []
    if evidence == "readable":
        assert result is False and lane.errors == [error] and not lane.workers.PENDING
        return
    assert result is True and lane.errors == [] and len(lane.workers.PENDING) == 1
    held = lane.workers.PENDING[0]
    assert budget_hold_fact(held)["reason"] == "authority"
    assert "_resume_selection" not in held and not held.get("_budget_pause_resume")
    assert stop_presentation(held)["reason"]["text"] == "After a technical problem"


def _run_with_body(lane, body):
    """The lane with BODY in place of the task body."""
    return lane.run(_host(lane.root, body))


def test_a_worker_s_immediate_stop_during_the_wait_leaves_the_terminal_to_the_supervisor(lane, monkeypatch):
    """A pooled task's preparation wait ended by the owner's immediate Stop returns no events, exactly as
    the body does: no worker-authored terminal before the supervisor's cancellation settles the task."""
    from ouroboros.agent import OuroborosAgent
    from ouroboros.task_results import load_task_result, write_task_result

    write_task_result(lane.root, "S", "running", billing_group=BINDING)
    store.close_connections_under(lane.root)
    _hold(lane, store.store_path(lane.root))
    monkeypatch.setattr("ouroboros.utils.in_worker_process", lambda: True)
    lane.on_sleep = lambda count: count == 1 and _accept_stop(lane)
    host = _host(lane.root, lambda _t: pytest.fail("the body never starts"))
    task = {**lane.task, "_is_direct_chat": False}
    assert OuroborosAgent.handle_task(host, task) == []
    _release(lane)
    assert load_task_result(lane.root, "S")["status"] == "running"


def test_a_busy_control_read_never_hides_a_local_stop_or_a_readable_pause(tmp_path, monkeypatch):
    """The strict probe at a send's dispatch window: a control store it cannot read is a gap only after
    every control it CAN read was checked: the local finalize control first, then the owner's Pause."""
    from ouroboros import llm_attempt, owner_pause
    from ouroboros.model_wait import task_model_wait_scope
    from ouroboros.owner_mailbox import KIND_FINALIZE_NOW, write_owner_message
    from ouroboros.task_results import write_task_result

    write_task_result(tmp_path, "S", "running")
    task = {"id": "S", "type": "task", "_attempt": 1}
    store.close_connections_under(tmp_path)
    blocker = exclusive_hold(store.store_path(tmp_path))
    try:
        with task_model_wait_scope(task=task, drive_root=tmp_path, event_queue=None, worker_slot_held=False) as owner:
            with pytest.raises(store.StoreBusy):
                owner.control_reason(strict_current=True)  # the legacy cancel latch is unreadable now
            monkeypatch.setattr(owner_pause, "scope_fence", lambda: {"state": "paused", "reason": "owner_pause"})
            with pytest.raises(llm_attempt.PhysicalDispatchInterrupted) as paused:
                llm_attempt.require_physical_dispatch_window()
            assert paused.value.control_reason == "owner_pause"
            monkeypatch.setattr(owner_pause, "scope_fence", lambda: {})
            assert write_owner_message(tmp_path, "deadline", "S", msg_id="grace-1", kind=KIND_FINALIZE_NOW)
            assert owner.control_reason(strict_current=True) == "finalize_requested"
    finally:
        blocker.close()
        store.close_connections_under(tmp_path)


def test_a_pause_pressed_during_the_wait_is_refused_honestly_and_writes_nothing(lane):
    """Pause acceptance itself needs the store. Pressed through the real door while the unstarted turn
    waits on the held store, it is answered with a retryable refusal (503) and no fence is recorded: no
    durable Pause is claimed that was never written. The turn keeps its own wait and runs once."""
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient

    from ouroboros.gateway.task_pause import api_task_pause
    from ouroboros.owner_pause import read_fence
    from ouroboros.task_results import write_task_result

    write_task_result(lane.root, "S", "requested", billing_group=BINDING)
    store.close_connections_under(lane.root)
    app = Starlette(routes=[Route("/api/tasks/{task_id}/pause", api_task_pause, methods=["POST"])])
    app.state.drive_root = lane.root
    answers = []
    _hold(lane, store.store_path(lane.root))

    def on_sleep(count):
        if count == 1:
            with TestClient(app) as client:
                answers.append(client.post("/api/tasks/S/pause", json={"request_id": "pause-1", "actor": "owner"}))
            _release(lane)

    lane.on_sleep = on_sleep
    assert lane.run() is True
    assert answers[0].status_code == 503, answers[0].text
    assert read_fence(lane.root, "S") == {}
    assert lane.body == [lane.task] and not lane.workers.PENDING


def _press(lane, route, endpoint, path, body=None):
    """One press of a real owner-control door (Starlette, in process; no port, no live data)."""
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient

    app = Starlette(routes=[Route(route, endpoint, methods=["POST"])])
    app.state.drive_root = lane.root
    with TestClient(app) as client:
        return client.post(path, **({"json": body} if body is not None else {}))


def test_an_ordinary_stop_accepted_during_the_wait_ends_the_turn_before_its_body(lane, monkeypatch):
    """The real Stop door, pressed while the unstarted turn waits on the held store. Refused while the store
    is held (its acceptance reads the store), with nothing recorded. Pressed in a moment the store is free,
    it is accepted against the LIVE waiting turn: the intent is durable before the answer and stays open
    (202, the cooperative stop armed), never a ``cancelled`` row over a turn nobody seemed to own. The
    turn's wait, meeting the held store again, observes it and ends with no body run; once the turn is
    gone the intent's own owner settles the row ``cancelled``."""
    from ouroboros.cancel_intents import active_intent
    from ouroboros.gateway.tasks import api_task_cancel
    from ouroboros.task_results import load_task_result, write_task_result
    from supervisor.task_lifecycle import drive_cancel_intent_scope

    monkeypatch.setenv("OUROBOROS_DIRECT_TURN_STOP_WAIT_SEC", "0.2")
    write_task_result(lane.root, "S", "requested", billing_group=BINDING, chat_id=7)
    store.close_connections_under(lane.root)
    _hold(lane, store.store_path(lane.root))
    answers, recorded = [], []

    def on_sleep(count):
        if count == 1:
            answers.append(_press(lane, "/api/tasks/{task_id}/cancel", api_task_cancel, "/api/tasks/S/cancel"))
            recorded.append(active_intent(lane.root, "S"))
            _release(lane)
            answers.append(_press(lane, "/api/tasks/{task_id}/cancel", api_task_cancel, "/api/tasks/S/cancel"))
            recorded.append(active_intent(lane.root, "S"))
            assert load_task_result(lane.root, "S")["status"] == "requested", "no cancelled row over a live turn"
            _hold(lane, store.store_path(lane.root))
        else:
            _release(lane)  # a turn that did not see the Stop would run its body now

    lane.on_sleep = on_sleep
    assert lane.run() is False
    assert answers[0].status_code == 503 and recorded[0] is None, answers[0].text
    assert answers[1].status_code == 202 and answers[1].json()["cancel_state"] == "pending", answers[1].text
    assert recorded[1] and recorded[1]["request_id"]
    assert lane.body == [] and len(lane.sleeps) == 1 and lane.released == ["S"]
    assert lane.errors[0].before_task_start is True and lane.errors[0].control_reason == "cancelled"
    assert not lane.workers.PENDING, "a Stop is never a paused or saved-on-stop hold"
    _release(lane)
    assert drive_cancel_intent_scope("S") == "cancelled"
    assert load_task_result(lane.root, "S")["status"] == "cancelled" and active_intent(lane.root, "S") is None


def test_a_pause_accepted_during_the_wait_parks_the_input_before_its_body(lane, monkeypatch):
    """A refused press writes nothing. An accepted Pause survives a later blocked read, parks the
    original input without starting its body, and ordinary Resume selects that same input once."""
    from ouroboros.gateway.task_pause import api_task_pause
    from ouroboros.owner_pause import fence_closed, read_fence
    from ouroboros.task_results import load_task_result, write_task_result
    from supervisor import queue
    from ouroboros import model_wait
    clock = model_wait.time
    monkeypatch.setattr(model_wait, "time", SimpleNamespace(monotonic=lambda: lane.clock[0], time=clock.time))
    lane.task["execution_timeout_sec"] = 2000.0
    from tests.test_budget_pause_holds import _idle_worker

    write_task_result(lane.root, "S", "requested", billing_group=BINDING, chat_id=7)
    usage_store.migrate_from_journal(lane.root)
    _hold(lane, store.store_path(lane.root))
    answers = []
    def on_sleep(count):
        lane.clock[0] += 7.0
        if count == 1:
            press = ("/api/tasks/{task_id}/pause", api_task_pause, "/api/tasks/S/pause")
            answers.append(_press(lane, *press, {"request_id": "held", "actor": "owner"}))
            _release(lane)
            answers.append(_press(lane, *press, {"request_id": "free", "actor": "owner"}))
            _hold(lane, store.store_path(lane.root))
        elif count == 2:
            assert lane.body == [] and not lane.workers.PENDING
            _release(lane)
        else:
            pytest.fail("accepted Pause must park once its authority is readable")

    lane.on_sleep = on_sleep
    assert lane.run() is True, lane.errors
    assert answers[0].status_code == 503 and answers[1].status_code in (200, 202)
    assert lane.body == [] and lane.errors == [] and len(lane.sleeps) == 2
    [parked] = lane.workers.PENDING
    assert parked["text"] == lane.task["text"] and parked["id"] == "S"
    assert parked["_budget_pause"]["before_task_start"] is True
    assert parked["_execution_handoff_sec"] == 14.0
    from supervisor.restart_retention import retained_pending
    assert retained_pending(parked), "Restart preserves the closed Pause; it grants no execution"
    assert "checkpoint" not in parked["_budget_pause"]
    assert not load_task_result(lane.root, "S").get("budget_pause")
    assert fence_closed(read_fence(lane.root, "S"))
    sent = []
    _idle_worker(lane.workers, sent)
    lane.workers.assign_tasks()
    assert sent == [], "a closed Pause cannot auto-dispatch"
    granted = queue.resume_budget_paused_task("S")
    assert granted["ok"] and granted["owner_pause_released"], granted
    lane.workers.assign_tasks()
    assert [row["id"] for row in sent] == ["S"] and not lane.workers.PENDING
    assert sent[0]["_execution_consumed_sec"] == 14.0, "Resume cannot reset the execution ceiling"
    def body(task):
        assert model_wait.current_model_wait().executed_seconds() >= 14.0
        lane.body.append(task)
        return []
    _host(lane.root, body).handle_task(sent[0])
    lane.workers.assign_tasks()
    assert len(lane.body) == len(sent) == 1


def test_a_pause_after_prestart_storage_wait_parks_the_real_loop_and_its_resume_runs_once(
        lane, monkeypatch, provider_catalog_offline):
    """After storage recovers, Pause during context preparation parks the real loop's exact source.
    Resume, assignment, a pre-body billing failure and death recovery preserve that same source/grant;
    the next Resume executes it once. Provider, context and OS-worker facts remain synthetic.
    The separate pre-body test covers an accepted Pause while preparation is still waiting.
    """
    from dataclasses import replace
    from functools import partial
    import queue as stdqueue

    from ouroboros import agent as agent_module, agent_task_pipeline, budget_pause, settings_integrity, utils
    from ouroboros.agent import Env, OuroborosAgent
    from ouroboros.artifacts import read_actor_source_bytes
    from ouroboros.gateway.task_pause import api_task_pause
    from ouroboros.owner_pause import read_fence
    from ouroboros.owner_wait import direct_owner_wait, worker_owner_wait
    from ouroboros.task_results import load_task_result, write_task_result
    from ouroboros.work_stop import stop_presentation
    from supervisor import queue, worker_health, worker_process
    from tests.test_budget_pause_holds import _idle_worker
    from tests.test_context_fit_v664 import _plan

    write_task_result(lane.root, "S", "requested", billing_group=BINDING, chat_id=7)
    usage_store.migrate_from_journal(lane.root)
    store.close_connections_under(lane.root)
    repo = lane.root.parent / "repo"
    repo.mkdir()
    monkeypatch.setattr(lane.workers, "REPO_DIR", repo)
    sends, reflections = [], []

    class Provider:
        def default_model(self):
            return "openai-compatible::test"

        def chat(self, **kwargs):
            sends.append(json.loads(json.dumps(kwargs.get("messages"), default=str)))
            return {"content": "Resumed answer.", "tool_calls": []}, {"cost": 0.0}

    def actor(*, pooled=False):
        if pooled:
            agent = OuroborosAgent(Env(repo_dir=repo, drive_root=lane.root))
            # worker_main binds this continuation owner before accepting a task.
            agent.owner_wait_callback = partial(worker_owner_wait, 0, stdqueue.Queue(), lane.workers.get_event_q())
        else:
            # The real direct factory supplies direct_owner_wait; construction of
            # OuroborosAgent alone has no continuation owner and must hold warm.
            agent = lane.workers._get_chat_agent()
            assert agent.owner_wait_callback is direct_owner_wait
        agent.llm = Provider()
        return agent

    def messages(*, task, ctx, **_kwargs):
        assert callable(ctx.owner_wait_callback), "the native lane must bind its continuation owner"
        # Like the real builder, publish the captured plan on the context: the
        # real cold Resume rebinds it after restoring the saved transcript.
        if not answers:
            answers.append(_press(lane, "/api/tasks/{task_id}/pause", api_task_pause, "/api/tasks/S/pause",
                                  {"request_id": "context-pause", "actor": "owner"}))
        ctx.context_fit_plan = replace(_plan(), model=Provider().default_model(), provider="openai-compatible",
                                       user_content_json=json.dumps(task["text"]))
        return ctx.context_fit_plan.messages_for("max"), {"budget_remaining": 5.0}

    monkeypatch.setattr(OuroborosAgent, "_log_worker_boot_once", lambda self: None)
    monkeypatch.setattr(agent_module, "build_llm_messages", messages)
    monkeypatch.setattr(agent_task_pipeline, "_run_post_task_processing_async",
                        lambda _env, task, *_a, **_k: reflections.append(task["id"]))
    monkeypatch.setattr("ouroboros.claudexor_daemon.ensure_owned_gateway",
                        lambda *_a, **_k: pytest.fail("no daemon is reached"))
    answers = []

    def on_sleep(count):
        _release(lane)

    lane.on_sleep = on_sleep
    _hold(lane, store.store_path(lane.root))
    assert lane.run(actor()) is True, lane.errors
    assert answers[0].status_code in (200, 202), [a.text for a in answers]
    assert sends == [] and lane.errors == [] and lane.released == ["S"]
    pause = budget_pause.budget_pause_row(lane.root, "S")
    assert pause["rail"] == budget_pause.RAIL_OWNER_PAUSE and pause["source_ref"]
    assert pause["state"] == budget_pause.STATE_PAUSED
    assert pause["owner_fence_id"] == answers[0].json()["fence_id"]
    saved_bytes = read_actor_source_bytes(lane.root, "S", pause["source_ref"])
    saved = json.loads(saved_bytes)
    assert saved["task_id"] == "S" and saved["task_attempt"] == lane.task["_attempt"]
    assert json.dumps(saved["messages"]).count("keep this input") == 1
    [parked] = lane.workers.PENDING
    assert parked["_budget_pause"]["checkpoint"]["pause_id"] == pause["pause_id"]
    owner_fence = read_fence(lane.root, "S")
    assert owner_fence["fence_id"] == pause["owner_fence_id"]
    fences = {"owner": owner_fence, "budget": queue.BUDGET_ROOT_FENCES}
    assert stop_presentation(parked, fences)["reason"]["text"] == "You paused it"

    def resume_to_worker():
        granted = queue.resume_budget_paused_task("S")
        assert granted["ok"] and granted["exact_continuation"], granted
        sent = []
        _idle_worker(lane.workers, sent)
        lane.workers.assign_tasks()
        assert [row["id"] for row in sent] == ["S"] and "S" in lane.workers.RUNNING
        assert sent[0]["_budget_pause_resume"]["pause_id"] == pause["pause_id"]
        return sent

    first = resume_to_worker()
    first_grant_id = first[0]["_budget_pause_resume"]["grant_id"]
    from ouroboros import model_wait, usage_admission

    billing_read = usage_admission.task_billing_fields
    billing_unreadable, billing_busy = False, []
    resumed_reads = {"room": 0, "billing": 0}
    resumed_trace, resumed_sleeps = [], []
    sleep_start, event_start = len(lane.sleeps), len(lane.events)

    def observed(stage, read):
        def call(*args, **kwargs):
            if stage in resumed_reads:
                resumed_reads[stage] += 1
            resumed_trace.append({"stage": stage, "phase": "entered"})
            try:
                result = read(*args, **kwargs)
            except Exception as exc:
                resumed_trace.append({"stage": stage, "phase": "raised", "error": repr(exc),
                                      "reason": getattr(exc, "reason", None),
                                      "control_reason": getattr(exc, "control_reason", None)})
                raise
            resumed_trace.append({"stage": stage, "phase": "returned",
                                  "result": result if stage == "control" else type(result).__name__})
            return result
        return call

    def resumed_diagnostic(returned=None):
        def event_facts(events):
            if events is None:
                return None
            return [{key: row[key] for key in ("type", "task_id", "status", "reason_code", "data")
                     if key in row} for row in events]

        return json.dumps({
            "returned": event_facts(returned), "resumed_reads": resumed_reads, "billing_busy": billing_busy,
            "trace": resumed_trace,
            "on_sleep_counts": resumed_sleeps, "new_sleeps": len(lane.sleeps) - sleep_start,
            "events": event_facts(lane.events[event_start:]),
            "pending_events": event_facts(runner._pending_events) if returned is None else None,
            "provider_calls": len(sends), "reflections": reflections,
        }, default=str, sort_keys=True)

    def read_resumed_billing(*args, **kwargs):
        if billing_unreadable:
            raise PermissionError("billing preparation unreadable")
        try:
            return billing_read(*args, **kwargs)
        except store.StoreBusy as exc:
            billing_busy.append(exc.reason)
            raise

    def on_resumed_sleep(_count):
        nonlocal billing_unreadable
        resumed_sleeps.append(_count)
        resumed_trace.append({"stage": "sleep", "count": _count, "billing_unreadable": billing_unreadable})
        assert resumed_reads == {"room": 1, "billing": 1} and billing_busy == ["contention"], resumed_diagnostic()
        _release(lane)
        billing_unreadable = True

    # Fail billing's own retry after its real StoreBusy and sleep. The room
    # preparation already succeeded and must not run again.
    lane.on_sleep = on_resumed_sleep
    _hold(lane, store.store_path(lane.root))
    runner = actor(pooled=True)
    with monkeypatch.context() as worker_context:
        # Only the synthetic worker is a worker process. Resume, assignment and
        # death recovery below still run as the supervisor owning the live queue.
        worker_context.setenv(utils.WORKER_PROCESS_ENV, "1")
        worker_context.setattr(settings_integrity, "actor_settings_snapshot",
                               observed("room", settings_integrity.actor_settings_snapshot))
        worker_context.setattr(usage_admission, "task_billing_fields",
                               observed("billing", read_resumed_billing))
        worker_context.setattr(model_wait.TaskModelWait, "control_reason",
                               observed("control", model_wait.TaskModelWait.control_reason))
        worker_context.setattr(runner, "_handle_task_scoped", observed("body", runner._handle_task_scoped))
        with pytest.raises(PermissionError, match="billing preparation unreadable") as failed:
            returned = runner.handle_task(first[0])
            pytest.fail("Resumed preparation returned instead of raising PermissionError: "
                        + resumed_diagnostic(returned))
        assert resumed_reads == {"room": 1, "billing": 2} and billing_busy == ["contention"], resumed_diagnostic()
        assert resumed_sleeps == [sleep_start + 1], resumed_diagnostic()
        assert not any(row["stage"] == "body" for row in resumed_trace) and reflections == [], resumed_diagnostic()
        assert failed.value.before_task_start is True and sends == [], resumed_diagnostic()
        worker_process._record_task_exception(runner, first[0], failed.value, lane.root)
    assert load_task_result(lane.root, "S")["status"] not in ("failed", "cancelled", "completed")
    # Only OS liveness is synthetic: recovery must validate this worker,
    # RUNNING row and attempt through its real current-job guard.
    worker = lane.workers.WORKERS[0]
    worker.proc.is_alive = lambda: False
    monkeypatch.setattr(lane.workers, "send_with_budget", lambda *_a, **_k: None)
    worker_health._recover_crashed_task_without_terminal({
        "worker": worker, "worker_id": 0, "task_id": "S", "task": first[0], "attempt": first[0]["_attempt"],
        "meta": lane.workers.RUNNING["S"], "exitcode": 1, "drive_root": str(lane.root)}, queue)
    [again] = lane.workers.PENDING
    assert "S" not in lane.workers.RUNNING and "_budget_pause_resume" not in again
    assert again["_budget_pause"]["checkpoint"]["pause_id"] == pause["pause_id"]
    kept = budget_pause.budget_pause_row(lane.root, "S")
    assert kept["source_ref"] == pause["source_ref"] and not (kept.get("grant") or {}).get("consumed_at")
    assert read_actor_source_bytes(lane.root, "S", kept["source_ref"]) == saved_bytes

    second = resume_to_worker()
    assert second[0]["_budget_pause_resume"]["grant_id"] != first_grant_id
    with monkeypatch.context() as worker_context:
        worker_context.setenv(utils.WORKER_PROCESS_ENV, "1")
        events = actor(pooled=True).handle_task(second[0])
    assert len(sends) == 1, sends
    assert json.dumps(sends[0]).count("keep this input") == 1
    assert budget_pause.budget_pause_row(lane.root, "S")["grant"]["consumed_at"]
    result = load_task_result(lane.root, "S")
    assert result["status"] == "completed" and result["result"] == "Resumed answer.", result
    assert [event.get("type") for event in events].count("task_done") == 1
    assert not lane.workers.PENDING and reflections == ["S"]


def test_a_start_that_resumes_saved_work_keeps_it_when_its_wait_ends_before_the_body(lane):
    """Not starting THIS attempt is not having nothing saved. A start that resumes a continuation
    (``governing_continuation``) whose preparation wait a bound ends leaves the decision to its pause
    record, as after any body; the record is in the held store, so the turn is held for an explicit
    decision as unreadable authority, never concluded as an error that drops what it resumes. The direct
    lane admits only fresh turns today (a resumed direct pause continues on a pooled worker, the case
    above); this keeps the lane's own recovery honest if one ever reaches it."""
    from ouroboros.task_results import write_task_result
    from ouroboros.work_stop import stop_presentation
    from supervisor.events_budget import budget_hold_fact

    write_task_result(lane.root, "S", "requested", billing_group=BINDING)
    store.close_connections_under(lane.root)
    lane.task["_working_recovery"] = {"from_attempt": 1, "started_at": 1.0}
    _hold(lane, store.store_path(lane.root))
    lane.on_sleep = lambda count: count == 1 and lane.task.update(deadline_at="2000-01-01T00:00:00Z")
    assert lane.run() is True
    assert lane.body == [] and lane.errors == [] and lane.released == ["S"]
    [held] = lane.workers.PENDING
    assert budget_hold_fact(held)["reason"] == "authority" and "_budget_pause_resume" not in held
    assert stop_presentation(held)["reason"]["text"] == "After a technical problem"


@pytest.mark.parametrize("mode", ["PERSIST", "WAL"])
def test_outer_chat_entry_waits_on_global_money_and_runs_the_original_input_once(lane, monkeypatch, mode):
    """The public chat lane registers the input before its strict global-budget read meets a lock;
    the real agent keeps that exact turn visible past 900 s and invokes its body once after recovery."""
    from supervisor import worker_chat_lane
    from supervisor.active_activity import get_direct_activity_registry

    monkeypatch.setattr(store, "_journal_mode", lambda *_: mode)
    usage_store.migrate_from_journal(lane.root)
    host = _host(lane.root, lambda task: lane.body.append(task) or [])
    monkeypatch.setattr(lane.workers, "_get_chat_agent", lambda: host)
    monkeypatch.setattr(worker_chat_lane, "uuid", SimpleNamespace(uuid4=lambda: SimpleNamespace(hex="S")))
    monkeypatch.setattr(worker_chat_lane, "owner_conversation_admitted", lambda *_a: True)
    _hold(lane, lane.root / usage_store.STORE_REL)
    def on_sleep(count):
        assert get_direct_activity_registry().get("S") is not None
        assert host._busy and host._current_task_id == "S" and lane.body == []
        lane.clock[0] += 1001.0
        if count == 2:
            _release(lane)
    lane.on_sleep = on_sleep
    worker_chat_lane.handle_chat_direct(7, "keep this input", task_metadata={"client_message_id": "m-1"})
    assert not lane.errors and not lane.replies and len(lane.sleeps) == 2
    assert len(lane.body) == 1 and lane.body[0]["text"] == "keep this input"
    assert lane.body[0]["id"] == "S" and get_direct_activity_registry().get("S") is None


@pytest.mark.parametrize("remaining", [0.0, 2.0])
def test_global_money_is_rechecked_after_prestart_contention(lane, remaining):
    """The strict gate rereads settled usage after contention, before the waiting turn's body."""
    from ouroboros import usage_accounting as ua
    from supervisor import state

    usage_store.migrate_from_journal(lane.root)
    reservation = ua.reserve_attempt(ua.AttemptRequest(
        model="m", provider="openai", reservation_usd=10.0, global_limit_usd=10.0,
        drive_root=lane.root, task_id="other", root_task_id="other", source="test"))
    ua.mark_dispatched(reservation)
    assert state.budget_remaining({}, strict=True) == pytest.approx(10.0)
    _hold(lane, lane.root / usage_store.STORE_REL)
    def on_sleep(count):
        assert count == 1 and lane.body == []
        _release(lane)
        # The retry has already captured budget_remaining; change the temporary ledger
        # it reads by settling another task, keeping the real strict gate throughout.
        ua.settle_attempt(reservation, cost_usd=10.0 - remaining, cost_final=True)
    lane.on_sleep = on_sleep
    assert lane.run() is (remaining > 0)
    assert len(lane.sleeps) == 1 and lane.released == ["S"] and not lane.workers.PENDING
    assert lane.body == ([lane.task] if remaining > 0 else [])
    assert state.budget_remaining({}, strict=True) == pytest.approx(remaining)
    if remaining == 0:
        assert len(lane.errors) == len(lane.replies) == 1
        assert lane.errors[0].control_reason == "budget_exhausted"
        assert lane.replies[0][1]["system_type"] == "task_not_started"
        assert "Budget exhausted" in lane.replies[0][0][1]
    else:
        assert lane.errors == [] and lane.replies == []


@pytest.mark.parametrize("rollback_confirmed", [False, True])
def test_unconfirmed_money_outcome_never_retries_prestart(lane, monkeypatch, rollback_confirmed):
    from ouroboros.usage_ledger import UsageLockUnavailable
    from supervisor import state

    calls = []
    def read(*_args, **_kwargs):
        calls.append(1)
        if len(calls) == 1:
            error = UsageLockUnavailable("held", reason="contention")
            error.rollback_confirmed = rollback_confirmed
            raise error
        return 1.0
    monkeypatch.setattr(state, "budget_remaining", read)
    assert lane.run() is rollback_confirmed
    assert len(calls) == (2 if rollback_confirmed else 1)
    assert len(lane.sleeps) == int(rollback_confirmed)
    assert len(lane.body) == int(rollback_confirmed)


def test_prestart_pause_proof_cannot_requeue_a_prior_attempt_or_saved_work(lane):
    """The real queue consumer rejects a forged/stale prestart park before moving RUNNING."""
    from supervisor import events_budget, queue
    from ouroboros.task_results import write_task_result

    write_task_result(lane.root, "S", "running", billing_group=BINDING)
    for extra in ({"_attempt": 2}, {"_budget_pause_resume": {"pause_id": "saved"}}):
        task = {**lane.task, **extra}
        lane.workers.RUNNING["S"] = {"task": task, "attempt": task.get("_attempt", 1)}
        event = {"type": "budget_pause", "task_id": "S", "task_attempt": 1, "resource_limit": {
            "before_task_start": True, "task_attempt": 1, "replay_safe": True, "physical_calls": 0,
            "cause": "owner_pause", "fence_id": "stale", "root_task_id": "S", "scope": "root",
            "execution_consumed_sec": 0.0}}
        with pytest.raises(ValueError, match="fresh attempt"):
            events_budget._handle_budget_pause(event, SimpleNamespace(
                RUNNING=lane.workers.RUNNING, PENDING=lane.workers.PENDING, DRIVE_ROOT=lane.root))
        assert lane.workers.RUNNING["S"]["task"] == task and not lane.workers.PENDING
        assert not queue.BUDGET_ROOT_FENCES


def test_stop_after_accepted_prestart_pause_still_wins_while_storage_is_busy(lane, monkeypatch):
    from ouroboros.gateway.task_pause import api_task_pause
    from ouroboros.gateway.tasks import api_task_cancel
    from ouroboros.task_results import write_task_result

    monkeypatch.setenv("OUROBOROS_DIRECT_TURN_STOP_WAIT_SEC", "0.2")
    write_task_result(lane.root, "S", "requested", billing_group=BINDING, chat_id=7)
    _hold(lane, store.store_path(lane.root))
    answers = []
    def on_sleep(count):
        assert count == 1
        _release(lane)
        answers.append(_press(lane, "/api/tasks/{task_id}/pause", api_task_pause, "/api/tasks/S/pause",
                              {"request_id": "pause-before-stop", "actor": "owner"}))
        answers.append(_press(lane, "/api/tasks/{task_id}/cancel", api_task_cancel, "/api/tasks/S/cancel"))
        _hold(lane, store.store_path(lane.root))
    lane.on_sleep = on_sleep
    assert lane.run() is False
    assert answers[0].status_code in (200, 202) and answers[1].status_code == 202
    assert lane.body == [] and not lane.workers.PENDING
    assert lane.errors[0].control_reason == "cancelled"
    assert lane.replies[0][1]["progress_meta"]["task_terminal_status"] == "cancelled"


@pytest.mark.parametrize("control", ["stop", "pause", "deadline", "none"])
def test_control_inside_successful_strict_budget_read_prevents_body(lane, monkeypatch, control):
    """SQLite waits inside a successful read: no exception reaches retry_presend.

    A real control door lands while the money file is locked. The read then succeeds
    once, and the final preparation probe must still observe that accepted control.
    """
    import threading
    import time

    from ouroboros import _usage_wait
    from ouroboros.gateway.task_pause import api_task_pause
    from ouroboros.gateway.tasks import api_task_cancel
    from ouroboros.task_results import write_task_result
    from ouroboros.usage_ledger import USAGE_LOCK_TIMEOUT_SEC
    from supervisor import state

    write_task_result(lane.root, "S", "requested", billing_group=BINDING, chat_id=7)
    usage_store.migrate_from_journal(lane.root)
    # Leave time for the real HTTP consumer to accept the control inside SQLite's
    # bounded acquisition, without converting this into the already-covered retry:
    # the read keeps production's wait (``short_waits`` would end it first).
    monkeypatch.setattr(_usage_wait, "ACQUISITION_SLICE_SEC", 2.0)
    monkeypatch.setattr(usage_store, "USAGE_LOCK_TIMEOUT_SEC", USAGE_LOCK_TIMEOUT_SEC)
    monkeypatch.setenv("OUROBOROS_DIRECT_TURN_STOP_WAIT_SEC", "0.01")
    entered = threading.Event()
    reads, answers, failures = [], [], []
    strict_read = state.budget_remaining

    def read(*args, **kwargs):
        reads.append("entered")
        entered.set()
        result = strict_read(*args, **kwargs)
        reads.append("returned")
        return result

    monkeypatch.setattr(state, "budget_remaining", read)
    _hold(lane, lane.root / usage_store.STORE_REL)

    def control_then_unlock():
        try:
            assert entered.wait(5)
            time.sleep(0.03)
            assert reads == ["entered"] and not lane.body
            if control == "stop":
                answers.append(_press(lane, "/api/tasks/{task_id}/cancel", api_task_cancel, "/api/tasks/S/cancel"))
            elif control == "pause":
                answers.append(_press(lane, "/api/tasks/{task_id}/pause", api_task_pause, "/api/tasks/S/pause",
                                      {"request_id": "inside-read", "actor": "owner"}))
            elif control == "deadline":
                lane.task["deadline_at"] = "2000-01-01T00:00:00Z"
            assert reads == ["entered"] and not lane.body
        except BaseException as exc:
            failures.append(exc)
        finally:
            _release(lane)

    controller = threading.Thread(target=control_then_unlock)
    controller.start()
    try:
        result = lane.run()
    finally:
        controller.join(5)
    assert not controller.is_alive() and not failures, failures
    assert all(answer.status_code in (200, 202) for answer in answers)
    assert reads == ["entered", "returned"] and not lane.sleeps
    assert len(lane.body) == int(control == "none")
    if control == "pause":
        assert result and lane.workers.PENDING[0]["text"] == lane.task["text"]
        assert lane.workers.PENDING[0]["_budget_pause"]["before_task_start"]
    elif control != "none":
        assert not result and not lane.workers.PENDING
        assert lane.errors[0].before_task_start
        assert lane.errors[0].control_reason == ("cancelled" if control == "stop" else "deadline")
    else:
        assert result and lane.body == [lane.task] and not lane.errors


@pytest.mark.parametrize("restart", [False, True], ids=["resume", "restart_resume"])
def test_inline_image_prestart_pause_survives_real_queue_consumers(lane, monkeypatch, restart):
    """Public inline-image ingress, accepted Pause, snapshot restore, Resume and
    assignment feed the original pixels to the actual model-content builder.
    """
    import base64
    import io
    from PIL import Image

    from ouroboros.context import build_user_content
    from ouroboros.gateway.task_pause import api_task_pause
    from ouroboros.task_results import write_task_result
    from supervisor import queue, worker_chat_lane
    from tests.test_budget_pause_holds import _idle_worker

    stream = io.BytesIO()
    Image.new("RGB", (3, 2), (12, 34, 56)).save(stream, "PNG")
    pixels = stream.getvalue()
    encoded = base64.b64encode(pixels).decode("ascii")
    original = {"text": "keep this input", "image_base64": encoded, "image_mime": "image/png",
                "image_caption": "original caption"}
    expected = build_user_content(original)
    metadata = {"client_message_id": "m-image", "origin_message_ref": {"kind": "test", "id": "m-image"},
                "origin_message_text": original["text"]}
    host = _host(lane.root, lambda task: lane.body.append(task) or [])
    monkeypatch.setattr(lane.workers, "_get_chat_agent", lambda: host)
    monkeypatch.setattr(worker_chat_lane, "uuid", SimpleNamespace(uuid4=lambda: SimpleNamespace(hex="S")))
    monkeypatch.setattr(worker_chat_lane, "owner_conversation_admitted", lambda *_a: True)
    write_task_result(lane.root, "S", "requested", billing_group=BINDING, chat_id=7)
    usage_store.migrate_from_journal(lane.root)
    _hold(lane, store.store_path(lane.root))
    answers = []

    def pause(count):
        assert count == 1
        _release(lane)
        answers.append(_press(lane, "/api/tasks/{task_id}/pause", api_task_pause, "/api/tasks/S/pause",
                              {"request_id": "inline-image", "actor": "owner"}))

    lane.on_sleep = pause
    worker_chat_lane.handle_chat_direct(7, original["text"],
        image_data=(encoded, original["image_mime"], original["image_caption"]), task_metadata=metadata)
    assert answers[0].status_code in (200, 202) and not lane.body and not lane.errors
    [parked] = lane.workers.PENDING
    assert "image_base64" not in parked and encoded not in json.dumps(lane.events)
    assert queue.persist_queue_snapshot(reason="inline-image-test")
    snapshot = queue.QUEUE_SNAPSHOT_PATH.read_text()
    assert encoded not in snapshot
    if restart:
        lane.workers.PENDING.clear()
        queue.BUDGET_ROOT_FENCES.clear()
        store.close_connections_under(lane.root)
        assert queue.restore_pending_from_snapshot() == 1
    sent = []
    _idle_worker(lane.workers, sent)
    lane.workers.assign_tasks()
    assert not sent
    granted = queue.resume_budget_paused_task("S")
    assert granted["ok"] and granted["owner_pause_released"], granted
    lane.workers.assign_tasks()
    [resumed] = sent
    assert resumed["id"] == "S" and resumed["text"] == original["text"]
    assert resumed["origin_message_ref"] == metadata["origin_message_ref"]
    assert resumed["origin_message_text"] == original["text"]
    observed = []
    _host(lane.root, lambda task: observed.append(build_user_content(task)) or []).handle_task(resumed)
    assert len(observed) == 1
    assert observed[0][0] == expected[0]
    actual_images = [block["image_url"] for block in observed[0] if block["type"] == "image_url"]
    expected_images = [block["image_url"] for block in expected if block["type"] == "image_url"]
    assert actual_images == expected_images and len(actual_images) == 1
    assert base64.b64decode(actual_images[0]["url"].split(",", 1)[1]) == pixels
    lane.workers.assign_tasks()
    assert len(sent) == 1 and not lane.workers.PENDING
