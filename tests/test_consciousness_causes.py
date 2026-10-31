"""Mixed carriers share controls and money, but only one actor owns the clock."""

import itertools
from types import SimpleNamespace

import pytest

from ouroboros import consciousness as clock_module
from ouroboros.consciousness import BackgroundConsciousness, LAST_WAKE_STATE_KEY, OBSERVATION_STATE_KEY
from ouroboros.consciousness_wake import wake_task_metadata
from supervisor.active_activity import get_direct_activity_registry
from supervisor.consciousness_carriers import finish_pooled_wake, queued_wakes
from tests._budget_pause_exact_helpers import _install_queue
from tests import test_consciousness as clocks
from tests.test_consciousness import T0, FLOOR, DEFAULT, _launched

clock = clocks.clock

OWN = wake_task_metadata("act", "heartbeat")
FOCUS = {**OWN, "wake_cause": "focus", "consciousness_cause": "place:mail"}
STARTED = {"initiator": "consciousness", "work_kind": "consciousness", "consciousness_cause": "started"}
OWNER = {"work_kind": "owner"}


def _task(task_id, metadata, **extra):
    return {"id": task_id, "type": "task", "text": task_id, "metadata": dict(metadata), **extra}


def _direct(monkeypatch, turns):
    from supervisor import workers
    registry = get_direct_activity_registry()
    for task_id in turns:
        registry.register(task_id, 7)
    monkeypatch.setattr(workers, "direct_chat_turn", turns.get)


@pytest.mark.parametrize("order", list(itertools.permutations(["own", "focus", "child"])))
def test_only_admitted_clock_actor_finishes_once_in_any_order(clock, monkeypatch, order):
    _launched(clock)
    boundary = clock.store[OBSERVATION_STATE_KEY]
    before = clock.clock.next_wake_at
    monkeypatch.setattr(clock_module.time, "time", lambda: T0 + 5000)
    for cause in order:
        task_id, metadata = {"own": ("wake0001", OWN), "focus": ("focus1", FOCUS),
                             "child": ("kid1", STARTED)}[cause]
        clock.clock._wake_finished(task_id, False, metadata=metadata)
        assert clock.clock.next_wake_at == (T0 + 5000 + DEFAULT * 2 if "own" in order[:order.index(cause) + 1] else before)
        assert clock.store[OBSERVATION_STATE_KEY] == boundary
    settled = clock.clock.next_wake_at
    clock.clock._wake_finished("wake0001", False, metadata=OWN)
    clock.clock._wake_finished("wake0001", True, metadata=FOCUS)  # id alone is insufficient
    assert clock.clock.next_wake_at == settled and clock.clock._backoff == 2
    assert clock.store[LAST_WAKE_STATE_KEY] == T0 + 5000


def test_handoff_and_restart_finish_the_same_logical_wake_once(clock, monkeypatch):
    from supervisor import queue
    _launched(clock)
    monkeypatch.setattr(queue, "PENDING", [_task("wake0001", OWN)])
    monkeypatch.setattr(queue, "RUNNING", {})
    deadline = clock.clock.next_wake_at
    clock.clock._wake_finished("wake0001", True)
    assert clock.clock.next_wake_at == deadline and LAST_WAKE_STATE_KEY not in clock.store
    assert clock.clock._in_flight == "" and clock.clock.live_wake() == "wake0001"
    restarted = BackgroundConsciousness(clock.root, clock.root / "repo", lambda: 7, now=T0 + 1000)
    monkeypatch.setattr(clock_module.time, "time", lambda: T0 + 5000)
    result = {"execution_owner": {"kind": "pooled"}, "metadata": OWN, "status": "completed"}
    ctx = SimpleNamespace(consciousness=restarted)
    finish_pooled_wake(ctx, "wake0001", {**result, "metadata": FOCUS})
    assert LAST_WAKE_STATE_KEY not in clock.store
    monkeypatch.setattr(queue, "RUNNING", {"wake0001": {"task": queue.PENDING.pop()}})
    finish_pooled_wake(ctx, "wake0001", result)
    assert restarted.live_wake() == "wake0001"
    queue.RUNNING.clear()
    assert restarted.live_wake() == ""
    assert restarted.next_wake_at == T0 + 5000 + DEFAULT
    finish_pooled_wake(ctx, "wake0001", {**result, "status": "failed"})
    assert restarted.next_wake_at == T0 + 5000 + DEFAULT
    again = BackgroundConsciousness(clock.root, clock.root / "repo", lambda: 7, now=T0 + 5000)
    again._wake_finished("wake0001", False, metadata=OWN, pooled=True)
    assert again.next_wake_at == T0 + 5000 + DEFAULT


def test_durable_handoff_waits_for_pool_terminal_even_before_queue_restore(clock, monkeypatch):
    from ouroboros.task_results import write_task_result
    from supervisor import queue
    _launched(clock)
    monkeypatch.setattr(queue, "PENDING", [])
    monkeypatch.setattr(queue, "RUNNING", {})
    write_task_result(clock.root, "wake0001", "scheduled", execution_owner={"kind": "pooled"}, metadata=OWN)
    clock.clock._wake_finished("wake0001", True)
    assert LAST_WAKE_STATE_KEY not in clock.store
    monkeypatch.setattr(clock_module.time, "time", lambda: T0 + 5000)
    finish_pooled_wake(SimpleNamespace(consciousness=clock.clock), "wake0001",
                       {"status": "failed", "execution_owner": {"kind": "pooled"}, "metadata": OWN})
    assert clock.clock.next_wake_at == T0 + 5000 + DEFAULT * 2


def test_completion_requires_the_matching_actor_id_and_own_clock_cause(clock, monkeypatch):
    _launched(clock)
    deadline = clock.clock.next_wake_at
    monkeypatch.setattr(clock_module.time, "time", lambda: T0 + 5000)
    clock.clock._wake_finished("another-own-wake", False, metadata=OWN, pooled=True)
    assert clock.clock.next_wake_at == deadline and LAST_WAKE_STATE_KEY not in clock.store
    clock.clock._wake_finished("wake0001", False, metadata=FOCUS, pooled=True)
    assert clock.clock.next_wake_at == deadline and LAST_WAKE_STATE_KEY not in clock.store
    clock.clock._wake_finished("wake0001", False, metadata=OWN, pooled=True)
    assert clock.clock.next_wake_at == T0 + 5000 + DEFAULT * 2


def test_pool_callback_filters_cause_before_calling_the_clock():
    calls = []
    ctx = SimpleNamespace(consciousness=SimpleNamespace(_wake_finished=lambda *args, **kwargs: calls.append((args, kwargs))))
    pooled = {"execution_owner": {"kind": "pooled"}, "status": "completed"}
    for metadata in (FOCUS, STARTED, OWNER):
        finish_pooled_wake(ctx, "other", {**pooled, "metadata": metadata})
    finish_pooled_wake(ctx, "own", {**pooled, "metadata": OWN, "execution_owner": {"kind": "direct"}})
    assert calls == []
    finish_pooled_wake(ctx, "own", {**pooled, "metadata": OWN})
    assert calls == [(("own", True), {"metadata": OWN, "pooled": True})]


def test_held_census_and_snapshot_use_only_pending_own_wakes(clock, monkeypatch):
    from supervisor import queue

    _launched(clock)
    monkeypatch.setattr(queue, "PENDING", [_task("wake0001", OWN), _task("focus", FOCUS), _task("owner", OWNER)])
    monkeypatch.setattr(queue, "RUNNING", {"other": {"task": _task("other", OWN)}})
    assert queued_wakes(held_only=True) == ["wake0001"]
    assert queued_wakes() == ["wake0001", "other"]
    assert clock.clock.status_snapshot()["live_wake_held"] is True
    queue.RUNNING["wake0001"] = {"task": queue.PENDING.pop(0)}
    assert queued_wakes(held_only=True) == []
    assert clock.clock.status_snapshot()["live_wake_held"] is False
    assert "owner_turn_live" not in clock.clock.status_snapshot()


def test_pooled_terminal_leaves_direct_custody_until_its_callback_returns(clock, monkeypatch):
    finished = _launched(clock)
    monkeypatch.setattr(clock_module.time, "time", lambda: T0 + 5000)
    clock.clock._wake_finished("wake0001", True, metadata=OWN, pooled=True)
    assert clock.clock.live_wake() == "wake0001"
    settled = clock.clock.next_wake_at
    finished("wake0001", False)
    assert clock.clock.live_wake() == "" and clock.clock.next_wake_at == settled


def test_focus_descendants_and_owner_do_not_delay_an_overdue_clock(clock, monkeypatch):
    from supervisor import queue
    monkeypatch.setattr(queue, "PENDING", [_task("focus-pooled", FOCUS), _task("kid", STARTED)])
    monkeypatch.setattr(queue, "RUNNING", {})
    turns = {tid: _task(tid, FOCUS) for tid in ("focus1", "focus2")}
    _direct(monkeypatch, turns)
    assert queued_wakes() == [] and clock.clock.live_wake() == ""
    turns["owner"] = _task("owner", OWNER)
    get_direct_activity_registry().register("owner", 7)
    assert clock.clock.live_wake() == ""
    assert clock.clock.tick(T0 + FLOOR + 1) == "launched"
    queue.PENDING.append(_task("wake0001", OWN))
    assert queued_wakes() == ["wake0001"] and clock.clock.live_wake() == "wake0001"


def test_counted_roots_are_exactly_autonomous_independent_roots(tmp_path, monkeypatch):
    q, _, _ = _install_queue(tmp_path, monkeypatch)
    q.PENDING.extend([_task("own", OWN), _task("focus", FOCUS), _task("started", STARTED),
                      _task("owner-commission", {**STARTED, "work_kind": "owner"}),
                      _task("kid", STARTED, parent_task_id="started", delegation_role="subagent"),
                      _task("kid-role-missing", STARTED, parent_task_id="started"),
                      _task("continuation", STARTED, _consciousness_continuation=True),
                      # Unmarked system/owner work is checked at the consciousness level, yet
                      # it is not the mind's own task: it takes no slot and no allowance.
                      _task("unmarked-system", {})])
    q.RUNNING.update(started={"task": _task("started", STARTED)}, other={"task": _task("other", STARTED)})
    assert q.live_consciousness_root_count() == 2
    assert {r["task_id"] for r in q.live_consciousness_roots()} == {"started", "other"}
    q.PENDING.clear()
    q.RUNNING.clear()
    assert q.live_consciousness_root_count() == 0


def test_stop_reaches_direct_pool_children_and_lent_focus_but_leaves_owner(clock, monkeypatch):
    from supervisor import queue
    from ouroboros import conversation_lease
    turns = {"own": _task("own", OWN), "focus": _task("focus", FOCUS), "owner": _task("owner", OWNER),
             "commissioned": _task("commissioned", {**STARTED, "work_kind": "owner"})}
    _direct(monkeypatch, turns)
    monkeypatch.setattr(queue, "PENDING", [_task("started", STARTED), _task("kid", STARTED), _task("owner-root", OWNER)])
    monkeypatch.setattr(queue, "RUNNING", {})
    monkeypatch.setattr(conversation_lease, "live_authors", lambda **kw: ["focus", "lent"])
    stopped = []
    monkeypatch.setattr(clock.clock, "_stop_live_wake", stopped.append)
    monkeypatch.setattr(clock_module.threading, "Thread", lambda *, target, args, **kw: SimpleNamespace(start=lambda: target(*args)))
    assert clock.clock.stop().startswith("Background consciousness disabled")
    assert sorted(stopped) == ["focus", "kid", "lent", "own", "started"]
    assert clock.clock.enabled is False


@pytest.mark.parametrize("metadata", [OWN, FOCUS, STARTED])
def test_recovered_autonomous_work_rechecks_off_and_panic_at_assignment(tmp_path, monkeypatch, metadata):
    from supervisor import state
    from supervisor.worker_assignment import _choose_candidate
    q, _, workers = _install_queue(tmp_path, monkeypatch)
    enabled, panic = {"value": False}, {"value": False}
    monkeypatch.setattr(state, "load_state", lambda: {"bg_consciousness_enabled": enabled["value"]})
    monkeypatch.setattr(clock_module, "panic_blocks_wake", lambda root: panic["value"])
    workers.PENDING.append(_task("restored", metadata, depth=0, admitted_dispatch="none"))
    choose = lambda: _choose_candidate(q, 5.0, set(), set(), {}, set())
    assert choose() == (None, False)
    enabled["value"] = True
    assert choose() == (0, False)
    panic["value"] = True
    assert choose() == (None, False)
    workers.PENDING[0]["metadata"] = OWNER
    assert choose() == (0, False)


def test_off_switch_stops_only_work_consciousness_started(tmp_path, monkeypatch):
    """Unmarked work is checked at the consciousness level, yet Off never stops it from dispatching."""
    from supervisor import consciousness_carriers as carriers
    from supervisor import state

    monkeypatch.setattr(state, "load_state", lambda: {"bg_consciousness_enabled": False})
    monkeypatch.setattr("ouroboros.consciousness.panic_blocks_wake", lambda root: False)
    unmarked = {"id": "system", "metadata": {"work_kind": "consciousness"}}
    started = {"id": "started", "metadata": {"initiator": "consciousness", "work_kind": "consciousness"}}
    commissioned = {"id": "owner", "metadata": {"initiator": "consciousness", "work_kind": "owner"}}
    assert carriers.autonomous_work_enabled(unmarked, tmp_path) is True
    assert carriers.autonomous_work_enabled(commissioned, tmp_path) is True
    assert carriers.autonomous_work_enabled(started, tmp_path) is False
    monkeypatch.setattr(state, "load_state", lambda: {"bg_consciousness_enabled": True})
    assert carriers.autonomous_work_enabled(started, tmp_path) is True

def test_turning_off_cancels_only_work_consciousness_started(monkeypatch):
    """The stop sweep and the dispatch gate select the same work: origin and kind together."""
    from supervisor import consciousness_carriers as carriers
    from supervisor import queue

    pending = [
        {"id": "system-job", "metadata": {"work_kind": "consciousness"}},
        {"id": "historical", "metadata": {}},
        {"id": "commissioned", "metadata": {"initiator": "consciousness", "work_kind": "owner"}},
        {"id": "autonomous-job", "metadata": {"initiator": "consciousness", "work_kind": "consciousness"}},
    ]
    running = {"w1": {"task": {"id": "autonomous-child", "metadata": {
        "initiator": "consciousness", "work_kind": "consciousness", "consciousness_cause": "started"}}}}
    monkeypatch.setattr(queue, "PENDING", pending)
    monkeypatch.setattr(queue, "RUNNING", running)
    assert carriers.autonomous_tasks() == ["autonomous-job", "autonomous-child"]
