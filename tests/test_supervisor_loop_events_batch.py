"""The supervisor loop's events phase is BOUNDED and pays one budget-projection write per turn.

Two starvation classes of the same live incident: a producer that keeps the worker
event queue non-empty used to hold the loop inside its drain until the queue was
empty, so an owner message waited behind every queued event; and every ``llm_usage``
event paid its own ledger render plus STATE_LOCK write. One events pass now drains at
most N events or T seconds (``runtime_limits``), FIFO, the remainder next turn; bridge
intake runs every turn; and the compatibility projection is written once per turn
after intake, kept dirty on failure and retried no more often than a bounded interval.
Every guard is pinned in both directions.
"""

from __future__ import annotations

import queue
import time
from types import SimpleNamespace

import pytest

from ouroboros import server_liveness as sl


def _ctx(dispatched: list):
    return SimpleNamespace(dispatched=dispatched, restarts=[], DRIVE_ROOT=None)


@pytest.fixture
def dispatch_spy(monkeypatch):
    """Route ``dispatch_event`` to the ctx so a test sees order and count."""
    from supervisor import events as events_mod

    monkeypatch.setattr(events_mod, "dispatch_event", lambda evt, ctx: ctx.dispatched.append(evt["n"]))


def _liveness():
    return [time.monotonic(), {}, time.thread_time(), None]


def test_one_events_pass_is_count_bounded_and_intake_runs_with_events_still_queued(monkeypatch, dispatch_spy):
    """A producer that keeps the queue non-empty still lets an owner message be taken
    by intake within one turn: the pass stops at its bound, the remainder stays queued
    in FIFO order and the turn's next step (intake) sees the queue non-empty."""
    monkeypatch.setattr(sl, "SUPERVISOR_EVENT_BATCH_MAX_EVENTS", 5)
    monkeypatch.setattr(sl, "SUPERVISOR_EVENT_BATCH_MAX_SEC", 60.0)
    event_q: "queue.Queue" = queue.Queue()
    for n in range(12):
        event_q.put({"type": "worker_event", "n": n, "ts": "2026-09-25T00:00:00+00:00"})
    ctx = _ctx([])
    liveness = _liveness()
    inbox = ["owner message"]
    turns = []

    def one_turn():
        backlog = sl.drain_worker_events(event_q, ctx, liveness, on_restart=lambda evt, c: None)
        taken = inbox.pop() if inbox else None  # bridge intake of this turn
        turns.append((backlog, event_q.qsize(), taken))

    one_turn()
    assert turns == [(True, 7, "owner message")], turns
    assert ctx.dispatched == [0, 1, 2, 3, 4]
    assert liveness[sl._LAG] is not None, "drained events are lag-observed"
    one_turn()
    one_turn()
    assert ctx.dispatched == list(range(12))
    assert turns[1:] == [(True, 2, None), (False, 0, None)]


def test_one_events_pass_is_time_bounded(monkeypatch):
    from supervisor import events as events_mod

    monkeypatch.setattr(sl, "SUPERVISOR_EVENT_BATCH_MAX_EVENTS", 1000)
    monkeypatch.setattr(sl, "SUPERVISOR_EVENT_BATCH_MAX_SEC", 0.05)
    monkeypatch.setattr(events_mod, "dispatch_event",
                        lambda evt, ctx: (ctx.dispatched.append(evt["n"]), time.sleep(0.03)))
    event_q: "queue.Queue" = queue.Queue()
    for n in range(50):
        event_q.put({"type": "worker_event", "n": n})
    ctx = _ctx([])

    assert sl.drain_worker_events(event_q, ctx, _liveness(), on_restart=lambda evt, c: None) is True
    assert 1 <= len(ctx.dispatched) < 50
    assert event_q.qsize() == 50 - len(ctx.dispatched)


def test_restart_requests_are_routed_and_an_empty_queue_ends_the_pass(dispatch_spy):
    event_q: "queue.Queue" = queue.Queue()
    event_q.put({"type": "worker_event", "n": 1})
    event_q.put({"type": "restart_request", "n": 2})
    event_q.put({"type": "worker_event", "n": 3})
    ctx = _ctx([])

    backlog = sl.drain_worker_events(
        event_q, ctx, _liveness(), on_restart=lambda evt, c: c.restarts.append(evt["n"]))

    assert backlog is False and ctx.dispatched == [1, 3] and ctx.restarts == [2]


def test_flush_writes_once_when_dirty_and_clears_only_on_success(monkeypatch):
    writes = []
    ctx = SimpleNamespace(budget_projection_dirty=True,
                          update_budget_from_usage=lambda usage: writes.append(usage) or True)

    sl.flush_budget_projection(ctx)
    assert writes == [{}] and ctx.budget_projection_dirty is False
    sl.flush_budget_projection(ctx)
    assert writes == [{}], "a clean context pays nothing"

    monkeypatch.setattr(sl, "BUDGET_PROJECTION_RETRY_SEC", 30.0)
    clock = [1000.0]
    monkeypatch.setattr(sl.time, "monotonic", lambda: clock[0])
    outcomes = [False, RuntimeError("ledger unreadable"), True]

    def writer(usage):
        writes.append(usage)
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    ctx = SimpleNamespace(budget_projection_dirty=True, update_budget_from_usage=writer)
    sl.flush_budget_projection(ctx)
    assert len(writes) == 2 and ctx.budget_projection_dirty is True, "a refused write keeps the flag"
    sl.flush_budget_projection(ctx)
    assert len(writes) == 2, "no retry inside the bounded interval"
    clock[0] += 30.0
    sl.flush_budget_projection(ctx)
    assert len(writes) == 3 and ctx.budget_projection_dirty is True, "an exception keeps the flag too"
    clock[0] += 30.0
    sl.flush_budget_projection(ctx)
    assert len(writes) == 4 and ctx.budget_projection_dirty is False and not outcomes


def test_n_llm_usage_events_in_one_turn_produce_one_write_after_intake(monkeypatch, tmp_path):
    """N events in one drain -> one writer call, after bridge intake; the event
    rows carry ``deferred``."""
    import json

    from supervisor import events as events_mod

    (tmp_path / "logs").mkdir()
    order = []
    ctx = SimpleNamespace(DRIVE_ROOT=tmp_path, dispatched=[], RUNNING={},
                          bridge=SimpleNamespace(push_log=lambda payload: None),
                          update_budget_from_usage=lambda usage: order.append("write") or True)
    monkeypatch.setattr(events_mod, "dispatch_event",
                        lambda evt, c: events_mod._handle_llm_usage(evt, c))
    event_q: "queue.Queue" = queue.Queue()
    for n in range(8):
        event_q.put({"type": "llm_usage", "task_id": f"t{n}", "usage": {"prompt_tokens": 1, "cost": 0.01}})

    sl.drain_worker_events(event_q, ctx, _liveness(), on_restart=lambda evt, c: None)
    order.append("intake")
    sl.flush_budget_projection(ctx)

    assert order == ["intake", "write"]
    rows = [json.loads(line) for line in (tmp_path / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 8 and {row["projection_update_status"] for row in rows} == {"deferred"}
    assert ctx.budget_projection_dirty is False
