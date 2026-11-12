"""The delegated body's tick reads what it owes work to (FINAL_PLAN E2; tests/test_cost_live_set.py's held axis).

The body (``delegate_collection.BodyRuntime``) ticks every few seconds over the held custody runs, and the held set
is mostly settled runs that keep an undisposed patch for days: they owe the body nothing. These laws hold its tick
to the change:

- a held row is decoded once per stamp, and what it owes the body (an observation, a collection) is judged once
  per stamp: a resting run costs a tick nothing;
- the held set is listed once per tick period by the body's own clock, or at once after this process wrote custody
  (its own jobs' effects); another process's write is seen at the next period, as each tick saw it before;
- a run named by an explicit request (``ensure_runs``) is read afresh, and every job receives its own entry.
"""
from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from ouroboros import delegate_collection as body
from ouroboros import delegate_custody as custody
from ouroboros import delegate_custody_current as current
from ouroboros import store
from ouroboros.gateways.claudexor import ClaudexorUnavailable


def _no_daemon():
    raise ClaudexorUnavailable("daemon_unreachable", "no daemon in this test")


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    """The real body with its driver parked; the test drives its real tick on a controlled clock."""
    release, entered, tick = threading.Event(), threading.Event(), body.BodyRuntime.tick

    def park(_runtime):
        entered.set()
        assert release.wait(30)

    monkeypatch.setattr(body.BodyRuntime, "tick", park)
    clock = [100.0]
    monkeypatch.setattr(body, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    started = body.start_body_runtime(tmp_path, gateway_factory=_no_daemon)
    assert entered.wait(5)

    def step():
        """One tick, then every job it submitted to its end (a refused read included)."""
        tick(started)
        with started.lock:
            jobs = [*started.reads.values(), *started.collections.values()]
        for job in jobs:
            try:
                job.result(timeout=10)
            except Exception:
                pass

    try:
        yield started, clock, step
    finally:
        started.close(wait=False)
        release.set()
        started.close(wait=True)


def _resting(run_id: str) -> dict:
    """A settled run custody holds for its undisposed patch, with nothing for the body to observe or collect."""
    return {"run_id": run_id, "task_id": f"{run_id}-task", "root_task_id": f"{run_id}-task", "settled": True,
            "terminal_state": "", "snapshot_id": f"snap-{run_id}", "mode": "agent"}


def _lay(root, resting: int, *, open_run: bool = True) -> None:
    with store.open_store(root, write=True) as txn:
        for index in range(resting):
            txn.put_custody(f"rest{index:03d}", _resting(f"rest{index:03d}"), state=current.HELD)
    if open_run:
        entry = custody.RunCustody(run_id="open", task_id="open-task", invocation_id="inv-open", route_id="test",
                                   model="test", access="readonly", mode="ask", ledger_root=str(root))
        assert custody.record_started(root, entry)


def _count(monkeypatch, module, name) -> list:
    calls, real = [], getattr(module, name)
    monkeypatch.setattr(module, name, lambda *a, **k: calls.append(a) or real(*a, **k))
    return calls


@pytest.mark.parametrize("resting", [1, 50])
def test_a_resting_held_run_costs_a_tick_nothing(tmp_path, monkeypatch, runtime, resting):
    """Equal at 1 and 50 settled held runs: the next period's tick lists once, decodes and judges nothing."""
    started, clock, step = runtime
    _lay(tmp_path, resting)
    step()
    decodes, owed, listings = (_count(monkeypatch, current, "_decode"), _count(monkeypatch, body, "collection_due"),
                               _count(monkeypatch, current, "held_stamped"))
    clock[0] += 10.0  # the next tick period: the held set is listed again
    step()
    assert (len(listings), len(decodes), len(owed)) == (1, 0, 0)
    assert set(started._held_entries) == {"open", *(f"rest{index:03d}" for index in range(resting))}


def test_only_a_run_that_owes_the_body_gets_a_job(tmp_path, monkeypatch, runtime):
    started, clock, step = runtime
    _lay(tmp_path, 20)
    observed = []
    real = body.BodyRuntime._observe
    monkeypatch.setattr(body.BodyRuntime, "_observe", lambda self, entry: observed.append(entry) or real(self, entry))
    step()
    assert [entry.run_id for entry in observed] == ["open"], "the settled majority owes no observation"
    held = started._held_entries["open"].entry
    assert observed[0] is not held and observed[0].run_id == held.run_id, "a job never holds the body's own entry"


def test_a_tick_in_the_same_period_reuses_its_listing_unless_this_process_wrote_custody(tmp_path, monkeypatch, runtime):
    started, clock, step = runtime
    _lay(tmp_path, 3)
    step()
    listings, decodes = _count(monkeypatch, current, "held_stamped"), _count(monkeypatch, current, "_decode")
    step()
    step()
    assert listings == [], "the same period: the read jobs wrote nothing, nothing is listed again"
    custody.record_containment_fault(tmp_path, current.run(tmp_path, "rest001"), "test_fault")  # this process writes
    decodes.clear()  # the write's own reduction decoded its row: only the tick's decodes count below
    step()
    assert len(listings) == 1 and len(decodes) == 1, "its own write is listed at once, one row decoded again"
    assert started._held_entries["rest001"].entry.containment_fault.get("reason") == "test_fault"
    with store.open_store(tmp_path, write=True) as txn:  # another process's write
        txn.put_custody("rest002", {**_resting("rest002"), "containment_fault": {"reason": "elsewhere"}}, state=current.HELD)
    step()
    assert len(listings) == 1, "another process's write waits for the next period"
    clock[0] += 3.0
    step()
    assert len(listings) == 2
    assert started._held_entries["rest002"].entry.containment_fault == {"reason": "elsewhere"}


def test_a_requested_run_is_read_afresh_and_jobs_get_their_own_entry(tmp_path, monkeypatch, runtime):
    started, clock, step = runtime
    _lay(tmp_path, 1)
    step()
    assert started.ensure_runs([("open", "inv-open")])
    reads = _count(monkeypatch, current, "run")
    handed = []
    real = body.BodyRuntime._observe
    monkeypatch.setattr(body.BodyRuntime, "_observe", lambda self, entry: handed.append(entry) or real(self, entry))
    step()
    assert ("open" in [args[1] for args in reads]), "an explicit request reads its run now, listed or not"
    assert handed and handed[-1] is not started._held_entries["open"].entry, "a job never mutates the held memory"
