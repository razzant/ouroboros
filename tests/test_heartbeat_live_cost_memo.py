"""Cost law L4: a heartbeat is an in-memory liveness stamp and one frame; it reads no store.

E3 (FINAL_PLAN): handling a heartbeat opens no store transaction — no ``current.sqlite``
custody snapshot, no ``usage.sqlite`` read, no path resolution — and stamps liveness from
the beat's own source time, so a drained backlog never makes a task look freshly alive.
The root's cost figure on the frame is the one ``LIVE_ROOT_COSTS`` holds; the supervisor's
once-per-turn usage flush reads it again only for a root whose money moved (its usage) or
that a heartbeat met for the first time. Each guard is pinned both ways: an unchanged pass
reads nothing, a change still produces its one read.
"""
from __future__ import annotations

import datetime as dt
import pathlib
import time
import types

import pytest

from tests import fixtures_usage_store as money

data_root = money.data_root


def _iso(epoch: float) -> str:
    return dt.datetime.fromtimestamp(epoch, tz=dt.timezone.utc).isoformat()


def _ctx(root, running):
    pushed = []
    return types.SimpleNamespace(DRIVE_ROOT=root, RUNNING=running,
                                 bridge=types.SimpleNamespace(push_log=pushed.append)), pushed


class _Reads:
    """Every store-shaped read the old drain path paid, counted while ``armed``."""

    def __init__(self, monkeypatch):
        from ouroboros import delegate_custody_current, usage_accounting

        self.breakdowns, self.snapshots, self.resolves = [], [], []
        breakdown, snapshot, resolve = (usage_accounting.usage_breakdown,
                                        delegate_custody_current.task_snapshot, pathlib.Path.resolve)
        monkeypatch.setattr(usage_accounting, "usage_breakdown",
                            lambda *a, **k: self.breakdowns.append(k) or breakdown(*a, **k))
        monkeypatch.setattr(delegate_custody_current, "task_snapshot",
                            lambda *a, **k: self.snapshots.append(a) or snapshot(*a, **k))
        self.armed = False

        def counted_resolve(path, *a, **k):
            if self.armed:
                self.resolves.append(str(path))
            return resolve(path, *a, **k)

        monkeypatch.setattr(pathlib.Path, "resolve", counted_resolve)

    def beat(self, handler, evt, ctx):
        """One heartbeat under the store meter; returns the meter it filled."""
        from ouroboros import store_timing

        before = (len(self.breakdowns), len(self.snapshots))
        self.armed = True
        try:
            with store_timing.metered() as meter:
                handler(evt, ctx)
        finally:
            self.armed = False
        assert (len(self.breakdowns), len(self.snapshots)) == before, "a heartbeat read money or custody"
        assert self.resolves == [], "a heartbeat resolved a path"
        return store_timing.snapshot(meter)


def test_root_heartbeats_read_no_store_and_the_flush_reads_only_a_changed_root(data_root, monkeypatch):
    from ouroboros import delegate_custody as custody
    from ouroboros import usage_store
    from ouroboros.projects_registry import project_chat_for_task_tree
    from ouroboros.server_liveness import flush_budget_projection
    from ouroboros.task_results import STATUS_RUNNING, write_task_result
    from supervisor.events_budget import _handle_llm_usage
    from supervisor.events_worker_reports import _handle_task_heartbeat

    usage_store.migrate_from_journal(data_root)
    money._settle(data_root, cost=1.25, cost_final=True, task_id="root", root_task_id="root")
    assert custody.record_started(data_root, custody.RunCustody(
        run_id="open-run", task_id="root", root_task_id="root", route_id="test"))
    (data_root / "logs").mkdir(exist_ok=True)
    write_task_result(data_root, "root", STATUS_RUNNING, root_task_id="root", chat_id=1)
    write_task_result(data_root, "child", STATUS_RUNNING, root_task_id="root", parent_task_id="root",
                      delegation_role="subagent", chat_id=1)
    # The room address every frame of a task shares (heartbeats, sends, logs) is resolved once per
    # task per bindings revision by the routing seam; an earlier frame of these tasks paid it.
    project_chat_for_task_tree(data_root, "root", "", "root")
    project_chat_for_task_tree(data_root, "child", "root", "root")
    started = time.time() - 60
    running = {
        "root": {"task": {"id": "root", "type": "task", "root_task_id": "root"}, "started_at": started},
        "child": {"task": {"id": "child", "type": "task", "root_task_id": "root", "parent_task_id": "root",
                           "delegation_role": "subagent"}, "started_at": started},
    }
    ctx, pushed = _ctx(data_root, running)
    ctx.update_budget_from_usage = lambda usage: True
    reads = _Reads(monkeypatch)

    def beat(task_id):
        meter = reads.beat(_handle_task_heartbeat, {"task_id": task_id, "phase": "running",
                                                    "ts": _iso(time.time())}, ctx)
        assert meter == {}, f"the heartbeat opened a store transaction: {meter}"
        return pushed[-1]

    # First sight: no figure yet (absent, never zero) and still no read on the drain.
    first = beat("root")
    assert "accounted_upper_bound_usd_with_children" not in first and "cost_accounting_status" not in first
    flush_budget_projection(ctx)
    assert len(reads.breakdowns) == 1 and reads.breakdowns[0]["include_custody"] is True

    held = beat("root")
    assert held["accounted_upper_bound_usd_with_children"] == pytest.approx(1.25)
    assert held["cost_custody"]["open_run_ids"] == ["open-run"]
    assert held["cost_final"] is False and held["cost_with_children_partial"] is True
    child = beat("child")
    assert "cost_accounting_status" not in child, "a child carries no root figure"

    # Rest law: beats and flushes over unchanged money read nothing.
    for _ in range(3):
        beat("root")
        flush_budget_projection(ctx)
    assert len(reads.breakdowns) == 1

    # A real change: the child's usage row marks its root, and exactly one read follows.
    money._settle(data_root, cost=0.75, cost_final=True, task_id="child", root_task_id="root")
    _handle_llm_usage({"type": "llm_usage", "task_id": "child", "usage": {"prompt_tokens": 1, "cost": 0.75}}, ctx)
    flush_budget_projection(ctx)
    flush_budget_projection(ctx)
    assert len(reads.breakdowns) == 2
    assert beat("root")["accounted_upper_bound_usd_with_children"] == pytest.approx(2.0)

    # A root that left RUNNING is forgotten by the next flush with work; it is never read again.
    del running["root"]
    _handle_llm_usage({"type": "llm_usage", "task_id": "child", "root_task_id": "root",
                       "usage": {"prompt_tokens": 1, "cost": 0.0}}, ctx)
    flush_budget_projection(ctx)
    assert len(reads.breakdowns) == 2


def test_heartbeat_liveness_is_the_beats_own_source_time(tmp_path):
    from supervisor.events_worker_reports import _handle_task_heartbeat

    now = time.time()
    meta = {"task": {"id": "t", "type": "task", "parent_task_id": "root", "root_task_id": "root"},
            "started_at": now - 100}
    ctx, pushed = _ctx(tmp_path, {"t": meta})

    _handle_task_heartbeat({"task_id": "t", "phase": "running", "ts": _iso(now - 50)}, ctx)
    assert meta["last_heartbeat_at"] == pytest.approx(now - 50, abs=0.01), "the beat's own time, not its handling"
    # A beat spoken earlier, drained later (the telemetry backlog), never moves the stamp back or its phase.
    _handle_task_heartbeat({"task_id": "t", "phase": "start", "ts": _iso(now - 80)}, ctx)
    assert meta["last_heartbeat_at"] == pytest.approx(now - 50, abs=0.01) and meta["heartbeat_phase"] == "running"
    # A beat spoken before this attempt started belongs to an earlier attempt.
    _handle_task_heartbeat({"task_id": "t", "phase": "running", "ts": _iso(now - 200)}, ctx)
    assert meta["last_heartbeat_at"] == pytest.approx(now - 50, abs=0.01)
    # A beat without a source time is no liveness evidence; one dated ahead is clamped to now.
    _handle_task_heartbeat({"task_id": "t", "phase": "running"}, ctx)
    assert meta["last_heartbeat_at"] == pytest.approx(now - 50, abs=0.01)
    _handle_task_heartbeat({"task_id": "t", "phase": "running", "ts": _iso(now + 3600)}, ctx)
    assert now <= meta["last_heartbeat_at"] <= time.time()
    assert len(pushed) == 5, "every beat of a running task still reaches the live view"


def test_a_failed_figure_read_keeps_the_last_figure_and_retries_no_sooner_than_the_bound(tmp_path, monkeypatch):
    from ouroboros import cost_projection, usage_accounting

    memo = cost_projection.LiveRootCosts()
    task = {"id": "root", "type": "task", "root_task_id": "root"}
    running = {"root": {"task": task}}
    answers = [{"accounted_usd": 3.0, "attempt_counts": {"settled": 1}, "priced_rows": 1,
                "tracked_nonfinal_rows": 0, "accounting_open_rows": 0}, OSError("ledger busy")]

    def read(_root, **_kwargs):
        answer = answers.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return answer

    monkeypatch.setattr(usage_accounting, "usage_breakdown", read)
    clock = [1000.0]
    monkeypatch.setattr(cost_projection.time, "monotonic", lambda: clock[0])
    assert memo.fields(tmp_path, "root", task) == {}
    memo.refresh(running, tmp_path)
    assert memo.fields(tmp_path, "root", task)["accounted_upper_bound_usd_with_children"] == 3.0
    memo.mark(tmp_path, "root")
    memo.refresh(running, tmp_path)  # the failed read
    assert memo.fields(tmp_path, "root", task)["accounted_upper_bound_usd_with_children"] == 3.0
    answers.append(OSError("not read inside the bound"))
    memo.refresh(running, tmp_path)
    assert len(answers) == 1, "inside the retry bound nothing is read again"
    from ouroboros.runtime_limits import BUDGET_PROJECTION_RETRY_SEC

    clock[0] += BUDGET_PROJECTION_RETRY_SEC
    memo.refresh(running, tmp_path)
    assert answers == [], "after the bound the root is read again"
    assert memo.fields(tmp_path, "root", task)["accounted_upper_bound_usd_with_children"] == 3.0
