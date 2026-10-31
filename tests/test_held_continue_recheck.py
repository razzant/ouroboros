"""A writer-held Continue is re-checked on the body cadence, not only at a member's terminal.

A Continue admitted while its predecessor's writers were not proven settled waits under
``continuation_writer_unsettled``. Its blockers can settle with no tree member reaching a
terminal (a delegated run's custody closes, a handoff resolves, a process exits, a late
phase ends), so the body maintenance schedules ``recheck_held_continuations`` while such a
row is pending. A writer hold installs no root fence: the held row itself is the hint, never
the owner-Pause fence gate. Each held predecessor gets one ``release_settled_continuations``,
the ordinary selection that re-checks the writers and selects a row once.

Real admission, real delegated custody and the real selection run here; the other live
blocker kinds are faked at the census seam (``action_writers``), one per kind.
"""
from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from ouroboros import delegate_custody as dc
from tests._budget_pause_exact_helpers import _install_queue
from tests.test_owner_continue import NONCE, _interrupted

pytestmark = pytest.mark.serial  # module-global latches and the delegate custody cache

PRED = "pred-1"
DUTY = "held_continuation_recheck"


def _tick(monkeypatch, run=()):
    """One body tick with the 20 s cadence due; only the named duties actually run (inline)."""
    import ouroboros.server_maintenance as sm

    started = []

    def start(latch, name, target, args):
        started.append(name)
        if name in run:
            target(*args)  # the duty releases its own latch
        else:
            latch.release()
        return True

    monkeypatch.setattr(sm, "_start_maintenance_thread", start)
    monkeypatch.setattr(sm, "_LAST_CANCEL_INTENT_SWEEP", [0.0])
    monkeypatch.setattr(sm, "_LAST_COLD_CUSTODY_SWEEP", [0.0])
    sm._periodic_supervisor_maintenance([time.time()], [time.time()])
    assert not sm._CONTINUATION_SWEEP_LOCK.locked()
    return started


def _held(workers, ack):
    from supervisor.events_budget import HOLD_CONTINUATION_WRITER, budget_hold_fact

    row = next(task for task in workers.PENDING if task["id"] == ack["successor_task_id"])
    assert budget_hold_fact(row)["reason"] == HOLD_CONTINUATION_WRITER
    return row


def _selections(root, task_id):
    rows = [json.loads(line) for line in (root / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    return [row for row in rows if row.get("type") == "budget_hold_selected" and row.get("task_id") == task_id]


def test_a_retained_delegated_run_releases_its_continue_through_the_tick_without_any_terminal(tmp_path, monkeypatch):
    from supervisor.continuation_admission import admit_continuation
    from supervisor.events_budget import budget_hold_fact

    queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    _interrupted(tmp_path, PRED)
    dc._CUSTODY.clear()
    assert dc.record_started(tmp_path, dc.RunCustody(
        run_id="run-leaf", task_id=PRED, route_id="codex", model="leaf-model",
        snapshot_id="snap-leaf", authority_source="active_workspace", ledger_root=str(tmp_path)))
    dc._CUSTODY.clear()
    ack = admit_continuation(PRED, action_nonce=NONCE)
    assert ack["ok"] and ack["held"], ack
    held = _held(workers, ack)
    assert [(b["kind"], b.get("run_id")) for b in budget_hold_fact(held)["blockers"]] == [("delegated_run", "run-leaf")]
    assert queue.BUDGET_ROOT_FENCES == {}, "a writer hold installs no fence"

    # The run is still retained: the duty is scheduled (no fence needed) and releases nothing.
    started = _tick(monkeypatch, run=(DUTY,))
    assert DUTY in started and "owner_pause_settlement" not in started
    assert budget_hold_fact(held)["blockers"][0]["run_id"] == "run-leaf"

    # The run's custody closes; no member of the tree reaches a terminal.
    dc._CUSTODY.clear()
    assert dc.emit(tmp_path, dc.SETTLED, {"run_id": "run-leaf", "task_id": PRED, "route": "codex",
                                          "model": "leaf-model", "state": "completed"})
    dc._CUSTODY.clear()
    assert DUTY in _tick(monkeypatch, run=(DUTY,))
    assert budget_hold_fact(held) is None
    assert held["_resume_selection"]["selected_by"] == "reconciliation"
    assert len(_selections(tmp_path, held["id"])) == 1
    # Released: the next tick has no held row to schedule for.
    assert DUTY not in _tick(monkeypatch)


LIVE_BLOCKERS = [
    {"kind": "delegated_run", "task_id": PRED, "run_id": "run-1", "state": "running"},
    {"kind": "model_handoff", "attempt_id": "attempt-1", "review_owned": False},
    {"kind": "tool_handoff", "task_id": PRED, "operation": {"tool": "external_write"}},
    {"kind": "owned_process", "task_id": PRED, "pid": 4242},
    {"kind": "custody_unreadable", "task_id": PRED},
    {"kind": "late_phase_running", "task_id": PRED},
    {"kind": "merge_operation", "task_id": PRED, "receipt_id": "merge-1"},
    {"kind": "workspace_executor", "task_id": PRED, "record_id": "exec-1", "executor_type": "local"},
]


def _admit_held_behind(tmp_path, monkeypatch, census):
    from supervisor import continuation_admission as admission

    monkeypatch.setattr(admission, "action_writers", lambda q, predecessor, **kw: [dict(row) for row in census])
    queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    _interrupted(tmp_path, PRED)
    ack = admission.admit_continuation(PRED, action_nonce=NONCE)
    assert ack["ok"] and ack["held"] and ack["blockers"] == census, ack
    return queue, workers, _held(workers, ack)


@pytest.mark.parametrize("blocker", LIVE_BLOCKERS, ids=[row["kind"] for row in LIVE_BLOCKERS])
def test_every_live_blocker_kind_holds_until_the_census_clears_then_the_duty_releases(tmp_path, monkeypatch, blocker):
    from supervisor.continuation_admission import recheck_held_continuations
    from supervisor.events_budget import HOLD_CONTINUATION_WRITER, budget_hold_fact

    census = [dict(blocker)]
    _queue, _workers, held = _admit_held_behind(tmp_path, monkeypatch, census)
    assert recheck_held_continuations() == []
    assert budget_hold_fact(held)["reason"] == HOLD_CONTINUATION_WRITER
    assert _selections(tmp_path, held["id"]) == []

    census.clear()
    assert recheck_held_continuations() == [held["id"]]
    assert budget_hold_fact(held) is None
    assert recheck_held_continuations() == [], "a selected row is no longer held"


def test_concurrent_passes_and_a_terminal_reconciliation_select_the_row_once(tmp_path, monkeypatch):
    from supervisor import events_budget
    from supervisor.continuation_admission import recheck_held_continuations, release_settled_continuations

    census = [{"kind": "owned_process", "task_id": PRED, "pid": 4242}]
    _queue, _workers, held = _admit_held_behind(tmp_path, monkeypatch, census)
    census.clear()
    # Two duty passes and a member terminal's reconciliation observe at the same moment.
    barrier = threading.Barrier(3, timeout=10)
    real_observe = events_budget.observe_held_budget_selection

    def observe(q, task_id):
        barrier.wait()
        return real_observe(q, task_id)

    monkeypatch.setattr(events_budget, "observe_held_budget_selection", observe)
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(recheck_held_continuations), pool.submit(recheck_held_continuations),
                   pool.submit(release_settled_continuations, PRED)]
        released = [future.result(timeout=20) for future in futures]
    assert sorted(task_id for result in released for task_id in result) == [held["id"]]
    assert len(_selections(tmp_path, held["id"])) == 1


def test_no_writer_hold_schedules_nothing_and_a_newer_hold_is_never_released(tmp_path, monkeypatch):
    from supervisor import continuation_admission as admission
    from supervisor.events_budget import HOLD_OWNER_RESTART, budget_hold_fact
    from supervisor.restart_retention import hold_for_owner_restart

    queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    workers.PENDING.append({"id": "plain", "root_task_id": "plain"})
    calls = []
    real_release = admission.release_settled_continuations
    monkeypatch.setattr(admission, "release_settled_continuations",
                        lambda root: calls.append(root) or real_release(root))
    assert admission.held_continuation_predecessors(workers.PENDING) == []
    assert DUTY not in _tick(monkeypatch, run=(DUTY,))
    assert admission.recheck_held_continuations() == [] and calls == []

    # A Restart that caught the held Continue replaced its writer hold: no longer this duty's row.
    census = [{"kind": "custody_unreadable", "task_id": PRED}]
    _queue, workers, held = _admit_held_behind(tmp_path, monkeypatch, census)
    assert admission.held_continuation_predecessors(workers.PENDING) == [PRED]
    hold_for_owner_restart(held, tmp_path)
    assert budget_hold_fact(held)["reason"] == HOLD_OWNER_RESTART
    census.clear()
    assert admission.held_continuation_predecessors(workers.PENDING) == []
    assert DUTY not in _tick(monkeypatch, run=(DUTY,))
    assert admission.recheck_held_continuations() == []
    assert budget_hold_fact(held)["reason"] == HOLD_OWNER_RESTART
