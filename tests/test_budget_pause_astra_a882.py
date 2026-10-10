"""Regression tests for the four Astra run-a882315dbcd7 findings closed in #1196
(commit 73d9c9a2): an orphaned durable Resume grant is revoked instead of
answering ``resume_already_granted`` forever; an unreadable custody chain is
``custody_read=failed``, never "no open runs"; a pending (START_REQUESTED,
unbound) invocation of THIS task is unknown custody; and a parallel tool batch
that leaves on ``UsageAccountingError`` waits for its already-started calls.

Fixtures come from the exact budget-pause suite (``tests/_budget_pause_exact_helpers``).
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

from tests._budget_pause_exact_helpers import (  # noqa: F401 -- shared fixtures of the exact-pause suite
    _install_queue,
    _loop_ctx,
    _parked,
)


def _idle_worker(workers, sent):
    workers.WORKERS[0] = SimpleNamespace(wid=0, busy_task_id=None, reaping=False,
                                         in_q=SimpleNamespace(put=lambda t: sent.append(dict(t))))


def _recording_writer(monkeypatch, budget_pause):
    """Wrap the real durable writer so a test can see every row it was asked to write."""
    real_writer = budget_pause.set_budget_pause
    written = []

    def _record(root, task_id, row, expected_pause_id=None, **kwargs):
        written.append((task_id, dict(row), dict(kwargs)))
        return real_writer(root, task_id, row, expected_pause_id, **kwargs)

    monkeypatch.setattr(budget_pause, "set_budget_pause", _record)
    return written


def _crash_restored_carrier(task, handoff):
    """The queue row a restore sees after a crash BETWEEN the durable grant and the
    snapshot: only the prior ``_budget_pause`` marker, no handoff, no hold."""
    stale = {key: value for key, value in task.items()
             if key not in {"_budget_pause_resume", "budget_resumed_at", "_budget_pause_hold"}}
    stale["_budget_pause"] = dict(handoff["pause"])
    return stale


# --- #1: orphaned grant without a queue carrier ------------------------------------

def test_a_durable_grant_no_queue_row_carries_is_revoked_and_the_owner_resume_mints_anew(tmp_path, monkeypatch):
    """Astra a882 #1: durable row RESUME_GRANTED, the restored queue carrier holds
    only ``_budget_pause`` (no ``_budget_pause_resume``, no hold). The owner's
    Resume revokes the orphan (``orphaned_grant_without_carrier``) and mints a
    NEW generation-2 grant instead of answering ``resume_already_granted`` forever."""
    from ouroboros import budget_pause

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda _st, **_k: 5.0)
    task, _row = _parked(tmp_path, monkeypatch, task_id="a882-orphan")
    first = queue.resume_budget_paused_task("a882-orphan")
    assert first["ok"] is True
    first_grant = first["grant_id"]
    handoff = task["_budget_pause_resume"]
    assert handoff["grant_id"] == first_grant
    assert budget_pause.budget_pause_row(tmp_path, "a882-orphan")["state"] == budget_pause.STATE_RESUME_GRANTED

    # Crash between the durable grant and the queue snapshot: restore brings back
    # the pre-grant row. Nothing in PENDING/RUNNING carries ``first_grant``.
    workers.PENDING.remove(task)
    restored = _crash_restored_carrier(task, handoff)
    workers.PENDING.append(restored)
    assert "_budget_pause_resume" not in restored and "_budget_pause_hold" not in restored
    written = _recording_writer(monkeypatch, budget_pause)

    second = queue.resume_budget_paused_task("a882-orphan")
    assert second.get("error") is None, second
    assert second["ok"] is True and second["grant_id"] != first_grant
    assert second["grant_generation"] == 2
    assert "released_hold" not in second  # there was no hold to release
    # The orphan was revoked FIRST, typed, under its own identity (CAS on the granted state).
    revocations = [(row, kwargs) for _tid, row, kwargs in written
                   if isinstance(row.get("grant"), dict) and row["grant"].get("revoke_reason")]
    assert len(revocations) == 1
    revoked_row, cas = revocations[0]
    assert revoked_row["state"] == budget_pause.STATE_PAUSED
    assert revoked_row["grant"]["grant_id"] == first_grant and revoked_row["grant"]["revoked_at"]
    assert revoked_row["grant"]["revoke_reason"] == "orphaned_grant_without_carrier"
    assert cas["expected_state"] == budget_pause.STATE_RESUME_GRANTED and cas["expected_grant_id"] == first_grant
    # The durable row now names the new grant; the old one is dead for good.
    row = budget_pause.budget_pause_row(tmp_path, "a882-orphan")
    assert row["state"] == budget_pause.STATE_RESUME_GRANTED
    assert row["grant"]["grant_id"] == second["grant_id"] and row["resume_generation"] == 2
    ctx, _limit = _loop_ctx(tmp_path, "a882-orphan")
    with pytest.raises(ValueError):
        budget_pause.load_budget_pause(ctx, {"pause_id": row["pause_id"], "grant_id": first_grant})
    # The restored row is now the live carrier and dispatches under the new grant.
    assert restored["_budget_pause_resume"]["grant_id"] == second["grant_id"]
    assert "_budget_pause" not in restored
    sent = []
    _idle_worker(workers, sent)
    workers.assign_tasks()
    assert [t["_budget_pause_resume"]["grant_id"] for t in sent] == [second["grant_id"]]


def test_a_grant_still_carried_by_a_queue_row_is_not_orphaned(tmp_path, monkeypatch):
    """Negative control for #1: while a PENDING or RUNNING row carries the matching
    ``_budget_pause_resume``, a second Resume is still ``resume_already_granted``
    and nothing is revoked or re-minted (single use holds)."""
    from ouroboros import budget_pause

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda _st, **_k: 5.0)
    task, _row = _parked(tmp_path, monkeypatch, task_id="a882-carried")
    first = queue.resume_budget_paused_task("a882-carried")
    assert first["ok"] is True
    first_grant = first["grant_id"]
    handoff = task["_budget_pause_resume"]
    written = _recording_writer(monkeypatch, budget_pause)

    # A stale ``_budget_pause`` copy is located first, but the real carrier is still PENDING.
    stale = _crash_restored_carrier(task, handoff)
    workers.PENDING.insert(0, stale)
    refused = queue.resume_budget_paused_task("a882-carried")
    assert refused["error"] == "resume_already_granted" and refused["grant_id"] == first_grant

    # The carrier dispatched to a worker (RUNNING, grant not yet consumed): still carried.
    workers.PENDING.remove(task)
    workers.RUNNING["a882-carried"] = {"task": task, "wid": 0}
    refused_running = queue.resume_budget_paused_task("a882-carried")
    assert refused_running["error"] == "resume_already_granted" and refused_running["grant_id"] == first_grant

    assert written == []  # no revocation, no new grant was written
    row = budget_pause.budget_pause_row(tmp_path, "a882-carried")
    assert row["state"] == budget_pause.STATE_RESUME_GRANTED
    assert row["grant"]["grant_id"] == first_grant and not row["grant"].get("revoked_at")
    assert row["resume_generation"] == 1
    assert task["_budget_pause_resume"]["grant_id"] == first_grant and "_budget_pause" in stale


# --- #2: unreadable custody chain is UNKNOWN, never "no open runs" -----------------

def test_an_unreadable_custody_chain_is_a_failed_read_not_no_open_runs(tmp_path, monkeypatch):
    """Astra a882 #2: ``replay`` falls back leniently (an unreadable segment reads
    as empty), so the observer probes the chain first: unreadable → ``custody_read``
    ``failed`` with ``coverage_basis`` ``custody_unreadable``, and a grant refuses typed."""
    from ouroboros import budget_pause
    from ouroboros import delegate_custody as custody
    from ouroboros import delegate_pending

    monkeypatch.setattr(custody, "custody_log_unreadable", lambda _root: True)
    monkeypatch.setattr(custody, "replay", lambda _root, rows=None: {})
    monkeypatch.setattr(delegate_pending, "pending_invocations", lambda _root, rows=None: [])

    observed = budget_pause.observe_task_runs(tmp_path, "a882-unreadable")
    assert observed["custody_read"] == "failed" and observed["runs"] == []
    assert observed["coverage_basis"] == "custody_unreadable"
    assert "custody_log_unreadable" in observed["error"]

    # Same probe, readable chain: the empty replay is a positively established absence.
    monkeypatch.setattr(custody, "custody_log_unreadable", lambda _root: False)
    clean = budget_pause.observe_task_runs(tmp_path, "a882-unreadable")
    assert clean["custody_read"] == "ok" and clean["coverage_basis"] == "no_open_runs"


def test_an_unreadable_custody_chain_refuses_the_owner_resume_typed(tmp_path, monkeypatch):
    """End to end for #2: the grant side re-reads custody through the same body
    and refuses ``external_custody_unreadable`` while the chain is unreadable."""
    from ouroboros import delegate_custody as custody

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda _st, **_k: 5.0)
    task, _row = _parked(tmp_path, monkeypatch, task_id="a882-unreadable-grant")
    monkeypatch.setattr(custody, "replay", lambda _root, rows=None: {})
    monkeypatch.setattr(custody, "custody_log_unreadable", lambda _root: True)
    refused = queue.resume_budget_paused_task("a882-unreadable-grant")
    assert refused["error"] == "external_custody_unreadable"
    assert "custody_log_unreadable" in refused["detail"]
    assert "_budget_pause" in task and "_budget_pause_resume" not in task
    monkeypatch.setattr(custody, "custody_log_unreadable", lambda _root: False)
    assert queue.resume_budget_paused_task("a882-unreadable-grant")["ok"] is True


# --- #3: a pending (unbound) invocation of THIS task is unknown custody -------------

def _pending_row(task_id, invocation_id, route="claudexor"):
    return {"invocation_id": invocation_id, "task_id": task_id, "route": route,
            "surface": "delegate", "request": {}}


def test_a_pending_invocation_of_this_task_is_unknown_custody_never_absence(tmp_path, monkeypatch):
    """Astra a882 #3: a START_REQUESTED whose response was lost has no run id yet
    but may be a live remote writer. It is observed as ``stop_unknown`` /
    ``pending_invocation_unbound`` under ``coverage_basis`` ``pending_invocations_unbound``;
    another task's pending invocation is not this task's custody."""
    from ouroboros import budget_pause
    from ouroboros import delegate_custody as custody
    from ouroboros import delegate_pending

    monkeypatch.setattr(custody, "custody_log_unreadable", lambda _root: False)
    monkeypatch.setattr(custody, "replay", lambda _root, rows=None: {})
    pending = [_pending_row("a882-pending", "inv-mine"), _pending_row("someone-else", "inv-theirs", route="codex")]
    monkeypatch.setattr(delegate_pending, "pending_invocations", lambda _root, rows=None: list(pending))

    observed = budget_pause.observe_task_runs(tmp_path, "a882-pending")
    assert observed["custody_read"] == "ok"
    assert observed["coverage_basis"] == "pending_invocations_unbound"
    assert observed["runs"] == [{
        "run_id": "", "invocation_id": "inv-mine", "route": "claudexor",
        "cost_coverage": "unproven_preterminal", "stop_policy": "reconcile_first", "review_owned": False,
        "state": budget_pause.EXTERNAL_STOP_UNKNOWN, "stop_outcome": "pending_invocation_unbound",
        "detail": "",
    }]
    assert observed["runs"][0]["state"] != budget_pause.EXTERNAL_STOP_CONFIRMED  # unsettled for a grant

    # Only the OTHER task's invocation is pending: this task has no open custody.
    other_only = budget_pause.observe_task_runs(tmp_path, "a882-nobody")
    assert other_only["custody_read"] == "ok" and other_only["coverage_basis"] == "no_open_runs"
    assert other_only["runs"] == []


def test_a_pending_invocation_keeps_the_owner_resume_refused_until_it_binds_or_fails(tmp_path, monkeypatch):
    """End to end for #3: the grant treats the unbound invocation as unsettled
    custody (``external_runs_unsettled``), records the observation on the durable
    row, and admits once the invocation is no longer pending."""
    from ouroboros import budget_pause
    from ouroboros import delegate_custody as custody
    from ouroboros import delegate_pending

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda _st, **_k: 5.0)
    task, _row = _parked(tmp_path, monkeypatch, task_id="a882-pending-grant")
    monkeypatch.setattr(custody, "custody_log_unreadable", lambda _root: False)
    monkeypatch.setattr(custody, "replay", lambda _root, rows=None: {})
    pending = [_pending_row("a882-pending-grant", "inv-1"), _pending_row("other", "inv-2")]
    monkeypatch.setattr(delegate_pending, "pending_invocations", lambda _root, rows=None: list(pending))

    refused = queue.resume_budget_paused_task("a882-pending-grant")
    assert refused["error"] == "external_runs_unsettled"
    assert refused["runs"] == [{"run_id": "", "state": budget_pause.EXTERNAL_STOP_UNKNOWN,
                                "stop_outcome": "pending_invocation_unbound"}]
    assert refused["action"] == "wait_for_delegated_runs_to_settle_or_cancel_them"
    assert "_budget_pause" in task and "_budget_pause_resume" not in task
    row = budget_pause.budget_pause_row(tmp_path, "a882-pending-grant")
    assert row["state"] == budget_pause.STATE_PAUSED
    assert row["external_runs"]["coverage_basis"] == "pending_invocations_unbound"
    assert row["external_runs"]["runs"][0]["invocation_id"] == "inv-1"

    # The invocation bound (or failed definitively): only the other task's remains.
    pending[:] = [_pending_row("other", "inv-2")]
    granted = queue.resume_budget_paused_task("a882-pending-grant")
    assert granted["ok"] is True
    assert task["_budget_pause_resume"]["external_runs"]["coverage_basis"] == "no_open_runs"


# --- #4: a parallel batch leaving on UsageAccountingError waits for started calls --

def test_a_parallel_batch_raising_usage_accounting_error_waits_for_already_started_calls(tmp_path, monkeypatch):
    """Astra a882 #4: when one parallel call raises ``UsageAccountingError`` the
    executor shutdown must WAIT for the calls that already started (each bounded
    by its own tool timeout) so an in-flight wrapper is never stranded; queued
    ones are cancelled. Call B is proven finished before ``handle_tool_calls``
    re-raises."""
    from ouroboros import loop_tool_execution as loop_tools
    from ouroboros.usage_accounting import UsageAccountingError

    b_started = threading.Event()
    b_done = threading.Event()
    marks = {}

    def fake_execute(_tools, tc, _drive_logs, _timeout, _task_id, _stateful):
        call_id = str(tc["id"])
        if call_id == "call_a":
            assert b_started.wait(timeout=5.0), "call B never started"
            raise UsageAccountingError("budget rail mid-batch")
        b_started.set()
        time.sleep(0.4)  # the in-flight wrapper, still running when A raises
        marks["b_finished_at"] = time.monotonic()
        b_done.set()
        return {"tool_call_id": call_id, "fn_name": "read_file", "result": "ok", "is_error": False,
                "tool_args": {}, "args_for_log": {}, "is_code_tool": False, "result_meta": {}}

    monkeypatch.setattr(loop_tools, "tool_calls_can_run_parallel", lambda _calls: True)
    monkeypatch.setattr(loop_tools, "_get_tool_timeout", lambda *_a, **_k: 5)
    monkeypatch.setattr(loop_tools, "_execute_with_timeout", fake_execute)
    tools = SimpleNamespace(_ctx=SimpleNamespace(_request_wire_custom_receipts=(), _current_llm_call_meta={}),
                            CODE_TOOLS=set())
    tool_calls = [
        {"id": "call_a", "type": "function", "function": {"name": "read_file", "arguments": "{}"}},
        {"id": "call_b", "type": "function", "function": {"name": "read_file", "arguments": "{}"}},
    ]
    try:
        with pytest.raises(UsageAccountingError, match="mid-batch"):
            loop_tools.handle_tool_calls(tool_calls, tools, tmp_path, "a882-batch", None, [], {}, lambda _s: None)
        raised_at = time.monotonic()
        assert b_started.is_set()
        # The re-raise happened only AFTER the started call B ran to completion.
        assert b_done.is_set(), "handle_tool_calls re-raised while call B was still running"
        assert raised_at >= marks["b_finished_at"]
        # Completion and ordering prove the wait; platform clock ticks need not measure sleep exactly.
    finally:
        b_done.wait(timeout=2.0)  # never leak the worker thread past the test


# --- Astra run-6fe5bf761449 follow-ups: one snapshot, and malformed custody lines ----

def _write_event_log(root, lines):
    from ouroboros import delegate_custody as custody

    path = custody.event_log_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(lines))
    return path


def test_both_custody_projections_come_from_one_snapshot(tmp_path, monkeypatch):
    """Astra 6fe5 #1: replay and pending_invocations must fold the SAME rows, so a
    START_REQUESTED that becomes STARTED between two reads is seen by one of them."""
    from ouroboros import budget_pause
    from ouroboros import delegate_custody as custody
    from ouroboros import delegate_pending

    snapshot = [{"type": custody.START_REQUESTED, "invocation_id": "inv-x", "task_id": "snap-1"}]
    seen = {}
    from ouroboros import delegate_custody_memo

    monkeypatch.setattr(custody, "custody_log_unreadable", lambda _root: False)
    monkeypatch.setattr(delegate_custody_memo, "custody_rows_with_integrity",
                        lambda _root, _needle: (tuple(snapshot), 0))

    def _replay(_root, rows=None):
        seen["replay"] = rows
        return {}

    def _pending(_root, rows=None):
        seen["pending"] = rows
        return [{"invocation_id": "inv-x", "task_id": "snap-1", "route": "codex"}]

    monkeypatch.setattr(custody, "replay", _replay)
    monkeypatch.setattr(delegate_pending, "pending_invocations", _pending)
    observed = budget_pause.observe_task_runs(tmp_path, "snap-1")
    assert seen["replay"] is not None and seen["replay"] == seen["pending"] == snapshot
    assert observed["coverage_basis"] == "pending_invocations_unbound"


def test_a_malformed_custody_line_naming_the_task_is_unknown_custody(tmp_path):
    """Astra 6fe5 #2, on a REAL event log: a START_REQUESTED joined onto a torn
    prefix is unparseable; the memo used to skip it silently and the observer then
    proved "no open runs". Now it is an incomplete read, never absence."""
    import json

    from ouroboros import budget_pause
    from ouroboros import delegate_custody as custody

    start = json.dumps({"type": custody.START_REQUESTED, "invocation_id": "inv-torn",
                        "task_id": "torn-1"}).encode()
    _write_event_log(tmp_path, [b'{"type": "llm_round", "x": 1', start + b"\n"])
    observed = budget_pause.observe_task_runs(tmp_path, "torn-1")
    assert observed["custody_read"] == "failed"
    assert "custody_rows_incomplete" in observed["error"]
    # Another task is not blocked by this task's torn line.
    other = budget_pause.observe_task_runs(tmp_path, "someone-else")
    assert other["custody_read"] == "ok" and other["coverage_basis"] == "no_open_runs"


def test_a_huge_torn_prefix_cannot_truncate_the_task_id_away(tmp_path):
    """Astra 2bc1 #1: a >64 KiB torn unrelated prefix with this task's
    START_REQUESTED joined after it must still count as this task's malformed line."""
    import json

    from ouroboros import budget_pause
    from ouroboros import delegate_custody as custody

    start = json.dumps({"type": custody.START_REQUESTED, "invocation_id": "inv-big",
                        "task_id": "big-1"}).encode()
    prefix = b'{"type": "delegate_run_note", "blob": "' + b"x" * 200_000
    _write_event_log(tmp_path, [prefix, start + b"\n"])
    observed = budget_pause.observe_task_runs(tmp_path, "big-1")
    assert observed["custody_read"] == "failed"


def test_a_bypassed_refresh_in_the_same_read_is_unknown(tmp_path, monkeypatch):
    """Astra 2bc1 #2: when the one refresh that produced the rows bypassed the memo
    (lenient read), the observer cannot prove absence from those rows."""
    from ouroboros import budget_pause
    from ouroboros import delegate_custody_memo

    _write_event_log(tmp_path, [b""])
    monkeypatch.setattr(delegate_custody_memo, "_refresh", lambda _root: (None, ()))
    observed = budget_pause.observe_task_runs(tmp_path, "bypass-1")
    assert observed["custody_read"] == "failed" and "unknown" in observed["error"]
