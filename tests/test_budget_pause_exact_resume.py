"""Exact mid-run budget pause, supervisor side (#1196): the park of the SAME
task id, its completion after a worker death, the owner-granted Resume and
its revocation, the loop-side consumption of a grant and the Q10 planning
threshold refresh. The loop-side pause and HOLD tests are in
``test_budget_pause_exact``; the shared fixtures live in
``tests._budget_pause_exact_helpers``.
"""

from __future__ import annotations

import json
import pathlib
import time
from types import SimpleNamespace

import pytest

from tests._budget_pause_exact_helpers import (
    _install_queue, _loop_ctx, _parked, _pause, _running_row, _supervisor_ctx,
)


# --------------------------------------------------------------------------- supervisor: park

def test_exact_pause_event_parks_same_task_id_and_confirms_row(tmp_path, monkeypatch):
    from ouroboros import budget_pause
    from supervisor.events import _handle_budget_pause

    queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    ctx, _limit, pause = _pause(tmp_path, monkeypatch, scope="root")
    budget_pause.end_dispatch_fence(ctx.task_id)
    task = {"id": ctx.task_id, "type": "task", "chat_id": 3, "root_task_id": ctx.task_id, "_attempt": 1}
    worker = SimpleNamespace(busy_task_id=ctx.task_id)
    workers.RUNNING[ctx.task_id] = {"task": task, "worker_id": 0, "attempt": 1}
    workers.WORKERS[0] = worker
    persisted, pushed = [], []
    sctx = _supervisor_ctx(tmp_path, workers, queue, persisted, pushed)

    event = budget_pause.pause_event(task, pause)
    assert event["resource_limit"]["exact_continuation"] is True
    _handle_budget_pause({**event, "worker_id": 0}, sctx)

    assert workers.RUNNING == {} and worker.busy_task_id is None
    marker = workers.PENDING[0]["_budget_pause"]
    assert marker["exact_continuation"] is True and marker["checkpoint"]["pause_id"] == pause["pause_id"]
    assert marker["fence_id"] and queue.BUDGET_ROOT_FENCES[ctx.task_id]["status"] == "paused"
    assert budget_pause.budget_pause_row(tmp_path, ctx.task_id)["state"] == budget_pause.STATE_PAUSED
    assert persisted == ["budget_pause_exact_continuation"]
    assert pushed[0]["type"] == "budget_scope_paused" and pushed[0]["pause_id"] == pause["pause_id"]
    assert "checkpoint" not in pushed[0]
    # The queue predicate already reads the marker: no false "queued".
    from supervisor.queue_transitions import budget_pause_fact

    assert budget_pause_fact(workers.PENDING[0])["exact_continuation"] is True


def test_late_park_confirmation_never_regresses_a_live_grant(tmp_path, monkeypatch):
    """F1: the park confirmation is a compare-and-set on the pause AND its state.

    The park's row reading is taken before the queue transition; a writer that
    moved the row on in between (an owner Resume granting it) must not be
    overwritten by a late ``paused`` carrying the stale grant-less row, and the
    owner-facing projection must not be rewritten to say paused either. The
    supervisor publishes the anomaly as itself instead of a false pause.
    """
    from ouroboros import budget_pause
    from ouroboros.task_results import load_task_result
    from supervisor.events import _handle_budget_pause

    queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    ctx, _limit, pause = _pause(tmp_path, monkeypatch, task_id="late-1")
    budget_pause.end_dispatch_fence("late-1")
    task = {"id": "late-1", "type": "task", "chat_id": 0, "root_task_id": "late-1", "_attempt": 1}
    workers.RUNNING["late-1"] = {"task": task, "worker_id": 0, "attempt": 1}
    workers.WORKERS[0] = SimpleNamespace(busy_task_id="late-1")
    row = budget_pause.budget_pause_row(tmp_path, "late-1")
    grant = {"grant_id": "g-live", "single_use": True, "generation": 1}
    persisted, pushed = [], []
    sctx = _supervisor_ctx(tmp_path, workers, queue, persisted, pushed)

    def _snapshot_then_grant(reason=""):
        # A writer this queue lock does not cover moves the row on mid-park.
        budget_pause.set_budget_pause(
            tmp_path, "late-1", {**row, "state": budget_pause.STATE_RESUME_GRANTED,
                                 "grant": grant, "resume_generation": 1},
            expected_pause_id=str(row["pause_id"]))
        persisted.append(reason)
        return True

    sctx.persist_queue_snapshot = _snapshot_then_grant
    _handle_budget_pause({**budget_pause.pause_event(task, pause), "worker_id": 0}, sctx)

    # The park itself still happened: the SAME task id is parked, never dropped.
    assert workers.RUNNING == {} and workers.PENDING[0]["id"] == "late-1"
    after = budget_pause.budget_pause_row(tmp_path, "late-1")
    assert after["state"] == budget_pause.STATE_RESUME_GRANTED
    assert after["grant"]["grant_id"] == "g-live"  # the live grant is intact
    assert "paused_confirmed_at" not in after
    assert pushed[-1]["type"] == "budget_pause_park_superseded"
    assert pushed[-1]["owner_visible"] is False and "toast_once" not in pushed[-1]
    assert pushed[-1]["park_state"] == budget_pause.STATE_RESUME_GRANTED
    # The owner-facing status projection belongs to the newer writer, not to us.
    assert load_task_result(tmp_path, "late-1", strict=True).get("reason_code") != "budget_paused"


def test_exact_pause_event_without_durable_row_is_refused(tmp_path, monkeypatch):
    from supervisor.events import _handle_budget_pause

    queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    _running_row(tmp_path, "ghost")
    workers.RUNNING["ghost"] = {"task": {"id": "ghost", "type": "task"}, "worker_id": 0, "attempt": 1}
    sctx = _supervisor_ctx(tmp_path, workers, queue, [], [])
    with pytest.raises(ValueError):
        _handle_budget_pause({"type": "budget_pause", "task_id": "ghost", "worker_id": 0,
                              "resource_limit": {"exact_continuation": True,
                                                 "checkpoint": {"pause_id": "nope"}}}, sctx)
    assert "ghost" in workers.RUNNING  # nothing moved without a record


def test_worker_death_during_pausing_completes_the_park_not_a_retry(tmp_path, monkeypatch):
    from ouroboros import budget_pause
    from supervisor import worker_health

    queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    ctx, _limit, pause = _pause(tmp_path, monkeypatch)
    budget_pause.end_dispatch_fence(ctx.task_id)
    task = {"id": ctx.task_id, "type": "task", "_attempt": 1, "budget_drive_root": str(tmp_path)}
    meta = {"task": task, "worker_id": 0, "attempt": 1}
    workers.RUNNING[ctx.task_id] = meta
    workers.WORKERS[0] = SimpleNamespace(busy_task_id=ctx.task_id)
    monkeypatch.setattr(worker_health, "_dead_job_is_current", lambda job: True)
    monkeypatch.setattr(queue, "persist_queue_snapshot", lambda reason="": True)
    job = {"worker": workers.WORKERS[0], "task_id": ctx.task_id, "task": task, "meta": meta,
           "worker_id": 0, "exitcode": 1, "drive_root": str(tmp_path)}
    assert worker_health._complete_exact_budget_pause_after_death(job, tmp_path, task, ctx.task_id, 1) == (True, True)
    assert workers.RUNNING == {} and workers.PENDING[0]["_budget_pause"]["exact_continuation"] is True
    assert budget_pause.budget_pause_row(tmp_path, ctx.task_id)["pause_source"] == "worker_death_during_pausing"
    # A different attempt (a retry that never saw the checkpoint) is neither
    # adopted nor fenced by this attempt's evidence.
    assert worker_health._complete_exact_budget_pause_after_death(job, tmp_path, task, ctx.task_id, 2) == (False, False)


def test_worker_death_holding_an_unconsumed_grant_reparks_instead_of_terminalizing(tmp_path, monkeypatch):
    """F4: a grant nothing consumed is revoked and the SAME task id returns to its
    exact pause. The loop writes ``consumed_at`` before any new effect, so an
    unconsumed grant proves the continuation never started — a refused
    continuation load kills the worker exactly here, and the saved pause must
    survive it. A CONSUMED grant is ordinary crash custody and never reopened."""
    from ouroboros import budget_pause
    from supervisor import worker_health

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda _st, **_k: 5.0)
    task, _row = _parked(tmp_path, monkeypatch, task_id="death-1")
    assert queue.resume_budget_paused_task("death-1")["ok"] is True
    grant_id = task["_budget_pause_resume"]["grant_id"]
    # Dispatched, then the worker dies before the loop could consume the grant.
    workers.PENDING.remove(task)
    meta = {"task": task, "worker_id": 0, "attempt": 1}
    workers.RUNNING["death-1"] = meta
    workers.WORKERS[0] = SimpleNamespace(busy_task_id="death-1")
    monkeypatch.setattr(worker_health, "_dead_job_is_current", lambda job: True)
    monkeypatch.setattr(queue, "persist_queue_snapshot", lambda reason="": True)
    job = {"worker": workers.WORKERS[0], "task_id": "death-1", "task": dict(task), "meta": meta,
           "worker_id": 0, "exitcode": 1, "drive_root": str(tmp_path)}
    assert worker_health._complete_exact_budget_pause_after_death(job, tmp_path, task, "death-1", 1) == (True, True)
    assert workers.RUNNING == {}
    parked = workers.PENDING[0]
    assert parked["id"] == "death-1" and parked["_budget_pause"]["exact_continuation"] is True
    assert "_budget_pause_resume" not in parked  # the spent handoff left with the park
    row = budget_pause.budget_pause_row(tmp_path, "death-1")
    assert row["state"] == budget_pause.STATE_PAUSED
    assert row["grant"]["grant_id"] == grant_id
    assert row["grant"]["revoke_reason"] == "worker_death_before_consumption"
    assert row["pause_source"] == "worker_death_before_grant_consumed"
    # The owner may Resume the same id again; the dead grant is dead for good.
    ctx, _limit = _loop_ctx(tmp_path, "death-1")
    with pytest.raises(ValueError):
        budget_pause.load_budget_pause(ctx, {"pause_id": row["pause_id"], "grant_id": grant_id})
    assert queue.resume_budget_paused_task("death-1")["ok"] is True


def test_worker_death_after_a_consumed_grant_is_not_reopened(tmp_path, monkeypatch):
    """The other half of F4: a consumed grant means the task RAN. Its death keeps
    the ordinary custody path — the pause is not re-armed over running work."""
    from ouroboros import budget_pause
    from supervisor import worker_health

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda _st, **_k: 5.0)
    task, _row = _parked(tmp_path, monkeypatch, task_id="death-2")
    assert queue.resume_budget_paused_task("death-2")["ok"] is True
    granted = budget_pause.budget_pause_row(tmp_path, "death-2")
    consumed = {**dict(granted["grant"]), "consumed_at": time.time()}
    budget_pause.set_budget_pause(
        tmp_path, "death-2", {**granted, "state": budget_pause.STATE_RESUMED, "grant": consumed},
        expected_pause_id=str(granted["pause_id"]),
        expected_state=budget_pause.STATE_RESUME_GRANTED,
        expected_grant_id=str(granted["grant"]["grant_id"]))
    workers.PENDING.remove(task)
    meta = {"task": task, "worker_id": 0, "attempt": 1}
    workers.RUNNING["death-2"] = meta
    workers.WORKERS[0] = SimpleNamespace(busy_task_id="death-2")
    monkeypatch.setattr(worker_health, "_dead_job_is_current", lambda job: True)
    job = {"worker": workers.WORKERS[0], "task_id": "death-2", "task": dict(task), "meta": meta,
           "worker_id": 0, "exitcode": 1, "drive_root": str(tmp_path)}
    # Not re-armed, but the consumed grant still fences the ordinary crash retry.
    assert worker_health._complete_exact_budget_pause_after_death(job, tmp_path, task, "death-2", 1) == (False, True)
    assert "death-2" in workers.RUNNING and workers.PENDING == []
    after = budget_pause.budget_pause_row(tmp_path, "death-2")
    assert after["state"] == budget_pause.STATE_RESUMED and not after["grant"].get("revoked_at")


# --------------------------------------------------------------------------- supervisor: resume

def test_resume_refuses_while_money_is_still_exhausted(tmp_path, monkeypatch):
    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    task, _row = _parked(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda _st, **_k: 0.0)
    result = queue.resume_budget_paused_task(task["id"])
    assert result["error"] == "budget_still_exhausted"
    assert "_budget_pause" in workers.PENDING[0] and "_budget_pause_resume" not in workers.PENDING[0]


def test_resume_refuses_cancel_intent_and_paused_root(tmp_path, monkeypatch):
    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda _st, **_k: 5.0)

    task, _row = _parked(tmp_path, monkeypatch)
    monkeypatch.setattr("ouroboros.cancel_intents.has_active_intent", lambda *_a, **_k: True)
    assert queue.resume_budget_paused_task(task["id"])["error"] == "cancel_intent_active"
    monkeypatch.setattr("ouroboros.cancel_intents.has_active_intent", lambda *_a, **_k: False)

    workers.PENDING[:] = []
    root, _r = _parked(tmp_path, monkeypatch, task_id="root-1", scope="root")
    child, _c = _parked(tmp_path, monkeypatch, task_id="child-1", root_task_id="root-1")
    refused = queue.resume_budget_paused_task("child-1")
    assert refused["error"] == "root_still_paused" and refused["action"] == "resume_root_first"


def test_root_resume_mints_single_use_grant_and_only_makes_children_eligible(tmp_path, monkeypatch):
    from ouroboros import budget_pause

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda _st, **_k: 5.0)
    root, root_row = _parked(tmp_path, monkeypatch, task_id="root-2", scope="root")
    child, _c = _parked(tmp_path, monkeypatch, task_id="child-2", root_task_id="root-2")
    # Backdate the pause so the paused interval is measurable and separate.
    budget_pause.set_budget_pause(tmp_path, "root-2", {**budget_pause.budget_pause_row(tmp_path, "root-2"),
                                                         "paused_at": time.time() - 30.0})

    granted = queue.resume_budget_paused_task("root-2")
    assert granted["ok"] is True and granted["exact_continuation"] is True
    assert granted["eligible_descendants"] == ["child-2"]
    assert granted["paused_duration_sec"] >= 29
    assert "root-2" not in queue.BUDGET_ROOT_FENCES
    handoff = root["_budget_pause_resume"]
    assert handoff["grant_id"] == granted["grant_id"] and handoff["pause"]["exact_continuation"] is True
    assert handoff["started_at"] == root_row["started_at"]  # never moved
    row = budget_pause.budget_pause_row(tmp_path, "root-2")
    assert row["state"] == budget_pause.STATE_RESUME_GRANTED and row["grant"]["single_use"] is True
    # The child stayed paused: eligibility is not release.
    assert "_budget_pause" in child and "_budget_pause_resume" not in child
    # Second grant for the same pause is refused while a queue carrier holds it
    # (single use). A stale carrier WITHOUT the handoff (crash between the durable
    # grant and the snapshot) is the orphan case, covered in
    # tests/test_budget_pause_astra_a882.py: it is revoked and re-granted.
    second = queue.resume_budget_paused_task("root-2")
    assert second["ok"] is False and second["error"] == "task_not_budget_paused"
    assert budget_pause.budget_pause_row(tmp_path, "root-2")["grant"]["grant_id"] == granted["grant_id"]
    # Now the child may be selected explicitly (Q9) under the resumed root.
    assert queue.resume_budget_paused_task("child-2")["ok"] is True


def test_root_grant_refuses_a_cached_tree_snapshot_and_admits_a_fresh_read(tmp_path, monkeypatch):
    """Negative then positive: a ROOT-scope grant reads the tree ledger NOW. A
    0-age cached snapshot beside a read that just failed refuses typed
    (``root_accounting_unavailable``); a working ledger admits, and spend at
    the actual root cap still refuses (``root_hard_cap_exhausted``)."""
    from ouroboros import usage_accounting

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda _st, **_k: 5.0)
    task, _row = _parked(tmp_path, monkeypatch, task_id="strict-root", scope="root")
    usage_accounting._stash_root_accounting("strict-root", {"settled_usd": 1.0, "accounted_usd": 1.0}, 10.0)
    monkeypatch.setattr(usage_accounting, "usage_projection",
                        lambda *_a, **_k: (_ for _ in ()).throw(OSError("ledger unavailable")))
    refused = queue.resume_budget_paused_task("strict-root")
    assert refused["error"] == "root_accounting_unavailable" and refused["action"] == "retry_or_cancel"
    assert "_budget_pause" in task and "_budget_pause_resume" not in task
    monkeypatch.setattr(usage_accounting, "usage_projection",
                        lambda *_a, **_k: {"settled_usd": 10.0, "accounted_usd": 10.0, "limit_usd": 10.0})
    assert queue.resume_budget_paused_task("strict-root")["error"] == "root_hard_cap_exhausted"
    # #1487: a $25 unresolved bound beside $1 of known spend is exposure, not spending:
    # explicit Resume within the authorized $10 is granted.
    monkeypatch.setattr(usage_accounting, "usage_projection",
                        lambda *_a, **_k: {"settled_usd": 1.0, "accounted_usd": 26.0, "limit_usd": 10.0})
    assert queue.resume_budget_paused_task("strict-root")["ok"] is True


def test_grant_is_revoked_when_money_vanishes_before_dispatch(tmp_path, monkeypatch):
    from ouroboros import budget_pause
    from supervisor.budget_resume import revoke_exact_budget_resume

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda _st, **_k: 5.0)
    task, _row = _parked(tmp_path, monkeypatch, task_id="revoke-1")
    assert queue.resume_budget_paused_task("revoke-1")["ok"] is True
    assert task["_budget_pause_resume"]["grant_generation"] == 1
    monkeypatch.setattr(state, "budget_remaining", lambda _st, **_k: 0.0)
    sent = []
    workers.WORKERS[0] = SimpleNamespace(wid=0, busy_task_id=None, reaping=False,
                                         in_q=SimpleNamespace(put=lambda t: sent.append(t)))
    workers.assign_tasks()
    assert sent == []
    assert task.get("_budget_pause", {}).get("exact_continuation") is True
    assert "_budget_pause_resume" not in task
    row = budget_pause.budget_pause_row(tmp_path, "revoke-1")
    assert row["state"] == budget_pause.STATE_PAUSED and row["grant"]["revoke_reason"] == "budget_exhausted_before_dispatch"
    assert revoke_exact_budget_resume(task, "again") is False  # nothing granted now
    # A stale copy of the revoked handoff can never revive the continuation.
    ctx, _limit = _loop_ctx(tmp_path, "revoke-1")
    with pytest.raises(ValueError):
        budget_pause.load_budget_pause(ctx, {"pause_id": row["pause_id"], "grant_id": row["grant"]["grant_id"]})


def test_granted_task_dispatches_with_original_started_at_and_paused_carrier(tmp_path, monkeypatch):
    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda _st, **_k: 5.0)
    task, row = _parked(tmp_path, monkeypatch, task_id="dispatch-1")
    assert queue.resume_budget_paused_task("dispatch-1")["ok"] is True
    sent = []
    workers.WORKERS[0] = SimpleNamespace(wid=0, busy_task_id=None, reaping=False,
                                         in_q=SimpleNamespace(put=lambda t: sent.append(dict(t))))
    workers.assign_tasks()
    assert [t["id"] for t in sent] == ["dispatch-1"]
    meta = workers.RUNNING["dispatch-1"]
    assert meta["started_at"] == pytest.approx(float(row["started_at"]))
    assert meta["budget_paused_sec"] > 0
    assert sent[0]["_budget_pause_resume"]["grant_id"]


def test_paused_row_survives_stale_snapshot_restore_without_waking(tmp_path, monkeypatch):
    queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    task, _row = _parked(tmp_path, monkeypatch, task_id="stale-1")
    queue.persist_queue_snapshot(reason="test")
    workers.PENDING[:] = []
    snap_path = queue.QUEUE_SNAPSHOT_PATH
    snap = json.loads(snap_path.read_text())
    snap["ts"] = "2000-01-01T00:00:00+00:00"  # far older than any freshness window
    snap_path.write_text(json.dumps(snap))
    assert queue.restore_pending_from_snapshot() == 1
    assert workers.PENDING[0]["id"] == "stale-1"
    assert workers.PENDING[0]["_budget_pause"]["exact_continuation"] is True  # still paused, not dispatched


def test_restore_returns_undispatched_grant_to_its_pause(tmp_path, monkeypatch):
    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda _st, **_k: 5.0)
    task, _row = _parked(tmp_path, monkeypatch, task_id="restart-grant")
    assert queue.resume_budget_paused_task("restart-grant")["ok"] is True
    queue.persist_queue_snapshot(reason="test")
    workers.PENDING[:] = []
    assert queue.restore_pending_from_snapshot() == 1
    restored = workers.PENDING[0]
    assert "_budget_pause_resume" not in restored and restored["_budget_pause"]["exact_continuation"] is True


# --------------------------------------------------------------------------- loop-side resume

def test_resume_consumes_grant_restores_cognition_and_never_reexecutes(tmp_path, monkeypatch):
    from ouroboros import budget_pause, owner_wait

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda _st, **_k: 5.0)
    from tests import _budget_pause_exact_helpers as helpers
    original_ctx = helpers._loop_ctx

    def with_sent_disclosure(*args, **kwargs):
        context, limits = original_ctx(*args, **kwargs)
        limits.accumulated_usage["request_wire"] = {"applied_effort": "low", "original_requested_effort": "high"}
        return context, limits

    monkeypatch.setattr(helpers, "_loop_ctx", with_sent_disclosure)
    task, row = _parked(tmp_path, monkeypatch, task_id="loop-1")
    assert queue.resume_budget_paused_task("loop-1")["ok"] is True
    handoff = task["_budget_pause_resume"]
    ctx, _limit = _loop_ctx(tmp_path, "loop-1")
    ctx.budget_pause_resume = handoff
    tools = SimpleNamespace(_ctx=ctx)
    state_blob = budget_pause.load_budget_pause(ctx)
    assert state_blob["pause_id"] == row["pause_id"]
    monkeypatch.setattr(owner_wait, "rebind_restored_route", lambda *_a, **_k: (None, "max"))
    messages, trace, usage, seen = [], {}, {}, set()
    model, effort, use_local, mode, round_idx, plan = budget_pause.resume_paused_loop(
        tools, state_blob, messages, trace, usage, seen, budget_remaining_usd=5.0)
    assert (model, round_idx, mode) == ("m", 4, "max")
    assert effort == "high"  # restored original preference, independent of old wire disclosures
    assert usage["request_wire"]["applied_effort"] == "low"
    assert usage["cost"] == 1.25 and "execution_status" not in usage
    # The unanswered call is closed as UNKNOWN, not re-run, not declared un-run.
    unknown = [m for m in messages if m.get("role") == "tool" and m.get("tool_call_id") == "call_b"]
    assert len(unknown) == 1 and "UNKNOWN" in unknown[0]["content"] and "NOT re-executed" in unknown[0]["content"]
    notice = messages[-1]["content"]
    assert "budget pause" in notice
    # Custody is re-observed FRESH at the grant (owner Q8) and that reading rides the
    # row: the disclosure names it, never the pause-time summary. This drive holds no
    # delegated run at Resume time, so the pause row's stale "run-1" must NOT resurface.
    assert "Delegated runs this task holds (re-observed at this Resume):\n- none" in notice
    assert "run-1" not in notice
    assert "never start a second writer" in notice
    fresh = budget_pause.budget_pause_row(tmp_path, "loop-1")["external_runs"]
    assert fresh["custody_read"] == "ok" and fresh["runs"] == []
    consumed = budget_pause.budget_pause_row(tmp_path, "loop-1")
    assert consumed["state"] == budget_pause.STATE_RESUMED and consumed["grant"]["consumed_at"]
    assert not budget_pause.dispatch_fenced("loop-1")
    # A hard rail keeps both wrap-up reservations; only a graceful rail relaxes (Q10).
    assert ctx._budget_resume_last_fit_relaxed is False
    assert usage["budget_pause_resume"]["last_fit_relaxed"] is False
    assert usage["budget_pause_resume"]["grant_generation"] == 1
    # The grant is single-use: a second load refuses.
    with pytest.raises(ValueError):
        budget_pause.load_budget_pause(ctx, handoff)


def test_resume_labels_a_pause_time_run_list_as_not_re_observed(tmp_path, monkeypatch):
    """The other branch of the same disclosure: when the row carries no fresh reading,
    the checkpoint's pause-time copy is disclosed and NAMED as un-re-observed history —
    a stale list must never read as a current one."""
    from ouroboros import budget_pause, owner_wait

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda _st, **_k: 5.0)
    task, row = _parked(tmp_path, monkeypatch, task_id="loop-2")
    assert queue.resume_budget_paused_task("loop-2")["ok"] is True
    ctx, _limit = _loop_ctx(tmp_path, "loop-2")
    ctx.budget_pause_resume = task["_budget_pause_resume"]
    state_blob = budget_pause.load_budget_pause(ctx)
    # Drop the grant's fresh observation, keeping the checkpoint's pause-time copy.
    state_blob["_pause_row"] = {k: v for k, v in state_blob["_pause_row"].items()
                               if k != "external_runs"}
    state_blob["external_runs"] = {"runs": [{"run_id": "run-1", "state": "stop_requested",
                                            "stop_outcome": "requested"}]}
    monkeypatch.setattr(owner_wait, "rebind_restored_route", lambda *_a, **_k: (None, "max"))
    messages = []
    budget_pause.resume_paused_loop(SimpleNamespace(_ctx=ctx), state_blob, messages, {}, {}, set(),
                                   budget_remaining_usd=5.0)
    notice = messages[-1]["content"]
    assert "(as recorded at the pause, NOT re-observed)" in notice
    assert "run-1: stop_requested" in notice
    assert "never start a second writer" in notice


def test_load_refuses_foreign_or_missing_grant(tmp_path, monkeypatch):
    from ouroboros import budget_pause

    ctx, _limit, pause = _pause(tmp_path, monkeypatch, task_id="grant-1")
    budget_pause.end_dispatch_fence("grant-1")
    with pytest.raises(ValueError):
        budget_pause.load_budget_pause(ctx, {"pause_id": pause["pause_id"], "grant_id": "forged"})


def test_graceful_rail_refreshes_planning_threshold_within_authorized_money(tmp_path, monkeypatch):
    from ouroboros import budget_pause, task_pacing
    from ouroboros import loop_budget

    ctx, _limit = _loop_ctx(tmp_path)
    ctx._cost_ceiling = task_pacing.CostCeiling(state="active", ceiling_usd=7.0, root_cap_usd=10.0,
                                                planning_margin_usd=3.0, basis="root_cap_minus_margin")
    ctx._accumulated_usage = {"cost": 8.0}
    # Every number is read from the AUTHORITATIVE ledger at Resume time: a fresh
    # wallet observation and a fresh, undegraded root-accounting read.
    monkeypatch.setattr(loop_budget, "_wrapup_global_remaining", lambda: 100.0)
    monkeypatch.setattr(loop_budget, "_loop_tree_accounting",
                        lambda **_k: {"settled_usd": 8.0, "accounted_usd": 8.0, "age_sec": 0.0})
    monkeypatch.setattr(task_pacing, "resolve_budget_profile", lambda _c: {"cost_hard_stop_pct": 50})
    disclosure = budget_pause._refresh_planning_threshold(ctx, budget_remaining_usd=100.0)
    assert disclosure["refreshed"] is True
    # The wallet is the ledger projection, never the dispatch-time number.
    assert disclosure["wallet_basis"] == "ledger_projection"
    assert disclosure["global_remaining_usd"] == 100.0
    # This tree read carries no cap of its own, so the start-of-task cap stands and
    # its provenance is disclosed rather than assumed.
    assert disclosure["root_cap_usd"] == 10.0 and disclosure["root_cap_basis"] == "start_of_task"
    # min(cap - spent = 2, 50% of global remaining = 50) added on top of spend: no immediate re-pause.
    assert ctx._cost_ceiling.ceiling_usd == pytest.approx(10.0)
    assert ctx._cost_ceiling.root_cap_usd == 10.0 and ctx._cost_ceiling.basis.startswith("owner_resume_refresh")
    # The hard tree cap is untouched by the refresh: spend AT the cap leaves no
    # authorized room, and the owner's explicit act cannot invent any.
    ctx._accumulated_usage = {"cost": 10.0}
    monkeypatch.setattr(loop_budget, "_loop_tree_accounting",
                        lambda **_k: {"settled_usd": 10.0, "accounted_usd": 10.0, "age_sec": 0.0})
    spent = budget_pause._refresh_planning_threshold(ctx, budget_remaining_usd=100.0)
    assert spent["refreshed"] is False and spent["reason"] == "no_authorized_room"


def test_threshold_refresh_refuses_every_unknown_or_stale_money_fact(tmp_path, monkeypatch):
    """Negative (owner Q10): unknown money is NOT room. A wallet the ledger cannot
    answer, a degraded or stale tree read and unknown tree spend each REFUSE the
    refresh, and the dispatch-time number is only ever DISCLOSED, never spent."""
    from ouroboros import budget_pause, loop_budget, task_pacing

    ctx, _limit = _loop_ctx(tmp_path)

    def _ceiling():
        return task_pacing.CostCeiling(state="active", ceiling_usd=7.0, root_cap_usd=10.0,
                                       planning_margin_usd=3.0, basis="root_cap_minus_margin")

    monkeypatch.setattr(task_pacing, "resolve_budget_profile", lambda _c: {"cost_hard_stop_pct": 50})
    # No ceiling at all: there is no threshold to move.
    ctx._cost_ceiling = None
    assert budget_pause._refresh_planning_threshold(ctx, budget_remaining_usd=100.0) == {
        "refreshed": False, "reason": "no_ceiling"}
    # The ledger cannot answer the wallet: the dispatch-time value is DISCLOSED only.
    ctx._cost_ceiling = _ceiling()
    ctx._accumulated_usage = {"cost": 8.0}
    monkeypatch.setattr(loop_budget, "_wrapup_global_remaining", lambda: None)
    monkeypatch.setattr(loop_budget, "_loop_tree_accounting",
                        lambda **_k: {"settled_usd": 8.0, "accounted_usd": 8.0, "age_sec": 0.0})
    assert budget_pause._refresh_planning_threshold(ctx, budget_remaining_usd=100.0) == {
        "refreshed": False, "reason": "wallet_unavailable", "wallet_basis": "ledger_unavailable",
        "dispatch_time_remaining_usd": 100.0}
    assert ctx._cost_ceiling.ceiling_usd == 7.0  # the paused threshold is untouched
    # With a fresh wallet, every unusable TREE read still refuses.
    monkeypatch.setattr(loop_budget, "_wrapup_global_remaining", lambda: 100.0)
    for tree, reason in (
        (None, "tree_spend_unavailable"),
        ({"settled_usd": 8.0, "accounted_usd": 8.0, "age_sec": 0.0, "integrity_degraded": True}, "tree_accounting_degraded"),
        ({"settled_usd": None, "accounted_usd": None, "age_sec": 0.0}, "tree_spend_unknown"),
    ):
        ctx._cost_ceiling = _ceiling()
        monkeypatch.setattr(loop_budget, "_loop_tree_accounting", lambda _t=tree, **_k: _t)
        refused = budget_pause._refresh_planning_threshold(ctx, budget_remaining_usd=100.0)
        assert refused["refreshed"] is False and refused["reason"] == reason
        assert ctx._cost_ceiling.ceiling_usd == 7.0


def test_threshold_refresh_reads_the_ledger_now_never_a_cached_snapshot(tmp_path, monkeypatch):
    """Negative then positive (owner Q10): the tree read is STRICT. A root snapshot
    cached a moment ago (an earlier display refresh, a reservation) is not room
    when the ledger cannot answer NOW; the same call with a working ledger reads
    the fresh number and moves the threshold within it."""
    from ouroboros import budget_pause, loop_budget, task_pacing, usage_accounting
    from ouroboros.usage_accounting import UsageScope, usage_scope

    ctx, _limit = _loop_ctx(tmp_path)
    ctx._cost_ceiling = task_pacing.CostCeiling(state="active", ceiling_usd=7.0, root_cap_usd=10.0,
                                                planning_margin_usd=3.0, basis="root_cap_minus_margin")
    ctx._accumulated_usage = {"cost": 8.0}
    monkeypatch.setattr(task_pacing, "resolve_budget_profile", lambda _c: {"cost_hard_stop_pct": 50})
    monkeypatch.setattr(loop_budget, "_wrapup_global_remaining", lambda: 100.0)
    usage_accounting._stash_root_accounting("q10-root", {"settled_usd": 8.0, "accounted_usd": 8.0}, 10.0)  # fresh, 0-age display cache
    monkeypatch.setattr(usage_accounting, "usage_projection",
                        lambda *_a, **_k: (_ for _ in ()).throw(OSError("ledger unavailable")))
    with usage_scope(UsageScope(drive_root=tmp_path, task_id="q10-task", root_task_id="q10-root")):
        refused = budget_pause._refresh_planning_threshold(ctx, budget_remaining_usd=100.0)
        assert refused == {"refreshed": False, "reason": "tree_spend_unavailable", "wallet_basis": "ledger_projection"}
        assert ctx._cost_ceiling.ceiling_usd == 7.0
        monkeypatch.setattr(usage_accounting, "usage_projection",
                            lambda *_a, **_k: {"settled_usd": 8.0, "accounted_usd": 8.0, "limit_usd": 10.0})
        granted = budget_pause._refresh_planning_threshold(ctx, budget_remaining_usd=100.0)
    assert granted["refreshed"] is True and granted["root_cap_basis"] == "root_accounting"
    assert ctx._cost_ceiling.ceiling_usd == pytest.approx(10.0)


# --------------------------------------------------------------------------- gateway / UI facts

def test_state_phase_budget_pausing_reads_the_durable_row(tmp_path, monkeypatch):
    from ouroboros import budget_pause
    from ouroboros.gateway import state as gw_state

    ctx, _limit, _pause_row = _pause(tmp_path, monkeypatch, task_id="phase-1")
    budget_pause.end_dispatch_fence("phase-1")
    row = {"_attempt": 1, "budget_drive_root": str(tmp_path)}
    assert gw_state._managed_task_budget_pausing(tmp_path, row, "phase-1") is True
    assert gw_state._managed_task_budget_pausing(tmp_path, {"_attempt": 2}, "phase-1") is False
    assert gw_state._managed_task_budget_pausing(tmp_path, {"_attempt": 1}, "absent") is False


def test_resume_child_tool_only_targets_own_children(monkeypatch):
    from ouroboros.tools import join_ledger

    ctx = SimpleNamespace(task_id="parent-1", task_metadata={})
    monkeypatch.setattr(join_ledger, "_status_drive_root", lambda _c: pathlib.Path("/tmp"))
    monkeypatch.setattr(join_ledger, "_is_own_child", lambda _c, _r, tid, **_kw: tid == "child-x")
    monkeypatch.setattr(join_ledger, "_publish_tool_result", lambda _c, result: result.text)
    monkeypatch.setattr(join_ledger, "_record_child_decision_beacon", lambda *_a, **_k: None)
    emitted = []
    monkeypatch.setattr("ouroboros.tools.control._emit_control_event",
                        lambda _c, evt: (emitted.append(evt) or "live"))
    assert "not a child" in join_ledger._resume_child_task(ctx, "stranger-1", "why")
    text = join_ledger._resume_child_task(ctx, "child-x", "still needed")
    assert "Resume requested" in text and "REQUEST" in text
    assert emitted[0]["type"] == "budget_resume_child" and emitted[0]["requested_by"] == "parent-1"


def test_resume_child_task_is_policy_covered_and_exposed_beside_its_own_family():
    """F7: the Q9 selection verb was registered but named nowhere else — it fell
    through to the default LLM safety check, was invisible in the round-one
    envelope and to delegated children (who must select their OWN paused
    children), and was not withheld from a consciousness wake at Observe, which
    may not start work. It is declared beside ``cancel_task``, the verb whose
    authority it mirrors; the supervisor still re-checks lineage and the root's
    live grant, so no owner authority is widened."""
    from ouroboros import safety
    from ouroboros import tool_capabilities as caps
    from ouroboros.consciousness_authority import disabled_tools_for
    from ouroboros.tools import join_ledger

    assert any(entry.name == "resume_child_task" for entry in join_ledger.get_tools())
    assert safety.TOOL_POLICY["resume_child_task"] == safety.POLICY_SKIP
    for names in (caps.CORE_TOOL_NAMES, caps.LOCAL_READONLY_SUBAGENT_TOOL_NAMES,
                  caps.ACTING_SUBAGENT_TOOL_NAMES):
        assert "cancel_task" in names  # the family it belongs to
        assert "resume_child_task" in names
    # It STARTS work, so an Observe-level wake does without it (В10').
    assert "resume_child_task" in caps.OBSERVE_WORLD_MUTATION_TOOLS
    assert "resume_child_task" in disabled_tools_for("observe")
    assert "resume_child_task" not in disabled_tools_for("full")


def _repo_file(*parts):
    return pathlib.Path(__file__).resolve().parents[1].joinpath(*parts).read_text()


def test_activity_rows_show_a_held_budget_row_as_paused_not_queued():
    """Scope note (static pin; the browser check is the parent's): a row whose root
    fence was lifted carries an unselected HOLD and nothing will dispatch it, so
    listing it as plain "queued" promises work that cannot start."""
    source = _repo_file("web", "modules", "activity.js")
    assert "_budget_pause_hold" in source and "heldRow" in source
    assert "|| heldRow(t)" in source  # consulted by the pending-row pause predicate


def test_runbook_does_not_promise_a_managed_outage_window_the_runtime_has_no_rail_for():
    """F8: with no deadline and an unlimited absolute ceiling, a managed task's
    transport-outage episode has NO window of its own — the 6h operation-window
    fallback belongs to other operations. The runbook names the optional rails an
    operator can set instead of promising a timeout that does not exist."""
    source = _repo_file("devtools", "benchmarks", "continual_learning", "RUNBOOK.md")
    assert "6h operation window from episode entry" not in source
    assert "OUROBOROS_TASK_ABS_CEILING_SEC" in source and "idle reaper" in source


# --------------------------------------------------------------------------- consumption publication holds

def _granted_loop(tmp_path, monkeypatch, task_id):
    from ouroboros import budget_pause, owner_wait

    queue, state, _workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda _st, **_k: 5.0)
    task, row = _parked(tmp_path, monkeypatch, task_id=task_id)
    assert queue.resume_budget_paused_task(task_id)["ok"] is True
    ctx, _limit = _loop_ctx(tmp_path, task_id)
    ctx.budget_pause_resume = task["_budget_pause_resume"]
    state_blob = budget_pause.load_budget_pause(ctx)
    monkeypatch.setattr(owner_wait, "rebind_restored_route", lambda *_a, **_k: (None, "max"))
    return budget_pause, ctx, state_blob, row


def test_failed_grant_consumption_publication_holds_then_consumes_never_terminalizes(tmp_path, monkeypatch):
    """#1196 review finding 2: a raise from the state=resumed/consumed_at write used
    to leave ``resume_paused_loop`` on the generic loop exception path and end in
    FAILED (``_task_exception_terminal``). A publication that fails HOLDS — typed,
    nonterminal, retrying the SAME compare-and-set — and consumes once it lands."""
    budget_pause, ctx, state_blob, _row = _granted_loop(tmp_path, monkeypatch, "consume-hold")
    real = budget_pause.set_budget_pause
    failures = {"left": 2}

    def flaky(*args, **kwargs):
        if failures["left"] > 0:
            failures["left"] -= 1
            raise OSError("disk full")
        return real(*args, **kwargs)

    monkeypatch.setattr(budget_pause, "set_budget_pause", flaky)
    published = []
    monkeypatch.setattr(budget_pause, "_publish_hold", lambda _ctx, hold: published.append(dict(hold)))
    usage = {}
    budget_pause.resume_paused_loop(SimpleNamespace(_ctx=ctx), state_blob, [], {}, usage, set(),
                                   budget_remaining_usd=5.0)
    consumed = budget_pause.budget_pause_row(tmp_path, "consume-hold")
    assert consumed["state"] == budget_pause.STATE_RESUMED and consumed["grant"]["consumed_at"]
    assert "budget_pause_hold" not in usage and "execution_status" not in usage
    assert [hold["hold_reason"] for hold in published] == [budget_pause.HOLD_GRANT_CONSUMPTION_UNWRITABLE]
    assert published[0]["state"] == budget_pause.STATE_RESUME_GRANTED and "OSError" in published[0]["error"]


def test_grant_consumption_hold_ended_by_control_retains_the_grant_and_checkpoint(tmp_path, monkeypatch):
    from ouroboros.model_wait import ModelWaitInterrupted
    from tests._budget_pause_exact_helpers import _controls

    budget_pause, ctx, state_blob, row = _granted_loop(tmp_path, monkeypatch, "consume-stop")

    def unwritable(*_args, **_kwargs):
        raise OSError("read-only drive")

    monkeypatch.setattr(budget_pause, "set_budget_pause", unwritable)
    monkeypatch.setattr(budget_pause, "_hold_control_reason", _controls("", "deadline"))
    usage = {}
    with pytest.raises(ModelWaitInterrupted) as raised:
        budget_pause.resume_paused_loop(SimpleNamespace(_ctx=ctx), state_blob, [], {}, usage, set(),
                                       budget_remaining_usd=5.0)
    assert raised.value.control_reason == "deadline"
    durable = budget_pause.budget_pause_row(tmp_path, "consume-stop")
    grant = state_blob["_pause_row"]["grant"]
    # Grant and checkpoint identities are exactly what the row carried: nothing consumed, nothing rewritten.
    assert durable["state"] == budget_pause.STATE_RESUME_GRANTED
    assert durable["grant"]["grant_id"] == grant["grant_id"] and not durable["grant"].get("consumed_at")
    assert durable["source_ref"] == row["source_ref"] and durable["pause_id"] == row["pause_id"]
    assert usage["budget_pause_hold"]["hold_reason"] == budget_pause.HOLD_GRANT_CONSUMPTION_UNWRITABLE
    assert usage["budget_pause_hold"]["ended_by"] == "deadline"
    assert usage["exact_pause_unavailable"] == "hold_ended_by_control"


def test_grant_consumption_over_a_revoked_grant_reparks_under_the_live_pause(tmp_path, monkeypatch):
    budget_pause, ctx, state_blob, row = _granted_loop(tmp_path, monkeypatch, "consume-revoked")
    live = budget_pause.budget_pause_row(tmp_path, "consume-revoked")
    budget_pause.set_budget_pause(tmp_path, "consume-revoked", {
        **live, "state": budget_pause.STATE_PAUSED, "grant": {**live["grant"], "revoked_at": time.time()}})
    usage = {}
    with pytest.raises(budget_pause.BudgetPauseRequested) as raised:
        budget_pause.resume_paused_loop(SimpleNamespace(_ctx=ctx), state_blob, [], {}, usage, set(),
                                       budget_remaining_usd=5.0)
    assert raised.value.pause["pause_id"] == row["pause_id"]
    assert raised.value.pause["state"] == budget_pause.STATE_PAUSED and "budget_pause_hold" not in usage


@pytest.mark.parametrize("stamp", ["", "not-a-timestamp"])
def test_paused_row_survives_a_snapshot_with_an_invalid_timestamp_while_ordinary_rows_do_not(
        tmp_path, monkeypatch, stamp):
    """#1196 review finding 6: a readable snapshot with valid paused rows but a
    missing/invalid ``ts`` returned zero before ``_retain_snapshot_pending`` ran,
    so no paused carrier was restored and Resume answered task_not_pending over
    an intact durable checkpoint. Timestamp validity gates ORDINARY rows only."""
    queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    _parked(tmp_path, monkeypatch, task_id="ts-paused")
    workers.PENDING.append({"id": "ts-ordinary", "type": "task", "chat_id": 0, "_attempt": 1, "text": "x"})
    queue.persist_queue_snapshot(reason="test")
    workers.PENDING[:] = []
    snap = json.loads(queue.QUEUE_SNAPSHOT_PATH.read_text())
    if stamp:
        snap["ts"] = stamp
    else:
        snap.pop("ts", None)
    queue.QUEUE_SNAPSHOT_PATH.write_text(json.dumps(snap))
    assert queue.restore_pending_from_snapshot() == 1
    assert [task["id"] for task in workers.PENDING] == ["ts-paused"]
    assert workers.PENDING[0]["_budget_pause"]["exact_continuation"] is True  # retained, not woken
    rows = [json.loads(line) for line in (tmp_path / "logs" / "supervisor.jsonl").read_text().splitlines()]
    assert any(row["type"] == "queue_restore_snapshot_timestamp_invalid" for row in rows)


# --------------------------------------------------------------------------- public status plane

def test_public_status_of_a_paused_forked_root_is_the_pause_not_the_replicas_running(tmp_path, monkeypatch):
    """S30's public red (#1196): a forked root's worker mirror (its child drive's
    ``running`` row, blank cost planes) was overlaid on the canonical ``scheduled``
    row, and the queue merge preferred ``running`` over the PENDING carrier. The
    canonical row carrying a LIVE pause owns its lifecycle and accounting planes
    (the park writes the ledger-derived cost fields onto it), and a PENDING row
    parked under a ``_budget_pause`` marker is never ``running``."""
    from ouroboros import budget_pause
    from ouroboros.task_results import STATUS_RUNNING, STATUS_SCHEDULED, load_task_result, write_task_result
    from ouroboros.task_status import load_effective_task_result
    from supervisor.events import _handle_budget_pause

    queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    child_drive = tmp_path / "state" / "headless_tasks" / "forked-1" / "data"
    # The canonical row of a forked root names its child drive (agent.py binds it at start).
    write_task_result(tmp_path, "forked-1", STATUS_RUNNING, child_drive_root=str(child_drive),
                      budget_drive_root=str(tmp_path), result="Task is running.")
    _ctx, _limit, pause = _pause(tmp_path, monkeypatch, task_id="forked-1")
    budget_pause.end_dispatch_fence("forked-1")
    task = {"id": "forked-1", "type": "task", "chat_id": 0, "root_task_id": "forked-1", "_attempt": 1,
            "budget_drive_root": str(tmp_path), "child_drive_root": str(child_drive)}
    workers.RUNNING["forked-1"] = {"task": task, "worker_id": 0, "attempt": 1}
    workers.WORKERS[0] = SimpleNamespace(busy_task_id="forked-1")
    sctx = _supervisor_ctx(tmp_path, workers, queue, [], [])
    sctx.persist_queue_snapshot = queue.persist_queue_snapshot  # the REAL snapshot: the reader opens it
    _handle_budget_pause({**budget_pause.pause_event(task, pause), "worker_id": 0}, sctx)
    workers.WORKERS.clear()
    # The worker's pre-pause mirror on the child drive: still ``running``, blank cost planes.
    write_task_result(child_drive, "forked-1", STATUS_RUNNING, result="Task is running.",
                      total_rounds=None, accounted_upper_bound_usd=None)
    authority = load_task_result(tmp_path, "forked-1")
    assert authority["status"] == STATUS_SCHEDULED and authority["reason_code"] == budget_pause.REASON_CODE
    # Written at the park from the ledger: an int, never the mirror's None.
    assert authority["total_rounds"] == 0 and authority["cost_accounting_status"] == "available"
    assert load_task_result(child_drive, "forked-1")["status"] == STATUS_RUNNING  # the mirror IS stale
    for materialize in (False, True):
        effective = load_effective_task_result(tmp_path, "forked-1", materialize_artifacts=materialize)
        assert effective["status"] == STATUS_SCHEDULED, (materialize, effective["status"])
        assert effective["reason_code"] == budget_pause.REASON_CODE
        assert effective["resource_limit"]["status"] == budget_pause.STATUS_PAUSED_EXACT
        assert effective["total_rounds"] == 0 and effective["cost_accounting_status"] == "available"
        assert effective["budget_pause"]["state"] == budget_pause.STATE_PAUSED
        assert "paused exactly" in effective["result"]
    # The queue merge alone (the window between the persisted snapshot and the
    # status write, or a stale ``running`` authority mirror): the PENDING carrier
    # under its ``_budget_pause`` marker is not running.
    write_task_result(tmp_path, "forked-1", STATUS_RUNNING)  # forward progress, not a blocked regression
    assert load_task_result(tmp_path, "forked-1")["status"] == STATUS_RUNNING
    assert load_effective_task_result(tmp_path, "forked-1", materialize_artifacts=False)["status"] == STATUS_SCHEDULED
    # Terminal still wins over the carrier, as before.
    from ouroboros.task_status import _merge_queue_status

    assert _merge_queue_status("completed", "scheduled", workers.PENDING[0]) == "completed"
    assert _merge_queue_status("running", "scheduled", {"id": "plain"}) == "running"  # the requeue race, unchanged
    # Astra run-e0bb1ca3487c A2/A3: only a TYPED marker (bool exact_continuation)
    # is a pause; {} or a truthy string is malformed metadata, not a pause.
    for malformed in ({"_budget_pause": {}}, {"_budget_pause": {"exact_continuation": "false"}},
                      {"_budget_pause": "paused"}):
        assert _merge_queue_status("running", "scheduled", {"id": "x", **malformed}) == "running", malformed
    assert _merge_queue_status("running", "scheduled", {"id": "x", "_budget_pause": {"exact_continuation": False}}) == "scheduled"
    from supervisor.queue_snapshot import _exact_pause_row

    assert _exact_pause_row({"_budget_pause": {"exact_continuation": True}})
    assert not _exact_pause_row({"_budget_pause": {"exact_continuation": "false"}})
    assert not _exact_pause_row({"_budget_pause": {"exact_continuation": 1}})
    # Astra C1: a replica that already reached a TERMINAL status is the child's
    # real outcome; a canonical pause the copyback has not yet cleared never
    # hides it (only the stale nonterminal ``running`` mirror yields).
    write_task_result(tmp_path, "forked-1", STATUS_SCHEDULED, reason_code=budget_pause.REASON_CODE)
    write_task_result(child_drive, "forked-1", "completed", result="FINAL ANSWER: done", error="")
    effective = load_effective_task_result(tmp_path, "forked-1", materialize_artifacts=False)
    assert effective["status"] == "completed" and effective["result"] == "FINAL ANSWER: done"


def test_park_publishes_an_unavailable_cost_projection_over_stale_amounts(tmp_path, monkeypatch):
    """Astra run-e0bb1ca3487c A4: a failed ledger reconstruction returns the
    explicit unavailable projection; the park publishes it too, so an older
    ``available`` amount cannot survive the failed refresh merged in."""
    from ouroboros import budget_pause
    from ouroboros.task_results import STATUS_RUNNING, load_task_result, write_task_result
    from supervisor import events_budget
    from supervisor.events import _handle_budget_pause

    queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    write_task_result(tmp_path, "stale-1", STATUS_RUNNING, budget_drive_root=str(tmp_path),
                      accounted_upper_bound_usd=7.5, total_rounds=9, cost_accounting_status="available")
    _ctx, _limit, pause = _pause(tmp_path, monkeypatch, task_id="stale-1")
    budget_pause.end_dispatch_fence("stale-1")
    from supervisor import state as sup_state

    def _broken(task_id, fields=False, drive_root=None):
        assert fields
        return {"cost_accounting_status": "unavailable", "cost_final": False,
                "cost_accounting_error": "ledger_unavailable", "accounted_upper_bound_usd": None,
                "total_rounds": None, "ledger_integrity_degraded": True}

    monkeypatch.setattr(sup_state, "reconstruct_task_cost", _broken)
    task = {"id": "stale-1", "type": "task", "chat_id": 0, "root_task_id": "stale-1", "_attempt": 1,
            "budget_drive_root": str(tmp_path)}
    workers.RUNNING["stale-1"] = {"task": task, "worker_id": 0, "attempt": 1}
    workers.WORKERS[0] = SimpleNamespace(busy_task_id="stale-1")
    sctx = _supervisor_ctx(tmp_path, workers, queue, [], [])
    _handle_budget_pause({**budget_pause.pause_event(task, pause), "worker_id": 0}, sctx)
    workers.WORKERS.clear()
    row = load_task_result(tmp_path, "stale-1")
    assert row["reason_code"] == budget_pause.REASON_CODE, events_budget
    assert row["cost_accounting_status"] == "unavailable"
    assert row["accounted_upper_bound_usd"] is None and row["total_rounds"] is None


# --------------------------------------------------------------------------- shutdown: the pause survives the epoch

SERVER_SHUTDOWN_REASON = ("Server shut down (external stop/restart signal) before this task "
                          "finished; the task was interrupted, not a worker crash.")


def _shutdown_pool(workers, monkeypatch):
    """Bind the pool's terminal ``task_done`` bus to a list (no real event queue)."""
    events: list = []
    monkeypatch.setattr(workers, "_EVENT_Q_SHUTDOWN", False, raising=False)
    monkeypatch.setattr(workers, "get_event_q", lambda: SimpleNamespace(put=events.append), raising=False)
    return events


def test_graceful_shutdown_leaves_an_exact_pause_carrier_and_its_row_untouched(tmp_path, monkeypatch):
    """The lifespan teardown's ``kill_workers`` (no ``preserve_pending`` outside a
    managed update) drains PENDING and cancels every row as interrupted work. An
    exact mid-run pause is saved work of the same id, not interrupted work: its
    carrier stays PENDING with its marker, its durable row stays ``paused``, no
    task_done is published, the final snapshot still carries it, and the next
    boot parks the same id again, unheld (#1196: a pause survives the epoch)."""
    from ouroboros import budget_pause
    from ouroboros.task_results import STATUS_CANCELLED, STATUS_SCHEDULED, load_task_result, write_task_result
    from supervisor.events_budget import budget_hold_fact

    queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    events = _shutdown_pool(workers, monkeypatch)
    carrier, _row = _parked(tmp_path, monkeypatch, task_id="epoch-paused")
    write_task_result(tmp_path, "epoch-paused", STATUS_SCHEDULED, reason_code=budget_pause.REASON_CODE)
    write_task_result(tmp_path, "epoch-plain", STATUS_SCHEDULED)
    workers.PENDING.append({"id": "epoch-plain", "type": "task", "chat_id": 0, "_attempt": 1, "text": "x"})

    workers.kill_workers(force=True, terminal_status="cancelled", result_reason=SERVER_SHUTDOWN_REASON)

    assert [row["id"] for row in workers.PENDING] == ["epoch-paused"]
    kept = workers.PENDING[0]
    assert kept["_budget_pause"]["exact_continuation"] is True and not kept.get("_terminalization_retry")
    row = load_task_result(tmp_path, "epoch-paused")
    assert row["status"] == STATUS_SCHEDULED and row["budget_pause"]["state"] == budget_pause.STATE_PAUSED
    assert row["budget_pause"]["pause_id"] == carrier["_budget_pause"]["checkpoint"]["pause_id"]
    assert [e["task_id"] for e in events if e.get("type") == "task_done"] == ["epoch-plain"]
    assert load_task_result(tmp_path, "epoch-plain")["status"] == STATUS_CANCELLED
    snap = json.loads(queue.QUEUE_SNAPSHOT_PATH.read_text())
    assert [r["id"] for r in snap["pending"]] == ["epoch-paused"] and snap["running"] == []
    assert snap["pending"][0]["task"]["_budget_pause"]["exact_continuation"] is True
    rows = [json.loads(line) for line in (tmp_path / "logs" / "supervisor.jsonl").read_text().splitlines()]
    cleanup = [r for r in rows if r["type"] == "zombie_prevention_cleanup"][-1]
    assert cleanup["drained_pending"] == ["epoch-plain"] and cleanup["retained_budget_paused"] == ["epoch-paused"]
    # The next boot: the restore re-validates the durable authority and parks the same id, unheld.
    workers.PENDING[:] = []
    assert queue.restore_pending_from_snapshot() == 1
    restored = workers.PENDING[0]
    assert restored["id"] == "epoch-paused" and restored["_budget_pause"]["exact_continuation"] is True
    assert budget_hold_fact(restored) is None
    assert load_task_result(tmp_path, "epoch-paused")["budget_pause"]["state"] == budget_pause.STATE_PAUSED


def test_planned_shutdown_keeps_a_paused_child_of_an_interrupted_root_and_drops_its_unstarted_sibling(
        tmp_path, monkeypatch):
    """``kill_workers(preserve_pending=True)`` cancels the never-started children of
    an interrupted root (``pending_parent_interrupted``). A child parked under its
    exact pause is saved work, not a child that did not start: it stays."""
    from ouroboros import budget_pause
    from ouroboros.task_results import STATUS_CANCELLED, STATUS_RUNNING, STATUS_SCHEDULED, load_task_result, write_task_result

    queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    events = _shutdown_pool(workers, monkeypatch)
    write_task_result(tmp_path, "root-live", STATUS_RUNNING, chat_id=0)
    workers.RUNNING["root-live"] = {"task": {"id": "root-live", "type": "task", "chat_id": 0,
                                             "root_task_id": "root-live"}, "worker_id": 0, "attempt": 1}
    _parked(tmp_path, monkeypatch, task_id="child-paused", root_task_id="root-live",
            extra={"parent_task_id": "root-live", "delegation_role": "subagent"})
    write_task_result(tmp_path, "child-fresh", STATUS_SCHEDULED, chat_id=0)
    workers.PENDING.append({"id": "child-fresh", "type": "task", "chat_id": 0, "_attempt": 1, "text": "x",
                            "parent_task_id": "root-live", "root_task_id": "root-live"})

    workers.kill_workers(force=True, terminal_status="cancelled", result_reason=SERVER_SHUTDOWN_REASON,
                         preserve_pending=True)

    assert [row["id"] for row in workers.PENDING] == ["child-paused"]
    assert workers.PENDING[0]["_budget_pause"]["exact_continuation"] is True
    assert not workers.PENDING[0].get("_terminalization_retry")
    assert load_task_result(tmp_path, "root-live")["status"] == STATUS_CANCELLED
    assert load_task_result(tmp_path, "child-fresh")["status"] == STATUS_CANCELLED
    assert load_task_result(tmp_path, "child-paused")["budget_pause"]["state"] == budget_pause.STATE_PAUSED
    assert sorted(e["task_id"] for e in events if e.get("type") == "task_done") == ["child-fresh", "root-live"]


def test_unplanned_stop_restore_keeps_a_paused_child_of_a_fenced_root_and_marks_its_unstarted_sibling(
        tmp_path, monkeypatch):
    """The unplanned door (#1104): restore fences the snapshot's RUNNING root and
    hands its never-started PENDING children the ``pending_parent_interrupted``
    marker. The paused child keeps its exact marker instead, unmarked and unheld."""
    from ouroboros.task_results import STATUS_RUNNING, write_task_result
    from supervisor.events_budget import budget_hold_fact

    queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    write_task_result(tmp_path, "root-live", STATUS_RUNNING, chat_id=0)
    _parked(tmp_path, monkeypatch, task_id="child-paused", root_task_id="root-live",
            extra={"parent_task_id": "root-live", "delegation_role": "subagent"})
    write_task_result(tmp_path, "child-fresh", "scheduled", chat_id=0)
    workers.PENDING.append({"id": "child-fresh", "type": "task", "chat_id": 0, "_attempt": 1, "text": "x",
                            "parent_task_id": "root-live", "root_task_id": "root-live"})
    queue.persist_queue_snapshot(reason="test")
    snap = json.loads(queue.QUEUE_SNAPSHOT_PATH.read_text())
    snap["running"] = [{"id": "root-live", "task": {"id": "root-live", "chat_id": 0}}]
    queue.QUEUE_SNAPSHOT_PATH.write_text(json.dumps(snap))
    workers.PENDING[:] = []

    fenced: list = []
    assert queue.restore_pending_from_snapshot(terminalized=fenced) == 1
    assert fenced == ["root-live"]
    by_id = {str(row["id"]): row for row in workers.PENDING}
    assert set(by_id) == {"child-paused", "child-fresh"}
    assert by_id["child-fresh"]["_terminalization_retry"]["trigger"] == "pending_parent_interrupted"
    paused = by_id["child-paused"]
    assert paused["_budget_pause"]["exact_continuation"] is True
    # (the snapshot whitelist round-trips the key as None; a marker is a dict)
    assert not paused.get("_terminalization_retry") and budget_hold_fact(paused) is None
    rows = [json.loads(line) for line in (tmp_path / "logs" / "supervisor.jsonl").read_text().splitlines()]
    restored = [row for row in rows if row["type"] == "queue_restored_from_snapshot"][-1]
    assert restored["pending_parent_interrupted"] == ["child-fresh"]
