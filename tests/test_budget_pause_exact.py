"""Exact mid-run budget pause and owner-granted same-ID Resume (#1196).

Static authoring note: these tests were WRITTEN against the candidate but NOT
RUN by their author (no runtime imports were permitted in that lane); the
parent's isolated harness is the first execution.

The pausing half is a HOLD, not a fallback: once the dispatch fence closes the
task stays fenced and nonterminal until its producers are quiescent and its
continuation is stored. These tests therefore drive the hold with a stubbed
``_hold_control_reason`` (the task's existing control rail) rather than waiting
on a real clock, and pin a shortened ``_HOLD_POLL_SEC``.
The supervisor-side half (park, worker death, Resume grants, threshold refresh)
continues in ``test_budget_pause_exact_resume``.
"""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest

from tests._budget_pause_exact_helpers import (
    _controls, _fast_hold, _install_queue, _loop_ctx, _mock_pause_observation, _pause, _quiet_external,
    _running_row, _supervisor_ctx,
)


# --------------------------------------------------------------------------- program counter

def test_unanswered_tool_calls_are_execution_unknown_not_replayed(tmp_path, monkeypatch):
    from ouroboros import budget_pause

    messages = [
        {"role": "assistant", "tool_calls": [{"id": "a"}, {"id": "b"}, {"id": "c"}]},
        {"role": "tool", "tool_call_id": "b", "content": "ok"},
    ]
    assert budget_pause.pending_tool_call_ids(messages) == ["a", "c"]
    assert budget_pause.pending_tool_call_ids([{"role": "assistant", "content": "hi"}]) == []
    # The program counter the pause row carries: unanswered calls of the last
    # batch are EXECUTION-UNKNOWN (never re-executed); a batch with none is a boundary.
    ctx, _limit, pause = _pause(tmp_path, monkeypatch)
    budget_pause.end_dispatch_fence(ctx.task_id)
    point = pause["resume_point"]
    assert point["round_idx"] == 4 and point["phase"] == "partial_tool_batch_unknown"
    assert point["unanswered_tool_call_ids"] == ["call_b"]
    assert point["unanswered_policy"] == "not_re_executed_execution_unknown"
    _running_row(tmp_path, "boundary-1")
    _ctx, limit_ctx = _loop_ctx(tmp_path, "boundary-1")
    limit_ctx.messages.append({"role": "tool", "tool_call_id": "call_b", "content": "done b"})
    with pytest.raises(budget_pause.BudgetPauseRequested) as raised:
        budget_pause.request_pause(limit_ctx, rail=budget_pause.RAIL_GLOBAL_EXHAUSTED,
                                   scope="global", reason_text="x")
    budget_pause.end_dispatch_fence("boundary-1")
    point = raised.value.pause["resume_point"]
    assert point["phase"] == "boundary" and point["unanswered_tool_call_ids"] == []


# --------------------------------------------------------------------------- loop-side pause

def test_direct_actor_pauses_under_its_own_task_id_and_its_event_carries_its_record(tmp_path, monkeypatch):
    """A direct owner-chat turn is ELIGIBLE (#1196): it never had an admission-written
    RUNNING row, so the pause writes one first; its event carries the turn's own
    record (minus inline image bytes) because RUNNING is not its carrier."""
    from ouroboros import budget_pause
    from ouroboros.task_results import load_task_result

    ctx, limit_ctx = _loop_ctx(tmp_path, "direct-1", direct=True)
    ctx.current_chat_id = 42
    _fast_hold(monkeypatch, budget_pause)
    _quiet_external(monkeypatch, budget_pause)
    # Eligible: the pause below raises instead of returning ``exact_pause_unavailable``.
    with pytest.raises(budget_pause.BudgetPauseRequested) as raised:
        budget_pause.request_pause(limit_ctx, rail=budget_pause.RAIL_GLOBAL_EXHAUSTED,
                                   scope="global", reason_text="x")
    budget_pause.end_dispatch_fence("direct-1")
    row = load_task_result(tmp_path, "direct-1", strict=True)
    assert row["_is_direct_chat"] is True and row["chat_id"] == 42
    assert row["budget_pause"]["is_direct_chat"] is True and row["budget_pause"]["source_ref"]
    task = {"id": "direct-1", "type": "task", "chat_id": 42, "text": "hello", "_is_direct_chat": True,
            "image_base64": "AAAA", "origin_message_ref": {"chat_id": 42}, "metadata": {"k": "v"}}
    event = budget_pause.pause_event(task, raised.value.pause)
    assert event["_is_direct_chat"] is True and event["resource_limit"]["exact_continuation"] is True
    carried = event["task"]
    assert carried["id"] == "direct-1" and carried["_is_direct_chat"] is True
    assert "image_base64" not in carried and carried["origin_message_ref"] == {"chat_id": 42}
    assert carried["metadata"] == {"k": "v"} and carried["_attempt"] == 1 and carried["depth"] == 0
    # A pooled task's event carries no record: RUNNING is its carrier.
    assert "task" not in budget_pause.pause_event({"id": "direct-1", "type": "task"}, raised.value.pause)
    # A context with no continuation owner at all is still excluded, loudly: the
    # pause returns without fencing and names the reason on the usage row.
    ctx.owner_wait_callback = None
    assert budget_pause.request_pause(limit_ctx, rail=budget_pause.RAIL_GLOBAL_EXHAUSTED,
                                      scope="global", reason_text="x") is None
    assert limit_ctx.accumulated_usage["exact_pause_unavailable"] == "no_continuation_owner"


def test_direct_turn_pause_event_parks_the_carried_record_in_pending(tmp_path, monkeypatch):
    """The supervisor parks a direct turn from its OWN record: same task id, lane fact
    kept, queue-order facts minted, snapshot persisted, census phase paused."""
    from ouroboros import budget_pause
    from supervisor.events import _handle_budget_pause
    from supervisor.queue_transitions import budget_pause_fact

    queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    ctx, limit_ctx = _loop_ctx(tmp_path, "direct-2", direct=True)
    ctx.current_chat_id = 7
    _fast_hold(monkeypatch, budget_pause)
    _quiet_external(monkeypatch, budget_pause)
    with pytest.raises(budget_pause.BudgetPauseRequested) as raised:
        budget_pause.request_pause(limit_ctx, rail=budget_pause.RAIL_GLOBAL_EXHAUSTED,
                                   scope="global", reason_text="x")
    budget_pause.end_dispatch_fence("direct-2")
    task = {"id": "direct-2", "type": "task", "chat_id": 7, "text": "hello", "_is_direct_chat": True,
            "metadata": {"origin_message_ref": {"chat_id": 7}}}
    persisted, pushed = [], []
    sctx = _supervisor_ctx(tmp_path, workers, queue, persisted, pushed)
    assert workers.RUNNING == {}  # a direct turn is never in RUNNING
    _handle_budget_pause(budget_pause.pause_event(task, raised.value.pause), sctx)
    parked = workers.PENDING[0]
    assert parked["id"] == "direct-2" and parked["_is_direct_chat"] is True
    assert parked["_budget_pause"]["exact_continuation"] is True
    assert parked["_queue_seq"] and parked["queued_at"] and "priority" in parked
    assert budget_pause_fact(parked)["exact_continuation"] is True
    assert budget_pause.budget_pause_row(tmp_path, "direct-2")["state"] == budget_pause.STATE_PAUSED
    assert persisted == ["budget_pause_exact_continuation"]
    assert pushed[0]["type"] == "budget_scope_paused" and pushed[0]["task_id"] == "direct-2"
    # The snapshot keeps the lane fact, so a restart restores the same direct row.
    queue.persist_queue_snapshot(reason="test")
    snap = json.loads(queue.QUEUE_SNAPSHOT_PATH.read_text())
    assert snap["pending"][0]["task"]["_is_direct_chat"] is True
    # The census reads it as a paused DIRECT activity under the same id.
    from ouroboros.gateway import state as gw_state

    rows = gw_state._chat_activities_snapshot_safe(tmp_path, direct_turns=[])
    assert [(r["activity_id"], r["kind"], r["phase"]) for r in rows if r["activity_id"] == "direct-2"] == [
        ("direct-2", "direct_chat", "budget_paused")]
    # An event without the record cannot park a turn that is not running.
    bare = budget_pause.pause_event({"id": "direct-2", "type": "task"}, raised.value.pause)
    workers.PENDING[:] = []
    with pytest.raises(RuntimeError):
        _handle_budget_pause(bare, sctx)


def test_task_event_addressing_stamps_the_direct_lane_fact_from_the_running_row(tmp_path):
    """A resumed direct turn runs on a pooled worker: its frames keep the lane fact."""
    from supervisor.log_addressing import address_task_event

    running = {"d-1": {"task": {"id": "d-1", "chat_id": 5, "_is_direct_chat": True}}}
    payload = address_task_event(running, tmp_path, {"task_id": "d-1", "type": "tool_call_started"})
    assert payload["_is_direct_chat"] is True and payload["chat_id"] == 5
    managed = address_task_event({"m-1": {"task": {"id": "m-1", "chat_id": 5}}}, tmp_path, {"task_id": "m-1"})
    assert "_is_direct_chat" not in managed


def test_pause_writes_source_and_row_before_raising_and_closes_fence(tmp_path, monkeypatch):
    from ouroboros import budget_pause
    from ouroboros.artifacts import read_actor_source_bytes

    ctx, limit_ctx, pause = _pause(tmp_path, monkeypatch)
    try:
        assert pause["state"] == budget_pause.STATE_PAUSING
        assert pause["exact_continuation"] is True and pause["replay_safe"] is False
        assert pause["resume_point"]["unanswered_tool_call_ids"] == ["call_b"]
        row = budget_pause.budget_pause_row(tmp_path, ctx.task_id)
        assert row["pause_id"] == pause["pause_id"] and row["task_attempt"] == 1
        state = json.loads(read_actor_source_bytes(tmp_path, ctx.task_id, row["source_ref"]))
        # The rail's terminal projection must not travel into the continuation.
        assert "execution_status" not in state["usage"] and "reason_code" not in state["usage"]
        assert state["usage"]["cost"] == 1.25 and state["round_idx"] == 4
        assert state["messages"] == limit_ctx.messages and state["seen"] == ["m1"]
        assert limit_ctx.accumulated_usage["reason_code"] == "budget_paused"
        assert budget_pause.dispatch_fenced(ctx.task_id)
        # Ledger seam: no NEW send under the fenced task.
        from ouroboros import usage_accounting as ua

        with pytest.raises(ua.DispatchFenced):
            ua.reserve_attempt(ua.AttemptRequest(model="m", provider="test", drive_root=tmp_path,
                                                 task_id=ctx.task_id, root_task_id=ctx.task_id))
    finally:
        budget_pause.end_dispatch_fence(ctx.task_id)


def test_storage_failure_keeps_a_fenced_nonterminal_hold_and_buys_no_model_call(tmp_path, monkeypatch):
    """A write that fails is a RETAINED hold, never a claimed pause and never
    the old paid terminal rail: the fence stays closed and the task keeps its
    worker until its own control ends the hold."""
    from ouroboros import budget_pause, utils
    from ouroboros.model_wait import ModelWaitInterrupted

    _running_row(tmp_path, "hold-store")
    ctx, limit_ctx = _loop_ctx(tmp_path, "hold-store")
    monkeypatch.setattr(utils, "update_json_locked",
                        lambda *_a, **_k: (_ for _ in ()).throw(OSError("read-only file system")))
    published = []
    monkeypatch.setattr(budget_pause, "_publish_hold", lambda _c, row: published.append(row))
    monkeypatch.setattr(budget_pause, "_HOLD_POLL_SEC", 0.01)
    monkeypatch.setattr(budget_pause, "_hold_control_reason", _controls("", "", "cancelled"))
    with pytest.raises(ModelWaitInterrupted):
        budget_pause.request_pause(limit_ctx, rail=budget_pause.RAIL_GRACEFUL_CEILING,
                                   scope="root", reason_text="x")
    usage = limit_ctx.accumulated_usage
    assert usage["exact_pause_unavailable"] == "hold_ended_by_control"
    assert usage["budget_pause_hold"]["hold_reason"] == budget_pause.HOLD_PAUSE_RECORD_UNWRITABLE
    assert "read-only file system" in usage["budget_pause_hold"]["error"]
    assert usage["budget_pause_hold"]["ended_by"] == "cancelled"
    # The hold was owner-visible while it lasted, and announced once per reason
    # (a poll interval is not a ledger cadence), then closed explicitly.
    still_held = [row for row in published if row["state"] == budget_pause.STATE_PAUSING]
    assert [row["hold_reason"] for row in still_held] == [budget_pause.HOLD_PAUSE_RECORD_UNWRITABLE]
    assert published[-1]["state"] == "hold_ended" and published[-1]["ended_by"] == "cancelled"
    # No pause was claimed, no wrap-up was bought, and the fence never reopened.
    assert "budget_pause" not in usage and usage.get("reason_code") == "budget_exhausted"
    assert budget_pause.dispatch_fenced("hold-store")
    budget_pause.end_dispatch_fence("hold-store")


def test_failed_checkpoint_publication_retries_the_prepared_snapshot_not_a_rebuild(tmp_path, monkeypatch):
    """A publication that fails after quiescence is retried with the SAME prepared
    snapshot: custody is observed (stops requested) and the source stored ONCE, not
    once per poll. A producer going live again discards that snapshot, so the next
    quiescence re-observes custody before the pause is published."""
    from ouroboros import budget_pause, owner_wait

    _running_row(tmp_path, "hold-retry")
    ctx, limit_ctx = _loop_ctx(tmp_path, "hold-retry")
    monkeypatch.setattr(budget_pause, "_HOLD_POLL_SEC", 0.01)
    monkeypatch.setattr(budget_pause, "_hold_control_reason", lambda _c: "")
    observed, stored = [], []
    monkeypatch.setattr(budget_pause, "observe_task_runs", lambda _root, _task_id, **_kw: observed.append(1) or {
        "runs": [{"run_id": "run-1", "state": "stop_requested", "stop_outcome": "requested"}],
        "observed_at": time.time(), "custody_read": "ok", "coverage_basis": "test"})
    real_store = owner_wait.store_continuation_source
    monkeypatch.setattr(owner_wait, "store_continuation_source",
                        lambda *a, **k: stored.append(1) or real_store(*a, **k))
    quiescence = iter([True, True, True, False, True])  # settled, settled, settled, live again, settled
    monkeypatch.setattr(budget_pause, "local_producer_observation",
                        lambda _c, timeout_sec: {"quiescent": next(quiescence, True), "review_attempts": {},
                                                 "tool_futures": {}})
    real_set = budget_pause.set_budget_pause
    failures = {"left": 3}

    def flaky_publish(root, task_id, row, **kw):
        if kw.get("expected_pause_id") and row.get("source_ref") and failures["left"] > 0:
            failures["left"] -= 1
            raise OSError("disk full")
        return real_set(root, task_id, row, **kw)

    monkeypatch.setattr(budget_pause, "set_budget_pause", flaky_publish)
    with pytest.raises(budget_pause.BudgetPauseRequested) as raised:
        budget_pause.request_pause(limit_ctx, rail=budget_pause.RAIL_GLOBAL_EXHAUSTED,
                                   scope="global", reason_text="x")
    # Three failed publications reused one prepared snapshot; the unsettled
    # interlude forced exactly one fresh observation before the durable pause.
    assert observed == [1, 1] and stored == [1, 1]
    assert failures["left"] == 0 and raised.value.pause["state"] == budget_pause.STATE_PAUSING
    assert budget_pause.budget_pause_row(tmp_path, "hold-retry")["source_ref"] == raised.value.pause["source_ref"]
    assert "budget_pause_hold" not in limit_ctx.accumulated_usage
    budget_pause.end_dispatch_fence("hold-retry")


def test_stop_during_a_hold_ends_it_through_the_existing_control_rail(tmp_path, monkeypatch):
    from ouroboros import budget_pause
    from ouroboros.model_wait import ModelWaitInterrupted

    _running_row(tmp_path, "hold-stop")
    ctx, limit_ctx = _loop_ctx(tmp_path, "hold-stop")
    monkeypatch.setattr(budget_pause, "_HOLD_POLL_SEC", 0.01)
    monkeypatch.setattr(budget_pause, "local_producer_observation",
                        lambda _c, timeout_sec: {"quiescent": False, "review_attempts": {},
                                                 "tool_futures": {}})
    monkeypatch.setattr(budget_pause, "_hold_control_reason", _controls("", "", "cancelled"))
    with pytest.raises(ModelWaitInterrupted) as raised:
        budget_pause.request_pause(limit_ctx, rail=budget_pause.RAIL_GLOBAL_EXHAUSTED,
                                   scope="global", reason_text="x")
    assert raised.value.control_reason == "cancelled"
    # The pausing row opened; the control closed it as abandoned rather than
    # leaving a half-written pause claiming to be durable.
    row = budget_pause.budget_pause_row(tmp_path, "hold-stop")
    assert row["state"] == budget_pause.STATE_ABANDONED
    assert row["abandon_reason"] == "hold_ended_by_control:cancelled"
    # An abandoned row is no pause/consumption evidence: the reaper's one row
    # read neither parks the id nor fences the ordinary crash retry.
    from supervisor import worker_health

    assert worker_health._complete_exact_budget_pause_after_death(
        {}, tmp_path, {"id": "hold-stop"}, "hold-stop", 1) == (False, False)
    assert budget_pause.dispatch_fenced("hold-stop")  # no fence reopen
    budget_pause.end_dispatch_fence("hold-stop")


def test_hold_controls_are_the_existing_ones_and_a_finalize_request_is_not_one(tmp_path, monkeypatch):
    """The hold borrows the task's own control rail and invents none: Stop,
    Panic, an explicit deadline and the finite lifetime end it; 'finalize now'
    and a closed wait do not abort a pause the fence already committed to."""
    from ouroboros import budget_pause, cancel_intents, model_wait

    ctx, _limit = _loop_ctx(tmp_path, "controls-1")
    monkeypatch.setattr(cancel_intents, "cancel_pending", lambda *_a, **_k: False)
    monkeypatch.setattr(model_wait, "current_model_wait",
                        lambda: SimpleNamespace(control_reason=lambda: "finalize_requested"))
    assert budget_pause._hold_control_reason(ctx) == ""
    monkeypatch.setattr(model_wait, "current_model_wait",
                        lambda: SimpleNamespace(control_reason=lambda: "absolute_ceiling"))
    assert budget_pause._hold_control_reason(ctx) == "absolute_ceiling"
    # No bound wait: the owner-stop flags the restore gate reads answer instead.
    monkeypatch.setattr(model_wait, "current_model_wait", lambda: None)
    assert budget_pause._hold_control_reason(ctx) == ""
    (tmp_path / "state").mkdir(exist_ok=True)
    (tmp_path / "state" / "panic_stop.flag").write_text("panic")
    assert budget_pause._hold_control_reason(ctx) == "panic"


def test_pending_review_attempt_blocks_release_then_pauses_once_it_settles(tmp_path, monkeypatch):
    """Quiescence is the release gate, not a refusal: the task holds while an
    already-sent review attempt is open and pauses exactly when it settles."""
    from ouroboros import budget_pause, review_custody as rc

    _running_row(tmp_path, "busy-1")
    ctx, limit_ctx = _loop_ctx(tmp_path, "busy-1")
    monkeypatch.setattr(budget_pause, "_HOLD_POLL_SEC", 0.01)
    _quiet_external(monkeypatch, budget_pause)
    holds = []
    monkeypatch.setattr(budget_pause, "_publish_hold", lambda _c, row: holds.append(row))
    open_attempt = rc.ActiveReviewAttempt(key="k", operation_id="op-open", wave_key="task_acceptance|busy-1|r")
    with rc._ACTIVE_LOCK:
        rc._ACTIVE["busy-k"] = open_attempt
    # The attempt settles through its OWN custody path after the first hold.
    monkeypatch.setattr(budget_pause, "_hold_control_reason",
                        lambda _ctx: (open_attempt.event.set() if holds else None) or "")
    try:
        with pytest.raises(budget_pause.BudgetPauseRequested) as raised:
            budget_pause.request_pause(limit_ctx, rail=budget_pause.RAIL_GLOBAL_EXHAUSTED,
                                       scope="global", reason_text="x")
    finally:
        with rc._ACTIVE_LOCK:
            rc._ACTIVE.pop("busy-k", None)
        budget_pause.end_dispatch_fence("busy-1")
    assert holds[0]["hold_reason"] == budget_pause.HOLD_PRODUCERS_UNSETTLED
    assert holds[0]["unsettled"]["review_attempts"][0]["operation_id"] == "op-open"
    assert holds[0]["state"] == budget_pause.STATE_PAUSING  # nonterminal throughout
    # The row was never abandoned: the same pause id carries through to the row.
    row = budget_pause.budget_pause_row(tmp_path, "busy-1")
    assert row["state"] == budget_pause.STATE_PAUSING and row["source_ref"]
    assert row["pause_id"] == raised.value.pause["pause_id"]
    assert "budget_pause_hold" not in limit_ctx.accumulated_usage


def test_timed_out_tool_future_blocks_release_until_its_settlement_callback_finishes(tmp_path):
    """``future.done()`` is not callback-complete: a call abandoned at its own
    timeout keeps the task unquiescent until the late settlement callback that
    owns its effects has finished.

    The registering OWNER pins the row until it releases, so the instant between
    the caller's own timeout and its ``hold_tool_settlement`` claim can never
    read as settled-and-unheld: the claim is taken while the pin still holds.
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from ouroboros import budget_pause

    ctx, _limit = _loop_ctx(tmp_path, "tool-1")
    gate = threading.Event()
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        future = executor.submit(gate.wait, 5.0)
        owner_release = budget_pause.register_tool_future(ctx, "call_slow", "run_command", future)
        drain = budget_pause.drain_local_tool_futures(ctx, timeout_sec=0.05)
        assert drain["drained"] is False
        assert drain["unsettled"] == [{"operation_id": "call_slow", "tool": "run_command",
                                       "state": "running"}]
        # The timeout path claims the row BEFORE the worker settles, while the
        # registering owner still pins it; only then does the owner hand over.
        release = budget_pause.hold_tool_settlement(ctx, "call_slow")
        owner_release()
        gate.set()
        assert future.result(timeout=5.0) is True
        # ``done()`` is true now, but the late callback still owns the effects.
        assert budget_pause.drain_local_tool_futures(ctx, timeout_sec=0.05)["drained"] is False
        release()
        settled = budget_pause.drain_local_tool_futures(ctx, timeout_sec=1.0)
        assert settled["drained"] is True
        assert settled["settled"] == [{"operation_id": "call_slow", "tool": "run_command"}]
    finally:
        gate.set()
        executor.shutdown(wait=True)
        budget_pause.forget_tool_scope(ctx)


def test_neither_half_of_the_settlement_protocol_alone_is_quiescence(tmp_path):
    """Negative: an owner release over a still-running future is not quiescence, and a
    registration interleaved with a finished-but-unreleased row never prunes it into
    false quiescence."""
    from concurrent.futures import Future

    from ouroboros import budget_pause

    ctx, _limit = _loop_ctx(tmp_path, "tool-3")
    try:
        running = Future()
        owner_running = budget_pause.register_tool_future(ctx, "call_running", "run_command", running)
        # The owner is done deciding, but the future has NOT finished: not quiescent.
        owner_running()
        assert budget_pause.drain_local_tool_futures(ctx, timeout_sec=0.02)["unsettled"] == [
            {"operation_id": "call_running", "tool": "run_command", "state": "running"}]
        running.set_result("late")
        assert budget_pause.drain_local_tool_futures(ctx, timeout_sec=1.0)["drained"] is True
        # A FINISHED future whose owner has not released is unsettled, and the next
        # registration must prune only the released row, never the pinned one.
        pinned = Future()
        pinned.set_result("x")
        budget_pause.register_tool_future(ctx, "call_pinned", "read_file", pinned)  # pin kept
        third = Future()
        third.set_result("y")
        owner_third = budget_pause.register_tool_future(ctx, "call_third", "read_file", third)
        drain = budget_pause.drain_local_tool_futures(ctx, timeout_sec=0.02)
        assert drain["drained"] is False
        assert sorted(row["operation_id"] for row in drain["unsettled"]) == ["call_pinned", "call_third"]
        owner_third()
        drain = budget_pause.drain_local_tool_futures(ctx, timeout_sec=0.5)
        assert drain["drained"] is False
        assert [row["operation_id"] for row in drain["unsettled"]] == ["call_pinned"]
        assert [row["operation_id"] for row in drain["settled"]] == ["call_third"]
    finally:
        budget_pause.forget_tool_scope(ctx)


def test_tool_future_quiescence_is_scoped_to_one_attempt_and_prunes_itself(tmp_path):
    from concurrent.futures import Future

    from ouroboros import budget_pause

    ctx, _limit = _loop_ctx(tmp_path, "tool-2", attempt=1)
    later, _l = _loop_ctx(tmp_path, "tool-2", attempt=2)
    done = Future()
    done.set_result("x")
    try:
        release_a = budget_pause.register_tool_future(ctx, "call_a", "read_file", done)
        # Finished, but the registering owner still pins it: not yet settled.
        assert budget_pause.drain_local_tool_futures(ctx, timeout_sec=0.05)["drained"] is False
        release_a()
        assert budget_pause.drain_local_tool_futures(ctx, timeout_sec=0.5)["drained"] is True
        # A later attempt never inherits a previous attempt's observations.
        assert budget_pause.drain_local_tool_futures(later, timeout_sec=0.0) == {
            "drained": True, "registry": "ok", "settled": [], "unsettled": []}
        # A settled, unheld AND released row is pruned by the next registration.
        second = Future()
        second.set_result("y")
        release_b = budget_pause.register_tool_future(ctx, "call_b", "read_file", second)
        release_b()
        assert [row["operation_id"] for row in
                budget_pause.drain_local_tool_futures(ctx, timeout_sec=0.5)["settled"]] == ["call_b"]
    finally:
        budget_pause.forget_tool_scope(ctx)
        budget_pause.forget_tool_scope(later)


def test_unreadable_external_custody_is_held_as_unknown_not_clean(tmp_path, monkeypatch):
    from ouroboros import budget_pause

    from ouroboros import delegate_custody as custody

    _running_row(tmp_path, "custody-1")
    ctx, limit_ctx = _loop_ctx(tmp_path, "custody-1")
    _fast_hold(monkeypatch, budget_pause)
    # The loop side resolves THIS context's custody root; one it cannot resolve
    # is held on the pause row as UNKNOWN, never as "no runs".
    monkeypatch.setattr(custody, "custody_root", lambda _c: (_ for _ in ()).throw(OSError("boom")))
    with pytest.raises(budget_pause.BudgetPauseRequested) as raised:
        budget_pause.request_pause(limit_ctx, rail=budget_pause.RAIL_GLOBAL_EXHAUSTED,
                                   scope="global", reason_text="x")
    budget_pause.end_dispatch_fence("custody-1")
    observed = raised.value.pause["external_runs"]
    assert observed["custody_read"] == "failed" and observed["runs"] == []
    assert "boom" in observed["error"]
    # The supervisor-side twin shares the body: a replay that fails is the same typed fact.
    monkeypatch.setattr(custody, "replay", lambda _r, rows=None: (_ for _ in ()).throw(OSError("rows torn")))
    grant_side = budget_pause.observe_task_runs(tmp_path, ctx.task_id)
    assert grant_side["custody_read"] == "failed" and "rows torn" in grant_side["error"]


def test_pausing_row_exists_before_any_wait_and_fences_crash_retry(tmp_path, monkeypatch):
    """The durable ``pausing`` row opens BEFORE the task waits on anything, so a
    death while it is still settling meets the crash-retry fence, not a replay."""
    from ouroboros import budget_pause
    from ouroboros.model_wait import ModelWaitInterrupted

    _running_row(tmp_path, "drain-1")
    ctx, limit_ctx = _loop_ctx(tmp_path, "drain-1")
    monkeypatch.setattr(budget_pause, "_HOLD_POLL_SEC", 0.01)
    seen = {}

    def _observe_during_hold(_ctx, *, timeout_sec):
        seen["row"] = budget_pause.budget_pause_row(tmp_path, "drain-1")
        raise RuntimeError("simulated death while settling")

    monkeypatch.setattr(budget_pause, "local_producer_observation", _observe_during_hold)
    monkeypatch.setattr(budget_pause, "_hold_control_reason", _controls("", "", "panic"))
    with pytest.raises(ModelWaitInterrupted):
        budget_pause.request_pause(limit_ctx, rail=budget_pause.RAIL_GLOBAL_EXHAUSTED,
                                   scope="global", reason_text="x")
    assert seen["row"]["state"] == budget_pause.STATE_PAUSING and seen["row"]["source_ref"] is None
    # An observation that RAISED proves nothing about quiescence: it holds.
    held = limit_ctx.accumulated_usage["budget_pause_hold"]
    assert held["hold_reason"] == budget_pause.HOLD_PRODUCERS_UNSETTLED
    assert "simulated death" in held["unsettled"]["observation_error"]
    budget_pause.end_dispatch_fence("drain-1")
    # While that row was live (no source yet) the crash-retry fence already held;
    # an unreadable record fails closed the same way: the reaper's one row read
    # parks nothing and fences the retry.
    from supervisor import worker_health

    monkeypatch.setattr(budget_pause, "budget_pause_row",
                        lambda *_a: (_ for _ in ()).throw(OSError("unreadable")))
    assert worker_health._complete_exact_budget_pause_after_death(
        {}, tmp_path, {"id": "drain-1"}, "drain-1", 1) == (False, True)


def test_light_extraction_is_not_dispatched_under_the_fence(monkeypatch):
    from ouroboros import budget_pause, review_verdict_extraction as rve
    from ouroboros import usage_accounting as ua

    monkeypatch.setattr(ua, "current_usage_scope", lambda: ua.UsageScope(task_id="fenced-task"))
    budget_pause.begin_dispatch_fence("fenced-task")
    try:
        canonical, usage = rve._extract_verdict_via_light_model("free text verdict", contract="c")
        assert canonical is None
        assert usage["reason_code"] == "budget_pausing_no_extraction"
        assert usage["dispatch"] == "not_dispatched"
    finally:
        budget_pause.end_dispatch_fence("fenced-task")


def test_local_review_drain_lists_unsettled_attempts_without_settling_them():
    from ouroboros import budget_pause, review_custody as rc

    settled = rc.ActiveReviewAttempt(key="triad|t9|r", operation_id="op-settled", wave_key="task_acceptance|t9|r")
    settled.event.set()
    open_attempt = rc.ActiveReviewAttempt(key="triad|t9|r2", operation_id="op-open", wave_key="task_acceptance|t9|r2")
    foreign = rc.ActiveReviewAttempt(key="triad|other|r", operation_id="op-foreign", wave_key="task_acceptance|other|r")
    with rc._ACTIVE_LOCK:
        rc._ACTIVE.update({"k1": settled, "k2": open_attempt, "k3": foreign})
    try:
        drain = budget_pause.drain_local_review_attempts("t9", timeout_sec=0.05)
        assert [row["operation_id"] for row in drain["settled"]] == ["op-settled"]
        assert [row["operation_id"] for row in drain["unsettled"]] == ["op-open"]
        assert drain["drained"] is False
        assert not open_attempt.event.is_set()  # never settled by the drain
    finally:
        with rc._ACTIVE_LOCK:
            for key in ("k1", "k2", "k3"):
                rc._ACTIVE.pop(key, None)


def test_the_round_start_stamp_stays_out_of_the_stored_continuation(tmp_path, monkeypatch):
    """A paused round resumes on the same round id; a carried monotonic start would stretch
    its llm_round duration over the whole pause, so the stored usage drops it and keeps the rest."""
    from ouroboros import budget_pause
    from ouroboros.artifacts import read_actor_source_bytes
    from ouroboros.loop_llm_call import ROUND_STARTED_KEY

    _running_row(tmp_path, "stamp-task")
    ctx, limit_ctx = _loop_ctx(tmp_path, "stamp-task")
    limit_ctx.accumulated_usage[ROUND_STARTED_KEY] = {"round_id": "e:round:4", "at": 1.0}
    _fast_hold(monkeypatch, budget_pause)
    _mock_pause_observation(monkeypatch, budget_pause, [
        {"run_id": "run-1", "state": "stop_requested", "stop_outcome": "requested"}])
    try:
        with pytest.raises(budget_pause.BudgetPauseRequested):
            budget_pause.request_pause(limit_ctx, rail=budget_pause.RAIL_GLOBAL_EXHAUSTED, scope="global",
                                       reason_text="money gone", root_task_id="stamp-task")
        row = budget_pause.budget_pause_row(tmp_path, ctx.task_id)
        state = json.loads(read_actor_source_bytes(tmp_path, ctx.task_id, row["source_ref"]))
        assert ROUND_STARTED_KEY not in state["usage"] and state["usage"]["cost"] == 1.25
    finally:
        budget_pause.end_dispatch_fence(ctx.task_id)
