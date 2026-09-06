"""A cooperative stop interrupts only the unsent paid transport-repeat grant."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import httpx
import pytest

from ouroboros import cancel_intents, loop, loop_llm_call, loop_transport, owner_mailbox
from ouroboros import usage_accounting as accounting
from ouroboros.outcomes import REASON_OWNER_REQUESTED_FINALIZATION
from ouroboros.task_results import write_task_result
from supervisor.owner_stop import REASON_OWNER_STOPPED_DIRECT_TURN, owner_stop_control_id
from tests.test_transport_death_retry import (
    EMPTY_RESPONSE, _LedgerLLM, _ScriptedLLM, _events, _ledger, _loop_kwargs,
    _no_chain, _primary_call, _status_failure,
)


@pytest.mark.parametrize("after_attempt,owner_grace", [(1, True), (1, False), (2, False)])
def test_stop_arriving_in_real_backoff_keeps_only_actual_physical_attempts(
    tmp_path, monkeypatch, after_attempt, owner_grace,
):
    canonical = tmp_path / "canonical"
    execution = tmp_path / "execution" if owner_grace else canonical
    canonical.mkdir()
    execution.mkdir(exist_ok=True)
    write_task_result(canonical, "t-death", "running",
                      **({"child_drive_root": str(execution)} if owner_grace else {}))
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.delenv("USE_LOCAL_FALLBACK", raising=False)
    monkeypatch.setattr(loop, "_run_cross_model_fallback_chain", _no_chain)
    llm = _LedgerLLM(canonical, *([lambda: httpx.ReadError("controlled socket death")] * 3))
    kwargs = _loop_kwargs(execution, llm, [])
    kwargs["drive_logs"] = execution / "logs"
    ctx = kwargs["tools"]._ctx
    ctx.task_id, ctx.task_attempt, ctx.budget_drive_root = "t-death", 1, canonical
    ctx.is_direct_chat = not owner_grace
    ctx.task_metadata = {"root_task_id": "t-death"}
    first_empty_check = threading.Event()
    real_wait = loop_transport.interruptible_wait_sleep
    def observed_wait(seconds, check):
        def checked():
            result = check()
            if not result and llm.calls == after_attempt:
                first_empty_check.set()
            return result
        return real_wait(seconds, checked)
    monkeypatch.setattr(loop_transport, "interruptible_wait_sleep", observed_wait)
    def write_stop():
        assert first_empty_check.wait(10)
        if owner_grace:
            intent = cancel_intents.request_cancel(canonical, "t-death", source="isolated-test",
                requested_stop_policy=cancel_intents.STOP_POLICY_FINALIZE)
            text, mid = REASON_OWNER_REQUESTED_FINALIZATION, owner_stop_control_id(intent)
        else:
            text, mid = REASON_OWNER_STOPPED_DIRECT_TURN, "direct-stop"
        assert owner_mailbox.write_owner_message(execution, text, "t-death", msg_id=mid,
                                                  kind=owner_mailbox.KIND_FINALIZE_NOW)
        return mid
    with accounting.usage_scope(accounting.UsageScope(drive_root=canonical, task_id="t-death",
             root_task_id="t-death", global_limit_usd=100.0)), ThreadPoolExecutor(max_workers=1) as pool:
        writing = pool.submit(write_stop)
        _text, usage, trace = loop.run_llm_loop(**kwargs)
        mid = writing.result(timeout=5)
    assert llm.calls == after_attempt
    rows = _ledger(canonical)
    assert len({row["attempt_id"] for row in rows}) == after_attempt
    assert [row["state"] for row in rows] == ["reserved", "dispatched", "unresolved"] * after_attempt
    assert accounting.usage_projection(canonical)["unresolved_upper_bound_usd"] == float(after_attempt)
    assert usage.get(loop_llm_call.TRANSPORT_DEATHS_KEY, {}).get("count", 0) == after_attempt - 1
    assert usage["_last_llm_retry_same_request"] is False
    assert usage["_last_llm_error_kind"] == "provider_outcome_unknown"
    assert trace["forced_finalization"]["source"] == "provider_outcome_unknown_no_resend"
    assert ("owner requested Wrap up" if owner_grace else "owner requested Stop") in usage["terminal_provider_notice"]
    assert "no terminal provider outcome" in usage["terminal_provider_notice"]
    assert [row["reason_code"] for row in _events(execution / "logs", "llm_not_dispatched")] == ["finalize_control_pending"]
    assert not _events(execution / "logs", "llm_retry_deadline_exhausted")
    assert mid not in ctx._loop_mailbox_seen_ids
    assert mid not in owner_mailbox.acknowledged_task_message_ids(execution, "t-death", attempt_key=1)
    if owner_grace:
        assert not cancel_intents.active_intent(canonical, "t-death").get("control_drained_at")
    else:
        assert getattr(ctx, "_skip_post_task_synthesis", False)  # same Stop-now contract after loop exit
        from ouroboros import agent_task_pipeline

        post_task_calls = []
        monkeypatch.setattr(agent_task_pipeline, "_run_post_task_processing_async",
                            lambda *_a, **_k: post_task_calls.append(True))
        task = {"id": "t-death", "type": "task", "chat_id": 7, "_is_direct_chat": True}
        agent_task_pipeline.emit_task_results(SimpleNamespace(drive_root=canonical, repo_dir=canonical),
            None, None, [], task, _text, usage, trace, start_time=0.0, drive_logs=canonical / "logs", ctx=ctx)
        assert task["_skip_post_task_synthesis"] is True
        assert post_task_calls == []


@pytest.mark.parametrize("kind,revoked,seen,expected", [
    ("owner_text", False, False, False), ("hurry", False, False, False),
    ("finalize_now", True, False, False), ("finalize_now", False, True, False),
    ("finalize_now", False, False, True),
])
def test_only_current_unseen_finalize_controls_request_a_retry_stop(tmp_path, kind, revoked, seen, expected):
    ctx = SimpleNamespace(drive_root=tmp_path, task_id="t-death", task_attempt=1,
                          _loop_mailbox_seen_ids={"control"} if seen else set())
    owner_mailbox.write_owner_message(tmp_path, "deadline", "t-death", msg_id="control", kind=kind)
    if revoked:
        owner_mailbox.revoke_owner_control(tmp_path, "t-death", "control")
    assert loop_transport.transport_repeat_stop_requested(ctx) is expected
    assert ctx._loop_mailbox_seen_ids == ({"control"} if seen else set())


def test_stale_owner_grace_control_cannot_cancel_a_repeat(tmp_path):
    ctx = SimpleNamespace(drive_root=tmp_path, task_id="t-death", task_attempt=1, _loop_mailbox_seen_ids=set())
    owner_mailbox.write_owner_message(tmp_path, REASON_OWNER_REQUESTED_FINALIZATION,
        "t-death", msg_id="ownerstop:old-request", kind=owner_mailbox.KIND_FINALIZE_NOW)
    assert not loop_transport.transport_repeat_stop_requested(ctx)


def test_deadline_refusal_is_not_misattributed_to_an_unchecked_control(tmp_path):
    def forbidden():
        raise AssertionError("deadline already refused this wait")
    usage = {"_last_llm_error_kind": "provider_outcome_unknown", "_last_llm_retry_same_request": True,
             loop_llm_call.TRANSPORT_DEATHS_KEY: {"round_id": "r", "count": 1, "backoff_sec": 4.0}}
    ctx = loop_llm_call._LlmErrorContext(task_id="t-death", task_type="task", execution_id="e",
        round_id="r", llm_call_id="old-call", round_idx=1, attempt=0, model="fixture",
        request_ref=None, drive_logs=tmp_path, event_queue=None, accumulated_usage=usage,
        deadline_ts=time.time() + 1, stop_retry_check=forbidden)
    assert loop_llm_call._stop_after_llm_error(ctx)
    assert not _events(tmp_path, "llm_not_dispatched")
    assert len(_events(tmp_path, "llm_retry_deadline_exhausted")) == 1
    assert usage["_last_llm_error_kind"] == "provider_outcome_unknown"
    assert loop_llm_call.TRANSPORT_DEATHS_KEY not in usage


@pytest.mark.parametrize("empty", [False, True])
def test_control_callback_does_not_change_other_transient_or_empty_retry_contracts(tmp_path, monkeypatch, empty):
    def forbidden():
        raise AssertionError("only a paid transport-death repeat opts into this stop check")
    def sleep(_seconds, _deadline, **options):
        assert options == {}
        return True
    monkeypatch.setattr(loop_llm_call, "_sleep_within_deadline", sleep)
    llm = _ScriptedLLM(EMPTY_RESPONSE if empty else lambda: _status_failure(503))
    message, _cost = _primary_call(llm, tmp_path, {}, stop_retry_check=forbidden)
    assert message["content"] == "done" and llm.calls == 2


def test_paid_repeat_empty_peek_reuses_existing_wait_proof(tmp_path, monkeypatch):
    ctx = SimpleNamespace(drive_root=tmp_path, task_id="t-death", task_attempt=1, _loop_mailbox_seen_ids={"old"})
    owner_mailbox.write_owner_message(tmp_path, "seen text", "t-death", msg_id="old")
    calls = []
    original = owner_mailbox.drain_owner_entries
    def read(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(owner_mailbox, "drain_owner_entries", read)
    peek = owner_mailbox.OwnerMailboxPeek()
    for _ in range(4):
        assert not loop_transport.transport_repeat_stop_requested(ctx, mailbox_peek=peek)
    assert len(calls) == 1
    owner_mailbox.write_owner_message(tmp_path, REASON_OWNER_STOPPED_DIRECT_TURN, "t-death", msg_id="stop", kind=owner_mailbox.KIND_FINALIZE_NOW)
    assert loop_transport.transport_repeat_stop_requested(ctx, mailbox_peek=peek)
    assert ctx._transport_repeat_control_reason == REASON_OWNER_STOPPED_DIRECT_TURN
    assert ctx._loop_mailbox_seen_ids == {"old"}


@pytest.mark.parametrize("with_leaf", [False, True])
def test_wrapup_reason_survives_the_live_delegate_hold(tmp_path, monkeypatch, with_leaf):
    from ouroboros import claudexor_daemon, delegate_custody, delegate_progress
    from tests.test_delegate_hold import _configured_registry, _start_leaf, _loop_kwargs as hold_kwargs

    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.delenv("USE_LOCAL_FALLBACK", raising=False)
    registry = _configured_registry(tmp_path, task_id="t-death")
    registry._ctx.budget_drive_root = tmp_path
    registry._ctx.task_attempt = 1
    if with_leaf:
        _start_leaf(tmp_path, task_id="t-death", run_id="fixture-leaf")

    def death():
        intent = cancel_intents.request_cancel(tmp_path, "t-death", requested_stop_policy=cancel_intents.STOP_POLICY_FINALIZE)
        owner_mailbox.write_owner_message(tmp_path, REASON_OWNER_REQUESTED_FINALIZATION, "t-death",
            msg_id=owner_stop_control_id(intent), kind=owner_mailbox.KIND_FINALIZE_NOW)
        return httpx.ReadError("controlled post-dispatch failure")

    llm = _LedgerLLM(tmp_path, death)
    kwargs = hold_kwargs(tmp_path, registry, [])
    kwargs["llm"] = llm
    monkeypatch.setattr(claudexor_daemon, "ensure_owned_gateway", lambda **kw: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(delegate_progress, "bounded_poll", lambda *a, **kw: {"summary": {"state": "running"}})
    monkeypatch.setattr(delegate_custody, "release_task_runs", lambda *a, **kw: None)
    with accounting.usage_scope(accounting.UsageScope(
            drive_root=tmp_path, task_id="t-death", root_task_id="t-death", global_limit_usd=100.0)):
        _text, usage, _trace = loop.run_llm_loop(**kwargs)
    assert llm.calls == 1
    assert accounting.usage_projection(tmp_path)["unresolved_upper_bound_usd"] == 1.0
    assert "owner requested Wrap up" in usage["terminal_provider_notice"]
