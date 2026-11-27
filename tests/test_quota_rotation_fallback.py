"""Quota recovery order: Auto account rotation, then the configured fallback, then the owner.

Engine evidence, serving bundle ``state/cx/3.14.0-8849e922607b/claudexord.bundle.cjs``:
``resolve38`` resolves ONE account per model operation before dispatch and
``ModelOperations.execute`` invokes it once ("A model operation may send inference
only once"). A vendor HTTP quota refusal is a settled, failed result naming that
account; ``observe`` ingests its ``resetsAt``/``retryAfterMs`` as the account's
cooldown. A refusal before dispatch is the engine's own verdict, and only its
``poolCause=quota`` says every compatible account is quota-blocked.
"""

from __future__ import annotations

import asyncio
import json
import queue
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest

from ouroboros.send_clock import CLOCK_NOTE_PREFIX
from ouroboros import fallback_cooldown, loop, loop_llm_call, model_wait
from ouroboros import llm_claudexor as transport
from ouroboros import usage_accounting as ua
from ouroboros.loop_llm_call import call_llm_with_retry, provider_no_call_source
from ouroboros.loop_model_call import RESOURCE_REFUSAL_KEY
from ouroboros.loop_transport import provider_recovery_hint
from ouroboros.model_slots import MODEL_ACCOUNTS_KEY
from ouroboros.presence_runner import _presence_delivery
from tests.test_llm_claudexor import MODEL, ROUTE, ledger, result
from tests.test_llm_claudexor import setup as gateway_fixture
from tests.test_model_wait import live_wait as wait_fixture
from tests.test_model_wait_controls import _loop_tools
from tests.test_subscription_main_wait import main_call as main_call_fixture
from tests._usage_store_testing import dispatched_attempts

setup = gateway_fixture
live_wait = wait_fixture
main_call = main_call_fixture
ROUTE_B = {**ROUTE, "credentialProfileId": "account-b", "accountFingerprint": "fingerprint-b"}
RESET = "2099-01-01T00:00:00Z"
FALLBACK = "openai::alternate"


def _vendor_quota(route=ROUTE, *, reset=True, http=True):
    """The vendor refused this account over HTTP: dispatched, settled, nothing generated."""
    context = {"vendorCode": "usage_limit_reached", **({"httpStatus": 429} if http else {}),
               **({"resetsAt": RESET} if reset else {})}
    return result(outcome="failed", route=route, problem={
        "code": "subscription_window_exhausted", "message": "Codex model request was refused (HTTP 429).",
        "retryable": True, "context": context})


def _pool_quota():
    """The engine's own verdict before dispatch: every compatible account is quota-blocked."""
    return result(outcome="failed", route={}, problem={
        "code": "subscription_window_exhausted", "retryable": False,
        "message": "Every available model account is blocked by subscription quota",
        "context": {"source": "codex", "poolCause": "quota", "resetsAt": RESET}})


def _final(text):
    row = result()
    row["message"] = {"role": "assistant", "content": text}
    return row


def _history():
    """A conversation whose last same-route answer came from account-a."""
    return [{"role": "user", "content": "go"}, result()["message"],
            {"role": "tool", "tool_call_id": "a", "content": "A"},
            {"role": "tool", "tool_call_id": "b", "content": "B"}]


def _rows(root, kind):
    path = root / "logs" / "events.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    return [row for row in rows if row.get("type") == kind]


def _waits(events):
    return [event for event in list(events.queue) if event.get("type") == "task_model_wait"]


# -- account rotation (transport) ---------------------------------------------------------------


@pytest.mark.parametrize("asynchronous", [False, True])
def test_unpinned_vendor_quota_reasks_the_same_payload_and_the_engine_picks_the_next_account(setup, asynchronous):
    root, gateway, client = setup
    gateway.results, gateway.dispatch = [_vendor_quota(), result(route=ROUTE_B)], ["response_received"] * 2
    call = client.chat_async if asynchronous else client.chat
    answer = call(_history(), MODEL, None, "high", model_role="main", cache_affinity="affinity")
    message, usage = asyncio.run(answer) if asynchronous else answer
    assert message["content"] == result()["message"]["content"]
    assert usage["claudexor"]["route"]["credentialProfileId"] == "account-b"
    assert usage["claudexor"]["account_rotation"] == {"refused_accounts": ["account-a"]}
    first, second = (payload for payload, _key in gateway.uploads)
    assert first["account"] == {"mode": "auto", "preferredProfileId": "account-a"}
    # Same model, effort, messages and tools: only the refused account's preference is gone.
    assert second == {**first, "account": {"mode": "auto"}}
    assert len(gateway.accepted_operations) == 2 and [row["state"] for row in ledger(root)].count("settled") == 2
    assert [(row["account"], row["disposition"]) for row in _rows(root, "model_account_rotation")] == [
        ("account-a", "reask")]


@pytest.mark.parametrize("dispatch", ["response_received", "not_started"])
def test_a_pinned_route_is_never_rotated(setup, monkeypatch, dispatch):
    _root, gateway, client = setup
    monkeypatch.setenv(MODEL_ACCOUNTS_KEY, json.dumps({"main": "account-a"}))
    gateway.results, gateway.dispatch = [_vendor_quota()], [dispatch]
    with pytest.raises(transport.ClaudexorModelError) as refused:
        client.chat(_history(), MODEL, model_role="main", cache_affinity="affinity")
    assert [payload["account"] for payload, _key in gateway.uploads] == [{"mode": "pin", "profileId": "account-a"}]
    assert refused.value.account_rotation == {
        "refused_accounts": ["account-a"], "stop": "pinned_account", "pool_exhausted": False}
    assert len(gateway.accepted_operations) == 1


@pytest.mark.parametrize("second,stop,pool_exhausted", [
    (_vendor_quota(), "engine_reselected_refused_account", False),
    (_pool_quota(), "pool_exhausted", True),
])
def test_rotation_ends_on_engine_evidence_and_claims_exhaustion_only_from_the_pool_verdict(
        setup, second, stop, pool_exhausted):
    """A re-selected refused account is not asked a third time, and proves nothing about the pool."""
    root, gateway, client = setup
    gateway.results = [_vendor_quota(), second]
    gateway.dispatch = ["response_received", "response_received" if second["route"] else "not_started"]
    with pytest.raises(transport.ClaudexorModelError) as refused:
        client.chat(_history(), MODEL, model_role="main", cache_affinity="affinity")
    assert len(gateway.accepted_operations) == 2
    assert refused.value.account_rotation == {
        "refused_accounts": ["account-a"], "stop": stop, "pool_exhausted": pool_exhausted}
    assert [row["disposition"] for row in _rows(root, "model_account_rotation")] == ["reask", stop]


@pytest.mark.parametrize("refusal,dispatch,stop", [
    (_pool_quota(), "not_started", "pool_exhausted"),
    (_vendor_quota(http=False), "response_received", "generation_not_excluded"),
])
def test_an_engine_verdict_or_a_stream_level_refusal_is_never_asked_again(setup, refusal, dispatch, stop):
    root, gateway, client = setup
    gateway.results, gateway.dispatch = [refusal], [dispatch]
    with pytest.raises(transport.ClaudexorModelError) as refused:
        client.chat(_history(), MODEL, model_role="main")
    assert refused.value.account_rotation["stop"] == stop and len(gateway.accepted_operations) == 1


def test_an_unknown_outcome_is_never_rotated_or_resent(setup):
    root, gateway, client = setup
    gateway.results, gateway.dispatch = [result(outcome="unknown")], ["unknown"]
    with pytest.raises(transport.ClaudexorModelError) as unknown:
        client.chat(_history(), MODEL, model_role="main")
    assert unknown.value.code == "model_outcome_unknown" and len(gateway.accepted_operations) == 1
    assert not _rows(root, "model_account_rotation")


def test_a_spent_send_budget_ends_rotation(setup):
    _root, gateway, client = setup
    gateway.results, gateway.dispatch = [_vendor_quota(), result(route=ROUTE_B)], ["response_received"] * 2
    with ua.physical_attempt_limit(1), pytest.raises(transport.ClaudexorModelError) as refused:
        client.chat(_history(), MODEL, model_role="main")
    assert refused.value.account_rotation["stop"] == "send_budget_spent"
    assert len(gateway.accepted_operations) == 1


# -- the round: configured fallback before the owner, and the owner from the retained refusal ------


@pytest.fixture
def one_fallback(monkeypatch):
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", FALLBACK)
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.setattr(fallback_cooldown, "is_cooling_down", lambda *_args: False)


def _api_fallback(ctx, monkeypatch, *, answer):
    """The configured API fallback answers or fails; the subscription primary stays real."""
    calls, remote = [], ctx.llm._chat_remote

    def send(target, messages, schemas, *args, **kwargs):
        if target["provider"] == "claudexor":
            return remote(target, messages, schemas, *args, **kwargs)
        calls.append(target["provider"])
        if not answer:
            raise RuntimeError("fixture API fallback failed")
        return {"role": "assistant", "content": "Finished by the fallback"}, {
            "prompt_tokens": 1, "completion_tokens": 1, "cost": 0.0, "provider": "openai"}

    monkeypatch.setattr(ctx.llm, "_chat_remote", send)
    return calls


def _run(ctx, tools, events):
    return loop.run_llm_loop(ctx.messages, tools, ctx.llm, ctx.drive_logs, lambda *_args, **_kwargs: None,
                             queue.Queue(), task_id="task-one", drive_root=ctx.drive_root, event_queue=events)


def test_after_the_fallback_fails_the_owner_wait_opens_from_the_retained_refusal(main_call, one_fallback, monkeypatch):
    """No primary generation opens the owner question; the primary sends again only after it resolves."""
    ctx, gateway, owner, events, _decide, _observations = main_call
    tools = _loop_tools(ctx, owner)
    gateway.results = [_pool_quota(), _final("Finished after the owner wait")]
    gateway.dispatch = ["not_started", "response_received"]
    api_calls = _api_fallback(ctx, monkeypatch, answer=False)
    at_owner_question = []
    catalog = ctx.llm.claudexor_model_catalog

    def observed_catalog(*args, **kwargs):
        at_owner_question.append((len(gateway.accepted_operations), list(api_calls),
                                  len({event["wait_id"] for event in _waits(events)})))
        return catalog(*args, **kwargs)

    monkeypatch.setattr(ctx.llm, "claudexor_model_catalog", observed_catalog)
    text, _usage, _trace = _run(ctx, tools, events)
    assert text == "Finished after the owner wait"
    # The wait was already open, after the fallback, with only the ORIGINAL refusal sent.
    assert at_owner_question[0] == (1, ["openai"], 1)
    waits = _waits(events)
    assert len({event["wait_id"] for event in waits}) == 1 and waits[-1]["resolution"] == "resource_available"
    assert waits[0]["reason"] == "quota" and waits[0]["reset_at"] == RESET
    assert waits[0]["account_rotation"] == {"refused_accounts": [], "stop": "pool_exhausted", "pool_exhausted": True}
    assert len(gateway.accepted_operations) == 2


def test_a_fallback_answer_leaves_no_owner_question_and_no_refusal(main_call, one_fallback, monkeypatch):
    ctx, gateway, owner, events, _decide, _observations = main_call
    tools = _loop_tools(ctx, owner)
    gateway.results, gateway.dispatch = [_pool_quota()], ["not_started"]
    api_calls = _api_fallback(ctx, monkeypatch, answer=True)
    monkeypatch.setattr(ctx.llm, "claudexor_model_catalog", lambda *_a, **_kw: pytest.fail("no owner question"))
    text, usage, _trace = _run(ctx, tools, events)
    assert text == "Finished by the fallback" and api_calls == ["openai"]
    assert not _waits(events) and RESOURCE_REFUSAL_KEY not in usage and len(gateway.accepted_operations) == 1


def test_an_eligible_unknown_fallback_outcome_still_opens_the_primary_refusal_wait(main_call, one_fallback, monkeypatch):
    """Owner decision 1A: a candidate's eligible unknown outcome does not fence the round. With no
    configured route left, the primary's retained refusal opens its own visible owner wait (never a
    sleep to its reset), and the send after that wait is a NEW generation carrying one facts-only
    input naming the unknown candidate attempt; the old attempt keeps its own identity."""
    ctx, gateway, owner, _events, _decide, _observations = main_call
    gateway.results, gateway.dispatch = [_pool_quota()], ["not_started"]
    assert loop._call_round_model(ctx)[0] is None and ctx.tools._ctx._deferred_resource_refusal is not None
    original = loop._call_round_model
    sent, asked = [], []

    def candidate(round_call):
        sent.append((round_call.active_model, round_call.defer_resource_wait, [dict(row) for row in round_call.messages]))
        if round_call.defer_resource_wait is False:  # the retained refusal's own send after the owner wait
            return {"role": "assistant", "content": "after the wait"}, 0.0, "max"
        round_call.accumulated_usage.update(_last_llm_error_kind="provider_outcome_unknown", _pending_transport_outcome={
            "physical_attempt_id": "fallback-unknown", "operation_id": "", "outcome": "unknown"})
        return None, 0.0, "max"

    monkeypatch.setattr(loop, "_call_round_model", candidate)
    monkeypatch.setattr(model_wait.ResourceDeferral, "ask_owner",
                        lambda deferral, _waiter: asked.append(deferral.fact["reason"]))
    message, *_rest = loop._run_cross_model_fallback_chain(
        llm=ctx.llm, ctx=ctx.tools._ctx, tools=ctx.tools, messages=ctx.messages, active_model=ctx.active_model,
        active_use_local=False, tool_schemas=[], active_effort="medium", max_retries=3, drive_logs=ctx.drive_logs,
        task_id=ctx.task_id, round_idx=1, event_queue=_events, accumulated_usage=ctx.accumulated_usage,
        task_type="task", emit_progress=lambda *_a, **_kw: None, context_fit_plan=ctx.context_fit_plan,
        active_context_mode="max")
    monkeypatch.setattr(loop, "_call_round_model", original)
    assert message["content"] == "after the wait" and asked == ["quota"]
    assert [(model, defer) for model, defer, _rows in sent] == [(FALLBACK, True), (MODEL, False)]
    assert not any("[Transport recovery]" in str(row.get("content")) for row in sent[0][2])
    notices = [row for row in sent[-1][2] if "[Transport recovery]" in str(row.get("content"))]
    assert len(notices) == 1 and "fallback-unknown" in notices[0]["content"]
    assert RESOURCE_REFUSAL_KEY not in ctx.accumulated_usage and len(gateway.accepted_operations) == 1


def test_a_readable_unknown_fallback_operation_fences_the_round(main_call, one_fallback, monkeypatch):
    """Disallowed counterpart: an unknown whose accepted operation could not be READ may still
    finish, so nothing further is sent — not the owner question's send, not a forced final."""
    ctx, gateway, owner, _events, _decide, _observations = main_call
    gateway.results, gateway.dispatch = [_pool_quota()], ["not_started"]
    assert loop._call_round_model(ctx)[0] is None
    original = loop._call_round_model

    def candidate(round_call):
        assert round_call.defer_resource_wait is True
        round_call.accumulated_usage.update(_last_llm_error_kind="provider_outcome_unknown", _pending_transport_outcome={
            "physical_attempt_id": "still-readable", "outcome": "unknown", "same_operation_recoverable": True})
        return None, 0.0, "max"

    monkeypatch.setattr(loop, "_call_round_model", candidate)
    monkeypatch.setattr(model_wait.ResourceDeferral, "ask_owner",
                        lambda *_a: pytest.fail("no owner wait may send over a readable operation"))
    message, *_rest = loop._run_cross_model_fallback_chain(
        llm=ctx.llm, ctx=ctx.tools._ctx, tools=ctx.tools, messages=ctx.messages, active_model=ctx.active_model,
        active_use_local=False, tool_schemas=[], active_effort="medium", max_retries=3, drive_logs=ctx.drive_logs,
        task_id=ctx.task_id, round_idx=1, event_queue=_events, accumulated_usage=ctx.accumulated_usage,
        task_type="task", emit_progress=lambda *_a, **_kw: None, context_fit_plan=ctx.context_fit_plan,
        active_context_mode="max")
    monkeypatch.setattr(loop, "_call_round_model", original)
    assert message is None and not _waits(_events) and len(gateway.accepted_operations) == 1
    assert provider_no_call_source(ctx.accumulated_usage, False)[0] == "provider_outcome_unknown_no_resend"


def test_a_candidate_defers_only_while_a_later_route_or_the_owner_question_follows(main_call, monkeypatch):
    ctx, _gateway, _owner, _events, _decide, _observations = main_call
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", "openai::one,openai::two")
    monkeypatch.setattr(fallback_cooldown, "is_cooling_down", lambda *_args: False)
    monkeypatch.setattr(loop, "_rebind_context_fit_plan", lambda plan, *_a, **_kw: (plan, "max"))
    seen = []
    monkeypatch.setattr(loop, "_call_round_model",
                        lambda call: seen.append((call.active_model, call.defer_resource_wait)) or (None, 0.0, "max"))
    loop._run_cross_model_fallback_chain(
        llm=ctx.llm, ctx=ctx.tools._ctx, tools=ctx.tools, messages=ctx.messages, active_model=ctx.active_model,
        active_use_local=False, tool_schemas=[], active_effort="medium", max_retries=1, drive_logs=ctx.drive_logs,
        task_id=ctx.task_id, round_idx=1, event_queue=None, accumulated_usage={}, task_type="task",
        emit_progress=lambda *_a, **_kw: None, context_fit_plan=ctx.context_fit_plan, active_context_mode="max")
    assert seen == [("openai::one", True), ("openai::two", False)]


def test_presence_retains_a_fallback_only_quota_refusal(main_call, one_fallback, monkeypatch):
    ctx, _gateway, owner, _events, _decide, _observations = main_call
    tools = _loop_tools(ctx, owner)
    monkeypatch.setattr(loop, "_rebind_context_fit_plan", lambda plan, *_a, **_kw: (plan, "max"))
    observed = []

    def quota_candidate(call):
        observed.append(call.defer_resource_wait)
        call.tools._ctx._deferred_resource_refusal = SimpleNamespace(
            fact={"reason": "quota", "reset_at": RESET},
            terminal=lambda **fields: {"reason": "quota", "reset_at": RESET, "temporary": True, **fields})
        call.accumulated_usage["_last_llm_error_kind"] = "quota_exhausted"
        return None, 0.0, "max"

    monkeypatch.setattr(loop, "_call_round_model", quota_candidate)
    usage = {"_last_llm_error_kind": "bad_request"}  # primary was NOT the resource refusal
    message, *_ = loop._run_cross_model_fallback_chain(
        llm=ctx.llm, ctx=tools._ctx, tools=tools, messages=ctx.messages, active_model=ctx.active_model,
        active_use_local=False, tool_schemas=[], active_effort="medium", max_retries=1,
        drive_logs=ctx.drive_logs, task_id=ctx.task_id, round_idx=1, event_queue=None,
        accumulated_usage=usage, task_type="presence", emit_progress=lambda *_a, **_kw: None,
        context_fit_plan=ctx.context_fit_plan, active_context_mode="max")
    assert message is None and observed == [True]
    assert usage[RESOURCE_REFUSAL_KEY]["reason"] == "quota"
    assert provider_no_call_source(usage, False)[0] == "resource_refusal_no_resend"


def test_presence_keeps_intermediate_fallback_refusal_after_later_bad_request(main_call, monkeypatch):
    ctx, _gateway, owner, _events, _decide, _observations = main_call
    tools = _loop_tools(ctx, owner)
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", "openai::one,openai::two")
    monkeypatch.setattr(fallback_cooldown, "is_cooling_down", lambda *_args: False)
    monkeypatch.setattr(loop, "_rebind_context_fit_plan", lambda plan, *_a, **_kw: (plan, "max"))
    seen = []

    def candidate(call):
        seen.append(call.active_model)
        # Mirror _dispatch_round_model's per-candidate transient slot reset.
        call.tools._ctx._deferred_resource_refusal = None
        if call.active_model == "openai::one":
            call.tools._ctx._deferred_resource_refusal = SimpleNamespace(
                fact={"reason": "quota", "reset_at": RESET},
                terminal=lambda **fields: {"reason": "quota", "reset_at": RESET, "temporary": True, **fields})
            call.accumulated_usage["_last_llm_error_kind"] = "quota_exhausted"
        else:
            call.accumulated_usage["_last_llm_error_kind"] = "bad_request"
        return None, 0.0, "max"

    monkeypatch.setattr(loop, "_call_round_model", candidate)
    usage = {"_last_llm_error_kind": "bad_request"}
    message, *_ = loop._run_cross_model_fallback_chain(
        llm=ctx.llm, ctx=tools._ctx, tools=tools, messages=ctx.messages, active_model=ctx.active_model,
        active_use_local=False, tool_schemas=[], active_effort="medium", max_retries=1,
        drive_logs=ctx.drive_logs, task_id=ctx.task_id, round_idx=1, event_queue=None,
        accumulated_usage=usage, task_type="presence", emit_progress=lambda *_a, **_kw: None,
        context_fit_plan=ctx.context_fit_plan, active_context_mode="max")
    assert message is None and seen == ["openai::one", "openai::two"]
    assert usage[RESOURCE_REFUSAL_KEY]["fallbacks_tried"] == seen
    assert provider_no_call_source(usage, False)[0] == "resource_refusal_no_resend"


def test_ordinary_task_keeps_intermediate_fallback_quota_for_owner_wait(main_call, monkeypatch):
    """A non-quota primary and a later bad route must not erase an earlier fallback's quota."""
    ctx, _gateway, owner, _events, _decide, _observations = main_call
    tools = _loop_tools(ctx, owner)
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", "openai::one,openai::two")
    monkeypatch.setattr(fallback_cooldown, "is_cooling_down", lambda *_args: False)
    rebounds = []

    def rebind(plan, *_args, **kwargs):
        account = kwargs.get("credential_profile_id")
        rebounds.append(account)
        if account == "account-b":
            return replace(plan, model_route={"accountFingerprint": "B", "credentialProfileId": account}), "max"
        return plan, "max"

    monkeypatch.setattr(loop, "_rebind_context_fit_plan", rebind)
    waiter = SimpleNamespace(waits_allowed=True, overrides={})
    monkeypatch.setattr(model_wait, "current_model_wait", lambda: waiter)
    seen, asked = [], []

    def candidate(call):
        seen.append((call.active_model, call.defer_resource_wait))
        call.tools._ctx._deferred_resource_refusal = None
        if call.active_model == "openai::one":
            if asked:
                assert asked == [waiter]
                assert call.context_fit_plan.model_route["accountFingerprint"] == "B"
                return {"role": "assistant", "content": "after owner wait"}, 0.0, "max"
            call.tools._ctx._deferred_resource_refusal = SimpleNamespace(
                fact={"reason": "quota", "reset_at": RESET},
                ask_owner=lambda _waiter: (asked.append(_waiter), waiter.overrides.update(  # the wait's answer: a dict
                    {"fallback:0": {"model_account_override": "account-b"}}), {})[-1],
                terminal=lambda **fields: {"reason": "quota", "reset_at": RESET, **fields})
            call.accumulated_usage["_last_llm_error_kind"] = "quota_exhausted"
            return None, 0.0, "max"
        assert call.active_model == "openai::two"
        call.accumulated_usage["_last_llm_error_kind"] = "bad_request"
        return None, 0.0, "max"

    monkeypatch.setattr(loop, "_call_round_model", candidate)
    usage = {"_last_llm_error_kind": "bad_request"}
    message, *_ = loop._run_cross_model_fallback_chain(
        llm=ctx.llm, ctx=tools._ctx, tools=tools, messages=ctx.messages, active_model=ctx.active_model,
        active_use_local=False, tool_schemas=[], active_effort="medium", max_retries=1,
        drive_logs=ctx.drive_logs, task_id=ctx.task_id, round_idx=1, event_queue=None,
        accumulated_usage=usage, task_type="task", emit_progress=lambda *_a, **_kw: None,
        context_fit_plan=ctx.context_fit_plan, active_context_mode="max")
    assert message["content"] == "after owner wait"
    assert seen == [("openai::one", True), ("openai::two", True), ("openai::one", False)]
    assert tools._ctx.active_model == "openai::one"
    assert rebounds[-1] == "account-b"
    assert RESOURCE_REFUSAL_KEY not in usage


def _two_route_walk(ctx, owner, monkeypatch, overrides=None):
    """Two configured routes with live waiting, every recovery read through the round's own caller."""
    tools = _loop_tools(ctx, owner)
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", "openai::one,openai::two")
    monkeypatch.setattr(fallback_cooldown, "is_cooling_down", lambda *_args: False)
    monkeypatch.setattr(loop, "_rebind_context_fit_plan", lambda plan, *_a, **_kw: (plan, "max"))
    waiter = SimpleNamespace(waits_allowed=True, overrides=dict(overrides or {}))
    monkeypatch.setattr(model_wait, "current_model_wait", lambda: waiter)
    reconciled = []
    monkeypatch.setattr(loop, "_reconcile_transport_wait",
                        lambda episode, _ctx, **kw: reconciled.append(kw["error_kind"]) or episode)
    return tools, waiter, reconciled


def _recover(ctx, tools, entry):
    ctx.accumulated_usage["_last_llm_error_kind"] = entry
    return loop._recover_failed_round(ctx, tools, None, None, context_fit_plan=ctx.context_fit_plan,
                                      active_context_mode="max", emit_progress=lambda *_a, **_kw: None)


@pytest.mark.parametrize("entry", ["transport_unavailable", "bad_request"])
def test_the_last_backup_keeps_its_quota_for_the_rounds_own_outage_wait(main_call, monkeypatch, entry):
    """The round's own outage owns its wait: the last configured route's quota refusal is retained for it, never
    waited on in that route's call, and the round goes back to its own transport wait with no owner question or
    terminal refusal. A round without a wait of its own lets the last route wait on its quota in its call."""
    ctx, _gateway, owner, _events, _decide, _observations = main_call
    tools, _waiter, reconciled = _two_route_walk(ctx, owner, monkeypatch)
    seen = []

    def candidate(call):
        seen.append((call.active_model, call.defer_resource_wait))
        call.tools._ctx._deferred_resource_refusal = None
        if call.active_model == "openai::one":
            call.accumulated_usage["_last_llm_error_kind"] = "bad_request"
            return None, 0.0, "max"
        if not call.defer_resource_wait:  # its own call waits on its quota, then answers
            return {"role": "assistant", "content": "after its own quota wait"}, 0.0, "max"
        call.tools._ctx._deferred_resource_refusal = SimpleNamespace(
            fact={"reason": "quota", "reset_at": RESET},
            ask_owner=lambda _waiter: pytest.fail("the round's own outage owns its wait"),
            terminal=lambda **_fields: pytest.fail("a refusal kept for the round's own wait is no terminal"))
        call.accumulated_usage["_last_llm_error_kind"] = "quota_exhausted"
        return None, 0.0, "max"

    monkeypatch.setattr(loop, "_call_round_model", candidate)
    message, model, *_rest = _recover(ctx, tools, entry)
    if entry == "transport_unavailable":
        assert message is None and model == ctx.active_model
        assert seen == [("openai::one", True), ("openai::two", True)]
        assert ctx.accumulated_usage["_last_llm_error_kind"] == entry and reconciled == [entry]
        assert RESOURCE_REFUSAL_KEY not in ctx.accumulated_usage
    else:
        assert message["content"] == "after its own quota wait" and model == "openai::two"
        assert seen == [("openai::one", True), ("openai::two", False)]


@pytest.mark.parametrize("answer", ["switch", "retry", "earlier_switch", "reselect_same", "switch_in_its_call"])
def test_an_owners_switch_stays_the_owners_choice_after_the_backup_answers(main_call, monkeypatch, answer):
    """A configured route's quota refusal reaches the owner after the walk (or, for the last route, in its own
    call). A Switch on that card is the owner's choice of the route and stays one when the route answers and is
    adopted, even a Switch to the very binding an earlier card already chose. Retry of an earlier card's choice,
    or of none, adopts the route as the configured fallback it is: no new owner choice is recorded."""
    from ouroboros.route_choice import explicit

    ctx, _gateway, owner, _events, _decide, _observations = main_call
    picked = {"model": "openai::owner-pick", "use_local": False, "model_account_override": ""}
    tools, waiter, _reconciled = _two_route_walk(
        ctx, owner, monkeypatch, {"fallback:0": dict(picked)} if answer in {"earlier_switch", "reselect_same"} else None)
    tools._ctx.primary_route = {"model": ctx.active_model, "use_local": False, "role": "main"}
    tools._ctx.event_queue = leads = queue.Queue()
    asked, sent = [], []

    def ask_owner(_waiter):  # the wait's answer: the call's own values, a Switch's choice over them
        asked.append(answer)
        if answer in {"switch", "reselect_same"}:  # as ``TaskModelWait.wait`` applies a Switch: a new entry, same fields or not
            waiter.overrides["fallback:0"] = dict(picked)
        return {"model": "openai::one" if answer == "retry" else picked["model"], "use_local": False}

    def candidate(call):
        sent.append((call.active_model, call.model_role, call.defer_resource_wait))
        call.tools._ctx._deferred_resource_refusal = None
        if asked:
            return {"role": "assistant", "content": "after the owner's answer"}, 0.0, "max"
        if answer == "switch_in_its_call" and call.model_role == "fallback:1":
            # The last route waits in its own call; the owner's Switch reprepares it there (``_reprepare_waiting_main``).
            waiter.overrides["fallback:1"] = dict(picked)
            call.tools._ctx.route_choice = explicit("wait_card")
            call.active_model = picked["model"]
            return {"role": "assistant", "content": "after the owner's answer"}, 0.0, "max"
        if call.model_role == "fallback:0" and answer != "switch_in_its_call":
            call.tools._ctx._deferred_resource_refusal = SimpleNamespace(
                fact={"reason": "quota", "reset_at": RESET}, ask_owner=ask_owner,
                terminal=lambda **fields: {"reason": "quota", "reset_at": RESET, **fields})
            call.accumulated_usage["_last_llm_error_kind"] = "quota_exhausted"
        else:
            call.accumulated_usage["_last_llm_error_kind"] = "bad_request"
        return None, 0.0, "max"

    monkeypatch.setattr(loop, "_call_round_model", candidate)
    message, model, *_rest = _recover(ctx, tools, "bad_request")
    expected = "openai::one" if answer == "retry" else picked["model"]
    assert message["content"] == "after the owner's answer" and model == expected
    if answer == "switch_in_its_call":  # its own call, not a deferral, held the card
        assert asked == [] and sent[-1] == ("openai::two", "fallback:1", False)
    else:
        assert asked == [answer] and sent[-1] == (expected, "fallback:0", False)
    lead = [row["lead"] for row in list(leads.queue) if row.get("type") == "work_lead"][-1]
    assert lead["model"] == expected
    if answer in {"switch", "reselect_same", "switch_in_its_call"}:
        assert tools._ctx.route_choice == explicit("wait_card") and (lead["basis"], lead["cause"]) == ("owner", "wait_card")
    else:
        assert tools._ctx.route_choice == explicit("fallback") and lead["basis"] == "fallback" and lead["cause"] != "wait_card"


_PICKED = {"model": "openai::owner-pick", "use_local": False, "model_account_override": ""}


def _wait_card_walk(main_call, monkeypatch, preset=None):
    """The walks below: Main, two configured routes and a live owner wait whose card may switch a route's role, with
    the loop's own checkpoint writer bound to the round (``limit_ctx``)."""
    from ouroboros import loop_model_call, route_choice

    ctx, gateway, owner, *_rest = main_call
    tools, _waiter, _reconciled = _two_route_walk(ctx, owner, monkeypatch, preset)
    owner.overrides.clear()
    owner.overrides.update(preset or {})
    monkeypatch.setattr(model_wait, "current_model_wait", lambda: owner)
    monkeypatch.setenv("OUROBOROS_MODEL", ctx.active_model)
    monkeypatch.setenv("USE_LOCAL_MAIN", "false")
    monkeypatch.setattr(loop, "_rebind_context_fit_plan", lambda plan, *_a, model, model_role="", **_kw: (
        replace(plan, model=model, provider="openai", model_role=model_role or "main", model_route={}), "max"))
    monkeypatch.setattr(loop, "_measure_round_main_fit", lambda *_a, **_kw: None)
    monkeypatch.setattr(loop_llm_call, "_prepare_main_messages", lambda msgs, **_kw: msgs)
    monkeypatch.setattr(loop_model_call, "_fit_route_tool_ceiling", lambda *_a, **_kw: None)
    slot = tools._ctx
    slot.task_id, slot.drive_root, slot.budget_drive_root = ctx.task_id, ctx.drive_root, ctx.drive_root
    slot.task_model_override = slot.task_use_local_override = slot.task_route_source = None
    slot.active_model, slot.active_use_local, slot.active_effort, slot.active_context_mode = (
        ctx.active_model, False, "medium", "max")
    slot.primary_route = {"model": ctx.active_model, "use_local": False, "role": "main"}
    slot.route_choice, slot.model_wait_context, slot.event_queue = route_choice.following("main"), owner, queue.Queue()
    return ctx, gateway, owner, tools, SimpleNamespace(
        tools=tools, messages=ctx.messages, llm_trace={"tool_calls": []}, accumulated_usage=ctx.accumulated_usage,
        owner_msg_seen=set(), round_idx=ctx.round_idx, tool_schemas=ctx.tool_schemas, budget_tail="tool")


def _route(tool_ctx, primary):
    """The work's next move and its chooser, its published lead, and what a child it starts inherits."""
    from ouroboros import lead_fact, supervisor_route

    lead = lead_fact.lead_fact(tool_ctx)
    child = supervisor_route._inherited_cognition(tool_ctx, {"OUROBOROS_MODEL": primary})
    return {"model": tool_ctx.active_model, "choice": tool_ctx.route_choice,
            "lead": (lead["model"], lead["basis"], lead.get("cause")),
            "child": (child["model"], child["chosen_by"], child.get("follows_role"))}


def _resumed_route(ctx, saved, primary):
    """``_route`` of a new attempt restored from one saved working state, as a lost worker's successor starts."""
    from ouroboros import owner_wait
    from ouroboros.tools.registry import ToolRegistry

    rtools = ToolRegistry(repo_dir=ctx.drive_root, drive_root=ctx.drive_root)
    rctx = rtools._ctx
    rctx.task_id, rctx.task_attempt, rctx.drive_root, rctx.budget_drive_root, rctx.task_metadata = (
        ctx.task_id, 2, ctx.drive_root, ctx.drive_root, {})
    with model_wait.task_model_wait_scope(
        task={"id": ctx.task_id, "_attempt": 2, "drive_root": str(ctx.drive_root)},
        drive_root=ctx.drive_root, event_queue=None, worker_slot_held=False, owner_control=lambda: None,
    ) as rwaiter:
        rwaiter.restore_continuation(saved.get("model_wait") or {}, started_at=saved.get("started_at"))
        rctx.model_wait_context = rwaiter
        owner_wait.restore_continuation_state(rtools, saved, [], {}, {}, set())
        rctx.context_fit_plan = SimpleNamespace(model=rctx.active_model, provider="openai",
                                                model_role=saved.get("context_model_role") or "main")
        return _route(rctx, primary)


@pytest.mark.parametrize("answer", [
    "switch", "retry", "earlier_switch", "reselect_same",
    "switch_in_its_call", "retry_in_its_call_with_earlier_override", "failed_after_switch",
])
def test_intermediate_checkpoint_after_wait_card_matches_adoption(main_call, monkeypatch, answer):
    """Intermediate checkpoint on output exhaustion matches adoption for parent and inheriting child."""
    from ouroboros import loop_model_call, loop_output_exhaustion as output_recovery, route_choice, working_checkpoint as wc

    picked = dict(_PICKED)
    preset = ({"fallback:0": dict(picked)} if answer in {"earlier_switch", "reselect_same"}
              else {"fallback:1": dict(picked)} if answer == "retry_in_its_call_with_earlier_override" else None)
    ctx, _gw, owner, tools, limit_ctx = _wait_card_walk(main_call, monkeypatch, preset)
    primary, slot = ctx.active_model, tools._ctx
    asked, in_call = [], answer in {"switch_in_its_call", "retry_in_its_call_with_earlier_override"}

    def ask_owner(_w):
        asked.append(answer)
        if answer in {"switch", "reselect_same", "failed_after_switch"}:
            owner.overrides["fallback:0"] = dict(picked)
            return dict(picked)
        return {"model": "openai::one" if answer == "retry" else picked["model"], "use_local": False}

    def in_call_transport(_llm, _msgs, _model, _schemas, *_a, model_role="", **_kw):
        if answer == "switch_in_its_call":
            owner.overrides[model_role] = dict(picked)
        owner.reprepare(model_role, {"messages": list(ctx.messages), "model": picked["model"],
                                     "use_local": False, "model_role": model_role, "tools": []})
        ctx.accumulated_usage.update(_last_llm_error_kind=output_recovery.KIND,
            _last_llm_output_exhausted={"finish_reason": "length", "model": picked["model"]})
        return None, 0.0

    def fake_once(call):
        call.tools._ctx._deferred_resource_refusal = None
        if in_call:
            if call.model_role == "fallback:0":
                call.accumulated_usage["_last_llm_error_kind"] = "bad_request"
                return None, 0.0, "max"
            return loop_model_call._dispatch_round_model(call, None, attempt_cap=1)[0], 0.0, call.active_context_mode
        if not asked:
            if call.model_role == "fallback:0":
                call.tools._ctx._deferred_resource_refusal = SimpleNamespace(
                    fact={"reason": "quota", "reset_at": RESET}, ask_owner=ask_owner,
                    terminal=lambda **fields: {"reason": "quota", "reset_at": RESET, **fields})
                call.accumulated_usage["_last_llm_error_kind"] = "quota_exhausted"
            else:
                call.accumulated_usage["_last_llm_error_kind"] = "bad_request"
            return None, 0.0, "max"
        if answer == "failed_after_switch":
            call.accumulated_usage["_last_llm_error_kind"] = "bad_request"
            return None, 0.0, "max"
        call.accumulated_usage.update(_last_llm_error_kind=output_recovery.KIND,
            _last_llm_output_exhausted={"finish_reason": "length", "model": call.active_model})
        return None, 0.0, "max"

    monkeypatch.setattr(loop_model_call, "_call_round_model_once", fake_once)
    monkeypatch.setattr(loop, "call_llm_with_retry", in_call_transport)
    with wc.model_call_scope(limit_ctx):
        message, model, *_rest = _recover(ctx, tools, "bad_request")
    if answer == "failed_after_switch":
        assert message is None and model == primary and slot.route_choice == route_choice.following("main")
        assert not wc.read_working(ctx.drive_root, ctx.task_id, 1)[1]
        return

    expected = "openai::one" if answer == "retry" else picked["model"]
    assert message is None and model == expected
    _, saved = wc.read_working(ctx.drive_root, ctx.task_id, 1)
    assert saved and saved["route"]["active_model"] == expected
    live, restored = _route(slot, primary), _resumed_route(ctx, saved, primary)
    switched = answer in {"switch", "reselect_same", "switch_in_its_call"}
    expected_choice = route_choice.explicit("wait_card" if switched else "fallback")
    assert slot.route_choice == saved["route"]["route_choice"] == restored["choice"] == expected_choice
    assert live["lead"][1:] == restored["lead"][1:] and live["child"][:2] == restored["child"][:2]
    if switched:
        assert restored["lead"][1:] == ("owner", "wait_card") and restored["child"][:2] == (expected, "parent_switch")
    else:
        assert restored["lead"][1] == "fallback" and restored["lead"][2] != "wait_card"
        assert restored["child"] == (primary, "inherited", "main")


@pytest.mark.parametrize("answer", [
    "switch_fails", "retry_fails", "earlier_switch_fails", "switch_length", "switch_unsent", "switch_paused", "switch_stopped",
])
def test_work_lost_during_a_walk_for_an_unfinished_reply_resumes_on_the_live_route(main_call, monkeypatch, answer):
    """An unfinished reply's owed turn claims every send of its round, the walk's trials included. Main refuses that
    turn, B refuses on quota, the owner answers B's card. B's real request then fails with HTTP 400, ends empty at its
    length limit, never leaves the host, or meets the owner's Pause or Stop between its saved claim and its send. Each
    version written from B's claim to the walk's exit is one a lost worker resumes from: B's claim stays pending, so a
    new execution ends it unrepeated, until B's end is saved once with the route the work keeps. A reverted B is
    nobody's route, an adopted Switch stays the owner's, a Stop or Pause goes on unchanged, and no route is sent twice."""
    import httpx
    import openai
    from ouroboros import loop_model_call, loop_output_exhaustion as output_recovery, pricing, request_wire_contract
    from ouroboros import route_choice, working_checkpoint as wc
    from ouroboros.llm import LLMClient

    preset = {"fallback:0": dict(_PICKED)} if answer == "earlier_switch_fails" else None
    ctx, gateway, owner, tools, limit_ctx = _wait_card_walk(main_call, monkeypatch, preset)
    primary, slot, key, ctx.tools = ctx.active_model, tools._ctx, output_recovery.UNFINISHED_KEY, tools
    adopted, control = answer == "switch_length", {"switch_paused": "owner_pause", "switch_stopped": "cancelled"}.get(answer)
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-not-live")
    monkeypatch.setattr(request_wire_contract, "canonical_wire_evidence_root", lambda: ctx.drive_root)
    monkeypatch.setattr(pricing, "estimate_cost_optional", lambda *_a, **_kw: None)
    monkeypatch.setattr(ua, "_reservation_cost", lambda _request: 1.0)
    monkeypatch.setattr(LLMClient, "_get_supported_parameters", lambda *_a, **_kw: None)
    sent, written, write = [], [], wc._write  # the routes' real HTTP sends; each working state, as written

    def respond(request):
        body = json.loads(request.content)
        if answer == "switch_unsent" and body["model"] == "owner-pick":
            raise httpx.ConnectError("connection refused", request=request)  # no request bytes left the host
        sent.append(body["model"])
        if not adopted or body["model"] != "owner-pick":
            return httpx.Response(400, json={"error": {"message": "Upstream request failed"}})
        delta, usage = {"role": "assistant", "content": ""}, {"prompt_tokens": 100, "completion_tokens": 50, "cost": 0}
        reply = {"id": "fake", "model": body["model"], "usage": usage, "choices": [
            {"index": 0, "finish_reason": "length", "delta" if body.get("stream") else "message": delta}]}
        if body.get("stream"):
            frame = json.dumps({**reply, "object": "chat.completion.chunk"})
            return httpx.Response(200, headers={"content-type": "text/event-stream"},
                                  content=f"data: {frame}\n\ndata: [DONE]\n\n")
        return httpx.Response(200, json={**reply, "object": "chat.completion"})

    def recorded_write(*args, **kwargs):
        write(*args, **kwargs)
        written.append((len(sent), state := wc.read_working(ctx.drive_root, ctx.task_id, 1)[1]))
        if control and asked and "pending" in state["usage"][key]:  # the owner's control lands after B's claim
            monkeypatch.setattr(owner, "control_reason", lambda **_kw: control)

    sdk = openai.OpenAI(api_key="fixture-not-live", base_url="https://synthetic.invalid/v1", max_retries=0,
                        http_client=httpx.Client(transport=httpx.MockTransport(respond)))
    monkeypatch.setattr(ctx.llm, "_get_remote_client", lambda *_a, **_kw: sdk)
    monkeypatch.setattr(wc, "_write", recorded_write)
    refusal = {"code": "unsupported_parameter", "message": "unsupported_parameter"}
    gateway.results = [result(outcome="failed", route=ROUTE, problem=refusal)]  # Main refuses
    gateway.dispatch, asked, real_once = ["response_received"], [], loop_model_call._call_round_model_once

    def ask_owner(_w):  # as ``TaskModelWait.wait`` answers: a Switch stores a new entry, a Retry the call's values
        asked.append(len(written))
        if answer.startswith("switch"):
            owner.overrides["fallback:0"] = dict(_PICKED)
            return dict(_PICKED)
        return {"model": "openai::one" if answer == "retry_fails" else _PICKED["model"], "use_local": False}

    def once(call):  # B's first request is refused on quota and retained for its card; every other send is real
        call.tools._ctx._deferred_resource_refusal = None
        if call.model_role == "fallback:0" and not asked:
            call.tools._ctx._deferred_resource_refusal = SimpleNamespace(
                fact={"reason": "quota", "reset_at": RESET}, ask_owner=ask_owner,
                terminal=lambda **fields: {"reason": "quota", "reset_at": RESET, **fields})
            call.accumulated_usage["_last_llm_error_kind"] = "quota_exhausted"
            return None, 0.0, "max"
        return real_once(call)

    monkeypatch.setattr(loop_model_call, "_call_round_model_once", once)
    slot._round_response_facts = {"finish_reason": "length", "finish_reason_present": True}
    cut = {"role": "assistant", "content": "Both files were read; the first count is 12, while the second"}
    with wc.model_call_scope(limit_ctx):
        assert output_recovery.unfinished_reply(limit_ctx, cut) is None  # its informed turn is owed
        assert loop_model_call._call_round_model(ctx)[0] is None  # Main refuses the owed turn
        assert ctx.accumulated_usage["_last_llm_error_kind"] == "bad_request"
        try:
            message, model, *_rest = loop._recover_failed_round(
                ctx, tools, None, None, context_fit_plan=ctx.context_fit_plan, active_context_mode="max",
                emit_progress=lambda *_a, **_kw: None)
            stopped = None
        except model_wait.ModelWaitInterrupted as interrupted:  # the owner's own control, as it was raised
            message, model, stopped = None, slot.active_model, interrupted.control_reason

    assert stopped == control
    retried, reached = "one" if answer == "retry_fails" else "owner-pick", answer.endswith(("fails", "length"))
    assert len(asked) == 1 and len(gateway.creates) == 1 and sent == ["two", *[retried] * reached]  # none sent twice
    assert (message, model) == (None, _PICKED["model"] if adopted else primary)
    live = _route(slot, primary)
    assert live["choice"] == (route_choice.explicit("wait_card") if adopted else route_choice.following("main"))
    assert slot.context_fit_plan.model == model  # the view a Pause or the next round saves
    versions = written[asked[0]:]  # from B's claim to the walk's exit
    claimed = versions[0][1]["usage"][key]["attempt_id"]
    for sends, state in versions:
        record = state["usage"][key]
        assert (record["stage"], record["finishes"], record["attempt_id"]) == ("informed", ["length"], claimed)
        if "pending" in record:  # B's claim, saved before its send: a new execution ends it unrepeated
            probe = SimpleNamespace(accumulated_usage=json.loads(json.dumps(state["usage"])), active_context_mode="max",
                                    tools=SimpleNamespace(_ctx=SimpleNamespace()))
            output_recovery.call(probe, lambda _call: pytest.fail("a saved pending claim is never sent again"))
            assert sends == 1 and probe.accumulated_usage["_last_llm_error_kind"] == "output_recovery_interrupted"
        else:  # B's end (known, or refunded unsent: the turn stays owed), saved only with the route the work keeps
            assert _resumed_route(ctx, state, primary) == live
    assert [("pending" in state["usage"][key], sends) for sends, state in versions] == [(True, 1), (False, len(sent))]
    if reached:  # the failure row saved with B's end is B's own
        known = {name: ctx.accumulated_usage["_last_llm_failure"][name] for name in ("model", "failure_code", "physical_attempt_id")}
        assert known["failure_code"] == (output_recovery.KIND if adopted else "bad_request") and known["physical_attempt_id"] == claimed
        assert {name: versions[-1][1]["usage"]["_last_llm_failure"][name] for name in known} == known


def test_owner_selected_account_rebinds_fallback_before_physical_send(main_call, monkeypatch):
    """An A->B switch on the refused fallback must bind B's capacity in the ledger.

    Drive the real round dispatcher and fake engine transport, not a mocked
    _call_round_model: the second operation's physical-context receipt is the
    consumer of the reprepare that an in-memory route assertion cannot cover.
    """
    from copy import deepcopy
    from ouroboros import context
    from ouroboros.model_slots import MODEL_ACCOUNTS_KEY

    ctx, gateway, waiter, events, _decide, _observations = main_call
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", MODEL + ",openai::dead-fallback")
    monkeypatch.setenv("OUROBOROS_FALLBACK_ATTEMPTS_PER_MODEL", "1")
    monkeypatch.setenv(MODEL_ACCOUNTS_KEY, json.dumps({"fallback": ["account-a", ""]}))
    monkeypatch.setattr(fallback_cooldown, "is_cooling_down", lambda *_args: False)
    gateway.results = [_pool_quota(), result(route=ROUTE_B)]
    gateway.dispatch = ["not_started", "response_received"]
    ctx.active_model = "openai::unavailable-primary"
    ctx.context_fit_plan = replace(ctx.context_fit_plan, model=ctx.active_model, provider="openai")
    original = deepcopy(ctx.messages)
    observed = []
    api_calls = _api_fallback(ctx, monkeypatch, answer=False)

    def route(task, **_kwargs):
        account = task.get("credential_profile_id") or "account-a"
        observed.append((task["model"], account if task["model"] == MODEL else ""))
        subscription = task["model"] == MODEL
        return {"model": task["model"], "provider": "claudexor" if subscription else "openai"}, SimpleNamespace(
            route_fp=f"capacity-{account}" if subscription else "api-route",
            status="confirmed", stale=False,
            window_tokens=160_000 if account == "account-a" else 240_000,
            source_id="codex" if subscription else "", source="advertised",
            credential_profile_id=account if subscription else "",
            account_fingerprint=f"fingerprint-{account[-1]}" if subscription else "")

    monkeypatch.setattr(context, "_context_fit_route", route)
    asked = []

    def owner_choice(_deferral, controller):
        assert controller is waiter
        asked.append(len(gateway.accepted_operations))
        controller.overrides["fallback:0"] = {"model_account_override": "account-b"}
        return {"resolution": "owner_selected_account"}

    monkeypatch.setattr(model_wait.ResourceDeferral, "ask_owner", owner_choice)
    ctx.accumulated_usage["_last_llm_error_kind"] = "bad_request"
    message, active_model, _local, plan, _mode = loop._run_cross_model_fallback_chain(
        llm=ctx.llm, ctx=ctx.tools._ctx, tools=ctx.tools, messages=ctx.messages,
        active_model=ctx.active_model, active_use_local=False, tool_schemas=[],
        active_effort="medium", max_retries=3, drive_logs=ctx.drive_logs,
        task_id=ctx.task_id, round_idx=ctx.round_idx, event_queue=events,
        accumulated_usage=ctx.accumulated_usage, task_type="task",
        emit_progress=lambda *_a, **_kw: None, context_fit_plan=ctx.context_fit_plan,
        active_context_mode="max")
    assert message and active_model == MODEL and asked == [1]
    assert api_calls == ["openai"]  # the next configured route failed before owner selection
    assert observed[0] == (MODEL, "account-a") and observed[-1] == (MODEL, "account-b")
    assert ("openai::dead-fallback", "") in observed
    assert [upload[0]["account"] for upload in gateway.uploads] == [
        {"mode": "pin", "profileId": "account-a"},
        {"mode": "pin", "profileId": "account-b"}]
    dispatched = dispatched_attempts(ctx.drive_root)
    assert [row["physical_context"]["route_fp"] for row in dispatched] == [
        "capacity-account-a", "capacity-account-b"]
    # The attempt's frame: the line its request is admitted at (the stated total less the reply floor) plus the reserve.
    assert dispatched[-1]["physical_context"]["capacity_total_tokens"] == 240_000 - 8_192 + 65_536
    assert plan.route_fp == "capacity-account-b" and plan.window_tokens == 240_000
    # completed tools are not replayed; the answered send's clock line follows them
    from tests.test_subscription_main_wait import _without_context_facts

    canonical = _without_context_facts(ctx.messages)
    assert canonical[-3:-1] == original[-2:] and canonical[-1]["content"].startswith(CLOCK_NOTE_PREFIX)
    assert len(gateway.accepted_operations) == 2


def test_owner_selected_account_rebinds_primary_before_physical_send(main_call, monkeypatch):
    """The retained primary, not only a fallback, must replace A's capacity with B's."""
    from ouroboros import context

    ctx, gateway, waiter, events, _decide, _observations = main_call
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", "openai::dead-fallback")
    monkeypatch.setenv(MODEL_ACCOUNTS_KEY, json.dumps({"main": "account-a"}))
    monkeypatch.setattr(fallback_cooldown, "is_cooling_down", lambda *_args: False)
    ctx.context_fit_plan = replace(ctx.context_fit_plan, window_tokens=900_000, route_fp="capacity-account-a")
    gateway.results = [_pool_quota(), result(route=ROUTE_B)]
    gateway.dispatch = ["not_started", "response_received"]
    api_calls = _api_fallback(ctx, monkeypatch, answer=False)
    observed = []

    def route(task, **_kwargs):
        if task["model"] != MODEL:
            return {"model": task["model"], "provider": "openai"}, SimpleNamespace(
                route_fp="api-route", status="confirmed", stale=False, window_tokens=300_000,
                source_id="", source="advertised", credential_profile_id="", account_fingerprint="")
        account = task.get("credential_profile_id") or "account-a"
        observed.append(account)
        return {"model": MODEL, "provider": "claudexor"}, SimpleNamespace(
            route_fp=f"capacity-{account}", status="confirmed", stale=False,
            window_tokens=900_000 if account == "account-a" else 240_000,
            source_id="codex", source="advertised", credential_profile_id=account,
            account_fingerprint=f"fingerprint-{account[-1]}")

    monkeypatch.setattr(context, "_context_fit_route", route)
    asked = []

    def owner_choice(_deferral, controller):
        assert controller is waiter
        asked.append(len(gateway.accepted_operations))
        controller.overrides["main"] = {"model_account_override": "account-b"}
        return {"resolution": "owner_selected_account"}

    monkeypatch.setattr(model_wait.ResourceDeferral, "ask_owner", owner_choice)
    assert loop._call_round_model(ctx)[0] is None
    message, active_model, _local, plan, _mode = loop._run_cross_model_fallback_chain(
        llm=ctx.llm, ctx=ctx.tools._ctx, tools=ctx.tools, messages=ctx.messages,
        active_model=ctx.active_model, active_use_local=False, tool_schemas=[],
        active_effort="medium", max_retries=3, drive_logs=ctx.drive_logs,
        task_id=ctx.task_id, round_idx=ctx.round_idx, event_queue=events,
        accumulated_usage=ctx.accumulated_usage, task_type="task",
        emit_progress=lambda *_a, **_kw: None, context_fit_plan=ctx.context_fit_plan,
        active_context_mode="max")
    assert message and active_model == MODEL and asked == [1] and api_calls == ["openai"]
    assert observed[-1] == "account-b"
    assert [upload[0]["account"] for upload in gateway.uploads] == [
        {"mode": "pin", "profileId": "account-a"},
        {"mode": "pin", "profileId": "account-b"}]
    dispatched = dispatched_attempts(ctx.drive_root)
    assert dispatched[-1]["physical_context"]["route_fp"] == "capacity-account-b"
    # The attempt's frame: the line its request is admitted at (the stated total less the reply floor) plus the reserve.
    assert dispatched[-1]["physical_context"]["capacity_total_tokens"] == 240_000 - 8_192 + 65_536
    assert plan.route_fp == "capacity-account-b" and plan.window_tokens == 240_000
    assert len(gateway.accepted_operations) == 2


# -- Presence: never waits, typed temporary refusal, no speech ------------------------------------


@pytest.fixture
def presence(main_call, monkeypatch):
    ctx, gateway, owner, events, _decide, _observations = main_call
    owner.task["_presence_turn"] = True

    def never(*_args, **_kwargs):
        pytest.fail("a Presence turn must not wait for quota or the owner")

    monkeypatch.setattr(model_wait, "time", SimpleNamespace(monotonic=time.monotonic, time=time.time, sleep=never))
    monkeypatch.setattr(loop_llm_call, "_sleep_within_deadline", never)
    monkeypatch.setattr(ctx.llm, "claudexor_model_catalog", never)
    return main_call


@pytest.mark.parametrize("configured_fallback", [True, False])
def test_presence_quota_turn_never_waits_and_ends_in_a_typed_temporary_refusal_without_speech(
        presence, monkeypatch, configured_fallback):
    ctx, gateway, owner, events, _decide, _observations = presence
    if configured_fallback:
        monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", FALLBACK)
        monkeypatch.setattr(fallback_cooldown, "is_cooling_down", lambda *_args: False)
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    api_calls = _api_fallback(ctx, monkeypatch, answer=False)
    gateway.results, gateway.dispatch = [_pool_quota()], ["not_started"]
    text, usage, _trace = _run(ctx, _loop_tools(ctx, owner), events)
    assert not _waits(events) and len(gateway.accepted_operations) == 1
    assert api_calls == (["openai"] if configured_fallback else [])
    refusal = usage[RESOURCE_REFUSAL_KEY]
    assert {key: refusal[key] for key in ("reason", "role", "model", "fallbacks_tried", "owner_wait", "temporary")} == {
        "reason": "quota", "role": "main", "model": MODEL, "owner_wait": "not_allowed", "temporary": True,
        "fallbacks_tried": [FALLBACK] if configured_fallback else []}
    assert refusal["account_rotation"]["pool_exhausted"] is True and refusal["reset_at"] == RESET
    assert provider_no_call_source(usage, False) == ("resource_refusal_no_resend", True)
    assert usage["execution_status"] == "infra_failed"
    assert usage["reason_code"] == "resource_refusal_no_resend"
    # Host-authored terminal: the correspondent hears nothing.
    assert _presence_delivery("message", text, str(usage.get("terminal_origin") or "")) == ("silent", "")


def test_presence_sign_in_refusal_is_typed_without_an_owner_wait(presence, monkeypatch):
    ctx, gateway, _owner, events, _decide, _observations = presence
    refusal = _pool_quota()
    refusal["problem"] = {**refusal["problem"], "code": "auth_required", "context": {"poolCause": "auth"}}
    gateway.results, gateway.dispatch = [refusal], ["not_started"]
    assert loop._call_round_model(ctx)[0] is None
    assert not _waits(events) and ctx.accumulated_usage[RESOURCE_REFUSAL_KEY]["reason"] == "auth"


@pytest.mark.parametrize("role", ["light", "vision"])
def test_presence_helper_roles_such_as_safety_refuse_at_once_instead_of_waiting(presence, role):
    """Safety keeps its own fail-closed refusal; it only stops waiting inside a Presence turn."""
    ctx, gateway, _owner, events, _decide, _observations = presence
    gateway.results, gateway.dispatch = [_pool_quota()], ["not_started"]
    with pytest.raises(transport.ClaudexorModelNotDispatched):
        ctx.llm.chat([{"role": "user", "content": "check"}], MODEL, model_role=role)
    assert not _waits(events) and len(gateway.accepted_operations) == 1


# -- no timer, no same-request resend; the owner wait needs evidence, not the same refused account --


def test_main_retry_never_sleeps_to_a_reset_nor_resends_a_spent_window(tmp_path, monkeypatch):
    monkeypatch.setattr(loop_llm_call, "_sleep_within_deadline", lambda *_a, **_kw: pytest.fail("slept to a reset"))
    refusal = transport.ClaudexorModelError(_vendor_quota()["problem"], route=ROUTE)
    refusal.physical_attempt_capture = SimpleNamespace(state="settled", attempt_id="a-1", provider="claudexor",
                                                       route_is_loopback=False)
    calls = []

    class Client:
        def chat(self, **kwargs):
            calls.append(kwargs)
            raise refusal

    usage = {}
    message, _cost = call_llm_with_retry(Client(), [{"role": "user", "content": "go"}], MODEL, None, "medium", 3,
                                         tmp_path / "logs", "task-one", 1, None, usage, deadline_ts=None)
    assert message is None and len(calls) == 1
    assert usage["_last_llm_retry_same_request"] is False and usage["_last_llm_resource_refusal"] == "quota"


def test_dated_pool_classifier_retains_forecast_without_inventing_quota():
    error = transport.ClaudexorModelNotDispatched({"code": "credential_pool_exhausted",
        "message": "No usable route", "context": {"poolCause": "unavailable", "resetsAt": RESET}}, route=ROUTE)
    fact = loop_llm_call.classify_llm_exception(error)
    assert fact.kind == "provider_error" and fact.provider_code == "credential_pool_exhausted"
    assert fact.reset_at == RESET and fact.retry_same_request is False and fact.retry_after_sec is None
    quota = loop_llm_call.classify_llm_exception(transport.ClaudexorModelError(_vendor_quota()["problem"], route=ROUTE))
    assert quota.kind == "subscription_window_exhausted"  # a proved quota window retains its own meaning


@pytest.mark.parametrize("presence", [False, True])
def test_a_dated_unavailable_pool_never_sleeps_to_its_reset_anywhere(tmp_path, monkeypatch, presence):
    """#1376 overlap / #1409 case 1: a dated pool refusal is a typed resource refusal of its own
    (``unavailable``, never quota) that is neither resent nor slept on, where waiting is allowed and
    where it is not; the round's configured routes and the visible wait own what follows."""
    monkeypatch.setattr(loop_llm_call, "_sleep_within_deadline", lambda *_a, **_kw: pytest.fail("slept to a reset"))
    dated = transport.ClaudexorModelNotDispatched({"code": "credential_pool_exhausted", "message": "no account",
                                                   "context": {"poolCause": "unavailable", "resetsAt": RESET}})
    dated.physical_attempt_capture = SimpleNamespace(state="released", attempt_id="a-1", provider="claudexor",
                                                     route_is_loopback=False)
    calls = []

    class Client:
        def chat(self, **_kwargs):
            calls.append(1)
            raise dated

    usage = {}
    with model_wait.task_model_wait_scope(task={"id": "task-one", "_presence_turn": presence}, drive_root=tmp_path,
                                          event_queue=None, worker_slot_held=False):
        message, _cost = call_llm_with_retry(Client(), [{"role": "user", "content": "go"}], MODEL, None, "medium", 3,
                                             tmp_path / "logs", "task-one", 1, None, usage, deadline_ts=None)
    assert message is None and calls == [1]
    assert usage["_last_llm_resource_refusal"] == "unavailable" and usage["_last_llm_reset_at"] == RESET
    assert "_last_llm_retry_after_sec" not in usage
    assert model_wait.model_wait_reason(dated) == "unavailable"
    undated = transport.ClaudexorModelNotDispatched({"code": "credential_pool_exhausted", "message": "no account",
                                                     "context": {"poolCause": "unavailable"}})
    assert model_wait.model_wait_reason(undated) == ""  # an undated pool proves no resource fact


@pytest.mark.parametrize("code,context,outage", [
    ("credential_pool_exhausted", {"poolCause": "mixed"}, False),
    ("model_account_unavailable", {}, True),
])
def test_an_engine_resource_verdict_is_not_a_network_outage(code, context, outage):
    refusal = transport.ClaudexorModelNotDispatched({"code": code, "message": code, "context": context})
    refusal.physical_attempt_capture = SimpleNamespace(state="released", provider="claudexor", route_is_loopback=False)
    assert (loop_llm_call.classify_llm_exception(refusal).kind == "transport_unavailable") is outage


def test_an_owner_wait_without_reset_evidence_does_not_resend_to_the_same_refused_account(live_wait, monkeypatch):
    root, gateway, client, _owner, events, _decide = live_wait
    gateway.results = [_vendor_quota(reset=False), _vendor_quota(reset=False), result(route=ROUTE_B)]
    gateway.dispatch = ["response_received"] * 3
    clock = SimpleNamespace(now=time.monotonic())
    monkeypatch.setattr(model_wait, "time", SimpleNamespace(
        monotonic=lambda: clock.now, time=time.time, sleep=lambda seconds: setattr(clock, "now", clock.now + seconds)))
    chosen = iter(["account-a", "account-b"])
    seen = []

    def catalog(*_args, **_kwargs):
        seen.append((next(chosen), len(gateway.accepted_operations)))
        return {"source": "codex", "credentialProfileId": seen[-1][0], "models": [{"id": "exact-model"}]}

    monkeypatch.setattr(client, "claudexor_model_catalog", catalog)
    answer, usage = client.chat(_history(), MODEL, model_role="main", cache_affinity="affinity")
    assert usage["claudexor"]["route"]["credentialProfileId"] == "account-b"
    # The engine's choice of the refused account, with no cooldown behind it, resent nothing.
    assert seen == [("account-a", 2), ("account-b", 2)] and len(gateway.accepted_operations) == 3
    waits = _waits(events)
    assert waits[0]["account_rotation"]["stop"] == "engine_reselected_refused_account"
    assert waits[0]["account_rotation"]["pool_exhausted"] is False
    assert waits[-1]["resolution"] == "resource_available"


def test_the_terminal_hint_claims_every_account_only_from_the_pool_verdict():
    base = {"_last_llm_error_kind": "subscription_window_exhausted", "_last_llm_reset_at": RESET}
    proven = provider_recovery_hint({**base, RESOURCE_REFUSAL_KEY: {
        "account_rotation": {"stop": "pool_exhausted", "pool_exhausted": True}, "fallbacks_tried": [FALLBACK]}})
    unproven = provider_recovery_hint({**base, RESOURCE_REFUSAL_KEY: {
        "account_rotation": {"stop": "engine_reselected_refused_account", "pool_exhausted": False}}})
    assert "every compatible account blocked" in proven and FALLBACK in proven
    assert "every compatible account" not in unproven and "unproven" in unproven
    assert "scheduled" not in proven + unproven and "nothing sleeps" in unproven


def test_fallback_keeps_original_effort_instead_of_previous_sent_or_reported(main_call, one_fallback, monkeypatch):
    ctx, _gateway, _owner, events, _decide, _observations = main_call
    ctx.accumulated_usage["request_wire"] = {"original_requested_effort": "ultra", "applied_effort": "low"}
    seen = []

    def candidate(round_call):
        seen.append(round_call.active_effort)
        return {"role": "assistant", "content": "done"}, 0.0, "max"

    monkeypatch.setattr(loop, "_call_round_model", candidate)
    message, *_rest = loop._run_cross_model_fallback_chain(
        llm=ctx.llm, ctx=ctx.tools._ctx, tools=ctx.tools, messages=ctx.messages, active_model=ctx.active_model,
        active_use_local=False, tool_schemas=[], active_effort="ultra", max_retries=1, drive_logs=ctx.drive_logs,
        task_id=ctx.task_id, round_idx=1, event_queue=events, accumulated_usage=ctx.accumulated_usage,
        task_type="task", emit_progress=lambda *_a, **_kw: None, context_fit_plan=ctx.context_fit_plan,
        active_context_mode="max")
    assert message["content"] == "done" and seen == ["ultra"]
