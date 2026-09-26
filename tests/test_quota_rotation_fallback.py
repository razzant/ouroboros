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

from ouroboros import fallback_cooldown, loop, loop_llm_call, model_wait
from ouroboros import llm_claudexor as transport
from ouroboros import usage_accounting as ua
from ouroboros.loop_llm_call import RETRY_WALL_EXHAUSTED_KEY, call_llm_with_retry, provider_no_call_source
from ouroboros.loop_model_call import RESOURCE_REFUSAL_KEY
from ouroboros.loop_transport import provider_recovery_hint
from ouroboros.model_slots import MODEL_ACCOUNTS_KEY
from ouroboros.presence_runner import _presence_delivery
from tests.test_llm_claudexor import MODEL, ROUTE, ledger, result
from tests.test_llm_claudexor import setup as gateway_fixture
from tests.test_model_wait import live_wait as wait_fixture
from tests.test_model_wait_controls import _loop_tools
from tests.test_subscription_main_wait import main_call as main_call_fixture

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


def test_an_unknown_fallback_outcome_asks_no_owner_and_sends_nothing_more(main_call, one_fallback, monkeypatch):
    ctx, gateway, owner, _events, _decide, _observations = main_call
    gateway.results, gateway.dispatch = [_pool_quota()], ["not_started"]
    assert loop._call_round_model(ctx)[0] is None and ctx.tools._ctx._deferred_resource_refusal is not None
    original = loop._call_round_model

    def candidate(round_call):
        assert round_call.defer_resource_wait is True  # the primary's owner question would follow
        round_call.accumulated_usage["_last_llm_error_kind"] = "provider_outcome_unknown"
        return None, 0.0, "max"

    monkeypatch.setattr(loop, "_call_round_model", candidate)
    message, *_rest = loop._run_cross_model_fallback_chain(
        llm=ctx.llm, ctx=ctx.tools._ctx, tools=ctx.tools, messages=ctx.messages, active_model=ctx.active_model,
        active_use_local=False, tool_schemas=[], active_effort="medium", max_retries=3, drive_logs=ctx.drive_logs,
        task_id=ctx.task_id, round_idx=1, event_queue=_events, accumulated_usage=ctx.accumulated_usage,
        task_type="task", emit_progress=lambda *_a, **_kw: None, context_fit_plan=ctx.context_fit_plan,
        active_context_mode="max")
    monkeypatch.setattr(loop, "_call_round_model", original)
    assert message is None and not _waits(_events) and len(gateway.accepted_operations) == 1
    refusal = ctx.accumulated_usage[RESOURCE_REFUSAL_KEY]
    assert (refusal["owner_wait"], refusal["fallbacks_tried"]) == ("not_asked", [FALLBACK])
    # The unknown-outcome fence still outranks: nothing is resent, not even a forced final.
    assert provider_no_call_source(ctx.accumulated_usage, False)[0] == "provider_outcome_unknown_no_resend"
    assert provider_no_call_source({RESOURCE_REFUSAL_KEY: refusal}, True) == ("resource_refusal_no_resend", True)


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
                ask_owner=lambda _waiter: (asked.append(_waiter), waiter.overrides.update(
                    {"fallback:0": {"model_account_override": "account-b"}})),
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
    rows = ledger(ctx.drive_root)
    dispatched = [row for row in rows if row["state"] == "dispatched"]
    assert [row["physical_context"]["route_fp"] for row in dispatched] == [
        "capacity-account-a", "capacity-account-b"]
    assert dispatched[-1]["physical_context"]["capacity_total_tokens"] == 240_000
    assert plan.route_fp == "capacity-account-b" and plan.window_tokens == 240_000
    assert ctx.messages[-2:] == original[-2:]  # completed tools are not replayed
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
    dispatched = [row for row in ledger(ctx.drive_root) if row["state"] == "dispatched"]
    assert dispatched[-1]["physical_context"]["route_fp"] == "capacity-account-b"
    assert dispatched[-1]["physical_context"]["capacity_total_tokens"] == 240_000
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
    assert message is None and len(calls) == 1 and usage[RETRY_WALL_EXHAUSTED_KEY] is True


@pytest.mark.parametrize("presence", [False, True])
def test_an_unproven_dated_pool_keeps_its_reset_timer_only_where_waiting_is_allowed(tmp_path, monkeypatch, presence):
    slept = []
    monkeypatch.setattr(loop_llm_call, "_sleep_within_deadline", lambda seconds, *_a, **_kw: slept.append(seconds) and False)
    dated = transport.ClaudexorModelNotDispatched({"code": "credential_pool_exhausted", "message": "no account",
                                                   "context": {"poolCause": "unavailable", "resetsAt": RESET}})
    dated.physical_attempt_capture = SimpleNamespace(state="released", attempt_id="a-1", provider="claudexor",
                                                     route_is_loopback=False)

    class Client:
        def chat(self, **_kwargs):
            raise dated

    with model_wait.task_model_wait_scope(task={"id": "task-one", "_presence_turn": presence}, drive_root=tmp_path,
                                          event_queue=None, worker_slot_held=False):
        call_llm_with_retry(Client(), [{"role": "user", "content": "go"}], MODEL, None, "medium", 3,
                            tmp_path / "logs", "task-one", 1, None, {}, deadline_ts=None)
    assert (slept == []) is presence and all(seconds > 3600 for seconds in slept)


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
