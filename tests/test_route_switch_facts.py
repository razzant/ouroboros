"""#1580: what a host-driven route switch tells the acting model, read from its next real input.

Real round loop, fallback chain, round dispatcher and physical-attempt ledger; only each
model's socket is scripted. After a switch the next input names both bindings, the
triggering failure as recorded before the answer cleared it, the answering call's own money
as its producer stated it and the primary's own last accepted call. Unknown stays unknown,
a known zero stays zero, a tariff comes only from the cache, and a later price never
rewrites what was sent.
"""

from __future__ import annotations

import json
import re

import pytest

from ouroboros import claudexor_daemon, llm_probe, pricing
from ouroboros import usage_accounting as ua
from ouroboros.gateways.claudexor import ClaudexorUnavailable
from tests import test_unknown_fallback_first as fallback_fixtures
from tests.test_completion_selection import finish
from tests.test_llm_claudexor import setup as gateway_fixture
from tests.test_model_wait import live_wait as wait_fixture
from tests.test_subscription_main_wait import main_call as main_call_fixture
from tests.test_unknown_fallback_first import _death, _run

data_root = fallback_fixtures.data_root
setup, live_wait, main_call = gateway_fixture, wait_fixture, main_call_fixture
PRIMARY = "claudexor::codex=gpt-test"
FALLBACK = "fb/one"
# The fallback's own answering call: provider-reported cash, a cold cache write.
SOL = {"prompt_tokens": 512_206, "cached_tokens": 0, "cache_write_tokens": 512_203, "completion_tokens": 900,
       "cost": 2.568527, "cost_final": True, "provider": "openrouter", "resolved_model": FALLBACK}
# A Codex result on engine 4.4: cash and billing unknown, a token valuation that is not cash,
# and an included processing mode that qualifies service, not this call's billing.
CODEX = {"prompt_tokens": 20, "cached_tokens": 12, "cache_write_tokens": None, "completion_tokens": 7,
         "cost": None, "cost_final": False, "provider": "claudexor", "resolved_model": PRIMARY,
         "cost_evidence": {"knowledge": "unknown", "billing": "unknown", "cashUsd": None, "estimatedUsd": None,
                           "valuationUsd": 12.5, "valuationKnowledge": "exact", "source": "provider",
                           "provenance": ["fixture"], "processing": {"kind": "included", "source": "fixture",
                                                                     "nativeMode": None}}}


def _tools(name: str = "chat_history") -> dict:
    return {"role": "assistant", "content": "", "tool_calls": [{"id": f"{name}-call", "type": "function",
            "function": {"name": name, "arguments": "{}"}}]}


class _Routes:
    """Each model's replies in order: ``(message, usage)`` or an exception factory; every send is ledgered."""

    def __init__(self, root, **scripts):
        self.root, self.sent = root, []
        self.scripts = {model: list(steps) for model, steps in scripts.items()}

    def default_model(self):
        return PRIMARY

    def chat(self, **kwargs):
        model = kwargs["model"]
        self.sent.append((model, [dict(row) for row in kwargs["messages"]]))
        step = self.scripts[model].pop(0)

        def send():
            if callable(step):
                error = step()
                raise RuntimeError("Connection error.") from error
            return step

        request = ua.AttemptRequest(model=model, provider="openrouter", reservation_usd=1.0, drive_root=self.root,
                                    task_id="t-1a", root_task_id="t-1a", source="test.route_switch_facts")
        ua.execute_physical_attempt(request, send, extractor=lambda reply: (
            {key: reply[1].get(key) for key in ("prompt_tokens", "completion_tokens")},
            reply[1].get("cost"), reply[1].get("cost_final") is True))
        return dict(step[0]), dict(step[1])


@pytest.fixture
def quiet_primary(monkeypatch):
    """No engine and no catalog answer for the primary; every price fetch is recorded, none is made."""
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", FALLBACK)
    monkeypatch.setattr(llm_probe, "upstream_transport_reachable", lambda *_a, **_kw: {})

    def no_engine(**_kwargs):
        raise ClaudexorUnavailable("daemon_not_discovered", "no owned engine in this fixture")

    monkeypatch.setattr(claudexor_daemon, "read_owned_gateway", no_engine)
    fetches = []
    monkeypatch.setattr(pricing, "_fetch_live_rows", lambda *args: fetches.append(args) or {})
    return fetches


CALL_ID = r"llm_[0-9a-f]{32}"


def _facts(messages, *, ids: bool = False) -> list:
    """Each sent ``[ROUTE FACTS]`` line; call ids are dropped unless asked for (they are random per run)."""
    rows = [match for row in messages for match in re.findall(r"\[ROUTE FACTS\][^\n]*", str(row.get("content")))]
    return rows if ids else [re.sub(rf"(?: |, ){CALL_ID}", "", row) for row in rows]


def _switch(tmp_path, root, *, answering=SOL, prior=CODEX, after=None):
    """Round 1 on the primary (unless ``prior`` is None), the primary dies, the fallback answers with a tool call."""
    primary = ([(_tools(), prior)] if prior is not None else []) + [_death]
    llm = _Routes(root, **{PRIMARY: primary,
                           FALLBACK: [(_tools(), answering)] + (after or [(finish("done"), dict(SOL))])})
    text, usage, trace, registry = _run(tmp_path, llm)
    return llm, text, usage, trace, registry


def test_the_next_real_input_names_both_bindings_the_failure_and_the_answering_calls_own_money(
        data_root, tmp_path, quiet_primary):
    llm, text, usage, _trace, _registry = _switch(tmp_path, data_root)
    assert text == "done" and [model for model, _ in llm.sent] == [PRIMARY, PRIMARY, FALLBACK, FALLBACK]
    assert _facts(llm.sent[2][1]) == []  # the answering candidate itself is never told
    [facts] = _facts(llm.sent[3][1])
    # Exact provenance: the named calls are the fallback's accepted round-2 answer and the primary's round-1 one.
    accepted = {(ref["model"], ref["round"]): ref["llm_call_id"] for ref in usage["llm_call_refs"]
                if "round" in ref and not ref.get("failure_code")}
    assert re.findall(CALL_ID, _facts(llm.sent[3][1], ids=True)[0]) == [accepted[(FALLBACK, 2)], accepted[(PRIMARY, 1)]]
    assert facts.startswith(f"[ROUTE FACTS] This task's primary route is {PRIMARY} (role main; account Auto) via "
                            f"claudexor; {FALLBACK} (role fallback:0) via openrouter answered this round after {PRIMARY} "
                            "failed with provider_outcome_unknown")
    # The triggering failure exactly as recorded before the fallback's answer cleared it.
    assert "That attempt's outcome and cost stay unknown." in facts
    assert 'Recorded error text: "' in facts and "Connection error." in facts
    # The answering call's OWN reported cash and cache split, not the task total or a reservation.
    assert ("That answering call: cost $2.5685 reported, billing basis not reported; "
            "tokens input 512,206, cache read 0, cache write 512,203; tariff unknown.") in facts
    # The primary's own last accepted call: unknown cash and unknown engine billing stay unknown.
    assert ("The primary's own last accepted call in this task (round 1): cost unknown, "
            "engine billing unknown; tariff unknown.") in facts
    for absent in ("$12.5", "included", "$0.00", "subscription", "metered", "free", "No catalog check"):
        assert absent not in facts, absent
    assert "No timer returns this task" in facts and 'switch_model(primary="wait")' in facts
    assert quiet_primary == []  # no catalog fetch for the facts, and no estimate was needed


def test_the_facts_never_borrow_the_task_total_or_an_earlier_primary_price(data_root, tmp_path, quiet_primary):
    """Distinguishing control: an API primary that cost $0.40 earlier makes the task total $2.97."""
    api_primary = {**SOL, "prompt_tokens": 100, "cached_tokens": 90, "cache_write_tokens": 0, "cost": 0.4,
                   "provider": "openrouter", "resolved_model": PRIMARY}
    llm, _text, usage, _trace, _registry = _switch(tmp_path, data_root, prior=api_primary)
    [facts] = _facts(llm.sent[3][1])
    assert "That answering call: cost $2.5685 reported" in facts
    assert "(round 1): cost $0.4000 reported, billing basis not reported" in facts
    assert "$2.9685" not in facts and "$1.0000" not in facts  # neither the total nor the unknown attempt's hold
    assert usage["cost"] == pytest.approx(2.968527 + 2.568527)  # the loop's own total kept counting


@pytest.mark.parametrize("answering,expected", [
    ({"cost": None, "cost_final": False, "provider": "claudexor",
      "cost_evidence": {"knowledge": "unknown", "billing": "subscription_entitlement", "cashUsd": None}},
     "That answering call: cost unknown, engine billing subscription_entitlement;"),
    ({"cost": 0.0, "cost_final": True, "provider": "claudexor",
      "cost_evidence": {"knowledge": "exact", "billing": "proven_zero", "cashUsd": 0.0}},
     "That answering call: cost $0.0000 reported, engine billing proven_zero;"),
    ({"cost": 0.0, "cost_final": True, "provider": "openrouter"},
     "That answering call: cost $0.0000 reported, billing basis not reported;"),
    ({"cost": 0.0312, "cost_final": False, "cost_estimated": True, "provider": "openrouter"},
     "That answering call: cost $0.0312 estimated, billing basis not reported;"),
    ({"cost": None, "cost_final": False, "provider": "claudexor",
      "cost_evidence": {"knowledge": "unknown", "billing": "unknown", "cashUsd": None, "valuationUsd": 3.0,
                        "valuationKnowledge": "exact", "processing": {"kind": "paid_credits", "source": "fixture"}}},
     "That answering call: cost unknown, engine billing unknown;"),
], ids=["subscription-entitlement-unknown-cash", "engine-proven-zero", "reported-zero", "estimated", "processing-not-billing"])
def test_each_money_state_reaches_the_input_as_its_producer_stated_it(data_root, tmp_path, quiet_primary,
                                                                     answering, expected):
    llm, *_ = _switch(tmp_path, data_root, answering={**SOL, **answering})
    [facts] = _facts(llm.sent[3][1])
    assert expected in facts
    assert "paid_credits" not in facts and "$3.0" not in facts  # a processing kind or a valuation is no charge
    if answering["cost"] is None:
        assert "That answering call: cost unknown" in facts and "$0.0000" not in facts.split("The primary")[0]


def test_a_switch_before_any_primary_answer_says_the_primary_was_not_observed(data_root, tmp_path, quiet_primary):
    llm, *_ = _switch(tmp_path, data_root, prior=None)
    assert [model for model, _ in llm.sent] == [PRIMARY, FALLBACK, FALLBACK]
    [facts] = _facts(llm.sent[2][1])
    assert "No accepted primary call is retained in this task's route-facts record; tariff unknown." in facts
    assert "(round" not in facts.split("No accepted primary call")[1]


def test_a_pinned_primary_and_a_local_fallback_keep_their_bindings_and_no_price(data_root, tmp_path, quiet_primary,
                                                                                monkeypatch):
    import json

    from ouroboros.model_slots import MODEL_ACCOUNTS_KEY

    monkeypatch.setenv(MODEL_ACCOUNTS_KEY, json.dumps({"main": "acct-a"}))
    monkeypatch.setenv("USE_LOCAL_FALLBACK", "true")
    llm, *_ = _switch(tmp_path, data_root, answering={**SOL, "cost": None, "cost_final": False, "provider": "openrouter"})
    [facts] = _facts(llm.sent[3][1])
    assert (f"primary route is {PRIMARY} (role main; account acct-a) via claudexor; {FALLBACK} (local) "
            "(role fallback:0) answered this round") in facts
    answer = facts.split("That answering call")[1].split("The primary's")[0]
    assert answer == ": local route, no provider charge; tokens input 512,206, cache read 0, cache write 512,203. "
    assert "(round 1): cost unknown, engine billing unknown; tariff unknown." in facts  # the pinned primary's own call


def test_a_cached_tier_tariff_is_quoted_without_http_and_a_later_price_never_rewrites_the_sent_facts(
        data_root, tmp_path, quiet_primary, monkeypatch):
    import time as clock

    key = ("openrouter", FALLBACK)
    tiered = pricing.PricingSchedule((1.0, 0.1, 1.25, 5.0), tiers=((200_000, (2.0, 0.2, 2.5, 10.0)),),
                                     source={"service_tier": "default"})
    monkeypatch.setitem(pricing._cached_pricing, key, {"endpoint-a": tiered})
    monkeypatch.setitem(pricing._pricing_fetched_at, key, clock.time())
    llm = _Routes(data_root, **{PRIMARY: [(_tools(), CODEX), _death],
                                FALLBACK: [(_tools(), SOL), (_tools(), dict(SOL)), (finish("done"), dict(SOL))]})
    original = llm.chat

    def chat(**kwargs):
        reply = original(**kwargs)
        if len(llm.sent) == 4:  # the fallback's first own round just carried the facts; then the price moves
            monkeypatch.setitem(pricing._cached_pricing, key, {"endpoint-a": pricing.PricingSchedule(
                (9.0, 0.9, 9.0, 90.0), source={"service_tier": "default"})})
        return reply

    llm.chat = chat
    text, *_ = _run(tmp_path, llm)
    assert text == "done" and [model for model, _ in llm.sent] == [PRIMARY, PRIMARY] + [FALLBACK] * 3
    first = _facts(llm.sent[3][1])
    assert len(first) == 1
    assert ("tokens input 512,206, cache read 0, cache write 512,203; cached catalog tariff $2.00/M in, "
            "$10.00/M out, cache read $0.20/M (estimated input tier).") in first[0]  # the input's own tier
    # The next own round carries the same bytes once; the later tariff never reached them.
    assert _facts(llm.sent[4][1]) == first and "$9.00" not in first[0]
    assert quiet_primary == []  # the facts read the cache only; nothing fetched a price


def test_the_answer_after_an_owner_wait_gets_the_same_facts_from_its_own_recorded_call(main_call, monkeypatch):
    """The second producer: an earlier fallback's quota refusal waits for the owner after the later route
    fails, then its retained send answers; the facts read that send's own call and the round's first failure."""
    from types import SimpleNamespace

    from ouroboros import fallback_cooldown, loop
    from ouroboros.primary_route_observation import PRIMARY_CALL_KEY
    from tests.test_model_wait_controls import _loop_tools
    from tests.test_quota_rotation_fallback import RESET

    ctx, _gateway, owner, _events, _decide, _observations = main_call
    tools = _loop_tools(ctx, owner)
    tools._ctx.primary_route = {"model": ctx.active_model, "use_local": False, "role": "main"}
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", "openai::one,openai::two")
    monkeypatch.setattr(fallback_cooldown, "is_cooling_down", lambda *_args: False)
    monkeypatch.setattr(loop, "_rebind_context_fit_plan", lambda plan, *_a, **_kw: (plan, "max"))
    asked, seen = [], []

    def candidate(call):
        seen.append((call.active_model, call.defer_resource_wait))
        call.tools._ctx._deferred_resource_refusal = None  # the dispatcher's per-candidate slot reset
        if call.active_model == "openai::one" and call.defer_resource_wait:  # its refusal waits for the owner
            call.tools._ctx._deferred_resource_refusal = SimpleNamespace(
                fact={"reason": "quota", "reset_at": RESET}, ask_owner=lambda _waiter: asked.append("quota"),
                terminal=lambda **fields: {"reason": "quota", "reset_at": RESET, **fields})
            call.accumulated_usage.update(_last_llm_error_kind="quota_exhausted", _last_llm_error="fallback quota refusal")
            return None, 0.0, "max"
        if call.active_model == "openai::two":
            call.accumulated_usage.update(_last_llm_error_kind="bad_request", _last_llm_error="second route refused")
            return None, 0.0, "max"
        call.accumulated_usage["_last_round_cache_usage"] = {
            "round": 4, "model": "one", "provider": "openai", "billing": None, "prompt_tokens": 9_000,
            "cached_tokens": 8_000, "cache_write_tokens": 0, "cost": 0.0123, "cost_final": True,
            "reported_tokens": {"prompt_tokens": 9_000, "completion_tokens": 40, "cached_tokens": 8_000,
                                "cache_write_tokens": 0}}
        return {"role": "assistant", "content": "after the owner wait"}, 0.0123, "max"

    monkeypatch.setattr(loop, "_call_round_model", candidate)
    # The primary's failing attempt as its producer records it: its own row carries its complete failure.
    usage = {"_last_llm_error_kind": "bad_request", "_last_llm_error": "HTTP 400: the primary refused this request",
             "_last_llm_status_code": 400, "llm_call_refs": [{
                 "ts": "2026-10-21T01:00:00Z", "failure_code": "bad_request", "status_code": 400, "provider_code": None,
                 "resource_refusal": "", "error": "HTTP 400: the primary refused this request"}],
             PRIMARY_CALL_KEY: {"round": 3, "model": ctx.active_model, "provider": "claudexor", "billing": "unknown",
                                "cost": None, "cost_final": False, "binding": [ctx.active_model, False, ""]}}
    usage["_last_llm_failure"] = usage["llm_call_refs"][-1]  # the producer retains the row by identity
    message, model, *_ = loop._run_cross_model_fallback_chain(
        llm=ctx.llm, ctx=tools._ctx, tools=tools, messages=ctx.messages, active_model=ctx.active_model,
        active_use_local=False, tool_schemas=[], active_effort="medium", max_retries=1, drive_logs=ctx.drive_logs,
        task_id=ctx.task_id, round_idx=4, event_queue=None, accumulated_usage=usage, task_type="task",
        emit_progress=lambda *_a, **_kw: None, context_fit_plan=ctx.context_fit_plan, active_context_mode="max")
    assert message["content"] == "after the owner wait" and model == "openai::one" and asked == ["quota"]
    assert seen == [("openai::one", True), ("openai::two", True), ("openai::one", False)]
    facts = tools._ctx._route_facts_pending
    assert f"openai::one (role fallback:0) via openai answered this round after {ctx.active_model} failed with " \
           "bad_request (HTTP 400) at 2026-10-21T01:00:00Z." in facts
    # The round's triggering failure, not either candidate's later failure that replaced the last-error keys.
    assert 'Recorded error text: "HTTP 400: the primary refused this request"' in facts
    assert "fallback quota" not in facts and "second route" not in facts
    assert ("That answering call: cost $0.0123 reported, billing basis not reported; "
            "tokens input 9,000, cache read 8,000, cache write 0") in facts
    assert "own last accepted call in this task (round 3): cost unknown, engine billing unknown" in facts


# -- r2: each fact from its own record: the binding a call answered on, the attempt that failed --------

API_FALLBACK = "openai::alternate"


class _Refused(RuntimeError):
    """A direct API route's own HTTP 400: classified bad_request, never resent as the same request."""

    status_code = 400


@pytest.mark.parametrize("chosen,pin,own", [
    ("claudexor::codex=exact-model", "account-b", False),
    ("openai::second", "", False),
    ("claudexor::codex=exact-model", "account-a", True),
], ids=["owner-moves-the-pin", "owner-overrides-the-model", "owner-rechooses-the-same-binding"])
def test_a_wait_card_choice_never_lends_an_earlier_binding_its_primary_call(main_call, monkeypatch, chosen, pin, own):
    """The real dispatcher, owner wait and fallback chain. Account-a answers round 1. In round 2 its refusal
    waits for the owner after the fallback fails, and the wait card moves the primary role; that binding fails
    before answering, again in round 3, where the fallback answers. The facts name the binding a send applies
    now; the round-1 call stays the earlier binding's unless the owner chose that same binding again."""
    from copy import deepcopy

    from ouroboros import fallback_cooldown, loop
    from ouroboros.llm_attempt import _attempt_request, _candidate_before_dispatch
    from ouroboros.model_slots import MODEL_ACCOUNTS_KEY
    from ouroboros.primary_route_observation import PRIMARY_CALL_KEY
    from tests.test_llm_claudexor import MODEL, ROUTE, result
    from tests.test_model_wait_controls import _loop_tools
    from tests.test_subscription_main_wait import ROUTE_B, _failed

    ctx, gateway, owner, events, decide, _observations = main_call
    monkeypatch.setenv(MODEL_ACCOUNTS_KEY, json.dumps({"main": "account-a"}))
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", API_FALLBACK)
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.setattr(fallback_cooldown, "is_cooling_down", lambda *_args: False)
    tools = ctx.tools = _loop_tools(ctx, owner)
    tools._ctx.primary_route = {"model": MODEL, "use_local": False, "role": "main"}
    engine = chosen == MODEL  # the moved primary still sends through the engine
    refused = _failed("model_unavailable", ROUTE_B if pin == "account-b" else ROUTE)
    gateway.results = [result(), _failed("subscription_window_exhausted")] + ([refused, deepcopy(refused)] if engine else [])
    gateway.dispatch = ["response_received", "not_started"] + (["not_started"] * 2 if engine else [])
    decisions, api = [], []

    def catalog(*_args, **_kwargs):  # the owner answers the wait's first catalog check from the wait card
        if not decisions:
            wait = next(event for event in reversed(list(events.queue)) if event.get("type") == "task_model_wait")
            decisions.append(decide({"request_id": "move-primary", "decision_id": f"model_wait:task-one:{wait['wait_id']}",
                                     "revision": wait["revision"], "action": "switch", "model": chosen,
                                     "credential_profile_id": pin, "use_local": False, "persist_role": False}).status_code)
        return {}

    monkeypatch.setattr(ctx.llm, "claudexor_model_catalog", catalog)
    remote = ctx.llm._chat_remote

    def send(target, messages, schemas, *args, **kwargs):
        if target["provider"] == "claudexor":
            return remote(target, messages, schemas, *args, **kwargs)
        api.append(target["resolved_model"])
        if target["resolved_model"] != "alternate" or api.count("alternate") == 1:
            raise _Refused("HTTP 400: this route refused the request")
        body = {"messages": deepcopy(messages), "tools": schemas or [], "model": "alternate"}
        request = _attempt_request(target, body)
        usage = {"prompt_tokens": 900, "completion_tokens": 3, "cost": 0.0042, "cost_final": True,
                 "provider": "openai", "resolved_model": "alternate"}
        return ua.execute_physical_attempt(
            request, lambda: ({"role": "assistant", "content": "the fallback answered"}, usage),
            extractor=lambda value: (value[1], 0.0042, True), before_dispatch=_candidate_before_dispatch(body, request))

    monkeypatch.setattr(ctx.llm, "_chat_remote", send)

    def chain(round_idx):
        return loop._run_cross_model_fallback_chain(
            llm=ctx.llm, ctx=tools._ctx, tools=tools, messages=ctx.messages, active_model=ctx.active_model,
            active_use_local=ctx.active_use_local, tool_schemas=[], active_effort="medium", max_retries=1,
            drive_logs=ctx.drive_logs, task_id=ctx.task_id, round_idx=round_idx, event_queue=events,
            accumulated_usage=ctx.accumulated_usage, task_type="task", emit_progress=lambda *_a, **_kw: None,
            context_fit_plan=ctx.context_fit_plan, active_context_mode="max")

    assert loop._call_round_model(ctx)[0] is not None  # round 1: the pinned account-a answers
    ctx.round_idx = 2
    assert loop._call_round_model(ctx)[0] is None  # account-a's refusal is deferred to the configured route
    message, ctx.active_model, ctx.active_use_local, *_ = chain(2)
    assert message is None and decisions == [202]
    assert owner.overrides == {"main": {"model": chosen, "use_local": False, "model_account_override": pin}}
    ctx.round_idx = 3
    assert loop._call_round_model(ctx)[0] is None
    message, *_ = chain(3)
    assert message["content"] == "the fallback answered"
    assert api == (["alternate", "alternate"] if engine else ["alternate", "second", "second", "alternate"])
    facts = re.sub(rf"(?: |, ){CALL_ID}", "", tools._ctx._route_facts_pending)
    if own:
        assert "The primary's own last accepted call in this task (round 1): cost unknown, engine billing unknown; " \
               "tariff unknown." in facts
    else:  # the earlier binding's call is never the current binding's evidence
        assert "own last accepted call" not in facts
    if not engine:  # the overridden model is named, never the configured one beside the new account
        assert f"is {MODEL}" not in facts and "account Auto" not in facts
    earlier = f"{MODEL} (role main; account account-a) via claudexor"
    assert (("earlier binding" not in facts) if own else
            ("The one retained primary call does not establish an accepted call on its current binding; tariff unknown. "
             f"Its last accepted call was on the earlier binding {earlier} (round 1): cost unknown, "
             "engine billing unknown.") in facts)
    now = (f"{chosen} (role main; account {pin}) via claudexor" if engine else f"{chosen} (role main) via openai")
    assert facts.startswith(f"[ROUTE FACTS] This task's primary route is {now}; {API_FALLBACK} (role fallback:0) via "
                            "openai answered this round after ")
    assert "cost $0.0042 reported, billing basis not reported" in facts  # the fallback's own call
    assert ctx.accumulated_usage[PRIMARY_CALL_KEY]["binding"] == [MODEL, False, "account-a"]  # round 1's, as recorded


def test_a_primary_call_recorded_without_its_binding_stays_unattributed():
    """A record that names no binding (a legacy shape) is never relabelled as the current binding's."""
    from types import SimpleNamespace

    from ouroboros.primary_route_observation import PRIMARY_CALL_KEY, route_facts_text

    ctx = SimpleNamespace(primary_route={"model": PRIMARY, "use_local": False, "role": "main"})
    legacy = {"round": 1, "llm_call_id": "llm_legacy", "model": PRIMARY, "provider": "claudexor", "billing": "unknown",
              "cost": None, "cost_final": False}
    text = route_facts_text(ctx, model=FALLBACK, use_local=False, role="fallback:0", failed_model=PRIMARY,
                            failure={"failure_code": "bad_request"}, waiter=None, usage={PRIMARY_CALL_KEY: legacy})
    assert ("The one retained primary call does not establish an accepted call on its current binding; tariff unknown. "
            "Its last accepted call (round 1, llm_legacy) has no recorded binding: cost unknown, "
            "engine billing unknown.") in text
    assert "own last accepted call" not in text


def test_returning_to_binding_a_does_not_deny_its_earlier_accepted_call(main_call, monkeypatch):
    """Real round dispatch records A, then B, then an A refusal; only B's accepted call is retained."""
    from ouroboros import loop
    from ouroboros.model_slots import MODEL_ACCOUNTS_KEY
    from ouroboros.primary_route_observation import PRIMARY_CALL_KEY, route_facts_text
    from tests.test_llm_claudexor import MODEL, result
    from tests.test_subscription_main_wait import ROUTE_B, _failed

    ctx, gateway, owner, *_ = main_call
    monkeypatch.setenv(MODEL_ACCOUNTS_KEY, json.dumps({"main": "account-a"}))
    ctx.tools._ctx.primary_route = {"model": MODEL, "use_local": False, "role": "main"}
    gateway.results = [result(), result(route=ROUTE_B), _failed("model_unavailable")]
    gateway.dispatch = ["response_received", "response_received", "not_started"]
    assert loop._call_round_model(ctx)[0] is not None
    first = dict(ctx.accumulated_usage[PRIMARY_CALL_KEY])
    owner.overrides = {"main": {"model": MODEL, "use_local": False, "model_account_override": "account-b"}}
    ctx.round_idx = 2
    assert loop._call_round_model(ctx)[0] is not None
    last = dict(ctx.accumulated_usage[PRIMARY_CALL_KEY])
    owner.overrides["main"]["model_account_override"] = "account-a"
    ctx.round_idx = 3
    assert loop._call_round_model(ctx)[0] is None
    assert first["binding"] == [MODEL, False, "account-a"] and last["binding"] == [MODEL, False, "account-b"]
    assert ctx.accumulated_usage[PRIMARY_CALL_KEY] == last
    text = route_facts_text(ctx.tools._ctx, model=API_FALLBACK, use_local=False, role="fallback:0", failed_model=MODEL,
                            failure={"failure_code": "model_unavailable"}, waiter=owner, usage=ctx.accumulated_usage,
                            round_idx=3)
    assert f"primary route is {MODEL} (role main; account account-a)" in text
    assert "The one retained primary call does not establish an accepted call on its current binding" in text
    assert "No accepted call" not in text and "own last accepted call" not in text
    assert "earlier binding" in text and "account-b" in text and last["llm_call_id"] in text
    assert first["llm_call_id"] not in text  # no history reconstruction or invented per-binding coverage


@pytest.mark.parametrize("refusal", [False, True], ids=["same-kind-and-text", "fallback-http-400"])
def test_second_fallback_pass_keeps_the_actual_failures_identity(data_root, tmp_path, monkeypatch, refusal):
    """The next real input after primary -> failed fallback -> recovered fallback names one actual failure."""
    from itertools import count

    from ouroboros import loop_llm_call, loop_transport
    from tests.test_unknown_fallback_first import PRIMARY as primary, _RouteLLM

    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", FALLBACK)
    monkeypatch.setattr(loop_transport, "upstream_transport_reachable", lambda *_a, **_kw: {})
    monkeypatch.setattr(loop_transport, "interruptible_wait_sleep", lambda *_a: False)
    ticks = count()
    monkeypatch.setattr(loop_llm_call, "utc_now_iso", lambda: f"2026-10-21T00:00:00.{next(ticks):06d}Z")
    llm = _RouteLLM(data_root, **{primary: [_death], FALLBACK: [
        (lambda: _Refused("HTTP 400: fb/one refused this request")) if refusal else _death]})
    chat, replies = llm.chat, []

    def answer(**kwargs):
        _msg, usage = chat(**kwargs)
        replies.append(kwargs)
        return (_tools() if len(replies) == 1 else finish("done")), usage

    llm.chat = answer
    text, usage, *_ = _run(tmp_path, llm, direct=True)
    assert text == "done" and [model for model, _ in llm.sent] == [primary, FALLBACK, FALLBACK, FALLBACK]
    failed = [ref for ref in usage["llm_call_refs"] if ref.get("failure_code")]
    assert [ref["model"] for ref in failed] == [primary, FALLBACK]
    assert failed[0]["ts"] != failed[1]["ts"]
    [facts] = _facts(llm.sent[-1][1])
    own = failed[1]
    assert f"after {own['model']} failed with {own['failure_code']}" in facts
    assert f"at {own['ts']}." in facts and failed[0]["ts"] not in facts
    assert f"after {primary} failed" not in facts
    assert f"Recorded error text: {json.dumps(own['error'])}." in facts
    assert ("HTTP 400" in facts) is refusal
    assert ("That attempt's outcome and cost stay unknown." in facts) is (not refusal)
    assert (failed[0]["error"] == own["error"]) is (not refusal)


class _Http503(RuntimeError):
    status_code = 503


def _http_503():
    raise _Http503("HTTP 503: upstream overloaded")


def test_a_later_unknown_attempt_never_inherits_an_earlier_attempts_http_status(
        data_root, tmp_path, quiet_primary, monkeypatch):
    """One primary round sends twice: a 503, then a death whose outcome is unknown. The facts describe the
    attempt that triggered the switch from its own row; the 503 stays that earlier attempt's own fact."""
    from ouroboros import loop_llm_call

    monkeypatch.setattr(loop_llm_call, "_sleep_within_deadline", lambda *_a, **_kw: True)  # the 503's backoff
    llm = _Routes(data_root, **{PRIMARY: [(_tools(), CODEX), _http_503, _death],
                                FALLBACK: [(_tools(), SOL), (finish("done"), dict(SOL))]})
    text, usage, *_ = _run(tmp_path, llm)
    assert text == "done" and [model for model, _ in llm.sent] == [PRIMARY] * 3 + [FALLBACK] * 2
    [facts] = _facts(llm.sent[4][1])
    assert "503" not in facts and "overloaded" not in facts
    failed = [ref for ref in usage["llm_call_refs"] if ref.get("failure_code")]
    assert (f"after {PRIMARY} failed with provider_outcome_unknown (code RuntimeError) at {failed[1]['ts']}. "
            "That attempt's outcome and cost stay unknown.") in facts
    assert 'Recorded error text: "' in facts and "Connection error." in facts
    assert [(ref["failure_code"], ref["status_code"]) for ref in failed] == [  # each attempt keeps its own status
        ("provider_transient", 503), ("provider_outcome_unknown", None)]


@pytest.mark.parametrize("message,kind", [
    ("Refused (HTTP 400).", "bad_request"),
    ("prompt is too long: 250000 tokens > 200000 maximum", "context_overflow"),
    ("unknown", "provider_outcome_unknown"),
], ids=["after-another-kind", "after-the-same-kind", "after-unknown"])
def test_a_refusal_without_its_own_attempt_row_borrows_no_earlier_attempts_fields(main_call, monkeypatch, message, kind):
    """A pre-send refusal leaves no attempt row. Its facts keep its code and its own words; an earlier
    attempt's row (of another kind, or of the same kind) lends it no status, time or text."""
    from copy import deepcopy

    from ouroboros import fallback_cooldown, loop
    from ouroboros.llm_attempt import _attempt_request, _candidate_before_dispatch
    from ouroboros.loop_llm_call import call_llm_with_retry
    from tests.test_context_presend_fit import _large
    from tests.test_llm_claudexor import MODEL, result
    from tests.test_model_wait_controls import _loop_tools

    ctx, gateway, owner, *_ = main_call
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", API_FALLBACK)
    monkeypatch.setattr(fallback_cooldown, "is_cooling_down", lambda *_args: False)
    tools = ctx.tools = _loop_tools(ctx, owner)
    tools._ctx.primary_route = {"model": MODEL, "use_local": False, "role": "main"}
    unknown = kind == "provider_outcome_unknown"
    gateway.results = [result(outcome="unknown") if unknown else result(
        outcome="failed", problem={"code": "invalid_request", "message": message,
                                   "retryable": False, "context": {"httpStatus": 400}})]
    gateway.dispatch = ["unknown" if unknown else "response_received"]
    if kind == "context_overflow":  # equal kind AND words still cannot identify a different attempt
        monkeypatch.setattr("ouroboros.loop_llm_call._short_error_text",
                            lambda _text: "No result room in the final candidate; pre-send size recovery.")
    assert call_llm_with_retry(ctx.llm, ctx.messages, MODEL, [], "medium", 1, ctx.drive_logs, ctx.task_id, 1, None,
                               ctx.accumulated_usage, "task", model_role="main")[0] is None
    [earlier] = [ref for ref in ctx.accumulated_usage["llm_call_refs"] if ref.get("failure_code")]
    ctx.round_idx = 2
    _large(ctx, irreducible=True)
    assert loop._call_round_model(ctx)[0] is None and len(gateway.creates) == 1  # the refusal sent nothing
    assert ctx.accumulated_usage["llm_call_refs"][-1] == earlier  # and added no row of its own
    remote = ctx.llm._chat_remote

    def send(target, messages, schemas, *args, **kwargs):
        if target["provider"] == "claudexor":
            return remote(target, messages, schemas, *args, **kwargs)
        body = {"messages": deepcopy(messages), "tools": schemas or [], "model": "alternate"}
        request = _attempt_request(target, body)
        usage = {"prompt_tokens": 90, "completion_tokens": 3, "cost": 0.001, "cost_final": True,
                 "provider": "openai", "resolved_model": "alternate"}
        return ua.execute_physical_attempt(
            request, lambda: ({"role": "assistant", "content": "the fallback answered"}, usage),
            extractor=lambda value: (value[1], 0.001, True), before_dispatch=_candidate_before_dispatch(body, request))

    from types import SimpleNamespace

    from ouroboros import context

    measured = context._context_fit_route

    def roomy(task, **kwargs):  # the fallback's window holds the input the primary's cannot
        route, evidence = measured(task, **kwargs)
        return route, (SimpleNamespace(**{**vars(evidence), "window_tokens": 2_000_000})
                       if route["provider"] == "openai" else evidence)

    monkeypatch.setattr(context, "_context_fit_route", roomy)
    monkeypatch.setattr(ctx.llm, "_chat_remote", send)
    message_out, *_ = loop._run_cross_model_fallback_chain(
        llm=ctx.llm, ctx=tools._ctx, tools=tools, messages=ctx.messages, active_model=MODEL, active_use_local=False,
        tool_schemas=[], active_effort="medium", max_retries=1, drive_logs=ctx.drive_logs, task_id=ctx.task_id,
        round_idx=2, event_queue=None, accumulated_usage=ctx.accumulated_usage, task_type="task",
        emit_progress=lambda *_a, **_kw: None, context_fit_plan=ctx.context_fit_plan, active_context_mode="max")
    assert message_out["content"] == "the fallback answered"
    facts = tools._ctx._route_facts_pending
    assert (f"after {MODEL} failed with context_overflow at an unrecorded time. Recorded error text: "
            '"No result room in the final candidate; pre-send size recovery."') in facts
    assert "HTTP 400" not in facts and earlier["ts"] not in facts and (unknown or message not in facts)
    assert earlier["failure_code"] == kind and earlier["error"]  # its own row
    assert not earlier["status_code"] if unknown else earlier["status_code"] == 400


def test_an_empty_answer_names_its_own_attempt_and_no_earlier_status(data_root, tmp_path, quiet_primary, monkeypatch):
    """The empty-response path through the real loop: a 503, then empty answers until the primary's own
    budget is spent; the facts name the last empty attempt's own row, code, time and words, no status."""
    from ouroboros import loop_llm_call

    monkeypatch.setattr(loop_llm_call, "_sleep_within_deadline", lambda *_a, **_kw: True)
    empty = ({"role": "assistant", "content": ""}, {"prompt_tokens": 5, "completion_tokens": 0, "cost": 0.0,
                                                     "cost_final": True})
    llm = _Routes(data_root, **{PRIMARY: [(_tools(), CODEX), _http_503] + [empty] * 12,
                                FALLBACK: [(_tools(), SOL), (finish("done"), dict(SOL))]})
    text, usage, *_ = _run(tmp_path, llm)
    sent = [model for model, _ in llm.sent]
    assert text == "done" and sent[:3] == [PRIMARY] * 3 and sent[-2:] == [FALLBACK] * 2 and sent.count(FALLBACK) == 2
    failed = [ref for ref in usage["llm_call_refs"] if ref.get("failure_code")]
    assert failed[0]["status_code"] == 503 and len(failed) >= 2
    assert all("status_code" not in ref and ref["error"] for ref in failed[1:])  # the empty attempts' own rows
    [facts] = _facts(llm.sent[-1][1])
    assert (f"after {PRIMARY} failed with {failed[-1]['failure_code']} at {failed[-1]['ts']}. "
            f"Recorded error text: {json.dumps(failed[-1]['error'], ensure_ascii=False)}.") in facts
    assert "503" not in facts and "overloaded" not in facts
