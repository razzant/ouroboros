"""Host review sampling hints remain distinct from explicit generation options.

The strict fake mirrors harness-codex/src/responses.ts::validateCodexModelOptions:
any present temperature/maxOutputTokens is refused before provider dispatch.
This contract is also checked against the real C validator in sprint evidence;
the permanent O suite needs neither a C checkout nor provider credentials.
"""

import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest

from ouroboros import capability_evidence as ce
from ouroboros.llm import LLMClient
from ouroboros.llm_claudexor import ClaudexorModelError
from ouroboros.review_records import ReviewRequest, ReviewSlot
from tests.test_llm_claudexor import MODEL, ledger, result, setup as gateway_fixture
from tests.test_model_wait import live_wait as wait_fixture

setup = gateway_fixture
live_wait = wait_fixture


def _strict(gateway, monkeypatch):
    create = gateway.create_model_operation

    def checked(ref, *, idempotency_key):
        options = gateway.uploads[-1][0]["options"]
        for option in ("temperature", "maxOutputTokens"):
            if option in options:
                failed = result(outcome="failed", problem={
                    "code": "unsupported_parameter", "message": "Explicit option unsupported",
                    "context": {"parameter": option}})
                failed["message"] = None
                gateway.results = [failed]
                gateway.dispatch = ["not_started"]
        return create(ref, idempotency_key=idempotency_key)

    monkeypatch.setattr(gateway, "create_model_operation", checked)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("explicit", [None, 0, 0.2])
def test_raw_default_hint_defers_but_explicit_temperature_stays_strict(setup, monkeypatch, asynchronous, explicit):
    root, gateway, client = setup
    _strict(gateway, monkeypatch)
    kwargs = dict(messages=[{"role": "user", "content": "Review"}], model=MODEL,
                  temperature=explicit, default_temperature=0.2)

    def call():
        return asyncio.run(client.chat_async(**kwargs)) if asynchronous else client.chat(**kwargs)

    if explicit is None:
        _message, usage = call()
        assert usage["prompt_tokens"] == 20
        assert "temperature" not in gateway.uploads[0][0]["options"]
        assert ledger(root)[-1]["state"] == "settled"
    else:
        with pytest.raises(ClaudexorModelError) as raised:
            call()
        assert raised.value.code == "unsupported_parameter"
        assert gateway.uploads[0][0]["options"]["temperature"] == explicit
        assert ledger(root)[-1]["state"] == "released"
    assert len(gateway.accepted_operations) == 1
    assert "default_temperature" not in gateway.uploads[0][0]["options"]
    assert kwargs["temperature"] is explicit and kwargs["default_temperature"] == 0.2


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("explicit,expected", [(None, 0.2), (0, 0), (0.7, 0.7)])
def test_api_physical_payload_keeps_host_default_and_explicit_precedence(asynchronous, explicit, expected):
    from tests.test_llm_provider_golden import _observe
    from tests.test_multiprovider_conformance import PROVIDER_DRIVERS

    driver = PROVIDER_DRIVERS["openai"]
    spec = driver.spec([driver.success_step], chat_kwargs={"temperature": explicit, "default_temperature": 0.2})
    spec["call"]["name"] = "chat_async" if asynchronous else "chat"
    observed = _observe(spec)
    assert "raised" not in observed
    assert observed["sends"][0]["payload"]["temperature"] == expected
    assert "default_temperature" not in observed["sends"][0]["payload"]
    assert observed["physical_attempts"][0]["state"] == "settled"


@pytest.mark.parametrize("asynchronous", [False, True])
def test_local_hint_preserves_the_previous_explicit_sampling_path(asynchronous):
    from tests.test_llm_provider_golden import _observe
    from tests.test_multiprovider_conformance import PROVIDER_DRIVERS

    driver = PROVIDER_DRIVERS["local"]
    observations = []
    for kwargs in ({"temperature": 0.2}, {"default_temperature": 0.2}):
        spec = driver.spec([driver.success_step], chat_kwargs=kwargs)
        spec["call"]["name"] = "chat_async" if asynchronous else "chat"
        observations.append(_observe(spec))
    assert all("raised" not in observed for observed in observations)
    assert observations[0]["sends"] == observations[1]["sends"]


@pytest.mark.parametrize("asynchronous", [False, True])
def test_wait_switch_restores_api_hint_and_later_override_defers_again(live_wait, monkeypatch, asynchronous):
    root, gateway, client, controller, _events, _decide = live_wait
    quota = result(outcome="failed", problem={"code": "subscription_window_exhausted", "message": "quota"})
    gateway.results = [quota, result()]
    gateway.dispatch = ["not_started", "response_received"]
    _strict(gateway, monkeypatch)
    api_calls, waited = [], []

    def api(_target, _messages, _tools, _effort, _max, _choice, temperature, *_args, **_kwargs):
        api_calls.append(temperature)
        return {"content": "API review"}, {"prompt_tokens": 3}

    # Async native Anthropic already shares the threaded synchronous transport seam.
    monkeypatch.setattr(client, "_chat_anthropic", api)

    def change(_client, _error, values, *_args):
        waited.append(dict(values))
        override = {"model": "anthropic::other-model", "use_local": False, "model_account_override": ""}
        controller.overrides["reviewer:critic"] = override
        return {**values, **override}

    monkeypatch.setattr(controller, "wait", change)
    kwargs = dict(messages=[{"role": "user", "content": "Same review"}], model=MODEL,
                  model_role="reviewer:critic", default_temperature=0.2)
    call = lambda: asyncio.run(client.chat_async(**kwargs)) if asynchronous else client.chat(**kwargs)
    _message, usage = call()
    assert api_calls == [0.2] and usage["model_role_route"]["model"] == "anthropic::other-model"
    assert waited[0]["temperature"] is None and waited[0]["default_temperature"] == 0.2
    controller.overrides["reviewer:critic"] = {"model": MODEL, "use_local": False, "model_account_override": "account-a"}
    # A reused caller originally naming API must not carry its resolved 0.2 as explicit.
    kwargs["model"] = "anthropic::other-model"
    _message, usage = call()
    assert usage["prompt_tokens"] == 20 and len(gateway.accepted_operations) == 2
    assert all("temperature" not in payload["options"] for payload, _key in gateway.uploads)
    assert [row["state"] for row in ledger(root)].count("released") == 1


@pytest.mark.parametrize("surface", ["triad", "two_part", "plan", "native_plan"])
def test_actual_review_authors_reach_strict_raw_dispatch(setup, monkeypatch, surface):
    from ouroboros import config
    from ouroboros.tools import plan_review_runtime
    from tests.review_pool_rosters import pool_roster, pool_seat, set_review_pool
    from ouroboros.tools.review_multi_model import _query_model
    from ouroboros.tools.registry import ToolContext

    root, gateway, client = setup
    _strict(gateway, monkeypatch)
    completed = result()
    completed["message"] = {"content": '[{"severity":"advisory","item":"x","evidence":"e","recommendation":"r"}]'}
    gateway.results = [completed]
    monkeypatch.setattr(config, "DATA_DIR", root)
    monkeypatch.setattr(ce, "resolve_review_token_density", lambda *_a, **_kw: (1.0, "measured"))
    monkeypatch.setattr(LLMClient, "claudexor_model_catalog", staticmethod(lambda source, profile=None, **_kw: {
        "source": source, "credentialProfileId": profile or "account-a", "accountFingerprint": "fingerprint-a",
        "observedAt": ce.utc_now_iso(), "provenance": "fixture",
        "models": [{"id": "exact-model", "contextWindow": 800_000, "maxOutputTokens": 4096}]}))
    ctx = ToolContext(repo_dir=root, drive_root=root, task_id="task-one")
    if surface == "triad":
        async def triad():
            return await _query_model(client, MODEL, [{"role": "user", "content": "Review"}], asyncio.Semaphore(1),
                                      ctx, slot_id="critic", session_profile="account-a", effort="high")
        _model, payload, _extra = asyncio.run(triad())
        assert "error" not in payload
    elif surface == "two_part":
        # The retrieving seat of the one wave: its own two-part brief, no packet.
        async def two_part():
            return await _query_model(client, MODEL, [], asyncio.Semaphore(1), ctx, slot_id="critic",
                                      session_profile="account-a", effort="high", native_retrieval=True,
                                      session_task="Review the staged change", session_root=str(root))
        _model, payload, _extra = asyncio.run(two_part())
        assert "error" not in payload
    else:
        set_review_pool(monkeypatch, pool_roster(pool_seat(
            "critic", MODEL, profile_id="account-a", delivery="native" if surface == "native_plan" else "packet")))
        slots = plan_review_runtime.plan_review_slots()
        assert slots[0].temperature is None and slots[0].default_temperature == 0.2
        rows = asyncio.run(plan_review_runtime.run_plan_review_slots(
            ctx, slots, system_prompt="Review", user_content="Plan", session_task="Review the repository.",
            session_root=str(root), output_contract="Return JSON findings"))
        assert rows[0]["text"] == completed["message"]["content"] and not rows[0]["error"]
    payload = gateway.uploads[0][0]
    assert "temperature" not in payload["options"]
    assert "default_temperature" not in payload["options"]
    assert "maxOutputTokens" not in payload["options"]
    assert payload["account"] == {"mode": "pin", "profileId": "account-a"}
    assert len(gateway.accepted_operations) == 1 and ledger(root)[-1]["state"] == "settled"


def test_review_custody_distinguishes_hint_from_explicit_without_changing_old_keys():
    from ouroboros.review_custody import _attempt_key

    request = ReviewRequest(surface="test", goal="Review")
    slot = ReviewSlot("critic", MODEL)
    old_request = SimpleNamespace(**{key: value for key, value in vars(request).items() if key != "default_temperature"})
    old_slot = SimpleNamespace(**{key: value for key, value in vars(slot).items() if key != "default_temperature"})
    assert _attempt_key(request, slot) == _attempt_key(old_request, old_slot)
    assert len({_attempt_key(request, candidate) for candidate in (
        slot, replace(slot, temperature=0.2), replace(slot, default_temperature=0.2))}) == 3


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("request_value,slot_value,expected", [(0, 0.6, 0), (None, 0, 0), (0.7, 0.6, 0.7)])
def test_explicit_review_temperature_beats_both_host_hints(setup, monkeypatch, native, request_value, slot_value, expected):
    from ouroboros.review_execution import ApiChatReviewExecutor, ReviewAssignment
    from ouroboros.review_native_episode import NativeToolRoundReviewExecutor

    root, gateway, client = setup
    _strict(gateway, monkeypatch)
    request = ReviewRequest(surface="test", goal="Review", temperature=request_value, default_temperature=0.2)
    slot = ReviewSlot("critic", MODEL, temperature=slot_value, default_temperature=0.3)
    executor_type = NativeToolRoundReviewExecutor if native else ApiChatReviewExecutor
    executor = executor_type(ReviewAssignment(request, slot, call_id="explicit"), llm=client)
    kwargs = executor._chat_kwargs([{"role": "user", "content": "Review"}], [], 32) if native else executor._kwargs()
    assert kwargs["temperature"] == expected and kwargs["default_temperature"] == 0.2
    with pytest.raises(ClaudexorModelError) as raised:
        client.chat(**kwargs)
    assert raised.value.code == "unsupported_parameter"
    assert gateway.uploads[0][0]["options"]["temperature"] == expected
    assert len(gateway.accepted_operations) == 1 and ledger(root)[-1]["state"] == "released"
