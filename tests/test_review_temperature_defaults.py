"""Host review sampling hints remain distinct from explicit generation options.

The strict fake mirrors harness-codex/src/responses.ts::validateCodexModelOptions:
any present temperature/maxOutputTokens is refused before provider dispatch.
This contract is also checked against the real C validator in sprint evidence;
the permanent O suite needs neither a C checkout nor provider credentials.
"""

import asyncio
import contextlib
import copy
import json
import time
from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace

import pytest

from ouroboros import capability_evidence as ce
from ouroboros import config
from ouroboros import request_wire_contract as wire
from ouroboros.deadline_utils import utc_now
from ouroboros.llm import LLMClient
from ouroboros.llm_claudexor import ClaudexorModelError
from ouroboros.request_wire_contract import (
    HINT_WITHDRAWAL_REASON,
    infer_tool_dialect,
    payload_effort,
    validate_wire_action,
    wire_action_identity,
)
from ouroboros.request_wire_recovery import (
    current_wire_candidate,
    plan_next_wire_retry,
    prepare_wire_payload_for_send,
    request_wire_call_scope,
    resolve_temperature_hint,
)
from ouroboros.review_records import ReviewRequest, ReviewSlot
from tests.test_llm_claudexor import MODEL, ledger, result, setup as gateway_fixture
from tests.test_llm_provider_golden import _ROUTE_ENV_NAMES, _install_fakes, _Recorder
from tests.test_model_wait import live_wait as wait_fixture
from tests.test_multiprovider_conformance import _ANTHROPIC_SUCCESS, PROVIDER_DRIVERS, _openai_success
from tests._cases import chosen_rows

setup = gateway_fixture
live_wait = wait_fixture

OMITTED = "omitted"
KIMI_REFUSAL = "invalid temperature: only 1 is allowed for this model"
FIELD_REFUSAL = "temperature is not supported"
HINT_ACTION = {"kind": "drop_field", "fields": ["temperature"], "reason_code": HINT_WITHDRAWAL_REASON}
FIELD_ACTION = {"kind": "drop_field", "fields": ["temperature"], "reason_code": "provider_unsupported_field"}
ROUTES = ("openai-compatible", "openrouter", "anthropic")
TOOLS = [{"type": "function", "function": {"name": "probe", "parameters": {"type": "object", "properties": {}}}}]


class _Provider(_Recorder):
    """One route's rule behind the real SDK seams: a refusal it returns is sent as that lane's refusal."""

    def __init__(self, root):
        super().__init__([])
        self.root, self.sent = root, []
        self.rule = lambda _payload: None

    def record(self, transport, *, payload, **extra):
        refusal = self.rule(copy.deepcopy(payload))
        message, node = refusal if isinstance(refusal, tuple) else (refusal, {})
        self.script = [_refusal_step(transport, message, node) if refusal else (
            _ANTHROPIC_SUCCESS if transport == "requests.post" else _openai_success())]
        return super().record(transport, payload=payload, **extra)

    def temperatures(self):
        """What the wire carried since the last look: a list of values, ``OMITTED`` for no field."""
        seen = [send["payload"].get("temperature", OMITTED) for send in self.sends]
        self.sent.extend(self.sends)
        self.sends.clear()
        return seen


def _refusal_step(transport, message, node):
    status = node.get("status", 400)
    error = {"message": message, "type": "invalid_request_error",
             **{k: v for k, v in node.items() if k not in {"status", "body_error"}}}
    if node.get("body_error"):  # An HTTP 200 whose body is the provider's refusal.
        return {"kind": "response", "body": {"choices": [], "usage": {}, "error": {**error, "code": status}}}
    if transport == "requests.post":  # The native lane reads an HTTP response, the SDK lanes an exception.
        return {"status_code": status, "reason": "Bad Request", "json": {"type": "error", "error": error}}
    return {"kind": "error", "status_code": status, "message": message, "body": {"error": error}}


@pytest.fixture
def api(tmp_path, monkeypatch):
    """Real LLMClient sends over the transport fakes, all on one data root and so one store."""
    for name in _ROUTE_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("OUROBOROS_SETTINGS_PATH", str(tmp_path / "settings.json"))
    monkeypatch.setenv("TOTAL_BUDGET", "1000")
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "SETTINGS_PATH", tmp_path / "settings.json")
    monkeypatch.setattr(LLMClient, "_SUPPORTED_PARAMS_FETCHED", True)
    monkeypatch.setattr(LLMClient, "_CAPABILITIES_FETCH_OK", True)
    monkeypatch.setattr(LLMClient, "_SUPPORTED_PARAMS_CACHE", {})
    for driver in PROVIDER_DRIVERS.values():
        for key, value in driver.env.items():
            monkeypatch.setenv(key, value)
    provider = _Provider(tmp_path)
    with contextlib.ExitStack() as stack:
        _install_fakes(stack, provider, {})
        yield provider


def _caller(model, *, asynchronous=False, no_proxy=False):
    """One logical call through the requested transport path; each resolves its own route."""
    client = LLMClient()

    def call(**kwargs):
        kwargs = {"messages": [{"role": "user", "content": "Review"}], "model": model, "no_proxy": no_proxy,
                  "reasoning_effort": "high", "max_tokens": 64, **kwargs}
        return asyncio.run(client.chat_async(**kwargs)) if asynchronous else client.chat(**kwargs)

    return call


def _records(root):
    path = root / "state" / wire.REQUEST_WIRE_STATE_FILE
    if not path.exists():
        return []
    return [record for entry in json.loads(path.read_text())["profiles"].values() for record in entry["records"]]


def _only_one(payload):
    return KIMI_REFUSAL if payload.get("temperature", 1) != 1 else None


def _applied(usage):
    return [(item["source"], item["action"]) for item in usage["request_wire"]["applied_actions"]]


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


@pytest.mark.parametrize(("explicit", "asynchronous"), [(None, False), (0, True), (0.2, False)])
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


@pytest.mark.parametrize(("explicit", "expected", "asynchronous"), [
    (None, 0.2, False),
    (0, 0, True),
    (0.7, 0.7, False),
])
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
    api_calls, waited, origins = [], [], []

    def api(target, _messages, _tools, _effort, _max, _choice, temperature, *_args, **_kwargs):
        api_calls.append(temperature)
        origins.append(target.get("temperature_hint"))
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
    assert origins == [0.2], "the hint resolved after the wait stays withdrawable on its effective route"
    assert waited[0]["temperature"] is None and waited[0]["default_temperature"] == 0.2
    controller.overrides["reviewer:critic"] = {"model": MODEL, "use_local": False, "model_account_override": "account-a"}
    # A reused caller originally naming API must not carry its resolved 0.2 as explicit.
    kwargs["model"] = "anthropic::other-model"
    _message, usage = call()
    assert usage["prompt_tokens"] == 20 and len(gateway.accepted_operations) == 2
    assert all("temperature" not in payload["options"] for payload, _key in gateway.uploads)
    assert [row["state"] for row in ledger(root)].count("released") == 1


@pytest.mark.parametrize("surface", ["triad", "two_part", "native_plan"])
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


@pytest.mark.parametrize(("request_value", "slot_value", "expected", "native"), [
    (0, 0.6, 0, True),
    (None, 0, 0, False),
    (0.7, 0.6, 0.7, True),
])
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


@pytest.mark.parametrize("no_proxy", [False, True], ids=["cached", "no_proxy"])
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("route", chosen_rows(ROUTES, 0, 2))
def test_refused_hint_is_withdrawn_remembered_and_explicit_values_stay(api, route, asynchronous, no_proxy):
    api.rule = _only_one
    call = _caller(PROVIDER_DRIVERS[route].model, asynchronous=asynchronous, no_proxy=no_proxy)

    # A same-number explicit value on a fresh route is the provider's error: one send, nothing learned.
    with pytest.raises(Exception, match="only 1 is allowed"):
        call(temperature=0.2, default_temperature=0.2)
    assert api.temperatures() == [0.2] and _records(api.root) == []

    # The hint is withdrawn after its refusal, and only the settled success teaches.
    _message, usage = call(default_temperature=0.2)
    assert api.temperatures() == [0.2, OMITTED]
    assert _applied(usage) == [("pending", HINT_ACTION)]
    learned = _records(api.root)
    assert [record["action"] for record in learned] == [HINT_ACTION]

    # It replays on later hints without extending its age.
    _message, usage = call(default_temperature=0.2)
    assert api.temperatures() == [OMITTED]
    assert _applied(usage) == [("durable", HINT_ACTION)] and _records(api.root) == learned

    # A later valid explicit value and the same number given explicitly are never its business.
    _message, usage = call(temperature=1, default_temperature=0.2)
    assert api.temperatures() == [1] and _applied(usage) == []
    with pytest.raises(Exception, match="only 1 is allowed"):
        call(temperature=0.2)
    assert api.temperatures() == [0.2] and _records(api.root) == learned

    # Origin never reaches the provider, on any send of any path.
    for send in api.sent:
        assert not {"temperature_hint", "default_temperature"} & set(send["payload"])


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
def test_a_withdrawal_that_still_fails_teaches_nothing(api, asynchronous):
    api.rule = lambda _payload: KIMI_REFUSAL  # refused with and without the hint alike
    call = _caller(PROVIDER_DRIVERS["openai-compatible"].model, asynchronous=asynchronous)
    for _attempt in range(2):
        with pytest.raises(Exception, match="only 1 is allowed"):
            call(default_temperature=0.2)
        assert api.temperatures() == [0.2, OMITTED], "one withdrawn resend, then the provider's error"
    assert _records(api.root) == []


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
def test_a_body_refusal_reaches_the_same_withdrawal_and_teaches_only_after_success(api, asynchronous):
    call = _caller(PROVIDER_DRIVERS["openrouter"].model, asynchronous=asynchronous)
    api.rule = lambda payload: (KIMI_REFUSAL, {"body_error": True}) if "temperature" in payload else None
    _message, usage = call(default_temperature=0.2)
    assert api.temperatures() == [0.2, OMITTED]
    assert _applied(usage) == [("pending", HINT_ACTION)]
    assert [record["action"] for record in _records(api.root)] == [HINT_ACTION]


def test_withdrawal_memory_is_exact_route_and_expires(api):
    api.rule = _only_one
    call = _caller(PROVIDER_DRIVERS["openai-compatible"].model)
    call(default_temperature=0.2)
    assert api.temperatures() == [0.2, OMITTED]

    api.rule = lambda _payload: None  # every route now takes the hint, so only memory could remove it
    _caller("openai-compatible::model-y")(default_temperature=0.2)
    assert api.temperatures() == [0.2], "another model keeps its hint"
    _caller(PROVIDER_DRIVERS["openrouter"].model)(default_temperature=0.2)
    assert api.temperatures() == [0.2], "another provider keeps its hint"
    call(default_temperature=0.2, tools=TOOLS)
    assert api.temperatures() == [0.2], "another tool shape is another profile"

    # The owner's accepted cost: the same profile suppresses it in another mode where it might work.
    call(default_temperature=0.2, reasoning_effort="low")
    assert api.temperatures() == [OMITTED]

    def age(days):
        path = api.root / "state" / wire.REQUEST_WIRE_STATE_FILE
        store = json.loads(path.read_text())
        for entry in store["profiles"].values():
            for record in entry["records"]:
                record["observed_at"] = (utc_now() - timedelta(days=days)).isoformat()
        path.write_text(json.dumps(store))

    age(13.9)
    call(default_temperature=0.2)
    assert api.temperatures() == [OMITTED]
    age(14)
    call(default_temperature=0.2)
    assert api.temperatures() == [0.2], "the 14-day record is gone, and replay never renewed it"


def _plan(error, *, sent=0.2, hint=True, **fields):
    """One real preparation and retry decision for the sent temperature's origin."""
    target = {"provider": "openrouter", "resolved_model": "vendor/future", "usage_model": "openrouter/vendor/future",
              "base_url": "https://openrouter.example/v1"}
    temperature = resolve_temperature_hint(target, None if hint else sent, sent if hint else 0.2)
    payload = {"model": "vendor/future", "messages": [{"role": "user", "content": "x"}], "reasoning_effort": "high",
               "temperature": temperature, "max_tokens": 8, **fields}
    with request_wire_call_scope():
        first = prepare_wire_payload_for_send(target, payload, api_surface="chat.completions")
        retry = plan_next_wire_retry(first, error=error, target=target)
        return retry, [item.action["reason_code"] for item in current_wire_candidate().applied_actions]


class _Refused(RuntimeError):
    def __init__(self, message, status=400, **node):
        super().__init__(message)
        self.status_code = status
        self.body = {"error": {"message": message, "type": "invalid_request_error", **node}}


@pytest.mark.parametrize("error", [
    _Refused(KIMI_REFUSAL),
    _Refused("temperature can only be 1 when reasoning_effort is high"),  # its own statement, another field beside it
    _Refused("invalid value", status=422, param="temperature", allowed_values=["0", 1.0]),  # every member readable
], ids=lambda error: error.body["error"]["message"][:40])
def test_bounded_value_refusal_withdraws_a_hint_and_never_an_explicit_value(error):
    retry, reasons = _plan(error)
    assert retry is not None and "temperature" not in retry and reasons == [HINT_WITHDRAWAL_REASON]
    assert _plan(error, hint=False)[0] is None, "the same refusal of an explicit value is the provider's error"


@pytest.mark.parametrize("sent,error", [
    (1, _Refused(KIMI_REFUSAL)),  # already the value it allows
    (0.2, _Refused("temperature can only be [0, 1]")),
    # The negative enum clause returns early, and its structured list still counts.
    (0.2, _Refused("no allowed values match; temperature must be 1", allowed_values=[0.2, 1])),
], ids=lambda value: value.body["error"]["message"][:40] if isinstance(value, _Refused) else str(value))
def test_a_refusal_that_does_not_exclude_the_hint_withdraws_nothing(sent, error):
    assert _plan(error, sent=sent)[0] is None


@pytest.mark.parametrize("code", ["insufficient_quota", "invalid_auth",
                                  "context_length_exceeded"])
def test_a_non_compatibility_code_keeps_the_classifier_admission(code):
    assert _plan(_Refused(KIMI_REFUSAL, code=code))[0] is None


@pytest.mark.parametrize(("message", "hint"), [
    pytest.param("unsupported temperature: not only 1 is allowed", True, id='unsupported temperature: not only 1 is allowed-hint'),
    pytest.param("unsupported temperature: only 1 or less is allowed", False, id='unsupported temperature: only 1 or less is allowed-explicit'),
    pytest.param("unsupported temperature: must be at most 1", True, id='unsupported temperature: must be at most 1-hint'),
    pytest.param("unsupported temperature: must not be 1", False, id='unsupported temperature: must not be 1-explicit'),
    pytest.param("unsupported temperature: must not exceed 1", True, id='unsupported temperature: must not exceed 1-hint'),
    pytest.param("unsupported temperature: cannot exceed 1", False, id='unsupported temperature: cannot exceed 1-explicit'),
    pytest.param("unsupported temperature: only 1 is not allowed", True, id='unsupported temperature: only 1 is not allowed-hint'),
    pytest.param("temperature is unsupported by this provider. only 1 is allowed", False, id='temperature is unsupported by this provider. only 1 is allowed-explicit'),
    pytest.param("invalid parameter: temperature can only be [0, 1; 0.2]", True, id='invalid parameter: temperature can only be [0, 1; 0.2]-hint'),  # legacy markers classify this as a field refusal
])
def test_a_statement_of_values_it_cannot_use_keeps_an_explicit_value_and_withdraws_the_hint(message, hint):
    retry, reasons = _plan(_Refused(message), hint=hint)
    if hint:  # the field's own refusal still withdraws the optional hint
        assert retry is not None and "temperature" not in retry and reasons == [HINT_WITHDRAWAL_REASON]
    else:
        assert retry is None


@pytest.mark.parametrize(("error", "hint"), [
    pytest.param(_Refused("unsupported value: temperature must be one of [0, 1"), True, id='unclosed-list-hint'),
    pytest.param(_Refused("invalid parameter: temperature must be one of [0, 1; 0.2]"), False, id='semicolon-list-explicit'),
    pytest.param(_Refused("unsupported temperature: only 1 is allowed", allowed_values=[0, 1, "2e-1"]), True, id='typed-unreadable-member-hint'),
    pytest.param(_Refused("invalid parameter: temperature must be [0, 1; 0.2]", code="invalid_value"), False, id='value-code-explicit'),
    pytest.param(_Refused("temperature is not supported; must be at most 1", param="temperature", code="unsupported_value"), True, id='unsupported-value-code-hint'),
])
def test_an_unreadable_value_refusal_beside_capability_wording_repairs_nothing(error, hint):
    # Recognized value evidence prevents hint withdrawal; the typed/code rows each exercise a separate guard.
    assert _plan(error, hint=hint)[0] is None


@pytest.mark.parametrize(("refusal", "origin", "asynchronous"), [
    pytest.param("temperature rejected, only 3 attempts are allowed", "default_temperature", False, id='a-count-sync'),
    pytest.param("temperature is fine, but custom_field must be one of 0, 1", "default_temperature", True, id='read-parameter-async'),
    pytest.param((KIMI_REFUSAL, {"code": "insufficient_quota"}), "default_temperature", False, id='quota-code-sync'),
    pytest.param("unsupported temperature: only 1 or less is allowed", "temperature", True, id='explicit-bound-async'),
    pytest.param("invalid max_tokens: only 1 is allowed for temperature-less models", "default_temperature", False, id='a-compound-sync'),
    pytest.param("unsupported value: temperature must be one of [0, 1", "default_temperature", True, id='unreadable-value-list-async'),
    pytest.param(("unsupported temperature: only 1 is allowed", {"allowed_values": [0, 1, "2e-1"]}), "default_temperature", False, id='unreadable-typed-list-sync'),
    pytest.param(("invalid parameter: temperature must be [0, 1; 0.2]", {"code": "invalid_value"}), "default_temperature", True, id='unreadable-value-code-async'),
    pytest.param(("temperature is not supported; must be at most 1", {"param": "temperature", "code": "unsupported_value"}), "default_temperature", False, id='unsupported-value-code-sync'),
])
def test_a_refusal_that_authorizes_nothing_reaches_the_caller_unrepaired(api, refusal, origin, asynchronous):
    api.rule = lambda _payload: refusal
    with pytest.raises(Exception, match="attempts|custom_field|only 1|must be"):
        _caller(PROVIDER_DRIVERS["openai-compatible"].model, asynchronous=asynchronous)(**{origin: 0.2})
    assert api.temperatures() == [0.2] and _records(api.root) == []


@pytest.mark.parametrize(("values", "excludes_hint", "asynchronous"), [
    pytest.param("0, 1, or 0.2", False, False, id='0, 1, or 0.2-False-sync'),
    pytest.param("0, 1, and 0.2", False, True, id='0, 1, and 0.2-False-async'),
    pytest.param("[0, 1, 0.2]", False, False, id='[0, 1, 0.2]-False-sync'),
    pytest.param("0, 1, or 2", True, True, id='0, 1, or 2-True-async'),
    pytest.param("0, 1, and 2", True, False, id='0, 1, and 2-True-sync'),
    pytest.param("[0, 1, 2]", True, True, id='[0, 1, 2]-True-async'),
    pytest.param("0, 1, or exactly 0.2", False, False, id='0, 1, or exactly 0.2-False-sync'),  # incomplete lists authorize nothing
    pytest.param("0, 1, and", False, True, id='0, 1, and-False-async'),  # incomplete lists authorize nothing
    pytest.param("0, 1, or 2, got 0.2", True, False, id='0, 1, or 2, got 0.2-True-sync'),  # a known trailing clause is not another allowed value
    pytest.param("[0, 1], or 0.2", False, True, id='[0, 1], or 0.2-False-async'),
    pytest.param("0, 1 (or 0.2)", False, False, id='0, 1 (or 0.2)-False-sync'),
    pytest.param("(0, 1), or 0.2", False, True, id='(0, 1), or 0.2-False-async'),
    pytest.param("0, 1 for temperature, or 0.2", False, False, id='0, 1 for temperature, or 0.2-False-sync'),
    pytest.param("(0, 1)", True, True, id='(0, 1)-True-async'),
    pytest.param("(0, 0.2)", False, False, id='(0, 0.2)-False-sync'),
    pytest.param("[0, 1; 0.2]", False, True, id='[0, 1; 0.2]-False-async'),  # never closed
    pytest.param("[0, 1", False, False, id='[0, 1-False-sync'),  # never closed
    pytest.param("(0, 1; 0.2)", False, True, id='(0, 1; 0.2)-False-async'),  # never closed
    pytest.param("(0, 1", False, False, id='(0, 1-False-sync'),  # never closed
    pytest.param("(0, 1]", False, True, id='(0, 1]-False-async'),  # closed by the wrong bracket, or none
    pytest.param("[0, 1)", False, False, id='[0, 1)-False-sync'),  # closed by the wrong bracket, or none
    pytest.param("0, 1]", False, True, id='0, 1]-False-async'),  # closed by the wrong bracket, or none
    pytest.param("0, 1)", False, False, id='0, 1)-False-sync'),  # closed by the wrong bracket, or none
    pytest.param("[0, 1], [0.2]", False, True, id='[0, 1], [0.2]-False-async'),
    pytest.param("[0, 1] or [0.2]", False, False, id='[0, 1] or [0.2]-False-sync'),
    pytest.param("0, 1; 0.2", False, True, id='0, 1; 0.2-False-async'),  # the list runs on
    pytest.param("0, 1. 0.2", False, False, id='0, 1. 0.2-False-sync'),  # the list runs on
    pytest.param("0, 1\n0.2", False, True, id='0, 1\n0.2-False-async'),  # the list runs on
    pytest.param("0, 1...", False, False, id='0, 1...-False-sync'),  # the list runs on
    pytest.param("[0, 1]; or 0.2", False, True, id='[0, 1]; or 0.2-False-async'),
    pytest.param("0, 1. and 0.2", False, False, id='0, 1. and 0.2-False-sync'),
    pytest.param("0, 1. . .", False, True, id='0, 1. . .-False-async'),
    pytest.param("0, 1 . . . 0.2", False, False, id='0, 1 . . . 0.2-False-sync'),
    pytest.param("[ 0, 1 ]", True, True, id='[ 0, 1 ]-True-async'),
    pytest.param("( 0 , 1 )", True, False, id='( 0 , 1 )-True-sync'),
    pytest.param("[0, 1].", True, True, id='[0, 1].-True-async'),
    pytest.param("[0, 1]; top_p must be 1", True, False, id='[0, 1]; top_p must be 1-True-sync'),
    pytest.param("[(0, 1)]", False, True, id='[(0, 1)]-False-async'),  # beyond the grammar's one bracket: unread, so nothing is withdrawn
    pytest.param("((0, 1))", False, False, id='((0, 1))-False-sync'),  # beyond the grammar's one bracket: unread, so nothing is withdrawn
])
def test_the_complete_allowed_list_decides_withdrawal_and_learning(api, values, excludes_hint, asynchronous):
    refusal = "temperature must be one of " + values
    api.rule = lambda payload: refusal if "temperature" in payload else None
    call = _caller(PROVIDER_DRIVERS["openai-compatible"].model, asynchronous=asynchronous)
    if excludes_hint:
        _message, usage = call(default_temperature=0.2)
        assert api.temperatures() == [0.2, OMITTED]
        assert _applied(usage) == [("pending", HINT_ACTION)]
        assert [record["action"] for record in _records(api.root)] == [HINT_ACTION]
    else:
        with pytest.raises(Exception, match="temperature must be one of"):
            call(default_temperature=0.2)
        assert api.temperatures() == [0.2] and _records(api.root) == []
    learned = _records(api.root)
    with pytest.raises(Exception, match="temperature must be one of"):
        call(temperature=0.2)
    assert api.temperatures() == [0.2] and _records(api.root) == learned


UNREADABLE_LISTS = {
    "exponent": [0, 1, "2e-1"], "bool": [0, 1, True], "null": [0, 1, None], "word": [0, 1, "x"], "nested": [0, 1, [0.2]],
    "negative": [0, 1, -1], "small-float": [0, 1, 1e-05], "not-a-list": "0, 1",
}


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("node", [("invalid value", {"param": "temperature"}), (KIMI_REFUSAL, {})], ids=["typed", "prose"])
@pytest.mark.parametrize("values", chosen_rows(UNREADABLE_LISTS.values(), 0, 7), ids=chosen_rows(UNREADABLE_LISTS.keys(), 0, 7))
def test_a_structured_list_it_cannot_read_whole_withdraws_nothing(api, node, values, asynchronous):
    """One unreadable member may be the value that allows the hint: the provider's refusal stands, nothing is learned."""
    message, fields = node
    api.rule = lambda payload: (message, {**fields, "allowed_values": values}) if "temperature" in payload else None
    call = _caller(PROVIDER_DRIVERS["openai-compatible"].model, asynchronous=asynchronous)
    for origin in ("default_temperature", "temperature"):
        with pytest.raises(Exception, match=message):
            call(**{origin: 0.2})
        assert api.temperatures() == [0.2] and _records(api.root) == []


@pytest.mark.parametrize("values", [[0, 1, 10 ** 400], [0, 1, 10 ** 5000], [float("inf")]],
                         ids=["huge-int", "unprintable-int", "inf"])
def test_an_extreme_structured_member_cannot_break_the_refusal(values):
    # A member this extreme must not raise in place of the provider's refusal.
    assert _plan(_Refused("invalid value", param="temperature", allowed_values=values), hint=False)[0] is None
    assert _plan(_Refused(KIMI_REFUSAL, allowed_values=[0.2, *values]))[0] is None


@pytest.mark.parametrize("asynchronous,field,option", [
    (False, "response_format", {"response_format": {"type": "json_object"}}),
    (False, "prompt_cache_key", {"cache_affinity": "seat"}),
    (True, "prompt_cache_key", {"cache_affinity": "seat"}),
], ids=["sync-response-format", "sync-cache-key", "async-cache-key"])
def test_a_send_left_unregistered_still_withdraws_its_hint_and_teaches_nothing(api, asynchronous, field, option):
    def rule(payload):
        if payload_effort(payload):
            return "model does not support thinking"
        if field in payload:
            return f"{field} is not supported"
        return "temperature can only be 1" if "temperature" in payload else None

    api.rule = rule
    call = _caller(PROVIDER_DRIVERS["openai-compatible"].model, asynchronous=asynchronous)
    call(default_temperature=0.2, **option)
    assert _sends(api) == [("none", 0.2, "high"), ("none", 0.2, ""), ("none", 0.2, ""), ("none", OMITTED, "")]
    assert _records(api.root) == [], "no candidate was bound to that success"

    # An explicit value carries no mark: the same three refusals are the provider's verdict.
    with pytest.raises(Exception, match="temperature can only be 1"):
        call(temperature=0.2, **option)
    assert _sends(api) == [("none", 0.2, "high"), ("none", 0.2, ""), ("none", 0.2, "")]
    assert _records(api.root) == []


@pytest.mark.parametrize("message", [
    "only " * 20_000,
    "must be " * 12_500,
    "temperature must be " + "(" * 100_000,
], ids=["only", "must-be", "brackets"])
def test_a_pathological_refusal_is_read_fast(message):
    """A provider's words cannot hold the interpreter: every run stays bounded, or a refusal blocks all threads."""
    started = time.perf_counter()
    _plan(_Refused(message))
    assert time.perf_counter() - started < 3


@pytest.mark.parametrize(("message", "hint"), [
    pytest.param(FIELD_REFUSAL, True, id='temperature is not supported-hint'),
    pytest.param("temperature is unsupported", False, id='temperature is unsupported-explicit'),
    pytest.param("unsupported parameter: temperature", True, id='unsupported parameter: temperature-hint'),
    pytest.param("temperature is not supported; top_p must be 1", False, id='temperature is not supported; top_p must be 1-explicit'),
    pytest.param("temperature is not supported, top-p must be 1", True, id='temperature is not supported, top-p must be 1-hint'),
    pytest.param("temperature is not supported for this model; max tokens must be 1", False, id='temperature is not supported for this model; max tokens must be 1-explicit'),
    pytest.param("temperature is not supported, must be removed", True, id='temperature is not supported, must be removed-hint'),  # no number: no statement of its values
    pytest.param("temperature is not supported; only default is allowed", False, id='temperature is not supported; only default is allowed-explicit'),
    pytest.param("temperature is not supported; only gpt-4 models accept it", True, id='temperature is not supported; only gpt-4 models accept it-hint'),
    pytest.param("temperature 0.2 is not supported", False, id='temperature 0.2 is not supported-explicit'),  # a legacy limit, not this rule: it may mean only the value, yet the field goes
    pytest.param("max_tokens is too large, must be 4096 or less; temperature is not supported", True, id='max_tokens is too large, must be 4096 or less; temperature is not supported-hint'),  # An unbound list that never ended states nothing of temperature: the legacy repair answers, as it always did.
    pytest.param("temperature is unsupported by this provider. must be at most 1", False, id='temperature is unsupported by this provider. must be at most 1-explicit'),
])
def test_genuine_field_refusals_keep_their_historical_repair(message, hint):
    retry, reasons = _plan(_Refused(message), hint=hint)
    assert retry is not None and "temperature" not in retry and reasons == ["provider_unsupported_field"]


@pytest.mark.parametrize("hint", [True, False], ids=["hint", "explicit"])
def test_a_typed_field_refusal_keeps_its_repair_beside_another_fields_bound(hint):
    retry, reasons = _plan(_Refused("temperature is not supported; top_p must be 1", param="temperature"), hint=hint)
    assert retry is not None and "temperature" not in retry and reasons == ["provider_unsupported_field"]


@pytest.mark.parametrize("hint", [True, False], ids=["hint", "explicit"])
def test_an_unreadable_list_states_nothing_that_cancels_a_field_refusal(hint):
    retry, reasons = _plan(_Refused(FIELD_REFUSAL, param="temperature", allowed_values="0, 1"), hint=hint)
    assert retry is not None and "temperature" not in retry and reasons == ["provider_unsupported_field"]


@pytest.mark.parametrize("hint", [True, False], ids=["hint", "explicit"])
def test_a_refused_sibling_field_goes_alone_beside_a_refused_temperature_value(hint):
    retry, reasons = _plan(_Refused("top_p is not supported; temperature must be 1"), hint=hint, top_p=0.9)
    assert retry is not None and "top_p" not in retry and retry["temperature"] == 0.2
    assert reasons == ["provider_unsupported_field"]


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
def test_field_evidence_and_hint_evidence_never_rewrite_each_other(api, asynchronous):
    call = _caller(PROVIDER_DRIVERS["openai-compatible"].model, asynchronous=asynchronous)
    api.rule = _only_one
    call(default_temperature=0.2)
    assert api.temperatures() == [0.2, OMITTED]

    # The route later refuses the field itself: the explicit path keeps its inherited repair and memory.
    api.rule = lambda payload: FIELD_REFUSAL if "temperature" in payload else None
    _message, usage = call(temperature=1)
    assert api.temperatures() == [1, OMITTED] and _applied(usage) == [("pending", FIELD_ACTION)]
    assert sorted(record["action"]["reason_code"] for record in _records(api.root)) == sorted(
        [HINT_WITHDRAWAL_REASON, "provider_unsupported_field"])

    # Field evidence still covers every value; an inherited record is not relabelled by replay.
    before = _records(api.root)
    _message, usage = call(temperature=1)
    assert api.temperatures() == [OMITTED] and _applied(usage) == [("durable", FIELD_ACTION)]
    assert _records(api.root) == before


def test_hint_withdrawal_is_a_closed_action_of_its_own():
    assert validate_wire_action(HINT_ACTION) == HINT_ACTION
    assert validate_wire_action({**HINT_ACTION, "fields": ["top_p"]}) == {}
    assert validate_wire_action({**HINT_ACTION, "fields": ["temperature", "top_p"]}) == {}
    assert wire_action_identity(HINT_ACTION) != wire_action_identity(FIELD_ACTION)


def _ladder(payload):
    return infer_tool_dialect(payload), payload.get("temperature", OMITTED), payload_effort(payload)


def _sends(api):
    sent = [_ladder(send["payload"]) for send in api.sends]
    api.temperatures()
    return sent


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
def test_prior_dialect_recovery_goes_first_and_the_hint_goes_on_its_rung(api, asynchronous):
    def rule(payload):
        custom = infer_tool_dialect(payload) == "openai_chat_custom"
        if custom and "temperature" in payload:
            return "custom tools cannot be used; temperature can only be 1"
        return "temperature can only be 1" if "temperature" in payload else None

    api.rule = rule
    call = _caller(PROVIDER_DRIVERS["openai"].model, asynchronous=asynchronous)
    _message, usage = call(default_temperature=0.2, tools=TOOLS)
    assert _sends(api) == [("openai_chat_custom", 0.2, "high"), ("function", 0.2, "high"), ("function", OMITTED, "high")]
    assert _applied(usage) == [("pending", HINT_ACTION)]

    with pytest.raises(Exception, match="temperature can only be 1"):
        call(temperature=0.2, tools=TOOLS)
    assert _sends(api) == [("openai_chat_custom", 0.2, "high"), ("function", 0.2, "high")]


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
def test_a_later_dialect_rebuild_does_not_restore_a_withdrawn_hint(api, asynchronous):
    def rule(payload):
        if "temperature" in payload:
            return "temperature can only be 1"
        return "custom tools cannot be used" if infer_tool_dialect(payload) == "openai_chat_custom" else None

    api.rule = rule
    call = _caller(PROVIDER_DRIVERS["openai"].model, asynchronous=asynchronous)
    _message, usage = call(default_temperature=0.2, tools=TOOLS)
    assert _sends(api) == [
        ("openai_chat_custom", 0.2, "high"), ("openai_chat_custom", OMITTED, "high"), ("function", OMITTED, "high"),
    ]
    assert _applied(usage) == [("pending", HINT_ACTION)], "the success teaches the rung that was accepted"
    assert [record["action"] for record in _records(api.root)] == [HINT_ACTION]

    # Rebuilt for its own rung, that memory needs the hint's origin to survive registration; an
    # explicit value at the same point goes out, and the provider's verdict on it stands.
    api.rule = lambda payload: "custom tools cannot be used" if infer_tool_dialect(
        payload) == "openai_chat_custom" else None
    call(default_temperature=0.2, tools=TOOLS)
    assert _sends(api) == [("openai_chat_custom", 0.2, "high"), ("function", OMITTED, "high")]
    call(temperature=0.2, tools=TOOLS)
    assert _sends(api) == [("openai_chat_custom", 0.2, "high"), ("function", 0.2, "high")]


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
def test_a_rung_memory_stays_withdrawn_on_the_next_rung_it_reaches(api, asynchronous):
    api.rule = lambda payload: "temperature can only be 1" if "temperature" in payload else None
    call = _caller(PROVIDER_DRIVERS["openai"].model, asynchronous=asynchronous)
    call(default_temperature=0.2, tools=TOOLS)
    assert _sends(api) == [("openai_chat_custom", 0.2, "high"), ("openai_chat_custom", OMITTED, "high")]

    api.rule = lambda payload: "custom tools cannot be used" if infer_tool_dialect(
        payload) == "openai_chat_custom" else None
    _message, usage = call(default_temperature=0.2, tools=TOOLS)
    assert _sends(api) == [("openai_chat_custom", OMITTED, "high"), ("function", OMITTED, "high")]
    assert _applied(usage) == [("pending", HINT_ACTION)]


def test_the_task_local_rung_keeps_it_withdrawn_and_teaches_nothing(api):
    def rule(payload):
        if "temperature" in payload:
            return "temperature can only be 1"
        if infer_tool_dialect(payload) == "openai_chat_custom":
            return "custom tools cannot be used"
        return "function tools are not supported" if payload_effort(payload) != "none" else None

    api.rule = rule
    _message, usage = _caller(PROVIDER_DRIVERS["openai"].model)(default_temperature=0.2, tools=TOOLS)
    assert _sends(api) == [
        ("openai_chat_custom", 0.2, "high"), ("openai_chat_custom", OMITTED, "high"),
        ("function", OMITTED, "high"), ("function", OMITTED, "none"),
    ]
    assert usage["request_wire"]["task_local"] is True and _records(api.root) == []


@pytest.mark.parametrize("hint_first", [False, True], ids=["reasoning-first", "hint-first"])
def test_reasoning_recovery_composes_with_the_withdrawal(api, hint_first):
    def rule(payload):
        effort_refused = payload_effort(payload) == "high"
        hint_refused = "temperature" in payload
        if hint_refused and (hint_first or not effort_refused):
            return KIMI_REFUSAL
        return "reasoning_effort value 'high' is not supported" if effort_refused else None

    api.rule = rule
    _message, usage = _caller(PROVIDER_DRIVERS["openai-compatible"].model)(default_temperature=0.2)
    assert _sends(api) == (
        [("none", 0.2, "high"), ("none", OMITTED, "high"), ("none", OMITTED, "medium")] if hint_first
        else [("none", 0.2, "high"), ("none", 0.2, "medium"), ("none", OMITTED, "medium")])
    assert {action["reason_code"] for _source, action in _applied(usage)} == {
        HINT_WITHDRAWAL_REASON, "provider_prescribed_value"}
    assert len(_records(api.root)) == 2


@pytest.mark.parametrize(("hint_first", "asynchronous"), [
    pytest.param(False, False, id='carrier-first-sync'),
    pytest.param(True, True, id='hint-first-async'),
])
def test_a_dropped_carrier_composes_with_the_withdrawal_and_its_memory(api, hint_first, asynchronous):
    def rule(payload):
        carrier_refused = bool(payload_effort(payload))
        if payload.get("temperature", 1) != 1 and (hint_first or not carrier_refused):
            return KIMI_REFUSAL
        return "model does not support thinking" if carrier_refused else None

    api.rule = rule
    call = _caller(PROVIDER_DRIVERS["openai-compatible"].model, asynchronous=asynchronous)
    call(default_temperature=0.2)
    if hint_first:
        assert _sends(api) == [("none", 0.2, "high"), ("none", OMITTED, "high"), ("none", OMITTED, "")]
        # That memory was taught beside the carrier, whose learned drop is another profile: one more
        # refusal teaches it there, and a route that once succeeded never starts failing.
        call(default_temperature=0.2)
        assert _sends(api) == [("none", 0.2, ""), ("none", OMITTED, "")]
    else:
        assert _sends(api) == [("none", 0.2, "high"), ("none", 0.2, ""), ("none", OMITTED, "")]

    _message, usage = call(default_temperature=0.2)
    assert _sends(api) == [("none", OMITTED, "")]
    assert [source for source, _action in _applied(usage)] == ["durable", "durable"]
    assert ("durable", HINT_ACTION) in _applied(usage)

    # The carrier's memory is still only the carrier's: explicit values go out, and the verdict stands.
    learned = _records(api.root)
    call(temperature=1)
    assert _sends(api) == [("none", 1, "")]
    with pytest.raises(Exception, match="only 1 is allowed"):
        call(temperature=0.2)
    assert _sends(api) == [("none", 0.2, "")] and _records(api.root) == learned


@pytest.mark.parametrize("surface", ["triad", "executor"])
def test_actual_review_authors_reach_the_withdrawal_as_a_hint(api, surface, tmp_path):
    from ouroboros.review_execution import ApiChatReviewExecutor, ReviewAssignment
    from ouroboros.tools.registry import ToolContext
    from ouroboros.tools.review_multi_model import _query_model

    api.rule = _only_one
    model = PROVIDER_DRIVERS["openai-compatible"].model
    client = LLMClient()
    if surface == "triad":
        async def triad():
            return await _query_model(client, model, [{"role": "user", "content": "Review"}], asyncio.Semaphore(1),
                                      ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="task-one"),
                                      slot_id="critic", effort="high")
        _model, payload, _extra = asyncio.run(triad())
        assert "error" not in payload, payload
    else:
        request = ReviewRequest(surface="test", goal="Review", messages=[{"role": "user", "content": "Review"}],
                                default_temperature=0.2)
        executor = ApiChatReviewExecutor(ReviewAssignment(request, ReviewSlot("critic", model), call_id="seat"), llm=client)
        client.chat(**executor._kwargs())
    assert api.temperatures() == [0.2, OMITTED]

    explicit = ReviewSlot("critic", model, temperature=0.2, default_temperature=0.3)
    request = ReviewRequest(surface="test", goal="Review", messages=[{"role": "user", "content": "Review"}],
                            default_temperature=0.2)
    executor = ApiChatReviewExecutor(ReviewAssignment(request, explicit, call_id="explicit"), llm=client)
    with pytest.raises(Exception, match="only 1 is allowed"):
        client.chat(**executor._kwargs())
    assert api.temperatures() == [0.2], "a seat's explicit temperature is never treated as the hint"
