"""Processing reaches the exact physical send without changing cognitive input."""

import asyncio
import copy
import hashlib
import json
import sys
from types import SimpleNamespace

import pytest

from ouroboros import config, model_wait, pricing, usage_accounting as ua
from ouroboros.llm import LLMClient
from ouroboros.llm_attempt import apply_processing_preference, processing_contract_headers
from tests._usage_store_testing import ledger_rows


@pytest.fixture
def transport(tmp_path, monkeypatch):
    root = tmp_path / "data"
    root.mkdir()
    monkeypatch.setattr(config, "DATA_DIR", root)
    monkeypatch.setattr(config, "SETTINGS_PATH", root / "settings.json")
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(root))
    monkeypatch.setenv("OUROBOROS_PROCESSING_PREFERENCE", "")
    monkeypatch.setenv("OUROBOROS_MODEL_PROCESSING_PREFERENCES", "{}")
    monkeypatch.setenv("TOTAL_BUDGET", "100")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(pricing, "estimate_cost_optional", lambda *a, **kw: None)
    monkeypatch.setattr(ua, "estimate_cost_optional", lambda *a, **kw: None)
    monkeypatch.setattr("ouroboros.llm.transport_no_proxy", lambda: False)  # the spawner's transport policy fact
    monkeypatch.setattr(LLMClient, "_SUPPORTED_PARAMS_FETCHED", True)
    monkeypatch.setattr(LLMClient, "_SUPPORTED_PARAMS_CACHE", {})
    monkeypatch.setattr(LLMClient, "_get_supported_parameters", lambda *a: None)
    sent = []

    class Response:
        status_code = 200

        def __init__(self, candidate):
            self.candidate = candidate

        def model_dump(self):
            if "speed" in self.candidate:
                return self.json()
            return {"choices": [{"message": {"role": "assistant", "content": "ok"}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 1, "cost": 0},
                    "service_tier": self.candidate.get("service_tier")}

        def json(self):
            return {"content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn",
                    "usage": {"input_tokens": 10, "output_tokens": 1,
                              "speed": self.candidate.get("speed")}}

    def create(**candidate):
        sent.append(copy.deepcopy(candidate))
        return Response(candidate)

    async def create_async(**candidate):
        return create(**candidate)

    client = LLMClient(api_key="test-key")
    monkeypatch.setattr(client, "_get_remote_client", lambda target: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    monkeypatch.setattr(client, "_get_async_remote_client", lambda target: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create_async))))

    def post(url, *, headers, json, **kwargs):
        sent.append({"payload": copy.deepcopy(json), "headers": copy.deepcopy(headers)})
        return Response(json)

    monkeypatch.setattr("requests.post", post)
    with ua.usage_scope(ua.UsageScope(drive_root=root, task_id="processing", root_task_id="processing")):
        yield root, client, sent


@pytest.mark.parametrize(("preference", "provider", "asynchronous"), [
    ("standard", "openai", False),
    ("fast", "openrouter", True),
    ("ultrafast", "openrouter", True),
    ("economy", "anthropic", False),
])
def test_native_processing_is_in_the_exact_accounted_request(transport, provider, preference, asynchronous):
    root, client, sent = transport
    model = f"{provider}::test-model"
    kwargs = dict(messages=[{"role": "user", "content": "unchanged input"}], model=model,
                  model_role="light", processing_preference=preference, max_tokens=123,
                  reasoning_effort="high")
    message, usage = asyncio.run(client.chat_async(**kwargs)) if asynchronous else client.chat(**kwargs)
    assert message["content"] == "ok"
    assert len(sent) == 1
    expected = ({"standard": "standard", "fast": "fast", "economy": "standard"}
                if provider == "anthropic" else
                {"standard": "default", "fast": "priority", "economy": "flex", "ultrafast": "ultrafast"})[preference]
    candidate = sent[0]["payload"] if provider == "anthropic" else sent[0]
    assert candidate["speed" if provider == "anthropic" else "service_tier"] == expected
    assert "processing_preference" not in candidate
    assert candidate.get("max_completion_tokens", candidate.get("max_tokens")) == 123
    if provider == "anthropic":
        assert ("fast-mode-2026-02-01" in sent[0]["headers"].get("anthropic-beta", "")) == (preference == "fast")
    rows = ledger_rows(root)
    assert [row["state"] for row in rows] == ["settled"]
    assert all(row["processing_preference"] == preference for row in rows)
    assert all(row["submitted_processing_mode"] == expected for row in rows)
    assert usage["processing"]["requested"] == preference
    assert usage["processing"]["submittedNative"] == expected
    assert usage["processing"]["observedNative"] == [expected]
    assert rows[-1]["processing"] == usage["processing"]
    from ouroboros.llm_attempt import _canonical_candidate_bytes
    assert rows[-1]["candidate_raw_sha256"] == hashlib.sha256(_canonical_candidate_bytes(candidate)).hexdigest()


def test_explicit_standard_does_not_inherit_global_fast(transport, monkeypatch):
    _root, client, sent = transport
    monkeypatch.setenv("OUROBOROS_PROCESSING_PREFERENCE", "fast")
    monkeypatch.setenv("OUROBOROS_MODEL_PROCESSING_PREFERENCES", '{"light":"standard"}')
    for role in ("main", "light"):
        client.chat([{"role": "user", "content": "same"}], "openai::same-model", model_role=role)
    client.chat([{"role": "user", "content": "same"}], "openai::same-model", processing_preference="")
    assert [item.get("service_tier") for item in sent] == ["priority", "default", None]


@pytest.mark.parametrize(("preference", "suffix", "reported"), [
    ("standard", "", None), ("ultrafast", "", 0.0), ("", ":variant", 1.25),
    ("standard", ":variant", None),
])
def test_private_router_uses_only_its_own_reported_price(transport, monkeypatch, preference, suffix, reported):
    from ouroboros.llm_stream import AssembledResponse

    root, _, _ = transport
    slug = "anthropic/synthetic-tariff-scope" + suffix
    client = LLMClient(api_key="synthetic-private-key", base_url="https://private.invalid/v1")
    lookup, sent = [], []

    def foreign_estimate(*args, **kwargs):
        lookup.append((args, kwargs))
        return 17.0  # A warm public tariff under the same model name.

    def create(**payload):
        sent.append(payload)
        usage = {"prompt_tokens": 11, "completion_tokens": 2, "cached_tokens": 3,
                 "provider_reported_cost_only": False}
        if reported is not None:
            usage["cost"] = reported
        return AssembledResponse({"choices": [{"message": {"role": "assistant", "content": "ok"},
                                               "finish_reason": "stop"}], "usage": usage})

    monkeypatch.setattr(pricing, "estimate_cost_optional", foreign_estimate)
    monkeypatch.setattr(ua, "estimate_cost_optional", foreign_estimate)
    monkeypatch.setattr("requests.get", lambda *a, **kw: pytest.fail("unexpected public GET"))
    monkeypatch.setattr(client, "_get_remote_client", lambda target: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    ua.stash_task_cache_split("processing", slug, 8, provider="openrouter", ttl_seconds=300)
    _, usage = client.chat([{"role": "user", "content": "hello"}], slug, max_tokens=8,
                           processing_preference=preference)
    [row] = ledger_rows(root)
    assert len(sent) == 1 and sent[0]["model"] == slug
    assert row["state"] == "settled" and row["reservation_upper_bound_usd"] is None
    assert (row["provider"], row["model"]) == (usage["provider"], usage["resolved_model"]) == ("openrouter", slug)
    assert row["cost_usd"] == usage["cost"] == reported
    assert row["cost_final"] is usage["cost_final"] is (reported is not None)
    assert row["provider_reported_cost_only"] is usage["provider_reported_cost_only"] is True
    assert ua.last_task_cache_split("processing", slug, provider="openrouter") == 8
    assert "synthetic-private-key" not in str(row) and "private.invalid" not in str(usage)
    assert lookup == []
    # Main/Safety's final display pass must respect the same terminal attempt.
    assert pricing.normalize_usage_cost(usage, model=slug, use_local=False) == (reported, slug, "openrouter", False)
    assert lookup == []


def test_public_router_prices_the_echo_and_keeps_the_ordinary_pool_out_of_the_receipt(transport, monkeypatch):
    from ouroboros.llm_stream import AssembledResponse

    root, client, _ = transport
    slug, lookups = "vendor/synthetic-tariff-scope", []

    def estimate(model, *args, processing_mode, allow_live_fetch, **kwargs):
        assert model == slug
        lookups.append((processing_mode, allow_live_fetch))
        return {"ordinary": 5.0, "priority": 0.25}[processing_mode]

    def create(**payload):
        assert "service_tier" not in payload
        return AssembledResponse({"choices": [{"message": {"role": "assistant", "content": "ok"},
                                               "finish_reason": "stop"}], "service_tier": "priority",
                                  "usage": {"prompt_tokens": 8, "completion_tokens": 1,
                                            "provider_reported_cost_only": True}})

    monkeypatch.setattr(pricing, "estimate_cost_optional", estimate)
    monkeypatch.setattr(ua, "estimate_cost_optional", estimate)
    monkeypatch.setattr(client, "_get_remote_client", lambda _: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    _, usage = client.chat([{"role": "user", "content": "hello"}], slug, processing_preference="", max_tokens=8)
    [row] = ledger_rows(root)
    assert row["reservation_upper_bound_usd"] == 5.0 and row["cost_usd"] == usage["cost"] == 0.25
    assert not row["cost_final"] and usage["cost_estimated"] and "provider_reported_cost_only" not in usage
    assert lookups[0] == ("ordinary", False) and [mode for mode, _ in lookups[1:]] == ["priority", "priority"]
    assert row["processing"] == usage["processing"]
    assert usage["processing"]["submittedNative"] is None and usage["processing"]["observed"] == "fast"


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("provider", ["openrouter", "openai"])
@pytest.mark.parametrize(("echo", "observed"), [
    ("ultrafast", "ultrafast"), ("priority", "fast"), ("default", "standard"), (None, "unknown"),
])
def test_ultrafast_receipt_uses_only_this_responses_or_router_reply(transport, monkeypatch, provider, stream, echo, observed):
    from tests.test_openai_responses import _Response, _Wire, _body, _message, _sse

    root, client, sent = transport
    monkeypatch.setenv("OPENAI_API_SURFACE", "responses")
    forged = {"observed": "ultrafast", "observedNative": ["ultrafast"]}

    def create(**candidate):
        sent.append(candidate)
        usage = {"cost": 0, "processing": forged, "service_tier": "ultrafast"}
        if provider == "openai":
            body = _body(_message("ok"), service_tier=echo, usage=usage)
            wire = _sse({"type": "response.completed", "response": body})
        else:
            message = {"role": "assistant", "content": "ok"}
            body = {"choices": [{"message": message}], "usage": usage, "service_tier": echo}
            chunks = [{"choices": [{"index": 0, "delta": message, "finish_reason": "stop"}], "service_tier": "ultrafast"},
                      {"choices": [], "usage": usage, "service_tier": echo}]
            wire = ("".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n").encode()
        return _Wire(wire) if stream else _Response(body)

    # The other API has no create function, so choosing the wrong surface fails the call.
    api = SimpleNamespace(create=create)
    remote = SimpleNamespace(responses=api) if provider == "openai" else SimpleNamespace(chat=SimpleNamespace(completions=api))
    monkeypatch.setattr(client, "_get_remote_client", lambda _target: remote)
    message, usage = client.chat([{"role": "user", "content": "hello"}], f"{provider}::same-model",
                                 processing_preference="ultrafast", stream=stream)
    assert message["content"] == "ok"
    assert [payload["service_tier"] for payload in sent] == ["ultrafast"]
    receipt = usage["processing"]
    assert (receipt["requested"], receipt["submitted"], receipt["submittedNative"]) == ("ultrafast",) * 3
    assert (receipt["observed"], receipt["observedNative"]) == (observed, [echo] if echo else [])
    assert receipt["reason"] == ("provider_mode_changed" if observed in {"fast", "standard"} else None)
    row, = ledger_rows(root)
    assert row["state"] == "settled" and row["processing"] == receipt
    binding = row["provider_receipt_binding"]
    assert binding["credential_sha256"] == hashlib.sha256(b"test-key").hexdigest()
    if provider == "openai":
        assert binding["api_surface"] == "responses" and binding["endpoint"] == "https://api.openai.com/v1"
    else:
        assert binding["endpoint"] == "https://openrouter.ai/api/v1"


@pytest.mark.parametrize(("provider", "base", "surface", "expected"), [
    ("openai", "https://api.openai.com/v1", "chat_completions", ("standard", "economy", "fast")),
    ("openai", "https://api.openai.com/v1", "responses", ("standard", "economy", "fast", "ultrafast")),
    ("openrouter", "https://openrouter.ai/api/v1", None, ("standard", "economy", "fast", "ultrafast")),
    ("anthropic", "https://api.anthropic.com/v1", None, ("standard", "fast")),
    ("openai", "https://gateway.invalid/v1", "responses", None),
    ("openrouter", "https://gateway.invalid/v1", None, None),
    ("openrouter", "https://openrouter.ai/api/v1", "messages", None),
    ("anthropic", "https://api.anthropic.com/v1", "responses", None),
    ("openai-compatible", "https://example.invalid/v1", "responses", None),
    ("future-adapter", "https://example.invalid/v1", None, None),
    ("claudexor", "", None, None),  # the engine declares its own ceiling
])
def test_catalog_adapter_modes_use_the_same_route_as_send(provider, base, surface, expected):
    from ouroboros.llm_attempt import processing_transport_modes

    target = {"provider": provider, "base_url": base, "api_surface": surface, "connection": "named"}
    assert processing_transport_modes(target) == expected
    for preference in ("standard", "economy", "fast", "ultrafast"):
        payload = {}
        apply_processing_preference({**target, "processing_preference": preference}, payload)
        if expected and preference in expected:
            assert payload == ({"speed": preference} if provider == "anthropic" else {"service_tier": {
                "standard": "default", "economy": "flex", "fast": "priority", "ultrafast": "ultrafast"}[preference]})
        elif expected and provider == "anthropic" and preference == "economy":
            assert payload == {"speed": "standard"}
        else:
            assert payload == {}


@pytest.mark.parametrize("provider", ["local", "deepseek", "minimax", "zai", "cloudru", "gigachat", "openai-compatible"])
def test_ordinary_only_host_adapters_match_their_wire(provider):
    from ouroboros.llm_routing import processing_transport_modes

    target = {"provider": provider}
    assert processing_transport_modes(target) == ("standard",)
    for preference in ("standard", "economy", "fast", "ultrafast"):
        payload = {"model": "fixture"}
        apply_processing_preference({**target, "processing_preference": preference}, payload)
        assert payload == {"model": "fixture"}


@pytest.mark.parametrize("provider,payload", [
    ("openai", {"service_tier": "default"}),
    ("anthropic", {"speed": "standard"}),
    ("openai-compatible", {"model": "custom"}),
])
def test_native_override_and_unknown_transport_shape_are_preserved(provider, payload):
    before = copy.deepcopy(payload)
    apply_processing_preference({"provider": provider, "processing_preference": "fast"}, payload)
    assert payload == before


@pytest.mark.parametrize("override", ["default", None])
def test_sdk_extra_body_overrides_the_projected_tier_even_if_null(override):
    from ouroboros.llm_attempt import submitted_processing_mode

    target = {"provider": "openai", "base_url": "https://api.openai.com/v1", "api_surface": "responses",
              "processing_preference": "ultrafast"}
    candidate = {"extra_body": {"service_tier": override}}
    apply_processing_preference(target, candidate)
    assert candidate == {"extra_body": {"service_tier": override}}
    assert target["processing_native_origin"] == "native_override"
    assert submitted_processing_mode(target, candidate) == (override or "")


def test_fast_headers_preserve_other_beta_intent():
    target = {"provider": "anthropic", "contract_headers": {"anthropic-beta": "another-beta"}}
    assert processing_contract_headers(target, {"speed": "fast"})["anthropic-beta"] == "another-beta,fast-mode-2026-02-01"
    assert target["contract_headers"]["anthropic-beta"] == "another-beta"
    assert processing_contract_headers({"provider": "anthropic"}, {"speed": "standard"}) == {}
    assert processing_contract_headers({**target, "contract_headers": {"anthropic-beta": "another-beta,fast-mode-2026-02-01"}},
                                       {"speed": "standard"}) == {"anthropic-beta": "another-beta"}


@pytest.mark.parametrize("asynchronous", [False, True])
def test_quota_reentry_keeps_captured_preference_and_same_owner(transport, monkeypatch, asynchronous):
    from ouroboros.llm_claudexor import ClaudexorModelNotDispatched

    root, _client, _sent = transport
    seen = []
    monkeypatch.setenv("OUROBOROS_PROCESSING_PREFERENCE", "fast")

    def invoke(self, *, model, model_role, processing_preference=None, model_poll_control=None):
        seen.append(processing_preference)
        if len(seen) == 1:
            error = ClaudexorModelNotDispatched({"code": "subscription_window_exhausted"})
            error.physical_attempt_capture = SimpleNamespace(state="released", attempt_id="attempt-one")
            raise error
        return {}, {}

    async def invoke_async(self, *, model, model_role, processing_preference=None, model_poll_control=None):
        return invoke(self, model=model, model_role=model_role, processing_preference=processing_preference,
                      model_poll_control=model_poll_control)

    wrapped = model_wait.model_waitable(invoke_async if asynchronous else invoke)
    with model_wait.task_model_wait_scope(task={"id": "processing"}, drive_root=root,
            event_queue=None, worker_slot_held=False, owner_control=lambda: None) as owner:
        def wait(llm, error, values, *args):
            monkeypatch.setenv("OUROBOROS_PROCESSING_PREFERENCE", "standard")
            assert model_wait.current_model_wait() is owner
            return values
        monkeypatch.setattr(owner, "wait", wait)
        result = wrapped(object(), model="claudexor::codex/model", model_role="light")
        if asynchronous:
            result = asyncio.run(result)
    assert seen == ["fast", "fast"]
    assert result[1]["ledger_attempt_ids"] == ["attempt-one"]


@pytest.mark.parametrize(("refusal", "asynchronous"), [("typed", False), ("capacity", True), ("unsupported", True)])
def test_typed_no_start_reprices_standard_without_changing_custom_tools(transport, monkeypatch, asynchronous, refusal):
    from ouroboros import send_clock
    from ouroboros._usage_money import cash_contribution
    from ouroboros._usage_rows import _physical_call_count
    from ouroboros.llm_attempt import ProcessingNotStarted
    from ouroboros.request_wire_recovery import current_wire_candidate

    root, client, _sent = transport
    reservations, candidates, catalogs = [], [], []
    monkeypatch.setattr(ua, "_reservation_cost", lambda request: (
        reservations.append((request.submitted_processing_mode, request.candidate_raw_sha256)) or
        (0.01 if request.submitted_processing_mode == "flex" else 0.04)))
    target = {**client._resolve_remote_target("openai::same-model"), "processing_preference": "economy"}
    payload = client._build_remote_kwargs(target, [{"role": "user", "content": "unchanged"}],
        "high", 123, "required", None, [{"type": "function", "function": {
            "name": "inspect", "description": "Read the requested value", "parameters": {
                "type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"]}}}])

    def create(**candidate):
        candidates.append(copy.deepcopy(candidate))
        catalogs.append(current_wire_candidate().custom_catalog_sha256)
        if len(candidates) == 1:
            if refusal == "typed":
                raise ProcessingNotStarted(RuntimeError("documented no-start receipt"), reason="capacity")
            error = RuntimeError("Structured provider refusal")
            error.status_code = 429 if refusal == "capacity" else 400
            error.body = {"error": {"code": "resource_unavailable" if refusal == "capacity" else "unsupported_parameter",
                                    "param": "service_tier"}}
            raise error
        return SimpleNamespace(model_dump=lambda: {"choices": [{"message": {
            "role": "assistant", "content": "ok"}}], "usage": {"cost": 0.04}})

    async def create_async(**candidate):
        return create(**candidate)

    from datetime import datetime, timedelta, timezone
    samples = iter(datetime(2026, 9, 26, tzinfo=timezone.utc) + timedelta(minutes=i) for i in range(10))
    monkeypatch.setattr(send_clock, "_now", lambda: next(samples))
    with send_clock.MainSendClock(send_clock.SendClockPolicy()).bound() as clock:
        if asynchronous:
            asyncio.run(client._create_chat_completion_with_retries_async(create_async, payload, target))
        else:
            client._create_chat_completion_with_retries(create, payload, target)
        assert len(candidates) == 2 and len(clock.notes) == 2 and clock.notes[0] != clock.notes[1]
        first, second = [send_clock.split_clock_note(p)[1] for p in candidates]
    assert {**first, "service_tier": "default"} == second
    assert candidates[1]["tools"][0]["type"] == "custom"
    assert catalogs[0] and catalogs[0] == catalogs[1]
    assert [mode for mode, _digest in reservations] == ["flex", "default"]
    assert reservations[0][1] != reservations[1][1]
    rows = ledger_rows(root)
    finals = list({row["attempt_id"]: row for row in rows}.values())
    assert [row["state"] for row in finals] == ["settled", "settled"]
    assert (finals[0]["settle_reason"], finals[0]["cost_usd"], finals[0]["cost_final"]) == ("processing_not_started", None, False)
    assert _physical_call_count(finals[0]) == 1 and float(cash_contribution(finals[0])[-1]) == 0.01
    assert [row["reservation_upper_bound_usd"] for row in finals] == [0.01, 0.04]
    assert not (root / "state/request_wire_compatibility.json").exists()
    assert payload["service_tier"] == "flex"


@pytest.mark.parametrize(("provider", "preference"), [("openai", "economy"), ("openrouter", "ultrafast")])
@pytest.mark.parametrize("status", [None, 400, 403, 429])
def test_unknown_or_unqualified_failure_never_falls_back(transport, provider, preference, status):
    root, client, _sent = transport
    calls = []
    target = {**client._resolve_remote_target(f"{provider}::same-model"), "processing_preference": preference}
    payload = client._build_remote_kwargs(target, [{"role": "user", "content": "input"}],
        "high", 123, "auto", None, None)

    def create(**candidate):
        calls.append(candidate)
        if status is None:
            raise TimeoutError("outcome unknown")
        error = RuntimeError("Provider refused the request")
        error.status_code = status
        error.body = {"error": {"code": "insufficient_quota" if status == 429 else "invalid_request_error",
                                "param": "service_tier"}}
        raise error

    with pytest.raises(TimeoutError if status is None else RuntimeError):
        client._create_chat_completion_with_retries(create, payload, target)
    assert len(calls) == 1
    rows = ledger_rows(root)
    assert rows[-1]["state"] == "unresolved"


_REFUSAL_COUNTERS = {
    "token_alias": {"completion_tokens": 0, "output_tokens": 7,
                    "_reported_token_usage": {"completion_tokens": 0}},
    "returned_token_details": {"input_tokens_details": {"cached_tokens": 7},
                               "prompt_tokens_details": {"cached_tokens": 0}},
    "native_cache_write": {"cache_creation": {"ephemeral_1h_input_tokens": 3}, "cache_write_tokens_by_ttl": {"1h": 0}},
    "reasoning_alias": {"reasoning_tokens": 0, "output_tokens_details": {"reasoning_tokens": 7}},
    "malformed_token": {"completion_tokens": 0, "output_tokens": False},
    "returned_audio": {"completion_tokens_details": {"audio_tokens": 2}},
    "video": {"prompt_tokens_details": {"video_tokens": 1}},
    "unquantified_modality": {"promptTokensDetails": [{"modality": "AUDIO", "tokenCount": None}]},
    "accepted_prediction": {"completion_tokens_details": {"accepted_prediction_tokens": 1}},
    "returned_rejected_prediction": {"completion_tokens_details": {"rejected_prediction_tokens": 1}},
    "malformed_reasoning": {"reasoning_tokens": -1},
    "returned_malformed_reasoning": {"reasoning_tokens": 0, "output_tokens_details": {"reasoning_tokens": False}},
}

# Each row describes a cumulative stream report, terminal usage, and the final price (if proven).
_REFUSAL_STREAM_PRICES = {
    "stream_total_below_wallets": ({"is_byok": True, "cost": 0.25, "total_cost": 0.375},
                                  {"is_byok": True, "cost_details": {"upstream_inference_cost": 0.25}}, None),
    "terminal_total_below_wallets": (
        {"is_byok": True, "cost": 0.25},
        {"total_cost": 0.375, "is_byok": True, "cost_details": {"upstream_inference_cost": 0.25}}, None),
    "repeated_gateway": ({"is_byok": True, "cost": 0.25, "total_cost": 0.375},
                         {"is_byok": True, "cost": 0.25}, 0.375),
    "equal_wallet_total": ({"is_byok": True, "cost": 0.125, "cost_details": {"upstream_inference_cost": 0.25}},
                           {"cost": 0.375, "completion_tokens": 2}, 0.375),
    "decreased_total": ({"cost": 0.25}, {"cost": 0.125}, None),
}


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("evidence", ["unpriced", "sdk_zero", "charged", "output", "upstream", "charged_with_cause",
                                      "sdk_price", "invalid_price", "stream_receipt", "returned_charge", "returned_unpriced",
                                      "returned_zero", "conflicting_totals", "returned_conflicting_totals",
                                      "quoted_json", "quoted_refusal", "quoted_native_delta", "quoted_unknown_frame",
                                      "quoted_malformed_frame",
                                      "byok", "byok_partial", "strict", "returned_answer", "returned_finish", *_REFUSAL_COUNTERS,
                                      *_REFUSAL_STREAM_PRICES,
                                      "nested_under_bound", "nested_compatible", "sdk_price_under_bound",
                                      "alternate_under_bound", "stream_under_bound", "split_only", "split_under_bound"])
def test_openrouter_ultrafast_retries_only_a_clean_refusal_at_standard(transport, monkeypatch, asynchronous, evidence):
    import httpx
    from ouroboros.llm_attempt import ProcessingNotStarted, physical_attempt_headroom, single_physical_send
    from ouroboros.observability import read_blob_ref

    root, client, sent = transport
    native = {"code": "unsupported_value", "param": "service_tier"}
    envelope = {"error": native}
    refusal = RuntimeError("unsupported session_id")  # Also matches the generic cache-parameter repair.
    refusal.status_code, refusal.body = 400, native
    if evidence.startswith("charged") or evidence == "returned_charge":
        envelope["usage"] = {"cost": 0.125}
    if evidence.endswith("conflicting_totals"):
        envelope["usage"] = {"cost": 0, "total_cost": 0.125}
    if evidence == "returned_zero":
        envelope["_stream_receipt"] = {"model_output_received": True}
        envelope["usage"] = {"cost": 0, "prompt_tokens": 0, "input_tokens": None,
                             "completion_tokens": 0, "output_tokens": 0, "private_extension": {"modality": "AUDIO"},
                             "input_tokens_details": {"cached_tokens": 0, "unknown_counter": 7},
                             "completion_tokens_details": {"audio_tokens": None, "accepted_prediction_tokens": 0,
                                                          "rejected_prediction_tokens": None},
                             "_reported_token_usage": {"completion_tokens": 7, "audio_tokens": 3},
                             "cache_write_tokens_by_ttl": {"1h": 3}}
    if evidence in _REFUSAL_COUNTERS:
        envelope["usage"] = {"cost": 0, **_REFUSAL_COUNTERS[evidence]}
    if evidence in _REFUSAL_STREAM_PRICES:
        refusal.stream_usage, envelope["usage"], expected_price = _REFUSAL_STREAM_PRICES[evidence]
    if evidence in {"byok", "byok_partial"}:
        envelope["usage"] = {"is_byok": True, "cost": 0.125}
    if evidence == "byok":
        envelope["usage"].update(total_cost=0.375, cost_details={"upstream_inference_cost": 0.25})
    if evidence in {"nested_under_bound", "nested_compatible", "sdk_price_under_bound", "alternate_under_bound",
                    "stream_under_bound", "split_only", "split_under_bound"}:
        native["usage"] = {"is_byok": True, "cost": 0.25}
    if evidence == "nested_under_bound":
        envelope["total_cost_usd"] = 0.125
    if evidence == "nested_compatible":  # Repeated gateway evidence is one component, not two charges.
        envelope.update(total_cost_usd=0.375, usage=dict(native["usage"]))
    if evidence == "alternate_under_bound":
        refusal.body = {"error": {"code": native["code"], "param": native["param"]}, "total_cost_usd": 0.125}
    if evidence == "stream_under_bound":
        refusal.stream_usage = {"cost": 0.125}
    if evidence in {"split_only", "split_under_bound"}:
        envelope["usage"] = {"is_byok": True, "cost_details": {"upstream_inference_cost": 0.25}}
    if evidence == "split_under_bound":
        envelope["total_cost_usd"] = 0.375
    if evidence == "charged_with_cause":
        refusal.__cause__ = httpx.ConnectError("upstream unavailable")
        refusal.stream_usage = {"cost": 0}
    if evidence == "output":
        envelope.update(choices=[{"delta": {"content": "partial"}}], usage={"cost": 0})
    if evidence == "returned_finish":
        envelope.update(choices=[{"finish_reason": "length", "delta": {}}], usage={"cost": 0})
    if evidence == "stream_receipt":
        refusal.stream_receipt = {"model_output_received": True}
    if evidence == "upstream":
        envelope["openrouter_metadata"] = {"attempts": [{"provider": "upstream", "status": 503}]}
    if evidence.startswith("quoted_"):
        prior = {"quoted_native_delta": 'event: content_block_delta\ndata: {"type":"content_block_delta",'
                 '"index":0,"delta":{"type":"text_delta","text":"partial"}}\n\n',
                 "quoted_unknown_frame": 'data: {"type":"unknown_event"}\n\n',
                 "quoted_malformed_frame": 'data: {not JSON}\n\n'}.get(evidence, "")
        raw_error = '{"type":"error","error":{"type":"rate_limit_error","status_code":429}}'
        raw = raw_error if evidence == "quoted_json" else f"{prior}event: error\ndata: {raw_error}\n\n"
        native["metadata"] = {"provider_name": "synthetic-upstream", "raw": raw}
    refusal.response = SimpleNamespace(status_code=400, json=lambda: envelope)
    if evidence in {"sdk_price", "sdk_price_under_bound"}:
        refusal.response.total_cost_usd = 0.125
    if evidence == "sdk_zero":
        refusal.response.total_cost_usd = 0
    if evidence == "invalid_price":
        envelope["total_cost_usd"] = "invalid"

    def create(**candidate):
        sent.append(copy.deepcopy(candidate))
        if len(sent) == 1:
            if evidence.startswith("returned_") or evidence == "strict":
                native["status_code"] = 400
                if evidence == "returned_answer":
                    envelope.update(choices=[{"message": {"content": "usable"}}], usage={"cost": 0.125})
                return SimpleNamespace(status_code=200, model_dump=lambda: envelope)
            raise refusal
        return SimpleNamespace(model_dump=lambda: {"choices": [{"message": {"content": "ok"}}], "usage": {"cost": 0}})

    async def async_create(**candidate):
        return create(**candidate)

    monkeypatch.setattr(client, "_get_remote_client", lambda _: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    monkeypatch.setattr(client, "_get_async_remote_client", lambda _: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=async_create))))
    kwargs = dict(messages=[{"role": "user", "content": "hello"}], model="openrouter::same-model",
                  processing_preference="ultrafast", cache_affinity="synthetic-session", max_tokens=123)

    def call():
        return asyncio.run(client.chat_async(**kwargs)) if asynchronous else client.chat(**kwargs)

    if evidence == "returned_answer":
        message, _ = call()
        assert message["content"] == "usable" and [p["service_tier"] for p in sent] == ["ultrafast"]
        [row] = ledger_rows(root)
        assert (row["state"], row["cost_usd"], row["cost_final"]) == ("settled", 0.125, True)
        return
    if evidence == "strict":
        with single_physical_send(lambda _reservation: None), pytest.raises(ProcessingNotStarted):
            call()
        assert [p["service_tier"] for p in sent] == ["ultrafast"]
        [row] = ledger_rows(root)
        assert (row["settle_reason"], row["cost_usd"], row["cost_final"]) == ("processing_not_started", None, False)
        return
    if evidence not in {"unpriced", "sdk_zero", "returned_unpriced", "returned_zero", "quoted_json", "quoted_refusal"}:
        with pytest.raises(RuntimeError) as failed:
            call()
        assert failed.value.processing_refusal_conflict
        assert [p["service_tier"] for p in sent] == ["ultrafast"]
        [row] = ledger_rows(root)
        retained = read_blob_ref(root, row["physical_failure"]["evidence_ref"])
        if evidence == "alternate_under_bound":
            assert retained["body"] == failed.value.body == refusal.body
            assert retained["alternate_body"] == envelope
        else:
            assert retained["body"] == failed.value.body == envelope
        if evidence == "sdk_price_under_bound":
            assert retained["reported_total_cost_usd"] == 0.125
        if evidence == "stream_under_bound":
            assert retained["stream_usage"] == {"cost": 0.125}
        if evidence in _REFUSAL_STREAM_PRICES:
            assert retained["stream_usage"] == refusal.stream_usage
            expected_state = "unresolved" if expected_price is None else "settled"
            assert (row["state"], row.get("cost_usd"), bool(row.get("cost_final"))) == (
                expected_state, expected_price, expected_price is not None)
            if evidence == "equal_wallet_total":
                assert row["cost_basis"] == "byok_gateway_plus_upstream" and row["completion_tokens"] == 2
        elif evidence.startswith("charged") or evidence in {"returned_charge", "sdk_price"}:
            assert (row["state"], row["cost_usd"], row["cost_final"]) == ("settled", 0.125, True)
        elif evidence in {"byok", "nested_compatible"}:
            assert (row["state"], row["cost_usd"], row["cost_final"]) == ("settled", 0.375, True)
            if evidence == "byok":
                assert row["cost_basis"] == "byok_gateway_plus_upstream"
        else:
            assert row["state"] == "unresolved" and row.get("cost_usd") is None and not row.get("cost_final")
        return
    with ua.physical_attempt_limit(2):
        _, usage = call()
        assert physical_attempt_headroom() == 0
    assert [p["service_tier"] for p in sent] == ["ultrafast", "default"]
    first_payload, second_payload = sent
    assert first_payload["max_tokens"] == 123
    assert second_payload == {**first_payload, "service_tier": "default"}
    first, second = ledger_rows(root)
    assert (first["state"], first["settle_reason"]) == ("settled", "processing_not_started")
    if evidence in {"sdk_zero", "returned_zero"}:
        assert (first["cost_usd"], first["cost_final"]) == (0, True)
    else:
        assert (first["cost_usd"], first["cost_final"]) == (None, False)
    assert first["physical_failure"]["provider_status_code"] == 400
    assert first["physical_failure"]["stage"] == (
        "response_body_error" if evidence.startswith("returned_") else "raised_exception")
    assert second["state"] == "settled"
    assert first["provider_receipt_binding"] == second["provider_receipt_binding"]
    assert second["processing"] == usage["processing"] == {
        "requested": "ultrafast", "submitted": "standard", "submittedNative": "default",
        "observed": "unknown", "observedNative": [], "source": "host_request", "reason": "submitted_mode_differs"}


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(("prior", "terminal", "cost"), [
    pytest.param([{"choices": [{"index": 0, "delta": {"role": "assistant"}}]}],
                 {"usage": {"cost": 0}}, 0.0, id="literal-zero-without-retry-proof"),
    pytest.param([], {}, None, id="typed-error-is-not-a-price"),
    pytest.param([{"choices": [{"index": 0, "delta": {}, "finish_reason": "length"}]}],
                 {"usage": {"cost": 0}}, None, id="prior-finish-without-text"),
    pytest.param([{"usage": {"completion_tokens": 2}}, {"usage": {"completion_tokens": 0}}],
                 {"usage": {"cost": 0}}, None, id="prior-counter-overwritten-with-zero"),
    pytest.param([{"openrouter_metadata": {"attempts": [{"status": 400}]}}],
                 {"openrouter_metadata": {"attempts": []}, "usage": {"cost": 0}},
                 None, id="prior-routing-evidence-overwritten"),
    pytest.param([{"usage": {"cost": 0.25, "completion_tokens": 2, "cached_tokens": 1, "cache_write_tokens": 1}},
                  {"usage": {"completion_tokens": 4, "cache_write_tokens": 3}}],
                 {"usage": {"prompt_tokens": 7, "completion_tokens": 5, "cached_tokens": 0}},
                 0.25, id="prior-bill-freshest-count-per-metric"),
    pytest.param([{"usage": {"cost": 0.125}}],
                 {"usage": {"is_byok": False, "cost": 0.25}}, 0.25, id="higher-ordinary-total"),
    pytest.param([{"usage": {"cost": 0.25}}],
                 {"usage": {"cost": 0.125}}, None, id="terminal-total-below-prior"),
    pytest.param([{"usage": {"is_byok": True, "cost": 0.25}}],
                 {"usage": {"cost": 0.25}}, None, id="unlabelled-gateway-is-not-a-total"),
    pytest.param([{"usage": {"is_byok": True, "cost": 0.25}}],
                 {"usage": {"is_byok": False, "cost": 0.25}}, None, id="contradictory-wallet-labels"),
    pytest.param([{"usage": {"is_byok": True, "cost": 0.125,
                             "cost_details": {"upstream_inference_cost": 0.25}}}],
                 {"usage": {"cost": 0.375, "completion_tokens": 2}}, 0.375, id="corroborated-wallet-total"),
    pytest.param([{"usage": {"is_byok": True, "cost": 0.25}}],
                 {"usage": {"is_byok": True, "cost_details": {"upstream_inference_cost": 0.25},
                            "total_cost": 0.375}}, None, id="terminal-total-below-labelled-wallets"),
])
def test_chat_sse_tier_refusal_reads_prior_reports_before_settlement(transport, monkeypatch, asynchronous,
                                                                    prior, terminal, cost):
    from ouroboros.llm_stream import ProviderStreamError
    from ouroboros.observability import read_blob_ref
    from tests.test_openrouter_transport_evidence import Wire

    root, client, sent = transport
    error_body = {**terminal, "error": {"status_code": 400, "code": "unsupported_value", "param": "service_tier"}}

    def frame(body):
        return ("data: " + json.dumps(body) + "\n\n").encode()

    def create(**candidate):
        sent.append(candidate)
        assert len(sent) == 1
        return Wire([frame(body) for body in (*prior, error_body)])

    async def create_async(**candidate):
        return create(**candidate)

    monkeypatch.setattr(client, "_get_remote_client", lambda _: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    monkeypatch.setattr(client, "_get_async_remote_client", lambda _: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create_async))))
    kwargs = dict(messages=[{"role": "user", "content": "once"}], model="openrouter::same-model",
                  stream=True, processing_preference="ultrafast")

    with pytest.raises(ProviderStreamError) as failed:
        asyncio.run(client.chat_async(**kwargs)) if asynchronous else client.chat(**kwargs)
    assert failed.value.processing_refusal_conflict
    assert [p["service_tier"] for p in sent] == ["ultrafast"]
    [first] = ledger_rows(root)

    assert (first["state"], first.get("cost_usd"), bool(first.get("cost_final"))) == (
        "settled" if cost is not None else "unresolved", cost, cost is not None)
    retained = read_blob_ref(root, first["physical_failure"]["evidence_ref"])
    assert retained["body"] == error_body
    assert retained["stream_prior_usage"] == [body for body in prior if "choices" not in body]
    assert retained["stream_receipt"]["complete"] is False
    if terminal.get("usage", {}).get("completion_tokens") == 5:
        assert (first["prompt_tokens"], first["completion_tokens"], first["cached_tokens"],
                first["cache_write_tokens"]) == (7, 5, 0, 3)
    if cost == 0.375:
        assert first["cost_basis"] == "byok_gateway_plus_upstream" and first["completion_tokens"] == 2


@pytest.mark.parametrize(("explicit_error", "prior_cost", "terminal_total", "expected"), [
    (True, 0.0, None, None), (True, 0.125, None, 0.125),
    (False, 0.0, None, None), (True, 0.0, 0.0, 0.0),
    pytest.param(True, 0.125, 0.25, 0.25, id="higher-terminal-total"),
    pytest.param(True, 0.25, 0.125, None, id="terminal-total-below-prior"),
    pytest.param(False, 0.125, None, 0.125, id="transport-close-keeps-prior-bill"),
])
def test_chat_sse_failure_does_not_revive_rejected_zero(
        transport, monkeypatch, explicit_error, prior_cost, terminal_total, expected):
    from ouroboros.llm_stream import IncompleteProviderStream, ProviderStreamError
    from tests.test_openrouter_transport_evidence import Wire

    root, client, sent = transport

    def create(**candidate):
        sent.append(candidate)
        frames = [{"choices": [{"index": 0, "delta": {"content": "partial"}}]},
                  {"usage": {"cost": prior_cost, "prompt_tokens": 0, "completion_tokens": 0}, "choices": []}]
        if explicit_error:
            frames.append({"error": {"code": 502, "message": "failed"}})
            if terminal_total is not None:
                frames[-1]["total_cost_usd"] = terminal_total
        return Wire([("data: " + json.dumps(frame) + "\n\n").encode() for frame in frames])

    monkeypatch.setattr(client, "_get_remote_client", lambda _: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    with pytest.raises(ProviderStreamError if explicit_error else IncompleteProviderStream):
        client.chat([{"role": "user", "content": "once"}], "openrouter::same-model", stream=True)
    assert len(sent) == 1
    [row] = ledger_rows(root)
    assert (row["state"], row.get("cost_usd"), bool(row.get("cost_final"))) == (
        "unresolved" if expected is None else "settled", expected, expected is not None)


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
def test_returned_error_finish_never_retries_partial_chat_or_projects_final_zero(transport, monkeypatch, asynchronous):
    from ouroboros._usage_response import UnusableReplyError

    root, client, sent = transport
    body = {"error": {"code": 502, "message": "temporary upstream failure"},
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "cost": 0},
            "choices": [{"finish_reason": "error", "message": {"role": "assistant", "content": "partial"}}]}

    def create(**candidate):
        sent.append(candidate)
        assert len(sent) == 1
        return SimpleNamespace(model_dump=lambda: copy.deepcopy(body))

    async def create_async(**candidate):
        return create(**candidate)

    monkeypatch.setattr(client, "_get_remote_client", lambda _: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    monkeypatch.setattr(client, "_get_async_remote_client", lambda _: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create_async))))
    args = ([{"role": "user", "content": "once"}], "openrouter::same-model")
    kwargs = {"stream": False, "processing_preference": "standard"}
    with pytest.raises(UnusableReplyError) as failed:
        asyncio.run(client.chat_async(*args, **kwargs)) if asynchronous else client.chat(*args, **kwargs)
    usage = failed.value.usage
    assert len(sent) == 1 and usage["response_finish_reason"] == "error"
    assert failed.value.reply_message["content"] == "partial"
    assert usage.get("provider_error") and (usage["cost"], usage["cost_final"]) == (None, False)
    [attempt] = ledger_rows(root)
    assert (attempt.get("cost_usd"), bool(attempt.get("cost_final"))) == (None, False)


@pytest.mark.parametrize(("frames", "raised", "expected"), [
    pytest.param([{"total_cost_usd": 0},
                  {"choices": [{"index": 0, "delta": {"content": "partial"}}]},
                  {"error": {"code": 502}}], "ProviderStreamError", None, id="midstream-zero"),
    pytest.param([{"usage": {"cost": 0.125}},
                  {"choices": [{"index": 0, "delta": {"content": "partial"}, "finish_reason": "error"}],
                   "total_cost_usd": 0.25}], "ProviderStreamError", 0.25, id="choice-ends-with-price"),
    pytest.param([{"total_cost_usd": 0.25}], "IncompleteProviderStream", 0.25, id="eof-after-price-only"),
])
def test_r3_stream_prices_follow_wire(transport, monkeypatch, frames, raised, expected):
    from ouroboros import llm_stream
    from tests.test_openrouter_transport_evidence import Wire

    root, client, sent = transport

    def create(**candidate):
        sent.append(candidate)
        return Wire([("data: " + json.dumps(frame) + "\n\n").encode() for frame in frames])

    monkeypatch.setattr(client, "_get_remote_client", lambda _: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    with pytest.raises(getattr(llm_stream, raised)):
        client.chat([{"role": "user", "content": "once"}], "openrouter::same-model", stream=True)
    assert len(sent) == 1
    [row] = ledger_rows(root)
    assert (row.get("cost_usd"), bool(row.get("cost_final"))) == (expected, expected is not None)


def test_r3_chat_error_finish_without_error_object_is_not_free(transport, monkeypatch):
    from ouroboros._usage_response import UnusableReplyError

    root, client, sent = transport
    body = {"usage": {"prompt_tokens": 0, "completion_tokens": 0, "cost": 0},
            "choices": [{"finish_reason": "error", "message": {"role": "assistant", "content": "partial"}}]}

    def create(**candidate):
        sent.append(candidate)
        return SimpleNamespace(model_dump=lambda: copy.deepcopy(body))

    monkeypatch.setattr(client, "_get_remote_client", lambda _: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    with pytest.raises(UnusableReplyError) as failed:
        client.chat([{"role": "user", "content": "once"}], "openrouter::same-model")
    assert len(sent) == 1
    [row] = ledger_rows(root)
    assert (row.get("cost_usd"), bool(row.get("cost_final"))) == (None, False)
    assert (failed.value.usage["cost"], failed.value.usage["cost_final"]) == (None, False)


@pytest.mark.parametrize("full_bill", [True, False], ids=["explicit-full-zero", "partial-gateway-zero"])
def test_r3_responses_failed_price_agrees_between_client_and_ledger(transport, monkeypatch, full_bill):
    from ouroboros._usage_response import UnusableReplyError
    from tests.test_openai_responses import _body, _message, _target

    root, client, sent = transport
    body = (_body(status="failed", usage={"input_tokens": 10}, total_cost_usd=0) if full_bill else
            _body(_message("partial", status="incomplete"), status="failed", usage={"cost": 0}))

    def create(**candidate):
        sent.append(candidate)
        return SimpleNamespace(model_dump=lambda: copy.deepcopy(body))

    monkeypatch.setattr(client, "_resolve_remote_target", lambda _model: _target())
    monkeypatch.setattr(client, "_get_remote_client", lambda _: SimpleNamespace(responses=SimpleNamespace(create=create)))
    with pytest.raises(UnusableReplyError) as failed:
        client.chat([{"role": "user", "content": "once"}], "openai::gpt-6.1-terra")
    assert len(sent) == 1
    [row] = ledger_rows(root)
    expected = (0.0, True) if full_bill else (None, False)
    assert (failed.value.usage["cost"], failed.value.usage["cost_final"]) == expected
    assert (row.get("cost_usd"), bool(row.get("cost_final"))) == expected


def test_partial_ultrafast_stream_never_authorizes_a_second_send(transport, monkeypatch):
    from ouroboros.llm_stream import IncompleteProviderStream
    from tests.test_openai_responses import _Wire

    root, client, sent = transport
    def create(**candidate):
        sent.append(candidate)
        chunk = {"service_tier": "priority", "choices": [{"index": 0, "delta": {"content": "partial"}}]}
        return _Wire(f"data: {json.dumps(chunk)}\n\n".encode())

    monkeypatch.setattr(client, "_get_remote_client", lambda _: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    with pytest.raises(IncompleteProviderStream):
        client.chat([{"role": "user", "content": "once"}], "openrouter::same-model", stream=True,
                    processing_preference="ultrafast")
    assert [p["service_tier"] for p in sent] == ["ultrafast"]
    row, = ledger_rows(root)
    assert row["state"] == "unresolved" and not row.get("processing")


@pytest.mark.parametrize("charged", [False, True])
def test_anthropic_fast_rate_refusal_settles_then_sends_standard(transport, monkeypatch, charged):
    import requests

    root, client, sent = transport
    success = requests.post

    def post(url, **kwargs):
        if not sent:
            sent.append({"payload": copy.deepcopy(kwargs["json"]), "headers": kwargs["headers"]})
            body = {"error": {"type": "rate_limit_error", "message": "Fast rate limit"}}
            if charged:
                body["usage"] = {"cost": 0.125}
            return SimpleNamespace(status_code=429, text=json.dumps(body), reason="Too Many Requests",
                                   url=url, json=lambda: body)
        return success(url, **kwargs)

    monkeypatch.setattr("requests.post", post)
    def call():
        return client.chat([{"role": "user", "content": "same input"}],
                           "anthropic::test-model", processing_preference="fast")

    if charged:
        with pytest.raises(requests.HTTPError):
            call()
        assert [entry["payload"]["speed"] for entry in sent] == ["fast"]
        [row] = ledger_rows(root)
        assert (row["state"], row["cost_usd"]) == ("settled", 0.125)
        return
    message, _usage = call()
    assert message["content"] == "ok"
    assert [entry["payload"]["speed"] for entry in sent] == ["fast", "standard"]
    assert "fast-mode-2026-02-01" in sent[0]["headers"].get("anthropic-beta", "")
    assert "fast-mode-2026-02-01" not in sent[1]["headers"].get("anthropic-beta", "")
    rows = ledger_rows(root)
    assert [row["state"] for row in rows] == ["settled", "settled"]
    assert rows[0]["settle_reason"] == "processing_not_started"


def _known_sibling_charge_before_the_retry(monkeypatch, root, task_id, first_mode):
    """Price the first mode at $0.01; before the standard retry is priced, a sibling's
    $0.02 lands at a final price: the tree's KNOWN spend reaches the $0.02 cap (#1487)."""
    landed = []

    def price(request):
        if request.submitted_processing_mode == first_mode:
            return 0.01
        if not landed:
            landed.append(True)
            with ua.usage_scope(ua.UsageScope(drive_root=root, task_id="sibling", root_task_id=task_id)):
                held = ua.reserve_attempt(ua.AttemptRequest(model="m", provider="p", reservation_usd=0.02))
                ua.mark_dispatched(held)
                ua.settle_attempt(held, {}, cost_usd=0.02, cost_final=True)
        return 0.04

    monkeypatch.setattr(ua, "_reservation_cost", price)


def test_standard_retry_is_refused_when_known_spend_reached_the_budget(transport, monkeypatch):
    """The retry gets its own admission: known spend that reached the cap after the
    first attempt was admitted refuses the retry's reservation; nothing is sent."""
    from ouroboros.llm_attempt import ProcessingNotStarted

    root, client, _sent = transport
    calls = []
    _known_sibling_charge_before_the_retry(monkeypatch, root, "processing", "flex")
    target = {**client._resolve_remote_target("openai::same-model"), "processing_preference": "economy"}
    payload = client._build_remote_kwargs(target, [{"role": "user", "content": "input"}],
        "high", 123, "auto", None, None)

    def create(**candidate):
        calls.append(candidate)
        raise ProcessingNotStarted(RuntimeError("typed no-start fixture"), reason="capacity")

    with ua.usage_scope(ua.UsageScope(drive_root=root, task_id="processing", root_task_id="processing", root_limit_usd=0.02)):
        with pytest.raises(ua.BudgetExceeded):
            client._create_chat_completion_with_retries(create, payload, target)
    assert len(calls) == 1
    finals = {row["attempt_id"]: row for row in ledger_rows(root)}.values()
    assert [(row["task_id"], row["state"]) for row in finals] == [("processing", "settled"), ("sibling", "settled")]


@pytest.mark.parametrize("provider,expected", [("openrouter", "default"), ("anthropic", "standard")])
def test_direct_web_helper_carries_its_own_role_preference(transport, monkeypatch, provider, expected):
    from ouroboros import llm

    root, _client, _sent = transport
    sent, clients = [], []
    monkeypatch.setenv("OUROBOROS_PROCESSING_PREFERENCE", "fast")
    monkeypatch.setenv("OUROBOROS_MODEL_PROCESSING_PREFERENCES", '{"websearch":"standard"}')

    def create(**candidate):
        sent.append(candidate)
        return {"usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0}}

    def factory(**kwargs):
        clients.append(kwargs)
        def messages_create(*, model, messages, max_tokens, tools, extra_body=None):
            return create(model=model, messages=messages, max_tokens=max_tokens,
                          tools=tools, **(extra_body or {}))
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
                               messages=SimpleNamespace(create=messages_create))

    if provider == "openrouter":
        monkeypatch.setattr("ouroboros.net_transport.web_search_openai_client", factory)
        llm.openrouter_web_search_server_tool(api_key="test", model="model", query="query", search_context_size="low")
    else:
        monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(Anthropic=factory))
        llm.anthropic_web_search_server_tool(api_key="test", model="model", query="query")
    assert len(sent) == 1
    assert sent[0]["speed" if provider == "anthropic" else "service_tier"] == expected
    rows = ledger_rows(root)
    assert rows[-1]["submitted_processing_mode"] == expected


def _search_client_double(monkeypatch, usage):
    """The search SDK client: records its construction and each send, answers with ``usage``."""
    sent, clients = [], []

    def create(**candidate):
        sent.append(copy.deepcopy(candidate))
        return {"choices": [{"message": {"role": "assistant", "content": "answer"}}], "usage": dict(usage)}

    def factory(**kwargs):
        clients.append(kwargs)
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    monkeypatch.setattr("ouroboros.net_transport.web_search_openai_client", factory)
    return sent, clients


@pytest.mark.parametrize(("preference", "reply", "shown"), [
    (None, {"cost": 0.0042}, (0.0042, True)),
    ("standard", {"total_cost": 0.0042}, (0.0042, True)),  # no ``cost``: the reply's total is still its price
    (None, {"cost": 0}, (0.0, True)),
    (None, {"cost": "invalid"}, (None, False)),
    (None, {"is_byok": True, "cost": 0.125, "cost_details": {"upstream_inference_cost": 0.25}}, (0.375, True)),
    (None, {"is_byok": True, "cost": 0.125}, (None, False)),
    (None, {"is_byok": True, "cost": 0.125, "cost_details": {"upstream_inference_cost": -1}}, (None, False)),
    (None, {}, (None, False)),  # unreported: unknown and non-final, no tariff in its place
], ids=["cost", "total_cost", "zero", "invalid", "byok", "byok-without-upstream", "byok-invalid-upstream", "unknown"])
def test_named_openrouter_search_sends_and_bills_on_its_own_resolved_connection(transport, monkeypatch, preference,
                                                                                reply, shown):
    """The search leg resolves its connection once and sends on that target, never a synthesized public one: the
    row's key and headers, its provider routing merged as a chat send merges it, its billing identity on the ledger
    and the event, the tier as any send projects it. The server tool's per-use charge leaves the reservation unknown
    though a token tariff is at hand; the event carries the reply's own normalized price and finality, never a
    second tariff read, and no secret."""
    import ouroboros.tools.search as search
    from ouroboros.llm_routing import OPENROUTER_BASE_URL, OPENROUTER_METADATA_HEADERS
    from ouroboros.openrouter_attribution import OPENROUTER_APP_HEADERS
    from ouroboros.provider_models import CONNECTIONS_KEY

    root, _client, _sent = transport
    monkeypatch.setenv(CONNECTIONS_KEY, json.dumps([{"kind": "openrouter", "name": "byok", "api_key": "row-secret-key",
                                                     "provider_routing": {"only": ["vendor-a"]}}]))
    monkeypatch.setenv("OUROBOROS_OR_PROVIDER", json.dumps({"data_collection": "deny"}))
    monkeypatch.setattr(ua, "estimate_cost_optional", lambda *a, **kw: 9.0)  # what a token bound would read
    monkeypatch.setattr(pricing, "estimate_cost_optional", lambda *a, **kw: pytest.fail("the event re-priced the reply"))
    resolved = []
    real_target = search._search_target
    monkeypatch.setattr(search, "_search_target", lambda *a: resolved.append(a) or real_target(*a))
    sent, clients = _search_client_double(monkeypatch, {"prompt_tokens": 7, "completion_tokens": 3, **reply})
    ctx = SimpleNamespace(pending_events=[], task_metadata={})
    search._web_search_openrouter(ctx, "query", connection="byok", processing_preference=preference)
    assert resolved == [("openrouter", "byok", "openai/gpt-5.5")]
    [client], [candidate] = clients, sent
    assert (client["api_key"], client["base_url"]) == ("row-secret-key", OPENROUTER_BASE_URL)
    assert client["default_headers"] == {**OPENROUTER_APP_HEADERS, **OPENROUTER_METADATA_HEADERS}
    assert candidate["extra_body"]["provider"] == {"data_collection": "deny", "only": ["vendor-a"]}
    assert candidate.get("service_tier") == ("default" if preference else None)
    [row] = ledger_rows(root)
    assert (row["provider"], row["model"]) == ("openrouter", "openrouter@byok::openai/gpt-5.5")
    assert (row["reservation_upper_bound_usd"], row["reservation_basis"]) == (None, "opaque_unknown")
    assert row["unpriced_add_on"] and not row.get("provider_reported_cost_only")
    assert (row["cost_usd"], row["cost_final"]) == shown
    # An inherited preference is no selection: the row then carries no processing keys at all (``usage_accounting``).
    assert ("submitted_processing_mode" in row) is bool(preference)
    assert row.get("submitted_processing_mode", "") == ("default" if preference else "")
    [event] = ctx.pending_events
    assert (event["model"], event["cost"], event["cost_final"]) == ("openrouter@byok::openai/gpt-5.5", *shown)
    assert "row-secret-key" not in json.dumps(event, default=str) and "row-secret-key" not in str(row)
    assert event["usage"]["cost"] == event["cost"]
    from supervisor.events_budget import _handle_llm_usage

    supervisor = SimpleNamespace(DRIVE_ROOT=root, bridge=SimpleNamespace(push_log=lambda row: None))
    _handle_llm_usage(event, supervisor)
    persisted = json.loads((root / "logs" / "events.jsonl").read_text().splitlines()[-1])
    assert (persisted["cost"], persisted["cost_final"]) == shown
    assert persisted["cost_known"] is (shown[0] is not None)
    assert persisted["cost_invalid"] is (reply.get("cost") == "invalid")
    assert not persisted["cost_estimated"]


@pytest.mark.parametrize("source,expected", [("web_search", None), ("loop", 0.125)])
def test_persisting_an_older_search_event_preserves_its_explicit_unknown(transport, source, expected):
    from supervisor.events_budget import _handle_llm_usage

    root, _, _ = transport
    supervisor = SimpleNamespace(DRIVE_ROOT=root, bridge=SimpleNamespace(push_log=lambda row: None))
    _handle_llm_usage({"source": source, "provider": "openrouter", "usage": {"is_byok": True, "cost": 0.125},
                       "cost": None, "cost_final": False}, supervisor)
    row = json.loads((root / "logs" / "events.jsonl").read_text().splitlines()[-1])
    assert row["cost"] == expected and row["cost_known"] is (expected is not None)
    assert not row["cost_final"] and not row["cost_invalid"]


@pytest.mark.parametrize("server_search", [False, True])
def test_chat_estimates_tokens_only_when_they_cover_the_call(transport, monkeypatch, server_search):
    root, client, sent = transport
    monkeypatch.setenv("OUROBOROS_MAIN_WEB_SEARCH", "openrouter")
    monkeypatch.setattr(pricing, "estimate_cost_optional", lambda *a, **k: 0.2)
    monkeypatch.setattr(ua, "estimate_cost_optional", lambda *a, **k: 0.2)
    body = {"choices": [{"message": {"role": "assistant", "content": "answer"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 1}}
    if server_search:
        body["usage"].update(is_byok=True, cost=0.125)  # Gateway fee only, upstream charge missing.

    def create(**candidate):
        sent.append(candidate)
        return SimpleNamespace(model_dump=lambda: body)

    monkeypatch.setattr(client, "_get_remote_client", lambda target: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    _, usage = client.chat([{"role": "user", "content": "query"}], "openrouter::fixture-model",
        tools=[{"type": "function", "function": {"name": "read", "parameters": {"type": "object", "properties": {}}}}],
        allow_server_web_search=server_search)
    [candidate], [attempt] = sent, ledger_rows(root)
    assert any(tool["type"] == "openrouter:web_search" for tool in candidate["tools"]) is server_search
    cost, _, _, estimated = pricing.normalize_usage_cost(usage, model=attempt["model"], use_local=False)
    assert cost == attempt["cost_usd"] == (None if server_search else 0.2)
    assert estimated is (not server_search) and not usage["cost_final"]


def test_plain_chat_can_still_reask_a_router_verdict(transport, monkeypatch):
    root, client, sent = transport
    error = RuntimeError("router verdict")
    error.status_code = 502
    error.openrouter_metadata = {"attempts": [{"provider": "upstream", "status": 504}]}

    def create(**candidate):
        sent.append(candidate)
        if len(sent) == 1:
            raise error
        return SimpleNamespace(model_dump=lambda: {
            "choices": [{"message": {"role": "assistant", "content": "answer"}}], "usage": {"cost": 0}})

    monkeypatch.setattr(client, "_get_remote_client", lambda target: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    message, _ = client.chat([{"role": "user", "content": "hello"}], "openrouter::fixture-model")
    assert message["content"] == "answer" and len(sent) == len(ledger_rows(root)) == 2
    routing = sent[1]["extra_body"]["provider"]
    assert routing["only"] == ["upstream"] and routing["allow_fallbacks"] is False


def test_search_on_a_custom_openrouter_endpoint_is_sent_there_and_never_priced_by_the_public_tariff(
        transport, monkeypatch):
    """The OpenRouter kind names no endpoint of its own (``PROVIDER_KINDS``): every configured leg, a named one too,
    resolves the public base. The custom variant is a target resolved on a client's own base: its send goes there with
    that target's key and headers, and only its report prices it, an unknown amount staying unknown."""
    from ouroboros import llm
    from ouroboros.provider_models import PROVIDER_KINDS

    root, _client, _sent = transport
    assert [field for field, _flat, _secret in PROVIDER_KINDS["openrouter"].fields] == ["api_key"]
    base = "https://private.invalid/v1"
    target = LLMClient(api_key="custom-secret", base_url=base).resolve_remote_target("openai/gpt-5.5")
    for module in (ua, pricing):
        monkeypatch.setattr(module, "estimate_cost_optional", lambda *a, **kw: pytest.fail("the public tariff was read"))
    sent, clients = _search_client_double(monkeypatch, {"prompt_tokens": 7, "completion_tokens": 3})
    llm.openrouter_web_search_server_tool(api_key=target["api_key"], target=target, model="openai/gpt-5.5",
                                          query="query", search_context_size="low")
    [client], [candidate] = clients, sent
    assert (client["base_url"], client["api_key"], client["default_headers"]) == (
        base, "custom-secret", target["default_headers"])
    assert candidate["tools"][0]["type"] == "openrouter:web_search" and "service_tier" not in candidate
    [row] = ledger_rows(root)
    assert row["provider_reported_cost_only"] is True and row["reservation_upper_bound_usd"] is None
    assert (row["cost_usd"], row["cost_final"]) == (None, False)


@pytest.mark.parametrize("provider,preference,field,native", [
    ("openai", "economy", "service_tier", "flex"),
    ("anthropic", "fast", "speed", "fast"),
])
def test_explicit_native_option_is_not_downgraded_by_preference_fallback(transport, provider, preference, field, native):
    from ouroboros.llm_attempt import ProcessingNotStarted, _finalized_physical_candidate, _attempt_request
    from ouroboros.request_wire_contract import physical_candidate_sha256
    from ouroboros.request_wire_recovery import plan_next_wire_retry, request_wire_call_scope

    _root, client, _sent = transport
    target = {**client._resolve_remote_target(f"{provider}::same-model"), "processing_preference": preference}
    payload = {"model": "same-model", "messages": [{"role": "user", "content": "input"}], field: native}
    apply_processing_preference(target, payload)
    assert target["processing_native_origin"] == "native_override"
    with request_wire_call_scope():
        candidate = _finalized_physical_candidate(target, payload, "messages" if provider == "anthropic" else "chat.completions")
        request = _attempt_request(target, candidate)
        error = ProcessingNotStarted(RuntimeError("typed no-start fixture"), reason="capacity")
        error.physical_attempt_capture = SimpleNamespace(state="settled", candidate_raw_sha256=physical_candidate_sha256(candidate))
        assert plan_next_wire_retry(candidate, error=error, target=target) is None
        assert request.submitted_processing_mode == native


@pytest.mark.parametrize("model", ["openai/other-model", "openai/same-model"])
def test_foreign_capture_never_supplies_submitted_mode_to_display(transport, model):
    _root, client, _sent = transport
    ua.adopt_physical_attempt_capture(ua.PhysicalAttemptCapture(
        attempt_id="foreign", model=model, provider="openai", state="settled",
        candidate_measurement_kind="canonical_json_v1", processing_preference="economy", submitted_processing_mode="flex"))
    target = {**client._resolve_remote_target("openai::same-model"), "processing_preference": "fast"}
    _, usage = client._normalize_remote_response({"choices": [{"message": {"content": "ok"}}],
        "usage": {"cost": 0}, "service_tier": "default"}, target)
    assert usage["processing"]["requested"] == "fast"
    assert usage["processing"]["submittedNative"] is None
    assert usage["processing"]["observed"] == "standard"


def test_second_key_without_an_echo_has_its_own_unknown_receipt(transport, monkeypatch):
    root, client, sent = transport

    def client_for(target):
        def create(**candidate):
            sent.append(candidate)
            body = {"choices": [{"message": {"role": "assistant", "content": "ok"}}], "usage": {"cost": 0}}
            if target["api_key"] == "first-key":
                body["service_tier"] = "priority"
            return SimpleNamespace(model_dump=lambda: body)
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    monkeypatch.setattr(client, "_get_remote_client", client_for)
    monkeypatch.setenv("OPENAI_API_SURFACE", "chat_completions")
    keys = ("first-key", "second-key")
    receipts = []
    for key in keys:
        monkeypatch.setenv("OPENAI_API_KEY", key)
        _, usage = client.chat([{"role": "user", "content": "same request"}], "openai::same-model",
                               processing_preference="fast")
        receipts.append(usage["processing"])
    assert [item["service_tier"] for item in sent] == ["priority", "priority"]
    rows = ledger_rows(root)
    assert [row["processing"] for row in rows] == receipts
    assert [r["observed"] for r in receipts] == ["fast", "unknown"]
    assert receipts[1]["observedNative"] == []
    assert [row["provider_receipt_binding"]["credential_sha256"] for row in rows] == [
        hashlib.sha256(key.encode()).hexdigest() for key in keys]
