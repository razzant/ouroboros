"""Native search payloads reuse the ordinary physical-send recovery owner."""

import copy
import hashlib
import json
import sys
from types import SimpleNamespace

import pytest

from ouroboros import llm, pricing, usage_accounting as ua
from ouroboros.llm_attempt import _canonical_candidate_bytes
from ouroboros.tools import search as search_module
from tests.test_web_search import (
    _FakeStream, _make_completed_event, _make_event,
    ctx as _search_ctx, mock_openai as _search_mock_openai, patch_env as _search_patch_env,
)
from tests.test_usage_accounting import data_root as _usage_data_root
from tests._usage_store_testing import ledger_rows

data_root = _usage_data_root
ctx, mock_openai, patch_env = _search_ctx, _search_mock_openai, _search_patch_env  # the search module's fixtures


def test_search_ultrafast_uses_responses_even_with_chat_configured(ctx, patch_env, mock_openai, monkeypatch):
    monkeypatch.setenv("OPENAI_API_SURFACE", "chat_completions")
    done = _make_completed_event()
    body = done.response.model_dump()
    done.response.model_dump = lambda: {**body, "service_tier": "priority"}
    mock_openai.responses.create.return_value = _FakeStream([
        _make_event("response.output_text.delta", delta="answer"), done])
    result = json.loads(search_module._web_search_openai(ctx, "query", model="same", _processing_preference="ultrafast"))
    assert result["answer"] == "answer"
    mock_openai.responses.create.assert_called_once()
    assert mock_openai.responses.create.call_args.kwargs["service_tier"] == "ultrafast"
    receipt = ctx.pending_events[0]["usage"]["processing"]
    assert (receipt["submittedNative"], receipt["observedNative"], receipt["reason"]) == (
        "ultrafast", ["priority"], "provider_mode_changed")


def test_named_responses_search_binds_key_model_receipt_and_event(data_root, ctx, patch_env, mock_openai, monkeypatch):
    from ouroboros.provider_models import CONNECTIONS_KEY

    ctx.task_metadata["budget_drive_root"] = str(data_root)
    monkeypatch.setenv(CONNECTIONS_KEY, json.dumps([{"kind": "openai", "name": "work", "api_key": "selected-key"}]))
    monkeypatch.setenv("OPENAI_BASE_URL", "https://builtin.invalid/v1")
    monkeypatch.setattr(ua, "estimate_cost_optional", lambda *a, **k: None)
    mock_openai.responses.create.return_value = _FakeStream([
        _make_event("response.output_text.delta", delta="answer"), _make_completed_event()])
    result = json.loads(search_module._web_search(ctx, "query", model="openai@work::same"))
    [row], [event] = ledger(data_root), ctx.pending_events
    sdk = sys.modules["openai"].OpenAI.call_args.kwargs
    assert (sdk["api_key"], sdk["base_url"]) == ("selected-key", "https://api.openai.com/v1")
    assert result["model"] == mock_openai.responses.create.call_args.kwargs["model"] == "same"
    assert row["model"] == event["model"] == "openai@work::same"
    assert row["provider_receipt_binding"]["credential_sha256"] == hashlib.sha256(b"selected-key").hexdigest()
    assert event["ledger_attempt_ids"] == [row["attempt_id"]]


def test_search_stream_read_error_with_status_does_not_repeat_generation(ctx, patch_env, mock_openai):
    def partial():
        yield _make_event("response.output_text.delta", delta="partial")
        error = RuntimeError("connection closed while reading")
        error.status_code = 503
        raise error

    mock_openai.responses.create.return_value = partial()
    result = json.loads(search_module._web_search_openai(ctx, "query", model="same", _processing_preference="ultrafast"))
    assert result["reason_code"] == "provider_outcome_unknown"
    mock_openai.responses.create.assert_called_once()


class FastCapacityRefusal(RuntimeError):
    status_code = 429
    body = {"error": {"type": "rate_limit_error", "message": "Fast rate limit"}}


class Response:
    def __init__(self, candidate):
        self.candidate = candidate
        self.content = [{"type": "text", "text": "answer"}]

    @property
    def usage(self):
        return self.model_dump()["usage"]

    def model_dump(self):
        if any(tool["type"] == "web_search_20250305" for tool in self.candidate.get("tools", [])):
            usage = {"input_tokens": 10, "output_tokens": 1, "cost": 0.005}
            if "speed" in self.candidate:
                usage["speed"] = self.candidate["speed"]
            return {"content": self.content, "usage": usage}
        return {"choices": [{"message": {"role": "assistant", "content": "answer"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 1, "cost": 0.005},
                "service_tier": self.candidate.get("service_tier")}


def ledger(root):
    return ledger_rows(root)


@pytest.fixture
def helpers(data_root, monkeypatch):
    monkeypatch.setenv("OUROBOROS_CONTEXT_MODE", "low")
    monkeypatch.setenv("OUROBOROS_PROCESSING_PREFERENCE", "fast")
    monkeypatch.setenv("OUROBOROS_MODEL_PROCESSING_PREFERENCES", "{}")
    monkeypatch.setattr(pricing, "estimate_cost_optional", lambda *args, **kwargs: None)
    monkeypatch.setattr(ua, "estimate_cost_optional", lambda *args, **kwargs: None)
    state = SimpleNamespace(sent=[], clients=[], headers=[], error=None, body=None, sdk_cost=None)

    def create(candidate):
        state.sent.append(copy.deepcopy(candidate))
        if state.error is not None and len(state.sent) == 1:
            raise state.error
        if state.body is not None and len(state.sent) == 1:
            return SimpleNamespace(model_dump=lambda: state.body, total_cost_usd=state.sdk_cost)
        return Response(candidate)

    def factory(**kwargs):
        state.clients.append(kwargs)

        def messages_create(*, model, messages, max_tokens, tools, extra_body=None, extra_headers=None, timeout=None):
            state.headers.append(dict(extra_headers or {}))
            return create(dict(model=model, messages=messages, max_tokens=max_tokens, tools=tools,
                               **(extra_body or {})))

        def chat_create(**candidate):
            candidate.pop("timeout", None)
            return create(candidate)

        return SimpleNamespace(messages=SimpleNamespace(create=messages_create),
                               chat=SimpleNamespace(completions=SimpleNamespace(create=chat_create)))

    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(Anthropic=factory))
    monkeypatch.setattr("ouroboros.net_transport.web_search_openai_client", factory)
    with ua.usage_scope(ua.UsageScope(drive_root=data_root, task_id="helper", root_task_id="helper")):
        yield data_root, state


def test_anthropic_search_refusal_reprices_same_native_body_as_standard(helpers, monkeypatch):
    root, state = helpers
    state.error = FastCapacityRefusal("Fast capacity exhausted")
    priced = []

    def reservation(request):
        priced.append((request.submitted_processing_mode, request.candidate_raw_sha256))
        return 0.01 if request.submitted_processing_mode == "fast" else 0.04

    monkeypatch.setattr(ua, "_reservation_cost", reservation)
    response = llm.anthropic_web_search_server_tool(api_key="test", model="model", query="query")
    first, second = state.sent
    assert second == {**first, "speed": "standard"}
    assert first == {"model": "model", "messages": [{"role": "user", "content": "query"}],
                     "max_tokens": 2048, "speed": "fast", "tools": [
                         {"type": "web_search_20250305", "name": "web_search", "max_uses": 5}]}
    assert state.headers == [{"anthropic-beta": "fast-mode-2026-02-01"}, {}]
    rows = ledger(root)
    finals = list({row["attempt_id"]: row for row in rows}.values())
    assert [row["state"] for row in finals] == ["settled", "settled"]
    assert (finals[0]["settle_reason"], finals[0]["cost_usd"], finals[0]["cost_final"]) == ("processing_not_started", None, False)
    # each attempt's current row keeps the reservation bound it was admitted with
    assert [row["reservation_upper_bound_usd"] for row in finals] == [0.01, 0.04]
    assert [mode for mode, _hash in priced] == ["fast", "standard"]
    for candidate, row in zip(state.sent, finals):
        assert row["candidate_raw_sha256"] == hashlib.sha256(_canonical_candidate_bytes(candidate)).hexdigest()
        assert row["processing_preference"] == "fast" and row["source"] == "web_search.anthropic"
    assert finals[0]["candidate_raw_sha256"] != finals[1]["candidate_raw_sha256"]
    assert finals[1]["processing"]["requested"] == "fast"
    assert finals[1]["processing"]["observed"] == "standard"
    assert response.model_dump()["usage"]["speed"] == "standard"
    assert state.clients[0]["max_retries"] == 0


def _auto_search_from(monkeypatch, provider):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    if provider == "openrouter":
        monkeypatch.setenv("OPENROUTER_API_KEY", "router-key")
    else:
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "messages-key")
    monkeypatch.setenv("OUROBOROS_WEBSEARCH_BACKEND", "auto")
    monkeypatch.setenv("OUROBOROS_WEBSEARCH_MODEL", "")
    monkeypatch.setattr("ouroboros.search_routes._package_discoverable", lambda name: True)
    monkeypatch.setattr(search_module, "_web_search_ddgs", lambda *a, **k: pytest.fail("unexpected DDGS search"))


@pytest.mark.parametrize("provider,returned_body", [
    ("openrouter", False), ("openrouter", True), ("anthropic", False),
])
def test_charged_opaque_failure_stops_at_the_first_sdk_call(helpers, ctx, monkeypatch, provider, returned_body):
    root, state = helpers
    _auto_search_from(monkeypatch, provider)
    metadata = {"attempts": [{"provider": "upstream", "status": 504}]}
    body = {"error": {"code": 502, "message": "gateway failure"}, "usage": {"cost": 0.125},
            "openrouter_metadata": metadata}
    if returned_body:
        state.body = body
    else:
        error = RuntimeError("gateway failure")
        error.status_code, error.body, error.openrouter_metadata = 502, body, metadata
        state.error = error

    result = json.loads(search_module._web_search(ctx, "synthetic query"))
    assert len(state.sent) == 1
    [row], [leg] = ledger(root), result["legs_tried"]
    assert leg["source"] == provider and leg["ledger_attempt_ids"] == [row["attempt_id"]]
    assert result["reason_code"] == leg["outcome"] == "provider_outcome_unknown"
    assert (row["state"], row["cost_usd"], row["cost_final"]) == ("settled", 0.125, True)


@pytest.mark.parametrize("provider,delivery,usage,upstream,expected_cost", [
    ("anthropic", "raised", {"cost": 0.125}, None, 0.125),
    ("openrouter", "raised", {"cost": 0.125}, None, 0.125),
    ("openrouter", "returned", {"cost": 0.125}, None, 0.125),
    ("openrouter", "sdk_price", {}, None, 0.125),
    ("openrouter", "raised", {"is_byok": True, "cost": 0.125}, None, None),
    ("openrouter", "returned", {"is_byok": True, "cost": 0.125}, None, None),
    # A zero beside work or an upstream attempt is not a proved free call: the amount stays unknown and non-final.
    ("openrouter", "returned", {"cost": 0}, 504, None),
    ("openrouter", "returned", {"cost": 0}, "raw_timeout", None),
    ("openrouter", "raised", {"cost": 0}, 502, None),
    ("openrouter", "raised", {"cost": 0}, "dispatched", None),
    ("anthropic", "raised", {"cost": 0, "server_tool_use": {"web_search_requests": 1}}, None, None),
])
def test_typed_refusal_with_charge_work_or_upstream_attempt_stops_auto_search(
        helpers, ctx, monkeypatch, provider, delivery, usage, upstream, expected_cost):
    root, state = helpers
    _auto_search_from(monkeypatch, provider)
    if provider == "openrouter":
        monkeypatch.setenv("OUROBOROS_MODEL_PROCESSING_PREFERENCES", '{"websearch":"economy"}')
    native = {"status_code": 429, "code": "resource_unavailable", "type": "rate_limit_error"}
    body = {"error": native, "usage": usage}
    if delivery != "raised":
        if delivery == "sdk_price":
            state.sdk_cost = expected_cost
        if upstream == "raw_timeout":
            native["metadata"] = {"provider_name": "upstream", "raw": 'data: {"error":{"code":504}}'}
        elif upstream:
            body["openrouter_metadata"] = {"attempts": [{"status": upstream, "provider": "upstream"}]}
        state.body = body
    else:
        error = RuntimeError("rate limit")
        error.status_code = 429
        # The error can be bare or wrapped in .body while usage lives on the received response.
        error.body = {"error": native} if provider == "anthropic" else native
        error.response = SimpleNamespace(json=lambda: body)
        if upstream:
            error.__cause__ = RuntimeError("upstream send failed")
            if upstream == "dispatched":
                error.__cause__.physical_attempt_capture = ua.PhysicalAttemptCapture(
                    attempt_id="upstream", model="fixture", provider=provider, state="dispatched",
                    candidate_measurement_kind="opaque")
            else:
                error.__cause__.response = SimpleNamespace(status_code=upstream)
        state.error = error

    result = json.loads(search_module._web_search(ctx, "query"))
    [sent], [row], [leg] = state.sent, ledger(root), result["legs_tried"]
    assert sent["service_tier" if provider == "openrouter" else "speed"] == ("flex" if provider == "openrouter" else "fast")
    assert leg["source"] == provider and leg["ledger_attempt_ids"] == [row["attempt_id"]]
    assert result["reason_code"] == leg["outcome"] == "provider_outcome_unknown"
    assert row["state"] == ("settled" if delivery != "raised" or expected_cost is not None else "unresolved")
    assert (row.get("cost_usd"), bool(row.get("cost_final"))) == (expected_cost, expected_cost is not None)
    if delivery != "raised":
        # A received reply is accounted once and shown once: one event, bound to that attempt, carrying its row's
        # amount and finality (the reply's own SDK price included), never a second charge or an estimate.
        [event] = ctx.pending_events
        assert event["ledger_attempt_ids"] == [row["attempt_id"]]
        assert (event["cost"], event["cost_final"]) == (expected_cost, expected_cost is not None)
        assert (event["cost"], event["cost_final"]) == (row.get("cost_usd"), bool(row.get("cost_final")))


@pytest.mark.parametrize("metadata", [{}, {"provider_name": "upstream", "raw": 'data: {"error":{"type":"rate_limit_error"}}'}])
def test_uncharged_router_body_refusal_can_reach_the_next_search_source(helpers, ctx, monkeypatch, metadata):
    root, state = helpers
    _auto_search_from(monkeypatch, "openrouter")
    state.body = {"error": {"code": 429, "type": "rate_limit_error", "metadata": metadata}, "usage": {"cost": 0},
                  "openrouter_metadata": {"attempt": 0, "attempts": []}}
    result = json.loads(search_module._web_search(ctx, "query"))
    assert result["answer"] == "answer" and len(state.sent) == 2
    first, second = ledger(root)
    assert [(leg["source"], leg["ledger_attempt_ids"]) for leg in result["legs_tried"]] == [
        ("openrouter", [first["attempt_id"]]), ("anthropic", [second["attempt_id"]])]
    assert (first["state"], first["cost_usd"], first["cost_final"]) == ("settled", 0, True)


def test_openrouter_search_can_replace_a_free_flex_request_once(helpers):
    root, state = helpers
    error = RuntimeError("Flex unavailable")
    error.status_code, error.body = 429, {"error": {"code": "resource_unavailable"}, "usage": {"cost": 0}}
    state.error = error
    response = llm.openrouter_web_search_server_tool(api_key="fixture", model="fixture", query="query",
        search_context_size="low", processing_preference="economy")
    first, second = state.sent
    assert first["service_tier"] == "flex" and second == {**first, "service_tier": "default"}
    assert response.model_dump()["service_tier"] == "default"
    first_row, second_row = ledger(root)
    assert (first_row["state"], first_row["settle_reason"], first_row["cost_usd"]) == ("settled", "processing_not_started", 0)
    assert first_row["cost_final"] is True
    assert second_row["state"] == "settled"


def test_a_tier_error_with_a_reported_charge_cannot_release_the_search(helpers):
    root, state = helpers
    error = RuntimeError("Flex unavailable")
    error.status_code = 429
    error.body = {"error": {"code": "resource_unavailable"}, "usage": {"cost": 0.125}}
    state.error = error
    with pytest.raises(RuntimeError) as failure:
        llm.openrouter_web_search_server_tool(api_key="fixture", model="fixture", query="query",
            search_context_size="low", processing_preference="economy")
    assert failure.value is error and len(state.sent) == 1
    [row] = ledger(root)
    assert (row["state"], row["cost_usd"], row["cost_final"]) == ("settled", 0.125, True)


@pytest.mark.parametrize("status,native,usage,cost", [
    (500, {"message": "provider failure"}, {"cost": 0.125}, 0.125),
    (429, {"code": "rate_limit_exceeded"}, {"cost": 0.125}, 0.125),
    (400, {"code": "unsupported_service_tier"}, {"total_cost": 0.25}, 0.25),
    pytest.param(429, {"code": "resource_unavailable", "usage": {"output_tokens": 1}}, {"cost": 0.375}, 0.375,
                 id="tier-refusal-inner-work-outer-charge"),
    # A raised failure that reports zero beside its own output has no proved amount: the bound stays open.
    pytest.param(429, {"code": "rate_limit_exceeded"}, {"cost": 0, "output_tokens": 2}, None, id="zero-with-output"),
])
def test_responses_create_error_settles_the_reported_cost_without_a_resend(
        data_root, ctx, patch_env, mock_openai, monkeypatch, status, native, usage, cost):
    ctx.task_metadata["budget_drive_root"] = str(data_root)
    monkeypatch.setenv("OPENROUTER_API_KEY", "fallback")
    monkeypatch.setenv("OUROBOROS_MODEL_PROCESSING_PREFERENCES", '{"websearch":"economy"}')
    monkeypatch.setattr(search_module, "_web_search_openrouter", lambda *a, **k: pytest.fail("search repeated"))
    error = RuntimeError("provider failure")
    error.status_code, error.body = status, native
    # The SDK can attach partial work to the inner error and keep the price only in the HTTP envelope.
    wire_error = {key: value for key, value in native.items() if key != "usage"}
    error.response = SimpleNamespace(json=lambda: {"error": wire_error, "usage": usage})
    mock_openai.responses.create.side_effect = error
    result = json.loads(search_module._web_search(ctx, "query"))
    mock_openai.responses.create.assert_called_once()
    [row], [leg] = ledger(data_root), result["legs_tried"]
    assert leg["ledger_attempt_ids"] == [row["attempt_id"]] and leg["outcome"] == "provider_outcome_unknown"
    assert (row["state"], row.get("cost_usd"), bool(row.get("cost_final"))) == (
        "settled" if cost is not None else "unresolved", cost, cost is not None)


@pytest.mark.parametrize("event_type,started,usage,state,cost,reason", [
    ("error", True, {}, "unresolved", None, "provider_outcome_unknown"),
    ("error", True, {"cost": 0.125}, "settled", 0.125, "provider_outcome_unknown"),
    ("sdk_error", False, {"cost": 0.125}, "settled", 0.125, "provider_outcome_unknown"),
    ("response.failed", True, {"cost": 0.125}, "settled", 0.125, "provider_response_failed"),
    ("response.incomplete", True, {"cost": 0.125}, "settled", 0.125, "provider_response_failed"),
    ("response.failed", False, {"cost": 0.125}, "settled", 0.125, "provider_response_failed"),
    ("failed_sdk_price", False, {}, "settled", 0.125, "provider_response_failed"),
    # The terminal failure was received, so the attempt is accounted; its zero beside output is not a final price.
    ("response.failed", False, {"cost": 0, "output_tokens": 1}, "settled", None, "provider_response_failed"),
])
def test_responses_stream_failure_keeps_its_own_evidence_and_settlement(
        data_root, ctx, patch_env, mock_openai, monkeypatch, event_type, started, usage, state, cost, reason):
    ctx.task_metadata["budget_drive_root"] = str(data_root)
    monkeypatch.setenv("OPENROUTER_API_KEY", "fallback")
    monkeypatch.setattr(search_module, "_web_search_openrouter", lambda *a, **k: pytest.fail("search already started"))
    native = {"status_code": 503, "type": "overloaded_error", "message": "failed"}

    def stream():
        if started:
            yield _make_event("response.web_search_call.searching")
        if event_type == "sdk_error":
            error = RuntimeError("stream failed")
            error.body, error.status_code = {"error": native, "usage": usage}, 503
            raise error
        fields = {"error": native, "usage": usage}
        if event_type == "failed_sdk_price":
            yield _make_event("response.failed", response=SimpleNamespace(model_dump=lambda: fields, total_cost_usd=cost))
        else:
            yield _make_event(event_type, **(fields if event_type == "error" else {"response": SimpleNamespace(**fields)}))

    mock_openai.responses.create.return_value = stream()
    result = json.loads(search_module._web_search(ctx, "query"))
    mock_openai.responses.create.assert_called_once()
    assert sys.modules["openai"].OpenAI.call_args.kwargs["max_retries"] == 0
    [row], [leg] = ledger(data_root), result["legs_tried"]
    assert leg["ledger_attempt_ids"] == [row["attempt_id"]] and result["reason_code"] == reason
    assert (row["state"], row.get("cost_usd"), bool(row.get("cost_final"))) == (state, cost, cost is not None)


@pytest.mark.parametrize("named", [True, False], ids=["named", "builtin"])
def test_anthropic_search_keeps_the_selected_key_with_its_messages_endpoint(helpers, ctx, monkeypatch, named):
    from ouroboros.provider_models import CONNECTIONS_KEY

    root, state = helpers
    monkeypatch.setenv("ANTHROPIC_API_KEY", "builtin-key")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://builtin.invalid/prefix")
    monkeypatch.setenv(CONNECTIONS_KEY, json.dumps([{"kind": "anthropic", "name": "work", "api_key": "selected-key"}]))
    selection = "anthropic@work::same" if named else "anthropic::same"
    result = json.loads(search_module._web_search(ctx, "query", model=selection))
    [sdk], [sent], [row], [event] = state.clients, state.sent, ledger(root), ctx.pending_events
    assert (sdk["api_key"], sdk["base_url"]) == (
        ("selected-key", "https://api.anthropic.com") if named else ("builtin-key", "https://builtin.invalid/prefix"))
    assert result["answer"] == "answer" and sent["model"] == "same"
    assert state.headers[0]["anthropic-version"] == "2023-06-01"
    assert row["model"] == event["model"] == (selection if named else "anthropic/same")
    assert event["ledger_attempt_ids"] == [row["attempt_id"]]
    if named:
        assert state.headers[0]["anthropic-beta"] == "fast-mode-2026-02-01"
        assert row["provider_receipt_binding"]["endpoint"] == "https://api.anthropic.com/v1"
        assert row["processing"] == event["usage"]["processing"]
        assert row["processing"]["observed"] == "fast"
    else:
        assert "speed" not in sent and event["usage"]["provider_reported_cost_only"]


def test_standard_search_gets_its_own_budget_admission(helpers, monkeypatch):
    """The standard retry's own reservation is admitted again on KNOWN spend (#1487):
    a sibling's final $0.02 that landed after the fast attempt refuses it."""
    root, state = helpers
    state.error = FastCapacityRefusal("Fast capacity exhausted")
    landed = []

    def price(request):
        if request.submitted_processing_mode == "fast":
            return 0.01
        if not landed:
            landed.append(True)
            with ua.usage_scope(ua.UsageScope(drive_root=root, task_id="sibling", root_task_id="helper")):
                held = ua.reserve_attempt(ua.AttemptRequest(model="m", provider="p", reservation_usd=0.02))
                ua.mark_dispatched(held)
                ua.settle_attempt(held, {}, cost_usd=0.02, cost_final=True)
        return 0.04

    monkeypatch.setattr(ua, "_reservation_cost", price)
    with ua.usage_scope(ua.UsageScope(drive_root=root, task_id="helper", root_task_id="helper", root_limit_usd=0.02)):
        with pytest.raises(ua.BudgetExceeded):
            llm.anthropic_web_search_server_tool(api_key="test", model="model", query="query")
    assert len(state.sent) == 1
    finals = {row["attempt_id"]: row for row in ledger(root)}.values()
    assert [(row["task_id"], row["state"]) for row in finals] == [("helper", "settled"), ("sibling", "settled")]


@pytest.mark.parametrize(("incomplete", "provider"), [(False, "anthropic"), (True, "openrouter")])
def test_unknown_search_attempt_never_resends(helpers, provider, incomplete):
    root, state = helpers
    state.error = TimeoutError("Connection ended after dispatch")
    if incomplete:
        state.error.stream_incomplete = True
    call = getattr(llm, f"{provider}_web_search_server_tool")
    kwargs = {"search_context_size": "low"} if provider == "openrouter" else {}
    with pytest.raises(TimeoutError):
        call(api_key="test", model="model", query="query", **kwargs)
    assert len(state.sent) == 1
    assert ledger(root)[-1]["state"] == "unresolved"


@pytest.mark.parametrize("provider,field,standard", [
    ("anthropic", "speed", "standard"), ("openrouter", "service_tier", "default"),
])
def test_public_helpers_preserve_explicit_standard_and_native_tool_payload(helpers, provider, field, standard):
    root, state = helpers
    call = getattr(llm, f"{provider}_web_search_server_tool")
    kwargs = {"search_context_size": "low"} if provider == "openrouter" else {}
    call(api_key="test", model="same-model", query="query", processing_preference="standard", **kwargs)
    assert len(state.sent) == 1
    assert state.sent[0][field] == standard
    assert state.sent[0]["model"] == "same-model"
    assert state.sent[0]["tools"][0]["type"] == (
        "web_search_20250305" if provider == "anthropic" else "openrouter:web_search")
    assert "thinking" not in state.sent[0] and "output_config" not in state.sent[0]
    assert ledger(root)[-1]["processing"]["requested"] == "standard"
    assert ledger(root)[-1]["source"] == f"web_search.{provider}"
