"""Native search payloads reuse the ordinary physical-send recovery owner."""

import copy
import hashlib
import sys
from types import SimpleNamespace

import pytest

from ouroboros import llm, pricing, usage_accounting as ua
from ouroboros.llm_attempt import _canonical_candidate_bytes
from tests.test_usage_accounting import data_root as _usage_data_root
from tests._usage_store_testing import ledger_rows

data_root = _usage_data_root


class FastCapacityRefusal(RuntimeError):
    status_code = 429
    body = {"error": {"type": "rate_limit_error", "message": "Fast rate limit"}}


class Response:
    def __init__(self, candidate):
        self.candidate = candidate

    def model_dump(self):
        if "speed" in self.candidate:
            return {"content": [{"type": "text", "text": "answer"}], "usage": {
                "input_tokens": 10, "output_tokens": 1, "speed": self.candidate["speed"], "cost": 0.005}}
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
    state = SimpleNamespace(sent=[], clients=[], error=None)

    def create(candidate):
        state.sent.append(copy.deepcopy(candidate))
        if state.error is not None and len(state.sent) == 1:
            raise state.error
        return Response(candidate)

    def factory(**kwargs):
        state.clients.append(kwargs)

        def messages_create(*, model, messages, max_tokens, tools, extra_body=None, timeout=None):
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
    assert "fast-mode-2026-02-01" in state.clients[0]["default_headers"]["anthropic-beta"]
    rows = ledger(root)
    finals = list({row["attempt_id"]: row for row in rows}.values())
    assert [row["state"] for row in finals] == ["released", "settled"]
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
    assert [(row["task_id"], row["state"]) for row in finals] == [("helper", "released"), ("sibling", "settled")]


@pytest.mark.parametrize("provider", ["anthropic", "openrouter"])
@pytest.mark.parametrize("incomplete", [False, True])
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
