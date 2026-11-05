"""Exact reasoning duplication is removed only from the OpenRouter send copy."""

import copy
import json

import pytest

from ouroboros.llm import LLMClient
from tests.test_llm_provider_golden import _observe
from tests._cases import chosen_rows


MODEL = "unlisted/reasoner"
CALL = {"id": "read-1", "type": "function",
        "function": {"name": "read_file", "arguments": '{"path":"source.py"}'}}


def _text(text, **metadata):
    return {"type": "reasoning.text", "text": text, **metadata}


def _history(carriers):
    return [
        {"role": "system", "content": "Read the complete source."},
        {"role": "user", "content": "Review this change."},
        {"role": "assistant", "content": "Reading the source.", "tool_calls": [CALL],
         "response_id": "response-1", "provider_metadata": {"future": [False, None, 7]},
         **carriers},
        {"role": "tool", "tool_call_id": CALL["id"], "content": "Exact source bytes."},
    ]


def _build(history, model=MODEL):
    client = LLMClient(api_key="fixture-not-live")
    target = client._resolve_remote_target(model)
    payload = client._build_remote_kwargs(
        target, history, "high", 1024, "auto", None, None, skip_capability_fetch=True,
    )
    client._normalize_payload_cache_ttl(target, payload)
    return payload


@pytest.mark.parametrize("parts", [
    ["one exact thought"],
    ["\n Préserver e\u0301", " 🐍\t", "tail\r\n"],
    ["\n \t"],
])
def test_complete_ordered_text_duplicate_only_leaves_the_send_copy(parts):
    details = [_text(part, index=index, id=f"part-{index}", format="provider-v1",
                     signature="opaque==\n" * 3000,
                     future_metadata={"nested": [None, False, {"text": "unchanged"}]})
               for index, part in enumerate(parts)]
    history = _history({"reasoning": "".join(parts), "reasoning_details": details,
                        "reasoning_content": "direct-only echo"})
    canonical = json.dumps(history, ensure_ascii=False)
    expected = copy.deepcopy(history)
    expected[2].pop("reasoning")
    expected[2].pop("reasoning_content")

    payload = _build(history)

    assert json.dumps(history, ensure_ascii=False) == canonical
    assert payload["messages"] == expected
    assert json.dumps(payload["messages"][2]["reasoning_details"], ensure_ascii=False) == json.dumps(
        details, ensure_ascii=False,
    )
    assert payload["messages"][2]["reasoning_details"] is not details
    assert payload["extra_body"]["provider"]["allow_fallbacks"] is False


@pytest.mark.parametrize("carriers", chosen_rows([
    pytest.param({"reasoning": "alpha"}, id="flat-only"),
    pytest.param({"reasoning_details": [_text("alpha")]}, id="details-only"),
    pytest.param({"reasoning": "", "reasoning_details": [_text("")]}, id="empty-flat"),
    pytest.param({"reasoning": "alpha", "reasoning_details": []}, id="empty-list"),
    pytest.param({"reasoning": "alpha", "reasoning_details": None}, id="null-details"),
    pytest.param({"reasoning": "alpha", "reasoning_details": _text("alpha")}, id="object-details"),
    pytest.param({"reasoning": "alpha", "reasoning_details": "alpha"}, id="string-details"),
    pytest.param({"reasoning": "alpha", "reasoning_details": [_text("beta")]}, id="different"),
    pytest.param({"reasoning": "alpha beta", "reasoning_details": [_text("alpha")]}, id="partial"),
    pytest.param({"reasoning": "alpha", "reasoning_details": [_text(" alpha ")]}, id="no-trimming"),
    pytest.param({"reasoning": "alpha\nbeta", "reasoning_details": [_text("alpha"), _text("beta")]},
                 id="no-added-separator"),
    pytest.param({"reasoning": "alphabeta", "reasoning_details": [_text("beta"), _text("alpha")]},
                 id="ordered"),
    pytest.param({"reasoning": "alpha", "reasoning_details": [_text("alpha"), _text("alpha")]},
                 id="no-skipping-duplicate-part"),
    pytest.param({"reasoning": "alpha", "reasoning_details": [
        {"type": "reasoning.summary", "text": "alpha", "summary": "alpha"}]}, id="summary"),
    pytest.param({"reasoning": "alpha", "reasoning_details": [
        {"type": "reasoning.encrypted", "data": "opaque=="}]}, id="encrypted-only"),
    pytest.param({"reasoning": "alpha", "reasoning_details": [
        _text("alpha"), {"type": "reasoning.encrypted", "data": "opaque=="}]}, id="mixed-encrypted"),
    pytest.param({"reasoning": "alpha", "reasoning_details": [
        _text("alpha"), {"type": "reasoning.summary", "summary": ""}]}, id="mixed-summary"),
    pytest.param({"reasoning": "alpha", "reasoning_details": [
        _text("alpha"), {"type": "reasoning.future", "text": ""}]}, id="unknown"),
    pytest.param({"reasoning": "alpha", "reasoning_details": [
        {"type": " reasoning.text ", "text": "alpha"}]}, id="no-type-normalization"),
    pytest.param({"reasoning": "alpha", "reasoning_details": [_text("alpha"), None]}, id="null-part"),
    pytest.param({"reasoning": "alpha", "reasoning_details": [_text("alpha"), ""]}, id="string-part"),
    pytest.param({"reasoning": "alpha", "reasoning_details": [
        _text("alpha"), {"type": "reasoning.text"}]}, id="missing-text"),
    pytest.param({"reasoning": "alpha", "reasoning_details": [
        _text("alpha"), {"text": ""}]}, id="missing-type"),
    *[pytest.param({"reasoning": "alpha", "reasoning_details": [_text("alpha"), _text(value)]},
                   id=f"nonstring-part-{type(value).__name__}") for value in [None, 0, False, [], {}]],
    *[pytest.param({"reasoning": value, "reasoning_details": [_text("alpha")]},
                   id=f"nonstring-flat-{type(value).__name__}") for value in [None, 0, False, [], {}]],
], 0, 19, 32))
def test_non_equivalent_empty_mixed_and_unknown_forms_stay_exact(carriers):
    history = _history({"reasoning_content": "", **carriers})
    before = copy.deepcopy(history)
    expected = copy.deepcopy(history)
    expected[2].pop("reasoning_content")

    payload = _build(history)

    assert payload["messages"] == expected
    assert history == before


@pytest.mark.parametrize("role", ["user", "tool"])
def test_non_assistant_matching_reasoning_fields_stay_exact(role):
    history = _history({})
    index = next(i for i, message in enumerate(history) if message["role"] == role)
    history[index].update({"reasoning": "exact", "reasoning_details": [_text("exact")],
                           "reasoning_content": "direct-only echo"})
    before = copy.deepcopy(history)
    expected = copy.deepcopy(history)
    expected[index].pop("reasoning_content")

    assert _build(history)["messages"] == expected
    assert history == before


@pytest.mark.parametrize("model, retained", [
    ("minimax::MiniMax-M3.1", {"reasoning_content", "reasoning_details"}),
    ("openai::gpt-5.6-terra", set()),
    ("zai::glm-5", set()),
    ("cloudru::zai-org/GLM-4.7", set()),
    ("openai-compatible::local-model", set()),
])
def test_direct_provider_carriers_keep_their_existing_projection(model, retained):
    carriers = {"reasoning": "exact", "reasoning_content": "direct echo",
                "reasoning_details": [_text("exact", signature="opaque==")]}
    history = _history(carriers)
    before = copy.deepcopy(history)

    payload = _build(history, model)

    assert {key: payload["messages"][2][key] for key in carriers
            if key in payload["messages"][2]} == {key: carriers[key] for key in retained}
    assert history == before


@pytest.mark.parametrize(("no_proxy", "method"), [(False, "chat"), (True, "chat_async")])
def test_sync_async_physical_send_preserves_details_and_seals_deduplicated_payload(method, no_proxy):
    details = [_text("exact", signature="opaque==", future={"parts": [2, 1]})]
    history = _history({"reasoning": "exact", "reasoning_details": details})
    observed = _observe({
        "env": {"OPENROUTER_API_KEY": "or-fixture-key"},
        "transport": [{"kind": "response", "body": {
            "id": "gen-replay", "choices": [{"index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": "done"}}],
            "usage": {"prompt_tokens": 40, "completion_tokens": 4, "cost": 0.001},
        }}],
        "call": {"kind": "method", "name": method, "kwargs": {
            "model": MODEL, "messages": history, "no_proxy": no_proxy,
        }},
    })

    assert "raised" not in observed, observed.get("raised")
    assert len(observed["sends"]) == len(observed["physical_attempts"]) == 1
    send, attempt = observed["sends"][0], observed["physical_attempts"][0]
    assert "reasoning" not in send["payload"]["messages"][2]
    assert send["payload"]["messages"][2]["reasoning_details"] == details
    assert send["payload_sha256"] == attempt["candidate_raw_sha256"]
    assert observed["returned"]["usage"]["request_wire"]["candidate_sha256"] == send["payload_sha256"]
