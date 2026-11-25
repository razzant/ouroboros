"""Direct OpenAI Responses: the connection-selected API under the shared client.

Home of ``ouroboros/openai_responses.py`` and ``ouroboros/openai_responses_custody.py``: the request projection,
the reply reading and its usability facts, exact item custody and replay, the stream assembler, and the
public ``LLMClient.chat`` call sites that send through them.
"""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

import ouroboros.llm_fallback as llm_fallback
from ouroboros._usage_response import UnusableReplyError, reply_usability, usage_reply_usability
from ouroboros.anthropic_native_custody import (
    context_custody_proxy,
    observability_custody_projection,
    physical_custody_projection,
    public_custody_projection,
    scrub_native_custody,
)
from ouroboros.llm import LLMClient
from ouroboros.llm_stream import IncompleteProviderStream, ProviderStreamError, ResponsesAccumulator
from ouroboros.openai_responses import (
    ResponsesProjectionError,
    completion_view,
    input_items,
    responses_payload,
)
from ouroboros.openai_responses_custody import (
    REPLAYED_TARGET_KEY,
    RESPONSES_CONSUMED_KEY,
    RESPONSES_RECEIPT_KEY,
    responses_tool_unit_active,
)
from ouroboros.request_wire_contract import canonical_sha256
from ouroboros.usage_accounting import PhysicalAttemptCapture

MODEL = "gpt-6.1-terra"


def _target(**overrides):
    return {"provider": "openai", "resolved_model": MODEL, "usage_model": f"openai/{MODEL}",
            "base_url": "https://api.openai.com/v1", "default_headers": {}, "api_surface": "responses",
            "connection": "", "supports_openrouter_extensions": False, "supports_generation_cost": False,
            **overrides}


def _tool(name="read_file", **function):
    return {"type": "function", "function": {"name": name, "description": "Read one file.",
                                             "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                                             **function}}


def _message(text, phase=None, status="completed", ident="msg_1"):
    item = {"id": ident, "type": "message", "role": "assistant", "status": status,
            "content": [{"type": "output_text", "text": text, "annotations": []}]}
    return {**item, **({"phase": phase} if phase else {})}


def _call(call_id="call_1", name="read_file", arguments='{"path":"a"}', status="completed"):
    return {"id": f"fc_{call_id}", "type": "function_call", "call_id": call_id, "name": name,
            "arguments": arguments, "status": status}


REASONING = {"id": "rs_1", "type": "reasoning", "summary": [], "encrypted_content": "OPAQUE-ENCRYPTED"}
REFUSAL = {"id": "msg_r", "type": "message", "role": "assistant", "status": "completed",
           "content": [{"type": "refusal", "refusal": "I can't help with that request."}]}
USAGE = {"input_tokens": 1000, "input_tokens_details": {"cached_tokens": 600, "cache_write_tokens": 0},
         "output_tokens": 50, "output_tokens_details": {"reasoning_tokens": 30}, "total_tokens": 1050}


def _body(*items, status="completed", reason=None, usage=USAGE, **extra):
    body = {"id": "resp_1", "object": "response", "status": status, "model": MODEL, "output": list(items),
            "usage": copy.deepcopy(usage) if usage is not None else None, **extra}
    if status == "incomplete":
        body["incomplete_details"] = {"reason": reason} if reason else {}
    return body


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def model_dump(self):
        return copy.deepcopy(self.payload)


def _install(monkeypatch, client, replies, target=None):
    """The real chat path with the provider's ``responses.create`` and the attempt executor stubbed."""
    captured, queue, holder = [], list(replies), {"capture": None}

    def create(**kwargs):
        captured.append(copy.deepcopy(kwargs))
        item = queue.pop(0)
        if isinstance(item, BaseException):
            raise item
        return _Response(item)

    remote = SimpleNamespace(responses=SimpleNamespace(create=create),
                             chat=SimpleNamespace(completions=SimpleNamespace(create=None)))
    monkeypatch.setattr(client, "_resolve_remote_target", lambda _model: dict(target or _target()))
    monkeypatch.setattr(client, "_get_remote_client", lambda _target: remote)
    monkeypatch.setattr(client, "_make_no_proxy_client", lambda *_a, **_k: (remote, SimpleNamespace(close=lambda: None)))

    def execute(request, send, before_dispatch=None):
        response = send()
        attempt_id = f"attempt-{len(captured)}"
        holder["capture"] = PhysicalAttemptCapture(
            attempt_id=attempt_id, model=request.model, provider=request.provider, state="settled",
            candidate_measurement_kind="canonical_json_v1", candidate_raw_sha256=request.candidate_raw_sha256,
            candidate_manifest_ref={"path": f"physical/{attempt_id}.json", "call_id": attempt_id,
                                    "sha256": canonical_sha256(attempt_id)})
        return response

    monkeypatch.setattr(llm_fallback, "_execute_candidate", execute)
    monkeypatch.setattr(llm_fallback, "last_physical_attempt_capture", lambda: holder["capture"])
    return captured


# --- request projection --------------------------------------------------------------------------------

def test_payload_projects_roles_media_tools_and_options_without_inventing_any():
    messages = [
        {"role": "system", "content": "Governance.", "_host_context_kind": "stable"},
        {"role": "user", "content": [{"type": "text", "text": "look", "cache_control": {"type": "ephemeral"}},
                                     {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA",
                                                                         "detail": "high"}}]},
        {"role": "assistant", "content": "Reading.", "reasoning_details": [{"type": "reasoning.encrypted"}],
         "tool_calls": [{"id": "x" * 80, "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "x" * 80, "content": [{"type": "text", "text": "body"},
                                                               {"type": "image", "source": {"type": "base64",
                                                                "media_type": "image/jpeg", "data": "QQ=="}}]},
        {"role": "user", "content": "clock line"},
    ]
    payload = responses_payload(_target(), messages, effort="xhigh", max_tokens=4096, tool_choice={
        "type": "function", "function": {"name": "read_file"}}, temperature=None,
        tools=[_tool(), _tool("strict_one", strict=True)], response_format={
            "type": "json_schema", "json_schema": {"name": "verdict", "schema": {"type": "object"}, "strict": True}},
        cache_identity="ouroboros-key", stream=True)
    system, user, assistant, call, output, clock = payload["input"]
    assert system == {"role": "system", "content": "Governance."}
    assert user["content"] == [{"type": "input_text", "text": "look"},
                               {"type": "input_image", "image_url": "data:image/png;base64,AAAA", "detail": "high"}]
    assert assistant == {"role": "assistant", "content": "Reading."}  # no phase or item id is invented
    assert call["type"] == "function_call" and call["call_id"] == output["call_id"] and len(call["call_id"]) <= 64
    assert output["output"][1] == {"type": "input_image", "image_url": "data:image/jpeg;base64,QQ=="}
    assert clock == {"role": "user", "content": "clock line"}  # host rows keep their role
    assert [tool["strict"] for tool in payload["tools"]] == [False, True]  # omitted is false, authored stays
    assert payload["tool_choice"] == {"type": "function", "name": "read_file"}
    assert payload["text"] == {"format": {"type": "json_schema", "name": "verdict", "schema": {"type": "object"},
                                          "strict": True}}
    assert (payload["store"], payload["max_output_tokens"], payload["reasoning"], payload["stream"],
            payload["prompt_cache_key"]) == (False, 4096, {"effort": "xhigh"}, True, "ouroboros-key")
    assert "previous_response_id" not in payload and "stream_options" not in payload
    assert "reasoning_details" not in json.dumps(payload)


def test_an_input_branch_without_a_responses_form_is_refused_never_dropped():
    with pytest.raises(ResponsesProjectionError):
        input_items(_target(), [{"role": "user", "content": [{"type": "input_audio", "data": "..."}]}])
    with pytest.raises(ResponsesProjectionError):
        input_items(_target(), [{"role": "user", "content": [{"type": "image_url", "image_url": {}}]}])


# --- reply reading and usability -----------------------------------------------------------------------

def _normalize(body, target=None):
    return LLMClient(api_key="k")._normalize_remote_response(body, target or _target(), skip_cost_fetch=True)


def test_completed_call_reply_keeps_exact_usage_and_unknown_cost_unknown():
    message, usage = _normalize(_body(REASONING, _message("Reading.", "commentary"), _call()))
    assert message["content"] == "Reading."
    assert message["tool_calls"] == [{"id": "call_1", "type": "function",
                                      "function": {"name": "read_file", "arguments": '{"path":"a"}'}}]
    assert (usage["prompt_tokens"], usage["completion_tokens"], usage["cached_tokens"]) == (1000, 50, 600)
    assert usage["output_tokens_details"] == {"reasoning_tokens": 30}  # output already includes reasoning
    assert "cache_write_tokens" not in usage  # the wire's nested zero stays in details
    assert usage["response_finish_reason"] == "tool_calls" and usage["response_status"] == "completed"
    assert usage["cost"] is None and usage["cost_final"] is False  # no tariff offline: unknown, never zero
    assert usage_reply_usability(usage, message) == "usable"


@pytest.mark.parametrize("field", ["cache_write_tokens", "cache_creation_tokens", "cache_creation_input_tokens", None])
def test_next_round_distinguishes_a_reported_zero_write_from_absence(field, monkeypatch):
    from ouroboros import pricing
    from ouroboros.loop_llm_call import _record_round_cache_facts
    from ouroboros.transcript_prefix import previous_cache_fact

    inputs = []
    monkeypatch.setattr(pricing, "estimate_cost_optional", lambda *args, cache_usage, **kw: inputs.append(cache_usage))
    details = {"cached_tokens": 0}
    if field is not None:
        details[field] = 0
    _, usage = _normalize(_body(_message("Done."), usage={**USAGE, "input_tokens_details": details}))
    assert "cache_write_tokens" not in usage
    assert inputs == [{"cached_tokens": 0, "cache_write_tokens": 0 if field else None,
                       "cache_write_tokens_by_ttl": None, "prompt_cache_ttl": None}]
    history = {}
    _record_round_cache_facts(history, usage, round_idx=1, model=MODEL, provider="openai")
    expected = "write unknown" if field is None else "write 0 tokens"
    assert expected in previous_cache_fact(history)


@pytest.mark.parametrize("charge", [0, 0.0123], ids=["reported-zero", "reported-charge"])
def test_a_reported_response_charge_reaches_the_ledger_and_the_loop_alike(charge, monkeypatch):
    """A Response's own ``total_cost_usd`` is the settled amount in both lanes; a catalog estimate never replaces it."""
    from ouroboros import pricing
    from ouroboros._usage_response import usage_from_response

    monkeypatch.setattr(pricing, "estimate_cost_optional", lambda *args, **kwargs: 0.99)
    body = _body(_message("Done."), total_cost_usd=charge)
    _facts, ledger_cost, ledger_final = usage_from_response(_Response(body))
    _message_, usage = _normalize(body)
    loop_cost, _model, _provider, estimated = pricing.normalize_usage_cost(usage, model=MODEL, use_local=False,
                                                                           provider="openai")
    assert ledger_cost == usage["cost"] == loop_cost == charge
    assert ledger_final is usage["cost_final"] is True and estimated is False


def test_absent_usage_stays_absent_never_zero():
    _message_, usage = _normalize(_body(_message("Done."), usage=None))
    assert "prompt_tokens" not in usage and "completion_tokens" not in usage and usage["cost"] is None


def test_completed_commentary_is_healthy_intermediate_progress():
    message, usage = _normalize(_body(REASONING, _message("I will look first.", "commentary")))
    assert usage["response_finish_reason"] == "commentary"
    assert usage_reply_usability(usage, message) == "intermediate"
    final_message, final_usage = _normalize(_body(_message("Answer.", "final_answer")))
    assert final_usage["response_finish_reason"] == "stop"
    assert usage_reply_usability(final_usage, final_message) == "usable"


@pytest.mark.parametrize(("items", "finish"), [
    ((REASONING, REFUSAL), "refusal"),
    ((REASONING, _message("Let me check first.", "commentary"), REFUSAL), "refusal"),  # never intermediate progress
    ((REASONING, REFUSAL, _call()), "tool_calls"),  # complete calls keep their own finish and run
])
def test_a_completed_refusal_ends_as_the_providers_refusal_in_its_own_words(items, finish):
    message, usage = _normalize(_body(*items))
    assert message["refusal"] == "I can't help with that request." and usage["response_finish_reason"] == finish
    assert usage_reply_usability(usage, message) == ("unusable" if finish == "refusal" else "usable")


@pytest.mark.parametrize("reason", ["content_filter", "max_messages", "steered", None])
def test_unsuitable_incomplete_terminals_are_no_result_and_never_executable(reason):
    message, usage = _normalize(_body(_message('{"verdict": "PASS"}', status="incomplete"),
                                      _call(), status="incomplete", reason=reason))
    assert "tool_calls" not in message  # a call of a reply that did not complete never runs
    assert usage["response_status"] == "incomplete" and usage["response_finish_reason"] == reason
    assert usage_reply_usability(usage, message) == "unusable"
    assert [item["type"] for item in usage["response_withheld_output"]] == ["function_call"]
    assert usage["prompt_tokens"] == 1000  # spend is kept whatever the usability
    assert message[RESPONSES_RECEIPT_KEY]["status"] == "incomplete"  # original items retained, never replayed


def test_output_limit_incomplete_keeps_the_genuine_cut_semantics():
    message, usage = _normalize(_body(_message("partial", status="incomplete"), status="incomplete",
                                      reason="max_output_tokens"))
    assert usage["response_finish_reason"] == "max_output_tokens"
    assert usage_reply_usability(usage, message) == "usable"  # the existing cut recovery reads it


def test_failed_reply_is_a_provider_body_error_with_its_output_retained():
    message, usage = _normalize(_body(_message("half an answer", status="incomplete"), status="failed",
                                      error={"code": "server_error", "message": "The server had an error."}))
    assert message["content"] == "" and "response_finish_reason" not in usage
    assert usage["provider_error"]["code"] == 500 and usage["response_status"] == "failed"
    assert usage["response_withheld_output"][0]["text"] == "half an answer"
    assert reply_usability(None, "failed", message) == "unusable"


def test_unknown_items_and_server_transformations_are_disclosed():
    transformations = [{"id": "rs_0", "type": "reasoning", "action": "drop", "reason": "prefix_mismatch"}]
    _message_, usage = _normalize(_body({"id": "ws_1", "type": "web_search_call"}, _message("Done."),
                                        input_transformations=transformations, reasoning={"context": "all_turns"}))
    assert usage["response_unexpected_items"] == ["web_search_call"]
    assert usage["response_input_transformations"] == transformations
    assert usage["response_reasoning_context"] == "all_turns"


def test_provider_usage_cannot_spoof_host_terminal_facts():
    completion, _facts = completion_view(_body(_message("Done.")))
    completion["usage"]["response_status"] = "incomplete"
    message, usage = LLMClient(api_key="k")._normalize_remote_response(
        {"choices": completion["choices"], "usage": completion["usage"]},
        {**_target(), "api_surface": "chat_completions"}, skip_cost_fetch=True)
    assert "response_status" not in usage and usage_reply_usability(usage, message) == "usable"


# --- custody -------------------------------------------------------------------------------------------

def _assistant(message):
    return {"role": "assistant", **{key: value for key, value in message.items() if key != "role"}}


def test_a_completed_turn_replays_whole_on_its_exact_route_only():
    message, _usage = _normalize(_body(REASONING, _message("Reading.", "commentary"), _call()))
    history = [{"role": "user", "content": "go"}, _assistant(message),
               {"role": "tool", "tool_call_id": "call_1", "content": "body"}]
    target = _target()
    items = input_items(target, history)
    assert items[1:4] == [REASONING, _message("Reading.", "commentary"), _call()]
    assert items[4] == {"type": "function_call_output", "call_id": "call_1", "output": "body"}
    assert len(target[REPLAYED_TARGET_KEY]) == 1
    for other in (_target(resolved_model="gpt-6.1-sol"), _target(connection="work"),
                  _target(base_url="https://proxy.example/v1")):
        assert input_items(other, history)[1] == {"role": "assistant", "content": "Reading."}


def test_an_edited_or_partly_answered_turn_projects_canonically_without_item_ids():
    message, _usage = _normalize(_body(REASONING, _message("Reading.", "commentary"), _call()))
    edited = {**_assistant(message), "content": "Reading (folded)."}
    assert input_items(_target(), [edited, {"role": "tool", "tool_call_id": "call_1", "content": "b"}])[0] == {
        "role": "assistant", "content": "Reading (folded)."}
    unanswered = input_items(_target(), [_assistant(message)])
    assert all("id" not in item for item in unanswered)
    assert unanswered[1]["type"] == "function_call" and "encrypted_content" not in json.dumps(unanswered)


def test_receipts_never_reach_another_route_or_a_public_view():
    message, _usage = _normalize(_body(REASONING, _message("Answer.", "final_answer")))
    row = _assistant(message)
    assert RESPONSES_RECEIPT_KEY not in scrub_native_custody([row])[0]
    chat_copy = LLMClient._copy_messages_with_cache_policy([row], allow_message_cache_control=False,
                                                           flatten_tool_content_blocks=True)
    assert RESPONSES_RECEIPT_KEY not in chat_copy[0]
    switched = LLMClient.sanitize_reasoning_on_model_switch([row], "openai::gpt-6.1-terra", "openai::gpt-6.1-sol")
    assert RESPONSES_RECEIPT_KEY not in switched[0]
    assert RESPONSES_RECEIPT_KEY not in public_custody_projection(row)
    projected = observability_custody_projection(row)["_openai_responses_custody"]
    assert projected["item_types"] == ["reasoning", "message"] and "OPAQUE" not in json.dumps(projected)
    assert context_custody_proxy(row)["_openai_responses_opaque_bytes"]  # the replayed bytes are still sized
    sealed = physical_custody_projection({"model": MODEL, "input": [REASONING, _message("Answer.")]})
    assert "OPAQUE-ENCRYPTED" not in json.dumps(sealed) and sealed["input"][0]["id"] == "rs_1"


def test_a_replayed_turn_stays_active_until_an_answered_request_consumed_it():
    first, _usage = _normalize(_body(REASONING, _message("Reading.", "commentary"), _call()))
    history = [_assistant(first), {"role": "tool", "tool_call_id": "call_1", "content": "b"}]
    assert responses_tool_unit_active(history, 0, 1)
    target = _target()
    input_items(target, history)
    answer, _usage = _normalize(_body(_message("Done.", "final_answer", ident="msg_2")), target)
    assert answer[RESPONSES_CONSUMED_KEY] == target[REPLAYED_TARGET_KEY]
    assert not responses_tool_unit_active([*history, _assistant(answer)], 0, 1)


# --- stream ----------------------------------------------------------------------------------------------

def _feed(*events):
    accumulator = ResponsesAccumulator()
    for event in events:
        accumulator.accept(event["type"], json.dumps(event))
    return accumulator


def test_stream_settles_only_on_a_terminal_response_event():
    created = {"type": "response.created", "response": {"id": "resp_1", "status": "in_progress", "output": []}}
    done = {"type": "response.output_item.done", "output_index": 0, "item": _message("hi")}
    with pytest.raises(IncompleteProviderStream):
        _feed(created, {"type": "response.output_text.delta", "delta": "hi"}, done).result()
    body = _feed(created, done, {"type": "response.completed", "response": {
        "id": "resp_1", "status": "completed", "output": [], "usage": USAGE}}).result()
    assert body["output"] == [_message("hi")] and body["status"] == "completed"
    with pytest.raises(ProviderStreamError):
        _feed(created, {"type": "error", "code": "server_error", "message": "boom"})


# --- public call sites ---------------------------------------------------------------------------------

def test_tool_less_one_shot_call_gets_typed_non_success_before_parsing(monkeypatch):
    client = LLMClient(api_key="k")
    _install(monkeypatch, client, [
        _body(_message('{"safe": true}', "commentary")),
        _body(_message('{"safe": true}', status="incomplete"), status="incomplete", reason="content_filter"),
        _body(REFUSAL),
        _body(_message('{"safe": true}', "final_answer")),
    ])
    for usability in ("intermediate", "unusable", "unusable"):
        with pytest.raises(UnusableReplyError) as raised:
            client.chat([{"role": "user", "content": "judge"}], "openai::gpt-6.1-terra")
        assert raised.value.usability == usability and raised.value.usage["prompt_tokens"] == 1000
    assert raised.value.reply_message["refusal"] == "I can't help with that request."  # the refusal's own words
    message, _usage = client.chat([{"role": "user", "content": "judge"}], "openai::gpt-6.1-terra")
    assert json.loads(message["content"]) == {"safe": True}


def test_an_output_cut_is_no_one_shot_result_but_stays_main_s_to_recover(monkeypatch):
    """A tool-less caller has no output-cut recovery: a reply its provider left incomplete at the limit is refused
    typed even when its text is valid JSON. A tool loop gets the same reply back, usable, for its own recovery."""
    client = LLMClient(api_key="k")
    cut = _body(_message('{"safe": true}', status="incomplete"), status="incomplete", reason="max_output_tokens")
    _install(monkeypatch, client, [cut, cut])
    with pytest.raises(UnusableReplyError) as raised:
        client.chat([{"role": "user", "content": "judge"}], "openai::gpt-6.1-terra")
    assert raised.value.usability == "unfinished" and raised.value.usage["response_finish_reason"] == "max_output_tokens"
    message, usage = client.chat([{"role": "user", "content": "go"}], "openai::gpt-6.1-terra", tools=[_tool()])
    assert usage_reply_usability(usage, message) == "usable" and usage["response_finish_reason"] == "max_output_tokens"


def test_the_real_client_replays_a_turn_exactly_on_its_next_request(monkeypatch):
    """Two ``LLMClient.chat`` turns, not ``input_items`` alone: the second physical body carries the first turn's
    reasoning, phase and call id in order before the tool result; an owner follow-up after the answer replays
    both turns; the same history on another model of the connection projects canonically, no opaque item."""
    client = LLMClient(api_key="k")
    second_reasoning = {**REASONING, "id": "rs_2", "encrypted_content": "OPAQUE-2"}
    captured = _install(monkeypatch, client, [
        _body(REASONING, _message("Reading.", "commentary"), _call()),
        _body(second_reasoning, _message("Done.", "final_answer", ident="msg_2")),
        _body(_message("More.", "final_answer", ident="msg_3"))])
    history = [{"role": "user", "content": "go"}]
    first, _usage = client.chat(history, "openai::gpt-6.1-terra", tools=[_tool()])
    history += [dict(first, role="assistant"), {"role": "tool", "tool_call_id": "call_1", "content": "file body"}]
    answer, _usage = client.chat(history, "openai::gpt-6.1-terra", tools=[_tool()])
    assert captured[1]["input"][1:5] == [REASONING, _message("Reading.", "commentary"), _call(),
                                         {"type": "function_call_output", "call_id": "call_1", "output": "file body"}]
    assert answer[RESPONSES_CONSUMED_KEY]
    history += [dict(answer, role="assistant"), {"role": "user", "content": "and now?"}]
    client.chat(history, "openai::gpt-6.1-terra", tools=[_tool()])
    assert [(item.get("type") or item.get("role"), item.get("phase")) for item in captured[2]["input"][-3:]] == [
        ("reasoning", None), ("message", "final_answer"), ("user", None)]
    other = _install(monkeypatch, client, [_body(_message("Sol.", "final_answer", ident="msg_4"))],
                     target=_target(resolved_model="gpt-6.1-sol", usage_model="openai/gpt-6.1-sol"))
    client.chat(history, "openai::gpt-6.1-sol", tools=[_tool()])
    assert "OPAQUE" not in json.dumps(other[0]["input"]) and all("id" not in item for item in other[0]["input"])


class _Wire:
    status_code = 200

    def __init__(self, raw):
        self.raw, self.headers = raw, {}

    def iter_bytes(self):
        for offset in range(0, len(self.raw), 7):  # splits across frames and UTF-8/JSON boundaries
            yield self.raw[offset:offset + 7]

    def close(self):
        pass


def _sse(*events):
    return b"".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode() for event in events)


_CREATED = {"type": "response.created", "response": {"id": "resp_1", "status": "in_progress", "output": []}}


@pytest.mark.parametrize(("events", "outcome"), [
    ([_CREATED, {"type": "response.output_item.done", "output_index": 0, "item": _call()},
      {"type": "response.completed", "response": _body(_call())}], "tool_calls"),
    ([_CREATED, {"type": "response.output_text.delta", "delta": "Rea"}], IncompleteProviderStream),
    ([_CREATED, {"type": "error", "code": "server_error", "message": "boom"}], ProviderStreamError),
    ([_CREATED, {"type": "response.failed", "response": _body(status="failed", error={"code": "server_error",
                                                                                       "message": "boom"})}], "failed"),
    ([_CREATED, {"type": "response.incomplete", "response": _body(_call(), status="incomplete", reason="content_filter")}],
     "content_filter"),
])
def test_a_streamed_reply_settles_only_on_its_typed_terminal_through_the_real_client(monkeypatch, events, outcome):
    """One physical send each: EOF without a terminal stays an unknown outcome, a provider ``error`` event its own
    stream error, a failed terminal a provider body error, an incomplete one a reply whose calls never run."""
    client = LLMClient(api_key="k")
    captured = _install(monkeypatch, client, [])
    remote = SimpleNamespace(responses=SimpleNamespace(create=lambda **kwargs: (
        captured.append(kwargs), SimpleNamespace(response=_Wire(_sse(*events))))[1]))
    monkeypatch.setattr(client, "_get_remote_client", lambda _target: remote)
    monkeypatch.setattr(client, "_make_no_proxy_client", lambda *_a, **_k: (remote, SimpleNamespace(close=lambda: None)))
    if isinstance(outcome, type):
        with pytest.raises(outcome):
            client.chat([{"role": "user", "content": "go"}], "openai::gpt-6.1-terra", tools=[_tool()], stream=True)
    else:
        message, usage = client.chat([{"role": "user", "content": "go"}], "openai::gpt-6.1-terra", tools=[_tool()],
                                     stream=True)
        assert usage.get("response_finish_reason") == outcome or usage["response_status"] == outcome
        assert bool(message.get("tool_calls")) is (outcome == "tool_calls")
    assert len(captured) == 1 and captured[0]["stream"] is True


def test_vision_query_call_site_refuses_an_intermediate_reply(monkeypatch):
    client = LLMClient(api_key="k")
    _install(monkeypatch, client, [_body(_message("Let me zoom in first.", "commentary"))])
    monkeypatch.setattr("ouroboros.vision_routing.prepare_messages_for_send", lambda messages, **_k: messages)
    with pytest.raises(UnusableReplyError):
        client.vision_query("what is shown?", [], model="openai::gpt-6.1-terra")


def test_tool_loop_call_returns_commentary_for_its_owner_to_continue(monkeypatch):
    client = LLMClient(api_key="k")
    captured = _install(monkeypatch, client, [_body(REASONING, _message("Checking.", "commentary"))])
    message, usage = client.chat([{"role": "user", "content": "go"}], "openai::gpt-6.1-terra", tools=[_tool()])
    assert usage_reply_usability(usage, message) == "intermediate"
    assert captured[0]["store"] is False and captured[0]["tools"][0]["strict"] is False


def test_effort_value_refusal_repairs_on_the_same_route_without_stripping(monkeypatch):
    class _Refusal(Exception):
        status_code = 400
        code = "unsupported_value"
        param = "reasoning.effort"

        def __init__(self):
            message = ("Unsupported value: 'reasoning.effort' does not support 'xhigh' with this model. "
                       "Supported values are: 'low', 'medium', and 'high'.")
            super().__init__(message)
            self.body = {"error": {"code": self.code, "param": self.param, "message": message,
                                   "type": "invalid_request_error"}}

    client = LLMClient(api_key="k")
    captured = _install(monkeypatch, client, [_Refusal(), _body(_message("Answer.", "final_answer"))])
    message, usage = client.chat([{"role": "user", "content": "go"}], "openai::gpt-6.1-terra", tools=[_tool()],
                                 reasoning_effort="xhigh")
    assert [request["reasoning"] for request in captured] == [{"effort": "xhigh"}, {"effort": "high"}]
    assert captured[0]["input"] == captured[1]["input"]  # nothing replayed was stripped
    assert usage["request_wire"]["requested_effort"] == "xhigh" and usage["request_wire"]["applied_effort"] == "high"
    assert message["content"] == "Answer."


def test_an_invalid_api_surface_refuses_before_any_send():
    from ouroboros.llm_attempt import ConnectionUnavailable
    from ouroboros.llm_routing import resolve_target

    with pytest.raises(ConnectionUnavailable) as refused:
        resolve_target("openai::gpt-6.1-terra", {"OPENAI_API_KEY": "sk-test", "OPENAI_API_SURFACE": "auto"})
    assert refused.value.reason == "invalid_api_surface"
    assert resolve_target("openai::gpt-5.6-terra", {"OPENAI_API_KEY": "sk-test"})["api_surface"] == "chat_completions"
    named = resolve_target("openai@work::gpt-6.1-terra", {"OUROBOROS_CONNECTIONS": [
        {"kind": "openai", "name": "work", "api_key": "sk-w", "api_surface": "Responses"}]})
    assert named["api_surface"] == "responses"


@pytest.mark.parametrize(("surface", "reply", "ok"), [
    ("responses", {"status": "completed", "output": [_call(name="provider_test", arguments='{"ok":true}')]}, True),
    ("responses", {"status": "completed", "output": [_message("OK")]}, False),
    ("chat_completions", {"choices": [{"finish_reason": "tool_calls", "message": {"role": "assistant", "content": None,
     "tool_calls": [{"id": "c", "type": "custom", "custom": {"name": "provider_test", "input": '{"ok":true}'}}]}}]},
     True),
    ("chat_completions", {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "OK"}}]}, False),
])
def test_provider_test_exercises_a_function_call_through_the_selected_api(monkeypatch, surface, reply, ok):
    from ouroboros import llm_probe

    sent = []

    class _Remote:
        def with_options(self, **_kwargs):
            return self

        def __getattr__(self, name):
            endpoint = SimpleNamespace(create=lambda **payload: (sent.append((name, payload)), reply)[1])
            return SimpleNamespace(create=endpoint.create, completions=endpoint)

    monkeypatch.setattr(LLMClient, "_new_remote_client", staticmethod(lambda _target: _Remote()))
    monkeypatch.setattr(llm_probe, "accounted_one_shot", lambda _target, candidate, send, **_k: send(candidate))
    result = LLMClient().probe_provider_readiness(
        "openai::gpt-6.1-terra", settings={"OPENAI_API_KEY": "sk-test", "OPENAI_API_SURFACE": surface})
    assert result["ok"] is ok and (ok or result["error"] == "No function call returned")
    [(endpoint, payload)] = sent
    assert endpoint == ("responses" if surface == "responses" else "chat")
    assert payload["tool_choice"] == "required" and payload["tools"]  # never a text-only "Reply OK"
    budget = payload.get("max_output_tokens") or payload.get("max_completion_tokens")
    assert budget >= 1024


def test_public_async_call_sends_responses_and_keeps_the_one_shot_rule(tmp_path, monkeypatch):
    """The async lane takes the same API and reading: a real attempt through the usage ledger, the exact items
    retained, and a tool-less intermediate reply refused before any caller parses it."""
    import asyncio

    from ouroboros.usage_accounting import UsageScope, usage_scope

    client = LLMClient(api_key="k")
    captured, replies = [], [_body(REASONING, _message("Reading.", "commentary"), _call()),
                             _body(_message("Plan first.", "commentary"))]

    async def create(**kwargs):
        captured.append(kwargs)
        return _Response(replies.pop(0))

    remote = SimpleNamespace(responses=SimpleNamespace(create=create))
    monkeypatch.setattr(client, "_resolve_remote_target", lambda _model: _target())
    monkeypatch.setattr(client, "_get_async_remote_client", lambda _target: remote)
    monkeypatch.setattr(client, "_make_no_proxy_async_client", lambda *_a, **_k: (remote, None))
    with usage_scope(UsageScope(drive_root=tmp_path, task_id="async-responses")):
        message, usage = asyncio.run(client.chat_async([{"role": "user", "content": "go"}], "openai::gpt-6.1-terra",
                                                       tools=[_tool()]))
        with pytest.raises(UnusableReplyError):
            asyncio.run(client.chat_async([{"role": "user", "content": "go"}], "openai::gpt-6.1-terra"))
    assert captured[0]["store"] is False and "messages" not in captured[0]
    assert message["tool_calls"][0]["id"] == "call_1" and RESPONSES_RECEIPT_KEY in message
    assert len(usage["ledger_attempt_ids"]) == 1 and usage["response_status"] == "completed"
