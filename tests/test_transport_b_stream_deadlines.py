"""B5/B6 physical-send contracts. All providers are in-memory or loopback fakes."""

from __future__ import annotations

import asyncio
import base64
import copy
import json
import pathlib
from datetime import datetime, timezone
from types import SimpleNamespace

import httpx
import pytest

from ouroboros import deadline_utils, model_wait, usage_accounting as ua
from ouroboros.llm import LLMClient
from ouroboros.llm_attempt import PhysicalDispatchInterrupted
from ouroboros.llm_stream import (
    AssembledResponse, IncompleteProviderStream, ProviderStreamError, RejectedProviderStream,
)
from ouroboros.loop_llm_call import classify_llm_exception
from tests._usage_store_testing import ledger_rows

WIRE_CORPUS = pathlib.Path(__file__).parent / "fixtures" / "llm_wire"


MESSAGES = [{"role": "user", "content": "preserve the complete answer"}]
TOOLS = [{"type": "function", "function": {"name": "lookup", "parameters": {
    "type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"],
    "additionalProperties": False}}}]


def target(provider="openrouter"):
    return {"provider": provider, "resolved_model": "vendor/test-stream", "usage_model": "vendor/test-stream",
            "base_url": "https://provider.invalid/v1", "supports_openrouter_extensions": provider == "openrouter",
            "supports_generation_cost": False, "api_key": "fixture"}


def payload(**extra):
    return {"model": "vendor/test-stream", "messages": copy.deepcopy(MESSAGES), "max_tokens": 1024,
            "reasoning_effort": "high", **extra}


def completion(**extra):
    return {"id": "gen-test", "object": "chat.completion", "model": "vendor/test-stream",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "done"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "cost": 0.25}, **extra}


def chunk(delta=None, finish=None, *, index=0, **extra):
    return {"id": "gen-test", "object": "chat.completion.chunk", "model": "vendor/test-stream",
            "choices": [{"index": index, "delta": delta or {}, "finish_reason": finish}], **extra}


def sse(*events, done=True):
    wire = b""
    for event in events:
        if isinstance(event, tuple):
            kind, body = event
            wire += f"event: {kind}\r\n".encode()
        else:
            body = event
        wire += ("data: " + json.dumps(body, ensure_ascii=False) + "\r\n\r\n").encode()
    return wire + (b"data: [DONE]\r\n\r\n" if done else b"")


class WireResponse:
    status_code = 200
    reason = "OK"
    url = "https://provider.invalid/v1/messages"

    def __init__(self, wire, *, failure=None, step=None, chunk_size=13):
        self.wire, self.failure, self.step, self.chunk_size = wire, failure, step, chunk_size
        self.headers = {"x-generation-id": "header-generation"}
        self.closed = False

    def iter_bytes(self):
        # Fragment through UTF-8, CRLF and JSON boundaries (13 bytes by default).
        for offset in range(0, len(self.wire), self.chunk_size):
            if self.step:
                self.step()
            yield self.wire[offset:offset + self.chunk_size]
        if self.failure:
            raise self.failure

    def iter_content(self, **kwargs):
        return self.iter_bytes()

    async def aiter_bytes(self):
        for item in self.iter_bytes():
            yield item

    def close(self):
        self.closed = True

    async def aclose(self):
        self.close()


class Rejected(RuntimeError):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status_code = status
        self.body = {"error": {"type": "invalid_request_error", "message": message}}


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    import ouroboros.request_wire_contract as wire
    import ouroboros.pricing as pricing

    monkeypatch.setattr(wire, "canonical_wire_evidence_root", lambda: tmp_path)
    monkeypatch.setattr(pricing, "estimate_cost_optional", lambda *a, **k: None)
    monkeypatch.setattr(ua, "_reservation_cost", lambda request: 1.0)
    monkeypatch.setattr(LLMClient, "_get_supported_parameters", lambda *a, **k: None)
    monkeypatch.setenv("TOTAL_BUDGET", "100")
    with ua.usage_scope(ua.UsageScope(drive_root=tmp_path, task_id="stream-task", root_task_id="stream-task")):
        yield tmp_path


def rows(root):
    return ledger_rows(root)


def run_driver(create, kwargs, route, *, asynchronous=False):
    client = LLMClient()
    if asynchronous:
        async def send(**values):
            return create(**values)
        return asyncio.run(client._create_chat_completion_with_retries_async(send, kwargs, route))
    return client._create_chat_completion_with_retries(create, kwargs, route)


@pytest.mark.parametrize("asynchronous", [False, True])
def test_json_and_stream_parity_with_usage_tail(isolated, asynchronous):
    wire = sse(chunk({"role": "assistant", "content": "do"}), chunk({"content": "ne"}, "stop"),
               chunk({"content": "", "role": "assistant"}, "stop", usage=completion()["usage"]))
    response = WireResponse(wire, step=lambda: assert_unsettled(isolated))
    result = run_driver(lambda **k: SimpleNamespace(response=response), payload(stream=True), target(),
                        asynchronous=asynchronous).model_dump()
    receipt = result.pop("_stream_receipt")
    assert result == completion()
    client = LLMClient()
    assert client._normalize_remote_response(result, target(), skip_cost_fetch=True) == client._normalize_remote_response(
        completion(), target(), skip_cost_fetch=True)
    assert receipt["complete"] is True and receipt["manifest_ref"]
    assert response.closed
    assert [row["state"] for row in rows(isolated)] == ["settled"]
    assert rows(isolated)[-1]["cost_usd"] == 0.25


def assert_unsettled(root):
    assert rows(root)[-1]["state"] == "dispatched"


def test_recorded_gemini_multicall_frames_through_the_driver(isolated):
    """Real OpenRouter frames (google/gemini-3.8-flash, two tool calls, high reasoning)
    through the physical driver at 13-byte fragmentation: every ``reasoning_details``
    delta carries ``index: 0``, so records are reassembled by type transition — the four
    ``reasoning.text`` deltas become ONE text record and the ``reasoning.encrypted``
    thought signature stays discrete — while ``tool_calls`` keep their stable indices.
    The wire ``index`` is preserved as recorded (round-trip accepted with any indices)."""
    wire = (WIRE_CORPUS / "openrouter" / "gemini-3.8-flash" / "multicall_stream.sse").read_bytes()
    response = WireResponse(wire, step=lambda: assert_unsettled(isolated))
    result = run_driver(lambda **k: response, payload(stream=True), target()).model_dump()
    receipt = result.pop("_stream_receipt")
    msg = result["choices"][0]["message"]
    assert [call["id"] for call in msg["tool_calls"]] == ["call_2216119", "call_2216120"]
    assert [json.loads(call["function"]["arguments"]) for call in msg["tool_calls"]] == [
        {"query": "ouroboros"}, {"query": "serpent"}]
    assert all(call["function"]["name"] == "lookup" and call["type"] == "function" for call in msg["tool_calls"])
    text, encrypted = msg["reasoning_details"]
    assert text["type"] == "reasoning.text" and text["format"] == "google-gemini-v1" and text["index"] == 0
    assert text["text"].startswith("**Exploring Etymological Roots**")
    assert text["text"].count("**") == 8  # four streamed deltas, one record
    assert msg["reasoning"] == text["text"]
    assert encrypted["type"] == "reasoning.encrypted" and encrypted["id"] == "call_2216119" and encrypted["index"] == 0
    assert encrypted["data"].startswith("AY89a19L6dC0Kq4QOSn6Z5Qz")
    assert result["choices"][0]["finish_reason"] == "tool_calls" and result["provider"] == "Google"
    assert result["usage"]["cost"] == 0.00358725 and result["usage"]["completion_tokens"] == 944
    assert receipt["complete"] is True and receipt["anomalies"] == {"count": 0, "first": []}
    assert response.closed
    assert [row["state"] for row in rows(isolated)] == ["settled"]
    assert rows(isolated)[-1]["cost_usd"] == 0.00358725
    normalized_msg, normalized_usage = LLMClient()._normalize_remote_response(result, target(), skip_cost_fetch=True)
    assert [detail["type"] for detail in normalized_msg["reasoning_details"]] == ["reasoning.text", "reasoning.encrypted"]
    assert normalized_usage["response_provider"] == "Google"


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("failure", ["eof", "missing_finish", "error_first", "error_mid", "error_free_claim",
                                     "disconnect", "cancel", "malformed"])
def test_incomplete_stream_never_settles_or_replays(isolated, asynchronous, failure):
    wire = sse(chunk({"content": "partial", "tool_calls": [{"index": 0, "id": "t", "type": "function",
                           "function": {"name": "lookup", "arguments": '{"q":"ok"}'}}]}), done=False)
    error = {"error": {"code": "server_error", "message": "stream_options unsupported; upstream lost"}}
    if failure == "error_free_claim":
        error = {"error": {"code": 404, "message": "No endpoints found; Error code: 404"}}
    exception = None
    if failure == "missing_finish":
        wire += b"data: [DONE]\n\n"
    elif failure.startswith("error"):
        wire = (wire if failure == "error_mid" else b"") + sse(error, done=False)
    elif failure == "disconnect":
        exception = httpx.ReadError("read lost")
    elif failure == "cancel":
        exception = asyncio.CancelledError()
    elif failure == "malformed":
        wire += b"data: {broken}\n\n"
    response = WireResponse(wire, failure=exception)
    calls = []
    def send(**kw):
        calls.append(kw)
        return response
    with pytest.raises(BaseException) as caught:
        run_driver(send, payload(stream=True), target(), asynchronous=asynchronous)
    exc = caught.value
    assert len(calls) == 1 and response.closed
    from ouroboros.transport_custody import _capture_on_chain, transport_exception_cause
    assert _capture_on_chain(exc).state == "unresolved"
    while not hasattr(exc, "stream_receipt") and transport_exception_cause(exc) is not None:
        exc = transport_exception_cause(exc)
    # missing_finish reached [DONE]: complete wire judged unusable (a verdict, not an
    # unknown outcome); without a usage frame its money stays unknown like the rest.
    rejected = failure == "missing_finish"
    assert isinstance(exc, RejectedProviderStream) is rejected
    assert exc.stream_receipt["complete"] is rejected
    assert exc.stream_receipt["generation_id"] == "header-generation"
    assert rows(isolated)[-1]["state"] == "unresolved"
    assert all(row["state"] != "settled" for row in rows(isolated))
    if failure.startswith("error"):
        assert isinstance(exc, ProviderStreamError) and exc.body["error"] == error["error"]
        assert exc.code == "" and exc.stream_usage is None
        assert exc.status_code == (404 if failure == "error_free_claim" else 200)
    from ouroboros.observability import read_blob_ref
    manifest = json.loads((isolated / exc.stream_receipt["manifest_ref"]["path"]).read_text())
    projection = read_blob_ref(isolated, manifest["full_payload_ref"])
    evidence = read_blob_ref(isolated, projection["private_wire_ref"])
    assert base64.b64decode(evidence["wire_base64"]) == wire
    assert evidence["attempt_id"] == exc.physical_attempt_capture.attempt_id


def test_complete_stream_without_usage_keeps_money_unknown(isolated):
    response = WireResponse(sse(chunk({"content": "done"}, "stop")))
    result = run_driver(lambda **k: response, payload(stream=True), target())
    assert result.model_dump()["choices"][0]["message"]["content"] == "done"
    assert rows(isolated)[-1]["cost_usd"] is None
    assert rows(isolated)[-1]["cost_final"] is False


def test_sse_bom_multiline_data_and_split_unicode(isolated):
    event = json.dumps(chunk({"content": "π\u2028done"}, "stop", usage=completion()["usage"]), ensure_ascii=False, indent=2)
    wire = b"\xef\xbb\xbf: comment\r\n\r\n" + b"".join(b"data: " + line.encode() + b"\r\n" for line in event.split("\n"))
    wire += b"\r\ndata: [DONE]\r\n\r\n"
    result = run_driver(lambda **kw: WireResponse(wire), payload(stream=True), target())
    assert result.model_dump()["choices"][0]["message"]["content"] == "π\u2028done"


@pytest.mark.parametrize("delta", [
    {"tool_calls": 1},
    {"function_call": {"name": "lookup"}},
    {"tool_calls": [{"index": 0, "id": "t", "type": "function", "function": {"name": "lookup"}}]},
])
def test_rejected_terminal_body_settles_usage_and_classifies_provider_error(isolated, delta):
    """Drained to the terminal frame, but the body cannot be executed: the verdict is a
    ``RejectedProviderStream`` (no ``code`` — never an unknown outcome) that carries the
    usage frame it read, the ledger row settles at the reported cost, and the classifier
    files ``provider_error`` with no same-model repeat (the cross-model chain may run)."""
    with pytest.raises(RejectedProviderStream) as caught:
        run_driver(lambda **kw: WireResponse(sse(chunk(delta, "stop", usage=completion()["usage"]))),
                   payload(stream=True), target())
    exc = caught.value
    assert not getattr(exc, "code", None) and exc.stream_rejected and exc.stream_incomplete
    assert "choice 0" in str(exc)
    assert exc.stream_usage == completion()["usage"]
    assert exc.stream_receipt["complete"] is True and exc.stream_receipt["manifest_ref"]
    assert exc.physical_attempt_capture.state == "settled"
    assert [row["state"] for row in rows(isolated)] == ["settled"]
    assert rows(isolated)[-1]["cost_usd"] == 0.25 and rows(isolated)[-1]["prompt_tokens"] == 10
    classification = classify_llm_exception(exc)
    assert (classification.kind, classification.retry_same_request) == ("provider_error", False)


def test_terminal_length_with_a_partial_tool_call_returns_like_non_stream(isolated):
    """``finish_reason=length`` after ``[DONE]`` is complete wire: the assembler returns
    the exhausted tool call exactly as the non-stream path would, and the loop owns it."""
    wire = sse(chunk({"tool_calls": [{"index": 0, "id": "t", "type": "function",
                                      "function": {"name": "lookup", "arguments": '{"q":'}}]},
                     "length", usage=completion()["usage"]))
    result = run_driver(lambda **kw: WireResponse(wire), payload(stream=True), target()).model_dump()
    choice = result["choices"][0]
    assert choice["finish_reason"] == "length"
    assert choice["message"]["tool_calls"] == [
        {"id": "t", "type": "function", "function": {"name": "lookup", "arguments": '{"q":'}}]
    assert result["_stream_receipt"]["complete"] is True and result["_stream_receipt"]["anomalies"]["count"] == 0
    assert [row["state"] for row in rows(isolated)] == ["settled"]


_FORGIVEN_SHAPES = {
    "non_json_frame": (
        lambda: b"data: {broken}\r\n\r\n" + sse(chunk({"role": "assistant", "content": "done"}, "stop", usage=completion()["usage"])),
        ["chunk is not JSON; skipped"]),
    "non_object_chunk": (
        lambda: sse([1, 2], chunk({"role": "assistant", "content": "done"}, "stop", usage=completion()["usage"])),
        ["chunk is list, not an object; skipped"]),
    "choices_not_a_list": (
        lambda: sse({"id": "gen-test", "choices": {"index": 0}}, chunk({"content": "done"}, "stop", usage=completion()["usage"])),
        ["choices is dict, not a list; skipped"]),
    "choice_not_an_object": (
        lambda: sse({"id": "gen-test", "choices": ["oops"]}, chunk({"content": "done"}, "stop", usage=completion()["usage"])),
        ["choice at position 0 is not an object; skipped"]),
    "logprobs_not_an_object": (
        lambda: sse({"id": "gen-test", "object": "chat.completion.chunk", "model": "vendor/test-stream",
                     "choices": [{"index": 0, "delta": {"content": "done"}, "finish_reason": "stop", "logprobs": 5}],
                     "usage": completion()["usage"]}),
        ["choice 0: logprobs is int, not an object; skipped"]),
    "reasoning_item_without_type": (
        lambda: sse(chunk({"reasoning_details": [{"text": "why"}]}), chunk({"content": "done"}, "stop", usage=completion()["usage"])),
        ["reasoning_details: item without type"]),
    "tool_call_index_not_an_integer": (
        lambda: sse(chunk({"tool_calls": [{"index": "zero", "id": "t", "type": "function",
                                           "function": {"name": "lookup", "arguments": "{}"}}]}, "tool_calls",
                          usage=completion()["usage"])),
        ["tool_calls: index 'zero' is not a non-negative integer; treated as absent", "tool_calls: item without index; appended"]),
    "finish_reason_not_a_string": (
        lambda: sse(chunk({"content": "done"}, 7), chunk({}, "stop", usage=completion()["usage"])),
        ["choice 0: finish_reason 7 is not a non-empty string; ignored"]),
    "content_after_finish": (
        lambda: sse(chunk({"content": "do"}, "stop"), chunk({"content": "ne"}, None, usage=completion()["usage"])),
        ["choice 0: content after finish_reason 'stop'; accepted"]),
    "model_conflict": (
        lambda: sse(chunk({"content": "do"}), chunk({"content": "ne"}, "stop", usage=completion()["usage"], model="vendor/other")),
        ["model: 'vendor/test-stream' then 'vendor/other'; kept first"]),
    "object_delta_onto_text": (
        lambda: sse(chunk({"content": "done"}), chunk({"content": {"x": 1}}, "stop", usage=completion()["usage"])),
        ["content: object delta onto str; kept first shape"]),
    "list_delta_onto_text": (
        lambda: sse(chunk({"content": "done"}), chunk({"content": [1]}, "stop", usage=completion()["usage"])),
        ["content: list delta onto str; kept first shape"]),
    "text_delta_onto_scalar": (
        lambda: sse(chunk({"content": 7}), chunk({"content": "done"}, "stop", usage=completion()["usage"])),
        ["content: text delta onto int; kept first shape"]),
    # [DONE] and a later frame in ONE network read: the assembler drains the whole chunk.
    "data_after_done": (
        lambda: WireResponse(sse(chunk({"content": "done"}, "stop", usage=completion()["usage"]))
                             + b"data: {\"id\": \"late\"}\r\n\r\n", chunk_size=1 << 16),
        ["data after [DONE]; ignored"]),
}


@pytest.mark.parametrize("shape", sorted(_FORGIVEN_SHAPES))
def test_form_irregularities_never_raise_before_the_terminal_verdict(isolated, shape):
    """The #856 class itself: a local form judgment inside ``accept()`` is a note, never a raise.
    Every forgiven shape assembles a complete reply, settles, and names the forgiven fact in the
    receipt; none of them is an unknown outcome."""
    build, expected = _FORGIVEN_SHAPES[shape]
    wire = build()
    response = wire if isinstance(wire, WireResponse) else WireResponse(wire)
    result = run_driver(lambda **kw: response, payload(stream=True), target()).model_dump()
    receipt = result["_stream_receipt"]
    assert receipt["complete"] is True
    assert all(note in receipt["anomalies"]["first"] for note in expected), receipt["anomalies"]
    assert result["choices"][0]["finish_reason"] in ("stop", "tool_calls")
    assert rows(isolated)[-1]["state"] == "settled"


def test_clean_close_after_every_choice_finished_is_terminal_framing(isolated):
    """The provider closed the body cleanly after the choice's ``finish_reason`` but never sent
    ``[DONE]``: both witnesses of terminal framing count, so the reply is complete (disclosed in the
    receipt), settles, and is not an unknown outcome."""
    wire = sse(chunk({"role": "assistant", "content": "done"}, "stop", usage=completion()["usage"]), done=False)
    result = run_driver(lambda **kw: WireResponse(wire), payload(stream=True), target()).model_dump()
    msg, usage = LLMClient()._normalize_remote_response(result, target(), skip_cost_fetch=True)
    assert msg["content"] == "done" and result["_stream_receipt"]["complete"] is True
    assert usage["stream_receipt"]["anomalies"] == {"count": 1, "first": [
        "stream closed without [DONE] after every choice finished"]}
    assert rows(isolated)[-1]["state"] == "settled"


@pytest.mark.parametrize("payload_key,field", [("function", "arguments"), ("custom", "input")])
def test_tool_call_without_type_is_structurally_complete(isolated, payload_key, field):
    """``type`` is not part of structural completeness (id, name, arguments are): a call whose
    fragments never carried it returns exactly as the non-stream path returns it, for either payload."""
    call = {"index": 0, "id": "t", payload_key: {"name": "lookup", field: '{"q":"x"}'}}
    wire = sse(chunk({"tool_calls": [call]}, "tool_calls", usage=completion()["usage"]))
    result = run_driver(lambda **kw: WireResponse(wire), payload(stream=True), target()).model_dump()
    assert result["choices"][0]["message"]["tool_calls"] == [{"id": "t", payload_key: {"name": "lookup", field: '{"q":"x"}'}}]
    assert result["_stream_receipt"]["anomalies"]["count"] == 0 and rows(isolated)[-1]["state"] == "settled"


def test_clean_close_with_a_missing_expected_choice_keeps_response_unknown_and_price(isolated):
    """``n=2``, one choice finished, the body closes without ``[DONE]`` and without the second
    choice: the reply remains an unknown outcome, while its received price is final."""
    wire = sse(chunk({"role": "assistant", "content": "only one"}, "stop", usage=completion()["usage"]), done=False)
    with pytest.raises(IncompleteProviderStream):
        run_driver(lambda **kw: WireResponse(wire), payload(stream=True, n=2), target())
    assert rows(isolated)[-1]["state"] == "settled"
    assert rows(isolated)[-1]["cost_usd"] == 0.25 and rows(isolated)[-1]["cost_final"] is True


def test_later_usage_snapshot_overrides_an_earlier_one(isolated):
    """Usage frames are cumulative snapshots: the last one read is the one settled, so an early
    partial snapshot never becomes the attempt's cost."""
    early = {"id": "gen-test", "object": "chat.completion.chunk", "model": "vendor/test-stream", "choices": [],
             "usage": {"prompt_tokens": 10, "completion_tokens": 0, "cost": 0.01}}
    wire = sse(early, chunk({"role": "assistant", "content": "done"}), chunk({}, "stop", usage=completion()["usage"]))
    result = run_driver(lambda **kw: WireResponse(wire), payload(stream=True), target()).model_dump()
    assert result["usage"] == completion()["usage"]
    assert rows(isolated)[-1]["state"] == "settled" and rows(isolated)[-1]["cost_usd"] == 0.25


def test_identity_conflict_is_forgiven_and_disclosed_in_the_receipt(isolated):
    """First value wins for identity scalars (a tool call's ``id``, the envelope ``id``) and for a
    choice's ``finish_reason``; each conflict is a fact in ``usage["stream_receipt"]["anomalies"]``,
    never a raise."""
    wire = sse(chunk({"tool_calls": [{"index": 0, "id": "call_a", "type": "function",
                                      "function": {"name": "lookup", "arguments": '{"q":'}}]}),
               chunk({"tool_calls": [{"index": 0, "id": "call_b", "function": {"arguments": '"ok"}'}}]},
                     "tool_calls", usage=completion()["usage"], id="gen-other"),
               chunk({}, "stop"))
    result = run_driver(lambda **kw: WireResponse(wire), payload(stream=True), target()).model_dump()
    msg, usage = LLMClient()._normalize_remote_response(result, target(), skip_cost_fetch=True)
    assert msg["tool_calls"][0]["id"] == "call_a" and msg["response_id"] == "gen-test"
    assert json.loads(msg["tool_calls"][0]["function"]["arguments"]) == {"q": "ok"}
    assert result["choices"][0]["finish_reason"] == "tool_calls"
    assert usage["stream_receipt"]["anomalies"] == {"count": 3, "first": [
        "id: 'gen-test' then 'gen-other'; kept first",
        "id: 'call_a' then 'call_b'; kept first",
        "choice 0: finish_reason 'tool_calls' then 'stop'; kept first",
    ]}
    assert rows(isolated)[-1]["state"] == "settled"


def test_finish_frame_without_delta_still_completes_the_choice(isolated):
    """A terminal frame that carries ``finish_reason`` but no ``delta`` is a shape
    irregularity, not a missing terminal: the choice finishes, the reply settles, and
    the forgiven shape is disclosed in the receipt."""
    terminal = {"id": "gen-test", "object": "chat.completion.chunk", "model": "vendor/test-stream",
                "choices": [{"index": 0, "finish_reason": "stop"}], "usage": completion()["usage"]}
    wire = sse(chunk({"role": "assistant", "content": "done"}), terminal)
    result = run_driver(lambda **kw: WireResponse(wire), payload(stream=True), target()).model_dump()
    msg, usage = LLMClient()._normalize_remote_response(result, target(), skip_cost_fetch=True)
    assert msg["content"] == "done" and result["choices"][0]["finish_reason"] == "stop"
    assert usage["stream_receipt"]["anomalies"] == {"count": 1, "first": [
        "choice 0: delta is NoneType, not an object; treated as empty",
    ]}
    assert rows(isolated)[-1]["state"] == "settled"


def test_lone_choice_with_a_foreign_index_is_remapped_on_a_single_choice_stream(isolated):
    """A single-choice reply whose only choice carries index 1 is a form irregularity after
    ``[DONE]``: it is remapped to choice 0 and disclosed, never an unknown outcome."""
    wire = sse(chunk({"role": "assistant", "content": "done"}, "stop", index=1, usage=completion()["usage"]))
    result = run_driver(lambda **kw: WireResponse(wire), payload(stream=True), target()).model_dump()
    assert result["choices"][0]["index"] == 0 and result["choices"][0]["message"]["content"] == "done"
    msg, usage = LLMClient()._normalize_remote_response(result, target(), skip_cost_fetch=True)
    assert usage["stream_receipt"]["anomalies"]["first"] == ["choice at position 0: index 1; used 0"]
    assert rows(isolated)[-1]["state"] == "settled"


def test_missing_choice_after_done_is_rejected_not_unknown(isolated):
    """``n=2`` with only one choice at ``[DONE]``: the wire is complete, the body is unusable —
    a rejection with usage (settled), not an unknown outcome."""
    wire = sse(chunk({"role": "assistant", "content": "only one"}, "stop", usage=completion()["usage"]))
    with pytest.raises(RejectedProviderStream) as caught:
        run_driver(lambda **kw: WireResponse(wire), payload(stream=True, n=2), target())
    assert "choices [0] present, 2 expected" in str(caught.value) and caught.value.stream_receipt["complete"] is True
    assert rows(isolated)[-1]["state"] == "settled"


def test_non_text_scalar_after_text_keeps_the_first_shape(isolated):
    """First shape wins in both directions: text established by earlier frames is not
    overwritten by a later malformed non-text scalar; the conflict is disclosed."""
    wire = sse(chunk({"role": "assistant", "content": "ok"}), chunk({"content": 7}),
               chunk({}, "stop", usage=completion()["usage"]))
    result = run_driver(lambda **kw: WireResponse(wire), payload(stream=True), target()).model_dump()
    msg, usage = LLMClient()._normalize_remote_response(result, target(), skip_cost_fetch=True)
    assert msg["content"] == "ok"
    assert usage["stream_receipt"]["anomalies"] == {"count": 1, "first": ["content: int delta onto str; kept first shape"]}
    assert rows(isolated)[-1]["state"] == "settled"


def test_index_less_tool_call_fragment_continues_the_last_call(isolated):
    """A tool-call delta without ``index`` is a fragment of the last call when its type/id
    are compatible (merged, disclosed); a fresh id is a new call (appended, disclosed)."""
    wire = sse(chunk({"tool_calls": [{"index": 0, "id": "t", "type": "function",
                                      "function": {"name": "lookup", "arguments": '{"q":'}}]}),
               chunk({"tool_calls": [{"function": {"arguments": '"ok"}'}}]}),
               chunk({"tool_calls": [{"id": "u", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}]}),
               chunk({}, "tool_calls", usage=completion()["usage"]))
    result = run_driver(lambda **kw: WireResponse(wire), payload(stream=True), target()).model_dump()
    assert result["choices"][0]["message"]["tool_calls"] == [
        {"id": "t", "type": "function", "function": {"name": "lookup", "arguments": '{"q":"ok"}'}},
        {"id": "u", "type": "function", "function": {"name": "lookup", "arguments": "{}"}},
    ]
    assert result["_stream_receipt"]["anomalies"]["first"] == [
        "tool_calls: item without index; merged into the last call",
        "tool_calls: item without index; appended",
    ]


def test_index_less_annotations_stay_separate_records(isolated):
    """Lists that never carry an index (annotations) keep their append semantics: two
    citations of the same type are two harvested sources, never merged into one."""
    wire = sse(chunk({"content": "cite", "annotations": [
                   {"type": "url_citation", "url_citation": {"url": "https://a.test", "title": "A"}}]}),
               chunk({"annotations": [{"type": "url_citation", "url_citation": {"url": "https://b.test", "title": "B"}}]},
                     "stop", usage=completion()["usage"]))
    result = run_driver(lambda **kw: WireResponse(wire), payload(stream=True), target()).model_dump()
    _msg, usage = LLMClient()._normalize_remote_response(result, target(), skip_cost_fetch=True)
    assert [source["url"] for source in usage["web_search_sources"]] == ["https://a.test", "https://b.test"]
    assert usage["stream_receipt"]["anomalies"]["count"] == 0


@pytest.mark.parametrize("usage_first", [False, True])
def test_mid_stream_error_chunk_classifies_by_body_and_settles_only_with_usage(isolated, usage_first):
    """Doc-shaped fixture: the corpus holds no recorded error chunk (the recorded refusals
    are ``content_filter`` finishes, not errors), so this is OpenRouter's documented
    mid-stream ``{"error": {"code": 502, "message": ...}}`` chunk. A provider fact, not an
    unknown outcome: the body code becomes the status the classifier files
    (``provider_transient``), and custody settles only when a usage frame was read first."""
    error = {"id": "gen-test", "object": "chat.completion.chunk",
             "error": {"code": 502, "message": "Upstream provider error"},
             "choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": "error"}]}
    frames = [chunk({"content": "par"})]
    if usage_first:
        frames.append({"id": "gen-test", "choices": [], "usage": completion()["usage"]})
    frames.append(error)
    with pytest.raises(ProviderStreamError) as caught:
        run_driver(lambda **kw: WireResponse(sse(*frames, done=False)), payload(stream=True), target())
    exc = caught.value
    assert exc.code == "" and exc.status_code == 502 and exc.provider_message == "Upstream provider error"
    assert exc.stream_receipt["complete"] is False
    assert exc.stream_usage == (completion()["usage"] if usage_first else None)
    assert exc.physical_attempt_capture.state == ("settled" if usage_first else "unresolved")
    assert rows(isolated)[-1]["state"] == ("settled" if usage_first else "unresolved")
    if usage_first:
        assert rows(isolated)[-1]["cost_usd"] == 0.25
    classification = classify_llm_exception(exc)
    assert (classification.kind, classification.retry_same_request, classification.status_code) == (
        "provider_transient", True, 502)


@pytest.mark.parametrize("usage_first", [False, True])
@pytest.mark.parametrize("shape", ["finish_reason_error", "code_less_error_frame"])
def test_code_less_stream_error_is_a_provider_verdict_not_an_unknown_outcome(isolated, usage_first, shape):
    """An SSE error without an HTTP-shaped code — the provider's own ``finish_reason: "error"``, or an
    ``{"error": {"type": ...}}`` frame (the overload/api_error shape) — is the provider's terminal verdict on
    this stream: ``provider_error`` with no same-request repeat, whether or not a usage frame was read
    first; it never reopens the unknown-outcome continuation. A usage snapshot carried by the error frame
    itself settles the attempt."""
    frames = [chunk({"content": "par"})]
    if usage_first:
        frames.append({"id": "gen-test", "choices": [], "usage": completion()["usage"]})
    if shape == "finish_reason_error":
        frames.append(chunk({"content": ""}, "error"))
    else:
        frames.append({"id": "gen-test", "error": {"type": "overloaded_error", "message": "Overloaded"},
                       **({"usage": completion()["usage"]} if usage_first else {})})
    with pytest.raises(ProviderStreamError) as caught:
        run_driver(lambda **kw: WireResponse(sse(*frames, done=False)), payload(stream=True), target())
    exc = caught.value
    assert exc.stream_rejected and exc.code == "" and exc.status_code == 200
    assert rows(isolated)[-1]["state"] == ("settled" if usage_first else "unresolved")
    classification = classify_llm_exception(exc)
    assert (classification.kind, classification.retry_same_request) == ("provider_error", False)


def test_error_frame_carrying_its_own_usage_settles_the_attempt(isolated):
    """The usage snapshot on the error frame itself is the latest money fact: it settles the attempt
    instead of leaving an unresolved upper bound behind a known outcome."""
    frames = [chunk({"content": "par"}),
              {"id": "gen-test", "error": {"code": 502, "message": "Upstream provider error"},
               "usage": completion()["usage"]}]
    with pytest.raises(ProviderStreamError) as caught:
        run_driver(lambda **kw: WireResponse(sse(*frames, done=False)), payload(stream=True), target())
    assert caught.value.stream_usage == completion()["usage"]
    assert rows(isolated)[-1]["state"] == "settled" and rows(isolated)[-1]["cost_usd"] == 0.25


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("field", ["stream", "stream_options"])
def test_stream_rejection_uses_existing_wire_recovery(isolated, asynchronous, field):
    calls = []
    def send(**kw):
        calls.append(kw)
        if len(calls) == 1:
            raise Rejected(f"Unsupported parameter: '{field}'")
        if kw.get("stream"):
            return WireResponse(sse(chunk({"content": "done"}, "stop", usage=completion()["usage"])))
        return AssembledResponse(completion())
    run_driver(send, payload(stream=True, stream_options={"include_usage": True}), target(), asynchronous=asynchronous)
    assert len(calls) == 2 and field not in calls[1]
    assert ("stream" in calls[1]) == (field == "stream_options")
    assert calls[1]["max_tokens"] == 1024 and calls[1]["reasoning_effort"] == "high"


@pytest.mark.parametrize("asynchronous", [False, True])
def test_each_recovery_reads_current_deadline(isolated, monkeypatch, asynchronous):
    now = [100.0]
    monkeypatch.setattr(model_wait, "monotonic_now", lambda slot=None: now[0])
    calls = []
    def send(**kw):
        calls.append(kw)
        now[0] += 3
        if len(calls) == 1:
            raise Rejected("temperature unsupported")
        if len(calls) == 2:
            raise Rejected("response_format unsupported")
        return AssembledResponse(completion())
    with model_wait.execution_deadline_scope(110):
        run_driver(send, payload(temperature=0.2, response_format={"type": "json_object"}, timeout=600),
                   target(), asynchronous=asynchronous)
    assert [call["timeout"] for call in calls] == [10, 7, 4]


@pytest.mark.parametrize("asynchronous", [False, True])
def test_exhaustion_preserves_earlier_paid_capture(isolated, monkeypatch, asynchronous):
    now = [100.0]
    monkeypatch.setattr(model_wait, "monotonic_now", lambda slot=None: now[0])
    calls = []
    def send(**kw):
        calls.append(kw)
        now[0] = 111
        raise Rejected("temperature unsupported")
    with model_wait.execution_deadline_scope(110), pytest.raises(PhysicalDispatchInterrupted) as caught:
        run_driver(send, payload(temperature=0.2), target(), asynchronous=asynchronous)
    from ouroboros.transport_custody import is_pre_dispatch_transport_failure
    assert len(calls) == 1
    assert not is_pre_dispatch_transport_failure(caught.value)
    assert caught.value.physical_attempt_capture.state == "unresolved"
    assert caught.value.physical_attempt_capture.attempt_id == rows(isolated)[0]["attempt_id"]
    assert [(row["state"], row["revision"]) for row in rows(isolated)] == [("unresolved", 4)]
    assert rows(isolated)[0]["physical_failure"]


def test_expired_initial_window_reserves_nothing(isolated, monkeypatch):
    run_driver(lambda **kw: AssembledResponse(completion()), payload(), target())
    previous_rows = rows(isolated)
    assert ua.last_physical_attempt_capture() is not None
    monkeypatch.setattr(model_wait, "monotonic_now", lambda slot=None: 100)
    with model_wait.execution_deadline_scope(100), pytest.raises(PhysicalDispatchInterrupted):
        run_driver(lambda **kw: pytest.fail("dispatched"), payload(), target())
    assert rows(isolated) == previous_rows
    assert ua.last_physical_attempt_capture() is None


def test_slow_candidate_preparation_rechecks_before_dispatch(isolated, monkeypatch):
    import ouroboros.observability as observability
    now = [100]
    monkeypatch.setattr(model_wait, "monotonic_now", lambda slot=None: now[0])
    original = observability.persist_physical_candidate
    def persist(*a, **kw):
        result = original(*a, **kw)
        now[0] = 111
        return result
    monkeypatch.setattr(observability, "persist_physical_candidate", persist)
    with model_wait.execution_deadline_scope(110), pytest.raises(PhysicalDispatchInterrupted) as caught:
        run_driver(lambda **kw: pytest.fail("dispatched"), payload(), target())
    assert [row["state"] for row in rows(isolated)] == ["released"]
    assert caught.value.physical_attempt_capture.candidate_manifest_ref


@pytest.mark.parametrize("asynchronous", [False, True])
def test_slow_recovery_preparation_keeps_both_attempt_receipts(isolated, monkeypatch, asynchronous):
    import ouroboros.observability as observability
    from ouroboros.transport_custody import is_pre_dispatch_transport_failure
    now, prepared, sent = [100], [], []
    monkeypatch.setattr(model_wait, "monotonic_now", lambda slot=None: now[0])
    original = observability.persist_physical_candidate
    def persist(*a, **kw):
        result = original(*a, **kw)
        prepared.append(result)
        if len(prepared) == 2:
            now[0] = 111
        return result
    def send(**kw):
        sent.append(kw)
        raise Rejected("temperature unsupported")
    monkeypatch.setattr(observability, "persist_physical_candidate", persist)
    with model_wait.execution_deadline_scope(110), pytest.raises(PhysicalDispatchInterrupted) as caught:
        run_driver(send, payload(temperature=0.2), target(), asynchronous=asynchronous)
    assert len(sent) == 1
    assert caught.value.physical_attempt_capture.state == "unresolved"
    assert caught.value.deadline_attempt_capture.state == "released"
    assert caught.value.deadline_attempt_capture.candidate_manifest_ref
    assert not is_pre_dispatch_transport_failure(caught.value)
    assert [row["state"] for row in rows(isolated)] == ["unresolved", "released"]


def test_later_free_recovery_cannot_replace_earlier_unknown_custody(isolated, monkeypatch):
    now, sent = [100], []
    monkeypatch.setattr(model_wait, "monotonic_now", lambda slot=None: now[0])
    def send(**kw):
        sent.append(kw)
        if len(sent) == 1:
            raise Rejected("temperature unsupported")
        now[0] = 111
        raise Rejected("No endpoints found; response_format unsupported", status=404)
    with model_wait.execution_deadline_scope(110), pytest.raises(PhysicalDispatchInterrupted) as caught:
        run_driver(send, payload(temperature=0.2, response_format={"type": "json_object"}), target())
    assert len(sent) == 2 and rows(isolated)[-1]["state"] == "settled"
    assert caught.value.physical_attempt_capture.attempt_id == rows(isolated)[0]["attempt_id"]
    assert caught.value.physical_attempt_capture.state == "unresolved"


def test_quota_pause_is_execution_only_and_network_wait_spends_time(isolated, monkeypatch):
    now = [100.0]
    monkeypatch.setattr(model_wait.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(deadline_utils, "utc_now", lambda: datetime.fromtimestamp(now[0], timezone.utc))
    owner = model_wait.TaskModelWait(task={"id": "clock"}, drive_root=isolated, event_queue=None,
                                    worker_slot_held=False, owner_control=lambda: None)
    token = model_wait._CURRENT.set(owner)
    try:
        with model_wait.execution_deadline_scope(110):
            owner.clocks[""].enter("quota", now[0])
            now[0] = 120
            assert model_wait.dispatch_deadline_remaining_sec() == 10
            with model_wait.calendar_scope(datetime.fromtimestamp(115, timezone.utc).isoformat()):
                assert model_wait.dispatch_deadline_remaining_sec() == 0
            owner.clocks[""].leave("quota", now[0])
            now[0] = 123
            assert model_wait.dispatch_deadline_remaining_sec() == 7
    finally:
        model_wait._CURRENT.reset(token)


@pytest.mark.parametrize("asynchronous", [False, True])
def test_default_timeout_and_no_proxy_phases(isolated, monkeypatch, asynchronous):
    seen = []
    run_driver(lambda **kw: (seen.append(kw), AssembledResponse(completion()))[1], payload(), target(), asynchronous=asynchronous)
    assert "timeout" not in seen[0]
    timeout = LLMClient._no_proxy_timeout(120)
    assert timeout.connect == 30 and timeout.read == 120
    monkeypatch.setattr(model_wait, "monotonic_now", lambda slot=None: 100)
    with model_wait.execution_deadline_scope(105):
        run_driver(lambda **kw: (seen.append(kw), AssembledResponse(completion()))[1], payload(timeout=timeout), target(), asynchronous=asynchronous)
    assert seen[-1]["timeout"].as_dict() == {"connect": 5, "read": 5, "write": 5, "pool": 5}


def native_events():
    blocks = [{"type": "thinking", "thinking": "reasoning", "signature": "private-signature"},
              {"type": "redacted_thinking", "data": "private-redacted"},
              {"type": "text", "text": "read this", "citations": [{"type": "char_location", "cited_text": "source"}]},
              {"type": "tool_use", "id": "native-tool", "name": "lookup", "input": {"q": "yes"}}]
    body = {"id": "msg-test", "type": "message", "role": "assistant", "model": "vendor/test-stream", "content": [],
            "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": 5, "output_tokens": 1, "cache_read_input_tokens": 10}}
    events = [("message_start", {"type": "message_start", "message": body})]
    for index, block in enumerate(blocks):
        start = copy.deepcopy(block)
        deltas = []
        if block["type"] == "thinking":
            start.update(thinking="", signature="")
            deltas = [{"type": "thinking_delta", "thinking": "reason"}, {"type": "thinking_delta", "thinking": "ing"},
                      {"type": "signature_delta", "signature": "private-"}, {"type": "signature_delta", "signature": "signature"}]
        elif block["type"] == "tool_use":
            start["input"] = {}
            deltas = [{"type": "input_json_delta", "partial_json": '{"q":'}, {"type": "input_json_delta", "partial_json": '"yes"}'}]
        elif block["type"] == "text":
            start.update(text="", citations=[])
            deltas = [{"type": "text_delta", "text": "read "}, {"type": "text_delta", "text": "this"},
                      {"type": "citations_delta", "citation": block["citations"][0]}]
        events.append(("content_block_start", {"type": "content_block_start", "index": index, "content_block": start}))
        events.extend(("content_block_delta", {"type": "content_block_delta", "index": index, "delta": delta}) for delta in deltas)
        events.append(("content_block_stop", {"type": "content_block_stop", "index": index}))
    usage = {"output_tokens": 12, "cache_creation_input_tokens": 3,
             "cache_creation": {"ephemeral_1h_input_tokens": 3}, "cost": 0.4}
    events.extend([("ping", {"type": "ping"}),
                   ("message_delta", {"type": "message_delta", "delta": {}, "usage": {"output_tokens": 5, "cost": 0.2}}),
                   ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": usage}),
                   ("message_stop", {"type": "message_stop"})])
    return events, {**body, "content": blocks, "stop_reason": "tool_use", "usage": {**body["usage"], **usage}}


@pytest.mark.parametrize("no_proxy", [False, True])
def test_native_sse_private_receipt_and_json_parity(isolated, monkeypatch, no_proxy):
    import requests
    from ouroboros.anthropic_native_custody import anthropic_replay_scope, native_content_for_replay, public_custody_projection
    events, expected = native_events()
    sent = []
    response = WireResponse(sse(*events, done=False), step=lambda: assert_unsettled(isolated))
    def post(*a, **kw):
        sent.append(kw)
        return response
    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setattr(requests.Session, "post", post)
    client = LLMClient()
    msg, usage = client._chat_anthropic(target("anthropic"), MESSAGES, TOOLS, "high", 1024, "auto",
                                      no_proxy=no_proxy, stream=True)
    json_msg, json_usage = client._normalize_anthropic_response(expected, target("anthropic"),
                                                             prompt_cache_ttl=usage.get("prompt_cache_ttl"))
    usage.pop("stream_receipt")
    usage.pop("request_wire", None)
    assert msg == json_msg and usage == json_usage
    with anthropic_replay_scope():
        assert native_content_for_replay(msg, target("anthropic"), ["native-tool"]) == expected["content"]
    assert "private-signature" not in json.dumps(public_custody_projection(msg))
    assert sent[0]["stream"] is True and sent[0]["json"]["stream"] is True
    assert response.closed and rows(isolated)[-1]["cost_usd"] == 0.4


@pytest.mark.parametrize("omit", ["message_stop", "content_block_stop"])
def test_native_incomplete_blocks_or_message_cannot_return_tools(isolated, monkeypatch, omit):
    import requests
    events, expected = native_events()
    events = [(kind, body) for kind, body in events if kind != omit]
    monkeypatch.setattr(requests, "post", lambda *a, **k: WireResponse(sse(*events, done=False)))
    with pytest.raises(IncompleteProviderStream):
        LLMClient()._chat_anthropic(target("anthropic"), MESSAGES, TOOLS, "high", 1024, "auto", stream=True)
    row = rows(isolated)[-1]
    if omit == "message_stop":
        assert row["state"] == "settled" and row["cost_usd"] == 0.4
    else:
        assert row["state"] == "unresolved"  # No final message_delta was accepted.


def test_native_unusable_body_after_message_stop_is_rejected_and_settled(isolated, monkeypatch):
    """A thinking block that never received its signature is unusable, but ``message_stop`` and the
    usage arrived: the native path judges once after terminal framing exactly like the Chat path —
    ``RejectedProviderStream`` with the usage, a settled ledger row, and ``provider_error`` (no retry of
    the same request, no unknown-outcome continuation)."""
    import requests
    events, _expected = native_events()
    events = [(kind, body) for kind, body in events
              if not (kind == "content_block_delta" and (body.get("delta") or {}).get("type") == "signature_delta")]
    monkeypatch.setattr(requests, "post", lambda *a, **k: WireResponse(sse(*events, done=False)))
    with pytest.raises(RejectedProviderStream) as caught:
        LLMClient()._chat_anthropic(target("anthropic"), MESSAGES, TOOLS, "high", 1024, "auto", stream=True)
    assert "thinking block lacks its complete signature" in str(caught.value)
    assert caught.value.stream_usage["input_tokens"] == 5 and not hasattr(caught.value, "code")
    assert rows(isolated)[-1]["state"] == "settled"
    verdict = classify_llm_exception(caught.value)
    assert (verdict.kind, verdict.retry_same_request) == ("provider_error", False)


def test_native_rejection_without_message_delta_keeps_money_unknown(isolated, monkeypatch):
    """``message_stop`` arrived but no ``message_delta`` did (no stop_reason, no final counters): the
    body is rejected, and the ledger keeps its unresolved upper bound instead of settling on
    ``message_start``'s lower-bound snapshot — the same policy as the mid-stream error frame."""
    import requests
    events, _expected = native_events()
    events = [(kind, body) for kind, body in events if kind != "message_delta"]
    monkeypatch.setattr(requests, "post", lambda *a, **k: WireResponse(sse(*events, done=False)))
    with pytest.raises(RejectedProviderStream) as caught:
        LLMClient()._chat_anthropic(target("anthropic"), MESSAGES, TOOLS, "high", 1024, "auto", stream=True)
    assert "no stop_reason after message_stop" in str(caught.value) and caught.value.stream_usage is None
    assert rows(isolated)[-1]["state"] == "unresolved"
    assert classify_llm_exception(caught.value).kind == "provider_error"


def test_native_non_json_frame_is_the_typed_unknown_outcome(isolated, monkeypatch):
    """A truncated ``data:`` line in a native stream is ``IncompleteProviderStream`` (typed
    ``model_outcome_unknown``), never a raw ``JSONDecodeError`` whose classification would rest
    on custody state alone."""
    import requests
    events, _expected = native_events()
    wire = sse(*events[:3], done=False) + b"data: {broken\r\n\r\n"
    monkeypatch.setattr(requests, "post", lambda *a, **k: WireResponse(wire))
    with pytest.raises(IncompleteProviderStream) as caught:
        LLMClient()._chat_anthropic(target("anthropic"), MESSAGES, TOOLS, "high", 1024, "auto", stream=True)
    assert "not JSON" in str(caught.value) and caught.value.code == "model_outcome_unknown"
    assert rows(isolated)[-1]["state"] == "unresolved"


def test_native_max_tokens_with_tool_blocks_returns_like_non_stream(isolated, monkeypatch):
    """``stop_reason == max_tokens`` beside a complete tool block is a finished reply the loop reads
    (parity with the non-stream native path, which surfaces ``stop_reason``); it is not a rejection."""
    import requests
    events, expected = native_events()
    for kind, body in events:
        if kind == "message_delta":
            body["delta"]["stop_reason"] = "max_tokens"
    monkeypatch.setattr(requests, "post", lambda *a, **k: WireResponse(sse(*events, done=False)))
    message, usage = LLMClient()._chat_anthropic(target("anthropic"), MESSAGES, TOOLS, "high", 1024, "auto", stream=True)
    assert message["stop_reason"] == "max_tokens" and message["tool_calls"][0]["function"]["name"] == "lookup"
    assert rows(isolated)[-1]["state"] == "settled"


@pytest.mark.parametrize("before_content", [False, True])
@pytest.mark.parametrize("error,status,kind", [
    ({"type": "overloaded_error", "message": "Overloaded"}, 200, "provider_error"),
    ({"type": "api_error", "code": 502, "message": "Bad gateway"}, 502, "provider_transient"),
])
def test_native_http_200_sse_error_keeps_producer_facts_and_custody(isolated, monkeypatch, before_content,
                                                                    error, status, kind):
    """An explicit SSE error is a provider fact, not ``model_outcome_unknown``: a numeric
    body code becomes the status the classifier files (502 → provider_transient); a
    type-only body (the overload shape) is the provider's own verdict on the stream
    (``stream_rejected`` → provider_error, no same-request repeat), never an unknown
    outcome. Custody stays unresolved either way: no final usage frame was read, and
    ``message_start``'s snapshot is only a lower bound."""
    import requests
    events, _ = native_events()
    event = {"type": "error", "error": error}
    response = WireResponse(sse(*(events[:3] if not before_content else []), ("error", event), done=False))
    monkeypatch.setattr(requests, "post", lambda *a, **kw: response)
    with pytest.raises(ProviderStreamError) as caught:
        LLMClient()._chat_anthropic(target("anthropic"), MESSAGES, TOOLS, "high", 1024, "auto", stream=True)
    exc = caught.value
    assert exc.body == event and exc.type == error["type"]
    assert exc.code == "" and exc.status_code == status and exc.stream_usage is None
    assert exc.stream_rejected is (status == 200)
    assert exc.stream_receipt["generation_id"] == "header-generation" and exc.stream_receipt["complete"] is False
    assert response.closed and rows(isolated)[-1]["state"] == "unresolved"
    assert classify_llm_exception(exc).kind == kind


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("no_proxy", [False, True])
def test_public_remote_chat_carries_stream_deadline_and_custom_receipts(isolated, monkeypatch, asynchronous, no_proxy):
    client = LLMClient()
    calls = []
    wire = sse(chunk({"tool_calls": [{"index": 0, "id": "c", "type": "custom", "custom": {"name": "lookup", "input": '{"q":"yes"}'}}]},
                     "tool_calls", usage=completion()["usage"]))
    def send(**kw):
        calls.append(kw)
        return WireResponse(wire)
    async def async_send(**kw):
        return send(**kw)
    sdk = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=async_send if asynchronous else send)))
    class Closer:
        def close(self):
            pass
        async def aclose(self):
            pass
    monkeypatch.setattr(client, "_resolve_remote_target", lambda model: target("openai"))
    monkeypatch.setattr(client, "_get_remote_client", lambda t: sdk)
    monkeypatch.setattr(client, "_get_async_remote_client", lambda t: sdk)
    monkeypatch.setattr(client, "_make_no_proxy_client", lambda t, timeout: (sdk, Closer()))
    monkeypatch.setattr(client, "_make_no_proxy_async_client", lambda t, timeout: (sdk, Closer()))
    monkeypatch.setattr(deadline_utils, "utc_now", lambda: datetime.fromtimestamp(100, timezone.utc))
    kwargs = dict(messages=MESSAGES, model="openai::test", tools=TOOLS, stream=True, caller_deadline_ts=110,
                  reasoning_effort="high", max_tokens=1024, no_proxy=no_proxy)
    msg, usage = asyncio.run(client.chat_async(**kwargs)) if asynchronous else client.chat(**kwargs)
    assert msg["tool_calls"][0]["function"] == {"name": "lookup", "arguments": '{"q":"yes"}'}
    assert usage["_request_wire_custom_receipts"][0].allows_execution
    assert calls[0]["stream"] is True and calls[0]["stream_options"] == {"include_usage": True}
    assert calls[0]["max_completion_tokens"] == 1024 and calls[0]["reasoning_effort"] == "high"
    timeout = calls[0]["timeout"]
    assert (timeout.read if isinstance(timeout, httpx.Timeout) else timeout) == 10


def test_late_complete_stream_settles_original_attempt(isolated, monkeypatch):
    now = [100]
    monkeypatch.setattr(model_wait, "monotonic_now", lambda slot=None: now[0])
    def advance():
        now[0] = 200
    response = WireResponse(sse(chunk({"content": "done"}, "stop", usage=completion()["usage"])), step=advance)
    with model_wait.execution_deadline_scope(110):
        result = run_driver(lambda **kw: response, payload(stream=True), target())
    assert result.model_dump()["choices"][0]["message"]["content"] == "done"
    assert [row["state"] for row in rows(isolated)] == ["settled"]


def test_cancelled_control_during_recovery_keeps_paid_custody(isolated):
    from ouroboros.task_results import write_task_result
    write_task_result(isolated, "stream-task", "running", root_task_id="stream-task")
    reason = [None]
    calls = []
    def send(**kw):
        calls.append(kw)
        reason[0] = "cancelled"
        raise Rejected("temperature unsupported")
    with model_wait.task_model_wait_scope(
        task={"id": "stream-task"}, drive_root=isolated, event_queue=None,
        worker_slot_held=False, owner_control=lambda: reason[0],
    ) as owner:
        assert owner.control_reason() is None
        with pytest.raises(PhysicalDispatchInterrupted) as caught:
            run_driver(send, payload(temperature=0.2), target())
        assert owner.control_reason() == "cancelled"
    assert len(calls) == 1 and caught.value.control_reason == "cancelled"
    assert caught.value.physical_attempt_capture.state == "unresolved"
    assert [row["state"] for row in rows(isolated)] == ["unresolved"]


@pytest.mark.parametrize("no_proxy", [False, True])
def test_native_recovery_rereads_execution_deadline(isolated, monkeypatch, no_proxy):
    import requests
    now = [100]
    monkeypatch.setattr(model_wait, "monotonic_now", lambda slot=None: now[0])
    seen = []
    def post(*a, **kw):
        seen.append(kw)
        now[0] += 3
        if len(seen) == 1:
            raise Rejected("temperature unsupported")
        return SimpleNamespace(status_code=200, json=lambda: native_events()[1])
    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setattr(requests.Session, "post", post)
    with model_wait.execution_deadline_scope(110):
        LLMClient()._chat_anthropic(target("anthropic"), MESSAGES, TOOLS, "high", 1024, "auto",
                                   temperature=0.2, no_proxy=no_proxy)
    assert [kw["timeout"] for kw in seen] == [10, 7]
    assert seen[0]["json"]["max_tokens"] == seen[1]["json"]["max_tokens"] == 1024


@pytest.mark.parametrize("asynchronous", [False, True])
def test_local_and_model_operation_transports_receive_no_stream_option(isolated, monkeypatch, asynchronous):
    import ouroboros.llm_claudexor as operations
    client = LLMClient()
    seen = []
    def local(messages, tools, max_tokens, choice, *, timeout=None, reasoning_effort=None):
        seen.append({"timeout": timeout, "reasoning_effort": reasoning_effort})
        return {"content": "local"}, {}
    def operation(*args, **kwargs):
        seen.append(kwargs)
        return {"content": "operation"}, {}
    async def async_operation(*args, **kwargs):
        return operation(*args, **kwargs)
    monkeypatch.setattr(client, "_chat_local", local)
    monkeypatch.setattr(client, "_resolve_remote_target", lambda model: target("claudexor"))
    monkeypatch.setattr(operations, "chat_claudexor", operation)
    monkeypatch.setattr(operations, "chat_claudexor_async", async_operation)
    for use_local in (True, False):
        kwargs = dict(messages=MESSAGES, model="claudexor::fixture::model", stream=True, use_local=use_local)
        result = asyncio.run(client.chat_async(**kwargs)) if asynchronous else client.chat(**kwargs)
        assert result[0]["content"] == ("local" if use_local else "operation")
    assert all("stream" not in kwargs and "stream_options" not in kwargs for kwargs in seen)


@pytest.mark.serial
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("no_proxy", [False, True])
@pytest.mark.parametrize("terminal", [False, True])
def test_actual_sdk_loopback_sse_and_cleanup(isolated, monkeypatch, asynchronous, no_proxy, terminal):
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    observed = []
    wire = b": fixture comment\n\n" + sse(chunk({"content": "done"}, "stop"),
                                          {"id": "gen-test", "choices": [], "usage": completion()["usage"]}, done=terminal)
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            observed.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("X-Generation-Id", "loopback-generation")
            self.send_header("Content-Length", str(len(wire)))
            self.end_headers()
            self.wfile.write(wire)
            self.wfile.flush()
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = LLMClient()
    route = {**target("openai-compatible"), "base_url": f"http://127.0.0.1:{server.server_port}/v1"}
    monkeypatch.setattr(client, "_resolve_remote_target", lambda model: route)
    try:
        kwargs = dict(messages=MESSAGES, model="openai-compatible::fixture", stream=True, no_proxy=no_proxy, timeout=5)
        if asynchronous:
            async def call_and_close():
                try:
                    return await client.chat_async(**kwargs)
                finally:
                    for sdk in client._async_remote_clients.values():
                        await sdk.close()
            operation = lambda: asyncio.run(call_and_close())
        else:
            operation = lambda: client.chat(**kwargs)
        msg, usage = operation()
        assert msg["content"] == "done"
        assert usage["stream_receipt"]["generation_id"] == "loopback-generation"
        # A body closed cleanly after every choice finished is terminal framing even without
        # ``[DONE]`` (a compatible endpoint that omits it must not turn every reply into an
        # unknown outcome); the missing witness is disclosed in the receipt.
        assert usage["stream_receipt"]["anomalies"]["count"] == (0 if terminal else 1)
        assert observed[0]["stream"] is True and observed[0]["stream_options"]["include_usage"] is True
        assert [row["state"] for row in rows(isolated)] == ["settled"]
    finally:
        for sdk in client._remote_clients.values():
            sdk.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert not thread.is_alive()


@pytest.mark.serial
def test_native_async_cancellation_keeps_worker_until_physical_terminal(isolated, monkeypatch):
    import threading
    import requests
    entered, release = threading.Event(), threading.Event()
    events, expected = native_events()
    response = WireResponse(sse(*events, done=False))
    original_iter = response.iter_content
    def chunks(**kwargs):
        entered.set()
        assert release.wait(5)
        yield from original_iter(**kwargs)
    response.iter_content = chunks
    monkeypatch.setattr(requests, "post", lambda *a, **kw: response)
    client = LLMClient()
    monkeypatch.setattr(client, "_resolve_remote_target", lambda model: target("anthropic"))
    async def call():
        pending = asyncio.create_task(client.chat_async(messages=MESSAGES, model="anthropic::fixture", stream=True))
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            pending.cancel()
            await asyncio.sleep(0)
            assert not pending.done()
            assert_unsettled(isolated)
            release.set()
            with pytest.raises(asyncio.CancelledError) as caught:
                await pending
            from ouroboros.transport_custody import _capture_on_chain
            assert _capture_on_chain(caught.value).state == "settled"
        finally:
            release.set()
    asyncio.run(call())
    assert response.closed and rows(isolated)[-1]["state"] == "settled"
