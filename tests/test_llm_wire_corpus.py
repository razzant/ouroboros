"""Replay of recorded provider wire (``tests/fixtures/llm_wire``) through the stream assembler.

Every ``.sse`` is real SSE recorded from a live route with a synthetic prompt (see the
corpus README for the recording and redaction rules). The assembler must consume each one
without an exception, at any byte fragmentation, and normalize to the same shape the
non-streaming JSON body of the same request takes. The ``.sse``/``.json`` pairs are SEPARATE
live requests, so parity is structural — message keys, tool names, argument key sets, the
``reasoning_details`` type sequence, finish reason and usage key names — never exact text.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from ouroboros.llm import LLMClient
from ouroboros.llm_stream import consume_stream

WIRE = pathlib.Path(__file__).parent / "fixtures" / "llm_wire"
CHUNK_SIZES = (0, 1024, 7)  # whole body, 1 KiB, and 7-byte splits across UTF-8/JSON boundaries

# Per-case expectations, keyed by the fixture's corpus-relative path. Every chat-shaped
# ``.sse`` in the corpus must have a row here (pinned below), so a new recording cannot
# land without stating what its assembly should look like.
EXPECTED = {
    "openrouter/gemini-3.8-flash/tool_stream.sse": {
        "reasoning": ["reasoning.encrypted"], "calls": ["lookup"], "encrypted_id": "call_"},
    "openrouter/gemini-3.8-flash/tool_stream_longreasoning.sse": {
        "reasoning": ["reasoning.text", "reasoning.encrypted"], "calls": ["lookup"], "encrypted_id": "call_"},
    "openrouter/gemini-3.8-flash/multicall_stream.sse": {
        "reasoning": ["reasoning.text", "reasoning.encrypted"], "calls": ["lookup", "lookup"], "encrypted_id": "call_"},
    "openrouter/gemini-3.8-flash/secondturn_stream.sse": {
        "reasoning": ["reasoning.text", "reasoning.encrypted"], "calls": ["lookup"], "encrypted_id": "call_"},
    "openrouter/grok-4.6/tool_stream.sse": {
        "reasoning": ["reasoning.summary", "reasoning.encrypted"], "calls": ["lookup"], "encrypted_id": "rs_"},
    "openrouter/gpt-5.6-sol/tool_stream.sse": {
        "reasoning": ["reasoning.encrypted"], "calls": ["lookup"], "encrypted_id": "rs_"},
    "openrouter/gpt-5.6-sol/multi_stream.sse": {
        "reasoning": ["reasoning.encrypted"], "calls": ["lookup", "lookup"], "encrypted_id": "rs_"},
    "openrouter/claude-sonnet-5/tool_stream.sse": {
        "reasoning": ["reasoning.text"], "calls": ["lookup", "lookup"]},
    "openrouter/claude-opus-5.5/twoblock_stream.sse": {
        "reasoning": ["reasoning.text", "reasoning.text"], "calls": ["run_command", "search_code"],
        "args": [{"cmd", "timeout_sec"}, {"max_results", "path", "query", "regex", "root"}]},
    "openrouter/claude-fable-5/refusal_stream.sse": {
        "reasoning": [], "calls": [], "finish": "content_filter", "refusal": True},
    "openrouter/claude-fable-5/refusal_stream_v2.sse": {
        "reasoning": [], "calls": [], "finish": "content_filter", "refusal": True},
    "openai/gpt-5.6-terra/custom_stream.sse": {
        "reasoning": [], "calls": ["lookup"], "kinds": ["custom"]},
    "openai/gpt-5.6-terra/function_none_stream.sse": {
        "reasoning": [], "calls": ["lookup", "lookup"]},
}
NATIVE = "anthropic/claude-sonnet-5/native_stream.sse"


def _rel(path: pathlib.Path) -> str:
    return path.relative_to(WIRE).as_posix()


def _streams():
    return sorted(path for path in WIRE.rglob("*.sse") if _rel(path) != NATIVE)


def _pairs():
    out = []
    for stream in sorted(WIRE.rglob("*_stream.sse")):
        sibling = stream.with_name(stream.name.replace("_stream.sse", "_nonstream.json"))
        if sibling.exists():
            out.append((stream, sibling))
    return out


class _Wire:
    """A minimal streamed HTTP response: raw bytes handed out in fixed-size pieces."""

    status_code = 200

    def __init__(self, raw: bytes, size: int):
        self.raw, self.size = raw, size
        self.headers: dict = {}
        self.closed = False

    def iter_bytes(self):
        step = self.size or len(self.raw)
        for offset in range(0, len(self.raw), step):
            yield self.raw[offset:offset + step]

    def iter_content(self, chunk_size=8192):
        return self.iter_bytes()

    def close(self):
        self.closed = True


def _target(path: pathlib.Path) -> dict:
    route, model = path.relative_to(WIRE).parts[:2]
    return {"provider": route, "resolved_model": model, "usage_model": model,
            "base_url": "https://provider.invalid/v1", "supports_openrouter_extensions": route == "openrouter",
            "supports_generation_cost": False, "api_key": "fixture"}


@pytest.fixture(autouse=True)
def _no_pricing_lookup(monkeypatch):
    import ouroboros.pricing as pricing

    monkeypatch.setattr(pricing, "estimate_cost_optional", lambda *a, **k: None)


def _assemble(path: pathlib.Path, size: int, *, native: bool = False) -> tuple[dict, dict]:
    response = _Wire(path.read_bytes(), size)
    body = consume_stream(response, native=native).model_dump()
    assert response.closed
    return body, body.pop("_stream_receipt")


def _normalized(path: pathlib.Path, body: dict) -> tuple[dict, dict]:
    return LLMClient()._normalize_remote_response(body, _target(path), skip_cost_fetch=True)


def _call_name(call: dict) -> str:
    return call[call["type"]]["name"]


def _call_args(call: dict) -> dict:
    return json.loads(call[call["type"]]["arguments" if call["type"] == "function" else "input"])


def test_every_recorded_chat_stream_has_expectations():
    assert {_rel(path) for path in _streams()} == set(EXPECTED)


@pytest.mark.parametrize("size", CHUNK_SIZES)
@pytest.mark.parametrize("path", _streams(), ids=_rel)
def test_recorded_chat_stream_assembles_completely(path, size):
    expected = EXPECTED[_rel(path)]
    body, receipt = _assemble(path, size)
    assert receipt["anomalies"] == {"count": 0, "first": []}, "a recorded wire is clean; irregularities would be disclosed here"
    assert receipt["complete"] is True
    for choice in body["choices"]:
        assert choice["finish_reason"]
        for call in choice["message"].get("tool_calls") or []:
            assert call["id"] and call["type"] in {"function", "custom"}
            assert _call_name(call)
            assert isinstance(call[call["type"]]["arguments" if call["type"] == "function" else "input"], str)
    msg, usage = _normalized(path, {**body, "_stream_receipt": receipt})
    assert usage["stream_receipt"]["anomalies"]["count"] == 0
    calls = msg.get("tool_calls") or []
    assert [_call_name(call) for call in calls] == expected["calls"]
    assert [call["type"] for call in calls] == expected.get("kinds", ["function"] * len(calls))
    assert [set(_call_args(call)) for call in calls] == expected.get("args", [{"query"}] * len(calls))
    details = msg.get("reasoning_details") or []
    assert [detail["type"] for detail in details] == expected["reasoning"]
    if "encrypted_id" in expected:
        encrypted = [detail for detail in details if detail["type"] == "reasoning.encrypted"]
        assert encrypted and all(detail["id"].startswith(expected["encrypted_id"]) for detail in encrypted)
    assert usage["response_finish_reason"] == expected.get("finish", "tool_calls")
    if expected.get("refusal"):
        assert msg["refusal"] and not calls
    if "usage" in body:
        assert usage["prompt_tokens"] and usage["completion_tokens"]


def _wire_opaque_values(path: pathlib.Path) -> list:
    values = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("data: ") and line != "data: [DONE]":
            for choice in json.loads(line[6:]).get("choices") or []:
                for item in (choice.get("delta") or {}).get("reasoning_details") or []:
                    values += [item[key] for key in ("signature", "data") if isinstance(item.get(key), str) and item[key]]
    return values


@pytest.mark.parametrize("path", _streams(), ids=_rel)
def test_each_opaque_wire_payload_is_one_whole_record_value(path):
    """A signature or encrypted payload is minted whole: each arrives in one delta and
    must stay one record's entire value, never fused with a neighbour's."""
    details = _assemble(path, 0)[0]["choices"][0]["message"].get("reasoning_details") or []
    assembled = [detail[key] for detail in details for key in ("signature", "data") if detail.get(key)]
    assert sorted(assembled) == sorted(_wire_opaque_values(path))


@pytest.mark.parametrize("path", _streams(), ids=_rel)
def test_recorded_chat_stream_assembly_is_fragmentation_invariant(path):
    bodies = [_assemble(path, size)[0] for size in CHUNK_SIZES]
    assert all(body == bodies[0] for body in bodies)


@pytest.mark.parametrize("stream,sibling", _pairs(), ids=lambda p: _rel(p) if p.suffix == ".sse" else "")
def test_recorded_stream_matches_its_nonstream_sibling_structurally(stream, sibling):
    if _rel(stream) == NATIVE:
        pytest.skip("native Messages parity is asserted by test_recorded_native_stream_assembles_blocks")
    text = sibling.read_text(encoding="utf-8")
    json_body = json.loads(text[text.index("{"):])
    streamed_msg, streamed_usage = _normalized(stream, _assemble(stream, 0)[0])
    json_msg, json_usage = _normalized(stream, json_body)
    assert set(streamed_msg) == set(json_msg)
    streamed_calls, json_calls = streamed_msg.get("tool_calls") or [], json_msg.get("tool_calls") or []
    assert [_call_name(call) for call in streamed_calls] == [_call_name(call) for call in json_calls]
    assert [set(_call_args(call)) for call in streamed_calls] == [set(_call_args(call)) for call in json_calls]
    assert ([detail["type"] for detail in streamed_msg.get("reasoning_details") or []]
            == [detail["type"] for detail in json_msg.get("reasoning_details") or []])
    # Per-record key sets: a streamed record that lost its opaque continuation payload
    # (``signature``/``data``/``id``) would differ from the non-stream sibling here.
    assert ([sorted(detail) for detail in streamed_msg.get("reasoning_details") or []]
            == [sorted(detail) for detail in json_msg.get("reasoning_details") or []])
    assert streamed_usage["response_finish_reason"] == json_usage["response_finish_reason"]
    assert set(streamed_usage) - {"stream_receipt"} == set(json_usage)


@pytest.mark.parametrize("size", CHUNK_SIZES)
def test_recorded_native_stream_assembles_blocks(size):
    """Direct Anthropic Messages SSE (thinking + text + two tool_use blocks), and its
    structural parity with the non-streaming body of the same request."""
    stream = WIRE / NATIVE
    body, receipt = _assemble(stream, size, native=True)
    assert receipt["anomalies"] == {"count": 0, "first": []} and receipt["complete"] is True
    blocks = body["content"]
    assert [block["type"] for block in blocks] == ["thinking", "text", "tool_use", "tool_use"]
    assert isinstance(blocks[0]["signature"], str) and blocks[0]["signature"]
    assert blocks[1]["text"].startswith("The two most relevant terms")
    assert [(block["name"], block["input"]) for block in blocks[2:]] == [
        ("lookup", {"query": "oura"}), ("lookup", {"query": "boros"})]
    assert body["stop_reason"] == "tool_use"
    assert body["usage"]["output_tokens"] == 186 and body["usage"]["input_tokens"] == 482
    json_body = json.loads((WIRE / "anthropic/claude-sonnet-5/native_nonstream.json").read_text(encoding="utf-8"))
    json_blocks = json_body["content"]
    assert [block["type"] for block in json_blocks] == [block["type"] for block in blocks]
    assert [block["name"] for block in json_blocks if block["type"] == "tool_use"] == ["lookup", "lookup"]
    assert [set(block["input"]) for block in json_blocks if block["type"] == "tool_use"] == [{"query"}, {"query"}]
    assert json_body["stop_reason"] == body["stop_reason"]
    assert set(json_body["usage"]) == set(body["usage"])
