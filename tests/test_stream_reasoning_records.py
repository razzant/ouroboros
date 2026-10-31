"""A signature closes the reasoning record it signs.

OpenRouter streams an Anthropic answer's thinking blocks as id-less ``reasoning.text``
deltas, each block ending in a signature-only delta. The frames below keep the shape
recorded from a live ``anthropic/claude-opus-5.5`` route (two blocks, ``index`` 0 then 1,
then the tool calls); a fused record carrying both texts and both signatures is refused
by every upstream ("Invalid `signature` in `thinking` block"), the split pair is accepted.
"""

from __future__ import annotations

import json

import pytest

from ouroboros.llm import LLMClient
from ouroboros.llm_stream import ChatAccumulator, consume_stream
from tests.test_transport_b_stream_deadlines import chunk, sse

ANTHROPIC = "anthropic-claude-v1"


@pytest.fixture(autouse=True)
def _no_pricing_lookup(monkeypatch):
    import ouroboros.pricing as pricing

    monkeypatch.setattr(pricing, "estimate_cost_optional", lambda *a, **k: None)


CALL = {"index": 0, "id": "toolu_01", "type": "function", "function": {"name": "lookup", "arguments": ""}}


def text_delta(text, index, fmt=ANTHROPIC):
    return {"content": "", "role": "assistant", "reasoning": text,
            "reasoning_details": [{"type": "reasoning.text", "text": text, "format": fmt, "index": index}]}


def signature_delta(signature, index):
    return {"content": "", "role": "assistant",
            "reasoning_details": [{"type": "reasoning.text", "signature": signature, "format": ANTHROPIC, "index": index}]}


def two_blocks(second_index=1):
    return [text_delta("Weigh the roots", 0), text_delta(" first.\n\n", 0), signature_delta("CAQS-first", 0),
            text_delta("Next I look both up.", second_index), signature_delta("CAQS-second", second_index),
            {"content": None, "role": "assistant", "tool_calls": [CALL]},
            {"content": None, "role": "assistant", "tool_calls": [{"index": 0, "function": {"arguments": '{"query":"oura"}'}}]}]


def assemble(deltas, finish="tool_calls"):
    accumulator = ChatAccumulator()
    for delta in deltas:
        accumulator.accept("", json.dumps(chunk(delta)))
    accumulator.accept("", json.dumps(chunk(finish=finish, usage={"prompt_tokens": 9, "completion_tokens": 7})))
    accumulator.accept("", "[DONE]")
    return accumulator.result()["choices"][0]["message"], accumulator.anomaly_facts()


@pytest.mark.parametrize("second_index", [1, 0], ids=["recorded-index", "repeated-index"])
def test_each_signed_block_stays_its_own_record(second_index):
    message, anomalies = assemble(two_blocks(second_index))
    assert message["reasoning_details"] == [
        {"type": "reasoning.text", "text": "Weigh the roots first.\n\n", "format": ANTHROPIC, "index": 0,
         "signature": "CAQS-first"},
        {"type": "reasoning.text", "text": "Next I look both up.", "format": ANTHROPIC, "index": second_index,
         "signature": "CAQS-second"},
    ]
    assert message["reasoning"] == "Weigh the roots first.\n\nNext I look both up."
    assert message["tool_calls"][0]["function"] == {"name": "lookup", "arguments": '{"query":"oura"}'}
    assert anomalies == {"count": 0, "first": []}


def test_signature_only_blocks_stay_separate_records():
    """Thinking returned without text (``display: omitted``) is a signature per block."""
    message, _ = assemble([signature_delta("CAQS-first", 0), signature_delta("CAQS-second", 1),
                           {"content": None, "role": "assistant", "tool_calls": [CALL]}])
    assert [detail.get("signature") for detail in message["reasoning_details"]] == ["CAQS-first", "CAQS-second"]


def test_a_contentless_item_after_the_signature_joins_the_signed_record():
    trailing = {"reasoning_details": [{"type": "reasoning.text", "text": "", "format": ANTHROPIC, "index": 0}]}
    message, _ = assemble([text_delta("Only thought.", 0), signature_delta("CAQS-only", 0), trailing,
                           {"content": "done"}], finish="stop")
    assert message["reasoning_details"] == [
        {"type": "reasoning.text", "text": "Only thought.", "format": ANTHROPIC, "index": 0, "signature": "CAQS-only"}]


@pytest.mark.parametrize("fmt", ["unknown", "google-gemini-v1"])
def test_unsigned_text_fragments_still_reassemble_into_one_record(fmt):
    message, _ = assemble([text_delta("part one, ", 0, fmt), text_delta("part two", 0, fmt), {"content": "done"}],
                          finish="stop")
    assert [(detail["text"], detail.get("signature")) for detail in message["reasoning_details"]] == [
        ("part one, part two", None)]


class _Wire:
    def __init__(self, raw, size):
        self.raw, self.size, self.headers = raw, size, {}

    def iter_bytes(self):
        step = self.size or len(self.raw)
        for offset in range(0, len(self.raw), step):
            yield self.raw[offset:offset + step]

    def close(self):
        pass


def test_split_records_survive_fragmentation_and_normalize_for_the_next_send():
    raw = sse(*(chunk(delta) for delta in two_blocks()),
              chunk(finish="tool_calls", usage={"prompt_tokens": 9, "completion_tokens": 7}))
    bodies = []
    for size in (0, 1, 7, 64):
        body = consume_stream(_Wire(raw, size)).model_dump()
        body.pop("_stream_receipt")
        bodies.append(body)
    assert all(body == bodies[0] for body in bodies)
    client = LLMClient(api_key="fixture-not-live")
    target = client._resolve_remote_target("anthropic/claude-opus-5.5")
    message, _usage = client._normalize_remote_response(bodies[0], target, skip_cost_fetch=True)
    assert [detail["signature"] for detail in message["reasoning_details"]] == ["CAQS-first", "CAQS-second"]
    history = [{"role": "user", "content": "Where does the word come from?"}, message,
               {"role": "tool", "tool_call_id": "toolu_01", "content": "oura: tail"}]
    payload = client._build_remote_kwargs(target, history, "high", 1024, "auto", None, None,
                                          skip_capability_fetch=True)
    sent = payload["messages"][1]
    assert sent["reasoning_details"] == message["reasoning_details"]
    assert "reasoning" not in sent  # the flat text equals the records' joined text and leaves the send copy
