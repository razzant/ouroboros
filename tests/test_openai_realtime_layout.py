"""Lossless OpenAI field packaging after the observed 256000-character refusal."""
from dataclasses import replace
import asyncio

import pytest

from ouroboros.contracts.realtime import canonical_json
from ouroboros.contracts.live_conversation import LiveOptions, Respond
from tests.test_grok_openai_protocol import providers as providers, deny_network as deny_network, Wire, selected


@pytest.mark.parametrize("role", ["system", "user", "assistant"])
@pytest.mark.parametrize("length", [0, 255999, 256000, 256001, 512001])
def test_item_text_fragments_keep_every_character_role_and_order(providers, role, length):
    text = ("Я🐍\n" * ((length + 2) // 3))[:length]
    parts = ["BEFORE\n", text, "\nAFTER", ""]
    value = replace(selected(), messages_json=canonical_json([
        {"role": "system", "content": "UNCHANGED GOVERNANCE"},
        {"role": role, "content": [{"type": "text", "text": p} for p in parts]},
    ]))
    events, _ = providers["openai"].module.layout(value)
    assert events[0]["session"]["instructions"] == "UNCHANGED GOVERNANCE"
    items = events[1:]
    assert items and all(e["item"]["role"] == role for e in items)
    actual = [p for e in items for p in e["item"]["content"]]
    assert all(len(p["text"]) <= 256000 for p in actual)
    assert all(p["type"] == ("output_text" if role == "assistant" else "input_text") for p in actual)
    assert "".join(p["text"] for p in actual).encode() == "".join(parts).encode()
    assert actual[-1]["text"] == ""  # Even an empty original content part is retained.
    for i, event in enumerate(items):
        assert event["previous_item_id"] == ("root" if i == 0 else items[i - 1]["item"]["id"])
    if length <= 256000:
        assert len(items) == 1 and [p["text"] for p in actual] == parts


def test_chunked_setup_is_fully_acknowledged_before_greeting(providers):
    text = "Large selected memory 🐍\n" * 15000
    value = replace(selected(), messages_json=canonical_json([
        {"role": "system", "content": "UNCHANGED GOVERNANCE"},
        {"role": "system", "content": text},
        {"role": "system", "content": "The owner opened this call."},
    ]))
    wire = Wire()
    adapter = providers["openai"].session(value, LiveOptions(), connect=wire.connect, observe=lambda *_: None)
    async def run():
        try:
            await adapter.open()
            assert not any(e["type"] == "response.create" for e in wire.sent)
            await adapter.send(Respond("The owner opened this call."))
            assert wire.sent[-1]["type"] == "response.create"
            memory = [p["text"] for e in wire.sent if e.get("item", {}).get("role") == "system"
                      for p in e["item"]["content"]]
            assert "".join(memory[:-1]) == text
            assert all(len(p) <= 256000 for p in memory)
        finally:
            await adapter.close()
        assert wire.closed
    asyncio.run(run())
