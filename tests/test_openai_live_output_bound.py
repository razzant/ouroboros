"""OpenAI response-bound grammar; connection reservations are retired."""
from __future__ import annotations

import asyncio
import copy
import json
from dataclasses import asdict, replace

import pytest

from ouroboros.contracts.live_conversation import LiveOptions
from ouroboros.contracts.realtime import RealtimeSetupError
from tests.test_grok_openai_protocol import providers as providers, deny_network as deny_network, Wire, selected


class OutputWire(Wire):
    """A full-session echo, with the documented output field grammar enforced."""

    def __init__(self, changed_at=None):
        super().__init__("openai")
        self.changed_at = changed_at
        self.acks = 0

    async def send(self, raw):
        event = json.loads(raw)
        value = (event.get("session") or event.get("response") or {}).get("max_output_tokens")
        if value is not None:
            assert value == "inf" or type(value) is int and 1 <= value <= 4096
        await super().send(raw)

    def put(self, event):
        if event.get("type") == "session.updated":
            self.acks += 1
            if self.acks == self.changed_at:
                event = copy.deepcopy(event)
                event["session"]["max_output_tokens"] = 4096
        super().put(event)


@pytest.mark.parametrize("allowance", [1, 256, 4096, 4097, 8192, 50000])
def test_openai_wire_field_preserves_the_ordinary_allowance(providers, allowance):
    value = replace(selected(), output_allowance_tokens=allowance)
    original = asdict(value)
    wire = OutputWire()
    adapter = providers["openai"].session(
        value, LiveOptions(), connect=wire.connect, observe=lambda *_: None)

    async def run():
        try:
            await adapter.open()
            assert (await adapter.receive())[0].kind == "ready"
        finally:
            await adapter.close()
    asyncio.run(run())
    assert asdict(value) == original and adapter.selected is value
    expected = allowance if allowance <= 4096 else "inf"
    assert adapter.wire["events"][0]["session"]["max_output_tokens"] == expected
    assert wire.sent[0]["session"]["max_output_tokens"] == expected
    assert adapter.output_bound_tokens == (allowance if allowance <= 4096 else None)
    assert wire.closed


@pytest.mark.parametrize("changed_at", [1, 2, 3])
def test_openai_output_field_ack_must_match_at_setup_vad_and_later_updates(providers, changed_at):
    wire = OutputWire(changed_at)
    adapter = providers["openai"].session(
        replace(selected(), output_allowance_tokens=8192), LiveOptions(model_output_bound_tokens=50000),
        connect=wire.connect, observe=lambda *_: None)

    async def run():
        try:
            with pytest.raises(RealtimeSetupError, match="session_mismatch.*max_output_tokens"):
                await adapter.open()
                await adapter.receive()  # ready
                await adapter.write({"type": "session.update", "session": {"type": "realtime"}})
                await adapter.receive()
        finally:
            await adapter.close()
    asyncio.run(run())
    assert wire.closed
    assert not any(row.get("type") == "response.create" for row in wire.sent)
