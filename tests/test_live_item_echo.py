"""OpenAI's 2026-10-26 image echo, also applied to xAI's shared item dialect.

The item shape is from checks/probe-image/echo.json (ids and image bytes replaced).
No xAI image probe was made; these tests do not enable Grok's unsupported vision.
"""
import asyncio
import copy

import pytest

from ouroboros.contracts.live_conversation import Video
from ouroboros.contracts.realtime import RealtimeSetupError
from tests.test_grok_openai_protocol import (
    Wire, deny_network as deny_network, drain, make, providers as providers, selected,
)


SENT = {"id": "probe_item", "type": "message", "role": "user", "content": [
    {"type": "input_text", "text": "probe frame"},
    {"type": "input_image", "image_url": "data:image/png;base64,cHJvYmU="},
]}
ECHO = {"id": "probe_item", "type": "message", "status": "completed", "role": "user", "content": [
    {"type": "input_text", "text": "probe frame"}, {"type": "input_image"},
]}


class ImageEchoWire(Wire):
    fault = ""

    def put(self, event):
        if event.get("type") not in {"conversation.item.added", "conversation.item.done"}:
            return super().put(event)
        event = copy.deepcopy(event)
        item = event["item"]
        item["status"] = "completed"
        for part in item.get("content", []):
            if part.get("type") == "input_image":
                part.pop("image_url", None)
        if self.fault == "text":
            item["content"][0]["text"] = "changed words"
        elif self.fault == "missing_part":
            item["content"].pop()
        elif self.fault == "order":
            item["content"].reverse()
        elif self.fault == "role":
            item["role"] = "assistant"
        # The observed OpenAI pair; xAI accepts added with completed status.
        for kind in ("conversation.item.added", "conversation.item.done"):
            super().put({**event, "type": kind})


def test_recorded_image_echo_trips_the_old_whole_content_comparison():
    for kind in ("conversation.item.added", "conversation.item.done"):
        event = {"type": kind, "previous_item_id": None, "item": ECHO}
        with pytest.raises(RealtimeSetupError, match="item_mismatch"):
            if any(event["item"].get(k) != v for k, v in SENT.items()):
                raise RealtimeSetupError("item_mismatch", "OpenAI changed host context or tool output")


@pytest.mark.parametrize("engine", ["openai", "grok"])
@pytest.mark.parametrize("phase", ["setup", "live"])
@pytest.mark.parametrize("fault", ["", "text", "missing_part", "order", "role"])
def test_image_echo_keeps_parts_text_and_item_authority(providers, engine, phase, fault):
    async def run():
        wire = ImageEchoWire(engine)
        adapter, _ = make(providers, engine, wire=wire,
                          value=selected([{"role": "system", "content": "Rules."}]))
        outgoing = {"type": "conversation.item.create", "previous_item_id": None, "item": copy.deepcopy(SENT)}
        try:
            if phase == "setup":
                # Exercise the native acknowledgement boundary, including Grok's
                # comparator, without claiming its text-only mapper accepts video.
                adapter.events.append(outgoing)
            else:
                await adapter.open()
                await drain(adapter, wire)
                adapter.outstanding[SENT["id"]] = copy.deepcopy(SENT)
            wire.fault = fault

            async def acknowledge():
                if phase == "setup":
                    await adapter.open()
                else:
                    await adapter.write(outgoing)
                await drain(adapter, wire)

            if fault:
                detail = "selected input" if phase == "setup" else "host context or tool output"
                with pytest.raises(RealtimeSetupError, match=f"item_mismatch.*changed {detail}"):
                    await asyncio.wait_for(acknowledge(), 2)
            else:
                await asyncio.wait_for(acknowledge(), 2)
                assert not adapter.outstanding and not wire.closed
                assert [event["item"] for event in wire.received if event.get("item")] == [ECHO, ECHO]
        finally:
            await adapter.close()
    asyncio.run(run())


def test_openai_camera_frame_survives_payload_free_acknowledgements(providers):
    async def run():
        wire = ImageEchoWire()
        adapter, _ = make(providers, "openai", wire=wire)
        try:
            await adapter.open()
            await drain(adapter, wire)
            await adapter.send(Video(b"synthetic-image", "image/png"))
            await drain(adapter, wire)
            assert not adapter.outstanding and not wire.closed
            assert wire.sent[-1]["item"]["content"][0]["type"] == "input_image"
            assert not any(event["type"] == "response.create" for event in wire.sent)
        finally:
            await adapter.close()
    asyncio.run(run())


@pytest.mark.parametrize("engine", ["openai", "grok"])
@pytest.mark.parametrize("sent,actual", [
    ({"type": "function_call_output", "call_id": "c", "output": "exact"},
     {"type": "function_call_output", "call_id": "c", "output": "changed"}),
    ({"type": "message", "role": "user", "content": [{"type": "input_audio", "audio": "exact"}]},
     {"type": "message", "role": "user", "content": [{"type": "input_audio"}]}),
])
def test_tool_output_and_audio_payloads_still_require_exact_echo(providers, engine, sent, actual):
    adapter, _ = make(providers, engine)
    adapter.outstanding["strict"] = {**sent, "id": "strict"}
    with pytest.raises(RealtimeSetupError, match="item_mismatch.*host context or tool output"):
        adapter.convert({"type": "conversation.item.done", "item": {**actual, "id": "strict"}})
