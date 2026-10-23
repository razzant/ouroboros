"""Provider contract consumers with fake sockets; no host/probe compatibility shim."""
from __future__ import annotations

import asyncio
import copy
import importlib.util
import json
import socket
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from ouroboros.contracts.live_conversation import Audio, LiveOptions, Respond, ToolOutputs
from ouroboros.contracts.realtime import RealtimeInput, USAGE_AXES, canonical_json


@pytest.fixture(autouse=True)
def deny_network(monkeypatch):
    def refused(*args, **kwargs):
        raise AssertionError("Provider tests use fake sockets only")
    monkeypatch.setattr(socket.socket, "connect", refused)
    monkeypatch.setattr(socket, "create_connection", refused)


@pytest.fixture
def providers(monkeypatch):
    result = {}
    for engine in ("grok", "openai"):
        directory = Path(__file__).resolve().parents[1] / "skills" / (engine + "_realtime")
        name = "_test_voice_" + engine
        spec = importlib.util.spec_from_file_location(name, directory / "plugin.py", submodule_search_locations=[str(directory)])
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        registrations = {}
        api = SimpleNamespace(register_realtime_provider=lambda name, **kw: registrations.update(kw),
                              get_settings=lambda _: pytest.fail("Injected sockets never read credentials"))
        module.register(api)
        result[engine] = SimpleNamespace(module=module, **registrations)
    yield result
    for name in list(sys.modules):
        if name.startswith("_test_voice_") and "." in name:
            sys.modules.pop(name)


def selected(messages=None, tools=()):
    return RealtimeInput("synthetic-model", canonical_json(messages if messages is not None else [
        {"role": "system", "content": "Host rules."},
        {"role": "system", "content": "The owner opened this call and can hear you."},
    ]), canonical_json(tools), 256, "nano", "core", "view")


class Wire:
    def __init__(self, engine="openai"):
        self.engine = engine
        self.sent, self.received, self.session = [], [], {}
        self.events = asyncio.Queue()
        self.closed = False
        self.after_close = lambda: None

    def put(self, event):
        self.events.put_nowait(copy.deepcopy(event))

    async def recv(self):
        value = await self.events.get()
        self.received.append(value)
        return canonical_json(value)

    async def send(self, raw):
        event = json.loads(raw)
        self.sent.append(event)
        if event["type"] == "session.update":
            for key, value in event["session"].items():
                if key == "audio":
                    for direction, fields in value.items():
                        self.session.setdefault("audio", {}).setdefault(direction, {}).update(fields)
                else:
                    self.session[key] = value
            self.put({"type": "session.updated", "session": self.session})
        elif event["type"] == "conversation.item.create":
            kind = "conversation.item.added" if self.engine == "grok" else "conversation.item.done"
            self.put({"type": kind, "item": event["item"],
                      "previous_item_id": event.get("previous_item_id")})
        elif event["type"] == "conversation.item.truncate":
            self.put({"type": "conversation.item.truncated", "item_id": event["item_id"],
                      "audio_end_ms": event["audio_end_ms"]})

    @asynccontextmanager
    async def connect(self, model):
        try:
            yield self
        finally:
            self.after_close()
            self.closed = True


def make(providers, engine, options=None, value=None, wire=None, observe=None):
    wire = wire or Wire(engine)
    adapter = providers[engine].session(value or selected(), options or LiveOptions(),
        connect=wire.connect, observe=observe or (lambda *_: None))
    return adapter, wire


async def drain(adapter, wire):
    while not wire.events.empty():
        await adapter.receive()


@pytest.mark.parametrize("engine", ["grok", "openai"])
def test_greeting_waits_for_setup_and_never_invents_owner_words(providers, engine):
    async def run():
        adapter, wire = make(providers, engine)
        try:
            await adapter.send(Respond("The owner opened this call and can hear you."))
            assert not wire.sent
            await adapter.open()
            assert (await adapter.receive())[0].kind == "ready"
            assert [e["type"] for e in wire.sent].count("response.create") == 1
            assert wire.sent[-1]["type"] == "response.create"
            assert not any(e.get("item", {}).get("role") == "user" for e in wire.sent)
        finally:
            await adapter.close()
    asyncio.run(run())


@pytest.mark.parametrize("engine", ["grok", "openai"])
@pytest.mark.parametrize("languages", [(), ("ru-RU", "en-US")])
def test_options_preserve_host_text_and_add_language_hints(providers, engine, languages):
    async def run():
        adapter, wire = make(providers, engine, LiveOptions(instructions="Feature then owner rules.", languages=languages, voice="eve" if engine == "grok" else "marin"))
        try:
            await adapter.open()
            session = wire.sent[0]["session"]
            text = session["instructions"]
            if engine == "openai":
                text += "\n\n" + "\n\n".join(part["text"] for event in wire.sent
                    if event.get("item", {}).get("role") == "system" for part in event["item"]["content"])
            assert text.startswith("Host rules.")
            assert "Feature then owner rules." in text
            assert text.index("The owner opened this call") < text.index("Feature then owner rules.")
            if engine == "grok":
                assert "languages" not in session and "language" not in session
                assert session["voice"] == "eve"
                assert ("Speech recognition should expect: ru-RU, en-US." in text) is bool(languages)
                if languages:
                    assert text.index("Feature then owner rules.") < text.index("Speech recognition")
            else:
                transcription = session["audio"]["input"]["transcription"]
                assert transcription.get("language") == ("ru" if languages else None)
                assert ("en-US" in text) is bool(languages)
                assert session["audio"]["output"]["voice"] == "marin"
                assert session["truncation"] == "disabled"
        finally:
            await adapter.close()
    asyncio.run(run())


@pytest.mark.parametrize("pause,ms", [("short", 300), ("normal", 500), ("long", 1000)])
@pytest.mark.parametrize("interruptions", [False, True])
def test_openai_turn_options_keep_responses_host_driven(providers, pause, ms, interruptions):
    async def run():
        adapter, wire = make(providers, "openai", LiveOptions(pause=pause, interruptions=interruptions))
        try:
            await adapter.open()
            vad = wire.session["audio"]["input"]["turn_detection"]
            assert vad == {"type": "server_vad", "silence_duration_ms": ms,
                           "create_response": False, "interrupt_response": interruptions}
            wire.put({"type": "input_audio_buffer.speech_started", "item_id": "owner"})
            events = await adapter.receive()
            if events[0].kind == "ready":
                events = await adapter.receive()
            assert any(e.kind == "input_started" for e in events)
            assert any(e.kind == "speech_started" for e in events) is interruptions
        finally:
            await adapter.close()
    asyncio.run(run())


@pytest.mark.parametrize("engine", ["grok", "openai"])
@pytest.mark.parametrize("silent", [(), ("a",), ("a", "b")])
def test_tool_outputs_only_open_response_for_nonsilent_calls(providers, engine, silent):
    async def run():
        adapter, wire = make(providers, engine)
        try:
            await adapter.open()
            await adapter.receive()
            adapter.waiting_tools.update(("a", "b"))
            tool = {"type": "function", "function": {"name": "new_tool", "parameters": {"type": "object"}}}
            await adapter.send(ToolOutputs(tuple({"role": "tool", "tool_call_id": cid, "content": "result"} for cid in ("a", "b")), (tool,), silent=silent))
            assert not any(e["type"] == "response.create" for e in wire.sent)
            await drain(adapter, wire)
            assert [e["type"] for e in wire.sent].count("response.create") == (0 if len(silent) == 2 else 1)
            assert len([e for e in wire.sent if e.get("item", {}).get("type") == "function_call_output"]) == 2
            assert adapter.tools == (tool,) and not adapter.waiting_tools
        finally:
            await adapter.close()
    asyncio.run(run())


@pytest.mark.parametrize("engine", ["grok", "openai"])
@pytest.mark.parametrize("status", ["completed", "cancelled", "failed", "incomplete"])
def test_response_usage_has_contract_axes_status_and_deduplicates(providers, engine, status):
    async def run():
        adapter, wire = make(providers, engine)
        raw = {"input_tokens": 20, "output_tokens": 9,
               "input_token_details": {"text_tokens": 4, "audio_tokens": 10, "image_tokens": 6, "cached_tokens": 3},
               "output_token_details": {"text_tokens": 2, "audio_tokens": 7}}
        event = {"type": "response.done", "response": {"id": "r", "status": status, "usage": raw, "output": []}}
        if engine == "grok":
            event["usage"] = raw
        try:
            await adapter.open()
            await adapter.receive()
            wire.put(event)
            events = await adapter.receive()
            done = next(e.data for e in events if e.kind == "response_done")
            assert done["response_id"] == "r" and done["status"] == status
            assert done["raw_usage"] == raw
            assert set(done["usage"]) <= set(USAGE_AXES)
            assert done["usage"]["text_input_tokens"] == 4
            assert done["usage"]["audio_output_tokens"] == 7
            assert done["usage"]["cached_input_tokens"] == 3
            if status in {"failed", "incomplete"}:
                assert events[-1].kind == "closed"
                assert events[-1].data == {"reason": status, "resumable": False}
            wire.put(event)
            assert await adapter.receive() == []
        finally:
            await adapter.close()
    asyncio.run(run())


@pytest.mark.parametrize("engine", ["grok", "openai"])
@pytest.mark.parametrize("details,expected", [(None, "cancelled"), ({"reason": "turn_detected"}, "interrupted")])
def test_cancelled_response_keeps_usage_with_nullable_status_details(providers, engine, details, expected):
    async def run():
        adapter, wire = make(providers, engine)
        try:
            await adapter.open()
            await adapter.receive()
            wire.put({"type": "response.done", "response": {"id": "r", "status": "cancelled",
                "status_details": details, "usage": {"output_token_details": {"audio_tokens": 2}}}})
            events = await adapter.receive()
            assert len(events) == 1 and events[0].kind == "response_done"
            assert events[0].data["status"] == expected
            assert events[0].data["usage"]["audio_output_tokens"] == 2
        finally:
            await adapter.close()
    asyncio.run(run())


def test_openai_renewal_waits_for_transcripts_tools_and_acks(providers):
    async def run():
        adapter, wire = make(providers, "openai")
        try:
            await adapter.open()
            await adapter.receive()
            assert adapter.renewal_ready
            await adapter.send(Audio(b"\x00\x00" * 20, 24000))
            assert adapter.renewal_ready  # silent frames are not speech evidence
            for event in (
                {"type": "input_audio_buffer.speech_started", "item_id": "owner"},
                {"type": "input_audio_buffer.speech_stopped", "item_id": "owner"},
                {"type": "input_audio_buffer.committed", "item_id": "owner"},
                {"type": "response.created", "response": {"id": "r"}},
                {"type": "response.output_audio.delta", "item_id": "spoken", "delta": "AAA="},
                {"type": "response.done", "response": {"id": "r", "status": "completed", "output": [
                    {"type": "function_call", "call_id": "c", "name": "end_call", "arguments": "{}"}]}},
                {"type": "conversation.item.input_audio_transcription.completed", "item_id": "owner", "transcript": "Bye"},
                {"type": "response.output_audio_transcript.done", "item_id": "spoken", "transcript": "Goodbye"},
            ):
                wire.put(event)
                await adapter.receive()
                assert not adapter.renewal_ready
            await adapter.send(ToolOutputs(({"role": "tool", "tool_call_id": "c", "content": "done"},), (), silent=("c",)))
            assert not adapter.renewal_ready
            await drain(adapter, wire)
            assert adapter.renewal_ready
        finally:
            await adapter.close()
    asyncio.run(run())


def test_grok_connected_time_includes_idle_and_closing_tail(providers, monkeypatch):
    async def run():
        now, observations = [10.0], []
        monkeypatch.setattr(providers["grok"].module, "monotonic", lambda: now[0])
        adapter, wire = make(providers, "grok", observe=lambda direction, raw: observations.append((direction, json.loads(raw))))
        await adapter.open()
        await adapter.receive()
        now[0] = 14.0
        assert await asyncio.wait_for(adapter.receive(), 2) == []
        samples = [v["usage"]["connected_seconds"] for d, v in observations if d == "usage"]
        assert samples[-1] == 4
        now[0] = 17.0
        wire.put({"type": "response.done", "response": {"id": "r", "status": "completed"}})
        done = (await adapter.receive())[0].data
        assert done["usage"]["connected_seconds"] == 7
        wire.after_close = lambda: now.__setitem__(0, 19.0)
        await adapter.close()
        samples = [v["usage"]["connected_seconds"] for d, v in observations if d == "usage"]
        assert samples[-1] == 9
        await adapter.close()
        assert len([1 for d, v in observations if d == "usage"]) == len(samples)
    asyncio.run(run())


@pytest.mark.parametrize("engine,levels", [("grok", ("none", "high")),
                                         ("openai", ("minimal", "low", "medium", "high", "xhigh"))])
def test_catalog_thinking_matches_documented_wire_and_default(providers, engine, levels):
    async def run():
        model = providers[engine].catalog.model()
        assert model.thinking_levels == levels and "thinking" in model.honours
        for level in ("", *levels):
            adapter, wire = make(providers, engine, LiveOptions(thinking=level))
            try:
                await adapter.open()
                assert wire.session.get("reasoning") == ({"effort": level} if level else None)
                assert ("voice" not in wire.session if engine == "grok" else "voice" not in wire.session["audio"]["output"])
            finally:
                await adapter.close()
    asyncio.run(run())


@pytest.mark.parametrize("engine", ["grok", "openai"])
def test_unknown_usage_stays_absent_and_tool_only_response_keeps_charge(providers, engine):
    async def run():
        adapter, wire = make(providers, engine)
        try:
            await adapter.open()
            await adapter.receive()
            wire.put({"type": "response.done", "response": {"id": "tool-only", "status": "completed",
                "usage": {"input_tokens": 20, "output_token_details": {"audio_tokens": 0, "text_tokens": 3}},
                "output": [{"type": "function_call", "call_id": "c", "name": "read_file", "arguments": "{}"}]}})
            events = await adapter.receive()
            assert [e.kind for e in events] == ["response_done", "tool_call"]
            usage = events[0].data["usage"]
            assert "text_input_tokens" not in usage and "audio_input_tokens" not in usage
            assert usage["audio_output_tokens"] == 0 and usage["text_output_tokens"] == 3
        finally:
            await adapter.close()
    asyncio.run(run())


def test_grok_outer_meter_does_not_hide_nested_token_axes(providers):
    async def run():
        adapter, wire = make(providers, "grok")
        try:
            await adapter.open()
            await adapter.receive()
            wire.put({"type": "response.done", "usage": {"billable_audio_seconds": 9}, "response": {
                "id": "r", "status": "completed", "usage": {"output_token_details": {"text_tokens": 3}}}})
            done = (await adapter.receive())[0].data
            assert done["usage"]["text_output_tokens"] == 3
            assert done["raw_usage"]["billable_audio_seconds"] == 9
            assert done["raw_usage"]["nested_usage"] == {"output_token_details": {"text_tokens": 3}}
        finally:
            await adapter.close()
    asyncio.run(run())


@pytest.mark.parametrize("changed", [False, True])
def test_openai_vad_echo_keeps_owner_option(providers, changed):
    from ouroboros.contracts.realtime import RealtimeSetupError
    class ChangedWire(Wire):
        def put(self, event):
            event = copy.deepcopy(event)
            vad = event.get("session", {}).get("audio", {}).get("input", {}).get("turn_detection")
            if changed and vad:
                vad["interrupt_response"] = True
            super().put(event)
    async def run():
        adapter, wire = make(providers, "openai", LiveOptions(interruptions=False), wire=ChangedWire())
        try:
            if changed:
                with pytest.raises(RealtimeSetupError, match="session_mismatch.*turn_detection"):
                    await adapter.open()
            else:
                await adapter.open()
                assert adapter.renewal_ready
        finally:
            await adapter.close()
    asyncio.run(run())


def test_openai_interrupted_audio_stays_silent_and_partial_caption_survives(providers):
    from ouroboros.contracts.live_conversation import Interrupt
    async def run():
        adapter, wire = make(providers, "openai")
        try:
            await adapter.open()
            await adapter.receive()
            wire.put({"type": "response.created", "response": {"id": "r"}})
            await adapter.receive()
            wire.put({"type": "response.output_audio_transcript.delta", "item_id": "spoken", "delta": "Partial words"})
            await adapter.receive()
            await adapter.send(Interrupt())
            wire.put({"type": "response.output_audio.delta", "response_id": "r", "item_id": "spoken", "delta": "AAA="})
            assert await adapter.receive() == []
            wire.put({"type": "response.done", "response": {"id": "r", "status": "cancelled", "usage": {
                "output_token_details": {"audio_tokens": 2}}}})
            done = (await adapter.receive())[0].data
            assert done["status"] == "interrupted" and done["usage"]["audio_output_tokens"] == 2
            captions = adapter.final_captions()
            assert captions[0].data["text"] == "Partial words" and captions[0].data["incomplete"]
            assert adapter.final_captions() == []
        finally:
            await adapter.close()
    asyncio.run(run())


def test_openai_images_remain_native_and_do_not_open_response(providers):
    from ouroboros.contracts.live_conversation import Video
    async def run():
        adapter, wire = make(providers, "openai")
        try:
            await adapter.open()
            await adapter.receive()
            await adapter.send(Video(b"fake-image", "image/png"))
            assert not adapter.renewal_ready
            await drain(adapter, wire)
            assert adapter.renewal_ready
            assert wire.sent[-1]["item"]["content"][0]["type"] == "input_image"
            assert not any(e["type"] == "response.create" for e in wire.sent)
        finally:
            await adapter.close()
    asyncio.run(run())
