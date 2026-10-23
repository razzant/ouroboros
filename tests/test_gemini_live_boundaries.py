"""Gemini native boundaries on fake sockets; no keys, network or device capture."""
from __future__ import annotations

import asyncio
import base64
import importlib.util
import json
import socket
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from ouroboros.contracts.live_conversation import Audio, LiveOptions, ToolOutputs
from ouroboros.contracts.realtime import RealtimeInput, canonical_json


@pytest.fixture(autouse=True)
def deny_network(monkeypatch):
    def refused(*args, **kwargs):
        raise AssertionError("Gemini tests must use fake sockets")
    monkeypatch.setattr(socket.socket, "connect", refused)
    monkeypatch.setattr(socket, "create_connection", refused)


@pytest.fixture
def gemini():
    path = Path(__file__).resolve().parents[1] / "skills/gemini_live/plugin.py"
    spec = importlib.util.spec_from_file_location("gemini_test_plugin", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Wire:
    def __init__(self):
        self.sent, self.incoming = [], asyncio.Queue()
        self.closed = False

    def put(self, event):
        self.incoming.put_nowait(event)

    async def send(self, raw):
        event = json.loads(raw)
        self.sent.append(event)
        if "setup" in event:
            self.put({"setupComplete": {}})

    async def recv(self):
        event = await self.incoming.get()
        if isinstance(event, Exception):
            raise event
        return json.dumps(event)

    @asynccontextmanager
    async def connect(self, model):
        try:
            yield self
        finally:
            self.closed = True


def selected(model="gemini-3.8-live", tools=(), messages=None):
    return RealtimeInput(model, canonical_json(messages or [
        {"role": "system", "content": "Host system rules."},
        {"role": "user", "content": "Original owner history."},
    ]), canonical_json(tools), 256, "nano", "core", "view")


def make_session(gemini, options=None, **kwargs):
    wire = Wire()
    session = gemini.GeminiSession(selected(**kwargs), options or LiveOptions(),
                                  connect=wire.connect, observe=lambda *_: None)
    return session, wire


async def receive(session, wire, event):
    wire.put(event)
    return await asyncio.wait_for(session.receive(), 1)


def audio_content(text="Spoken words", *, finished=True):
    return {"serverContent": {
        "modelTurn": {"parts": [{"inlineData": {"mimeType": "audio/pcm;rate=24000",
            "data": base64.b64encode(b"\x00\x00" * 480).decode()}}]},
        "outputTranscription": {"text": text, "finished": finished}}}


def test_continuous_silent_microphone_does_not_rearm_completed_input(gemini):
    async def run():
        session, wire = make_session(gemini)
        try:
            await session.open()
            await session.receive()
            await receive(session, wire, {"serverContent": {
                "inputTranscription": {"text": "Hello", "finished": True}}})
            await receive(session, wire, audio_content())
            await receive(session, wire, {"serverContent": {"turnComplete": True}})
            for _ in range(200):
                await session.send(Audio(b"\x00\x00" * 320, 16000))
                assert session.renewal_ready
            events = await receive(session, wire, {"goAway": {"timeLeft": "1s"}})
            assert any(e.kind == "renew" for e in events)
            assert session.renewal_ready
            assert len([r for r in wire.sent if "realtimeInput" in r]) == 200
        finally:
            await session.close()
    asyncio.run(run())


def test_native_activity_and_late_transcript_hold_renewal(gemini):
    async def run():
        session, wire = make_session(gemini)
        try:
            await session.open()
            await session.receive()
            assert session.renewal_ready
            events = await receive(session, wire, {"voiceActivity": {"type": "ACTIVITY_START"}})
            assert [e.kind for e in events] == ["input_started"]
            assert not session.renewal_ready
            await receive(session, wire, {"voiceActivity": {"type": "ACTIVITY_END"}})
            assert not session.renewal_ready  # VAD end is not the missing caption.
            await receive(session, wire, {"serverContent": {"turnComplete": True}})
            assert not session.renewal_ready
            events = await receive(session, wire, {"serverContent": {
                "inputTranscription": {"text": "Late words", "finished": True}}})
            assert any(e.kind == "caption" and e.data["final"] for e in events)
            assert any(e.kind == "input_done" for e in events)
            assert session.renewal_ready
        finally:
            await session.close()
    asyncio.run(run())


def test_partial_input_is_pending_without_vad_and_finishes_at_turn(gemini):
    async def run():
        session, wire = make_session(gemini)
        try:
            await session.open()
            await session.receive()
            events = await receive(session, wire, {"serverContent": {
                "inputTranscription": {"text": "A partial sentence"}}})
            assert any(e.kind == "input_started" for e in events)
            assert not session.renewal_ready
            events = await receive(session, wire, {"serverContent": {"turnComplete": True}})
            assert any(e.kind == "input_done" for e in events)
            assert session.renewal_ready
        finally:
            await session.close()
    asyncio.run(run())


def test_late_output_and_outstanding_tool_keep_their_boundaries(gemini):
    async def run():
        session, wire = make_session(gemini)
        try:
            await session.open()
            await session.receive()
            events = await receive(session, wire, audio_content("First ", finished=False))
            item = next(e.data["item_id"] for e in events if e.kind == "audio")
            await receive(session, wire, {"toolCall": {"functionCalls": [
                {"id": "call-1", "name": "read_file", "args": {}}]}})
            await receive(session, wire, {"serverContent": {"turnComplete": True}})
            assert not session.renewal_ready
            events = await receive(session, wire, {"serverContent": {
                "outputTranscription": {"text": "ending.", "finished": True}}})
            caption = next(e for e in events if e.kind == "caption" and e.data["final"])
            assert caption.data["item_id"] == item
            assert caption.data["text"] == "First ending."
            assert not session.renewal_ready  # The tool is still unanswered.
            await session.send(ToolOutputs(({"role": "tool", "tool_call_id": "call-1", "content": "Done"},), ()))
            assert session.renewal_ready
        finally:
            await session.close()
    asyncio.run(run())


def test_caption_without_audio_remains_pending_until_finished(gemini):
    async def run():
        session, wire = make_session(gemini)
        try:
            await session.open()
            await session.receive()
            await receive(session, wire, {"serverContent": {"outputTranscription": {"text": "First "}}})
            await receive(session, wire, {"serverContent": {"turnComplete": True}})
            assert not session.renewal_ready
            events = await receive(session, wire, {"serverContent": {
                "outputTranscription": {"text": "last.", "finished": True}}})
            assert next(e.data["text"] for e in events if e.kind == "caption" and e.data["final"]) == "First last."
            assert session.renewal_ready
        finally:
            await session.close()
    asyncio.run(run())


def test_schema_wakeup_keeps_a_socket_frame_that_arrived_together(gemini):
    async def run():
        session, wire = make_session(gemini)
        try:
            await session.open()
            await session.receive()
            receiving = asyncio.create_task(session.receive())
            await asyncio.sleep(0)
            wire.put({"serverContent": {"inputTranscription": {"text": "Must survive", "finished": True}}})
            schema = {"type": "function", "function": {"name": "added", "parameters": {"type": "object"}}}
            await session.send(ToolOutputs((), (schema,)))
            events = await asyncio.wait_for(receiving, 1)
            assert [e.kind for e in events] == ["renew"]
            events = await asyncio.wait_for(session.receive(), 1)
            assert any(e.kind == "caption" and e.data["text"] == "Must survive" for e in events)
        finally:
            await session.close()
    asyncio.run(run())
