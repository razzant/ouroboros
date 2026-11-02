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


def test_an_old_utterances_tail_cannot_close_new_barge_in_input(gemini):
    async def scenario():
        session, wire = make_session(gemini)
        try:
            await session.open(); await session.receive()
            await receive(session, wire, {'voiceActivity': {'type': 'ACTIVITY_START'}})
            first = await receive(session, wire, {'serverContent': {'inputTranscription': {'text': 'First owner words.'}}})
            first_id = next(e.data['item_id'] for e in first if e.kind == 'caption')
            await receive(session, wire, {'voiceActivity': {'type': 'ACTIVITY_END'}})
            boundary = await receive(session, wire, {'voiceActivity': {'type': 'ACTIVITY_START'}})
            assert [e.kind for e in boundary] == ['caption', 'input_done', 'input_started']
            assert boundary[0].data == dict(speaker='owner', text='First owner words.', item_id=first_id, final=True)
            events = await receive(session, wire, {'serverContent': {'interrupted': True, 'turnComplete': True}})
            assert not any(e.kind == 'input_done' for e in events)
            assert session.input_pending and not session.renewal_ready
            await receive(session, wire, {'serverContent': {'inputTranscription': {'text': 'New owner words.'}}})
            await receive(session, wire, {'voiceActivity': {'type': 'ACTIVITY_END'}})
            last = await receive(session, wire, {'serverContent': {'turnComplete': True}})
            caption = next(e.data for e in last if e.kind == 'caption' and e.data['final'])
            assert caption['text'] == 'New owner words.' and caption['item_id'] != first_id
            assert session.renewal_ready
        finally:
            await session.close()
    asyncio.run(scenario())


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


@pytest.mark.parametrize('model', ['gemini-3.8-live', 'gemini-3.8-live-extended-thinking'])
def test_completed_caption_without_finished_releases_quiet_and_late_tail_revises(gemini, model):
    async def run():
        session, wire = make_session(gemini, model=model)
        try:
            await session.open()
            await session.receive()
            await receive(session, wire, {"serverContent": {"outputTranscription": {"text": "First "}}})
            assert not session.renewal_ready
            await receive(session, wire, {"serverContent": {"turnComplete": True}})
            if 'thinking' in model:
                assert not session.renewal_ready, 'thinking still needs its native IDLE'
                await receive(session, wire, {"serverContent": {"turnComplete": True, "interactionStatus": "IDLE"}})
            assert session.renewal_ready
            events = await receive(session, wire, {"serverContent": {
                "outputTranscription": {"text": "last.", "finished": True}}})
            assert next(e.data["text"] for e in events if e.kind == "caption" and e.data["final"]) == "First last."
            assert next(e.data['replace'] for e in events if e.kind == 'caption' and e.data['final'])
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


def test_a_live_update_carries_images_as_inline_parts_not_base64_text(gemini):
    """A screenshot's image row in a Gemini call went as JSON text, its base64 counted as words (I1 review, 27.10)."""
    from ouroboros.contracts.live_conversation import ContextUpdate
    async def scenario():
        session, wire = make_session(gemini)
        await session.open()
        data = base64.b64encode(b"\x89PNG fake").decode()
        await session.send(ContextUpdate(({"role": "user", "content": [
            {"type": "text", "text": "[image: shot.png]"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + data}}]},)))
        turn = wire.sent[-1]["clientContent"]["turns"][0]
        assert turn["parts"][1] == {"inlineData": {"mimeType": "image/png", "data": data}}
        assert data not in turn["parts"][0]["text"] and "[Attached image 1 follows]" in turn["parts"][0]["text"]
        await session.send(ContextUpdate(({"role": "user", "content": "Plain words"},)))
        await session.close()
    asyncio.run(scenario())


def test_a_tool_change_is_named_as_needing_a_new_session(gemini):
    """28.10: a live Gemini session keeps its declarations; the driver asks this before delivering a result."""
    tool = {"type": "function", "function": {"name": "send_user_message", "description": "Send",
                                             "parameters": {"type": "object", "properties": {}}}}
    added = {"type": "function", "function": {"name": "newly_enabled", "description": "New",
                                              "parameters": {"type": "object", "properties": {}}}}
    session, _wire = make_session(gemini, tools=(tool,))
    assert session.tools_need_renewal((tool,)) is False
    assert session.tools_need_renewal((tool, added)) is True


def test_a_live_update_carries_no_host_delivery_marks(gemini):
    """The call driver marks its own rows (canonical facts, a requested look); Gemini serializes a whole update as
    text, so those marks reached the model as words until they were left out (28.10)."""
    from ouroboros.contracts.live_conversation import ContextUpdate
    async def scenario():
        session, wire = make_session(gemini)
        await session.open()
        await session.send(ContextUpdate(({"role": "user", "content": "[Host fact; not owner words]\nThe owner started "
                                           "showing the screen.", "_canonical": True, "_requested_look": True},)))
        text = wire.sent[-1]["clientContent"]["turns"][0]["parts"][0]["text"]
        assert "started showing the screen" in text and '"role":"user"' in text.replace(" ", "")
        assert "_canonical" not in text and "_requested_look" not in text
        await session.close()
    asyncio.run(scenario())


@pytest.mark.parametrize('model', ['gemini-3.8-live', 'gemini-3.8-live-extended-thinking'])
@pytest.mark.parametrize('new_caption_first', [False, True])
def test_old_turn_complete_cannot_finish_a_new_active_owner_caption(gemini, model, new_caption_first):
    """R19-1: while native VAD reports speech, an interrupted answer's completion leaves the words open."""
    async def scenario():
        session, wire = make_session(gemini, model=model)
        try:
            await session.open(); await session.receive()
            await receive(session, wire, {'voiceActivity': {'type': 'ACTIVITY_START'}})
            await receive(session, wire, {'serverContent': {'inputTranscription': {'text': 'First utterance.'}}})
            await receive(session, wire, {'voiceActivity': {'type': 'ACTIVITY_END'}})
            boundary = await receive(session, wire, {'voiceActivity': {'type': 'ACTIVITY_START'}})
            assert any(e.kind == 'caption' and e.data['final'] for e in boundary)
            fragment = {'serverContent': {'inputTranscription': {'text': 'New unfinished words'}}}
            if new_caption_first:
                await receive(session, wire, fragment)
            old = await receive(session, wire, {'serverContent': {'interrupted': True, 'turnComplete': True}})
            assert not any(e.kind == 'input_done' for e in old), 'old answer ended the current owner input'
            assert session.input_pending and not session.renewal_ready
            if not new_caption_first:
                await receive(session, wire, fragment)
            await receive(session, wire, {'voiceActivity': {'type': 'ACTIVITY_END'}})
            final = await receive(session, wire, {'serverContent': {'turnComplete': True}})
            assert [e.data['text'] for e in final if e.kind == 'caption' and e.data['final']] == ['New unfinished words']
            assert any(e.kind == 'input_done' for e in final) and session.renewal_ready
        finally:
            await session.close()
    asyncio.run(scenario())
