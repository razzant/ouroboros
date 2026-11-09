"""Recovered peer codec and Presence attribution using disposable synthetic data."""
from __future__ import annotations

import asyncio
import json
import struct

import pytest

from ouroboros import call_protocol as cp
from ouroboros.call_peers import LiveCallHub
from ouroboros.gateway.call_ws import serve_peer

PEER_CONTROL_BYTES = 16 * 1024  # published as call_socket.max_control_bytes
# Fewer characters than a peer's control may carry bytes, and far more UTF-8 bytes than that.
LONG_WORDS = "Синтетическая длинная реплика звонка 🐍. " * 400


def hello(role=cp.PEER_OWNER):
    return json.dumps({"type": "hello", "protocol": 1, "role": role, "mic": True, "speaker": True, "rate": 24000})


class PeerSocket:
    """The far side of the real socket door: one scripted peer and everything it is sent."""

    def __init__(self, *texts):
        self.incoming = asyncio.Queue()
        self.sent, self.frames, self.closed = [], [], []
        for text in texts:
            self.incoming.put_nowait({"type": "websocket.receive", "text": text})

    async def accept(self):
        pass

    async def receive(self):
        return await self.incoming.get()

    async def send_text(self, text):
        self.sent.append(json.loads(text))

    async def send_bytes(self, data):
        self.frames.append(data)

    async def close(self, **reason):
        self.closed.append(reason)
        self.incoming.put_nowait({"type": "websocket.disconnect"})


async def _joined(hub):
    while not hub.peers():
        await asyncio.sleep(.001)


@pytest.mark.parametrize("role,skill", [(cp.PEER_OWNER, ""), (cp.PEER_MEETING, "synthetic-transport")],
                         ids=["owner_surface", "companion"])
def test_long_queued_captions_survive_flush_and_close_and_reach_the_peer_whole(role, skill):
    assert len(LONG_WORDS) < PEER_CONTROL_BYTES < len(LONG_WORDS.encode("utf-8"))
    monologue = LONG_WORDS * 64  # no second, larger limit waits behind the first

    async def scenario():
        hub = LiveCallHub(cp.CallSpec("synthetic-long-caption", 1, "synthetic", "synthetic"))
        socket = PeerSocket(hello(role))
        door = asyncio.create_task(serve_peer(socket, hub, role=role, skill=skill))
        await asyncio.wait_for(_joined(hub), 5)
        # Nothing here awaits, so the door cannot drain: flush and close meet these still queued.
        hub.play(b"\x01\x00" * 480, "answer")
        hub.send_caption("ouroboros", LONG_WORDS, False, "answer")
        hub.send_caption("owner", monologue, True, "words")
        barge_in = hub.flush()
        hub.send_caption("ouroboros", LONG_WORDS + "!", True, "answer", replace=True)
        hub.close("finished")
        await asyncio.wait_for(door, 5)
        assert not socket.frames, "a flush retires queued audio of the older epoch"
        assert hub.epoch == barge_in + 1, "closing flushes again, over the same queue"
        last_flush = max(i for i, m in enumerate(socket.sent) if m["type"] == "flush")
        assert socket.sent[last_flush]["epoch"] == hub.epoch
        assert [(m["speaker"], m["text"], m["final"], m["item_id"]) for m in socket.sent[last_flush + 1:]
                if m["type"] == "caption"] == [("ouroboros", LONG_WORDS, False, "answer"), ("owner", monologue, True, "words"),
                                               ("ouroboros", LONG_WORDS + "!", True, "answer")]
        assert socket.sent[-1]["type"] == "state" and socket.sent[-1]["state"] == cp.STATE_CLOSED
        assert socket.closed and all(reason == {"code": 1000} for reason in socket.closed)
    asyncio.run(scenario())


@pytest.mark.parametrize("over", [0, 1], ids=["at_the_bound", "one_byte_over"])
def test_a_peers_control_stays_bounded_in_utf8_bytes_at_the_socket_door(over):
    def control(words):
        return json.dumps({"type": "context", "text": words, "source": "synthetic"}, ensure_ascii=False,
                          separators=(",", ":"))

    room = PEER_CONTROL_BYTES + over - len(control("").encode("utf-8"))
    words = "🐍" * (room // 4) + "x" * (room % 4)
    assert len(words) < PEER_CONTROL_BYTES / 3, "the bound counts bytes, never characters"
    assert len(control(words).encode("utf-8")) == PEER_CONTROL_BYTES + over

    async def scenario():
        hub = LiveCallHub(cp.CallSpec("synthetic-peer-bound", 1, "synthetic", "synthetic"))
        socket = PeerSocket(hello(), control(words))
        door = asyncio.create_task(serve_peer(socket, hub, role="owner"))
        events = []
        while not any(isinstance(event, (cp.PeerControl, cp.PeerLeft)) for event in events):
            event = hub.next_event(0)
            if event is None:
                await asyncio.sleep(.001)
            else:
                events.append(event)
        socket.incoming.put_nowait({"type": "websocket.disconnect"})
        await asyncio.wait_for(door, 5)
        while (event := hub.next_event(0)) is not None:
            events.append(event)
        hub.close("fixture_cleanup")
        return socket, events

    socket, events = asyncio.run(scenario())
    if over:
        assert socket.closed[0] == {"code": 1008, "reason": "invalid_call_control"}
        assert [type(e) for e in events] == [cp.PeerJoined, cp.PeerLeft] and events[1].reason == "invalid_call_control"
    else:
        assert [type(e) for e in events] == [cp.PeerJoined, cp.PeerControl, cp.PeerLeft]
        assert events[1].message["text"] == words


def test_bounded_codec_keeps_reading_server_controls(subtests):
    for kind in cp.SERVER_MESSAGE_TYPES:
        with subtests.test(kind=kind):
            message = {"type": kind}
            assert cp.decode_control(cp.encode_control(message)) == message


def test_a_long_server_caption_is_whole_json_beyond_the_bounded_peer_reader():
    caption = {"type": "caption", "speaker": "ouroboros", "text": LONG_WORDS, "final": True, "item_id": "answer",
               "replace": False}
    text = cp.encode_control(caption)
    assert len(text) < PEER_CONTROL_BYTES < len(text.encode("utf-8"))
    assert json.loads(text) == caption
    with pytest.raises(ValueError, match="too large"):
        cp.decode_control(text)  # the peer reader keeps its bound; nothing reads server text through it


def test_a_peer_cannot_send_the_hosts_own_messages():
    forged = json.dumps({"type": "caption", "speaker": "owner", "text": "Forged words.", "final": True, "item_id": "one"})

    async def scenario():
        hub = LiveCallHub(cp.CallSpec("synthetic-forged-caption", 1, "synthetic", "synthetic"))
        socket = PeerSocket(hello(), forged)
        await asyncio.wait_for(serve_peer(socket, hub, role="owner"), 5)
        hub.close("fixture_cleanup")
        return socket
    assert asyncio.run(scenario()).closed[0] == {"code": 1008, "reason": "invalid_call_control"}


@pytest.mark.parametrize("door", ["owner", "companion"])
def test_a_call_open_body_stays_bounded_before_decoding(door, tmp_path, monkeypatch):
    from starlette.applications import Starlette
    from starlette.testclient import TestClient

    from ouroboros.gateway import call_ws, host_call, host_service

    body = json.dumps({"chat_id": 1, "client_message_id": "я" * (PEER_CONTROL_BYTES // 2)}, ensure_ascii=False).encode("utf-8")
    if door == "owner":
        answer = TestClient(Starlette(routes=call_ws.call_routes())).post("/api/call/open", content=body)
        assert (answer.status_code, answer.json()["code"]) == (409, "control_too_large")
    else:
        async def authenticated(*_args):
            return "synthetic_skill", {}
        monkeypatch.setattr(host_call, "_authenticated", authenticated)
        answer = TestClient(host_service.create_host_service_app(tmp_path)).post("/call/open", content=body)
        assert (answer.status_code, answer.json()["code"]) == (413, "control_too_large")


def _loopback(app):
    """Serve ``app`` with uvicorn on a free loopback port, in its own thread."""
    import socket
    import threading
    import time

    import uvicorn

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning", lifespan="off"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, name="call-door-loopback", daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        assert thread.is_alive() and time.monotonic() < deadline, "the loopback server did not start"
        time.sleep(.01)
    return server, thread, sock


def test_words_no_text_frame_can_carry_end_the_call_at_their_sender_and_leave_no_phantom_peer():
    """A real socket, door, hub and driver; only the provider and the page are scripted.

    Malformed provider JSON can carry an unpaired surrogate, which no UTF-8 text frame can.
    Refused only inside a peer's socket, it left that dead peer registered, so a paid call
    kept running with nobody listening while later valid words went nowhere."""
    from starlette.applications import Starlette
    from websockets.asyncio.client import connect
    from websockets.exceptions import ConnectionClosed

    from ouroboros.call_peers import close_hub, open_owner_hub
    from ouroboros.gateway import call_ws
    from tests.test_live_conversation_driver import Host, started

    broken = "Synthetic partial answer \ud83d"
    hub, created = open_owner_hub(cp.CallSpec("synthetic-unencodable-caption", 1, "synthetic", "synthetic"))
    assert created
    server, thread, sock = _loopback(Starlette(routes=call_ws.call_routes()))
    received, closed = [], []

    async def until(predicate, facts):
        deadline = asyncio.get_running_loop().time() + 20  # a hang detector, not a latency claim
        while not predicate():
            assert asyncio.get_running_loop().time() < deadline, facts()
            await asyncio.sleep(.01)

    async def scenario():
        host = Host()
        host.hub = hub
        task, session = await started(host)
        try:
            async with connect(f"ws://127.0.0.1:{sock.getsockname()[1]}/ws/call/{hub.call_id}") as page:
                async def read():
                    try:
                        async for text in page:
                            received.append(json.loads(text))
                    except ConnectionClosed:
                        pass
                    closed.append(page.close_code)
                reader = asyncio.create_task(read())
                await page.send(hello())
                session.emit("ready", input_rate=24000, output_rate=24000)
                await until(lambda: any(m.get("state") == cp.STATE_LIVE for m in received), lambda: received)
                # Valid words, more UTF-8 bytes than a peer's control may carry, cross the real socket whole.
                session.emit("caption", speaker="owner", item_id="owner-long", text=LONG_WORDS, final=True)
                await until(lambda: any(m.get("text") == LONG_WORDS for m in received), lambda: closed)
                session.emit("caption", speaker="ouroboros", item_id="answer", text=broken, final=False)
                session.emit("caption", speaker="ouroboros", item_id="answer", text="Synthetic final answer.", final=True)
                await until(lambda: task.done() or closed, lambda: hub.peers())
                assert task.done(), {"call": "still running", "registered_peers": hub.peers(), "page_closed": closed}
                with pytest.raises(UnicodeEncodeError):
                    await task  # the words' sender ends the call, at its first unsendable caption
                close_hub(hub.call_id, "call_failed")  # the turn's own teardown
                await until(lambda: closed, lambda: received[-3:])
                await reader
            await until(lambda: not hub.peers(), lambda: {"registered_peers": hub.peers()})
            return host
        finally:
            close_hub(hub.call_id, "fixture_cleanup")
            await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 20)

    try:
        host = asyncio.run(scenario())
    finally:
        server.should_exit = True
        thread.join(10)
        sock.close()
    assert not thread.is_alive()
    captions = [(m["speaker"], m["text"], m["final"]) for m in received if m["type"] == "caption"]
    assert captions == [("owner", LONG_WORDS, True)], "nothing after the refused caption is shown"
    assert [(row["speaker"], row["text"]) for row in host.captions] == [("owner", LONG_WORDS)]
    assert received[-1]["type"] == "state" and received[-1]["state"] == cp.STATE_CLOSED
    assert received[-1]["reason"] == "call_failed" and closed == [1000]


def test_mixed_audio_and_meeting_bye_keep_nonowner_provenance():
    hub = LiveCallHub(cp.CallSpec("synthetic-mix", 1, "synthetic", "synthetic"), rates=(24000, 24000))
    for name, role in (("owner", cp.PEER_OWNER), ("guest", cp.PEER_MEETING)):
        hub.join(cp.PeerInfo(name, role, True, True), 24000)
        assert isinstance(hub.next_event(0), cp.PeerJoined)
        frame = cp.AudioFrame(cp.FRAME_AUDIO_IN, 0, 1, 24000, 0, struct.pack("<480h", *([1000] * 480)))
        assert hub.receive_audio(name, cp.decode_frame(cp.encode_frame(frame)))
    mixed = hub.next_event(.1)
    assert isinstance(mixed, cp.PeerAudio) and mixed.role == cp.PEER_MEETING
    assert struct.unpack("<480h", mixed.pcm) == (2000,) * 480
    hub.receive_control("guest", {"type": "bye"})
    assert not hub.ended_by_owner
    hub.receive_control("owner", {"type": "bye"})
    assert hub.ended_by_owner
    hub.close("fixture_cleanup")


@pytest.mark.parametrize("claimed_role", ["owner", "meeting"])
def test_shared_socket_parser_refuses_only_peer_role_escalation(claimed_role):
    class Socket:
        def __init__(self):
            self.closed = []
            self.received = False
        async def accept(self):
            pass
        async def receive(self):
            if self.received:
                return {"type": "websocket.disconnect"}
            self.received = True
            return {"type": "websocket.receive", "text": json.dumps({
                "type": "hello", "protocol": 1, "role": claimed_role,
                "mic": True, "speaker": True, "rate": 24000})}
        async def send_text(self, text):
            pass
        async def send_bytes(self, data):
            pass
        async def close(self, **reason):
            self.closed.append(reason)

    hub = LiveCallHub(cp.CallSpec("synthetic-role", 1, "synthetic", "synthetic"))
    socket = Socket()
    asyncio.run(serve_peer(socket, hub, role="meeting", skill="synthetic-transport"))
    if claimed_role == "owner":
        assert socket.closed[0] == {"code": 1008, "reason": "peer_role_forbidden"}
        assert hub.next_event(0) is None
    else:
        assert isinstance(hub.next_event(0), cp.PeerJoined)
        assert isinstance(hub.next_event(0), cp.PeerLeft)
        assert socket.closed == [{"code": 1000}]
    assert not hub.peers() and not hub.ended_by_owner
    hub.close("fixture_cleanup")


def test_presence_utterance_identity_and_mixed_audio_attribution_survive(tmp_path):
    from ouroboros.dialogue_provenance import run_origin
    from ouroboros.presence_call import record_call_utterance
    from ouroboros.presence_runner import presence_chat_id
    from tests._dialogue import chat_rows

    key = "synthetic:account:room:thread"
    task = {"id": "synthetic-presence-call", "chat_id": presence_chat_id(key), "metadata": {"presence": {
        "binding_id": "synthetic-binding", "transport_skill": "synthetic-transport", "behavior_skill": "synthetic-behavior",
        "event": {"source_event_id": "meeting:join", "provider": "synthetic", "account_id": "account",
                  "conversation_id": "room", "thread_id": "thread", "conversation_key": key,
                  "actor": {"platform_actor_id": "meeting", "display_name": "Synthetic meeting"},
                  "conversation": {}, "message": {}}}}}
    for _ in range(2):
        assert record_call_utterance(tmp_path, task, direction="in", text="Participant words.",
                                     utterance_id="one")
    assert record_call_utterance(tmp_path, task, direction="out", text="Generated reply.", utterance_id="two")
    rows = chat_rows(tmp_path)
    assert len(rows) == 2
    assert rows[0]["client_message_id"] == "meeting:join:heard:one"
    assert "identity" not in rows[0]["transport"]  # a call task without a focus has no admitted identity to carry
    for direction, name in (("in", "long-heard"), ("out", "long-spoken")):
        assert record_call_utterance(tmp_path, task, direction=direction, text=LONG_WORDS.strip(), utterance_id=name)
        assert chat_rows(tmp_path)[-1]["text"] == LONG_WORDS.strip()
    assert rows[1]["client_message_id"] == "meeting:join:spoken:two"
    assert rows[0]["sender_label"] == "meeting audio"
    assert rows[0]["transport"]["actor"]["display_name"] == "meeting audio"
    assert {row["presence_provenance"]["binding_id"] for row in rows} == {"synthetic-binding"}
    assert {row["chat_id"] for row in rows} == {task["chat_id"]}
    assert run_origin(task)["owner_ingress"] is False


def test_call_event_uses_ordinary_presence_binding_and_history_identity():
    from types import SimpleNamespace
    from ouroboros.presence_runner import PresenceTurnError, presence_chat_id, presence_turn_event, _stable_numeric_id
    from ouroboros.presence_bindings import conversation_key

    admission = SimpleNamespace(origin=SimpleNamespace(
        transport="synthetic", account_id="account", conversation_id="room", thread_id="thread"))
    payload = dict(source_event_id="event", provider="synthetic", account_id="account",
        conversation_id="room", thread_id="thread", conversation_key="owner-forged-room",
        actor={"platform_actor_id": "participant"}, conversation={}, message={}, text="Participant words")
    event = presence_turn_event(payload, admission)
    assert event.conversation_key == conversation_key("synthetic", "account", "room", "thread")
    assert presence_chat_id(event.conversation_key) == _stable_numeric_id("presence-conversation", event.conversation_key)
    for field in ("provider", "account_id", "conversation_id", "thread_id"):
        with pytest.raises(PresenceTurnError, match="presence_origin_mismatch"):
            presence_turn_event({**payload, field: "forged"}, admission)
    with pytest.raises(PresenceTurnError, match="presence_identity_missing"):
        presence_turn_event({**payload, "actor": {}}, admission)


@pytest.mark.parametrize("direction", ["in", "out"])
def test_call_utterance_waits_for_dialogue_writer_and_keeps_attribution(tmp_path, direction):
    """The #2082 required wait covers both heard and spoken meeting captions."""
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from ouroboros import dialogue_store
    from ouroboros.chat_references import reference_from_row
    from ouroboros.presence_call import record_call_utterance
    from ouroboros.presence_runner import presence_chat_id
    from tests._dialogue import chat_rows

    key = "synthetic:account:room:thread"
    task = {"id": "synthetic-call", "chat_id": presence_chat_id(key), "metadata": {"presence": {
        "binding_id": "binding", "transport_skill": "transport", "behavior_skill": "behavior",
        "event": {"source_event_id": "join", "provider": "synthetic", "account_id": "account",
                  "conversation_id": "room", "thread_id": "thread", "conversation_key": key,
                  "actor": {"display_name": "Meeting"}, "conversation": {}, "message": {}}}}}
    task["metadata"]["execution_transport"] = {"kind": "live_conversation"}
    dialogue_store.append(tmp_path, "progress", {"text": "warm"})
    holding, release, writing = threading.Event(), threading.Event(), threading.Event()

    def hold():
        with dialogue_store.transaction(tmp_path, write=True):
            holding.set()
            assert release.wait(5)

    def record():
        writing.set()
        return record_call_utterance(tmp_path, task, direction=direction, text="Caption under contention.", utterance_id="one")

    with ThreadPoolExecutor(max_workers=2) as pool:
        holder = pool.submit(hold)
        assert holding.wait(5)
        try:
            saved = pool.submit(record)
            assert writing.wait(5)
            # A non-waiting append would already have refused this held writer.
            assert not threading.Event().wait(.15) and not saved.done()
        finally:
            release.set()
        holder.result(5)
        assert saved.result(5)
    # Inbound replay rejoins the same accepted utterance.
    if direction == "in":
        assert record()
    [row] = chat_rows(tmp_path)
    assert row["text"] == "Caption under contention." and row["direction"] == direction
    assert row["transport"]["actor"]["display_name"] == ("meeting audio" if direction == "in" else "Meeting")
    if direction == "in":
        assert row["sender_label"] == "meeting audio"
    else:
        # #2236 keeps the conversation's actor as provenance, not the reply's author.
        assert not row.get("sender_label") and not row.get("sender_session_id")
    assert reference_from_row(row)["author_label"] == ("meeting audio" if direction == "in" else "Ouroboros")
    assert row["client_message_id"] == f"join:{'heard' if direction == 'in' else 'spoken'}:one"
    assert row["chat_id"] == task["chat_id"] and row["task_id"] == task["id"]


@pytest.mark.parametrize("direction", ["in", "out"])
@pytest.mark.parametrize("live", [False, True])
def test_presence_caption_requires_storage_for_calls_but_keeps_text_reply_best_effort(tmp_path, monkeypatch,
                                                                                    direction, live):
    from ouroboros import dialogue_store
    from ouroboros.presence_call import record_call_utterance

    task = {"id": "synthetic-call", "chat_id": 12, "metadata": {"presence": {
        "binding_id": "binding", "event": {"source_event_id": "join", "provider": "synthetic",
        "account_id": "account", "conversation_id": "room", "thread_id": "", "conversation_key": "room"}}}}
    if live:
        task["metadata"]["execution_transport"] = {"kind": "live_conversation"}
    quiet = []
    def unavailable(*args, **kwargs):
        raise OSError("synthetic unavailable store")
    monkeypatch.setattr(dialogue_store, "append", unavailable)
    monkeypatch.setattr(dialogue_store, "append_quietly", lambda *args: quiet.append(args))
    best_effort = direction == "out" and not live
    assert record_call_utterance(tmp_path, task, direction=direction, text="Caption", utterance_id="one") is best_effort
    assert bool(quiet) is best_effort
