"""Recovered peer codec and Presence attribution using disposable synthetic data."""
from __future__ import annotations

import asyncio
import json
import struct

import pytest

from ouroboros import call_protocol as cp
from ouroboros.call_peers import LiveCallHub
from ouroboros.gateway.call_ws import serve_peer


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
    assert row["sender_label"] == ("meeting audio" if direction == "in" else "Meeting")
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
