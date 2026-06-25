from supervisor.message_bus import LocalChatBridge


def test_bridge_preserves_generic_transport_metadata():
    bridge = LocalChatBridge()
    bridge.enqueue_local_message("hello", source="skill:bridge", transport={"kind": "messenger", "conversation_id": "42", "sender_label": "Messenger"})
    updates = bridge.get_updates(0, timeout=0)
    assert updates[0]["message"]["transport"] == {"kind": "messenger", "conversation_id": "42", "sender_label": "Messenger"}


def test_outbound_events_include_remembered_transport_metadata(monkeypatch):
    bridge = LocalChatBridge()
    events = []
    monkeypatch.setattr("supervisor.message_bus.publish_event", lambda topic, payload: events.append((topic, payload)))
    bridge.enqueue_local_message("hello", chat_id=77, transport={"kind": "messenger", "conversation_id": "abc", "sender_label": "Messenger"})
    bridge.get_updates(0, timeout=0)

    bridge.send_message(77, "reply")
    bridge.send_chat_action(77, "typing")
    bridge.send_photo(77, b"img", caption="photo")
    bridge.send_video(77, b"vid", caption="video", mime="video/mp4")

    assert all(payload.get("transport", {}).get("conversation_id") == "abc" for _topic, payload in events)


def test_transport_metadata_is_not_remembered_for_owner_chat(monkeypatch):
    bridge = LocalChatBridge()
    events = []
    monkeypatch.setattr("supervisor.message_bus.publish_event", lambda topic, payload: events.append((topic, payload)))
    bridge.enqueue_local_message("external", chat_id=1, source="skill:bridge", transport={"kind": "messenger", "conversation_id": "attacker", "sender_label": "Messenger"})
    bridge.get_updates(0, timeout=0)
    bridge.send_message(1, "owner reply")
    assert events[-1][1].get("transport") == {}


def test_transport_metadata_is_cleared_by_plain_message(monkeypatch):
    bridge = LocalChatBridge()
    events = []
    monkeypatch.setattr("supervisor.message_bus.publish_event", lambda topic, payload: events.append((topic, payload)))
    bridge.enqueue_local_message("external", chat_id=77, source="skill:bridge", transport={"kind": "messenger", "conversation_id": "abc", "sender_label": "Messenger"})
    bridge.get_updates(0, timeout=0)
    bridge.enqueue_local_message("plain", chat_id=77, source="web")
    bridge.get_updates(1, timeout=0)
    bridge.send_message(77, "reply")
    assert events[-1][1].get("transport") == {}
