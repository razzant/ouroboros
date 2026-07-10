import supervisor.message_bus as message_bus
import ouroboros.event_bus as event_bus


def _make_bridge(monkeypatch, settings=None):
    return message_bus.LocalChatBridge(settings or {})


def test_configure_from_settings_without_legacy_field(monkeypatch):
    """configure_from_settings remains a no-op compatibility path."""
    bridge = _make_bridge(monkeypatch)
    bridge.configure_from_settings({
        "TELEGRAM_BOT_TOKEN": "",
        "TELEGRAM_CHAT_ID": "999",
    })
    assert bridge.get_updates(offset=0, timeout=0) == []


def test_ui_send_enqueues_structured_message_and_broadcasts(monkeypatch):
    bridge = _make_bridge(monkeypatch)
    broadcasts = []
    bridge._broadcast_fn = broadcasts.append

    bridge.ui_send("hello", sender_session_id="sess-1", client_message_id="c-1")
    updates = bridge.get_updates(offset=0, timeout=1)

    assert broadcasts[0]["role"] == "user"
    assert broadcasts[0]["sender_session_id"] == "sess-1"
    assert broadcasts[0]["client_message_id"] == "c-1"
    assert updates[0]["message"]["text"] == "hello"
    assert updates[0]["message"]["source"] == "web"
    assert updates[0]["message"]["sender_session_id"] == "sess-1"
    assert updates[0]["message"]["client_message_id"] == "c-1"


def test_ui_send_preserves_suppress_chat_log_flag(monkeypatch):
    bridge = _make_bridge(monkeypatch)

    bridge.ui_send("FULL_PROMPT", broadcast=False, suppress_chat_log=True)
    updates = bridge.get_updates(offset=0, timeout=1)

    assert updates[0]["message"]["text"] == "FULL_PROMPT"
    assert updates[0]["message"]["suppress_chat_log"] is True


def test_send_photo_publishes_transport_event_with_payload(monkeypatch):
    bridge = _make_bridge(monkeypatch)
    events = []
    monkeypatch.setattr(event_bus, "publish_event", lambda topic, data: events.append((topic, data)))
    monkeypatch.setattr(message_bus, "publish_event", lambda topic, data: events.append((topic, data)))

    ok, _ = bridge.send_photo(123, b"img", caption="caption", mime="image/png")

    assert ok is True
    topic, payload = events[-1]
    assert topic == event_bus.CHAT_PHOTO
    assert payload["image_base64"]
    assert payload["caption"] == "caption"
    assert payload["mime"] == "image/png"


def test_send_video_publishes_transport_event_with_payload(monkeypatch):
    bridge = _make_bridge(monkeypatch)
    events = []
    monkeypatch.setattr(event_bus, "publish_event", lambda topic, data: events.append((topic, data)))
    monkeypatch.setattr(message_bus, "publish_event", lambda topic, data: events.append((topic, data)))

    ok, _ = bridge.send_video(123, b"vid", caption="trailer", mime="video/mp4")

    assert ok is True
    topic, payload = events[-1]
    assert topic == event_bus.CHAT_VIDEO
    assert payload["video_base64"]
    assert payload["caption"] == "trailer"
    assert payload["mime"] == "video/mp4"


def test_send_document_publishes_transport_event_with_payload(monkeypatch):
    bridge = _make_bridge(monkeypatch)
    events = []
    monkeypatch.setattr(event_bus, "publish_event", lambda topic, data: events.append((topic, data)))
    monkeypatch.setattr(message_bus, "publish_event", lambda topic, data: events.append((topic, data)))

    ok, _ = bridge.send_document(
        123, b"filebytes", filename="report.csv", caption="q3", mime="text/csv",
        download_url="/api/files/download?path=Desktop/report.csv",
    )

    assert ok is True
    topic, payload = events[-1]
    assert topic == event_bus.CHAT_DOCUMENT
    assert payload["file_base64"]
    assert payload["filename"] == "report.csv"
    assert payload["caption"] == "q3"
    assert payload["mime"] == "text/csv"
    assert payload["download_url"] == "/api/files/download?path=Desktop/report.csv"


def test_send_document_persists_compact_chat_row(monkeypatch, tmp_path):
    """A delivered document persists a base64-free chat.jsonl row so it can be
    rebuilt on reload (the durable download_url carries the bytes)."""
    import json

    bridge = _make_bridge(monkeypatch)
    bridge._broadcast_fn = lambda *_a, **_k: None
    monkeypatch.setattr(event_bus, "publish_event", lambda *_a, **_k: None)
    monkeypatch.setattr(message_bus, "publish_event", lambda *_a, **_k: None)
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    monkeypatch.setattr(message_bus, "load_state", lambda: {"session_id": "s", "owner_id": 7})

    ok, _ = bridge.send_document(
        123, b"filebytes", filename="report.csv", caption="q3", mime="text/csv",
        download_url="/api/files/download?path=Desktop/report.csv", task_id="t-1",
    )
    assert ok is True

    rows = [json.loads(line) for line in (tmp_path / "logs" / "chat.jsonl").read_text().splitlines() if line.strip()]
    doc_rows = [r for r in rows if r.get("type") == "document"]
    assert len(doc_rows) == 1
    row = doc_rows[0]
    assert row["direction"] == "out"
    assert row["chat_id"] == 123
    assert row["filename"] == "report.csv"
    assert row["mime"] == "text/csv"
    assert row["download_url"] == "/api/files/download?path=Desktop/report.csv"
    assert row["task_id"] == "t-1"
    assert row["text"] == "q3"
    assert row["caption"] == "q3"  # explicit caption survives reload
    assert "file_base64" not in row  # no base64 bloat in chat.jsonl


def test_push_log_broadcast_surfaces_chat_id(monkeypatch):
    """Live log frames surface the task's chat_id top-level so the browser's
    per-thread fan-out routes the live card to its project panel; events with
    no chat_id default to the main chat (0)."""
    bridge = _make_bridge(monkeypatch)
    frames = []
    bridge._broadcast_fn = frames.append

    bridge.push_log({"type": "tool_call", "task_id": "t1", "chat_id": 1234})
    bridge.push_log({"type": "tool_call", "task_id": "t2"})

    logs = [f for f in frames if f.get("type") == "log"]
    assert logs[0]["chat_id"] == 1234
    assert logs[0]["data"]["task_id"] == "t1"
    assert logs[1]["chat_id"] == 0

