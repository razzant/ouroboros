import pytest

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


def test_project_completion_summary_keeps_event_time_label_live_and_on_reload(
    monkeypatch, tmp_path,
):
    """The message bus is the one live/durable transport for Main's host row."""
    import asyncio
    import json
    from types import SimpleNamespace

    bridge = _make_bridge(monkeypatch)
    frames = []
    bridge._broadcast_fn = frames.append
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    monkeypatch.setattr(message_bus, "get_bridge", lambda: bridge)
    monkeypatch.setattr(
        message_bus, "load_state",
        lambda: {"session_id": "session-1", "owner_id": 7},
    )
    monkeypatch.setattr(
        message_bus, "_advance_project_visible_revision", lambda _chat_id: None,
    )
    monkeypatch.setattr(message_bus, "publish_event", lambda *_a, **_k: None)

    message_bus.send_with_budget(
        1,
        "Launch 🚀 › Ship release · Completed",
        task_id="opaque-root-id",
        role="system",
        system_type="project_completion_summary",
        progress_meta={
            "project_id": "opaque-project-id",
            "project_name": "Launch 🚀",
            "target_label": "Launch 🚀 › Ship release",
            "status": "completed",
        },
    )

    live = next(frame for frame in frames if frame.get("type") == "chat")
    assert live["task_id"] == "opaque-root-id"
    assert live["target_label"] == "Launch 🚀 › Ship release"
    assert live["system_type"] == "project_completion_summary"

    from ouroboros.gateway.history import make_chat_history_endpoint

    endpoint = make_chat_history_endpoint(tmp_path)
    response = asyncio.run(endpoint(SimpleNamespace(
        query_params={"chat_id": "1", "limit": "20"},
    )))
    messages = json.loads(response.body.decode("utf-8"))["messages"]
    replayed = next(
        row for row in messages
        if row.get("system_type") == "project_completion_summary"
    )
    assert replayed["task_id"] == live["task_id"]
    assert replayed["target_label"] == live["target_label"]
    assert replayed["project_id"] == live["project_id"]
    assert replayed["project_name"] == live["project_name"]
    assert replayed["status"] == live["status"]


def test_terminal_incident_persists_and_replays_as_system_without_raw_salvage(
    monkeypatch, tmp_path,
):
    import asyncio
    import json
    from types import SimpleNamespace

    from ouroboros.gateway.history import make_chat_history_endpoint
    from supervisor.terminal_delivery import project_terminal_result_event

    raw = "RAW SALVAGE " * 1000
    projected = project_terminal_result_event(
        tmp_path, {"chat_id": 1}, "terminal-task",
        result_text=raw, terminal_origin="host_salvage",
        base_event={"chat_id": 1, "task_id": "terminal-task", "text": raw},
    )
    bridge = _make_bridge(monkeypatch)
    frames = []
    bridge._broadcast_fn = frames.append
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    monkeypatch.setattr(message_bus, "get_bridge", lambda: bridge)
    monkeypatch.setattr(message_bus, "load_state", lambda: {"owner_id": 7})
    monkeypatch.setattr(message_bus, "_advance_project_visible_revision", lambda _chat_id: None)
    monkeypatch.setattr(message_bus, "publish_event", lambda *_a, **_k: None)

    message_bus.send_with_budget(
        1, projected["text"], task_id=projected["task_id"],
        role=projected["role"], system_type=projected["system_type"],
    )

    live = next(frame for frame in frames if frame.get("type") == "chat")
    assert live["role"] == "system"
    assert live["system_type"] == "terminal_incident"
    assert raw not in live["content"]
    durable = json.loads(
        (tmp_path / "logs" / "chat.jsonl").read_text(encoding="utf-8").splitlines()[-1]
    )
    assert durable["direction"] == "system"
    assert durable["type"] == "terminal_incident"
    assert raw not in durable["text"]

    response = asyncio.run(make_chat_history_endpoint(tmp_path)(SimpleNamespace(
        query_params={"chat_id": "1", "limit": "20"},
    )))
    replayed = json.loads(response.body.decode("utf-8"))["messages"][-1]
    assert replayed["role"] == "system"
    assert replayed["system_type"] == "terminal_incident"
    assert raw not in replayed["text"]


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


def test_send_links_broadcasts_publishes_and_persists_compact_row(monkeypatch, tmp_path):
    import json

    bridge = _make_bridge(monkeypatch)
    frames = []
    events = []
    bridge._broadcast_fn = frames.append
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    monkeypatch.setattr(message_bus, "load_state", lambda: {"session_id": "s", "owner_id": 7})
    monkeypatch.setattr(message_bus, "_advance_project_visible_revision", lambda _chat_id: None)
    monkeypatch.setattr(
        message_bus, "publish_event", lambda topic, data: events.append((topic, data)),
    )
    prefix = "https://example.com/"
    url = prefix + "a" * (2048 - len(prefix))
    actions = [{"label": "Report", "url": url}]

    ok, error = bridge.send_links(123, actions, title="Results", task_id="task-links")

    assert (ok, error) == (True, "ok")
    live = next(frame for frame in frames if frame.get("type") == "links")
    assert live["actions"] == actions
    assert live["title"] == "Results"
    assert live["task_id"] == "task-links"
    topic, payload = events[-1]
    assert topic == event_bus.CHAT_LINKS
    assert set(payload) == {"chat_id", "transport", "title", "actions", "ts"}
    assert payload["actions"] == actions
    row = json.loads((tmp_path / "logs" / "chat.jsonl").read_text().splitlines()[-1])
    assert row["type"] == "links"
    assert row["actions"] == actions
    assert row["title"] == "Results"
    assert row["task_id"] == "task-links"
    assert "file_base64" not in row


def test_send_links_caps_title_across_broadcast_event_and_persistence(monkeypatch, tmp_path):
    import json

    bridge = _make_bridge(monkeypatch)
    frames = []
    events = []
    bridge._broadcast_fn = frames.append
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    monkeypatch.setattr(message_bus, "load_state", lambda: {"owner_id": 7})
    monkeypatch.setattr(message_bus, "_advance_project_visible_revision", lambda _chat_id: None)
    monkeypatch.setattr(
        message_bus, "publish_event", lambda topic, data: events.append((topic, data)),
    )

    ok, error = bridge.send_links(
        123,
        [{"label": "Docs", "url": "https://example.com/docs"}],
        title="X" * 300,
    )

    assert (ok, error) == (True, "ok")
    live = next(frame for frame in frames if frame.get("type") == "links")
    assert len(live["title"]) == 240
    topic, payload = events[-1]
    assert topic == event_bus.CHAT_LINKS
    assert len(payload["title"]) == 240
    row = json.loads((tmp_path / "logs" / "chat.jsonl").read_text().splitlines()[-1])
    assert len(row["title"]) == 240


@pytest.mark.parametrize(
    ("label", "url"),
    [
        ("Valid", "https://[::1]:8080/x"),
        ("Valid", "https://example.com:8443/x"),
        ("Valid", "https://example.com/a%20b"),
        ("Docs and specs", "https://example.com/x"),
        ("Valid", "https://例え.jp/x"),
        ("Valid", "https://my_server.example.com/docs"),
        ("Valid", "https://host~tilde.example.com/x"),
    ],
    ids=[
        "ipv6-port", "hostname-port", "encoded-path", "label-space",
        "unicode-host", "underscore-host", "tilde-host",
    ],
)
def test_send_links_accepts_valid_link_actions(monkeypatch, tmp_path, label, url):
    import json

    bridge = _make_bridge(monkeypatch)
    frames = []
    events = []
    bridge._broadcast_fn = frames.append
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    monkeypatch.setattr(message_bus, "load_state", lambda: {"owner_id": 7})
    monkeypatch.setattr(message_bus, "_advance_project_visible_revision", lambda _chat_id: None)
    monkeypatch.setattr(
        message_bus, "publish_event", lambda topic, data: events.append((topic, data)),
    )
    actions = [{"label": label, "url": url}]

    ok, error = bridge.send_links(123, actions)

    assert (ok, error) == (True, "ok")
    assert frames[-1]["actions"] == actions
    assert events[-1][1]["actions"] == actions
    row = json.loads((tmp_path / "logs" / "chat.jsonl").read_text().splitlines()[-1])
    assert row["actions"] == actions


@pytest.mark.parametrize(
    ("label", "url"),
    [
        ("Report", "https://exa mple.com/report"),
        ("Report", "https://example.com/re\nport"),
        ("Report", "https://example.com/re\0port"),
        ("Report", "https://example.com/" + "a" * 2029),
        ("Report", "https://[::1"),
        ("Report", "https://example.com:99999/path"),
        ("Report", "https://example.com:bad/path"),
        ("Report", "https://:443/path"),
        ("Report", "https://@/path"),
        ("Report", "https://exa%20mple.com/x"),
        ("Report", "https://%zz/x"),
        ("Report", "https://[v1.foo]/x"),
        ("Report", "https://[v1.fe80::1]/p"),
        ("Report", "https://exa\u00a0mple.com/x"),
        ("Report", "https://example.com/a\u2028b"),
        ("Report", "https://example.com\\@evil.com/"),
        ("Bad\nLabel", "https://example.com"),
        ("Bad\u2028Label", "https://example.com"),
    ],
    ids=[
        "space", "newline", "nul", "over-2048", "malformed-ipv6",
        "port-out-of-range", "non-numeric-port", "empty-host-port", "userinfo-only",
        "encoded-host", "invalid-percent-host", "ipvfuture", "ipvfuture-colons",
        "no-break-space", "line-separator", "backslash-authority", "label-newline",
        "label-line-separator",
    ],
)
def test_send_links_refuses_dirty_action_without_side_effects(
    monkeypatch, tmp_path, label, url,
):
    bridge = _make_bridge(monkeypatch)
    frames = []
    events = []
    bridge._broadcast_fn = frames.append
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    monkeypatch.setattr(message_bus, "_advance_project_visible_revision", lambda _chat_id: None)
    monkeypatch.setattr(
        message_bus, "publish_event", lambda topic, data: events.append((topic, data)),
    )

    ok, _error = bridge.send_links(123, [{"label": label, "url": url}])

    assert ok is False
    assert frames == []
    assert events == []
    assert not (tmp_path / "logs" / "chat.jsonl").exists()


def test_send_document_persists_compact_chat_row(monkeypatch, tmp_path):
    """A delivered document persists a base64-free chat.jsonl row so it can be
    rebuilt on reload (the durable download_url carries the bytes)."""
    import json

    bridge = _make_bridge(monkeypatch)
    frames = []
    bridge._broadcast_fn = frames.append
    monkeypatch.setattr(event_bus, "publish_event", lambda *_a, **_k: None)
    monkeypatch.setattr(message_bus, "publish_event", lambda *_a, **_k: None)
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    monkeypatch.setattr(message_bus, "load_state", lambda: {"session_id": "s", "owner_id": 7})

    ok, _ = bridge.send_document(
        123, b"filebytes", filename="report.csv", caption="q3", mime="text/csv",
        download_url="/api/files/download?path=Desktop/report.csv", task_id="t-1",
    )
    assert ok is True
    live = next(frame for frame in frames if frame.get("type") == "document")
    assert live["task_id"] == "t-1"
    assert live["size_bytes"] == len(b"filebytes")

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
    assert row["size_bytes"] == len(b"filebytes")
    assert "file_base64" not in row  # no base64 bloat in chat.jsonl


def test_send_photo_and_video_persist_compact_rows_before_unread_revision(monkeypatch, tmp_path):
    """Durable Project unread never advances for media absent from history."""
    import json

    bridge = _make_bridge(monkeypatch)
    frames = []
    bridge._broadcast_fn = frames.append
    monkeypatch.setattr(event_bus, "publish_event", lambda *_a, **_k: None)
    monkeypatch.setattr(message_bus, "publish_event", lambda *_a, **_k: None)
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    monkeypatch.setattr(message_bus, "load_state", lambda: {"session_id": "s", "owner_id": 7})
    revisions = []
    monkeypatch.setattr(message_bus, "_advance_project_visible_revision", revisions.append)

    bridge.send_photo(123, b"image", caption="shot", mime="image/png", task_id="media-task")
    bridge.send_video(123, b"video", caption="clip", mime="video/mp4", task_id="media-task")

    media_frames = [frame for frame in frames if frame.get("type") in {"photo", "video"}]
    assert [frame["task_id"] for frame in media_frames] == ["media-task", "media-task"]
    # The LIVE frame carries both durable addresses: a packaged desktop shell
    # cannot hand its file bridge a data: URI, so without these the media is
    # only saveable after a history reload.
    for frame in media_frames:
        assert frame["download_url"].startswith("/api/tasks/media-task/artifacts/chat-media-")
        # compat resolves only under the configured file-browser root; this
        # fixture's DATA_DIR lives in /tmp (outside it), so absence is honest —
        # presence must still carry the launcher-gate-compatible form.
        compat = frame.get("download_url_compat")
        assert compat is None or compat.startswith("/api/files/download?path=")

    rows = [
        json.loads(line)
        for line in (tmp_path / "logs" / "chat.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert [(row["type"], row["text"], row["mime"]) for row in rows] == [
        ("photo", "shot", "image/png"),
        ("video", "clip", "video/mp4"),
    ]
    assert all("image_base64" not in row and "video_base64" not in row for row in rows)
    assert all(row["task_id"] == "media-task" for row in rows)
    assert all(row["download_url"].startswith("/api/tasks/media-task/artifacts/chat-media-") for row in rows)
    stored = list((tmp_path / "task_results" / "artifacts" / "media-task" / "chat_media").iterdir())
    assert {path.read_bytes() for path in stored} == {b"image", b"video"}
    assert revisions == [123, 123]


def test_send_photo_keeps_live_delivery_when_media_persistence_fails(monkeypatch, tmp_path):
    bridge = _make_bridge(monkeypatch)
    frames = []
    bridge._broadcast_fn = frames.append
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    monkeypatch.setattr(
        message_bus, "store_chat_media_bytes",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("disk full")),
    )
    monkeypatch.setattr(message_bus, "load_state", lambda: {"session_id": "s", "owner_id": 7})

    ok, _ = bridge.send_photo(123, b"image", caption="shot", task_id="media-task")

    assert ok is True
    assert next(frame for frame in frames if frame.get("type") == "photo")["image_base64"]
    import json
    row = json.loads((tmp_path / "logs" / "chat.jsonl").read_text().splitlines()[-1])
    assert row.get("download_url", "") == ""


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


def test_push_log_suppresses_negative_a2a_chat_from_browser_broadcast(monkeypatch):
    bridge = _make_bridge(monkeypatch)
    frames = []
    bridge._broadcast_fn = frames.append

    bridge.push_log({"type": "task_started", "task_id": "a2a", "chat_id": -1001})

    assert frames == []


def test_budget_line_replays_unresolved_attempt_not_stale_state(monkeypatch, tmp_path):
    from ouroboros import usage_accounting as ua
    from supervisor import state as state_module

    (tmp_path / "state").mkdir()
    (tmp_path / "logs").mkdir()
    (tmp_path / "state" / "state.json").write_text(
        '{"spent_usd":0,"spent_calls":0}\n', encoding="utf-8",
    )
    (tmp_path / "settings.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "logs" / "events.jsonl").write_text("", encoding="utf-8")
    ua.ensure_legacy_imported(tmp_path)
    reservation = ua.reserve_attempt(ua.AttemptRequest(
        model="openai/gpt-5.5",
        provider="openrouter",
        reservation_usd=1.0,
        drive_root=tmp_path,
        global_limit_usd=10.0,
    ))
    ua.mark_dispatched(reservation)
    ua.mark_unresolved(reservation, "timeout")

    stale = {
        "spent_usd": 0,
        "spent_calls": 0,
        "current_branch": "ouroboros",
        "current_sha": "abcdef123456",
    }

    def update_state(mutator):
        mutator(stale)
        return dict(stale)

    monkeypatch.setattr(state_module, "update_state", update_state)
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    monkeypatch.setattr(message_bus, "TOTAL_BUDGET_LIMIT", 10.0)

    line = message_bus.budget_line(force=True)

    assert "$1.0000 / $10.00" in line
    assert "unresolved <=$1.0000" in line
    assert "ouroboros@abcdef12" in line
    assert "$0.0000 / $10.00" not in line


def test_budget_line_fails_loud_on_mid_ledger_corruption(monkeypatch, tmp_path):
    from ouroboros import usage_accounting as ua
    from supervisor import state as state_module

    (tmp_path / "state").mkdir()
    (tmp_path / "logs").mkdir()
    (tmp_path / "state" / "state.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "settings.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "logs" / "events.jsonl").write_text("", encoding="utf-8")
    ua.ensure_legacy_imported(tmp_path)
    reservation = ua.reserve_attempt(ua.AttemptRequest(
        model="openai/gpt-5.5", provider="openrouter", reservation_usd=1.0,
        drive_root=tmp_path, global_limit_usd=10.0,
    ))
    ua.mark_dispatched(reservation)
    ua.mark_unresolved(reservation, "timeout")
    ledger = tmp_path / ua.LEDGER_REL
    rows = ledger.read_text(encoding="utf-8").splitlines()
    ledger.write_text(rows[0] + "\nnot-json\n" + "\n".join(rows[1:]) + "\n", encoding="utf-8")

    stale = {"spent_usd": 0, "current_branch": "ouroboros", "current_sha": "abc"}
    monkeypatch.setattr(
        state_module, "update_state",
        lambda mutator: (mutator(stale), dict(stale))[1],
    )
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    monkeypatch.setattr(message_bus, "TOTAL_BUDGET_LIMIT", 10.0)

    line = message_bus.budget_line(force=True)

    assert "Budget: unavailable (physical-attempt ledger error)" in line
    assert "$0.0000" not in line


def test_budget_line_marks_quarantined_tail_nonfinal(monkeypatch, tmp_path):
    from ouroboros import usage_accounting as ua
    from supervisor import state as state_module

    (tmp_path / "state").mkdir()
    (tmp_path / "logs").mkdir()
    (tmp_path / "state" / "state.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "settings.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "logs" / "events.jsonl").write_text("", encoding="utf-8")
    ua.ensure_legacy_imported(tmp_path)
    reservation = ua.reserve_attempt(ua.AttemptRequest(
        model="openai/gpt-5.5", provider="openrouter", reservation_usd=0.1,
        drive_root=tmp_path, global_limit_usd=10.0,
    ))
    ua.release_attempt(reservation)
    with (tmp_path / ua.LEDGER_REL).open("ab") as handle:
        handle.write(b'{"seq":')

    stale = {"spent_usd": 0, "current_branch": "ouroboros", "current_sha": "abc"}
    monkeypatch.setattr(
        state_module, "update_state",
        lambda mutator: (mutator(stale), dict(stale))[1],
    )
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    monkeypatch.setattr(message_bus, "TOTAL_BUDGET_LIMIT", 10.0)

    line = message_bus.budget_line(force=True)

    assert "cost_final no" in line
    assert "ledger_integrity DEGRADED (quarantined ledger tail)" in line


def _registry_with_project(tmp_path):
    from ouroboros.projects_registry import create_project

    project = create_project(tmp_path, "happy-farm", name="Happy Farm")
    return int(project["chat_id"])


def test_broadcast_stamps_project_thread_for_registry_chat_only(monkeypatch, tmp_path):
    """The broadcast choke stamps ``project_thread`` when the final chat_id is a
    reserved Project thread, so Main can reject a project it has not learned
    yet; main (1), legacy (0) and transport-shaped ids stay unstamped."""
    project_chat = _registry_with_project(tmp_path)
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    bridge = _make_bridge(monkeypatch)
    frames = []
    bridge._broadcast_fn = frames.append

    bridge.send_message(project_chat, "in project", task_id="t-proj", is_progress=True)
    bridge.send_message(1, "in main", task_id="t-main", is_progress=True)
    bridge.send_message(197422551, "via telegram", task_id="t-tg", is_progress=True)
    bridge.push_log({"type": "tool_call", "task_id": "t-proj", "chat_id": project_chat})
    bridge.push_log({"type": "tool_call", "task_id": "t-main"})

    chats = [f for f in frames if f.get("type") == "chat"]
    assert chats[0]["chat_id"] == project_chat and chats[0]["project_thread"] is True
    assert "project_thread" not in chats[1]
    assert "project_thread" not in chats[2]
    logs = [f for f in frames if f.get("type") == "log"]
    assert logs[0]["project_thread"] is True
    assert "project_thread" not in logs[1]


def test_media_broadcasts_stamp_project_thread(monkeypatch, tmp_path):
    project_chat = _registry_with_project(tmp_path)
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    monkeypatch.setattr(message_bus, "store_chat_media_bytes", lambda *a, **k: "")
    bridge = _make_bridge(monkeypatch)
    frames = []
    bridge._broadcast_fn = frames.append

    bridge.send_photo(project_chat, b"\x89PNG", caption="p")
    bridge.send_video(project_chat, b"\x00\x00", caption="v")
    bridge.send_document(project_chat, b"%PDF", filename="a.pdf", caption="d")
    bridge.send_links(project_chat, [{"label": "L", "url": "https://example.com"}])
    bridge.send_photo(1, b"\x89PNG", caption="main")

    media = [f for f in frames if f.get("type") in ("photo", "video", "document", "links")]
    assert [f.get("project_thread") for f in media] == [True, True, True, True, None]


def test_project_thread_lens_follows_registry_file(monkeypatch, tmp_path):
    """The lens is mtime-cached: a project created after the first lookup is
    visible without a process restart."""
    from ouroboros.projects_registry import create_project, project_thread_chat_ids

    assert project_thread_chat_ids(tmp_path) == frozenset()
    first = int(create_project(tmp_path, "one")["chat_id"])
    assert first in project_thread_chat_ids(tmp_path)


def test_project_thread_stamp_survives_meta_and_covers_typing_and_echo(monkeypatch, tmp_path):
    """The marker is the LAST writer keyed on the FINAL chat_id (progress meta can
    neither spoof nor erase it), and the typing + user-echo seams carry it too."""
    project_chat = _registry_with_project(tmp_path)
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    bridge = _make_bridge(monkeypatch)
    frames = []
    bridge._broadcast_fn = frames.append

    bridge.send_message(project_chat, "p", task_id="t1", is_progress=True,
                        progress_meta={"project_thread": False})
    bridge.send_message(1, "m", task_id="t2", is_progress=True,
                        progress_meta={"project_thread": True})
    bridge.send_chat_action(project_chat, "typing", activity_id="a1")
    bridge.send_chat_action(1, "typing", activity_id="a2")
    bridge.handle_web_message("hi", chat_id=project_chat)
    bridge.handle_web_message("hi main", chat_id=1)

    bridge.send_message(1, "rehomed", task_id="t3", is_progress=True,
                        progress_meta={"chat_id": project_chat})

    chats = [f for f in frames if f.get("type") == "chat" and f.get("role") != "user"]
    assert chats[0]["project_thread"] is True      # meta cannot erase
    assert "project_thread" not in chats[1]        # meta cannot spoof
    assert chats[2]["project_thread"] is True      # stamp keys on the FINAL chat_id
    typing = [f for f in frames if f.get("type") == "typing"]
    assert typing[0]["project_thread"] is True
    assert "project_thread" not in typing[1]
    echoes = [f for f in frames if f.get("role") == "user"]
    assert echoes[0]["project_thread"] is True
    assert "project_thread" not in echoes[1]
