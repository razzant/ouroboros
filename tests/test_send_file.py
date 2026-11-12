"""Tests for the send_file (document/file) tool and MIME detection."""
import base64
import types

import pytest

from tests.test_main_notice import host as host  # real supervisor/bus fixture

from ouroboros.tools.core import _MAX_LINK_ACTIONS, _send_links
from ouroboros.tools.core_artifacts import _send_file, _detect_document_mime, _MAX_DOCUMENT_FILE_BYTES
from ouroboros.gateway.files import download_url_for_local_file


def _make_ctx(chat_id=123, drive_root=None):
    return types.SimpleNamespace(
        current_chat_id=chat_id,
        pending_events=[],
        drive_root=drive_root,
        task_id="t-send-file",
        task_metadata={},
    )


class TestSendFile:
    @pytest.mark.parametrize("bind_before_send", [True, False])
    def test_delivered_document_follows_its_work_not_the_project_named_in_its_filename(
        self, tmp_path, host, bind_before_send,
    ):
        import queue

        from tests.test_room_appearances import _project, _born_in, _ctx, _read
        from ouroboros import dialogue_store as ds
        from ouroboros.projects_registry import bind_task_to_project
        from ouroboros.tools.core_artifacts import _send_file
        from supervisor.events_chat_delivery import _handle_send_document
        from supervisor.message_bus import get_bridge

        host.ctx.bridge = get_bridge()
        project = _project(tmp_path, "circles")
        other = _project(tmp_path, "other")
        _born_in(tmp_path, "audit-root", 1)

        def bind():
            bind_task_to_project(tmp_path, "audit-root", "circles", project,
                                 origin={"absent": "system"})

        if bind_before_send:
            bind()
        source = tmp_path / "Circles-now.html"
        source.write_text("<p>A project-specific audit.</p>", encoding="utf-8")
        q = queue.Queue()
        ctx = _ctx(tmp_path, "audit-root", 1, q)
        assert _send_file(ctx, file_path=str(source), caption="Audit preview").startswith("OK:")
        event = q.get_nowait()
        _handle_send_document(event, host.ctx)
        documents = [frame for frame in host.frames if frame.get("type") == "document"]
        assert len(documents) == 1
        frame = documents[0]
        assert frame["chat_id"] == (project if bind_before_send else 1)

        def delivered_in(chat):
            return [row for row in _read(tmp_path, chat)[1]["messages"]
                    if row.get("msg_type") == "document"]

        if not bind_before_send:
            # Merely naming Circles cannot pull a personal Main artifact into that room.
            assert len(delivered_in(1)) == 1
            assert not delivered_in(project)
            bind()
        # Placement also makes an earlier delivery readable without sending its bytes again.
        (document,) = delivered_in(project)
        assert document["filename"] == "Circles-now.html"
        assert document["download_url"] == frame["download_url"]
        assert not delivered_in(1) and not delivered_in(other)
        assert len([entry for entry in ds.iter_rows(tmp_path, "chat")
                    if entry.row.get("type") == "document"]) == 1

    def test_file_path_reads_document(self, tmp_path):
        doc = tmp_path / "report.csv"
        doc.write_text("a,b,c\n1,2,3\n", encoding="utf-8")

        ctx = _make_ctx(drive_root=tmp_path)
        result = _send_file(ctx, file_path=str(doc), caption="quarterly report")

        assert "OK" in result
        assert len(ctx.pending_events) == 1
        event = ctx.pending_events[0]
        assert event["type"] == "send_document"
        assert event["mime"] == "text/csv"
        assert event["filename"] == "report.csv"
        assert event["caption"] == "quarterly report"
        assert event["file_base64"] == base64.b64encode(doc.read_bytes()).decode()

    def test_unknown_extension_falls_back_to_octet_stream(self, tmp_path):
        blob = tmp_path / "data.bin"
        blob.write_bytes(b"\x00\x01\x02\x03")

        ctx = _make_ctx(drive_root=tmp_path)
        result = _send_file(ctx, file_path=str(blob))

        assert "OK" in result
        assert ctx.pending_events[0]["mime"] == "application/octet-stream"

    def test_chat_zero_is_valid(self, tmp_path):
        doc = tmp_path / "note.txt"
        doc.write_text("hi", encoding="utf-8")

        ctx = _make_ctx(chat_id=0, drive_root=tmp_path)
        result = _send_file(ctx, file_path=str(doc))

        assert "OK" in result
        assert ctx.pending_events[0]["chat_id"] == 0

    def test_no_active_chat_returns_error(self, tmp_path):
        doc = tmp_path / "note.txt"
        doc.write_text("hi", encoding="utf-8")

        ctx = _make_ctx(chat_id=None)
        result = _send_file(ctx, file_path=str(doc))

        assert "no active chat" in result.lower()
        assert ctx.pending_events == []

    def test_file_not_found(self, tmp_path):
        ctx = _make_ctx()
        # Absolute on every OS: a drive-less "/nonexistent/..." is workspace-relative on Windows.
        result = _send_file(ctx, file_path=str(tmp_path / "nonexistent" / "report.pdf"))
        assert "not found" in result.lower()

    def test_directory_is_rejected(self, tmp_path):
        ctx = _make_ctx(drive_root=tmp_path)
        result = _send_file(ctx, file_path=str(tmp_path))
        assert "not found" in result.lower()
        assert ctx.pending_events == []

    def test_large_file_has_captured_reference_without_inline_bytes(self, tmp_path):
        from ouroboros.gateway.files import resolve_task_file_reference
        big = tmp_path / "huge.bin"
        with big.open("wb") as handle:
            handle.seek(_MAX_DOCUMENT_FILE_BYTES)
            handle.write(b"last")
        ctx = _make_ctx(drive_root=tmp_path / "data")
        result = _send_file(ctx, file_path=str(big))
        assert result.startswith("OK")
        event = ctx.pending_events[0]
        assert event["file_base64"] == ""
        assert event["file_ref"]["size"] == big.stat().st_size
        stored = resolve_task_file_reference(ctx.drive_root, ctx.task_id, event["file_ref"])
        assert stored.is_file() and stored != big

    def test_no_input_returns_error(self):
        ctx = _make_ctx()
        result = _send_file(ctx)
        assert "provide" in result.lower()

    def test_event_carries_download_url_from_durable_artifact(self, tmp_path, monkeypatch):
        # File-browser root = tmp_path so the durable artifact copy (under the
        # task drive) resolves to a servable /api/files/download URL.
        monkeypatch.setenv("OUROBOROS_FILE_BROWSER_DEFAULT", str(tmp_path))
        doc = tmp_path / "report.pdf"
        doc.write_bytes(b"%PDF-1.4 test")

        ctx = _make_ctx(drive_root=tmp_path)
        result = _send_file(ctx, file_path=str(doc), caption="q4")

        assert "OK" in result
        event = ctx.pending_events[0]
        assert event["download_url"].startswith("/api/tasks/t-send-file/artifacts/")
        assert event["download_url_compat"].startswith("/api/files/download?path=")
        assert "task_results/artifacts" in event["download_url_compat"]

    def test_canonical_url_survives_unrelated_browser_root(self, tmp_path, monkeypatch):
        # The canonical task route is independent of the current Files root.
        other = tmp_path / "root"
        other.mkdir()
        monkeypatch.setenv("OUROBOROS_FILE_BROWSER_DEFAULT", str(other))
        doc = tmp_path / "outside.txt"
        doc.write_text("x", encoding="utf-8")

        ctx = _make_ctx(drive_root=tmp_path / "drive")
        result = _send_file(ctx, file_path=str(doc))

        assert "OK" in result
        assert ctx.pending_events[0]["download_url"].startswith("/api/tasks/t-send-file/artifacts/")
        assert ctx.pending_events[0]["download_url_compat"] == ""


class TestDownloadUrlForLocalFile:
    def test_inside_root_returns_relative_url(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OUROBOROS_FILE_BROWSER_DEFAULT", str(tmp_path))
        (tmp_path / "Desktop").mkdir()
        f = tmp_path / "Desktop" / "a b.pdf"
        f.write_text("x", encoding="utf-8")
        url = download_url_for_local_file(f)
        # Root-relative + URL-quoted (space -> %20), never absolute.
        assert url == "/api/files/download?path=Desktop/a%20b.pdf"

    def test_outside_root_returns_empty(self, tmp_path, monkeypatch):
        root = tmp_path / "root"
        root.mkdir()
        monkeypatch.setenv("OUROBOROS_FILE_BROWSER_DEFAULT", str(root))
        outside = tmp_path / "elsewhere.txt"
        outside.write_text("x", encoding="utf-8")
        assert download_url_for_local_file(outside) == ""


class TestDetectDocumentMime:
    def test_pdf_extension(self):
        assert _detect_document_mime("report.pdf") == "application/pdf"

    def test_csv_extension(self):
        assert _detect_document_mime("data.csv") == "text/csv"

    def test_unknown_extension(self):
        assert _detect_document_mime("blob.unknownext") == "application/octet-stream"


class TestSendLinks:
    def test_valid_actions_append_typed_pending_event(self):
        ctx = _make_ctx()

        result = _send_links(
            ctx,
            links=[
                {"label": "Report", "url": "https://example.com/report"},
                {"label": "Dashboard", "url": "http://example.com/dashboard"},
            ],
            title="Results",
        )

        assert result.startswith("OK:")
        assert ctx.pending_events == [{
            "type": "send_links",
            "chat_id": 123,
            "task_id": "t-send-file",
            "parent_task_id": "",
            "root_task_id": "",
            "title": "Results",
            "actions": [
                {"label": "Report", "url": "https://example.com/report"},
                {"label": "Dashboard", "url": "http://example.com/dashboard"},
            ],
        }]

    def test_title_is_accepted_for_review_chat(self):
        ctx = _make_ctx(chat_id=0)
        result = _send_links(
            ctx,
            links=[{"label": "Home", "url": "https://example.com"}],
            title="Useful links",
        )
        assert result.startswith("OK:")
        assert ctx.pending_events[0]["chat_id"] == 0
        assert ctx.pending_events[0]["title"] == "Useful links"

    @pytest.mark.parametrize(
        ("links", "error_code"),
        [
            (None, "SEND_LINKS_ARG_ERROR"),
            ([{"label": "Bad", "url": "javascript:alert(1)"}], "SEND_LINKS_URL_BLOCKED"),
            ([{"label": "Bad\u2028Label", "url": "https://example.com"}], "SEND_LINKS_ARG_ERROR"),
        ],
    )
    def test_validation_refusals_are_typed(self, links, error_code):
        ctx = _make_ctx()
        result = _send_links(ctx, links=links)
        assert error_code in result
        assert ctx.pending_events == []

    @pytest.mark.parametrize(
        ("label", "url"),
        [
            ("Valid", "https://[::1]:8080/x"),
            ("Docs and specs", "https://example.com/x"),
            ("Valid", "https://host~tilde.example.com/x"),
        ],
        ids=[
            "ipv6-port", "label-space",
            "tilde-host",
        ],
    )
    def test_accepts_valid_link_actions(self, label, url):
        ctx = _make_ctx()

        result = _send_links(ctx, links=[{"label": label, "url": url}])

        assert result.startswith("OK:")
        assert ctx.pending_events[0]["actions"] == [{"label": label, "url": url}]

    def test_accepts_url_at_2048_character_limit(self):
        ctx = _make_ctx()
        prefix = "https://example.com/"
        url = prefix + "a" * (2048 - len(prefix))

        assert _send_links(ctx, links=[{"label": "Limit", "url": url}]).startswith("OK:")
        assert ctx.pending_events[0]["actions"] == [{"label": "Limit", "url": url}]

    def test_rejects_more_than_twelve_actions(self):
        ctx = _make_ctx()
        links = [
            {"label": f"Link {index}", "url": f"https://example.com/{index}"}
            for index in range(_MAX_LINK_ACTIONS + 1)
        ]
        assert "SEND_LINKS_TOO_MANY" in _send_links(ctx, links=links)
        assert ctx.pending_events == []

    def test_requires_active_chat(self):
        ctx = _make_ctx(chat_id=None)
        result = _send_links(
            ctx, links=[{"label": "Home", "url": "https://example.com"}],
        )
        assert "SEND_LINKS_NO_CHAT" in result
        assert ctx.pending_events == []


def test_send_links_delivery_handler_prefers_bound_project_chat(monkeypatch):
    from supervisor import events_chat_delivery as chat_delivery_events

    sent = []
    ctx = types.SimpleNamespace(
        bridge=types.SimpleNamespace(
            send_links=lambda chat_id, actions, title="", task_id="": (
                sent.append((chat_id, actions, title, task_id)) or (True, "")
            ),
        ),
        append_jsonl=lambda *_a, **_k: None,
        DRIVE_ROOT=types.SimpleNamespace(__truediv__=lambda *_a: "unused"),
    )
    monkeypatch.setattr(chat_delivery_events, "_bound_project_chat_id", lambda *_a: 8123)

    chat_delivery_events.EVENT_HANDLERS["send_links"](
        {
            "chat_id": 1,
            "task_id": "task-links",
            "title": "References",
            "actions": [{"label": "Docs", "url": "https://example.com/docs"}],
        },
        ctx,
    )

    assert sent == [(
        8123,
        [{"label": "Docs", "url": "https://example.com/docs"}],
        "References",
        "task-links",
    )]


@pytest.mark.parametrize("name", [".gitignore", "ж" * 100 + ".txt"], ids=["dotfile", "utf8"])
def test_benign_filename_capture_stays_readable_by_document_bridge(tmp_path, name):
    from ouroboros.gateway.files import resolve_task_file_reference
    source = tmp_path / name
    source.write_text("build/\n")
    ctx = _make_ctx(drive_root=tmp_path / "data")
    assert _send_file(ctx, str(source)).startswith("OK")
    event = ctx.pending_events[0]
    assert event["filename"] == name
    assert not event["file_ref"]["path"].startswith(".")
    assert resolve_task_file_reference(ctx.drive_root, ctx.task_id, event["file_ref"]).read_text() == "build/\n"
