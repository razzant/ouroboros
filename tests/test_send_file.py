"""Tests for the send_file (document/file) tool and MIME detection."""
import base64
import types

from ouroboros.tools.core import _send_file, _detect_document_mime, _MAX_DOCUMENT_FILE_BYTES
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
    def test_file_path_reads_document(self, tmp_path):
        doc = tmp_path / "report.csv"
        doc.write_text("a,b,c\n1,2,3\n", encoding="utf-8")

        ctx = _make_ctx()
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

        ctx = _make_ctx()
        result = _send_file(ctx, file_path=str(blob))

        assert "OK" in result
        assert ctx.pending_events[0]["mime"] == "application/octet-stream"

    def test_chat_zero_is_valid(self, tmp_path):
        doc = tmp_path / "note.txt"
        doc.write_text("hi", encoding="utf-8")

        ctx = _make_ctx(chat_id=0)
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

    def test_file_not_found(self):
        ctx = _make_ctx()
        result = _send_file(ctx, file_path="/nonexistent/report.pdf")
        assert "not found" in result.lower()

    def test_directory_is_rejected(self, tmp_path):
        ctx = _make_ctx()
        result = _send_file(ctx, file_path=str(tmp_path))
        assert "not found" in result.lower()
        assert ctx.pending_events == []

    def test_file_too_large(self, tmp_path):
        big = tmp_path / "huge.bin"
        big.write_bytes(b"\x00" * (_MAX_DOCUMENT_FILE_BYTES + 1))

        ctx = _make_ctx()
        result = _send_file(ctx, file_path=str(big))
        assert "too large" in result.lower()

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
        assert event["download_url"].startswith("/api/files/download?path=")
        # The URL points at the durable artifact copy, not the original path.
        assert "task_results/artifacts" in event["download_url"]

    def test_event_download_url_empty_when_outside_browser_root(self, tmp_path, monkeypatch):
        # Root is an unrelated dir; the delivered file is not servable → "".
        other = tmp_path / "root"
        other.mkdir()
        monkeypatch.setenv("OUROBOROS_FILE_BROWSER_DEFAULT", str(other))
        doc = tmp_path / "outside.txt"
        doc.write_text("x", encoding="utf-8")

        ctx = _make_ctx(drive_root=tmp_path / "drive")
        result = _send_file(ctx, file_path=str(doc))

        assert "OK" in result
        assert ctx.pending_events[0]["download_url"] == ""


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
