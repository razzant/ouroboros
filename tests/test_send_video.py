"""Tests for send_video file_path support and MIME detection."""
import base64
import types

from ouroboros.tools import core_artifacts


def _make_ctx(chat_id=123):
    return types.SimpleNamespace(
        current_chat_id=chat_id,
        pending_events=[],
    )


class TestSendVideoFilePath:
    def test_file_path_reads_mp4(self, tmp_path):
        mp4 = tmp_path / "test.mp4"
        mp4.write_bytes(b'\x00\x00\x00\x18ftypmp42' + b'\x00' * 100)

        ctx = _make_ctx()
        result = core_artifacts._send_video(ctx, file_path=str(mp4), caption="promo clip")

        assert "OK" in result
        assert len(ctx.pending_events) == 1
        event = ctx.pending_events[0]
        assert event["type"] == "send_video"
        assert event["mime"] == "video/mp4"
        assert event["caption"] == "promo clip"
        assert event["video_base64"] == base64.b64encode(mp4.read_bytes()).decode()

    def test_chat_zero_is_valid(self, tmp_path):
        mp4 = tmp_path / "test.mp4"
        mp4.write_bytes(b'\x00\x00\x00\x18ftypmp42')

        ctx = _make_ctx(chat_id=0)
        result = core_artifacts._send_video(ctx, file_path=str(mp4))

        assert "OK" in result
        assert ctx.pending_events[0]["chat_id"] == 0

    def test_no_active_chat_returns_error(self, tmp_path):
        mp4 = tmp_path / "test.mp4"
        mp4.write_bytes(b'\x00\x00\x00\x18ftypmp42')

        ctx = _make_ctx(chat_id=None)
        result = core_artifacts._send_video(ctx, file_path=str(mp4))

        assert "no active chat" in result.lower()
        assert ctx.pending_events == []

    def test_file_not_found(self):
        ctx = _make_ctx()
        result = core_artifacts._send_video(ctx, file_path="/nonexistent/video.mp4")
        assert "not found" in result.lower()

    def test_file_too_large(self, tmp_path):
        big = tmp_path / "huge.mp4"
        big.write_bytes(
            b'\x00\x00\x00\x18ftypmp42'
            + b'\x00' * (core_artifacts._MAX_VIDEO_FILE_BYTES + 1)
        )

        ctx = _make_ctx()
        result = core_artifacts._send_video(ctx, file_path=str(big))
        assert "too large" in result.lower()

    def test_no_input_returns_error(self):
        ctx = _make_ctx()
        result = core_artifacts._send_video(ctx)
        assert "provide" in result.lower()


class TestDetectVideoMime:
    def test_mp4_extension(self):
        assert core_artifacts._detect_video_mime("movie.mp4", b"") == "video/mp4"

    def test_mp4_header(self):
        assert core_artifacts._detect_video_mime(
            "movie.dat", b'\x00\x00\x00\x18ftypmp42\x00'
        ) == "video/mp4"

    def test_webm_header(self):
        assert core_artifacts._detect_video_mime(
            "movie.dat", b'\x1a\x45\xdf\xa3\x00'
        ) == "video/webm"

    def test_non_video_extension_falls_back_to_mp4(self):
        assert core_artifacts._detect_video_mime(
            "movie.txt", b"not enough signature"
        ) == "video/mp4"
