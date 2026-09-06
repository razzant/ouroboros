"""Tests for send_photo file_path support and MIME detection."""
import base64
import types


from ouroboros.tools.core_artifacts import _send_photo, _detect_image_mime, _MAX_PHOTO_FILE_BYTES


def _make_ctx(chat_id=123, screenshot_b64=None):
    return types.SimpleNamespace(
        current_chat_id=chat_id,
        browser_state=types.SimpleNamespace(last_screenshot_b64=screenshot_b64),
        pending_events=[],
    )


class TestSendPhotoFilePath:
    def test_file_path_reads_png(self, tmp_path):
        png = tmp_path / "test.png"
        png.write_bytes(b'\x89PNG\r\n\x1a\n' + b'\x00' * 100)

        ctx = _make_ctx()
        result = _send_photo(ctx, file_path=str(png), caption="test shot")

        assert "OK" in result
        assert len(ctx.pending_events) == 1
        assert ctx.pending_events[0]["type"] == "send_photo"
        assert ctx.pending_events[0]["mime"] == "image/png"

    def test_file_not_found(self):
        ctx = _make_ctx()
        result = _send_photo(ctx, file_path="/nonexistent/image.png")
        assert "not found" in result.lower()

    def test_file_too_large(self, tmp_path):
        big = tmp_path / "huge.png"
        big.write_bytes(b'\x89PNG\r\n\x1a\n' + b'\x00' * (_MAX_PHOTO_FILE_BYTES + 1))

        ctx = _make_ctx()
        result = _send_photo(ctx, file_path=str(big))
        assert "too large" in result.lower()

    def test_no_input_returns_error(self):
        ctx = _make_ctx()
        result = _send_photo(ctx)
        assert "Provide either" in result


class TestSendPhotoBase64Fallback:
    def test_base64_still_works(self):
        ctx = _make_ctx()
        b64 = base64.b64encode(b'\x00' * 200).decode()
        result = _send_photo(ctx, image_base64=b64)
        assert "OK" in result
        assert len(ctx.pending_events) == 1

    def test_last_screenshot_reference(self):
        b64 = base64.b64encode(b'\x00' * 200).decode()
        ctx = _make_ctx(screenshot_b64=b64)
        result = _send_photo(ctx, image_base64="__last_screenshot__")
        assert "OK" in result

    def test_last_screenshot_missing(self):
        ctx = _make_ctx(screenshot_b64=None)
        result = _send_photo(ctx, image_base64="__last_screenshot__")
        assert "No screenshot" in result


class TestDetectImageMime:
    def test_png(self):
        assert _detect_image_mime(b'\x89PNG\r\n\x1a\n\x00') == "image/png"

    def test_jpeg(self):
        assert _detect_image_mime(b'\xff\xd8\xff\xe0') == "image/jpeg"

    def test_gif(self):
        assert _detect_image_mime(b'GIF89a') == "image/gif"

    def test_webp(self):
        assert _detect_image_mime(b'RIFF\x00\x00\x00\x00WEBP') == "image/webp"

    def test_unknown(self):
        assert _detect_image_mime(b'\x00\x00\x00\x00') == "application/octet-stream"
