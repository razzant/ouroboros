"""Smoke tests for VLM (Vision Language Model) support."""

import sys
import os
import time
import unittest
from unittest.mock import MagicMock, patch
import pathlib
import pytest

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


@pytest.mark.serial
def test_vision_query_with_timeout_returns_without_waiting_for_hung_worker():
    from ouroboros.tools.vision import _vision_query_with_timeout

    started = time.monotonic()
    with unittest.TestCase().assertRaises(TimeoutError):
        _vision_query_with_timeout(None, prompt="x", images=[], model="m", timeout=0.01, _test_sleep_sec=2)
    assert time.monotonic() - started < 0.5


class TestLLMVisionQuery(unittest.TestCase):
    """Test LLMClient.vision_query() message format."""

    def test_vision_query_url_format(self):
        """vision_query builds correct message format for URL images."""
        from ouroboros.llm import LLMClient

        client = LLMClient(api_key="test-key")

        captured_messages = []

        def mock_chat(messages, model, tools=None, reasoning_effort="low", max_tokens=1024, tool_choice="auto", **kwargs):
            captured_messages.extend(messages)
            return {"content": "I see a test image."}, {"prompt_tokens": 10, "completion_tokens": 5}

        client.chat = mock_chat

        text, usage = client.vision_query(
            prompt="What do you see?",
            images=[{"url": "https://example.com/test.png"}],
            model="anthropic/claude-sonnet-4.6",
        )

        self.assertEqual(text, "I see a test image.")
        self.assertEqual(len(captured_messages), 1)
        content = captured_messages[0]["content"]
        self.assertIsInstance(content, list)
        self.assertEqual(len(content), 2)
        self.assertEqual(content[0]["type"], "text")
        self.assertEqual(content[0]["text"], "What do you see?")
        self.assertEqual(content[1]["type"], "image_url")
        self.assertIn("url", content[1]["image_url"])
        self.assertEqual(content[1]["image_url"]["url"], "https://example.com/test.png")

    def test_vision_query_base64_format(self):
        """vision_query builds correct data URI for base64 images."""
        from ouroboros.llm import LLMClient

        client = LLMClient(api_key="test-key")
        captured_messages = []

        def mock_chat(messages, model, tools=None, reasoning_effort="low", max_tokens=1024, tool_choice="auto", **kwargs):
            captured_messages.extend(messages)
            return {"content": "Base64 image description."}, {}

        client.chat = mock_chat

        fake_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        text, _ = client.vision_query(
            prompt="Describe this.",
            images=[{"base64": fake_b64, "mime": "image/png"}],
        )

        self.assertEqual(text, "Base64 image description.")
        content = captured_messages[0]["content"]
        image_part = content[1]
        self.assertTrue(image_part["image_url"]["url"].startswith("data:image/png;base64,"))
        self.assertIn(fake_b64, image_part["image_url"]["url"])

    def test_vision_query_multiple_images(self):
        """vision_query handles multiple images in one call."""
        from ouroboros.llm import LLMClient

        client = LLMClient(api_key="test-key")
        captured_messages = []

        def mock_chat(messages, model, tools=None, reasoning_effort="low", max_tokens=1024, tool_choice="auto", **kwargs):
            captured_messages.extend(messages)
            return {"content": "Two images."}, {}

        client.chat = mock_chat

        client.vision_query(
            prompt="Compare these images.",
            images=[
                {"url": "https://example.com/img1.png"},
                {"url": "https://example.com/img2.png"},
            ],
        )

        content = captured_messages[0]["content"]
        self.assertEqual(len(content), 3)  # text + 2 images

    def test_vision_query_empty_images(self):
        """vision_query works with no images (just text)."""
        from ouroboros.llm import LLMClient

        client = LLMClient(api_key="test-key")

        def mock_chat(messages, model, tools=None, reasoning_effort="low", max_tokens=1024, tool_choice="auto", **kwargs):
            return {"content": "Text only."}, {}

        client.chat = mock_chat

        text, _ = client.vision_query(prompt="Hello", images=[])
        self.assertEqual(text, "Text only.")

    def test_vision_query_forces_short_no_proxy_timeout(self):
        """vision_query uses a one-shot client timeout instead of the global tool timeout."""
        from ouroboros.llm import LLMClient

        client = LLMClient(api_key="test-key")
        captured = {}

        def mock_chat(**kwargs):
            captured.update(kwargs)
            return {"content": "ok"}, {}

        client.chat = mock_chat

        text, _ = client.vision_query(prompt="Hello", images=[], reasoning_effort="medium", timeout=75.0)

        self.assertEqual(text, "ok")
        self.assertEqual(captured["reasoning_effort"], "medium")
        self.assertTrue(captured["no_proxy"])
        self.assertEqual(captured["timeout"], 75.0)


    def test_downscale_image_enforces_provider_byte_cap(self):
        from PIL import Image
        import io
        import random
        from unittest.mock import patch

        from ouroboros.tools import vision

        rng = random.Random(0)
        raw_pixels = bytes(rng.getrandbits(8) for _ in range(256 * 256 * 3))
        img = Image.frombytes("RGB", (256, 256), raw_pixels)
        buf = io.BytesIO()
        img.save(buf, format="PNG")

        with patch.object(vision, "_VLM_MAX_PROVIDER_BYTES", 20_000), \
             patch.object(vision, "_VLM_MAX_IMAGE_SIDE", 256):
            capped, mime = vision._downscale_image_for_vlm(buf.getvalue(), "image/png")

        self.assertEqual(mime, "image/jpeg")
        self.assertLessEqual(len(capped), 20_000)


class TestAnalyzeScreenshotTool(unittest.TestCase):
    """Test the analyze_screenshot tool."""

    def _make_ctx(self, with_screenshot=True):
        from ouroboros.tools.registry import ToolContext, BrowserState
        ctx = MagicMock(spec=ToolContext)
        ctx.browser_state = BrowserState()
        ctx.event_queue = None
        ctx.task_id = "test-task"
        ctx.current_task_type = "task"
        if with_screenshot:
            ctx.browser_state.last_screenshot_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        else:
            ctx.browser_state.last_screenshot_b64 = None
        return ctx

    def test_no_screenshot_returns_warning(self):
        """analyze_screenshot returns warning when no screenshot available."""
        from ouroboros.tools.vision import _analyze_screenshot

        ctx = self._make_ctx(with_screenshot=False)
        result = _analyze_screenshot(ctx, prompt="What do you see?")
        self.assertIn("⚠️", result)
        self.assertIn("screenshot", result.lower())

    def test_analyze_screenshot_calls_vlm(self):
        """analyze_screenshot calls VLM with the screenshot base64."""
        from ouroboros.tools.vision import _analyze_screenshot

        ctx = self._make_ctx(with_screenshot=True)

        with patch("ouroboros.tools.vision._get_llm_client") as mock_get_client, \
             patch("ouroboros.tools.vision._vision_query_with_timeout") as mock_vlm:
            mock_client = MagicMock()
            mock_client.default_model.return_value = "openai/gpt-5.5"
            mock_get_client.return_value = mock_client
            mock_vlm.return_value = ("Beautiful UI.", {"prompt_tokens": 100, "completion_tokens": 20})

            result = _analyze_screenshot(ctx, prompt="Describe the UI.")

        self.assertEqual(result, "Beautiful UI.")
        mock_vlm.assert_called_once()
        call_kwargs = mock_vlm.call_args
        # Check that base64 image was passed
        images = call_kwargs[1].get("images") or call_kwargs[0][1]
        self.assertEqual(len(images), 1)
        self.assertIn("base64", images[0])
        # C2.1/C2.2: the VLM call must use a VISION-CAPABLE model (routed to a
        # capable slot), not blindly the active/default model.
        from ouroboros.provider_models import supports_vision
        self.assertTrue(supports_vision(call_kwargs[1]["model"]))
        self.assertEqual(call_kwargs[1]["reasoning_effort"], "medium")
        self.assertEqual(call_kwargs[1]["timeout"], 90.0)

    def test_analyze_screenshot_failure_is_tool_error_prefixed(self):
        from ouroboros.tools.vision import _analyze_screenshot
        from ouroboros.loop_tool_execution import _extract_result_metadata, _is_tool_execution_failure

        ctx = self._make_ctx(with_screenshot=True)
        with patch("ouroboros.tools.vision._get_llm_client") as mock_get_client, \
             patch("ouroboros.tools.vision._vision_query_with_timeout") as mock_vlm:
            mock_client = MagicMock()
            mock_client.default_model.return_value = "openai/gpt-5.5"
            mock_get_client.return_value = mock_client
            mock_vlm.side_effect = RuntimeError("provider failed")

            result = _analyze_screenshot(ctx, prompt="Describe the UI.")

        self.assertTrue(result.startswith("⚠️ VLM_ANALYSIS_FAILED"))
        self.assertTrue(_is_tool_execution_failure(True, result))
        self.assertEqual(_extract_result_metadata("analyze_screenshot", result, True)["status"], "vlm_error")


class TestVlmQueryTool(unittest.TestCase):
    """Test the vlm_query tool."""

    def _make_ctx(self):
        from ouroboros.tools.registry import ToolContext, BrowserState
        ctx = MagicMock(spec=ToolContext)
        ctx.browser_state = BrowserState()
        ctx.event_queue = None
        ctx.task_id = "test-task"
        ctx.current_task_type = "task"
        return ctx

    def test_vlm_query_requires_image(self):
        """vlm_query returns error when no image provided."""
        from ouroboros.tools.vision import _vlm_query

        ctx = self._make_ctx()
        result = _vlm_query(ctx, prompt="What is this?")
        self.assertIn("⚠️", result)

    def test_vlm_query_with_url(self):
        """vlm_query calls VLM with URL image."""
        from ouroboros.tools.vision import _vlm_query

        ctx = self._make_ctx()

        with patch("ouroboros.tools.vision._get_llm_client") as mock_get_client, \
             patch("ouroboros.tools.vision._vision_query_with_timeout") as mock_vlm:
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client
            mock_vlm.return_value = ("A logo.", {})

            result = _vlm_query(ctx, prompt="What is the logo?", image_url="https://example.com/logo.png")

        self.assertEqual(result, "A logo.")
        call_kwargs = mock_vlm.call_args
        images = call_kwargs[1].get("images") or call_kwargs[0][1]
        self.assertEqual(images[0]["url"], "https://example.com/logo.png")

    def _make_uploads_dir(self):
        """Create a temp uploads directory and patch _allowed_file_roots to point there."""
        import tempfile
        tmpdir = tempfile.mkdtemp()
        uploads = pathlib.Path(tmpdir) / "uploads"
        uploads.mkdir()
        return tmpdir, uploads

    def test_vlm_query_with_file_path(self):
        """vlm_query reads a local PNG from uploads dir and passes base64 to VLM."""
        import base64 as b64mod
        from ouroboros.tools.vision import _vlm_query

        ctx = self._make_ctx()

        # Minimal valid 1x1 PNG (89 bytes)
        png_bytes = (
            b'\x89PNG\r\n\x1a\n'
            b'\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01'
            b'\x08\x02\x00\x00\x00\x90wS\xde'
            b'\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N'
            b'\x00\x00\x00\x00IEND\xaeB`\x82'
        )

        tmpdir, uploads = self._make_uploads_dir()
        img_path = uploads / "test.png"
        img_path.write_bytes(png_bytes)

        try:
            with patch("ouroboros.tools.vision._allowed_file_roots", return_value=[uploads]):
                with patch("ouroboros.tools.vision._get_llm_client") as mock_get_client, \
                     patch("ouroboros.tools.vision._vision_query_with_timeout") as mock_vlm:
                    mock_client = MagicMock()
                    mock_get_client.return_value = mock_client
                    mock_vlm.return_value = ("A small PNG.", {})

                    result = _vlm_query(ctx, prompt="What is this?", file_path=str(img_path))

            self.assertEqual(result, "A small PNG.")
            call_kwargs = mock_vlm.call_args
            images = call_kwargs[1].get("images") or call_kwargs[0][1]
            self.assertEqual(len(images), 1)
            self.assertIn("base64", images[0])
            self.assertEqual(images[0]["mime"], "image/png")
            self.assertEqual(b64mod.b64decode(images[0]["base64"]), png_bytes)
        finally:
            import shutil
            shutil.rmtree(tmpdir)

    def test_vlm_query_file_not_found(self):
        """vlm_query returns error for missing file path."""
        import shutil
        from ouroboros.tools.vision import _vlm_query

        ctx = self._make_ctx()
        tmpdir, uploads = self._make_uploads_dir()
        try:
            with patch("ouroboros.tools.vision._allowed_file_roots", return_value=[uploads]):
                result = _vlm_query(ctx, prompt="Describe this.", file_path=str(uploads / "missing.png"))
            self.assertIn("⚠️", result)
            self.assertIn("not found", result.lower())
        finally:
            shutil.rmtree(tmpdir)

    def test_vlm_query_non_image_rejected(self):
        """vlm_query rejects non-image files (fail-closed MIME)."""
        from ouroboros.tools.vision import _vlm_query

        ctx = self._make_ctx()
        tmpdir, uploads = self._make_uploads_dir()
        txt_path = uploads / "notes.txt"
        txt_path.write_bytes(b"this is plain text, not an image")

        try:
            with patch("ouroboros.tools.vision._allowed_file_roots", return_value=[uploads]):
                result = _vlm_query(ctx, prompt="What is this?", file_path=str(txt_path))
            self.assertIn("⚠️", result)
            self.assertIn("supported image", result.lower())
        finally:
            import shutil
            shutil.rmtree(tmpdir)

    def test_vlm_query_path_outside_uploads_rejected(self):
        """vlm_query rejects paths outside the allowed uploads directory."""
        from ouroboros.tools.vision import _vlm_query

        ctx = self._make_ctx()
        tmpdir, uploads = self._make_uploads_dir()
        # Create a PNG outside the uploads dir
        outside_path = pathlib.Path(tmpdir) / "secret.png"
        outside_path.write_bytes(b'\x89PNG\r\n\x1a\n' + b'\x00' * 50)

        try:
            with patch("ouroboros.tools.vision._allowed_file_roots", return_value=[uploads]):
                result = _vlm_query(ctx, prompt="What is this?", file_path=str(outside_path))
            self.assertIn("⚠️", result)
            self.assertIn("uploads directory", result)
        finally:
            import shutil
            shutil.rmtree(tmpdir)

    def test_vlm_query_data_dir_env_isolation(self):
        """When OUROBOROS_DATA_DIR is set, only that dir's uploads/ is allowed (not ~/Ouroboros/data/uploads)."""
        import shutil
        import os as os_mod
        from ouroboros.tools.vision import _vlm_query

        ctx = self._make_ctx()
        # Configure a custom data dir
        tmpdir, custom_uploads = self._make_uploads_dir()
        # The "default" home uploads path is different
        pathlib.Path("~/Ouroboros/data/uploads").expanduser().resolve()

        # Create a valid PNG in home_uploads area (mocked via a separate temp dir)
        home_tmp, _ = self._make_uploads_dir()
        home_uploads_mock = pathlib.Path(home_tmp) / "uploads"
        home_uploads_mock.mkdir(exist_ok=True)
        img_path = home_uploads_mock / "test.png"
        png_bytes = b'\x89PNG\r\n\x1a\n' + b'\x00' * 50
        img_path.write_bytes(png_bytes)

        try:
            # With OUROBOROS_DATA_DIR pointing to custom tmpdir, home path is NOT allowed
            with patch.dict(os_mod.environ, {"OUROBOROS_DATA_DIR": str(pathlib.Path(tmpdir))}):
                # We call the real _allowed_file_roots (not patched) here
                from ouroboros.tools.vision import _allowed_file_roots
                roots = _allowed_file_roots()
                # Two env-derived roots: the custom uploads AND the skill-state
                # tree (state/skills), where reviewed skills write screenshots.
                self.assertEqual(len(roots), 2)
                self.assertEqual(roots[0], pathlib.Path(tmpdir).resolve() / "uploads")
                self.assertEqual(roots[1], pathlib.Path(tmpdir).resolve() / "state" / "skills")
                # Attempt to read image from home_uploads_mock — should be rejected
                with patch("ouroboros.tools.vision._allowed_file_roots", return_value=roots):
                    result = _vlm_query(ctx, prompt="test", file_path=str(img_path))
            self.assertIn("⚠️", result)
            self.assertIn("uploads directory", result)
        finally:
            shutil.rmtree(tmpdir)
            shutil.rmtree(home_tmp)

    def test_vlm_query_symlink_escape_rejected(self):
        """vlm_query rejects a symlink inside uploads that points outside uploads."""
        import shutil
        from ouroboros.tools.vision import _vlm_query

        ctx = self._make_ctx()
        tmpdir, uploads = self._make_uploads_dir()
        # Create a real PNG outside uploads
        outside = pathlib.Path(tmpdir) / "secret.png"
        outside.write_bytes(b'\x89PNG\r\n\x1a\n' + b'\x00' * 50)
        # Create symlink inside uploads pointing to outside file
        symlink = uploads / "link.png"
        symlink.symlink_to(outside)

        try:
            with patch("ouroboros.tools.vision._allowed_file_roots", return_value=[uploads]):
                result = _vlm_query(ctx, prompt="test", file_path=str(symlink))
            # The symlink resolves outside uploads/ so it must be rejected
            self.assertIn("⚠️", result)
            self.assertIn("uploads directory", result)
        finally:
            shutil.rmtree(tmpdir)

    def test_vlm_query_tool_registered(self):
        """vlm_query and analyze_screenshot tools are properly registered."""
        import pathlib
        from ouroboros.tools.registry import ToolRegistry

        registry = ToolRegistry(
            repo_dir=pathlib.Path("/tmp"),
            drive_root=pathlib.Path("/tmp"),
        )
        tools = registry.available_tools()
        self.assertIn("analyze_screenshot", tools, "analyze_screenshot must be registered")
        self.assertIn("vlm_query", tools, "vlm_query must be registered")


if __name__ == "__main__":
    unittest.main()
