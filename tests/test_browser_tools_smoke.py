from __future__ import annotations

import base64
import http.server
import os
import socketserver
import threading

import pytest

from ouroboros.tools.browser import _browse_page, _browser_action, cleanup_browser
from ouroboros.contracts.task_constraint import TaskConstraint
from ouroboros.server_entrypoint import bound_service_socket
from ouroboros.tools.registry import ToolContext


pytestmark = pytest.mark.browser
_EXPECTED_BROWSER_ENGINES = {
    item.strip().lower()
    for item in os.environ.get("OUROBOROS_EXPECT_BROWSER_ENGINES", "").split(",")
    if item.strip()
}


class _StaticHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802 - stdlib callback name
        body = b"<html><body><h1>Browser smoke OK</h1></body></html>"
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        return


@pytest.fixture()
def static_page_url():
    server = socketserver.TCPServer(("127.0.0.1", 0), _StaticHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()


def test_browser_tools_launch_real_chromium(tmp_path, static_page_url, monkeypatch):
    from ouroboros.tools import browser as browser_mod

    subagent_ctx = ToolContext(
        repo_dir=tmp_path,
        drive_root=tmp_path,
        task_constraint=TaskConstraint(mode="local_readonly_subagent", allow_enable=False),
    )
    with bound_service_socket(tmp_path, "main", "127.0.0.1", 0) as listener:
        own_control = f"http://127.0.0.1:{listener.getsockname()[1]}"
        assert "BROWSER_LOCAL_READONLY_BLOCKED" in _browse_page(subagent_ctx, url=own_control)
    assert "origin_not_granted" in _browse_page(subagent_ctx, url="http://192.168.1.1")
    assert "origin_not_granted" in _browse_page(subagent_ctx, url="http://10.0.0.1")
    assert "BROWSER_LOCAL_READONLY_BLOCKED" in _browse_page(subagent_ctx, url="http://169.254.1.1")
    assert "BROWSER_LOCAL_READONLY_BLOCKED" in _browse_page(subagent_ctx, url="http://[::]/")
    assert "BROWSER_LOCAL_READONLY_BLOCKED" in _browse_page(subagent_ctx, url=f"file://{tmp_path / 'settings.json'}")
    assert "BROWSER_LOCAL_READONLY_BLOCKED" in _browser_action(subagent_ctx, action="evaluate", value="1 + 1")

    install_flags = []

    def fake_ensure_playwright_installed(*, engine="chromium", allow_install=True):
        install_flags.append((engine, allow_install))
        raise RuntimeError("missing browser")

    with monkeypatch.context() as m:
        m.setattr(browser_mod, "_playwright_ready", False)
        m.setattr(browser_mod, "_ensure_playwright_installed", fake_ensure_playwright_installed)
        with pytest.raises(RuntimeError, match="missing browser"):
            browser_mod._ensure_browser(subagent_ctx)
    assert install_flags == [("chromium", False)]

    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    expect_chromium = "chromium" in _EXPECTED_BROWSER_ENGINES or "all" in _EXPECTED_BROWSER_ENGINES
    try:
        try:
            text = _browse_page(ctx, url=static_page_url)
        except Exception as exc:
            if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
                if expect_chromium:
                    raise AssertionError("expected Chromium browser executable is missing") from exc
                pytest.skip(str(exc))
            raise
        if text.startswith("⚠️ BROWSER_INFRA_ERROR"):
            if "Executable doesn't exist" in text or "playwright install" in text.lower():
                if expect_chromium:
                    pytest.fail(text)
                pytest.skip(text)
            pytest.skip(text)
        assert "Browser smoke OK" in text

        screenshot = _browser_action(ctx, action="screenshot")
        if screenshot.startswith("⚠️ BROWSER_INFRA_ERROR"):
            if expect_chromium:
                pytest.fail(screenshot)
            pytest.skip(screenshot)
        raw = base64.b64decode(ctx.browser_state.last_screenshot_b64 or "")
        assert raw.startswith(b"\x89PNG\r\n\x1a\n")
    finally:
        cleanup_browser(ctx)


def test_browser_tools_launch_real_webkit_mobile_device(tmp_path, static_page_url):
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    expect_webkit = "webkit" in _EXPECTED_BROWSER_ENGINES or "all" in _EXPECTED_BROWSER_ENGINES
    try:
        try:
            text = _browse_page(ctx, url=static_page_url, engine="webkit", device="iPhone 13")
        except Exception as exc:
            msg = str(exc)
            if "Executable doesn't exist" in msg or "playwright install" in msg.lower():
                if expect_webkit:
                    raise AssertionError("expected WebKit browser executable is missing") from exc
                pytest.skip(msg)
            raise
        if text.startswith("⚠️ BROWSER_INFRA_ERROR"):
            if expect_webkit:
                pytest.fail(text)
            pytest.skip(text)
        assert "Browser smoke OK" in text
        assert getattr(ctx.browser_state, "_browser_engine", "") == "webkit"
        assert getattr(ctx.browser_state, "_browser_device", "") == "iPhone 13"
    finally:
        cleanup_browser(ctx)
