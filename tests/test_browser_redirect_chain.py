"""Every redirect hop meets the same target decision before any page result."""
from __future__ import annotations

import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
from types import SimpleNamespace

import pytest

from ouroboros.contracts.task_constraint import TaskConstraint
from ouroboros.server_entrypoint import bound_service_socket
from ouroboros.tools import browser
from ouroboros.tools.registry import ToolContext, ToolRegistry


class FakeRequest:
    def __init__(self, url, redirected_from=None):
        self.url, self.redirected_from = url, redirected_from


def chain(*urls):
    request = None
    for url in urls:
        request = FakeRequest(url, request)
    return SimpleNamespace(request=request)


class FakePage:
    def __init__(self, response):
        self.url, self.response, self.calls = "about:blank", response, []

    def set_default_timeout(self, ms):
        pass

    def goto(self, url, **_kwargs):
        self.url = self.response.request.url
        return self.response

    def inner_text(self, selector):
        self.calls.append("inner_text")
        return "FINAL_ALLOWED_CONTENT"

    def screenshot(self, **_kwargs):
        self.calls.append("screenshot")
        return b"png"


@pytest.fixture
def readonly_child(tmp_path, monkeypatch):
    from ouroboros import config
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "unrelated")
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path / "data",
                      task_constraint=TaskConstraint(mode="local_readonly_subagent", allow_enable=False))
    tools = ToolRegistry(tmp_path, ctx.drive_root)
    tools.set_context(ctx)
    return ctx, tools


def test_allowed_blocked_allowed_chain_withholds_content(readonly_child, monkeypatch):
    ctx, tools = readonly_child
    page = FakePage(chain("http://127.0.0.1:3000/start", "http://192.168.7.7/middle", "http://127.0.0.1:3000/final"))
    monkeypatch.setattr(browser, "_ensure_browser", lambda ctx, *_a, **_k: (page, ctx.browser_state))
    result = tools.execute("browse_page", {"url": "http://127.0.0.1:3000/start"})
    assert "origin_not_granted" in result and "192.168.7.7" in result
    assert "FINAL_ALLOWED_CONTENT" not in result and page.calls == []
    page.response = chain("http://127.0.0.1:3000/start", "http://127.0.0.1:3000/final")
    assert "FINAL_ALLOWED_CONTENT" in tools.execute("browse_page", {"url": "http://127.0.0.1:3000/start"})


def test_observed_navigation_chain_keeps_actions_refused_until_a_new_document(readonly_child, monkeypatch):
    ctx, tools = readonly_child
    page = FakePage(chain("http://127.0.0.1:3000/final"))
    page.url = "http://127.0.0.1:3000/final"
    monkeypatch.setattr(browser, "_ensure_browser", lambda ctx, *_a, **_k: (page, ctx.browser_state))
    ctx.browser_state.navigations = [chain("http://127.0.0.1:3000/start", "http://10.9.8.7/middle", page.url)]
    refused = tools.execute("browser_action", {"action": "screenshot"})
    assert "origin_not_granted" in refused and "10.9.8.7" in refused and page.calls == []
    ctx.browser_state.navigations = [chain("http://127.0.0.1:3000/final")]
    assert "Screenshot captured" in tools.execute("browser_action", {"action": "screenshot"})


@pytest.mark.browser
@pytest.mark.serial
@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_real_redirect_through_a_control_endpoint_withholds_content(tmp_path, engine):
    """An allowed dev page redirects through the actual Ouroboros main endpoint and back.

    The browser follows the whole chain natively; the child never receives the
    final content, keeps refusing actions on that document, and works again on a
    clean navigation. The root browser keeps the native redirect capability.
    """
    pytest.importorskip("playwright.sync_api")
    root = tmp_path / "data"
    hits: dict = {"app": [], "control": []}
    origins: dict = {}

    class App(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            hits["app"].append(self.path)
            if self.path == "/start":
                self.send_response(302); self.send_header("Location", origins["control"] + "/middle"); self.end_headers(); return
            body = b"<!doctype html><h1>FINAL_ALLOWED_CONTENT</h1>"
            self.send_response(200); self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

        def log_message(self, *_args):
            pass

    class Control(App):
        def do_GET(self):  # noqa: N802
            hits["control"].append(self.path)
            self.send_response(302); self.send_header("Location", origins["app"] + "/final"); self.end_headers()

    app = ThreadingHTTPServer(("127.0.0.1", 0), App)
    origins["app"] = f"http://127.0.0.1:{app.server_port}"
    app_thread = threading.Thread(target=app.serve_forever, daemon=True)
    app_thread.start()
    ctx = ToolContext(repo_dir=tmp_path, drive_root=root,
                      task_constraint=TaskConstraint(mode="local_readonly_subagent", allow_enable=False))
    root_ctx = ToolContext(repo_dir=tmp_path, drive_root=root)
    tools = ToolRegistry(tmp_path, root)
    tools.set_context(ctx)
    try:
        with bound_service_socket(root, "main", "127.0.0.1", 0) as listener:
            control = ThreadingHTTPServer(("127.0.0.1", 0), Control, bind_and_activate=False)
            control.socket.close()
            control.socket, control.server_address = listener, listener.getsockname()
            control.server_activate()
            origins["control"] = f"http://127.0.0.1:{listener.getsockname()[1]}"
            control_thread = threading.Thread(target=control.serve_forever, daemon=True)
            control_thread.start()
            try:
                result = tools.execute("browse_page", {"url": origins["app"] + "/start", "engine": engine})
                if "not already installed" in result:
                    if engine in os.environ.get("OUROBOROS_EXPECT_BROWSER_ENGINES", "").split(","):
                        pytest.fail(result)
                    pytest.skip(result)
                print("dispatched hops:", hits)
                assert "actual Ouroboros control-service endpoint" in result, result
                assert "FINAL_ALLOWED_CONTENT" not in result
                assert "actual Ouroboros control-service endpoint" in tools.execute(
                    "browser_action", {"action": "screenshot"})
                assert ctx.browser_state.last_screenshot_b64 is None
                clean = tools.execute("browse_page", {"url": origins["app"] + "/final", "engine": engine})
                assert "FINAL_ALLOWED_CONTENT" in clean, clean
                assert "Screenshot captured" in tools.execute("browser_action", {"action": "screenshot"})
                # Real tasks own separate worker threads; one Playwright Sync API loop per thread.
                browser.cleanup_browser(ctx)
                assert "FINAL_ALLOWED_CONTENT" in browser._browse_page(
                    root_ctx, origins["app"] + "/start", engine=engine)
            finally:
                control.shutdown()
                control_thread.join(timeout=5)
                assert not control_thread.is_alive()
    finally:
        browser.cleanup_browser(ctx)
        browser.cleanup_browser(root_ctx)
        app.shutdown(); app.server_close(); app_thread.join(timeout=5)
        assert not app_thread.is_alive()
