"""Every navigation target meets the browser's decision before any page result: each redirect hop the same target
policy, and this installation's own UI page opens as an observer view (DESIGN "Unread")."""
from __future__ import annotations

import ipaddress
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
from types import SimpleNamespace

import pytest

from ouroboros import browser_policy, config, server_process
from ouroboros.contracts.task_constraint import TaskConstraint
from ouroboros.server_entrypoint import bound_service_socket
from ouroboros.tools import browser
from ouroboros.tools.registry import ToolContext, ToolRegistry
from tests._browser_fakes import close_generations


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

    async def goto(self, url, **_kwargs):
        self.url = self.response.request.url
        return self.response

    async def inner_text(self, selector):
        self.calls.append("inner_text")
        return "FINAL_ALLOWED_CONTENT"

    async def screenshot(self, **_kwargs):
        self.calls.append("screenshot")
        return b"png"

    async def wait_for_function(self, *_args, **_kwargs):
        return None

    async def evaluate(self, *_args, **_kwargs):
        return None


@pytest.fixture
def readonly_child(tmp_path, monkeypatch):
    from ouroboros import config
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "unrelated")
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path / "data",
                      task_constraint=TaskConstraint(mode="local_readonly_subagent", allow_enable=False))
    tools = ToolRegistry(tmp_path, ctx.drive_root)
    tools.set_context(ctx)
    yield ctx, tools
    close_generations(ctx.browser_state)


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


# The observer view: ``browse_page`` adds ``observer=1`` only when the UI document (``/``) is proven this installation's
# main UI by ``browser_policy.runtime_service_kind`` (the live binding and process facts), keeping the address's
# spelling; the page then moves no read cursor (``web/modules/read_ahead.js``). An API or file path, another service, an
# unrelated port (a candidate UI included), an unknown or unreadable identity and any other site open as given.
MAIN, HOST, OTHER = 9471, 9472, 9474  # this installation's UI, its Host Service, an unrelated application


def _installation(tmp_path, monkeypatch, host="127.0.0.1"):
    """A data root whose main UI (bound on ``host``) and Host Service are this live process, as a server records them;
    ``ouroboros.lan`` resolves to loopback and ``unresolvable.lan`` to nothing, without a resolver."""
    root = tmp_path / "data"
    root.mkdir()
    monkeypatch.setattr(config, "DATA_DIR", root)
    server_process.record_service_binding(root, "main", host, MAIN, pid=os.getpid())
    server_process.record_service_binding(root, "host_service", "127.0.0.1", HOST, pid=os.getpid())
    resolve = browser_policy._resolved_addresses

    def names(name):
        if name == "ouroboros.lan":
            return {ipaddress.ip_address("127.0.0.1")}
        if name == "unresolvable.lan":
            raise OSError("no address for unresolvable.lan")
        return resolve(name)

    monkeypatch.setattr(browser_policy, "_resolved_addresses", names)
    return ToolContext(repo_dir=tmp_path, drive_root=root)


@pytest.mark.parametrize("url, expected", [
    (f"http://127.0.0.1:{MAIN}", f"http://127.0.0.1:{MAIN}?observer=1"),
    (f"http://127.0.0.1:{MAIN}/", f"http://127.0.0.1:{MAIN}/?observer=1"),
    (f"http://127.0.0.1:{MAIN}/?a=b%20c&flag#/projects/p1",
     f"http://127.0.0.1:{MAIN}/?a=b%20c&flag&observer=1#/projects/p1"),
    (f"http://localhost:{MAIN}/#/settings", f"http://localhost:{MAIN}/?observer=1#/settings"),
    (f"http://ouroboros.lan:{MAIN}/", f"http://ouroboros.lan:{MAIN}/?observer=1"),  # a name the proof resolves
    (f"http://127.0.0.1:{MAIN}/?observer=1", f"http://127.0.0.1:{MAIN}/?observer=1"),  # already asked
    # The builtin's own visits stay observers: an exit in the address it was given does not take the mode away.
    (f"http://127.0.0.1:{MAIN}/?observer=0", f"http://127.0.0.1:{MAIN}/?observer=0&observer=1"),
])
def test_the_proven_ui_page_opens_as_an_observer_with_its_spelling_kept(tmp_path, monkeypatch, url, expected):
    ctx = _installation(tmp_path, monkeypatch)
    assert browser_policy.runtime_service_kind(url, ctx) == "main"
    assert browser._observer_url(url, ctx) == expected


@pytest.mark.parametrize("url", [
    f"http://127.0.0.1:{MAIN}/api/state",  # the UI's own API, files and assets are no view
    f"http://127.0.0.1:{MAIN}/api/files/download?path=notes.md",
    f"http://127.0.0.1:{MAIN}/static/app.js",
    f"http://127.0.0.1:{HOST}/",  # the Host Service, a proven service but not the UI
    f"http://127.0.0.1:{OTHER}/",  # an unrelated application, or a candidate UI on another port
    f"http://unresolvable.lan:{MAIN}/",  # the recorded port behind a name the proof cannot resolve: unknown
    "http://192.0.2.10:9474/",
    "file:///tmp/index.html",
])
def test_anything_but_the_proven_ui_page_opens_as_given(tmp_path, monkeypatch, url):
    ctx = _installation(tmp_path, monkeypatch)
    assert browser._observer_url(url, ctx) == url


def test_a_ui_bound_to_every_interface_is_proven_by_a_local_name(tmp_path, monkeypatch):
    ctx = _installation(tmp_path, monkeypatch, host="0.0.0.0")
    assert browser._observer_url(f"http://ouroboros.lan:{MAIN}/", ctx) == f"http://ouroboros.lan:{MAIN}/?observer=1"


@pytest.mark.parametrize("kind", ["local_model", "host_service", browser_policy.SERVICE_IDENTITY_UNKNOWN, "", "boom"])
def test_only_a_main_proof_asks_for_the_observer_mode(tmp_path, monkeypatch, kind):
    ctx = _installation(tmp_path, monkeypatch)

    def identity(_url, _ctx):
        if kind == "boom":
            raise RuntimeError("bindings unreadable")
        return kind

    monkeypatch.setattr(browser_policy, "runtime_service_kind", identity)
    assert browser._observer_url(f"http://127.0.0.1:{MAIN}/", ctx) == f"http://127.0.0.1:{MAIN}/"


class _RecordingPage(FakePage):
    def __init__(self):
        super().__init__(None)
        self.asked = []

    async def goto(self, url, **_kwargs):
        self.asked.append(url)
        self.response = chain(url)
        self.url = url
        return self.response


def test_browse_page_navigates_to_the_observer_address_only_for_the_ui(tmp_path, monkeypatch):
    ctx = _installation(tmp_path, monkeypatch)
    tools = ToolRegistry(tmp_path, ctx.drive_root)
    tools.set_context(ctx)
    page = _RecordingPage()
    monkeypatch.setattr(browser, "_ensure_browser", lambda ctx, *_a, **_k: (page, ctx.browser_state))
    try:
        results = [tools.execute("browse_page", {"url": url}) for url in (
            f"http://127.0.0.1:{MAIN}/#/projects", f"http://127.0.0.1:{OTHER}/start")]
    finally:
        close_generations(ctx.browser_state)
    assert page.asked == [f"http://127.0.0.1:{MAIN}/?observer=1#/projects", f"http://127.0.0.1:{OTHER}/start"], results
    assert all("FINAL_ALLOWED_CONTENT" in result for result in results), results


@pytest.mark.browser
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
                # The child's session closes before the root opens its own.
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
