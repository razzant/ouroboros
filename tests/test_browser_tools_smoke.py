from __future__ import annotations

import base64
import http.server
import json
import os
import pathlib
import socketserver
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from ouroboros import owner_pause
from ouroboros.task_results import write_task_result
from ouroboros.tools.browser import _browse_page, _browser_action, cleanup_browser
from ouroboros.contracts.task_constraint import TaskConstraint
from ouroboros.server_entrypoint import bound_service_socket
from ouroboros.tools.registry import ToolContext, ToolRegistry


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
        cleanup_browser(ctx)  # Independent Sync API contexts cannot nest on one thread.
        local = tmp_path / "outside-start" / "index.html"
        local.parent.mkdir()
        local.write_text("<h1>Parent-readable local artifact</h1>", encoding="utf-8")
        try:
            assert "Parent-readable local artifact" in _browse_page(subagent_ctx, url=local.as_uri())
            assert "BROWSER_LOCAL_READONLY_BLOCKED" in _browser_action(subagent_ctx, action="evaluate", value="1 + 1")
        finally:
            cleanup_browser(subagent_ctx)

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


# #1606: the repaired built-in, through the registry and its sticky executor, against the
# production module-widget mount (web/modules/widget_module.js) and a plain iframe.
_REPO = pathlib.Path(__file__).resolve().parents[1]
_WIDGET = ("const root=document.querySelector('#root'); root.innerHTML=`<h2>VISIBLE_MODULE_CONTROL</h2>"
           "<button id=\"bridge\">Read through bridge</button><p id=\"answer\">Not read yet</p>`;"
           "document.querySelector('#bridge').onclick=async()=>{const r=await OuroborosWidget.fetch("
           "'/api/extensions/repro/probe'); document.querySelector('#answer').textContent=await r.text();};")
_WIDGET_PAGE = """<!doctype html><html lang="en"><meta charset="utf-8"><title>srcdoc observer</title>
<body><h1>VISIBLE_PARENT_CONTROL</h1><iframe id="plain" src="/plain.html"></iframe>
<section><div id="mount"></div></section><pre id="status">Loading</pre>
<script type="module">import {mountModuleWidget} from '/web/modules/widget_module.js';
window.dispose = await mountModuleWidget(document.querySelector('#mount'), {skill: 'repro', ws_prefix: 'repro.'},
  {kind: 'module', entry: 'widget.js', height: 240});
document.querySelector('#status').textContent = 'Parent mounted';
window.probe = () => { const f = document.querySelector('#mount iframe');
  return {srcdocLength: f.srcdoc.length, csp: f.srcdoc.includes('Content-Security-Policy'),
          sandbox: f.getAttribute('sandbox')}; };</script></body></html>"""


@pytest.fixture()
def widget_page_url():
    bodies = {"/": (_WIDGET_PAGE, "text/html"), "/plain.html": ("<p id='plain-ok'>PLAIN_FRAME_OK</p>", "text/html"),
              "/api/extensions/repro/module/widget.js": (_WIDGET, "text/javascript"),
              "/api/extensions/repro/probe": ("BRIDGE_CONSUMER_OK", "text/plain")}

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(_REPO), **kwargs)

        def do_GET(self):  # noqa: N802 - stdlib callback name
            if self.path not in bodies:
                return super().do_GET()
            body, kind = bodies[self.path][0].encode(), bodies[self.path][1]
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            return

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_builtin_renders_the_production_srcdoc_widget_and_its_bridge(tmp_path, monkeypatch, widget_page_url, engine):
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    import ouroboros.safety as safety

    monkeypatch.setattr(safety, "check_safety", lambda *_a, **_k: (True, ""))
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
    write_task_result(tmp_path, "root", "running", root_task_id="root")
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx.task_id = registry._ctx.root_task_id = "root"
    expected = engine in _EXPECTED_BROWSER_ENGINES or "all" in _EXPECTED_BROWSER_ENGINES
    frame = "#mount iframe >> internal:control=enter-frame >> "
    with ThreadPoolExecutor(max_workers=1) as executor:
        def call(name, **args):
            return owner_pause.submit_tool(registry._ctx, name, executor.submit, registry.execute_result,
                                           name, args).result()
        try:
            page = call("browse_page", url=widget_page_url, engine=engine, wait_for="text=Parent mounted")
            if "Executable doesn't exist" in page.text or "playwright install" in page.text.lower():
                if expected:
                    pytest.fail(page.text[:500])
                pytest.skip(page.text[:300])
            assert "VISIBLE_PARENT_CONTROL" in (page.producer_text or ""), page.text
            conditions = page.host_annotations[-1]
            assert conditions.startswith(f"Browser conditions: {engine} ") and "except iframe.contentWindow" in conditions
            probe = call("browser_action", action="evaluate", value="JSON.stringify(window.probe())")
            facts = json.loads(probe.producer_text)
            assert facts["srcdocLength"] > 1000 and facts["csp"], facts  # the native setter ran
            assert facts["sandbox"] == "allow-scripts allow-pointer-lock allow-downloads"  # opaque origin kept
            assert probe.host_annotations == (conditions,)
            plain = call("browser_action", action="wait", selector="#plain >> internal:control=enter-frame >> #plain-ok")
            assert json.loads(plain.text)["status"] == "reached", plain.text
            visible = call("browser_action", action="wait", selector=frame + "text=VISIBLE_MODULE_CONTROL")
            assert json.loads(visible.text)["status"] == "reached", visible.text
            assert call("browser_action", action="click", selector=frame + "#bridge").status == "ok"
            answered = call("browser_action", action="wait", selector=frame + "#answer:has-text('BRIDGE_CONSUMER_OK')")
            assert json.loads(answered.text)["status"] == "reached", answered.text
            shot = call("browser_action", action="screenshot")
            assert shot.host_annotations == (conditions,)
            raw = base64.b64decode(registry._ctx.browser_state.last_screenshot_b64 or "")
            assert raw.startswith(b"\x89PNG")
            (tmp_path / f"srcdoc-widget-{engine}.png").write_bytes(raw)  # retained for visual inspection
        finally:
            executor.submit(cleanup_browser, registry._ctx).result()
