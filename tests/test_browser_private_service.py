"""A delegated browser completes a real task against its granted private service."""
from __future__ import annotations

import base64
from hashlib import sha256
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import io
import json
import os
from pathlib import Path
import queue
import socket
import threading
from types import SimpleNamespace

import pytest
from PIL import Image

from ouroboros.browser_policy import runtime_service_kind
from ouroboros.contracts.task_constraint import TaskConstraint
from ouroboros.server_process import read_service_bindings
from ouroboros.tools import browser, vision
from ouroboros.tools.control import _schedule_task
from ouroboros.tools.registry import ToolContext, ToolRegistry
from ouroboros.vision_routing import VisionRoutingContext, prepare_messages_for_send
from tests.test_ui_smoke_playwright import direct_server_with_data  # noqa: F401
from tests._shared import configure_test_subagent


@pytest.fixture(params=["chromium", "webkit"])
def private_browser(request, direct_server_with_data, tmp_path, monkeypatch):  # noqa: F811
    engine = request.param
    host = os.environ.get("OUROBOROS_TEST_PRIVATE_BROWSER_HOST") or socket.gethostname()
    ips = [info[4][0] for info in socket.getaddrinfo(host, None, family=socket.AF_INET)
           if ipaddress.ip_address(info[4][0]).is_private and not ipaddress.ip_address(info[4][0]).is_loopback]
    if not ips:
        if os.environ.get("OUROBOROS_EXPECT_PRIVATE_BROWSER") == "1":
            pytest.fail("this host has no browser-visible private interface")
        pytest.skip("a real private interface is required")
    address = os.environ.get("OUROBOROS_TEST_PRIVATE_BROWSER_ADDRESS") or ips[0]
    assert address in ips, "the selected DNS name must resolve to the actual private service address"
    hits = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            hits.append(("GET", self.path))
            if self.path == "/redirect":
                self.send_response(302); self.send_header("Location", "/"); self.end_headers(); return
            if self.path == "/foreign-redirect":
                self.send_response(302); self.send_header("Location", foreign_origin); self.end_headers(); return
            body, mime = {
                "/style.css": (b"body{background:#e4f4ed;font:24px sans-serif;padding:32px}", "text/css"),
                "/app.js": (b"document.querySelector('#ready').textContent='Private service ready';", "application/javascript"),
                "/icon.svg": (b'<svg xmlns="http://www.w3.org/2000/svg" width="60" height="60"><rect width="60" height="60" fill="green"/></svg>', "image/svg+xml"),
            }.get(self.path, (b'''<!doctype html><link rel="stylesheet" href="/style.css"><h1 id="ready">Loading</h1>
                <img src="/icon.svg"><button id="act" onclick="fetch('/api/owner/context-mode',{method:'POST'}).then(r=>r.text()).then(t=>document.querySelector('#result').textContent=t)">Verify action</button>
                <p id="result"></p><script src="/app.js"></script>''', "text/html"))
            self.send_response(200); self.send_header("Content-Type", mime); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

        def do_POST(self):  # noqa: N802
            hits.append(("POST", self.path))
            body = b"Assigned service action succeeded"
            self.send_response(200); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

        def log_message(self, *_args):
            pass

    service = ThreadingHTTPServer((address, 0), Handler)
    thread = threading.Thread(target=service.serve_forever, daemon=True)
    thread.start()
    ctx = None
    foreign = foreign_thread = root_ctx = None
    try:
        origin = f"http://{address}:{service.server_port}"
        named_origin = f"http://{host}:{service.server_port}"
        canonical = direct_server_with_data["data_dir"]
        foreign_hits = []
        class Foreign(Handler):
            def do_GET(self):  # noqa: N802
                foreign_hits.append(self.path)
                super().do_GET()
        foreign = ThreadingHTTPServer((address, 0), Foreign)
        foreign_thread = threading.Thread(target=foreign.serve_forever, daemon=True)
        foreign_thread.start()
        foreign_origin = f"http://{address}:{foreign.server_port}"
        child = tmp_path / "child-data"
        ctx = ToolContext(repo_dir=tmp_path, drive_root=child, budget_drive_root=canonical,
                          task_constraint=TaskConstraint(mode="acting_subagent", surface="external_workspace", write_root=str(tmp_path)))
        ctx.task_metadata = {"budget_drive_root": str(canonical)}
        parent = ToolContext(repo_dir=tmp_path, drive_root=canonical, is_direct_chat=True)
        parent.task_id, parent.task_depth, parent.current_chat_id = "browser-parent", 0, 1
        parent.event_queue = queue.Queue()
        selected = configure_test_subagent(monkeypatch)
        scheduled = _schedule_task(parent, subagent_id=selected, objective="Verify the assigned private UI",
                                   expected_output="Screenshot and findings", allowed_origins=[origin, named_origin])
        assert "error" not in scheduled.lower(), scheduled
        ctx.task_contract = parent.event_queue.get_nowait()["task_contract"]
        ctx.messages = []
        tools = ToolRegistry(tmp_path, child)
        tools.set_context(ctx)
        root_ctx = ToolContext(repo_dir=tmp_path, drive_root=child, budget_drive_root=canonical)
        yield SimpleNamespace(ctx=ctx, tools=tools, canonical=canonical, engine=engine,
                              actual_main=direct_server_with_data["url"], origin=origin,
                              named_origin=named_origin, hits=hits, foreign_origin=foreign_origin,
                              foreign_hits=foreign_hits, root_ctx=root_ctx)
    finally:
        if ctx is not None:
            browser.cleanup_browser(ctx)
        if root_ctx is not None:
            browser.cleanup_browser(root_ctx)
        if foreign is not None:
            foreign.shutdown(); foreign.server_close(); foreign_thread.join(timeout=5)
            assert not foreign_thread.is_alive()
        service.shutdown(); service.server_close(); thread.join(timeout=5)
        assert not thread.is_alive()


@pytest.mark.browser
def test_granted_private_page_resources_actions_and_actual_control_identity(private_browser):
    case = private_browser
    ctx, tools, canonical, engine = case.ctx, case.tools, case.canonical, case.engine
    actual_main, origin, named_origin = case.actual_main, case.origin, case.named_origin
    hits, foreign_hits = case.hits, case.foreign_hits
    foreign_origin, root_ctx = case.foreign_origin, case.root_ctx
    bindings = read_service_bindings(canonical)
    assert {"main", "host_service"} <= set(bindings)
    assert runtime_service_kind(actual_main, ctx) == "main"
    assert "BROWSER_LOCAL_READONLY_BLOCKED" in tools.execute("browse_page", {"url": actual_main})
    host_service_url = f"http://127.0.0.1:{bindings['host_service']['port']}/identity"
    assert runtime_service_kind(host_service_url, ctx) == "host_service"
    assert "BROWSER_LOCAL_READONLY_BLOCKED" in tools.execute("browse_page", {"url": host_service_url})
    result = tools.execute("browse_page", {"url": origin + "/redirect", "engine": engine})
    assert "Private service ready" in result, result
    assert {("GET", "/style.css"), ("GET", "/app.js"), ("GET", "/icon.svg")} <= set(hits)
    assert "Clicked" in tools.execute("browser_action", {"action": "click", "selector": "#act"})
    assert json.loads(tools.execute("browser_action", {"action": "wait", "selector": "#result", "state": "visible"}))["status"] == "reached"
    assert "Assigned service action succeeded" in ctx.browser_state.page.inner_text("#result")
    assert ("POST", "/api/owner/context-mode") in hits
    denied_fetch = tools.execute("browser_action", {"action": "evaluate",
        "value": f"() => fetch({json.dumps(foreign_origin)}).then(() => 'unexpected').catch(() => 'denied')"})
    assert "origin_not_granted" in denied_fetch and not foreign_hits
    assert "origin_not_granted" in tools.execute("browse_page", {"url": foreign_origin, "engine": engine})
    assert not foreign_hits
    # Real tasks own separate worker threads. This sequential test retires one
    # actor before opening another Playwright Sync API loop on the same thread.
    browser.cleanup_browser(ctx)
    assert "Private service ready" in browser._browse_page(root_ctx, foreign_origin, engine=engine)
    assert foreign_hits
    before_settings = sha256((canonical / "settings.json").read_bytes()).hexdigest()
    forbidden_url = actual_main + "/api/owner/context-mode"
    browser._browse_page(root_ctx, actual_main, engine=engine)
    rejected = browser._browser_action(root_ctx, "evaluate", value=
        f"() => fetch({json.dumps(forbidden_url)},{{method:'POST',body:JSON.stringify({{mode:'max'}})}}).catch(() => 'denied')")
    assert "BROWSER_OWNER_CONTROL_BLOCKED" in rejected
    assert sha256((canonical / "settings.json").read_bytes()).hexdigest() == before_settings
    browser.cleanup_browser(root_ctx)
    assert "Private service ready" in tools.execute("browse_page", {"url": named_origin, "engine": engine})
    ctx.task_constraint = TaskConstraint(mode="local_readonly_subagent", allow_enable=False)
    assert "Private service ready" in tools.execute("browse_page", {"url": origin, "engine": engine})
    assert "cannot run arbitrary" in tools.execute("browser_action", {"action": "evaluate", "value": "1+1"})
    assert "Screenshot captured" in tools.execute("browser_action", {"action": "screenshot"})
    raw = base64.b64decode(ctx.browser_state.last_screenshot_b64)
    shot = canonical / "state/skills/browser-proof/jobs/proof/output/screen.png"
    shot.parent.mkdir(parents=True, exist_ok=True); shot.write_bytes(raw)
    assert vision.attach_local_image_to_context(ctx, str(shot))[0]
    sent = prepare_messages_for_send(ctx.messages, routing=VisionRoutingContext("google/gemini-3.5-flash", object(), {}))
    block = next(b for m in sent if isinstance(m.get("content"), list) for b in m["content"] if b.get("type") == "image_url")
    prepared = base64.b64decode(block["image_url"]["url"].split(",", 1)[1])
    assert block["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert shot.read_bytes() == raw
    assert Path(block["_source_path"]).read_bytes() == prepared
    with Image.open(io.BytesIO(raw)) as source, Image.open(io.BytesIO(prepared)) as received:
        assert source.size == (1920, 1080) and source.format == "PNG"
        assert received.size == (1600, 900) and received.format == "JPEG"
        icon = ctx.browser_state.page.locator("img").bounding_box()
        center = (round((icon["x"] + icon["width"] / 2) * 1600 / 1920),
                  round((icon["y"] + icon["height"] / 2) * 900 / 1080))
        assert all(abs(actual - expected) < 15 for actual, expected in zip(received.getpixel(center), (0, 128, 0)))
    if out := os.environ.get("OUROBOROS_BROWSER_EVIDENCE_OUT"):
        Path(out).mkdir(parents=True, exist_ok=True)
        (Path(out) / f"private-service-{engine}.png").write_bytes(raw)
        (Path(out) / f"private-service-{engine}-vision.jpg").write_bytes(prepared)


@pytest.mark.browser
@pytest.mark.xfail(strict=True, reason=(
    "Owner-accepted residual (external audit 2026-09-06, item 2/6, answer A): Chromium and "
    "WebKit follow HTTP redirects natively and Playwright route callbacks see only the first "
    "URL of a chain, so the forbidden hop's GET is dispatched before the post-navigation chain "
    "check withholds the content. Pre-request prevention is not promised; this server counter "
    "stays a strict negative control so a change in the native behaviour is a visible fact, "
    "never a silently weakened oracle."))
def test_redirect_to_ungranted_private_service_is_denied_before_dispatch(private_browser):
    """Server-observed dispatch is independent of the later content refusal.

    The content refusal itself is proven elsewhere (the chain check); this
    control asks the stronger question the native transport cannot answer.
    """
    case = private_browser
    result = case.tools.execute("browse_page", {
        "url": case.origin + "/foreign-redirect", "engine": case.engine})
    assert "origin_not_granted" in result, result
    assert not case.foreign_hits, "the forbidden redirect request was sent before post-navigation refusal"
