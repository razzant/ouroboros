"""Real browser return/startup/auth; provider exchange is a deterministic fixture.

The fixture uses the real builtin mount and NetworkAuthGate. It does not grant
provider access or qualify Google client registration, PKCE, or a native window.
"""
from __future__ import annotations

import base64
from contextlib import contextmanager
import json
import os
from pathlib import Path
import socket
import threading
import time
from urllib.parse import urlencode

import pytest

pytestmark = pytest.mark.ui_browser
REPO = Path(__file__).resolve().parents[1]


@contextmanager
def return_server(monkeypatch, *, remote, prefix):
    import uvicorn
    from starlette.applications import Starlette
    from starlette.responses import HTMLResponse, JSONResponse, PlainTextResponse
    from starlette.routing import Route
    from ouroboros import server_auth
    from ouroboros.server_web import ServedUI, UiSelection, _Generation

    monkeypatch.setattr(server_auth, "get_configured_network_password", lambda: "fixture-password" if remote else "")
    monkeypatch.setattr(server_auth, "_auth_secret_cache", b"\x02" * 32)
    seen, completed = [], []
    async def complete(request):
        payload = await request.json()
        completed.append(payload)
        if payload.get("error"):
            return JSONResponse({"ok": False, "error": "Permission was not granted."}, status_code=400)
        return JSONResponse({"ok": True, "email": "long.personal.mailbox@example.test"})
    async def language(_request):
        return JSONResponse({"language": "en", "entries": {}})
    async def identity(request):
        account = request.query_params.get("account_id", "A")
        readers = [{"id": f"reader-{i}", "kind": "user", "label": f"long.mailbox.alias-{i:02}@example.test"}
                   for i in range(20)] if account == "B" else []
        return JSONResponse({"account_id": account, "text": f"Account {account} ready", "tone": "ok",
                             "identity": {"sends_as": {"id": f"writer-{account}", "kind": "user",
                                                        "label": f"mailbox-{account}@example.test"}, "reads_as": readers}})
    async def places_view(_request):
        return HTMLResponse('''<!doctype html><html class="ouro-ui"><head><meta name="viewport" content="width=device-width">
        <script src="/onboarding/static/theme.js"></script>
        <link rel="stylesheet" href="/onboarding/static/ui.css"><link rel="stylesheet" href="/onboarding/static/style.css">
        <link rel="stylesheet" href="/onboarding/static/settings.css"><link rel="stylesheet" href="/onboarding/static/model_roles.css">
        <link rel="stylesheet" href="/onboarding/static/consciousness.css"></head><body><script type="module">
        import {openPlaceSheet} from '/onboarding/static/modules/place_sheet.js';
        import {normalizePlace} from '/onboarding/static/modules/consciousness_api.js';
        window.openAccount = account => openPlaceSheet(normalizePlace({place_id: account, account_id: account,
          connector: 'calendar', provider: 'service', label: 'Mailbox ' + account, state: 'active', revision: 1, safety_mode: 'full',
          source: {url: '/api/extensions/calendar/status'}, facts: ['direct_message'], owner_accounts: []}),
          {readRooms: async () => ({rooms: []}), readModels: async () => ({models: []})});
        </script></body></html>''')
    served = ServedUI(REPO)
    served._current = _Generation(UiSelection("none"), None)
    app = Starlette(routes=[Route("/api/extensions/calendar/oauth/complete", complete, methods=["POST"]),
                            Route("/api/extensions/calendar/status", identity), Route("/fixture", places_view),
                            Route("/api/ui/i18n", language), *served.mounts()])
    gate = server_auth.NetworkAuthGate(app, served_ui=served)
    async def proxy(scope, receive, send):
        if scope["type"] == "http":
            seen.append((scope["method"], scope["path"], dict(scope["headers"]).get(b"sec-fetch-site")))
            if prefix:
                if not scope["path"].startswith(prefix + "/"):
                    await PlainTextResponse("Not found", status_code=404)(scope, receive, send)
                    return
                scope = {**scope, "path": scope["path"][len(prefix):]}
            if remote:
                scope = {**scope, "client": ("198.51.100.20", 1234)}
        await gate(scope, receive, send)
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    server = uvicorn.Server(uvicorn.Config(proxy, log_level="warning", lifespan="off"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert server.started
    try:
        yield f"http://127.0.0.1:{sock.getsockname()[1]}", seen, completed
    finally:
        server.should_exit = True
        thread.join(10)
        sock.close()
        assert not thread.is_alive()


@pytest.mark.parametrize("remote,prefix,theme", [(False, "/instance", "light"), (True, "", "dark")])
def test_provider_return_runs_real_page_after_navigation_and_owner_login(monkeypatch, remote, prefix, theme):
    if os.environ.get("OUROBOROS_RUN_UI_SMOKE") != "1":
        pytest.skip("set OUROBOROS_RUN_UI_SMOKE=1 to run browser UI smoke")
    from playwright.sync_api import sync_playwright

    state = base64.urlsafe_b64encode(json.dumps({"v": 1, "skill": "calendar", "nonce": "fixture"}).encode()).decode().rstrip("=")
    with return_server(monkeypatch, remote=remote, prefix=prefix) as (origin, seen, completed), sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 390, "height": 844}, color_scheme=theme)
        try:
            page = context.new_page()
            page.route("https://provider.example.test/start", lambda route: route.fulfill(content_type="text/html", body="<a id='return'>Return</a>"))
            address = origin + prefix + "/onboarding/static/provider-return.html?" + urlencode({"code": "fixture-code", "state": state})
            page.goto("https://provider.example.test/start")
            page.locator("#return").evaluate("(node, address) => node.href = address", address)
            page.locator("#return").click()
            if remote:
                page.locator("#password").fill("fixture-password")
                page.locator("button[type=submit]").click()
            page.locator('[data-return-status][data-tone="ok"]').wait_for()
            assert page.locator("[data-return-account]").inner_text() == "long.personal.mailbox@example.test"
            assert completed == [{"code": "fixture-code", "state": state}]
            assert "code=" not in page.url and "state=" not in page.url
            assert any(method == "GET" and "provider-return.html" in path and site == b"cross-site" for method, path, site in seen)
            assert any(method == "POST" and path.endswith("/oauth/complete") and site == b"same-origin" for method, path, site in seen)
            assert page.locator("[data-return-home]").get_attribute("href") == origin + prefix + "/"
            assert not page.evaluate("document.documentElement.scrollWidth > innerWidth")
            evidence = os.environ.get("OUROBOROS_UI_EVIDENCE_DIR")
            if evidence:
                Path(evidence).mkdir(parents=True, exist_ok=True)
                page.screenshot(path=str(Path(evidence) / f"provider-return-{theme}.png"))
            page.goto(origin + prefix + "/onboarding/static/provider-return.html?" + urlencode({"error": "access_denied", "state": state}))
            page.locator('[data-return-status][data-tone="warn"]').wait_for()
            assert page.locator("[data-return-detail]").inner_text() == "Permission was not granted."
            assert completed[-1]["error"] == "access_denied"
        finally:
            context.close()
            browser.close()


@pytest.mark.parametrize("width", [390, 1000])
def test_place_sheet_paints_each_accounts_identity_and_readiness(monkeypatch, width):
    if os.environ.get("OUROBOROS_RUN_UI_SMOKE") != "1":
        pytest.skip("set OUROBOROS_RUN_UI_SMOKE=1 to run browser UI smoke")
    from playwright.sync_api import sync_playwright

    with return_server(monkeypatch, remote=False, prefix="") as (origin, _seen, _completed), sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            page = browser.new_page(viewport={"width": width, "height": 900})
            page.goto(origin + "/fixture")
            page.wait_for_function("typeof window.openAccount === 'function'")
            for account in ("A", "B"):
                page.evaluate("account => window.openAccount(account)", account)
                sheet = page.locator('[data-place-section="accounts"]')
                sheet.get_by_text(f"mailbox-{account}@example.test", exact=False).wait_for()
                assert f"Account {account} ready" in page.locator("[data-place-body]").inner_text()
                assert sheet.locator('[data-place-act="confirm"]').get_attribute("data-actor") == f"writer-{account}"
                assert f"mailbox-{'B' if account == 'A' else 'A'}@example.test" not in sheet.inner_text()
                if account == "B":
                    assert "long.mailbox.alias-19@example.test" in sheet.inner_text()
                    assert sheet.locator("[data-i18n-authored]").count() == 21
                assert page.locator("[data-place-body]").evaluate("node => node.scrollWidth <= node.clientWidth")
                evidence = os.environ.get("OUROBOROS_UI_EVIDENCE_DIR")
                if evidence:
                    Path(evidence).mkdir(parents=True, exist_ok=True)
                    sheet.scroll_into_view_if_needed()
                    page.screenshot(path=str(Path(evidence) / f"place-account-{account}-{width}.png"))
                page.locator("[data-place-close]").click()
        finally:
            browser.close()
