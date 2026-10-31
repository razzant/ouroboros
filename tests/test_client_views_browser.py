"""Selected client entry pages through a real core and a non-loopback password gate.

The existing separate-host tests exercise ``UI_DIR=none``. Here every TLS request reaches
the core, which selects the example's main/setup/login files. The proxy supplies a synthetic
nonlocal visitor because the real browser runs on loopback. No owner keys or paid models.
"""
from __future__ import annotations

import json
import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse

import pytest

from tests.separate_ui_support import (
    PASSWORD,
    REPO_ROOT,
    SYNTHETIC_PEER,
    BrowserSession,
    Caddy,
    caddy_binary,
    copy_interface,
    evidence_dir,
    free_port,
    make_certificate,
    real_core,
    wait_until,
)

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]
EXAMPLE = REPO_ROOT / "docs/examples/client_views"
BROWSERS = ["chromium", "webkit"]


def _client_files(tmp_path: Path, *, copied_builtin: bool = False) -> Path:
    client = tmp_path / "client"
    if copied_builtin:
        copy_interface(REPO_ROOT / "web", client)
        for name in ("onboarding.html", "setup.js"):
            shutil.copy2(EXAMPLE / name, client / name)
        shutil.copytree(EXAMPLE / "auth", client / "auth")
    else:
        shutil.copytree(EXAMPLE, client)
    return client


@contextmanager
def _selected_core(tmp_path, request, client):
    if os.environ.get("OUROBOROS_RUN_UI_SMOKE") != "1":
        pytest.skip("set OUROBOROS_RUN_UI_SMOKE=1 for real client-view qualification")
    binary = caddy_binary()
    if binary is None:
        pytest.skip("SEPARATE_UI_CADDY must name the existing Caddy binary")
    evidence = evidence_dir(request.node.name, tmp_path)
    proxy = None
    core = None
    try:
        with real_core(tmp_path, provider=False, ui_dir=client) as core:
            root = tmp_path / "proxy"
            root.mkdir()
            certificate = make_certificate(root / "tls")
            port = free_port()
            site = root / "site.Caddyfile"
            site.write_text("""{$SITE} {
    tls {$CERT} {$KEY}
    log
    log_append upstream {http.reverse_proxy.upstream.hostport}
    reverse_proxy {$CORE} {
        header_up X-Forwarded-For """ + SYNTHETIC_PEER + """
    }
}
""", encoding="utf-8")
            proxy = Caddy("client-proxy", binary, root / "process", site=site, port=port,
                          variables={"SITE": f"https://127.0.0.1:{port}", "CERT": certificate.cert,
                                     "KEY": certificate.key, "CORE": f"127.0.0.1:{core.port}"})
            try:
                proxy.start()
                yield SimpleNamespace(core=core, proxy=proxy, certificate=certificate,
                                      url=f"https://127.0.0.1:{port}", evidence=evidence)
            finally:
                proxy.stop()
    finally:
        for process in (proxy, core.process if core else None):
            if process and process.log_path.exists():
                shutil.copy2(process.log_path, evidence / process.log_path.name)
        if proxy is not None:
            (evidence / "routing.json").write_text(json.dumps(proxy.access(), indent=2), encoding="utf-8")


def _session(playwright, stack, browser_name):
    viewport = (390, 844) if browser_name == "webkit" else (1360, 900)
    return BrowserSession(playwright, stack, browser_name, stack.evidence, viewport=viewport)


def _custom_sign_in(page, password=PASSWORD):
    page.get_by_label("Password", exact=True).fill(password)
    page.get_by_role("button", name="Sign in", exact=True).click()


def _check_geometry(page):
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), "horizontal overflow"


def _painted_document(document):
    """Wait for the document's real load/style/font work, then a browser paint opportunity.

    DOM visibility alone precedes a nested frame's first composited frame in Chromium.
    Pixels still require inspection; these conditions do not declare a screenshot correct.
    """
    document.locator("html").evaluate("""async () => {
        if (document.readyState !== 'complete') {
            await new Promise(resolve => window.addEventListener('load', resolve, {once: true}));
        }
        await document.fonts.ready;
        const styles = [...document.querySelectorAll('link[rel="stylesheet"]')];
        if (styles.some(link => !link.sheet)) throw new Error('A stylesheet did not load');
        document.body.getBoundingClientRect();
        await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
    }""")


def _keyless_models_step(document, endpoint):
    """The real Accounts → Models route, without a subscription-only quick-start button."""
    document.locator("#next-btn").wait_for(timeout=30_000)
    for section in ("api-access", "more-providers"):
        details = document.locator(f'[data-collapse="{section}"]')
        if details.get_attribute("open") is None:
            details.locator("summary").first.click()
    document.locator("#openai-compatible-url").fill(endpoint)
    document.locator("#next-btn").click()
    document.locator('[data-model-role="main"]').wait_for()
    _painted_document(document)


@pytest.mark.parametrize("browser_name", BROWSERS)
@pytest.mark.parametrize("scheme", ["http", "https"])
def test_example_message_ids_work_on_insecure_and_secure_origins(browser_name, scheme):
    """Execute exact example bytes in real browser origins; network/API/WS are small stubs.

    This pins Web Crypto availability, not TLS or backend behaviour (the real-core cases
    below own those claims). Restoring randomUUID must fail on the insecure origin.
    """
    if os.environ.get("OUROBOROS_RUN_UI_SMOKE") != "1":
        pytest.skip("set OUROBOROS_RUN_UI_SMOKE=1 for browser API qualification")
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        browser = getattr(playwright, browser_name).launch(headless=True)
        try:
            context = browser.new_context(ignore_https_errors=True)
            page = context.new_page()
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.add_init_script("""window.sent = [];
                window.WebSocket = class {
                    static OPEN = 1;
                    readyState = 1;
                    addEventListener(type, callback) {
                        if (type === 'open') setTimeout(callback, 0);
                    }
                    send(value) { window.sent.push(JSON.parse(value)); }
                    close() {}
                };""")

            def route(request):
                path = urlparse(request.request.url).path
                if path == "/api/onboarding/bootstrap":
                    request.fulfill(json={"ready": True, "bootstrap": {}})
                elif path == "/api/state":
                    request.fulfill(json={"served_ui": {"source": "directory"}})
                else:
                    member = "index.html" if path == "/" else path.removeprefix("/static/").lstrip("/")
                    source = EXAMPLE / member
                    kind = "text/javascript" if source.suffix == ".js" else (
                        "text/css" if source.suffix == ".css" else "text/html")
                    if source.is_file():
                        request.fulfill(body=source.read_bytes(), content_type=kind)
                    else:
                        request.fulfill(status=404, body="")

            page.route("**/*", route)
            page.goto(f"{scheme}://client-views.test/")
            page.locator("#send:not([disabled])").wait_for()
            assert page.evaluate("isSecureContext") is (scheme == "https")
            for text in ("First message", "Second message"):
                page.locator("#message").fill(text)
                page.locator("#send").click()
            sent = page.evaluate("window.sent")
            assert [item["content"] for item in sent] == ["First message", "Second message"]
            ids = [item["client_message_id"] for item in sent]
            assert all(isinstance(value, str) and value for value in ids) and len(set(ids)) == 2
            assert not errors
        finally:
            browser.close()


@pytest.mark.parametrize("browser_name", BROWSERS)
def test_independent_pages_sign_in_configure_and_use_the_real_core(tmp_path, request, browser_name):
    """No private SPA imports: source bytes, refusal/session, atomic save, HTTP and real WS."""
    from playwright.sync_api import sync_playwright

    client = _client_files(tmp_path)
    with _selected_core(tmp_path, request, client) as stack, sync_playwright() as playwright:
        core = stack.core
        settings = core.data_dir / "settings.json"
        assert not settings.exists()
        session = _session(playwright, stack, browser_name)
        page = session.page
        try:
            page.goto(stack.url + "/?entry=custom")
            page.get_by_role("heading", name="Welcome to your client").wait_for()
            assert page.url.startswith(stack.url + "/auth/login?")
            assert page.locator("body").get_attribute("data-client-view") == "login"
            assert session.context.cookies() == []
            for path in ("/auth/style.css", "/auth/login.js"):
                result = session.api("GET", path)
                assert result["status"] == 200, (path, result)
            for path in ("/api/state", "/api/onboarding/bootstrap", "/static/client.js",
                         "/onboarding/static/ui.css", "/api/extensions/not-installed"):
                assert session.api("GET", path)["status"] == 401, path
            assert session.api("GET", "/auth/ui.css")["status"] == 404  # builtin ui.css is not a fallback
            assert not settings.exists(), "page/resource/readiness reads must not author settings"
            _check_geometry(page)
            session.shot("1-custom-sign-in")

            with page.expect_response(lambda r: r.url.endswith("/auth/login") and r.request.method == "POST") as wrong:
                _custom_sign_in(page, "wrong-password")
            assert wrong.value.status == 401
            page.get_by_role("alert").filter(has_text="Invalid password").wait_for()
            assert not session.context.cookies()
            assert page.locator("body").get_attribute("data-client-view") == "login"
            session.shot("2-wrong-password")

            with page.expect_response(lambda r: r.url.endswith("/auth/login") and r.request.method == "POST") as signed:
                _custom_sign_in(page)
            assert signed.value.status == 200
            page.get_by_role("heading", name="Set up your client").wait_for()
            assert any(url == stack.url + "/?entry=custom" for url in session.loads())
            cookie = next(c for c in session.context.cookies() if c["name"] == "ouroboros_auth")
            assert cookie["secure"] and cookie["httpOnly"] and cookie["sameSite"] == "Lax"
            page.locator("#save:not([disabled])").wait_for()
            assert not settings.exists()
            assert session.api("GET", "/api/onboarding/bootstrap")["json"]["ready"] is False
            assert session.api("GET", "/static/modules/onboarding_wizard.js")["status"] == 404
            page.get_by_label("Endpoint URL").fill(core.model.base_url)
            page.get_by_label("Model ID").fill("mock-model")
            _check_geometry(page)
            session.shot("3-custom-setup")
            # A partial 200 must not close setup or silently re-submit a write.
            page.route("**/api/onboarding/complete",
                       lambda route: route.fulfill(status=200, json={"ok": True}), times=1)
            page.get_by_role("button", name="Save setup").click()
            page.get_by_role("status").filter(has_text="incomplete completion receipt").wait_for()
            assert page.get_by_role("button", name="Save setup").is_disabled()
            assert not settings.exists()
            page.get_by_role("button", name="Check current status").click()
            page.get_by_role("status").filter(has_text="No save is retried automatically").wait_for()
            assert not settings.exists()
            # An explicit page reload starts another draft; the genuine receipt below works.
            page.reload()
            page.locator("#save:not([disabled])").wait_for()
            page.get_by_label("Endpoint URL").fill(core.model.base_url)
            page.get_by_label("Model ID").fill("mock-model")
            with page.expect_response(lambda r: r.url.endswith("/api/onboarding/complete")) as saved:
                page.get_by_role("button", name="Save setup").click()
            assert saved.value.status == 200
            assert saved.value.request.post_data_json["OPENAI_COMPATIBLE_BASE_URL"] == core.model.base_url
            page.get_by_role("heading", name="Your own interface").wait_for(timeout=60_000)
            page.get_by_role("status").filter(has_text="Connected").wait_for(timeout=60_000)
            assert page.locator("body").get_attribute("data-served-ui") == "directory"
            assert session.api("GET", "/api/onboarding")["status"] == 204
            core.wait_ready()
            core.prove_origin()

            page.get_by_label("Message", exact=True).fill("A message from the independent client")
            page.get_by_role("button", name="Send", exact=True).click()
            page.locator('#messages [data-role="user"][data-accepted="true"]').wait_for(timeout=60_000)
            page.locator('#messages [data-role="assistant"]').filter(has_text="OK").wait_for(timeout=60_000)
            session.shot("4-custom-main")
            _check_geometry(page)
            assert session.unexpected_errors() == []
            written = json.loads(settings.read_text(encoding="utf-8"))
            assert written["OPENAI_COMPATIBLE_BASE_URL"] == core.model.base_url
            assert not written.get("OPENAI_COMPATIBLE_API_KEY")
            assert written.get("OUROBOROS_ONBOARDING_COMPLETED_AT")
            assert "OUROBOROS_UI_DIR" not in written
            completions = [r for r in core.jsonl("logs/events.jsonl")
                           if r.get("type") == "owner_api_action" and r.get("action") == "onboarding_complete"]
            assert len(completions) == 1
            assert any(call["main"] for call in core.model.calls)
        finally:
            session.close()
        wait_until(lambda: any(r["request"]["uri"] == "/ws" and r.get("status") == 101
                               for r in stack.proxy.access()), 15, "proxy has no completed WS upgrade record")
        records = stack.proxy.access()
        assert {r.get("upstream") for r in records} == {f"127.0.0.1:{core.port}"}
        assert not [r for r in records if r["request"]["uri"].startswith("/onboarding/static/")
                    and r.get("status") == 200], "custom journey borrowed builtin resources"


@pytest.mark.parametrize("browser_name", BROWSERS)
def test_copied_builtin_parent_recovers_a_broken_selected_setup(tmp_path, request, browser_name):
    """The real copied overlay has an action outside the unusable custom iframe."""
    from playwright.sync_api import sync_playwright

    client = _client_files(tmp_path, copied_builtin=True)
    (client / "onboarding.html").write_text(
        '<!doctype html><html><head><link rel="stylesheet" href="/auth/style.css"></head>'
        '<body><main><h1>Broken custom setup</h1></main>'
        '<script>throw new Error("deliberate custom setup failure")</script></body></html>', encoding="utf-8")
    (client / "modules/onboarding_wizard.js").write_text(
        'throw new Error("selected wizard must not run in builtin recovery");', encoding="utf-8")
    with _selected_core(tmp_path, request, client) as stack, sync_playwright() as playwright:
        session = _session(playwright, stack, browser_name)
        page = session.page
        try:
            page.goto(stack.url + "/")
            page.get_by_role("heading", name="Welcome to your client").wait_for()
            _custom_sign_in(page)
            frame = page.frame_locator("iframe.onboarding-frame")
            frame.get_by_role("heading", name="Broken custom setup").wait_for(timeout=60_000)
            recovery = page.get_by_role("button", name="Open built-in setup", exact=True)
            assert recovery.get_attribute("data-onboarding-recovery") is not None
            assert recovery.is_visible()
            bounds = recovery.bounding_box()
            assert bounds and bounds["x"] >= 0 and bounds["x"] + bounds["width"] <= page.viewport_size["width"]
            _painted_document(frame)
            _painted_document(page)
            session.shot("1-broken-custom-frame")
            recovery.click()
            assert "builtin=1" in page.locator("iframe.onboarding-frame").get_attribute("src")
            _keyless_models_step(frame, stack.core.model.base_url)
            _painted_document(page)
            session.shot("2-builtin-recovery")
            _check_geometry(page)
            assert not (stack.core.data_dir / "settings.json").exists()
            assert [e for e in session.unexpected_errors() if "deliberate custom setup failure" not in e] == []
        finally:
            session.close()
        assert any(r["request"]["uri"].startswith("/onboarding/static/modules/onboarding_wizard.js")
                   and r.get("status") == 200 for r in stack.proxy.access())
        assert not any(r["request"]["uri"].startswith("/static/modules/onboarding_wizard.js")
                       for r in stack.proxy.access())


@pytest.mark.parametrize("custom_failure", ["missing-views", "broken-login"])
def test_builtin_recovery_is_a_whole_working_page_before_and_after_auth(tmp_path, request, custom_failure):
    from playwright.sync_api import sync_playwright

    client = _client_files(tmp_path)
    if custom_failure == "missing-views":
        (client / "auth/index.html").unlink()
        (client / "onboarding.html").unlink()
        entry = "/onboarding"
    else:
        (client / "auth/index.html").write_text('<h1>Broken custom login</h1>', encoding="utf-8")
        entry = "/onboarding?builtin=1"
    with _selected_core(tmp_path, request, client) as stack, sync_playwright() as playwright:
        session = _session(playwright, stack, "chromium")
        page = session.page
        try:
            page.goto(stack.url + entry)
            page.get_by_role("button", name="Unlock", exact=True).wait_for()
            assert page.get_by_role("heading", name="Broken custom login").count() == 0
            session.sign_in("wrong-password")
            page.get_by_text("Invalid password.").wait_for()
            assert not session.context.cookies()
            session.sign_in()
            page.locator("#next-btn").wait_for(timeout=30_000)
            assert page.locator("body[data-client-view]").count() == 0
            _keyless_models_step(page, stack.core.model.base_url)
            session.shot(custom_failure + "-builtin-setup")
            assert not (stack.core.data_dir / "settings.json").exists()
            assert session.unexpected_errors() == []
        finally:
            session.close()
        assert not [r for r in stack.proxy.access() if r["request"]["uri"] in ("/auth/login.js", "/auth/style.css")], (
            "builtin recovery loaded custom assets")
