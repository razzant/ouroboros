"""Screens before the interface wear the owner's appearance, in real engines (#1111, owner decision A).

Real here: WebKit and Chromium; the gate's own login form through ``NetworkAuthGate`` on a live socket,
seen by a peer beyond loopback (a phone on the network), then its Unlock; the Telegram sidecar's own
bootstrap response, script and Content-Security-Policy; the launcher's own pre-server window pages (its
``html=`` arguments, evaluated from launcher.py). Stand-ins: the device's appearance
(``prefers-color-scheme`` emulation); storage the browser blocks (a ``localStorage`` getter that throws,
as with blocked site data); Telegram's SDK (a stub reporting its colour scheme); the network between
Telegram and the sidecar (routes answer with the sidecar's own responses); and the native window around a
pre-server page (the page alone at that window's content size, its title bar taken as 28 px). The macOS
window's first frame is native: tests/test_launcher_window_chrome.py pins its contract, and only a built
app shows its pixels. ``OUROBOROS_UI_EVIDENCE_DIR`` keeps a screenshot of each screen in each theme.
"""
from __future__ import annotations

import ast
import asyncio
import json
import os
import pathlib
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

pytestmark = [pytest.mark.ui_browser]

REPO = pathlib.Path(__file__).resolve().parents[1]
PASSWORD = "network-gate-test-password"
ROOMS = {"light": "rgb(255, 255, 255)", "dark": "rgb(15, 15, 17)"}
ENGINES = ["chromium", "webkit"]
PUBLIC_ORIGIN = "https://quiet-river-42.trycloudflare.com"
TELEGRAM_SDK = "https://telegram.org/js/telegram-web-app.js"

# Before any page script: the stored choice, or none, and storage readable or blocked.
STORAGE = """(([saved, blocked]) => {
    try { if (saved === null) localStorage.removeItem('ouroboros.theme'); else localStorage.setItem('ouroboros.theme', saved); }
    catch (_) { /* an opaque document has no storage of its own */ }
    if (blocked) Object.defineProperty(window, 'localStorage', { configurable: true,
        get() { throw new DOMException('The operation is insecure.', 'SecurityError'); } });
    window.__cspViolations = [];
    addEventListener('securitypolicyviolation', (event) => __cspViolations.push(event.violatedDirective));
})(%s)"""


def _storage(saved, blocked) -> str:
    return STORAGE % json.dumps([saved, blocked])

# WCAG contrast of an element's text over what is painted under it (translucent layers composited).
CONTRAST = """(selector) => {
    const parse = (value) => { const m = /rgba?\\(([^)]+)\\)/.exec(value); if (!m) return null;
        const p = m[1].split(/[\\s,\\/]+/).filter(Boolean).map(Number); return [p[0], p[1], p[2], p.length > 3 ? p[3] : 1]; };
    const over = (top, under) => [0, 1, 2].map((i) => top[i] * top[3] + under[i] * (1 - top[3])).concat(1);
    const element = document.querySelector(selector);
    const layers = [];
    for (let node = element; node; node = node.parentElement) layers.push(parse(getComputedStyle(node).backgroundColor));
    let under = [255, 255, 255, 1];
    for (const layer of layers.reverse()) if (layer && layer[3] > 0) under = over(layer, under);
    const ink = over(parse(getComputedStyle(element).color), under);
    const lum = (c) => { const f = (v) => ((v /= 255) <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4);
        return 0.2126 * f(c[0]) + 0.7152 * f(c[1]) + 0.0722 * f(c[2]); };
    const [a, b] = [lum(ink), lum(under)].sort((x, y) => y - x);
    return Math.round(((a + 0.05) / (b + 0.05)) * 100) / 100;
}"""


def _shoot(page, name: str) -> None:
    """Keep what the engine painted, when the run asked for evidence."""
    root = os.environ.get("OUROBOROS_UI_EVIDENCE_DIR")
    if root:
        path = pathlib.Path(root) / "theme-before-interface" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(path))


def _painted(page) -> dict:
    return page.evaluate("""() => ({theme: document.documentElement.dataset.theme,
        room: getComputedStyle(document.body).backgroundColor,
        wide: document.documentElement.scrollWidth > innerWidth,
        tall: document.documentElement.scrollHeight > innerHeight,
        storage: (() => { try { localStorage.length; return 'readable'; } catch (error) { return error.name; } })()})""")


@pytest.fixture
def engines():
    if os.environ.get("OUROBOROS_RUN_UI_SMOKE") != "1":
        pytest.skip("set OUROBOROS_RUN_UI_SMOKE=1 to run browser UI smoke")
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        launched = {}
        try:
            yield lambda name: launched.setdefault(name, getattr(playwright, name).launch(headless=True))
        finally:
            for browser in launched.values():
                browser.close()


@pytest.fixture
def gate_url(monkeypatch):
    """The real gate over the builtin interface tree, answering a peer beyond loopback."""
    import uvicorn
    from starlette.applications import Starlette
    from starlette.responses import HTMLResponse
    from starlette.routing import Route

    from ouroboros import server_auth
    from ouroboros.server_web import ServedUI

    monkeypatch.setattr(server_auth, "get_configured_network_password", lambda: PASSWORD)
    monkeypatch.setattr(server_auth, "_auth_secret_cache", b"\x01" * 32)  # no key file in a data root

    async def unlocked(_request):
        return HTMLResponse("<!doctype html><title>Unlocked</title><p id='unlocked'>Unlocked</p>")

    gate = server_auth.NetworkAuthGate(Starlette(routes=[Route("/", unlocked)]), served_ui=ServedUI(REPO))

    async def phone(scope, receive, send):
        if scope["type"] in {"http", "websocket"}:
            scope = {**scope, "client": ("198.51.100.23", 50000)}
        await gate(scope, receive, send)

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    server = uvicorn.Server(uvicorn.Config(phone, log_level="warning", lifespan="off"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert server.started
    try:
        yield f"http://127.0.0.1:{sock.getsockname()[1]}"
    finally:
        server.should_exit = True
        thread.join(10)
        sock.close()
        assert not thread.is_alive()


LOGIN_CASES = [  # saved choice, the device's appearance, storage blocked -> painted
    ("light", "dark", False, "light"),
    ("dark", "light", False, "dark"),
    ("system", "light", False, "light"),
    ("system", "dark", False, "dark"),
    (None, "light", False, "light"),
    (None, "dark", False, "dark"),
    ("light", "dark", True, "dark"),
    (None, "light", True, "light"),
]


@pytest.mark.parametrize("engine", ENGINES)
def test_the_login_form_follows_the_saved_choice_else_the_device(engines, gate_url, engine):
    browser = engines(engine)
    for saved, device, blocked, expected in LOGIN_CASES:
        context = browser.new_context(viewport={"width": 390, "height": 844}, device_scale_factor=2,
                                      color_scheme=device)
        page = context.new_page()
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.add_init_script(_storage(saved, blocked))
        response = page.goto(gate_url + "/", wait_until="load")
        case = (saved, device, blocked)
        assert response.status == 401, case
        painted = _painted(page)
        assert painted["storage"] == ("SecurityError" if blocked else "readable"), case
        assert (painted["theme"], painted["room"]) == (expected, ROOMS[expected]), case
        assert not painted["wide"] and not painted["tall"], case
        assert page.get_attribute("#password", "autofocus") is not None, case  # headless WebKit ignores it
        for selector in (".gate h1", ".gate p", ".gate label", ".gate .btn-primary"):
            assert page.evaluate(CONTRAST, selector) >= 4.5, (case, selector)
        if saved == expected and not blocked and saved in ("light", "dark"):
            _shoot(page, f"login-{engine}-{expected}.png")
        assert errors == [], case
        context.close()


@pytest.mark.parametrize("engine", ENGINES)
def test_a_wrong_password_keeps_the_choice_and_the_right_one_unlocks(engines, gate_url, engine):
    browser = engines(engine)
    for saved, device in (("light", "dark"), ("dark", "light")):
        context = browser.new_context(viewport={"width": 390, "height": 844}, device_scale_factor=2,
                                      color_scheme=device)
        page = context.new_page()
        page.add_init_script(_storage(saved, False))
        page.goto(gate_url + "/", wait_until="load")
        page.get_by_label("Password", exact=True).fill("not the password")
        with page.expect_navigation():
            page.get_by_role("button", name="Unlock", exact=True).click()
        alert = page.get_by_role("alert")
        assert alert.inner_text() == "Invalid password."
        assert _painted(page)["theme"] == saved
        assert page.evaluate(CONTRAST, ".gate .gate-error") >= 4.5, saved
        error_ink = page.evaluate("getComputedStyle(document.documentElement).getPropertyValue('--status-error-fg').trim()")
        assert alert.evaluate("(node) => getComputedStyle(node).color") == page.evaluate(
            "(ink) => { const probe = document.createElement('i'); probe.style.color = ink; document.body.append(probe);"
            " const value = getComputedStyle(probe).color; probe.remove(); return value; }", error_ink)
        page.locator("#password").focus()
        assert page.evaluate("getComputedStyle(document.activeElement).boxShadow") != "none"  # the field's ring
        page.keyboard.press("Alt+Tab" if engine == "webkit" else "Tab")  # macOS WebKit tabs to buttons with Option
        assert page.evaluate("document.activeElement.textContent.trim()") == "Unlock"
        assert page.evaluate("getComputedStyle(document.activeElement).outlineStyle") == "solid"
        _shoot(page, f"login-error-{engine}-{saved}.png")
        page.get_by_label("Password", exact=True).fill(PASSWORD)
        with page.expect_navigation():
            page.get_by_role("button", name="Unlock", exact=True).click()
        assert page.locator("#unlocked").inner_text() == "Unlocked"
        context.close()


def _sidecar(**seams):
    """The skill's own sidecar module (loaded as its tests load it) and one gateway on the public origin."""
    from tests.test_telegram_miniapp_sidecar import BOT_TOKEN, OWNER_ID, sidecar_module

    gateway = sidecar_module.TelegramProxySidecar(BOT_TOKEN, OWNER_ID, 8765, seams=sidecar_module.SidecarSeams(**seams))
    gateway.set_public_url(PUBLIC_ORIGIN + "/")
    return sidecar_module, gateway


def _sidecar_responses() -> dict:
    """The sidecar's own answers to an unauthenticated Telegram client: the bootstrap page and its script."""
    import httpx

    module, gateway = _sidecar()

    async def fetch():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=gateway), base_url=PUBLIC_ORIGIN) as client:
            return {path: await client.get(path) for path in ("/", module.BOOTSTRAP_JS_PATH)}

    with ThreadPoolExecutor(1) as pool:  # Playwright's sync API already runs a loop on this thread
        answers = pool.submit(asyncio.run, fetch()).result()
    return {path: (answer.status_code, {key: value for key, value in answer.headers.items()
                                        if key not in ("content-length", "content-encoding")}, answer.content)
            for path, answer in answers.items()}


def _telegram_network(answers: dict, stub: str):
    """The network a Telegram client sees: Telegram's SDK (the stub) and the sidecar's own answers."""
    def answer(route):
        url = route.request.url
        if url == TELEGRAM_SDK:
            return route.fulfill(status=200, content_type="application/javascript", body=stub)
        path = url[len(PUBLIC_ORIGIN):] if url.startswith(PUBLIC_ORIGIN) else None
        if path in answers:
            status, headers, body = answers[path]
            return route.fulfill(status=status, headers=headers, body=body)
        if path == "/__telegram/auth":
            return None  # Telegram's launch data is checked elsewhere: the page stays connecting
        return route.abort()

    return answer


TELEGRAM_CASES = [  # saved choice, Telegram's scheme, the device's, storage blocked -> painted
    ("light", "dark", "dark", False, "light"),
    ("dark", "light", "light", False, "dark"),
    ("system", "light", "dark", False, "light"),  # System is Telegram's here, not the device's
    (None, "dark", "light", False, "dark"),
    (None, None, "light", False, "light"),  # no scheme from Telegram: the device's
    ("light", "dark", "light", True, "dark"),  # storage blocked: Telegram's
]


@pytest.mark.parametrize("engine", ENGINES)
def test_the_telegram_connecting_page_follows_the_saved_choice_else_telegram(engines, engine):
    answers = _sidecar_responses()
    browser = engines(engine)
    for saved, scheme, device, blocked, expected in TELEGRAM_CASES:
        for launch in ("connecting", "outside"):
            context = browser.new_context(viewport={"width": 390, "height": 844}, device_scale_factor=2,
                                          color_scheme=device)
            page = context.new_page()
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            stub = ("window.Telegram = {WebApp: {colorScheme: %s, initData: %s, ready() { window.__ready = true; },"
                    " expand() {}}};") % ("undefined" if scheme is None else repr(scheme),
                                          repr("query_id=x" if launch == "connecting" else ""))

            page.route("**/*", _telegram_network(answers, stub))
            page.add_init_script(_storage(saved, blocked))
            page.goto(PUBLIC_ORIGIN + "/", wait_until="domcontentloaded")
            status = page.locator("#status")
            words = "Connecting to Ouroboros…" if launch == "connecting" else "Open this app from its Telegram bot menu."
            page.wait_for_function("(words) => document.getElementById('status').textContent === words", arg=words)
            case = (saved, scheme, device, blocked, launch)
            painted = _painted(page)
            assert (painted["theme"], painted["room"]) == (expected, ROOMS[expected]), case
            assert status.get_attribute("role") == "status"
            assert page.evaluate(CONTRAST, "#status") >= 4.5, case
            assert page.evaluate("window.__cspViolations") == [], case
            assert page.evaluate("Boolean(window.__ready)") is (launch == "connecting"), case
            if saved in ("light", "dark") and not blocked:
                _shoot(page, f"telegram-{launch}-{engine}-{expected}.png")
            assert errors == [], case
            context.close()


class _PublicOrigin:
    """The network of a Telegram client after a genuine launch: Telegram's SDK (the stub), and the public origin
    answered by the sidecar's own ASGI app, which validates the launch data, keeps the session and proxies the
    core's builtin page. The sidecar runs on a loop of its own: Playwright's sync API owns this thread's.
    Bounded at the handoff: the interface's own scripts (all but theme.js) are not served, so the page the
    sidecar hands off to paints from theme.js and the stylesheets alone, and nothing boots behind it."""

    def __init__(self, gateway, stub: str):
        import httpx

        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=gateway), base_url=PUBLIC_ORIGIN)
        self.stub, self.seen = stub, []

    def __call__(self, route):
        request = route.request
        if request.url == TELEGRAM_SDK:
            return route.fulfill(status=200, content_type="application/javascript", body=self.stub)
        if not request.url.startswith(PUBLIC_ORIGIN + "/"):
            return route.abort()
        target = request.url[len(PUBLIC_ORIGIN):]
        path = target.split("?", 1)[0]
        if path.startswith("/static/") and path.endswith(".js") and path != "/static/theme.js":
            return route.abort()
        headers = {key: value for key, value in request.all_headers().items() if key != "host" and not key.startswith(":")}
        answer = asyncio.run_coroutine_threadsafe(self.client.request(
            request.method, target, headers=headers, content=request.post_data_buffer), self.loop).result(15)
        self.seen.append((request.method, path, answer.status_code))
        return route.fulfill(status=answer.status_code, body=answer.content, headers={
            key: value for key, value in answer.headers.multi_items()
            if key not in ("content-length", "content-encoding", "transfer-encoding")})

    def close(self):
        asyncio.run_coroutine_threadsafe(self.client.aclose(), self.loop).result(5)
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(5)


# Logs each document's painted palette: the bootstrap once parsed (it may leave before its load), the page it
# hands off to once its stylesheets have loaded.
PAINTED = """(() => {
    const core = () => !!document.querySelector('meta[name="ouroboros-ui-owner"]');
    const report = () => console.log('painted ' + JSON.stringify({core: core(),
        host: document.documentElement.dataset.ouroborosHost || '', theme: document.documentElement.dataset.theme,
        room: getComputedStyle(document.body).backgroundColor}));
    addEventListener('DOMContentLoaded', () => { if (!core()) report(); });
    addEventListener('load', () => { if (core()) report(); });
})();"""


@pytest.mark.parametrize("engine", ENGINES)
def test_a_genuine_telegram_launch_hands_off_in_the_same_palette(engines, engine):
    """Launch data signed with the bot token passes the sidecar's own check; its session cookie lets the page go
    to ``/``, which the sidecar proxies to the core's builtin page (stamped as the Telegram host). The connecting
    page and the page it hands off to paint the same palette: the saved choice, or with System and no saved choice
    the appearance Telegram and the device agree on. (The interface reads Telegram's scheme only once its Telegram
    adapter binds, after its scripts load: a device that disagrees with Telegram is that adapter's case.)"""
    import httpx
    from starlette.applications import Starlette
    from starlette.routing import Mount, Route

    from ouroboros.server_web import ServedUI
    from tests.test_telegram_miniapp_sidecar import signed_init_data

    served = ServedUI(REPO)
    core = Starlette(routes=[Route("/", served.index_page), Mount("/static", app=served.static)])
    browser = engines(engine)
    for saved, scheme in (("light", "dark"), ("dark", "light"), (None, "light"), (None, "dark")):
        _module, gateway = _sidecar(http_client=httpx.AsyncClient(transport=httpx.ASGITransport(app=core)))
        stub = ("window.Telegram = {WebApp: {colorScheme: %s, initData: %s, ready() {}, expand() {}}};"
                % (json.dumps(scheme), json.dumps(signed_init_data(auth_date=int(time.time())))))
        network = _PublicOrigin(gateway, stub)
        context = browser.new_context(viewport={"width": 390, "height": 844}, device_scale_factor=2,
                                      color_scheme=scheme)  # with System the device agrees with Telegram here
        page = context.new_page()
        painted, errors = [], []
        page.on("console", lambda message: painted.append(json.loads(message.text[8:]))
                if message.text.startswith("painted ") else None)
        page.on("pageerror", lambda error: errors.append(str(error)))
        try:
            page.route("**/*", network)
            page.add_init_script(_storage(saved, False))
            page.add_init_script(PAINTED)
            page.goto(PUBLIC_ORIGIN + "/", wait_until="domcontentloaded")
            deadline = time.monotonic() + 15
            while len(painted) < 2 and time.monotonic() < deadline:
                page.wait_for_timeout(50)  # console events arrive after their document's own
            expected = saved or scheme
            case = (saved, scheme)
            assert [(entry["core"], entry["theme"], entry["room"]) for entry in painted] == [
                (False, expected, ROOMS[expected]), (True, expected, ROOMS[expected])], (case, painted)
            assert painted[1]["host"] == "telegram", case
            steps = [(method, path, status) for method, path, status in network.seen if not path.startswith("/static/")]
            assert steps == [("GET", "/", 200), ("GET", "/__telegram/bootstrap.js", 200),
                             ("POST", "/__telegram/auth", 204), ("GET", "/__telegram/session", 204),
                             ("GET", "/", 200)], (case, network.seen)
            assert ("GET", "/static/theme.js", 200) in network.seen, case
            assert errors == [], case
            if saved:
                _shoot(page, f"telegram-handoff-{engine}-{expected}.png")
        finally:
            context.close()
            network.close()


def _launcher_pages() -> dict:
    """Each pre-server window's own page, as launcher.py builds it on macOS (its ``html=`` argument, evaluated)."""
    import launcher
    from ouroboros.platform_layer import git_install_hint

    pages = {}
    for node in ast.walk(ast.parse((REPO / "launcher.py").read_text(encoding="utf-8"))):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "create_window":
            html = next((keyword.value for keyword in node.keywords if keyword.arg == "html"), None)
            if html is None:
                continue
            size = {keyword.arg: keyword.value.value for keyword in node.keywords if keyword.arg in ("width", "height")}
            source = compile(ast.Expression(html), "launcher.py", "eval")
            page = eval(source, {"_window_page": launcher._window_page, "IS_MACOS": True, "_hint": git_install_hint()})  # noqa: S307
            pages[node.args[0].value] = (page, size["width"], size["height"] - 28)  # the ordinary title bar
    assert sorted(pages) == ["Ouroboros", "Ouroboros — Setup Required", "Ouroboros — Startup Failed"]
    return pages


@pytest.mark.parametrize("engine", ENGINES)
def test_windows_before_the_server_follow_the_device(engines, engine):
    pages = _launcher_pages()
    browser = engines(engine)
    for title, (html, width, height) in pages.items():
        for device in ("light", "dark"):
            context = browser.new_context(viewport={"width": width, "height": height}, device_scale_factor=2,
                                          color_scheme=device)
            page = context.new_page()
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.set_content(html, wait_until="load")
            painted = _painted(page)
            assert (painted["theme"], painted["room"]) == (device, ROOMS[device]), (title, device)
            assert not painted["wide"] and not painted["tall"], (title, device)
            for selector in ("h2", ".window p"):
                assert page.evaluate(CONTRAST, selector) >= 4.5, (title, device, selector)
            if page.locator("#install-btn").count():
                assert page.evaluate(CONTRAST, "#install-btn") >= 4.5, (title, device)
                page.evaluate("document.getElementById('status').textContent = 'Installing... A system dialog may appear.'")
                assert page.evaluate(CONTRAST, "#status") >= 4.5, (title, device)
            _shoot(page, f"window-{title.split('—')[-1].strip().lower().replace(' ', '-')}-{engine}-{device}.png")
            assert errors == [], (title, device)
            context.close()
