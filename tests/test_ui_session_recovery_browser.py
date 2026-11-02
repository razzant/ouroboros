"""A separately hosted interface signs in, expires and survives a core outage.

The core is the real ``server.py`` with its network password. The interface is the
unchanged SPA served by an independent static host, and both sit behind one browser
origin through a reverse proxy that routes ``/`` and ``/static/*`` to the interface
host and everything else to the core. The proxy forwards a LAN client address the
way a deployed proxy does, so the core's gate sees a non-loopback browser; the
browser itself only ever talks to loopback.

Proven here with Chromium and WebKit: an empty cookie jar opening ``/`` reaches the
gate's own sign-in, a wrong password stays there, the right one returns to the
interface with HTTP and WebSocket working; an expired session is refused at the
socket handshake as an ordinary close (never 4401) and hands over to sign-in again
without resending what the page had queued; a core that is down when the page loads
is waited for and unblocks the page without a click. Page ownership (`served_ui`
and the core marker) is executable in web/tests/ws_recovery.test.js.
"""
from __future__ import annotations

import asyncio
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import threading
import time
from urllib.parse import urlsplit
import urllib.request

import pytest

from tests.test_ui_smoke_playwright import direct_server_with_data as direct_server_with_data
from tests.ui_console_errors import PageErrors

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]

PASSWORD = "ui-smoke-password"  # the real-server fixture's network password
LAN_CLIENT = "198.51.100.23"
_HOP_HEADERS = {"connection", "keep-alive", "x-forwarded-for", "x-forwarded-proto", "x-real-ip", "forwarded"}


class _InterfaceHost(ThreadingHTTPServer):
    """An independent static host for the SPA bytes: no core code, no state."""

    def __init__(self, web: Path):
        self.web, self.paths = web, []

        class Handler(SimpleHTTPRequestHandler):
            def translate_path(handler, path):  # noqa: N805 - the handler instance
                route = urlsplit(path).path
                handler.server.paths.append(route)
                if route == "/":
                    return str(web / "index.html")
                if route.startswith("/static/"):
                    return str(web / route[len("/static/"):])
                return str(web / "__not_served__")

            def end_headers(handler):  # noqa: N805
                handler.send_header("Cache-Control", "no-store")
                super().end_headers()

            def log_message(handler, *args):  # noqa: N805
                return

        super().__init__(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.serve_forever, name="interface-host", daemon=True)
        self.thread.start()

    def close(self):
        self.shutdown()
        self.server_close()
        self.thread.join(10)


class _SameOriginProxy:
    """One origin over two hosts, forwarding the client the way a deployed proxy does."""

    def __init__(self, interface_port: int, core_port: int):
        self.interface_port, self.core_port = interface_port, core_port
        self.records, self._sockets = [], set()
        self.loop = asyncio.new_event_loop()
        started = threading.Event()
        self.thread = threading.Thread(target=self._run, args=(started,), name="same-origin-proxy", daemon=True)
        self.thread.start()
        assert started.wait(10)

    def _run(self, started):
        asyncio.set_event_loop(self.loop)
        self.server = self.loop.run_until_complete(asyncio.start_server(self._handle, "127.0.0.1", 0))
        self.port = self.server.sockets[0].getsockname()[1]
        started.set()
        self.loop.run_forever()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    async def _handle(self, reader, writer):
        try:
            head = await reader.readuntil(b"\r\n\r\n")
        except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, ConnectionError):
            writer.close()
            return
        request_line, *lines = head.decode("latin-1").split("\r\n")
        path = urlsplit(request_line.split(" ")[1]).path
        upstream = "interface" if path == "/" or path.startswith("/static/") else "core"
        upgrade = any(line.lower().startswith("upgrade:") and "websocket" in line.lower() for line in lines)
        kept = [line for line in lines if line and (line.split(":", 1)[0].strip().lower() not in _HOP_HEADERS
                                                     or (upgrade and line.lower().startswith("connection:")))]
        # Overwrite, never append: a client-sent X-Forwarded-For must not reach the core.
        kept += [f"X-Forwarded-For: {LAN_CLIENT}", "X-Forwarded-Proto: http"]
        if not upgrade:
            kept.append("Connection: close")
        record = {"path": path, "upstream": upstream, "websocket": upgrade, "status": None}
        self.records.append(record)
        try:
            up_reader, up_writer = await asyncio.open_connection(
                "127.0.0.1", self.interface_port if upstream == "interface" else self.core_port)
        except OSError:
            record["status"] = 502
            writer.write(b"HTTP/1.1 502 Bad Gateway\r\nContent-Type: text/plain\r\n"
                         b"Content-Length: 11\r\nConnection: close\r\n\r\nBad Gateway")
            await writer.drain()
            writer.close()
            return
        up_writer.write(("\r\n".join([request_line, *kept]) + "\r\n\r\n").encode("latin-1"))
        pair = (writer, up_writer)
        if upgrade:
            self._sockets.add(pair)

        async def pump(source, sink, observe=False):
            try:
                while chunk := await source.read(65536):
                    if observe and record["status"] is None:
                        record["status"] = int(chunk.split(b" ", 2)[1])
                    sink.write(chunk)
                    await sink.drain()
            except (ConnectionError, asyncio.CancelledError, ValueError, IndexError):
                pass
            finally:
                sink.close()

        try:
            await asyncio.gather(pump(reader, up_writer), pump(up_reader, writer, observe=True))
        finally:
            self._sockets.discard(pair)

    def drop_websockets(self) -> int:
        """Cut every upgraded connection, as a network blip does."""
        pairs = list(self._sockets)

        def cut():
            for client, upstream in pairs:
                client.close()
                upstream.close()
        self.loop.call_soon_threadsafe(cut)
        return len(pairs)

    def close(self):
        async def stop():
            self.server.close()
            for client, upstream in list(self._sockets):
                client.close()
                upstream.close()
        asyncio.run_coroutine_threadsafe(stop(), self.loop).result(10)
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(10)


# Close codes and every main-frame address this page lives through.
_OBSERVE = """
window.__socketCloses = JSON.parse(sessionStorage.getItem('__socketCloses') || '[]');
const NativeSocket = window.WebSocket;
window.WebSocket = class extends NativeSocket {
    constructor(...args) {
        super(...args);
        this.addEventListener('close', (event) => {
            window.__socketCloses.push({ code: event.code, clean: event.wasClean });
            sessionStorage.setItem('__socketCloses', JSON.stringify(window.__socketCloses));
        });
    }
};
"""


@pytest.fixture
def separate_ui(direct_server_with_data):
    core = direct_server_with_data
    host = _InterfaceHost(core["repo_dir"] / "web")
    proxy = _SameOriginProxy(host.server_address[1], int(urlsplit(core["url"]).port))
    try:
        yield {**core, "host": host, "proxy": proxy}
    finally:
        proxy.close()
        host.close()


def _evidence(core) -> Path:
    root = Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", core["data_dir"].parent)) / "session-recovery"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _connected(page) -> bool:
    return bool(page.evaluate("() => Boolean(window.__ouroWs?.isConnected?.())"))


def _sign_in(page, password: str):
    page.get_by_label("Password", exact=True).fill(password)
    page.get_by_role("button", name="Unlock", exact=True).click()


def _expired_session(data_dir: Path, *, age_sec: int) -> str:
    """A session the core itself signed, `age_sec` old (the production signer and key)."""
    from ouroboros import server_auth

    key = bytes.fromhex((data_dir / "state" / "auth_secret.key").read_text(encoding="utf-8").strip())
    previous = server_auth._auth_secret_cache
    server_auth._auth_secret_cache = key
    try:
        return server_auth._session_value(PASSWORD, int(time.time()) - age_sec)
    finally:
        server_auth._auth_secret_cache = previous


def _launch(playwright, browser_name):
    return getattr(playwright, browser_name).launch(headless=True)


def _at_sign_in(next_path: str | None = None):
    """A predicate for the gate's sign-in address, optionally with its exact return."""
    def matches(address: str) -> bool:
        parts = urlsplit(address)
        return parts.path == "/auth/login" and (next_path is None or parts.query == f"next={next_path}")
    return matches


def _chat_frame(text: str) -> str:
    # The composer's own frame shape (chat.js), sent through the page's real socket manager.
    return ("() => window.__ouroWs.send({type: 'chat', content: %s, sender_session_id: 'session-recovery-test'})"
            ".status" % json.dumps(text))


def _history_text(page) -> str:
    return json.dumps(page.evaluate(
        "async () => (await (await fetch('/api/chat/history', {cache: 'no-store'})).json())"))


@pytest.mark.parametrize("browser_name", ["chromium", "webkit"])
def test_separate_interface_signs_in_returns_and_recovers_an_expired_session(separate_ui, browser_name):
    pytest.importorskip("playwright.sync_api")
    from playwright.sync_api import sync_playwright

    proxy, host, url = separate_ui["proxy"], separate_ui["host"], separate_ui["proxy"].url
    evidence = _evidence(separate_ui)
    with sync_playwright() as playwright:
        browser = _launch(playwright, browser_name)
        try:
            context = browser.new_context(viewport={"width": 1200, "height": 820})
            context.add_init_script(_OBSERVE)
            page = context.new_page()
            errors, addresses = PageErrors(page, browser_name), []
            page.on("framenavigated", lambda frame: frame == page.main_frame and addresses.append(frame.url))

            # An empty cookie jar opens the interface host's `/`: the page itself is
            # served without the gate (its own bytes, unmarked), and its readiness
            # read is what is refused.
            with urllib.request.urlopen(url + "/", timeout=10) as anonymous:  # noqa: S310 - loopback test proxy
                assert anonymous.status == 200
                assert b"ouroboros-ui-owner" not in anonymous.read()
            assert context.cookies() == []
            first = page.goto(url + "/")
            assert first.status == 200
            page.wait_for_url(_at_sign_in("%2F"), timeout=20_000)
            assert any(r["path"] == "/api/onboarding" and r["status"] == 401 for r in proxy.records)
            page.screenshot(path=str(evidence / f"{browser_name}-1-sign-in.png"))

            _sign_in(page, "not-the-password")
            page.get_by_text("Invalid password.").wait_for()
            assert "/auth/login" in page.url
            assert not any(c["name"] == "ouroboros_auth" for c in context.cookies())
            page.screenshot(path=str(evidence / f"{browser_name}-2-wrong-password.png"))

            _sign_in(page, PASSWORD)
            page.wait_for_url(url + "/", timeout=20_000)
            page.wait_for_function("() => window.__ouroWs?.isConnected?.() === true", timeout=30_000)
            page.locator("#chat-input").wait_for(state="visible")
            assert page.locator("#onboarding-overlay").count() == 0
            cookie = next(c for c in context.cookies() if c["name"] == "ouroboros_auth")
            assert cookie["httpOnly"] is True
            assert page.evaluate("async () => (await fetch('/api/state', {cache: 'no-store'})).status") == 200
            page.screenshot(path=str(evidence / f"{browser_name}-3-returned.png"))

            # The core never answered `/`, and the interface host never saw an API path.
            assert all(r["upstream"] == "interface" for r in proxy.records if r["path"] == "/")
            assert not [p for p in host.paths if p.startswith(("/api/", "/ws", "/auth/"))]

            # The composer's frame shape is accepted while signed in (control for
            # the queued-message negative below).
            control = f"delivered-while-signed-in-{browser_name}-{time.time_ns()}"
            assert page.evaluate(_chat_frame(control)) == "sent"
            deadline = time.monotonic() + 30
            while control not in _history_text(page):
                assert time.monotonic() < deadline, "the control message never reached history"
                time.sleep(0.5)

            # A signed-but-expired session (the core's own signer, 31 days old) is
            # refused; the same signer's fresh session is accepted (positive control).
            data_dir = separate_ui["data_dir"]
            api = playwright.request.new_context()
            try:
                fresh = api.get(url + "/api/state", headers={
                    "Cookie": f"ouroboros_auth={_expired_session(data_dir, age_sec=60)}"})
                assert fresh.status == 200
                stale = api.get(url + "/api/state", headers={
                    "Cookie": f"ouroboros_auth={_expired_session(data_dir, age_sec=31 * 24 * 3600)}"})
                assert stale.status == 401
            finally:
                api.dispose()
            context.add_cookies([{"name": "ouroboros_auth", "value": _expired_session(data_dir, age_sec=31 * 24 * 3600),
                                  "url": url, "httpOnly": True, "sameSite": "Lax"}])
            assert context.request.get(url + "/api/state").status == 401

            closes_before = len(page.evaluate("() => window.__socketCloses"))
            assert proxy.drop_websockets() >= 1
            page.wait_for_function("() => window.__ouroWs?.isConnected?.() === false")
            # Sent while the socket is down: the page queues it, and nothing may
            # resend it after signing in again.
            unsent = f"queued-before-sign-in-{browser_name}-{time.time_ns()}"
            assert page.evaluate(_chat_frame(unsent)) == "queued"
            page.wait_for_url(_at_sign_in("%2F"), timeout=45_000)
            closes = page.evaluate("() => window.__socketCloses")[closes_before:]
            refused = [r for r in proxy.records if r["websocket"] and r["status"] == 403]
            assert refused, "the gate refuses the expired handshake at HTTP level"
            assert closes and all(c["code"] == 1006 and c["clean"] is False for c in closes), closes
            assert not any(c["code"] == 4401 for c in closes)
            assert any(r["path"] == "/api/state" and r["status"] == 401 for r in proxy.records)
            page.screenshot(path=str(evidence / f"{browser_name}-4-expired.png"))

            _sign_in(page, PASSWORD)
            page.wait_for_url(url + "/", timeout=20_000)
            page.wait_for_function("() => window.__ouroWs?.isConnected?.() === true", timeout=30_000)
            time.sleep(3)
            history = _history_text(page)
            assert control in history
            assert unsent not in history, "a queued message must not be resent after signing in"
            (evidence / f"{browser_name}-sign-in.json").write_text(json.dumps({
                "addresses": addresses, "proxy": proxy.records, "socket_closes": closes,
                "interface_host_paths": host.paths, "page_errors": errors}, indent=2), encoding="utf-8")
            # The page leaving for sign-in cancels fetches (WebKit's report of them is set aside).
            assert errors.unexpected() == [], errors
        finally:
            browser.close()


@pytest.mark.parametrize("browser_name", ["chromium", "webkit"])
def test_a_core_down_at_load_is_waited_for_and_unblocks_without_a_click(separate_ui, browser_name):
    pytest.importorskip("playwright.sync_api")
    from playwright.sync_api import sync_playwright

    proxy, url = separate_ui["proxy"], separate_ui["proxy"].url
    evidence = _evidence(separate_ui)
    with sync_playwright() as playwright:
        browser = _launch(playwright, browser_name)
        try:
            context = browser.new_context(viewport={"width": 1200, "height": 820})
            page = context.new_page()
            page.goto(url + "/")
            page.wait_for_url(_at_sign_in("%2F"), timeout=20_000)
            _sign_in(page, PASSWORD)
            page.wait_for_function("() => window.__ouroWs?.isConnected?.() === true", timeout=30_000)

            separate_ui["stop_server"]()
            clicks = []
            page.expose_function("__retryClicked", lambda: clicks.append(time.monotonic()))
            page.goto(url + "/")
            card = page.locator("[data-onboarding-failure]")
            card.wait_for(state="visible", timeout=20_000)
            assert "HTTP 502" in card.inner_text()
            page.evaluate("""() => document.querySelector('[data-onboarding-retry]')
                .addEventListener('click', () => window.__retryClicked())""")
            page.screenshot(path=str(evidence / f"{browser_name}-5-core-down.png"))

            separate_ui["start_server"]()
            page.locator("#onboarding-overlay").wait_for(state="detached", timeout=60_000)
            page.wait_for_function("() => window.__ouroWs?.isConnected?.() === true", timeout=60_000)
            assert clicks == [], "the core's return unblocked the page by itself"
            assert any(r["path"] == "/api/onboarding" and r["status"] == 502 for r in proxy.records)
            assert any(r["path"] == "/api/onboarding" and r["status"] == 204 for r in proxy.records)
            page.screenshot(path=str(evidence / f"{browser_name}-6-core-back.png"))
        finally:
            browser.close()
