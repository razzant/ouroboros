"""Tests for the minimal network password gate.

The gate tests pin the GATE, so they take the configured password through the
module's resolver seam (``get_configured_network_password``) instead of the
ambient environment: one windows-latest xdist worker saw a password with none
configured and then no password right after one was set — ambient os.environ
pollution from an earlier module on the same ``--dist loadscope`` worker (the
conftest snapshot restores os.environ between tests; a daemon thread applying a
settings dict to the environment does not wait for it). Exactly ONE test reads
the real resolution order, and it asserts the pre-state it relies on by name so a
polluted worker fails there, not as a downstream 200/404.
"""

import contextlib
import os
import pathlib
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route, WebSocketRoute
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import ouroboros.server_auth as server_auth


async def _ok(_: Request) -> JSONResponse:
    return JSONResponse({"ok": True})


def _make_client(monkeypatch, password: str = "secret") -> TestClient:
    monkeypatch.setattr(server_auth, "get_configured_network_password", lambda: password)
    app = server_auth.NetworkAuthGate(Starlette(routes=[
        Route("/", endpoint=_ok),
        Route("/api/health", endpoint=_ok),
        Route("/api/secret", endpoint=_ok),
    ]))
    return TestClient(app)


def test_configured_password_resolution_env_over_settings_then_empty():
    """The ONE resolution pin, on the PURE resolver: env wins over settings, a blank env
    falls through to settings, a missing/blank/unreadable settings value is the empty
    password. It reads neither os.environ nor a module attribute, so neither polluter
    class (a leaked daemon re-applying settings to the environment; a started-and-never-
    stopped patch of the module wrapper on the same xdist worker — the macos-latest
    rc.11 red: the wrapper answered '' with settings patched to a password) can reach it."""
    key = server_auth.NETWORK_PASSWORD_KEY
    resolve = server_auth.resolve_network_password
    assert resolve(None, dict) == "" and resolve("", lambda: {}) == ""
    assert resolve(None, lambda: {key: " from-settings "}) == "from-settings"
    assert resolve(" from-env ", lambda: {key: " from-settings "}) == "from-env"
    assert resolve("   ", lambda: {key: " from-settings "}) == "from-settings"
    assert resolve(None, lambda: {key: "  "}) == ""
    assert resolve("", lambda: {"other": "x"}) == ""

    def _unreadable():
        raise RuntimeError("settings unreadable")

    assert resolve(None, _unreadable) == ""


def test_wrapper_reads_the_environment_then_the_settings_loader(monkeypatch):
    """The wrapper hands the pure resolver the live environment value and the module's
    settings loader — pinned through the resolver seam, immune to a leaked patch."""
    key = server_auth.NETWORK_PASSWORD_KEY
    seen = []
    monkeypatch.setattr(server_auth, "resolve_network_password", lambda env, loader: seen.append((env, loader)) or "r")
    monkeypatch.setenv(key, "env-value")
    assert server_auth.get_configured_network_password() == "r"
    assert seen == [("env-value", server_auth.load_settings)]


def test_validate_network_auth_configuration_allows_open_bind_without_password(monkeypatch):
    monkeypatch.setattr(server_auth, "get_configured_network_password", lambda: "")

    assert server_auth.validate_network_auth_configuration("127.0.0.1") is None
    assert server_auth.validate_network_auth_configuration("0.0.0.0") is None


def test_get_network_auth_startup_warning_warns_but_allows_open_bind(monkeypatch):
    monkeypatch.setattr(server_auth, "get_configured_network_password", lambda: "")

    assert server_auth.get_network_auth_startup_warning("127.0.0.1") is None
    warning = server_auth.get_network_auth_startup_warning("0.0.0.0")
    assert warning is not None
    assert "without OUROBOROS_NETWORK_PASSWORD" in warning

    monkeypatch.setattr(server_auth, "get_configured_network_password", lambda: "secret")
    assert server_auth.get_network_auth_startup_warning("0.0.0.0") is None


def test_network_auth_gate_is_open_without_a_configured_password(monkeypatch):
    with _make_client(monkeypatch, password="") as client:
        assert client.get("/api/secret").status_code == 200


def test_network_auth_gate_blocks_non_local_requests(monkeypatch):
    with _make_client(monkeypatch) as client:
        html_resp = client.get("/", follow_redirects=False)
        assert html_resp.status_code == 401
        assert "Enter the network password" in html_resp.text

        api_resp = client.get("/api/secret")
        assert api_resp.status_code == 401
        assert api_resp.json()["error"] == "Authentication required."

        health_resp = client.get("/api/health")
        assert health_resp.status_code == 200


def test_network_auth_gate_accepts_header_and_login_cookie(monkeypatch):
    with _make_client(monkeypatch) as client:
        header_resp = client.get("/", headers={"x-ouroboros-password": "secret"})
        assert header_resp.status_code == 200
        assert header_resp.json() == {"ok": True}

    with _make_client(monkeypatch) as client:
        login_resp = client.post(
            "/auth/login",
            json={"password": "secret", "next": "/"},
            follow_redirects=False,
        )
        assert login_resp.status_code == 200

        cookie_resp = client.get("/")
        assert cookie_resp.status_code == 200
        assert cookie_resp.json() == {"ok": True}


def test_login_next_url_is_escaped(monkeypatch):
    with _make_client(monkeypatch) as client:
        resp = client.get('/auth/login?next=/"><script>alert(1)</script>', follow_redirects=False)
        assert resp.status_code == 200
        # The form inlines its own appearance script; the requested markup appears nowhere.
        assert "alert(1)" not in resp.text and '"><script>' not in resp.text
        assert 'value="/"' in resp.text


WEB = pathlib.Path(__file__).resolve().parents[1] / "web"


def _inlined(page: str) -> tuple[str, str]:
    """The page's inlined appearance owner and token sheet, as the browser parses them."""
    script = re.search(r"<script>(.*?)</script>", page, re.S)
    style = re.search(r"<style>(.*?)</style>", page, re.S)
    return (script.group(1) if script else "", style.group(1) if style else "")


def test_builtin_form_wears_the_interface_appearance_before_any_asset(monkeypatch):
    """The form answers before the gate lets anything load, so it carries theme.js and ui.css itself:
    this browser's saved Light / Dark / System, else the device's (tests/test_theme_before_interface_browser.py
    renders it). Nothing behind the gate becomes public for it."""
    with _make_client(monkeypatch) as client:
        for page in (client.get("/", follow_redirects=False), client.get("/auth/login?next=/api/secret")):
            script, style = _inlined(page.text)
            assert script == (WEB / "theme.js").read_text(encoding="utf-8")
            assert style == (WEB / "ui.css").read_text(encoding="utf-8")
            assert '<html lang="en" class="ouro-ui">' in page.text
            assert page.text.index("<script>") < page.text.index("<style>")  # the palette is chosen before any style
            assert 'class="ui-control"' in page.text and 'class="btn btn-primary"' in page.text
            assert page.headers["cache-control"] == "no-store"
        assert client.get("/static/theme.js").status_code == 401
        assert client.get("/static/ui.css").status_code == 401


def test_wrong_password_keeps_the_appearance_and_announces_the_error(monkeypatch):
    with _make_client(monkeypatch) as client:
        resp = client.post("/auth/login", data={"password": "wrong", "next": "/api/secret"}, follow_redirects=False)
        assert resp.status_code == 401
        assert '<p class="gate-error" role="alert">Invalid password.</p>' in resp.text
        assert _inlined(resp.text)[0] == (WEB / "theme.js").read_text(encoding="utf-8")
        assert 'value="/api/secret"' in resp.text


def test_builtin_form_under_a_selected_interface_wears_only_builtin_files(monkeypatch, tmp_path):
    """A selected directory without its own login page gets the builtin form, never mixed with that
    directory's files (docs/development/13-gateway-boundary-pattern.md)."""
    from ouroboros.server_web import ServedUI

    selected = tmp_path / "selected"
    selected.mkdir()
    (selected / "index.html").write_text("<!doctype html><title>selected</title>", encoding="utf-8")
    (selected / "theme.js").write_text("/* SELECTED-THEME */", encoding="utf-8")
    (selected / "ui.css").write_text("/* SELECTED-TOKENS */", encoding="utf-8")
    served = ServedUI(pathlib.Path(__file__).resolve().parents[1])
    app = Starlette(routes=[Route("/", endpoint=_ok)])
    monkeypatch.setattr("ouroboros.server_process.record_applied_restart_settings", lambda settings: None)
    assert served.apply(app, {"OUROBOROS_UI_DIR": str(selected)}).source == "directory"
    monkeypatch.setattr(server_auth, "get_configured_network_password", lambda: "secret")
    with TestClient(server_auth.NetworkAuthGate(app, served_ui=served)) as client:
        for page in (client.get("/", follow_redirects=False), client.get("/auth/login")):
            assert page.status_code in (200, 401)
            assert "SELECTED-" not in page.text
            assert _inlined(page.text) == ((WEB / "theme.js").read_text(encoding="utf-8"),
                                           (WEB / "ui.css").read_text(encoding="utf-8"))


def test_builtin_form_without_readable_assets_stays_a_usable_form(tmp_path, caplog):
    page = server_auth._login_page("/", "Invalid password.", web_dir=tmp_path / "missing")
    # No appearance owner is inlined; only the saved-conversation release below runs.
    assert re.findall(r"<script>(.*?)</script>", page, re.S) == [server_auth._END_OF_SESSION_SCRIPT]
    assert '<meta name="color-scheme" content="light dark">' in page
    assert 'name="password"' in page and "Unlock" in page and "Invalid password." in page
    assert "var(--bg-primary, Canvas)" in page  # the system's own colours, readable in either appearance
    assert "was not inlined" in caplog.text


_PREVIEW_SOURCE = (WEB / "modules" / "chat_preview.js").read_text(encoding="utf-8")
PREVIEW_KEY, PREVIEW_FAMILY = (re.search(rf"export const {name} = '([^']+)'", _PREVIEW_SOURCE).group(1)
                               for name in ("PREVIEW_KEY", "PREVIEW_FAMILY"))


def test_every_sign_out_and_ended_session_reaches_the_form_that_releases_the_saved_conversation(monkeypatch):
    """GET or POST /auth/logout and any page whose session ended land on the builtin form before the interface can run
    again. Its script removes this browser's saved conversation text (web/modules/chat_preview.js: the record
    PREVIEW_KEY, then every key of PREVIEW_FAMILY), nothing else the browser keeps, after the appearance owner
    (web/tests/chat_preview.test.js runs it against a store, tests/test_phone_chat_reopen_browser.py in real engines)."""
    theme = (WEB / "theme.js").read_text(encoding="utf-8")
    with _make_client(monkeypatch) as client:
        for response in (client.get("/auth/logout"), client.post("/auth/logout", data={"next": "/api/secret"}, headers={"accept": "text/html"}),
                         client.get("/auth/login?next=/"), client.get("/api/secret", headers={"accept": "text/html"})):
            assert response.status_code in (200, 401) and "Enter the network password" in response.text
            scripts = re.findall(r"<script>(.*?)</script>", response.text, re.S)
            assert scripts == [theme, server_auth._END_OF_SESSION_SCRIPT], response.url
    assert f'removeItem("{PREVIEW_KEY}")' in server_auth._END_OF_SESSION_SCRIPT
    assert f'indexOf("{PREVIEW_FAMILY}")===0' in server_auth._END_OF_SESSION_SCRIPT



def test_a_held_settings_read_leaves_the_loop_serving_and_admits_its_request_when_released(monkeypatch):
    """The gate reads the settings document for a scope on its own reader thread: while that read waits on the
    disk, an already-open socket keeps answering on the loop, and the request is admitted when the read returns."""
    monkeypatch.delenv(server_auth.NETWORK_PASSWORD_KEY, raising=False)
    entered, release = threading.Event(), threading.Event()
    reads: list[int] = []

    def settings():
        reads.append(threading.get_ident())
        if len(reads) > 1:  # the socket's own handshake read passes; the next scope's read is held
            entered.set()
            assert release.wait(10), "the test never released the held read"
        return {}

    monkeypatch.setattr(server_auth, "load_settings", settings)
    loop_threads: list[int] = []

    async def echo(websocket):
        await websocket.accept()
        loop_threads.append(threading.get_ident())
        with contextlib.suppress(WebSocketDisconnect):
            while True:
                await websocket.send_text(await websocket.receive_text())

    def round_trip(socket):
        socket.send_text("still here")
        return socket.receive_text()

    app = server_auth.NetworkAuthGate(Starlette(routes=[Route("/api/secret", endpoint=_ok),
                                                        WebSocketRoute("/ws", echo)]))
    with TestClient(app) as client, client.websocket_connect("/ws") as socket, \
            ThreadPoolExecutor(max_workers=2) as pool:
        held = pool.submit(client.get, "/api/secret")
        try:
            assert entered.wait(5), "the request never reached its settings read"
            assert pool.submit(round_trip, socket).result(timeout=5) == "still here", "the loop waited for the disk"
            assert not held.done()
        finally:
            release.set()
        assert held.result(timeout=10).status_code == 200
    assert loop_threads and loop_threads[0] not in reads, "every settings read ran off the ASGI loop"


def _echo_socket_app(*routes):
    """The gate around a socket that echoes, plus ``routes``; returns the app and the loop threads that served it."""
    loop_threads: list[int] = []

    async def echo(websocket):
        await websocket.accept()
        loop_threads.append(threading.get_ident())
        with contextlib.suppress(WebSocketDisconnect):
            while True:
                await websocket.send_text(await websocket.receive_text())

    return Starlette(routes=[*routes, WebSocketRoute("/ws", echo)]), loop_threads


def _round_trip(socket):
    socket.send_text("still here")
    return socket.receive_text()


def test_a_held_sign_in_page_read_leaves_the_loop_serving(monkeypatch):
    """Every sign-in page the gate renders (the form, the form a refused page gets, the form after a wrong password)
    reads theme.js and ui.css and names the appearance scope on the gate's reader: while such a read waits on the
    disk an open socket keeps answering, and each page arrives whole once the read returns."""
    from ouroboros import server_web, standalone_page
    from ouroboros.server_web import ServedUI

    monkeypatch.setenv(server_auth.NETWORK_PASSWORD_KEY, "secret")  # no settings read: only the page waits
    entered, release = threading.Event(), threading.Event()
    reads: list[int] = []
    inline, scope_of = standalone_page._inline, server_web.appearance_scope

    def held_inline(path, tag):
        reads.append(threading.get_ident())
        if path.name == "theme.js":
            entered.set()
            assert release.wait(10), "the test never released the held read"
        return inline(path, tag)

    monkeypatch.setattr(standalone_page, "_inline", held_inline)
    monkeypatch.setattr(server_web, "appearance_scope",
                        lambda root: reads.append(threading.get_ident()) or scope_of(root))
    app, loop_threads = _echo_socket_app()
    pages = (("get", "/auth/login?next=/", {}, 200), ("get", "/api/secret", {"headers": {"accept": "text/html"}}, 401),
             ("post", "/auth/login", {"data": {"password": "wrong", "next": "/"}}, 401))
    with TestClient(server_auth.NetworkAuthGate(app, served_ui=ServedUI(pathlib.Path(__file__).resolve().parents[1]))) \
            as client, client.websocket_connect("/ws", headers={"x-ouroboros-password": "secret"}) as socket, \
            ThreadPoolExecutor(max_workers=2) as pool:
        for method, path, kwargs, status in pages:
            entered.clear()
            release.clear()
            held = pool.submit(getattr(client, method), path, **kwargs)
            try:
                assert entered.wait(5), f"{path} never reached its asset read"
                assert pool.submit(_round_trip, socket).result(timeout=5) == "still here", "the loop waited"
                assert not held.done()
            finally:
                release.set()
            response = held.result(timeout=10)
            assert response.status_code == status and "Enter the network password" in response.text, path
    assert loop_threads and reads and loop_threads[0] not in reads, "every page read ran off the ASGI loop"


def test_a_selected_interfaces_sign_in_page_is_read_off_the_loop(monkeypatch, tmp_path):
    from ouroboros.server_web import ServedUI

    selected = tmp_path / "selected"
    (selected / "auth").mkdir(parents=True)
    (selected / "index.html").write_text("<!doctype html><title>selected</title>", encoding="utf-8")
    (selected / "auth" / "index.html").write_text("<!doctype html><title>SELECTED-SIGN-IN</title>", encoding="utf-8")
    threads: list[str] = []
    read = ServedUI.custom_page
    monkeypatch.setattr(ServedUI, "custom_page",
                        lambda self, page: threads.append(threading.current_thread().name) or read(self, page))
    served = ServedUI(pathlib.Path(__file__).resolve().parents[1])
    app = Starlette(routes=[Route("/", endpoint=_ok)])
    monkeypatch.setattr("ouroboros.server_process.record_applied_restart_settings", lambda settings: None)
    assert served.apply(app, {"OUROBOROS_UI_DIR": str(selected)}).source == "directory"
    monkeypatch.setattr(server_auth, "get_configured_network_password", lambda: "secret")
    with TestClient(server_auth.NetworkAuthGate(app, served_ui=served)) as client:
        assert "SELECTED-SIGN-IN" in client.get("/auth/login").text
        refused = client.get("/", headers={"accept": "text/html"}, follow_redirects=False)
        assert refused.status_code == 303 and refused.headers["location"] == "/auth/login?next=%2F"
    assert threads == ["auth-gate-reader", "auth-gate-reader"], threads


def test_a_password_in_the_environment_takes_no_settings_read_and_no_reader(monkeypatch):
    monkeypatch.setenv(server_auth.NETWORK_PASSWORD_KEY, " env-secret ")
    monkeypatch.setattr(server_auth, "load_settings", lambda: pytest.fail("the environment's password read settings"))
    monkeypatch.setattr(server_auth._READER, "submit", lambda read: pytest.fail("the environment's password waited"))
    app = server_auth.NetworkAuthGate(Starlette(routes=[Route("/api/secret", endpoint=_ok)]))
    with TestClient(app) as remote:
        assert remote.get("/api/secret").status_code == 401
        assert remote.get("/api/secret", headers={"x-ouroboros-password": "wrong"}).status_code == 401
        assert remote.get("/api/secret", headers={"x-ouroboros-password": "env-secret"}).status_code == 200
    with TestClient(app, client=("127.0.0.1", 50000)) as local:
        assert local.get("/api/secret").status_code == 200, "loopback stays ungated"


def test_the_environment_is_read_once_so_a_cleared_password_sends_no_settings_read_onto_the_loop(monkeypatch):
    """A settings save clears the environment's password between two reads of it: the value that chose the inline
    path is the value applied, so no settings read follows on the loop (and none waits on the reader)."""
    key = server_auth.NETWORK_PASSWORD_KEY
    reads: list[str] = []

    class _ClearedAfterOneRead(dict):
        def get(self, name, default=None):
            if name != key:
                return os.environ.get(name, default)
            reads.append(name)
            return " env-secret " if len(reads) == 1 else default

    monkeypatch.setattr(server_auth, "os", SimpleNamespace(environ=_ClearedAfterOneRead(), chmod=os.chmod))
    monkeypatch.setattr(server_auth, "load_settings", lambda: pytest.fail("settings read after the env decided"))
    monkeypatch.setattr(server_auth._READER, "submit", lambda read: pytest.fail("the environment's password waited"))
    app = server_auth.NetworkAuthGate(Starlette(routes=[Route("/api/secret", endpoint=_ok)]))
    with TestClient(app) as remote:
        assert remote.get("/api/secret", headers={"x-ouroboros-password": "env-secret"}).status_code == 200
    assert reads == [key], "one read of the environment per scope"


def test_a_malformed_expired_or_future_cookie_never_touches_the_session_key(monkeypatch, tmp_path):
    """Only a cookie whose shape and age could make it a session brings the key into memory (off the loop); one
    that cannot is refused without it, as before the key moved off the loop."""
    from ouroboros import config

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(server_auth, "_auth_secret_cache", None)
    calls: list[str] = []
    original = server_auth._auth_secret
    monkeypatch.setattr(server_auth, "_auth_secret",
                        lambda: calls.append(threading.current_thread().name) or original())
    now, name = int(time.time()), server_auth.AUTH_COOKIE_NAME
    with _make_client(monkeypatch) as client:
        for cookie in ("garbage", "abc.def", f"{now - server_auth._AUTH_SESSION_TTL_SEC - 60}.0", f"{now + 3600}.0"):
            assert client.get("/api/secret", headers={"cookie": f"{name}={cookie}"}).status_code == 401
        assert calls == [] and not (tmp_path / "state" / "auth_secret.key").exists(), calls
        assert client.get("/api/secret", headers={"cookie": f"{name}={now}.0"}).status_code == 401
    assert calls[:1] == ["auth-gate-reader"], "a cookie that could be a session is checked against the key"


def test_the_first_session_key_is_made_off_the_loop_and_only_for_a_cookie_or_a_login(monkeypatch, tmp_path):
    from ouroboros import config

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(server_auth, "_auth_secret_cache", None)
    calls: list[tuple[str, bool]] = []
    original = server_auth._auth_secret

    def recording():
        calls.append((threading.current_thread().name, server_auth._auth_secret_cache is None))
        return original()

    monkeypatch.setattr(server_auth, "_auth_secret", recording)
    with _make_client(monkeypatch) as client:
        assert client.get("/api/health").status_code == 200
        assert client.get("/api/secret").status_code == 401
        assert calls == [], "no cookie, no login: the key is never touched"
        assert client.post("/auth/login", json={"password": "secret", "next": "/"}).status_code == 200
        assert client.get("/api/secret").status_code == 200
    assert calls[0] == ("auth-gate-reader", True), calls
    assert not any(missing for _thread, missing in calls[1:]), "every later use found the key in memory"
    key = (tmp_path / "state" / "auth_secret.key").read_text(encoding="utf-8").strip()
    assert bytes.fromhex(key) == server_auth._auth_secret_cache


def test_concurrent_first_uses_read_or_create_one_session_key(monkeypatch, tmp_path):
    from ouroboros import config

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(server_auth, "_auth_secret_cache", None)
    minted = iter(f"{n:064x}" for n in range(1, 100))

    def slow_token(_nbytes):  # every first use overlaps the others inside the creation
        time.sleep(0.05)
        return next(minted)

    monkeypatch.setattr(server_auth.secrets, "token_hex", slow_token)
    with ThreadPoolExecutor(max_workers=4) as pool:
        keys = list(pool.map(lambda _n: server_auth._auth_secret(), range(4)))
    assert len(set(keys)) == 1
    assert (tmp_path / "state" / "auth_secret.key").read_text(encoding="utf-8").strip() == keys[0].hex()
