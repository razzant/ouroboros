r"""The complete builtin interface, hosted apart from a real core, behind an actual TLS reverse proxy.

Topology (``tests/separate_ui_support.py``): the real ``server.py`` of this tree with
``OUROBOROS_UI_DIR=none`` and a network password; a Caddy interface host serving a copy of
``web/``; a Caddy front proxy with this run's certificate. Both Caddy configurations are
``docs/examples/separate_ui`` as published; the only change is the forwarded visitor address
(``SYNTHETIC_PEER``), because the browser here is on loopback and the core never gates loopback.
The verbatim example is checked separately.

What a test here counts as proof is the producer's own record: the proxy's and the interface
host's access logs (which upstream answered each request), the core's log and data files, the
core's HTTP answers and the bytes served. The browser's DOM is the consumer under test, not the
oracle. Requires ``OUROBOROS_RUN_UI_SMOKE=1`` and ``SEPARATE_UI_CADDY`` (an absolute Caddy v2 path).
Run from the repository root under the environment boundary (writes only beneath its temporary root
and ``OUROBOROS_UI_EVIDENCE_DIR``)::

    SEPARATE_UI_CADDY=/abs/path/caddy PLAYWRIGHT_BROWSERS_PATH=$HOME/Library/Caches/ms-playwright \
    OUROBOROS_RUN_UI_SMOKE=1 OUROBOROS_UI_EVIDENCE_DIR=/tmp/separate-ui-evidence \
    python -I -S scripts/safe_test.py --temp-parent /tmp/sui -- /abs/venv/bin/python -m pytest \
        -o addopts="" -p no:cacheprovider -m ui_browser tests/test_separate_ui_browser.py

The same environment with ``-- /abs/venv/bin/python -m tests.separate_ui_support`` starts the stack for a
person to browse (``tests/separate_ui_support.py`` ``main``).
"""
from __future__ import annotations

import hashlib
import http.client
import json
import os
import shutil
import socket
import ssl
import subprocess
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path

import pytest

from tests.separate_ui_support import (
    CADDY_ENV,
    CERT_SUBJECT,
    CONNECTED,
    DISCONNECTED,
    EXAMPLE,
    PASSWORD,
    PROBE,
    SYNTHETIC_PEER,
    BrowserSession,
    caddy_binary,
    evidence_dir,
    install_widgets,
    listening_sockets,
    real_core,
    separate_stack,
    wait_until,
)

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]

CORE_PREFIXES = ("/api", "/auth", "/onboarding", "/ws")


def _require_stack() -> None:
    if os.environ.get("OUROBOROS_RUN_UI_SMOKE") != "1":
        pytest.skip("set OUROBOROS_RUN_UI_SMOKE=1 to run browser UI qualification")
    if caddy_binary() is None:
        pytest.skip(f"set {CADDY_ENV} to an absolute Caddy v2 binary")


def _keep_logs(stack, evidence: Path) -> None:
    """Copy the three processes' logs and the routing record next to the screenshots (no key file)."""
    for process in (stack.core.process, stack.proxy, stack.interface):
        if process.log_path.exists():
            shutil.copy2(process.log_path, evidence / process.log_path.name)
    shutil.copy2(stack.certificate.cert, evidence / "cert.pem")
    (evidence / "routed.json").write_text(json.dumps({
        "routed": stack.routed(), "interface_paths": stack.interface_paths(),
        "core_pids": stack.core.process.pids, "core_exits": stack.core.process.exits,
        "core_boots": stack.core.boots(),
        "proxy_exits": stack.proxy.exits, "interface_exits": stack.interface.exits,
        "certificate_sha256": stack.certificate.sha256, "spki_pin": stack.certificate.spki_pin,
        "interface_manifest": stack.manifest, "verbatim_example": stack.verbatim, **stack.extra,
    }, indent=2), encoding="utf-8")


@contextmanager
def _kept_stack(tmp_path, request, **core_options):
    """A started stack whose logs are copied to the evidence folder after every process stopped."""
    _require_stack()
    evidence = evidence_dir(request.node.name, tmp_path)
    built = None
    try:
        with real_core(tmp_path, **core_options) as core, separate_stack(core, tmp_path / "stack") as built:
            built.extra["evidence"] = str(evidence)
            yield built
    finally:
        if built is not None:
            _keep_logs(built, evidence)


@pytest.fixture
def stack(tmp_path, request):
    with _kept_stack(tmp_path, request) as built:
        yield built


class _Client:
    """A plain HTTPS client of the proxy that trusts this run's certificate only."""

    def __init__(self, stack):
        self.host, self.port = "127.0.0.1", stack.proxy.port
        self.context = stack.certificate.ssl_context()

    def request(self, method, path, *, body=None, headers=None):
        connection = http.client.HTTPSConnection(self.host, self.port, context=self.context, timeout=30)
        try:
            payload = json.dumps(body).encode() if isinstance(body, (dict, list)) else body
            heads = dict(headers or {})
            if isinstance(body, (dict, list)):
                heads.setdefault("Content-Type", "application/json")
            connection.request(method, path, body=payload, headers=heads)
            response = connection.getresponse()
            return response.status, {k.lower(): v for k, v in response.getheaders()}, response.read(), \
                response.headers.get_all("Set-Cookie") or []
        finally:
            connection.close()


def _routed_after_sockets_close(stack, minimum: int = 1) -> list[dict]:
    """The proxy's access log once it holds ``minimum`` upgraded sockets. Caddy writes a WebSocket's line
    when that connection ends, so the line for a just-closed socket can trail the browser's close."""
    return wait_until(lambda: (rows := stack.routed()) and sum(
        1 for r in rows if r["uri"] == "/ws" and r["status"] == 101) >= minimum and rows, 15,
        f"fewer than {minimum} upgraded sockets in the proxy's access log")


def test_one_tls_origin_routes_each_prefix_to_its_owner_and_keeps_the_password_gate(stack):
    client = _Client(stack)
    interface = stack.interface_root

    # Real TLS: a client that does not trust this run's certificate is refused.
    with pytest.raises(urllib.error.URLError) as refused:
        urllib.request.urlopen(stack.url + "/", context=ssl.create_default_context(), timeout=10)  # noqa: S310
    assert isinstance(refused.value.reason, ssl.SSLCertVerificationError)
    with socket.create_connection(("127.0.0.1", stack.proxy.port), timeout=10) as raw, \
            stack.certificate.ssl_context().wrap_socket(raw, server_hostname="127.0.0.1") as tls:
        assert tls.version() == "TLSv1.3"
        assert hashlib.sha256(tls.getpeercert(binary_form=True)).hexdigest() == stack.certificate.sha256

    # The interface host's copy answers / and /static; the core itself serves neither.
    status, headers, body, _ = client.request("GET", "/")
    assert status == 200 and body == (interface / "index.html").read_bytes()
    assert b"ouroboros-ui-owner" not in body and headers["cache-control"] == "no-cache, must-revalidate"
    status, _, body, _ = client.request("GET", "/static/modules/api_client.js")
    assert status == 200 and body == (interface / "modules" / "api_client.js").read_bytes()
    for path in ("/", "/static/modules/api_client.js"):
        with pytest.raises(urllib.error.HTTPError) as direct:
            urllib.request.urlopen(stack.core.url + path, timeout=10)  # noqa: S310 - loopback core
        assert direct.value.code == 404
    assert not [p for p in interface.rglob("*") if p.suffix == ".py" or p.name in {"settings.json", "VERSION"}]

    # Open core routes answer anyone; the core's own setup assets are behind the gate like the rest.
    status, _, body, _ = client.request("GET", "/api/health")
    assert status == 200 and json.loads(body)["abi_version"]
    assert client.request("GET", "/onboarding/static/modules/api_client.js")[0] == 401

    # The gate: the forwarded visitor is not loopback, and no client header makes it so.
    for spoof in ({}, {"X-Forwarded-For": "127.0.0.1"}, {"X-Real-IP": "127.0.0.1"},
                  {"Forwarded": "for=127.0.0.1;proto=https"}, {"X-Forwarded-For": "127.0.0.1, ::1"}):
        status, _, body, _ = client.request("GET", "/api/state", headers=spoof)
        assert (status, json.loads(body)) == (401, {"error": "Authentication required."}), spoof
    status, _, body, cookies = client.request("POST", "/auth/login", body={"password": "wrong", "next": "/"})
    assert status == 401 and not cookies
    status, _, body, cookies = client.request("POST", "/auth/login", body={"password": PASSWORD, "next": "/\\evil"})
    assert status == 200 and json.loads(body)["next"] == "/"
    attributes = {part.strip().lower() for part in cookies[0].split(";")[1:]}
    assert {"httponly", "secure", "samesite=lax"} <= attributes, cookies  # Secure: the core saw TLS
    cookie = cookies[0].split(";", 1)[0]

    status, _, body, _ = client.request("GET", "/onboarding/static/modules/api_client.js", headers={"Cookie": cookie})
    assert status == 200 and body == (stack.core.checkout / "web" / "modules" / "api_client.js").read_bytes()
    status, _, body, _ = client.request("GET", "/api/state", headers={"Cookie": cookie})
    state = json.loads(body)
    assert status == 200 and state["served_ui"] == {"source": "none"}
    status, _, body, _ = client.request("GET", "/api/contract", headers={"Cookie": cookie})
    contract = json.loads(body)
    assert tuple(contract["core_paths"]) == CORE_PREFIXES and tuple(contract["interface_paths"]) == ("/", "/static")
    matcher = next(line.split()[2:] for line in (stack.core.checkout / EXAMPLE / "Caddyfile")
                   .read_text(encoding="utf-8").splitlines() if line.strip().startswith("@core path"))
    assert sorted(matcher) == sorted([p for c in contract["core_paths"] for p in (c, c + "/*")])

    # Unknown paths are answered by their owner, never with the interface's index page.
    status, _, body, _ = client.request("GET", "/no-such-page")
    assert status == 404 and b"<html" not in body.lower()
    status, _, body, _ = client.request("GET", "/api/no-such-route", headers={"Cookie": cookie})
    assert status == 404 and b"<html" not in body.lower()

    # The socket: refused at the handshake without a session, upgraded with one.
    from websockets.exceptions import InvalidStatus
    from websockets.sync.client import connect

    wss = f"wss://127.0.0.1:{stack.proxy.port}/ws"
    with pytest.raises(InvalidStatus) as handshake:
        connect(wss, ssl=client.context, open_timeout=15).close()
    assert handshake.value.response.status_code == 403
    with connect(wss, ssl=client.context, additional_headers={"Cookie": cookie}, open_timeout=15) as ws:
        frame = json.loads(ws.recv(timeout=40))
        assert frame.get("type")

    # The producers' records: every core prefix reached the core, everything else the interface host.
    routed = _routed_after_sockets_close(stack)
    for row in routed:
        path = row["uri"].split("?", 1)[0]
        owner = "core" if any(path == p or path.startswith(p + "/") for p in CORE_PREFIXES) else "interface"
        assert row["role"] == owner, row
    assert any(r["uri"] == "/ws" and r["role"] == "core" and r["websocket"] for r in routed)
    assert {p.split("?", 1)[0] for p in stack.interface_paths()} == {"/", "/static/modules/api_client.js",
                                                                     "/no-such-page"}
    for process in (stack.proxy, stack.interface):
        sockets = listening_sockets(process.proc.pid)
        if sockets is not None:
            assert sockets and all("127.0.0.1:" in line and "UDP" not in line for line in sockets), sockets
    stack.extra["verdicts"] = {"tls": "TLSv1.3", "visitor": SYNTHETIC_PEER, "served_ui": state["served_ui"],
                               "certificate_subject": CERT_SUBJECT}


def test_the_verbatim_example_overwrites_a_visitor_forwarded_claim(tmp_path, request):
    """Unchanged example, loopback browser: the visitor is the real peer, whatever it claims."""
    _require_stack()
    evidence = evidence_dir(request.node.name, tmp_path)
    with real_core(tmp_path) as core, separate_stack(core, tmp_path / "stack", verbatim=True) as stack:
        try:
            client = _Client(stack)
            # A claim to be a LAN visitor would be gated if it reached the core; the example
            # replaces it with the loopback peer, which the core does not gate.
            status, _, body, _ = client.request("GET", "/api/state", headers={"X-Forwarded-For": "198.51.100.99"})
            assert status == 200 and json.loads(body)["served_ui"] == {"source": "none"}
            assert client.request("GET", "/")[0] == 200
        finally:
            _keep_logs(stack, evidence)


BROWSERS = ["chromium", "webkit"]
_OK_REPLY = """(scope) => [...document.querySelectorAll(scope + ' .chat-bubble.assistant')]
    .some(bubble => /\\bOK\\b/.test(bubble.innerText))"""


def _session(playwright, stack, browser_name):
    return BrowserSession(playwright, stack, browser_name, Path(stack.extra["evidence"]))


@pytest.mark.parametrize("browser_name", BROWSERS)
def test_an_empty_browser_signs_in_on_the_shared_origin_and_returns_with_http_and_socket(stack, browser_name):
    pytest.importorskip("playwright.sync_api")
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        session = _session(playwright, stack, browser_name)
        page = session.page
        try:
            assert session.context.cookies() == []
            first = page.goto(stack.url + "/")
            assert first.status == 200
            details = first.security_details()
            assert details["subjectName"] == CERT_SUBJECT and details["protocol"] == "TLS 1.3", details
            page.wait_for_url(lambda address: address == stack.url + "/auth/login?next=%2F", timeout=30_000)
            session.shot("1-sign-in")
            session.sign_in("not-the-password")
            page.get_by_text("Invalid password.").wait_for()
            assert not [c for c in session.context.cookies() if c["name"] == "ouroboros_auth"]
            session.shot("2-wrong-password")
            session.sign_in()
            page.wait_for_url(stack.url + "/", timeout=30_000)
            page.wait_for_function(CONNECTED, timeout=60_000)
            page.locator("#chat-input").wait_for(state="visible")
            assert page.locator("#onboarding-overlay").count() == 0
            cookie = next(c for c in session.context.cookies() if c["name"] == "ouroboros_auth")
            assert cookie["secure"] is True and cookie["httpOnly"] is True and cookie["sameSite"] == "Lax"
            assert session.api("GET", "/api/state", None)["json"]["served_ui"] == {"source": "none"}
            page.locator("#chat-status").filter(has_text="Online").wait_for(timeout=30_000)
            session.shot("3-returned")
            # Interface page, the core's sign-in, its refusal (a POSTed page), the interface page again.
            assert session.loads() == [stack.url + "/", stack.url + "/auth/login?next=%2F",
                                       stack.url + "/auth/login", stack.url + "/"]

            # Expiry: a session the core itself signed 31 days ago is refused (one signed a minute ago is
            # the control); after a network blip the page hands over to sign-in and comes back.
            client = _Client(stack)
            fresh, expired = (_signed_session(stack.core.data_dir, age) for age in (60, 31 * 24 * 3600))
            controls = [client.request("GET", "/api/state", headers={"Cookie": f"ouroboros_auth={value}"})[0]
                        for value in (fresh, expired)]
            session.context.add_cookies([{"name": "ouroboros_auth", "value": expired, "url": stack.url,
                                          "httpOnly": True, "secure": True, "sameSite": "Lax"}])
            stack.proxy.stop()
            page.wait_for_function(DISCONNECTED, timeout=30_000)
            stack.proxy.start()
            page.wait_for_url(stack.url + "/auth/login?next=%2F", timeout=60_000)
            session.shot("4-expired")
            session.sign_in()
            page.wait_for_url(stack.url + "/", timeout=30_000)
            page.wait_for_function(CONNECTED, timeout=60_000)
            assert session.unexpected_errors() == []
        finally:
            session.close()
    assert controls == [200, 401], controls
    routed = _routed_after_sockets_close(stack)
    # Every load of / reached the interface host (a repeat may revalidate: no-cache).
    assert [r["status"] in (200, 304) for r in routed if r["uri"] == "/"] == [True] * 3
    assert {r["role"] for r in routed if r["uri"] == "/" or r["uri"].startswith("/static/")} == {"interface"}
    assert any(r["uri"] == "/api/onboarding" and r["status"] == 401 for r in routed)
    assert [r["status"] for r in routed if r["uri"].startswith("/auth/login") and r["method"] == "POST"] == [
        401, 303, 303]
    # The expired session was refused at the socket handshake as well as over HTTP.
    assert any(r["uri"] == "/ws" and r["status"] == 403 for r in routed)
    assert any(r["uri"] == "/ws" and r["status"] == 101 and r["role"] == "core" for r in routed)
    assert not [p for p in stack.interface_paths() if p.startswith(("/api", "/auth", "/ws", "/onboarding"))]


def _signed_session(data_dir: Path, age_sec: int) -> str:
    """A session value the core's own signer would issue ``age_sec`` ago, with the core's own key."""
    from ouroboros import server_auth

    key = bytes.fromhex((data_dir / "state" / "auth_secret.key").read_text(encoding="utf-8").strip())
    previous, server_auth._auth_secret_cache = server_auth._auth_secret_cache, key
    try:
        return server_auth._session_value(PASSWORD, int(time.time()) - age_sec)
    finally:
        server_auth._auth_secret_cache = previous


def _history_notice(page) -> str:
    """The Main feed's persistent history note ("Some saved history is not loaded ..."), if shown."""
    return page.evaluate("""() => [...document.querySelectorAll(
        '#page-chat .chat-history-status, #page-chat .chat-load-older.has-gaps :is(.chat-load-older-note, .chat-history-gap)')]
        .filter(node => !node.hidden).map(node => node.textContent.trim()).join(' | ')""")


def _send(page, scope: str, text: str) -> str:
    """Send through the visible composer of ``scope``; return the message's own client id."""
    if scope == "#chat-messages":
        page.fill("#chat-input", text)
        page.click("#chat-send")
        bubble = page.locator("#chat-messages .chat-bubble.user", has_text=text)
    else:
        page.locator(f"{scope} textarea").fill(text)
        page.locator(f"{scope} .chat-send-inline").click()
        bubble = page.locator(f"{scope} .chat-bubble.user", has_text=text)
    bubble.last.wait_for(timeout=30_000)
    page.wait_for_function(_OK_REPLY, arg=scope, timeout=60_000)
    return bubble.last.get_attribute("data-client-message-id")


@pytest.mark.parametrize("browser_name", BROWSERS)
def test_the_interface_and_the_core_keep_their_own_lifetimes(stack, browser_name):
    """Projects, a correlated chat task and its history survive a proxy blip, a core stop/start and
    an interface host stop/start; nothing reloads the page except the explicit reload."""
    pytest.importorskip("playwright.sync_api")
    from playwright.sync_api import sync_playwright

    core = stack.core
    with sync_playwright() as playwright:
        session = _session(playwright, stack, browser_name)
        page = session.page
        try:
            session.open_signed_in()
            signed_in = len(session.loads())

            # A project made in the interface is the core's: its registry answers it.
            page.locator("#nav-projects-add").click()
            page.locator("[data-np-name]").fill("Separate host room")
            with page.expect_response(lambda r: r.url.endswith("/api/projects") and r.request.method == "POST") as made:
                page.locator("[data-np-create]").click()
            assert made.value.status == 200, made.value.text()
            project = made.value.json().get("project") or made.value.json()
            row = page.locator(f'.nav-project-row[data-project-id="{project["id"]}"]')
            row.wait_for(timeout=30_000)
            listed = session.api("GET", "/api/projects", None)["json"]
            assert any(p["id"] == project["id"] for p in listed["projects"]), listed

            # The project's own thread: the interface opens a new project's panel itself (app.js); a row
            # click here would toggle that panel closed again, so the test only waits for it.
            page.locator("#project-panel.open").wait_for(timeout=30_000)
            room_id = _send(page, "#project-panel", f"separate host room {browser_name}")
            room_task = wait_until(lambda: core.task_for_message(room_id), 30, "no task for the room message")
            room_history = session.api("GET", f"/api/chat/history?chat_id={project['chat_id']}", None)["json"]
            assert any(m.get("client_message_id") == room_id for m in room_history["messages"]), room_history
            session.shot("1-project-room")
            page.locator("#project-panel-close").click()
            page.locator("#project-panel.open").wait_for(state="detached", timeout=10_000)

            # Main chat: the owner message and the task the core started for it are one correlation.
            main_text = f"separate host main {browser_name}"
            main_id = _send(page, "#chat-messages", main_text)
            task = wait_until(lambda: core.task_for_message(main_id), 30, "no task_received for the message")
            replies = wait_until(lambda: [r for r in core.jsonl("logs/chat.jsonl") if r.get("direction") == "out"
                                          and r.get("task_id") == task["id"]], 30, "no reply row for the task")
            history = session.api("GET", "/api/chat/history?chat_id=1", None)["json"]["messages"]
            assert any(m.get("client_message_id") == main_id and m.get("role") == "user" for m in history)
            assert any(m.get("task_id") == task["id"] and m.get("role") == "assistant" for m in history)
            session.shot("2-main-chat")

            # A network blip: the proxy restarts and every connection drops.
            stack.proxy.stop()
            page.wait_for_function(DISCONNECTED, timeout=30_000)
            stack.proxy.start()
            page.wait_for_function(CONNECTED, timeout=60_000)
            notices = {"after_proxy_blip": _history_notice(page)}

            # The core stops and starts as its own process; the page waits and resynchronizes.
            first_pid = core.pid
            core.stop()
            page.wait_for_function(DISCONNECTED, timeout=30_000)
            page.locator("#chat-status").filter(has_not_text="Online").wait_for(timeout=30_000)
            session.shot("3-core-stopped")
            core.start()
            page.wait_for_function(CONNECTED, timeout=90_000)
            page.locator("#chat-status").filter(has_text="Online").wait_for(timeout=30_000)
            assert core.pid != first_pid and core.boots() == 2
            assert len(session.loads()) == signed_in, "the core's return never reloads a page it does not own"
            history = session.api("GET", "/api/chat/history?chat_id=1", None)["json"]["messages"]
            assert any(m.get("client_message_id") == main_id for m in history)
            notices["after_core_restart"] = _history_notice(page)
            session.shot("4-core-back")

            # The interface host stops: the core and the open page keep working; only / is gone.
            stack.interface.stop()
            assert session.api("GET", "/api/state", None)["status"] == 200
            assert page.evaluate(CONNECTED)
            down = session.api("GET", "/", None)["status"]
            session.shot("5-interface-host-stopped")
            stack.interface.start()
            page.reload()
            page.wait_for_function(CONNECTED, timeout=60_000)
            page.wait_for_function(_OK_REPLY, arg="#chat-messages", timeout=30_000)
            assert core.pid == core.process.pids[-1] and len(core.process.pids) == 2
            assert len(session.loads()) == signed_in + 1
            notices["after_explicit_reload"] = _history_notice(page)
            assert notices["after_explicit_reload"] == "", "a reloaded page reads its whole history"
            session.shot("6-interface-host-back")
            assert session.unexpected_errors() == []
        finally:
            session.close()
    stack.extra["journey"] = {"project": project, "main": {"client_message_id": main_id, "task_id": task["id"],
                                                           "reply_rows": len(replies)},
                              "room": {"client_message_id": room_id, "task_id": room_task["id"]},
                              "interface_down_status": down, "main_history_notices": notices}
    assert down == 502
    routed = _routed_after_sockets_close(stack, minimum=3)
    assert [r for r in routed if r["uri"] == "/" and r["status"] == 502 and r["role"] != "core"]
    assert sum(1 for r in routed if r["uri"] == "/ws" and r["status"] == 101) >= 3


def _open_activity_controls(page, task_id: str):
    page.click('[data-nav-page="dashboard"]')
    page.click('[data-dashboard-tab="activity"]')
    button = page.locator(f'[data-activity-section="queue"] [data-id="{task_id}"]')
    button.wait_for(state="visible", timeout=30_000)
    button.click()
    menu = page.locator("body > .task-control-menu")
    menu.wait_for(state="visible")
    return menu


def test_stop_ends_a_held_task_and_the_core_records_the_stop(stack):
    """A live task held in its model call is stopped from the interface; the core's records say so."""
    pytest.importorskip("playwright.sync_api")
    from playwright.sync_api import sync_playwright

    core = stack.core
    entered = core.model.hold("HOLD-STOP-7Q")
    with sync_playwright() as playwright:
        session = _session(playwright, stack, "chromium")
        page = session.page
        try:
            session.open_signed_in()
            page.fill("#chat-input", "HOLD-STOP-7Q: wait in your first model call")
            page.click("#chat-send")
            assert entered.wait(60), "the task never reached its model call"
            actor = wait_until(lambda: next((a for a in session.api("GET", "/api/state", None)["json"]
                                             .get("active_chat_activities", []) if a["kind"] == "direct_chat"), None),
                               30, "no live direct turn")
            task_id = actor["activity_id"]
            menu = _open_activity_controls(page, task_id)
            session.shot("1-stop-menu")
            # A direct turn stops cooperatively: its held model call returns once Stop is sent.
            page.on("request", lambda request: request.method == "POST" and request.url.endswith(
                f"/api/tasks/{task_id}/cancel") and threading.Timer(0.3, core.model.release).start())
            with page.expect_response(lambda r: r.url.endswith(f"/api/tasks/{task_id}/cancel")) as stopped:
                menu.locator('[data-task-control="stop_now"]').click()
            receipt = stopped.value
            sent, answer = receipt.request.post_data_json, receipt.json()
            assert receipt.status in (200, 202) and answer.get("ok") is True, answer
            assert sent["cascade"] is True and sent["stop_action_id"], sent
            wait_until(lambda: not any(a["activity_id"] == task_id for a in session.api("GET", "/api/state", None)
                                       ["json"].get("active_chat_activities", [])), 60, "the task never settled")
            intents = wait_until(lambda: [r for r in core.jsonl("logs/supervisor.jsonl") if r.get("type") ==
                                          "cancel_intent" and r.get("task_id") == task_id and r.get("event") == "settled"]
                                 and [r for r in core.jsonl("logs/supervisor.jsonl") if r.get("type") == "cancel_intent"
                                      and r.get("task_id") == task_id], 30, "the core never settled the stop")
            record = session.api("GET", f"/api/tasks/{task_id}", None)
            session.shot("2-stopped")
            assert session.unexpected_errors() == []
        finally:
            core.model.release()
            session.close()
    stack.extra["stop"] = {"task_id": task_id, "request": sent, "response": answer,
                           "cancel_intents": intents, "task_record": record}
    requested = next(r for r in intents if r["event"] == "requested")
    settled = next(r for r in intents if r["event"] == "settled")
    assert requested["source"] == "http_cascade" and settled["outcome"] == "cancelled", intents
    assert record["status"] == 200, record
    assert [r for r in stack.routed() if r["uri"] == f"/api/tasks/{task_id}/cancel" and r["role"] == "core"
            and r["status"] in (200, 202)]


def _seed_late_quiz(core, question: str) -> tuple[str, str]:
    """A question its (completed) task asked in Main, written by the core's own writers."""
    from ouroboros import dialogue_store
    from ouroboros.owner_quiz import record_asked, record_publication
    from ouroboros.task_results import write_task_result

    task_id, quiz_id = "separate-ui-late-task", "late-q"
    write_task_result(core.data_dir, task_id, "completed", chat_id=1)
    record_asked(core.data_dir, task_id, quiz_id=quiz_id, question=question, options=["Keep", "Change"], chat_id=1)
    # The publication writer binds the card's address before exposing it; only that address takes a late answer (#1899).
    record_publication(core.data_dir, task_id, quiz_id, 1)
    quiz = {"quiz_id": quiz_id, "question": question, "state": "open", "chat_id": 1,
            "options": [{"label": "Keep"}, {"label": "Change"}], "asked_at": "2026-10-13T00:00:00Z"}
    dialogue_store.append(core.data_dir, "chat", {"ts": "2026-10-13T00:00:00Z", "direction": "out", "chat_id": 1,
                                                  "text": question, "type": "quiz", "task_id": task_id, "quiz": quiz})
    return task_id, quiz_id


def test_questions_are_answered_once_live_and_late_through_the_proxy(stack):
    """A live question (the model calls escalate) and a late one (its task already ended):
    the card's answer is recorded once; a repeat is a duplicate; a competing answer is refused."""
    pytest.importorskip("playwright.sync_api")
    from playwright.sync_api import sync_playwright

    core = stack.core
    late_task, late_quiz = _seed_late_quiz(core, "Late question: keep the separate host?")
    core.model.ask("QUIZ-LIVE-7Q", {"question": "Live question: which colour should the separate host use?",
                                    "options": [{"label": "Teal", "recommended": True}, {"label": "Amber"}],
                                    "stake": "qualification only", "wait_for_answer": True})
    responses = []
    with sync_playwright() as playwright:
        session = _session(playwright, stack, "chromium")
        page = session.page
        try:
            session.open_signed_in()
            page.on("response", lambda r: r.url.endswith("/api/decisions") and responses.append(r))
            # Live: the task waits on the card.
            page.fill("#chat-input", "QUIZ-LIVE-7Q: ask me first")
            page.click("#chat-send")
            asked = page.locator(".chat-quiz-card", has_text="Live question")
            asked.wait_for(timeout=60_000)
            live_task, live_quiz = asked.get_attribute("data-task-id"), asked.get_attribute("data-quiz-id")
            card = page.locator(f'.chat-quiz-card[data-task-id="{live_task}"][data-quiz-id="{live_quiz}"]')
            wait_until(lambda: ((_task_record(core, live_task).get("owner_wait") or {}).get("state") == "waiting"),
                       30, "the task is not waiting for the owner")
            session.shot("1-live-question")
            card.locator(".chat-quiz-option", has_text="Amber").click()
            page.locator(f'.chat-quiz-card[data-task-id="{live_task}"][data-state="answered"]').wait_for(timeout=30_000)
            page.wait_for_function(_OK_REPLY, arg="#chat-messages", timeout=60_000)
            # Late: the seeded card of a completed task still takes one answer, delivered as a message.
            late = page.locator(f'.chat-quiz-card[data-task-id="{late_task}"]')
            late.wait_for(timeout=30_000)
            late.locator(".chat-quiz-option", has_text="Change").click()
            page.locator(f'.chat-quiz-card[data-task-id="{late_task}"][data-state="answered"]').wait_for(timeout=30_000)
            session.shot("2-answered")
            wait_until(lambda: len(responses) >= 2, 30, "two answers were not both sent")
            answers = {r.request.post_data_json["decision_id"]: (r.status, r.request.post_data_json, r.json())
                       for r in responses}
            replays = {}
            for decision, (_status, sent, _body) in list(answers.items()):
                again = session.api("POST", "/api/decisions", sent)
                rival = session.api("POST", "/api/decisions", {**sent, "request_id": sent["request_id"] + "-rival",
                                                              "option_index": 0})
                replays[decision] = {"again": again, "rival": rival}
            assert session.unexpected_errors() == []
        finally:
            session.close()
    live_id, late_id = f"quiz:{live_task}:{live_quiz}", f"quiz:{late_task}:{late_quiz}"
    stack.extra["questions"] = {"answers": answers, "replays": replays}
    status, sent, body = answers[live_id]
    assert status == 200 and body["state"] == "answered" and body["duplicate"] is False and sent["option_index"] == 1
    status, sent, body = answers[late_id]
    assert status == 200 and body.get("answered_after_terminal") is True and body.get("forwarded") is True, body
    for decision in (live_id, late_id):
        assert replays[decision]["again"]["status"] == 200 and replays[decision]["again"]["json"]["duplicate"] is True
        assert replays[decision]["rival"]["status"] == 409, replays[decision]["rival"]
        assert replays[decision]["rival"]["json"]["error"] == "quiz_closed"
    live_record = _task_record(core, live_task)
    assert live_record["owner_quiz"][live_quiz]["state"] == "answered"
    assert live_record["owner_quiz"][live_quiz]["answered_index"] == 1
    late_record = _task_record(core, late_task)["owner_quiz"][late_quiz]
    assert late_record["state"] == "answered" and late_record.get("answered_after_terminal") is True
    # The late answer reached Main once, as the owner's own message.
    delivered = [r for r in core.jsonl("logs/chat.jsonl") if r.get("direction") == "in"
                 and r.get("client_message_id") == f"quiz_late_answer:{late_task}:{late_quiz}"]
    assert len(delivered) == 1, delivered


def _task_record(core, task_id: str) -> dict:
    """The core's durable task result, read by its own reader from its data root."""
    from ouroboros.task_results import load_task_result

    return load_task_result(core.data_dir, task_id) or {}


def test_settings_save_reports_the_restart_the_core_requires(stack):
    pytest.importorskip("playwright.sync_api")
    from playwright.sync_api import expect, sync_playwright

    core = stack.core
    settings_path = core.data_dir / "settings.json"
    with sync_playwright() as playwright:
        session = _session(playwright, stack, "chromium")
        page = session.page
        try:
            session.open_signed_in()

            def open_settings():
                page.locator('[data-nav-page="settings"]').click()
                expect(page.locator("#btn-save-settings")).to_be_enabled(timeout=30_000)

            open_settings()
            wait_until(lambda: "OUROBOROS_MAX_WORKERS" not in session.api("GET", "/api/settings", None)["json"]
                       ["_meta"]["restart_state"]["unknown_keys"], 30, "worker count never applied")
            page.locator("#s-workers").evaluate(
                "(node) => {node.value='2'; node.dispatchEvent(new Event('input', {bubbles: true}));}")
            with page.expect_response(lambda r: r.url.endswith("/api/settings") and r.request.method == "POST") as saved:
                page.locator("#btn-save-settings").click()
                if page.locator("[data-confirm-ok]").is_visible():
                    page.locator("[data-confirm-ok]").click()
            expect(page.locator("#settings-status")).to_contain_text("saved", timeout=30_000)
            save_request = saved.value.request.post_data_json  # Playwright objects are read inside the session
            expect(page.locator("#btn-restart-now")).to_be_visible()
            expect(page.locator("#settings-restart-status")).to_contain_text("Restart Ouroboros to apply")
            session.shot("1-restart-needed")
            written = json.loads(settings_path.read_text(encoding="utf-8"))
            state = session.api("GET", "/api/settings", None)["json"]["_meta"]["restart_state"]

            # A documented refusal writes nothing (the preference file is the oracle).
            preferences = core.data_dir / "state" / "ui_preferences.json"
            before = preferences.read_bytes() if preferences.exists() else None
            refused = session.api("POST", "/api/ui/preferences", {"separate_ui_unknown_key": True})
            unchanged = (preferences.read_bytes() if preferences.exists() else None) == before

            core.restart()
            page.wait_for_function(CONNECTED, timeout=90_000)
            page.reload()
            page.wait_for_function(CONNECTED, timeout=60_000)
            open_settings()
            after = session.api("GET", "/api/settings", None)["json"]["_meta"]["restart_state"]
            expect(page.locator("#btn-restart-now")).to_be_hidden(timeout=30_000)
            session.shot("2-applied")
            assert session.unexpected_errors() == []
        finally:
            session.close()
    stack.extra["settings"] = {"save_request": save_request, "restart_state": state,
                               "refused": refused, "file_unchanged": unchanged, "after_restart": after}
    assert written["OUROBOROS_MAX_WORKERS"] in (2, "2")
    assert "OUROBOROS_UI_DIR" not in written, "a save never authors the operator's interface choice"
    assert "OUROBOROS_MAX_WORKERS" in state["restart_keys"] and state["restart_required"] is True
    assert refused["status"] == 400 and unchanged, refused
    assert after["restart_required"] is False and "OUROBOROS_MAX_WORKERS" not in after["restart_keys"]


@pytest.mark.parametrize("browser_name", BROWSERS)
def test_uploaded_bytes_come_back_identical_through_the_proxy(stack, browser_name):
    pytest.importorskip("playwright.sync_api")
    from playwright.sync_api import sync_playwright

    core = stack.core
    payload = os.urandom(768 * 1024)
    digest = hashlib.sha256(payload).hexdigest()
    with sync_playwright() as playwright:
        session = _session(playwright, stack, browser_name)
        page = session.page
        try:
            session.open_signed_in()
            # A chat attachment: multipart through the proxy, stored and served back by the core.
            with page.expect_response(lambda r: r.url.endswith("/api/chat/upload") and r.request.method == "POST") as up:
                page.locator("#chat-file-input").set_input_files(
                    [{"name": "separate-ui.bin", "mimeType": "application/octet-stream", "buffer": payload}])
                page.locator("#chat-attachment-preview .attach-badge").wait_for()
                page.fill("#chat-input", "bytes through the proxy")
                page.click("#chat-send")
            uploaded = up.value.json()
            served = session.fetch_bytes(uploaded["view"]["url"])
            ranged = session.fetch_bytes(uploaded["view"]["url"], {"Range": "bytes=0-15"})
            page.wait_for_function(_OK_REPLY, arg="#chat-messages", timeout=60_000)
            session.shot("1-attachment")

            # The Files page: a dropped file is written under the operator's root and downloads intact.
            page.locator('[data-nav-page="files"]').click()
            page.locator("#page-files").wait_for(state="visible")
            page.evaluate("""(bytes) => {
                const transfer = new DataTransfer();
                transfer.items.add(new File([new Uint8Array(bytes)], 'dropped.bin', {type: 'application/octet-stream'}));
                document.querySelector('.files-layout').dispatchEvent(new DragEvent('drop',
                    {bubbles: true, cancelable: true, dataTransfer: transfer}));
            }""", list(payload[:65536]))
            page.locator("#files-preview-meta, #files-preview-status").filter(has_text="Upload complete").first.wait_for(
                timeout=30_000)
            page.locator(".files-entry").filter(has_text="dropped.bin").first.click()
            with page.expect_download() as download:
                page.locator("#files-download").click()
            fetched = Path(download.value.path()).read_bytes()
            session.shot("2-files")
            assert session.unexpected_errors() == []
        finally:
            session.close()
    stored = core.data_dir / "uploads" / uploaded["filename"]
    inbound = [r for r in core.jsonl("logs/chat.jsonl") if r.get("direction") == "in" and r.get("attachments")]
    stack.extra["bytes"] = {"sha256": digest, "upload": uploaded, "inbound": inbound, "served": served,
                            "ranged": ranged}
    assert uploaded["sha256"] == digest and uploaded["size"] == len(payload)
    assert hashlib.sha256(stored.read_bytes()).hexdigest() == digest
    assert (served["status"], served["sha256"], served["size"]) == (200, digest, len(payload))
    assert served["headers"]["x-content-type-options"] == "nosniff"
    assert (ranged["status"], bytes(ranged["head"])) == (206, payload[:16])
    assert [ref["sha256"] for ref in inbound[-1]["attachments"]] == [digest]
    dropped = Path(core.env["OUROBOROS_FILE_BROWSER_DEFAULT"]) / "dropped.bin"
    assert dropped.read_bytes() == payload[:65536] and fetched == payload[:65536]
    multipart = [r for r in stack.routed() if r["uri"] in ("/api/chat/upload", "/api/files/upload")]
    assert {(r["uri"], r["role"], r["status"]) for r in multipart} == {
        ("/api/chat/upload", "core", 200), ("/api/files/upload", "core", 200)}


@pytest.mark.parametrize("browser_name", BROWSERS)
def test_widgets_and_the_author_kit_run_through_the_proxy_under_the_password(stack, browser_name):
    """A module widget's bridge, blob resources, fault and disposer, and the published author kit's
    module and route iframe, on a core that serves no interface, signed in through the proxy."""
    pytest.importorskip("playwright.sync_api")
    from playwright.sync_api import sync_playwright

    core = stack.core
    install_widgets(core)
    card = lambda key: f'[data-widget-key="{key}"]'  # noqa: E731
    with sync_playwright() as playwright:
        session = _session(playwright, stack, browser_name)
        page = session.page
        try:
            session.open_signed_in()
            toggles = {name: session.api("POST", f"/api/skills/{name}/toggle", {"enabled": True})
                       for name in ("author_ui_kit", PROBE)}
            assert all(r["status"] == 200 for r in toggles.values()), toggles
            page.locator('[data-nav-page="widgets"]').click()
            page.locator("#page-widgets").wait_for(state="visible")

            probe = page.frame_locator(f'{card(PROBE + ":probe")} iframe')
            for key, value in (("data", "separate-ui"), ("module", "blob-module-ok"), ("blob-image", "1")):
                probe.locator(f'#root[data-{key}="{value}"]').wait_for(timeout=30_000)
            # A load that bypasses the bridge: its outcome is the engine's cookie policy (compared below).
            probe.locator("#root[data-passive]").wait_for(timeout=30_000)
            passive = probe.locator("#root").get_attribute("data-passive")
            fault = page.locator(f'{card(PROBE + ":fault")} [data-widget-status][data-widget-fault]')
            fault.wait_for(timeout=30_000)
            fault_text = fault.inner_text()

            page.locator(f'{card("author_ui_kit:module")} [data-widget-power]').click()
            controls = page.frame_locator(f'{card("author_ui_kit:module")} iframe')
            controls.get_by_role("heading", name="Author controls").wait_for(timeout=30_000)
            controls.get_by_role("button", name="Preview").click()
            preview = controls.locator('[role="status"]').inner_text()
            page.locator(f'{card("author_ui_kit:page")} [data-widget-power]').click()
            frame_src = page.locator(f'{card("author_ui_kit:page")} iframe').get_attribute("src")
            routed_page = page.frame_locator(f'{card("author_ui_kit:page")} iframe')
            routed_page.get_by_role("heading", name="Author controls").wait_for(timeout=30_000)
            session.shot("1-widgets", full_page=True)

            # Leaving the page disposes the module; its hook still reaches the core through the bridge.
            page.locator('[data-nav-page="chat"]').click()
            hits = wait_until(lambda: (r := session.api("GET", f"/api/extensions/{PROBE}/hits", None))["json"]
                              and r["json"]["disposed"] >= 1 and r["json"], 30, "the disposer never reached the core")
            # The deliberate fault is the fault card's own frame error; nothing else may fail.
            assert [e for e in session.unexpected_errors() if "separate-ui deliberate fault" not in e] == []
        finally:
            session.close()
    stack.extra["widgets"] = {"toggles": toggles, "fault": fault_text, "preview": preview, "page_src": frame_src,
                              "hits": hits, "passive_load": passive}
    # The fault reaches its card in both engines; WebKit masks the frame's message as "Script error.".
    assert fault_text.startswith("Widget script error: "), fault_text
    assert "separate-ui deliberate fault" in fault_text or browser_name == "webkit", fault_text
    assert preview == "My notes: List, enabled", preview
    assert frame_src == "/api/extensions/author_ui_kit/page"
    routed = stack.routed()
    statuses = {(r["uri"], r["status"]) for r in routed if r["uri"].startswith("/api/extensions/")}
    # Chromium sends no session from the sandboxed frame (401, refused); WebKit was observed to send it.
    passive_status = {"refused": 401, "loaded": 200}[passive]
    for expected in ((f"/api/extensions/{PROBE}/data", 200), (f"/api/extensions/{PROBE}/lib", 200),
                     (f"/api/extensions/{PROBE}/logo", 200), (f"/api/extensions/{PROBE}/logo?passive=1", passive_status),
                     (f"/api/extensions/{PROBE}/disposed", 200), ("/api/extensions/author_ui_kit/author-kit", 200),
                     ("/api/extensions/author_ui_kit/page", 200), (f"/api/extensions/{PROBE}/module/widget.js", 200)):
        assert expected in statuses, (expected, sorted(statuses))
    assert {r["role"] for r in routed if r["uri"].startswith("/api/extensions/")} == {"core"}
    assert not [p for p in stack.interface_paths() if p.startswith(("/api", "/auth", "/ws", "/onboarding"))]


def test_update_status_restart_and_panic_are_the_cores_own_process_facts(stack):
    """Read-only update facts; Restart re-executes the core in place; Panic ends it with its exit
    code, while the interface host keeps serving the page."""
    pytest.importorskip("playwright.sync_api")
    from playwright.sync_api import expect, sync_playwright

    core = stack.core
    head = subprocess.run(["git", "-C", str(core.checkout), "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True, timeout=30).stdout.strip()
    with sync_playwright() as playwright:
        session = _session(playwright, stack, "chromium")
        page = session.page
        try:
            session.open_signed_in()
            page.click('[data-nav-page="dashboard"]')
            page.click('[data-dashboard-tab="updates"]')
            expect(page.locator("#updates-summary")).to_contain_text("Managed updates are unavailable", timeout=30_000)
            update = session.api("GET", "/api/update/status", None)
            preflight = session.api("POST", "/api/update/preflight", {})
            session.shot("1-updates")

            # Restart from the chat header: one process, a new server generation.
            page.click('[data-nav-page="chat"]')
            pid, boots, loads = core.pid, core.boots(), len(session.loads())
            page.locator('[data-chat-command="restart"]').click()
            page.locator("[data-confirm-ok]").click()
            wait_until(lambda: core.boots() == boots + 1, 90, "no new server generation started")
            core.wait_ready()
            page.wait_for_function(CONNECTED, timeout=90_000)
            page.locator("#chat-status").filter(has_text="Online").wait_for(timeout=30_000)
            restarted = {"same_pid": core.pid == pid and core.process.running, "loads_unchanged":
                         len(session.loads()) == loads, "uptime": session.api("GET", "/api/state", None)["json"]
                         .get("uptime"), "shutdown_rows": [r for r in core.jsonl("logs/supervisor.jsonl")
                                                           if r.get("type") == "server_shutdown"]}
            session.shot("2-restarted")

            # Panic: the core stops everything and exits; the interface host is untouched.
            page.locator('#page-chat [data-chat-command="panic"]').click()
            expect(page.locator("[data-confirm-ok]")).to_have_count(0)
            exit_code = core.process.wait_exit(90)
            page.wait_for_function(DISCONNECTED, timeout=30_000)
            after = {"api": session.api("GET", "/api/state", None)["status"],
                     "interface": session.api("GET", "/", None)["status"]}
            session.shot("3-after-panic")
        finally:
            session.close()
    flag = core.data_dir / "state" / "panic_stop.flag"
    stack.extra["process_facts"] = {"head": head, "update_status": update, "preflight": preflight,
                                    "restart": restarted, "panic_exit_code": exit_code, "after_panic": after,
                                    "panic_flag": flag.read_text(encoding="utf-8").strip() if flag.exists() else None}
    assert update["status"] == 200 and update["json"]["running_sha"] == head
    assert "managed_updates_unavailable" in update["json"].get("warnings", [])
    assert preflight["status"] == 200 and preflight["json"]["merge_plan"]["available"] is False, preflight
    assert restarted["same_pid"] and restarted["loads_unchanged"], restarted
    assert any(r.get("cause") == "restart_requested" and r.get("restart_exit") is True
               for r in restarted["shutdown_rows"]), restarted["shutdown_rows"]
    assert "Re-executing direct server mode" in core.process.log_text()
    assert exit_code == 99 and stack.extra["process_facts"]["panic_flag"] == "panic"
    assert after == {"api": 502, "interface": 200}, after


def test_a_fresh_install_is_set_up_without_a_key_through_the_proxy(tmp_path, request):
    """No settings at all: the separately hosted page signs in, shows the core's own setup wizard,
    completes it against a loopback OpenAI-compatible endpoint with no key, and then chats. The
    endpoint's own record of every model call shows that no credential reached it."""
    pytest.importorskip("playwright.sync_api")
    from playwright.sync_api import sync_playwright

    with _kept_stack(tmp_path, request, provider=False) as stack, sync_playwright() as playwright:
        core = stack.core
        assert not (core.data_dir / "settings.json").exists()
        authorization = []  # the Authorization header (None: absent) of each call the keyless endpoint answered

        def recorded(handler, handle=core.model._handle):
            authorization.append(handler.headers.get("Authorization"))
            return handle(handler)

        core.model._handle = recorded
        session = _session(playwright, stack, "chromium")
        page = session.page
        try:
            page.goto(stack.url + "/")
            page.wait_for_url(lambda address: address.split("?")[0].endswith("/auth/login"), timeout=30_000)
            session.sign_in()
            page.wait_for_url(stack.url + "/", timeout=30_000)
            before = session.api("GET", "/api/onboarding/bootstrap", None)
            wizard = page.frame_locator("#onboarding-overlay iframe.onboarding-frame")
            wizard.locator('[data-collapse="api-access"] > summary').click(timeout=30_000)
            wizard.locator('[data-collapse="more-providers"] > summary').click()
            wizard.locator("#openai-compatible-url").fill(core.model.base_url)
            session.shot("1-setup-provider")
            wizard.locator("#next-btn").click()
            wizard.locator('[data-model-role="main"] [data-model-role-model]').fill("openai-compatible::mock-model")
            for _step in range(3):
                wizard.locator("#next-btn").click()
            wizard.locator(".summary-card").first.wait_for(timeout=30_000)
            session.shot("2-setup-summary")
            with page.expect_response(lambda r: r.url.endswith("/api/onboarding/complete")) as completed:
                wizard.locator("#next-btn").click()
            # The overlay navigates at once on success, so the body is gone; the core's records answer.
            done = (completed.value.status, completed.value.request.post_data_json)
            page.locator("#onboarding-overlay").wait_for(state="detached", timeout=60_000)
            page.wait_for_function(CONNECTED, timeout=60_000)
            probe = session.api("GET", "/api/onboarding", None)["status"]
            after = session.api("GET", "/api/onboarding/bootstrap", None)
            # Completion starts the supervisor in this same process; its workers follow.
            ready = lambda state: (state.get("supervisor_ready") is True and not state.get("supervisor_error")  # noqa: E731
                                   and state.get("workers_total"))
            states = []
            wait_until(lambda: ready(states.append(session.api("GET", "/api/state", None)["json"]) or states[-1]),
                       120, "no running worker pool after setup")
            core.prove_origin()
            page.fill("#chat-input", "first words after a keyless setup")
            page.click("#chat-send")
            refused = lambda: [r for r in core.jsonl("logs/events.jsonl") if r.get("type") == "llm_api_error"  # noqa: E731
                               and "Missing credentials" in str(r.get("error"))]
            reply = wait_until(lambda: refused() or page.evaluate(_OK_REPLY, "#chat-messages"), 90,
                               "neither a reply nor a provider error")
            session.shot("3-first-chat")
            assert session.unexpected_errors() == []
        finally:
            session.close()
        written = json.loads((core.data_dir / "settings.json").read_text(encoding="utf-8"))
        completions = [r for r in core.jsonl("logs/events.jsonl")
                       if r.get("type") == "owner_api_action" and r.get("action") == "onboarding_complete"]
        stack.extra["setup"] = {"bootstrap_before": before["json"]["ready"], "complete_status": done[0],
                                "sent_keys": sorted(done[1]), "completion_events": completions, "probe_after": probe,
                                "bootstrap_after": after["json"]["ready"], "first_chat": reply,
                                "model_calls": len(core.model.calls), "model_call_authorization": authorization}
    assert before["status"] == 200 and before["json"]["ready"] is False
    assert before["json"]["bootstrap"].get("freshInstall") is True
    assert done[0] == 200 and len(completions) == 1, completions
    assert done[1]["OPENAI_COMPATIBLE_BASE_URL"] == core.model.base_url and not done[1].get("OPENAI_COMPATIBLE_API_KEY")
    assert probe == 204 and after["json"]["ready"] is True
    assert written["OPENAI_COMPATIBLE_BASE_URL"] == core.model.base_url
    assert not written.get("OPENAI_COMPATIBLE_API_KEY") and written.get("OUROBOROS_ONBOARDING_COMPLETED_AT")
    assert "OUROBOROS_UI_DIR" not in written
    assert reply is True, f"setup accepted the keyless endpoint; the first chat failed: {reply}"
    assert any(c["main"] for c in core.model.calls), "the first chat reached the keyless provider"
    assert authorization and set(authorization) == {None}, authorization
