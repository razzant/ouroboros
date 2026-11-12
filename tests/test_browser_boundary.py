"""The local API's browser boundary, through the real NetworkAuthGate.

Every refused request must leave the handler unrun (no effect at all); native clients without
browser headers, the core's own pages and passive files keep working. The recording app stands
in for the whole router.
"""

from __future__ import annotations

import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route, WebSocketRoute
from starlette.testclient import TestClient
from starlette.websockets import WebSocket, WebSocketDisconnect
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

import ouroboros.browser_boundary as browser_boundary
import ouroboros.server_auth as server_auth

OWN = "http://testserver"
HANDLED: list[tuple[str, str, bytes]] = []


async def _record(request: Request) -> JSONResponse:
    HANDLED.append((request.method, request.url.path, await request.body()))
    return JSONResponse({"ok": True})


async def _ws(websocket: WebSocket) -> None:
    HANDLED.append(("WS", "/ws", b""))
    await websocket.accept()
    await websocket.send_json({"ok": True})
    await websocket.close()


_METHODS = ["GET", "HEAD", "POST", "PUT", "DELETE", "PATCH"]


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(server_auth, "get_configured_network_password", lambda: "")
    HANDLED.clear()
    app = server_auth.NetworkAuthGate(Starlette(routes=[
        Route("/", endpoint=_record),
        Route("/static/{path:path}", endpoint=_record),
        Route("/onboarding", endpoint=_record),
        Route("/api/{path:path}", endpoint=_record, methods=_METHODS),
        WebSocketRoute("/ws", endpoint=_ws),
    ]))
    with TestClient(app) as test_client:
        yield test_client


def _refused(response):
    assert response.status_code == 403, response.text
    assert response.json()["code"] == "cross_origin_request"
    assert HANDLED == []


@pytest.mark.parametrize("headers", [
    {},                                                         # native: CLI, launcher, sidecar, skills
    {"sec-fetch-site": "none"},                                 # a navigation the owner typed
    {"origin": "HTTP://TESTSERVER:80"},                         # case and the default port are the same origin
])
def test_native_and_same_origin_requests_reach_the_handler(client, headers):
    response = client.post("/api/reset", headers=headers)
    assert response.status_code == 200 and HANDLED == [("POST", "/api/reset", b"")]


@pytest.mark.parametrize(("method", "headers"), [
    pytest.param("POST", {"origin": "http://evil.example", "sec-fetch-site": "cross-site"}, id='POST-headers0'),
    pytest.param("POST", {"origin": "http://testserver:3000", "sec-fetch-site": "same-site"}, id='POST-headers3'),
    pytest.param("POST", {"origin": "null", "sec-fetch-site": "same-origin"}, id='POST-headers6'),
    pytest.param("POST", {"sec-fetch-site": "cross-site"}, id='POST-headers9'),  # a no-cors image or form with no Origin
    pytest.param("GET", {"origin": "http://evil.example"}, id='GET-headers1'),  # no Fetch Metadata
    pytest.param("GET", {"origin": "https://testserver"}, id='GET-headers4'),  # another scheme
    pytest.param("GET", {"origin": "http://testserver/path"}, id='GET-headers7'),  # malformed
    pytest.param("GET", {"sec-fetch-site": "same-site"}, id='GET-headers10'),
    pytest.param("DELETE", {"origin": "http://testserver:3000"}, id='DELETE-headers2'),  # same host, another port
    pytest.param("DELETE", {"origin": "null"}, id='DELETE-headers5'),  # an opaque frame or sandboxed page
    pytest.param("DELETE", {"origin": "chrome-extension://abc"}, id='DELETE-headers8'),
])
def test_other_sites_pages_never_reach_a_handler(client, headers, method):
    _refused(client.request(method, "/api/reset", headers=headers))


@pytest.mark.parametrize(("method", "path"), [
    ("GET", "/"),
    ("GET", "/static/modules/app.js"),
    ("GET", "/api/extensions/demo/ui/rev1/ui/icon.png"),
    ("HEAD", "/onboarding"),
    ("HEAD", "/api/extensions/demo/module/widget.js"),
])
def test_passive_files_load_from_any_page(client, path, method):
    response = client.request(method, path, headers={"origin": "null", "sec-fetch-site": "cross-site"})
    assert response.status_code == 200 and HANDLED and HANDLED[0][1] == path


@pytest.mark.parametrize("method,path", [
    ("POST", "/api/extensions/demo/module/widget.js"),           # only GET/HEAD of extension files is passive
    ("GET", "/api/extensions/demo/ping"),                        # a skill's own route may act on GET
    # Only the static /ui/{revision}/{path} route is passive; a shorter /ui/<x> is a dynamic route, not a file.
    ("GET", "/api/extensions/demo/ui/action"),
])
def test_effects_behind_passive_looking_paths_stay_guarded(client, method, path):
    _refused(client.request(method, path, headers={"origin": "null", "sec-fetch-site": "cross-site"}))


@pytest.mark.parametrize("headers", [{}, {"origin": OWN}, {"origin": "https://ouroboros.example", "sec-fetch-site": "same-origin"}])
def test_native_and_same_origin_websockets_are_accepted(client, headers):
    with client.websocket_connect("/ws", headers=headers) as ws:
        assert ws.receive_json() == {"ok": True}
    assert HANDLED == [("WS", "/ws", b"")]


@pytest.mark.parametrize("headers", [
    {"origin": "http://evil.example"}, {"origin": "null"},
    {"origin": "http://evil.example", "sec-fetch-site": "cross-site"},
])
def test_other_sites_websockets_close_before_accept(client, headers):
    with pytest.raises(WebSocketDisconnect) as closed:
        with client.websocket_connect("/ws", headers=headers):
            pass
    assert closed.value.code == 4403 and HANDLED == []


# A JSON body needs no label (curl's ``-d`` sends a form label); what a route makes of its body is its own
# reading (gateway._helpers.request_json_or), so the boundary hands every body on byte for byte.
@pytest.mark.parametrize("method,path,content_type,body", [
    ("POST", "/api/settings", None, b'{"OUROBOROS_MODEL": "x"}'),
    ("POST", "/api/settings", "application/merge-patch+json; charset=utf-8", b"{}"),
    ("DELETE", "/api/chat/upload", "text/plain", b'{"filename": "probe.txt"}'),
    ("POST", "/api/reset", "text/plain", b""),
])
def test_every_body_reaches_the_handler_unchanged(client, method, path, content_type, body):
    headers = {"origin": OWN} | ({"content-type": content_type} if content_type else {})
    response = client.request(method, path, content=body, headers=headers)
    assert response.status_code == 200 and HANDLED == [(method, path, body)]


def test_an_unknown_length_body_is_handed_on_whole(client):
    def chunks(*parts):
        yield from parts

    response = client.post("/api/settings", content=chunks(b"", b'{"a"', b": 1}"), headers={"origin": OWN})
    assert response.status_code == 200 and HANDLED == [("POST", "/api/settings", b'{"a": 1}')]
    HANDLED.clear()
    response = client.post("/api/tasks/t1/cancel", content=chunks(b""), headers={"origin": OWN})
    assert response.status_code == 200 and HANDLED == [("POST", "/api/tasks/t1/cancel", b"")]


@pytest.mark.parametrize("content_type", [None, "text/plain", "application/json"])
def test_another_sites_body_is_refused_whatever_it_holds(client, content_type):
    """The labels a cross-site form or ``no-cors`` fetch can send, and the one it cannot: provenance decides."""
    headers = {"origin": "http://evil.example", "sec-fetch-site": "cross-site"}
    _refused(client.post("/api/settings", content=b'{"OUROBOROS_MODEL": "x"}',
                         headers=headers | ({"content-type": content_type} if content_type else {})))


# One public origin behind a reverse proxy. Chromium (151 and 157 tested) sends a WebSocket handshake with
# ``Origin`` alone, no Fetch Metadata, so the core's own socket passes only when the browser's Host, port
# included, and its HTTPS scheme reach the core: uvicorn applies X-Forwarded-Proto only from an address in
# FORWARDED_ALLOW_IPS (here the test client, when trusted), and nothing reads X-Forwarded-Host.
PAGE = "https://agent.example"


@pytest.fixture
def refusals(monkeypatch, caplog):
    monkeypatch.setattr(browser_boundary, "_refused_seen", set())
    caplog.set_level("WARNING", logger="ouroboros.browser_boundary")
    return lambda: [r.getMessage() for r in caplog.records if r.name == "ouroboros.browser_boundary"]


def _behind_proxy(monkeypatch, *, trusted: bool) -> TestClient:
    monkeypatch.setattr(server_auth, "get_configured_network_password", lambda: "")
    HANDLED.clear()
    gate = server_auth.NetworkAuthGate(Starlette(routes=[
        Route("/api/{path:path}", endpoint=_record, methods=_METHODS), WebSocketRoute("/ws", endpoint=_ws)]))
    return TestClient(ProxyHeadersMiddleware(gate, trusted_hosts=["testclient"] if trusted else "127.0.0.1"))


@pytest.mark.parametrize("host,origin", [("agent.example", PAGE), ("agent.example:8443", f"{PAGE}:8443")])
def test_a_proxy_passing_the_browsers_host_and_https_admits_the_pages_socket(monkeypatch, host, origin):
    with _behind_proxy(monkeypatch, trusted=True) as client, client.websocket_connect(
            "/ws", headers={"host": host, "x-forwarded-proto": "https", "origin": origin}) as ws:
        assert ws.receive_json() == {"ok": True}
    assert HANDLED == [("WS", "/ws", b"")]


@pytest.mark.parametrize("trusted,headers,origin,expected", [
    # The proxy names its upstream instead of the browser's Host; X-Forwarded-Host is never read.
    (True, {"host": "127.0.0.1:8765", "x-forwarded-host": "agent.example"}, PAGE, "https://127.0.0.1:8765"),
    (True, {"host": "agent.example"}, f"{PAGE}:8443", PAGE),  # the browser's port dropped
    (False, {"host": "agent.example"}, PAGE, "http://agent.example"),  # X-Forwarded-Proto from an untrusted address
])
def test_a_proxy_that_loses_host_or_https_gets_the_socket_refused_and_logged(
        monkeypatch, refusals, trusted, headers, origin, expected):
    with _behind_proxy(monkeypatch, trusted=trusted) as client:
        with pytest.raises(WebSocketDisconnect) as closed, client.websocket_connect(
                "/ws", headers={**headers, "x-forwarded-proto": "https", "origin": origin}):
            pass
        assert closed.value.code == 4403 and HANDLED == []
        # The page's own fetches still pass by the browser's Sec-Fetch-Site: only the live socket fails.
        response = client.post("/api/reset", headers={**headers, "x-forwarded-proto": "https", "origin": origin,
                                                      "sec-fetch-site": "same-origin"})
        assert response.status_code == 200 and HANDLED == [("POST", "/api/reset", b"")]
    assert refusals() == [f"browser_boundary refused WEBSOCKET /ws origin={origin} expected={expected} sec-fetch-site=-"]


def test_a_refusal_logs_sanitized_origins_once_per_expected_origin(client, refusals):
    secret = "https://user:hunter2@evil.example/next?token=t0ps3cret"
    for _ in range(2):
        _refused(client.post("/api/reset?password=hunter2", headers={"origin": secret, "sec-fetch-site": "cross-site"}))
    _refused(client.post("/api/reset", headers={"origin": "HTTPS://EVIL.EXAMPLE:443"}))
    # The same page against a corrected Host is a new fact; a Host with credentials is never written.
    _refused(client.post("/api/reset", headers={"origin": "HTTPS://EVIL.EXAMPLE:443", "host": "core.example"}))
    _refused(client.post("/api/reset", headers={"origin": "http://evil.example", "host": "user:hunter2@core"}))
    # A decoded path is written as a bounded URL path: no line breaks, no unbounded text.
    _refused(client.post("/api/x%0Aforged" + "a" * 300, headers={"origin": "http://evil.example"}))
    lines = refusals()
    assert lines[:4] == [
        "browser_boundary refused POST /api/reset origin=malformed expected=http://testserver sec-fetch-site=cross-site",
        "browser_boundary refused POST /api/reset origin=https://evil.example expected=http://testserver sec-fetch-site=-",
        "browser_boundary refused POST /api/reset origin=https://evil.example expected=http://core.example sec-fetch-site=-",
        "browser_boundary refused POST /api/reset origin=http://evil.example expected=malformed sec-fetch-site=-",
    ]
    path = lines[4].split(" ")[3]
    assert len(lines) == 5 and "\n" not in lines[4]
    assert path.startswith("/api/x%0Aforgedaaa") and path.endswith("…") and len(path) == 201
    assert not any(word in line for line in lines for word in ("hunter2", "token", "t0ps3cret", "user"))
