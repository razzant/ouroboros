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
    {"origin": OWN},                                            # the core's page, no Fetch Metadata (older WebKit)
    {"origin": OWN, "sec-fetch-site": "same-origin"},
    {"sec-fetch-site": "none"},                                 # a navigation the owner typed
    # Behind a proxy that terminates TLS or rewrites Host, the browser's own verdict decides.
    {"origin": "https://ouroboros.example", "sec-fetch-site": "same-origin"},
    {"origin": "HTTP://TESTSERVER:80"},                         # case and the default port are the same origin
])
def test_native_and_same_origin_requests_reach_the_handler(client, headers):
    response = client.post("/api/reset", headers=headers)
    assert response.status_code == 200 and HANDLED == [("POST", "/api/reset", b"")]


@pytest.mark.parametrize("headers", [
    {"origin": "http://evil.example", "sec-fetch-site": "cross-site"},
    {"origin": "http://evil.example"},                          # no Fetch Metadata
    {"origin": "http://testserver:3000"},                       # same host, another port
    {"origin": "http://testserver:3000", "sec-fetch-site": "same-site"},
    {"origin": "https://testserver"},                           # another scheme
    {"origin": "null"},                                         # an opaque frame or sandboxed page
    {"origin": "null", "sec-fetch-site": "same-origin"},
    {"origin": "http://testserver/path"},                       # malformed
    {"origin": "chrome-extension://abc"},
    {"sec-fetch-site": "cross-site"},                           # a no-cors image or form with no Origin
    {"sec-fetch-site": "same-site"},
])
@pytest.mark.parametrize("method", ["POST", "GET", "DELETE"])
def test_other_sites_pages_never_reach_a_handler(client, headers, method):
    _refused(client.request(method, "/api/reset", headers=headers))


@pytest.mark.parametrize("path", [
    "/", "/onboarding", "/static/modules/app.js",
    "/api/extensions/demo/module/widget.js", "/api/extensions/demo/ui/rev1/ui/icon.png",
])
@pytest.mark.parametrize("method", ["GET", "HEAD"])
def test_passive_files_load_from_any_page(client, path, method):
    response = client.request(method, path, headers={"origin": "null", "sec-fetch-site": "cross-site"})
    assert response.status_code == 200 and HANDLED and HANDLED[0][1] == path


@pytest.mark.parametrize("method,path", [
    ("POST", "/api/extensions/demo/module/widget.js"),           # only GET/HEAD of extension files is passive
    ("GET", "/api/extensions/demo/ping"),                        # a skill's own route may act on GET
    # Only the static /ui/{revision}/{path} route is passive; a shorter /ui/<x> is a dynamic route, not a file.
    ("GET", "/api/extensions/demo/ui/action"),
    ("GET", "/api/extensions/demo/ui"),
    ("GET", "/api/model-catalog"),
    ("POST", "/"),
    ("GET", "/auth/logout"),
])
def test_effects_behind_passive_looking_paths_stay_guarded(client, method, path):
    _refused(client.request(method, path, headers={"origin": "null", "sec-fetch-site": "cross-site"}))


@pytest.mark.parametrize("headers", [{}, {"origin": OWN}, {"origin": "https://ouroboros.example", "sec-fetch-site": "same-origin"}])
def test_native_and_same_origin_websockets_are_accepted(client, headers):
    with client.websocket_connect("/ws", headers=headers) as ws:
        assert ws.receive_json() == {"ok": True}
    assert HANDLED == [("WS", "/ws", b"")]


@pytest.mark.parametrize("headers", [
    {"origin": "http://evil.example"}, {"origin": "http://testserver:3000"}, {"origin": "null"},
    {"origin": "http://evil.example", "sec-fetch-site": "cross-site"},
])
def test_other_sites_websockets_close_before_accept(client, headers):
    with pytest.raises(WebSocketDisconnect) as closed:
        with client.websocket_connect("/ws", headers=headers):
            pass
    assert closed.value.code == 4403 and HANDLED == []


def _json_refused(response):
    assert response.status_code == 415, response.text
    assert response.json()["code"] == "json_content_type_required"
    assert HANDLED == []


@pytest.mark.parametrize("content_type", ["text/plain", "text/plain;charset=UTF-8",
                                          "application/x-www-form-urlencoded", "multipart/form-data; boundary=x", None])
def test_a_json_contract_refuses_a_body_of_another_type(client, content_type):
    headers = {"origin": OWN} | ({"content-type": content_type} if content_type else {})
    _json_refused(client.post("/api/settings", content=b'{"OUROBOROS_MODEL": "x"}', headers=headers))


def test_an_unknown_length_body_is_decided_by_its_first_chunk_and_replayed(client):
    def chunks(*parts):
        yield from parts

    _json_refused(client.post("/api/settings", content=chunks(b"", b'{"a": 1}'), headers={"origin": OWN}))
    response = client.post("/api/tasks/t1/cancel", content=chunks(b""), headers={"origin": OWN})
    assert response.status_code == 200 and HANDLED == [("POST", "/api/tasks/t1/cancel", b"")]
    HANDLED.clear()
    response = client.post("/api/settings", content=chunks(b'{"a"', b": 1}"),
                           headers={"origin": OWN, "content-type": "application/json"})
    assert response.status_code == 200 and HANDLED == [("POST", "/api/settings", b'{"a": 1}')]


def test_the_method_is_part_of_the_json_contract(client):
    """POST /api/chat/upload is a multipart upload; DELETE at the same path takes JSON (endpoint_index)."""
    response = client.post("/api/chat/upload", content=b"--x--\r\n",
                           headers={"origin": OWN, "content-type": "multipart/form-data; boundary=x"})
    assert response.status_code == 200 and HANDLED == [("POST", "/api/chat/upload", b"--x--\r\n")]
    HANDLED.clear()
    _json_refused(client.request("DELETE", "/api/chat/upload", content=b'{"filename": "probe.txt"}',
                                 headers={"origin": OWN, "content-type": "text/plain"}))
    response = client.request("DELETE", "/api/chat/upload", content=b'{"filename": "probe.txt"}',
                              headers={"origin": OWN, "content-type": "application/json"})
    assert response.status_code == 200 and HANDLED == [("DELETE", "/api/chat/upload", b'{"filename": "probe.txt"}')]


@pytest.mark.parametrize("path,content_type,body", [
    ("/api/settings", "application/json", b"{}"),
    ("/api/settings", "application/merge-patch+json; charset=utf-8", b"{}"),
    ("/api/reset", "text/plain", b""),                           # a body-less POST needs no type
    ("/api/chat/upload", "multipart/form-data; boundary=x", b"--x--\r\n"),
    ("/api/files/upload", "multipart/form-data; boundary=x", b"--x--\r\n"),
    ("/api/extensions/demo/transcribe", "audio/webm", b"\x1a\x45"),  # a skill route defines its own body
])
def test_json_and_declared_non_json_bodies_pass(client, path, content_type, body):
    response = client.post(path, content=body, headers={"origin": OWN, "content-type": content_type})
    assert response.status_code == 200 and HANDLED == [("POST", path, body)]
