"""Tests for the minimal network password gate."""

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

import ouroboros.server_auth as server_auth


async def _ok(_: Request) -> JSONResponse:
    return JSONResponse({"ok": True})


def _make_client(monkeypatch) -> TestClient:
    monkeypatch.setenv(server_auth.NETWORK_PASSWORD_KEY, "secret")
    monkeypatch.setattr(server_auth, "load_settings", lambda: {})
    app = server_auth.NetworkAuthGate(Starlette(routes=[
        Route("/", endpoint=_ok),
        Route("/api/health", endpoint=_ok),
        Route("/api/secret", endpoint=_ok),
    ]))
    return TestClient(app)


def test_validate_network_auth_configuration_allows_open_bind_without_password(monkeypatch):
    monkeypatch.delenv(server_auth.NETWORK_PASSWORD_KEY, raising=False)
    monkeypatch.setattr(server_auth, "load_settings", lambda: {})

    assert server_auth.validate_network_auth_configuration("127.0.0.1") is None
    assert server_auth.validate_network_auth_configuration("0.0.0.0") is None


def test_get_network_auth_startup_warning_warns_but_allows_open_bind(monkeypatch):
    monkeypatch.delenv(server_auth.NETWORK_PASSWORD_KEY, raising=False)
    monkeypatch.setattr(server_auth, "load_settings", lambda: {})

    assert server_auth.get_network_auth_startup_warning("127.0.0.1") is None
    warning = server_auth.get_network_auth_startup_warning("0.0.0.0")
    assert warning is not None
    assert "without OUROBOROS_NETWORK_PASSWORD" in warning

    monkeypatch.setenv(server_auth.NETWORK_PASSWORD_KEY, "secret")
    assert server_auth.get_network_auth_startup_warning("0.0.0.0") is None


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


def test_loopback_login_cookie_unlocks_owner_connections(monkeypatch):
    """The owner-only connections namespace is authenticated even on loopback.

    Settings establishes the signed browser session through the ordinary
    ``/auth/login`` flow — which is why login/logout are handled BEFORE the
    loopback bypass — so the submitted password is never retained in JavaScript
    (RWS v2, D6).
    """

    monkeypatch.setenv(server_auth.NETWORK_PASSWORD_KEY, "secret")
    monkeypatch.setattr(server_auth, "load_settings", lambda: {})
    inner = Starlette(routes=[
        Route("/api/owner/connections", endpoint=_ok),
    ])
    client = TestClient(
        server_auth.NetworkAuthGate(inner),
        client=("127.0.0.1", 12345),
    )

    locked = client.get("/api/owner/connections")
    assert locked.status_code == 401
    assert locked.json()["error_code"] == "owner_auth_required"

    login = client.post(
        "/auth/login",
        json={"password": "secret", "next": "/"},
    )
    assert login.status_code == 200
    assert client.get("/api/owner/connections").json() == {"ok": True}


def test_login_next_url_is_escaped(monkeypatch):
    with _make_client(monkeypatch) as client:
        resp = client.get('/auth/login?next=/"><script>alert(1)</script>', follow_redirects=False)
        assert resp.status_code == 200
        assert "<script>" not in resp.text
        assert 'value="/"' in resp.text
