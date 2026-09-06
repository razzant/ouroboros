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

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

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
        assert "<script>" not in resp.text
        assert 'value="/"' in resp.text
