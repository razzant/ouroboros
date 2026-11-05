"""Cyber's explicit password choice reaches the existing network-auth path."""

import json

import pytest
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route, WebSocketRoute
from starlette.testclient import TestClient

from ouroboros import config, server_auth
from ouroboros.gateway import settings
from tests.test_owner_settings_write_seam import _settings_app, isolated_settings  # noqa: F401


@pytest.fixture(params=["pro", "cyber_pro"])
def access(request, isolated_settings, monkeypatch):  # noqa: F811
    for key in config.SETTINGS_DEFAULTS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("OUROBOROS_TRUST_NONLOCAL_BIND_WITHOUT_PASSWORD", raising=False)
    config.initialize_runtime_mode_baseline(request.param)
    yield request.param


def _post(monkeypatch, path, initial, payload, *, bound_host):
    # An install that already carries the consciousness level: loading it runs no one-time copy.
    initial.setdefault("OUROBOROS_CONSCIOUSNESS_SAFETY_MODE", "full")
    path.write_text(json.dumps(initial))
    config.apply_settings_to_env(config.load_settings())
    app = _settings_app(monkeypatch, path)
    app.state.bind_host = bound_host
    monkeypatch.setattr(settings, "_apply_settings_to_env", config.apply_settings_to_env)
    monkeypatch.setattr(server_auth, "load_settings", config.load_settings)
    return TestClient(app).post("/api/settings", json=payload)


def _assert_open_http_and_websocket():
    async def http(request):
        return JSONResponse({"reached": "app"})

    async def websocket(socket):
        await socket.accept()
        await socket.send_text("reached app")
        await socket.close()

    app = server_auth.NetworkAuthGate(Starlette(routes=[
        Route("/api/ordinary", http), WebSocketRoute("/ws", websocket),
    ]))
    # In-memory ASGI transport; no live listener or authentication operation.
    with TestClient(app, client=("192.0.2.25", 32100)) as client:
        assert client.get("/api/ordinary").json() == {"reached": "app"}
        with client.websocket_connect("/ws") as socket:
            assert socket.receive_text() == "reached app"


@pytest.mark.parametrize("scenario", ["enable_network", "clear_password"])
def test_explicit_empty_password_save_uses_existing_auth_path(isolated_settings, access, monkeypatch, scenario):  # noqa: F811
    initial = {"OUROBOROS_RUNTIME_MODE": access, "OUROBOROS_SERVER_HOST": "127.0.0.1",
               "OUROBOROS_NETWORK_PASSWORD": "", "OPENAI_API_KEY": "synthetic-provider-marker"}
    if scenario == "clear_password":
        initial.update(OUROBOROS_SERVER_HOST="0.0.0.0", OUROBOROS_NETWORK_PASSWORD="synthetic-password")
    payload = {"OUROBOROS_SERVER_HOST": "0.0.0.0" if scenario == "enable_network" else "127.0.0.1",
               "OUROBOROS_NETWORK_PASSWORD": ""}
    response = _post(monkeypatch, isolated_settings, initial, payload, bound_host=initial["OUROBOROS_SERVER_HOST"])
    if access == "pro":
        assert response.status_code == 400 and response.json()["saved"] is False
        assert json.loads(isolated_settings.read_text()) == initial
        return
    assert response.status_code == 200, response.text
    saved = json.loads(isolated_settings.read_text())
    assert saved["OUROBOROS_NETWORK_PASSWORD"] == ""
    assert saved["OPENAI_API_KEY"] == initial["OPENAI_API_KEY"]
    assert server_auth.get_configured_network_password() == ""
    assert server_auth.validate_network_auth_configuration("0.0.0.0") is None
    assert server_auth.get_network_auth_startup_warning("0.0.0.0")
    from ouroboros.server_runtime import has_startup_ready_provider
    assert has_startup_ready_provider(saved)
    _assert_open_http_and_websocket()


def test_specific_lan_bind_retains_supported_form_error(isolated_settings, access, monkeypatch):  # noqa: F811
    initial = {"OUROBOROS_RUNTIME_MODE": access, "OUROBOROS_SERVER_HOST": "127.0.0.1"}
    response = _post(monkeypatch, isolated_settings, initial,
        {"OUROBOROS_SERVER_HOST": "192.0.2.10", "OUROBOROS_NETWORK_PASSWORD": ""}, bound_host="127.0.0.1")
    assert response.status_code == 400 and response.json()["saved"] is False
    assert "Specific LAN IP" in response.json()["error"]
    assert json.loads(isolated_settings.read_text()) == initial


def test_cyber_does_not_clear_a_configured_password_incidentally(isolated_settings, access, monkeypatch):  # noqa: F811
    initial = {"OUROBOROS_RUNTIME_MODE": access, "OUROBOROS_SERVER_HOST": "0.0.0.0",
               "OUROBOROS_NETWORK_PASSWORD": "synthetic-password"}
    response = _post(monkeypatch, isolated_settings, initial, {"TOTAL_BUDGET": 25}, bound_host="0.0.0.0")
    assert response.status_code == 200
    assert server_auth.get_configured_network_password() == "synthetic-password"


@pytest.mark.parametrize("host,password,trusted,warning", [
    ("", "", False, ""),
    ("127.0.0.1", "", False, ""),
    ("0.0.0.0", "synthetic-password", False, ""),
    ("0.0.0.0", "", False,
     "Server Bind Host is non-localhost and Network Password is empty; "
     "after restart the app will be reachable on the network without a password."),
    ("0.0.0.0", "", True,
     "OUROBOROS_TRUST_NONLOCAL_BIND_WITHOUT_PASSWORD=1 allows this "
     "non-localhost bind without Ouroboros's internal Network Password. "
     "Use only behind ingress auth, VPN, private networking, or an auth proxy."),
])
def test_post_save_exposure_notice_is_unchanged(monkeypatch, host, password, trusted, warning):
    current = {"OUROBOROS_SERVER_HOST": host, "OUROBOROS_NETWORK_PASSWORD": password}
    before = dict(current)
    monkeypatch.setattr(settings, "_trust_nonlocal_bind_without_password_enabled", lambda: trusted)
    assert settings._network_exposure_warning(current) == warning
    assert current == before


def test_post_save_notice_failure_does_not_reclassify_saved_settings(monkeypatch):
    def unavailable(host):
        raise OSError("synthetic notice lookup failure")
    monkeypatch.setattr(server_auth, "is_loopback_host", unavailable)
    assert settings._network_exposure_warning({"OUROBOROS_SERVER_HOST": "0.0.0.0"}) == ""
