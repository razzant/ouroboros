"""Cyber's explicit password choice reaches the existing network-auth path."""

import json
import pathlib

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


@pytest.mark.parametrize("saved_host,launch_host,password", [
    ("192.0.2.10", None, "synthetic-password"),
    ("192.0.2.10", None, ""),
    (None, "192.0.2.10", "synthetic-password"),
    ("", "192.0.2.10", "synthetic-password"),
    ("127.0.0.1", "192.0.2.10", "synthetic-password"),
])
def test_unrelated_secret_save_preserves_the_source_of_a_manual_bind(
        isolated_settings, access, monkeypatch, saved_host, launch_host, password):  # noqa: F811
    host_key = "OUROBOROS_SERVER_HOST"
    initial = {"OUROBOROS_RUNTIME_MODE": access, "OUROBOROS_NETWORK_PASSWORD": password}
    if saved_host is not None:
        initial[host_key] = saved_host
    if launch_host:
        monkeypatch.setenv(host_key, launch_host)
    response = _post(monkeypatch, isolated_settings, initial, {"EXAMPLE_API_KEY": "synthetic-key"},
                     bound_host=launch_host or saved_host)
    assert response.status_code == 200, response.text
    saved = json.loads(isolated_settings.read_text())
    assert saved["EXAMPLE_API_KEY"] == "synthetic-key" and saved["OUROBOROS_NETWORK_PASSWORD"] == password
    assert (host_key in saved, saved.get(host_key)) == (host_key in initial, saved_host)


@pytest.mark.parametrize("access", ["pro"], indirect=True)
@pytest.mark.parametrize("refusals", [1, None])
def test_a_document_the_save_cannot_read_never_loses_its_saved_bind(
        isolated_settings, access, monkeypatch, refusals):  # noqa: F811
    """The save's own read of the document fails after its digest. Once: it is read again behind the writer's lock,
    and the saved bind and password stay. For good: the save is refused and the bytes stay. Neither rebuilds them
    from an empty document, which only an absent one is (the first save's, tests/test_owner_settings_write_seam.py)."""
    host_key, password_key = "OUROBOROS_SERVER_HOST", "OUROBOROS_NETWORK_PASSWORD"
    initial = {"OUROBOROS_RUNTIME_MODE": access, host_key: "0.0.0.0", password_key: "synthetic-password"}
    real_view, real_open, left = settings.settings_read_view, pathlib.Path.open, []

    def refused(path, *args, **kwargs):  # injected: chmod is not a refusal on every platform
        if path == isolated_settings and left and left[0]:
            left[0] -= 1
            raise PermissionError(13, "injected: settings unreadable for this read")
        return real_open(path, *args, **kwargs)

    def document(*args, **kwargs):  # only the save's document read meets the refusal
        left[:] = [refusals or float("inf")]
        try:
            return real_view(*args, **kwargs)
        finally:
            left.clear()

    monkeypatch.setattr(pathlib.Path, "open", refused)
    monkeypatch.setattr(settings, "settings_read_view", document)
    response = _post(monkeypatch, isolated_settings, initial, {"TOTAL_BUDGET": "25"}, bound_host="0.0.0.0")
    monkeypatch.setattr(pathlib.Path, "open", real_open)
    saved = json.loads(isolated_settings.read_text())
    if refusals:
        assert response.status_code == 200, response.text
        assert (saved[host_key], saved[password_key], saved["TOTAL_BUDGET"]) == ("0.0.0.0", "synthetic-password", 25.0)
    else:
        assert response.status_code == 409, response.text
        assert response.json()["code"] == "settings_unreadable" and response.json()["saved"] is False
        assert saved == initial


@pytest.mark.parametrize("access,host,password,refusal", [
    ("pro", "192.0.2.10", "synthetic-password", "Specific LAN IP"),
    ("cyber_pro", "192.0.2.10", "synthetic-password", "Specific LAN IP"),
    ("pro", "0.0.0.0", "", "requires a Network Password"),
], indirect=["access"])
def test_form_rules_apply_when_persisting_a_launch_only_bind(
        isolated_settings, access, monkeypatch, host, password, refusal):  # noqa: F811
    monkeypatch.setenv("OUROBOROS_SERVER_HOST", host)
    initial = {"OUROBOROS_RUNTIME_MODE": access, "OUROBOROS_NETWORK_PASSWORD": password}
    response = _post(monkeypatch, isolated_settings, initial, {"OUROBOROS_SERVER_HOST": host}, bound_host=host)
    assert response.status_code == 400 and refusal in response.json()["error"]
    assert json.loads(isolated_settings.read_text()) == initial


@pytest.mark.parametrize("access", ["pro"], indirect=True)
@pytest.mark.parametrize("saved_password,form_password,expected_password", [
    (None, None, None),
    ("", "***set***", None),
    ("synthetic-saved", None, "synthetic-saved"),
    (None, "synthetic-new", "synthetic-new"),
])
def test_new_saved_network_bind_requires_a_durable_password(
        isolated_settings, access, monkeypatch, saved_password, form_password, expected_password):  # noqa: F811
    host_key, password_key = "OUROBOROS_SERVER_HOST", "OUROBOROS_NETWORK_PASSWORD"
    monkeypatch.setenv(password_key, "synthetic-launch-only")
    initial = {"OUROBOROS_RUNTIME_MODE": access, host_key: "127.0.0.1"}
    if saved_password is not None:
        initial[password_key] = saved_password
    payload = {host_key: "0.0.0.0"}
    if form_password is not None:
        payload[password_key] = form_password
    response = _post(monkeypatch, isolated_settings, initial, payload, bound_host="127.0.0.1")
    saved = json.loads(isolated_settings.read_text())
    if expected_password is None:
        assert response.status_code == 400 and response.json()["saved"] is False
        assert "requires a Network Password" in response.json()["error"]
        assert "synthetic-launch-only" not in response.text and saved == initial
    else:
        assert response.status_code == 200
        assert saved[host_key] == payload[host_key] and saved[password_key] == expected_password


@pytest.mark.parametrize("access", ["pro"], indirect=True)
def test_unsetting_a_bind_keeps_the_launch_network_fields_out_of_the_document(isolated_settings, access, monkeypatch):  # noqa: F811
    host_key, password_key = "OUROBOROS_SERVER_HOST", "OUROBOROS_NETWORK_PASSWORD"
    monkeypatch.setenv(host_key, "0.0.0.0")
    monkeypatch.setenv(password_key, "synthetic-launch-only")
    initial = {"OUROBOROS_RUNTIME_MODE": access, host_key: "127.0.0.1"}
    response = _post(monkeypatch, isolated_settings, initial, {"unset": [host_key]}, bound_host="127.0.0.1")
    assert response.status_code == 200
    saved = json.loads(isolated_settings.read_text())
    assert host_key not in saved and password_key not in saved


@pytest.mark.parametrize("access", ["pro"], indirect=True)
@pytest.mark.parametrize("password", [None, ""])
def test_undoing_a_saved_bind_writes_no_address_and_keeps_the_password_rule(
        isolated_settings, access, monkeypatch, password):  # noqa: F811
    host_key, password_key = "OUROBOROS_SERVER_HOST", "OUROBOROS_NETWORK_PASSWORD"
    # The operator launched on a specific LAN address (the environment is the bind host's authority); Settings saved
    # a wildcard over it. Undo writes no address: the launch one is in effect again, and the address rule is not its.
    monkeypatch.setenv(host_key, "192.0.2.10")
    initial = {"OUROBOROS_RUNTIME_MODE": access, host_key: "0.0.0.0", password_key: "synthetic-password"}
    payload = {"unset": [host_key]} if password is None else {"unset": [host_key], password_key: password}
    response = _post(monkeypatch, isolated_settings, initial, payload, bound_host="192.0.2.10")
    saved = json.loads(isolated_settings.read_text())
    if password is None:
        assert response.status_code == 200, response.text
        assert host_key not in saved and saved[password_key] == "synthetic-password"
    else:  # the same save clears the password: that address would listen without one, so the save is refused
        assert response.status_code == 400 and "requires a Network Password" in response.json()["error"]
        assert saved == initial


@pytest.mark.parametrize("access", ["pro"], indirect=True)
def test_password_clear_is_refused_for_a_saved_manual_bind_before_restart(isolated_settings, access, monkeypatch):  # noqa: F811
    initial = {"OUROBOROS_RUNTIME_MODE": access, "OUROBOROS_SERVER_HOST": "192.0.2.10",
               "OUROBOROS_NETWORK_PASSWORD": "synthetic-password"}
    response = _post(monkeypatch, isolated_settings, initial, {"OUROBOROS_NETWORK_PASSWORD": ""}, bound_host="127.0.0.1")
    assert response.status_code == 400 and "requires a Network Password" in response.json()["error"]
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
