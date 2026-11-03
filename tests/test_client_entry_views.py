"""Selected complete entry pages share the applied UI owner and keep core authority."""

from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from ouroboros import server_auth, server_process
from ouroboros.gateway import onboarding_host
from ouroboros.server_web import ServedUI

REPO = Path(__file__).resolve().parents[1]


async def _private(_request):
    return JSONResponse({"private": True})


@pytest.fixture
def entry_app(tmp_path, monkeypatch):
    monkeypatch.setattr(server_process, "_applied_restart_settings", {})
    monkeypatch.setattr(server_auth, "get_configured_network_password", lambda: "secret")
    monkeypatch.setattr(server_auth, "_auth_secret_cache", b"test-selected-login")
    monkeypatch.setattr(onboarding_host, "_live_setup", lambda: ({}, {}))
    selected = tmp_path / "selected"
    (selected / "auth").mkdir(parents=True)
    for name, text in {
        "index.html": "private main", "app.js": "private main script",
        "onboarding.html": "<!doctype html><title>Selected setup</title>\r\n<script src='/static/app.js'></script>",
        "auth/index.html": "<!doctype html><title>Selected login</title><script src='./login.js'></script>",
        "auth/login.js": "public login script", "auth/login": "reserved asset", "auth/logout": "reserved asset",
    }.items():
        (selected / name).write_bytes(text.encode("utf-8"))
    served = ServedUI(REPO)
    app = Starlette(routes=[
        Route("/", endpoint=served.index_page), Route("/api/health", endpoint=_private),
        Route("/api/secret", endpoint=_private), Route("/api/onboarding/bootstrap", endpoint=_private),
        Route("/api/extensions/test/value", endpoint=_private),
        Route("/onboarding", endpoint=onboarding_host.onboarding_page), *served.mounts(),
    ])
    app.state.drive_root = tmp_path / "data"
    served.apply(app, {"OUROBOROS_UI_DIR": str(selected)})
    return app, selected, served


def test_setup_selects_whole_page_and_builtin_recovery_keeps_its_graph(entry_app):
    app, selected, served = entry_app
    with TestClient(app) as client:
        page = client.get("/onboarding")
        assert page.content == (selected / "onboarding.html").read_bytes()
        assert page.headers["cache-control"] == "no-store"
        assert client.get("/static/app.js").text == "private main script"
        recovered = client.get("/onboarding?builtin=1")
        assert recovered.content != page.content and "/onboarding/static/" in recovered.text
        assert client.get("/onboarding/static/ui.css").content == (REPO / "web/ui.css").read_bytes()
        (selected / "onboarding.html").unlink()
        assert client.get("/onboarding").content == recovered.content
    assert served.selection.source == "directory"


def test_setup_without_selection_owner_remains_builtin(monkeypatch):
    monkeypatch.setattr(onboarding_host, "_live_setup", lambda: ({}, {}))
    with TestClient(Starlette(routes=[Route("/onboarding", endpoint=onboarding_host.onboarding_page)])) as client:
        response = client.get("/onboarding")
    assert response.status_code == 200 and "/onboarding/static/" in response.text


def test_optional_entrypoints_do_not_change_the_builtin_source(tmp_path):
    web = tmp_path / "web"
    (web / "auth").mkdir(parents=True)
    for name in ("index.html", "onboarding.html", "auth/index.html"):
        (web / name).write_text("not a selected directory", encoding="utf-8")
    served = ServedUI(tmp_path)
    assert served.custom_page("onboarding.html") is None
    assert served.custom_page("auth/index.html") is None
    with TestClient(Starlette(routes=served.mounts())) as client:
        assert client.get("/auth/index.html").status_code == 404


@pytest.mark.parametrize("relative_path", ["onboarding.html", "auth/index.html"])
def test_unreadable_optional_page_falls_back_with_diagnostic(entry_app, monkeypatch, caplog, relative_path):
    app, selected, served = entry_app
    path = selected / relative_path
    assert served.custom_page(relative_path) is not None
    original = Path.read_bytes

    def unavailable(candidate):
        if candidate == path:
            raise PermissionError("fixture page unreadable")
        return original(candidate)

    monkeypatch.setattr(Path, "read_bytes", unavailable)
    with TestClient(server_auth.NetworkAuthGate(app, served_ui=served)) as client:
        route = "/onboarding" if relative_path == "onboarding.html" else "/auth/login"
        response = client.get(route, headers={"X-Ouroboros-Password": "secret"})
    assert response.status_code == 200
    assert "Selected setup" not in response.text and "Selected login" not in response.text
    assert "using builtin" in caplog.text and "fixture page unreadable" in caplog.text


def test_optional_views_and_assets_use_only_the_applied_directory(entry_app):
    app, selected, served = entry_app
    settings = {"OUROBOROS_UI_DIR": str(selected)}
    served.apply(app, settings)
    with TestClient(app) as client:
        settings["OUROBOROS_UI_DIR"] = "none"  # saving is not applying another generation
        assert "Selected setup" in served.custom_page("onboarding.html")
        assert client.get("/auth/login.js").text == "public login script"
        for mode in ("none", "", str(selected / "missing")):
            served.apply(app, {"OUROBOROS_UI_DIR": mode})
            assert served.custom_page("onboarding.html") is None
            assert client.get("/auth/login.js").status_code == 404
        for path in (selected / "auth").iterdir():
            path.unlink()
        (selected / "auth").rmdir()
        served.apply(app, {"OUROBOROS_UI_DIR": str(selected)})
        assert client.get("/auth/login.js").status_code == 404
        (selected / "auth").mkdir()
        (selected / "auth/login.js").write_text("added asset", encoding="utf-8")
        assert client.get("/auth/login.js").text == "added asset"


def test_only_selected_login_assets_are_public_under_the_network_gate(entry_app):
    app, selected, served = entry_app
    with TestClient(server_auth.NetworkAuthGate(app, served_ui=served)) as client:
        entry = client.get("/auth/login?next=/private")
        assert entry.status_code == 200 and entry.content == (selected / "auth/index.html").read_bytes()
        assert entry.headers["cache-control"] == "no-store"
        assert client.get("/auth/login.js").text == "public login script"
        assert client.head("/auth/login.js").status_code == 200
        assert client.post("/auth/login.js").status_code == 401
        assert client.get("/auth/missing.js").status_code == 404
        for path in ("/static/app.js", "/api/secret", "/api/onboarding/bootstrap",
                     "/api/extensions/test/value", "/authentic/login.js"):
            response = client.get(path, headers={"accept": "application/json"}, follow_redirects=False)
            assert response.status_code == 401, (path, response.status_code)
        assert client.get("/auth/%2e%2e/app.js").status_code == 404
        assert client.get("/api/health").status_code == 200


def test_custom_login_preserves_destination_and_json_and_form_authority(entry_app):
    app, _selected, served = entry_app
    with TestClient(server_auth.NetworkAuthGate(app, served_ui=served)) as client:
        protected = client.get("/?room=one", follow_redirects=False)
        assert protected.status_code == 303
        redirect = urlsplit(protected.headers["location"])
        assert redirect.path == "/auth/login" and parse_qs(redirect.query) == {"next": ["/?room=one"]}
        wrong = client.post("/auth/login", json={"password": "wrong", "next": "/"})
        assert wrong.status_code == 401 and wrong.json() == {"error": "Invalid password."}
        assert server_auth.AUTH_COOKIE_NAME not in client.cookies
        wrong_form = client.post("/auth/login", data={"password": "wrong", "next": "/"})
        assert wrong_form.status_code == 401 and "Invalid password." in wrong_form.text
        assert "Selected login" not in wrong_form.text
        success = client.post("/auth/login", json={"password": "secret", "next": "/\\evil.example"})
        assert success.status_code == 200 and success.json() == {"ok": True, "next": "/"}
        cookie = success.headers["set-cookie"].lower()
        assert "httponly" in cookie and "samesite=lax" in cookie
        assert client.get("/static/app.js").text == "private main script"
        assert "Selected setup" in client.get("/onboarding").text
        form = client.post("/auth/login", data={"password": "secret", "next": "/?room=one"}, follow_redirects=False)
        assert form.status_code == 303 and form.headers["location"] == "/?room=one"


def test_builtin_setup_recovery_stays_builtin_through_authentication(entry_app):
    app, selected, served = entry_app
    with TestClient(server_auth.NetworkAuthGate(app, served_ui=served)) as client:
        direct = client.get("/auth/login?builtin=1&next=/onboarding?builtin=1")
        assert "Enter the network password" in direct.text and "Selected login" not in direct.text
        recovery = client.get("/onboarding?builtin=1", headers={"accept": "text/html"}, follow_redirects=False)
        assert recovery.status_code == 401 and "Enter the network password" in recovery.text
        assert 'value="/onboarding?builtin=1"' in recovery.text and "Selected login" not in recovery.text
        signed_in = client.post("/auth/login", data={"password": "secret", "next": "/onboarding?builtin=1"})
        assert signed_in.status_code == 200 and "/onboarding/static/" in signed_in.text
        assert "Selected setup" not in signed_in.text
        client.cookies.clear()
        (selected / "auth/index.html").unlink()
        assert "Enter the network password" in client.get("/auth/login").text
        assert client.get("/", follow_redirects=False).status_code == 401


@pytest.mark.parametrize("inactive", ["no_password", "loopback"])
def test_public_auth_assets_do_not_activate_login_operations(entry_app, monkeypatch, inactive):
    app, _selected, served = entry_app
    if inactive == "no_password":
        monkeypatch.setattr(server_auth, "get_configured_network_password", lambda: "")
    else:
        monkeypatch.setattr(server_auth, "_scope_client_host", lambda _scope: "127.0.0.1")
    with TestClient(server_auth.NetworkAuthGate(app, served_ui=served)) as client:
        assert client.get("/auth/login.js").text == "public login script"
        assert client.post("/auth/login.js").status_code == 405
        for path in ("/auth/login", "/auth/logout"):
            assert client.get(path).status_code == 404
            assert client.post(path, json={"password": "secret"}).status_code == 404
        assert client.get("/static/app.js").status_code == 200
