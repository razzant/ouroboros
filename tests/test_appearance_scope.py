"""The installation's appearance scope on the core's own entry pages (#1111).

``web/theme.js`` keeps the Light / Dark / System choice in one host-only cookie this
scope names, so the choice follows the server to another port of the same host and
stays apart from another installation's (``web/tests/theme.test.js`` owns the browser
side). Pinned here: the scope is a stable digest of the installation's own identity;
``/``, ``/onboarding`` and the network login form name it before ``theme.js``, and only
while the builtin interface serves; the Telegram Mini App and a selected interface
directory keep their origin's store; the cookie never hides the gate's own session; the
real-server browser lanes prove ``/`` byte for byte with exactly the scope their data root names.
"""

from __future__ import annotations

import json
import pathlib
import re
from types import SimpleNamespace

import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from ouroboros import install_identity, server_auth
from ouroboros.gateway import onboarding_host
from ouroboros.server_web import APPEARANCE_SCOPE_META, ServedUI, appearance_scope
from tests import candidate_checkout

REPO = pathlib.Path(__file__).resolve().parents[1]
META = re.compile(r'<meta name="ouroboros-appearance-scope" content="([0-9a-f]{16})">')
OWNER = {"X-Ouroboros-Password": "secret"}


def _mint(root: pathlib.Path, identity: str, recorded_root: pathlib.Path | None = None) -> pathlib.Path:
    """The record ``install_identity.initialize_install`` writes, as a copied root would carry it too."""
    path = root / install_identity.IDENTITY_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"id": identity, "root": str(install_identity.layout_root(recorded_root or root)),
                                "created_at": "2026-10-27T00:00:00Z"}), encoding="utf-8")
    return root


def test_the_scope_is_a_stable_digest_of_the_installations_own_identity(tmp_path):
    one, two = _mint(tmp_path / "one", "a" * 32), _mint(tmp_path / "two", "b" * 32)
    scope = appearance_scope(one)
    assert re.fullmatch(r"[0-9a-f]{16}", scope) and scope == appearance_scope(one) != appearance_scope(two)
    # Not minted yet, a copy still carrying its original's record, a record never readable: no scope.
    assert appearance_scope(tmp_path / "fresh") == appearance_scope(None) == ""
    assert appearance_scope(_mint(tmp_path / "copy", "a" * 32, recorded_root=one)) == ""
    (tmp_path / "broken" / "state").mkdir(parents=True)
    (tmp_path / "broken" / "state" / "install.json").write_text("{", encoding="utf-8")
    assert appearance_scope(tmp_path / "broken") == ""
    # A running installation's record that fails to read again keeps its identity, so a page served
    # then still names the scope its cookie lives under; a removed record is no identity at all.
    (one / install_identity.IDENTITY_PATH).write_text('{"id": ', encoding="utf-8")
    assert appearance_scope(one) == scope
    (one / install_identity.IDENTITY_PATH).unlink()
    assert appearance_scope(one) == ""


def test_theme_js_reads_exactly_the_scope_the_core_names():
    source = (REPO / "web" / "theme.js").read_text(encoding="utf-8")
    assert f"meta[name=\"{APPEARANCE_SCOPE_META}\"]" in source
    assert "/^[0-9a-f]{16}$/" in source and "`ouroboros-theme-${scope}`" in source


@pytest.fixture
def gate(tmp_path, monkeypatch):
    """The real entry pages behind the real gate, as a peer beyond loopback sees them."""
    monkeypatch.setattr("ouroboros.server_process.record_applied_restart_settings", lambda settings: None)
    monkeypatch.setattr(onboarding_host, "_live_setup", lambda: ({}, {}))
    monkeypatch.setattr(server_auth, "get_configured_network_password", lambda: "secret")

    def build(ui_dir: str = "", root: pathlib.Path | None = None) -> TestClient:
        served = ServedUI(REPO)
        app = Starlette(routes=[Route("/", endpoint=served.index_page),
                                Route("/onboarding", endpoint=onboarding_host.onboarding_page), *served.mounts()])
        app.state.drive_root = _mint(tmp_path / "data", "c" * 32) if root is None else root
        served.apply(app, {"OUROBOROS_UI_DIR": ui_dir})
        return TestClient(server_auth.NetworkAuthGate(app, served_ui=served))

    return build


def test_the_builtin_entry_pages_name_the_scope_before_theme_js(gate, tmp_path):
    client = gate()
    scope = appearance_scope(tmp_path / "data")
    pages = {"index": (client.get("/", headers=OWNER).text, '<script src="/static/theme.js">'),
             "onboarding": (client.get("/onboarding", headers=OWNER).text, '<script src="/onboarding/static/theme.js">'),
             "login": (client.get("/", headers={"Accept": "text/html"}).text, "<script>")}
    for name, (page, script) in pages.items():
        assert META.findall(page) == [scope], name
        assert 0 < META.search(page).start() < page.index(script), name
        assert "c" * 32 not in page, name  # a digest: the identity itself never reaches a browser
    # The Telegram Mini App keeps its origin's store: its proxy origin never moves with the port.
    telegram = {**OWNER, "X-Ouroboros-Telegram-MiniApp": "1"}
    assert META.findall(client.get("/", headers=telegram).text) == []
    assert META.findall(client.get("/onboarding", headers=telegram).text) == []


def test_a_selected_directory_or_an_unminted_identity_names_no_scope(gate, tmp_path):
    custom = tmp_path / "custom"
    custom.mkdir()
    (custom / "index.html").write_text("<!doctype html><html><head><title>c</title></head><body>c</body></html>",
                                       encoding="utf-8")
    for client in (gate(str(custom)), gate(root=tmp_path / "fresh")):
        for path in ("/", "/onboarding", "/onboarding?builtin=1"):
            assert META.findall(client.get(path, headers=OWNER).text) == [], path
        # The login form inlines theme.js, which names the meta it looks for: match the meta itself.
        assert META.findall(client.get("/", headers={"Accept": "text/html"}).text) == []


def test_the_browser_harness_proves_the_index_by_exactly_the_scope_its_data_root_names(tmp_path, monkeypatch):
    """``tests/candidate_checkout.assert_served_candidate`` qualifies a real server's ``/`` byte for byte: the
    checkout's index plus the marker and the scope the real ServedUI names for that server's data root. The scope
    is derived, never stripped: another installation's, a missing or an extra head fact, or another byte fails."""
    monkeypatch.setattr("ouroboros.server_process.record_applied_restart_settings", lambda settings: None)

    def served_index(root: pathlib.Path) -> bytes:
        served = ServedUI(REPO)
        app = Starlette(routes=[Route("/", endpoint=served.index_page)])
        app.state.drive_root = root
        served.apply(app, {"OUROBOROS_UI_DIR": ""})
        return TestClient(app).get("/").content

    url, sentinel, version = "http://127.0.0.1:4343", b"candidate sentinel", "9.9.9+candidate.proof"
    replies = {"/api/state": json.dumps({"supervisor_ready": True, "workers_total": 1,
                                         "served_ui": {"source": "builtin"}}).encode(),
               "/onboarding/static/modules/api_client.js": b"api client\n",
               "/onboarding/static/candidate-sentinel.txt": sentinel,
               "/api/health": json.dumps({"runtime_version": version}).encode()}
    monkeypatch.setattr(candidate_checkout, "_fetch", lambda address, timeout=5: replies[address.removeprefix(url)])
    monkeypatch.setattr(candidate_checkout, "verify_checkout", lambda checkout, state: None)
    monkeypatch.setattr("ouroboros.server_process.read_service_bindings",
                        lambda root: {"main": {"pid": 4242, "port": 4343}})
    files = {"web/index.html": ((REPO / "web" / "index.html").read_bytes(),),
             "web/modules/api_client.js": (b"api client\n",)}
    proof = SimpleNamespace(state=SimpleNamespace(files=files), sentinel_bytes=sentinel, version_text=version)

    def qualifies(page: bytes, root: pathlib.Path) -> bool:
        replies["/"] = page
        try:
            candidate_checkout.assert_served_candidate(url, tmp_path, root, 4242, proof)
        except AssertionError as error:
            assert str(error) == "web/index.html"
            return False
        return True

    data, other, fresh = _mint(tmp_path / "data", "d" * 32), _mint(tmp_path / "other", "e" * 32), tmp_path / "fresh"
    page, foreign, unminted = served_index(data), served_index(other), served_index(fresh)
    meta = META.search(page.decode()).group(0).encode()
    assert META.findall(foreign.decode()) == [appearance_scope(other)] != [appearance_scope(data)]
    assert META.findall(unminted.decode()) == []
    assert qualifies(page, data) and qualifies(unminted, fresh)
    for wrong, root in ((foreign, data),  # another installation's scope
                        (unminted, data),  # the scope missing
                        (page, fresh),  # a scope this data root never minted
                        (page.replace(meta, meta + b'<meta name="ouroboros-extra" content="1">'), data),
                        (page[:-1] + bytes([page[-1] ^ 1]), data)):  # the checkout's own bytes changed
        assert not qualifies(wrong, root)


async def _ok(_: Request) -> JSONResponse:
    return JSONResponse({"ok": True})


@pytest.mark.parametrize("order", ["theme first", "session first"])
def test_the_theme_cookie_never_hides_the_gates_session(monkeypatch, order):
    monkeypatch.setattr(server_auth, "get_configured_network_password", lambda: "secret")
    monkeypatch.setattr(server_auth, "_auth_secret_cache", b"\x01" * 32)  # no key file in a data root
    session = f"{server_auth.AUTH_COOKIE_NAME}={server_auth._session_value('secret')}"
    theme = "ouroboros-theme-0123456789abcdef=system"
    cookie = "; ".join([theme, session] if order == "theme first" else [session, theme])
    with TestClient(server_auth.NetworkAuthGate(Starlette(routes=[Route("/api/secret", endpoint=_ok)]))) as client:
        assert client.get("/api/secret", headers={"Cookie": cookie}).json() == {"ok": True}
