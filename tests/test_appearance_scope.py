"""The installation's appearance scope and the data profile's saved-preview scope on the core's own pages (#1111, #3294).

``web/theme.js`` keeps the Light / Dark / System choice in one host-only cookie this
scope names, so the choice follows the server to another port of the same host and
stays apart from another installation's (``web/tests/theme.test.js`` owns the browser
side). Pinned here: the scope is a stable digest of the installation's own identity;
``/``, ``/onboarding`` and the network login form name it before ``theme.js``, and only
while the builtin interface serves; the Telegram Mini App and a selected interface
directory keep their origin's store; the cookie never hides the gate's own session; the
real-server browser lanes prove ``/`` byte for byte with exactly the scope their data root names.

The builtin index also names the data profile its saved chat preview belongs to
(``web/modules/chat_preview.js``): the ``initialization_id`` a current state and its completed
witness both prove, as a digest. The installation scope is no profile proof: an owner Reset keeps
``install.json`` but mints a new identity, and the same rooms must open on their canonical history.
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
from ouroboros.server_web import (
    APPEARANCE_SCOPE_META, CHAT_PREVIEW_SCOPE_META, ServedUI, appearance_scope, chat_preview_scope,
    data_profile_scope)
from tests import candidate_checkout

REPO = pathlib.Path(__file__).resolve().parents[1]
META = re.compile(r'<meta name="ouroboros-appearance-scope" content="([0-9a-f]{16})">')
PREVIEW = re.compile(r'<meta name="ouroboros-chat-preview-scope" content="([0-9a-f]{16})">')
OWNER = {"X-Ouroboros-Password": "secret"}
# What a state copy needs before ``supervisor.state.read_state`` calls it current: every required control
# is present (the optional ones may stay absent), and the initialization identity is the witness's.
_STATE = {"evolution_mode_enabled": False, "evolution_owner_stopped": False, "bg_consciousness_enabled": False,
          "owner_id": None, "owner_chat_id": None, "owner_external_id": None, "owner_external_chat_id": None}


def _mint(root: pathlib.Path, identity: str, recorded_root: pathlib.Path | None = None) -> pathlib.Path:
    """The record ``install_identity.initialize_install`` writes, as a copied root would carry it too."""
    path = root / install_identity.IDENTITY_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"id": identity, "root": str(install_identity.layout_root(recorded_root or root)),
                                "created_at": "2026-10-27T00:00:00Z"}), encoding="utf-8")
    return root


def _profile(root: pathlib.Path, identity: str, *, phase: str = "complete", backup_only: bool = False) -> pathlib.Path:
    """The pair ``read_state`` proves: a primary state and the initialization witness of the same identity.

    ``backup_only`` puts the state in ``state.last_good.json`` alone: display data, never current
    authority, so it is no profile proof either."""
    state = root / "state"
    state.mkdir(parents=True, exist_ok=True)
    copy = json.dumps({"initialization_id": identity, **_STATE}, indent=2)
    (state / "state.last_good.json" if backup_only else state / "state.json").write_text(copy, encoding="utf-8")
    if backup_only:
        (state / "state.json").unlink(missing_ok=True)
    (state / "state.initialized.json").write_text(
        json.dumps({"initialization_id": identity, "phase": phase}), encoding="utf-8")
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
    monkeypatch.setattr(onboarding_host, "_live_setup", lambda _request: ({}, {}))
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


def test_the_builtin_index_names_the_data_profile_its_saved_preview_belongs_to(gate, tmp_path):
    root = _mint(tmp_path / "data", "c" * 32)
    identity = "f" * 32
    _profile(root, identity)
    client = gate(root=root)
    page = client.get("/", headers=OWNER).text
    scope = chat_preview_scope(root)
    assert PREVIEW.findall(page) == [scope]
    assert 0 < PREVIEW.search(page).start() < page.index('<script src="/static/theme.js">')
    assert re.fullmatch(r"[0-9a-f]{16}", scope) and scope != appearance_scope(root)
    assert identity not in page, "a digest: the initialization identity itself never reaches a browser"
    # Only the chat document runs a saved preview; the onboarding wizard and the login form keep the theme fact alone.
    assert PREVIEW.findall(client.get("/onboarding", headers=OWNER).text) == []
    assert PREVIEW.findall(client.get("/", headers={"Accept": "text/html"}).text) == []
    # The Telegram Mini App is the same chat document, so it gets the profile fact; the theme cookie name stays off
    # because its proxy origin never moves with the port.
    telegram = {**OWNER, "X-Ouroboros-Telegram-MiniApp": "1"}
    mini_app = client.get("/", headers=telegram).text
    assert PREVIEW.findall(mini_app) == [scope] and META.findall(mini_app) == []


def test_only_a_current_state_and_its_completed_witness_name_the_preview_scope(tmp_path):
    root = _mint(tmp_path / "data", "a" * 32)
    assert chat_preview_scope(root) == ""
    _profile(root, "b" * 32)
    scope, appearance = chat_preview_scope(root), appearance_scope(root)
    assert re.fullmatch(r"[0-9a-f]{16}", scope)
    # An owner Reset keeps the installation identity and mints a new profile identity: the theme scope holds, the
    # saved preview's does not, so rooms that reuse their numbers open on their canonical history.
    _profile(root, "d" * 32)
    assert appearance_scope(root) == appearance and chat_preview_scope(root) not in ("", scope)
    # A pending witness, a witness of another identity, and a backup-only state are no proof.
    _profile(root, "e" * 32, phase="pending")
    assert chat_preview_scope(root) == ""
    _profile(root, "e" * 32)
    (root / "state" / "state.initialized.json").write_text(
        json.dumps({"initialization_id": "f" * 32, "phase": "complete"}), encoding="utf-8")
    assert chat_preview_scope(root) == ""
    _profile(root, "e" * 32, backup_only=True)
    assert chat_preview_scope(root) == ""
    # An unreadable pair is no proof either, and the page still serves.
    (root / "state" / "state.json").write_text("{", encoding="utf-8")
    assert chat_preview_scope(root) == ""


def test_chat_preview_reads_exactly_the_profile_scope_the_core_names():
    source = (REPO / "web" / "modules" / "chat_preview.js").read_text(encoding="utf-8")
    assert CHAT_PREVIEW_SCOPE_META in source and "/^[0-9a-f]{16}$/" in source
    assert "ouroboros-appearance-scope" not in source, "the theme's installation scope is no profile proof"


def test_the_browser_harness_proves_the_index_by_exactly_the_scope_its_data_root_names(tmp_path, monkeypatch):
    """The harness proves the index bytes, the installation's appearance scope and one preview scope slot.

    ServedUI owns the preview value: it freezes one scope per generation, while the harness checks
    its place and shape. An unrelated byte change or missing or extra fact still fails the proof.
    """
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
    unset = _mint(tmp_path / "unset", "f" * 32)
    _profile(data, "1" * 32)
    _profile(other, "2" * 32)
    page, foreign, unminted = served_index(data), served_index(other), served_index(fresh)
    meta = META.search(page.decode()).group(0).encode()
    assert META.findall(foreign.decode()) == [appearance_scope(other)] != [appearance_scope(data)]
    assert PREVIEW.findall(page.decode()) == [chat_preview_scope(data)] != [chat_preview_scope(other)]
    assert META.findall(unminted.decode()) == []
    cold = served_index(unset)
    assert chat_preview_scope(fresh) == chat_preview_scope(unset) == ""
    for generation_only in (unminted, cold):
        assert len(PREVIEW.findall(generation_only.decode())) == 1
    assert qualifies(page, data) and qualifies(unminted, fresh) and qualifies(cold, unset)
    for wrong, root in ((foreign, data),  # another installation's scope
                        (unminted, data),  # the scope missing
                        (page, fresh),  # a scope this data root never minted
                        (page.replace(meta, meta + b'<meta name="ouroboros-extra" content="1">'), data),
                        (page[:-1] + bytes([page[-1] ^ 1]), data)):  # the checkout's own bytes changed
        assert not qualifies(wrong, root)
    preview = PREVIEW.search(page.decode()).group(0).encode()
    assert not qualifies(page.replace(preview, b"", 1), data)
    assert not qualifies(page.replace(preview, preview * 2, 1), data)
    for value in ("0" * 15, "0" * 17, "ABCDEF0123456789"):
        malformed = page.replace(preview, f'<meta name="{CHAT_PREVIEW_SCOPE_META}" content="{value}">'.encode(), 1)
        assert not qualifies(malformed, data)

    # A fresh generation of the same installation must serve the new profile's digest.
    # Verify that producer directly, rather than asking the static fixture to emulate it.
    _profile(data, "3" * 32)
    successor_index = served_index(data)
    assert PREVIEW.findall(successor_index.decode()) == [chat_preview_scope(data)] != PREVIEW.findall(page.decode())
    assert qualifies(successor_index, data)


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


# --- #3294 (R14): the serving WebSocket connection admits chat frames against its confirmed data profile ---


def _ws_client(root, monkeypatch, *, applied=True):
    """The real ``ws_endpoint`` behind a fake bridge that records admitted ``ui_send`` payloads."""
    import supervisor.message_bus as message_bus
    from starlette.routing import WebSocketRoute
    from ouroboros.gateway.ws import ws_endpoint

    sent = []
    bridge = SimpleNamespace(ui_send=lambda payload, **kwargs: sent.append(payload))
    monkeypatch.setattr(message_bus, "try_get_bridge", lambda: bridge)
    app = Starlette(routes=[WebSocketRoute("/ws", endpoint=ws_endpoint)])
    app.state.drive_root = root
    if applied:
        monkeypatch.setattr("ouroboros.server_process.record_applied_restart_settings", lambda settings: None)
        ServedUI(REPO).apply(app, {})
    return TestClient(app), sent


def _drain(sent, want, seconds=5.0):
    import time

    deadline = time.monotonic() + seconds
    while len(sent) < want and time.monotonic() < deadline:
        time.sleep(0.05)
    return list(sent)


def test_profile_scope_mismatch_needs_positive_proof_on_both_sides():
    from ouroboros.gateway.ws import _profile_scope_mismatch

    assert _profile_scope_mismatch("a" * 16, "b" * 16) is True
    assert _profile_scope_mismatch("a" * 16, "a" * 16) is False
    assert _profile_scope_mismatch("", "b" * 16) is False  # the connection proves no profile
    assert _profile_scope_mismatch("a" * 16, "") is False  # a legacy frame claims nothing
    assert _profile_scope_mismatch("", "") is False


def test_ws_chat_admission_binds_to_the_serving_connections_confirmed_profile(tmp_path, monkeypatch):
    root = _mint(tmp_path / "data", "c" * 32)
    _profile(root, "f" * 32)
    scope = data_profile_scope(root)
    assert re.fullmatch(r"[0-9a-f]{16}", scope)
    client, sent = _ws_client(root, monkeypatch)
    with client:
        with client.websocket_connect("/ws") as ws:
            # The connection issues its confirmed profile once, as the socket's first frame.
            issued = json.loads(ws.receive_text())
            assert issued == {"type": "connection_profile", "profile_scope": scope}
            # A page that proves a different profile: the frame is refused, never saved.
            ws.send_text(json.dumps({"type": "chat", "content": "old profile words", "client_message_id": "cm-old",
                                     "profile_scope": "0" * 16}))
            refused = json.loads(ws.receive_text())
            assert refused["type"] == "chat" and refused["system_type"] == "initialization_notice"
            assert refused["client_message_id"] == "cm-old" and "different data profile" in refused["content"]
            # The same profile's claim, and a legacy frame carrying no claim: both admitted.
            ws.send_text(json.dumps({"type": "chat", "content": "hi", "client_message_id": "cm-1",
                                     "profile_scope": scope}))
            ws.send_text(json.dumps({"type": "chat", "content": "legacy", "client_message_id": "cm-2"}))
            # One more refusal drains the loop's direct replies in order.
            ws.send_text(json.dumps({"type": "chat", "content": "again", "client_message_id": "cm-3",
                                     "profile_scope": "0" * 16}))
            again = json.loads(ws.receive_text())
            assert again["client_message_id"] == "cm-3" and "different data profile" in again["content"]
    assert sorted(_drain(sent, 2)) == ["hi", "legacy"]


def test_ws_admission_stays_open_when_the_connection_proves_no_profile(tmp_path, monkeypatch):
    root = _mint(tmp_path / "data", "c" * 32)  # no state/witness pair: no proven profile
    assert data_profile_scope(root) == ""
    client, sent = _ws_client(root, monkeypatch, applied=False)
    with client:
        with client.websocket_connect("/ws") as ws:
            # No proven profile: nothing is issued — the first frame is the schema log below.
            ws.send_text(json.dumps({"type": "chat", "content": "probe", "profile_scope": 123}))
            first = json.loads(ws.receive_text())
            assert first["type"] == "log" and "profile_scope" in first["data"]["message"]
            # Claims against no proven connection scope are no proof either: admitted as before.
            ws.send_text(json.dumps({"type": "chat", "content": "hi", "client_message_id": "cm-1",
                                     "profile_scope": "a" * 16}))
            ws.send_text(json.dumps({"type": "chat", "content": "x", "client_message_id": "cm-2",
                                     "profile_scope": "b" * 16}))
    assert sorted(_drain(sent, 2)) == ["hi", "x"]


def test_ws_issues_the_connections_confirmed_profile_once_at_accept(tmp_path, monkeypatch):
    """The accept-time issuance carries exactly the digest the drive proves, as the socket's
    first frame, and a same-profile chat frame after it is admitted as before (#3294)."""
    root = _mint(tmp_path / "data", "c" * 32)
    _profile(root, "f" * 32)
    scope = data_profile_scope(root)
    client, sent = _ws_client(root, monkeypatch)
    with client:
        with client.websocket_connect("/ws") as ws:
            first = json.loads(ws.receive_text())
            assert first == {"type": "connection_profile", "profile_scope": scope}
            ws.send_text(json.dumps({"type": "chat", "content": "hi", "client_message_id": "cm-1",
                                     "profile_scope": scope}))
    assert _drain(sent, 1) == ["hi"]


def test_ingress_rejects_a_non_string_profile_scope_claim(tmp_path, monkeypatch):
    root = _mint(tmp_path / "data", "c" * 32)
    _profile(root, "f" * 32)
    client, sent = _ws_client(root, monkeypatch)
    with client:
        with client.websocket_connect("/ws") as ws:
            assert json.loads(ws.receive_text())["type"] == "connection_profile"
            ws.send_text(json.dumps({"type": "chat", "content": "hi", "profile_scope": 123}))
            frame = json.loads(ws.receive_text())
            assert frame["type"] == "log" and "profile_scope" in frame["data"]["message"]
    assert _drain(sent, 0) == []
