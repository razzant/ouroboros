"""Tests T1 and T2 for the dedicated POSIX letters door (ouroboros/peer_door.py & peer_channel.py)."""
from __future__ import annotations

import asyncio
import os
import pathlib
import shutil
import stat
import tempfile
import threading
from contextlib import ExitStack
from types import SimpleNamespace

import pytest
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from ouroboros import peer_channel, peer_door, peer_links
from ouroboros.install_identity import acting_as
from tests._peer_letters_shared import init_control_state, introduce, ready_queue, receiver_client

_POSIX_DOOR_ONLY = pytest.mark.skipif(
    not peer_channel.is_door_supported(),
    reason="POSIX Unix-domain letters door only",
)


def _serve_door_in_thread(root: pathlib.Path):
    """Run ``peer_door.serve_peer_door`` on a dedicated asyncio event loop thread; returns (boot, stop_fn)."""
    ready = threading.Event()
    stop_ev = threading.Event()
    err: list[BaseException] = []
    boot = SimpleNamespace(peer_door_task=None, peer_door_server=None, peer_door_listener=ExitStack())

    def _runner():
        async def _main():
            try:
                await peer_door.serve_peer_door(root, boot)
            except BaseException as exc:  # noqa: BLE001
                err.append(exc)
            finally:
                ready.set()
            while not stop_ev.is_set():
                await asyncio.sleep(0.02)
            from ouroboros.server_boot import stop_embedded_services

            await stop_embedded_services(boot)

        asyncio.run(_main())

    thread = threading.Thread(target=_runner, name="test-peer-door", daemon=True)
    thread.start()
    assert ready.wait(timeout=10.0), "peer door startup timed out"
    if err:
        raise err[0]

    def _stop():
        stop_ev.set()
        thread.join(timeout=10.0)

    return boot, _stop


# ── T1: Letters door surface, auth isolation, and main-port separation ───────

def test_t1_door_refuses_owner_routes_without_network_password(tmp_path, monkeypatch):
    """Without any network password configured, the letters door refuses every owner/UI/WS route,
    refuses valid link keys on non-letter routes, never treats owner password/cookie as owner,
    and requires an invitation token on /api/peer-links/introduce."""
    monkeypatch.delenv("OUROBOROS_NETWORK_PASSWORD", raising=False)
    pair = introduce(tmp_path, monkeypatch)
    door_app = peer_door.create_peer_door_app(pair.root_b)

    async def _as_b(scope, receive, send):
        with acting_as(pair.root_b):
            await door_app(scope, receive, send)

    owner_paths = [
        ("GET", "/"),
        ("GET", "/api/state"),
        ("GET", "/api/settings"),
        ("GET", "/api/tasks"),
        ("GET", "/api/files/list"),
        ("POST", "/api/peer-links/invite"),
        ("POST", "/api/peer-links/accept"),
        ("POST", f"/api/peer-links/{pair.link_id}/confirm"),
        ("POST", f"/api/peer-links/{pair.link_id}/revoke"),
        ("GET", f"/api/peer-links/{pair.link_id}"),
    ]

    with TestClient(_as_b, base_url="http://peer.localhost", client=None) as client:
        # 1. Requests without peer headers to owner/UI routes are refused.
        for method, path in owner_paths:
            resp = client.request(method, path)
            assert resp.status_code == 403, (method, path, resp.status_code, resp.text)
            assert resp.json()["refusal_code"] == "peer_route_forbidden", (path, resp.json())

        # 2. WebSocket through the door is refused with 4403 (with or without link key).
        for hdrs in ({}, pair.headers()):
            with pytest.raises(WebSocketDisconnect) as closed:
                with client.websocket_connect("/ws", headers=hdrs):
                    pass
            assert closed.value.code == 4403

        # 3. A valid link key on any non-letter route is refused.
        for method, path in owner_paths:
            resp = client.request(method, path, headers=pair.headers())
            assert resp.status_code == 403, (method, path, resp.status_code, resp.text)
            assert resp.json()["refusal_code"] == "peer_route_forbidden"

        # 4. Owner password header or valid session cookie never grants owner authority on the door.
        from ouroboros.server_auth import AUTH_COOKIE_NAME, _session_value

        monkeypatch.setenv("OUROBOROS_NETWORK_PASSWORD", "top-secret-pw")
        cookie = _session_value("top-secret-pw")
        for hdrs in (
            {"X-Ouroboros-Password": "top-secret-pw"},
            {"Authorization": "Bearer top-secret-pw"},
            {"Cookie": f"{AUTH_COOKIE_NAME}={cookie}"},
        ):
            r_state = client.get("/api/state", headers=hdrs)
            assert r_state.status_code == 403 and r_state.json()["refusal_code"] == "peer_route_forbidden"
            r_inv = client.post("/api/peer-links/invite", json={"may_start": True}, headers=hdrs)
            assert r_inv.status_code == 403 and r_inv.json()["refusal_code"] == "peer_route_forbidden"

        # 5. /api/peer-links/introduce requires a valid invitation token.
        no_tok = client.post("/api/peer-links/introduce", json={})
        assert no_tok.status_code == 401 and no_tok.json()["refusal_code"] == "invite_unknown"
        bad_tok = client.post(
            "/api/peer-links/introduce",
            json={"link_id": "a" * 32, "install_id": "x", "url": "http://127.0.0.1:9", "key_sha256": "0" * 64},
            headers={"X-Ouroboros-Invite": "not-a-real-token"},
        )
        assert bad_tok.status_code == 401 and bad_tok.json()["refusal_code"] == "invite_unknown"

        # 6. A valid link key DOES reach letter routes even with client=None (UDS scope).
        missing = client.get("/api/letters/no-such-letter", headers=pair.headers())
        assert missing.status_code == 404 and missing.json()["refusal_code"] == "letter_not_found"


def test_t1_main_port_never_routes_letters_channel(tmp_path, monkeypatch):
    """On the main port, POST /api/letters/channel never reaches api_letters_channel
    (refused 403 peer_route_forbidden with link key; 404/405 without)."""
    pair = introduce(tmp_path, monkeypatch)
    called: list[str] = []
    monkeypatch.setattr(
        peer_channel, "set_channel_session",
        lambda *a, **k: (called.append("reached"), {"ok": True})[1],
    )
    with receiver_client(pair, monkeypatch) as client:
        with_key = client.post(
            "/api/letters/channel",
            json={"session": "000000000001"},
            headers=pair.headers(),
        )
        assert with_key.status_code == 403 and with_key.json()["refusal_code"] == "peer_route_forbidden"
        without_key = client.post("/api/letters/channel", json={"session": "000000000001"})
        assert without_key.status_code in {404, 405}
    assert called == [], "POST /api/letters/channel must never reach the announce handler on the main port"


def test_t1_get_and_head_letter_id_channel_allowed_on_both_listeners(tmp_path, monkeypatch):
    """C14: Only POST /api/letters/channel is door-only; GET and HEAD for letter_id=='channel'
    work on both the main port and the dedicated letters door (404 absent, 200 present)."""
    pair = introduce(tmp_path, monkeypatch)
    ready_queue(monkeypatch, pair.root_b)
    door_app = peer_door.create_peer_door_app(pair.root_b)

    async def _as_b(scope, receive, send):
        with acting_as(pair.root_b):
            await door_app(scope, receive, send)

    with receiver_client(pair, monkeypatch) as main_client, TestClient(
        _as_b, base_url="http://peer.localhost", client=None
    ) as door_client:
        # Before letter "channel" exists: GET and HEAD return 404 letter_not_found on both listeners.
        for client in (main_client, door_client):
            g_absent = client.get("/api/letters/channel", headers=pair.headers())
            assert g_absent.status_code == 404 and g_absent.json()["refusal_code"] == "letter_not_found"
            h_absent = client.head("/api/letters/channel", headers=pair.headers())
            assert h_absent.status_code == 404

        # Create a real work letter whose letter_id is literally "channel".
        created = main_client.post(
            "/api/letters",
            json={"letter_id": "channel", "text": "Count files", "protocol": 1},
            headers=pair.headers(),
        )
        assert created.status_code == 200, created.text

        # GET and HEAD return 200 on both main port and dedicated door; POST stays 403 on main port.
        for client in (main_client, door_client):
            g_present = client.get("/api/letters/channel", headers=pair.headers())
            assert g_present.status_code == 200 and g_present.json()["letter_id"] == "channel"
            h_present = client.head("/api/letters/channel", headers=pair.headers())
            assert h_present.status_code == 200
        post_main = main_client.post("/api/letters/channel", json={"session": "000000000001"}, headers=pair.headers())
        assert post_main.status_code == 403 and post_main.json()["refusal_code"] == "peer_route_forbidden"


def test_t1_door_defers_assignment_while_supervisor_not_ready(tmp_path, monkeypatch):
    """C12: The letters door shares supervisor_ready_event with the main app so work assignments
    are deferred 503 unconfirmed during startup recovery even after workers exist."""
    pair = introduce(tmp_path, monkeypatch)
    ready_queue(monkeypatch, pair.root_b)
    ready_ev = threading.Event()  # unset: startup recovery still running
    door_app = peer_door.create_peer_door_app(pair.root_b, supervisor_ready_event=ready_ev, repo_dir=pair.repo)

    async def _as_b(scope, receive, send):
        with acting_as(pair.root_b):
            await door_app(scope, receive, send)

    with TestClient(_as_b, base_url="http://peer.localhost", client=None) as door_client:
        resp = door_client.post(
            "/api/letters",
            json={"letter_id": "boot-race-1", "text": "Do work during startup", "protocol": 1},
            headers=pair.headers(),
        )
        assert resp.status_code == 503, (resp.status_code, resp.text)
        body = resp.json()
        assert body["error"] == "unconfirmed"
        assert "supervisor is still starting" in str(body.get("detail") or "")
        assert pair.row("boot-race-1") == {}, "no task admitted before supervisor_ready_event is set"

        # Once startup completes, the same letter is admitted cleanly.
        ready_ev.set()
        ok = door_client.post(
            "/api/letters",
            json={"letter_id": "boot-race-1", "text": "Do work during startup", "protocol": 1},
            headers=pair.headers(),
        )
        assert ok.status_code == 200 and ok.json()["state"] == "accepted_running", ok.text
        assert pair.row("boot-race-1")

    # Verify server.lifespan wires supervisor_ready_event & repo_dir through server_boot.serve_peer_door -> peer_door.
    captured: dict = {}

    async def _fake_serve(root, boot, *, supervisor_ready_event=None, repo_dir=None):
        captured.update(root=root, boot=boot, ready=supervisor_ready_event, repo_dir=repo_dir)

    import types
    import server
    from ouroboros import config, server_boot

    monkeypatch.setattr(peer_door, "serve_peer_door", _fake_serve)
    monkeypatch.setattr(server, "BOOT", types.SimpleNamespace(reset=lambda: None, serving=lambda: None, stopping=lambda: None))
    monkeypatch.setattr(server, "admit_data_root", lambda *_: None)
    monkeypatch.setattr(server.process_scope, "initialize_installation_scope", lambda *_: None)
    monkeypatch.setattr(server.store_switch, "start", lambda *_a, **_k: None)
    monkeypatch.setattr(server.store_switch, "when_ready", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "bind_body_lifespan", lambda *_a, **_k: (lambda: None))
    monkeypatch.setattr(server.served_ui, "apply", lambda *_: None)
    monkeypatch.setattr(config, "initialize_runtime_mode_baseline", lambda: None)
    monkeypatch.setattr(config, "load_settings", lambda: {})
    monkeypatch.setattr(server, "has_startup_ready_provider", lambda _s: False)
    monkeypatch.setattr(server, "_set_ws_event_loop", lambda *_: None)
    monkeypatch.setattr(server, "_supervisor_thread", None)
    monkeypatch.setattr(server, "_supervisor_ready", ready_ev)
    monkeypatch.setattr(server, "_supervisor_stop", threading.Event())
    monkeypatch.setattr(server, "_extensions_ready", threading.Event())
    monkeypatch.setattr(server, "begin_owned_stop", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "stop_owned_work", lambda *_a, **_k: None)
    monkeypatch.setattr(server_boot, "stop_extension_loop", lambda: None)
    monkeypatch.setattr(server, "_shutdown_supervisor_event_bus", lambda: None)

    recorded_steps: list[tuple] = []

    def _capture_boot(steps, *, stop=None):
        recorded_steps.extend(steps)

    monkeypatch.setattr(server, "start_boot_coordinator", _capture_boot)
    app = types.SimpleNamespace(state=types.SimpleNamespace(drive_root=pair.root_b))

    async def _exercise_lifespan():
        async with server.lifespan(app):
            door_steps = [fn for name, fn in recorded_steps if name == "peer_door"]
            assert len(door_steps) == 1, recorded_steps
            await asyncio.to_thread(door_steps[0])

    asyncio.run(_exercise_lifespan())
    assert captured["root"] == pair.root_b
    assert captured["ready"] is ready_ev
    assert captured["repo_dir"] == server.REPO_DIR


# ── T2: Door directory permissions, lstat verification, path length, no XDG ─

def test_t2_unsupported_platform_refuses_door_and_keeps_main_port_working(tmp_path, monkeypatch):
    """C8: When the POSIX letters door is unsupported (e.g. Windows), door start records typed
    peer_door_unsupported, remote invites refuse 501, and main-port owner API + v1 invites keep working."""
    from starlette.applications import Starlette

    from ouroboros.gateway.router import collect_routes
    from ouroboros.server_auth import NetworkAuthGate

    monkeypatch.setattr(peer_channel, "is_door_supported", lambda: False)
    root = tmp_path / "win_inst" / "data"
    (root / "memory").mkdir(parents=True)
    init_control_state(root, enabled=False)

    boot, stop = _serve_door_in_thread(root)
    try:
        st = peer_channel.door_status(root)
        assert st["listening"] is False and st["error_code"] == "peer_door_unsupported", st
        with pytest.raises(peer_links.PeerLinkError) as e1:
            peer_channel.resolve_door_dir(root)
        assert e1.value.code == "peer_door_unsupported" and e1.value.status == 501
        with pytest.raises(peer_links.PeerLinkError) as e2:
            peer_channel.require_door_running(root)
        assert e2.value.code == "peer_door_unsupported" and e2.value.status == 501

        starlette_app = Starlette(routes=collect_routes(data_dir=root))
        starlette_app.state.drive_root = root
        starlette_app.state.repo_dir = pathlib.Path(__file__).resolve().parents[1]
        gate = NetworkAuthGate(starlette_app)

        async def _scoped(scope, receive, send):
            with acting_as(root):
                await gate(scope, receive, send)

        with TestClient(_scoped, base_url="http://127.0.0.1:8772") as client:
            assert client.get("/api/health").status_code == 200
            assert client.get("/api/places").status_code == 200
            r_remote = client.post("/api/peer-links/invite", json={"remote": True, "may_start": False})
            assert r_remote.status_code == 501 and r_remote.json()["refusal_code"] == "peer_door_unsupported"
            r_local = client.post(
                "/api/peer-links/invite",
                json={"own_url": "http://127.0.0.1:8772", "may_start": False},
            )
            assert r_local.status_code == 200 and r_local.json()["invite"].startswith("ouroboros-invite:1:")
    finally:
        stop()


@_POSIX_DOOR_ONLY
def test_t2_door_dir_0700_socket_0600_and_live_uds_serving(tmp_path):
    """Starting the door creates a 0700 directory and a 0600 door.sock and serves HTTP over UDS."""
    short_dir = pathlib.Path(tempfile.mkdtemp(prefix="obd-", dir="/tmp"))
    root = short_dir / "inst"
    root.mkdir()
    boot, stop = _serve_door_in_thread(root)
    try:
        status = peer_channel.door_status(root)
        assert status["listening"] is True, status
        door_dir = pathlib.Path(status["door_dir"])
        sock_path = pathlib.Path(status["socket_path"])
        assert stat.S_IMODE(os.lstat(door_dir).st_mode) == 0o700
        assert stat.S_ISSOCK(os.lstat(sock_path).st_mode)
        assert stat.S_IMODE(os.lstat(sock_path).st_mode) == 0o600
        # Real UDS HTTP request hits the door and gets peer_route_forbidden on /api/state.
        with peer_channel.peer_http_client(channel_path=sock_path, timeout_sec=5.0) as client:
            resp = client.get("/api/state")
            assert resp.status_code == 403 and resp.json()["refusal_code"] == "peer_route_forbidden"
    finally:
        stop()
        sock_still = pathlib.Path(status["socket_path"]).exists() if "status" in locals() else False
        shutil.rmtree(short_dir, ignore_errors=True)
    assert not sock_still, "door socket unlinked on clean shutdown"


@_POSIX_DOOR_ONLY
def test_t2_door_refuses_bad_mode_symlink_and_foreign_owner_without_failing_server(tmp_path, monkeypatch):
    """Wrong mode, symlink directory, or foreign UID prevent door startup with typed status
    and leave the main server able to continue."""
    # Case 1: wrong mode (0755)
    with tempfile.TemporaryDirectory(prefix="ob-t2m-", dir="/tmp") as folder:
        root_mode = pathlib.Path(folder)
        door_dir = peer_channel.resolve_door_dir(root_mode)
        door_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(door_dir, 0o755)
        boot, stop = _serve_door_in_thread(root_mode)
        try:
            st = peer_channel.door_status(root_mode)
            assert st["listening"] is False and st["error_code"] == "peer_door_dir_invalid", st
        finally:
            stop()
            os.chmod(door_dir, 0o700)
            door_dir.rmdir()

    # Case 2: symlink door dir
    target = tmp_path / "real_target"
    target.mkdir(mode=0o700)
    with tempfile.TemporaryDirectory(prefix="ob-t2s-", dir="/tmp") as folder:
        root_sym = pathlib.Path(folder)
        door_dir = peer_channel.resolve_door_dir(root_sym)
        door_dir.parent.mkdir(parents=True, exist_ok=True)
        if door_dir.exists() or door_dir.is_symlink():
            door_dir.unlink()
        os.symlink(target, door_dir)
        with pytest.raises(peer_links.PeerLinkError) as exc:
            peer_channel.ensure_door_dir(root_sym, create=True)
        assert exc.value.code == "peer_door_dir_invalid"

    # Case 3: foreign UID on door dir
    short_uid = pathlib.Path(tempfile.mkdtemp(prefix="obu-", dir="/tmp"))
    try:
        root_uid = short_uid / "u"
        root_uid.mkdir()
        real_dir = peer_channel.ensure_door_dir(root_uid, create=True)
        real_lstat = os.lstat

        def _fake_lstat(path, *a, **k):
            res = real_lstat(path, *a, **k)
            if pathlib.Path(path) == real_dir:
                return SimpleNamespace(st_mode=res.st_mode, st_uid=res.st_uid + 1, st_ino=res.st_ino)
            return res

        monkeypatch.setattr(os, "lstat", _fake_lstat)
        with pytest.raises(peer_links.PeerLinkError) as exc:
            peer_channel.verify_door_dir(root_uid)
        assert exc.value.code == "peer_door_dir_invalid"
    finally:
        shutil.rmtree(short_uid, ignore_errors=True)


@_POSIX_DOOR_ONLY
def test_t2_long_data_dir_falls_back_to_short_deterministic_path_ignoring_xdg(tmp_path, monkeypatch):
    """A data_dir > 104 bytes yields /tmp/ouroboros-peer-<uid>/<16hex> with longest socket <= 100 bytes,
    and XDG_RUNTIME_DIR never influences the chosen path."""
    long_dir = tmp_path / ("very_long_directory_segment_" * 4) / "data"
    long_dir.mkdir(parents=True)
    assert len(str(long_dir).encode("utf-8")) > 104

    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "xdg-run"))
    path_with_xdg = peer_channel.resolve_door_dir(long_dir)
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    path_without_xdg = peer_channel.resolve_door_dir(long_dir)

    assert path_with_xdg == path_without_xdg
    assert str(path_with_xdg).startswith(f"/tmp/ouroboros-peer-{os.getuid()}/")
    longest = path_with_xdg / f"r-{'0' * 16}-{'0' * 12}.sock"
    assert len(str(longest).encode("utf-8")) <= peer_channel.MAX_SOCKET_PATH_BYTES


@_POSIX_DOOR_ONLY
def test_q6_introduce_enforces_listener_transport_identity(tmp_path, monkeypatch):
    """Q6: v1 invitations are accepted only on the main port; v2 invitations only on the letters door."""
    import hashlib
    from starlette.applications import Starlette
    from ouroboros.gateway.router import collect_routes
    from ouroboros.server_auth import NetworkAuthGate

    short_dir = pathlib.Path(tempfile.mkdtemp(prefix="obq6-", dir="/tmp"))
    try:
        root_b = short_dir / "b" / "data"
        (root_b / "memory").mkdir(parents=True)
        init_control_state(root_b, enabled=False)

        # Build both apps: main port (door_only=False) and letters door (door_only=True).
        main_starlette = Starlette(routes=collect_routes(data_dir=root_b))
        main_starlette.state.drive_root = root_b
        main_starlette.state.repo_dir = pathlib.Path(__file__).resolve().parents[1]
        main_app = NetworkAuthGate(main_starlette)

        door_app = peer_door.create_peer_door_app(root_b)
        _boot, stop_door = _serve_door_in_thread(root_b)
        try:
            with acting_as(root_b):
                v1_inv = peer_links.create_invite(root_b, own_url="http://127.0.0.1:8771", may_start=False)
                v1_parsed = peer_links.parse_invite(v1_inv["invite"])
                v2_inv = peer_links.create_invite(root_b, may_start=False, remote=True)
                v2_parsed = peer_links.parse_invite(v2_inv["invite"])
                sess = peer_channel.next_channel_session()

                # 1. Valid v1 invite + v1 body sent to the letters door -> 400 invite_transport_mismatch, invite NOT consumed.
                v1_body = {
                    "link_id": v1_parsed["link_id"], "install_id": "inst-a",
                    "url": "http://127.0.0.1:8772", "label": "A",
                    "key_sha256": hashlib.sha256(b"k1").hexdigest(), "protocol": 1,
                }
                with TestClient(door_app) as door_client:
                    r1 = door_client.post("/api/peer-links/introduce", json=v1_body,
                                          headers={"X-Ouroboros-Invite": v1_parsed["token"]})
                assert r1.status_code == 400 and r1.json().get("refusal_code") == "invite_transport_mismatch", r1.text

                # 2. Valid v2 invite + v2 body sent to the main port -> 400 invite_transport_mismatch, invite NOT consumed.
                # Create the receiver socket so v2 introduce passes verify_channel_socket on unmodified code.
                import socket as _sock
                r_sock_path = peer_channel.receiver_socket_path(root_b, v2_parsed["link_id"], sess)
                rs = _sock.socket(_sock.AF_UNIX, _sock.SOCK_STREAM)
                rs.bind(str(r_sock_path))
                os.chmod(r_sock_path, 0o600)
                try:
                    v2_body = {
                        "link_id": v2_parsed["link_id"], "install_id": "inst-a",
                        "session": sess, "label": "A",
                        "key_sha256": hashlib.sha256(b"k2").hexdigest(), "protocol": 1,
                    }
                    with TestClient(main_app, client=("127.0.0.1", 12345)) as main_client:
                        r2 = main_client.post("/api/peer-links/introduce", json=v2_body,
                                              headers={"X-Ouroboros-Invite": v2_parsed["token"]})
                    assert r2.status_code == 400 and r2.json().get("refusal_code") == "invite_transport_mismatch", r2.text
                finally:
                    rs.close()
                    r_sock_path.unlink(missing_ok=True)

                # Both invites remain unused in the registry.
                reg = peer_links.read_registry(root_b)
                assert v1_parsed["link_id"] not in reg["links"]
                assert v2_parsed["link_id"] not in reg["links"]
        finally:
            stop_door()
    finally:
        shutil.rmtree(short_dir, ignore_errors=True)


def test_p6_peer_door_endpoints_and_shapes_match_mounted_routes():
    """P6: endpoint_index.PEER_DOOR_ENDPOINTS and PEER_DOOR_SHAPES stay in lockstep with peer_door_routes()."""
    from ouroboros.gateway import endpoint_index

    mounted = {
        f"{method} {route.path}"
        for route in peer_door.peer_door_routes()
        for method in sorted(route.methods or ())
        if method != "HEAD"
    }
    indexed = set(endpoint_index.PEER_DOOR_ENDPOINTS)
    assert indexed == mounted, f"indexed={sorted(indexed)} mounted={sorted(mounted)}"
    assert set(endpoint_index.PEER_DOOR_SHAPES).issubset(indexed)
