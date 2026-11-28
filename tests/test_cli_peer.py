"""Test T7: non-interactive CLI commands for linked Ouroboroses (ouroboros/cli_peer.py)."""
from __future__ import annotations

import asyncio
import contextlib
import http.client
import json
import pathlib
import shutil
import socket
import tempfile
import threading
from types import SimpleNamespace

import pytest
import uvicorn
from starlette.applications import Starlette

from ouroboros import cli, peer_channel, peer_door, peer_links
from ouroboros.gateway.router import collect_routes
from ouroboros.install_identity import acting_as
from ouroboros.server_auth import NetworkAuthGate
from tests._peer_letters_shared import init_control_state
from tests.test_peer_channel import UnixSocketForwarder

_POSIX_DOOR_ONLY = pytest.mark.skipif(
    not peer_channel.is_door_supported(),
    reason="POSIX Unix-domain letters door only",
)


@contextlib.contextmanager
def _running_owner_server_with_door(root: pathlib.Path, *, start_door: bool = True):
    """Run one real main HTTP server (ephemeral loopback port) + optional POSIX letters door."""
    (root / "memory").mkdir(parents=True, exist_ok=True)
    init_control_state(root, enabled=False)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.listen(64)

    starlette_app = Starlette(routes=collect_routes(data_dir=root))
    starlette_app.state.drive_root = root
    starlette_app.state.repo_dir = pathlib.Path(__file__).resolve().parents[1]
    gate = NetworkAuthGate(starlette_app)

    async def _scoped(scope, receive, send):
        with acting_as(root):
            await gate(scope, receive, send)

    ready = threading.Event()
    stop_ev = threading.Event()
    boot = SimpleNamespace(peer_door_task=None, peer_door_server=None, peer_door_listener=contextlib.ExitStack())
    err: list[BaseException] = []

    def _run():
        async def _main():
            from ouroboros.server_boot import stop_embedded_services
            from ouroboros.server_process import _embedded_uvicorn_server

            main_srv = _embedded_uvicorn_server(uvicorn.Config(_scoped, log_level="warning", log_config=None))
            main_task = asyncio.create_task(main_srv.serve(sockets=[sock]))
            try:
                if start_door:
                    await peer_door.serve_peer_door(root, boot)
            except BaseException as exc:  # noqa: BLE001
                err.append(exc)
            finally:
                ready.set()
            while not stop_ev.is_set():
                await asyncio.sleep(0.02)
            main_srv.should_exit = True
            with contextlib.suppress(Exception):
                await asyncio.wait_for(main_task, timeout=5.0)
            await stop_embedded_services(boot)

        asyncio.run(_main())

    thread = threading.Thread(target=_run, name="test-cli-peer-srv", daemon=True)
    thread.start()
    assert ready.wait(timeout=10.0)
    if err:
        raise err[0]
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        stop_ev.set()
        thread.join(timeout=10.0)
        with contextlib.suppress(OSError):
            sock.close()


def _run_cli(capsys, argv: list[str]) -> tuple[int, str, str]:
    code = cli.main(argv)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


@_POSIX_DOOR_ONLY
def test_t7_cli_link_lifecycle_json_and_pending_code(tmp_path, capsys):
    """link invite --remote, show (pending code + list), confirm, revoke work non-interactively with --json."""
    short_dir = pathlib.Path(tempfile.mkdtemp(prefix="obcli-", dir="/tmp"))
    try:
        root_b = short_dir / "b" / "data"
        root_a = short_dir / "a" / "data"
        (root_a / "memory").mkdir(parents=True)
        init_control_state(root_a, enabled=False)

        with _running_owner_server_with_door(root_b, start_door=True) as url_b:
            # 1. link invite --remote --may-start --json
            rc, out, _ = _run_cli(capsys, [
                "--url", url_b, "link", "invite", "--remote", "--may-start",
                "--label", "Laptop", "--own-label", "ServerB", "--note", "Run checks", "--json",
            ])
            assert rc == 0, out
            invited = json.loads(out)
            assert invited["invite"].startswith("ouroboros-invite:2:"), invited
            assert invited["remote"] is True and invited["door_dir"]
            link_id = invited["link_id"]

            # 2. A accepts via channel forwarder -> link on B is pending with 6-digit code.
            peer_channel.ensure_door_dir(root_a, create=True)
            sess = peer_channel.next_channel_session()
            door_b = pathlib.Path(peer_channel.door_status(root_b)["socket_path"])
            sock_l = peer_channel.opener_socket_path(root_a, link_id)
            sock_r = peer_channel.receiver_socket_path(root_b, link_id, sess)
            fwd_l = UnixSocketForwarder(sock_l, door_b, mode="forward")
            fwd_r = UnixSocketForwarder(sock_r, door_b, mode="forward")  # placeholder socket for introduce lstat
            try:
                with acting_as(root_a):
                    accepted_a = peer_links.accept_invite(root_a, invited["invite"], session=sess,
                                                          label="ServerB", own_label="Laptop", may_start=False)
            finally:
                fwd_l.close()
                fwd_r.close()

            # 3. link show <link_id> --json returns pending 6-digit verification code!
            rc, out, _ = _run_cli(capsys, ["--url", url_b, "link", "show", link_id, "--json"])
            assert rc == 0, out
            shown = json.loads(out)["link"]
            assert shown["link_id"] == link_id
            assert shown["code"] == accepted_a["code"] and len(shown["code"]) == 6
            assert shown["code_check"] == "pending"

            # 4. link confirm <link_id> --json confirms and activates.
            rc, out, _ = _run_cli(capsys, ["--url", url_b, "link", "confirm", link_id, "--json"])
            assert rc == 0, out
            confirmed = json.loads(out)
            assert confirmed["code_check"] == "confirmed" and confirmed["state"] == "active"

            # 5. link show --json lists all links with channel metadata.
            rc, out, _ = _run_cli(capsys, ["--url", url_b, "link", "show", "--json"])
            assert rc == 0, out
            listed = json.loads(out)["links"]
            assert len(listed) == 1 and listed[0]["link_id"] == link_id and listed[0]["transport"] == "channel"

            # 6. link revoke <link_id> --json revokes.
            rc, out, _ = _run_cli(capsys, ["--url", url_b, "link", "revoke", link_id, "--json"])
            assert rc == 0, out
            revoked = json.loads(out)
            assert revoked["state"] == "removed" and revoked["revoked_at"]
            for selector, key in (([link_id], "link"), ([], "links")):
                rc, out, _ = _run_cli(capsys, ["--url", url_b, "link", "show", *selector, "--json"])
                assert rc == 0, out
                shown = json.loads(out)[key]
                shown = shown[0] if isinstance(shown, list) else shown
                assert shown["state"] == "removed", shown
                rc, out, _ = _run_cli(capsys, ["--url", url_b, "link", "show", *selector])
                assert rc == 0 and "state=removed" in out, out

            # 7. Unknown link returns non-zero exit code + typed refusal JSON.
            rc, out, _ = _run_cli(capsys, ["--url", url_b, "link", "show", "0" * 32, "--json"])
            assert rc != 0
            err_payload = json.loads(out)
            assert err_payload["ok"] is False and err_payload["refusal_code"] == "link_unknown"
    finally:
        shutil.rmtree(short_dir, ignore_errors=True)


def test_t7_cli_typed_refusals_when_door_stopped_or_on_windows(tmp_path, monkeypatch, capsys):
    """Typed non-zero refusals when the letters door is not running or on Windows."""
    root = tmp_path / "nodoor" / "data"
    expected_without_door = "peer_door_not_running" if peer_channel.is_door_supported() else "peer_door_unsupported"
    with _running_owner_server_with_door(root, start_door=False) as url:
        # Door not running on POSIX (or unsupported on Windows) -> typed refusal
        rc, out, _ = _run_cli(capsys, ["--url", url, "link", "invite", "--remote", "--json"])
        assert rc != 0
        payload = json.loads(out)
        assert payload["ok"] is False and payload["refusal_code"] == expected_without_door, payload

        # Windows (unsupported platform) -> peer_door_unsupported
        monkeypatch.setattr(peer_channel, "is_door_supported", lambda: False)
        rc2, out2, _ = _run_cli(capsys, ["--url", url, "link", "invite", "--remote", "--json"])
        assert rc2 != 0
        payload2 = json.loads(out2)
        assert payload2["ok"] is False and payload2["refusal_code"] == "peer_door_unsupported", payload2


def test_t7_cli_link_show_secondary_places_failure_returns_single_error_json(monkeypatch, capsys):
    """C13: When /api/places fails after /api/peer-links/{id} succeeds, link show <id> --json
    emits exactly one valid JSON error object and returns a non-zero exit code."""
    from ouroboros import cli_peer
    from ouroboros.cli import CLIError

    link_id = "a" * 32

    class _FakeClient:
        def request(self, method: str, path: str, body=None):
            if path == f"/api/peer-links/{link_id}":
                return {"link": {"link_id": link_id, "place_id": "place-1", "code": "123456", "code_check": "pending"}}
            if path == "/api/places":
                err = CLIError("places unavailable")
                err.status_code = 503
                err.payload = {"error": "places_unreadable", "refusal_code": "places_unreadable"}
                raise err
            raise AssertionError(f"unexpected call: {method} {path}")

    monkeypatch.setattr(cli_peer, "_client", lambda _args: _FakeClient())
    rc, out, _ = _run_cli(capsys, ["--url", "http://127.0.0.1:9", "link", "show", link_id, "--json"])
    assert rc != 0, (rc, out)
    payload = json.loads(out)  # fails with JSONDecodeError if two JSON documents were emitted
    assert payload["ok"] is False and payload["refusal_code"] == "places_unreadable", payload


def test_t7_cli_has_no_accept_or_channel_subcommands():
    """Delivery 1 CLI omits link accept and link channel."""
    parser = cli.build_parser()
    for removed in ("accept", "channel"):
        with pytest.raises(SystemExit):
            parser.parse_args(["link", removed])


def test_p1_cli_link_show_excludes_unlinked_places_and_reports_unreadable_registry(monkeypatch, capsys):
    """P1: link show lists only real link records (never substituting conversation_id for an unlinked
    residual place) and preserves a typed error when peer_links.json is unreadable."""
    from ouroboros import cli_peer

    unlinked_place = {
        "place_id": "place-residual", "connector": "ouroboros", "label": "RefusedPeer",
        "conversation_id": "deadbeef" * 4, "state": "paused",
        "link": None, "peer": {"mine": True, "may_start": False},
        "peer_effective": {"messages": False, "assignments": False, "reason": "unlinked"},
    }
    real_place = {
        "place_id": "place-real", "connector": "ouroboros", "label": "RealPeer",
        "conversation_id": "1" * 32, "state": "active",
        "link": {"link_id": "1" * 32, "code": "654321", "code_check": "confirmed",
                 "peer_install_id": "inst-p", "created_at": "2026-11-27T00:00:00Z", "revoked_at": ""},
        "peer": {"mine": True, "may_start": True},
        "peer_effective": {"messages": True, "assignments": True, "reason": ""},
    }
    unreadable_place = {
        "place_id": "place-err", "connector": "ouroboros", "label": "Corrupt",
        "conversation_id": "2" * 32, "state": "active",
        "link": None, "peer": {"mine": True, "may_start": False},
        "peer_effective": {"messages": None, "assignments": None, "reason": "unknown"},
    }

    # Case 1: unlinked residual place is excluded; only real link is returned.
    class _ClientList:
        def request(self, method, path, body=None):
            assert method == "GET" and path == "/api/places"
            return {"places": [unlinked_place, real_place]}

    monkeypatch.setattr(cli_peer, "_client", lambda _args: _ClientList())
    rc, out, _ = _run_cli(capsys, ["--url", "http://127.0.0.1:9", "link", "show", "--json"])
    assert rc == 0, out
    links = json.loads(out)["links"]
    assert [row["link_id"] for row in links] == ["1" * 32], links

    # Case 2: unreadable peer_links.json yields typed nonzero error, not empty list or fake link_id.
    class _ClientUnreadable:
        def request(self, method, path, body=None):
            return {"places": [unreadable_place]}

    monkeypatch.setattr(cli_peer, "_client", lambda _args: _ClientUnreadable())
    rc2, out2, _ = _run_cli(capsys, ["--url", "http://127.0.0.1:9", "link", "show", "--json"])
    assert rc2 != 0, (rc2, out2)
    payload2 = json.loads(out2)
    assert payload2["ok"] is False and payload2["refusal_code"] == "peer_links_unreadable", payload2


def test_p2_cli_peer_handles_interrupted_http_response_with_typed_json(monkeypatch, capsys):
    """P2: IncompleteRead / RemoteDisconnected / ConnectionResetError emit one typed JSON error
    with nonzero exit and mark mutating POSTs as outcome='unknown'."""
    import urllib.request

    class _HealthOK:
        def __enter__(self):
            return self
        def __exit__(self, *_a):
            return None
        def read(self):
            return b'{"ok": true}'

    for exc_factory in (
        lambda: http.client.IncompleteRead(b'{"partial":'),
        lambda: http.client.RemoteDisconnected("server closed mid-response"),
        lambda: ConnectionResetError("connection reset by peer"),
    ):
        def _boom(req, *_a, **_k):
            if req.full_url.endswith("/api/health"):
                return _HealthOK()
            raise exc_factory()

        monkeypatch.setattr(urllib.request, "urlopen", _boom)

        # Mutating POST (link revoke): typed JSON, nonzero exit, outcome='unknown'.
        rc_post, out_post, _ = _run_cli(capsys, ["--url", "http://127.0.0.1:9", "link", "revoke", "a" * 32, "--json"])
        assert rc_post != 0, (rc_post, out_post)
        post_body = json.loads(out_post)
        assert post_body["ok"] is False and post_body["error"] == "response_interrupted", post_body
        assert post_body.get("outcome") == "unknown", post_body

        # Read-only GET (link show): typed JSON, nonzero exit.
        rc_get, out_get, _ = _run_cli(capsys, ["--url", "http://127.0.0.1:9", "link", "show", "--json"])
        assert rc_get != 0, (rc_get, out_get)
        get_body = json.loads(out_get)
        assert get_body["ok"] is False and get_body["error"] == "response_interrupted", get_body


def test_c2_cli_peer_truncated_http_error_body_reports_response_interrupted(monkeypatch, capsys):
    """C2: Truncated/interrupted HTTP error body (e.g. 503 whose exc.read() raises IncompleteRead)
    is caught cleanly as StreamInterruptedCLIError with status_code preserved and emits typed JSON."""
    import urllib.error
    import urllib.request

    class _HealthOK:
        def __enter__(self):
            return self
        def __exit__(self, *_a):
            return None
        def read(self):
            return b'{"ok": true}'

    class _Truncated503(urllib.error.HTTPError):
        def __init__(self):
            super().__init__("http://127.0.0.1:9/api/peer-links/revoke", 503, "Service Unavailable", {}, None)
        def read(self, amt=None):
            raise http.client.IncompleteRead(b'{"error": "partial')

    def _boom(req, *_a, **_k):
        if req.full_url.endswith("/api/health"):
            return _HealthOK()
        raise _Truncated503()

    monkeypatch.setattr(urllib.request, "urlopen", _boom)
    rc_post, out_post, _ = _run_cli(capsys, ["--url", "http://127.0.0.1:9", "link", "revoke", "b" * 32, "--json"])
    assert rc_post == 2, (rc_post, out_post)
    post_body = json.loads(out_post)
    assert post_body["ok"] is False and post_body["error"] == "response_interrupted", post_body
    assert post_body.get("status") == 503 and post_body.get("outcome") == "unknown", post_body

    rc_get, out_get, _ = _run_cli(capsys, ["--url", "http://127.0.0.1:9", "link", "show", "--json"])
    assert rc_get == 2, (rc_get, out_get)
    get_body = json.loads(out_get)
    assert get_body["ok"] is False and get_body["error"] == "response_interrupted", get_body
    assert get_body.get("status") == 503 and "outcome" not in get_body, get_body
