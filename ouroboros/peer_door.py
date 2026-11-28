"""Dedicated ASGI letters door on a POSIX Unix-domain socket.

The letters door exposes only peer-authenticated letter routes (``/api/letters*`` and
``POST /api/letters/channel``) and one-time invitation introduction (``POST /api/peer-links/introduce``).
It runs as its own Starlette ASGI application wrapped in ``PeerDoorAuthGate``, with no owner branch:
requests without a valid link key or invitation token, requests for any other path, and all WebSocket
connections are refused regardless of network password or owner cookies.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import pathlib
import socket
import stat
from typing import Any

import uvicorn
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.types import ASGIApp, Receive, Scope, Send

from ouroboros import peer_channel
from ouroboros.server_auth import authenticate_peer_scope

log = logging.getLogger(__name__)


def peer_door_routes() -> list[Route]:
    """Exact route table of the dedicated letters door (handlers shared with the main port)."""
    from ouroboros.gateway.peer_letters_api import peer_door_letter_routes

    return peer_door_letter_routes()


class PeerDoorAuthGate:
    """Peer-only ASGI gate for the dedicated letters door: never has an owner branch."""

    def __init__(self, app: ASGIApp, *, drive_root: pathlib.Path) -> None:
        self.app = app
        self._drive_root = pathlib.Path(drive_root)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in {"http", "websocket"}:
            await self.app(scope, receive, send)
            return
        await authenticate_peer_scope(
            scope,
            receive,
            send,
            app=self.app,
            root=self._drive_root,
            require_direct=False,
            door_only=True,
        )


def create_peer_door_app(
    drive_root: Any,
    *,
    supervisor_ready_event: Any = None,
    repo_dir: Any = None,
) -> PeerDoorAuthGate:
    """Build the dedicated letters-door ASGI application for ``drive_root``."""
    root = pathlib.Path(drive_root)
    starlette_app = Starlette(routes=peer_door_routes())
    starlette_app.state.drive_root = root
    if supervisor_ready_event is not None:
        starlette_app.state.supervisor_ready_event = supervisor_ready_event
    if repo_dir is not None:
        starlette_app.state.repo_dir = pathlib.Path(repo_dir)
    return PeerDoorAuthGate(starlette_app, drive_root=root)


@contextlib.contextmanager
def bound_peer_door_socket(root: Any):
    """Verify the 0700 door directory, remove any stale own door socket, bind AF_UNIX, and yield ``(sock, door_dir, sock_path)``."""
    door_dir, owned = peer_channel._ensure_door_dir(root, create=True)
    try:
        with _bound_socket(door_dir) as bound:
            yield bound
    finally:
        if owned is not None:
            with contextlib.suppress(OSError):
                current = door_dir.lstat()
                if (current.st_dev, current.st_ino) == (owned.st_dev, owned.st_ino):
                    door_dir.rmdir()  # empty only; a peer's nonempty directory is untouched


@contextlib.contextmanager
def _bound_socket(door_dir: pathlib.Path):
    from ouroboros.peer_links import PeerLinkError

    sock_path = door_dir / peer_channel.DOOR_SOCKET_NAME
    uid = os.getuid()
    try:
        st = os.lstat(sock_path)
    except FileNotFoundError:
        st = None
    except OSError as exc:
        raise PeerLinkError("peer_door_socket_invalid", f"cannot inspect door socket: {exc}", 503) from exc
    if st is not None:
        if not stat.S_ISSOCK(st.st_mode):
            raise PeerLinkError("peer_door_socket_invalid", "door socket path exists and is not a Unix socket", 503)
        if st.st_uid != uid:
            raise PeerLinkError("peer_door_socket_invalid", "door socket is not owned by the current user", 503)
        try:
            sock_path.unlink()
        except OSError as exc:
            raise PeerLinkError("peer_door_socket_invalid", f"stale door socket could not be removed: {exc}", 503) from exc

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    bound_ino = None
    try:
        sock.bind(str(sock_path))
        os.chmod(sock_path, 0o600)
        sock.listen(64)
        bound_ino = os.lstat(sock_path).st_ino
        yield sock, door_dir, sock_path
    finally:
        with contextlib.suppress(OSError):
            sock.close()
        if bound_ino is not None:
            with contextlib.suppress(OSError):
                cur = os.lstat(sock_path)
                if stat.S_ISSOCK(cur.st_mode) and cur.st_ino == bound_ino:
                    sock_path.unlink(missing_ok=True)


async def serve_peer_door(
    root: pathlib.Path,
    boot: Any,
    *,
    supervisor_ready_event: Any = None,
    repo_dir: Any = None,
) -> None:
    """Start the dedicated letters door on the server's ASGI event loop; never blocks main server boot on refusal."""
    from ouroboros.peer_links import PeerLinkError
    from ouroboros.server_process import _embedded_uvicorn_server

    if not peer_channel.is_door_supported():
        peer_channel.register_door_status(
            root,
            listening=False,
            error_code="peer_door_unsupported",
            error="the letters door is supported on POSIX systems only",
        )
        log.info("Peer letters door skipped: unsupported platform")
        return

    try:
        door_socket, door_dir, sock_path = boot.peer_door_listener.enter_context(bound_peer_door_socket(root))
    except (PeerLinkError, OSError) as exc:
        code = getattr(exc, "code", "") or "peer_door_bind_failed"
        peer_channel.register_door_status(root, listening=False, error_code=code, error=str(exc))
        log.warning("Peer letters door did not start (%s): %s", code, exc)
        return

    door_app = create_peer_door_app(root, supervisor_ready_event=supervisor_ready_event, repo_dir=repo_dir)
    config = uvicorn.Config(door_app, log_level="warning", log_config=None)
    boot.peer_door_server = _embedded_uvicorn_server(config)
    peer_channel.register_door_status(root, listening=True, door_dir=str(door_dir), socket_path=str(sock_path))

    def _on_done(_task: Any) -> None:
        peer_channel.unregister_door_status(root)
        boot.peer_door_listener.close()

    boot.peer_door_task = asyncio.create_task(
        boot.peer_door_server.serve(sockets=[door_socket]),
        name="peer-letters-door",
    )
    boot.peer_door_task.add_done_callback(_on_done)
    log.info("Peer letters door listening on %s", sock_path)
