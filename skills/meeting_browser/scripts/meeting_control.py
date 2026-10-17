"""Private control channel between the meeting tools and the capture companion.

The tools run inside the turn that called them (a worker or a short-lived
extension child); the companion runs under the host's supervisor and is the one
process that holds audio devices and the call socket. They meet on loopback TCP:
the companion binds 127.0.0.1 on a free port and publishes ``control.json``
(port, random token, pid) in the skill's state directory with owner-only
permissions; a tool reads it, connects, sends one JSON request line carrying the
token and reads one JSON reply line. Loopback TCP behaves the same on macOS,
Linux and Windows (an AF_UNIX path under a deep state directory can exceed the
platform's socket path limit); the token keeps any process that cannot read the
state directory out.
"""

from __future__ import annotations

import hmac
import json
import os
import pathlib
import secrets
import socket
import socketserver
import threading
from typing import Any, Callable, Dict, Optional

CONTROL_FILE = "control.json"
# Transport bounds, not policy: one request or reply is a small JSON object.
MAX_MESSAGE_BYTES = 64 * 1024
CONNECT_TIMEOUT_SEC = 3.0

Handler = Callable[[str, Dict[str, Any]], Dict[str, Any]]


class ControlUnavailable(RuntimeError):
    """The companion is not running or not answering; the text says why in plain words."""


def _write_private(path: pathlib.Path, data: bytes) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
    os.replace(str(tmp), str(path))


def read_endpoint(state_dir: pathlib.Path) -> Dict[str, Any]:
    path = pathlib.Path(state_dir) / CONTROL_FILE
    try:
        if path.stat().st_size > MAX_MESSAGE_BYTES:
            raise ControlUnavailable("the meeting companion's control file is unreadable")
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ControlUnavailable(
            "the meeting companion is not running (the skill may be disabled or still starting)"
        ) from None
    except (OSError, ValueError):
        raise ControlUnavailable("the meeting companion's control file is unreadable") from None
    if not isinstance(data, dict) or not isinstance(data.get("port"), int) or not data.get("token"):
        raise ControlUnavailable("the meeting companion's control file is unreadable")
    return data


def _read_line(sock: socket.socket) -> bytes:
    chunks = bytearray()
    while not chunks.endswith(b"\n"):
        piece = sock.recv(4096)
        if not piece:
            break
        chunks.extend(piece)
        if len(chunks) > MAX_MESSAGE_BYTES:
            raise ControlUnavailable("the meeting companion sent an oversized reply")
    return bytes(chunks)


def request(state_dir: pathlib.Path, op: str, args: Optional[Dict[str, Any]] = None,
            *, timeout_sec: float = 30.0) -> Dict[str, Any]:
    """Send one operation to the companion and return its JSON reply."""
    endpoint = read_endpoint(state_dir)
    line = json.dumps({"token": endpoint["token"], "op": op, "args": dict(args or {})}).encode("utf-8")
    try:
        with socket.create_connection(("127.0.0.1", endpoint["port"]), timeout=CONNECT_TIMEOUT_SEC) as sock:
            sock.settimeout(timeout_sec)
            sock.sendall(line + b"\n")
            raw = _read_line(sock)
    except ControlUnavailable:
        raise
    except (OSError, socket.timeout):
        raise ControlUnavailable(
            "the meeting companion is not answering (it may be restarting); try again shortly"
        ) from None
    try:
        reply = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise ControlUnavailable("the meeting companion sent an unreadable reply") from None
    if not isinstance(reply, dict):
        raise ControlUnavailable("the meeting companion sent an unreadable reply")
    return reply


class _Server(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = False


class ControlServer:
    """The companion's end: authenticate each request line and hand it to ``handler``."""

    def __init__(self, state_dir: pathlib.Path, handler: Handler):
        self.state_dir = pathlib.Path(state_dir)
        self._handler = handler
        self._token = secrets.token_urlsafe(32)
        self._server: Optional[_Server] = None
        self._thread: Optional[threading.Thread] = None

    @property
    def port(self) -> int:
        return self._server.server_address[1] if self._server is not None else 0

    def start(self) -> None:
        owner = self

        class _Request(socketserver.StreamRequestHandler):
            def handle(self) -> None:
                self.request.settimeout(60.0)
                raw = self.rfile.readline(MAX_MESSAGE_BYTES + 1)
                reply = owner._answer(raw)
                self.wfile.write(json.dumps(reply, ensure_ascii=False).encode("utf-8") + b"\n")

        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._server = _Server(("127.0.0.1", 0), _Request)
        self._thread = threading.Thread(target=self._server.serve_forever, name="meeting-control",
                                        daemon=True)
        self._thread.start()
        _write_private(self.state_dir / CONTROL_FILE, json.dumps({
            "port": self.port, "token": self._token, "pid": os.getpid(),
        }).encode("utf-8"))

    def _answer(self, raw: bytes) -> Dict[str, Any]:
        if len(raw) > MAX_MESSAGE_BYTES:
            return {"ok": False, "code": "MEETING_CONTROL_REFUSED", "message": "request too large"}
        try:
            message = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            return {"ok": False, "code": "MEETING_CONTROL_REFUSED", "message": "unreadable request"}
        if not isinstance(message, dict) or not hmac.compare_digest(
                str(message.get("token") or ""), self._token):
            return {"ok": False, "code": "MEETING_CONTROL_REFUSED", "message": "not authorized"}
        args = message.get("args") if isinstance(message.get("args"), dict) else {}
        try:
            return self._handler(str(message.get("op") or ""), args)
        except Exception as exc:  # the reply is the error channel of this transport
            return {"ok": False, "code": "MEETING_COMPANION_ERROR",
                    "message": f"the meeting companion failed: {type(exc).__name__}: {exc}"}

    def stop(self) -> None:
        server, self._server = self._server, None
        if server is not None:
            server.shutdown()
            server.server_close()
        try:
            current = json.loads((self.state_dir / CONTROL_FILE).read_text(encoding="utf-8"))
            if isinstance(current, dict) and current.get("token") == self._token:
                (self.state_dir / CONTROL_FILE).unlink()
        except (OSError, ValueError):
            pass
