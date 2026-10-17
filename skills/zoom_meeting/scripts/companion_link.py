"""How the zoom_meeting tools and the zoom_bot companion reach each other.

The tools run in a short-lived child per call; the companion is the one long-lived
process that owns the Zoom Meeting SDK. They meet in the skill's private state dir:

- a control channel: an AF_UNIX socket there, or loopback TCP when the socket path is
  too long for the platform. One JSON request and one JSON reply per connection,
  authenticated by a random token kept in an owner-only file beside it, so another
  local process cannot drive the meeting even when it can reach the port;
- ``status.json``: the companion's last published state. The tools read it when the
  companion does not answer, so a stopped companion is reported as stopped, with what
  it was doing, rather than as an empty meeting;
- the app identity both sides derive from the Zoom app's client id (never from the
  secret). It names the Presence account a binding admits.

Stdlib only and free of sibling imports: the companion imports it as a top-level
module, the plugin as ``.scripts.companion_link``.
"""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import json
import os
import secrets
import socket
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Optional

CONTROL_FILE = "control.json"
SOCKET_FILE = "control.sock"
STATUS_FILE = "status.json"
PROVIDER = "zoom"
# sun_path holds 104 bytes on macOS and 108 on Linux, terminator included.
UNIX_PATH_LIMIT = 100
MAX_MESSAGE_BYTES = 64 * 1024
_CONNECTION_TIMEOUT_SEC = 5.0

Handler = Callable[[str, Dict[str, Any]], Dict[str, Any]]


class ControlUnavailable(RuntimeError):
    """The companion did not answer: not running, not yet listening, or gone."""


def app_account_id(client_id: str) -> str:
    """The Presence account id of one Zoom Meeting SDK app.

    A binding admits exactly one provider account. The client id identifies the app,
    so its digest is stable across restarts and installs; the secret never enters it.
    """
    digest = hashlib.sha256(f"zoom-meeting-sdk-app:{client_id.strip()}".encode("utf-8")).hexdigest()
    return f"zoom-sdk-app-{digest[:16]}"


def conversation_key(account_id: str, meeting_number: str) -> str:
    """The host's conversation identity for one meeting (``presence_bindings.conversation_key``)."""
    return f"{PROVIDER}:{account_id}:{meeting_number}:0"


def write_private_json(path: Path, payload: Dict[str, Any]) -> None:
    """Atomically replace ``path`` with owner-only JSON (each writer its own temp file)."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
        os.replace(tmp, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            tmp.unlink()


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def write_status(state_dir: Path, payload: Dict[str, Any]) -> None:
    write_private_json(Path(state_dir) / STATUS_FILE, payload)


def read_status(state_dir: Path) -> Optional[Dict[str, Any]]:
    return read_json(Path(state_dir) / STATUS_FILE)


def _read_line(conn: socket.socket) -> bytes:
    data = b""
    while b"\n" not in data:
        chunk = conn.recv(4096)
        if not chunk:
            break
        data += chunk
        if len(data) > MAX_MESSAGE_BYTES:
            raise ValueError("control message too large")
    return data.split(b"\n", 1)[0]


class ControlServer:
    """The companion's end: one accept thread, requests served one at a time.

    Handlers only read state or schedule work on the SDK loop, so serving requests
    one by one keeps the companion's state changes ordered without extra locking.
    """

    def __init__(self, state_dir: Path, handler: Handler) -> None:
        self._state_dir = Path(state_dir)
        self._handler = handler
        self._token = secrets.token_urlsafe(32)
        self._sock: Optional[socket.socket] = None
        self._socket_path: Optional[Path] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.address: Dict[str, Any] = {}

    def start(self) -> None:
        path = self._state_dir / SOCKET_FILE
        if hasattr(socket, "AF_UNIX") and len(os.fsencode(str(path))) < UNIX_PATH_LIMIT:
            with contextlib.suppress(FileNotFoundError):
                path.unlink()
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            old_umask = os.umask(0o077)
            try:
                sock.bind(str(path))
            finally:
                os.umask(old_umask)
            self._socket_path = path
            self.address = {"transport": "unix", "path": str(path)}
        else:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(("127.0.0.1", 0))
            self.address = {"transport": "tcp", "host": "127.0.0.1", "port": sock.getsockname()[1]}
        sock.listen(8)
        sock.settimeout(0.5)
        self._sock = sock
        write_private_json(self._state_dir / CONTROL_FILE, {**self.address, "token": self._token, "pid": os.getpid()})
        self._thread = threading.Thread(target=self._serve, name="zoom-bot-control", daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            with conn:
                self._handle(conn)

    def _handle(self, conn: socket.socket) -> None:
        try:
            conn.settimeout(_CONNECTION_TIMEOUT_SEC)
            request = json.loads(_read_line(conn).decode("utf-8"))
            if not isinstance(request, dict) or not hmac.compare_digest(
                    str(request.get("token") or ""), self._token):
                reply: Dict[str, Any] = {"ok": False, "code": "ZOOM_CONTROL_DENIED",
                                         "error": "control request without the companion's token"}
            else:
                args = request.get("args") if isinstance(request.get("args"), dict) else {}
                reply = self._handler(str(request.get("op") or ""), args)
        except Exception as exc:  # one bad request never stops the channel
            reply = {"ok": False, "code": "ZOOM_CONTROL_FAILED", "error": type(exc).__name__}
        with contextlib.suppress(OSError):
            conn.sendall(json.dumps(reply, ensure_ascii=False).encode("utf-8") + b"\n")

    def close(self) -> None:
        self._stop.set()
        if self._sock is not None:
            with contextlib.suppress(OSError):
                self._sock.close()
        if self._thread is not None:
            self._thread.join(timeout=2)
        if self._socket_path is not None:
            with contextlib.suppress(FileNotFoundError):
                self._socket_path.unlink()
        control = read_json(self._state_dir / CONTROL_FILE)
        if control and control.get("token") == self._token:
            with contextlib.suppress(FileNotFoundError):
                (self._state_dir / CONTROL_FILE).unlink()


def control_request(state_dir: Path, op: str, args: Optional[Dict[str, Any]] = None, *,
                    timeout: float = 10.0) -> Dict[str, Any]:
    """The tools' end: one request to the companion, or ``ControlUnavailable``."""
    info = read_json(Path(state_dir) / CONTROL_FILE)
    if not info or not info.get("token"):
        raise ControlUnavailable("the companion has not opened its control channel")
    conn: Optional[socket.socket] = None
    try:
        if info.get("transport") == "unix":
            conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            conn.settimeout(timeout)
            conn.connect(str(info.get("path") or ""))
        else:
            conn = socket.create_connection(("127.0.0.1", int(info.get("port") or 0)), timeout=timeout)
        conn.settimeout(timeout)
        payload = {"token": info["token"], "op": op, "args": dict(args or {})}
        conn.sendall(json.dumps(payload, ensure_ascii=False).encode("utf-8") + b"\n")
        reply = json.loads(_read_line(conn).decode("utf-8"))
    except (OSError, ValueError) as exc:
        raise ControlUnavailable(f"{type(exc).__name__}: {exc}") from exc
    finally:
        if conn is not None:
            conn.close()
    if not isinstance(reply, dict):
        raise ControlUnavailable("the companion answered with something other than a JSON object")
    return reply
