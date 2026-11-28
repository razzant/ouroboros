"""Channel transport and POSIX letters-door custody for linked Ouroboroses.

A cross-machine link speaks only to its own derived POSIX Unix-domain socket inside an owner-only (0700)
door directory (``door.sock``, ``l-<link16>.sock``, ``r-<link16>-<session12>.sock``). Queue storage,
channel fault classification, and event-driven flushing live in ``ouroboros/peer_channel_queue.py``.
"""
from __future__ import annotations

import contextlib
import errno
import hashlib
import logging
import os
import pathlib
import re
import socket
import stat
import threading
import time
from typing import Any, Callable, Mapping

from ouroboros.platform_layer import IS_MACOS, IS_WINDOWS
from ouroboros.utils import utc_now_iso

log = logging.getLogger(__name__)

DOOR_SOCKET_NAME = "door.sock"
# Reserve room for the longest derived socket name (r-<16hex>-<12hex>.sock = 36 bytes) under 100 bytes
# (macOS sun_path is 104 bytes, Linux sun_path is 108 bytes).
MAX_SOCKET_PATH_BYTES = 100
_LONGEST_SOCKET_NAME_BYTES = len("r-" + "0" * 16 + "-" + "0" * 12 + ".sock")  # 36 bytes
SUN_PATH_LIMIT = 103 if IS_MACOS else 107
SESSION_RE = re.compile(r"^[0-9a-f]{12}$")
_RECEIVER_SOCK_RE = re.compile(r"^r-([0-9a-f]{16})-([0-9a-f]{12})\.sock$")
ROLE_OPENER = "opener"
ROLE_RECEIVER = "receiver"
TRANSPORT_URL = "url"
TRANSPORT_CHANNEL = "channel"
CHANNEL_ADDRESS_LABEL = "Another computer (channel)"
CHANNEL_BASE_URL = "http://peer.localhost"
_CHANNEL_ANNOUNCE_TIMEOUT_SEC = 15.0
_DOOR_STATUS_LOCK = threading.Lock()
_DOOR_STATUS: dict[str, dict[str, Any]] = {}
_CONNECT_ERRNOS = frozenset({errno.ENOENT, errno.ECONNREFUSED, errno.EACCES})


def is_door_supported() -> bool:
    """Whether this platform supports the POSIX Unix-domain letters door."""
    return not IS_WINDOWS and hasattr(socket, "AF_UNIX") and hasattr(os, "getuid")


def _canonical_root(root: Any) -> str:
    return os.path.realpath(str(pathlib.Path(root)))


def _door_tag(canonical_root: str) -> str:
    return hashlib.sha256(canonical_root.encode("utf-8")).hexdigest()[:16]


def link_tag(link_id: str) -> str:
    """16 lowercase hex characters derived from ``sha256(link_id)``."""
    return hashlib.sha256(str(link_id).encode("utf-8")).hexdigest()[:16]


def validate_session(value: Any, *, field: str = "session") -> str:
    """Strictly require a 12-character lowercase hex session identifier."""
    from ouroboros.peer_links import PeerLinkError

    if not isinstance(value, str) or not SESSION_RE.fullmatch(value):
        raise PeerLinkError("invalid_session", f"{field} must be 12 lowercase hexadecimal characters", 400)
    return value


def _session_int(value: str) -> int:
    return int(value, 16) if value and SESSION_RE.fullmatch(value) else -1


def next_channel_session(prev_session: str = "") -> str:
    """Strictly increasing 12-hex session number: ``max(now_ms, prev + 1)``."""
    now_ms = int(time.time() * 1000)
    prev_val = _session_int(str(prev_session or ""))
    nxt = max(now_ms, prev_val + 1)
    if nxt > 0xFFFFFFFFFFFF:
        from ouroboros.peer_links import PeerLinkError

        raise PeerLinkError("channel_session_exhausted", "introduce the linked Ouroboros again", 409)
    return f"{nxt:012x}"


def allocate_channel_session(root: Any, link_id: str) -> str:
    """Choose the next strictly increasing 12-hex session number for ``link_id`` on the opening side."""
    from ouroboros import peer_links

    row = peer_links.read_registry(root)["links"].get(str(link_id or "")) or {}
    return next_channel_session(str(row.get("channel_session") or ""))


def _fits_door_dir(directory: pathlib.Path) -> bool:
    longest = len(str(directory).encode("utf-8")) + 1 + _LONGEST_SOCKET_NAME_BYTES
    return longest <= MAX_SOCKET_PATH_BYTES and longest <= SUN_PATH_LIMIT


def resolve_door_dir(root: Any) -> pathlib.Path:
    """Deterministic POSIX peer door directory for ``root`` without creating it on disk.

    Candidate order:
    1. ``<data_dir>/state/peer`` if its longest derived socket path (``r-<16hex>-<12hex>.sock``) is ``<= 100`` bytes.
    2. ``/tmp/ouroboros-peer-<uid>/<16-hex-of-realpath(data_dir)>``.
    """
    from ouroboros.peer_links import PeerLinkError

    if not is_door_supported():
        raise PeerLinkError("peer_door_unsupported", "the letters door is supported on POSIX systems only", 501)
    canonical = _canonical_root(root)
    tag = _door_tag(canonical)
    candidate_state = pathlib.Path(canonical) / "state" / "peer"
    if _fits_door_dir(candidate_state):
        return candidate_state
    uid = os.getuid()
    return pathlib.Path("/tmp") / f"ouroboros-peer-{uid}" / tag


def door_socket_path(root: Any) -> pathlib.Path:
    """Deterministic socket path ``<peer_dir>/door.sock`` for this installation's letters door."""
    return resolve_door_dir(root) / DOOR_SOCKET_NAME


def opener_socket_path(root: Any, link_id: str) -> pathlib.Path:
    """Derived opener-side channel socket ``<peer_dir>/l-<link16>.sock``."""
    return resolve_door_dir(root) / f"l-{link_tag(link_id)}.sock"


def receiver_socket_path(root: Any, link_id: str, session: str) -> pathlib.Path:
    """Derived receiver-side channel socket ``<peer_dir>/r-<link16>-<session12>.sock``."""
    sess = validate_session(session)
    return resolve_door_dir(root) / f"r-{link_tag(link_id)}-{sess}.sock"


def _verify_dir_lstat(path: pathlib.Path, *, uid: int, label: str = "door directory") -> os.stat_result:
    """Strictly verify via ``lstat`` that ``path`` is a real directory, not a symlink, owned by ``uid``, mode 0700."""
    from ouroboros.peer_links import PeerLinkError

    try:
        st = os.lstat(path)
    except FileNotFoundError as exc:
        raise PeerLinkError("peer_door_dir_invalid", f"{label} does not exist: {path}", 503) from exc
    except OSError as exc:
        raise PeerLinkError("peer_door_dir_invalid", f"{label} cannot be inspected: {exc}", 503) from exc
    if stat.S_ISLNK(st.st_mode):
        raise PeerLinkError("peer_door_dir_invalid", f"{label} must not be a symlink", 503)
    if not stat.S_ISDIR(st.st_mode):
        raise PeerLinkError("peer_door_dir_invalid", f"{label} is not a directory", 503)
    if st.st_uid != uid:
        raise PeerLinkError("peer_door_dir_invalid", f"{label} is not owned by the current user", 503)
    if stat.S_IMODE(st.st_mode) != 0o700:
        raise PeerLinkError("peer_door_dir_invalid", f"{label} must have mode 0700", 503)
    return st


def _create_or_verify_0700_dir(path: pathlib.Path, *, uid: int, create: bool, label: str) -> os.stat_result | None:
    from ouroboros.peer_links import PeerLinkError

    created = False
    if create:
        try:
            os.mkdir(path, 0o700)
            created = True
        except FileExistsError:
            pass
        except OSError as exc:
            raise PeerLinkError("peer_door_dir_invalid", f"{label} could not be created: {exc}", 503) from exc
        else:
            try:
                os.chmod(path, 0o700)
            except OSError as exc:
                raise PeerLinkError("peer_door_dir_invalid", f"{label} permissions could not be set: {exc}", 503) from exc
    found = _verify_dir_lstat(path, uid=uid, label=label)
    return found if created else None


def _ensure_door_dir(root: Any, *, create: bool) -> tuple[pathlib.Path, os.stat_result | None]:
    """Return the verified directory and a cleanup identity only for a fallback we just made."""
    from ouroboros.peer_links import PeerLinkError

    door_dir = resolve_door_dir(root)
    uid = os.getuid()
    canonical_state = pathlib.Path(_canonical_root(root)) / "state" / "peer"
    if door_dir == canonical_state:
        if create:
            try:
                door_dir.parent.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise PeerLinkError("peer_door_dir_invalid", f"door parent could not be created: {exc}", 503) from exc
        _create_or_verify_0700_dir(door_dir, uid=uid, create=create, label="door directory")
        return door_dir, None
    _create_or_verify_0700_dir(door_dir.parent, uid=uid, create=create, label="door parent directory")
    return door_dir, _create_or_verify_0700_dir(door_dir, uid=uid, create=create, label="door directory")


def ensure_door_dir(root: Any, *, create: bool = True) -> pathlib.Path:
    """Resolve, optionally create (0700), and lstat-verify this installation's letters-door directory."""
    return _ensure_door_dir(root, create=create)[0]


def verify_door_dir(root: Any) -> pathlib.Path:
    """Verify an existing letters-door directory via ``lstat`` without creating or modifying it."""
    return ensure_door_dir(root, create=False)


def verify_channel_socket(root: Any, sock_path: pathlib.Path, *, require_exists: bool = True) -> pathlib.Path:
    """Verify ``door_dir`` (0700, owned, not symlink) and ``sock_path`` via ``lstat`` (socket, not symlink, owned)."""
    from ouroboros.peer_links import PeerLinkError

    if not is_door_supported():
        raise PeerLinkError("peer_door_unsupported", "channel links require the POSIX letters door", 501)
    door_dir = verify_door_dir(root)
    uid = os.getuid()
    real_door = pathlib.Path(os.path.realpath(str(door_dir)))
    if sock_path.parent != door_dir or sock_path.name == DOOR_SOCKET_NAME:
        raise PeerLinkError("peer_channel_refused", "derived channel socket path is outside the door directory", 400)
    try:
        st = os.lstat(sock_path)
    except FileNotFoundError as exc:
        if require_exists:
            raise PeerLinkError("channel_end_missing", f"channel socket does not exist: {sock_path.name}", 400) from exc
        return sock_path
    except OSError as exc:
        raise PeerLinkError("peer_channel_refused", f"channel socket cannot be inspected: {exc}", 400) from exc
    if stat.S_ISLNK(st.st_mode):
        raise PeerLinkError("peer_channel_refused", "channel socket must not be a symlink", 400)
    if not stat.S_ISSOCK(st.st_mode):
        raise PeerLinkError("peer_channel_refused", "channel end must be a Unix socket", 400)
    if st.st_uid != uid:
        raise PeerLinkError("peer_channel_refused", "channel socket is not owned by the current user", 400)
    if pathlib.Path(os.path.realpath(str(sock_path))).parent != real_door:
        raise PeerLinkError("peer_channel_refused", "channel socket resolves outside the door directory", 400)
    return sock_path


def register_door_status(root: Any, *, listening: bool, door_dir: str = "", socket_path: str = "",
                         error_code: str = "", error: str = "") -> None:
    with _DOOR_STATUS_LOCK:
        _DOOR_STATUS[_canonical_root(root)] = {
            "listening": bool(listening), "door_dir": str(door_dir or ""),
            "socket_path": str(socket_path or ""), "error_code": str(error_code or ""),
            "error": str(error or ""),
        }


def unregister_door_status(root: Any) -> None:
    with _DOOR_STATUS_LOCK:
        _DOOR_STATUS.pop(_canonical_root(root), None)


def door_status(root: Any) -> dict[str, Any]:
    """Current in-process status of the letters door for ``root``."""
    with _DOOR_STATUS_LOCK:
        row = _DOOR_STATUS.get(_canonical_root(root))
    if row is not None:
        return dict(row)
    code = "peer_door_unsupported" if not is_door_supported() else "peer_door_not_running"
    msg = ("the letters door is supported on POSIX systems only" if code == "peer_door_unsupported"
           else "the letters door is not running on this installation")
    return {"listening": False, "door_dir": "", "socket_path": "", "error_code": code, "error": msg}


def require_door_running(root: Any) -> tuple[str, str]:
    """Require a live, lstat-verified letters door for ``root``; return ``(door_dir, door_socket)``."""
    from ouroboros.peer_links import PeerLinkError

    if not is_door_supported():
        raise PeerLinkError("peer_door_unsupported", "the letters door is supported on POSIX systems only", 501)
    status = door_status(root)
    if not status.get("listening"):
        code = str(status.get("error_code") or "peer_door_not_running")
        detail = str(status.get("error") or "the letters door is not running on this installation")
        raise PeerLinkError(code, detail, 501 if code == "peer_door_unsupported" else 503)
    verified_dir = verify_door_dir(root)
    sock = verified_dir / DOOR_SOCKET_NAME
    try:
        st = os.lstat(sock)
    except OSError as exc:
        raise PeerLinkError("peer_door_not_running", "the letters door socket is missing", 503) from exc
    if not stat.S_ISSOCK(st.st_mode) or st.st_uid != os.getuid():
        raise PeerLinkError("peer_door_dir_invalid", "the letters door socket is invalid", 503)
    return str(verified_dir), str(sock)


def is_channel_link(link: Any) -> bool:
    """Whether an ``OutgoingLink`` or registry link row uses the channel transport."""
    if isinstance(link, Mapping):
        return str(link.get("transport") or TRANSPORT_URL) == TRANSPORT_CHANNEL
    return str(getattr(link, "transport", TRANSPORT_URL) or TRANSPORT_URL) == TRANSPORT_CHANNEL


def link_identity(link: Any) -> tuple[str, str, str, str]:
    """Stable identity tuple for a link (excludes the channel socket/session so reconnects preserve observation)."""
    if is_channel_link(link):
        return (str(link.link_id), str(link.place_id), TRANSPORT_CHANNEL, str(link.peer_install_id))
    return (str(link.link_id), str(link.place_id), str(link.peer_url), str(link.peer_install_id))


def channel_socket_for_link(link: Any) -> pathlib.Path:
    """Derive the local Unix socket path for a channel ``OutgoingLink``."""
    from ouroboros.peer_links import PeerLinkError

    if isinstance(link, Mapping):
        raise TypeError("channel_socket_for_link requires an OutgoingLink")
    role = str(getattr(link, "channel_role", "") or "")
    if role == ROLE_OPENER:
        return opener_socket_path(link.drive_root, link.link_id)
    if role == ROLE_RECEIVER:
        session = str(getattr(link, "channel_session", "") or "")
        return receiver_socket_path(link.drive_root, link.link_id, session)
    raise PeerLinkError("peer_channel_refused", "channel link has an unknown role", 400)


def channel_socket_for_row(root: Any, link_id: str, row: Mapping[str, Any]) -> pathlib.Path:
    """Derive the local Unix socket path for a registry link row."""
    from ouroboros.peer_links import PeerLinkError

    role = str(row.get("channel_role") or "")
    if role == ROLE_OPENER:
        return opener_socket_path(root, link_id)
    if role == ROLE_RECEIVER:
        return receiver_socket_path(root, link_id, str(row.get("channel_session") or ""))
    raise PeerLinkError("peer_channel_refused", "channel link has an unknown role", 400)


def link_target_valid(link: Any) -> tuple[bool, str, str]:
    """Validate an ``OutgoingLink`` transport target: strict loopback URL or valid channel role/dir/lstat.

    For a channel link, a temporarily absent socket file is allowed here so the send layer classifies it as
    pre-wire ``channel_unavailable`` (waiting for connection without burning passes); an invalid door dir,
    invalid session/role, symlink, non-socket or foreign owner is refused immediately.
    """
    from ouroboros import peer_links

    if is_channel_link(link):
        if str(getattr(link, "peer_url", "") or "").strip():
            return False, "peer_channel_refused", "a channel link must not also carry an HTTP URL; nothing was sent"
        try:
            sock = channel_socket_for_link(link)
            verify_channel_socket(link.drive_root, sock, require_exists=False)
        except peer_links.PeerLinkError as exc:
            return False, exc.code, f"{exc}; nothing was sent"
        return True, "", ""
    if not peer_links.loopback_url(link.peer_url):
        return False, "peer_url_refused", (
            "URL links must be loopback addresses on this computer; another computer uses a channel invitation; "
            "nothing was sent"
        )
    return True, "", ""


def _make_channel_transport(resolver: Callable[[], str | pathlib.Path]) -> Any:
    """Build an ``httpx.BaseTransport`` that resolves the current channel socket on every request."""
    import httpx

    class _ClosingByteStream(httpx.SyncByteStream):
        def __init__(self, inner: Any, transport: httpx.HTTPTransport) -> None:
            self._inner = inner
            self._transport = transport

        def __iter__(self):
            yield from self._inner

        def close(self) -> None:
            try:
                self._inner.close()
            finally:
                self._transport.close()

    class _DynamicChannelTransport(httpx.BaseTransport):
        def handle_request(self, request: httpx.Request) -> httpx.Response:
            sock_path = str(resolver() or "")
            if not sock_path:
                raise FileNotFoundError("channel socket path is empty")
            transport = httpx.HTTPTransport(
                uds=sock_path,
                limits=httpx.Limits(max_keepalive_connections=0),
            )
            try:
                response = transport.handle_request(request)
            except BaseException:
                transport.close()
                raise
            response.stream = _ClosingByteStream(response.stream, transport)
            return response

    return _DynamicChannelTransport()


def peer_http_client(*, peer_url: str = "", channel_path: str | pathlib.Path = "",
                     channel_resolver: Callable[[], str | pathlib.Path] | None = None,
                     timeout_sec: float = 20.0, connect_timeout_sec: float = 5.0) -> Any:
    """Create a synchronous ``httpx.Client`` for a loopback URL or a per-request Unix-domain channel socket."""
    import httpx

    timeout = httpx.Timeout(timeout_sec, connect=connect_timeout_sec)
    if channel_resolver is not None or channel_path:
        resolver = channel_resolver or (lambda fixed=str(channel_path): fixed)
        return httpx.Client(transport=_make_channel_transport(resolver), base_url=CHANNEL_BASE_URL,
                            trust_env=False, follow_redirects=False, timeout=timeout)
    return httpx.Client(base_url=str(peer_url).rstrip("/"), trust_env=False,
                        follow_redirects=False, timeout=timeout)


def _cleanup_stale_receiver_sockets(door_dir: pathlib.Path, link_id: str, *,
                                    older_than_session: str = "", uid: int) -> None:
    """Remove ``r-<link16>-<s>.sock`` files of ``link_id`` (only ``s < older_than_session`` when given)."""
    want_tag = link_tag(link_id)
    cutoff = _session_int(older_than_session) if older_than_session else -1
    try:
        entries = list(door_dir.iterdir())
    except OSError:
        return
    for entry in entries:
        name = entry.name
        if name == DOOR_SOCKET_NAME:
            continue
        match = _RECEIVER_SOCK_RE.fullmatch(name)
        if not match or match.group(1) != want_tag:
            continue
        if older_than_session and _session_int(match.group(2)) >= cutoff:
            continue
        try:
            st = os.lstat(entry)
            if stat.S_ISSOCK(st.st_mode) and st.st_uid == uid:
                entry.unlink(missing_ok=True)
        except OSError:
            log.debug("stale channel socket %s could not be removed", entry, exc_info=True)


def set_channel_session(root: Any, link_id: str, session: Any) -> dict[str, Any]:
    """Handle an incoming ``POST /api/letters/channel`` announcement (R6).

    - ``session < current``: refused ``channel_session_stale`` (409), nothing changes.
    - ``session == current``: accepted on either role ("I am here again"); nothing changes, owed/queued resumed.
    - ``session > current`` on opener side (meaning receiver tried to advance): refused ``channel_session_stale`` (409).
    - ``session > current`` on receiver side: verifies ``r-<link16>-<session>.sock`` via ``lstat``, atomically
      advances ``channel_session``, removes only older ``r-<link16>-<s>.sock`` of this link, and resumes owed/queued.
    """
    from ouroboros import peer_links

    link_id = peer_links._valid_link_id(link_id)
    sess = validate_session(session)
    sess_val = _session_int(sess)

    # Pre-read to validate existence/role/direction and lstat-check the derived socket before locking registry.
    row = peer_links.read_registry(root)["links"].get(link_id)
    if row is None or row.get("revoked_at"):
        raise peer_links.PeerLinkError("link_unknown", "no such active link", 401)
    if not is_channel_link(row):
        raise peer_links.PeerLinkError("peer_channel_refused", "this link is not a channel link", 400)
    role = str(row.get("channel_role") or "")
    cur = str(row.get("channel_session") or "")
    cur_val = _session_int(cur)
    if cur_val >= 0 and sess_val < cur_val:
        raise peer_links.PeerLinkError("channel_session_stale",
                                       "announced channel session is older than the active session", 409)
    if sess_val > cur_val and role != ROLE_RECEIVER:
        raise peer_links.PeerLinkError("channel_session_stale",
                                       "only the opening side may advance the channel session", 409)

    door_dir: pathlib.Path | None = None
    if sess_val > cur_val:
        verify_channel_socket(root, receiver_socket_path(root, link_id, sess), require_exists=True)
        door_dir = verify_door_dir(root)

    def apply(document: dict) -> dict[str, Any]:
        live = document["links"].get(link_id)
        if live is None or live.get("revoked_at"):
            raise peer_links.PeerLinkError("link_unknown", "no such active link", 401)
        if not is_channel_link(live):
            raise peer_links.PeerLinkError("peer_channel_refused", "this link is not a channel link", 400)
        live_role = str(live.get("channel_role") or "")
        live_cur = str(live.get("channel_session") or "")
        live_val = _session_int(live_cur)
        if live_val >= 0 and sess_val < live_val:
            raise peer_links.PeerLinkError("channel_session_stale",
                                           "announced channel session is older than the active session", 409)
        if sess_val == live_val:
            live["channel_announce_seq"] = int(live.get("channel_announce_seq") or 0) + 1
            return {"ok": True, "link_id": link_id, "session": sess, "changed": False}
        if live_role != ROLE_RECEIVER:
            raise peer_links.PeerLinkError("channel_session_stale",
                                           "only the opening side may advance the channel session", 409)
        live["channel_session"] = sess
        live["channel_updated_at"] = utc_now_iso()
        live["channel_announce_seq"] = int(live.get("channel_announce_seq") or 0) + 1
        return {"ok": True, "link_id": link_id, "session": sess, "changed": True}

    result = peer_links.update_registry(root, apply)
    if result.get("changed") and door_dir is not None:
        _cleanup_stale_receiver_sockets(door_dir, link_id, older_than_session=sess, uid=os.getuid())
    return result


def remove_link_channel_sockets(root: Any, link_id: str, row: Mapping[str, Any]) -> None:
    """Best-effort cleanup of this link's derived sockets in our verified door dir on revocation."""
    if not is_door_supported() or not is_channel_link(row):
        return
    try:
        door_dir = verify_door_dir(root)
    except Exception:  # noqa: BLE001 - revocation never fails on door dir inspection
        return
    uid = os.getuid()
    if row.get("channel_role") == ROLE_RECEIVER:
        _cleanup_stale_receiver_sockets(door_dir, link_id, older_than_session="", uid=uid)
    elif row.get("channel_role") == ROLE_OPENER:
        candidate = door_dir / f"l-{link_tag(link_id)}.sock"
        with contextlib.suppress(OSError):
            st = os.lstat(candidate)
            if stat.S_ISSOCK(st.st_mode) and st.st_uid == uid:
                candidate.unlink(missing_ok=True)


def announce_channel(root: Any, link_id: str, session: Any = "") -> dict[str, Any]:
    """Internal R6 channel announcement: POST ``{"session"}`` to peer's ``POST /api/letters/channel``.

    - Opener may announce a strictly greater session (new SSH session) or its current session ("I am back").
    - Receiver may announce only its current session ("I am back").
    - Lower session or greater session from receiver is refused ``channel_session_stale`` (409).
    - On 200: opener records the new session if greater, and this side resumes owed replies & queued messages.
    """
    import httpx

    from ouroboros import peer_links

    link_id = peer_links._valid_link_id(link_id)
    row = peer_links.read_registry(root)["links"].get(link_id)
    if row is None or row.get("revoked_at"):
        raise peer_links.PeerLinkError("link_unknown", "no such active link", 401)
    if not is_channel_link(row):
        raise peer_links.PeerLinkError("peer_channel_refused", "this link is not a channel link", 400)
    role = str(row.get("channel_role") or "")
    cur = str(row.get("channel_session") or "")
    sess = validate_session(session if session != "" else cur)
    cur_val, sess_val = _session_int(cur), _session_int(sess)
    if cur_val >= 0 and sess_val < cur_val:
        raise peer_links.PeerLinkError("channel_session_stale",
                                       "session number must not be older than the active session", 409)
    if role == ROLE_RECEIVER and cur_val >= 0 and sess_val > cur_val:
        raise peer_links.PeerLinkError("channel_session_stale",
                                       "only the opening side may advance the channel session", 409)
    own_sock = verify_channel_socket(root, channel_socket_for_row(root, link_id, row), require_exists=True)
    secret = peer_links._read_secret(root, link_id)
    if not secret:
        raise peer_links.PeerLinkError("link_unknown", "this link has no key here", 401)
    headers = {
        peer_links.LINK_HEADER: f"{link_id}.{secret}",
        peer_links.INSTALL_HEADER: peer_links._current_install_id(root) or str(row.get("own_install_id") or ""),
        peer_links.PROTOCOL_HEADER: str(peer_links.PROTOCOL),
        "Content-Type": "application/json",
    }
    try:
        with peer_http_client(channel_path=own_sock, timeout_sec=_CHANNEL_ANNOUNCE_TIMEOUT_SEC) as client:
            response = client.post("/api/letters/channel", json={"session": sess}, headers=headers)
    except (httpx.HTTPError, OSError) as exc:
        raise peer_links.PeerLinkError(
            "channel_unreachable",
            f"the peer letters door did not answer over the channel: {type(exc).__name__}",
            502,
        ) from exc
    if response.status_code != 200:
        try:
            body = response.json()
        except ValueError:
            body = {}
        code = str(body.get("refusal_code") or body.get("error") or "") if isinstance(body, dict) else ""
        detail = str(body.get("detail") or code or response.status_code) if isinstance(body, dict) else str(response.status_code)
        if response.status_code == 401 and code == "link_unknown":
            with contextlib.suppress(Exception):
                peer_links.mark_remote_unknown(root, link_id)
        status = response.status_code if 400 <= response.status_code < 500 else 502
        raise peer_links.PeerLinkError(code or "channel_announce_refused", detail, status)
    try:
        ack = response.json()
    except ValueError as exc:
        raise peer_links.PeerLinkError(
            "malformed_response", "the peer letters door answered channel announce without JSON", 502,
        ) from exc
    if (not isinstance(ack, dict) or ack.get("ok") is not True
            or ack.get("link_id") != link_id or ack.get("session") != sess):
        raise peer_links.PeerLinkError(
            "malformed_response", "the peer letters door returned an invalid channel announce acknowledgement", 502,
        )

    def record_announce(document: dict) -> None:
        current = document["links"].get(link_id)
        if isinstance(current, dict) and not current.get("revoked_at"):
            if role == ROLE_OPENER and sess_val > _session_int(str(current.get("channel_session") or "")):
                current["channel_session"] = sess
                current["channel_updated_at"] = utc_now_iso()
            current["channel_announce_seq"] = int(current.get("channel_announce_seq") or 0) + 1

    peer_links.update_registry(root, record_announce)

    from ouroboros import peer_channel_queue

    peer_channel_queue.trigger_link_flush(root, link_id)
    return {"ok": True, "link_id": link_id, "session": sess}


def announce_existing_channels_at_start(root: Any) -> None:
    """At installation startup, announce the current session for each confirmed channel link whose end exists."""
    from ouroboros import peer_links

    if not is_door_supported():
        return
    try:
        links = peer_links.read_registry(root)["links"]
    except Exception:  # noqa: BLE001 - startup never fails on unreadable registry
        return
    targets: list[tuple[str, str]] = []
    for lid, row in links.items():
        if (not isinstance(row, dict) or not is_channel_link(row) or row.get("revoked_at")
                or not row.get("code_confirmed_at") or row.get("remote_unknown_at")):
            continue
        cur = str(row.get("channel_session") or "")
        if not SESSION_RE.fullmatch(cur):
            continue
        try:
            sock = channel_socket_for_row(root, lid, row)
            if sock.exists():
                targets.append((lid, cur))
        except Exception:  # noqa: BLE001
            continue
    if not targets:
        return

    def _run() -> None:
        from ouroboros.install_identity import acting_as

        with acting_as(root):
            for lid, cur in targets:
                try:
                    announce_channel(root, lid, cur)
                except Exception:  # noqa: BLE001 - peer may be offline at our startup
                    log.debug("startup channel announce for link %s did not reach peer", lid, exc_info=True)

    threading.Thread(target=_run, name="peer-channel-start-announce", daemon=True).start()


def is_channel_connect_failure(exc: BaseException) -> bool:
    """Whether ``exc`` proves nothing left on our channel socket (absent file, ECONNREFUSED, EACCES, ConnectTimeout)."""
    import httpx

    if isinstance(exc, (FileNotFoundError, ConnectionRefusedError, PermissionError,
                        httpx.ConnectError, httpx.ConnectTimeout)):
        return True
    if isinstance(exc, OSError) and exc.errno in _CONNECT_ERRNOS:
        return True
    cause = exc.__cause__
    if isinstance(cause, BaseException) and cause is not exc:
        return is_channel_connect_failure(cause)
    return False
