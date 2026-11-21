"""Pairing the Telegram bridge with its owner by a one-time code, and the identity of a bot token.

Standard library only and free of relative imports: the core loads this one file by its path in the skill
(the manifest's ``connection_card`` names ``lib/telegram_pairing.py:<function>``), whether the bridge runs or not.

Pairing. A connection card (the setup wizard's phone card, or the skill's card later) mints a code
(``new_code``), shows ``https://t.me/<bot>?start=<code>`` as a QR code, an Open in Telegram button and a copyable
``/start <code>``, and saves the code in Settings as ``TELEGRAM_PAIRING_CODE`` (the manifest's secret field
``pairingCode``), then grants that key to this skill: a newly saved key is one the bridge may read only with its
grant. The bridge reads it as it needs it, so a saved, replaced or cleared code applies at once. A code is pending
while the bridge can read it and has not consumed it; the bridge records the SHA-256 digest of the code it consumed
in ``pairing_consumed.json`` in its own state folder. While a code is pending only a private chat's ``/start <code>``
binds the owner (``first_writer_may_bind``); with none, or one this skill may not read, the first positive private
chat binds, as before.

Identity. ``bot_identity`` sends one bare ``getMe`` and says which bot a token names, or why it could not:
Telegram refused the token, or Telegram could not be reached from here. Nothing it returns carries the token or
the URL, it raises nothing, logs nothing, writes nothing and calls no other method.
"""
from __future__ import annotations

import contextlib
import hashlib
import hmac
import json
import os
import re
import secrets
import ssl
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Optional

CODE_KEY = "TELEGRAM_PAIRING_CODE"
CLOUD_API_BASE = "https://api.telegram.org"
CONSUMED_FILE = "pairing_consumed.json"
SETTINGS_FILE = "settings.json"  # the skill's own settings (scripts/telegram_settings.py)
_CODE = re.compile(r"[A-Za-z0-9_-]{1,64}")  # what Telegram carries in a start parameter
_TOKEN = re.compile(r"(\d{1,20}):[A-Za-z0-9_-]{1,128}")
_USERNAME = re.compile(r"[A-Za-z0-9_]{5,32}")
_DIGEST = re.compile(r"[0-9a-f]{64}")
_BASE_RULE = "TELEGRAM_API_BASE must be http(s)://host[:port] with no path, query, fragment or credentials."
_DEFAULT_PORTS = {"http": 80, "https": 443}
_MAX_ANSWER_BYTES = 65536


def normalize_code(value: Any) -> str:
    """``value`` when a ``start`` parameter can carry it exactly (1-64 letters, digits, ``_`` or ``-``), else ""."""
    return value if isinstance(value, str) and _CODE.fullmatch(value) else ""


def new_code() -> str:
    """A fresh pairing code: 32 characters, 192 random bits."""
    return secrets.token_urlsafe(24)


def code_digest(code: str) -> str:
    return hashlib.sha256(str(code).encode("utf-8")).hexdigest()


def same_digest(left: Any, right: Any) -> bool:
    """Whether two SHA-256 hex digests are the same one, compared in constant time."""
    return (isinstance(left, str) and isinstance(right, str) and bool(_DIGEST.fullmatch(left))
            and bool(_DIGEST.fullmatch(right)) and hmac.compare_digest(left, right))


def start_command(text: Any) -> Optional[str]:
    """The payload of ``/start``, ``/start@bot``, ``/start <payload>``: "" when bare, None for any other text.
    The payload is returned as sent; ``normalize_code`` decides whether it can be a code."""
    parts = text.strip().split(None, 1) if isinstance(text, str) else []
    head = parts[0].lower() if parts else ""
    if head != "/start" and not (head.startswith("/start@") and len(head) > 7 and head.count("@") == 1):
        return None
    return parts[1].strip() if len(parts) > 1 else ""


def canonical_api_base(value: Any = None) -> str:
    """The Bot API origin to call: the cloud when empty, else a declared ``TELEGRAM_API_BASE`` accepted as the
    bridge accepts it (``scripts/telegram_settings.validate_telegram_api_base``). Raises ValueError otherwise."""
    text = str(value or "").strip().rstrip("/")
    if not text:
        return CLOUD_API_BASE
    try:
        parts = urllib.parse.urlsplit(text)
        port = parts.port
    except ValueError:
        raise ValueError(_BASE_RULE) from None
    host = parts.hostname or ""
    if ("#" in text or "?" in text or parts.scheme not in _DEFAULT_PORTS or not host or parts.path
            or parts.username or parts.password or (port is not None and not 0 < port <= 65535)):
        raise ValueError(_BASE_RULE)
    origin = f"{parts.scheme}://{f'[{host}]' if ':' in host else host}"
    return origin if port in (None, _DEFAULT_PORTS[parts.scheme]) else f"{origin}:{port}"


def saved_api_base(state_dir: Any) -> Optional[str]:
    """The ``TELEGRAM_API_BASE`` saved in the skill's own settings, or None (the cloud) when none is saved."""
    _status, settings = _read_json(Path(state_dir) / SETTINGS_FILE)
    value = settings.get("TELEGRAM_API_BASE")
    return value.strip() if isinstance(value, str) and value.strip() else None


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001 — urllib's signature
        return None  # never followed: the token travels in the path, so a redirect ends as a refused answer


def _tls() -> ssl.SSLContext:
    try:
        import certifi  # the CA bundle the bridge's own HTTP client trusts, when this Python has it
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


def bot_identity(token: Any, api_base: Any = None, *, timeout: float = 8.0) -> dict:
    """Which bot ``token`` names, from one bare ``getMe`` to ``api_base`` (the cloud when None).

    ``{"ok": True, "username": str, "id": int}``; ``{"ok": False, "cause": "refused"}`` when Telegram rejected
    the token (HTTP 401 or 404, or ``ok: false`` with error 401) or it cannot be a token at all;
    ``{"ok": False, "cause": "unreachable"}`` for anything else: an invalid base, a transport failure, a
    timeout, a server error, a redirect (never followed) or an answer that names no bot of this token. Proxies of
    the environment (on macOS also the system's) apply as for any Python request; the skill's ``TELEGRAM_PROXY``
    does not, so an install that reaches Telegram only through it reads unreachable here.
    """
    text = token.strip() if isinstance(token, str) else ""
    matched = _TOKEN.fullmatch(text)
    if not matched:
        return {"ok": False, "cause": "refused"}
    try:
        url = f"{canonical_api_base(api_base)}/bot{text}/getMe"
        opener = urllib.request.build_opener(_NoRedirect(), urllib.request.HTTPSHandler(context=_tls()))
        with opener.open(urllib.request.Request(url, headers={"Accept": "application/json"}), timeout=timeout) as answer:
            status, raw = int(answer.status), answer.read(_MAX_ANSWER_BYTES + 1)
    except urllib.error.HTTPError as answer:  # an error status or a redirect: its body may still say why
        status, raw = int(answer.code), b""
        with contextlib.suppress(Exception), answer:
            raw = answer.read(_MAX_ANSWER_BYTES + 1)
    except Exception:  # noqa: BLE001 — every other failure is "not reachable from here", never its text
        return {"ok": False, "cause": "unreachable"}
    try:
        payload = json.loads(raw.decode("utf-8")) if len(raw) <= _MAX_ANSWER_BYTES else None
    except ValueError:
        payload = None
    payload = payload if isinstance(payload, dict) else {}
    if status in (401, 404) or (payload.get("ok") is False and payload.get("error_code") == 401):
        return {"ok": False, "cause": "refused"}
    result = payload.get("result") if status == 200 and payload.get("ok") is True else None
    bot = result.get("id") if isinstance(result, dict) else None
    username = result.get("username") if isinstance(result, dict) else None
    if (type(bot) is not int or bot != int(matched.group(1)) or not isinstance(username, str)
            or not _USERNAME.fullmatch(username)):
        return {"ok": False, "cause": "unreachable"}
    return {"ok": True, "username": username, "id": bot}


def username_ok(value: Any) -> bool:
    """Whether ``value`` is a Telegram username a ``t.me`` link may carry as is."""
    return isinstance(value, str) and bool(_USERNAME.fullmatch(value))


def _read_json(path: Path) -> tuple[str, dict]:
    """``("missing" | "ok" | "unreadable", object)``: the object is ``{}`` unless the file holds one."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return "missing", {}
    except (OSError, ValueError):
        return "unreadable", {}
    return ("ok", value) if isinstance(value, dict) else ("unreadable", {})


def _consumed(state_dir: Any) -> str:
    """The digest of the code that bound the owner last; "" when none, or when its record cannot be read (a code
    then stays usable until it binds again, rather than leaving the owner unable to pair)."""
    _status, data = _read_json(Path(state_dir) / CONSUMED_FILE)
    digest = data.get("digest")
    return digest if isinstance(digest, str) and _DIGEST.fullmatch(digest) else ""


def pending(state_dir: Any, code: Any) -> bool:
    """Whether ``code`` (the saved ``TELEGRAM_PAIRING_CODE`` the bridge can read; "" when none) still waits."""
    code = normalize_code(code)
    return bool(code) and not same_digest(code_digest(code), _consumed(state_dir))


def matching_digest(state_dir: Any, code: Any, payload: Any) -> str:
    """The digest of ``code`` when a ``/start`` ``payload`` is that code and it still waits, else ""."""
    code, sent = normalize_code(code), normalize_code(payload)
    if not code or not sent or not pending(state_dir, code):
        return ""
    digest = code_digest(code)
    return digest if same_digest(code_digest(sent), digest) else ""


def record_consumed(state_dir: Any, digest: str) -> None:
    """The bridge's own record that the code of ``digest`` bound the owner: it never binds again. Replaces the
    file atomically; ``mkstemp`` creates it readable by this user alone."""
    path = Path(state_dir) / CONSUMED_FILE
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump({"digest": digest, "consumed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, stream)
        os.replace(temporary, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temporary)
        raise


def first_writer_may_bind(code_pending: bool) -> bool:
    """Whether the first positive private chat may bind the owner. Owner fork Q4, answer A: not while a pairing
    code waits (only the phone that opens its link binds). Answer B, the first writer still wins: ``return True``."""
    return not code_pending


def local_status(state_dir: Any, code: Any) -> dict:
    """``{"owner_bound", "pending"}`` from the skill's own files and the saved code the caller read."""
    _status, settings = _read_json(Path(state_dir) / SETTINGS_FILE)
    try:
        owner = int(str(settings.get("TELEGRAM_CHAT_ID") or "").strip() or 0)
    except ValueError:
        owner = 0
    return {"owner_bound": owner > 0, "pending": pending(state_dir, code)}


__all__ = [
    "CLOUD_API_BASE", "CODE_KEY", "CONSUMED_FILE", "bot_identity", "canonical_api_base", "code_digest",
    "first_writer_may_bind", "local_status", "matching_digest", "new_code", "normalize_code", "pending",
    "record_consumed", "same_digest", "saved_api_base", "start_command", "username_ok",
]
