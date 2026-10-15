"""Private settings authority shared by the Telegram extension and companion."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Optional

import httpx
from starlette.responses import JSONResponse

try:  # Package import from plugin.py.
    from .platform_support import acquire_file_lock, path_is_link_or_reparse, release_file_lock
except ImportError:  # Direct execution from scripts/companion.py.
    from platform_support import acquire_file_lock, path_is_link_or_reparse, release_file_lock


MINIAPP_MARKER_HEADER = "X-Ouroboros-Telegram-MiniApp"
_SETTINGS_NAME = "settings.json"
_LOCK_NAME = ".settings.lock"
_TRUE = frozenset({"on", "true", "1", "yes"})


class _SocksCredentialsFilter(logging.Filter):
    # HTTPcore 1.0 logs the SOCKS auth tuple at DEBUG, before authentication.
    # Redact that single diagnostic without changing transport or logger levels.
    _telegram_proxy_auth_filter = True

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        if message.startswith("setup_socks5_connection.started ") and " auth=" in message:
            record.msg = message.split(" auth=", 1)[0] + " auth=***"
            record.args = ()
        return True


_socks_logger = logging.getLogger("httpcore.socks")
if not any(getattr(item, "_telegram_proxy_auth_filter", False) for item in _socks_logger.filters):
    _socks_logger.addFilter(_SocksCredentialsFilter())


class TelegramSettingsError(RuntimeError):
    pass


def _settings_path(state_dir: Path) -> Path:
    root = Path(state_dir).expanduser()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path_is_link_or_reparse(root) or not root.is_dir():
        raise TelegramSettingsError("Telegram state must be a real directory.")
    root = root.resolve(strict=True)
    path = root / _SETTINGS_NAME
    if path_is_link_or_reparse(path):
        raise TelegramSettingsError("Telegram settings are an unsafe link.")
    return path


def load_settings(state_dir: Path) -> dict[str, Any]:
    path = _settings_path(state_dir)
    try:
        path.stat()
    except FileNotFoundError:
        return {}
    except OSError as exc:
        raise TelegramSettingsError("Could not read Telegram settings.") from exc
    if not path.is_file():
        raise TelegramSettingsError("Telegram settings are invalid.")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise TelegramSettingsError("Could not read Telegram settings.") from exc
    except (TypeError, ValueError) as exc:
        raise TelegramSettingsError("Telegram settings are invalid.") from exc
    if not isinstance(value, dict):
        raise TelegramSettingsError("Telegram settings are invalid.")
    return value


def owner_chat_id(state_dir: Path) -> int:
    try:
        value = int(str(load_settings(state_dir).get("TELEGRAM_CHAT_ID") or "").strip())
    except (TypeError, ValueError):
        return 0
    return value if value > 0 else 0


def validate_telegram_proxy(value: Any) -> str | None:
    """Parse the Telegram-only proxy without echoing its possible credentials."""
    text = str(value or "").strip()
    if not text:
        return None
    try:
        url = httpx.Proxy(text).url
        valid = (
            url.scheme in ("http", "https", "socks5", "socks5h")
            and bool(url.host)
            and (url.port is not None or url.scheme in ("http", "https"))
            and (url.port is None or 0 < url.port <= 65535)
            and url.raw_path == b"/" and not url.query and not url.fragment
        )
    except Exception:
        valid = False
    if not valid:
        raise ValueError(
            "TELEGRAM_PROXY must be scheme://[user:password@]host[:port] "
            "with scheme http, https, socks5 or socks5h (SOCKS requires a port)."
        )
    return text


def telegram_proxy(state_dir: Path) -> str | None:
    """The same skill-local setting for bridge, notifier and companion APIs."""
    return validate_telegram_proxy(load_settings(state_dir).get("TELEGRAM_PROXY"))


def validate_telegram_api_base(value: Any) -> str:
    """Parse TELEGRAM_API_BASE (self-hosted Bot API server) without credentials.

    Accepts http/https scheme://host[:port] with no path beyond an optional
    trailing slash; anything else is rejected so the token-bearing URL is
    never silently assembled from a malformed base. The returned string is a
    canonical origin (scheme://host[:port]) rebuilt from the parsed URL: no
    fragment/query ever survives, not even an empty "#" delimiter that would
    otherwise swallow the /bot<token>/... suffix during string concatenation.
    """
    text = str(value or "").strip().rstrip("/")
    if not text:
        return "https://api.telegram.org"
    if "#" in text or "?" in text:
        raise ValueError(
            "TELEGRAM_API_BASE must be http(s)://host[:port] with no path, "
            "query, fragment or credentials."
        )
    canonical: Optional[str] = None
    try:
        url = httpx.URL(text)
        valid = (
            url.scheme in ("http", "https")
            and bool(url.host)
            and (url.port is None or 0 < url.port <= 65535)
            and url.raw_path in (b"/", b"")
            and not url.query and not url.fragment
            and not url.username and not url.password
        )
        if valid:
            host = url.host or ""
            # httpx.URL.host returns IPv6 literals WITHOUT brackets; rebuild
            # the origin with them so http://[::1]:8081 round-trips through
            # settings persistence and client revalidation.
            bracketed = f"[{host}]" if ":" in host else host
            origin = f"{url.scheme}://{bracketed}"
            if url.port is not None:
                origin += f":{url.port}"
            canonical = origin
    except Exception:
        valid = False
    if not valid:
        raise ValueError(
            "TELEGRAM_API_BASE must be http(s)://host[:port] with no path, "
            "query, fragment or credentials."
        )
    return canonical or "https://api.telegram.org"


def telegram_api_base(state_dir: Path) -> str:
    """The Bot API base URL setting; defaults to the cloud endpoint."""
    return validate_telegram_api_base(load_settings(state_dir).get("TELEGRAM_API_BASE"))


def validate_telegram_files_dir(value: Any) -> Optional[str]:
    """Parse TELEGRAM_API_FILES_DIR: a native absolute directory path or empty.

    Accepts Unix (``/store``), Windows drive (``C:\\store``) and UNC
    (``\\\\server\\share\\store``) spellings, with optional trailing slashes.
    Path-illegal characters are rejected by the pathlib round-trip. The
    filesystem root of either spelling ("/" or a drive root) is rejected: it
    would make the disk-fallback confinement vacuous (every resolvable path
    lies under it), letting a malicious server response name any readable
    file for exfiltration into the owner chat.
    """
    text = str(value or "").strip()
    if not text:
        return None
    if "\x00" in text:
        raise ValueError(
            "TELEGRAM_API_FILES_DIR must be an absolute directory path "
            "(Unix /path, Windows drive C:\\path, or UNC \\\\server\\share\\path) "
            "in this platform's native spelling."
        )
    # Normalize FIRST and classify the normalized spelling: os.path.normpath is
    # idempotent, so whatever passes once passes unchanged on reload. Spellings
    # are also kept NATIVE to the running platform: on Windows a POSIX path
    # normalizes to a single-backslash root that is neither drive nor UNC and
    # is rejected at save time; on POSIX a Windows drive/UNC spelling would
    # otherwise be treated by pathlib as a relative name and resolved against
    # the working directory instead of the operator's intended absolute
    # location, so it is rejected too (round-6 A1).
    normalized = os.path.normpath(text)
    if os.name == "nt":
        allowed = PureWindowsPath(normalized).is_absolute()
        candidate = PureWindowsPath(normalized)
    else:
        allowed = normalized.startswith("/")
        candidate = PurePosixPath(normalized)
    if not allowed:
        raise ValueError(
            "TELEGRAM_API_FILES_DIR must be an absolute directory path native "
            "to this platform (Unix /path; Windows C:\\path or \\\\server\\share\\path)."
        )
    # Root-level stores are refused on every spelling, including aliases that
    # only normalize down to a root ("/tmp/..", "C:/store/.."): confinement to
    # such a store would be vacuous.
    if len(candidate.parts) <= 1:
        raise ValueError(
            "TELEGRAM_API_FILES_DIR must name a concrete directory, not the filesystem root."
        )
    return normalized


def telegram_files_dir(state_dir: Path) -> Optional[str]:
    """The local server file store; None keeps HTTP-only downloads."""
    return validate_telegram_files_dir(load_settings(state_dir).get("TELEGRAM_API_FILES_DIR"))


def miniapp_enabled(state_dir: Path) -> bool:
    raw = str(load_settings(state_dir).get("TELEGRAM_MINIAPP_ENABLED") or "on").strip().lower()
    return raw in _TRUE


def merge_settings(state_dir: Path, patch: dict[str, Any]) -> dict[str, Any]:
    path = _settings_path(state_dir)
    lock_path = path.with_name(_LOCK_NAME)
    if path_is_link_or_reparse(lock_path):
        raise TelegramSettingsError("Telegram settings lock is unsafe.")
    try:
        fd = acquire_file_lock(lock_path, timeout_sec=2.0)
    except Exception as exc:
        raise TelegramSettingsError("Telegram settings are busy.") from exc
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        current = load_settings(path.parent)
        current.update(patch)
        encoded = json.dumps(current, ensure_ascii=False, indent=2, sort_keys=True)
        tmp.write_text(encoded, encoding="utf-8")
        try:
            tmp.chmod(0o600)
        except OSError:
            pass
        os.replace(tmp, path)
        return current
    except OSError as exc:
        raise TelegramSettingsError("Could not persist Telegram settings.") from exc
    finally:
        release_file_lock(fd)
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def request_may_change_owner(request: Any) -> bool:
    headers = getattr(request, "headers", {})
    if str(headers.get(MINIAPP_MARKER_HEADER, "")).strip():
        return False
    client = getattr(request, "client", None)
    host = str(getattr(client, "host", "") or "").strip()
    return host in {"127.0.0.1", "::1"}


# The exact keys the declarative Settings form owns: POST accepts only these and
# the GET that hydrates the form returns only non-secret fields. The bot token
# belongs to Secrets; the proxy is a masked, write-only skill setting.
_SETTINGS_FORM_KEYS = (
    "TELEGRAM_CHAT_ID", "TELEGRAM_MAX_UPDATES_PER_POLL", "TELEGRAM_MIRROR_MODE",
    "TELEGRAM_COMMAND_MODE", "TELEGRAM_SILENT_MODE",
    "TELEGRAM_SUBAGENT_CARDS", "TELEGRAM_MIRROR_PROGRESS", "TELEGRAM_NOTIFY_TASKS",
    "TELEGRAM_NOTIFY_BUDGET", "TELEGRAM_MINIAPP_ENABLED", "TELEGRAM_PROXY",
    "TELEGRAM_API_BASE", "TELEGRAM_API_FILES_DIR",
)

def make_settings_save(api):
    async def _settings_save(request):
        if str(getattr(request, "method", "POST") or "POST").upper() == "GET":
            # Hydration read for the Settings form: only the form's own keys
            # that are actually stored, as strings, so the UI shows what is
            # saved instead of the schema's first option.
            try:
                stored = load_settings(Path(api.get_state_dir()))
            except TelegramSettingsError as exc:
                return JSONResponse({"ok": False, "message": str(exc)}, status_code=409)
            values = {key: str(stored[key]) for key in _SETTINGS_FORM_KEYS
                      if key in stored and key != "TELEGRAM_PROXY"}
            if "TELEGRAM_PROXY" in stored:
                values["telegram_proxy_status"] = "Configured" if stored["TELEGRAM_PROXY"] else "Not configured"
            if stored.get("TELEGRAM_API_BASE"):
                values["telegram_api_base_status"] = str(stored["TELEGRAM_API_BASE"])
            return JSONResponse(values)
        try:
            data = await request.json()
        except (TypeError, ValueError):
            return JSONResponse(
                {"ok": False, "message": "Invalid Telegram settings payload."},
                status_code=400,
            )
        if not isinstance(data, dict):
            return JSONResponse(
                {"ok": False, "message": "Invalid Telegram settings payload."},
                status_code=400,
            )
        payload = {key: data.get(key) for key in _SETTINGS_FORM_KEYS if key in data}
        # Password fields intentionally hydrate empty. An untouched field must
        # preserve its saved value; removal is a separate, explicit form action.
        if data.get("clear_telegram_proxy") is True:
            payload["TELEGRAM_PROXY"] = ""
        elif "TELEGRAM_PROXY" in payload:
            try:
                proxy = validate_telegram_proxy(payload["TELEGRAM_PROXY"])
            except ValueError as exc:
                return JSONResponse({"ok": False, "message": str(exc), "error": str(exc)}, status_code=400)
            if proxy is None:
                payload.pop("TELEGRAM_PROXY")
            else:
                payload["TELEGRAM_PROXY"] = proxy
        if data.get("clear_telegram_api_base") is True:
            payload["TELEGRAM_API_BASE"] = ""
        elif "TELEGRAM_API_BASE" in payload:
            raw = str(payload.get("TELEGRAM_API_BASE") or "").strip()
            if not raw:
                # An untouched/emptied text field preserves the saved value;
                # removal is the separate clear checkbox above.
                payload.pop("TELEGRAM_API_BASE", None)
            else:
                try:
                    payload["TELEGRAM_API_BASE"] = validate_telegram_api_base(raw)
                except ValueError as exc:
                    return JSONResponse({"ok": False, "message": str(exc), "error": str(exc)}, status_code=400)
        if data.get("clear_telegram_api_files_dir") is True:
            payload["TELEGRAM_API_FILES_DIR"] = ""
        elif "TELEGRAM_API_FILES_DIR" in payload:
            try:
                files_dir = validate_telegram_files_dir(payload["TELEGRAM_API_FILES_DIR"])
            except ValueError as exc:
                return JSONResponse({"ok": False, "message": str(exc), "error": str(exc)}, status_code=400)
            if files_dir is None:
                payload.pop("TELEGRAM_API_FILES_DIR")
            else:
                payload["TELEGRAM_API_FILES_DIR"] = files_dir
        owner_ignored = False
        if "TELEGRAM_CHAT_ID" in payload and not request_may_change_owner(request):
            payload.pop("TELEGRAM_CHAT_ID", None)
            owner_ignored = True
        try:
            merge_settings(Path(api.get_state_dir()), payload)
        except TelegramSettingsError as exc:
            return JSONResponse(
                {"ok": False, "message": str(exc)},
                status_code=409,
            )
        message = "Telegram settings saved."
        if "TELEGRAM_PROXY" in payload:
            message += " Saved Telegram proxy: " + ("configured." if payload["TELEGRAM_PROXY"] else "cleared.")
            message += " Disable and re-enable the skill to apply the proxy to all Telegram calls."
        if owner_ignored:
            message += " Owner binding was left unchanged."
        return JSONResponse({"ok": True, "owner_ignored": owner_ignored, "message": message})
    return _settings_save


class TelegramSettingsObserver:
    """Read-only companion view of this skill's own settings."""

    def __init__(self, state_dir: Path, _core_port: int) -> None:
        self.state_dir = state_dir

    def owner_chat_id(self) -> int:
        return owner_chat_id(self.state_dir)

    def safe_for_exposure(self, owner: int) -> bool:
        settings = load_settings(self.state_dir)
        try:
            current_owner = int(str(settings.get("TELEGRAM_CHAT_ID") or "").strip())
        except (TypeError, ValueError):
            current_owner = 0
        enabled = str(settings.get("TELEGRAM_MINIAPP_ENABLED") or "on").strip().lower() in _TRUE
        return owner > 0 and current_owner == owner and enabled

__all__ = [
    "MINIAPP_MARKER_HEADER",
    "TelegramSettingsError",
    "TelegramSettingsObserver",
    "load_settings",
    "merge_settings",
    "make_settings_save",
    "miniapp_enabled",
    "owner_chat_id",
    "request_may_change_owner",
    "telegram_api_base",
    "telegram_files_dir",
    "telegram_proxy",
    "validate_telegram_api_base",
    "validate_telegram_files_dir",
    "validate_telegram_proxy",
]
