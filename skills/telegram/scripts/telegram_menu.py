"""Crash-safe ownership of one private Telegram chat menu button.

The companion is the only caller. It keeps the pre-Mini-App button so a normal
disable can restore it, and records every URL rotation before touching
Telegram so a restart can reconcile an ambiguous remote result.

Each bot keeps its own ledger, named by the id Telegram's getMe reports for the
token (a rotated token of the same bot keeps it): a replacement bot neither reads
nor clears a ledger another bot could not restore. The older chat-only ledger is
never changed or removed. A bot whose current menu shows a Mini App button it
records derives its own ledger from it and notes that basis with the source's
digest: an inference from the menu now, not proof of which bot wrote that file.
When the chat-only ledger names an earlier owner, the companion reads that chat's
menu once before exposing a new owner, so a ledger derived there is restored like any other.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx

from platform_support import fsync_directory, path_is_link_or_reparse
from telegram_settings import TelegramSettingsError, telegram_api_base, telegram_proxy


_LEGACY_NAME, _LEGACY_SCHEMA = "menu_button_snapshot.json", 2  # chat-only, written before ledgers named a bot
_SNAPSHOT_SCHEMA = 3
_BOT_ID_RE = re.compile(r"^[1-9][0-9]{0,19}$")
_TUNNEL_HOST_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.trycloudflare\.com$")


class TelegramMenuError(RuntimeError):
    """A safe, token-free Telegram menu lifecycle error."""


class TelegramMenuTransportError(TelegramMenuError):
    """Telegram could not be observed reliably; retry without changing ownership."""


class TelegramMenuRejectedError(TelegramMenuError):
    """Telegram rejected the configured bot, owner, or request."""


class TelegramMenuConflictError(TelegramMenuError):
    """The remote button no longer matches this skill's durable ownership ledger."""


def normalize_public_url(value: Any) -> str:
    raw = str(value or "").strip()
    try:
        parsed = urlsplit(raw)
        port = parsed.port
    except ValueError as exc:
        raise TelegramMenuError("Tunnel URL is invalid.") from exc
    host = str(parsed.hostname or "").lower().rstrip(".")
    if parsed.scheme.lower() != "https" or parsed.username or parsed.password:
        raise TelegramMenuError("Tunnel URL must be an uncredentialed HTTPS URL.")
    if port not in (None, 443) or not _TUNNEL_HOST_RE.fullmatch(host):
        raise TelegramMenuError("Tunnel URL is not an exact trycloudflare.com host.")
    if parsed.path not in ("", "/") or parsed.query or parsed.fragment:
        raise TelegramMenuError("Tunnel URL must be the origin root.")
    return urlunsplit(("https", host, "/", "", ""))


def menu_button(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TelegramMenuError("Telegram returned an invalid menu button.")
    kind = str(value.get("type") or "").strip()
    if kind in {"default", "commands"}:
        return {"type": kind}
    if kind != "web_app" or not isinstance(value.get("web_app"), dict):
        raise TelegramMenuError("Telegram returned an unsupported menu button.")
    text = str(value.get("text") or "").strip()
    url = str(value["web_app"].get("url") or "").strip()
    if not text or len(text) > 64 or not url or len(url) > 2048:
        raise TelegramMenuError("Telegram returned an incomplete menu button.")
    return {"type": "web_app", "text": text, "web_app": {"url": url}}


def web_app_button(url: str, text: str = "Ouroboros") -> dict[str, Any]:
    label = str(text or "").strip().replace("\r", " ").replace("\n", " ")
    if not label or len(label) > 64:
        raise TelegramMenuError("Telegram button text must be 1-64 characters.")
    return {
        "type": "web_app",
        "text": label,
        "web_app": {"url": normalize_public_url(url)},
    }


def _ledger_name(bot: str) -> str:
    if not _BOT_ID_RE.fullmatch(str(bot or "")):
        raise TelegramMenuError("The Telegram bot identity is not verified.")
    return f"menu_button_snapshot.bot-{bot}.json"


def _bot_identity(me: Any) -> str:
    """The bot id Telegram's getMe reported, which names its ledger; never read from the token's text."""
    raw = me.get("id") if isinstance(me, dict) else None
    try:
        bot = "" if isinstance(raw, bool) else str(int(raw))
    except (TypeError, ValueError):
        bot = ""
    if not _BOT_ID_RE.fullmatch(bot):
        raise TelegramMenuError("Telegram returned an invalid bot identity.")
    return bot


def _read_ledger(path: Path, *, raw: bool = False) -> Any:
    """One ledger file's JSON (its exact bytes with ``raw``), or None when absent; an unsafe, oversized or
    unreadable file is a conflict."""
    if not path.exists():
        return None
    if path_is_link_or_reparse(path):
        raise TelegramMenuConflictError("Menu rollback snapshot is an unsafe link.")
    try:
        if path.stat().st_size > 16_384:
            raise TelegramMenuConflictError("Menu rollback snapshot is oversized.")
        data = path.read_bytes()
        return data if raw else json.loads(data.decode("utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise TelegramMenuConflictError("Menu rollback snapshot is unreadable.") from exc


def _installed_by(record: dict[str, Any], observed: dict[str, Any]) -> bool:
    """``observed`` is a Mini App button ``record`` lists as installed: a tunnel URL is specific to one install,
    so the match is strong evidence, though not proof of which bot wrote ``record``."""
    ours = [record.get("owned"), *([record.get("to")] if record.get("to") != record.get("original") else [])]
    return observed.get("type") == "web_app" and observed in ours


def snapshot_owner_chat_id(state_dir: str | Path, bot: str) -> int | None:
    """Read only the durable menu owner of ``bot``'s ledger before constructing an owner-bound manager."""

    payload = _read_ledger(Path(state_dir).expanduser().resolve() / _ledger_name(bot))
    if payload is None:
        return None
    try:
        owner = int(payload.get("chat_id")) if isinstance(payload, dict) else 0
    except (TypeError, ValueError) as exc:
        raise TelegramMenuConflictError("Menu rollback snapshot owner is unreadable.") from exc
    if owner <= 0 or payload.get("schema") != _SNAPSHOT_SCHEMA or payload.get("bot") != bot:
        raise TelegramMenuConflictError("Menu rollback snapshot owner is invalid.")
    return owner


def chat_only_owner_chat_id(state_dir: str | Path) -> int | None:
    """The owner the chat-only ledger names; None when it is absent, unreadable or invalid. It names no bot, so it
    proves nothing until a bot's menu in that chat shows a button it records (``derive_from_chat_only``)."""
    try:
        payload = _read_ledger(Path(state_dir).expanduser().resolve() / _LEGACY_NAME)
        legacy = isinstance(payload, dict) and payload.get("schema") == _LEGACY_SCHEMA
        owner = int(payload.get("chat_id")) if legacy else 0
    except (TelegramMenuError, TypeError, ValueError):
        return None
    return owner if owner > 0 else None


class TelegramMenuManager:
    def __init__(
        self,
        token: str,
        chat_id: int,
        state_dir: str | Path,
        *,
        api_root: str | None = None,
        client: httpx.AsyncClient | None = None,
        bot: str = "",
    ) -> None:
        self._token = str(token or "").strip()
        self.chat_id = int(chat_id)
        if not self._token:
            raise TelegramMenuError("TELEGRAM_BOT_TOKEN is missing.")
        if self.chat_id <= 0:
            raise TelegramMenuError("Owner binding must be a positive private chat ID.")
        self.state_dir = Path(state_dir).expanduser().resolve()
        self.state_dir.mkdir(parents=True, exist_ok=True)
        # None follows the skill's saved TELEGRAM_API_BASE at every call, as the bridge does,
        # so install, checks and the rollback restore reach the server the bot now uses.
        self._api_root = str(api_root).rstrip("/") if api_root else None
        self._client = client
        self.last_rejected_api_root: str | None = None
        # The bot getMe reported for this token (verify_private_owner or the first ledger access);
        # a caller that already verified the same token passes it on.
        self.bot = _bot_identity({"id": bot}) if bot else ""

    @property
    def snapshot_path(self) -> Path:
        """This bot's own ledger."""
        path = self.state_dir / _ledger_name(self.bot)
        try:
            path.relative_to(self.state_dir)
        except ValueError as exc:
            raise TelegramMenuError("Menu snapshot escaped skill state.") from exc
        if path_is_link_or_reparse(path):
            raise TelegramMenuConflictError("Menu rollback snapshot is an unsafe link.")
        return path

    async def _call(
        self,
        method: str,
        payload: dict[str, Any],
        *,
        request_timeout_sec: float | None = None,
    ) -> Any:
        try:
            api_root = self._api_root or telegram_api_base(self.state_dir)
        except (TelegramSettingsError, ValueError):
            raise TelegramMenuTransportError("Telegram Bot API settings are unavailable or invalid.") from None
        owned_client = self._client is None
        client = self._client or httpx.AsyncClient(
            timeout=12,
            follow_redirects=False,
            trust_env=False,
            proxy=telegram_proxy(self.state_dir),
        )
        request_options: dict[str, Any] = {}
        if request_timeout_sec is not None:
            timeout = float(request_timeout_sec)
            if timeout <= 0:
                raise TelegramMenuError("Telegram request timeout must be positive.")
            request_options["timeout"] = timeout
        try:
            response = await client.post(
                f"{api_root}/bot{self._token}/{method}",
                json=payload,
                **request_options,
            )
        except httpx.TimeoutException:
            raise TelegramMenuTransportError(f"Telegram API timed out during {method}.") from None
        except httpx.HTTPError as exc:
            raise TelegramMenuTransportError(
                f"Telegram API transport failed during {method} ({type(exc).__name__})."
            ) from None
        finally:
            if owned_client:
                await client.aclose()
        try:
            body = response.json()
        except ValueError as exc:
            raise TelegramMenuTransportError(f"Telegram API returned non-JSON during {method}.") from exc
        if response.status_code == 429 or response.status_code >= 500:
            raise TelegramMenuTransportError(f"Telegram API is temporarily unavailable during {method}.")
        if response.status_code >= 400 or not isinstance(body, dict) or not body.get("ok"):
            # Telegram descriptions may echo the request URL, which contains the
            # token.  Deliberately expose no remote free text here.
            self.last_rejected_api_root = api_root
            raise TelegramMenuRejectedError(f"Telegram API rejected {method}.")
        return body.get("result")

    async def _identify(self, *, request_timeout_sec: float | None = None) -> None:
        if not self.bot:
            self.bot = _bot_identity(await self._call("getMe", {}, request_timeout_sec=request_timeout_sec))

    async def verify_private_owner(self) -> str:
        bot = await self._call("getMe", {})
        self.bot = _bot_identity(bot)
        chat = await self._call("getChat", {"chat_id": self.chat_id})
        if not isinstance(chat, dict) or str(chat.get("type") or "") != "private":
            raise TelegramMenuError("The configured Telegram owner chat is not private.")
        try:
            observed = int(chat.get("id"))
        except (TypeError, ValueError) as exc:
            raise TelegramMenuError("Telegram returned an invalid owner chat ID.") from exc
        if observed != self.chat_id:
            raise TelegramMenuError("Telegram returned a different owner chat ID.")
        username = str(bot.get("username") or "").strip() if isinstance(bot, dict) else ""
        return f"@{username}" if username else "Telegram bot"

    async def current(self, *, request_timeout_sec: float | None = None) -> dict[str, Any]:
        result = await self._call(
            "getChatMenuButton",
            {"chat_id": self.chat_id},
            request_timeout_sec=request_timeout_sec,
        )
        return menu_button(result)

    async def _set(
        self,
        button: dict[str, Any],
        *,
        request_timeout_sec: float | None = None,
    ) -> None:
        await self._call(
            "setChatMenuButton",
            {"chat_id": self.chat_id, "menu_button": button},
            request_timeout_sec=request_timeout_sec,
        )

    def _record(self, phase: str, original: dict[str, Any], owned: dict[str, Any] | None, *,
                adopted_from: dict[str, Any] | None = None, **transition: dict[str, Any]) -> dict[str, Any]:
        """One ledger state; ``adopted_from`` (a ledger derived from the chat-only one) stays on every later state."""
        return {"schema": _SNAPSHOT_SCHEMA, "bot": self.bot, "chat_id": self.chat_id, "phase": phase,
                "original": original, "owned": owned, **transition,
                **({"adopted_from": adopted_from} if adopted_from else {})}

    def _parse(self, raw: Any, schema: int) -> dict[str, Any]:
        if not isinstance(raw, dict) or raw.get("schema") != schema:
            raise TelegramMenuError("Menu rollback snapshot has an unsupported schema.")
        if schema == _SNAPSHOT_SCHEMA and raw.get("bot") != self.bot:
            raise TelegramMenuError("Menu rollback snapshot belongs to another bot.")
        try:
            chat_id = int(raw.get("chat_id"))
        except (TypeError, ValueError) as exc:
            raise TelegramMenuError("Menu rollback snapshot has an invalid owner.") from exc
        if chat_id != self.chat_id:
            raise TelegramMenuError("Menu rollback snapshot belongs to another owner.")
        phase = str(raw.get("phase") or "")
        if phase not in {"installed", "mutating"}:
            raise TelegramMenuError("Menu rollback snapshot has an invalid phase.")
        owned = raw.get("owned")
        transition = ({"from": menu_button(raw.get("from")), "to": menu_button(raw.get("to"))}
                      if phase == "mutating" else {})
        adopted = raw.get("adopted_from") if schema == _SNAPSHOT_SCHEMA else None
        return self._record(phase, menu_button(raw.get("original")), None if owned is None else menu_button(owned),
                            adopted_from=adopted if isinstance(adopted, dict) else None, **transition)

    def _load(self, observed: dict[str, Any] | None = None) -> dict[str, Any] | None:
        """This bot's ledger; else one derived from the chat-only ledger when ``observed`` shows a Mini App button
        it records. The chat-only file is left byte for byte as it is."""
        raw = _read_ledger(self.snapshot_path)
        if raw is not None:
            return self._parse(raw, _SNAPSHOT_SCHEMA)
        try:
            data = _read_ledger(self.state_dir / _LEGACY_NAME, raw=True) if observed else None
            legacy = None if data is None else self._parse(json.loads(data), _LEGACY_SCHEMA)
        except (TelegramMenuError, ValueError):
            return None  # unreadable or another owner's: nothing proves whose it is, so it stays as it is
        if legacy is None or not _installed_by(legacy, observed):
            return None
        # The basis is this bot's menu now, not a proven past writer; the digest names the exact source bytes.
        legacy["adopted_from"] = {"ledger": _LEGACY_NAME, "schema": _LEGACY_SCHEMA, "basis": "current_menu_match",
                                  "sha256": hashlib.sha256(data).hexdigest()}
        self._write(legacy)
        return legacy

    async def derive_from_chat_only(self, *, request_timeout_sec: float | None = None) -> bool:
        """For a bot without a ledger of its own: read this chat's menu once and derive the ledger (``_load``) only
        when it shows a Mini App button the chat-only ledger records. True when this bot now holds one to restore."""
        await self._identify(request_timeout_sec=request_timeout_sec)
        if self.snapshot_path.exists() or not (self.state_dir / _LEGACY_NAME).exists():
            return False
        return self._load(await self.current(request_timeout_sec=request_timeout_sec)) is not None

    def _write(self, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        if len(encoded.encode("utf-8")) > 16_384:
            raise TelegramMenuError("Menu rollback snapshot is unexpectedly large.")
        path = self.snapshot_path
        tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{secrets.token_hex(8)}")
        try:
            with tmp.open("x", encoding="utf-8") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                tmp.chmod(0o600)
            except OSError:
                pass
            os.replace(tmp, path)
            fsync_directory(path.parent)
        except OSError as exc:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            raise TelegramMenuError("Could not persist Telegram rollback state.") from exc

    def _delete(self, path: Path | None = None) -> None:
        try:
            (path or self.snapshot_path).unlink(missing_ok=True)
            fsync_directory(self.state_dir)
        except OSError as exc:
            raise TelegramMenuError("Could not remove Telegram rollback state.") from exc

    def _installed_snapshot(
        self,
        original: dict[str, Any],
        owned: dict[str, Any],
        adopted_from: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._record("installed", original, owned, adopted_from=adopted_from)

    async def _reconcile(self, snapshot: dict[str, Any], observed: dict[str, Any]) -> dict[str, Any] | None:
        if snapshot["phase"] == "installed":
            if observed == snapshot["owned"]:
                return snapshot
            if observed == snapshot["original"]:
                # Telegram already has the exact pre-Mini-App button.  Treat a
                # surviving installed snapshot as a completed rollback rather
                # than blocking the next enabled generation forever.
                self._delete()
                return None
            raise TelegramMenuConflictError(
                "Telegram menu changed outside this skill; refusing to overwrite it."
            )

        before = snapshot["from"]
        after = snapshot["to"]
        if observed == after:
            stable = self._installed_snapshot(snapshot["original"], after, snapshot.get("adopted_from"))
            self._write(stable)
            return stable
        if observed == before:
            if snapshot["owned"] is None:
                # First install did not reach Telegram.
                self._delete()
                return None
            stable = self._installed_snapshot(snapshot["original"], before, snapshot.get("adopted_from"))
            self._write(stable)
            return stable
        raise TelegramMenuConflictError(
            "Telegram menu drifted during an interrupted update; refusing to overwrite it."
        )

    async def check_owned(self, url: str, text: str = "Ouroboros") -> None:
        """Confirm current ownership without ever mutating Telegram."""

        target = web_app_button(url, text)
        observed = await self.current()
        await self._identify()
        loaded = self._load(observed)
        if loaded is None:
            raise TelegramMenuConflictError("Telegram menu ownership snapshot is missing.")
        snapshot = await self._reconcile(loaded, observed)
        if snapshot is None or snapshot["owned"] != target or observed != target:
            raise TelegramMenuConflictError("Telegram menu no longer matches this Mini App URL.")

    async def check_snapshot_owned(self) -> None:
        """Confirm the durable owned button without knowing its prior URL."""

        observed = await self.current()
        await self._identify()
        loaded = self._load(observed)
        if loaded is None:
            raise TelegramMenuConflictError("Telegram menu ownership snapshot is missing.")
        snapshot = await self._reconcile(loaded, observed)
        if snapshot is None or snapshot["owned"] is None or observed != snapshot["owned"]:
            raise TelegramMenuConflictError("Telegram menu no longer matches its ownership snapshot.")

    async def install(self, url: str, text: str = "Ouroboros") -> bool:
        """Install or rotate the button. Return True only after a remote mutation."""
        target = web_app_button(url, text)
        observed = await self.current()
        await self._identify()
        loaded = self._load(observed)
        snapshot = await self._reconcile(loaded, observed) if loaded else None
        if snapshot is not None and snapshot["owned"] == target:
            return False

        original = snapshot["original"] if snapshot else observed
        previous_owned = snapshot["owned"] if snapshot else None
        adopted_from = snapshot.get("adopted_from") if snapshot else None
        self._write(self._record("mutating", original, previous_owned, adopted_from=adopted_from,
                                 **{"from": observed, "to": target}))
        try:
            await self._set(target)
            confirmed = await self.current()
        except Exception:
            # Leave the transaction durable.  The next start distinguishes
            # pre-set, post-set, and third-party drift without guessing.
            raise
        if confirmed != target:
            raise TelegramMenuError("Telegram did not confirm the installed menu button.")
        self._write(self._installed_snapshot(original, target, adopted_from))
        return True

    async def restore(self, *, request_timeout_sec: float | None = None) -> bool:
        """Restore the exact pre-Mini-App button without clobbering external changes."""
        await self._identify(request_timeout_sec=request_timeout_sec)
        if not self.snapshot_path.exists() and not (self.state_dir / _LEGACY_NAME).exists():
            return False
        observed = await self.current(request_timeout_sec=request_timeout_sec)
        snapshot = self._load(observed)
        snapshot = await self._reconcile(snapshot, observed) if snapshot else None
        if snapshot is None:
            return False
        original = snapshot["original"]
        if observed == original:
            self._delete()
            return False
        self._write(self._record("mutating", original, snapshot["owned"], adopted_from=snapshot.get("adopted_from"),
                                 **{"from": observed, "to": original}))
        await self._set(original, request_timeout_sec=request_timeout_sec)
        confirmed = await self.current(request_timeout_sec=request_timeout_sec)
        if confirmed != original:
            raise TelegramMenuError("Telegram did not confirm the restored menu button.")
        self._delete()
        return True


__all__ = [
    "TelegramMenuConflictError",
    "TelegramMenuError",
    "TelegramMenuManager",
    "TelegramMenuRejectedError",
    "TelegramMenuTransportError",
    "chat_only_owner_chat_id",
    "menu_button",
    "normalize_public_url",
    "snapshot_owner_chat_id",
    "web_app_button",
]
