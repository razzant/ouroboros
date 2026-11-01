"""Crash-safe ownership of one private Telegram chat menu button.

The companion is the only caller. It keeps the pre-Mini-App button so a normal
disable can restore it, and records every URL rotation before touching
Telegram so a restart can reconcile an ambiguous remote result.
"""

from __future__ import annotations

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


_SNAPSHOT_NAME = "menu_button_snapshot.json"
_SNAPSHOT_SCHEMA = 2
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


def snapshot_owner_chat_id(state_dir: str | Path) -> int | None:
    """Read only the durable menu owner before constructing an owner-bound manager."""

    root = Path(state_dir).expanduser().resolve()
    path = root / _SNAPSHOT_NAME
    if not path.exists():
        return None
    if path_is_link_or_reparse(path):
        raise TelegramMenuConflictError("Menu rollback snapshot is an unsafe link.")
    try:
        if path.stat().st_size > 16_384:
            raise TelegramMenuConflictError("Menu rollback snapshot is oversized.")
        payload = json.loads(path.read_text(encoding="utf-8"))
        owner = int(payload.get("chat_id")) if isinstance(payload, dict) else 0
    except TelegramMenuConflictError:
        raise
    except (OSError, ValueError, TypeError) as exc:
        raise TelegramMenuConflictError("Menu rollback snapshot owner is unreadable.") from exc
    if not isinstance(payload, dict) or payload.get("schema") != _SNAPSHOT_SCHEMA or owner <= 0:
        raise TelegramMenuConflictError("Menu rollback snapshot owner is invalid.")
    return owner


class TelegramMenuManager:
    def __init__(
        self,
        token: str,
        chat_id: int,
        state_dir: str | Path,
        *,
        api_root: str | None = None,
        client: httpx.AsyncClient | None = None,
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

    @property
    def snapshot_path(self) -> Path:
        path = self.state_dir / _SNAPSHOT_NAME
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

    async def verify_private_owner(self) -> str:
        bot = await self._call("getMe", {})
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

    def _load(self) -> dict[str, Any] | None:
        path = self.snapshot_path
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError) as exc:
            raise TelegramMenuError("Menu rollback snapshot is unreadable.") from exc
        if not isinstance(raw, dict) or raw.get("schema") != _SNAPSHOT_SCHEMA:
            raise TelegramMenuError("Menu rollback snapshot has an unsupported schema.")
        try:
            chat_id = int(raw.get("chat_id"))
        except (TypeError, ValueError) as exc:
            raise TelegramMenuError("Menu rollback snapshot has an invalid owner.") from exc
        if chat_id != self.chat_id:
            raise TelegramMenuError("Menu rollback snapshot belongs to another owner.")
        phase = str(raw.get("phase") or "")
        if phase not in {"installed", "mutating"}:
            raise TelegramMenuError("Menu rollback snapshot has an invalid phase.")
        original = menu_button(raw.get("original"))
        owned = raw.get("owned")
        if owned is not None:
            owned = menu_button(owned)
        result: dict[str, Any] = {
            "schema": _SNAPSHOT_SCHEMA,
            "chat_id": chat_id,
            "phase": phase,
            "original": original,
            "owned": owned,
        }
        if phase == "mutating":
            result["from"] = menu_button(raw.get("from"))
            result["to"] = menu_button(raw.get("to"))
        return result

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

    def _delete(self) -> None:
        try:
            self.snapshot_path.unlink(missing_ok=True)
            fsync_directory(self.state_dir)
        except OSError as exc:
            raise TelegramMenuError("Could not remove Telegram rollback state.") from exc

    def _installed_snapshot(
        self,
        original: dict[str, Any],
        owned: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "schema": _SNAPSHOT_SCHEMA,
            "chat_id": self.chat_id,
            "phase": "installed",
            "original": original,
            "owned": owned,
        }

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
            stable = self._installed_snapshot(snapshot["original"], after)
            self._write(stable)
            return stable
        if observed == before:
            if snapshot["owned"] is None:
                # First install did not reach Telegram.
                self._delete()
                return None
            stable = self._installed_snapshot(snapshot["original"], before)
            self._write(stable)
            return stable
        raise TelegramMenuConflictError(
            "Telegram menu drifted during an interrupted update; refusing to overwrite it."
        )

    async def check_owned(self, url: str, text: str = "Ouroboros") -> None:
        """Confirm current ownership without ever mutating Telegram."""

        target = web_app_button(url, text)
        observed = await self.current()
        loaded = self._load()
        if loaded is None:
            raise TelegramMenuConflictError("Telegram menu ownership snapshot is missing.")
        snapshot = await self._reconcile(loaded, observed)
        if snapshot is None or snapshot["owned"] != target or observed != target:
            raise TelegramMenuConflictError("Telegram menu no longer matches this Mini App URL.")

    async def check_snapshot_owned(self) -> None:
        """Confirm the durable owned button without knowing its prior URL."""

        observed = await self.current()
        loaded = self._load()
        if loaded is None:
            raise TelegramMenuConflictError("Telegram menu ownership snapshot is missing.")
        snapshot = await self._reconcile(loaded, observed)
        if snapshot is None or snapshot["owned"] is None or observed != snapshot["owned"]:
            raise TelegramMenuConflictError("Telegram menu no longer matches its ownership snapshot.")

    async def install(self, url: str, text: str = "Ouroboros") -> bool:
        """Install or rotate the button. Return True only after a remote mutation."""
        target = web_app_button(url, text)
        observed = await self.current()
        loaded = self._load()
        snapshot = await self._reconcile(loaded, observed) if loaded else None
        if snapshot is not None and snapshot["owned"] == target:
            return False

        original = snapshot["original"] if snapshot else observed
        previous_owned = snapshot["owned"] if snapshot else None
        transaction = {
            "schema": _SNAPSHOT_SCHEMA,
            "chat_id": self.chat_id,
            "phase": "mutating",
            "original": original,
            "owned": previous_owned,
            "from": observed,
            "to": target,
        }
        self._write(transaction)
        try:
            await self._set(target)
            confirmed = await self.current()
        except Exception:
            # Leave the transaction durable.  The next start distinguishes
            # pre-set, post-set, and third-party drift without guessing.
            raise
        if confirmed != target:
            raise TelegramMenuError("Telegram did not confirm the installed menu button.")
        self._write(self._installed_snapshot(original, target))
        return True

    async def restore(self, *, request_timeout_sec: float | None = None) -> bool:
        """Restore the exact pre-Mini-App button without clobbering external changes."""
        snapshot = self._load()
        if snapshot is None:
            return False
        observed = await self.current(request_timeout_sec=request_timeout_sec)
        snapshot = await self._reconcile(snapshot, observed)
        if snapshot is None:
            return False
        original = snapshot["original"]
        if observed == original:
            self._delete()
            return False
        transaction = {
            "schema": _SNAPSHOT_SCHEMA,
            "chat_id": self.chat_id,
            "phase": "mutating",
            "original": original,
            "owned": snapshot["owned"],
            "from": observed,
            "to": original,
        }
        self._write(transaction)
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
    "menu_button",
    "normalize_public_url",
    "snapshot_owner_chat_id",
    "web_app_button",
]
