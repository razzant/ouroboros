"""Who becomes the Telegram bridge's owner: the setup wizard's one-time link, else the first private writer.

The setup wizard's card (DESIGN «Phone via Telegram», ``ouroboros/onboarding_connections.py``) shows a QR of
``https://t.me/<bot>?start=<code>`` and, once the owner finishes setup, gives the code to this skill's
``settings/save`` route, which keeps only its SHA-256 (``TELEGRAM_PAIRING_DIGEST``). Until that post lands the core's
starter record still holds it (``awaited_code_digest``), so the bridge knows a link waits from its first poll.
(``scripts/telegram_settings.py`` keeps the digest; it repeats the code's form because the companion imports it alone.)

While a link waits, only a private ``/start <that code>`` makes its chat the owner chat (``TELEGRAM_CHAT_ID``); any other
private message is answered and binds nobody. Where no link was ever shown (a server, Docker or an older install), the
owner's rule of 2026-09-02 holds: the first private writer is the owner. Binding is the skill's own settings write;
the core's emergency door needs no step of its own (``ouroboros.server_control.PanicIngress`` admits a positive
private sender while its owner slot is free), so the first /panic from the bound chat stops in every command mode.

``bot_identity`` is the card's token check and stays stdlib-only at import: the host loads this file by path from the
app's own copy of the skill, whether the bridge runs or not.
"""
from __future__ import annotations

import hashlib
import json
import re
import urllib.error
import urllib.request
from typing import Any, Optional

DIGEST_KEY = "TELEGRAM_PAIRING_DIGEST"
CODE_FIELD = "TELEGRAM_PAIRING_CODE"  # what the wizard's post names; never stored as such
CLOUD_API_BASE = "https://api.telegram.org"
_CODE = re.compile(r"[A-Za-z0-9_-]{22,64}")  # what the wizard mints, within what a start parameter carries
_TOKEN = re.compile(r"\d{1,20}:[A-Za-z0-9_-]{1,128}")
_START = re.compile(r"/start(?:@[A-Za-z0-9_]{1,64})?(?:\s+(\S+))?\s*", re.IGNORECASE)
_MAX_ANSWER = 65536


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args: Any, **kwargs: Any) -> None:  # a token never follows a redirect
        return None


def bot_identity(token: Any, api_base: Any = None, *, timeout: float = 8.0) -> dict:
    """Which bot ``token`` names: ``{"ok": True, "username": …}``, ``{"ok": False, "cause": "refused"}`` when Telegram
    rejected it, or ``{"ok": False, "cause": "unreachable"}``. One bare ``getMe``; the answer carries neither the token
    nor the address, and nothing is logged or written."""
    token = str(token or "").strip()
    if not _TOKEN.fullmatch(token):
        return {"ok": False, "cause": "refused"}
    base = str(api_base or CLOUD_API_BASE).rstrip("/")
    opener = urllib.request.build_opener(_NoRedirect)  # env proxies apply as for any urllib request
    try:
        with opener.open(f"{base}/bot{token}/getMe", timeout=timeout) as answer:
            body = json.loads(answer.read(_MAX_ANSWER).decode("utf-8"))
    except urllib.error.HTTPError as refused:
        return {"ok": False, "cause": "refused" if refused.code in (401, 403, 404) else "unreachable"}
    except Exception:  # noqa: BLE001 — any transport or parse failure: Telegram could not be asked from here
        return {"ok": False, "cause": "unreachable"}
    username = str(((body or {}).get("result") or {}).get("username") or "") if isinstance(body, dict) else ""
    if not (isinstance(body, dict) and body.get("ok") is True and re.fullmatch(r"[A-Za-z0-9_]{5,64}", username)):
        return {"ok": False, "cause": "unreachable"}
    return {"ok": True, "username": username}


def start_payload(text: Any) -> Optional[str]:
    """``None`` for anything but ``/start``; ``""`` for a bare Start or a payload that is no code; else the code."""
    match = _START.fullmatch(str(text or "").strip())
    if match is None:
        return None
    payload = match.group(1) or ""
    return payload if _CODE.fullmatch(payload) else ""


def digest(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def _awaited_by_setup(data_dir: Any) -> Optional[str]:
    """The digest of the link the wizard showed and still holds for this skill (before its post lands), or None."""
    try:
        from ouroboros.onboarding_connections import awaited_code_digest
    except ImportError:  # an older core: the post is the only carrier
        return None
    return awaited_code_digest(data_dir, "telegram")


def unbound_private(text: Any, settings: dict, data_dir: Any) -> str:
    """For a private message while no owner chat is bound: ``"bind"`` when this chat becomes the owner (the waiting
    link's own code, or the first writer where no link waits), else ``"wait"``."""
    waiting = {value for value in (str(settings.get(DIGEST_KEY) or "").strip(), _awaited_by_setup(data_dir)) if value}
    if not waiting:
        return "bind"
    payload = start_payload(text)
    return "bind" if payload and digest(payload) in waiting else "wait"


async def bind_unbound(api, client, update: dict, chat_id: int, lang: str) -> bool:
    """The bridge's owner rule for a private update while no owner chat is bound (the caller checked private and
    positive): bind ``chat_id`` when ``unbound_private`` says so, else tell a writer that the bot waits for its owner's
    link and bind nobody. Bridge-side only: its imports stay inside so the card's ``bot_identity`` loads by path."""
    import pathlib

    from ..scripts.telegram_settings import load_settings, merge_settings
    from .telegram_api import _LOCALIZED_TEXTS
    from .telegram_state import _data_dir

    state_dir = pathlib.Path(api.get_state_dir())
    callback = bool(update.get("callback_query"))
    text = None if callback else (update.get("message") or {}).get("text")
    if unbound_private(text, load_settings(state_dir), _data_dir(api)) == "bind":
        merge_settings(state_dir, {"TELEGRAM_CHAT_ID": str(chat_id), DIGEST_KEY: ""})
        return True
    if not callback:
        await client.send_message(chat_id, _LOCALIZED_TEXTS[lang]["pairing_waits"])
    return False


__all__ = ["CODE_FIELD", "DIGEST_KEY", "bind_unbound", "bot_identity", "digest", "start_payload", "unbound_private"]
