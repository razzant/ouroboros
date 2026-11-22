"""Presence runtime registry, /mode command handling, and non-owner update routing for plugin.py."""

from __future__ import annotations

import asyncio
import os
import pathlib
import sqlite3
import time
import weakref
from typing import Any, Callable, Dict, Optional

from starlette.responses import JSONResponse

from ..scripts.telegram_settings import TelegramSettingsError, merge_settings
from . import telegram_quiz
from .miniapp_registration import _read_status
from .presence.host import PresenceHostClient
from .presence.runtime import PresenceRuntime
from .telegram_connection import _bridge_status
from .telegram_mirror import (
    MIRROR_MODE_CONTROL_ONLY,
    MIRROR_MODE_MAIN,
    OWNER_CHAT_MODE_DIRECT,
    OWNER_CHAT_MODE_PRESENCE,
    VALID_MIRROR_MODES,
    VALID_OWNER_CHAT_MODES,
    owner_chat_mode,
)
from .telegram_panels import _MODE_TEXTS, build_mode_panel, presence_readiness_hint
from .telegram_poll_binding import InjectNotAccepted, update_key
from .telegram_state import _load_settings

_PRESENCE_HINT_THROTTLE_SEC = 60.0
_LAST_PRESENCE_HINT: Dict[tuple[str, int], float] = {}
_PRESENCE_RUNTIMES: "weakref.WeakKeyDictionary[Any, PresenceRuntime]" = weakref.WeakKeyDictionary()
_PRESENCE_RUNTIMES_BY_STATE: "weakref.WeakValueDictionary[str, PresenceRuntime]" = weakref.WeakValueDictionary()


def _skill_token_provider(api: Any) -> Callable[[], str]:
    def _read() -> str:
        getter = getattr(api, "get_skill_token", None)
        if not callable(getter):
            return ""
        token_obj = getter()
        use = getattr(token_obj, "use_in_request", None)
        return str(use() if callable(use) else (token_obj or "")).strip()

    return _read


def presence_runtime(api: Any, *, trust_env: bool = False) -> PresenceRuntime:
    """One PresenceRuntime per skill load (and state dir), sharing the bridge's bot token."""
    try:
        existing = _PRESENCE_RUNTIMES.get(api)
    except TypeError:
        existing = None
    if existing is not None:
        return existing
    state_dir = pathlib.Path(api.get_state_dir())
    key = os.path.realpath(state_dir)
    if key in _PRESENCE_RUNTIMES_BY_STATE and _PRESENCE_RUNTIMES_BY_STATE[key].api is api:
        return _PRESENCE_RUNTIMES_BY_STATE[key]
    try:
        host_port = int(os.environ.get("OUROBOROS_HOST_SERVICE_PORT", "8767"))
    except (TypeError, ValueError):
        host_port = 8767
    submitter = PresenceHostClient(
        state_dir=state_dir,
        token_provider=_skill_token_provider(api),
        host_base=f"http://127.0.0.1:{host_port}",
    )
    runtime = PresenceRuntime(
        api=api,
        state_dir=state_dir,
        submitter=submitter,
        trust_env=trust_env,
    )
    try:
        _PRESENCE_RUNTIMES[api] = runtime
    except TypeError:
        pass
    _PRESENCE_RUNTIMES_BY_STATE[key] = runtime
    return runtime


def try_presence_runtime(api: Any, *, trust_env: bool = False) -> Optional[PresenceRuntime]:
    try:
        return presence_runtime(api, trust_env=trust_env)
    except (OSError, sqlite3.Error):
        return None


async def mode_panel_for_chat(
    api: Any,
    chat_id: int,
    lang: str,
    *,
    is_bg_active: Callable[[Any], bool],
    runtime_getter: Callable[[Any], Optional[PresenceRuntime]],
) -> tuple[str, list[list[dict]]]:
    settings = _load_settings(api)
    bg_enabled: bool | None = None
    place_state = ""
    if owner_chat_mode(settings) == OWNER_CHAT_MODE_PRESENCE:
        bg_enabled = is_bg_active(api)
        runtime = runtime_getter(api)
        if runtime is not None:
            try:
                info = await runtime.place_readiness(str(chat_id))
                place_state = str(info.get("state") or "")
            except Exception:
                place_state = "unavailable"
    return build_mode_panel(settings, lang, place_state=place_state, bg_enabled=bg_enabled)


async def handle_mode_command(
    api: Any,
    client: Any,
    chat_id: int,
    text: str,
    lang: str,
    *,
    panel_builder: Callable[[Any, int, str], Any],
) -> None:
    live_pinned = str(_load_settings(api).get("TELEGRAM_CHAT_ID") or "").strip()
    if not live_pinned or str(chat_id) != live_pinned:
        return
    parts = text.strip().split()
    m = _MODE_TEXTS[lang]
    if len(parts) <= 1:
        header, keyboard = await panel_builder(api, chat_id, lang)
        await client.send_message_with_inline_keyboard(chat_id, header, keyboard)
        return
    sub = parts[1].lower()
    note = ""
    if len(parts) == 2 and sub in VALID_OWNER_CHAT_MODES:
        merge_settings(pathlib.Path(api.get_state_dir()), {"TELEGRAM_OWNER_CHAT_MODE": sub})
        note = m["mode_updated_presence"] if sub == OWNER_CHAT_MODE_PRESENCE else m["mode_updated_direct"]
    elif len(parts) == 3 and sub == "mirror" and parts[2].lower() in VALID_MIRROR_MODES:
        chosen = parts[2].lower()
        merge_settings(pathlib.Path(api.get_state_dir()), {"TELEGRAM_MIRROR_MODE": chosen})
        note = m.format("mirror_updated", mode=chosen)
    elif len(parts) == 3 and sub == "quizzes" and parts[2].lower() in {"on", "off"}:
        chosen = parts[2].lower()
        merge_settings(pathlib.Path(api.get_state_dir()), {"TELEGRAM_MIRROR_QUIZZES": chosen})
        note = m["quizzes_updated_on"] if chosen == "on" else m["quizzes_updated_off"]
    elif len(parts) == 3 and sub == "preset" and parts[2].lower() in {"presence_control", "direct_main"}:
        chosen = parts[2].lower()
        if chosen == "presence_control":
            merge_settings(
                pathlib.Path(api.get_state_dir()),
                {"TELEGRAM_OWNER_CHAT_MODE": OWNER_CHAT_MODE_PRESENCE, "TELEGRAM_MIRROR_MODE": MIRROR_MODE_CONTROL_ONLY},
            )
            note = m["preset_presence_applied"]
        else:
            merge_settings(
                pathlib.Path(api.get_state_dir()),
                {"TELEGRAM_OWNER_CHAT_MODE": OWNER_CHAT_MODE_DIRECT, "TELEGRAM_MIRROR_MODE": MIRROR_MODE_MAIN},
            )
            note = m["preset_direct_applied"]
    else:
        await client.send_message(chat_id, m["mode_usage"])
        return
    header, keyboard = await panel_builder(api, chat_id, lang)
    await client.send_message_with_inline_keyboard(chat_id, f"{note}\n\n{header}", keyboard)


async def handle_non_owner_update(
    api: Any,
    client: Any,
    update: Dict[str, Any],
    pinned_chat: str,
    *,
    runtime_getter: Callable[[Any], Optional[PresenceRuntime]],
    lang: str = "en",
    post: Any = None,
) -> None:
    """Route groups, channels, other private chats, membership changes and owner edits/reactions into Presence."""
    if update.get("callback_query"):
        return
    runtime = runtime_getter(api)
    if runtime is None:
        raise InjectNotAccepted("Presence runtime unavailable")
    record_only = False
    for key in ("message", "edited_message"):
        payload = update.get(key)
        if isinstance(payload, dict):
            chat = payload.get("chat") or {}
            reply_to = payload.get("reply_to_message") or {}
            if str(chat.get("type") or "") == "private" and isinstance(reply_to, dict) and reply_to:
                chat_id = int(chat.get("id") or 0)
                quiz_ref = telegram_quiz.quiz_for_message(api, client.bot_id, chat_id, int(reply_to.get("message_id") or 0))
                if quiz_ref and quiz_ref.get("secret"):
                    if callable(post) and pinned_chat and str(chat_id) == str(pinned_chat):
                        await telegram_quiz.answer_from_reply(
                            api, client, quiz_ref, "", chat_id=chat_id,
                            request_key=update_key(update), lang=lang, post=post,
                        )
                    return  # Secret replies and edits never touch SQLite custody or Presence turns.
    for key in ("edited_message", "message_reaction", "message_reaction_count"):
        payload = update.get(key)
        if isinstance(payload, dict):
            chat = payload.get("chat") or {}
            if (
                pinned_chat
                and str(chat.get("id") or "") == str(pinned_chat)
                and str(chat.get("type") or "") == "private"
            ):
                if key == "edited_message":
                    chat_id = int(chat.get("id") or 0)
                    reply_to = payload.get("reply_to_message") or {}
                    quiz_ref = (
                        telegram_quiz.quiz_for_message(api, client.bot_id, chat_id, int(reply_to.get("message_id") or 0))
                        if isinstance(reply_to, dict) and reply_to
                        else None
                    )
                    edit_text = str(payload.get("text") or payload.get("caption") or "").strip()
                    if quiz_ref is not None or edit_text.startswith("/"):
                        record_only = True
                        break
                try:
                    settings = _load_settings(api)
                except TelegramSettingsError:
                    settings = {}
                record_only = owner_chat_mode(settings) != OWNER_CHAT_MODE_PRESENCE
            break
    await runtime.ingest_update(update, account_id=client.bot_id, record_only=record_only)


async def maybe_send_presence_hint(
    api: Any,
    client: Any,
    chat_id: int,
    lang: str,
    runtime: PresenceRuntime,
    *,
    is_bg_active: Callable[[Any], bool],
) -> None:
    """Best-effort throttled hint when an owner DM message enters Presence while Presence cannot reply yet."""
    key = (os.path.realpath(api.get_state_dir()), int(chat_id))
    now = time.monotonic()
    last = _LAST_PRESENCE_HINT.get(key)
    if last is not None and now - last < _PRESENCE_HINT_THROTTLE_SEC:
        return
    bg_active = is_bg_active(api)
    place_state = ""
    try:
        info = await runtime.place_readiness(str(chat_id))
        place_state = str(info.get("state") or "")
    except Exception:
        place_state = "unavailable"
    hint = presence_readiness_hint(lang, place_state=place_state, bg_enabled=bg_active)
    if not hint:
        return
    _LAST_PRESENCE_HINT[key] = now
    try:
        await client.send_message(chat_id, hint)
    except Exception:
        pass


def make_presence_status(api: Any, *, runtime_getter: Callable[[Any], PresenceRuntime]):
    """Circles place-connector ``GET status`` route combining Presence custody and bridge state."""
    async def status(request):
        query_params = getattr(request, "query_params", None)
        requested_account = str((query_params.get("account_id") if query_params else "") or "").strip()

        def _read() -> Dict[str, Any]:
            runtime = runtime_getter(api)
            try:
                settings = _load_settings(api)
            except TelegramSettingsError:
                settings = {}
            return runtime.status_payload(
                _bridge_status(api),
                _read_status(api),
                settings,
                requested_account_id=requested_account,
            )

        return JSONResponse(await asyncio.to_thread(_read))

    return status
