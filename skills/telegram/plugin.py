from __future__ import annotations

import asyncio
import contextlib
import json
import os
import pathlib
import re
import time
from typing import Any, Dict

import httpx
from starlette.responses import JSONResponse

from ouroboros.host_notice import notice_line, with_instruction
from ouroboros.net_transport import env_proxies_configured
from ouroboros.project_facts import project_display_name
from ouroboros.utils import in_worker_process

from .lib.telegram_api import (
    OUTBOUND_JOURNAL, TELEGRAM_RETRY_INITIAL_SEC, TelegramBindingChanged, TelegramClient,
    TelegramRequestRejected, TelegramTokenMissing, TelegramTransportError,
    is_transient_telegram_error, markdown_to_telegram_html, message_chunks,
    replacing_request, next_telegram_retry_delay, _LOCALIZED_TEXTS, _MAX_TELEGRAM_UPLOAD_BYTES, _u16len,
)
from .lib.telegram_state import (
    _state_file, _load_settings, _is_silent_mode_enabled,
    _get_silent_msg, _set_silent_msg, _clear_silent_msg, _subagent_cards_enabled,
    _mirror_progress_enabled, _render_subagent_card, _data_dir,
    _load_runtime_state, _child_row_held_for_root, _child_final_report,
    _mirror_options, _message_ref, _owner_message_seen, _owner_ingress_fields, _replaceable_progress,
    _quiet_progress, _reply_target, _reply_ingress_transport, _carried_here, _bridged,
)
from .lib import telegram_i18n, telegram_inbound, telegram_quiz
from .lib.telegram_origin import origin_label
from .lib.telegram_health import _collect_health, _build_menu_tasks
from .lib.telegram_mirror import (
    MIRROR_MODE_CONTROL_ONLY,
    MIRROR_MODE_MAIN,
    OWNER_CHAT_MODE_DIRECT,
    OWNER_CHAT_MODE_PRESENCE,
    VALID_MIRROR_MODES,
    VALID_OWNER_CHAT_MODES,
    mirror_quizzes_enabled as _mirror_quizzes_enabled,
    owner_chat_mode as _owner_chat_mode,
    target_chat as _target_chat,
)
from .lib.telegram_panels import (
    _MODE_TEXTS,
    _bot_commands,
    _build_language_keyboard,
    _build_menu_keyboard,
    _build_menu_mind,
    _build_menu_settings,
    _build_menu_status,
    build_mode_panel,
    settings_section_schema,
)
from .lib.telegram_media_handlers import (
    is_native_audio_document as _is_native_audio_document,
    make_document_handler,
    make_links_handler,
    make_photo_handler,
    make_video_handler,
)
from .lib.telegram_presence_bridge import (
    handle_mode_command,
    handle_non_owner_update,
    make_presence_status,
    maybe_send_presence_hint,
    mode_panel_for_chat,
    presence_runtime,
    try_presence_runtime,
)
from .lib.presence.tools import register_presence_tools
from .lib.telegram_notifier import _make_notifier, _notify_enabled, _completion_text, _relay_text, _routing_notice_text
from .lib.telegram_outbox import OutboxDeferred, OutboxStopped, outbox_for
from .lib.telegram_share import make_prepare_share
from .lib.telegram_typing import TypingCue, host_reply_preparation
from .lib.telegram_poll_binding import (
    InjectNotAccepted, PollCursor, bot_id, handle_poll_batch, park_until_settings_change,
    saved_server as _saved_server, server_fingerprint, update_key,
)
from .lib.telegram_owner_auth import (
    api_base_matches_live, is_literal_slash_message, is_literal_span, is_pairing_command,
    is_panic_command, is_panic_text, issue_owner_pairing_code, owner_matches_live, redeem_owner_pairing_code,
)
from .lib.telegram_secret_guard import (
    begin_secret_send, classify_secret_exposure, finish_secret_send, record_secret_message,
)
from .lib.miniapp_registration import _read_status, register as register_miniapp
from .lib.telegram_token import _report_failure, _wait_for_bot_token, binding_for, make_token_observer
from .lib.telegram_connection import (  # status of the token, the bridge and the Mini App
    _COMMAND_MODE_FULL, _COMMAND_MODE_STRICT, _VALID_COMMAND_MODES, _fail_permanent_poll_rejection,
    _make_status, _save_bridge_status, poll_failure_reason, register_connection,
)
from .lib.telegram_connection import _bridge_status, _connection_fields  # noqa: F401 — the plugin's status facade
from .scripts.telegram_settings import (
    TelegramSettingsError,
    merge_settings,
    make_settings_save as _make_settings_save,
    request_may_change_owner,
    telegram_proxy,
)

# Decided once in the server process: a proxy-routed install keeps its only egress, every
# other install is isolated from ambient proxy and SSL_CERT env; the worker guard mirrors
# the core's macOS fork-safety rule. Residual: a CA-only install (custom CA via
# SSL_CERT_FILE/SSL_CERT_DIR, no proxy) loses Telegram egress until that CA is trusted
# system-wide or an ambient proxy is set. The explicit skill proxy is independent
# of this legacy bridge behavior; companion Telegram calls do not trust the env.
_HONOR_ENV_PROXIES = (not in_worker_process()) and env_proxies_configured()


def _bridge_client(api, event=None, chat_id: int = 0,
                   server: tuple[str, str | None] | None = None) -> tuple[TelegramClient, int]:
    """One bridge client and the token generation it is bound to: the accepted token (held in memory,
    lib/telegram_token.py) is read ONCE, and the same snapshot names the bot of its message references, so a
    change during construction makes the client refuse its first request instead of mixing two bots. The
    skill-local proxy and the saved server apply. No token raises ``TelegramTokenMissing``: a wait, not an error."""
    binding = binding_for(api)
    if binding.state == "pending":
        binding.read_now(api)  # before the observer's first read: once, never per request
    token, generation = binding.snapshot()
    if not token:
        raise TelegramTokenMissing("TELEGRAM_BOT_TOKEN is missing")
    api_base_url, files_dir = server or _saved_server(api)
    state_dir = pathlib.Path(api.get_state_dir())
    def _guard(sdir=state_dir, base=api_base_url, cid=int(chat_id or 0)) -> bool:
        return api_base_matches_live(sdir, base) and (cid <= 0 or owner_matches_live(sdir, cid))
    client = TelegramClient(token, trust_env=_HONOR_ENV_PROXIES,
                            proxy=telegram_proxy(state_dir),
                            **(_mirror_options(api, bot_id(token), event, chat_id) if event is not None else {}),
                            origin_label=origin_label(event, telegram_i18n.language()) if event is not None else "",
                            api_base_url=api_base_url, files_dir=files_dir, binding=binding, generation=generation,
                            binding_guard=_guard)
    return client, generation


def _telegram_client(api, event=None, chat_id: int = 0, server: tuple[str, str | None] | None = None) -> TelegramClient:
    return _bridge_client(api, event, chat_id, server)[0]


_POLL_GENERATION_STARTED = time.time()

_SLASH_COMMAND_RE = re.compile(r"^\s*/[A-Za-z]")
_MAIN_CMD_RE = re.compile(r"^/(?:main|m)(?:@[A-Za-z0-9_]+)?(?:\s+(.*))?$", re.IGNORECASE | re.DOTALL)


def _presence_runtime(api):
    return presence_runtime(api, trust_env=_HONOR_ENV_PROXIES)


def _try_presence_runtime(api):
    return try_presence_runtime(api, trust_env=_HONOR_ENV_PROXIES)

# In strict/safe modes slash commands are still controlled locally. In
# full_access mode a reviewed+granted chat transport is allowed to forward the
# same raw owner commands that the local UI accepts.
_COMMAND_TRANSLATIONS: dict[str, str] = {
    "/status": "/status",
    "/bg status": "/bg status",
    "/bg start": "/bg start",
    "/bg stop": "/bg stop",
    "/bg": "/bg",
}


# Which translation keys are available in safe mode (full_access forwards raw)
_SAFE_TRANSLATION_KEYS = frozenset({"/status", "/bg status", "/bg"})

def _setting_int(settings: Dict[str, Any], key: str, default: int, *, minimum: int = 1, maximum: int = 100) -> int:
    try:
        value = int(settings.get(key) or default)
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


def _translate_command(text: str, command_mode: str) -> str | None:
    """Return allowed chat text for a Telegram command, or None to reject."""
    if not _SLASH_COMMAND_RE.match(str(text or "")):
        return text  # Not a slash command — pass through unchanged
    normalized = str(text or "").strip().lower()
    if command_mode == _COMMAND_MODE_STRICT:
        return None  # All slash commands blocked
    if command_mode == _COMMAND_MODE_FULL:
        return str(text or "").strip()
    # Determine which translations are available for this mode
    for cmd_key in sorted(_SAFE_TRANSLATION_KEYS, key=len, reverse=True):
        if normalized == cmd_key:
            return _COMMAND_TRANSLATIONS[cmd_key]
    return None  # Unrecognized slash command — reject


def _is_exact_bot_command(text: str, command: str) -> bool:
    normalized = str(text or "").strip().lower()
    return normalized == command or (
        normalized.startswith(command + "@")
        and " " not in normalized
        and normalized.count("@") == 1
    )


async def _mode_panel_for_chat(api, chat_id: int, lang: str) -> tuple[str, list[list[dict]]]:
    return await mode_panel_for_chat(
        api, chat_id, lang, is_bg_active=_is_bg_consciousness_active, runtime_getter=_try_presence_runtime
    )


async def _handle_mode_command(api, client, chat_id: int, text: str, lang: str) -> None:
    await handle_mode_command(api, client, chat_id, text, lang, panel_builder=_mode_panel_for_chat)


async def _choose_language(api, requested: str, lang: str, notify) -> str:
    """Relay the owner's choice (a tag, a name, a description) to the host's ONE language
    writer (Host Service ``/ui/language`` → ``ui_i18n.choose_language``) and tell the owner
    through ``notify(text)``. Returns the tag now in effect, or "" when nothing changed."""
    t = _LOCALIZED_TEXTS[lang]

    async def tell(text: str) -> None:
        # Every reply follows the host's settled answer: a token change ends the reply, never the choice,
        # so the update is not given again and the host is not asked twice.
        try:
            await notify(text)
        except TelegramBindingChanged:
            api.log("info", "Telegram language reply not sent: the bot token changed; the choice stands.")

    try:
        status, body = await _host_post(api, "/ui/language", {"language": requested})
    except Exception as exc:
        await tell(t.format("lang_failed", reason=type(exc).__name__))
        return ""
    if status < 400:
        new_lang = str(body.get("language") or "")
        await tell(_LOCALIZED_TEXTS[new_lang].format("lang_changed", language=telegram_i18n.label(new_lang)))
        return new_lang
    code = str(body.get("code") or "")
    if code == "language_needs_model":
        await tell(t["lang_needs_model"])
    else:
        await tell(t.format("lang_failed", reason=str(body.get("error") or body.get("message") or status)))
    return ""


async def _migrate_bridge_language(api, settings: Dict[str, Any], lang: str) -> str:
    """Once: the bridge's retired private ``TELEGRAM_LANGUAGE`` becomes the install's
    interface language (an owner who chose Russian here before the install-wide language
    existed keeps Russian — now everywhere). Marked done whatever the host answered, so a
    refusal is logged once and never retried on every start."""
    target = telegram_i18n.migration_target(settings)
    if not target:
        return lang
    try:
        status, body = await _host_post(api, "/ui/language", {"language": target})
    except Exception as exc:
        api.log("warning", f"Telegram language migration deferred ({type(exc).__name__}); retried at the next start.")
        return lang
    outcome = str(body.get("language") or target) if status < 400 else f"refused:{body.get('code') or status}"
    merge_settings(pathlib.Path(api.get_state_dir()), {"TELEGRAM_LANGUAGE_MIGRATED": outcome})
    if status < 400:
        api.log("info", f"The bridge language {target} became the install's interface language.")
        return str(body.get("language") or target)
    api.log("warning", f"Telegram language migration refused by the host ({outcome}).")
    return lang


def _host_headers(api) -> Dict[str, str]:
    return {"X-Skill-Token": api.get_skill_token().use_in_request()}


async def _host_post(api, path: str, payload: Dict[str, Any]) -> tuple[int, Dict[str, Any]]:
    """POST one JSON payload to the loopback Host Service; return ``(status, body)``."""
    try:
        port = int(os.environ.get("OUROBOROS_HOST_SERVICE_PORT", "8767"))
    except (TypeError, ValueError):
        raise RuntimeError("Host Service port is invalid.") from None
    if not 1 <= port <= 65535:
        raise RuntimeError("Host Service port is invalid.")
    async with httpx.AsyncClient(
        timeout=60,
        trust_env=False,
        follow_redirects=False,
    ) as client:
        response = await client.post(
            f"http://127.0.0.1:{port}{path}",
            headers=_host_headers(api),
            json=payload,
        )
    try:
        body = response.json()
    except Exception:
        body = {}
    return int(response.status_code), (body if isinstance(body, dict) else {})


def _is_panic_text(text: Any) -> bool:
    return is_panic_text(text)


def _panic_update(update: Dict[str, Any]) -> bool:
    """Whether an update is a genuine /panic command (never code/literal or /panicking)."""
    return is_panic_command(update.get("message") or {})


# The pause before an update the host did not accept is asked again; any other failure is never retried.
_NOT_ACCEPTED_RETRY_SEC = 2.0


async def _inject(api, payload: Dict[str, Any]) -> None:
    settings = _load_settings(api)
    pinned_chat = str(settings.get("TELEGRAM_CHAT_ID") or "").strip()
    if not pinned_chat or str(payload.get("chat_id") or "").strip() != pinned_chat:
        api.log("warning", "Host inject refused: TELEGRAM_CHAT_ID is unbound or does not match sender.")
        return
    try:
        status, body = await _host_post(api, "/chat/inject", payload)
    except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout):
        raise InjectNotAccepted("Host inject unreachable before acceptance") from None
    if status == 429 or (status == 503 and str(body.get("code") or "") == "not_accepted"):
        raise InjectNotAccepted(str(body.get("error") or f"Host inject did not accept (HTTP {status})"))
    if status >= 400 or body.get("ok") is False:
        raise RuntimeError(f"Host inject returned HTTP {status}")
    # A new user turn starts here — break the silent-mode chain so the next
    # outbound message begins a fresh bubble rather than overwriting the last.
    try:
        chat_id = int(payload.get("chat_id") or 0)
        if chat_id:
            _clear_silent_msg(api, chat_id)
    except (TypeError, ValueError):
        pass


def _extract_sender_label(sender: dict, fallback_chat_id: int) -> str:
    """Build a human-readable sender label from a Telegram user dict."""
    return (
        str(sender.get("username") or "").strip()
        or " ".join(
            str(part).strip()
            for part in (sender.get("first_name"), sender.get("last_name"))
            if part
        )
        or f"Telegram {sender.get('id') or fallback_chat_id}"
    )


def _is_bg_consciousness_active(api) -> bool:
    """Check if background consciousness is actively enabled in state.json."""
    state_file = _data_dir(api) / "state" / "state.json"
    if state_file.exists():
        try:
            state_data = json.loads(state_file.read_text(encoding="utf-8"))
            return bool(state_data.get("bg_consciousness_enabled") or False)
        except Exception:
            pass
    return False


async def _compile_status_text(api, lang: str = "en") -> str:
    """Generate a clean HTML metrics block from state/settings."""
    runtime_state = await _load_runtime_state(api)
    try:
        spent_usd = (
            float(runtime_state["spent_usd"])
            if runtime_state.get("spent_usd") is not None
            else None
        )
    except (TypeError, ValueError):
        spent_usd = None
    try:
        total_budget = (
            float(runtime_state["budget_limit"])
            if runtime_state.get("budget_limit") is not None
            else None
        )
    except (TypeError, ValueError):
        total_budget = None
    branch = str(runtime_state.get("branch") or "unavailable")
    bg_value = runtime_state.get("bg_consciousness_enabled")
    bg_enabled = bool(bg_value) if isinstance(bg_value, bool) else None

    t = _LOCALIZED_TEXTS[lang]

    bg_status_raw = "unavailable"
    if bg_enabled is not None:
        bg_status_raw = t["bg_active_label"] if bg_enabled else t["bg_sleeping_label"]

    spent_spec = "unavailable" if spent_usd is None else f"{spent_usd:.4f}"
    unbounded = t["budget_unbounded"]
    total_spec = (
        "unavailable" if total_budget is None
        else unbounded if total_budget <= 0
        else f"{total_budget:.2f}"
    )
    rem_spec = (
        "unavailable" if spent_usd is None or total_budget is None
        else unbounded if total_budget <= 0
        else f"{max(0.0, total_budget - spent_usd):.4f}"
    )

    status_str = t.format("metrics_budget_status", spent_usd=spent_spec, total_budget=total_spec, rem=rem_spec,
                          branch=branch, bg_status=bg_status_raw)
    status_str += "\n" + await asyncio.to_thread(_collect_health, api, lang, runtime_state)
    return status_str


def _poller_preferences(api):
    settings = _load_settings(api)
    pinned = str(settings.get("TELEGRAM_CHAT_ID") or "").strip()
    maximum = _setting_int(settings, "TELEGRAM_MAX_UPDATES_PER_POLL", 20, minimum=1, maximum=100)
    mode = str(settings.get("TELEGRAM_COMMAND_MODE") or _COMMAND_MODE_FULL).strip().lower()
    mode = mode if mode in _VALID_COMMAND_MODES else _COMMAND_MODE_STRICT
    # The install's interface language: one choice for the web UI, the desktop window and
    # this bot (OUROBOROS_UI_LANGUAGE); "" reads as English.
    return settings, pinned, maximum, mode, telegram_i18n.language()


_LAST_COMMAND_SCOPE: Dict[str, tuple[str, str, str, tuple[str, str, str] | None, float]] = {}


def _should_sync_commands(state_key: str, bot_id_val: str, pinned_chat: str, api_base: str = "") -> bool:
    target = (bot_id_val, pinned_chat, api_base)
    synced_bot, synced_chat, synced_base, failed_target, failed_at = _LAST_COMMAND_SCOPE.get(state_key, ("", "", "", None, 0.0))
    return (synced_bot, synced_chat, synced_base) != target and (failed_target != target or time.monotonic() - failed_at >= 60.0)


async def _sync_bot_commands(api, client, command_mode: str, lang: str, pinned_chat: str) -> None:
    """Keep owner admin commands scoped to the bound owner DM and clear the public default scope."""
    state_key, api_base = os.path.realpath(api.get_state_dir()), str(getattr(client, "api_base_origin", "") or "")
    prev_bot, prev_chat, prev_base, _, _ = _LAST_COMMAND_SCOPE.get(state_key, ("", "", "", None, 0.0))
    try:
        if prev_bot == client.bot_id and prev_chat and prev_chat != pinned_chat and prev_chat.isdigit() and int(prev_chat) > 0:
            await client.call("setMyCommands", data={"commands": "[]", "scope": json.dumps({"type": "chat", "chat_id": int(prev_chat)})})
            prev_bot, prev_chat = "", ""
        await client.call("setMyCommands", data={"commands": "[]", "scope": json.dumps({"type": "default"})})
        if pinned_chat.isdigit() and int(pinned_chat) > 0:
            await client.call("setMyCommands", data={"commands": json.dumps(_bot_commands(command_mode, lang)),
                                                     "scope": json.dumps({"type": "chat", "chat_id": int(pinned_chat)})})
        _LAST_COMMAND_SCOPE[state_key] = (client.bot_id, pinned_chat, api_base, None, 0.0)
        api.log("info", "Telegram bot commands configured successfully")
    except TelegramBindingChanged:
        raise  # a stale client is never reported ready
    except Exception as exc:
        _LAST_COMMAND_SCOPE[state_key] = (prev_bot, prev_chat, prev_base, (client.bot_id, pinned_chat, api_base), time.monotonic())
        api.log("warning", f"Failed to set Telegram bot commands: {exc}")


_CMD_SYNC_TASKS: Dict[str, asyncio.Task[Any]] = {}


def _schedule_command_sync(api, client, command_mode: str, lang: str, pinned_chat: str) -> None:
    state_key = os.path.realpath(api.get_state_dir())
    if not _should_sync_commands(state_key, client.bot_id, pinned_chat, str(getattr(client, "api_base_origin", "") or "")): return
    prev = _CMD_SYNC_TASKS.get(state_key)
    if prev is not None and not prev.done(): return
    async def _bg() -> None:
        try:
            with contextlib.suppress(TelegramBindingChanged, asyncio.CancelledError): await _sync_bot_commands(api, client, command_mode, lang, pinned_chat)
        finally:
            if _CMD_SYNC_TASKS.get(state_key) is asyncio.current_task(): _CMD_SYNC_TASKS.pop(state_key, None)
    _CMD_SYNC_TASKS[state_key] = asyncio.create_task(_bg())


async def _authorized_owner(api, client, update, pinned_chat: str, lang: str) -> str | None:
    callback, message = update.get("callback_query") or {}, update.get("message") or {}
    chat = ((callback.get("message") or {}).get("chat") or (message.get("chat") or {}))
    sender = callback.get("from") or message.get("from") or {}
    chat_id, user_id = int(chat.get("id") or 0), int(sender.get("id") or 0)
    if chat_id <= 0 or user_id <= 0:
        return None
    live_pinned = str(_load_settings(api).get("TELEGRAM_CHAT_ID") or "").strip()
    if pinned_chat and not live_pinned:
        return None  # owner binding was cleared mid-batch; never rebind or keep old access
    active_pinned = live_pinned
    if not active_pinned:
        # Never infer owner authority from an arbitrary private DM; require explicit
        # Ouroboros Settings binding or a one-time pairing code issued from Ouroboros.
        if (not callback and str(chat.get("type") or "") == "private" and chat_id == user_id
                and redeem_owner_pairing_code(api.get_state_dir(), chat_id, str(message.get("text") or ""))):
            active_pinned = str(chat_id)
            update["_owner_paired"] = True
            try:
                await _sync_bot_commands(api, client, _poller_preferences(api)[3], lang, active_pinned)
            except TelegramBindingChanged:
                raise
            except Exception:
                pass
        else:
            if callback:
                with contextlib.suppress(Exception):
                    await client.answer_callback_query(str(callback.get("id") or ""), text=_LOCALIZED_TEXTS[lang]["not_authorized"])
            return None
    if str(chat.get("type") or "private") == "private" and str(chat_id) == active_pinned and str(user_id) == active_pinned:
        return active_pinned
    if callback:
        with contextlib.suppress(Exception):
            await client.answer_callback_query(str(callback.get("id") or ""), text=_LOCALIZED_TEXTS[lang]["not_authorized"])
    return None


async def _edit_panel(api, client, chat_id: int, message_id: int, text: str, keyboard: list[list[dict]]) -> None:
    if await client.edit_message_text_with_inline_keyboard(chat_id, message_id, text, keyboard):
        return
    api.log("warning", "Telegram panel edit failed; sending a fresh panel.")
    await client.send_message_with_inline_keyboard(chat_id, text, keyboard)


async def _validate_bot(api, client, command_mode: str, lang: str, generation: int | None = None) -> None:
    """Validate the token (getMe) and finish bridge + Presence startup bookkeeping."""
    me_payload = await client.call("getMe")
    me = me_payload.get("result") if isinstance(me_payload, dict) and isinstance(me_payload.get("result"), dict) else {}
    raw_me_id = me.get("id")
    if client.bot_id:
        if type(raw_me_id) is not int or raw_me_id <= 0 or str(raw_me_id) != client.bot_id:
            raise TelegramRequestRejected("Telegram getMe bot identity does not match token.", status_code=401)
    me_id = str(raw_me_id if type(raw_me_id) is int and raw_me_id > 0 else client.bot_id or "").strip()
    me_username = str(me.get("username") or "").strip()
    if generation is not None:
        binding = binding_for(api)
        if not binding.current(generation):
            raise TelegramBindingChanged(getattr(client, "accepted", 0))
        binding.mark_validated(generation)
    runtime = _try_presence_runtime(api)
    if runtime is not None:
        api_base = str(getattr(client, "api_base_origin", "") or "")
        runtime.on_bot_validated(client.token, me_id, me_username, api_base=api_base)
    _save_bridge_status(api, "ready", bot=client.bot_id, generation=generation)
    api.log("info", f"Telegram poller started (command_mode={command_mode}, lang={lang})")


async def _start_poller(api):
    client = generation = None  # once built: the client, and the exact generation a failure below belongs to
    try:
        _settings, pinned_chat, max_updates, command_mode, lang = _poller_preferences(api)
        lang = await _migrate_bridge_language(api, _settings, lang)
        server = _saved_server(api)
        client, generation = _bridge_client(api, server=server)  # the exact generation this client was built from
        cursor = PollCursor(_state_file(api, "poll_offset.json"), server[0], bot_id(client.token))
        try:
            await _validate_bot(api, client, command_mode, lang, generation)
            _schedule_command_sync(api, client, command_mode, lang, pinned_chat)
            validated = True
        except Exception as exc:
            if not is_transient_telegram_error(exc):
                raise
            # A dead network at startup must not burn the supervised-restart
            # budget: stay alive and revalidate from the polling loop once
            # transport returns. Permanent rejections still raise below.
            validated = False
            # Truthful status while validation is deferred (#376): neither
            # ready nor failed. The in-loop validation overwrites it.
            _save_bridge_status(api, "degraded", "telegram_startup_deferred", bot=client.bot_id, generation=generation)
            api.log("warning", "Telegram token validation deferred by a transport failure; polling waits for the network.")
        return client, generation, server, cursor, pinned_chat, max_updates, command_mode, lang, validated
    except (TelegramBindingChanged, TelegramTokenMissing):
        raise  # the token changed or went away: the caller waits and starts again, nothing failed
    except TelegramSettingsError:
        _save_bridge_status(api, "error", "settings_invalid")
        api.log("error", "Telegram settings are invalid; owner binding is closed.")
        raise
    except Exception as exc:
        bot = client.bot_id if client is not None else bot_id(binding_for(api).snapshot()[0])
        if isinstance(exc, TelegramRequestRejected):  # Telegram itself refused the token (getMe)
            _save_bridge_status(api, "error", "telegram_startup_failed", bot=bot, generation=generation)
            api.log("error", f"Telegram token validation failed: {exc}")
        elif isinstance(exc, ValueError):  # a saved proxy or Bot API server that cannot be used; not the token
            _save_bridge_status(api, "error", "connection_settings_invalid", bot=bot, generation=generation)
            api.log("error", f"Telegram connection settings are invalid: {exc}")
        else:
            _save_bridge_status(api, "error", "startup_error", bot=bot, generation=generation)
            api.log("error", f"Telegram bridge could not start ({type(exc).__name__}): {exc}")
        raise


async def _handle_poll_callback(api, client, update, pinned_chat, command_mode, lang, confirm):
    """Handle an authorized inline panel press and return the language for this poll."""
    callback_query = update["callback_query"]
    cb_id = str(callback_query.get("id") or "")
    cb_data = str(callback_query.get("data") or "").strip()
    cb_message = callback_query.get("message") or {}
    cb_message_id = int(cb_message.get("message_id") or 0)
    cb_chat = cb_message.get("chat") or {}
    cb_chat_id = int(cb_chat.get("id") or 0)
    cb_sender = callback_query.get("from") or {}
    if (
        not pinned_chat
        or str(cb_chat_id) != pinned_chat
        or str(cb_sender.get("id") or "") != pinned_chat
    ):
        await client.answer_callback_query(cb_id, text=_LOCALIZED_TEXTS[lang]["not_authorized"])
        return lang

    # --- Dynamic Tab Navigation (Category 1) ---
    if cb_data.startswith("nav:"):
        target = cb_data.split(":", 1)[1]
        await client.answer_callback_query(cb_id)
        if target == "menu":
            header, keyboard = _build_menu_keyboard(command_mode, lang)
            await _edit_panel(api, client, cb_chat_id, cb_message_id, header, keyboard)
        elif target == "status":
            info_text = await _compile_status_text(api, lang)
            header, keyboard = _build_menu_status(command_mode, lang, info_text)
            await _edit_panel(api, client, cb_chat_id, cb_message_id, header, keyboard)
        elif target == "mind":
            bg_enabled = _is_bg_consciousness_active(api)
            header, keyboard = _build_menu_mind(command_mode, lang, bg_enabled)
            await _edit_panel(api, client, cb_chat_id, cb_message_id, header, keyboard)
        elif target == "language":
            header, keyboard = _build_language_keyboard(lang)
            await _edit_panel(api, client, cb_chat_id, cb_message_id, header, keyboard)
        elif target == "tasks":
            header, keyboard = await asyncio.to_thread(  # the window's works (/api/state)
                _build_menu_tasks, await _load_runtime_state(api), command_mode, lang
            )
            await _edit_panel(api, client, cb_chat_id, cb_message_id, header, keyboard)
        elif target == "settings":
            header, keyboard = _build_menu_settings(api, command_mode, lang)
            await _edit_panel(api, client, cb_chat_id, cb_message_id, header, keyboard)
        elif target == "mode":
            header, keyboard = await _mode_panel_for_chat(api, cb_chat_id, lang)
            await _edit_panel(api, client, cb_chat_id, cb_message_id, header, keyboard)
        return lang

    # --- Command Actions / Control (Category 2) ---
    if cb_data.startswith("cmd_act:"):
        action = cb_data.split(":", 1)[1]

        if action == "update_status":
            await client.answer_callback_query(cb_id, text=_LOCALIZED_TEXTS[lang]["updating_status"])
            info_text = await _compile_status_text(api, lang)
            header, keyboard = _build_menu_status(command_mode, lang, info_text)
            await _edit_panel(api, client, cb_chat_id, cb_message_id, header, keyboard)
            return lang

        elif action == "toggle_silent":
            # Toggle TELEGRAM_SILENT_MODE and refresh the Settings panel.
            # This is a display preference (no LLM injection), so it is
            # allowed in every command_mode including strict.
            local_settings = _load_settings(api)
            currently_on = _is_silent_mode_enabled(local_settings)
            new_value = "off" if currently_on else "on"
            merge_settings(pathlib.Path(api.get_state_dir()), {"TELEGRAM_SILENT_MODE": new_value})
            confirm()  # the toggle is applied: a later token change must not replay it
            # Clear any stale tracked message id for this chat so the
            # next outbound starts a fresh bubble in either direction.
            _clear_silent_msg(api, cb_chat_id)
            toast_key = "quiet_toggled_on" if new_value == "on" else "quiet_toggled_off"
            await client.answer_callback_query(cb_id, text=_LOCALIZED_TEXTS[lang][toast_key])
            header, keyboard = _build_menu_settings(api, command_mode, lang)
            await _edit_panel(api, client, cb_chat_id, cb_message_id, header, keyboard)
            return lang

        elif action == "toggle_quizzes":
            local_settings = _load_settings(api)
            new_value = "off" if _mirror_quizzes_enabled(local_settings) else "on"
            merge_settings(pathlib.Path(api.get_state_dir()), {"TELEGRAM_MIRROR_QUIZZES": new_value})
            confirm()
            m = _MODE_TEXTS[lang]
            await client.answer_callback_query(
                cb_id, text=m["quizzes_updated_on"] if new_value == "on" else m["quizzes_updated_off"]
            )
            header, keyboard = await _mode_panel_for_chat(api, cb_chat_id, lang)
            await _edit_panel(api, client, cb_chat_id, cb_message_id, header, keyboard)
            return lang

        elif action == "bg_thoughts":
            # The retired thoughts button of a panel sent earlier: settle its spinner
            # and show the current panel, reading nothing.
            await client.answer_callback_query(cb_id)
            bg_enabled = _is_bg_consciousness_active(api)
            header, keyboard = _build_menu_mind(command_mode, lang, bg_enabled)
            await _edit_panel(api, client, cb_chat_id, cb_message_id, header, keyboard)
            return lang

        elif action in ("bg_start", "bg_stop"):
            if command_mode != _COMMAND_MODE_FULL:
                await client.answer_callback_query(cb_id, text=_LOCALIZED_TEXTS[lang]["restricted_safe"])
                return lang
            await client.answer_callback_query(cb_id, text=_LOCALIZED_TEXTS[lang]["injecting_consciousness"])
            translated = "/bg start" if action == "bg_start" else "/bg stop"
            sender_name = _extract_sender_label(cb_sender, cb_chat_id)
            sender_label = f"Telegram ({sender_name})"
            await _inject(api, {
                "text": translated,
                "chat_id": cb_chat_id,
                "user_id": int(cb_sender.get("id") or cb_chat_id or 1),
                "source": "telegram",
                "sender_label": sender_label,
                "transport": {
                    "kind": "telegram",
                    "conversation_id": str(cb_chat_id),
                    "sender_label": sender_label,
                },
                "image_base64": "",
                "image_mime": "",
                "image_caption": "",
                # The press is the owner's; the command is this button's words, not theirs.
                "ui_control": {"name": "telegram_mind_panel"},
            })
            confirm()  # the host took the command: it is not injected again
            # Give it a tiny moment to commit setting then refresh mind panel
            await asyncio.sleep(0.8)
            bg_enabled = _is_bg_consciousness_active(api)
            header, keyboard = _build_menu_mind(command_mode, lang, bg_enabled)
            await _edit_panel(api, client, cb_chat_id, cb_message_id, header, keyboard)
            return lang

    # --- Handle owner DM chat mode, mirror mode and preset buttons ---
    if cb_data.startswith("set_mode:"):
        chosen = cb_data.split(":", 1)[1].strip().lower()
        if chosen in VALID_OWNER_CHAT_MODES:
            merge_settings(pathlib.Path(api.get_state_dir()), {"TELEGRAM_OWNER_CHAT_MODE": chosen})
            confirm()
            m = _MODE_TEXTS[lang]
            toast = m["mode_updated_presence"] if chosen == OWNER_CHAT_MODE_PRESENCE else m["mode_updated_direct"]
            await client.answer_callback_query(cb_id, text=toast)
            header, keyboard = await _mode_panel_for_chat(api, cb_chat_id, lang)
            await _edit_panel(api, client, cb_chat_id, cb_message_id, header, keyboard)
            return lang

    if cb_data.startswith("set_mirror:"):
        chosen = cb_data.split(":", 1)[1].strip().lower()
        if chosen in VALID_MIRROR_MODES:
            merge_settings(pathlib.Path(api.get_state_dir()), {"TELEGRAM_MIRROR_MODE": chosen})
            confirm()
            await client.answer_callback_query(cb_id, text=_MODE_TEXTS[lang].format("mirror_updated", mode=chosen))
            header, keyboard = await _mode_panel_for_chat(api, cb_chat_id, lang)
            await _edit_panel(api, client, cb_chat_id, cb_message_id, header, keyboard)
            return lang

    if cb_data.startswith("set_preset:"):
        chosen = cb_data.split(":", 1)[1].strip().lower()
        m = _MODE_TEXTS[lang]
        if chosen == "presence_control":
            merge_settings(
                pathlib.Path(api.get_state_dir()),
                {"TELEGRAM_OWNER_CHAT_MODE": OWNER_CHAT_MODE_PRESENCE, "TELEGRAM_MIRROR_MODE": MIRROR_MODE_CONTROL_ONLY},
            )
            confirm()
            await client.answer_callback_query(cb_id, text=m["preset_presence_applied"])
            header, keyboard = await _mode_panel_for_chat(api, cb_chat_id, lang)
            await _edit_panel(api, client, cb_chat_id, cb_message_id, header, keyboard)
            return lang
        if chosen == "direct_main":
            merge_settings(
                pathlib.Path(api.get_state_dir()),
                {"TELEGRAM_OWNER_CHAT_MODE": OWNER_CHAT_MODE_DIRECT, "TELEGRAM_MIRROR_MODE": MIRROR_MODE_MAIN},
            )
            confirm()
            await client.answer_callback_query(cb_id, text=m["preset_direct_applied"])
            header, keyboard = await _mode_panel_for_chat(api, cb_chat_id, lang)
            await _edit_panel(api, client, cb_chat_id, cb_message_id, header, keyboard)
            return lang

    # --- Handle language selection buttons ---
    if cb_data.startswith("set_lang:"):
        requested = cb_data.split(":", 1)[1]

        async def _toast(text: str, _cb_id: str = cb_id) -> None:
            await client.answer_callback_query(_cb_id, text=text)

        new_lang = await _choose_language(api, requested, lang, _toast)
        confirm()  # the host settled the choice: it is not asked again
        if new_lang or requested == "en":
            lang = new_lang
        # Smoothly return to menu panel in the language now in effect
        header, keyboard = _build_menu_keyboard(command_mode, lang)
        await _edit_panel(api, client, cb_chat_id, cb_message_id, header, keyboard)
        return lang

    # --- Quiz answers (#472): a tapped option reaches the host's
    # decision ingress; the owner is already verified above.
    if cb_data.startswith("qz:"):
        await telegram_quiz.answer_from_callback(
            api, client, cb_data, cb_id=cb_id,
            request_key=update_key(update), lang=lang, post=_host_post,
        )
        return lang

    if cb_data.startswith(("set_model:", "set_budget:")):
        await client.answer_callback_query(
            cb_id,
            text=_LOCALIZED_TEXTS[lang]["use_miniapp"],
        )
        return lang

    await client.answer_callback_query(cb_id, text=_LOCALIZED_TEXTS[lang]["unknown_command"])
    return lang


async def _poller(api) -> None:
    binding = binding_for(api)

    def accepted_token() -> str:
        return binding.snapshot()[0]

    while True:
        await _wait_for_bot_token(api, binding)
        fingerprint = server_fingerprint(api, accepted_token())
        try:
            started = await _start_poller(api)
            break
        except TelegramRequestRejected as exc:
            runtime = _try_presence_runtime(api)
            if runtime is not None:
                rejected_token = fingerprint[0] if fingerprint is not None else accepted_token()
                runtime.on_token_rejected(rejected_token, str(exc))
            await park_until_settings_change(api, fingerprint, binding)
        except (TelegramBindingChanged, TelegramTokenMissing):
            continue
    client, generation, server, cursor, pinned_chat, max_updates, command_mode, lang, validated = started

    retry_delay = TELEGRAM_RETRY_INITIAL_SEC
    degraded_cause = ""
    degraded = "" if validated else "telegram_startup_deferred"  # the degraded reason last persisted, if any
    while True:
        revalidating = False
        try:
            if not accepted_token():
                runtime = _try_presence_runtime(api)
                if runtime is not None:
                    runtime.on_token_cleared()
                await _wait_for_bot_token(api, binding)
            saved = _saved_server(api)
            token_changed = not binding.current(generation)
            if saved != server or token_changed:
                # A Save reaches polling at the next cycle, as it reaches every send at once:
                # getUpdates, the replies and the downloads made here follow it. A new server
                # or token validates there again; a new server or another bot is polled from its
                # own oldest update, while a rotated token of the same bot keeps its position.
                validated = validated and not token_changed and server is not None and saved[0] == server[0]
                (client, generation), server = _bridge_client(api, server=saved), saved
                cursor.follow(saved[0], bot_id(client.token))
                if token_changed:
                    runtime = _try_presence_runtime(api)
                    if runtime is not None:
                        runtime.on_token_changed()
                    _save_bridge_status(api, "connecting", "telegram_token_changed", bot=client.bot_id, generation=generation)
                    api.log("info", "Telegram bot token changed; polling follows the saved token.")
                    degraded = ""
                else:
                    if not validated:
                        runtime = _try_presence_runtime(api)
                        if runtime is not None:
                            runtime.on_token_changed()
                    api.log("info", "Telegram Bot API settings changed; polling follows the saved server.")
            if not validated:
                revalidating = True
                await _validate_bot(api, client, command_mode, lang, generation)  # it records ready
                revalidating, validated, degraded = False, True, ""
                _schedule_command_sync(api, client, command_mode, lang, pinned_chat)
            updates = await client.get_updates(cursor.offset)
            if not binding.current(generation):
                # Answered for the previous token: neither confirmed nor handled, so the same bot's
                # rotated token receives these updates again and another bot never sees them.
                continue
            runtime = _try_presence_runtime(api)
            if runtime is not None:
                runtime.on_poll_ok()
            if degraded_cause:
                api.log("info", f"Telegram poller recovered after {degraded_cause}; polling resumed.")
                degraded_cause = ""
            if degraded:
                _save_bridge_status(api, "ready", bot=client.bot_id, generation=generation)
                degraded = ""
            retry_delay = TELEGRAM_RETRY_INITIAL_SEC
            if updates:
                local_settings, pinned_chat, max_updates, command_mode, lang = _poller_preferences(api)
            elif validated:
                try:
                    pinned_chat = str(_load_settings(api).get("TELEGRAM_CHAT_ID") or "").strip()
                except Exception:
                    pass
            deferred, lang = await handle_poll_batch(
                api, client, updates, cursor, binding, generation, pinned_chat, max_updates, command_mode, lang,
                authorize=_authorized_owner, is_panic=_panic_update, on_message=_handle_owner_message,
                on_callback=_handle_poll_callback, on_non_owner=_handle_non_owner_update,
            )
            # Never block emergency polling behind slow Telegram menu syncs.
            if validated and not deferred: _schedule_command_sync(api, client, command_mode, lang, pinned_chat)
            await asyncio.sleep(_NOT_ACCEPTED_RETRY_SEC if deferred else 0.1)
        except (TelegramBindingChanged, TelegramTokenMissing):
            continue  # the next cycle builds the client of the accepted token, or waits for one
        except (TelegramRequestRejected, TelegramTransportError) as exc:
            # Only getUpdates and getMe reach here: a 409 is another poller of this token, which may stop.
            if isinstance(exc, TelegramRequestRejected) and not exc.transient and exc.status_code != 409:
                runtime = _try_presence_runtime(api)
                if runtime is not None:
                    runtime.on_token_rejected(client.token, str(exc))
                _fail_permanent_poll_rejection(api, exc, revalidating, client.bot_id, generation)
                await park_until_settings_change(api, (client.token, server), binding)
                server, validated, degraded = None, False, ""
                continue
            # Transition logging only: one line entering degraded, one on
            # recovery; consecutive transient failures stay quiet while the
            # monotone backoff (reset by any successful poll) paces retries.
            runtime = _try_presence_runtime(api)
            if runtime is not None:
                runtime.on_poll_error(str(exc))
            if not degraded_cause:
                degraded_cause = type(exc).__name__
                api.log("warning", f"Telegram poller degraded ({degraded_cause}): {exc}")
            reason = poll_failure_reason(exc, revalidating)  # the status follows each change of cause
            if reason != degraded:
                _save_bridge_status(api, "degraded", reason, bot=client.bot_id, generation=generation)
                degraded = reason
            await asyncio.sleep(retry_delay)
            retry_delay = next_telegram_retry_delay(retry_delay)
        except TelegramSettingsError:
            _save_bridge_status(api, "error", "settings_invalid", generation=generation)
            api.log("error", "Telegram settings are invalid; owner binding is closed.")
            raise
        except Exception as exc:
            _save_bridge_status(api, "error", "poller_failed", bot=client.bot_id, generation=generation)
            api.log("error", f"Telegram poller failed ({type(exc).__name__}).")
            raise


async def _handle_non_owner_update(api, client, update: Dict[str, Any], pinned_chat: str) -> None:
    await handle_non_owner_update(
        api,
        client,
        update,
        pinned_chat,
        runtime_getter=_try_presence_runtime,
        lang=telegram_i18n.language(),
        post=_host_post,
    )


async def _maybe_send_presence_hint(api, client, chat_id: int, lang: str, runtime) -> None:
    await maybe_send_presence_hint(api, client, chat_id, lang, runtime, is_bg_active=_is_bg_consciousness_active)


async def _record_owner_sighting(api, client, update: Dict[str, Any]) -> None:
    runtime = _try_presence_runtime(api)
    if runtime is not None:
        try:
            await runtime.ingest_update(update, account_id=client.bot_id, record_only=True)
        except Exception:
            pass


async def _handle_owner_message(
    api, client, update: Dict[str, Any], update_id: int, command_mode: str, lang: str, *, reply_target=None,
) -> None:
    """One authorized owner message: local /menu /mode /language /help /panic, /main override,
    command-mode translation, quiz replies, then Presence custody (presence mode) or Main inject (direct mode).
    Owner binding is already enforced by the caller (explicit TELEGRAM_CHAT_ID or one-time pairing).
    """
    ingested_to_presence = False

    def mark_ingested() -> None:
        nonlocal ingested_to_presence
        ingested_to_presence = True

    try:
        await _handle_owner_message_inner(
            api, client, update, update_id, command_mode, lang,
            reply_target=reply_target, mark_ingested=mark_ingested,
        )
    finally:
        if not ingested_to_presence:
            await _record_owner_sighting(api, client, update)


async def _handle_owner_message_inner(
    api, client, update: Dict[str, Any], update_id: int, command_mode: str, lang: str,
    *, reply_target=None, mark_ingested=lambda: None,
) -> None:
    message = update.get("message") or {}
    chat = message.get("chat") or {}
    sender = message.get("from") or {}
    chat_id = int(chat.get("id") or 0)
    _owner_message_seen(api, client.bot_id, chat_id, int(message.get("message_id") or 0))
    # Owner binding + filtering is already enforced at the top of
    # the update loop (explicit binding or one-time pairing), so chat_id is
    # guaranteed to equal the pinned owner chat here.
    text = str(message.get("text") or message.get("caption") or "").strip()
    caption = str(message.get("caption") or "").strip()
    cleaned_text = text.lower().strip()
    literal_slash = is_literal_slash_message(message)
    if not literal_slash and _is_exact_bot_command(cleaned_text, "/menu"):
        header, keyboard = _build_menu_keyboard(command_mode, lang)
        if keyboard:
            await client.send_message_with_inline_keyboard(chat_id, header, keyboard)
        else:
            await client.send_message(chat_id, header)
        return
    if not literal_slash and (_is_exact_bot_command(cleaned_text, "/mode") or cleaned_text.startswith("/mode ") or cleaned_text.startswith("/mode@")):
        await _handle_mode_command(api, client, chat_id, text, lang)
        return
    if not literal_slash and _is_exact_bot_command(cleaned_text, "/language"):
        header, keyboard = _build_language_keyboard(lang)
        await client.send_message_with_inline_keyboard(chat_id, header, keyboard)
        return
    if not literal_slash and _is_exact_bot_command(cleaned_text, "/help"):
        await client.send_message(chat_id, _LOCALIZED_TEXTS[lang]["help_text"])
        return

    # Never forward /pair <code...> or /start <pairing_code> into chat history or Presence custody.
    if update.get("_owner_paired") or is_pairing_command(api, message):
        await client.send_message(chat_id, _LOCALIZED_TEXTS[lang].get("paired_ok", "Owner chat confirmed."))
        return

    if _panic_update(update):
        # If someone sent /panic as a reply to a secret card, quarantine its message_id too.
        api_base = str(getattr(client, "api_base_origin", "") or "")
        classify_secret_exposure(api, client.bot_id, message, api_base=api_base)
        if int(message.get("date") or 0) < int(_POLL_GENERATION_STARTED):
            await client.send_message(chat_id, _LOCALIZED_TEXTS[lang]["panic_earlier"])
            return
        try:
            sender_label = f"Telegram ({_extract_sender_label(sender, chat_id)})"
            await asyncio.wait_for(_inject(api, {**_owner_ingress_fields(api, client.bot_id, chat_id, message, update_id),
                "text": "/panic", "chat_id": chat_id,
                "sender_label": sender_label, "transport": {"kind": "telegram", "conversation_id": str(chat_id), "sender_label": sender_label},
                "user_id": int(sender.get("id") or chat_id), "source": "telegram"}), timeout=2)
        except InjectNotAccepted:
            raise
        except Exception:
            api.log("warning", "Immediate Telegram Panic was not confirmed; it will not be retried.")
            try:
                await client.send_message(chat_id, _LOCALIZED_TEXTS[lang]["panic_unconfirmed"])
            except TelegramBindingChanged:
                pass  # the notice lost its client; the unknown Panic is still not retried
        return

    # Fail-closed secret exposure check BEFORE any download, command routing, Presence ingest, or inject:
    # direct replies check quiz_for_message once; quotes/edits/forwards/uncertain cards check secret_guard.
    api_base = str(getattr(client, "api_base_origin", "") or "")
    reply_to = message.get("reply_to_message") or {}
    quiz_ref = None
    if isinstance(reply_to, dict) and reply_to:
        try:
            quiz_ref = telegram_quiz.quiz_for_message(api, client.bot_id, chat_id, int(reply_to.get("message_id") or 0))
        except Exception:
            quiz_ref = {"chat_id": chat_id, "message_id": int(reply_to.get("message_id") or 0), "options": [], "secret": True, "unreadable": True}
    if quiz_ref and quiz_ref.get("secret"):
        record_secret_message(api, client.bot_id, chat_id, int(message.get("message_id") or 0), api_base=api_base, kind="refused_reply")
        await telegram_quiz.answer_from_reply(api, client, quiz_ref, "", chat_id=chat_id,
                                              request_key=update_key(update), lang=lang, post=_host_post)
        return
    secret_ref = classify_secret_exposure(
        api, client.bot_id, message, api_base=api_base, is_edit=False,
        known_non_secret_fn=lambda cid, mid: bool(_message_ref(api, client.bot_id, cid, mid)),
    )
    if secret_ref is not None:
        await telegram_quiz.answer_from_reply(api, client, secret_ref, "", chat_id=chat_id,
                                              request_key=update_key(update), lang=lang, post=_host_post)
        return

    if not literal_slash and (cleaned_text.startswith("/language ") or cleaned_text.startswith("/language@")):
        requested = text.split(None, 1)[1].strip() if len(text.split(None, 1)) > 1 else ""
        if requested:
            async def _reply(message: str) -> None:
                await client.send_message(chat_id, message)

            await _choose_language(api, requested, lang, _reply)
            return

    # Explicit one-shot Main escape hatch: `/main <text>` or `/m <text>` bypasses Presence mode.
    force_direct = False
    main_match = None if literal_slash else _MAIN_CMD_RE.match(text)
    if main_match:
        inner = (main_match.group(1) or "").strip()
        if not inner and not (message.get("photo") or []) and telegram_inbound.inbound_file(message) is None:
            await client.send_message(chat_id, _MODE_TEXTS[lang]["main_empty"])
            return
        force_direct = True
        raw_src = str(message.get("text") or "") if str(message.get("text") or "").strip() else str(message.get("caption") or "")
        lead_off = len(raw_src) - len(raw_src.lstrip())
        inner_start = lead_off + main_match.start(1)
        text = inner
        cap_match = _MAIN_CMD_RE.match(caption) if caption else None
        if cap_match:
            caption = (cap_match.group(1) or "").strip()
        cap_start = (len(str(message.get("caption") or "")) - len(str(message.get("caption") or "").lstrip()) + cap_match.start(1)) if cap_match else None
        literal_slash = is_literal_span(message, text, start=inner_start) or (bool(caption) and is_literal_span(message, caption, start=cap_start))

    is_slash_cmd = (not literal_slash) and bool(_SLASH_COMMAND_RE.match(text) or (caption and _SLASH_COMMAND_RE.match(caption)))

    # Translate commands to safe natural-language text (code/pre literal text stays non-command).
    safe_text = text if literal_slash else _translate_command(text, command_mode)
    safe_caption = caption if literal_slash else (_translate_command(caption, command_mode) if caption else caption)
    if safe_text is None or safe_caption is None:
        if command_mode == _COMMAND_MODE_STRICT:
            await client.send_message(
                chat_id,
                _LOCALIZED_TEXTS[lang]["slash_blocked_strict"],
            )
        else:
            await client.send_message(
                chat_id,
                _LOCALIZED_TEXTS[lang]["slash_blocked_mode"],
            )
        return

    if not force_direct and quiz_ref is not None and quiz_ref.get("ambiguous") and not is_slash_cmd:
        # An old card's number names no question for sure: an answer or a turn meant for it stays
        # here, before any download; a command keeps its dispatch, as below.
        await telegram_quiz.answer_from_reply(api, client, quiz_ref, "", chat_id=chat_id,
                                              request_key=update_key(update), lang=lang, post=_host_post)
        return
    if not force_direct and safe_text and not is_slash_cmd and isinstance(reply_to, dict) and reply_to:
        if quiz_ref is not None:
            # A reply to a quiz card is the owner's own answer to
            # that question, not a new chat turn (#472).
            # Commands keep their existing dispatch even in replies. Other
            # answers, including code-formatted command names, stay verbatim.
            answer_text = str(message.get("text") or message.get("caption") or "")
            await telegram_quiz.answer_from_reply(
                api, client, quiz_ref, answer_text, chat_id=chat_id,
                request_key=update_key(update), lang=lang, post=_host_post,
            )
            return

    if literal_slash:
        # Defense in depth: re-quote leading-slash code/pre text so neither Host nor legacy paths treat it as a command.
        if safe_text.startswith("/"):
            safe_text = f"`{safe_text}`"
        if safe_caption.startswith("/"):
            safe_caption = f"`{safe_caption}`"

    # Dual-lane Owner DM routing: in `presence` mode, ordinary non-command messages commit to
    # Presence SQLite custody and fail closed (never silently falling back to Main `/chat/inject`).
    if not force_direct and not is_slash_cmd and _owner_chat_mode(_load_settings(api)) == OWNER_CHAT_MODE_PRESENCE:
        presence_rt = _try_presence_runtime(api)
        if presence_rt is None:
            raise InjectNotAccepted("Presence runtime unavailable")
        inserted = await presence_rt.ingest_update(update, account_id=client.bot_id, record_only=False)
        mark_ingested()
        if inserted:
            await _maybe_send_presence_hint(api, client, chat_id, lang, presence_rt)
        return
    def read_target():
        return _reply_target(api, client.bot_id, chat_id, message)

    target = reply_target(read_target) if reply_target is not None else read_target()
    photos = message.get("photo") or []
    image_base64 = ""
    image_mime = ""
    if photos:
        file_id = str((photos[-1] or {}).get("file_id") or "").strip()
        if file_id:
            image_base64, image_mime = await client.download_photo(file_id)
    inbound = None if photos else telegram_inbound.inbound_file(message)
    if inbound is not None and inbound.get("refusal"):
        await client.send_message(chat_id, telegram_inbound.refusal_text(inbound, lang))
        return
    if not safe_text and not image_base64 and inbound is None:
        # Acknowledge an unsupported message kind (sticker, poll,
        # location…) instead of silently swallowing it.
        if telegram_inbound.unsupported_kind(message):
            await client.send_message(chat_id, telegram_inbound.unsupported_text(lang))
        return
    sender_name = _extract_sender_label(sender, chat_id)
    sender_label = f"Telegram ({sender_name})"
    parked = None
    if inbound is not None:
        # Documents, video, audio and voice ride the host's shared
        # attachment path (#668): parked in this skill's state dir,
        # copied by the host into data/uploads, removed here after.
        parked = await telegram_inbound.park_inbound_file(api, client, inbound)
    try:
        await _inject(api, {
            **_owner_ingress_fields(api, client.bot_id, chat_id, message, update_id, target),
            "text": safe_text,
            "chat_id": chat_id,
            "user_id": int(sender.get("id") or chat_id or 1),
            "source": "telegram",
            "sender_label": sender_label,
            "transport": {"kind": "telegram", "conversation_id": str(chat_id), "sender_label": sender_label,
                          **_reply_ingress_transport(target, client.bot_id, int(message.get("message_id") or 0))},
            "image_base64": image_base64,
            "image_mime": image_mime,
            "image_caption": safe_caption,
            **({"literal_text": True} if literal_slash else {}),
            **({"attachments": [parked.spec]} if parked is not None else {}),
        })
    finally:
        if parked is not None:
            parked.cleanup()
    if target.get("unlinked"):
        try:
            await client.send_message(chat_id, _LOCALIZED_TEXTS[lang]["reply_not_linked"])
        except Exception:
            api.log("warning", "Telegram note about an unlinked reply could not be delivered.")


def _make_poller(api):
    return lambda: _poller(api)


def _make_outbound(api, cue: TypingCue | None = None):
    async def handle(event: Dict[str, Any]) -> None:
        try:
            local_settings, event = _load_settings(api), _bridged(event)  # a helper's root address, as it reaches here
            chat_id = _target_chat(local_settings, event)
            if not chat_id:
                return
            lang = telegram_i18n.language()  # a client (and so a bot token) only where a message goes

            # Subagent lifecycle → one dedicated bubble per subagent, edited in
            # place across its lifecycle (not a flood of new messages). Since 6.22
            # the supervisor emits one is_progress chat.outbound per subagent state
            # transition; mirroring them raw spams the chat / collapses over the
            # real reply in silent mode.
            sub_event = str(event.get("subagent_event") or "").strip().lower()
            if sub_event:
                if _subagent_cards_enabled(local_settings):
                    await _render_subagent_card(api, _telegram_client(api, event, chat_id), chat_id, event, sub_event, lang,
                                                quiet=_is_silent_mode_enabled(local_settings))
                return

            # Card-internal host rows: the web shows them inside a task card, and
            # this transport has none. A child's row stays out while its root is
            # unfinished (the lifecycle bubble above keeps saying `failed`, the
            # root accounts for its children). A root's own placed row is an
            # ordinary progress note here and follows the progress toggle below.
            if _child_row_held_for_root(api, event):
                return

            # A child's own final report speaks to its parent: the web keeps it inside the child's
            # card and the root answers here. Its deliberate messages and host notices are typed and go.
            if _child_final_report(event):
                return

            # Progress telemetry and a root's own card rows → dropped by default (a child's row waits above,
            # then goes as before); the typing indicator already signals "working". Opt in via the toggle.
            # A row the host types as a terminal receipt (Wrap up's saved work) is no progress: it goes.
            if (event.get("is_progress") or (event.get("card_row") and event.get("delegation_role") != "subagent"
                                             and event.get("terminal_receipt") != "retained_work")) and not _mirror_progress_enabled(local_settings):
                return

            # A routing receipt is a typed annotation (suppress_bubble, no text): its owner notice is ONE message
            # answering the owner's message, and a landed act stays silent. A refusal's paired System row stays in the
            # web, and is skipped here only when it says this ACK carried its notice.
            if event.get("role") == "system" and event.get("routing_notice_carrier") == "routing_ack":
                return
            if str(event.get("annotation_type") or "") == "routing_ack":
                notice, picker = _routing_notice_text(event, lang)
                if notice:
                    if picker:
                        _clear_silent_msg(api, chat_id)
                    await _telegram_client(api, event, chat_id).send_message(chat_id, notice, parse_mode="")
                return

            # The host's one line (a notice's phrase and step, the technical pause), each part translated first: the
            # frame's own line when the host assembled one, so the phone says what the window and Copy say. The phone
            # has no Details: a notice's typed one-line instruction (a Git command) comes under that line.
            line = event["notice"].get("line") if isinstance(event.get("notice"), dict) else None
            text = line if isinstance(line, str) and line else notice_line(event, lang, drive_root=telegram_i18n.drive_root())
            text = with_instruction(event, text)
            # A relay quotes the owner's own words above Ouroboros's note; they are sent verbatim, as plain text.
            relayed = event.get("system_type") == "relay_entry" and isinstance(event.get("relay"), dict)
            if relayed:
                text = _relay_text(event["relay"], text, lang)
            completion_summary = event.get("role") == "system" and event.get("system_type") == "project_completion_summary"
            if completion_summary:
                notice = event.get("completion_notice")
                if not _notify_enabled(local_settings, "TELEGRAM_NOTIFY_TASKS"):
                    # Preserved intermediate output is content, not the optional status line.
                    # Unrecoverable legacy metadata must not erase unknown content.
                    text = str(notice.get("salvage") or "") if isinstance(notice, dict) else text
                elif isinstance(notice, dict) and isinstance(notice.get("neutral"), bool) and isinstance(notice.get("detail"), str):
                    text = _completion_text(event, str(event.get("task_id") or ""), lang)
            if not text:
                return
            client = _telegram_client(api, event, chat_id)
            # Honor the host's markdown hint: plain text (markdown=False) is sent
            # verbatim so literal *, _, `, [] aren't mis-parsed as Telegram
            # formatting; an absent/True hint renders markdown→HTML as before.
            parse_mode = "" if completion_summary or relayed or event.get("markdown") is False else "HTML"

            # Silent mode mutes each authored message and the final answer, each its own message, and technical
            # progress, the host's typed notes included; only a one-message progress note is tracked, for the next
            # note of its episode to replace. Any other host row, a typed problem among them, keeps its sound and run.
            host_row = event.get("role") == "system" and not _quiet_progress(event)
            quiet = _is_silent_mode_enabled(local_settings) and not host_row
            single = quiet and event.get("is_progress") is True and len(message_chunks(
                text, parse_mode, limit=getattr(client, "text_limit", 4096))) == 1
            tracked_msg_id = _get_silent_msg(api, client.bot_id, chat_id) if single else 0

            # An answer never lands under, or revives, the typing it ends (DESIGN §9).
            async with _cue_message(cue, chat_id) as delivered:
                with contextlib.ExitStack() as fallbacks:
                    # editMessageText returns False on any failure (too old, deleted, parse error): send instead.
                    if tracked_msg_id and _replaceable_progress(api, client.bot_id, chat_id, tracked_msg_id, event):
                        edited = await client.edit_message_text(chat_id, tracked_msg_id, text, parse_mode=parse_mode)
                        if not edited:
                            fallbacks.enter_context(replacing_request())
                        if not edited and parse_mode:
                            edited = await client.edit_message_text(chat_id, tracked_msg_id, text, parse_mode="")
                            if not edited:
                                fallbacks.enter_context(replacing_request())
                        if edited:
                            delivered()
                            return
                    msg_id = await client.send_message(chat_id, text, parse_mode=parse_mode, **({"disable_notification": True} if quiet else {}))
                    if msg_id:
                        delivered()

            if single and msg_id:
                _set_silent_msg(api, client.bot_id, chat_id, msg_id)  # an edit of a split note would erase its last part
            elif not host_row:
                _clear_silent_msg(api, chat_id)
        except Exception as exc:
            _report_failure(api, "outbound", exc)
    return handle


def _make_typing(cue: TypingCue):
    async def handle(_event: Dict[str, Any]) -> None:
        # A start or resume is a wake, never itself a cue: a helper, a wake-up or a
        # mirrored start shows typing only if the host says a reply is prepared (DESIGN §9).
        cue.wake()
    return handle


def _typing_cue(api) -> TypingCue:
    """The bridge's one typing refresher over the host's reply-preparation projection, paced per accepted token."""
    def targets(state: Dict[str, Any]) -> set:  # a proven address types only in its bot and chat (TypingCue._send)
        settings, bot = _load_settings(api), bot_id(binding_for(api).snapshot()[0])
        rows = [row for row in state.get("chats") or [] if isinstance(row, dict)]
        return {
            chat
            for row in rows
            if (transport := row.get("transport") or {}) is not None
            and (chat := _target_chat(settings, {"chat_id": row.get("chat_id"), "transport": transport}))
            and _carried_here(bot, transport, chat)
        }

    return TypingCue(read_state=lambda: host_reply_preparation(_data_dir(api)), targets=targets,
                     client_factory=lambda: _telegram_client(api), warn=lambda text: api.log("warning", text),
                     generation=lambda: binding_for(api).snapshot()[1])


def _cue_message(cue: TypingCue | None, chat_id: int):
    """The cue's bracket around one message to ``chat_id`` (an empty one without a cue)."""
    return cue.message(chat_id) if cue is not None else contextlib.nullcontext(lambda: None)


def _bridged_handler(handler):
    async def handle(event: Dict[str, Any]) -> None:
        await handler(_bridged(event))

    return handle


def _make_photo(api):
    return _bridged_handler(make_photo_handler(
        api,
        load_settings=lambda a: _load_settings(a),
        target_chat=lambda s, e: _target_chat(s, e),
        client_factory=lambda *a, **k: _telegram_client(*a, **k),
    ))


def _make_video(api):
    return _bridged_handler(make_video_handler(
        api,
        load_settings=lambda a: _load_settings(a),
        target_chat=lambda s, e: _target_chat(s, e),
        client_factory=lambda *a, **k: _telegram_client(*a, **k),
    ))


def _make_document(api):
    return _bridged_handler(make_document_handler(
        api,
        load_settings=lambda a: _load_settings(a),
        target_chat=lambda s, e: _target_chat(s, e),
        client_factory=lambda *a, **k: _telegram_client(*a, **k),
        read_status=lambda a: _read_status(a),
    ))


def _make_links(api):
    return _bridged_handler(make_links_handler(
        api,
        load_settings=lambda a: _load_settings(a),
        target_chat=lambda s, e: _target_chat(s, e),
        client_factory=lambda *a, **k: _telegram_client(*a, **k),
    ))


def _make_quiz(api, cue: TypingCue | None = None):
    async def handle(event: Dict[str, Any]) -> None:
        try:
            event = _bridged(event)
            local_settings = _load_settings(api)
            chat_id = _target_chat(local_settings, event)
            if not chat_id:
                return
            client = _telegram_client(api, event, chat_id)
            question = str(event.get("question") or "").strip()
            raw_options = event.get("options") if isinstance(event.get("options"), list) else []
            # Shared quiz contract cap: ouroboros.tools.core._MAX_QUIZ_OPTIONS.
            # A secret request is a note on where to enter it: no options, no reply prompt.
            secret = isinstance(event.get("secret_request"), dict)
            plain_labels, details, recommended_index = telegram_quiz.card_options([] if secret else raw_options, limit=6)
            # Buttons and the remembered record keep the starred caption; an open
            # question (no options) is a card without buttons (TZ-2 B1).
            labels = telegram_quiz.button_labels(plain_labels, recommended_index)
            if not question:
                return
            task_id = str(event.get("task_id") or "").strip()
            quiz_id = str(event.get("quiz_id") or "").strip()
            if not task_id or not quiz_id:
                return  # the host refuses anonymous quizzes; nothing to answer to
            _clear_silent_msg(api, chat_id)
            stake = str(event.get("stake") or "").strip()
            assumption = str(event.get("assumption") or "").strip()
            lang = _poller_preferences(api)[4]
            wait_for_answer = event.get("wait_for_answer") is True and not secret
            # Both optional: events from an older host carry neither.
            project_name = str(event.get("project_name") or "").strip()
            host_facts = str(event.get("host_facts") or "").strip()
            # Sends use the event client's fixed origin, so EVERY explanation part gets it.
            # Edits use the remembered text on an unlabelled client (including the poller's).
            heading = origin_label(event, lang)
            # The label already names the source: a legacy Project line would repeat it. A
            # minted ``proj_<hex>`` is no name at all. An empty label (a failed host lookup
            # among them) keeps the named Project line.
            render_project_name = "" if heading or not project_display_name(project_name, project_name) else project_name
            card = {
                "project_name": render_project_name, "option_details": details,
                "recommended_index": recommended_index, "host_facts": host_facts, "lang": lang,
            }
            body = telegram_quiz.render_quiz_text(
                question, plain_labels, stake, assumption, wait_for_answer=wait_for_answer, **card)
            compact = telegram_quiz.render_compact_text(
                plain_labels, project_name=render_project_name, recommended_index=recommended_index, lang=lang)
            prefix = f"{heading}\n" if heading else ""  # TelegramClient.origin_prefix's plain form
            token = telegram_quiz.mint_token(task_id, quiz_id)
            # One button per option; a reply to the card is a free-form answer.
            # Both reach the host's decision ingress (#472). An overflowing card
            # arrives as plain parts followed by the compact keyboard message.
            # Creation holds the card's lock so a lifecycle fact cannot edit (or
            # miss) a card mid-send.
            sent_ids: list[int] = []
            api_base = str(getattr(client, "api_base_origin", "") or "")
            track_secret = secret and not bool(getattr(OUTBOUND_JOURNAL.get(), "probing", False))
            secret_op = begin_secret_send(api, client.bot_id, chat_id, api_base=api_base) if track_secret else ""
            secret_confirmed = False
            try:
                async with telegram_quiz.card_lock(token), _cue_message(cue, chat_id) as delivered:
                    message_id, overflowed = await telegram_quiz.send_quiz_card(
                        client, chat_id, body=body, compact=compact,
                        hint_text=(telegram_quiz.hint_secret(lang) if secret else telegram_quiz.hint(lang) if labels
                                   else telegram_quiz.hint_open(lang)),
                        keyboard=telegram_quiz.quiz_keyboard(token, labels),
                        reserve=telegram_quiz.answered_edit_reserve(lang) + _u16len(prefix),
                        **({"sent_ids": sent_ids, "on_part": lambda mid: telegram_quiz.remember_secret_message(
                            api, client.bot_id, chat_id, mid, task_id, quiz_id)} if track_secret else {}),
                    )
                    delivered()
                    if track_secret:
                        all_secret_ids = [int(x or 0) for x in [*sent_ids, message_id]]
                        secret_confirmed = bool(int(message_id or 0) > 0 and all(x > 0 for x in all_secret_ids))
                        finish_secret_send(
                            api, secret_op, bot_id=client.bot_id, chat_id=chat_id, api_base=api_base,
                            confirmed_ids=all_secret_ids, uncertain=not secret_confirmed,
                            reason="unknown_message_id" if not secret_confirmed else "", task_id=task_id, quiz_id=quiz_id,
                        )
                    settled_text = (compact if overflowed else
                                    telegram_quiz.render_quiz_text(question, plain_labels, stake, "", **card)
                                    if wait_for_answer else body)
                    telegram_quiz.remember_quiz(api, client.bot_id, token, {
                        "task_id": task_id, "quiz_id": quiz_id, "chat_id": chat_id,
                        "message_id": int(message_id or 0), "options": labels,
                        "text": prefix + settled_text,
                        **({"secret": True, "message_ids": sent_ids} if secret else {}),
                    })
                    await telegram_quiz.apply_retained_fact(
                        api, token, lang, client_factory=lambda: _telegram_client(api), published=event)
            except BaseException as exc:
                if track_secret and not secret_confirmed:
                    unsent_exc = (bool(getattr(exc, "unsent", False))
                                  or isinstance(exc, (OutboxDeferred, OutboxStopped))
                                  or type(exc).__name__ == "_WouldSend")
                    uncertain = not (unsent_exc and all(int(x or 0) > 0 for x in sent_ids))
                    finish_secret_send(
                        api, secret_op, bot_id=client.bot_id, chat_id=chat_id, api_base=api_base,
                        confirmed_ids=sent_ids, uncertain=uncertain, reason=type(exc).__name__, task_id=task_id, quiz_id=quiz_id,
                    )
                raise
        except Exception as exc:
            _report_failure(api, "quiz", exc)
    return handle


def _make_quiz_state(api):
    """Telegram has no reload: a sent card follows its question's lifecycle (TZ-2 B2),
    one fact at a time per card and never before the card itself is remembered."""
    async def handle(event: Dict[str, Any]) -> None:
        try:
            await telegram_quiz.follow_lifecycle(
                api, event, _poller_preferences(api)[4],
                client_factory=lambda: _telegram_client(api))
        except Exception as exc:
            _report_failure(api, "quiz state", exc)
    return handle


def _make_presence_status(api):
    return make_presence_status(api, runtime_getter=_presence_runtime)


def _make_owner_pair(api):
    async def _pair(request):
        if not request_may_change_owner(request):
            return JSONResponse(
                {"ok": False, "error": "Owner pairing codes can only be issued from local Ouroboros on loopback."},
                status_code=403,
                headers={"Cache-Control": "no-store"},
            )
        issued = await asyncio.to_thread(issue_owner_pairing_code, api)
        return JSONResponse(
            {"ok": True, **issued, "command": f"/pair {issued['code']}"},
            headers={"Cache-Control": "no-store"},
        )
    return _pair


def register(api):
    telegram_i18n.configure(_data_dir(api))
    binding = binding_for(api)
    binding.read_now(api)  # the token in force from the first send; the observer keeps it current
    presence = _presence_runtime(api)
    if callable(getattr(api, "on_unload", None)):
        def _unload() -> None:
            binding.close()
            sync_task = _CMD_SYNC_TASKS.pop(os.path.realpath(api.get_state_dir()), None)
            if sync_task is not None and not sync_task.done(): sync_task.cancel()
            presence.stop(wait_sec=2.0)

        api.on_unload(_unload)  # unload, Off or a revoked grant: no client or worker of this load sends again
    cue = _typing_cue(api)  # Telegram failures stay inside it, so they never spend restart budget
    api.register_supervised_task("poller", _make_poller(api), restart_policy="on_failure", max_restarts=10)
    api.register_supervised_task("presence", presence.run, restart_policy="on_failure", max_restarts=10)
    api.register_supervised_task("notifier", _make_notifier(api, trust_env=_HONOR_ENV_PROXIES), restart_policy="on_failure", max_restarts=10)
    api.register_supervised_task("typing_cue", cue.run, restart_policy="on_failure", max_restarts=10)
    # Every mirrored message goes through one durable, ordered journal (lib/telegram_outbox.py).
    outbox = outbox_for(api)
    api.register_supervised_task("outbox", outbox.run, restart_policy="on_failure", max_restarts=10)
    register_presence_tools(api, presence)

    def routed(event):  # each content handler's first filter: no target chat, nothing to save
        return bool(_target_chat(_load_settings(api), _bridged(event)))
    api.subscribe_event("chat.outbound", outbox.subscriber("outbound", _make_outbound(api, cue), decide=True, routed=routed))
    api.subscribe_event("chat.typing", _make_typing(cue))
    api.subscribe_event("chat.photo", outbox.subscriber("photo", _make_photo(api), routed=routed))
    api.subscribe_event("chat.video", outbox.subscriber("video", _make_video(api), routed=routed))
    api.subscribe_event("chat.document", outbox.subscriber("document", _make_document(api), routed=routed))
    api.subscribe_event("chat.links", outbox.subscriber("links", _make_links(api), routed=routed))
    api.subscribe_event("chat.quiz", outbox.subscriber("quiz", _make_quiz(api, cue), routed=routed))
    api.subscribe_event("chat.quiz_state", outbox.subscriber(  # a card's edit: only once a card is remembered
        "quiz_state", _make_quiz_state(api), routed=lambda event: telegram_quiz.card_follows(api, event)))
    # GET status feeds Circles place discovery and connector health; settings/save hydrates/saves Telegram settings.
    api.register_route("status", handler=_make_presence_status(api), methods=("GET",))
    api.register_route("settings/save", handler=_make_settings_save(api), methods=("GET", "POST"))
    api.register_route("owner/pair", handler=_make_owner_pair(api), methods=("POST",))
    api.register_route("miniapp/status", handler=_make_status(api), methods=("POST",))
    # Share… in the Mini App: the bridge's own client prepares the owner's message for Telegram's chat picker.
    api.register_route("miniapp/prepare-share", handler=make_prepare_share(api, _bridge_client), methods=("POST",))
    register_connection(api)  # Connection's status and its kept-copies form
    api.register_settings_section("telegram", title="Telegram", schema=settings_section_schema())
    # Unsupported OS/architecture degrades inside register_miniapp(). Invalid
    # host runtime or unsafe/unwritable state must fail the complete load rather
    # than leave a partially registered skill claiming to be healthy.
    companion = "miniapp_gateway" if register_miniapp(api) else ""
    # One observer applies a saved, replaced or cleared token to every part, the Mini App included,
    # independent of the poller's long poll or a parked rejection.
    def _on_token_change() -> None:
        presence.on_token_changed()
        cue.wake()

    api.register_supervised_task("token_watch", make_token_observer(api, companion, _on_token_change),
                                 restart_policy="on_failure", max_restarts=10)
