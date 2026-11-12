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

from ouroboros.net_transport import env_proxies_configured
from ouroboros.utils import in_worker_process

from .lib.telegram_api import (
    TELEGRAM_RETRY_INITIAL_SEC,
    TelegramBindingChanged,
    TelegramClient,
    TelegramRequestRejected,
    TelegramTokenMissing,
    TelegramTransportError,
    is_transient_telegram_error,
    markdown_to_telegram_html,
    message_chunks,
    next_telegram_retry_delay,
    _LOCALIZED_TEXTS,
    _MAX_TELEGRAM_UPLOAD_BYTES,
)
from .lib.telegram_state import (
    _state_file, _load_settings, _is_silent_mode_enabled,
    _get_silent_msg, _set_silent_msg, _clear_silent_msg, _subagent_cards_enabled,
    _mirror_progress_enabled, _render_subagent_card, _data_dir,
    _load_runtime_state, _child_row_held_for_root, _child_final_report,
    _mirror_options, _old_reference_unlinked, _owner_message_seen, _owner_ingress_fields, _replaceable_progress,
    _quiet_progress,
)
from .lib import telegram_i18n, telegram_inbound, telegram_quiz
from .lib.telegram_health import _collect_health, _build_menu_tasks
from .lib.telegram_notifier import _make_notifier, _notify_enabled, _completion_text, _relay_text, _routing_notice_text
from .lib.telegram_share import make_prepare_share
from .lib.telegram_typing import TypingCue, host_reply_preparation
from .lib.telegram_poll_binding import (
    PollCursor, bot_id, park_until_settings_change, saved_server as _saved_server, server_fingerprint, update_key,
)
from .lib.miniapp_registration import _read_status, register as register_miniapp
from .lib.telegram_token import _report_failure, _wait_for_bot_token, binding_for, make_token_observer
from .lib.telegram_connection import (  # status of the token, the bridge and the Mini App
    _COMMAND_MODE_FULL, _COMMAND_MODE_STRICT, _VALID_COMMAND_MODES, _fail_permanent_poll_rejection,
    _make_connection, _make_status, _save_bridge_status, connection_components, poll_failure_reason,
)
from .lib.telegram_connection import _bridge_status, _connection_fields  # noqa: F401 — the plugin's status facade
from .scripts.telegram_settings import (
    TelegramSettingsError,
    merge_settings,
    make_settings_save as _make_settings_save,
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
    api_base_url, files_dir = server or _saved_server(api)
    client = TelegramClient(token, trust_env=_HONOR_ENV_PROXIES,
                            proxy=telegram_proxy(pathlib.Path(api.get_state_dir())),
                            **(_mirror_options(api, bot_id(token), event, chat_id) if event is not None else {}),
                            api_base_url=api_base_url, files_dir=files_dir, binding=binding, generation=generation)
    return client, generation


def _telegram_client(api, event=None, chat_id: int = 0, server: tuple[str, str | None] | None = None) -> TelegramClient:
    return _bridge_client(api, event, chat_id, server)[0]


_POLL_GENERATION_STARTED = time.time()

_SLASH_COMMAND_RE = re.compile(r"^\s*/[A-Za-z]")

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


def _build_menu_keyboard(command_mode: str, lang: str = "en") -> tuple[str, list[list[dict]]]:
    """Return (header_text, inline_keyboard_rows) for the /menu command."""
    t = _LOCALIZED_TEXTS[lang]
    if command_mode == _COMMAND_MODE_STRICT:
        return (
            t["menu_title_strict"],
            [[{"text": t["btn_settings"], "callback_data": "nav:settings"}]],
        )

    header = t.format("menu_title", command_mode=command_mode, language=telegram_i18n.label(lang))
    keyboard = [
        [
            {"text": t["btn_metrics"], "callback_data": "nav:status"},
            {"text": t["btn_mind"], "callback_data": "nav:mind"},
        ],
        [
            {"text": t["btn_tasks"], "callback_data": "nav:tasks"},
        ],
        [
            {"text": t["btn_settings"], "callback_data": "nav:settings"},
        ]
    ]
    return header, keyboard


def _build_menu_status(command_mode: str, lang: str = "en", info_text: str = "") -> tuple[str, list[list[dict]]]:
    """Return status header and keyboard with Refresh and Back button."""
    t = _LOCALIZED_TEXTS[lang]
    header = t.format("metrics_title", info_text=info_text)
    keyboard = [
        [{"text": t["btn_refresh"], "callback_data": "cmd_act:update_status"}],
        [{"text": t["btn_back"], "callback_data": "nav:menu"}]
    ]
    return header, keyboard


def _build_menu_mind(command_mode: str, lang: str = "en", bg_enabled: bool = False) -> tuple[str, list[list[dict]]]:
    """Return mind controlling header and buttons."""
    t = _LOCALIZED_TEXTS[lang]
    state_str = t["mind_state_active"] if bg_enabled else t["mind_state_sleeping"]
    header = t.format("mind_title", state_str=state_str)

    row = []
    if command_mode == _COMMAND_MODE_FULL:
        if bg_enabled:
            row.append({"text": t["btn_stop_bg"], "callback_data": "cmd_act:bg_stop"})
        else:
            row.append({"text": t["btn_start_bg"], "callback_data": "cmd_act:bg_start"})

    keyboard = []
    if row:
        keyboard.append(row)
    keyboard.append([{"text": t["btn_back"], "callback_data": "nav:menu"}])
    return header, keyboard


def _bot_commands(command_mode: str, lang: str = "") -> list:
    """Localized '/' menu: owner Panic in every mode; other controls in full_access."""
    t = _LOCALIZED_TEXTS[lang]
    cmds = [
        {"command": "menu", "description": t["cmd_menu"]},
        {"command": "language", "description": t["cmd_language"]},
        {"command": "status", "description": t["cmd_status"]},
        {"command": "help", "description": t["cmd_help"]},
        {"command": "panic", "description": t["cmd_panic"]},
    ]
    if command_mode == _COMMAND_MODE_FULL:
        cmds += [
            {"command": "evolve", "description": t["cmd_evolve"]},
            {"command": "bg", "description": t["cmd_bg"]},
            {"command": "review", "description": t["cmd_review"]},
            {"command": "restart", "description": t["cmd_restart"]},
        ]
    return cmds


def _build_menu_settings(api, command_mode: str, lang: str = "en") -> tuple[str, list[list[dict]]]:
    """Return (header_text, inline_keyboard_rows) for the Settings panel."""
    t = _LOCALIZED_TEXTS[lang]
    header = t["settings_title"]
    silent_on = _is_silent_mode_enabled(_load_settings(api))
    silent_label = t["btn_silent_on"] if silent_on else t["btn_silent_off"]
    keyboard = [
        [
            {"text": t["btn_language"], "callback_data": "nav:language"},
        ],
        [
            {"text": silent_label, "callback_data": "cmd_act:toggle_silent"},
        ],
        [{"text": t["btn_back"], "callback_data": "nav:menu"}]
    ]
    return header, keyboard


def _build_language_keyboard(lang: str = "") -> tuple[str, list[list[dict]]]:
    """(header_text, inline_keyboard_rows) for the install's interface language: English,
    every language with a translation memory on disk (the current one marked), and the
    hint that any other language is one ``/language <name>`` away."""
    t = _LOCALIZED_TEXTS[lang]
    header = t.format("lang_title", language=telegram_i18n.label(lang))
    rows = [[{"text": t["lang_english"], "callback_data": "set_lang:en"}]]
    for item in telegram_i18n.known_languages():
        tag = str(item.get("language") or "")
        if not tag or telegram_i18n.english(tag):
            continue
        mark = "• " if tag == lang else ""
        rows.append([{"text": f"{mark}{item.get('label') or tag}", "callback_data": f"set_lang:{tag}"}])
    rows.append([{"text": t["btn_back"], "callback_data": "nav:menu"}])
    return header, rows


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


def _target_chat(settings: Dict[str, Any], event: Dict[str, Any]) -> int:
    mirror_mode = str(settings.get("TELEGRAM_MIRROR_MODE") or "all").strip().lower()
    configured = str(settings.get("TELEGRAM_CHAT_ID") or "").strip()
    if configured:
        try:
            chat_id = int(configured)
        except ValueError:
            return 0
        if mirror_mode == "all":
            # Mirror everything (web UI + Telegram) to the pinned chat
            return chat_id
        # telegram_only: only forward events that originate from Telegram transport
        transport = event.get("transport") if isinstance(event.get("transport"), dict) else {}
        if transport.get("kind") == "telegram":
            return chat_id
        # ...and the owner's notes, skill notices and the Safety Supervisor's notices: System rows
        # written for the owner, which an owner who left the web should still get on the phone.
        if event.get("role") == "system" and event.get("system_type") in {"reminder", "skill_notice", "safety_notice"}:
            return chat_id
        return 0
    # No pinned chat configured — only forward events that originate from
    # a Telegram transport conversation so local UI events are never leaked.
    transport = event.get("transport") if isinstance(event.get("transport"), dict) else {}
    if transport.get("kind") != "telegram":
        return 0
    try:
        return int(transport.get("conversation_id") or 0)
    except (TypeError, ValueError):
        return 0


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


async def _inject(api, payload: Dict[str, Any]) -> None:
    settings = _load_settings(api)
    pinned_chat = str(settings.get("TELEGRAM_CHAT_ID") or "").strip()
    if not pinned_chat:
        api.log("warning", "Host inject refused: TELEGRAM_CHAT_ID is not configured or bound.")
        return
    status, _body = await _host_post(api, "/chat/inject", payload)
    if status >= 400:
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
    status_str += "\n" + await asyncio.to_thread(_collect_health, api, lang)
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


async def _authorized_owner(api, client, update, pinned_chat: str, lang: str) -> str | None:
    callback = update.get("callback_query") or {}
    message = update.get("message") or {}
    chat = ((callback.get("message") or {}).get("chat") or (message.get("chat") or {}))
    sender = callback.get("from") or message.get("from") or {}
    chat_id = int(chat.get("id") or 0)
    user_id = int(sender.get("id") or 0)
    if chat_id <= 0 or user_id <= 0:
        return None
    if not pinned_chat:
        if str(chat.get("type") or "") != "private" or chat_id != user_id:
            return None
        merge_settings(pathlib.Path(api.get_state_dir()), {"TELEGRAM_CHAT_ID": str(chat_id)})
        pinned_chat = str(chat_id)
    if str(chat_id) == pinned_chat and str(user_id) == pinned_chat:
        return pinned_chat
    if callback:
        try:
            await client.answer_callback_query(
                str(callback.get("id") or ""),
                text=_LOCALIZED_TEXTS[lang]["not_authorized"],
            )
        except Exception:
            pass
    return None


async def _edit_panel(api, client, chat_id: int, message_id: int, text: str, keyboard: list[list[dict]]) -> None:
    if await client.edit_message_text_with_inline_keyboard(chat_id, message_id, text, keyboard):
        return
    api.log("warning", "Telegram panel edit failed; sending a fresh panel.")
    await client.send_message_with_inline_keyboard(chat_id, text, keyboard)


async def _validate_bot(api, client, command_mode: str, lang: str, generation: int | None = None) -> None:
    """Validate the token (getMe) and finish bridge startup bookkeeping; ready names the exact binding proved."""
    await client.call("getMe")
    # Menu discovery follows command mode; owner Panic has its own immediate door.
    try:
        await client.call("setMyCommands", data={"commands": json.dumps(_bot_commands(command_mode, lang))})
        api.log("info", "Telegram bot commands configured successfully")
    except TelegramBindingChanged:
        raise  # a stale client is never reported ready
    except Exception as exc:
        api.log("warning", f"Failed to set Telegram bot commands: {exc}")
    if generation is not None:
        binding = binding_for(api)
        if not binding.current(generation):
            raise TelegramBindingChanged(getattr(client, "accepted", 0))
        binding.mark_validated(generation)
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
        except TelegramRequestRejected:
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
                    _save_bridge_status(api, "connecting", "telegram_token_changed", bot=client.bot_id, generation=generation)
                    api.log("info", "Telegram bot token changed; polling follows the saved token.")
                    degraded = ""
                else:
                    api.log("info", "Telegram Bot API settings changed; polling follows the saved server.")
            if not validated:
                revalidating = True
                await _validate_bot(api, client, command_mode, lang, generation)  # it records ready
                revalidating, validated, degraded = False, True, ""
            offset, handled = cursor.offset, []
            updates = await client.get_updates(offset)
            if not binding.current(generation):
                # Answered for the previous token: neither confirmed nor handled, so the same bot's
                # rotated token receives these updates again and another bot never sees them.
                continue
            if degraded_cause:
                api.log("info", f"Telegram poller recovered after {degraded_cause}; polling resumed.")
                degraded_cause = ""
            if degraded:
                _save_bridge_status(api, "ready", bot=client.bot_id, generation=generation)
                degraded = ""
            retry_delay = TELEGRAM_RETRY_INITIAL_SEC
            if updates:
                local_settings, pinned_chat, max_updates, command_mode, lang = _poller_preferences(api)

            for update in updates[:max_updates]:
                if not binding.current(generation):
                    break  # the rest stays unconfirmed for the new token's client
                authorized_chat, confirmed = None, (offset, len(handled))
                try:  # Isolate one bad update without changing offset/retry semantics.
                    update_id = int(update.get("update_id") or 0)
                    if update_id > 0:
                        # Confirm what the server returned, even below the offset: a server
                        # that lost its queue answers from its oldest update, again each poll.
                        offset = update_id + 1
                    handled.append(update_key(update))
                    if cursor.is_replay(update):
                        continue

                    authorized_chat = await _authorized_owner(api, client, update, pinned_chat, lang)
                    if not authorized_chat:
                        continue
                    pinned_chat = authorized_chat

                    # --- Handle callback queries (inline button presses) ---
                    callback_query = update.get("callback_query")
                    if callback_query:
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
                            continue

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
                                header, keyboard = await asyncio.to_thread(
                                    _build_menu_tasks, api, command_mode, lang
                                )
                                await _edit_panel(api, client, cb_chat_id, cb_message_id, header, keyboard)
                            elif target == "settings":
                                header, keyboard = _build_menu_settings(api, command_mode, lang)
                                await _edit_panel(api, client, cb_chat_id, cb_message_id, header, keyboard)
                            continue

                        # --- Command Actions / Control (Category 2) ---
                        if cb_data.startswith("cmd_act:"):
                            action = cb_data.split(":", 1)[1]

                            if action == "update_status":
                                await client.answer_callback_query(cb_id, text=_LOCALIZED_TEXTS[lang]["updating_status"])
                                info_text = await _compile_status_text(api, lang)
                                header, keyboard = _build_menu_status(command_mode, lang, info_text)
                                await _edit_panel(api, client, cb_chat_id, cb_message_id, header, keyboard)
                                continue

                            elif action == "toggle_silent":
                                # Toggle TELEGRAM_SILENT_MODE and refresh the Settings panel.
                                # This is a display preference (no LLM injection), so it is
                                # allowed in every command_mode including strict.
                                local_settings = _load_settings(api)
                                currently_on = _is_silent_mode_enabled(local_settings)
                                new_value = "off" if currently_on else "on"
                                merge_settings(pathlib.Path(api.get_state_dir()), {"TELEGRAM_SILENT_MODE": new_value})
                                confirmed = None  # the toggle is applied: a later token change must not replay it
                                # Clear any stale tracked message id for this chat so the
                                # next outbound starts a fresh bubble in either direction.
                                _clear_silent_msg(api, cb_chat_id)
                                toast_key = "quiet_toggled_on" if new_value == "on" else "quiet_toggled_off"
                                await client.answer_callback_query(cb_id, text=_LOCALIZED_TEXTS[lang][toast_key])
                                header, keyboard = _build_menu_settings(api, command_mode, lang)
                                await _edit_panel(api, client, cb_chat_id, cb_message_id, header, keyboard)
                                continue

                            elif action == "bg_thoughts":
                                # The retired thoughts button of a panel sent earlier: settle its spinner
                                # and show the current panel, reading nothing.
                                await client.answer_callback_query(cb_id)
                                bg_enabled = _is_bg_consciousness_active(api)
                                header, keyboard = _build_menu_mind(command_mode, lang, bg_enabled)
                                await _edit_panel(api, client, cb_chat_id, cb_message_id, header, keyboard)
                                continue

                            elif action in ("bg_start", "bg_stop"):
                                if command_mode != _COMMAND_MODE_FULL:
                                    await client.answer_callback_query(cb_id, text=_LOCALIZED_TEXTS[lang]["restricted_safe"])
                                    continue
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
                                confirmed = None  # the host took the command: it is not injected again
                                # Give it a tiny moment to commit setting then refresh mind panel
                                await asyncio.sleep(0.8)
                                bg_enabled = _is_bg_consciousness_active(api)
                                header, keyboard = _build_menu_mind(command_mode, lang, bg_enabled)
                                await _edit_panel(api, client, cb_chat_id, cb_message_id, header, keyboard)
                                continue

                        # --- Handle language selection buttons ---
                        if cb_data.startswith("set_lang:"):
                            requested = cb_data.split(":", 1)[1]

                            async def _toast(text: str, _cb_id: str = cb_id) -> None:
                                await client.answer_callback_query(_cb_id, text=text)

                            new_lang = await _choose_language(api, requested, lang, _toast)
                            confirmed = None  # the host settled the choice: it is not asked again
                            if new_lang or requested == "en":
                                lang = new_lang
                            # Smoothly return to menu panel in the language now in effect
                            header, keyboard = _build_menu_keyboard(command_mode, lang)
                            await _edit_panel(api, client, cb_chat_id, cb_message_id, header, keyboard)
                            continue

                        # --- Quiz answers (#472): a tapped option reaches the host's
                        # decision ingress; the owner is already verified above.
                        if cb_data.startswith("qz:"):
                            await telegram_quiz.answer_from_callback(
                                api, client, cb_data, cb_id=cb_id,
                                request_key=update_key(update), lang=lang, post=_host_post,
                            )
                            continue

                        if cb_data.startswith(("set_model:", "set_budget:")):
                            await client.answer_callback_query(
                                cb_id,
                                text=_LOCALIZED_TEXTS[lang]["use_miniapp"],
                            )
                            continue

                        await client.answer_callback_query(cb_id, text=_LOCALIZED_TEXTS[lang]["unknown_command"])
                        continue

                    # --- Handle regular messages ---
                    await _handle_owner_message(api, client, update, update_id, command_mode, lang)
                except TelegramBindingChanged:
                    # Nothing more goes out through this token. An update stopped before its effect was
                    # accepted (a download, a reply before the host took it) stays unconfirmed, so the same
                    # bot's next client is given it again; the host accepts an owner message id and a quiz
                    # request once. One whose local or host effect was applied here (``confirmed`` cleared)
                    # stays confirmed: replaying it would apply that effect twice (a toggle would undo itself).
                    if confirmed is not None:
                        offset, handled = confirmed[0], handled[:confirmed[1]]
                        api.log("info", "Telegram bot token changed while an update was handled; it stays unconfirmed.")
                    else:
                        api.log("info", "Telegram bot token changed after an update was applied; its reply was not sent.")
                    break
                except Exception as exc:
                    api.log(
                        "warning",
                        f"Telegram update processing failed ({type(exc).__name__}).",
                    )
                    if authorized_chat:
                        try:
                            notice = _LOCALIZED_TEXTS[lang]["update_failed"]
                            await client.send_message(int(authorized_chat), notice)
                        except Exception:
                            api.log("warning", "Telegram update failure notice could not be delivered.")
            if updates and (handled or offset != cursor.offset):
                cursor.save(offset, handled)
            await asyncio.sleep(0.1)
        except (TelegramBindingChanged, TelegramTokenMissing):
            continue  # the next cycle builds the client of the accepted token, or waits for one
        except (TelegramRequestRejected, TelegramTransportError) as exc:
            # Only getUpdates and getMe reach here: a 409 is another poller of this token, which may stop.
            if isinstance(exc, TelegramRequestRejected) and not exc.transient and exc.status_code != 409:
                _fail_permanent_poll_rejection(api, exc, revalidating, client.bot_id, generation)
                await park_until_settings_change(api, (client.token, server), binding)
                server, validated, degraded = None, False, ""
                continue
            # Transition logging only: one line entering degraded, one on
            # recovery; consecutive transient failures stay quiet while the
            # monotone backoff (reset by any successful poll) paces retries.
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


async def _handle_owner_message(
    api, client, update: Dict[str, Any], update_id: int, command_mode: str, lang: str,
) -> None:
    """One authorized owner message: local /menu /language /help, command-mode
    translation, a reply to a quiz card (#472), then text/photo/file relay (#668).
    Owner binding is already enforced by the caller (TOFU for every command mode).
    """
    message = update.get("message") or {}
    chat = message.get("chat") or {}
    sender = message.get("from") or {}
    chat_id = int(chat.get("id") or 0)
    _owner_message_seen(api, client.bot_id, chat_id, int(message.get("message_id") or 0))
    # Owner binding + filtering is already enforced at the top of
    # the update loop (TOFU for all command modes), so chat_id is
    # guaranteed to equal the pinned owner chat here.
    text = str(message.get("text") or message.get("caption") or "").strip()
    caption = str(message.get("caption") or "").strip()
    # Handle /menu command locally — always allowed
    cleaned_text = text.lower().strip()
    is_menu_cmd = _is_exact_bot_command(cleaned_text, "/menu")
    if is_menu_cmd:
        header, keyboard = _build_menu_keyboard(command_mode, lang)
        if keyboard:
            await client.send_message_with_inline_keyboard(chat_id, header, keyboard)
        else:
            await client.send_message(chat_id, header)
        return

    # Handle /language locally — always allowed. Bare: the keyboard. With an argument
    # ("/language Quenya", "/language pt-BR", "/language invent a language and translate
    # everything into it"): the owner's words go to the host's one language writer.
    is_lang_cmd = _is_exact_bot_command(cleaned_text, "/language")
    if is_lang_cmd:
        header, keyboard = _build_language_keyboard(lang)
        await client.send_message_with_inline_keyboard(chat_id, header, keyboard)
        return
    # Handle /help command locally — always allowed
    is_help_cmd = _is_exact_bot_command(cleaned_text, "/help")
    if is_help_cmd:
        help_text = _LOCALIZED_TEXTS[lang]["help_text"]
        await client.send_message(chat_id, help_text)
        return

    if text.lower().startswith("/panic"):
        # Compare Telegram's whole-second date with the host clock at import.
        # This assumes aligned clocks: a host ahead rejects fresh commands until
        # Telegram reaches that second; a host behind may admit retained ones.
        # The boundary second is indistinguishable from fresh input; accept it.
        if int(message.get("date") or 0) < int(_POLL_GENERATION_STARTED):
            await client.send_message(chat_id, _LOCALIZED_TEXTS[lang]["panic_earlier"])
            return
        try:
            sender_label = f"Telegram ({_extract_sender_label(sender, chat_id)})"
            await asyncio.wait_for(_inject(api, {**_owner_ingress_fields(api, client.bot_id, chat_id, message, update_id),
                "text": "/panic", "chat_id": chat_id,
                "sender_label": sender_label, "transport": {"kind": "telegram", "conversation_id": str(chat_id), "sender_label": sender_label},
                "user_id": int(sender.get("id") or chat_id), "source": "telegram"}), timeout=2)
        except Exception:
            api.log("warning", "Immediate Telegram Panic was not confirmed; it will not be retried.")
            await client.send_message(chat_id, _LOCALIZED_TEXTS[lang]["panic_unconfirmed"])
        return

    # Local commands are done. Look up the replied-to card once, before any payload forwarding;
    # a secret reply stays local even when it has no text or resembles a slash command.
    reply_to = message.get("reply_to_message") or {}
    quiz_ref = None
    if isinstance(reply_to, dict) and reply_to:
        quiz_ref = telegram_quiz.quiz_for_message(api, client.bot_id, chat_id, int(reply_to.get("message_id") or 0))
    if quiz_ref and quiz_ref.get("secret"):
        await telegram_quiz.answer_from_reply(api, client, quiz_ref, "", chat_id=chat_id,
                                              request_key=update_key(update), lang=lang, post=_host_post)
        return

    if cleaned_text.startswith("/language ") or cleaned_text.startswith("/language@"):
        requested = text.split(None, 1)[1].strip() if len(text.split(None, 1)) > 1 else ""
        if requested:
            async def _reply(message: str) -> None:
                await client.send_message(chat_id, message)

            await _choose_language(api, requested, lang, _reply)
            return

    # Translate commands to safe natural-language text.
    # _translate_command returns None when the command is rejected.
    safe_text = _translate_command(text, command_mode)
    safe_caption = _translate_command(caption, command_mode) if caption else caption
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

    if quiz_ref is not None and quiz_ref.get("ambiguous") and not _SLASH_COMMAND_RE.match(text):
        # An old card's number names no question for sure: an answer or a turn meant for it stays
        # here, before any download; a command keeps its dispatch, as below.
        await telegram_quiz.answer_from_reply(api, client, quiz_ref, "", chat_id=chat_id,
                                              request_key=update_key(update), lang=lang, post=_host_post)
        return
    if safe_text and not _SLASH_COMMAND_RE.match(text) and isinstance(reply_to, dict) and reply_to:
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
            **_owner_ingress_fields(api, client.bot_id, chat_id, message, update_id),
            "text": safe_text,
            "chat_id": chat_id,
            "user_id": int(sender.get("id") or chat_id or 1),
            "source": "telegram",
            "sender_label": sender_label,
            "transport": {
                "kind": "telegram",
                "conversation_id": str(chat_id),
                "sender_label": sender_label,
            },
            "image_base64": image_base64,
            "image_mime": image_mime,
            "image_caption": safe_caption,
            **({"attachments": [parked.spec]} if parked is not None else {}),
        })
    finally:
        if parked is not None:
            parked.cleanup()
    if _old_reference_unlinked(api, client.bot_id, chat_id, message):
        try:
            await client.send_message(chat_id, _LOCALIZED_TEXTS[lang]["reply_not_linked"])
        except Exception:
            api.log("warning", "Telegram note about an unlinked reply could not be delivered.")


def _make_poller(api):
    return lambda: _poller(api)


def _make_outbound(api, cue: TypingCue | None = None):
    async def handle(event: Dict[str, Any]) -> None:
        try:
            local_settings = _load_settings(api)
            chat_id = _target_chat(local_settings, event)
            if not chat_id:
                return
            client = _telegram_client(api, event, chat_id)
            lang = telegram_i18n.language()

            # Subagent lifecycle → one dedicated bubble per subagent, edited in
            # place across its lifecycle (not a flood of new messages). Since 6.22
            # the supervisor emits one is_progress chat.outbound per subagent state
            # transition; mirroring them raw spams the chat / collapses over the
            # real reply in silent mode.
            sub_event = str(event.get("subagent_event") or "").strip().lower()
            if sub_event:
                if _subagent_cards_enabled(local_settings):
                    await _render_subagent_card(api, client, chat_id, event, sub_event, lang, quiet=_is_silent_mode_enabled(local_settings))
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
                    await client.send_message(chat_id, notice, parse_mode="")
                return

            text = str(event.get("text") or "").strip()
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
            # Honor the host's markdown hint: plain text (markdown=False) is sent
            # verbatim so literal *, _, `, [] aren't mis-parsed as Telegram
            # formatting; an absent/True hint renders markdown→HTML as before.
            parse_mode = "" if completion_summary or relayed or event.get("markdown") is False else "HTML"

            # Silent mode mutes each authored message and the final answer, each its own message, and technical
            # progress, the host's typed notes included; only a one-message progress note is tracked, for the next
            # note of its episode to replace. Any other host row, a typed problem among them, keeps its sound and run.
            host_row = event.get("role") == "system" and not _quiet_progress(event)
            quiet = _is_silent_mode_enabled(local_settings) and not host_row
            single = quiet and event.get("is_progress") is True and len(message_chunks(text, parse_mode)) == 1
            tracked_msg_id = _get_silent_msg(api, client.bot_id, chat_id) if single else 0

            # An answer never lands under, or revives, the typing it ends (DESIGN §9).
            async with _cue_message(cue, chat_id) as delivered:
                # editMessageText returns False on any failure (too old, deleted, parse error): send instead.
                if tracked_msg_id and _replaceable_progress(api, client.bot_id, chat_id, tracked_msg_id, event):
                    edited = await client.edit_message_text(chat_id, tracked_msg_id, text, parse_mode=parse_mode)
                    if not edited and parse_mode:
                        edited = await client.edit_message_text(chat_id, tracked_msg_id, text, parse_mode="")
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
    def targets(state: Dict[str, Any]) -> set:
        settings = _load_settings(api)
        rows = [row for row in state.get("chats") or [] if isinstance(row, dict)]
        return {chat for row in rows if (chat := _target_chat(settings, {"transport": row.get("transport") or {}}))}

    return TypingCue(read_state=lambda: host_reply_preparation(_data_dir(api)), targets=targets,
                     client_factory=lambda: _telegram_client(api), warn=lambda text: api.log("warning", text),
                     generation=lambda: binding_for(api).snapshot()[1])


def _cue_message(cue: TypingCue | None, chat_id: int):
    """The cue's bracket around one message to ``chat_id`` (an empty one without a cue)."""
    return cue.message(chat_id) if cue is not None else contextlib.nullcontext(lambda: None)


def _make_photo(api):
    async def handle(event: Dict[str, Any]) -> None:
        try:
            local_settings = _load_settings(api)
            chat_id = _target_chat(local_settings, event)
            client = _telegram_client(api, event, chat_id)
            image_base64 = str(event.get("image_base64") or "").strip()
            if chat_id and image_base64:
                # Media cannot replace a text bubble in Telegram — break the
                # silent chain so the next outbound starts a fresh message.
                _clear_silent_msg(api, chat_id)
                await client.send_photo(
                    chat_id,
                    image_base64,
                    caption=str(event.get("caption") or ""),
                    mime=str(event.get("mime") or "image/png"),
                )
        except Exception as exc:
            _report_failure(api, "photo", exc)
    return handle


def _make_video(api):
    async def handle(event: Dict[str, Any]) -> None:
        try:
            local_settings = _load_settings(api)
            chat_id = _target_chat(local_settings, event)
            client = _telegram_client(api, event, chat_id)
            video_base64 = str(event.get("video_base64") or "").strip()
            if chat_id and video_base64:
                # Media cannot replace a text bubble — reset silent tracking.
                _clear_silent_msg(api, chat_id)
                caption = str(event.get("caption") or "")
                mime = str(event.get("mime") or "video/mp4")
                import base64 as _base64
                files = {"video": ("video.mp4", _base64.b64decode(video_base64), mime)}
                data = {"chat_id": str(chat_id), "caption": markdown_to_telegram_html(caption), "parse_mode": "HTML"}
                await client.call("sendVideo", data=data, files=files, timeout=40)
        except Exception as exc:
            _report_failure(api, "video", exc)
    return handle


def _make_document(api):
    async def handle(event: Dict[str, Any]) -> None:
        file_handle = None
        try:
            local_settings = _load_settings(api)
            chat_id = _target_chat(local_settings, event)
            client = _telegram_client(api, event, chat_id)
            file_base64 = str(event.get("file_base64") or "").strip()
            file_ref = event.get("file_ref") if isinstance(event.get("file_ref"), dict) else None
            if chat_id and (file_base64 or file_ref):
                # Media/files cannot replace a text bubble — reset silent tracking.
                _clear_silent_msg(api, chat_id)
                import base64 as _base64
                filename = str(event.get("filename") or "file")
                caption = str(event.get("caption") or "")
                if file_ref:
                    from ouroboros.gateway.files import resolve_task_file_reference

                    task_id = str(event.get("task_id") or file_ref.get("task_id") or "")
                    source = await asyncio.to_thread(resolve_task_file_reference, _data_dir(api), task_id, file_ref)
                    if file_ref["size"] > _MAX_TELEGRAM_UPLOAD_BYTES:
                        t = _LOCALIZED_TEXTS[telegram_i18n.language()]
                        notice = t.format("file_too_large_notice", filename=filename)
                        status = await asyncio.to_thread(_read_status, api)
                        if status.get("state") == "ready" and status.get("public_url"):
                            try:
                                await client.send_message_with_inline_keyboard(chat_id, notice,
                                    [[{"text": t["btn_open_app"], "web_app": {"url": status["public_url"]}}]])
                                return
                            except TelegramRequestRejected as exc:
                                if not exc.plain_retry_safe:
                                    raise
                        await client.send_message(chat_id, notice + " " + t["file_open_app_hint"], parse_mode="")
                        return
                    file_handle = source.open("rb")
                    file_bytes = file_handle
                else:
                    file_bytes = _base64.b64decode(file_base64)
                mime = str(event.get("mime") or "application/octet-stream")
                if _is_native_audio_document(filename, mime):
                    try:
                        await client.send_audio(
                            chat_id,
                            file_bytes,
                            filename=filename,
                            caption=caption,
                            mime=mime,
                        )
                    except TelegramRequestRejected as exc:
                        # Fall back to a plain document only on a definitive
                        # format rejection (HTTP 400). Auth failures and
                        # transient errors (429/5xx) re-raise: retrying the
                        # upload there risks a duplicate delivery.
                        if exc.status_code != 400 or exc.transient:
                            raise
                        if file_handle is not None:
                            file_handle.seek(0)
                        await client.send_document(
                            chat_id,
                            file_bytes,
                            filename=filename,
                            caption=caption,
                        )
                else:
                    await client.send_document(
                        chat_id,
                        file_bytes,
                        filename=filename,
                        caption=caption,
                    )
        except Exception as exc:
            _report_failure(api, "document", exc)
        finally:
            if file_handle is not None:
                file_handle.close()
    return handle


def _is_native_audio_document(filename: str, mime: str) -> bool:
    extension = pathlib.Path(str(filename or "")).suffix.casefold()
    normalized_mime = str(mime or "").split(";", 1)[0].strip().casefold()
    return extension in {".mp3", ".m4a"} or normalized_mime in {
        "audio/mpeg",
        "audio/mp3",
        "audio/mp4",
        "audio/x-m4a",
    }


def _make_links(api):
    async def handle(event: Dict[str, Any]) -> None:
        try:
            local_settings = _load_settings(api)
            chat_id = _target_chat(local_settings, event)
            client = _telegram_client(api, event, chat_id)
            if not chat_id:
                return
            title = str(event.get("title") or "").strip() or "Links"
            raw_actions = event.get("actions") if isinstance(event.get("actions"), list) else []
            actions = []
            for action in raw_actions:
                if not isinstance(action, dict):
                    continue
                label = str(action.get("label") or "").strip()
                url = str(action.get("url") or "").strip()
                if label and url:
                    actions.append((label, url))
            # Shared link-action contract cap: ouroboros.tools.core._MAX_LINK_ACTIONS.
            actions = actions[:12]
            if not actions:
                return
            _clear_silent_msg(api, chat_id)
            keyboard = [[{"text": label, "url": url}] for label, url in actions]
            try:
                await client.send_message_with_inline_keyboard(chat_id, title, keyboard)
            except TelegramRequestRejected as exc:
                if not exc.plain_retry_safe:
                    raise
                api.log("warning", "Telegram links keyboard failed; sending a plain list.")
                plain_text = "\n".join([title, *(f"{label} — {url}" for label, url in actions)])
                await client.send_message(chat_id, plain_text, parse_mode="")
        except Exception as exc:
            _report_failure(api, "links", exc)
    return handle


def _make_quiz(api, cue: TypingCue | None = None):
    async def handle(event: Dict[str, Any]) -> None:
        try:
            local_settings = _load_settings(api)
            chat_id = _target_chat(local_settings, event)
            client = _telegram_client(api, event, chat_id)
            if not chat_id:
                return
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
            card = {
                "project_name": project_name, "option_details": details,
                "recommended_index": recommended_index, "host_facts": host_facts, "lang": lang,
            }
            body = telegram_quiz.render_quiz_text(
                question, plain_labels, stake, assumption, wait_for_answer=wait_for_answer, **card)
            compact = telegram_quiz.render_compact_text(
                plain_labels, project_name=project_name, recommended_index=recommended_index, lang=lang)
            token = telegram_quiz.mint_token(task_id, quiz_id)
            # One button per option; a reply to the card is a free-form answer.
            # Both reach the host's decision ingress (#472). An overflowing card
            # arrives as plain parts followed by the compact keyboard message.
            # Creation holds the card's lock so a lifecycle fact cannot edit (or
            # miss) a card mid-send.
            async with telegram_quiz.card_lock(token), _cue_message(cue, chat_id) as delivered:
                message_id, overflowed = await telegram_quiz.send_quiz_card(
                    client, chat_id, body=body, compact=compact,
                    hint_text=(telegram_quiz.hint_secret(lang) if secret else telegram_quiz.hint(lang) if labels
                               else telegram_quiz.hint_open(lang)),
                    keyboard=telegram_quiz.quiz_keyboard(token, labels),
                    reserve=telegram_quiz.answered_edit_reserve(lang),
                )
                delivered()
                if overflowed:
                    settled_text = compact
                elif wait_for_answer:
                    # The answer edit keeps optional history, not a live waiting claim.
                    settled_text = telegram_quiz.render_quiz_text(question, plain_labels, stake, "", **card)
                else:
                    settled_text = body
                telegram_quiz.remember_quiz(api, client.bot_id, token, {
                    "task_id": task_id, "quiz_id": quiz_id, "chat_id": chat_id,
                    "message_id": int(message_id or 0), "options": labels,
                    "text": settled_text, **({"secret": True} if secret else {}),
                })
                await telegram_quiz.apply_retained_fact(api, token, lang, client_factory=lambda: client, published=event)
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


def register(api):
    telegram_i18n.configure(_data_dir(api))
    binding = binding_for(api)
    binding.read_now(api)  # the token in force from the first send; the observer keeps it current
    if callable(getattr(api, "on_unload", None)):
        api.on_unload(binding.close)  # unload, Off or a revoked grant: no client of this load sends again
    cue = _typing_cue(api)  # Telegram failures stay inside it, so they never spend restart budget
    api.register_supervised_task("poller", _make_poller(api), restart_policy="on_failure", max_restarts=10)
    api.register_supervised_task("notifier", _make_notifier(api, trust_env=_HONOR_ENV_PROXIES), restart_policy="on_failure", max_restarts=10)
    api.register_supervised_task("typing_cue", cue.run, restart_policy="on_failure", max_restarts=10)
    api.subscribe_event("chat.outbound", _make_outbound(api, cue))
    api.subscribe_event("chat.typing", _make_typing(cue))
    api.subscribe_event("chat.photo", _make_photo(api))
    api.subscribe_event("chat.video", _make_video(api))
    api.subscribe_event("chat.document", _make_document(api))
    api.subscribe_event("chat.links", _make_links(api))
    api.subscribe_event("chat.quiz", _make_quiz(api, cue))
    api.subscribe_event("chat.quiz_state", _make_quiz_state(api))
    # GET hydrates the declarative form with what is stored; POST saves it.
    api.register_route("settings/save", handler=_make_settings_save(api), methods=("GET", "POST"))
    api.register_route("miniapp/status", handler=_make_status(api), methods=("POST",))
    # Share… in the Mini App: the bridge's own client prepares the owner's message for Telegram's chat picker.
    api.register_route("miniapp/prepare-share", handler=make_prepare_share(api, _bridge_client), methods=("POST",))
    api.register_route("connection", handler=_make_connection(api), methods=("GET", "POST"))
    api.register_settings_section(
        "telegram",
        title="Telegram",
        schema={
            "components": [
                {
                    "type": "markdown",
                    "text": (
                        "The bot token lives in Settings → Secrets (TELEGRAM_BOT_TOKEN). While this skill is on, a saved, "
                        "replaced or cleared token reaches the text bridge and the Mini App within seconds; saving it grants, "
                        "turns on or resumes nothing; Connection below shows what actually runs. "
                        "If Telegram needs a proxy, configure Telegram proxy below. It applies to all Telegram API calls, "
                        "including the menu button; the Mini App tunnel and web traffic keep their existing routes.\n\n"
                        "**Command mode**: controls which slash commands can be sent from Telegram. "
                        "Use `/menu` in Telegram to see available commands as inline buttons.\n\n"
                        "**Mirror mode**: *all* mirrors every chat message (including web UI) to Telegram — requires Chat ID. "
                        "*Telegram only* mirrors Telegram-originated conversations, plus Ouroboros's reminders, skill notices and Safety notices "
                        "when a Chat ID is set.\n\n"
                        "**Mini App Beta** uses a best-effort Cloudflare Quick Tunnel with no SLA and no SSE support. "
                        "It targets native Telegram clients; Telegram WebA/WebK are not supported."
                    ),
                },
                *connection_components(),
                {
                    "type": "form",
                    "route": "settings/save",
                    "method": "POST",
                    "fields": [
                        {"name": "telegram_proxy_status", "label": "Saved Telegram proxy (at form load)",
                         "type": "text", "disabled": True, "default": "Not configured"},
                        {"name": "TELEGRAM_PROXY", "label": "Telegram proxy", "type": "password",
                         "placeholder": "scheme://[user:password@]host:port",
                         "help": "Optional HTTP, HTTPS, SOCKS5 or SOCKS5h proxy for Telegram only. "
                                 "Leave empty to keep the saved proxy. Disable and re-enable the skill after changing it."},
                        {"name": "clear_telegram_proxy", "label": "Clear saved Telegram proxy", "type": "checkbox",
                         "help": "Remove the proxy instead of replacing it."},
                        {"name": "TELEGRAM_API_BASE", "label": "Bot API base URL", "type": "text",
                         "placeholder": "https://api.telegram.org",
                         "help": "Origin of a telegram-bot-api server you run, e.g. http://127.0.0.1:8081. Every "
                                 "Telegram call goes there; polling after its current long poll or retry wait (up to "
                                 "about a minute). Empty: the cloud. Moving the bot to another server: quit first (skill's notes)."},
                        {"name": "TELEGRAM_API_FILES_DIR", "label": "Bot API files directory", "type": "text",
                         "placeholder": "/absolute/path/to/server-file-store",
                         "help": "For a server started with --local: its file store, mounted at the same absolute "
                                 "path here. Downloads read files only inside it. Empty: HTTP downloads only."},
                        {"name": "TELEGRAM_COMMAND_MODE", "label": "Command mode", "type": "select",
                         "options": [
                             {"value": "full_access", "label": "Full access (default)"},
                             {"value": "safe_commands", "label": "Safe commands only"},
                             {"value": "strict", "label": "Strict"},
                         ],
                         "help": "Panic is available in every mode. Full access forwards raw owner commands. "
                                 "Safe otherwise allows /status and /bg status only; Strict blocks other slash commands.",
                         "placeholder": "full_access"},
                        {"name": "TELEGRAM_MIRROR_MODE", "label": "Mirror mode", "type": "select",
                         "options": [
                             {"value": "all", "label": "Mirror all messages (web + Telegram)"},
                             {"value": "telegram_only", "label": "Telegram conversations, plus reminders, skill notices and Safety notices"},
                         ],
                         "placeholder": "all"},
                        {"name": "TELEGRAM_CHAT_ID", "label": "Telegram Chat ID", "type": "text", "placeholder": "required for 'all' mode"},
                        {"name": "TELEGRAM_MAX_UPDATES_PER_POLL", "label": "Max updates per poll", "type": "number", "placeholder": "20"},
                        {"name": "TELEGRAM_SILENT_MODE", "label": "Silent mode", "type": "select",
                         "options": [
                             {"value": "off", "label": "Off"},
                             {"value": "on", "label": "On (without sound)"},
                         ],
                         "help": "On sends replies, the final answer and progress without sound; a progress update replaces the one just before it. Questions, files and problems keep their sound.",
                         "placeholder": "off"},
                        {"name": "TELEGRAM_SUBAGENT_CARDS", "label": "Subagent cards", "type": "select",
                         "options": [
                             {"value": "on", "label": "On"},
                             {"value": "off", "label": "Off"},
                         ],
                         "help": "One updating message per subagent.",
                         "placeholder": "on"},
                        {"name": "TELEGRAM_MIRROR_PROGRESS", "label": "Mirror progress telemetry", "type": "select",
                         "options": [
                             {"value": "off", "label": "Off (default)"},
                             {"value": "on", "label": "On"},
                         ],
                         "help": "On streams the main agent's progress; Off keeps replies only.",
                         "placeholder": "off"},
                        {"name": "TELEGRAM_MINIAPP_ENABLED", "label": "Telegram Mini App", "type": "select",
                         "options": [
                             {"value": "on", "label": "On (default after owner binding)"},
                             {"value": "off", "label": "Off — keep text bridge only"},
                         ],
                         "placeholder": "on"},
                        {"name": "TELEGRAM_NOTIFY_TASKS", "label": "Notify on task completion", "type": "select",
                         "options": [
                             {"value": "off", "label": "Off — no extra completion messages"},
                             {"value": "on", "label": "On — completion messages"},
                         ],
                         "help": "Off also suppresses extra failure and cancellation notices. Replies, files and "
                                 "questions still arrive; task details retain all outcomes and review evidence.",
                         "placeholder": "off"},
                        {"name": "TELEGRAM_NOTIFY_BUDGET", "label": "Notify on budget thresholds", "type": "select",
                         "options": [
                             {"value": "off", "label": "Off"},
                             {"value": "on", "label": "On — ⚠️ at 80% / 90% / 100%"},
                         ],
                         "placeholder": "off"},
                    ],
                    "submit_label": "Save Telegram settings",
                },
            ]
        },
    )
    # Unsupported OS/architecture degrades inside register_miniapp(). Invalid
    # host runtime or unsafe/unwritable state must fail the complete load rather
    # than leave a partially registered skill claiming to be healthy.
    companion = "miniapp_gateway" if register_miniapp(api) else ""
    # One observer applies a saved, replaced or cleared token to every part, the Mini App included,
    # independent of the poller's long poll or a parked rejection.
    api.register_supervised_task("token_watch", make_token_observer(api, companion, cue.wake),
                                 restart_policy="on_failure", max_restarts=10)
