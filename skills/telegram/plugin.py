from __future__ import annotations

import asyncio
import json
import os
import pathlib
import re
from typing import Any, Dict

import httpx
from starlette.responses import JSONResponse

# Native skills are version-resynced with Ouroboros core; reuse its truncation
# SSOT instead of vendoring a divergent payload-local copy.
from ouroboros.net_transport import env_proxies_configured
from ouroboros.utils import in_worker_process, truncate_review_artifact

from .lib.telegram_api import (
    TELEGRAM_RETRY_INITIAL_SEC,
    TelegramClient,
    TelegramRequestRejected,
    TelegramTransportError,
    is_transient_telegram_error,
    markdown_to_telegram_html,
    next_telegram_retry_delay,
    _LOCALIZED_TEXTS,
)
from .lib.telegram_state import (
    _state_file, _load_settings, _is_silent_mode_enabled,
    _get_silent_msg, _set_silent_msg, _clear_silent_msg, _subagent_cards_enabled,
    _mirror_progress_enabled, _render_subagent_card, _data_dir,
    _jsonl_tail, _load_runtime_state, _read_json_file,
)
from .lib import telegram_inbound, telegram_quiz
from .lib.telegram_health import _collect_health, _build_menu_tasks
from .lib.telegram_notifier import _make_notifier
from .lib.miniapp_registration import _read_status, register as register_miniapp
from .scripts.telegram_settings import (
    TelegramSettingsError,
    merge_settings,
    request_may_change_owner,
)

# Decided once in the server process: a proxy-routed install keeps its only egress, every
# other install is isolated from ambient proxy and SSL_CERT env; the worker guard mirrors
# the core's macOS fork-safety rule. Residual: a CA-only install (custom CA via
# SSL_CERT_FILE/SSL_CERT_DIR, no proxy) loses Telegram egress until that CA is trusted
# system-wide or a proxy is set; the menu client stays pinned either way.
_HONOR_ENV_PROXIES = (not in_worker_process()) and env_proxies_configured()

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

_COMMAND_MODE_STRICT = "strict"
_COMMAND_MODE_SAFE = "safe_commands"
_COMMAND_MODE_FULL = "full_access"
_VALID_COMMAND_MODES = frozenset({_COMMAND_MODE_STRICT, _COMMAND_MODE_SAFE, _COMMAND_MODE_FULL})


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

    header = t["menu_title"].format(command_mode=command_mode, lang=lang.upper())
    keyboard = [
        [
            {"text": t["btn_metrics"], "callback_data": "nav:status"},
            {"text": t["btn_mind"], "callback_data": "nav:mind"},
        ],
        [
            {"text": "📋 Задачи" if lang == "ru" else "📋 Tasks", "callback_data": "nav:tasks"},
        ],
        [
            {"text": t["btn_settings"], "callback_data": "nav:settings"},
        ]
    ]
    return header, keyboard


def _build_menu_status(command_mode: str, lang: str = "en", info_text: str = "") -> tuple[str, list[list[dict]]]:
    """Return status header and keyboard with Refresh and Back button."""
    t = _LOCALIZED_TEXTS[lang]
    header = t["metrics_title"].format(info_text=info_text)
    keyboard = [
        [{"text": t["btn_refresh"], "callback_data": "cmd_act:update_status"}],
        [{"text": t["btn_back"], "callback_data": "nav:menu"}]
    ]
    return header, keyboard


def _build_menu_mind(command_mode: str, lang: str = "en", bg_enabled: bool = False, thoughts_text: str = "") -> tuple[str, list[list[dict]]]:
    """Return mind controlling header and buttons."""
    t = _LOCALIZED_TEXTS[lang]
    state_str = t["mind_state_active"] if bg_enabled else t["mind_state_sleeping"]
    header = t["mind_title"].format(state_str=state_str)
    if thoughts_text:
        header += t["mind_thoughts"].format(thoughts_text=thoughts_text)

    row = []
    if command_mode == _COMMAND_MODE_FULL:
        if bg_enabled:
            row.append({"text": t["btn_stop_bg"], "callback_data": "cmd_act:bg_stop"})
        else:
            row.append({"text": t["btn_start_bg"], "callback_data": "cmd_act:bg_start"})

    keyboard = []
    if row:
        keyboard.append(row)
    keyboard.append([{"text": t["btn_thoughts"], "callback_data": "cmd_act:bg_thoughts"}])
    keyboard.append([{"text": t["btn_back"], "callback_data": "nav:menu"}])
    return header, keyboard


def _load_recent_thoughts(api) -> str:
    """Read the last few blocks from progress.jsonl and build a text snapshot."""
    progress_file = _data_dir(api) / "logs" / "progress.jsonl"
    if not progress_file.exists():
        return "_No thoughts log created yet._"
    try:
        entries, _prefix_omitted = _jsonl_tail(
            progress_file,
            max_entries=40,
            tail_bytes=256 * 1024,
        )
        recent = []
        for elem in reversed(entries):
            text = str(elem.get("text") or elem.get("message") or elem.get("thoughts") or "").strip()
            if not text or len(text) <= 10:
                continue
            text = text.replace("`", "").replace("*", "").replace("#", "")
            preview = truncate_review_artifact(text, 100)
            timestamp = str(elem.get("ts") or elem.get("timestamp") or elem.get("created_at") or "")
            task_id = str(elem.get("task_id") or "").strip()
            if preview != text:
                ref = "data/logs/progress.jsonl"
                if timestamp:
                    ref += f" @ {timestamp}"
                if task_id:
                    ref += f" task {task_id}"
                preview += f"\n  Full ref: {ref}"
            if timestamp:
                time_match = re.search(r"T(\d{2}:\d{2})", timestamp)
                time_str = f"[{time_match.group(1)}] " if time_match else f"[{timestamp[:10]}] "
            else:
                time_str = ""
            recent.append(f"• {time_str}{preview}")
            if len(recent) >= 4:
                break
        if not recent:
            return "_Thoughts log is empty or waiting for next cycle._"
        return "\n".join(recent) + "\n\nShowing up to 4 recent entries. Full source: data/logs/progress.jsonl"
    except Exception as exc:
        return f"_Failed to read log: {exc}_"


def _bot_commands(command_mode: str) -> list:
    """Telegram '/' menu command list. Owner-control commands appear only in
    full_access — they already forward via the raw-command path, so listing them
    here just makes them discoverable/tappable (no new authority)."""
    cmds = [
        {"command": "menu", "description": "Interactive panel / Меню"},
        {"command": "language", "description": "Select language / Выбор языка"},
        {"command": "status", "description": "Request status / Статус"},
        {"command": "help", "description": "Usage guide / Справка"},
    ]
    if command_mode == _COMMAND_MODE_FULL:
        cmds += [
            {"command": "evolve", "description": "Start an evolution campaign / Запустить эволюцию"},
            {"command": "bg", "description": "Background consciousness on/off / Фоновое сознание"},
            {"command": "review", "description": "Run a self-review / Само-ревью"},
            {"command": "restart", "description": "Restart the agent / Перезапуск"},
            {"command": "panic", "description": "Emergency stop / Аварийный стоп"},
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


def _build_language_keyboard(lang: str = "en") -> tuple[str, list[list[dict]]]:
    """Return (header_text, inline_keyboard_rows) for language selection."""
    t = _LOCALIZED_TEXTS[lang]
    header = t["lang_title"]
    rows = [
        [
            {"text": t["lang_en"], "callback_data": "set_lang:en"},
            {"text": t["lang_ru"], "callback_data": "set_lang:ru"}
        ],
        [{"text": t["btn_back"], "callback_data": "nav:menu"}]
    ]
    return header, rows


def _make_settings_save(api):
    async def _settings_save(request):
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
        allowed = {"TELEGRAM_CHAT_ID", "TELEGRAM_MAX_UPDATES_PER_POLL", "TELEGRAM_MIRROR_MODE", "TELEGRAM_COMMAND_MODE", "TELEGRAM_LANGUAGE", "TELEGRAM_SILENT_MODE", "TELEGRAM_SUBAGENT_CARDS", "TELEGRAM_MIRROR_PROGRESS", "TELEGRAM_NOTIFY_TASKS", "TELEGRAM_NOTIFY_BUDGET", "TELEGRAM_MINIAPP_ENABLED"}
        payload = {key: data.get(key) for key in allowed if key in data}
        owner_ignored = False
        if "TELEGRAM_CHAT_ID" in payload and not request_may_change_owner(request):
            payload.pop("TELEGRAM_CHAT_ID", None)
            owner_ignored = True
        try:
            merge_settings(pathlib.Path(api.get_state_dir()), payload)
        except TelegramSettingsError as exc:
            return JSONResponse(
                {"ok": False, "message": str(exc)},
                status_code=409,
            )
        message = "Telegram settings saved."
        if owner_ignored:
            message += " Owner binding was left unchanged."
        return JSONResponse({"ok": True, "owner_ignored": owner_ignored, "message": message})
    return _settings_save


def _bridge_status(api) -> dict[str, Any]:
    try:
        settings = _load_settings(api)
    except TelegramSettingsError:
        return {
            "state": "error",
            "owner_bound": False,
            "poller": "failed",
            "command_mode": _COMMAND_MODE_STRICT,
            "mirror_mode": "all",
            "reason_code": "settings_invalid",
        }
    try:
        owner_bound = int(str(settings.get("TELEGRAM_CHAT_ID") or "").strip()) > 0
    except (TypeError, ValueError):
        owner_bound = False
    try:
        token_configured = bool(str(api.get_settings(["TELEGRAM_BOT_TOKEN"]).get("TELEGRAM_BOT_TOKEN") or "").strip())
    except Exception:
        token_configured = False
    command_mode = str(settings.get("TELEGRAM_COMMAND_MODE") or _COMMAND_MODE_FULL).strip().lower()
    if command_mode not in _VALID_COMMAND_MODES:
        command_mode = _COMMAND_MODE_STRICT
    mirror_mode = str(settings.get("TELEGRAM_MIRROR_MODE") or "all").strip().lower()
    if mirror_mode not in {"all", "telegram_only"}:
        mirror_mode = "all"
    state = "ready" if token_configured and owner_bound else (
        "waiting_owner" if token_configured else "missing_token"
    )
    poller = "configured" if token_configured else "blocked_missing_token"
    reason_code = ""
    runtime = _read_json_file(_state_file(api, "bridge_status.json"))
    if token_configured and str(runtime.get("state") or "") == "error":
        state = "error"
        poller = "failed"
        reason_code = str(runtime.get("reason_code") or "telegram_rejected")[:64]
    elif token_configured and str(runtime.get("state") or "") == "degraded":
        state = "degraded"
        poller = "degraded"
        reason_code = str(runtime.get("reason_code") or "telegram_startup_deferred")[:64]
    result = {
        "state": state,
        "owner_bound": owner_bound,
        "poller": poller,
        "command_mode": command_mode,
        "mirror_mode": mirror_mode,
    }
    if reason_code:
        result["reason_code"] = reason_code
    return result


def _save_bridge_status(api, state: str, reason_code: str = "") -> None:
    path = _state_file(api, "bridge_status.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    try:
        tmp.write_text(
            json.dumps({"state": state, "reason_code": reason_code}),
            encoding="utf-8",
        )
        tmp.replace(path)
    except OSError:
        api.log("warning", "Telegram bridge status could not be persisted.")


def _make_status(api):
    async def status(_request):
        return JSONResponse({
            "bridge": _bridge_status(api),
            "mini_app": _read_status(api),
        })

    return status


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


def _load_offset(api) -> int:
    path = _state_file(api, "poll_offset.json")
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return int(data.get("offset") or 0)
    except Exception:
        pass
    return 0


def _save_offset(api, offset: int) -> None:
    path = _state_file(api, "poll_offset.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"offset": offset}), encoding="utf-8")
    tmp.replace(path)


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
    unbounded = "без лимита" if lang == "ru" else "unbounded"
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

    template = t["metrics_budget_status"]
    template = template.replace("{spent_usd:.4f}", "{spent_usd_str}")
    template = template.replace("{total_budget:.2f}", "{total_budget_str}")
    template = template.replace("{rem:.4f}", "{rem_str}")

    status_str = template.format(
        spent_usd_str=spent_spec,
        total_budget_str=total_spec,
        rem_str=rem_spec,
        branch=branch,
        bg_status=bg_status_raw
    )
    status_str += "\n" + await asyncio.to_thread(_collect_health, api, lang)
    return status_str


def _poller_preferences(api):
    settings = _load_settings(api)
    pinned = str(settings.get("TELEGRAM_CHAT_ID") or "").strip()
    maximum = _setting_int(settings, "TELEGRAM_MAX_UPDATES_PER_POLL", 20, minimum=1, maximum=100)
    mode = str(settings.get("TELEGRAM_COMMAND_MODE") or _COMMAND_MODE_FULL).strip().lower()
    mode = mode if mode in _VALID_COMMAND_MODES else _COMMAND_MODE_STRICT
    language = str(settings.get("TELEGRAM_LANGUAGE") or "en").strip().lower()
    language = language if language in ("en", "ru") else "en"
    return settings, pinned, maximum, mode, language


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


async def _validate_bot(api, client, command_mode: str, lang: str) -> None:
    """Validate the token (getMe) and finish bridge startup bookkeeping."""
    await client.call("getMe")
    # The owner-control commands (evolve/bg/review/restart/panic) only
    # appear in full_access — they already forward via the raw-command
    # path; listing them here just makes them discoverable/tappable.
    try:
        await client.call("setMyCommands", data={"commands": json.dumps(_bot_commands(command_mode))})
        api.log("info", "Telegram bot commands configured successfully")
    except Exception as exc:
        api.log("warning", f"Failed to set Telegram bot commands: {exc}")

    _save_bridge_status(api, "ready")
    api.log("info", f"Telegram poller started (command_mode={command_mode}, lang={lang})")


async def _start_poller(api):
    try:
        protected_settings = api.get_settings(["TELEGRAM_BOT_TOKEN"])
        _settings, pinned_chat, max_updates, command_mode, lang = _poller_preferences(api)
        client = TelegramClient(protected_settings.get("TELEGRAM_BOT_TOKEN", ""), trust_env=_HONOR_ENV_PROXIES)
        offset = _load_offset(api)
        try:
            await _validate_bot(api, client, command_mode, lang)
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
            _save_bridge_status(api, "degraded", "telegram_startup_deferred")
            api.log("warning", "Telegram token validation deferred by a transport failure; polling waits for the network.")
        return client, offset, pinned_chat, max_updates, command_mode, lang, validated
    except TelegramSettingsError:
        _save_bridge_status(api, "error", "settings_invalid")
        api.log("error", "Telegram settings are invalid; owner binding is closed.")
        raise
    except Exception as exc:
        _save_bridge_status(api, "error", "telegram_startup_failed")
        api.log("error", f"Telegram token validation failed: {exc}")
        raise


def _fail_permanent_poll_rejection(api, exc, revalidating: bool) -> None:
    """Persist bridge status + log for a permanent polling rejection."""
    if revalidating:
        # A permanent getMe rejection during in-loop revalidation
        # is the startup token-validation failure, merely deferred
        # by a dead network at startup — label it the same way.
        _save_bridge_status(api, "error", "telegram_startup_failed")
        api.log("error", f"Telegram token validation failed: {exc}")
    else:
        _save_bridge_status(api, "error", "telegram_rejected")
        api.log("error", "Telegram polling was permanently rejected.")


async def _poller(api) -> None:
    client, offset, pinned_chat, max_updates, command_mode, lang, validated = await _start_poller(api)

    retry_delay = TELEGRAM_RETRY_INITIAL_SEC
    degraded_cause = ""
    while True:
        revalidating = False
        try:
            if not validated:
                revalidating = True
                await _validate_bot(api, client, command_mode, lang)
                revalidating = False
                validated = True
            updates = await client.get_updates(offset)
            if degraded_cause:
                api.log("info", f"Telegram poller recovered after {degraded_cause}; polling resumed.")
                degraded_cause = ""
            retry_delay = TELEGRAM_RETRY_INITIAL_SEC
            if updates:
                local_settings, pinned_chat, max_updates, command_mode, lang = _poller_preferences(api)

            for update in updates[:max_updates]:
                authorized_chat = None
                try:  # Isolate one bad update without changing offset/retry semantics.
                    update_id = int(update.get("update_id") or 0)
                    if update_id >= offset:
                        offset = update_id + 1

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
                                # Clear any stale tracked message id for this chat so the
                                # next outbound starts a fresh bubble in either direction.
                                _clear_silent_msg(api, cb_chat_id)
                                toast_key = "silent_toggled_on" if new_value == "on" else "silent_toggled_off"
                                await client.answer_callback_query(cb_id, text=_LOCALIZED_TEXTS[lang][toast_key])
                                header, keyboard = _build_menu_settings(api, command_mode, lang)
                                await _edit_panel(api, client, cb_chat_id, cb_message_id, header, keyboard)
                                continue

                            elif action == "bg_thoughts":
                                await client.answer_callback_query(cb_id, text=_LOCALIZED_TEXTS[lang]["extracting_thoughts"])
                                bg_enabled = _is_bg_consciousness_active(api)
                                thoughts = await asyncio.to_thread(_load_recent_thoughts, api)
                                header, keyboard = _build_menu_mind(command_mode, lang, bg_enabled, thoughts)
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
                                })
                                # Give it a tiny moment to commit setting then refresh mind panel
                                await asyncio.sleep(0.8)
                                bg_enabled = _is_bg_consciousness_active(api)
                                header, keyboard = _build_menu_mind(command_mode, lang, bg_enabled)
                                await _edit_panel(api, client, cb_chat_id, cb_message_id, header, keyboard)
                                continue

                        # --- Handle language selection buttons ---
                        if cb_data.startswith("set_lang:"):
                            new_lang = cb_data.split(":", 1)[1]
                            if new_lang in ("en", "ru"):
                                merge_settings(pathlib.Path(api.get_state_dir()), {"TELEGRAM_LANGUAGE": new_lang})

                                lang = new_lang
                                await client.answer_callback_query(cb_id, text=_LOCALIZED_TEXTS[lang]["lang_changed"])

                                # Smoothly return to menu panel in updated language
                                header, keyboard = _build_menu_keyboard(command_mode, lang)
                                await _edit_panel(api, client, cb_chat_id, cb_message_id, header, keyboard)
                                continue

                        # --- Quiz answers (#472): a tapped option reaches the host's
                        # decision ingress; the owner is already verified above.
                        if cb_data.startswith("qz:"):
                            await telegram_quiz.answer_from_callback(
                                api, client, cb_data, cb_id=cb_id,
                                update_id=update_id, lang=lang, post=_host_post,
                            )
                            continue

                        if cb_data.startswith(("set_model:", "set_budget:")):
                            await client.answer_callback_query(
                                cb_id,
                                text=("Используйте Mini App." if lang == "ru" else "Use the Mini App."),
                            )
                            continue

                        await client.answer_callback_query(cb_id, text=_LOCALIZED_TEXTS[lang]["unknown_command"])
                        continue

                    # --- Handle regular messages ---
                    await _handle_owner_message(api, client, update, update_id, command_mode, lang)
                except Exception as exc:
                    api.log(
                        "warning",
                        f"Telegram update processing failed ({type(exc).__name__}).",
                    )
                    if authorized_chat:
                        try:
                            notice = (
                                "Не удалось передать это обновление Telegram в Ouroboros."
                                if lang == "ru"
                                else "Could not deliver that Telegram update to Ouroboros."
                            )
                            await client.send_message(int(authorized_chat), notice)
                        except Exception:
                            api.log("warning", "Telegram update failure notice could not be delivered.")
            if updates:
                _save_offset(api, offset)
            await asyncio.sleep(0.1)
        except (TelegramRequestRejected, TelegramTransportError) as exc:
            if isinstance(exc, TelegramRequestRejected) and not exc.transient:
                _fail_permanent_poll_rejection(api, exc, revalidating)
                raise
            # Transition logging only: one line entering degraded, one on
            # recovery; consecutive transient failures stay quiet while the
            # monotone backoff (reset by any successful poll) paces retries.
            if not degraded_cause:
                degraded_cause = type(exc).__name__
                api.log("warning", f"Telegram poller degraded ({degraded_cause}): {exc}")
            await asyncio.sleep(retry_delay)
            retry_delay = next_telegram_retry_delay(retry_delay)
        except TelegramSettingsError:
            _save_bridge_status(api, "error", "settings_invalid")
            api.log("error", "Telegram settings are invalid; owner binding is closed.")
            raise
        except Exception as exc:
            _save_bridge_status(api, "error", "poller_failed")
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

    # Handle /language command locally — always allowed
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

    reply_to = message.get("reply_to_message") or {}
    if safe_text and not _SLASH_COMMAND_RE.match(text) and isinstance(reply_to, dict) and reply_to:
        quiz_ref = telegram_quiz.quiz_for_message(
            api, chat_id, int(reply_to.get("message_id") or 0),
        )
        if quiz_ref is not None:
            # A reply to a quiz card is the owner's own answer to
            # that question, not a new chat turn (#472).
            # Commands keep their existing dispatch even in replies. Other
            # answers, including code-formatted command names, stay verbatim.
            answer_text = str(message.get("text") or message.get("caption") or "")
            await telegram_quiz.answer_from_reply(
                api, client, quiz_ref, answer_text, chat_id=chat_id,
                update_id=update_id, lang=lang, post=_host_post,
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


def _make_poller(api):
    return lambda: _poller(api)


def _make_outbound(api):
    async def handle(event: Dict[str, Any]) -> None:
        try:
            protected_settings = api.get_settings(["TELEGRAM_BOT_TOKEN"])
            local_settings = _load_settings(api)
            client = TelegramClient(protected_settings.get("TELEGRAM_BOT_TOKEN", ""), trust_env=_HONOR_ENV_PROXIES)
            chat_id = _target_chat(local_settings, event)
            if not chat_id:
                return
            lang = str(local_settings.get("TELEGRAM_LANGUAGE") or "en").strip().lower()

            # Subagent lifecycle → one dedicated bubble per subagent, edited in
            # place across its lifecycle (not a flood of new messages). Since 6.22
            # the supervisor emits one is_progress chat.outbound per subagent state
            # transition; mirroring them raw spams the chat / collapses over the
            # real reply in silent mode.
            sub_event = str(event.get("subagent_event") or "").strip().lower()
            if sub_event:
                if _subagent_cards_enabled(local_settings):
                    await _render_subagent_card(api, client, chat_id, event, sub_event, lang)
                return

            # Generic (non-subagent) progress telemetry → dropped by default; the
            # typing indicator already signals "working". Opt in via the toggle.
            if event.get("is_progress") and not _mirror_progress_enabled(local_settings):
                return

            # Routing receipts ride the outbound bus as typed annotations with
            # suppress_bubble (no text). The only one worth a Telegram push is
            # the actionable refusal: a numbered destination list (#198). The
            # owner replies with a number in plain words — the LLM router reads
            # this list from history and dispatches the choice (P5 LLM-first,
            # no keyword gate) — or taps the picker card in the web UI.
            if str(event.get("annotation_type") or "") == "routing_ack":
                if str(event.get("status") or "") != "needs_manual_target":
                    return
                raw_options = event.get("options") if isinstance(event.get("options"), list) else []
                from ouroboros.project_dialogue import routing_option_label

                labels = [routing_option_label(option) for option in raw_options]
                labels = [label for label in labels if label][:8]
                if not labels:
                    return
                lines = ["I couldn't pick a destination for your last message. Options:"]
                lines.extend(f"{index}. {label}" for index, label in enumerate(labels, 1))
                if len(raw_options) > len(labels):
                    lines.append(f"…and {len(raw_options) - len(labels)} more in the web chat.")
                lines.append("Reply with a number, or tap an option in the web chat.")
                _clear_silent_msg(api, chat_id)
                await client.send_message(chat_id, "\n".join(lines), parse_mode="")
                return

            text = str(event.get("text") or "").strip()
            if not text:
                return
            # Honor the host's markdown hint: plain text (markdown=False) is sent
            # verbatim so literal *, _, `, [] aren't mis-parsed as Telegram
            # formatting; an absent/True hint renders markdown→HTML as before.
            parse_mode = "" if event.get("markdown") is False else "HTML"

            silent_on = _is_silent_mode_enabled(local_settings)
            tracked_msg_id = _get_silent_msg(api, chat_id) if silent_on else 0

            # Silent mode: try to edit the previously tracked message in-place.
            # editMessageText returns False on any failure (too old, deleted,
            # identical content, parse error) so we fall back to sendMessage.
            if silent_on and tracked_msg_id:
                edited = await client.edit_message_text(chat_id, tracked_msg_id, text, parse_mode=parse_mode)
                if not edited and parse_mode:
                    edited = await client.edit_message_text(chat_id, tracked_msg_id, text, parse_mode="")
                if edited:
                    return
                # Edit failed (likely too old or already identical) — clear
                # tracking and fall through to sendMessage path.
                _clear_silent_msg(api, chat_id)

            msg_id = await client.send_message(chat_id, text, parse_mode=parse_mode)

            if silent_on and msg_id:
                _set_silent_msg(api, chat_id, msg_id)
        except Exception as exc:
            api.log("error", f"Telegram outbound error: {exc}")
    return handle


def _make_typing(api):
    async def handle(event: Dict[str, Any]) -> None:
        try:
            protected_settings = api.get_settings(["TELEGRAM_BOT_TOKEN"])
            local_settings = _load_settings(api)
            client = TelegramClient(protected_settings.get("TELEGRAM_BOT_TOKEN", ""), trust_env=_HONOR_ENV_PROXIES)
            chat_id = _target_chat(local_settings, event)
            if chat_id:
                await client.send_chat_action(chat_id, "typing")
        except Exception as exc:
            api.log("error", f"Telegram typing error: {exc}")
    return handle


def _make_photo(api):
    async def handle(event: Dict[str, Any]) -> None:
        try:
            protected_settings = api.get_settings(["TELEGRAM_BOT_TOKEN"])
            local_settings = _load_settings(api)
            client = TelegramClient(protected_settings.get("TELEGRAM_BOT_TOKEN", ""), trust_env=_HONOR_ENV_PROXIES)
            chat_id = _target_chat(local_settings, event)
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
            api.log("error", f"Telegram photo error: {exc}")
    return handle


def _make_video(api):
    async def handle(event: Dict[str, Any]) -> None:
        try:
            protected_settings = api.get_settings(["TELEGRAM_BOT_TOKEN"])
            local_settings = _load_settings(api)
            client = TelegramClient(protected_settings.get("TELEGRAM_BOT_TOKEN", ""), trust_env=_HONOR_ENV_PROXIES)
            chat_id = _target_chat(local_settings, event)
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
            api.log("error", f"Telegram video error: {exc}")
    return handle


def _make_document(api):
    async def handle(event: Dict[str, Any]) -> None:
        try:
            protected_settings = api.get_settings(["TELEGRAM_BOT_TOKEN"])
            local_settings = _load_settings(api)
            client = TelegramClient(protected_settings.get("TELEGRAM_BOT_TOKEN", ""), trust_env=_HONOR_ENV_PROXIES)
            chat_id = _target_chat(local_settings, event)
            file_base64 = str(event.get("file_base64") or "").strip()
            if chat_id and file_base64:
                # Media/files cannot replace a text bubble — reset silent tracking.
                _clear_silent_msg(api, chat_id)
                import base64 as _base64
                filename = str(event.get("filename") or "file")
                caption = str(event.get("caption") or "")
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
            api.log("error", f"Telegram document error: {exc}")
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
            protected_settings = api.get_settings(["TELEGRAM_BOT_TOKEN"])
            local_settings = _load_settings(api)
            client = TelegramClient(protected_settings.get("TELEGRAM_BOT_TOKEN", ""), trust_env=_HONOR_ENV_PROXIES)
            chat_id = _target_chat(local_settings, event)
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
            api.log("error", f"Telegram links error: {exc}")
    return handle


def _make_quiz(api):
    async def handle(event: Dict[str, Any]) -> None:
        try:
            protected_settings = api.get_settings(["TELEGRAM_BOT_TOKEN"])
            local_settings = _load_settings(api)
            client = TelegramClient(protected_settings.get("TELEGRAM_BOT_TOKEN", ""), trust_env=_HONOR_ENV_PROXIES)
            chat_id = _target_chat(local_settings, event)
            if not chat_id:
                return
            question = str(event.get("question") or "").strip()
            raw_options = event.get("options") if isinstance(event.get("options"), list) else []
            labels = []
            for option in raw_options:
                if isinstance(option, dict):
                    label = str(option.get("label") or "").strip()
                    if label:
                        labels.append(label)
            # Shared quiz contract cap: ouroboros.tools.core._MAX_QUIZ_OPTIONS.
            labels = labels[:6]
            if not question or len(labels) < 2:
                return
            task_id = str(event.get("task_id") or "").strip()
            quiz_id = str(event.get("quiz_id") or "").strip()
            if not task_id or not quiz_id:
                return  # the host refuses anonymous quizzes; nothing to answer to
            _clear_silent_msg(api, chat_id)
            stake = str(event.get("stake") or "").strip()
            assumption = str(event.get("assumption") or "").strip()
            lang = _poller_preferences(api)[4]
            body = telegram_quiz.render_quiz_text(question, labels, stake, assumption)
            token = telegram_quiz.mint_token(task_id, quiz_id)
            # One button per option; a reply to the card is a free-form answer.
            # Both reach the host's decision ingress (#472).
            message_id = await client.send_message_with_inline_keyboard(
                chat_id, f"{body}\n{telegram_quiz.hint(lang)}",
                telegram_quiz.quiz_keyboard(token, labels), parse_mode="",
            )
            telegram_quiz.remember_quiz(api, token, {
                "task_id": task_id, "quiz_id": quiz_id, "chat_id": chat_id,
                "message_id": int(message_id or 0), "options": labels, "text": body,
            })
        except Exception as exc:
            api.log("error", f"Telegram quiz error: {exc}")
    return handle


def register(api):
    api.register_supervised_task("poller", _make_poller(api), restart_policy="on_failure", max_restarts=10)
    api.register_supervised_task("notifier", _make_notifier(api, trust_env=_HONOR_ENV_PROXIES), restart_policy="on_failure", max_restarts=10)
    api.subscribe_event("chat.outbound", _make_outbound(api))
    api.subscribe_event("chat.typing", _make_typing(api))
    api.subscribe_event("chat.photo", _make_photo(api))
    api.subscribe_event("chat.video", _make_video(api))
    api.subscribe_event("chat.document", _make_document(api))
    api.subscribe_event("chat.links", _make_links(api))
    api.subscribe_event("chat.quiz", _make_quiz(api))
    api.register_route("settings/save", handler=_make_settings_save(api), methods=("POST",))
    api.register_route("miniapp/status", handler=_make_status(api), methods=("POST",))
    api.register_settings_section(
        "telegram",
        title="Telegram",
        schema={
            "components": [
                {
                    "type": "markdown",
                    "text": (
                        "Set TELEGRAM_BOT_TOKEN in Settings → Secrets, grant it to this skill, then configure the options below.\n\n"
                        "**Command mode**: controls which slash commands can be sent from Telegram. "
                        "Use `/menu` in Telegram to see available commands as inline buttons.\n\n"
                        "**Mirror mode**: *all* mirrors every chat message (including web UI) to Telegram — requires Chat ID. "
                        "*Telegram only* mirrors only Telegram-originated conversations.\n\n"
                        "**Mini App Beta** uses a best-effort Cloudflare Quick Tunnel with no SLA and no SSE support. "
                        "It targets native Telegram clients; Telegram WebA/WebK are not supported."
                    ),
                },
                {
                    "type": "form",
                    "route": "settings/save",
                    "method": "POST",
                    "fields": [
                        {"name": "TELEGRAM_LANGUAGE", "label": "Language / Язык", "type": "select",
                         "options": [
                             {"value": "en", "label": "🇬🇧 English"},
                             {"value": "ru", "label": "🇷🇺 Русский"},
                         ],
                         "placeholder": "en"},
                        {"name": "TELEGRAM_COMMAND_MODE", "label": "Command mode", "type": "select",
                         "options": [
                             {"value": "full_access", "label": "Full access (default) — raw owner commands incl. /panic, /restart"},
                             {"value": "safe_commands", "label": "Safe — allow /status, /bg status only"},
                             {"value": "strict", "label": "Strict — block all slash commands from Telegram"},
                         ],
                         "placeholder": "full_access"},
                        {"name": "TELEGRAM_MIRROR_MODE", "label": "Mirror mode", "type": "select",
                         "options": [
                             {"value": "all", "label": "Mirror all messages (web + Telegram)"},
                             {"value": "telegram_only", "label": "Telegram conversations only"},
                         ],
                         "placeholder": "all"},
                        {"name": "TELEGRAM_CHAT_ID", "label": "Telegram Chat ID", "type": "text", "placeholder": "required for 'all' mode"},
                        {"name": "TELEGRAM_MAX_UPDATES_PER_POLL", "label": "Max updates per poll", "type": "number", "placeholder": "20"},
                        {"name": "TELEGRAM_SILENT_MODE", "label": "Silent mode (edit-in-place)", "type": "select",
                         "options": [
                             {"value": "off", "label": "Off — each thought is a new message"},
                             {"value": "on", "label": "On — replace the previous thought in-place"},
                         ],
                         "placeholder": "off"},
                        {"name": "TELEGRAM_SUBAGENT_CARDS", "label": "Subagent cards", "type": "select",
                         "options": [
                             {"value": "on", "label": "On — one updating message per subagent"},
                             {"value": "off", "label": "Off — hide subagent activity"},
                         ],
                         "placeholder": "on"},
                        {"name": "TELEGRAM_MIRROR_PROGRESS", "label": "Mirror progress telemetry", "type": "select",
                         "options": [
                             {"value": "off", "label": "Off (default) — replies only (clean chat)"},
                             {"value": "on", "label": "On — stream the main agent's progress"},
                         ],
                         "placeholder": "off"},
                        {"name": "TELEGRAM_MINIAPP_ENABLED", "label": "Telegram Mini App", "type": "select",
                         "options": [
                             {"value": "on", "label": "On (default after owner binding)"},
                             {"value": "off", "label": "Off — keep text bridge only"},
                         ],
                         "placeholder": "on"},
                        {"name": "TELEGRAM_NOTIFY_TASKS", "label": "Notify on task completion", "type": "select",
                         "options": [
                             {"value": "off", "label": "Off"},
                             {"value": "on", "label": "On — ✅ Task done · cost · rounds"},
                         ],
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
                {
                    "type": "action",
                    "id": "miniapp_status",
                    "route": "miniapp/status",
                    "method": "POST",
                    "submit_label": "Refresh Mini App status",
                    "busy_label": "Checking...",
                    "fields": [],
                },
            ]
        },
    )
    # Unsupported OS/architecture degrades inside register_miniapp(). Invalid
    # host runtime or unsafe/unwritable state must fail the complete load rather
    # than leave a partially registered skill claiming to be healthy.
    register_miniapp(api)
