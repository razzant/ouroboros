"""Interactive Telegram control panels, localized /mode texts, and declarative Settings schema."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from . import telegram_i18n
from .telegram_api import _LOCALIZED_TEXTS
from .telegram_connection import _COMMAND_MODE_FULL, _COMMAND_MODE_STRICT, connection_components
from .telegram_i18n import Index
from .telegram_mirror import (
    MIRROR_MODE_ALL,
    MIRROR_MODE_CONTROL_ONLY,
    MIRROR_MODE_MAIN,
    MIRROR_MODE_TELEGRAM_ONLY,
    OWNER_CHAT_MODE_DIRECT,
    OWNER_CHAT_MODE_PRESENCE,
    mirror_mode,
    mirror_quizzes_enabled,
    owner_chat_mode,
)
from .telegram_state import _is_silent_mode_enabled, _load_settings

_MODE_TEXTS_EN = {
    "btn_mode": "🧭 Mode & Mirror",
    "cmd_mode": "Owner chat & mirror mode",
    "cmd_main": "Send directly to Main chat",
    "mode_title": (
        "🧭 **Telegram Routing & Mirror Mode**\n\n"
        "• **Owner DM chat:** `{owner_mode}`\n"
        "• **Mirror stream:** `{mirror_mode}`\n"
        "• **Mirror questions (`main`/`control_only`):** `{quizzes}`"
        "{readiness}"
        "\n\nIn `presence` mode, ordinary DM messages go to Presence (Circles / Focus), "
        "while `/menu`, `/mode`, `/main <text>`, `/status`, `/bg`, `/panic`, quizzes and the Mini App keep working."
    ),
    "mode_readiness_ready": "\n• **DM Place:** Active (Background Consciousness on)",
    "mode_readiness_bg_off": "\n⚠️ **Warning:** Background Consciousness is **sleeping**. Run `/bg start` or enable it in the web UI so Presence can respond.",
    "mode_readiness_not_connected": "\n⚠️ **Warning:** This owner DM is not connected as a Place yet. Open **Circles → Add a place → Telegram** (or use `/main <text>` / switch back to `direct`).",
    "mode_readiness_held": "\n⚠️ **Warning:** This owner DM Place is **{state}**. Set it to **Active** in Circles for Presence replies to be sent.",
    "mode_readiness_unavailable": "\n• **DM Place:** status unavailable right now",
    "btn_owner_direct": "💬 Direct → Main",
    "btn_owner_presence": "🌐 Presence (Circles)",
    "btn_mirror_all": "All",
    "btn_mirror_main": "Main",
    "btn_mirror_tg": "Telegram only",
    "btn_mirror_control": "Control only",
    "btn_preset_presence": "✨ Preset: Presence + Control",
    "btn_preset_direct": "⚡ Preset: Direct + Main",
    "btn_quizzes_on": "❓ Quizzes: ON",
    "btn_quizzes_off": "❓ Quizzes: OFF",
    "mode_updated_direct": "✅ Owner DM switched to direct (Main chat).",
    "mode_updated_presence": "✅ Owner DM switched to presence (Circles). Commands and /main still go straight to Main.",
    "mirror_updated": "✅ Mirror mode set to {mode}.",
    "quizzes_updated_on": "✅ Question cards will mirror in main / control_only modes.",
    "quizzes_updated_off": "✅ Non-Telegram question cards muted in main / control_only modes.",
    "preset_presence_applied": "✅ Applied: Owner DM = presence, Mirror = control_only.",
    "preset_direct_applied": "✅ Applied: Owner DM = direct, Mirror = main.",
    "mode_usage": (
        "Usage:\n"
        "• `/mode` — open mode panel\n"
        "• `/mode direct` | `/mode presence`\n"
        "• `/mode mirror all|main|telegram_only|control_only`\n"
        "• `/mode quizzes on|off`\n"
        "• `/main <message>` — send one message straight to Main chat"
    ),
    "main_empty": "Send a message after `/main`, for example: `/main check build status`.",
    "presence_hint_bg_off": (
        "💤 Saved to Presence custody, but Background Consciousness is off — Presence won't wake until you "
        "run `/bg start`, use `/main <text>`, or switch `/mode direct`."
    ),
    "presence_hint_not_connected": (
        "🧭 Saved to Presence custody, but this chat isn't connected as a Telegram Place yet. "
        "Connect it in Circles → Add a place, or use `/main <text>` / `/mode direct`."
    ),
    "presence_hint_held": (
        "🔇 Saved to Presence custody, but this Place is currently {state}. "
        "Set it to Active in Circles to let Presence reply, or use `/main <text>` / `/mode direct`."
    ),
}
_MODE_TEXTS = Index("mode", _MODE_TEXTS_EN, "the Telegram bot's mode switcher: owner chat mode, mirror modes, presets and readiness hints")


def _mark(active: bool, label: str) -> str:
    return f"• {label}" if active else label


def _build_menu_keyboard(command_mode: str, lang: str = "en") -> Tuple[str, List[List[dict]]]:
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
        [{"text": t["btn_tasks"], "callback_data": "nav:tasks"}],
        [{"text": t["btn_settings"], "callback_data": "nav:settings"}],
    ]
    return header, keyboard


def _build_menu_status(command_mode: str, lang: str = "en", info_text: str = "") -> Tuple[str, List[List[dict]]]:
    """Return status header and keyboard with Refresh and Back button."""
    t = _LOCALIZED_TEXTS[lang]
    header = t.format("metrics_title", info_text=info_text)
    keyboard = [
        [{"text": t["btn_refresh"], "callback_data": "cmd_act:update_status"}],
        [{"text": t["btn_back"], "callback_data": "nav:menu"}],
    ]
    return header, keyboard


def _build_menu_mind(command_mode: str, lang: str = "en", bg_enabled: bool = False) -> Tuple[str, List[List[dict]]]:
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
    """Localized '/' menu: menu first, owner Panic in every mode; extra controls in full_access."""
    t = _LOCALIZED_TEXTS[lang]
    m = _MODE_TEXTS[lang]
    cmds = [
        {"command": "menu", "description": t["cmd_menu"]},
        {"command": "mode", "description": m["cmd_mode"]},
        {"command": "main", "description": m["cmd_main"]},
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


def _build_menu_settings(api, command_mode: str, lang: str = "en") -> Tuple[str, List[List[dict]]]:
    """Return (header_text, inline_keyboard_rows) for the Settings panel."""
    t = _LOCALIZED_TEXTS[lang]
    m = _MODE_TEXTS[lang]
    header = t["settings_title"]
    silent_on = _is_silent_mode_enabled(_load_settings(api))
    silent_label = t["btn_silent_on"] if silent_on else t["btn_silent_off"]
    keyboard = [
        [{"text": m["btn_mode"], "callback_data": "nav:mode"}],
        [{"text": t["btn_language"], "callback_data": "nav:language"}],
        [{"text": silent_label, "callback_data": "cmd_act:toggle_silent"}],
        [{"text": t["btn_back"], "callback_data": "nav:menu"}],
    ]
    return header, keyboard


def _build_language_keyboard(lang: str = "") -> Tuple[str, List[List[dict]]]:
    """(header_text, inline_keyboard_rows) for the install's interface language."""
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


def build_mode_panel(
    settings: Dict[str, Any],
    lang: str = "en",
    *,
    place_state: str = "",
    bg_enabled: Optional[bool] = None,
) -> Tuple[str, List[List[dict]]]:
    """Build the interactive /mode panel showing owner DM lane, mirror mode, quizzes and presets."""
    t = _LOCALIZED_TEXTS[lang]
    m = _MODE_TEXTS[lang]
    cur_owner = owner_chat_mode(settings)
    cur_mirror = mirror_mode(settings)
    cur_quizzes = mirror_quizzes_enabled(settings)
    readiness = ""
    if cur_owner == OWNER_CHAT_MODE_PRESENCE:
        if bg_enabled is False:
            readiness = m["mode_readiness_bg_off"]
        elif place_state == "not_connected":
            readiness = m["mode_readiness_not_connected"]
        elif place_state in {"active", "legacy"}:
            readiness = m["mode_readiness_ready"]
        elif place_state and place_state != "unavailable":
            readiness = m.format("mode_readiness_held", state=place_state)
        else:
            readiness = m["mode_readiness_unavailable"]
    header = m.format(
        "mode_title",
        owner_mode=cur_owner,
        mirror_mode=cur_mirror,
        quizzes="on" if cur_quizzes else "off",
        readiness=readiness,
    )
    keyboard = [
        [
            {"text": _mark(cur_owner == OWNER_CHAT_MODE_DIRECT, m["btn_owner_direct"]), "callback_data": "set_mode:direct"},
            {"text": _mark(cur_owner == OWNER_CHAT_MODE_PRESENCE, m["btn_owner_presence"]), "callback_data": "set_mode:presence"},
        ],
        [
            {"text": _mark(cur_mirror == MIRROR_MODE_ALL, m["btn_mirror_all"]), "callback_data": "set_mirror:all"},
            {"text": _mark(cur_mirror == MIRROR_MODE_MAIN, m["btn_mirror_main"]), "callback_data": "set_mirror:main"},
        ],
        [
            {"text": _mark(cur_mirror == MIRROR_MODE_TELEGRAM_ONLY, m["btn_mirror_tg"]), "callback_data": "set_mirror:telegram_only"},
            {"text": _mark(cur_mirror == MIRROR_MODE_CONTROL_ONLY, m["btn_mirror_control"]), "callback_data": "set_mirror:control_only"},
        ],
        [
            {"text": m["btn_quizzes_on"] if cur_quizzes else m["btn_quizzes_off"], "callback_data": "cmd_act:toggle_quizzes"},
        ],
        [
            {"text": m["btn_preset_presence"], "callback_data": "set_preset:presence_control"},
            {"text": m["btn_preset_direct"], "callback_data": "set_preset:direct_main"},
        ],
        [{"text": t["btn_back"], "callback_data": "nav:settings"}],
    ]
    return header, keyboard


def presence_readiness_hint(lang: str, *, place_state: str = "", bg_enabled: Optional[bool] = None) -> str:
    """Actionable one-line hint when an owner DM message enters Presence while it cannot reply yet."""
    m = _MODE_TEXTS[lang]
    if bg_enabled is False:
        return m["presence_hint_bg_off"]
    if place_state == "not_connected":
        return m["presence_hint_not_connected"]
    if place_state and place_state not in {"active", "legacy", "unavailable"}:
        return m.format("presence_hint_held", state=place_state)
    return ""


def settings_section_schema() -> Dict[str, Any]:
    """Declarative Telegram settings schema combining bridge, Mini App, mode/mirror and Presence fields."""
    return {
        "components": [
            {
                "type": "markdown",
                "text": (
                    "The bot token lives in Settings → Accounts and keys (TELEGRAM_BOT_TOKEN). While this skill is on, a saved, "
                    "replaced or cleared token reaches the text bridge, Presence transport and the Mini App within seconds; "
                    "saving it grants, turns on or resumes nothing; Connection below shows what actually runs. "
                    "If Telegram needs a proxy, configure Telegram proxy below. It applies to all Telegram API calls, "
                    "including Presence and the menu button; the Mini App tunnel and web traffic keep their existing routes.\n\n"
                    "**Owner DM chat mode**: *direct* sends ordinary owner DM messages straight to Main chat. "
                    "*presence* routes ordinary owner DM messages into Presence (Circles / Focus) while keeping `/menu`, "
                    "`/mode`, `/main <text>`, `/status`, `/bg`, `/panic`, quizzes and the Mini App live on the owner control lane.\n\n"
                    "**Command mode**: controls which slash commands can be sent from Telegram. "
                    "Use `/menu` or `/mode` in Telegram to inspect or switch settings via inline buttons.\n\n"
                    "**Mirror mode**: *all* mirrors every chat message (including web UI) to Telegram — requires Chat ID. "
                    "*main* mirrors Main chat and owner/Presence notices without raw Project chatter. "
                    "*Telegram only* mirrors Telegram-originated conversations, plus Ouroboros's reminders, skill notices and Safety notices "
                    "when a Chat ID is set. "
                    "*Control only* mirrors Telegram command/chat replies, owner/Presence notices and optional question cards. "
                    "Presence rooms are always delivered by the Presence outbox and never double-mirrored.\n\n"
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
                    {"name": "TELEGRAM_OWNER_CHAT_MODE", "label": "Owner DM chat mode", "type": "select",
                     "options": [
                         {"value": "direct", "label": "Direct — ordinary DM messages go to Main chat (default)"},
                         {"value": "presence", "label": "Presence — ordinary DM messages go to Circles; commands stay live"},
                     ],
                     "help": "Controls ordinary owner DM messages only. /menu, /mode, /main <text>, /panic, slash commands, "
                             "quizzes and the Mini App always work.",
                     "placeholder": "direct"},
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
                         {"value": "main", "label": "Main chat + notices (no Project chatter)"},
                         {"value": "telegram_only", "label": "Telegram conversations, plus reminders, skill notices and Safety notices"},
                         {"value": "control_only", "label": "Control only (command replies, owner notices, optional questions)"},
                     ],
                     "placeholder": "all"},
                    {"name": "TELEGRAM_MIRROR_QUIZZES", "label": "Mirror question cards (main / control_only)", "type": "select",
                     "options": [
                         {"value": "on", "label": "On (default)"},
                         {"value": "off", "label": "Off — only Telegram-originated questions"},
                     ],
                     "placeholder": "on"},
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
                    {"name": "words", "label": "Presence words to notice (one per line)", "type": "textarea",
                     "help": "Optional wake signal words for Circles places (`signal.words`). Applies to new messages without restart."},
                    {"name": "management_group_id", "label": "Presence management group ID", "type": "text",
                     "placeholder": "Optional numeric Telegram group ID (e.g. -100...)",
                     "help": "Marks this group in Presence event facts. Grants no owner permissions."},
                ],
                "submit_label": "Save Telegram settings",
            },
        ]
    }
