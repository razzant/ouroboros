"""What the Telegram bridge is configured with and what actually runs: one status for Settings › Telegram.

The bridge's state is the poller's record for the exact token generation in force (lib/telegram_token.py), so a
saved token is configuration, never "ready" or "refused", until the poller tried that token in this load; the
saved file is the record's copy and proves nothing alone. The Mini App's status is its companion's own heartbeat.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, Dict

from starlette.responses import JSONResponse

from ..scripts.telegram_settings import TelegramSettingsError
from .miniapp_registration import _read_status
from .telegram_state import _load_settings, _state_file
from .telegram_token import accepted_token, binding_for

_COMMAND_MODE_STRICT = "strict"
_COMMAND_MODE_SAFE = "safe_commands"
_COMMAND_MODE_FULL = "full_access"
_VALID_COMMAND_MODES = frozenset({_COMMAND_MODE_STRICT, _COMMAND_MODE_SAFE, _COMMAND_MODE_FULL})


def _bridge_status(api) -> dict[str, Any]:
    """The text bridge as it is: the accepted token's state, and ready only for the token the poller validated."""
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
    binding = binding_for(api)
    if binding.state == "pending":
        binding.read_now(api)
    token = binding.snapshot()[0]
    token_state = binding.state if binding.state in {"configured", "missing", "unknown"} else "unknown"
    command_mode = str(settings.get("TELEGRAM_COMMAND_MODE") or _COMMAND_MODE_FULL).strip().lower()
    if command_mode not in _VALID_COMMAND_MODES:
        command_mode = _COMMAND_MODE_STRICT
    mirror_mode = str(settings.get("TELEGRAM_MIRROR_MODE") or "all").strip().lower()
    if mirror_mode not in {"all", "telegram_only"}:
        mirror_mode = "all"
    reason_code = ""
    # The poller's record for this exact generation: a replaced token's refusal or wait never speaks for this one.
    recorded, recorded_reason = binding.noted()
    if not token:
        state, poller = ("missing_token" if token_state == "missing" else "settings_unknown"), "waiting_token"
    elif recorded == "error":
        state, poller = "error", "failed"
        reason_code = (recorded_reason or "telegram_rejected")[:64]
    elif recorded == "degraded":
        state, poller = "degraded", "degraded"
        reason_code = (recorded_reason or "telegram_startup_deferred")[:64]
    elif recorded == "ready" and binding.is_validated():  # this exact binding's getMe, not a saved record
        state, poller = ("ready" if owner_bound else "waiting_owner"), "polling"
    else:
        state, poller = "connecting", "connecting"
    result = {
        "state": state,
        "token": token_state,
        "owner_bound": owner_bound,
        "poller": poller,
        "command_mode": command_mode,
        "mirror_mode": mirror_mode,
    }
    if reason_code:
        result["reason_code"] = reason_code
    return result


def _save_bridge_status(api, state: str, reason_code: str = "", bot: str = "", generation: int | None = None) -> None:
    """Record the bridge state for the client ``generation`` that met it (default: the token in force now). A record
    for a generation no longer in force is dropped: a late answer of a replaced token never becomes current."""
    binding = binding_for(api)
    if generation is None:
        accepted_token(api)  # the token in force now, read once if nothing was read yet, as for a first client
        generation = binding.snapshot()[1]
    if not binding.note(generation, state, reason_code):
        return
    path = _state_file(api, "bridge_status.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    try:
        tmp.write_text(
            json.dumps({"state": state, "reason_code": reason_code, **({"bot": bot} if bot else {})}),
            encoding="utf-8",
        )
        tmp.replace(path)
    except OSError:
        api.log("warning", "Telegram bridge status could not be persisted.")


def poll_failure_reason(exc: BaseException, revalidating: bool) -> str:
    """The degraded reason a retried poll failure records: a webhook or another poller holds the token (409),
    a validation still waits for the network, or Telegram stopped answering a working bridge."""
    if getattr(exc, "status_code", None) == 409:
        return "telegram_conflict"
    return "telegram_startup_deferred" if revalidating else "telegram_unreachable"


def _fail_permanent_poll_rejection(api, exc, revalidating: bool, bot: str, generation: int | None = None) -> None:
    """Persist bridge status + log for a permanent polling rejection of the token ``generation``."""
    if revalidating:
        # A permanent getMe rejection during in-loop revalidation
        # is the startup token-validation failure, merely deferred
        # by a dead network at startup — label it the same way.
        _save_bridge_status(api, "error", "telegram_startup_failed", bot=bot, generation=generation)
        api.log("error", f"Telegram token validation failed: {exc}")
    else:
        _save_bridge_status(api, "error", "telegram_rejected", bot=bot, generation=generation)
        api.log("error", "Telegram polling was permanently rejected.")


def _make_status(api):
    async def status(_request):
        return JSONResponse({
            "bridge": _bridge_status(api),
            "mini_app": _read_status(api),
        })

    return status


_TOKEN_WORDS = {
    "configured": "Saved",
    "missing": "Not set (Settings → Secrets)",
    "unknown": "Can't read saved settings now",
}
_KEPT_TOKEN = "; the current token stays in use"
_BRIDGE_WORDS = {
    "ready": "Connected",
    "waiting_owner": "Connected; send the bot a private message to bind its owner",
    "connecting": "Connecting with the saved token…",
    "missing_token": "Waiting for a bot token",
    "settings_unknown": "Waiting: saved settings can't be read",
}
# What stopped or slowed the bridge, by the reason it recorded, in the owner's words: a refused token, a
# setting below, the network and an internal failure each name their own next step. Never the raw code.
_TOKEN_REFUSED = "Stopped: Telegram refused this token; save a new one in Secrets"
_REASON_WORDS = {
    "telegram_rejected": _TOKEN_REFUSED,
    "telegram_startup_failed": _TOKEN_REFUSED,
    "settings_invalid": "Stopped: fix the Telegram settings below and save",
    "connection_settings_invalid": "Stopped: the proxy or Bot API server below is invalid",
    "telegram_conflict": "Waiting: a webhook or another copy uses this token",
    "telegram_startup_deferred": "Can't reach Telegram yet; retrying",
    "telegram_unreachable": "Can't reach Telegram now; retrying",
}
_STATE_FALLBACK = {
    "degraded": "Can't reach Telegram yet; retrying",
    "error": "Stopped by an internal error; see the skill's log",
}
# Each state the companion publishes (scripts/runtime_status.py) or its reader infers, in the owner's words.
_MINIAPP_WORDS = {
    "ready": "Ready",
    "disabled": "Off; the text bridge stays on",
    "waiting_token": "Waiting for a bot token",
    "waiting_owner": "Waiting: send the bot a private message to bind its owner",
    "starting": "Starting",
    "verifying": "Checking its public link",
    "syncing_menu": "Setting the bot's menu button",
    "reconnecting": "Reconnecting; it retries by itself",
    "stopping": "Stopping",
    "stale": "Not running: no recent heartbeat",
    "unavailable": "Unavailable on this computer",
}
# States whose cause or next step only the companion's own bounded sentence names: shown after the lead.
_MINIAPP_LEADS = {
    "degraded": "Retrying",
    "blocked": "Stopped",
    "reconciling": "Restoring the previous owner's menu button",
    "error": "Stopped",
    "stopped": "Stopped",
    "rollback_pending": "Stopped",
}


def connection_components() -> list[dict[str, Any]]:
    """The Connection form of Settings › Telegram: read-only fields ``_connection_fields`` fills."""
    return [
        {"type": "markdown", "text": "Connection: the saved token and what actually runs now."},
        {
            "type": "form",
            "id": "connection",
            "route": "connection",
            "method": "POST",
            "fields": [
                # Read-only and wrapping: a state with its next step stays whole on a phone.
                {"name": "token_status", "label": "Bot token", "type": "textarea", "disabled": True},
                {"name": "bridge_status", "label": "Text bridge", "type": "textarea", "disabled": True},
                {"name": "miniapp_status", "label": "Mini App", "type": "textarea", "disabled": True},
            ],
            "submit_label": "Refresh status",
            "busy_label": "Checking…",
        },
    ]


def _connection_fields(api) -> Dict[str, Any]:
    """What is configured and what actually runs, as the Settings › Telegram Connection form shows it."""
    bridge = _bridge_status(api)
    mini_app = _read_status(api)
    binding = binding_for(api)
    if bridge["state"] in _STATE_FALLBACK:
        bridge_words = _REASON_WORDS.get(str(bridge.get("reason_code") or ""), _STATE_FALLBACK[bridge["state"]])
    else:
        bridge_words = _BRIDGE_WORDS.get(bridge["state"], bridge["state"])
    mini_state, sentence = str(mini_app.get("state") or ""), str(mini_app.get("message") or "").strip()
    lead = _MINIAPP_LEADS.get(mini_state, "")
    # Never the raw state: a state this module does not know shows the companion's sentence alone.
    mini_words = _MINIAPP_WORDS.get(mini_state) or (f"{lead}: {sentence}" if lead and sentence else lead
                                                     or sentence or "Unknown")
    if binding.companion in {"retained", "stop_unconfirmed"}:
        mini_words = "Applying the token once the old Mini App has stopped"
    elif binding.companion == "start_failed":
        mini_words = "Couldn't start with the saved token; retrying"
    token_words = _TOKEN_WORDS.get(bridge.get("token", ""), "Unknown")
    if bridge.get("token") == "unknown" and binding.snapshot()[0]:
        token_words += _KEPT_TOKEN  # only when a token was accepted before: a cold start has none to keep
    fields = {
        "token_status": token_words,
        "bridge_status": bridge_words,
        "miniapp_status": mini_words,
    }
    # The three fields are the status; the form shows them in place, so the result line only confirms the check.
    return {**fields, "message": "Status checked.", "bridge": bridge, "mini_app": mini_app}


def _make_connection(api):
    async def connection(_request):
        # Status reads files (and once the settings): off the event loop.
        return JSONResponse(await asyncio.to_thread(_connection_fields, api))

    return connection
