"""What the Telegram bridge is configured with and what actually runs: one status for Settings › Telegram.

The bridge's state is the poller's record for the exact token generation in force (lib/telegram_token.py), so a
saved token is configuration, never "ready" or "refused", until the poller tried that token in this load; the
saved file is the record's copy and proves nothing alone. The Mini App's status is its companion's own heartbeat;
delivery is the outbound journal's own account (lib/telegram_outbox.py): what waits and how many recent deliveries
were not confirmed. The second form lists each of those by its time, thing, room, parts, outcome and where its
content is, beside the only copies kept of them. Names and labels are shown on one line with their control
characters escaped, and are never read back: what a press deletes is its own read-only field of handles the journal
made (``delete_ids``), and only once the owner ticks the box. A deletion's answer leaves that field empty, so a box
still ticked deletes nothing until the page is opened again; a deleted copy's outcome stays.
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Any, Dict, Optional

from starlette.responses import JSONResponse

from ..scripts.telegram_settings import TelegramSettingsError
from .miniapp_registration import _read_status
from .telegram_outbox import outbox_for
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
# Waits whose cause the companion names by its reason; the time it planned the next attempt for follows if set.
_MINIAPP_REASONS = {
    "cloudflare_rate_limited": "Waiting: Cloudflare limits new links from this network",
    "public_marker_pending": "Checking its new public link while Cloudflare registers it",
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
                {"name": "delivery_status", "label": "Delivery", "type": "textarea", "disabled": True},
            ],
            "submit_label": "Refresh status",
            "busy_label": "Checking…",
        },
        {
            # Deliveries not confirmed beside the copies kept of them: two lists side by side on a wide screen, and
            # the deletion right under them; a phone stacks them.
            "type": "form",
            "id": "kept_copies",
            "route": "connection/copies",
            "method": "POST",
            "fields": [
                {"name": "not_delivered", "label": "Delivery issues", "type": "textarea", "disabled": True},
                {"name": "kept_copies", "label": "Kept copies", "type": "textarea", "disabled": True,
                 "help": "Never sent again; they stay until you delete them. Ouroboros saves one to your files when "
                         "you ask with its id."},
                # The tick shares its column with Delete copies on a wide screen; on a phone it comes before the
                # ids it confirms, then the button.
                {"name": "confirm_delete", "label": "Delete the kept copies these ids name", "type": "checkbox"},
                {"name": "delete_ids", "label": "Ids of the kept copies listed", "type": "textarea", "disabled": True,
                 "help": "A copy kept later is not among them. Empty after a deletion until you open this page again."},
            ],
            "submit_label": "Delete copies",
            "busy_label": "Deleting…",
        },
    ]


# Why the oldest waiting message waits, by the cause its operation recorded; any other cause is the network's.
_WAIT_WORDS = {"HTTP 429": "Telegram asked to slow down", "token_changed": "switching to the saved bot token"}
_KINDS = {"reply": "Reply", "notice": "Notice", "completion": "Completion notice", "salvaged result": "Salvaged result",
          "saved work": "Saved work", "progress": "Progress note", "routing notice": "Routing notice",
          "subagent card": "Subagent card", "photo": "Photo", "video": "Video", "file": "File", "links": "Links",
          "question": "Question", "question update": "Question update"}
_STATE_WORDS = {"unknown": "May not have arrived (no clear answer from Telegram; not sent again)",
                "refused": "Refused by Telegram", "superseded": "Not sent: no longer mirrored"}
_HANDLE = re.compile(r"\d{12}-[0-9a-f]{8}|stray-[0-9a-f]{16}")  # what the journal names: never a name or label
_OUTCOME_REASONS = {"expired": "Not sent within a day", "token_changed": "Stopped: the bot token changed",
                    "changed": "Stopped: it changed before its retry", "full": "Not sent: the outbox was full",
                    "no handler": "Stopped: nothing here can send it now"}
# Where a kept copy's content is, by the class its end found (lib/telegram_copies.py); any other kept file is one the
# outbox could not remove.
_KEPT_WORDS = {"only": "Only copy kept on this computer (Kept copies).",
               "reference": "Its message and the file's reference are kept on this computer, not the file (Kept copies)."}


def _when(at: float, now: float) -> str:
    """This computer's clock time, with the day when it is not today."""
    if not at:
        return "Time unknown"
    local = time.localtime(at)
    clock = time.strftime("%H:%M", local)
    return clock if local[:3] == time.localtime(now)[:3] else f"{local.tm_mday} {time.strftime('%b', local)} {clock}"


def _size(size: Any) -> str:
    size = int(size or 0)
    return f"{size / 1048576:.1f} MB" if size >= 1048576 else f"{max(1, round(size / 1024))} KB"


def _shown(text: Any) -> str:
    """A name or label as one visible line: a line break, other control, format or separator character (a bidi
    override among them) shows as its escape, so a name never adds, ends or reorders a line."""
    return "".join(ch if ch == " " or ch.isprintable() else f"\\u{ord(ch):04x}" for ch in str(text or ""))


def _thing(subject: Dict[str, Any], parts: int = 1) -> str:
    """The thing and its room: ``Photo, 2.1 MB · Main``; a document by its name, a long reply by its parts."""
    kind = str(subject.get("kind") or "")
    thing = _KINDS.get(kind, "Message")
    if kind == "file":
        thing = f"File {_shown(subject.get('name'))}".strip()
    if kind in {"photo", "video", "file"} and subject.get("size"):
        thing += f", {_size(subject['size'])}"
    if parts > 1:
        thing += f" in {parts} parts"
    room, label = int(subject.get("room") or 0), _shown(subject.get("label"))
    where = "Main" + (f" · {label}" if label else "") if room == 1 else label or ("a Project room" if room else "")
    return f"{thing} · {where}" if where else thing


def _outcome_words(outcome: Dict[str, Any]) -> str:
    """What happened and where its content is now."""
    reason = str(outcome.get("reason") or "")
    said = _STATE_WORDS.get(str(outcome.get("state") or "")) or _OUTCOME_REASONS.get(reason) or (
        "Not sent: the outbox couldn't save it" if reason.startswith("journal") else
        "Stopped: its saved copy can't be read" if reason.startswith("event") else "Stopped before sending")
    if int(outcome.get("parts") or 0) > 1:
        said += f" ({outcome.get('arrived', 0)} of {outcome['parts']} parts arrived)"
    if outcome.get("held"):
        return f"{said}. " + _KEPT_WORDS.get(str(outcome.get("copy") or ""), "A copy stays on this computer (Kept copies).")
    if outcome.get("deleted"):
        return f"{said}. Its kept copy was deleted."
    if outcome.get("home"):
        return f"{said}. Its content is in the app."
    return said + ("." if outcome.get("copy") == "replaceable" else ". No copy was kept.")


def _delivery_words(summary: Dict[str, Any]) -> str:
    """The outbound journal in the owner's words: whether it can save, what waits and why, how many recent
    deliveries were not confirmed (each listed under Delivery issues, with what is known of it)."""
    words = []
    if summary.get("fault"):  # it fails closed: nothing is sent without a saved record
        words.append(f"Degraded: the outbox can't save ({summary['fault']}), so nothing new is sent")
    if not_sent := int(summary.get("not_sent") or 0):
        words.append(f"{not_sent} not sent because the outbox couldn't hold them")
    if not summary.get("loaded"):
        return ". ".join(words + ["Starting"])
    waiting = int(summary.get("waiting") or 0)
    if waiting:
        cause = str(summary.get("cause") or "")
        why = _WAIT_WORDS.get(cause, "can't reach Telegram" if cause else "")
        retry = int(summary.get("retry_in_sec") or 0)
        words.append(f"{waiting} waiting to be sent" + (f" ({why})" if why else "")
                     + (f"; next try in {retry} s" if retry else ""))
    if total := int(summary.get("outcomes_total") or 0):
        words.append(f"{total} recent delivery {'issue' if total == 1 else 'issues'} (see Delivery issues)")
    return ". ".join(words) or "Nothing waits to be sent"


def _outcomes_words(summary: Dict[str, Any], now: float) -> str:
    """Recent deliveries not confirmed, newest first: each its thing over what happened and where its content is."""
    if not summary.get("loaded"):
        return "Starting"
    outcomes = summary.get("outcomes") or []
    # One item per paragraph: wrapped lines start at the field's edge on a phone too.
    items = [f"{_when(outcome['at'], now)} · {_thing(outcome['subject'], outcome['parts'])}\n{_outcome_words(outcome)}"
             for outcome in outcomes]
    if (earlier := int(summary.get("outcomes_total") or 0) - len(outcomes)) > 0:
        items.append(f"{earlier} earlier not listed.")
    return "\n\n".join(items) or "No recent delivery issues."


def _kept_words(summary: Dict[str, Any], now: float) -> str:
    """The kept copies, each its thing over the id Ouroboros and Delete copies know it by. Display only: never read
    back, so a name can neither add nor hide what a press deletes."""
    if not summary.get("loaded"):
        return "Starting"
    items = []
    for item in summary.get("kept") or []:
        if item.get("stray"):  # by its exact file name: a record that cannot be read proves nothing
            what = ("Saved message without its delivery record; it may have been sent" if item["stray"] == "orphan"
                    else "Delivery record that can't be read")
            size = "size unknown" if item.get("size") is None else _size(item["size"])
            items.append(f"{_when(item['at'], now)} · {what}, {size} · file {_shown(item['name'])}\n{item['id']}")
        elif item.get("copy") == "reference":  # its message and the file's reference: no file's bytes, so no size
            name = f"{item['subject'].get('name') or 'file'} (reference only)"
            items.append(f"{_when(item['at'], now)} · {_thing({**item['subject'], 'size': 0, 'name': name})}\n{item['id']}")
        else:
            items.append(f"{_when(item['at'], now)} · {_thing(item['subject'])}\n{item['id']}")
    if (more := int(summary.get("kept_total") or 0) - len(items)) > 0:
        items.append(f"{more} older kept copies are listed once these are deleted.")
    if total := int(summary.get("kept_total") or 0):  # no bound of their own, so the list says what they take
        size, unknown = int(summary.get("kept_bytes") or 0), int(summary.get("kept_unknown") or 0)
        took = (f"{total} kept {'copy takes' if total == 1 else 'copies take'} {_size(size)}" if size or not unknown
                else f"{total} kept {'copy' if total == 1 else 'copies'}")
        unread = "" if not unknown else "; its size can't be read" if total == 1 else f"; the size of {unknown} can't be read"
        items.insert(0, f"{took} on this computer{unread}.")
    return "\n\n".join(items) or "No copies are kept."


def _shown_ids(text: Any) -> Optional[list[str]]:
    """The handles ``delete_ids`` held, one per line; None when a line is not one (the field is the journal's own
    words, so anything else is not what the page showed)."""
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    return lines if all(_HANDLE.fullmatch(line) for line in lines) else None


async def _clear(outbox, ids: list[str]) -> Dict[str, int]:
    """Delete on the outbox's own loop: the journal's records belong to it."""
    loop, here = outbox._loop, asyncio.get_running_loop()
    if loop is None or loop is here:
        return await outbox.clear(ids)
    if not loop.is_running():
        raise RuntimeError("the Telegram outbox is not running")
    return await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(outbox.clear(ids), loop))


def _lists(summary: Dict[str, Any], *, armed: bool) -> Dict[str, str]:
    """Both lists, and the handles a press deletes: exactly the kept copies listed, or none once disarmed."""
    now = time.time()
    shown = "\n".join(str(item["id"]) for item in summary.get("kept") or []) if armed and summary.get("loaded") else ""
    return {"not_delivered": _outcomes_words(summary, now), "kept_copies": _kept_words(summary, now),
            "delete_ids": shown}


def _make_copies(api):
    """GET lists the delivery issues and the kept copies, armed with exactly the handles listed; POST deletes the
    handles of ``delete_ids`` once the owner ticked the box, and answers disarmed: the renderer keeps the tick, so a
    second press finds nothing to delete until the page is opened again."""
    async def copies(request):
        outbox = outbox_for(api)
        if request.method != "POST":
            return JSONResponse({**_lists(outbox.summary(), armed=True), "confirm_delete": False})
        try:
            values = await request.json()
        except ValueError:
            values = None
        if not isinstance(values, dict) or values.get("confirm_delete") is not True:
            return JSONResponse({"error": "Tick “Delete the kept copies these ids name” first; nothing was deleted."},
                                status_code=400)
        ids = _shown_ids(values.get("delete_ids"))
        if ids is None:
            return JSONResponse({"error": "These are not ids this page listed; nothing was deleted."}, status_code=400)
        if not ids:
            return JSONResponse({**_lists(outbox.summary(), armed=False),
                                 "message": "Nothing to delete. Open this page again to delete the copies kept now."})
        try:
            done = await _clear(outbox, ids)
        except RuntimeError as exc:
            return JSONResponse({"error": f"Nothing was deleted: {exc}."}, status_code=503)
        summary = outbox.summary()
        said = [f"Deleted {done['deleted']} kept {'copy' if done['deleted'] == 1 else 'copies'}"]
        if done["failed"]:
            said.append(f"{done['failed']} could not be deleted (see the skill's log)")
        if done["missing"]:
            said.append(f"{done['missing']} no longer kept")
        if remaining := int(summary.get("kept_total") or 0):
            said.append(f"{remaining} other kept {'copy stays' if remaining == 1 else 'copies stay'}; open this page "
                        "again to delete " + ("it" if remaining == 1 else "them"))
        return JSONResponse({**_lists(summary, armed=False), "message": ". ".join(said) + "."})

    return copies


def register_connection(api) -> None:
    """The Connection form's route and its kept-copies form's route."""
    api.register_route("connection", handler=_make_connection(api), methods=("GET", "POST"))
    api.register_route("connection/copies", handler=_make_copies(api), methods=("GET", "POST"))


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
    if cause := _MINIAPP_REASONS.get(str(mini_app.get("reason_code") or "")):
        at = int(mini_app.get("next_retry_at_epoch") or 0)
        mini_words = cause + (f"; next automatic attempt planned for {_when(at, time.time())}" if at else "")
    if binding.companion in {"retained", "stop_unconfirmed"}:
        mini_words = "Applying the token once the old Mini App has stopped"
    elif binding.companion == "start_failed":
        mini_words = "Couldn't start with the saved token; retrying"
    token_words = _TOKEN_WORDS.get(bridge.get("token", ""), "Unknown")
    if bridge.get("token") == "unknown" and binding.snapshot()[0]:
        token_words += _KEPT_TOKEN  # only when a token was accepted before: a cold start has none to keep
    delivery = outbox_for(api).summary()
    fields = {
        "token_status": token_words,
        "bridge_status": bridge_words,
        "miniapp_status": mini_words,
        "delivery_status": _delivery_words(delivery),
    }
    # The fields are the status; the form shows them in place, so the result line only confirms the check.
    return {**fields, "message": "Status checked.", "bridge": bridge, "mini_app": mini_app, "delivery": delivery}


def _make_connection(api):
    async def connection(_request):
        # Status reads files (and once the settings): off the event loop.
        return JSONResponse(await asyncio.to_thread(_connection_fields, api))

    return connection
