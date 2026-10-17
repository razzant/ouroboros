"""zoom_meeting: this install joins Zoom meetings through the Zoom Meeting SDK.

The skill has isolated dependencies, so these tools run in a short-lived child per
call. They check the request, choose the Presence binding that will govern the call,
and hand the join to the long-lived ``zoom_bot`` companion, which owns the SDK. When
meeting audio flows, the companion opens the Presence call through the Host Service:
one long live turn under that binding's behavior and capability ceiling, with the
people in the meeting as correspondents, never as the owner.
"""

from __future__ import annotations

import pathlib
import re
import time
import urllib.parse
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .scripts.companion_link import (
    PROVIDER,
    ControlUnavailable,
    app_account_id,
    control_request,
    read_status,
)

SKILL_NAME = "zoom_meeting"
COMPANION = "zoom_bot"
# How long zoom_join waits for Zoom's first answer (auth, join, waiting room) before
# reporting "joining"; bounded by the tool's own timeout.
JOIN_ANSWER_WAIT_SEC = 20.0
_POLL_SEC = 0.5
_FIELD_LIMIT = 200
_ZOOM_DOMAINS = ("zoom.us", "zoom.com")
_MEETING_PATH = re.compile(r"^/(?:j|w|s|wc/join)/(\d+)(?:/|$)")
_WEB_CLIENT_PATH = re.compile(r"^/wc/(\d+)(?:/|$)")
_SEPARATORS = re.compile(r"[\s\-.]")
_JOINING_STATES = ("joining",)

_sleep = time.sleep
_monotonic = time.monotonic


class ToolRefusal(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code, self.message = code, message

    def render(self) -> str:
        return f"⚠️ {self.code}: {self.message}"


def _bounded(text: Any, limit: int = _FIELD_LIMIT) -> str:
    value = str(text or "")
    return value if len(value) <= limit else f"{value[:limit]} ...[{len(value) - limit} more characters omitted]"


def _checked_number(raw: str) -> str:
    if not raw.isdigit() or not 0 < int(raw) < 2 ** 64:
        raise ToolRefusal("ZOOM_MEETING_INVALID", f"{raw!r} is not a Zoom meeting number")
    return raw


def parse_meeting(meeting: str) -> Tuple[str, str]:
    """``(meeting_number, passcode_from_link)`` from a join link or a meeting ID.

    Parsed by URL structure only: a Zoom host with ``/j/``, ``/w/``, ``/s/`` or
    ``/wc/`` paths, a ``confno`` query, or the ``zoommtg://`` desktop link; the
    passcode comes from the link's ``pwd`` when it carries one.
    """
    raw = str(meeting or "").strip()
    if not raw:
        raise ToolRefusal("ZOOM_MEETING_INVALID", "meeting is empty: pass a Zoom join link or the meeting ID")
    compact = _SEPARATORS.sub("", raw)
    if compact.isdigit():
        return _checked_number(compact), ""
    parts = urllib.parse.urlsplit(raw if "://" in raw else f"https://{raw}")
    scheme, host = parts.scheme.lower(), (parts.hostname or "").lower()
    query = urllib.parse.parse_qs(parts.query)
    number = ""
    if scheme in ("zoommtg", "zoomus"):
        number = (query.get("confno") or [""])[0]
    elif scheme in ("http", "https") and any(host == d or host.endswith("." + d) for d in _ZOOM_DOMAINS):
        if parts.path.startswith("/my/"):
            raise ToolRefusal("ZOOM_MEETING_INVALID", "a personal link name (zoom.us/my/...) carries no "
                                                      "meeting number; ask for the numeric meeting ID")
        match = _MEETING_PATH.match(parts.path) or _WEB_CLIENT_PATH.match(parts.path)
        number = match.group(1) if match else (query.get("confno") or [""])[0]
    elif host == "zoomgov.com" or host.endswith(".zoomgov.com"):
        raise ToolRefusal("ZOOM_MEETING_INVALID", "ZoomGov meetings need the zoomgov.com SDK domain, "
                                                  "which this skill does not use")
    else:
        raise ToolRefusal("ZOOM_MEETING_INVALID", f"not a Zoom join link: {_bounded(raw, 120)!r}")
    if not number:
        raise ToolRefusal("ZOOM_MEETING_INVALID", "the link carries no meeting number")
    return _checked_number(number), (query.get("pwd") or [""])[0]


def requester_facts(ctx: Any) -> Dict[str, str]:
    """Who asked to join, as the Presence event's actor: the Presence correspondent of
    the calling turn when there is one, else the calling task."""
    metadata = dict(getattr(ctx, "task_metadata", None) or {})
    presence = metadata.get("presence") if isinstance(metadata.get("presence"), dict) else {}
    event = presence.get("event") if isinstance(presence.get("event"), dict) else {}
    actor = event.get("actor") if isinstance(event.get("actor"), dict) else {}
    task_id = str(getattr(ctx, "task_id", "") or "")
    if actor:
        actor_id = actor.get("platform_actor_id") or actor.get("id") or ""
        name = actor.get("display_name") or actor.get("name") or actor.get("username") or ""
        return {"id": _bounded(f"{event.get('provider') or 'presence'}:{actor_id}"), "display_name": _bounded(name),
                "requested_via": _bounded(f"presence:{event.get('provider') or ''}"), "task_id": task_id}
    via = metadata.get("source") or getattr(ctx, "current_task_type", "") or "task"
    return {"id": f"task:{task_id}" if task_id else "unknown", "display_name": "",
            "requested_via": _bounded(via), "task_id": task_id}


def select_binding(bindings: Iterable[Any], *, requested: str, account_id: str, meeting_number: str) -> Any:
    """The binding argument, or the one enabled binding whose transport is this skill."""
    ours = [b for b in bindings if b.transport_skill == SKILL_NAME and b.enabled]
    if requested:
        chosen = next((b for b in ours if b.binding_id == requested), None)
        if chosen is None:
            raise ToolRefusal("ZOOM_BINDING_UNKNOWN", f"binding {requested} is not an enabled Presence binding "
                                                      f"whose transport is {SKILL_NAME}")
    elif len(ours) == 1:
        chosen = ours[0]
    elif not ours:
        raise ToolRefusal("ZOOM_BINDING_MISSING", (
            f"no enabled Presence binding uses {SKILL_NAME} as its transport. The owner creates one with "
            f"transport '{PROVIDER}', account '{account_id}' and conversation '*' (any meeting) or this "
            "meeting's number, picks its behavior skill and turns call mode on for it"))
    else:
        listed = ", ".join(f"{b.binding_id} ({b.behavior_skill})" for b in ours)
        raise ToolRefusal("ZOOM_BINDING_REQUIRED", f"several Zoom bindings exist: {listed}; pass binding_id")
    origin = chosen.origin
    if (origin.transport != PROVIDER or origin.account_id != account_id
            or origin.conversation_id not in ("*", meeting_number) or origin.thread_id):
        raise ToolRefusal("ZOOM_BINDING_MISMATCH", (
            f"binding {chosen.binding_id} admits transport '{origin.transport}', account '{origin.account_id}', "
            f"conversation '{origin.conversation_id}'; this join is transport '{PROVIDER}', account "
            f"'{account_id}', meeting {meeting_number}"))
    return chosen


def render_status(status: Dict[str, Any]) -> str:
    """The companion's own sentence first, then the facts it carries."""
    lines = [str(status.get("message") or status.get("state") or "unknown")]
    facts = [f"{label}: {status[key]}" for key, label in (
        ("meeting_number", "meeting"), ("topic", "topic"), ("display_name", "as"), ("binding_id", "binding"),
        ("call_id", "call"), ("since", "since"), ("account_id", "Presence account")) if status.get(key)]
    if facts:
        lines.append(" · ".join(facts))
    audio = status.get("audio") if isinstance(status.get("audio"), dict) else {}
    if audio.get("status") == "FAIL":
        lines.append(f"meeting audio may not arrive here: {audio.get('detail') or 'unknown reason'} "
                     "(SKILL.md, Setup, says how to provide it)")
    outcome = status.get("last_outcome") if isinstance(status.get("last_outcome"), dict) else None
    if outcome and outcome.get("message") and status.get("state") == "idle":
        lines.append(f"last: {outcome['message']} ({outcome.get('at', '')})")
    return "\n".join(lines)


def _companion(state_dir: pathlib.Path, op: str, args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    try:
        return control_request(state_dir, op, args)
    except ControlUnavailable as exc:
        known = read_status(state_dir)
        hint = f" Last published state: {render_status(known)}" if known else ""
        raise ToolRefusal("ZOOM_COMPANION_UNAVAILABLE", (
            f"the {COMPANION} companion is not answering ({exc}). It starts when the skill is enabled with its "
            f"dependencies installed; its health is on the Skills page.{hint}"))


def _require_ready(status: Dict[str, Any], account_id: str) -> None:
    sdk = status.get("sdk") if isinstance(status.get("sdk"), dict) else {}
    if not sdk.get("available"):
        raise ToolRefusal("ZOOM_SDK_UNAVAILABLE", f"this host cannot run the Zoom Meeting SDK: "
                                                  f"{sdk.get('detail') or 'unknown reason'}")
    if not status.get("credentials") or status.get("account_id") != account_id:
        raise ToolRefusal("ZOOM_COMPANION_STALE", (
            f"the {COMPANION} companion started without the current Zoom app credentials (they were saved or "
            "granted after it started); turn the zoom_meeting skill off and on so it starts with them"))
    if status.get("state") != "idle":
        raise ToolRefusal("ZOOM_BUSY", f"already busy: {render_status(status)}\nCall zoom_leave first.")


def _await_first_answer(state_dir: pathlib.Path, join_id: str) -> Dict[str, Any]:
    deadline = _monotonic() + JOIN_ANSWER_WAIT_SEC
    status: Dict[str, Any] = {}
    while True:
        status = _companion(state_dir, "status")
        outcome = status.get("last_outcome") if isinstance(status.get("last_outcome"), dict) else {}
        if join_id and status.get("state") == "idle" and outcome.get("join_id") == join_id:
            raise ToolRefusal("ZOOM_JOIN_FAILED", str(outcome.get("message") or "the join ended"))
        if status.get("state") not in _JOINING_STATES or _monotonic() >= deadline:
            return status
        _sleep(_POLL_SEC)


def _data_root(api: Any, ctx: Any) -> pathlib.Path:
    info = api.get_runtime_info() if hasattr(api, "get_runtime_info") else {}
    root = str((info or {}).get("data_dir") or getattr(ctx, "drive_root", "") or "")
    if not root:
        raise ToolRefusal("ZOOM_BINDINGS_UNREADABLE", "the install's data directory is unknown here, so its "
                                                      "Presence bindings cannot be read")
    return pathlib.Path(root)


def _bindings(api: Any, ctx: Any) -> List[Any]:
    from ouroboros.presence_bindings import PresenceBindingError, list_presence_bindings

    try:
        return list(list_presence_bindings(_data_root(api, ctx)))
    except PresenceBindingError as exc:
        raise ToolRefusal("ZOOM_BINDINGS_UNREADABLE", f"the Presence bindings could not be read ({exc.code})")


def register(api: Any) -> None:
    api.register_companion_process(COMPANION)
    state_dir = pathlib.Path(api.get_state_dir())

    def zoom_join(ctx: Any, meeting: str = "", passcode: str = "", display_name: str = "",
                  binding_id: str = "", on_behalf_token: str = "") -> str:
        try:
            number, link_passcode = parse_meeting(meeting)
            settings = api.get_settings(["ZOOM_CLIENT_ID", "ZOOM_CLIENT_SECRET"])
            client_id = str(settings.get("ZOOM_CLIENT_ID") or "").strip()
            if not client_id or not str(settings.get("ZOOM_CLIENT_SECRET") or "").strip():
                raise ToolRefusal("ZOOM_CREDENTIALS_MISSING", (
                    "Zoom app credentials are not set: save the Meeting SDK app's ZOOM_CLIENT_ID and "
                    "ZOOM_CLIENT_SECRET in Settings → Secrets and grant them to zoom_meeting"))
            account_id = app_account_id(client_id)
            binding = select_binding(_bindings(api, ctx), requested=str(binding_id or "").strip(),
                                     account_id=account_id, meeting_number=number)
            _require_ready(_companion(state_dir, "status"), account_id)
            reply = _companion(state_dir, "join", {
                "meeting_number": number, "passcode": str(passcode or "").strip() or link_passcode,
                "display_name": str(display_name or "").strip(), "binding_id": binding.binding_id,
                "account_id": account_id, "on_behalf_token": str(on_behalf_token or "").strip(),
                "requested_by": requester_facts(ctx),
            })
            if not reply.get("ok"):
                raise ToolRefusal(str(reply.get("code") or "ZOOM_JOIN_REFUSED"), str(reply.get("error") or reply))
            status = _await_first_answer(state_dir, str(reply.get("join_id") or ""))
        except ToolRefusal as refusal:
            return refusal.render()
        guidance = ("The call opens by itself once the meeting host allows the audio; "
                    "zoom_status follows the join, zoom_leave ends it.")
        return f"{render_status(status)}\n{guidance}"

    def zoom_leave(ctx: Any) -> str:
        try:
            reply = _companion(state_dir, "leave")
        except ToolRefusal as refusal:
            return refusal.render()
        return render_status(reply) if reply.get("left") else f"not in a Zoom meeting\n{render_status(reply)}"

    def zoom_status(ctx: Any) -> str:
        try:
            return render_status(_companion(state_dir, "status"))
        except ToolRefusal as refusal:
            return refusal.render()

    api.register_tool("zoom_join", handler=zoom_join, timeout_sec=60, description=(
        "Join a Zoom meeting by voice through this install's Zoom Meeting SDK companion. The meeting "
        "becomes one Presence call governed by the chosen binding's behavior and ceiling; participants see "
        "the display name. The meeting host approves the bot's audio in Zoom on every join before the call "
        "opens. Returns the companion's state after Zoom's first answer."), schema={
        "type": "object",
        "properties": {
            "meeting": {"type": "string", "description": "Zoom join link (https://...zoom.us/j/<number>?pwd=...) "
                                                         "or the numeric meeting ID."},
            "passcode": {"type": "string", "description": "Meeting passcode; omit when the link carries pwd=."},
            "display_name": {"type": "string", "description": "Name participants see; default: this install's "
                                                              "name."},
            "binding_id": {"type": "string", "description": "Presence binding that governs the call; needed only "
                                                            "when several Zoom bindings exist."},
            "on_behalf_token": {"type": "string", "description": "Zoom OBF token for a meeting hosted outside the "
                                                                 "Zoom app's own account; its authorizing user must "
                                                                 "already be in the meeting."},
        },
        "required": ["meeting"],
    })
    api.register_tool("zoom_leave", handler=zoom_leave, timeout_sec=30, description=(
        "Leave the current Zoom meeting: ends the call and cleans the SDK up."),
        schema={"type": "object", "properties": {}})
    api.register_tool("zoom_status", handler=zoom_status, timeout_sec=30, description=(
        "The Zoom companion's state: idle, joining, waiting room, waiting for the host to allow audio, in the "
        "call, and the last outcome."), schema={"type": "object", "properties": {}})
