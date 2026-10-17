"""meeting_audio: Ouroboros hears the owner's meeting from inside a live call.

The tools never open a call. Each finds the call it belongs to from its own
invocation: the host admits a call as a turn whose task carries
``metadata.live_call.call_id``, and that metadata reaches a tool's ``ctx`` both
in-process and in an extension child. Outside a call there is nothing to join,
and the refusal says so in plain words. The work itself happens in the
``meeting_capture`` companion (the one process that holds audio devices and the
call socket), reached through the private control channel in this skill's state
directory.
"""

from __future__ import annotations

import json
import pathlib
import sys
from typing import Any, Dict

from .scripts.meeting_control import ControlUnavailable, request

LISTEN_TIMEOUT_SEC = 50
CONTROL_TIMEOUT_SEC = 25

NO_CALL = ("I can only listen to a meeting during a live call. Ask the owner to press Call, "
           "then ask me again.")
SPEAK_UNAVAILABLE = {
    "darwin": ("speaking mode is not available on macOS because it needs an audio driver; the browser "
               "participant (the meeting_browser skill) is the driver-free alternative"),
    "win32": ("speaking mode is not available on Windows because it needs an audio driver; the browser "
              "participant (the meeting_browser skill) is the driver-free alternative"),
}


def _platform() -> str:
    return sys.platform


def _call_id(ctx: Any) -> str:
    metadata = getattr(ctx, "task_metadata", None)
    live = metadata.get("live_call") if isinstance(metadata, dict) else None
    return str(live.get("call_id") or "").strip() if isinstance(live, dict) else ""


def _reply(payload: Dict[str, Any]) -> str:
    """JSON for the mind; a refusal leads with the typed line the host records as a failure."""
    body = json.dumps(payload, ensure_ascii=False)
    if payload.get("ok") is False:
        return f"⚠️ {payload.get('code') or 'MEETING_REFUSED'}: {payload.get('message') or ''}\n{body}"
    return body


def _speak_refusal() -> Dict[str, Any]:
    platform = _platform()
    if platform.startswith("linux"):
        return {}
    message = SPEAK_UNAVAILABLE.get(platform, f"speaking mode is not available on {platform}")
    return {"ok": False, "code": "MEETING_SPEAK_UNSUPPORTED", "message": message}


class MeetingTools:
    def __init__(self, state_dir: pathlib.Path):
        self.state_dir = pathlib.Path(state_dir)

    def _ask(self, op: str, args: Dict[str, Any], timeout_sec: float) -> str:
        try:
            return _reply(request(self.state_dir, op, args, timeout_sec=timeout_sec))
        except ControlUnavailable as exc:
            return _reply({"ok": False, "code": "MEETING_COMPANION_UNAVAILABLE", "message": str(exc)})

    def listen_start(self, ctx: Any, app: str = "") -> str:
        call_id = _call_id(ctx)
        if not call_id:
            return _reply({"ok": False, "code": "MEETING_NO_CALL", "message": NO_CALL})
        return self._ask("listen_start", {"call_id": call_id, "app": str(app or "").strip()}, LISTEN_TIMEOUT_SEC)

    def listen_stop(self, ctx: Any) -> str:
        return self._ask("listen_stop", {}, CONTROL_TIMEOUT_SEC)

    def speak_start(self, ctx: Any) -> str:
        refusal = _speak_refusal()
        if refusal:
            return _reply(refusal)
        call_id = _call_id(ctx)
        if not call_id:
            return _reply({"ok": False, "code": "MEETING_NO_CALL", "message": NO_CALL})
        return self._ask("speak_start", {"call_id": call_id}, CONTROL_TIMEOUT_SEC)

    def speak_stop(self, ctx: Any) -> str:
        refusal = _speak_refusal()
        if refusal:
            return _reply(refusal)
        return self._ask("speak_stop", {}, CONTROL_TIMEOUT_SEC)

    def status(self, ctx: Any) -> str:
        try:
            reply = request(self.state_dir, "status", {}, timeout_sec=CONTROL_TIMEOUT_SEC)
        except ControlUnavailable as exc:
            reply = {"ok": False, "code": "MEETING_COMPANION_UNAVAILABLE", "message": str(exc)}
        return _reply({**reply, "in_call": bool(_call_id(ctx))})


_NO_ARGS = {"type": "object", "properties": {}}


def register(api: Any) -> None:
    tools = MeetingTools(pathlib.Path(api.get_state_dir()))
    api.register_tool(
        "meeting_listen_start", tools.listen_start,
        description=(
            "Listen to the owner's meeting on this computer during the current live call: the meeting "
            "application's sound joins the call as meeting audio beside the owner's voice, while your "
            "spoken hints still reach only the owner. Works only inside a live call; outside one, ask the "
            "owner to press Call. `app` names the application playing the meeting (for example 'zoom', "
            "'Microsoft Teams', or 'Google Chrome' for a meeting in a browser); empty listens to the "
            "meeting applications that are running. The first use on macOS asks the owner for the "
            "Screen & System Audio Recording permission. The result says what is heard and whether it "
            "can contain your own voice."
        ),
        schema={"type": "object", "properties": {"app": {
            "type": "string", "default": "",
            "description": "Meeting application name, or empty for the running meeting applications.",
        }}},
        timeout_sec=LISTEN_TIMEOUT_SEC + 10,
    )
    api.register_tool(
        "meeting_listen_stop", tools.listen_stop,
        description="Stop listening to the meeting; speaking mode, if on, stops too.",
        schema=_NO_ARGS, timeout_sec=CONTROL_TIMEOUT_SEC + 5,
    )
    api.register_tool(
        "meeting_speak_start", tools.speak_start,
        description=(
            "Linux only: let the meeting hear you through a software microphone named 'Ouroboros' (no "
            "driver; the owner selects it as the microphone in the meeting application). Needs listening "
            "to be on. On macOS and Windows it is unavailable by design, because it would need an audio "
            "driver; the meeting_browser participant is the alternative there."
        ),
        schema=_NO_ARGS, timeout_sec=CONTROL_TIMEOUT_SEC + 5,
    )
    api.register_tool(
        "meeting_speak_stop", tools.speak_stop,
        description="Stop speaking into the meeting and remove the 'Ouroboros' microphone; listening continues.",
        schema=_NO_ARGS, timeout_sec=CONTROL_TIMEOUT_SEC + 5,
    )
    api.register_tool(
        "meeting_status", tools.status,
        description=(
            "What the meeting companion is doing: idle, listening (to which application and how), or "
            "speaking, plus the last reason it stopped and whether this turn is a live call."
        ),
        schema=_NO_ARGS, timeout_sec=CONTROL_TIMEOUT_SEC + 5,
    )
    api.register_companion_process("meeting_capture")
