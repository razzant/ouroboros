"""meeting_browser: Ouroboros's own browser window joins a meeting as a participant.

Experimental: a known vendor's meeting can be joined, and until that vendor's
participant probe passes the join result says so. Like the meeting_audio tools, these tools never open a call: a join
belongs to the live call whose turn invoked it (``metadata.live_call.call_id``
on the task, visible in ``ctx`` in-process and in an extension child), and the
``meeting_participant`` companion does the work over the private control
channel in this skill's state directory.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any, Dict

from .scripts.meeting_control import ControlUnavailable, request

JOIN_TIMEOUT_SEC = 120
CONTROL_TIMEOUT_SEC = 30

NO_CALL = ("I can only join a meeting as a participant during a live call. Ask the owner to press Call, "
           "then ask me again.")


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


class ParticipantTools:
    def __init__(self, state_dir: pathlib.Path):
        self.state_dir = pathlib.Path(state_dir)

    def _ask(self, op: str, args: Dict[str, Any], timeout_sec: float) -> Dict[str, Any]:
        try:
            return request(self.state_dir, op, args, timeout_sec=timeout_sec)
        except ControlUnavailable as exc:
            return {"ok": False, "code": "MEETING_COMPANION_UNAVAILABLE", "message": str(exc)}

    def join(self, ctx: Any, url: str = "") -> str:
        call_id = _call_id(ctx)
        if not call_id:
            return _reply({"ok": False, "code": "MEETING_NO_CALL", "message": NO_CALL})
        return _reply(self._ask("join", {"call_id": call_id, "url": str(url or "").strip()}, JOIN_TIMEOUT_SEC))

    def leave(self, ctx: Any) -> str:
        return _reply(self._ask("leave", {}, CONTROL_TIMEOUT_SEC))

    def status(self, ctx: Any) -> str:
        return _reply({**self._ask("status", {}, CONTROL_TIMEOUT_SEC), "in_call": bool(_call_id(ctx))})


def register(api: Any) -> None:
    tools = ParticipantTools(pathlib.Path(api.get_state_dir()))
    api.register_tool(
        "meeting_join", tools.join,
        description=(
            "Experimental: join a Zoom web, Google Meet or Microsoft Teams meeting as a participant in "
            "Ouroboros's own visible browser window, during the current live call. The meeting then hears "
            "your voice and you hear the meeting, without any audio driver. Until a service's participant "
            "probe has passed, the join result says so and the page may leave you listen-only. It may need a "
            "click on its Join button in that window (use the computer-use tools), and the owner signs "
            "Ouroboros's own account in there once."
        ),
        schema={"type": "object", "properties": {"url": {
            "type": "string", "description": "The https meeting link."}}, "required": ["url"]},
        timeout_sec=JOIN_TIMEOUT_SEC + 10,
    )
    api.register_tool(
        "meeting_leave", tools.leave,
        description="Leave the meeting and close Ouroboros's browser window.",
        schema={"type": "object", "properties": {}}, timeout_sec=CONTROL_TIMEOUT_SEC + 5,
    )
    api.register_tool(
        "meeting_status", tools.status,
        description=(
            "Whether Ouroboros's browser participant is in a meeting and whether the meeting really receives "
            "its voice (speaking) or not yet (listen-only), with the page title and address."
        ),
        schema={"type": "object", "properties": {}}, timeout_sec=CONTROL_TIMEOUT_SEC + 5,
    )
    api.register_companion_process("meeting_participant")
