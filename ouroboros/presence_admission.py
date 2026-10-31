"""Resolve authenticated connector events to host places; no behavior or tool ceiling."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ouroboros.places import Place, PlaceError, resolve_place
from ouroboros.presence_bindings import PresenceEndpoint


class PresenceAdmissionError(ValueError):
    def __init__(self, code: str, field: str, message: str = "") -> None:
        self.code, self.field = code, field
        super().__init__(message or f"{code}: {field}")


@dataclass(frozen=True)
class PresenceAdmission:
    """Frozen place choice; legacy fields remain readable for already-admitted work."""
    binding_id: str
    transport_skill: str
    origin: PresenceEndpoint
    destination: PresenceEndpoint
    behavior_skill: str = ""
    instructions: str = ""
    context_topics: tuple[str, ...] = ()
    model_slot: str = ""
    inline_max_rounds: int | None = None
    skill_content_hash: str = ""
    profile_fingerprint: str = ""
    state_fingerprint: str = ""
    selection_fingerprint: str = ""
    capability_ceiling: Any = None
    workspace_root: str = ""
    place: Place | None = None
    allowance: Mapping[str, Any] | None = None


def admit_presence_turn(*, drive_root: Path, authenticated_transport_skill: str, binding_id: str = "",
                        global_max_rounds: int | None = None, repo_path: str | None = None,
                        provider: str = "", account_id: str = "", conversation_id: str = "",
                        thread_id: str = "") -> PresenceAdmission:
    """Authentication remains the Host's job; this resolves scope, never owner identity."""
    try:
        if binding_id:
            from ouroboros.places import load_places
            place = next((p for p in load_places(drive_root)
                          if binding_id in (p.place_id, *p.legacy_binding_ids)), None)
            if place is not None and place.connector != authenticated_transport_skill:
                raise PresenceAdmissionError("presence_binding_wrong_transport", "binding_id")
        else:
            place = resolve_place(drive_root, connector=authenticated_transport_skill, provider=provider,
                                  account_id=account_id, conversation_id=conversation_id, thread_id=thread_id)
    except PlaceError as exc:
        raise PresenceAdmissionError(exc.code, exc.field, str(exc)) from exc
    if place is None:
        raise PresenceAdmissionError("place_not_connected", "place", "Connect this place in Consciousness > Places.")
    if ((provider and place.provider != provider)
            or (place.account_id and account_id and place.account_id != account_id)
            or (place.conversation_id and conversation_id and place.conversation_id != conversation_id)
            or (place.thread_id and (provider or thread_id) and place.thread_id != thread_id)):
        raise PresenceAdmissionError("presence_origin_mismatch", "place")
    origin = PresenceEndpoint(place.provider, place.account_id or account_id, place.conversation_id or "*", place.thread_id)
    destination = (PresenceEndpoint(**place.destination) if place.destination else
                   PresenceEndpoint(place.provider, place.account_id or account_id, conversation_id or place.conversation_id or "*", thread_id))
    return PresenceAdmission(binding_id=(place.legacy_binding_ids or (place.place_id,))[0],
                             transport_skill=place.connector, origin=origin, destination=destination,
                             instructions="\n\n".join(note.text for note in place.notes),
                             context_topics=place.context_topics, model_slot=place.model,
                             workspace_root=place.working_folder, place=place)


def focus_enabled(drive_root: Path) -> bool:
    from ouroboros.consciousness import panic_blocks_wake
    from supervisor.state import control_is, read_state
    return not panic_blocks_wake(drive_root) and control_is(read_state(drive_root).projection(), "bg_consciousness_enabled", True)


def focus_allowance(drive_root: Path) -> dict:
    from ouroboros.consciousness_allowance import allowance_window
    return allowance_window(drive_root)
