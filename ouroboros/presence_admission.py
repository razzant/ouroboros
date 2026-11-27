"""Resolve authenticated connector events to host places; no behavior or tool ceiling."""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from ouroboros.places import Place, PlaceError, place_records, resolve_place
from ouroboros.places_migration import complete_wake_import, finishing_declarations, import_unfinished
from ouroboros.presence_bindings import PresenceEndpoint


class PresenceAdmissionError(ValueError):
    def __init__(self, code: str, field: str, message: str = "") -> None:
        self.code, self.field = code, field
        super().__init__(message or f"{code}: {field}")


class WakeImportPending(PresenceAdmissionError):
    code = "wake_import_declarations_pending"
    retry_after = 60

    def __init__(self):
        super().__init__(self.code, "place", "This connector's wake declarations are not available yet. Retry this event.")


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
    circle: Mapping[str, Any] | None = None  # circle words with both origins, and model, at admission
    # The words the place's live declarations give its facts (``presence_events.connector_fact_labels``), as a door read
    # them; a carrier takes the whole map from its seed, by value: display only, never a fact's meaning.
    fact_labels: Mapping[str, str] | None = None
    # The snapshot this admission read left its legacy connection's wake import unfinished
    # (``places_migration.import_unfinished``): only then may a captured arrival read the store again to finish it.
    unfinished_import: bool = False
    # The connection an owner_bot event arrived under (``telegram_owner_bot``), checked at the event door and again at
    # every re-admission of that event; None for every other place.
    connected_revision: int | None = None


def origin_outside(origin: PresenceEndpoint, provider: str, account_id: str, conversation_id: str, thread_id: str) -> bool:
    """Whether the event's provider coordinates fall outside this connection."""
    return (provider != origin.transport or account_id != origin.account_id
            or (origin.conversation_id != "*" and conversation_id != origin.conversation_id)
            or bool(origin.thread_id and thread_id != origin.thread_id))


def _scoped_place(root, records, connector, binding_id, *, provider, account_id, conversation_id, thread_id, live_event,
                  connected_revision=None):
    from ouroboros.telegram_owner_bot import FEED_CONNECTOR, admits, is_owner_bot_place, owner_bridge_bot_id
    # The owner's own bot (A) enters only through its exact owner_bot place, named, never a broader or B place.
    owner_bot = owner_bridge_bot_id(root) if connector == FEED_CONNECTOR else ""
    if binding_id:
        place = next((p for p in records.places.values() if binding_id in (p.place_id, *p.legacy_binding_ids)), None)
        if place is not None and place.connector != connector:
            raise PresenceAdmissionError("presence_binding_wrong_transport", "binding_id")
    else:
        place = resolve_place(root, connector=connector, provider=provider, account_id=account_id,
                              conversation_id=conversation_id, thread_id=thread_id, places=records.places.values(),
                              owner_bot=owner_bot)
    if place is not None and not admits(place, account_id=account_id, conversation_id=conversation_id,
                                        thread_id=thread_id, owner_bot=owner_bot, named=bool(binding_id)):
        place = None
    if place is None:
        raise PresenceAdmissionError("place_not_connected", "place", "Add this place in Circles.")
    if connected_revision is not None and is_owner_bot_place(place) and place.connected_revision != connected_revision:
        raise PresenceAdmissionError("presence_origin_mismatch", "connected_revision")  # arrived under another connection
    if ((provider and place.provider != provider)
            or (place.account_id and account_id and place.account_id != account_id)
            or (place.conversation_id and conversation_id and place.conversation_id != conversation_id)
            or (place.thread_id and (provider or thread_id) and place.thread_id != thread_id)):
        raise PresenceAdmissionError("presence_origin_mismatch", "place")
    origin = PresenceEndpoint(place.provider, place.account_id or account_id, place.conversation_id or "*", place.thread_id)
    if live_event and origin_outside(origin, provider, account_id, conversation_id, thread_id):
        raise PresenceAdmissionError("presence_origin_mismatch", "place")
    return place, origin


def admit_presence_turn(*, drive_root: Path, authenticated_transport_skill: str, binding_id: str = "",
                        global_max_rounds: int | None = None, repo_path: str | None = None,
                        provider: str = "", account_id: str = "", conversation_id: str = "",
                        thread_id: str = "", live_event: bool = False, persist_import: bool = True,
                        connected_revision: int | None = None) -> PresenceAdmission:
    """Resolve a place and its circle from one snapshot. A live door previews the scope with persist_import=False
    until the event passes capture. Only its subsequent admission can finish the proven import. ``connected_revision``
    (the event door's, ``-1`` when the sender named none): an owner_bot place admits only at its current connection."""
    from ouroboros.telegram_owner_bot import is_owner_bot_place

    def read_place():
        records = place_records(drive_root, persist_import=persist_import)
        place, origin = _scoped_place(drive_root, records, authenticated_transport_skill, binding_id, provider=provider,
                                      account_id=account_id, conversation_id=conversation_id, thread_id=thread_id,
                                      live_event=live_event, connected_revision=connected_revision)
        return records, place, origin

    try:
        records, place, origin = read_place()
        if live_event and persist_import and complete_wake_import(drive_root, records, place):
            # Completion used the current record under the lock. Read the resulting place and circle together.
            records, place, origin = read_place()
    except PlaceError as exc:
        raise PresenceAdmissionError(exc.code, exc.field, str(exc)) from exc
    destination = (PresenceEndpoint(**place.destination) if place.destination else
                   PresenceEndpoint(place.provider, place.account_id or account_id, conversation_id or place.conversation_id or "*", thread_id))
    return PresenceAdmission(binding_id=(place.legacy_binding_ids or (place.place_id,))[0],
                             transport_skill=place.connector, origin=origin, destination=destination,
                             instructions="\n\n".join(note.text for note in place.notes),
                             context_topics=place.context_topics, model_slot=place.model,
                             workspace_root=place.working_folder, place=place,
                             circle=records.circle_context(place.place_id),
                             unfinished_import=import_unfinished(records, place),
                             connected_revision=connected_revision if is_owner_bot_place(place) else None)


def admit_captured_wakes(root: Path, admission: PresenceAdmission, event) -> PresenceAdmission:
    """Only a captured, validated arrival can finish its legacy connection's wake import. It reads the store again
    only while its own snapshot left that import unfinished and a live declaration can finish it: a sealed import,
    or one with nothing published to apply, costs no read here (``decide_attention`` reads the current choices)."""
    if not admission.unfinished_import or not finishing_declarations(admission.place):
        return admission
    current = admit_presence_turn(drive_root=root, authenticated_transport_skill=admission.transport_skill,
                                  binding_id=admission.place.place_id, provider=event.provider,
                                  account_id=event.account_id, conversation_id=event.conversation_id,
                                  thread_id=event.thread_id, live_event=True,
                                  connected_revision=admission.connected_revision)
    return replace(current, fact_labels=admission.fact_labels)


def switch_cause(drive_root: Path) -> str:
    """The switch by its one cause, the one reading behind intake (``focus_state``) and the places' quiet words
    (``place_projection.switch_state``): ``on``; ``panic`` (a Panic waits, or Panic turned it off); ``off`` (the
    owner's switch known Off, or the mind's own stop); ``unknown`` (a switch this read cannot know)."""
    from ouroboros.consciousness import PANIC_STOP, STOP_SOURCE_KEY, panic_blocks_wake
    from supervisor.state import control_is, control_value, read_state
    if panic_blocks_wake(drive_root):
        return "panic"
    projection = read_state(drive_root).projection()
    if control_is(projection, "bg_consciousness_enabled", True):
        return "on"
    if not control_value(projection, "bg_consciousness_enabled")[0]:
        return "unknown"
    known, source = control_value(projection, STOP_SOURCE_KEY)  # an unconfirmed source names no Panic
    return "panic" if known and source == PANIC_STOP else "off"


def focus_state(drive_root: Path) -> str:
    """``on``; ``off`` (the owner's switch known Off, or a Panic); ``unknown`` (a switch this read cannot know).
    Intake admits only ``on``; a view claims Off only for ``off``. ``switch_cause`` folded: a Panic is Off here."""
    cause = switch_cause(drive_root)
    return "off" if cause == "panic" else cause


def focus_enabled(drive_root: Path) -> bool:
    return focus_state(drive_root) == "on"


def focus_allowance(drive_root: Path) -> dict:
    from ouroboros.consciousness_allowance import allowance_window
    return allowance_window(drive_root)
