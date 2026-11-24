"""Meeting peers use the ordinary place admission, execution registry and history.

A reviewed transport token and a connected place never mint owner speech. A call
speaks in its place, so it opens only in an Active place whose current choices
would start a focus for the event (wake match, consciousness on and not Advises,
allowance left); the request is heard in the room first, as a text event is. It
runs the same loaded feature/provider pair as an owner call, under the turn gate.
"""
from __future__ import annotations
import contextvars
import logging
import secrets
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping
from ouroboros.presence_admission import admit_captured_wakes, admit_presence_turn, PresenceAdmissionError
from ouroboros.presence_events import connector_fact_labels, outside_origin, parse_event_wire
from ouroboros.presence_runner import (PresenceTurnEvent, PresenceTurnError, PROACTIVE_TURNS,
    _configured_gate, _task_id, _log_dialogue, _stable_numeric_id, _stored_turn, capture_presence_event, presence_chat_id,
    presence_event_identity, presence_turn_replay, run_presence_turn)

log = logging.getLogger(__name__)
MEETING_AUDIO_LABEL = "meeting audio"

class PresenceCallRefused(ValueError):
    def __init__(self, code, message):
        self.code, self.message = code, message
        super().__init__(message)

@dataclass(frozen=True)
class PresenceCallOpened:
    call_id: str
    task_id: str


def _require_waking_place(drive_root, admission, event, captured) -> None:
    """Refuse before any hub opens unless this event would start a focus in an Active place now.

    A call is speech: Silent and Paused places, consciousness Off or Panic, and Advises (which
    holds automatic replies) refuse it. A refused request stays a heard arrival in the room.
    Only the admission is checked; the call's speech does not cross the Supervisor per utterance.
    """
    from ouroboros.config import get_consciousness_autonomy
    from ouroboros.consciousness_authority import LEVEL_OBSERVE, normalize_level
    from ouroboros.place_intake import WAKE_IMPORT_PENDING, decide_attention, settle_heard

    if admission.place is None:
        decision, details = "silent", {"reason": "disconnected"}
    else:  # Active is read with the owner's other current choices, after capture (``decide_attention``)
        decision, details = decide_attention(Path(drive_root), admission, event, active_only=True)
    if decision == "focus" and normalize_level(get_consciousness_autonomy()) == LEVEL_OBSERVE:
        decision, details = "silent", {"reason": "advises"}
    if decision == "retry":
        if details.get("reason") == WAKE_IMPORT_PENDING:
            raise PresenceCallRefused(WAKE_IMPORT_PENDING,
                                      "This place's updated wake declarations are not admitted yet.")
        raise PresenceCallRefused("consciousness_allowance_exhausted", "The consciousness allowance is used up.")
    if decision != "focus":
        reason = str(details.get("reason") or "heard")
        settle_heard(Path(drive_root), captured, event, reason)
        raise PresenceCallRefused(f"place_{reason}", f"This place does not take a call for this event ({reason}).")


def open_presence_call(*, drive_root, repo_dir, skill_name, binding_id="", event,
                       label="", event_queue=None, agent_factory=None):
    from ouroboros.call_protocol import CallSpec, OPENED_BY_SKILL
    from ouroboros.call_peers import open_hub, close_hub
    from ouroboros.live_conversation import LiveConversationTurn, feature_configuration
    from supervisor.message_bus import get_bridge

    try:
        # The connector's live declared signals and words, as /presence/turn admits them (one Host, one registry).
        turn_event = parse_event_wire(event)
        admission = admit_presence_turn(drive_root=drive_root, authenticated_transport_skill=skill_name,
            binding_id=binding_id, provider=turn_event.provider, account_id=turn_event.account_id,
            conversation_id=turn_event.conversation_id, thread_id=turn_event.thread_id,
            live_event=True, persist_import=False)
        if outside_origin(admission, turn_event):
            raise PresenceTurnError("presence_origin_mismatch", "event")
        admission = replace(admission, fact_labels=connector_fact_labels(drive_root, admission, turn_event))
        task_id = _task_id(admission, turn_event)
        identity = presence_event_identity(admission.binding_id, turn_event)
        if admission.place and admission.place.legacy_binding_ids:
            # A legacy connection's stored turn can predate its room's capture record, sealed import or not: its
            # identity is checked before this event writes a room row or finishes the import.
            _stored_turn(Path(drive_root), task_id, identity)
        # Heard before any decision, as a text event is (host_service_places).
        captured = capture_presence_event(admission, turn_event, drive_root=Path(drive_root))
        admission = admit_captured_wakes(Path(drive_root), admission, turn_event)
    except (PresenceAdmissionError, PresenceTurnError) as exc:
        raise PresenceCallRefused(exc.code, str(exc)) from exc
    _require_waking_place(drive_root, admission, turn_event, captured)
    with feature_configuration("live_conversation:call") as (feature, config):
        if not config.enabled:
            raise PresenceCallRefused("calls_disabled", "Calls (beta) are off.")
        spec = CallSpec(secrets.token_hex(16), presence_chat_id(turn_event.conversation_key),
            config.provider, config.model, config.voice, opened_by=OPENED_BY_SKILL,
            presence_binding_id=admission.binding_id, label=str(label), client_message_id=turn_event.source_event_id)
        turn = LiveConversationTurn(feature, config, spec, publish=get_bridge().broadcast)
        spec = turn.spec
        lease = _configured_gate(Path(drive_root)).try_acquire(turn_event.conversation_key)
        if lease is None:
            raise PresenceCallRefused("presence_call_busy", "This conversation or the Presence capacity is busy.")
        try:
            if presence_turn_replay(Path(drive_root), task_id, turn_event.conversation_key, identity, 0) is not None:
                raise PresenceCallRefused("presence_call_settled", "This event's turn already ended.")
            open_hub(spec)
            def run(held):
                try:
                    return run_presence_turn(admission=admission, event=turn_event, repo_dir=Path(repo_dir),
                        drive_root=Path(drive_root), event_queue=event_queue, agent_factory=agent_factory,
                        admitted=held, live_turn=turn)
                finally:
                    close_hub(spec.call_id, "presence_ended")
            execution, started = PROACTIVE_TURNS.start_thread(task_id, identity=identity,
                admit=lambda: lease, run=run, context=contextvars.copy_context())
            if not started:
                close_hub(spec.call_id, "presence_busy")
                lease.release()
                raise PresenceCallRefused("presence_call_busy", "This event already has a live turn.")
        except BaseException:
            close_hub(spec.call_id, "presence_refused")
            lease.release()
            raise
        return PresenceCallOpened(spec.call_id, task_id)


def _event_from_task(task: Mapping[str, Any]) -> PresenceTurnEvent:
    """The opening event a call's utterances are named after, with its focus's admitted identity (``focus.identity``):
    a heard row keeps the account the place sends as, so the room's newest arrivals never lose it to a long call."""
    presence = task["metadata"]["presence"]
    stored = presence["event"]
    focus = task["metadata"].get("focus")
    return PresenceTurnEvent(
        source_event_id=str(stored["source_event_id"]), provider=str(stored["provider"]),
        account_id=str(stored["account_id"]), conversation_id=str(stored["conversation_id"]),
        thread_id=str(stored["thread_id"]), conversation_key=str(stored["conversation_key"]),
        actor=dict(stored.get("actor") or {}), conversation=dict(stored.get("conversation") or {}),
        message=dict(stored.get("message") or {}), text=str(presence.get("observed_text") or ""),
        delivery_reporting_version=int(presence.get("delivery_reporting_version") or 0),
        identity=focus.get("identity") if isinstance(focus, Mapping) else None,
    )


def record_call_utterance(drive_root: Path, task: Mapping[str, Any], *, direction: str, text: str,
                          speaker_label: str = "", utterance_id: str = "") -> bool:
    """Retain one finalized utterance of a live call as a Presence dialogue row.

    The driver calls this for every finalized correspondent utterance (``direction="in"``,
    attributed to the participant label the platform reported, else to "meeting audio")
    and for every utterance Ouroboros spoke (``direction="out"``). The row carries the
    call's task and conversation exactly like a text turn's rows, so history, memory and
    the previous-turn pointer read the call as one conversation. An inbound row is named
    by the event and its ``utterance_id`` (the engine's item id; a fresh name without
    one): the store keeps one inbound row per name, so the event's own name would keep
    only the call's opening row, while the same item recorded twice stays one row.
    False means the row could not be retained; the host ends the call with
    ``history_write`` instead of continuing with a missing canonical utterance.
    """
    text = str(text or "").strip()
    if direction not in {"in", "out"} or not text:
        return False
    event = _event_from_task(task)
    if direction == "in":  # what the connector reported with the opening (a change, its files, its send time and order)
        event = replace(event, message={})  # stays the opening row's, never a heard utterance's (``call_utterance``)
    chat_id = int(task["chat_id"])
    if direction == "out":
        speaker, user_id = None, 0
    else:
        label = str(speaker_label or "").strip() or MEETING_AUDIO_LABEL
        speaker = {"display_name": label, "platform_actor_id": f"meeting-audio:{label}"}
        user_id = _stable_numeric_id("presence-actor-log", speaker["platform_actor_id"])
    name = f"{event.source_event_id}:{'heard' if direction == 'in' else 'spoken'}:{utterance_id or secrets.token_hex(8)}"
    try:
        _log_dialogue(Path(drive_root), direction=direction, chat_id=chat_id, user_id=user_id, text=text,
                      event=event, task=task, task_id=str(task["id"]), speaker=speaker, client_message_id=name)
    except PresenceTurnError:
        log.warning("Presence call %s: a %s utterance could not be retained", task.get("id"), direction)
        return False
    return True


def call_utterance(row: Mapping[str, Any]) -> bool:
    """A row ``record_call_utterance`` wrote, by the name it gave it after its event (``<event>:heard:<item>``,
    ``:spoken:``); the event's own row keeps the event's name. Its identity, focus and, on one written before heard
    utterances stopped carrying it, its message are the call's opening facts, never this utterance's."""
    provenance = row.get("presence_provenance")
    event = str(provenance.get("source_event_id") or "") if isinstance(provenance, Mapping) else ""
    return bool(event) and str(row.get("client_message_id") or "").startswith((f"{event}:heard:", f"{event}:spoken:"))
