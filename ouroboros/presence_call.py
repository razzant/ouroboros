"""Presence call mode: a reviewed binding opens one live audio session as one long turn.

A transport skill that joined a meeting (a Zoom companion, a capture peer) asks the
Host to open a call for an owner-created binding. The same admission that admits a
text turn admits the call: the reviewed profile, the owner's selections and the
positive capability ceiling. The call is then the ordinary Presence turn pipeline
(``presence_runner.run_presence_turn``) with ``metadata["live_call"]`` set, so the
driver runs the realtime session instead of the text loop while the admission
record, the lost-attempt refusal, the previous-turn pointer and the dialogue rows
stay exactly what a text turn produces. Opening never waits: the gate lease is
taken with one attempt and a busy conversation or a full cap is a typed refusal.
"""

from __future__ import annotations

import contextvars
import logging
import secrets
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping

from ouroboros.presence_admission import PresenceAdmission, PresenceAdmissionError, admit_presence_turn
from ouroboros.presence_gate import PresenceTurnLease, _configured_gate
from ouroboros.presence_runner import (
    PresenceTurnError,
    PresenceTurnEvent,
    PresenceTurnResult,
    _log_dialogue,
    _stable_numeric_id,
    _stored_turn,
    _retry_target,
    _task_id,
    presence_chat_id,
    presence_event_identity,
    presence_turn_event,
    presence_turn_replay,
    run_presence_turn,
)
from ouroboros.presence_runtime import EXECUTION_CALL
from ouroboros.task_results import STATUS_INTERRUPTED, STATUS_RUNNING, is_reconciled_presence_placeholder

if TYPE_CHECKING:  # the call vocabulary is agent-core (D01); Presence reaches it lazily, as the domain graph pins
    from ouroboros.call_protocol import CallSpec

log = logging.getLogger(__name__)

MEETING_AUDIO_LABEL = "meeting audio"


class PresenceCallRefused(ValueError):
    """The call was not opened; ``code`` is the typed reason and ``message`` is plain."""

    def __init__(self, code: str, message: str) -> None:
        self.code = str(code or "presence_call_refused")
        self.message = str(message or self.code)
        super().__init__(self.message)


@dataclass(frozen=True)
class PresenceCallOpened:
    call_id: str
    task_id: str


@dataclass
class PresenceCallRun:
    """One live call's process-local facts: its thread, the lease it holds and how it ended."""

    spec: CallSpec
    task_id: str
    lease: PresenceTurnLease
    thread: threading.Thread | None = None
    result: PresenceTurnResult | None = None
    error: BaseException | None = None
    ended: threading.Event | None = None


_RUNS_LOCK = threading.Lock()
_RUNS: dict[str, PresenceCallRun] = {}


def presence_call_run(call_id: str) -> PresenceCallRun | None:
    """The live (or just-ended, until collected) run of one call; None when unknown here."""
    with _RUNS_LOCK:
        return _RUNS.get(str(call_id or ""))


def _open_hub(spec: CallSpec) -> Any:
    """The per-call hub registry (``call_peers``); imported late so the call door loads
    without the audio stack, and so tests stub this seam."""
    from ouroboros import call_peers

    return call_peers.open_hub(spec)


def _close_hub(call_id: str) -> None:
    from ouroboros import call_peers

    call_peers.close_hub(call_id)


def _close_hub_quietly(call_id: str) -> None:
    try:
        _close_hub(call_id)
    except Exception:  # the call is already over; a hub that cannot close is logged, not raised
        log.warning("Presence call %s: hub close failed", call_id, exc_info=True)


def _call_setting(key: str, default: str = "") -> str:
    from ouroboros.config import SETTINGS_DEFAULTS, runtime_setting

    raw = runtime_setting(key, SETTINGS_DEFAULTS.get(key, default))
    return str(raw if raw is not None else "").strip()


def _call_route() -> tuple[str, str, str]:
    """``(engine, model, voice)`` from the owner's call settings (Gemini Live); a missing model is a refusal."""
    model = _call_setting("OUROBOROS_CALL_MODEL_GOOGLE")
    if not model:
        raise PresenceCallRefused("presence_call_model_unset", "OUROBOROS_CALL_MODEL_GOOGLE is empty")
    return "google", model, _call_setting("OUROBOROS_CALL_VOICE_GOOGLE")


def _call_max_minutes() -> int | None:
    """``OUROBOROS_PRESENCE_CALL_MAX_MINUTES``: empty means no ceiling; a set value is a positive
    integer, anything else is refused rather than quietly replaced."""
    raw = _call_setting("OUROBOROS_PRESENCE_CALL_MAX_MINUTES")
    if not raw:
        return None
    try:
        minutes = int(raw)
    except ValueError:
        minutes = 0
    if minutes < 1:
        raise PresenceCallRefused("presence_call_invalid_setting",
                                  f"OUROBOROS_PRESENCE_CALL_MAX_MINUTES must be a positive integer or empty, not {raw!r}")
    return minutes


def _refuse_lost_or_settled(drive_root: Path, admission: PresenceAdmission, event: PresenceTurnEvent) -> str:
    """Under the lease: the task id this call runs as, or a refusal for an event already used.

    The same facts ``run_presence_turn`` checks under the conversation lock, read here so
    the door answers before a hub exists: a settled turn is not reopened, and a lost
    attempt (a running row while the lock was free) keeps its unknown outcome — no
    re-join, no second paid session.
    """
    first_id = _task_id(admission, event)
    identity = presence_event_identity(admission.binding_id, event)
    if presence_turn_replay(drive_root, first_id, event.conversation_key, identity, 0) is not None:
        raise PresenceCallRefused("presence_call_settled", "this event's turn already ended; it is not reopened")
    task_id = _retry_target(drive_root, first_id, identity)
    stored = _stored_turn(drive_root, task_id, identity)
    if is_reconciled_presence_placeholder(stored) or str(stored.get("status") or "") in {
            STATUS_RUNNING, STATUS_INTERRUPTED}:
        raise PresenceCallRefused("presence_attempt_outcome_unknown",
                                  "an earlier attempt of this event was lost with an unknown outcome; "
                                  "it is retained, not rerun")
    return task_id


def open_presence_call(
    *,
    drive_root: Path,
    repo_dir: Path,
    skill_name: str,
    binding_id: str,
    event: Mapping[str, Any],
    label: str = "",
    event_queue: Any = None,
    agent_factory: Callable[..., Any] | None = None,
    on_finished: Callable[[PresenceCallRun], None] | None = None,
) -> PresenceCallOpened:
    """Admit and start one live call for a Presence binding; returns once its thread runs.

    ``event`` is the same object ``/presence/turn`` takes. Refusals are
    ``PresenceCallRefused`` with the admission's own codes, ``presence_call_not_enabled``
    (the effective runtime ``execution`` is not ``call``), ``presence_call_busy`` (the
    conversation has a live turn or every active slot is taken), the retained-outcome
    codes, and ``presence_call_hub_unavailable``. The hub is opened before the turn starts
    and closed on any refusal after that point. ``agent_factory`` and ``on_finished`` are
    test and host seams; the public shape is the keyword set the Host door passes.
    """
    from ouroboros.call_protocol import OPENED_BY_SKILL, CallSpec

    drive_root = Path(drive_root)
    try:
        admission = admit_presence_turn(
            drive_root=drive_root, authenticated_transport_skill=skill_name, binding_id=binding_id,
            global_max_rounds=_global_max_rounds(),
        )
        turn_event = presence_turn_event(event, admission)
    except PresenceAdmissionError as exc:
        raise PresenceCallRefused(exc.code, str(exc)) from exc
    if admission.execution != EXECUTION_CALL:
        raise PresenceCallRefused(
            "presence_call_not_enabled",
            f"binding {binding_id} runs as text turns; set the behavior skill's presence runtime "
            "execution to call (profile runtime_defaults.execution or the owner override) to allow a live call",
        )
    engine, model, voice = _call_route()
    max_minutes = _call_max_minutes()

    gate = _configured_gate(drive_root)
    lease = gate.try_acquire(turn_event.conversation_key)
    if lease is None:
        raise PresenceCallRefused(
            "presence_call_busy",
            f"presence is busy: this conversation already has a live turn or every active slot is taken "
            f"({gate.active_count()} of {gate.max_active} active)",
        )
    call_id = f"call-{secrets.token_hex(12)}"
    try:
        try:
            task_id = _refuse_lost_or_settled(drive_root, admission, turn_event)
        except PresenceTurnError as exc:
            raise PresenceCallRefused(exc.code, str(exc)) from exc
        spec = CallSpec(
            call_id=call_id, chat_id=presence_chat_id(turn_event.conversation_key), engine=engine, model=model,
            voice=voice, opened_by=OPENED_BY_SKILL, presence_binding_id=admission.binding_id,
            max_minutes=max_minutes, label=str(label or ""), client_message_id=turn_event.source_event_id,
        )
        try:
            _open_hub(spec)
        except Exception as exc:
            raise PresenceCallRefused("presence_call_hub_unavailable", f"the call hub could not open: {exc}") from exc
        run = PresenceCallRun(spec=spec, task_id=task_id, lease=lease, ended=threading.Event())
        try:
            _start_call_thread(run, admission=admission, event=turn_event, repo_dir=Path(repo_dir),
                               drive_root=drive_root, event_queue=event_queue, agent_factory=agent_factory,
                               on_finished=on_finished)
        except BaseException as exc:
            _close_hub_quietly(call_id)
            raise PresenceCallRefused("presence_call_not_started", f"the call thread could not start: {exc}") from exc
    except BaseException:
        lease.release()
        raise
    return PresenceCallOpened(call_id=call_id, task_id=task_id)


def _global_max_rounds() -> int | None:
    from ouroboros.loop import _resolve_loop_max_rounds

    return _resolve_loop_max_rounds()


def _start_call_thread(run: PresenceCallRun, *, admission: PresenceAdmission, event: PresenceTurnEvent,
                       repo_dir: Path, drive_root: Path, event_queue: Any,
                       agent_factory: Callable[..., Any] | None,
                       on_finished: Callable[[PresenceCallRun], None] | None) -> None:
    """Run the turn on its own named thread; the lease is released on return, raise or stop."""
    call_id = run.spec.call_id

    def body() -> None:
        try:
            run.result = run_presence_turn(
                admission=admission, event=event, repo_dir=repo_dir, drive_root=drive_root,
                event_queue=event_queue, agent_factory=agent_factory, admitted=run.lease, live_call=run.spec,
            )
        except BaseException as exc:  # the fact is retained on the run; a thread has no caller to raise to
            run.error = exc
            log.warning("Presence call %s ended with %s: %s", call_id, type(exc).__name__, exc)
        finally:
            run.lease.release()
            _close_hub_quietly(call_id)
            with _RUNS_LOCK:
                _RUNS.pop(call_id, None)
            if run.ended is not None:
                run.ended.set()
            if on_finished is not None:
                try:
                    on_finished(run)
                except Exception:
                    log.warning("Presence call %s: on_finished failed", call_id, exc_info=True)

    with _RUNS_LOCK:
        _RUNS[call_id] = run
    thread = threading.Thread(target=contextvars.copy_context().run, args=(body,),
                              name=f"presence-call-{call_id}", daemon=True)
    run.thread = thread
    try:
        thread.start()
    except BaseException:
        with _RUNS_LOCK:
            _RUNS.pop(call_id, None)
        raise


def _event_from_task(task: Mapping[str, Any]) -> PresenceTurnEvent:
    presence = task["metadata"]["presence"]
    stored = presence["event"]
    return PresenceTurnEvent(
        source_event_id=str(stored["source_event_id"]), provider=str(stored["provider"]),
        account_id=str(stored["account_id"]), conversation_id=str(stored["conversation_id"]),
        thread_id=str(stored["thread_id"]), conversation_key=str(stored["conversation_key"]),
        actor=dict(stored.get("actor") or {}), conversation=dict(stored.get("conversation") or {}),
        message=dict(stored.get("message") or {}), text=str(presence.get("observed_text") or ""),
        delivery_reporting_version=int(presence.get("delivery_reporting_version") or 0),
    )


def record_call_utterance(drive_root: Path, task: Mapping[str, Any], *, direction: str, text: str,
                          speaker_label: str = "") -> bool:
    """Retain one finalized utterance of a live call as a Presence dialogue row.

    The driver calls this for every finalized correspondent utterance (``direction="in"``,
    attributed to the participant label the platform reported, else to "meeting audio")
    and for every utterance Ouroboros spoke (``direction="out"``). The row carries the
    call's task and conversation exactly like a text turn's rows, so history, memory and
    the previous-turn pointer read the call as one conversation. False means the row
    could not be written; the driver discloses that, the call is not ended for it.
    """
    text = str(text or "").strip()
    if direction not in {"in", "out"} or not text:
        return False
    event = _event_from_task(task)
    chat_id = int(task["chat_id"])
    if direction == "out":
        speaker, user_id = None, 0
    else:
        label = str(speaker_label or "").strip() or MEETING_AUDIO_LABEL
        speaker = {"display_name": label, "platform_actor_id": f"meeting-audio:{label}"}
        user_id = _stable_numeric_id("presence-actor-log", speaker["platform_actor_id"])
    try:
        _log_dialogue(Path(drive_root), direction=direction, chat_id=chat_id, user_id=user_id, text=text,
                      event=event, task=task, task_id=str(task["id"]), speaker=speaker)
    except PresenceTurnError:
        log.warning("Presence call %s: a %s utterance could not be retained", task.get("id"), direction)
        return False
    return True


__all__ = [
    "MEETING_AUDIO_LABEL",
    "PresenceCallOpened",
    "PresenceCallRefused",
    "PresenceCallRun",
    "open_presence_call",
    "presence_call_run",
    "record_call_utterance",
]
