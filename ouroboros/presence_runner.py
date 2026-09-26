"""Fresh-agent execution for bounded, host-admitted presence turns."""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextvars
import hashlib
import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ouroboros.artifacts import stage_task_attachments
from ouroboros.contracts.task_contract import attach_task_contract
from ouroboros.presence_admission import PresenceAdmission
from ouroboros.presence_authority import presence_ceiling_payload
from ouroboros.task_results import (
    STATUS_CANCELLED,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_INTERRUPTED,
    STATUS_RUNNING,
    is_reconciled_presence_placeholder,
    load_task_result,
    task_result_path,
)
from ouroboros.utils import append_jsonl, atomic_write_json, iter_jsonl_objects, read_json_dict, utc_now_iso

log = logging.getLogger(__name__)


class PresenceTurnError(ValueError):
    def __init__(
        self,
        code: str,
        field: str,
        *,
        attachment_manifest: Sequence[Mapping[str, Any]] = (),
        turn_ref: str = "",
        work_ref: str = "",
    ) -> None:
        self.code = str(code or "presence_turn_failed")
        self.field = str(field or "presence_turn")
        self.turn_ref = str(turn_ref or "")
        self.work_ref = str(work_ref or "")
        self.attachment_manifest = [
            dict(row) for row in attachment_manifest if isinstance(row, Mapping)
        ]
        super().__init__(f"{self.code}: {self.field}")


@dataclass(frozen=True)
class PresenceTurnEvent:
    source_event_id: str
    provider: str
    account_id: str
    conversation_id: str
    thread_id: str
    conversation_key: str
    actor: Mapping[str, Any]
    conversation: Mapping[str, Any]
    message: Mapping[str, Any]
    text: str
    delivery_reporting_version: int = 0


@dataclass(frozen=True)
class PresenceTurnResult:
    outcome: str
    text: str
    task_id: str
    work_ref: str = ""
    delivery_reporting_version: int = 0


def _presence_delivery(outcome: str, text: str, terminal_origin: str, *, legacy: bool = False) -> tuple[str, str]:
    """Project speech from producer facts, never from its wording or task status."""
    from ouroboros.task_finalization import HOST_AUTHORED_TERMINAL_ORIGINS, TERMINAL_ORIGIN_MODEL_FINAL

    if outcome not in {"message", "silent", "tool_delivered", "deferred"}:
        outcome = "message"
    if outcome not in {"message", "deferred"}:
        return outcome, ""
    # Unknown origin remains explicit stored-row compatibility, not evidence
    # authorizing new speech. Even an old frozen body cannot override known host provenance.
    authored = terminal_origin == TERMINAL_ORIGIN_MODEL_FINAL
    if not authored and not (legacy and terminal_origin not in HOST_AUTHORED_TERMINAL_ORIGINS):
        return ("deferred" if outcome == "deferred" else "silent"), ""
    return outcome, text


def presence_result_from_stored(stored: Mapping[str, Any], task_id: str) -> PresenceTurnResult:
    """One replay projection for cached turns and completed delegated work."""
    from ouroboros.task_finalization import TERMINAL_ORIGIN_MODEL_FINAL, provider_terminal_body, terminal_notice_text

    metadata = stored.get("metadata") if isinstance(stored.get("metadata"), dict) else {}
    text = metadata["presence_result_text"] if "presence_result_text" in metadata else stored.get("result")
    origin = str(stored.get("terminal_origin") or "")
    if origin == TERMINAL_ORIGIN_MODEL_FINAL and text and "result" in stored:
        raw, notice = str(stored["result"] or ""), terminal_notice_text(stored)
        # Undo only the recorded host composition. Older explicit reply bodies
        # could differ from the raw final; neither they nor deliberate emptiness are guessed away.
        if notice and text == provider_terminal_body(raw, notice):
            text = raw
    # Unknown-origin compatibility covers completed rows only: a failed row may
    # carry host text (an orphan reconcile or exception notice), never a reply.
    outcome, text = _presence_delivery(
        str(metadata.get("presence_outcome") or "message"), str(text or ""), origin,
        legacy=str(stored.get("status") or "") == "completed",
    )
    if _terminal_refusal(stored):
        # A refused or unproven attempt has no reply to deliver: the forced rail may
        # still stamp model_final over a round-one draft it salvaged before the quota
        # refusal, and the deferred-work view must not hand that draft to the correspondent.
        outcome, text = "silent", ""
    return PresenceTurnResult(
        outcome=outcome, text=text, task_id=task_id,
        work_ref=str(metadata.get("presence_work_ref") or ""),
        delivery_reporting_version=int((metadata.get("presence") or {}).get("delivery_reporting_version") == 1),
    )


def build_presence_result_event(task: dict[str, Any], text: str, ctx: Any, *, terminal_origin: str = "",
                                retain_scheduled_handoff: bool = False) -> dict[str, Any]:
    """Freeze typed delivery metadata before the ordinary durable result write."""
    from ouroboros.task_finalization import HOST_AUTHORED_TERMINAL_ORIGINS

    completion = getattr(ctx, "_presence_completion", None)
    completion = completion if (
        isinstance(completion, dict) and getattr(ctx, "_presence_completion_accepted", False)
    ) else {}
    # A forced final declares its outward delivery beside the internal record; without
    # a valid declaration the record never becomes conversation speech (owner Q4).
    declared = getattr(ctx, "_presence_forced_declaration", None)
    if not completion and isinstance(declared, dict):
        completion = declared if declared.get("status") == "declared" else {"outcome": "silent"}
        # Only a message/deferred body is speech; a tool_delivered note stays context even
        # when owed work below turns the outcome into deferred.
        text = str(completion.get("message") or "") if completion.get("outcome") in {"message", "deferred"} else ""
    note = str(completion.get("message") or "") if completion.get("outcome") == "tool_delivered" else ""
    outcome = str(completion.get("outcome") or "message").strip()
    handoff = getattr(ctx, "_swarm_handoff_attempt", None)
    handoff = handoff if isinstance(handoff, dict) else {}
    work_ref = (
        str(handoff.get("task_id") or "")
        if str(handoff.get("status") or "") == "scheduled"
        else ""
    )
    if outcome == "deferred" and not work_ref:
        outcome = "message"
    if work_ref and (retain_scheduled_handoff or terminal_origin in HOST_AUTHORED_TERMINAL_ORIGINS):
        # A failed/forced or host-replaced final still owes an admitted child's result.
        # Transports poll only deferred outcomes, even when no reply was authored.
        outcome = "deferred"
    outcome, result_text = _presence_delivery(outcome, str(text or ""), terminal_origin)
    metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    metadata["presence_outcome"] = outcome
    metadata["presence_result_text"] = result_text
    if work_ref:
        metadata["presence_work_ref"] = work_ref
    if not getattr(ctx, "_presence_completion_accepted", False) and isinstance(declared, dict):
        metadata["presence_declaration"] = {key: declared[key] for key in ("status", "reason") if declared.get(key)}
    task["metadata"] = metadata
    return {
        "type": "presence_result",
        "task_id": str(task.get("id") or ""),
        "outcome": outcome,
        "text": result_text,
        # Speech and the internal finish note stay separate even when owed work
        # changes tool_delivered into deferred for the polling contract.
        "message": result_text,
        **({"finish_note": note} if note else {}),
        "work_ref": work_ref,
        "ts": utc_now_iso(),
    }


_GATE_POLL_SEC = 0.05


def _try_exclusive(fd: int) -> bool:
    from ouroboros.platform_layer import file_lock_exclusive_nb

    try:
        file_lock_exclusive_nb(fd)
    except OSError:
        return False
    return True


class PresenceTurnLease:
    """The gate resources one admitted turn holds; its owner releases them exactly once."""

    def __init__(self, conversation_key: str, resources: ExitStack) -> None:
        self.conversation_key = conversation_key
        self._resources: ExitStack | None = resources
        self._lock = threading.Lock()

    def release(self) -> None:
        with self._lock:
            resources, self._resources = self._resources, None
        if resources is not None:
            resources.close()


class PresenceTurnGate:
    """Cross-process cap plus one active turn for each conversation.

    ``run`` waits on a thread (the tool path); ``admit`` takes the same resources as a
    coroutine for the Host, whose queued turns must hold no thread while they wait.
    """

    def __init__(self, max_active: int = 2, *, state_root: Path | None = None) -> None:
        self._max_active = max(1, int(max_active))
        self._slots = threading.BoundedSemaphore(max(1, int(max_active)))
        self._guard = threading.Lock()
        self._conversations: dict[str, threading.Lock] = {}
        self._state_root = Path(state_root).resolve(strict=False) if state_root is not None else None
        self._claimed_slots: set[int] = set()

    def _gate_file(self, name: str) -> Path:
        root = self._state_root / "presence_turn_gate"
        root.mkdir(parents=True, exist_ok=True)
        return root / name

    def _conversation(self, conversation_key: str) -> tuple[str, threading.Lock]:
        key = str(conversation_key or "").strip()
        if not key:
            raise PresenceTurnError("presence_conversation_key_required", "conversation_key")
        with self._guard:
            return key, self._conversations.setdefault(key, threading.Lock())

    def _conversation_file(self, key: str) -> Path:
        return self._gate_file(f"conversation-{hashlib.sha256(key.encode('utf-8')).hexdigest()}.lock")

    @contextmanager
    def _file_lock(self, path: Path):
        from ouroboros.platform_layer import file_lock_exclusive, file_unlock

        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            file_lock_exclusive(fd)
            yield
        finally:
            try:
                file_unlock(fd)
            finally:
                os.close(fd)

    def _try_slot(self) -> Callable[[], None] | None:
        """Claim one free active slot without waiting: its release, or None while all are taken."""
        if self._state_root is None:
            return self._slots.release if self._slots.acquire(blocking=False) else None
        from ouroboros.platform_layer import file_unlock

        for index in range(self._max_active):
            with self._guard:
                if index in self._claimed_slots:
                    continue
                self._claimed_slots.add(index)
            try:
                fd = os.open(self._gate_file(f"slot-{index}.lock"), os.O_CREAT | os.O_RDWR, 0o600)
            except BaseException:  # an unopenable slot is an error, never a slot left claimed
                with self._guard:
                    self._claimed_slots.discard(index)
                raise
            if not _try_exclusive(fd):  # another process holds it
                os.close(fd)
                with self._guard:
                    self._claimed_slots.discard(index)
                continue

            def release(index: int = index, fd: int = fd) -> None:
                try:
                    file_unlock(fd)
                finally:
                    os.close(fd)
                    with self._guard:
                        self._claimed_slots.discard(index)

            return release
        return None

    @contextmanager
    def _file_slot(self):
        while (release := self._try_slot()) is None:
            time.sleep(_GATE_POLL_SEC)
        try:
            yield
        finally:
            release()

    def run(self, conversation_key: str, callback: Callable[[], PresenceTurnResult]) -> PresenceTurnResult:
        key, conversation_lock = self._conversation(conversation_key)
        with conversation_lock:
            if self._state_root is None:
                with self._slots:
                    return callback()
            with self._file_lock(self._conversation_file(key)):
                with self._file_slot():
                    return callback()

    async def admit(self, conversation_key: str) -> PresenceTurnLease:
        """Take what ``run`` takes, in the same order, as a coroutine.

        Every attempt is non-blocking and every wait is an asyncio sleep, so a turn queued
        behind its conversation or the active cap parks no thread. Cancellation releases
        whatever was already taken; success hands it all to the returned lease.
        """
        from ouroboros.platform_layer import file_unlock

        key, conversation_lock = self._conversation(conversation_key)
        with ExitStack() as held:
            while not conversation_lock.acquire(blocking=False):
                await asyncio.sleep(_GATE_POLL_SEC)
            held.callback(conversation_lock.release)
            if self._state_root is not None:
                fd = os.open(self._conversation_file(key), os.O_CREAT | os.O_RDWR, 0o600)
                held.callback(os.close, fd)
                while not _try_exclusive(fd):
                    await asyncio.sleep(_GATE_POLL_SEC)
                held.callback(file_unlock, fd)
            while (release_slot := self._try_slot()) is None:
                await asyncio.sleep(_GATE_POLL_SEC)
            held.callback(release_slot)
            return PresenceTurnLease(key, held.pop_all())


_GATES_LOCK = threading.Lock()
_GATES: dict[tuple[str, int], PresenceTurnGate] = {}
_LIVE_LOCK = threading.Lock()
_LIVE_PRESENCE_TASKS: set[str] = set()


def presence_turn_is_live(task_id: str) -> bool:
    """Process-local liveness for the orphan reconciler; never an owner-addressable actor."""
    with _LIVE_LOCK:
        return str(task_id or "") in _LIVE_PRESENCE_TASKS


def _configured_gate(drive_root: Path | None = None) -> PresenceTurnGate:
    from ouroboros.config import SETTINGS_DEFAULTS, _bounded_positive_int_setting

    limit = _bounded_positive_int_setting(
        "OUROBOROS_PRESENCE_MAX_ACTIVE",
        default=int(SETTINGS_DEFAULTS["OUROBOROS_PRESENCE_MAX_ACTIVE"]),
        hard_max=20,
    )
    state_root = Path(drive_root).resolve(strict=False) / "state" if drive_root is not None else None
    key = (str(state_root or ""), limit)
    with _GATES_LOCK:
        return _GATES.setdefault(key, PresenceTurnGate(limit, state_root=state_root))


async def admit_configured_gate(drive_root: Path, conversation_key: str) -> PresenceTurnLease:
    """Coroutine admission into the configured gate ``run_presence_turn`` would otherwise take."""
    return await _configured_gate(Path(drive_root)).admit(conversation_key)


def _stable_numeric_id(prefix: str, value: str) -> int:
    digest = hashlib.sha256(f"{prefix}\0{value}".encode("utf-8")).digest()
    return (1 << 40) + (int.from_bytes(digest[:6], "big") & ((1 << 40) - 1))


def presence_turn_task_id(binding_id: str, source_event_id: str) -> str:
    """The durable id of one turn; the Host joins a retry to its live execution by it."""
    digest = hashlib.sha256(f"{binding_id}\0{source_event_id}".encode("utf-8")).hexdigest()
    return f"presence-{digest[:24]}"


def presence_retry_proof(task: Mapping[str, Any], usage: Mapping[str, Any],
                         trace: Mapping[str, Any], ctx: Any) -> dict[str, str]:
    """Certificate for the narrow first-round engine-not-started refusal.

    Missing evidence, a received response, an earlier tool, a handoff, or an
    undated refusal cannot mint it. A private host diagnostic is not speech;
    the absence of a delivery receipt is NOT used as proof of no effect.
    """
    refusal = usage.get("resource_refusal")
    metadata = task.get("metadata") if isinstance(task.get("metadata"), Mapping) else {}
    if not (ctx is not None and usage.get("_presence_pre_dispatch_only") is True
            and isinstance(refusal, Mapping) and refusal.get("temporary") is True
            and not usage.get("rounds") and not trace.get("tool_calls")
            and not getattr(ctx, "_swarm_handoff_attempt", None)
            and isinstance(metadata.get("presence_event_identity"), str)):
        return {}
    reset_at = str(refusal.get("reset_at") or "")
    try:
        reset = datetime.fromisoformat(reset_at.replace("Z", "+00:00"))
    except ValueError:
        return {}
    if reset.tzinfo is None or reset <= datetime.now(timezone.utc):
        return {}
    return {"kind": "first_round_engine_not_started", "event_identity": metadata["presence_event_identity"],
            "reset_at": reset_at, "task_id": str(task.get("id") or "")}


def presence_unknown_outcome(usage: Mapping[str, Any]) -> dict[str, str]:
    """The durable marker of a dispatched attempt whose outcome the loop never resolved.

    Spelled by the loop's own no-resend predicate (``provider_no_call_source``: the sticky
    ``provider_outcome_unknown`` kind, or a round still holding a transport-death record;
    a usable response clears both), so the forced rail and the Host guard answer one
    question the same way. Neither the rail's fallback text nor a draft it salvaged as
    ``model_final`` answers the event; the trace source alone cannot say so, because a
    retained draft is recorded under its own source.
    """
    from ouroboros.loop_llm_call import provider_no_call_source

    source, _wall = provider_no_call_source(dict(usage), False)
    if source != "provider_outcome_unknown_no_resend":
        return {}
    pending = usage.get("_pending_transport_outcome")
    pending = pending if isinstance(pending, Mapping) else {}
    return {"source": source, "error_kind": str(usage.get("_last_llm_error_kind") or ""),
            "operation_id": str(pending.get("operation_id") or "")}


def _successor_id(task_id: str) -> str:
    return "presence-" + hashlib.sha256(f"presence-retry\0{task_id}".encode("utf-8")).hexdigest()[:24]


def _retry_target(drive_root: Path, first_id: str, identity: str, *, claim: bool = False) -> str:
    """Follow immutable predecessor links; only the conversation-lock holder may claim one.

    The current failed row stays intact. A crashed claim points at an unused
    successor, whose own RUNNING barrier still precedes every model/tool effect.
    """
    from ouroboros.task_results import write_task_result

    current_id = first_id
    visited: set[str] = set()
    while current_id not in visited:
        visited.add(current_id)
        row = _stored_turn(drive_root, current_id, identity)
        next_id = str(row.get("presence_retry_next") or "")
        if next_id:
            if next_id != _successor_id(current_id):
                raise PresenceTurnError("presence_result_unreadable", "source_event_id", turn_ref=current_id)
            current_id = next_id
            continue
        if str(row.get("reason_code") or "") != "resource_refusal_no_resend":
            return current_id
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        proof = metadata.get("presence_retry_proof")
        if not (isinstance(proof, dict) and proof.get("kind") == "first_round_engine_not_started"
                and proof.get("event_identity") == identity and proof.get("task_id") == current_id):
            return current_id
        try:
            reset = datetime.fromisoformat(str(proof.get("reset_at") or "").replace("Z", "+00:00"))
        except ValueError:
            return current_id
        if reset.tzinfo is None or reset > datetime.now(timezone.utc) or not claim:
            return current_id
        successor = _successor_id(current_id)

        def link(existing: dict, _fields: dict) -> dict | None:
            _assert_event_identity(existing, current_id, identity)
            if existing.get("presence_retry_next") == successor:
                return None
            existing_meta = existing.get("metadata") if isinstance(existing.get("metadata"), dict) else {}
            if (existing.get("status") != STATUS_FAILED or
                    existing.get("reason_code") != "resource_refusal_no_resend" or
                    existing_meta.get("presence_retry_proof") != proof or
                    existing.get("presence_retry_next")):
                raise PresenceTurnError("presence_result_unreadable", "source_event_id", turn_ref=current_id)
            return {"presence_retry_next": successor}

        try:
            write_task_result(drive_root, current_id, STATUS_FAILED,
                              strict_existing_dict=True, _field_projector=link)
        except (OSError, ValueError) as exc:
            if isinstance(exc, PresenceTurnError):
                raise
            raise PresenceTurnError("presence_result_unreadable", "source_event_id", turn_ref=current_id) from exc
        if _stored_turn(drive_root, current_id, identity).get("presence_retry_next") != successor:
            raise PresenceTurnError("presence_result_unreadable", "source_event_id", turn_ref=current_id)
        current_id = successor
    raise PresenceTurnError("presence_result_unreadable", "source_event_id", turn_ref=current_id)


def _task_id(admission: PresenceAdmission, event: PresenceTurnEvent) -> str:
    return presence_turn_task_id(admission.binding_id, event.source_event_id)


def presence_event_identity(binding_id: str, event: PresenceTurnEvent) -> str:
    """Stable source-event identity; retry-local file paths and reporting negotiation are not identity."""
    facts = (binding_id, event.source_event_id, event.provider, event.account_id,
             event.conversation_id, event.thread_id,
             str(event.actor.get("platform_actor_id") or event.actor.get("id") or ""), event.text)
    return hashlib.sha256(json.dumps(facts, ensure_ascii=False).encode("utf-8")).hexdigest()


def _assert_event_identity(stored: Mapping[str, Any], task_id: str, identity: str) -> None:
    if not stored or not identity:
        return
    metadata = stored.get("metadata") if isinstance(stored.get("metadata"), dict) else {}
    prior = str(metadata.get("presence_event_identity") or "")
    presence = metadata.get("presence") if isinstance(metadata.get("presence"), dict) else {}
    old = presence.get("event") if isinstance(presence.get("event"), dict) else {}
    if not prior and old and all(old.get(key) is not None for key in (
            "source_event_id", "provider", "account_id", "conversation_id", "thread_id", "actor")):
        # Older turns already stored the exact event and observed text, before a separate
        # digest was introduced. Reconstruct its identity instead of stranding valid replies.
        prior = presence_event_identity(str(presence.get("binding_id") or ""), PresenceTurnEvent(
            source_event_id=str(old["source_event_id"]), provider=str(old["provider"]),
            account_id=str(old["account_id"]), conversation_id=str(old["conversation_id"]),
            thread_id=str(old["thread_id"]), conversation_key=str(old.get("conversation_key") or ""),
            actor=old["actor"] if isinstance(old["actor"], Mapping) else {},
            conversation={}, message={}, text=str(presence.get("observed_text") or "")))
    if prior and prior != identity:
        raise PresenceTurnError("presence_event_identity_conflict", "source_event_id", turn_ref=task_id)
    if (not prior and not is_reconciled_presence_placeholder(stored)
            and str(stored.get("status") or "") in {STATUS_COMPLETED, STATUS_FAILED}):
        # An incomplete legacy terminal cannot authorize cross-room replay.
        raise PresenceTurnError("presence_event_identity_unproven", "source_event_id", turn_ref=task_id)


def _stored_turn(drive_root: Path, task_id: str, identity: str = "") -> dict[str, Any]:
    """Read Presence authority without converting an unreadable row into a new event.

    A generic fail-soft read can quarantine a malformed task result and return None.
    That is correct for a list, but a transport retry must not treat an already
    admitted event as never started. Quarantine keeps the original id occupied.
    """
    from ouroboros.task_result_schema import TASK_RESULT_QUARANTINE_DIR

    path = task_result_path(drive_root, task_id, create=False)
    quarantined = path.parent / TASK_RESULT_QUARANTINE_DIR
    try:
        current = load_task_result(drive_root, task_id, strict=True) or {}
        if (quarantined / path.name).exists() or any(quarantined.glob(f"{task_id}.*.json")):
            # A strictly admitted, source-bound late terminal is newer authority than an old
            # quarantined attempt. Nothing else, including an unrelated or running row, clears it.
            if str(current.get("status") or "") not in {STATUS_COMPLETED, STATUS_FAILED} or not identity:
                raise PresenceTurnError("presence_result_unreadable", "source_event_id", turn_ref=task_id)
            _assert_event_identity(current, task_id, identity)
        _assert_event_identity(current, task_id, identity)
        return current
    except (OSError, ValueError) as exc:
        if isinstance(exc, PresenceTurnError):
            raise
        raise PresenceTurnError("presence_result_unreadable", "source_event_id", turn_ref=task_id) from exc


def _terminal_refusal(stored: Mapping[str, Any]) -> str:
    """The typed refusal a durable Presence row demands before its event may be acknowledged.

    The durable terminal cause decides, never a draft or the in-memory envelope. A confirmed
    resource refusal keeps the event with the transport (``presence_resources_unavailable``).
    A row without a canonical terminal of its own (RUNNING/INTERRUPTED, a terminal write that
    failed after the start barrier, or the host's reconcile of such a row: the orphan
    placeholder or a stale row marked from its own already-failed axes) and a terminal under
    the unknown-outcome fence (the pipeline's ``presence_unknown_outcome`` marker, stamped from
    the loop's own no-resend predicate: a dispatched attempt stayed unresolved; the forced rail
    words it ``provider_unavailable`` and may salvage a round-one draft as ``model_final``) are
    attempts whose external effect is unproven (``presence_attempt_outcome_unknown``): no model
    answered this event, so completed/silent would let the adapter drop it. Every other
    infrastructure terminal (a confirmed outage, an overflow) is not refused here: it keeps the
    deferred projection, where ``_presence_delivery`` decides speech from terminal authorship
    and an admitted child stays pollable. Empty: the row may answer.
    """
    if str(stored.get("reason_code") or "") == "resource_refusal_no_resend":
        return "presence_resources_unavailable"
    if (str(stored.get("status") or "") not in {STATUS_COMPLETED, STATUS_FAILED}
            or str(stored.get("status_reconciled_from") or "") in {STATUS_RUNNING, STATUS_INTERRUPTED}):
        return "presence_attempt_outcome_unknown"
    metadata = stored.get("metadata") if isinstance(stored.get("metadata"), dict) else {}
    if isinstance(metadata.get("presence_unknown_outcome"), dict):
        return "presence_attempt_outcome_unknown"
    return ""


def _cached_result(drive_root: Path, task_id: str, identity: str = "") -> PresenceTurnResult | None:
    if identity:
        task_id = _retry_target(drive_root, task_id, identity)
    stored = _stored_turn(drive_root, task_id, identity)
    refusal = _terminal_refusal(stored)
    if refusal == "presence_resources_unavailable":
        metadata = stored.get("metadata") if isinstance(stored.get("metadata"), dict) else {}
        if identity and isinstance(metadata.get("presence_retry_proof"), dict):
            proof = metadata["presence_retry_proof"]
            try:
                reset = datetime.fromisoformat(str(proof.get("reset_at") or "").replace("Z", "+00:00"))
            except ValueError:
                reset = None
            if (proof.get("kind") == "first_round_engine_not_started" and
                    proof.get("event_identity") == identity and proof.get("task_id") == task_id and
                    reset is not None and reset.tzinfo is not None and reset <= datetime.now(timezone.utc)):
                return None  # the conversation-lock holder claims its successor
        # A failed quota/fallback attempt is not a completed/silent transport answer;
        # retain the event with its adapter and never replay a draft as external speech.
        _notify_unresolved_turn(drive_root, task_id)
        raise PresenceTurnError("presence_resources_unavailable", "source_event_id", turn_ref=task_id,
                                work_ref=str(metadata.get("presence_work_ref") or ""))
    if refusal and (str(stored.get("status") or "") not in {STATUS_COMPLETED, STATUS_FAILED}
                    or is_reconciled_presence_placeholder(stored)):
        return None  # a host-lost turn is not a result; the later admission guard refuses regeneration
    if refusal:
        # A failed infrastructure terminal never earns a retry certificate: unknown is not
        # not_started. Retain the event and any already scheduled work; ask the owner once.
        metadata = stored.get("metadata") if isinstance(stored.get("metadata"), dict) else {}
        _notify_unresolved_turn(drive_root, task_id)
        raise PresenceTurnError(refusal, "source_event_id", turn_ref=task_id,
                                work_ref=str(metadata.get("presence_work_ref") or ""))
    return presence_result_from_stored(stored, task_id)


def _notify_unresolved_turn(drive_root: Path, task_id: str) -> None:
    """Serialize one owner's recovery notice across concurrent Host replay readers.

    Reuse the per-conversation gate's process and file locks, keyed by the physical
    task id, without taking an execution slot. A crash after the required chat
    write but before the stamp may still repeat a question; it cannot mark an
    undelivered question as delivered. No new notification authority is stored.
    """
    try:
        gate = _configured_gate(drive_root)
        key, local_lock = gate._conversation(f"owner-notice:{task_id}")
        with local_lock, gate._file_lock(gate._conversation_file(key)):
            _write_unresolved_notice(drive_root, task_id)
    except Exception:
        log.warning("Presence recovery question lock failed for %s", task_id, exc_info=True)


def _write_unresolved_notice(drive_root: Path, task_id: str) -> None:
    stored = load_task_result(drive_root, task_id) or {}
    if stored.get("presence_recovery_owner_notified"):
        return
    try:
        from ouroboros.contracts.chat_id_policy import WEB_UI_CHAT_ID
        from supervisor import message_bus

        # Unit callers and unrelated installations must not write via a process-global
        # bridge bound to another data root. The real Host shares this canonical root.
        if message_bus.DATA_DIR is None or Path(message_bus.DATA_DIR).resolve() != drive_root.resolve():
            return
        if message_bus.try_get_bridge() is None:
            return
        message_bus.send_with_budget(
            WEB_UI_CHAT_ID,
            (f"Presence turn {task_id} has no available model route. Its original event remains "
             "with the transport; check the subscription account or configured fallback in Main."
             if str(stored.get("reason_code") or "") == "resource_refusal_no_resend" else
             f"Presence turn {task_id} has an unconfirmed previous effect. Its original "
             "event remains with the transport; I will not start another model or tool "
             "attempt automatically. Please answer in Main after checking the prior "
             "operation and external delivery, so we can decide how to recover it."),
            role="system", system_type="presence_recovery_required", require_write=True,
            ensure_record_boundary=True,
        )
        from ouroboros.task_results import write_task_result

        write_task_result(drive_root, task_id, str(stored["status"]), strict_existing_dict=True,
                          presence_recovery_owner_notified=utc_now_iso())
    except Exception:
        log.warning("Presence recovery question could not be recorded for %s", task_id, exc_info=True)


def _live_task_rows(drive_root: Path, task_id: str, conversation_key: str) -> list[dict[str, Any]]:
    """This task's rows in the live chat generation, in order; an attempt starts with its inbound row.

    A receipt addressed to another conversation (a tool send elsewhere) is not this conversation's
    delivery, however similar its text, so only receipts carrying this key are kept.
    """
    rows = []
    for row in iter_jsonl_objects(Path(drive_root) / "logs" / "chat.jsonl"):
        if row.get("task_id") != task_id:
            continue
        transport = row.get("transport") if isinstance(row.get("transport"), Mapping) else {}
        if row.get("type") == "presence_delivery" and str(transport.get("conversation_key") or "") != conversation_key:
            continue
        rows.append(row)
    return rows


def _turn_sends(rows: Sequence[Mapping[str, Any]], own_start: int = 0,
                same_generation: bool = False) -> tuple[list[str] | None, int]:
    """``(confirmed v1 sends or None, parts whose latest receipt is uncertain)`` among *rows*.

    A later receipt for a part settles an earlier one. An attempt's rows follow its inbound row, so
    that row in the live file means every receipt is there too and the whole task counts. Without
    it a rotated archive may hold receipts: the sends are unknown (None), not zero, unless the
    caller vouches that this execution's own rows (``own_start`` onwards) all landed in the
    generation that was live when it started (``same_generation``); a rotation during the execution
    leaves them unknown. Uncertain parts are neither confirmed nor refused, so they may have landed.
    """
    if any(row.get("direction") == "in" for row in rows):
        own_start = 0
    elif not same_generation:
        return None, 0
    latest: dict[tuple[str, str], tuple[str, str]] = {}
    for row in rows[own_start:]:
        transport = row.get("transport") if isinstance(row.get("transport"), Mapping) else {}
        delivery = transport.get("delivery") if isinstance(transport.get("delivery"), Mapping) else {}
        if row.get("type") == "presence_delivery" and delivery.get("state"):
            latest[(str(delivery.get("delivery_id")), str(delivery.get("part_id")))] = (
                str(delivery["state"]), str(row.get("text") or ""))
    return ([text for state, text in latest.values() if state in {"delivered", "accepted"}],
            sum(1 for state, _text in latest.values() if state == "uncertain"))


def _previous_turn_path(drive_root: Path, conversation_key: str) -> Path:
    digest = hashlib.sha256(conversation_key.encode("utf-8")).hexdigest()
    return Path(drive_root) / "state" / "presence_turn_gate" / f"last-{digest}.json"


def _read_previous_turn(drive_root: Path, conversation_key: str) -> dict[str, Any] | None:
    """Last executed turn of this exact conversation; a rebuildable projection."""
    row = read_json_dict(_previous_turn_path(drive_root, conversation_key)) or {}
    return row if row.get("conversation_key") == conversation_key else None


def _previous_turn_source_view(drive_root: Path, pointer: dict[str, Any]) -> dict[str, Any]:
    """Read a legacy deferred pointer's speech from its canonical result, without rewriting history.

    Old tool-delivered notes used the same `message` slot as replies. Owed work
    changed their outcome to deferred, hiding that provenance. A missing source
    proves neither speech nor a note; the context must say it is unverified.
    """
    if (pointer.get("outcome") != "deferred" or not pointer.get("message")
            or "finish_note" in pointer):
        return pointer
    row = load_task_result(drive_root, str(pointer.get("task_id") or "")) or {}
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    observed = metadata.get("presence_result_text")
    if metadata.get("presence_outcome") == "deferred" and isinstance(observed, str):
        if observed == pointer["message"]:
            return pointer  # source proves an authored partial reply, not a note
        if not observed:
            return {**pointer, "message": "", "finish_note": pointer["message"]}
    return {**pointer, "message": "", "previous_text_unverified": True}


def _write_previous_turn(drive_root: Path, conversation_key: str, task_id: str, *, outcome: str, message: str,
                         sends: Sequence[str], work_ref: str, finished_at: str, delivery: str,
                         finish_note: str = "") -> None:
    """Best effort: the pointer is a projection, and a turn that already answered is not failed over it."""
    try:
        atomic_write_json(_previous_turn_path(drive_root, conversation_key), {
            "conversation_key": conversation_key, "task_id": task_id, "outcome": outcome, "message": message,
            **({"finish_note": finish_note} if finish_note else {}),
            "transport_sends": [text for text in sends if text], "work_ref": work_ref,
            "finished_at": finished_at, "delivery": delivery,
        })
    except OSError:
        log.warning("presence previous-turn pointer not written for %s (task %s); the next replay of this turn "
                    "rebuilds it", conversation_key, task_id, exc_info=True)


def _pointer_behind(drive_root: Path, conversation_key: str, task_id: str) -> bool:
    """A settled turn whose pointer never landed (lost between its terminal write and the pointer write).

    Completed rows and authored failed rows (a model-final reply on a failed task) both replay speech;
    a host placeholder and a host-authored failure never do, so they are not turns to point at.
    """
    from ouroboros.task_finalization import TERMINAL_ORIGIN_MODEL_FINAL

    stored = load_task_result(drive_root, task_id) or {}
    status = str(stored.get("status") or "")
    if status != STATUS_COMPLETED and not (
            status == STATUS_FAILED and str(stored.get("terminal_origin") or "") == TERMINAL_ORIGIN_MODEL_FINAL):
        return False
    pointer = _read_previous_turn(drive_root, conversation_key)
    return pointer is None or (
        pointer.get("task_id") != task_id and str(pointer.get("finished_at") or "") <= str(stored.get("ts") or ""))


def presence_turn_replay(drive_root: Path, task_id: str, conversation_key: str,
                         identity: str = "") -> PresenceTurnResult | None:
    """A settled turn's durable answer, returned without the gate; None when the turn must (re)run."""
    cached = _cached_result(Path(drive_root), task_id, identity)
    return None if cached is None or _pointer_behind(Path(drive_root), conversation_key, cached.task_id) else cached


def _delivery_state(reporting_version: int, sends: Sequence[str] | None, text: str) -> str:
    """One rule for the live write and the repair.

    Tool sends confirm mid-turn; the adapter-delivered reply's receipt arrives only after the turn
    returns, so a reply text that is not among the confirmed sends is at most partly confirmed.
    """
    if not reporting_version or sends is None or not (sends or text):
        return "unknown"  # v0 never confirms; None = this turn's receipts left the live generation
    if not sends:
        return "authored"
    return "confirmed" if not text or text in sends else "partly confirmed"


def _log_dialogue(
    drive_root: Path,
    *,
    direction: str,
    chat_id: int,
    user_id: int,
    text: str,
    event: PresenceTurnEvent,
    task: Mapping[str, Any],
    task_id: str,
) -> None:
    from ouroboros.dialogue_provenance import presence_provenance_from_task

    state = read_json_dict(drive_root / "state" / "state.json") or {}
    written = append_jsonl(
        drive_root / "logs" / "chat.jsonl",
        ensure_record_boundary=(direction == "in"),  # a re-run never re-logs: this row must be parseable
        obj={
            "ts": utc_now_iso(),
            "session_id": state.get("session_id"),
            "direction": direction,
            "chat_id": chat_id,
            "user_id": user_id,
            "text": text,
            "format": "markdown" if direction == "out" else "",
            "source": f"presence:{event.provider}",
            "sender_label": str(event.actor.get("display_name") or event.actor.get("username") or ""),
            "sender_session_id": str(event.actor.get("platform_actor_id") or event.actor.get("id") or ""),
            "client_message_id": event.source_event_id,
            "presence_event_identity": presence_event_identity(str(task["metadata"]["presence"]["binding_id"]), event),
            "transport": {
                "provider": event.provider,
                "account_id": event.account_id,
                "conversation_id": event.conversation_id,
                "thread_id": event.thread_id,
                "conversation_key": event.conversation_key,
                "actor": dict(event.actor),
                "conversation": dict(event.conversation),
                "message": dict(event.message),
                **({"delivery": {"state": "authored"}} if direction == "out" else {}),
            },
            "presence_provenance": presence_provenance_from_task(task),
            "task_id": task_id,
        },
    )
    if direction == "in" and written is False:
        # A re-run of a lost attempt relies on the inbound row having landed (it never re-logs), so
        # a turn whose row cannot be written fails here, before the model runs; the transport retries.
        raise PresenceTurnError("chat_log_unwritable", "chat")


def _build_task(
    admission: PresenceAdmission,
    event: PresenceTurnEvent,
    *,
    drive_root: Path,
    staged_files: Sequence[Path],
    physical_task_id: str = "",
    lost_attempt: bool = False,
    prior_rows: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    from ouroboros.config import runtime_setting

    task_id = physical_task_id or _task_id(admission, event)
    chat_id = _stable_numeric_id("presence-conversation", event.conversation_key)
    actor_id = _stable_numeric_id(
        "presence-actor",
        f"{event.provider}:{event.account_id}:{event.actor.get('platform_actor_id') or event.actor.get('id') or ''}",
    )
    presence_context = {
        "binding_id": admission.binding_id,
        "transport_skill": admission.transport_skill,
        "behavior_skill": admission.behavior_skill,
        "profile_fingerprint": admission.profile_fingerprint,
        "instructions": admission.instructions,
        "context_topics": list(admission.context_topics),
        "observed_text": str(event.text or ""),
        "delivery_reporting_version": event.delivery_reporting_version,
        "event": {
            "source_event_id": event.source_event_id,
            "provider": event.provider,
            "account_id": event.account_id,
            "conversation_id": event.conversation_id,
            "thread_id": event.thread_id,
            "conversation_key": event.conversation_key,
            "actor": dict(event.actor),
            "conversation": dict(event.conversation),
            "message": dict(event.message),
            "origin": admission.origin.__dict__,
            "destination": admission.destination.__dict__,
        },
    }
    previous_turn = _read_previous_turn(drive_root, event.conversation_key)
    if previous_turn:
        previous_turn = _previous_turn_source_view(drive_root, previous_turn)
        if previous_turn.get("work_ref"):   # the deferred child's fate is read from its canonical row, never stored
            work_ref = str(previous_turn["work_ref"])
            child = load_task_result(drive_root, work_ref) or {}
            status = str(child.get("status") or "")
            answered = presence_result_from_stored(child, work_ref).text if status == STATUS_COMPLETED else ""
            record = str(child.get("result") or "") if status == STATUS_COMPLETED and not answered else ""
            previous_turn = {**previous_turn, "work_status": status or "absent", "work_result": answered,
                             "work_record": record[:300] + (" …(truncated)" if len(record) > 300 else "")}
        presence_context["previous_turn"] = previous_turn
    if lost_attempt:
        # Unknown (None) when the transport reports no receipts or the attempt's rows left the live generation.
        sent, uncertain = _turn_sends(prior_rows) if event.delivery_reporting_version else (None, 0)
        presence_context["previous_attempt"] = {
            "delivered_count": None if sent is None else len(sent),
            "delivered": None if sent is None else [text for text in sent if text],
            "uncertain_count": uncertain,
        }
    metadata: dict[str, Any] = {
        "source": "presence",
        "client_message_id": event.source_event_id,
        "presence_event_identity": presence_event_identity(admission.binding_id, event),
        "inline_max_rounds": admission.inline_max_rounds,
        "presence": presence_context,
    }
    if admission.model_slot == "light":
        from ouroboros.config import get_light_model

        metadata["model"] = get_light_model()
        metadata["use_local_model"] = runtime_setting("USE_LOCAL_LIGHT", "").lower() in {"true", "1"}
    task: dict[str, Any] = {
        "id": task_id,
        "type": "presence",
        "chat_id": chat_id,
        "actor_id": str(event.actor.get("platform_actor_id") or event.actor.get("id") or actor_id),
        "text": str(event.text or "").strip(),
        "_is_direct_chat": True,
        "_presence_turn": True,
        "context_requires_development": False,
        "metadata": metadata,
        "task_contract": {"capability_ceiling": presence_ceiling_payload(admission.capability_ceiling)},
    }
    if admission.workspace_root:
        task.update(
            workspace_root=admission.workspace_root,
            workspace_mode="external",
            memory_mode="shared",
        )
    manifest = stage_task_attachments(
        drive_root,
        task_id,
        [{"path": str(path), "label": path.name} for path in staged_files],
    )
    # Partial staging is the default for initial-task ingress (В25c, capinv-447):
    # good attachments stage, rejected ones ride along as disclosed manifest
    # rows — mirrors the gateway task API default. A FULLY-rejected set stays
    # atomic: the turn would run with none of its declared material.
    if manifest:
        from ouroboros.artifacts import (
            attachment_manifest_all_rejected,
            remove_staged_attachments,
        )

        if attachment_manifest_all_rejected(manifest):
            remove_staged_attachments(manifest)
            raise PresenceTurnError(
                "presence_attachment_admission_rejected",
                "staged_files",
                attachment_manifest=manifest,
            )
    if manifest:
        from ouroboros.gateway.tasks import _render_attachment_lines

        # The manifest is task authority, not merely presentation prose.  Keep
        # every staged/rejected declaration on the canonical carrier before the
        # task contract is normalized so a later promotion or child can inherit
        # and materialize the exact inputs.
        from ouroboros.artifacts import attachment_manifest_projection
        authority = attachment_manifest_projection(drive_root, task_id, manifest)
        task.update(authority)
        task["attachments"] = authority["attachment_manifest"]
        task["attachment_images"] = [
            dict(item) for item in manifest
            if str(item.get("status") or "staged") == "staged" and item.get("is_image")
        ]
        rendered = _render_attachment_lines(authority)
        if rendered:
            task["text"] = f"{task['text']}\n\n[ATTACHMENTS]\n{rendered}\n[END_ATTACHMENTS]".strip()
    if not task["text"]:
        task["text"] = "(attachments received)" if manifest else "(empty presence event)"
    return attach_task_contract(task)


def run_presence_turn(
    *,
    admission: PresenceAdmission,
    event: PresenceTurnEvent,
    repo_dir: Path,
    drive_root: Path,
    staged_files: Sequence[Path] = (),
    event_queue: Any = None,
    agent_factory: Callable[..., Any] | None = None,
    gate: PresenceTurnGate | None = None,
    admitted: PresenceTurnLease | None = None,
) -> PresenceTurnResult:
    """Run one bounded turn; adapters retain durable provider custody.

    ``admitted`` is a lease the caller already took for this conversation (``PresenceTurnGate.admit``):
    the turn runs under it instead of waiting on ``gate``, and the caller keeps releasing it.
    """

    task_id = _task_id(admission, event)
    identity = presence_event_identity(admission.binding_id, event)
    cached = presence_turn_replay(Path(drive_root), task_id, event.conversation_key, identity)
    if cached is not None:
        return cached

    def execute() -> PresenceTurnResult:
        second_cached = _cached_result(Path(drive_root), task_id, identity)
        if second_cached is not None:
            # A turn lost between its terminal write and its pointer write replays from the durable
            # row; its pointer is rebuilt here, under the conversation lock, so no newer turn is undone.
            physical_id = second_cached.task_id
            if _pointer_behind(Path(drive_root), event.conversation_key, physical_id):
                stored = load_task_result(Path(drive_root), physical_id) or {}
                sends = (_turn_sends(_live_task_rows(Path(drive_root), physical_id, event.conversation_key))[0]
                         if second_cached.delivery_reporting_version else [])
                _write_previous_turn(Path(drive_root), event.conversation_key, physical_id, outcome=second_cached.outcome,
                                     message=second_cached.text, sends=sends or [], work_ref=second_cached.work_ref,
                                     finished_at=str(stored.get("ts") or ""),
                                     delivery=_delivery_state(second_cached.delivery_reporting_version, sends,
                                                              second_cached.text))
            return second_cached
        return _execute_live()

    def _execute_live() -> PresenceTurnResult:
        nonlocal task_id
        task_id = _retry_target(Path(drive_root), task_id, identity, claim=True)
        with _LIVE_LOCK:
            _LIVE_PRESENCE_TASKS.add(task_id)
        try:
            return _run_current()
        finally:
            with _LIVE_LOCK:
                _LIVE_PRESENCE_TASKS.discard(task_id)

    def _run_current() -> PresenceTurnResult:
        # Both locks are held: no other execution of this conversation runs, so a running or
        # interrupted row of this task (not yet reconciled) belongs to a lost attempt too.
        stored = _stored_turn(Path(drive_root), task_id, identity)
        if str(stored.get("status") or "") == STATUS_CANCELLED:
            # An owner's Stop is not an absent turn. Never regenerate work after it,
            # nor acknowledge the original transport event as completed/silent.
            raise PresenceTurnError("presence_turn_cancelled", "source_event_id", turn_ref=task_id)
        lost_attempt = is_reconciled_presence_placeholder(stored) or str(stored.get("status") or "") in {
            STATUS_RUNNING, STATUS_INTERRUPTED}
        if lost_attempt:
            # A dead local stack is not evidence the provider or an external tool never acted.
            # In particular a dispatched operation administratively settled as "abandoned"
            # retains an unknown effect. Retain the original event with its transport, and
            # leave this row untouched: no new inbound row, agent, or paid generation. A
            # positively pre-thread failure wrote no running row and reaches the normal path;
            # a later canonical terminal result wins the cached-replay check above.
            _notify_unresolved_turn(Path(drive_root), task_id)
            raise PresenceTurnError("presence_attempt_outcome_unknown", "source_event_id", turn_ref=task_id)
        def chat_generation() -> tuple[int, int] | None:  # rotation renames the live file: a new inode
            try:
                stat = os.stat(Path(drive_root) / "logs" / "chat.jsonl")
            except OSError:
                return None
            return (stat.st_dev, stat.st_ino)

        generation = chat_generation()  # the live file this execution's rows land in, read before the rows
        prior_rows = _live_task_rows(Path(drive_root), task_id, event.conversation_key)
        for prior in prior_rows:
            if prior.get("direction") == "in" and prior.get("presence_event_identity") not in (None, identity):
                raise PresenceTurnError("presence_event_identity_conflict", "source_event_id", turn_ref=task_id)
        task = _build_task(
            admission,
            event,
            drive_root=Path(drive_root),
            staged_files=tuple(Path(item) for item in staged_files),
            physical_task_id=task_id,
            lost_attempt=lost_attempt,
            prior_rows=prior_rows,
        )
        chat_id = int(task["chat_id"])
        actor_id = _stable_numeric_id("presence-actor-log", str(task.get("actor_id") or ""))
        own_start = len(prior_rows)
        # A lost attempt logged the message (here or in a rotated archive); an attempt that died before its
        # running write left only that row. Re-logging would also let a later retry mistake the fresh row
        # for complete receipt coverage, so the row is written only when no attempt has a trace at all.
        if (not lost_attempt and task_id == _task_id(admission, event)
                and not any(row.get("direction") == "in" for row in prior_rows)):
            _log_dialogue(
                Path(drive_root),
                direction="in",
                chat_id=chat_id,
                user_id=actor_id,
                text=event.text or task["text"],
                event=event,
                task=task,
                task_id=task_id,
            )
        if generation is None:  # the inbound log just created the live file: that is this execution's generation
            generation = chat_generation()
        if agent_factory is None:
            from ouroboros.agent import make_agent

            factory = make_agent
        else:
            factory = agent_factory
        agent = factory(
            repo_dir=str(repo_dir),
            drive_root=str(drive_root),
            event_queue=event_queue,
        )
        # The generic agent's RUNNING writer logs and continues on failure. Presence cannot:
        # after a crash, absence of that row would otherwise authorize a second model/tool effect.
        # This existing task-result authority is published and read back BEFORE handle_task.
        from ouroboros.task_results import write_task_result

        try:
            write_task_result(Path(drive_root), task_id, STATUS_RUNNING,
                              create_only=True, strict_existing_dict=True,
                              metadata=task["metadata"], chat_id=chat_id,
                              _is_direct_chat=True, source="presence", result="Task is running.")
            start = _stored_turn(Path(drive_root), task_id, identity)
            if str(start.get("status") or "") != STATUS_RUNNING or (
                    start.get("metadata") or {}).get("presence_event_identity") != identity:
                raise ValueError("Presence start record was not bound to this event")
        except (OSError, ValueError) as exc:
            raise PresenceTurnError("presence_start_unwritable", "source_event_id", turn_ref=task_id) from exc
        events = agent.handle_task(task)
        # The model can have a draft reply before every allowed route refuses quota, and the
        # terminal write can fail after the start barrier. The durable terminal cause, not that
        # draft or the presence_result envelope, determines whether the transport may
        # acknowledge the original event; a row still RUNNING here is an unproven effect.
        terminal = _stored_turn(Path(drive_root), task_id, identity)
        row = next((item for item in events if item.get("type") == "presence_result"), None)
        refusal = _terminal_refusal(terminal)
        if refusal:
            _notify_unresolved_turn(Path(drive_root), task_id)
            metadata = terminal.get("metadata") if isinstance(terminal.get("metadata"), dict) else {}
            # Scheduled work survives the refusal: the durable ref when the terminal landed, else the
            # host-built handoff fact of this execution (the terminal write itself may have failed).
            work_ref = str(metadata.get("presence_work_ref") or (row or {}).get("work_ref") or "")
            raise PresenceTurnError(refusal, "source_event_id", turn_ref=task_id, work_ref=work_ref)
        if not isinstance(row, dict):
            raise PresenceTurnError("presence_result_missing", "presence_result")
        result = PresenceTurnResult(
            outcome=str(row.get("outcome") or "message"),
            text=str(row.get("text") or ""),
            task_id=task_id,
            work_ref=str(row.get("work_ref") or ""),
            delivery_reporting_version=event.delivery_reporting_version,
        )
        if result.outcome in {"message", "deferred"} and result.text and not result.delivery_reporting_version:
            _log_dialogue(
                Path(drive_root),
                direction="out",
                chat_id=chat_id,
                user_id=0,
                text=result.text,
                event=event,
                task=task,
                task_id=task_id,
            )
        # Still under the conversation lock. A replay writes only through the cached-replay repair, under
        # this same lock, and it leaves a pointer that names a newer turn alone.
        sends: list[str] | None = []
        if event.delivery_reporting_version:
            sends = _turn_sends(_live_task_rows(Path(drive_root), task_id, event.conversation_key), own_start,
                                same_generation=generation is None or chat_generation() == generation)[0]
        _write_previous_turn(Path(drive_root), event.conversation_key, task_id, outcome=result.outcome,
                             message=str(row.get("message") or result.text), sends=sends or [],
                             work_ref=result.work_ref, finished_at=utc_now_iso(),
                             delivery=_delivery_state(event.delivery_reporting_version, sends, result.text),
                             finish_note=str(row.get("finish_note") or ""))
        return result

    if admitted is not None:
        if admitted.conversation_key != str(event.conversation_key or "").strip():
            raise PresenceTurnError("presence_admission_conversation_mismatch", "conversation_key")
        return execute()
    return (gate or _configured_gate(Path(drive_root))).run(event.conversation_key, execute)


class PresenceTurnNotStarted(RuntimeError):
    """Nothing ran: admission was interrupted or no thread could start; retry the same event."""


class PresenceTurnExecution:
    """One live host turn: the result every waiter shares and the admission that starts it."""

    __slots__ = ("turn_id", "identity", "result", "admission")

    def __init__(self, turn_id: str, identity: str = "") -> None:
        self.turn_id = turn_id
        self.identity = identity
        self.result: concurrent.futures.Future = concurrent.futures.Future()
        # RUNNING from birth: a cancelled waiter's wrapper calls Future.cancel(), which a running
        # future refuses, so no HTTP waiter can cancel the result other waiters share.
        self.result.set_running_or_notify_cancel()
        self.admission: asyncio.Task | None = None


class PresenceTurnExecutions:
    """The Host's live presence turns by durable turn id: start one, or join the one already running.

    A turn is work, not its request. It queues on the gate as a coroutine (no thread, shared or
    owned), then runs on its own daemon thread with the admitting request's context variables
    (settings, usage and wait scopes), as ``asyncio.to_thread`` carried them. ``reserve``d
    capacity returns only at true settlement: a cancelled or disconnected waiter leaves the turn
    and its capacity in place, and a retry of the same event joins it. A turn the process exits
    under stays host-lost for the transport's retry; the daemon thread never extends a drain.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._live: dict[str, PresenceTurnExecution] = {}

    def live(self) -> list[str]:
        with self._lock:
            return list(self._live)

    def start_or_join(
        self,
        turn_id: str,
        *,
        identity: str = "",
        reserve: Callable[[], bool],
        release: Callable[[], None],
        admit: Callable[[], Any],
        run: Callable[[PresenceTurnLease], Any],
    ) -> tuple[PresenceTurnExecution | None, bool]:
        """``(execution, started)``; ``(None, False)`` when ``reserve`` refuses a new turn.

        A live turn is joined before capacity is consulted: a retry never needs a new slot.
        Call from the event loop; ``admit`` is a coroutine function returning the lease.
        """
        with self._lock:
            live = self._live.get(turn_id)
            if live is not None:
                if identity and live.identity and identity != live.identity:
                    raise PresenceTurnError("presence_event_identity_conflict", "source_event_id", turn_ref=turn_id)
                return live, False
            if not reserve():
                return None, False
            execution = self._live[turn_id] = PresenceTurnExecution(turn_id, identity)
        admission = self._admit_and_start(execution, admit, run, release)
        try:
            execution.admission = asyncio.get_running_loop().create_task(admission)
        except BaseException:
            admission.close()
            self._settle(execution, release, error=PresenceTurnNotStarted("presence turn admission did not start"))
            raise
        execution.admission.add_done_callback(lambda task: self._admission_ended(execution, release, task))
        return execution, True

    def _admission_ended(self, execution: PresenceTurnExecution, release: Callable[[], None],
                         task: asyncio.Task) -> None:
        # Cancellation lands only before the thread starts — in ``admit`` or before the first step,
        # where the coroutine's own handlers never run (e.g. a loop shutting down).
        if task.cancelled() and not execution.result.done():
            self._settle(execution, release, error=PresenceTurnNotStarted("presence turn admission was interrupted"))

    async def _admit_and_start(self, execution: PresenceTurnExecution, admit: Callable[[], Any],
                               run: Callable[[PresenceTurnLease], Any], release: Callable[[], None]) -> None:
        try:
            lease = await admit()
        except asyncio.CancelledError:
            raise  # settled by _admission_ended
        except BaseException as exc:
            self._settle(execution, release, error=exc)
            if not isinstance(exc, Exception):
                raise
            return
        try:
            thread = threading.Thread(
                target=contextvars.copy_context().run, args=(self._execute, execution, run, lease, release),
                name=f"presence-turn-{execution.turn_id}", daemon=True,
            )
            thread.start()
        except BaseException as exc:  # e.g. "can't start new thread": nothing ran, so nothing stays held
            self._release_lease(execution, lease)
            error = PresenceTurnNotStarted("presence turn thread could not start")
            error.__cause__ = exc
            self._settle(execution, release, error=error)
            if not isinstance(exc, Exception):
                raise

    def _execute(self, execution: PresenceTurnExecution, run: Callable[[PresenceTurnLease], Any],
                 lease: PresenceTurnLease, release: Callable[[], None]) -> None:
        outcome: Any = None
        error: BaseException | None = None
        try:
            outcome = run(lease)
        except BaseException as exc:  # relayed unchanged to every waiter
            error = exc
        self._release_lease(execution, lease)
        self._settle(execution, release, result=outcome, error=error)

    @staticmethod
    def _release_lease(execution: PresenceTurnExecution, lease: PresenceTurnLease) -> None:
        try:
            lease.release()
        except Exception:  # every release callback still ran; settlement must not be lost with it
            log.warning("presence turn %s could not release its gate lease", execution.turn_id, exc_info=True)

    def _settle(self, execution: PresenceTurnExecution, release: Callable[[], None], *,
                result: Any = None, error: BaseException | None = None) -> None:
        # Capacity first, then the id retires, then every waiter sees the outcome. The live set is
        # custody the Host reads the moment a waiter observes the terminal, and this usually runs on
        # the turn's own thread while the waiter wakes on the loop: an id published after its outcome
        # could outlive a response already on the wire. A retry that takes the lock before this joins
        # the running turn; one that lands after retirement rechecks durable authority — a turn that
        # reached the model left its RUNNING/terminal row, so it replays or refuses; only a turn that
        # failed before the start barrier may have no row, and it never generated anything to repeat.
        try:
            release()
        except Exception:
            log.warning("presence turn %s could not return its capacity", execution.turn_id, exc_info=True)
        with self._lock:
            if self._live.get(execution.turn_id) is execution:
                del self._live[execution.turn_id]
        if error is None:
            execution.result.set_result(result)
        else:
            execution.result.set_exception(error)


__all__ = [
    "PresenceTurnError",
    "PresenceTurnEvent",
    "PresenceTurnExecution",
    "PresenceTurnExecutions",
    "PresenceTurnGate",
    "PresenceTurnLease",
    "PresenceTurnNotStarted",
    "PresenceTurnResult",
    "admit_configured_gate",
    "build_presence_result_event",
    "presence_turn_is_live",
    "presence_turn_replay",
    "presence_turn_task_id",
    "run_presence_turn",
]
