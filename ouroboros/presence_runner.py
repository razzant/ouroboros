"""Fresh-agent execution for bounded, host-admitted presence turns."""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextvars
import functools
import hashlib
import json
import logging
import os
import threading
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ouroboros.artifacts import stage_task_attachments
from ouroboros.contracts.task_contract import attach_task_contract
from ouroboros.presence_admission import PresenceAdmission, PresenceAdmissionError
from ouroboros.presence_authority import presence_ceiling_payload
# The gate (cap, serialization, lease) lives in presence_gate; these names stay
# importable here because callers and tests reach them through the runner.
from ouroboros.presence_gate import (  # noqa: F401
    _GATE_POLL_SEC,
    _GATE_STOP_POLL_SEC,
    _GATES,
    _GATES_LOCK,
    _LIVE_LOCK,
    _LIVE_PRESENCE_TASKS,
    PresenceTurnGate,
    PresenceTurnLease,
    _configured_gate,
    _try_exclusive,
    admit_configured_gate,
    presence_turn_is_live,
)
from ouroboros.task_results import (
    STATUS_CANCELLED,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_INTERRUPTED,
    STATUS_RUNNING,
    is_reconciled_presence_placeholder,
    load_task_result,
    task_results_dir,
)
from ouroboros.utils import append_jsonl, atomic_write_json, iter_jsonl_objects, read_json_dict, utc_now_iso
from ouroboros.store import StoreError

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
        disposition: str = "",
        delivery_reporting_version: int = 0,
    ) -> None:
        self.code = str(code or "presence_turn_failed")
        self.field = str(field or "presence_turn")
        self.turn_ref = str(turn_ref or "")
        self.work_ref = str(work_ref or "")
        self.disposition = disposition
        self.delivery_reporting_version = delivery_reporting_version
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
    # Negotiated separately from receipts and never part of the event identity
    # (``presence_continuation``): 1 lets the transport take the initial envelope early.
    continuation_version: int = 0


@dataclass(frozen=True)
class PresenceTurnResult:
    outcome: str
    text: str
    task_id: str
    work_ref: str = ""
    delivery_reporting_version: int = 0
    # ``continuing``: the initial envelope of a live author; ``completed``: the author ended.
    status: str = "completed"
    continuation_ref: str = ""
    output_ref: str = ""
    continuation_version: int = 0


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
    presence = metadata.get("presence") if isinstance(metadata.get("presence"), dict) else {}
    return PresenceTurnResult(
        outcome=outcome, text=text, task_id=task_id,
        work_ref=str(metadata.get("presence_work_ref") or ""),
        delivery_reporting_version=int(presence.get("delivery_reporting_version") == 1),
        output_ref=str(metadata.get("presence_output_ref") or "") if text else "",
    )


def build_presence_result_event(task: dict[str, Any], text: str, ctx: Any, *, terminal_origin: str = "",
                                retain_scheduled_handoff: bool = False) -> dict[str, Any]:
    """Freeze typed delivery metadata before the ordinary durable result write."""
    from ouroboros.presence_continuation import _handoff_ref, terminal_output
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
    work_ref = _handoff_ref(ctx)
    if outcome == "deferred" and not work_ref:
        outcome = "message"
    if work_ref and (retain_scheduled_handoff or terminal_origin in HOST_AUTHORED_TERMINAL_ORIGINS):
        # A failed/forced or host-replaced final still owes an admitted child's result.
        # Transports poll only deferred outcomes, even when no reply was authored.
        outcome = "deferred"
    outcome, result_text = _presence_delivery(outcome, str(text or ""), terminal_origin)
    # An output this author already released at a review wait is never spoken twice.
    outcome, result_text, output_ref = terminal_output(ctx, outcome, result_text)
    metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    metadata["presence_outcome"] = outcome
    metadata["presence_result_text"] = result_text
    if output_ref:
        metadata["presence_output_ref"] = output_ref
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
        "output_ref": output_ref,
        "ts": utc_now_iso(),
    }


def _stable_numeric_id(prefix: str, value: str) -> int:
    digest = hashlib.sha256(f"{prefix}\0{value}".encode("utf-8")).digest()
    return (1 << 40) + (int.from_bytes(digest[:6], "big") & ((1 << 40) - 1))


def presence_chat_id(conversation_key: str) -> int:
    """The chat id every turn of one conversation shares; a live call reuses it exactly."""
    return _stable_numeric_id("presence-conversation", conversation_key)


# The exact event payload the Host door and the call door accept; nothing else is a Presence event.
PRESENCE_EVENT_FIELDS = frozenset({
    "source_event_id", "provider", "account_id", "conversation_id", "thread_id",
    "conversation_key", "actor", "conversation", "message", "text",
})


def presence_turn_event(event_payload: Any, admission: PresenceAdmission, *,
                        delivery_reporting_version: int = 0, continuation_version: int = 0) -> PresenceTurnEvent:
    """Build the one event shape a Presence turn runs on from a transport payload.

    Shared by ``/presence/turn`` and the call door so both check the same facts: the
    payload shape, that the provider facts match the owner-created binding, and that the
    identity facts are present. The transport authenticates provider facts, but it does
    not get to choose the concurrency/history identity: the conversation key is derived
    from the binding-checked origin facts, never taken from the payload.
    """
    from ouroboros.presence_bindings import conversation_key

    if not isinstance(event_payload, Mapping) or set(event_payload) != PRESENCE_EVENT_FIELDS:
        raise PresenceAdmissionError("presence_event_invalid", "event", "invalid presence event")
    provider = str(event_payload.get("provider") or "").strip()
    account_id = str(event_payload.get("account_id") or "").strip()
    conversation_id = str(event_payload.get("conversation_id") or "").strip()
    thread_id = str(event_payload.get("thread_id") or "").strip()
    if (
        provider != admission.origin.transport
        or account_id != admission.origin.account_id
        or (admission.origin.conversation_id != "*" and conversation_id != admission.origin.conversation_id)
        or (admission.origin.thread_id and thread_id != admission.origin.thread_id)
    ):
        raise PresenceAdmissionError("presence_origin_mismatch", "event",
                                     "presence event does not match its owner-created binding")
    event = PresenceTurnEvent(
        source_event_id=str(event_payload["source_event_id"] or "").strip(),
        provider=provider,
        account_id=account_id,
        conversation_id=conversation_id,
        thread_id=thread_id,
        conversation_key=conversation_key(provider, account_id, conversation_id, thread_id),
        actor=dict(event_payload["actor"]) if isinstance(event_payload["actor"], Mapping) else {},
        conversation=dict(event_payload["conversation"]) if isinstance(event_payload["conversation"], Mapping) else {},
        message=dict(event_payload["message"]) if isinstance(event_payload["message"], Mapping) else {},
        text=str(event_payload["text"] or ""),
        delivery_reporting_version=delivery_reporting_version,
        continuation_version=continuation_version,
    )
    if not event.source_event_id or not event.conversation_key or not event.actor:
        raise PresenceAdmissionError("presence_identity_missing", "event",
                                     "presence event is missing identity facts")
    return event


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


def _retry_reset(row: Mapping[str, Any], task_id: str, identity: str) -> datetime | None:
    """Only the source-bound, pre-dispatch certificate can permit event regeneration."""
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    proof = metadata.get("presence_retry_proof")
    if not (identity and row.get("status") == STATUS_FAILED
            and row.get("reason_code") == "resource_refusal_no_resend"
            and isinstance(proof, dict) and proof.get("kind") == "first_round_engine_not_started"
            and proof.get("event_identity") == identity and proof.get("task_id") == task_id):
        return None
    try:
        reset = datetime.fromisoformat(str(proof.get("reset_at") or "").replace("Z", "+00:00"))
    except ValueError:
        return None
    return reset if reset.tzinfo is not None else None


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
        reset = _retry_reset(row, current_id, identity)
        if reset is None or reset > datetime.now(timezone.utc) or not claim:
            return current_id
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        proof = metadata.get("presence_retry_proof")
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
        except (OSError, ValueError, StoreError) as exc:
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

    quarantined = task_results_dir(drive_root, create=False) / TASK_RESULT_QUARANTINE_DIR
    try:
        current = load_task_result(drive_root, task_id, strict=True) or {}
        if (quarantined / f"{task_id}.json").exists() or any(quarantined.glob(f"{task_id}.*.json")):
            # A strictly admitted, source-bound late terminal is newer authority than an old
            # quarantined attempt. Nothing else, including an unrelated or running row, clears it.
            if str(current.get("status") or "") not in {STATUS_COMPLETED, STATUS_FAILED} or not identity:
                raise PresenceTurnError("presence_result_unreadable", "source_event_id", turn_ref=task_id)
        _assert_event_identity(current, task_id, identity)
        return current
    except (OSError, ValueError, StoreError) as exc:
        if isinstance(exc, PresenceTurnError):
            raise
        raise PresenceTurnError("presence_result_unreadable", "source_event_id", turn_ref=task_id) from exc


def _terminal_refusal(stored: Mapping[str, Any]) -> str:
    """Classify an unresolved attempt without certifying silence or absence of effects.

    A missing terminal or the loop's durable unknown marker cannot authorize event
    regeneration. The runner proves local loss under the conversation gate before
    retaining a nonterminal row. Resource refusal is retryable only with its
    source-bound pre-dispatch certificate. Other terminals retain ordinary authored
    output projection, including a real admitted child's reference.
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


def _refused_turn(drive_root: Path, task_id: str, stored: Mapping[str, Any], code: str,
                  *, identity: str = "", work_ref: str = "") -> PresenceTurnError:
    """Retain admitted work separately from input FIFO custody; never certify a reply."""
    metadata = stored.get("metadata") if isinstance(stored.get("metadata"), dict) else {}
    if work_ref and not metadata.get("presence_work_ref"):
        # The final result write may have failed after the actual handoff. Retain that
        # host-built reference before acknowledging custody, including across HTTP loss.
        from ouroboros.task_store import patch_document

        def retain_child(current: dict) -> dict:
            meta = dict(current.get("metadata") or {})
            meta.setdefault("presence_work_ref", work_ref)
            return {**current, "metadata": meta}

        try:
            stored = patch_document(drive_root, task_id, retain_child)
        except (OSError, ValueError, StoreError) as exc:
            raise PresenceTurnError("presence_result_unreadable", "source_event_id", turn_ref=task_id) from exc
        metadata = stored.get("metadata") or {}
    presence = metadata.get("presence") if isinstance(metadata.get("presence"), dict) else {}
    retryable = code == "presence_resources_unavailable" and _retry_reset(stored, task_id, identity) is not None
    _notify_unresolved_turn(drive_root, task_id)
    from ouroboros.presence_recovery import retained_turn_facts

    return PresenceTurnError(code, "source_event_id", turn_ref=task_id,
                             work_ref=str(retained_turn_facts(stored, task_id).get("work_ref") or ""),
                             disposition="retry" if retryable else "retained",
                             delivery_reporting_version=int(presence.get("delivery_reporting_version") == 1))


def _cached_result(drive_root: Path, task_id: str, identity: str = "") -> PresenceTurnResult | None:
    if identity:
        task_id = _retry_target(drive_root, task_id, identity)
    stored = _stored_turn(drive_root, task_id, identity)
    refusal = _terminal_refusal(stored)
    if refusal == "presence_resources_unavailable":
        reset = _retry_reset(stored, task_id, identity)
        if reset is not None and reset <= datetime.now(timezone.utc):
            return None  # the conversation-lock holder claims its successor
        raise _refused_turn(drive_root, task_id, stored, refusal, identity=identity)
    if refusal and (str(stored.get("status") or "") not in {STATUS_COMPLETED, STATUS_FAILED}
                    or is_reconciled_presence_placeholder(stored)):
        return None  # the conversation gate distinguishes a live author from a lost attempt
    if refusal:
        raise _refused_turn(drive_root, task_id, stored, refusal, identity=identity)
    return presence_result_from_stored(stored, task_id)


def _notify_unresolved_turn(drive_root: Path, task_id: str) -> None:
    """Serialize source-bound recovery facts across concurrent Host replay readers.

    Reuse the existing gate locks without taking an execution slot. The canonical
    room fact has its own stable identity; no public reply or owner question is sent.
    """
    try:
        gate = _configured_gate(drive_root)
        key, local_lock = gate._conversation(f"recovery-fact:{task_id}")
        with local_lock, gate._file_lock(gate._conversation_file(key)):
            _write_unresolved_notice(drive_root, task_id)
    except Exception:
        log.warning("Presence recovery fact lock failed for %s", task_id, exc_info=True)


def _write_unresolved_notice(drive_root: Path, task_id: str) -> None:
    from ouroboros.presence_recovery import record_unresolved_turn

    record_unresolved_turn(drive_root, task_id)


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


def _first_lent_at(stored: Mapping[str, Any]) -> str:
    """Pointer chronology starts at the first yield, even when an author parks again."""
    continued = stored.get("presence_continuation")
    return str(continued.get("first_lent_at") or continued.get("lent_at") or "") if isinstance(continued, dict) else ""


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
                         finish_note: str = "", lent_at: str = "", continuing: bool = False,
                         strict: bool = False) -> None:
    """Best effort: the pointer is a projection, and a turn that already answered is not failed over it.

    It also lists the conversation's still-continuing authors (``open_turns``, refs only:
    their facts stay on their task rows). An author that yielded its conversation at
    its first ``lent_at`` never replaces a pointer a newer turn wrote since; a later
    park keeps it in the open list and its terminal removes it from that list.
    ``strict`` (an author about to yield) raises instead, and reads the write back first:
    a successor must never run without seeing that the yielded author is still open.
    """
    pointer = {
        "conversation_key": conversation_key, "task_id": task_id, "outcome": outcome, "message": message,
        **({"finish_note": finish_note} if finish_note else {}),
        "transport_sends": [text for text in sends if text], "work_ref": work_ref,
        "finished_at": finished_at, "delivery": delivery, **({"continuing": True} if continuing else {}),
    }
    try:
        gate = _configured_gate(Path(drive_root))
        key, local_lock = gate._conversation(f"projection:{conversation_key}")
        with local_lock, gate._file_lock(gate._conversation_file(key)):
            current = _read_previous_turn(drive_root, conversation_key) or {}
            open_turns = [ref for ref in current.get("open_turns") or [] if ref != task_id and str(
                (load_task_result(drive_root, ref) or {}).get("status") or STATUS_RUNNING) not in {
                STATUS_COMPLETED, STATUS_FAILED, STATUS_CANCELLED}]
            newer = bool(lent_at) and current.get("task_id") not in (None, "", task_id) and str(
                current.get("finished_at") or "") > lent_at
            row = {key: value for key, value in (current if newer else pointer).items() if key != "open_turns"}
            written = {**row, **({"open_turns": [*open_turns, *([task_id] if continuing else [])]}
                                 if open_turns or continuing else {})}
            atomic_write_json(_previous_turn_path(drive_root, conversation_key), written)
            if strict and _read_previous_turn(drive_root, conversation_key) != written:
                raise OSError("presence previous-turn pointer did not read back")
    except Exception:
        if strict:
            raise
        log.warning("presence previous-turn pointer not written for %s (task %s); the next replay of this turn "
                    "rebuilds it", conversation_key, task_id, exc_info=True)


def record_continuing_turn(drive_root: Path, conversation_key: str, task_id: str, *, outcome: str, message: str,
                           work_ref: str, lent_at: str) -> None:
    """Record an open author without replacing a turn that ran since its first yield.

    Raises when the projection cannot be written and read back: such an author keeps its
    conversation, since a successor would otherwise run without seeing it is still open.
    """
    _write_previous_turn(drive_root, conversation_key, task_id, outcome=outcome, message=message, sends=[],
                         work_ref=work_ref, finished_at=lent_at, delivery="unknown", continuing=True, strict=True,
                         lent_at=_first_lent_at(load_task_result(drive_root, task_id) or {}) or lent_at)


def _pointer_speech(stored: Mapping[str, Any], outcome: str, message: str) -> tuple[str, str]:
    """A terminal that re-selected an output it released at a review wait says nothing new;
    its pointer still names that released speech (``presence_continuation``)."""
    continued = stored.get("presence_continuation") if isinstance(stored.get("presence_continuation"), dict) else {}
    spoken = [item for item in continued.get("outputs") or [] if isinstance(item, dict) and item.get("text")]
    if message or not spoken:
        return outcome, message
    return str(spoken[-1].get("outcome") or outcome), str(spoken[-1]["text"])


def _repair_pointer(drive_root: Path, conversation_key: str, physical_id: str) -> None:
    """Rebuild the pointer of a settled turn lost between its terminal write and its pointer write.

    A pointer a newer turn wrote is never undone: the projection lock and the yielded
    author's ``lent_at`` decide that in ``_write_previous_turn``.
    """
    stored = load_task_result(Path(drive_root), physical_id) or {}
    if _terminal_refusal(stored) or not _pointer_behind(Path(drive_root), conversation_key, physical_id):
        return
    settled = presence_result_from_stored(stored, physical_id)
    sends = (_turn_sends(_live_task_rows(Path(drive_root), physical_id, conversation_key))[0]
             if settled.delivery_reporting_version else [])
    outcome, message = _pointer_speech(stored, settled.outcome, settled.text)
    _write_previous_turn(Path(drive_root), conversation_key, physical_id, outcome=outcome, message=message,
                         sends=sends or [], work_ref=settled.work_ref, finished_at=str(stored.get("ts") or ""),
                         delivery=_delivery_state(settled.delivery_reporting_version, sends, message, outcome),
                         lent_at=_first_lent_at(stored))


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
    # A continuing author pointed at itself when it yielded; a pointer naming another turn
    # written after that belongs to a newer turn, however late this one completed. A
    # pointer still calling this settled turn continuing is its yield, not its terminal.
    settled = _first_lent_at(stored) or str(stored.get("ts") or "")
    if pointer is not None and pointer.get("task_id") == task_id:
        return bool(pointer.get("continuing"))
    return pointer is None or str(pointer.get("finished_at") or "") <= settled


def presence_turn_replay(drive_root: Path, task_id: str, conversation_key: str,
                         identity: str = "", continuation_version: int = 0) -> PresenceTurnResult | None:
    """A settled turn's durable answer, returned without the gate; None when the turn must (re)run.

    A version-1 consumer gets a continuing author's stored initial envelope (write-once),
    whatever the author did since; its terminal is read from the continuation poll. When
    that author has settled since, its lost pointer is rebuilt before the envelope returns.
    """
    if continuation_version and identity:
        from ouroboros.presence_continuation import replay_envelope

        envelope = replay_envelope(Path(drive_root), task_id, identity)
        if envelope is not None:
            _repair_pointer(Path(drive_root), conversation_key, envelope.task_id)
            return envelope
    cached = _cached_result(Path(drive_root), task_id, identity)
    return None if cached is None or _pointer_behind(Path(drive_root), conversation_key, cached.task_id) else cached


def _delivery_state(reporting_version: int, sends: Sequence[str] | None, text: str, outcome: str = "message") -> str:
    """One rule for the live write and the repair.

    Tool sends confirm mid-turn; the adapter-delivered reply's receipt arrives only after the turn
    returns, so a reply text that is not among the confirmed sends is at most partly confirmed.
    """
    # Legacy tool_delivered messages are finish notes, not unconfirmed adapter speech.
    if outcome not in {"message", "deferred"}:
        text = ""
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
    speaker: Mapping[str, Any] | None = None,
) -> None:
    """Append one Presence dialogue row. ``speaker`` names who spoke when it is not the
    event's actor (a live call's correspondent utterance); the event identity stays the
    turn's own, so a later replay of the same event still recognizes its rows."""
    from types import SimpleNamespace
    from ouroboros import chat_chain
    from ouroboros.dialogue_provenance import presence_provenance_from_task

    actor = dict(speaker) if speaker is not None else dict(event.actor)
    state = read_json_dict(drive_root / "state" / "state.json") or {}
    row = {
        "ts": utc_now_iso(),
        "session_id": state.get("session_id"),
        "direction": direction,
        "chat_id": chat_id,
        "user_id": user_id,
        "text": text,
        "format": "markdown" if direction == "out" else "",
        "source": f"presence:{event.provider}",
        "sender_label": str(actor.get("display_name") or actor.get("username") or ""),
        "sender_session_id": str(actor.get("platform_actor_id") or actor.get("id") or ""),
        "client_message_id": event.source_event_id,
        "presence_event_identity": presence_event_identity(str(task["metadata"]["presence"]["binding_id"]), event),
        "transport": {
            "provider": event.provider,
            "account_id": event.account_id,
            "conversation_id": event.conversation_id,
            "thread_id": event.thread_id,
            "conversation_key": event.conversation_key,
            "actor": actor,
            "conversation": dict(event.conversation),
            "message": dict(event.message),
            **({"delivery": {"state": "authored"}} if direction == "out" else {}),
        },
        "presence_provenance": presence_provenance_from_task(task),
        "task_id": task_id,
    }
    if direction == "in":
        # Retain the transport's exact contextual observations before publishing their address.
        # They remain dated context of this event, never additional incoming speech or work.
        source = {"recorded_at": row["ts"], "source_event_id": event.source_event_id,
                  "text": text, "transport": row["transport"]}
        row["context_source_ref"] = chat_chain.retain_memory_source(
            SimpleNamespace(drive_root=drive_root, task_id=task_id), "presence-event",
            json.dumps(source, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"), "json")
    written = append_jsonl(
        drive_root / "logs" / "chat.jsonl", obj=row,
        ensure_record_boundary=(direction == "in"),  # a re-run never re-logs: this row must be parseable
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
    live_call: Any = None,
) -> dict[str, Any]:
    """The fresh task one turn runs as. ``live_call`` (a ``call_protocol.CallSpec``) marks a
    turn that is one live audio session: the driver reads ``metadata["live_call"]``."""
    from ouroboros.config import runtime_setting

    task_id = physical_task_id or _task_id(admission, event)
    chat_id = presence_chat_id(event.conversation_key)
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
        from ouroboros.presence_continuation import continuation_status

        # Authors that yielded this conversation while their result awaits review: refs whose
        # facts are read from their own canonical rows, never copied into the pointer.
        open_turns = [{"task_id": ref, "status": continuation_status(load_task_result(drive_root, ref) or {})}
                      for ref in previous_turn.pop("open_turns", None) or [] if ref != physical_task_id]
        if open_turns:
            presence_context["open_turns"] = open_turns
        if previous_turn.get("continuing"):
            # The pointer records the yield; whether that author still lives is its row's fact.
            status = continuation_status(load_task_result(drive_root, str(previous_turn.get("task_id") or "")) or {})
            if status != STATUS_RUNNING:
                previous_turn = {**previous_turn, "continuing": False, "author_status": status}
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
    metadata: dict[str, Any] = {
        "source": "presence",
        "client_message_id": event.source_event_id,
        "presence_event_identity": presence_event_identity(admission.binding_id, event),
        "inline_max_rounds": admission.inline_max_rounds,
        "presence": presence_context,
    }
    if live_call is not None:
        metadata["live_call"] = live_call.as_metadata()
    if admission.model_slot == "light":
        from ouroboros.config import get_light_model

        metadata["model"] = get_light_model()
        metadata["use_local_model"] = runtime_setting("USE_LOCAL_LIGHT", "").lower() in {"true", "1"}
    task: dict[str, Any] = {
        "id": task_id,
        "type": "presence",
        "source": "presence",
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
            attachment_manifest_all_rejected, attachment_manifest_projection, remove_staged_attachments,
        )

        if attachment_manifest_all_rejected(manifest):
            remove_staged_attachments(manifest)
            raise PresenceTurnError(
                "presence_attachment_admission_rejected",
                "staged_files",
                attachment_manifest=manifest,
            )
        from ouroboros.gateway.tasks import _render_attachment_lines

        # The manifest is task authority, not merely presentation prose.  Keep
        # every staged/rejected declaration on the canonical carrier before the
        # task contract is normalized so a later promotion or child can inherit
        # and materialize the exact inputs.
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
    live_call: Any = None,
) -> PresenceTurnResult:
    """Run one bounded turn; adapters retain durable provider custody.

    ``admitted`` is a lease the caller already took for this conversation (``PresenceTurnGate.admit``):
    the turn runs under it instead of waiting on ``gate``, and the caller keeps releasing it.
    Either lease may be lent at a qualified review wait (``presence_continuation``).
    ``live_call`` turns the task into one live audio session (``presence_call``); every
    other step — admission record, lost-attempt refusal, pointer, dialogue rows — is the same.
    """

    task_id = _task_id(admission, event)
    identity = presence_event_identity(admission.binding_id, event)
    cached = presence_turn_replay(Path(drive_root), task_id, event.conversation_key, identity,
                                  event.continuation_version)
    if cached is not None:
        return cached
    lease = admitted

    def execute() -> PresenceTurnResult:
        nonlocal task_id
        if event.continuation_version:
            from ouroboros.presence_continuation import replay_envelope

            # The author may have yielded (and even ended) since the unlocked check above.
            envelope = replay_envelope(Path(drive_root), task_id, identity)
            if envelope is not None:
                _repair_pointer(Path(drive_root), event.conversation_key, envelope.task_id)
                return envelope
        second_cached = _cached_result(Path(drive_root), task_id, identity)
        if second_cached is not None:
            # A turn lost between its terminal write and its pointer write replays from the durable
            # row; its pointer is rebuilt here, under the conversation lock, so no newer turn is undone.
            _repair_pointer(Path(drive_root), event.conversation_key, second_cached.task_id)
            return second_cached
        task_id = _retry_target(Path(drive_root), task_id, identity, claim=True)
        live_key = (str(Path(drive_root).resolve()), task_id)
        with _LIVE_LOCK:
            _LIVE_PRESENCE_TASKS.add(live_key)
        try:
            return _run_current()
        finally:
            with _LIVE_LOCK:
                _LIVE_PRESENCE_TASKS.discard(live_key)

    def _run_current() -> PresenceTurnResult:
        # Both locks are held: no other execution of this conversation runs, so a running or
        # interrupted row of this task (not yet reconciled) belongs to a lost attempt too.
        stored = _stored_turn(Path(drive_root), task_id, identity)
        if str(stored.get("status") or "") == STATUS_CANCELLED:
            # An owner's Stop is not an absent turn. Never regenerate work after it,
            # nor acknowledge the original transport event as completed/silent.
            raise _refused_turn(Path(drive_root), task_id, stored, "presence_turn_cancelled")
        lost_attempt = is_reconciled_presence_placeholder(stored) or str(stored.get("status") or "") in {
            STATUS_RUNNING, STATUS_INTERRUPTED}
        if lost_attempt:
            # A dead local stack is not evidence the provider or an external tool never acted.
            # In particular a dispatched operation administratively settled as "abandoned"
            # retains an unknown effect. Retain the original event with its transport, and
            # leave this row untouched: no new inbound row, agent, or paid generation. A
            # positively pre-thread failure wrote no running row and reaches the normal path;
            # a later canonical terminal result wins the cached-replay check above.
            raise _refused_turn(Path(drive_root), task_id, stored, "presence_attempt_outcome_unknown")
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
            live_call=live_call,
        )
        chat_id = int(task["chat_id"])
        actor_id = _stable_numeric_id("presence-actor-log", str(task.get("actor_id") or ""))
        log_dialogue = functools.partial(_log_dialogue, Path(drive_root), chat_id=chat_id,
                                        event=event, task=task, task_id=task_id)
        own_start = len(prior_rows)
        # A lost attempt logged the message (here or in a rotated archive); an attempt that died before its
        # running write left only that row. Re-logging would also let a later retry mistake the fresh row
        # for complete receipt coverage, so the row is written only when no attempt has a trace at all.
        if (task_id == _task_id(admission, event)
                and not any(row.get("direction") == "in" for row in prior_rows)):
            log_dialogue(direction="in", user_id=actor_id, text=event.text or task["text"])
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
        from ouroboros.presence_continuation import bind_review_wait

        bind_review_wait(agent, lease=lease, event=event, task_id=task_id, identity=identity,
                         drive_root=Path(drive_root))
        # The generic agent's RUNNING writer logs and continues on failure. Presence cannot:
        # after a crash, absence of that row would otherwise authorize a second model/tool effect.
        # This existing task-result authority is published and read back BEFORE handle_task.
        from ouroboros.task_results import write_task_result
        from ouroboros.task_status import execution_owner_record

        try:
            write_task_result(Path(drive_root), task_id, STATUS_RUNNING,
                              execution_owner=execution_owner_record(drive_root, task, "presence"),
                              create_only=True, strict_existing_dict=True,
                              metadata=task["metadata"], chat_id=chat_id,
                              _is_direct_chat=True, source=task["source"], result="Task is running.")
            start = _stored_turn(Path(drive_root), task_id, identity)
            if str(start.get("status") or "") != STATUS_RUNNING or (
                    start.get("metadata") or {}).get("presence_event_identity") != identity:
                raise ValueError("Presence start record was not bound to this event")
        except (OSError, ValueError, StoreError) as exc:
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
            metadata = terminal.get("metadata") if isinstance(terminal.get("metadata"), dict) else {}
            # Scheduled work survives the refusal: the durable ref when the terminal landed, else the
            # host-built handoff fact of this execution (the terminal write itself may have failed).
            work_ref = str(metadata.get("presence_work_ref") or (row or {}).get("work_ref") or "")
            raise _refused_turn(Path(drive_root), task_id, terminal, refusal,
                                identity=identity, work_ref=work_ref)
        if not isinstance(row, dict):
            raise PresenceTurnError("presence_result_missing", "presence_result")
        continued = terminal.get("presence_continuation") if isinstance(terminal.get("presence_continuation"), dict) else {}
        result = PresenceTurnResult(
            outcome=str(row.get("outcome") or "message"),
            text=str(row.get("text") or ""),
            task_id=task_id,
            work_ref=str(row.get("work_ref") or ""),
            delivery_reporting_version=event.delivery_reporting_version,
            continuation_ref=task_id if continued else "",
            output_ref=str(row.get("output_ref") or ""),
            continuation_version=event.continuation_version,
        )
        if result.outcome in {"message", "deferred"} and result.text and not result.delivery_reporting_version:
            log_dialogue(direction="out", user_id=0, text=result.text)
        # Still under the conversation lock. A replay writes only through the cached-replay repair, under
        # this same lock, and it leaves a pointer that names a newer turn alone.
        sends: list[str] | None = []
        if event.delivery_reporting_version:
            sends = _turn_sends(_live_task_rows(Path(drive_root), task_id, event.conversation_key), own_start,
                                same_generation=generation is None or chat_generation() == generation)[0]
        outcome, message = _pointer_speech(terminal, result.outcome, str(row.get("message") or result.text))
        _write_previous_turn(Path(drive_root), event.conversation_key, task_id, outcome=outcome, message=message,
                             sends=sends or [], work_ref=result.work_ref, finished_at=utc_now_iso(),
                             delivery=_delivery_state(event.delivery_reporting_version, sends, result.text or message, outcome),
                             finish_note=str(row.get("finish_note") or ""), lent_at=_first_lent_at(terminal))
        return result

    if admitted is not None:
        if admitted.conversation_key != str(event.conversation_key or "").strip():
            raise PresenceTurnError("presence_admission_conversation_mismatch", "conversation_key")
        return execute()
    lease = (gate or _configured_gate(Path(drive_root))).acquire(event.conversation_key)
    try:
        return execute()
    finally:
        lease.release()


class PresenceTurnNotStarted(RuntimeError):
    """Nothing ran: admission was interrupted or no thread could start; retry the same event."""


class PresenceTurnExecution:
    """One live host turn: the results every waiter shares and the admission that starts it.

    ``initial`` is the first envelope (a continuing author's, or the terminal itself) and
    ``result`` the author's terminal (#1536). The reserved capacity returns exactly once,
    at whichever comes first; the turn stays live until its terminal.
    """

    __slots__ = ("turn_id", "identity", "result", "initial", "admission", "_capacity", "_returned", "_lock")

    def __init__(self, turn_id: str, identity: str = "") -> None:
        self.turn_id = turn_id
        self.identity = identity
        self.result: concurrent.futures.Future = concurrent.futures.Future()
        self.initial: concurrent.futures.Future = concurrent.futures.Future()
        # RUNNING from birth: a cancelled waiter's wrapper calls Future.cancel(), which a running
        # future refuses, so no HTTP waiter can cancel the result other waiters share.
        self.result.set_running_or_notify_cancel()
        self.initial.set_running_or_notify_cancel()
        self.admission: asyncio.Task | None = None
        self._capacity: Callable[[], None] | None = None
        self._returned = False
        self._lock = threading.Lock()

    def return_capacity(self, fallback: Callable[[], None] | None = None) -> None:
        with self._lock:
            if self._returned:
                return
            self._returned, release = True, self._capacity or fallback
        if release is not None:
            try:
                release()
            except Exception:
                log.warning("presence turn %s could not return its capacity", self.turn_id, exc_info=True)

    def publish_initial(self, value: Any) -> bool:
        """The durable initial envelope of a continuing author: its waiters answer, its capacity returns."""
        with self._lock:
            if self.initial.done():
                return False
            self.initial.set_result(value)
        self.return_capacity()
        return True


class PresenceTurnExecutions:
    """The Host's live presence turns by durable turn id: start one, or join the one already running.

    A turn queues on the gate as a coroutine (no thread), then runs on its own daemon thread
    with the admitting request's context variables (settings, usage and wait scopes), as
    ``asyncio.to_thread`` carried them. Both versions return ``reserve``d capacity at a qualified
    park or terminal settlement; version 0's HTTP waiter stays open until terminal. Cancelling or
    disconnecting a waiter leaves the turn and any held capacity in place; retries join it.
    A process exit leaves the turn host-lost for retry; the daemon thread never extends a drain.
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
            execution._capacity = release
        admission = self._admit_and_start(execution, admit, run, release)
        try:
            execution.admission = asyncio.get_running_loop().create_task(admission)
        except BaseException:
            admission.close()
            self._settle(execution, release, error=PresenceTurnNotStarted("presence turn admission did not start"))
            raise
        execution.admission.add_done_callback(lambda task: self._admission_ended(execution, release, task))
        return execution, True

    def start_thread(self, turn_id: str, *, identity: str, admit: Callable[[], PresenceTurnLease],
                     run: Callable[[PresenceTurnLease], Any],
                     context: contextvars.Context) -> tuple[PresenceTurnExecution, bool]:
        """The proactive form of ``start_or_join``: admission waits on the turn's own thread."""
        with self._lock:
            live = self._live.get(turn_id)
            if live is not None:
                if identity and live.identity and identity != live.identity:
                    raise PresenceTurnError("presence_event_identity_conflict", "source_event_id", turn_ref=turn_id)
                return live, False
            execution = self._live[turn_id] = PresenceTurnExecution(turn_id, identity)

        def body() -> None:
            try:
                lease = admit()
            except BaseException as exc:
                self._settle(execution, lambda: None, error=exc)
                return
            self._execute(execution, run, lease, lambda: None)

        try:
            threading.Thread(target=context.run, args=(body,), name=f"presence-turn-{turn_id}", daemon=True).start()
        except BaseException as exc:
            error = PresenceTurnNotStarted("presence turn thread could not start")
            error.__cause__ = exc
            self._settle(execution, lambda: None, error=error)
            raise error from exc
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
        lease.execution = execution  # the author may publish its initial envelope through its lease
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
        # A continuing author returned its capacity with its initial envelope; that is never repeated.
        execution.return_capacity(release)
        with self._lock:
            if self._live.get(execution.turn_id) is execution:
                del self._live[execution.turn_id]
        for future in (execution.initial, execution.result):
            with execution._lock:
                if future.done():
                    continue
                if error is None:
                    future.set_result(result)
                else:
                    future.set_exception(error)


# Initiated turns run under this process's own execution registry (the Host keeps its own).
PROACTIVE_TURNS = PresenceTurnExecutions()


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
    "presence_chat_id",
    "presence_turn_event",
    "presence_turn_is_live",
    "presence_turn_replay",
    "presence_turn_task_id",
    "run_presence_turn",
]
