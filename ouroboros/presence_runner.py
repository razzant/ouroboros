"""Fresh-agent execution for bounded, host-admitted presence turns."""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import threading
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ouroboros.artifacts import stage_task_attachments, resolve_attachment_manifest
from ouroboros.contracts.task_contract import attach_task_contract
from ouroboros.presence_admission import PresenceAdmission
from ouroboros.presence_events import PresenceTurnError, PresenceTurnEvent, presence_turn_event as presence_turn_event
from ouroboros.presence_authority import presence_ceiling_payload
from ouroboros.conversation_lease import PresenceTurnGate, PresenceTurnLease
from ouroboros.presence_executions import (  # noqa: F401 -- the Host's registry, re-exported for its importers
    PROACTIVE_TURNS, PresenceTurnExecution, PresenceTurnExecutions, PresenceTurnNotStarted, _find_execution,
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
from ouroboros.utils import atomic_write_json, read_json_dict, utc_now_iso
from ouroboros.store import StoreError

log = logging.getLogger(__name__)


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
    # Version 2: ``{source_event_id, message}`` of the last input this output covers.
    reply_anchor: Mapping[str, Any] | None = None


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


def presence_result_from_stored(stored: Mapping[str, Any], task_id: str, drive_root=None) -> PresenceTurnResult:
    """One replay projection for cached turns and completed delegated work."""
    from ouroboros.task_finalization import TERMINAL_ORIGIN_MODEL_FINAL, provider_terminal_body, terminal_notice_text

    metadata = stored.get("metadata") if isinstance(stored.get("metadata"), dict) else {}
    text = metadata["presence_result_text"] if "presence_result_text" in metadata else stored.get("result")
    origin = str(stored.get("terminal_origin") or "")
    if origin == TERMINAL_ORIGIN_MODEL_FINAL and text and "result" in stored:
        raw = str(stored["result"] or "")
        # Undo only the recorded host composition, in either custody rendering. Older explicit reply
        # bodies could differ from the raw final; neither they nor deliberate emptiness are guessed away.
        notices = (terminal_notice_text(stored), terminal_notice_text(stored, include_empty=True))
        if any(notice and text == provider_terminal_body(raw, notice) for notice in notices):
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
    from ouroboros.place_reply import project_reply
    return project_reply(drive_root, metadata, PresenceTurnResult(
        outcome=outcome, text=text, task_id=task_id,
        work_ref=str(metadata.get("presence_work_ref") or ""),
        delivery_reporting_version=int(presence.get("delivery_reporting_version") == 1),
        output_ref=str(metadata.get("presence_output_ref") or "") if text else "",
        reply_anchor=metadata.get("presence_reply_anchor"),
    ))


# The conversation's own delivery facts a Presence result stamps on its task's metadata, below.
PRESENCE_DELIVERY_KEYS = ("presence_outcome", "presence_result_text", "presence_output_ref", "presence_work_ref",
                          "presence_reply_anchor", "presence_declaration")


def build_presence_result_event(task: dict[str, Any], text: str, ctx: Any, *, terminal_origin: str = "",
                                retain_scheduled_handoff: bool = False, silent: bool = False) -> dict[str, Any]:
    """Freeze typed delivery metadata before the ordinary durable result write.

    ``silent`` is the terminal's own quiet fact (``usage["wake_silent"]``): the candidate that
    actually ended the loop, never an earlier completion request a forced exit outlived."""
    from ouroboros.presence_continuation import _handoff_ref, terminal_output
    from ouroboros.task_finalization import HOST_AUTHORED_TERMINAL_ORIGINS

    from ouroboros.place_reply import prepare_focus_reply, reply_carrier
    carrier = reply_carrier(task.get("metadata") or {})
    if carrier:
        # Ordinary exits have already returned flags to the mind. Forced exits cannot buy a
        # new round: preserve the flag as a held reply instead of releasing unchecked speech.
        allowed, refusal = prepare_focus_reply(ctx, str(text or ""), quiet=terminal_origin != "model_final" or silent)
        if not allowed:
            from ouroboros.place_reply import record_notice
            from ouroboros.tool_access import canonical_data_root
            record_notice(canonical_data_root(ctx), place_id=carrier["place_id"],
                          conversation_key=carrier["conversation_key"], task_id=task["id"],
                          text="Reply held after finalization: " + refusal, identity=task["id"] + ":forced-hold")
            ctx._presence_completion = {"outcome": "silent", "quiet": True}
            ctx._presence_completion_accepted = True
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
    if completion.get("quiet"):
        # Not speech, so an owed handoff below polls with an empty body; the mind's own
        # silent words stay its finish note, host text never becomes one.
        note, text = (str(text or "") if silent and terminal_origin == "model_final" else ""), ""
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
    from ouroboros.presence_continuation import reply_anchor
    if anchor := (reply_anchor(ctx, metadata) if task.get("_presence_turn") else {}):  # never a promoted root's
        metadata["presence_reply_anchor"] = anchor
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
        **({"reply_anchor": anchor} if anchor else {}),
        "ts": utc_now_iso(),
    }


_GATES_LOCK = threading.Lock()
_GATES: dict[tuple[str, int], PresenceTurnGate] = {}
_LIVE_LOCK = threading.Lock()
_LIVE_PRESENCE_TASKS: set[tuple[str, str]] = set()


def presence_turn_is_live(task_id: str, *, drive_root: Path | None = None) -> bool:
    """Root-scoped local liveness; never an owner-addressable actor."""
    with _LIVE_LOCK:
        if drive_root is not None:
            return (str(Path(drive_root).resolve()), str(task_id or "")) in _LIVE_PRESENCE_TASKS
        return any(tid == str(task_id or "") for _root, tid in _LIVE_PRESENCE_TASKS)


def presence_execution_witness(task_id: str, *, drive_root: Path) -> bool | None:
    """Whether this turn's stack still runs: True witnessed, False absent, None unknown.

    This process's registry or the cross-process author census (one locked inode per live stack,
    kept while a parked author lends its conversation) witnesses it; a nonterminal row is no witness.
    An unreadable census is None: custody keeps a Stop pending on it (a false dead would settle it
    ``not_found`` while the turn spends), but it is never reported as a witnessed run.
    """
    if presence_turn_is_live(task_id, drive_root=drive_root):
        return True
    from ouroboros.conversation_lease import live_authors

    try:
        return str(task_id or "") in live_authors(drive_root=drive_root)
    except (OSError, ValueError):
        log.warning("Presence author census unreadable for %s; liveness unknown", task_id, exc_info=True)
        return None


def presence_turn_owned(row: Any, task_id: str, *, drive_root: Path) -> bool:
    """Live custody of an unsettled Presence row for ``task_has_live_ownership`` (#2012). Only such
    a row reads the census; a witnessed or unknown stack stays owned, as every unknown does there."""
    from supervisor.cancel_publication import live_presence_row

    return live_presence_row(row) and presence_execution_witness(task_id, drive_root=drive_root) is not False


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


async def admit_configured_gate(drive_root: Path, conversation_key: str, *, wait: bool = True) -> PresenceTurnLease:
    """Coroutine admission into the configured gate ``run_presence_turn`` would otherwise take.

    ``wait=False`` (continuation version 2) tries every resource once: a busy one is a typed
    retry that leaves the event in its transport's custody, holding nothing and waiting nowhere.
    """
    gate = _configured_gate(Path(drive_root))
    if wait:
        return await gate.admit(conversation_key)
    lease, busy = gate.try_admit(conversation_key)
    if lease is None:
        error = PresenceTurnError("presence_capacity_full", busy, disposition="retry")
        error.retry_after = 1 if busy == "conversation" else 5
        raise error
    return lease


def _stable_numeric_id(prefix: str, value: str) -> int:
    digest = hashlib.sha256(f"{prefix}\0{value}".encode("utf-8")).digest()
    return (1 << 40) + (int.from_bytes(digest[:6], "big") & ((1 << 40) - 1))


def presence_chat_id(conversation_key: str) -> int:
    """The unchanged ordinary Presence history identity, shared by call ingress."""
    return _stable_numeric_id("presence-conversation", conversation_key)


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
    if str(stored.get("status") or "") == STATUS_CANCELLED:
        return "presence_turn_cancelled"  # the owner's Stop: retained, never a reply or a re-run
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
    return presence_result_from_stored(stored, task_id, drive_root)


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
    """This task's chat rows (the dialogue store's task probe), in order; an attempt starts with
    its inbound row.

    A receipt addressed to another conversation (a tool send elsewhere) is not this conversation's
    delivery, however similar its text, so only receipts carrying this key are kept.
    """
    from ouroboros import dialogue_store

    rows = []
    for stored in dialogue_store.task_rows(Path(drive_root), [task_id], stream="chat"):
        row = stored.row
        if row.get("task_id") != task_id:
            continue
        transport = row.get("transport") if isinstance(row.get("transport"), Mapping) else {}
        if row.get("type") == "presence_delivery" and str(transport.get("conversation_key") or "") != conversation_key:
            continue
        rows.append(row)
    return rows


def _latest_receipts(rows: Sequence[Mapping[str, Any]], own_start: int = 0,
                     same_generation: bool = False) -> list[Mapping[str, Any]] | None:
    """Each delivery part's latest receipt row among *rows* (a later one settles an earlier), else None.

    An attempt's rows follow its inbound row, so that row means every receipt is there and the whole task
    counts. Without it the trace is incomplete (None), unless the caller vouches that this execution's own
    rows (``own_start`` onwards) all landed (``same_generation``)."""
    if any(row.get("direction") == "in" for row in rows):
        own_start = 0
    elif not same_generation:
        return None
    latest: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in rows[own_start:]:
        transport = row.get("transport") if isinstance(row.get("transport"), Mapping) else {}
        delivery = transport.get("delivery") if isinstance(transport.get("delivery"), Mapping) else {}
        if row.get("type") == "presence_delivery" and delivery.get("state"):
            latest[(str(delivery.get("delivery_id")), str(delivery.get("part_id")))] = row
    return list(latest.values())


def _turn_sends(rows: Sequence[Mapping[str, Any]], own_start: int = 0,
                same_generation: bool = False) -> tuple[list[str] | None, int]:
    """``(confirmed v1 sends or None, parts whose latest receipt is uncertain)`` over ``_latest_receipts``.

    None is unknown, not zero; an uncertain part may have landed. A wordless typed reaction receipt is an
    action (``_turn_actions``), no sent part; words sent beside a reaction are one."""
    from ouroboros.dialogue_provenance import wordless_reaction
    latest = _latest_receipts(rows, own_start, same_generation)
    if latest is None:
        return None, 0
    parts = [(str(row["transport"]["delivery"]["state"]), str(row.get("text") or ""))
             for row in latest if not wordless_reaction(row)]
    return ([text for state, text in parts if state in {"delivered", "accepted"}],
            sum(1 for state, _text in parts if state == "uncertain"))


def _turn_actions(rows: Sequence[Mapping[str, Any]], own_start: int = 0,
                  same_generation: bool = False) -> list[dict[str, Any]]:
    """The typed reaction receipts over ``_latest_receipts``, each with its part's state: actions, never speech.
    Without a complete trace there are none; an older connector's own ``reaction`` key is no typed action."""
    from ouroboros.dialogue_provenance import reaction_fact
    actions = []
    for row in _latest_receipts(rows, own_start, same_generation) or []:
        fact = reaction_fact(row["transport"].get("message"))
        if not fact:
            continue
        target = fact.get("target") if isinstance(fact.get("target"), Mapping) else {}
        actions.append({**{key: fact[key] for key in ("operation", "old", "new", "effect") if key in fact},
                        "target": {key: target[key] for key in ("message_id", "resolution") if key in target},
                        "state": str(row["transport"]["delivery"]["state"])})
    return actions


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
                         strict: bool = False, actions: Sequence[Mapping[str, Any]] = ()) -> None:
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
        **({"transport_actions": [dict(action) for action in actions]} if actions else {}),
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
                         actions=_turn_actions(_live_task_rows(drive_root, task_id, conversation_key), same_generation=True),
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
    rows = _live_task_rows(Path(drive_root), physical_id, conversation_key) if settled.delivery_reporting_version else []
    sends = _turn_sends(rows)[0] if settled.delivery_reporting_version else []
    outcome, message = _pointer_speech(stored, settled.outcome, settled.text)
    _write_previous_turn(Path(drive_root), conversation_key, physical_id, outcome=outcome, message=message,
                         sends=sends or [], actions=_turn_actions(rows), work_ref=settled.work_ref,
                         finished_at=str(stored.get("ts") or ""),
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
    if (stored.get("metadata") or {}).get("focus_heard_only"):
        return False
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
    from ouroboros.place_arrivals import heard_replay
    heard = heard_replay(drive_root, task_id, identity, continuation_version)
    if heard is not None:
        return heard
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
    client_message_id: str = "",
    ts: str = "",
    historical_import: Mapping[str, Any] | None = None,
) -> Any:
    """Write one room row; an inbound one returns its store receipt. ``ts``/``historical_import``:
    an imported earlier message keeps its provider send time and immutable marker (``presence_history``).
    Every new arrival, live or imported, keeps its provider send time and order (``source_order``)."""
    from ouroboros import chat_chain
    from ouroboros.dialogue_provenance import ANONYMOUS_REACTIONS, presence_provenance_from_task, reaction_fact
    from ouroboros.presence_history import source_order_block

    state = read_json_dict(drive_root / "state" / "state.json") or {}
    at = ts or utc_now_iso()
    block = source_order_block(event.message, received=at, sent=ts if historical_import is not None else "") \
        if direction == "in" else None
    row = {
        "ts": at,
        "session_id": state.get("session_id"),
        "direction": direction,
        **({"type": "place_arrival"} if direction == "in" and task["metadata"].get("focus") else {}),
        "chat_id": chat_id,
        "user_id": user_id,
        "text": text,
        "format": "markdown" if direction == "out" else "",
        "source": f"presence:{event.provider}",
        **({"sender_label": str((speaker or event.actor).get("display_name") or event.actor.get("username") or
                                (ANONYMOUS_REACTIONS if event.actor.get("kind") == "anonymous" and reaction_fact(event.message)
                                 else "")),
            "sender_session_id": str((speaker or event.actor).get("platform_actor_id") or event.actor.get("id") or "")}
           if direction == "in" else {}),
        "client_message_id": client_message_id or event.source_event_id,
        "presence_event_identity": presence_event_identity(str(task["metadata"]["presence"]["binding_id"]), event),
        "transport": {
            "provider": event.provider,
            "account_id": event.account_id,
            "conversation_id": event.conversation_id,
            "thread_id": event.thread_id,
            "conversation_key": event.conversation_key,
            "actor": dict(speaker or event.actor),
            "conversation": dict(event.conversation),
            "message": dict(event.message),
            **({"delivery": {"state": "authored"}} if direction == "out" else {}),
        },
        "presence_provenance": presence_provenance_from_task(task),
        "task_id": task_id,
        **({"place_id": task["metadata"]["presence"]["place_id"]}
           if task["metadata"]["presence"].get("place_id") else {}),
        **{key: task[key] for key in ("attachment_manifest", "attachment_manifest_ref") if key in task},
        **({"historical_import": dict(historical_import)} if historical_import is not None else {}),
        **({"source_order": block} if block is not None else {}),
    }
    if direction == "in" and (identity := getattr(event, "identity", None)) is not None:
        row["transport"]["identity"] = identity
    if direction == "in" and not task["metadata"].get("focus"):
        # The transport's exact contextual observations are this row's, read by its address (no copy).
        # They remain dated context of this event, never additional incoming speech or work.
        row["context_source_ref"] = dict(chat_chain.ROW_CONTEXT)
    from ouroboros import dialogue_store
    # Inbound acceptance and both live-call captions require storage before acknowledging the row.
    # Ordinary text replies keep the best-effort retry buffer.
    live = (task.get("metadata", {}).get("execution_transport") or {}).get("kind") == "live_conversation"
    if direction != "in" and not live:
        dialogue_store.append_quietly(drive_root, "chat", row)  # the reply: best effort, retry buffer
        return None
    try:
        return dialogue_store.append(drive_root, "chat", row)
    except (StoreError, OSError, ValueError) as exc:
        log.warning("Presence %s row not stored in %s", direction, drive_root, exc_info=True)
        raise PresenceTurnError("chat_log_unwritable", "chat") from exc


def current_previous_turn(drive_root: Path, conversation_key: str, *,
                          exclude_task_id: str = "") -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """The conversation's last executed turn and its still-continuing authors, read now.

    Read-only: no replay and no pointer write. An inbound turn and a scheduled root of the
    same conversation share it, so both see the conversation as it is, never as a copied
    carrier froze it when the work was scheduled (#1589).
    """
    previous_turn = _read_previous_turn(drive_root, conversation_key)
    if not previous_turn:
        return None, []
    from ouroboros.presence_continuation import continuation_status

    # Authors that yielded this conversation while their result awaits review: refs whose
    # facts are read from their own canonical rows, never copied into the pointer.
    open_turns = [{"task_id": ref, "status": continuation_status(load_task_result(drive_root, ref) or {}, drive_root)}
                  for ref in previous_turn.pop("open_turns", None) or [] if ref != exclude_task_id]
    if previous_turn.get("continuing"):
        # The pointer records the yield; whether that author still lives is its row's fact.
        status = continuation_status(load_task_result(drive_root, str(previous_turn.get("task_id") or "")) or {}, drive_root)
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
    return previous_turn, open_turns


def _build_task(
    admission: PresenceAdmission,
    event: PresenceTurnEvent,
    *,
    drive_root: Path,
    staged_files: Sequence[Path],
    physical_task_id: str = "",
    retained_manifest: list | None = None,
) -> dict[str, Any]:
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
    if admission.place is not None:
        from ouroboros.places import place_summary
        presence_context.update(place_id=admission.place.place_id, place=place_summary(admission.place))
        if admission.circle:  # absent outside a circle, so a frame recorded before circles still re-renders
            presence_context["circle"] = dict(admission.circle)
        presence_context["event"]["facts"] = list(event.facts) if event.facts is not None else ["legacy_addressed"]
    previous_turn, open_turns = current_previous_turn(drive_root, event.conversation_key,
                                                     exclude_task_id=physical_task_id)
    if open_turns:
        presence_context["open_turns"] = open_turns
    if previous_turn:
        presence_context["previous_turn"] = previous_turn
    metadata: dict[str, Any] = {
        "source": "presence",
        "client_message_id": event.source_event_id,
        "presence_event_identity": presence_event_identity(admission.binding_id, event),
        "inline_max_rounds": admission.inline_max_rounds,
        "presence": presence_context,
    }
    if admission.place is None and admission.model_slot == "light":
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
        "task_contract": ({"capability_ceiling": presence_ceiling_payload(admission.capability_ceiling)}
                          if admission.capability_ceiling is not None else {}),
    }
    if admission.place is not None:
        from ouroboros.presence_context import focus_metadata
        from ouroboros.consciousness_authority import apply_consciousness_authority
        metadata.pop("inline_max_rounds", None)
        metadata.update(focus_metadata(admission, event, task_id, chat_id, drive_root))
        from ouroboros.presence_context import _own_work_section
        from ouroboros.place_intake import conversation_work_section
        metadata["focus_work_here"] = "\n\n".join(part for part in (
            _own_work_section(drive_root, metadata["presence"], task_id),
            conversation_work_section(drive_root, metadata["presence"])) if part)
        apply_consciousness_authority(task)
    if admission.workspace_root:
        task.update(
            workspace_root=admission.workspace_root,
            workspace_mode="external",
            memory_mode="shared",
        )
    manifest = retained_manifest if retained_manifest is not None else stage_task_attachments(
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
    if not task["text"]:  # a file that did not arrive or content without words is never an empty event
        from ouroboros.presence_message_facts import carries_content
        task["text"] = ("(attachments received)" if manifest else
                        "(no text)" if carries_content(event.message) else "(empty presence event)")
    return attach_task_contract(task)


def capture_presence_event(admission: PresenceAdmission, event: PresenceTurnEvent, *, drive_root: Path,
                           staged_files: Sequence[Path] = ()) -> dict:
    """Retain source and attachments exactly once before every attention decision."""
    task_id = _task_id(admission, event)
    identity = presence_event_identity(admission.binding_id, event)
    from ouroboros.place_arrivals import intake_lock
    with intake_lock(drive_root, event.conversation_key):
        rows = _live_task_rows(drive_root, task_id, event.conversation_key)
        prior = next((row for row in rows if row.get("direction") == "in"), None)
        if prior is not None and prior.get("presence_event_identity") != identity:
            raise PresenceTurnError("presence_event_identity_conflict", "source_event_id", turn_ref=task_id)
        task = _build_task(admission, event, drive_root=drive_root, staged_files=staged_files,
                           retained_manifest=resolve_attachment_manifest(drive_root, task_id, prior) if prior else None)
        if prior is None:
            _log_dialogue(drive_root, direction="in", chat_id=task["chat_id"],
                          user_id=_stable_numeric_id("presence-actor-log", str(task["actor_id"])),
                          text=event.text, event=event, task=task, task_id=task_id)
        return task


def finish_turn(drive_root: Path, task: Mapping[str, Any], task_id: str, identity: str, event: PresenceTurnEvent,
                events: Sequence[Mapping[str, Any]], *, own_start: int) -> PresenceTurnResult:
    """A turn's end, still under its conversation lock: the durable terminal decides, a refused or unknown
    one is retained (``_refused_turn``), an unreported reply is logged at its conversation and the pointer
    names this turn without undoing a newer one. Shared by the Host's carrier and a moved turn's saved
    continuation (``presence_carrier.carry_saved_turn``)."""
    log_dialogue = functools.partial(_log_dialogue, Path(drive_root), chat_id=int(task["chat_id"]),
                                     event=event, task=task, task_id=task_id)
    # Only the durable terminal proves the outcome; a draft or RUNNING row cannot authorize acknowledgement.
    terminal = _stored_turn(Path(drive_root), task_id, identity)
    if terminal.get("wake_summary") and terminal.get("suggested_name") and not terminal.get("title"):
        from supervisor.worker_chat_lane import _broadcast_task_named
        _broadcast_task_named({"type": "task_named", "task_id": task_id, "suggested_name": terminal["suggested_name"]})
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
        reply_anchor=row.get("reply_anchor"),
    )
    from ouroboros.place_reply import project_reply
    result = project_reply(drive_root, task["metadata"], result)
    if result.outcome in {"message", "deferred"} and result.text and not result.delivery_reporting_version:
        log_dialogue(direction="out", user_id=0, text=result.text)
    # Still under the conversation lock. A replay writes only through the cached-replay repair, under
    # this same lock, and it leaves a pointer that names a newer turn alone.
    sends: list[str] | None = []
    actions: list[dict[str, Any]] = []
    if event.delivery_reporting_version:
        rows = _live_task_rows(Path(drive_root), task_id, event.conversation_key)
        sends = _turn_sends(rows, own_start, same_generation=True)[0]  # the store holds every row this execution wrote
        actions = _turn_actions(rows, own_start, same_generation=True)
    outcome, message = _pointer_speech(terminal, result.outcome, str(row.get("message") or result.text))
    _write_previous_turn(Path(drive_root), event.conversation_key, task_id, outcome=outcome, message=message,
                         sends=sends or [], actions=actions, work_ref=result.work_ref, finished_at=utc_now_iso(),
                         delivery=_delivery_state(event.delivery_reporting_version, sends, result.text or message, outcome),
                         finish_note=str(row.get("finish_note") or ""), lent_at=_first_lent_at(terminal))
    return result


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
    live_turn: Any = None,
    carrier_owner: Mapping[str, Any] | None = None,
    event_sink: Callable[[list], None] | None = None,
    request_waits: bool | None = None,
) -> PresenceTurnResult:
    """Run one bounded turn; adapters retain durable provider custody.

    ``admitted`` is a lease the caller already took for this conversation (``PresenceTurnGate.admit``):
    the turn runs under it instead of waiting on ``gate``, and the caller keeps releasing it.
    Either lease may be lent at a qualified review wait (``presence_continuation``).
    ``carrier_owner`` is the lane admission's execution-owner record (its carrier facts ride the
    row's own ``execution_owner``); ``event_sink`` receives the private owner-contact frames the
    runner returns to the bus when this process has no bus of its own (a carrier);
    ``request_waits`` states whether a Host request waits on this turn's result (the admission
    fact the post-task rule reads, ``agent_task_pipeline.presence_request_waits``).
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
        from ouroboros.presence_recovery import CUSTODY_QUEUED, CUSTODY_UNKNOWN, custody_refusal, queue_custody

        stored = _stored_turn(Path(drive_root), task_id, identity)
        custody = queue_custody(Path(drive_root), task_id, stored) if stored else ""
        # The queue keeps it: no author, no loss stamp. What a carrier cannot read, its own lane row already answered:
        # ``dispatch_presence_turn`` admits no id the queue holds, so its retried turn stays the lost attempt it is.
        if custody == CUSTODY_QUEUED or (custody == CUSTODY_UNKNOWN and carrier_owner is None):
            raise custody_refusal(stored, task_id, custody)
        live_key = (str(Path(drive_root).resolve()), task_id)
        with _LIVE_LOCK:
            _LIVE_PRESENCE_TASKS.add(live_key)
        try:
            lease.bind_author(task_id)
            return _run_current()
        finally:
            kept = False
            try:
                # Inline turns emit no task_done. Only a proven terminal row ends
                # the question/wait projections; an unknown attempt stays unknown.
                terminal = _stored_turn(Path(drive_root), task_id, identity)
                if terminal.get("status") in {STATUS_COMPLETED, STATUS_FAILED, STATUS_CANCELLED}:
                    from supervisor.queue_transitions import reconcile_terminal_task_projections
                    reconcile_terminal_task_projections(Path(drive_root), task_id)
                from ouroboros.presence_recovery import queue_kept
                kept = queue_kept(terminal)  # a parked turn's fence is the queue's
            except Exception:
                log.warning("Presence %s terminal projections remain unconfirmed", task_id, exc_info=True)
            try:
                from supervisor.queue_transitions import release_turn_acceptance_fence  # no task_done ends a turn
                if not kept:
                    release_turn_acceptance_fence(Path(drive_root), task_id)  # before "not live": no open row outlives it
            finally:  # a turn whose release failed still stops reading live
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
            retained_manifest=next((resolve_attachment_manifest(drive_root, task_id, r) for r in prior_rows
                                    if r.get("direction") == "in"), None),
        )
        if request_waits is not None:
            task["metadata"]["presence_request_waits"] = bool(request_waits)
        if live_turn is not None:
            from ouroboros.live_conversation import LiveConversationTurn
            if not isinstance(live_turn, LiveConversationTurn) or live_turn.spec.presence_binding_id != admission.binding_id:
                raise PresenceTurnError("presence_call_identity", "binding_id")
            task["metadata"]["execution_transport"] = live_turn.request
            task["model"], task["use_local_model"] = live_turn.model, False
        if event.continuation_version == 2:  # this turn's outputs name their reply anchor; never in the root carrier
            task["metadata"]["presence_continuation_version"] = event.continuation_version
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
        if admission.place is not None:
            from ouroboros.place_intake import decide_attention, settle_heard
            decision, allowance = decide_attention(Path(drive_root), admission, event)
            if decision == "silent":
                return settle_heard(Path(drive_root), task, event, allowance["reason"],
                                    captured_task_id=_task_id(admission, event))
            if decision == "retry":
                error = PresenceTurnError("consciousness_allowance_exhausted", "allowance")
                error.disposition, error.retry_after = "retry", allowance["retry_after"]
                raise error
            task["metadata"]["focus_allowance"] = allowance
            remaining = allowance.get("remaining_usd")
            if remaining is not None:
                task["metadata"]["root_cost_ceiling_usd"] = min(float(remaining), float(
                    task["metadata"].get("root_cost_ceiling_usd", remaining)))
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
        if live_turn is not None:
            agent._realtime_turn = live_turn
        from ouroboros.presence_continuation import bind_review_wait

        bind_review_wait(agent, lease=lease, event=event, task_id=task_id, identity=identity,
                         drive_root=Path(drive_root))
        if carrier_owner:
            task["_execution_owner"] = dict(carrier_owner)  # the lane's carrier facts ride the presence record
        # Read back RUNNING before execution: absence could authorize a duplicate after a crash.
        from ouroboros.task_results import write_task_result
        from ouroboros.task_status import execution_owner_record

        from ouroboros.place_arrivals import heard_replay, intake_lock, retain_focus_source
        try:
            with intake_lock(drive_root, event.conversation_key):
                heard = heard_replay(drive_root, task_id, identity, event.continuation_version)
                if heard is not None:
                    return heard
                write_task_result(Path(drive_root), task_id, STATUS_RUNNING,
                                  execution_owner=execution_owner_record(drive_root, task, "presence"),
                                  create_only=True, strict_existing_dict=True,
                                  metadata=task["metadata"], chat_id=chat_id,
                                  _is_direct_chat=True, source=task["source"], result="Task is running.")
                start = _stored_turn(Path(drive_root), task_id, identity)
                if str(start.get("status") or "") != STATUS_RUNNING or (
                        start.get("metadata") or {}).get("presence_event_identity") != identity:
                    raise ValueError("Presence start record was not bound to this event")
                if admission.place is not None:
                    retain_focus_source(drive_root, _task_id(admission, event))
        except (OSError, ValueError, StoreError) as exc:
            raise PresenceTurnError("presence_start_unwritable", "source_event_id", turn_ref=task_id) from exc
        try:
            from supervisor.worker_chat_lane import name_wake_on_admission
            name_wake_on_admission(drive_root, task, broadcast=(
                (lambda msg: event_queue.put(dict(msg))) if event_queue is not None else None))
            lease.intake_agent = agent
            # The author's intake is a durable fact beside its mailbox (``conversation_lease``):
            # a Host in another process offers an arrival to this author only while it reads.
            # Closed after the body returns (its wait scope already closed), never inside the
            # agent's release: a store write there sat between the final's publication and the
            # wait's close, where a consumer reads "no author left".
            from ouroboros.conversation_lease import close_author_intake, open_author_intake
            intake_open = False
            try:
                intake_open = open_author_intake(Path(drive_root), task_id, event.conversation_key)
            except Exception:
                log.debug("author intake fact not written for %s", task_id, exc_info=True)
            try:
                events = agent.handle_task(task)
            finally:
                if intake_open:
                    try:
                        close_author_intake(Path(drive_root), task_id)
                    except Exception:
                        log.debug("author intake fact not closed for %s", task_id, exc_info=True)
            # Unlike a pooled worker, this runner has no end-of-task event drain.
            # Private owner contact still uses the existing durable delivery owners
            # when there was no live queue (or its enqueue acknowledgement failed).
            from ouroboros.presence_owner_contact import flush_owner_events
            flush_owner_events(Path(drive_root), events, sink=event_sink)
        except BaseException:  # its executor ended without a terminal: record that end, never a re-run
            _notify_unresolved_turn(Path(drive_root), task_id)
            raise
        from ouroboros.presence_carrier import park_moved_turn

        # A moved turn that saved an exact pause parks in the queue (#2566): its waiters are answered from
        # that custody, and this author keeps the conversation until its own effects settled.
        parked = park_moved_turn(Path(drive_root), task, events, agent=agent, lease=lease, event=event,
                                 identity=identity, sink=event_sink)
        if parked is not None:
            raise parked
        return finish_turn(Path(drive_root), task, task_id, identity, event, events, own_start=own_start)

    if admitted is not None:
        if admitted.conversation_key != str(event.conversation_key or "").strip():
            raise PresenceTurnError("presence_admission_conversation_mismatch", "conversation_key")
        return execute()
    lease = (gate or _configured_gate(Path(drive_root))).acquire(event.conversation_key)
    try:
        return execute()
    finally:
        lease.release()


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
    "current_previous_turn",
    "presence_execution_witness",
    "presence_turn_is_live",
    "presence_turn_owned",
    "presence_turn_replay",
    "presence_turn_task_id",
    "run_presence_turn",
]
