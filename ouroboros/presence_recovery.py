"""Retained Presence facts, projected from the existing task and conversation sources.

The runner owns eligibility and execution custody. This module never retries a turn:
it says whether an attempt can still change (``attempt_ended``, the one rule the work
view and the orphan reconciler share) and records an ended attempt's observation in the
existing room biography. A task-local receipt suppresses ordinary repeated writes;
the stable summary id identifies an append/receipt crash gap without a history scan.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

WORK_OUTPUT_VERSION = 1  # Canonical output_ref, producer task_id and independent child_work_ref.


def retained_turn_facts(stored: Mapping[str, Any], task_id: str) -> dict[str, Any]:
    """Facts for an already-classified retained turn, never an authored answer.

    No absence of a tool/delivery record proves absence of an effect. A child
    reference is copied only from actual admitted work; the original is not its
    own child. Callers decide retained versus retry from their current authority.
    """
    metadata = stored.get("metadata") if isinstance(stored.get("metadata"), Mapping) else {}
    presence = metadata.get("presence") if isinstance(metadata.get("presence"), Mapping) else {}
    event = presence.get("event") if isinstance(presence.get("event"), Mapping) else {}
    reason, status = str(stored.get("reason_code") or ""), str(stored.get("status") or "")
    code = ("presence_resources_unavailable" if reason == "resource_refusal_no_resend" else
            "presence_turn_cancelled" if status == "cancelled" else "presence_attempt_outcome_unknown")
    facts: dict[str, Any] = {
        "code": code, "turn_ref": task_id, "recorded_status": status,
        "reason_code": reason, "source_event_id": str(event.get("source_event_id") or ""),
        "conversation_key": str(event.get("conversation_key") or ""),
        "delivery_reporting_version": int(presence.get("delivery_reporting_version") == 1),
    }
    continuation = stored.get("presence_continuation")
    continuation = continuation if isinstance(continuation, Mapping) else {}
    initial = continuation.get("initial") if isinstance(continuation.get("initial"), Mapping) else {}
    child = metadata.get("presence_work_ref") or continuation.get("child_work_ref") or initial.get("work_ref")
    if isinstance(child, str) and child.strip():
        facts["work_ref"] = child.strip()
    unknown = metadata.get("presence_unknown_outcome")
    if isinstance(unknown, Mapping):
        facts["uncertainty"] = {key: str(unknown[key]) for key in ("source", "error_kind", "operation_id")
                                if unknown.get(key) not in (None, "")}
    reconciled = stored.get("status_reconciled_from")
    if reconciled:
        facts["status_reconciled_from"] = str(reconciled)
    return facts


def record_unresolved_turn(drive_root: Path, task_id: str) -> bool:
    """Append one source-backed room observation under the runner's per-task lock.

    This is not a terminal transition or an owner question. Keep the CURRENT
    status when stamping the append receipt, even if a late result landed after
    the observation. A failed append gets no receipt and can be tried again.
    """
    from ouroboros.dialogue_provenance import presence_provenance_fields
    from ouroboros.presence_runner import _terminal_refusal
    from ouroboros.project_dialogue import append_canonical_task_summary
    from ouroboros.task_results import load_task_result, write_task_result
    from ouroboros.utils import utc_now_iso

    stored = load_task_result(drive_root, task_id, strict=True) or {}
    metadata = stored.get("metadata") if isinstance(stored.get("metadata"), Mapping) else {}
    presence = metadata.get("presence")
    if not isinstance(presence, Mapping) or not stored.get("status") or not _terminal_refusal(stored):
        return False  # a canonical answer needs no retained observation
    summary_id = f"presence-retained:{task_id}"
    receipt = stored.get("presence_recovery_recorded")
    if isinstance(receipt, Mapping) and receipt.get("summary_id") == summary_id:
        return False
    facts = retained_turn_facts(stored, task_id)
    event = presence.get("event") if isinstance(presence.get("event"), Mapping) else {}
    observed_at = utc_now_iso()
    row = {
        "ts": observed_at, "direction": "system", "type": "task_summary", "text": "",
        "summary_kind": "host_task_facts", "summary_id": summary_id, "task_id": task_id,
        "chat_id": stored.get("chat_id"), "status": str(stored["status"]),
        "_is_direct_chat": bool(stored.get("_is_direct_chat")),
        "source": str(stored.get("source") or ""), "presence_recovery": facts,
        "source_event": copy.deepcopy(dict(event)), **presence_provenance_fields(stored),
        "outcome_final": False, "outcome_authority": "retained_presence_observation",
    }
    if isinstance(presence.get("observed_text"), str):
        row["source_text"] = presence["observed_text"]
    if not append_canonical_task_summary(drive_root, row):
        return False

    def stamp(current: dict, _patch: dict) -> dict | None:
        if not current or not current.get("status"):
            return None
        return {"status": current["status"], "presence_recovery_recorded": {
            "summary_id": summary_id, "written_at": observed_at,
        }}

    write_task_result(drive_root, task_id, str(stored["status"]), strict_existing_dict=True,
                      _field_projector=stamp)
    return True


def attempt_ended(stored: Mapping[str, Any], drive_root: Path | None = None) -> bool:
    """Whether nothing can still write this Presence attempt's row.

    A positive live observation here wins over every record: an execution still finishing
    in this process (it may yet back-fill its child) is never cut short. Then only a positive
    end counts: a recorded terminal; the runner's own end marker (``presence_recovery_recorded``:
    it saw this attempt's executor end, or proved it lost under the conversation gate, an
    attempt fact a living process cannot contradict); or a gone controller witness (the
    execution owner's, or a legacy parked author's). A missing local registry entry, or a
    missing, live or unobservable witness, is no end: the attempt stays pending.
    """
    from ouroboros.presence_runner import presence_turn_is_live
    from ouroboros.review_operation import controller_state

    if presence_turn_is_live(str(stored.get("task_id") or stored.get("id") or ""), drive_root=drive_root):
        return False
    if str(stored.get("status") or "") in {"completed", "failed", "cancelled"} or stored.get("presence_recovery_recorded"):
        return True
    owner = stored.get("execution_owner") if isinstance(stored.get("execution_owner"), Mapping) else {}
    parked = stored.get("presence_continuation") if isinstance(stored.get("presence_continuation"), Mapping) else {}
    return controller_state(owner.get("controller") or parked.get("author_controller")) == "dead"


def retained_work_view(stored: Mapping[str, Any], reference: str,
                       drive_root: Path | None = None) -> tuple[int, dict[str, Any]]:
    """Pending while the attempt's producer can still write it; terminal once it cannot.

    An ended attempt without a canonical answer is never ``completed``: it is ``failed``
    (``cancelled`` when recorded so) with no authored text, its typed facts and any
    released outputs. Its real child keeps its own reference and is observed on its own.
    """
    metadata = stored.get("metadata") if isinstance(stored.get("metadata"), Mapping) else {}
    presence = metadata.get("presence") if isinstance(metadata.get("presence"), Mapping) else {}
    from ouroboros.presence_continuation import work_view

    status = str(stored.get("status") or "")
    from ouroboros.presence_runner import _terminal_refusal, presence_result_from_stored

    continued = work_view(stored, reference, drive_root)  # a continuing author's own poll
    if status in {"completed", "failed", "cancelled"} and not _terminal_refusal(stored):
        # A canonical answer is final the moment it is recorded.
        if continued is not None:
            return continued
        result = presence_result_from_stored(stored, reference, drive_root)
        return 200, {"ok": True, "status": status, "outcome": result.outcome, "text": result.text,
                     "work_ref": reference, "task_id": result.task_id, "output_ref": result.output_ref,
                     "child_work_ref": result.work_ref,
                     "delivery_reporting_version": result.delivery_reporting_version}
    if not attempt_ended(stored, drive_root):
        if continued is None:
            return 202, {"ok": True, "status": "pending", "work_ref": reference, "task_id": reference,
                         "delivery_reporting_version": presence.get("delivery_reporting_version", 0)}
        # A live author stays observed even after its row recorded a terminal without an answer:
        # its execution may still back-fill a child before it ends.
        return 202, {**{key: value for key, value in continued[1].items()
                        if key not in {"outcome", "text", "output_ref"}}, "status": "pending"}
    # Already released continuation outputs keep their exact identities. A draft
    # salvaged from the unanswered final round is never newly selected speech.
    facts = retained_turn_facts(stored, reference)
    child = facts.pop("work_ref", "") or (continued[1].get("child_work_ref", "") if continued else "")
    released = {key: continued[1][key] for key in (
        "outputs", "continuation_ref", "continuation_version") if key in continued[1]} if continued else {}
    return 200, {"ok": True, "status": "cancelled" if status == "cancelled" else "failed",
                 "outcome": "silent", "text": "", "output_ref": "", **facts,
                 "work_ref": reference, "task_id": reference, "child_work_ref": child, **released}
