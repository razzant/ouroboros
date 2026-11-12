"""Retained Presence facts, projected from the existing task and conversation sources.

The runner owns eligibility and execution custody. This module never retries a turn:
it says whether an attempt can still change (``attempt_ended``, the one rule the work
view and the orphan reconciler share) and records an ended attempt's observation in the
existing room biography. A task-local receipt suppresses ordinary repeated writes;
the stable summary id identifies an append/receipt crash gap without a history scan.

A moved turn's saved work is the queue's (``queue_custody``): its exact pause, its queue
row, or a queue continuation of it. Every reader asks that one rule before a start, a
loss stamp or an end, and a custody no read confirms stays ``unknown``, never confirmed.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

WORK_OUTPUT_VERSION = 1  # Canonical output_ref, producer task_id and independent child_work_ref.
CUSTODY_TERMINAL, CUSTODY_QUEUED, CUSTODY_UNKNOWN = "terminal", "queued", "unknown"
CUSTODY_HOLDS = frozenset({CUSTODY_QUEUED, CUSTODY_UNKNOWN})


def queue_kept(row: Mapping[str, Any]) -> bool:
    """The durable facts of saved work the queue keeps: a live exact pause with its source, a
    queue-written ``scheduled`` status, or a queue continuation (``pooled`` owner) of the turn."""
    from ouroboros.budget_pause import LIVE_PAUSE_STATES

    pause = row.get("budget_pause") if isinstance(row.get("budget_pause"), Mapping) else {}
    owner = row.get("execution_owner") if isinstance(row.get("execution_owner"), Mapping) else {}
    return bool((pause.get("state") in LIVE_PAUSE_STATES and pause.get("source_ref"))
                or str(row.get("status") or "") == "scheduled" or owner.get("kind") == "pooled")


def _queue_row(drive_root: Path, task_id: str) -> str | None:
    """Where the queue of ``drive_root`` holds this id, read only in the server process: ``pending``,
    ``running`` (a project-lane continuation), ``lane`` (a carried turn's own conversation row) or "".
    ``listed`` before this process restored the saved queue (the boot window between the Host and the
    restore): the saved document names it and nothing has ruled on it yet. None when unreadable here."""
    try:
        from ouroboros.utils import in_worker_process

        if in_worker_process():
            return None
        from supervisor import queue as q
        from supervisor.queue_snapshot import saved_queue_unowned

        # This process's live maps answer for its own root once it took (restored or published) the saved queue;
        # before that, the saved document is the queue's only word on this id.
        if Path(q.DRIVE_ROOT).resolve() != Path(drive_root).resolve() or saved_queue_unowned():
            from ouroboros.queue_observation import load_queue_document

            document = load_queue_document(Path(drive_root))
            if document.get("_snapshot_invalid"):
                return None
            return "listed" if any(isinstance(row, dict) and str(row.get("id") or row.get("task_id") or "") == task_id
                                   for lane in ("running", "pending") for row in document.get(lane) or ()) else ""
        if not q._queue_lock.acquire(timeout=5.0):
            return None
        try:
            if any(str(row.get("id") or "") == task_id for row in q.PENDING):
                return "pending"
            meta = q.RUNNING.get(task_id)
            if not isinstance(meta, dict):
                return ""
            return "lane" if meta.get("lane") == "conversation" else "running"
        finally:
            q._queue_lock.release()
    except Exception:
        return None


def queue_custody(drive_root: Path | None, task_id: str, stored: Mapping[str, Any] | None = None) -> str:
    """The queue's custody of one Presence id, read before any start, loss stamp or end.

    ``terminal``: a canonical terminal, which outranks every pause fact. ``queued``: saved work the
    queue keeps (``queue_kept``), or its PENDING row or project-lane continuation in this server's live
    queue. ``unknown``: an unreadable row, an unreadable queue (a worker, a lock timeout, a failed read), or
    the saved queue document naming the id before this server restored it; never presented as confirmed.
    ``""``: only after a read that does not hold the id, or a turn's own carried conversation row, so its own
    liveness rules apply. Only a moved text turn's work is ever saved (``presence_carrier.moved_text_turn``;
    an unreadable binding may be one): any other turn reads ``""`` here, as before. Never raises.
    """
    if drive_root is None or not task_id:
        return ""
    from ouroboros.presence_carrier import moved_text_turn

    try:
        from ouroboros.task_results import load_task_result

        row = dict(stored) if stored is not None else (load_task_result(drive_root, task_id, strict=True) or {})
    except Exception:
        try:  # an unreadable row: only a Project binding (a possible move) leaves its custody unknown
            from ouroboros.projects_registry import project_binding_for_task

            binding = project_binding_for_task(Path(drive_root), task_id, strict=True) or {}
        except Exception:
            return CUSTODY_UNKNOWN
        return CUSTODY_UNKNOWN if binding.get("project_id") and not binding.get("origin_bound") else ""
    if str(row.get("status") or "") in {"completed", "failed", "cancelled"}:
        return CUSTODY_TERMINAL
    if not moved_text_turn(drive_root, task_id, row.get("metadata"), unreadable=True):
        return ""
    if queue_kept(row):
        return CUSTODY_QUEUED
    where = _queue_row(Path(drive_root), task_id)
    if where is None or where == "listed":
        return CUSTODY_UNKNOWN
    return CUSTODY_QUEUED if where in {"pending", "running"} else ""


def custody_refusal(stored: Mapping[str, Any], task_id: str, custody: str) -> Any:
    """The typed retained answer of an event whose turn the queue keeps (``presence_turn_parked``) or whose
    custody no read confirms (``presence_turn_custody_unknown``): never a start, a new author or a loss."""
    from ouroboros.presence_events import PresenceTurnError

    facts = retained_turn_facts(stored, task_id)
    code = (facts["code"] if custody == CUSTODY_TERMINAL else
            "presence_turn_parked" if custody == CUSTODY_QUEUED else "presence_turn_custody_unknown")
    error = PresenceTurnError(code, "source_event_id", turn_ref=task_id, work_ref=str(facts.get("work_ref") or ""),
                              disposition="retained", delivery_reporting_version=facts["delivery_reporting_version"])
    if custody != CUSTODY_TERMINAL:
        error.custody = custody
    return error


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
    missing, live or unobservable witness, is no end: the attempt stays pending. Nor is a
    controller's end or an end marker while the queue keeps (or may keep) the same id
    (``queue_custody``): a moved turn's saved work goes on there.
    """
    from ouroboros.presence_runner import presence_turn_is_live
    from ouroboros.review_operation import controller_state

    task_id = str(stored.get("task_id") or stored.get("id") or "")
    if presence_turn_is_live(task_id, drive_root=drive_root):
        return False
    if str(stored.get("status") or "") in {"completed", "failed", "cancelled"}:
        return True
    if task_id and queue_custody(drive_root, task_id, stored) in CUSTODY_HOLDS:
        return False
    if stored.get("presence_recovery_recorded"):
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
