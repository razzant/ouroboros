"""A frozen schedule admission is Continue's source, never invented owner words."""
from __future__ import annotations

import copy


def admission_basis(result, task_id):
    receipt = result.get("schedule_admission")
    if not isinstance(receipt, dict):
        return None
    task = receipt.get("task") or {}
    if not isinstance(task, dict):
        return None
    meta = task.get("metadata") or {}
    if not isinstance(meta, dict):
        return None
    occurrence = meta.get("schedule_occurrence") or {}
    if not isinstance(occurrence, dict):
        return None
    if (task.get("id") != task_id or not receipt.get("token") or not receipt.get("schedule_id")
            or occurrence.get("token") != receipt["token"]
            or occurrence.get("schedule_id") != receipt["schedule_id"] or not task.get("text")):
        return None
    basis = meta.get("schedule_basis") or {}
    if not isinstance(basis, dict):
        return None
    return {"source": "schedule_admission", "schedule_id": receipt["schedule_id"],
            "occurrence_token": receipt["token"], "objective": str(task["text"]),
            "context": copy.deepcopy(task.get("context", "")),
            "author": copy.deepcopy(basis.get("author") or meta.get("objective_author") or "unknown (legacy admission)"),
            "origin": copy.deepcopy(basis.get("origin") or meta.get("followup_origin") or receipt.get("followup_origin") or {})}


def recorded_origin(result):
    """The origin id the host retained for the schedule admission this record's work runs on, else "".

    Its own receipt, or the basis a Continue keeps by value when the record's host admission names it: metadata alone
    can be any caller's words. Reads nothing beyond the record it is given."""
    from ouroboros.dialogue_provenance import presence_continue_record

    basis = admission_basis(result, result.get("task_id"))
    if basis is None and presence_continue_record(result):
        kept = result
        for key in ("metadata", "continuation", "owner_sources"):
            kept = kept.get(key) if isinstance(kept.get(key), dict) else {}
        basis = kept.get("schedule")
    return str((basis.get("origin") or {}).get("task_id") or "") if valid_basis(basis) else ""


def valid_basis(basis):
    return (isinstance(basis, dict) and basis.get("source") == "schedule_admission"
            and all(isinstance(basis.get(key), str) and basis[key].strip()
                    for key in ("schedule_id", "occurrence_token", "objective"))
            and isinstance(basis.get("origin", {}), dict))
