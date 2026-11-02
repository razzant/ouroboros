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


def valid_basis(basis):
    return (isinstance(basis, dict) and basis.get("source") == "schedule_admission"
            and all(isinstance(basis.get(key), str) and basis[key].strip()
                    for key in ("schedule_id", "occurrence_token", "objective"))
            and isinstance(basis.get("origin", {}), dict))
