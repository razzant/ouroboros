"""Durable pre-STARTED invocation replay, extracted from custody's facade."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def pending_invocations(
    drive_root: Any, rows: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Return request rows with no bound run and no definite refusal."""

    from ouroboros import delegate_custody as c

    found: Dict[str, Dict[str, Any]] = {}
    state: Dict[str, str] = {}
    source = rows if rows is not None else c._iter_rows(c.event_log_path(drive_root))
    for row in source:
        invocation_id = str(row.get("invocation_id") or "")
        if not invocation_id:
            continue
        kind = str(row.get("type") or "")
        if kind == c.START_REQUESTED and invocation_id not in found:
            found[invocation_id] = {
                "invocation_id": invocation_id,
                "task_id": str(row.get("task_id") or ""),
                "request": row.get("request") if isinstance(row.get("request"), dict) else None,
                "route": str(row.get("route") or ""),
                "project_id": str(row.get("project_id") or ""),
                "project_owned": bool(row.get("project_owned")),
                "project_persistent": bool(row.get("project_persistent")),
                "idempotency_key": str(row.get("idempotency_key") or ""),
                "root_task_id": str(row.get("root_task_id") or ""),
                "parent_task_id": str(row.get("parent_task_id") or ""),
                "snapshot_id": str(row.get("snapshot_id") or ""),
                "execution_root": str(row.get("execution_root") or ""),
                "baseline_sha": str(row.get("baseline_sha") or ""),
                "target_root": str(row.get("target_root") or ""),
                "authority_source": str(row.get("authority_source") or ""),
                "resource_ref": row.get("resource_ref") if isinstance(row.get("resource_ref"), dict) else {},
                "selected_subagent_id": str(row.get("selected_subagent_id") or ""),
                "config_fingerprint": str(row.get("config_fingerprint") or ""),
                "work_order_fingerprint": str(row.get("work_order_fingerprint") or ""),
                "authority_fingerprint": str(row.get("authority_fingerprint") or ""),
            }
        elif kind == c.STARTED:
            state[invocation_id] = "started"
        elif (
            kind == c.START_FAILED
            and row.get("definite") is True
            and state.get(invocation_id) != "started"
        ):
            state[invocation_id] = "failed_definite"
    return [
        record for invocation_id, record in found.items()
        if state.get(invocation_id, "pending") == "pending"
        and isinstance(record["request"], dict) and record["request"]
    ]


__all__ = ["pending_invocations"]
