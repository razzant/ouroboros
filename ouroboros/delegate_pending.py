"""Durable pre-STARTED invocation replay, extracted from custody's facade."""

from __future__ import annotations

import copy
import pathlib
from typing import Any, Dict, List, Optional


def pending_invocations(
    drive_root: Any, rows: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Return request rows with no bound run and no definite refusal."""

    from ouroboros import delegate_custody as c

    from ouroboros.delegate_custody_current import active, pending
    if rows is None and active(drive_root):
        return pending(drive_root)
    found: Dict[str, Dict[str, Any]] = {}
    state: Dict[str, str] = {}
    source = rows if rows is not None else c.custody_rows(drive_root)
    for row in source:
        invocation_id = str(row.get("invocation_id") or "")
        if not invocation_id:
            continue
        kind = str(row.get("type") or "")
        if kind == c.START_REQUESTED and invocation_id not in found:
            found[invocation_id] = {
                "invocation_id": invocation_id,
                "task_id": str(row.get("task_id") or ""),
                "surface": str(row.get("surface") or ""),
                "slot_id": str(row.get("slot_id") or ""),
                "operation_id": str(row.get("operation_id") or ""),
                "request": row.get("request") if isinstance(row.get("request"), dict) else None,
                "request_ref": row.get("request_ref"),
                "request_locator": row.get("request_locator"),
                "route": str(row.get("route") or ""),
                "project_id": str(row.get("project_id") or ""),
                "project_owned": bool(row.get("project_owned")),
                # Absence is a fact (legacy rows): the recovery fallback
                # derives persistence from the stored request.
                **({"project_persistent": bool(row["project_persistent"])}
                   if "project_persistent" in row else {}),
                "idempotency_key": str(row.get("idempotency_key") or ""),
                "root_task_id": str(row.get("root_task_id") or ""),
                "parent_task_id": str(row.get("parent_task_id") or ""),
                "category": str(row.get("category") or "subagent"),
                "source": str(row.get("source") or "delegated_subagent"),
                **{key: str(row.get(key) or "") for key in c.REVIEW_ATTRIBUTION_KEYS},
                "snapshot_id": str(row.get("snapshot_id") or ""),
                "execution_root": str(row.get("execution_root") or ""),
                "baseline_sha": str(row.get("baseline_sha") or ""),
                "target_root": str(row.get("target_root") or ""),
                "authority_source": str(row.get("authority_source") or ""),
                # A pending continuation names the run whose snapshot it takes over.
                **{key: str(row.get(key) or "") for key in ("continuation_of", "capture_id", "snapshot_task_id")},
                # Copies: the source rows may be the shared, read-only custody memo.
                "resource_ref": copy.deepcopy(row.get("resource_ref")) if isinstance(row.get("resource_ref"), dict) else {},
                "selected_subagent_id": str(row.get("selected_subagent_id") or ""),
                "config_fingerprint": str(row.get("config_fingerprint") or ""),
                "work_order_fingerprint": str(row.get("work_order_fingerprint") or ""),
                "work_order_coverage": str(row.get("work_order_coverage") or ""),
                "authority_fingerprint": str(row.get("authority_fingerprint") or ""),
                "work_order_source_request": (
                    copy.deepcopy(row.get("work_order_source_request"))
                    if isinstance(row.get("work_order_source_request"), dict) else {}
                ),
            }
        elif kind == c.STARTED:
            state[invocation_id] = "started"
        elif (
            kind == c.START_FAILED
            and row.get("definite") is True
            and state.get(invocation_id) != "started"
        ):
            state[invocation_id] = "failed_definite"
    pending = []
    for invocation_id, record in found.items():
        if state.get(invocation_id, "pending") != "pending":
            continue
        # Resolve only survivors, not every historical start on each sweep.
        body = request_body(drive_root, record)
        ref = record.pop("request_ref")
        locator = record.pop("request_locator")
        # An unreadable stored body does not discharge the pending start.
        if body or ref is not None or locator is not None:
            record["request"] = body
            pending.append(record)
    return pending


def request_body(drive_root: Any, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Resolve the canonical replay envelope, legacy inline first, then raw CAS.

    The raw CAS blob holds the full unredacted envelope, stored before the
    start-requested row. Unreadable references leave the request unknown for
    the caller's existing refusal paths; never rebuild a paid invocation from
    current settings.
    """
    inline = row.get("request")
    if isinstance(inline, dict) and inline:
        return inline
    locator = row.get("request_locator")
    if locator is not None:
        # A memo row carries the legacy inline body's location, not the body.
        from ouroboros.delegate_custody_memo import read_locator_request

        located = read_locator_request(drive_root, locator, invocation_id=str(row.get("invocation_id") or ""))
        if located is not None:
            return located
    ref = row.get("request_ref")
    if not isinstance(ref, dict) or not ref:
        return None
    from ouroboros.observability import read_blob_ref

    try:
        body = read_blob_ref(pathlib.Path(drive_root), ref, expected_kind="json")
    except Exception:
        return None
    return body if isinstance(body, dict) and body else None


__all__ = ["pending_invocations", "request_body"]
