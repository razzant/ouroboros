"""Frozen 761a992b lifecycle writer and fanout producer, without a Git dependency.

These function bodies are verbatim exports. Unchanged lifecycle/store helpers
remain production; the companion loader replaces only its writer import. Tests
bind the producer's lazy loader import to that frozen loader. This fixture does
not import the candidate's write_task_result (the previous test's blind spot).
The private review evidence additionally runs seven complete baseline modules.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from ouroboros.task_results import (
    _TRULY_TERMINAL_STATUSES,
    _is_status_regression,
    _refuse_occupied,
    lifecycle_named,
    merge_review_projection,
    normalize_task_result_cost_planes,
    patch_document,
    preserve_terminal_attempt,
    require_writable_task_result_schema,
    resolve_task_lineage,
    stamp_task_result_schema,
    terminal_time_patch,
    utc_now_iso,
    validate_task_id,
)
from ouroboros.tools.registry import ToolContext
from ouroboros.utils import append_jsonl

log = logging.getLogger(__name__)
_SCHEDULE_EMIT_LOCK = threading.Lock()


def write_task_result(
    results_drive_root: Any,
    task_id: str,
    status: str,
    *,
    _field_projector: Optional[Callable[[Dict[str, Any], Dict[str, Any]], Optional[Dict[str, Any]]]] = None,
    strict_existing_dict: bool = False,
    create_only: bool = False,
    _terminal_observed: bool = False,
    _terminal_time_source: Optional[Dict[str, Any]] = None,
    **fields: Any,
) -> Dict[str, Any]:
    """Merge-write a task result inside one store write transaction.

    Workers, supervisor and gateways share CURRENT under that transaction: the monotonic
    reducer decides terminal races, so natural completion wins a late cancel.
    Discarding a result remains the separate explicit ``discard_child_result``.
    ``_field_projector`` runs after review-publication selection under the same
    lock, deriving fields/status from CURRENT or publishing its verified refs;
    repeating the incoming-review merge afterward would undo that handoff.
    ``strict_existing_dict`` is accepted for its callers: every stored row is a dict.
    ``create_only`` initializes only absence, before projection or timestamps.
    The body is one store transaction: the merge over CURRENT, the typed columns and
    obligation flags of the merged document, the custody and handoff riders, COMMIT.
    """
    path = f"task:{validate_task_id(task_id)}"
    if create_only:
        _refuse_occupied(results_drive_root, task_id, write=True)
    explicit_ts = str(fields.pop("ts", "") or "")
    from ouroboros.task_custody import capture_owner_mail, capture_unread_mail, merge_unread_mail
    from ouroboros.task_store import Reprepare, prepared_write

    prepared: Dict[str, Any] = {}
    transition: Dict[str, bool] = {}  # set by the reducer when THIS write turns the row truly terminal

    def _write(needs: set) -> Dict[str, Any]:
        # TZ-1 V10: the mailbox bytes are read BEFORE the row lock (a bounded union happens
        # under it); a write the projector turns terminal, or a first row, asks once
        # (``Reprepare``) and the whole write repeats with that input prepared out here
        # (what an earlier round prepared is kept, not read again).
        if "mail" not in prepared and (status in _TRULY_TERMINAL_STATUSES or "mail" in needs):
            prepared["mail"] = (capture_unread_mail(results_drive_root, task_id), capture_owner_mail(results_drive_root, task_id))
        if "rollup" not in prepared and (create_only or "rollup" in needs) and "swarm_fanout_rollup" not in fields:
            from ouroboros.swarm_rollup import initial_swarm_rollup
            prepared["rollup"] = initial_swarm_rollup(results_drive_root)
        return patch_document(results_drive_root, task_id, lambda existing: _merge(existing, prepared))

    def _merge(existing: Dict[str, Any], prepared: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        transition.clear()
        if strict_existing_dict and existing and (
            str(existing.get("task_id") or "") != str(task_id)
            or not isinstance(existing.get("status"), str)
            or not str(existing.get("status") or "").strip()
        ):
            raise ValueError(
                f"task result authority is unreadable or invalid: {path}"
            )
        # ABI 7.0: every write stamps the row; another schema's row is never downgraded.
        require_writable_task_result_schema(existing, path)
        if create_only and existing:
            return None
        prepared_fields = dict(fields)
        if not existing and "swarm_fanout_rollup" not in prepared_fields:
            if "rollup" not in prepared:
                raise Reprepare("rollup")
            prepared_fields["swarm_fanout_rollup"] = prepared["rollup"]
        if "review_projection" in prepared_fields:
            prepared_fields["review_projection"] = merge_review_projection(
                existing.get("review_projection"), prepared_fields["review_projection"],
            )
        projected_fields = _field_projector(existing, {**prepared_fields, "status": status}) if _field_projector else prepared_fields
        if projected_fields is None:  # projector saw a terminal/stale row: no mutation
            return None
        projected_status = str(projected_fields.pop("status", status))
        projected_fields.pop("terminal_time", None)  # ordinary fields cannot manufacture clock authority
        # Monotonic lifecycle: no stale mirror may overwrite a terminal outcome.
        existing_status = str(existing.get("status") or "")
        if existing and _is_status_regression(existing_status, projected_status):
            # Debugging a "stuck" task: the only signal that a stale/late write was dropped.
            log.debug("Blocked status regression %s -> %s for task %s",
                      existing.get("status"), projected_status, task_id)
            return None
        # The lifecycle axis names the status this accepted write lands: in the axes it carries,
        # or in the current axes it keeps across a status change. Same-status enrichment leaves
        # stored axes alone, so untouched history is never migrated.
        carried = "outcome_axes" in projected_fields
        if carried or projected_status != existing_status:
            axes = projected_fields["outcome_axes"] if carried else existing.get("outcome_axes")
            named = lifecycle_named(axes, projected_status)
            if named is not axes:
                projected_fields["outcome_axes"] = named
        if isinstance(projected_fields.get("delegate_terminal_reconciliation"), dict):
            # #1573: no writer, a wholesale terminal one included, unpublishes a retained output.
            from ouroboros.delegate_terminal import keep_published_receipts
            projected_fields["delegate_terminal_reconciliation"] = keep_published_receipts(
                existing, projected_fields["delegate_terminal_reconciliation"])
        # Only this accepted transition can originate terminal delivery debt:
        # enrichment/replica fields cannot adopt historical terminal rows or
        # erase provenance persisted before the separate readiness write.
        projected_fields.pop("canonical_terminal_projection_origin", None)
        if projected_status in _TRULY_TERMINAL_STATUSES and existing_status not in _TRULY_TERMINAL_STATUSES:
            if "mail" not in prepared:
                raise Reprepare("mail")  # the projector turned the row terminal: the capture happens outside
            transition["terminal"] = True
            captured, owner_mail = prepared["mail"]
            merged = {**existing, **projected_fields}
            lineage_keys = ("root_task_id", "parent_task_id", "delegation_role", "original_task_id", "timeout_retry_from")
            if resolve_task_lineage(task_id, metadata=merged.get("metadata"),
                                    **{key: merged.get(key) for key in lineage_keys})["is_root_task"]:
                projected_fields["canonical_terminal_projection_origin"] = "terminal_transition"
                # Batch4: the owner's exact rows, ACKed ones included, outlive the mailbox cleanup.
                projected_fields["owner_mailbox"] = merge_unread_mail(existing.get("owner_mailbox"), owner_mail)
            # TZ-1 V10: this accepted transition keeps the mail no attempt read (no ACK written);
            # later late mail joins through settlement, never through a rejected write.
            projected_fields["unread_mailbox"] = merge_unread_mail(projected_fields.get("unread_mailbox"), captured)
        # Unread-mail custody only grows: no replica or partial write can shrink it.
        projected_fields["unread_mailbox"] = merge_unread_mail(
            existing.get("unread_mailbox"), projected_fields.get("unread_mailbox"))
        if projected_fields["unread_mailbox"] is None:
            projected_fields.pop("unread_mailbox")
        projected_fields = preserve_terminal_attempt(existing, projected_fields)
        now = utc_now_iso()
        projected_fields.update(terminal_time_patch(existing, projected_fields, status=projected_status,
            task_id=task_id, observed_at=now if _terminal_observed else None, replica=_terminal_time_source))
        # ABI-3: shared deep cost normalization handles both rows/nested planes; incoming values win.
        return stamp_task_result_schema(normalize_task_result_cost_planes({
            **normalize_task_result_cost_planes(existing),
            **projected_fields,
            "task_id": task_id,
            "status": projected_status,
            "ts": explicit_ts or str(existing.get("ts") or now),
            "updated_at": now,
        }))

    stored = prepared_write(_write)
    if transition.get("terminal"):
        # Prior safety flags are in-process retry authority, never a durable grant.
        # Child completion cannot clear its still-running root's history.
        if stored and str(stored.get("root_task_id") or (stored.get("metadata") or {}).get("root_task_id") or task_id) == task_id:
            from ouroboros.safety import clear_safety_work
            clear_safety_work(task_id)
        # The accepted transition into a truly terminal status ends the task: processes its commands left behind are
        # ended by scope (marker + recorded groups) outside this store transaction, on positive evidence only; a pause,
        # retained hold, handoff or reopenable placeholder never reaches here (DEVELOPMENT "Process Custody Rule").
        try:  # a task that launched nothing costs nothing
            from ouroboros.process_scope import note_task_terminal_transition
            note_task_terminal_transition(results_drive_root, task_id)
        except Exception:
            log.debug("task scope cleanup could not be started for %s; the sweep owns it", task_id, exc_info=True)
    from ouroboros.swarm_receipt import refresh_for_task
    refresh_for_task(results_drive_root, stored)
    return stored


def _emit_swarm_fanout(
    ctx: ToolContext,
    *,
    parent_task_id: str,
    root_task_id: str,
    depth: int,
    task_group_id: str,
    task_ids: List[str],
    role: str,
    requested_model_lane: str,
    objective: str,
    emitted_live: bool,
) -> None:
    """Emit one durable swarm_fanout telemetry event per fan-out emission.

    The name avoids task_/llm_/tool_ prefixes and the event sets no
    delegation_role/subagent_task_id, so the Logs UI never renders a phantom
    child card or folds it into a grouped-task lane (web/modules/log_events.js).
    fanout_interval_sec reuses ``_last_fanout_ts`` under the emit lock (no new
    persistent state).
    """
    now = time.time()
    with _SCHEDULE_EMIT_LOCK:
        prev = float(getattr(ctx, "_last_fanout_ts", 0.0) or 0.0)
        fanout_interval = round(now - prev, 3) if prev > 0 else None
        setattr(ctx, "_last_fanout_ts", now)
    evt = {
        "ts": utc_now_iso(),
        "type": "swarm_fanout",
        "task_id": parent_task_id,
        "parent_task_id": parent_task_id,
        "root_task_id": root_task_id,
        "depth": depth,
        "task_group_id": task_group_id,
        "requested_count": len(task_ids),
        "task_ids": task_ids,
        "role": role,
        # The REQUEST. What the children actually ran on is a per-child DISPATCH
        # fact and lives on each child's own record — a fan-out event written before
        # any child started cannot know it, and `effective_model_lanes` used to
        # claim it anyway.
        "requested_model_lane": requested_model_lane,
        "slot_count": len(task_ids),
        "objective_preview": objective[:200],
        "emitted_live": bool(emitted_live),
        "fanout_interval_sec": fanout_interval,
    }
    try:
        append_jsonl(ctx.drive_logs() / "events.jsonl", evt)
        from ouroboros.swarm_rollup import load_swarm_rollup
        load_swarm_rollup(ctx.drive_logs().parent, parent_task_id)
    except Exception:
        log.debug("Failed to emit swarm_fanout telemetry", exc_info=True)
