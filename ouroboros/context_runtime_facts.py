"""The runtime section's FACT builders: what the host can honestly say it knows.

Extracted whole from ``context.py`` at its module ceiling (v7 leaf) so the
facts the runtime section renders keep one home: the project room a task sits in,
the budget rails it runs under, and the
configured delegation route with its honestly-labeled historical observations.
Each returns a plain projection and reads no context state, so nothing here can
change what the section MEANS — only what it reports. ``context`` re-exports every
name, so historical imports and monkeypatch targets keep working unchanged.
"""

from __future__ import annotations

import logging
import pathlib
from typing import Any, Dict, List, Optional

from ouroboros.task_pacing import in_task_cost_ceiling_disclosure as _in_task_cost_ceiling
from ouroboros.config import runtime_setting

log = logging.getLogger(__name__)


def task_execution_clock_fact(task: Dict[str, Any], ctx: Any) -> Dict[str, Any]:
    """Current finite execution-ceiling estimate, not a calendar deadline.

    Quota/budget pauses can move the estimate after this context is assembled;
    an unknown start or unlimited ceiling yields null instead of a false date.
    """
    import datetime
    import math
    import time
    from ouroboros.config import get_task_abs_ceiling_sec
    from ouroboros.deadline_utils import parse_deadline_ts
    from ouroboros.model_wait import current_model_wait, execution_elapsed_seconds

    raw = getattr(ctx, "task_started_at", None) or task.get("started_at")
    try:
        start = float(raw)
    except (TypeError, ValueError):
        parsed = parse_deadline_ts(raw)
        start = parsed.timestamp() if parsed is not None else 0.0
    ceiling = get_task_abs_ceiling_sec()
    started = datetime.datetime.fromtimestamp(start, datetime.timezone.utc).isoformat() if start > 0 and math.isfinite(start) else None
    projected = None
    if started and ceiling is not None:
        now = time.time()
        owner = current_model_wait()
        elapsed = (owner.executed_seconds() if owner is not None and owner.task_id == str(task.get("id") or "")
                   else execution_elapsed_seconds({**task, "started_at": start}, now))
        projected = datetime.datetime.fromtimestamp(now + max(0.0, ceiling - elapsed),
                                                    datetime.timezone.utc).isoformat()
    return {"started_at": started, "absolute_ceiling_at": projected,
            "absolute_ceiling_at_basis": "current estimate; quota or budget pauses may move it" if projected else "not_set"}


def _queue_context_fact(task: Dict[str, Any]) -> Dict[str, Any]:
    """One dated canonical-queue view, frozen with the task's ContextCore."""
    from ouroboros.config import DATA_DIR, get_max_active_subagents_per_root, get_max_workers
    from ouroboros.task_status import _load_queue_snapshot, queue_snapshot_observation

    root = pathlib.Path(task.get("budget_drive_root") or DATA_DIR)
    snapshot = _load_queue_snapshot(root)
    fact = {**queue_snapshot_observation(snapshot), "source_root": str(root),
            "max_workers": int(get_max_workers()),
            "max_active_subagents_per_root": int(get_max_active_subagents_per_root())}
    if snapshot.get("_snapshot_missing") or snapshot.get("_snapshot_invalid"):
        return {**fact, "note": "Queue observation unavailable; current capacity is unknown."}
    for key in ("running", "pending"):
        rows = snapshot.get(key)
        fact[key + "_count"] = sum(isinstance(row, dict) for row in rows) if isinstance(rows, list) else None
    fact["reaping_count"] = snapshot.get("reaping_count")
    fact["worker_total"] = snapshot.get("worker_total")
    assignable = snapshot.get("assignable_idle_workers")
    if assignable is not None:
        fact["free_worker_slots"] = max(0, int(assignable))
        fact["free_worker_slots_basis"] = "recorded_assignable_idle_workers"
    elif fact["running_count"] is not None:
        fact["free_worker_slots"] = max(0, fact["max_workers"] - fact["running_count"]
                                          - int(fact["reaping_count"] or 0))
        fact["free_worker_slots_basis"] = "legacy_estimate_from_configured_limit"
    else:
        fact["free_worker_slots"] = None
        fact["free_worker_slots_basis"] = "unknown"
    fact["note"] = (
        "Last recorded queue counts at ts; freshness and age were measured when this context "
        "was built and do not refresh during the task. Stale or unknown observations do not "
        "establish current load. Legacy free-slot estimates are not measured capacity. "
        "Scheduling owns admission; these observations reserve no slots."
    )
    return fact


def _project_room_fact(task: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The room's active folder and ordinary tool target, or None.

    Conversation keeps its own lifecycle; selecting a room changes the physical
    target without requiring a managed workspace or Git repository. Promotion
    continues to use its separate workspace admission contract.
    """
    try:
        _room_pid = str(task.get("project_id") or "").strip()
        if _room_pid and not str(task.get("workspace_root") or "").strip():
            from ouroboros.config import DATA_DIR as _DATA_DIR
            from ouroboros.projects_registry import get_project as _get_project
            from ouroboros.workspace_admission import room_chat_lens_dir as _room_lens

            _room = _get_project(_DATA_DIR, _room_pid) or {}
            _room_wd = str(_room.get("working_dir") or "").strip()
            if _room_wd:
                # Same resolver the agent uses for the tool lens, so the stated rule
                # and the actual tool surface cannot diverge (the robot incident).
                _lens_dir, _room_note = _room_lens(_DATA_DIR, _room_pid)
                _lens_active = bool(task.get("_is_direct_chat")) and bool(_lens_dir)
                fact = {
                    "project_id": _room_pid,
                    "working_dir": _room_wd,
                    "rule": (
                        (
                            "This room's active_workspace is working_dir for file reads, "
                            "writes and edits, the default shell cwd, VCS and selected "
                            "delegation. The tools retain their own requirements and "
                            "explicit task constraints. Ouroboros governance remains at "
                            "system_repo; select that root explicitly to work on the body. "
                            "Direct work and promotion are both available."
                        )
                        if _lens_active
                        else (
                            "This project has a working folder. Tasks promoted from this room "
                            "run with it as their active workspace by default; pass "
                            "workspace='none' to promote a folder-less task."
                        )
                    ),
                }
                if _room_note:
                    fact["working_dir_warning"] = _room_note
                return fact
    except Exception:
        log.debug("Failed to inject project_room working_dir fact", exc_info=True)
    return None


def _runtime_budget_info(env: Any, task: Dict[str, Any], ctx: Any = None) -> Dict[str, Any]:
    """Start-of-task budget block: global projection + the STATIC per-task tree cap,
    written once at task start so the cached prefix stays byte-stable (DEVELOPMENT
    cache_friendliness item 22); live tree spend rides only the cache-breaking
    surfaces (checkpoint/pacing/milestones)."""
    try:
        from ouroboros.usage_accounting import usage_projection
        from ouroboros.settings_setup_contract import resolve_total_budget_usd

        total_usd = resolve_total_budget_usd()
        budget_root = pathlib.Path(task.get("budget_drive_root") or env.drive_root)
        projection = usage_projection(budget_root, global_limit_usd=total_usd)
        spent_usd = float(projection.get("accounted_usd") or 0.0)
        budget_info = {
            "status": "available" if total_usd is not None else "no_global_limit", "total_usd": total_usd,
            "spent_usd": spent_usd, "remaining_usd": None if total_usd is None else total_usd - spent_usd,
            "reserved_usd": float(projection.get("reserved_usd") or 0.0),
            "unresolved_upper_bound_usd": float(projection.get("unresolved_upper_bound_usd") or 0.0),
            "unknown_unmetered": int(projection.get("unknown_unmetered") or 0),
        }
    except Exception:
        log.error("Budget authority unavailable for runtime context", exc_info=True)
        budget_info = {"status": "unavailable"}
    try:
        root_cap = float(runtime_setting("OUROBOROS_PER_TASK_COST_USD", "0") or 0)
    except (TypeError, ValueError):
        root_cap = 0.0
    if root_cap > 0:
        budget_info["per_task_tree_cap_usd"] = root_cap
        budget_info["per_task_tree_cap_rule"] = (
            "Hard cap for THIS task's WHOLE tree (own model calls + all subagents), enforced "
            "by the physical-attempt ledger: dispatches are refused once the tree's accounted "
            "spend reaches it and the task is force-stopped. Budget checkpoints during the task report the live tree number."
        )
    if ctx is not None:
        budget_info["in_task_cost_ceiling"] = _in_task_cost_ceiling(ctx, budget_info.get("remaining_usd"))
    return budget_info




def _delegation_capability_fact() -> Optional[Dict[str, Any]]:
    """B4-lite: honestly-labeled HISTORICAL delegation observations.

    Deliberately NOT live health — receipts prove what the last execution did,
    not what a lane can do now; live lane facts arrive from plan-review wave
    rows and typed delegate refusals. Pure bounded file reads over the existing
    receipt projections: no daemon probes, no new health authority. Absent
    receipt files mean absent observations, never "healthy". Fail-soft on its
    own (None on any failure) so a problem here never drops the surrounding
    capabilities digest.
    """
    try:
        from ouroboros.reviewer_slot_config import reviewer_slot_last_executions
        from ouroboros.subagent_history import recorded_handle
        from ouroboros.subagents import subagent_last_delegation

        def _observed_label(ts: Any) -> str:
            # Timestamp only: the verbatim "historical, not live health" disclaimer
            # lives ONCE in the note below, never repeated per row.
            return f"last observed at {str(ts or '').strip() or 'unknown time'}"

        delegation: Dict[str, Any] = {
            "note": (
                "Every row here is historical, not live health (the last "
                "recorded execution per reviewer slot / delegated run): "
                "live lane facts arrive from plan-review wave rows and typed "
                "delegate refusals. A missing row means no observation on "
                "record — never healthy."
            ),
        }
        slot_rows: List[Dict[str, Any]] = []
        for slot_id, row in sorted(reviewer_slot_last_executions().items()):
            if not isinstance(row, dict):
                continue
            status = str(row.get("status") or "").strip()
            fact: Dict[str, Any] = {
                "slot": str(slot_id),
                "outcome": (("ok" if status == "ok" else "failed") if status
                            else "unknown"),
                "observed": _observed_label(row.get("ts")),
            }
            requested = row.get("requested") if isinstance(row.get("requested"), dict) else {}
            effective = row.get("effective") if isinstance(row.get("effective"), dict) else {}
            if requested.get("profile_id"):
                fact["requested_profile"] = str(requested["profile_id"])
            if effective.get("profile_id"):
                fact["applied_profile"] = str(effective["profile_id"])
            # B1's typed failure facts, forwarded only when recorded (a dated
            # window carries reset_at without a code and an undated one the
            # code without a reset — read both independently).
            for key in ("failure_code", "reset_at"):
                if row.get(key):
                    fact[key] = row[key]
            slot_rows.append(fact)
        if slot_rows:
            delegation["reviewer_slots_last"] = slot_rows
        last = subagent_last_delegation()
        if isinstance(last, dict) and last:
            last_fact = {
                "route": str(last.get("route") or ""),
                "requested_model": str(last.get("requested_model") or ""),
                "applied_model": str(last.get("applied_model") or ""),
                "observed": _observed_label(last.get("ts")),
            }
            if last.get("requested_profile"):
                last_fact["requested_profile"] = str(last["requested_profile"])
            if last.get("applied_profile"):
                last_fact["applied_profile"] = str(last["applied_profile"])
            # Model-facing actor names are handles computed from each record's
            # OWN facts; the stored key stays in the durable receipt file.
            if last.get("selected_subagent_id"):
                last_fact["selected_subagent_id"] = recorded_handle(last)
            for key in ("outcome", "failure_code", "reset_at", "occurred_at", "observed_at"):
                if key in last:
                    last_fact[key] = last[key]
            delegation["subagent_last_delegation"] = last_fact
            rows = last.get("latest_by_subagent")
            if isinstance(rows, dict) and rows:
                delegation["subagents_last_executions"] = [
                    {**row, "selected_subagent_id": recorded_handle(row)}
                    for row in rows.values() if isinstance(row, dict)
                ]
        if len(delegation) == 1:
            return None
        return delegation
    except Exception:
        log.debug("Failed to build delegation capability fact", exc_info=True)
        return None
