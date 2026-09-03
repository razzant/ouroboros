"""Core health/state HTTP endpoints for the gateway boundary."""

from __future__ import annotations

import asyncio
import logging
import os
import pathlib
import time
from typing import Any, Callable, Dict

from starlette.requests import Request
from starlette.responses import JSONResponse

from ouroboros import get_version
from ouroboros.gateway._helpers import json_exception, request_drive_root
from ouroboros.post_task_checkpoint import post_task_synthesis_is_open

log = logging.getLogger(__name__)


def _state_attr(request: Request, name: str, default: Any = None) -> Any:
    state = getattr(request.app, "state", None)
    return getattr(state, name, default) if state is not None else default


async def api_health(_request: Request) -> JSONResponse:
    runtime_version = get_version()
    app_version = os.environ.get("OUROBOROS_APP_VERSION", "").strip() or runtime_version
    return JSONResponse({
        "status": "ok",
        # legacy field for backward compatibility
        "version": runtime_version,
        "runtime_version": runtime_version,
        "app_version": app_version,
    })


def _state_snapshot(request: Request) -> Dict[str, Any]:
    """Collect every heavy synchronous input for the ``/api/state`` payload.

    Runs inside ``asyncio.to_thread`` (pattern shared with gateway/history.py)
    so state.json reads, the usage-ledger projection, the evolution snapshot,
    and the projects/bindings reads cannot block the event loop for every
    concurrent request.
    """
    from ouroboros.tools.github import github_token_from_env_or_settings
    from ouroboros.usage_accounting import ensure_legacy_imported, usage_breakdown, usage_projection
    from supervisor.queue import get_evolution_status_snapshot
    from supervisor.state import TOTAL_BUDGET_LIMIT, load_state
    from supervisor.workers import PENDING, RUNNING, WORKERS

    st = load_state()
    alive = 0
    total_w = 0
    try:
        alive = sum(1 for w in WORKERS.values() if w.proc.is_alive())
        total_w = len(WORKERS)
    except Exception:
        pass
    # ``0`` is the documented unbounded budget, not a request to invent the
    # historical $10 default.  Server startup initializes the supervisor
    # value from settings; keeping zero here makes that state explicit.
    limit = max(0.0, float(TOTAL_BUDGET_LIMIT or 0.0))
    drive_root = request_drive_root(request)
    accounting_available = True
    try:
        from ouroboros.usage_ledger import DISPLAY_LOCK_TIMEOUT_SEC

        ensure_legacy_imported(drive_root)
        # /api/state is polled by benchmark watchers during 64-lane runs: the
        # ledger reads degrade to the last validated snapshot under write
        # contention instead of parking this to_thread for the monetary timeout.
        breakdown = usage_breakdown(
            drive_root,
            lock_timeout_sec=DISPLAY_LOCK_TIMEOUT_SEC,
            allow_stale=True,
        )
        # include_roots=False: /api/state serializes named scalars only, so the
        # per-root map would be built per poll and thrown away (O(N×roots) work
        # with zero readers on this path). The slim projection still carries
        # limit_usd/remaining_known_usd for the evolution budget snapshot below.
        accounting = (
            usage_projection(
                drive_root,
                global_limit_usd=limit,
                include_roots=False,
                lock_timeout_sec=DISPLAY_LOCK_TIMEOUT_SEC,
                allow_stale=True,
            )
            if limit > 0
            else dict(breakdown)
        )
    except Exception:
        log.exception("Physical-attempt accounting unavailable for /api/state")
        accounting_available = False
        breakdown, accounting = {}, {}
    # Compatibility header/bar uses the conservative dispatch authority:
    # settled + live reservations + unresolved upper bounds.  Actual paid
    # cost remains separately visible as accounting.settled_usd/confirmed.
    spent = float(accounting.get("accounted_usd") or 0.0) if accounting_available else None
    # De-triplication: hand the evolution snapshot the projection this request
    # already computed, so budget_remaining does not replay the ledger again —
    # but ONLY when this request's drive root IS the supervisor's root and the
    # computation SUCCEEDED. A failed computation passes nothing, so the
    # snapshot computes (and fails) itself and the "accounting unavailable =>
    # evolution paused" disclosure keeps coming from its own strict attempt.
    budget_projection = None
    if accounting_available and limit > 0:
        try:
            from supervisor import state as supervisor_state

            if (
                pathlib.Path(supervisor_state.DRIVE_ROOT).resolve(strict=False)
                == pathlib.Path(drive_root).resolve(strict=False)
            ):
                budget_projection = accounting
        except Exception:
            budget_projection = None
    evolution_state = (
        get_evolution_status_snapshot(budget_projection=budget_projection)
        if budget_projection is not None
        else get_evolution_status_snapshot()
    )
    task_bindings = _task_bindings_safe(request)
    return {
        "st": st,
        "workers_alive": alive,
        "workers_total": total_w,
        "pending_count": len(PENDING),
        "running_count": len(RUNNING),
        "limit": limit,
        "accounting_available": accounting_available,
        "accounting": accounting,
        "breakdown": breakdown,
        "spent": spent,
        "evolution_state": evolution_state,
        "github_token_configured": bool(github_token_from_env_or_settings()),
        "projects": _projects_summary_safe(request),
        "project_chat_ids": _project_chat_ids_safe(request),
        "task_bindings": task_bindings,
        "active_direct_turns": (
            _direct_turns_snapshot_safe()
        ),
        "active_chat_activities": _chat_activities_snapshot_safe(drive_root, task_bindings),
    }


def _direct_turns_snapshot_safe() -> list:
    try:
        from supervisor.active_activity import get_direct_activity_registry

        return get_direct_activity_registry().snapshot()
    except Exception:
        return []


def _epoch_or_zero(value: Any) -> float:
    """Epoch seconds from a float or an ISO-8601 string (queued_at); else 0.0."""
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        pass
    try:
        from datetime import datetime

        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError):
        return 0.0


# task_id -> ((mtime_ns, size), finalizing) so the poll re-reads a root's
# durable result only when the file actually changed (projection over replay).
_FINALIZING_MEMO: Dict[str, tuple] = {}
_FINALIZING_MEMO_MAX = 64


def _managed_task_finalizing(drive_root: Any, task_id: str) -> bool:
    """True while the root's post-task synthesis checkpoint is OPEN.

    An open checkpoint (``pending_once`` | ``running``) is the canonical
    durable signal that the final answer was stored but post-task synthesis —
    and therefore ``task_done`` — is still pending (post_task_checkpoint.py).
    Stat-memoized; never raises.
    """
    try:
        from ouroboros.task_results import task_results_dir

        path = task_results_dir(pathlib.Path(drive_root), create=False) / f"{task_id}.json"
        stat = path.stat()
    except Exception:
        _FINALIZING_MEMO.pop(task_id, None)
        return False
    key = (stat.st_mtime_ns, stat.st_size)
    memo = _FINALIZING_MEMO.get(task_id)
    if memo is not None and memo[0] == key:
        return memo[1]
    try:
        from ouroboros.utils import read_json_dict

        data = read_json_dict(path) or {}
    except Exception:
        return False
    checkpoint = data.get("root_phase_checkpoint")
    synthesis = str(checkpoint.get("post_task_synthesis") or "") if isinstance(checkpoint, dict) else ""
    finalizing = post_task_synthesis_is_open(synthesis)
    if len(_FINALIZING_MEMO) >= _FINALIZING_MEMO_MAX:
        _FINALIZING_MEMO.clear()
    _FINALIZING_MEMO[task_id] = (key, finalizing)
    return finalizing


def _chat_activities_snapshot_safe(drive_root: Any, task_bindings: Any = None) -> list:
    """Direct/ephemeral turns plus ROOT managed queue tasks as ONE activity list.

    Additive beside ``active_direct_turns`` (kept unchanged for compatibility):
    the client hydrates managed-task visibility from the queue authority —
    ``queued`` (PENDING), ``working`` (RUNNING), or ``finalizing`` (RUNNING
    with an open post-task checkpoint) — instead of relying on transient
    typing frames. ``task_bindings`` (the same projection the snapshot already
    serves) re-homes a mid-run "turn into project" conversion, whose queue row
    still carries the original chat. Never raises.
    """
    activities = _direct_turns_snapshot_safe()
    try:
        from supervisor import queue as queue_mod
        from ouroboros.task_results import resolve_task_lineage

        bindings = task_bindings if isinstance(task_bindings, dict) else {}
        with queue_mod._queue_lock:
            pending_rows = [dict(task) for task in queue_mod.PENDING]
            fence_rows = {
                str(key): dict(value)
                for key, value in queue_mod.BUDGET_ROOT_FENCES.items()
                if isinstance(value, dict)
            }
            running_rows = [
                (
                    str(task_id),
                    dict(meta.get("task") or {}) if isinstance(meta, dict) else {},
                    _epoch_or_zero(meta.get("started_at")) if isinstance(meta, dict) else 0.0,
                )
                for task_id, meta in queue_mod.RUNNING.items()
            ]

        def _is_root(task_id: str, row: Dict[str, Any]) -> bool:
            try:
                return bool(resolve_task_lineage(
                    task_id,
                    metadata=row.get("metadata"),
                    root_task_id=row.get("root_task_id"),
                    parent_task_id=row.get("parent_task_id"),
                    delegation_role=row.get("delegation_role"),
                    original_task_id=row.get("original_task_id"),
                    timeout_retry_from=row.get("timeout_retry_from"),
                )["is_root_task"])
            except Exception:
                return False

        def _activity(task_id: str, row: Dict[str, Any], phase: str, started_at: float) -> Dict[str, Any]:
            binding = bindings.get(task_id) if isinstance(bindings.get(task_id), dict) else {}
            return {
                "activity_id": task_id,
                "chat_id": int(binding.get("chat_id") or row.get("chat_id") or 0),
                "project_id": str(binding.get("project_id") or row.get("project_id") or ""),
                "client_message_id": "",
                "kind": "managed_task",
                "phase": phase,
                "started_at": started_at,
            }

        from supervisor.queue_transitions import budget_pause_fact

        for row in pending_rows:
            task_id = str(row.get("id") or "")
            if task_id and _is_root(task_id, row):
                # #322 (P1): a budget-paused member must not masquerade as
                # "queued" — nothing will dispatch it until an explicit resume.
                phase = "budget_paused" if budget_pause_fact(row, fence_rows) else "queued"
                activities.append(_activity(task_id, row, phase, _epoch_or_zero(row.get("queued_at"))))
        for task_id, row, started_at in running_rows:
            if task_id and _is_root(task_id, row):
                phase = "finalizing" if _managed_task_finalizing(drive_root, task_id) else "working"
                activities.append(_activity(task_id, row, phase, started_at))
    except Exception:
        log.debug("Managed-activity snapshot unavailable for /api/state", exc_info=True)
    return activities


async def api_state(request: Request) -> JSONResponse:
    try:
        from ouroboros.config import (
            get_context_mode,
            get_runtime_mode,
            get_safety_mode,
            get_skills_repo_path,
        )

        snap = await asyncio.to_thread(_state_snapshot, request)
        st = snap["st"]
        limit = snap["limit"]
        accounting = snap["accounting"]
        breakdown = snap["breakdown"]
        accounting_available = snap["accounting_available"]
        spent = snap["spent"]
        evolution_state = snap["evolution_state"]
        bg_requested = bool(st.get("bg_consciousness_enabled"))
        describe_bg_state: Callable[[bool], dict[str, Any]] | None = _state_attr(
            request,
            "describe_bg_consciousness_state",
        )
        bg_state = describe_bg_state(bg_requested) if describe_bg_state else {}
        supervisor_ready = _state_attr(request, "supervisor_ready_event")
        get_supervisor_error = _state_attr(request, "get_supervisor_error")
        app_start = float(_state_attr(request, "app_start", time.time()) or time.time())
        return JSONResponse({
            "uptime": int(time.time() - app_start),
            "workers_alive": snap["workers_alive"],
            "workers_total": snap["workers_total"],
            "pending_count": snap["pending_count"],
            "running_count": snap["running_count"],
            "spent_usd": round(spent, 4) if spent is not None else None,
            "budget_limit": limit,
            "budget_pct": (
                round((spent / limit * 100) if limit > 0 else 0, 1)
                if spent is not None else None
            ),
            "branch": st.get("current_branch", "ouroboros"),
            "sha": (st.get("current_sha") or "")[:8],
            "evolution_enabled": bool(st.get("evolution_mode_enabled")),
            "bg_consciousness_enabled": bg_requested,
            "evolution_cycle": int(st.get("evolution_cycle") or 0),
            "evolution_state": evolution_state,
            "bg_consciousness_state": bg_state,
            "spent_calls": (
                int(breakdown.get("physical_calls") or 0) if accounting_available else None
            ),
            "supervisor_ready": bool(supervisor_ready.is_set()) if supervisor_ready else False,
            "supervisor_error": get_supervisor_error() if callable(get_supervisor_error) else None,
            "runtime_mode": get_runtime_mode(),
            "context_mode": get_context_mode(),
            # Frozen one-window compatibility field. Persistent auto-Low is retired.
            "context_mode_auto_low": False,
            "safety_mode": get_safety_mode(),
            "skills_repo_configured": bool(get_skills_repo_path()),
            "github_token_configured": snap["github_token_configured"],
            "accounting": {
                "available": accounting_available,
                "authority": "physical_attempt_ledger",
                "settled_usd": (
                    float(accounting.get("settled_usd") or 0.0) if accounting_available else None
                ),
                "confirmed_usd": (
                    float(accounting.get("confirmed_usd") or 0.0) if accounting_available else None
                ),
                "estimated_usd": (
                    float(accounting.get("estimated_usd") or 0.0) if accounting_available else None
                ),
                "reserved_usd": (
                    float(accounting.get("reserved_usd") or 0.0) if accounting_available else None
                ),
                "unresolved_upper_bound_usd": (
                    float(accounting.get("unresolved_upper_bound_usd") or 0.0)
                    if accounting_available else None
                ),
                "accounted_usd": (
                    float(accounting.get("accounted_usd") or 0.0) if accounting_available else None
                ),
                "unknown_unmetered": (
                    int(accounting.get("unknown_unmetered") or 0) if accounting_available else None
                ),
                "cost_final": bool(accounting.get("cost_final")) if accounting_available else False,
                "integrity_degraded": (
                    bool(accounting.get("integrity_degraded")) if accounting_available else True
                ),
                "attempt_counts": dict(accounting.get("attempt_counts") or {}),
                "limit_usd": limit,
                "remaining_known_usd": (
                    float(accounting.get("remaining_known_usd") or 0.0)
                    if accounting_available and limit > 0
                    else None
                ),
                **({"error_code": "ledger_unavailable"} if not accounting_available else {}),
            },
            "projects": snap["projects"],
            "project_chat_ids": snap["project_chat_ids"],
            "task_bindings": snap["task_bindings"],
            "active_direct_turns": snap.get("active_direct_turns") or [],
            "active_chat_activities": snap.get("active_chat_activities") or [],
        })
    except Exception as exc:
        return json_exception(exc)


def _projects_summary_safe(request: Request) -> list:
    """Compact registered-projects list for the sidebar (never raises)."""
    try:
        from ouroboros.projects_registry import projects_summary

        return projects_summary(request_drive_root(request))
    except Exception:
        return []


def _task_bindings_safe(request: Request) -> dict:
    """{task_id: {project_id, chat_id}} for tasks bound to a project. The frontend
    uses this to recognise a project-scoped task card: it suppresses the stray
    "turn into project" button (P2) AND turns the card into a pointer that opens
    the bound project's panel (F4). Never raises."""
    try:
        from ouroboros.projects_registry import all_task_project_bindings

        return {
            str(k): {"project_id": str(v.get("project_id") or ""), "chat_id": int(v.get("chat_id") or 0)}
            for k, v in (all_task_project_bindings(request_drive_root(request)) or {}).items()
        }
    except Exception:
        return {}


def _project_chat_ids_safe(request: Request) -> list:
    """COMPLETE (uncapped, all-status) registered project chat_ids for the live
    WS fan-out isolation SSOT — distinct from the capped/filtered sidebar list,
    so isolation never lapses for projects beyond the summary limit or hidden
    rows. Never raises."""
    try:
        from ouroboros.projects_registry import reserved_project_chat_ids

        return sorted(reserved_project_chat_ids(request_drive_root(request)))
    except Exception:
        return []


__all__ = ["api_health", "api_state"]
