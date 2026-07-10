"""Core health/state HTTP endpoints for the gateway boundary."""

from __future__ import annotations

import os
import time
from typing import Any, Callable

from starlette.requests import Request
from starlette.responses import JSONResponse

from ouroboros import get_version
from ouroboros.gateway._helpers import json_exception, request_drive_root


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


async def api_state(request: Request) -> JSONResponse:
    try:
        from supervisor.state import TOTAL_BUDGET_LIMIT, load_state
        from supervisor.workers import PENDING, RUNNING, WORKERS
        from supervisor.queue import get_evolution_status_snapshot
        from ouroboros.config import get_context_mode, get_runtime_mode, get_safety_mode, get_skills_repo_path
        from ouroboros.tools.github import github_token_from_env_or_settings

        st = load_state()
        alive = 0
        total_w = 0
        try:
            alive = sum(1 for w in WORKERS.values() if w.proc.is_alive())
            total_w = len(WORKERS)
        except Exception:
            pass
        spent = float(st.get("spent_usd") or 0.0)
        limit = float(TOTAL_BUDGET_LIMIT or 10.0)
        evolution_state = get_evolution_status_snapshot()
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
            "workers_alive": alive,
            "workers_total": total_w,
            "pending_count": len(PENDING),
            "running_count": len(RUNNING),
            "spent_usd": round(spent, 4),
            "budget_limit": limit,
            "budget_pct": round((spent / limit * 100) if limit > 0 else 0, 1),
            "branch": st.get("current_branch", "ouroboros"),
            "sha": (st.get("current_sha") or "")[:8],
            "evolution_enabled": bool(st.get("evolution_mode_enabled")),
            "bg_consciousness_enabled": bg_requested,
            "evolution_cycle": int(st.get("evolution_cycle") or 0),
            "evolution_state": evolution_state,
            "bg_consciousness_state": bg_state,
            "spent_calls": int(st.get("spent_calls") or 0),
            "supervisor_ready": bool(supervisor_ready.is_set()) if supervisor_ready else False,
            "supervisor_error": get_supervisor_error() if callable(get_supervisor_error) else None,
            "runtime_mode": get_runtime_mode(),
            "context_mode": get_context_mode(),
            "safety_mode": get_safety_mode(),
            "skills_repo_configured": bool(get_skills_repo_path()),
            "github_token_configured": bool(github_token_from_env_or_settings()),
            "projects": _projects_summary_safe(request),
            "project_chat_ids": _project_chat_ids_safe(request),
            "task_bindings": _task_bindings_safe(request),
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
        from ouroboros.projects_registry import registered_project_chat_ids

        return sorted(registered_project_chat_ids(request_drive_root(request)))
    except Exception:
        return []


__all__ = ["api_health", "api_state"]
