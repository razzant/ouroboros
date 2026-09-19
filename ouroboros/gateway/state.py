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


def _git_checkout_identity(repo_dir: Any) -> tuple:
    """(branch|None, sha) of the ACTUAL runtime git checkout, read-only.

    W2-F3: on a source-mode install nothing ever writes ``current_branch`` /
    ``current_sha`` into state.json (the supervisor stamps them only on managed
    update/reset flows), so ``/api/state`` answered ``sha: ""`` / ``branch:
    null`` while the process demonstrably ran from a real checkout. This reads
    the identity from the checkout itself — pure stdlib file reads (``.git`` may
    be a worktree pointer file; branch refs live in the common dir, loose or
    packed) rather than a ``git`` subprocess, because ``/api/state`` is the UI's
    poll path. A detached HEAD honestly has no branch; any unreadable layout
    degrades to ``(None, "")`` — never an invented identity.
    """
    try:
        gitpath = pathlib.Path(repo_dir) / ".git"
        if gitpath.is_file():
            pointer = gitpath.read_text(encoding="utf-8").strip()
            if not pointer.startswith("gitdir:"):
                return None, ""
            gitdir = (gitpath.parent / pointer.split(":", 1)[1].strip()).resolve()
        elif gitpath.is_dir():
            gitdir = gitpath
        else:
            return None, ""
        head = (gitdir / "HEAD").read_text(encoding="utf-8").strip()
        if not head.startswith("ref: "):
            # Detached HEAD: a bare commit id and honestly no branch.
            return None, head if len(head) >= 7 else ""
        ref = head[5:].strip()
        branch = ref[len("refs/heads/"):] if ref.startswith("refs/heads/") else ref
        commondir = gitdir
        pointer_file = gitdir / "commondir"
        if pointer_file.is_file():
            commondir = (gitdir / pointer_file.read_text(encoding="utf-8").strip()).resolve()
        loose = commondir / ref
        if loose.is_file():
            return branch, loose.read_text(encoding="utf-8").strip()
        packed = commondir / "packed-refs"
        if packed.is_file():
            for line in packed.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.endswith(f" {ref}") and not line.startswith(("#", "^")):
                    return branch, line.split(" ", 1)[0]
        return branch, ""
    except Exception:
        return None, ""


def _runtime_repo_identity(st: Dict[str, Any]) -> tuple:
    """(branch, sha) for ``/api/state``: supervisor-stamped state values win
    (they carry managed update/reset provenance); the live checkout fills the
    source-mode gap; nothing is invented when both are silent."""
    branch = st.get("current_branch") or None
    sha = str(st.get("current_sha") or "")
    if branch and sha:
        return branch, sha
    from ouroboros.config import REPO_DIR

    checkout_branch, checkout_sha = _git_checkout_identity(REPO_DIR)
    return branch or checkout_branch, sha or checkout_sha


def _evolution_state_public(evolution_state: Dict[str, Any]) -> Dict[str, Any]:
    """ABI-3 projection boundary for ``/api/state`` (fix-round-3).

    Campaign history rows are durable supervisor state: a row written by an
    older release still carries the retired ``cost_usd`` spelling. Resolve the
    pair deprecated-wins and emit the honest name only — copy-on-write, since
    the snapshot dict aliases shared supervisor campaign state.
    """
    campaign = evolution_state.get("campaign") if isinstance(evolution_state, dict) else None
    if not isinstance(campaign, dict) or not isinstance(campaign.get("history"), list):
        return evolution_state
    from ouroboros.cost_projection import with_cost_aliases

    history = [
        with_cost_aliases(row) if isinstance(row, dict) else row
        for row in campaign["history"]
    ]
    return {**evolution_state, "campaign": {**campaign, "history": history}}


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


def _describe_bg(request: Request) -> Callable[[bool], dict[str, Any]] | None:
    return _state_attr(request, "describe_bg_consciousness_state")


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
        ensure_legacy_imported(drive_root)
        breakdown = usage_breakdown(drive_root)
        # include_roots=False: /api/state serializes named scalars only, so the
        # per-root map would be built per poll and thrown away (O(N×roots) work
        # with zero readers on this path). The slim projection still carries
        # limit_usd/remaining_known_usd for the evolution budget snapshot below.
        accounting = (
            usage_projection(drive_root, global_limit_usd=limit, include_roots=False)
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
    evolution_state = _evolution_state_public(
        get_evolution_status_snapshot(budget_projection=budget_projection)
        if budget_projection is not None
        else get_evolution_status_snapshot()
    )
    activity_availability = {"complete": True}
    task_bindings = _task_bindings_safe(request, availability=activity_availability)
    direct_turns = _direct_turns_snapshot_safe(availability=activity_availability)
    activities = _chat_activities_snapshot_safe(
        drive_root, task_bindings, direct_turns=direct_turns, availability=activity_availability,
    )
    return {
        "st": st,
        # Resolved here so the checkout file reads stay on the snapshot thread.
        "runtime_identity": _runtime_repo_identity(st),
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
        # The alarm's snapshot reads the usage ledger (a cross-process lock): computed HERE,
        # on the worker thread with the rest of the snapshot, never on the event loop.
        "bg_state": (_describe_bg(request)(bool(st.get("bg_consciousness_enabled"))) if _describe_bg(request) else {}),
        "github_token_configured": bool(github_token_from_env_or_settings()),
        "projects": _projects_summary_safe(request),
        "project_chat_ids": _project_chat_ids_safe(request),
        "task_bindings": task_bindings,
        "active_direct_turns": direct_turns,
        "active_chat_activities": activities,
        "active_chat_activities_complete": activity_availability["complete"],
    }


def _direct_turns_snapshot_safe(*, availability=None) -> list:
    try:
        from supervisor.active_activity import get_direct_activity_registry

        return get_direct_activity_registry().snapshot()
    except Exception:
        if availability is not None:
            availability["complete"] = False
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


# Exact root/task path -> stat-keyed finalizing and question display facts.
_FINALIZING_MEMO: Dict[tuple, tuple] = {}
_FINALIZING_MEMO_MAX = 64


def _task_activity_facts(drive_root: Any, task_id: str) -> dict:
    """One stat-keyed read serves finalizing and the current required question."""
    memo_id = (str(pathlib.Path(drive_root).resolve()), task_id)
    try:
        from ouroboros.task_results import task_results_dir

        path = task_results_dir(pathlib.Path(drive_root), create=False) / f"{task_id}.json"
        stat = path.stat()
    except Exception:
        _FINALIZING_MEMO.pop(memo_id, None)
        return {}
    # Atomic replacement can preserve size and mtime within one timestamp tick.
    key = (str(path), stat.st_dev, stat.st_ino, stat.st_ctime_ns, stat.st_mtime_ns, stat.st_size)
    memo = _FINALIZING_MEMO.get(memo_id)
    if memo is not None and memo[0] == key:
        return memo[1]
    try:
        from ouroboros.utils import read_json_dict

        data = read_json_dict(path)
        if not isinstance(data, dict):
            return {}
    except Exception:
        return {}
    checkpoint = data.get("root_phase_checkpoint")
    synthesis = str(checkpoint.get("post_task_synthesis") or "") if isinstance(checkpoint, dict) else ""
    wait = data.get("owner_wait") if isinstance(data.get("owner_wait"), dict) else {}
    quizzes = data.get("owner_quiz") if isinstance(data.get("owner_quiz"), dict) else {}
    quiz = quizzes.get(str(wait.get("quiz_id") or ""), {})
    facts = {"finalizing": post_task_synthesis_is_open(synthesis),
             "owner_wait": {key: wait[key] for key in ("quiz_id", "state", "resume_reason")
                            if key in wait},
             # The census pointer is the same complete row history and the live delivery carry
             # (project_dialogue.project_question_pointer): display fields ride along.
             "quiz": {key: quiz[key] for key in ("quiz_id", "state", "asked_at", "wait_for_answer", "question",
                                                 "options", "option_details", "stake", "assumption",
                                                 "recommended_index", "answered_index", "comment", "wait_ended_at")
                      if isinstance(quiz, dict) and key in quiz}}
    if len(_FINALIZING_MEMO) >= _FINALIZING_MEMO_MAX:
        _FINALIZING_MEMO.clear()
    _FINALIZING_MEMO[memo_id] = (key, facts)
    return facts


def _managed_task_finalizing(drive_root: Any, task_id: str) -> bool:
    return bool(_task_activity_facts(drive_root, task_id).get("finalizing"))


def _chat_activities_snapshot_safe(drive_root: Any, task_bindings: Any = None, *, direct_turns=None, availability=None) -> list:
    """Direct turns plus ROOT managed queue tasks as ONE activity list.

    Additive beside ``active_direct_turns`` (kept unchanged for compatibility):
    the client hydrates managed-task visibility from the queue authority —
    ``queued`` (PENDING), ``working`` (RUNNING), or ``finalizing`` (RUNNING
    with an open post-task checkpoint) — instead of relying on transient
    typing frames. ``task_bindings`` (the same projection the snapshot already
    serves) re-homes direct and managed activities after a mid-run project
    conversion, while their source records retain the original chat; an
    ``origin_bound`` row is a convert-gate fact only and re-homes nothing. A
    post-task wait keeps the task row's own ``_is_direct_chat`` fact as its
    ``kind``, so a direct turn waiting for a model after its answer is never
    relabelled a managed task. Never raises.
    """
    direct_rows = direct_turns if direct_turns is not None else _direct_turns_snapshot_safe()
    activities = [dict(row) for row in direct_rows]
    bindings = task_bindings if isinstance(task_bindings, dict) else {}
    try:
        from supervisor import queue as queue_mod
        from ouroboros.task_results import resolve_task_lineage

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
                    {**(meta.get("task") or {}), "_attempt": int(meta.get("attempt")
                        or (meta.get("task") or {}).get("_attempt") or 1)} if isinstance(meta, dict) else {},
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
                if availability is not None:
                    availability["complete"] = False
                return False

        def _activity(task_id: str, row: Dict[str, Any], phase: str, started_at: float) -> Dict[str, Any]:
            return {
                "activity_id": task_id,
                "chat_id": int(row.get("chat_id") or 0),
                "project_id": str(row.get("project_id") or ""),
                "client_message_id": "",
                "kind": "direct_chat" if row.get("_is_direct_chat") else "managed_task",
                "phase": phase,
                "started_at": started_at,
                "task_attempt": int(row.get("_attempt") or 1),
                **({"model_waits": row["model_waits"]} if row.get("model_waits") else {}),
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
        from ouroboros.post_task_checkpoint import post_task_model_waits
        visible = {row["activity_id"]: row for row in activities}
        for owner in post_task_model_waits(drive_root):
            waits = owner.snapshot()["model_waits"]
            if owner.task_id in visible:
                visible[owner.task_id].update(phase="finalizing", model_waits=waits, task_attempt=owner.attempt)
            else:
                row = {**owner.task, "model_waits": waits}
                activities.append(_activity(owner.task_id, row, "finalizing", _epoch_or_zero(row.get("queued_at"))))
    except Exception:
        if availability is not None:
            availability["complete"] = False
        log.debug("Managed-activity snapshot unavailable for /api/state", exc_info=True)
    # Bindings are already the gateway's {project_id, chat_id} projection.
    # Apply them to every copied activity before resolving Project questions.
    # A DURABLE binding re-homes a converted card; an origin-bound one is a
    # convert-gate fact only (#902) — that task was never bound, and its chat
    # rows are still where it was started, so moving its live card into the
    # project room would strand the card from its own history.
    for activity in activities:
        binding = bindings.get(str(activity.get("activity_id") or ""))
        if isinstance(binding, dict) and not binding.get("origin_bound"):
            for key in ("chat_id", "project_id"):
                if binding.get(key):
                    activity[key] = binding[key]
    try:
        from ouroboros.project_dialogue import project_question_pointer
        from ouroboros.projects_registry import list_reserved_projects

        projects = {str(row["id"]): row for row in list_reserved_projects(drive_root)}
        for activity in activities:
            facts = _task_activity_facts(drive_root, str(activity.get("activity_id") or ""))
            wait = facts.get("owner_wait", {})
            if not wait.get("quiz_id"):
                continue
            pointer = project_question_pointer(
                {"task_id": activity["activity_id"], "quiz_id": wait["quiz_id"], "wait_for_answer": True},
                facts.get("quiz"), projects.get(str(activity.get("project_id") or "")), wait,
            )
            if pointer:
                activity["required_question"] = pointer
    except Exception:
        # Optional display detail cannot disprove the copied live-id census.
        log.debug("Required-question activity detail unavailable", exc_info=True)
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
        runtime_branch, runtime_sha = snap["runtime_identity"]
        limit = snap["limit"]
        accounting = snap["accounting"]
        breakdown = snap["breakdown"]
        accounting_available = snap["accounting_available"]
        spent = snap["spent"]
        evolution_state = snap["evolution_state"]
        bg_requested = bool(st.get("bg_consciousness_enabled"))
        bg_state = snap.get("bg_state") or {}
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
            # W2-F3: state values when the supervisor stamped them, else the
            # actual runtime checkout (source-mode installs never stamp state).
            # The retired bare "ouroboros" default was dead code — load_state
            # always seeds the key (None), so the default never fired — and a
            # guessed branch is not identity.
            "branch": runtime_branch,
            "sha": (runtime_sha or "")[:8],
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
            "active_chat_activities_complete": snap.get("active_chat_activities_complete") is True,
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


def _task_bindings_safe(request: Request, *, availability=None) -> dict:
    """{task_id: {project_id, chat_id}} for tasks BOUND to a project. The frontend
    uses this to recognise a bound task card: it suppresses the stray "turn into
    project" button (P2) AND turns the card into a pointer that opens the bound
    project's panel (F4).

    Bound is not the same as project-SCOPED: a headless/CLI run carries a
    project_id for lease and memory without ever being bound (that stays the
    owner-facing convert/promote act), so it is absent here BY DESIGN. It needs
    no button gate either — such a run is addressed to its project thread at
    admission, so it never mints a card in Main. Never raises.

    ORIGIN-bound tasks are included too (#902). One owner message spawns several
    task ids, and an ADOPTING conversion (#900) binds only the card that was
    clicked: the message's other live cards stayed task-unbound and went on
    offering "Turn into project" for work that already has one. The project an
    origin already has is resolved by ``project_id_for_origin`` — the one owner of
    that fact, including its legacy several-projects-per-origin tie-break — over
    the SAME live lanes the sibling claim walks; the room id comes from the
    binding that names the chosen project, so no second source can disagree.
    Such a row carries ``origin_bound``: it closes the convert gate and points at
    the project, and deliberately does NOT re-home the task's live card, whose
    chat rows are still in the chat the task was started from."""
    try:
        from ouroboros.projects_registry import all_task_project_bindings

        drive_root = request_drive_root(request)
        bindings = {
            str(k): {"project_id": str(v.get("project_id") or ""), "chat_id": int(v.get("chat_id") or 0)}
            for k, v in (all_task_project_bindings(drive_root, strict=True) or {}).items()
        }
    except Exception:
        if availability is not None:
            availability["complete"] = False
        return {}
    try:
        from ouroboros.projects_registry import live_origin_lanes, project_id_for_origin

        rooms = {row["project_id"]: row["chat_id"] for row in bindings.values()}
        for task_id, origin_ref in live_origin_lanes():
            if task_id in bindings:
                continue
            project_id = str(project_id_for_origin(drive_root, origin_ref) or "")
            room = rooms.get(project_id)
            if room:
                bindings[task_id] = {
                    "project_id": project_id, "chat_id": room, "origin_bound": True,
                }
    except Exception:
        # Fail OPEN on the enrichment only: the durable task-keyed answer above is
        # complete on its own, and the residual is the stray button, not a wrong one.
        log.debug("origin-bound task projection unavailable", exc_info=True)
    return bindings


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
