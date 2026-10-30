"""Core health/state HTTP endpoints for the gateway boundary."""

from __future__ import annotations

import asyncio
import logging
import os
import pathlib
import time
from typing import Any, Callable, Dict, Optional

from starlette.requests import Request
from starlette.responses import JSONResponse

from ouroboros import get_version
from ouroboros.settings_scales import effort_range
from ouroboros.gateway._helpers import json_exception, request_drive_root
from ouroboros.post_task_checkpoint import post_task_synthesis_is_open, post_task_synthesis_is_terminal

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


def _boot_document(request: Request) -> Dict[str, Any]:
    """The ``boot`` object: this process's phases and steps (``server_liveness.BOOT``), the
    supervisor latches the server attached to the app, the store's named-job rows."""
    from ouroboros.server_liveness import boot_document

    return boot_document(request_drive_root(request), getattr(request.app, "state", None))


async def api_health(request: Request) -> JSONResponse:
    """Answers 200 as soon as the socket serves, during boot too; ``boot`` carries the phase."""
    from ouroboros.gateway.schema import GATEWAY_ABI_VERSION

    runtime_version = get_version()
    app_version = os.environ.get("OUROBOROS_APP_VERSION", "").strip() or runtime_version
    boot = await asyncio.to_thread(_boot_document, request)  # one indexed store read, off the loop
    return JSONResponse({
        "status": "ok",
        # legacy field for backward compatibility
        "version": runtime_version,
        "runtime_version": runtime_version,
        "app_version": app_version,
        # The wire-contract carrier a client checks before relying on GET /api/contract.
        "abi_version": GATEWAY_ABI_VERSION,
        "boot": boot,
    })


async def api_contract(_request: Request) -> JSONResponse:
    """``GatewayContractResponse``: the gateway ABI derived from the endpoint and envelope indexes."""
    from ouroboros.gateway.schema import contract_document

    return JSONResponse(contract_document(), headers={"Cache-Control": "no-store"})


def _served_ui_state(request: Request) -> Dict[str, str]:
    """This generation's applied main interface. The lifespan attaches the server's ServedUI;
    before that (or in an app assembled without one) the routes still serve the builtin default."""
    served_ui = _state_attr(request, "served_ui")
    return served_ui.selection.state() if served_ui is not None else {"source": "builtin"}


def _describe_bg(request: Request) -> Callable[[bool], dict[str, Any]] | None:
    return _state_attr(request, "describe_bg_consciousness_state")


async def api_consciousness(request: Request) -> JSONResponse:
    """The Dashboard's read-only status, spend by cause, place labels and recent wakes."""
    describe = _state_attr(request, "describe_consciousness")
    if describe is None:
        return JSONResponse({"status": "unavailable", "recent_wakes": []}, status_code=503,
                            headers={"Cache-Control": "no-store"})
    return JSONResponse(await asyncio.to_thread(describe), headers={"Cache-Control": "no-store"})


def _state_snapshot(request: Request) -> Dict[str, Any]:
    """Collect every heavy synchronous input for the ``/api/state`` payload.

    Runs inside ``asyncio.to_thread`` (pattern shared with gateway/history.py)
    so state.json reads, the usage-ledger projection, the evolution snapshot,
    and the projects/bindings reads cannot block the event loop for every
    concurrent request.
    """
    from ouroboros.tools.github import github_token_from_env_or_settings
    from ouroboros.config import load_settings
    from ouroboros.composer_cognition import main_cognition, main_model_choice
    from ouroboros.route_choice import configured_references
    from ouroboros.subagent_runtime import subagents_availability

    settings = load_settings()
    from ouroboros.usage_accounting import usage_projection, usage_writer_snapshot
    from supervisor.queue import get_evolution_status_snapshot
    from supervisor.state import TOTAL_BUDGET_LIMIT, control_value, load_state
    from supervisor.workers import PENDING, RUNNING, WORKERS

    st = load_state()
    bg_known, bg_value = control_value(st, "bg_consciousness_enabled")
    bg_enabled = bool(bg_value) if bg_known else None  # unknown is never shown as off (#1307)
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
    boot = _boot_document(request)
    accounting_available = True
    try:
        # The writer's slim snapshot (totals, marker, OpenRouter bucket): this response
        # serializes ``physical_calls`` and scalar accounting fields only, so the five
        # grouped axes of ``usage_breakdown`` would be rendered per poll and thrown away.
        # Its private provenance keys stay off the wire even when the unbounded-budget
        # branch reuses the mapping directly as its accounting projection.
        # /api/state is polled: both reads are display reads (a short wait, then the
        # accounting is reported unavailable) instead of parking this worker thread.
        breakdown = {
            key: value
            for key, value in usage_writer_snapshot(drive_root, allow_stale=True).items()
            if not str(key).startswith("_")
        }
        # include_roots=False: /api/state serializes named scalars only, so the
        # per-root map would be built per poll and thrown away (O(N×roots) work
        # with zero readers on this path). The slim projection still carries
        # limit_usd/remaining_known_usd for the evolution budget snapshot below.
        accounting = (
            usage_projection(drive_root, global_limit_usd=limit, include_roots=False, allow_stale=True)
            if limit > 0
            else dict(breakdown)
        )
    except Exception:
        log.exception("Physical-attempt accounting unavailable for /api/state")
        accounting_available = False
        breakdown, accounting = {}, {}
    # Compatibility header/bar shows the dispatch authority's own number: KNOWN
    # spend (settled: confirmed + estimated, #1487). Reservations and unresolved
    # upper bounds stay separately visible in the accounting block.
    spent = float(accounting.get("settled_usd") or 0.0) if accounting_available else None
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
    from ouroboros.projects_registry import bindings_revision

    revision = bindings_revision(drive_root)  # stamped BEFORE the read: the map is never older than it
    task_bindings = _task_bindings_safe(request, availability=activity_availability)
    bindings_complete = activity_availability["complete"]
    direct_turns = _direct_turns_snapshot_safe(availability=activity_availability)
    decision_owners, work_facts = [], {}
    activities = _chat_activities_snapshot_safe(
        drive_root, task_bindings, direct_turns=direct_turns, availability=activity_availability,
        decision_owners=decision_owners, work_facts=work_facts,
    )
    from ouroboros.gateway.main_decisions import main_decisions_snapshot
    decisions = main_decisions_snapshot(drive_root, decision_owners, task_bindings,
                                       activities_complete=activity_availability["complete"],
                                       bindings_complete=bindings_complete)
    from ouroboros.gateway.work_projection import works_snapshot
    works = works_snapshot(drive_root, decision_owners, work_facts, complete=activity_availability["complete"])
    # Registered-project facts (never raise): the compact sidebar list, and the
    # COMPLETE (uncapped, all-status) project chat_ids for the live WS fan-out
    # isolation SSOT — distinct from the capped/filtered sidebar list, so
    # isolation never lapses for projects beyond the summary limit or hidden rows.
    try:
        from ouroboros.projects_registry import projects_summary

        projects = projects_summary(request_drive_root(request), include_pinned=True)
    except Exception:
        projects = []
    try:
        from ouroboros.projects_registry import reserved_project_chat_ids

        project_chat_ids = sorted(reserved_project_chat_ids(request_drive_root(request)))
    except Exception:
        project_chat_ids = []
    from ouroboros.gateway.ui_preferences import main_room_revision

    main_visible_revision = main_room_revision(request_drive_root(request))
    return {
        "st": st,
        # Resolved here so the checkout file reads stay on the snapshot thread.
        "runtime_identity": _runtime_repo_identity(st),
        "workers_alive": alive,
        "boot": boot,
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
        "bg_state": (_describe_bg(request)(bg_enabled) if _describe_bg(request) else {}),
        "github_token_configured": bool(github_token_from_env_or_settings()),
        "subagents_mode": main_cognition(settings)["subagents_mode"],
        "subagents_availability": subagents_availability(settings),
        "model_choice": main_model_choice(settings), "model_routes": configured_references(),
        "projects": projects,
        "project_chat_ids": project_chat_ids,
        "main_visible_revision": main_visible_revision,
        "task_bindings": task_bindings,
        "task_bindings_revision": revision if bindings_complete else "",
        "active_direct_turns": direct_turns,
        "active_chat_activities": activities,
        "active_chat_activities_complete": activity_availability["complete"],
        **decisions,
        **works,
    }


def _direct_turns_snapshot_safe(*, availability=None) -> list:
    try:
        from supervisor.active_activity import get_direct_activity_registry

        registry = get_direct_activity_registry()
        if availability is None:
            return registry.snapshot()
        return registry.snapshot(availability=availability)
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


def _answer_stored(synthesis: Any) -> bool:
    """A post-answer checkpoint state; the pre-answer cost ``refresh`` pseudo-status is not one."""
    return post_task_synthesis_is_open(synthesis) or post_task_synthesis_is_terminal(synthesis)


# Exact root/task path -> stat-keyed outcome, finalizing and question display facts. Sized for a
# census that includes running children (the Main decision owners): a smaller bound evicts the
# census on every poll and re-reads each document twice.
_FINALIZING_MEMO: Dict[tuple, tuple] = {}
_FINALIZING_MEMO_MAX = 1024


def _task_activity_facts(drive_root: Any, task_id: str) -> dict:
    """One signature-keyed read serves outcome, finalizing, retry and question facts.

    ``{}`` is an absent document; a read that could not answer is ``{"unavailable": True}``."""
    from ouroboros.task_store import read_document
    from ouroboros.store_hot_facts import read
    from ouroboros.task_activity_facts import card_facts

    memo_id = (str(pathlib.Path(drive_root).resolve()), task_id)
    try:
        hot = read(drive_root, task_id)
        key = (hot["write_seq"], hot["updated_at"]) if hot else None
    except Exception:
        _FINALIZING_MEMO.pop(memo_id, None)
        return {"unavailable": True}
    if key is None:
        _FINALIZING_MEMO.pop(memo_id, None)
        return {}
    memo = _FINALIZING_MEMO.get(memo_id)
    if memo is not None and memo[0] == key:
        return _with_card(memo[1], hot)
    try:
        data, key = read_document(drive_root, task_id, signed=True)
        if not isinstance(data, dict):
            return {"unavailable": True}
    except Exception:
        return {"unavailable": True}
    checkpoint = data.get("root_phase_checkpoint")
    synthesis = str(checkpoint.get("post_task_synthesis") or "") if isinstance(checkpoint, dict) else ""
    wait = data.get("owner_wait") if isinstance(data.get("owner_wait"), dict) else {}
    quizzes = data.get("owner_quiz") if isinstance(data.get("owner_quiz"), dict) else {}
    quiz = quizzes.get(str(wait.get("quiz_id") or ""), {})
    # Outcome knowledge does not end live activity: a failed answer can still
    # have open post-task work. Carry the same canonical axes as history/detail.
    from ouroboros.outcomes import normalize_outcome_axes

    display = {key: data[key] for key in ("status", "reason_code", "timeout_retry_from", "original_task_id",
                                          "delegated_runs_unreconciled") if key in data}
    if isinstance(data.get("model_waits"), dict):
        display["model_waits"] = data["model_waits"]
    if data.get("status") or data.get("outcome_axes"):
        display["outcome_axes"] = normalize_outcome_axes(data)
    if isinstance(data.get("cancel_origin"), dict):
        display["cancel_origin"] = dict(data["cancel_origin"])  # the stop the status fold reads beside the axes
    if synthesis:
        display["root_phase_checkpoint"] = {"post_task_synthesis": synthesis}
    if _answer_stored(synthesis):
        # An answered root whose late work remains names it where no card does:
        # Activity shows it beside that work's controls.
        from ouroboros.projects_registry import task_presentation_name
        from ouroboros.terminal_time import terminal_time_fact

        # The executor's own terminal fact or nothing: generic ts is never an answer time.
        display["late_work"] = {"title": task_presentation_name(data),
                                "answered_at": terminal_time_fact(data).get("occurred_at") or ""}
    facts = {"finalizing": post_task_synthesis_is_open(synthesis), "late_phase": synthesis, "card": card_facts(data),
             "display": display,
             # An answered root's census row carries only its own routing identity.
             "row": {key: data[key] for key in ("chat_id", "project_id", "_is_direct_chat", "queued_at")
                     if key in data},
             "owner_wait": {key: wait[key] for key in ("quiz_id", "state", "resume_reason", "reason")
                            if key in wait},
             # The census pointer is the same complete row history and the live delivery carry
             # (project_dialogue.project_question_pointer): display fields ride along.
             "quiz": {key: quiz[key] for key in ("quiz_id", "state", "asked_at", "wait_for_answer", "question",
                                                 "options", "option_details", "stake", "assumption",
                                                 "recommended_index", "answered_index", "comment", "wait_ended_at",
                                                 "host_facts", "materials", "closed_reason", "closed_at",
                                                 "closed_by_task_id", "closed_actor", "closed_request_id", "delivery")
                      if isinstance(quiz, dict) and key in quiz}}
    if len(_FINALIZING_MEMO) >= _FINALIZING_MEMO_MAX:
        _FINALIZING_MEMO.clear()
    _FINALIZING_MEMO[memo_id] = (key, facts)
    return _with_card(facts, hot)


def _with_card(facts: dict, hot: dict) -> dict:
    """The card from the hot row this poll read, over a copy of the memoized facts: a row-only card
    write moves no document signature, so the memo alone would show a stale card."""
    from ouroboros.store_task_cards import row_values

    return {**facts, "card": {**facts["card"], **row_values(hot)}}


def _managed_task_finalizing(drive_root: Any, task_id: str) -> bool:
    from ouroboros.store_hot_facts import read

    try:
        return post_task_synthesis_is_open((read(drive_root, task_id) or {}).get("post_task_synthesis"))
    except Exception:
        return False


def _pending_resume_available(q: Any, task: dict, *, owner_paused: bool) -> bool:
    """Present the current release rule; root custody still owns member selection."""
    from ouroboros.work_stop import continuation_refusal, current_stop
    from ouroboros.owner_pause import root_actor
    from supervisor.events_budget import live_root_resume_grant
    from supervisor.queue_transitions import budget_pause_fact

    stop = current_stop(task, {"budget": q.BUDGET_ROOT_FENCES})
    if owner_paused or not (stop.get("release") in {"resume", "owner"} or continuation_refusal(task)):
        return False
    task_id = str(task.get("id") or "")
    root_id = str(task.get("root_task_id") or task_id)
    result_root = pathlib.Path(task.get("budget_drive_root") or q.DRIVE_ROOT)
    if root_actor(result_root, root_id, task_id, strict=True):
        return True
    fence = q.BUDGET_ROOT_FENCES.get(root_id) or {}
    root_parked = fence.get("status") in {"active", "paused"} or any(
        row.get("id") == root_id and budget_pause_fact(row, q.BUDGET_ROOT_FENCES) is not None for row in q.PENDING)
    return not root_parked and not live_root_resume_grant(q, root_id, result_root)


def _finishing_reviews(drive_root: Any, row: Dict[str, Any], task_id: str) -> bool:
    """Whether review work the owner's Pause lets finish still runs, as its census recorded it.
    The census names tasks and model sends alike, so it is never a reviewer count."""
    try:
        from ouroboros.owner_pause import read_fence

        fence = read_fence(pathlib.Path(row.get("budget_drive_root") or drive_root),
                           str(row.get("root_task_id") or task_id))
        return fence.get("state") == "paused" and bool(fence.get("finishing_reviews"))
    except Exception:
        return False


def _raw_work_facts(row: dict) -> dict:
    """The queue facts a census row drops that the work projection reads (gateway/work_projection.py)."""
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    continuation = metadata.get("continuation") if isinstance(metadata.get("continuation"), dict) else {}
    facts = {key: row[key] for key in ("_budget_pause_hold", "_budget_pause", "title", "suggested_name", "focus",
                                        "root_task_id", "parent_task_id", "original_task_id", "budget_drive_root", "owner_wait")
             if row.get(key)}
    edges = {key: continuation[key] for key in ("predecessor_task_id", "work_id", "work_name") if continuation.get(key)}
    return {**facts, **({"continuation": edges} if edges else {})}


def _activity_pause_cause(row: dict, fence: dict) -> str:
    """Explain a parked census row from its existing typed control, never its phase name."""
    from ouroboros.work_stop import stop_presentation

    root_id = str(row.get("root_task_id") or row.get("id") or "")
    return stop_presentation(row, {"budget": {root_id: fence}})["pause_cause"]


def _direct_activity_pause_projection(drive_root: Any, row: dict, availability: Any) -> dict:
    """Project a direct actor's saved warm Pause without changing its registry.

    A requested fence alone cannot say the author reached its boundary. The
    existing owner-wait record supplies that fact; the settled root fence then
    supplies Paused and the separate review count, just as for managed tasks.
    """
    from ouroboros.work_stop import owner_fence_state, parked_stop_view

    activity = dict(row)
    task_id = str(row.get("activity_id") or "")
    state = owner_fence_state(drive_root, row, task_id)
    if state == "unknown":
        activity["phase"] = "unknown"
        if availability is not None:
            availability["complete"] = False
    elif state in {"requested", "paused"}:
        from ouroboros.store_hot_facts import read

        try:
            own = read(row.get("budget_drive_root") or drive_root, task_id) or {}
        except Exception:
            own = {}
        if own.get("owner_wait_state") == "waiting" and own.get("owner_wait_reason") == "owner_pause":
            view = parked_stop_view(drive_root, row, task_id, {})
            activity.update(phase="budget_paused", live=False, stop=view["stop"],
                            refusal=view["refusal"], pause_cause=view["pause_cause"])
            if state == "paused" and (finishing := _finishing_reviews(drive_root, row, task_id)):
                activity["finishing_reviews"] = finishing
    return activity


def _chat_activities_snapshot_safe(drive_root: Any, task_bindings: Any = None, *, direct_turns=None, availability=None,
                                  decision_owners=None, owners_only: bool = False, question_details: bool = True,
                                  work_facts=None) -> list:
    """Copy live owners first, then read their control/phase facts in one store transaction."""
    from ouroboros.post_task_checkpoint import post_task_model_waits
    from ouroboros.store_hot_facts import snapshot
    from supervisor import queue as queue_mod
    from supervisor.owner_pause_control import warm_paused_member

    direct_rows = direct_turns if direct_turns is not None else _direct_turns_snapshot_safe(availability=availability)
    try:
        with queue_mod._queue_lock:
            pending_rows = [dict(task) for task in queue_mod.PENDING]
            fence_rows = {str(key): dict(value) for key, value in queue_mod.BUDGET_ROOT_FENCES.items()
                          if isinstance(value, dict)}
            running_rows = [
                (str(task_id), {**(meta.get("task") or {}), "_attempt": int(meta.get("attempt")
                    or (meta.get("task") or {}).get("_attempt") or 1),
                    "model_activity": (meta.get("active_llm_call") or {}).get("model_activity"),
                    "owner_wait": dict(meta.get("owner_wait") or {}),
                    "_owner_pause_warm": warm_paused_member(meta)} if isinstance(meta, dict) else {},
                 _epoch_or_zero(meta.get("started_at")) if isinstance(meta, dict) else 0.0)
                for task_id, meta in queue_mod.RUNNING.items()]
    except Exception:
        if availability is not None:
            availability["complete"] = False
        pending_rows, fence_rows, running_rows = [], {}, []
    try:
        late_owners = list(post_task_model_waits(drive_root))
    except Exception:
        if availability is not None:
            availability["complete"] = False
        late_owners = []
    owners = [(str(row.get("activity_id") or ""), row) for row in direct_rows]
    owners += [(str(row.get("id") or ""), row) for row in pending_rows]
    owners += [(tid, row) for tid, row, _started in running_rows]
    owners += [(owner.task_id, owner.task) for owner in late_owners]
    owners += [(tid, {}) for tid in fence_rows]
    requests = []
    for tid, row in owners:
        root = row.get("budget_drive_root") or drive_root
        requests.extend(((drive_root, tid), (root, tid), (root, str(row.get("root_task_id") or tid))))
    if owners_only:  # which work is current, without the per-row control facts
        return _chat_activities_snapshot(drive_root, task_bindings, direct_rows, pending_rows, fence_rows,
                                         running_rows, late_owners, availability, decision_owners, True,
                                         question_details, work_facts)
    with snapshot(requests):
        return _chat_activities_snapshot(drive_root, task_bindings, direct_rows, pending_rows, fence_rows,
                                         running_rows, late_owners, availability, decision_owners, False,
                                         question_details, work_facts)


def _chat_activities_snapshot(drive_root, task_bindings, direct_rows, pending_rows, fence_rows,
                              running_rows, late_owners, availability, decision_owners, owners_only,
                              question_details=True, work_facts=None) -> list:
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
    relabelled a managed task. ``owners_only`` keeps the same owners and rooms
    but skips every per-row phase, pause and question projection (each a store
    read): a reader that needs only which work is current (one room's history)
    must not pay for the whole census. ``question_details=False`` skips optional Project/question
    enrichment but retains phases, fences, waits and current result facts. ``work_facts`` collects,
    per activity, the queue facts the rows drop that the work projection reads (``_raw_work_facts``). Never raises.
    """
    activities = [dict(row) if owners_only else _direct_activity_pause_projection(drive_root, row, availability)
                  for row in direct_rows]
    census_ids = {row.get("activity_id") for row in activities}
    bindings = task_bindings if isinstance(task_bindings, dict) else {}
    try:
        from ouroboros.task_results import resolve_task_lineage

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

        def _activity(task_id: str, row: Dict[str, Any], phase: str, started_at: float, *, root: bool = True,
                      can_resume: bool = False, owner_resumed: bool = False, live: bool = False,
                      view: Optional[dict] = None) -> Dict[str, Any]:
            from ouroboros.project_admission import project_hold_fact
            from ouroboros.task_activity_facts import card_facts
            from ouroboros.work_stop import parked_stop_view

            if phase == "unknown" and availability is not None:
                availability["complete"] = False
            view = view or {}
            if phase in {"budget_paused", "budget_pausing"}:
                # A queue row's view comes with its control (``queue_control``); an answered root's is read here.
                view = view or parked_stop_view(drive_root, row, task_id, {} if owner_resumed else fence_rows.get(
                    str(row.get("root_task_id") or task_id), {}), owner_resumed=owner_resumed)
                if phase == "budget_pausing" and not live:
                    phase = "budget_paused"
            return {
                "activity_id": task_id,
                "detached": card_facts(row)["detached"],
                "chat_id": int(row.get("chat_id") or 0),
                "project_id": str(row.get("project_id") or ""),
                "client_message_id": "",
                "kind": "direct_chat" if row.get("_is_direct_chat") else "managed_task",
                "phase": phase,
                # Cancel stops a lineage-resolved root (``ChatOutbound.cancelable``); Resume is the
                # parked row's own release rule (``work_stop.current_stop``).
                "can_resume": bool(can_resume), "cancelable": bool(root),
                **({"pause_cause": view["pause_cause"], "stop": view["stop"], "refusal": view["refusal"],
                    "live": live} if view else {}),
                **({"finishing_reviews": finishing} if phase == "budget_paused" and (
                    finishing := _finishing_reviews(drive_root, row, task_id)) else {}),
                "started_at": started_at,
                "task_attempt": int(row.get("_attempt") or 1),
                **{key: row[key] for key in ("timeout_retry_from", "original_task_id") if row.get(key)},
                # A child is never a census row: its lineage serves only the Main decision projection.
                **({} if root else {key: row[key] for key in ("parent_task_id", "root_task_id", "delegation_role")
                                    if row.get(key)}),
                **({"model_waits": row["model_waits"]} if row.get("model_waits") else {}),
                **({"model_activity": row["model_activity"]} if row.get("model_activity")
                   and row["model_activity"].get("task_attempt") == int(row.get("_attempt") or 1) else {}),
                **({"project_admission_hold": project_hold_fact(row)}
                   if row.get("_project_admission_restore_hold") else {}),
            }

        from types import SimpleNamespace

        from ouroboros.work_stop import owner_fence_state, queue_control

        registries = SimpleNamespace(PENDING=pending_rows, BUDGET_ROOT_FENCES=fence_rows, DRIVE_ROOT=drive_root,
                                     RUNNING={task_id: {"task": row} for task_id, row, _started in running_rows})
        active_roots = set()  # logical roots represented by the current physical queue actors
        for row in pending_rows:
            task_id = str(row.get("id") or "")
            if task_id:
                root = _is_root(task_id, row)
                if root:
                    census_ids.add(task_id)
                    active_roots.add(str(row.get("root_task_id") or task_id))
                if owners_only:
                    activities.append(_activity(task_id, row, "queued", _epoch_or_zero(row.get("queued_at")), root=root))
                    continue
                control = queue_control(drive_root, row, fence_rows)  # the mind's readers read it alike
                phase, interrupted = control["phase"], control["owner_resumed"]
                resume = interrupted or (phase in {"budget_paused", "budget_pausing"} and _pending_resume_available(
                    registries, row, owner_paused=(fence_rows.get(str(row.get("root_task_id") or task_id)) or {}).get(
                        "cause") == "owner_pause" or owner_fence_state(drive_root, row, task_id) == "paused"))
                activities.append(_activity(task_id, row, phase, _epoch_or_zero(row.get("queued_at")), root=root,
                                            can_resume=resume, owner_resumed=interrupted, view=control["view"]))
                if work_facts is not None:
                    work_facts[task_id] = _raw_work_facts(row)
        from ouroboros.post_task_checkpoint import post_task_synthesis_in_flight
        from ouroboros.review_operation import task_has_live_review_operation
        for task_id, row, started_at in running_rows:
            if task_id:
                root = _is_root(task_id, row)
                if root:
                    census_ids.add(task_id)
                    active_roots.add(str(row.get("root_task_id") or task_id))
                if owners_only:
                    activities.append(_activity(task_id, row, "working", started_at, root=root))
                    continue
                # #1196: a RUNNING root whose durable budget_pause row says
                # "pausing" is neither working nor paused yet — additive phase.
                control = queue_control(drive_root, row, fence_rows, running=True)
                phase = control["phase"]
                if phase == "running":
                    phase = "finalizing" if _managed_task_finalizing(drive_root, task_id) else "working"
                activities.append(_activity(task_id, row, phase, started_at, root=root,
                                            live=not row.get("_owner_pause_warm"), view=control["view"]))
                if work_facts is not None:  # whether a settling Pause's member still executes, or waits parked warm
                    warm = {"owner_pause_warm": True} if row.get("_owner_pause_warm") else {}
                    work_facts[task_id] = {**_raw_work_facts(row), "running": not warm, **warm}
        visible = {row["activity_id"]: row for row in activities}
        for root_id, latch in fence_rows.items():
            # An answered root under the owner's Pause whose late work is the tree's
            # remaining work: its answer stays Done while the Pause ends its upkeep
            # and a review's unsent preparation waits for Resume.
            if latch.get("cause") != "owner_pause" or root_id in visible:
                continue
            facts = _task_activity_facts(drive_root, root_id)
            if root_id in active_roots and not (
                    facts.get("finalizing") or post_task_synthesis_in_flight(drive_root, root_id)
                    or task_has_live_review_operation(drive_root, root_id)):
                continue  # the logical fence alone is no second actor; genuine late work stays
            # The fence's own census is the truth of "Paused": reviewers already
            # launched finish separately and never keep the tree Pausing.
            if owner_fence_state(drive_root, {"root_task_id": root_id}, root_id) == "paused":
                phase = "budget_paused"
            elif (facts.get("finalizing") or post_task_synthesis_in_flight(drive_root, root_id)
                  or task_has_live_review_operation(drive_root, root_id, sent_only=True)):
                phase = "budget_pausing"
            elif task_has_live_review_operation(drive_root, root_id):
                phase = "budget_paused"  # an unsent review operation deferred until Resume
            elif any(str(item.get("root_task_id") or "") == root_id for item in pending_rows) or any(
                    str(item.get("root_task_id") or "") == root_id for _tid, item, _at in running_rows):
                phase = "budget_paused"
            else:
                continue
            row = dict(facts.get("row") or {})
            live = phase == "budget_pausing" and post_task_synthesis_in_flight(drive_root, root_id)
            visible[root_id] = _activity(root_id, row, phase, _epoch_or_zero(row.get("queued_at")), live=live)
            if work_facts is not None and live:  # its late work still runs
                work_facts[root_id] = {**_raw_work_facts(row), "running": True}
            census_ids.add(root_id)
            activities.append(visible[root_id])
        for owner in late_owners:
            census_ids.add(owner.task_id)
            if work_facts is not None:
                work_facts.setdefault(owner.task_id, _raw_work_facts(owner.task))
            waits = owner.snapshot()["model_waits"]
            if owner.task_id in visible:
                # A Pause ending the late phase stays "pausing", never back to finalizing.
                pausing = visible[owner.task_id]["phase"] in {"budget_pausing", "budget_paused"}
                visible[owner.task_id].update(model_waits=waits, task_attempt=owner.attempt,
                                              **({} if pausing else {"phase": "finalizing"}))
            else:
                row = {**owner.task, "model_waits": waits}
                activities.append(_activity(owner.task_id, row, "finalizing", _epoch_or_zero(row.get("queued_at")),
                                            root=_is_root(owner.task_id, owner.task)))
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
    owners_by_id = {row["activity_id"]: row for row in activities}
    for activity in activities:
        # A child re-homes through its own, then its parent's or root's durable binding.
        binding = next((bindings[key] for key in (activity.get("activity_id"), activity.get("parent_task_id"),
            activity.get("root_task_id")) if isinstance(bindings.get(key), dict)
            and not bindings[key].get("origin_bound")), None)
        if binding is None and not activity.get("project_id"):
            binding = next((owners_by_id[key] for key in (activity.get("parent_task_id"), activity.get("root_task_id"))
                if key in owners_by_id and owners_by_id[key].get("project_id")), None)
        if isinstance(binding, dict) and not binding.get("origin_bound"):
            for key in ("chat_id", "project_id", "project_name"):
                if binding.get(key):
                    activity[key] = binding[key]
    if owners_only:
        return [row for row in activities if row.get("activity_id") in census_ids]
    try:
        from ouroboros.project_dialogue import owner_wait_projection, project_question_pointer
        from ouroboros.projects_registry import list_reserved_projects

        projects = {str(row["id"]): row for row in list_reserved_projects(drive_root)} if question_details else {}
    except Exception:
        # Without the registry no row's question can be resolved: every row
        # discloses that instead of reading as "no question pending".
        log.debug("Required-question activity detail unavailable", exc_info=True)
        for activity in activities:
            activity["required_question_unavailable"] = True
        projects = {}  # Known wait state survives missing optional Project detail.
    for activity in activities:
        try:
            from ouroboros.task_activity_facts import enrich_activity
            facts = _task_activity_facts(drive_root, str(activity.get("activity_id") or ""))
            if question_details:
                enrich_activity(activity, facts, projects)
            display = dict(facts.get("display") or {})
            recorded_waits = display.pop("model_waits", {})
            attempt = int(activity.get("task_attempt") or 1)
            # Live post-task/direct owners and canonical result facts may land
            # at different times. Merge identities at their current attempt,
            # choosing the newest revision; an old result must not erase a
            # current review wait or restore an already settled one.
            waits = {}
            for source in (recorded_waits, activity.get("model_waits") or {}):
                for wait_id, wait in source.items():
                    if not isinstance(wait, dict) or wait.get("task_attempt") != attempt:
                        continue
                    revision = wait.get("revision")
                    previous = (waits.get(wait_id) or {}).get("revision")
                    # An unreadable live revision cannot authorize the older
                    # result's controls; retain its gap for the Main projector.
                    if (type(revision) is not int or type(previous) is not int
                            or revision >= previous):
                        waits[wait_id] = wait
            activity.update(display)
            if waits:
                activity["model_waits"] = waits
            if not question_details:
                continue
            wait = facts.get("owner_wait", {})
            if not wait.get("quiz_id"):
                continue
            # The waited quiz's own id rides along: Main's work list goes to that exact question
            # (a Main task has no Project pointer to carry it).
            activity["owner_wait"] = {**owner_wait_projection(wait["quiz_id"], wait, facts.get("quiz")),
                                      "quiz_id": str(wait["quiz_id"]),
                                      "quiz_state": (facts.get("quiz") or {}).get("state", "unknown")}
            pointer = project_question_pointer(
                {"task_id": activity["activity_id"], "quiz_id": wait["quiz_id"], "wait_for_answer": True},
                facts.get("quiz"), projects.get(str(activity.get("project_id") or "")), wait,
            )
        except Exception:
            # Optional display detail cannot disprove the copied live-id census,
            # but a row that MAY be blocked on an answer must say it is unknown.
            log.debug("Required-question activity detail unavailable", exc_info=True)
            activity["required_question_unavailable"] = True
            continue
        if pointer:
            activity["required_question"] = pointer
        else:
            # A recorded quiz wait with no readable Project pointer: the wait is
            # real, its detail is not; disclose rather than animate.
            activity["required_question_unavailable"] = True
    if decision_owners is not None:
        decision_owners.extend(activities)  # waits belong to their emitting owner, including children
    return [row for row in activities if row.get("activity_id") in census_ids]


def current_work_in_room(drive_root: Any, task_bindings: Any, matches: Callable[[Any, dict], bool]) -> list:
    """Census roots a recent history page addresses as current work of its room.

    An answered root whose late work alone remains (the answer stays; its upkeep or a
    deferred review is still open) is not current conversation: its card stays beside
    its delivered answer and Activity offers that work's controls, so it is never
    re-addressed at the end of a page that no longer holds its own rows. Only this room's
    owners pay for their facts: the census of every room's work stays with ``/api/state``."""
    in_room = [str(row["activity_id"]) for row in _chat_activities_snapshot_safe(drive_root, task_bindings, owners_only=True)
               if row.get("activity_id") and row.get("chat_id") is not None
               and matches(row["chat_id"], {**row, "task_id": row["activity_id"]})]
    return [task_id for task_id in in_room if not _answer_stored(_task_activity_facts(drive_root, task_id).get("late_phase"))]


async def api_state(request: Request) -> JSONResponse:
    """``StateResponse``, never stored: an open page compares its ``sha`` to spot a new build."""
    try:
        from ouroboros.config import (
            get_context_mode,
            get_runtime_mode,
            get_safety_mode,
            get_skills_repo_path,
        )

        snap = await asyncio.to_thread(_state_snapshot, request)
        st = snap["st"]
        since = str(request.query_params.get("bindings_since") or "")  # the revision of a map the client holds
        unchanged = bool(since) and since == snap["task_bindings_revision"]  # a failed read's "" never is
        runtime_branch, runtime_sha = snap["runtime_identity"]
        limit = snap["limit"]
        accounting = snap["accounting"]
        breakdown = snap["breakdown"]
        accounting_available = snap["accounting_available"]
        spent = snap["spent"]
        evolution_state = snap["evolution_state"]
        from supervisor.state import STATE_READ_KEY, control_value

        # A read that is not current never renders its controls as known (#1307).
        state_read = st.get(STATE_READ_KEY) if isinstance(st.get(STATE_READ_KEY), dict) else {}
        evolution_known, evolution_enabled = control_value(st, "evolution_mode_enabled")
        bg_known, bg_requested = control_value(st, "bg_consciousness_enabled")
        bg_requested = bool(bg_requested) if bg_known else None
        bg_state = snap.get("bg_state") or {}
        supervisor_ready = _state_attr(request, "supervisor_ready_event")
        get_supervisor_error = _state_attr(request, "get_supervisor_error")
        app_start = float(_state_attr(request, "app_start", time.time()) or time.time())
        response = JSONResponse({
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
            "evolution_enabled": bool(evolution_enabled) if evolution_known else None,
            "bg_consciousness_enabled": bg_requested,
            "state_quality": {"quality": str(state_read.get("quality") or "current"),
                              "source": str(state_read.get("source") or "primary"),
                              "unconfirmed": list(state_read.get("unconfirmed") or [])},
            "evolution_cycle": int(st.get("evolution_cycle") or 0),
            "evolution_state": evolution_state,
            "bg_consciousness_state": bg_state,
            "spent_calls": (
                int(breakdown.get("physical_calls") or 0) if accounting_available else None
            ),
            "supervisor_ready": bool(supervisor_ready.is_set()) if supervisor_ready else False,
            "supervisor_error": get_supervisor_error() if callable(get_supervisor_error) else None,
            "boot": snap["boot"],  # the boot phases the header paints (Starting… → chat ready → Online)
            "runtime_mode": get_runtime_mode(),
            "context_mode": get_context_mode(),
            "subagents_mode": snap["subagents_mode"],
            "subagents_availability": snap["subagents_availability"],
            # Frozen one-window compatibility field. Persistent auto-Low is retired.
            "context_mode_auto_low": False,
            "effort_range": effort_range(),  # the owner's range, the tolerant read
            "model_choice": snap["model_choice"], "model_routes": snap["model_routes"],  # Main chat's model picker
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
            "main_visible_revision": snap.get("main_visible_revision"),
            "task_bindings": None if unchanged else snap["task_bindings"],
            "task_bindings_revision": snap["task_bindings_revision"],
            "task_bindings_unchanged": unchanged,
            "active_direct_turns": snap.get("active_direct_turns") or [],
            "active_chat_activities": snap.get("active_chat_activities") or [],
            "active_chat_activities_complete": snap.get("active_chat_activities_complete") is True,
            "current_questions": snap.get("current_questions") or [],
            "current_questions_complete": snap.get("current_questions_complete") is True,
            "current_questions_gaps": snap.get("current_questions_gaps") or [],
            "main_attention": snap.get("main_attention") or [],
            "main_attention_complete": snap.get("main_attention_complete") is True,
            "main_attention_gaps": snap.get("main_attention_gaps") or [],
            "works": snap.get("works") or [],
            "rooms": snap.get("rooms") or {},
            "work_summary": snap.get("work_summary") or {},
            "works_complete": snap.get("works_complete") is True,
            "works_gaps": snap.get("works_gaps") or [],
            "served_ui": _served_ui_state(request),
        })
    except Exception as exc:
        response = json_exception(exc)
    response.headers["Cache-Control"] = "no-store"
    return response


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

    """
    try:
        from ouroboros.projects_registry import all_task_project_bindings

        return all_task_project_bindings(request_drive_root(request), strict=True)
    except Exception:
        if availability is not None:
            availability["complete"] = False
        return {}


__all__ = ["api_contract", "api_health", "api_state"]
