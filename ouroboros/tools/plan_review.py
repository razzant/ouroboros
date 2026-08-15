"""Pre-implementation Atlas-backed design review tool."""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import json
import logging
import pathlib
import time
from dataclasses import dataclass, replace
from datetime import timedelta

from ouroboros.config import SETTINGS_DEFAULTS
from ouroboros.deadline_utils import parse_deadline_ts, utc_now as _planning_now
from ouroboros.tools.plan_review_runtime import (  # noqa: F401 -- re-exported for callers and tests
    resolve_plan_class as _resolve_plan_class,
)
from ouroboros.tools.review_synthesis import (  # noqa: F401 -- re-exported for callers and tests
    resolve_plan_context_level as _resolve_plan_context_level,
)
from ouroboros.tools.plan_review_setup import (
    _open_plan_subject_roots,
    _resolve_plan_shape,
)
from ouroboros.planning_evidence import planning_evidence_horizon
from ouroboros.task_results import (
    load_plan_review_state,
    mark_current_plan_review_unavailable,
    plan_review_gate_projection,
    plan_review_wave,
    plan_review_wave_handoffs,
    plan_review_wave_task_ids,
    plan_review_handoff_snapshot_path,
    persist_plan_review_handoff_snapshot,
    persist_plan_review_handoffs,
    record_plan_review_collection,
    record_plan_review_consumed,
    record_plan_review_attempt,
    record_plan_review_result,
    record_plan_review_scout,
    represent_plan_review,
    reserve_plan_review_wave,
)
from ouroboros.task_status import FINAL_STATUSES, find_child_tasks, wait_for_effective_tasks
from ouroboros.tools.plan_review_runtime import (
    PLAN_REVIEW_MAX_TOKENS as _PLAN_REVIEW_MAX_TOKENS,
    PLAN_REVIEW_SLOT_TIMEOUT_SEC as _PLAN_REVIEW_SLOT_TIMEOUT_SEC,
    classify_reviewer_error as _classify_reviewer_error,  # noqa: F401 — test-compat re-export
    get_review_models as _get_review_models,
    load_plan_checklist as _load_plan_checklist,
    plan_deadline_skip as _plan_deadline_skip,
    record_raw_plan_request_attempt as _record_raw_plan_request_attempt,
    reviewed_handoff_hashes as _reviewed_handoff_hashes,
    run_plan_review_slots as _run_plan_review_slots,
    validate_plan_request_envelope as _validate_plan_request_envelope,
)
from ouroboros.tools.registry import ToolContext, ToolEntry
from ouroboros.tools.review_context_atlas import (
    ReviewContextAtlasRequest,
    atlas_assembly_failed,
    atlas_assembly_failure_reason,
    compile_review_context_atlas,
)
from ouroboros.tools.review_helpers import (
    build_head_snapshot_section,
    load_governance_doc,
    review_wave_budget_gate,
)
from ouroboros.tools.review_synthesis import (
    emit_plan_review_usage as _emit_plan_review_usage,  # noqa: F401 — test-compat re-export
    PLAN_REVIEW_CONTROL_PREFIX,
    VACUOUS_CLAIMS_NOTE as _VACUOUS_CLAIMS_NOTE,
    VACUOUS_DISPOSITION_NOTE as _VACUOUS_DISPOSITION_NOTE,
    addressable_plan_findings,
    vacuous_acceptance_claims as _vacuous_acceptance_claims,
    vacuous_review_disposition as _vacuous_review_disposition,
    all_planning_tasks_terminal as _all_planning_tasks_known_terminal,
    assemble_plan_raw_results as _assemble_plan_raw_results,
    bounded_planning_reason as _bounded_planning_reason,
    build_plan_review_system_prompt,
    build_plan_review_user_content,
    completed_planning_handoffs as _completed_planning_handoffs,
    format_planning_handoffs as _format_planning_handoffs,
    format_plan_review_output as _format_output,
    parse_plan_review_signal,
    minted_plan_slot_ids as _minted_plan_slot_ids,  # noqa: F401 — test-compat re-export
    per_slot_input_token_limits as _per_slot_input_token_limits,
    plan_review_fingerprint as _plan_request_fingerprint,
    plan_context_target_tokens as _plan_context_target_tokens,
    plan_slot_fit_with_identity as _plan_slot_fit_with_identity,
    plan_text_fingerprint,
    quorum_input_token_limit as _quorum_input_token_limit,
    planning_handoff_selection as _planning_handoff_selection,
    planning_scout_framing as _planning_scout_framing,
    planning_scout_wave_plan as _planning_scout_wave_plan,
    planning_swarm_context as _planning_swarm_context,
    render_plan_context_degradation as _render_plan_context_degradation,
    render_plan_review_result as _render_existing_plan_review,
    replay_closed_review_disposition as _replay_closed_review_disposition,
    summarize_plan_review_results as _summarize_plan_review_results,
    validate_plan_review_disposition,
)
from ouroboros.utils import estimate_tokens, read_json_dict, utc_now_iso

_addressable_plan_findings = addressable_plan_findings
_parse_aggregate_signal = parse_plan_review_signal
_build_system_prompt = build_plan_review_system_prompt
_build_user_content = build_plan_review_user_content

log = logging.getLogger(__name__)

# Scout-wave admission prices ONE opening round per scout (a deliberate lower bound: a wave
# that cannot fund even that must not start; the per-attempt reservation rail covers the rest).
_PLAN_SCOUT_MAX_TOKENS = 8192
# Wrapper covers the shared scout cutoff plus one reviewer slot below the hard timeout.
_PLAN_SWARM_MAX_WAIT_DEFAULT_SEC = int(SETTINGS_DEFAULTS["OUROBOROS_PLAN_TASK_SWARM_MAX_WAIT_SEC"])  # config SSOT (no DRY mirror)
_PLAN_REVIEW_WRAPPER_TIMEOUT_SEC = _PLAN_SWARM_MAX_WAIT_DEFAULT_SEC + _PLAN_REVIEW_SLOT_TIMEOUT_SEC + 60
_PLAN_TASK_TOOL_TIMEOUT_SEC = _PLAN_REVIEW_WRAPPER_TIMEOUT_SEC + 10


def _effective_swarm_max_wait() -> float:
    from ouroboros.config import get_plan_task_swarm_max_wait_sec
    return min(get_plan_task_swarm_max_wait_sec(), float(_PLAN_SWARM_MAX_WAIT_DEFAULT_SEC))

@dataclass(frozen=True)
class _PlanReviewRequest:
    plan: str
    goal: str
    files_to_touch: list
    context_level: str = ""
    context_notes: str = ""
    include_tests: bool = False
    plan_class: str = ""
    scope: dict | None = None
    review_disposition: dict | None = None


@dataclass
class _PlanReviewFinalization:
    request: _PlanReviewRequest
    raw_results: list[dict]
    models: list[str]
    estimated_tokens: int
    subject_repo: pathlib.Path
    governance_repo: pathlib.Path
    planning_handoffs: dict
    state_root: pathlib.Path
    state_task_id: str
    request_fingerprint: str
    degraded_scout_note: str
    reviewed_result_hashes: dict[str, str]
    requested_context_level: str = ""
    effective_context_level: str = ""
    context_degradation_reason: str = ""


def get_tools():
    return [
        ToolEntry(
            name="plan_task",
            schema={
                "name": "plan_task",
                "description": (
                    "Run a pre-implementation design review of a proposed plan. It first starts a small "
                    "local-readonly planning-scout subagent swarm and waits for every started scout to "
                    "finish or reach one shared OUROBOROS_PLAN_TASK_SWARM_MAX_WAIT_SEC cutoff "
                    "for raw handoffs, then runs the configured reviewer slots (an arbitrary N, "
                    "duplicates allowed) in parallel. Call this BEFORE writing any code for non-trivial tasks (>2 files or >50 lines "
                    "of changes). The agent chooses the context level: minimal includes governance docs, the plan, "
                    "and touched-file snapshots; localized/broad/constitutional add a generated repository Atlas. "
                    "Reviewers identify forgotten touchpoints, implicit contract "
                    "violations, simpler alternatives, and Bible/architecture compliance issues — before you've "
                    "written a single line. Uses the reviewer slots configured in OUROBOROS_REVIEW_MODELS (same "
                    "slot as the commit triad); duplicate model IDs are allowed and count as separate stochastic "
                    "slots. Returns structured feedback from every reviewer slot with detailed explanations and "
                    "alternative approaches. GREEN closes the exact plan fingerprint; REVIEW_REQUIRED "
                    "is closed by a second call containing review_disposition ONLY, bound to the latest "
                    "stored fingerprint and requiring no new LLM call; "
                    "Blocking REVISE_PLAN requires changed plan text and a fresh review; "
                    "advisory mode may proceed under loud disclosure with main-agent rationale."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "plan": {"type": "string", "description": "Review mode only: describe what you plan to implement, including files, key decisions, and non-goals. Omit this field in disposition mode."},
                        "goal": {"type": "string", "description": "Review mode only: the high-level goal. Omit this field in disposition mode."},
                        "plan_class": {
                            "type": "string",
                            "enum": ["self_mod", "external", "creative", "research"],
                            "description": (
                                "What KIND of plan this is — your own classification: self_mod (changes to the "
                                "Ouroboros system repo — full governance pack), external (an external codebase/"
                                "workspace), creative (content/design/site deliverables), research (investigation/"
                                "analysis). The host STRUCTURALLY escalates to self_mod when files_to_touch resolve "
                                "under the system repo. Non-self_mod classes get a leaner doc pack (ARCHITECTURE as "
                                "a navigation map) and task-fit scout framing."
                            ),
                        },
                        "files_to_touch": {"type": "array", "description": "Review mode only: repo-relative files you plan to modify. Their HEAD snapshots inform reviewers, and the list enters the review fingerprint. Omit this field in disposition mode; never resend it with review_disposition.", "items": {"type": "string"}},
                        "context_level": {
                            "type": "string",
                            "enum": ["minimal", "localized", "broad", "constitutional"],
                            "description": (
                                "Agent-chosen repository context level. Choose explicitly: minimal omits generated "
                                "Atlas context but keeps governance docs and touched-file snapshots; localized adds "
                                "a small Atlas around files_to_touch; broad is for shared contracts; constitutional "
                                "is for self-evolution/immune surfaces. For non-self_mod plan classes it may be "
                                "omitted and defaults to minimal."
                            ),
                        },
                        "context_notes": {
                            "type": "string",
                            "default": "",
                            "description": "Optional agent-chosen notes explaining why this context level/evidence is appropriate.",
                        },
                        "scope": {
                            "type": "object",
                            "additionalProperties": False,
                            "description": (
                                "Optional structured intent boundary shown beside the goal: what is in scope, "
                                "mandatory invariants, non-goals, the existing seam selected for extension, "
                                "explicitly rejected expansions, and optional pre-work acceptance claims."
                            ),
                            "properties": {
                                "in_scope": {"type": "array", "items": {"type": "string"}},
                                "invariants": {"type": "array", "items": {"type": "string"}},
                                "non_goals": {"type": "array", "items": {"type": "string"}},
                                "selected_seam": {"type": "string"},
                                "rejected_expansions": {"type": "array", "items": {"type": "string"}},
                                "acceptance_claims": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": (
                                        "Optional pre-work acceptance claims: concrete, checkable statements of "
                                        "what 'done' means (plain strings). They enter the plan fingerprint when "
                                        "set, reviewers see them beside the goal, and the CLOSED plan's claims "
                                        "bind task acceptance (ids claim_1..N in list order — link "
                                        "verify_and_record receipts via criterion_id). Keep the list stable "
                                        "across re-plans (append-only) so receipt links survive; omit unless "
                                        "you can state real checks — empty/blank values are treated as absent."
                                    ),
                                },
                            },
                        },
                        "review_disposition": {
                            "type": "object",
                            "additionalProperties": False,
                            "description": (
                                "Disposition mode only: resolve the latest integrated, non-degraded "
                                "REVIEW_REQUIRED result without another reviewer call. Send ONLY this "
                                "field; do not resend plan, goal, scope, files, or context. Every reported "
                                "finding id must appear exactly once."
                            ),
                            "properties": {
                                "review_fingerprint": {"type": "string"},
                                "items": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "additionalProperties": False,
                                        "properties": {
                                            "finding_id": {"type": "string"},
                                            "decision": {
                                                "type": "string",
                                                "enum": ["accept", "reject", "defer"],
                                            },
                                            "rationale": {"type": "string"},
                                            "plan_revision": {
                                                "type": "string",
                                                "description": (
                                                    "Required for accept: a concrete reference to the "
                                                    "corresponding revision/implementation adjustment."
                                                ),
                                            },
                                        },
                                        "required": ["finding_id", "decision", "rationale"],
                                    },
                                },
                            },
                            "required": ["review_fingerprint", "items"],
                        },
                        "include_tests": {
                            "type": "boolean",
                            "default": False,
                            "description": "Whether generated Atlas context may include related tests.",
                        },
                    },
                    # The handler enforces two mutually exclusive modes: plan+goal for a
                    # fresh review, or review_disposition alone for a stored result.
                    "required": [],
                },
            },
            handler=_handle_plan_task,
            timeout_sec=_PLAN_TASK_TOOL_TIMEOUT_SEC,
        )
    ]


def _handle_plan_task(ctx: ToolContext, **params) -> str:
    raw_disposition = params.get("review_disposition")
    disposition_present = "review_disposition" in params
    vacuous_disposition = _vacuous_review_disposition(raw_disposition)
    envelope_fields = sorted(set(params) - {"review_disposition"})
    if disposition_present and raw_disposition is not None and not vacuous_disposition:
        if envelope_fields:
            return (
                "ERROR: PLAN_REVIEW_DISPOSITION_MIXED_ENVELOPE: disposition mode accepts "
                "review_disposition only. Do not resend plan, goal, scope, files_to_touch, "
                "or context fields; edits require a new review-mode call without "
                "review_disposition. No plan attempt was recorded."
            )
        if not isinstance(raw_disposition, dict):
            return "ERROR: PLAN_REVIEW_DISPOSITION_INVALID: review_disposition must be an object"
        claimed = str(raw_disposition.get("review_fingerprint") or "").strip()
        if not claimed:
            return "ERROR: PLAN_REVIEW_DISPOSITION_INVALID: review_fingerprint is required"
        result = _reuse_or_disposition_plan_review(ctx, claimed, raw_disposition)
        return result or "ERROR: PLAN_REVIEW_DISPOSITION_UNBINDABLE: stored review disappeared"
    if disposition_present and vacuous_disposition and not envelope_fields:
        return (
            "ERROR: PLAN_REVIEW_DISPOSITION_EMPTY: submit non-empty plan and goal for "
            "review mode, or a "
            "complete review_disposition as the only field for disposition mode. "
            "No plan attempt was recorded."
        )
    try:
        state_root, state_task_id = _planning_state_location(ctx)
        _record_raw_plan_request_attempt(params, state_root, state_task_id, reason="plan_envelope_pending")
    except (OSError, TimeoutError, ValueError) as exc:
        return f"ERROR: PLAN_REVIEW_STATE_PERSIST_FAILED: {exc}"
    request = _PlanReviewRequest(
        plan=str(params.get("plan") or ""),
        goal=str(params.get("goal") or ""),
        files_to_touch=list(params.get("files_to_touch") or []),
        context_level=str(params.get("context_level") or ""),
        context_notes=str(params.get("context_notes") or ""),
        include_tests=bool(params.get("include_tests", False)),
        plan_class=str(params.get("plan_class") or ""),
        scope=params.get("scope"),
        review_disposition=None,
    )
    try:
        try:
            asyncio.get_running_loop()
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                result = pool.submit(
                    asyncio.run,
                    asyncio.wait_for(
                        _run_plan_review_async(ctx, request), timeout=_PLAN_REVIEW_WRAPPER_TIMEOUT_SEC,
                    ),
                ).result(timeout=_PLAN_REVIEW_WRAPPER_TIMEOUT_SEC + 5)
        except RuntimeError:
            result = asyncio.run(
                asyncio.wait_for(
                    _run_plan_review_async(ctx, request), timeout=_PLAN_REVIEW_WRAPPER_TIMEOUT_SEC,
                )
            )
        if isinstance(result, str) and vacuous_disposition:
            result += _VACUOUS_DISPOSITION_NOTE
        if isinstance(result, str) and _vacuous_acceptance_claims(params.get("scope")):
            result += _VACUOUS_CLAIMS_NOTE
        return result
    except (concurrent.futures.TimeoutError, asyncio.TimeoutError):
        return _plan_unavailable(
            ctx, f"ERROR: Plan review timed out after {_PLAN_REVIEW_WRAPPER_TIMEOUT_SEC}s.", "review_timeout")
    except Exception as e:
        log.error("plan_task failed: %s", e, exc_info=True)
        return _plan_unavailable(ctx, f"ERROR: Plan review failed: {e}", "review_failed")


def _plan_unavailable(ctx: ToolContext, message: str, reason: str) -> str:
    """Persist a retryable availability outcome after canonical validation."""
    try:
        root, task_id = _planning_state_location(ctx)
        mark_current_plan_review_unavailable(root, task_id, reason=reason)
    except (OSError, TimeoutError, ValueError) as exc:
        return f"{message}\nERROR: PLAN_REVIEW_STATE_PERSIST_FAILED: {exc}"
    return message


def _persist_planning_handoffs(ctx: ToolContext, handoffs: dict) -> dict:
    task_id = str(getattr(ctx, "task_id", "") or "plan_review")
    return persist_plan_review_handoffs(ctx.drive_root, task_id, handoffs)


def _persist_planning_snapshot(ctx: ToolContext, handoffs: dict) -> dict:
    """Freeze the scout-handoff input once, immediately before panel dispatch."""
    task_id = str(getattr(ctx, "task_id", "") or "plan_review")
    return persist_plan_review_handoff_snapshot(ctx.drive_root, task_id, handoffs)


def _planning_handoff_path(ctx: ToolContext) -> pathlib.Path:
    task_id = str(getattr(ctx, "task_id", "") or "plan_review")
    return pathlib.Path(ctx.drive_root) / "task_results" / "artifacts" / task_id / "plan_task_handoffs.json"


def _planning_state_location(ctx: ToolContext) -> tuple[pathlib.Path, str]:
    root = pathlib.Path(str(getattr(ctx, "budget_drive_root", "") or ctx.drive_root))
    task_id = str(getattr(ctx, "task_id", "") or "").strip()
    if not task_id:
        raise ValueError(
            "PLAN_REVIEW_TASK_ID_REQUIRED: durable review state must belong to a real task"
        )
    return root, task_id


def _collect_planning_handoffs(
    ctx: ToolContext, *, task_ids: list[str], schedule_outputs: list[str], fingerprint: str,
    wait_timeout: float, max_wait: float = 0.0, intended_scouts: list[dict] | None = None,
    cutoff_at: str = "",
) -> dict:
    """Wait for every started scout until terminal or the one shared cutoff."""
    status_root = pathlib.Path(str(getattr(ctx, "budget_drive_root", "") or ctx.drive_root))
    slice_sec = max(0.25, float(wait_timeout or 0))
    ceiling = max(0.0, float(max_wait or slice_sec))
    cutoff = parse_deadline_ts(cutoff_at) if cutoff_at else _planning_now() + timedelta(seconds=ceiling)
    if cutoff is None:
        raise ValueError("PLAN_REVIEW_STATE_INVALID: scout_cutoff_at is malformed")
    start = time.monotonic()
    remaining_at_start = max(0.0, (cutoff - _planning_now()).total_seconds())
    stop_reason = ""
    waited: dict = {}
    while True:
        remaining = (cutoff - _planning_now()).total_seconds()
        if remaining <= 0.01:
            waited = wait_for_effective_tasks(
                status_root, task_ids, timeout_sec=0.0, mode="all_terminal", poll_interval_sec=0.25,
            )
            stop_reason = "ceiling"
            break
        waited = wait_for_effective_tasks(
            status_root,
            task_ids,
            timeout_sec=min(slice_sec, remaining),
            mode="all_terminal",
            poll_interval_sec=0.25,
        )
        tasks = waited.get("tasks") if isinstance(waited.get("tasks"), dict) else {}
        if waited.get("all_terminal") or _all_planning_tasks_known_terminal(task_ids, tasks or {}):
            break
        # Heartbeats are diagnostic; every scout gets the same terminal-or-cutoff window.
    tasks = waited.get("tasks") if isinstance(waited.get("tasks"), dict) else {}
    attempts = intended_scouts or [
        {"role": str((tasks.get(task_id) or {}).get("role") or ""),
         "schedule_status": "started", "task_ids": [task_id], "schedule_reason": ""}
        for task_id in task_ids
    ]
    included_task_ids, omissions = _planning_handoff_selection(attempts, tasks or {}, stop_reason)
    handoffs = {
        "schema_version": 1,
        "ts": utc_now_iso(),
        "request_fingerprint": fingerprint,
        "task_ids": task_ids,
        "schedule_outputs": schedule_outputs,
        "scout_cutoff_at": cutoff.isoformat(),
        "wait": waited,
        "wait_stop_reason": stop_reason,
        "wait_elapsed_sec": round(time.monotonic() - start, 2),
        "wait_remaining_at_start_sec": round(remaining_at_start, 2),
        "included_task_ids": included_task_ids,
        "omissions": omissions,
        "consumed_task_ids": [],
    }
    handoffs["artifact"] = _persist_planning_handoffs(ctx, handoffs)
    return handoffs


def _planning_swarm_timing(ctx: ToolContext) -> tuple[float, float]:
    from ouroboros.config import get_plan_task_swarm_timeout_sec

    wait_timeout, max_wait = get_plan_task_swarm_timeout_sec(), _effective_swarm_max_wait()
    metadata = getattr(ctx, "task_metadata", {})
    metadata = metadata if isinstance(metadata, dict) else {}
    deadline = parse_deadline_ts(metadata.get("deadline_at"))
    if deadline is not None:
        remaining = (deadline - _planning_now()).total_seconds()
        max_wait = 0.0 if remaining <= 0 else min(max_wait, remaining / 4.0)
    event_queue = getattr(ctx, "event_queue", None)
    live = event_queue is not None and event_queue.__class__.__module__ in {
        "queue",
        "multiprocessing.queues",
        "multiprocessing.managers",
    }
    if not live:
        wait_timeout = min(wait_timeout, 0.25)
        max_wait = min(max_wait, wait_timeout)
    return wait_timeout, max_wait


def _planning_direct_children(ctx: ToolContext) -> dict[str, dict]:
    """Read durable direct-child authority for scheduling recovery."""
    root, parent_id = _planning_state_location(ctx)
    try:
        rows = find_child_tasks(
            root,
            parent_task_id=parent_id,
            root_task_id="",
            exclude_task_id=parent_id,
            scope="direct",
        )
    except Exception:
        log.debug("plan_task could not read durable child authority", exc_info=True)
        return {}
    return {
        str(row.get("task_id") or row.get("id") or ""): row
        for row in rows
        if isinstance(row, dict) and str(row.get("task_id") or row.get("id") or "")
    }


def _scheduled_side_channel_ids(ctx: ToolContext) -> list[str]:
    records = getattr(ctx, "_last_scheduled_subagents", [])
    if not isinstance(records, list):
        return []
    return list(dict.fromkeys(
        str(task_id)
        for record in records
        if isinstance(record, dict)
        for task_id in (record.get("task_ids") or [])
        if str(task_id)
    ))

def _schedule_planning_scouts(
    ctx: ToolContext, wave: dict, *, fingerprint: str, objective: str, constraints: str, context: str,
    deadline_at: str = "",
) -> dict:
    from ouroboros.tools.control import _schedule_task

    root, parent_id = _planning_state_location(ctx)
    for attempt in wave.get("intended_scouts") or []:
        if str(attempt.get("schedule_status") or "") != "pending":
            continue
        role = str(attempt.get("role") or "")
        before_side = set(_scheduled_side_channel_ids(ctx))
        before_durable = set(_planning_direct_children(ctx))
        try:
            # `deadline_at` became a public schedule_subagent parameter in v6.87.7, so the
            # scout deadline now rides the same channel any parent would use. Narrowing is
            # enforced downstream: the earlier of this and the parent's deadline wins.
            output = _schedule_task(
                ctx, objective=objective, deadline_at=deadline_at,
                expected_output=("A concise planning handoff with sections: summary, missed_touchpoints, "
                                 "risks, suggested_scope_adjustments, tests_to_run, blockers."),
                role=role, context=context, constraints=constraints, memory_mode="forked", model_lane="light",
            )
            reason = _bounded_planning_reason(output)
        except Exception as exc:
            output, reason = "", _bounded_planning_reason(f"{type(exc).__name__}: {exc}")
        after_side = [task_id for task_id in _scheduled_side_channel_ids(ctx) if task_id not in before_side]
        after_durable = [
            task_id
            for task_id, row in _planning_direct_children(ctx).items()
            if task_id not in before_durable and str(row.get("role") or "") == role
        ]
        after = list(dict.fromkeys(after_side + after_durable))
        if len(after) > 1:
            raise ValueError(
                "PLAN_REVIEW_STATE_INVALID: one planning scout intent issued multiple child ids"
            )
        status = "started" if after else "failed"
        if after:
            reason = reason or "scheduled"
        else:
            reason = _bounded_planning_reason(
                "host issued no task id" + (f"; {reason}" if reason else "")
            )
        wave = record_plan_review_scout(
            root, parent_id, fingerprint=fingerprint, role=role, schedule_status=status,
            task_ids=after, reason=reason,
        )
    return wave


def _recover_pending_planning_scouts(ctx: ToolContext, state: dict, wave: dict, *, fingerprint: str) -> dict:
    """Resolve an interrupted schedule from durable child rows before declaring omission."""
    root, parent_id = _planning_state_location(ctx)
    assigned = {
        str(task_id)
        for stored_wave in state.get("waves") or []
        for attempt in stored_wave.get("intended_scouts") or []
        for task_id in (attempt.get("task_ids") or [])
        if str(task_id)
    }
    children = _planning_direct_children(ctx)
    created_at = parse_deadline_ts(wave.get("created_at"))
    for attempt in list(wave.get("intended_scouts") or []):
        if str(attempt.get("schedule_status") or "") != "pending":
            continue
        role = str(attempt.get("role") or "")
        candidates: list[str] = []
        if created_at is not None:
            for task_id, row in children.items():
                row_ts = parse_deadline_ts(row.get("ts"))
                if (
                    task_id not in assigned
                    and str(row.get("role") or "") == role
                    and row_ts is not None
                    and row_ts >= created_at
                ):
                    candidates.append(task_id)
        if len(candidates) == 1:
            status = "started"
            reason = "recovered durable issued child id after interrupted scheduling"
            assigned.add(candidates[0])
        else:
            status = "unknown"
            reason = (
                "scheduling was interrupted and durable child authority was ambiguous"
                if len(candidates) > 1
                else "scheduling was interrupted before any issued child id was durably recoverable"
            )
            candidates = []
        wave = record_plan_review_scout(
            root,
            parent_id,
            fingerprint=fingerprint,
            role=role,
            schedule_status=status,
            task_ids=candidates,
            reason=reason,
        )
    return wave


def _collect_host_planning_wave(
    ctx: ToolContext, wave: dict, *, fingerprint: str, wait_timeout: float, max_wait: float,
) -> tuple[dict, dict]:
    root, parent_id = _planning_state_location(ctx)
    task_ids = plan_review_wave_task_ids(wave)
    handoffs = _collect_planning_handoffs(
        ctx, task_ids=task_ids,
        schedule_outputs=[str(item.get("schedule_reason") or "") for item in wave.get("intended_scouts") or []],
        fingerprint=fingerprint, wait_timeout=wait_timeout, max_wait=max_wait,
        intended_scouts=list(wave.get("intended_scouts") or []),
        cutoff_at=str(wave.get("scout_cutoff_at") or ""),
    )
    wave = record_plan_review_collection(
        root, parent_id, fingerprint=fingerprint,
        included_task_ids=list(handoffs.get("included_task_ids") or []),
        omissions=list(handoffs.get("omissions") or []),
        stop_reason=str(handoffs.get("wait_stop_reason") or ""),
    )
    handoffs.update({key: value for key, value in plan_review_wave_handoffs(wave).items() if key != "wait"})
    handoffs["artifact"] = _persist_planning_handoffs(ctx, handoffs)
    return handoffs, wave


def _retry_reviewed_wave_handoffs(ctx: ToolContext, wave: dict) -> dict:
    """Reuse the exact scout snapshot seen by the first reviewer attempt."""
    handoffs = plan_review_wave_handoffs(wave)
    path = plan_review_handoff_snapshot_path(
        ctx.drive_root, str(getattr(ctx, "task_id", "") or "plan_review"),
        str(wave.get("request_fingerprint") or ""),
    )
    stored = read_json_dict(path)
    error = ""
    if not isinstance(stored, dict):
        error = "reviewed planning handoff snapshot is missing or invalid"
    elif stored.get("request_fingerprint") != handoffs.get("request_fingerprint"):
        error = "reviewed planning handoff snapshot belongs to another plan"
    elif stored.get("audit_only") is not True or stored.get("authoritative") is not False:
        error = "reviewed planning handoff snapshot has invalid provenance"
    else:
        handoffs["wait"] = stored.get("wait") if isinstance(stored.get("wait"), dict) else {}
        try:
            actual_hashes = _reviewed_handoff_hashes(handoffs)
        except ValueError as exc:
            error = str(exc)
        else:
            expected_hashes = dict(wave.get("reviewed_result_hashes") or {})
            if expected_hashes and actual_hashes != expected_hashes:
                error = "reviewed planning handoff snapshot hash does not match durable review state"
            elif (
                isinstance(wave.get("review"), dict)
                and wave.get("included_task_ids")
                and not expected_hashes
            ):
                error = "reviewed planning handoff snapshot has no durable evidence hashes"
    handoffs["artifact"] = {
        "kind": "plan_task_handoffs",
        **({"error": error} if error else {"name": path.name, "path": str(path)}),
    }
    return handoffs


def _start_planning_swarm(
    ctx: ToolContext,
    request: _PlanReviewRequest,
    fingerprint: str,
) -> dict:
    """Reserve/resume the scout wave for an ALREADY-COMPUTED binding fingerprint.

    The caller passes it in: recomputing it here from the host-RESOLVED request would
    key the wave under a different identity than the one the agent can name."""
    from ouroboros.config import get_finalization_grace_sec, get_light_model, get_max_workers

    plan = request.plan
    files_to_touch = request.files_to_touch
    context_level = request.context_level
    plan_class = request.plan_class or "self_mod"
    wait_timeout, max_wait = _planning_swarm_timing(ctx)
    root, parent_id = _planning_state_location(ctx)
    try:
        state = load_plan_review_state(root, parent_id)
        wave = plan_review_wave(state, fingerprint)
        resumed = wave is not None
        created = False
        review = wave.get("review") if isinstance((wave or {}).get("review"), dict) else {}
        orphan_snapshot = not review and plan_review_handoff_snapshot_path(
            ctx.drive_root, str(getattr(ctx, "task_id", "") or "plan_review"), fingerprint,
        ).is_file()
        snapshot_retry = bool(wave and (review.get("reviewer_slots_degraded") is True or orphan_snapshot))
        if snapshot_retry:
            handoffs = _retry_reviewed_wave_handoffs(ctx, wave)
            tasks = (
                handoffs.get("wait", {}).get("tasks", {})
                if isinstance(handoffs.get("wait"), dict) else {}
            )
            return {
                "started": not bool((handoffs.get("artifact") or {}).get("error")),
                "task_ids": plan_review_wave_task_ids(wave),
                "handoffs": handoffs,
                "resumed": True,
                "degraded_evidence": not bool(_completed_planning_handoffs(tasks or {})),
                **({"error": handoffs["artifact"]["error"]}
                   if (handoffs.get("artifact") or {}).get("error") else {}),
            }
        if wave is None:
            try:
                from ouroboros.config import get_max_active_subagents_per_root
                _cap = get_max_active_subagents_per_root()
            except Exception:
                _cap = 3
            _desired = 2 if context_level in {"broad", "constitutional"} or len(files_to_touch or []) > 3 else 1
            roles = [f"planning-scout-{idx + 1}" for idx in range(max(1, min(int(_cap or 1), _desired)))]
            scope_claims = request.scope.get("acceptance_claims") if isinstance(request.scope, dict) else None
            wave, created = reserve_plan_review_wave(
                root, parent_id, fingerprint=fingerprint, plan_text_hash=plan_text_fingerprint(plan),
                scout_roles=roles, cutoff_at=(_planning_now() + timedelta(seconds=max_wait)).isoformat(),
                acceptance_claims=scope_claims if isinstance(scope_claims, list) else None,
            )
            resumed = not created
            if created:
                objective, constraints = _planning_scout_framing(plan_class)
                context = _planning_swarm_context(
                    plan=plan, goal=request.goal, files_to_touch=files_to_touch,
                    context_level=context_level, context_notes=request.context_notes,
                    scope=request.scope,
                )
                # Admission for a NEW wave ONLY. The recovery/collection path below gathers
                # handoffs that are already PAID — gating those would abandon spend, not save it.
                # The worker-capacity refusal lives inside the wave plan (max_workers < 2).
                scout_deadline, refusal = _planning_scout_wave_plan(
                    str(wave.get("scout_cutoff_at") or ""), max_workers=get_max_workers(),
                    grace_sec=get_finalization_grace_sec(), now=_planning_now(),
                )
                admission = None if refusal else review_wave_budget_gate(
                    ctx, surface="plan_task_scouts", max_completion_tokens=_PLAN_SCOUT_MAX_TOKENS,
                    models=[get_light_model()] * len(wave.get("intended_scouts") or []),
                    prompt_chars=len(objective) + len(constraints) + len(context),
                )
                if admission is not None:
                    refusal = (
                        "the scout wave was declined before dispatch — estimated ~$"
                        f"{admission.get('estimated_wave_usd')} exceeds the remaining root budget "
                        f"${admission.get('remaining_usd')} (limit ${admission.get('limit_usd')})"
                    )
                if refusal:
                    for attempt in list(wave.get("intended_scouts") or []):
                        wave = record_plan_review_scout(
                            root, parent_id, fingerprint=fingerprint, role=str(attempt.get("role") or ""),
                            schedule_status="failed", task_ids=[], reason=refusal,
                        )
                else:
                    wave = _schedule_planning_scouts(
                        ctx, wave, fingerprint=fingerprint, objective=objective, constraints=constraints,
                        context=context, deadline_at=scout_deadline,
                    )
        if not created and any(
            str(item.get("schedule_status") or "") == "pending"
            for item in wave.get("intended_scouts") or []
        ):
            wave = _recover_pending_planning_scouts(ctx, state, wave, fingerprint=fingerprint)
        handoffs, wave = _collect_host_planning_wave(
            ctx, wave, fingerprint=fingerprint, wait_timeout=wait_timeout, max_wait=max_wait,
        )
    except (OSError, TimeoutError, ValueError) as exc:
        return {"started": False, "error": f"ERROR: PLAN_SCOUT_WAVE_STATE_PERSIST_FAILED: {exc}"}

    task_ids = plan_review_wave_task_ids(wave)
    wait_payload = handoffs.get("wait") if isinstance(handoffs.get("wait"), dict) else {}
    tasks = wait_payload.get("tasks") if isinstance(wait_payload.get("tasks"), dict) else {}
    completed = _completed_planning_handoffs(tasks or {})
    if not (handoffs.get("artifact") or {}).get("path"):
        return {"started": False, "error": "ERROR: raw planning handoff audit could not be saved.",
                "task_ids": task_ids, "handoffs": handoffs, "resumed": resumed}
    return {"started": True, "task_ids": task_ids, "handoffs": handoffs, "resumed": resumed,
            "degraded_evidence": not bool(completed)}


def _mark_planning_handoffs_consumed(ctx: ToolContext, handoffs: dict) -> dict:
    """Mark exactly the handoffs embedded in the reviewer request as consumed."""
    included = [str(item) for item in (handoffs.get("included_task_ids") or []) if str(item)]
    from ouroboros.tools.join_ledger import (
        CHILD_RESULT_DISPOSITION_TYPE,
        _record_child_result_disposition,
    )

    reviewed_hashes = handoffs.get("reviewed_result_hashes")
    if not isinstance(reviewed_hashes, dict) or set(reviewed_hashes) != set(included):
        # Compatibility for callers/tests that consume before the paid review is
        # stored. Production resume always uses the durable exact hash mapping.
        reviewed_hashes = _reviewed_handoff_hashes(handoffs)

    disposition_warnings: list[dict] = []
    for child_task_id in included:
        recorded = _record_child_result_disposition(
            ctx,
            {
                "type": CHILD_RESULT_DISPOSITION_TYPE,
                "child_task_id": child_task_id,
                "disposition": "integrated",
                "child_result_sha256": str(reviewed_hashes[child_task_id]),
            },
            "The exact planning scout handoff was embedded in the plan-review request.",
        )
        if "CHILD_RESULT_STALE" in recorded:
            disposition_warnings.append({
                "task_id": child_task_id,
                "code": "CHILD_RESULT_STALE",
                "detail": _bounded_planning_reason(recorded),
            })
        elif not recorded.startswith("OK:"):
            raise ValueError(recorded)
    root, task_id = _planning_state_location(ctx)
    wave = record_plan_review_consumed(
        root, task_id, fingerprint=str(handoffs.get("request_fingerprint") or ""),
        consumed_task_ids=included,
        disposition_warnings=disposition_warnings,
    )
    handoffs.update({key: value for key, value in plan_review_wave_handoffs(wave).items() if key != "wait"})
    handoffs.pop("reviewed_result_hashes", None)
    handoffs.pop("review_evidence_status", None)
    handoffs.setdefault("wait", {})
    handoffs["artifact"] = _persist_planning_handoffs(ctx, handoffs)
    return wave


def _capture_late_planning_audit(ctx: ToolContext, handoffs: dict) -> None:
    """Record late omitted results for audit without feeding or reopening review."""
    omitted_ids = [
        str(item.get("task_id") or "")
        for item in (handoffs.get("omissions") or [])
        if isinstance(item, dict) and str(item.get("task_id") or "")
    ]
    if not omitted_ids:
        return
    status_root = pathlib.Path(str(getattr(ctx, "budget_drive_root", "") or ctx.drive_root))
    current = wait_for_effective_tasks(
        status_root,
        omitted_ids,
        timeout_sec=0.0,
        mode="all_terminal",
        poll_interval_sec=0.25,
    )
    tasks = current.get("tasks") if isinstance(current.get("tasks"), dict) else {}
    late_tasks = {
        task_id: row
        for task_id, row in (tasks or {}).items()
        if isinstance(row, dict)
        and str(row.get("status") or "").strip().lower() in FINAL_STATUSES
        and str(row.get("result") or "").strip()
    }
    if not late_tasks:
        return
    handoffs["late_audit"] = {
        "captured_at": utc_now_iso(),
        "affects_review": False,
        "tasks": late_tasks,
    }
    handoffs["artifact"] = _persist_planning_handoffs(ctx, handoffs)




def _apply_review_disposition(
    ctx: ToolContext,
    audit: dict,
    review: dict,
    fingerprint: str,
    disposition: dict,
) -> str:
    updated, error = validate_plan_review_disposition(review, fingerprint, disposition)
    if error or updated is None:
        return error
    updated["disposition"]["recorded_at"] = utc_now_iso()
    root, task_id = _planning_state_location(ctx)
    try:
        wave = record_plan_review_result(
            root, task_id, fingerprint=fingerprint, review=updated, require_latest=True,
        )
    except (OSError, TimeoutError, ValueError) as exc:
        return "ERROR: PLAN_REVIEW_STATE_PERSIST_FAILED: " + str(exc)
    audit.update(plan_review_wave_handoffs(wave))
    persisted = _persist_planning_handoffs(ctx, audit)
    if persisted.get("error"):
        return "ERROR: PLAN_REVIEW_STATE_PERSIST_FAILED: " + str(persisted["error"])
    audit["artifact"] = persisted
    return _planning_disposition_warning_note(audit) + _render_existing_plan_review(updated)


def _planning_disposition_warning_note(handoffs: dict) -> str:
    warnings = handoffs.get("disposition_warnings")
    count = len(warnings) if isinstance(warnings, list) else 0
    if not count:
        return ""
    return (
        "⚠️ PLANNING SCOUT SNAPSHOT CHANGED: "
        f"{count} reviewer-included scout result(s) changed after the exact snapshot "
        "was sent to the panel. The reviewed snapshot remains plan evidence; the newer "
        "result is audit-only and was not marked integrated.\n\n"
    )


def _reuse_or_disposition_plan_review(
    ctx: ToolContext,
    fingerprint: str,
    review_disposition: dict | None,
    plan_text_hash: str = "",
) -> str | None:
    if _vacuous_review_disposition(review_disposition):
        review_disposition = None  # vacuous == absent (see review_synthesis)
    try:
        root, task_id = _planning_state_location(ctx)
        state = load_plan_review_state(root, task_id)
    except (OSError, TimeoutError, ValueError) as exc:
        return "ERROR: PLAN_REVIEW_STATE_INVALID: " + str(exc)
    wave = plan_review_wave(state, fingerprint)
    review = wave.get("review") if isinstance((wave or {}).get("review"), dict) else {}
    expected_fp = str(review.get("request_fingerprint") or "")
    latest = str(state.get("latest_review_fingerprint") or "")
    attempt = state.get("current_attempt") if isinstance(state.get("current_attempt"), dict) else {}
    disposition_is_current = (
        str(attempt.get("status") or "") == "open"
        and str(attempt.get("fingerprint") or "") == fingerprint
    )
    if review_disposition is not None and (
        fingerprint != latest or not disposition_is_current
    ):
        return (
            "ERROR: PLAN_REVIEW_DISPOSITION_STALE: disposition mode can close only the "
            "latest still-current review; a newer or unavailable plan attempt supersedes "
            f"it (latest={latest or 'none'}, claimed={fingerprint}). No plan attempt was recorded."
        )
    prior_revise = next((
        item for item in state.get("waves") or []
        if isinstance(item.get("review"), dict)
        and str(item["review"].get("aggregate_signal") or "") == "REVISE_PLAN"
        and str(item.get("request_fingerprint") or "") != fingerprint
        and plan_text_hash and str(item["review"].get("plan_text_hash") or "") == plan_text_hash
    ), None)
    if prior_revise is not None:
        return (
            "ERROR: PLAN_REVIEW_REVISION_REQUIRED: blocking mode requires changed plan "
            "text and a fresh fingerprint; advisory mode may instead proceed only under "
            "the host's loud disclosure."
        )
    if not review or expected_fp != fingerprint:
        if review_disposition is None:
            return None
        return (
            "ERROR: PLAN_REVIEW_DISPOSITION_UNBINDABLE: the latest fingerprint has no "
            "stored reviewer result. No plan attempt was recorded."
        )
    if review_disposition is not None and (
        str((wave or {}).get("phase") or "") != "reviewed"
        or str((wave or {}).get("review_evidence_status") or "integrated") != "integrated"
    ):
        return (
            "ERROR: PLAN_REVIEW_DISPOSITION_NOT_READY: the latest reviewer result is not "
            "fully integrated into durable scout evidence. No plan attempt was recorded."
        )
    if str((wave or {}).get("review_evidence_status") or "") == "pending":
        try:
            wave = _mark_planning_handoffs_consumed(ctx, dict(wave or {}))
            state = load_plan_review_state(root, task_id)
        except (OSError, TimeoutError, ValueError) as exc:
            return "ERROR: PLAN_REVIEW_STATE_PERSIST_FAILED: " + str(exc)
        review = wave.get("review") if isinstance(wave.get("review"), dict) else {}
        if str(wave.get("review_evidence_status") or "") != "integrated":
            return (
                "ERROR: PLAN_REVIEW_STATE_PERSIST_FAILED: stored panel result remains "
                "pending evidence integration."
            )
    audit = plan_review_wave_handoffs(wave or {})
    _capture_late_planning_audit(ctx, audit)
    audit["artifact"] = _persist_planning_handoffs(ctx, audit)
    if audit["artifact"].get("error"):
        return "ERROR: PLAN_REVIEW_STATE_PERSIST_FAILED: " + str(audit["artifact"]["error"])
    if review.get("reviewer_slots_degraded") is True:
        if review_disposition is not None:
            return (
                "ERROR: PLAN_REVIEW_RETRY_REQUIRED: reviewer availability cannot be "
                "closed by disposition. Re-call plan_task with the same unchanged plan "
                "and omit review_disposition; the existing scout wave will be reused."
            )
        return None
    aggregate = str(review.get("aggregate_signal") or "")
    if aggregate == "REVISE_PLAN":
        return (
            "ERROR: PLAN_REVIEW_REVISION_REQUIRED: this unchanged fingerprint received "
            "REVISE_PLAN. Blocking mode requires changed plan text and a fresh panel; "
            "advisory mode may proceed only under loud host disclosure and the main "
            "agent's rationale. A disposition cannot override it."
        )
    if review_disposition is not None and aggregate != "REVIEW_REQUIRED":
        return (
            "ERROR: PLAN_REVIEW_DISPOSITION_NOT_APPLICABLE: the latest review does not "
            "require a main-agent disposition."
        )
    if bool(review.get("closed")):
        if review_disposition is not None:
            replayed = _replay_closed_review_disposition(
                review, fingerprint, review_disposition,
            )
            return (
                replayed
                if replayed.startswith("ERROR:")
                else _planning_disposition_warning_note(audit) + replayed
            )
        return _planning_disposition_warning_note(audit) + _render_existing_plan_review(
            review, cached=True,
        )
    if review_disposition is not None:
        return _apply_review_disposition(ctx, audit, review, fingerprint, review_disposition)
    if str(state.get("latest_review_fingerprint") or "") != fingerprint:
        try:
            represented = represent_plan_review(
                root, task_id, fingerprint=fingerprint,
            )
        except (OSError, TimeoutError, ValueError) as exc:
            return "ERROR: PLAN_REVIEW_STATE_PERSIST_FAILED: " + str(exc)
        represented_review = (
            represented.get("review")
            if isinstance(represented.get("review"), dict)
            else review
        )
        return _planning_disposition_warning_note(audit) + _render_existing_plan_review(
            represented_review, cached=True,
        )
    return (
        "ERROR: PLAN_REVIEW_DISPOSITION_REQUIRED: this unchanged fingerprint already "
        "received REVIEW_REQUIRED. Re-call plan_task with review_disposition as the only "
        "field, covering "
        f"every finding. fingerprint={fingerprint}; finding_ids="
        + json.dumps([
            item.get("finding_id") for item in (review.get("findings") or [])
            if isinstance(item, dict)
        ], ensure_ascii=False)
    )


def _finalize_plan_review_output(
    ctx: ToolContext,
    finalization: _PlanReviewFinalization,
) -> str:
    """Persist the authoritative review result and render its public projection."""
    raw_results = finalization.raw_results
    planning_handoffs = finalization.planning_handoffs
    request_fingerprint = finalization.request_fingerprint
    ctx._last_plan_review_raw_results = raw_results
    ctx._last_plan_review_estimated_tokens = finalization.estimated_tokens
    ctx._last_plan_review_subject_root = str(finalization.subject_repo)
    ctx._last_plan_review_governance_root = str(finalization.governance_repo)
    summary = _summarize_plan_review_results(raw_results)
    aggregate_signal = str(summary["aggregate_signal"])
    reviewer_slots_degraded = bool(summary.get("degraded_count"))
    availability_only = bool(
        reviewer_slots_degraded
        and not (summary.get("review_required_count") or summary.get("revise_count"))
    )
    review_record = {
        "schema_version": 1,
        "request_fingerprint": request_fingerprint,
        "plan_text_hash": plan_text_fingerprint(finalization.request.plan),
        "aggregate_signal": aggregate_signal,
        "findings": list(summary["findings"]),
        "reviewed_at": utc_now_iso(),
        "closed": aggregate_signal == "GREEN",
        "reviewer_slots_degraded": reviewer_slots_degraded,
        "included_task_ids": list(planning_handoffs.get("included_task_ids") or []),
        "omitted_task_ids": [
            str(item.get("task_id") or "")
            for item in (planning_handoffs.get("omissions") or [])
            if isinstance(item, dict) and str(item.get("task_id") or "")
        ],
        "requested_context_level": finalization.requested_context_level,
        "effective_context_level": finalization.effective_context_level,
    }
    if finalization.context_degradation_reason:
        review_record["context_degradation_reason"] = finalization.context_degradation_reason
    if planning_handoffs:
        try:
            wave = record_plan_review_result(
                finalization.state_root, finalization.state_task_id,
                fingerprint=request_fingerprint, review=review_record,
                reviewed_result_hashes=finalization.reviewed_result_hashes,
            )
        except (OSError, TimeoutError, ValueError) as exc:
            return f"ERROR: PLAN_REVIEW_STATE_PERSIST_FAILED: {exc}"
        if not availability_only and str(wave.get("review_evidence_status") or "") == "pending":
            try:
                wave = _mark_planning_handoffs_consumed(ctx, dict(wave))
            except (OSError, TimeoutError, ValueError) as exc:
                return f"ERROR: PLAN_REVIEW_STATE_PERSIST_FAILED: {exc}"
        planning_handoffs.update({
            key: value for key, value in plan_review_wave_handoffs(wave).items() if key != "wait"
        })
        artifact = _persist_planning_handoffs(ctx, planning_handoffs)
        if artifact.get("error"):
            return "ERROR: PLAN_REVIEW_STATE_PERSIST_FAILED: " + str(artifact["error"])
        planning_handoffs["artifact"] = artifact
        _capture_late_planning_audit(ctx, planning_handoffs)
    if availability_only:
        try:
            record_plan_review_attempt(
                finalization.state_root,
                finalization.state_task_id,
                fingerprint=request_fingerprint,
                status="unavailable",
                reason="reviewer_unavailable",
            )
        except (OSError, TimeoutError, ValueError) as exc:
            return f"ERROR: PLAN_REVIEW_STATE_PERSIST_FAILED: {exc}"

    if reviewer_slots_degraded:
        next_step = (
            "Reviewer availability prevented an authoritative verdict. Re-submit the same "
            "unchanged plan in review mode without review_disposition; the existing scout "
            "wave is reused while the reviewer panel retries."
        )
    elif aggregate_signal == "GREEN":
        next_step = "Proceed with the reviewed plan."
    elif aggregate_signal == "REVIEW_REQUIRED":
        next_step = (
            "Re-call plan_task with review_disposition as the only field, naming this "
            "fingerprint and covering every finding id. Do not resend or edit the plan, "
            "goal, scope, files, or context; this path makes no new LLM call. Unanimous "
            "GREEN is not required: the main agent may accept, reject, or defer each finding. "
            "Blocking mode requires that closure; advisory mode may proceed under the "
            "host-owned loud disclosure without pretending the review was GREEN."
        )
    else:
        next_step = (
            "Blocking mode requires changed plan text and a fresh fingerprint review. "
            "Advisory mode may proceed only under loud host disclosure and the main "
            "agent's rationale. A disposition cannot override REVISE_PLAN."
        )
    footer = "\n".join([
        "", "## Plan Review Contract", "", f"**Plan fingerprint:** `{request_fingerprint}`",
        next_step, "",
        PLAN_REVIEW_CONTROL_PREFIX + json.dumps(
            {"outcome": aggregate_signal, "closed": aggregate_signal == "GREEN"},
            separators=(",", ":"),
        ),
        f"PLAN_REVIEW_OUTCOME: {aggregate_signal}", f"AGGREGATE: {aggregate_signal}",
    ])
    if availability_only:
        availability_note = (
            "⚠️ REVIEWER AVAILABILITY: this output is audit evidence, not a durable "
            "review verdict; retry the same plan fingerprint.\n\n"
        )
    elif reviewer_slots_degraded:
        availability_note = (
            "⚠️ REVIEWER AVAILABILITY: substantive findings were stored, but the "
            "review remains open until the same reviewer panel retries.\n\n"
        )
    else:
        availability_note = ""
    context_note = _render_plan_context_degradation(
        finalization.requested_context_level,
        finalization.effective_context_level,
        finalization.context_degradation_reason,
    )
    return (
        context_note
        + finalization.degraded_scout_note
        + availability_note
        + _planning_disposition_warning_note(planning_handoffs)
        + _format_output(
            raw_results,
            finalization.models,
            finalization.request.goal,
            finalization.estimated_tokens,
            availability_only=availability_only,
            reviewer_retry_required=reviewer_slots_degraded,
        )
        + "\n\n"
        + footer
    )


def _compile_plan_atlas(
    ctx: ToolContext,
    request: _PlanReviewRequest,
    subject_repo: pathlib.Path,
    governance_repo: pathlib.Path,
    snapshot_included: frozenset[str],
    fixed_prompt_tokens: int,
    plan_budget_limit: int,
):
    canonical_docs = {
        "BIBLE.md", "docs/DEVELOPMENT.md", "docs/ARCHITECTURE.md", "docs/CHECKLISTS.md",
    }
    return compile_review_context_atlas(ReviewContextAtlasRequest(
        repo_dir=subject_repo,
        anchors=tuple(request.files_to_touch),
        already_included=frozenset(
            set(snapshot_included)
            | (canonical_docs if subject_repo == governance_repo else set())
        ),
        fixed_prompt_tokens=fixed_prompt_tokens,
        target_total_tokens=_plan_context_target_tokens(request.context_level),
        hard_total_tokens=plan_budget_limit,
        include_tests=request.include_tests,
        title=f"Generated Plan Review Atlas ({request.context_level})",
        drive_root=pathlib.Path(ctx.drive_root),
    ))


async def _run_plan_review_async(
    ctx: ToolContext,
    request: _PlanReviewRequest,
    *,
    planning_handoff_override: tuple[str, str] | None = None,
    additional_context: str = "",
) -> str:
    """Own the review's resources for exactly the review's lifetime.

    The stack is passed EXPLICITLY into the body rather than stashed on ctx: a remote
    review holds a materialized mirror, and the donor made that mirror's lifetime a
    property of a mutable ctx attribute that a decorator deleted and two unrelated
    modules read. A parameter cannot be deleted by someone else."""

    with contextlib.ExitStack() as stack:
        return await _run_plan_review_body(
            ctx,
            request,
            exit_stack=stack,
            planning_handoff_override=planning_handoff_override,
            additional_context=additional_context,
        )






async def _run_plan_review_body(
    ctx: ToolContext,
    request: _PlanReviewRequest,
    *,
    exit_stack: contextlib.ExitStack,
    planning_handoff_override: tuple[str, str] | None = None,
    additional_context: str = "",
) -> str:
    plan = request.plan
    goal = request.goal
    files_to_touch = request.files_to_touch
    context_level = request.context_level
    context_notes = request.context_notes
    include_tests = request.include_tests
    plan_class = request.plan_class
    if request.review_disposition is not None and not _vacuous_review_disposition(
        request.review_disposition
    ):
        return (
            "ERROR: PLAN_REVIEW_DISPOSITION_MIXED_ENVELOPE: disposition mode accepts "
            "review_disposition only; a review request cannot carry both modes."
        )
    if request.review_disposition is not None:
        request = replace(request, review_disposition=None)
    try:
        state_root, state_task_id = _planning_state_location(ctx)
    except ValueError as exc:
        return f"ERROR: PLAN_REVIEW_STATE_INVALID: {exc}"
    scope, validation_error = _validate_plan_request_envelope(request, state_root, state_task_id)
    if validation_error:
        return validation_error
    from ouroboros import config as _cfg
    deadline_skip = _plan_deadline_skip(ctx)
    deadline_blocked = bool(deadline_skip)
    try:
        has_prior_state = bool(load_plan_review_state(state_root, state_task_id).get("waves"))
    except (OSError, TimeoutError, ValueError) as exc:
        return f"ERROR: PLAN_REVIEW_STATE_INVALID: {exc}"
    request_fingerprint = _plan_request_fingerprint(
        plan=plan, goal=goal, files_to_touch=files_to_touch, context_level=context_level,
        context_notes=context_notes, plan_class=plan_class, scope=scope, include_tests=include_tests,
    )
    try:
        record_plan_review_attempt(state_root, state_task_id, fingerprint=request_fingerprint)
    except (OSError, TimeoutError, ValueError) as exc:
        return f"ERROR: PLAN_REVIEW_STATE_PERSIST_FAILED: {exc}"
    if not deadline_blocked or has_prior_state:
        remote_snapshot, governance_repo, subject_repo, roots_error = _open_plan_subject_roots(
            ctx, files_to_touch, exit_stack,
        )
        if roots_error:
            return roots_error
        resolved_request, shape_error = _resolve_plan_shape(
            ctx, request, plan_class=plan_class, context_level=context_level,
            files_to_touch=files_to_touch, scope=scope,
        )
        if shape_error:
            return shape_error
        if deadline_blocked:
            try:
                decision = plan_review_gate_projection(
                    load_plan_review_state(state_root, state_task_id), "blocking")
            except (OSError, TimeoutError, ValueError) as exc:
                return f"ERROR: PLAN_REVIEW_STATE_INVALID: {exc}"
            if str(decision.get("status") or "") == "closed":
                existing = _reuse_or_disposition_plan_review(
                    ctx, request_fingerprint, None, plan_text_fingerprint(plan),
                )
                if existing is not None:
                    return existing
            if str(decision.get("status") or "") != "closed":
                try:
                    record_plan_review_attempt(
                        state_root, state_task_id, fingerprint=request_fingerprint,
                        status="rail_degraded", reason="plan_task_deadline")
                except (OSError, TimeoutError, ValueError) as exc:
                    return f"ERROR: PLAN_REVIEW_STATE_PERSIST_FAILED: {exc}"
                return _plan_deadline_skip(ctx, emit=True) or deadline_skip
        else:
            existing = _reuse_or_disposition_plan_review(
                ctx, request_fingerprint, None, plan_text_fingerprint(plan),
            )
            if existing is not None:
                return existing
    if deadline_blocked:
        try:
            record_plan_review_attempt(
                state_root, state_task_id, fingerprint=request_fingerprint,
                status="rail_degraded", reason="plan_task_deadline")
        except (OSError, TimeoutError, ValueError) as exc:
            return f"ERROR: PLAN_REVIEW_STATE_PERSIST_FAILED: {exc}"
        return _plan_deadline_skip(ctx, emit=True) or deadline_skip
    # #116: a malformed structured reviewer-slot config must refuse loudly here
    # instead of running the panel on the silently projected default models.
    from ouroboros.reviewer_slot_config import reviewer_slot_config_error

    if err := reviewer_slot_config_error():
        return _plan_unavailable(
            ctx,
            f"ERROR: Invalid reviewer-slot configuration blocks plan review — {err}. "
            "Fix Review lanes on the Agents tab in Settings.",
            "reviewer_slot_config_invalid")
    if not list(_cfg.get_review_models() or []):
        return _plan_unavailable(
            ctx, "ERROR: No review models configured. Set OUROBOROS_REVIEW_MODELS in settings.",
            "review_models_unconfigured")
    configured_models = _get_review_models()
    slot_limits = _per_slot_input_token_limits(
        configured_models, output_reserve=_PLAN_REVIEW_MAX_TOKENS, tokenizer_margin=155_000)
    plan_budget_limit = _quorum_input_token_limit(
        configured_models, slot_limits
    )  # quorum, not the smallest window
    degraded_scout_note = ""
    planning_handoffs: dict = {}
    reviewed_result_hashes: dict[str, str] = {}
    if planning_handoff_override is not None:
        planning_handoff_raw, planning_handoff_compact = planning_handoff_override
    else:
        swarm = _start_planning_swarm(ctx, resolved_request, request_fingerprint)
        if not swarm.get("started"):
            return _plan_unavailable(
                ctx, str(swarm.get("error") or "ERROR: plan_task planning swarm failed closed."),
                "planning_scout_unavailable")
        planning_handoffs = dict(swarm.get("handoffs") or {})
        try:
            reviewed_result_hashes = _reviewed_handoff_hashes(planning_handoffs)
        except ValueError as exc:
            return _plan_unavailable(
                ctx, f"ERROR: PLAN_REVIEW_STATE_INVALID: {exc}", "planning_evidence_invalid")
        planning_handoff_raw = _format_planning_handoffs(planning_handoffs, raw=True)
        planning_handoff_compact = _format_planning_handoffs(planning_handoffs, raw=False)
        degraded_scout_note = (
            "⚠️ DEGRADED PLANNING EVIDENCE: one or more intended scouts produced no usable "
            "handoff before the shared cutoff; reviewers received the complete "
            "omissions manifest.\n\n"
            if swarm.get("degraded_evidence") else ""
        )
    checklist = _load_plan_checklist()
    bible_text = load_governance_doc(governance_repo, "BIBLE.md", on_missing="explicit")
    dev_md = load_governance_doc(governance_repo, "docs/DEVELOPMENT.md", on_missing="explicit")
    arch_md = load_governance_doc(governance_repo, "docs/ARCHITECTURE.md", on_missing="explicit")
    checklists_md = load_governance_doc(governance_repo, "docs/CHECKLISTS.md", on_missing="explicit")
    # Non-self-mod plans use the lossless ARCHITECTURE navigation map; self-mod keeps it whole.
    if resolved_request.plan_class != "self_mod" and arch_md.strip():
        from ouroboros.context_layout import generate_doc_nav_map

        arch_md = generate_doc_nav_map(
            arch_md, title="ARCHITECTURE.md", rel_path="docs/ARCHITECTURE.md"
        )
    ctx.emit_progress_fn("📐 plan_task: reading planned-touch file snapshots…")
    head_snapshots = ""
    # Atlas inclusion claims only snapshots that survived; omission markers stay explicit.
    snapshot_included: frozenset[str] = frozenset()
    if files_to_touch:
        head_snapshots, snapshot_included = build_head_snapshot_section(
            subject_repo,
            files_to_touch,
            verified_filesystem_snapshot=remote_snapshot is not None,
        )
    requested_context_level = resolved_request.context_level
    effective_context_level = requested_context_level
    context_degradation_reason = ""
    placeholder = "__GENERATED_PLAN_ATLAS_PENDING__"

    def build_prompt(level: str, degradation_reason: str = "") -> tuple[str, str, int]:
        effective_request = replace(resolved_request, context_level=level)
        system = _build_system_prompt(
            checklist, bible_text, dev_md, arch_md, checklists_md,
            context_level=level, plan_class=resolved_request.plan_class,
        )
        user, stable_len = _build_user_content(
            effective_request, head_snapshots,
            placeholder if level != "minimal" else "",
            _render_plan_context_degradation(
                requested_context_level, level, degradation_reason,
            ),
        )
        user += "\n\n" + planning_evidence_horizon(
            ctx, governance_repo=governance_repo, subject_repo=subject_repo, scope=scope,
        )
        if planning_handoff_raw:
            user += "\n\n" + planning_handoff_raw
        if additional_context:
            user += "\n\n" + additional_context
        return system, user, stable_len

    system_prompt, user_content, user_stable_len = build_prompt(effective_context_level)
    fixed_prompt_tokens = estimate_tokens(system_prompt + user_content)
    if effective_context_level != "minimal":
        ctx.emit_progress_fn(
            f"📐 plan_task: building {effective_context_level} Generated Plan Review Atlas…"
        )
        try:
            atlas = _compile_plan_atlas(
                ctx, resolved_request, subject_repo, governance_repo, snapshot_included,
                fixed_prompt_tokens, plan_budget_limit,
            )
        except Exception as e:
            return _plan_unavailable(
                ctx, f"ERROR: Failed to build review context atlas: {e}", "review_atlas_failed")

        if atlas_assembly_failed(atlas):
            context_degradation_reason = atlas_assembly_failure_reason(atlas)
            effective_context_level = "minimal"
            system_prompt, user_content, user_stable_len = build_prompt(
                effective_context_level, context_degradation_reason,
            )
        else:
            # Replace only the stable-prefix slot, not a copy quoted by the plan/snapshot.
            slot = user_content.rfind(placeholder, 0, user_stable_len)
            if slot < 0:
                return _plan_unavailable(
                    ctx, "ERROR: Failed to build review context atlas: placeholder missing.",
                    "review_atlas_invalid")
            user_content = user_content[:slot] + atlas.text + user_content[slot + len(placeholder):]
            user_stable_len += len(atlas.text) - len(placeholder)
    estimated_tokens = estimate_tokens(system_prompt + user_content)
    if estimated_tokens > plan_budget_limit and planning_handoff_raw:
        user_content = user_content.replace(planning_handoff_raw, planning_handoff_compact)
        estimated_tokens = estimate_tokens(system_prompt + user_content)
    models, callable_slot_ids, oversize_results, fit_error = _plan_slot_fit_with_identity(
        configured_models, slot_limits, estimated_tokens)
    if fit_error and effective_context_level != "minimal":
        context_degradation_reason = (
            f"the {effective_context_level} prompt (~{estimated_tokens:,} tokens) exceeded "
            "enough calibrated reviewer input caps to leave fewer than quorum callable"
        )
        effective_context_level = "minimal"
        system_prompt, user_content, user_stable_len = build_prompt(
            effective_context_level, context_degradation_reason,
        )
        estimated_tokens = estimate_tokens(system_prompt + user_content)
        if estimated_tokens > plan_budget_limit and planning_handoff_raw:
            user_content = user_content.replace(planning_handoff_raw, planning_handoff_compact)
            estimated_tokens = estimate_tokens(system_prompt + user_content)
        models, callable_slot_ids, oversize_results, fit_error = _plan_slot_fit_with_identity(
            configured_models, slot_limits, estimated_tokens)
    if fit_error:
        return _plan_unavailable(
            ctx,
            _render_plan_context_degradation(
                requested_context_level, effective_context_level, context_degradation_reason,
            ) + fit_error,
            "review_context_unavailable",
        )
    context_degradation_note = _render_plan_context_degradation(
        requested_context_level, effective_context_level, context_degradation_reason,
    )

    # Decline an unaffordable whole wave before paying for partial slots; fail open on unknowns.
    _admission = review_wave_budget_gate(
        ctx, surface="plan_review", models=models,
        prompt_chars=len(system_prompt) + len(user_content),
        max_completion_tokens=_PLAN_REVIEW_MAX_TOKENS,
    )
    if _admission is not None:
        return _plan_unavailable(
            ctx,
            context_degradation_note
            + "⚠️ PLAN_REVIEW_SKIPPED_BUDGET: the reviewer wave was declined before "
            f"dispatch — estimated cost ~${_admission.get('estimated_wave_usd')} exceeds "
            f"the remaining root budget ${_admission.get('remaining_usd')} "
            f"(limit ${_admission.get('limit_usd')}). No reviewer was called. "
            "Shrink the plan context, split the plan, or raise the per-task budget.",
            "review_budget_unavailable",
        )
    if planning_handoffs:
        snapshot = _persist_planning_snapshot(ctx, planning_handoffs)
        if snapshot.get("error"):
            return _plan_unavailable(
                ctx,
                context_degradation_note
                + "ERROR: PLAN_REVIEW_STATE_PERSIST_FAILED: " + str(snapshot["error"]),
                "planning_evidence_snapshot_failed")
    ctx.emit_progress_fn(
        f"📐 plan_task: running {len(models)} parallel reviewers "
        f"(context={effective_context_level}, ~{estimated_tokens:,} tokens each)…"
    )

    raw_results = _assemble_plan_raw_results(oversize_results, await _run_plan_review_slots(
        ctx, models, system_prompt, user_content,
        user_stable_len=user_stable_len, slot_ids=callable_slot_ids,
    ))
    return _finalize_plan_review_output(ctx, _PlanReviewFinalization(
        request=request,
        raw_results=raw_results,
        models=models,
        estimated_tokens=estimated_tokens,
        subject_repo=subject_repo,
        governance_repo=governance_repo,
        planning_handoffs=planning_handoffs,
        state_root=state_root,
        state_task_id=state_task_id,
        request_fingerprint=request_fingerprint,
        degraded_scout_note=degraded_scout_note,
        reviewed_result_hashes=reviewed_result_hashes,
        requested_context_level=requested_context_level,
        effective_context_level=effective_context_level,
        context_degradation_reason=context_degradation_reason,
    ))
