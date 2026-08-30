"""LLM tool loop: call model, execute tools, repeat until final response."""

from __future__ import annotations

import functools
import json
import hashlib
import os
import queue
import pathlib
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional, Tuple

import logging

from ouroboros.llm import LLMClient, normalize_reasoning_effort, add_usage
from ouroboros import task_pacing
from ouroboros.config import adaptive_quorum, get_acceptance_fence_wait_max_rounds, get_context_mode, get_light_model, get_review_enforcement, get_task_review_mode, resolve_effort
from ouroboros.review_cycles import REASON_REVIEW_CYCLES_EXHAUSTED
from ouroboros.outcomes import ACCEPTANCE_ACCEPTED, ACCEPTANCE_BYPASS_REASON_BY_RAIL, ACCEPTANCE_BYPASS_REASONS, ACCEPTANCE_DECISION_STATUSES, ACCEPTANCE_FINALIZED_UNACCEPTED, ACCEPTANCE_REVISION_REQUESTED, REASON_ACCEPTANCE_REVIEW_SKIPPED_DEADLINE_RESERVE, REASON_DELIVERY_CONTROL_DEGRADED, REASON_OWNER_REQUESTED_FINALIZATION, RESULT_INFRA_FAILED, extract_final_answer, latest_agent_defined_verification, latest_unreconciled_failed_verification, latest_unreconciled_masked_verification, reviewable_effect_projection, should_nudge_verification, turn_has_reviewable_effects
from ouroboros.observability import new_execution_id
from ouroboros.tool_policy import CAPABILITY_OMISSION_HEADER, format_capability_omissions, initial_tool_schemas, list_non_core_tools, swarm_router_turn
from ouroboros.tools.registry import ToolRegistry
from ouroboros.context import build_user_content
from ouroboros.context_budget import ContextReclaimRequest
from ouroboros.context_compaction import compact_tool_history_llm, context_reclaim_transcript_sha256
from ouroboros.deadline_utils import parse_deadline_ts, utc_now
from ouroboros.utils import estimate_tokens, sanitize_tool_result_for_log, truncate_review_artifact
from ouroboros.usage_accounting import (
    BudgetExceeded,
    PhysicalAttemptContext,
    PhysicalAttemptPreconditionFailed,
    last_physical_attempt_capture,
)
from ouroboros.task_finalization import (
    TERMINAL_ORIGIN_HOST_SALVAGE,
    TERMINAL_ORIGIN_MODEL_FINAL,
)
from supervisor.owner_stop import (
    _mark_owner_stop_control_drained,
    _narrow_round_deadline,
    _owner_stop_control_is_current,
    _owner_stop_window_elapsed,
)

from ouroboros.loop_tool_execution import (
    StatefulToolExecutor,
    handle_tool_calls,
    prune_reclaim_trace_refs,
    reclaim_negative_memo,
    reclaim_trace_refs,
)
from ouroboros.loop_llm_call import call_llm_with_retry, emit_llm_usage_event, forced_response_is_incomplete, forced_response_parts, provider_no_call_source
from ouroboros.delegate_hold import (
    close_hold as _delegate_hold_close,
    hold_step as _delegate_hold_step,
    latch_after_unknown as _delegate_hold_latch,
)
from ouroboros.loop_transport import (
    TransportWaitEpisode,
    end_episode_budget as _end_episode_budget,
    fallback_chain_allowed as _fallback_chain_allowed,
    finalize_now_transport_terminal as _finalize_now_transport_terminal,
    last_assistant_text as _last_assistant_text,
    provider_terminal_fallback_text as _provider_terminal_fallback_text,
    reconcile_transport_wait as _reconcile_transport_wait,
    task_deadline_epoch as _task_deadline_epoch,
    transport_wait_step as _transport_wait_step,
)
from ouroboros.pricing import estimate_cost_optional

# Backward-compat alias for source-inspecting/monkeypatched tests.
_call_llm_with_retry = call_llm_with_retry

log = logging.getLogger(__name__)

@dataclass
class DeliveryCandidate:
    """Loop-local complete answer retained across service/finalization rounds."""

    full_text: str
    content_sha256: str
    revision: int
    evidence_revision: int
    evidence_fingerprint: str
    acceptance_binding: Dict[str, Any]
    finalization_control: str = "candidate"
    repair_attempted: bool = False
    degraded: bool = False
    degraded_reason: str = ""
    model_text: str = ""

@dataclass
class _CompactionRoundContext:
    tools: ToolRegistry
    drive_root: Optional[pathlib.Path]
    drive_logs: pathlib.Path
    task_id: str
    round_idx: int
    event_queue: Optional[queue.Queue]
    emit_progress: Callable[[str], None]


def _handle_text_response(
    content: Optional[str],
    llm_trace: Dict[str, Any],
    accumulated_usage: Dict[str, Any],
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """Handle LLM response without tool calls (final response)."""
    safe_content = sanitize_tool_result_for_log(content or "")
    if safe_content.strip():
        llm_trace["reasoning_notes"].append(safe_content.strip())
        accumulated_usage["terminal_origin"] = TERMINAL_ORIGIN_MODEL_FINAL
    return safe_content, accumulated_usage, llm_trace


def _skill_names_touched_by_trace(llm_trace: Dict[str, Any]) -> List[str]:
    names: List[str] = []
    for call in llm_trace.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        tool = str(call.get("tool") or "")
        if tool not in {"write_file", "edit_text"}:
            continue
        args = call.get("args") if isinstance(call.get("args"), dict) else {}
        bucket = str(args.get("bucket") or "").strip().lower()
        skill_name = str(args.get("skill_name") or "").strip()
        if bucket in {"external", "clawhub", "ouroboroshub"} and skill_name:
            if skill_name not in names:
                names.append(skill_name)
            continue
        candidates = [str(args.get("path") or "")]
        for raw in candidates:
            norm = raw.replace("\\", "/").strip().lstrip("/")
            if norm.startswith("data/"):
                norm = norm[len("data/"):]
            parts = pathlib.PurePosixPath(norm).parts
            if len(parts) >= 3 and parts[0] == "skills" and parts[1] in {"external", "clawhub", "ouroboroshub", "native"}:
                name = parts[2]
                if name and name not in names:
                    names.append(name)
    return names


def _skill_finalization_message(drive_root: pathlib.Path, llm_trace: Dict[str, Any]) -> str:
    names = _skill_names_touched_by_trace(llm_trace)
    if not names:
        return ""
    try:
        from ouroboros.skill_loader import find_skill
        from ouroboros.skill_readiness import skill_readiness_for_execution
    except Exception:
        return ""
    blockers: List[str] = []
    for name in names:
        try:
            skill = find_skill(pathlib.Path(drive_root), name)
            if skill is None or not getattr(skill, "is_self_authored", False):
                continue
            readiness = skill_readiness_for_execution(pathlib.Path(drive_root), skill)
            ready = readiness.ready
        except Exception:
            continue
        if not ready:
            blockers.append(
                f"{skill.name}: status={skill.review.status!r}, "
                f"blockers={readiness.blockers}"
            )
    if not blockers:
        return ""
    return (
        "⚠️ SKILL_NOT_FINALIZED: You edited self-authored skill payloads but "
        "they are not ready yet. Call skill_review for each skill before "
        "declaring the task done. Current blockers: " + "; ".join(blockers)
    )


def _force_plan_decision(
    ctx: Any,
    _llm_trace: Dict[str, Any],
    *,
    hard_rail: str = "",
) -> Dict[str, Any]:
    """Project force-plan finalization from existing review + policy SSOTs.

    Body extracted to ``owner_hurry.force_plan_decision`` (the hurry latch makes
    the projection task-locally advisory for reviewed/open/unavailable states —
    §19.7.2 item 9); unlatched behavior is byte-identical.
    """
    from ouroboros.owner_hurry import force_plan_decision

    return force_plan_decision(
        ctx, _llm_trace, hard_rail=hard_rail,
        enforcement=get_review_enforcement(),
    )


def _force_plan_reminder(decision: Dict[str, Any]) -> str:
    from ouroboros.owner_hurry import plan_review_reminder

    return plan_review_reminder(decision)


def _force_plan_disclosure(
    ctx: Any,
    llm_trace: Dict[str, Any],
    *,
    forced_reason: str = "",
) -> str:
    # Normal finalization reuses the reducer projection that decided this
    # exact candidate. The trace copy is presentation-only, granting no
    # permission; forced rails recompute with their explicit rail input.
    from ouroboros.owner_hurry import plan_review_disclosure

    projected = llm_trace.get("force_plan_decision")
    decision = (
        projected
        if not forced_reason and isinstance(projected, dict)
        else _force_plan_decision(ctx, llm_trace, hard_rail=forced_reason)
    )
    return plan_review_disclosure(decision, forced_reason)


def _swarm_handoff_attempt(ctx: Any) -> Dict[str, Any]:
    attempt = getattr(ctx, "_swarm_handoff_attempt", None)
    return dict(attempt) if isinstance(attempt, dict) else {}


def _check_budget_limits(
    ctx: "_RoundLimitContext",
    budget_remaining_usd: Optional[float],
    cost_ceiling: Optional["task_pacing.CostCeiling"] = None,
) -> Optional[Tuple[str, Dict[str, Any], Dict[str, Any]]]:
    """Return a final-response tuple when budget limits require stopping.

    ``cost_ceiling`` is the typed in-task stop resolved ONCE at loop start
    (``task_pacing.resolve_cost_ceiling``). Only an ``active`` ceiling stops
    here; ``exhausted_soft_land`` fires at the round top. The deciding spend
    is the root subtree's ledger-accounted number when a root cap exists (the
    fence counts the TREE, not own calls); own cost is the DISCLOSED fallback
    and diagnostic. Unknown spend never becomes $0. The axes are INDEPENDENT
    (v6.91): ``budget_remaining_usd`` None only means no finite GLOBAL budget
    (TOTAL_BUDGET unset — the GAIA-shaped run) and must not silence a live
    per-task ROOT CAP; with neither, the ceiling resolves ``disabled`` and
    the whole cost axis stays silent, as before."""
    accumulated_usage = ctx.accumulated_usage
    raw_task_cost = accumulated_usage.get("cost")
    task_cost = float(raw_task_cost) if raw_task_cost is not None else None

    if budget_remaining_usd is not None and budget_remaining_usd <= 0:
        finish_reason = "🚫 Task rejected. Total budget exhausted. Please increase TOTAL_BUDGET in settings."
        accumulated_usage["execution_status"] = "failed"
        accumulated_usage["reason_code"] = "budget_exhausted"
        if ctx.round_idx <= 1:
            trace = ctx.llm_trace if isinstance(ctx.llm_trace, dict) else {}
            router_result = _forced_swarm_router_result(ctx, trace, "budget_exhausted")
            if router_result is not None:
                return router_result
            tool_ctx = getattr(getattr(ctx, "tools", None), "_ctx", None)
            suffix = (
                _force_plan_disclosure(tool_ctx, trace, forced_reason="budget_exhausted")
                if tool_ctx is not None else ""
            )
            # A forced sink like every other: a queued/headless root still
            # OWED a panel; returning without the record left `not_eligible /
            # run_count=0` — as if no panel was owed. Pure ledger write.
            _record_forced_finalization(
                ctx,
                trace,
                reason_code="budget_exhausted",
                source="host_budget_rejection_before_work",
                candidate=None,
            )
            return _compose_delivery_suffix(finish_reason, suffix), accumulated_usage, trace
        return _forced_final_answer(
            ctx,
            prompt=(
                "[BUDGET LIMIT] Total budget exhausted. Produce your best final answer NOW "
                "from the verified work so far; clearly mark anything unverified or "
                "incomplete. An honest best-effort result is the expected outcome here."
            ),
            fallback_text=finish_reason,
            reason_code="budget_exhausted",
        )
    # The pre-v6.91 per-task soft "[COST NOTE]" is gone: since v6.64.0 the
    # same settings key hard-fences the whole TREE at the ledger, so an
    # own-cost note keyed to it could never fire before the fence (proven
    # live: silent through two tree deaths); v6.56.0 milestones are the nudge.

    if cost_ceiling is None or cost_ceiling.state != task_pacing.COST_CEILING_ACTIVE:
        return None
    tree_info = _loop_tree_accounting(refresh=True, max_age_sec=_TREE_ACCOUNTING_MAX_STALE_SEC)
    tree_cost = tree_info.get("accounted_usd") if isinstance(tree_info, dict) else None
    deciding, spend_basis = task_pacing.resolve_deciding_spend(
        tree_cost_usd=tree_cost,
        task_cost_usd=task_cost,
        root_cap_usd=cost_ceiling.root_cap_usd,
    )
    ceiling_usd = cost_ceiling.ceiling_usd
    if deciding is not None and ceiling_usd is not None and deciding > ceiling_usd:
        if spend_basis == task_pacing.SPEND_BASIS_TREE:
            spent_text = (
                f"Task tree spent ${deciding:.3f} (ledger-accounted incl. in-flight holds, "
                f"subagents included; own calls ${task_cost:.3f})"
                if task_cost is not None
                else f"Task tree spent ${deciding:.3f} (ledger-accounted incl. in-flight holds)"
            )
        elif spend_basis == task_pacing.SPEND_BASIS_OWN_TREE_UNKNOWN:
            # Stopping on a disclosed lower bound beats not stopping at all, but
            # the substitution is stated, never silent (BIBLE P1).
            spent_text = (
                f"Task spent ${deciding:.3f} on its OWN calls (the tree-accounted total "
                "is unavailable right now, so subagent spend is not included — this is a "
                "lower bound)"
            )
        else:
            spent_text = f"Task spent ${deciding:.3f}"
        cap_text = (
            f"; the hard tree cap is ${cost_ceiling.root_cap_usd:.2f}"
            if cost_ceiling.root_cap_usd is not None else ""
        )
        finish_reason = (
            f"{spent_text}, over the in-task cost ceiling ${ceiling_usd:.2f}{cap_text}. "
            "Budget exhausted."
        )
        # The basis rides the usage record too, so a later reader can tell a
        # tree-decided stop from an own-cost stand-in without parsing prose.
        accumulated_usage["cost_stop_spend_basis"] = spend_basis
        return _forced_final_answer(
            ctx,
            prompt=(
                f"[BUDGET LIMIT] {finish_reason} Produce your best final answer now from "
                "the verified work so far; clearly mark anything unverified or incomplete. "
                "An honest best-effort result is the expected outcome here, not a failure."
            ),
            fallback_text=finish_reason,
            reason_code="budget_exhausted",
        )
    # The old round-gated "[INFO] ... Wrap up if possible" nudge is replaced by
    # the latched cost milestones in task_pacing (transport: _inject_round_checkpoints).

    return None


def _resolve_task_cost_ceiling(
    ctx: Any, budget_remaining_usd: Optional[float],
) -> "task_pacing.CostCeiling":
    """The typed in-task cost stop, resolved ONCE at loop start.

    The root cap comes from the bound usage scope — the SAME
    ``OUROBOROS_PER_TASK_COST_USD``-derived value the ledger fence enforces
    (agent.py wires it as ``UsageScope.root_limit_usd``), so the graceful stop
    and the fence can never disagree about the cap."""
    root_cap = None
    try:
        from ouroboros.usage_accounting import current_usage_scope

        scope = current_usage_scope()
        root_cap = getattr(scope, "root_limit_usd", None) if scope is not None else None
    except Exception:
        log.debug("Usage scope unavailable for cost ceiling resolution", exc_info=True)
    return task_pacing.resolve_cost_ceiling(
        budget_remaining_usd,
        task_pacing.resolve_budget_profile(ctx),
        root_cap_usd=root_cap,
    )


# Bounded staleness for the two DECIDING cost surfaces (ceiling check,
# milestone note): a round can block 900s in wait_tasks while children spend,
# and the pacing refresh covers only deadline-less tasks — such a round pays
# ONE real projection read, never per-round (e4a87344).
_TREE_ACCOUNTING_MAX_STALE_SEC = 120.0


def _loop_tree_accounting(
    *, refresh: bool, max_age_sec: float = 30.0,
) -> Optional[Dict[str, Any]]:
    """The root subtree's accounted spend for the CURRENT task's tree (nullable).

    Reads the reserve-time scope telemetry for free; ``refresh=True`` may do
    one real ledger projection read when the stash is older than
    ``max_age_sec``. Callers: loop start / 600s pacing note / 15-round
    checkpoint (cache-breaking, small max_age), plus the two DECIDING
    surfaces (ceiling check + milestone note) on the wider
    ``_TREE_ACCOUNTING_MAX_STALE_SEC`` bound — free while rounds are shorter
    (every dispatch refreshes the stash); never per-round unconditionally
    (e4a87344). Only under a root cap; None otherwise (unknown ≠ $0)."""
    try:
        from ouroboros.usage_accounting import (
            current_usage_scope,
            last_root_accounting,
            refresh_root_accounting,
        )

        scope = current_usage_scope()
        if scope is None or not scope.root_task_id or scope.root_limit_usd is None:
            return None
        if refresh:
            return refresh_root_accounting(scope.drive_root, scope.root_task_id, max_age_sec=max_age_sec)
        return last_root_accounting(scope.root_task_id)
    except Exception:
        log.debug("Tree accounting telemetry unavailable", exc_info=True)
        return None


def _soft_land_exhausted_ceiling(
    limit_ctx: "_RoundLimitContext",
    cost_ceiling: "task_pacing.CostCeiling",
) -> Optional[Tuple[str, Dict[str, Any], Dict[str, Any]]]:
    """Typed soft landing (v6.91): a root cap at or below the planning margin
    leaves no working room — enter the existing graceful best-effort wrap-up
    BEFORE spending a work round; never run uncapped (the pre-typed shape
    resolved this to the same None as "unlimited"). The ledger fence stays the
    untouched backstop. Returns the forced-final tuple, or None when the
    ceiling is not in the ``exhausted_soft_land`` state."""
    if cost_ceiling.state != task_pacing.COST_CEILING_EXHAUSTED_SOFT_LAND:
        return None
    cap_text = (
        f"${cost_ceiling.root_cap_usd:.2f}"
        if cost_ceiling.root_cap_usd is not None else "the per-task tree cap"
    )
    margin_text = (
        f"${cost_ceiling.planning_margin_usd:.2f}"
        if cost_ceiling.planning_margin_usd is not None else "the wrap-up planning margin"
    )
    soft_land_reason = (
        f"Per-task tree cap {cap_text} leaves no working room above the "
        f"wrap-up planning margin ({margin_text}). Budget exhausted."
    )
    return _forced_final_answer(
        limit_ctx,
        prompt=(
            f"[BUDGET LIMIT] {soft_land_reason} Produce your best final answer "
            "NOW from the verified work so far; clearly mark anything unverified "
            "or incomplete. An honest best-effort result is the expected outcome "
            "here, not a failure."
        ),
        fallback_text=soft_land_reason,
        reason_code="budget_exhausted",
    )


def _build_recent_tool_trace(messages: List[Dict[str, Any]], window: int = 15) -> str:
    """Build a compact recent-tool trace for the self-check prompt."""
    all_calls: List[str] = []
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                fn = tc.get("function", {})
                name = fn.get("name", "")
                args = fn.get("arguments", "")
                if isinstance(args, dict):
                    args = json.dumps(args, sort_keys=True)
                args_str = str(args)
                summary = f"{name}({args_str[:80]})" if len(args_str) > 80 else f"{name}({args_str})"
                all_calls.append(summary)
    recent = all_calls[-window:] if all_calls else []
    if not recent:
        return ""
    return "Recent tool calls (oldest first):\n" + "\n".join(f"  {i+1}. {c}" for i, c in enumerate(recent))


def _emit_checkpoint_event(
    event_queue: Optional[queue.Queue],
    task_id: str,
    drive_logs: Optional[pathlib.Path],
    data: Dict[str, Any],
) -> bool:
    """Emit a task_checkpoint via event queue or direct events.jsonl append."""
    from ouroboros.loop_llm_call import _emit_live_log
    payload = {"type": "task_checkpoint", "task_id": task_id, **data}
    if event_queue is not None:
        _emit_live_log(event_queue, payload)
    elif drive_logs:
        try:
            from ouroboros.utils import append_jsonl, utc_now_iso
            append_jsonl(drive_logs / "events.jsonl", {"ts": utc_now_iso(), **payload})
        except Exception:
            pass


def _extract_plain_text_from_content(content: Any) -> str:
    """Extract text from strings or multipart content for transcript sealing."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content) if content is not None else ""


def _append_or_merge_user_message(messages: List[Dict[str, Any]], text: str) -> None:
    """Append a user message without creating consecutive user turns."""
    _append_or_merge_user_content(messages, text)


def _evict_stale_image_blocks(messages: List[Dict[str, Any]], *, incoming: int = 0) -> None:
    """Keep only the newest MAX_LIVE_IMAGE_BLOCKS image blocks in the transcript.

    Single counter across ALL image sources (owner uploads, browser
    screenshots, transport injections). Evicted blocks become a text
    placeholder carrying the caption and re-view path: the dialogue HORIZON
    survives while the heavy payload drops (P1 — granularity varies, history
    never silently vanishes). ``incoming`` reserves room for imminent blocks.
    """
    from ouroboros.context_budget import MAX_LIVE_IMAGE_BLOCKS

    image_refs: List[tuple] = []  # (message_idx, block_idx)
    for m_idx, msg in enumerate(messages):
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for b_idx, block in enumerate(content):
            if isinstance(block, dict) and str(block.get("type") or "") in ("image_url", "image"):
                image_refs.append((m_idx, b_idx))
    excess = len(image_refs) + max(0, int(incoming)) - MAX_LIVE_IMAGE_BLOCKS
    if excess <= 0:
        return
    for m_idx, b_idx in image_refs[:excess]:
        content = messages[m_idx]["content"]
        block = content[b_idx]
        caption = str(block.get("_caption") or "").strip()
        source_path = str(block.get("_source_path") or "").strip()
        placeholder = "[image evicted"
        if caption:
            placeholder += f": {caption}"
        if source_path:
            # view_image re-views the local file natively. VLM tools are vision/local-media
            # tools, not _WEB_TOOLS; benchmark isolation withholds them by name.
            placeholder += f"; re-view: view_image path={source_path}"
        placeholder += "]"
        content[b_idx] = {"type": "text", "text": placeholder}


def _append_or_merge_user_content(messages: List[Dict[str, Any]], content: Any) -> None:
    """Append user content without flattening multipart blocks."""
    if isinstance(content, list):
        incoming_images = sum(
            1 for b in content
            if isinstance(b, dict) and str(b.get("type") or "") in ("image_url", "image")
        )
        if incoming_images:
            _evict_stale_image_blocks(messages, incoming=incoming_images)
    if messages and messages[-1].get("role") == "user":
        prior = messages[-1].get("content")
        if isinstance(content, list):
            new_blocks = list(content)
            if isinstance(prior, list):
                messages[-1] = {"role": "user", "content": list(prior) + new_blocks}
                return
            prior_text = prior if isinstance(prior, str) else str(prior or "")
            prefix_block = [{"type": "text", "text": prior_text.rstrip() + "\n\n---\n\n"}] if prior_text else []
            messages[-1] = {"role": "user", "content": prefix_block + new_blocks}
            return
        text = str(content or "")
        if isinstance(prior, list):
            messages[-1] = {
                "role": "user",
                "content": list(prior) + [{"type": "text", "text": "\n\n---\n\n" + text}],
            }
            return
        prior_text = prior if isinstance(prior, str) else str(prior or "")
        messages[-1] = {
            "role": "user",
            "content": (prior_text.rstrip() + "\n\n---\n\n" + text) if prior_text else text,
        }
        return
    messages.append({"role": "user", "content": content})


def _owner_marked_content(content: Any) -> Any:
    """Mark direct owner injections with the same priority tag as mailbox messages."""
    prefix = "[Message from my human]: "
    if isinstance(content, list):
        blocks = [dict(block) if isinstance(block, dict) else block for block in content]
        for block in blocks:
            if isinstance(block, dict) and str(block.get("type") or "") in {"text", "input_text"}:
                block["text"] = prefix + str(block.get("text") or "")
                return blocks
        return [{"type": "text", "text": prefix.rstrip()}] + blocks
    return prefix + str(content or "")


def _record_owner_directive(
    ctx: Any,
    *,
    source: str,
    content: Any,
    msg_id: str = "",
) -> None:
    """Retain the task-local owner corpus across transcript compaction.

    This is deliberately a provenance-preserving list, not a semantic decision
    parser: reviewers interpret the owner's verbatim words.  Structural control
    messages never call this helper.
    """
    if ctx is None:
        return
    if isinstance(content, str) and not content.strip():
        return
    if content in (None, [], {}):
        return
    directives = getattr(ctx, "_owner_directives", None)
    if not isinstance(directives, list):
        directives = []
        setattr(ctx, "_owner_directives", directives)
    stable_id = str(msg_id or "").strip()
    if stable_id and any(
        isinstance(row, dict) and str(row.get("msg_id") or "") == stable_id
        for row in directives
    ):
        return
    try:
        frozen_content = json.loads(json.dumps(content, ensure_ascii=False, default=str))
    except (TypeError, ValueError):
        frozen_content = str(content)
    row = {"source": str(source or "owner"), "content": frozen_content}
    if stable_id:
        row["msg_id"] = stable_id
    directives.append(row)


def _initialize_owner_directives(ctx: Any, messages: List[Dict[str, Any]]) -> None:
    """Capture the canonical initial user turn before system notices are added."""
    existing = getattr(ctx, "_owner_directives", None)
    if isinstance(existing, list) and existing:
        return
    for message in messages:
        if isinstance(message, dict) and str(message.get("role") or "") == "user":
            _record_owner_directive(
                ctx,
                source="initial_user",
                content=message.get("content"),
            )
            return


def _task_acceptance_eligible(
    mode: str,
    llm_trace: Dict[str, Any],
    is_direct_chat: bool,
    *,
    is_root_task: bool = True,
    is_ephemeral_turn: bool = False,
    task_contract: Optional[Dict[str, Any]] = None,
) -> tuple[bool, str]:
    """Return ``(host_should_review, trigger_reason)``.

    ``auto`` and ``required`` are effect-gated: the host enforces review
    when the turn produced reviewable effects (commit / deliverable / repo /
    workspace / skill write), declared a typed deliverable/criterion, or is
    not a direct-chat turn (queued / headless / scheduled). Read-only
    research and ordinary tool use in direct chat do not justify a
    three-reviewer panel; ephemeral routing turns are presentation/control
    decisions. ``off`` never reviews. Gates on typed contracts and observable
    runtime facts (P3 immune gate), never message content (P5)."""
    if mode == "off":
        return False, "off"
    if not is_root_task:
        return False, "skipped_child_advisory"
    if is_ephemeral_turn:
        return False, "skipped_ephemeral_control"
    if mode in {"auto", "required"}:
        prefix = "required" if mode == "required" else "auto"
        if turn_has_reviewable_effects(llm_trace):
            return True, f"{prefix}_effect"
        if not is_direct_chat:
            return True, f"{prefix}_nondirect"
        contract = task_contract if isinstance(task_contract, dict) else {}
        if (
            str(contract.get("expected_output") or "").strip()
            or bool(contract.get("acceptance_criteria"))
            or bool(contract.get("success_criteria"))
            or bool(contract.get("acceptance_claims"))
        ):
            return True, f"{prefix}_contract"
        return False, "skipped_conversation"
    return False, "skipped_unknown_mode"


def _begin_task_acceptance_fence(ctx: Any, task_id: str) -> tuple[bool, Any]:
    """Optional seam implemented by the supervisor under its queue lock."""
    admission_lock = getattr(ctx, "owner_message_admission_lock", None)
    admission_agent = getattr(ctx, "owner_message_admission_agent", None)
    if admission_lock is not None and admission_agent is not None:
        with admission_lock:
            ctx._task_acceptance_owner_generation = int(getattr(admission_agent, "_owner_message_generation", 0) or 0)
    existing = getattr(ctx, "_task_acceptance_fence_token", None)
    if existing is not None:
        inspect = getattr(ctx, "inspect_acceptance_fence", None)
        if callable(inspect):
            try:
                refreshed = inspect(token=str(existing))
                ctx._task_acceptance_queue_descendants = (
                    list(refreshed.get("queue_descendants") or [])
                    if isinstance(refreshed, dict) else []
                )
                if isinstance(refreshed, dict):
                    ctx._task_acceptance_fence_generation = int(
                        refreshed.get("owner_message_generation") or 0
                    )
            except Exception:
                log.debug("Queue-owned acceptance fence inspection failed", exc_info=True)
                return False, existing
        return True, existing
    callback = getattr(ctx, "begin_acceptance_fence", None)
    if not callable(callback):
        return True, None  # one-minor/direct-context compatibility
    try:
        meta = getattr(ctx, "task_metadata", {})
        meta = meta if isinstance(meta, dict) else {}
        response = callback(
            root_task_id=str(
                meta.get("root_task_id") or getattr(ctx, "root_task_id", "") or task_id
            ),
            task_id=str(task_id),
        )
    except Exception:
        log.debug("Queue-owned acceptance fence begin failed", exc_info=True)
        return False, None
    if isinstance(response, dict):
        token = response.get("token")
        ctx._task_acceptance_queue_descendants = list(response.get("queue_descendants") or [])
        ctx._task_acceptance_fence_generation = int(
            response.get("owner_message_generation") or 0
        )
    else:
        token = response
        ctx._task_acceptance_queue_descendants = []
        ctx._task_acceptance_fence_generation = None
    if token in (None, False, ""):
        return False, None
    ctx._task_acceptance_fence_token = token
    return True, token


def _end_task_acceptance_fence(
    ctx: Any, *, outcome: str, admission_locked: bool = False,
) -> bool:
    token = getattr(ctx, "_task_acceptance_fence_token", None)
    if token is None and str(outcome) == "revision":
        token = getattr(ctx, "_task_acceptance_sealed_fence_token", None)
    callback = getattr(ctx, "end_acceptance_fence", None)
    admission_lock = getattr(ctx, "owner_message_admission_lock", None)
    admission_agent = getattr(ctx, "owner_message_admission_agent", None)
    acquired = False
    try:
        if admission_lock is not None and admission_agent is not None and not admission_locked:
            admission_lock.acquire()
            acquired = True
        expected_owner_generation = getattr(ctx, "_task_acceptance_owner_generation", None)
        direct_generation_mismatch = bool(
            expected_owner_generation is not None
            and admission_agent is not None
            and int(getattr(admission_agent, "_owner_message_generation", 0) or 0)
            != int(expected_owner_generation)
        )
        effective_outcome = "revision" if direct_generation_mismatch else str(outcome)
        if token is None or not callable(callback):
            ctx._task_acceptance_fence_generation_mismatch = direct_generation_mismatch
            return True
        expected_queue_generation = getattr(ctx, "_task_acceptance_fence_generation", None)
        if expected_queue_generation is None:
            response = callback(token=token, outcome=effective_outcome)
        else:
            response = callback(
                token=token,
                outcome=effective_outcome,
                expected_generation=int(expected_queue_generation),
            )
    except Exception:
        log.debug("Queue-owned acceptance fence transition failed", exc_info=True)
        return False
    finally:
        if acquired:
            admission_lock.release()
    if isinstance(response, dict) and not bool(response.get("ok", True)):
        return False
    status = str((response or {}).get("status") or "") if isinstance(response, dict) else ""
    generation_mismatch = bool(
        direct_generation_mismatch
        or (isinstance(response, dict) and response.get("generation_mismatch"))
    )
    ctx._task_acceptance_fence_generation_mismatch = generation_mismatch
    ctx._task_acceptance_fence_token = None
    ctx._task_acceptance_fence_generation = None
    ctx._task_acceptance_queue_descendants = []
    if status == "sealed" or (not status and effective_outcome != "revision"):
        ctx._task_acceptance_sealed_fence_token = token
    else:
        ctx._task_acceptance_sealed_fence_token = None
    return True


def _supersede_delivery_acceptance_binding(
    tools: ToolRegistry,
    llm_trace: Dict[str, Any],
    candidate: DeliveryCandidate,
    *,
    reason: str,
) -> bool:
    """Invalidate the exact host verdict bound to a changed delivery candidate.

    The run remains in ``review_runs`` as audit evidence, but neither the
    candidate nor ``review_decision`` may keep pointing at it after answer text
    or answer-invalidating evidence changes.  Negative superseded verdicts stay
    available to the outcome reducer's fail-closed path.
    """

    decision = (
        dict(llm_trace.get("review_decision") or {})
        if isinstance(llm_trace.get("review_decision"), dict)
        else {}
    )
    candidate_binding = (
        dict(candidate.acceptance_binding or {})
        if isinstance(candidate.acceptance_binding, dict)
        else {}
    )
    exact_bindings = {
        (str(panel_id), str(binding_hash))
        for panel_id, binding_hash in (
            (candidate_binding.get("panel_id"), candidate_binding.get("binding_hash")),
            (decision.get("panel_id"), decision.get("binding_hash")),
        )
        if panel_id and binding_hash
    }
    run_record: Optional[Dict[str, Any]] = None
    if exact_bindings:
        for run in reversed(llm_trace.get("review_runs") or []):
            if not isinstance(run, dict):
                continue
            if run.get("authority") != "host_root" or run.get("superseded_by_revision"):
                continue
            run_candidate = str(
                run.get("candidate_hash") or run.get("candidate_sha256") or ""
            )
            run_binding = (
                str(run.get("panel_id") or ""),
                str(run.get("binding_hash") or ""),
            )
            if run_candidate != candidate.content_sha256 or run_binding not in exact_bindings:
                continue
            run_record = run
            break

    decision_was_bound = bool(decision.get("panel_id") and decision.get("binding_hash"))
    candidate_was_bound = bool(exact_bindings)
    if run_record is None and not decision_was_bound and not candidate_was_bound:
        return False
    if run_record is not None:
        run_record["superseded_by_revision"] = True
        run_record["superseded_reason"] = reason
        run_record["enforcement_impact"] = "requires_revision"

    for key in ("panel_id", "binding_hash", "panel_reused"):
        decision.pop(key, None)
    decision.update({
        "eligibility": "pending_delivery_acceptance",
        "trigger": reason,
    })
    llm_trace["review_decision"] = decision
    candidate_binding.update({
        "acceptance_status": "unaccepted",
        "authoritative": False,
        "panel_id": "",
        "binding_hash": "",
    })
    candidate_binding.pop("review_evidence_revision", None)
    candidate.acceptance_binding = candidate_binding
    tools._ctx._task_acceptance_reviewed = False
    llm_trace.pop("root_phase_checkpoint", None)
    _set_acceptance_decision(llm_trace, {
        "status": ACCEPTANCE_REVISION_REQUESTED,
        "reason": "delivery_binding_superseded",
        "source": "delivery_candidate_binding",
        "rationale": (
            "The delivery candidate or its evidence binding changed after host "
            "acceptance; the prior panel is retained only as superseded audit evidence."
        ),
    })
    return True


def _supersede_task_acceptance_for_owner_followup(
    ctx: Any,
    llm_trace: Dict[str, Any],
    *,
    admission_locked: bool = False,
) -> bool:
    """Invalidate a paid verdict whose immutable evidence predates an owner follow-up."""
    released = _end_task_acceptance_fence(
        ctx, outcome="revision", admission_locked=admission_locked,
    )
    for run in reversed(llm_trace.get("review_runs") or []):
        if (
            isinstance(run, dict)
            and run.get("authority") == "host_root"
            and not run.get("superseded_by_revision")
        ):
            run["superseded_by_revision"] = True
            run["superseded_reason"] = "owner_followup_after_acceptance_evidence"
            run["enforcement_impact"] = "requires_revision"
            break
    ctx._task_acceptance_reviewed = False
    ctx._task_acceptance_fence_generation_mismatch = False
    llm_trace.pop("root_phase_checkpoint", None)
    llm_trace["review_decision"] = {
        "eligibility": "pending_owner_followup",
        "trigger": "owner_followup_after_acceptance",
    }
    _set_acceptance_decision(llm_trace, {
        "status": ACCEPTANCE_REVISION_REQUESTED,
        "reason": "owner_followup",
        "source": "owner_followup",
        "rationale": "The owner added a directive after acceptance evidence was frozen; re-review is required.",
    })
    return released


def _task_acceptance_owner_generation_changed(ctx: Any) -> bool:
    """Check direct and queue-owned owner generations without closing the fence."""

    expected_owner = getattr(ctx, "_task_acceptance_owner_generation", None)
    admission_agent = getattr(ctx, "owner_message_admission_agent", None)
    if (
        expected_owner is not None
        and admission_agent is not None
        and int(getattr(admission_agent, "_owner_message_generation", 0) or 0)
        != int(expected_owner)
    ):
        return True
    expected_queue = getattr(ctx, "_task_acceptance_fence_generation", None)
    token = getattr(ctx, "_task_acceptance_fence_token", None)
    inspect = getattr(ctx, "inspect_acceptance_fence", None)
    if expected_queue is None or token is None or not callable(inspect):
        return False
    try:
        state = inspect(token=str(token))
        return bool(
            isinstance(state, dict)
            and int(state.get("owner_message_generation") or 0) != int(expected_queue)
        )
    except Exception:
        return True


def _supersede_task_acceptance_for_evidence_change(
    ctx: Any,
    llm_trace: Dict[str, Any],
    run_record: Optional[Dict[str, Any]],
    reason: str,
    messages: List[Dict[str, Any]],
    emit_progress: Callable[[str], None],
) -> None:
    """Invalidate an acceptance boundary when frozen evidence changes before delivery."""

    if isinstance(run_record, dict):
        run_record["superseded_by_revision"] = True
        run_record["superseded_reason"] = reason
        run_record["enforcement_impact"] = "requires_revision"
    _end_task_acceptance_fence(ctx, outcome="revision")
    ctx._task_acceptance_reviewed = False
    ctx._task_acceptance_fence_generation_mismatch = False
    llm_trace.pop("root_phase_checkpoint", None)
    llm_trace["review_decision"] = {
        "eligibility": "pending_evidence_refresh",
        "trigger": reason,
    }
    _set_acceptance_decision(llm_trace, {
        "status": ACCEPTANCE_REVISION_REQUESTED,
        "reason": "evidence_refresh",
        "source": "host_acceptance_evidence_refresh",
        "rationale": (
            "Task or child evidence changed after acceptance evidence was frozen; "
            "the prior boundary was superseded before it could authorize delivery."
        ),
    })
    _append_or_merge_user_message(
        messages,
        "[TASK ACCEPTANCE REFRESH] Task or child evidence changed after acceptance "
        "evidence was frozen. Re-read the latest evidence and produce one complete "
        "replacement answer before the next host acceptance review.",
    )
    emit_progress(
        "Task acceptance review superseded: task or child evidence changed before delivery."
    )


def _task_acceptance_subtree_snapshot(
    ctx: Any, drive_root: Optional[pathlib.Path], task_id: str,
) -> tuple[bool, List[Dict[str, Any]]]:
    """Return recursive terminal/quiescent state using the existing task SSOT."""
    if drive_root is None:
        try:
            drive_root = pathlib.Path(getattr(ctx, "drive_root"))
        except (TypeError, OSError, ValueError):
            return False, []
    try:
        from ouroboros.task_status import SETTLED_STATUSES, find_child_tasks
        from ouroboros.depth_evidence import task_depth_provenance
        from ouroboros.tools.join_ledger import _child_result_sha256

        meta = getattr(ctx, "task_metadata", {})
        meta = meta if isinstance(meta, dict) else {}
        root_id = str(meta.get("root_task_id") or getattr(ctx, "root_task_id", "") or task_id)
        status_root = pathlib.Path(str(
            meta.get("budget_drive_root")
            or getattr(ctx, "budget_drive_root", "")
            or drive_root
        ))
        rows = find_child_tasks(
            status_root,
            parent_task_id=str(task_id),
            root_task_id=root_id,
            exclude_task_id=str(task_id),
            scope="subtree",
        )
        compact = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            row_task_id = str(row.get("task_id") or row.get("id") or "")
            status = str(row.get("status") or "unknown")
            projected = {
                "task_id": row_task_id,
                "parent_task_id": str(row.get("parent_task_id") or ""),
                "status": status,
                "artifact_status": str(row.get("artifact_status") or ""),
            }
            if depth_provenance := task_depth_provenance(row):
                projected["depth_provenance"] = depth_provenance
            if status in SETTLED_STATUSES:
                projected["child_result_sha256"] = _child_result_sha256(row)
            compact.append(projected)
        # Acceptance needs true quiescence: SETTLED statuses only. A child
        # with a pending durable cancel intent stays non-quiescent until
        # custody settles it (guaranteed by the cancel-intent watchdog).
        queue_rows = [
            {
                "task_id": str(row.get("task_id") or ""),
                "parent_task_id": "",
                "status": str(row.get("status") or "running"),
                "artifact_status": "",
                "source": "supervisor_queue",
            }
            for row in (getattr(ctx, "_task_acceptance_queue_descendants", None) or [])
            if isinstance(row, dict)
        ]
        return (
            not queue_rows and all(row["status"] in SETTLED_STATUSES for row in compact),
            compact + queue_rows,
        )
    except Exception:
        log.debug("Unable to establish task-acceptance subtree quiescence", exc_info=True)
        return False, []


def _mark_root_acceptance_checkpoint(
    ctx: Any, llm_trace: Dict[str, Any], *, status: str, pass_index: int = 0,
) -> None:
    """Minimal in-result phase checkpoint; no parallel acceptance journal."""
    from ouroboros.task_results import resolve_task_lineage

    meta = getattr(ctx, "task_metadata", {})
    meta = meta if isinstance(meta, dict) else {}
    task_id = str(getattr(ctx, "task_id", "") or "")
    lineage = resolve_task_lineage(
        task_id,
        metadata=meta,
        root_task_id=getattr(ctx, "root_task_id", None),
        parent_task_id=getattr(ctx, "parent_task_id", None),
        delegation_role=getattr(ctx, "delegation_role", None),
        original_task_id=getattr(ctx, "original_task_id", None),
        timeout_retry_from=getattr(ctx, "timeout_retry_from", None),
    )
    if not lineage["is_root_task"]:
        return
    llm_trace["root_phase_checkpoint"] = {
        "phase": "task_acceptance",
        "status": str(status),
        "pass_index": max(0, int(pass_index)),
        "post_task_synthesis": "pending_once",
    }


def _latch_final_answer_marker(
    llm_trace: Dict[str, Any],
    content: str | None,
    current_tool_calls: list | None = None,
) -> None:
    """Anytime capture for explicit FINAL ANSWER markers.

    Marker-only: do not mine prose. The tool-call count stamp preserves the
    existing stale-answer invariant: later grounding invalidates this fallback
    unless the model emits a newer marker.
    """
    # Opt-in CANDIDATES latch (v6.54.4): an explicit block ("CANDIDATES:" on
    # its own line, "- " items) latches candidate interpretations beside the
    # final answer so the acceptance reviewer can adjudicate ambiguity.
    # Marker-only, like FINAL ANSWER — no prose mining; no block = unchanged.
    text = content or ""
    try:
        lines = text.splitlines()
        marker_idx = next(
            (i for i, line in enumerate(lines) if line.strip() == "CANDIDATES:"),
            None,
        )
        if marker_idx is not None:
            # Marker-only, like FINAL ANSWER (adversarial r2 #4): the block
            # is the "- " items IMMEDIATELY after the marker line; the first
            # non-item line ends it. No substring or distant-bullet harvest.
            candidates: list = []
            for line in lines[marker_idx + 1:]:
                if line.strip().startswith("- "):
                    candidates.append(line.strip()[2:].strip()[:300])
                else:
                    break
            if candidates:
                llm_trace["candidate_answers"] = candidates[:8]
    except Exception:
        pass
    answer = extract_final_answer(text)
    if not answer:
        return
    llm_trace["best_valid_final_answer"] = answer
    del current_tool_calls
    llm_trace["best_valid_final_answer_tools"] = len(llm_trace.get("tool_calls") or [])


def _server_web_allowed_by_task(ctx: Any) -> bool:
    contract = getattr(ctx, "task_contract", {}) if isinstance(getattr(ctx, "task_contract", {}), dict) else {}
    resources = contract.get("allowed_resources") if isinstance(contract.get("allowed_resources"), dict) else {}
    forbidden_names = {"web", "allow_web", "network", "allow_network", "internet", "external_network"}
    return not any(resources.get(name) is False for name in forbidden_names)


ACCEPTANCE_REASON_UNSPECIFIED = "unspecified"
# Closed set of typed acceptance reasons (v6.78.0): every value is a fact
# the host already computed or the exit branch's name; none derives from model
# prose. `unspecified` is only the fail-closed fallback for a forgotten reason.
ACCEPTANCE_DECISION_REASONS = (
    "clean_pass",
    "clean_pass_obligations_closed",
    "no_actionable_changes",
    "delivery_binding_superseded",
    "owner_followup",
    "evidence_refresh",
    "improvement_capsule",
    "dialogue_terminal",
    "open_obligations",
    "capsule_spent",
    "improvement_window_closed",
    "reviewer_fail_no_capsule",
    "review_degraded",
    "fence_reopen_failed",
    "infra_failure",
    # Bounded fence wait exhausted (CyberGym full1507): the queue-owned
    # admission fence stayed unavailable past the configured round cap, so the
    # task terminalized as infra_failed instead of spinning paid rounds.
    "acceptance_fence_unavailable",
    # Owner Q2A: the forced children_unabsorbed rail runs the panel but cannot
    # grant a requested improvement pass; the dangling revision terminalizes.
    "revision_unavailable_on_forced_rail",
    REASON_ACCEPTANCE_REVIEW_SKIPPED_DEADLINE_RESERVE,
    # Forced-rail acceptance bypass (closed set, outcomes.py SSOT): stamped by
    # `_record_forced_acceptance_bypass` when the panel was owed but a rail fired.
    *sorted(ACCEPTANCE_BYPASS_REASONS),
    ACCEPTANCE_REASON_UNSPECIFIED,
)


def _set_acceptance_decision(llm_trace: Dict[str, Any], decision: Dict[str, Any]) -> None:
    """The ONLY merge point for the host acceptance decision (v6.78.0, owner Q23).

    Every host exit funnels here and leaves in one of the three canonical
    owner-facing states (``ACCEPTANCE_DECISION_STATUSES``) plus a typed
    ``reason`` naming WHICH exit. A status outside the trio fails closed to
    ``finalized_unaccepted`` with its raw token surviving as ``reason`` — no
    fourth state, no lost token. The agent's stance (``agent_disposition``/
    ``agent_rationale``) carries forward, never overwritten (after P4.1 the
    agent writes no status at all)."""
    previous = llm_trace.get("acceptance_decision") if isinstance(llm_trace.get("acceptance_decision"), dict) else {}
    merged = dict(decision)
    status = str(merged.get("status") or "")
    reason = str(merged.get("reason") or "")
    if status not in ACCEPTANCE_DECISION_STATUSES:
        merged["status"] = ACCEPTANCE_FINALIZED_UNACCEPTED
        reason = reason or status or ACCEPTANCE_REASON_UNSPECIFIED
    merged["reason"] = reason
    for key in ("agent_disposition", "agent_rationale"):
        if previous.get(key) and not merged.get(key):
            merged[key] = previous.get(key)
    llm_trace["acceptance_decision"] = merged


def _collect_acceptance_obligations(llm_trace: Dict[str, Any], result: Any) -> None:
    """Typed PER-TASK obligations from critical contributing findings (v6.54.4).

    Required+blocking path only. Each critical finding WITH a concrete
    recommendation becomes one open obligation in llm_trace (never the durable
    commit review_state — a separate SSOT). Clean finalization asks for an
    agent disposition per obligation (v6.54.0); time/pass gates and the
    forced-finalization escape hatches bound the loop, so a deadline never
    hangs here. v6.60.0 widening (S1-lite, owner quiz 18b): when the AGGREGATE
    verdict itself is failing — signal FAIL, or worst tier
    blocked_with_evidence — contributing reviewers' HIGH findings with a
    concrete recommendation also become obligations (the PB incident). On a
    PASS (incl. with-dissent) the bar stays critical-only, so the blocking
    lane cannot creep into taxing clean runs with hygiene items."""
    import hashlib

    from ouroboros.review_substrate import _contributing_actors, aggregate_outcome_tier

    contributing = {str(a.get("slot_id", "")) for a in _contributing_actors(result)}
    obligations = llm_trace.setdefault("acceptance_obligations", [])
    by_id = {str(o.get("id")): o for o in obligations if isinstance(o, dict)}
    # No contributing actors (all parse-degraded / no quorum) => no
    # authoritative verdict: manufacture NO blocking obligations — else one
    # parse-degraded slot's critical finding would gate finalization, the
    # class the capsule refuses (r1); obligations ride CONTRIBUTING slots.
    if not contributing:
        return
    _agg_failing = (
        str(getattr(result, "aggregate_signal", "") or "").upper() == "FAIL"
        or aggregate_outcome_tier(result) == "blocked_with_evidence"
    )
    _obligation_severities = {"critical", "high"} if _agg_failing else {"critical"}
    # Ids already created or reopened by THIS panel pass: slots of one panel
    # routinely raise the same finding (typed re_raise copies the catalog
    # id); without it the second slot's dupe would falsely bump
    # reopened_count and overwrite reviewer_rebuttal_response (fable r1 #1).
    touched_this_pass: set[str] = set()
    for finding in (getattr(result, "parsed_findings", None) or []):
        if not isinstance(finding, dict):
            continue
        if str(finding.get("severity") or "").strip().lower() not in _obligation_severities:
            continue
        if str(finding.get("slot_id", "")) not in contributing:
            continue
        recommendation = " ".join(str(finding.get("recommendation") or "").split()).strip()
        if not recommendation:
            continue
        item = str(finding.get("item") or "finding").strip()
        # v6.74.0 (A3): obligation identity is reviewer-authored. A re_raise
        # MUST name an existing catalog id (the host validates existence
        # only); a missing/unknown id fails closed to `new` with a disclosed
        # note — a reworded re-raise cannot mint a fresh hash id.
        kind = str(finding.get("disposition_kind") or "").strip().lower()
        claimed_id = str(finding.get("obligation_id") or "").strip()
        unbound_note = ""
        if kind == "re_raise":
            row = by_id.get(claimed_id)
            if row is not None:
                if claimed_id not in touched_this_pass:
                    touched_this_pass.add(claimed_id)
                    _reopen_obligation_row(row, finding)
                continue
            unbound_note = f"re_raise_unbound:{claimed_id or 'missing_id'}"
        oid = "ob-" + hashlib.sha256(
            json.dumps([item, recommendation], ensure_ascii=False).encode("utf-8")
        ).hexdigest()[:12]
        if oid in by_id:
            # Reviewer-authored identity (triad r2, sol): only an UNTYPED
            # legacy finding may reopen via byte-identical text (v6.71.1
            # compat). A typed "new"/unbound "re_raise" matching a settled row
            # must NOT resurrect the settled rebuttal — sloppiness DISCLOSED.
            row = by_id[oid]
            if not kind and oid not in touched_this_pass:
                touched_this_pass.add(oid)
                _reopen_obligation_row(row, finding)
            elif kind:
                notes = row.setdefault("notes", [])
                note = unbound_note or f"typed_new_matched_existing:{oid}"
                if note not in notes:
                    notes.append(note)
            continue
        row = {
            "id": oid,
            "item": item,
            "recommendation": recommendation,
            "status": "open",
            "disposition": "",
            "disposition_reason": "",
        }
        if unbound_note:
            row["notes"] = [unbound_note]
        by_id[oid] = row
        touched_this_pass.add(oid)
        obligations.append(row)


def _reopen_obligation_row(row: Dict[str, Any], finding: Dict[str, Any]) -> None:
    """Reopen a re-raised obligation WITHOUT wiping the agent's argument (A3).

    The prior disposition/reason survive as ``previous_disposition`` /
    ``previous_reason`` and ``reopened_count`` increments, so the agent can see
    its rebuttal was overruled (previously indistinguishable from a fresh
    finding) and the next reviewer receives the prior argument to adjudicate.
    The reviewer's stated reason for maintaining the finding rides along."""
    if str(row.get("disposition") or "").strip() or str(row.get("status") or "") == "agent_disposed":
        row["previous_disposition"] = str(
            row.get("disposition") or row.get("status") or ""
        )
        row["previous_reason"] = str(row.get("disposition_reason") or "")
    row["reopened_count"] = int(row.get("reopened_count") or 0) + 1
    row["disposition"] = ""
    row["disposition_reason"] = ""
    row["status"] = "open"
    reviewer_response = " ".join(str(finding.get("evidence") or "").split()).strip()
    if reviewer_response:
        row["reviewer_rebuttal_response"] = truncate_review_artifact(
            reviewer_response, limit=600,
        )


def _open_acceptance_obligations(llm_trace: Dict[str, Any]) -> List[Dict[str, Any]]:
    # An agent-filed disposition (status="agent_disposed") is a
    # CLAIM/rebuttal, not a settlement: the row is pending until a host panel
    # adjudicates (PASS settles; re-raise reopens). SSOT: review_evidence.
    from ouroboros.review_evidence import obligation_is_pending

    return [
        o for o in (llm_trace.get("acceptance_obligations") or [])
        if obligation_is_pending(o)
    ]


def _dispose_obligations_on_clean_pass(
    llm_trace: Dict[str, Any],
    result: Any,
    open_obligations: List[Dict[str, Any]],
    dissent_noted: bool,
) -> bool:
    """If the re-review is a CLEAN PASS (aggregate PASS and not degraded), close
    the open obligations as disposed_by_re_review and record the accepted verdict;
    return True. A DEGRADED/no-quorum run proves nothing → returns False, leaving
    the honest best-effort labeling to the caller."""
    if not open_obligations:
        return False
    from ouroboros.review_substrate import task_acceptance_is_clean

    if not task_acceptance_is_clean(result):
        return False
    for ob in open_obligations:
        if str(ob.get("status") or "") == "agent_disposed":
            # The clean panel ACCEPTED the agent's filed disposition (a
            # rebuttal it chose not to re-raise): keep that disposition/reason
            # as provenance, record the host settlement distinctly (r6) —
            # never rewrite a rejected rebuttal into "addressed by revision".
            ob["status"] = "disposed_rebuttal_accepted"
            continue
        ob["disposition"] = "addressed"
        ob["disposition_reason"] = "resolved by revision: the clean re-review returned no findings"
        ob["status"] = "disposed_by_re_review"
    _set_acceptance_decision(llm_trace, {
        "status": ACCEPTANCE_ACCEPTED,
        "reason": "clean_pass_obligations_closed",
        "source": "task_acceptance_review",
        "rationale": "Clean PASS re-review; open obligations closed by the revision (dissent, if any, stays advisory).",
        "dissent_noted": dissent_noted,
    })
    return True


def _format_obligations_clause(open_obligations: List[Dict[str, Any]]) -> str:
    # v6.74.0 (A4): disagreement is recorded ONLY via
    # obligation_dispositions — the old "or address them directly" prose read
    # as a third channel; fixing the work just makes the next panel clean.
    if not open_obligations:
        return ""
    lines = [
        "",
        "OPEN OBLIGATIONS (blocking review policy). Either FIX the work so the next review "
        "panel finds it clean, or record your disagreement via the task_acceptance_review "
        "tool's obligation_dispositions (addressed / rejected / deferred + reason) — "
        "dispositions are the ONLY channel the reviewer adjudicates:",
    ]
    for o in open_obligations[:5]:
        line = f"  {o.get('id')}: {o.get('item')} — {o.get('recommendation')}"
        reopened = int(o.get("reopened_count") or 0)
        if reopened > 0:
            line += f" [re-raised ×{reopened}"
            if str(o.get("previous_disposition") or "").strip():
                line += (
                    f"; your '{o.get('previous_disposition')}' rebuttal was overruled"
                )
                response = str(o.get("reviewer_rebuttal_response") or "").strip()
                if response:
                    line += f" — reviewer: {response}"
            line += "]"
        lines.append(line)
    if len(open_obligations) > 5:
        lines.append(f"  (+{len(open_obligations) - 5} more in the task record)")
    return "\n".join(lines)


# The host-forced acceptance-review checklist (module constant for the size
# gate). v6.60.0 adds the explicit SCOPE-CUT question — a silent/unjustified
# narrowing is a high-severity finding, which under blocking enforcement
# becomes a typed obligation.
_ACCEPTANCE_REVIEW_CHECKLIST = (
    "Check whether the claimed result follows from the tool trace, "
    "whether errors/timeouts/artifacts were handled honestly, and "
    "whether each explicit original requirement was verified through "
    "the interface/surface the task itself names (not a weaker "
    "surrogate self-test), and "
    "whether the final response should be changed before release. "
    "SCOPE CUTS (v6.60.0): did the agent knowingly narrow the task's scope "
    "(dropped/limited requirements, simplified formats, skipped inputs)? "
    "A DISCLOSED, task-justified cut is honest best_effort; an unjustified "
    "or silent cut is a finding — name it with severity high and a concrete "
    "recommendation (under blocking enforcement it becomes an obligation). "
    "Classify the deliverable tier (solved / best_effort / "
    "blocked_with_evidence) and name the single highest-value change "
    "that would move it one tier up. If the task asks for a specific "
    "value or short answer, check the FINAL ANSWER line matches the "
    "requested format exactly."
)


@dataclass
class _TaskAcceptanceContext:
    tools: ToolRegistry
    content: str
    task_id: str
    task_type: str
    llm_trace: Dict[str, Any]
    drive_root: Optional[pathlib.Path]
    messages: List[Dict[str, Any]]
    emit_progress: Callable[[str], None]
    mode: str
    subtree_statuses: List[Dict[str, Any]]
    budget_profile: Any
    passes_done: int
    evidence: Dict[str, Any] = field(default_factory=dict)
    review_binding: Dict[str, Any] = field(default_factory=dict)
    # One pre-rendered rails line (money/time/rounds/passes headroom) built
    # in loop.py from each real source, fed into the improvement capsule
    # (v6.74.0 A1, owner Q6); the capsule builder never gains ctx.
    rails_line: str = ""


def _acceptance_dialogue_quorum(result: Any) -> int:
    """The quorum the panel itself used (policy min_successful_slots), with the
    adaptive_quorum fallback for records that lost the policy dict."""
    request = getattr(result, "request", None)
    policy = request.get("policy") if isinstance(request, dict) else {}
    try:
        quorum = int((policy or {}).get("min_successful_slots") or 0)
    except (TypeError, ValueError):
        quorum = 0
    if quorum <= 0:
        quorum = adaptive_quorum(len(getattr(result, "actors", None) or []) or 1)
    return max(1, quorum)


def _attach_dialogue_to_host_run(llm_trace: Dict[str, Any], dialogue: Dict[str, Any]) -> None:
    """Persist the dialogue-status vote distribution on the authoritative host
    run record so the review projection carries it for audit (A5)."""
    for run in reversed(llm_trace.get("review_runs") or []):
        if (
            isinstance(run, dict)
            and run.get("authority") == "host_root"
            and not run.get("superseded_by_revision")
        ):
            run["dialogue"] = dict(dialogue)
            return


def _mark_agent_acceptance_runs_advisory(llm_trace: Dict[str, Any]) -> None:
    """Keep agent-invoked reviews as evidence without granting root authority."""
    for run in llm_trace.get("review_runs") or []:
        if not isinstance(run, dict) or run.get("authority") == "host_root":
            continue
        request = run.get("request") if isinstance(run.get("request"), dict) else {}
        if str(request.get("surface") or "") != "task_acceptance":
            continue
        run["authority"] = "agent_advisory"
        # Compatibility with the objective reducer: non-authoritative historical
        # runs stay fully auditable but cannot worst-case the host/root verdict.
        run["superseded_by_revision"] = True
        run["superseded_reason"] = "non_authoritative_agent_acceptance_review"


def _latest_agent_acceptance_evidence(llm_trace: Dict[str, Any]) -> Dict[str, Any]:
    """Return the latest validated root self-call packet for host review.

    ``process_tool_results`` records only typed, non-authoritative root
    deferrals here.  The payload is already bounded and redacted by the shared
    evidence builder; the host builder will redact it again while assigning the
    explicit ``agent_supplied`` provenance.
    """
    for call in reversed(llm_trace.get("acceptance_evidence_calls") or []):
        if not isinstance(call, dict):
            continue
        if (
            str(call.get("status") or "") != "deferred_to_host_acceptance"
            or call.get("authoritative") is not False
        ):
            continue
        evidence = call.get("agent_supplied")
        if isinstance(evidence, dict):
            return dict(evidence)
    return {}


def _build_host_acceptance_evidence(ctx: _TaskAcceptanceContext) -> Dict[str, Any]:
    """Build the one bounded host packet shared by binding and reviewer input."""
    from ouroboros.review_evidence import build_task_acceptance_evidence

    committed_this_turn = any(
        isinstance(call, dict)
        and str(call.get("tool") or "") in ("commit_reviewed", "vcs_commit_reviewed")
        and str(call.get("status") or "") == "ok"
        for call in (ctx.llm_trace.get("tool_calls") or [])
    )
    evidence = build_task_acceptance_evidence(
        ctx.tools._ctx,
        llm_trace=ctx.llm_trace,
        drive_root=ctx.drive_root,
        task_id=ctx.task_id,
        task_type=ctx.task_type,
        agent_evidence=_latest_agent_acceptance_evidence(ctx.llm_trace),
        include_recent_commit=committed_this_turn,
        canonical_subject=str(ctx.content or ""),
        subtree_statuses=ctx.subtree_statuses,
    )
    # Owner Q2A: the forced children_unabsorbed rail stashes the process debt
    # (undispositioned children) so the panel sees it; part of the binding hash.
    undecided = getattr(ctx.tools._ctx, "_forced_undispositioned_children", None)
    if isinstance(undecided, list) and undecided:
        evidence["undispositioned_children"] = undecided
    return evidence


def _execute_task_acceptance_panel(ctx: _TaskAcceptanceContext) -> Any:
    """Perform the one substantive host panel over the pre-bound evidence."""
    from ouroboros.review_evidence import task_acceptance_evidence_revision
    from ouroboros.review_substrate import (
        HARDNESS_ADVISORY_VISIBLE,
        ReviewRequest,
        ReviewRunResult,
        reviewer_slots,
        run_review_request,
    )
    from ouroboros.review_dispatch import (
        TaskAcceptanceDispatchUnavailable,
        bind_task_acceptance_paid_dispatch,
        run_zero_physical_task_acceptance as _free_dispatch,
        task_acceptance_preclaim_refusal,
    )

    evidence = ctx.evidence or _build_host_acceptance_evidence(ctx)
    slots = reviewer_slots(effort=resolve_effort("review"), role_hint="task acceptance")
    request = ReviewRequest(
        surface="task_acceptance",
        goal=(
            _extract_plain_text_from_content(ctx.messages[1].get("content"))
            if len(ctx.messages) > 1 else ""
        ),
        subject=str(ctx.content or ""),
        evidence=evidence,
        checklist=_ACCEPTANCE_REVIEW_CHECKLIST,
        policy={
            "full_output_enters_context": False,
            "hardness": HARDNESS_ADVISORY_VISIBLE,
            "min_successful_slots": adaptive_quorum(len(slots)),
            "fail_closed_on_errors": True,
            "classify_outcome_tier": True,
            "max_physical_attempts_per_actor": 2,
        },
        task_id=ctx.task_id, retry_key=f"task_acceptance:{task_acceptance_evidence_revision(evidence)}",
    )
    if not slots:
        return ReviewRunResult(
            request={"surface": "task_acceptance", "task_id": str(ctx.task_id)},
            actors=[],
            parsed_findings=[],
            aggregate_signal="DEGRADED",
            degraded=True,
            degraded_reasons=["no_review_slots"],
        )
    # Budget admission for the whole acceptance wave (v6.69.0): a wave that
    # cannot fit the remaining root budget is declined up front as a terminal
    # DEGRADED (no-quorum semantics) instead of dying mid-wave. The estimate
    # renders the REAL per-slot message pair; the rare second physical
    # attempt is not multiplied in — fail-open coarse filter, no reservation.
    from ouroboros.tools.review_helpers import review_wave_budget_gate

    try:
        from ouroboros.review_substrate import _messages_char_count, _request_messages

        _prompt_chars = _messages_char_count(_request_messages(request, slots[0])) if slots else 0
    except Exception:
        _prompt_chars = len(json.dumps(evidence, ensure_ascii=False, default=str))
    _admission = review_wave_budget_gate(
        ctx.tools._ctx,
        surface="task_acceptance",
        models=[getattr(slot, "model", "") for slot in slots],
        prompt_chars=_prompt_chars,
    )
    if _admission is not None:
        return ReviewRunResult(
            request={"surface": "task_acceptance", "task_id": str(ctx.task_id)},
            actors=[],
            parsed_findings=[],
            aggregate_signal="DEGRADED",
            degraded=True,
            degraded_reasons=[
                "review_wave_budget_insufficient: estimated "
                f"~${_admission.get('estimated_wave_usd')} > remaining "
                f"${_admission.get('remaining_usd')} (no reviewer was called)"
            ],
        )
    free_result = _free_dispatch(
        request, slots, drive_root=ctx.drive_root or ctx.tools._ctx.drive_root, usage_ctx=ctx.tools._ctx)
    if free_result is not None:
        return free_result
    refusal = task_acceptance_preclaim_refusal(ctx)
    if refusal is not None:
        return refusal
    # Q6: bind the exact tree wallet to the target's physical-dispatch stamp.
    # Route/candidate refusals remain free; one strict stamp gates every slot.
    started = time.monotonic()
    try:
        with bind_task_acceptance_paid_dispatch(ctx) as usage_ctx:
            result = run_review_request(
                request, slots=slots,
                drive_root=(pathlib.Path(ctx.drive_root) if ctx.drive_root is not None
                            else pathlib.Path(ctx.tools._ctx.drive_root)),
                usage_ctx=usage_ctx,
            )
    except TaskAcceptanceDispatchUnavailable as exc:
        return ReviewRunResult(
            request={"surface": "task_acceptance", "task_id": str(ctx.task_id)},
            actors=[], parsed_findings=[], aggregate_signal="DEGRADED", degraded=True,
            degraded_reasons=[f"{exc} (no reviewer was called)"],
        )
    duration_sec = round(time.monotonic() - started, 3)
    try:
        from ouroboros.utils import append_jsonl, utc_now_iso

        append_jsonl(
            task_pacing.acceptance_timing_events_path(ctx.tools._ctx),
            {
                "ts": utc_now_iso(),
                "type": "task_acceptance_review_timing",
                "task_id": str(ctx.task_id),
                "duration_sec": duration_sec,
                "pass_index": ctx.passes_done,
                "aggregate_signal": str(result.aggregate_signal or ""),
            },
        )
    except Exception:
        log.debug("Failed to persist task-acceptance timing event", exc_info=True)
    return result


def _record_host_acceptance_run(ctx: _TaskAcceptanceContext, result: Any) -> Dict[str, Any]:
    """Append the authoritative host result after demoting agent-tool evidence."""
    _mark_agent_acceptance_runs_advisory(ctx.llm_trace)
    for prior in ctx.llm_trace.get("review_runs") or []:
        if (
            isinstance(prior, dict)
            and prior.get("authority") == "host_root"
            and not prior.get("superseded_by_revision")
        ):
            prior["superseded_by_revision"] = True
            prior["superseded_reason"] = "atomically_replaced_by_host_root_review"
    run_record = dict(getattr(result, "__dict__", {}) or {})
    for key in (
        "request", "actors", "parsed_findings", "aggregate_signal", "degraded",
        "degraded_reasons", "single_reviewer_no_diversity",
    ):
        if key not in run_record and hasattr(result, key):
            run_record[key] = getattr(result, key)
    run_record["authority"] = "host_root"
    run_record.update(ctx.review_binding or {})
    aggregate = str(run_record.get("aggregate_signal") or "DEGRADED").upper()
    run_record["enforcement_impact"] = (
        "allows_completion"
        if aggregate == "PASS"
        else "degrades_completion"
    )
    ctx.llm_trace.setdefault("review_runs", []).append(run_record)
    seen = getattr(ctx.tools._ctx, "_task_acceptance_seen_bindings", None)
    binding_hash = str(run_record.get("binding_hash") or "")
    if isinstance(seen, dict) and binding_hash:
        seen[binding_hash] = run_record
    return run_record


def _set_applied_host_acceptance_impact(
    run_record: Any,
    result: Any,
    *,
    requires_revision: bool,
) -> None:
    """Record what the host actually did with a panel result."""
    if not isinstance(run_record, dict):
        return
    if requires_revision:
        run_record["enforcement_impact"] = "requires_revision"
        return
    from ouroboros.review_substrate import task_acceptance_is_clean

    run_record["enforcement_impact"] = (
        "allows_completion" if task_acceptance_is_clean(result) else "degrades_completion"
    )


def _apply_task_acceptance_result(
    ctx: _TaskAcceptanceContext,
    result: Any,
    *,
    record_run: bool = True,
    reused: bool = False,
) -> bool:
    """Apply one panel result; return whether the agent must take another round."""
    from ouroboros.review_substrate import (
        DIALOGUE_CONTINUE,
        aggregate_dialogue_status,
        build_improvement_capsule,
        dissent_findings,
        task_acceptance_is_clean,
    )

    if record_run:
        _record_host_acceptance_run(ctx, result)
    dissent = dissent_findings(result)
    blocking_lane = ctx.mode == "required" and get_review_enforcement() == "blocking"
    # A REUSED panel (unchanged binding) is the SAME reviewer act applied
    # again: re-collecting would mutate reviewer-authored state with no new
    # input, and the shifted evidence revision would buy a fresh paid panel
    # for a byte-identical resubmit (fable r2 #1); rows already collected.
    if blocking_lane and not reused:
        _collect_acceptance_obligations(ctx.llm_trace, result)
    open_obligations = _open_acceptance_obligations(ctx.llm_trace) if blocking_lane else []
    # v6.74.0 (A1): the capsule leads with the verdict, the concrete open
    # obligation ids, and the pre-rendered rails line (money/time/rounds/passes).
    capsule = build_improvement_capsule(
        result,
        rails_line=ctx.rails_line,
        open_obligations=open_obligations,
    )
    # v6.74.0 (A5): the reviewers' typed dialogue judgement, reduced over
    # ALL contract-valid actors with the panel's own quorum; persisted for
    # audit on the authoritative run record whatever branch applies below.
    dialogue = aggregate_dialogue_status(
        result, quorum=_acceptance_dialogue_quorum(result),
    )
    _attach_dialogue_to_host_run(ctx.llm_trace, dialogue)
    dialogue_terminal = dialogue["status"] != DIALOGUE_CONTINUE
    if task_acceptance_is_clean(result):
        ctx.tools._ctx._task_acceptance_reviewed = True
        _end_task_acceptance_fence(ctx.tools._ctx, outcome="terminal")
        _mark_root_acceptance_checkpoint(
            ctx.tools._ctx, ctx.llm_trace, status="pass", pass_index=ctx.passes_done,
        )
        if not _dispose_obligations_on_clean_pass(
            ctx.llm_trace, result, open_obligations, bool(dissent),
        ):
            _set_acceptance_decision(ctx.llm_trace, {
                "status": ACCEPTANCE_ACCEPTED,
                "reason": "clean_pass",
                "source": "task_acceptance_review",
                "rationale": "Quorum PASS classified the deliverable solved with criterion evidence.",
                "dissent_noted": bool(dissent),
            })
        ctx.emit_progress("Task acceptance review: PASS (clean acceptance).")
        return False

    budget_snapshot = task_pacing.build_budget_snapshot(
        ctx.tools._ctx, profile=ctx.budget_profile,
    )
    pass_ok, pass_reason = task_pacing.improvement_pass_allowed(
        budget_snapshot,
        ctx.passes_done,
        ctx.budget_profile,
        required_blocking=blocking_lane,
        estimated_sec=task_pacing.acceptance_review_estimate_sec(
            ctx.tools._ctx, passes_done=ctx.passes_done + 1,
        ),
        ctx=ctx.tools._ctx,
    )
    if dialogue_terminal:
        # v6.74.0 (A5): a reviewer quorum judged the dialogue no longer
        # actionable (unreachable_here / stable_disagreement). Finalize via
        # the EXISTING honest path recording BOTH positions in one
        # owner-visible line — reviewer authorship, not a host timer.
        ctx.tools._ctx._task_acceptance_reviewed = True
        _end_task_acceptance_fence(ctx.tools._ctx, outcome="terminal")
        _mark_root_acceptance_checkpoint(
            ctx.tools._ctx,
            ctx.llm_trace,
            status=str(result.aggregate_signal or "DEGRADED").lower(),
            pass_index=ctx.passes_done,
        )
        _set_acceptance_decision(ctx.llm_trace, {
            # The with/without-obligations distinction moves from the status token to
            # the `open_obligations` id list this branch already records.
            "status": ACCEPTANCE_FINALIZED_UNACCEPTED,
            "reason": "dialogue_terminal",
            "source": "task_acceptance_review",
            "rationale": (
                f"Reviewer quorum judged the dialogue {dialogue['status']}; "
                "finalizing honestly with both positions recorded "
                f"({len(open_obligations)} open obligation(s))."
            ),
            "dialogue_status": dialogue["status"],
            "dialogue_votes": dialogue["votes"],
            "dissent_noted": bool(dissent),
            "open_obligations": [str(item.get("id")) for item in open_obligations],
        })
        ctx.emit_progress(
            f"Task acceptance review: {result.aggregate_signal} — reviewer quorum judged "
            f"the dialogue {dialogue['status']}; finalizing with "
            f"{len(open_obligations)} open obligation(s)."
        )
        return False
    if capsule and pass_ok:
        _set_acceptance_decision(ctx.llm_trace, {
            "status": ACCEPTANCE_REVISION_REQUESTED,
            "reason": "improvement_capsule",
            "source": "task_acceptance_review",
            "rationale": "A compact advisory improvement capsule was fed back for one bounded revision pass.",
            "dissent_noted": bool(dissent),
        })
        ctx.tools._ctx._task_acceptance_improvement_passes = ctx.passes_done + 1
        if not _end_task_acceptance_fence(ctx.tools._ctx, outcome="revision"):
            ctx.tools._ctx._task_acceptance_reviewed = True
            _set_acceptance_decision(ctx.llm_trace, {
                "status": ACCEPTANCE_FINALIZED_UNACCEPTED,
                "reason": "fence_reopen_failed",
                "source": "task_acceptance_fence",
                "rationale": "The revision could not safely reopen queue admission at the dispatch boundary.",
            })
            return False
        if open_obligations:
            capsule += _format_obligations_clause(open_obligations)
        if ctx.content and ctx.content.strip():
            ctx.messages.append({"role": "assistant", "content": ctx.content})
        _append_or_merge_user_message(ctx.messages, capsule)
        ctx.emit_progress(
            f"Task acceptance review: {result.aggregate_signal} — improvement note fed back."
        )
        return True

    ctx.tools._ctx._task_acceptance_reviewed = True
    _end_task_acceptance_fence(ctx.tools._ctx, outcome="terminal")
    _mark_root_acceptance_checkpoint(
        ctx.tools._ctx,
        ctx.llm_trace,
        status=str(result.aggregate_signal or "DEGRADED").lower(),
        pass_index=ctx.passes_done,
    )
    if _dispose_obligations_on_clean_pass(
        ctx.llm_trace, result, open_obligations, bool(dissent),
    ):
        ctx.emit_progress(
            f"Task acceptance review: {result.aggregate_signal} (clean pass; obligations closed)."
        )
        return False
    aggregate_signal = str(result.aggregate_signal or "DEGRADED").upper()
    if aggregate_signal == "DEGRADED":
        _set_acceptance_decision(ctx.llm_trace, {
            "status": ACCEPTANCE_FINALIZED_UNACCEPTED,
            "reason": "review_degraded",
            "source": "task_acceptance_review",
            "rationale": "Acceptance reviewers did not reach a valid quorum.",
            "degraded_reasons": list(getattr(result, "degraded_reasons", []) or []),
            "open_obligations": [str(item.get("id")) for item in open_obligations],
        })
        # Per-slot causes were always in the structured decision; the
        # owner-visible line said only "no valid quorum", forcing a dig
        # through task_results for WHICH slot failed and why (v6.70.0).
        _degraded_reasons = list(getattr(result, "degraded_reasons", []) or [])
        # Bounded PREVIEW for the chat line only — the complete causes live in
        # the structured decision record (owner-facing full copy, per the
        # v6.70.0 honesty invariant).
        _reason_note = "; ".join(
            truncate_review_artifact(str(r), limit=300).replace("\n", " ")
            for r in _degraded_reasons[:4]
        )
        if len(_degraded_reasons) > 4:
            _reason_note += f" (+{len(_degraded_reasons) - 4} more in the task result)"
        ctx.emit_progress(
            "Task acceptance review: DEGRADED (no valid quorum; not recorded as PASS)."
            + (f" Causes: {_reason_note}" if _reason_note else "")
        )
        return False
    if capsule and open_obligations:
        _set_acceptance_decision(ctx.llm_trace, {
            "status": ACCEPTANCE_FINALIZED_UNACCEPTED,
            "reason": pass_reason if pass_reason == REASON_REVIEW_CYCLES_EXHAUSTED else "open_obligations",
            "source": "task_acceptance_review",
            "rationale": (
                f"Improvement gates exhausted ({pass_reason or 'passes spent'}) with "
                f"{len(open_obligations)} open obligation(s); finalizing honestly."
            ),
            "dissent_noted": bool(dissent),
            "open_obligations": [str(item.get("id")) for item in open_obligations],
        })
        ctx.emit_progress(
            f"Task acceptance review: {result.aggregate_signal} — finalizing with "
            f"{len(open_obligations)} open obligation(s) ({pass_reason or 'passes spent'})."
        )
    elif capsule:
        _set_acceptance_decision(ctx.llm_trace, {
            "status": ACCEPTANCE_FINALIZED_UNACCEPTED,
            "reason": (
                pass_reason if pass_reason == REASON_REVIEW_CYCLES_EXHAUSTED else
                "improvement_window_closed"
                if (not ctx.passes_done and pass_reason)
                else "capsule_spent"
            ),
            "source": "task_acceptance_review",
            "rationale": (
                f"Improvement window closed before any capsule pass ({pass_reason})."
                if not ctx.passes_done and pass_reason
                else "The bounded acceptance-review capsule was already spent; finalizing with the current answer."
            ),
            "dissent_noted": bool(dissent),
        })
        ctx.emit_progress(
            f"Task acceptance review: {result.aggregate_signal} "
            "(improvement note already fed back; finalizing)."
        )
    elif aggregate_signal == "FAIL":
        _set_acceptance_decision(ctx.llm_trace, {
            "status": ACCEPTANCE_FINALIZED_UNACCEPTED,
            "reason": "reviewer_fail_no_capsule",
            "source": "task_acceptance_review",
            "rationale": "A valid acceptance reviewer FAIL had no additional capsule text.",
            "dissent_noted": bool(dissent),
        })
        ctx.emit_progress("Task acceptance review: FAIL (finalizing with a failed review verdict).")
    elif open_obligations:
        _set_acceptance_decision(ctx.llm_trace, {
            "status": ACCEPTANCE_FINALIZED_UNACCEPTED,
            "reason": "open_obligations",
            "source": "task_acceptance_review",
            "rationale": (
                f"Re-review was not a clean PASS ({result.aggregate_signal}); "
                f"{len(open_obligations)} obligation(s) stay open — finalizing honestly."
            ),
            "dissent_noted": bool(dissent),
            "open_obligations": [str(item.get("id")) for item in open_obligations],
        })
        ctx.emit_progress(f"Task acceptance review: {result.aggregate_signal} (no changes suggested).")
    else:
        _set_acceptance_decision(ctx.llm_trace, {
            # Round-9 CRITICAL 1: fall-through AFTER
            # `task_acceptance_is_clean` refused the panel, so it cannot mint
            # `accepted` (reserved for clean acceptance). Reachable: a
            # reviewer claims `solved` with a MISSING criterion and the
            # improvement cap spent — nothing actionable, yet not "accepted";
            # the typed reason names WHY; tier honesty rides `outcome_tier`.
            "status": ACCEPTANCE_FINALIZED_UNACCEPTED,
            "reason": "no_actionable_changes",
            "source": "task_acceptance_review",
            "rationale": (
                f"Re-review was not a clean acceptance ({result.aggregate_signal}) and "
                "suggested no actionable changes; finalizing honestly without acceptance."
            ),
            "dissent_noted": bool(dissent),
        })
        ctx.emit_progress(
            f"Task acceptance review: {result.aggregate_signal} — not a clean acceptance "
            "and no actionable changes were suggested; finalizing without acceptance."
        )
    return False


def _record_acceptance_infra_failure(ctx: _TaskAcceptanceContext, exc: Exception) -> bool:
    """Finish an eligible mandatory panel as DEGRADED, never as a silent skip."""
    ctx.tools._ctx._task_acceptance_reviewed = True
    _end_task_acceptance_fence(ctx.tools._ctx, outcome="degraded")
    _mark_root_acceptance_checkpoint(
        ctx.tools._ctx,
        ctx.llm_trace,
        status="review_degraded",
        pass_index=ctx.passes_done,
    )
    safe_error = _extract_plain_text_from_content(str(exc))[:2000]
    _mark_agent_acceptance_runs_advisory(ctx.llm_trace)
    run_record = {
        "request": {"surface": "task_acceptance", "task_id": ctx.task_id},
        "actors": [],
        "parsed_findings": [{
            "severity": "critical",
            "item": "task_acceptance_infra_failure",
            "evidence": f"{type(exc).__name__}: {safe_error}",
            "recommendation": "Do not report semantic success unless the failure is explicitly accounted for.",
        }],
        "aggregate_signal": "DEGRADED",
        "degraded": True,
        "degraded_reasons": [f"{type(exc).__name__}: {safe_error}"],
        "authority": "host_root",
        **(ctx.review_binding or {}),
        "enforcement_impact": "degrades_completion",
    }
    ctx.llm_trace.setdefault("review_runs", []).append(run_record)
    seen = getattr(ctx.tools._ctx, "_task_acceptance_seen_bindings", None)
    binding_hash = str(run_record.get("binding_hash") or "")
    if isinstance(seen, dict) and binding_hash:
        seen[binding_hash] = run_record
    _set_acceptance_decision(ctx.llm_trace, {
        "status": ACCEPTANCE_FINALIZED_UNACCEPTED,
        "reason": "infra_failure",
        "source": "task_acceptance_review",
        "rationale": "The mandatory host acceptance panel failed before a valid quorum.",
        "degraded_reasons": [f"{type(exc).__name__}: {safe_error}"],
    })
    ctx.emit_progress("Task acceptance review: DEGRADED after host review infrastructure failure.")
    return False


def _prior_acceptance_run(
    tools_ctx: Any, llm_trace: Dict[str, Any], binding_hash: str,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """Locate the authoritative host run already recorded for this binding:
    first the trace (survives requeue replay), then the process-local
    ``_task_acceptance_seen_bindings`` cache. Returns (cache, prior_run)."""
    seen_bindings = getattr(tools_ctx, "_task_acceptance_seen_bindings", None)
    if not isinstance(seen_bindings, dict):
        seen_bindings = {}
        tools_ctx._task_acceptance_seen_bindings = seen_bindings
    prior_run = next(
        (
            run for run in reversed(llm_trace.get("review_runs") or [])
            if isinstance(run, dict)
            and run.get("authority") == "host_root"
            and not run.get("superseded_by_revision")
            and str(run.get("binding_hash") or "") == binding_hash
        ),
        None,
    )
    cached_run = seen_bindings.get(binding_hash)
    if (
        prior_run is None
        and isinstance(cached_run, dict)
        and not cached_run.get("superseded_by_revision")
    ):
        prior_run = cached_run
    return seen_bindings, prior_run


def _direct_context_fence_state(tools_ctx: Any, fence_token: Any) -> Any:
    """Review-binding fence state: the queue-owned token when present, else the
    direct-chat generations (no queue fence exists for a direct context)."""
    if fence_token is not None:
        return fence_token
    return {
        "state": "direct_context",
        "owner_generation": getattr(tools_ctx, "_task_acceptance_owner_generation", None),
        "queue_generation": getattr(tools_ctx, "_task_acceptance_fence_generation", None),
    }


def _run_task_acceptance_review_once(
    *,
    tools: ToolRegistry,
    content: str,
    task_id: str,
    task_type: str,
    llm_trace: Dict[str, Any],
    drive_root: Optional[pathlib.Path],
    messages: List[Dict[str, Any]],
    emit_progress: Callable[[str], None],
) -> bool:
    """Run the root-owned acceptance gate once for the current deliverable.
    Loop-side rails facts arrive via the ``_acceptance_loop_rails`` ctx stash
    (set by ``_no_tool_final_answer``; keeps the signature at 8 params)."""
    mode = get_task_review_mode()
    _latch_final_answer_marker(llm_trace, content)
    if getattr(tools._ctx, "_task_acceptance_reviewed", False):
        return False
    from ouroboros.task_results import resolve_task_lineage

    meta = getattr(tools._ctx, "task_metadata", {})
    meta = meta if isinstance(meta, dict) else {}
    lineage = resolve_task_lineage(
        task_id or getattr(tools._ctx, "task_id", ""),
        metadata=meta,
        root_task_id=getattr(tools._ctx, "root_task_id", None),
        parent_task_id=getattr(tools._ctx, "parent_task_id", None),
        delegation_role=getattr(tools._ctx, "delegation_role", None),
        original_task_id=getattr(tools._ctx, "original_task_id", None),
        timeout_retry_from=getattr(tools._ctx, "timeout_retry_from", None),
    )
    eligible, trigger = _task_acceptance_eligible(
        mode,
        llm_trace,
        bool(getattr(tools._ctx, "is_direct_chat", False)),
        is_root_task=bool(lineage["is_root_task"]),
        is_ephemeral_turn=bool(getattr(tools._ctx, "is_ephemeral_turn", False)),
        task_contract=(
            tools._ctx.task_contract
            if isinstance(getattr(tools._ctx, "task_contract", None), dict)
            else {}
        ),
    )
    agent_called = any(
        isinstance(call, dict) and str(call.get("tool") or "") == "task_acceptance_review"
        for call in (llm_trace.get("tool_calls") or [])
    )
    agent_review_present = any(
        isinstance(run, dict)
        and isinstance(run.get("request"), dict)
        and str((run.get("request") or {}).get("surface") or "") == "task_acceptance"
        and str(run.get("aggregate_signal") or "").strip()
        for run in (llm_trace.get("review_runs") or [])
    )
    if agent_review_present:
        _mark_agent_acceptance_runs_advisory(llm_trace)
        trigger = f"{trigger}_after_agent_advisory"
    elif agent_called:
        trigger = f"{trigger}_after_agent_tool"
    llm_trace["review_decision"] = {
        "eligibility": "eligible" if eligible else "not_eligible", "trigger": trigger,
    }
    if not eligible:
        return False
    # Owner hurry (§19.7.2 item 8): AFTER structural eligibility is known,
    # BEFORE acceptance-fence/quiescence/reviewer admission, an armed latch
    # skips the next otherwise-eligible panel with the typed reason — no
    # reviewer calls (an in-flight panel is never cancelled/relabeled).
    from ouroboros.owner_hurry import acceptance_skip_applied, effective_budget_profile

    if acceptance_skip_applied(
        tools._ctx, llm_trace, task_id=task_id, drive_root=drive_root,
        set_decision=_set_acceptance_decision, emit_progress=emit_progress,
    ):
        return False
    fence_ok, _fence_token = _begin_task_acceptance_fence(tools._ctx, task_id)
    if not fence_ok:
        llm_trace["review_decision"] = {
            "eligibility": "acceptance_fence_failed", "trigger": trigger,
        }
        # Bounded wait (CyberGym full1507 postmortem): each fence-unavailable
        # round used to burn one paid LLM round until the 4h deadline. Count
        # consecutive failures and terminalize as infra_failed at the config
        # cap instead; the leaked supervisor-side fence is cleared by
        # task_done / the reaper / the dead-owner sweep.
        fence_failures = int(getattr(tools._ctx, "_task_acceptance_fence_failures", 0) or 0) + 1
        tools._ctx._task_acceptance_fence_failures = fence_failures
        if fence_failures >= get_acceptance_fence_wait_max_rounds():
            tools._ctx._task_acceptance_fence_infra_failed = True
            _set_acceptance_decision(llm_trace, {
                "status": ACCEPTANCE_FINALIZED_UNACCEPTED,
                "reason": "acceptance_fence_unavailable",
                "source": "acceptance_fence",
                "rationale": (
                    f"The queue-owned admission fence stayed unavailable for "
                    f"{fence_failures} consecutive rounds; terminalizing as an "
                    "infrastructure failure instead of burning paid rounds "
                    "until the deadline."
                ),
            })
            emit_progress(
                "Task acceptance review could not acquire the queue-owned admission "
                "fence; terminalizing as an infrastructure failure."
            )
            return False
        _append_or_merge_user_message(
            messages,
            "[TASK ACCEPTANCE WAIT] The supervisor could not atomically close "
            "subtask admission. Do not finalize or spawn more work; retry after the "
            "queue fence is available.",
        )
        emit_progress("Task acceptance review waiting for the queue-owned admission fence.")
        return True
    tools._ctx._task_acceptance_fence_failures = 0
    quiescent, subtree_statuses = _task_acceptance_subtree_snapshot(
        tools._ctx, drive_root, task_id,
    )
    if not quiescent:
        llm_trace["review_decision"] = {
            "eligibility": "waiting_for_quiescence",
            "trigger": trigger,
            "live_descendants": [
                row for row in subtree_statuses
                if str(row.get("status") or "")
                not in {"completed", "failed", "cancelled", "rejected_duplicate"}
            ],
        }
        _append_or_merge_user_message(
            messages,
            "[TASK ACCEPTANCE WAIT] The root acceptance review requires the recursive "
            "subtree to be terminal. Absorb or explicitly cancel the remaining child "
            "tasks before finalizing.",
        )
        emit_progress("Task acceptance review waiting for recursive subtree quiescence.")
        return True
    # §19.7.2 item 7: ONE effective profile (remaining improvement passes ->
    # 0 under an armed hurry latch) feeds EVERY acceptance-pacing read below
    # — the improvement_pass_allowed call and the rails display alike.
    budget_profile = effective_budget_profile(
        tools._ctx, task_pacing.resolve_budget_profile(tools._ctx),
    )
    budget_snapshot = task_pacing.build_budget_snapshot(tools._ctx, profile=budget_profile)
    passes_done = int(getattr(tools._ctx, "_task_acceptance_improvement_passes", 0))
    launch_ok, launch_reason = task_pacing.review_launch_allowed(
        budget_snapshot,
        estimated_sec=task_pacing.acceptance_review_estimate_sec(
            tools._ctx, passes_done=passes_done,
        ),
    )
    if not launch_ok:
        tools._ctx._task_acceptance_reviewed = True
        _end_task_acceptance_fence(tools._ctx, outcome="terminal")
        _mark_root_acceptance_checkpoint(
            tools._ctx, llm_trace, status=launch_reason, pass_index=passes_done,
        )
        llm_trace["review_decision"].update({"skipped": launch_reason})
        # The pacing launch reason is now the typed REASON, not the status;
        # `outcomes.derive_loop_outcome` keys on that PAIR (see its comment).
        _set_acceptance_decision(llm_trace, {
            "status": ACCEPTANCE_FINALIZED_UNACCEPTED, "reason": launch_reason,
            "source": "task_pacing",
            "rationale": (
                f"Remaining {budget_snapshot.remaining_sec:.0f}s is inside the finalization "
                f"reserve ({budget_snapshot.reserve_sec:.0f}s); finalizing without review."
            ),
        })
        emit_progress("Task acceptance review skipped: inside the finalization reserve.")
        return False
    review_ctx = _TaskAcceptanceContext(
        tools=tools,
        content=content,
        task_id=task_id,
        task_type=task_type,
        llm_trace=llm_trace,
        drive_root=drive_root,
        messages=messages,
        emit_progress=emit_progress,
        mode=mode,
        subtree_statuses=subtree_statuses,
        budget_profile=budget_profile,
        passes_done=passes_done,
        evidence={},
        review_binding={},
        rails_line=task_pacing.acceptance_rails_line(
            budget_snapshot,
            budget_profile,
            passes_done,
            getattr(tools._ctx, "_acceptance_loop_rails", None),
            required_blocking=(
                mode == "required" and get_review_enforcement() == "blocking"
            ), workspace=task_pacing._workspace_delivery(tools._ctx),
        ),
    )
    try:
        from types import SimpleNamespace

        from ouroboros.review_evidence import task_acceptance_evidence_revision
        from ouroboros.review_substrate import build_review_binding

        review_ctx.evidence = _build_host_acceptance_evidence(review_ctx)
        review_ctx.review_binding = build_review_binding(
            candidate=content,
            evidence=review_ctx.evidence,
            fence_token_or_state=_direct_context_fence_state(tools._ctx, _fence_token),
        )
        binding_hash = str(review_ctx.review_binding.get("binding_hash") or "")
        seen_bindings, prior_run = _prior_acceptance_run(
            tools._ctx, llm_trace, binding_hash,
        )
        reused_result = None
        if prior_run is not None:
            seen_bindings[binding_hash] = prior_run
            if prior_run not in (llm_trace.get("review_runs") or []):
                llm_trace.setdefault("review_runs", []).append(dict(prior_run))
            llm_trace["review_decision"].update({
                "panel_reused": True,
                "panel_id": str(prior_run.get("panel_id") or ""),
                "binding_hash": binding_hash,
            })
            emit_progress(
                "Task acceptance review: reusing the authoritative result for the unchanged binding."
            )
            # Re-run the normal semantic application (gates, outcome axis,
            # obligations, fence) without appending or paying for another panel.
            reused_result = SimpleNamespace(**prior_run)
        elif binding_hash in seen_bindings:
            # A process-local attempt without its authoritative trace is not
            # safe to repeat or silently accept. The infra-degraded path below
            # records the missing authority and closes finalization honestly.
            raise RuntimeError("acceptance binding was attempted but its host run is unavailable")
        else:
            seen_bindings[binding_hash] = None
        llm_trace["review_decision"].update({
            "panel_id": str(review_ctx.review_binding.get("panel_id") or ""),
            "binding_hash": binding_hash,
        })
        messages_before_apply = list(messages)
        obligations_were_present = "acceptance_obligations" in llm_trace
        obligations_before_apply = [
            dict(row) if isinstance(row, dict) else row
            for row in (llm_trace.get("acceptance_obligations") or [])
        ]
        passes_before_apply = int(
            getattr(tools._ctx, "_task_acceptance_improvement_passes", 0) or 0
        )
        panel_result = reused_result or _execute_task_acceptance_panel(review_ctx)
        run_record = (
            prior_run
            if reused_result is not None
            else _record_host_acceptance_run(review_ctx, panel_result)
        )
        if _task_acceptance_owner_generation_changed(tools._ctx):
            _supersede_task_acceptance_for_owner_followup(tools._ctx, llm_trace)
            emit_progress(
                "Task acceptance review superseded: an owner follow-up arrived during the panel."
            )
            return True
        fresh_quiescent, fresh_subtree_statuses = _task_acceptance_subtree_snapshot(
            tools._ctx, drive_root, task_id,
        )
        fresh_review_ctx = replace(
            review_ctx,
            subtree_statuses=fresh_subtree_statuses,
            evidence={},
        )
        fresh_evidence_revision = task_acceptance_evidence_revision(
            _build_host_acceptance_evidence(fresh_review_ctx)
        )
        frozen_evidence_revision = str(
            review_ctx.review_binding.get("evidence_revision") or ""
        )
        stale_reason = ""
        if not fresh_quiescent:
            stale_reason = "host_acceptance_subtree_became_non_quiescent"
        elif fresh_evidence_revision != frozen_evidence_revision:
            stale_reason = "host_acceptance_evidence_revision_changed"
        if stale_reason:
            _supersede_task_acceptance_for_evidence_change(
                tools._ctx,
                llm_trace,
                run_record,
                stale_reason,
                messages,
                emit_progress,
            )
            return True
        another_round = _apply_task_acceptance_result(
            review_ctx,
            panel_result,
            record_run=False,
            reused=reused_result is not None,
        )
        if getattr(tools._ctx, "_task_acceptance_fence_generation_mismatch", False):
            messages[:] = messages_before_apply
            if obligations_were_present:
                llm_trace["acceptance_obligations"] = obligations_before_apply
            else:
                llm_trace.pop("acceptance_obligations", None)
            tools._ctx._task_acceptance_improvement_passes = passes_before_apply
            _supersede_task_acceptance_for_owner_followup(tools._ctx, llm_trace)
            emit_progress(
                "Task acceptance review superseded: an owner follow-up arrived during the panel."
            )
            return True
        _set_applied_host_acceptance_impact(
            run_record,
            panel_result,
            requires_revision=another_round,
        )
        return another_round
    except Exception as exc:
        log.debug("Mandatory task acceptance review failed", exc_info=True)
        return _record_acceptance_infra_failure(review_ctx, exc)


def _adopt_fallback_route(
    ctx: Any,
    tools: ToolRegistry,
    fallback_model: str,
    fallback_use_local: bool,
    messages: List[Dict[str, Any]],
    fallback_messages: List[Dict[str, Any]],
    context_fit_plan: Any,
    active_context_mode: str,
    tool_schemas: List[Dict[str, Any]],
    accumulated_usage: Dict[str, Any],
) -> tuple:
    """Round-4 C1.1: adopt a SUCCESSFUL cross-family fallback as the active
    route for the rest of the loop. Otherwise a later round (esp. a tool
    loop) replays THIS fallback's reasoning/thinking back to the original
    primary family with no model-switch sanitizer firing (active_model never
    changed) — the cross-family signature replay, in reverse. Adopting the
    sanitized transcript keeps the old family's provider-private blocks off
    the switched route (a later switch_model/override re-triggers the
    round-start sanitizer); the caller already rebound the context-fit plan
    to this exact route, so adoption makes that tested projection canonical.
    Returns ``(active_model, active_use_local, context_fit_plan, context_mode)``."""
    ctx.active_model = fallback_model
    messages[:] = fallback_messages
    if context_fit_plan is not None:
        tools._ctx.context_fit_plan = context_fit_plan
        tools._ctx.messages = messages
        tools._ctx.active_context_mode = active_context_mode
        # _call_round_model already recorded the accepted candidate's complete
        # same-basis fit facts. Do not replace them with a raw char estimate.
    return fallback_model, fallback_use_local, context_fit_plan, active_context_mode


def _snapshot_context_fit_usage(usage: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in usage.items() if key.startswith("_context_")}


def _restore_context_fit_usage(
    usage: Dict[str, Any],
    snapshot: Dict[str, Any],
) -> None:
    for key in tuple(usage):
        if key.startswith("_context_"):
            usage.pop(key, None)
    usage.update(snapshot)


def _run_cross_model_fallback_chain(
    *, llm, ctx, tools, messages, active_model, active_use_local, tool_schemas,
    active_effort, max_retries, drive_logs, task_id, round_idx, event_queue,
    accumulated_usage, task_type, emit_progress, context_fit_plan,
    active_context_mode,
) -> tuple:
    """Try fallbacks; unknown dispatch stops the chain."""
    from ouroboros import fallback_cooldown as _fcd
    from ouroboros.config import get_fallback_models
    from ouroboros.loop_llm_call import _COOLDOWN_ERROR_KINDS as _cooldown_kinds

    def _cooled(model: str, use_local: bool) -> None:
        if str(accumulated_usage.get("_last_llm_error_kind") or "") in _cooldown_kinds:
            _fcd.mark_cooldown(model, use_local)

    _cooled(active_model, active_use_local)
    primary_context_usage = _snapshot_context_fit_usage(accumulated_usage)
    fallback_use_local = os.environ.get("USE_LOCAL_FALLBACK", "").lower() in ("true", "1")
    attempt_cap = _fcd.attempts_per_model()
    msg = None
    for fallback_model in get_fallback_models(active_model):
        if _fcd.is_cooling_down(fallback_model, fallback_use_local):
            continue
        deadline = _task_deadline_epoch(tools)
        if deadline and time.time() >= deadline:
            break
        ptag = " (local)" if active_use_local else ""
        ftag = " (local)" if fallback_use_local else ""
        emit_progress(f"⚡ Fallback: {active_model}{ptag} → {fallback_model}{ftag}")
        # Cross-FAMILY fallback must not replay the primary's
        # provider-private reasoning to a different family (the GLM->Claude
        # 400 "Invalid signature" death); the SSOT sanitizer no-ops same-family.
        fallback_messages = LLMClient.sanitize_reasoning_on_model_switch(messages, active_model, fallback_model)
        # Bind exact route evidence and choose its deterministic projection
        # BEFORE physical dispatch: the fallback's first request must not
        # inherit the failed primary route's Max projection/fingerprint. It
        # then uses the ordinary single confirmed-overflow Low retry path.
        candidate_plan, candidate_mode = _rebind_context_fit_plan(
            context_fit_plan,
            tools,
            fallback_messages,
            model=fallback_model,
            use_local=fallback_use_local,
            preferred_mode=str(
                getattr(context_fit_plan, "preferred_mode", "") or active_context_mode
            ),
            tool_schemas=tool_schemas,
        )
        msg, _cost, candidate_mode = _call_round_model(
            _RoundModelCallContext(
                llm=llm,
                messages=fallback_messages,
                tools=tools,
                context_fit_plan=candidate_plan,
                active_model=fallback_model,
                tool_schemas=tool_schemas,
                active_effort=active_effort,
                max_retries=max_retries,
                drive_logs=drive_logs,
                task_id=task_id,
                round_idx=round_idx,
                event_queue=event_queue,
                accumulated_usage=accumulated_usage,
                task_type=task_type,
                active_use_local=fallback_use_local,
                active_context_mode=candidate_mode,
                drive_root=pathlib.Path(drive_logs).parent,
                attempt_cap=attempt_cap,
            )
        )
        if msg is not None:
            (
                active_model,
                active_use_local,
                context_fit_plan,
                active_context_mode,
            ) = _adopt_fallback_route(
                ctx,
                tools,
                fallback_model,
                fallback_use_local,
                messages,
                fallback_messages,
                candidate_plan,
                candidate_mode,
                tool_schemas,
                accumulated_usage,
            )
            break
        tools._ctx.context_fit_plan = context_fit_plan
        tools._ctx.messages = messages
        tools._ctx.active_context_mode = active_context_mode
        _restore_context_fit_usage(accumulated_usage, primary_context_usage)
        if str(accumulated_usage.get("_last_llm_error_kind") or "") in ("provider_outcome_unknown", "deadline_exhausted", "transport_unavailable"):
            break
        _cooled(fallback_model, fallback_use_local)
    return (
        msg,
        active_model,
        active_use_local,
        context_fit_plan,
        active_context_mode,
    )


def _load_direct_child_results(
    status_root: pathlib.Path,
    task_id: str,
    root_task_id: str,
) -> list[Dict[str, Any]]:
    """Read this task's direct children (plan review spawns none)."""

    from ouroboros.task_status import find_child_tasks

    return [
        row for row in find_child_tasks(
            pathlib.Path(status_root),
            parent_task_id=task_id,
            root_task_id=root_task_id,
            exclude_task_id=task_id,
            scope="direct",
        )
        if isinstance(row, dict)
    ]


def _compute_subagent_handoff(tools: Any, drive_root: Any, task_id: str, content: Any) -> str:
    """C3.4 pre-finalization child absorption: build the bounded subagent-handoff
    reminder when a finished child's status/result changed since the last refresh, or
    a nonterminal child is unacknowledged in the final text. Returns "" when there is
    nothing to inject. Scans the SAME status root get_task_result uses
    (budget_drive_root, not the forked drive_root — else nested grandchildren in
    forked child drives are missed). Never raises."""
    if drive_root is None or not task_id:
        return ""
    try:
        from ouroboros.task_status import FINAL_STATUSES, format_subagent_absorption_message

        metadata = getattr(tools._ctx, "task_metadata", {}) if isinstance(getattr(tools._ctx, "task_metadata", {}), dict) else {}
        status_drive_root = pathlib.Path(
            str(metadata.get("budget_drive_root") or getattr(tools._ctx, "budget_drive_root", "") or "")
            or drive_root
        )
        children = _load_direct_child_results(
            status_drive_root,
            task_id,
            str(metadata.get("root_task_id") or task_id),
        )
        # Exact-hash dispositions suppress the unchanged result only: if
        # status, result, trace, or artifact identity changes, the disposition
        # goes stale and this reminder re-opens without parsing prose.
        children = [
            child for child in children
            if _child_disposition_state(child) not in {
                "integrated", "irrelevant", "deferred", "discarded", "cancelled",
            }
        ]
        from ouroboros.tools.join_ledger import _child_result_sha256

        signature = "|".join(
            f"{child.get('task_id') or child.get('id')}:{_child_result_sha256(child)}"
            for child in children
        )
        previous = getattr(tools._ctx, "_subagent_handoff_signature", "")
        nonterminal_children = [
            child for child in children
            if str(child.get("status") or "").strip().lower() not in FINAL_STATUSES
        ]
        # P5: the reminder is suppressed ONLY by structured signals — a
        # child discarded/cancelled (filtered above) or absorbed (unchanged
        # signature), NEVER by parsing final PROSE; fires once per CHANGE, and
        # finalizing with unhandled children appends a loud orphan note (P1).
        _ = nonterminal_children  # (kept for readability; trigger is change-based)
        if children and signature and signature != previous:
            tools._ctx._subagent_handoff_signature = signature
            tools._ctx._child_absorption_reminded = False
            _absorb_budget = 160_000 if str(get_context_mode()).lower() == "max" else 60_000
            return format_subagent_absorption_message(
                children, parent_task_id=task_id, budget_chars=_absorb_budget,
            )
    except Exception:
        log.debug("Failed to build subagent handoff reminder", exc_info=True)
    return ""


def _maybe_inject_self_check(
    round_idx: int,
    max_rounds: int,
    messages: List[Dict[str, Any]],
    accumulated_usage: Dict[str, Any],
    emit_progress: Callable[[str], None],
    *,
    event_queue: Optional[queue.Queue] = None,
    task_id: str = "",
    drive_logs: Optional[pathlib.Path] = None,
) -> bool:
    """Inject a normal user-turn self-check and emit one checkpoint event."""
    REMINDER_INTERVAL = 15
    if round_idx <= 1 or round_idx % REMINDER_INTERVAL != 0 or round_idx >= max_rounds:
        return False
    # Non-incrementing round re-entries (e.g. free redials): one self-check per round.
    if accumulated_usage.get("_self_check_round") == round_idx:
        return False
    accumulated_usage["_self_check_round"] = round_idx

    ctx_tokens = sum(
        estimate_tokens(_extract_plain_text_from_content(m.get("content")))
        for m in messages
    )
    raw_task_cost = accumulated_usage.get("cost")
    task_cost = float(raw_task_cost) if raw_task_cost is not None else None
    cost_text = f"${task_cost:.2f}" if task_cost is not None else "unknown"
    checkpoint_num = round_idx // REMINDER_INTERVAL

    # Tree spend under a root cap (v6.91): the checkpoint is already a
    # cache-breaking user turn — one of the RARE surfaces allowed a live
    # ledger number (DEVELOPMENT cache_friendliness item 22). The fence
    # counts the whole tree; own cost alone hid two tree deaths.
    tree_line = ""
    tree_accounted: Optional[float] = None
    tree_cap: Optional[float] = None
    tree_info = _loop_tree_accounting(refresh=True, max_age_sec=30.0)
    if isinstance(tree_info, dict) and tree_info.get("accounted_usd") is not None:
        tree_accounted = float(tree_info["accounted_usd"])
        raw_cap = tree_info.get("root_limit_usd")
        tree_cap = float(raw_cap) if raw_cap is not None else None
        cap_text = f" of ${tree_cap:.2f} hard tree cap" if tree_cap is not None else ""
        tree_line = (
            f"Task tree spend: ~${tree_accounted:.2f}{cap_text} "
            "(ledger-accounted incl. in-flight holds, subagents included)\n"
        )

    tool_trace = _build_recent_tool_trace(messages)

    reminder = (
        f"[CHECKPOINT {checkpoint_num} — round {round_idx}/{max_rounds}]\n"
        f"Context: ~{ctx_tokens} tokens | Cost so far: {cost_text} | "
        f"Rounds remaining: {max_rounds - round_idx}\n"
        f"{tree_line}"
    )
    if tool_trace:
        reminder += f"\n{tool_trace}\n"
    reminder += (
        "\nThis is a periodic self-check, not a command to stop. "
        "Glance at your recent tool-call trace above and briefly consider:\n"
        "- Are you still making progress toward the task, or repeating the same actions?\n"
        "- Is the current approach still the right one, or should you narrow scope / try a different angle?\n"
        "- If you are waiting on a long build/download/training run or have independent branches of investigation, consider schedule_subagent for a focused parallel handoff.\n"
        "- If the task is effectively done, first re-check the literal original requirements one by one "
        "against the specified interface/path/format/service, then wrap up by replying with your final answer in plain text (no tool call). "
        "Otherwise continue with the most valuable next step.\n"
        "\nNo special format required — just think, then act."
    )

    # Merge into a prior user turn to avoid Anthropic consecutive-role 400s,
    # preserving multipart blocks so images/cache markers survive.
    _append_or_merge_user_message(messages, reminder)
    emit_progress(
        f"Checkpoint {checkpoint_num} at round {round_idx}: "
        f"~{ctx_tokens} tokens, {cost_text} spent"
    )

    checkpoint_payload: Dict[str, Any] = {
        "checkpoint_number": checkpoint_num,
        "round": round_idx,
        "max_rounds": max_rounds,
        "context_tokens": ctx_tokens,
        "task_cost": task_cost,
    }
    if tree_accounted is not None:
        checkpoint_payload["tree_accounted_usd"] = round(tree_accounted, 4)
        checkpoint_payload["tree_cap_usd"] = round(tree_cap, 4) if tree_cap is not None else None
    _emit_checkpoint_event(event_queue, task_id, drive_logs, checkpoint_payload)

    return True


def _maybe_inject_time_budget_milestone(
    messages: List[Dict[str, Any]],
    tools: ToolRegistry,
    *,
    event_queue: Optional[queue.Queue] = None,
    task_id: str = "",
    drive_logs: Optional[pathlib.Path] = None,
    round_idx: int = 0,
    accumulated_usage: Optional[Dict[str, Any]] = None,
) -> bool:
    """Thin transport over the task_pacing SSOT (v6.54.4): the milestone content,
    thresholds, and seen-state live in ouroboros/task_pacing.py; this wrapper only
    appends the note and emits the checkpoint event."""
    note = task_pacing.build_time_budget_note(
        tools._ctx, round_idx=round_idx, accumulated_usage=accumulated_usage,
        # A real ledger read happens ONLY when the pacing note actually fires
        # (per 600s bucket) — the note is a cache-breaking user turn already.
        tree_cost_provider=lambda: _loop_tree_accounting(refresh=True, max_age_sec=30.0),
    )
    if note is None:
        return False
    _append_or_merge_user_message(messages, note.text)
    _emit_checkpoint_event(event_queue, task_id, drive_logs, note.checkpoint)
    return True


def _maybe_inject_cost_budget_milestone(
    messages: List[Dict[str, Any]],
    tools: ToolRegistry,
    *,
    budget_remaining_usd: Optional[float],
    cost_ceiling: Optional["task_pacing.CostCeiling"],
    accumulated_usage: Optional[Dict[str, Any]],
    event_queue: Optional[queue.Queue] = None,
    task_id: str = "",
    drive_logs: Optional[pathlib.Path] = None,
) -> bool:
    """Thin transport over the task_pacing cost axis (v6.56.0): content,
    thresholds, and latch state live in ouroboros/task_pacing.py. The deciding
    spend under a root cap is the tree-accounted stash (free read; refreshed by
    every dispatch) with a bounded staleness cap — never a per-round ledger
    read, see ``_TREE_ACCOUNTING_MAX_STALE_SEC``."""
    ceiling_usd = (
        cost_ceiling.ceiling_usd
        if cost_ceiling is not None and cost_ceiling.state == task_pacing.COST_CEILING_ACTIVE
        else None
    )
    tree_info = _loop_tree_accounting(
        refresh=True, max_age_sec=_TREE_ACCOUNTING_MAX_STALE_SEC,
    )
    tree_cost = tree_info.get("accounted_usd") if isinstance(tree_info, dict) else None
    note = task_pacing.build_cost_budget_note(
        tools._ctx,
        start_remaining_usd=budget_remaining_usd,
        cost_ceiling_usd=ceiling_usd,
        task_cost=(accumulated_usage or {}).get("cost"),
        tree_cost_usd=tree_cost,
        # Whether a tree cap exists at all decides if own cost is the complete
        # picture or a disclosed lower bound (task_pacing.resolve_deciding_spend).
        root_cap_usd=(cost_ceiling.root_cap_usd if cost_ceiling is not None else None),
    )
    if note is None:
        return False
    _append_or_merge_user_message(messages, note.text)
    _emit_checkpoint_event(event_queue, task_id, drive_logs, note.checkpoint)
    return True


from ouroboros.nanny_pacing import (
    _nanny_burn_phrase,
    _nanny_metered_since_delegate_activity,
    _nanny_reminder_due,
    _note_nanny_delegate_activity,
)


def _maybe_inject_nanny_economics_reminder(
    round_idx: int,
    messages: List[Dict[str, Any]],
    tools: ToolRegistry,
    emit_progress: Callable[[str], None],
    *,
    event_queue: Optional[queue.Queue] = None,
    task_id: str = "",
    drive_logs: Optional[pathlib.Path] = None,
) -> bool:
    """The periodic half of the nanny-economics reminder (poltergeist phase B).

    A plain user-message reminder in the self-checkpoint style — the loop's
    checkpoints are ordinary user turns, never protocol (ARCHITECTURE: "Loop
    self-checkpoints remain plain user-message reminders"). Fires between
    rounds, while the burn is happening — the finalization nudge alone
    arrives after the money is spent. Proportional, unbounded in count: each
    further threshold-width of metered rounds re-arms it (owner 2=B — no cap)."""
    ctx = tools._ctx
    if not getattr(ctx, "_nanny_route_dispatched", False):
        return False
    rounds, cost, due = _nanny_reminder_due(ctx, round_idx)
    if not due:
        return False
    # The fire cursor is the metered-progress mark AT this firing (round + cost),
    # so the dual-axis re-arm in `_nanny_reminder_due` measures both axes from
    # the same instant. Cleared on delegate activity.
    _progress_mark = getattr(ctx, "_nanny_metered_progress", None)
    ctx._nanny_reminder_mark = (dict(_progress_mark) if isinstance(_progress_mark, dict)
                                else {"round": int(round_idx), "cost": 0.0})
    # R2-7c: before the first delegate verb there IS no "last delegated-run
    # activity" — the burn is measured from task start, and the wording says
    # so instead of implying an activity that never happened.
    _baseline_known = isinstance(getattr(ctx, "_nanny_delegate_baseline", None), dict)
    if _baseline_known:
        since_phrase = (
            "since your last act of delegation (delegate_start / "
            "schedule_subagent), supervision included"
        )
    else:
        since_phrase = "since this task started (no act of delegation yet)"
    # BR1-3: never an unconditional "$0" claim — the owner's wording law is
    # typed cost classes: known-zero only on a settled $0 spend, never "free"
    # unqualified (estimated/undisclosed spend is never zero).
    reminder = (
        "[NANNY ECONOMICS REMINDER]\n"
        f"You are a harness-dispatched NANNY and you have spent {_nanny_burn_phrase(rounds, cost)} "
        f"{since_phrase}. A subscription-lane delegated run has known-zero "
        "marginal cost only when its settled spend reports $0 (estimated or "
        "undisclosed spend is never zero); every round you think yourself is "
        "metered API money.\n"
        "This is a reminder, not a stop. Consider: delegate the remaining work "
        "(delegate_start / delegate_wait — follow-up work and fixes are delegated too), "
        "and keep your own rounds for judgment: acceptance, integration, honest "
        "settlement. A deliberate switch_model raise for that judgment is "
        "sanctioned — finish it and drop back. If this work genuinely must run "
        "on metered tokens, continue deliberately and say why in your result."
    )
    _append_or_merge_user_message(messages, reminder)
    # Owner decision (2026-08-15): no owner-chat progress line — the model sees
    # the reminder and the typed task_checkpoint below carries observability.
    _emit_checkpoint_event(event_queue, task_id, drive_logs, {
        "checkpoint_kind": "nanny_economics_reminder",
        "round": round_idx,
        "metered_rounds_since_delegate_activity": rounds,
        "metered_cost_since_delegate_activity_usd": round(cost, 4),
    })
    return True


def _inject_round_checkpoints(
    *,
    round_idx: int,
    max_rounds: int,
    messages: List[Dict[str, Any]],
    accumulated_usage: Dict[str, Any],
    emit_progress: Callable[[str], None],
    tools: ToolRegistry,
    event_queue: Optional[queue.Queue],
    task_id: str,
    drive_logs: Optional[pathlib.Path],
    budget_remaining_usd: Optional[float] = None,
    cost_ceiling: Optional["task_pacing.CostCeiling"] = None,
) -> bool:
    """Inject the per-round self-check and the time-budget / intrinsic-pacing
    milestone AFTER owner messages, so the checkpoint is the LLM-call tail (a
    normal user turn). Returns whether any was injected (routine compaction is
    skipped that round when so)."""
    checkpoint = _maybe_inject_self_check(
        round_idx, max_rounds, messages, accumulated_usage, emit_progress,
        event_queue=event_queue, task_id=task_id, drive_logs=drive_logs,
    )
    time_budget = _maybe_inject_time_budget_milestone(
        messages, tools, event_queue=event_queue, task_id=task_id, drive_logs=drive_logs,
        round_idx=round_idx, accumulated_usage=accumulated_usage,
    )
    cost_budget = _maybe_inject_cost_budget_milestone(
        messages, tools,
        budget_remaining_usd=budget_remaining_usd, cost_ceiling=cost_ceiling,
        accumulated_usage=accumulated_usage,
        event_queue=event_queue, task_id=task_id, drive_logs=drive_logs,
    )
    nanny_economics = _maybe_inject_nanny_economics_reminder(
        round_idx, messages, tools, emit_progress,
        event_queue=event_queue, task_id=task_id, drive_logs=drive_logs,
    )
    return bool(checkpoint or time_budget or cost_budget or nanny_economics)


def seal_task_transcript(
    messages: List[Dict[str, Any]],
    keep_active: int = 5,
    min_prefix_tokens: int = 2048,
) -> None:
    """Mark one stable old tool-result boundary for provider prompt caching."""
    for msg in messages:
        if msg.get("role") != "tool":
            continue
        content = msg.get("content")
        if isinstance(content, list):
            # Flatten the old sealed boundary before choosing a new one.
            msg["content"] = _extract_plain_text_from_content(content)

    tool_indices = [
        i for i, m in enumerate(messages)
        if m.get("role") == "tool"
    ]
    if len(tool_indices) <= keep_active:
        return

    seal_candidate_idx = tool_indices[-(keep_active + 1)]

    prefix_text_len = sum(
        len(_extract_plain_text_from_content(m.get("content", "")))
        for m in messages[: seal_candidate_idx + 1]
    )
    prefix_tokens = prefix_text_len // 4  # rough 4-chars-per-token estimate

    if prefix_tokens < min_prefix_tokens:
        return

    candidate = messages[seal_candidate_idx]
    plain_text = str(candidate.get("content", ""))
    if not plain_text.strip():
        # Anthropic 400s on cache_control attached to an empty text block; never seal
        # an empty tool output as the cache anchor (turns the whole task unanswerable).
        plain_text = "(no tool output)"
    candidate["content"] = [
        {
            "type": "text",
            "text": plain_text,
            "cache_control": {"type": "ephemeral"},
        }
    ]


def _setup_dynamic_tools(tools_registry, tool_schemas, messages):
    """Attach list/enable tool handlers and mutate the active schema list."""
    enabled_extra: set = set()
    active_tool_names = {
        str(schema.get("function", {}).get("name") or "").strip()
        for schema in tool_schemas
        if str(schema.get("function", {}).get("name") or "").strip()
    }

    def _handle_list_tools(ctx=None, **kwargs):
        omissions = (
            tools_registry.capability_omissions()
            if hasattr(tools_registry, "capability_omissions")
            else []
        )
        non_core = [
            t for t in list_non_core_tools(tools_registry)
            if t["name"] not in active_tool_names
        ]
        if not non_core:
            if not omissions:
                return "All tools are already in your active set."
            lines = ["All currently discovered tools are already in your active set.", ""]
            lines.extend(format_capability_omissions(omissions))
            return "\n".join(lines)
        lines = [f"**{len(non_core)} additional tools available** (use `enable_tools` to activate):\n"]
        for t in non_core:
            lines.append(f"- **{t['name']}**: {t['description'][:120]}")
        if omissions:
            lines.extend(format_capability_omissions(
                omissions, header="\n" + CAPABILITY_OMISSION_HEADER,
            ))
        return "\n".join(lines)

    def _handle_enable_tools(ctx=None, tools: str = "", **kwargs):
        names = [n.strip() for n in tools.split(",") if n.strip()]
        enabled, hidden, not_found = [], [], []
        for name in names:
            schema = tools_registry.get_schema_by_name(name)
            if schema and name not in active_tool_names:
                tool_schemas.append(schema)
                enabled_extra.add(name)
                active_tool_names.add(name)
                enabled.append(f"{name} (registered late)")
            elif name in active_tool_names:
                enabled.append(f"{name} (already active)")
            else:
                # F3 (2026-08-10 saga): a policy-filtered tool is not "Not found" —
                # answer with the typed reason so the agent stops guessing names.
                reason = (
                    tools_registry.policy_hidden_reason(name)
                    if hasattr(tools_registry, "policy_hidden_reason") else None
                )
                if reason:
                    hidden.append(f"{name} — {reason}")
                else:
                    not_found.append(name)
        parts = []
        if enabled:
            parts.append(
                "✅ Tools are registered in the active capability envelope: "
                + ", ".join(enabled)
            )
        if hidden:
            parts.append(
                "🚫 Hidden by policy (the tool exists but this task cannot use it): "
                + "; ".join(hidden)
            )
        if not_found:
            parts.append(f"❌ Not found: {', '.join(not_found)}")
        return "\n".join(parts) if parts else "No tools specified."

    tools_registry.override_handler("list_available_tools", _handle_list_tools)
    tools_registry.override_handler("enable_tools", _handle_enable_tools)

    non_core_count = len(list_non_core_tools(tools_registry))
    if non_core_count > 0:
        _append_or_merge_user_message(
            messages,
            (
                "[SYSTEM NOTICE]\n"
                f"You have {len(tool_schemas)} core tools loaded. "
                f"There are {non_core_count} additional tools available "
                f"(use `list_available_tools` to see them, `enable_tools` to activate). "
                f"Core tools cover most tasks. Enable extras only when needed."
            ),
        )
    omissions = (
        tools_registry.capability_omissions()
        if hasattr(tools_registry, "capability_omissions")
        else []
    )
    if omissions:
        _append_or_merge_user_message(
            messages,
            "[SYSTEM NOTICE]\n" + "\n".join(format_capability_omissions(omissions)),
        )

    return tool_schemas, enabled_extra


def _drain_incoming_messages(
    messages: List[Dict[str, Any]],
    incoming_messages: queue.Queue,
    drive_root: Optional[pathlib.Path],
    task_id: str,
    event_queue: Optional[queue.Queue],
    _owner_msg_seen: set,
    owner_ctx: Any = None,
) -> Dict[str, Any]:
    """Injects dialogue; returns typed controls."""
    controls: Dict[str, Any] = {}
    while not incoming_messages.empty():
        try:
            injected = incoming_messages.get_nowait()
            if isinstance(injected, dict):
                owner_content = build_user_content(injected)
                _record_owner_directive(
                    owner_ctx,
                    source="direct_incoming",
                    content=owner_content,
                    msg_id=str(
                        injected.get("client_message_id")
                        or injected.get("msg_id")
                        or ""
                    ),
                )
                _append_or_merge_user_content(messages, _owner_marked_content(owner_content))
            else:
                _record_owner_directive(
                    owner_ctx, source="direct_incoming", content=injected,
                )
                _append_or_merge_user_message(messages, _owner_marked_content(injected))
        except queue.Empty:
            break

    if drive_root is not None and task_id:
        from ouroboros.owner_mailbox import KIND_FINALIZE_NOW, KIND_HURRY, KIND_OWNER_TEXT, KIND_TASK_MESSAGE, acknowledge_transcript_entry, deliver_task_message, drain_owner_entries

        if owner_ctx:
            owner_ctx._loop_mailbox_seen_ids = _owner_msg_seen
        attempt = getattr(owner_ctx, "task_attempt", None) or (1 if owner_ctx is not None else None)
        for entry in drain_owner_entries(drive_root, task_id, _owner_msg_seen, attempt):
            kind = entry.get("kind") or KIND_OWNER_TEXT
            if kind == KIND_FINALIZE_NOW:
                text = str(entry.get("text") or "deadline")
                first_line = text.splitlines()[0].strip() if text else ""
                if first_line == REASON_OWNER_REQUESTED_FINALIZATION:
                    if not _owner_stop_control_is_current(
                        owner_ctx,
                        drive_root,
                        task_id,
                        str(entry.get("msg_id") or ""),
                    ):
                        continue
                    # Owner-stop budget starts at delivery; first drain wins.
                    if not _mark_owner_stop_control_drained(
                        owner_ctx, drive_root, task_id,
                    ):
                        continue
                    if not _owner_stop_control_is_current(
                        owner_ctx,
                        drive_root,
                        task_id,
                        str(entry.get("msg_id") or ""),
                    ):
                        continue
                else:
                    opened = parse_deadline_ts(entry.get("ts"))
                    if opened is not None:
                        controls["finalize_deadline_ts"] = (
                            opened.timestamp()
                            + task_pacing.effective_finalization_reserve_sec(owner_ctx)
                        )
                controls["finalize_now"] = text
                continue
            if kind == KIND_HURRY:
                # HQ1 no-chat contract (§19.7.2 item 6): a typed hurry
                # control routes structurally — never via
                # _record_owner_directive, _owner_marked_content, messages, or
                # owner_message_injected.
                from ouroboros.owner_hurry import apply_latch

                apply_latch(owner_ctx, entry, event_queue=event_queue)
                controls["hurry"] = str(entry.get("msg_id") or "hurry")
                continue
            dmsg = entry.get("text") or ""
            if kind == KIND_TASK_MESSAGE:
                deliver_task_message(entry, task_id, event_queue, lambda text: _append_or_merge_user_message(messages, text))
                acknowledge_transcript_entry(drive_root, task_id, entry)
                continue
            _record_owner_directive(
                owner_ctx,
                source="owner_mailbox",
                content=dmsg,
                msg_id=str(entry.get("msg_id") or ""),
            )
            from ouroboros.client_surface import noted_owner_text

            _append_or_merge_user_message(messages, _owner_marked_content(noted_owner_text(owner_ctx, entry, dmsg)))
            acknowledge_transcript_entry(drive_root, task_id, entry)
            if event_queue is not None:
                try:
                    event_queue.put_nowait({
                        "type": "owner_message_injected",
                        "task_id": task_id,
                        "text": dmsg,
                    })
                except Exception:
                    pass
    return controls


def _context_reclaim_passes(tool_ctx: Any) -> set[Tuple[str, str]]:
    passes = getattr(tool_ctx, "_context_reclaim_passes", None)
    if not isinstance(passes, set):
        passes = set()
        tool_ctx._context_reclaim_passes = passes
    return passes


def _context_reclaim_materializations(tool_ctx: Any) -> set[Tuple[str, str]]:
    materialized = getattr(tool_ctx, "_context_reclaim_materializations", None)
    if not isinstance(materialized, set):
        materialized = set()
        tool_ctx._context_reclaim_materializations = materialized
    return materialized


def _context_overflow_retries(tool_ctx: Any) -> set[Tuple[str, str]]:
    retries = getattr(tool_ctx, "_context_overflow_retries", None)
    if not isinstance(retries, set):
        retries = set()
        tool_ctx._context_overflow_retries = retries
    return retries


def _run_round_compaction(
    messages: List[Dict[str, Any]],
    ctx: _CompactionRoundContext,
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Run only an explicit manual reclaim; Main fit owns automatic decisions."""
    pending = getattr(ctx.tools._ctx, "_pending_compaction", None)
    if pending is None:
        return messages, None
    ctx.tools._ctx._pending_compaction = None
    rebuilt, receipt, usage = compact_tool_history_llm(
        messages,
        keep_recent=max(0, int(pending)),
        drive_root=ctx.drive_root or pathlib.Path(ctx.drive_logs).parent,
        task_id=ctx.task_id,
        negative_memo=reclaim_negative_memo(ctx.tools._ctx),
        trace_refs_by_tool_call_id=reclaim_trace_refs(ctx.tools._ctx),
    )
    _emit_checkpoint_event(ctx.event_queue, ctx.task_id, ctx.drive_logs, {
        "checkpoint_kind": "context_reclaim_manual",
        "round": ctx.round_idx,
        "status": receipt.status,
        "reclaimed_tokens": receipt.reclaimed_tokens,
        "goal_reached": receipt.goal_reached,
        "checkpoint_ref": receipt.checkpoint_ref,
    })
    if receipt.status in {"checkpoint_failed", "summarizer_failed", "binding_mismatch"}:
        ctx.emit_progress(
            f"⚠️ Context compaction kept the transcript unchanged ({receipt.status})."
        )
    if receipt.status == "applied":
        prune_reclaim_trace_refs(ctx.tools._ctx, rebuilt)
    return rebuilt, usage


@dataclass
class _RoundLimitContext:
    messages: List[Dict[str, Any]]
    llm: LLMClient
    active_model: str
    active_effort: str
    max_retries: int
    drive_logs: pathlib.Path
    task_id: str
    round_idx: int
    event_queue: Optional[queue.Queue]
    accumulated_usage: Dict[str, Any]
    task_type: str
    active_use_local: bool
    max_rounds: int
    deadline_ts: Optional[float] = None
    # Drive root for durable salvage (latest_llm_response_text) on the provider-death
    # path; optional so existing positional construction stays valid.
    drive_root: Optional[pathlib.Path] = None
    # STATUS/budget drive root + root task id for the forced-finalization
    # orphan note: child results live under the parent BUDGET drive, not a
    # (possibly forked) drive_root — same root get_task_result uses.
    status_drive_root: Optional[pathlib.Path] = None
    root_task_id: str = ""
    delivery_candidate: Optional[DeliveryCandidate] = None
    tools: Optional[ToolRegistry] = None
    llm_trace: Optional[Dict[str, Any]] = None
    incoming_messages: Optional[queue.Queue] = None
    owner_msg_seen: Optional[set] = None
    forced_service_evidence_fingerprint: str = ""


def _account_compaction_usage(
    accumulated_usage: Dict[str, Any],
    compaction_usage: Dict[str, Any],
    event_queue: Optional[queue.Queue],
    task_id: str,
) -> None:
    """Fold a compaction pass's usage into the loop totals and emit its llm_usage
    event (light-model lane). Extracted verbatim from ``run_llm_loop`` for the
    300-line function gate; behavior unchanged."""
    add_usage(accumulated_usage, compaction_usage)
    _cm = get_light_model()
    _cc = (
        float(compaction_usage["cost"])
        if compaction_usage.get("cost") is not None
        else estimate_cost_optional(
            _cm,
            int(compaction_usage.get("prompt_tokens") or 0),
            int(compaction_usage.get("completion_tokens") or 0),
            cache_usage={
                "cached_tokens": int(compaction_usage.get("cached_tokens") or 0),
                "cache_write_tokens": int(compaction_usage.get("cache_write_tokens") or 0),
                "prompt_cache_ttl": compaction_usage.get("prompt_cache_ttl"),
            },
            provider=str(compaction_usage.get("provider") or "openrouter"),
        )
    )
    emit_llm_usage_event(event_queue, task_id, _cm, compaction_usage, _cc, "compaction")


def _handle_round_limit(ctx: _RoundLimitContext) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    finish_reason = f"⚠️ Task exceeded MAX_ROUNDS ({ctx.max_rounds}). Consider decomposing into subtasks via schedule_subagent."
    prompt = (
        f"[ROUND_LIMIT] {finish_reason} Produce your best final answer now from the "
        "verified work so far; clearly mark anything unverified or incomplete. An honest "
        "best-effort result is the expected outcome here, not a failure."
    )
    return _forced_final_answer(ctx, prompt=prompt, fallback_text=finish_reason, reason_code="round_limit")


def _handle_forced_finalization(ctx: _RoundLimitContext, reason: str) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """Finalize within the cooperative grace window."""
    reason_lines = str(reason or "").splitlines()
    if reason_lines and reason_lines[0].strip() == REASON_OWNER_REQUESTED_FINALIZATION:
        return _handle_owner_stop_finalization(ctx, str(reason))
    fallback = f"⚠️ Task reached {reason or 'deadline'}; finalization grace produced no answer."
    prompt = (
        f"[FINALIZE_NOW] The supervisor opened a finalization grace window (reason: {reason or 'deadline'}). "
        "The task will be stopped shortly. Produce your best final answer NOW from the verified "
        "work so far; clearly mark anything unverified or incomplete. An honest best-effort "
        "result is the expected outcome here, not a failure."
    )
    return _forced_final_answer(ctx, prompt=prompt, fallback_text=fallback, reason_code="finalization_grace")


def _handle_owner_stop_finalization(
    ctx: _RoundLimitContext, control_text: str,
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """Owner-requested finalization: zero or one tool-less model turn."""
    live_trace = getattr(ctx, "llm_trace", None)
    llm_trace = live_trace if isinstance(live_trace, dict) else {}
    candidate = _current_delivery_candidate(ctx, llm_trace)
    if candidate is not None:
        _finalize_forced_services(ctx, llm_trace)
        ctx.accumulated_usage["execution_status"] = "failed"
        ctx.accumulated_usage["reason_code"] = REASON_OWNER_REQUESTED_FINALIZATION
        return _forced_fallback_result(
            ctx, llm_trace, candidate.full_text, REASON_OWNER_REQUESTED_FINALIZATION,
            retained_source="owner_stop_retained_candidate",
        )
    fallback = (
        "⚠️ The owner requested finalize-then-stop; no final answer could be "
        "produced inside the grace window."
    )
    if _owner_stop_window_elapsed(ctx):
        _finalize_forced_services(ctx, llm_trace)
        ctx.accumulated_usage["execution_status"] = "failed"
        ctx.accumulated_usage["reason_code"] = REASON_OWNER_REQUESTED_FINALIZATION
        return _forced_fallback_result(
            ctx, llm_trace, fallback, REASON_OWNER_REQUESTED_FINALIZATION,
            source="owner_stop_window_elapsed",
        )
    child_block = "\n".join(str(control_text or "").splitlines()[1:]).strip()
    prompt = (
        "[OWNER_STOP] The owner asked this task to summarize and stop now. "
        "Produce your best final answer NOW from the verified work so far; "
        "clearly mark anything unverified or incomplete. An honest best-effort "
        "result is the expected outcome here, not a failure. Do not start new work."
        + (f"\n\n{child_block}" if child_block else "")
    )
    return _forced_final_answer(
        ctx, prompt=prompt, fallback_text=fallback,
        reason_code="owner_requested_finalization", single_semantic_turn=True,
    )


def _handle_provider_unavailable(
    ctx: _RoundLimitContext, *, error_kind: str = "provider_unavailable",
    wait_cause: str = "", waited: bool = False, wait_eligible: bool = True,
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """Provider-death rail wrapper: every arm carries a terminal provenance;
    ``setdefault`` keeps explicit stamps authoritative. Not every exit is a
    provider death: the deadline grace arm can return a MODEL-AUTHORED final
    (``deadline_local``) and a scheduled swarm handoff pops its reason code —
    both keep their legacy shape."""
    text, usage, llm_trace = _provider_unavailable_result(
        ctx, error_kind=error_kind, wait_cause=wait_cause, waited=waited,
        wait_eligible=wait_eligible,
    )
    if str(usage.get("reason_code") or "") not in ("", "deadline_local"):
        usage.setdefault("terminal_origin", TERMINAL_ORIGIN_HOST_SALVAGE)
    return text, usage, llm_trace


def _provider_unavailable_result(
    ctx: _RoundLimitContext, *, error_kind: str = "provider_unavailable",
    wait_cause: str = "", waited: bool = False, wait_eligible: bool = True,
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """Salvage provider failure without an unsafe retry. ``wait_cause`` is the
    transport-wait episode's latched cause (survives later overwrites of the
    mutable ``_last_llm_error_kind``); ``waited``/``wait_eligible`` keep the
    terminal text honest for zero-wait turns."""
    kind = str(error_kind or "")
    is_context_overflow = kind == "context_overflow"
    is_transport_wait = str(wait_cause or "") == "transport_unavailable"
    is_deadline_exhausted = kind == "deadline_exhausted" or str(ctx.accumulated_usage.get("_last_llm_error_kind") or "") == "deadline_exhausted"
    forced_reason = "deadline_local" if is_deadline_exhausted else "provider_unavailable"
    llm_trace = getattr(ctx, "llm_trace", None)
    llm_trace = llm_trace if isinstance(llm_trace, dict) else {}
    candidate = _live_delivery_candidate(ctx)
    salvaged = candidate.full_text if candidate is not None else _last_assistant_text(ctx.messages)
    if candidate is None and not salvaged and ctx.drive_root is not None:
        try:
            from ouroboros.observability import latest_llm_response_text
            salvaged = latest_llm_response_text(pathlib.Path(ctx.drive_root), ctx.task_id) or ""
        except Exception:
            log.debug("latest_llm_response_text salvage failed", exc_info=True)
    if salvaged:
        fallback = salvaged
    else:
        fallback = _provider_terminal_fallback_text(
            ctx.accumulated_usage, is_context_overflow=is_context_overflow,
            is_transport_wait=is_transport_wait, waited=waited,
            wait_eligible=wait_eligible,
            is_deadline_exhausted=is_deadline_exhausted,
        )
    if is_context_overflow:
        text, usage, llm_trace = _forced_fallback_result(
            ctx, llm_trace, fallback, "context_overflow",
            source="context_overflow_local_salvage",
        )
        usage.update(
            execution_status=RESULT_INFRA_FAILED,
            reason_code="llm_api_error",
            _last_llm_error_kind="context_overflow",
        )
        return text, usage, llm_trace
    if is_transport_wait:
        # No-resend terminal: salvage, no forced-final call over a dead
        # egress. Stamp BEFORE the composer (owner-stop pattern): a SCHEDULED
        # swarm handoff clears it; guard mirrors the sibling below.
        live_trace = getattr(ctx, "llm_trace", None)
        llm_trace = live_trace if isinstance(live_trace, dict) else {}
        ctx.accumulated_usage["execution_status"] = RESULT_INFRA_FAILED
        ctx.accumulated_usage["reason_code"] = "provider_unavailable"
        text, usage, llm_trace = _forced_fallback_result(
            ctx, llm_trace, fallback, reason_code="provider_unavailable",
            source="transport_unavailable_no_resend",
        )
        if str(usage.get("reason_code") or "") == "provider_unavailable":
            usage["execution_status"] = RESULT_INFRA_FAILED
        return text, usage, llm_trace
    # No-call shapes; see provider_no_call_source
    no_call, wall = provider_no_call_source(ctx.accumulated_usage, is_deadline_exhausted)
    if no_call:
        if wall:
            _finalize_forced_services(ctx, llm_trace)
            _drain_forced_owner_directives(ctx, llm_trace)
        text, usage, llm_trace = _forced_fallback_result(
            ctx, llm_trace, fallback, reason_code="provider_unavailable",
            source=no_call, provider_terminal=wall,
        )
        if usage.get("execution_status") is not None:
            usage.update(execution_status=RESULT_INFRA_FAILED, reason_code="provider_unavailable")
        return text, usage, llm_trace
    prompt = (
        "[DEADLINE] Primary model work reached the owner deadline. Produce the best final answer now from verified work and state what remains undone."
        if is_deadline_exhausted else
        "[PROVIDER_UNAVAILABLE] The model provider failed to return a usable response. "
        "The task is being INTERRUPTED by this outage, not completed. Summarize the "
        "verified work so far and state plainly what remains undone."
    )
    text, usage, llm_trace = _forced_final_answer(
        ctx, prompt=prompt, fallback_text=fallback, reason_code=forced_reason,
        provider_terminal=not is_deadline_exhausted,
    )
    if not is_deadline_exhausted and str(usage.get("reason_code") or "") == "provider_unavailable":
        usage["execution_status"] = RESULT_INFRA_FAILED
        usage.setdefault("terminal_origin", TERMINAL_ORIGIN_HOST_SALVAGE)
    return text, usage, llm_trace


def _maybe_deadline_local_finalize(
    ctx: _RoundLimitContext, tools: ToolRegistry
) -> Optional[Tuple[str, Dict[str, Any], Dict[str, Any]]]:
    """Loop-local graceful finalization on a REAL task deadline.

    Headless runs (benchmarks, harbor) often get no supervisor finalize_now:
    the process is killed at the deadline, discarding any best-effort
    artifact. With a real deadline_at set and less than the
    finalization-grace window left, self-finalize one tool-less best answer —
    independent of the supervisor — so a deadline NEVER returns emptiness.
    Never fires without a real deadline_at (no synthesized deadline;
    leaderboard timeouts stay legal)."""
    meta = getattr(tools._ctx, "task_metadata", {})
    if not isinstance(meta, dict):
        return None
    deadline = parse_deadline_ts(meta.get("deadline_at"))
    if deadline is None:
        return None
    ctx.deadline_ts = deadline.timestamp()
    remaining = (deadline - utc_now()).total_seconds()
    if remaining > task_pacing.effective_finalization_reserve_sec(tools._ctx):
        return None
    prompt = (
        f"[DEADLINE] The task deadline ({meta.get('deadline_at')}) is ~{max(0.0, remaining)/60:.1f} min away "
        "and the run will stop at it. Produce your best final answer NOW from the verified work so far; "
        "clearly mark anything unverified or incomplete. An honest best-effort result is the expected "
        "outcome here, not a failure."
    )
    fallback = "⚠️ Task reached its deadline; local finalization produced no answer."
    return _forced_final_answer(ctx, prompt=prompt, fallback_text=fallback, reason_code="deadline_local")


def _maybe_early_finalize(
    limit_ctx: _RoundLimitContext, tools: ToolRegistry, controls: Dict[str, Any],
    *, transport_episode: Optional[TransportWaitEpisode] = None,
) -> Optional[Tuple[str, Dict[str, Any], Dict[str, Any]]]:
    """Consume supervisor grace first, then a local deadline."""
    if controls.get("finalize_now"):
        if transport_episode is not None:
            # Every finalize_now flavor during an active outage takes the
            # honest no-resend terminal (rationale in the helper); control
            # bookkeeping was done by the drain, terminalization is prompt.
            return _finalize_now_transport_terminal(
                transport_episode, drive_logs=limit_ctx.drive_logs,
                task_id=limit_ctx.task_id, model=limit_ctx.active_model,
                handle_provider_unavailable=functools.partial(
                    _handle_provider_unavailable, limit_ctx),
            )
        if controls.get("finalize_deadline_ts") is not None:
            _narrow_round_deadline(
                limit_ctx, controls["finalize_deadline_ts"],
            )
        return _handle_forced_finalization(limit_ctx, str(controls["finalize_now"]))
    if transport_episode is not None:
        # An active episode owns the deadline sliver: its last free redial +
        # no-resend terminal replace the paid deadline_local finalize call.
        return None
    return _maybe_deadline_local_finalize(limit_ctx, tools)


def _finalize_limit_ctx(
    ctx: "_RoundLimitContext",
    tools: Any,
    llm_trace: Optional[Dict[str, Any]] = None,
) -> "_RoundLimitContext":
    """Resolve the deadline + STATUS/budget drive root + root task id from the live
    ToolContext onto an already-constructed round-limit context (child results live under
    the parent BUDGET drive, not the forked drive_root), then attach the live tool/trace
    references needed to publish a forced DeliveryCandidate. Returns the same (mutated)
    context."""
    meta = getattr(tools._ctx, "task_metadata", {}) if isinstance(getattr(tools._ctx, "task_metadata", {}), dict) else {}
    ctx.deadline_ts = _task_deadline_epoch(tools)
    ctx.status_drive_root = pathlib.Path(
        str(meta.get("budget_drive_root") or getattr(tools._ctx, "budget_drive_root", "") or "")
        or (ctx.drive_root if ctx.drive_root is not None else pathlib.Path(ctx.drive_logs).parent)
    )
    ctx.root_task_id = str(meta.get("root_task_id") or ctx.task_id)
    candidate = getattr(tools._ctx, "_delivery_candidate", None)
    ctx.delivery_candidate = candidate if isinstance(candidate, DeliveryCandidate) else None
    ctx.tools = tools
    ctx.llm_trace = llm_trace
    return ctx


def _direct_child_results(ctx: _RoundLimitContext) -> list[Dict[str, Any]]:
    """Read this node's direct children from the existing task-status authority."""

    try:
        status_root = ctx.status_drive_root or ctx.drive_root or pathlib.Path(ctx.drive_logs).parent
        if status_root is None or not ctx.task_id:
            return []
        return _load_direct_child_results(
            pathlib.Path(status_root),
            ctx.task_id,
            str(ctx.root_task_id or ctx.task_id),
        )
    except Exception:
        return []


def _child_disposition_state(child: Dict[str, Any]) -> str:
    """Return cancellation or the current task-tree exact-hash disposition."""

    # Explicit cancellation is lifecycle authority and wins every completion
    # race; late scratch results are not projected or recovered. Only a SETTLED
    # ``cancelled`` counts as handled (GR2-8c): ``cancel_requested`` is intent,
    # not outcome — treating it as done suppressed the handoff reminder, so
    # such a child stays cancel-pending until custody settles.
    if (
        str(child.get("parent_decision") or "").strip().lower() == "cancelled"
        and str(child.get("status") or "").strip().lower() == "cancelled"
    ):
        return "cancelled"
    try:
        from ouroboros.tools.join_ledger import _current_child_result_disposition

        current = _current_child_result_disposition(child)
        if current:
            return current
    except Exception:
        pass
    return ""


def _project_child_result_dispositions(
    ctx: _RoundLimitContext,
    llm_trace: Dict[str, Any],
) -> None:
    """Expose a compact exact-hash projection for acceptance/outcome reducers."""

    try:
        from ouroboros.tools.join_ledger import _child_result_sha256

        current = []
        for child in _direct_child_results(ctx):
            disposition = _child_disposition_state(child)
            if disposition not in {"integrated", "irrelevant", "deferred"}:
                continue
            current.append({
                "child_task_id": str(child.get("task_id") or child.get("id") or ""),
                "disposition": disposition,
                "child_result_sha256": _child_result_sha256(child),
            })
        llm_trace["child_result_dispositions"] = {
            "current": current,
            "deferred_count": sum(row["disposition"] == "deferred" for row in current),
        }
    except Exception:
        llm_trace["child_result_dispositions"] = {"current": [], "deferred_count": 0}


def _delivery_evidence_state(
    tools: ToolRegistry,
    ctx: _RoundLimitContext,
    llm_trace: Dict[str, Any],
) -> tuple[int, str]:
    """Fingerprint only evidence that can invalidate a complete answer."""

    from ouroboros.outcomes import read_context_verification_receipts
    from ouroboros.tools.join_ledger import _child_result_sha256

    owner_directives = getattr(tools._ctx, "_owner_directives", [])
    owner_directives = owner_directives if isinstance(owner_directives, list) else []
    children = []
    for child in _direct_child_results(ctx):
        children.append({
            "task_id": str(child.get("task_id") or child.get("id") or ""),
            "status": str(child.get("status") or ""),
            "sha256": _child_result_sha256(child),
            "disposition": _child_disposition_state(child),
        })
    receipt_root = pathlib.Path(
        str(
            getattr(tools._ctx, "drive_root", "")
            or ctx.drive_root
            or ctx.status_drive_root
            or ctx.drive_logs.parent
        )
    )
    evidence = {
        "owner_directives": owner_directives,
        "tool_effects": reviewable_effect_projection(llm_trace),
        # The typed plan-review control is not a filesystem effect, but it
        # changes whether a pre-plan answer is grounded.
        "plan_review_receipts": [
            {
                "index": index,
                "outcome": call.get("plan_review_outcome"),
                "closed": call.get("plan_review_closed"),
                "result": call.get("result"),
            }
            for index, call in enumerate(llm_trace.get("tool_calls") or [])
            if isinstance(call, dict) and call.get("plan_review_outcome")
        ],
        "children": children,
        "verification_receipts": read_context_verification_receipts(
            tools._ctx, ctx.task_id, fallback_root=receipt_root,
        ),
        # Task-scoped service teardown can register declared outputs or
        # surface an output-finalization failure. Those facts arise outside an
        # ordinary tool call, so bind their stable projection explicitly; else
        # a host acceptance panel could review the pre-teardown state.
        "service_finalization": _service_finalization_evidence(llm_trace),
    }
    fingerprint = hashlib.sha256(json.dumps(
        evidence,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")).hexdigest()
    previous = str(getattr(tools._ctx, "_delivery_evidence_fingerprint", "") or "")
    revision = int(getattr(tools._ctx, "_delivery_evidence_revision", 0) or 0)
    if fingerprint != previous:
        candidate = getattr(tools._ctx, "_delivery_candidate", None)
        if (
            isinstance(candidate, DeliveryCandidate)
            and bool(candidate.evidence_fingerprint)
            and candidate.evidence_fingerprint != fingerprint
        ):
            _supersede_delivery_acceptance_binding(
                tools,
                llm_trace,
                candidate,
                reason="delivery_evidence_changed_after_host_acceptance",
            )
        revision += 1
        tools._ctx._delivery_evidence_fingerprint = fingerprint
        tools._ctx._delivery_evidence_revision = revision
    return revision, fingerprint


def _service_finalization_evidence(llm_trace: Dict[str, Any]) -> list[Dict[str, Any]]:
    """Return the stable, answer-relevant part of service finalization events."""

    rows: list[Dict[str, Any]] = []
    stable_fields = (
        "service_id",
        "name",
        "task_id",
        "lifecycle",
        "backend",
        "pid",
        "port",
        "artifact_outputs",
        "artifact_output_failed",
        "artifact_audit_gap",
        "log_finalization",
    )
    for event in llm_trace.get("verification_events") or []:
        if not isinstance(event, dict) or str(event.get("kind") or "") not in {
            "services_stopped",
            "services_kept",
            "service_finalization_error",
        }:
            continue
        services = []
        for service in event.get("services") or []:
            if not isinstance(service, dict):
                continue
            services.append({
                key: service.get(key)
                for key in stable_fields
                if service.get(key) not in (None, "", [], {})
            })
        rows.append({
            "kind": str(event.get("kind") or ""),
            "services": services,
            "error": str(event.get("error") or ""),
        })
    return rows


def _unaccepted_delivery_binding(
    tools: ToolRegistry,
    candidate_hash: str,
) -> Dict[str, Any]:
    fence_value = str(
        getattr(tools._ctx, "_task_acceptance_sealed_fence_token", "")
        or "unsealed"
    )
    return {
        "candidate_sha256": candidate_hash,
        "evidence_revision": int(getattr(tools._ctx, "_delivery_evidence_revision", 0) or 0),
        "acceptance_status": "unaccepted",
        "authoritative": False,
        "panel_id": "",
        "binding_hash": "",
        "fence_hash": hashlib.sha256(fence_value.encode("utf-8")).hexdigest(),
    }


def _delivery_acceptance_binding(
    tools: ToolRegistry,
    llm_trace: Dict[str, Any],
    candidate_hash: str,
) -> Dict[str, Any]:
    """Refresh a candidate from one exact, complete, active host-root verdict."""

    binding = _unaccepted_delivery_binding(tools, candidate_hash)
    review_decision = llm_trace.get("review_decision") if isinstance(llm_trace.get("review_decision"), dict) else {}
    expected_panel = str(review_decision.get("panel_id") or "")
    expected_binding = str(review_decision.get("binding_hash") or "")
    # Candidate text alone is not a review identity: the same full answer
    # can be regenerated after tool/child/verification evidence changes.
    # Refresh host authority only from the panel this pass names; an older
    # exact-text run must never be rediscovered by hash-only scan.
    if not expected_panel or not expected_binding:
        return binding
    for raw_run in reversed(llm_trace.get("review_runs") or []):
        if not isinstance(raw_run, dict):
            continue
        if raw_run.get("authority") != "host_root" or raw_run.get("superseded_by_revision"):
            continue
        run_candidate = str(
            raw_run.get("candidate_hash") or raw_run.get("candidate_sha256") or ""
        )
        if run_candidate != candidate_hash:
            continue
        run_panel = str(raw_run.get("panel_id") or "")
        run_binding = str(raw_run.get("binding_hash") or "")
        if not run_panel or not run_binding:
            continue
        if run_panel != expected_panel:
            continue
        if run_binding != expected_binding:
            continue
        verdict = str(
            raw_run.get("aggregate_signal") or raw_run.get("semantic_verdict") or ""
        ).strip().lower()
        if not verdict:
            continue
        binding.update({
            "acceptance_status": verdict,
            "authoritative": True,
            "panel_id": run_panel,
            "binding_hash": run_binding,
            "fence_hash": str(raw_run.get("fence_hash") or binding["fence_hash"]),
            "review_evidence_revision": str(raw_run.get("evidence_revision") or ""),
        })
        break
    return binding


def _publish_delivery_candidate(
    tools: ToolRegistry,
    candidate: DeliveryCandidate,
    llm_trace: Dict[str, Any],
) -> None:
    """Publish hashes/control state only; the complete text remains loop-local."""

    current_fp = str(getattr(tools._ctx, "_delivery_evidence_fingerprint", "") or "")
    llm_trace["delivery_candidate"] = {
        "content_sha256": candidate.content_sha256,
        "revision": candidate.revision,
        "evidence_revision": candidate.evidence_revision,
        "evidence_fingerprint": candidate.evidence_fingerprint,
        "evidence_current": candidate.evidence_fingerprint == current_fp,
        "acceptance_binding": dict(candidate.acceptance_binding),
        "finalization_control": candidate.finalization_control,
        "degraded": candidate.degraded,
        "degraded_reason": candidate.degraded_reason,
    }


def _replace_delivery_candidate(
    tools: ToolRegistry,
    ctx: _RoundLimitContext,
    llm_trace: Dict[str, Any],
    full_text: str,
    *,
    control: str,
    model_text: Optional[str] = None,
) -> DeliveryCandidate:
    full_text = sanitize_tool_result_for_log(full_text)
    model_text = sanitize_tool_result_for_log(
        full_text if model_text is None else model_text
    )
    previous_candidate = getattr(tools._ctx, "_delivery_candidate", None)
    if isinstance(previous_candidate, DeliveryCandidate):
        _supersede_delivery_acceptance_binding(
            tools,
            llm_trace,
            previous_candidate,
            reason="delivery_candidate_replaced",
        )
    evidence_revision, evidence_fingerprint = _delivery_evidence_state(tools, ctx, llm_trace)
    content_hash = hashlib.sha256(full_text.encode("utf-8")).hexdigest()
    revision = int(getattr(tools._ctx, "_delivery_candidate_revision", 0) or 0) + 1
    tools._ctx._delivery_candidate_revision = revision
    candidate = DeliveryCandidate(
        full_text=full_text,
        content_sha256=content_hash,
        revision=revision,
        evidence_revision=evidence_revision,
        evidence_fingerprint=evidence_fingerprint,
        acceptance_binding=_unaccepted_delivery_binding(tools, content_hash),
        finalization_control=control,
        model_text=model_text,
    )
    tools._ctx._delivery_candidate = candidate
    tools._ctx._delivery_control_required = False
    _publish_delivery_candidate(tools, candidate, llm_trace)
    return candidate


def _ensure_explicit_acceptance_binding(candidate: DeliveryCandidate) -> None:
    """Keep an exact historical binding, or state explicitly that none exists."""

    binding = dict(candidate.acceptance_binding or {})
    if binding.get("authoritative") is not True:
        binding.update({
            "acceptance_status": "unaccepted",
            "authoritative": False,
            "panel_id": "",
            "binding_hash": "",
        })
        binding.pop("review_evidence_revision", None)
    candidate.acceptance_binding = binding


def _forced_unaccepted_binding(
    tools: ToolRegistry,
    candidate: DeliveryCandidate,
    reason_code: str,
) -> Dict[str, Any]:
    """Bind a newly generated forced answer without borrowing an older verdict."""

    binding = _unaccepted_delivery_binding(tools, candidate.content_sha256)
    binding.update({
        "acceptance_status": "unaccepted",
        "authoritative": False,
        "degraded": True,
        "degraded_reason": reason_code,
        "panel_id": "",
        "binding_hash": "",
    })
    binding.pop("review_evidence_revision", None)
    return binding


def _live_delivery_candidate(ctx: _RoundLimitContext) -> Optional[DeliveryCandidate]:
    tools = getattr(ctx, "tools", None)
    if tools is not None:
        candidate = getattr(tools._ctx, "_delivery_candidate", None)
        if isinstance(candidate, DeliveryCandidate):
            return candidate
    candidate = getattr(ctx, "delivery_candidate", None)
    return candidate if isinstance(candidate, DeliveryCandidate) else None


def _current_delivery_candidate(
    ctx: Optional[_RoundLimitContext],
    llm_trace: Dict[str, Any],
) -> Optional[DeliveryCandidate]:
    """Return a retained answer only after checking live answer-invalidating evidence."""

    if ctx is None or getattr(ctx, "tools", None) is None:
        return None
    candidate = _live_delivery_candidate(ctx)
    if candidate is None:
        return None
    evidence_revision, evidence_fingerprint = _delivery_evidence_state(
        ctx.tools, ctx, llm_trace,
    )
    if (
        candidate.evidence_revision != evidence_revision
        or candidate.evidence_fingerprint != evidence_fingerprint
    ):
        return None
    return candidate


def _degrade_retained_delivery_candidate(
    ctx: _RoundLimitContext,
    llm_trace: Dict[str, Any],
    candidate: DeliveryCandidate,
    *,
    control: str,
    reason_code: str,
) -> DeliveryCandidate:
    """Publish a current unchanged candidate while preserving its exact verdict binding."""

    candidate.degraded = True
    candidate.degraded_reason = reason_code
    candidate.finalization_control = control
    _ensure_explicit_acceptance_binding(candidate)
    tools = getattr(ctx, "tools", None)
    if tools is not None:
        _publish_delivery_candidate(tools, candidate, llm_trace)
    ctx.delivery_candidate = candidate
    return candidate


def _record_forced_acceptance_bypass(
    ctx: _RoundLimitContext,
    llm_trace: Dict[str, Any],
    reason_code: str,
) -> None:
    """Typed acceptance-bypass record on a forced rail — a LEDGER write, never a gate.

    The panel's only launch site is the voluntary no-tool finalization, so
    forced exits used to leave the review axis at {skipped, not_eligible,
    run_count:0} — indistinguishable from "no panel warranted". Stamp the
    terminal truth instead: eligibility is evaluated PURE against the live
    trace (no fence begin, quiescence wait, panel, model round, or prompt text
    — forced exits are the v6.29 honesty/salvage shelf, byte-identical); an
    OWED-but-bypassed panel lands as ``finalized_unaccepted`` with a
    closed-enum reason (`ACCEPTANCE_BYPASS_REASON_BY_RAIL`, v6.54.4
    deadline-reserve precedent generalized; v6.74.4). Reason tokens stay
    ledger-only (v6.61.4 token-parroting class). Never raises."""
    rail_reason = ACCEPTANCE_BYPASS_REASON_BY_RAIL.get(str(reason_code or ""))
    if rail_reason is None:
        return
    # A rail that deliberately cleared the failure state (a confirmed swarm routing
    # handoff) terminalized nothing reviewable here — the admitted managed task gets
    # its own acceptance lifecycle.
    if not str(ctx.accumulated_usage.get("reason_code") or ""):
        return
    tools_ctx = getattr(getattr(ctx, "tools", None), "_ctx", None)
    if tools_ctx is None:
        return
    # A recorded host decision (panel ran, pacing skip, supersede) wins; the
    # bypass record exists only for the no-host-verdict shape. "Host
    # decision" = a canonical status — NOT the status-less agent-stance dict
    # merged when task_acceptance_review defers to the host (that left the
    # bypass unrecorded when owed); `_set_acceptance_decision` stamps.
    decision = llm_trace.get("acceptance_decision")
    if isinstance(decision, dict) and str(decision.get("status") or "") in ACCEPTANCE_DECISION_STATUSES:
        return
    if getattr(tools_ctx, "_task_acceptance_reviewed", False):
        return
    trigger = f"bypassed_{reason_code}"
    try:
        from ouroboros.task_results import resolve_task_lineage

        meta = getattr(tools_ctx, "task_metadata", {})
        meta = meta if isinstance(meta, dict) else {}
        lineage = resolve_task_lineage(
            str(ctx.task_id or getattr(tools_ctx, "task_id", "") or ""),
            metadata=meta,
            root_task_id=getattr(tools_ctx, "root_task_id", None),
            parent_task_id=getattr(tools_ctx, "parent_task_id", None),
            delegation_role=getattr(tools_ctx, "delegation_role", None),
            original_task_id=getattr(tools_ctx, "original_task_id", None),
            timeout_retry_from=getattr(tools_ctx, "timeout_retry_from", None),
        )
        eligible, probe_trigger = _task_acceptance_eligible(
            get_task_review_mode(),
            llm_trace,
            bool(getattr(tools_ctx, "is_direct_chat", False)),
            is_root_task=bool(lineage["is_root_task"]),
            is_ephemeral_turn=bool(getattr(tools_ctx, "is_ephemeral_turn", False)),
            task_contract=(
                tools_ctx.task_contract
                if isinstance(getattr(tools_ctx, "task_contract", None), dict)
                else {}
            ),
        )
    except Exception:
        # A mid-round dying trace may not support the probe; record the honest
        # unknown instead of crashing the salvage path.
        log.debug("Forced acceptance-bypass eligibility probe failed", exc_info=True)
        llm_trace["review_decision"] = {"eligibility": "unknown", "trigger": trigger}
        return
    if not eligible:
        # Explicitly "no panel warranted" — now distinguishable from "not evaluated".
        llm_trace["review_decision"] = {"eligibility": "not_eligible", "trigger": probe_trigger}
        return
    llm_trace["review_decision"] = {"eligibility": "eligible", "trigger": trigger}
    _set_acceptance_decision(llm_trace, {
        "status": ACCEPTANCE_FINALIZED_UNACCEPTED,
        "reason": rail_reason,
        "source": "forced_finalization",
    })


def _record_forced_finalization(
    ctx: _RoundLimitContext,
    llm_trace: Dict[str, Any],
    *,
    reason_code: str,
    source: str,
    candidate: Optional[DeliveryCandidate],
) -> None:
    # Forced exits bypass the normal no-tool finalization gate. Project
    # child dispositions here, after services/evidence and the candidate
    # refresh, so every forced return exposes the same terminal child-result
    # truth to the outcome reducer.
    _project_child_result_dispositions(ctx, llm_trace)
    # Common terminal recorder = the ONE seam over the LLM-seam forced
    # answer (`_forced_final_answer`) and the no-spend host-fallback fence
    # path (`_handle_budget_exceeded` -> `_forced_fallback_result`).
    _record_forced_acceptance_bypass(ctx, llm_trace, reason_code)
    binding = dict(candidate.acceptance_binding or {}) if candidate is not None else {}
    tools = getattr(ctx, "tools", None)
    current_fingerprint = str(
        getattr(getattr(tools, "_ctx", None), "_delivery_evidence_fingerprint", "")
        or ""
    )
    current_revision = int(
        getattr(getattr(tools, "_ctx", None), "_delivery_evidence_revision", 0)
        or 0
    )
    llm_trace["forced_finalization"] = {
        "reason_code": reason_code,
        "source": source,
        "degraded": True,
        "candidate_sha256": candidate.content_sha256 if candidate is not None else "",
        "candidate_revision": candidate.revision if candidate is not None else None,
        "evidence_revision": candidate.evidence_revision if candidate is not None else None,
        "current_evidence_revision": current_revision,
        "evidence_current": bool(
            candidate is not None
            and candidate.evidence_fingerprint == current_fingerprint
        ),
        "acceptance_status": str(binding.get("acceptance_status") or "unaccepted"),
        "acceptance_authoritative": bool(binding.get("authoritative", False)),
    }


def _merge_finalization_trace(
    llm_trace: Dict[str, Any],
    returned_trace: Any,
) -> Dict[str, Any]:
    """Merge a forced-path trace without duplicating the live trace object."""

    if not isinstance(returned_trace, dict) or returned_trace is llm_trace:
        return llm_trace
    for key, value in returned_trace.items():
        if isinstance(value, list) and isinstance(llm_trace.get(key), list):
            for item in value:
                if item not in llm_trace[key]:
                    llm_trace[key].append(item)
        elif isinstance(value, dict) and isinstance(llm_trace.get(key), dict):
            llm_trace[key].update(value)
        else:
            llm_trace[key] = value
    return llm_trace


def _delivery_control_prompt(candidate: DeliveryCandidate, *, keep_allowed: bool) -> str:
    keep_line = (
        "keep is allowed because no answer-invalidating evidence changed."
        if keep_allowed
        else "keep is NOT allowed because owner/tool/child/verification evidence changed."
    )
    return (
        "[DELIVERY_FINALIZATION_CONTROL]\n"
        f"A complete answer candidate (revision {candidate.revision}, sha256 "
        f"{candidate.content_sha256[:12]}) is retained by the loop; do not replace it with a "
        f"service notice. {keep_line}\n"
        "Return exactly one JSON object and no other text:\n"
        '{"delivery_control":"keep"}\n'
        "or\n"
        '{"delivery_control":"replace","full_answer":"<the complete user-facing answer>"}'
    )


def _delivery_replace_required(candidate: DeliveryCandidate) -> bool:
    """Return whether a typed full replacement is mandatory for this control round."""

    return candidate.finalization_control.startswith(
        ("effect_revision_required", "skill_revision_required")
    )


def _delivery_keep_allowed(
    candidate: DeliveryCandidate,
    evidence_revision: int,
    evidence_fingerprint: str,
) -> bool:
    return (
        not _delivery_replace_required(candidate)
        and candidate.evidence_revision == evidence_revision
        and candidate.evidence_fingerprint == evidence_fingerprint
    )


def _arm_delivery_control(
    tools: ToolRegistry,
    ctx: _RoundLimitContext,
    llm_trace: Dict[str, Any],
    *,
    control: str = "awaiting_control",
) -> None:
    candidate = getattr(tools._ctx, "_delivery_candidate", None)
    if not isinstance(candidate, DeliveryCandidate):
        return
    evidence_revision, evidence_fingerprint = _delivery_evidence_state(tools, ctx, llm_trace)
    candidate.finalization_control = control
    candidate.repair_attempted = False
    tools._ctx._delivery_control_required = True
    _append_or_merge_user_message(
        ctx.messages,
        _delivery_control_prompt(
            candidate,
            keep_allowed=_delivery_keep_allowed(
                candidate, evidence_revision, evidence_fingerprint,
            ),
        ),
    )
    _publish_delivery_candidate(tools, candidate, llm_trace)


def _hold_delivery_for_skill_action(
    tools: ToolRegistry,
    llm_trace: Dict[str, Any],
) -> None:
    """Retain the answer while an unresolved skill lifecycle gate requires action."""

    candidate = getattr(tools._ctx, "_delivery_candidate", None)
    if not isinstance(candidate, DeliveryCandidate):
        return
    candidate.finalization_control = "skill_action_or_revision_required"
    candidate.repair_attempted = False
    tools._ctx._delivery_control_required = False
    _publish_delivery_candidate(tools, candidate, llm_trace)


def _parse_delivery_control_object(
    raw: str,
) -> tuple[Optional[Dict[str, Any]], bool]:
    """Parse a delivery-control object while rejecting duplicate JSON keys.

    The boolean preserves protocol intent for the repair path when a duplicate
    ``delivery_control`` or ``full_answer`` key made the object invalid.
    """

    duplicate_protocol_key = False

    def _unique_object(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
        nonlocal duplicate_protocol_key
        result: Dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                if key in {"delivery_control", "full_answer"}:
                    duplicate_protocol_key = True
                raise ValueError(f"duplicate key: {key}")
            result[key] = value
        return result

    try:
        payload = json.loads(raw, object_pairs_hook=_unique_object)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None, duplicate_protocol_key
    if not isinstance(payload, dict):
        return None, False
    return payload, False


def _resolve_delivery_control(
    content: Any,
    tools: ToolRegistry,
    ctx: _RoundLimitContext,
    llm_trace: Dict[str, Any],
) -> tuple[str, str]:
    """Return ``retry`` or a complete answer text before any existing gate runs."""

    candidate = getattr(tools._ctx, "_delivery_candidate", None)
    required = bool(getattr(tools._ctx, "_delivery_control_required", False))
    if not isinstance(candidate, DeliveryCandidate):
        return "fresh", _extract_plain_text_from_content(content)
    raw = _extract_plain_text_from_content(content).strip()
    parsed, duplicate_protocol_key = _parse_delivery_control_object(raw)
    # ANY parsed object carrying the protocol key is control intent,
    # regardless of verb/value — an unknown verb is a mangled protocol
    # attempt, never prose (raw JSON leaked to chat); validity judged below.
    is_control_intent = duplicate_protocol_key or (
        isinstance(parsed, dict) and "delivery_control" in parsed
    )
    if not required:
        if _delivery_replace_required(candidate):
            # A writer/skill action cannot silently turn a short acknowledgement
            # into the new complete answer, even if a caller lost the transient
            # required latch. The candidate's typed control state is authoritative.
            required = True
            tools._ctx._delivery_control_required = True
        elif candidate.finalization_control == "skill_action_or_revision_required":
            # Preserve the historical bounded skill gate: an actual tool
            # action or a reconsidered full prose answer may proceed, but a
            # typed keep cannot acknowledge the gate. No delivery JSON prompt
            # before the action — it would conflict with the instruction to
            # call the skill lifecycle tool.
            if not is_control_intent:
                return "fresh", _extract_plain_text_from_content(content)
            candidate.finalization_control = "skill_revision_required"
            required = True
            tools._ctx._delivery_control_required = True
        else:
            # An owner revision starts an ordinary substantive answer round.
            # If the model still follows the prior typed instruction, honor
            # that control structurally; service/effect/skill rounds are
            # handled by the replace-required branch above.
            if not (
                candidate.finalization_control == "owner_revision_required"
                and is_control_intent
            ):
                return "fresh", _extract_plain_text_from_content(content)
            tools._ctx._delivery_control_required = True
    evidence_revision, evidence_fingerprint = _delivery_evidence_state(tools, ctx, llm_trace)
    error = "control must be one exact JSON object"
    selected = str(parsed.get("delivery_control") or "") if isinstance(parsed, dict) else ""
    valid = False
    replacement = ""
    if selected == "keep" and set(parsed) == {"delivery_control"}:
        valid = _delivery_keep_allowed(
            candidate, evidence_revision, evidence_fingerprint,
        )
        error = "keep cannot bind changed evidence; send replace with the complete answer"
    elif selected == "replace" and set(parsed) == {"delivery_control", "full_answer"}:
        replacement_value = parsed.get("full_answer")
        if isinstance(replacement_value, str):
            replacement = replacement_value
        valid = isinstance(replacement_value, str) and bool(replacement.strip())
        error = "replace requires a non-empty complete full_answer"

    if valid and selected == "keep":
        tools._ctx._delivery_control_required = False
        candidate.finalization_control = "keep"
        candidate.acceptance_binding = _delivery_acceptance_binding(
            tools, llm_trace, candidate.content_sha256,
        )
        _publish_delivery_candidate(tools, candidate, llm_trace)
        return "resolved", candidate.full_text
    if valid and selected == "replace":
        updated = _replace_delivery_candidate(
            tools, ctx, llm_trace, replacement, control="replace",
        )
        return "resolved", updated.full_text

    if not candidate.repair_attempted:
        candidate.repair_attempted = True
        candidate.finalization_control = (
            f"{candidate.finalization_control}_repair_requested"
            if _delivery_replace_required(candidate)
            else "repair_requested"
        )
        if raw:
            ctx.messages.append({"role": "assistant", "content": raw})
        _append_or_merge_user_message(
            ctx.messages,
            "[DELIVERY_CONTROL_REPAIR] Invalid finalization control: " + error + ".\n"
            + _delivery_control_prompt(
                candidate,
                keep_allowed=_delivery_keep_allowed(
                    candidate, evidence_revision, evidence_fingerprint,
                ),
            ),
        )
        _publish_delivery_candidate(tools, candidate, llm_trace)
        return "retry", ""

    tools._ctx._delivery_control_required = False
    candidate.degraded = True
    candidate.degraded_reason = "invalid_delivery_control_after_repair"
    candidate.finalization_control = "degraded_preserve"
    # The control failed, not the retained text: bind that unchanged text to
    # the evidence the failed control was meant to acknowledge so the stale
    # check cannot reopen another control round. Still explicitly unaccepted;
    # the host acceptance gate judges this exact pair before publication.
    candidate.evidence_revision = evidence_revision
    candidate.evidence_fingerprint = evidence_fingerprint
    candidate.acceptance_binding = _unaccepted_delivery_binding(
        tools, candidate.content_sha256,
    )
    llm_trace["reasoning_notes"].append(
        "Delivery finalization control remained invalid after one repair; preserved the prior complete answer."
    )
    _publish_delivery_candidate(tools, candidate, llm_trace)
    return "degraded", candidate.full_text


def _compose_delivery_suffix(full_text: str, suffix: str) -> str:
    """Compose one host-owned suffix into the exact delivered/candidate text."""

    text = str(full_text or "")
    note = str(suffix or "")
    if not note or text.endswith(note):
        return text
    return text + note


def _forced_orphan_note(ctx: _RoundLimitContext, *, include_terminal: bool = True) -> str:
    """A bounded note listing children the parent did NOT explicitly handle
    (discard/cancel), appended to a finalization so paid child work is never
    SILENTLY orphaned (P1; P5 — no prose parsing). On a FORCED finalization
    (deadline / provider death / finalize_now, ``include_terminal=True``) the
    parent may not have seen completions: RUNNING and COMPLETED-undecided are
    both reported. On a NORMAL no-tool finalization
    (``include_terminal=False``) the agent saw every change, so only
    STILL-RUNNING undecided children — genuinely orphaned by finalizing
    mid-flight — are reported. Never raises."""
    try:
        from ouroboros.task_status import FINAL_STATUSES

        children = _direct_child_results(ctx)
        claimed = _claimed_child_dispositions(ctx)

        def _undecided(c: Dict[str, Any]) -> bool:
            if _child_disposition_state(c) in {
                "integrated", "irrelevant", "deferred", "discarded", "cancelled",
            }:
                return False  # explicitly handled
            if not include_terminal and str(c.get("status") or "").strip().lower() in FINAL_STATUSES:
                return False  # completed children were already surfaced via the reminder
            return True

        undecided = [c for c in children if _undecided(c)]
        deferred = [c for c in children if _child_disposition_state(c) == "deferred"]

        def _label(c: Dict[str, Any]) -> str:
            tid = str(c.get("task_id") or c.get("id") or "?")
            st = str(c.get("status") or "?").strip().lower()
            lifecycle = "running" if st not in FINAL_STATUSES else st
            # W2: a child whose LATEST blackboard decision row no longer
            # binds the current result was READ and decided — say that, not
            # "unread"; only what the ledger PROVES: the row EXISTS, the
            # binding did not. Scoped to children the projection left
            # UNDECIDED: a carried disposition (deferred / integrated /
            # irrelevant / discarded / cancelled) is no failed binding —
            # "re-submit to close it" would be false there.
            claim = claimed.get(tid) if not _child_disposition_state(c) else None
            if claim is not None:
                disposition, row_sha = claim
                from ouroboros.tools.join_ledger import _child_result_sha256

                if _child_result_sha256(c) != row_sha:
                    detail = (
                        f"{disposition} recorded for an EARLIER result hash; the current "
                        "result is not bound — re-inspect and re-submit the current hash"
                    )
                else:
                    detail = (
                        f"{disposition} recorded for this exact result hash but not carried "
                        "by this round's disposition projection — re-submit to close it"
                    )
                return f"{tid} [{lifecycle}; {detail}]"
            terminal = str(c.get("child_status") or "").strip().lower()
            if terminal and terminal != st:
                return f"{tid} [{lifecycle}; terminal_result={terminal}]"
            return f"{tid} [{lifecycle}]"

        notes: list[str] = []
        if undecided:
            listed = "; ".join(_label(c) for c in undecided[:10])
            more = f" (+{len(undecided) - 10} more)" if len(undecided) > 10 else ""
            lead = "finalized under a hard limit with" if include_terminal else "finalized with"
            detail = (
                "running ones may be incomplete, completed ones may be UNREAD"
                if include_terminal else
                "still-running children not absorbed or discarded"
            )
            notes.append(
                f"\n\n⚠️ NOTE: {lead} {len(undecided)} child task(s) not explicitly absorbed or "
                f"discarded — {detail}: {listed}{more}. Inspect with get_task_result(<id>) / "
                f"peek_task(<id>)."
            )
        if deferred:
            listed = "; ".join(_label(c) for c in deferred[:10])
            more = f" (+{len(deferred) - 10} more)" if len(deferred) > 10 else ""
            notes.append(
                f"\n\n⚠️ DEFERRED CHILD RESULTS: {listed}{more}. These exact results were "
                "explicitly deferred, so this answer is degraded/best-effort rather than clean solved."
            )
        return "".join(notes)
    except Exception:
        return ""


def _claimed_child_dispositions(ctx: _RoundLimitContext) -> Dict[str, tuple]:
    """task_id -> (disposition, row_sha) from THIS parent's latest blackboard
    decision rows (W2). Consulted only for children the disposition projection
    left undecided: a row that exists but no longer binds is audit evidence of a
    claimed-but-failed disposition write, and the forced orphan note must say so
    instead of calling the child unread. Pure read, never raises."""
    try:
        from ouroboros.task_tree_ledger import CHILD_RESULT_DISPOSITION_TYPE, tree_ledger_rows

        status_root = (
            getattr(ctx, "status_drive_root", None)
            or getattr(ctx, "drive_root", None)
        )
        root_id = str(getattr(ctx, "root_task_id", "") or getattr(ctx, "task_id", "") or "")
        parent_id = str(getattr(ctx, "task_id", "") or "")
        if status_root is None or not root_id or not parent_id:
            return {}
        claims: Dict[str, tuple] = {}
        for row in tree_ledger_rows(root_id, data_root=pathlib.Path(status_root)):
            payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
            if (
                str(row.get("kind") or "") == "decision"
                and str(payload.get("type") or "") == CHILD_RESULT_DISPOSITION_TYPE
                and str(row.get("task_id") or "") == parent_id
                and str(payload.get("child_task_id") or "")
            ):
                # Later rows win: the ledger is append-only and the newest decision
                # is the one whose failure to bind is worth naming.
                claims[str(payload["child_task_id"])] = (
                    str(payload.get("disposition") or ""),
                    str(payload.get("child_result_sha256") or ""),
                )
        return claims
    except Exception:
        return {}


def _undispositioned_children(ctx: _RoundLimitContext) -> list[Dict[str, Any]]:
    try:
        return [
            child for child in _direct_child_results(ctx)
            if _child_disposition_state(child) not in {
                "integrated", "irrelevant", "deferred", "discarded", "cancelled",
            }
        ]
    except Exception:
        return []


def _maybe_enforce_child_absorption_gate(
    tools: ToolRegistry,
    limit_ctx: _RoundLimitContext,
    content: Any,
    messages: List[Dict[str, Any]],
    emit_progress: Callable[[str], None],
    llm_trace: Dict[str, Any],
) -> Optional[Tuple[str, Dict[str, Any], Dict[str, Any]] | str]:
    undecided = _undispositioned_children(limit_ctx)
    if not undecided:
        return None
    if not getattr(tools._ctx, "_child_absorption_reminded", False):
        tools._ctx._child_absorption_reminded = True
        if content and str(content).strip():
            messages.append({"role": "assistant", "content": content})
        from ouroboros.tools.join_ledger import _child_result_sha256

        listed = "; ".join(
            f"{c.get('task_id') or c.get('id') or '?'} [{c.get('status') or 'unknown'}] "
            f"sha256={_child_result_sha256(c)}"
            for c in undecided[:10]
        )
        reminder = (
            "[CHILD_ABSORPTION_REQUIRED]\n"
            "You have child result(s) without a current exact-hash disposition: "
            f"{listed}. Before a clean final answer, inspect unfinished children or record a "
            "tree_note(kind='decision') payload with type=child_result_disposition, child_task_id, "
            "disposition=integrated|irrelevant|deferred, and the shown child_result_sha256. "
            "To disposition several children in ONE call, pass a children array instead: "
            "payload={'type': 'child_result_disposition', 'children': [{'child_task_id': ..., "
            "'disposition': ..., 'child_result_sha256': ...}, ...]}. "
            "discard_child_result remains the shorthand for irrelevant. This is a bounded reminder; "
            "ignoring it will finalize best_effort, not clean."
        )
        _append_or_merge_user_message(messages, reminder)
        emit_progress("Child absorption reminder injected before final response.")
        llm_trace["reasoning_notes"].append("Child absorption reminder injected before final response.")
        return "continue"
    text, usage, forced_trace = _forced_final_answer(
        limit_ctx,
        prompt=(
            "[FINALIZE_WITH_UNABSORBED_CHILDREN]\n"
            "You still have child results without exact dispositions and already received one "
            "child-absorption reminder. Produce an honest best-effort final answer now; name the "
            "unabsorbed or unfinished children explicitly."
        ),
        fallback_text="⚠️ Finalized best-effort with undispositioned child results.",
        reason_code="children_unabsorbed",
    )
    _merge_finalization_trace(llm_trace, forced_trace)
    _run_forced_children_acceptance(
        tools, limit_ctx, undecided, text, messages, emit_progress, llm_trace,
    )
    return text, usage, llm_trace


def _run_forced_children_acceptance(
    tools: ToolRegistry,
    limit_ctx: _RoundLimitContext,
    undecided: list[Dict[str, Any]],
    text: str,
    messages: List[Dict[str, Any]],
    emit_progress: Callable[[str], None],
    llm_trace: Dict[str, Any],
) -> None:
    """Content acceptance still runs on the forced children_unabsorbed rail (owner Q2A).

    The panel uses the ORDINARY entry point
    (`_run_task_acceptance_review_once`) after the forced answer text exists
    but BEFORE the loop seals it; the evidence packet carries the
    undispositioned children via the ctx stash. The forced rail can never
    take another model round, so a ``True`` return terminalizes here: a
    requested improvement pass downgrades to ``finalized_unaccepted``; a WAIT
    shape that never ran the panel keeps the typed acceptance-bypass verdict
    from `_record_forced_finalization`. Never raises — salvage outranks review."""
    if not str(text or "").strip():
        return
    tools_ctx = tools._ctx
    try:
        from ouroboros.tools.join_ledger import _child_result_sha256

        debt = [
            {
                "task_id": str(c.get("task_id") or c.get("id") or ""),
                "status": str(c.get("status") or "unknown"),
                "child_result_sha256": _child_result_sha256(c),
            }
            for c in undecided[:20]
            if isinstance(c, dict)
        ]
        if len(undecided) > 20:
            # Explicit omission marker: a >20-child debt list must not read as complete.
            debt.append({"omitted": len(undecided) - 20, "total": len(undecided)})
        tools_ctx._forced_undispositioned_children = debt
        another_round = _run_task_acceptance_review_once(
            tools=tools,
            content=str(text),
            task_id=limit_ctx.task_id,
            task_type=limit_ctx.task_type,
            llm_trace=llm_trace,
            drive_root=limit_ctx.drive_root,
            messages=messages,
            emit_progress=emit_progress,
        )
        if not another_round:
            return
        tools_ctx._task_acceptance_reviewed = True
        _end_task_acceptance_fence(tools_ctx, outcome="terminal")
        decision = llm_trace.get("acceptance_decision")
        status = str(decision.get("status") or "") if isinstance(decision, dict) else ""
        if status == ACCEPTANCE_REVISION_REQUESTED:
            # A panel DID run and asked for an improvement pass; record the honest
            # terminal state instead of leaving a dangling revision request.
            _set_acceptance_decision(llm_trace, {
                "status": ACCEPTANCE_FINALIZED_UNACCEPTED,
                "reason": "revision_unavailable_on_forced_rail",
                "source": "forced_finalization",
                "rationale": (
                    "The acceptance panel requested an improvement pass, but the "
                    "forced children_unabsorbed rail cannot take another model round."
                ),
            })
            emit_progress(
                "Task acceptance ran on the forced rail; the requested improvement "
                "pass is unavailable, finalizing unaccepted."
            )
    except Exception:
        log.debug("Forced children_unabsorbed acceptance run failed", exc_info=True)
    finally:
        tools_ctx._forced_undispositioned_children = None


def _enforce_swarm_actions(
    content: str,
    messages: List[Dict[str, Any]],
    tools: ToolRegistry,
    llm_trace: Dict[str, Any],
    emit_progress: Callable[[str], None],
) -> bool:
    """Hold normal finalization while routing or blocking plan work is open."""

    if swarm_router_turn(tools._ctx) and not _swarm_handoff_attempt(tools._ctx):
        if content.strip():
            messages.append({"role": "assistant", "content": content})
        reminder = (
            "[SWARM_ROUTING_INTENT] Admit exactly one new managed root now with "
            "promote_chat_to_task, or from Main route_to_project for a clearly matching "
            "existing Project. Do not answer inline or steer an existing task."
        )
        _append_or_merge_user_message(messages, reminder)
        llm_trace["reasoning_notes"].append(reminder)
        emit_progress("Swarm routing action required before final response.")
        return True

    decision = _force_plan_decision(tools._ctx, llm_trace)
    if decision.get("required"):
        llm_trace["force_plan_decision"] = decision
    if decision.get("allow"):
        return False
    if content.strip():
        messages.append({"role": "assistant", "content": content})
    reminder = _force_plan_reminder(decision)
    _append_or_merge_user_message(messages, reminder)
    llm_trace["reasoning_notes"].append(reminder)
    emit_progress("Plan-review action required before final response.")
    return True


def _no_tool_final_answer(
    content: Any,
    limit_ctx: _RoundLimitContext,
    llm_trace: Dict[str, Any],
    tools: ToolRegistry,
    incoming_messages: queue.Queue,
    owner_msg_seen: set,
    emit_progress: Callable[[str], None],
) -> Optional[Tuple[str, Dict[str, Any], Dict[str, Any]]]:
    """Run the no-tool finalization gates; ``None`` requests another model round."""
    messages = limit_ctx.messages
    control_state, controlled_content = _resolve_delivery_control(
        content, tools, limit_ctx, llm_trace,
    )
    if control_state == "retry":
        return None
    content = controlled_content
    _project_child_result_dispositions(limit_ctx, llm_trace)
    if control_state == "fresh" and str(content or "").strip():
        candidate = _replace_delivery_candidate(
            tools, limit_ctx, llm_trace, str(content), control="candidate",
        )
        content = candidate.full_text
    else:
        candidate = getattr(tools._ctx, "_delivery_candidate", None)
        if isinstance(candidate, DeliveryCandidate):
            content = candidate.full_text

    if _enforce_swarm_actions(
        str(content or ""), messages, tools, llm_trace, emit_progress,
    ):
        return None
    handoff_msg = _compute_subagent_handoff(tools, limit_ctx.drive_root, limit_ctx.task_id, content)
    if handoff_msg:
        if content and content.strip():
            messages.append({"role": "assistant", "content": content})
        _append_or_merge_user_message(messages, f"[SYSTEM REMINDER]\n{handoff_msg}")
        emit_progress("Subagent handoff status refreshed before final response.")
        llm_trace["reasoning_notes"].append("Subagent handoff status refreshed before final response.")
        _arm_delivery_control(tools, limit_ctx, llm_trace)
        return None
    absorption_result = _maybe_enforce_child_absorption_gate(
        tools, limit_ctx, content, messages, emit_progress, llm_trace,
    )
    if absorption_result == "continue":
        _arm_delivery_control(tools, limit_ctx, llm_trace)
        return None
    if absorption_result is not None:
        return absorption_result
    skill_finalization_was_injected = bool(
        getattr(tools._ctx, "_skill_finalization_injected", False)
    )
    if _maybe_inject_finalization_nudges(
        tools, limit_ctx.drive_root, limit_ctx.task_id, llm_trace, content, messages, emit_progress,
    ):
        skill_finalization_injected_now = (
            not skill_finalization_was_injected
            and bool(getattr(tools._ctx, "_skill_finalization_injected", False))
        )
        # Skill finalization is an action gate, not a service notice.
        # Preserve the candidate without a conflicting JSON-only instruction:
        # the next round may run the required tool or give the historically
        # allowed reconsidered answer; a typed keep cannot close it.
        if skill_finalization_injected_now:
            _hold_delivery_for_skill_action(tools, llm_trace)
        else:
            _arm_delivery_control(tools, limit_ctx, llm_trace)
        return None

    # Declared service outputs and teardown failures are acceptance evidence,
    # not postscript cleanup: finalize them before the host panel and, when
    # that changes evidence, require one complete replacement answer bound to
    # the new revision. The finally-path reuses the same idempotent helper.
    service_exit_ctx = _LoopExitContext(
        tools=tools,
        drive_root=limit_ctx.drive_root,
        task_id=limit_ctx.task_id,
        event_queue=limit_ctx.event_queue,
        drive_logs=limit_ctx.drive_logs,
        accumulated_usage=limit_ctx.accumulated_usage,
        llm_trace=llm_trace,
    )
    if _finalize_task_services(service_exit_ctx):
        evidence_revision, evidence_fingerprint = _delivery_evidence_state(
            tools, limit_ctx, llm_trace,
        )
        candidate = getattr(tools._ctx, "_delivery_candidate", None)
        if (
            isinstance(candidate, DeliveryCandidate)
            and (
                candidate.evidence_revision != evidence_revision
                or candidate.evidence_fingerprint != evidence_fingerprint
            )
        ):
            if content and str(content).strip():
                messages.append({"role": "assistant", "content": str(content)})
            llm_trace["reasoning_notes"].append(
                "Task services were finalized before acceptance; the complete answer must bind the resulting evidence."
            )
            _arm_delivery_control(tools, limit_ctx, llm_trace)
            return None

    _project_child_result_dispositions(limit_ctx, llm_trace)
    plan_suffix = _force_plan_disclosure(tools._ctx, llm_trace)
    orphan_suffix = _forced_orphan_note(limit_ctx, include_terminal=False)
    normal_suffix = plan_suffix + orphan_suffix
    composed_content = _compose_delivery_suffix(str(content or ""), normal_suffix)
    candidate = getattr(tools._ctx, "_delivery_candidate", None)
    if composed_content and (
        not isinstance(candidate, DeliveryCandidate)
        or candidate.full_text != composed_content
    ):
        candidate = _replace_delivery_candidate(
            tools,
            limit_ctx,
            llm_trace,
            composed_content,
            control="host_suffix" if normal_suffix else "candidate",
            model_text=str(content or ""),
        )
    if isinstance(candidate, DeliveryCandidate):
        if orphan_suffix:
            candidate.degraded = True
            candidate.degraded_reason = "host_child_status_suffix"
            _publish_delivery_candidate(tools, candidate, llm_trace)
        elif plan_suffix:
            candidate.degraded = True
            candidate.degraded_reason = "plan_review_advisory"
            _publish_delivery_candidate(tools, candidate, llm_trace)
        content = candidate.full_text

    tools._ctx._acceptance_loop_rails = {
        "round_idx": limit_ctx.round_idx,
        "max_rounds": limit_ctx.max_rounds,
        "task_cost_usd": limit_ctx.accumulated_usage.get("cost"),
    }
    # v6.78.0 (owner Q20/Q22): mirror the host-attested native-retrieval
    # fact into the trace so `build_task_acceptance_evidence` can show the
    # reviewer whether the answer was grounded in fetched pages. Reviewer-side
    # only — the agent gets the improvement capsule, not the evidence packet.
    _retrieval = limit_ctx.accumulated_usage.get("retrieval")
    if isinstance(_retrieval, dict) and _retrieval:
        llm_trace["retrieval"] = dict(_retrieval)
    if _run_task_acceptance_review_once(
        tools=tools,
        content=content or "",
        task_id=limit_ctx.task_id,
        task_type=limit_ctx.task_type,
        llm_trace=llm_trace,
        drive_root=limit_ctx.drive_root,
        messages=messages,
        emit_progress=emit_progress,
    ):
        # v6.71.1: an acceptance improvement pass is an ORDINARY substantive
        # answer round — do NOT arm delivery-control: layering "return
        # exactly one JSON object" on OPEN OBLIGATIONS plus the self-check
        # froze the model into resubmitting the same answer. The next
        # free-form answer re-enters the acceptance panel (blocking not
        # weakened); other lanes still arm where JSON keep/replace is needed.
        return None
    if bool(getattr(tools._ctx, "_task_acceptance_fence_infra_failed", False)):
        # The bounded fence wait is exhausted: terminalize as infra_failed
        # through the host-salvage seam instead of finalizing past a review
        # that never ran.
        text, usage, fence_trace = _forced_fallback_result(
            limit_ctx,
            llm_trace,
            "⚠️ The task could not start its acceptance review: the queue-owned "
            "admission fence stayed unavailable. Any files written so far are "
            "preserved in the workspace.",
            "acceptance_fence_unavailable",
            source="acceptance_fence_unavailable",
        )
        usage.update(
            execution_status=RESULT_INFRA_FAILED,
            reason_code="acceptance_fence_unavailable",
        )
        return text, usage, fence_trace
    candidate = getattr(tools._ctx, "_delivery_candidate", None)
    if isinstance(candidate, DeliveryCandidate):
        candidate.acceptance_binding = _delivery_acceptance_binding(
            tools, llm_trace, candidate.content_sha256,
        )
        _publish_delivery_candidate(tools, candidate, llm_trace)

    # Close delivery under the same lock as routing, then drain once. A follow-up
    # either forces another round or is rejected after the fence, never stranded.
    admission_lock = getattr(tools._ctx, "owner_message_admission_lock", None)
    admission_agent = getattr(tools._ctx, "owner_message_admission_agent", None)
    if admission_lock is not None and admission_agent is not None:
        before_directives = len(getattr(tools._ctx, "_owner_directives", []) or [])
        acceptance_was_terminal = bool(
            getattr(tools._ctx, "_task_acceptance_reviewed", False)
            or getattr(tools._ctx, "_task_acceptance_sealed_fence_token", None)
        )
        provisional_assistant = {"role": "assistant", "content": content} if content else None
        if provisional_assistant is not None:
            messages.append(provisional_assistant)
        with admission_lock:
            admission_agent._accepting_owner_messages = False
            post_controls = _drain_incoming_messages(
                messages, incoming_messages, limit_ctx.drive_root, limit_ctx.task_id,
                limit_ctx.event_queue, owner_msg_seen, owner_ctx=tools._ctx,
            )
        if len(getattr(tools._ctx, "_owner_directives", []) or []) > before_directives:
            with admission_lock:
                if acceptance_was_terminal:
                    _supersede_task_acceptance_for_owner_followup(
                        tools._ctx, llm_trace, admission_locked=True,
                    )
                if (
                    getattr(admission_agent, "_busy", False)
                    and str(getattr(admission_agent, "_current_task_id", "") or "") == limit_ctx.task_id
                ):
                    admission_agent._accepting_owner_messages = True
            if acceptance_was_terminal:
                emit_progress(
                    "Task acceptance review superseded: an owner follow-up arrived before finalization."
                )
            # An owner directive is a substantive revision request, not a service
            # notification. The next complete response creates a fresh candidate.
            tools._ctx._delivery_control_required = False
            if isinstance(candidate, DeliveryCandidate):
                candidate.finalization_control = "owner_revision_required"
                _delivery_evidence_state(tools, limit_ctx, llm_trace)
                _publish_delivery_candidate(tools, candidate, llm_trace)
            return None
        if provisional_assistant is not None and messages[-1] is provisional_assistant:
            messages.pop()
        if post_controls.get("finalize_now"):
            text, usage, forced_trace = _maybe_early_finalize(
                limit_ctx, tools, post_controls,
            )
            _merge_finalization_trace(llm_trace, forced_trace)
            return text, usage, llm_trace
    _project_child_result_dispositions(limit_ctx, llm_trace)
    evidence_revision, evidence_fingerprint = _delivery_evidence_state(
        tools, limit_ctx, llm_trace,
    )
    candidate = getattr(tools._ctx, "_delivery_candidate", None)
    if (
        isinstance(candidate, DeliveryCandidate)
        and (
            candidate.evidence_revision != evidence_revision
            or candidate.evidence_fingerprint != evidence_fingerprint
        )
    ):
        acceptance_was_terminal = bool(
            getattr(tools._ctx, "_task_acceptance_reviewed", False)
            or getattr(tools._ctx, "_task_acceptance_sealed_fence_token", None)
        )
        if acceptance_was_terminal:
            decision = (
                llm_trace.get("review_decision")
                if isinstance(llm_trace.get("review_decision"), dict)
                else {}
            )
            expected_panel = str(decision.get("panel_id") or "")
            expected_binding = str(decision.get("binding_hash") or "")
            active_run = next(
                (
                    run
                    for run in reversed(llm_trace.get("review_runs") or [])
                    if isinstance(run, dict)
                    and run.get("authority") == "host_root"
                    and not run.get("superseded_by_revision")
                    and str(run.get("panel_id") or "") == expected_panel
                    and str(run.get("binding_hash") or "") == expected_binding
                ),
                None,
            )
            _supersede_task_acceptance_for_evidence_change(
                tools._ctx,
                llm_trace,
                active_run,
                "delivery_evidence_changed_after_host_acceptance",
                messages,
                emit_progress,
            )
        if candidate.full_text:
            messages.append({"role": "assistant", "content": candidate.full_text})
        llm_trace["reasoning_notes"].append(
            "Delivery evidence changed after host acceptance; a complete replacement answer is required."
        )
        _arm_delivery_control(tools, limit_ctx, llm_trace)
        return None
    if isinstance(candidate, DeliveryCandidate):
        candidate.acceptance_binding = _delivery_acceptance_binding(
            tools, llm_trace, candidate.content_sha256,
        )
        _publish_delivery_candidate(tools, candidate, llm_trace)
        content = candidate.full_text
    return _handle_text_response(
        str(content or ""),
        llm_trace,
        limit_ctx.accumulated_usage,
    )


def _finalize_forced_services(
    ctx: _RoundLimitContext,
    llm_trace: Dict[str, Any],
) -> None:
    """Finalize services and expose their stable projection before forced synthesis."""

    tools = getattr(ctx, "tools", None)
    if tools is None:
        return
    _finalize_task_services(_LoopExitContext(
        tools=tools,
        drive_root=ctx.drive_root,
        task_id=ctx.task_id,
        event_queue=ctx.event_queue,
        drive_logs=ctx.drive_logs,
        accumulated_usage=ctx.accumulated_usage,
        llm_trace=llm_trace,
    ))
    _delivery_evidence_state(tools, ctx, llm_trace)
    projection = _service_finalization_evidence(llm_trace)
    if not projection:
        return
    payload = json.dumps(
        projection,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    fingerprint = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    if ctx.forced_service_evidence_fingerprint == fingerprint:
        return
    from ouroboros.observability import redact_projection

    ctx.forced_service_evidence_fingerprint = fingerprint
    safe_payload = truncate_review_artifact(
        str(redact_projection(payload).value),
        limit=8000,
    )
    _append_or_merge_user_message(
        ctx.messages,
        "[SERVICE_FINALIZATION_EVIDENCE]\n"
        "Task services were finalized before forced synthesis. Incorporate this "
        f"evidence and disclose any failure honestly:\n{safe_payload}",
    )


def _drain_forced_owner_directives(
    ctx: _RoundLimitContext,
    llm_trace: Dict[str, Any],
) -> bool:
    """Drain typed owner input after a forced call and advance answer evidence."""

    tools = getattr(ctx, "tools", None)
    if tools is None:
        return False
    incoming = ctx.incoming_messages
    if incoming is None:
        incoming = queue.Queue()
    seen = ctx.owner_msg_seen
    if not isinstance(seen, set):
        seen = set()
        ctx.owner_msg_seen = seen
    directives = getattr(tools._ctx, "_owner_directives", None)
    before = len(directives) if isinstance(directives, list) else 0
    _drain_incoming_messages(
        ctx.messages,
        incoming,
        ctx.drive_root,
        ctx.task_id,
        ctx.event_queue,
        seen,
        owner_ctx=tools._ctx,
    )
    directives = getattr(tools._ctx, "_owner_directives", None)
    after = len(directives) if isinstance(directives, list) else 0
    if after <= before:
        return False
    candidate = _live_delivery_candidate(ctx)
    binding = (
        candidate.acceptance_binding
        if isinstance(candidate, DeliveryCandidate)
        and isinstance(candidate.acceptance_binding, dict)
        else {}
    )
    if (
        binding.get("authoritative") is True
        or bool(getattr(tools._ctx, "_task_acceptance_reviewed", False))
        or bool(getattr(tools._ctx, "_task_acceptance_sealed_fence_token", None))
    ):
        _supersede_task_acceptance_for_owner_followup(tools._ctx, llm_trace)
    _delivery_evidence_state(tools, ctx, llm_trace)
    return True


def _call_forced_model_once(ctx: _RoundLimitContext) -> str:
    response_meta: Dict[str, Any] = {}
    final_msg, _final_cost = call_llm_with_retry(
        ctx.llm,
        ctx.messages,
        ctx.active_model,
        None,
        ctx.active_effort,
        ctx.max_retries,
        ctx.drive_logs,
        ctx.task_id,
        ctx.round_idx,
        ctx.event_queue,
        ctx.accumulated_usage,
        ctx.task_type,
        use_local=ctx.active_use_local,
        deadline_ts=ctx.deadline_ts,
        response_meta_out=response_meta,
        transport_reserve_sec=0.0,
    )
    ctx.accumulated_usage["_forced_response_meta"] = response_meta
    return str((final_msg or {}).get("content") or "").strip()


def _publish_model_forced_candidate(
    ctx: _RoundLimitContext,
    llm_trace: Dict[str, Any],
    full_text: str,
    reason_code: str,
) -> Optional[DeliveryCandidate]:
    """Replace the retained answer and old verdict."""

    tools = getattr(ctx, "tools", None)
    if tools is None:
        return None
    candidate = _replace_delivery_candidate(
        tools,
        ctx,
        llm_trace,
        full_text,
        control=f"forced_replace:{reason_code}",
    )
    candidate.acceptance_binding = _forced_unaccepted_binding(
        tools, candidate, reason_code,
    )
    candidate.degraded = True
    candidate.degraded_reason = reason_code
    _publish_delivery_candidate(tools, candidate, llm_trace)
    ctx.delivery_candidate = candidate
    return candidate


def _publish_stale_forced_candidate(
    ctx: _RoundLimitContext,
    llm_trace: Dict[str, Any],
    stale_candidate: DeliveryCandidate,
    reason_code: str,
    suffix: str,
) -> Optional[DeliveryCandidate]:
    """Preserve old text."""

    tools = getattr(ctx, "tools", None)
    if tools is None:
        return None
    current_revision, _current_fingerprint = _delivery_evidence_state(
        tools, ctx, llm_trace,
    )
    disclosure = (
        "\n\n⚠️ STALE-EVIDENCE NOTICE — RESUME REQUIRED (host): The preserved "
        "answer above was produced before newer task evidence reached the loop. "
        "It has not been regenerated or accepted against that newer evidence and "
        "does not claim to incorporate it. Resume the task to produce and review "
        "a complete answer against the latest evidence."
    )
    full_text = _compose_delivery_suffix(
        _compose_delivery_suffix(stale_candidate.full_text, suffix),
        disclosure,
    )
    candidate = _replace_delivery_candidate(
        tools,
        ctx,
        llm_trace,
        full_text,
        control=f"forced_stale_preserve:{reason_code}",
    )
    # A host disclosure cannot make the preserved model text current.
    candidate.evidence_revision = stale_candidate.evidence_revision
    candidate.evidence_fingerprint = stale_candidate.evidence_fingerprint
    candidate.acceptance_binding = _forced_unaccepted_binding(
        tools, candidate, reason_code,
    )
    candidate.acceptance_binding.update({
        "evidence_revision": stale_candidate.evidence_revision,
        "current_evidence_revision": current_revision,
        "stale_evidence": True,
    })
    candidate.degraded = True
    candidate.degraded_reason = reason_code
    _publish_delivery_candidate(tools, candidate, llm_trace)
    ctx.delivery_candidate = candidate
    return candidate


def _forced_fallback_result(
    ctx: _RoundLimitContext,
    llm_trace: Dict[str, Any],
    fallback_text: str,
    reason_code: str,
    *,
    source: str = "host_fallback",
    retained_source: str = "",
    retained_control: str = "",
    provider_terminal: bool = False,
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """Return a current or fallback candidate."""

    router_result = _forced_swarm_router_result(ctx, llm_trace, reason_code)
    if router_result is not None:
        return router_result
    tool_ctx = getattr(getattr(ctx, "tools", None), "_ctx", None)
    plan_suffix = (
        _force_plan_disclosure(tool_ctx, llm_trace, forced_reason=reason_code)
        if tool_ctx is not None else ""
    )
    suffix = plan_suffix + _forced_orphan_note(ctx)
    live_candidate = _live_delivery_candidate(ctx)
    fallback_is_retained_model_text = (
        isinstance(live_candidate, DeliveryCandidate)
        and fallback_text == live_candidate.full_text
    )
    candidate = _current_delivery_candidate(ctx, llm_trace)
    if candidate is not None:
        composed = (
            candidate.model_text or candidate.full_text if provider_terminal else
            sanitize_tool_result_for_log(_compose_delivery_suffix(candidate.full_text, suffix))
        )
        if provider_terminal:
            ctx.accumulated_usage.update(
                terminal_origin=TERMINAL_ORIGIN_MODEL_FINAL,
                terminal_plan_review_open=bool(plan_suffix),
            )
        if composed != candidate.full_text:
            candidate = _publish_model_forced_candidate(
                ctx, llm_trace, composed, reason_code,
            )
            ctx.accumulated_usage["_best_effort_extracted"] = True
            _record_forced_finalization(
                ctx,
                llm_trace,
                reason_code=reason_code,
                source=(
                    f"{retained_source}_with_host_suffix"
                    if retained_source else "retained_candidate_with_host_suffix"
                ),
                candidate=candidate,
            )
            return composed, ctx.accumulated_usage, llm_trace
        _degrade_retained_delivery_candidate(
            ctx,
            llm_trace,
            candidate,
            control=retained_control or f"forced_preserve:{reason_code}",
            reason_code=reason_code,
        )
        ctx.accumulated_usage["_best_effort_extracted"] = True
        _record_forced_finalization(
            ctx,
            llm_trace,
            reason_code=reason_code,
            source=retained_source or "retained_candidate",
            candidate=candidate,
        )
        return candidate.full_text, ctx.accumulated_usage, llm_trace

    if fallback_is_retained_model_text and live_candidate is not None:
        candidate = _publish_stale_forced_candidate(
            ctx,
            llm_trace,
            live_candidate,
            reason_code,
            suffix,
        )
        if candidate is not None:
            if provider_terminal:
                ctx.accumulated_usage["terminal_origin"] = TERMINAL_ORIGIN_HOST_SALVAGE
            ctx.accumulated_usage["_best_effort_extracted"] = True
            _record_forced_finalization(
                ctx,
                llm_trace,
                reason_code=reason_code,
                source=f"{source}_stale_evidence_resume_required",
                candidate=candidate,
            )
            return candidate.full_text, ctx.accumulated_usage, llm_trace

    composed = sanitize_tool_result_for_log(_compose_delivery_suffix(fallback_text, suffix))
    candidate = _publish_model_forced_candidate(
        ctx, llm_trace, composed, reason_code,
    )
    if fallback_is_retained_model_text:
        ctx.accumulated_usage["_best_effort_extracted"] = True
    if provider_terminal:
        ctx.accumulated_usage["terminal_origin"] = TERMINAL_ORIGIN_HOST_SALVAGE
    _record_forced_finalization(
        ctx,
        llm_trace,
        reason_code=reason_code,
        source=source,
        candidate=candidate,
    )
    return composed, ctx.accumulated_usage, llm_trace


def _forced_swarm_router_result(
    ctx: _RoundLimitContext,
    llm_trace: Dict[str, Any],
    reason_code: str,
) -> Optional[Tuple[str, Dict[str, Any], Dict[str, Any]]]:
    """Use deterministic routing text only when a real rail ends the router."""

    tools = getattr(ctx, "tools", None)
    if tools is None or not swarm_router_turn(tools._ctx):
        return None
    attempt = _swarm_handoff_attempt(tools._ctx)
    status = str(attempt.get("status") or "not_attempted")
    task_id = str(attempt.get("task_id") or "")
    if status == "scheduled":
        text = f"✅ Swarm admitted managed task {task_id}. Work continues in that task."
    elif status == "unconfirmed":
        text = (
            f"⚠️ Swarm attempted managed task {task_id}, but admission was not confirmed. "
            "No second routing event was emitted; keep the task id for reconciliation."
        )
    elif status == "rejected":
        detail = str(attempt.get("reason") or "admission rejected")
        text = f"⚠️ Swarm could not admit a new managed task ({detail}). No retry was emitted."
    else:
        text = (
            f"⚠️ Swarm reached the task-wide rail `{reason_code}` before a managed-root "
            "admission attempt completed. No inline work was published."
        )
    full_text = _compose_delivery_suffix(text, _forced_orphan_note(ctx))
    candidate = _replace_delivery_candidate(
        tools, ctx, llm_trace, full_text, control=f"forced_swarm_router:{reason_code}",
    )
    if status != "scheduled":
        candidate.degraded = True
        candidate.degraded_reason = reason_code
    _publish_delivery_candidate(tools, candidate, llm_trace)
    if status == "scheduled":
        # The short acknowledgement hit a rail, but the requested managed work
        # was already durably admitted. Keep that successful handoff truthful.
        ctx.accumulated_usage.pop("execution_status", None)
        ctx.accumulated_usage.pop("reason_code", None)
    else:
        ctx.accumulated_usage.update(execution_status="failed", reason_code=reason_code)
    _record_forced_finalization(
        ctx,
        llm_trace,
        reason_code=reason_code,
        source="host_swarm_routing_fallback",
        candidate=candidate,
    )
    return candidate.full_text, ctx.accumulated_usage, llm_trace


def _resolve_forced_delivery_control(
    tools_ctx: Any,
    extracted: str,
) -> Tuple[str, str]:
    """Resolve an armed delivery-control object without another model round."""
    if tools_ctx is None or not extracted:
        return extracted, ""
    candidate = getattr(tools_ctx, "_delivery_candidate", None)
    candidate = candidate if isinstance(candidate, DeliveryCandidate) else None
    armed = bool(getattr(tools_ctx, "_delivery_control_required", False)) or (
        candidate is not None and _delivery_replace_required(candidate)
    )
    if not armed:
        return extracted, ""
    tools_ctx._delivery_control_required = False
    parsed, duplicate_protocol_key = _parse_delivery_control_object(extracted)
    # Protocol intent: any parsed object with the protocol key (unknown verb =
    # broken control, never prose), or JSON-looking text that fails to parse (a
    # mangled protocol attempt under the armed latch — the candidate is the answer).
    protocol_intent = duplicate_protocol_key or (
        ("delivery_control" in parsed)
        if isinstance(parsed, dict)
        else extracted.lstrip().startswith("{")
    )
    if not protocol_intent:
        # An ordinary prose answer under an armed latch: the fresh text stands.
        return extracted, ""
    selected = str(parsed.get("delivery_control") or "") if isinstance(parsed, dict) else ""
    if selected == "replace" and set(parsed) == {"delivery_control", "full_answer"}:
        replacement = parsed.get("full_answer")
        if isinstance(replacement, str) and replacement.strip():
            return replacement, ""
    elif selected == "keep" and set(parsed) == {"delivery_control"} and candidate is not None:
        return candidate.full_text, ""
    # Malformed/duplicate/invalid control: preserve the retained candidate (or,
    # with none retained, let the caller's fallback text stand) and say so.
    return (
        candidate.full_text if candidate is not None else "",
        REASON_DELIVERY_CONTROL_DEGRADED,
    )


def _forced_delegation_note(tools_ctx: Any, llm_trace: Dict[str, Any]) -> str:
    """The nanny postcondition's forced-path half, grounded in DURABLE custody.

    A forced finalization may not re-loop, so the substrate fact rides the
    one final prompt. `delegate_custody.task_execution_evidence` on the
    custody root (canonical/budget root — the Phase A split-root rule)
    decides, not just this execution's trace: succeeded → no note;
    started-but-unsettled → pending wording (no retry pressure);
    settled-without-success → truthful failure wording; zero started with
    readable evidence → no-delegation wording; unreadable → no accusation."""
    if not getattr(tools_ctx, "_nanny_route_dispatched", False):
        return ""
    try:
        from ouroboros import delegate_custody

        root = delegate_custody.custody_root(tools_ctx)
        log_path = delegate_custody.event_log_path(root)
        if log_path.exists():
            # _iter_rows swallows OSError, which would misread an unreadable log
            # as "zero runs" — probe readability so absence of rows is a fact.
            log_path.open("rb").close()
        evidence = delegate_custody.task_execution_evidence(
            root, str(getattr(tools_ctx, "task_id", "") or ""),
        )
    except Exception:
        log.debug("Forced-path custody evidence unreadable; nanny note skipped", exc_info=True)
        return ""
    started = int(evidence.get("delegated_runs_started") or 0)
    settled = int(evidence.get("delegated_runs_settled") or 0)
    if int(evidence.get("delegated_runs_succeeded") or 0):
        # Proportional silence must not extend to FORCED exits (grok / F16):
        # an overrun-forced wrap-up still owes the parent the honest-spend
        # line. One shot riding the single forced prompt — never a re-loop.
        rounds, cost = _nanny_metered_since_delegate_activity(tools_ctx)
        from ouroboros.task_pacing import NANNY_REMINDER_ROUNDS, NANNY_REMINDER_USD

        if rounds >= NANNY_REMINDER_ROUNDS or cost >= NANNY_REMINDER_USD:
            return (
                "\nNOTE: your delegated run(s) succeeded, but you have since spent "
                f"{_nanny_burn_phrase(rounds, cost)} beyond your last act of delegation. "
                "Account for that spend honestly in your answer."
            )
        return ""
    if started > settled:
        return (
            "\nNOTE: this task dispatched delegated run(s) that have not settled "
            f"yet ({started - settled} of {started} pending). State their status "
            "in your answer; do not claim the delegated work finished."
        )
    if settled:
        return (
            f"\nNOTE: this task's delegated run(s) settled WITHOUT success ({settled} "
            "run(s)). State that failure and its impact honestly in your answer."
        )
    if evidence.get("delegate_start_attempted") or any(
        str(c.get("tool") or "") == "delegate_start"
        for c in (llm_trace.get("tool_calls") or []) if isinstance(c, dict)
    ):
        # A durable or current-trace start attempt is not a refusal to delegate.
        return ""
    return (
        "\nNOTE: this task was dispatched onto the delegated substrate "
        "(executor=harness) and made no delegate_start calls — the work ran on "
        "metered API tokens. State why in your answer."
    )


def _forced_final_answer(
    ctx: _RoundLimitContext,
    *,
    prompt: str,
    fallback_text: str,
    reason_code: str,
    single_semantic_turn: bool = False,
    provider_terminal: bool = False,
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """Forced rail."""
    live_trace = getattr(ctx, "llm_trace", None)
    llm_trace = live_trace if isinstance(live_trace, dict) else {}
    _finalize_forced_services(ctx, llm_trace)
    if ctx.deadline_ts is not None and time.time() >= float(ctx.deadline_ts):
        ctx.accumulated_usage.update(execution_status="failed", reason_code=reason_code)
        return _forced_fallback_result(
            ctx, llm_trace, fallback_text, reason_code,
            source=f"{reason_code}_window_elapsed",
        )
    router_result = _forced_swarm_router_result(ctx, llm_trace, reason_code)
    if router_result is not None:
        return router_result
    tools_ctx = getattr(getattr(ctx, "tools", None), "_ctx", None)
    prompt += _forced_delegation_note(tools_ctx, llm_trace)
    _append_or_merge_user_message(ctx.messages, prompt)
    extracted = ""
    response_meta: Dict[str, Any] = {}
    for attempt in range(1 if single_semantic_turn else 2):
        try:
            ctx.accumulated_usage.pop("_forced_response_meta", None)
            extracted, response_meta = forced_response_parts(
                _call_forced_model_once(ctx), ctx.accumulated_usage,
            )
        except BudgetExceeded:
            _drain_forced_owner_directives(ctx, llm_trace)
            raise
        except Exception:
            log.warning("Failed to get final response after %s", reason_code, exc_info=True)
            extracted = ""
            response_meta = {}
        ctx.accumulated_usage["execution_status"] = "failed"
        ctx.accumulated_usage["reason_code"] = reason_code
        if not _drain_forced_owner_directives(ctx, llm_trace):
            break
        if str(ctx.accumulated_usage.get("_last_llm_error_kind") or "") == "provider_outcome_unknown":
            return _forced_fallback_result(
                ctx, llm_trace,
                "⚠️ Provider outcome unknown; directive retained. Resume the task without a blind resend.",
                reason_code,
                source="provider_outcome_unknown_no_resend",
            )
        if attempt == 1:
            return _forced_fallback_result(
                ctx,
                llm_trace,
                "⚠️ Another directive arrived. Resume the task for a current answer.",
                reason_code,
                source="late_owner_directive_requires_resume",
                provider_terminal=provider_terminal,
            )
        _finalize_forced_services(ctx, llm_trace)
        _append_or_merge_user_message(
            ctx.messages,
            "[FORCED_OWNER_REFRESH] Answer all current directives; ignore the stale draft.",
        )

    if provider_terminal and extracted and forced_response_is_incomplete(response_meta):
        return _forced_fallback_result(
            ctx, llm_trace, extracted, reason_code,
            source="forced_model_incomplete", provider_terminal=True,
        )

    extracted, control_degraded = _resolve_forced_delivery_control(tools_ctx, extracted)
    if extracted:
        ctx.accumulated_usage["_best_effort_extracted"] = True
        plan_suffix = (
            _force_plan_disclosure(tools_ctx, llm_trace, forced_reason=reason_code)
            if tools_ctx is not None else ""
        )
        if provider_terminal:
            ctx.accumulated_usage["terminal_plan_review_open"] = bool(plan_suffix)
        full_text = extracted if provider_terminal else _compose_delivery_suffix(
            extracted, plan_suffix + _forced_orphan_note(ctx),
        )
        if provider_terminal:
            ctx.accumulated_usage["terminal_origin"] = TERMINAL_ORIGIN_MODEL_FINAL
        candidate = _publish_model_forced_candidate(
            ctx, llm_trace, full_text, reason_code,
        )
        if control_degraded and candidate is not None:
            candidate.degraded_reason = control_degraded
            llm_trace.setdefault("reasoning_notes", []).append(
                "Forced finalization received an invalid delivery-control object; "
                "preserved the retained complete answer."
            )
            if getattr(ctx, "tools", None) is not None:
                _publish_delivery_candidate(ctx.tools, candidate, llm_trace)
        _record_forced_finalization(
            ctx,
            llm_trace,
            reason_code=reason_code,
            source="model",
            candidate=candidate,
        )
        return (
            candidate.full_text if candidate is not None else full_text,
            ctx.accumulated_usage,
            llm_trace,
        )
    return _forced_fallback_result(
        ctx,
        llm_trace,
        fallback_text,
        reason_code,
        provider_terminal=provider_terminal,
    )


def _apply_runtime_overrides(
    ctx: Any,
    active_model: str,
    active_use_local: bool,
    active_effort: str,
) -> Tuple[str, bool, str]:
    """Apply one-shot per-round model/locality/effort overrides from tool ctx."""
    if ctx.active_model_override:
        active_model = ctx.active_model_override
        ctx.active_model_override = None
    if getattr(ctx, "active_use_local_override", None) is not None:
        active_use_local = ctx.active_use_local_override
        ctx.active_use_local_override = None
    if ctx.active_effort_override:
        active_effort = normalize_reasoning_effort(ctx.active_effort_override, default=active_effort)
        ctx.active_effort_override = None
    return active_model, active_use_local, active_effort


def _apply_overrides_and_regate_mode(ctx, active_model, active_use_local, active_effort, active_context_mode):
    """Apply per-round overrides; route rebind never predicts a mode change."""
    active_model, active_use_local, active_effort = _apply_runtime_overrides(
        ctx, active_model, active_use_local, active_effort,
    )
    return active_model, active_use_local, active_effort, active_context_mode


def _rebind_context_fit_plan(
    plan: Any,
    tools: ToolRegistry,
    messages: List[Dict[str, Any]],
    *,
    model: str,
    use_local: bool,
    preferred_mode: str,
    tool_schemas: List[Dict[str, Any]],
) -> Tuple[Any, str]:
    """Recalibrate the captured immutable core for one new exact route.

    Route switches reuse the plan's already-rendered Low/Max projections; only
    exact-route evidence, calibration, and fit are rebound.  This avoids both a
    stale initial-route retry plan and a second context-builder/intent corpus.
    """
    if plan is None or not all(
        hasattr(plan, name) for name in ("max_projection", "low_projection", "core_sha256")
    ):
        raise RuntimeError(
            "CONTEXT_FIT_REBUILD_FAILED: immutable context core is unavailable for route switch"
        )
    from ouroboros.capability_evidence import is_known
    from ouroboros.context import _context_fit_route
    from ouroboros.context_fit import _failed_route_evidence, _route_calibration_ratio

    metadata = getattr(tools._ctx, "task_metadata", {})
    metadata = metadata if isinstance(metadata, dict) else {}
    task = {
        "model": model,
        "use_local_model": use_local,
        "task_metadata": metadata,
        "delegation_role": metadata.get("delegation_role"),
    }
    is_subagent = str(metadata.get("delegation_role") or "").lower() == "subagent"
    try:
        route, evidence = _context_fit_route(task, allow_fetch=not is_subagent)
    except Exception:
        log.debug("Route-switch capability probe failed; preserving unknown Max", exc_info=True)
        route, evidence = _failed_route_evidence(task)
    ratio = _route_calibration_ratio(
        None,  # canonical evidence root (one observation store)
        str(getattr(evidence, "route_fp", "") or ""),
        str(route.get("model") or model),
    )
    known_window = is_known(evidence, require_fresh=True)
    window_tokens = int(getattr(evidence, "window_tokens", 0) or 0)

    def project(projection: Any) -> Any:
        calibrated = int(int(projection.estimated_tokens or 0) * ratio)
        fits = (
            calibrated + int(plan.output_reserve_tokens or 0) <= window_tokens
            if known_window else None
        )
        return replace(
            projection,
            calibrated_tokens=calibrated,
            calibration_ratio=ratio,
            fits_known_window=fits,
        )

    max_projection = project(plan.max_projection)
    low_projection = project(plan.low_projection)
    preferred = preferred_mode if preferred_mode in {"low", "max"} else "max"
    initial_mode = preferred
    rebound = replace(
        plan,
        preferred_mode=preferred,
        initial_mode=initial_mode,
        model=str(route.get("model") or model),
        provider=str(route.get("provider") or ""),
        route_fp=str(getattr(evidence, "route_fp", "") or ""),
        status=str(getattr(evidence, "status", "") or ""),
        stale=bool(getattr(evidence, "stale", False)),
        window_tokens=window_tokens,
        max_projection=max_projection,
        low_projection=low_projection,
    )
    mode = initial_mode
    projected_prompt_tokens = rebound.projected_tokens_with_tools(mode, tool_schemas)
    messages[:] = rebound.reproject_transcript(messages, mode)
    tools._ctx.context_fit_plan = rebound
    tools._ctx.messages = messages
    tools._ctx.active_context_mode = mode
    try:
        _emit_checkpoint_event(
            getattr(tools._ctx, "event_queue", None),
            str(getattr(tools._ctx, "task_id", "") or ""),
            tools._ctx.drive_logs(),
            {
                "checkpoint_kind": "context_fit_route_rebound",
                "model": rebound.model,
                "route_fp": rebound.route_fp,
                "core_sha256": rebound.core_sha256,
                "preferred_mode": preferred,
                "effective_mode": mode,
                "evidence_status": rebound.status,
                "window_tokens": rebound.window_tokens,
                "projected_prompt_tokens": projected_prompt_tokens,
            },
        )
    except Exception:
        log.debug("Failed to emit route-switch context-fit checkpoint", exc_info=True)
    return rebound, mode


def _visible_round_text(content: Any) -> str:
    """The round's visible assistant text as a plain string. ``content`` may be
    a string OR a list of typed blocks; collect the ``text`` of every block
    EXCEPT reasoning ones (Anthropic ``thinking``/``redacted_thinking``,
    Gemini ``part.thought``) — the exact complement of
    extract_display_reasoning. A regular Gemini part carries ``text`` with NO
    ``type``, so key on the ABSENCE of a reasoning marker (not ``type ==
    'text'``) to avoid dropping real answer text; a non-empty block list never
    stringifies to a raw repr, and a thinking-only list correctly reads as 'no
    visible text' (narration falls back to readable reasoning)."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        out: List[str] = []
        for b in content:
            if not isinstance(b, dict):
                continue
            if str(b.get("type") or "") in ("thinking", "reasoning", "redacted_thinking") or b.get("thought") is True:
                continue  # reasoning/thinking blocks are display reasoning, not visible answer text
            txt = b.get("text")
            if isinstance(txt, str):
                out.append(txt)
        return "".join(out).strip()
    return ""


def _emit_round_progress(content: Any, msg: Dict[str, Any], emit_progress, llm_trace: Dict[str, Any]) -> None:
    """Emit redacted progress safely to users.

    Visible text is retained in ``reasoning_notes``. Provider reasoning stays
    display-only; the native message and transcript remain unchanged.
    """
    visible_text = _visible_round_text(content)
    if visible_text:
        safe_text = sanitize_tool_result_for_log(visible_text)
        emit_progress(safe_text)
        llm_trace["reasoning_notes"].append(safe_text)
    elif str(os.environ.get("OUROBOROS_REASONING_SUMMARY", "auto")).strip().lower() != "off":
        display_reasoning = LLMClient.extract_display_reasoning(msg)
        if display_reasoning:
            emit_progress(sanitize_tool_result_for_log(display_reasoning))


def _nanny_finalization_message(
    tools: ToolRegistry, drive_root: pathlib.Path, task_id: str,
    trace_attempted: bool = False,
) -> str:
    """The honest nanny reminder for a harness-dispatched child at finalization —
    or '' when no reminder is deserved.

    F4 (2026-08-10 saga): the old reminder accused children whose delegated
    runs CRASHED of "choosing" not to delegate, and fired even with the verbs
    policy-hidden. Two structural facts fix both: the task's own visible
    toolset and durable custody evidence (delegate_custody.
    task_execution_evidence) spanning the WHOLE task — per-execution llm_trace
    resets on continuation. `trace_attempted` is the third fact: a
    delegate_start in THIS execution's trace; it must not suppress the failure
    message (triad, e84475f2: delegate, run dies, finish by hand, finalize —
    one execution), only the accusation when custody has no rows yet (a
    pending/uncustodied start is an attempt, not a choice)."""
    try:
        if "delegate_start" not in set(tools.available_tools()):
            return ""  # the verbs are invisible here; "you chose not to" would be false
    except Exception:
        log.debug("nanny nudge: toolset visibility check failed", exc_info=True)
    evidence: Dict[str, Any] = {}
    try:
        from ouroboros.delegate_custody import custody_root, task_execution_evidence

        # Split-root fix (2026-08-10): custody WRITES land on the CANONICAL
        # (budget) root, but this read used the loop's drive_root — a
        # split-root child drive has no custody rows: nanny blind. Resolve
        # the writers' SAME root; drive_root stays the unit-stub fallback.
        try:
            evidence_root = custody_root(tools._ctx)
        except Exception:
            evidence_root = drive_root
        evidence = task_execution_evidence(evidence_root, str(task_id or ""))
    except Exception:
        log.debug("nanny nudge: custody evidence read failed", exc_info=True)
    if evidence.get("delegated_runs_succeeded"):
        # The route WAS used and worked — but "used once" is no permanent
        # license: the poltergeist children each ran ONE successful $0 run then
        # co-built for tens of opus rounds while this early return kept the
        # nudge silent. Silence is proportional to the measured burn since the
        # last delegated-run activity.
        rounds, cost = _nanny_metered_since_delegate_activity(tools._ctx)
        from ouroboros.task_pacing import NANNY_REMINDER_ROUNDS, NANNY_REMINDER_USD

        if rounds < NANNY_REMINDER_ROUNDS and cost < NANNY_REMINDER_USD:
            return ""
        return (
            "⚠️ NANNY_METERED_OVERRUN: your delegated run(s) succeeded, but you have "
            f"since spent {_nanny_burn_phrase(rounds, cost)} beyond your last act of "
            "delegation (supervision included). "
            "A successful run is verified and integrated, not rebuilt. If "
            "the remaining work is substantive, delegate it (a new delegate_start); "
            "if you are wrapping up, keep the wrap-up short and account for the "
            "metered spend honestly in your result."
        )
    started = int(evidence.get("delegated_runs_started") or 0)
    if not started and (evidence.get("evidence_read_failed") or not evidence):
        # Unreadable custody (a5e59bdf): configured -> unknown message; legacy -> silence.
        from ouroboros.subagent_bootstrap import configured_actor_finalization_message

        _actor = configured_actor_finalization_message(
            tools._ctx, task_id=str(task_id or ""), fallback_root=drive_root)
        return _actor or ""
    if not started and (trace_attempted or evidence.get("delegate_start_attempted")):
        # Pending, refused or uncustodied starts are still real attempts.
        return ""
    settled = int(evidence.get("delegated_runs_settled") or 0)
    failure_states = [str(s) for s in (evidence.get("delegated_run_failure_states") or [])]
    pending = max(0, started - settled)
    if pending:
        # PENDING ≠ FAILED (sol review, b49f8192): a STARTED row without
        # settlement may still be executing — calling it failed invites a
        # duplicate, and finalizing over it orphans the result. Outranks the
        # failed message: with a run in flight, "retry" is wrong even when an
        # earlier sibling died (still a fact below).
        failed_note = (
            f" {len(failure_states)} earlier run(s) already ended: {', '.join(failure_states)}."
            if failure_states else ""
        )
        return (
            "⚠️ NANNY_DELEGATED_RUN_PENDING: you routed work onto the delegated "
            f"substrate and {pending} delegated run(s) have started but not "
            "settled — they may still be executing. Do not finalize over an "
            "in-flight delegated run (its result would be orphaned) and do not "
            "start a duplicate: wait for or check it (delegate_wait) before "
            "finalizing, or cancel it (delegate_cancel) and say so." + failed_note
        )
    if started:
        states = ", ".join(failure_states) or "settled without a recorded terminal state"
        return (
            "⚠️ NANNY_DELEGATED_RUN_FAILED: you DID route work onto the delegated "
            f"substrate ({started} run(s) started), but none succeeded — your "
            f"delegated run(s) ended: {states}. Do not finalize as if delegation "
            "was never attempted: either retry it (delegate_start / delegate_wait) "
            "or state in your final answer that the delegated run failed and why "
            "the remaining work ran on metered API tokens."
        )
    from ouroboros.subagent_bootstrap import configured_actor_finalization_message

    _actor = configured_actor_finalization_message(
        tools._ctx, task_id=str(task_id or ""), fallback_root=drive_root)
    if _actor is not None:
        return _actor
    return (
        "⚠️ NANNY_DID_NOT_DELEGATE: this task was dispatched onto the delegated "
        "substrate (executor=harness), but you are finalizing with ZERO "
        "delegate_start calls — the work would end up billed to metered API "
        "tokens the parent asked to avoid. Either delegate the remaining work "
        "now (delegate_start / delegate_wait), or finalize with an explicit "
        "statement of WHY delegation was not used (route refused, work shape "
        "unsuited, deadline) so your parent sees the substrate decision."
    )


def _maybe_inject_finalization_nudges(
    tools: ToolRegistry, drive_root: Optional[pathlib.Path], task_id: str,
    llm_trace: Dict[str, Any], content: Optional[str], messages: List[Dict[str, Any]],
    emit_progress: Callable[[str], None],
) -> bool:
    """One-shot pre-finalization injections that each re-loop (return True): the skill
    finalization reminder, then the FR3 verify-before-done nudge. Extracted from
    run_llm_loop to keep it under the method size gate."""
    if drive_root is None:
        return False
    # Forked actors keep ordinary verification locally while actor-first
    # zero-run authority is canonical immediately. One non-empty replica must
    # not hide the other: verification/nudge decisions use the merged view.
    try:
        from ouroboros.outcomes import read_context_verification_receipts

        receipt_rows = read_context_verification_receipts(
            tools._ctx, task_id, fallback_root=drive_root,
        )
    except Exception:
        from ouroboros.outcomes import read_verification_receipts

        receipt_rows = read_verification_receipts(drive_root, task_id)
    if (getattr(tools._ctx, "_nanny_route_dispatched", False)
            and not getattr(tools._ctx, "_nanny_finalization_injected", False)):
        # Nanny postcondition (owner 2026-08-07): a harness-dispatched child
        # must not finalize as if that decision never existed. One structural
        # fact, one re-loop; delegating OR finalizing with a typed reason
        # both stay open — never a hard gate (P5). A delegate_start in THIS
        # trace rides into the message decision (triad, e84475f2);
        # suppressions live in _nanny_finalization_message.
        _trace_attempted = any(
            str(c.get("tool") or "") == "delegate_start"
            for c in (llm_trace.get("tool_calls") or [])
            if isinstance(c, dict)
        )
        tools._ctx._nanny_finalization_injected = True
        _nanny_msg = _nanny_finalization_message(
            tools, drive_root, task_id, trace_attempted=_trace_attempted,
        )
        if _nanny_msg:
            if content and content.strip():
                messages.append({"role": "assistant", "content": content})
            _append_or_merge_user_message(messages, f"[SYSTEM REMINDER]\n{_nanny_msg}")
            # Owner decision (2026-08-15): no owner-chat progress line — the
            # trace + typed task_checkpoint carry observability.
            _code = _nanny_msg.split(":", 1)[0].replace("⚠️", "").strip()
            _emit_checkpoint_event(
                getattr(tools._ctx, "event_queue", None), task_id,
                getattr(tools._ctx, "drive_logs", None),
                {"checkpoint_kind": "nanny_finalization_nudge",
                 "nanny_code": _code},
            )
            # B3: durable worker stamp that the nudge was really INJECTED (the
            # ctx flag is set even on suppression); read back at completion.
            from ouroboros.delegate_evidence import record_nanny_nudge_stamp

            record_nanny_nudge_stamp(tools._ctx, task_id, _code)
            llm_trace["reasoning_notes"].append(_nanny_msg)
            return True
    finalization_msg = _skill_finalization_message(drive_root, llm_trace)
    if finalization_msg and not getattr(tools._ctx, "_skill_finalization_injected", False):
        tools._ctx._skill_finalization_injected = True
        if content and content.strip():
            messages.append({"role": "assistant", "content": content})
        _append_or_merge_user_message(messages, f"[SYSTEM REMINDER]\n{finalization_msg}")
        emit_progress(finalization_msg)
        llm_trace["reasoning_notes"].append(finalization_msg)
        return True
    if not getattr(tools._ctx, "_verify_red_nudged", False):
        # Red-verification one-shot nudge: the latest host-attested verify
        # receipt is RED and unreconciled — finalizing over your own failing
        # check is a self-contradiction (P3/P12), distinct from receipt_absent
        # below ("no grounding" vs "grounding says FAIL"). BEFORE the FR3
        # verify nudge. Binary latch; advisory; forced-finalization paths
        # bypass it. Keyed on the typed receipt status, never content (P5).
        _failed_receipt = latest_unreconciled_failed_verification(
            drive_root, task_id, receipts=receipt_rows,
        )
        if _failed_receipt is not None:
            tools._ctx._verify_red_nudged = True
            _check = str(_failed_receipt.get("check") or "").strip()
            _rc = _failed_receipt.get("returncode")
            _on = f" on `{_check}`" if _check else ""
            _exit = f" (exit {_rc})" if _rc is not None else ""
            if content and content.strip():
                messages.append({"role": "assistant", "content": content})
            _append_or_merge_user_message(
                messages,
                "[SYSTEM REMINDER]\nYour latest host-attested verification is RED" + _on + _exit +
                ". Before a clean final answer, reconcile it: re-check it, explain why this check is "
                "not the task's acceptance contract, or fix and re-run verification. This is advisory — "
                "if you finalize anyway, make the residual risk explicit.",
            )
            emit_progress("Red-verification nudge injected before final response.")
            llm_trace["reasoning_notes"].append("Red-verification nudge injected before final response.")
            return True
    if not getattr(tools._ctx, "_verify_masked_nudged", False):
        # Exit-masking one-shot ADVISORY nudge (v6.52.2): a PASSING verify
        # check can LAUNDER the real exit code (`| tail`/`|| true` — the
        # false-green tutanota hit). Distinct from the red nudge; after it.
        # Binary latch; advisory; forced paths bypass it. Flag-driven on
        # typed receipt sensor, never content (P5).
        _masked_receipt = latest_unreconciled_masked_verification(
            drive_root, task_id, receipts=receipt_rows,
        )
        if _masked_receipt is not None:
            tools._ctx._verify_masked_nudged = True
            _mcheck = str(_masked_receipt.get("check") or "").strip()
            _mreasons = ", ".join(str(x) for x in (_masked_receipt.get("check_exit_masking_reasons") or []))
            _mon = f" on `{_mcheck}`" if _mcheck else ""
            _mwhy = f" ({_mreasons})" if _mreasons else ""
            if content and content.strip():
                messages.append({"role": "assistant", "content": content})
            _append_or_merge_user_message(
                messages,
                "[SYSTEM REMINDER]\nYour latest passing verification" + _mon + " uses a shell pipe" + _mwhy +
                " that can hide the real command's exit code, so a failing run could read as exit 0. "
                "Before a clean final answer, re-ground so the exit reflects the real result (drop the "
                "masking pipe / use the runner's own pass marker), or explain why it is reliable. This is "
                "advisory — if you finalize anyway, make the residual risk explicit.",
            )
            emit_progress("Masked-verification nudge injected before final response.")
            llm_trace["reasoning_notes"].append("Masked-verification nudge injected before final response.")
            return True
    if not getattr(tools._ctx, "_criterion_source_nudged", False):
        # Criterion-provenance one-shot ADVISORY nudge (v6.54.4): the latest
        # passing verification used an AGENT-DEFINED criterion with no stated
        # basis — green check, synthesized criterion. One reminder to confirm
        # equivalence with the task's real requirement (or state the basis via
        # criterion_basis). AFTER the masked nudge, BEFORE FR3. Flag-driven on
        # the typed receipt field, never content (P5); forced paths bypass.
        _agent_defined = latest_agent_defined_verification(
            drive_root, task_id, receipts=receipt_rows,
        )
        if _agent_defined is not None:
            tools._ctx._criterion_source_nudged = True
            _acheck = str(_agent_defined.get("check") or "").strip()
            _aon = f" (`{_acheck}`)" if _acheck else ""
            if content and content.strip():
                messages.append({"role": "assistant", "content": content})
            _append_or_merge_user_message(
                messages,
                "[SYSTEM REMINDER]\nYour latest passing verification" + _aon + " uses a success "
                "criterion YOU defined, not one the task states. Before finalizing, double-check the "
                "criterion is equivalent to what the task actually asks for (format, units, scope) — "
                "re-run verify_and_record with criterion_basis stating why it suffices, or adjust the "
                "check. Advisory only — if you finalize anyway, make the assumption explicit.",
            )
            emit_progress("Criterion-provenance nudge injected before final response.")
            llm_trace["reasoning_notes"].append("Criterion-provenance nudge injected before final response.")
            return True
    suppress_unavailable_zero_run_verify = False
    try:
        from ouroboros.outcomes import _terminal_zero_run_receipt_present
        from ouroboros.tool_access import active_tool_profile

        suppress_unavailable_zero_run_verify = (
            active_tool_profile(tools._ctx) == "local_readonly_subagent"
            and _terminal_zero_run_receipt_present(receipt_rows)
        )
    except Exception:
        pass
    if (
        not getattr(tools._ctx, "_verify_nudged", False)
        and not suppress_unavailable_zero_run_verify
        and should_nudge_verification(
            llm_trace, drive_root, task_id, receipts=receipt_rows,
        )
    ):
        # FR3 one-shot verify-before-done nudge: real effects, no
        # host-attested grounding yet. Binary latch, BEFORE the
        # acceptance-review gate so it reaches required and auto. Forced
        # finalization paths return earlier and bypass it (land best_effort).
        tools._ctx._verify_nudged = True
        if content and content.strip():
            messages.append({"role": "assistant", "content": content})
        _append_or_merge_user_message(
            messages,
            "[SYSTEM REMINDER]\nBefore finalizing: you produced a real deliverable but recorded no "
            "machine verification. Call verify_and_record — run your test/command (explicit_command/"
            "explicit_metric/visible_verifier), confirm the artifact exists (artifact_observation), or "
            "honestly declare no_visible_machine_contract — so the result is grounded, then continue.",
        )
        emit_progress("Verify-before-done nudge injected before final response.")
        llm_trace["reasoning_notes"].append("Verify-before-done nudge injected before final response.")
        return True
    # A3 one-shot no-op nudge: a declared deliverable (non-empty
    # expected_output) but NO tool calls, reviewable effects, or FINAL ANSWER
    # marker this turn — about-to-finalize-without-attempting (family of the
    # M2 expected_output_ungrounded flag). Own latch, AFTER the verify nudge;
    # never forces acceptance review; forced paths return earlier. Structural
    # facts only (no refusal-text matching).
    if (
        not getattr(tools._ctx, "_noop_attempt_nudged", False)
        and str(_contract_expected_output(tools._ctx)).strip()
        and not (llm_trace.get("tool_calls") or [])
        and not turn_has_reviewable_effects(llm_trace)
        and not extract_final_answer(content or "")
    ):
        tools._ctx._noop_attempt_nudged = True
        if content and content.strip():
            messages.append({"role": "assistant", "content": content})
        # v6.60.0: the nudge keys on expected_output SEMANTICS; it mentions the FINAL
        # ANSWER marker only when this task's contract actually declares the protocol.
        _marker_bit = (
            "no tool calls, no reviewable effects, no FINAL ANSWER"
            if _answer_protocol_active(tools._ctx)
            else "no tool calls, no reviewable effects, no delivered answer"
        )
        _append_or_merge_user_message(
            messages,
            "[SYSTEM REMINDER]\nThis task declares an expected output, but you are about to finalize "
            f"without having attempted it — {_marker_bit}. "
            "Actually attempt the task now (do the work / produce the deliverable / derive the answer), "
            "then finalize. If it is genuinely blocked, say so with the concrete blocker and evidence.",
        )
        emit_progress("No-op attempt nudge injected before final response.")
        llm_trace["reasoning_notes"].append("No-op attempt nudge injected before final response.")
        return True
    # P2 one-shot final-answer-marker nudge: REAL work + visible prose but
    # no FINAL ANSWER marker — the typed extractor would drop it, a forced
    # finalization score empty. Strengthen BEHAVIOR (agent marks its OWN
    # answer), never mine prose into a claimed answer (P5). Own latch, AFTER
    # verify/red/A3; forced paths return earlier. The protocol gate alone
    # suffices — it must not ALSO require expected_output: GAIA-shaped
    # contracts keep it empty; that extra gate once suppressed the only
    # salvage surface (v6.56.0: last-round refusal finalized empty).
    if (
        not getattr(tools._ctx, "_final_marker_nudged", False)
        and _answer_protocol_active(tools._ctx)  # v6.60.0: marker nudge is protocol-gated
        and content and content.strip()
        and not extract_final_answer(content or "")
        and ((llm_trace.get("tool_calls") or []) or turn_has_reviewable_effects(llm_trace))
    ):
        tools._ctx._final_marker_nudged = True
        messages.append({"role": "assistant", "content": content})
        _append_or_merge_user_message(
            messages,
            "[SYSTEM REMINDER]\nYou have done the work but have not marked a final answer. If you "
            "are done, end your response with a single line, exactly: FINAL ANSWER: <answer> — the "
            "bare deliverable only (a number / a few words / a short list), so it is captured even if "
            "the run is cut short. If you are not done, keep working.",
        )
        emit_progress("Final-answer marker nudge injected before final response.")
        llm_trace["reasoning_notes"].append("Final-answer marker nudge injected before final response.")
        return True
    return False


def _answer_protocol_active(ctx: Any) -> bool:
    """True when this task's contract declares answer_protocol="final_answer_line"
    (v6.60.0): the FINAL ANSWER marker instructions/nudges/pacing phrases are
    PROTOCOL-GATED — only adapter/exact-match tasks see them; ordinary chat and
    self-tasks never get marker prompting (the latch/extractor stay unconditional).
    Thin alias over the contracts SSOT gate."""
    from ouroboros.contracts.task_contract import answer_protocol_active

    return answer_protocol_active(ctx)


def _contract_expected_output(ctx: Any) -> str:
    """Read the declared expected_output (as carried on the task contract/metadata for the
    running ctx — the same declared field the M2 ungrounded flag keys on), for the A3 no-op nudge gate."""
    contract = getattr(ctx, "task_contract", {})
    if isinstance(contract, dict) and str(contract.get("expected_output") or "").strip():
        return str(contract.get("expected_output") or "")
    metadata = getattr(ctx, "task_metadata", {})
    if isinstance(metadata, dict):
        if str(metadata.get("expected_output") or "").strip():
            return str(metadata.get("expected_output") or "")
        meta_contract = metadata.get("task_contract")
        if isinstance(meta_contract, dict):
            return str(meta_contract.get("expected_output") or "")
    return ""


@dataclass
class _RoundModelCallContext:
    llm: LLMClient
    messages: List[Dict[str, Any]]
    tools: ToolRegistry
    context_fit_plan: Any
    active_model: str
    tool_schemas: List[Dict[str, Any]]
    active_effort: str
    max_retries: int
    drive_logs: pathlib.Path
    task_id: str
    round_idx: int
    event_queue: Optional[queue.Queue]
    accumulated_usage: Dict[str, Any]
    task_type: str
    active_use_local: bool
    active_context_mode: str
    drive_root: Optional[pathlib.Path]
    attempt_cap: Optional[int] = None


def _context_fit_round_id(ctx: _RoundModelCallContext) -> str:
    execution_id = str(ctx.accumulated_usage.setdefault("execution_id", new_execution_id()))
    return f"{execution_id}:round:{ctx.round_idx}"


def _main_context_profile(plan: Any, rendered_mode: str) -> str:
    if rendered_mode != "low":
        return "owner_max"
    # Effective Low is the sizing authority even when a bare env override
    # keeps owner intent Max for P3. A Low entered after a real Max overflow
    # is task-local and does not inherit the economy target T.
    return "owner_low" if str(getattr(plan, "preferred_mode", "")) == "low" else "task_local_low"


def _remember_main_fit(ctx: _RoundModelCallContext, disposition: Any) -> None:
    measurement = disposition.measurement
    usage = ctx.accumulated_usage
    usage["_context_route_fp"] = measurement.route_fp
    usage["_context_prompt_estimate"] = measurement.estimated_input_tokens
    usage["_context_fit_mode"] = measurement.rendered_mode
    usage["_context_profile"] = measurement.profile
    usage["_context_measurement_basis"] = measurement.measurement_basis
    usage["_context_measurement_density"] = measurement.measurement_density
    usage["_context_target_total_tokens"] = measurement.target_total_tokens
    usage["_context_capacity_total_tokens"] = measurement.capacity_total_tokens
    usage["_context_target_deficit_tokens"] = measurement.target_deficit_tokens
    usage["_context_capacity_deficit_tokens"] = measurement.capacity_deficit_tokens
    usage["_context_reclaim_goal_tokens"] = measurement.reclaim_goal_tokens
    usage["_context_target_miss"] = disposition.action == "send_target_miss"
    usage["_context_automatic_pass_used"] = disposition.automatic_pass_used
    usage["_context_predicted_capacity_miss"] = disposition.predicted_capacity_miss


def _measure_round_main_fit(
    ctx: _RoundModelCallContext,
    *,
    automatic_pass_used: bool,
) -> Any:
    plan = ctx.context_fit_plan
    if plan is None or str(ctx.active_model or "") != str(getattr(plan, "model", "") or ""):
        return None
    from ouroboros.context_fit import measure_main_fit

    rendered_mode = "low" if ctx.active_context_mode == "low" else "max"
    disposition = measure_main_fit(
        plan,
        ctx.messages,
        ctx.tool_schemas,
        profile=_main_context_profile(plan, rendered_mode),
        rendered_mode=rendered_mode,
        round_id=_context_fit_round_id(ctx),
        automatic_pass_used=automatic_pass_used,
        reasoning_effort=ctx.active_effort,
    )
    _remember_main_fit(ctx, disposition)
    return disposition


def _physical_context_for_fit(disposition: Any) -> PhysicalAttemptContext:
    measurement = disposition.measurement
    return PhysicalAttemptContext(
        profile=measurement.profile,
        rendered_mode=measurement.rendered_mode,
        measurement_basis=measurement.measurement_basis,
        route_fp=measurement.route_fp,
        round_id=measurement.round_id,
        target_total_tokens=measurement.target_total_tokens,
        capacity_total_tokens=measurement.capacity_total_tokens,
        context_target_miss=disposition.action == "send_target_miss",
        automatic_pass_used=disposition.automatic_pass_used,
    )


def _dispatch_round_model(
    ctx: _RoundModelCallContext,
    disposition: Any,
    *,
    attempt_cap: Optional[int],
    candidate_predicate: Optional[Callable[[Any], Any]] = None,
) -> Tuple[Any, float]:
    return call_llm_with_retry(
        ctx.llm,
        ctx.messages,
        ctx.active_model,
        ctx.tool_schemas,
        ctx.active_effort,
        ctx.max_retries,
        ctx.drive_logs,
        ctx.task_id,
        ctx.round_idx,
        ctx.event_queue,
        ctx.accumulated_usage,
        ctx.task_type,
        use_local=ctx.active_use_local,
        deadline_ts=_task_deadline_epoch(ctx.tools),
        transport_reserve_sec=task_pacing.get_finalization_grace_sec(),
        attempt_cap=attempt_cap,
        allow_server_web_search=_server_web_allowed_by_task(ctx.tools._ctx),
        physical_context=(
            _physical_context_for_fit(disposition) if disposition is not None else None
        ),
        candidate_predicate=candidate_predicate,
    )


def _run_main_reclaim(
    ctx: _RoundModelCallContext,
    disposition: Any,
    *,
    minimum_goal_tokens: int = 0,
) -> Any:
    measurement = disposition.measurement
    key = (measurement.route_fp, measurement.round_id)
    passes = _context_reclaim_passes(ctx.tools._ctx)
    if key in passes:
        return None
    request = ContextReclaimRequest(
        route_fp=measurement.route_fp,
        round_id=measurement.round_id,
        transcript_sha256=context_reclaim_transcript_sha256(ctx.messages),
        measurement_basis=measurement.measurement_basis,
        measurement_density=measurement.measurement_density,
        reclaim_goal_tokens=max(
            int(measurement.reclaim_goal_tokens),
            max(0, int(minimum_goal_tokens)),
        ),
        allow_partial_shrink=True,
    )
    rebuilt, receipt, usage = compact_tool_history_llm(
        ctx.messages,
        request=request,
        drive_root=pathlib.Path(ctx.drive_root or ctx.drive_logs.parent),
        task_id=ctx.task_id,
        negative_memo=reclaim_negative_memo(ctx.tools._ctx),
        trace_refs_by_tool_call_id=reclaim_trace_refs(ctx.tools._ctx),
    )
    passes.add(key)
    # The checkpoint is written only after non-empty selection and immediately
    # before map/fold, so it also covers a post-summary binding mismatch.
    if receipt.checkpoint_ref:
        _context_reclaim_materializations(ctx.tools._ctx).add(key)
    if usage:
        _account_compaction_usage(ctx.accumulated_usage, usage, ctx.event_queue, ctx.task_id)
    if receipt.status == "applied":
        ctx.messages[:] = rebuilt
        ctx.tools._ctx.messages = ctx.messages
        seal_task_transcript(ctx.messages)
        prune_reclaim_trace_refs(ctx.tools._ctx, ctx.messages)
    _emit_checkpoint_event(ctx.event_queue, ctx.task_id, ctx.drive_logs, {
        "type": "context_reclaim",
        "checkpoint_kind": "context_reclaim_automatic",
        "round": ctx.round_idx,
        "route_fp": measurement.route_fp,
        "round_id": measurement.round_id,
        "status": receipt.status,
        "reclaim_goal_tokens": request.reclaim_goal_tokens,
        "reclaimed_tokens": receipt.reclaimed_tokens,
        "goal_reached": receipt.goal_reached,
        "checkpoint_ref": receipt.checkpoint_ref,
    })
    return receipt


def _measure_after_reclaim(ctx: _RoundModelCallContext) -> Any:
    """Suppress a second pass while reporting whether a summarizer actually ran."""
    disposition = _measure_round_main_fit(ctx, automatic_pass_used=True)
    if disposition is None:
        return None
    key = (disposition.measurement.route_fp, disposition.measurement.round_id)
    used = key in _context_reclaim_materializations(ctx.tools._ctx)
    if disposition.automatic_pass_used != used:
        disposition = replace(disposition, automatic_pass_used=used)
        _remember_main_fit(ctx, disposition)
    return disposition


def _reproject_actual_overflow_low(ctx: _RoundModelCallContext) -> None:
    if ctx.active_context_mode == "low" or ctx.context_fit_plan is None:
        return
    ctx.messages[:] = ctx.context_fit_plan.reproject_transcript(ctx.messages, "low")
    ctx.active_context_mode = "low"
    ctx.tools._ctx.messages = ctx.messages
    ctx.tools._ctx.active_context_mode = "low"
    _emit_checkpoint_event(ctx.event_queue, ctx.task_id, ctx.drive_logs, {
        "checkpoint_kind": "context_fit_low_retry",
        "round": ctx.round_idx,
        "route_fp": str(getattr(ctx.context_fit_plan, "route_fp", "") or ""),
        "preferred_mode": str(getattr(ctx.context_fit_plan, "preferred_mode", "") or ""),
        "effective_mode": "low",
        "owner_visible": True,
    })


def _failed_capture_is_comparable(capture: Any) -> bool:
    return bool(
        capture is not None
        and capture.state in {"dispatched", "settled", "unresolved"}
        and capture.candidate_measurement_kind == "canonical_json_v1"
        and capture.candidate_raw_sha256
        and capture.candidate_context_size_bytes is not None
        and capture.physical_context is not None
    )


def _strict_context_shrink_predicate(failed: Any) -> Callable[[Any], bool]:
    def predicate(request: Any) -> bool:
        failed_context = failed.physical_context
        current_context = request.physical_context
        return bool(
            request.candidate_measurement_kind == "canonical_json_v1"
            and request.provider == failed.provider
            and request.model == failed.model
            and request.max_completion_tokens == failed.max_completion_tokens
            and current_context is not None
            and failed_context is not None
            and current_context.route_fp == failed_context.route_fp
            and current_context.round_id == failed_context.round_id
            and request.candidate_raw_sha256 != failed.candidate_raw_sha256
            and request.candidate_context_size_bytes is not None
            and int(request.candidate_context_size_bytes) < int(failed.candidate_context_size_bytes)
        )

    return predicate


def _emit_overflow_retry_skipped(ctx: _RoundModelCallContext, reason: str) -> None:
    _emit_checkpoint_event(ctx.event_queue, ctx.task_id, ctx.drive_logs, {
        "type": "context_overflow_retry_skipped",
        "round": ctx.round_idx,
        "route_fp": str(getattr(ctx.context_fit_plan, "route_fp", "") or ""),
        "reason": reason,
    })


def _call_round_model(ctx: _RoundModelCallContext) -> Tuple[Any, float, str]:
    """Measure, optionally reclaim, dispatch, and recover one Main round."""
    disposition = _measure_round_main_fit(ctx, automatic_pass_used=False)
    if disposition is not None:
        key = (disposition.measurement.route_fp, disposition.measurement.round_id)
        already_reclaimed = key in _context_reclaim_passes(ctx.tools._ctx)
        if disposition.action == "reclaim_once" and not already_reclaimed:
            _run_main_reclaim(ctx, disposition)
            already_reclaimed = True
        if already_reclaimed:
            disposition = _measure_after_reclaim(ctx)

    msg, cost = _dispatch_round_model(
        ctx,
        disposition,
        attempt_cap=ctx.attempt_cap,
    )
    if msg is not None or str(ctx.accumulated_usage.get("_last_llm_error_kind") or "") != "context_overflow":
        return msg, cost, ctx.active_context_mode

    # Snapshot immediately: a reclaim summarizer is itself physically receipted
    # and would otherwise replace the failed Main candidate in the ContextVar.
    failed_capture = last_physical_attempt_capture()
    if disposition is None:
        return msg, cost, ctx.active_context_mode
    _reproject_actual_overflow_low(ctx)
    reclaim_key = (disposition.measurement.route_fp, disposition.measurement.round_id)
    overflow_fit = (
        _measure_after_reclaim(ctx)
        if reclaim_key in _context_reclaim_passes(ctx.tools._ctx)
        else _measure_round_main_fit(ctx, automatic_pass_used=False)
    )
    if overflow_fit is None:
        return msg, cost, ctx.active_context_mode
    key = (overflow_fit.measurement.route_fp, overflow_fit.measurement.round_id)
    if key not in _context_reclaim_passes(ctx.tools._ctx):
        _run_main_reclaim(ctx, overflow_fit, minimum_goal_tokens=1)
        overflow_fit = _measure_after_reclaim(ctx)
        if overflow_fit is None:
            return msg, cost, ctx.active_context_mode

    retries = _context_overflow_retries(ctx.tools._ctx)
    if key in retries:
        _emit_overflow_retry_skipped(ctx, "route_round_retry_already_used")
        return msg, cost, ctx.active_context_mode
    if not _failed_capture_is_comparable(failed_capture):
        _emit_overflow_retry_skipped(ctx, "failed_candidate_not_comparable")
        return msg, cost, ctx.active_context_mode
    retries.add(key)
    try:
        retry_msg, retry_cost = _dispatch_round_model(
            ctx,
            overflow_fit,
            attempt_cap=1,
            candidate_predicate=_strict_context_shrink_predicate(
                failed_capture,
            ),
        )
    except PhysicalAttemptPreconditionFailed:
        _emit_overflow_retry_skipped(ctx, "context_candidate_not_strictly_smaller")
        return msg, cost, ctx.active_context_mode
    return retry_msg, retry_cost, ctx.active_context_mode


@dataclass
class _LoopExitContext:
    tools: ToolRegistry
    drive_root: Optional[pathlib.Path]
    task_id: str
    event_queue: Optional[queue.Queue]
    drive_logs: pathlib.Path
    accumulated_usage: Dict[str, Any]
    llm_trace: Dict[str, Any]


def _handle_budget_exceeded(
    exc: BudgetExceeded,
    ctx: _LoopExitContext,
    *,
    limit_ctx: Optional[_RoundLimitContext] = None,
    episode: Optional[TransportWaitEpisode] = None,
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """Apply the physical-attempt dispatch rail without spending a wrap-up call."""
    if episode is not None:
        # Budget rail fired mid transport-wait: close the episode's durable story.
        _end_episode_budget(
            episode, ctx.drive_logs, ctx.task_id,
            limit_ctx.active_model if limit_ctx is not None else "")
    physical_calls: Optional[int] = None
    try:
        from ouroboros.usage_accounting import usage_breakdown

        budget_root = (
            getattr(ctx.tools._ctx, "budget_drive_root", None)
            or ctx.drive_root
            or getattr(ctx.tools._ctx, "drive_root", None)
        )
        if budget_root is not None:
            attempt_evidence = usage_breakdown(
                pathlib.Path(budget_root), task_id=str(ctx.task_id),
            )
            physical_calls = int(attempt_evidence.get("physical_calls") or 0)
            if attempt_evidence.get("integrity_degraded"):
                physical_calls = None
    except Exception:
        log.exception("Could not inspect task attempts after budget rail")
    direct_chat = bool(getattr(ctx.tools._ctx, "is_direct_chat", False))
    replay_safe = physical_calls == 0 and not direct_chat
    scope = str(getattr(exc, "limit_scope", "global") or "global")
    resource_limit = {
        "status": "paused_before_dispatch" if replay_safe else "resource_limited",
        "scope": scope,
        "root_task_id": str(getattr(exc, "root_task_id", "") or ""),
        "physical_calls": physical_calls,
        "replay_safe": replay_safe,
        "auto_resume": False,
        "resume_policy": (
            "increase_or_reset_budget_then_retry"
            if direct_chat
            else ("manual_same_generation" if replay_safe else "cancel_or_new_run")
        ),
    }
    if replay_safe:
        raise exc
    ctx.accumulated_usage["execution_status"] = "failed"
    ctx.accumulated_usage["reason_code"] = "budget_exhausted"
    ctx.accumulated_usage["resource_limit"] = resource_limit
    ctx.llm_trace["resource_limit"] = resource_limit
    _emit_checkpoint_event(ctx.event_queue, ctx.task_id, ctx.drive_logs, {
        "checkpoint_kind": "budget_scope_paused",
        "owner_visible": True,
        "toast_once": f"{ctx.task_id}:budget-paused:{scope}",
        **resource_limit,
    })
    if (
        scope == "root"
        and ctx.event_queue is not None
        and not bool(getattr(ctx.tools._ctx, "is_direct_chat", False))
    ):
        try:
            ctx.event_queue.put_nowait({
                "type": "budget_root_fence",
                "task_id": ctx.task_id,
                "root_task_id": resource_limit["root_task_id"],
                "resource_limit": resource_limit,
            })
        except Exception:
            log.error("Could not publish root budget fence for %s", ctx.task_id, exc_info=True)
    # A physical budget rail is terminal for this execution. Finalize task
    # services before testing or creating a DeliveryCandidate so no
    # pre-teardown answer is published on stale service/output evidence. The
    # outer cleanup repeats this helper only as an idempotent safety net.
    if limit_ctx is not None:
        limit_ctx.tools = ctx.tools
        limit_ctx.llm_trace = ctx.llm_trace
        _finalize_forced_services(limit_ctx, ctx.llm_trace)
    else:
        _finalize_task_services(ctx)
    candidate_seen: Optional[DeliveryCandidate] = None
    if limit_ctx is not None:
        # The exception can arrive after a substantive answer entered a service
        # re-loop. Re-read the live evidence now; the round-start snapshot alone
        # cannot prove that candidate is still current.
        limit_ctx.tools = ctx.tools
        limit_ctx.llm_trace = ctx.llm_trace
        candidate_seen = _live_delivery_candidate(limit_ctx)
        current_candidate = _current_delivery_candidate(limit_ctx, ctx.llm_trace)
        if current_candidate is not None:
            return _forced_fallback_result(
                limit_ctx,
                ctx.llm_trace,
                current_candidate.full_text,
                "budget_exhausted",
                source="budget_host_fallback",
                retained_source="budget_preserve",
                retained_control="budget_preserve",
            )
        if candidate_seen is not None:
            candidate_seen.degraded = True
            candidate_seen.degraded_reason = "budget_exhausted"
            candidate_seen.finalization_control = "budget_stale_rejected"
            _publish_delivery_candidate(ctx.tools, candidate_seen, ctx.llm_trace)
    latched = str(ctx.llm_trace.get("best_valid_final_answer") or "").strip()
    latched_is_current = (
        latched
        and len(ctx.llm_trace.get("tool_calls") or [])
        <= int(ctx.llm_trace.get("best_valid_final_answer_tools") or 0)
    )
    if latched_is_current:
        ctx.accumulated_usage["_best_effort_extracted"] = True
        if limit_ctx is not None:
            return _forced_fallback_result(
                limit_ctx,
                ctx.llm_trace,
                latched,
                "budget_exhausted",
                source="budget_latched_fallback",
            )
        return latched, ctx.accumulated_usage, ctx.llm_trace
    if candidate_seen is not None and limit_ctx is not None:
        return _forced_fallback_result(
            limit_ctx,
            ctx.llm_trace,
            candidate_seen.full_text,
            "budget_exhausted",
            source="budget_stale_candidate_preserved",
        )
    message = (
        "🚫 Model budget exhausted before another model dispatch. Increase or reset "
        "the global/root budget, then retry or resume the request. Starting a new run "
        "before changing the budget will hit the same limit."
        if direct_chat
        else (
            "🚫 Resource limit reached before another model dispatch. The task was not "
            "auto-resumed; cancel it or start a new run unless the recorded checkpoint "
            "is explicitly replay-safe."
        )
    )
    if limit_ctx is not None:
        return _forced_fallback_result(
            limit_ctx,
            ctx.llm_trace,
            message,
            "budget_exhausted",
            source="budget_host_fallback",
        )
    return message, ctx.accumulated_usage, ctx.llm_trace


def _cleanup_loop_resources(
    stateful_executor: Any,
    ctx: _LoopExitContext,
) -> None:
    """Release attempt-scoped executors, services, and delegated runs."""
    if stateful_executor:
        try:
            from ouroboros.tools.browser import cleanup_browser

            stateful_executor.submit(cleanup_browser, ctx.tools._ctx).result(timeout=5)
        except Exception:
            log.debug("Browser cleanup on executor thread failed or timed out", exc_info=True)
        try:
            stateful_executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            log.warning("Failed to shutdown stateful executor", exc_info=True)
    _finalize_task_services(ctx)
    # The full DeliveryCandidate is loop-local: only its compact
    # hash/revision projection remains in llm_trace after this cleanup. Clear
    # it after the idempotent teardown safety net so cleanup cannot erase the
    # only complete answer before service evidence lands.
    ctx.tools._ctx._delivery_candidate = None
    ctx.tools._ctx._delivery_control_required = False
    if ctx.drive_root is None or not ctx.task_id:
        return
    try:
        from ouroboros.delegate_custody import custody_root, release_task_runs

        # A delegated run is a resource this task HOLDS, like a service or
        # an executor: a terminalized parent leaving one running has a
        # mutating process nothing is watching. The durable reconciler still
        # covers a worker dying before here; this is the ordinary path.
        release_task_runs(custody_root(ctx.tools._ctx), ctx.task_id)
    except Exception:
        log.debug("Failed to release delegated runs for task %s", ctx.task_id, exc_info=True)
def _service_identity_projection(service: Dict[str, Any]) -> Dict[str, Any]:
    """Bounded identity used to deduplicate idempotent teardown observations."""

    fields = (
        "service_id",
        "name",
        "task_id",
        "lifecycle",
        "backend",
        "pid",
        "port",
        "artifact_outputs",
        "artifact_output_failed",
        "artifact_audit_gap",
        "log_finalization",
    )
    return {
        key: service.get(key)
        for key in fields
        if service.get(key) not in (None, "", [], {})
    }


def _finalize_task_services(ctx: _LoopExitContext) -> bool:
    """Finalize newly observed task services and record answer-bound evidence.

    Returns True only when a new stopped/kept/error observation was added.  The
    same helper is safe both immediately before acceptance and from ``finally``.
    """

    if ctx.drive_root is None or not ctx.task_id:
        return False
    try:
        from ouroboros.tools.services import stop_task_services

        finalized = stop_task_services(ctx.tools._ctx)
        seen = getattr(ctx.tools._ctx, "_service_finalization_signatures", None)
        if not isinstance(seen, set):
            seen = set()
            ctx.tools._ctx._service_finalization_signatures = seen
        fresh = []
        for service in finalized:
            if not isinstance(service, dict):
                continue
            signature = hashlib.sha256(json.dumps(
                _service_identity_projection(service),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")).hexdigest()
            if signature in seen:
                continue
            seen.add(signature)
            fresh.append(service)
        stopped = [service for service in fresh if service.get("lifecycle") != "kept"]
        kept = [service for service in fresh if service.get("lifecycle") == "kept"]
        if stopped:
            _emit_checkpoint_event(ctx.event_queue, ctx.task_id, ctx.drive_logs, {
                "checkpoint_kind": "services_stopped",
                "services": stopped,
            })
            ctx.llm_trace.setdefault("verification_events", []).append({
                "kind": "services_stopped",
                "services": stopped,
            })
        if kept:
            _emit_checkpoint_event(ctx.event_queue, ctx.task_id, ctx.drive_logs, {
                "checkpoint_kind": "services_kept",
                "services": kept,
            })
            ctx.llm_trace.setdefault("verification_events", []).append({
                "kind": "services_kept",
                "services": kept,
            })
        return bool(stopped or kept)
    except Exception as exc:
        log.debug("Failed to stop task services", exc_info=True)
        event = {
            "kind": "service_finalization_error",
            "services": [],
            "error": f"{type(exc).__name__}: {exc}",
        }
        signature = hashlib.sha256(json.dumps(
            event, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")).hexdigest()
        seen = getattr(ctx.tools._ctx, "_service_finalization_signatures", None)
        if not isinstance(seen, set):
            seen = set()
            ctx.tools._ctx._service_finalization_signatures = seen
        if signature in seen:
            return False
        seen.add(signature)
        ctx.llm_trace.setdefault("verification_events", []).append(event)
        return True


def _prepare_post_tool_budget_context(
    tools: ToolRegistry,
    limit_ctx: _RoundLimitContext,
    llm_trace: Dict[str, Any],
    active_model: str,
    active_use_local: bool,
    active_effort: str,
) -> None:
    """Refresh candidate evidence and the actual route before budget wrap-up."""

    candidate = getattr(tools._ctx, "_delivery_candidate", None)
    if isinstance(candidate, DeliveryCandidate):
        skill_action_pending = (
            candidate.finalization_control == "skill_action_or_revision_required"
        )
        evidence_revision, evidence_fingerprint = _delivery_evidence_state(
            tools, limit_ctx, llm_trace,
        )
        if (
            candidate.evidence_revision != evidence_revision
            or candidate.evidence_fingerprint != evidence_fingerprint
        ):
            _arm_delivery_control(
                tools,
                limit_ctx,
                llm_trace,
                control="effect_revision_required",
            )
        elif skill_action_pending:
            _arm_delivery_control(
                tools,
                limit_ctx,
                llm_trace,
                control="skill_revision_required",
            )
    # Cross-model fallback can adopt a different route during this round.
    limit_ctx.active_model = active_model
    limit_ctx.active_use_local = active_use_local
    limit_ctx.active_effort = active_effort


def _resolve_loop_max_rounds() -> int:
    from ouroboros.config import SETTINGS_DEFAULTS

    default = int(SETTINGS_DEFAULTS["OUROBOROS_MAX_ROUNDS"])
    try:
        return max(1, int(os.environ.get("OUROBOROS_MAX_ROUNDS", str(default))))
    except (ValueError, TypeError):
        log.warning("Invalid OUROBOROS_MAX_ROUNDS, defaulting to %s", default)
        return default


def run_llm_loop(
    messages: List[Dict[str, Any]],
    tools: ToolRegistry,
    llm: LLMClient,
    drive_logs: pathlib.Path,
    emit_progress: Callable[[str], None],
    incoming_messages: queue.Queue,
    task_type: str = "",
    task_id: str = "",
    budget_remaining_usd: Optional[float] = None,
    event_queue: Optional[queue.Queue] = None,
    initial_effort: str = "medium",
    drive_root: Optional[pathlib.Path] = None,
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """Run the tool loop."""
    ctx = tools._ctx
    ctx._delivery_candidate, ctx._delivery_candidate_revision = None, 0
    ctx._delivery_control_required = False
    ctx._delivery_evidence_revision, ctx._delivery_evidence_fingerprint = 0, ""
    _initialize_owner_directives(ctx, messages)
    task_model_override = str(getattr(ctx, "task_model_override", "") or "").strip()
    active_model = task_model_override or llm.default_model()
    active_effort = initial_effort
    if getattr(ctx, "task_use_local_override", None) is not None:
        active_use_local = bool(ctx.task_use_local_override)
    else:
        active_use_local = os.environ.get("USE_LOCAL_MAIN", "").lower() in ("true", "1")
    # Unknown routes get one honest call; no synthetic short-window capacity.
    _preferred_context_mode = get_context_mode()
    context_fit_plan = getattr(ctx, "context_fit_plan", None)
    if (context_fit_plan is not None
            and str(getattr(context_fit_plan, "preferred_mode", "")) == _preferred_context_mode):
        active_context_mode = str(getattr(context_fit_plan, "initial_mode", "") or _preferred_context_mode)
    else:
        active_context_mode = _preferred_context_mode
    llm_trace: Dict[str, Any] = {"reasoning_notes": [], "tool_calls": []}
    accumulated_usage: Dict[str, Any] = {"_task_attempt": getattr(ctx, "task_attempt", None)}
    tools._ctx._accumulated_usage = ctx._accumulated_usage = accumulated_usage
    max_retries = 3
    cost_ceiling = _resolve_task_cost_ceiling(ctx, budget_remaining_usd)
    if cost_ceiling.root_cap_usd is not None:
        # A resumed/late-started tree member must see tree spend before its
        # first pacing surface, not a process-local empty stash.
        _loop_tree_accounting(refresh=True, max_age_sec=0.0)
    from ouroboros.tools import tool_discovery as _td
    _td.set_registry(tools)

    tool_schemas = initial_tool_schemas(tools)
    tool_schemas, _enabled_extra_tools = _setup_dynamic_tools(tools, tool_schemas, messages)
    tools._ctx.event_queue = event_queue
    tools._ctx.task_id = task_id
    tools._ctx.messages = messages
    stateful_executor = StatefulToolExecutor()
    exit_ctx = _LoopExitContext(
        tools, drive_root, task_id, event_queue, drive_logs, accumulated_usage, llm_trace,
    )
    _owner_msg_seen: set = set()
    MAX_ROUNDS = _resolve_loop_max_rounds()
    MAX_ROUNDS = min(MAX_ROUNDS, int(getattr(ctx, "inline_max_rounds", MAX_ROUNDS)))
    round_idx = 0
    free_redial = False
    transport_wait = None
    limit_ctx: Optional[_RoundLimitContext] = None
    try:
        while True:
            if free_redial:
                # A transport-wait redial re-enters the SAME logical round.
                free_redial = False
            else:
                round_idx += 1

            ctx = tools._ctx
            _prev_active_route = (active_model, active_use_local)
            _prev_active_model = active_model
            active_model, active_use_local, active_effort, active_context_mode = _apply_overrides_and_regate_mode(
                ctx, active_model, active_use_local, active_effort, active_context_mode,
            )
            if (active_model, active_use_local) != _prev_active_route:
                context_fit_plan, active_context_mode = _rebind_context_fit_plan(
                    context_fit_plan, tools, messages, model=active_model,
                    use_local=active_use_local, preferred_mode=_preferred_context_mode,
                    tool_schemas=tool_schemas,
                )
            if active_model != _prev_active_model:
                # A cross-FAMILY switch_model / per-task override: strip the
                # prior family's provider-private reasoning blocks from the
                # history so the new family does not 400 on a signature it
                # cannot validate (safe — loses only reasoning continuity).
                # Same family is a no-op.
                _sanitized = LLMClient.sanitize_reasoning_on_model_switch(messages, _prev_active_model, active_model)
                if _sanitized is not messages:
                    messages[:] = _sanitized
            ctx.active_context_mode = active_context_mode
            ctx.active_model = active_model
            ctx.active_effort = active_effort
            ctx.active_use_local = active_use_local

            # One forced-wrap-up context per round: consumed by the round-limit
            # path and the supervisor finalize_now control path below.
            limit_ctx = _RoundLimitContext(
                messages, llm, active_model, active_effort, max_retries, drive_logs,
                task_id, round_idx, event_queue, accumulated_usage, task_type,
                active_use_local, MAX_ROUNDS, drive_root=drive_root, llm_trace=llm_trace,
                incoming_messages=incoming_messages, owner_msg_seen=_owner_msg_seen)
            _finalize_limit_ctx(limit_ctx, tools, llm_trace)
            if round_idx > MAX_ROUNDS:
                # Live hold: a paid [ROUND_LIMIT] dial would be a resend (no wake receipt) — no-call unknown terminal.
                if _delegate_hold_close(tools, drive_logs=drive_logs, task_id=task_id, detail="round_limit") == "active":
                    accumulated_usage["_last_llm_error_kind"] = "provider_outcome_unknown"
                    text, accumulated_usage, forced_trace = _handle_provider_unavailable(limit_ctx, error_kind="provider_outcome_unknown")
                else:
                    text, accumulated_usage, forced_trace = _handle_round_limit(limit_ctx)
                _merge_finalization_trace(llm_trace, forced_trace)
                return text, accumulated_usage, llm_trace

            # Tuple, not a sum: an APPENDED short owner message must also read as new input (final-pair fable F1).
            _pre_drain_sig = (len(messages), len(str(messages[-1].get("content") or "")))
            _controls = _drain_incoming_messages(
                messages, incoming_messages, drive_root, task_id, event_queue,
                _owner_msg_seen, owner_ctx=ctx)
            if _delegate_hold_step(
                    tools, controls=_controls, messages=messages, drive_logs=drive_logs,
                    task_id=task_id, emit_progress=emit_progress,
                    new_input=(len(messages), len(str(messages[-1].get("content") or ""))) != _pre_drain_sig) == "terminal":
                # The no-call decision reads the usage record, not the error_kind argument — stamp first.
                accumulated_usage["_last_llm_error_kind"] = "provider_outcome_unknown"
                text, accumulated_usage, forced_trace = _handle_provider_unavailable(
                    limit_ctx, error_kind="provider_outcome_unknown")
                _merge_finalization_trace(llm_trace, forced_trace)
                return text, accumulated_usage, llm_trace
            # Per-round early exit: supervisor finalize_now, else loop-local real-deadline finalize.
            _early_final = _maybe_early_finalize(
                limit_ctx, tools, _controls, transport_episode=transport_wait)
            if _early_final is not None:
                text, accumulated_usage, forced_trace = _early_final
                _merge_finalization_trace(llm_trace, forced_trace)
                return text, accumulated_usage, llm_trace

            # Typed soft landing (v6.91): the ledger fence stays the untouched
            # backstop; an exhausted ceiling wraps up BEFORE spending a round.
            _soft_land = _soft_land_exhausted_ceiling(limit_ctx, cost_ceiling)
            if _soft_land is not None:
                text, accumulated_usage, forced_trace = _soft_land
                _merge_finalization_trace(llm_trace, forced_trace)
                return text, accumulated_usage, llm_trace

            _checkpoint_injected = _inject_round_checkpoints(
                round_idx=round_idx, max_rounds=MAX_ROUNDS, messages=messages, accumulated_usage=accumulated_usage,
                emit_progress=emit_progress, tools=tools, event_queue=event_queue, task_id=task_id,
                drive_logs=drive_logs, budget_remaining_usd=budget_remaining_usd, cost_ceiling=cost_ceiling)

            messages, _compaction_usage = _run_round_compaction(
                messages,
                _CompactionRoundContext(
                    tools=tools, drive_root=drive_root, drive_logs=drive_logs,
                    task_id=task_id, round_idx=round_idx,
                    event_queue=event_queue, emit_progress=emit_progress))
            if tools._ctx.messages is not messages:
                tools._ctx.messages = messages
            limit_ctx.messages = messages  # WA2: provider-death finalize must salvage the COMPACTED transcript
            if _compaction_usage:
                _account_compaction_usage(accumulated_usage, _compaction_usage, event_queue, task_id)

            seal_task_transcript(messages)

            msg, cost, active_context_mode = _call_round_model(
                _RoundModelCallContext(
                    llm=llm,
                    messages=messages,
                    tools=tools,
                    context_fit_plan=context_fit_plan,
                    active_model=active_model,
                    tool_schemas=tool_schemas,
                    active_effort=active_effort,
                    max_retries=max_retries,
                    drive_logs=drive_logs,
                    task_id=task_id,
                    round_idx=round_idx,
                    event_queue=event_queue,
                    accumulated_usage=accumulated_usage,
                    task_type=task_type,
                    active_use_local=active_use_local,
                    active_context_mode=active_context_mode,
                    drive_root=drive_root,
                )
            )
            tools._ctx._current_llm_call_meta = dict(accumulated_usage.get("_last_llm_call_meta") or {})

            last_error_kind = str(accumulated_usage.get("_last_llm_error_kind") or "")
            transport_wait = _reconcile_transport_wait(
                transport_wait, ctx, msg_present=msg is not None, error_kind=last_error_kind,
                drive_logs=drive_logs, task_id=task_id, model=active_model, emit_progress=emit_progress)
            if msg is None and _fallback_chain_allowed(ctx, last_error_kind, transport_wait):
                _episode_before_chain = transport_wait is not None
                (
                    msg,
                    active_model,
                    active_use_local,
                    context_fit_plan,
                    active_context_mode,
                ) = _run_cross_model_fallback_chain(
                    llm=llm, ctx=ctx, tools=tools, messages=messages, active_model=active_model,
                    active_use_local=active_use_local, tool_schemas=tool_schemas, active_effort=active_effort,
                    max_retries=max_retries, drive_logs=drive_logs, task_id=task_id, round_idx=round_idx,
                    event_queue=event_queue, accumulated_usage=accumulated_usage, task_type=task_type,
                    emit_progress=emit_progress, context_fit_plan=context_fit_plan,
                    active_context_mode=active_context_mode)
                # Post-chain reconcile with the FRESH kind: a MID-chain outage
                # latches too (see reconcile_transport_wait's docstring).
                transport_wait = _reconcile_transport_wait(
                    transport_wait, ctx, msg_present=msg is not None,
                    error_kind=str(accumulated_usage.get("_last_llm_error_kind") or ""),
                    drive_logs=drive_logs, task_id=task_id, model=active_model,
                    emit_progress=emit_progress, after_local_pass=_episode_before_chain)
            if msg is None and transport_wait is not None and _transport_wait_step(
                transport_wait, tools=tools,
                error_kind=str(accumulated_usage.get("_last_llm_error_kind") or ""),
                drive_root=drive_root, drive_logs=drive_logs, task_id=task_id, model=active_model,
                emit_progress=emit_progress, incoming_messages=incoming_messages, owner_msg_seen=_owner_msg_seen):
                free_redial = True
                continue
            if msg is None and _delegate_hold_latch(
                    tools, error_kind=last_error_kind, drive_logs=drive_logs,
                    task_id=task_id, emit_progress=emit_progress):  # hold latched -> next round top parks
                continue
            if msg is None:
                # Exact actor routes skip generic substitution and fail as infrastructure.
                text, accumulated_usage, forced_trace = _handle_provider_unavailable(
                    limit_ctx,
                    error_kind=str(accumulated_usage.get("_last_llm_error_kind") or "provider_unavailable"),
                    wait_cause=transport_wait.wait_cause if transport_wait is not None else "",
                    waited=transport_wait is not None and (
                        transport_wait.wait_iterations > 0 or transport_wait.redials > 0),
                    wait_eligible=transport_wait.wait_eligible if transport_wait is not None else True)
                _merge_finalization_trace(llm_trace, forced_trace)
                return text, accumulated_usage, llm_trace

            from ouroboros.openai_chat_dispatch import CUSTOM_RECEIPTS_USAGE_KEY

            tool_calls = msg.get("tool_calls") or []
            tools._ctx._request_wire_custom_receipts = accumulated_usage.pop(
                CUSTOM_RECEIPTS_USAGE_KEY,
                (),
            )
            content = msg.get("content")
            _latch_final_answer_marker(llm_trace, content, current_tool_calls=tool_calls)
            # Every metered response counts as nanny progress.
            _note_nanny_delegate_activity(tools._ctx, round_idx, accumulated_usage, [])
            if not tool_calls:
                final_result = _no_tool_final_answer(
                    content, limit_ctx, llm_trace, tools, incoming_messages,
                    _owner_msg_seen, emit_progress,
                )
                if final_result is None:
                    continue
                return final_result

            if getattr(tools._ctx, "_skill_finalization_injected", False):
                tools._ctx._skill_finalization_injected = False
            assistant_msg = dict(msg)
            assistant_msg.setdefault("role", "assistant")
            messages.append(assistant_msg)

            _emit_round_progress(content, msg, emit_progress, llm_trace)

            handle_tool_calls(
                tool_calls, tools, drive_logs, task_id, stateful_executor,
                messages, llm_trace, emit_progress
            )

            # Nanny-economics baseline (poltergeist phase B): mark the
            # round's metered progress; re-baseline when it touched a
            # delegated run. Exact tool-call transitions — no log scans.
            _note_nanny_delegate_activity(
                tools._ctx, round_idx, accumulated_usage, tool_calls,
            )

            _prepare_post_tool_budget_context(
                tools, limit_ctx, llm_trace, active_model, active_use_local, active_effort,
            )
            budget_result = _check_budget_limits(
                limit_ctx,
                budget_remaining_usd,
                cost_ceiling=cost_ceiling,
            )
            if budget_result is not None:
                text, accumulated_usage, budget_trace = budget_result
                _merge_finalization_trace(llm_trace, budget_trace)
                return text, accumulated_usage, llm_trace

    except BudgetExceeded as exc:
        _delegate_hold_close(tools, drive_logs=drive_logs, task_id=task_id, detail="budget")
        return _handle_budget_exceeded(
            exc, exit_ctx, limit_ctx=limit_ctx, episode=transport_wait)
    finally:
        # No stale active latch behind an in-process exit (a crash skips this frame, keeping the latch for recovery).
        _delegate_hold_close(tools, drive_logs=drive_logs, task_id=task_id, detail="loop_exit")
        _cleanup_loop_resources(stateful_executor, exit_ctx)
