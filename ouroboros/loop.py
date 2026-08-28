"""LLM tool loop: call model, execute tools, repeat until final response."""

from __future__ import annotations

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
from ouroboros.config import adaptive_quorum, get_context_mode, get_light_model, get_review_enforcement, get_task_review_mode, resolve_effort
from ouroboros.outcomes import ACCEPTANCE_ACCEPTED, ACCEPTANCE_DECISION_STATUSES, ACCEPTANCE_FINALIZED_UNACCEPTED, ACCEPTANCE_REVISION_REQUESTED, REASON_ACCEPTANCE_REVIEW_SKIPPED_DEADLINE_RESERVE, extract_final_answer, latest_agent_defined_verification, latest_unreconciled_failed_verification, latest_unreconciled_masked_verification, reviewable_effect_projection, should_nudge_verification, turn_has_reviewable_effects
from ouroboros.observability import new_call_id, persist_call
from ouroboros.tool_policy import CAPABILITY_OMISSION_HEADER, format_capability_omissions, initial_tool_schemas, list_non_core_tools
from ouroboros.tools.registry import ToolRegistry
from ouroboros.context import build_user_content, estimate_context_prompt_tokens
from ouroboros.context_budget import EMERGENCY_COMPACTION_CHARS, LOW_EMERGENCY_COMPACTION_CHARS
from ouroboros.context_compaction import _tool_round_spans, compact_tool_history_llm
from ouroboros.deadline_utils import parse_deadline_ts, utc_now
from ouroboros.utils import estimate_tokens, truncate_review_artifact
from ouroboros.usage_accounting import BudgetExceeded

from ouroboros.loop_tool_execution import (
    StatefulToolExecutor,
    handle_tool_calls,
)
from ouroboros.loop_llm_call import call_llm_with_retry, emit_llm_usage_event
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

@dataclass
class _CompactionRoundContext:
    tools: ToolRegistry
    drive_root: Optional[pathlib.Path]
    drive_logs: pathlib.Path
    task_id: str
    round_idx: int
    event_queue: Optional[queue.Queue]
    active_use_local: bool
    active_context_mode: str
    checkpoint_injected: bool
    emit_progress: Callable[[str], None]
    active_model: str = ""


def _estimate_messages_chars(messages: List[Dict[str, Any]]) -> int:
    """Estimate transcript size over the FULL message list (the system block,
    when present in ``messages``, is counted too — conservative for the
    window-derived emergency trigger)."""
    from ouroboros.context_budget import IMAGE_BLOCK_CHAR_EQUIVALENT

    total = 0
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    if str(block.get("type") or "") in ("image_url", "image"):
                        # Vision tokens are billed per tile, not per base64
                        # char: counting the raw payload made ONE image look
                        # like ~300K tokens and permanently wedged emergency
                        # compaction.
                        total += IMAGE_BLOCK_CHAR_EQUIVALENT
                        continue
                    # Count whole multipart blocks, including cache markers.
                    try:
                        import json as _json2
                        total += len(_json2.dumps(block, ensure_ascii=False))
                    except (TypeError, ValueError):
                        total += len(str(block))
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            try:
                import json as _json
                total += len(_json.dumps(tool_calls, ensure_ascii=False))
            except (TypeError, ValueError):
                total += sum(len(str(tc)) for tc in tool_calls)
        tc_id = msg.get("tool_call_id")
        if tc_id:
            total += len(str(tc_id))
    return total


def _provider_failure_hint(accumulated_usage: Dict[str, Any]) -> str:
    detail = " ".join(str(accumulated_usage.get("_last_llm_error") or "").split()).strip()
    if not detail:
        return ""
    return f" Last provider error: {detail}"


def _provider_recovery_hint(accumulated_usage: Dict[str, Any]) -> str:
    """Explain whether retrying later is likely to help."""
    kind = str(accumulated_usage.get("_last_llm_error_kind") or "").strip()
    if kind == "provider_outcome_unknown":
        return (
            " The dispatched request has no terminal provider outcome. No further "
            "replay, same-request retry, or paid fallback was sent after that "
            "unknown outcome because doing so could duplicate live work."
        )
    if accumulated_usage.get("context_overflow_suggest_low"):
        return (
            " ⚠️ The context overflowed the model window. Switching to low context "
            "mode (Settings → Behavior, or the chat toggle) fits ~200K / local "
            "models by serving ARCHITECTURE as a navigation map and compacting "
            "memory sooner — without changing the model or reasoning effort."
        )
    if kind in {"quota_exhausted", "auth_error", "request_too_large", "bad_request", "context_overflow"}:
        guidance = {
            "quota_exhausted": "The provider rejected the request for quota/billing reasons; retrying the same request will not help until the key/account limit changes.",
            "auth_error": "The provider rejected authentication/authorization; retrying the same request will not help until the configured key or provider access is fixed.",
            "request_too_large": "The provider rejected the request size/output-token shape; retrying the same request will not help without reducing context/output demand or changing model capacity.",
            "bad_request": "The provider rejected the request shape; retrying the same request will not help until the transcript/tool payload is fixed.",
            "context_overflow": "The context overflowed the model window; retrying the same request will not help without reducing context or changing model capacity.",
        }.get(kind, "Retrying the same provider request will not help until the underlying request/account issue changes.")
        return f" {guidance}"
    detail = str(accumulated_usage.get("_last_llm_error") or "").lower()
    if "prefill" in detail or "conversation must end with a user message" in detail:
        return (
            " This looks like a client-side transcript-shape error, not a "
            "provider outage; retrying the same input will not help."
        )
    if "provider returned incomplete response" in detail or "finish_reason=null" in detail:
        return (
            " The provider returned incomplete responses repeatedly; this may "
            "be transient, but it can also indicate malformed client input."
        )
    return " If background consciousness is running, it will retry when the provider recovers."


def _record_provider_recovery_action(
    accumulated_usage: Dict[str, Any],
    *,
    kind: str,
    count: int,
    model: str,
) -> None:
    """Project recovery actions backed by this round's physical-attempt events."""

    count = max(0, int(count or 0))
    if not count:
        return
    actions = accumulated_usage.setdefault("_provider_recovery_actions", [])
    if not isinstance(actions, list):
        actions = []
        accumulated_usage["_provider_recovery_actions"] = actions
    actions.append({"kind": str(kind), "count": count, "model": str(model or "")})


def _record_last_call_same_model_retries(
    accumulated_usage: Dict[str, Any], model: str,
) -> int:
    dispatched = int(accumulated_usage.get("_llm_dispatched_attempts") or 0)
    if dispatched > 1:
        _record_provider_recovery_action(
            accumulated_usage,
            kind="same_model_retry",
            count=dispatched - 1,
            model=model,
        )
    return dispatched


def _provider_recovery_action_hint(accumulated_usage: Dict[str, Any]) -> str:
    """Describe only recovery actions backed by typed physical-attempt facts."""

    actions = accumulated_usage.get("_provider_recovery_actions")
    if not isinstance(actions, list):
        return ""
    retries = sum(
        max(0, int(action.get("count") or 0))
        for action in actions
        if isinstance(action, dict) and action.get("kind") == "same_model_retry"
    )
    fallbacks = sum(
        max(0, int(action.get("count") or 0))
        for action in actions
        if isinstance(action, dict) and action.get("kind") == "cross_model_fallback"
    )
    facts: list[str] = []
    if retries:
        facts.append(f"{retries} same-model retry attempt(s)")
    if fallbacks:
        facts.append(f"{fallbacks} configured fallback route attempt(s)")
    if not facts:
        return ""
    return (
        " Recorded physical-attempt evidence confirms these recovery actions "
        "in the same episode: " + "; ".join(facts) + "."
    )


def _provider_unavailable_notice(accumulated_usage: Dict[str, Any]) -> str:
    """Host-authored provider terminal notice with evidence-bound claims."""

    kind = str(accumulated_usage.get("_last_llm_error_kind") or "").strip()
    if kind == "provider_outcome_unknown":
        lead = (
            "⚠️ SYSTEM NOTICE (host): A model request may have reached the "
            "provider, but no terminal provider outcome was recorded."
        )
    else:
        lead = "⚠️ SYSTEM NOTICE (host): The model provider returned no usable response."
    return (
        lead
        + _provider_failure_hint(accumulated_usage)
        + _provider_recovery_action_hint(accumulated_usage)
        + _provider_recovery_hint(accumulated_usage)
        + " Any files written so far are preserved in the workspace."
    )


def _handle_text_response(
    content: Optional[str],
    llm_trace: Dict[str, Any],
    accumulated_usage: Dict[str, Any],
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """Handle LLM response without tool calls (final response)."""
    if content and content.strip():
        llm_trace["reasoning_notes"].append(content.strip())
    return (content or ""), accumulated_usage, llm_trace


def _skill_names_touched_by_trace(llm_trace: Dict[str, Any]) -> List[str]:
    names: List[str] = []
    for call in llm_trace.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        tool = str(call.get("tool") or "")
        if tool not in {"write_file", "edit_text", "claude_code_edit"}:
            continue
        args = call.get("args") if isinstance(call.get("args"), dict) else {}
        bucket = str(args.get("bucket") or "").strip().lower()
        skill_name = str(args.get("skill_name") or "").strip()
        if bucket in {"external", "clawhub", "ouroboroshub"} and skill_name:
            if skill_name not in names:
                names.append(skill_name)
            continue
        candidates = [str(args.get("cwd") or "")] if tool == "claude_code_edit" else [str(args.get("path") or "")]
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


def _force_plan_completed(llm_trace: Dict[str, Any]) -> bool:
    """True when the latest plan_task control closes the force-plan gate.

    Reads the exact typed outcome/control captured from the FULL tool result at
    execution time.  GREEN closes directly; REVIEW_REQUIRED closes only after
    an exact-fingerprint disposition.  REVISE_PLAN and malformed/open controls
    keep the force-plan gate closed.
    """
    for call in reversed(llm_trace.get("tool_calls") or []):
        if not isinstance(call, dict):
            continue
        if str(call.get("tool") or "") != "plan_task":
            continue
        if bool(call.get("is_error")):
            return False
        return (
            call.get("plan_review_closed") is True
            and str(call.get("plan_review_outcome") or "")
            in {"GREEN", "REVIEW_REQUIRED"}
        )
    return False


def _force_plan_required(ctx: Any, llm_trace: Dict[str, Any]) -> bool:
    metadata = getattr(ctx, "task_metadata", {}) if isinstance(getattr(ctx, "task_metadata", {}), dict) else {}
    return bool(metadata.get("force_plan")) and not _force_plan_completed(llm_trace)


def _check_budget_limits(
    ctx: "_RoundLimitContext",
    budget_remaining_usd: Optional[float],
    cost_ceiling_usd: Optional[float] = None,
) -> Optional[Tuple[str, Dict[str, Any], Dict[str, Any]]]:
    """Return a final-response tuple when budget limits require stopping.

    ``cost_ceiling_usd`` is the in-task hard-stop resolved ONCE at loop start
    from ``task_contract.budget_profile.cost_hard_stop_pct``
    (``task_pacing.resolve_cost_ceiling_usd``); None means no in-task cost stop
    — the global budget-exhaustion gate below still applies."""
    if budget_remaining_usd is None:
        return None

    accumulated_usage = ctx.accumulated_usage
    messages = ctx.messages
    raw_task_cost = accumulated_usage.get("cost")
    task_cost = float(raw_task_cost) if raw_task_cost is not None else None

    if budget_remaining_usd <= 0:
        finish_reason = "🚫 Task rejected. Total budget exhausted. Please increase TOTAL_BUDGET in settings."
        accumulated_usage["execution_status"] = "failed"
        accumulated_usage["reason_code"] = "budget_exhausted"
        if ctx.round_idx <= 1:
            trace = ctx.llm_trace if isinstance(ctx.llm_trace, dict) else {}
            return finish_reason, accumulated_usage, trace
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

    from ouroboros.config import SETTINGS_DEFAULTS as _DEFAULTS
    _per_task_default = str(_DEFAULTS["OUROBOROS_PER_TASK_COST_USD"])
    per_task_limit = float(os.environ.get("OUROBOROS_PER_TASK_COST_USD", _per_task_default) or _per_task_default)
    if task_cost is not None and task_cost >= per_task_limit and ctx.round_idx % 10 == 0:
        _append_or_merge_user_message(
            messages,
            f"[COST NOTE] Task spent ${task_cost:.3f}, which is at or above the per-task soft threshold of ${per_task_limit:.2f}. Continue only if the expected value still justifies the cost.",
        )

    if cost_ceiling_usd is not None and task_cost is not None and task_cost > cost_ceiling_usd:
        finish_reason = (
            f"Task spent ${task_cost:.3f} (over the in-task cost ceiling ${cost_ceiling_usd:.2f} "
            f"of remaining ${budget_remaining_usd:.2f}). Budget exhausted."
        )
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


def _resolve_task_cost_ceiling(ctx: Any, budget_remaining_usd: Optional[float]) -> Optional[float]:
    """The in-task cost hard-stop, resolved ONCE at loop start from the start-of-
    task budget snapshot + task_contract.budget_profile (cost_hard_stop_pct
    None -> the historical 50%-of-remaining stop, 0 -> no in-task stop)."""
    return task_pacing.resolve_cost_ceiling_usd(
        budget_remaining_usd, task_pacing.resolve_budget_profile(ctx),
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


def _persist_compaction_checkpoint(
    messages: List[Dict[str, Any]],
    *,
    drive_root: Optional[pathlib.Path],
    drive_logs: pathlib.Path,
    task_id: str,
    reason: str,
    keep_recent: int,
    round_idx: int,
    event_queue: Optional[queue.Queue],
    checkpoint_kind: str = "pre_compaction_transcript",
    call_type: str = "compaction_checkpoint",
) -> bool:
    """Persist the canonical transcript before a deterministic context rebuild."""
    root = pathlib.Path(drive_root) if drive_root is not None else pathlib.Path(drive_logs).parent
    call_id = new_call_id("compaction_checkpoint")
    try:
        ref = persist_call(
            root,
            task_id=task_id,
            call_id=call_id,
            call_type=call_type,
            payload={
                "reason": reason,
                "keep_recent": keep_recent,
                "round": round_idx,
                "messages": messages,
            },
            manifest={
                "round": round_idx,
                "reason": reason,
                "keep_recent": keep_recent,
            },
        )
        _emit_checkpoint_event(event_queue, task_id, drive_logs, {
            "checkpoint_kind": checkpoint_kind,
            "round": round_idx,
            "reason": reason,
            "keep_recent": keep_recent,
            "checkpoint_ref": ref.get("manifest_ref"),
        })
        return True
    except Exception:
        log.debug("Failed to persist pre-compaction transcript checkpoint", exc_info=True)
        _emit_checkpoint_event(event_queue, task_id, drive_logs, {
            "checkpoint_kind": checkpoint_kind,
            "round": round_idx,
            "reason": reason,
            "keep_recent": keep_recent,
            "checkpoint_status": "failed",
        })
        return False


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
    placeholder carrying the caption and the re-view path, so the dialogue
    HORIZON is preserved while the heavy payload is dropped (P1: granularity
    varies, history does not silently vanish). ``incoming`` reserves room for
    blocks about to be appended.
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

    ``auto`` and ``required`` are effect-gated: the host enforces review when the
    turn produced reviewable effects (commit / deliverable / repo / workspace /
    skill write), declared a typed deliverable/criterion, or is not a direct-chat
    turn (queued / headless / scheduled). Read-only research and ordinary tool use
    in direct conversation do not by themselves justify a three-reviewer panel.
    Ephemeral routing turns are presentation/control decisions, not deliverables.
    ``off`` never reviews.
    This gates on typed contracts and observable runtime facts (P3 immune gate),
    not on message content (no P5 violation).
    """
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
            ctx._task_acceptance_owner_generation = int(
                getattr(admission_agent, "_owner_message_generation", 0) or 0
            )
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
        from ouroboros.tools.join_ledger import _child_result_sha256

        meta = getattr(ctx, "task_metadata", {})
        meta = meta if isinstance(meta, dict) else {}
        root_id = str(meta.get("root_task_id") or getattr(ctx, "root_task_id", "") or task_id)
        status_root = pathlib.Path(str(
            meta.get("budget_drive_root")
            or getattr(ctx, "budget_drive_root", "")
            or drive_root
        ))
        audit_only_task_ids: set[str] = set()
        try:
            from ouroboros.task_results import (
                load_plan_review_state,
                plan_review_audit_only_task_ids,
            )

            audit_only_task_ids = set(plan_review_audit_only_task_ids(
                load_plan_review_state(status_root, str(task_id))
            ))
        except Exception:
            # Missing/corrupt planning evidence must never hide a live child.
            log.debug(
                "Unable to load audit-only planning scouts for acceptance",
                exc_info=True,
            )
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
            if row_task_id in audit_only_task_ids:
                continue
            status = str(row.get("status") or "unknown")
            projected = {
                "task_id": row_task_id,
                "parent_task_id": str(row.get("parent_task_id") or ""),
                "status": status,
                "artifact_status": str(row.get("artifact_status") or ""),
            }
            if status in SETTLED_STATUSES:
                projected["child_result_sha256"] = _child_result_sha256(row)
            compact.append(projected)
        # Acceptance needs true quiescence. ``cancel_requested`` is terminal for
        # parent handoff reminders but the worker may still be exiting, so it is
        # deliberately excluded here via SETTLED_STATUSES.
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
            and str(row.get("task_id") or "") not in audit_only_task_ids
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
    # Opt-in CANDIDATES latch (v6.54.4): when the model enumerates candidate
    # interpretations/answers with an explicit block ("CANDIDATES:" on its own
    # line, one "- " item per line), latch them alongside the final answer so the
    # acceptance reviewer can adjudicate ambiguity. Marker-only, like FINAL
    # ANSWER — never prose mining; absent block leaves behavior unchanged.
    text = content or ""
    try:
        lines = text.splitlines()
        marker_idx = next(
            (i for i, line in enumerate(lines) if line.strip() == "CANDIDATES:"),
            None,
        )
        if marker_idx is not None:
            # Marker-only, like FINAL ANSWER (adversarial review r2 #4): the block
            # is the "- " items IMMEDIATELY following the marker line; the first
            # non-item line ends it. No substring-anywhere trigger, no harvesting
            # of a distant bullet list after intervening prose.
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
# Closed set of typed acceptance reasons (v6.78.0). Every value is either a fact the
# host already computed for another purpose or the name of the exit branch; nothing
# here is derived from model prose. `unspecified` is only the fail-closed fallback of
# `_set_acceptance_decision` for a future writer that forgets its reason.
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
    REASON_ACCEPTANCE_REVIEW_SKIPPED_DEADLINE_RESERVE,
    ACCEPTANCE_REASON_UNSPECIFIED,
)


def _set_acceptance_decision(llm_trace: Dict[str, Any], decision: Dict[str, Any]) -> None:
    """The ONLY merge point for the host acceptance decision (v6.78.0, owner Q23).

    Every host exit funnels here and can only leave in one of the three canonical
    owner-facing states (``ACCEPTANCE_DECISION_STATUSES``) plus a typed ``reason``
    naming WHICH exit it was. A status outside the trio fails closed to
    ``finalized_unaccepted`` and its raw token survives as the ``reason``, so a
    future writer can neither invent a fourth state nor lose its own token. The
    agent's stance (``agent_disposition``/``agent_rationale``) is carried forward,
    never overwritten — after P4.1 the agent no longer writes a status at all.
    """
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

    Active only on the required+blocking path. Each critical finding WITH a
    concrete recommendation becomes one open obligation in llm_trace (never the
    durable commit review_state — that ledger stays a separate SSOT). Clean
    finalization asks for an agent disposition per obligation via the existing
    v6.54.0 agent_disposition mechanism; time/pass gates and every forced-
    finalization escape hatch bound the loop, so a deadline never hangs here.

    v6.60.0 widening (S1-lite, owner quiz 18b): when the AGGREGATE verdict itself
    is failing — signal FAIL, or worst outcome tier blocked_with_evidence — the
    contributing reviewers' HIGH-severity findings with a concrete recommendation
    also become obligations (the PB incident: reviewers converged on a concrete
    "the deliverable misses X" at high severity, the task finalized clean anyway).
    On a PASS (including PASS-with-dissent) the bar stays critical-only, so the
    blocking lane cannot creep into taxing every clean run with hygiene items."""
    import hashlib

    from ouroboros.review_substrate import _contributing_actors, aggregate_outcome_tier

    contributing = {str(a.get("slot_id", "")) for a in _contributing_actors(result)}
    obligations = llm_trace.setdefault("acceptance_obligations", [])
    by_id = {str(o.get("id")): o for o in obligations if isinstance(o, dict)}
    # No contributing actors (all parse-degraded / no quorum) => no authoritative
    # verdict, so manufacture NO blocking obligations — otherwise a single
    # parse-degraded slot's critical finding would gate finalization, the same
    # class the improvement capsule already refuses to let a degraded slot inject
    # (adversarial review r1). A blocking obligation must ride a CONTRIBUTING slot.
    if not contributing:
        return
    _agg_failing = (
        str(getattr(result, "aggregate_signal", "") or "").upper() == "FAIL"
        or aggregate_outcome_tier(result) == "blocked_with_evidence"
    )
    _obligation_severities = {"critical", "high"} if _agg_failing else {"critical"}
    # Ids already created or reopened by THIS panel pass. Multiple slots of one
    # panel routinely raise the same finding (typed re_raise copies the exact
    # catalog id); without this, the second slot's duplicate would falsely
    # increment reopened_count on the very pass that first presented the
    # finding and overwrite reviewer_rebuttal_response (fable review r1 #1).
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
        # v6.74.0 (A3): obligation identity is reviewer-authored. A finding with
        # disposition_kind="re_raise" MUST name an existing catalog id; the host
        # only validates it exists. A re_raise with a missing/unknown id fails
        # closed to `new` with a disclosed note, so a reworded re-raise can no
        # longer silently mint a fresh hash id.
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
            # Reviewer-authored identity (commit triad r2, sol): only an UNTYPED
            # legacy finding may reopen via byte-identical text (v6.71.1
            # compat). A typed "new" or an unbound "re_raise" whose text happens
            # to match a settled row must NOT resurrect the agent's settled
            # rebuttal — the sloppy signal is DISCLOSED on the row instead.
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
    # An agent-filed disposition (status="agent_disposed") is a CLAIM/rebuttal, not
    # a settlement: the row stays pending until a host panel adjudicates it (clean
    # PASS settles it; a re-raise reopens it). Predicate SSOT: review_evidence.
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
            # The clean panel ACCEPTED the agent's filed disposition (a rebuttal
            # it chose not to re-raise): keep the agent's disposition/reason as
            # provenance and record the host settlement distinctly (final review
            # r6) — never rewrite a rejected rebuttal into "addressed by revision".
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
    # v6.74.0 (A4): disagreement is recorded ONLY via obligation_dispositions —
    # the old "or address them directly" prose read as a third channel; fixing
    # the work is described honestly (it helps by making the next panel clean).
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


# The host-forced acceptance-review checklist (module constant so the review
# function stays within the size gate). v6.60.0 adds the explicit SCOPE-CUT
# question — a silent/unjustified narrowing is a high-severity finding, which
# under blocking enforcement becomes a typed obligation.
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
    # One pre-rendered rails line (money/time/rounds/passes headroom) assembled
    # in loop.py from each real source and fed into the improvement capsule
    # (v6.74.0 A1, owner Q6); the capsule builder never gains a ctx parameter.
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
    return build_task_acceptance_evidence(
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


def _execute_task_acceptance_panel(ctx: _TaskAcceptanceContext) -> Any:
    """Perform the one substantive host panel over the pre-bound evidence."""
    from ouroboros.review_substrate import (
        HARDNESS_ADVISORY_VISIBLE,
        ReviewRequest,
        ReviewRunResult,
        reviewer_slots,
        run_review_request,
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
            "require_criterion_evidence": True,
            "max_physical_attempts_per_actor": 2,
        },
        task_id=ctx.task_id,
    )
    # Budget admission for the whole acceptance wave (v6.69.0): a wave that
    # cannot fit the remaining root budget is declined up front as a terminal
    # DEGRADED (no-quorum semantics) instead of dying mid-wave with paid
    # partial slots and a wedged pending review. The estimate renders the REAL
    # per-slot message pair (a compact evidence+checklist guess measured ~2.1x
    # short of the dispatched prompt); the rare second physical attempt per
    # actor is deliberately not multiplied in — this gate is a fail-open
    # coarse filter for waves that cannot fit, not a hard reservation.
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
    started = time.monotonic()
    result = run_review_request(
        request,
        slots=slots,
        drive_root=(
            pathlib.Path(ctx.drive_root)
            if ctx.drive_root is not None
            else pathlib.Path(ctx.tools._ctx.drive_root)
        ),
        usage_ctx=ctx.tools._ctx,
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
    # A REUSED panel (unchanged binding) is the SAME reviewer act applied again:
    # re-collecting would mutate reviewer-authored state with no new reviewer
    # input — reopened_count 0→1 mislabels a first presentation as re-raised,
    # and the changed catalog row shifts the evidence revision, buying a fresh
    # paid panel for a byte-identical resubmit (fable review r2 #1). The rows
    # were already collected when this exact panel first applied.
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
    # v6.74.0 (A5): the reviewers' typed dialogue judgement, reduced over ALL
    # contract-valid actors with the panel's own quorum; persisted for audit on
    # the authoritative run record regardless of which branch applies below.
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
    )
    if dialogue_terminal:
        # v6.74.0 (A5): a reviewer quorum judged the dialogue no longer
        # actionable (unreachable_here / stable_disagreement). Finalize through
        # the EXISTING honest path, recording BOTH positions — the reviewers'
        # findings stay in the run record, the agent's dispositions stay on the
        # obligation rows — with one owner-visible line. This is reviewer
        # authorship, not a host timer and not a unilateral agent give-up.
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
        # Per-slot causes were always recorded in the structured decision; the
        # owner-visible line used to say only "no valid quorum", forcing a dig
        # through task_results to learn WHICH slot failed and why (v6.70.0).
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
            "reason": "open_obligations",
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
            # Round-9 CRITICAL 1: this is the fall-through AFTER
            # `task_acceptance_is_clean(result)` already refused the panel, so it
            # cannot mint `accepted` — docs/ARCHITECTURE.md reserves that state for
            # clean acceptance ("quorum PASS, `solved`, and supported evidence for
            # every contributing criterion"). The reachable case is a reviewer
            # claiming `solved` while supplying a MISSING criterion with the
            # improvement-pass cap already spent: there is nothing actionable left
            # to feed back, but "nothing left to ask for" is not "accepted". The
            # typed reason is unchanged (`no_actionable_changes` still names WHY the
            # loop stops here) and the tier honesty keeps riding `outcome_tier`;
            # only the owner-facing state is now the honest one.
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


def _build_acceptance_rails_line(
    budget_snapshot: Any,
    budget_profile: Dict[str, Any],
    passes_done: int,
    loop_rails: Optional[Dict[str, Any]],
    *,
    required_blocking: bool,
    workspace: bool = False,
) -> str:
    try:
        return _build_acceptance_rails_line_inner(
            budget_snapshot, budget_profile, passes_done, loop_rails,
            required_blocking=required_blocking, workspace=workspace,
        )
    except Exception:
        # The rails line is advisory context; it must never take down the
        # acceptance path (fable review r2 #3 — make the docstring true).
        log.debug("acceptance rails line failed soft", exc_info=True)
        return ""


def _build_acceptance_rails_line_inner(
    budget_snapshot: Any,
    budget_profile: Dict[str, Any],
    passes_done: int,
    loop_rails: Optional[Dict[str, Any]],
    *,
    required_blocking: bool,
    workspace: bool = False,
) -> str:
    """One line naming every active termination source with its remaining
    headroom (v6.74.0 A1, owner Q6): money, time, rounds, review passes. Each
    rail comes from its real source — the usage ledger projection, the
    BudgetSnapshot, the loop's round counter, and the pacing pass cap — and an
    unavailable rail is omitted rather than guessed. For workspace deliveries
    the line also carries the tree directive (v6.74.4) — a delivery-state
    instruction, not a termination source. Fail-soft: never raises."""
    parts: List[str] = []
    rails = loop_rails if isinstance(loop_rails, dict) else {}
    try:
        money_bits: List[str] = []
        cost = rails.get("task_cost_usd")
        if cost is not None:
            money_bits.append(f"${float(cost):.2f} spent this task")
        try:
            from ouroboros.usage_accounting import current_usage_scope, usage_projection

            scope = current_usage_scope()
            if scope is not None and scope.root_task_id:
                projection = usage_projection(
                    scope.drive_root, root_task_id=scope.root_task_id,
                )
                remaining = projection.get("remaining_known_usd")
                if remaining is not None:
                    money_bits.append(f"${float(remaining):.2f} budget left")
        except Exception:
            log.debug("rails: budget projection unavailable", exc_info=True)
        if money_bits:
            parts.append("money: " + ", ".join(money_bits))
    except (TypeError, ValueError):
        pass
    try:
        if getattr(budget_snapshot, "has_deadline", False):
            parts.append(
                f"time: {max(0.0, budget_snapshot.remaining_sec) / 60:.0f} min left "
                f"({budget_snapshot.reserve_sec / 60:.0f} min finalization reserve)"
            )
    except (TypeError, ValueError):
        pass
    try:
        round_idx = rails.get("round_idx")
        max_rounds = rails.get("max_rounds")
        if round_idx is not None and max_rounds:
            parts.append(f"rounds: {int(round_idx)}/{int(max_rounds)}")
    except (TypeError, ValueError):
        pass
    try:
        cap = task_pacing.effective_max_improvement_passes(
            budget_profile,
            has_deadline=bool(getattr(budget_snapshot, "has_deadline", False)),
            required_blocking=required_blocking,
        )
        if cap is None:
            parts.append(
                f"review passes: {int(passes_done)} done, no local count cap "
                "(deadline/budget rails bind)"
            )
        else:
            passes_part = f"review passes: {int(passes_done)}/{int(cap)}"
            # v6.74.4 freeze directive (count axis): the pass launched at
            # cap-1 is the last one improvement_pass_allowed will admit, so
            # say so. cap==0 never feeds a capsule back; skip the clause, and
            # passes_done >= cap (supersede-reset re-review) is not a launch.
            if 0 <= int(passes_done) < int(cap) and int(passes_done) + 1 >= int(cap):
                passes_part += " — FINAL improvement pass, no further passes will run"
            parts.append(passes_part)
    except (TypeError, ValueError):
        pass
    try:
        # v6.74.4: EVERY workspace improvement capsule carries the tree
        # directive, not just the provably-final one — a deadline/cost rail
        # can end the loop between capsules (commit triad r1, sol), and the
        # tree ships as-is on any forced end.
        if workspace:
            parts.append(
                "workspace delivery: the deliverable is your working tree as "
                "it stands when the task ends — keep it in a VERIFIED state "
                "(rebuild, verify, and commit if the task calls for a commit) "
                "and revert unverified edits rather than shipping them"
            )
    except (TypeError, ValueError):
        pass
    return "; ".join(parts)


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
    fence_ok, _fence_token = _begin_task_acceptance_fence(tools._ctx, task_id)
    if not fence_ok:
        llm_trace["review_decision"] = {
            "eligibility": "acceptance_fence_failed", "trigger": trigger,
        }
        _append_or_merge_user_message(
            messages,
            "[TASK ACCEPTANCE WAIT] The supervisor could not atomically close "
            "subtask admission. Do not finalize or spawn more work; retry after the "
            "queue fence is available.",
        )
        emit_progress("Task acceptance review waiting for the queue-owned admission fence.")
        return True
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
    budget_profile = task_pacing.resolve_budget_profile(tools._ctx)
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
        rails_line=_build_acceptance_rails_line(
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
        fence_state: Any = _fence_token
        if fence_state is None:
            fence_state = {
                "state": "direct_context",
                "owner_generation": getattr(
                    tools._ctx, "_task_acceptance_owner_generation", None,
                ),
                "queue_generation": getattr(
                    tools._ctx, "_task_acceptance_fence_generation", None,
                ),
            }
        review_ctx.review_binding = build_review_binding(
            candidate=content,
            evidence=review_ctx.evidence,
            fence_token_or_state=fence_state,
        )
        binding_hash = str(review_ctx.review_binding.get("binding_hash") or "")
        seen_bindings = getattr(tools._ctx, "_task_acceptance_seen_bindings", None)
        if not isinstance(seen_bindings, dict):
            seen_bindings = {}
            tools._ctx._task_acceptance_seen_bindings = seen_bindings
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
            # A process-local attempt without its authoritative trace is not safe
            # to repeat or silently accept.  The ordinary infra-degraded path below
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
    """Round-4 C1.1: adopt a SUCCESSFUL cross-family fallback as the active route for the
    rest of the loop. Otherwise a subsequent round (esp. a tool loop) replays THIS
    fallback's reasoning/thinking back to the original primary family with no
    model-switch sanitizer firing (active_model never changed) — the cross-family
    signature replay, in reverse. Adopting the sanitized transcript as canonical means
    the switched route never carries the old family's provider-private blocks (a later
    switch_model/override re-triggers the round-start sanitizer normally). The caller
    has already rebound the shared context-fit plan to this exact route before its
    physical dispatch; adoption makes that tested projection canonical. Returns the new
    ``(active_model, active_use_local, context_fit_plan, context_mode)``."""
    ctx.active_model = fallback_model
    messages[:] = fallback_messages
    if context_fit_plan is not None:
        tools._ctx.context_fit_plan = context_fit_plan
        tools._ctx.messages = messages
        tools._ctx.active_context_mode = active_context_mode
        accumulated_usage["_context_route_fp"] = str(
            getattr(context_fit_plan, "route_fp", "") or ""
        )
        accumulated_usage["_context_prompt_estimate"] = estimate_context_prompt_tokens(
            messages, tool_schemas,
        )
        accumulated_usage["_context_fit_mode"] = active_context_mode
    return fallback_model, fallback_use_local, context_fit_plan, active_context_mode


def _run_cross_model_fallback_chain(
    *, llm, ctx, tools, messages, active_model, active_use_local, tool_schemas,
    active_effort, max_retries, drive_logs, task_id, round_idx, event_queue,
    accumulated_usage, task_type, emit_progress, context_fit_plan,
    active_context_mode,
) -> tuple:
    """F1 (v6.39): 429-aware cross-model fallback CHAIN. Mark the failed primary on
    cooldown if its last failure was transient (so a swarm stops stampeding it), then walk
    the configured fallback chain, skipping cooled-down models, until one responds. Each
    candidate gets a small per-candidate attempt cap so a multi-model chain cannot multiply
    into a long retry storm; every call stays deadline-aware. The bench (FALLBACKS==main)
    dedupes to an empty chain -> no cross-model fallback, by design. Returns the new
    ``(msg, active_model, active_use_local, context_fit_plan, context_mode)``;
    ``msg`` is None if the whole (cooled-down / empty) chain is exhausted,
    leaving the caller to join the provider-unavailable shelf."""
    from ouroboros import fallback_cooldown as _fcd
    from ouroboros.config import get_fallback_models
    from ouroboros.loop_llm_call import _COOLDOWN_ERROR_KINDS as _cooldown_kinds

    if str(accumulated_usage.get("_last_llm_error_kind") or "") == "provider_outcome_unknown":
        return (
            None,
            active_model,
            active_use_local,
            context_fit_plan,
            active_context_mode,
        )

    def _cooled(model: str, use_local: bool) -> None:
        if str(accumulated_usage.get("_last_llm_error_kind") or "") in _cooldown_kinds:
            _fcd.mark_cooldown(model, use_local)

    _cooled(active_model, active_use_local)
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
        # Cross-FAMILY fallback must not replay the primary's provider-private reasoning to
        # a different family (the GLM->Claude 400 "Invalid signature" death); the SSOT
        # sanitizer is a no-op same-family.
        fallback_messages = LLMClient.sanitize_reasoning_on_model_switch(messages, active_model, fallback_model)
        # Bind exact route evidence and choose its deterministic projection BEFORE
        # physical dispatch.  This prevents the fallback's first request from
        # inheriting the failed primary route's Max projection/fingerprint.  It
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
        dispatched_attempts = _record_last_call_same_model_retries(
            accumulated_usage, fallback_model,
        )
        if dispatched_attempts:
            _record_provider_recovery_action(
                accumulated_usage,
                kind="cross_model_fallback",
                count=1,
                model=fallback_model,
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
        # Candidate evidence was real for its dispatched attempts, but an
        # unaccepted route must not become the task's canonical plan/transcript.
        tools._ctx.context_fit_plan = context_fit_plan
        tools._ctx.messages = messages
        tools._ctx.active_context_mode = active_context_mode
        if str(accumulated_usage.get("_last_llm_error_kind") or "") == "provider_outcome_unknown":
            break
        _cooled(fallback_model, fallback_use_local)
    if msg is None and context_fit_plan is not None:
        accumulated_usage["_context_route_fp"] = str(
            getattr(context_fit_plan, "route_fp", "") or ""
        )
        accumulated_usage["_context_prompt_estimate"] = estimate_context_prompt_tokens(
            messages, tool_schemas,
        )
        accumulated_usage["_context_fit_mode"] = active_context_mode
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
    """Read direct children while excluding planning scouts retained for audit only."""

    from ouroboros.task_status import find_child_tasks

    children = [
        row for row in find_child_tasks(
            pathlib.Path(status_root),
            parent_task_id=task_id,
            root_task_id=root_task_id,
            exclude_task_id=task_id,
            scope="direct",
        )
        if isinstance(row, dict)
    ]
    try:
        from ouroboros.task_results import (
            load_plan_review_state,
            plan_review_audit_only_task_ids,
        )

        audit_only = set(plan_review_audit_only_task_ids(
            load_plan_review_state(pathlib.Path(status_root), task_id)
        ))
    except Exception:
        # Corrupt/unavailable planning evidence must not hide generic children.
        audit_only = set()
    return [
        row for row in children
        if str(row.get("task_id") or row.get("id") or "") not in audit_only
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
        # Exact-hash dispositions suppress the unchanged result only. If status,
        # result, trace, or artifact identity changes, the disposition becomes stale
        # and this reminder automatically re-opens without parsing prose.
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
        # P5: the reminder is suppressed ONLY by structured signals — a child explicitly
        # discarded/cancelled (already filtered out of `children` above) or absorbed (an
        # unchanged signature — the agent has already seen this exact state). It is NOT
        # suppressed by parsing the final PROSE for status words (a removed keyword gate
        # that could silently orphan a child). The reminder fires once per CHANGE (a child
        # appearing/progressing/completing re-surfaces it) rather than every round, so the
        # agent is informed without an unbreakable loop; if the agent then finalizes with
        # children still unhandled, the no-tool / forced finalization paths append a loud
        # orphan note via _forced_orphan_note (P1 — never a silent loss).
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

    ctx_tokens = sum(
        estimate_tokens(_extract_plain_text_from_content(m.get("content")))
        for m in messages
    )
    raw_task_cost = accumulated_usage.get("cost")
    task_cost = float(raw_task_cost) if raw_task_cost is not None else None
    cost_text = f"${task_cost:.2f}" if task_cost is not None else "unknown"
    checkpoint_num = round_idx // REMINDER_INTERVAL

    tool_trace = _build_recent_tool_trace(messages)

    reminder = (
        f"[CHECKPOINT {checkpoint_num} — round {round_idx}/{max_rounds}]\n"
        f"Context: ~{ctx_tokens} tokens | Cost so far: {cost_text} | "
        f"Rounds remaining: {max_rounds - round_idx}\n"
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

    _emit_checkpoint_event(event_queue, task_id, drive_logs, {
        "checkpoint_number": checkpoint_num,
        "round": round_idx,
        "max_rounds": max_rounds,
        "context_tokens": ctx_tokens,
        "task_cost": task_cost,
    })

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
    cost_ceiling_usd: Optional[float],
    accumulated_usage: Optional[Dict[str, Any]],
    event_queue: Optional[queue.Queue] = None,
    task_id: str = "",
    drive_logs: Optional[pathlib.Path] = None,
) -> bool:
    """Thin transport over the task_pacing cost axis (v6.56.0): content,
    thresholds, and latch state live in ouroboros/task_pacing.py."""
    note = task_pacing.build_cost_budget_note(
        tools._ctx,
        start_remaining_usd=budget_remaining_usd,
        cost_ceiling_usd=cost_ceiling_usd,
        task_cost=(accumulated_usage or {}).get("cost"),
    )
    if note is None:
        return False
    _append_or_merge_user_message(messages, note.text)
    _emit_checkpoint_event(event_queue, task_id, drive_logs, note.checkpoint)
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
    cost_ceiling_usd: Optional[float] = None,
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
        budget_remaining_usd=budget_remaining_usd, cost_ceiling_usd=cost_ceiling_usd,
        accumulated_usage=accumulated_usage,
        event_queue=event_queue, task_id=task_id, drive_logs=drive_logs,
    )
    return bool(checkpoint or time_budget or cost_budget)


def _last_assistant_text(messages: List[Dict[str, Any]]) -> str:
    """Last real assistant text already produced this task — salvaged into the
    terminal answer when provider-death prevents a fresh final response, so
    useful work is never silently discarded (workspace files persist on disk
    regardless)."""
    for m in reversed(messages or []):
        if isinstance(m, dict) and m.get("role") == "assistant":
            content = m.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
    return ""


def _task_deadline_epoch(tools: ToolRegistry) -> Optional[float]:
    """Task deadline as epoch seconds, for deadline-bounded LLM retry backoff."""
    meta = getattr(tools._ctx, "task_metadata", {})
    if not isinstance(meta, dict):
        return None
    deadline = parse_deadline_ts(meta.get("deadline_at"))
    return deadline.timestamp() if deadline is not None else None


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
        enabled, not_found = [], []
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
                not_found.append(name)
        parts = []
        if enabled:
            parts.append(
                "✅ Tools are registered in the active capability envelope: "
                + ", ".join(enabled)
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
    """Inject owner messages received during task execution.

    Returns typed control signals drained from the mailbox (currently
    ``{"finalize_now": reason}`` when the supervisor opened a finalization
    grace window); control entries are routed structurally, never injected
    as owner prose.
    """
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
        from ouroboros.owner_mailbox import KIND_FINALIZE_NOW, KIND_OWNER_TEXT, drain_owner_entries
        for entry in drain_owner_entries(drive_root, task_id=task_id, seen_ids=_owner_msg_seen):
            kind = entry.get("kind") or KIND_OWNER_TEXT
            if kind == KIND_FINALIZE_NOW:
                controls["finalize_now"] = str(entry.get("text") or "deadline")
                continue
            dmsg = entry.get("text") or ""
            _record_owner_directive(
                owner_ctx,
                source="owner_mailbox",
                content=dmsg,
                msg_id=str(entry.get("msg_id") or ""),
            )
            _append_or_merge_user_message(messages, _owner_marked_content(dmsg))
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


def _run_round_compaction(
    messages: List[Dict[str, Any]],
    ctx: _CompactionRoundContext,
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Run at most one transcript compaction for this round.

    Manual (pending) and emergency compaction always run; routine compaction
    covers the local lane and owner low context mode (v6.33.0: mode is the SSOT —
    the per-model small-window remote override was removed with the static window
    table), and is skipped on self-check checkpoint rounds to avoid a duplicate
    summarizer call. Each branch persists a forensic checkpoint before compacting
    (P1: no silent truncation). Returns the possibly-rebound message list and any
    compaction usage record.
    """
    pending_compaction = getattr(ctx.tools._ctx, "_pending_compaction", None)
    if pending_compaction is not None:
        if _persist_compaction_checkpoint(
            messages, drive_root=ctx.drive_root, drive_logs=ctx.drive_logs, task_id=ctx.task_id,
            reason="manual", keep_recent=int(pending_compaction),
            round_idx=ctx.round_idx, event_queue=ctx.event_queue,
        ):
            messages, usage = compact_tool_history_llm(
                messages,
                keep_recent=pending_compaction,
                drive_root=ctx.drive_root,
                task_id=ctx.task_id,
            )
            ctx.tools._ctx._pending_compaction = None
            return messages, usage
        ctx.emit_progress("⚠️ Context compaction skipped: forensic checkpoint could not be persisted.")
        return messages, None

    # The owner low/max context MODE is the SSOT for the agent's own operating
    # window (BIBLE P1, v6.33.0): low => 400K-char emergency trigger + routine
    # compaction; max => 1.2M-char emergency-only (cache-friendly). No per-model
    # window table; the reactive provider-overflow detector (context.py) drops the
    # agent to low mode if a route's real window turns out smaller than assumed.
    emergency_chars = LOW_EMERGENCY_COMPACTION_CHARS if ctx.active_context_mode == "low" else EMERGENCY_COMPACTION_CHARS
    if _estimate_messages_chars(messages) > emergency_chars:
        # keep_recent must stay BELOW the current span count or the compactor
        # no-ops (len(spans) <= keep_recent returns as-is): a transcript over
        # the emergency byte threshold with only ~50 huge rounds previously
        # never compacted at all. Halve the history (floor 6), but ALWAYS
        # clamp below the span count so even 2-6 huge rounds compact; with a
        # single round there is nothing older to summarize.
        span_count = len(_tool_round_spans(messages))
        emergency_keep_recent = min(50, max(6, span_count // 2), max(1, span_count - 1))
        if _persist_compaction_checkpoint(
            messages, drive_root=ctx.drive_root, drive_logs=ctx.drive_logs, task_id=ctx.task_id,
            reason="emergency_context_size", keep_recent=emergency_keep_recent,
            round_idx=ctx.round_idx, event_queue=ctx.event_queue,
        ):
            return compact_tool_history_llm(
                messages,
                keep_recent=emergency_keep_recent,
                drive_root=ctx.drive_root,
                task_id=ctx.task_id,
            )
        ctx.emit_progress("⚠️ Emergency compaction skipped: forensic checkpoint could not be persisted.")
        return messages, None

    # Routine compaction runs only when local or in low context mode; never on
    # checkpoint rounds. Max mode relies on emergency compaction alone to preserve
    # prompt-cache hits (mode is the SSOT — no per-model small-window override).
    if not ctx.checkpoint_injected and (ctx.active_use_local or ctx.active_context_mode == "low"):
        if ctx.round_idx > 6 and len(messages) > 40:
            if _persist_compaction_checkpoint(
                messages, drive_root=ctx.drive_root, drive_logs=ctx.drive_logs, task_id=ctx.task_id,
                reason="routine", keep_recent=20,
                round_idx=ctx.round_idx, event_queue=ctx.event_queue,
            ):
                return compact_tool_history_llm(
                    messages,
                    keep_recent=20,
                    drive_root=ctx.drive_root,
                    task_id=ctx.task_id,
                )
    return messages, None


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
    # STATUS/budget drive root + root task id for the forced-finalization orphan note:
    # child results live under the parent BUDGET drive, NOT the (possibly forked)
    # drive_root, so the orphan scan must use this — same root get_task_result uses.
    status_drive_root: Optional[pathlib.Path] = None
    root_task_id: str = ""
    delivery_candidate: Optional[DeliveryCandidate] = None
    tools: Optional[ToolRegistry] = None
    llm_trace: Optional[Dict[str, Any]] = None
    incoming_messages: Optional[queue.Queue] = None
    owner_msg_seen: Optional[set] = None
    forced_service_evidence_fingerprint: str = ""
    finalization_reason: str = ""


def _handle_round_limit(ctx: _RoundLimitContext) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    finish_reason = f"⚠️ Task exceeded MAX_ROUNDS ({ctx.max_rounds}). Consider decomposing into subtasks via schedule_subagent."
    prompt = (
        f"[ROUND_LIMIT] {finish_reason} Produce your best final answer now from the "
        "verified work so far; clearly mark anything unverified or incomplete. An honest "
        "best-effort result is the expected outcome here, not a failure."
    )
    return _forced_final_answer(ctx, prompt=prompt, fallback_text=finish_reason, reason_code="round_limit")


def _handle_forced_finalization(ctx: _RoundLimitContext, reason: str) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """Cooperative finalize-and-exit when the supervisor opens a grace window.

    The supervisor sends a typed finalize_now control through the owner
    mailbox when the task deadline/hard-timeout is reached; this extracts one
    tool-less best final answer inside the grace window so a deadline NEVER
    returns emptiness.
    """
    ctx.finalization_reason = " ".join(str(reason or "deadline").split())[:200]
    fallback = f"⚠️ Task reached {ctx.finalization_reason}; finalization grace produced no answer."
    prompt = (
        f"[FINALIZE_NOW] The supervisor opened a finalization grace window (reason: {ctx.finalization_reason}). "
        "The task will be stopped shortly. Produce your best final answer NOW from the verified "
        "work so far; clearly mark anything unverified or incomplete. An honest best-effort "
        "result is the expected outcome here, not a failure."
    )
    return _forced_final_answer(ctx, prompt=prompt, fallback_text=fallback, reason_code="finalization_grace")


def _handle_provider_unavailable(ctx: _RoundLimitContext) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """Provider terminalization with unknown-outcome no-resend custody."""
    # A stale DeliveryCandidate is still the best complete text available when
    # the provider is dead. _forced_fallback_result preserves its original
    # evidence provenance and adds a host-owned resume disclosure rather than
    # laundering unchanged text onto the newer evidence fingerprint.
    candidate = _live_delivery_candidate(ctx)
    salvaged = candidate.full_text if candidate is not None else _last_assistant_text(ctx.messages)
    if candidate is None and not salvaged and ctx.drive_root is not None:
        # B2: the current (possibly compacted) transcript may no longer hold the
        # last useful assistant text, but every LLM round was persisted — fall back
        # to the durable salvage source named by the plan (latest_llm_response_text).
        try:
            from ouroboros.observability import latest_llm_response_text
            salvaged = latest_llm_response_text(pathlib.Path(ctx.drive_root), ctx.task_id) or ""
        except Exception:
            log.debug("latest_llm_response_text salvage failed", exc_info=True)
    fallback = salvaged or ""
    notice = _provider_unavailable_notice(ctx.accumulated_usage)
    if str(ctx.accumulated_usage.get("_last_llm_error_kind") or "") == "provider_outcome_unknown":
        live_trace = getattr(ctx, "llm_trace", None)
        llm_trace = live_trace if isinstance(live_trace, dict) else {}
        _finalize_forced_services(ctx, llm_trace)
        ctx.accumulated_usage["execution_status"] = "failed"
        ctx.accumulated_usage["reason_code"] = "provider_unavailable"
        return _forced_fallback_result(
            ctx,
            llm_trace,
            fallback,
            "provider_unavailable",
            source="provider_outcome_unknown_no_resend",
            host_notice=notice,
        )
    prompt = (
        "[PROVIDER_UNAVAILABLE] The model provider failed to return a usable response. "
        "Produce your best final answer NOW from the verified work so far; clearly mark "
        "anything unverified or incomplete. An honest best-effort result is expected here, not a failure."
    )
    return _forced_final_answer(
        ctx,
        prompt=prompt,
        fallback_text=fallback,
        reason_code="provider_unavailable",
        host_notice=notice,
    )


def _maybe_deadline_local_finalize(
    ctx: _RoundLimitContext, tools: ToolRegistry
) -> Optional[Tuple[str, Dict[str, Any], Dict[str, Any]]]:
    """Loop-local graceful finalization on a REAL task deadline.

    Headless runs (benchmarks, harbor) frequently get no supervisor finalize_now:
    the process is simply killed at the deadline, discarding any best-effort
    artifact. When a real deadline_at is set and less than the finalization-grace
    window remains, self-finalize one tool-less best answer here — independent of
    the supervisor — so a deadline NEVER returns emptiness. Never fires without a
    real deadline_at (no synthesized deadline; leaderboard timeouts stay legal)."""
    meta = getattr(tools._ctx, "task_metadata", {})
    if not isinstance(meta, dict):
        return None
    deadline = parse_deadline_ts(meta.get("deadline_at"))
    if deadline is None:
        return None
    remaining = (deadline - utc_now()).total_seconds()
    # v6.55.0: the plain finalization GRACE emit-window (task_pacing SSOT), NOT
    # the pct reserve — this path fires just before the kill to emit one answer,
    # so a percentage-of-total reserve would amputate the working tail (a 6h task
    # would self-finalize ~54 min early on a 15% profile). The pct reserve is an
    # acceptance-review gate concept only.
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
    limit_ctx: _RoundLimitContext, tools: ToolRegistry, controls: Dict[str, Any]
) -> Optional[Tuple[str, Dict[str, Any], Dict[str, Any]]]:
    """One early-exit gate per round: supervisor finalize_now first, then a
    loop-local real-deadline finalize. Returns the forced answer or None."""
    if controls.get("finalize_now"):
        return _handle_forced_finalization(limit_ctx, str(controls["finalize_now"]))
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
    # race. Late scratch results are intentionally not projected or recovered.
    if (
        str(child.get("parent_decision") or "").strip().lower() == "cancelled"
        and str(child.get("status") or "").strip().lower()
        in {"cancel_requested", "cancelled"}
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

    from ouroboros.outcomes import read_verification_receipts
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
        str(getattr(tools._ctx, "drive_root", "") or ctx.drive_root or ctx.status_drive_root or ctx.drive_logs.parent)
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
        "verification_receipts": read_verification_receipts(receipt_root, ctx.task_id),
        # Task-scoped service teardown can register declared outputs or surface an
        # output-finalization failure.  Those facts are produced outside an ordinary
        # tool call, so bind their stable projection explicitly; otherwise a host
        # acceptance panel could review the pre-teardown state.
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
    # Candidate text alone is not a review identity: the same full answer can be
    # regenerated after tool/child/verification evidence changes.  Refresh host
    # authority only from the panel the current acceptance pass explicitly names;
    # an older exact-text run must never be rediscovered by a hash-only scan.
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
) -> DeliveryCandidate:
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


def _record_forced_finalization(
    ctx: _RoundLimitContext,
    llm_trace: Dict[str, Any],
    *,
    reason_code: str,
    source: str,
    candidate: Optional[DeliveryCandidate],
) -> None:
    # Forced exits bypass the normal no-tool finalization gate. Project child
    # dispositions here, after services/evidence and the returned candidate
    # have been refreshed, so every forced return exposes the same terminal
    # child-result truth to the outcome reducer.
    _project_child_result_dispositions(ctx, llm_trace)
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
        "finalization_reason": str(getattr(ctx, "finalization_reason", "") or ""),
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
    is_control_intent = duplicate_protocol_key or (
        isinstance(parsed, dict)
        and str(parsed.get("delivery_control") or "") in {"keep", "replace"}
    )
    if not required:
        if _delivery_replace_required(candidate):
            # A writer/skill action cannot silently turn a short acknowledgement
            # into the new complete answer, even if a caller lost the transient
            # required latch. The candidate's typed control state is authoritative.
            required = True
            tools._ctx._delivery_control_required = True
        elif candidate.finalization_control == "skill_action_or_revision_required":
            # Preserve the historical bounded skill gate: an actual tool action
            # or a reconsidered full prose answer may proceed, but a typed keep
            # cannot acknowledge the gate. Do not inject the delivery JSON prompt
            # before the action because it would conflict with the instruction to
            # call the skill lifecycle tool.
            if not is_control_intent:
                return "fresh", _extract_plain_text_from_content(content)
            candidate.finalization_control = "skill_revision_required"
            required = True
            tools._ctx._delivery_control_required = True
        else:
            # An owner revision starts an ordinary substantive answer round. If
            # the model nevertheless follows the prior typed instruction, honor
            # that control structurally; service/effect/skill rounds are handled
            # by the replace-required branch above.
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
    # The control failed, not the retained text. Bind that unchanged text to
    # the evidence the failed control was meant to acknowledge so the stale
    # check cannot reopen another control round. It remains explicitly
    # unaccepted; the ordinary host acceptance gate still judges this exact
    # candidate/evidence pair before publication.
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


def _compose_host_notice(full_text: str, host_notice: str) -> str:
    """Append one visibly host-authored notice with normal Markdown spacing."""

    text = str(full_text or "")
    notice = str(host_notice or "").strip()
    if not notice or text.endswith(notice):
        return text
    return text + ("\n\n" if text else "") + notice


def _forced_finalization_lead(ctx: _RoundLimitContext, reason_code: str) -> str:
    """Render a typed finalization cause; only real hard rails say hard limit."""

    if reason_code == "round_limit":
        return "Finalized under the hard round limit"
    if reason_code == "budget_exhausted":
        return "Finalized under a hard budget/resource limit"
    if reason_code == "deadline_local":
        return "Finalized because the task deadline was reached"
    if reason_code == "provider_unavailable":
        if str(ctx.accumulated_usage.get("_last_llm_error_kind") or "") == "provider_outcome_unknown":
            return "Finalized because a provider request had an unknown outcome"
        return "Finalized because the model provider was unavailable"
    if reason_code == "children_unabsorbed":
        return "Finalized best-effort after the bounded child-absorption reminder"
    if reason_code == "finalization_grace":
        detail = " ".join(
            str(getattr(ctx, "finalization_reason", "") or "unspecified").split()
        )[:200]
        return f"Finalized after a supervisor finalization request (reason: {detail})"
    return "Finalized through forced finalization"


def _forced_finalization_notice(
    ctx: _RoundLimitContext,
    reason_code: str,
    detail: str = "",
) -> str:
    """Compose one visibly host-authored cause plus optional custody detail."""

    prefix = "⚠️ FINALIZATION NOTICE (host): "
    extra = str(detail or "").strip()
    if extra.startswith("⚠️ SYSTEM NOTICE (host): "):
        extra = extra[len("⚠️ SYSTEM NOTICE (host): "):].strip()
    notice = f"{prefix}{_forced_finalization_lead(ctx, reason_code)}."
    return notice + (f" {extra}" if extra else "")


def _forced_orphan_note(
    ctx: _RoundLimitContext,
    *,
    include_terminal: bool = True,
    reason_code: str = "",
) -> str:
    """A bounded note listing children the parent did NOT explicitly handle (discard/cancel),
    appended to a finalization so paid child work is never SILENTLY orphaned (P1; P5 — no
    prose parsing). On a FORCED finalization (deadline / provider death / finalize_now,
    ``include_terminal=True``) the parent was cut off and may not have seen completions, so
    RUNNING and COMPLETED-undecided children are both reported. On a NORMAL no-tool
    finalization (``include_terminal=False``) the agent was reminded of every change
    (including completions) before choosing to finalize, so only STILL-RUNNING undecided
    children — genuinely orphaned by finalizing mid-flight — are reported. Never raises."""
    try:
        from ouroboros.task_status import FINAL_STATUSES

        children = _direct_child_results(ctx)

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
            terminal = str(c.get("child_status") or "").strip().lower()
            if terminal and terminal != st:
                return f"{tid} [{lifecycle}; terminal_result={terminal}]"
            return f"{tid} [{lifecycle}]"

        notes: list[str] = []
        if undecided:
            listed = "; ".join(_label(c) for c in undecided[:10])
            more = f" (+{len(undecided) - 10} more)" if len(undecided) > 10 else ""
            lead = "At forced finalization" if include_terminal else "Finalized"
            detail = (
                "running ones may be incomplete, completed ones may be UNREAD"
                if include_terminal else
                "still-running children not absorbed or discarded"
            )
            notes.append(
                f"\n\n⚠️ NOTE (host): {lead} with {len(undecided)} child task(s) not "
                f"explicitly absorbed or discarded — {detail}. This child list is a snapshot "
                "at finalization time; listed statuses may change and are not guaranteed to be "
                f"final outcomes. Snapshot: {listed}{more}. Inspect with "
                "`get_task_result(<id>)` / `peek_task(<id>)`."
            )
        if deferred:
            listed = "; ".join(_label(c) for c in deferred[:10])
            more = f" (+{len(deferred) - 10} more)" if len(deferred) > 10 else ""
            notes.append(
                "\n\n⚠️ DEFERRED CHILD RESULTS (host): This child list is a snapshot at "
                "finalization time; listed statuses may change and are not guaranteed to be final "
                f"outcomes. Snapshot: {listed}{more}. These exact results were explicitly deferred, "
                "so this answer is degraded/best-effort rather than clean solved."
            )
        return "".join(notes)
    except Exception:
        return ""


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
    return text, usage, llm_trace


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

    if _force_plan_required(tools._ctx, llm_trace):
        attempts = int(getattr(tools._ctx, "_force_plan_reminder_count", 0) or 0)
        if attempts >= 2:
            limit_ctx.accumulated_usage.update(
                execution_status="failed", reason_code="swarm_force_plan_not_called",
            )
            return (
                "⚠️ SWARM_INITIATIVE_BLOCKED: plan_task was required for this swarm task but was not called.",
                limit_ctx.accumulated_usage,
                llm_trace,
            )
        tools._ctx._force_plan_reminder_count = attempts + 1
        if content and content.strip():
            messages.append({"role": "assistant", "content": content})
        _append_or_merge_user_message(
            messages,
            "[SWARM_INITIATIVE] plan_task is required before finalizing this task. "
            "Call plan_task now with an appropriate context_level, then continue.",
        )
        emit_progress("Swarm force-plan reminder injected before final response.")
        llm_trace["reasoning_notes"].append("Swarm force-plan reminder injected before final response.")
        _arm_delivery_control(tools, limit_ctx, llm_trace)
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
        # Skill finalization is an action gate, not a service notice. Preserve
        # the candidate without adding a conflicting JSON-only instruction: the
        # next round may run the required tool or provide the historically
        # allowed reconsidered full answer, but a typed keep cannot close it.
        if skill_finalization_injected_now:
            _hold_delivery_for_skill_action(tools, llm_trace)
        else:
            _arm_delivery_control(tools, limit_ctx, llm_trace)
        return None

    # Declared service outputs and teardown failures are acceptance evidence, not
    # postscript cleanup.  Finalize them before the authoritative host panel and,
    # when that changes evidence, require one complete replacement answer bound to
    # the new revision.  The finally-path calls the same idempotent helper as a
    # safety net for forced/error exits.
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
    normal_suffix = _forced_orphan_note(limit_ctx, include_terminal=False)
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
        )
    if isinstance(candidate, DeliveryCandidate):
        if normal_suffix:
            candidate.degraded = True
            candidate.degraded_reason = "host_child_status_suffix"
            _publish_delivery_candidate(tools, candidate, llm_trace)
        content = candidate.full_text

    tools._ctx._acceptance_loop_rails = {
        "round_idx": limit_ctx.round_idx,
        "max_rounds": limit_ctx.max_rounds,
        "task_cost_usd": limit_ctx.accumulated_usage.get("cost"),
    }
    # v6.78.0 (owner Q20/Q22): mirror the host-attested native-retrieval fact into the
    # trace so `build_task_acceptance_evidence` can show the reviewer whether the answer
    # was grounded in fetched pages. Reviewer-side only — the agent never sees it (it
    # receives the improvement capsule, not the evidence packet).
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
        # v6.71.1: an acceptance improvement pass is an ORDINARY substantive answer
        # round (like an owner revision, loop.py `_replace_delivery_candidate`
        # "fresh" path) — do NOT arm delivery-control here. Arming layered a
        # "return exactly one JSON object" directive on top of the OPEN OBLIGATIONS
        # ("address them directly") and the periodic plain-text self-check, and the
        # three conflicting finalization instructions in one message froze the model
        # into no-tool rounds that resubmitted the same answer. The next free-form
        # answer re-enters the acceptance panel, so blocking is not weakened. Other
        # lanes (services/force-plan/handoff) still arm delivery-control where JSON
        # keep/replace is genuinely needed.
        return None
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
            text, usage, forced_trace = _handle_forced_finalization(
                limit_ctx, str(post_controls.get("finalize_now") or "deadline"),
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
    )
    return str((final_msg or {}).get("content") or "").strip()


def _publish_model_forced_candidate(
    ctx: _RoundLimitContext,
    llm_trace: Dict[str, Any],
    full_text: str,
    reason_code: str,
) -> Optional[DeliveryCandidate]:
    """Replace the retained answer and invalidate any verdict for the old SHA."""

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
    """Preserve useful old text without pretending it absorbed newer evidence."""

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
    # The host-added disclosure is current, but the substantive answer it
    # qualifies is not. Preserve the answer's original evidence provenance so
    # every projection remains conservative instead of laundering unchanged
    # text onto the newer fingerprint.
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
    host_notice: str = "",
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """Return one exact candidate; reuse only current unchanged full text."""

    host_notice = _forced_finalization_notice(ctx, reason_code, host_notice)
    orphan_suffix = _forced_orphan_note(ctx, reason_code=reason_code)
    notice_suffix = f"\n\n{host_notice}"
    suffix = notice_suffix + orphan_suffix
    live_candidate = _live_delivery_candidate(ctx)
    fallback_is_retained_model_text = (
        isinstance(live_candidate, DeliveryCandidate)
        and fallback_text == live_candidate.full_text
    )
    candidate = _current_delivery_candidate(ctx, llm_trace)
    if candidate is not None:
        composed = _compose_delivery_suffix(
            _compose_host_notice(candidate.full_text, host_notice),
            orphan_suffix,
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
                    if retained_source else (
                        f"{source}_with_host_suffix"
                        if source != "host_fallback"
                        else "retained_candidate_with_host_suffix"
                    )
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
        # The preserved candidate is a previously model-produced complete answer.
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
            ctx.accumulated_usage["_best_effort_extracted"] = True
            _record_forced_finalization(
                ctx,
                llm_trace,
                reason_code=reason_code,
                source=f"{source}_stale_evidence_resume_required",
                candidate=candidate,
            )
            return candidate.full_text, ctx.accumulated_usage, llm_trace

    composed = _compose_delivery_suffix(
        _compose_host_notice(fallback_text, host_notice),
        orphan_suffix,
    )
    candidate = _publish_model_forced_candidate(
        ctx, llm_trace, composed, reason_code,
    )
    if fallback_is_retained_model_text:
        ctx.accumulated_usage["_best_effort_extracted"] = True
    _record_forced_finalization(
        ctx,
        llm_trace,
        reason_code=reason_code,
        source=source,
        candidate=candidate,
    )
    return composed, ctx.accumulated_usage, llm_trace


def _forced_final_answer(
    ctx: _RoundLimitContext,
    *,
    prompt: str,
    fallback_text: str,
    reason_code: str,
    host_notice: str = "",
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """Force one tool-less final answer; stamp the typed forced-finalization
    reason code (the best_effort outcome gate reads it downstream)."""
    live_trace = getattr(ctx, "llm_trace", None)
    llm_trace = live_trace if isinstance(live_trace, dict) else {}
    _finalize_forced_services(ctx, llm_trace)
    _append_or_merge_user_message(ctx.messages, prompt)
    # A forced summary is a new provider episode. The caller may already have
    # materialized a notice for the outage that triggered it; any new failure
    # must not inherit action claims from that earlier episode.
    ctx.accumulated_usage["_provider_recovery_actions"] = []
    extracted = ""
    for attempt in range(2):
        try:
            extracted = _call_forced_model_once(ctx)
        except BudgetExceeded:
            _drain_forced_owner_directives(ctx, llm_trace)
            raise
        except Exception:
            log.warning("Failed to get final response after %s", reason_code, exc_info=True)
            extracted = ""
        ctx.accumulated_usage["execution_status"] = "failed"
        ctx.accumulated_usage["reason_code"] = reason_code
        if str(ctx.accumulated_usage.get("_last_llm_error_kind") or "") == "provider_outcome_unknown":
            _drain_forced_owner_directives(ctx, llm_trace)
            _finalize_forced_services(ctx, llm_trace)
            return _forced_fallback_result(
                ctx,
                llm_trace,
                fallback_text,
                reason_code,
                source="provider_outcome_unknown_no_resend",
                host_notice=_provider_unavailable_notice(ctx.accumulated_usage),
            )
        if not _drain_forced_owner_directives(ctx, llm_trace):
            break
        if attempt == 1:
            return _forced_fallback_result(
                ctx,
                llm_trace,
                (
                    "⚠️ A new owner directive arrived during the forced refresh and could "
                    "not be incorporated safely before the hard stop. Resume the task to "
                    "produce an answer bound to the latest directive."
                ),
                reason_code,
                source="late_owner_directive_requires_resume",
                host_notice=host_notice,
            )
        _finalize_forced_services(ctx, llm_trace)
        _append_or_merge_user_message(
            ctx.messages,
            "[FORCED_OWNER_REFRESH] A new typed owner directive arrived while the prior "
            "forced answer was being generated. Discard that stale draft and produce one "
            "new complete answer bound to every owner directive now present.",
        )

    if extracted:
        # Typed fact for the best_effort outcome gate: a REAL model answer
        # was extracted (host fallback strings never set this).
        ctx.accumulated_usage["_best_effort_extracted"] = True
        finalization_notice = _forced_finalization_notice(ctx, reason_code, host_notice)
        full_text = _compose_delivery_suffix(
            _compose_host_notice(extracted, finalization_notice),
            _forced_orphan_note(ctx, reason_code=reason_code),
        )
        candidate = _publish_model_forced_candidate(
            ctx, llm_trace, full_text, reason_code,
        )
        _record_forced_finalization(
            ctx,
            llm_trace,
            reason_code=reason_code,
            source="model_with_host_notice",
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
        host_notice=host_notice,
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


def _maybe_downgrade_max_unconfirmed(mode: str, use_local: bool, model: str = "", *, allow_fetch: bool = False) -> str:
    """Select Low only from positive exact-route evidence of a sub-1M window.

    Missing/stale/failed evidence is UNKNOWN, not an invented 200K capability:
    ordinary tasks try the owner-selected Max projection and may take the single
    task-local Low retry only after a real provider overflow.  The P3 commit gate
    has its own fail-closed >=1M contract and never calls this helper.
    """
    if mode != "max":
        return mode
    try:
        from ouroboros.capability_evidence import ONE_MILLION
        from ouroboros.context import _context_fit_route

        _route, evidence = _context_fit_route(
            {"model": model, "use_local_model": use_local},
            allow_fetch=allow_fetch,
        )
        known = (
            str(getattr(evidence, "status", "")) in {"confirmed", "asserted"}
            and not bool(getattr(evidence, "stale", False))
            and int(getattr(evidence, "window_tokens", 0) or 0) > 0
        )
        if known and int(evidence.window_tokens or 0) < ONE_MILLION:
            log.info(
                "Exact route evidence reports a sub-1M context window "
                "(%s tokens, use_local=%s); using the task-local Low projection.",
                evidence.window_tokens, use_local,
            )
            return "low"
    except Exception:
        log.debug("Context-fit capability check unavailable; preserving Max", exc_info=True)
    return mode


def _apply_overrides_and_regate_mode(ctx, active_model, active_use_local, active_effort, active_context_mode):
    """Apply per-round runtime overrides, then re-gate max-mode at point-of-use if the
    active route changed (a mid-loop switch_model / local-route change — the start-of-
    loop gate only saw the initial route). Positive small-window evidence selects Low;
    unknown evidence remains Max until a real overflow (v6.64)."""
    _route_before = (active_model, active_use_local)
    active_model, active_use_local, active_effort = _apply_runtime_overrides(
        ctx, active_model, active_use_local, active_effort,
    )
    if (active_model, active_use_local) != _route_before:
        active_context_mode = _maybe_downgrade_max_unconfirmed(
            get_context_mode(), active_use_local, active_model,
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
        pathlib.Path(tools._ctx.drive_root),
        str(getattr(evidence, "route_fp", "") or ""),
        str(route.get("model") or model),
    )
    known_window = (
        str(getattr(evidence, "status", "") or "") in {"confirmed", "asserted"}
        and not bool(getattr(evidence, "stale", False))
        and int(getattr(evidence, "window_tokens", 0) or 0) > 0
    )
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
    initial_mode = "low" if preferred == "max" and max_projection.fits_known_window is False else preferred
    rebound = replace(
        plan,
        preferred_mode=preferred,
        initial_mode=initial_mode,
        model=str(route.get("model") or model),
        provider=str(route.get("provider") or ""),
        route_fp=str(getattr(evidence, "route_fp", "") or ""),
        evidence_status=str(getattr(evidence, "status", "") or ""),
        evidence_stale=bool(getattr(evidence, "stale", False)),
        window_tokens=window_tokens,
        max_projection=max_projection,
        low_projection=low_projection,
    )
    mode = str(rebound.initial_mode_with_tools(tool_schemas) or initial_mode)
    projected_prompt_tokens = rebound.projected_tokens_with_tools("max", tool_schemas)
    if preferred == "max" and known_window:
        max_transcript = rebound.reproject_transcript(messages, "max")
        projected_prompt_tokens = int(
            estimate_context_prompt_tokens(max_transcript, tool_schemas) * ratio
        )
        if projected_prompt_tokens + int(rebound.output_reserve_tokens or 0) > window_tokens:
            mode = "low"
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
                "evidence_status": rebound.evidence_status,
                "window_tokens": rebound.window_tokens,
                "projected_prompt_tokens": projected_prompt_tokens,
            },
        )
    except Exception:
        log.debug("Failed to emit route-switch context-fit checkpoint", exc_info=True)
    return rebound, mode


def _visible_round_text(content: Any) -> str:
    """The round's visible assistant text as a plain string. A provider may return ``content`` as
    a string OR a list of typed blocks; collect the ``text`` of every block EXCEPT reasoning ones
    (Anthropic ``thinking``/``redacted_thinking``, Gemini ``part.thought``) — the exact complement
    of extract_display_reasoning. A regular Gemini part carries ``text`` with NO ``type``, so keying
    on the ABSENCE of a reasoning marker (not on ``type == 'text'``) avoids dropping real answer
    text; a non-empty block list never stringifies to a raw Python repr, and a thinking-only list
    correctly reads as 'no visible text' (letting narration fall back to readable reasoning)."""
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
    """Emit the round's progress bubble: the visible assistant text, or — for a pure tool-call round
    with no visible text — readable reasoning the provider already returned. The reasoning fallback
    is DISPLAY-ONLY: emitted to the UI bubble but NOT recorded in ``reasoning_notes`` (which feeds
    build_trace_summary / task summaries) and never appended to the transcript, so it cannot leak out
    of the display path into the durable trace or back to a provider. Gated by OUROBOROS_REASONING_SUMMARY."""
    visible_text = _visible_round_text(content)
    if visible_text:
        emit_progress(visible_text)
        llm_trace["reasoning_notes"].append(visible_text)
    elif str(os.environ.get("OUROBOROS_REASONING_SUMMARY", "auto")).strip().lower() != "off":
        display_reasoning = LLMClient.extract_display_reasoning(msg)
        if display_reasoning:
            emit_progress(display_reasoning)


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
        # Red-verification one-shot nudge: the agent's most recent host-attested verify
        # receipt is RED and unreconciled — finalizing over your own failing check is a
        # self-contradiction (Bible P3/P12), distinct from the receipt_absent case below
        # (that is "no grounding"; this is "grounding says FAIL"). Ordered BEFORE the FR3
        # verify nudge. Binary latch; advisory (the agent may still finalize with reasoning);
        # forced-finalization paths return earlier and bypass it. Structural — keyed on the
        # typed receipt status, never content (Bible P5). Benchmark-neutral wording.
        _failed_receipt = latest_unreconciled_failed_verification(drive_root, task_id)
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
        # Exit-masking one-shot ADVISORY nudge (v6.52.2): the agent's latest PASSING verify check
        # can LAUNDER the real exit code (a `| tail`/`grep`/`|| true` pipeline reports exit 0 even
        # when the underlying runner failed — the false-green tutanota hit). Distinct from the red
        # nudge (that is "grounding says FAIL"; this is "grounding says PASS but may be laundered").
        # Ordered AFTER the red nudge. Binary latch; ADVISORY (the agent may still finalize with
        # reasoning); forced-finalization paths return earlier and bypass it. Flag-driven on the
        # typed receipt sensor, never content (Bible P5). Benchmark-neutral wording.
        _masked_receipt = latest_unreconciled_masked_verification(drive_root, task_id)
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
        # Criterion-provenance one-shot ADVISORY nudge (v6.54.4): the latest passing
        # verification used an AGENT-DEFINED criterion with no stated basis — the check
        # is green, but the success criterion itself was synthesized. One reminder to
        # confirm equivalence with the task's real requirement (or state the basis via
        # criterion_basis). Ordered AFTER the masked nudge, BEFORE FR3. Flag-driven on
        # the typed receipt field, never content (P5); forced paths bypass earlier.
        _agent_defined = latest_agent_defined_verification(drive_root, task_id)
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
    if not getattr(tools._ctx, "_verify_nudged", False) and should_nudge_verification(llm_trace, drive_root, task_id):
        # FR3 one-shot verify-before-done nudge: real effects, no host-attested grounding
        # yet. Binary latch (not a tunable counter), sibling BEFORE the acceptance-review
        # gate so it reaches both required and auto. Forced finalization paths return
        # earlier and bypass it (they land best_effort).
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
    # A3 one-shot no-op nudge: a declared deliverable (non-empty expected_output) but the
    # turn made NO tool calls, produced NO reviewable effects, and carries NO FINAL ANSWER
    # marker — a structural about-to-finalize-without-attempting signal (same condition
    # family as the M2 expected_output_ungrounded flag). Own latch, ordered AFTER the verify
    # nudge; never forces acceptance review; forced-finalization paths return earlier and
    # bypass it. Structural facts only (no refusal-text matching).
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
    # P2 one-shot final-answer-marker nudge: the turn produced REAL work (tool calls or
    # reviewable effects) AND visible prose, but carries NO FINAL ANSWER marker — so the
    # typed extractor would drop it and a forced/deadline finalization would score empty
    # even though the answer is sitting in the prose. We strengthen the BEHAVIOR (ask the
    # agent to mark its OWN answer) rather than mining prose into a claimed answer (Bible P5;
    # codex-confirmed that prose-mining in core would harm ordinary users). Own latch,
    # ordered AFTER verify/red/A3 (verification grounding outranks formatting); mutually
    # exclusive with the A3 no-op nudge (which is the no-work case). Forced-finalization
    # paths return earlier and bypass it. Structural facts only (no content matching).
    # The protocol gate is sufficient: answer_protocol="final_answer_line" itself declares
    # a machine-extracted deliverable, so the nudge must not ALSO require a declared
    # expected_output — GAIA-shaped contracts carry the question in `objective` with
    # expected_output empty, and the extra gate silently suppressed the one salvage
    # surface (a v6.56.0 run finalized a last-round refusal with an empty typed answer
    # despite 24 tool calls of real research).
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


def _call_round_model(ctx: _RoundModelCallContext) -> Tuple[Any, float, str]:
    """Dispatch one ordinary round and its single confirmed-overflow Low retry."""
    plan = ctx.context_fit_plan
    if plan is not None and str(ctx.active_model or "") == str(getattr(plan, "model", "") or ""):
        ctx.accumulated_usage["_context_route_fp"] = str(getattr(plan, "route_fp", "") or "")
        ctx.accumulated_usage["_context_prompt_estimate"] = estimate_context_prompt_tokens(
            ctx.messages, ctx.tool_schemas,
        )
        ctx.accumulated_usage["_context_fit_mode"] = ctx.active_context_mode
    msg, cost = call_llm_with_retry(
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
        attempt_cap=ctx.attempt_cap,
        allow_server_web_search=_server_web_allowed_by_task(ctx.tools._ctx),
    )
    first_dispatched_attempts = int(
        ctx.accumulated_usage.get("_llm_dispatched_attempts") or 0
    )
    should_retry_low = (
        msg is None
        and plan is not None
        and str(ctx.active_model or "") == str(getattr(plan, "model", "") or "")
        and str(getattr(plan, "preferred_mode", "")) == "max"
        and ctx.active_context_mode != "low"
        and str(ctx.accumulated_usage.get("_last_llm_error_kind") or "") == "context_overflow"
        and not bool(ctx.accumulated_usage.get("_context_fit_low_retry_used"))
    )
    if not should_retry_low:
        return msg, cost, ctx.active_context_mode
    checkpoint_ok = _persist_compaction_checkpoint(
        ctx.messages,
        drive_root=ctx.drive_root,
        drive_logs=ctx.drive_logs,
        task_id=ctx.task_id,
        reason="confirmed_context_overflow_low_retry",
        keep_recent=max(0, len(_tool_round_spans(ctx.messages))),
        round_idx=ctx.round_idx,
        event_queue=ctx.event_queue,
        checkpoint_kind="pre_context_fit_low_retry",
        call_type="context_fit_checkpoint",
    )
    if not checkpoint_ok:
        return msg, cost, ctx.active_context_mode
    ctx.accumulated_usage["_context_fit_low_retry_used"] = True
    ctx.messages[:] = plan.reproject_transcript(ctx.messages, "low")
    ctx.tools._ctx.messages = ctx.messages
    ctx.tools._ctx.active_context_mode = "low"
    ctx.accumulated_usage["_context_prompt_estimate"] = estimate_context_prompt_tokens(
        ctx.messages, ctx.tool_schemas,
    )
    ctx.accumulated_usage["_context_fit_mode"] = "low"
    _emit_checkpoint_event(ctx.event_queue, ctx.task_id, ctx.drive_logs, {
        "checkpoint_kind": "context_fit_low_retry",
        "round": ctx.round_idx,
        "model": ctx.active_model,
        "route_fp": str(getattr(plan, "route_fp", "") or ""),
        "core_sha256": str(getattr(plan, "core_sha256", "") or ""),
        "preferred_mode": "max",
        "effective_mode": "low",
        "toast_once": f"{ctx.task_id}:context-fit-low:{ctx.round_idx}",
        "owner_visible": True,
    })
    try:
        msg, cost = call_llm_with_retry(
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
            attempt_cap=1,
            allow_server_web_search=_server_web_allowed_by_task(ctx.tools._ctx),
        )
    finally:
        ctx.accumulated_usage["_llm_dispatched_attempts"] = first_dispatched_attempts + int(
            ctx.accumulated_usage.get("_llm_dispatched_attempts") or 0)
    return msg, cost, "low"


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
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """Apply the physical-attempt dispatch rail without spending a wrap-up call."""
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
    # A physical budget rail is terminal for this execution.  Finalize task
    # services before testing or creating a DeliveryCandidate so no pre-teardown
    # answer can be published against stale service/output evidence.  The loop's
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
    """Release executor, task services, and mailbox after every loop exit."""
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
    # The full DeliveryCandidate is intentionally loop-local. Only its compact
    # hash/revision projection remains in llm_trace after this cleanup. Clear it
    # after the idempotent teardown safety net so cleanup cannot erase the only
    # complete answer before service evidence is collected.
    ctx.tools._ctx._delivery_candidate = None
    ctx.tools._ctx._delivery_control_required = False
    if ctx.drive_root is None or not ctx.task_id:
        return
    try:
        from ouroboros.owner_mailbox import cleanup_task_mailbox

        cleanup_task_mailbox(ctx.drive_root, ctx.task_id)
    except Exception:
        log.debug("Failed to cleanup task mailbox", exc_info=True)


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
    """Run the LLM-with-tools loop and return final text, usage, and trace."""
    ctx = tools._ctx
    ctx._delivery_candidate = None
    ctx._delivery_candidate_revision = 0
    ctx._delivery_control_required = False
    ctx._delivery_evidence_revision = 0
    ctx._delivery_evidence_fingerprint = ""
    _initialize_owner_directives(ctx, messages)
    task_model_override = str(getattr(ctx, "task_model_override", "") or "").strip()
    active_model = task_model_override or llm.default_model()
    active_effort = initial_effort
    if getattr(ctx, "task_use_local_override", None) is not None:
        active_use_local = bool(ctx.task_use_local_override)
    else:
        active_use_local = os.environ.get("USE_LOCAL_MAIN", "").lower() in ("true", "1")
    # Root probes exact-route fit; unknown routes get honest Max, never invented 200K.
    _ctx_meta = getattr(ctx, "task_metadata", {})
    _is_subagent = (
        isinstance(_ctx_meta, dict)
        and str(_ctx_meta.get("delegation_role") or "").strip().lower() == "subagent"
    )
    _preferred_context_mode = get_context_mode()
    context_fit_plan = getattr(ctx, "context_fit_plan", None)
    if (
        context_fit_plan is not None
        and str(getattr(context_fit_plan, "preferred_mode", "")) == _preferred_context_mode
    ):
        active_context_mode = str(getattr(context_fit_plan, "initial_mode", "") or _preferred_context_mode)
    else:
        active_context_mode = _maybe_downgrade_max_unconfirmed(
            _preferred_context_mode, active_use_local, active_model, allow_fetch=not _is_subagent,
        )
    llm_trace: Dict[str, Any] = {"reasoning_notes": [], "tool_calls": []}
    accumulated_usage: Dict[str, Any] = {}
    max_retries = 3
    cost_ceiling_usd = _resolve_task_cost_ceiling(ctx, budget_remaining_usd)
    from ouroboros.tools import tool_discovery as _td
    _td.set_registry(tools)

    tool_schemas = initial_tool_schemas(tools)
    tool_schemas, _enabled_extra_tools = _setup_dynamic_tools(tools, tool_schemas, messages)
    if context_fit_plan is not None and str(
        getattr(context_fit_plan, "preferred_mode", "")
    ) == _preferred_context_mode:
        fit_with_tools = getattr(context_fit_plan, "initial_mode_with_tools", None)
        if callable(fit_with_tools):
            tool_aware_mode = str(fit_with_tools(tool_schemas) or active_context_mode)
            if tool_aware_mode != active_context_mode:
                messages[:] = context_fit_plan.reproject_transcript(messages, tool_aware_mode)
                active_context_mode = tool_aware_mode

    if _preferred_context_mode == "max" and active_context_mode != "max":
        # Make the effective-vs-preferred downgrade owner-visible and durable.
        projected_prompt = 0
        project_with_tools = getattr(context_fit_plan, "projected_tokens_with_tools", None)
        if callable(project_with_tools):
            projected_prompt = int(project_with_tools("max", tool_schemas) or 0)
        _emit_checkpoint_event(event_queue, task_id, drive_logs, {
            "checkpoint_kind": "context_mode_downgraded",
            "preferred_mode": _preferred_context_mode,
            "effective_mode": active_context_mode,
            "model": active_model,
            "use_local": active_use_local,
            "reason": "known_route_projection_does_not_fit",
            "route_fp": str(getattr(context_fit_plan, "route_fp", "") or ""),
            "window_tokens": int(getattr(context_fit_plan, "window_tokens", 0) or 0),
            "projected_prompt_tokens": projected_prompt,
            "core_sha256": str(getattr(context_fit_plan, "core_sha256", "") or ""),
        })

    tools._ctx.event_queue = event_queue
    tools._ctx.task_id = task_id
    tools._ctx.messages = messages
    stateful_executor = StatefulToolExecutor()
    exit_ctx = _LoopExitContext(
        tools, drive_root, task_id, event_queue, drive_logs, accumulated_usage, llm_trace,
    )
    _owner_msg_seen: set = set()
    MAX_ROUNDS = _resolve_loop_max_rounds()
    round_idx = 0
    limit_ctx: Optional[_RoundLimitContext] = None
    try:
        while True:
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
                # A cross-FAMILY switch_model / per-task override mid-conversation:
                # proactively strip the prior family's provider-private reasoning/
                # thinking blocks from the canonical history so the new family does
                # not 400 on a signature it cannot validate (stripping is always
                # safe — it loses only reasoning continuity). Same family is a no-op.
                _sanitized = LLMClient.sanitize_reasoning_on_model_switch(messages, _prev_active_model, active_model)
                if _sanitized is not messages:
                    messages[:] = _sanitized
            ctx.active_context_mode = active_context_mode  # CW2: switch_model reads this to refuse a sub-1M switch while max-sized
            ctx.active_model = active_model  # publish the round's REAL model (incl. switch_model / per-task override) so tools (native screenshot vision-routing) don't read the stale global OUROBOROS_MODEL env

            # One forced-wrap-up context per round: consumed by the round-limit
            # path and the supervisor finalize_now control path below.
            limit_ctx = _RoundLimitContext(
                messages, llm, active_model, active_effort, max_retries, drive_logs,
                task_id, round_idx, event_queue, accumulated_usage, task_type,
                active_use_local, MAX_ROUNDS, drive_root=drive_root,
                incoming_messages=incoming_messages, owner_msg_seen=_owner_msg_seen,
            )
            _finalize_limit_ctx(limit_ctx, tools, llm_trace)
            if round_idx > MAX_ROUNDS:
                text, accumulated_usage, forced_trace = _handle_round_limit(limit_ctx)
                _merge_finalization_trace(llm_trace, forced_trace)
                return text, accumulated_usage, llm_trace

            _controls = _drain_incoming_messages(
                messages,
                incoming_messages,
                drive_root,
                task_id,
                event_queue,
                _owner_msg_seen,
                owner_ctx=ctx,
            )
            # Early-exit per round: supervisor finalize_now, else loop-local real-
            # deadline finalize (headless runs that get no finalize_now) — finalize
            # best-effort rather than be killed mid-step with nothing.
            _early_final = _maybe_early_finalize(limit_ctx, tools, _controls)
            if _early_final is not None:
                text, accumulated_usage, forced_trace = _early_final
                _merge_finalization_trace(llm_trace, forced_trace)
                return text, accumulated_usage, llm_trace

            _checkpoint_injected = _inject_round_checkpoints(
                round_idx=round_idx, max_rounds=MAX_ROUNDS, messages=messages, accumulated_usage=accumulated_usage,
                emit_progress=emit_progress, tools=tools, event_queue=event_queue, task_id=task_id,
                drive_logs=drive_logs, budget_remaining_usd=budget_remaining_usd, cost_ceiling_usd=cost_ceiling_usd)

            messages, _compaction_usage = _run_round_compaction(
                messages,
                _CompactionRoundContext(
                    tools=tools,
                    drive_root=drive_root,
                    drive_logs=drive_logs,
                    task_id=task_id,
                    round_idx=round_idx,
                    event_queue=event_queue,
                    active_use_local=active_use_local,
                    active_context_mode=active_context_mode,
                    checkpoint_injected=_checkpoint_injected,
                    emit_progress=emit_progress,
                    active_model=active_model,
                ),
            )
            if tools._ctx.messages is not messages:
                tools._ctx.messages = messages
            limit_ctx.messages = messages  # WA2: provider-death finalize must salvage the COMPACTED transcript
            if _compaction_usage:
                add_usage(accumulated_usage, _compaction_usage)
                _cm = get_light_model()
                _cc = (
                    float(_compaction_usage["cost"])
                    if _compaction_usage.get("cost") is not None
                    else estimate_cost_optional(
                        _cm,
                        int(_compaction_usage.get("prompt_tokens") or 0),
                        int(_compaction_usage.get("completion_tokens") or 0),
                        int(_compaction_usage.get("cached_tokens") or 0),
                        int(_compaction_usage.get("cache_write_tokens") or 0),
                        _compaction_usage.get("prompt_cache_ttl"),
                        provider=str(_compaction_usage.get("provider") or "openrouter"),
                    )
                )
                emit_llm_usage_event(event_queue, task_id, _cm, _compaction_usage, _cc, "compaction")

            seal_task_transcript(messages)

            # Per-round typed recovery projection. Only physical attempts proven
            # by usage/capture facts are added below; classifier decisions and
            # configured retry budgets are not action evidence.
            accumulated_usage["_provider_recovery_actions"] = []
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

            if msg is None:
                _record_last_call_same_model_retries(accumulated_usage, active_model)
                if (
                    str(accumulated_usage.get("_last_llm_error_kind") or "")
                    != "provider_outcome_unknown"
                ):
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
                if msg is None:
                    # Provider-death: join the unified honest best-effort shelf
                    # (deadline/budget/round-limit) instead of discarding useful
                    # workspace state with a bare error string.
                    text, accumulated_usage, forced_trace = _handle_provider_unavailable(limit_ctx)
                    _merge_finalization_trace(llm_trace, forced_trace)
                    return text, accumulated_usage, llm_trace

            tool_calls = msg.get("tool_calls") or []
            content = msg.get("content")
            _latch_final_answer_marker(llm_trace, content, current_tool_calls=tool_calls)
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

            _prepare_post_tool_budget_context(
                tools, limit_ctx, llm_trace, active_model, active_use_local, active_effort,
            )
            budget_result = _check_budget_limits(
                limit_ctx,
                budget_remaining_usd,
                cost_ceiling_usd=cost_ceiling_usd,
            )
            if budget_result is not None:
                text, accumulated_usage, budget_trace = budget_result
                _merge_finalization_trace(llm_trace, budget_trace)
                return text, accumulated_usage, llm_trace

    except BudgetExceeded as exc:
        return _handle_budget_exceeded(exc, exit_ctx, limit_ctx=limit_ctx)
    finally:
        _cleanup_loop_resources(stateful_executor, exit_ctx)
