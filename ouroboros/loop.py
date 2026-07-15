"""LLM tool loop: call model, execute tools, repeat until final response."""

from __future__ import annotations

import json
import os
import queue
import pathlib
import time
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, List, Optional, Tuple

import logging

from ouroboros.llm import LLMClient, normalize_reasoning_effort, add_usage
from ouroboros import task_pacing
from ouroboros.config import adaptive_quorum, get_context_mode, get_light_model, get_review_enforcement, get_task_review_mode, resolve_effort
from ouroboros.outcomes import extract_final_answer, latest_agent_defined_verification, latest_unreconciled_failed_verification, latest_unreconciled_masked_verification, should_nudge_verification, turn_has_reviewable_effects
from ouroboros.observability import new_call_id, persist_call
from ouroboros.tool_policy import initial_tool_schemas, list_non_core_tools
from ouroboros.tools.registry import ToolRegistry
from ouroboros.context import build_user_content, estimate_context_prompt_tokens
from ouroboros.context_budget import EMERGENCY_COMPACTION_CHARS, LOW_EMERGENCY_COMPACTION_CHARS, PROACTIVE_COMPACTION_CHARS
from ouroboros.context_compaction import _tool_round_spans, compact_tool_history_llm
from ouroboros.deadline_utils import parse_deadline_ts, utc_now
from ouroboros.utils import estimate_tokens
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
    if accumulated_usage.get("context_overflow_suggest_low"):
        return (
            " ⚠️ The context overflowed the model window. Switching to low context "
            "mode (Settings → Behavior, or the chat toggle) fits ~200K / local "
            "models by serving ARCHITECTURE as a navigation map and compacting "
            "memory sooner — without changing the model or reasoning effort."
        )
    kind = str(accumulated_usage.get("_last_llm_error_kind") or "").strip()
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
    """True when a reviewed plan_task completed in this trace.

    Reads the structured ``plan_review_aggregate`` flag captured from the FULL
    tool result at execution time (loop_tool_execution); the old substring
    check against the 700-char trace preview could never see the aggregate
    marker at the end of a long plan output, wedging swarm tasks in the
    force-plan reminder loop.
    """
    for call in llm_trace.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        if (
            str(call.get("tool") or "") == "plan_task"
            and not bool(call.get("is_error"))
            and bool(call.get("plan_review_aggregate"))
        ):
            return True
    return False


def _force_plan_required(ctx: Any, llm_trace: Dict[str, Any]) -> bool:
    metadata = getattr(ctx, "task_metadata", {}) if isinstance(getattr(ctx, "task_metadata", {}), dict) else {}
    return bool(metadata.get("force_plan")) and not _force_plan_completed(llm_trace)


def _check_budget_limits(
    budget_remaining_usd: Optional[float],
    accumulated_usage: Dict[str, Any],
    round_idx: int,
    messages: List[Dict[str, Any]],
    llm: LLMClient,
    active_model: str,
    active_effort: str,
    max_retries: int,
    drive_logs: pathlib.Path,
    task_id: str,
    event_queue: Optional[queue.Queue],
    llm_trace: Dict[str, Any],
    task_type: str = "task",
    use_local: bool = False,
    deadline_ts: Optional[float] = None,
    cost_ceiling_usd: Optional[float] = None,
) -> Optional[Tuple[str, Dict[str, Any], Dict[str, Any]]]:
    """Return a final-response tuple when budget limits require stopping.

    ``cost_ceiling_usd`` is the in-task hard-stop resolved ONCE at loop start
    from ``task_contract.budget_profile.cost_hard_stop_pct``
    (``task_pacing.resolve_cost_ceiling_usd``); None means no in-task cost stop
    — the global budget-exhaustion gate below still applies."""
    if budget_remaining_usd is None:
        return None

    raw_task_cost = accumulated_usage.get("cost")
    task_cost = float(raw_task_cost) if raw_task_cost is not None else None

    if budget_remaining_usd <= 0:
        finish_reason = "🚫 Task rejected. Total budget exhausted. Please increase TOTAL_BUDGET in settings."
        accumulated_usage["execution_status"] = "failed"
        accumulated_usage["reason_code"] = "budget_exhausted"
        # One bounded tool-less best-effort extraction before rejecting: if the
        # task already produced verified work, salvage it instead of returning
        # nothing (the typed best_effort outcome gate reads this reason code).
        if round_idx > 1:
            try:
                _append_or_merge_user_message(
                    messages,
                    "[BUDGET LIMIT] Total budget exhausted. Produce your best final answer NOW "
                    "from the verified work so far; clearly mark anything unverified or "
                    "incomplete. An honest best-effort result is the expected outcome here.",
                )
                final_msg, _cost = _call_llm_with_retry(
                    llm, messages, active_model, None, active_effort,
                    1, drive_logs, task_id, round_idx, event_queue, accumulated_usage, task_type,
                    use_local=use_local,
                    deadline_ts=deadline_ts,
                )
                accumulated_usage["execution_status"] = "failed"
                accumulated_usage["reason_code"] = "budget_exhausted"
                final_text = str((final_msg or {}).get("content") or "").strip()
                if final_text:
                    accumulated_usage["_best_effort_extracted"] = True
                    return final_text, accumulated_usage, llm_trace
            except Exception:
                log.warning("Failed to extract best-effort answer after budget exhaustion", exc_info=True)
        return finish_reason, accumulated_usage, llm_trace

    from ouroboros.config import SETTINGS_DEFAULTS as _DEFAULTS
    _per_task_default = str(_DEFAULTS["OUROBOROS_PER_TASK_COST_USD"])
    per_task_limit = float(os.environ.get("OUROBOROS_PER_TASK_COST_USD", _per_task_default) or _per_task_default)
    if task_cost is not None and task_cost >= per_task_limit and round_idx % 10 == 0:
        _append_or_merge_user_message(
            messages,
            f"[COST NOTE] Task spent ${task_cost:.3f}, which is at or above the per-task soft threshold of ${per_task_limit:.2f}. Continue only if the expected value still justifies the cost.",
        )

    if cost_ceiling_usd is not None and task_cost is not None and task_cost > cost_ceiling_usd:
        finish_reason = (
            f"Task spent ${task_cost:.3f} (over the in-task cost ceiling ${cost_ceiling_usd:.2f} "
            f"of remaining ${budget_remaining_usd:.2f}). Budget exhausted."
        )
        _append_or_merge_user_message(
            messages,
            f"[BUDGET LIMIT] {finish_reason} Produce your best final answer now from the "
            "verified work so far; clearly mark anything unverified or incomplete. An honest "
            "best-effort result is the expected outcome here, not a failure.",
        )
        try:
            final_msg, final_cost = _call_llm_with_retry(
                llm, messages, active_model, None, active_effort,
                max_retries, drive_logs, task_id, round_idx, event_queue, accumulated_usage, task_type,
                use_local=use_local,
                deadline_ts=deadline_ts,
            )
            accumulated_usage["execution_status"] = "failed"
            accumulated_usage["reason_code"] = "budget_exhausted"
            extracted = str((final_msg or {}).get("content") or "").strip()
            if extracted:
                accumulated_usage["_best_effort_extracted"] = True
                return extracted, accumulated_usage, llm_trace
            return finish_reason, accumulated_usage, llm_trace
        except Exception:
            log.warning("Failed to get final response after budget limit", exc_info=True)
            accumulated_usage["execution_status"] = "failed"
            accumulated_usage["reason_code"] = "budget_exhausted"
            return finish_reason, accumulated_usage, llm_trace
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
            break
    ctx._task_acceptance_reviewed = False
    ctx._task_acceptance_fence_generation_mismatch = False
    llm_trace.pop("root_phase_checkpoint", None)
    llm_trace["review_decision"] = {
        "eligibility": "pending_owner_followup",
        "trigger": "owner_followup_after_acceptance",
    }
    _set_acceptance_decision(llm_trace, {
        "status": "revision_requested",
        "source": "owner_followup",
        "rationale": "The owner added a directive after acceptance evidence was frozen; re-review is required.",
    })
    return released


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
        compact = [{
            "task_id": str(row.get("task_id") or row.get("id") or ""),
            "parent_task_id": str(row.get("parent_task_id") or ""),
            "status": str(row.get("status") or "unknown"),
            "artifact_status": str(row.get("artifact_status") or ""),
        } for row in rows if isinstance(row, dict)]
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
    meta = getattr(ctx, "task_metadata", {})
    meta = meta if isinstance(meta, dict) else {}
    task_id = str(getattr(ctx, "task_id", "") or "")
    root_id = str(meta.get("root_task_id") or getattr(ctx, "root_task_id", "") or task_id)
    if root_id and root_id != task_id:
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


def _set_acceptance_decision(llm_trace: Dict[str, Any], decision: Dict[str, Any]) -> None:
    previous = llm_trace.get("acceptance_decision") if isinstance(llm_trace.get("acceptance_decision"), dict) else {}
    merged = dict(decision)
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
    seen = {str(o.get("id")) for o in obligations if isinstance(o, dict)}
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
        oid = "ob-" + hashlib.sha256(
            json.dumps([item, recommendation], ensure_ascii=False).encode("utf-8")
        ).hexdigest()[:12]
        if oid in seen:
            continue
        seen.add(oid)
        obligations.append({
            "id": oid,
            "item": item,
            "recommendation": recommendation,
            "status": "open",
            "disposition": "",
            "disposition_reason": "",
        })


def _open_acceptance_obligations(llm_trace: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        o for o in (llm_trace.get("acceptance_obligations") or [])
        if isinstance(o, dict) and not str(o.get("disposition") or "").strip()
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
        ob["disposition"] = "addressed"
        ob["disposition_reason"] = "resolved by revision: the clean re-review returned no findings"
        ob["status"] = "disposed_by_re_review"
    _set_acceptance_decision(llm_trace, {
        "status": "accepted",
        "source": "task_acceptance_review",
        "rationale": "Clean PASS re-review; open obligations closed by the revision (dissent, if any, stays advisory).",
        "dissent_noted": dissent_noted,
    })
    return True


def _format_obligations_clause(open_obligations: List[Dict[str, Any]]) -> str:
    if not open_obligations:
        return ""
    lines = [
        "",
        "OPEN OBLIGATIONS (blocking review policy): give a disposition for each via the "
        "task_acceptance_review tool's obligation_dispositions (addressed / rejected / deferred + reason) "
        "or address them directly before your final answer:",
    ]
    for o in open_obligations[:5]:
        lines.append(f"  {o.get('id')}: {o.get('item')} — {o.get('recommendation')}")
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


def _execute_task_acceptance_panel(ctx: _TaskAcceptanceContext) -> Any:
    """Build immutable evidence and perform the one substantive host panel."""
    from ouroboros.review_evidence import build_task_acceptance_evidence
    from ouroboros.review_substrate import (
        HARDNESS_ADVISORY_VISIBLE,
        ReviewRequest,
        reviewer_slots,
        run_review_request,
    )

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
        include_recent_commit=committed_this_turn,
        canonical_subject=str(ctx.content or ""),
        subtree_statuses=ctx.subtree_statuses,
    )
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
    ctx.llm_trace.setdefault("review_runs", []).append(run_record)
    return run_record


def _apply_task_acceptance_result(ctx: _TaskAcceptanceContext, result: Any) -> bool:
    """Apply one panel result; return whether the agent must take another round."""
    from ouroboros.review_substrate import (
        build_improvement_capsule,
        dissent_findings,
        task_acceptance_is_clean,
    )

    _record_host_acceptance_run(ctx, result)
    capsule = build_improvement_capsule(result)
    dissent = dissent_findings(result)
    blocking_lane = ctx.mode == "required" and get_review_enforcement() == "blocking"
    if blocking_lane:
        _collect_acceptance_obligations(ctx.llm_trace, result)
    open_obligations = _open_acceptance_obligations(ctx.llm_trace) if blocking_lane else []
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
                "status": "accepted",
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
    if capsule and pass_ok:
        _set_acceptance_decision(ctx.llm_trace, {
            "status": "revision_requested",
            "source": "task_acceptance_review",
            "rationale": "A compact advisory improvement capsule was fed back for one bounded revision pass.",
            "dissent_noted": bool(dissent),
        })
        ctx.tools._ctx._task_acceptance_improvement_passes = ctx.passes_done + 1
        if not _end_task_acceptance_fence(ctx.tools._ctx, outcome="revision"):
            ctx.tools._ctx._task_acceptance_reviewed = True
            _set_acceptance_decision(ctx.llm_trace, {
                "status": "review_degraded",
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
            "status": "review_degraded",
            "source": "task_acceptance_review",
            "rationale": "Acceptance reviewers did not reach a valid quorum.",
            "degraded_reasons": list(getattr(result, "degraded_reasons", []) or []),
            "open_obligations": [str(item.get("id")) for item in open_obligations],
        })
        ctx.emit_progress(
            "Task acceptance review: DEGRADED (no valid quorum; not recorded as PASS)."
        )
        return False
    if capsule and open_obligations:
        _set_acceptance_decision(ctx.llm_trace, {
            "status": "best_effort_open_obligations",
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
            "status": "finalized_after_capsule",
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
            "status": "review_failed",
            "source": "task_acceptance_review",
            "rationale": "A valid acceptance reviewer FAIL had no additional capsule text.",
            "dissent_noted": bool(dissent),
        })
        ctx.emit_progress("Task acceptance review: FAIL (finalizing with a failed review verdict).")
    elif open_obligations:
        _set_acceptance_decision(ctx.llm_trace, {
            "status": "best_effort_open_obligations",
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
            "status": "accepted",
            "source": "task_acceptance_review",
            "rationale": "No actionable task-acceptance changes were suggested.",
            "dissent_noted": bool(dissent),
        })
        ctx.emit_progress(f"Task acceptance review: {result.aggregate_signal} (no changes suggested).")
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
    ctx.llm_trace.setdefault("review_runs", []).append({
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
    })
    _set_acceptance_decision(ctx.llm_trace, {
        "status": "review_degraded",
        "source": "task_acceptance_review",
        "rationale": "The mandatory host acceptance panel failed before a valid quorum.",
        "degraded_reasons": [f"{type(exc).__name__}: {safe_error}"],
    })
    ctx.emit_progress("Task acceptance review: DEGRADED after host review infrastructure failure.")
    return False


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
    """Run the root-owned acceptance gate once for the current deliverable."""
    mode = get_task_review_mode()
    _latch_final_answer_marker(llm_trace, content)
    if getattr(tools._ctx, "_task_acceptance_reviewed", False):
        return False
    meta = getattr(tools._ctx, "task_metadata", {})
    meta = meta if isinstance(meta, dict) else {}
    root_id = str(meta.get("root_task_id") or getattr(tools._ctx, "root_task_id", "") or task_id)
    eligible, trigger = _task_acceptance_eligible(
        mode,
        llm_trace,
        bool(getattr(tools._ctx, "is_direct_chat", False)),
        is_root_task=not root_id or root_id == str(task_id or getattr(tools._ctx, "task_id", "") or ""),
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
        "eligibility": "eligible" if eligible else "not_eligible",
        "trigger": trigger,
    }
    if not eligible:
        return False
    fence_ok, _fence_token = _begin_task_acceptance_fence(tools._ctx, task_id)
    if not fence_ok:
        llm_trace["review_decision"] = {
            "eligibility": "acceptance_fence_failed",
            "trigger": trigger,
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
        _set_acceptance_decision(llm_trace, {
            "status": launch_reason,
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
    )
    try:
        messages_before_apply = list(messages)
        obligations_were_present = "acceptance_obligations" in llm_trace
        obligations_before_apply = [
            dict(row) if isinstance(row, dict) else row
            for row in (llm_trace.get("acceptance_obligations") or [])
        ]
        passes_before_apply = int(
            getattr(tools._ctx, "_task_acceptance_improvement_passes", 0) or 0
        )
        another_round = _apply_task_acceptance_result(
            review_ctx, _execute_task_acceptance_panel(review_ctx),
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
        from ouroboros.task_status import FINAL_STATUSES, find_child_tasks, format_subagent_absorption_message

        metadata = getattr(tools._ctx, "task_metadata", {}) if isinstance(getattr(tools._ctx, "task_metadata", {}), dict) else {}
        status_drive_root = pathlib.Path(
            str(metadata.get("budget_drive_root") or getattr(tools._ctx, "budget_drive_root", "") or "")
            or drive_root
        )
        children = find_child_tasks(
            status_drive_root,
            parent_task_id=task_id,
            root_task_id=str(metadata.get("root_task_id") or task_id),
            exclude_task_id=task_id,
            # Absorption is a PER-NODE gate: a task absorbs only its OWN direct
            # children (each level absorbs its own; depth is handled by every level
            # doing this). scope="direct" stops a leaf from getting a false
            # children_unabsorbed reminder about its parent/siblings (v6.57.0).
            scope="direct",
        )
        # D#7: a child the parent EXPLICITLY decided about (discard_child_result /
        # cancel_task stamp parent_decision) is handled — drop it from the reminder so the
        # signal is the structured decision, not a phrase parsed from the final text (P5).
        children = [
            child for child in children
            if str(child.get("parent_decision") or "").strip().lower() not in ("discarded", "cancelled")
        ]
        signature = "|".join(
            f"{child.get('task_id') or child.get('id')}:{child.get('status')}:{len(str(child.get('result') or ''))}"
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
            lines = ["All currently discovered tools are already in your active set.", "", "[CAPABILITY_OMISSION_MANIFEST]"]
            for item in omissions:
                lines.append(
                    f"- {item.get('surface', 'unknown')}: {item.get('reason', 'unknown')} "
                    f"({item.get('error', 'no detail')})"
                )
            return "\n".join(lines)
        lines = [f"**{len(non_core)} additional tools available** (use `enable_tools` to activate):\n"]
        for t in non_core:
            lines.append(f"- **{t['name']}**: {t['description'][:120]}")
        if omissions:
            lines.append("\n[CAPABILITY_OMISSION_MANIFEST]")
            for item in omissions:
                lines.append(
                    f"- {item.get('surface', 'unknown')}: {item.get('reason', 'unknown')} "
                    f"({item.get('error', 'no detail')})"
                )
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
        lines = ["[CAPABILITY_OMISSION_MANIFEST]"]
        for item in omissions:
            lines.append(
                f"- {item.get('surface', 'unknown')}: {item.get('reason', 'unknown')} "
                f"({item.get('error') or item.get('resource') or 'no detail'})"
            )
        _append_or_merge_user_message(messages, "[SYSTEM NOTICE]\n" + "\n".join(lines))

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

    # Proactive compaction: fire before emergency when context is growing large
    # in max mode with a known window. Keeps 70% of recent history to avoid
    # emergency overflow on the next round.
    if (
        ctx.active_context_mode == "max"
        and not ctx.checkpoint_injected
        and _estimate_messages_chars(messages) > PROACTIVE_COMPACTION_CHARS
    ):
        span_count = len(_tool_round_spans(messages))
        proactive_keep_recent = min(50, max(6, int(span_count * 0.7)), max(1, span_count - 1))
        if _persist_compaction_checkpoint(
            messages, drive_root=ctx.drive_root, drive_logs=ctx.drive_logs, task_id=ctx.task_id,
            reason="proactive_context_size", keep_recent=proactive_keep_recent,
            round_idx=ctx.round_idx, event_queue=ctx.event_queue,
        ):
            return compact_tool_history_llm(
                messages,
                keep_recent=proactive_keep_recent,
                drive_root=ctx.drive_root,
                task_id=ctx.task_id,
            )

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
    fallback = f"⚠️ Task reached {reason or 'deadline'}; finalization grace produced no answer."
    prompt = (
        f"[FINALIZE_NOW] The supervisor opened a finalization grace window (reason: {reason or 'deadline'}). "
        "The task will be stopped shortly. Produce your best final answer NOW from the verified "
        "work so far; clearly mark anything unverified or incomplete. An honest best-effort "
        "result is the expected outcome here, not a failure."
    )
    return _forced_final_answer(ctx, prompt=prompt, fallback_text=fallback, reason_code="finalization_grace")


def _handle_provider_unavailable(ctx: _RoundLimitContext) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """Provider-death terminalization (P2 unified best-effort shelf): the model
    returned no usable response after the transport same-model reroute + retries
    (+ any configured cross-model fallback). Join the SAME honest best-effort
    shelf as deadline/budget/round-limit instead of discarding workspace state
    with a bare error string — one tool-less final answer (which itself benefits
    from the same-model reroute) and, failing that, the last assistant text
    already produced."""
    salvaged = _last_assistant_text(ctx.messages)
    if not salvaged and ctx.drive_root is not None:
        # B2: the current (possibly compacted) transcript may no longer hold the
        # last useful assistant text, but every LLM round was persisted — fall back
        # to the durable salvage source named by the plan (latest_llm_response_text).
        try:
            from ouroboros.observability import latest_llm_response_text
            salvaged = latest_llm_response_text(pathlib.Path(ctx.drive_root), ctx.task_id) or ""
        except Exception:
            log.debug("latest_llm_response_text salvage failed", exc_info=True)
    if salvaged:
        fallback = salvaged
    else:
        fallback = (
            "⚠️ The model provider returned no usable response after retries and same-model reroute."
            f"{_provider_failure_hint(ctx.accumulated_usage)}{_provider_recovery_hint(ctx.accumulated_usage)} "
            "Any files written so far are preserved in the workspace."
        )
    prompt = (
        "[PROVIDER_UNAVAILABLE] The model provider failed to return a usable response. "
        "Produce your best final answer NOW from the verified work so far; clearly mark "
        "anything unverified or incomplete. An honest best-effort result is expected here, not a failure."
    )
    return _forced_final_answer(ctx, prompt=prompt, fallback_text=fallback, reason_code="provider_unavailable")


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


def _finalize_limit_ctx(ctx: "_RoundLimitContext", tools: Any) -> "_RoundLimitContext":
    """Resolve the deadline + STATUS/budget drive root + root task id from the live
    ToolContext onto an already-constructed round-limit context (child results live under
    the parent BUDGET drive, not the forked drive_root). The dataclass itself bundles the
    13 per-round fields (so no >8-param builder function is needed — DEVELOPMENT param
    rule); this fills only the 3 ctx-derived fields. Returns the same (mutated) ctx."""
    meta = getattr(tools._ctx, "task_metadata", {}) if isinstance(getattr(tools._ctx, "task_metadata", {}), dict) else {}
    ctx.deadline_ts = _task_deadline_epoch(tools)
    ctx.status_drive_root = pathlib.Path(
        str(meta.get("budget_drive_root") or getattr(tools._ctx, "budget_drive_root", "") or "")
        or (ctx.drive_root if ctx.drive_root is not None else pathlib.Path(ctx.drive_logs).parent)
    )
    ctx.root_task_id = str(meta.get("root_task_id") or ctx.task_id)
    return ctx


def _forced_orphan_note(ctx: _RoundLimitContext, *, include_terminal: bool = True) -> str:
    """A bounded note listing children the parent did NOT explicitly handle (discard/cancel),
    appended to a finalization so paid child work is never SILENTLY orphaned (P1; P5 — no
    prose parsing). On a FORCED finalization (deadline / provider death / finalize_now,
    ``include_terminal=True``) the parent was cut off and may not have seen completions, so
    RUNNING and COMPLETED-undecided children are both reported. On a NORMAL no-tool
    finalization (``include_terminal=False``) the agent was reminded of every change
    (including completions) before choosing to finalize, so only STILL-RUNNING undecided
    children — genuinely orphaned by finalizing mid-flight — are reported. Never raises."""
    try:
        # Child results live under the parent BUDGET drive (status_drive_root), not the
        # forked drive_root — use the same root get_task_result / _compute_subagent_handoff
        # use, or a forked/nested finalization scans the wrong tree and omits the note.
        status_root = ctx.status_drive_root or ctx.drive_root or pathlib.Path(ctx.drive_logs).parent
        if status_root is None or not ctx.task_id:
            return ""
        from ouroboros.task_status import FINAL_STATUSES, find_child_tasks

        children = find_child_tasks(
            pathlib.Path(status_root),
            parent_task_id=ctx.task_id,
            root_task_id=str(ctx.root_task_id or ctx.task_id),
            exclude_task_id=ctx.task_id,
            scope="direct",  # orphan note is per-node: only MY direct children (v6.57.0)
        )

        def _undecided(c: Dict[str, Any]) -> bool:
            if str(c.get("parent_decision") or "").strip().lower() in ("discarded", "cancelled"):
                return False  # explicitly handled
            if not include_terminal and str(c.get("status") or "").strip().lower() in FINAL_STATUSES:
                return False  # completed children were already surfaced via the reminder
            return True

        undecided = [c for c in children if _undecided(c)]
        if not undecided:
            return ""

        def _label(c: Dict[str, Any]) -> str:
            tid = str(c.get("task_id") or c.get("id") or "?")
            st = str(c.get("status") or "?").strip().lower()
            return f"{tid} [{'running' if st not in FINAL_STATUSES else st}]"

        listed = "; ".join(_label(c) for c in undecided[:10])
        more = f" (+{len(undecided) - 10} more)" if len(undecided) > 10 else ""
        lead = "finalized under a hard limit with" if include_terminal else "finalized with"
        detail = (
            "running ones may be incomplete, completed ones may be UNREAD"
            if include_terminal else
            "still-running children not absorbed or discarded"
        )
        return (
            f"\n\n⚠️ NOTE: {lead} {len(undecided)} child task(s) not explicitly absorbed or "
            f"discarded — {detail}: {listed}{more}. Inspect with get_task_result(<id>) / "
            f"peek_task(<id>)."
        )
    except Exception:
        return ""


def _running_undecided_children(ctx: _RoundLimitContext) -> list[Dict[str, Any]]:
    try:
        status_root = ctx.status_drive_root or ctx.drive_root or pathlib.Path(ctx.drive_logs).parent
        if status_root is None or not ctx.task_id:
            return []
        from ouroboros.task_results import STATUS_RUNNING
        from ouroboros.task_status import FINAL_STATUSES, find_child_tasks

        children = find_child_tasks(
            pathlib.Path(status_root),
            parent_task_id=ctx.task_id,
            root_task_id=str(ctx.root_task_id or ctx.task_id),
            exclude_task_id=ctx.task_id,
            scope="direct",  # absorption gate is per-node: only MY direct children (v6.57.0)
        )
        out: list[Dict[str, Any]] = []
        for child in children:
            if str(child.get("parent_decision") or "").strip().lower() in ("discarded", "cancelled"):
                continue
            status = str(child.get("status") or "").strip().lower()
            if status in FINAL_STATUSES or status != STATUS_RUNNING:
                continue
            out.append(child)
        return out
    except Exception:
        return []


def _task_may_delegate(tools: ToolRegistry) -> bool:
    try:
        ctx = tools._ctx
        contract = getattr(ctx, "task_contract", {}) if isinstance(getattr(ctx, "task_contract", {}), dict) else {}
        metadata = getattr(ctx, "task_metadata", {}) if isinstance(getattr(ctx, "task_metadata", {}), dict) else {}
        if not contract and isinstance(metadata.get("task_contract"), dict):
            contract = metadata.get("task_contract")
        if not contract:
            return False
        budget = contract.get("delegation_budget") if isinstance(contract.get("delegation_budget"), dict) else {}
        return bool(budget.get("may_delegate", True) or budget.get("may_fan_out", True))
    except Exception:
        return False


def _maybe_enforce_child_absorption_gate(
    tools: ToolRegistry,
    limit_ctx: _RoundLimitContext,
    content: Any,
    messages: List[Dict[str, Any]],
    emit_progress: Callable[[str], None],
    llm_trace: Dict[str, Any],
) -> Optional[Tuple[str, Dict[str, Any], Dict[str, Any]] | str]:
    if not _task_may_delegate(tools):
        return None
    undecided = _running_undecided_children(limit_ctx)
    if not undecided:
        return None
    if not getattr(tools._ctx, "_child_absorption_reminded", False):
        tools._ctx._child_absorption_reminded = True
        if content and str(content).strip():
            messages.append({"role": "assistant", "content": content})
        listed = "; ".join(str(c.get("task_id") or c.get("id") or "?") for c in undecided[:10])
        reminder = (
            "[CHILD_ABSORPTION_REQUIRED]\n"
            "You still have RUNNING child task(s) in this task tree: "
            f"{listed}. Before a clean final answer, wait/inspect them with wait_task/get_task_result, "
            "or make an explicit decision with cancel_task / discard_child_result. This is a "
            "bounded reminder; ignoring it will finalize best_effort, not clean."
        )
        _append_or_merge_user_message(messages, reminder)
        emit_progress("Child absorption reminder injected before final response.")
        llm_trace["reasoning_notes"].append("Child absorption reminder injected before final response.")
        return "continue"
    text, usage, _discarded_trace = _forced_final_answer(
        limit_ctx,
        prompt=(
            "[FINALIZE_WITH_UNABSORBED_CHILDREN]\n"
            "You still have running child tasks and already received one child-absorption reminder. "
            "Produce an honest best-effort final answer now; name unabsorbed children explicitly."
        ),
        fallback_text="⚠️ Finalized best-effort with unabsorbed running child tasks.",
        reason_code="children_unabsorbed",
    )
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
        return None
    handoff_msg = _compute_subagent_handoff(tools, limit_ctx.drive_root, limit_ctx.task_id, content)
    if handoff_msg:
        if content and content.strip():
            messages.append({"role": "assistant", "content": content})
        _append_or_merge_user_message(messages, f"[SYSTEM REMINDER]\n{handoff_msg}")
        emit_progress("Subagent handoff status refreshed before final response.")
        llm_trace["reasoning_notes"].append("Subagent handoff status refreshed before final response.")
        return None
    absorption_result = _maybe_enforce_child_absorption_gate(
        tools, limit_ctx, content, messages, emit_progress, llm_trace,
    )
    if absorption_result == "continue":
        return None
    if absorption_result is not None:
        return absorption_result
    if _maybe_inject_finalization_nudges(
        tools, limit_ctx.drive_root, limit_ctx.task_id, llm_trace, content, messages, emit_progress,
    ):
        return None
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
        return None

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
            return None
        if provisional_assistant is not None and messages[-1] is provisional_assistant:
            messages.pop()
        if post_controls.get("finalize_now"):
            return _handle_forced_finalization(
                limit_ctx, str(post_controls.get("finalize_now") or "deadline"),
            )
    # Append a loud orphan note for any still-running child not absorbed/discarded.
    return _handle_text_response(
        (content or "") + _forced_orphan_note(limit_ctx, include_terminal=False),
        llm_trace,
        limit_ctx.accumulated_usage,
    )


def _forced_final_answer(
    ctx: _RoundLimitContext,
    *,
    prompt: str,
    fallback_text: str,
    reason_code: str,
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """Force one tool-less final answer; stamp the typed forced-finalization
    reason code (the best_effort outcome gate reads it downstream)."""
    llm_trace: Dict[str, Any] = {}
    _append_or_merge_user_message(ctx.messages, prompt)
    orphan_note = _forced_orphan_note(ctx)
    try:
        final_msg, _final_cost = call_llm_with_retry(
            ctx.llm, ctx.messages, ctx.active_model, None, ctx.active_effort,
            ctx.max_retries, ctx.drive_logs, ctx.task_id, ctx.round_idx, ctx.event_queue, ctx.accumulated_usage, ctx.task_type,
            use_local=ctx.active_use_local,
            deadline_ts=ctx.deadline_ts,
        )
        ctx.accumulated_usage["execution_status"] = "failed"
        ctx.accumulated_usage["reason_code"] = reason_code
        extracted = str((final_msg or {}).get("content") or "").strip()
        if extracted:
            # Typed fact for the best_effort outcome gate: a REAL model answer
            # was extracted (host fallback strings never set this).
            ctx.accumulated_usage["_best_effort_extracted"] = True
            return extracted + orphan_note, ctx.accumulated_usage, llm_trace
        return fallback_text + orphan_note, ctx.accumulated_usage, llm_trace
    except Exception:
        log.warning("Failed to get final response after %s", reason_code, exc_info=True)
        ctx.accumulated_usage["execution_status"] = "failed"
        ctx.accumulated_usage["reason_code"] = reason_code
        return fallback_text + orphan_note, ctx.accumulated_usage, llm_trace


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
    latched = str(ctx.llm_trace.get("best_valid_final_answer") or "").strip()
    latched_is_current = (
        latched
        and len(ctx.llm_trace.get("tool_calls") or [])
        <= int(ctx.llm_trace.get("best_valid_final_answer_tools") or 0)
    )
    if latched_is_current:
        ctx.accumulated_usage["_best_effort_extracted"] = True
        return latched, ctx.accumulated_usage, ctx.llm_trace
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
    return (
        message,
        ctx.accumulated_usage,
        ctx.llm_trace,
    )


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
    if ctx.drive_root is None or not ctx.task_id:
        return
    try:
        from ouroboros.tools.services import stop_task_services

        finalized = stop_task_services(ctx.tools._ctx)
        stopped = [service for service in finalized if service.get("lifecycle") != "kept"]
        kept = [service for service in finalized if service.get("lifecycle") == "kept"]
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
    except Exception:
        log.debug("Failed to stop task services", exc_info=True)
    try:
        from ouroboros.owner_mailbox import cleanup_task_mailbox

        cleanup_task_mailbox(ctx.drive_root, ctx.task_id)
    except Exception:
        log.debug("Failed to cleanup task mailbox", exc_info=True)


def _should_use_compact_tools(context_fit_plan=None) -> bool:
    """Decide whether to use compact tool schemas (name+description only, no parameters).

    Compact mode saves ~20K tokens by omitting parameter schemas from non-core
    tools. Enabled when:
    - OUROBOROS_COMPACT_TOOLS=true env var, OR
    - Context fit plan projects that MAX mode won't fit the known window
    """
    if os.environ.get("OUROBOROS_COMPACT_TOOLS", "").lower() in ("true", "1", "yes"):
        return True
    if context_fit_plan is None:
        return False
    try:
        known_window = (
            str(getattr(context_fit_plan, "evidence_status", "") or "") in {"confirmed", "asserted"}
            and not bool(getattr(context_fit_plan, "evidence_stale", True))
            and int(getattr(context_fit_plan, "window_tokens", 0) or 0) > 0
        )
        if not known_window:
            return False
        max_proj = getattr(context_fit_plan, "max_projection", None)
        if max_proj is None:
            return False
        projected = int(getattr(max_proj, "calibrated_tokens", 0) or 0)
        output_reserve = int(getattr(context_fit_plan, "output_reserve_tokens", 0) or 0)
        window = int(getattr(context_fit_plan, "window_tokens", 0) or 0)
        return projected + output_reserve > window
    except Exception:
        return False


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

    tool_schemas = initial_tool_schemas(tools, compact=_should_use_compact_tools(context_fit_plan))
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
            )
            _finalize_limit_ctx(limit_ctx, tools)
            if round_idx > MAX_ROUNDS:
                text, accumulated_usage, _ = _handle_round_limit(limit_ctx)
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
                text, accumulated_usage, _ = _early_final
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
                    text, accumulated_usage, _ = _handle_provider_unavailable(limit_ctx)
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

            budget_result = _check_budget_limits(
                budget_remaining_usd, accumulated_usage, round_idx, messages,
                llm, active_model, active_effort, max_retries, drive_logs,
                task_id, event_queue, llm_trace, task_type, active_use_local,
                deadline_ts=_task_deadline_epoch(tools), cost_ceiling_usd=cost_ceiling_usd)
            if budget_result is not None:
                return budget_result

    except BudgetExceeded as exc:
        return _handle_budget_exceeded(exc, exit_ctx)
    finally:
        _cleanup_loop_resources(stateful_executor, exit_ctx)
