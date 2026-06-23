"""LLM tool loop: call model, execute tools, repeat until final response."""

from __future__ import annotations

import json
import os
import queue
import pathlib
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import logging

from ouroboros.llm import LLMClient, normalize_reasoning_effort, add_usage
from ouroboros.config import adaptive_quorum, get_context_mode, get_finalization_grace_sec, get_light_model, get_pacing_interval_sec, get_task_review_mode, resolve_effort
from ouroboros.outcomes import turn_has_reviewable_effects
from ouroboros.observability import new_call_id, persist_call
from ouroboros.tool_policy import initial_tool_schemas, list_non_core_tools
from ouroboros.tools.registry import ToolRegistry
from ouroboros.context import build_user_content
from ouroboros.context_budget import EMERGENCY_COMPACTION_CHARS, LOW_EMERGENCY_COMPACTION_CHARS
from ouroboros.context_compaction import _tool_round_spans, compact_tool_history_llm
from ouroboros.deadline_utils import parse_deadline_ts, utc_now
from ouroboros.utils import estimate_tokens

from ouroboros.loop_tool_execution import (
    StatefulToolExecutor,
    handle_tool_calls,
)
from ouroboros.loop_llm_call import call_llm_with_retry, emit_llm_usage_event, estimate_cost

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
) -> Optional[Tuple[str, Dict[str, Any], Dict[str, Any]]]:
    """Return a final-response tuple when budget limits require stopping."""
    if budget_remaining_usd is None:
        return None

    task_cost = accumulated_usage.get("cost", 0)

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

    budget_pct = task_cost / budget_remaining_usd if budget_remaining_usd > 0 else 1.0

    from ouroboros.config import SETTINGS_DEFAULTS as _DEFAULTS
    _per_task_default = str(_DEFAULTS["OUROBOROS_PER_TASK_COST_USD"])
    per_task_limit = float(os.environ.get("OUROBOROS_PER_TASK_COST_USD", _per_task_default) or _per_task_default)
    if task_cost >= per_task_limit and round_idx % 10 == 0:
        _append_or_merge_user_message(
            messages,
            f"[COST NOTE] Task spent ${task_cost:.3f}, which is at or above the per-task soft threshold of ${per_task_limit:.2f}. Continue only if the expected value still justifies the cost.",
        )

    if budget_pct > 0.5:
        finish_reason = f"Task spent ${task_cost:.3f} (>50% of remaining ${budget_remaining_usd:.2f}). Budget exhausted."
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
    elif budget_pct > 0.3 and round_idx % 10 == 0:
        _append_or_merge_user_message(messages, f"[INFO] Task spent ${task_cost:.3f} of ${budget_remaining_usd:.2f}. Wrap up if possible.")

    return None


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
) -> None:
    """Persist the pre-compaction transcript so compaction is only a view."""
    root = pathlib.Path(drive_root) if drive_root is not None else pathlib.Path(drive_logs).parent
    call_id = new_call_id("compaction_checkpoint")
    try:
        ref = persist_call(
            root,
            task_id=task_id,
            call_id=call_id,
            call_type="compaction_checkpoint",
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
            "checkpoint_kind": "pre_compaction_transcript",
            "round": round_idx,
            "reason": reason,
            "keep_recent": keep_recent,
            "checkpoint_ref": ref.get("manifest_ref"),
        })
        return True
    except Exception:
        log.debug("Failed to persist pre-compaction transcript checkpoint", exc_info=True)
        _emit_checkpoint_event(event_queue, task_id, drive_logs, {
            "checkpoint_kind": "pre_compaction_transcript",
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
            # view_image (local-file, native context, NOT web-gated) re-views the image;
            # vlm_query is in _WEB_TOOLS and is blocked under allowed_resources.web=false.
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


def _task_acceptance_eligible(mode: str, llm_trace: Dict[str, Any], is_direct_chat: bool) -> tuple[bool, str]:
    """Return ``(host_should_review, trigger_reason)``.

    ``required`` is effect-gated: the host enforces review only when the turn
    produced reviewable work (commit / deliverable / repo / workspace / skill
    write) or the task is not a direct-chat turn (queued / headless / scheduled).
    Pure conversation with no reviewable effect is not reviewed even in
    ``required``. ``auto`` stays LLM-first (the agent elects via the visible
    task_acceptance_review tool); ``off`` never reviews. This gates on observable
    runtime effects (P3 immune gate), not on message content (no P5 violation).
    """
    if mode == "off":
        return False, "off"
    if mode == "required":
        if turn_has_reviewable_effects(llm_trace):
            return True, "required_effect"
        if not is_direct_chat:
            return True, "required_nondirect"
        return False, "skipped_conversation"
    return False, "skipped_auto"


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
    mode = get_task_review_mode()
    if getattr(tools._ctx, "_task_acceptance_reviewed", False):
        return False
    is_direct_chat = bool(getattr(tools._ctx, "is_direct_chat", False))
    eligible, trigger = _task_acceptance_eligible(mode, llm_trace, is_direct_chat)
    agent_called = any(
        isinstance(c, dict) and str(c.get("tool") or "") == "task_acceptance_review"
        for c in (llm_trace.get("tool_calls") or [])
    )
    agent_review_run = any(
        isinstance(run, dict)
        and str(((run.get("request") or {}) if isinstance(run.get("request"), dict) else {}).get("surface") or "") == "task_acceptance"
        and str(run.get("aggregate_signal") or "").strip()
        for run in (llm_trace.get("review_runs") or [])
    )
    if agent_called and agent_review_run:
        tools._ctx._task_acceptance_reviewed = True
        llm_trace["review_decision"] = {"eligibility": "already_reviewed", "trigger": "agent_called_tool_result"}
        return False
    if agent_called:
        llm_trace["review_decision"] = {"eligibility": "eligible", "trigger": "agent_called_tool"}
    else:
        llm_trace["review_decision"] = {
            "eligibility": "eligible" if eligible else "not_eligible",
            "trigger": trigger,
        }
    if not eligible:
        return False
    try:
        from ouroboros.review_substrate import (
            HARDNESS_ADVISORY_VISIBLE,
            ReviewRequest,
            build_improvement_capsule,
            reviewer_slots,
            run_review_request,
        )

        from ouroboros.review_evidence import collect_turn_diff

        # A commit only "happened this turn" when it actually LANDED. A
        # REVIEW_BLOCKED / GIT_ERROR commit attempt is intentionally NOT a
        # tool-execution error (is_error=False) but carries a non-ok structured
        # status (blocked/error), so gate on the structured status, not is_error —
        # else a blocked commit would surface an unrelated prior HEAD commit as
        # this turn's evidence.
        committed_this_turn = any(
            isinstance(c, dict)
            and str(c.get("tool") or "") in ("commit_reviewed", "vcs_commit_reviewed")
            and str(c.get("status") or "") == "ok"
            for c in (llm_trace.get("tool_calls") or [])
        )
        evidence = {
            "task_id": task_id,
            "task_type": task_type,
            "tool_calls": llm_trace.get("tool_calls") or [],
            "reasoning_notes": llm_trace.get("reasoning_notes") or [],
            "repo_diff": collect_turn_diff(tools._ctx, include_recent_commit=committed_this_turn),
        }
        slots = reviewer_slots(effort=resolve_effort("review"), role_hint="task acceptance")
        min_successful = adaptive_quorum(len(slots))
        request = ReviewRequest(
            surface="task_acceptance",
            goal=_extract_plain_text_from_content(messages[1].get("content")) if len(messages) > 1 else "",
            subject=str(content or ""),
            evidence=evidence,
            checklist=(
                "Check whether the claimed result follows from the tool trace, "
                "whether errors/timeouts/artifacts were handled honestly, and "
                "whether each explicit original requirement was verified through "
                "the interface/surface the task itself names (not a weaker "
                "surrogate self-test), and "
                "whether the final response should be changed before release. "
                "Classify the deliverable tier (solved / best_effort / "
                "blocked_with_evidence) and name the single highest-value change "
                "that would move it one tier up. If the task asks for a specific "
                "value or short answer, check the FINAL ANSWER line matches the "
                "requested format exactly."
            ),
            policy={
                "verdict_is_advisory": True,
                # advisory_visible: the FULL review stays on the objective axis;
                # only a compact improvement capsule (not the raw output) is fed
                # back to the agent — so "full output does not enter context" is
                # still truthful to the reviewer.
                "full_output_enters_context": False,
                "hardness": HARDNESS_ADVISORY_VISIBLE,
                "min_successful_slots": min_successful,
                "fail_closed_on_errors": True,
                "classify_outcome_tier": True,
            },
            task_id=task_id,
        )
        result = run_review_request(
            request,
            slots=slots,
            drive_root=pathlib.Path(drive_root) if drive_root is not None else pathlib.Path(tools._ctx.drive_root),
            usage_ctx=tools._ctx,
        )
        # Record the full verdict on the objective axis (audit/status), then feed
        # the agent a COMPACT improvement capsule (not the raw review) so a real
        # best_effort/blocked_with_evidence gets ONE bounded chance to improve.
        # _task_acceptance_reviewed (set above) caps it to a single pass; the
        # capsule is anti-derailment framed ("revise only if useful, don't mention
        # the review"), which is what the old full-output re-loop lacked when it
        # tanked metrics. An empty capsule (solved / nothing actionable) finalizes.
        run_record = result.__dict__
        llm_trace.setdefault("review_runs", []).append(run_record)
        capsule = build_improvement_capsule(result)
        capsule_already_injected = bool(getattr(tools._ctx, "_task_acceptance_capsule_injected", False))
        if capsule and not capsule_already_injected:
            # ONE bounded improvement pass: inject the capsule and re-loop. Bound the
            # CAPSULE (not the review) — we do NOT set _task_acceptance_reviewed here,
            # so the REVISED final deliverable is reviewed once more and ITS verdict
            # (not the pre-revision one) lands on the objective axis. The capsule is
            # bounded to a single injection so the loop cannot derail into endless
            # re-review even if the revision does further tool work. Mark THIS
            # pre-revision run superseded so the objective reducer (which worst-cases
            # across runs) does not let the stale FAIL poison the re-reviewed verdict;
            # the run is kept in the trace for forensics.
            run_record["superseded_by_revision"] = True
            tools._ctx._task_acceptance_capsule_injected = True
            # Preserve the model's just-produced final answer in the transcript
            # before the capsule, like the sibling re-loop paths — so the revise
            # round can actually revise its OWN deliverable, not reconstruct it.
            if content and content.strip():
                messages.append({"role": "assistant", "content": content})
            _append_or_merge_user_message(messages, capsule)
            emit_progress(f"Task acceptance review: {result.aggregate_signal} — improvement note fed back.")
            return True
        # Terminal review: nothing actionable, OR the one capsule was already spent on
        # a prior pass. Record THIS (final-deliverable) verdict and finalize so the
        # objective axis reflects the shipped answer, not a stale pre-revision one.
        tools._ctx._task_acceptance_reviewed = True
        if capsule:
            emit_progress(f"Task acceptance review: {result.aggregate_signal} (improvement note already fed back; finalizing).")
        else:
            emit_progress(f"Task acceptance review: {result.aggregate_signal} (no changes suggested).")
        return False
    except Exception as exc:
        if mode == "required":
            tools._ctx._task_acceptance_reviewed = True
            safe_error = _extract_plain_text_from_content(str(exc))[:2000]
            degraded_result = {
                "request": {"surface": "task_acceptance", "task_id": task_id},
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
            }
            # Label-only: record the degraded review on the objective axis; do
            # not inject it or force another round (same non-surrender rationale).
            llm_trace.setdefault("review_runs", []).append(degraded_result)
            return False
        log.debug("Task acceptance review skipped after failure", exc_info=True)
        return False


def _adopt_fallback_route(ctx: Any, fallback_model: str, fallback_use_local: bool,
                          messages: List[Dict[str, Any]], fallback_messages: List[Dict[str, Any]]) -> tuple:
    """Round-4 C1.1: adopt a SUCCESSFUL cross-family fallback as the active route for the
    rest of the loop. Otherwise a subsequent round (esp. a tool loop) replays THIS
    fallback's reasoning/thinking back to the original primary family with no
    model-switch sanitizer firing (active_model never changed) — the cross-family
    signature replay, in reverse. Adopting the sanitized transcript as canonical means
    the switched route never carries the old family's provider-private blocks (a later
    switch_model/override re-triggers the round-start sanitizer normally). Returns the
    new ``(active_model, active_use_local)``."""
    ctx.active_model = fallback_model
    messages[:] = fallback_messages
    return fallback_model, fallback_use_local


def _run_cross_model_fallback_chain(
    *, llm, ctx, tools, messages, active_model, active_use_local, tool_schemas,
    active_effort, max_retries, drive_logs, task_id, round_idx, event_queue,
    accumulated_usage, task_type, emit_progress,
) -> tuple:
    """F1 (v6.39): 429-aware cross-model fallback CHAIN. Mark the failed primary on
    cooldown if its last failure was transient (so a swarm stops stampeding it), then walk
    the configured fallback chain, skipping cooled-down models, until one responds. Each
    candidate gets a small per-candidate attempt cap so a multi-model chain cannot multiply
    into a long retry storm; every call stays deadline-aware. The bench (FALLBACKS==main)
    dedupes to an empty chain -> no cross-model fallback, by design. Returns the new
    ``(msg, active_model, active_use_local)``; ``msg`` is None if the whole (cooled-down /
    empty) chain is exhausted, leaving the caller to join the provider-unavailable shelf."""
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
        msg, _cost = call_llm_with_retry(
            llm, fallback_messages, fallback_model, tool_schemas, active_effort,
            max_retries, drive_logs, task_id, round_idx, event_queue, accumulated_usage, task_type,
            use_local=fallback_use_local, deadline_ts=deadline, attempt_cap=attempt_cap,
        )
        if msg is not None:
            active_model, active_use_local = _adopt_fallback_route(
                ctx, fallback_model, fallback_use_local, messages, fallback_messages
            )
            break
        _cooled(fallback_model, fallback_use_local)
    return msg, active_model, active_use_local


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
    task_cost = accumulated_usage.get("cost", 0)
    checkpoint_num = round_idx // REMINDER_INTERVAL

    tool_trace = _build_recent_tool_trace(messages)

    reminder = (
        f"[CHECKPOINT {checkpoint_num} — round {round_idx}/{max_rounds}]\n"
        f"Context: ~{ctx_tokens} tokens | Cost so far: ${task_cost:.2f} | "
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
        f"~{ctx_tokens} tokens, ${task_cost:.2f} spent"
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
    """Inject deadline-awareness at 50/25/10% remaining, never per-round.

    With no deadline_at (headless/benchmark runs), fall back to intrinsic
    self-pacing: surface the agent's OWN elapsed wall-clock / rounds / cost at a
    fixed cadence so it can decide when to wrap up. Both are ADVISORY — the model
    judges when to finalize; neither is a deterministic stop gate (P5)."""
    meta = getattr(tools._ctx, "task_metadata", {})
    if not isinstance(meta, dict):
        return False
    created = parse_deadline_ts(meta.get("created_at") or meta.get("started_at"))
    if created is None:
        created = getattr(tools._ctx, "_time_budget_started_at", None)
        if created is None:
            created = utc_now()
            tools._ctx._time_budget_started_at = created
    now = utc_now()
    deadline = parse_deadline_ts(meta.get("deadline_at"))
    if deadline is None:
        return _maybe_inject_intrinsic_pacing(
            messages, tools, created=created, now=now, round_idx=round_idx,
            accumulated_usage=accumulated_usage, event_queue=event_queue,
            task_id=task_id, drive_logs=drive_logs,
        )
    total = max(1.0, (deadline - created).total_seconds())
    remaining = (deadline - now).total_seconds()
    fraction_remaining = 0.0 if remaining <= 0 else remaining / total
    thresholds = ((0.50, "50%"), (0.25, "25%"), (0.10, "10%"))
    seen = getattr(tools._ctx, "_time_budget_milestones_seen", None)
    if not isinstance(seen, set):
        seen = set()
        tools._ctx._time_budget_milestones_seen = seen
    # Fire the TIGHTEST crossed milestone, not the coarsest. Starting a task
    # already past 50% (or 25%/10%) remaining must announce the real urgency
    # immediately instead of labelling it "50%" and cascading one threshold per
    # round (which lags reality and can pass the deadline before "10%" fires).
    # Mark every crossed label seen so coarser ones never fire redundantly.
    crossed = [(value, label) for value, label in thresholds if fraction_remaining <= value]
    unseen_crossed = [(value, label) for value, label in crossed if label not in seen]
    if not unseen_crossed:
        return False
    selected_label = unseen_crossed[-1][1]  # thresholds are coarse→fine
    for _value, label in crossed:
        seen.add(label)
    elapsed = max(0.0, (now - created).total_seconds())
    remaining_clamped = max(0.0, remaining)
    deadline_text = deadline.isoformat().replace("+00:00", "Z")
    _append_or_merge_user_message(
        messages,
        (
            f"[TIME BUDGET — {selected_label} remaining crossed]\n"
            f"Elapsed: ~{elapsed/60:.1f} min | Remaining: ~{remaining_clamped/60:.1f} min | "
            f"Deadline: {deadline_text}\n"
            "Use this as planning context, not as a command to stop. If a passing artifact "
            "or service already exists, prefer preserving and verifying it over speculative "
            "improvements. If not, focus on the shortest path to a verifiable result."
        ),
    )
    _emit_checkpoint_event(event_queue, task_id, drive_logs, {
        "checkpoint_kind": "time_budget_milestone",
        "milestone": selected_label,
        "elapsed_sec": round(elapsed, 3),
        "remaining_sec": round(remaining_clamped, 3),
        "deadline_at": deadline_text,
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
    return bool(checkpoint or time_budget)


def _maybe_inject_intrinsic_pacing(
    messages: List[Dict[str, Any]],
    tools: ToolRegistry,
    *,
    created,
    now,
    round_idx: int,
    accumulated_usage: Optional[Dict[str, Any]],
    event_queue: Optional[queue.Queue],
    task_id: str,
    drive_logs: Optional[pathlib.Path],
) -> bool:
    """No deadline: surface the agent's OWN elapsed / rounds / cost periodically.

    ADVISORY only — this gives the one mind awareness so IT can choose to wrap up.
    There is deliberately no deterministic time/round/cost stop here: finalization
    is P5-named semantic behavior and stays the model's judgment."""
    interval = get_pacing_interval_sec()
    if interval <= 0:
        return False
    elapsed = max(0.0, (now - created).total_seconds())
    bucket = int(elapsed // interval)
    if bucket <= 0:
        return False
    last_bucket = getattr(tools._ctx, "_pacing_bucket_seen", 0)
    if bucket <= last_bucket:
        return False
    tools._ctx._pacing_bucket_seen = bucket
    cost = float((accumulated_usage or {}).get("cost") or 0.0)
    _append_or_merge_user_message(
        messages,
        (
            f"[PACING — ~{elapsed/60:.0f} min elapsed]\n"
            f"Rounds so far: {round_idx} | Elapsed: ~{elapsed/60:.1f} min | Cost so far: ~${cost:.2f}\n"
            "Planning context, not a command to stop. Periodically confirm you are still on the "
            "shortest path to a verifiable result; if a passing artifact or service already exists, "
            "prefer preserving and verifying it over speculative improvements."
        ),
    )
    _emit_checkpoint_event(event_queue, task_id, drive_logs, {
        "checkpoint_kind": "intrinsic_pacing",
        "elapsed_sec": round(elapsed, 3),
        "rounds": int(round_idx),
        "cost": round(cost, 4),
    })
    return True


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
                _append_or_merge_user_content(messages, _owner_marked_content(build_user_content(injected)))
            else:
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
    if remaining > float(get_finalization_grace_sec()):
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


def _no_tool_final_answer(content, limit_ctx, llm_trace, accumulated_usage):
    """Finalize a no-tool turn, appending a loud orphan note for any STILL-RUNNING child
    not absorbed/discarded (P1 — never a silent loss; P5 — discard_child_result is how the
    agent suppresses it). Completed children were already surfaced via the handoff reminder."""
    return _handle_text_response(
        (content or "") + _forced_orphan_note(limit_ctx, include_terminal=False),
        llm_trace, accumulated_usage,
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
    """CW2 (v6.34.0): enforce the max-mode contract at the point of USE. Max is kept
    only when the ACTUAL active route — remote OR local (USE_LOCAL_MAIN, a local model,
    or a per-task switch_model override) — carries confirmed >=1M Capability Evidence
    (read-only, no network on the hot path). Local routes are probed for their local
    n_ctx, NOT skipped (CW7) — a 16K local model under OUROBOROS_CONTEXT_MODE=max must
    still fall back to low. Fail-closed to low on any unconfirmed/unprobeable route or
    probe error (BIBLE P1 cognitive-horizon). Composes with the reactive provider-
    overflow fallback; this is the preflight gate (settings-save only checks at write)."""
    if mode != "max":
        return mode
    try:
        from ouroboros.gateway.settings import _active_route_confirms_max
        if not _active_route_confirms_max(model=model, use_local=use_local, allow_fetch=allow_fetch):
            log.info(
                "Max context mode is not confirmed >=1M for the active route "
                "(use_local=%s); using low-mode compaction for this task (fail-closed, CW2).",
                use_local,
            )
            return "low"
    except Exception:
        log.debug("CW2 max-mode capability check failed; fail-closed to low", exc_info=True)
        return "low"
    return mode


def _apply_overrides_and_regate_mode(ctx, active_model, active_use_local, active_effort, active_context_mode):
    """Apply per-round runtime overrides, then re-gate max-mode at point-of-use if the
    active route changed (a mid-loop switch_model / local-route change — the start-of-
    loop gate only saw the initial route). Fail-closed to low (CW2)."""
    _route_before = (active_model, active_use_local)
    active_model, active_use_local, active_effort = _apply_runtime_overrides(
        ctx, active_model, active_use_local, active_effort,
    )
    if (active_model, active_use_local) != _route_before:
        active_context_mode = _maybe_downgrade_max_unconfirmed(
            get_context_mode(), active_use_local, active_model,
        )
    return active_model, active_use_local, active_effort, active_context_mode


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
    task_model_override = str(getattr(ctx, "task_model_override", "") or "").strip()
    active_model = task_model_override or llm.default_model()
    active_effort = initial_effort
    if getattr(ctx, "task_use_local_override", None) is not None:
        active_use_local = bool(ctx.task_use_local_override)
    else:
        active_use_local = os.environ.get("USE_LOCAL_MAIN", "").lower() in ("true", "1")
    # CW2: max-mode enforced at point-of-USE — fail-closed to low if the active route (incl. local n_ctx) no longer confirms >=1M (not just at settings-save); low-mode also compacts sooner.
    # H (v6.39): the start-of-loop gate does a LAZY probe-on-first-use (allow_fetch=True,
    # once per task) so a genuine >=1M route is actually confirmed even when max is the
    # default and the owner never toggled Low->Max; the per-round re-gate stays read-only.
    # Single-flight: ONLY a root/non-subagent task fires the network probe — subagents
    # stay read-only and share the parent's warm global Capability-Evidence store, so a
    # swarm cannot stampede the route's /models endpoint (the root probes first).
    _ctx_meta = getattr(ctx, "task_metadata", {})
    _is_subagent = (
        isinstance(_ctx_meta, dict)
        and str(_ctx_meta.get("delegation_role") or "").strip().lower() == "subagent"
    )
    _preferred_context_mode = get_context_mode()
    active_context_mode = _maybe_downgrade_max_unconfirmed(
        _preferred_context_mode, active_use_local, active_model, allow_fetch=not _is_subagent,
    )
    if _preferred_context_mode == "max" and active_context_mode != "max":
        # Observable effective-vs-preferred: the downgrade is no longer a silent log
        # line. Keep type=task_checkpoint (+ checkpoint_kind) so it is BOTH broadcast
        # live AND durably persisted to events.jsonl (the durable append is gated on
        # type==task_checkpoint), matching every other checkpoint emitter.
        _emit_checkpoint_event(event_queue, task_id, drive_logs, {
            "checkpoint_kind": "context_mode_downgraded",
            "preferred_mode": _preferred_context_mode,
            "effective_mode": active_context_mode,
            "model": active_model,
            "use_local": active_use_local,
            "reason": "route_unconfirmed_ge_1m",
        })

    llm_trace: Dict[str, Any] = {"reasoning_notes": [], "tool_calls": []}
    accumulated_usage: Dict[str, Any] = {}
    max_retries = 3
    from ouroboros.tools import tool_discovery as _td
    _td.set_registry(tools)

    tool_schemas = initial_tool_schemas(tools)
    tool_schemas, _enabled_extra_tools = _setup_dynamic_tools(tools, tool_schemas, messages)

    tools._ctx.event_queue = event_queue
    tools._ctx.task_id = task_id
    tools._ctx.messages = messages
    stateful_executor = StatefulToolExecutor()
    _owner_msg_seen: set = set()
    from ouroboros.config import SETTINGS_DEFAULTS as _DEFAULTS
    _max_rounds_default = int(_DEFAULTS["OUROBOROS_MAX_ROUNDS"])
    try:
        MAX_ROUNDS = max(1, int(os.environ.get("OUROBOROS_MAX_ROUNDS", str(_max_rounds_default))))
    except (ValueError, TypeError):
        MAX_ROUNDS = _max_rounds_default
        log.warning("Invalid OUROBOROS_MAX_ROUNDS, defaulting to %s", _max_rounds_default)
    round_idx = 0
    try:
        while True:
            round_idx += 1

            ctx = tools._ctx
            _prev_active_model = active_model
            active_model, active_use_local, active_effort, active_context_mode = _apply_overrides_and_regate_mode(
                ctx, active_model, active_use_local, active_effort, active_context_mode,
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

            _controls = _drain_incoming_messages(messages, incoming_messages, drive_root, task_id, event_queue, _owner_msg_seen)
            # Early-exit per round: supervisor finalize_now, else loop-local real-
            # deadline finalize (headless runs that get no finalize_now) — finalize
            # best-effort rather than be killed mid-step with nothing.
            _early_final = _maybe_early_finalize(limit_ctx, tools, _controls)
            if _early_final is not None:
                text, accumulated_usage, _ = _early_final
                return text, accumulated_usage, llm_trace

            _checkpoint_injected = _inject_round_checkpoints(
                round_idx=round_idx, max_rounds=MAX_ROUNDS, messages=messages,
                accumulated_usage=accumulated_usage, emit_progress=emit_progress, tools=tools,
                event_queue=event_queue, task_id=task_id, drive_logs=drive_logs,
            )

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
                _cc = float(_compaction_usage.get("cost") or 0) or estimate_cost(
                    _cm, int(_compaction_usage.get("prompt_tokens") or 0),
                    int(_compaction_usage.get("completion_tokens") or 0),
                    int(_compaction_usage.get("cached_tokens") or 0),
                    int(_compaction_usage.get("cache_write_tokens") or 0),
                    _compaction_usage.get("prompt_cache_ttl"))
                emit_llm_usage_event(event_queue, task_id, _cm, _compaction_usage, _cc, "compaction")

            # Provider cache boundary; unsupported providers strip cache_control in llm.py.
            seal_task_transcript(messages)

            msg, cost = call_llm_with_retry(
                llm, messages, active_model, tool_schemas, active_effort,
                max_retries, drive_logs, task_id, round_idx, event_queue, accumulated_usage, task_type,
                use_local=active_use_local,
                deadline_ts=_task_deadline_epoch(tools),
            )
            tools._ctx._current_llm_call_meta = dict(accumulated_usage.get("_last_llm_call_meta") or {})

            if msg is None:
                msg, active_model, active_use_local = _run_cross_model_fallback_chain(
                    llm=llm, ctx=ctx, tools=tools, messages=messages, active_model=active_model,
                    active_use_local=active_use_local, tool_schemas=tool_schemas, active_effort=active_effort,
                    max_retries=max_retries, drive_logs=drive_logs, task_id=task_id, round_idx=round_idx,
                    event_queue=event_queue, accumulated_usage=accumulated_usage, task_type=task_type,
                    emit_progress=emit_progress)
                if msg is None:
                    # Provider-death: join the unified honest best-effort shelf
                    # (deadline/budget/round-limit) instead of discarding useful
                    # workspace state with a bare error string.
                    text, accumulated_usage, _ = _handle_provider_unavailable(limit_ctx)
                    return text, accumulated_usage, llm_trace

            tool_calls = msg.get("tool_calls") or []
            content = msg.get("content")
            if not tool_calls:
                if _force_plan_required(tools._ctx, llm_trace):
                    attempts = int(getattr(tools._ctx, "_force_plan_reminder_count", 0) or 0)
                    if attempts >= 2:
                        accumulated_usage["execution_status"] = "failed"
                        accumulated_usage["reason_code"] = "swarm_force_plan_not_called"
                        return (
                            "⚠️ SWARM_INITIATIVE_BLOCKED: plan_task was required for this swarm task but was not called.",
                            accumulated_usage,
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
                    continue
                handoff_msg = _compute_subagent_handoff(tools, drive_root, task_id, content)
                if handoff_msg:
                    if content and content.strip():
                        messages.append({"role": "assistant", "content": content})
                    _append_or_merge_user_message(messages, f"[SYSTEM REMINDER]\n{handoff_msg}")
                    emit_progress("Subagent handoff status refreshed before final response.")
                    llm_trace["reasoning_notes"].append("Subagent handoff status refreshed before final response.")
                    continue
                finalization_msg = _skill_finalization_message(drive_root, llm_trace) if drive_root is not None else ""
                if finalization_msg and not getattr(tools._ctx, "_skill_finalization_injected", False):
                    tools._ctx._skill_finalization_injected = True
                    if content and content.strip():
                        messages.append({"role": "assistant", "content": content})
                    _append_or_merge_user_message(messages, f"[SYSTEM REMINDER]\n{finalization_msg}")
                    emit_progress(finalization_msg)
                    llm_trace["reasoning_notes"].append(finalization_msg)
                    continue
                if _run_task_acceptance_review_once(
                    tools=tools,
                    content=content or "",
                    task_id=task_id,
                    task_type=task_type,
                    llm_trace=llm_trace,
                    drive_root=drive_root,
                    messages=messages,
                    emit_progress=emit_progress,
                ):
                    continue
                return _no_tool_final_answer(content, limit_ctx, llm_trace, accumulated_usage)

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
                deadline_ts=_task_deadline_epoch(tools),
            )
            if budget_result is not None:
                return budget_result

    finally:
        if stateful_executor:
            try:
                from ouroboros.tools.browser import cleanup_browser
                stateful_executor.submit(cleanup_browser, tools._ctx).result(timeout=5)
            except Exception:
                log.debug("Browser cleanup on executor thread failed or timed out", exc_info=True)
            try:
                stateful_executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                log.warning("Failed to shutdown stateful executor", exc_info=True)
        if drive_root is not None and task_id:
            try:
                from ouroboros.tools.services import stop_task_services

                finalized_services = stop_task_services(tools._ctx)
                stopped_services = [s for s in finalized_services if s.get("lifecycle") != "kept"]
                kept_services = [s for s in finalized_services if s.get("lifecycle") == "kept"]
                if stopped_services:
                    _emit_checkpoint_event(event_queue, task_id, drive_logs, {
                        "checkpoint_kind": "services_stopped",
                        "services": stopped_services,
                    })
                    llm_trace.setdefault("verification_events", []).append({
                        "kind": "services_stopped",
                        "services": stopped_services,
                    })
                if kept_services:
                    # Survivors are deliberate (keep_alive / service_teardown=keep):
                    # record pid/port metadata so the external party that asked for
                    # them (verifier, owner) knows what it now owns.
                    _emit_checkpoint_event(event_queue, task_id, drive_logs, {
                        "checkpoint_kind": "services_kept",
                        "services": kept_services,
                    })
                    llm_trace.setdefault("verification_events", []).append({
                        "kind": "services_kept",
                        "services": kept_services,
                    })
            except Exception:
                log.debug("Failed to stop task services", exc_info=True)
            try:
                from ouroboros.owner_mailbox import cleanup_task_mailbox
                cleanup_task_mailbox(drive_root, task_id)
            except Exception:
                log.debug("Failed to cleanup task mailbox", exc_info=True)
