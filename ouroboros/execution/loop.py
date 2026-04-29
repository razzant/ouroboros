"""
Ouroboros — LLM tool loop (Orchestrator).

Core loop: send messages to LLM, execute tool calls, repeat until final response.
"""
from __future__ import annotations

import json
import logging
import os
import pathlib
import queue
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from ouroboros.llm import LLMClient, normalize_reasoning_effort, add_usage
from ouroboros.tools.registry import ToolRegistry
from ouroboros.context import compact_tool_history, compact_tool_history_llm
from ouroboros.utils import utc_now_iso, append_jsonl
from ouroboros.execution import (
    handle_tool_calls,
    handle_text_response,
    StatefulToolExecutor,
    READ_ONLY_PARALLEL_TOOLS,
    STATEFUL_BROWSER_TOOLS,
)

log = logging.getLogger(__name__)

# Context compaction thresholds (tokens) — prevent OOM on local LLM
CONTEXT_COMPACT_SOFT = 30000   # Compact at 30K estimated tokens
CONTEXT_COMPACT_HARD = 50000   # Force compact at 50K


def _maybe_inject_self_check(
    round_idx: int,
    max_rounds: int,
    messages: List[Dict[str, Any]],
    accumulated_usage: Dict[str, Any],
    emit_progress: Callable[[str], None],
) -> None:
    """Bible P5: soft self-check reminder every 50 rounds."""
    if round_idx > 0 and round_idx % 50 == 0:
        remaining = max_rounds - round_idx
        used_pct = round(100 * round_idx / max_rounds, 1)
        cost = round(float(accumulated_usage.get("cost") or 0), 4)
        msg = f"[Self-check] Round {round_idx}/{max_rounds} ({used_pct}% used, {remaining} left, cost: ${cost})"
        emit_progress(msg)
        messages.append({
            "role": "system",
            "content": (
                f"## Self-Check Reminder (Bible P5)\n\n"
                f"You are at round {round_idx}/{max_rounds} ({used_pct}% used). "
                f"Remaining budget: {remaining} rounds. "
                f"Total cost so far: ${cost}.\n\n"
                f"Before continuing:\n"
                f"1. Is your next action necessary and likely to advance the task?\n"
                f"2. Can you complete the task with fewer remaining rounds?\n"
                f"3. Should you finalize now with a summary?\n\n"
                f"Be efficient. Prioritize high-impact actions."
            ),
        })


def _setup_dynamic_tools(tools_registry, tool_schemas, messages):
    """Add meta-tools for tool discovery (Bible P4)."""
    enabled_extra_tools = set()
    # Add tool_discovery if the agent has shown need for it
    has_tool_ref = any(
        isinstance(m.get("content"), str) and "tool_ref" in m.get("content", "")
        for m in messages[-10:] if isinstance(m, dict)
    )
    if has_tool_ref:
        tool_schemas = tools_registry.schemas(core_only=False)
        enabled_extra_tools = {"tool_discovery", "tool_install"}
    return tool_schemas, enabled_extra_tools


def _drain_incoming_messages(
    messages: List[Dict[str, Any]],
    incoming_messages: queue.Queue,
    drive_root: Optional[pathlib.Path],
    task_id: str,
    event_queue: Optional[queue.Queue],
    _owner_msg_seen: set,
) -> None:
    """Drive mailbox + in-process queue → messages."""
    if drive_root is not None and task_id:
        try:
            from ouroboros.owner_inject import collect_owner_messages
            drive_msgs = collect_owner_messages(drive_root, task_id)
            for dm in drive_msgs:
                mid = dm.get("msg_id") or f"{dm.get('seq', 0)}"
                if mid in _owner_msg_seen:
                    continue
                _owner_msg_seen.add(mid)
                messages.append({
                    "role": "system",
                    "content": f"## Incoming Message (from {dm.get('source', 'owner')})\n\n{dm.get('text', '')}",
                })
                if event_queue is not None:
                    try:
                        event_queue.put_nowait({
                            "type": "task_progress",
                            "task_id": task_id,
                            "text": dm.get("text", "")[:200],
                        })
                    except Exception:
                        pass
        except Exception:
            log.debug("Failed to collect owner messages", exc_info=True)

    while not incoming_messages.empty():
        try:
            msg = incoming_messages.get_nowait()
            if isinstance(msg, dict) and msg.get("type") == "task_progress":
                messages.append({
                    "role": "system",
                    "content": f"## Incoming Message\n\n{msg.get('text', '')}",
                })
                if event_queue is not None:
                    try:
                        event_queue.put_nowait({
                            "type": "task_progress",
                            "task_id": task_id,
                            "text": msg.get("text", "")[:200],
                        })
                    except Exception:
                        pass
        except queue.Empty:
            break


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
    """
    Core LLM-with-tools loop.

    Sends messages to LLM, executes tool calls, retries on errors.
    LLM controls model/effort via switch_model tool (LLM-first, Bible P3).

    Args:
        budget_remaining_usd: If set, forces completion when task cost exceeds 50% of this budget
        initial_effort: Initial reasoning effort level (default "medium")

    Returns: (final_text, accumulated_usage, llm_trace)
    """
    # LLM-first: single default model, LLM switches via tool if needed
    active_model = llm.default_model()
    active_effort = initial_effort

    llm_trace: Dict[str, Any] = {"assistant_notes": [], "tool_calls": []}
    accumulated_usage: Dict[str, Any] = {}
    max_retries = 3
    # Wire module-level registry ref so tool_discovery handlers work outside run_llm_loop too
    from ouroboros.tools import tool_discovery as _td
    _td.set_registry(tools)

    # Selective tool schemas: core set + meta-tools for discovery.
    tool_schemas = tools.schemas(core_only=True)
    tool_schemas, _enabled_extra_tools = _setup_dynamic_tools(tools, tool_schemas, messages)

    # Set budget tracking on tool context for real-time usage events
    tools._ctx.event_queue = event_queue
    tools._ctx.task_id = task_id
    # Thread-sticky executor for browser tools (Playwright sync requires greenlet thread-affinity)
    stateful_executor = StatefulToolExecutor()
    # Dedup set for per-task owner messages from Drive mailbox
    _owner_msg_seen: set = set()
    try:
        MAX_ROUNDS = max(1, int(os.environ.get("OUROBOROS_MAX_ROUNDS", "200")))
    except (ValueError, TypeError):
        MAX_ROUNDS = 200
        log.warning("Invalid OUROBOROS_MAX_ROUNDS, defaulting to 200")
    round_idx = 0
    try:
        while True:
            round_idx += 1

            # Hard limit on rounds to prevent runaway tasks
            if round_idx > MAX_ROUNDS:
                finish_reason = f"⚠️ Task exceeded MAX_ROUNDS ({MAX_ROUNDS}). Consider decomposing into subtasks via schedule_task."
                messages.append({"role": "system", "content": f"[ROUND_LIMIT] {finish_reason}"})
                try:
                    final_msg, final_cost = _call_llm_with_retry(
                        llm, messages, active_model, None, active_effort,
                        max_retries, drive_logs, task_id, round_idx, accumulated_usage, task_type
                    )
                    if final_msg:
                        return (final_msg.get("content") or finish_reason), accumulated_usage, llm_trace
                    return finish_reason, accumulated_usage, llm_trace
                except Exception:
                    log.warning("Failed to get final response after round limit", exc_info=True)
                    return finish_reason, accumulated_usage, llm_trace

            # Soft self-check reminder every 50 rounds (LLM-first: agent decides, not code)
            _maybe_inject_self_check(round_idx, MAX_ROUNDS, messages, accumulated_usage, emit_progress)

            # Apply LLM-driven model/effort switch (via switch_model tool)
            ctx = tools._ctx
            if ctx.active_model_override:
                active_model = ctx.active_model_override
                ctx.active_model_override = None
            if ctx.active_effort_override:
                active_effort = normalize_reasoning_effort(ctx.active_effort_override, default=active_effort)
                ctx.active_effort_override = None

            # Inject owner messages (in-process queue + Drive mailbox)
            _drain_incoming_messages(messages, incoming_messages, drive_root, task_id, event_queue, _owner_msg_seen)

            # Compact old tool history when needed
            # Check for LLM-requested compaction first (via compact_context tool)
            pending_compaction = getattr(tools._ctx, '_pending_compaction', None)
            if pending_compaction is not None:
                messages = compact_tool_history_llm(messages, keep_recent=pending_compaction)
                tools._ctx._pending_compaction = None
            else:
                # Token-aware compaction: compact when context exceeds safe limits
                # Estimate token count (rough: ~4 chars per token for typical messages)
                estimated_tokens = sum(
                    len(str(m.get("content", ""))) // 4 +
                    len(str(m.get("tool_calls", []))) * 100
                    for m in messages
                )
                
                # Aggressive compaction for local mode to prevent OOM
                if estimated_tokens > CONTEXT_COMPACT_HARD:
                    messages = compact_tool_history(messages, keep_recent=10)
                elif estimated_tokens > CONTEXT_COMPACT_SOFT:
                    messages = compact_tool_history(messages, keep_recent=8)
                elif round_idx > 5:
                    messages = compact_tool_history(messages, keep_recent=6)
                elif round_idx > 3:
                    # Light compaction: only if messages list is very long (>60 items)
                    if len(messages) > 60:
                        messages = compact_tool_history(messages, keep_recent=6)

            # --- LLM call with retry ---
            msg, cost = _call_llm_with_retry(
                llm, messages, active_model, tool_schemas, active_effort,
                max_retries, drive_logs, task_id, round_idx, accumulated_usage, task_type
            )

            # Fallback to another model if primary model returns empty responses
            if msg is None:
                # Check if we're in local mode (no cloud API)
                base_url = os.environ.get("OUROBOROS_BASE_URL", "")
                is_local_mode = base_url and "openrouter.ai" not in base_url
                
                if is_local_mode:
                    # In local mode, don't try cloud fallback — just report failure
                    return (
                        f"⚠️ Failed to get a response from model {active_model} after {max_retries} attempts. "
                        f"Local mode: no cloud fallback available. Check your local LLM server."
                    ), accumulated_usage, llm_trace
                
                # Configurable fallback priority list (Bible P3: no hardcoded behavior)
                fallback_list_raw = os.environ.get(
                    "OUROBOROS_MODEL_FALLBACK_LIST",
                    "google/gemini-2.5-pro-preview,openai/o3,anthropic/claude-sonnet-4.6"
                )
                fallback_candidates = [m.strip() for m in fallback_list_raw.split(",") if m.strip()]
                fallback_model = None
                for candidate in fallback_candidates:
                    if candidate != active_model:
                        fallback_model = candidate
                        break
                if fallback_model is None:
                    return (
                        f"⚠️ Failed to get a response from the model after {max_retries} attempts. "
                        f"All fallback models match the active one. Try rephrasing your request."
                    ), accumulated_usage, llm_trace

                # Emit progress message so user sees fallback happening
                fallback_progress = f"⚡ Fallback: {active_model} → {fallback_model} after empty response"
                emit_progress(fallback_progress)

                # Try fallback model (don't increment round_idx — this is still same logical round)
                msg, fallback_cost = _call_llm_with_retry(
                    llm, messages, fallback_model, tool_schemas, active_effort,
                    max_retries, drive_logs, task_id, round_idx, accumulated_usage, task_type
                )

                # If fallback also fails, give up
                if msg is None:
                    return (
                        f"⚠️ Failed to get a response from the model after {max_retries} attempts. "
                        f"Fallback model ({fallback_model}) also returned no response."
                    ), accumulated_usage, llm_trace

                # Fallback succeeded — continue processing with this msg
                # (don't return — fall through to tool_calls processing below)

            tool_calls = msg.get("tool_calls") or []
            content = msg.get("content")
            # No tool calls — final response
            if not tool_calls:
                return _handle_text_response(content, llm_trace, accumulated_usage)

            # Process tool calls
            messages.append({"role": "assistant", "content": content or "", "tool_calls": tool_calls})

            if content and content.strip():
                emit_progress(content.strip())
                llm_trace["assistant_notes"].append(content.strip()[:320])

            error_count = handle_tool_calls(
                tool_calls, tools, drive_logs, task_id, stateful_executor,
                messages, llm_trace, emit_progress
            )

    finally:
        # Cleanup thread-sticky executor for stateful tools
        if stateful_executor:
            try:
                stateful_executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                log.warning("Failed to shutdown stateful executor", exc_info=True)
        # Cleanup per-task mailbox
        if drive_root is not None and task_id:
            try:
                from ouroboros.owner_inject import cleanup_task_mailbox
                cleanup_task_mailbox(drive_root, task_id)
            except Exception:
                log.debug("Failed to cleanup task mailbox", exc_info=True)


def _call_llm_with_retry(
    llm: LLMClient,
    messages: List[Dict[str, Any]],
    model: str,
    tools: Optional[List[Dict[str, Any]]],
    effort: str,
    max_retries: int,
    drive_logs: pathlib.Path,
    task_id: str,
    round_idx: int,
    accumulated_usage: Dict[str, Any],
    task_type: str = "",
) -> Tuple[Optional[Dict[str, Any]], float]:
    """
    Call LLM with retry logic, usage tracking.

    Returns:
        (response_message, cost) on success
        (None, 0.0) on failure after max_retries
    """
    msg = None

    for attempt in range(max_retries):
        try:
            kwargs = {"messages": messages, "model": model, "reasoning_effort": effort}
            if tools:
                kwargs["tools"] = tools
            resp_msg, usage = llm.chat(**kwargs)
            msg = resp_msg
            add_usage(accumulated_usage, usage)

            # Calculate cost (local mode: cost is always 0)
            cost = float(usage.get("cost") or 0)

            # Empty response = retry-worthy (model sometimes returns empty content with no tool_calls)
            tool_calls = msg.get("tool_calls") or []
            content = msg.get("content")
            if not tool_calls and (not content or not content.strip()):
                log.warning("LLM returned empty response (no content, no tool_calls), attempt %d/%d", attempt + 1, max_retries)

                # Log raw empty response for debugging
                append_jsonl(drive_logs / "events.jsonl", {
                    "ts": utc_now_iso(), "type": "llm_empty_response",
                    "task_id": task_id,
                    "round": round_idx, "attempt": attempt + 1,
                    "model": model,
                    "raw_content": repr(content)[:500] if content else None,
                    "raw_tool_calls": repr(tool_calls)[:500] if tool_calls else None,
                    "finish_reason": msg.get("finish_reason") or msg.get("stop_reason"),
                })

                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                # Last attempt — return None to trigger "could not get response"
                return None, cost

            # Count only successful rounds
            accumulated_usage["rounds"] = accumulated_usage.get("rounds", 0) + 1

            # Log per-round metrics
            _round_event = {
                "ts": utc_now_iso(), "type": "llm_round",
                "task_id": task_id,
                "round": round_idx, "model": model,
                "reasoning_effort": effort,
                "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                "completion_tokens": int(usage.get("completion_tokens") or 0),
                "cached_tokens": int(usage.get("cached_tokens") or 0),
                "cache_write_tokens": int(usage.get("cache_write_tokens") or 0),
                "cost_usd": cost,
            }
            append_jsonl(drive_logs / "events.jsonl", _round_event)
            return msg, cost

        except Exception as e:
            append_jsonl(drive_logs / "events.jsonl", {
                "ts": utc_now_iso(), "type": "llm_api_error",
                "task_id": task_id,
                "round": round_idx, "attempt": attempt + 1,
                "model": model, "error": repr(e),
            })
            if attempt < max_retries - 1:
                time.sleep(min(2 ** attempt * 2, 30))

    return None, 0.0
