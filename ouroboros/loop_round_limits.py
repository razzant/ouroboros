"""Round-limit and terminal-drain handling for the main loop: owner-stop drain
and its window, incoming-message drain, round compaction and its usage
accounting, the round-limit context, and the limit/forced/owner-stop/
provider-unavailable/deadline handlers. Extracted from loop.py (v7 L-B split);
loop.py re-exports every name."""

from __future__ import annotations

import functools
import pathlib
import queue

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
from ouroboros import task_pacing
from ouroboros.config import get_light_model
from ouroboros.context import build_user_content
from ouroboros.deadline_utils import parse_deadline_ts
from ouroboros.llm import LLMClient, add_usage
from ouroboros.loop_llm_call import emit_llm_usage_event
from ouroboros.loop_tool_execution import prune_reclaim_trace_refs, reclaim_negative_memo, reclaim_trace_refs
from ouroboros.loop_transport import TransportWaitEpisode, finalize_now_transport_terminal as _finalize_now_transport_terminal
from ouroboros.outcomes import REASON_OWNER_REQUESTED_FINALIZATION
from ouroboros.pricing import estimate_cost_optional
from ouroboros.task_finalization import TERMINAL_ORIGIN_HOST_NOTICE, TERMINAL_ORIGIN_HOST_SALVAGE
from ouroboros.tools.registry import ToolRegistry
from ouroboros.usage_accounting import invalidate_task_cache_splits
from supervisor.owner_stop import REASON_OWNER_STOPPED_DIRECT_TURN, _narrow_round_deadline, _owner_stop_control_is_current, _owner_stop_window_elapsed, handle_finalize_now_entry  # noqa: F401 -- _owner_stop_control_is_current stays a facade surface

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # annotation-only names; lazy under future annotations, never imported at runtime
    from ouroboros.loop_delivery import DeliveryCandidate


def _loop():
    """The parent loop module, read at call time.

    The loop's members stay monkeypatch-addressable at their historical
    ``ouroboros.loop`` bindings (tests rebind them there), so this leaf
    resolves every cross-reference through the module at each call instead
    of freezing whatever object a from-import saw at import time.
    """
    from ouroboros import loop

    return loop


@dataclass
class _CompactionRoundContext:
    tools: ToolRegistry
    drive_root: Optional[pathlib.Path]
    drive_logs: pathlib.Path
    task_id: str
    round_idx: int
    event_queue: Optional[queue.Queue]
    emit_progress: Callable[[str], None]


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
                _loop()._record_owner_directive(
                    owner_ctx,
                    source="direct_incoming",
                    content=owner_content,
                    msg_id=str(
                        injected.get("client_message_id")
                        or injected.get("msg_id")
                        or ""
                    ),
                )
                _loop()._append_or_merge_user_content(messages, _loop()._owner_marked_content(owner_content))
            else:
                _loop()._record_owner_directive(
                    owner_ctx, source="direct_incoming", content=injected,
                )
                _loop()._append_or_merge_user_message(messages, _loop()._owner_marked_content(injected))
        except queue.Empty:
            break

    if drive_root is not None and task_id:
        from ouroboros.owner_mailbox import KIND_FINALIZE_NOW, KIND_HURRY, KIND_OWNER_TEXT, KIND_QUIZ_ANSWER, KIND_TASK_MESSAGE, acknowledge_transcript_entry, deliver_quiz_answer, deliver_task_message, drain_owner_entries

        if owner_ctx:
            owner_ctx._loop_mailbox_seen_ids = _owner_msg_seen
        attempt = getattr(owner_ctx, "task_attempt", None) or (1 if owner_ctx is not None else None)
        for entry in drain_owner_entries(drive_root, task_id, _owner_msg_seen, attempt):
            kind = entry.get("kind") or KIND_OWNER_TEXT
            if kind == KIND_FINALIZE_NOW:
                handle_finalize_now_entry(entry, owner_ctx, drive_root, task_id, controls)
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
                deliver_task_message(entry, task_id, event_queue, lambda text: _loop()._append_or_merge_user_message(messages, text))
                acknowledge_transcript_entry(drive_root, task_id, entry)
                continue
            if kind == KIND_QUIZ_ANSWER:
                deliver_quiz_answer(entry, task_id, event_queue, lambda text: _loop()._append_or_merge_user_message(messages, text))
                acknowledge_transcript_entry(drive_root, task_id, entry)
                continue
            _loop()._record_owner_directive(
                owner_ctx,
                source="owner_mailbox",
                content=dmsg,
                msg_id=str(entry.get("msg_id") or ""),
            )
            from ouroboros.client_surface import noted_owner_text

            _loop()._append_or_merge_user_message(messages, _loop()._owner_marked_content(noted_owner_text(owner_ctx, entry, dmsg)))
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
    rebuilt, receipt, usage = _loop().compact_tool_history_llm(
        messages,
        keep_recent=max(0, int(pending)),
        drive_root=ctx.drive_root or pathlib.Path(ctx.drive_logs).parent,
        task_id=ctx.task_id,
        negative_memo=reclaim_negative_memo(ctx.tools._ctx),
        trace_refs_by_tool_call_id=reclaim_trace_refs(ctx.tools._ctx),
    )
    _loop()._emit_checkpoint_event(ctx.event_queue, ctx.task_id, ctx.drive_logs, {
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
        invalidate_task_cache_splits(ctx.task_id)
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
    # The round's exact tool envelope, so a forced wrap-up call keeps the same
    # provider prefix as the working round instead of rebuilding it tool-less.
    # LAST field: existing positional construction (tests included) stays valid.
    tool_schemas: Optional[List[Dict[str, Any]]] = None


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
    return _loop()._forced_final_answer(ctx, prompt=prompt, fallback_text=finish_reason, reason_code="round_limit")


def _handle_forced_finalization(ctx: _RoundLimitContext, reason: str) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """Finalize within the cooperative grace window."""
    reason_lines = str(reason or "").splitlines()
    if reason_lines and reason_lines[0].strip() == REASON_OWNER_REQUESTED_FINALIZATION:
        return _handle_owner_stop_finalization(ctx, str(reason))
    if reason_lines and reason_lines[0].strip() == REASON_OWNER_STOPPED_DIRECT_TURN:
        return _handle_direct_turn_hard_stop(ctx)
    fallback = f"⚠️ Task reached {reason or 'deadline'}; finalization grace produced no answer."
    prompt = (
        f"[FINALIZE_NOW] The supervisor opened a finalization grace window (reason: {reason or 'deadline'}). "
        "The task will be stopped shortly. Produce your best final answer NOW from the verified "
        "work so far; clearly mark anything unverified or incomplete. An honest best-effort "
        "result is the expected outcome here, not a failure."
    )
    return _loop()._forced_final_answer(ctx, prompt=prompt, fallback_text=fallback, reason_code="finalization_grace")


def _handle_direct_turn_hard_stop(
    ctx: _RoundLimitContext,
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """"Stop now" on an in-process direct-chat turn: ZERO further model calls.

    A pooled task's immediate stop kills its worker; the direct turn has no
    process of its own, so custody writes this control and the turn ends at
    its next round boundary with whatever delivery candidate it already
    holds, else the typed fallback — never a paid final turn (that is the
    graceful "Wrap up" contract, ``_handle_owner_stop_finalization``).

    The contract reaches past the loop: the killed pooled worker never runs
    its post-task synthesis, so the stopped direct turn must not either. The
    existing ``_skip_post_task_synthesis`` marker (the authority-refusal
    terminal's seam, honoured by ``post_task_checkpoint.is_root_post_task``)
    is recorded on the tool context here; ``emit_task_results`` copies it
    onto the task record before the root predicate runs, so no summary,
    reflection or consolidation bills after the owner's stop."""
    live_trace = getattr(ctx, "llm_trace", None)
    llm_trace = live_trace if isinstance(live_trace, dict) else {}
    tool_ctx = getattr(getattr(ctx, "tools", None), "_ctx", None)
    if tool_ctx is not None:
        tool_ctx._skip_post_task_synthesis = True
    _loop()._finalize_forced_services(ctx, llm_trace)
    ctx.accumulated_usage["execution_status"] = "failed"
    ctx.accumulated_usage["reason_code"] = REASON_OWNER_REQUESTED_FINALIZATION
    candidate = _loop()._current_delivery_candidate(ctx, llm_trace)
    if candidate is not None:
        return _loop()._forced_fallback_result(
            ctx, llm_trace, candidate.full_text, REASON_OWNER_REQUESTED_FINALIZATION,
            retained_source="owner_stop_retained_candidate",
        )
    return _loop()._forced_fallback_result(
        ctx, llm_trace,
        "⏹ The owner stopped this chat turn; no further work was done.",
        REASON_OWNER_REQUESTED_FINALIZATION,
        source="owner_stopped_direct_turn",
    )


def _handle_owner_stop_finalization(
    ctx: _RoundLimitContext, control_text: str,
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """Owner-requested finalization: zero or one tool-less model turn."""
    live_trace = getattr(ctx, "llm_trace", None)
    llm_trace = live_trace if isinstance(live_trace, dict) else {}
    candidate = _loop()._current_delivery_candidate(ctx, llm_trace)
    if candidate is not None:
        _loop()._finalize_forced_services(ctx, llm_trace)
        ctx.accumulated_usage["execution_status"] = "failed"
        ctx.accumulated_usage["reason_code"] = REASON_OWNER_REQUESTED_FINALIZATION
        return _loop()._forced_fallback_result(
            ctx, llm_trace, candidate.full_text, REASON_OWNER_REQUESTED_FINALIZATION,
            retained_source="owner_stop_retained_candidate",
        )
    fallback = (
        "⚠️ The owner requested finalize-then-stop; no final answer could be "
        "produced inside the grace window."
    )
    if _owner_stop_window_elapsed(ctx):
        _loop()._finalize_forced_services(ctx, llm_trace)
        ctx.accumulated_usage["execution_status"] = "failed"
        ctx.accumulated_usage["reason_code"] = REASON_OWNER_REQUESTED_FINALIZATION
        return _loop()._forced_fallback_result(
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
    return _loop()._forced_final_answer(
        ctx, prompt=prompt, fallback_text=fallback,
        reason_code="owner_requested_finalization", single_semantic_turn=True,
    )


def _handle_provider_unavailable(
    ctx: _RoundLimitContext, *, error_kind: str = "provider_unavailable",
    wait_cause: str = "", waited_sec: float = 0.0, interactive: bool = False,
    control_reason: Optional[str] = None,
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """Provider-death rail wrapper: every arm carries terminal provenance.
    The forced-finalization sink stamps ``host_notice``; retained/generated
    model candidates stamp ``model_final``. Read the recognized control here
    so ordinary and delegated-hold terminal paths share the same cause."""
    if control_reason is None:
        owner_ctx = getattr(ctx.tools, "_ctx", None)
        control_reason = str(getattr(owner_ctx, "_transport_repeat_control_reason", "") or "")
    text, usage, llm_trace = _loop()._provider_unavailable_result(
        ctx, error_kind=error_kind, wait_cause=wait_cause, waited_sec=waited_sec,
        interactive=interactive, control_reason=control_reason,
    )
    if str(usage.get("reason_code") or "") not in ("", "deadline_local") and usage.get(
            "terminal_origin") in (None, TERMINAL_ORIGIN_HOST_NOTICE):
        usage["terminal_origin"] = TERMINAL_ORIGIN_HOST_SALVAGE
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
    remaining = (deadline - _loop().utc_now()).total_seconds()
    if remaining > task_pacing.effective_finalization_reserve_sec(tools._ctx):
        return None
    prompt = (
        f"[DEADLINE] The task deadline ({meta.get('deadline_at')}) is ~{max(0.0, remaining)/60:.1f} min away "
        "and the run will stop at it. Produce your best final answer NOW from the verified work so far; "
        "clearly mark anything unverified or incomplete. An honest best-effort result is the expected "
        "outcome here, not a failure."
    )
    fallback = "⚠️ Task reached its deadline; local finalization produced no answer."
    return _loop()._forced_final_answer(ctx, prompt=prompt, fallback_text=fallback, reason_code="deadline_local")


def _maybe_early_finalize(
    limit_ctx: _RoundLimitContext, tools: ToolRegistry, controls: Dict[str, Any],
    *, transport_episode: Optional[TransportWaitEpisode] = None,
) -> Optional[Tuple[str, Dict[str, Any], Dict[str, Any]]]:
    """Consume supervisor grace first, then a local deadline."""
    if controls.get("finalize_now"):
        # The owner's "Stop now" on a direct turn costs no call, so it is
        # honest whatever the transport is doing — never reported to the owner
        # as a provider-outage terminal.
        control_text = str(controls["finalize_now"])
        if control_text.strip() and control_text.splitlines()[0].strip() == REASON_OWNER_STOPPED_DIRECT_TURN:
            return _handle_direct_turn_hard_stop(limit_ctx)
        if transport_episode is not None:
            # Every other finalize_now flavor during an active outage takes the
            # honest no-resend terminal (rationale in the helper); control
            # bookkeeping was done by the drain, terminalization is prompt.
            return _finalize_now_transport_terminal(
                transport_episode, drive_logs=limit_ctx.drive_logs,
                task_id=limit_ctx.task_id, model=limit_ctx.active_model,
                control_reason=control_text.splitlines()[0].strip(),
                handle_provider_unavailable=functools.partial(
                    _handle_provider_unavailable, limit_ctx),
            )
        if controls.get("finalize_deadline_ts") is not None:
            _narrow_round_deadline(
                limit_ctx, controls["finalize_deadline_ts"],
            )
        return _loop()._handle_forced_finalization(limit_ctx, str(controls["finalize_now"]))
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
    ctx.deadline_ts = _loop()._task_deadline_epoch(tools)
    ctx.status_drive_root = pathlib.Path(
        str(meta.get("budget_drive_root") or getattr(tools._ctx, "budget_drive_root", "") or "")
        or (ctx.drive_root if ctx.drive_root is not None else pathlib.Path(ctx.drive_logs).parent)
    )
    ctx.root_task_id = str(meta.get("root_task_id") or ctx.task_id)
    candidate = getattr(tools._ctx, "_delivery_candidate", None)
    ctx.delivery_candidate = candidate if isinstance(candidate, _loop().DeliveryCandidate) else None
    ctx.tools = tools
    ctx.llm_trace = llm_trace
    return ctx
