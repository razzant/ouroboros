"""Round-limit and terminal-drain handling for the main loop: owner-stop drain
and its window, incoming-message drain, round compaction and its usage
accounting, the round-limit context, and the limit/forced/owner-stop/
provider-unavailable/deadline handlers. Extracted from loop.py (v7 L-B split);
loop.py re-exports every name."""

from __future__ import annotations

import queue
import pathlib
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
import logging

from ouroboros.llm import LLMClient, add_usage
from ouroboros import task_pacing
from ouroboros.config import get_light_model
from ouroboros.outcomes import REASON_OWNER_REQUESTED_FINALIZATION, RESULT_INFRA_FAILED
from ouroboros.tools.registry import ToolRegistry
from ouroboros.context import build_user_content
from ouroboros.deadline_utils import parse_deadline_ts
from ouroboros.loop_tool_execution import prune_reclaim_trace_refs, reclaim_negative_memo, reclaim_trace_refs
from ouroboros.loop_llm_call import emit_llm_usage_event
# The owner-content appender is the messages leaf's own public name (L3): nothing
# rebinds it on the loop, so the sibling owner is imported directly (frozen
# sibling import: the name is a pure function no test patches — the same
# accepted class as the loop_llm_call sibling imports; if a test ever needs to
# intercept it, flip this to a late-bound read).
from ouroboros.loop_messages import _append_or_merge_user_content
from ouroboros.pricing import estimate_cost_optional

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # annotation-only names; lazy under future annotations, never imported at runtime
    from ouroboros.loop_delivery import DeliveryCandidate


# The parent logger name is pinned on purpose: records moved with their code
# keep the exact `%(name)s` every handler and reader saw before the split.
log = logging.getLogger("ouroboros.loop")


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


def _provider_failure_hint(accumulated_usage: Dict[str, Any]) -> str:
    detail = " ".join(str(accumulated_usage.get("_last_llm_error") or "").split()).strip()
    if not detail:
        return ""
    return f" Last provider error: {detail}"


def _provider_recovery_hint(accumulated_usage: Dict[str, Any]) -> str:
    """Explain whether retrying later is likely to help."""
    kind = str(accumulated_usage.get("_last_llm_error_kind") or "").strip()
    if kind == "subscription_window_exhausted":
        reset_at = str(accumulated_usage.get("_last_llm_reset_at") or "").strip()
        when = f" It resets at {reset_at}." if reset_at else ""
        return (
            " The subscription window for the delegated route is spent. This is "
            f"TRANSIENT, not a billing refusal — waiting cures it.{when} Retrying is "
            "scheduled against that reset time, not the ordinary short backoff."
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


def _task_deadline_epoch(tools: ToolRegistry) -> Optional[float]:
    """Task deadline as epoch seconds, for deadline-bounded LLM retry backoff."""
    meta = getattr(tools._ctx, "task_metadata", {})
    if not isinstance(meta, dict):
        return None
    deadline = parse_deadline_ts(meta.get("deadline_at"))
    return deadline.timestamp() if deadline is not None else None


def _mark_owner_stop_control_drained(
    owner_ctx: Any, drive_root: Optional[pathlib.Path], task_id: str,
) -> None:
    """Stamp the owner-stop finalize control's DELIVERY on the durable intent.

    The intent lives on the CANONICAL data root (``budget_drive_root`` first;
    a forked task's mailbox drive differs). Idempotent (first drain wins). A
    failed stamp is retried ONCE; still unconfirmed, a typed forensic event
    is appended and no extended budget is assumed: the sweep keeps the
    request+outer-cap deadline, and ``_owner_stop_window_elapsed`` reads the
    same unstamped intent, bounding the worker by that anchor."""
    try:
        from ouroboros.cancel_intents import active_intent, mark_finalize_control_drained

        root = (
            str(getattr(owner_ctx, "budget_drive_root", "") or "")
            or (str(drive_root) if drive_root is not None else "")
        )
        if not (root and task_id):
            return
        root_path = pathlib.Path(root)
        for _ in range(2):
            if mark_finalize_control_drained(root_path, task_id):
                return
            row = active_intent(root_path, task_id)
            if isinstance(row, dict) and str(row.get("control_drained_at") or ""):
                return  # already stamped: the durable anchor is confirmed
        from ouroboros.utils import append_jsonl, utc_now_iso

        append_jsonl(root_path / "logs" / "events.jsonl", {
            "ts": utc_now_iso(), "type": "owner_stop_stamp_failed",
            "task_id": task_id,
        })
    except Exception:
        log.debug("owner-stop drain stamp failed for %s", task_id, exc_info=True)


def _owner_stop_window_elapsed(ctx: "_RoundLimitContext") -> bool:
    """Whether the durable owner-stop deadline already passed at consume.

    Reads the SAME durable intent the custody sweep budgets from (no drain
    stamp -> the conservative request+outer-cap anchor). Fail-soft: an
    unreadable intent keeps the bounded summary running."""
    try:
        from ouroboros.cancel_intents import STOP_POLICY_FINALIZE, active_intent, stop_policy
        from ouroboros.config import get_finalization_grace_sec
        from supervisor.owner_stop import owner_stop_deadline_ts

        root = getattr(ctx, "status_drive_root", None) or ctx.drive_root
        if root is None or not ctx.task_id:
            return False
        intent = active_intent(pathlib.Path(root), ctx.task_id)
        if not isinstance(intent, dict) or stop_policy(intent) != STOP_POLICY_FINALIZE:
            return False
        deadline = owner_stop_deadline_ts(intent, float(get_finalization_grace_sec()))
        return time.time() >= deadline if deadline else True
    except Exception:
        log.debug("owner-stop window check failed for %s", ctx.task_id, exc_info=True)
        return False


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
                _append_or_merge_user_content(messages, _loop()._owner_marked_content(owner_content))
            else:
                _loop()._record_owner_directive(
                    owner_ctx, source="direct_incoming", content=injected,
                )
                _loop()._append_or_merge_user_message(messages, _loop()._owner_marked_content(injected))
        except queue.Empty:
            break

    if drive_root is not None and task_id:
        from ouroboros.owner_mailbox import KIND_FINALIZE_NOW, KIND_HURRY, KIND_OWNER_TEXT, drain_owner_entries
        for entry in drain_owner_entries(drive_root, task_id=task_id, seen_ids=_owner_msg_seen):
            kind = entry.get("kind") or KIND_OWNER_TEXT
            if kind == KIND_FINALIZE_NOW:
                text = str(entry.get("text") or "deadline")
                controls["finalize_now"] = text
                first_line = text.splitlines()[0].strip() if text else ""
                if first_line == REASON_OWNER_REQUESTED_FINALIZATION:
                    # Owner-stop budget starts at DELIVERY (1=A): stamp the drain
                    # so the custody sweep budgets the final turn from here, not
                    # the button press. First drain wins; fail-soft.
                    _mark_owner_stop_control_drained(owner_ctx, drive_root, task_id)
                continue
            if kind == KIND_HURRY:
                # HQ1 no-chat contract (§19.7.2 item 6): a typed hurry control is
                # routed structurally — never through _record_owner_directive,
                # _owner_marked_content, messages, or owner_message_injected.
                from ouroboros.owner_hurry import apply_latch

                apply_latch(owner_ctx, entry, event_queue=event_queue)
                controls["hurry"] = str(entry.get("msg_id") or "hurry")
                continue
            dmsg = entry.get("text") or ""
            _loop()._record_owner_directive(
                owner_ctx,
                source="owner_mailbox",
                content=dmsg,
                msg_id=str(entry.get("msg_id") or ""),
            )
            from ouroboros.client_surface import noted_owner_text

            _loop()._append_or_merge_user_message(
                messages, _loop()._owner_marked_content(noted_owner_text(owner_ctx, entry, dmsg)))
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
    """Cooperative finalize-and-exit when the supervisor opens a grace window.

    The supervisor sends a typed finalize_now control through the owner
    mailbox when the task deadline/hard-timeout is reached; this extracts one
    tool-less best final answer inside the grace window so a deadline NEVER
    returns emptiness. An OWNER-STOP control (its payload's first line is the
    typed ``owner_requested_finalization`` literal, optionally followed by the
    bounded child projection) routes to its own rail: the owner's stop must
    never persist the deadline's false reason (CF-02).
    """
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
    return _loop()._forced_final_answer(ctx, prompt=prompt, fallback_text=fallback, reason_code="finalization_grace")


def _handle_owner_stop_finalization(
    ctx: _RoundLimitContext, control_text: str,
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """Owner-requested finalization (Q1/Q3=A): ZERO or ONE tool-less model turn.

    A current valid complete DeliveryCandidate is reused with zero new model
    turns; otherwise exactly one logical tool-less call runs (transport retries
    keep the existing call seam; the generic second semantic refresh is
    structurally disabled — owner steering is fenced during a pending stop, so
    no late directive can arrive). The typed ``owner_requested_finalization``
    reason flows through the best-effort gate, so a successful synthesis
    terminalizes ``completed``/best-effort — never the deadline's
    ``acceptance_bypassed_deadline`` falsehood (CF-02)."""
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
        # An expired control never buys a paid summary: the honest fallback
        # rides the same typed rail and custody settles it.
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


def _handle_provider_unavailable(ctx: _RoundLimitContext) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """Provider-death terminalization: the model returned no usable response
    after the transport same-model reroute + retries (+ any configured
    cross-model fallback). SALVAGE like the other forced rails — one tool-less
    final answer (which itself benefits from the same-model reroute) and,
    failing that, the last assistant text already produced — but terminalize as
    an INFRA FAILURE, never as a completion: an outage interrupts the task with
    the objective unmet, and calling that "completed (best effort)" was a lie
    that hid a real outage from the owner (95 minutes of silence)."""
    # A stale DeliveryCandidate is still the best complete text available when
    # the provider is dead. _forced_fallback_result preserves its original
    # evidence provenance and adds a host-owned resume disclosure rather than
    # laundering unchanged text onto the newer evidence fingerprint.
    candidate = _loop()._live_delivery_candidate(ctx)
    salvaged = candidate.full_text if candidate is not None else _loop()._last_assistant_text(ctx.messages)
    if candidate is None and not salvaged and ctx.drive_root is not None:
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
        "The task is being INTERRUPTED by this outage, not completed. Summarize the "
        "verified work so far and state plainly what remains undone."
    )
    text, usage, llm_trace = _loop()._forced_final_answer(
        ctx, prompt=prompt, fallback_text=fallback, reason_code="provider_unavailable",
    )
    # Honesty (P1): a provider outage interrupts the task — it never "completes"
    # it. Stamp the infra-failure execution status so the outcome reducer lands
    # on infra_failed/provider (terminal: failed) instead of the old best-effort
    # promotion to "completed"; the salvage text still rides the result body.
    # Skipped when a swarm routing handoff already cleared the rail (the admitted
    # task owns its lifecycle). NOTE: "interrupted" is deliberately NOT used —
    # STATUS_INTERRUPTED is a pre-requeue, non-terminal state in this codebase.
    if str(usage.get("reason_code") or "") == "provider_unavailable":
        usage["execution_status"] = RESULT_INFRA_FAILED
    return text, usage, llm_trace


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
    remaining = (deadline - _loop().utc_now()).total_seconds()
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
    return _loop()._forced_final_answer(ctx, prompt=prompt, fallback_text=fallback, reason_code="deadline_local")


def _maybe_early_finalize(
    limit_ctx: _RoundLimitContext, tools: ToolRegistry, controls: Dict[str, Any]
) -> Optional[Tuple[str, Dict[str, Any], Dict[str, Any]]]:
    """One early-exit gate per round: supervisor finalize_now first, then a
    loop-local real-deadline finalize. Returns the forced answer or None."""
    if controls.get("finalize_now"):
        return _loop()._handle_forced_finalization(limit_ctx, str(controls["finalize_now"]))
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
