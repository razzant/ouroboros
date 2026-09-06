"""Mid-task steering notes injected into the transcript: the self-check, time and
cost milestones, nanny economics and delegate-activity metering, round
checkpoints, plan-forcing prompts, skill-finalization wording, finalization
nudges and the answer protocol. Extracted from loop.py (v7 L-B split); loop.py
re-exports every name."""

from __future__ import annotations

import json
import logging
import pathlib
import queue

from typing import Any, Callable, Dict, List, Optional
from ouroboros import task_pacing
from ouroboros.nanny_pacing import _nanny_burn_phrase, _nanny_metered_since_delegate_activity, _nanny_reminder_due
from ouroboros.outcomes import extract_final_answer, latest_agent_defined_verification, latest_unreconciled_failed_verification, latest_unreconciled_masked_verification, should_nudge_verification, turn_has_reviewable_effects
# D18b: the trace-touched skill-name scan lives with the readiness predicate that
# consumes it (upstream 0463c6bb); the historical private name stays bound here so
# loop.py's re-export and every existing caller keep addressing one object.
from ouroboros.skill_readiness import skill_names_touched_by_trace as _skill_names_touched_by_trace  # noqa: F401 -- historical name
from ouroboros.tools.registry import ToolRegistry
from ouroboros.utils import estimate_tokens


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


def _skill_finalization_message(drive_root: pathlib.Path, llm_trace: Dict[str, Any]) -> str:
    names = _loop()._skill_names_touched_by_trace(llm_trace)
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
    """Project force-plan finalization from existing review + policy SSOTs."""
    from ouroboros.owner_hurry import force_plan_decision

    return force_plan_decision(
        ctx, _llm_trace, hard_rail=hard_rail,
        enforcement=_loop().get_review_enforcement(),
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
        else _loop()._force_plan_decision(ctx, llm_trace, hard_rail=forced_reason)
    )
    return plan_review_disclosure(decision, forced_reason)


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
    cost_ceiling: Optional["task_pacing.CostCeiling"] = None,
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
        estimate_tokens(_loop()._extract_plain_text_from_content(m.get("content")))
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
    tree_info = _loop()._loop_tree_accounting(refresh=True, max_age_sec=30.0)
    rendered = task_pacing.tree_spend_line(tree_info, cost_ceiling)
    if rendered:
        tree_accounted = float(tree_info["accounted_usd"])
        raw_cap = tree_info.get("root_limit_usd")
        tree_cap = float(raw_cap) if raw_cap is not None else None
        tree_line = f"{rendered}\n"

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
    _loop()._append_or_merge_user_message(messages, reminder)
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
    _loop()._emit_checkpoint_event(event_queue, task_id, drive_logs, checkpoint_payload)

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
        tree_cost_provider=lambda: _loop()._loop_tree_accounting(refresh=True, max_age_sec=30.0),
    )
    if note is None:
        return False
    _loop()._append_or_merge_user_message(messages, note.text)
    _loop()._emit_checkpoint_event(event_queue, task_id, drive_logs, note.checkpoint)
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
    tree_info = _loop()._loop_tree_accounting(
        refresh=True, max_age_sec=_loop()._TREE_ACCOUNTING_MAX_STALE_SEC,
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
    _loop()._append_or_merge_user_message(messages, note.text)
    _loop()._emit_checkpoint_event(event_queue, task_id, drive_logs, note.checkpoint)
    return True


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
    _loop()._append_or_merge_user_message(messages, reminder)
    # Owner decision (2026-08-15): no owner-chat progress line — the model sees
    # the reminder and the typed task_checkpoint below carries observability.
    _loop()._emit_checkpoint_event(event_queue, task_id, drive_logs, {
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
        cost_ceiling=cost_ceiling,
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


def _forced_delegation_note(tools_ctx: Any, llm_trace: Dict[str, Any]) -> str:
    """Build the forced-path nanny note from durable delegation custody."""
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
        # "Used once" is no permanent license (the poltergeist children ran one
        # $0 run, then co-built for tens of opus rounds behind this early
        # return): silence is proportional to the burn since the last delegated run.
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
        # PENDING ≠ FAILED (b49f8192): a STARTED row without settlement may
        # still be executing — "failed" invites a duplicate, finalizing orphans
        # the result; outranks the failed message even if a sibling died.
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

    def _inject(reminder: str, note: str) -> bool:
        # The one nudge protocol every advisory injection below follows: land
        # the model's pending text, append the reminder, record the note once
        # (live progress AND the durable trail), re-loop.
        if content and content.strip():
            messages.append({"role": "assistant", "content": content})
        _loop()._append_or_merge_user_message(messages, f"[SYSTEM REMINDER]\n{reminder}")
        emit_progress(note)
        llm_trace["reasoning_notes"].append(note)
        return True

    if (getattr(tools._ctx, "_nanny_route_dispatched", False)
            and not getattr(tools._ctx, "_nanny_finalization_injected", False)):
        # Nanny postcondition (owner 2026-08-07): a harness-dispatched child must
        # not finalize as if that decision never existed — one structural fact,
        # one re-loop, never a hard gate (P5); suppressions live in
        # _nanny_finalization_message.
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
            _loop()._append_or_merge_user_message(messages, f"[SYSTEM REMINDER]\n{_nanny_msg}")
            # Owner decision (2026-08-15): no owner-chat progress line — the
            # trace + typed task_checkpoint carry observability.
            _code = _nanny_msg.split(":", 1)[0].replace("⚠️", "").strip()
            _loop()._emit_checkpoint_event(
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
    finalization_msg = _loop()._skill_finalization_message(drive_root, llm_trace)
    if finalization_msg and not getattr(tools._ctx, "_skill_finalization_injected", False):
        tools._ctx._skill_finalization_injected = True
        return _inject(finalization_msg, finalization_msg)
    if not getattr(tools._ctx, "_verify_red_nudged", False):
        # Red-verification one-shot nudge: the latest host-attested verify
        # receipt is RED and unreconciled (distinct from receipt_absent below).
        # Before FR3; binary latch; advisory; forced paths bypass; keyed on the
        # typed receipt status, never content (P5).
        _failed_receipt = latest_unreconciled_failed_verification(
            drive_root, task_id, receipts=receipt_rows,
        )
        if _failed_receipt is not None:
            tools._ctx._verify_red_nudged = True
            _check = str(_failed_receipt.get("check") or "").strip()
            _rc = _failed_receipt.get("returncode")
            _on = f" on `{_check}`" if _check else ""
            _exit = f" (exit {_rc})" if _rc is not None else ""
            return _inject(
                "Your latest host-attested verification is RED" + _on + _exit +
                ". Before a clean final answer, reconcile it: re-check it, explain why this check is "
                "not the task's acceptance contract, or fix and re-run verification. This is advisory — "
                "if you finalize anyway, make the residual risk explicit.",
                "Red-verification nudge injected before final response.",
            )
    if not getattr(tools._ctx, "_verify_masked_nudged", False):
        # Exit-masking one-shot ADVISORY nudge (v6.52.2): a passing verify can
        # launder the exit code (`| tail`/`|| true`). After the red nudge; binary
        # latch; forced paths bypass; typed receipt sensor, never content (P5).
        _masked_receipt = latest_unreconciled_masked_verification(
            drive_root, task_id, receipts=receipt_rows,
        )
        if _masked_receipt is not None:
            tools._ctx._verify_masked_nudged = True
            _mcheck = str(_masked_receipt.get("check") or "").strip()
            _mreasons = ", ".join(str(x) for x in (_masked_receipt.get("check_exit_masking_reasons") or []))
            _mon = f" on `{_mcheck}`" if _mcheck else ""
            _mwhy = f" ({_mreasons})" if _mreasons else ""
            return _inject(
                "Your latest passing verification" + _mon + " uses a shell pipe" + _mwhy +
                " that can hide the real command's exit code, so a failing run could read as exit 0. "
                "Before a clean final answer, re-ground so the exit reflects the real result (drop the "
                "masking pipe / use the runner's own pass marker), or explain why it is reliable. This is "
                "advisory — if you finalize anyway, make the residual risk explicit.",
                "Masked-verification nudge injected before final response.",
            )
    if not getattr(tools._ctx, "_criterion_source_nudged", False):
        # Criterion-provenance one-shot ADVISORY nudge (v6.54.4): a green check on
        # an agent-defined criterion with no basis gets one reminder; after the
        # masked nudge, before FR3; typed receipt field, never content (P5).
        _agent_defined = latest_agent_defined_verification(
            drive_root, task_id, receipts=receipt_rows,
        )
        if _agent_defined is not None:
            tools._ctx._criterion_source_nudged = True
            _acheck = str(_agent_defined.get("check") or "").strip()
            _aon = f" (`{_acheck}`)" if _acheck else ""
            return _inject(
                "Your latest passing verification" + _aon + " uses a success "
                "criterion YOU defined, not one the task states. Before finalizing, double-check the "
                "criterion is equivalent to what the task actually asks for (format, units, scope) — "
                "re-run verify_and_record with criterion_basis stating why it suffices, or adjust the "
                "check. Advisory only — if you finalize anyway, make the assumption explicit.",
                "Criterion-provenance nudge injected before final response.",
            )
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
        return _inject(
            "Before finalizing: you produced a real deliverable but recorded no "
            "machine verification. Call verify_and_record — run your test/command (explicit_command/"
            "explicit_metric/visible_verifier), confirm the artifact exists (artifact_observation), or "
            "honestly declare no_visible_machine_contract — so the result is grounded, then continue.",
            "Verify-before-done nudge injected before final response.",
        )
        emit_progress("Verify-before-done nudge injected before final response.")
        llm_trace["reasoning_notes"].append("Verify-before-done nudge injected before final response.")
        return True
    # A3 one-shot no-op nudge: a declared deliverable but no tool calls,
    # reviewable effects or FINAL ANSWER marker this turn (family of the M2
    # expected_output_ungrounded flag). Own latch after the verify nudge; never
    # forces acceptance review; structural facts only.
    if (
        not getattr(tools._ctx, "_noop_attempt_nudged", False)
        and str(_contract_expected_output(tools._ctx)).strip()
        and not (llm_trace.get("tool_calls") or [])
        and not turn_has_reviewable_effects(llm_trace)
        and not extract_final_answer(content or "")
    ):
        tools._ctx._noop_attempt_nudged = True
        # v6.60.0: the nudge keys on expected_output SEMANTICS; it mentions the FINAL
        # ANSWER marker only when this task's contract actually declares the protocol.
        _marker_bit = (
            "no tool calls, no reviewable effects, no FINAL ANSWER"
            if _answer_protocol_active(tools._ctx)
            else "no tool calls, no reviewable effects, no delivered answer"
        )
        return _inject(
            "This task declares an expected output, but you are about to finalize "
            f"without having attempted it — {_marker_bit}. "
            "Actually attempt the task now (do the work / produce the deliverable / derive the answer), "
            "then finalize. If it is genuinely blocked, say so with the concrete blocker and evidence.",
            "No-op attempt nudge injected before final response.",
        )
        emit_progress("No-op attempt nudge injected before final response.")
        llm_trace["reasoning_notes"].append("No-op attempt nudge injected before final response.")
        return True
    # P2 one-shot final-answer-marker nudge: real work + prose, no FINAL ANSWER
    # marker — the agent marks its OWN answer, prose is never mined (P5). Own
    # latch after verify/red/A3; the protocol gate alone suffices (v6.56.0).
    if (
        not getattr(tools._ctx, "_final_marker_nudged", False)
        and _answer_protocol_active(tools._ctx)  # v6.60.0: marker nudge is protocol-gated
        and content and content.strip()
        and not extract_final_answer(content or "")
        and ((llm_trace.get("tool_calls") or []) or turn_has_reviewable_effects(llm_trace))
    ):
        tools._ctx._final_marker_nudged = True
        return _inject(
            "You have done the work but have not marked a final answer. If you "
            "are done, end your response with a single line, exactly: FINAL ANSWER: <answer> — the "
            "bare deliverable only (a number / a few words / a short list), so it is captured even if "
            "the run is cut short. If you are not done, keep working.",
            "Final-answer marker nudge injected before final response.",
        )
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
    declared = str(contract.get("expected_output") or "") if isinstance(contract, dict) else ""
    if declared.strip():
        return declared
    metadata = getattr(ctx, "task_metadata", {})
    if isinstance(metadata, dict):
        declared = str(metadata.get("expected_output") or "")
        if declared.strip():
            return declared
        meta_contract = metadata.get("task_contract")
        if isinstance(meta_contract, dict):
            return str(meta_contract.get("expected_output") or "")
    return ""
