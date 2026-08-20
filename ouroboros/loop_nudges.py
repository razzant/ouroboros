"""Mid-task steering notes injected into the transcript: the self-check, time and
cost milestones, nanny economics and delegate-activity metering, round
checkpoints, plan-forcing prompts, skill-finalization wording, finalization
nudges and the answer protocol. Extracted from loop.py (v7 L-B split); loop.py
re-exports every name."""

from __future__ import annotations

import json
import queue
import pathlib
from typing import Any, Callable, Dict, List, Optional, Tuple
import logging

from ouroboros import task_pacing
from ouroboros.outcomes import (
    extract_final_answer,
    latest_agent_defined_verification,
    latest_unreconciled_failed_verification,
    latest_unreconciled_masked_verification,
    should_nudge_verification,
    turn_has_reviewable_effects,
)
from ouroboros.tools.registry import ToolRegistry
from ouroboros.utils import estimate_tokens


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
    # Normal finalization reuses the reducer projection that already decided
    # this exact candidate. The trace copy is presentation-only and cannot grant
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
) -> bool:
    """Inject a normal user-turn self-check and emit one checkpoint event."""
    REMINDER_INTERVAL = 15
    if round_idx <= 1 or round_idx % REMINDER_INTERVAL != 0 or round_idx >= max_rounds:
        return False

    ctx_tokens = sum(
        estimate_tokens(_loop()._extract_plain_text_from_content(m.get("content")))
        for m in messages
    )
    raw_task_cost = accumulated_usage.get("cost")
    task_cost = float(raw_task_cost) if raw_task_cost is not None else None
    cost_text = f"${task_cost:.2f}" if task_cost is not None else "unknown"
    checkpoint_num = round_idx // REMINDER_INTERVAL

    # Tree spend under a root cap (v6.91): the checkpoint is an already
    # cache-breaking user turn, so it is one of the RARE surfaces allowed to
    # carry a live ledger number (DEVELOPMENT cache_friendliness item 22). The
    # fence counts the whole tree, so own cost alone hid two tree deaths.
    tree_line = ""
    tree_accounted: Optional[float] = None
    tree_cap: Optional[float] = None
    tree_info = _loop()._loop_tree_accounting(refresh=True, max_age_sec=30.0)
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


# The verbs whose call IS delegated-run activity for the nanny-economics baseline.
# Exact tool-call transitions, observed in the loop as they happen — never a scan
# of the custody log or events.jsonl (the baseline must be free to read per round).
_DELEGATE_ACTIVITY_TOOLS = frozenset({
    "delegate_start", "delegate_wait", "delegate_cancel", "delegate_answer",
})


def _note_nanny_delegate_activity(
    ctx: Any, round_idx: int, accumulated_usage: Dict[str, Any],
    tool_calls: List[Dict[str, Any]],
) -> None:
    """Advance the nanny's metered-progress marker, and its delegate-activity baseline
    when this round actually touched a delegated run.

    Two process-local marks on the ToolContext, written once per round: what the task
    has spent so far (round index + accumulated cost), and where that stood at the
    LAST delegate-verb call. Their difference is the whole input of the proportional
    reminder — the poltergeist children burned $87 of opus rounds co-building around
    their $0 runs, and nothing measured the burn while it happened.
    """
    if not getattr(ctx, "_nanny_route_dispatched", False):
        return
    try:
        cost = float(accumulated_usage.get("cost") or 0.0)
    except (TypeError, ValueError):
        cost = 0.0
    mark = {"round": int(round_idx), "cost": cost}
    ctx._nanny_metered_progress = mark
    verbs = set()
    for call in tool_calls or []:
        fn = call.get("function") if isinstance(call, dict) else None
        name = str((fn or {}).get("name") or "").strip() if isinstance(fn, dict) else ""
        if name in _DELEGATE_ACTIVITY_TOOLS:
            verbs.add(name)
    if not verbs:
        return
    if verbs == {"delegate_wait"}:
        # R2-5: a wait is WATCHING, not delegating — it advances only the
        # ROUND half of the baseline. Preserving the COST half keeps the
        # dollar axis cumulative across waits: re-zeroing BOTH axes at every
        # wait never heard the reminder ($0.24/round probe), while a genuinely
        # holding nanny stays under the dollar threshold anyway.
        prior = getattr(ctx, "_nanny_delegate_baseline", None)
        prior_cost = float(prior.get("cost") or 0.0) if isinstance(prior, dict) else 0.0
        ctx._nanny_delegate_baseline = {"round": mark["round"], "cost": prior_cost}
    else:
        ctx._nanny_delegate_baseline = dict(mark)
    # Delegate activity also RE-ARMS the reminder: the fire cursor is
    # cleared so a cooldown earned BEFORE this activity can never mute
    # the reminder for burn that happens AFTER it (gemini, fix F1).
    ctx._nanny_reminder_mark = None


def _nanny_metered_since_delegate_activity(ctx: Any) -> Tuple[int, float]:
    """(rounds, dollars) this task's OWN metered loop has spent since the last
    delegate-verb call — zero before the first round is marked."""
    progress = getattr(ctx, "_nanny_metered_progress", None)
    progress = progress if isinstance(progress, dict) else {}
    baseline = getattr(ctx, "_nanny_delegate_baseline", None)
    baseline = baseline if isinstance(baseline, dict) else {}
    try:
        rounds = max(0, int(progress.get("round") or 0) - int(baseline.get("round") or 0))
    except (TypeError, ValueError):
        rounds = 0
    try:
        cost = max(0.0, float(progress.get("cost") or 0.0) - float(baseline.get("cost") or 0.0))
    except (TypeError, ValueError):
        cost = 0.0
    return rounds, cost


def _nanny_reminder_due(ctx: Any, round_idx: int) -> Tuple[int, float, bool]:
    """The measured burn plus whether the proportional reminder is due THIS round.

    Due when EITHER axis (rounds or dollars, ``task_pacing.NANNY_REMINDER_*``)
    crossed its threshold since the last delegate-verb call. The re-arm is
    dual-axis too (fix F1): the next firing waits for a further threshold-width
    on EITHER axis, so a fast dollar burn is never muted by round spacing. The
    first firing has no spacing gate; delegate activity clears the fire cursor
    (``_note_nanny_delegate_activity``). Proportional and repeating, never a cap
    (owner decision 2=B). With no delegate verb AND no prior firing, the first
    reminder fires early (``NANNY_FIRST_REMINDER_ROUNDS``, owner-approved
    2026-08-15) regardless of dollars; any delegate activity or re-arm restores
    the ordinary dual-axis thresholds unchanged."""
    from ouroboros.task_pacing import (
        NANNY_FIRST_REMINDER_ROUNDS, NANNY_REMINDER_ROUNDS, NANNY_REMINDER_USD,
    )

    rounds, cost = _nanny_metered_since_delegate_activity(ctx)
    round_threshold = NANNY_REMINDER_ROUNDS
    if (
        not isinstance(getattr(ctx, "_nanny_delegate_baseline", None), dict)
        and not isinstance(getattr(ctx, "_nanny_reminder_mark", None), dict)
    ):
        # No delegate verb AND no reminder yet: first firing comes early.
        round_threshold = NANNY_FIRST_REMINDER_ROUNDS
    if rounds < round_threshold and cost < NANNY_REMINDER_USD:
        return rounds, cost, False
    mark = getattr(ctx, "_nanny_reminder_mark", None)
    if not isinstance(mark, dict):
        return rounds, cost, True  # first firing: no spacing gate
    progress = getattr(ctx, "_nanny_metered_progress", None)
    progress = progress if isinstance(progress, dict) else {}
    try:
        rounds_since_fire = int(progress.get("round") or 0) - int(mark.get("round") or 0)
    except (TypeError, ValueError):
        rounds_since_fire = 0
    try:
        cost_since_fire = float(progress.get("cost") or 0.0) - float(mark.get("cost") or 0.0)
    except (TypeError, ValueError):
        cost_since_fire = 0.0
    if rounds_since_fire >= NANNY_REMINDER_ROUNDS or cost_since_fire >= NANNY_REMINDER_USD:
        return rounds, cost, True
    return rounds, cost, False


def _nanny_burn_phrase(rounds: int, cost: float) -> str:
    return (f"{rounds} of your own metered LLM rounds (~${cost:.2f})" if cost > 0
            else f"{rounds} of your own metered LLM rounds")


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

    A plain user-message reminder in the existing self-checkpoint style — the loop's
    checkpoints are ordinary user turns, never protocol (ARCHITECTURE: "Loop
    self-checkpoints remain plain user-message reminders"). It fires between rounds,
    while the burn is happening, because the finalization nudge alone arrives only
    after the money is spent. Proportional and unbounded in count: each further
    threshold-width of metered rounds re-arms it (owner 2=B — no round cap)."""
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
    # activity" — the burn is measured from the task's start, and the wording
    # says so instead of implying an activity that never happened.
    _baseline_known = isinstance(getattr(ctx, "_nanny_delegate_baseline", None), dict)
    since_phrase = ("since your last delegated-run activity" if _baseline_known
                    else "since this task started (no delegated-run activity yet)")
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
    """The nanny postcondition's forced-path half, grounded in DURABLE custody.

    A forced finalization may not re-loop, so the substrate fact rides the one
    final prompt. `delegate_custody.task_execution_evidence` on the custody root
    (canonical/budget root — the split-root rule Phase A fixed) decides, not just
    this execution's trace: succeeded → no note; started-but-unsettled → pending
    wording (no retry pressure); settled-without-success → truthful failure
    wording; zero started with readable evidence → the no-delegation wording;
    unreadable evidence → no accusation."""
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
        # The proportional silence must not extend to FORCED exits (grok / F16):
        # a wrap-up forced by an overrun still owes the parent the honest-spend
        # line. One shot, riding the single forced prompt — never a re-loop.
        rounds, cost = _nanny_metered_since_delegate_activity(tools_ctx)
        from ouroboros.task_pacing import NANNY_REMINDER_ROUNDS, NANNY_REMINDER_USD

        if rounds >= NANNY_REMINDER_ROUNDS or cost >= NANNY_REMINDER_USD:
            return (
                "\nNOTE: your delegated run(s) succeeded, but you have since spent "
                f"{_nanny_burn_phrase(rounds, cost)} with no delegated-run activity. "
                "Account for that metered spend honestly in your answer."
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
    if any(str(c.get("tool") or "") == "delegate_start"
           for c in (llm_trace.get("tool_calls") or []) if isinstance(c, dict)):
        # The trace shows a dispatch the durable rows have not recorded — never
        # accuse over evidence that is behind the task's own actions.
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
    runs CRASHED of "choosing" not to delegate, and fired even when the verbs
    were policy-hidden. Two structural facts fix both: the task's own visible
    toolset, and durable custody evidence (delegate_custody.
    task_execution_evidence), which spans the WHOLE task — per-execution
    llm_trace resets on continuation. `trace_attempted` is the third fact: a
    delegate_start in THIS execution's trace; it must not suppress the failure
    message (triad, e84475f2: delegate, run dies, finish by hand, finalize —
    all in ONE execution), only the accusation when custody has no rows yet (a
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
        # (budget) root, but this read used the loop's drive_root — a split-root
        # child drive has no custody rows, leaving the nanny blind. Resolve the
        # SAME root the writers use; drive_root stays the unit-stub fallback.
        try:
            evidence_root = custody_root(tools._ctx)
        except Exception:
            evidence_root = drive_root
        evidence = task_execution_evidence(evidence_root, str(task_id or ""))
    except Exception:
        log.debug("nanny nudge: custody evidence read failed", exc_info=True)
    if evidence.get("delegated_runs_succeeded"):
        # The route WAS used and worked — but "used once" is not a permanent
        # license: the poltergeist children each ran ONE successful $0 run,
        # then co-built for tens of opus rounds while this early return kept
        # the nudge silent. Silence is now proportional to the measured burn
        # since the last delegated-run activity.
        rounds, cost = _nanny_metered_since_delegate_activity(tools._ctx)
        from ouroboros.task_pacing import NANNY_REMINDER_ROUNDS, NANNY_REMINDER_USD

        if rounds < NANNY_REMINDER_ROUNDS and cost < NANNY_REMINDER_USD:
            return ""
        return (
            "⚠️ NANNY_METERED_OVERRUN: your delegated run(s) succeeded, but you have "
            f"since spent {_nanny_burn_phrase(rounds, cost)} with no delegated-run "
            "activity. A successful run is verified and integrated, not rebuilt. If "
            "the remaining work is substantive, delegate it (a new delegate_start); "
            "if you are wrapping up, keep the wrap-up short and account for the "
            "metered spend honestly in your result."
        )
    started = int(evidence.get("delegated_runs_started") or 0)
    if not started and (evidence.get("evidence_read_failed") or not evidence):
        # Zero attempts is an ACCUSATION and needs positively-established
        # evidence: an unreadable custody log (or a failed read above) proves
        # nothing (scope finding on a5e59bdf).
        return ""
    if not started and trace_attempted:
        # A start this trace saw but custody has no row for: pending settlement
        # or an uncustodied start — an attempt either way; neither accusation
        # fits, and the wait/cancel path owns its own disclosure.
        return ""
    settled = int(evidence.get("delegated_runs_settled") or 0)
    failure_states = [str(s) for s in (evidence.get("delegated_run_failure_states") or [])]
    pending = max(0, started - settled)
    if pending:
        # PENDING ≠ FAILED (sol review, b49f8192): a STARTED row with no
        # settlement may still be executing — calling it failed invites a
        # duplicate run, and finalizing over it orphans the result. Takes
        # precedence over the failed message: with a run in flight, "retry" is
        # wrong even when an earlier sibling died (still a fact below).
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
    if (getattr(tools._ctx, "_nanny_route_dispatched", False)
            and not getattr(tools._ctx, "_nanny_finalization_injected", False)):
        # Nanny postcondition (owner decision, 2026-08-07): a child dispatched
        # onto the delegated substrate must not finalize as if that decision
        # never existed. One structural fact, one re-loop; the child may still
        # delegate OR finalize with a typed reason — never a hard gate (P5).
        # A delegate_start in THIS trace rides into the message decision
        # (triad, e84475f2), where custody evidence separates a failed run
        # (NANNY_DELEGATED_RUN_FAILED) from a pending attempt (no message).
        # Suppression cases live in _nanny_finalization_message.
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
    finalization_msg = _skill_finalization_message(drive_root, llm_trace)
    if finalization_msg and not getattr(tools._ctx, "_skill_finalization_injected", False):
        tools._ctx._skill_finalization_injected = True
        if content and content.strip():
            messages.append({"role": "assistant", "content": content})
        _loop()._append_or_merge_user_message(messages, f"[SYSTEM REMINDER]\n{finalization_msg}")
        emit_progress(finalization_msg)
        llm_trace["reasoning_notes"].append(finalization_msg)
        return True
    if not getattr(tools._ctx, "_verify_red_nudged", False):
        # Red-verification one-shot nudge: the latest host-attested verify receipt
        # is RED and unreconciled — finalizing over your own failing check is a
        # self-contradiction (Bible P3/P12), distinct from receipt_absent below
        # ("no grounding" vs "grounding says FAIL"). Ordered BEFORE the FR3 verify
        # nudge. Binary latch; advisory; forced-finalization paths bypass it.
        # Keyed on the typed receipt status, never content (Bible P5).
        _failed_receipt = latest_unreconciled_failed_verification(drive_root, task_id)
        if _failed_receipt is not None:
            tools._ctx._verify_red_nudged = True
            _check = str(_failed_receipt.get("check") or "").strip()
            _rc = _failed_receipt.get("returncode")
            _on = f" on `{_check}`" if _check else ""
            _exit = f" (exit {_rc})" if _rc is not None else ""
            if content and content.strip():
                messages.append({"role": "assistant", "content": content})
            _loop()._append_or_merge_user_message(
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
        # false-green tutanota hit). Distinct from the red nudge; ordered
        # after it. Binary latch; advisory; forced paths bypass it. Flag-
        # driven on the typed receipt sensor, never content (Bible P5).
        _masked_receipt = latest_unreconciled_masked_verification(drive_root, task_id)
        if _masked_receipt is not None:
            tools._ctx._verify_masked_nudged = True
            _mcheck = str(_masked_receipt.get("check") or "").strip()
            _mreasons = ", ".join(str(x) for x in (_masked_receipt.get("check_exit_masking_reasons") or []))
            _mon = f" on `{_mcheck}`" if _mcheck else ""
            _mwhy = f" ({_mreasons})" if _mreasons else ""
            if content and content.strip():
                messages.append({"role": "assistant", "content": content})
            _loop()._append_or_merge_user_message(
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
            _loop()._append_or_merge_user_message(
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
        _loop()._append_or_merge_user_message(
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
    # expected_output) but the turn made NO tool calls, NO reviewable effects,
    # NO FINAL ANSWER marker — about-to-finalize-without-attempting (same
    # family as the M2 expected_output_ungrounded flag). Own latch, AFTER the
    # verify nudge; never forces acceptance review; forced paths return
    # earlier. Structural facts only (no refusal-text matching).
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
        _loop()._append_or_merge_user_message(
            messages,
            "[SYSTEM REMINDER]\nThis task declares an expected output, but you are about to finalize "
            f"without having attempted it — {_marker_bit}. "
            "Actually attempt the task now (do the work / produce the deliverable / derive the answer), "
            "then finalize. If it is genuinely blocked, say so with the concrete blocker and evidence.",
        )
        emit_progress("No-op attempt nudge injected before final response.")
        llm_trace["reasoning_notes"].append("No-op attempt nudge injected before final response.")
        return True
    # P2 one-shot final-answer-marker nudge: the turn produced REAL work AND
    # visible prose but no FINAL ANSWER marker — the typed extractor would drop
    # it and a forced/deadline finalization would score empty. Strengthen the
    # BEHAVIOR (ask the agent to mark its OWN answer), never mine prose into a
    # claimed answer (Bible P5). Own latch, ordered AFTER verify/red/A3
    # (grounding outranks formatting); mutually exclusive with the A3 no-op
    # nudge; forced paths return earlier. Structural facts only. The protocol
    # gate alone suffices: answer_protocol="final_answer_line" itself declares
    # a machine-extracted deliverable, so the nudge must not ALSO require a
    # declared expected_output — GAIA-shaped contracts keep expected_output
    # empty, and that extra gate once suppressed the only salvage surface
    # (a v6.56.0 run finalized a last-round refusal empty despite 24 calls).
    if (
        not getattr(tools._ctx, "_final_marker_nudged", False)
        and _answer_protocol_active(tools._ctx)  # v6.60.0: marker nudge is protocol-gated
        and content and content.strip()
        and not extract_final_answer(content or "")
        and ((llm_trace.get("tool_calls") or []) or turn_has_reviewable_effects(llm_trace))
    ):
        tools._ctx._final_marker_nudged = True
        messages.append({"role": "assistant", "content": content})
        _loop()._append_or_merge_user_message(
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
