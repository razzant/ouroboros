"""Host notices injected into the transcript: one-time time and cost threshold
crossings, plan-forcing prompts, skill-finalization facts, one-shot finalization
facts and the answer protocol. Extracted from loop.py (v7 L-B split); loop.py
re-exports every name. Notices state observations and the available actions;
periodic questions and strategy advice were removed (owner 11 Oct, #1748).
Every-round facts ride the facts line (``loop_messages``)."""

from __future__ import annotations

import logging
import pathlib
import queue

from typing import Any, Callable, Dict, List, Optional
from ouroboros import task_pacing
from ouroboros.nanny_pacing import nanny_burn_fact, nanny_metered_since_delegate_activity
from ouroboros.outcomes import _ROOT_WRITE_TOOLS, extract_final_answer, latest_unreconciled_failed_verification, result_unexamined, reviewable_effect_projection, turn_has_reviewable_effects
# D18b: the trace-touched skill-name scan lives with the readiness predicate that
# consumes it (upstream 0463c6bb); the historical private name stays bound here so
# loop.py's re-export and every existing caller keep addressing one object.
from ouroboros.skill_readiness import skill_names_touched_by_trace as _skill_names_touched_by_trace  # noqa: F401 -- historical name
from ouroboros.tools.registry import ToolRegistry


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
                f"blockers={readiness.blockers}, "
                f"agent_fixable={readiness.agent_fixable_blockers}, "
                f"owner_action={readiness.owner_action_blockers}, "
                f"next_actions={readiness.next_actions}"
            )
    if not blockers:
        return ""
    return (
        "⚠️ SKILL_NOT_FINALIZED: self-authored skills this task touched are not ready to "
        "execute. A fresh review does not supply a missing owner grant or override an "
        "intentional disablement, and an intentionally disabled skill may still satisfy the "
        "task. Current facts, each with its available next actions: " + "; ".join(blockers)
    )


def _plan_gate_decision(
    ctx: Any,
    _llm_trace: Dict[str, Any],
    *,
    hard_rail: str = "",
) -> Dict[str, Any]:
    """Project plan finalization from existing review + policy SSOTs."""
    from ouroboros.owner_hurry import plan_gate_decision

    return plan_gate_decision(ctx, _llm_trace, hard_rail=hard_rail)


def _plan_gate_reminder(decision: Dict[str, Any]) -> str:
    from ouroboros.owner_hurry import plan_review_reminder

    return plan_review_reminder(decision)


def _plan_gate_disclosure(
    ctx: Any,
    llm_trace: Dict[str, Any],
    *,
    forced_reason: str = "",
    author_stop: bool = False,
) -> str:
    # Normal finalization reuses the reducer projection that decided this
    # exact candidate. The trace copy is presentation-only, granting no
    # permission; forced rails (and a stop, which no gate decided) recompute,
    # and the decision they disclose replaces the projection the outcome reads
    # (none when no plan is engaged now: an earlier hold's view never outlives
    # it). Only the stop's own call says it stopped (``author_stop``).
    from ouroboros.owner_hurry import plan_review_disclosure

    decision = llm_trace.get("plan_gate", llm_trace.get("force_plan_decision"))
    if forced_reason or author_stop or not isinstance(decision, dict):
        for stale in ("plan_gate", "force_plan_decision"):
            llm_trace.pop(stale, None)
        decision = _loop()._plan_gate_decision(ctx, llm_trace, hard_rail=forced_reason)
        if decision.get("required"):
            llm_trace["plan_gate"] = decision
    return plan_review_disclosure(decision, forced_reason, author_stop=author_stop)


def _inject_note(note: Optional["task_pacing.PacingNote"], messages: List[Dict[str, Any]], tools: ToolRegistry,
                 accumulated_usage: Optional[Dict[str, Any]], event_queue: Optional[queue.Queue], task_id: str,
                 drive_logs: Optional[pathlib.Path]) -> bool:
    """Append one threshold notice to the round about to be sent and emit its checkpoint."""
    if note is None:
        return False
    note = task_pacing.with_resource_facts(note, tools._ctx, accumulated_usage)
    _loop()._append_or_merge_user_message(messages, note.text)
    _loop()._emit_checkpoint_event(event_queue, task_id, drive_logs, note.checkpoint)
    return True


def _maybe_inject_time_budget_milestone(
    messages: List[Dict[str, Any]],
    tools: ToolRegistry,
    *,
    event_queue: Optional[queue.Queue] = None,
    task_id: str = "",
    drive_logs: Optional[pathlib.Path] = None,
    accumulated_usage: Optional[Dict[str, Any]] = None,
) -> bool:
    """Thin transport over the task_pacing SSOT (v6.54.4): the notice content,
    thresholds, and seen-state live in ouroboros/task_pacing.py; this wrapper only
    appends the notice and emits the checkpoint event."""
    return _inject_note(task_pacing.build_time_budget_note(tools._ctx), messages, tools,
                        accumulated_usage, event_queue, task_id, drive_logs)


def _round_money(accumulated_usage: Dict[str, Any], round_idx: int) -> Dict[str, Any]:
    """This round's wallet and tree readings, shared with the round's facts line."""
    return task_pacing.round_money_observation(
        accumulated_usage, round_idx,
        tree_reader=lambda: _loop()._loop_tree_accounting(
            refresh=True, max_age_sec=_loop()._TREE_ACCOUNTING_MAX_STALE_SEC))


def _maybe_inject_cost_budget_milestone(
    messages: List[Dict[str, Any]],
    tools: ToolRegistry,
    *,
    budget_remaining_usd: Optional[float],
    cost_ceiling: Optional["task_pacing.CostCeiling"],
    accumulated_usage: Optional[Dict[str, Any]],
    round_idx: int = 0,
    active_use_local: bool = False,
    event_queue: Optional[queue.Queue] = None,
    task_id: str = "",
    drive_logs: Optional[pathlib.Path] = None,
) -> bool:
    """Thin transport over the task_pacing cost axes (v6.56.0; the wallet axis #1129).

    The tree axis decides on the tree's KNOWN spend from the root stash (refreshed
    by every dispatch) with a bounded staleness cap; the wallet axis compares the
    shared wallet's current known headroom with ``budget_remaining_usd``, what
    remained when this execution started. Both read this round's one money
    observation (``_round_money``), the same one its facts line renders. A local
    route spends nothing from the wallet, so its wallet axis waits."""
    usage = accumulated_usage if isinstance(accumulated_usage, dict) else {}
    money = _round_money(usage, round_idx)
    tree_info = money.get("tree")
    ceiling_usd = (
        cost_ceiling.ceiling_usd
        if cost_ceiling is not None and cost_ceiling.state == task_pacing.COST_CEILING_ACTIVE
        else None
    )
    task_cost = usage.get("cost")
    tree_note = task_pacing.build_cost_budget_note(
        tools._ctx,
        cost_ceiling_usd=ceiling_usd,
        task_cost=task_cost,
        tree_cost_usd=tree_info.get("settled_usd") if isinstance(tree_info, dict) else None,
        # Whether a tree cap exists at all decides if own cost is the complete
        # picture or a disclosed lower bound (task_pacing.resolve_deciding_spend).
        root_cap_usd=(cost_ceiling.root_cap_usd if cost_ceiling is not None else None),
        whole_work_group=isinstance(tree_info, dict) and tree_info.get("accounting_basis") == "billing_group",
    )
    wallet_note = None if active_use_local else task_pacing.build_wallet_budget_note(
        tools._ctx, baseline_usd=budget_remaining_usd, wallet=money.get("wallet"), task_cost=task_cost)
    injected = [_inject_note(note, messages, tools, accumulated_usage, event_queue, task_id, drive_logs)
                for note in (tree_note, wallet_note)]
    return any(injected)


def _inject_round_checkpoints(
    *,
    round_idx: int,
    messages: List[Dict[str, Any]],
    accumulated_usage: Dict[str, Any],
    tools: ToolRegistry,
    event_queue: Optional[queue.Queue],
    task_id: str,
    drive_logs: Optional[pathlib.Path],
    budget_remaining_usd: Optional[float] = None,
    cost_ceiling: Optional["task_pacing.CostCeiling"] = None,
    active_use_local: bool = False,
    llm_trace: Optional[Dict[str, Any]] = None,
) -> bool:
    """Inject this round's time and cost threshold notices AFTER owner messages, so a
    notice joins the LLM-call tail (a normal user turn, never a paid round of its
    own), then the locators of timed-out calls that settled since (``late_tool_results``):
    this already-authorized send carries them, nothing is scheduled for them.
    Returns whether any was injected."""
    from ouroboros.late_tool_results import deliver_late_results

    time_budget = _maybe_inject_time_budget_milestone(
        messages, tools, event_queue=event_queue, task_id=task_id, drive_logs=drive_logs,
        accumulated_usage=accumulated_usage,
    )
    cost_budget = _maybe_inject_cost_budget_milestone(
        messages, tools,
        budget_remaining_usd=budget_remaining_usd, cost_ceiling=cost_ceiling,
        accumulated_usage=accumulated_usage, round_idx=round_idx, active_use_local=active_use_local,
        event_queue=event_queue, task_id=task_id, drive_logs=drive_logs,
    )
    late = deliver_late_results(messages, tools._ctx, llm_trace, drive_logs, _loop()._append_or_merge_user_message)
    return bool(time_budget or cost_budget or late)


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
        # The forced prompt carries no facts line, so the nanny's own burn since its
        # anchors rides here as a fact (grok / F16) — no threshold, no advice, never a
        # re-loop; with no own rounds or recorded cost since, there is nothing to state,
        # unless an older checkpoint left the anchors unknown (unknown is not zero).
        baseline = getattr(tools_ctx, "_nanny_delegate_baseline", None)
        anchors_unknown = isinstance(baseline, dict) and "cost" not in baseline
        if not anchors_unknown and not any(nanny_metered_since_delegate_activity(tools_ctx)):
            return ""
        return "\nNOTE: this task's delegated run(s) succeeded. " + nanny_burn_fact(tools_ctx) + "."
    if started > settled:
        return (
            "\nNOTE: this task dispatched delegated run(s) that have not settled "
            f"yet ({started - settled} of {started} pending); their results are not in "
            "this transcript."
        )
    if settled:
        return (
            f"\nNOTE: this task's delegated run(s) settled WITHOUT success ({settled} "
            "run(s))."
        )
    if evidence.get("delegate_start_attempted") or any(
        str(c.get("tool") or "") == "delegate_start"
        for c in (llm_trace.get("tool_calls") or []) if isinstance(c, dict)
    ):
        # A durable or current-trace start attempt is not a refusal to delegate.
        return ""
    return (
        "\nNOTE: this task was dispatched onto the delegated substrate "
        "(executor=harness) and made no delegate_start calls; its work ran on "
        "this task's own model calls."
    )


_EFFECT_WORDS = {"commit_reviewed": "commit_reviewed call", "vcs_commit_reviewed": "vcs_commit_reviewed call",
                 "integrate_subagent_patch": "integrated patch", "integrate_delegated_patch": "integrated patch"}


def _changed_fact(llm_trace: Dict[str, Any]) -> str:
    """What the result changed, counted from the one effect projection review and the record read."""
    counts: Dict[str, int] = {}
    for effect in reviewable_effect_projection(llm_trace):
        tool, args = str(effect.get("tool") or ""), effect.get("args") or {}
        label = (_EFFECT_WORDS.get(tool)
                 or (f"written to {args.get('root') or 'active_workspace'}" if tool in _ROOT_WRITE_TOOLS else "")
                 or f"artifact registered by {tool} outputs")
        counts[label] = counts.get(label, 0) + 1
    return "; ".join(f"{label} x{n}" for label, n in counts.items())


def _recorded_fact(drive_root: Any, task_id: str, receipt_rows: List[Dict[str, Any]]) -> str:
    """What this task recorded about its result, by receipt status; a red check with its own words."""
    if not receipt_rows:
        return "no verification receipt"
    status = [str(row.get("status") or "") for row in receipt_rows]
    parts = []
    if status.count("pass"):
        parts.append(f"{status.count('pass')} check(s) passed before the last change")
    if status.count("observed"):
        parts.append(f"{status.count('observed')} path set(s) confirmed to exist")
    kinds = {str(row.get("contract_kind") or "") for row in receipt_rows if row.get("status") == "declared"}
    if "no_visible_machine_contract" in kinds:
        parts.append('"no machine check exists" declared')
    if kinds - {"no_visible_machine_contract"}:
        parts.append("a zero-run decision recorded")
    red = latest_unreconciled_failed_verification(drive_root, task_id, receipts=receipt_rows)
    if red is not None:
        check, code = str(red.get("check") or "").strip(), red.get("returncode")
        matcher = "; exit 0 but the expected text was not found" if code == 0 and red.get("matched") is False else ""
        check = check[:200] + ("…" if len(check) > 200 else "")
        parts.append(f"latest check RED{f' on `{check}`' if check else ''}"
                     f"{f' (exit {code}{matcher})' if code is not None else ''} with no later pass of the same check")
    return "; ".join(parts) or "no verification receipt"


def _reader_fact(ctx: Any) -> str:
    from ouroboros.task_results import resolve_ctx_lineage

    lineage = resolve_ctx_lineage(ctx)
    parent = str(lineage.get("parent_task_id") or "")
    return "your human" if lineage.get("is_root_task") or not parent else f"parent task {parent}"


def _running_children(tools: ToolRegistry, drive_root: Any, task_id: str) -> List[str]:
    """Direct children still running with no decision recorded: they outlive this task's end."""
    from ouroboros.loop_forced_finalization import HANDLED_CHILD_DISPOSITIONS
    from ouroboros.task_status import FINAL_STATUSES

    metadata = getattr(tools._ctx, "task_metadata", None)
    metadata = metadata if isinstance(metadata, dict) else {}
    root = pathlib.Path(str(metadata.get("budget_drive_root") or getattr(tools._ctx, "budget_drive_root", "") or "")
                        or drive_root)
    children = _loop()._load_direct_child_results(root, task_id, str(metadata.get("root_task_id") or task_id))
    return [str(c.get("task_id") or c.get("id") or "?") for c in children
            if str(c.get("status") or "").strip().lower() not in FINAL_STATUSES
            and _loop()._child_disposition_state(c) not in HANDLED_CHILD_DISPOSITIONS]


def _open_delegated_runs(ctx: Any, task_id: str) -> int:
    """This task's own delegated runs that started and have not settled (durable custody, the writers' root)."""
    from ouroboros.delegate_custody import custody_root, task_execution_evidence

    evidence = task_execution_evidence(custody_root(ctx), str(task_id or ""))
    if evidence.get("evidence_read_failed"):
        return 0
    return max(0, int(evidence.get("delegated_runs_started") or 0) - int(evidence.get("delegated_runs_settled") or 0))


def _zero_run_closed(ctx: Any, receipt_rows: List[Dict[str, Any]]) -> bool:
    """A read-only actor that already made its terminal no-leaf decision no longer has the verification tool:
    naming paths it cannot take would be false (the record still discloses the gap)."""
    try:
        from ouroboros.outcomes import _terminal_zero_run_receipt_present
        from ouroboros.tool_access import active_tool_profile

        return active_tool_profile(ctx) == "local_readonly_subagent" and _terminal_zero_run_receipt_present(receipt_rows)
    except Exception:
        return False


def _callable_tools(tools: Any) -> Optional[frozenset]:
    """The tools this actor may call now; None when unreadable, and then the notice withholds no path."""
    try:
        return frozenset(tools.available_tools())
    except Exception:
        return None


def _pre_delivery_notice(tools: ToolRegistry, drive_root: Any, task_id: str, llm_trace: Dict[str, Any],
                         receipt_rows: List[Dict[str, Any]]) -> str:
    """The one finish-door notice (audit 11.11, item 18; owner 12.11: keep the pushback at the exit and make
    acceptance and quality more precise): the facts the host holds before the first ordinary
    delivery, never a gate. It names what applies: a result nothing examined since it last changed
    (``outcomes.result_unexamined``; not where Required's floor reviews it at delivery), direct children still
    running, this task's own delegated runs still open. Typed facts only, never parsed prose (P5); a red check,
    a masking pipe and an agent-defined criterion were said when the check ran and stay on the record."""
    from ouroboros.config import get_task_review_mode
    from ouroboros.tools.review_result import required_root_wave

    lines: List[str] = []
    # A declaration that nothing is machine-checkable does not skip it: an early formal record must not buy the
    # door back (owner 12.11: such a record no longer counts as a check). The notice states the declaration.
    if (result_unexamined(llm_trace) and not required_root_wave(tools._ctx)
            and not _zero_run_closed(tools._ctx, receipt_rows)):
        mode = get_task_review_mode()  # Off refuses a result review; Required reviews only a root at delivery
        callable_tools = _callable_tools(tools)  # a path this actor cannot take is never offered
        can_review = mode != "off" and (callable_tools is None or "review_change" in callable_tools)
        paths = (["check the result against what was asked (verify_and_record runs the check and records what it "
                  "showed)"] if callable_tools is None or "verify_and_record" in callable_tools else [])
        paths += (["ask for a result review (review_change with subject=\"result\" and the seats you choose; its "
                   "verdict comes back before delivery)"] if can_review else [])
        deliver = "deliver it as it is and say in the answer what was not checked, which the record then shows"
        lines += [
            "Before delivery: no recorded check (verify_and_record) or result review has examined this result since it "
            "last changed.",
            f"Changed: {_changed_fact(llm_trace)}.",
            f"Recorded about it: {_recorded_fact(drive_root, task_id, receipt_rows)}.",
            f"This answer is read by: {_reader_fact(tools._ctx)}.",
            "Result review: off on this install (task review mode Off)." if mode == "off" else
            "Result review: none for this answer; this actor has no review_change." if not can_review else
            f"Result review: none for this answer; under {mode.capitalize()} a result review "
            f"{'of a child ' if mode == 'required' else ''}runs only when you ask for one.",
            "Ways forward: " + "; ".join([*paths, f"or {deliver}" if paths else deliver]) + ".",
        ]
    try:
        running = _running_children(tools, drive_root, task_id)
    except Exception:
        log.debug("pre-delivery notice: direct children unreadable", exc_info=True)
        running = []
    if running:
        listed = ", ".join(running[:10]) + (f" (+{len(running) - 10} more)" if len(running) > 10 else "")
        lines.append(f"Still running: direct children {listed}; after this task ends they keep running and spending, "
                     "and it reads nothing more from them (wait for them, cancel them, or leave them).")
    try:
        open_runs = _open_delegated_runs(tools._ctx, task_id)
    except Exception:
        log.debug("pre-delivery notice: delegated custody unreadable", exc_info=True)
        open_runs = 0
    if open_runs:
        lines.append(f"Open delegated runs: {open_runs} of this task's runs have not settled; ending this task now "
                     "cancels them on the next custody pass (delegate_wait shows each run's state).")
    if not lines:
        return ""
    if not lines[0].startswith("Before delivery"):
        lines.insert(0, "Before delivery:")
    return "\n".join([*lines, "Your next answer is delivered as you choose."])


def _claim_pre_delivery_notice(tools: ToolRegistry, drive_root: Any, task_id: str, llm_trace: Dict[str, Any],
                               receipt_rows: Optional[List[Dict[str, Any]]] = None) -> str:
    """The notice once per execution: the first door that shows it (this chain, or the handoff riding the same
    door) latches it, so one door is never two returns."""
    if drive_root is None or getattr(tools._ctx, "_pre_delivery_noticed", False):
        return ""
    if receipt_rows is None:
        from ouroboros.outcomes import read_context_verification_receipts

        receipt_rows = read_context_verification_receipts(tools._ctx, task_id, fallback_root=drive_root)
    notice = _pre_delivery_notice(tools, drive_root, task_id, llm_trace, receipt_rows)
    if notice:
        tools._ctx._pre_delivery_noticed = True
    return notice


def _maybe_inject_finalization_nudges(
    tools: ToolRegistry, drive_root: Optional[pathlib.Path], task_id: str,
    llm_trace: Dict[str, Any], content: Optional[str], messages: List[Dict[str, Any]],
    emit_progress: Callable[[str], None],
) -> bool:
    """One-shot pre-finalization injections that each re-loop (return True): the authoring
    handover fact, the skill finalization reminder, the one pre-delivery notice
    (``_pre_delivery_notice``) and the adapter's answer-protocol marker. Extracted from
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

    def _inject(reminder: str, note: str, cause: str, *, progress: bool = True) -> bool:
        # The one nudge protocol every advisory injection below follows: land
        # the model's pending text, append the reminder, record the note once
        # (live progress AND the durable trail), name the hold's typed cause
        # for the delivery owner's observation, re-loop.
        if content and content.strip():
            messages.append({"role": "assistant", "content": content})
        _loop()._append_or_merge_user_message(messages, f"[SYSTEM REMINDER]\n{reminder}")
        if progress:
            emit_progress(note)
        llm_trace["reasoning_notes"].append(note)
        tools._ctx._completion_hold_cause = cause
        return True

    # A host-driven route change hands an already-active authoring turn to a
    # new model.  The successor sees the canonical transcript, but without this
    # typed reminder its first short status response can look like an ordinary
    # final.  One recovery round is bounded by the existing loop/deadline/budget
    # rails; a second tool-less response is retained with a degraded execution
    # fact rather than silently painting a clean Done.
    handover = (getattr(tools._ctx, "_authoring_handover", None)
                or llm_trace.get("authoring_handover_incomplete"))
    if isinstance(handover, dict):
        baseline = int(handover.get("tool_calls_at_handover") or 0)
        from ouroboros.tool_capabilities import substantive_tool_calls
        if substantive_tool_calls((llm_trace.get("tool_calls") or [])[baseline:]):
            handover["status"] = "recovered"
            incomplete = llm_trace.pop("authoring_handover_incomplete", None)
            if isinstance(incomplete, dict):
                incomplete["status"] = "recovered"
            tools._ctx._authoring_handover = None
            usage = getattr(tools._ctx, "_accumulated_usage", {})
            if usage.get("reason_code") == "authoring_handover_incomplete":
                usage.pop("execution_status", None)
                usage.pop("reason_code", None)
        elif baseline > 0 and content and str(content).strip():
            if not bool(handover.get("recovery_prompted")):
                handover["recovery_prompted"] = True
                from_model = str(handover.get("from_model") or "the previous model")
                to_model = str(handover.get("to_model") or "the current model")
                return _inject(
                    f"A host-driven model handover occurred ({from_model} → {to_model}) "
                    "after the previous model had already used tools on this open task; the "
                    "plan and tool results above are that earlier work. No substantive tool "
                    "call has run since the handover. If the turn ends without one, the host "
                    "records this handover as incomplete (execution degraded); later tool "
                    "work clears that record.",
                    "Authoring handover recovery nudge injected before final response.", "authoring_handover",
                )
            tools._ctx._authoring_handover = None
            handover["status"] = "incomplete"
            handover["incomplete_observed"] = True
            usage = getattr(tools._ctx, "_accumulated_usage", None)
            if isinstance(usage, dict):
                usage["execution_status"] = "degraded"
                usage["reason_code"] = "authoring_handover_incomplete"
            llm_trace["authoring_handover_incomplete"] = handover
            # Existing one-shot readiness/verification nudges and acceptance
            # can still continue the task. Later tool work heals only this
            # warning; the history row retains the handover and its recovery.

    from ouroboros.tool_access import canonical_data_root

    finalization_msg = _loop()._skill_finalization_message(canonical_data_root(tools._ctx), llm_trace)
    if finalization_msg and not getattr(tools._ctx, "_skill_finalization_injected", False):
        tools._ctx._skill_finalization_injected = True
        return _inject(finalization_msg, finalization_msg, "skill_not_finalized")
    notice = _claim_pre_delivery_notice(tools, drive_root, task_id, llm_trace, receipt_rows)
    if notice:
        # A nanny's notice stays out of the owner chat's progress (owner 2026-08-15: nanny chat spam removed).
        return _inject(notice, "Pre-delivery facts shown before final response.", "pre_delivery_notice",
                       progress=not getattr(tools._ctx, "_nanny_route_dispatched", False))
    # No "expected output but nothing attempted" re-loop (A3, removed for #1748): plain
    # prose can itself be the requested deliverable, and the transcript already shows
    # the model its own calls.
    from ouroboros.tool_capabilities import substantive_tool_calls
    work_calls = substantive_tool_calls(llm_trace.get("tool_calls"))
    # P2 one-shot final-answer-marker nudge: real work + prose, no FINAL ANSWER
    # marker — the agent marks its OWN answer, prose is never mined (P5). Own
    # latch after verify/red; the protocol gate alone suffices (v6.56.0).
    if (
        not getattr(tools._ctx, "_final_marker_nudged", False)
        and _answer_protocol_active(tools._ctx)  # v6.60.0: marker nudge is protocol-gated
        and content and content.strip()
        and not extract_final_answer(content or "")
        and (work_calls or turn_has_reviewable_effects(llm_trace))
    ):
        tools._ctx._final_marker_nudged = True
        return _inject(
            "This task's answer protocol (final_answer_line) captures an answer only from a line "
            "exactly `FINAL ANSWER: <answer>` — the bare deliverable (a number / a few words / a "
            "short list) — and the response above has none. The latest such line remains eligible "
            "for interrupted-run delivery only while no later tool call has made it stale.",
            "Final-answer marker nudge injected before final response.", "final_answer_marker",
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
