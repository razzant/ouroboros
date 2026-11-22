"""Host notices injected into the transcript: one-time time and cost threshold
crossings, plan-forcing prompts, skill-finalization facts, one-shot finalization
facts and the answer protocol. Extracted from loop.py (v7 L-B split); loop.py
re-exports every name. Notices state observations and the available actions;
periodic questions and strategy advice were removed (owner 11 Oct, #1748).
Every-round facts ride the facts line (``loop_messages``)."""

from __future__ import annotations

import json
import logging
import pathlib
import queue

from itertools import groupby, takewhile
from typing import Any, Callable, Dict, List, Optional
from ouroboros import task_pacing
from ouroboros.nanny_pacing import nanny_burn_fact, nanny_metered_since_delegate_activity
from ouroboros._outcome_receipts import red_check_fact
from ouroboros.outcomes import (
    _OK_TOOL_STATUSES, _ROOT_WRITE_TOOLS, extract_final_answer, result_unexamined,
    reviewable_effect_projection, turn_has_reviewable_effects,
)
# D18b: the trace-touched skill-name scan lives with the readiness predicate that
# consumes it (upstream 0463c6bb); the historical private name stays bound here so
# loop.py's re-export and every existing caller keep addressing one object.
from ouroboros.skill_readiness import skill_names_touched_by_trace as _skill_names_touched_by_trace  # noqa: F401 -- historical name
from ouroboros.tool_error_words import check_cause, red_check_phrase
from ouroboros.tool_policy import BUILTIN_NAMESPACE
from ouroboros.tool_result_record import TOOL_RESULT_RECORD_KEY
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
            "\nNOTE: collection/accounting has not settled for "
            f"{started - settled} of {started} delegated run(s)."
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


def _named_red_check(red: Dict[str, Any]) -> str:
    """The red check for the mind alone: the command and its cause in the words of the owner's sentence, or the
    host's own words where it looked for the named files and ran no command."""
    if red["kind"] == "artifact_observation":
        return f"path observation — {red['check'] or check_cause(red)}"
    return f"`{red['check']}` — {check_cause(red)}" if red.get("check") else check_cause(red)


def _red_check_words(red: Dict[str, Any]) -> str:
    """The red check as the notice says it, and whether any check passed after it (``_outcome_receipts.red_check_fact``)."""
    if not red["passed_after"]:
        return f"the last recorded check did not pass ({_named_red_check(red)}), and none has passed since"
    return (f"a check did not pass ({_named_red_check(red)}); {red['passed_after']} passed after it, "
            "none recorded as a rerun of it")


def _recorded_fact(receipt_rows: List[Dict[str, Any]], red: Optional[Dict[str, Any]]) -> str:
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
    if red:
        parts.append(_red_check_words(red))
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
    """This task's own delegated runs with no recorded end (durable custody, the writers' root)."""
    from ouroboros.delegate_custody import TERMINAL_STATES, custody_root
    from ouroboros.delegate_custody_current import runs

    return sum(1 for entry in runs(custody_root(ctx), "task_id=?", (str(task_id or ""),)).values()
               if not entry.review_owned and not entry.settled and entry.terminal_state not in TERMINAL_STATES)


def _zero_run_closed(ctx: Any, receipt_rows: List[Dict[str, Any]]) -> bool:
    """A read-only actor that already made its terminal no-leaf decision no longer has the verification tool:
    naming paths it cannot take would be false (the record still discloses the gap)."""
    try:
        from ouroboros.outcomes import _terminal_zero_run_receipt_present
        from ouroboros.tool_access import active_tool_profile

        return active_tool_profile(ctx) == "local_readonly_subagent" and _terminal_zero_run_receipt_present(receipt_rows)
    except Exception:
        return False


RED_CHECK_NOTICE = "red_check_notice"  # the hold cause of the red check stated alone; beside the other facts the hold is theirs


def _pre_delivery_facts(tools: ToolRegistry, drive_root: Any, task_id: str, llm_trace: Dict[str, Any],
                        receipt_rows: List[Dict[str, Any]], told: bool = False, red_told: bool = False) -> tuple:
    """The finish-door notice and the two parts it states, ``(text, facts, red)``: never a gate, typed facts only,
    never parsed prose (P5). ``facts`` (owner 12.11: keep the pushback at the exit and make acceptance and quality
    more precise): a result nothing examined since it last changed (``outcomes.result_unexamined``; not where
    Required's floor reviews it at delivery), direct children still running, this task's own delegated runs with no
    recorded end. ``red`` (owner 15.11: on a failure, a chance to correct it): the last recorded check did not pass
    and no check has passed since (``_outcome_receipts.red_check_fact``), for every actor and review mode; a masking
    pipe and an agent-defined criterion were said when the check ran and stay on the record. A part already told
    (``told``, ``red_told``) is left out."""
    from ouroboros.config import get_task_review_mode
    from ouroboros.tools.review_result import required_root_wave

    lines: List[str] = []
    red = red_check_fact(receipt_rows)
    red_due = bool(red) and not red["passed_after"] and not red_told
    try:  # a path this actor cannot call is never offered (a level's withheld name stays shown, В31=B); unreadable: none
        callable_tools = frozenset(row["name"] for row in tools.callable_rows(BUILTIN_NAMESPACE))
    except Exception:
        callable_tools = None
    can_check = callable_tools is None or "verify_and_record" in callable_tools
    # A declaration that nothing is machine-checkable does not skip it: an early formal record must not buy the
    # door back (owner 12.11: such a record no longer counts as a check). The notice states the declaration.
    unexamined = (not told and result_unexamined(llm_trace) and not required_root_wave(tools._ctx)
                  and not _zero_run_closed(tools._ctx, receipt_rows))
    if unexamined:
        mode = get_task_review_mode()  # Off refuses a result review; Required reviews only a root at delivery
        can_review = mode != "off" and (callable_tools is None or "review_change" in callable_tools)
        paths = (["check the result against what was asked (verify_and_record runs the check and records what it "
                  "showed)"] if can_check else [])
        paths += (["ask for a result review (review_change with subject=\"result\" and the seats you choose; its "
                   "verdict comes back before delivery)"] if can_review else [])
        deliver = "deliver it as it is and say in the answer what was not checked, which the record then shows"
        lines += [
            "Before delivery: no recorded check (verify_and_record) or result review has examined this result since it "
            "last changed.",
            f"Changed: {_changed_fact(llm_trace)}.",
            f"Recorded about it: {_recorded_fact(receipt_rows, red)}.",
            f"This answer is read by: {_reader_fact(tools._ctx)}.",
            "Result review: off on this install (task review mode Off)." if mode == "off" else
            "Result review: none for this answer; this actor has no review_change." if not can_review else
            f"Result review: none for this answer; under {mode.capitalize()} a result review "
            f"{'of a child ' if mode == 'required' else ''}runs only when you ask for one.",
            "Ways forward: " + "; ".join([*paths, f"or {deliver}" if paths else deliver]) + ".",
        ]
    elif red_due:  # beside an unexamined result the red check is said once, in "Recorded about it"
        lines += [
            "Before delivery: the last recorded check did not pass, and no recorded check has passed since.",
            f"Check: {_named_red_check(red)}.",
            f"This answer is read by: {_reader_fact(tools._ctx)}; if you finish now, a clean ending's details will say: "
            f"\"{red_check_phrase({'verification_red': red})}\"",
            "Ways forward: " + ("change the result or the check, or run the check again as it is (verify_and_record); or "
                                if can_check else "change the result; or ")
            + "deliver the answer as it is and say in it what did not pass.",
        ]
    count = len(lines)
    if not told:
        try:
            running = _running_children(tools, drive_root, task_id)
        except Exception:
            log.debug("pre-delivery notice: direct children unreadable", exc_info=True)
            running = []
        if running:
            listed = ", ".join(running[:10]) + (f" (+{len(running) - 10} more)" if len(running) > 10 else "")
            lines.append(f"Still running: direct children {listed}; after this task ends they keep running and "
                         "spending, and it reads nothing more from them (wait for them, cancel them, or leave them).")
        try:
            open_runs = _open_delegated_runs(tools._ctx, task_id)
        except Exception:
            log.debug("pre-delivery notice: delegated custody unreadable", exc_info=True)
            open_runs = 0
        if open_runs:
            lines.append(f"Open delegated execution: {open_runs} of this task's runs have no recorded end; "
                         "ending this task now cancels those still running on the next custody pass "
                         "(delegate_wait shows each run's state).")
    if not lines:
        return "", False, False
    if not lines[0].startswith("Before delivery"):
        lines.insert(0, "Before delivery:")
    text = "\n".join([*lines, "Your next answer is not returned for these facts."])
    return text, unexamined or len(lines) > count, red_due


def _pre_delivery_notice(tools: ToolRegistry, drive_root: Any, task_id: str, llm_trace: Dict[str, Any],
                         receipt_rows: List[Dict[str, Any]]) -> str:
    """The whole finish-door notice of this moment, nothing told yet (``_pre_delivery_facts``)."""
    return _pre_delivery_facts(tools, drive_root, task_id, llm_trace, receipt_rows)[0]


def _claim_pre_delivery_notice(tools: ToolRegistry, drive_root: Any, task_id: str, llm_trace: Dict[str, Any],
                               receipt_rows: Optional[List[Dict[str, Any]]] = None) -> str:
    """The notice at this door: each of its two parts returns an answer once per execution, and the door that shows a
    part (this chain, or the handoff riding the same door) latches it, so one door is never two returns and the
    notice returns an answer at most twice, once for each part. It names the hold's cause for the delivery owner
    (``_completion_hold_cause``)."""
    ctx = tools._ctx
    told, red_told = bool(getattr(ctx, "_pre_delivery_noticed", False)), bool(getattr(ctx, "_red_check_noticed", False))
    if drive_root is None or (told and red_told):
        return ""
    if receipt_rows is None:
        from ouroboros.outcomes import read_context_verification_receipts

        receipt_rows = read_context_verification_receipts(ctx, task_id, fallback_root=drive_root)
    notice, facts, red = _pre_delivery_facts(tools, drive_root, task_id, llm_trace, receipt_rows, told, red_told)
    if notice:
        ctx._pre_delivery_noticed, ctx._red_check_noticed = told or facts, red_told or red
        ctx._completion_hold_cause = "pre_delivery_notice" if facts else RED_CHECK_NOTICE
    return notice


def _end_authoring_handover(ctx: Any, llm_trace: Dict[str, Any], handover: Dict[str, Any], status: str,
                            **facts: Any) -> None:
    """Record the successor's act on its own row. Earlier switches keep their history."""
    handover.update(status=status, **facts)
    llm_trace.pop("authoring_handover_incomplete", None)
    ctx._authoring_handover = None
    usage = getattr(ctx, "_accumulated_usage", None)
    if isinstance(usage, dict) and usage.get("reason_code") == "authoring_handover_incomplete":
        if usage.get("execution_status") == "degraded":
            usage.pop("execution_status")
        usage.pop("reason_code", None)


def _working_calls(calls: Any) -> list:
    """The calls that show a successor working: successful, neither a completion request nor schema discovery."""
    from ouroboros.tool_capabilities import SCHEMA_DISCOVERY_TOOLS, substantive_tool_calls

    return [call for call in substantive_tool_calls(calls) if call.get("is_error") is False
            and call.get("status") in _OK_TOOL_STATUSES and call.get("tool") not in SCHEMA_DISCOVERY_TOOLS]


def observe_handover_work(ctx: Any, trace: dict, calls: list) -> None:
    """The completed batch belongs to this author, even if the next round switches or ends."""
    handover = getattr(ctx, "_authoring_handover", None)
    if handover and _working_calls(calls):
        _end_authoring_handover(ctx, trace, handover, "recovered", work_observed=True)


def _worked_handover(ctx: Any, trace: dict, request: dict) -> Optional[dict]:
    """The row this request's own response ended by its work, for the choice the same response made.

    A batch ends the handover (``observe_handover_work``) before the delivery gates accept that response's
    completion request. The request names its response by the tool count it was staged under; the row is that
    response's only when the first successful work since the handover lies in the response's own round. A request
    held and then selected again, a later reply, another round's work and a forced reply (which stages nothing)
    name no such row, and an execution that holds the row only as another one's history has none to name."""
    rows = (getattr(ctx, "_accumulated_usage", None) or {}).get("authoring_handovers")
    routes, calls = trace.get("route_handovers"), trace.get("tool_calls") or []
    row = rows[-1] if isinstance(rows, list) and rows and isinstance(routes, list) and routes else None
    start = (request.get("observation") or {}).get("tool_count")
    if (not isinstance(row, dict) or row is not routes[-1] or row.get("status") != "recovered"
            or row.get("work_observed") is not True or type(start) is not int
            or type(baseline := row.get("tool_calls_at_handover")) is not int
            or not baseline <= start < len(calls) or not isinstance(calls[start], dict)):
        return None
    round_id = calls[start].get("round_id")
    own = [call for call in calls[start:] if isinstance(call, dict) and call.get("round_id") == round_id]
    return row if _working_calls(own) and not _working_calls(calls[baseline:start]) else None


def _solve_round(call: Any, execution: str, through: int) -> int:
    """Logical round of a usable solve, not a provider attempt or a forced reply."""
    if not isinstance(call, dict):
        return 0
    number = call.get("round")
    return number if (
        type(number) is int and 0 < number <= through and call.get("execution_id") == execution
        and call.get("round_id") == f"{execution}:round:{number}" and call.get("usable_solve_response") is True
        and isinstance(call.get("llm_call_id"), str) and call["llm_call_id"]
        and isinstance(call.get("model"), str) and call["model"].strip()
    ) else 0


def _runs(names: Any) -> list:
    """The order in which the name changed: consecutive repeats are one."""
    return [name for name, _ in groupby(names)]


def _receipt_authorship(state: dict, calls: list, handover: dict) -> tuple:
    """``(last usable reply, the handover's successor rounds)`` as the saved execution's own call receipts say.

    One source answers at every saved boundary: each usable solve keeps its receipt (execution, logical round,
    model) and each tool result names the round that asked for it. The rounds are the target's last series of
    replies, entered from the row's predecessor, read round after round from the predecessor's last tool round
    (the anchor); a round between two solves counts only when every attempt it recorded failed (a provider's
    refusal, an exhausted reply), or when it recorded none (a view refused before sending leaves no receipt) and
    the accepted routes, kept apart from the receipts, name the same authors in the same order. ``set()``: the
    target has not answered yet. ``None``: the receipts cannot say: a round neither record accounts for, a solve
    out of order, a predecessor the row does not name, or another author since. Returning to the same name
    reopens no row: where the target answered in an earlier series too, the last series is this row's only when
    the rows written at this same tool count name the receipts' last changes of author one for one and no
    earlier change is the same pair (each host switch after tools writes its own row; a switch the mind made
    writes none). Counts bound the slice, not its authorship. Old call rows stamp the execution and logical
    round, not the task attempt (``attempt`` is the provider retry): the verified checkpoint owns the task
    identity, so its usage's attempt must be its envelope's.
    """
    usage = state["usage"]
    attempt, through = state.get("task_attempt"), state.get("round_idx")
    execution, refs = usage.get("execution_id"), usage.get("llm_call_refs")
    if (type(attempt) is not int or attempt < 1 or type(usage.get("_task_attempt")) is not int
            or usage["_task_attempt"] != attempt or type(through) is not int
            or not isinstance(refs, list) or not isinstance(execution, str) or not execution):
        return {}, None
    refs = [ref for ref in refs if isinstance(ref, dict)]
    solves = [ref for ref in refs if ref.get("usable_solve_response") is True]
    if not solves or not all(_solve_round(ref, execution, through) for ref in solves):
        return {}, None
    reply, baseline = solves[-1], handover.get("tool_calls_at_handover")
    if type(baseline) is not int or not 0 < baseline <= len(calls) or not isinstance(calls[baseline - 1], dict):
        return reply, None
    failed: Dict[Any, bool] = {}  # a round whose every recorded attempt failed, none naming another execution
    for ref in refs:
        failed[ref.get("round")] = (failed.get(ref.get("round"), True) and bool(ref.get("failure_code"))
                                    and ref.get("execution_id") in (None, execution))
    accepted = usage.get("accepted_routes")  # written at each accepted reply whose route differs from the last one's
    routes = [row["to_binding"][0] for row in (accepted if isinstance(accepted, list) else ())
              if isinstance(row, dict) and isinstance(row.get("to_binding"), (list, tuple)) and row["to_binding"]]
    unsent = _runs(routes) == _runs(ref["model"] for ref in solves)  # no reply is missing from the receipts
    pair, anchor = (handover.get("from_model"), handover.get("to_model")), calls[baseline - 1].get("round_id")
    span, last, previous, met, turns = set(), 0, None, False, []  # turns: each change of author from the anchor on
    for ref in solves:
        if not last:
            if not anchor or ref["round_id"] != anchor:
                continue
        elif ref["round"] <= last or not all(failed.get(gap, unsent) for gap in range(last + 1, ref["round"])):
            return reply, None
        elif ref["model"] != previous:
            turns.append((previous, ref["model"]))
            span = set()
        if last and ref["model"] == pair[1]:  # the anchor itself is a tool round from before the handover
            met = True
            span.add(ref["round_id"])
        last, previous = ref["round"], ref["model"]
    if not last or turns[-1:] != [pair]:
        return reply, None if met or not last else set()
    if sum(turn[1] == pair[1] for turn in turns) > 1:  # several series: which departure the last row names
        rows = usage.get("authoring_handovers")
        if not isinstance(rows, list) or not rows or rows[-1] != handover:
            return reply, None
        named = [(row.get("from_model"), row.get("to_model")) for row in takewhile(
            lambda row: isinstance(row, dict) and row.get("tool_calls_at_handover") == baseline, reversed(rows))][::-1]
        if turns[-len(named):] != named or pair in turns[:-len(named)]:
            return reply, None
    return reply, span


_NO_DECLARATION = object()  # this spelling proves nothing of the call's arguments; keep its recorded copy.


def _declared_arguments(spelling: Any) -> Any:
    """The one object a declared ``arguments`` spelling holds, or ``_NO_DECLARATION``.

    One decoder for both readers of a saved batch. Text that does not parse, a JSON value that is not an
    object, and a save that wrote the object itself where its JSON text belongs are the same gap: they name
    no request. Shape is checked before it is parsed because ``json.loads`` answers a spelling that is not
    text with ``TypeError``, which a ``ValueError`` reader would not catch, and a resume is not the place to
    learn a save's shape from an exception."""
    if not isinstance(spelling, (str, bytes, bytearray)):
        return _NO_DECLARATION
    try:
        arguments = json.loads(spelling)
    except ValueError:
        return _NO_DECLARATION
    return arguments if isinstance(arguments, dict) else _NO_DECLARATION


def _recorded_call_arguments(call: Any, messages: Any) -> Any:
    """One recorded call's arguments as the transcript's one unambiguous naming of its ID and tool holds them.

    The fallback for saves written before the executor stamped the result record. Providers reuse a call ID
    between responses (tool_call_log), and those saves carry no axis saying which round spelled a naming: the
    first match could hand an earlier round's complete choice to this batch, and no compatibility rule ever
    required that pick. So a second naming of one ID, or one that cannot be decoded, ends the search with the
    call's own recorded copy — what those restores were proven to settle on before any transcript was read
    for it. ``_transcript_arguments`` binds the record era to the call's own batch instead. Nothing absent is
    inferred and no disk is read; the trace's rendered ``args`` stay the last resort."""
    call_id, tool = call.get("tool_call_id"), call.get("tool")
    if not isinstance(call_id, str) or not call_id:
        return call.get("args")  # a command that names no ID matches nothing anywhere in the transcript
    found = _NO_DECLARATION
    for message in messages if isinstance(messages, list) else ():
        for recorded in (message.get("tool_calls") or ()) if isinstance(message, dict) else ():
            function = recorded.get("function") if isinstance(recorded, dict) else None
            if (isinstance(function, dict) and recorded.get("id") == call_id
                    and function.get("name") == tool):
                if found is not _NO_DECLARATION:
                    return call.get("args")  # two namings: the save cannot say which round spelled them
                spelling = _declared_arguments(function.get("arguments"))
                if spelling is _NO_DECLARATION:
                    return call.get("args")  # this naming cannot be resolved: it proves nothing either
                found = spelling
    return call.get("args") if found is _NO_DECLARATION else found


def _batch_declaration(row: Any, tool_call_id: Any, tool: Any) -> Any:
    """The complete arguments one assistant batch declares under one call ID: a lone naming of this tool.

    Named twice in its own response, named by another command, unreadable or absent is no proof: ``_NO_DECLARATION``."""
    found = _NO_DECLARATION
    declarations = row.get("tool_calls") if isinstance(row, dict) else None
    for declared in declarations if isinstance(declarations, list) else ():
        function = declared.get("function") if isinstance(declared, dict) else None
        if not (isinstance(declared, dict) and declared.get("id") == tool_call_id and isinstance(function, dict)):
            continue
        if found is not _NO_DECLARATION or function.get("name") != tool:
            return _NO_DECLARATION
        found = _declared_arguments(function.get("arguments"))
        if found is _NO_DECLARATION:
            return _NO_DECLARATION
    return found


def _transcript_arguments(controls: list, messages: Any) -> list:
    """For each recorded command of one round (the caller checked their ``round_id``): the arguments as its own
    batch declares them in the transcript, or the call's recorded copy — never another round's spelling.

    The executor stamps each result row with the round it ran that call in (``tool_result_record``); the batch
    whose result records name this round is the batch to read, so a provider ID reused in an earlier valid
    response never lends its words. The walk starts at the transcript's end and stops once that batch closes:
    only the needed suffix is read. A call the records do not prove — a result naming another round, a record
    whose own invocation names another call or none (the gap ``read_tool_result_record`` reports as a
    mismatch), a record disagreeing with its row, a doubled or unreadable declaration — keeps its recorded
    ``args``: a broken link grants no authorship and breaks no restore. A save whose rows carry no record at
    all (the format before the stamp) keeps the unambiguous name match above. The trace renders ``args`` with
    truncation markers, so two spellings of one long choice can differ there; the transcript keeps the
    complete JSON the provider sent, which is what the live staging compared."""
    round_id = controls[0].get("round_id") if controls else None
    if not isinstance(messages, list) or not isinstance(round_id, str) or not round_id:
        return [_recorded_call_arguments(call, messages) for call in controls]
    wanted: Dict[Any, list] = {}
    for position, call in enumerate(controls):
        call_id = call.get("tool_call_id")
        if isinstance(call_id, str) and call_id:
            wanted.setdefault(call_id, []).append(position)
    decided, episode, stamped = {}, [], False
    for index in range(len(messages) - 1, -1, -1):
        row = messages[index]
        if not isinstance(row, dict):
            continue
        role = str(row.get("role") or "").lower()
        if role == "tool":
            record = row.get(TOOL_RESULT_RECORD_KEY)
            if not isinstance(record, dict):
                continue
            stamped = True
            invocation = record.get("invocation")
            call_id = row.get("tool_call_id")
            if (isinstance(invocation, dict) and invocation.get("round_id") == round_id
                    and invocation.get("tool_call_id") == call_id and call_id in wanted):
                episode.append(call_id)
            continue
        if role == "assistant" and episode:
            for call_id in dict.fromkeys(episode):  # one batch, however its rows repeat
                for position in wanted[call_id]:
                    arguments = _batch_declaration(row, call_id, controls[position].get("tool"))
                    decided[position] = controls[position].get("args") if arguments is _NO_DECLARATION else arguments
            break  # this was the round's own batch: whatever remains unproven keeps its copy, nobody else's.
    if not stamped:
        return [_recorded_call_arguments(call, messages) for call in controls]
    return [decided[position] if position in decided else call.get("args")
            for position, call in enumerate(controls)]


def _restore_completion_author(ctx: Any, observation: dict, calls: list, reply: dict, messages: Any) -> None:
    """The producer copies the round observation into a request; repair only its own completed batch's copy."""
    from ouroboros.tools.control_runtime import staged_request_identity

    index = observation.get("tool_count")
    if type(index) is not int or not 0 <= index < len(calls):
        return
    batch = calls[index:]
    if any(not isinstance(call, dict) or call.get("round_id") != reply["round_id"] for call in batch):
        return
    controls = [call for call in batch if call.get("completion_control") is True]
    said = [staged_request_identity(ctx, call.get("tool"), arguments)
            for call, arguments in zip(controls, _transcript_arguments(controls, messages))]
    # A response may repeat its one accepted command under several call IDs: staging admits the request it already
    # holds, however the repeat spells it. A refused or conflicting sibling, another command or another request:
    # the batch does not say one thing.
    if not controls or any(call.get("is_error") is not False or call.get("status") not in _OK_TOOL_STATUSES
                           or identity != said[0] for call, identity in zip(controls, said)):
        return
    for key in ("_completion_request", "_completion_selected"):
        request = getattr(ctx, key, None)
        snapshot = request.get("observation") if isinstance(request, dict) else None
        if (isinstance(snapshot, dict) and not snapshot.get("model") and request.get("reply_later") is False
                and request.get("source") == controls[0].get("tool")
                and {**snapshot, "model": None} == {**observation, "model": None}):
            setattr(ctx, key, {**request, "observation": {**snapshot, "model": reply["model"]}})


def restore_authoring_handover(ctx: Any, state: dict, trace: dict, usage: dict) -> None:
    """Relink only this task's final history row and settle what its saved facts establish, once on restore.

    A row this body wrote says itself that its target replied, beside the saved author. An older row says neither:
    the execution's own receipts answer for it (``_receipt_authorship``), and what they leave open stays open.
    Where no receipt can say anything, a saved author naming the target is still this body's record of its reply."""
    if not state.get("task_id") or state["task_id"] != ctx.task_id:
        return
    routes, rows = trace.get("route_handovers"), usage.get("authoring_handovers")
    if (not isinstance(routes, list) or not routes or not isinstance(row := routes[-1], dict)
            or not isinstance(rows, list) or not rows or rows[-1] != row):
        return
    rows[-1] = row  # JSON broke the live alias; earlier rows remain history.
    incomplete = row.get("status") == "incomplete" and trace.get("authoring_handover_incomplete") == row
    if incomplete:
        trace["authoring_handover_incomplete"] = row
    if row.get("status") != "pending" and not incomplete:
        return
    observation = state["delivery"].get("_completion_observation") or {}
    target, author, calls = row.get("to_model"), observation.get("model"), trace.get("tool_calls") or []
    reply, span = {}, None
    if row.get("reply_observed") is not True or not author:
        reply, span = _receipt_authorship(state, calls, row)
        if reply and not author:
            ctx._completion_observation = {**observation, "model": (author := reply["model"])}
    if not target or author not in (None, target) or not (
            row.get("reply_observed") is True or span or not reply and author == target):
        return  # A route pair alone proves no reply; its target's next one may still attach it.
    ctx._authoring_handover = row
    if span:
        row["reply_observed"] = True
        _restore_completion_author(ctx, observation, calls, reply, state.get("messages"))
        observe_handover_work(ctx, trace, [call for call in calls[row["tool_calls_at_handover"]:]
                                          if isinstance(call, dict) and call.get("round_id") in span])


def accept_handover_selection(ctx: Any, trace: dict, request: dict | None = None, *, forced_model: str = "") -> None:
    """An accepting exit records the choice of the model that made it, also on the row that the work of the same
    response already ended (``_worked_handover``): acceptance, never staging, puts a choice on a row."""
    if request is None:
        request = getattr(ctx, "_completion_selected", None) or {}
    handover = getattr(ctx, "_authoring_handover", None) or _worked_handover(ctx, trace, request)
    author = forced_model or (request.get("observation") or {}).get("model")
    if not handover or not author or author != handover.get("to_model"):
        return
    if request.get("action") == "stop":
        kind = "stop"
    elif request.get("wake_silent"):
        kind = "silent"
    elif request.get("action") == "finish" and (forced_model or handover.get("recovery_prompted")):
        kind = "answer_sha256" if request.get("answer_sha256") else "answer"
    else:
        return
    _end_authoring_handover(ctx, trace, handover, "selected", selection=kind)


def accept_preserved_handover_stop(ctx: Any, trace: dict) -> None:
    """A rail retaining the current candidate keeps its validated author stop, not its outcome."""
    request = getattr(ctx, "_completion_selected", None) or {}
    if request.get("action") == "stop":
        accept_handover_selection(ctx, trace, request)


def _note_incomplete_handover(ctx: Any, llm_trace: Dict[str, Any], handover: Dict[str, Any]) -> None:
    """Text after the one reminder is no choice. The mark names no author; another failure keeps its own code."""
    handover.update(status="incomplete", incomplete_observed=True)
    usage = getattr(ctx, "_accumulated_usage", None)
    if isinstance(usage, dict) and not any(usage.get(key) for key in ("execution_status", "result_status", "reason_code")):
        usage.update(execution_status="degraded", reason_code="authoring_handover_incomplete")
    llm_trace["authoring_handover_incomplete"] = handover


def _authoring_handover_step(ctx: Any, llm_trace: Dict[str, Any], content: Optional[str], restored: bool) -> str:
    """Offer one reminder before an ordinary finish. A selection can still be held downstream."""
    handover = getattr(ctx, "_authoring_handover", None)
    selected = getattr(ctx, "_completion_selected", None) or {}
    if not isinstance(handover, dict):
        # Restore links matching usage/trace rows only for this task. A pending legacy row
        # may leave the draft's author unknown; asking for a choice names no successor act.
        rows = (getattr(ctx, "_accumulated_usage", None) or {}).get("authoring_handovers") or []
        routes = llm_trace.get("route_handovers") or []
        row = rows[-1] if rows and routes and rows[-1] is routes[-1] else None
        if selected or not isinstance(row, dict) or row.get("status") != "pending" or row.get("reply_observed") is True:
            return ""
        if row.get("recovery_prompted"):
            # Asked once already, here or by the body that saved it. The row stays unbound: its author is unproven.
            if content and str(content).strip():
                _note_incomplete_handover(ctx, llm_trace, row)
        elif restored:
            row["recovery_prompted"] = True
            return ("This answer was retained around a model handover; the saved record does not confirm its author. "
                    "Continue the work, or explicitly select the complete answer to deliver, by text or answer_sha256.")
        return ""
    # Inherited quiet delivery is not a completion request. A saved choice by an unknown or
    # different author can still deliver, but does not answer this handover's question.
    if selected and (selected.get("wake_silent") or handover.get("recovery_prompted")
                     or (selected.get("observation") or {}).get("model") != handover.get("to_model")):
        return ""
    if content and str(content).strip():
        if not handover.get("recovery_prompted"):
            handover["recovery_prompted"] = True
            from_model = str(handover.get("from_model") or "the previous model")
            to_model = str(handover.get("to_model") or "the current model")
            return (f"A host-driven model handover occurred ({from_model} → {to_model}). The previous model's "
                    "tool results remain in the transcript. Your response above is retained and interim prose "
                    "stays held. Continue the work, or select the answer to deliver using its complete text "
                    "or an answer_sha256 offered below.")
        _note_incomplete_handover(ctx, llm_trace, handover)
    return ""


def _maybe_inject_finalization_nudges(
    tools: ToolRegistry, drive_root: Optional[pathlib.Path], task_id: str,
    llm_trace: Dict[str, Any], content: Optional[str], messages: List[Dict[str, Any]],
    emit_progress: Callable[[str], None],
    *, restored: bool = False,
) -> bool:
    """One-shot pre-finalization injections that each re-loop (return True): the authoring
    handover fact, the skill finalization reminder, the pre-delivery notice (``_claim_pre_delivery_notice``: each
    of its two parts once) and the adapter's answer-protocol marker. Extracted from
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

    reminder = _authoring_handover_step(tools._ctx, llm_trace, content, restored)
    if reminder:
        return _inject(reminder, "Authoring handover recovery nudge injected before final response.",
                       "authoring_handover")
    # The other one-shot facts below still reach a successor: settling the handover settles only its own question.

    from ouroboros.tool_access import canonical_data_root

    finalization_msg = _loop()._skill_finalization_message(canonical_data_root(tools._ctx), llm_trace)
    if finalization_msg and not getattr(tools._ctx, "_skill_finalization_injected", False):
        tools._ctx._skill_finalization_injected = True
        return _inject(finalization_msg, finalization_msg, "skill_not_finalized")
    notice = _claim_pre_delivery_notice(tools, drive_root, task_id, llm_trace, receipt_rows)
    if notice:
        # A nanny's notice stays out of the owner chat's progress (owner 2026-08-15: nanny chat spam removed).
        cause = getattr(tools._ctx, "_completion_hold_cause", "") or "pre_delivery_notice"
        return _inject(notice, "A check that did not pass was shown before final response." if cause == RED_CHECK_NOTICE
                       else "Pre-delivery facts shown before final response.", cause,
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
