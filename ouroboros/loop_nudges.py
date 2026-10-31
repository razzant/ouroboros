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
from ouroboros.outcomes import extract_final_answer, latest_agent_defined_verification, latest_unreconciled_failed_verification, latest_unreconciled_masked_verification, should_nudge_verification, turn_has_reviewable_effects
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

    return plan_gate_decision(
        ctx, _llm_trace, hard_rail=hard_rail,
        enforcement=_loop().get_review_enforcement(),
    )


def _plan_gate_reminder(decision: Dict[str, Any]) -> str:
    from ouroboros.owner_hurry import plan_review_reminder

    return plan_review_reminder(decision)


def _plan_gate_disclosure(
    ctx: Any,
    llm_trace: Dict[str, Any],
    *,
    forced_reason: str = "",
) -> str:
    # Normal finalization reuses the reducer projection that decided this
    # exact candidate. The trace copy is presentation-only, granting no
    # permission; forced rails recompute with their explicit rail input.
    from ouroboros.owner_hurry import plan_review_disclosure

    projected = llm_trace.get("plan_gate", llm_trace.get("force_plan_decision"))
    decision = (
        projected
        if not forced_reason and isinstance(projected, dict)
        else _loop()._plan_gate_decision(ctx, llm_trace, hard_rail=forced_reason)
    )
    return plan_review_disclosure(decision, forced_reason)


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


def _nanny_finalization_message(
    tools: ToolRegistry, drive_root: pathlib.Path, task_id: str,
    trace_attempted: bool = False,
) -> str:
    """The honest nanny reminder for a harness-dispatched child at finalization —
    or '' when no reminder is deserved.

    A nanny with no durable delegated start gets NANNY_DID_NOT_DELEGATE; an unsettled
    start gets NANNY_DELEGATED_RUN_PENDING (the custody fact a duplicate start would
    contradict); runs that all ended without success get NANNY_DELEGATED_RUN_FAILED.
    Each is a one-shot fact with the available verbs, not a gate. A succeeded run gets
    nothing here: the nanny's own burn rides the facts line every round.

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
        from ouroboros.tools import registry_guards

        if "delegate_start" not in set(tools.available_tools()) or "delegate_start" in registry_guards._disabled_tools(
                getattr(tools, "_ctx", None)):
            return ""  # the verb is invisible or withheld here; "you chose not to" would be false
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
        # The poltergeist children ran one $0 run, then co-built for tens of opus
        # rounds: that burn is now on the facts line of every round
        # (``nanny_pacing.nanny_burn_fact``), stated, not judged.
        return ""
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
            f"⚠️ NANNY_DELEGATED_RUN_PENDING: {pending} delegated run(s) of this task have "
            "started but not settled — they may still be executing. A final answer now "
            "leaves their results unread (orphaned), and a new start would duplicate "
            "them. Available: delegate_wait to check or wait, delegate_cancel to stop one."
            + failed_note
        )
    if started:
        states = ", ".join(failure_states) or "settled without a recorded terminal state"
        return (
            f"⚠️ NANNY_DELEGATED_RUN_FAILED: this task started {started} delegated run(s) "
            f"and none succeeded; they ended: {states}. Available: delegate_start for "
            "another run."
        )
    from ouroboros.subagent_bootstrap import configured_actor_finalization_message

    _actor = configured_actor_finalization_message(
        tools._ctx, task_id=str(task_id or ""), fallback_root=drive_root)
    if _actor is not None:
        return _actor
    return (
        "⚠️ NANNY_DID_NOT_DELEGATE: this task was dispatched onto the delegated "
        "substrate (executor=harness) and made zero delegate_start calls; its work "
        "ran on this task's own model calls. If it completes this way, its "
        "result discloses delegated_substrate_unused to the parent, together with the "
        "fact that it finalized after this notice. Available: delegate_start, "
        "delegate_wait."
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

    def _inject(reminder: str, note: str, cause: str) -> bool:
        # The one nudge protocol every advisory injection below follows: land
        # the model's pending text, append the reminder, record the note once
        # (live progress AND the durable trail), name the hold's typed cause
        # for the delivery owner's observation, re-loop.
        if content and content.strip():
            messages.append({"role": "assistant", "content": content})
        _loop()._append_or_merge_user_message(messages, f"[SYSTEM REMINDER]\n{reminder}")
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
            tools._ctx._completion_hold_cause = _code  # the producer's own typed code, as its rows carry it
            return True
    from ouroboros.tool_access import canonical_data_root

    finalization_msg = _loop()._skill_finalization_message(canonical_data_root(tools._ctx), llm_trace)
    if finalization_msg and not getattr(tools._ctx, "_skill_finalization_injected", False):
        tools._ctx._skill_finalization_injected = True
        return _inject(finalization_msg, finalization_msg, "skill_not_finalized")
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
                ", and no later receipt reconciles it. Available: re-check it, fix and re-run "
                "verification, or state why this check is not the task's acceptance contract. "
                "This notice is advisory, not a gate.",
                "Red-verification nudge injected before final response.", "verification_red",
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
                "Available: re-run it without the masking pipe or with the runner's own pass marker, "
                "or state why the result is reliable. This notice is advisory, not a gate.",
                "Masked-verification nudge injected before final response.", "verification_masked",
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
                "Your latest passing verification" + _aon + " uses a success criterion you defined "
                "(criterion_source agent_defined, no criterion_basis), not one the task states; the "
                "receipt carries that provenance to acceptance. Available: verify_and_record with a "
                "criterion_basis, or a check of the task's own stated criterion. This notice is advisory, not a gate.",
                "Criterion-provenance nudge injected before final response.", "criterion_agent_defined",
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
            "This turn has reviewable effects and no host-attested verification receipt; a clean "
            "finish records the receipt_absent warning (the result stays solved). Available "
            "verify_and_record kinds: explicit_command / explicit_metric / visible_verifier (run a "
            "check), artifact_observation (confirm an artifact exists), no_visible_machine_contract "
            "(declare that none exists).",
            "Verify-before-done nudge injected before final response.", "verification_absent",
        )
    # No "expected output but nothing attempted" re-loop (A3, removed for #1748): plain
    # prose can itself be the requested deliverable, and the transcript already shows
    # the model its own calls; the outcome keeps its expected_output_ungrounded signal.
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
