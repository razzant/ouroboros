"""The host-forced acceptance review run: the checklist, the panel execution,
the dialogue quorum, applying the panel result, infra-failure records and the
one-shot review entrypoint. Extracted from loop.py (v7 L-B split); loop.py
re-exports every name."""

from __future__ import annotations

import json
import pathlib
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional, Tuple
import logging

from ouroboros import task_pacing
from ouroboros.config import adaptive_quorum, resolve_effort
from ouroboros.review_cycles import REASON_REVIEW_CYCLES_EXHAUSTED
from ouroboros.outcomes import ACCEPTANCE_ACCEPTED, ACCEPTANCE_FINALIZED_UNACCEPTED, ACCEPTANCE_REVISION_REQUESTED
from ouroboros.tools.registry import ToolRegistry
from ouroboros.utils import truncate_review_artifact


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
    evidence: Dict[str, Any] = field(default_factory=dict)
    review_binding: Dict[str, Any] = field(default_factory=dict)
    # One pre-rendered rails line (money/time/rounds/passes headroom) assembled
    # in loop.py from each real source and fed into the improvement capsule
    # (v6.74.0 A1, owner Q6); the capsule builder never gains a ctx parameter.
    rails_line: str = ""


def _acceptance_dialogue_quorum(result: Any) -> int:
    """The quorum the panel itself used (policy min_successful_slots), with the
    adaptive_quorum fallback for records that lost the policy dict."""
    request = getattr(result, "request", None)
    policy = request.get("policy") if isinstance(request, dict) else {}
    try:
        quorum = int((policy or {}).get("min_successful_slots") or 0)
    except (TypeError, ValueError):
        quorum = 0
    if quorum <= 0:
        quorum = adaptive_quorum(len(getattr(result, "actors", None) or []) or 1)
    return max(1, quorum)


def _attach_dialogue_to_host_run(llm_trace: Dict[str, Any], dialogue: Dict[str, Any]) -> None:
    """Persist the dialogue-status vote distribution on the authoritative host
    run record so the review projection carries it for audit (A5)."""
    for run in reversed(llm_trace.get("review_runs") or []):
        if (
            isinstance(run, dict)
            and run.get("authority") == "host_root"
            and not run.get("superseded_by_revision")
        ):
            run["dialogue"] = dict(dialogue)
            return


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


def _latest_agent_acceptance_evidence(llm_trace: Dict[str, Any]) -> Dict[str, Any]:
    """Return the latest validated root self-call packet for host review.

    ``process_tool_results`` records only typed, non-authoritative root
    deferrals here.  The payload is already bounded and redacted by the shared
    evidence builder; the host builder will redact it again while assigning the
    explicit ``agent_supplied`` provenance.
    """
    for call in reversed(llm_trace.get("acceptance_evidence_calls") or []):
        if not isinstance(call, dict):
            continue
        if (
            str(call.get("status") or "") != "deferred_to_host_acceptance"
            or call.get("authoritative") is not False
        ):
            continue
        evidence = call.get("agent_supplied")
        if isinstance(evidence, dict):
            return dict(evidence)
    return {}


def _build_host_acceptance_evidence(ctx: _TaskAcceptanceContext) -> Dict[str, Any]:
    """Build the one bounded host packet shared by binding and reviewer input."""
    from ouroboros.review_evidence import build_task_acceptance_evidence

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
        agent_evidence=_latest_agent_acceptance_evidence(ctx.llm_trace),
        include_recent_commit=committed_this_turn,
        canonical_subject=str(ctx.content or ""),
        subtree_statuses=ctx.subtree_statuses,
    )
    # Owner Q2A: the forced children_unabsorbed rail stashes the process debt
    # (undispositioned children) so the panel sees it; part of the binding hash.
    undecided = getattr(ctx.tools._ctx, "_forced_undispositioned_children", None)
    if isinstance(undecided, list) and undecided:
        evidence["undispositioned_children"] = undecided
    return evidence


def _execute_task_acceptance_panel(ctx: _TaskAcceptanceContext) -> Any:
    """Perform the one substantive host panel over the pre-bound evidence."""
    from ouroboros.review_substrate import (
        HARDNESS_ADVISORY_VISIBLE,
        ReviewRequest,
        ReviewRunResult,
        reviewer_slots,
        run_review_request,
    )

    evidence = ctx.evidence or _build_host_acceptance_evidence(ctx)
    slots = reviewer_slots(effort=resolve_effort("review"), role_hint="task acceptance")
    request = ReviewRequest(
        surface="task_acceptance",
        goal=(
            _loop()._extract_plain_text_from_content(ctx.messages[1].get("content"))
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
            "max_physical_attempts_per_actor": 2,
        },
        task_id=ctx.task_id,
    )
    # Budget admission for the whole acceptance wave (v6.69.0): a wave that
    # cannot fit the remaining root budget is declined up front as a terminal
    # DEGRADED (no-quorum semantics) instead of dying mid-wave. The estimate
    # renders the REAL per-slot message pair; the rare second physical attempt
    # is deliberately not multiplied in — a fail-open coarse filter, not a
    # hard reservation.
    from ouroboros.tools.review_helpers import review_wave_budget_gate

    try:
        from ouroboros.review_substrate import _messages_char_count, _request_messages

        _prompt_chars = _messages_char_count(_request_messages(request, slots[0])) if slots else 0
    except Exception:
        _prompt_chars = len(json.dumps(evidence, ensure_ascii=False, default=str))
    _admission = review_wave_budget_gate(
        ctx.tools._ctx,
        surface="task_acceptance",
        models=[getattr(slot, "model", "") for slot in slots],
        prompt_chars=_prompt_chars,
    )
    if _admission is not None:
        return ReviewRunResult(
            request={"surface": "task_acceptance", "task_id": str(ctx.task_id)},
            actors=[],
            parsed_findings=[],
            aggregate_signal="DEGRADED",
            degraded=True,
            degraded_reasons=[
                "review_wave_budget_insufficient: estimated "
                f"~${_admission.get('estimated_wave_usd')} > remaining "
                f"${_admission.get('remaining_usd')} (no reviewer was called)"
            ],
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
    run_record.update(ctx.review_binding or {})
    aggregate = str(run_record.get("aggregate_signal") or "DEGRADED").upper()
    run_record["enforcement_impact"] = (
        "allows_completion"
        if aggregate == "PASS"
        else "degrades_completion"
    )
    ctx.llm_trace.setdefault("review_runs", []).append(run_record)
    seen = getattr(ctx.tools._ctx, "_task_acceptance_seen_bindings", None)
    binding_hash = str(run_record.get("binding_hash") or "")
    if isinstance(seen, dict) and binding_hash:
        seen[binding_hash] = run_record
    return run_record


def _set_applied_host_acceptance_impact(
    run_record: Any,
    result: Any,
    *,
    requires_revision: bool,
) -> None:
    """Record what the host actually did with a panel result."""
    if not isinstance(run_record, dict):
        return
    if requires_revision:
        run_record["enforcement_impact"] = "requires_revision"
        return
    from ouroboros.review_substrate import task_acceptance_is_clean

    run_record["enforcement_impact"] = (
        "allows_completion" if task_acceptance_is_clean(result) else "degrades_completion"
    )


def _apply_task_acceptance_result(
    ctx: _TaskAcceptanceContext,
    result: Any,
    *,
    record_run: bool = True,
    reused: bool = False,
) -> bool:
    """Apply one panel result; return whether the agent must take another round."""
    from ouroboros.review_substrate import (
        DIALOGUE_CONTINUE,
        aggregate_dialogue_status,
        build_improvement_capsule,
        dissent_findings,
        task_acceptance_is_clean,
    )

    if record_run:
        _record_host_acceptance_run(ctx, result)
    dissent = dissent_findings(result)
    blocking_lane = ctx.mode == "required" and _loop().get_review_enforcement() == "blocking"
    # A REUSED panel (unchanged binding) is the SAME reviewer act applied
    # again: re-collecting would mutate reviewer-authored state with no new
    # reviewer input, and the shifted evidence revision would buy a fresh paid
    # panel for a byte-identical resubmit (fable review r2 #1). The rows were
    # already collected when this exact panel first applied.
    if blocking_lane and not reused:
        _loop()._collect_acceptance_obligations(ctx.llm_trace, result)
    open_obligations = _loop()._open_acceptance_obligations(ctx.llm_trace) if blocking_lane else []
    # v6.74.0 (A1): the capsule leads with the verdict, the concrete open
    # obligation ids, and the pre-rendered rails line (money/time/rounds/passes).
    capsule = build_improvement_capsule(
        result,
        rails_line=ctx.rails_line,
        open_obligations=open_obligations,
    )
    # v6.74.0 (A5): the reviewers' typed dialogue judgement, reduced over ALL
    # contract-valid actors with the panel's own quorum; persisted for audit on
    # the authoritative run record regardless of which branch applies below.
    dialogue = aggregate_dialogue_status(
        result, quorum=_acceptance_dialogue_quorum(result),
    )
    _attach_dialogue_to_host_run(ctx.llm_trace, dialogue)
    dialogue_terminal = dialogue["status"] != DIALOGUE_CONTINUE
    if task_acceptance_is_clean(result):
        ctx.tools._ctx._task_acceptance_reviewed = True
        _loop()._end_task_acceptance_fence(ctx.tools._ctx, outcome="terminal")
        _loop()._mark_root_acceptance_checkpoint(
            ctx.tools._ctx, ctx.llm_trace, status="pass", pass_index=ctx.passes_done,
        )
        if not _loop()._dispose_obligations_on_clean_pass(
            ctx.llm_trace, result, open_obligations, bool(dissent),
        ):
            _loop()._set_acceptance_decision(ctx.llm_trace, {
                "status": ACCEPTANCE_ACCEPTED,
                "reason": "clean_pass",
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
        ctx=ctx.tools._ctx,
    )
    if dialogue_terminal:
        # v6.74.0 (A5): a reviewer quorum judged the dialogue no longer
        # actionable (unreachable_here / stable_disagreement). Finalize through
        # the EXISTING honest path, recording BOTH positions (findings in the
        # run record, dispositions on the obligation rows) with one owner-
        # visible line. Reviewer authorship — not a host timer or a
        # unilateral agent give-up.
        ctx.tools._ctx._task_acceptance_reviewed = True
        _loop()._end_task_acceptance_fence(ctx.tools._ctx, outcome="terminal")
        _loop()._mark_root_acceptance_checkpoint(
            ctx.tools._ctx,
            ctx.llm_trace,
            status=str(result.aggregate_signal or "DEGRADED").lower(),
            pass_index=ctx.passes_done,
        )
        _loop()._set_acceptance_decision(ctx.llm_trace, {
            # The with/without-obligations distinction moves from the status token to
            # the `open_obligations` id list this branch already records.
            "status": ACCEPTANCE_FINALIZED_UNACCEPTED,
            "reason": "dialogue_terminal",
            "source": "task_acceptance_review",
            "rationale": (
                f"Reviewer quorum judged the dialogue {dialogue['status']}; "
                "finalizing honestly with both positions recorded "
                f"({len(open_obligations)} open obligation(s))."
            ),
            "dialogue_status": dialogue["status"],
            "dialogue_votes": dialogue["votes"],
            "dissent_noted": bool(dissent),
            "open_obligations": [str(item.get("id")) for item in open_obligations],
        })
        ctx.emit_progress(
            f"Task acceptance review: {result.aggregate_signal} — reviewer quorum judged "
            f"the dialogue {dialogue['status']}; finalizing with "
            f"{len(open_obligations)} open obligation(s)."
        )
        return False
    if capsule and pass_ok:
        _loop()._set_acceptance_decision(ctx.llm_trace, {
            "status": ACCEPTANCE_REVISION_REQUESTED,
            "reason": "improvement_capsule",
            "source": "task_acceptance_review",
            "rationale": "A compact advisory improvement capsule was fed back for one bounded revision pass.",
            "dissent_noted": bool(dissent),
        })
        ctx.tools._ctx._task_acceptance_improvement_passes = ctx.passes_done + 1
        if not _loop()._end_task_acceptance_fence(ctx.tools._ctx, outcome="revision"):
            ctx.tools._ctx._task_acceptance_reviewed = True
            _loop()._set_acceptance_decision(ctx.llm_trace, {
                "status": ACCEPTANCE_FINALIZED_UNACCEPTED,
                "reason": "fence_reopen_failed",
                "source": "task_acceptance_fence",
                "rationale": "The revision could not safely reopen queue admission at the dispatch boundary.",
            })
            return False
        if open_obligations:
            capsule += _loop()._format_obligations_clause(open_obligations)
        if ctx.content and ctx.content.strip():
            ctx.messages.append({"role": "assistant", "content": ctx.content})
        _loop()._append_or_merge_user_message(ctx.messages, capsule)
        ctx.emit_progress(
            f"Task acceptance review: {result.aggregate_signal} — improvement note fed back."
        )
        return True

    ctx.tools._ctx._task_acceptance_reviewed = True
    _loop()._end_task_acceptance_fence(ctx.tools._ctx, outcome="terminal")
    _loop()._mark_root_acceptance_checkpoint(
        ctx.tools._ctx,
        ctx.llm_trace,
        status=str(result.aggregate_signal or "DEGRADED").lower(),
        pass_index=ctx.passes_done,
    )
    if _loop()._dispose_obligations_on_clean_pass(
        ctx.llm_trace, result, open_obligations, bool(dissent),
    ):
        ctx.emit_progress(
            f"Task acceptance review: {result.aggregate_signal} (clean pass; obligations closed)."
        )
        return False
    aggregate_signal = str(result.aggregate_signal or "DEGRADED").upper()
    if aggregate_signal == "DEGRADED":
        _loop()._set_acceptance_decision(ctx.llm_trace, {
            "status": ACCEPTANCE_FINALIZED_UNACCEPTED,
            "reason": "review_degraded",
            "source": "task_acceptance_review",
            "rationale": "Acceptance reviewers did not reach a valid quorum.",
            "degraded_reasons": list(getattr(result, "degraded_reasons", []) or []),
            "open_obligations": [str(item.get("id")) for item in open_obligations],
        })
        # Per-slot causes were always recorded in the structured decision; the
        # owner-visible line used to say only "no valid quorum", forcing a dig
        # through task_results to learn WHICH slot failed and why (v6.70.0).
        _degraded_reasons = list(getattr(result, "degraded_reasons", []) or [])
        # Bounded PREVIEW for the chat line only — the complete causes live in
        # the structured decision record (owner-facing full copy, per the
        # v6.70.0 honesty invariant).
        _reason_note = "; ".join(
            truncate_review_artifact(str(r), limit=300).replace("\n", " ")
            for r in _degraded_reasons[:4]
        )
        if len(_degraded_reasons) > 4:
            _reason_note += f" (+{len(_degraded_reasons) - 4} more in the task result)"
        ctx.emit_progress(
            "Task acceptance review: DEGRADED (no valid quorum; not recorded as PASS)."
            + (f" Causes: {_reason_note}" if _reason_note else "")
        )
        return False
    if capsule and open_obligations:
        _loop()._set_acceptance_decision(ctx.llm_trace, {
            "status": ACCEPTANCE_FINALIZED_UNACCEPTED,
            "reason": pass_reason if pass_reason == REASON_REVIEW_CYCLES_EXHAUSTED else "open_obligations",
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
        _loop()._set_acceptance_decision(ctx.llm_trace, {
            "status": ACCEPTANCE_FINALIZED_UNACCEPTED,
            "reason": (
                pass_reason if pass_reason == REASON_REVIEW_CYCLES_EXHAUSTED else
                "improvement_window_closed"
                if (not ctx.passes_done and pass_reason)
                else "capsule_spent"
            ),
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
        _loop()._set_acceptance_decision(ctx.llm_trace, {
            "status": ACCEPTANCE_FINALIZED_UNACCEPTED,
            "reason": "reviewer_fail_no_capsule",
            "source": "task_acceptance_review",
            "rationale": "A valid acceptance reviewer FAIL had no additional capsule text.",
            "dissent_noted": bool(dissent),
        })
        ctx.emit_progress("Task acceptance review: FAIL (finalizing with a failed review verdict).")
    elif open_obligations:
        _loop()._set_acceptance_decision(ctx.llm_trace, {
            "status": ACCEPTANCE_FINALIZED_UNACCEPTED,
            "reason": "open_obligations",
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
        _loop()._set_acceptance_decision(ctx.llm_trace, {
            # Round-9 CRITICAL 1: fall-through AFTER `task_acceptance_is_clean`
            # refused the panel, so it cannot mint `accepted` (ARCH reserves
            # that for clean acceptance). Reachable: a reviewer claims `solved`
            # with a MISSING criterion and the improvement-pass cap spent —
            # nothing actionable, but not "accepted". The typed reason names
            # WHY the loop stops; tier honesty keeps riding `outcome_tier`.
            "status": ACCEPTANCE_FINALIZED_UNACCEPTED,
            "reason": "no_actionable_changes",
            "source": "task_acceptance_review",
            "rationale": (
                f"Re-review was not a clean acceptance ({result.aggregate_signal}) and "
                "suggested no actionable changes; finalizing honestly without acceptance."
            ),
            "dissent_noted": bool(dissent),
        })
        ctx.emit_progress(
            f"Task acceptance review: {result.aggregate_signal} — not a clean acceptance "
            "and no actionable changes were suggested; finalizing without acceptance."
        )
    return False


def _record_acceptance_infra_failure(ctx: _TaskAcceptanceContext, exc: Exception) -> bool:
    """Finish an eligible mandatory panel as DEGRADED, never as a silent skip."""
    ctx.tools._ctx._task_acceptance_reviewed = True
    _loop()._end_task_acceptance_fence(ctx.tools._ctx, outcome="degraded")
    _loop()._mark_root_acceptance_checkpoint(
        ctx.tools._ctx,
        ctx.llm_trace,
        status="review_degraded",
        pass_index=ctx.passes_done,
    )
    safe_error = _loop()._extract_plain_text_from_content(str(exc))[:2000]
    _mark_agent_acceptance_runs_advisory(ctx.llm_trace)
    run_record = {
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
        **(ctx.review_binding or {}),
        "enforcement_impact": "degrades_completion",
    }
    ctx.llm_trace.setdefault("review_runs", []).append(run_record)
    seen = getattr(ctx.tools._ctx, "_task_acceptance_seen_bindings", None)
    binding_hash = str(run_record.get("binding_hash") or "")
    if isinstance(seen, dict) and binding_hash:
        seen[binding_hash] = run_record
    _loop()._set_acceptance_decision(ctx.llm_trace, {
        "status": ACCEPTANCE_FINALIZED_UNACCEPTED,
        "reason": "infra_failure",
        "source": "task_acceptance_review",
        "rationale": "The mandatory host acceptance panel failed before a valid quorum.",
        "degraded_reasons": [f"{type(exc).__name__}: {safe_error}"],
    })
    ctx.emit_progress("Task acceptance review: DEGRADED after host review infrastructure failure.")
    return False


def _prior_acceptance_run(
    tools_ctx: Any, llm_trace: Dict[str, Any], binding_hash: str,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """Locate the authoritative host run already recorded for this binding:
    first the trace (survives requeue replay), then the process-local
    ``_task_acceptance_seen_bindings`` cache. Returns (cache, prior_run)."""
    seen_bindings = getattr(tools_ctx, "_task_acceptance_seen_bindings", None)
    if not isinstance(seen_bindings, dict):
        seen_bindings = {}
        tools_ctx._task_acceptance_seen_bindings = seen_bindings
    prior_run = next(
        (
            run for run in reversed(llm_trace.get("review_runs") or [])
            if isinstance(run, dict)
            and run.get("authority") == "host_root"
            and not run.get("superseded_by_revision")
            and str(run.get("binding_hash") or "") == binding_hash
        ),
        None,
    )
    cached_run = seen_bindings.get(binding_hash)
    if (
        prior_run is None
        and isinstance(cached_run, dict)
        and not cached_run.get("superseded_by_revision")
    ):
        prior_run = cached_run
    return seen_bindings, prior_run


def _direct_context_fence_state(tools_ctx: Any, fence_token: Any) -> Any:
    """Review-binding fence state: the queue-owned token when present, else the
    direct-chat generations (no queue fence exists for a direct context)."""
    if fence_token is not None:
        return fence_token
    return {
        "state": "direct_context",
        "owner_generation": getattr(tools_ctx, "_task_acceptance_owner_generation", None),
        "queue_generation": getattr(tools_ctx, "_task_acceptance_fence_generation", None),
    }


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
    """Run the root-owned acceptance gate once for the current deliverable.
    Loop-side rails facts arrive via the ``_acceptance_loop_rails`` ctx stash
    (set by ``_no_tool_final_answer``; keeps the signature at 8 params)."""
    mode = _loop().get_task_review_mode()
    _loop()._latch_final_answer_marker(llm_trace, content)
    if getattr(tools._ctx, "_task_acceptance_reviewed", False):
        return False
    from ouroboros.task_results import resolve_task_lineage

    meta = getattr(tools._ctx, "task_metadata", {})
    meta = meta if isinstance(meta, dict) else {}
    lineage = resolve_task_lineage(
        task_id or getattr(tools._ctx, "task_id", ""),
        metadata=meta,
        root_task_id=getattr(tools._ctx, "root_task_id", None),
        parent_task_id=getattr(tools._ctx, "parent_task_id", None),
        delegation_role=getattr(tools._ctx, "delegation_role", None),
        original_task_id=getattr(tools._ctx, "original_task_id", None),
        timeout_retry_from=getattr(tools._ctx, "timeout_retry_from", None),
    )
    eligible, trigger = _loop()._task_acceptance_eligible(
        mode,
        llm_trace,
        bool(getattr(tools._ctx, "is_direct_chat", False)),
        is_root_task=bool(lineage["is_root_task"]),
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
        "eligibility": "eligible" if eligible else "not_eligible", "trigger": trigger,
    }
    if not eligible:
        return False
    # Owner hurry (§19.7.2 item 8): AFTER structural eligibility is known and
    # BEFORE acceptance-fence/quiescence/reviewer admission, an armed latch
    # skips the next otherwise-eligible panel with the typed reason — zero
    # reviewer calls (an already in-flight panel is never cancelled/relabeled).
    from ouroboros.owner_hurry import acceptance_skip_applied, effective_budget_profile

    if acceptance_skip_applied(
        tools._ctx, llm_trace, task_id=task_id, drive_root=drive_root,
        set_decision=_loop()._set_acceptance_decision, emit_progress=emit_progress,
    ):
        return False
    fence_ok, _fence_token = _loop()._begin_task_acceptance_fence(tools._ctx, task_id)
    if not fence_ok:
        llm_trace["review_decision"] = {
            "eligibility": "acceptance_fence_failed", "trigger": trigger,
        }
        _loop()._append_or_merge_user_message(
            messages,
            "[TASK ACCEPTANCE WAIT] The supervisor could not atomically close "
            "subtask admission. Do not finalize or spawn more work; retry after the "
            "queue fence is available.",
        )
        emit_progress("Task acceptance review waiting for the queue-owned admission fence.")
        return True
    quiescent, subtree_statuses = _loop()._task_acceptance_subtree_snapshot(
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
        _loop()._append_or_merge_user_message(
            messages,
            "[TASK ACCEPTANCE WAIT] The root acceptance review requires the recursive "
            "subtree to be terminal. Absorb or explicitly cancel the remaining child "
            "tasks before finalizing.",
        )
        emit_progress("Task acceptance review waiting for recursive subtree quiescence.")
        return True
    # §19.7.2 item 7: ONE effective profile (remaining improvement passes -> 0
    # under an armed hurry latch) feeds EVERY acceptance-pacing read below —
    # the real improvement_pass_allowed call and the rails display alike.
    budget_profile = effective_budget_profile(
        tools._ctx, task_pacing.resolve_budget_profile(tools._ctx),
    )
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
        _loop()._end_task_acceptance_fence(tools._ctx, outcome="terminal")
        _loop()._mark_root_acceptance_checkpoint(
            tools._ctx, llm_trace, status=launch_reason, pass_index=passes_done,
        )
        llm_trace["review_decision"].update({"skipped": launch_reason})
        # The pacing launch reason is now the typed REASON, not the status;
        # `outcomes.derive_loop_outcome` keys on that PAIR (see its comment).
        _loop()._set_acceptance_decision(llm_trace, {
            "status": ACCEPTANCE_FINALIZED_UNACCEPTED, "reason": launch_reason,
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
        evidence={},
        review_binding={},
        rails_line=task_pacing.acceptance_rails_line(
            budget_snapshot,
            budget_profile,
            passes_done,
            getattr(tools._ctx, "_acceptance_loop_rails", None),
            required_blocking=(
                mode == "required" and _loop().get_review_enforcement() == "blocking"
            ), workspace=task_pacing._workspace_delivery(tools._ctx),
        ),
    )
    try:
        from types import SimpleNamespace

        from ouroboros.review_evidence import task_acceptance_evidence_revision
        from ouroboros.review_substrate import build_review_binding

        review_ctx.evidence = _build_host_acceptance_evidence(review_ctx)
        review_ctx.review_binding = build_review_binding(
            candidate=content,
            evidence=review_ctx.evidence,
            fence_token_or_state=_direct_context_fence_state(tools._ctx, _fence_token),
        )
        binding_hash = str(review_ctx.review_binding.get("binding_hash") or "")
        seen_bindings, prior_run = _prior_acceptance_run(
            tools._ctx, llm_trace, binding_hash,
        )
        reused_result = None
        if prior_run is not None:
            seen_bindings[binding_hash] = prior_run
            if prior_run not in (llm_trace.get("review_runs") or []):
                llm_trace.setdefault("review_runs", []).append(dict(prior_run))
            llm_trace["review_decision"].update({
                "panel_reused": True,
                "panel_id": str(prior_run.get("panel_id") or ""),
                "binding_hash": binding_hash,
            })
            emit_progress(
                "Task acceptance review: reusing the authoritative result for the unchanged binding."
            )
            # Re-run the normal semantic application (gates, outcome axis,
            # obligations, fence) without appending or paying for another panel.
            reused_result = SimpleNamespace(**prior_run)
        elif binding_hash in seen_bindings:
            # A process-local attempt without its authoritative trace is not safe
            # to repeat or silently accept.  The ordinary infra-degraded path below
            # records the missing authority and closes finalization honestly.
            raise RuntimeError("acceptance binding was attempted but its host run is unavailable")
        else:
            seen_bindings[binding_hash] = None
        llm_trace["review_decision"].update({
            "panel_id": str(review_ctx.review_binding.get("panel_id") or ""),
            "binding_hash": binding_hash,
        })
        messages_before_apply = list(messages)
        obligations_were_present = "acceptance_obligations" in llm_trace
        obligations_before_apply = [
            dict(row) if isinstance(row, dict) else row
            for row in (llm_trace.get("acceptance_obligations") or [])
        ]
        passes_before_apply = int(
            getattr(tools._ctx, "_task_acceptance_improvement_passes", 0) or 0
        )
        panel_result = reused_result or _execute_task_acceptance_panel(review_ctx)
        run_record = (
            prior_run
            if reused_result is not None
            else _record_host_acceptance_run(review_ctx, panel_result)
        )
        if _loop()._task_acceptance_owner_generation_changed(tools._ctx):
            _loop()._supersede_task_acceptance_for_owner_followup(tools._ctx, llm_trace)
            emit_progress(
                "Task acceptance review superseded: an owner follow-up arrived during the panel."
            )
            return True
        fresh_quiescent, fresh_subtree_statuses = _loop()._task_acceptance_subtree_snapshot(
            tools._ctx, drive_root, task_id,
        )
        fresh_review_ctx = replace(
            review_ctx,
            subtree_statuses=fresh_subtree_statuses,
            evidence={},
        )
        fresh_evidence_revision = task_acceptance_evidence_revision(
            _build_host_acceptance_evidence(fresh_review_ctx)
        )
        frozen_evidence_revision = str(
            review_ctx.review_binding.get("evidence_revision") or ""
        )
        stale_reason = ""
        if not fresh_quiescent:
            stale_reason = "host_acceptance_subtree_became_non_quiescent"
        elif fresh_evidence_revision != frozen_evidence_revision:
            stale_reason = "host_acceptance_evidence_revision_changed"
        if stale_reason:
            _loop()._supersede_task_acceptance_for_evidence_change(
                tools._ctx,
                llm_trace,
                run_record,
                stale_reason,
                messages,
                emit_progress,
            )
            return True
        another_round = _apply_task_acceptance_result(
            review_ctx,
            panel_result,
            record_run=False,
            reused=reused_result is not None,
        )
        if getattr(tools._ctx, "_task_acceptance_fence_generation_mismatch", False):
            messages[:] = messages_before_apply
            if obligations_were_present:
                llm_trace["acceptance_obligations"] = obligations_before_apply
            else:
                llm_trace.pop("acceptance_obligations", None)
            tools._ctx._task_acceptance_improvement_passes = passes_before_apply
            _loop()._supersede_task_acceptance_for_owner_followup(tools._ctx, llm_trace)
            emit_progress(
                "Task acceptance review superseded: an owner follow-up arrived during the panel."
            )
            return True
        _set_applied_host_acceptance_impact(
            run_record,
            panel_result,
            requires_revision=another_round,
        )
        return another_round
    except Exception as exc:
        log.debug("Mandatory task acceptance review failed", exc_info=True)
        return _record_acceptance_infra_failure(review_ctx, exc)
