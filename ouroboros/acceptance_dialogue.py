"""Host acceptance dialogue: obligations, the reviewer-authored dialogue's
consumer, and the A-material admission that makes the loop converge.

Split out of ``loop.py`` (which stayed the owner of the fence, checkpoint,
panel-execution and message rails this module drives) so the acceptance
contract lives in one readable place. The reducer over the reviewers' typed
``dialogue_status`` votes stays in ``review_substrate.aggregate_dialogue_status``
— that is the vote SSOT; this module is its CONSUMER.

A-MATERIAL (owner ratification 2026-08-30). A paid acceptance panel is the
scarce resource, so its identity is what the agent actually CHANGED, not what
merely moved: ``paid_identity = sha256(candidate_hash + the sorted set of
nonempty (obligation_id, disposition, sha256(reason)) tuples)``. The evidence
revision — which every cosmetic tool call shifts — deliberately does NOT enter
it; it remains stale-packet detection for the supersede paths. A resubmit whose
paid identity is unchanged replays the recorded verdict for FREE and terminates
with the typed ``identical_acceptance_refused`` reason instead of buying
another panel and re-entering the improvement capsule. Majority voting was
REJECTED: one strong reviewer may still hold the loop open — but only with
material.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

from ouroboros import task_pacing
from ouroboros.config import adaptive_quorum, get_acceptance_fence_wait_max_rounds
from ouroboros.outcomes import (
    ACCEPTANCE_ACCEPTED,
    ACCEPTANCE_BYPASS_REASONS,
    ACCEPTANCE_DECISION_STATUSES,
    ACCEPTANCE_FINALIZED_UNACCEPTED,
    ACCEPTANCE_REVISION_REQUESTED,
    REASON_ACCEPTANCE_REVIEW_SKIPPED_DEADLINE_RESERVE,
    REASON_IDENTICAL_ACCEPTANCE_REFUSED,
)
from ouroboros.review_cycles import REASON_REVIEW_CYCLES_EXHAUSTED
from ouroboros.utils import truncate_review_artifact

log = logging.getLogger(__name__)


ACCEPTANCE_REASON_UNSPECIFIED = "unspecified"
# Closed set of typed acceptance reasons (v6.78.0): every value is a fact
# the host already computed or the exit branch's name; none derives from model
# prose. `unspecified` is only the fail-closed fallback for a forgotten reason.
ACCEPTANCE_DECISION_REASONS = (
    "clean_pass",
    "clean_pass_obligations_closed",
    "no_actionable_changes",
    "delivery_binding_superseded",
    "owner_followup",
    "evidence_refresh",
    "improvement_capsule",
    "dialogue_terminal",
    "open_obligations",
    "capsule_spent",
    "improvement_window_closed",
    "reviewer_fail_no_capsule",
    "review_degraded",
    "fence_reopen_failed",
    "infra_failure",
    # The queue-owned fence remained unavailable past its configured round
    # bound, so the task terminalized as infrastructure failure.
    "acceptance_fence_unavailable",
    # The pacing/wallet reason two branches below already STAMP (`pass_reason ==
    # REASON_REVIEW_CYCLES_EXHAUSTED`); it was missing from the closed set, so a
    # spent shared cap shipped a reason no reader could validate.
    REASON_REVIEW_CYCLES_EXHAUSTED,
    # A-material (2026-08-30): the resubmit carried no changed candidate and no new
    # obligation disposition, so the recorded verdict was replayed for free.
    REASON_IDENTICAL_ACCEPTANCE_REFUSED,
    # Owner Q2A: the forced children_unabsorbed rail runs the panel but cannot
    # grant a requested improvement pass; the dangling revision terminalizes.
    "revision_unavailable_on_forced_rail",
    REASON_ACCEPTANCE_REVIEW_SKIPPED_DEADLINE_RESERVE,
    # Forced-rail acceptance bypass (closed set, outcomes.py SSOT): stamped by
    # `_record_forced_acceptance_bypass` when the panel was owed but a rail fired.
    *sorted(ACCEPTANCE_BYPASS_REASONS),
    ACCEPTANCE_REASON_UNSPECIFIED,
)


def _set_acceptance_decision(llm_trace: Dict[str, Any], decision: Dict[str, Any]) -> None:
    """The ONLY merge point for the host acceptance decision (v6.78.0, owner Q23).

    Every host exit funnels here and leaves in one of the three canonical
    owner-facing states (``ACCEPTANCE_DECISION_STATUSES``) plus a typed
    ``reason`` naming WHICH exit. A status outside the trio fails closed to
    ``finalized_unaccepted`` with its raw token surviving as ``reason`` — no
    fourth state, no lost token. The agent's stance (``agent_disposition``/
    ``agent_rationale``) carries forward, never overwritten (after P4.1 the
    agent writes no status at all)."""
    previous = llm_trace.get("acceptance_decision") if isinstance(llm_trace.get("acceptance_decision"), dict) else {}
    merged = dict(decision)
    status = str(merged.get("status") or "")
    reason = str(merged.get("reason") or "")
    if status not in ACCEPTANCE_DECISION_STATUSES:
        merged["status"] = ACCEPTANCE_FINALIZED_UNACCEPTED
        reason = reason or status or ACCEPTANCE_REASON_UNSPECIFIED
    merged["reason"] = reason
    for key in ("agent_disposition", "agent_rationale"):
        if previous.get(key) and not merged.get(key):
            merged[key] = previous.get(key)
    llm_trace["acceptance_decision"] = merged


def acceptance_fence_failure_exhausted(
    tools_ctx: Any,
    llm_trace: Dict[str, Any],
    emit_progress: Callable[[str], None],
) -> bool:
    """Count a failed fence begin and terminalize after the configured bound."""
    failures = int(
        getattr(tools_ctx, "_task_acceptance_fence_failures", 0) or 0
    ) + 1
    tools_ctx._task_acceptance_fence_failures = failures
    if failures < get_acceptance_fence_wait_max_rounds():
        return False
    tools_ctx._task_acceptance_fence_infra_failed = True
    _set_acceptance_decision(llm_trace, {
        "status": ACCEPTANCE_FINALIZED_UNACCEPTED,
        "reason": "acceptance_fence_unavailable",
        "source": "acceptance_fence",
        "rationale": (
            "The queue-owned admission fence stayed unavailable for "
            f"{failures} consecutive rounds; terminalizing as infrastructure "
            "failure instead of burning paid rounds until the deadline."
        ),
    })
    emit_progress(
        "Task acceptance review could not acquire the queue-owned admission "
        "fence; terminalizing as an infrastructure failure."
    )
    return True


def finalize_acceptance_fence_failure(
    tools_ctx: Any,
    limit_ctx: Any,
    llm_trace: Dict[str, Any],
    forced_fallback: Callable[..., Tuple[str, Dict[str, Any], Dict[str, Any]]],
) -> Optional[Tuple[str, Dict[str, Any], Dict[str, Any]]]:
    """Build the host-salvaged infra result after bounded fence exhaustion."""
    if not bool(getattr(tools_ctx, "_task_acceptance_fence_infra_failed", False)):
        return None
    text, usage, trace = forced_fallback(
        limit_ctx,
        llm_trace,
        "⚠️ The task could not start its acceptance review: the queue-owned "
        "admission fence stayed unavailable. Any files written so far are "
        "preserved in the workspace.",
        "acceptance_fence_unavailable",
        source="acceptance_fence_unavailable",
    )
    usage.update(
        execution_status="infra_failed",
        reason_code="acceptance_fence_unavailable",
    )
    return text, usage, trace


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
        "allows_completion"
        if task_acceptance_is_clean(result)
        else "degrades_completion"
    )


def _direct_context_fence_state(tools_ctx: Any, fence_token: Any) -> Any:
    """Return the queue token or direct-chat generations for review binding."""
    if fence_token is not None:
        return fence_token
    return {
        "state": "direct_context",
        "owner_generation": getattr(
            tools_ctx, "_task_acceptance_owner_generation", None,
        ),
        "queue_generation": getattr(
            tools_ctx, "_task_acceptance_fence_generation", None,
        ),
    }


def _collect_acceptance_obligations(llm_trace: Dict[str, Any], result: Any) -> None:
    """Typed PER-TASK obligations from critical contributing findings (v6.54.4).

    Required+blocking path only. Each critical finding WITH a concrete
    recommendation becomes one open obligation in llm_trace (never the durable
    commit review_state — a separate SSOT). Clean finalization asks for an
    agent disposition per obligation (v6.54.0); time/pass gates and the
    forced-finalization escape hatches bound the loop, so a deadline never
    hangs here. v6.60.0 widening (S1-lite, owner quiz 18b): when the AGGREGATE
    verdict itself is failing — signal FAIL, or worst tier
    blocked_with_evidence — contributing reviewers' HIGH findings with a
    concrete recommendation also become obligations (the PB incident). On a
    PASS (incl. with-dissent) the bar stays critical-only, so the blocking
    lane cannot creep into taxing clean runs with hygiene items."""
    from ouroboros.review_substrate import _contributing_actors, aggregate_outcome_tier

    contributing = {str(a.get("slot_id", "")) for a in _contributing_actors(result)}
    obligations = llm_trace.setdefault("acceptance_obligations", [])
    by_id = {str(o.get("id")): o for o in obligations if isinstance(o, dict)}
    # No contributing actors (all parse-degraded / no quorum) => no
    # authoritative verdict: manufacture NO blocking obligations — else one
    # parse-degraded slot's critical finding would gate finalization, the
    # class the capsule refuses (r1); obligations ride CONTRIBUTING slots.
    if not contributing:
        return
    _agg_failing = (
        str(getattr(result, "aggregate_signal", "") or "").upper() == "FAIL"
        or aggregate_outcome_tier(result) == "blocked_with_evidence"
    )
    _obligation_severities = {"critical", "high"} if _agg_failing else {"critical"}
    # Ids already created or reopened by THIS panel pass: slots of one panel
    # routinely raise the same finding (typed re_raise copies the catalog
    # id); without it the second slot's dupe would falsely bump
    # reopened_count and overwrite reviewer_rebuttal_response (fable r1 #1).
    touched_this_pass: set[str] = set()
    for finding in (getattr(result, "parsed_findings", None) or []):
        if not isinstance(finding, dict):
            continue
        if str(finding.get("severity") or "").strip().lower() not in _obligation_severities:
            continue
        if str(finding.get("slot_id", "")) not in contributing:
            continue
        recommendation = " ".join(str(finding.get("recommendation") or "").split()).strip()
        if not recommendation:
            continue
        item = str(finding.get("item") or "finding").strip()
        # v6.74.0 (A3): obligation identity is reviewer-authored. A re_raise
        # MUST name an existing catalog id (the host validates existence
        # only); a missing/unknown id fails closed to `new` with a disclosed
        # note — a reworded re-raise cannot mint a fresh hash id.
        kind = str(finding.get("disposition_kind") or "").strip().lower()
        claimed_id = str(finding.get("obligation_id") or "").strip()
        unbound_note = ""
        if kind == "re_raise":
            row = by_id.get(claimed_id)
            if row is not None:
                if claimed_id not in touched_this_pass:
                    touched_this_pass.add(claimed_id)
                    _reopen_obligation_row(row, finding)
                continue
            unbound_note = f"re_raise_unbound:{claimed_id or 'missing_id'}"
        oid = "ob-" + hashlib.sha256(
            json.dumps([item, recommendation], ensure_ascii=False).encode("utf-8")
        ).hexdigest()[:12]
        if oid in by_id:
            # Reviewer-authored identity (triad r2, sol): only an UNTYPED
            # legacy finding may reopen via byte-identical text (v6.71.1
            # compat). A typed "new"/unbound "re_raise" matching a settled row
            # must NOT resurrect the settled rebuttal — sloppiness DISCLOSED.
            row = by_id[oid]
            if not kind and oid not in touched_this_pass:
                touched_this_pass.add(oid)
                _reopen_obligation_row(row, finding)
            elif kind:
                notes = row.setdefault("notes", [])
                note = unbound_note or f"typed_new_matched_existing:{oid}"
                if note not in notes:
                    notes.append(note)
            continue
        row = {
            "id": oid,
            "item": item,
            "recommendation": recommendation,
            "status": "open",
            "disposition": "",
            "disposition_reason": "",
        }
        if unbound_note:
            row["notes"] = [unbound_note]
        by_id[oid] = row
        touched_this_pass.add(oid)
        obligations.append(row)


def _reopen_obligation_row(row: Dict[str, Any], finding: Dict[str, Any]) -> None:
    """Reopen a re-raised obligation WITHOUT wiping the agent's argument (A3).

    The prior disposition/reason survive as ``previous_disposition`` /
    ``previous_reason`` and ``reopened_count`` increments, so the agent can see
    its rebuttal was overruled (previously indistinguishable from a fresh
    finding) and the next reviewer receives the prior argument to adjudicate.
    The reviewer's stated reason for maintaining the finding rides along."""
    if str(row.get("disposition") or "").strip() or str(row.get("status") or "") == "agent_disposed":
        row["previous_disposition"] = str(
            row.get("disposition") or row.get("status") or ""
        )
        row["previous_reason"] = str(row.get("disposition_reason") or "")
    row["reopened_count"] = int(row.get("reopened_count") or 0) + 1
    row["disposition"] = ""
    row["disposition_reason"] = ""
    row["status"] = "open"
    reviewer_response = " ".join(str(finding.get("evidence") or "").split()).strip()
    if reviewer_response:
        row["reviewer_rebuttal_response"] = truncate_review_artifact(
            reviewer_response, limit=600,
        )


def _open_acceptance_obligations(llm_trace: Dict[str, Any]) -> List[Dict[str, Any]]:
    # An agent-filed disposition (status="agent_disposed") is a
    # CLAIM/rebuttal, not a settlement: the row is pending until a host panel
    # adjudicates (PASS settles; re-raise reopens). SSOT: review_evidence.
    from ouroboros.review_evidence import obligation_is_pending

    return [
        o for o in (llm_trace.get("acceptance_obligations") or [])
        if obligation_is_pending(o)
    ]


def _dispose_obligations_on_clean_pass(
    llm_trace: Dict[str, Any],
    result: Any,
    open_obligations: List[Dict[str, Any]],
    dissent_noted: bool,
) -> bool:
    """If the re-review is a CLEAN PASS (aggregate PASS and not degraded), close
    the open obligations as disposed_by_re_review and record the accepted verdict;
    return True. A DEGRADED/no-quorum run proves nothing → returns False, leaving
    the honest best-effort labeling to the caller."""
    if not open_obligations:
        return False
    from ouroboros.review_substrate import task_acceptance_is_clean

    if not task_acceptance_is_clean(result):
        return False
    for ob in open_obligations:
        if str(ob.get("status") or "") == "agent_disposed":
            # The clean panel ACCEPTED the agent's filed disposition (a
            # rebuttal it chose not to re-raise): keep that disposition/reason
            # as provenance, record the host settlement distinctly (r6) —
            # never rewrite a rejected rebuttal into "addressed by revision".
            ob["status"] = "disposed_rebuttal_accepted"
            continue
        ob["disposition"] = "addressed"
        ob["disposition_reason"] = "resolved by revision: the clean re-review returned no findings"
        ob["status"] = "disposed_by_re_review"
    _set_acceptance_decision(llm_trace, {
        "status": ACCEPTANCE_ACCEPTED,
        "reason": "clean_pass_obligations_closed",
        "source": "task_acceptance_review",
        "rationale": "Clean PASS re-review; open obligations closed by the revision (dissent, if any, stays advisory).",
        "dissent_noted": dissent_noted,
    })
    return True


def _format_obligations_clause(open_obligations: List[Dict[str, Any]]) -> str:
    # v6.74.0 (A4): disagreement is recorded ONLY via
    # obligation_dispositions — the old "or address them directly" prose read
    # as a third channel; fixing the work just makes the next panel clean.
    if not open_obligations:
        return ""
    lines = [
        "",
        "OPEN OBLIGATIONS (blocking review policy). Either FIX the work so the next review "
        "panel finds it clean, or record your disagreement via the task_acceptance_review "
        "tool's obligation_dispositions (addressed / rejected / deferred + reason) — "
        "dispositions are the ONLY channel the reviewer adjudicates:",
    ]
    for o in open_obligations[:5]:
        line = f"  {o.get('id')}: {o.get('item')} — {o.get('recommendation')}"
        reopened = int(o.get("reopened_count") or 0)
        if reopened > 0:
            line += f" [re-raised ×{reopened}"
            if str(o.get("previous_disposition") or "").strip():
                line += (
                    f"; your '{o.get('previous_disposition')}' rebuttal was overruled"
                )
                response = str(o.get("reviewer_rebuttal_response") or "").strip()
                if response:
                    line += f" — reviewer: {response}"
            line += "]"
        lines.append(line)
    if len(open_obligations) > 5:
        lines.append(f"  (+{len(open_obligations) - 5} more in the task record)")
    return "\n".join(lines)


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


def _refuse_identical_acceptance(
    ctx: Any,
    result: Any,
    *,
    dialogue: Dict[str, Any],
    dissent: bool,
    open_obligations: List[Dict[str, Any]],
) -> bool:
    """Terminate a resubmit whose A-material paid identity was already bought.

    The recorded verdict is replayed for FREE and quoted; the improvement capsule
    is deliberately NOT re-entered. Feeding the note again asks for a round the
    agent has already answered with nothing new, and every such round shifted the
    evidence revision into a fresh paid binding — the 21-panel pump. The dialogue
    record and the decision row still land, so the replay stays auditable."""
    from ouroboros import loop

    ctx.tools._ctx._task_acceptance_reviewed = True
    loop._end_task_acceptance_fence(ctx.tools._ctx, outcome="terminal")
    loop._mark_root_acceptance_checkpoint(
        ctx.tools._ctx,
        ctx.llm_trace,
        status=str(result.aggregate_signal or "DEGRADED").lower(),
        pass_index=ctx.passes_done,
    )
    _set_acceptance_decision(ctx.llm_trace, {
        "status": ACCEPTANCE_FINALIZED_UNACCEPTED,
        "reason": REASON_IDENTICAL_ACCEPTANCE_REFUSED,
        "source": "task_acceptance_review",
        "rationale": (
            "No new material since the paid panel: neither the candidate answer nor "
            "any obligation disposition changed. Quoting the recorded verdict "
            f"({str(result.aggregate_signal or 'DEGRADED').upper()}; dialogue "
            f"{dialogue['status']}) with {len(open_obligations)} open "
            "obligation(s); no further round."
        ),
        "dialogue_status": dialogue["status"],
        "dialogue_votes": dialogue["votes"],
        "dissent_noted": dissent,
        "open_obligations": [str(item.get("id")) for item in open_obligations],
    })
    ctx.emit_progress(
        f"Task acceptance review: {result.aggregate_signal} — identical paid identity "
        "(no changed answer, no new obligation disposition); the recorded verdict "
        "stands and no further panel is bought."
    )
    return False


def _apply_task_acceptance_result(
    ctx: Any,
    result: Any,
    *,
    record_run: bool = True,
    reused: bool = False,
) -> bool:
    """Apply one panel result; return whether the agent must take another round."""
    # loop.py keeps the fence, checkpoint, panel-record and message rails; binding
    # them through the MODULE (not by copying the names in) keeps one definition
    # and one patch surface for callers that already reach for ``ouroboros.loop``.
    from ouroboros import loop
    from ouroboros.review_substrate import (
        DIALOGUE_TERMINAL_STATUSES,
        aggregate_dialogue_status,
        build_improvement_capsule,
        dissent_findings,
        task_acceptance_is_clean,
    )

    if record_run:
        loop._record_host_acceptance_run(ctx, result)
    dissent = dissent_findings(result)
    blocking_lane = ctx.mode == "required" and loop.get_review_enforcement() == "blocking"
    # A REUSED panel (unchanged binding) is the SAME reviewer act applied
    # again: re-collecting would mutate reviewer-authored state with no new
    # input, and the shifted evidence revision would buy a fresh paid panel
    # for a byte-identical resubmit (fable r2 #1); rows already collected.
    if blocking_lane and not reused:
        _collect_acceptance_obligations(ctx.llm_trace, result)
    open_obligations = _open_acceptance_obligations(ctx.llm_trace) if blocking_lane else []
    # v6.74.0 (A1): the capsule leads with the verdict, the concrete open
    # obligation ids, and the pre-rendered rails line (money/time/rounds/passes).
    capsule = build_improvement_capsule(
        result,
        rails_line=ctx.rails_line,
        open_obligations=open_obligations,
    )
    # v6.74.0 (A5): the reviewers' typed dialogue judgement, reduced over the
    # CONTRIBUTING actors with the panel's own quorum; persisted for audit on
    # the authoritative run record whatever branch applies below. `inconclusive`
    # (no well-formed vote at all) grants the dialogue NO authority: it is not a
    # terminal verdict and not a licence to continue — the existing non-dialogue
    # terminals below decide, exactly as they did before the dialogue existed.
    dialogue = aggregate_dialogue_status(
        result, quorum=_acceptance_dialogue_quorum(result),
    )
    _attach_dialogue_to_host_run(ctx.llm_trace, dialogue)
    dialogue_terminal = dialogue["status"] in DIALOGUE_TERMINAL_STATUSES
    if reused and getattr(result, "replayed_from_superseded", False):
        # A run superseded by an evidence revision replays ONLY into the typed
        # identical-refusal terminal — never into clean-PASS authorization: its
        # verdict predates the evidence change, so re-accepting would stamp a
        # stale PASS (and the trace's superseded rows would contradict the
        # applied decision — the delivery binding could never match). The
        # refusal is conservative and consistent: nothing new was bought,
        # nothing stale is re-authorized.
        return _refuse_identical_acceptance(
            ctx, result,
            dialogue=dialogue, dissent=bool(dissent), open_obligations=open_obligations,
        )
    if task_acceptance_is_clean(result):
        ctx.tools._ctx._task_acceptance_reviewed = True
        loop._end_task_acceptance_fence(ctx.tools._ctx, outcome="terminal")
        loop._mark_root_acceptance_checkpoint(
            ctx.tools._ctx, ctx.llm_trace, status="pass", pass_index=ctx.passes_done,
        )
        if not _dispose_obligations_on_clean_pass(
            ctx.llm_trace, result, open_obligations, bool(dissent),
        ):
            _set_acceptance_decision(ctx.llm_trace, {
                "status": ACCEPTANCE_ACCEPTED,
                "reason": "clean_pass",
                "source": "task_acceptance_review",
                "rationale": "Quorum PASS classified the deliverable solved with criterion evidence.",
                "dissent_noted": bool(dissent),
            })
        ctx.emit_progress("Task acceptance review: PASS (clean acceptance).")
        return False
    if reused:
        return _refuse_identical_acceptance(
            ctx, result,
            dialogue=dialogue, dissent=bool(dissent), open_obligations=open_obligations,
        )

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
    # A DEGRADED panel (no valid verdict quorum) cannot "judge" the dialogue:
    # a lone terminal vote from the one contributing slot must NOT shadow the
    # review_degraded path below, which is the only surface carrying the
    # per-slot causes and degraded_reasons the v6.70.0 honesty invariant (P1)
    # requires. Letting the dialogue-terminal branch fire here recorded a false
    # "reviewer quorum judged" rationale and silently dropped those causes.
    if dialogue_terminal and str(result.aggregate_signal or "DEGRADED").upper() != "DEGRADED":
        # v6.74.0 (A5): a reviewer quorum judged the dialogue no longer
        # actionable (unreachable_here / stable_disagreement). Finalize via
        # the EXISTING honest path recording BOTH positions in one
        # owner-visible line — reviewer authorship, not a host timer.
        ctx.tools._ctx._task_acceptance_reviewed = True
        loop._end_task_acceptance_fence(ctx.tools._ctx, outcome="terminal")
        loop._mark_root_acceptance_checkpoint(
            ctx.tools._ctx,
            ctx.llm_trace,
            status=str(result.aggregate_signal or "DEGRADED").lower(),
            pass_index=ctx.passes_done,
        )
        _set_acceptance_decision(ctx.llm_trace, {
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
        _set_acceptance_decision(ctx.llm_trace, {
            "status": ACCEPTANCE_REVISION_REQUESTED,
            "reason": "improvement_capsule",
            "source": "task_acceptance_review",
            "rationale": "A compact advisory improvement capsule was fed back for one bounded revision pass.",
            "dissent_noted": bool(dissent),
        })
        ctx.tools._ctx._task_acceptance_improvement_passes = ctx.passes_done + 1
        if not loop._end_task_acceptance_fence(ctx.tools._ctx, outcome="revision"):
            ctx.tools._ctx._task_acceptance_reviewed = True
            _set_acceptance_decision(ctx.llm_trace, {
                "status": ACCEPTANCE_FINALIZED_UNACCEPTED,
                "reason": "fence_reopen_failed",
                "source": "task_acceptance_fence",
                "rationale": "The revision could not safely reopen queue admission at the dispatch boundary.",
            })
            return False
        if open_obligations:
            capsule += _format_obligations_clause(open_obligations)
        if ctx.content and ctx.content.strip():
            ctx.messages.append({"role": "assistant", "content": ctx.content})
        loop._append_or_merge_user_message(ctx.messages, capsule)
        ctx.emit_progress(
            f"Task acceptance review: {result.aggregate_signal} — improvement note fed back."
        )
        return True

    ctx.tools._ctx._task_acceptance_reviewed = True
    loop._end_task_acceptance_fence(ctx.tools._ctx, outcome="terminal")
    loop._mark_root_acceptance_checkpoint(
        ctx.tools._ctx,
        ctx.llm_trace,
        status=str(result.aggregate_signal or "DEGRADED").lower(),
        pass_index=ctx.passes_done,
    )
    if _dispose_obligations_on_clean_pass(
        ctx.llm_trace, result, open_obligations, bool(dissent),
    ):
        ctx.emit_progress(
            f"Task acceptance review: {result.aggregate_signal} (clean pass; obligations closed)."
        )
        return False
    aggregate_signal = str(result.aggregate_signal or "DEGRADED").upper()
    if aggregate_signal == "DEGRADED":
        _set_acceptance_decision(ctx.llm_trace, {
            "status": ACCEPTANCE_FINALIZED_UNACCEPTED,
            "reason": "review_degraded",
            "source": "task_acceptance_review",
            "rationale": "Acceptance reviewers did not reach a valid quorum.",
            "degraded_reasons": list(getattr(result, "degraded_reasons", []) or []),
            "open_obligations": [str(item.get("id")) for item in open_obligations],
        })
        # Per-slot causes were always in the structured decision; the
        # owner-visible line said only "no valid quorum", forcing a dig
        # through task_results for WHICH slot failed and why (v6.70.0).
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
        _set_acceptance_decision(ctx.llm_trace, {
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
        _set_acceptance_decision(ctx.llm_trace, {
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
        _set_acceptance_decision(ctx.llm_trace, {
            "status": ACCEPTANCE_FINALIZED_UNACCEPTED,
            "reason": "reviewer_fail_no_capsule",
            "source": "task_acceptance_review",
            "rationale": "A valid acceptance reviewer FAIL had no additional capsule text.",
            "dissent_noted": bool(dissent),
        })
        ctx.emit_progress("Task acceptance review: FAIL (finalizing with a failed review verdict).")
    elif open_obligations:
        _set_acceptance_decision(ctx.llm_trace, {
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
        _set_acceptance_decision(ctx.llm_trace, {
            # Round-9 CRITICAL 1: fall-through AFTER
            # `task_acceptance_is_clean` refused the panel, so it cannot mint
            # `accepted` (reserved for clean acceptance). Reachable: a
            # reviewer claims `solved` with a MISSING criterion and the
            # improvement cap spent — nothing actionable, yet not "accepted";
            # the typed reason names WHY; tier honesty rides `outcome_tier`.
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


def _disposition_reason_sha256(reason: Any) -> str:
    """Content identity of one obligation-disposition reason; "" when blank.

    Mirrors ``commit_gate.compute_rebuttal_sha256`` on purpose: on BOTH gates an
    empty rebuttal is not an argument and buys no paid cycle."""
    text = str(reason or "").strip()
    if not text:
        return ""
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def acceptance_paid_identity(candidate_hash: str, llm_trace: Dict[str, Any]) -> str:
    """The identity ONE paid acceptance panel is claimed under (A-material).

    ``sha256(candidate_hash + the sorted set of nonempty (obligation_id,
    disposition, sha256(reason)) tuples)``. Exactly two things mint a new paid
    panel: a changed candidate answer, or an obligation disposition whose content
    the reviewers have not answered yet. The evidence revision is deliberately NOT
    in here — every cosmetic tool call moves it, which is how one task bought 21
    paid panels; it stays what it always was, stale-packet detection for the
    supersede paths. A disposition with an empty reason contributes nothing.
    Rows are read live from the agent's own ``acceptance_obligations`` (the
    ``task_acceptance_review`` tool stamps ``status="agent_disposed"`` there)."""
    material = sorted({
        (
            str(row.get("id") or "").strip(),
            str(row.get("disposition") or "").strip().lower(),
            _disposition_reason_sha256(row.get("disposition_reason")),
        )
        for row in (llm_trace.get("acceptance_obligations") or [])
        if isinstance(row, dict)
        and str(row.get("id") or "").strip()
        and str(row.get("disposition") or "").strip()
        and _disposition_reason_sha256(row.get("disposition_reason"))
    })
    payload = json.dumps(
        [str(candidate_hash or ""), [list(item) for item in material]],
        ensure_ascii=False, separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def bind_acceptance_paid_identity(
    review_binding: Dict[str, Any], llm_trace: Dict[str, Any],
) -> str:
    """Stamp the A-material paid identity onto a freshly built review binding.

    The binding keeps carrying its three hashes (the supersede paths still need
    the evidence revision); ``paid_identity`` rides ALONGSIDE them and is what the
    wallet claim and the free-replay lookup key on."""
    identity = acceptance_paid_identity(
        str(review_binding.get("candidate_hash") or ""), llm_trace,
    )
    review_binding["paid_identity"] = identity
    return identity


def acceptance_dialogue_history(llm_trace: Dict[str, Any], *, limit: int = 6) -> List[Dict[str, Any]]:
    """Bounded per-panel history of the dialogue so far, for the NEXT reviewer.

    Reviewers were adjudicating each round blind to the previous rounds' typed
    judgement, which is most of why the same finding kept being re-raised. The
    rows are tiny host facts already recorded on the run records; the caller
    attaches them to the evidence packet OUTSIDE the hashed material
    (``review_evidence.UNHASHED_EVIDENCE_KEYS``) so reading the history can never
    mint a fresh evidence revision — and therefore never a fresh paid binding."""
    rows: List[Dict[str, Any]] = []
    for run in (llm_trace.get("review_runs") or []):
        if not isinstance(run, dict) or str(run.get("authority") or "") != "host_root":
            continue
        dialogue = run.get("dialogue") if isinstance(run.get("dialogue"), dict) else {}
        votes = dialogue.get("votes") if isinstance(dialogue.get("votes"), dict) else {}
        rows.append({
            "round": len(rows) + 1,
            "aggregate_signal": str(run.get("aggregate_signal") or "").upper(),
            "dialogue_status": str(dialogue.get("status") or ""),
            "votes": {str(k): len(v or []) for k, v in votes.items()},
        })
    obligations = [
        row for row in (llm_trace.get("acceptance_obligations") or [])
        if isinstance(row, dict)
    ]
    if rows:
        rows[-1]["obligations_new"] = sum(
            1 for row in obligations if not int(row.get("reopened_count") or 0)
        )
        rows[-1]["obligations_re_raised"] = sum(
            1 for row in obligations if int(row.get("reopened_count") or 0)
        )
    return rows[-max(1, int(limit)):]


def _prior_acceptance_run(
    tools_ctx: Any,
    llm_trace: Dict[str, Any],
    binding_hash: str,
    *,
    paid_identity: str = "",
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """Locate the authoritative host run already recorded for this submission:
    first the trace (survives requeue replay), then the process-local
    ``_task_acceptance_seen_bindings`` cache. Returns (cache, prior_run).

    EITHER identity replays for free: the same binding hash (byte-identical
    submission, as before) OR the same A-material ``paid_identity`` — unchanged
    candidate answer and no new obligation disposition — which is the identity the
    tree's wallet actually bought."""
    seen_bindings = getattr(tools_ctx, "_task_acceptance_seen_bindings", None)
    if not isinstance(seen_bindings, dict):
        seen_bindings = {}
        tools_ctx._task_acceptance_seen_bindings = seen_bindings
    identity = str(paid_identity or "")

    def _matches(run: Any) -> bool:
        return isinstance(run, dict) and (
            str(run.get("binding_hash") or "") == binding_hash
            or bool(identity and str(run.get("paid_identity") or "") == identity)
        )

    prior_run = next(
        (
            run for run in reversed(llm_trace.get("review_runs") or [])
            if isinstance(run, dict)
            and run.get("authority") == "host_root"
            and not run.get("superseded_by_revision")
            and _matches(run)
        ),
        None,
    )
    if prior_run is None:
        prior_run = next(
            (
                run for run in reversed(list(seen_bindings.values()))
                if isinstance(run, dict)
                and not run.get("superseded_by_revision")
                and _matches(run)
            ),
            None,
        )
    if prior_run is None and identity:
        # A run superseded by an evidence revision is stale as a CURRENT
        # acceptance, but the wallet already bought its A-material. When the
        # resubmission carries the SAME paid identity (unchanged candidate, no
        # new nonempty disposition), the recorded verdict replays for free —
        # otherwise the dispatch claim refuses `binding_dispatch_already_claimed`
        # and the loop records a synthetic DEGRADED panel instead of the typed
        # identical-refusal terminal the contract requires. Evidence revision
        # is stale-DETECTION, never a paid-cycle mint (owner decision 5=A).
        superseded = next(
            (
                run for run in reversed(llm_trace.get("review_runs") or [])
                if isinstance(run, dict)
                and run.get("authority") == "host_root"
                and run.get("superseded_by_revision")
                and str(run.get("paid_identity") or "") == identity
            ),
            None,
        )
        if superseded is not None:
            prior_run = dict(superseded)
            prior_run["replayed_from_superseded"] = True
    return seen_bindings, prior_run
