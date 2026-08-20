"""Pure reducers from panel actor rows to a verdict, a tier, and a capsule.

Owns the read-only judgement layer above a completed review run: which actors
contributed to the aggregate, the worst outcome tier among them, the
release-clean acceptance bit, the reviewer-authored dialogue vote, the one
honest reason line naming the real blocker, minority dissent, and the
improvement capsule fed back to the agent. Every function here is a pure read
of records the coordinator already produced; nothing in this module runs a
reviewer, persists anything, or mutates a record.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List

# Tier vocabulary SSOT lives in outcomes.py; reuse it so a future tier rename
# cannot silently desync the capsule from the objective axis.
from ouroboros.outcomes import OUTCOME_TIER_BEST_EFFORT, OUTCOME_TIER_BLOCKED, OUTCOME_TIER_SOLVED
from ouroboros.review_records import ReviewRunResult
from ouroboros.utils import truncate_review_artifact


_TIER_ORDER = {OUTCOME_TIER_SOLVED: 0, OUTCOME_TIER_BEST_EFFORT: 1, OUTCOME_TIER_BLOCKED: 2}


_CRITERION_STATUSES = frozenset({"supported", "missing", "partial", "rejected"})


def _criteria_have_supported_evidence(criteria: Any) -> bool:
    return bool(isinstance(criteria, list) and criteria and all(
        isinstance(item, dict)
        and bool(str(item.get("criterion") or "").strip())
        and str(item.get("status") or "").strip().lower() == "supported"
        and bool(item.get("evidence_refs"))
        for item in criteria
    ))


def _criteria_shape_valid(criteria: Any, tier: str) -> bool:
    """Shape + tier coherence for a reviewer's criteria_used (v6.71.1).

    SHAPE: a non-empty list of {criterion, status ∈ enum}, and every 'supported'
    criterion names evidence_refs. COHERENCE: 'solved' still requires ALL criteria
    'supported' with refs — the release-clean bar (task_acceptance_is_clean) is
    unchanged; a non-solved tier (best_effort / blocked_with_evidence) may honestly
    carry partial/missing/rejected criteria. This lets an honest PASS that marks one
    criterion 'partial' contribute as a valid NON-clean vote instead of being demoted
    to parse_status=malformed — the old all-must-be-'supported' gate (the prompt itself
    offers 'partial') silently starved the honest-partial path and fueled acceptance
    loops (BIBLE P2/P3; the FAIL-veto and clean-solved contracts are untouched)."""
    if not (isinstance(criteria, list) and criteria):
        return False
    for item in criteria:
        if not isinstance(item, dict):
            return False
        if not str(item.get("criterion") or "").strip():
            return False
        status = str(item.get("status") or "").strip().lower()
        if status not in _CRITERION_STATUSES:
            return False
        if status == "supported" and not item.get("evidence_refs"):
            return False
    if str(tier or "").strip().lower() == OUTCOME_TIER_SOLVED:
        return _criteria_have_supported_evidence(criteria)
    return True


def _contributing_actors(result: ReviewRunResult) -> List[Dict[str, Any]]:
    """Actors whose verdict CONTRIBUTED to the aggregate, so a parse-degraded or
    non-responsive slot cannot inject a tier / coach / finding into a clean quorum
    result (Bible P3: one degraded slot must not poison the aggregate — the exact
    class the split-participation gate was built to avoid). For aggregate PASS only
    PASS actors speak; for FAIL only FAIL actors; for a DEGRADED/UNKNOWN aggregate
    only the cleanly-parsed PASS/FAIL actors may speak (never the degraded ones)."""
    actors = [a for a in (getattr(result, "actors", None) or []) if isinstance(a, dict)]
    agg = str(getattr(result, "aggregate_signal", "") or "").upper()
    if agg in ("PASS", "FAIL"):
        return [a for a in actors if str(a.get("signal", "")).upper() == agg]
    return [a for a in actors if str(a.get("signal", "")).upper() in ("PASS", "FAIL")]


def aggregate_outcome_tier(result: ReviewRunResult) -> str:
    """Worst-tier-wins across the actors that CONTRIBUTED to the aggregate verdict."""
    worst, worst_rank = "", -1
    for actor in _contributing_actors(result):
        parsed = actor.get("parsed") if isinstance(actor, dict) else None
        tier = str((parsed or {}).get("outcome_tier") or "").strip().lower() if isinstance(parsed, dict) else ""
        rank = _TIER_ORDER.get(tier, -1)
        if rank > worst_rank:
            worst_rank, worst = rank, tier
    return worst


def task_acceptance_is_clean(result: Any) -> bool:
    """Whether a task-acceptance verdict satisfies the release-clean contract.

    The evidence condition is UNCONDITIONAL (D-Q5 deleted the constant-true
    ``require_criterion_evidence`` knob — the v6.60.0 dead-key precedent), and a
    'supported' criterion counts only when ≥1 of its ``evidence_refs`` RESOLVED
    against the packet (host annotation stamped at panel time; absent on
    historical rows — forward-only). Both demote ONLY this clean bit onto the
    existing non-clean rails; parse validity/quorum/verdicts untouched (v6.71.1)."""
    if str(getattr(result, "aggregate_signal", "") or "").upper() != "PASS" or bool(getattr(result, "degraded", False)):
        return False
    contributing = _contributing_actors(result)
    if not contributing:
        return False
    for actor in contributing:
        parsed = actor.get("parsed") if isinstance(actor, dict) else None
        if not isinstance(parsed, dict) or str(parsed.get("outcome_tier") or "").lower() != OUTCOME_TIER_SOLVED:
            return False
        if not _criteria_have_supported_evidence(parsed.get("criteria_used")):
            return False
        if any(isinstance(r, dict) and not r.get("supported_evidence_resolves")
               for r in (actor.get("criteria_refs_unresolved") or [])):
            return False
    return True


# v6.74.0 (A5): reviewer-authored dialogue status. The reviewer — not a host
# counter or hash — judges whether the acceptance dialogue is still actionable.
DIALOGUE_CONTINUE = "continue_actionable"
DIALOGUE_UNREACHABLE = "unreachable_here"
DIALOGUE_STABLE_DISAGREEMENT = "stable_disagreement"
DIALOGUE_STATUS_VALUES = (DIALOGUE_CONTINUE, DIALOGUE_UNREACHABLE, DIALOGUE_STABLE_DISAGREEMENT)


def _contract_valid_actors(result: Any) -> List[Dict[str, Any]]:
    """Actors with a DELIBERATE, CONTRACT-VALID reviewer object: parsed dict,
    recognizable verdict, parse_status not "malformed". Wider than
    ``_contributing_actors`` (deliberate DEGRADED keeps its vote, sol #3) but a
    contract-DEMOTED/garbage response never votes terminal (commit triad #1)."""
    out: List[Dict[str, Any]] = []
    for actor in (getattr(result, "actors", None) or []):
        row = actor if isinstance(actor, dict) else asdict(actor)
        parsed = row.get("parsed")
        if str(row.get("parse_status") or "") == "malformed":
            continue
        if isinstance(parsed, dict) and str(
            parsed.get("verdict") or parsed.get("status") or ""
        ).strip().upper() in {"PASS", "FAIL", "DEGRADED"}:
            out.append(row)
    return out


def aggregate_dialogue_status(result: Any, *, quorum: int) -> Dict[str, Any]:
    """Pure reducer over the reviewers' typed ``dialogue_status`` votes (A5, P5):
    the host validates the enum, applies the caller's quorum, and transports the
    result. Precedence: any continue vote from a QUORUM-CONTRIBUTING actor keeps
    the loop; else a quorum of terminal votes terminates. Missing/invalid votes
    default to ``continue_actionable`` (fail-safe, backward-compatible).
    Returns ``{"status", "votes"}`` with the full distribution for audit."""
    contributing = {str(a.get("slot_id", "")) for a in _contributing_actors(result)}
    votes: Dict[str, List[str]] = {}
    for row in _contract_valid_actors(result):
        parsed = row.get("parsed") if isinstance(row.get("parsed"), dict) else {}
        vote = str(parsed.get("dialogue_status") or "").strip().lower()
        if vote not in DIALOGUE_STATUS_VALUES:
            vote = DIALOGUE_CONTINUE
        votes.setdefault(vote, []).append(str(row.get("slot_id", "")))
    continue_slots = votes.get(DIALOGUE_CONTINUE, [])
    unreachable = votes.get(DIALOGUE_UNREACHABLE, [])
    disagreement = votes.get(DIALOGUE_STABLE_DISAGREEMENT, [])
    terminal = unreachable + disagreement
    if any(slot in contributing for slot in continue_slots):
        status = DIALOGUE_CONTINUE
    elif len(terminal) >= max(1, int(quorum)):
        status = (
            DIALOGUE_UNREACHABLE
            if len(unreachable) >= len(disagreement)
            else DIALOGUE_STABLE_DISAGREEMENT
        )
    else:
        status = DIALOGUE_CONTINUE
    return {"status": status, "votes": votes}


def _unresolved_evidence_ref_labels(run: Any) -> List[str]:
    """The D-Q5 refs a contributing actor cited that did NOT resolve, as
    ``ref (basis)`` labels (or the panel-wide ``host_resolution_unavailable``).

    Pure read of the deciding detail already recorded on the actor rows, so
    ``panel_reason`` can name the REAL blocker: on a D-Q5 demotion every criterion
    IS marked supported with refs, and the criteria-support line would describe a
    condition that is already satisfied."""
    from ouroboros.review_evidence_refs import NON_RESOLVING_BASIS_KINDS

    labels: List[str] = []
    for actor in _contributing_actors(run):
        for row in (actor.get("criteria_refs_unresolved") or []):
            if not isinstance(row, dict) or row.get("supported_evidence_resolves"):
                continue
            if str(row.get("resolution_status") or ""):
                labels.append(str(row["resolution_status"]))
                continue
            for ref in (row.get("refs") or []):
                if not isinstance(ref, dict):
                    continue
                basis = str(ref.get("resolved_as") or "")
                if basis and basis not in NON_RESOLVING_BASIS_KINDS:
                    continue  # this one resolved; it is not the blocker
                labels.append(f"{str(ref.get('ref') or '?')[:80]} ({basis or 'no packet entry'})")
    return list(dict.fromkeys(labels))


def panel_reason(run: Any) -> str:
    """One honest reason line naming the REAL blocker (v6.74.0, A6); shared by
    the capsule header, the compact projection fallback, and progress lines.
    Accepts a ``ReviewRunResult`` or its dict/namespace record."""
    from types import SimpleNamespace

    if isinstance(run, dict):
        run = SimpleNamespace(**run)
    aggregate = str(getattr(run, "aggregate_signal", "") or "UNKNOWN").upper()
    tier = aggregate_outcome_tier(run)
    if aggregate == "PASS":
        if task_acceptance_is_clean(run):
            return "clean acceptance"
        # A D-Q5 demotion leaves every criterion marked supported WITH refs, so the
        # criteria-support line below would name a condition that is already
        # satisfied. Name the ref that actually decided instead.
        unresolved = _unresolved_evidence_ref_labels(run)
        if unresolved:
            more = f" (+{len(unresolved) - 3} more)" if len(unresolved) > 3 else ""
            return (
                f"tier={tier or 'unclassified'} — cited evidence does not resolve "
                f"against the packet: {', '.join(unresolved[:3])}{more}"
            )
        return (
            f"tier={tier or 'unclassified'} — a PASS is not release-clean until "
            "every criterion is supported"
        )
    if aggregate == "FAIL":
        fail_slots = {
            str(actor.get("slot_id", ""))
            for actor in _contributing_actors(run)
            if str(actor.get("signal", "")).upper() == "FAIL"
        }
        named = ""
        for actor in _contributing_actors(run):
            if str(actor.get("signal", "")).upper() != "FAIL":
                continue
            parsed = actor.get("parsed") if isinstance(actor.get("parsed"), dict) else {}
            for finding in (parsed.get("findings") or []):
                if isinstance(finding, dict):
                    named = str(finding.get("item") or finding.get("recommendation") or "").strip()
                    if named:
                        break
            if not named:
                named = str(parsed.get("summary") or "").strip()
            if named:
                break
        if not named:
            # Coordinator-flattened findings (slot_id-stamped) from FAIL slots.
            for finding in (getattr(run, "parsed_findings", None) or []):
                if isinstance(finding, dict) and str(finding.get("slot_id", "")) in fail_slots:
                    named = str(finding.get("item") or finding.get("recommendation") or "").strip()
                    if named:
                        break
        if named:
            compact = truncate_review_artifact(" ".join(named.split()), limit=300)
            return f"tier={tier or 'unclassified'} — {compact}"
        return f"tier={tier or 'unclassified'} — reviewer FAIL without a named finding"
    reasons = [str(r) for r in (getattr(run, "degraded_reasons", None) or []) if str(r)]
    if len(reasons) > 4:
        return "; ".join(reasons[:4]) + f" ⚠️ OMISSION NOTE: +{len(reasons) - 4} more causes in the run record"
    return "; ".join(reasons) or "no valid reviewer quorum"


def dissent_findings(result: ReviewRunResult, *, limit: int = 1) -> List[str]:
    """Compact dissent bullets from NON-contributing minority reviewers (v6.54.4).

    A cleanly-parsed reviewer whose verdict differs from the aggregate AND who
    carries a CONCRETE recommendation/alternative contributes ONE verbatim
    "[DISSENT — slot N]: ..." line. Not a veto — the aggregate stands; this ends
    the class where an aggregate-PASS silently discarded a minority FAIL whose
    concrete recommendation was correct (GAIA 3cef3a44). A DELIBERATE minority
    DEGRADED — the reviewer's own parsed verdict (the prompt's "cannot judge →
    return DEGRADED and explain" branch, which is exactly what the 3cef3a44
    reviewer returned) — may dissent too, but only on the strength of a concrete
    findings[].recommendation. Parse-fail placeholders (parsed=None),
    contract-demoted PASSes (their parsed verdict stays PASS — they agree with
    the aggregate), and coach-only DEGRADED stay excluded (no clean dissenting
    signal). ONE bullet by design (plan decision #13) — the first concrete
    dissenter speaks."""
    agg = str(getattr(result, "aggregate_signal", "") or "").upper()
    contributing_ids = {str(a.get("slot_id", "")) for a in _contributing_actors(result)}
    out: List[str] = []
    for actor in (getattr(result, "actors", None) or []):
        if not isinstance(actor, dict) or len(out) >= limit:
            continue
        slot_id = str(actor.get("slot_id", ""))
        signal = str(actor.get("signal", "")).upper()
        if slot_id in contributing_ids or signal == agg:
            continue
        parsed = actor.get("parsed") if isinstance(actor.get("parsed"), dict) else {}
        deliberate_degraded = (
            signal == "DEGRADED"
            and str(parsed.get("verdict") or "").strip().upper() == "DEGRADED"
        )
        if signal not in ("PASS", "FAIL") and not deliberate_degraded:
            continue
        recommendation = ""
        for finding in (parsed.get("findings") or []):
            if isinstance(finding, dict):
                recommendation = str(finding.get("recommendation") or "").strip()
                if recommendation:
                    break
        if not recommendation and not deliberate_degraded:
            recommendation = str(parsed.get("completion_coach") or "").strip()
        if not recommendation:
            continue  # a bare contrary verdict with no concrete alternative is noise
        compact = " ".join(recommendation.split())
        if len(compact) > 300:
            compact = compact[:300].rstrip() + "…"
        out.append(f"[DISSENT — {slot_id} said {signal}]: check this before finalizing — {compact}")
    return out


def build_improvement_capsule(
    result: ReviewRunResult,
    *,
    rails_line: str = "",
    open_obligations: List[Dict[str, Any]] | None = None,
) -> str:
    """Compact, anti-derailment "Final improvement note" fed back to the agent:
    the actual verdict + tier + real blocker (v6.74.0, A1 — today only the tier
    label printed), the concrete open obligation ids, one pre-rendered rails
    line (money/time/rounds/passes headroom, assembled by the caller from the
    real sources), exact-deduplicated actionable findings, and one
    completion_coach. Returns "" when there is nothing actionable. The full
    ReviewRunResult stays on the objective axis / trace; the agent sees only this
    capsule, so it does not rewrite its deliverable into a meta-essay about the
    review (the failure mode that made the host-forced path label-only).

    Tier, coach, and bullets are drawn ONLY from the actors that contributed to the
    aggregate verdict, so a single parse-degraded slot cannot inject a blocking note
    into an otherwise-clean quorum PASS."""
    aggregate_signal = str(getattr(result, "aggregate_signal", "") or "").upper()
    contributing = _contributing_actors(result)
    # A semantic DEGRADED verdict abstains from quorum, but a concrete finding is
    # still an owner-approved correction rail for the required+blocking re-drive.
    # Transport/parse placeholders and contract-demoted PASS/FAIL actors remain
    # excluded: only an explicitly parsed verdict=DEGRADED may supply ANY capsule
    # content (tier, coach, or finding) when the aggregate itself is DEGRADED.
    deliberate_degraded = [
        actor
        for actor in (getattr(result, "actors", None) or [])
        if (
            aggregate_signal == "DEGRADED"
            and isinstance(actor, dict)
            and str(actor.get("signal") or "").upper() == "DEGRADED"
            and isinstance(actor.get("parsed"), dict)
            and str(actor["parsed"].get("verdict") or "").strip().upper() == "DEGRADED"
        )
    ]
    eligible_actors = deliberate_degraded if aggregate_signal == "DEGRADED" else contributing
    eligible_slots = {str(actor.get("slot_id", "")) for actor in eligible_actors}
    tier = ""
    tier_rank = -1
    for actor in eligible_actors:
        parsed = actor.get("parsed") if isinstance(actor, dict) else None
        actor_tier = (
            str(parsed.get("outcome_tier") or "").strip().lower()
            if isinstance(parsed, dict)
            else ""
        )
        actor_rank = _TIER_ORDER.get(actor_tier, -1)
        if actor_rank > tier_rank:
            tier_rank, tier = actor_rank, actor_tier
    coach = ""
    for actor in eligible_actors:
        parsed = actor.get("parsed") if isinstance(actor, dict) else None
        if isinstance(parsed, dict) and not coach:
            coach = str(parsed.get("completion_coach") or "").strip()
        if coach:
            break
    bullets: List[str] = []
    seen_bullets: set[str] = set()
    for finding in (getattr(result, "parsed_findings", None) or []):
        if not isinstance(finding, dict):
            continue
        # Only findings from a contributing actor may surface in the capsule.
        if str(finding.get("slot_id", "")) not in eligible_slots:
            continue
        text = str(finding.get("recommendation") or finding.get("item") or "").strip()
        # Exact normalized deduplication only.  Do not introduce semantic
        # clustering or another findings authority for the improvement loop.
        dedup_key = " ".join(text.split())
        if text and dedup_key not in seen_bullets:
            seen_bullets.add(dedup_key)
            bullets.append(text)
    # A SOLVED review carries a (contract-required) completion_coach, but a coach
    # alone must NOT force a revise round on an already-solved deliverable — that
    # would re-loop EVERY clean required review. The capsule is actionable only
    # when there are real findings to act on OR the tier itself is incomplete
    # (best_effort/blocked). The coach is then included as the next step.
    dissent = dissent_findings(result)
    # A coach alone stays non-actionable for a clean SOLVED PASS, but it is the
    # bounded correction rail for a contributing FAIL.  The coordinator admits a
    # task-acceptance FAIL only when this function can return such a rail.
    actionable = (
        bool(bullets)
        or bool(dissent)
        or (
            aggregate_signal == "FAIL"
            and bool(coach)
        )
        or tier in (OUTCOME_TIER_BEST_EFFORT, OUTCOME_TIER_BLOCKED)
    )
    if not actionable:
        return ""
    # Lead with the actual outcome (A1): verdict + tier + the real blocker, so
    # the agent sees WHAT failed instead of a bare ledger label.
    header = f"[Final improvement note] Review verdict: {aggregate_signal or 'UNKNOWN'}"
    if tier:
        header += f" (tier: {tier})"
    header += f" — {panel_reason(result)}."
    lines = [header]
    open_ids = [
        str(o.get("id"))
        for o in (open_obligations or [])
        if isinstance(o, dict) and o.get("id")
    ]
    if open_ids:
        lines.append(
            f"Open blocking obligation(s) ({len(open_ids)}): " + ", ".join(open_ids) + "."
        )
    if rails_line:
        lines.append(f"Remaining headroom — {rails_line}.")
    # Dissent rides ON TOP of the capsule (v6.54.4): same anti-derailment frame,
    # never a veto — a minority reviewer with a concrete recommendation is a
    # "check this before finalizing" pointer, not a re-litigation of the verdict.
    lines += dissent
    lines += [f"- {b}" for b in bullets]
    if coach:
        lines.append(f"Highest-value next step: {coach}")
    lines.append(
        # The three real moves (A1): the old "revise only if it genuinely
        # improves the result; otherwise produce your normal final answer" tail
        # was the measured cause of the do-nothing resubmit loop (SWE 1b311217:
        # 7 passes, zero tool calls). The anti-derailment guards stay verbatim.
        "Three real moves are available: (1) FIX — change the work/answer so the next panel is "
        "clean; (2) REBUT — file obligation_dispositions (rejected + your reason) via the "
        "task_acceptance_review tool for findings you can show are wrong; the reviewer "
        "adjudicates the argument; (3) DECLARE UNREACHABLE — dispose an obligation as "
        "unsatisfiable in this environment (rejected + the concrete gap), and the reviewer "
        "judges reachability. Resubmitting the same answer with none of these moves changes "
        "nothing. "
        "Do not mention this review or the reviewer unless the user asked. "
        "The assessment tier above is an internal ledger label — never emit an internal ledger "
        "identifier as the deliverable itself."
    )
    return "\n".join(lines)
