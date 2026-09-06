"""Contract aggregation for completed review actor rows."""

from __future__ import annotations

from typing import Any, Callable, Dict, List

from ouroboros.triad_review import parse_review_findings


def contract_valid_actors(result: Any) -> List[Dict[str, Any]]:
    """Actors with a DELIBERATE, CONTRACT-VALID reviewer object: parsed dict,
    recognizable verdict, parse_status not "malformed" — so a contract-DEMOTED or
    garbage response never votes (commit triad #1).

    Owner ratification 2026-08-30: the acceptance-dialogue reducer now counts
    votes over ``_contributing_actors`` (a slot whose verdict did not reach the
    aggregate must not steer the loop either), and applies THIS predicate as the
    validity gate on top — the two are not the same test, and a hand-built or
    legacy row carrying a PASS/FAIL signal beside a malformed parse must still be
    unable to vote. It moved here from ``review_substrate`` with that change: this
    module is where the reviewer-row contract is decided, and the substrate sits
    exactly on its module-size cap."""
    from dataclasses import asdict

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


def continue_vote_is_well_formed(parsed: Dict[str, Any]) -> bool:
    """Whether a ``continue_actionable`` dialogue vote came with MATERIAL (owner
    ratification 2026-08-30, Rule 1; consumed by
    ``review_substrate.aggregate_dialogue_status``).

    Majority voting was rejected: one strong reviewer may still hold the
    acceptance loop open — but only while it can say what to do next. The same
    response must carry a concrete finding OR a completion_coach line; the
    correction-rail contract below already accepts coach-without-findings, so
    both count here. A bare "keep going" with neither is not a judgement the
    agent can act on: it bought one paid panel per round and never converged."""
    findings = parsed.get("findings")
    if isinstance(findings, list) and any(
        isinstance(item, dict)
        and str(item.get("item") or item.get("recommendation") or "").strip()
        for item in findings
    ):
        return True
    return bool(str(parsed.get("completion_coach") or "").strip())


def aggregate_review_actors(
    *,
    request: Any,
    slots: List[Any],
    actors: List[Any],
    slots_by_id: Dict[str, Any],
    actor_projection: Callable[[Any, str], Dict[str, Any]],
    criteria_shape_valid: Callable[[Any, str], bool],
    advisory_hardness: str,
) -> Dict[str, Any]:
    """Apply participation, tier-contract, quorum, and enforcement semantics."""
    all_findings: List[Dict[str, Any]] = []
    # Split participation faults (a slot errored / timed out / returned empty)
    # from parse-degraded (a slot produced a DEGRADED verdict or unparseable
    # text). Only a participation fault fail-closes: a single Markdown/non-JSON
    # slot must NOT poison a clean quorum PASS.
    actor_errors: List[str] = []
    parse_degraded: List[str] = []
    fail_count = 0
    pass_count = 0
    classify_tier = bool(
        request.surface == "task_acceptance"
        and (request.policy or {}).get("classify_outcome_tier")
    )
    valid_tiers = {"solved", "best_effort", "blocked_with_evidence"}
    is_advisory = (
        request.surface == "task_acceptance"
        or str((request.policy or {}).get("hardness") or "") == advisory_hardness
    )
    for actor in actors:
        if actor.status in {"error", "not_dispatched"}:
            actor_errors.append(f"{actor.slot_id}:{actor.error}")
        elif actor.status != "ok":
            actor_errors.append(f"{actor.slot_id}:{actor.status}")
        parsed, findings, signal = parse_review_findings(actor.raw_text)
        actor.parsed = parsed
        actor.signal = signal
        slot = slots_by_id.get(actor.slot_id)
        actor.actor_role = (
            str(getattr(slot, "role_hint", "") or "").strip()
            or f"{request.surface} reviewer"
        )
        truth = actor_projection(actor, request.surface)
        for key in (
            "model", "transport_status", "parse_status", "semantic_verdict", "provider",
            "coverage", "reason",
        ):
            setattr(actor, key, truth[key])
        all_findings.extend(
            {**item, "slot_id": actor.slot_id, "model": actor.model}
            for item in findings
        )
        tier = (
            str(parsed.get("outcome_tier") or "").strip().lower()
            if isinstance(parsed, dict) else ""
        )
        criteria = parsed.get("criteria_used") if isinstance(parsed, dict) else None
        criteria_ok = criteria_shape_valid(criteria, tier)
        contract_ok = (
            tier in valid_tiers
            and (
                bool(str((parsed or {}).get("completion_coach") or "").strip())
                or (is_advisory and tier == "solved")
            )
            and criteria_ok
        )
        if signal == "FAIL":
            has_concrete_finding = any(
                isinstance(item, dict)
                and bool(str(item.get("recommendation") or item.get("item") or "").strip())
                for item in findings
            )
            parsed_obj = parsed if isinstance(parsed, dict) else {}
            has_correction_rail = (
                bool(str(parsed_obj.get("completion_coach") or "").strip())
                or has_concrete_finding
                or tier in {"best_effort", "blocked_with_evidence"}
            )
            if classify_tier and (tier not in valid_tiers or not has_correction_rail):
                parse_degraded.append(
                    f"{actor.slot_id}:fail_missing_tier_or_correction_rail"
                )
                actor.signal = "DEGRADED"
                actor.parse_status = "malformed"
                actor.semantic_verdict = ""
                actor.reason = (
                    "Reviewer response violated the required outcome-tier or "
                    "correction-rail contract."
                )
            else:
                fail_count += 1
        elif signal == "PASS" and classify_tier and not contract_ok:
            parse_degraded.append(
                f"{actor.slot_id}:missing_tier_coach_or_criterion_evidence"
            )
            actor.signal = "DEGRADED"
            actor.parse_status = "malformed"
            actor.semantic_verdict = ""
            actor.reason = (
                "Reviewer response violated the required outcome-tier, coach, "
                "or criterion-evidence contract."
            )
        elif signal == "PASS":
            pass_count += 1
        elif signal == "DEGRADED":
            parse_degraded.append(f"{actor.slot_id}:degraded")

    min_successful = max(
        1, int((request.policy or {}).get("min_successful_slots") or 1)
    )
    fail_closed_on_errors = bool((request.policy or {}).get("fail_closed_on_errors"))
    degraded_reasons = actor_errors + parse_degraded
    if fail_count >= 1:
        aggregate = "FAIL"
    elif pass_count >= min_successful and not (
        fail_closed_on_errors and actor_errors and request.surface != "task_acceptance"
    ):
        aggregate = "PASS"
    else:
        aggregate = "DEGRADED"
        if not degraded_reasons:
            degraded_reasons.append(
                f"quorum_not_met: pass_count={pass_count} < min_successful={min_successful}"
            )

    participating_ids = {
        actor.slot_id
        for actor in actors
        if str(actor.signal or "").upper() in {"PASS", "FAIL"}
    }
    for actor in actors:
        actor.quorum_contribution = actor.slot_id in participating_ids
        if not actor.quorum_contribution:
            actor.enforcement_impact = "abstains"
        elif str(actor.signal or "").upper() == "FAIL":
            actor.enforcement_impact = "veto"
        else:
            actor.enforcement_impact = "supports_pass"
    return {
        "parsed_findings": all_findings,
        "aggregate_signal": aggregate,
        "degraded": aggregate == "DEGRADED",
        "degraded_reasons": degraded_reasons,
        "single_reviewer_no_diversity": len(slots) == 1,
    }
