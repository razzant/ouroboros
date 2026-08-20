"""Panel identity and the compact, redacted projection of a review run.

Owns the outward-facing view of a completed panel: transport-failure
classification, redaction of model-authored reason text, the per-actor and
per-panel projections that task results and the UI consume, the enforcement
impact label, and the two hashes that give a panel its identity (the actor
digest and the candidate/evidence/fence binding). Projection never decides a
verdict — it reads the records and the reducers and publishes them.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Any, Dict, List

from ouroboros.observability import redact_projection
from ouroboros.outcomes import OUTCOME_TIER_BEST_EFFORT, OUTCOME_TIER_BLOCKED, OUTCOME_TIER_SOLVED
from ouroboros.provider_models import provider_for_model
from ouroboros.review_records import (
    HARDNESS_ADVISORY_VISIBLE,
    HARDNESS_HARD_GATE,
    ReviewActorRecord,
    ReviewRequest,
)
from ouroboros.review_verdict import DIALOGUE_STATUS_VALUES, panel_reason


def _transport_error_status(error: Any) -> str:
    """Classify transport failures without depending on a non-empty message."""
    error_type = type(error).__name__ if isinstance(error, BaseException) else ""
    error_text = str(error or "")
    if (
        isinstance(error, TimeoutError)
        or "timeout" in error_type.casefold()
        or "timeout" in error_text.casefold()
        or "timed out" in error_text.casefold()
    ):
        return "timeout"
    return "provider_transport_error"


def _public_review_reason(value: Any) -> str:
    """Redact model-controlled reason text before publishing it in full.

    v6.70.0 honesty change (owner decision): reviewer rationale is a cognitive
    artifact (BIBLE P1 — multi-model review outputs must not fall back to
    generic transport truncation). The former 500/800-char caps destroyed the
    only owner-reachable copy of the reasoning (task_results carried the same
    truncated projection and the full observability blobs were unreferenced),
    so the projection now publishes the COMPLETE redacted text; secrets are
    still masked by redact_projection."""
    text = str(value or "")
    if not text:
        return ""
    return str(redact_projection(text).value)


def _review_actor_projection(actor: Any, surface: str) -> Dict[str, Any]:
    row = actor if isinstance(actor, dict) else asdict(actor)
    parsed = row.get("parsed") if isinstance(row.get("parsed"), (dict, list)) else None
    usage = row.get("usage") if isinstance(row.get("usage"), dict) else {}
    explicit_parse = str(row.get("parse_status") or "")
    semantic = str(row.get("semantic_verdict") or "").upper()
    if not semantic and isinstance(parsed, dict):
        semantic = str(parsed.get("verdict") or parsed.get("status") or "").upper()
    if not semantic:
        semantic = str(row.get("signal") or "").upper()
    valid = (
        explicit_parse != "malformed"
        and parsed is not None
        and semantic in {"PASS", "FAIL", "DEGRADED"}
    )
    error = str(row.get("error") or "")
    transport = str(row.get("transport_status") or "")
    if not transport:
        transport = (
            "success" if str(row.get("status") or "") in {"ok", "empty"}
            else _transport_error_status(error)
        )
    criteria = parsed.get("criteria_used") if isinstance(parsed, dict) else []
    criteria = criteria if isinstance(criteria, list) else []
    if isinstance(parsed, dict):
        parsed_findings = parsed.get("findings")
    elif isinstance(parsed, list):
        parsed_findings = parsed
    else:
        parsed_findings = []
    parsed_findings = (
        [item for item in parsed_findings if isinstance(item, dict)]
        if isinstance(parsed_findings, list)
        else []
    )
    reason = str(row.get("reason") or "")
    if not reason and isinstance(parsed, dict):
        reason = str(parsed.get("summary") or parsed.get("reason") or "")
    if not reason and isinstance(parsed, list):
        for item in parsed_findings:
            reason = str(
                item.get("summary")
                or item.get("reason")
                or item.get("evidence")
                or item.get("item")
                or item.get("recommendation")
                or ""
            )
            if reason:
                break
    reason = reason or error or ("Reviewer response was malformed or absent." if not valid else "")
    model = str(usage.get("resolved_model") or row.get("model") or "")
    provider = str(usage.get("provider") or row.get("provider") or "")
    if not provider:
        provider = provider_for_model(model) if model else "unknown"
    outcome_tier = (
        str(parsed.get("outcome_tier") or "").strip().lower()
        if isinstance(parsed, dict)
        else ""
    )
    if outcome_tier not in {
        OUTCOME_TIER_SOLVED, OUTCOME_TIER_BEST_EFFORT, OUTCOME_TIER_BLOCKED,
    }:
        outcome_tier = ""
    dialogue_vote = (
        str(parsed.get("dialogue_status") or "").strip().lower()
        if isinstance(parsed, dict)
        else ""
    )
    if dialogue_vote not in DIALOGUE_STATUS_VALUES:
        dialogue_vote = ""
    return {
        "slot_id": str(row.get("slot_id") or ""), "model": model, "provider": provider,
        "actor_role": str(row.get("actor_role") or f"{surface} reviewer"),
        "transport_status": transport,
        "parse_status": explicit_parse or ("valid" if valid else "malformed"),
        "semantic_verdict": semantic if valid else "",
        "outcome_tier": outcome_tier if valid else "",
        "dialogue_status": dialogue_vote if valid else "",
        "coverage": {
            "criteria_total": len(criteria),
            "findings": len(parsed_findings),
        },
        "quorum_contribution": bool(row.get("quorum_contribution")),
        "reason": _public_review_reason(reason),
        "enforcement_impact": str(row.get("enforcement_impact") or "abstains"),
        # Forensic pointer to the full raw reviewer response in the private
        # observability store (durable-copy reachability; never the raw text,
        # never absolute host paths — exported task records must not leak the
        # install layout). persist_call() nests the content hashes inside
        # redacted_projection_ref/manifest_ref; project them flat.
        "response_ref": _response_ref_projection(row.get("response_ref")),
    }


def _response_ref_projection(ref: Any) -> Dict[str, str]:
    if not isinstance(ref, dict):
        return {}
    out: Dict[str, str] = {}
    if ref.get("call_id"):
        out["call_id"] = str(ref["call_id"])
    projection_ref = ref.get("redacted_projection_ref")
    if isinstance(projection_ref, dict) and projection_ref.get("sha256"):
        out["sha256"] = str(projection_ref["sha256"])
    elif ref.get("sha256"):
        out["sha256"] = str(ref["sha256"])
    manifest_ref = ref.get("manifest_ref")
    if isinstance(manifest_ref, dict) and manifest_ref.get("sha256"):
        out["manifest_sha256"] = str(manifest_ref["sha256"])
    return out


def _review_enforcement_impact(run: Dict[str, Any]) -> str:
    if str(run.get("enforcement_impact") or ""):
        return str(run["enforcement_impact"])
    request = run.get("request") if isinstance(run.get("request"), dict) else {}
    hardness = str((request.get("policy") or {}).get("hardness") or "")
    signal = str(run.get("aggregate_signal") or "").upper()
    if str(run.get("authority") or "") == "agent_advisory" or hardness == HARDNESS_ADVISORY_VISIBLE:
        return "advisory"
    if signal == "PASS":
        return "allows_completion"
    return "blocks_completion" if signal == "FAIL" and hardness == HARDNESS_HARD_GATE else "degrades_completion"


def _review_panel_id(request: ReviewRequest, actors: List[ReviewActorRecord]) -> str:
    seed = {
        "surface": request.surface,
        "task_id": request.task_id,
        "actors": [
            [actor.slot_id, actor.model, actor.response_ref]
            for actor in actors
        ],
    }
    digest = hashlib.sha256(
        json.dumps(seed, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()
    return f"panel_{digest[:16]}"


def build_review_binding(
    *,
    candidate: str,
    evidence: Dict[str, Any],
    fence_token_or_state: Any,
) -> Dict[str, Any]:
    """Build the exact host-panel identity without introducing another ledger."""
    from ouroboros.review_evidence import task_acceptance_evidence_revision

    candidate_hash = hashlib.sha256(str(candidate or "").encode("utf-8")).hexdigest()
    evidence_revision = task_acceptance_evidence_revision(evidence)
    fence_value = (
        json.dumps(fence_token_or_state, sort_keys=True, separators=(",", ":"), default=str)
        if isinstance(fence_token_or_state, (dict, list, tuple))
        else str(fence_token_or_state or "direct_context")
    )
    fence_hash = hashlib.sha256(fence_value.encode("utf-8")).hexdigest()
    binding_payload = {
        "candidate_hash": candidate_hash,
        "evidence_revision": evidence_revision,
        "fence_hash": fence_hash,
    }
    binding_hash = hashlib.sha256(
        json.dumps(binding_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        **binding_payload,
        "binding_hash": binding_hash,
        "panel_id": f"panel_{binding_hash[:16]}",
    }


def compact_review_projection(review_runs: Any) -> Dict[str, Any]:
    """Project existing audit runs without copying raw prompts or responses."""
    panels: List[Dict[str, Any]] = []
    for index, raw_run in enumerate(review_runs or []):
        if not isinstance(raw_run, dict):
            continue
        request = raw_run.get("request") if isinstance(raw_run.get("request"), dict) else {}
        surface = str(request.get("surface") or "review")
        actors = [_review_actor_projection(actor, surface) for actor in (raw_run.get("actors") or []) if isinstance(actor, dict)]
        policy = request.get("policy") if isinstance(request.get("policy"), dict) else {}
        min_successful = max(1, int(policy.get("min_successful_slots") or 1))
        contributing = sum(1 for actor in actors if actor["quorum_contribution"])
        transport_statuses = [actor["transport_status"] for actor in actors]
        transport = (
            "success" if transport_statuses and all(s == "success" for s in transport_statuses)
            else ("partial" if "success" in transport_statuses else (
                "timeout" if transport_statuses and all(s == "timeout" for s in transport_statuses)
                else "provider_transport_error"
            ))
        )
        reasons = raw_run.get("degraded_reasons") if isinstance(raw_run.get("degraded_reasons"), list) else []
        panel: Dict[str, Any] = {
            "panel_id": str(raw_run.get("panel_id") or f"panel_{index + 1}"),
            "surface": surface,
            "authority": str(raw_run.get("authority") or "unspecified"),
            "aggregate_signal": str(raw_run.get("aggregate_signal") or "UNKNOWN").upper(),
            "transport_status": str(raw_run.get("transport_status") or transport),
            "parse_status": str(raw_run.get("parse_status") or (
                "valid" if actors and all(a["parse_status"] == "valid" for a in actors) else "malformed"
            )),
            "coverage": {
                "actors_configured": len(actors),
                "transport_success": sum(1 for actor in actors if actor["transport_status"] == "success"),
                "parse_valid": sum(1 for actor in actors if actor["parse_status"] == "valid"),
                "quorum_contributing": contributing,
            },
            "quorum": {"required": min_successful, "contributed": contributing, "configured": len(actors)},
            # v6.74.0 (A6): the fallback reason is the structured panel_reason
            # reducer — it names the real blocker (tier + finding / degraded
            # causes) instead of an opaque aggregate label. An explicitly
            # recorded reason still wins.
            "reason": _public_review_reason(
                str(raw_run.get("reason") or "; ".join(str(item) for item in reasons)
                    or panel_reason(raw_run)),
            ),
            "enforcement_impact": _review_enforcement_impact(raw_run),
            "actors": actors,
            "superseded": bool(raw_run.get("superseded_by_revision")),
        }
        if raw_run.get("single_reviewer_no_diversity"):
            panel["single_reviewer_no_diversity"] = True
        if isinstance(raw_run.get("dialogue"), dict):
            panel["dialogue"] = raw_run.get("dialogue")
        for key in (
            "candidate_hash", "evidence_revision", "fence_hash", "binding_hash",
        ):
            if raw_run.get(key) not in (None, ""):
                panel[key] = str(raw_run.get(key))
        panels.append(panel)
    return {"panels": panels}
