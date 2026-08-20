"""What the review actor records claim, and to whom.

Split by theme out of ``tests/test_review_substrate_v2.py``. This module owns
the actor truth: transport, parse and semantics reported separately, the
bounded compact projection with redaction before truncation, mixed-panel
participation counts and the stable review binding.
"""

import json

from ouroboros.review_substrate import ReviewRequest, ReviewSlot, run_review_request


class _MixedReviewTruthLLM:
    def chat(self, **kwargs):
        model = str(kwargs.get("model") or "")
        if "timeout" in model:
            # Some timeout types stringify to an empty message.  The transport
            # truth must come from the exception type, not incidental wording.
            raise TimeoutError()
        if "malformed" in model:
            return {"content": "not json"}, {}
        return {
            "content": json.dumps({
                "verdict": "DEGRADED",
                "outcome_tier": "best_effort",
                "summary": "Evidence coverage is incomplete.",
                "findings": [],
            }),
        }, {
            "provider": "openrouter",
            "resolved_model": "google/gemini-3.5-flash",
        }


def test_review_actor_truth_separates_transport_parse_and_semantics(tmp_path):
    from ouroboros.review_substrate import compact_review_projection

    result = run_review_request(
        ReviewRequest(
            surface="task_acceptance", goal="g", subject="candidate",
            policy={"min_successful_slots": 2}, task_id="truth",
        ),
        slots=[
            ReviewSlot("timeout", "anthropic/timeout-model", role_hint="acceptance reviewer"),
            ReviewSlot("malformed", "openai/malformed-model", role_hint="acceptance reviewer"),
            ReviewSlot("degraded", "google/degraded-model", role_hint="acceptance reviewer"),
        ],
        drive_root=tmp_path,
        llm=_MixedReviewTruthLLM(),
    )
    actors = {actor["slot_id"]: actor for actor in result.actors}
    assert result.panel_id.startswith("panel_")
    assert len(result.panel_id) == len("panel_") + 16
    assert actors["timeout"]["transport_status"] == "timeout"
    assert actors["timeout"]["parse_status"] == "malformed"
    assert actors["timeout"]["semantic_verdict"] == ""
    assert actors["malformed"]["transport_status"] == "success"
    assert actors["malformed"]["parse_status"] == "malformed"
    assert actors["degraded"]["parse_status"] == "valid"
    assert actors["degraded"]["semantic_verdict"] == "DEGRADED"
    assert actors["degraded"]["parsed"]["outcome_tier"] == "best_effort"
    assert actors["degraded"]["provider"] == "openrouter"
    assert actors["degraded"]["model"] == "google/gemini-3.5-flash"
    assert actors["degraded"]["actor_role"] == "acceptance reviewer"

    run = dict(result.__dict__)
    run.update({
        "authority": "host_root",
        "candidate_hash": "c" * 64,
        "evidence_revision": "e" * 64,
        "fence_hash": "f" * 64,
        "enforcement_impact": "degrades_completion",
    })
    panel = compact_review_projection([run])["panels"][0]
    assert panel["panel_id"] == result.panel_id
    assert panel["transport_status"] == "partial"
    assert panel["parse_status"] == "malformed"
    assert panel["quorum"] == {"required": 2, "contributed": 0, "configured": 3}
    assert len(panel["actors"]) == 3
    assert next(
        actor for actor in panel["actors"] if actor["slot_id"] == "degraded"
    )["outcome_tier"] == "best_effort"
    assert all("raw_text" not in actor for actor in panel["actors"])


def test_compact_review_projection_redacts_public_reasons_before_truncation():
    from ouroboros.review_substrate import compact_review_projection

    secret = "sk-or-" + ("ReviewSecret123" * 4)
    benign_reason = "Evidence coverage is incomplete, but the consumer flow is clear."
    actor_prefix = benign_reason + ("x" * 400) + " credential="
    panel_prefix = "Panel retained its benign diagnostic context. " + ("y" * 735)
    run = {
        "request": {"surface": "task_acceptance", "policy": {"min_successful_slots": 1}},
        "aggregate_signal": "DEGRADED",
        "reason": panel_prefix + " " + secret,
        "actors": [
            {
                "slot_id": "benign",
                "model": "model-safe",
                "status": "ok",
                "signal": "PASS",
                "parsed": {"verdict": "PASS", "summary": benign_reason, "findings": []},
                "quorum_contribution": True,
            },
            {
                "slot_id": "secret-bearing",
                "model": "model-secret",
                "status": "ok",
                "signal": "DEGRADED",
                "parsed": {
                    "verdict": "DEGRADED",
                    "summary": actor_prefix + secret,
                    "findings": [],
                },
            },
        ],
    }

    panel = compact_review_projection([run])["panels"][0]
    actors = {actor["slot_id"]: actor for actor in panel["actors"]}
    rendered = json.dumps(panel, ensure_ascii=False)

    assert actors["benign"]["reason"] == benign_reason
    assert benign_reason in actors["secret-bearing"]["reason"]
    assert "Panel retained its benign diagnostic context." in panel["reason"]
    assert secret not in rendered
    assert secret[:20] not in rendered
    assert "***REDACTED***" in actors["secret-bearing"]["reason"]
    assert "***REDACTED***" in panel["reason"]


class _MixedPassPassFailLLM:
    def chat(self, **kwargs):
        if str(kwargs.get("model") or "").endswith("-2"):
            body = {
                "verdict": "FAIL",
                "outcome_tier": "blocked_with_evidence",
                "completion_coach": "Resolve the verified acceptance gap.",
                "findings": [{
                    "severity": "high",
                    "item": "acceptance_gap",
                    "evidence": "The required behavior is not demonstrated.",
                    "recommendation": "Add independent evidence for the missing behavior.",
                }],
                "summary": "The candidate is not ready.",
            }
        else:
            body = {
                "verdict": "PASS",
                "outcome_tier": "solved",
                "completion_coach": "Ship the candidate.",
                # Production panels always required criteria evidence (the knob was
                # constant-true and is deleted): a contributing solved PASS carries
                # supported criteria with refs.
                "criteria_used": [{
                    "criterion": "candidate is verified", "status": "supported",
                    "evidence_refs": ["verification_summary"],
                }],
                "findings": [],
                "summary": "The candidate is ready.",
            }
        return {"content": json.dumps(body)}, {}


def test_mixed_panel_counts_valid_participation_independently_of_veto(tmp_path):
    from ouroboros.review_substrate import (
        aggregate_outcome_tier,
        compact_review_projection,
        task_acceptance_is_clean,
    )

    result = run_review_request(
        ReviewRequest(
            surface="task_acceptance",
            goal="g",
            subject="candidate",
            policy={"classify_outcome_tier": True, "min_successful_slots": 2},
            task_id="mixed-panel",
        ),
        slots=[ReviewSlot(f"s{i}", f"m-{i}") for i in range(3)],
        drive_root=tmp_path,
        llm=_MixedPassPassFailLLM(),
    )

    assert result.aggregate_signal == "FAIL"
    assert aggregate_outcome_tier(result) == "blocked_with_evidence"
    assert task_acceptance_is_clean(result) is False
    actors = {actor["slot_id"]: actor for actor in result.actors}
    assert all(actor["quorum_contribution"] is True for actor in actors.values())
    assert actors["s0"]["enforcement_impact"] == "supports_pass"
    assert actors["s1"]["enforcement_impact"] == "supports_pass"
    assert actors["s2"]["enforcement_impact"] == "veto"

    run = dict(result.__dict__)
    run["authority"] = "host_root"
    panel = compact_review_projection([run])["panels"][0]
    assert panel["aggregate_signal"] == "FAIL"
    assert panel["quorum"] == {"required": 2, "contributed": 3, "configured": 3}
    assert panel["coverage"]["quorum_contributing"] == 3
    assert [actor["enforcement_impact"] for actor in panel["actors"]] == [
        "supports_pass",
        "supports_pass",
        "veto",
    ]


class _ArrayReviewTruthLLM:
    def chat(self, **_kwargs):
        return {
            "content": json.dumps([{
                "verdict": "FAIL",
                "item": "missing_visual_evidence",
                "evidence": "No inspected screenshot is attached.",
                "recommendation": "Inspect the captured consumer flow.",
            }]),
        }, {
            "provider": "openrouter",
            "resolved_model": "anthropic/claude-fable-5",
        }


def test_review_actor_truth_preserves_array_coverage_and_physical_route(tmp_path):
    result = run_review_request(
        ReviewRequest(
            surface="multi_model_review",
            goal="g",
            subject="candidate",
            policy={"min_successful_slots": 1},
            task_id="array-truth",
        ),
        slots=[ReviewSlot("array", "anthropic/array-model")],
        drive_root=tmp_path,
        llm=_ArrayReviewTruthLLM(),
    )

    actor = result.actors[0]
    assert actor["parse_status"] == "valid"
    assert actor["semantic_verdict"] == "FAIL"
    assert actor["coverage"]["findings"] == 1
    assert actor["reason"] == "No inspected screenshot is attached."
    assert actor["provider"] == "openrouter"
    assert actor["model"] == "anthropic/claude-fable-5"

def test_review_binding_is_stable_and_tracks_each_exact_input():
    from ouroboros.review_substrate import build_review_binding

    base = build_review_binding(
        candidate="answer", evidence={"claims": ["verified"]}, fence_token_or_state="fence-1",
    )
    assert base == build_review_binding(
        candidate="answer", evidence={"claims": ["verified"]}, fence_token_or_state="fence-1",
    )
    assert base["candidate_hash"] != build_review_binding(
        candidate="changed", evidence={"claims": ["verified"]}, fence_token_or_state="fence-1",
    )["candidate_hash"]
    assert base["evidence_revision"] != build_review_binding(
        candidate="answer", evidence={"claims": ["changed"]}, fence_token_or_state="fence-1",
    )["evidence_revision"]
    assert base["binding_hash"] != build_review_binding(
        candidate="answer", evidence={"claims": ["verified"]}, fence_token_or_state="fence-2",
    )["binding_hash"]
    assert len(base["fence_hash"]) == 64
    assert "fence_token_or_state" not in base
    assert "fence-1" not in json.dumps(base)
