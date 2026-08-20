"""v6.71.1 — acceptance-review convergence (rebuttal wiring, partial coherence,
evidence-parity). These pin the surfaces that let a required+blocking acceptance
review terminate by REVIEWER AGREEMENT (never a unilateral agent give-up) instead
of looping on the same answer. See the trace-audit finalization-loop class."""

from __future__ import annotations

import json
import tempfile
import types as _t
from pathlib import Path

from ouroboros.review_substrate import (
    _criteria_shape_valid,
    _criteria_have_supported_evidence,
    _render_prompt_parts,
    run_review_request,
    task_acceptance_is_clean,
    ReviewRequest,
    ReviewSlot,
    OUTCOME_TIER_SOLVED,
)
from ouroboros.review_evidence import build_task_acceptance_evidence, _ACCEPT_RESULT_CAP
from ouroboros.tool_capabilities import DEFAULT_TOOL_RESULT_LIMIT


# ── 1.2 partial coherence: an honest partial no longer becomes "malformed" ──────

def test_criteria_shape_valid_partial_at_non_solved_tier_is_valid():
    partial = [{"criterion": "c", "status": "partial"}]
    # honest partial contributes as a valid NON-clean vote at best_effort
    assert _criteria_shape_valid(partial, "best_effort") is True
    assert _criteria_shape_valid(partial, "blocked_with_evidence") is True
    # ...but NOT at solved (incoherent — solved still requires all supported)
    assert _criteria_shape_valid(partial, "solved") is False


def test_criteria_shape_valid_solved_still_requires_all_supported_with_refs():
    supported = [{"criterion": "c", "status": "supported", "evidence_refs": ["r"]}]
    assert _criteria_shape_valid(supported, "solved") is True
    # solved coherence is exactly the release-clean bar (unchanged)
    assert _criteria_have_supported_evidence(supported) is True
    # supported without refs is invalid at any tier
    assert _criteria_shape_valid(
        [{"criterion": "c", "status": "supported"}], OUTCOME_TIER_SOLVED
    ) is False


def test_criteria_shape_valid_rejects_bad_shape():
    assert _criteria_shape_valid([], "best_effort") is False
    assert _criteria_shape_valid([{"criterion": "", "status": "partial"}], "best_effort") is False
    assert _criteria_shape_valid([{"criterion": "c", "status": "weird"}], "best_effort") is False
    assert _criteria_shape_valid("notalist", "best_effort") is False


# ── 1.1 rebuttal wire: the reviewer sees the host obligation catalog + rule ──────

def _acc_ctx(dr: Path):
    return _t.SimpleNamespace(
        task_contract={"requirements": "do X", "expected_output": "42"},
        task_metadata={}, drive_root=str(dr), task_id="acc", repo_dir=str(dr),
    )


def test_acceptance_evidence_carries_host_obligation_catalog():
    dr = Path(tempfile.mkdtemp())
    trace = {
        "tool_calls": [],
        "acceptance_obligations": [
            {"id": "ob-abc123def456", "item": "cover edge case", "recommendation": "add a test for empty input",
             "status": "open", "disposition": "rejected", "disposition_reason": "empty input is rejected upstream"},
        ],
    }
    ev = build_task_acceptance_evidence(_acc_ctx(dr), llm_trace=trace, drive_root=dr, task_id="acc")
    cat = ev.get("acceptance_obligations")
    assert cat and cat[0]["id"] == "ob-abc123def456"
    # host facts only (id/item/recommendation/status); the disposition REASON is not
    # in the host-attested section (it rides agent_supplied — clean provenance, P3)
    assert cat[0]["item"] == "cover edge case"
    assert "add a test" in cat[0]["recommendation"]
    assert "disposition_reason" not in cat[0]
    assert ev["__provenance__"]["acceptance_obligations"] == "host_attested"


def test_acceptance_prompt_has_obligation_rebuttal_rule():
    req = ReviewRequest(surface="task_acceptance", goal="ship the verified result",
                        policy={"classify_outcome_tier": True})
    slot = ReviewSlot(slot_id="slot_1", model="m", effort="high")
    stable, _task_stable, _dyn = _render_prompt_parts(req, slot)
    assert "OBLIGATION REBUTTALS" in stable
    # close to the commit-gate rebuttal wording; no new softness
    assert "if the argument is genuinely valid, do not re-raise" in stable
    # rebuttal is NEVER itself evidence for a criterion (P3 preserved)
    assert "NEVER itself evidence for a criterion" in stable


# ── 1.0 evidence-parity: reviewer sees the actor's view, not a hidden 700 ────────

def test_acceptance_evidence_parity_reviewer_sees_full_actor_result():
    assert _ACCEPT_RESULT_CAP == DEFAULT_TOOL_RESULT_LIMIT  # SSOT, not a third hidden cap
    dr = Path(tempfile.mkdtemp())
    # the actor saw a big run_command result; the reviewer must see far more than 700
    trace = {"tool_calls": [{"tool": "run_command", "status": "ok", "result": "A" * 12000}]}
    ev = build_task_acceptance_evidence(_acc_ctx(dr), llm_trace=trace, drive_root=dr, task_id="acc")
    traj = ev["tool_trajectory"]
    assert traj and len(traj[0]["result"]) > 6000          # not the old 700/4000
    assert len(traj[0]["result"]) <= _ACCEPT_RESULT_CAP + 300


def test_acceptance_evidence_parity_respects_per_tool_limits():
    # A verification tool with an 80k actor window (run_command) keeps parity past
    # the default window — the decisive pytest output at char 20k stays visible.
    dr = Path(tempfile.mkdtemp())
    trace = {"tool_calls": [{"tool": "run_command", "status": "ok",
                             "result": ("B" * 30000) + "DECISIVE-EVIDENCE-MARKER"}]}
    ev = build_task_acceptance_evidence(_acc_ctx(dr), llm_trace=trace, drive_root=dr, task_id="acc")
    assert "DECISIVE-EVIDENCE-MARKER" in ev["tool_trajectory"][0]["result"]


def test_budget_ladder_recaps_heavy_trajectory_instead_of_core_overflow():
    # Adversarial-review finding (Phase 1 round 1): 20 retained heavy calls could
    # exceed _ACCEPT_TOTAL_BUDGET and force __immutable_core_overflow__ → every
    # reviewer slot short-circuits DEGRADED with zero model calls. The ladder must
    # re-cap trajectory results (disclosed) before declaring the packet unreviewable.
    import json as _json
    from ouroboros.review_evidence import _ACCEPT_TOTAL_BUDGET

    dr = Path(tempfile.mkdtemp())
    trace = {"tool_calls": [
        {"tool": "run_command", "status": "ok", "result": f"call-{i}-" + ("X" * 25000)}
        for i in range(25)
    ]}
    ev = build_task_acceptance_evidence(_acc_ctx(dr), llm_trace=trace, drive_root=dr, task_id="acc")
    assert "__immutable_core_overflow__" not in ev              # never unreviewable from trajectory alone
    blob = _json.dumps(ev, ensure_ascii=False, default=str)
    assert len(blob) <= _ACCEPT_TOTAL_BUDGET + 5000             # bounded (+ omission-note slack)
    sections = {o.get("section") for o in ev.get("omissions_manifest", [])}
    assert "tool_trajectory_results" in sections                # the shed is DISCLOSED (P1)


def test_budget_ladder_recaps_trajectory_before_destroying_rebuttal_channel():
    # Round-2 MAJOR: post-parity the trajectory ALONE routinely exceeds the budget,
    # so the re-cap must run with the other trajectory steps — otherwise a tiny
    # agent_supplied (the obligation_dispositions rebuttal channel) and artifact
    # previews are destroyed as pure collateral of routine trajectory weight.
    from ouroboros.review_evidence import _ACCEPT_TOTAL_BUDGET

    dr = Path(tempfile.mkdtemp())
    art_dir = dr / "task_results" / "artifacts" / "acc"
    art_dir.mkdir(parents=True)
    (art_dir / "report.txt").write_text("small artifact preview text", encoding="utf-8")
    trace = {"tool_calls": [
        {"tool": "run_command", "status": "ok", "result": f"call-{i}-" + ("X" * 25000)}
        for i in range(25)
    ]}
    agent_evidence = {"agent_decision": {"obligation_dispositions": [
        {"id": "ob-abc123def456", "disposition": "rejected", "reason": "handled upstream"},
    ]}}
    ev = build_task_acceptance_evidence(
        _acc_ctx(dr), llm_trace=trace, drive_root=dr, task_id="acc",
        agent_evidence=agent_evidence,
    )
    assert len(json.dumps(ev, ensure_ascii=False, default=str)) <= _ACCEPT_TOTAL_BUDGET + 5000
    # the rebuttal channel survives STRUCTURED (not a {"__truncated__": ...} blob)
    supplied = ev["agent_supplied"]
    assert "__truncated__" not in supplied
    assert supplied["agent_decision"]["obligation_dispositions"][0]["id"] == "ob-abc123def456"
    # artifact previews survive too — they are last resorts, not collateral
    previews = [a.get("preview") for a in ev.get("artifacts", []) if isinstance(a, dict)]
    assert previews and "(omitted for budget — manifest only)" not in previews


def test_actor_truncation_marker_survives_into_acceptance_packet():
    # Triad round-2: the trace stores the ACTOR's view — for an over-limit raw
    # result that is `cap chars + "... (truncated from N ...)"`. The packet's
    # per-tool cap must NOT chop that marker off (truncate_review_artifact's
    # anti-waste floor passes the ~47-over-cap value whole), or the reviewer
    # loses the original raw-size provenance the actor saw (evidence-parity/P1).
    from ouroboros.loop_tool_execution import _truncate_tool_result

    dr = Path(tempfile.mkdtemp())
    actor_view = _truncate_tool_result("Z" * 500_000, "run_command")
    assert "truncated from 500000" in actor_view          # the actor's own marker
    trace = {"tool_calls": [{"tool": "run_command", "status": "ok", "result": actor_view}]}
    ev = build_task_acceptance_evidence(_acc_ctx(dr), llm_trace=trace, drive_root=dr, task_id="acc")
    assert "truncated from 500000" in ev["tool_trajectory"][0]["result"]


def test_acceptance_obligation_catalog_caps_disposed_history_only():
    # Round-2 MINOR + triad r4 CRITICAL: OPEN obligations are active blocking state
    # (a clean PASS closes them), so they must ALL ship to the reviewer — only the
    # HISTORICAL disposed rows are count-capped, with a disclosed omission.
    from ouroboros.review_evidence import _ACCEPT_OBLIGATIONS_MAX

    dr = Path(tempfile.mkdtemp())
    obligations = [
        {"id": f"ob-open-{i}", "item": f"open item {i}", "recommendation": "fix it",
         "status": "open", "disposition": ""}
        for i in range(30)
    ] + [
        {"id": f"ob-disp-{i}", "item": f"disposed item {i}", "recommendation": "fixed",
         "status": "disposed_by_re_review", "disposition": "addressed"}
        for i in range(30)
    ]
    trace = {"tool_calls": [], "acceptance_obligations": obligations}
    ev = build_task_acceptance_evidence(_acc_ctx(dr), llm_trace=trace, drive_root=dr, task_id="acc")
    cat = ev["acceptance_obligations"]
    assert len(cat) == _ACCEPT_OBLIGATIONS_MAX == 40
    ids = {row["id"] for row in cat}
    assert all(f"ob-open-{i}" in ids for i in range(30))         # every open row retained
    assert sum(1 for i in ids if i.startswith("ob-disp-")) == 10  # most-recent disposed fill
    assert {"section": "acceptance_obligations", "omitted": 20, "reason": "count_cap_disposed_only"} \
        in ev["omissions_manifest"]


def test_open_obligations_are_never_hidden_even_past_the_cap():
    # Triad r4: >_ACCEPT_OBLIGATIONS_MAX OPEN rows must ALL reach the reviewer —
    # hiding an open row would let _dispose_obligations_on_clean_pass close
    # obligations the panel never adjudicated (P1/P3).
    from ouroboros.review_evidence import _ACCEPT_OBLIGATIONS_MAX

    dr = Path(tempfile.mkdtemp())
    n_open = _ACCEPT_OBLIGATIONS_MAX + 15
    obligations = [
        {"id": f"ob-open-{i}", "item": f"open item {i}", "recommendation": "fix it",
         "status": "open", "disposition": ""}
        for i in range(n_open)
    ] + [
        {"id": "ob-disp-0", "item": "old disposed", "recommendation": "done",
         "status": "disposed_by_re_review", "disposition": "addressed"},
    ]
    trace = {"tool_calls": [], "acceptance_obligations": obligations}
    ev = build_task_acceptance_evidence(_acc_ctx(dr), llm_trace=trace, drive_root=dr, task_id="acc")
    cat_ids = {row["id"] for row in ev["acceptance_obligations"]}
    assert all(f"ob-open-{i}" in cat_ids for i in range(n_open))  # no open row hidden
    assert "ob-disp-0" not in cat_ids                             # disposed history clipped
    assert {"section": "acceptance_obligations", "omitted": 1, "reason": "count_cap_disposed_only"} \
        in ev["omissions_manifest"]


def test_recap_near_boundary_antiwaste_refusals_cannot_fake_core_overflow():
    # Triad r4 finding 2 (verified FALSE POSITIVE, pinned): truncate_review_artifact
    # refuses cuts smaller than its ~65-char marker, but the -400/call haircut in the
    # share formula over-provisions for exactly that, so refused near-boundary results
    # are already inside the budget envelope: after the re-cap the packet ALWAYS fits
    # (share > floor) and __immutable_core_overflow__ can only mean genuine core
    # dominance. Mixed near/far-over-share trajectory proves it end to end.
    import json as _json
    from ouroboros.review_evidence import _ACCEPT_TOTAL_BUDGET

    dr = Path(tempfile.mkdtemp())
    # 10 huge results force the re-cap; 10 sit near the eventual share so their
    # individual cuts are refused by the anti-waste floor.
    trace = {"tool_calls": [
        {"tool": "run_command", "status": "ok", "result": f"big-{i}-" + ("X" * 40000)}
        for i in range(10)
    ] + [
        {"tool": "run_command", "status": "ok", "result": f"near-{i}-" + ("Y" * 11000)}
        for i in range(10)
    ]}
    ev = build_task_acceptance_evidence(_acc_ctx(dr), llm_trace=trace, drive_root=dr, task_id="acc")
    assert "__immutable_core_overflow__" not in ev
    assert len(_json.dumps(ev, ensure_ascii=False, default=str)) <= _ACCEPT_TOTAL_BUDGET


# ── 1.2 coordinator-level pin: the call-site (not just the helper) is guarded ────

class _HonestPartialLLM:
    def chat(self, **_kwargs):
        return {"content": json.dumps({
            "verdict": "PASS",
            "outcome_tier": "best_effort",
            "completion_coach": "add the missing edge-case test to reach solved",
            "criteria_used": [
                {"criterion": "core works", "status": "supported", "evidence_refs": ["verification_summary"]},
                {"criterion": "edge case covered", "status": "partial"},
            ],
            "findings": [],
            "summary": "honest partial",
        })}, {}


def test_honest_partial_pass_contributes_at_coordinator_level(tmp_path):
    # Round-2 MAJOR test gap: the review_substrate call-site (`_criteria_shape_valid`
    # instead of `_criteria_have_supported_evidence`) had no regression net — a
    # one-line revert kept all helper-level tests green while demoting every honest
    # partial PASS back to DEGRADED (the quorum-starvation class 1.2 removes).
    result = run_review_request(
        ReviewRequest(
            surface="task_acceptance",
            goal="g",
            policy={
                "min_successful_slots": 2,
                "classify_outcome_tier": True,
                "require_criterion_evidence": True,
            },
            task_id="root",
        ),
        slots=[ReviewSlot(slot_id=f"s{i}", model=f"m{i}") for i in range(3)],
        drive_root=tmp_path,
        llm=_HonestPartialLLM(),
    )
    assert result.aggregate_signal == "PASS"                     # contributes, not malformed
    assert task_acceptance_is_clean(result) is False             # ...but never release-clean


# ── 1.3 an acceptance improvement pass is an ordinary answer round ──────────────

def test_acceptance_revision_round_does_not_arm_delivery_control(tmp_path, monkeypatch):
    import queue as _q

    from tests.test_delivery_forced_finalization import _forced_test_context

    loop, registry, ctx, trace = _forced_test_context(tmp_path)
    from ouroboros import loop_delivery

    monkeypatch.setattr(loop_delivery, "_compute_subagent_handoff", lambda *_a, **_k: None)
    monkeypatch.setattr(loop, "_maybe_inject_finalization_nudges", lambda *_a, **_k: False)
    # the panel requested another improvement pass (capsule fed back → True)
    monkeypatch.setattr(loop, "_run_task_acceptance_review_once", lambda **_k: True)

    result = loop._no_tool_final_answer(
        "Revised answer draft.", ctx, trace, registry, _q.Queue(), set(), lambda _t: None,
    )
    assert result is None                                        # another model round
    # v6.71.1: the revision round is an ORDINARY substantive answer round — the
    # JSON-only delivery control must NOT be armed on the acceptance path (it
    # conflicted with OPEN OBLIGATIONS prose + the periodic self-check and froze
    # the model into identical no-tool resubmits).
    assert not bool(getattr(registry._ctx, "_delivery_control_required", False))
