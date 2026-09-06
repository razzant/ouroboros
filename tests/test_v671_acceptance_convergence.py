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
    monkeypatch.setattr(loop, "_compute_subagent_handoff", lambda *_a, **_k: None)
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


# ── AP1: the packet ceiling follows the review quorum's real windows ───────────

def test_acceptance_packet_budget_falls_back_to_the_floor_without_calibration():
    from ouroboros.review_evidence import _ACCEPT_TOTAL_BUDGET, acceptance_packet_budget_chars

    # No slots at all, and a slot list whose models calibrate to nothing, both
    # read the historical floor: an unknown route never gets a THINNER packet.
    assert acceptance_packet_budget_chars([]) == _ACCEPT_TOTAL_BUDGET
    assert acceptance_packet_budget_chars(None) == _ACCEPT_TOTAL_BUDGET
    blank = [ReviewSlot(slot_id="slot_1", model="", effort="high")]
    assert acceptance_packet_budget_chars(blank) == _ACCEPT_TOTAL_BUDGET


def test_acceptance_packet_budget_uses_the_quorum_window_and_dense_chars(monkeypatch):
    from ouroboros.review_evidence import (
        ACCEPTANCE_PROMPT_OVERHEAD_CHARS,
        _ACCEPT_DENSE_CHARS_PER_TOKEN,
        _ACCEPT_TOTAL_BUDGET,
        acceptance_packet_budget_chars,
    )
    from ouroboros.tools import review_synthesis

    caps = {"wide-1": 900_000, "wide-2": 800_000, "narrow": 120_000}
    monkeypatch.setattr(
        review_synthesis, "per_slot_input_token_limits",
        lambda models, **kwargs: {str(m): caps.get(str(m), 0) for m in models},
    )
    slots = [
        ReviewSlot(slot_id="slot_1", model="wide-1", effort="high"),
        ReviewSlot(slot_id="slot_2", model="wide-2", effort="high"),
        ReviewSlot(slot_id="slot_3", model="narrow", effort="high"),
    ]
    budget = acceptance_packet_budget_chars(slots)
    # quorum of 3 is 2, so the second-largest cap sizes the shared prompt; the
    # narrow slot drops out of the quorum instead of shrinking everyone's packet.
    expected = int(800_000 * _ACCEPT_DENSE_CHARS_PER_TOKEN) - ACCEPTANCE_PROMPT_OVERHEAD_CHARS
    assert budget == expected > _ACCEPT_TOTAL_BUDGET
    # A retrieving slot brings its own tools and is not sized against this pack.
    retrieving = _t.SimpleNamespace(model="wide-1", max_tokens=16_384, retrieves=True)
    assert acceptance_packet_budget_chars([retrieving]) == _ACCEPT_TOTAL_BUDGET


def test_predecessor_envelope_sheds_first_and_is_disclosed():
    dr = Path(tempfile.mkdtemp())
    ctx = _t.SimpleNamespace(
        task_contract={
            "requirements": "do X", "expected_output": "42",
            "predecessor_authority": {
                "source": {"kind": "task_result", "task_id": "prev-task"},
                "final_answer": "P" * 60_000,
            },
        },
        task_metadata={}, drive_root=str(dr), task_id="acc", repo_dir=str(dr),
    )
    trace = {"tool_calls": [
        {"tool": "run_command", "status": "ok", "result": f"call-{i}-" + ("X" * 25000)}
        for i in range(25)
    ]}
    ev = build_task_acceptance_evidence(ctx, llm_trace=trace, drive_root=dr, task_id="acc")
    stub = ev["task_contract"]["predecessor_authority"]
    assert stub["kind"] == "predecessor_authority_omitted_for_budget"
    assert stub["previous_task_id"] == "prev-task"
    assert stub["omitted_chars"] > 60_000
    rows = [o for o in ev["omissions_manifest"]
            if o.get("section") == "task_contract.predecessor_authority"]
    assert rows and rows[0]["reason"] == "evidence_budget"
    # ...and it sheds BEFORE the trajectory tail is re-capped for the same bytes.
    assert ev["omissions_manifest"].index(rows[0]) == 0


def test_repo_diff_previews_last_and_keeps_its_source_ref(monkeypatch):
    from ouroboros import review_evidence

    dr = Path(tempfile.mkdtemp())
    ref = {"kind": "artifact", "path": "task_results/artifacts/acc/repo_diff.txt"}
    monkeypatch.setattr(
        review_evidence, "collect_turn_diff",
        lambda *a, **k: "D" * 300_000,
    )
    monkeypatch.setattr(
        review_evidence, "_accept_owner_directives", lambda *a, **k: "",
    )
    ev = build_task_acceptance_evidence(
        _acc_ctx(dr), llm_trace={"tool_calls": []}, drive_root=dr, task_id="acc",
    )
    # Without a durable source ref the diff is NOT previewed — it stays exact and
    # the packet honestly overflows instead of losing bytes nobody can recover.
    assert len(ev["repo_diff"]) == 300_000
    assert "repo_diff" not in {o.get("section") for o in ev["omissions_manifest"]}

    ev2 = build_task_acceptance_evidence(
        _acc_ctx(dr), llm_trace={"tool_calls": []}, drive_root=dr, task_id="acc",
    )
    ev2["repo_diff_source_ref"] = ref
    from ouroboros.review_evidence import _accept_enforce_budget

    ev2["repo_diff"] = "D" * 300_000
    ev2["omissions_manifest"] = []
    ev2.pop("__immutable_core_overflow__", None)
    ev2.pop("__budget_note__", None)
    shed = _accept_enforce_budget(ev2)
    assert len(shed["repo_diff"]) <= 20_100
    assert shed["repo_diff_source_ref"] == ref
    row = [o for o in shed["omissions_manifest"] if o.get("section") == "repo_diff"]
    assert row and row[0]["source_ref"] == ref
    assert "__immutable_core_overflow__" not in shed
    assert "__unresolved_partial_artifacts__" not in shed


def test_core_overflow_reason_names_the_largest_sections():
    from ouroboros.review_evidence import _accept_enforce_budget
    from ouroboros.review_dispatch import task_acceptance_zero_physical_refusal

    ev = {
        "owner_requirements_and_decisions": "O" * 300_000,
        "task_contract": {"requirements": "R" * 50_000},
        "verification_summary": {"note": "V" * 10_000},
    }
    out = _accept_enforce_budget(ev)
    reason = out["__immutable_core_overflow__"]["reason"]
    assert reason.startswith("packet exceeds budget after every disclosed shed")
    assert "owner_requirements_and_decisions=" in reason
    assert "task_contract=" in reason
    refusal = task_acceptance_zero_physical_refusal(out)
    assert refusal["status"] == "degraded_core_overflow"
    assert "owner_requirements_and_decisions=" in refusal["summary"]


def test_budget_shed_partials_no_longer_veto_the_panel(tmp_path):
    from ouroboros.review_dispatch import task_acceptance_zero_physical_refusal

    shed_only = {"__unresolved_partial_artifacts__": [
        {"tool": "read_file", "status": "not_materialized_for_reviewer",
         "source_ref": {"kind": "artifact", "path": "p"}},
    ]}
    assert task_acceptance_zero_physical_refusal(shed_only) == {}
    genuine = {"__unresolved_partial_artifacts__": [
        {"tool": "read_file", "status": "not_materialized_for_reviewer", "source_ref": {}},
        {"tool": "artifact_manifest", "status": "source_unavailable", "source_ref": {}},
    ]}
    assert task_acceptance_zero_physical_refusal(genuine)["status"] == "degraded_partial_source"


def test_packet_budget_is_memoised_once_per_task(tmp_path, monkeypatch):
    """A ceiling that moved between the binding build and the staleness rebuild
    would flip the evidence revision, supersede the paid identity and terminalize
    the task on an identical-acceptance refusal. The context resolves it ONCE."""
    from dataclasses import replace

    from ouroboros import loop
    from ouroboros.review_evidence import task_acceptance_evidence_revision

    ctx = _t.SimpleNamespace(
        task_contract={"requirements": "do X"}, task_metadata={},
        drive_root=str(tmp_path), task_id="acc", repo_dir=str(tmp_path),
    )
    review_ctx = loop._TaskAcceptanceContext(
        tools=_t.SimpleNamespace(_ctx=ctx), content="done", task_id="acc",
        task_type="", llm_trace={"tool_calls": []}, drive_root=tmp_path,
        messages=[{}, {"content": "goal"}], emit_progress=lambda *_: None,
        mode="auto", subtree_statuses=[], budget_profile=None, passes_done=0,
        packet_budget_chars=1_000_000,
    )
    first = task_acceptance_evidence_revision(loop._build_host_acceptance_evidence(review_ctx))
    again = task_acceptance_evidence_revision(
        loop._build_host_acceptance_evidence(replace(review_ctx, evidence={})),
    )
    assert first == again


# ── AP5: an OPEN plan wave binds nothing, and says so ─────────────────────────

def _record_wave(root: Path, task_id: str, *, closed: bool, cycle: int = 1) -> None:
    from ouroboros.task_results import record_plan_review_wave

    record_plan_review_wave(root, task_id, {
        "schema_version": 2,
        "cycle_index": cycle,
        "request_fingerprint": f"{cycle:064x}",
        "spec": {"goal": "g", "acceptance_claims": [
            {"id": "claim_1", "claim": "the widget renders", "priority": "must"},
            {"id": "claim_2", "claim": "the counter increments", "priority": "should"},
        ]},
        "findings": [],
        "aggregate": "GREEN" if closed else "REVISE_PLAN",
        "closed": closed,
        "paid": True,
        "dispositions": [],
    })


def test_an_open_plan_wave_is_disclosed_as_binding_nothing():
    dr = Path(tempfile.mkdtemp())
    _record_wave(dr, "acc", closed=False)
    ev = build_task_acceptance_evidence(
        _acc_ctx(dr), llm_trace={"tool_calls": []}, drive_root=dr, task_id="acc",
    )
    assert ev["acceptance_claims_source"] == "none_open_plan_wave"
    exhibit = ev["plan_claims_exhibit"]
    assert exhibit["binding"] == "not bound: wave open"
    assert exhibit["cycle_index"] == 1
    assert len(exhibit["acceptance_claims"]) == 2
    assert "the widget renders" in json.dumps(exhibit)
    # Nothing was bound: the contract keeps no claims and no support refs.
    assert "acceptance_claims" not in ev["task_contract"]
    assert "acceptance_support_refs" not in ev
    assert ev["__provenance__"]["plan_claims_exhibit"] == "host_attested"


def test_a_closed_wave_still_binds_and_leaves_no_exhibit():
    dr = Path(tempfile.mkdtemp())
    _record_wave(dr, "acc", closed=True)
    ev = build_task_acceptance_evidence(
        _acc_ctx(dr), llm_trace={"tool_calls": []}, drive_root=dr, task_id="acc",
    )
    assert ev["acceptance_claims_source"] == "plan_review"
    assert "plan_claims_exhibit" not in ev
    claims = ev["task_contract"]["acceptance_claims"]
    assert [row["id"] for row in claims] == ["claim_1", "claim_2"]


def test_ingress_claims_win_over_an_open_wave_with_no_exhibit():
    dr = Path(tempfile.mkdtemp())
    _record_wave(dr, "acc", closed=False)
    ctx = _t.SimpleNamespace(
        task_contract={
            "requirements": "do X", "expected_output": "42",
            "acceptance_claims": [{"id": "ingress_1", "claim": "the ingress claim"}],
        },
        task_metadata={}, drive_root=str(dr), task_id="acc", repo_dir=str(dr),
    )
    ev = build_task_acceptance_evidence(
        ctx, llm_trace={"tool_calls": []}, drive_root=dr, task_id="acc",
    )
    assert ev["acceptance_claims_source"] == "ingress_contract"
    assert "plan_claims_exhibit" not in ev


def test_a_task_with_no_wave_at_all_stays_silent():
    dr = Path(tempfile.mkdtemp())
    ev = build_task_acceptance_evidence(
        _acc_ctx(dr), llm_trace={"tool_calls": []}, drive_root=dr, task_id="acc",
    )
    assert "acceptance_claims_source" not in ev
    assert "plan_claims_exhibit" not in ev


# ── AP6: a forced rail closes a dangling revision request ────────────────────

def _decision(status: str, reason: str = "") -> dict:
    trace: dict = {"acceptance_decision": {
        "status": status, "reason": reason,
        "agent_disposition": "deferred", "agent_rationale": "the host decides",
    }, "review_decision": {"run_count": 2}}
    return trace


def test_a_terminal_decision_is_never_overwritten():
    from ouroboros.loop_acceptance import terminalize_dangling_revision

    for status in ("accepted", "finalized_unaccepted"):
        trace = _decision(status, reason="clean_quorum")
        assert terminalize_dangling_revision(trace, rail="round_limit") is False
        assert trace["acceptance_decision"]["status"] == status
        assert trace["acceptance_decision"]["reason"] == "clean_quorum"
        assert "acceptance_bypassed_round_limit" not in json.dumps(trace)


def test_a_dangling_revision_terminalizes_on_every_forced_rail():
    from ouroboros.loop_acceptance import terminalize_dangling_revision

    for rail, prior in (
        ("round_limit", "delivery_binding_superseded"),
        ("budget_exhausted", "improvement_capsule"),
        ("children_unabsorbed", "improvement_capsule"),
    ):
        trace = _decision("revision_requested", reason=prior)
        assert terminalize_dangling_revision(trace, rail=rail) is True
        decision = trace["acceptance_decision"]
        assert decision["status"] == "finalized_unaccepted"
        assert decision["reason"] == "revision_unavailable_on_forced_rail"
        assert decision["source"] == "forced_finalization"
        # The rationale names the PRIOR reason: "the panel requested an
        # improvement pass" is false for the superseded-binding shape.
        assert prior in decision["rationale"]
        assert rail in decision["rationale"]
        # The agent's own stance survives, and no bypass reason is stamped over
        # a panel that really ran.
        assert decision["agent_disposition"] == "deferred"
        assert decision["agent_rationale"] == "the host decides"
        assert "acceptance_bypassed" not in json.dumps(decision)
        # The paid audit trail is untouched.
        assert trace["review_decision"]["run_count"] == 2


def test_the_terminal_pair_keeps_the_objective_best_effort():
    from ouroboros.outcomes import _ACCEPTANCE_BLOCKED_TERMINAL_REASONS

    assert "revision_unavailable_on_forced_rail" not in _ACCEPTANCE_BLOCKED_TERMINAL_REASONS
