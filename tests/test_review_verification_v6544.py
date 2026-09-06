"""Focused tests for the v6.54.4 review+verification commit (Commit 2 of the sprint).

Covers: budget_profile in the task contract (2.1), the task_pacing SSOT
(snapshot/reserve/gates incl. the plan's adversarial time-schema cases), the
review dissent layer + obligations (2.2), criterion provenance + CANDIDATES
latch + web answer_type (2.3).
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from ouroboros import task_pacing
from ouroboros.contracts.task_contract import build_task_contract, normalize_budget_profile


def _deadline_ctx(remaining_sec: float, *, total_sec: float = 1000.0, profile=None):
    now = datetime.now(timezone.utc)
    created = now - timedelta(seconds=total_sec - remaining_sec)
    deadline = now + timedelta(seconds=remaining_sec)
    return SimpleNamespace(
        task_metadata={
            "created_at": created.isoformat(),
            "deadline_at": deadline.isoformat(),
        },
        task_contract={"budget_profile": normalize_budget_profile(profile or {})},
    )


# ---------------------------------------------------------------------------
# 2.1 — budget_profile contract


def test_normalize_budget_profile_defaults_are_conservative():
    out = normalize_budget_profile(None)
    assert out == {
        "improvement_policy": "fixed",
        "max_improvement_passes": None,
        "reserve_finalization_pct": None,
        "cost_hard_stop_pct": None,
    }
    out2 = normalize_budget_profile({
        "improvement_policy": "ADAPTIVE",
        "max_improvement_passes": "3",
        "reserve_finalization_pct": 150,
        "cost_hard_stop_pct": 150,
    })
    assert out2["improvement_policy"] == "adaptive"
    assert out2["max_improvement_passes"] == 3
    assert out2["reserve_finalization_pct"] == 100  # clamped
    assert out2["cost_hard_stop_pct"] == 100  # clamped
    # ABI 7.0 (Q10=A): retired spellings normalize AWAY - the alias policy
    # falls back to "fixed" and the stall knob is not part of the shape.
    gone = normalize_budget_profile(
        {"improvement_policy": "until_deadline", "stall_rounds_threshold": 12})
    assert gone["improvement_policy"] == "fixed"
    assert "stall_rounds_threshold" not in gone
    # 0 is a MEANINGFUL value (no in-task cost stop), preserved verbatim.
    assert normalize_budget_profile({"cost_hard_stop_pct": 0})["cost_hard_stop_pct"] == 0


def test_budget_profile_rides_the_contract_and_inherits():
    task = {
        "id": "t1", "type": "task", "description": "x",
        "budget_profile": {
            "improvement_policy": "adaptive",
            "max_improvement_passes": 2,
            "cost_hard_stop_pct": 0,
        },
    }
    contract = build_task_contract(task)
    assert contract["budget_profile"]["improvement_policy"] == "adaptive"
    # Child inheritance via the parent-contract spread (metadata.task_contract).
    child = build_task_contract({
        "id": "c1", "type": "task", "description": "y",
        "metadata": {"task_contract": {**contract, "objective": "y"}},
    })
    assert child["budget_profile"]["improvement_policy"] == "adaptive"
    assert child["budget_profile"]["max_improvement_passes"] == 2
    # The meaningful 0 (no in-task cost stop) survives inheritance verbatim.
    assert child["budget_profile"]["cost_hard_stop_pct"] == 0


# ---------------------------------------------------------------------------
# 2.1 — task_pacing snapshot and gates (plan §193-200 time schema)


def test_snapshot_without_deadline_disables_time_axis():
    ctx = SimpleNamespace(task_metadata={}, task_contract={})
    snap = task_pacing.build_budget_snapshot(ctx)
    assert snap.has_deadline is False
    assert snap.spendable_sec == float("inf")
    ok, reason = task_pacing.review_launch_allowed(snap)
    assert ok and reason == ""


def test_review_launch_blocked_inside_reserve(monkeypatch):
    monkeypatch.setenv("OUROBOROS_FINALIZATION_GRACE_SEC", "120")
    monkeypatch.setenv("OUROBOROS_ACCEPTANCE_REVIEW_EST_SEC", "90")
    ctx = _deadline_ctx(remaining_sec=100.0)  # inside the 120s grace reserve
    snap = task_pacing.build_budget_snapshot(ctx)
    assert snap.inside_reserve
    ok, reason = task_pacing.review_launch_allowed(snap)
    assert not ok and reason == "review_skipped_deadline_reserve"


def test_review_launch_allowed_above_reserve(monkeypatch):
    monkeypatch.setenv("OUROBOROS_FINALIZATION_GRACE_SEC", "120")
    monkeypatch.setenv("OUROBOROS_ACCEPTANCE_REVIEW_EST_SEC", "90")
    ctx = _deadline_ctx(remaining_sec=400.0)
    snap = task_pacing.build_budget_snapshot(ctx)
    ok, _ = task_pacing.review_launch_allowed(snap)
    assert ok


def test_improvement_passes_bounded_by_count_without_deadline():
    profile = normalize_budget_profile({})
    snap = task_pacing.build_budget_snapshot(SimpleNamespace(task_metadata={}, task_contract={}))
    ok, _ = task_pacing.improvement_pass_allowed(snap, 0, profile)
    assert ok  # default max passes = 1, first pass allowed
    ok2, reason2 = task_pacing.improvement_pass_allowed(snap, 1, profile)
    assert not ok2 and reason2 == "improvement_passes_exhausted"


def test_improvement_pass_blocked_when_time_exhausted_mid_cycle(monkeypatch):
    """Plan adversarial case (г): time runs out mid-cycle → finalize."""
    monkeypatch.setenv("OUROBOROS_FINALIZATION_GRACE_SEC", "120")
    monkeypatch.setenv("OUROBOROS_ACCEPTANCE_REVIEW_EST_SEC", "90")
    profile = normalize_budget_profile({"max_improvement_passes": 5})
    ctx = _deadline_ctx(remaining_sec=150.0)  # spendable=30 <= floor=200
    snap = task_pacing.build_budget_snapshot(ctx)
    ok, reason = task_pacing.improvement_pass_allowed(snap, 0, profile)
    assert not ok and reason == "improvement_window_inside_reserve"


def test_reserve_uses_profile_pct_when_larger_than_grace(monkeypatch):
    monkeypatch.setenv("OUROBOROS_FINALIZATION_GRACE_SEC", "60")
    profile = normalize_budget_profile({"reserve_finalization_pct": 20})
    ctx = _deadline_ctx(remaining_sec=500.0, total_sec=1000.0, profile={"reserve_finalization_pct": 20})
    snap = task_pacing.build_budget_snapshot(ctx, profile=profile)
    assert snap.reserve_sec == pytest.approx(200.0, rel=0.05)  # 20% of 1000 > 60s grace


def test_finalize_emit_window_is_grace_not_pct_reserve(monkeypatch):
    """Adversarial r1 #6/#15: the deadline-finalize/tool-clamp emit window is the
    plain GRACE, NOT the pct reserve — else a long task's tail is amputated (a 6h
    task would self-finalize ~54 min early on a 15% profile). The pct reserve stays
    a review-gate concept, living only on the snapshot."""
    monkeypatch.setenv("OUROBOROS_FINALIZATION_GRACE_SEC", "120")
    # A big-budget task on a 15% profile: snapshot reserve is large (pct), but the
    # emit window the finalize path uses must be the small grace (120s).
    ctx = _deadline_ctx(remaining_sec=3600.0, total_sec=21600.0,
                        profile={"reserve_finalization_pct": 15})
    snap = task_pacing.build_budget_snapshot(ctx)
    assert snap.reserve_sec == pytest.approx(3240.0, rel=0.05)  # 15% of 6h (gate reserve)
    assert task_pacing.effective_finalization_reserve_sec(ctx) == pytest.approx(120.0)  # emit window
    # No deadline -> still the grace window.
    assert task_pacing.effective_finalization_reserve_sec(
        SimpleNamespace(task_metadata={})) == pytest.approx(120.0)


# ---------------------------------------------------------------------------
# 2.1 — milestone content moved to task_pacing (behavior parity)


def test_time_budget_note_fires_tightest_crossed_milestone():
    ctx = _deadline_ctx(remaining_sec=50.0, total_sec=1000.0)  # 5% remaining
    note = task_pacing.build_time_budget_note(ctx)
    assert note is not None
    assert note.checkpoint["milestone"] == "10%"
    # v6.60.0: the flush clause fires, but its FINAL ANSWER phrase is protocol-gated —
    # this ctx declares no answer_protocol, so the deliverable-save wording stays
    # while the marker phrase is absent.
    assert "WRITE your best current deliverable now" in note.text
    assert "FINAL ANSWER:" not in note.text
    # All coarser milestones marked seen — nothing refires.
    assert task_pacing.build_time_budget_note(ctx) is None

    # With the protocol declared, the marker phrase rides the same 10% flush.
    proto_ctx = _deadline_ctx(remaining_sec=50.0, total_sec=1000.0)
    proto_ctx.task_contract["answer_protocol"] = "final_answer_line"
    proto_note = task_pacing.build_time_budget_note(proto_ctx)
    assert proto_note is not None and "FINAL ANSWER:" in proto_note.text


def test_intrinsic_pacing_note_without_deadline(monkeypatch):
    monkeypatch.setenv("OUROBOROS_PACING_INTERVAL_SEC", "600")
    now = datetime.now(timezone.utc)
    ctx = SimpleNamespace(
        task_metadata={"created_at": (now - timedelta(seconds=1300)).isoformat()},
        task_contract={},
    )
    note = task_pacing.build_time_budget_note(ctx, round_idx=7, accumulated_usage={"cost": 1.25})
    assert note is not None and note.checkpoint["checkpoint_kind"] == "intrinsic_pacing"
    assert "Rounds so far: 7" in note.text


def test_budget_snapshot_latches_fallback_created():
    """Fable-5 cumulative review F4: a metadata-poor task (deadline but no
    created_at/started_at) must LATCH its fallback anchor like the note path
    does, so consecutive snapshots keep a stable total instead of re-anchoring
    to "now" and silently degrading the pct reserve toward the grace floor."""
    now = datetime.now(timezone.utc)
    ctx = SimpleNamespace(
        task_metadata={"deadline_at": (now + timedelta(seconds=1000)).isoformat()},
        task_contract={},
    )
    snap1 = task_pacing.build_budget_snapshot(ctx)
    assert snap1.has_deadline
    assert getattr(ctx, "_time_budget_started_at", None) is not None
    snap2 = task_pacing.build_budget_snapshot(ctx)
    assert abs(snap2.total_sec - snap1.total_sec) < 1.0


# ---------------------------------------------------------------------------
# 2.2 — dissent layer


def _mk_result(actors, findings, aggregate):
    return SimpleNamespace(
        actors=actors, parsed_findings=findings, aggregate_signal=aggregate,
        degraded=False, degraded_reasons=[], request={},
    )


def test_dissent_findings_surface_concrete_minority():
    from ouroboros.review_substrate import build_improvement_capsule, dissent_findings

    actors = [
        {"slot_id": "slot_1", "signal": "PASS", "parsed": {"outcome_tier": "solved", "completion_coach": "ship"}},
        {"slot_id": "slot_2", "signal": "PASS", "parsed": {"outcome_tier": "solved", "completion_coach": "ship"}},
        {"slot_id": "slot_3", "signal": "FAIL", "parsed": {
            "outcome_tier": "best_effort",
            "findings": [{"severity": "critical", "item": "answer_mismatch",
                          "recommendation": "The gold answer includes bell pepper — re-check the list"}],
        }},
    ]
    result = _mk_result(actors, [], "PASS")
    dissent = dissent_findings(result)
    assert len(dissent) == 1
    assert "slot_3" in dissent[0] and "bell pepper" in dissent[0]
    capsule = build_improvement_capsule(result)
    assert "[DISSENT — slot_3" in capsule  # dissent alone makes the capsule actionable


def test_dissent_ignores_bare_contrary_verdicts_and_degraded():
    from ouroboros.review_substrate import dissent_findings

    actors = [
        {"slot_id": "slot_1", "signal": "PASS", "parsed": {"outcome_tier": "solved"}},
        {"slot_id": "slot_2", "signal": "FAIL", "parsed": {"findings": []}},  # no concrete rec
        {"slot_id": "slot_3", "signal": "DEGRADED", "parsed": {"completion_coach": "noise"}},
    ]
    assert dissent_findings(_mk_result(actors, [], "PASS")) == []


def test_dissent_admits_deliberate_degraded_with_concrete_recommendation():
    """The motivating GAIA 3cef3a44 class (codex cumulative-review finding): a
    minority reviewer that consciously returned verdict=DEGRADED (the prompt's
    "cannot judge → DEGRADED and explain" branch) WITH a concrete
    findings[].recommendation must surface as dissent."""
    from ouroboros.review_substrate import build_improvement_capsule, dissent_findings

    actors = [
        {"slot_id": "slot_1", "signal": "PASS", "parsed": {"outcome_tier": "solved", "completion_coach": "ship"}},
        {"slot_id": "slot_2", "signal": "PASS", "parsed": {"outcome_tier": "solved", "completion_coach": "ship"}},
        {"slot_id": "slot_3", "signal": "DEGRADED", "parsed": {
            "verdict": "DEGRADED",
            "findings": [{"severity": "major", "item": "unverified_answer",
                          "recommendation": "Evidence does not cover the gold list — re-open the source and re-check bell pepper"}],
        }},
    ]
    result = _mk_result(actors, [], "PASS")
    dissent = dissent_findings(result)
    assert len(dissent) == 1
    assert "slot_3 said DEGRADED" in dissent[0] and "bell pepper" in dissent[0]
    assert "[DISSENT — slot_3" in build_improvement_capsule(result)


def test_dissent_still_excludes_parse_fail_demoted_and_coach_only_degraded():
    """Only a DELIBERATE DEGRADED verdict with a concrete findings recommendation
    dissents: parse-fail placeholders (parsed=None), contract-demoted PASSes
    (parsed verdict stays PASS — they agreed with the aggregate), and coach-only
    DEGRADED all stay excluded."""
    from ouroboros.review_substrate import dissent_findings

    actors = [
        {"slot_id": "slot_1", "signal": "PASS", "parsed": {"outcome_tier": "solved", "completion_coach": "ship"}},
        {"slot_id": "slot_2", "signal": "DEGRADED", "parsed": None},  # parse-fail placeholder
        {"slot_id": "slot_3", "signal": "DEGRADED", "parsed": {
            # Contract-demoted PASS (missing_tier_or_coach): not a dissenting voice.
            "verdict": "PASS",
            "findings": [{"recommendation": "looks demoted, not dissenting"}],
        }},
        {"slot_id": "slot_4", "signal": "DEGRADED", "parsed": {
            # Deliberate DEGRADED but coach-only — too vague to redirect finalization.
            "verdict": "DEGRADED", "completion_coach": "maybe check something",
        }},
    ]
    assert dissent_findings(_mk_result(actors, [], "PASS")) == []


# ---------------------------------------------------------------------------
# 2.2 — obligations collection and disposition


def test_collect_acceptance_obligations_critical_contributing_only():
    from ouroboros.loop import _collect_acceptance_obligations, _open_acceptance_obligations

    actors = [
        {"slot_id": "slot_1", "signal": "FAIL", "parsed": {}},
        {"slot_id": "slot_2", "signal": "PASS", "parsed": {}},
    ]
    findings = [
        {"slot_id": "slot_1", "severity": "critical", "item": "broken_output",
         "recommendation": "Fix the CSV header row"},
        {"slot_id": "slot_1", "severity": "advisory", "item": "style", "recommendation": "nit"},
        {"slot_id": "slot_2", "severity": "critical", "item": "non_contrib",
         "recommendation": "should not land (PASS actor on FAIL aggregate)"},
        {"slot_id": "slot_1", "severity": "critical", "item": "no_rec", "recommendation": ""},
    ]
    llm_trace: dict = {}
    _collect_acceptance_obligations(llm_trace, _mk_result(actors, findings, "FAIL"))
    obs = llm_trace["acceptance_obligations"]
    assert len(obs) == 1 and obs[0]["item"] == "broken_output" and obs[0]["status"] == "open"
    assert len(_open_acceptance_obligations(llm_trace)) == 1
    # Idempotent re-collect (same finding does not duplicate).
    _collect_acceptance_obligations(llm_trace, _mk_result(actors, findings, "FAIL"))
    assert len(llm_trace["acceptance_obligations"]) == 1
    # Disposition closes it.
    obs[0]["disposition"] = "addressed"
    assert _open_acceptance_obligations(llm_trace) == []


def test_agent_disposed_obligation_lifecycle_stays_pending_until_host_settles():
    """Triad r3 + codex v6.71.1: an agent disposition (a rebuttal) is a CLAIM, not a
    settlement — the row stays PENDING until a host panel adjudicates it. A panel
    re-raise reopens the row outright; a clean PASS settles it (disposed_by_re_review)."""
    from ouroboros.loop import (
        _collect_acceptance_obligations,
        _dispose_obligations_on_clean_pass,
        _open_acceptance_obligations,
    )

    actors = [{"slot_id": "slot_1", "signal": "FAIL", "parsed": {}}]
    findings = [{"slot_id": "slot_1", "severity": "critical", "item": "broken_output",
                 "recommendation": "Fix the CSV header row"}]
    llm_trace: dict = {}
    _collect_acceptance_obligations(llm_trace, _mk_result(actors, findings, "FAIL"))
    ob = llm_trace["acceptance_obligations"][0]
    # The agent files a rejection rebuttal (the apply path in loop_tool_execution):
    # the row is a claim awaiting adjudication — it must STAY pending.
    ob["disposition"] = "rejected"
    ob["disposition_reason"] = "not applicable in my view"
    ob["status"] = "agent_disposed"
    assert len(_open_acceptance_obligations(llm_trace)) == 1
    # The next panel round re-raises the SAME finding: the rebuttal was rejected,
    # the row reopens outright (no hidden disposed state).
    _collect_acceptance_obligations(llm_trace, _mk_result(actors, findings, "FAIL"))
    assert len(llm_trace["acceptance_obligations"]) == 1
    reopened = llm_trace["acceptance_obligations"][0]
    assert reopened["status"] == "open" and reopened["disposition"] == ""
    # The agent re-files; the panel then ACCEPTS (clean PASS): host settlement.
    reopened["disposition"] = "rejected"
    reopened["disposition_reason"] = "not applicable in my view"
    reopened["status"] = "agent_disposed"
    pending = _open_acceptance_obligations(llm_trace)
    assert len(pending) == 1
    clean_actors = [{"slot_id": "slot_1", "signal": "PASS", "parsed": {
        "outcome_tier": "solved",
        "criteria_used": [{"criterion": "c", "status": "supported", "evidence_refs": ["r"]}],
    }}]
    assert _dispose_obligations_on_clean_pass(
        llm_trace, _mk_result(clean_actors, [], "PASS"), pending, False,
    ) is True
    settled = llm_trace["acceptance_obligations"][0]
    # The panel accepted the agent's REBUTTAL (did not re-raise): the agent's own
    # disposition/reason stay as provenance, the host settlement is distinct.
    assert settled["status"] == "disposed_rebuttal_accepted"
    assert settled["disposition"] == "rejected"
    assert settled["disposition_reason"] == "not applicable in my view"
    assert _open_acceptance_obligations(llm_trace) == []


def test_obligations_clause_formats_ids():
    from ouroboros.loop import _format_obligations_clause

    clause = _format_obligations_clause([
        {"id": "ob-12345678", "item": "broken_output", "recommendation": "Fix the CSV header"},
    ])
    assert "ob-12345678" in clause and "obligation_dispositions" in clause


# ---------------------------------------------------------------------------
# 2.3 — criterion provenance


def test_receipt_defaults_to_agent_defined(tmp_path, monkeypatch):
    from ouroboros.tools.verify import _verify_and_record

    ctx = SimpleNamespace(
        task_id="t-crit", drive_root=tmp_path, task_metadata={}, task_contract={},
        repo_dir=tmp_path / "repo",
    )
    monkeypatch.setattr("ouroboros.tools.verify._confine_cwd", lambda *a, **k: (str(tmp_path), ""), raising=False)
    _verify_and_record(
        ctx, contract_kind="artifact_observation",
        artifact_paths=[str(tmp_path / "missing.txt")],
    )
    receipts_path = tmp_path / "task_results" / "artifacts" / "t-crit" / "verification_receipts.jsonl"
    rows = [json.loads(line) for line in receipts_path.read_text().splitlines()]
    assert rows and rows[-1]["criterion_source"] == "agent_defined"


def test_latest_agent_defined_verification_helper(tmp_path):
    from ouroboros.outcomes import latest_agent_defined_verification
    from ouroboros.utils import append_jsonl

    receipts = tmp_path / "task_results" / "artifacts" / "t-x" / "verification_receipts.jsonl"
    receipts.parent.mkdir(parents=True)
    append_jsonl(receipts, {"status": "pass", "criterion_source": "task_stated"})
    assert latest_agent_defined_verification(tmp_path, "t-x") is None
    append_jsonl(receipts, {"status": "pass", "criterion_source": "agent_defined"})
    hit = latest_agent_defined_verification(tmp_path, "t-x")
    assert hit is not None and hit["criterion_source"] == "agent_defined"
    # A stated basis reconciles the nudge condition.
    append_jsonl(receipts, {"status": "pass", "criterion_source": "agent_defined", "criterion_basis": "task asks exactly this"})
    assert latest_agent_defined_verification(tmp_path, "t-x") is None


# ---------------------------------------------------------------------------
# 2.3 — CANDIDATES latch


def test_candidates_block_latched_with_final_answer():
    from ouroboros.loop import _latch_final_answer_marker

    llm_trace: dict = {"tool_calls": []}
    content = (
        "Research complete.\n\nCANDIDATES:\n- 42 — counting only published papers\n"
        "- 45 — including preprints\n\nFINAL ANSWER: 42"
    )
    _latch_final_answer_marker(llm_trace, content)
    assert llm_trace["best_valid_final_answer"] == "42"
    assert llm_trace["candidate_answers"] == [
        "42 — counting only published papers",
        "45 — including preprints",
    ]


def test_no_candidates_block_leaves_trace_unchanged():
    from ouroboros.loop import _latch_final_answer_marker

    llm_trace: dict = {"tool_calls": []}
    _latch_final_answer_marker(llm_trace, "FINAL ANSWER: 7")
    assert "candidate_answers" not in llm_trace
    assert llm_trace["best_valid_final_answer"] == "7"


def test_candidates_marker_only_ignores_inline_prose_mention():
    """Adversarial r2 #4: a mid-sentence 'CANDIDATES:' followed later by an
    ordinary markdown bullet list must NOT latch those bullets — the marker is
    line-anchored and the items must be adjacent to it."""
    from ouroboros.loop import _latch_final_answer_marker

    llm_trace: dict = {"tool_calls": []}
    content = (
        "I evaluated several CANDIDATES: see the comparison below.\n\n"
        "- source A is stale\n- source B contradicts it\n\nFINAL ANSWER: B"
    )
    _latch_final_answer_marker(llm_trace, content)
    assert llm_trace["best_valid_final_answer"] == "B"
    assert "candidate_answers" not in llm_trace


def test_candidates_marker_only_stops_at_first_non_item_line():
    """The block ends at the first non-'- ' line even when items follow later."""
    from ouroboros.loop import _latch_final_answer_marker

    llm_trace: dict = {"tool_calls": []}
    content = (
        "CANDIDATES:\n- 42 — published only\nprose interrupts here\n"
        "- 99 — not part of the block\n\nFINAL ANSWER: 42"
    )
    _latch_final_answer_marker(llm_trace, content)
    assert llm_trace["candidate_answers"] == ["42 — published only"]


# ---------------------------------------------------------------------------
# review round 1 — integration regressions


def _acceptance_harness(monkeypatch, tmp_path, review_result, *, enforcement="blocking",
                        deadline_remaining=None, passes_done=0, prior_trace=None,
                        budget_profile=None):
    import ouroboros.loop as loop_mod
    import ouroboros.review_substrate as rs

    monkeypatch.setattr(loop_mod, "get_task_review_mode", lambda: "required")
    monkeypatch.setattr(loop_mod, "get_review_enforcement", lambda: enforcement)
    monkeypatch.setattr(rs, "triad_delivery_slots", lambda **k: [object(), object(), object()])
    monkeypatch.setattr(rs, "run_review_request", lambda *a, **k: review_result)
    meta = {}
    if deadline_remaining is not None:
        now = datetime.now(timezone.utc)
        meta = {
            "created_at": (now - timedelta(seconds=1000 - deadline_remaining)).isoformat(),
            "deadline_at": (now + timedelta(seconds=deadline_remaining)).isoformat(),
        }
    contract_task = {
        "id": "t",
        "root_task_id": "t",
        "delegation_role": "root",
        "metadata": meta,
    }
    if budget_profile is not None:
        contract_task["budget_profile"] = budget_profile
    contract = build_task_contract(contract_task)
    from ouroboros.task_results import STATUS_RUNNING, write_task_result

    write_task_result(
        tmp_path,
        "t",
        STATUS_RUNNING,
        root_task_id="t",
        delegation_role="root",
        task_contract=contract,
        result="Task is running.",
    )
    ctx = SimpleNamespace(
        _task_acceptance_reviewed=False, is_direct_chat=False,
        drive_root=str(tmp_path), root_task_id="t", delegation_role="root",
        task_metadata={**meta, "root_task_id": "t", "budget_drive_root": str(tmp_path)},
        task_contract=contract,
    )
    if passes_done:
        ctx._task_acceptance_improvement_passes = passes_done
    trace = dict(prior_trace or {})
    trace.setdefault("tool_calls", [{"tool": "write_file", "args": {"path": "x.py"}}])
    messages = [{"role": "system", "content": ""}, {"role": "user", "content": "goal"}]
    tools = SimpleNamespace(_ctx=ctx)
    out = loop_mod._run_task_acceptance_review_once(
        tools=tools, content="done", task_id="t", task_type="task",
        llm_trace=trace, drive_root=None, messages=messages, emit_progress=lambda _m, *, incident=None: None,
    )
    return out, ctx, trace, messages


def _blocked_result():
    import ouroboros.review_substrate as rs

    return rs.ReviewRunResult(
        request={"surface": "task_acceptance"},
        actors=[{"signal": "FAIL", "slot_id": "s0",
                 "parsed": {"outcome_tier": "blocked_with_evidence", "completion_coach": "fix it"}}],
        parsed_findings=[{"slot_id": "s0", "severity": "critical", "item": "broken",
                          "recommendation": "fix the header"}],
        aggregate_signal="FAIL",
    )


def test_required_blocking_actionable_fail_vetoes_and_redrives(
    monkeypatch, tmp_path,
):
    import ouroboros.review_substrate as rs

    class _Panel:
        def chat(self, **kwargs):
            failed = str(kwargs.get("model") or "") == "contradictory"
            verdict = "FAIL" if failed else "PASS"
            return {"content": json.dumps({
                "verdict": verdict,
                "outcome_tier": "best_effort" if failed else "solved",
                "completion_coach": "run the independent verification" if failed else "",
                "criteria_used": [{
                    "criterion": "deliverable is verified",
                    "status": "supported",
                    "evidence_refs": ["verification_summary"],
                }],
                "findings": [],
                "summary": verdict,
            })}, {}

    panel = rs.run_review_request(
        rs.ReviewRequest(
            surface="task_acceptance",
            goal="g",
            policy={
                "min_successful_slots": 2,
                "classify_outcome_tier": True,
                "require_criterion_evidence": True,
            },
            task_id="root",
        ),
        slots=[
            rs.ReviewSlot(slot_id="s1", model="pass-1"),
            rs.ReviewSlot(slot_id="s2", model="pass-2"),
            rs.ReviewSlot(slot_id="s3", model="contradictory"),
        ],
        drive_root=tmp_path,
        llm=_Panel(),
    )
    assert panel.aggregate_signal == "FAIL"
    assert panel.actors[2]["signal"] == "FAIL"

    out, _ctx, trace, messages = _acceptance_harness(
        monkeypatch, tmp_path, panel, enforcement="blocking",
    )
    assert out is True
    assert len(messages) == 4
    assert "run the independent verification" in messages[-1]["content"]
    assert trace["acceptance_decision"]["status"] == "revision_requested"
    assert trace["review_runs"][0]["aggregate_signal"] == "FAIL"
    assert trace["acceptance_obligations"] == []


def test_required_blocking_bare_fail_abstains_without_false_veto(monkeypatch, tmp_path):
    import ouroboros.review_substrate as rs

    class _MinimalFail:
        def chat(self, **_kwargs):
            return {"content": json.dumps({"verdict": "FAIL", "findings": []})}, {}

    panel = rs.run_review_request(
        rs.ReviewRequest(
            surface="task_acceptance",
            goal="g",
            policy={
                "min_successful_slots": 1,
                "classify_outcome_tier": True,
                "require_criterion_evidence": True,
            },
            task_id="root",
        ),
        slots=[rs.ReviewSlot(slot_id="s1", model="minimal-fail")],
        drive_root=tmp_path,
        llm=_MinimalFail(),
    )
    assert panel.aggregate_signal == "DEGRADED"
    assert panel.actors[0]["parsed"] == {"verdict": "FAIL", "findings": []}
    assert panel.actors[0]["signal"] == "DEGRADED"
    assert rs.build_improvement_capsule(panel) == ""

    out, _ctx, trace, messages = _acceptance_harness(
        monkeypatch, tmp_path, panel, enforcement="blocking",
    )
    assert out is False
    assert len(messages) == 2  # no empty or fabricated revision capsule
    assert trace["acceptance_decision"]["status"] == "finalized_unaccepted"
    assert trace["acceptance_decision"]["reason"] == "review_degraded"
    assert trace["acceptance_obligations"] == []


def test_review_skipped_inside_reserve_finalizes_loudly(monkeypatch, tmp_path):
    from ouroboros.outcomes import derive_loop_outcome

    monkeypatch.setenv("OUROBOROS_FINALIZATION_GRACE_SEC", "120")
    out, ctx, trace, messages = _acceptance_harness(
        monkeypatch, tmp_path, _blocked_result(), deadline_remaining=60,
    )
    assert out is False and len(messages) == 2  # no review round, no injection
    assert trace["review_decision"]["skipped"] == "review_skipped_deadline_reserve"
    # v6.78.0: the pacing reason is the typed `reason`; the status is canonical.
    assert trace["acceptance_decision"]["status"] == "finalized_unaccepted"
    assert trace["acceptance_decision"]["reason"] == "review_skipped_deadline_reserve"
    assert "review_runs" not in trace  # the review never ran

    outcome = derive_loop_outcome("FINAL ANSWER: best available answer", {}, trace)
    axes = outcome["outcome_axes"]
    assert axes["execution"]["status"] == "degraded"
    assert axes["execution"]["reason_code"] == "review_skipped_deadline_reserve"
    assert axes["objective"] == {
        "status": "degraded",
        "source": "task_acceptance_deadline_reserve",
        "review_status": "skipped",
    }
    assert axes["review"]["status"] == "skipped"
    assert axes["review"]["run_count"] == 0
    assert "aggregate_signals" not in axes["review"]  # no synthetic reviewer verdict


def test_obligations_finalize_best_effort_when_passes_exhausted(monkeypatch, tmp_path):
    prior = {"acceptance_obligations": [
        {"id": "ob-1", "item": "broken", "recommendation": "fix", "status": "open", "disposition": ""},
    ]}
    out, ctx, trace, _messages = _acceptance_harness(
        monkeypatch, tmp_path, _blocked_result(), passes_done=1, prior_trace=prior,
        budget_profile={"max_improvement_passes": 1},
    )
    assert out is False  # gates exhausted -> finalize
    decision = trace["acceptance_decision"]
    assert decision["status"] == "finalized_unaccepted"
    assert decision["reason"] == "open_obligations"
    assert decision["open_obligations"]  # visible in the outcome


def test_advisory_enforcement_keeps_todays_behavior_plus_dissent(monkeypatch, tmp_path):
    """Plan adversarial case (б): advisory users — no obligations layer at all."""
    out, ctx, trace, messages = _acceptance_harness(
        monkeypatch, tmp_path, _blocked_result(), enforcement="advisory",
    )
    assert out is True  # capsule re-loop exactly as today
    assert "acceptance_obligations" not in trace
    assert "OPEN OBLIGATIONS" not in messages[-1]["content"]


def test_clean_re_review_disposes_open_obligations(monkeypatch, tmp_path):
    import ouroboros.review_substrate as rs

    solved = rs.ReviewRunResult(
        request={"surface": "task_acceptance"},
        actors=[{"signal": "PASS", "slot_id": "s0",
                 "parsed": {"outcome_tier": "solved", "completion_coach": "ship",
                            "criteria_used": [{"criterion": "deliverable is verified", "status": "supported",
                                               "evidence_refs": ["verification_summary"]}]}}],
        parsed_findings=[], aggregate_signal="PASS",
    )
    prior = {"acceptance_obligations": [
        {"id": "ob-1", "item": "broken", "recommendation": "fix", "status": "open", "disposition": ""},
    ]}
    out, ctx, trace, _messages = _acceptance_harness(
        monkeypatch, tmp_path, solved, prior_trace=prior,
    )
    assert out is False
    ob = trace["acceptance_obligations"][0]
    assert ob["status"] == "disposed_by_re_review" and ob["disposition"] == "addressed"
    assert trace["acceptance_decision"]["status"] == "accepted"


def test_clean_pass_with_dissent_bullet_still_disposes_obligations(monkeypatch, tmp_path):
    """Adversarial r1 #5: a lone advisory DISSENT bullet makes the capsule
    non-empty, but on a CLEAN PASS re-review it must NOT block obligation
    disposal nor mislabel the task best_effort — dissent stays advisory."""
    import ouroboros.review_substrate as rs

    # A clean PASS aggregate with a non-contributing minority FAIL carrying a
    # concrete recommendation -> dissent_findings yields one bullet -> capsule
    # is non-empty even though nothing is actionable for the majority.
    clean_with_dissent = rs.ReviewRunResult(
        request={"surface": "task_acceptance"},
        actors=[
            {"signal": "PASS", "slot_id": "s0", "parsed": {"outcome_tier": "solved", "completion_coach": "ship",
                                                           "criteria_used": [{"criterion": "verified", "status": "supported", "evidence_refs": ["verification_summary"]}]}},
            {"signal": "PASS", "slot_id": "s1", "parsed": {"outcome_tier": "solved", "completion_coach": "ship",
                                                           "criteria_used": [{"criterion": "verified", "status": "supported", "evidence_refs": ["verification_summary"]}]}},
            {"signal": "FAIL", "slot_id": "s2", "parsed": {
                "outcome_tier": "best_effort",
                "findings": [{"severity": "critical", "item": "doubt",
                              "recommendation": "double-check the edge case before shipping"}]}},
        ],
        parsed_findings=[], aggregate_signal="PASS",
    )
    assert rs.build_improvement_capsule(clean_with_dissent).strip()  # capsule IS non-empty (dissent)
    prior = {"acceptance_obligations": [
        {"id": "ob-1", "item": "broken", "recommendation": "fix", "status": "open", "disposition": ""},
    ]}
    out, ctx, trace, _messages = _acceptance_harness(
        monkeypatch, tmp_path, clean_with_dissent, passes_done=1, prior_trace=prior,
        budget_profile={"max_improvement_passes": 1},
    )
    assert out is False
    ob = trace["acceptance_obligations"][0]
    assert ob["status"] == "disposed_by_re_review" and ob["disposition"] == "addressed"
    assert trace["acceptance_decision"]["status"] == "accepted"
    assert trace["acceptance_decision"]["dissent_noted"] is True


def test_pass_best_effort_is_not_clean_and_does_not_close_obligations(monkeypatch, tmp_path):
    import ouroboros.review_substrate as rs

    partial = rs.ReviewRunResult(
        request={"surface": "task_acceptance"},
        actors=[{
            "signal": "PASS",
            "slot_id": "s0",
            "parsed": {
                "outcome_tier": "best_effort",
                "completion_coach": "finish the missing criterion",
            },
        }],
        parsed_findings=[],
        aggregate_signal="PASS",
    )
    prior = {"acceptance_obligations": [{
        "id": "ob-1",
        "item": "missing",
        "recommendation": "finish it",
        "status": "open",
        "disposition": "",
    }]}
    out, _ctx, trace, _messages = _acceptance_harness(
        monkeypatch,
        tmp_path,
        partial,
        passes_done=1,
        prior_trace=prior,
        budget_profile={"max_improvement_passes": 1},
    )
    assert out is False
    assert trace["acceptance_obligations"][0]["status"] == "open"
    assert trace["acceptance_decision"]["status"] == "finalized_unaccepted"
    assert trace["acceptance_decision"]["reason"] == "open_obligations"


def test_no_obligations_when_contributing_set_empty():
    """Adversarial r1 #8: a no-quorum review (empty contributing set) has no
    authoritative verdict — it must manufacture NO blocking obligations, even
    from a parse-degraded slot's critical finding."""
    from ouroboros.loop import _collect_acceptance_obligations, _open_acceptance_obligations

    # All actors DEGRADED (no PASS/FAIL) -> _contributing_actors == [].
    actors = [
        {"slot_id": "d0", "signal": "DEGRADED", "parsed": {}},
        {"slot_id": "d1", "signal": "DEGRADED", "parsed": {}},
    ]
    findings = [{"slot_id": "d0", "severity": "critical", "item": "phantom",
                 "recommendation": "would have blocked finalization"}]
    llm_trace: dict = {}
    _collect_acceptance_obligations(llm_trace, _mk_result(actors, findings, "DEGRADED"))
    assert _open_acceptance_obligations(llm_trace) == []


def test_degraded_re_review_keeps_obligations_open_without_false_pass(monkeypatch, tmp_path):
    """Review round 3: a DEGRADED/no-quorum empty-capsule re-review proves nothing —
    obligations stay open and the finalization is honestly labeled."""
    import ouroboros.review_substrate as rs

    degraded = rs.ReviewRunResult(
        request={"surface": "task_acceptance"},
        actors=[], parsed_findings=[], aggregate_signal="DEGRADED",
        degraded=True, degraded_reasons=["quorum failure"],
    )
    prior = {"acceptance_obligations": [
        {"id": "ob-1", "item": "broken", "recommendation": "fix", "status": "open", "disposition": ""},
    ]}
    out, ctx, trace, _messages = _acceptance_harness(
        monkeypatch, tmp_path, degraded, prior_trace=prior,
    )
    assert out is False
    ob = trace["acceptance_obligations"][0]
    assert ob["status"] == "open" and not ob["disposition"]
    assert trace["acceptance_decision"]["status"] == "finalized_unaccepted"
    assert trace["acceptance_decision"]["reason"] == "review_degraded"
    assert trace["acceptance_decision"]["open_obligations"] == ["ob-1"]


def test_semantic_degraded_actor_abstains_but_steers_revision(monkeypatch, tmp_path):
    """A deliberate semantic DEGRADED remains outside quorum/obligations, while
    its concrete recommendation supplies the required+blocking correction rail."""
    import ouroboros.review_substrate as rs

    degraded = rs.ReviewRunResult(
        request={"surface": "task_acceptance"},
        actors=[{
            "signal": "DEGRADED",
            "slot_id": "s0",
            "parsed": {
                "verdict": "DEGRADED",
                "outcome_tier": "best_effort",
                "completion_coach": "change the deliverable",
            },
        }],
        parsed_findings=[{
            "slot_id": "s0",
            "severity": "high",
            "item": "unconfirmed",
            "recommendation": "change the deliverable",
        }],
        aggregate_signal="DEGRADED",
        degraded=True,
        degraded_reasons=["quorum_not_met"],
    )
    out, _ctx, trace, messages = _acceptance_harness(
        monkeypatch, tmp_path, degraded,
    )
    assert out is True
    assert len(messages) == 4
    assert "change the deliverable" in messages[-1]["content"]
    assert trace["acceptance_decision"]["status"] == "revision_requested"
    assert trace["acceptance_obligations"] == []


def test_no_quorum_pass_cannot_steer_degraded_revision():
    """A below-quorum PASS is not a semantic DEGRADED reviewer and therefore
    cannot inject its tier, coach, or findings into the correction capsule."""
    import ouroboros.review_substrate as rs

    degraded = rs.ReviewRunResult(
        request={"surface": "task_acceptance"},
        actors=[{
            "signal": "PASS",
            "slot_id": "s0",
            "parsed": {
                "verdict": "PASS",
                "outcome_tier": "best_effort",
                "completion_coach": "change the deliverable",
            },
        }],
        parsed_findings=[{
            "slot_id": "s0",
            "severity": "high",
            "item": "unconfirmed",
            "recommendation": "change the deliverable",
        }],
        aggregate_signal="DEGRADED",
        degraded=True,
        degraded_reasons=["quorum_not_met"],
    )
    assert rs.build_improvement_capsule(degraded) == ""


def test_tool_capture_applies_obligation_dispositions():
    from ouroboros.loop_tool_execution import handle_tool_calls  # noqa: F401  (import sanity)
    # Direct unit of the capture block: simulate the parsed tool payload path.
    import json as _json


    llm_trace = {"acceptance_obligations": [
        {"id": "ob-1", "item": "broken", "recommendation": "fix", "status": "open", "disposition": ""},
        {"id": "ob-2", "item": "other", "recommendation": "fix2", "status": "open", "disposition": ""},
    ]}
    parsed = {
        "aggregate_signal": "PASS",
        "agent_decision": {
            "disposition": "partial", "rationale": "r", "source": "agent_task_acceptance_review_tool",
            "obligation_dispositions": [
                {"id": "ob-1", "disposition": "addressed", "reason": "fixed header"},
            ],
        },
    }
    exec_result = {
        "tool_call_id": "c1", "fn_name": "task_acceptance_review",
        "result": "<full_review>\n" + _json.dumps(parsed) + "\n</full_review>",
        "is_error": False, "tool_args": {},
    }
    # Reuse the real capture path via the private helper contract: emulate what
    # handle_tool_calls does for this fn (kept in sync with loop_tool_execution).
    raw = str(exec_result.get("result") or "")
    payload = raw.split("<full_review>", 1)[1].rsplit("</full_review>", 1)[0].strip()
    parsed2 = _json.loads(payload)
    agent_decision = parsed2.get("agent_decision")
    by_id = {str(e.get("id")): e for e in agent_decision["obligation_dispositions"]}
    for ob in llm_trace["acceptance_obligations"]:
        entry = by_id.get(str(ob.get("id")))
        if entry:
            ob["disposition"] = str(entry.get("disposition") or "")
            ob["status"] = "agent_disposed"
    assert llm_trace["acceptance_obligations"][0]["status"] == "agent_disposed"
    assert llm_trace["acceptance_obligations"][1]["status"] == "open"


def test_agent_tool_payload_carries_dissent_noted(monkeypatch, tmp_path):
    """Review round 2: DISSENT is recorded on the agent-called path too."""
    import ouroboros.review_substrate as rs
    from ouroboros.tools.review import _handle_task_acceptance_review

    result = rs.ReviewRunResult(
        request={"surface": "task_acceptance"},
        actors=[
            {"slot_id": "s1", "signal": "PASS", "parsed": {"outcome_tier": "solved", "completion_coach": "ship"}},
            {"slot_id": "s2", "signal": "PASS", "parsed": {"outcome_tier": "solved", "completion_coach": "ship"}},
            {"slot_id": "s3", "signal": "FAIL", "parsed": {"findings": [
                {"severity": "critical", "item": "gap", "recommendation": "check the units"}]}},
        ],
        parsed_findings=[], aggregate_signal="PASS",
    )
    monkeypatch.setattr(rs, "triad_delivery_slots", lambda **k: [object(), object(), object()])
    monkeypatch.setattr(rs, "run_review_request", lambda *a, **k: result)
    monkeypatch.setattr(
        "ouroboros.review_evidence.build_task_acceptance_evidence",
        lambda ctx, **k: {"claim": "x"},
    )
    ctx = SimpleNamespace(
        task_id="t", drive_root=tmp_path,
        task_metadata={"root_task_id": "root", "parent_task_id": "root"},
        task_contract={},
    )
    out = _handle_task_acceptance_review(ctx, claim="done", goal="g")
    assert '"dissent_noted": true' in out


def test_capture_preserves_dissent_with_agent_decision():
    """Review round 3: dissent_noted survives when the same payload also carries
    an agent_decision (the capture must merge, not overwrite)."""
    parsed = {
        "dissent_noted": True,
        "agent_decision": {"disposition": "partial", "rationale": "r",
                           "source": "agent_task_acceptance_review_tool"},
    }
    llm_trace: dict = {}
    # Mirror the capture logic order (dissent first, then agent_decision merge).
    if parsed.get("dissent_noted"):
        dec = llm_trace.get("acceptance_decision") or {}
        dec["dissent_noted"] = True
        llm_trace["acceptance_decision"] = dec
    agent_decision = parsed["agent_decision"]
    llm_trace["acceptance_decision"] = {
        "status": agent_decision["disposition"],
        "source": agent_decision["source"],
        "rationale": agent_decision["rationale"],
        "agent_disposition": agent_decision["disposition"],
        "agent_rationale": agent_decision["rationale"],
        **({"dissent_noted": True} if parsed.get("dissent_noted") else {}),
    }
    assert llm_trace["acceptance_decision"]["dissent_noted"] is True
    assert llm_trace["acceptance_decision"]["agent_disposition"] == "partial"


def test_dissent_emits_exactly_one_bullet_for_two_minorities():
    from ouroboros.review_substrate import dissent_findings

    actors = [
        {"slot_id": "s1", "signal": "PASS", "parsed": {"outcome_tier": "solved"}},
        {"slot_id": "s2", "signal": "PASS", "parsed": {"outcome_tier": "solved"}},
        {"slot_id": "s3", "signal": "PASS", "parsed": {"outcome_tier": "solved"}},
        {"slot_id": "s4", "signal": "FAIL", "parsed": {"findings": [
            {"severity": "critical", "item": "a", "recommendation": "first concrete dissent"}]}},
        {"slot_id": "s5", "signal": "FAIL", "parsed": {"findings": [
            {"severity": "critical", "item": "b", "recommendation": "second concrete dissent"}]}},
    ]
    out = dissent_findings(_mk_result(actors, [], "PASS"))
    assert len(out) == 1 and "s4" in out[0]


# ---------------------------------------------------------------------------
# 2.3 — web_search answer_type


def test_web_search_results_carry_answer_type(monkeypatch, tmp_path):
    import sys, types

    from ouroboros.tools import search as search_mod

    class _FakeStream:
        def __iter__(self):
            return iter([
                types.SimpleNamespace(type="response.output_text.delta", delta="fresh answer"),
                types.SimpleNamespace(
                    type="response.completed",
                    response=types.SimpleNamespace(output_text="fresh answer", output=[], usage=None),
                ),
            ])

    class _Responses:
        def create(self, **kwargs):
            if kwargs.get("stream"):
                return _FakeStream()
            return types.SimpleNamespace(output_text="fresh answer", output=[])

    class _Client:
        def __init__(self, api_key=None, base_url=None, **kwargs):
            self.responses = _Responses()

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=_Client))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    out = search_mod._web_search(SimpleNamespace(task_id="t", drive_root=tmp_path), query="q")
    data = json.loads(out)
    assert data.get("answer_type") == "summary"


def test_time_budget_flush_tree_sentence_workspace_gated():
    """v6.74.4 time-axis freeze: the 10% flush gains a commit-neutral tree
    sentence ONLY for workspace deliveries; non-workspace text stays
    byte-identical (cache/latch semantics untouched)."""
    base_ctx = _deadline_ctx(remaining_sec=50.0, total_sec=1000.0)
    base_note = task_pacing.build_time_budget_note(base_ctx)
    assert base_note is not None and "working tree ships as-is" not in base_note.text

    ws_ctx = _deadline_ctx(remaining_sec=50.0, total_sec=1000.0)
    ws_ctx.workspace_root = "/tmp/ws"  # fallback attribute path
    ws_note = task_pacing.build_time_budget_note(ws_ctx)
    assert ws_note is not None and "working tree ships as-is" in ws_note.text
    # Commit-neutral on purpose: acting self_worktree subagents cannot commit.
    assert "commit" not in ws_note.text.lower()
    # The canonical authority wins over the raw attribute when present.
    canon_ctx = _deadline_ctx(remaining_sec=50.0, total_sec=1000.0)
    canon_ctx.workspace_root = "/tmp/ws"
    canon_ctx.is_workspace_mode = lambda: False
    canon_note = task_pacing.build_time_budget_note(canon_ctx)
    assert canon_note is not None and "working tree ships as-is" not in canon_note.text


def test_time_budget_flush_non_workspace_text_fully_identical(monkeypatch):
    """codex finding 4 / commit triad r4: TRUE byte-equality of the WHOLE
    non-workspace note — clock frozen, both contexts built from identical
    deadline metadata, full-text comparison with no stripping."""
    frozen = datetime.now(timezone.utc)
    monkeypatch.setattr(task_pacing, "utc_now", lambda: frozen)
    created = (frozen - timedelta(seconds=950)).isoformat()
    deadline = (frozen + timedelta(seconds=50)).isoformat()

    def _ctx():
        return SimpleNamespace(
            task_metadata={"created_at": created, "deadline_at": deadline},
            task_contract={"budget_profile": normalize_budget_profile({})},
        )

    plain = task_pacing.build_time_budget_note(_ctx())
    denied_ctx = _ctx()
    denied_ctx.workspace_root = "/tmp/ws"
    denied_ctx.is_workspace_mode = lambda: False
    denied = task_pacing.build_time_budget_note(denied_ctx)
    assert plain is not None and denied is not None
    assert plain.text == denied.text


def test_cost_wrapup_tree_sentence_workspace_gated(monkeypatch):
    """v6.74.4 (commit triad r1 advisory, sol): the cost wrap-up note — a
    sibling forced-end channel — carries the same commit-neutral tree sentence
    for workspace deliveries only; non-workspace text stays byte-identical."""

    def _note(with_ws):
        ctx = SimpleNamespace(
            task_metadata={}, task_contract={},
            _cost_budget_milestones_seen={"50%", "25%", "10%"},
            _cost_wrapup_seen=False,
        )
        if with_ws:
            ctx.workspace_root = "/tmp/ws"
        return task_pacing.build_cost_budget_note(
            ctx, start_remaining_usd=None, cost_ceiling_usd=10.0, task_cost=8.5,
        )

    plain = _note(False)
    ws = _note(True)
    assert plain is not None and ws is not None
    assert "working tree ships as-is" not in plain.text
    assert "working tree ships as-is" in ws.text
    assert ws.text.replace(task_pacing._TREE_FLUSH_SENTENCE, "") == plain.text


def test_cost_late_first_milestone_carries_tree_sentence_for_workspace():
    """v6.74.4 (commit triad r2, sol advisory): a workspace task whose FIRST
    cost milestone fires already past the wrap-up fraction must still see the
    tree sentence (the suppressed wrap-up was its only carrier)."""
    ctx = SimpleNamespace(task_metadata={}, task_contract={},
                          workspace_root="/tmp/ws")
    note = task_pacing.build_cost_budget_note(
        ctx, start_remaining_usd=None, cost_ceiling_usd=10.0, task_cost=9.5)
    assert note is not None
    assert note.checkpoint["checkpoint_kind"] == "cost_budget_milestone"
    assert "working tree ships as-is" in note.text
    # Non-workspace late milestone stays byte-identical (no tree sentence).
    plain_ctx = SimpleNamespace(task_metadata={}, task_contract={})
    plain = task_pacing.build_cost_budget_note(
        plain_ctx, start_remaining_usd=None, cost_ceiling_usd=10.0, task_cost=9.5)
    assert plain is not None and "working tree ships as-is" not in plain.text
    assert note.text.replace(task_pacing._TREE_FLUSH_SENTENCE, "") == plain.text
    # An EARLY milestone (before the wrap-up fraction) is unchanged even for
    # workspace tasks — the wrap-up note will carry the sentence later.
    early_ctx = SimpleNamespace(task_metadata={}, task_contract={},
                                workspace_root="/tmp/ws")
    early = task_pacing.build_cost_budget_note(
        early_ctx, start_remaining_usd=None, cost_ceiling_usd=10.0, task_cost=6.0)
    assert early is not None and "working tree ships as-is" not in early.text
