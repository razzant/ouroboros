"""v6.74.0 — the acceptance review becomes a reviewer-authored terminating dialogue.

Mandatory adversarial coverage per DEVELOPMENT.md "Loop / State-Machine Changes":
malformed reviewer output, unknown/stale ``obligation_id`` on a re_raise, partial
panel failure, multi-slot dialogue-status disagreement (reducer precedence),
replay durability of obligation rows, false completion, and the
backward-compatible default when the new fields are absent. Plus the B1 cache
segmentation contract (two marked segments, slot label at the tail, breakpoint
cap asserted on the final payload).
"""
from __future__ import annotations

import json
from types import SimpleNamespace as NS

import pytest

import ouroboros.loop as loop_mod
from ouroboros import loop_acceptance_review
from ouroboros import task_pacing
from ouroboros.review_substrate import (
    DIALOGUE_CONTINUE,
    DIALOGUE_STABLE_DISAGREEMENT,
    DIALOGUE_UNREACHABLE,
    ReviewRequest,
    ReviewSlot,
    _render_prompt_parts,
    _request_messages,
    aggregate_dialogue_status,
    assert_cache_breakpoint_cap,
    build_improvement_capsule,
    panel_reason,
)


def _actor(slot_id, signal, parsed, *, parse_status="valid"):
    return {
        "slot_id": slot_id,
        "signal": signal,
        "parsed": parsed,
        "parse_status": parse_status,
    }


def _result(actors, aggregate="FAIL", findings=None, degraded_reasons=None):
    return NS(
        aggregate_signal=aggregate,
        degraded=(aggregate == "DEGRADED"),
        actors=actors,
        parsed_findings=findings or [],
        degraded_reasons=degraded_reasons or [],
        request={"policy": {"min_successful_slots": 2}},
    )


# ── A5: dialogue-status reducer ─────────────────────────────────────────────


def test_dialogue_default_is_continue_when_field_absent():
    # Backward-compatible fail-safe: no reviewer emits the field -> continue.
    res = _result([
        _actor("s1", "FAIL", {"verdict": "FAIL"}),
        _actor("s2", "FAIL", {"verdict": "FAIL"}),
    ])
    out = aggregate_dialogue_status(res, quorum=2)
    assert out["status"] == DIALOGUE_CONTINUE
    assert out["votes"] == {DIALOGUE_CONTINUE: ["s1", "s2"]}


def test_dialogue_malformed_vote_defaults_to_continue():
    res = _result([
        _actor("s1", "FAIL", {"verdict": "FAIL", "dialogue_status": "give_up_now"}),
        _actor("s2", "FAIL", {"verdict": "FAIL", "dialogue_status": 42}),
    ])
    assert aggregate_dialogue_status(res, quorum=2)["status"] == DIALOGUE_CONTINUE


def test_dialogue_quorum_of_terminal_votes_terminates():
    res = _result([
        _actor("s1", "FAIL", {"verdict": "FAIL", "dialogue_status": "unreachable_here"}),
        _actor("s2", "FAIL", {"verdict": "FAIL", "dialogue_status": "unreachable_here"}),
        _actor("s3", "FAIL", {"verdict": "FAIL", "dialogue_status": "stable_disagreement"}),
    ])
    out = aggregate_dialogue_status(res, quorum=2)
    assert out["status"] == DIALOGUE_UNREACHABLE
    assert set(out["votes"][DIALOGUE_UNREACHABLE]) == {"s1", "s2"}


def test_dialogue_contributing_continue_beats_terminal_votes():
    # Precedence: any continue vote from a QUORUM-CONTRIBUTING actor keeps the loop.
    res = _result([
        _actor("s1", "FAIL", {"verdict": "FAIL", "dialogue_status": "continue_actionable"}),
        _actor("s2", "FAIL", {"verdict": "FAIL", "dialogue_status": "unreachable_here"}),
        _actor("s3", "FAIL", {"verdict": "FAIL", "dialogue_status": "unreachable_here"}),
    ])
    assert aggregate_dialogue_status(res, quorum=2)["status"] == DIALOGUE_CONTINUE


def test_dialogue_degraded_slot_vote_counts_toward_terminal_quorum():
    # sol finding #3: a DEGRADED (non-contributing) slot's deliberate terminal
    # vote must count — the reducer runs over ALL contract-valid actors.
    res = _result(
        [
            _actor("s1", "DEGRADED", {"verdict": "DEGRADED", "dialogue_status": "unreachable_here"}),
            _actor("s2", "FAIL", {"verdict": "FAIL", "dialogue_status": "stable_disagreement"}),
            # transport-dead slot: no parsed object -> no vote at all
            _actor("s3", "", None, parse_status="malformed"),
        ],
        aggregate="FAIL",
    )
    out = aggregate_dialogue_status(res, quorum=2)
    assert out["status"] == DIALOGUE_STABLE_DISAGREEMENT or out["status"] == DIALOGUE_UNREACHABLE
    # tie 1:1 resolves to unreachable_here (>= comparison)
    assert out["status"] == DIALOGUE_UNREACHABLE


def test_dialogue_garbage_dict_without_verdict_cannot_vote_terminal():
    # ext review r2: a dict-shaped parsed object WITHOUT a recognizable verdict
    # is not a deliberate reviewer act — its terminal "vote" must not count.
    res = _result(
        [
            _actor("s1", "", {"dialogue_status": "unreachable_here"}, parse_status="malformed"),
            _actor("s2", "", {"garbage": True, "dialogue_status": "unreachable_here"}, parse_status="malformed"),
        ],
        aggregate="DEGRADED",
    )
    assert aggregate_dialogue_status(res, quorum=2)["status"] == DIALOGUE_CONTINUE


def test_dialogue_transport_dead_panel_stays_continue():
    # Pure transport failure (no parsed objects anywhere) must NOT terminate:
    # no contract-valid votes exist, so the fail-safe default keeps the loop.
    res = _result(
        [
            _actor("s1", "", None, parse_status="malformed"),
            _actor("s2", "", None, parse_status="malformed"),
        ],
        aggregate="DEGRADED",
    )
    assert aggregate_dialogue_status(res, quorum=2)["status"] == DIALOGUE_CONTINUE


# ── A3: reviewer-authored obligation identity ───────────────────────────────


def _finding(slot="s1", severity="critical", item="cover edge case",
             recommendation="add a test", **extra):
    return {
        "slot_id": slot, "severity": severity, "item": item,
        "recommendation": recommendation, **extra,
    }


def _fail_result_with(findings):
    actors = [
        _actor("s1", "FAIL", {"verdict": "FAIL", "outcome_tier": "best_effort",
                              "completion_coach": "fix it"}),
        _actor("s2", "FAIL", {"verdict": "FAIL", "outcome_tier": "best_effort",
                              "completion_coach": "fix it"}),
    ]
    return NS(
        aggregate_signal="FAIL", degraded=False, actors=actors,
        parsed_findings=findings, degraded_reasons=[],
        request={"policy": {"min_successful_slots": 2}},
    )


def test_re_raise_with_valid_id_reopens_row_and_preserves_rebuttal():
    trace = {"acceptance_obligations": [{
        "id": "ob-known1known1", "item": "cover edge case",
        "recommendation": "add a test", "status": "agent_disposed",
        "disposition": "rejected", "disposition_reason": "the case is unreachable",
    }]}
    result = _fail_result_with([
        _finding(item="REWORDED edge-case gap", recommendation="really add the test",
                 disposition_kind="re_raise", obligation_id="ob-known1known1",
                 evidence="the rebuttal ignores the reachable input class"),
    ])
    loop_mod._collect_acceptance_obligations(trace, result)
    rows = trace["acceptance_obligations"]
    assert len(rows) == 1  # NO new hash id was minted for the reworded re-raise
    row = rows[0]
    assert row["status"] == "open" and row["disposition"] == ""
    assert row["previous_disposition"] == "rejected"
    assert row["previous_reason"] == "the case is unreachable"
    assert row["reopened_count"] == 1
    assert "reachable input class" in row["reviewer_rebuttal_response"]


def test_re_raise_with_unknown_id_fails_closed_to_new_with_note():
    trace = {"acceptance_obligations": []}
    result = _fail_result_with([
        _finding(disposition_kind="re_raise", obligation_id="ob-doesnotexist"),
    ])
    loop_mod._collect_acceptance_obligations(trace, result)
    rows = trace["acceptance_obligations"]
    assert len(rows) == 1
    assert rows[0]["id"].startswith("ob-")
    assert rows[0]["notes"] == ["re_raise_unbound:ob-doesnotexist"]
    assert rows[0]["status"] == "open"


def test_re_raise_with_missing_id_fails_closed_to_new_with_note():
    trace = {"acceptance_obligations": []}
    result = _fail_result_with([_finding(disposition_kind="re_raise")])
    loop_mod._collect_acceptance_obligations(trace, result)
    assert trace["acceptance_obligations"][0]["notes"] == ["re_raise_unbound:missing_id"]


def test_byte_identical_re_raise_without_marker_still_reopens():
    trace = {"acceptance_obligations": []}
    result = _fail_result_with([_finding()])
    loop_mod._collect_acceptance_obligations(trace, result)
    row = trace["acceptance_obligations"][0]
    row["status"] = "agent_disposed"
    row["disposition"] = "rejected"
    row["disposition_reason"] = "argument"
    loop_mod._collect_acceptance_obligations(trace, result)
    assert len(trace["acceptance_obligations"]) == 1
    assert row["status"] == "open" and row["reopened_count"] == 1
    assert row["previous_disposition"] == "rejected"


def test_duplicate_finding_from_two_slots_in_one_pass_does_not_false_reopen():
    # fable review r1 #1: two slots of ONE panel raising the same finding must
    # create the row once with NO reopen — a first presentation must never show
    # the agent "[re-raised ×1]" or inflate reopened_count for the reviewer.
    trace = {"acceptance_obligations": []}
    result = _fail_result_with([_finding(slot="s1"), _finding(slot="s2")])
    loop_mod._collect_acceptance_obligations(trace, result)
    rows = trace["acceptance_obligations"]
    assert len(rows) == 1
    assert int(rows[0].get("reopened_count") or 0) == 0
    assert "previous_disposition" not in rows[0]
    # ...and a typed re_raise duplicated across slots reopens exactly ONCE.
    rows[0].update(status="agent_disposed", disposition="rejected", disposition_reason="arg")
    oid = rows[0]["id"]
    result2 = _fail_result_with([
        _finding(slot="s1", disposition_kind="re_raise", obligation_id=oid),
        _finding(slot="s2", disposition_kind="re_raise", obligation_id=oid),
    ])
    loop_mod._collect_acceptance_obligations(trace, result2)
    assert len(trace["acceptance_obligations"]) == 1
    assert rows[0]["reopened_count"] == 1


def test_reused_panel_application_leaves_obligation_rows_byte_identical(monkeypatch, tmp_path):
    # fable review r2 #1: a REUSED panel (unchanged binding) is the same
    # reviewer act applied again — it must not mutate reviewer-authored rows
    # (no reopened_count bump, no reviewer_rebuttal_response overwrite), or a
    # byte-identical resubmit would shift the evidence revision and buy a
    # fresh paid panel.
    import copy

    findings = [_finding(evidence="it fails")]
    actors = [
        _actor("s1", "FAIL", {"verdict": "FAIL", "outcome_tier": "best_effort",
                              "completion_coach": "fix"}),
        _actor("s2", "FAIL", {"verdict": "FAIL", "outcome_tier": "best_effort",
                              "completion_coach": "fix"}),
    ]
    result = NS(
        aggregate_signal="FAIL", degraded=False, actors=actors,
        parsed_findings=findings, degraded_reasons=[],
        request={"policy": {"min_successful_slots": 2}},
    )
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
    monkeypatch.setattr(loop_mod, "_end_task_acceptance_fence", lambda *_a, **_k: True)
    monkeypatch.setattr(loop_mod, "_mark_root_acceptance_checkpoint", lambda *a, **k: None)
    tool_ctx = NS(
        _task_acceptance_seen_bindings={}, _task_acceptance_improvement_passes=0,
        task_metadata={}, task_contract={}, drive_root=str(tmp_path),
    )
    trace = {"review_runs": [], "reasoning_notes": []}
    ctx = loop_acceptance_review._TaskAcceptanceContext(
        tools=NS(_ctx=tool_ctx), content="candidate", task_id="t-reuse",
        task_type="task", llm_trace=trace, drive_root=None, messages=[],
        emit_progress=lambda _m: None, mode="required", subtree_statuses=[],
        budget_profile={}, passes_done=0,
    )
    loop_acceptance_review._apply_task_acceptance_result(ctx, result)
    rows_after_first = copy.deepcopy(trace["acceptance_obligations"])
    assert int(rows_after_first[0].get("reopened_count") or 0) == 0
    # the reused application of the SAME panel must be a pure re-read
    loop_acceptance_review._apply_task_acceptance_result(ctx, result, record_run=False, reused=True)
    assert trace["acceptance_obligations"] == rows_after_first


def test_obligation_rows_survive_json_replay():
    # Replay durability: the reopened row round-trips through JSON (the shape
    # persisted into task_results / evidence) without losing the argument.
    trace = {"acceptance_obligations": []}
    result = _fail_result_with([_finding()])
    loop_mod._collect_acceptance_obligations(trace, result)
    trace["acceptance_obligations"][0].update(
        status="agent_disposed", disposition="rejected", disposition_reason="arg",
    )
    loop_mod._collect_acceptance_obligations(trace, result)
    replayed = json.loads(json.dumps(trace))
    row = replayed["acceptance_obligations"][0]
    assert row["previous_disposition"] == "rejected"
    assert row["reopened_count"] == 1


def test_evidence_catalog_carries_prior_argument():
    from ouroboros.review_evidence import _accept_obligation_row

    row = {
        "id": "ob-1", "item": "x", "recommendation": "y", "status": "open",
        "previous_disposition": "rejected", "previous_reason": "my argument",
        "reopened_count": 2,
    }
    out = _accept_obligation_row(row)
    assert out["previous_agent_disposition"] == "rejected"
    assert out["previous_agent_reason"] == "my argument"
    assert out["reopened_count"] == 2
    plain = _accept_obligation_row({"id": "ob-2", "item": "x", "recommendation": "y"})
    assert "previous_agent_disposition" not in plain and "reopened_count" not in plain


# ── A4: obligations clause ──────────────────────────────────────────────────


def test_obligations_clause_shows_overrule_and_single_channel():
    clause = loop_mod._format_obligations_clause([{
        "id": "ob-1", "item": "gap", "recommendation": "fix",
        "reopened_count": 1, "previous_disposition": "rejected",
        "reviewer_rebuttal_response": "the argument fails because X",
    }])
    assert "obligation_dispositions" in clause
    assert "address them directly" not in clause
    assert "rebuttal was overruled" in clause
    assert "the argument fails because X" in clause


# ── A1/A6: capsule header, rails, three moves, panel_reason ─────────────────


def test_capsule_leads_with_verdict_blocker_rails_and_three_moves():
    result = _fail_result_with([_finding()])
    capsule = build_improvement_capsule(
        result,
        rails_line="money: $1.00 spent; time: 10 min left; review passes: 1 done",
        open_obligations=[{"id": "ob-1"}, {"id": "ob-2"}],
    )
    assert "Review verdict: FAIL (tier: best_effort)" in capsule
    assert "Open blocking obligation(s) (2): ob-1, ob-2." in capsule
    assert "Remaining headroom — money: $1.00 spent" in capsule
    assert "FIX" in capsule and "REBUT" in capsule and "DECLARE UNREACHABLE" in capsule
    # the measured do-nothing tail is gone; anti-derailment guards stay verbatim
    assert "otherwise produce your normal final answer" not in capsule
    assert "Do not mention this review or the reviewer" in capsule
    assert "never emit an internal ledger" in capsule


def test_capsule_without_new_args_is_backward_compatible():
    result = _fail_result_with([_finding()])
    capsule = build_improvement_capsule(result)
    assert capsule and "Review verdict: FAIL" in capsule
    assert "Open blocking obligation(s)" not in capsule
    assert "Remaining headroom" not in capsule


def test_panel_reason_names_real_blocker():
    fail = _fail_result_with([_finding(item="missing UI verification")])
    assert "missing UI verification" in panel_reason(fail)
    degraded = _result([], aggregate="DEGRADED", degraded_reasons=["slot_1:timeout"])
    assert "slot_1:timeout" in panel_reason(degraded)
    # dict-shaped run records (compact projection path) work too
    assert "slot_1:timeout" in panel_reason({
        "aggregate_signal": "DEGRADED", "actors": [], "parsed_findings": [],
        "degraded_reasons": ["slot_1:timeout"], "request": {},
    })


def test_single_reviewer_note_is_label_not_reason(tmp_path):
    # A6: the diversity note must not displace the real blocker in the reason.
    from ouroboros.review_substrate import compact_review_projection

    run = {
        "aggregate_signal": "FAIL", "actors": [], "parsed_findings": [],
        "degraded_reasons": [], "request": {"surface": "task_acceptance", "policy": {}},
        "single_reviewer_no_diversity": True, "authority": "host_root",
    }
    panel = compact_review_projection([run])["panels"][0]
    assert panel["single_reviewer_no_diversity"] is True
    assert "single_reviewer_no_diversity" not in panel["reason"]


# ── A5 wiring: dialogue-terminal finalization is honest (no false completion) ─


def _apply_harness(monkeypatch, result, *, obligations=None, tmp_path=None):
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
    fences: list = []
    monkeypatch.setattr(loop_mod, "_end_task_acceptance_fence", lambda ctx, outcome: fences.append(outcome) or True)
    monkeypatch.setattr(loop_mod, "_mark_root_acceptance_checkpoint", lambda *a, **k: None)
    tool_ctx = NS(
        _task_acceptance_seen_bindings={},
        _task_acceptance_improvement_passes=0,
        task_metadata={},
        task_contract={},
        drive_root=str(tmp_path or "/tmp"),
    )
    trace = {"review_runs": [], "reasoning_notes": []}
    if obligations is not None:
        trace["acceptance_obligations"] = obligations
    ctx = loop_acceptance_review._TaskAcceptanceContext(
        tools=NS(_ctx=tool_ctx),
        content="candidate",
        task_id="t-dialogue",
        task_type="task",
        llm_trace=trace,
        drive_root=None,
        messages=[],
        emit_progress=lambda _m: None,
        mode="required",
        subtree_statuses=[],
        budget_profile={},
        passes_done=0,
        rails_line="rails",
    )
    another = loop_acceptance_review._apply_task_acceptance_result(ctx, result)
    return another, trace, tool_ctx, fences


def test_dialogue_terminal_finalizes_honestly_with_both_positions(monkeypatch, tmp_path):
    findings = [_finding()]
    actors = [
        _actor("s1", "FAIL", {"verdict": "FAIL", "outcome_tier": "best_effort",
                              "completion_coach": "fix", "dialogue_status": "stable_disagreement"}),
        _actor("s2", "FAIL", {"verdict": "FAIL", "outcome_tier": "best_effort",
                              "completion_coach": "fix", "dialogue_status": "stable_disagreement"}),
    ]
    result = NS(
        aggregate_signal="FAIL", degraded=False, actors=actors,
        parsed_findings=findings, degraded_reasons=[],
        request={"policy": {"min_successful_slots": 2}},
    )
    another, trace, tool_ctx, fences = _apply_harness(monkeypatch, result, tmp_path=tmp_path)
    assert another is False  # NOT re-driven: the reviewers ended the dialogue
    decision = trace["acceptance_decision"]
    # v6.78.0: one canonical terminal status; the with/without-obligations
    # distinction lives on the `open_obligations` list asserted below.
    assert decision["status"] == "finalized_unaccepted"  # never a clean accept
    assert decision["reason"] == "dialogue_terminal"
    assert decision["dialogue_status"] == DIALOGUE_STABLE_DISAGREEMENT
    assert decision["dialogue_votes"][DIALOGUE_STABLE_DISAGREEMENT] == ["s1", "s2"]
    assert decision["open_obligations"]  # the obligations stay recorded, not wiped
    assert tool_ctx._task_acceptance_reviewed is True
    assert fences == ["terminal"]
    # both positions persisted: reviewer votes on the run record
    host_runs = [r for r in trace["review_runs"] if r.get("authority") == "host_root"]
    assert host_runs and host_runs[-1]["dialogue"]["status"] == DIALOGUE_STABLE_DISAGREEMENT


def test_dialogue_continue_still_re_drives(monkeypatch, tmp_path):
    findings = [_finding()]
    actors = [
        _actor("s1", "FAIL", {"verdict": "FAIL", "outcome_tier": "best_effort",
                              "completion_coach": "fix", "dialogue_status": "continue_actionable"}),
        _actor("s2", "FAIL", {"verdict": "FAIL", "outcome_tier": "best_effort",
                              "completion_coach": "fix", "dialogue_status": "unreachable_here"}),
    ]
    result = NS(
        aggregate_signal="FAIL", degraded=False, actors=actors,
        parsed_findings=findings, degraded_reasons=[],
        request={"policy": {"min_successful_slots": 2}},
    )
    another, trace, _tool_ctx, fences = _apply_harness(monkeypatch, result, tmp_path=tmp_path)
    assert another is True  # a contributing continue keeps the loop
    assert fences == ["revision"]
    assert trace["acceptance_decision"]["status"] == "revision_requested"
    # the capsule content itself (verdict header, rails, three moves) is pinned
    # by the dedicated capsule tests above; here the loop must merely re-drive.


def test_absent_dialogue_fields_keep_legacy_re_drive(monkeypatch, tmp_path):
    # Backward compatibility: nothing emitted -> behavior identical to v6.71.x.
    findings = [_finding()]
    actors = [
        _actor("s1", "FAIL", {"verdict": "FAIL", "outcome_tier": "best_effort",
                              "completion_coach": "fix"}),
        _actor("s2", "FAIL", {"verdict": "FAIL", "outcome_tier": "best_effort",
                              "completion_coach": "fix"}),
    ]
    result = NS(
        aggregate_signal="FAIL", degraded=False, actors=actors,
        parsed_findings=findings, degraded_reasons=[],
        request={"policy": {"min_successful_slots": 2}},
    )
    another, _trace, _tool_ctx, fences = _apply_harness(monkeypatch, result, tmp_path=tmp_path)
    assert another is True
    assert fences == ["revision"]


# ── B1: prompt segmentation ─────────────────────────────────────────────────


def _acc_request(evidence=None):
    return ReviewRequest(
        surface="task_acceptance",
        goal="ship the verified result",
        scope="the task",
        subject="candidate answer",
        checklist="the checklist",
        evidence=evidence or {},
        policy={"classify_outcome_tier": True, "min_successful_slots": 2},
        task_id="cache-task",
    )


def test_prompt_parts_split_governance_task_stable_dynamic():
    stable, task_stable, dynamic = _render_prompt_parts(
        _acc_request({"k": "v1"}), ReviewSlot("slot_1", "m"),
    )
    assert "independent Ouroboros reviewer slot" in stable
    assert "dialogue_status" in stable and "disposition_kind" in stable
    assert "REACHABILITY" in stable and "REVIEW REGISTER" in stable
    assert "Review goal:" in task_stable and "Checklist" in task_stable
    assert "Subject:" in dynamic and "Evidence packet:" in dynamic
    # slot label at the TAIL of the mutable part, never at byte 0
    assert dynamic.rstrip().endswith("Slot: slot_1")
    assert not dynamic.startswith("Slot:")


def test_task_stable_segment_is_byte_identical_across_passes():
    slot = ReviewSlot("slot_1", "m")
    s1, t1, d1 = _render_prompt_parts(_acc_request({"k": "pass-1 evidence"}), slot)
    s2, t2, d2 = _render_prompt_parts(_acc_request({"k": "pass-2 evidence"}), slot)
    assert s1 == s2 and t1 == t2  # cache-marked segments stable across passes
    assert d1 != d2               # the evidence tail honestly changes


def test_same_pass_slots_share_whole_prefix():
    req = _acc_request({"k": "v"})
    s1, t1, d1 = _render_prompt_parts(req, ReviewSlot("slot_1", "m"))
    s2, t2, d2 = _render_prompt_parts(req, ReviewSlot("slot_2", "m"))
    assert (s1, t1) == (s2, t2)
    # the dynamic parts differ ONLY in the trailing slot label
    assert d1[: d1.rfind("Slot:")] == d2[: d2.rfind("Slot:")]


def test_request_messages_mark_two_segments_and_pass_cap():
    messages = _request_messages(_acc_request(), ReviewSlot("slot_1", "m"))
    system = messages[0]["content"]
    marked = [b for b in system if isinstance(b, dict) and b.get("cache_control")]
    assert len(marked) == 2
    assert isinstance(messages[1]["content"], str)
    assert_cache_breakpoint_cap(messages)  # 2 <= 4


def test_breakpoint_cap_rejects_over_marked_payload():
    block = {"type": "text", "text": "x", "cache_control": {"type": "ephemeral"}}
    messages = [{"role": "system", "content": [dict(block) for _ in range(5)]}]
    with pytest.raises(AssertionError):
        assert_cache_breakpoint_cap(messages)


# ── C5: cost-finality wait helpers (triad r1 tests_affected advisory) ────────


def test_cli_cost_finality_wait_branches(monkeypatch):
    from ouroboros import cli as cli_mod

    calls = []

    class _Client:
        def request(self, method, path, body=None, timeout=None):
            calls.append(path)
            return {"status": "completed", "cost_final": True}

    # non-completed status: read immediately, no poll
    r = {"status": "failed"}
    assert cli_mod._await_cost_finality(_Client(), "t", r) is r and not calls
    # no explicit partial fields: read immediately
    r2 = {"status": "completed"}
    assert cli_mod._await_cost_finality(_Client(), "t", r2) is r2 and not calls
    # already final: read immediately
    r3 = {"status": "completed", "cost_final": True}
    assert cli_mod._await_cost_finality(_Client(), "t", r3) is r3 and not calls
    # explicitly partial: polls until final
    r4 = {"status": "completed", "cost_final": False, "cost_with_children_partial": True}
    out = cli_mod._await_cost_finality(_Client(), "t", r4, grace_sec=5.0)
    assert out["cost_final"] is True and calls


def test_pb_cost_finality_wait_branches(monkeypatch):
    from devtools.benchmarks.programbench import programbench_adapter as pb

    calls = []
    monkeypatch.setattr(
        pb, "ouroboros_api_request",
        lambda base, method, path, body=None, timeout=None: calls.append(path)
        or {"status": "completed", "cost_final": True},
    )
    r = {"status": "cancelled"}
    assert pb._await_cost_finality("http://x", "t", r) is r and not calls
    r2 = {"status": "degraded"}
    assert pb._await_cost_finality("http://x", "t", r2) is r2 and not calls
    r3 = {"status": "completed", "cost_final": False}
    out = pb._await_cost_finality("http://x", "t", r3, grace_sec=5.0)
    assert out["cost_final"] is True and calls


def test_dialogue_quorum_fallback_uses_adaptive_quorum():
    # Pins the fallback branch (record without policy.min_successful_slots) and
    # with it the adaptive_quorum import in loop.py (commit triad advisory).
    result = NS(request={}, actors=[_actor(f"s{i}", "FAIL", {"verdict": "FAIL"}) for i in range(3)])
    assert loop_acceptance_review._acceptance_dialogue_quorum(result) == 2  # adaptive_quorum(3)
    assert loop_acceptance_review._acceptance_dialogue_quorum(NS(request=None, actors=[])) == 1


def test_contract_demoted_actor_cannot_vote_terminal():
    # commit triad sol #1: parse_status="malformed" (contract-demoted or garbage
    # with a verdict-shaped dict) must not satisfy terminal quorum.
    res = _result(
        [
            _actor("s1", "DEGRADED", {"verdict": "FAIL", "dialogue_status": "unreachable_here"},
                   parse_status="malformed"),
            _actor("s2", "DEGRADED", {"verdict": "FAIL", "dialogue_status": "unreachable_here"},
                   parse_status="malformed"),
        ],
        aggregate="DEGRADED",
    )
    assert aggregate_dialogue_status(res, quorum=2)["status"] == DIALOGUE_CONTINUE


def test_typed_findings_cannot_resurrect_settled_row_via_content_match():
    # commit triad r2 (sol, obl-0002): a typed "new" or an UNBOUND re_raise whose
    # text matches a settled row must NOT reopen it — the sloppy signal is
    # disclosed on the row; only an UNTYPED legacy finding reopens by content.
    trace = {"acceptance_obligations": []}
    loop_mod._collect_acceptance_obligations(trace, _fail_result_with([_finding()]))
    row = trace["acceptance_obligations"][0]
    row.update(status="agent_disposed", disposition="rejected", disposition_reason="arg")
    loop_mod._collect_acceptance_obligations(trace, _fail_result_with([
        _finding(disposition_kind="re_raise", obligation_id="ob-wrongid00000"),
    ]))
    assert len(trace["acceptance_obligations"]) == 1
    assert row["status"] == "agent_disposed"  # settled state NOT resurrected
    assert int(row.get("reopened_count") or 0) == 0
    assert "re_raise_unbound:ob-wrongid00000" in row.get("notes", [])
    loop_mod._collect_acceptance_obligations(trace, _fail_result_with([
        _finding(disposition_kind="new"),
    ]))
    assert row["status"] == "agent_disposed"
    assert any(n.startswith("typed_new_matched_existing:") for n in row.get("notes", []))
    # untyped legacy finding still reopens by byte-identical content
    loop_mod._collect_acceptance_obligations(trace, _fail_result_with([_finding()]))
    assert row["status"] == "open" and row["reopened_count"] == 1


# ── v6.74.4: count-axis freeze directive in the rails line ───────────────────


def _rails(passes_done, *, cap=6, workspace=False, required_blocking=False):
    snap = NS(has_deadline=False)
    profile = {"max_improvement_passes": cap} if cap is not None else {}
    return task_pacing._acceptance_rails_line_inner(
        snap, profile, passes_done, None,
        required_blocking=required_blocking, workspace=workspace,
    )


def test_rails_final_pass_freeze_directive_workspace():
    # The pass launched at cap-1 is the last one improvement_pass_allowed
    # admits: the FINAL marker must ride that rails line. The tree directive
    # rides EVERY workspace rails line (commit triad r1, sol: a deadline/cost
    # rail can end the loop between capsules), and never a non-workspace one.
    line = _rails(5, cap=6, workspace=True)
    assert "review passes: 5/6" in line
    assert "FINAL improvement pass, no further passes will run" in line
    assert "working tree as it stands" in line and "VERIFIED state" in line
    # Non-workspace: factual finality only, no tree directive.
    plain = _rails(5, cap=6, workspace=False)
    assert "FINAL improvement pass" in plain
    assert "working tree" not in plain


def test_rails_freeze_directive_absent_off_final_and_edge_caps(monkeypatch):
    # Non-final pass: no FINAL marker, but the workspace tree directive is
    # always present for workspace deliveries.
    mid = _rails(3, cap=6, workspace=True)
    assert "FINAL" not in mid and "review passes: 3/6" in mid
    assert "working tree as it stands" in mid
    assert "working tree" not in _rails(3, cap=6, workspace=False)
    # cap == 0 never feeds a capsule back — no misleading FINAL rail.
    zero = _rails(0, cap=0, workspace=True)
    assert "review passes: 0/0" in zero and "FINAL" not in zero
    # Passes already exhausted (supersede-reset re-review): not a launch.
    spent = _rails(6, cap=6, workspace=True)
    assert "review passes: 6/6" in spent and "FINAL" not in spent
    # No local cap (required+blocking with the shared review-cycle cap set to
    # unlimited — D10/D20: otherwise the shared cap binds) — no count-axis clause.
    monkeypatch.setenv("OUROBOROS_REVIEW_MAX_CYCLES", "unlimited")
    unbounded = _rails(4, cap=None, workspace=True, required_blocking=True)
    assert "no local count cap" in unbounded and "FINAL" not in unbounded


@pytest.mark.parametrize("passes_done,cap", [(4, 6), (5, 6), (6, 6), (0, 1), (0, 0), (3, None)])
def test_rails_final_clause_matches_pacing_seam(passes_done, cap):
    # codex finding 4: the FINAL clause must appear exactly when the pacing
    # SSOT admits the CURRENT pass but would deny the NEXT one — same
    # profile/counters through both surfaces, so drift between
    # improvement_pass_allowed and the rails renderer cannot stay green.
    from ouroboros import task_pacing

    profile = {"max_improvement_passes": cap} if cap is not None else {}
    snap = NS(has_deadline=False, spendable_sec=1e9)
    required_blocking = cap is None
    now_ok, _ = task_pacing.improvement_pass_allowed(
        snap, passes_done, profile, required_blocking=required_blocking)
    next_ok, _ = task_pacing.improvement_pass_allowed(
        snap, passes_done + 1, profile, required_blocking=required_blocking)
    line = _rails(passes_done, cap=cap, workspace=True,
                  required_blocking=required_blocking)
    assert ("FINAL improvement pass" in line) == (now_ok and not next_ok)
