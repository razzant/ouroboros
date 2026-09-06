"""v6.78.0 (phase P4) — the host acceptance-decision state machine, driven through the
REAL writers, plus the two transparency facts that ride the same evidence packet.

The harness half of phase P4: every terminal branch of the host acceptance state machine
is exercised through `loop`'s actual writers so the canonical status + typed reason pair
is pinned per branch (the before/after transition table), the outcome reducer's
status+reason predicate is pinned separately because losing that pairing is a silent
false green, and the capability-omission formatter and the native-`retrieval` fact are
checked where they reach (and where they must NOT reach) a reviewer. The PURE receipt
identity/reconciliation core those nudges consult lives in
`test_v678_receipt_reconciliation.py`.

Table-driven and offline: no live reviewer panel, no ambient checkout state.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

import ouroboros.loop as loop_mod
from ouroboros.loop_acceptance import (
    ACCEPTANCE_DECISION_REASONS,
    _set_acceptance_decision,
    _supersede_task_acceptance_for_evidence_change,
    _supersede_task_acceptance_for_owner_followup,
)
from ouroboros.loop_acceptance_review import (
    _apply_task_acceptance_result,
    _record_acceptance_infra_failure,
)
from ouroboros.outcomes import (
    ACCEPTANCE_ACCEPTED,
    ACCEPTANCE_FINALIZED_UNACCEPTED,
    ACCEPTANCE_REVISION_REQUESTED,
    REASON_ACCEPTANCE_REVIEW_SKIPPED_DEADLINE_RESERVE,
    derive_loop_outcome,
)


# ---------------------------------------------------------------------------
# helpers


def _actor(slot_id, signal, parsed):
    return {"slot_id": slot_id, "signal": signal, "parsed": parsed}


def _result(*, aggregate, actors, findings=(), degraded=False, degraded_reasons=(), quorum=1,
            policy=None):
    import ouroboros.review_substrate as rs

    base_policy = {"min_successful_slots": quorum}
    base_policy.update(policy or {})
    return rs.ReviewRunResult(
        request={"surface": "task_acceptance", "policy": base_policy},
        actors=list(actors),
        parsed_findings=list(findings),
        aggregate_signal=aggregate,
        degraded=degraded,
        degraded_reasons=list(degraded_reasons),
    )


def _apply_ctx(tmp_path, *, prior_trace=None, passes_done=0, budget_profile=None, fence_ok=True):
    """Minimal _TaskAcceptanceContext over a fake tool ctx (no queue, no provider)."""
    tool_ctx = SimpleNamespace(
        _task_acceptance_reviewed=False,
        _task_acceptance_improvement_passes=passes_done,
        drive_root=str(tmp_path),
        task_metadata={},
        task_contract={"budget_profile": budget_profile} if budget_profile else {},
        is_direct_chat=False,
        end_acceptance_fence=(lambda **_k: {"ok": True}) if fence_ok else (lambda **_k: {"ok": False}),
        _task_acceptance_fence_token="tok",
    )
    trace = dict(prior_trace or {})
    trace.setdefault("tool_calls", [{"tool": "write_file", "args": {"path": "x.py"}}])
    return loop_mod._TaskAcceptanceContext(
        tools=SimpleNamespace(_ctx=tool_ctx),
        content="done",
        task_id="t",
        task_type="task",
        llm_trace=trace,
        drive_root=None,
        messages=[{"role": "user", "content": "goal"}],
        emit_progress=lambda _m, *, incident=None: None,
        mode="required",
        subtree_statuses=[],
        budget_profile=budget_profile or {},
        passes_done=passes_done,
    )


_OPEN_OBLIGATION = {
    "id": "ob-1", "item": "broken", "recommendation": "fix",
    "status": "open", "disposition": "",
}

# The clean-pass fixture carries supported criteria WITH evidence_refs: the
# evidence condition is unconditional since D-Q5 deleted the constant-true
# require_criterion_evidence knob (production panels always ran it True, so a
# real clean PASS always looked exactly like this).
_CLEAN_PASS = dict(
    aggregate="PASS",
    actors=[_actor("s0", "PASS", {
        "verdict": "PASS", "outcome_tier": "solved",
        "criteria_used": [{
            "criterion": "deliverable is verified", "status": "supported",
            "evidence_refs": ["verification_summary"],
        }],
    })],
)
# A PASS whose `solved` claim lacks supported criterion evidence: NOT clean, yet the
# capsule has nothing actionable (no findings, no dissent, solved tier) — the A14 branch.
_NON_CLEAN_PASS_NO_CAPSULE = dict(
    aggregate="PASS",
    policy={"require_criterion_evidence": True},
    actors=[_actor("s0", "PASS", {
        "verdict": "PASS", "outcome_tier": "solved",
        "criteria_used": [{"criterion": "deliverable is verified", "status": "missing"}],
    })],
)
# A contributing FAIL with a coach and NO findings: the capsule is the bounded
# correction rail, but no obligation is ever minted (obligations need a finding).
_FAIL_COACH_ONLY = dict(
    aggregate="FAIL",
    actors=[_actor("s0", "FAIL", {
        "verdict": "FAIL", "outcome_tier": "best_effort", "completion_coach": "fix it",
        "dialogue_status": "continue_actionable",
    })],
)
_ACTIONABLE_FAIL = dict(
    aggregate="FAIL",
    actors=[_actor("s0", "FAIL", {
        "verdict": "FAIL", "outcome_tier": "blocked_with_evidence",
        "completion_coach": "fix it", "dialogue_status": "continue_actionable",
    })],
    findings=[{"slot_id": "s0", "severity": "critical", "item": "broken",
               "recommendation": "fix the header"}],
)
_BARE_FAIL = dict(
    aggregate="FAIL",
    actors=[_actor("s0", "FAIL", {"verdict": "FAIL"})],
)
_NO_QUORUM = dict(
    aggregate="DEGRADED", actors=[], degraded=True, degraded_reasons=["quorum failure"],
)
_DIALOGUE_TERMINAL = dict(
    aggregate="FAIL",
    actors=[_actor("s0", "FAIL", {
        "verdict": "FAIL", "outcome_tier": "best_effort", "completion_coach": "fix",
        "dialogue_status": "stable_disagreement",
    })],
    findings=[{"slot_id": "s0", "severity": "critical", "item": "broken",
               "recommendation": "fix the header"}],
)


# ---------------------------------------------------------------------------
# P4.2 — the three-state collapse, one row per terminal branch


@pytest.mark.parametrize(
    "label,result_kwargs,kwargs,expected",
    [
        ("clean_pass", _CLEAN_PASS, {}, (ACCEPTANCE_ACCEPTED, "clean_pass")),
        (
            "clean_pass_obligations_closed", _CLEAN_PASS,
            {"prior_trace": {"acceptance_obligations": [dict(_OPEN_OBLIGATION)]}},
            (ACCEPTANCE_ACCEPTED, "clean_pass_obligations_closed"),
        ),
        (
            # Round-9 CRITICAL 1: this row used to expect `accepted`. The panel is a
            # PASS whose `solved` claim carries a MISSING criterion, so the clean
            # predicate has already refused it — `accepted` is exactly the state the
            # documented requirement reserves for clean acceptance.
            "no_actionable_changes", _NON_CLEAN_PASS_NO_CAPSULE,
            {"passes_done": 1, "budget_profile": {"max_improvement_passes": 1}},
            (ACCEPTANCE_FINALIZED_UNACCEPTED, "no_actionable_changes"),
        ),
        ("improvement_capsule", _ACTIONABLE_FAIL, {}, (ACCEPTANCE_REVISION_REQUESTED, "improvement_capsule")),
        (
            "fence_reopen_failed", _ACTIONABLE_FAIL, {"fence_ok": False},
            (ACCEPTANCE_FINALIZED_UNACCEPTED, "fence_reopen_failed"),
        ),
        ("dialogue_terminal", _DIALOGUE_TERMINAL, {}, (ACCEPTANCE_FINALIZED_UNACCEPTED, "dialogue_terminal")),
        ("review_degraded", _NO_QUORUM, {}, (ACCEPTANCE_FINALIZED_UNACCEPTED, "review_degraded")),
        (
            "open_obligations_gates_exhausted", _ACTIONABLE_FAIL,
            {
                "passes_done": 1, "budget_profile": {"max_improvement_passes": 1},
                "prior_trace": {"acceptance_obligations": [dict(_OPEN_OBLIGATION)]},
            },
            (ACCEPTANCE_FINALIZED_UNACCEPTED, "open_obligations"),
        ),
        (
            "capsule_spent", _FAIL_COACH_ONLY,
            {"passes_done": 1, "budget_profile": {"max_improvement_passes": 1}},
            (ACCEPTANCE_FINALIZED_UNACCEPTED, "capsule_spent"),
        ),
        (
            "improvement_window_closed", _FAIL_COACH_ONLY,
            {"passes_done": 0, "budget_profile": {"max_improvement_passes": 0}},
            (ACCEPTANCE_FINALIZED_UNACCEPTED, "improvement_window_closed"),
        ),
        (
            "reviewer_fail_no_capsule", _BARE_FAIL,
            {"passes_done": 1, "budget_profile": {"max_improvement_passes": 1}},
            (ACCEPTANCE_FINALIZED_UNACCEPTED, "reviewer_fail_no_capsule"),
        ),
    ],
)
def test_terminal_branch_maps_to_canonical_status_and_typed_reason(
    monkeypatch, tmp_path, label, result_kwargs, kwargs, expected,
):
    monkeypatch.setattr(loop_mod, "get_review_enforcement", lambda: "blocking")
    ctx = _apply_ctx(tmp_path, **kwargs)
    _apply_task_acceptance_result(ctx, _result(**result_kwargs), record_run=False)
    decision = ctx.llm_trace["acceptance_decision"]
    assert (decision["status"], decision["reason"]) == expected, label
    assert decision["reason"] in ACCEPTANCE_DECISION_REASONS


_NON_CLEAN_FIXTURES = (
    ("non_clean_pass_no_capsule", _NON_CLEAN_PASS_NO_CAPSULE,
     {"passes_done": 1, "budget_profile": {"max_improvement_passes": 1}}),
    ("fail_coach_only_spent", _FAIL_COACH_ONLY,
     {"passes_done": 1, "budget_profile": {"max_improvement_passes": 1}}),
    ("actionable_fail_spent", _ACTIONABLE_FAIL,
     {"passes_done": 1, "budget_profile": {"max_improvement_passes": 1}}),
    ("actionable_fail_spent_with_obligations", _ACTIONABLE_FAIL,
     {"passes_done": 1, "budget_profile": {"max_improvement_passes": 1},
      "prior_trace": {"acceptance_obligations": [dict(_OPEN_OBLIGATION)]}}),
    ("bare_fail", _BARE_FAIL,
     {"passes_done": 1, "budget_profile": {"max_improvement_passes": 1}}),
    ("improvement_window_closed", _FAIL_COACH_ONLY,
     {"passes_done": 0, "budget_profile": {"max_improvement_passes": 0}}),
    ("no_quorum", _NO_QUORUM, {}),
    ("dialogue_terminal", _DIALOGUE_TERMINAL, {}),
    ("actionable_fail_fence_reopen_failed", _ACTIONABLE_FAIL, {"fence_ok": False}),
)


@pytest.mark.parametrize("label,result_kwargs,kwargs", _NON_CLEAN_FIXTURES)
def test_accepted_is_authorised_only_by_clean_acceptance(
    monkeypatch, tmp_path, label, result_kwargs, kwargs,
):
    """Round-9 CRITICAL 1 (the state-machine invariant, not one branch).

    `docs/ARCHITECTURE.md` reserves the `accepted` state for clean acceptance —
    "Clean means quorum PASS, `solved`, and supported evidence for every contributing
    criterion" — and `review_substrate.task_acceptance_is_clean` is the predicate that
    decides it. Every fixture below is one the predicate has already REFUSED, so none
    of them may leave the state machine holding `accepted`, whatever the capsule and
    obligation shape happens to be. The old bug was the final no-capsule/no-obligation
    fall-through: a reviewer saying `solved` while supplying a MISSING criterion, with
    the improvement-pass cap exhausted, landed as `accepted` — the clean predicate's
    refusal silently overruled by the branch that runs after it.
    """
    from ouroboros.review_substrate import task_acceptance_is_clean

    monkeypatch.setattr(loop_mod, "get_review_enforcement", lambda: "blocking")
    result = _result(**result_kwargs)
    assert not task_acceptance_is_clean(result), f"{label}: fixture is not the non-clean case"
    ctx = _apply_ctx(tmp_path, **kwargs)
    _apply_task_acceptance_result(ctx, result, record_run=False)
    decision = ctx.llm_trace["acceptance_decision"]
    assert decision["status"] != ACCEPTANCE_ACCEPTED, f"{label}: {decision}"
    assert decision["reason"] in ACCEPTANCE_DECISION_REASONS, label


def test_clean_acceptance_still_authorises_accepted(monkeypatch, tmp_path):
    """Positive control for the invariant above: the fix must not make `accepted`
    unreachable — the clean predicate still authorises it, with and without
    obligations to close."""
    from ouroboros.review_substrate import task_acceptance_is_clean

    monkeypatch.setattr(loop_mod, "get_review_enforcement", lambda: "blocking")
    result = _result(**_CLEAN_PASS)
    assert task_acceptance_is_clean(result)
    for prior in ({}, {"acceptance_obligations": [dict(_OPEN_OBLIGATION)]}):
        ctx = _apply_ctx(tmp_path, prior_trace=dict(prior))
        _apply_task_acceptance_result(ctx, result, record_run=False)
        assert ctx.llm_trace["acceptance_decision"]["status"] == ACCEPTANCE_ACCEPTED


def test_non_clean_pass_no_capsule_owner_line_does_not_read_as_acceptance(monkeypatch, tmp_path):
    """The owner-visible line for the fixed branch must say the task was finalized
    without acceptance. The old line ("no changes suggested") was true and useless:
    it read identically to a clean pass while the decision said `accepted`."""
    monkeypatch.setattr(loop_mod, "get_review_enforcement", lambda: "blocking")
    lines: list[str] = []
    ctx = _apply_ctx(
        tmp_path, passes_done=1, budget_profile={"max_improvement_passes": 1},
    )
    ctx.emit_progress = lines.append
    _apply_task_acceptance_result(ctx, _result(**_NON_CLEAN_PASS_NO_CAPSULE), record_run=False)
    assert lines and "finaliz" in lines[-1].lower(), lines
    assert "without acceptance" in lines[-1].lower(), lines


def test_bare_fail_reaches_the_reviewer_fail_branch_not_a_capsule():
    """Guard for the fixture above: a bare veto really produces NO capsule, so the
    `reviewer_fail_no_capsule` row exercises the intended branch."""
    import ouroboros.review_substrate as rs

    assert rs.build_improvement_capsule(_result(**_BARE_FAIL)) == ""
    assert rs.build_improvement_capsule(_result(**_ACTIONABLE_FAIL)).strip()


def test_infra_failure_branch_is_finalized_unaccepted(tmp_path):
    ctx = _apply_ctx(tmp_path)
    assert _record_acceptance_infra_failure(ctx, RuntimeError("boom")) is False
    decision = ctx.llm_trace["acceptance_decision"]
    assert decision["status"] == ACCEPTANCE_FINALIZED_UNACCEPTED
    assert decision["reason"] == "infra_failure"
    assert decision["degraded_reasons"] == ["RuntimeError: boom"]
    # The synthetic DEGRADED run record — the typed expression of "never a silent
    # skip" — is unchanged by the host-status collapse.
    assert ctx.llm_trace["review_runs"][-1]["aggregate_signal"] == "DEGRADED"


def test_supersede_paths_request_a_revision_with_their_own_reason(tmp_path):
    ctx = SimpleNamespace(
        _task_acceptance_reviewed=True,
        _task_acceptance_fence_generation_mismatch=False,
        end_acceptance_fence=lambda **_k: True,
        _task_acceptance_fence_token="tok",
    )
    trace = {"review_runs": []}
    _supersede_task_acceptance_for_owner_followup(ctx, trace)
    assert trace["acceptance_decision"]["status"] == ACCEPTANCE_REVISION_REQUESTED
    assert trace["acceptance_decision"]["reason"] == "owner_followup"

    trace2 = {"review_runs": []}
    _supersede_task_acceptance_for_evidence_change(
        ctx, trace2, None, "host_acceptance_evidence_revision_changed", [], lambda _m: None,
    )
    assert trace2["acceptance_decision"]["status"] == ACCEPTANCE_REVISION_REQUESTED
    assert trace2["acceptance_decision"]["reason"] == "evidence_refresh"
    # The caller's trigger keeps its existing home; no new detail surface was added.
    assert trace2["review_decision"]["trigger"] == "host_acceptance_evidence_revision_changed"


# ---------------------------------------------------------------------------
# P4.2 readers — the status+reason predicate (gap G1) and the projection (G2)


@pytest.mark.parametrize(
    "decision,eligibility,degrades",
    [
        (
            {"status": ACCEPTANCE_FINALIZED_UNACCEPTED,
             "reason": REASON_ACCEPTANCE_REVIEW_SKIPPED_DEADLINE_RESERVE},
            "eligible", True,
        ),
        # Same canonical status, DIFFERENT reason -> this specific degradation must
        # NOT fire (otherwise every terminal acceptance would claim a deadline skip).
        (
            {"status": ACCEPTANCE_FINALIZED_UNACCEPTED, "reason": "review_degraded"},
            "eligible", False,
        ),
        # Right reason, not eligible -> unchanged behaviour.
        (
            {"status": ACCEPTANCE_FINALIZED_UNACCEPTED,
             "reason": REASON_ACCEPTANCE_REVIEW_SKIPPED_DEADLINE_RESERVE},
            "not_eligible", False,
        ),
        # Forced-rail bypass reasons (closed enum) ride the SAME (status, reason,
        # eligibility) key — an eligible panel bypassed by a rail is never clean.
        (
            {"status": ACCEPTANCE_FINALIZED_UNACCEPTED,
             "reason": "acceptance_bypassed_budget_exhausted"},
            "eligible", True,
        ),
        (
            {"status": ACCEPTANCE_FINALIZED_UNACCEPTED,
             "reason": "acceptance_bypassed_round_limit"},
            "eligible", True,
        ),
        (
            {"status": ACCEPTANCE_FINALIZED_UNACCEPTED,
             "reason": "acceptance_bypassed_deadline"},
            "eligible", True,
        ),
        (
            {"status": ACCEPTANCE_FINALIZED_UNACCEPTED,
             "reason": "acceptance_bypassed_provider_unavailable"},
            "eligible", True,
        ),
        (
            {"status": ACCEPTANCE_FINALIZED_UNACCEPTED,
             "reason": "acceptance_bypassed_children_unabsorbed"},
            "eligible", True,
        ),
        # Bypass reason without eligibility (unknown / not_eligible) never degrades.
        (
            {"status": ACCEPTANCE_FINALIZED_UNACCEPTED,
             "reason": "acceptance_bypassed_budget_exhausted"},
            "not_eligible", False,
        ),
        (
            {"status": ACCEPTANCE_FINALIZED_UNACCEPTED,
             "reason": "acceptance_bypassed_budget_exhausted"},
            "unknown", False,
        ),
    ],
)
def test_deadline_reserve_degradation_keys_on_status_plus_reason(decision, eligibility, degrades):
    outcome = derive_loop_outcome(
        "FINAL ANSWER: best available answer",
        {},
        {
            "acceptance_decision": dict(decision),
            "review_decision": {"eligibility": eligibility, "trigger": "auto_nondirect"},
            "tool_calls": [],
        },
    )
    axes = outcome["outcome_axes"]
    is_deadline_reserve = (
        decision["reason"] == REASON_ACCEPTANCE_REVIEW_SKIPPED_DEADLINE_RESERVE
    )
    if degrades:
        assert axes["execution"]["status"] == "degraded"
        # The typed reason itself is the degradation reason_code — deadline-reserve
        # keeps its historical token, a forced-rail bypass carries its own.
        assert axes["execution"]["reason_code"] == decision["reason"]
        assert axes["objective"]["status"] == "degraded"
        assert axes["objective"]["source"] == (
            "task_acceptance_deadline_reserve"
            if is_deadline_reserve else "task_acceptance_forced_bypass"
        )
    else:
        assert axes["execution"]["status"] == "ok"
        assert axes["objective"].get("source") not in {
            "task_acceptance_deadline_reserve", "task_acceptance_forced_bypass",
        }


def test_forced_rail_axes_are_the_production_shape(tmp_path, monkeypatch):
    """The table above feeds `derive_loop_outcome` a usage dict a forced rail
    cannot produce (`usage={}`), so it pins the pair-key, not the delivered shape.
    Driven through the REAL round-limit rail, a stamped bypass ALWAYS arrives with
    `usage.execution_status='failed'` — every writer of `usage.reason_code` writes
    it — so the outcome lands on the stronger best_effort/failed classification and
    the pair-keyed degrade never decides. The owner-visible "a panel was owed and
    did not run" fact rides the REVIEW axis, which is the claim that must hold."""
    from tests.test_delivery_forced_finalization import _forced_test_context

    loop, _registry, limit_ctx, _trace = _forced_test_context(tmp_path)
    monkeypatch.setattr(
        loop, "call_llm_with_retry",
        lambda *_args, **_kwargs: (
            {"role": "assistant", "content": "Best answer before the round limit."}, 0.0,
        ),
    )

    text, usage, trace = loop._handle_round_limit(limit_ctx)

    # The rail's own execution truth: failed usage + the rail reason code.
    assert usage["execution_status"] == "failed"
    assert usage["reason_code"] == "round_limit"
    assert trace["acceptance_decision"]["reason"] == "acceptance_bypassed_round_limit"

    axes = derive_loop_outcome(text, usage, trace)["outcome_axes"]
    # NOT "degraded/acceptance_bypassed_round_limit": the honest rail reason wins.
    assert axes["execution"]["status"] == "best_effort"
    assert axes["execution"]["reason_code"] == "round_limit"
    # ...and the bypass is carried where it is owner-visible and unambiguous.
    assert axes["review"]["eligibility"] == "eligible"
    assert axes["review"]["trigger"] == "bypassed_round_limit"
    assert axes["review"]["run_count"] == 0
    assert axes["review"]["acceptance_decision"]["reason"] == (
        "acceptance_bypassed_round_limit"
    )


def test_round_one_budget_rejection_is_also_a_covered_forced_sink(tmp_path):
    """The total-budget rejection at round<=1 returns its notice directly instead
    of going through `_forced_final_answer`, so it used to reach the ledger with
    NO bypass record at all: `eligibility=not_evaluated / run_count=0` — exactly
    the "indistinguishable from no panel warranted" shape the typed bypass closes.
    Nothing was produced, so the record is the whole remedy: one ledger write."""
    from tests.test_delivery_forced_finalization import _forced_test_context

    loop, _registry, limit_ctx, _trace = _forced_test_context(tmp_path)
    limit_ctx.round_idx = 1

    result = loop._check_budget_limits(limit_ctx, 0.0)

    assert result is not None
    text, usage, trace = result
    assert text.startswith("🚫 Task rejected.")
    assert usage["reason_code"] == "budget_exhausted"
    assert trace["review_decision"] == {
        "eligibility": "eligible", "trigger": "bypassed_budget_exhausted",
    }
    assert trace["acceptance_decision"]["reason"] == "acceptance_bypassed_budget_exhausted"
    assert trace["acceptance_decision"]["source"] == "forced_finalization"
    # No candidate existed — the record says so instead of inventing one.
    assert trace["forced_finalization"]["candidate_sha256"] == ""
    assert trace["forced_finalization"]["source"] == "host_budget_rejection_before_work"

    axes = derive_loop_outcome(text, usage, trace)["outcome_axes"]
    assert axes["review"]["eligibility"] == "eligible"
    assert axes["review"]["run_count"] == 0


def test_acceptance_projection_carries_the_typed_reason():
    from ouroboros.outcomes import _acceptance_decision_projection

    out = _acceptance_decision_projection({
        "status": ACCEPTANCE_FINALIZED_UNACCEPTED,
        "reason": "review_degraded",
        "source": "task_acceptance_review",
        "rationale": "no quorum",
        "open_obligations": ["ob-1"],
    })
    assert out["status"] == ACCEPTANCE_FINALIZED_UNACCEPTED
    assert out["reason"] == "review_degraded"
    assert out["open_obligations"] == ["ob-1"]
    # Historical records (pre-collapse tokens, no reason) still project verbatim:
    # the projection is a passthrough and no normalizer was introduced.
    legacy = _acceptance_decision_projection({"status": "best_effort_open_obligations"})
    assert legacy["status"] == "best_effort_open_obligations"
    assert legacy["reason"] == ""


# ---------------------------------------------------------------------------
# P4.3 — the reconciliation narrowing stays ADVISORY


def test_semantically_different_green_yields_one_advisory_nudge_and_no_gate(monkeypatch, tmp_path):
    """The panel's required case: a red receipt plus a later, semantically different
    passing receipt (`pytest tests/x.py` -> `pytest tests/x.py -v`) produces AT MOST
    ONE advisory nudge, the acceptance panel can still reach PASS, and the forced
    finalization path is untouched (it returns before the injector runs)."""
    from ouroboros.outcomes import append_verification_receipt
    from ouroboros.review_substrate import task_acceptance_is_clean

    drive_root = tmp_path
    append_verification_receipt(drive_root, "t", {"status": "fail", "check": "pytest tests/x.py"})
    append_verification_receipt(drive_root, "t", {"status": "pass", "check": "pytest tests/x.py -v"})

    tool_ctx = SimpleNamespace(drive_root=str(tmp_path))
    tools = SimpleNamespace(_ctx=tool_ctx)
    trace = {"reasoning_notes": [], "tool_calls": [{"tool": "write_file", "args": {"path": "x.py"}}]}
    messages: list = []
    monkeypatch.setattr(loop_mod, "_skill_finalization_message", lambda *_a, **_k: "")

    fired = [
        loop_mod._maybe_inject_finalization_nudges(
            tools, tmp_path, "t", trace, "answer", messages, lambda _m: None,
        )
        for _ in range(4)
    ]
    assert fired[0] is True                     # one advisory nudge
    assert fired.count(True) == 1               # ...exactly one, latched
    nudges = [m for m in messages if "RED" in str(m.get("content") or "")]
    assert len(nudges) == 1
    assert "advisory" in str(nudges[0]["content"])

    # ADVISORY, not a gate: a clean panel PASS is still clean with the red on file.
    assert task_acceptance_is_clean(_result(**_CLEAN_PASS)) is True


def test_red_receipt_reconciled_by_the_same_check_raises_no_nudge(monkeypatch, tmp_path):
    from ouroboros.outcomes import append_verification_receipt

    append_verification_receipt(tmp_path, "t", {"status": "fail", "check": "pytest tests/x.py"})
    append_verification_receipt(tmp_path, "t", {"status": "pass", "check": "pytest  tests/x.py "})
    monkeypatch.setattr(loop_mod, "_skill_finalization_message", lambda *_a, **_k: "")
    tools = SimpleNamespace(_ctx=SimpleNamespace(drive_root=str(tmp_path)))
    trace = {"reasoning_notes": [], "tool_calls": [{"tool": "write_file", "args": {"path": "x.py"}}]}
    assert loop_mod._maybe_inject_finalization_nudges(
        tools, tmp_path, "t", trace, "answer", [], lambda _m: None,
    ) is False


# ---------------------------------------------------------------------------
# P4.4 — transparency


def test_capability_omission_manifest_has_one_formatter_with_the_richest_detail():
    from ouroboros.tool_policy import CAPABILITY_OMISSION_HEADER, format_capability_omissions

    lines = format_capability_omissions([
        {"surface": "tools", "reason": "disabled_by_contract", "tools": ["web_search", "browse_page"]},
        {"surface": "extensions", "reason": "resource_blocked", "resource": "network=false"},
        {"surface": "mcp", "reason": "load_failed", "error": "boom"},
        "not-a-dict",
    ])
    assert lines[0] == CAPABILITY_OMISSION_HEADER
    # The real withheld NAMES are rendered (owner Q20), not "no detail".
    assert lines[1] == "- tools: disabled_by_contract (web_search, browse_page)"
    assert lines[2] == "- extensions: resource_blocked (network=false)"
    assert lines[3] == "- mcp: load_failed (boom)"
    assert len(lines) == 4
    assert format_capability_omissions([], header="") == []


def test_only_one_capability_omission_formatter_remains_in_the_tree():
    """The five copies are gone: no module renders the manifest line itself."""
    import pathlib

    import ouroboros.loop_tool_execution as lte  # noqa: F401  (tree sanity import)

    root = pathlib.Path(loop_mod.__file__).parent
    hits = [
        path.name
        for path in root.rglob("*.py")
        if "item.get('reason', 'unknown')" in path.read_text(encoding="utf-8")
        or 'item.get("reason", "unknown")' in path.read_text(encoding="utf-8")
    ]
    assert hits == ["tool_policy.py"], hits


def test_retrieval_fold_accumulates_counts_and_capped_urls():
    from ouroboros.loop_llm_call import fold_retrieval_usage

    accumulated: dict = {}
    fold_retrieval_usage(accumulated, {"prompt_tokens": 10})
    assert "retrieval" not in accumulated              # no search -> no fact at all

    fold_retrieval_usage(accumulated, {
        "server_tool_use": {"web_search_requests": 2},
        "web_search_sources": [
            {"url": "https://a.example/1", "title": "t", "content": "snippet"},
            {"url": "https://a.example/1"},
        ],
    })
    fold_retrieval_usage(accumulated, {
        "web_search_sources": [{"url": "https://b.example/2"}],
    })
    record = accumulated["retrieval"]
    assert record["web_search_requests"] == 3          # 2 + citations-without-counter
    assert record["source_count"] == 3
    assert record["urls"] == ["https://a.example/1", "https://b.example/2"]

    assert record["urls_omitted"] == 0                # nothing dropped -> say so
    assert len(record["urls_identity_sha256"]) == 64  # full-set identity, always

    # cap + no titles/snippets ever — and the cap DISCLOSES what it dropped (round 2)
    many = {"web_search_sources": [{"url": f"https://c.example/{i}"} for i in range(40)]}
    fresh: dict = {}
    fold_retrieval_usage(fresh, many)
    assert len(fresh["retrieval"]["urls"]) == 20
    assert fresh["retrieval"]["urls_omitted"] == 20   # exact count, not silence
    assert fresh["retrieval"]["source_count"] == 40
    assert all(isinstance(u, str) for u in fresh["retrieval"]["urls"])
    long_url = {"web_search_sources": [{"url": "https://d.example/" + "x" * 500}]}
    fresh2: dict = {}
    fold_retrieval_usage(fresh2, long_url)
    # per-URL bound goes through the SSOT truncator -> carries its own omission note
    assert "OMISSION NOTE" in fresh2["retrieval"]["urls"][0]
    assert fresh2["retrieval"]["urls"][0].startswith("https://d.example/xxx")


def test_retrieval_reaches_the_acceptance_reviewer_host_attested(tmp_path):
    from ouroboros.review_evidence import build_task_acceptance_evidence

    ctx = SimpleNamespace(task_metadata={}, task_contract={}, drive_root=str(tmp_path))
    ev = build_task_acceptance_evidence(
        ctx,
        llm_trace={
            "tool_calls": [],
            "retrieval": {
                "web_search_requests": 2, "source_count": 2,
                "urls": ["https://a.example/1", "https://b.example/2"],
            },
        },
        drive_root=None,
        task_id="t",
        canonical_subject="answer",
    )
    assert ev["retrieval"] == {
        "web_search_requests": 2,
        "source_count": 2,
        "urls": ["https://a.example/1", "https://b.example/2"],
        "urls_omitted": 0,
    }
    assert ev["__provenance__"]["retrieval"] == "host_attested"


def test_absent_retrieval_is_not_a_deficiency_and_leaves_the_verdict_unchanged(tmp_path):
    """A correct knowledge-only answer (`retrieval=none`) must produce the same packet
    shape and the same panel verdict as before this feature, and the rules must say
    absence is neutral so a reviewer cannot read it as a gap."""
    from ouroboros.review_evidence import build_task_acceptance_evidence
    from ouroboros.triad_review import ACCEPTANCE_SURFACE_RULES

    ctx = SimpleNamespace(task_metadata={}, task_contract={}, drive_root=str(tmp_path))
    ev = build_task_acceptance_evidence(
        ctx, llm_trace={"tool_calls": []}, drive_root=None, task_id="t",
        canonical_subject="Moscow",
    )
    assert "retrieval" not in ev
    assert "retrieval" not in ev["__provenance__"]
    assert "FACTUAL CONTEXT, not a criterion" in ACCEPTANCE_SURFACE_RULES
    assert "must NOT be treated as a gap" in ACCEPTANCE_SURFACE_RULES

    # Same evidence -> same panel verdict as today (offline stub panel).
    import ouroboros.review_substrate as rs

    class _Panel:
        seen: list = []

        def chat(self, **kwargs):
            _Panel.seen.append(kwargs)
            return {"content": json.dumps({
                "verdict": "PASS", "outcome_tier": "solved", "findings": [],
                "criteria_used": [{"criterion": "answer is correct", "status": "supported",
                                   "evidence_refs": ["canonical_payload"]}],
                "summary": "PASS",
            })}, {}

    panel = rs.run_review_request(
        rs.ReviewRequest(
            surface="task_acceptance", goal="capital of Russia",
            policy={"min_successful_slots": 1, "classify_outcome_tier": True,
                    "require_criterion_evidence": True},
            task_id="root", evidence=ev,
        ),
        slots=[rs.ReviewSlot(slot_id="s1", model="stub")],
        drive_root=tmp_path,
        llm=_Panel(),
    )
    assert panel.aggregate_signal == "PASS"
    assert rs.task_acceptance_is_clean(panel) is True


def test_retrieval_fact_is_never_shown_to_the_agent(tmp_path):
    """Reviewer-side only: the agent receives the improvement capsule, which carries
    no evidence sections at all — so the retrieval fact cannot leak back to it."""
    from ouroboros.review_substrate import build_improvement_capsule

    result = _result(**_ACTIONABLE_FAIL)
    result.request = {
        "surface": "task_acceptance",
        "policy": {"min_successful_slots": 1},
        "evidence": {"retrieval": {"urls": ["https://secret.example/leak"]}},
    }
    capsule = build_improvement_capsule(result, rails_line="rails")
    assert capsule
    assert "secret.example" not in capsule
    assert "retrieval" not in capsule.lower()


def test_agent_called_evidence_tool_never_builds_the_retrieval_section(tmp_path):
    """The other evidence caller (`tools/review.py`, the agent-facing tool) passes NO
    `llm_trace`, so the retrieval fact is never built there — and its reply carries
    only section NAMES plus the agent's own section, never host evidence values."""
    from ouroboros.review_evidence import build_task_acceptance_evidence

    ctx = SimpleNamespace(task_metadata={}, task_contract={}, drive_root=str(tmp_path))
    ev = build_task_acceptance_evidence(
        ctx, drive_root=None, task_id="t", canonical_subject="answer",
        agent_evidence={"claim": "done"},
    )
    assert "retrieval" not in ev
    assert "retrieval" not in sorted(k for k in ev if k != "__provenance__")


# ---------------------------------------------------------------------------
# frozen-ABI guard


def test_canonical_statuses_reach_the_review_projection_without_new_required_keys():
    """The three-state strings ride the frozen `review_projection` surface; only the
    additive optional `reason` key was introduced."""
    from ouroboros.outcomes import _review_axis

    axis = _review_axis({
        "acceptance_decision": {
            "status": ACCEPTANCE_FINALIZED_UNACCEPTED, "reason": "dialogue_terminal",
            "source": "task_acceptance_review", "rationale": "terminal",
        },
        "review_decision": {"eligibility": "eligible", "trigger": "auto_nondirect"},
        "review_runs": [{"authority": "host_root", "aggregate_signal": "FAIL", "actors": []}],
    })
    assert set(axis["acceptance_decision"]) == {
        "status", "reason", "source", "rationale", "agent_disposition", "agent_rationale",
    }


def test_deadline_reserve_writer_and_reader_move_together(monkeypatch, tmp_path):
    """End-to-end pairing of writer A16 and reader G1 through the real skip path."""
    import ouroboros.review_substrate as rs

    monkeypatch.setattr(loop_mod, "get_task_review_mode", lambda: "required")
    monkeypatch.setattr(loop_mod, "get_review_enforcement", lambda: "blocking")
    monkeypatch.setattr(rs, "triad_delivery_slots", lambda **_k: [object()])
    monkeypatch.setenv("OUROBOROS_FINALIZATION_GRACE_SEC", "120")
    now = datetime.now(timezone.utc)
    ctx = SimpleNamespace(
        _task_acceptance_reviewed=False, is_direct_chat=False, drive_root=str(tmp_path),
        task_metadata={
            "created_at": (now - timedelta(seconds=940)).isoformat(),
            "deadline_at": (now + timedelta(seconds=60)).isoformat(),
        },
        task_contract={},
    )
    trace = {"tool_calls": [{"tool": "write_file", "args": {"path": "x.py"}}]}
    assert loop_mod._run_task_acceptance_review_once(
        tools=SimpleNamespace(_ctx=ctx), content="done", task_id="t", task_type="task",
        llm_trace=trace, drive_root=None,
        messages=[{"role": "user", "content": "goal"}], emit_progress=lambda _m, *, incident=None: None,
    ) is False
    decision = trace["acceptance_decision"]
    assert decision["status"] == ACCEPTANCE_FINALIZED_UNACCEPTED
    assert decision["reason"] == REASON_ACCEPTANCE_REVIEW_SKIPPED_DEADLINE_RESERVE
    axes = derive_loop_outcome("FINAL ANSWER: x", {}, trace)["outcome_axes"]
    assert axes["execution"]["reason_code"] == REASON_ACCEPTANCE_REVIEW_SKIPPED_DEADLINE_RESERVE
    assert axes["objective"]["source"] == "task_acceptance_deadline_reserve"


def test_merge_point_is_the_only_status_writer_outside_the_agent_stance_merge():
    """P4.1: only `_set_acceptance_decision` and the agent STANCE merge assign
    `acceptance_decision`; no other module writes the host status."""
    import pathlib

    root = pathlib.Path(loop_mod.__file__).parent
    writers = sorted(
        path.name for path in root.rglob("*.py")
        if 'llm_trace["acceptance_decision"] =' in path.read_text(encoding="utf-8")
        or '["acceptance_decision"] = _dec' in path.read_text(encoding="utf-8")
    )
    # The v7 L-B split moved `_set_acceptance_decision` (and with it the single
    # host-status assignment) into the loop_acceptance leaf; the loop.py name is
    # a facade re-export of the same object.
    assert writers == ["loop_acceptance.py", "loop_tool_execution.py"], writers
    trace: dict = {}
    _set_acceptance_decision(trace, {"status": ACCEPTANCE_ACCEPTED, "reason": "clean_pass"})
    assert trace["acceptance_decision"]["status"] == ACCEPTANCE_ACCEPTED


def test_retrieval_urls_are_bounded_with_a_disclosed_count_and_a_full_set_hash(tmp_path):
    """Round-2 CRITICAL 2: the acceptance rules call `retrieval.urls` "the URLs it
    fetched", so silently dropping URLs past the cap (and silently shortening each one)
    misdescribes host-attested evidence. The bound stays; the silence does not."""
    from ouroboros.loop_llm_call import fold_retrieval_usage
    from ouroboros.review_evidence import build_task_acceptance_evidence

    accumulated: dict = {}
    fold_retrieval_usage(accumulated, {
        "web_search_sources": [{"url": f"https://e.example/{i}"} for i in range(33)],
    })
    record = accumulated["retrieval"]
    assert len(record["urls"]) == 20 and record["urls_omitted"] == 13

    ctx = SimpleNamespace(task_metadata={}, task_contract={}, drive_root=str(tmp_path))
    ev = build_task_acceptance_evidence(
        ctx, llm_trace={"tool_calls": [], "retrieval": record},
        drive_root=None, task_id="t", canonical_subject="answer",
    )
    assert len(ev["retrieval"]["urls"]) == 20
    assert ev["retrieval"]["urls_omitted"] == 13
    assert ev["retrieval"]["urls_identity_sha256"] == record["urls_identity_sha256"]
    assert ev["__provenance__"]["retrieval"] == "host_attested"

    # The full-set hash is order-sensitive and covers the URLs the capped list dropped:
    # a different fetched set cannot share it.
    other: dict = {}
    fold_retrieval_usage(other, {
        "web_search_sources": [{"url": f"https://e.example/{i}"} for i in range(32)],
    })
    assert other["retrieval"]["urls"] == record["urls"]  # identical bounded evidence...
    assert other["retrieval"]["urls_identity_sha256"] != record["urls_identity_sha256"]


def test_two_distinct_urls_sharing_a_rendered_prefix_are_never_silently_collapsed():
    """Round-3 BIBLE P1: `fold_retrieval_usage` deduplicated AFTER rendering. Two
    DISTINCT long URLs with the same retained prefix and the same length render
    identically, so the second was dropped while `urls_omitted` still said 0 — a
    fetched URL lost from evidence that promises an EXACT omission count. Dedup is on
    the full RAW url, so every distinct URL is either carried or counted."""
    from ouroboros.loop_llm_call import fold_retrieval_usage

    stem = "https://f.example/" + "q" * 400
    first, second = stem + "/alpha", stem + "/omega"   # same prefix, equal length
    assert len(first) == len(second) and first != second

    acc: dict = {}
    fold_retrieval_usage(acc, {"web_search_sources": [{"url": first}, {"url": second}]})
    record = acc["retrieval"]
    # The two render byte-identically — which is exactly what made the loss silent.
    assert record["source_count"] == 2
    assert len(set(record["urls"])) == 1

    # The invariant: 2 distinct fetched URLs, fully represented OR exactly accounted for.
    assert len(record["urls"]) + record["urls_omitted"] == 2
    assert (len(record["urls"]) == 2 and record["urls_omitted"] == 0) or (
        len(record["urls"]) == 1 and record["urls_omitted"] == 1
    )

    # A genuine REPEAT of the same raw url is still a repeat, not a phantom second fetch.
    again: dict = {}
    fold_retrieval_usage(again, {"web_search_sources": [{"url": first}, {"url": first}]})
    assert len(again["retrieval"]["urls"]) == 1
    assert again["retrieval"]["urls_omitted"] == 0

    # The rolling chain hash already covers the RAW values, so it separates the two sets
    # the rendered list cannot: it must not inherit the same collision.
    assert record["urls_identity_sha256"] != again["retrieval"]["urls_identity_sha256"]
