"""A-material acceptance convergence (owner ratification 2026-08-30).

The failure this pins: on unlimited review cycles ONE task ran 21 paid acceptance
panels. Two pumps drove it — a reducer where any single `continue_actionable`
vote (and every missing/invalid vote) kept the loop, and a paid identity that
included the evidence revision, which any cosmetic tool call moves. The contract
here: a NEW paid panel is admissible only when the candidate answer changed or a
new nonempty obligation disposition appeared; everything else replays the
recorded verdict for free and terminalizes `identical_acceptance_refused`.

Offline and deterministic: no reviewer panel, no provider, no queue.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import ouroboros.loop as loop_mod
from ouroboros.loop_acceptance_review import (
    _apply_task_acceptance_result,
    acceptance_paid_identity,
    bind_acceptance_paid_identity,
)
from ouroboros.contracts.task_contract import build_task_contract
from ouroboros.outcomes import (
    ACCEPTANCE_FINALIZED_UNACCEPTED,
    ACCEPTANCE_REVISION_REQUESTED,
    OBJECTIVE_FAIL,
    OUTCOME_TIER_BLOCKED,
    REASON_IDENTICAL_ACCEPTANCE_REFUSED,
    derive_loop_outcome,
)
from ouroboros.review_evidence import (
    UNHASHED_ACCEPTANCE_DIALOGUE_HISTORY_KEY,
    task_acceptance_evidence_revision,
)
from ouroboros.review_substrate import (
    DIALOGUE_CONTINUE,
    DIALOGUE_UNREACHABLE,
    ReviewRunResult,
    aggregate_dialogue_status,
    review_binding_hash,
)
from ouroboros.task_results import (
    STATUS_RUNNING,
    claim_task_acceptance_review_cycle,
    load_task_acceptance_review_state,
    write_task_result,
)


# ---------------------------------------------------------------------------
# helpers


def _actor(slot_id, signal, parsed):
    return {"slot_id": slot_id, "signal": signal, "parsed": parsed}


def _fail_result(*, dialogue_status="continue_actionable", coach="tighten the parser",
                 findings=None):
    parsed = {"verdict": "FAIL", "outcome_tier": "best_effort"}
    if coach:
        parsed["completion_coach"] = coach
    if dialogue_status:
        parsed["dialogue_status"] = dialogue_status
    if findings is not None:
        parsed["findings"] = findings
    return ReviewRunResult(
        request={"surface": "task_acceptance", "policy": {"min_successful_slots": 1}},
        actors=[_actor("s0", "FAIL", parsed)],
        parsed_findings=list(findings or []),
        aggregate_signal="FAIL",
    )


def _ctx(tmp_path, *, trace=None, mode="required", passes_done=0, messages=None):
    tool_ctx = SimpleNamespace(
        _task_acceptance_reviewed=False,
        _task_acceptance_improvement_passes=passes_done,
        _task_acceptance_seen_bindings={},
        drive_root=str(tmp_path),
        task_metadata={},
        task_contract={},
        is_direct_chat=False,
    )
    return loop_mod._TaskAcceptanceContext(
        tools=SimpleNamespace(_ctx=tool_ctx),
        content="the deliverable",
        task_id="t",
        task_type="task",
        llm_trace=trace if trace is not None else {"tool_calls": []},
        drive_root=None,
        messages=messages if messages is not None else [{"role": "user", "content": "goal"}],
        emit_progress=lambda _m, *, incident=None: None,
        mode=mode,
        subtree_statuses=[],
        budget_profile={"max_improvement_passes": 99},
        passes_done=passes_done,
    )


def _no_fence(monkeypatch):
    monkeypatch.setattr(loop_mod, "_end_task_acceptance_fence", lambda *_a, **_k: True)
    monkeypatch.setattr(loop_mod, "_mark_root_acceptance_checkpoint", lambda *_a, **_k: None)
    monkeypatch.setattr(loop_mod, "get_review_enforcement", lambda: "blocking")


def _binding(candidate_hash: str, evidence_revision: str) -> dict:
    fields = {
        "candidate_hash": candidate_hash,
        "evidence_revision": evidence_revision,
        "fence_hash": "f" * 64,
    }
    return {**fields, "binding_hash": review_binding_hash(**fields)}


# ---------------------------------------------------------------------------
# (a) the 21-panel scenario converges


def test_persistently_optimistic_reviewer_without_material_cannot_buy_a_second_panel(
    monkeypatch, tmp_path,
):
    """The reproduction, reduced: unlimited cycles, one reviewer that keeps voting
    `continue_actionable` with a coach, and an agent that changes NOTHING. Panel 1
    is paid and asks for a revision; the resubmit carries the same paid identity,
    so panel 2 is a free replay that terminalizes instead of pumping."""
    monkeypatch.setenv("OUROBOROS_REVIEW_MAX_CYCLES", "unlimited")
    _no_fence(monkeypatch)
    trace: dict = {"tool_calls": [], "review_runs": []}
    messages = [{"role": "user", "content": "goal"}]
    ctx = _ctx(tmp_path, trace=trace, messages=messages)
    result = _fail_result()

    # Panel 1: a well-formed continue keeps the loop and feeds the capsule.
    assert _apply_task_acceptance_result(ctx, result, record_run=False) is True
    assert trace["acceptance_decision"]["status"] == ACCEPTANCE_REVISION_REQUESTED
    messages_after_panel_1 = list(messages)

    # The agent resubmits with no changed answer and no new disposition: the paid
    # identity is unchanged, so the recorded panel replays for free.
    identity_before = acceptance_paid_identity("cand", trace)
    identity_after = acceptance_paid_identity("cand", trace)
    assert identity_before == identity_after

    ctx.tools._ctx._task_acceptance_reviewed = False
    assert _apply_task_acceptance_result(
        ctx, result, record_run=False, reused=True,
    ) is False
    decision = trace["acceptance_decision"]
    assert decision["status"] == ACCEPTANCE_FINALIZED_UNACCEPTED
    assert decision["reason"] == REASON_IDENTICAL_ACCEPTANCE_REFUSED
    assert "FAIL" in decision["rationale"]  # the recorded verdict is quoted
    # The capsule was NOT re-entered: no second improvement note reached the agent.
    assert messages == messages_after_panel_1
    assert ctx.tools._ctx._task_acceptance_reviewed is True


def test_identical_refusal_terminal_is_a_blocked_objective_not_a_green_one():
    """Value-keyed reader sweep: the refusal rides the SAME status+reason key as
    `review_cycles_exhausted`, so the last panel's proposed tier cannot render as
    a solved-clean objective."""
    trace = {
        "acceptance_decision": {
            "status": ACCEPTANCE_FINALIZED_UNACCEPTED,
            "reason": REASON_IDENTICAL_ACCEPTANCE_REFUSED,
            "source": "task_acceptance_review",
        },
        "review_runs": [{
            "authority": "host_root",
            "request": {"surface": "task_acceptance"},
            "aggregate_signal": "PASS",
            "actors": [_actor("s0", "PASS", {"verdict": "PASS", "outcome_tier": "solved"})],
        }],
        "review_decision": {"eligibility": "eligible", "trigger": "root"},
    }
    objective = derive_loop_outcome("FINAL ANSWER: x", {}, trace)["outcome_axes"]["objective"]
    assert objective["status"] == OBJECTIVE_FAIL
    assert objective["outcome_tier"] == OUTCOME_TIER_BLOCKED
    assert objective["reason"] == REASON_IDENTICAL_ACCEPTANCE_REFUSED


# ---------------------------------------------------------------------------
# (b) one strong reviewer WITH material still holds the loop open


def test_single_reviewer_with_a_new_finding_keeps_the_loop_and_new_candidate_pays(
    monkeypatch, tmp_path,
):
    _no_fence(monkeypatch)
    finding = {"slot_id": "s0", "severity": "critical", "item": "unhandled EOF",
               "recommendation": "close the reader"}
    result = _fail_result(findings=[finding], coach="")
    assert aggregate_dialogue_status(result, quorum=2)["status"] == DIALOGUE_CONTINUE
    trace: dict = {"tool_calls": [], "review_runs": []}
    ctx = _ctx(tmp_path, trace=trace)
    assert _apply_task_acceptance_result(ctx, result, record_run=False) is True
    assert trace["acceptance_decision"]["reason"] == "improvement_capsule"
    # After the revision the candidate hash moves, so the next panel is admissible.
    assert acceptance_paid_identity("cand-v1", trace) != acceptance_paid_identity("cand-v2", trace)


# ---------------------------------------------------------------------------
# (c) a junk continue abstains and one terminal vote ends the dialogue


def test_junk_continue_abstains_so_a_single_terminal_vote_terminalizes(
    monkeypatch, tmp_path,
):
    result = ReviewRunResult(
        request={"surface": "task_acceptance", "policy": {"min_successful_slots": 2}},
        actors=[
            _actor("s0", "FAIL", {"verdict": "FAIL", "outcome_tier": "best_effort",
                                  "dialogue_status": "continue_actionable"}),
            _actor("s1", "FAIL", {"verdict": "FAIL", "outcome_tier": "best_effort",
                                  "completion_coach": "stop here",
                                  "dialogue_status": "unreachable_here"}),
        ],
        parsed_findings=[],
        aggregate_signal="FAIL",
    )
    assert aggregate_dialogue_status(result, quorum=2)["status"] == DIALOGUE_UNREACHABLE
    _no_fence(monkeypatch)
    trace: dict = {"tool_calls": [], "review_runs": []}
    ctx = _ctx(tmp_path, trace=trace)
    assert _apply_task_acceptance_result(ctx, result, record_run=False) is False
    assert trace["acceptance_decision"]["reason"] == "dialogue_terminal"


def test_degraded_panel_lone_terminal_vote_does_not_shadow_the_degraded_causes(
    monkeypatch, tmp_path,
):
    """MAJOR 6: a DEGRADED panel (no valid verdict quorum) cannot 'judge' the
    dialogue — a lone terminal vote from the one contributing slot must fall to
    review_degraded, which is the only surface carrying per-slot causes and
    degraded_reasons (the v6.70.0 honesty invariant, P1)."""
    result = ReviewRunResult(
        request={"surface": "task_acceptance", "policy": {"min_successful_slots": 2}},
        actors=[
            {"slot_id": "s0", "parse_status": "malformed", "parsed": None},
            {"slot_id": "s1", "parse_status": "malformed", "parsed": None},
            _actor("s2", "FAIL", {"verdict": "FAIL", "outcome_tier": "best_effort",
                                  "completion_coach": "n/a",
                                  "dialogue_status": "unreachable_here"}),
        ],
        parsed_findings=[],
        aggregate_signal="DEGRADED",
    )
    setattr(result, "degraded_reasons", ["slot_0 transport_failed: 502 from provider"])
    _no_fence(monkeypatch)
    trace: dict = {"tool_calls": [], "review_runs": []}
    ctx = _ctx(tmp_path, trace=trace)
    assert _apply_task_acceptance_result(ctx, result, record_run=False) is False
    decision = trace["acceptance_decision"]
    assert decision["reason"] == "review_degraded"
    assert decision["degraded_reasons"] == ["slot_0 transport_failed: 502 from provider"]


# ---------------------------------------------------------------------------
# (d) a new nonempty rebuttal buys exactly ONE paid panel; an empty one buys none


def _obligation(reason: str, disposition: str = "rejected") -> dict:
    return {
        "id": "ob-1", "item": "broken", "recommendation": "fix",
        "status": "agent_disposed", "disposition": disposition,
        "disposition_reason": reason,
    }


def test_new_rebuttal_buys_one_paid_panel_and_an_empty_reason_buys_nothing(tmp_path):
    write_task_result(
        tmp_path, "root-a", STATUS_RUNNING, root_task_id="root-a",
        task_contract=build_task_contract({}),
    )
    trace: dict = {"acceptance_obligations": []}
    cand = "c" * 64

    # No disposition at all, then a disposition with an EMPTY reason: the identity
    # must not move — an empty rebuttal is not an argument (commit-gate parity).
    first = acceptance_paid_identity(cand, trace)
    trace["acceptance_obligations"] = [_obligation("")]
    assert acceptance_paid_identity(cand, trace) == first

    # A nonempty reason IS new material.
    trace["acceptance_obligations"] = [_obligation("the API contract forbids it")]
    second = acceptance_paid_identity(cand, trace)
    assert second != first

    def _claim(identity, evidence_revision):
        binding = _binding(cand, evidence_revision)
        binding["paid_identity"] = identity
        return claim_task_acceptance_review_cycle(
            tmp_path, "root-a", binding, claimed_by_task_id="root-a",
        )

    assert _claim(first, "a" * 64)["status"] == "claimed"
    # A cosmetic tool call moved the evidence revision -> a brand-new binding hash
    # -> under the OLD rule this bought a paid panel. It must not.
    moved = _claim(first, "b" * 64)
    assert (moved["status"], moved["reason"]) == ("unknown", "binding_dispatch_already_claimed")
    # The new rebuttal buys exactly one.
    assert _claim(second, "c" * 64)["status"] == "claimed"
    assert _claim(second, "d" * 64)["reason"] == "binding_dispatch_already_claimed"
    state = load_task_acceptance_review_state(tmp_path, "root-a")
    assert len(state["claims_by_binding"]) == 2


def test_reused_lookup_accepts_either_binding_or_paid_identity(tmp_path):
    """`_prior_acceptance_run` compat: byte-identical binding replays as before,
    AND an unchanged paid identity under a moved binding replays too."""
    identity = "e" * 64
    prior = {"authority": "host_root", "binding_hash": "1" * 64, "paid_identity": identity,
             "aggregate_signal": "FAIL"}
    trace = {"review_runs": [prior]}
    tool_ctx = SimpleNamespace(_task_acceptance_seen_bindings={})
    _seen, found = loop_mod._prior_acceptance_run(tool_ctx, trace, "1" * 64)
    assert found is prior
    _seen, found = loop_mod._prior_acceptance_run(
        tool_ctx, trace, "9" * 64, paid_identity=identity,
    )
    assert found is prior
    _seen, found = loop_mod._prior_acceptance_run(
        tool_ctx, trace, "9" * 64, paid_identity="f" * 64,
    )
    assert found is None


def test_bind_paid_identity_stamps_the_binding_without_touching_its_hashes():
    binding = _binding("a" * 64, "b" * 64)
    before = dict(binding)
    identity = bind_acceptance_paid_identity(binding, {"acceptance_obligations": []})
    assert binding["paid_identity"] == identity
    assert {k: binding[k] for k in before} == before


# ---------------------------------------------------------------------------
# (e) reading the dialogue history must not re-price the packet


def test_dialogue_history_does_not_change_the_evidence_revision():
    """The pin for task 6: the history is reviewer-VISIBLE but outside the hashed
    material, so growing it can never mint a fresh evidence revision — and
    therefore never a fresh paid binding."""
    packet = {"task_type": "task", "verification_summary": "ok"}
    with_history = {
        **packet,
        UNHASHED_ACCEPTANCE_DIALOGUE_HISTORY_KEY: [
            {"round": 1, "aggregate_signal": "FAIL", "dialogue_status": "continue_actionable"},
        ],
    }
    more_history = {
        **packet,
        UNHASHED_ACCEPTANCE_DIALOGUE_HISTORY_KEY: [
            {"round": 1, "aggregate_signal": "FAIL", "dialogue_status": "continue_actionable"},
            {"round": 2, "aggregate_signal": "FAIL", "dialogue_status": "unreachable_here"},
        ],
    }
    revision = task_acceptance_evidence_revision(packet)
    assert task_acceptance_evidence_revision(with_history) == revision
    assert task_acceptance_evidence_revision(more_history) == revision
    # Any OTHER key still re-prices, so this is an exclusion and not a hole.
    assert task_acceptance_evidence_revision({**packet, "extra": 1}) != revision


def test_dialogue_history_rows_are_bounded_and_carry_the_panel_facts():
    from ouroboros.loop_acceptance_review import acceptance_dialogue_history

    trace = {
        "review_runs": [
            {"authority": "host_root", "aggregate_signal": "FAIL",
             "dialogue": {"status": "continue_actionable", "votes": {"continue_actionable": ["s0"]}}},
            {"authority": "agent_advisory", "aggregate_signal": "PASS"},
            {"authority": "host_root", "aggregate_signal": "FAIL",
             "dialogue": {"status": "unreachable_here", "votes": {"unreachable_here": ["s0", "s1"]}}},
        ],
        "acceptance_obligations": [
            {"id": "ob-1", "reopened_count": 2}, {"id": "ob-2"},
        ],
    }
    rows = acceptance_dialogue_history(trace, limit=2)
    assert [row["round"] for row in rows] == [1, 2]  # advisory runs never count
    assert rows[1]["dialogue_status"] == "unreachable_here"
    assert rows[1]["votes"] == {"unreachable_here": 2}
    assert rows[1]["obligations_new"] == 1 and rows[1]["obligations_re_raised"] == 1
    assert len(acceptance_dialogue_history(trace, limit=1)) == 1
    # The rows must survive the JSON round-trip the evidence packet performs.
    assert json.loads(json.dumps(rows)) == rows


def test_reviewer_rebuttal_response_reaches_the_next_panels_obligation_catalog():
    from ouroboros.review_evidence import _accept_obligation_row

    row = _accept_obligation_row({
        "id": "ob-1", "item": "broken", "recommendation": "fix", "status": "open",
        "reopened_count": 1, "previous_disposition": "rejected",
        "previous_reason": "the contract forbids it",
        "reviewer_rebuttal_response": "the contract was amended in v2; " + "x" * 900,
    })
    assert row["previous_agent_disposition"] == "rejected"
    assert row["previous_reviewer_response"].startswith("the contract was amended in v2")
    # Bounded at 600 chars with the cognitive-artifact omission note appended, the
    # same shape `previous_agent_reason` already ships (never a silent [:N] clip).
    assert "OMISSION NOTE: truncated at 600 chars" in row["previous_reviewer_response"]


# ---------------------------------------------------------------------------
# (f) the advisory lane is untouched


def test_advisory_lane_collects_no_obligations_and_gains_no_counters(monkeypatch, tmp_path):
    """Obligations are a Required+Blocking construct. On the advisory lane the new
    branches must add nothing: no obligation rows, no new trace keys, and the
    ordinary capsule path still runs."""
    _no_fence(monkeypatch)
    monkeypatch.setattr(loop_mod, "get_review_enforcement", lambda: "advisory")
    trace: dict = {"tool_calls": [], "review_runs": []}
    ctx = _ctx(tmp_path, trace=trace, mode="auto")
    assert _apply_task_acceptance_result(ctx, _fail_result(), record_run=False) is True
    assert "acceptance_obligations" not in trace
    assert trace["acceptance_decision"]["reason"] == "improvement_capsule"
    assert set(trace) == {"tool_calls", "review_runs", "acceptance_decision"}
    # And the identical-resubmit refusal still terminalizes on this lane.
    ctx.tools._ctx._task_acceptance_reviewed = False
    assert _apply_task_acceptance_result(
        ctx, _fail_result(), record_run=False, reused=True,
    ) is False
    assert trace["acceptance_decision"]["reason"] == REASON_IDENTICAL_ACCEPTANCE_REFUSED
    assert trace["acceptance_decision"]["open_obligations"] == []


# ---------------------------------------------------------------------------
# (g) restart durability


def test_paid_identity_claims_survive_a_reload_of_the_ledger(tmp_path):
    """The claim ledger is the root task result on disk: a restarted process must
    still refuse a paid dispatch for material the tree already bought."""
    write_task_result(
        tmp_path, "root-b", STATUS_RUNNING, root_task_id="root-b",
        task_contract=build_task_contract({}),
    )
    identity = "1" * 64
    binding = {**_binding("a" * 64, "b" * 64), "paid_identity": identity}
    assert claim_task_acceptance_review_cycle(
        tmp_path, "root-b", binding, claimed_by_task_id="root-b",
    )["status"] == "claimed"

    reloaded = load_task_acceptance_review_state(tmp_path, "root-b")
    assert [row["paid_identity"] for row in reloaded["claims_by_binding"].values()] == [identity]
    # Same material, different binding, after the reload: still refused.
    moved = {**_binding("a" * 64, "c" * 64), "paid_identity": identity}
    again = claim_task_acceptance_review_cycle(
        tmp_path, "root-b", moved, claimed_by_task_id="root-b",
    )
    assert (again["status"], again["reason"]) == ("unknown", "binding_dispatch_already_claimed")

    # And the free-refusal projection agrees before any dispatch is attempted.
    from ouroboros.task_results import project_task_acceptance_review_capacity

    ctx = SimpleNamespace(
        task_id="root-b", root_task_id="root-b", task_metadata={},
        drive_root=str(tmp_path), budget_drive_root=str(tmp_path),
    )
    projection = project_task_acceptance_review_capacity(
        ctx, binding_hash=moved["binding_hash"], task_id="root-b", paid_identity=identity,
    )
    assert projection["binding_seen"] is True


def test_superseded_paid_identity_still_replays_the_identical_refusal(tmp_path):
    """sol M2: a panel paid for identity I, then superseded by an evidence
    revision, then resubmitted with the SAME identity must free-replay to the
    typed identical-refusal terminal — not fall through to the wallet's
    binding_dispatch_already_claimed synthetic DEGRADED."""
    from ouroboros.loop import _prior_acceptance_run
    from types import SimpleNamespace

    run = {
        "authority": "host_root", "binding_hash": "bh-old",
        "paid_identity": "pi-1", "panel_id": "p1",
        "aggregate_signal": "FAIL", "superseded_by_revision": True,
        "superseded_reason": "evidence_revision",
    }
    trace = {"review_runs": [run]}
    ctx = SimpleNamespace()
    # New binding hash (evidence moved) but the SAME paid identity.
    seen, prior = _prior_acceptance_run(ctx, trace, "bh-new", paid_identity="pi-1")
    assert prior is not None
    assert prior["paid_identity"] == "pi-1"
    assert prior["replayed_from_superseded"] is True
    # A fresh (non-superseded) run still wins over the superseded fallback.
    fresh = dict(run, superseded_by_revision=False, panel_id="p2", binding_hash="bh-mid")
    trace2 = {"review_runs": [run, fresh]}
    _, prior2 = _prior_acceptance_run(SimpleNamespace(), trace2, "bh-new", paid_identity="pi-1")
    assert prior2 is not None and prior2["panel_id"] == "p2"
    assert "replayed_from_superseded" not in prior2
    # A DIFFERENT identity buys nothing from the superseded run.
    _, prior3 = _prior_acceptance_run(SimpleNamespace(), trace, "bh-new", paid_identity="pi-2")
    assert prior3 is None


def test_superseded_clean_pass_replays_into_refusal_never_reaccepts(monkeypatch, tmp_path):
    """final-lane sol MAJOR: a clean-PASS panel superseded by an evidence
    revision must NOT re-authorize on an identical resubmission — the verdict
    predates the evidence change. The replay lands in the typed refusal
    terminal (conservative and consistent with the superseded trace rows)."""
    result = ReviewRunResult(
        request={"surface": "task_acceptance", "policy": {"min_successful_slots": 1}},
        actors=[_actor("s0", "PASS", {"verdict": "PASS", "outcome_tier": "solved",
                                      "dialogue_status": "unreachable_here"})],
        parsed_findings=[],
        aggregate_signal="PASS",
    )
    setattr(result, "replayed_from_superseded", True)
    _no_fence(monkeypatch)
    trace: dict = {"tool_calls": [], "review_runs": []}
    ctx = _ctx(tmp_path, trace=trace)
    assert _apply_task_acceptance_result(ctx, result, reused=True, record_run=False) is False
    decision = trace["acceptance_decision"]
    assert decision["reason"] == "identical_acceptance_refused"
