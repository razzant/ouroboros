"""C4 (BUG 3 Layer A): feed past attempt outcomes + the active objective into the chooser prompt.

Past attempts are evidence the chooser weighs, not a permanent ban; no sentence pushes it to
promote because a schedule fired, and it is not told to prefer the backlog."""
import json

from ouroboros.post_task_evolution import _DECISION_PROMPT, _closed_objectives_digest


def test_decision_prompt_has_closed_and_active_sections_and_formats():
    rendered = _DECISION_PROMPT.format(
        task_facts="f", backlog="b", capability="c",
        closed="- [NO_OP] X", active_objective="Y",
    )
    assert "[PAST ATTEMPTS" in rendered and "not a permanent verdict" in rendered
    assert "needs a reason its earlier outcome does not cover" in rendered
    assert "The cadence already decided" not in rendered and "Prefer items already in the backlog" not in rendered
    assert "CLOSED / DROPPED" not in rendered
    assert "[ACTIVE CAMPAIGN OBJECTIVE" in rendered
    assert "- [NO_OP] X" in rendered and "Y" in rendered
    # the JSON example's escaped braces must survive .format()
    assert '"promote": true|false' in rendered


def _write_ledger(tmp_path, rows):
    state = tmp_path / "state"
    state.mkdir(exist_ok=True)
    (state / "evolution_checkpoints.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows), encoding="utf-8")


def test_closed_digest_from_structured_ledger(tmp_path):
    _write_ledger(tmp_path, [
        {"task_id": "t1", "campaign_objective": "Add API key guard",
         "transaction": {"cycle_outcome": "waiting_for_restart"}},
        {"task_id": "t1", "kind": "cycle_outcome", "cycle_outcome": "no_op",
         "campaign_objective": "Add API key guard"},
        # blocked_with_evidence: execution ok, but the OBJECTIVE axis is blocked (the bug's signal
        # the capability digest is blind to)
        {"task_id": "t2", "campaign_objective": "Modify the evolution machinery",
         "outcome_axes": {"execution": {"status": "ok"},
                          "objective": {"outcome_tier": "blocked_with_evidence"}}},
        {"task_id": "t3", "kind": "cycle_outcome", "cycle_outcome": "absorbed",
         "campaign_objective": "Add CLI ref tool"},
    ])
    digest = _closed_objectives_digest(tmp_path)
    assert "Add API key guard" in digest                 # no_op -> dropped
    assert "Modify the evolution machinery" in digest     # blocked_with_evidence (structural source)
    assert "Add CLI ref tool" in digest                   # absorbed = already shipped
    assert "[BLOCKED]" in digest
    assert "[NO_OP]" in digest or "[ABANDONED]" in digest


def test_closed_digest_dedups_by_fingerprint(tmp_path):
    _write_ledger(tmp_path, [
        {"task_id": "a", "kind": "cycle_outcome", "cycle_outcome": "no_op",
         "campaign_objective": "Add API key guard"},
        {"task_id": "b", "kind": "cycle_outcome", "cycle_outcome": "no_op",
         "campaign_objective": "Add  API  key  guard"},  # whitespace-only variant
    ])
    assert _closed_objectives_digest(tmp_path).count("Add ") == 1  # one deduped line


def test_closed_digest_empty_when_no_ledger(tmp_path):
    assert _closed_objectives_digest(tmp_path) == ""


def test_review_cap_stop_is_not_permanent_closure_even_when_cycle_is_no_op(tmp_path):
    _write_ledger(tmp_path, [
        {"task_id": "old", "campaign_objective": "Retained correction",
         "transaction": {"cycle_outcome": "no_op"}},
        {"task_id": "current", "campaign_objective": "Retained correction",
         "transaction": {"cycle_outcome": "no_op", "cleanup_stash": "retained-current"},
         "outcome_axes": {"objective": {"outcome_tier": "blocked_with_evidence",
                                        "reason": "review_cycles_exhausted"}}},
        {"task_id": "unavailable", "campaign_objective": "Reviewer unavailable",
         "transaction": {"cycle_outcome": "no_op"},
         "outcome_axes": {"review": {"status": "degraded", "acceptance_decision": {
             "status": "finalized_unaccepted", "reason": "review_degraded"}}}},
        {"task_id": "plan", "campaign_objective": "Retained plan",
         "outcome_axes": {"objective": {"outcome_tier": "blocked_with_evidence",
                                        "reason": "plan_review_quorum_unreachable"}}},
    ])
    assert _closed_objectives_digest(tmp_path) == ""


def test_actual_absorption_or_abandonment_still_closes_a_review_blocked_cycle(tmp_path):
    _write_ledger(tmp_path, [
        {"task_id": "shipped", "campaign_objective": "Shipped objective",
         "outcome_axes": {"objective": {"reason": "review_cycles_exhausted"}}},
        {"task_id": "shipped", "kind": "cycle_outcome", "cycle_outcome": "absorbed"},
        {"task_id": "abandoned", "campaign_objective": "Abandoned objective",
         "outcome_axes": {"objective": {"reason": "review_cycles_exhausted"}}},
        {"task_id": "abandoned", "kind": "cycle_outcome", "cycle_outcome": "abandoned"},
    ])
    digest = _closed_objectives_digest(tmp_path)
    assert "Shipped objective" in digest
    assert "Abandoned objective" in digest


def test_a_running_result_review_holds_closure_only_when_it_seats_a_counted_reviewer(tmp_path):
    """A task whose own result review still runs is waiting, not closed (``_pending_result_review_tasks``). An
    advisers-only one (Cyber Pro, #3194: heard, never counted) holds nothing, so its no-op stays a past attempt."""
    from ouroboros import review_ledger as ledger
    from tests.test_review_ledger import _facts, _plan, _raw

    for task_id, additional in (("counted", False), ("adviser", True)):
        running = [_raw("seat", "model-seat", "pending", parts=("result",), operation_state="in_flight",
                        late_result_pending=True)]
        plan = [_plan("seat", "model-seat", parts=("result",), additional=additional)]
        record = ledger.build_wave_record(_facts(running, rows=plan, task_id=task_id, pending=True), surface="result")
        assert record.state == ledger.STATE_PENDING, "both records are still running"
        ledger.write_record(tmp_path, record)
    _write_ledger(tmp_path, [
        {"task_id": "counted", "kind": "cycle_outcome", "cycle_outcome": "no_op", "campaign_objective": "Counted review runs"},
        {"task_id": "adviser", "kind": "cycle_outcome", "cycle_outcome": "no_op", "campaign_objective": "Adviser alone runs"},
    ])
    digest = _closed_objectives_digest(tmp_path)
    assert "Counted review runs" not in digest and "[NO_OP] Adviser alone runs" in digest, digest
