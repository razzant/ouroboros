"""C4 (BUG 3 Layer A): feed the CLOSED/DROPPED set + active objective into the chooser prompt."""
import json

from ouroboros.post_task_evolution import _DECISION_PROMPT, _closed_objectives_digest


def test_decision_prompt_has_closed_and_active_sections_and_formats():
    rendered = _DECISION_PROMPT.format(
        reflection="r", backlog="b", capability="c",
        closed="- [NO_OP] X", active_objective="Y", force_note="",
    )
    assert "[CLOSED / DROPPED" in rendered
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
