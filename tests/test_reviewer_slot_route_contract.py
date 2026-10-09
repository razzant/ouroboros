"""Exact review-pool route and effort-authority regressions (PR-3, one transition).

The pool's rows are catalog rows: a session row promises ONE concrete harness
route exactly as a lane row did, the «Выполняется как» projection keeps a wave's
declared effort apart from the row's own, and a compound session effort pins the
commit contract fingerprint against global effort drift.
"""

import pytest

from tests.review_pool_rosters import pool_roster, pool_seat


@pytest.mark.parametrize(("target", "fragment"), [
    # ``off`` passes the catalog's spelling check but names no harness (the lane
    # row's own rule, kept on the pool row); the malformed spellings are the
    # catalog parser's typed refusals before the pool ever sees them.
    ("off", "does not name a concrete harness route"),
    ("OFF", "does not name a concrete harness route"),
    ("=malformed", "session harness must match"),
    (":high", "legacy ':effort'"),
])
def test_a_pool_session_row_must_name_a_concrete_harness(target, fragment, monkeypatch):
    from ouroboros.reviewer_slot_config import review_pool_rows, review_pool_state

    roster = pool_roster(pool_seat("sess", target, kind="agent_session"),
                         pool_seat("api", "openai/gpt-5.6-sol"))
    monkeypatch.setenv("OUROBOROS_SUBAGENTS", roster)
    with pytest.raises(ValueError, match=fragment):
        review_pool_rows()
    state = review_pool_state(roster)
    assert state["state"] == "error" and fragment.replace("\\", "") in state["error"]


def test_last_execution_projection_keeps_a_declared_effort_apart_from_the_row(tmp_path, monkeypatch):
    """«Выполняется как» must not show the agent's one-off panel strength as the
    row's saved configuration: requested.effort is the ROW's effort ('' when the
    declaration filled it) and the declaration rides its own field."""
    from types import SimpleNamespace

    from ouroboros import reviewer_slot_config
    from ouroboros.review_substrate import ReviewSlot

    monkeypatch.setattr(reviewer_slot_config, "_last_execution_path", lambda: tmp_path / "last.json")
    slots = {
        "declared": ReviewSlot(slot_id="declared", model="m/a", effort="max", declared_effort="max"),
        "own": ReviewSlot(slot_id="own", model="m/b", effort="low"),
    }
    actors = [SimpleNamespace(slot_id=sid, status="ok", usage={}) for sid in slots]
    reviewer_slot_config.record_reviewer_slot_executions("plan_review", actors, slots)
    last = reviewer_slot_config.reviewer_slot_last_executions()
    assert last["declared"]["requested"]["effort"] == "" and last["declared"]["requested"]["declared_effort"] == "max"
    assert last["own"]["requested"]["effort"] == "low" and "declared_effort" not in last["own"]["requested"]


def test_compound_effort_stabilizes_commit_fingerprint_against_global_drift(
    monkeypatch,
):
    from ouroboros.tools.commit_gate import commit_review_contract_fingerprint
    from tests.review_pool_rosters import set_review_pool

    def pool(second: str) -> str:
        return pool_roster(pool_seat("grok-row", "cursor=cursor-grok-4.6-xhigh", kind="agent_session"),
                           pool_seat("agy-row", second, kind="agent_session"))

    # The pool's compound session rows carry their effort in the route slug: a
    # changed global effort moves nothing on them.
    set_review_pool(monkeypatch, pool("agy=gemini-3.1-pro-max-fast"))
    monkeypatch.setenv("OUROBOROS_EFFORT_REVIEW", "low")
    first = commit_review_contract_fingerprint()

    monkeypatch.setenv("OUROBOROS_EFFORT_REVIEW", "high")
    assert commit_review_contract_fingerprint() == first

    set_review_pool(monkeypatch, pool("agy=gemini-3.1-pro-xhigh-fast"))
    assert commit_review_contract_fingerprint() != first
