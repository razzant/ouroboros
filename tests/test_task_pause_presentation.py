"""Pause causes remain display facts, independent of the shared parked phase."""
import pytest

from ouroboros.gateway.state import _activity_pause_cause


@pytest.mark.parametrize("reason", ["budget", "owner", "sleep"])
def test_exact_pause_cause_comes_from_marker(reason):
    assert _activity_pause_cause({"_budget_pause": {"reason": reason}}, {}) == reason


def test_owner_tree_and_restart_hold_do_not_inherit_a_budget_label():
    assert _activity_pause_cause({"_budget_pause": {"reason": "budget"}}, {"cause": "owner_pause"}) == "owner"
    assert _activity_pause_cause({"_budget_pause_hold": {"reason": "owner_restart_hold"}}, {}) == "restart"
    assert _activity_pause_cause({}, {}) == "unknown"
    assert _activity_pause_cause({"reason_code": "budget_exhausted"}, {}) == "budget"


@pytest.mark.serial
@pytest.mark.parametrize("cause", ["budget", "owner", "sleep"])
def test_census_carries_typed_pause_cause_for_a_parked_direct_root(tmp_path, monkeypatch, cause):
    from ouroboros.gateway.state import _chat_activities_snapshot_safe
    from ouroboros.task_results import write_task_result
    from supervisor import queue
    task = {"id": "paused", "root_task_id": "paused", "chat_id": 1, "_is_direct_chat": True,
            "_budget_pause": {"reason": cause, "status": "paused_exact_continuation"}}
    monkeypatch.setattr(queue, "PENDING", [task])
    monkeypatch.setattr(queue, "RUNNING", {})
    monkeypatch.setattr(queue, "BUDGET_ROOT_FENCES", {})
    write_task_result(tmp_path, "paused", "scheduled", budget_pause={"reason": cause, "state": "paused"})
    rows = _chat_activities_snapshot_safe(tmp_path, direct_turns=[])
    row = next(row for row in rows if row["activity_id"] == "paused")
    assert row["kind"] == "direct_chat" and row["phase"] == "budget_paused"
    assert row["pause_cause"] == cause
