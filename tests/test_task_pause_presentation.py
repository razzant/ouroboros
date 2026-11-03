"""Pause causes remain display facts, independent of the shared parked phase."""
import pytest

from ouroboros.gateway.state import _activity_pause_cause


@pytest.mark.parametrize("reason", ["budget", "sleep"])
def test_exact_pause_cause_comes_from_marker(reason):
    assert _activity_pause_cause({"_budget_pause": {"reason": reason}}, {}) == reason


def test_owner_tree_and_restart_hold_do_not_inherit_a_budget_label():
    fence = {"cause": "owner_pause", "status": "active"}
    assert _activity_pause_cause({"_budget_pause": {"reason": "budget"}}, fence) == "owner"
    assert _activity_pause_cause({"_budget_pause_hold": {"reason": "owner_restart_hold"}}, {}) == "restart"
    # #1563: work saved before the application stopped waits under the same published cause.
    assert _activity_pause_cause({"_budget_pause_hold": {"reason": "saved_work_hold"}}, {}) == "restart"
    # Once the owner's Resume released (selected) that hold, a later owner Pause names itself.
    for reason in ("owner_restart_hold", "saved_work_hold"):
        released = {"_resume_selection": {"selected_by": "owner"}}
        assert _activity_pause_cause(released, fence) == "owner"
    assert _activity_pause_cause({}, {}) == "unknown"
    # Terminal history is not a current stop; an orphaned owner marker is Q9 selection.
    assert _activity_pause_cause({"reason_code": "budget_exhausted"}, {}) == "unknown"
    assert _activity_pause_cause({"_budget_pause": {"reason": "owner"}}, {}) == "unknown"


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
    assert row["pause_cause"] == ("unknown" if cause == "owner" else cause)
    if cause == "owner":
        assert row["stop"]["cause"] == "awaiting_selection"


@pytest.mark.parametrize("synthesis,named", [("running", True), ("pending_once", True), ("refresh", False), ("", False)])
def test_only_a_stored_answer_names_its_late_work(tmp_path, synthesis, named):
    """Activity names an answered root's remaining work; a working turn whose checkpoint holds
    only the pre-answer cost "refresh" pseudo-status is not answered and is never so labelled."""
    from ouroboros.gateway.state import _task_activity_facts
    from ouroboros.task_results import write_task_result

    write_task_result(tmp_path, "root", "running" if synthesis in {"refresh", ""} else "completed",
                      suggested_name="Audit the release", chat_id=1,
                      **({"root_phase_checkpoint": {"post_task_synthesis": synthesis}} if synthesis else {}))
    display = _task_activity_facts(tmp_path, "root")["display"]
    assert ("late_work" in display) is named
    if named:
        assert display["late_work"]["title"] == "Audit the release"


def test_late_work_dates_the_answer_only_by_the_executor_terminal_fact(tmp_path):
    from ouroboros.gateway.state import _task_activity_facts
    from ouroboros.task_results import write_task_result
    from ouroboros.terminal_time import terminal_time_fact

    running = {"root_phase_checkpoint": {"post_task_synthesis": "running"}}
    observed = write_task_result(tmp_path, "observed", "completed", _terminal_observed=True,
                                 suggested_name="Audit the release", **running)
    occurred = terminal_time_fact(observed)["occurred_at"]
    assert occurred, "the executor observed this terminal"
    assert _task_activity_facts(tmp_path, "observed")["display"]["late_work"] == {
        "title": "Audit the release", "answered_at": occurred}
    # A record without the executor's own observation keeps an unknown time, never its row ts.
    write_task_result(tmp_path, "legacy", "completed", suggested_name="Legacy", **running)
    assert _task_activity_facts(tmp_path, "legacy")["display"]["late_work"]["answered_at"] == ""
