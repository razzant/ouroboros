"""Activity keeps the current worker's warm sleep through the server's shared work projection."""
from __future__ import annotations

import time

import pytest

from tests._budget_pause_exact_helpers import _install_queue
from tests._work_projection_shared import state_fields


@pytest.mark.parametrize("change, activity", [
    ({}, "waiting"),
    ({"task_attempt": 1}, "working"),
    ({"state": "resumed"}, "working"),
    ({"reason": "owner_pause"}, "working"),
    ({"sleep": {"mode": "cold"}}, "working"),
    ({"sleep": None}, "working"),
])
def test_current_warm_sleep_reaches_work_without_claiming_an_owner_turn(tmp_path, monkeypatch, change, activity):
    from ouroboros.task_results import write_task_result

    _queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    task = {"id": "asleep", "type": "task", "root_task_id": "asleep", "chat_id": 1, "_attempt": 2}
    write_task_result(tmp_path, "asleep", "running", root_task_id="asleep", chat_id=1)
    workers.RUNNING["asleep"] = {
        "task": task, "attempt": 2, "started_at": time.time(),
        "owner_wait": {"state": "waiting", "reason": "sleep", "sleep": {"mode": "warm"},
                       "task_attempt": 2, **change},
    }
    fields = state_fields(tmp_path)
    [work] = fields["works"]
    assert work["activity"] == activity
    assert work["live"] is (activity == "working")
    assert work["turn"] == [] and work["release"] == "self"
    assert [reason["code"] for reason in work["because"]] == (["sleeping"] if activity == "waiting" else [])
    assert not any(control["action"] == "resume" for control in work["controls"])
    # The physical checkpoint stays internal; the census owner_wait still belongs to questions.
    assert "sleep" not in (fields["active_chat_activities"][0].get("owner_wait") or {})


def test_settled_pause_takes_precedence_over_an_older_worker_sleep():
    from ouroboros.gateway.work_status import classify_member

    row = {"activity_id": "asleep", "phase": "working", "task_attempt": 2}
    raw = {"pause_by": {"actor": "owner"}, "owner_wait": {
        "state": "waiting", "reason": "sleep", "sleep": {"mode": "warm"}, "task_attempt": 2}}
    assert classify_member(row, raw)["activity"] == "waiting"
    state = classify_member({**row, "phase": "budget_paused", "pause_cause": "owner"}, raw)
    assert state["activity"] == "paused"
    assert [reason["code"] for reason in state["reasons"]] == ["paused_by_owner"]
