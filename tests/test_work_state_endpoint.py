"""/api/state end to end: the real census builder feeds the work projection (a held Continue, an empty install)."""
from __future__ import annotations

from ouroboros.task_results import write_task_result
from tests.test_main_state_projection import gateway  # noqa: F401 -- real gateway handlers over a temp data root
from tests.test_swarm_host_admission import host  # noqa: F401 -- the gateway fixture's companion


def _helpers(gateway, root, count, chat_id=42):  # noqa: F811
    for index in range(count):
        task_id = f"{root}-h{index}"
        gateway.running[task_id] = {"attempt": 1, "started_at": 100.0 + index, "task": {
            "id": task_id, "parent_task_id": root, "root_task_id": root, "chat_id": chat_id,
            "delegation_role": "subagent", "type": "task"}}
        write_task_result(gateway.root, task_id, "running", parent_task_id=root, root_task_id=root, chat_id=chat_id)


def test_a_held_continue_with_live_helpers_is_one_working_work_with_stop_earlier(gateway):  # noqa: F811
    write_task_result(gateway.root, "p", "failed", suggested_name="Release 7.7", chat_id=42)
    _helpers(gateway, "p", 3)
    gateway.pending.append({
        "id": "c", "type": "task", "root_task_id": "c", "chat_id": 42, "delegation_role": "root",
        "queued_at": 200.0, "title": "Continue: Release 7.7",
        "_budget_pause_hold": {"reason": "continuation_writer_unsettled", "held_at": "2026-10-17T11:00:00Z",
                               "selected": False, "dispatchable": False},
        "metadata": {"continuation": {"predecessor_task_id": "p", "work_id": "p", "work_name": "Release 7.7"}}})
    write_task_result(gateway.root, "c", "scheduled", root_task_id="c", chat_id=42)
    data = gateway.client.get("/api/state").json()
    assert data["works_complete"] is True, data["works_gaps"]
    [work] = [item for item in data["works"] if item["work_id"] == "p"]
    assert work["name"] == "Release 7.7" and work["episode"] == {"task_id": "c", "live": True}
    assert work["activity"] == "working" and work["turn"] == []
    assert work["because"][0]["code"] == "earlier_attempt_finishing" and work["because"][0]["count"] == 3
    actions = [(entry["action"], entry["target"]) for entry in work["controls"]]
    assert ("stop_earlier", "p") in actions and ("stop", "c") in actions
    assert not any(action in {"resume", "pause"} for action, _ in actions)
    room = data["rooms"]["42"]
    assert room["activity"] == "working" and room["amber"] is False and room["mark"] == "moving"
    # The frozen census is unchanged: the held Continue is still its own root row, helpers stay out of it.
    census_ids = {row["activity_id"] for row in data["active_chat_activities"]}
    assert "c" in census_ids and not any(task_id.startswith("p-h") for task_id in census_ids)


def test_without_its_hold_the_same_continue_is_plain_queued_work(gateway):  # noqa: F811
    write_task_result(gateway.root, "p", "failed", chat_id=42)
    gateway.pending.append({"id": "c", "type": "task", "root_task_id": "c", "chat_id": 42, "delegation_role": "root",
                            "queued_at": 200.0,
                            "metadata": {"continuation": {"predecessor_task_id": "p", "work_id": "p"}}})
    write_task_result(gateway.root, "c", "scheduled", root_task_id="c", chat_id=42)
    data = gateway.client.get("/api/state").json()
    [work] = [item for item in data["works"] if item["work_id"] == "p"]
    assert work["activity"] == "queued" and not any(entry["action"] == "stop_earlier" for entry in work["controls"])


def test_an_empty_install_emits_every_work_field(gateway):  # noqa: F811
    data = gateway.client.get("/api/state").json()
    assert data["works"] == [] and data["rooms"] == {} and data["works_complete"] is True and data["works_gaps"] == []
    assert data["work_summary"]["confirmed"] is True
