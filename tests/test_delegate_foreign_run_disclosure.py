"""Foreign-run refusals disclose the facts the caller needs to stop retrying.

The ``run_not_owned`` guard stays a refusal for every mutating delegate verb —
there is deliberately no read-only run surface — but the refusal now carries
``owner_task_id`` / ``run_settled`` / ``run_terminal_state`` so a foreign
caller learns the run is over and whom to ask (``get_task_result(owner)`` is
the legitimate ownership-free cross-task read). The terminal-state string is
replayed off the SETTLED row, which previously dropped it on the floor.
"""

import json

from ouroboros import delegate_custody as custody
from ouroboros.delegate_shared import _owned_run


class _Ctx:
    def __init__(self, task_id, drive_root):
        self.task_id = task_id
        self.drive_root = str(drive_root)
        self.budget_drive_root = str(drive_root)
        self.task_metadata = {"root_task_id": task_id}


def test_foreign_refusal_carries_owner_and_settlement_facts(tmp_path):
    custody._CUSTODY.clear()
    try:
        entry = custody.RunCustody(
            task_id="task-owner", route_id="codex", model="m",
            project_id="prj", project_owned=False,
        )
        entry.settled = True
        entry.terminal_state = "succeeded"
        custody._CUSTODY["run-x"] = entry

        refusal, returned = _owned_run(_Ctx("task-other", tmp_path), "delegate_wait", "run-x")
        assert returned is None
        assert (refusal.status, refusal.code) == ("error", "TOOL_REPORTED_FAILURE")
        out = json.loads(refusal.text)
        assert out["reason"] == "run_not_owned"
        assert out["owner_task_id"] == "task-owner"
        assert out["run_settled"] is True
        assert out["run_terminal_state"] == "succeeded"
    finally:
        custody._CUSTODY.clear()


def test_unknown_refusal_names_the_crosstask_read(tmp_path):
    custody._CUSTODY.clear()
    refusal, returned = _owned_run(_Ctx("task-a", tmp_path), "delegate_cancel", "run-nowhere")
    assert returned is None
    assert (refusal.status, refusal.code) == ("error", "TOOL_REPORTED_FAILURE")
    out = json.loads(refusal.text)
    assert out["reason"] == "run_ownership_unknown"
    assert "get_task_result" in out.get("hint", "")


def test_settled_row_state_replays_into_terminal_state(tmp_path):
    rows = [
        {"type": custody.STARTED, "run_id": "r1", "task_id": "t1",
         "route": "codex", "shape": {}},
        {"type": custody.SETTLED, "run_id": "r1", "task_id": "t1",
         "state": "failed"},
    ]
    state = custody.replay(tmp_path, rows=rows)
    assert state["r1"].settled is True
    assert state["r1"].terminal_state == "failed"


def test_settled_row_without_state_stays_empty(tmp_path):
    rows = [
        {"type": custody.STARTED, "run_id": "r2", "task_id": "t1",
         "route": "codex", "shape": {}},
        {"type": custody.SETTLED, "run_id": "r2", "task_id": "t1"},
    ]
    state = custody.replay(tmp_path, rows=rows)
    assert state["r2"].settled is True
    assert state["r2"].terminal_state == ""
