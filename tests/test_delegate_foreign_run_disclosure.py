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
    # The kind is unknown here: the general pointer, as before.
    assert out["hint"] == ("The run may belong to a different drive or the id may be mistyped; "
                           "get_task_result(<task_id>) is the ownership-free way to read another task's "
                           "delegated-run outcome.")


def test_a_task_id_given_as_a_run_id_names_the_exact_read(tmp_path, monkeypatch):
    """Both directions: a task this drive records is named with its ready read; a contract that withholds
    get_task_result, and a store that cannot be read, keep the general pointer and the typed refusal."""
    from ouroboros import task_status
    from ouroboros.store import StoreBusy
    from ouroboros.task_results import write_task_result

    custody._CUSTODY.clear()
    write_task_result(tmp_path, "kidtask01", "completed", result="done")
    refusal, _ = _owned_run(_Ctx("task-a", tmp_path), "delegate_wait", "kidtask01")
    out = json.loads(refusal.text)
    assert out["reason"] == "run_ownership_unknown"
    assert out["hint"] == ("That id names a task, not a delegated run: "
                           "get_task_result(task_id='kidtask01') reads it.")
    withheld = _Ctx("task-a", tmp_path)
    withheld.task_metadata["disabled_tools"] = ["get_task_result"]
    refusal, _ = _owned_run(withheld, "delegate_wait", "kidtask01")
    assert json.loads(refusal.text)["hint"].startswith("The run may belong to a different drive")

    def busy(*_args, **_kwargs):
        raise StoreBusy("contention")

    monkeypatch.setattr(task_status, "load_effective_task_result", busy)
    refusal, _ = _owned_run(_Ctx("task-a", tmp_path), "delegate_cancel", "kidtask01")
    out = json.loads(refusal.text)
    assert out["reason"] == "run_ownership_unknown" and out["hint"].startswith("The run may belong")


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
