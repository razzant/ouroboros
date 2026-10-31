"""D1: anonymized carrier shapes from the census, with no live task contents."""

import copy
import json

import pytest

from ouroboros.work_stop import SELECTION_KEY, continuation_refusal, current_stop
from supervisor.work_stop_migration import FORMAT_KEY, normalize_snapshot
from tests._budget_pause_exact_helpers import _install_queue, _parked


def _snapshot(*tasks):
    return {"pending": [{"id": t["id"], "task": t} for t in tasks], "running": [], "budget_root_fences": []}


def _pause(reason="owner", **extra):
    return {"reason": reason, "status": "paused_exact_continuation", "exact_continuation": True,
            "checkpoint": {"pause_id": "current", "pause_generation": 2}, **extra}


@pytest.mark.parametrize("reason", ["panic_hold", "Ouroboros: the owner asked to pause before updating"])
@pytest.mark.parametrize("has_current", [True, False])
def test_old_copied_reason_never_overrules_current_pause(reason, has_current):
    task = {"id": "root", "root_task_id": "root", "_budget_pause_hold": {
        "reason": reason, "selected": False, "detail": "root_resume_generation_stale"}}
    doc = {"task_id": "root", "status": "scheduled"}
    if has_current:
        task["_budget_pause"] = _pause()
        doc["owner_pause"] = {"state": "requested", "fence_id": "current", "by": {"actor": "ouroboros"}}
    snap = _snapshot(task)
    changes = normalize_snapshot(snap, [doc])
    stop = current_stop(task, {"owner": doc.get("owner_pause")})
    assert stop["cause"] == ("pause" if has_current else "awaiting_selection")
    assert stop["release"] == "resume"
    assert ("_budget_pause_hold" not in task) is has_current
    assert any(row["action"] == "actor_text_archived" for row in changes) == reason.startswith("Ouroboros:")
    once = copy.deepcopy(snap)
    assert normalize_snapshot(snap, [doc]) == [] and snap == once


@pytest.mark.parametrize("has_current", [True, False])
def test_prose_alone_is_archived_without_inventing_a_stop(has_current):
    text = "Ouroboros: the owner asked to pause before updating"
    task = {"id": "root", "_budget_pause_hold": {"reason": text}}
    document = {"task_id": "root", "status": "scheduled"}
    if has_current:
        task["_budget_pause"] = _pause()
        document["owner_pause"] = {"state": "requested", "fence_id": "current"}
    snapshot = _snapshot(task)
    changes = normalize_snapshot(snapshot, [document])
    stop = current_stop(task, {"owner": document.get("owner_pause")})
    assert (stop["cause"] if stop else None) == ("pause" if has_current else None)
    assert "_budget_pause_hold" not in task
    assert {"task_id": "root", "action": "actor_text_archived", "text": text} in changes
    once = copy.deepcopy(snapshot)
    assert normalize_snapshot(snapshot, [document]) == [] and snapshot == once


def test_selected_members_keep_only_q9_bindings_and_new_pause_discards_old_selection():
    binding = {"fence_id": "f", "root_task_id": "root", "selected_by": "root", "selected_at": "then",
               "root_resume_basis": "durable", "root_grant_id": "g", "root_resume_generation": 2}
    chosen = {"id": "child", "root_task_id": "root", "_budget_pause_hold": {
        **binding, "selected": True, "reason": "panic_hold", "detail": "released"}}
    paused = {**copy.deepcopy(chosen), "id": "repaused", "_budget_pause": _pause()}
    snapshot = _snapshot(chosen, paused)
    snapshot["budget_root_fences"] = [{"root_task_id": "root", "status": "paused", "fence_id": "f"}]
    normalize_snapshot(snapshot)
    assert {k: chosen[SELECTION_KEY][k] for k in binding} == binding
    assert not {"reason", "detail", "release", "selected"} & chosen[SELECTION_KEY].keys()
    assert "_budget_pause_hold" not in chosen
    assert SELECTION_KEY not in paused and "_budget_pause_hold" not in paused


@pytest.mark.parametrize("reason,release", [("panic_hold", "owner"), ("saved_work_hold", "resume"),
                                          ("owner_restart_hold", "resume"), ("unrecognized_code", "resume")])
def test_unselected_old_holds_gain_release_or_become_point_refusals(reason, release):
    task = {"id": "held", "_budget_pause_hold": {"reason": reason, "stop_cause": "panic", "selected": False}}
    snap = _snapshot(task)
    normalize_snapshot(snap)
    if reason == "unrecognized_code":
        assert "_budget_pause_hold" not in task
        assert continuation_refusal(task)["reason"] == reason
    else:
        hold = task["_budget_pause_hold"]
        assert hold["release"] == release and hold["door"] == "panic"
        assert "stop_cause" not in hold and "selected" not in hold


def test_child_external_tail_keeps_its_current_point_and_settlement():
    task = {"id": "child", "root_task_id": "root", "_budget_pause": _pause(settlement="external_writers_running")}
    before = copy.deepcopy(task)
    normalize_snapshot(_snapshot(task))
    assert task == before
    assert current_stop(task)["cause"] == "awaiting_selection"


def test_ghost_shapes_are_addressable_but_never_selected_or_terminally_revived():
    documents = [{"task_id": "saved", "status": "scheduled", "resource_limit": {
        "status": "budget_hold", "reason": "saved_work_hold", "stop_cause": "panic"}},
        {"task_id": "orphan", "root_task_id": "finished", "parent_task_id": "finished", "status": "scheduled",
         "budget_pause": {"state": "pausing", "reason": "owner", "pause_id": "p", "settlement": "external_writers_running"}},
        {"task_id": "sleep", "status": "scheduled", "budget_pause": {"state": "paused", "reason": "sleep"}},
        {"task_id": "finished", "status": "completed", "budget_pause": {"state": "paused", "reason": "owner"}},
        {"task_id": "stopped", "status": "cancelled", "resource_limit": {"status": "budget_hold", "reason": "panic_hold"}}]
    snapshot = _snapshot()
    before = copy.deepcopy(documents)
    changes = normalize_snapshot(snapshot, documents)
    assert documents == before
    assert {c["task_id"] for c in changes if c["action"] == "queue_locator_restored"} == {"saved", "orphan", "sleep"}
    for row in snapshot["pending"]:
        assert current_stop(row["task"])["release"] == "resume"
        assert SELECTION_KEY not in row["task"] and "_budget_pause_resume" not in row["task"]
        assert row["task"].get("admitted_dispatch") != "none"


def test_format_stamp_survives_absent_stops_and_empty_snapshot(tmp_path, monkeypatch):
    queue, _, workers = _install_queue(tmp_path, monkeypatch)
    path = tmp_path / "state" / "queue_snapshot.json"
    path.write_text(json.dumps(_snapshot()))
    assert queue.restore_pending_from_snapshot() == 0
    assert json.loads(path.read_text())[FORMAT_KEY] == 1
    assert not workers.PENDING
    from supervisor import work_stop_migration
    monkeypatch.setattr(work_stop_migration, "normalize_snapshot", lambda *_a: pytest.fail("format already converted"))
    assert queue.restore_pending_from_snapshot() == 0


def test_real_restore_normalizes_before_retention_and_repairs_missing_locator(tmp_path, monkeypatch):
    from ouroboros.task_results import write_task_result
    from supervisor.events_budget import BUDGET_HOLD_KEY

    queue, _, workers = _install_queue(tmp_path, monkeypatch)
    task, _ = _parked(tmp_path, monkeypatch, task_id="root")
    task[BUDGET_HOLD_KEY] = {"reason": "panic_hold", "detail": "root_resume_generation_stale"}
    _parked(tmp_path, monkeypatch, task_id="ghost")
    workers.PENDING.pop()
    write_task_result(tmp_path, "saved", "scheduled", chat_id=0,
                      resource_limit={"status": "budget_hold", "reason": "saved_work_hold", "stop_cause": "panic"})
    queue.persist_queue_snapshot()
    path = tmp_path / "state" / "queue_snapshot.json"
    legacy = json.loads(path.read_text())
    legacy.pop(FORMAT_KEY)
    path.write_text(json.dumps(legacy))
    workers.PENDING.clear()
    assert queue.restore_pending_from_snapshot() == 3
    tasks = {t["id"]: t for t in workers.PENDING}
    assert BUDGET_HOLD_KEY not in tasks["root"]
    assert tasks["ghost"]["_budget_pause"]["checkpoint"]["pause_id"]
    assert tasks["saved"][BUDGET_HOLD_KEY]["door"] == "panic"
    from tests.test_budget_pause_holds import _idle_worker
    sent = []
    _idle_worker(workers, sent)
    workers.assign_tasks()
    assert sent == []
    assert json.loads(path.read_text())[FORMAT_KEY] == 1
