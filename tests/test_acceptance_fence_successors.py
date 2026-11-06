"""#1183: S1 successors of an ended attempt are not closed by that attempt's own fence.

The aborted-update return (``restart_retention.recover_aborted_update``, captured
process death) and the boot restore of retained saved work (a previous supervisor
generation) are the handoffs; each is driven through its real door.
"""
from __future__ import annotations

import json

import pytest



@pytest.fixture(autouse=True)
def isolated_fences(monkeypatch):
    """The cases compare whole fence maps: never inherit an earlier test's rows."""
    from supervisor import queue, task_lifecycle

    budget_fences = {}
    monkeypatch.setattr(queue, "ACCEPTANCE_FENCES", {})
    monkeypatch.setattr(queue, "BUDGET_ROOT_FENCES", budget_fences)
    monkeypatch.setattr(task_lifecycle, "BUDGET_ROOT_FENCES", budget_fences)


def test_an_aborted_update_returns_saved_work_without_its_dead_attempts_fence(tmp_path, monkeypatch):
    from supervisor import workers
    from tests.test_update_abort_retention import _assert_returned, _system

    # The real respawn caches ``_system``'s fake multiprocessing context in ``workers._CTX``;
    # restore that global at teardown so later tests get the real context again.
    monkeypatch.setattr(workers, "_CTX", None)
    system = _system(tmp_path, monkeypatch)
    system.saved()
    meta = system.workers.RUNNING["saved"]
    worker = system.workers.WORKERS[meta["worker_id"]]
    ack = system.queue.transition_acceptance_fence(
        action="begin", token="c" * 32, task_id="saved", root_task_id="saved",
        pooled_owner={"task_attempt": 1, "pid": worker.proc.pid})
    assert ack["status"] == "active"
    sealed = {"token": "d" * 32, "root_task_id": "other-root", "task_id": "other-root", "status": "sealed",
              "owner_message_generation": 0, "pooled_owner": {"task_attempt": 1, "pid": worker.proc.pid}}
    system.queue.ACCEPTANCE_FENCES["other-root"] = dict(sealed)
    assert system.control._quiesce_repo_writers("auto_merge") == []
    system.control._respawn_workers_after_failed_update()
    _assert_returned(system, tmp_path)
    assert system.queue.ACCEPTANCE_FENCES == {"other-root": sealed}  # only the proven-dead attempt's own row
    persisted = json.loads(system.queue.QUEUE_SNAPSHOT_PATH.read_text())
    assert [row["root_task_id"] for row in persisted["acceptance_fences"]] == ["other-root"]
    system.workers.assign_tasks()
    [commands] = system.launches
    assert commands.get_nowait()["id"] == "saved"


@pytest.mark.parametrize("fence", ["own_previous_attempt", "legacy_unattributed", "sealed"])
def test_boot_restore_keeps_a_saved_successor_out_of_its_ended_attempts_fence(tmp_path, monkeypatch, fence):
    from ouroboros.task_results import load_task_result
    from tests._budget_pause_exact_helpers import _install_queue
    from tests.test_restart_retention import _pool_events
    from tests.test_restart_saved_work import _working_run

    queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    _pool_events(workers, monkeypatch)
    _working_run(tmp_path, workers, "saved-root")
    row = {"token": "e" * 32, "root_task_id": "saved-root", "task_id": "saved-root", "status": "active",
           "owner_message_generation": 0, "pooled_owner": {"task_attempt": 1, "pid": 4242}}
    if fence == "legacy_unattributed":
        row.pop("pooled_owner")
    elif fence == "sealed":
        row["status"] = "sealed"
    monkeypatch.setattr(queue, "ACCEPTANCE_FENCES", {"saved-root": row})
    # The application stops with the fence in its final snapshot; the next process
    # starts with an empty in-memory map and reads only that snapshot.
    workers.kill_workers(force=True, terminal_status="cancelled", result_reason="Server shutdown.",
                         stop_source="server_shutdown", retain_saved_work=True)
    assert json.loads(queue.QUEUE_SNAPSHOT_PATH.read_text())["acceptance_fences"] == [row]
    queue.ACCEPTANCE_FENCES.clear()
    workers.PENDING[:] = []
    workers.RUNNING.clear()
    queue.restore_pending_from_snapshot()
    restored = [task for task in workers.PENDING if task["id"] == "saved-root"]
    if fence == "own_previous_attempt":
        # Attempt 2 continues the saved work; attempt 1's begin closes nothing for it.
        assert restored and restored[0]["_attempt"] == 2
        assert restored[0]["_working_recovery"]["from_attempt"] == 1
        assert load_task_result(tmp_path, "saved-root")["status"] != "cancelled"
    else:
        # Unattributed and sealed rows keep their existing boot meaning: the root
        # entered acceptance, so the restore cancels instead of returning work.
        assert not restored
        assert load_task_result(tmp_path, "saved-root")["status"] == "cancelled"


@pytest.mark.parametrize("carrier", ["plain_retry", "runtime_recovery_carrier"])
def test_boot_restore_never_infers_an_ended_attempt_from_the_attempt_number(tmp_path, monkeypatch, carrier):
    """An ordinary QUEUED retry of attempt 2 is no validated stopped-generation handoff (only
    a saved RUNNING row recovery turned into its successor is): attempt 1's active fence stays,
    with its existing boot meaning (the root entered acceptance)."""
    from ouroboros.task_results import load_task_result, write_task_result
    from tests._budget_pause_exact_helpers import _install_queue
    from tests.test_restart_retention import _pool_events

    queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    _pool_events(workers, monkeypatch)
    write_task_result(tmp_path, "retry-root", "scheduled", chat_id=0, root_task_id="retry-root")
    retry = {"id": "retry-root", "type": "task", "chat_id": 0, "_attempt": 2, "root_task_id": "retry-root"}
    if carrier == "runtime_recovery_carrier":  # a queued crash/timeout retry's own carrier, not a boot handoff
        retry["_working_recovery"] = {"source_task_id": "retry-root", "from_attempt": 1, "cause": "worker_crash"}
    workers.PENDING.append(retry)
    row = {"token": "f" * 32, "root_task_id": "retry-root", "task_id": "retry-root", "status": "active",
           "owner_message_generation": 0, "pooled_owner": {"task_attempt": 1, "pid": 4242}}
    monkeypatch.setattr(queue, "ACCEPTANCE_FENCES", {"retry-root": row})
    workers.kill_workers(force=True, terminal_status="cancelled", result_reason="Server shutdown.",
                         stop_source="server_shutdown", retain_saved_work=True)
    snap = json.loads(queue.QUEUE_SNAPSHOT_PATH.read_text())
    assert snap["acceptance_fences"] == [row]
    assert [item["task"]["id"] for item in snap["pending"]] == ["retry-root"] and not snap["running"]
    queue.ACCEPTANCE_FENCES.clear()
    workers.PENDING[:] = []
    workers.RUNNING.clear()
    queue.restore_pending_from_snapshot()  # the snapshot fence decides the root entered acceptance
    assert not [task for task in workers.PENDING if task["id"] == "retry-root"]
    assert load_task_result(tmp_path, "retry-root")["status"] == "cancelled"
