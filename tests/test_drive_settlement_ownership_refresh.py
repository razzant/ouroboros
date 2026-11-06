"""#2709: the ownership read a drive settlement carries into its interlock is refreshed after Phase A.

``settle_child_drive`` asks the supervisor's probe (``task_settlement_liveness``) outside the queue lock,
publishes the drive's deliverables and writes the canonical row (Phase A), then asks once more under the
queue lock (Phase B). The locked probe re-checks the ownership read prepared before the lock against the
store's signature; a signature Phase A itself moved used to read as "changed" and the drive stayed with the
reason ``task_live`` although nothing ran. The read is now refreshed after Phase A and before the interlock,
so the drive goes; a row another writer moves after that refresh still refuses; a store busy under the
lock is reported as ``store_busy``, never as a live task. Real supervisor queue maps, real store, real
probe; isolated roots only.
"""
from __future__ import annotations

import pytest

from ouroboros import headless, task_custody, task_results
from ouroboros.store import StoreBusy
from supervisor import queue, queue_transitions, workers
from supervisor.task_ownership import TaskOwnershipRead
from tests.test_child_drive_settlement import TASK, _capture

PROBE = queue_transitions.task_settlement_liveness


@pytest.fixture
def env(tmp_path, monkeypatch):
    """A supervisor process with empty live maps whose durable rows live in ``data``."""
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setattr(queue, "DRIVE_ROOT", data)
    monkeypatch.setattr(queue, "RUNNING", {})
    monkeypatch.setattr(queue, "PENDING", [])
    monkeypatch.setattr(queue, "INITIALIZED", True)
    monkeypatch.setattr(queue, "QUEUE_MAX_RETRIES", 3)
    monkeypatch.setattr(workers, "WORKERS", {})
    monkeypatch.setattr(workers, "direct_chat_turn", lambda _: None)
    monkeypatch.setattr(queue_transitions, "post_task_synthesis_in_flight", lambda *_: False)
    return data


def _publishing_drive(data, tmp_path):
    """A cancelled split task whose child captured a file copy-back never published: Phase A has a
    publication to make and a custody row to write."""
    drive = headless.prepare_task_drive(data, TASK, "empty")
    record = _capture(tmp_path, drive)
    task_results.write_task_result(drive, TASK, "running", artifacts=[record])
    task_results.write_task_result(data, TASK, "cancelled", result="Cancelled.", child_drive_root=str(drive))
    return drive


def _settle(data, drive, live=PROBE):
    return task_custody.settle_child_drive(data, TASK, drive, live=live,
                                           guard=lambda: queue_transitions.task_settlement_interlock())


def test_a_drive_whose_phase_a_wrote_its_own_custody_row_is_freed_in_phase_b(env, tmp_path):
    drive = _publishing_drive(env, tmp_path)
    assert _settle(env, drive) == {"status": "removed", "reason": "", "published": 1}
    assert not drive.exists()
    row = task_results.load_task_result(env, TASK)
    assert row["status"] == "cancelled" and any(item.get("immutable") for item in row["artifacts"])


def test_a_row_another_writer_moves_after_the_refresh_still_refuses(env, tmp_path, monkeypatch):
    """The interlock's recheck keeps its teeth: a write that lands between the refreshed read and the
    queue lock is seen there and the drive stays (its reason names the moved row, not a live task)."""
    drive = _publishing_drive(env, tmp_path)
    real = queue_transitions.task_settlement_interlock

    def interlock_after_a_foreign_write(stop=None):
        task_results.write_task_result(env, TASK, "cancelled", result="Cancelled.", child_drive_root=str(drive),
                                       foreign_note="another publisher")
        return real(stop=stop)

    outcome = task_custody.settle_child_drive(env, TASK, drive, live=PROBE, guard=interlock_after_a_foreign_write)
    assert outcome["status"] == "retained" and outcome["reason"] == "liveness_unknown", outcome
    assert drive.is_dir()
    assert _settle(env, drive)["status"] == "removed", "the next pass, with nothing else moving, settles it"


def test_a_running_occupant_still_keeps_its_drive_and_is_named_live(env, tmp_path):
    drive = _publishing_drive(env, tmp_path)
    queue.RUNNING[TASK] = {"task": {"id": TASK}}
    assert _settle(env, drive) == {"status": "retained", "reason": "task_live", "published": 0}
    assert drive.is_dir()
    queue.RUNNING.clear()
    assert _settle(env, drive)["status"] == "removed"


def test_a_store_busy_under_the_lock_is_reported_busy_never_live(env, tmp_path, monkeypatch):
    """The locked recheck never waits; a busy store is unknown ownership with its cause, so the drive
    waits for the next pass as ``store_busy`` (Free's "busy right now"), not as a task that runs."""
    drive = _publishing_drive(env, tmp_path)
    real = TaskOwnershipRead._stamp

    def busy_under_the_lock(self, task_id):
        if queue._queue_lock._is_owned():
            raise StoreBusy("ownership re-check found the store busy", reason="contention")
        return real(self, task_id)

    monkeypatch.setattr(TaskOwnershipRead, "_stamp", busy_under_the_lock)
    with pytest.raises(StoreBusy):
        PROBE(TASK)
    assert _settle(env, drive) == {"status": "retained", "reason": "store_busy", "published": 0}
    assert drive.is_dir()
    monkeypatch.setattr(TaskOwnershipRead, "_stamp", real)
    assert _settle(env, drive)["status"] == "removed"


def test_the_probe_answers_false_only_for_proven_absence(env):
    task_results.write_task_result(env, TASK, "completed", result="done")
    assert PROBE(TASK) is False
    queue.PENDING.append({"id": TASK})
    assert PROBE(TASK) is True
