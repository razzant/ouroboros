"""The schedule scan observes policy outside Q and leaves idle disabled rows alone."""
import copy

import pytest

from tests.test_schedule_occurrence import q as _q, _row, _rows
from tests.test_g1_followup_policy import world as _world, register, stop

q, world = _q, _world


def test_disabled_rows_without_occurrences_do_no_store_work(q, monkeypatch):
    from ouroboros import store, task_store
    from supervisor import queue_schedules as schedules, followup_policy

    rows = [{"id": f"off-{i}", "enabled": False, "source": "task_followup",
             "followup_origin": {"task_id": "origin", "root_task_id": "origin"},
             "last_task_id": f"old-{i}", "trigger": {"type": "cron", "expr": "* * * * *"},
             # The outcome of the last run is already cached, so the pass owes no result read.
             "last_outcome": {"task_id": f"old-{i}", "state": "completed", "quiet": False,
                              "notice": False, "ended_at": None, "line": ""}}
            for i in range(33)]
    schedules._write_scheduled_tasks({"tasks": rows}, q.root)
    before = schedules.load_schedule_store(q.root)
    def unexpected(*_args, **_kwargs):
        pytest.fail("disabled idle schedules must not read origins, receipts or policy")
    monkeypatch.setattr(store.Txn, "body_text", unexpected)
    from contextlib import contextmanager
    original = task_store.open_store
    @contextmanager
    def off_lock(*args, **kwargs):
        assert not q.queue._queue_lock._is_owned()
        with original(*args, **kwargs) as txn:
            yield txn
    monkeypatch.setattr(task_store, "open_store", off_lock)
    monkeypatch.setattr(followup_policy, "policy_view", unexpected)
    schedules.check_scheduled_tasks()
    assert schedules.load_schedule_store(q.root) == before
    assert q.pending == []


def test_disabled_occurrence_still_settles(q):
    from ouroboros.task_results import write_task_result
    _row(q, intent={"kind": "system_repo"})
    q.queue.check_scheduled_tasks()
    task = q.pending.pop()
    assert not _rows(q)["s1"]["enabled"]
    write_task_result(q.root, task["id"], "completed")
    q.queue.check_scheduled_tasks()
    assert "occurrence" not in _rows(q)["s1"]


def test_policy_scan_is_outside_q_and_unchanged_due_work_admits(world, monkeypatch):
    from supervisor import followup_policy, queue, queue_schedules as schedules
    register(world)
    reads = []
    original = followup_policy.policy_view
    def observe(*args):
        reads.append(queue._queue_lock._is_owned())
        return original(*args)
    monkeypatch.setattr(followup_policy, "policy_view", observe)
    schedules.check_scheduled_tasks()
    assert reads[0] is False  # scan
    assert len(world[2]) == 1  # final admission still verifies current controls


@pytest.mark.parametrize("control", ["stop", "pause"])
def test_control_landing_after_offlock_observation_blocks_admission(world, monkeypatch, control):
    from ouroboros import owner_pause
    from supervisor import followup_policy, queue, queue_schedules as schedules
    register(world)
    original = followup_policy.policy_view
    observed = []
    def race(*args):
        view = original(*args)
        if not observed:
            assert not queue._queue_lock._is_owned()
            observed.append(True)
            if control == "stop":
                stop(world[1])
            else:
                monkeypatch.setattr(owner_pause, "read_fence", lambda *_a, **_k: {
                    "state": "paused", "fence_id": "new-pause"})
        return view
    monkeypatch.setattr(followup_policy, "policy_view", race)
    schedules.check_scheduled_tasks()
    assert world[2] == []
    record = schedules.load_schedule_store(world[1])["tasks"][0]
    assert record.get("followup_hold") or record.get("followup_wait")


def test_row_edit_during_policy_read_is_never_overwritten(world, monkeypatch):
    from supervisor import followup_policy, queue, queue_schedules as schedules
    record = register(world)
    original = followup_policy.policy_view
    def edit(root, data, row):
        assert not queue._queue_lock._is_owned()
        view = original(root, data, row)
        with schedules.schedule_transaction(root):
            current = schedules.load_schedule_store(root)
            changed = next(r for r in current["tasks"] if r["id"] == record["id"])
            changed["enabled"] = False
            changed["name"] = "Owner changed this during observation"
            schedules._write_scheduled_tasks(current, root)
        return view
    monkeypatch.setattr(followup_policy, "policy_view", edit)
    schedules.check_scheduled_tasks()
    assert world[2] == []
    current = schedules.load_schedule_store(world[1])["tasks"][0]
    assert current == {**copy.deepcopy(record), "enabled": False, "name": "Owner changed this during observation"}


@pytest.mark.parametrize("control", ["none", "stop", "pause"])
def test_reenabled_row_rechecks_controls_before_admission(world, control):
    from ouroboros.owner_pause import install_fence
    from supervisor import queue_schedules as schedules
    from tests.test_g1_followup_policy import ORIGIN

    registered = register(world)
    root = world[1]
    def set_enabled(enabled):
        with schedules.schedule_transaction(root):
            data = schedules.load_schedule_store(root)
            row = next(row for row in data["tasks"] if row["id"] == registered["id"])
            row["enabled"] = enabled
            schedules._write_scheduled_tasks(data, root)
    set_enabled(False)
    if control == "stop":
        stop(root)
    elif control == "pause":
        install_fence(root, ORIGIN, request_id="disabled-pause")
    schedules.check_scheduled_tasks()
    assert not world[2]
    set_enabled(True)
    schedules.check_scheduled_tasks()
    assert len(world[2]) == (1 if control == "none" else 0)
    current = schedules.load_schedule_store(root)["tasks"][0]
    assert bool(current.get("followup_hold") or current.get("followup_wait")) == (control != "none")
