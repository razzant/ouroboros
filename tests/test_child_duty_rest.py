"""Child jobs rest on their own sources; retention wakes at actual age eligibility."""
import os
import time

import pytest

from ouroboros import maintenance, maintenance_jobs, maintenance_retirement, plan_record, store
from ouroboros.task_results import write_task_result
from tests._cost_meter import BOUNDARY, CostMeter, HISTORY
from tests.test_plan_record import _participant, _record


def measured_rest(root, job):
    store.close_connections()
    with CostMeter(root) as meter:
        token = BOUNDARY.set("rest")
        try:
            assert maintenance.run_job(root, job) is None
        finally:
            BOUNDARY.reset(token)
    counts = meter.snapshot("rest")["counts"]
    assert counts.get("writes", 0) == 0
    assert all(counts.get(f"{kind}:{category}", 0) == 0
               for category in HISTORY for kind in ("open", "read_bytes"))
    return {key: counts.get(key, 0) for key in ("vm_steps", "statements", "json_loads")}


def test_retirement_rests_after_unrelated_completion_and_wakes_at_earliest_age(tmp_path, monkeypatch):
    costs = []
    now = time.time()
    for scale in (1, 30):
        root = tmp_path / f"root{scale:02}"
        shots = root / "uploads/screenshots"
        shots.mkdir(parents=True)
        for index in range(10 * scale):
            path = shots / f"shot{index:04}.png"
            path.write_bytes(b"kept")
            os.utime(path, (now, now))
        maintenance_retirement.record_startup(root, gaps=True)
        assert maintenance.run_job(root, "runtime_retirement")["state"] == "done"
        write_task_result(root, "unrelated", "completed")
        costs.append(measured_rest(root, "runtime_retirement"))
        deadline = maintenance_jobs.job(root, "runtime_retirement")["next_due"]
        (root / "services").mkdir()  # directory activity is no reason to traverse history
        with monkeypatch.context() as clock:
            clock.setattr("ouroboros.retention.time.time", lambda: deadline - 1)
            measured_rest(root, "runtime_retirement")
            assert len(list(shots.iterdir())) == 10 * scale
            clock.setattr("ouroboros.retention.time.time", lambda: deadline + 1)
            assert maintenance.run_job(root, "runtime_retirement")["state"] == "done"
            assert list(shots.iterdir()) == []
            measured_rest(root, "runtime_retirement")
    assert costs[0] == costs[1]


def test_child_requires_recorded_startup_success_and_keeps_unknown_recovery_links(tmp_path):
    write_task_result(tmp_path, "settled", "completed", ts="2000-01-01T00:00:00+00:00")
    tree = tmp_path / "task_trees/settled"
    tree.mkdir(parents=True)
    os.utime(tree, (1, 1))
    maintenance.run_job(tmp_path, "runtime_retirement")
    assert tree.exists(), "missing startup result is unknown, not success"
    maintenance_retirement.record_startup(tmp_path, gaps=True)
    maintenance.run_job(tmp_path, "runtime_retirement")
    assert tree.exists()
    maintenance_retirement.record_startup(tmp_path, gaps=False)
    maintenance.run_job(tmp_path, "runtime_retirement")
    assert not tree.exists()


def test_malformed_retention_date_does_not_abort_a_task_write_or_schedule_retirement(tmp_path):
    from ouroboros import task_store

    write_task_result(tmp_path, "ended", "completed")
    (tmp_path / "task_trees/ended").mkdir(parents=True)
    assert maintenance_jobs.job(tmp_path, "runtime_retirement") is None
    for sequence, timestamp in enumerate(("not-a-timestamp", "12345.5"), start=2):
        with store.open_store(tmp_path, write=True) as txn:
            txn.conn.execute("UPDATE tasks SET drive_retention_at=? WHERE task_id='ended'", (timestamp,))
        handoff = {"state": "closed", "revision": sequence}
        assert task_store.patch_handoff(tmp_path, "ended", "model_waits", "wait", lambda _: handoff) == handoff
        with store.open_store(tmp_path, write=False) as txn:
            assert txn.row("ended", "write_seq") == sequence
            assert txn.handoff("ended", "wait") == handoff, "the task writer committed through the retirement seam"
        job = maintenance_jobs.job(tmp_path, "runtime_retirement")
        if sequence == 2:
            assert job is None, "malformed retention data supplies no deadline"
        else:
            assert job["next_due"] == 12345.5 + maintenance_retirement.get_gc_retention_days() * 86400
            assert job["arrival"]


def test_ended_plan_job_rests_flat_and_wakes_on_terminal_or_producer_facts(tmp_path, monkeypatch):
    from ouroboros.observability import persist_call
    costs = []
    for scale in (1, 30):
        root = tmp_path / f"root{scale:02}"
        for index in range(5 * scale):
            record = _record([_participant("s1")])
            write_task_result(root, f"ended{index:04}", "completed", plan_record=record)
        assert maintenance.run_job(root, "ended_plan_asks")["state"] == "done"
        write_task_result(root, "unrelated", "completed")
        with monkeypatch.context() as clock:
            clock.setattr("ouroboros.utils.utc_now_iso", lambda: "2099-01-01T00:00:00+00:00")
            costs.append(measured_rest(root, "ended_plan_asks"))
        write_task_result(root, "new-ended", "completed")
        measured_rest(root, "ended_plan_asks")
        seen, load = [], plan_record.load_plan_record
        def observe(root, task_id):
            seen.append(task_id)
            return load(root, task_id)
        with monkeypatch.context() as changed:
            changed.setattr(plan_record, "load_plan_record", observe)
            write_task_result(root, "ended0000", "completed", plan_record=record, result="changed plan")
            assert maintenance.run_job(root, "ended_plan_asks") is not None
        assert seen == ["ended0000"]
        persist_call(root, task_id="ended0000", call_id="producer", call_type="plan_review_response",
                     payload={"producer_outcome": {}}, manifest={"surface": "plan_review", "producer_complete": True})
        seen.clear()
        with monkeypatch.context() as produced:
            produced.setattr(plan_record, "load_plan_record", observe)
            assert maintenance.run_job(root, "ended_plan_asks") is not None
        assert seen == ["ended0000"]
        assert maintenance.run_job(root, "ended_plan_asks") is None
    assert costs[0] == costs[1]


def test_a_racing_retirement_input_is_not_recorded_as_examined(tmp_path, monkeypatch):
    from ouroboros import delegate_state_sweep
    def arrive(root):
        maintenance_retirement.artifact_changed(root, timestamp=1)
        return {}
    monkeypatch.setattr(delegate_state_sweep, "sweep_settled_delegate_state", arrive)
    maintenance.run_job(tmp_path, "runtime_retirement")
    row = maintenance_jobs.job(tmp_path, "runtime_retirement")
    assert maintenance_retirement.due(tmp_path, row, {}, True)


def test_new_retained_images_schedule_their_own_age_after_an_empty_pass(tmp_path, monkeypatch):
    from ouroboros.image_preparation import retain_original
    maintenance.run_job(tmp_path, "runtime_retirement")
    original = retain_original(tmp_path / "uploads/views", b"original", "source.png")
    first = maintenance_jobs.job(tmp_path, "runtime_retirement")["next_due"]
    measured_rest(tmp_path, "runtime_retirement")
    later = original.parent / "later.png"
    later.write_bytes(b"later")
    os.utime(later, (original.stat().st_mtime + 86400, original.stat().st_mtime + 86400))
    maintenance_retirement.artifact_changed(tmp_path, timestamp=later.stat().st_mtime)
    with monkeypatch.context() as clock:
        clock.setattr("ouroboros.retention.time.time", lambda: first + 1)
        maintenance.run_job(tmp_path, "runtime_retirement")
        assert not original.exists() and later.exists()
        measured_rest(tmp_path, "runtime_retirement")
        second = maintenance_jobs.job(tmp_path, "runtime_retirement")["next_due"]
        assert second > first
        clock.setattr("ouroboros.retention.time.time", lambda: second + 1)
        maintenance.run_job(tmp_path, "runtime_retirement")
        assert not later.exists()
        measured_rest(tmp_path, "runtime_retirement")


def test_racing_plan_publication_survives_the_examined_signature(tmp_path, monkeypatch):
    from ouroboros import plan_collection_job
    write_task_result(tmp_path, "plan", "completed", plan_record=_record([_participant("s1")]))
    assert maintenance.run_job(tmp_path, "ended_plan_asks") == {"state": "done", "items": 0}
    plan_collection_job.producer_published(tmp_path, ["plan", "first", "digest"])
    load = plan_record.load_plan_record
    def arrive(root, task_id):
        plan_collection_job.producer_published(root, [task_id, "second", "digest"])
        return load(root, task_id)
    with monkeypatch.context() as race:
        race.setattr(plan_record, "load_plan_record", arrive)
        assert maintenance.run_job(tmp_path, "ended_plan_asks") == {"state": "done", "items": 0}
    with store.open_store(tmp_path, write=False) as txn:
        assert txn.get_row(plan_collection_job.CHANGED, "plan") == {"producer": ["plan", "second", "digest"]}
    assert maintenance.run_job(tmp_path, "ended_plan_asks") == {"state": "done", "items": 0}
    measured_rest(tmp_path, "ended_plan_asks")


def test_metrics_child_discovers_a_death_after_append_and_rests_after_catchup(tmp_path, monkeypatch):
    from ouroboros import subagent_metrics
    from ouroboros.utils import append_jsonl
    from tests.test_subagent_metrics import event
    subagent_metrics.publish(tmp_path, event("already-folded"))
    maintenance.run_job(tmp_path, "subagent_metrics")

    class ProcessExit(BaseException):
        pass

    def append(path, row):
        job = maintenance_jobs.job(tmp_path, "subagent_metrics")
        assert subagent_metrics.identity_key(row) == job["publication_owed"]
        return append_jsonl(path, row)

    with monkeypatch.context() as crash:
        crash.setattr(subagent_metrics, "append_jsonl", append)
        crash.setattr(subagent_metrics, "_publish_suffix", lambda *a: (_ for _ in ()).throw(ProcessExit()))
        with pytest.raises(ProcessExit):
            subagent_metrics.publish(tmp_path, event("saved-before-death"))
    assert maintenance.run_job(tmp_path, "subagent_metrics")["state"] == "done"
    assert subagent_metrics.read(tmp_path, "worker")["rows"][0]["dispatched"] == 2
    assert maintenance_jobs.job(tmp_path, "subagent_metrics")["publication_owed"] is None
    measured_rest(tmp_path, "subagent_metrics")


def test_completed_metrics_job_ignores_unrelated_events(tmp_path):
    from ouroboros import subagent_metrics
    from ouroboros.utils import append_jsonl
    from tests.test_subagent_metrics import event

    subagent_metrics.publish(tmp_path, event())
    maintenance.run_job(tmp_path, "subagent_metrics")
    for index in range(3):
        append_jsonl(tmp_path / "logs/events.jsonl", {"type": "unrelated_event", "index": index})
        measured_rest(tmp_path, "subagent_metrics")
