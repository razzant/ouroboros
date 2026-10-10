"""Deferred timeout work cannot acquire a replacement worker's authority."""
import queue
import time
from types import SimpleNamespace

import pytest

from ouroboros import headless
from ouroboros.task_results import load_task_result, write_task_result
from supervisor import events, task_reaper, worker_pool_lifecycle
from tests.test_phase1_orchestration import _FakeProc, _patch_queue


class FiniteQueue(queue.Queue):
    def get(self, *args, **kwargs):
        if self.empty():
            raise KeyboardInterrupt("finite test pump ended")
        return super().get(*args, **kwargs)


@pytest.fixture
def timed_out(tmp_path, monkeypatch):
    from supervisor import queue as q, workers

    old = SimpleNamespace(wid=7, proc=_FakeProc(), busy_task_id="old-task", reaping=False)
    _patch_queue(q, workers, monkeypatch, tmp_path, {7: old})
    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(workers, "RUNNING", q.RUNNING)
    monkeypatch.setattr(workers, "PENDING", q.PENDING)
    monkeypatch.setattr(task_reaper, "_deferred_reap_jobs", [])
    monkeypatch.setattr("supervisor.cancel_publication._audit_delegated_runs_on_kill",
                        lambda *_a, **_kw: {"unreconciled": []})
    monkeypatch.setattr("ouroboros.tools.services.archive_task_service_logs", lambda *_a, **_kw: None)
    jobs = FiniteQueue()
    monkeypatch.setattr(task_reaper, "reap_queue", jobs)
    monkeypatch.setattr(q, "_reap_queue", jobs)
    frames, pushed, spawned = [], [], []
    monkeypatch.setattr(workers, "get_event_q", lambda: SimpleNamespace(put=frames.append))
    monkeypatch.setattr(workers, "respawn_worker", worker_pool_lifecycle.respawn_worker)
    monkeypatch.setattr(worker_pool_lifecycle, "_spawn_worker_slot",
                        lambda wid, captured, **_kw: spawned.append((wid, captured)) or True)
    child = headless.prepare_task_drive(tmp_path, "old-task", "empty")
    task = {"id": "old-task", "type": "task", "chat_id": 5, "_attempt": 1,
            "deadline_at": "2000-01-01T00:00:00Z", "drive_root": str(child),
            "delegation_role": "subagent", "task_constraint": {"mode": "local_readonly_subagent"}}
    q.RUNNING["old-task"] = {"task": task, "worker_id": 7, "attempt": 1,
                              "started_at": time.time() - 1000, "last_heartbeat_at": time.time() - 1000}
    write_task_result(tmp_path, "old-task", "running")
    write_task_result(child, "old-task", "completed", result="Saved old answer", artifact_status="ready")
    q.enforce_task_timeouts()
    assert jobs.qsize() == 1 and old.reaping and old.busy_task_id is None
    job = jobs.queue[0]
    assert job["worker"] is old and job["drive_root"] == str(tmp_path) and job["attempt"] == 1
    ctx = SimpleNamespace(DRIVE_ROOT=tmp_path, RUNNING=q.RUNNING, WORKERS=workers.WORKERS,
                          bridge=SimpleNamespace(push_log=pushed.append),
                          send_with_budget=lambda *_a, **_kw: None,
                          persist_queue_snapshot=lambda **_kw: None)
    return SimpleNamespace(q=q, workers=workers, root=tmp_path, old=old, task=task, child=child,
                           jobs=jobs, job=job, spawned=spawned, frames=frames, pushed=pushed, ctx=ctx)


def pump_once():
    with pytest.raises(KeyboardInterrupt, match="finite test pump"):
        task_reaper.reaper_loop()


@pytest.mark.parametrize("closure", ["confirmed", "unconfirmed: process table unreadable"])
def test_confirmed_timeout_sweeps_bridge_scope_without_rolling_back(timed_out, monkeypatch, closure):
    from ouroboros import mcp_task_sessions

    t = timed_out
    swept = []
    monkeypatch.setattr(mcp_task_sessions, "settle_dead_task",
                        lambda root, task_id: swept.append((root, task_id)) or {"closure": closure})
    pump_once()
    assert swept == [(t.root, "old-task")]
    for frame in t.frames:
        events.dispatch_event(frame, t.ctx)
    assert load_task_result(t.root, "old-task")["status"] == "completed"


def test_unconfirmed_timeout_worker_death_does_not_sweep_bridge(timed_out, monkeypatch):
    from ouroboros import mcp_task_sessions

    swept = []
    monkeypatch.setattr(mcp_task_sessions, "settle_dead_task",
                        lambda *_a: swept.append(True))
    monkeypatch.setattr(task_reaper, "_kill_and_confirm_worker_dead", lambda *_a: False)
    monkeypatch.setattr(task_reaper, "_hold_wedged_worker", lambda *_a: None)
    pump_once()
    assert swept == []
    assert load_task_result(timed_out.root, "old-task")["status"] == "running"


@pytest.mark.parametrize("replacement", [False, True])
def test_deferred_timeout_recovers_files_without_replacing_a_new_busy_worker(timed_out, monkeypatch, replacement):
    t = timed_out
    with monkeypatch.context() as fault:
        fault.setattr(headless, "copy_child_task_result", lambda *_a: (_ for _ in ()).throw(OSError("disk refused")))
        pump_once()
    assert task_reaper._deferred_reap_jobs == [t.job] and t.frames == [] and t.spawned == []
    if replacement:
        newer = SimpleNamespace(wid=7, busy_task_id="new-task", reaping=False,
                                proc=SimpleNamespace(is_alive=lambda: True))
        t.workers.WORKERS[7] = newer
        t.q.RUNNING["new-task"] = {"task": {"id": "new-task", "_attempt": 1}, "worker_id": 7, "attempt": 1}
    task_reaper._retry_deferred_reap_jobs()
    pump_once()
    assert task_reaper._deferred_reap_jobs == [] and t.q.PENDING == []
    for frame in t.frames:
        events.dispatch_event(frame, t.ctx)
    assert load_task_result(t.root, "old-task")["result"] == "Saved old answer"
    assert [e["status"] for e in t.pushed if e.get("type") == "task_done"] == ["completed"]
    if replacement:
        assert t.spawned == [] and t.workers.WORKERS[7] is newer
        assert newer.busy_task_id == "new-task" and not newer.reaping and "new-task" in t.q.RUNNING
    else:
        assert t.spawned == [(7, t.old)]  # Actual respawn owner selected the original.


def test_same_id_new_attempt_prevents_stale_file_adoption(timed_out, monkeypatch):
    t = timed_out
    newer = SimpleNamespace(wid=7, busy_task_id="old-task", reaping=False,
                            proc=SimpleNamespace(is_alive=lambda: True))
    t.workers.WORKERS[7] = newer
    t.q.RUNNING["old-task"] = {"task": {"id": "old-task", "_attempt": 2}, "worker_id": 7, "attempt": 2}
    monkeypatch.setattr(headless, "prepare_terminal_task_files", lambda *_a: pytest.fail("stale source adopted"))
    pump_once()
    assert t.frames == [] and t.spawned == [] and newer.busy_task_id == "old-task"
    assert load_task_result(t.root, "old-task")["status"] == "running"


def test_changed_root_recovers_only_the_original_files(timed_out, monkeypatch):
    t = timed_out
    newer_root = t.root / "other-install"
    monkeypatch.setattr(t.q, "DRIVE_ROOT", newer_root)
    monkeypatch.setattr(t.workers, "DRIVE_ROOT", newer_root)
    newer = SimpleNamespace(wid=7, busy_task_id="new-task", reaping=False,
                            proc=SimpleNamespace(is_alive=lambda: True))
    t.workers.WORKERS[7] = newer
    pump_once()
    assert load_task_result(t.root, "old-task")["result"] == "Saved old answer"
    assert load_task_result(newer_root, "old-task") is None
    assert t.spawned == [] and t.frames == [] and t.workers.WORKERS[7] is newer


def test_generation_change_during_join_cannot_control_replacement(timed_out):
    t = timed_out
    newer = SimpleNamespace(wid=7, busy_task_id="new-task", reaping=False,
                            proc=SimpleNamespace(is_alive=lambda: True))
    def joined(**_kwargs):
        t.workers.WORKERS[7] = newer
    t.old.proc.join = joined
    pump_once()
    for frame in t.frames:
        events.dispatch_event(frame, t.ctx)
    assert t.spawned == [] and newer.busy_task_id == "new-task" and not newer.reaping
    assert load_task_result(t.root, "old-task")["result"] == "Saved old answer"


class StopAfterRetirement(BaseException):
    pass


@pytest.mark.parametrize("replacement", [False, True])
def test_captured_timeout_job_closes_only_its_dead_local_invocation(timed_out, monkeypatch, replacement):
    from ouroboros.task_results import load_task_result, write_task_result
    from supervisor import task_reaper

    t = timed_out
    # The REAL timeout producer already captured t.job, removed RUNNING and
    # cleared the slot's busy id. The retained fake Process is positively dead.
    assert t.job["meta"]["task"]["id"] == "old-task"
    assert "old-task" not in t.q.RUNNING and t.old.busy_task_id is None
    t.old.proc.pid, t.old.proc.exitcode = 880001, -9
    t.old.process_birth = "owned-worker-birth"
    claim = {"tool": "local-tool", "state": "claimed", "task_id": "old-task",
             "root_task_id": "old-task", "local_owner": {
                 "pid": 880001, "process_birth": "owned-worker-birth", "task_attempt": 1}}
    write_task_result(t.root, "old-task", "running", launch_handoffs={"op": claim})
    monkeypatch.setattr("supervisor.worker_pool_lifecycle.kill_worker_tree", lambda *_a, **_kw: None)
    monkeypatch.setattr(t.workers, "_reconcile_confirmed_dead_review_owner", lambda *_a: None)
    # Stop after the real retirement seam, before unrelated terminal copyback.
    monkeypatch.setattr("ouroboros.tools.services.archive_task_service_logs",
                        lambda *_a, **_kw: (_ for _ in ()).throw(StopAfterRetirement()))
    if replacement:
        new = SimpleNamespace(wid=7, busy_task_id="old-task", reaping=False,
                              proc=SimpleNamespace(is_alive=lambda: True), process_birth="new-birth")
        t.workers.WORKERS[7] = new
        t.q.RUNNING["old-task"] = {"task": {"id": "old-task", "_attempt": 2},
                                    "worker_id": 7, "attempt": 2}
        task_reaper.reap_timed_out_task(t.job)
        assert load_task_result(t.root, "old-task")["launch_handoffs"] == {"op": claim}
        assert t.workers.WORKERS[7] is new and new.busy_task_id == "old-task"
    else:
        with pytest.raises(StopAfterRetirement):
            task_reaper.reap_timed_out_task(t.job)
        row = load_task_result(t.root, "old-task")
        assert not row["launch_handoffs"], row
        assert row["retired_tool_invocations"]["op"]["effect_outcome"] == "unknown"
        assert row["retired_tool_invocations"]["op"]["replay_authorized"] is False
