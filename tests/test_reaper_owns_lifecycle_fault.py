"""A refused ``task_done`` racing the reaper belongs to the reaper, not the fault resolver.

CyberGym r8 (2026-09-04): at the 2 h wall the reaper popped a task out of
RUNNING and killed its worker; the worker's own ``task_done`` (completed) landed
in that window and was refused as a lifecycle fault. The resolver terminalized
the task as ``failed`` — clobbering the completed result the reaper's post-kill
re-check would have mirrored. 4 solved tasks were published as
``task_done_lifecycle_fault``.
"""

from __future__ import annotations

import types

from ouroboros.task_results import load_task_result
from ouroboros.utils import append_jsonl
from supervisor import task_reaper
from supervisor.events import _resolve_lifecycle_fault


def _ctx(tmp_path, running, slot):
    return types.SimpleNamespace(
        DRIVE_ROOT=tmp_path,
        RUNNING=running,
        WORKERS={0: slot},
        append_jsonl=append_jsonl,
        persist_queue_snapshot=lambda **_kw: True,
    )


def _reset():
    task_reaper._forget_task_reaping("r8-task")
    task_reaper._forget_reaper_publishing("r8-task")


def test_fault_resolver_defers_while_the_reaper_is_mid_kill(tmp_path):
    _reset()
    running = {"r8-task": {"task": {"id": "r8-task"}}}
    slot = types.SimpleNamespace(busy_task_id="r8-task", reaping=True)
    task_reaper.note_task_reaping("r8-task")
    try:
        _resolve_lifecycle_fault(
            {"task_id": "r8-task", "status": "completed", "worker_id": 0},
            _ctx(tmp_path, running, slot), "completed", detail="durable-result fault",
        )
        assert "r8-task" in running
        assert load_task_result(tmp_path, "r8-task") in (None, {})
    finally:
        _reset()


def test_fault_resolver_defers_until_the_reaper_has_published(tmp_path):
    """Confirmed death clears reaping custody (the acceptance-fence predicate
    needs that) but the reaper still owns the terminal row until its job ends."""
    _reset()
    running = {"r8-task": {"task": {"id": "r8-task"}}}
    slot = types.SimpleNamespace(busy_task_id="r8-task", reaping=True)
    task_reaper.note_task_reaping("r8-task")
    workers_mod = types.SimpleNamespace(_reconcile_confirmed_dead_review_owner=lambda pid: None)
    try:
        task_reaper._release_confirmed_dead_acceptance_owner(
            workers_mod, types.SimpleNamespace(pid=4242), "r8-task",
        )
        assert task_reaper.task_reaping_in_progress("r8-task") is False
        assert task_reaper.reaper_owns_task_row("r8-task") is True

        _resolve_lifecycle_fault(
            {"task_id": "r8-task", "status": "completed", "worker_id": 0},
            _ctx(tmp_path, running, slot), "completed", detail="durable-result fault",
        )
        assert "r8-task" in running
        assert load_task_result(tmp_path, "r8-task") in (None, {})

        # The reaper loop's job boundary ends terminal-row ownership.
        task_reaper._forget_reaper_publishing("r8-task")
        assert task_reaper.reaper_owns_task_row("r8-task") is False
    finally:
        _reset()


def test_unowned_fault_is_still_terminalized(tmp_path):
    _reset()
    running = {"r8-task": {"task": {"id": "r8-task"}}}
    slot = types.SimpleNamespace(busy_task_id="r8-task", reaping=False)
    _resolve_lifecycle_fault(
        {"task_id": "r8-task", "status": "running", "worker_id": 0},
        _ctx(tmp_path, running, slot), "running",
    )
    assert "r8-task" not in running
    stored = load_task_result(tmp_path, "r8-task") or {}
    assert stored.get("reason_code") == "task_done_lifecycle_fault"


def test_reaper_loop_job_boundary_forgets_publication_even_on_failure(monkeypatch):
    _reset()
    import queue as _queue

    jobs: _queue.Queue = _queue.Queue()
    jobs.put({"task_id": "r8-task"})
    monkeypatch.setattr(task_reaper, "reap_queue", jobs)

    def boom(job):
        task_reaper._note_reaper_publishing(job["task_id"])
        raise RuntimeError("publication failed")

    monkeypatch.setattr(task_reaper, "reap_timed_out_task", boom)

    class _Stop(BaseException):  # the loop swallows Exception from get()
        pass

    real_get = jobs.get
    calls = {"n": 0}

    def get_once(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] > 1:
            raise _Stop()
        return real_get(*args, **kwargs)

    monkeypatch.setattr(jobs, "get", get_once)
    try:
        task_reaper.reaper_loop()
    except _Stop:
        pass
    assert task_reaper.reaper_owns_task_row("r8-task") is False
