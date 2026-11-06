"""Host teardown must reach the real terminal consumer after file preparation."""

import json
import queue
from types import SimpleNamespace

import pytest

from ouroboros import headless
from ouroboros.task_results import load_task_result, write_task_result
from supervisor import events_task_done, worker_health, workers
from tests.test_task_done_durable_vs_transport import _mk_ctx


@pytest.fixture
def terminal_host(tmp_path, monkeypatch):
    from supervisor import queue as tasks, state, task_reaper

    tasks.init(tmp_path)
    state.init(tmp_path)
    sent, pushed, respawned, events = [], [], [], []
    ctx = _mk_ctx(tmp_path, sent, pushed)
    proc = SimpleNamespace(pid=0, exitcode=-11, is_alive=lambda: False)
    worker = SimpleNamespace(wid=0, proc=proc, busy_task_id="host-terminal",
                             reaping=False, readiness_exhausted=False, active_capacity=True)
    for module in (tasks, workers):
        monkeypatch.setattr(module, "RUNNING", ctx.RUNNING)
        monkeypatch.setattr(module, "DRIVE_ROOT", tmp_path)
    ctx.WORKERS[0] = worker
    monkeypatch.setattr(workers, "WORKERS", ctx.WORKERS)
    monkeypatch.setattr(workers, "PENDING", ctx.PENDING)
    monkeypatch.setattr(tasks, "PENDING", ctx.PENDING)
    monkeypatch.setattr(workers, "REPO_DIR", tmp_path / "repo")
    monkeypatch.setattr(workers, "_LAST_SPAWN_TIME", 0)
    monkeypatch.setattr(workers, "_WORKER_POOL_DISABLED_REASON", "")
    monkeypatch.setattr(workers, "_reconcile_confirmed_dead_review_owner", lambda *_a: None)
    monkeypatch.setattr(workers, "respawn_worker", lambda wid: respawned.append(wid))
    monkeypatch.setattr(workers, "send_with_budget", ctx.send_with_budget)
    monkeypatch.setattr(workers, "load_state", lambda: {})
    jobs = queue.Queue()
    monkeypatch.setattr(tasks, "_reap_queue", jobs)
    monkeypatch.setattr(task_reaper, "reap_queue", jobs)
    monkeypatch.setattr(task_reaper, "_deferred_reap_jobs", [])
    monkeypatch.setattr(tasks, "_ensure_reaper_started", lambda: None)

    class ImmediateDrain:
        def put(self, event):
            events.append(dict(event))
            # The consumer may run before put returns. No timing sleeps or
            # emitted-frame-only assertions can hide the former handoff race.
            events_task_done._handle_task_done(event, ctx)

    monkeypatch.setattr(workers, "get_event_q", ImmediateDrain)
    return SimpleNamespace(ctx=ctx, worker=worker, jobs=jobs, events=events,
                           pushed=pushed, respawned=respawned, root=tmp_path)


@pytest.mark.parametrize(("split", "task_type", "exitcode", "attempt", "status"), [
    (False, "task", -11, 2, "failed"),  # a signal death at the last attempt: the retry bound, not the signal, is terminal
    (True, "deep_self_review", -11, 2, "failed"),
    (True, "evolution", 1, 1, "cancelled"),
])
def test_crash_terminal_reaches_durable_and_live_consumers(terminal_host, task_type, exitcode, attempt, status, split):
    host = terminal_host
    task = {"id": "host-terminal", "type": task_type, "chat_id": 0, "_attempt": attempt}
    if split:
        child = host.root / "state/headless_tasks/host-terminal/data"
        task["drive_root"] = str(child)
        write_task_result(child, task["id"], "running", result="unfinished")
    write_task_result(host.root, task["id"], "running", chat_id=0,
                      **({"child_drive_root": str(child)} if split else {}))
    host.worker.proc.exitcode = exitcode
    host.ctx.RUNNING[task["id"]] = {"task": task, "worker_id": 0, "attempt": attempt}
    workers.ensure_workers_healthy()
    job = host.jobs.get_nowait()
    assert job["kind"] == "confirmed_dead_worker"
    worker_health.recover_confirmed_dead_worker(job)
    assert not host.ctx.RUNNING and host.jobs.empty()
    assert host.respawned == [0]
    assert load_task_result(host.root, task["id"], strict=True)["status"] == status
    rows = [json.loads(line) for line in (host.root / "logs/events.jsonl").read_text().splitlines()]
    assert len([row for row in rows if row.get("type") == "task_done"]) == 1
    assert [row["status"] for row in host.pushed if row.get("type") == "task_done"] == [status]


def test_legacy_host_frame_recovery_keeps_captured_worker_identity(terminal_host):
    host = terminal_host
    task = {"id": "host-terminal", "type": "task", "chat_id": 0, "_attempt": 1}
    write_task_result(host.root, task["id"], "completed", result="kept", artifact_status="ready")
    host.ctx.RUNNING[task["id"]] = {"task": task, "worker_id": 0, "attempt": 1}
    # This represents a host frame already queued before withdrawal completed.
    events_task_done._handle_task_done({"type": "task_done", "task_id": task["id"],
                                       "status": "completed", "chat_id": 0}, host.ctx)
    job = host.jobs.get_nowait()
    assert job["kind"] == "terminal_file_recovery"
    worker_health._recover_terminal_files(job)
    assert host.events[-1]["worker_id"] == 0
    assert not host.ctx.RUNNING and host.jobs.empty()
    assert [row["status"] for row in host.pushed if row.get("type") == "task_done"] == ["completed"]
    assert headless.terminal_task_files_ready(host.root, task, load_task_result(host.root, task["id"]))


def test_crash_terminal_write_failure_keeps_the_same_recovery_owner(terminal_host, monkeypatch):
    from ouroboros import task_results
    from supervisor.task_reaper import TerminalFileRecoveryPending

    host = terminal_host
    task = {"id": "host-terminal", "type": "task", "chat_id": 0, "_attempt": 2}  # its one retry spent: terminal
    write_task_result(host.root, task["id"], "running")
    meta = {"task": task, "worker_id": 0, "attempt": 2}
    host.ctx.RUNNING[task["id"]] = meta
    workers.ensure_workers_healthy()
    job = host.jobs.get_nowait()
    original_write = task_results.write_task_result

    def unavailable(root, task_id, status, **fields):
        if status == "failed":
            raise OSError("terminal storage temporarily unavailable")
        return original_write(root, task_id, status, **fields)

    monkeypatch.setattr(task_results, "write_task_result", unavailable)
    with pytest.raises(TerminalFileRecoveryPending):
        worker_health.recover_confirmed_dead_worker(job)
    assert host.ctx.RUNNING[task["id"]] is meta
    assert not host.events and not host.respawned
    assert load_task_result(host.root, task["id"])["status"] == "running"
    monkeypatch.setattr(task_results, "write_task_result", original_write)
    worker_health.recover_confirmed_dead_worker(job)
    assert not host.ctx.RUNNING and host.respawned == [0]
    assert [row["status"] for row in host.pushed if row.get("type") == "task_done"] == ["failed"]
