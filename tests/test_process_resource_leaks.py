"""Regression tests for process/resource leak fixes (PR-C).

Covers:
  * #9  — reap orphaned worker process groups from a prior server instance.
  * #4/#6 — respawn closes the old worker queue under the queue lock.
  * #7  — emergency cleanup joins killed children.
  * #3  — a cancelled subagent's child drive is removed immediately.
"""

from __future__ import annotations

import pathlib

REPO = pathlib.Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


# ───────────────────────── #3: subagent drive cleanup ───────────────────────

def test_remove_subagent_task_drive(tmp_path):
    from ouroboros.headless import (
        HEADLESS_TASKS_DIR,
        TASK_DRIVES_DIR,
        remove_subagent_task_drive,
    )

    tid = "abcd1234"
    headless_dir = tmp_path / HEADLESS_TASKS_DIR / tid / "data"
    drive_dir = tmp_path / TASK_DRIVES_DIR / tid
    headless_dir.mkdir(parents=True)
    drive_dir.mkdir(parents=True)

    assert remove_subagent_task_drive(tmp_path, tid) is True
    assert not (tmp_path / HEADLESS_TASKS_DIR / tid).exists()
    assert not (tmp_path / TASK_DRIVES_DIR / tid).exists()

    # idempotent / no error when nothing to remove
    assert remove_subagent_task_drive(tmp_path, tid) is False
    # invalid task id is rejected, not raised
    assert remove_subagent_task_drive(tmp_path, "../escape") is False


def test_remove_task_scratch_never_promotes_forged_terminal_result(tmp_path):
    from ouroboros.headless import TASK_DRIVES_DIR, remove_subagent_task_drive
    from ouroboros.task_results import STATUS_CANCELLED, load_task_result, write_task_result

    tid = "directchild"
    write_task_result(
        tmp_path,
        tid,
        STATUS_CANCELLED,
        parent_task_id="parent1",
        root_task_id="parent1",
        delegation_role="subagent",
    )
    child_drive = tmp_path / TASK_DRIVES_DIR / tid
    write_task_result(
        child_drive,
        tid,
        "completed",
        result="late complete result",
        trace_summary="late trace",
        artifact_status="ready",
        artifacts=[],
    )

    assert remove_subagent_task_drive(tmp_path, tid) is True
    stored = load_task_result(tmp_path, tid) or {}
    assert "terminal_child_result_snapshot" not in stored
    assert not child_drive.exists()


def test_remove_subagent_drive_does_not_promote_custom_late_result(tmp_path):
    from ouroboros.headless import TASK_DRIVES_DIR, remove_subagent_task_drive
    from ouroboros.task_results import STATUS_CANCELLED, load_task_result, write_task_result

    tid = "customchild"
    custom_child = tmp_path / "isolated-child-data"
    write_task_result(
        custom_child,
        tid,
        "completed",
        result="authoritative late result",
        trace_summary="late trace",
        artifact_status="ready",
        artifacts=[],
    )
    write_task_result(
        tmp_path,
        tid,
        STATUS_CANCELLED,
        parent_task_id="parent1",
        root_task_id="parent1",
        delegation_role="subagent",
        child_drive_root=str(custom_child),
    )
    scratch = tmp_path / TASK_DRIVES_DIR / tid
    scratch.mkdir(parents=True)

    assert remove_subagent_task_drive(tmp_path, tid) is True
    stored = load_task_result(tmp_path, tid) or {}
    assert stored["status"] == STATUS_CANCELLED
    assert "terminal_child_result_snapshot" not in stored
    assert "authoritative late result" not in str(stored)
    assert not scratch.exists()


def test_cancel_running_subagent_removes_drive_source():
    # The cancellation custody family lives in supervisor/cancel_custody.py; its
    # settlement PUBLICATION half (where the drive cleanup runs) was split into
    # supervisor/cancel_publication.py at the module-size boundary.
    custody_src = _read("supervisor/cancel_custody.py")
    publish_src = _read("supervisor/cancel_publication.py")
    assert "remove_subagent_task_drive(q.DRIVE_ROOT, str(task_id))" in publish_src
    assert "delegation_role" in publish_src  # gated on subagent role
    # ORDER matters: the drive may only be reclaimed after the process is confirmed
    # dead, or a still-running worker loses its scratch out from under it. The
    # death confirmation lives in custody, which only then calls the publish
    # step; inside the publish step the cleanup follows the terminal emit.
    assert "survived kill escalation" in custody_src
    assert custody_src.index("survived kill escalation") < custody_src.index(
        "_publish_cancelled_task(\n"
    )
    assert publish_src.index("_emit_cancel_task_done") < publish_src.index(
        "remove_subagent_task_drive"
    )


# ───────────────────────── #9: orphan worker reaping ────────────────────────

def test_reap_orphaned_workers(tmp_path, monkeypatch):
    import ouroboros.platform_layer as pl
    from ouroboros.utils import atomic_write_json
    from supervisor import workers

    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path, raising=False)
    workers.WORKERS.clear()
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "state").mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        workers._worker_pids_path(),
        {"server_pid": 999999, "workers": [{"pid": 111}, {"pid": 222}, {"pid": 333}]},
    )

    # 111 = alive + ours (own session leader) → must be killed
    # 222 = dead (empty cmdline) → skipped
    # 333 = alive but unrelated process (pid reused) → skipped
    cmds = {111: "python -B -c multiprocessing.spawn", 222: "", 333: "/usr/bin/SomethingElse"}
    monkeypatch.setattr(pl, "process_command", lambda pid: cmds.get(int(pid), ""))
    monkeypatch.setattr(pl, "process_group_id", lambda pid: int(pid))  # session leader
    killed_groups, killed_pids = [], []
    monkeypatch.setattr(pl, "kill_process_group_id", lambda pgid: killed_groups.append(int(pgid)))
    monkeypatch.setattr(pl, "force_kill_pid", lambda pid: killed_pids.append(int(pid)))

    n = workers.reap_orphaned_workers()

    assert n == 1
    assert killed_pids == [111]
    assert killed_groups == [111]


def test_record_worker_pids_roundtrip(tmp_path, monkeypatch):
    from ouroboros.utils import read_json_dict
    from supervisor import workers

    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path, raising=False)
    (tmp_path / "state").mkdir(parents=True, exist_ok=True)

    class _FakeProc:
        def __init__(self, pid):
            self.pid = pid

    monkeypatch.setattr(
        workers, "WORKERS",
        {0: workers.Worker(wid=0, proc=_FakeProc(4242), in_q=None)},
        raising=False,
    )
    workers._record_worker_pids()
    data = read_json_dict(workers._worker_pids_path()) or {}
    assert {"pid": 4242} in (data.get("workers") or [])


# ───────────────────── #4/#6 + #7: source contracts ─────────────────────────

def test_respawn_closes_old_queue_under_lock():
    # respawn moved to the pool-lifecycle owner; the pool still spawns.
    src = _read("supervisor/worker_pool_lifecycle.py")
    assert "with _queue_lock:" in src
    assert "old.in_q.close()" in src
    assert "old.in_q.cancel_join_thread()" in src


def test_spawn_reaps_orphans_and_records_pids():
    src = _read("supervisor/workers.py")
    assert "reap_orphaned_workers()" in src
    assert "_record_worker_pids()" in src
    # reap guards against PID reuse and only group-kills its own setsid session
    assert "if pgid and pgid == pid:" in _read("supervisor/worker_pool_lifecycle.py")


def test_emergency_cleanup_joins_children():
    src = _read("server.py")
    # force_kill is followed by a join so the Process object is reaped
    assert "child.join(timeout=2)" in src
