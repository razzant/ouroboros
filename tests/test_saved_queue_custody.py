"""The saved queue across a generation that has not restored it yet (razzant/ouroboros#1952).

Until a restore read ``queue_snapshot.json`` and decided every row, or the process published one
itself, no publication of a process that holds no work replaces a snapshot that still names pending
or running rows, and none replaces it while another thread restores it (``supervisor/queue_snapshot.
_keeps_saved_queue``): a stop, Update or Rollback during the boot keeps every saved Pause for the next
generation, whose Resume finds it.
"""
from __future__ import annotations

import json
import pathlib

import pytest

from tests._budget_pause_exact_helpers import _install_queue, _parked


@pytest.fixture(autouse=True)
def _a_generation_holding_nothing_of_its_own(monkeypatch):
    """Before its restore a generation holds no work: the direct-turn registry and the fence maps
    are this test's own, never ones an earlier test in the same worker left behind."""
    from supervisor import active_activity, queue, queue_snapshot

    registry = active_activity.DirectActivityRegistry()
    monkeypatch.setattr(active_activity, "get_direct_activity_registry", lambda: registry)
    monkeypatch.setattr(queue, "ACCEPTANCE_FENCES", {})
    monkeypatch.setattr(queue, "BUDGET_ROOT_FENCES", {})
    monkeypatch.setattr(queue_snapshot, "_RESTORED", set())
    monkeypatch.setattr(queue_snapshot, "_RESTORING", {})


@pytest.mark.parametrize("door", ["kill_workers", "publication", "update_quiesce", "rollback_aborted"])
def test_a_stop_before_the_queue_restore_keeps_every_saved_pause_for_resume(tmp_path, monkeypatch, door):
    """#1952: a generation that stops before its supervisor restored the queue holds no work of its own
    (a stop door during the boot, Update or Rollback pressed then, a boot without a provider); no
    publication of it may replace the saved snapshot with that emptiness, so the next generation
    restores the saved Pause and the owner's Resume finds it. An aborted door still readmits the pool.
    Once a generation restored the queue, an emptied queue is published as before."""
    from ouroboros import process_custody
    from ouroboros.gateway import control
    from ouroboros.tools import services
    from supervisor import git_ops, queue_snapshot

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda _st, **_k: 5.0)
    monkeypatch.setattr(queue_snapshot, "_RESTORED", set())
    _parked(tmp_path, monkeypatch, task_id="saved-root")

    def saved_ids():
        return [row["id"] for row in json.loads(queue.QUEUE_SNAPSHOT_PATH.read_text())["pending"]]

    def shutdown():
        workers.kill_workers(force=True, terminal_status="cancelled", result_reason="Server shutdown.",
                             stop_source="server_shutdown", retain_saved_work=True, preserve_pending=True)

    def stop():
        if door == "kill_workers":
            shutdown()
        elif door == "publication":
            assert queue.persist_queue_snapshot(reason="pre_restart_exit") is None
        elif door == "update_quiesce":  # the Update door's pool stop, then the restart exit's shutdown
            assert workers.kill_workers_for_update(result_reason="Owner requested update") == []
            shutdown()
        else:  # the Rollback door stops the pool, the checkout switch fails, the pool is readmitted
            monkeypatch.setattr(git_ops, "DRIVE_ROOT", tmp_path, raising=False)
            monkeypatch.setattr(services, "kill_all_services", lambda *_a, **_k: [])
            monkeypatch.setattr(process_custody, "quiesce_custodied_services", lambda *_a, **_k: (True, []))
            started = []
            monkeypatch.setattr(workers, "ensure_worker_pool_started", lambda **kw: started.append(kw) or True)
            try:
                assert control._quiesce_repo_writers("manual_rollback") == []
                control._respawn_workers_after_failed_update()
            finally:
                workers.open_repo_writer_admission()
            assert started, "the aborted door readmits the pool"

    shutdown()
    assert saved_ids() == ["saved-root"]  # generation A stopped with its saved Pause
    workers.PENDING[:] = []
    queue.init(tmp_path)  # generation B: fresh maps, nothing published, nothing restored
    stop()
    assert saved_ids() == ["saved-root"], "an unrestored empty queue never replaces the saved one"
    queue.init(tmp_path)  # generation C restores it
    assert queue.restore_pending_from_snapshot() == 1
    assert queue.resume_budget_paused_task("saved-root")["ok"] is True
    workers.PENDING[:] = []  # its work ended: an emptied queue it restored is published
    assert queue.persist_queue_snapshot(reason="test") is True and saved_ids() == []


@pytest.mark.parametrize("boundary", ["stopped_during_the_restore", "snapshot_unreadable_at_the_restore"])
def test_a_restore_that_did_not_finish_grants_no_custody_of_the_saved_queue(tmp_path, monkeypatch, boundary):
    """The saved queue passes into this process's custody only when a restore read it and decided every
    row: a stop while another thread restores it (rows not adopted yet), or a restore that could not
    read it (a transient error), leaves the snapshot to the next generation, whose Resume finds it."""
    import threading

    from supervisor import queue_snapshot
    from ouroboros import delegate_recovery

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda _st, **_k: 5.0)
    monkeypatch.setattr(queue_snapshot, "_RESTORED", set())
    _parked(tmp_path, monkeypatch, task_id="saved-root")
    assert queue.persist_queue_snapshot(reason="test") is True
    saved = queue.QUEUE_SNAPSHOT_PATH.read_bytes()
    workers.PENDING[:] = []
    queue.init(tmp_path)  # the next generation, before its restore

    def stop():
        workers.kill_workers(force=True, terminal_status="cancelled", result_reason="Server shutdown.",
                             stop_source="server_shutdown", retain_saved_work=True, preserve_pending=True)

    if boundary == "stopped_during_the_restore":
        entered, release = threading.Event(), threading.Event()
        real_door = delegate_recovery.boot_door

        class Stopped(BaseException):
            pass

        def held(root, **kwargs):
            if threading.current_thread().name == "restore":
                entered.set()
                assert release.wait(10)
                raise Stopped()  # the process exits before any row is adopted
            return real_door(root, **kwargs)

        monkeypatch.setattr(delegate_recovery, "boot_door", held)

        def restore():
            try:
                queue.restore_pending_from_snapshot()
            except Stopped:
                pass

        thread = threading.Thread(target=restore, name="restore")
        thread.start()
        try:
            assert entered.wait(10)
            stop()
        finally:
            release.set()
            thread.join(10)
        monkeypatch.setattr(delegate_recovery, "boot_door", real_door)
    else:
        real_read = pathlib.Path.read_text

        def unreadable(path, *args, **kwargs):
            if path == queue.QUEUE_SNAPSHOT_PATH:
                raise PermissionError("a transient read error")
            return real_read(path, *args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(pathlib.Path, "read_text", unreadable)
            assert queue.restore_pending_from_snapshot() == 0
        stop()
    assert queue.QUEUE_SNAPSHOT_PATH.read_bytes() == saved
    queue_snapshot._RESTORED.clear()
    queue.init(tmp_path)  # the following generation
    assert queue.restore_pending_from_snapshot() == 1
    assert queue.resume_budget_paused_task("saved-root")["ok"] is True


@pytest.mark.parametrize("orphan", [False, True], ids=["no_direct_turn", "caught_direct_turn"])
@pytest.mark.parametrize("read_failure", [False, True], ids=["read", "unreadable"])
def test_settling_a_caught_direct_turn_grants_no_custody_of_an_unread_snapshot(tmp_path, monkeypatch, orphan,
                                                                               read_failure):
    """The restore settles a direct turn the stop caught even when the snapshot could not be read; that
    hand-over to cancellation custody never certifies the unread saved queue, so the boot's next stop
    and startup publication keep it and the next generation's Resume finds the saved Pause."""
    from supervisor import queue_snapshot
    from tests.test_direct_roots_adoption import TID, _crashed_turn, _next_generation

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda _st, **_k: 5.0)
    _parked(tmp_path, monkeypatch, task_id="saved-root")
    assert queue.persist_queue_snapshot(reason="seed") is True
    if orphan:
        _crashed_turn(tmp_path)  # a durable direct execution owner naming the ended process
        _next_generation(monkeypatch, tmp_path)
    workers.PENDING.clear()
    queue.init(tmp_path)  # the next generation discovers the caught direct turn
    assert (TID in queue.PRIOR_DIRECT_ROOTS.get("task_ids", [])) is orphan
    real_read, fenced = pathlib.Path.read_text, []

    def transient(path, *args, **kwargs):
        if read_failure and path == queue.QUEUE_SNAPSHOT_PATH:
            raise OSError(5, "a transient read error")
        return real_read(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(pathlib.Path, "read_text", transient)
        queue.restore_pending_from_snapshot(terminalized=fenced)
    assert (TID in fenced) is orphan, "the caught direct turn still goes to cancellation custody"
    workers.kill_workers(force=True, terminal_status="cancelled", result_reason="Server shutdown.",
                         stop_source="server_shutdown", retain_saved_work=True, preserve_pending=True)
    queue.persist_queue_snapshot(reason="startup")
    workers.PENDING.clear()
    queue_snapshot._RESTORED.clear()
    queue.init(tmp_path)  # the following generation
    assert queue.restore_pending_from_snapshot() == 1
    assert queue.resume_budget_paused_task("saved-root")["ok"] is True


def test_a_continue_before_the_restore_waits_and_the_saved_queue_survives(tmp_path, monkeypatch):
    """A Continue admits a successor into this process's queue; before the process took the saved
    queue into custody that publication would replace it, and the restore that follows skips a
    queue already holding work. The Continue is refused as retryable instead; after the restore the
    same press is admitted and the saved Pause still resumes."""
    from supervisor.continuation_admission import admit_continuation
    from tests.test_owner_continue import NONCE, _interrupted

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda _st, **_k: 5.0)
    _parked(tmp_path, monkeypatch, task_id="saved-root")
    workers.kill_workers(force=True, terminal_status="cancelled", result_reason="Server shutdown.",
                         stop_source="server_shutdown", retain_saved_work=True, preserve_pending=True)
    saved = queue.QUEUE_SNAPSHOT_PATH.read_bytes()
    workers.PENDING[:] = []
    queue.init(tmp_path)  # the next generation serves its UI before its restore
    _interrupted(tmp_path, "pred-1")
    assert admit_continuation("pred-1", action_nonce=NONCE) == {"ok": False, "error": "queue_not_restored"}
    assert not workers.PENDING and queue.QUEUE_SNAPSHOT_PATH.read_bytes() == saved
    assert queue.restore_pending_from_snapshot() >= 1
    admitted = admit_continuation("pred-1", action_nonce=NONCE)
    assert admitted.get("ok") is True, admitted
    assert queue.resume_budget_paused_task("saved-root")["ok"] is True
