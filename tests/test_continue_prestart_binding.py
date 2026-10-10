"""A Continue that never started hands its folder and deadline to its own Continue.

The real chain, with nothing written into a row by the test: an external-folder
root A is Continued as B; B is dispatched to a worker that never starts it, and
the graceful shutdown's own kill (application-stop retention included) cancels
B as ``server_shutdown`` — a B still merely queued would be retained instead, so
this is the door that makes a never-started B Continuable. The owner Continues
B as C; the next stop retains C and the boot's snapshot restore brings it back.
C must run in A's folder under A's deadline. The replay variant loses B's
result row after admission and lets the same nonce recover it.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests._budget_pause_exact_helpers import _install_queue
from tests.test_owner_continue import NONCE, _interrupted

DEADLINE = "2099-01-01T00:00:00+00:00"  # the one ``_interrupted`` gives A


def _shutdown(workers):
    """The lifespan teardown's kill (``server.py``): its cleanup args and its retention."""
    import server
    from ouroboros.server_restart import _shutdown_task_cleanup_args

    status, reason = _shutdown_task_cleanup_args(restart_requested=False)
    assert workers.kill_workers(force=True, terminal_status=status, result_reason=reason,
                                stop_source="server_shutdown", **server._restart_cleanup_kwargs())


def _binding(row):
    return row.get("workspace_root"), row.get("workspace_mode"), row.get("deadline_at")


@pytest.mark.parametrize("replay", [False, True])
def test_never_started_continue_keeps_folder_and_deadline_for_its_continue(tmp_path, monkeypatch, replay):
    from ouroboros.task_results import load_task_result, task_result_path
    from supervisor.continuation_admission import admit_continuation

    queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(workers, "_EVENT_Q_SHUTDOWN", False, raising=False)
    monkeypatch.setattr(workers, "get_event_q", lambda: SimpleNamespace(put=lambda _event: None), raising=False)
    folder = tmp_path / "external-folder"
    folder.mkdir()
    expected = (str(folder), "external", DEADLINE)
    _interrupted(tmp_path, workspace_root=str(folder), workspace_mode="external")

    b = admit_continuation("pred-1", action_nonce=NONCE)["successor_task_id"]
    assert _binding(load_task_result(tmp_path, b)) == expected
    if replay:
        # A stop between the queue snapshot and B's result write: the same nonce recovers it.
        task_result_path(tmp_path, b).unlink()
        recovered = admit_continuation("pred-1", action_nonce=NONCE)
        assert recovered["recovered"] is True and recovered["successor_task_id"] == b
        assert _binding(load_task_result(tmp_path, b)) == expected

    sent = []
    workers.WORKERS[0] = SimpleNamespace(
        wid=0, busy_task_id=None, reaping=False, in_q=SimpleNamespace(put=sent.append),
        proc=SimpleNamespace(pid=None, exitcode=None, is_alive=lambda: False, join=lambda timeout=None: None))
    workers.assign_tasks()
    assert [task["id"] for task in sent] == [b] and b in workers.RUNNING  # handed over, never started
    _shutdown(workers)
    stopped = load_task_result(tmp_path, b)
    assert stopped["status"] == "cancelled" and stopped["cancel_origin"]["source"] == "server_shutdown"
    assert _binding(stopped) == expected

    c_ack = admit_continuation(b, action_nonce=NONCE)
    assert c_ack["ok"] is True, c_ack
    c = c_ack["successor_task_id"]
    admitted = next(task for task in workers.PENDING if task["id"] == c)
    assert _binding(admitted) == expected
    assert _binding(load_task_result(tmp_path, c)) == expected
    _shutdown(workers)
    assert [task["id"] for task in workers.PENDING] == [c]  # accepted work is retained, not cancelled
    workers.PENDING.clear()
    assert queue.restore_pending_from_snapshot() == 1
    restored = workers.PENDING[0]
    assert restored["id"] == c and _binding(restored) == expected
