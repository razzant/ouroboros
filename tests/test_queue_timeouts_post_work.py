"""The solve ceiling does not cut a settled root's running post-work (chapter 05),
including a split Project root whose settled row has not been copied back yet."""

import queue as stdqueue
import time

import pytest

from ouroboros.task_results import write_task_result
from supervisor import queue, workers
from tests.test_owner_wait_pool import pool as _pool  # noqa: F401 - fixture reuse (RUNNING/worker pool)

pool = _pool  # noqa: F811 - pytest fixture re-export


@pytest.mark.parametrize("settled", [True, False])
def test_split_root_post_work_is_exempt_from_the_solve_ceiling_once_its_actor_drive_settled(
    pool, monkeypatch, settled,
):
    """Finding 7: the exemption read solve settlement from the CANONICAL row, but a
    split root settles its actor drive first and the canonical row keeps ``running``
    until copyback, so the root could still be reaped on the solve ceiling during
    post-work. Settlement is read from the actor's drive; the post-work phase from
    the canonical checkpoint authority. An unsettled root still hits the ceiling."""
    actor = pool.root / "child-drive"
    actor.mkdir()
    pool.meta["task"].update({"root_task_id": "owner", "drive_root": str(actor),
                              "budget_drive_root": str(pool.root)})
    pool.meta["last_progress_at"] = time.time()  # post-work is progressing; only the ceiling is at stake
    # Canonical: still `running` until copyback, but it IS the checkpoint authority.
    write_task_result(pool.root, "owner", "running",
                      root_phase_checkpoint={"post_task_synthesis": "running"})
    write_task_result(actor, "owner", "completed" if settled else "running", result="answer")
    monkeypatch.setattr(queue, "get_task_idle_timeout_sec", lambda: 10)
    monkeypatch.setattr(queue, "get_per_call_timeout_ceiling_sec", lambda: 1)
    monkeypatch.setattr(queue, "get_task_abs_ceiling_sec", lambda: 1000)  # runtime is ~4000s
    monkeypatch.setattr(queue, "FINALIZATION_GRACE_SEC", 0)
    monkeypatch.setattr(queue, "_ensure_reaper_started", lambda: None)
    reaps = stdqueue.Queue()
    monkeypatch.setattr(queue, "_reap_queue", reaps)
    queue._enforce_task_timeouts_locked(workers, time.time(), 0, {})
    if settled:
        assert reaps.empty() and "owner" in workers.RUNNING
    else:
        assert reaps.get_nowait()["terminal_reason"] == "absolute_ceiling"
