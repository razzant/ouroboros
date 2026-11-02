"""Drive a conversation-lane carrier inside the test process.

Production admits an owner turn or a wake to a lane worker (``supervisor.conversation_lane``)
and the worker runs ``worker_process.run_worker_task``. A test that wants the body to run takes
the dispatched task off the lane slot's queue and runs that same body here, synchronously, with
the actor it chose: the admission stays the real one, the carrier is this thread.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional

from ouroboros.task_carrier import LANE_CONVERSATION


def install_lane_spare(workers_module: Any, wid: int = 90, *, pid: int = 2147480000) -> Any:
    """A READY idle conversation carrier in the pool table, as the boot spare is."""
    from tests.worker_pool_support import affirm_ready

    class _Queue(list):
        def put(self, item: Any) -> None:
            self.append(item)

    slot = workers_module.Worker(wid, SimpleNamespace(pid=pid + wid, is_alive=lambda: True), _Queue(),
                                 lane=LANE_CONVERSATION)
    affirm_ready(slot)
    workers_module.WORKERS[wid] = slot
    return slot


def dispatched_tasks(slot: Any) -> List[Dict[str, Any]]:
    """The tasks the admission put on this carrier's queue, oldest first (the queue is drained)."""
    queued = slot.in_q
    if isinstance(queued, list):
        tasks, queued[:] = list(queued), []
        return tasks
    tasks = []
    while True:
        try:
            tasks.append(queued.get_nowait())
        except Exception:
            return tasks


def carry(workers_module: Any, slot: Any, actor: Any, *, out_q: Any = None,
          on_failure: Optional[Callable[[Any, BaseException], bool]] = None) -> List[Dict[str, Any]]:
    """Run every task dispatched to ``slot`` through the real worker body with ``actor``, here.

    ``out_q`` defaults to the pool's event bus (``workers.get_event_q()``). The slot's
    ``busy_task_id`` is released as the supervisor's ``task_done`` handler would.
    """
    from supervisor.worker_process import run_worker_task

    bus = out_q if out_q is not None else workers_module.get_event_q()
    ran = []
    for task in dispatched_tasks(slot):
        try:
            run_worker_task(actor, task, wid=slot.wid, out_q=bus, repo_dir=str(workers_module.REPO_DIR),
                            drive_root=str(workers_module.DRIVE_ROOT), on_failure=on_failure)
        finally:
            if slot.busy_task_id == task.get("id"):
                slot.busy_task_id = None
        ran.append(task)
    return ran


def lane_rows(workers_module: Any) -> Dict[str, Dict[str, Any]]:
    return {tid: meta for tid, meta in workers_module.RUNNING.items()
            if isinstance(meta, dict) and meta.get("lane") == LANE_CONVERSATION}
