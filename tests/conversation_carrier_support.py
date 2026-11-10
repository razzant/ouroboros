"""Drive a conversation-lane carrier inside the test process.

Production admits an owner turn or a wake to a lane worker (``supervisor.conversation_lane``)
and the worker runs ``worker_process.run_worker_task``. A test that wants the body to run takes
the dispatched task off the lane slot's queue and runs that same body here, synchronously, with
the actor it chose: the admission stays the real one, the carrier is this thread.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional

from ouroboros.task_carrier import EXECUTOR_WORKER, LANE_CONVERSATION


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
    ``busy_task_id`` is released as the supervisor's ``task_done`` handler would, and the
    carrier's ``direct_turn_bound`` frame lands on the RUNNING row as the lane's handler lays it, and its
    keyed terminal final (a system ``send_message`` frame) reaches ``send_with_budget`` as the supervisor's would.
    """
    from supervisor.worker_process import run_worker_task

    bus = _LaneFrames(out_q if out_q is not None else workers_module.get_event_q(), workers_module)
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


def host_lane(monkeypatch, root) -> None:
    """A standalone Host without an owned supervisor queue (the Host-only tests): isolated lane tables
    and roots, no pool generation, so turn bodies take the disclosed degraded executor in this process.
    A server that owns the queue but has not armed its pool instead refuses Presence with a typed retry."""
    from supervisor import queue, workers

    isolate_lane_pool(monkeypatch, root, spares=0)
    monkeypatch.setattr(queue, "INITIALIZED", False)
    monkeypatch.setattr(workers, "REPO_DIR", root / "repo")


def booted_host_lane(monkeypatch, workers_module) -> None:
    """Arm an installed server table whose lane carrier exhausted readiness, preserving inline authors.

    The fixture has no supervisor loop, so its bus delivers Presence frames to the same handlers here;
    other frames stay queued for inspection. Admission, degraded execution and settlement remain real.
    """
    from queue import Queue

    from ouroboros.presence_carrier import handle_presence_turn_initial, handle_presence_turn_settled
    from supervisor import conversation_lane

    class Frames(Queue):
        def put(self, item, *args, **kwargs):
            kind = item.get("type") if isinstance(item, dict) else None
            if kind in ("presence_turn_initial", "presence_turn_settled"):
                handler = handle_presence_turn_initial if kind == "presence_turn_initial" else handle_presence_turn_settled
                handler(item, workers_module)
            else:
                super().put(item, *args, **kwargs)

    monkeypatch.setattr(conversation_lane, "_ARMED_TABLE", None)
    monkeypatch.setattr(conversation_lane, "_SPAWN_REFUSED", [""])
    conversation_lane.arm(workers_module.WORKERS)
    install_lane_spare(workers_module).readiness_exhausted = True
    frames = Frames()
    monkeypatch.setattr(workers_module, "get_event_q", lambda create=True: frames)


class _LaneFrames:
    """The test's stand-in for the supervisor's frame consumer: every frame reaches ``inner``; the
    carrier's bound facts (``direct_turn_bound``) reach the RUNNING row as ``conversation_lane`` lays them."""

    def __init__(self, inner: Any, workers_module: Any) -> None:
        self._inner, self._workers = inner, workers_module

    def put(self, item: Any, *args: Any, **kwargs: Any) -> None:
        self._inner.put(item, *args, **kwargs)
        if not isinstance(item, dict):
            return
        if item.get("type") == "direct_turn_bound":
            from supervisor.conversation_lane import handle_direct_turn_bound

            handle_direct_turn_bound(item, SimpleNamespace(RUNNING=self._workers.RUNNING))
        elif (item.get("type") == "send_message" and item.get("role") == "system"
              and "task_terminal_status" in (item.get("progress_meta") or {})):
            # The carrier's keyed final (``conversation_exception_terminal``): the supervisor's
            # ``send_message`` handler would deliver it with ``send_with_budget``; so does this stand-in.
            self._workers.send_with_budget(
                int(item.get("chat_id") or 0), str(item.get("text") or ""), task_id=str(item.get("task_id") or ""),
                progress_meta=dict(item.get("progress_meta") or {}), role="system",
                system_type=str(item.get("system_type") or ""))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def lane_rows(workers_module: Any) -> Dict[str, Dict[str, Any]]:
    return {tid: meta for tid, meta in workers_module.RUNNING.items()
            if isinstance(meta, dict) and meta.get("lane") == LANE_CONVERSATION}


def isolate_lane_pool(monkeypatch, tmp_path, *, spares: int = 1, event_q: Any = None) -> List[Any]:
    """A fresh pool table on ``tmp_path`` for lane tests: empty RUNNING/PENDING/WORKERS shared by the
    workers and queue modules, the snapshot path, no fences, a disarmed lane (no replenishment), a server
    that must never construct an agent, and ``spares`` READY idle lane carriers (returned)."""
    import pytest

    from supervisor import conversation_lane, queue, workers

    running, pending, slots = {}, [], {}
    for module in (workers, queue):
        monkeypatch.setattr(module, "DRIVE_ROOT", tmp_path)
        monkeypatch.setattr(module, "RUNNING", running)
        monkeypatch.setattr(module, "PENDING", pending)
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(queue, "QUEUE_SNAPSHOT_PATH", tmp_path / "state/queue_snapshot.json")
    monkeypatch.setattr(queue, "ACCEPTANCE_FENCES", {})
    monkeypatch.setattr(queue, "BUDGET_ROOT_FENCES", {})
    monkeypatch.setattr(workers, "WORKERS", slots)
    monkeypatch.setattr(workers, "_WORKER_POOL_DISABLED_REASON", "")
    monkeypatch.setattr(workers, "_get_chat_agent", lambda: pytest.fail("the server constructed an agent"))
    monkeypatch.setattr(conversation_lane, "_ARMED_TABLE", None)
    monkeypatch.setattr(conversation_lane, "_SPAWN_REFUSED", [""])  # a fresh generation records no refusal
    if event_q is not None:
        monkeypatch.setattr(workers, "get_event_q", lambda: event_q)
    return [install_lane_spare(workers, 90 + index) for index in range(spares)]


def lane_row(workers_module: Any, task: Dict[str, Any], *, wid: int = 97, executor: str = EXECUTOR_WORKER,
             pid: Optional[int] = 2147480097) -> Any:
    """Install TASK as a RUNNING conversation-lane row held by carrier slot ``wid``, as admission and
    dispatch leave it: a worker carrier (a pid) or an inline executor's pseudo-slot (``pid=None``)."""
    import time

    owner = dict(task.get("_execution_owner") or {"kind": "direct"})
    owner.update(lane=LANE_CONVERSATION, executor=executor)
    task["_execution_owner"] = owner
    task.setdefault("root_task_id", str(task["id"]))
    proc = SimpleNamespace(pid=pid, is_alive=lambda: True, exitcode=None,  # a stop can join and terminate it
                           join=lambda *_a, **_k: None, terminate=lambda: None, kill=lambda: None)
    slot = workers_module.Worker(wid, proc, SimpleNamespace(put=lambda row: None), busy_task_id=str(task["id"]),
                                 lane=LANE_CONVERSATION, executor=executor)
    slot.booting, slot.reaping = False, False
    workers_module.WORKERS[wid] = slot
    now = time.time()
    workers_module.RUNNING[str(task["id"])] = {
        "task": dict(task), "worker_id": wid, "attempt": int(task.get("_attempt") or 1), "started_at": now,
        "last_heartbeat_at": now, "last_progress_at": now, "budget_paused_sec": 0.0, "soft_sent": False,
        "lane": LANE_CONVERSATION, "client_message_id": "", "reply_expected": True, "fresh_identity": False,
    }
    return slot


def park_lane_pause(event: Dict[str, Any], *, worker_id: Optional[int] = None) -> None:
    """The pooled park of a lane turn's exact pause: the supervisor's ``budget_pause`` handler over the
    live pool table (RUNNING -> PENDING), as the event bus runs it for every carried task. ``worker_id``
    stamps the frame as the carrier's body does (``run_worker_task``), so the park releases that slot."""
    from supervisor import queue, workers
    from supervisor.events_budget import _handle_budget_pause
    from supervisor.message_bus import try_get_bridge

    if worker_id is not None:
        event = {**event, "worker_id": worker_id}
    bridge = try_get_bridge() or SimpleNamespace(push_log=lambda *_a, **_k: None)
    shim = SimpleNamespace(RUNNING=workers.RUNNING, PENDING=workers.PENDING, WORKERS=workers.WORKERS,
                           DRIVE_ROOT=workers.DRIVE_ROOT, sort_pending=queue.sort_pending,
                           persist_queue_snapshot=queue.persist_queue_snapshot, bridge=bridge)
    _handle_budget_pause(event, shim)


def inline_synchronously(monkeypatch) -> None:
    """Run every inline executor body on the calling thread: the carrier is this thread, no thread is left."""
    from supervisor.inline_carrier import InlineCarrier

    def start(self) -> None:
        self._started = True
        self._thread.run()
        if self.exitcode is None:
            self.exitcode = 0

    monkeypatch.setattr(InlineCarrier, "start", start)
    monkeypatch.setattr(InlineCarrier, "is_alive", lambda self: False)
    monkeypatch.setattr(InlineCarrier, "join", lambda self, timeout=None: None)
