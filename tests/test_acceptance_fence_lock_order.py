"""Runtime proof of the fence's in-process lock order: admission lock -> ``_queue_lock``.

A direct turn applies its acceptance fence inside the supervisor process, so the
queue lock is now taken on the direct agent's own thread — at ``end`` while the
turn's owner-message admission lock is held. Every other party that touches both
locks (steering an owner message to the turn, Stop through
``arm_direct_chat_turn``, project-chat routing, the turn's own finalization, and
``task_done`` clearing the fence) must agree on the same order, or the process
deadlocks. The real code of each party runs here concurrently with in-process
begin/end behind an order-checking admission lock and a bounded join.
"""

from __future__ import annotations

import queue as stdqueue
import threading
from types import SimpleNamespace

from tests.test_acceptance_fence import _isolated_queue

TURN_ID = "direct-1"
ROUNDS = 25
JOIN_SEC = 60.0


class _OrderCheckedLock:
    """The turn's admission lock; records any blocking acquire made while the
    acquiring thread already owns the queue lock (the reverse order)."""

    def __init__(self, queue_lock, violations):
        self._lock = threading.Lock()
        self._queue_lock = queue_lock
        self._violations = violations

    def acquire(self, blocking=True, timeout=-1):
        if blocking and self._queue_lock._is_owned():
            self._violations.append(threading.current_thread().name)
        return self._lock.acquire(blocking, timeout)

    def release(self):
        self._lock.release()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *_exc):
        self.release()


def _direct_agent(tmp_path, queue_mod, chat_id, violations):
    from ouroboros.agent import Env, OuroborosAgent

    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    agent = object.__new__(OuroborosAgent)
    agent.env = Env(repo_dir=repo, drive_root=tmp_path)
    agent._event_queue = stdqueue.Queue()
    agent._owner_message_admission_lock = _OrderCheckedLock(queue_mod._queue_lock, violations)
    agent._owner_message_generation = 0
    agent._busy = agent._accepting_owner_messages = True
    agent._current_task_id, agent._current_chat_id = TURN_ID, chat_id
    agent._current_task_metadata = {"project_id": "racer", "title": "Direct turn"}
    agent._current_task_text = "work in the project room"
    agent._task_started_ts = agent._last_activity_ts = 1000.0
    agent._fence_in_process = lambda: True  # a direct turn runs in the queue's process
    return agent


def test_order_checker_reports_the_reverse_order(monkeypatch, tmp_path):
    """The guard fires: queue lock first, admission second is exactly what it records."""
    queue_mod, _pending = _isolated_queue(monkeypatch, tmp_path)
    violations: list = []
    lock = _OrderCheckedLock(queue_mod._queue_lock, violations)
    with lock:
        with queue_mod._queue_lock:
            pass
    assert violations == []
    with queue_mod._queue_lock:
        with lock:
            pass
    assert len(violations) == 1


def test_in_process_fence_keeps_admission_before_queue_lock(monkeypatch, tmp_path):
    from ouroboros.loop import _begin_task_acceptance_fence, _end_task_acceptance_fence
    from ouroboros.projects_registry import create_project
    from ouroboros.server_owner_routing import _route_project_chat_to_running_task
    from supervisor import workers
    from supervisor.active_activity import get_direct_activity_registry
    from supervisor.steering import _handle_steer_task

    queue_mod, pending = _isolated_queue(monkeypatch, tmp_path)
    chat_id = int(create_project(tmp_path, "racer")["chat_id"])
    violations: list = []
    agent = _direct_agent(tmp_path, queue_mod, chat_id, violations)
    get_direct_activity_registry().register(TURN_ID, chat_id, project_id="racer", actor=agent)
    supervisor_ctx = SimpleNamespace(
        DRIVE_ROOT=tmp_path, RUNNING=queue_mod.RUNNING, PENDING=pending, bridge=None,
        persist_queue_snapshot=queue_mod.persist_queue_snapshot,
        send_with_budget=lambda *_a, **_k: None,
    )
    turn_ctx = SimpleNamespace(
        task_metadata={"root_task_id": TURN_ID}, task_id=TURN_ID, drive_root=tmp_path,
        owner_message_admission_lock=agent._owner_message_admission_lock,
        owner_message_admission_agent=agent,
        begin_acceptance_fence=agent._begin_acceptance_fence,
        inspect_acceptance_fence=agent._inspect_acceptance_fence,
        end_acceptance_fence=agent._end_acceptance_fence,
        _task_acceptance_fence_token=None, _task_acceptance_sealed_fence_token=None,
        _loop_mailbox_seen_ids=set(),
    )
    done = {"begun": 0, "sealed": 0, "steered": 0, "armed": 0, "routed": 0, "cleared": 0}
    errors: list = []
    start = threading.Barrier(5)

    def party(body):
        def run():
            try:
                start.wait(timeout=20)
                for index in range(ROUNDS):
                    body(index)
            except Exception as exc:  # noqa: BLE001 — every failure is reported by the test
                errors.append(f"{threading.current_thread().name}: {type(exc).__name__}: {exc}")
        return run

    def turn(_index):
        # A panel cycle, then the turn's own finalization: release under the admission
        # lock (an owner follow-up superseded the answer), reopen, seal.
        opened, _token = _begin_task_acceptance_fence(turn_ctx, TURN_ID)
        if not opened:
            return  # the sealed row of the previous round is not cleared yet
        done["begun"] += 1
        _begin_task_acceptance_fence(turn_ctx, TURN_ID)  # refresh: in-process inspect
        with agent._owner_message_admission_lock:
            _end_task_acceptance_fence(turn_ctx, outcome="revision", admission_locked=True)
        turn_ctx._task_acceptance_owner_generation = None  # the follow-up was consumed
        if _begin_task_acceptance_fence(turn_ctx, TURN_ID)[0] and _end_task_acceptance_fence(
                turn_ctx, outcome="terminal") and turn_ctx._task_acceptance_sealed_fence_token:
            done["sealed"] += 1
        turn_ctx._task_acceptance_sealed_fence_token = None

    def steering(index):
        before = agent._owner_message_generation
        _handle_steer_task({
            "target_task_id": TURN_ID, "chat_id": chat_id, "message": f"steer {index}",
            "client_message_id": f"steer-{index}",
        }, supervisor_ctx)
        done["steered"] += int(agent._owner_message_generation > before)

    def stop(index):
        done["armed"] += int(workers.arm_direct_chat_turn(TURN_ID, lambda _turn: f"control-{index}") is not None)

    def routing(index):
        routed = _route_project_chat_to_running_task(
            supervisor_ctx, chat_id, f"route {index}", client_message_id=f"route-{index}")
        done["routed"] += int(routed == TURN_ID)

    def task_done(_index):
        with queue_mod._queue_lock:  # the ``task_done`` seam: pop RUNNING, then clear the fence
            queue_mod.RUNNING.pop(TURN_ID, None)
        done["cleared"] += int(queue_mod.clear_acceptance_fence_for_root(TURN_ID))
        queue_mod.persist_queue_snapshot(reason="task_done")

    threads = [
        threading.Thread(target=party(body), name=body.__name__, daemon=True)
        for body in (turn, steering, stop, routing, task_done)
    ]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=JOIN_SEC)
        stuck = [thread.name for thread in threads if thread.is_alive()]
    finally:
        get_direct_activity_registry().unregister(TURN_ID)
    assert not stuck, f"deadlock: {stuck} never finished"
    assert errors == []
    assert violations == [], f"queue lock held while taking the admission lock: {violations}"
    # Every party really ran against the in-process fence.
    assert done["begun"] and done["sealed"], done
    assert done["steered"] and done["armed"] and done["routed"], done
    assert agent._event_queue.empty()  # the direct turn never asked through the event queue
    assert not (tmp_path / "state" / "acceptance_fence_acks").exists()
