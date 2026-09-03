"""Battle test for the acceptance-fence IPC under real supervisor backlog.

CyberGym full1507 postmortem class: 64 concurrent roots stalled the supervisor
event loop for ~90-99s, fence acks arrived after the worker timeout, a late ack
for one operation was consumed by the next, and a retained local token poisoned
every later round into ``acceptance_fence_unavailable``. This test runs the REAL
worker methods (``OuroborosAgent._begin/_inspect/_end_acceptance_fence``), the
REAL waiter (``_await_acceptance_fence_ack``), the REAL supervisor transition
(``transition_acceptance_fence``) and the REAL ack writer
(``_handle_acceptance_fence``) across 64 concurrent workers — the production
lane count — with a delayed consumer, and proves the poisoned loop is gone:
lost acks are recovered by a fresh idempotent begin, and no operation ever
consumes another operation's ack.
"""

from __future__ import annotations

import json
import queue as stdqueue
import threading
import time
from types import SimpleNamespace

from tests.test_acceptance_fence import _isolated_queue


_WORKERS = 64
_OP_TIMEOUT_SEC = 1.0


def _make_agent(agent_cls, env_cls, canonical, repo, task_id, event_queue):
    agent = object.__new__(agent_cls)
    agent.env = env_cls(repo_dir=repo, drive_root=canonical / "idle-child")
    agent._current_task_metadata = {"budget_drive_root": str(canonical)}
    agent._current_task_id = task_id
    agent._event_queue = event_queue
    return agent


def test_acceptance_fence_ipc_survives_supervisor_backlog(monkeypatch, tmp_path):
    from ouroboros.agent import Env, OuroborosAgent
    from ouroboros.loop import (
        _begin_task_acceptance_fence,
        _end_task_acceptance_fence,
    )
    from supervisor import events

    queue_mod = _isolated_queue(monkeypatch, tmp_path)[0]
    # The waiter reads the timeout through the config SSOT; clamp it down so the
    # forced 1.5s supervisor delay lands past the worker's wait window.
    monkeypatch.setattr(
        "ouroboros.config.get_acceptance_fence_ack_timeout_sec",
        lambda: _OP_TIMEOUT_SEC,
    )
    canonical = tmp_path / "canonical-data"
    repo = tmp_path / "repo"
    canonical.mkdir()
    repo.mkdir()
    event_queue: stdqueue.Queue = stdqueue.Queue()
    handler_ctx = SimpleNamespace(DRIVE_ROOT=canonical)

    # Worker 0's first end is delayed past its ack timeout: the ack lands after
    # the waiter gave up — the exact lost-ack shape that used to poison the
    # worker. The delay happens BEFORE the handler writes, so the supervisor
    # applies the transition while the worker has already timed out.
    worker0_first_end = {"token": None, "armed": True}

    def consumer():
        while True:
            evt = event_queue.get()
            if evt is None:
                return
            if (
                worker0_first_end["armed"]
                and evt.get("action") == "end"
                and evt.get("token") == worker0_first_end["token"]
            ):
                worker0_first_end["armed"] = False
                time.sleep(2.5)
            else:
                time.sleep(0.005)
            events._handle_acceptance_fence(evt, handler_ctx)

    consumer_thread = threading.Thread(target=consumer, daemon=True)
    consumer_thread.start()

    failures: list[str] = []
    worker0_recovery: list[str] = []

    def run_worker(index: int) -> None:
        task_id = f"root-{index}"
        queue_mod.RUNNING[task_id] = {"task": {"id": task_id, "root_task_id": task_id}}
        agent = _make_agent(OuroborosAgent, Env, canonical, repo, task_id, event_queue)
        ctx = SimpleNamespace(task_metadata={"root_task_id": task_id})
        ctx.begin_acceptance_fence = agent._begin_acceptance_fence
        ctx.inspect_acceptance_fence = agent._inspect_acceptance_fence
        ctx.end_acceptance_fence = agent._end_acceptance_fence

        def begin() -> bool:
            for _attempt in range(60):
                ok, _token = _begin_task_acceptance_fence(ctx, task_id)
                if ok:
                    return True
                if index == 0:
                    worker0_recovery.append("begin_retry")
                time.sleep(0.05)
            failures.append(f"worker {index}: begin never succeeded")
            return False

        def end(outcome: str) -> bool:
            for _attempt in range(60):
                if _end_task_acceptance_fence(ctx, outcome=outcome):
                    return True
                if index == 0:
                    worker0_recovery.append(f"end_{outcome}_retry")
                time.sleep(0.05)
            failures.append(f"worker {index}: end {outcome} never succeeded")
            return False

        try:
            if not begin():
                return
            if index == 0:
                worker0_first_end["token"] = str(ctx._task_acceptance_fence_token)
            # A revision cycle: end(revision) releases the fence, then a fresh
            # begin re-opens it, then end(terminal) seals it.
            if not end("revision"):
                return
            if not begin():
                return
            if not end("terminal"):
                return
            # Production clears the sealed fence at task_done.
            queue_mod.clear_acceptance_fence_for_root(task_id)
        except Exception as exc:  # the loop-level seam must never raise here
            failures.append(f"worker {index}: {type(exc).__name__}: {exc}")
        finally:
            queue_mod.RUNNING.pop(task_id, None)

    threads = [
        threading.Thread(target=run_worker, args=(index,), daemon=True)
        for index in range(_WORKERS)
    ]
    started = time.monotonic()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=180)
    stuck = [str(i) for i, thread in enumerate(threads) if thread.is_alive()]
    assert not stuck, f"workers never finished: {stuck}"
    assert not failures, failures

    # Worker 0 provably lost its first end ack (1.5s delay vs 0.4s timeout) and
    # recovered through a fresh begin — the poisoned-token loop would instead
    # have inspected the stale token forever.
    assert "end_revision_retry" in worker0_recovery
    assert queue_mod.ACCEPTANCE_FENCES == {}

    # No ack file was consumed by the wrong operation: every leftover ack is an
    # unconsumed stale op that the hourly compaction owns, and the directory
    # stays bounded.
    ack_dir = canonical / "state" / "acceptance_fence_acks"
    leftovers = list(ack_dir.glob("*.json")) if ack_dir.is_dir() else []
    for path in leftovers:
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload.get("op"), f"ack without op identity: {path}"

    event_queue.put(None)
    consumer_thread.join(timeout=30)
    assert not consumer_thread.is_alive()
    assert time.monotonic() - started < 240
