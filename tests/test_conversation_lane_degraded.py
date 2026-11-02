"""The degraded executor is chosen only on a physical fact, runs the same worker body on a server
thread, and is disclosed (owner decision 2A; E7).

A refused carrier spawn or a repo-writer marker window runs the admitted turn inline with the
owner's notice, a journal row and a vitals count; a merely busy lane leaves the row awaiting its
carrier. A live call is the other inline executor (``executor=inline_call``), named in its row.
"""

import json
import threading
from types import SimpleNamespace

import pytest

from ouroboros.task_carrier import EXECUTOR_INLINE_DEGRADED, LANE_CONVERSATION
from supervisor import conversation_lane, inline_carrier, worker_chat_lane, workers
from tests.test_owner_wait_pool import InputQueue, Process, pool as pool  # noqa: F401 -- fixture
from tests.worker_pool_support import affirm_ready

pytestmark = pytest.mark.serial
_REAL_THREAD = threading.Thread  # captured at collection: the pool fixture stubs the lifecycle's threads


class Actor:
    def __init__(self):
        self.ran, self.owner_wait_callback, self._event_queue = [], None, None
        self.env = SimpleNamespace(drive_root=None, drive_path=lambda rel: None)

    def handle_task(self, task):
        self.ran.append(task)
        return [{"type": "task_done", "task_id": task["id"], "chat_id": task["chat_id"], "status": "completed"}]


@pytest.fixture
def degraded(pool, monkeypatch):
    from supervisor import message_bus

    sent = []
    monkeypatch.setattr(message_bus, "_BRIDGE", SimpleNamespace(send_chat_action=lambda *a, **k: None,
                                                                 broadcast=lambda m: None, push_log=lambda *a, **k: None))
    monkeypatch.setattr(workers, "send_with_budget", lambda *a, **k: sent.append((a, k)))
    monkeypatch.setattr(conversation_lane, "_ARMED_TABLE", id(workers.WORKERS))
    actor = Actor()
    monkeypatch.setattr(workers, "_get_chat_agent", lambda: actor)
    monkeypatch.setattr(inline_carrier, "_INLINE", type(inline_carrier._INLINE)(maxlen=50))
    monkeypatch.setattr(inline_carrier, "threading", SimpleNamespace(Thread=_REAL_THREAD))  # a real carrier thread
    monkeypatch.setattr(inline_carrier, "_DEGRADED_TOTAL", [0])
    (pool.root / "logs").mkdir(parents=True, exist_ok=True)
    pool.actor, pool.sent = actor, sent
    return pool


def _inline_slot():
    return next((w for w in workers.WORKERS.values() if w.executor == EXECUTOR_INLINE_DEGRADED), None)


def _await_inline(slot):
    slot.proc.join(10)
    assert not slot.proc.is_alive()


@pytest.mark.parametrize("fact", ["spawn_refused", "marker_window"])
def test_a_physical_fact_runs_the_turn_inline_and_discloses_it(degraded, monkeypatch, fact):
    if fact == "spawn_refused":
        monkeypatch.setattr(conversation_lane, "_SPAWN_REFUSED", ["carrier_spawn_refused:OSError"])
    else:
        monkeypatch.setattr(worker_chat_lane, "marker_window_closed", lambda: True)
    worker_chat_lane.handle_chat_direct(1, "hello", None, task_metadata={"client_message_id": "cm-d"})
    slot = _inline_slot()
    assert slot is not None and slot.lane == LANE_CONVERSATION and slot.proc.pid is None
    _await_inline(slot)
    assert degraded.actor.ran and degraded.actor.ran[0]["text"] == "hello"
    assert "task_contract" in degraded.actor.ran[0]  # the same first step a carrier runs
    tid = degraded.actor.ran[0]["id"]
    meta = workers.RUNNING[tid]
    assert meta["executor"] == EXECUTOR_INLINE_DEGRADED and meta["task"]["_execution_owner"]["executor"] == EXECUTOR_INLINE_DEGRADED
    assert slot.wid not in workers.WORKERS  # the pseudo-slot ends with its thread
    vitals = inline_carrier.inline_vitals()
    assert vitals["degraded_total"] == 1 and vitals["inline_executions"][-1]["task_id"] == tid
    reason = vitals["inline_executions"][-1]["reason"]
    assert reason == ("carrier_spawn_refused:OSError" if fact == "spawn_refused" else "repo_writer_gate_closed")
    notice = [k for a, k in degraded.sent if k.get("progress_meta", {}).get("task_incident") == "conversation_degraded_executor"]
    assert len(notice) == 1
    rows = [json.loads(line) for line in (degraded.root / "logs" / "supervisor.jsonl").read_text().splitlines()]
    assert [row["executor"] for row in rows if row["type"] == "conversation_inline_executor"] == [EXECUTOR_INLINE_DEGRADED]


def test_a_busy_lane_never_selects_the_degraded_executor(degraded):
    busy = workers.Worker(5, Process(5005), InputQueue(), lane=LANE_CONVERSATION, busy_task_id="other")
    affirm_ready(busy)
    workers.WORKERS[5] = busy
    worker_chat_lane.handle_chat_direct(1, "hello", None, task_metadata={"client_message_id": "cm-b"})
    assert _inline_slot() is None and not degraded.actor.ran
    meta = next(m for m in workers.RUNNING.values() if m.get("lane") == LANE_CONVERSATION and m["task"]["text"] == "hello")
    assert meta["worker_id"] is None  # awaiting its carrier: the next ready one, or the spare being replenished
    assert inline_carrier.inline_vitals()["degraded_total"] == 0


def test_a_booting_spare_is_a_carrier_that_will_come(degraded, monkeypatch):
    monkeypatch.setattr(conversation_lane, "_SPAWN_REFUSED", ["carrier_spawn_refused:OSError"])
    booting = workers.Worker(6, Process(5006), InputQueue(), lane=LANE_CONVERSATION, reaping=True, booting=True)
    workers.WORKERS[6] = booting
    worker_chat_lane.handle_chat_direct(1, "hello", None, task_metadata={"client_message_id": "cm-c"})
    assert _inline_slot() is None and not degraded.actor.ran
