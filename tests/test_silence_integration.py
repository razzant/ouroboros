"""Focused #1376 consumer probes; no provider, real process or real control effects."""
from types import SimpleNamespace

import pytest

from ouroboros.task_results import load_task_result, write_task_result
from supervisor import queue
from tests.test_silence_attention_boundaries import (
    attention as attention, host as host,
    no_external_effects as no_external_effects,
)
from tests.test_silence_attention import observation


@pytest.mark.parametrize(("selected", "mode"), [("root", "warm"), ("root", "slot"), ("middle", "cold")])
def test_late_waiter_receives_one_attention_without_readmission(attention, monkeypatch, mode, selected):
    from ouroboros import model_sleep
    from ouroboros.owner_mailbox import drain_owner_entries
    from tests.test_model_sleep import _ctx
    from supervisor import silence_attention
    task = attention.meta["task"]
    task["parent_task_id"] = "parent" if selected == "root" else "middle"
    if selected == "middle":
        attention.running["middle"] = {"task": {"id": "middle", "parent_task_id": "parent"},
                                       "attempt": 1, "started_at": 4999, "last_progress_at": 4999}
        write_task_result(attention.root, "middle", "running")
    parent = {"task": {"id": "parent"}, "attempt": 1, "started_at": 4999, "last_progress_at": 4999}
    attention.running["parent"] = parent
    write_task_result(attention.root, "parent", "running")
    attention.tick()
    attention.flush()  # published before any waiter exists
    ctx = _ctx(attention.root, "parent", root="parent")
    chosen = model_sleep.selectors(ctx, tasks=[selected])
    checkpoint = {"reason": "sleep", "state": "waiting", "task_attempt": 1, "sleep": chosen}
    if mode == "cold":
        attention.running.pop("parent")
        queue.PENDING.append({"id": "parent", "_budget_pause": checkpoint})
    elif mode == "warm":
        parent["owner_wait"] = checkpoint
    else:
        parent["active_operation_leases"] = {"wait": {"kind": "in_slot_wait", "until_ts": 9000,
                                                     "wait_for_tasks": [selected]}}
    attention.tick()
    attention.flush()
    assert model_sleep.wake_reason(ctx, chosen) == "work_attention:root"
    mail = drain_owner_entries(attention.root, "parent")
    assert len(mail) == 1 and mail[0]["work_attention"]["waiting_for_task_id"] == selected
    assert not any(str(t["id"]).startswith("silence-") for t in queue.PENDING)
    monkeypatch.setattr(silence_attention, "_publish", attention.forbidden)
    for _ in range(3):
        attention.tick()
        attention.flush()
    # Draining exposes mail; until acknowledged the existing row stays readable.
    assert drain_owner_entries(attention.root, "parent") == mail


@pytest.mark.parametrize("mode", ["warm", "cold", "slot"])
def test_real_late_wait_consumers_wake(attention, monkeypatch, mode):
    from ouroboros.owner_mailbox import drain_owner_entries
    from ouroboros.tools import control_task_results
    from ouroboros import budget_pause
    from ouroboros.owner_wait import checkpoint_owner_wait, direct_owner_wait, _sleep_wake
    from tests.test_model_sleep import _ctx, _cold_park
    from supervisor import state, workers
    from supervisor.sleep_wake import wake_ready_sleepers
    from supervisor.worker_owner_wait import handle_owner_wait

    attention.meta["task"]["parent_task_id"] = "parent"
    write_task_result(attention.root, "parent", "running")
    parent = {"task": {"id": "parent", "root_task_id": "parent"}, "worker_id": 1,
              "attempt": 1, "started_at": 4900.0, "last_progress_at": 4999.0}
    attention.running["parent"] = parent
    attention.tick()
    attention.flush()
    ctx = _ctx(attention.root, "parent", root="parent")
    ctx.event_queue = attention.bus
    if mode == "cold":
        monkeypatch.setattr(state, "budget_remaining", lambda *a, **kw: 5.0)
        monkeypatch.setattr(budget_pause, "observe_task_runs", lambda *a, **kw: {"custody_read": "ok", "runs": []})
        _cold_park(attention.root, monkeypatch, workers, task_id="parent", tasks=["root"])
        parked = next(t for t in queue.PENDING if t["id"] == "parent")
        assert wake_ready_sleepers(queue) == []
        attention.tick()
        attention.flush()
        outcome = wake_ready_sleepers(queue)
        assert outcome[0]["ok"] and outcome[0]["ready"]["reason"] == "work_attention:root"
        assert parked["_budget_pause_resume"]["selected_by"] == "sleep_wake"
    elif mode == "warm":
        assert '"sleep_armed"' in control_task_results._await_messages(ctx, mode="warm", tasks=["root"])
        checkpoint = checkpoint_owner_wait(ctx, [{"role": "user", "content": "await helper"}], {}, {}, 1, [], set())
        workers.WORKERS[1] = SimpleNamespace(wid=1, proc=SimpleNamespace(pid=77, is_alive=lambda: True),
            busy_task_id="parent", reaping=False, active_capacity=True, in_q=SimpleNamespace(put=lambda x: None))
        handle_owner_wait({"task_id": "parent", "wait_id": checkpoint["wait_id"], "task_attempt": 1,
            "worker_id": 1, "pid": 77, "phase": "park", "checkpoint": checkpoint},
            SimpleNamespace(RUNNING=attention.running, WORKERS=workers.WORKERS, DRIVE_ROOT=attention.root))
        assert _sleep_wake(ctx, checkpoint) == ""
        attention.tick()
        attention.flush()
        assert direct_owner_wait(ctx, checkpoint) == "work_attention:root"
    else:
        def wait(root, ids, **kwargs):
            attention.drain()
            assert kwargs["on_poll"]({}, {}) is None
            attention.tick()
            attention.flush()
            early = kwargs["on_poll"]({}, {})
            assert early["reason"] == "owner_mailbox_pending"
            return {"early_return": early, "tasks": {}, "elapsed_sec": 0}
        monkeypatch.setattr(control_task_results, "wait_for_effective_tasks", wait)
        assert "interrupted by an unread message" in control_task_results._wait_for_task(
            ctx, "root", 30, sleep_mode="in_slot")  # the named slot adapter; an unnamed mode sleeps (owner Q2=A)
    assert len(drain_owner_entries(attention.root, "parent")) == 1
    assert not any(str(t["id"]).startswith("silence-") for t in queue.PENDING)


@pytest.mark.parametrize("unavailable", ["missing", "other_task", "held"])
def test_no_republication_for_unavailable_late_waiter(attention, monkeypatch, unavailable):
    from supervisor import silence_attention
    attention.meta["task"]["parent_task_id"] = "parent"
    attention.tick()
    attention.flush()
    if unavailable != "missing":
        parent = {"task": {"id": "parent"}, "attempt": 1, "started_at": 4999, "last_progress_at": 4999,
                  "active_operation_leases": {"wait": {"kind": "in_slot_wait", "until_ts": 9000,
                    "wait_for_tasks": ["other" if unavailable == "other_task" else "root"]}}}
        if unavailable == "owner":
            parent["owner_wait"] = {"state": "waiting", "task_attempt": 1, "reason": "owner"}
        elif unavailable == "held":
            parent["task"]["_budget_pause_hold"] = {"reason": "owner"}
        attention.running["parent"] = parent
    monkeypatch.setattr(silence_attention, "_publish", attention.forbidden)
    attention.tick()
    attention.flush()


@pytest.mark.parametrize("barrier", ["parent", "store"])
def test_unchanged_attention_avoids_tick_reads_and_recovers(attention, monkeypatch, barrier):
    from ouroboros.owner_mailbox import drain_owner_entries
    from ouroboros.config import BODY_MAINTENANCE_INTERVAL_SEC
    from ouroboros.store import StoreError
    from supervisor import silence_attention

    available = [False]
    if barrier == "parent":
        attention.meta["task"]["parent_task_id"] = "parent"
        attention.running["parent"] = {"task": {"id": "parent"}, "attempt": 1,
            "started_at": 4999, "last_progress_at": 4999, "owner_wait": {
                "state": "waiting", "task_attempt": 1, "reason": "sleep", "sleep": {"tasks": ["root"]}}}
        write_task_result(attention.root, "parent", "running")
    publishes, reads = [], []
    publish = silence_attention._publish
    def counted_publish(*args):
        publishes.append(attention.clock[0])
        return publish(*args)
    def read(root, task_id, **kwargs):
        reads.append(task_id)
        if barrier == "store" and task_id == "root" and not available[0]:
            raise StoreError("temporary source unavailability")
        return load_task_result(root, task_id, **kwargs)
    monkeypatch.setattr(silence_attention, "_publish", counted_publish)
    monkeypatch.setattr("ouroboros.task_results.load_task_result", read)
    attention.tick()
    attention.flush()
    initial_reads = len(reads)
    for _ in range(9):
        attention.clock[0] += 0.5
        attention.tick()
        attention.flush()
    print("ATTENTION_REPEATS", barrier, "publishes", len(publishes), "source_reads", len(reads))
    assert len(publishes) == 1 and len(reads) == initial_reads
    assert queue.PENDING == []
    if barrier == "parent":
        # Delivered once to the waiting parent and once to the source; nothing to retry.
        assert observation(attention)["parent_delivered"] and observation(attention)["self_delivered"]
        assert len(drain_owner_entries(attention.root, "parent")) == 1
        return
    available[0] = True  # the unreadable source recovers on the existing maintenance cadence
    attention.clock[0] += BODY_MAINTENANCE_INTERVAL_SEC
    attention.tick()
    attention.flush()
    assert len(publishes) == 2 and queue.PENDING == []
    assert observation(attention)["state"] == "pending_attention"


def test_ordinary_progress_keeps_the_certificate_and_recovery_republishes(attention):
    """Only an open episode projects its fact into the certified queue; another row's
    progress stamps never change the certificate; the source's own recovery does."""
    from ouroboros.task_status import _load_queue_snapshot

    busy = {"task": {"id": "busy", "root_task_id": "busy"}, "attempt": 1, "worker_id": 2,
            "started_at": 4900.0, "last_progress_at": 4950.0}
    attention.running["busy"] = busy
    attention.tick()
    attention.flush()

    def running(task_id):
        return next(row for row in _load_queue_snapshot(attention.root)["running"] if row["id"] == task_id)

    assert queue.persist_queue_snapshot(reason="certificate_probe")
    generation = queue.QUEUE_PUBLICATION[2]
    assert not {"last_progress_at", "silence_attention"} & set(running("busy"))
    assert running("root")["silence_attention"]["state"] == "pending_attention"
    for stamp in (4960.0, 4970.0):
        busy["last_progress_at"] = stamp
        assert queue.persist_queue_snapshot(reason="certificate_probe")
        assert queue.QUEUE_PUBLICATION[2] == generation
    attention.meta["last_progress_at"] = attention.clock[0]  # the source's own progress
    attention.tick()
    attention.flush()
    assert observation(attention)["state"] == "progress_observed"
    assert queue.persist_queue_snapshot(reason="certificate_probe")
    assert queue.QUEUE_PUBLICATION[2] != generation
    assert running("root")["silence_attention"]["state"] == "progress_observed"
    assert queue.PENDING == [] and "root" in attention.running and queue._reap_queue.empty()
