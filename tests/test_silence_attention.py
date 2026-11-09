"""Unknown live silence is a fact for a mind, never a timeout verdict."""

from types import SimpleNamespace

import pytest

from tests.test_idle_observation import host as _host, _controls
from ouroboros.owner_mailbox import drain_owner_entries
from ouroboros.task_results import load_task_result, write_task_result
from supervisor import queue, workers

host = _host


@pytest.fixture
def attention(host, monkeypatch):
    from supervisor import queue_snapshot, silence_attention, workspace_preparation as preparation

    jobs = []
    monkeypatch.setattr(preparation, "_process_owner", lambda: {"pid": 11, "birth": 1})
    generation = preparation.PreparationGeneration()
    monkeypatch.setattr(generation, "submit", lambda job, lane: jobs.append(job) or True)
    monkeypatch.setattr(generation, "current_lane", lambda: preparation.ADMISSION_LANE)
    monkeypatch.setattr("supervisor.workspace_preparation.current_generation", lambda: generation)
    monkeypatch.setattr(silence_attention, "time", SimpleNamespace(time=lambda: host.clock[0]))
    for name in ("BUDGET_ROOT_FENCES", "ACCEPTANCE_FENCES", "ADMISSION_RESERVATIONS", "RETURNS_NAMED"):
        monkeypatch.setattr(queue, name, {})
    monkeypatch.setattr(workers, "_worker_pool_execution_state", lambda: {"available": True, "disabled_reason": ""})
    monkeypatch.setattr(queue, "QUEUE_SEQ_COUNTER_REF", {"value": 0})
    monkeypatch.setattr(queue, "QUEUE_SNAPSHOT_PATH", host.root / "state" / "queue_snapshot.json")
    monkeypatch.setattr(queue, "QUEUE_PUBLICATION", None)
    monkeypatch.setattr(queue, "persist_queue_snapshot", queue_snapshot.persist_queue_snapshot)
    monkeypatch.setattr(workers, "PENDING", queue.PENDING)
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(host.root))
    monkeypatch.setenv("OUROBOROS_SETTINGS_PATH", str(host.root / "settings.json"))
    monkeypatch.setenv("TOTAL_BUDGET", "1000")
    monkeypatch.setattr("ouroboros.platform_layer.process_start_time", lambda pid: None)

    def forbidden(*args, **kwargs):
        raise AssertionError("No process death or real worker launch is permitted in this test")

    monkeypatch.setattr("supervisor.worker_pool_lifecycle.kill_worker_tree", forbidden)
    monkeypatch.setattr("supervisor.task_reaper._kill_and_confirm_worker_dead", forbidden)
    host.forbidden = forbidden
    write_task_result(host.root, "root", "running", root_task_id="root", task_attempt=1,
                      billing_group={"billing_group_id": "root", "billing_group_limit_usd": 10,
                        "billing_group_limit_source": "initial_task_admission", "billing_group_limit_revision": "original"})
    host.meta["task"]["metadata"] = {"work_kind": "owner"}

    def flush():
        while jobs:
            jobs.pop(0)()

    host.flush = flush
    host.generation = generation
    return host


def observation(host):
    return (load_task_result(host.root, "root") or {}).get("silence_attention", {})


def test_heartbeat_only_never_finalizes_or_reaps(host):
    host.bus.put({"type": "task_heartbeat", "task_id": "root"})
    host.drain()
    host.tick()
    host.clock[0] += 10000
    host.tick()
    assert _controls(host) == []
    assert queue._reap_queue.empty()
    assert host.running["root"] is host.meta


def test_waiting_parent_receives_one_attention_through_actual_wait_consumer(attention, monkeypatch):
    from ouroboros.tools import control_task_results
    from ouroboros.tools.registry import ToolContext

    attention.meta["task"]["parent_task_id"] = "parent"
    attention.running["parent"] = {
        "task": {"id": "parent", "root_task_id": "parent"}, "attempt": 1,
        "started_at": 4000.0, "last_progress_at": 4999.0,
    }
    ctx = ToolContext(repo_dir=attention.root, drive_root=attention.root, task_id="parent",
                      event_queue=attention.bus)
    ctx.task_attempt = 1
    ctx.root_task_id = "parent"

    def wait(root, ids, **kwargs):
        assert ids == ["root"]
        attention.drain()  # actual in_slot_wait producer -> cognitive_operations reducer
        assert kwargs["on_poll"]({}, {}) is None
        for _ in range(2):
            attention.tick()
            attention.flush()
        early = kwargs["on_poll"]({}, {})
        assert early["reason"] == "owner_mailbox_pending"
        return {"early_return": early, "tasks": {}, "elapsed_sec": 0}

    monkeypatch.setattr(control_task_results, "wait_for_effective_tasks", wait)
    result = control_task_results._wait_for_task(ctx, "root", 30)
    assert "interrupted by an unread message" in result
    mail = drain_owner_entries(attention.root, "parent")
    assert len(mail) == 1 and mail[0]["source_task_id"] == "root"
    assert observation(attention)["recipient_task_id"] == "parent"
    assert queue.PENDING == []


def test_real_admission_generation_publishes_off_queue_lock(attention, monkeypatch):
    from supervisor import workspace_preparation as preparation

    monkeypatch.setattr(preparation, "_process_owner", lambda: {"pid": 11, "birth": 1})
    generation = preparation.PreparationGeneration()
    monkeypatch.setattr(preparation, "current_generation", lambda: generation)
    try:
        attention.tick()
        assert generation.lane_idle(preparation.ADMISSION_LANE, 15)
        assert queue.PENDING == []
        assert observation(attention)["self_delivered"]
    finally:
        generation.close()
        assert generation.lane_idle(preparation.ADMISSION_LANE, 15)


@pytest.mark.parametrize("selected", ["root", "middle"])
def test_warm_parent_waiting_for_helper_terminal_wakes_on_one_attention(attention, monkeypatch, selected):
    from ouroboros.owner_wait import checkpoint_owner_wait, direct_owner_wait, _sleep_wake
    from ouroboros.tools.control_task_results import _await_messages
    from supervisor.worker_owner_wait import handle_owner_wait
    from tests.test_model_sleep import _ctx

    attention.meta["task"]["parent_task_id"] = "parent" if selected == "root" else "middle"
    if selected == "middle":
        write_task_result(attention.root, "middle", "running")
        attention.running["middle"] = {"task": {"id": "middle", "parent_task_id": "parent"},
                                        "attempt": 1, "started_at": 4990.0, "last_progress_at": 4999.0}
    write_task_result(attention.root, "parent", "running")
    ctx = _ctx(attention.root, "parent", root="parent")
    assert '"sleep_armed"' in _await_messages(ctx, mode="warm", tasks=[selected])
    checkpoint = checkpoint_owner_wait(ctx, [{"role": "user", "content": "await helper"}], {}, {}, 1, [], set())
    commands = []
    worker = SimpleNamespace(wid=1, proc=SimpleNamespace(pid=77, is_alive=lambda: True),
                             busy_task_id="parent", reaping=False, active_capacity=True,
                             in_q=SimpleNamespace(put=commands.append))
    workers.WORKERS[1] = worker
    parent = {"task": {"id": "parent"}, "worker_id": 1, "attempt": 1,
              "started_at": 4900.0, "last_progress_at": 4999.0}
    attention.running["parent"] = parent
    handle_owner_wait({"task_id": "parent", "wait_id": checkpoint["wait_id"], "task_attempt": 1,
                       "worker_id": 1, "pid": 77, "phase": "park", "checkpoint": checkpoint},
                      SimpleNamespace(RUNNING=attention.running, WORKERS=workers.WORKERS, DRIVE_ROOT=attention.root))
    assert parent["sleep_parked_at"] and not worker.active_capacity
    assert _sleep_wake(ctx, checkpoint) == ""
    for _ in range(2):
        attention.tick()
        attention.flush()
    assert _sleep_wake(ctx, checkpoint) == "work_attention:root"
    assert direct_owner_wait(ctx, checkpoint) == "work_attention:root"
    mail = drain_owner_entries(attention.root, "parent")
    assert len(mail) == 1 and mail[0]["provenance"] == "system"
    assert mail[0]["work_attention"]["waiting_for_task_id"] == selected
    assert observation(attention)["parent_delivered"] and queue.PENDING == []


@pytest.mark.parametrize("paused", [False, True])
def test_cold_parent_waiting_for_helper_terminal_uses_existing_wake_grant(attention, monkeypatch, paused):
    from ouroboros import budget_pause, owner_pause
    from supervisor import state
    from supervisor.sleep_wake import wake_ready_sleepers
    from tests.test_model_sleep import _cold_park

    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_k: 5.0)
    monkeypatch.setattr(budget_pause, "observe_task_runs", lambda *_a, **_k: {"custody_read": "ok", "runs": []})
    attention.meta["task"]["parent_task_id"] = "parent"
    _cold_park(attention.root, monkeypatch, workers, task_id="parent", tasks=["root"])
    parked = next(row for row in queue.PENDING if row["id"] == "parent")
    assert "parent" not in queue.RUNNING
    assert wake_ready_sleepers(queue) == []
    if paused:
        owner_pause.install_fence(attention.root, "parent", request_id="owner-pause")
    for _ in range(2):
        attention.tick()
        attention.flush()
    outcome = wake_ready_sleepers(queue)
    assert outcome[0]["ready"]["reason"] == "work_attention:root"
    if paused:
        assert outcome[0]["veto"] == "owner_pause"
        assert "_budget_pause_resume" not in parked
    else:
        assert outcome[0]["ok"]
        assert parked["_budget_pause_resume"]["selected_by"] == "sleep_wake"
    assert len(queue.PENDING) == 1 and queue.PENDING[0] is parked
    assert observation(attention)["parent_delivered"]
    assert len(drain_owner_entries(attention.root, "parent")) == 1


def test_real_owner_answer_wait_is_unavailable_even_with_old_helper_wait_lease(attention):
    attention.meta["task"]["parent_task_id"] = "parent"
    attention.running["parent"] = {
        "task": {"id": "parent"}, "attempt": 1, "started_at": 4900.0, "last_progress_at": 4999.0,
        "owner_wait": {"state": "waiting", "task_attempt": 1, "reason": "owner", "source_ref": {"x": 1}},
        "active_operation_leases": {"old": {"kind": "in_slot_wait", "until_ts": 9000, "wait_for_tasks": ["root"]}},
    }
    attention.tick()
    attention.flush()
    assert queue.PENDING == [] and not observation(attention).get("parent_delivered")


def test_sleep_attention_is_bound_to_the_selected_task_and_system_fact(attention):
    from ouroboros import model_sleep
    from ouroboros.owner_mailbox import write_task_message
    from tests.test_model_sleep import _ctx

    write_task_result(attention.root, "parent", "running")
    write_task_result(attention.root, "other", "running")
    ctx = _ctx(attention.root, "parent", root="parent")
    chosen = model_sleep.selectors(ctx, tasks=["other"])
    attention.meta["task"]["parent_task_id"] = "parent"
    attention.running["parent"] = {
        "task": {"id": "parent"}, "attempt": 1, "started_at": 4990.0,
        "owner_wait": {"state": "waiting", "task_attempt": 1, "reason": "sleep", "sleep": chosen},
    }
    attention.tick()
    attention.flush()
    assert queue.PENDING == [] and not observation(attention).get("parent_delivered")
    assert model_sleep.wake_reason(ctx, chosen) == ""
    write_task_message(attention.root, "ordinary system mail about selected task", "parent",
                       source_task_id="other", provenance="system")
    assert model_sleep.wake_reason(ctx, chosen) == ""
    write_task_message(attention.root, "an unrelated silence", "parent", source_task_id="root", provenance="system",
                       work_attention={"kind": "silence", "task_id": "root", "episode_id": "e",
                                       "waiting_for_task_id": "root"})
    assert model_sleep.wake_reason(ctx, chosen) == ""


@pytest.mark.parametrize("wait", [{"wait_for_tasks": ["unrelated"]}, {"kind": "owner_wait"}, {}])
def test_unrelated_or_unavailable_parent_does_not_take_responsibility(attention, wait):
    attention.meta["task"]["parent_task_id"] = "parent"
    attention.running["parent"] = {
        "task": {"id": "parent"}, "attempt": 1, "started_at": 4990.0, "last_progress_at": 4999.0,
        "active_operation_leases": {"wait": {"kind": "in_slot_wait", "until_ts": 9000, **wait}},
    }
    attention.tick()
    attention.flush()
    assert queue.PENDING == []
    assert not observation(attention).get("recipient_task_id")
    assert drain_owner_entries(attention.root, "parent") == []


def test_root_without_a_waiting_parent_gets_one_fact_and_one_self_mail_and_no_row(attention, monkeypatch):
    monkeypatch.setattr(queue, "load_state", lambda: {"bg_consciousness_enabled": False})
    for _ in range(3):
        attention.tick()
        attention.flush()
    fact = observation(attention)
    assert fact["self_delivered"] and not fact.get("recipient_task_id")
    assert "diagnostic_id" not in fact and "attention_wait" not in fact
    mail = drain_owner_entries(attention.root, "root")
    assert len(mail) == 1 and mail[0]["kind"] == "task_message"
    assert "not a stop request" in mail[0]["text"]
    # No host role is minted: nothing queued, no worker, source untouched.
    assert queue.PENDING == [] and workers.WORKERS == {}
    assert queue._reap_queue.empty() and attention.running["root"] is attention.meta


def test_terminal_file_does_not_release_or_kill_source(attention):
    worker = SimpleNamespace(busy_task_id="root", proc=SimpleNamespace(
        is_alive=lambda: True, terminate=attention.forbidden, kill=attention.forbidden))
    workers.WORKERS[0] = worker
    write_task_result(attention.root, "root", "completed", result="An answer is not process death.")
    attention.tick()
    attention.flush()
    assert load_task_result(attention.root, "root")["status"] == "completed"
    assert "root" in attention.running and queue._reap_queue.empty()
    assert queue.PENDING == []
    assert workers.WORKERS[0] is worker and worker.busy_task_id == "root"


def test_delivery_failure_retries_same_episode_without_duplicate_consumption(attention, monkeypatch):
    from ouroboros.config import BODY_MAINTENANCE_INTERVAL_SEC
    from supervisor import silence_attention

    save, calls = silence_attention._save, []

    def interrupted(*args):
        calls.append(True)
        if len(calls) == 2:
            raise OSError("lost acknowledgement after delivery and admission")
        return save(*args)

    monkeypatch.setattr(silence_attention, "_save", interrupted)
    for _ in range(3):
        attention.tick()
        attention.flush()
        attention.clock[0] += BODY_MAINTENANCE_INTERVAL_SEC
    assert queue.PENDING == []
    assert len(drain_owner_entries(attention.root, "root")) == 1


def test_delivered_episode_publishes_nothing_again_until_a_waiter_appears(attention, monkeypatch):
    """Both directions of `_submit`'s early return: a published, self-delivered fact with no
    new waiter is not published again, but a waiting parent that appears later gets it once."""
    from ouroboros.config import BODY_MAINTENANCE_INTERVAL_SEC
    from supervisor import silence_attention

    publishes = []
    publish = silence_attention._publish
    monkeypatch.setattr(silence_attention, "_publish", lambda *a: publishes.append(a[1]) or publish(*a))
    attention.meta["task"]["parent_task_id"] = "parent"
    write_task_result(attention.root, "parent", "running")
    attention.tick()
    attention.flush()
    assert publishes == ["root"] and observation(attention)["self_delivered"]
    first_mail = drain_owner_entries(attention.root, "root")  # draining does not consume
    assert len(first_mail) == 1
    # (a) no new waiter: not even after the retry cadence has elapsed
    for _ in range(3):
        attention.clock[0] += BODY_MAINTENANCE_INTERVAL_SEC
        attention.tick()
        attention.flush()
    assert publishes == ["root"] and drain_owner_entries(attention.root, "root") == first_mail
    assert not observation(attention).get("parent_delivered")
    # (b) the same episode reaches a waiting parent that appears later, exactly once
    episode = observation(attention)["episode_id"]
    attention.running["parent"] = {
        "task": {"id": "parent"}, "attempt": 1, "started_at": 4999.0, "last_progress_at": 4999.0,
        "active_operation_leases": {"wait": {"kind": "in_slot_wait", "until_ts": 99999,
                                             "wait_for_tasks": ["root"]}}}
    for _ in range(3):
        attention.tick()
        attention.flush()
    assert publishes == ["root", "root"]
    assert observation(attention)["episode_id"] == episode and observation(attention)["parent_delivered"]
    assert len(drain_owner_entries(attention.root, "parent")) == 1
    assert drain_owner_entries(attention.root, "root") == first_mail
    assert queue.PENDING == []


def test_silence_mail_leads_with_the_date_in_words():
    """The mail is the only surface a waiting parent (or the work itself) sees: its first words, which the
    200-character row preview shows, say since when, in UTC, not a JSON record."""
    from supervisor.silence_attention import _mail_text

    text = _mail_text({"episode_id": "ep-1", "last_progress_at": 1794190200.0, "threshold_sec": 900.0})
    assert text.startswith("No recognized progress or active wait since 2026-11-09 02:10 UTC (idle window 900 s;")
    assert "silence episode ep-1" in text and "not a stop request" in text
    assert "{" not in text[:200]
