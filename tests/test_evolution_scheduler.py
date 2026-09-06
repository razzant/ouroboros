"""What the evolution scheduler enqueues, replaces and refuses, and what assignment dispatches.

Split out of ``tests/test_evolution_state_integrity_v3.py`` by theme: the bare flag with no
campaign, the active campaign with no source, the owner resume that repairs a legacy
source, the transaction attach and its owner-stop recheck, the uncommitted transaction
replaced only when no worker is reaping, and the exact claim assignment must see.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from tests._evolution_state_shared import (
    _CaptureQueue,
    _active_transaction,
)


def _assignment_case(tmp_path, monkeypatch, task_id="assign-evo"):
    from supervisor import evolution_lifecycle, queue, state, workers

    state.init(tmp_path)
    monkeypatch.setattr(state, "TOTAL_BUDGET_LIMIT", 0.0)
    pending, running = [], {}
    monkeypatch.setattr(workers, "PENDING", pending)
    monkeypatch.setattr(workers, "RUNNING", running)
    workers.init(tmp_path, tmp_path, 1)
    campaign = evolution_lifecycle.start_evolution_campaign("Improve", source="test")
    state.update_state(lambda live: live.update(
        evolution_mode_enabled=True,
        evolution_owner_stopped=False,
    ))
    tx = evolution_lifecycle.begin_evolution_transaction(task_id, cycle=1, campaign=campaign)
    task = {
        "id": task_id,
        "type": "evolution",
        "text": "Improve",
        "metadata": {"evolution_transaction": dict(tx)},
    }
    pending.append(task)
    inbox, events = _CaptureQueue(), _CaptureQueue()
    worker = SimpleNamespace(wid=1, busy_task_id=None, reaping=False, in_q=inbox)
    monkeypatch.setattr(workers, "WORKERS", {1: worker})
    monkeypatch.setattr(workers, "get_event_q", lambda: events)
    monkeypatch.setattr(queue, "persist_queue_snapshot", lambda reason="": None)
    monkeypatch.setattr(evolution_lifecycle, "evolution_block_reason", lambda: "")
    return workers, task, tx, worker, inbox, events


def test_scheduler_disables_a_bare_flag_without_campaign(tmp_path, monkeypatch):
    from supervisor import queue, state

    state.init(tmp_path)
    queue.init(tmp_path)
    pending = []
    queue.init_queue_refs(pending, {}, {"value": 0})
    live = state.load_state()
    live.update({
        "owner_chat_id": 1,
        "evolution_mode_enabled": True,
        "post_task_autostop": True,
    })
    state.save_state(live)
    sent = []
    monkeypatch.setattr(queue, "send_with_budget", lambda *args, **kwargs: sent.append(args[1]))

    queue.enqueue_evolution_task_if_needed()

    assert pending == []
    assert state.load_state()["evolution_mode_enabled"] is False
    assert state.load_state()["post_task_autostop"] is False
    assert "active campaign authority" in sent[0]
    event = json.loads((tmp_path / "logs" / "events.jsonl").read_text().splitlines()[-1])
    assert event["type"] == "evolution_authority_missing"


def test_scheduler_refuses_active_campaign_without_source(tmp_path, monkeypatch):
    from supervisor import evolution_lifecycle, queue, state

    state.init(tmp_path)
    queue.init(tmp_path)
    queue.init_queue_refs([], {}, {"value": 0})
    campaign = evolution_lifecycle.start_evolution_campaign("Improve", source="test")
    campaign.pop("source")
    assert evolution_lifecycle._write_evolution_campaign(campaign) is True
    live = state.load_state()
    live.update({"owner_chat_id": 1, "evolution_mode_enabled": True})
    state.save_state(live)
    monkeypatch.setattr(queue, "send_with_budget", lambda *a, **k: None)

    queue.enqueue_evolution_task_if_needed()

    assert state.load_state()["evolution_mode_enabled"] is False


def test_owner_resume_repairs_missing_legacy_campaign_source(tmp_path):
    from supervisor import evolution_lifecycle, queue, state

    state.init(tmp_path)
    queue.init(tmp_path)
    campaign = evolution_lifecycle.start_evolution_campaign("Improve", source="test")
    campaign["status"] = "paused"
    campaign.pop("source")
    assert evolution_lifecycle._write_evolution_campaign(campaign) is True

    resumed = evolution_lifecycle.start_evolution_campaign("", source="owner_chat")

    assert resumed["status"] == "active"
    assert resumed["source"] == "owner_chat"


def test_scheduler_does_not_enqueue_when_transaction_attach_fails(tmp_path, monkeypatch):
    from supervisor import evolution_lifecycle, queue, state

    state.init(tmp_path)
    queue.init(tmp_path)
    pending = []
    queue.init_queue_refs(pending, {}, {"value": 0})
    evolution_lifecycle.start_evolution_campaign("Improve", source="test")
    live = state.load_state()
    live.update({"owner_chat_id": 1, "evolution_mode_enabled": True})
    state.save_state(live)
    monkeypatch.setattr(queue, "begin_evolution_transaction", lambda *a, **k: {})
    monkeypatch.setattr(queue, "send_with_budget", lambda *a, **k: None)

    queue.enqueue_evolution_task_if_needed()

    assert pending == []
    assert state.load_state()["evolution_mode_enabled"] is False


def test_transaction_attach_rechecks_owner_stop_under_state_lock(tmp_path):
    from supervisor import evolution_lifecycle, queue, state

    state.init(tmp_path)
    queue.init(tmp_path)
    campaign = evolution_lifecycle.start_evolution_campaign("Improve", source="test")
    live = state.load_state()
    live.update({"evolution_mode_enabled": False, "evolution_owner_stopped": True})
    state.save_state(live)

    tx = evolution_lifecycle.begin_evolution_transaction(
        "too-late", cycle=1, campaign=campaign,
    )

    assert tx == {}
    assert "active_transaction" not in evolution_lifecycle._read_evolution_campaign()


def test_scheduler_replaces_uncommitted_transaction_lost_before_enqueue(tmp_path, monkeypatch):
    from supervisor import evolution_lifecycle, queue, state

    state.init(tmp_path)
    queue.init(tmp_path)
    pending = []
    queue.init_queue_refs(pending, {}, {"value": 0})
    campaign = evolution_lifecycle.start_evolution_campaign("Improve", source="test")
    live = state.load_state()
    live.update({"owner_chat_id": 1, "evolution_mode_enabled": True})
    state.save_state(live)
    lost = evolution_lifecycle.begin_evolution_transaction(
        "lost-before-enqueue", cycle=1, campaign=campaign,
    )
    monkeypatch.setattr(queue, "send_with_budget", lambda *a, **k: None)

    queue.enqueue_evolution_task_if_needed()

    assert len(pending) == 1
    replacement = pending[0]["metadata"]["evolution_transaction"]
    assert replacement["transaction_id"] != lost["transaction_id"]
    stored = evolution_lifecycle._read_evolution_campaign()
    assert stored["active_transaction"]["transaction_id"] == replacement["transaction_id"]
    assert stored["transaction_history"][-1]["abandoned_reason"] == "dispatch_not_persisted"


def test_scheduler_does_not_replace_transaction_while_worker_is_reaping(tmp_path, monkeypatch):
    from supervisor import evolution_lifecycle, queue

    _campaign, tx = _active_transaction(tmp_path, task_id="reaping-evolution")
    pending = []
    queue.init_queue_refs(pending, {}, {"value": 0})
    assert evolution_lifecycle.update_evolution_transaction(
        tx["task_id"], dispatch_status="reaping",
    )
    monkeypatch.setattr(queue, "send_with_budget", lambda *args, **kwargs: None)

    queue.enqueue_evolution_task_if_needed()

    assert pending == []
    stored = evolution_lifecycle._read_evolution_campaign()["active_transaction"]
    assert stored["transaction_id"] == tx["transaction_id"]
    assert stored["dispatch_status"] == "reaping"


def test_timeout_marks_evolution_reaping_before_scheduler_can_replace_it(tmp_path, monkeypatch):
    from supervisor import evolution_lifecycle, queue

    _campaign, tx = _active_transaction(tmp_path, task_id="timeout-evolution")
    pending = []
    running = {
        tx["task_id"]: {
            "task": {
                "id": tx["task_id"],
                "type": "evolution",
                "chat_id": 1,
                "metadata": {"evolution_transaction": dict(tx)},
            },
            "started_at": 1.0,
            "last_heartbeat_at": 1.0,
            "worker_id": 7,
            "attempt": 1,
        }
    }
    queue.init_queue_refs(pending, running, {"value": 0})
    worker = SimpleNamespace(busy_task_id=tx["task_id"], proc=None, reaping=False)
    workers_view = SimpleNamespace(WORKERS={7: worker})
    reaper_jobs = _CaptureQueue()
    monkeypatch.setattr(queue, "FINALIZATION_GRACE_SEC", 0)
    monkeypatch.setattr(queue, "get_task_idle_timeout_sec", lambda: 1)
    monkeypatch.setattr(queue, "get_per_call_timeout_ceiling_sec", lambda: 1)
    monkeypatch.setattr(queue, "get_task_abs_ceiling_sec", lambda: 10)
    monkeypatch.setattr(queue, "_ensure_reaper_started", lambda: None)
    monkeypatch.setattr(queue, "_reap_queue", reaper_jobs)
    monkeypatch.setattr(queue, "persist_queue_snapshot", lambda reason="": True)
    monkeypatch.setattr(queue, "send_with_budget", lambda *args, **kwargs: None)

    queue._enforce_task_timeouts_locked(
        workers_view, now=1000.0, owner_chat_id=1,
        st={"evolution_mode_enabled": True},
    )

    assert running == {}
    assert worker.reaping is True
    assert len(reaper_jobs.items) == 1
    stored = evolution_lifecycle._read_evolution_campaign()["active_transaction"]
    assert stored["dispatch_status"] == "reaping"

    queue.enqueue_evolution_task_if_needed()
    assert pending == []
    assert evolution_lifecycle._read_evolution_campaign()["active_transaction"][
        "transaction_id"
    ] == tx["transaction_id"]


def test_assignment_dispatches_exact_uncommitted_evolution_claim(tmp_path, monkeypatch):
    workers, task, _tx, worker, inbox, events = _assignment_case(tmp_path, monkeypatch)

    workers.assign_tasks()

    assert inbox.items == [task]
    assert worker.busy_task_id == task["id"]
    assert workers.RUNNING[task["id"]]["task"] == task
    assert events.items == []


def test_assignment_rejects_stale_or_committed_evolution_claim(tmp_path, monkeypatch):
    from ouroboros.task_results import load_task_result
    from supervisor import evolution_lifecycle

    workers, task, tx, worker, inbox, events = _assignment_case(tmp_path, monkeypatch)
    task["metadata"]["evolution_transaction"]["task_id"] = "other-task"

    workers.assign_tasks()

    assert inbox.items == []
    assert workers.RUNNING == {}
    assert worker.busy_task_id is None
    stored = load_task_result(tmp_path, task["id"])
    assert stored["status"] == "cancelled"
    assert stored["reason_code"] == "evolution_authority_missing"
    assert stored["authority_reason"] == "task_mismatch"
    assert events.items[-1]["metadata"]["evolution_transaction"]["task_id"] == "other-task"

    workers, task, tx, _worker, inbox, _events = _assignment_case(
        tmp_path / "committed", monkeypatch, task_id="committed-evo",
    )
    campaign = evolution_lifecycle._read_evolution_campaign()
    assert evolution_lifecycle.record_evolution_commit(
        campaign["id"], tx["transaction_id"], tx["task_id"], "a" * 40,
    )["ok"] is True

    workers.assign_tasks()

    assert inbox.items == []
    assert load_task_result(tmp_path / "committed", task["id"])["authority_reason"] == (
        "transaction_already_committed"
    )


def test_assignment_keeps_invalid_evolution_pending_when_cancel_write_fails(
    tmp_path, monkeypatch,
):
    from supervisor import workers as workers_module

    workers, task, _tx, worker, inbox, events = _assignment_case(tmp_path, monkeypatch)
    task["metadata"]["evolution_transaction"]["task_id"] = "other-task"
    monkeypatch.setattr(
        "ouroboros.task_results.write_task_result",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("disk full")),
    )

    workers.assign_tasks()

    assert workers_module.PENDING == [task]
    assert worker.busy_task_id is None
    assert inbox.items == []
    assert events.items == []


def test_benchmark_seed_creates_campaign_before_enabling(tmp_path):
    from devtools.benchmarks.common.server_runner import seed_owner_state

    seed_owner_state(tmp_path, evolution_enabled=True)

    state = json.loads((tmp_path / "state" / "state.json").read_text())
    campaign = json.loads((tmp_path / "state" / "evolution_campaign.json").read_text())
    assert campaign["status"] == "active"
    assert campaign["id"]
    assert state["evolution_mode_enabled"] is True
