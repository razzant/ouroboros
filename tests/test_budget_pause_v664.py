"""Dispatch-boundary budget pause and explicit replay-safe resume tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest


def _install_queue(tmp_path, monkeypatch):
    from supervisor import queue, state, workers

    state.init(tmp_path, total_budget_limit=10.0)
    queue.init(tmp_path)
    workers.DRIVE_ROOT = tmp_path
    queue.DRIVE_ROOT = tmp_path
    workers.PENDING[:] = []
    workers.RUNNING.clear()
    workers.WORKERS.clear()
    queue.BUDGET_ROOT_FENCES.clear()
    queue.init_queue_refs(workers.PENDING, workers.RUNNING, workers.QUEUE_SEQ_COUNTER_REF)
    monkeypatch.setattr(workers, "load_state", lambda: {"owner_chat_id": 0})
    return queue, state, workers


def test_queued_task_does_not_auto_resume_after_budget_increase(tmp_path, monkeypatch):
    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    sent = []
    worker = SimpleNamespace(
        wid=0,
        busy_task_id=None,
        reaping=False,
        in_q=SimpleNamespace(put=lambda task: sent.append(dict(task))),
    )
    workers.WORKERS[0] = worker
    task = {"id": "paused-task", "type": "task", "chat_id": 0, "priority": 1}
    workers.PENDING.append(task)

    monkeypatch.setattr(state, "budget_remaining", lambda _st, **_kwargs: 0.0)
    workers.assign_tasks()
    pause = workers.PENDING[0]["_budget_pause"]
    assert pause["replay_safe"] is True
    assert pause["auto_resume"] is False
    assert sent == []

    monkeypatch.setattr(state, "budget_remaining", lambda _st, **_kwargs: 10.0)
    workers.assign_tasks()
    assert sent == []
    assert workers.PENDING[0]["id"] == "paused-task"

    resumed = queue.resume_budget_paused_task("paused-task")
    assert resumed == {"ok": True, "task_id": "paused-task", "same_generation": True}
    workers.assign_tasks()
    assert [item["id"] for item in sent] == ["paused-task"]
    assert workers.PENDING == []


def test_resume_rejects_replay_unsafe_pause(tmp_path, monkeypatch):
    queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    workers.PENDING.append({
        "id": "unsafe-task",
        "type": "task",
        "_budget_pause": {
            "status": "resource_limited",
            "physical_calls": 1,
            "replay_safe": False,
        },
    })

    assert queue.resume_budget_paused_task("unsafe-task") == {
        "ok": False,
        "error": "replay_unsafe",
        "action": "cancel_or_new_run",
    }
    assert "_budget_pause" in workers.PENDING[0]


def test_budget_pause_survives_queue_snapshot_restore(tmp_path, monkeypatch):
    queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    workers.PENDING.append({
        "id": "restart-paused",
        "type": "task",
        "chat_id": 0,
        "_budget_pause": {
            "status": "paused_before_dispatch",
            "physical_calls": 0,
            "replay_safe": True,
            "auto_resume": False,
        },
    })
    queue.persist_queue_snapshot(reason="test_budget_pause")
    workers.PENDING[:] = []

    restored = queue.restore_pending_from_snapshot()

    assert restored == 1
    assert workers.PENDING[0]["_budget_pause"]["auto_resume"] is False


def test_worker_budget_pause_event_requeues_same_task_without_task_done(tmp_path, monkeypatch):
    from supervisor.events import _handle_budget_pause

    queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    pushed = []
    persisted = []
    task = {
        "id": "worker-paused",
        "type": "task",
        "chat_id": 7,
        "workspace_root": "/work/subject",
        "_queue_seq": 11,
    }
    worker = SimpleNamespace(busy_task_id="worker-paused")
    workers.RUNNING["worker-paused"] = {"task": task, "worker_id": 0}
    workers.WORKERS[0] = worker
    ctx = SimpleNamespace(
        DRIVE_ROOT=tmp_path,
        RUNNING=workers.RUNNING,
        PENDING=workers.PENDING,
        WORKERS=workers.WORKERS,
        sort_pending=lambda: None,
        persist_queue_snapshot=lambda reason="": persisted.append(reason),
        bridge=SimpleNamespace(push_log=lambda event: pushed.append(event)),
    )
    pause = {
        "status": "paused_before_dispatch",
        "scope": "root",
        "root_task_id": "worker-root",
        "physical_calls": 0,
        "replay_safe": True,
        "auto_resume": False,
        "resume_policy": "manual_same_generation",
    }

    _handle_budget_pause({
        "type": "budget_pause",
        "task_id": "worker-paused",
        "worker_id": 0,
        "resource_limit": pause,
    }, ctx)

    assert ctx.RUNNING == {}
    assert ctx.PENDING[0]["workspace_root"] == "/work/subject"
    assert ctx.PENDING[0]["_queue_seq"] == 11
    stored_pause = ctx.PENDING[0]["_budget_pause"]
    assert stored_pause["root_task_id"] == "worker-root"
    assert stored_pause["replay_safe"] is True
    assert stored_pause["physical_calls"] == 0
    assert stored_pause["fence_id"]
    assert worker.busy_task_id is None
    assert persisted == ["budget_pause_before_dispatch"]
    assert pushed[0]["type"] == "budget_scope_paused"
    assert queue.BUDGET_ROOT_FENCES["worker-root"]["status"] == "paused"
    assert set(queue.BUDGET_ROOT_FENCES["worker-root"]) == {
        "status", "scope", "root_task_id", "fence_id", "auto_resume", "paused_at",
    }


def test_root_budget_fence_is_one_durable_marker_without_subtree_reclassification(
    tmp_path, monkeypatch,
):
    from ouroboros.usage_accounting import AttemptRequest, BudgetExceeded, reserve_attempt
    from supervisor.events import _handle_budget_root_fence

    queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    pending = {
        "id": "safe-sibling", "type": "task", "chat_id": 1,
        "root_task_id": "root-budget", "_attempt": 1,
    }
    current = {
        "id": "spent-current", "type": "task", "chat_id": 1,
        "root_task_id": "root-budget", "_attempt": 1,
    }
    workers.PENDING.append(pending)
    workers.RUNNING[current["id"]] = {"task": current, "worker_id": 0}
    workers.WORKERS[0] = SimpleNamespace(busy_task_id=current["id"])
    ctx = SimpleNamespace(
        DRIVE_ROOT=tmp_path,
        RUNNING=workers.RUNNING,
        PENDING=workers.PENDING,
        WORKERS=workers.WORKERS,
        sort_pending=queue.sort_pending,
        persist_queue_snapshot=queue.persist_queue_snapshot,
        bridge=SimpleNamespace(push_log=lambda _event: None),
    )

    _handle_budget_root_fence({
        "type": "budget_root_fence",
        "task_id": current["id"],
        "worker_id": 0,
        "resource_limit": {
            "scope": "root", "root_task_id": "root-budget",
            "physical_calls": 1, "replay_safe": False,
        },
    }, ctx)

    fence = queue.BUDGET_ROOT_FENCES["root-budget"]
    assert set(fence) == {
        "status", "scope", "root_task_id", "fence_id", "auto_resume", "paused_at",
    }
    assert fence["status"] == "paused"
    assert workers.RUNNING[current["id"]]["task"] == current
    assert workers.WORKERS[0].busy_task_id == current["id"]
    assert workers.PENDING == [pending]

    blocked = queue.enqueue_task({
        "id": "new-child", "type": "task", "chat_id": 1,
        "root_task_id": "root-budget", "parent_task_id": current["id"],
    })
    assert blocked["_admission_blocked"] == "root_budget_fence"
    with pytest.raises(BudgetExceeded) as exc_info:
        reserve_attempt(AttemptRequest(
            model="local/test", provider="local", drive_root=tmp_path,
            task_id=current["id"], root_task_id="root-budget",
            global_limit_usd=100.0, root_limit_usd=100.0,
        ))
    assert exc_info.value.limit_scope == "root"

    workers.PENDING[:] = []
    workers.RUNNING.clear()
    queue.BUDGET_ROOT_FENCES.clear()
    assert queue.restore_pending_from_snapshot() == 1
    assert queue.BUDGET_ROOT_FENCES["root-budget"]["status"] == "paused"
    assert workers.PENDING[0]["id"] == pending["id"]


def test_root_budget_resume_checks_one_task_and_clears_marker(tmp_path, monkeypatch):
    queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    fence_id = "root-fence-id"
    queue.BUDGET_ROOT_FENCES["safe-root"] = {
        "status": "paused", "scope": "root", "root_task_id": "safe-root",
        "fence_id": fence_id, "auto_resume": False, "paused_at": "now",
    }
    workers.PENDING.append({
        "id": "safe-child", "type": "task", "chat_id": 1,
        "root_task_id": "safe-root",
    })
    monkeypatch.setattr(queue, "reconstruct_task_cost", lambda *_a, **_k: {
        "cost_accounting_status": "available",
        "total_rounds": 0,
        "ledger_integrity_degraded": False,
    })

    result = queue.resume_budget_paused_task("safe-child")

    assert result == {"ok": True, "task_id": "safe-child", "same_generation": True}
    assert "safe-root" not in queue.BUDGET_ROOT_FENCES
    assert workers.PENDING[0]["budget_resumed_at"]


def test_root_budget_resume_refuses_unsafe_pending_sibling(tmp_path, monkeypatch):
    queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    fence_id = "root-fence-id"
    queue.BUDGET_ROOT_FENCES["mixed-root"] = {
        "status": "paused", "scope": "root", "root_task_id": "mixed-root",
        "fence_id": fence_id, "auto_resume": False, "paused_at": "now",
    }
    safe = {"id": "safe-child", "type": "task", "root_task_id": "mixed-root"}
    unsafe = {
        "id": "retry-child", "type": "task", "root_task_id": "mixed-root",
        "_attempt": 2, "original_task_id": "first-child",
    }
    workers.PENDING.extend([safe, unsafe])
    monkeypatch.setattr(queue, "reconstruct_task_cost", lambda *_a, **_k: {
        "cost_accounting_status": "available",
        "total_rounds": 0,
        "ledger_integrity_degraded": False,
    })

    result = queue.resume_budget_paused_task("safe-child")

    assert result == {
        "ok": False,
        "error": "root_replay_unsafe",
        "unsafe_task_ids": ["retry-child"],
        "action": "cancel_or_new_run",
    }
    assert "mixed-root" in queue.BUDGET_ROOT_FENCES
    assert "budget_resumed_at" not in safe
