"""Hurry seeds only missing pooled lifecycle truth, before its owner-only projection."""

from __future__ import annotations

import json
import queue as stdqueue
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from ouroboros import task_results as results
from ouroboros.gateway.task_hurry import _admit_hurry_locked, api_task_hurry
from ouroboros.owner_mailbox import KIND_HURRY, _mailbox_path, drain_owner_entries
from ouroboros.utils import atomic_write_json
from supervisor import queue, state, task_reaper, worker_health, workers
from tests._usage_store_testing import ledger_rows


@pytest.fixture
def pool(tmp_path, monkeypatch):
    from supervisor import git_ops, task_lifecycle

    pending, running = [], {}
    for module in (queue, workers):
        monkeypatch.setattr(module, "DRIVE_ROOT", tmp_path)
        monkeypatch.setattr(module, "PENDING", pending)
        monkeypatch.setattr(module, "RUNNING", running)
    for name in ("ADMISSION_RESERVATIONS", "ACCEPTANCE_FENCES", "BUDGET_ROOT_FENCES"):
        monkeypatch.setattr(queue, name, {})
    monkeypatch.setattr(task_lifecycle, "BUDGET_ROOT_FENCES", queue.BUDGET_ROOT_FENCES)
    monkeypatch.setattr(task_lifecycle, "CANCELLED_ROOT_FENCES", {})
    monkeypatch.setattr(task_lifecycle, "_ACTIVE_CASCADE_FENCES", {})
    monkeypatch.setattr(queue, "QUEUE_SEQ_COUNTER_REF", {"value": 0})
    monkeypatch.setattr(queue, "QUEUE_SNAPSHOT_PATH", tmp_path / "state/queue_snapshot.json")
    for name, value in {
        "DRIVE_ROOT": tmp_path, "STATE_PATH": tmp_path / "state/state.json",
        "STATE_LAST_GOOD_PATH": tmp_path / "state/state.last_good.json",
        "STATE_LOCK_PATH": tmp_path / "locks/state.lock", "TOTAL_BUDGET_LIMIT": 0,
    }.items():
        monkeypatch.setattr(state, name, value)
    for module in (workers, git_ops):
        monkeypatch.setattr(module, "REPO_DIR", tmp_path / "repo")
    monkeypatch.setattr(git_ops, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(workers, "MAX_WORKERS", 1)
    monkeypatch.setattr(workers, "_WORKER_POOL_DISABLED_REASON", "")
    monkeypatch.setattr(workers, "_repo_writer_gate_reason", "")
    monkeypatch.setattr(workers, "CRASH_TS", [])
    monkeypatch.setattr(workers, "_LAST_SPAWN_TIME", 0)
    monkeypatch.setattr(workers, "QUEUE_MAX_RETRIES", 1)
    monkeypatch.setattr(workers, "load_state", state.load_state)
    monkeypatch.setattr(queue, "load_state", state.load_state)
    monkeypatch.setattr(workers, "send_with_budget", lambda *a, **k: None)
    monkeypatch.setattr(queue, "send_with_budget", lambda *a, **k: None)
    monkeypatch.setattr(workers, "_reconcile_confirmed_dead_review_owner", lambda *a: None)
    monkeypatch.setattr("ouroboros.tools.services.archive_task_service_logs", lambda *a: None)
    monkeypatch.setattr("ouroboros.delegate_recovery.reconcile_unrecoverable_task", lambda *a: None)
    events, jobs, respawns = stdqueue.Queue(), stdqueue.Queue(), []
    monkeypatch.setattr(workers, "get_event_q", lambda: events)
    monkeypatch.setattr(queue, "_reap_queue", jobs)
    monkeypatch.setattr(queue, "_ensure_reaper_started", lambda: None)
    monkeypatch.setattr(task_reaper, "_deferred_reap_jobs", [])
    monkeypatch.setattr(workers, "respawn_worker", respawns.append)
    proc = SimpleNamespace(pid=None, alive=True, exitcode=None)
    proc.is_alive = lambda: proc.alive
    slot = workers.Worker(0, proc, stdqueue.Queue())
    monkeypatch.setattr(workers, "WORKERS", {0: slot})
    state.save_state({})  # an initialized install: only explicit init creates state (#1307)
    state.update_state(lambda row: row.update(owner_chat_id=1))
    return SimpleNamespace(root=tmp_path, slot=slot, events=events, jobs=jobs, respawns=respawns)


def _enqueue_origin(pool, monkeypatch, origin):
    if origin in {"review", "restore"}:
        task_id = queue.queue_deep_self_review_task("owner:/review", force=True, chat_id=1)
        if origin == "restore":
            from supervisor.events_budget import HOLD_SAVED_WORK, budget_hold_fact

            queue.PENDING.clear()
            assert queue.restore_pending_from_snapshot() == 1
            # This is an application-crash restore, with no acknowledged Restart
            # transaction. The accepted task waits for the owner's Resume (#1563).
            assert budget_hold_fact(queue.PENDING[0])["reason"] == HOLD_SAVED_WORK
            assert queue.resume_budget_paused_task(task_id)["ok"]
            assert budget_hold_fact(queue.PENDING[0]) is None
    elif origin == "evolution":
        from supervisor import git_ops

        # Only Git observation is replaced; campaign claim/admission are real.
        monkeypatch.setattr(git_ops, "git_capture", lambda argv: (0, "baseline", ""))
        atomic_write_json(pool.root / "state/evolution_campaign.json", {
            "id": "campaign", "source": "owner", "status": "active", "objective": "inspect",
        })
        state.update_state(lambda row: row.update(evolution_mode_enabled=True))
        queue.enqueue_evolution_task_if_needed()
        task_id = queue.PENDING[0]["id"]
    elif origin == "plain":  # a receipt-less row: a legacy snapshot or a failed resolver receipt
        task_id = queue.enqueue_task({"id": "plain", "type": "task", "chat_id": 1})["id"]
    else:
        from supervisor import update_merge

        task_id = "update_assisted_merge_test"
        tx = {"task_id": task_id, "phase": "assisted_resolution", "owner_chat_id": 1,
              "target_sha": "target", "pre_update_sha": "baseline", "resolver_submitted_id": ""}
        update_merge.write_update_tx(tx)
        assert update_merge.enqueue_assisted_resolution_task(tx) == task_id
        assert not workers.worker_pool_admission_state()["available"]
        assert workers.repo_writer_task_allowed(queue.PENDING[0])
    assert len(queue.PENDING) == 1
    if origin == "plain":  # hurry seeds the receipt-less row's missing lifecycle
        assert not results.task_result_path(pool.root, task_id, create=False).exists()
    else:  # host producers admit with their own positive scheduled receipt
        receipt = results.load_task_result(pool.root, task_id, strict=True)
        assert receipt["status"] == "scheduled" and receipt["host_admission"]["status"] == "accepted"
    return task_id


def _hurry(root, task_id):
    app = Starlette(routes=[Route("/api/tasks/{task_id}/hurry", api_task_hurry, methods=["POST"])])
    app.state.drive_root = root
    with TestClient(app) as client:
        return client.post(f"/api/tasks/{task_id}/hurry", json={"request_id": "request-1"})


@pytest.mark.parametrize("origin", ["review", "evolution", "assisted", "restore", "plain"])
@pytest.mark.parametrize("phase", ["pending", "assigned"])
@pytest.mark.parametrize("exitcode", [-11, 1])
def test_real_admission_hurry_and_pre_running_death_recover(pool, monkeypatch, origin, phase, exitcode):
    task_id = _enqueue_origin(pool, monkeypatch, origin)
    before_hurry = results.load_task_result(pool.root, task_id, strict=True)
    if phase == "assigned":
        workers.assign_tasks()
        assert task_id in queue.RUNNING
        before_hurry = results.load_task_result(pool.root, task_id, strict=True)
        assert before_hurry["status"] == ("requested" if origin == "plain" else "scheduled")
        assert before_hurry["admitted_dispatch"] == "possible"
    response = _hurry(pool.root, task_id)
    assert response.status_code == 200, response.text
    row = results.load_task_result(pool.root, task_id, strict=True)
    if phase == "pending" and origin == "plain":
        # Only a receipt-less row is seeded, from its exact admitted queue facts.
        assert row["status"] == "scheduled"
        admitted = queue.PENDING[0]
        persisted = {key for key in (
            "type", "chat_id", "metadata", "task_contract", "parent_task_id", "delegation_role",
            "project_id", "workspace_root", "workspace_mode", "memory_mode", "budget_drive_root",
            "queued_at", "admitted_dispatch", "_admission_owner_token", "origin_message_text",
            "origin_message_ref", "objective", "title", "suggested_name", "original_task_id", "timeout_retry_from",
            "deadline_at", "root_cost_ceiling_usd", "billing_group", "task_constraint", "objective_author",
            "owner_corpus", "task_group_id", "task_group",
        ) if key in admitted}
        assert set(row) == ({"task_id", "status", "_schema_version", "ts", "updated_at", "owner_hurry",
                             "root_task_id", "description", "task_attempt"} | persisted)
        assert all(row[key] == admitted[key] for key in persisted)
        assert row["root_task_id"] == task_id and row["description"] == admitted.get("text", "")
        # The host attempt key (terminal_time witness), never the queue-private `_attempt`.
        assert row["task_attempt"] == admitted["_attempt"]
    else:
        assert {key: value for key, value in row.items() if key != "owner_hurry"} == before_hurry
    assert row["owner_hurry"]["attempt_key"] == 1
    assert [entry["kind"] for entry in drain_owner_entries(pool.root, task_id)] == [KIND_HURRY]
    assert not (pool.root / "logs/chat.jsonl").exists()
    assert not ledger_rows(pool.root)
    if phase == "pending":
        workers.assign_tasks()
    assert pool.slot.in_q.get_nowait()["id"] == task_id
    assert task_id in queue.RUNNING and not queue.PENDING
    pool.slot.proc.alive, pool.slot.proc.exitcode = False, exitcode
    workers.ensure_workers_healthy()
    job = pool.jobs.get_nowait()
    assert job["kind"] == "confirmed_dead_worker"
    worker_health.recover_confirmed_dead_worker(job)
    assert task_id not in queue.RUNNING and pool.jobs.empty()
    assert pool.respawns == [0]
    row = results.load_task_result(pool.root, task_id, strict=True)
    if exitcode < 0:
        assert row["status"] == "failed" and row["reason_code"] == "worker_crash_signal"
        assert len([event for event in pool.events.queue if event.get("type") == "task_done"]) == 1
        assert not queue.PENDING and row["owner_hurry"]["request_id"] == "request-1"
    else:
        assert row["status"] == "interrupted"
        assert [(task["id"], task["_attempt"]) for task in queue.PENDING] == [(task_id, 2)]
        assert "owner_hurry" not in row
        assert row["owner_hurry_history"][-1]["request_id"] == "request-1"


def test_initializer_holds_queue_lock_and_does_not_rewrite_existing_wait(pool, monkeypatch):
    task_id = _enqueue_origin(pool, monkeypatch, "plain")
    actual = results.write_task_result
    calls = []

    def observe(*args, **kwargs):
        with ThreadPoolExecutor(max_workers=1) as executor:
            assert executor.submit(queue._queue_lock.acquire, False).result() is False
        calls.append(kwargs)
        return actual(*args, **kwargs)

    monkeypatch.setattr(results, "write_task_result", observe)
    _admit_hurry_locked(task_id)
    assert len(calls) == 1
    assert calls[0]["create_only"] is True and calls[0]["strict_existing_dict"] is True
    row = results.load_task_result(pool.root, task_id, strict=True)
    assert {key: value for key, value in calls[0].items() if key not in {"create_only", "strict_existing_dict"}} == {
        key: value for key, value in row.items() if key not in {"task_id", "status", "_schema_version", "ts", "updated_at"}}
    assert row["root_task_id"] == task_id and row["metadata"] == queue.PENDING[0]["metadata"]
    actual(pool.root, task_id, "running", owner_wait={"state": "waiting", "wait_id": "w", "source_ref": {"sha256": "source"}},
           metadata={"grant": "preserved"}, started_at="original", model_waits={"quota": "kept"})
    path = results.task_result_path(pool.root, task_id)
    before = path.read_bytes()
    calls.clear()
    _admit_hurry_locked(task_id)
    assert not calls and path.read_bytes() == before
    assert _hurry(pool.root, task_id).status_code == 200
    row = results.load_task_result(pool.root, task_id, strict=True)
    assert {key: value for key, value in row.items() if key != "owner_hurry"} == json.loads(before)


def test_hurry_does_not_release_restored_work_before_owner_resume(pool):
    from supervisor.events_budget import HOLD_SAVED_WORK, budget_hold_fact

    task_id = queue.queue_deep_self_review_task("owner:/review", force=True, chat_id=1)
    queue.PENDING.clear()
    assert queue.restore_pending_from_snapshot() == 1
    [restored] = queue.PENDING
    hold = dict(budget_hold_fact(restored))
    assert hold["reason"] == HOLD_SAVED_WORK
    workers.assign_tasks()
    assert not queue.RUNNING and pool.slot.in_q.empty()
    response = _hurry(pool.root, task_id)
    assert response.status_code == 200, response.text
    assert budget_hold_fact(restored) == hold
    workers.assign_tasks()
    assert not queue.RUNNING and pool.slot.in_q.empty()
    assert results.load_task_result(pool.root, task_id)["owner_hurry"]["request_id"] == "request-1"
    assert queue.resume_budget_paused_task(task_id)["ok"]
    workers.assign_tasks()
    assert pool.slot.in_q.get_nowait()["id"] == task_id
    assert pool.slot.in_q.empty() and set(queue.RUNNING) == {task_id}


@pytest.mark.parametrize("status", ["scheduled", "running", "interrupted", "completed", "cancelled", "future_status"])
def test_create_only_observes_late_row_under_real_file_lock(tmp_path, monkeypatch, status):
    from ouroboros import platform_layer, utils

    path = results.task_result_path(tmp_path, "race")
    entered, contender, release = threading.Event(), threading.Event(), threading.Event()
    late = {"task_id": "race", "status": status, "_schema_version": 1, "updated_at": "original",
            "cost_usd": 7, "metadata": {"late_grant": True}, "owner_wait": {"wait_id": "kept"}}
    written = []
    actual_atomic, actual_acquire = utils.atomic_write_json, platform_layer.acquire_exclusive_file_lock

    def record_write(*args, **kwargs):
        actual_atomic(*args, **kwargs)
        written.append(path.read_bytes())

    def acquire(*args, **kwargs):
        if entered.is_set():
            contender.set()
        return actual_acquire(*args, **kwargs)

    def publish(current):
        assert not current
        entered.set()
        assert release.wait(5)
        return late

    monkeypatch.setattr(utils, "atomic_write_json", record_write)
    monkeypatch.setattr(platform_layer, "acquire_exclusive_file_lock", acquire)
    with ThreadPoolExecutor(max_workers=2) as executor:
        holder = executor.submit(utils.update_json_locked, path, publish)
        assert entered.wait(5)
        creator = executor.submit(results.write_task_result, tmp_path, "race", "scheduled",
                                  create_only=True, strict_existing_dict=True,
                                  metadata={"late_grant": False}, result="must not land")
        try:
            assert contender.wait(5)
        finally:
            release.set()
        assert holder.result() == late and creator.result() == late
    assert len(written) == 1 and path.read_bytes() == written[0]


def test_create_only_creates_once_and_default_writer_still_merges(tmp_path):
    first = results.write_task_result(tmp_path, "new", "scheduled", create_only=True, strict_existing_dict=True)
    assert first["status"] == "scheduled" and "create_only" not in first
    second = results.write_task_result(tmp_path, "new", "running", result="normal update")
    assert second["status"] == "running" and second["result"] == "normal update"


@pytest.mark.parametrize("raw", [
    "{broken", "[]", "{}",
    '{"_schema_version":1,"owner_hurry":{}}',
    '{"_schema_version":1,"task_id":"other","status":"running"}',
    '{"_schema_version":2,"task_id":"bad","status":"running"}',
    '{"_schema_version":true,"task_id":"bad","status":"running"}',
])
def test_invalid_existing_bytes_refuse_create_and_hurry(pool, monkeypatch, raw):
    from ouroboros import owner_hurry

    queue.enqueue_task({"id": "bad", "type": "task", "chat_id": 1})
    path = results.task_result_path(pool.root, "bad")
    path.write_text(raw, encoding="utf-8")
    before = path.read_bytes()
    with pytest.raises(ValueError):
        results.write_task_result(pool.root, "bad", "scheduled", create_only=True, strict_existing_dict=True)
    monkeypatch.setattr(owner_hurry, "record_requested", lambda *a, **k: pytest.fail("unknown authority reached hurry"))
    assert _hurry(pool.root, "bad").status_code == 503
    assert path.read_bytes() == before and not _mailbox_path(pool.root, "bad").exists()


@pytest.mark.parametrize("failure", ["read", "write"])
def test_storage_failure_does_not_write_hurry(pool, monkeypatch, failure):
    from ouroboros import owner_hurry

    tid = _enqueue_origin(pool, monkeypatch, "plain")
    def fail(*args, **kwargs):
        raise OSError("storage unavailable")
    monkeypatch.setattr(results, "load_task_result" if failure == "read" else "write_task_result", fail)
    monkeypatch.setattr(owner_hurry, "record_requested", lambda *a, **k: pytest.fail("failed seed reached hurry"))
    assert _hurry(pool.root, tid).status_code == 503
    assert not results.task_result_path(pool.root, tid, create=False).exists()
    assert not _mailbox_path(pool.root, tid).exists()


def test_direct_fallback_never_initializes_lifecycle(pool, monkeypatch):
    task = {"id": "direct", "type": "task", "chat_id": 1, "_is_direct_chat": True}
    monkeypatch.setattr(workers, "direct_chat_turn", lambda task_id: task)
    monkeypatch.setattr(results, "write_task_result", lambda *a, **k: pytest.fail("direct lifecycle initialized"))
    assert _hurry(pool.root, "direct").status_code == 200
    row = results.load_task_result(pool.root, "direct")
    assert set(row) == {"owner_hurry", "_schema_version"}, "keep the existing direct fallback contract"


@pytest.mark.parametrize("accepting", [False, True])
def test_actual_direct_actor_admission_keeps_its_existing_contract(pool, monkeypatch, accepting):
    from supervisor.active_activity import get_direct_activity_registry

    actor = SimpleNamespace(_busy=True, _accepting_owner_messages=accepting,
                            _current_task_id="actor", _current_task_metadata={}, _current_task_text="work")
    get_direct_activity_registry().register("actor", 1, actor=actor,
                                           kind="direct_chat")
    monkeypatch.setattr(results, "write_task_result", lambda *a, **k: pytest.fail("direct seed"))
    assert _hurry(pool.root, "actor").status_code == (200 if accepting else 404)
    row = results.load_task_result(pool.root, "actor") or {}
    assert "status" not in row


def test_initializer_runs_only_after_existing_admission_checks(pool, monkeypatch):
    from ouroboros.cancel_intents import request_cancel

    child = queue.enqueue_task({"id": "child", "parent_task_id": "root", "root_task_id": "root",
                                "delegation_role": "subagent", "chat_id": 1})
    assert not child.get("_admission_blocked"), child
    queue.enqueue_task({"id": "sealed", "chat_id": 1})
    queue.ACCEPTANCE_FENCES["sealed"] = {"status": "sealed"}
    queue.enqueue_task({"id": "cancelled", "chat_id": 1})
    request_cancel(pool.root, "cancelled", requested_by="owner", reason="stop")
    monkeypatch.setattr(results, "write_task_result", lambda *a, **k: pytest.fail("refused task seed"))
    for tid, code in [("child", 409), ("sealed", 409), ("cancelled", 409), ("absent", 404)]:
        assert _hurry(pool.root, tid).status_code == code
        assert not results.task_result_path(pool.root, tid, create=False).exists()
