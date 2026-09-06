"""Cross-process transport and durable admission regressions for chat promotion."""

from __future__ import annotations

import json
import multiprocessing as mp
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import pytest


pytestmark = pytest.mark.serial


@pytest.fixture(autouse=True)
def _isolated_projects_root(tmp_path_factory, monkeypatch):
    """Q10=A auto-provisions a genesis workspace for file-less project promotes;
    keep it out of the real ~/Ouroboros/projects."""
    monkeypatch.setenv(
        "OUROBOROS_SUBAGENT_PROJECTS_ROOT",
        str(tmp_path_factory.mktemp("projects_root")),
    )


@pytest.fixture(autouse=True)
def _isolate_event_bus_shutdown_latch():
    """A prior TestClient lifespan must not leak its shutdown latch into tests."""
    import supervisor.workers as workers

    workers._EVENT_Q_SHUTDOWN = False
    try:
        yield
    finally:
        workers._EVENT_Q_SHUTDOWN = False


def _child_put(queue, payload):
    queue.put(payload)


def test_manager_event_bus_accepts_real_spawn_child_after_generation_setup(
    monkeypatch, tmp_path,
):
    import supervisor.workers as workers

    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(workers, "_CTX", None)
    monkeypatch.setattr(workers, "_EVENT_Q", None)
    monkeypatch.setattr(workers, "_EVENT_Q_MANAGER", None)
    monkeypatch.setattr(workers, "_EVENT_Q_GENERATION", "")
    monkeypatch.setattr(workers, "_EVENT_Q_SHUTDOWN", False)
    queue = workers.get_event_q()
    manager = workers._EVENT_Q_MANAGER
    ctx = mp.get_context(workers._WORKER_START_METHOD)
    child = ctx.Process(target=_child_put, args=(queue, {"type": "probe", "value": 7}))
    try:
        child.start()
        child.join(10)
        assert child.exitcode == 0
        assert queue.get(timeout=2) == {"type": "probe", "value": 7}
        ledger = [
            json.loads(line)
            for line in (tmp_path / "state" / "process_ledger.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert ledger[-1]["pid"] == manager._process.pid
        assert ledger[-1]["purpose"] == "supervisor_event_queue_manager"
        assert ledger[-1]["scope"] == "session"
        assert ledger[-1]["pgid"] == 0
    finally:
        if child.is_alive():
            child.terminate()
            child.join(2)
        workers.shutdown_event_q()
        assert not manager._process.is_alive()


def test_concurrent_pool_start_cannot_orphan_a_generation(monkeypatch, tmp_path):
    import supervisor.workers as workers

    event_q = object()
    fake_ctx = MagicMock()
    fake_ctx.Queue.return_value = object()
    created = []

    def make_process(*_args, **_kwargs):
        proc = MagicMock(pid=1000 + len(created))
        created.append(proc)
        return proc

    fake_ctx.Process.side_effect = make_process
    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(workers, "_CTX", fake_ctx)
    monkeypatch.setattr(workers.mp, "get_context", lambda _method: fake_ctx)
    monkeypatch.setattr(workers, "_EVENT_Q", event_q)
    monkeypatch.setattr(workers, "_EVENT_Q_GENERATION", "test-generation")
    monkeypatch.setattr(workers, "WORKERS", {})
    monkeypatch.setattr(workers, "reap_orphaned_workers", lambda: 0)
    monkeypatch.setattr(workers, "_record_worker_pids", lambda: None)
    monkeypatch.setattr(workers, "_verify_worker_sha_after_spawn", lambda *_args: None)
    barrier = threading.Barrier(2)

    def start():
        barrier.wait()
        try:
            workers.spawn_workers(1)
            return "started"
        except RuntimeError:
            return "refused"

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(start) for _ in range(2)]
        outcomes = sorted(future.result() for future in futures)
    assert outcomes == ["refused", "started"]
    assert len(created) == 1
    assert len(workers.WORKERS) == 1


def test_single_slot_respawn_starts_child_without_queue_lock(monkeypatch, tmp_path):
    import supervisor.workers as workers

    lock_was_free = []

    class ProbeProcess:
        pid = 4321
        daemon = False

        def start(self):
            def probe():
                with workers._queue_lock:
                    lock_was_free.append(True)

            thread = threading.Thread(target=probe)
            thread.start()
            thread.join(1)
            assert not thread.is_alive(), "proc.start() ran while queue lock was held"

        def is_alive(self):
            return True

    fake_ctx = MagicMock()
    fake_ctx.Queue.return_value = MagicMock()
    fake_ctx.Process.return_value = ProbeProcess()
    old = workers.Worker(
        wid=0,
        proc=MagicMock(pid=111),
        in_q=MagicMock(),
        busy_task_id=None,
        reaping=True,
    )
    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(workers, "WORKERS", {0: old})
    monkeypatch.setattr(workers, "_get_ctx", lambda: fake_ctx)
    monkeypatch.setattr(workers, "get_event_q", lambda: object())
    monkeypatch.setattr(workers, "_record_worker_pids", lambda: None)
    # The readiness seam is stubbed: a fake pid must never reach the real teardown.
    handed = []
    monkeypatch.setattr(workers, "_verify_worker_sha_after_spawn", lambda slots, *_rest: handed.append(dict(slots)))

    assert workers.respawn_worker(0) is True
    assert lock_was_free == [True]
    assert workers.WORKERS[0] is not old
    # Installed unassignable: the readiness seam opens it once the child confirms ready.
    assert workers.WORKERS[0].reaping is True
    deadline = time.time() + 5
    while not handed and time.time() < deadline:
        time.sleep(0.01)
    assert handed == [{0: workers.WORKERS[0]}]


def test_worker_pool_respawn_reuses_process_event_bus_and_refuses_live_pool(monkeypatch, tmp_path):
    import supervisor.workers as workers

    event_q = object()
    fake_ctx = MagicMock()
    fake_ctx.Queue.side_effect = [object(), object()]
    processes = [MagicMock(pid=101), MagicMock(pid=102)]
    fake_ctx.Process.side_effect = processes
    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(workers, "_CTX", fake_ctx)
    monkeypatch.setattr(workers.mp, "get_context", lambda _method: fake_ctx)
    monkeypatch.setattr(workers, "_EVENT_Q", event_q)
    monkeypatch.setattr(workers, "_EVENT_Q_GENERATION", "test-generation")
    monkeypatch.setattr(workers, "_WORKER_POOL_DISABLED_REASON", "worker_crash_storm")
    monkeypatch.setattr(workers, "WORKERS", {})
    monkeypatch.setattr(workers, "reap_orphaned_workers", lambda: 0)
    monkeypatch.setattr(workers, "_record_worker_pids", lambda: None)
    monkeypatch.setattr(workers.threading, "Thread", MagicMock())

    workers.spawn_workers(1)
    assert fake_ctx.Process.call_args_list[0].kwargs["args"][2] is event_q
    assert workers._EVENT_Q is event_q
    assert workers._WORKER_POOL_DISABLED_REASON == ""

    try:
        workers.spawn_workers(1)
    except RuntimeError as exc:
        assert "requires an empty pool" in str(exc)
    else:
        raise AssertionError("spawn_workers replaced a live pool")
    assert fake_ctx.Process.call_count == 1

    workers.WORKERS.clear()
    workers.spawn_workers(1)
    assert fake_ctx.Process.call_args_list[1].kwargs["args"][2] is event_q
    assert workers._EVENT_Q is event_q


def test_promote_rejects_before_side_effects_when_pool_disabled(tmp_path):
    from ouroboros.task_results import load_task_result
    from supervisor.events import _handle_promote_chat_to_task

    rows = []
    ctx = types.SimpleNamespace(
        DRIVE_ROOT=tmp_path,
        WORKERS={},
        PENDING=[],
        bridge=None,
        append_jsonl=lambda _path, row: rows.append(row),
        persist_queue_snapshot=lambda **_kwargs: True,
        enqueue_task=lambda _task: (_ for _ in ()).throw(AssertionError("must not enqueue")),
        load_state=lambda: {"owner_chat_id": 1},
    )

    outcome = _handle_promote_chat_to_task(
        {
            "type": "promote_chat_to_task",
            "task_id": "nowork01",
            "routing_token": "nowork-token",
            "objective": "Build it",
            "project_id": "must-not-exist",
            "project_name": "Must Not Exist",
            "chat_id": 1,
        },
        ctx,
    )

    assert outcome["reason"] == "worker_pool_unavailable"
    assert not (tmp_path / "state" / "projects.json").exists()
    stored = load_task_result(tmp_path, "nowork01")
    assert stored["status"] == "failed"
    assert stored["promotion_admission"]["status"] == "rejected"
    assert stored["promotion_admission"]["reason"] == "worker_pool_unavailable"
    assert any(row["type"] == "promote_chat_to_task_rejected" for row in rows)


def test_tool_snapshot_precheck_skips_source_side_effects(monkeypatch, tmp_path):
    from ouroboros.tools import control

    state = tmp_path / "state"
    state.mkdir(parents=True)
    (state / "queue_snapshot.json").write_text(
        json.dumps({"worker_pool_disabled_reason": "worker_crash_storm"}),
        encoding="utf-8",
    )
    ctx = types.SimpleNamespace(
        event_queue=None,
        pending_events=[],
        current_chat_id=1,
        drive_root=tmp_path,
    )
    out = control._promote_chat_to_task(
        ctx,
        "Build",
        project_name="Must Not Exist",
        source="/tmp/must-not-attach",
        predecessor_task_id="",
    )
    assert out.startswith("PROMOTE_REJECTED:")
    assert "worker_crash_storm" in out
    assert ctx.pending_events == []


def test_busy_or_reaping_workers_still_allow_queue_admission():
    from supervisor.workers import worker_pool_admission_state

    ctx = types.SimpleNamespace(
        WORKERS={0: types.SimpleNamespace(busy_task_id="other", reaping=True)},
    )
    assert worker_pool_admission_state(ctx)["available"] is True


def test_real_event_queue_reaches_dispatch_and_confirms_durable_admission(
    monkeypatch, tmp_path,
):
    import supervisor.workers as workers
    from ouroboros.task_results import load_task_result
    from ouroboros.tools import control, control_events
    from supervisor.events import _handle_promote_chat_to_task

    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(control_events, "_PROMOTE_CONFIRM_TIMEOUT_SEC", 15.0)
    monkeypatch.setattr(control_events, "_PROMOTE_CONFIRM_POLL_SEC", 0.01)
    pending = []

    def enqueue(task):
        pending.append(dict(task))
        return task

    def persist_snapshot(**_kwargs):
        path = tmp_path / "state" / "queue_snapshot.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"pending": pending, "pending_count": len(pending)}),
            encoding="utf-8",
        )
        return True

    handler_ctx = types.SimpleNamespace(
        DRIVE_ROOT=tmp_path,
        WORKERS={0: types.SimpleNamespace(busy_task_id=None, reaping=False)},
        PENDING=pending,
        bridge=None,
        append_jsonl=lambda *_args, **_kwargs: None,
        persist_queue_snapshot=persist_snapshot,
        enqueue_task=enqueue,
        load_state=lambda: {"owner_chat_id": 1},
    )
    queue = mp.get_context("spawn").Queue()
    tool_ctx = types.SimpleNamespace(
        event_queue=queue,
        pending_events=[],
        current_chat_id=1,
        drive_root=tmp_path,
        task_metadata={},
    )
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(control._promote_chat_to_task, tool_ctx, "Build the racer", predecessor_task_id="")
            event = queue.get(timeout=10)
            assert event["type"] == "promote_chat_to_task"
            outcome = _handle_promote_chat_to_task(event, handler_ctx)
            result_text = future.result(timeout=15)
        task_id = event["task_id"]
        assert outcome == {"status": "scheduled", "task_id": task_id}
        assert result_text.startswith(f"OK: task {task_id}")
        assert "accepted and durably scheduled" in result_text
        assert any(task["id"] == task_id for task in pending)
        snapshot = json.loads((tmp_path / "state" / "queue_snapshot.json").read_text())
        assert snapshot["pending_count"] == 1
        stored = load_task_result(tmp_path, task_id)
        assert stored["status"] == "scheduled"
        assert stored["promotion_admission"] == {
            "status": "scheduled",
            "routing_token": event["routing_token"],
            "confirmed_at": stored["promotion_admission"]["confirmed_at"],
            "queue_snapshot_persisted": True,
            "source_note": "",
            "reason": "",
            "routing_receipt_required": False,
            "routing_receipt_status": "not_applicable",
        }
    finally:
        queue.close()
        queue.cancel_join_thread()


def test_route_to_project_waits_for_same_durable_admission(monkeypatch, tmp_path):
    import supervisor.workers as workers
    from ouroboros.projects_registry import create_project
    from ouroboros.project_dialogue import chat_annotation_receipt
    from ouroboros.task_results import (
        claim_task_acceptance_review_cycle,
        load_task_result,
        review_binding_hash,
    )
    from ouroboros.tools import control, control_events
    from supervisor.events import _handle_promote_chat_to_task

    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(control_events, "_PROMOTE_CONFIRM_TIMEOUT_SEC", 15.0)
    monkeypatch.setattr(control_events, "_PROMOTE_CONFIRM_POLL_SEC", 0.01)
    create_project(tmp_path, "racer", name="Racer")
    pending = []

    def enqueue(task):
        admitted = dict(task)
        admitted["task_contract"] = {
            **task["task_contract"],
            "source": "queue_admitted_test",
        }
        pending.append(admitted)
        return admitted

    handler_ctx = types.SimpleNamespace(
        DRIVE_ROOT=tmp_path,
        WORKERS={0: types.SimpleNamespace()},
        PENDING=pending,
        bridge=None,
        append_jsonl=lambda *_args, **_kwargs: None,
        persist_queue_snapshot=lambda **_kwargs: True,
        enqueue_task=enqueue,
        load_state=lambda: {"owner_chat_id": 1},
    )
    queue = mp.get_context("spawn").Queue()
    tool_ctx = types.SimpleNamespace(
        event_queue=queue,
        pending_events=[],
        current_chat_id=1,
        drive_root=tmp_path,
        task_metadata={"client_message_id": "route-owner-1"},
    )
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                control._route_to_project,
                tool_ctx,
                "racer",
                "Continue the racer",
                "belongs there",
                predecessor_task_id="",
            )
            event = queue.get(timeout=10)
            outcome = _handle_promote_chat_to_task(event, handler_ctx)
            text = future.result(timeout=15)
        assert outcome["status"] == "scheduled"
        assert text.startswith("✉️ Routed to project 'Racer'")
        assert "durably scheduled" in text
        task = next(row for row in pending if row["id"] == event["task_id"])
        assert task["budget_drive_root"] == str(tmp_path)
        assert task["drive_root"] != str(tmp_path)
        stored = load_task_result(tmp_path, event["task_id"])
        assert stored["root_task_id"] == event["task_id"]
        assert stored["delegation_role"] == "root"
        assert stored["task_contract"] == task["task_contract"]
        candidate_hash = "a" * 64
        evidence_revision = "b" * 64
        fence_hash = "c" * 64
        claim = claim_task_acceptance_review_cycle(
            tmp_path,
            event["task_id"],
            {
                "binding_hash": review_binding_hash(
                    candidate_hash=candidate_hash,
                    evidence_revision=evidence_revision,
                    fence_hash=fence_hash,
                ),
                "candidate_hash": candidate_hash,
                "evidence_revision": evidence_revision,
                "fence_hash": fence_hash,
            },
            claimed_by_task_id=event["task_id"],
        )
        assert claim["status"] == "claimed"
        receipt = chat_annotation_receipt(
            tmp_path, "route-owner-1", event["routing_token"]
        )
        assert receipt["action"] == "route_to_project"
        assert receipt["status"] == "scheduled"
    finally:
        queue.close()
        queue.cancel_join_thread()


def test_manual_target_tool_waits_for_durable_handler_receipt(monkeypatch, tmp_path):
    from ouroboros.tools import control, control_events
    from supervisor.events import _handle_routing_manual_target

    monkeypatch.setattr(control_events, "_PROMOTE_CONFIRM_TIMEOUT_SEC", 15.0)
    monkeypatch.setattr(control_events, "_PROMOTE_CONFIRM_POLL_SEC", 0.01)
    queue = mp.get_context("spawn").Queue()
    tool_ctx = types.SimpleNamespace(
        event_queue=queue,
        pending_events=[],
        current_chat_id=1,
        drive_root=tmp_path,
        task_metadata={
            "client_message_id": "manual-owner-1",
            "routing_contract": {"manual_options": [{"kind": "new_task"}]},
        },
    )
    handler_ctx = types.SimpleNamespace(DRIVE_ROOT=tmp_path, bridge=None)
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                control._route_to_project,
                tool_ctx,
                "missing-project",
                "Continue",
                "uncertain target",
                predecessor_task_id="",
            )
            event = queue.get(timeout=10)
            _handle_routing_manual_target(event, handler_ctx)
            text = future.result(timeout=15)
        assert text.startswith("⚠️ NEEDS_MANUAL_TARGET")
        assert 'Host-validated options: [{"kind": "new_task"}]' in text
    finally:
        queue.close()
        queue.cancel_join_thread()


def test_steer_tool_reports_delivery_only_after_mailbox_receipt(
    monkeypatch, tmp_path,
):
    import supervisor.queue as supervisor_queue
    from ouroboros.owner_mailbox import drain_owner_messages
    from ouroboros.tools import control, control_events
    from supervisor.events import _handle_steer_task

    monkeypatch.setattr(supervisor_queue, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(control_events, "_PROMOTE_CONFIRM_TIMEOUT_SEC", 15.0)
    monkeypatch.setattr(control_events, "_PROMOTE_CONFIRM_POLL_SEC", 0.01)
    target = {"id": "target01", "chat_id": 1}
    queue = mp.get_context("spawn").Queue()
    tool_ctx = types.SimpleNamespace(
        event_queue=queue,
        pending_events=[],
        current_chat_id=1,
        drive_root=tmp_path,
        task_metadata={"client_message_id": "steer-owner-1"},
    )
    handler_ctx = types.SimpleNamespace(
        DRIVE_ROOT=tmp_path,
        RUNNING={},
        PENDING=[target],
        bridge=None,
        get_chat_agent=lambda: None,
        persist_queue_snapshot=lambda **_kwargs: True,
    )
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(control._steer_task, tool_ctx, "target01", "Use the new data")
            event = queue.get(timeout=10)
            _handle_steer_task(event, handler_ctx)
            text = future.result(timeout=15)
        assert text.startswith("✉️ Steering task target01")
        assert "durably confirmed" in text
        assert drain_owner_messages(tmp_path, "target01") == ["Use the new data"]
    finally:
        queue.close()
        queue.cancel_join_thread()


def test_stale_live_transport_returns_unconfirmed_not_ok(monkeypatch, tmp_path):
    from ouroboros.tools import control, control_events

    monkeypatch.setattr(control_events, "_PROMOTE_CONFIRM_TIMEOUT_SEC", 0.05)
    monkeypatch.setattr(control_events, "_PROMOTE_CONFIRM_POLL_SEC", 0.005)
    stale_queue = mp.get_context("spawn").Queue()
    ctx = types.SimpleNamespace(
        event_queue=stale_queue,
        pending_events=[],
        current_chat_id=1,
        drive_root=tmp_path,
        task_metadata={},
    )
    try:
        out = control._promote_chat_to_task(ctx, "Never drained", predecessor_task_id="")
        event = stale_queue.get(timeout=10)
        assert event["type"] == "promote_chat_to_task"
        assert out.startswith("PROMOTE_UNCONFIRMED:")
        assert "Do not report this task as created" in out
        assert not (tmp_path / "task_results" / f"{event['task_id']}.json").exists()
    finally:
        stale_queue.close()
        stale_queue.cancel_join_thread()


def test_stale_task_result_and_receipt_cannot_confirm_new_admission_token(tmp_path):
    from ouroboros.task_results import STATUS_SCHEDULED, write_task_result
    from ouroboros.tools.control import _wait_for_promotion_admission

    old_token = "a" * 32
    new_token = "b" * 32
    write_task_result(
        tmp_path,
        "deadbeef",
        STATUS_SCHEDULED,
        promotion_admission={"status": "scheduled", "routing_token": old_token},
    )
    ctx = types.SimpleNamespace(drive_root=tmp_path)
    assert _wait_for_promotion_admission(
        ctx, "deadbeef", new_token, timeout_sec=0.0,
    ) == {
        "status": "unconfirmed",
        "reason": "confirmation_timeout",
    }


def test_unpicklable_control_event_fails_before_feeder_thread(tmp_path):
    from ouroboros.tools.control import _emit_control_event

    ctx = types.SimpleNamespace(
        event_queue=mp.get_context("spawn").Queue(),
        pending_events=[],
        drive_root=tmp_path,
    )
    try:
        mode = _emit_control_event(
            ctx,
            {"type": "promote_chat_to_task", "task_id": "pickle01", "bad": lambda: None},
        )
        assert mode == "serialization_failed"
        assert ctx.pending_events == []
        rows = [
            json.loads(line)
            for line in (tmp_path / "logs" / "supervisor.jsonl").read_text().splitlines()
        ]
        assert rows[-1]["type"] == "control_event_serialization_failed"
        assert rows[-1]["task_id"] == "pickle01"
    finally:
        ctx.event_queue.close()
        ctx.event_queue.cancel_join_thread()


def test_snapshot_persistence_failure_rolls_back_pending(monkeypatch, tmp_path):
    import supervisor.workers as workers
    from ouroboros.task_results import load_task_result
    from supervisor.events import _handle_promote_chat_to_task

    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    pending = []

    def enqueue(task):
        pending.append(dict(task))
        return task

    ctx = types.SimpleNamespace(
        DRIVE_ROOT=tmp_path,
        WORKERS={0: types.SimpleNamespace()},
        PENDING=pending,
        bridge=None,
        enqueue_task=enqueue,
        persist_queue_snapshot=lambda **_kwargs: False,
        load_state=lambda: {"owner_chat_id": 1},
        append_jsonl=lambda *_args, **_kwargs: None,
    )
    outcome = _handle_promote_chat_to_task(
        {
            "type": "promote_chat_to_task",
            "task_id": "snapfail",
            "routing_token": "snapfail-token",
            "objective": "Build",
        },
        ctx,
    )
    assert outcome["reason"] == "queue_snapshot_persist_failed"
    assert pending == []
    stored = load_task_result(tmp_path, "snapfail")
    assert stored["promotion_admission"]["status"] == "rejected"


def test_missing_snapshot_persister_fails_closed(monkeypatch, tmp_path):
    import supervisor.workers as workers

    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    pending = []
    ctx = types.SimpleNamespace(
        enqueue_task=lambda task: pending.append(dict(task)) or task,
        load_state=lambda: {"owner_chat_id": 1},
    )
    outcome = workers.promote_chat_to_task(
        {"task_id": "no-persister", "objective": "Build"},
        ctx,
    )
    assert outcome["status"] == "needs_manual_target"
    assert outcome["reason"] == "queue_snapshot_persist_unavailable"
    assert len(pending) == 1


def test_routing_receipt_failure_cannot_produce_positive_confirmation(
    monkeypatch, tmp_path,
):
    import supervisor.workers as workers
    from ouroboros.task_results import load_task_result
    from supervisor.events import _handle_promote_chat_to_task

    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(
        "ouroboros.project_dialogue.append_chat_annotation",
        lambda *_args, **_kwargs: False,
    )
    pending = []

    def enqueue(task):
        pending.append(dict(task))
        return task

    ctx = types.SimpleNamespace(
        DRIVE_ROOT=tmp_path,
        WORKERS={0: types.SimpleNamespace()},
        PENDING=pending,
        bridge=None,
        enqueue_task=enqueue,
        persist_queue_snapshot=lambda **_kwargs: True,
        load_state=lambda: {"owner_chat_id": 1},
        append_jsonl=lambda *_args, **_kwargs: None,
    )
    outcome = _handle_promote_chat_to_task(
        {
            "type": "promote_chat_to_task",
            "task_id": "receiptfail",
            "routing_token": "receiptfail-token",
            "objective": "Build",
            "client_message_id": "owner-receipt-fail",
            "chat_id": 1,
        },
        ctx,
    )
    assert outcome["reason"] == "routing_annotation_persist_failed"
    assert outcome["status"] == "unconfirmed"
    assert len(pending) == 1
    stored = load_task_result(tmp_path, "receiptfail")
    assert stored["promotion_admission"]["status"] == "unconfirmed"


def test_gateway_refuses_explicitly_disabled_pool_before_task_side_effects(
    monkeypatch, tmp_path,
):
    import supervisor.workers as workers
    from ouroboros.gateway.tasks import _supervisor_ready_error

    ready = threading.Event()
    ready.set()
    request = types.SimpleNamespace(
        app=types.SimpleNamespace(
            state=types.SimpleNamespace(supervisor_ready_event=ready),
        )
    )
    monkeypatch.setattr(workers, "WORKERS", {})
    monkeypatch.setattr(workers, "_WORKER_POOL_DISABLED_REASON", "worker_crash_storm")
    response = _supervisor_ready_error(request)
    payload = json.loads(response.body)
    assert response.status_code == 503
    assert payload["reason_code"] == "worker_pool_unavailable"
    assert payload["worker_pool_disabled_reason"] == "worker_crash_storm"
    assert not (tmp_path / "task_results").exists()


def test_admission_reservation_rejects_tokenless_competing_enqueue(
    monkeypatch, tmp_path,
):
    import supervisor.queue as supervisor_queue

    pending = []
    monkeypatch.setattr(supervisor_queue, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(supervisor_queue, "PENDING", pending)
    monkeypatch.setattr(supervisor_queue, "RUNNING", {})
    monkeypatch.setattr(supervisor_queue, "ADMISSION_RESERVATIONS", {})
    assert supervisor_queue.reserve_task_admission(
        "owned-task", "owner-token", drive_root=tmp_path
    )["status"] == "reserved"

    loser = supervisor_queue.enqueue_task({"id": "owned-task", "type": "task"})

    assert loser["_admission_blocked"] == "admission_reservation_owned"
    assert pending == []
    assert supervisor_queue.ADMISSION_RESERVATIONS == {
        "owned-task": "owner-token"
    }


def test_exact_id_ingress_fails_closed_on_unreadable_result(monkeypatch, tmp_path):
    import supervisor.queue as supervisor_queue
    import supervisor.workers as workers

    result_path = tmp_path / "task_results" / "malformed-id.json"
    result_path.parent.mkdir()
    malformed = b"{not-json"
    result_path.write_bytes(malformed)
    monkeypatch.setattr(supervisor_queue, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(supervisor_queue, "PENDING", [])
    monkeypatch.setattr(supervisor_queue, "RUNNING", {})
    monkeypatch.setattr(supervisor_queue, "ADMISSION_RESERVATIONS", {})

    reservation = supervisor_queue.reserve_task_admission(
        "malformed-id", "reservation-token", drive_root=tmp_path,
    )
    queued = supervisor_queue.enqueue_task({
        "id": "malformed-id",
        "type": "task",
        "_require_unique_task_id": True,
    })
    duplicate_reason = workers._promote_duplicate_reason(
        "malformed-id", types.SimpleNamespace(
            DRIVE_ROOT=tmp_path, PENDING=[], RUNNING={},
        ),
    )

    assert reservation == {"status": "blocked", "reason": "task_id_lookup_failed"}
    assert queued["_admission_blocked"] == "task_id_lookup_failed"
    assert duplicate_reason == "task_id_lookup_failed"
    assert supervisor_queue.PENDING == []
    assert result_path.read_bytes() == malformed

    empty_path = tmp_path / "task_results" / "empty-id.json"
    empty_path.write_text("{}\n", encoding="utf-8")
    assert supervisor_queue.reserve_task_admission(
        "empty-id", "empty-token", drive_root=tmp_path,
    ) == {"status": "blocked", "reason": "task_id_lookup_failed"}
    empty_queued = supervisor_queue.enqueue_task({
        "id": "empty-id", "type": "task", "_require_unique_task_id": True,
    })
    assert empty_queued["_admission_blocked"] == "task_id_lookup_failed"
    assert empty_path.read_text(encoding="utf-8") == "{}\n"


def test_promote_lookup_failure_never_overwrites_exact_result(tmp_path):
    from supervisor.events import _persist_promote_rejection

    result_path = tmp_path / "task_results" / "promote-corrupt.json"
    result_path.parent.mkdir()
    original = b"{corrupt"
    result_path.write_bytes(original)
    _persist_promote_rejection(
        types.SimpleNamespace(DRIVE_ROOT=tmp_path),
        {"task_id": "promote-corrupt", "routing_token": "token"},
        {"task_id": "promote-corrupt", "reason": "task_id_lookup_failed"},
    )
    assert result_path.read_bytes() == original


def test_project_registry_lookup_failure_prevents_clone(monkeypatch, tmp_path):
    from ouroboros.promotion_source import resolve_promote_source

    monkeypatch.setattr(
        "ouroboros.projects_registry.get_reserved_project",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("registry unreadable")),
    )
    monkeypatch.setattr(
        "ouroboros.project_sources.clone_project_repo",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("clone must not start")
        ),
    )
    ctx = types.SimpleNamespace(DRIVE_ROOT=tmp_path, REPO_DIR=tmp_path / "repo")

    folder, note, error, project_id, source_created = resolve_promote_source(
        ctx, "https://github.com/example/project.git", "project"
    )

    assert folder == ""
    assert note == ""
    assert error.startswith("project_lookup_failed: OSError: registry unreadable")
    assert project_id == "project"
    assert source_created is False


def test_duplicate_promote_uses_negative_annotation_without_overwriting_result(
    monkeypatch, tmp_path,
):
    import supervisor.queue as supervisor_queue
    import supervisor.workers as workers
    from ouroboros.task_results import STATUS_SCHEDULED, load_task_result, write_task_result
    from ouroboros.tools.control import _wait_for_promotion_admission
    from supervisor.events import _handle_promote_chat_to_task

    old_token = "old-token"
    new_token = "new-token"
    write_task_result(
        tmp_path,
        "duplicate-task",
        STATUS_SCHEDULED,
        promotion_admission={"status": "scheduled", "routing_token": old_token},
    )
    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(supervisor_queue, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(supervisor_queue, "PENDING", [])
    monkeypatch.setattr(supervisor_queue, "RUNNING", {})
    monkeypatch.setattr(supervisor_queue, "ADMISSION_RESERVATIONS", {})
    rows = []
    ctx = types.SimpleNamespace(
        DRIVE_ROOT=tmp_path,
        WORKERS={0: types.SimpleNamespace()},
        bridge=None,
        append_jsonl=lambda _path, row: rows.append(row),
    )
    outcome = _handle_promote_chat_to_task(
        {
            "type": "promote_chat_to_task",
            "task_id": "duplicate-task",
            "routing_token": new_token,
            "objective": "Competing request",
            "chat_id": 1,
            "client_message_id": "duplicate-owner-message",
        },
        ctx,
    )

    assert outcome["reason"] == "duplicate_task_id"
    confirmation = _wait_for_promotion_admission(
        types.SimpleNamespace(drive_root=tmp_path),
        "duplicate-task",
        new_token,
        client_message_id="duplicate-owner-message",
        timeout_sec=0.0,
    )
    assert confirmation["status"] == "needs_manual_target"
    assert confirmation["reason"] == "duplicate_task_id"
    assert load_task_result(tmp_path, "duplicate-task")["promotion_admission"] == {
        "status": "scheduled",
        "routing_token": old_token,
    }


def test_source_resolution_runs_off_supervisor_loop_and_continues_once(
    monkeypatch, tmp_path,
):
    import queue as thread_queue

    import supervisor.queue as supervisor_queue
    import supervisor.workers as workers
    from ouroboros.task_results import load_task_result
    from supervisor.events import _handle_promote_chat_to_task

    pending = []
    started = threading.Event()
    release = threading.Event()
    continuation_bus = thread_queue.Queue()

    def slow_resolve(_ctx, _source, project_id):
        started.set()
        assert release.wait(2)
        return "", "source checked", "", project_id, False

    monkeypatch.setattr(
        "ouroboros.promotion_source.resolve_promote_source", slow_resolve
    )
    monkeypatch.setattr(workers, "get_event_q", lambda: continuation_bus)
    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(workers, "WORKERS", {0: types.SimpleNamespace()})
    monkeypatch.setattr(workers, "_WORKER_POOL_DISABLED_REASON", "")
    monkeypatch.setattr(supervisor_queue, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(supervisor_queue, "PENDING", pending)
    monkeypatch.setattr(supervisor_queue, "RUNNING", {})
    monkeypatch.setattr(supervisor_queue, "ADMISSION_RESERVATIONS", {})
    ctx = types.SimpleNamespace(
        DRIVE_ROOT=tmp_path,
        WORKERS={0: types.SimpleNamespace()},
        PENDING=pending,
        bridge=None,
        enqueue_task=supervisor_queue.enqueue_task,
        persist_queue_snapshot=lambda **_kwargs: True,
        load_state=lambda: {"owner_chat_id": 1},
        append_jsonl=lambda *_args, **_kwargs: None,
    )
    event = {
        "type": "promote_chat_to_task",
        "task_id": "source-task",
        "routing_token": "source-token",
        "objective": "Inspect source",
        "source": "https://github.com/example/project.git",
        "chat_id": 1,
    }

    first = _handle_promote_chat_to_task(event, ctx)
    assert first == {"status": "preparing", "task_id": "source-task"}
    assert started.wait(0.5)
    assert pending == []
    duplicate = _handle_promote_chat_to_task(event, ctx)
    assert duplicate == {"status": "preparing", "task_id": "source-task"}

    release.set()
    continuation = continuation_bus.get(timeout=2)
    final = _handle_promote_chat_to_task(continuation, ctx)

    assert final["status"] == "scheduled"
    assert [row["id"] for row in pending] == ["source-task"]
    assert load_task_result(tmp_path, "source-task")["promotion_admission"][
        "routing_token"
    ] == "source-token"
