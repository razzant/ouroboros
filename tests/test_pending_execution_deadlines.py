"""#939 (owner choice 2A): an explicit calendar deadline also ends a queued row, without dispatch.

Driven through the real queue, timeout tick, assignment, reaper job, result and
delivery seams. No default queue deadline exists; the execution duration never
creates one. Ordinary resource holds cannot erase it. Stop, a saved terminal,
owner Pause and continuation custody win; a failed publication retains the row.
"""
from __future__ import annotations

import datetime
import json
import queue as stdqueue
import time
from types import SimpleNamespace

import pytest

from ouroboros.task_results import load_task_result, write_task_result
from supervisor import queue, workers
from tests.test_swarm_host_admission import host  # noqa: F401
from tests._store_shared import delete_result, result_exists, unreadable_result
from tests.worker_pool_support import affirm_ready

pytestmark = pytest.mark.serial


@pytest.fixture
def clock(host, monkeypatch):  # noqa: F811
    from supervisor import queue_timeouts

    now, frames = [time.time()], []
    jobs = stdqueue.Queue()
    monkeypatch.setattr(queue, "_ensure_reaper_started", lambda: None)
    monkeypatch.setattr(queue, "_reap_queue", jobs)
    monkeypatch.setattr(queue_timeouts, "_pending_deadline_jobs", {})

    def reap():
        while not jobs.empty():
            job = jobs.get_nowait()
            assert job["kind"] == "pending_deadline"
            try:
                queue_timeouts._reap_pending_deadline(job)
            finally:
                jobs.task_done()

    monkeypatch.setattr(time, "time", lambda: now[0])
    monkeypatch.setattr(queue, "load_state", lambda: {})
    monkeypatch.setattr(workers, "load_state", lambda: {})
    monkeypatch.setattr(workers, "get_event_q", lambda **_kw: SimpleNamespace(put=frames.append))
    monkeypatch.setattr("supervisor.state.budget_remaining", lambda *_a, **_kw: 5)
    monkeypatch.setattr("supervisor.worker_owner_wait.maintain_owner_wait_capacity", lambda: None)
    workers.WORKERS.clear()  # no worker capacity: the timeout tick alone settles a pending-only queue
    return SimpleNamespace(now=now, frames=frames, jobs=jobs, reap=reap)


def add_task(host, clock, *, task_id="pending", deadline=True, source="task", **fields):  # noqa: F811
    task = {"id": task_id, "type": "task", "chat_id": 1, "text": "Solve the queued work",
            "execution_timeout_sec": 5, **fields}
    if deadline:
        stamp = datetime.datetime.fromtimestamp(clock.now[0] + 30, datetime.timezone.utc).isoformat()
        if source == "task":
            task["deadline_at"] = stamp
        else:
            task[source] = {**task.get(source, {}), "deadline_at": stamp}
    admitted = queue.enqueue_task(task)
    assert not admitted.get("_admission_blocked")
    queue.ensure_control_task_result(task_id)
    return admitted


def _done(clock):
    return [frame["status"] for frame in clock.frames if frame.get("type") == "task_done"]


def _resource_hold(host, clock, monkeypatch, kind, *, deadline=True):  # noqa: F811
    """Real admission/hold producers, without a worker or a paid attempt."""
    from supervisor.events_budget import _set_root_budget_pause_locked, hold_root_resume_descendants

    if kind in {"root_budget", "root_selection"}:
        add_task(host, clock, task_id="root", deadline=False)
    task = add_task(host, clock, deadline=deadline,
                    **({"root_task_id": "root", "parent_task_id": "root", "depth": 1,
                        "delegation_role": "subagent"} if kind.startswith("root_") else {}))
    if kind == "global_budget":
        with monkeypatch.context() as fault:
            fault.setattr("supervisor.state.budget_remaining", lambda *_a, **_kw: 0)
            workers.assign_tasks()
        assert task["_budget_pause"]["status"] == "paused_before_dispatch"
        assert task["_budget_pause"]["replay_safe"] is True
    elif kind.startswith("root_"):
        with queue._queue_lock:
            fence = _set_root_budget_pause_locked("root", {"reason": "budget"})
            if kind == "root_selection":
                held, _, _ = hold_root_resume_descendants(
                    queue, "root", fence, {"grant_id": "root-grant", "generation": 1})
                assert held == ["pending"]
                assert "cause" not in task["_budget_pause_hold"]
    assert task["admitted_dispatch"] == "none"
    return task


@pytest.mark.parametrize("kind", ["global_budget", "root_budget", "root_selection"])
@pytest.mark.parametrize("deadline", [False, True], ids=["duration-only", "explicit-calendar"])
def test_never_started_resource_wait_reaches_deadline_outcome_without_resume(
    host, clock, monkeypatch, kind, deadline,  # noqa: F811
):
    task = _resource_hold(host, clock, monkeypatch, kind, deadline=deadline)
    clock.now[0] += 100
    queue.enforce_task_timeouts()
    clock.reap()
    result = load_task_result(host.root, "pending")
    if deadline:
        assert task not in host.pending and result["status"] == "failed"
        assert result["reason_code"] == "deadline" and _done(clock) == ["failed"]
    else:
        assert task in host.pending and result["status"] == "scheduled"
        assert not clock.frames and clock.jobs.empty()
    assert not host.running and not host.attempts


def _project_hold(host, clock, tmp_path, monkeypatch):  # noqa: F811
    """A live accepted Project loses its result, then waits under the real assignment hold."""
    from tests.test_project_hold_recovery import accepted
    from ouroboros import projects_registry

    # Admission needs a declared pool even when no slot is free to dispatch.
    workers.WORKERS[0] = SimpleNamespace(busy_task_id="occupied", reaping=False)
    try:
        task = accepted(host, tmp_path, tid="pending")
    finally:
        workers.WORKERS.clear()
    task["deadline_at"] = datetime.datetime.fromtimestamp(
        clock.now[0] + 30, datetime.timezone.utc).isoformat()
    unreadable = unreadable_result(monkeypatch, "pending")
    workers.assign_tasks()
    assert task["_project_admission_restore_hold"]["reason"] == "project_routing_fence_lookup_failed"
    unreadable["on"] = False
    # Scope remains unavailable, but the accepted row's no-dispatch receipt is readable again.
    projects_registry._registry_path(host.root).write_text("{torn", encoding="utf-8")
    return task


@pytest.mark.parametrize("held", ["global_budget", "project"])
def test_api_queued_deadline_uses_real_admission_receipt(host, clock, tmp_path, monkeypatch, held):  # noqa: F811
    from starlette.requests import Request
    from ouroboros.gateway.tasks import api_tasks_create
    from ouroboros import projects_registry

    body = {"description": "API queued work", "timeout_sec": 30, "execution_timeout_sec": 5}
    if held == "project":
        projects_registry.create_project(host.root, "target", name="Original Project")
        body["project_id"] = "target"

    async def receive():
        return {"type": "http.request", "body": json.dumps(body).encode()}

    app = SimpleNamespace(state=SimpleNamespace(drive_root=host.root, repo_dir=workers.REPO_DIR))
    request = Request({"type": "http", "method": "POST", "path": "/api/tasks", "headers": [], "app": app}, receive)
    workers.WORKERS[0] = SimpleNamespace(busy_task_id="occupied", reaping=False)
    try:
        response = host.run_async(api_tasks_create(request))
    finally:
        workers.WORKERS.clear()
    assert response.status_code == 200, response.body
    [task] = host.pending
    task_id = json.loads(response.body)["task_id"]
    stored = load_task_result(host.root, task_id)
    assert stored["api_admission"]["status"] == "accepted"
    assert task["admitted_dispatch"] == "none" and task["execution_timeout_sec"] == 5
    if held == "global_budget":
        with monkeypatch.context() as fault:
            fault.setattr("supervisor.state.budget_remaining", lambda *_a, **_kw: 0)
            workers.assign_tasks()
        assert task["_budget_pause"]["replay_safe"] is True
    else:
        unreadable = unreadable_result(monkeypatch, task_id)
        workers.assign_tasks()
        assert task["_project_admission_restore_hold"]
        unreadable["on"] = False
        projects_registry._registry_path(host.root).write_text("{torn", encoding="utf-8")
    clock.now[0] += 100
    queue.enforce_task_timeouts()
    clock.reap()
    result = load_task_result(host.root, task_id)
    assert result["status"] == "failed" and result["reason_code"] == "deadline"
    assert not host.pending and not host.running and not host.attempts
    assert _done(clock) == ["failed"]


def test_never_started_project_wait_expires_on_the_tick_without_capacity(host, clock, tmp_path, monkeypatch):  # noqa: F811
    task = _project_hold(host, clock, tmp_path, monkeypatch)
    basis = task["_project_admission"]
    clock.now[0] += 100
    queue.enforce_task_timeouts()
    clock.reap()
    result = load_task_result(host.root, "pending")
    assert result["status"] == "failed" and result["reason_code"] == "deadline"
    assert task["_project_admission"] == basis  # no fresh scope or replay authority
    assert not host.pending and not host.running and not host.attempts
    assert _done(clock) == ["failed"]


@pytest.mark.parametrize("veto", ["owner_pause", "result_missing", "result_unreadable", "possible", "started",
                                    "unknown_dispatch", "exact_pause", "resume", "owner_wait", "working_recovery"])
def test_project_deadline_does_not_take_protected_or_unknown_custody(
    host, clock, tmp_path, monkeypatch, veto,  # noqa: F811
):
    from supervisor.owner_pause_control import request_owner_pause

    task = _project_hold(host, clock, tmp_path, monkeypatch)
    if veto == "owner_pause":
        assert request_owner_pause("pending", request_id="owner-pause")["ok"]
    elif veto.startswith("result_"):
        delete_result(host.root, "pending") if veto == "result_missing" else unreadable_result(monkeypatch, "pending")
    elif veto == "possible":
        # A stale queued 'none' must not override the durable pre-handoff claim.
        write_task_result(host.root, "pending", "scheduled", admitted_dispatch="possible")
    elif veto == "started":
        write_task_result(host.root, "pending", "scheduled", started_at="2026-01-01T00:00:00Z")
    elif veto == "unknown_dispatch":
        task.pop("admitted_dispatch")
    else:
        field, value = {
            "exact_pause": ("_budget_pause", {"exact_continuation": True, "reason": "budget"}),
            "resume": ("_budget_pause_resume", {"grant_id": "granted"}),
            "owner_wait": ("_owner_wait_resume", {"started_at": clock.now[0] - 5}),
            "working_recovery": ("_working_recovery", {"source_task_id": "pending", "from_attempt": 1}),
        }[veto]
        task[field] = value
    clock.now[0] += 100
    queue.enforce_task_timeouts()
    workers.assign_tasks()  # Project revalidation must apply the same custody precedence.
    clock.reap()
    assert host.pending == [task] and "_terminalization_retry" not in task
    assert not clock.frames and clock.jobs.empty() and not host.attempts
    if not veto.startswith("result_"):
        assert load_task_result(host.root, "pending")["status"] == "scheduled"


@pytest.mark.parametrize("kind", ["global_budget", "root_budget", "root_selection"])
def test_owner_pause_over_a_resource_wait_survives_the_deadline(host, clock, monkeypatch, kind):  # noqa: F811
    from supervisor.owner_pause_control import request_owner_pause

    task = _resource_hold(host, clock, monkeypatch, kind)
    root_id = str(task.get("root_task_id") or task["id"])
    assert request_owner_pause(root_id, request_id="pause-resource-wait")["ok"]
    # Even if the queue latch has not landed, the durable Pause remains authoritative.
    queue.BUDGET_ROOT_FENCES.clear()
    clock.now[0] += 100
    queue.enforce_task_timeouts()
    clock.reap()
    assert task in host.pending and "_terminalization_retry" not in task
    assert load_task_result(host.root, "pending")["status"] == "scheduled"
    assert not clock.frames and not host.attempts


def test_owner_pause_accepted_after_expiry_marking_wins_before_settlement(host, clock):  # noqa: F811
    from supervisor.owner_pause_control import request_owner_pause

    task = add_task(host, clock)
    clock.now[0] += 100
    queue.enforce_task_timeouts()
    assert clock.jobs.qsize() == 1
    assert request_owner_pause("pending", request_id="pause-before-settlement")["ok"]
    clock.reap()
    assert host.pending == [task] and load_task_result(host.root, "pending")["status"] == "scheduled"
    assert not clock.frames and not host.attempts


def test_a_held_receipt_lost_after_expiry_marking_is_not_recreated(host, clock, monkeypatch):  # noqa: F811
    task = _resource_hold(host, clock, monkeypatch, "global_budget")
    clock.now[0] += 100
    queue.enforce_task_timeouts()
    assert clock.jobs.qsize() == 1
    delete_result(host.root, "pending")
    clock.reap()
    assert host.pending == [task] and not result_exists(host.root, "pending")
    assert not clock.frames and not host.attempts


def test_an_unreadable_held_child_row_is_custody_not_a_tick_failure(host, clock, monkeypatch):  # noqa: F811
    task = _resource_hold(host, clock, monkeypatch, "root_budget")
    unreadable = unreadable_result(monkeypatch, "pending")  # the root's own row stays readable
    clock.now[0] += 100
    queue.enforce_task_timeouts()
    clock.reap()
    assert host.pending[-1] is task and "_terminalization_retry" not in task and clock.jobs.empty()
    unreadable["on"] = False
    queue.enforce_task_timeouts()
    clock.reap()
    assert task not in host.pending and load_task_result(host.root, "pending")["reason_code"] == "deadline"


@pytest.mark.parametrize("stop_during_custody", ["none", "requested", "settled"])
def test_expiry_uses_one_offloop_reaper_job_and_keeps_the_queue_responsive(
    host, clock, monkeypatch, stop_during_custody,  # noqa: F811
):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    task = add_task(host, clock)
    clock.now[0] += 100
    entered, release = Event(), Event()

    def slow_custody(*_a, **_kw):
        entered.set()
        assert release.wait(5), "test did not release custody"

    monkeypatch.setattr(workers, "_audit_delegate_terminal_custody", slow_custody)
    queue.enforce_task_timeouts()
    queue.enforce_task_timeouts()
    workers.assign_tasks()
    assert clock.jobs.qsize() == 1  # one reservation per exact row
    assert host.pending == [task] and not host.attempts
    assert load_task_result(host.root, "pending")["status"] == "scheduled"
    assert not clock.frames

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(clock.reap)
        try:
            assert entered.wait(5)
            assert queue._queue_lock.acquire(timeout=1), "custody still holds the queue lock"
            queue._queue_lock.release()
            queue.enforce_task_timeouts()  # another tick runs while remote custody is outstanding
            assert clock.jobs.empty()
            if stop_during_custody != "none":
                from ouroboros.cancel_intents import request_cancel

                request_cancel(host.root, "pending", source="test", reason="Stop during remote custody")
                if stop_during_custody == "settled":
                    assert queue.cancel_task_custody("pending") in {queue.CANCEL_CANCELLED, queue.CANCEL_ALREADY_SETTLED}
                    assert not host.pending
                    stop_frames = list(clock.frames)
        finally:
            release.set()
        future.result(timeout=5)
    if stop_during_custody == "settled":
        assert clock.frames == stop_frames
        assert load_task_result(host.root, "pending")["status"] == "cancelled"
        return
    if stop_during_custody == "requested":
        assert host.pending == [task] and not clock.frames  # Stop wins; the deadline settles nothing
        assert load_task_result(host.root, "pending")["status"] == "scheduled"
        assert queue.cancel_task_custody("pending") in {queue.CANCEL_CANCELLED, queue.CANCEL_ALREADY_SETTLED}
        assert load_task_result(host.root, "pending")["status"] == "cancelled"
        return
    assert not host.pending and not host.attempts
    assert _done(clock) == ["failed"]


def test_one_expiry_pass_does_not_rescan_pending_per_expired_row(host, clock, monkeypatch):  # noqa: F811
    """The locked pass reserves the rows it just read; it does not rescan PENDING under
    the queue lock once per expired row. The reaper keeps its own membership check."""
    for index in range(4):
        add_task(host, clock, task_id=f"pending-{index}")
    clock.now[0] += 100
    passes = []

    class CountingPending(list):
        def __iter__(self):
            passes.append(1)
            return super().__iter__()

    counted = CountingPending(host.pending)
    for module in (workers, queue):
        monkeypatch.setattr(module, "PENDING", counted)
    queue.enforce_task_timeouts()
    assert clock.jobs.qsize() == 4
    assert len(passes) < 4  # the pass and its snapshot, not one scan per expired row
    for module in (workers, queue):
        monkeypatch.setattr(module, "PENDING", host.pending)
    clock.reap()
    assert not host.pending and _done(clock) == ["failed"] * 4


def test_a_stale_deadline_job_cannot_settle_a_replaced_pending_row(host, clock):  # noqa: F811
    add_task(host, clock)
    clock.now[0] += 100
    queue.enforce_task_timeouts()
    replacement = dict(host.pending[0])
    host.pending[:] = [replacement]
    clock.reap()
    assert host.pending == [replacement] and not clock.frames
    assert load_task_result(host.root, "pending")["status"] == "scheduled"
    queue.enforce_task_timeouts()
    clock.reap()
    assert not host.pending
    assert load_task_result(host.root, "pending")["reason_code"] == "deadline"


@pytest.mark.parametrize("source", ["task", "metadata", "task_contract"])
def test_an_explicit_deadline_expires_without_capacity_but_a_duration_never_does(host, clock, source):  # noqa: F811
    add_task(host, clock, source=source)
    unbounded = add_task(host, clock, task_id="no-calendar-cap", deadline=False)
    clock.now[0] += 100
    queue.enforce_task_timeouts()
    clock.reap()
    result = load_task_result(host.root, "pending")
    assert result["status"] == "failed" and result["reason_code"] == "deadline"
    assert host.pending == [unbounded] and not host.running and not host.attempts
    assert load_task_result(host.root, "no-calendar-cap")["status"] == "scheduled"
    types = [frame["type"] for frame in clock.frames]
    assert types.index("send_message") < types.index("task_done")  # the outbox owes delivery first
    from supervisor.terminal_delivery import pending_deliveries

    assert any(row["task_id"] == "pending" for row in pending_deliveries(host.root))


@pytest.mark.parametrize("failure", ["result", "delivery"])
def test_a_failed_publication_keeps_custody_through_snapshot_restore(host, clock, monkeypatch, failure):  # noqa: F811
    add_task(host, clock, _execution_handoff_sec=2)
    clock.now[0] += 100
    with monkeypatch.context() as fault:
        if failure == "result":
            import ouroboros.task_results as results

            original = results.write_task_result

            def refuse_expiry(root, task_id, status, **fields):
                if status == "failed":
                    raise OSError("isolated injected disk refusal")
                return original(root, task_id, status, **fields)

            fault.setattr(results, "write_task_result", refuse_expiry)
        else:
            fault.setattr("supervisor.terminal_delivery.register_pending_delivery", lambda *_a, **_kw: False)
        queue.enforce_task_timeouts()
        clock.reap()
    assert host.pending[0]["_terminalization_retry"]["trigger"] == "deadline"
    assert not _done(clock)
    assert queue.persist_queue_snapshot()
    host.pending.clear()
    assert queue.restore_pending_from_snapshot() == 1
    assert host.pending[0]["execution_timeout_sec"] == 5 and host.pending[0]["_execution_handoff_sec"] == 2
    workers.assign_tasks()
    clock.reap()
    assert not host.pending and not host.running and not host.attempts
    assert load_task_result(host.root, "pending")["reason_code"] == "deadline"
    assert _done(clock) == ["failed"]


@pytest.mark.parametrize("held", ["_owner_hold", "_budget_pause", "_budget_pause_hold"])
def test_paused_and_held_rows_keep_their_own_custody_at_the_deadline(host, clock, held):  # noqa: F811
    add_task(host, clock, **{held: {"reason": "confirmed pause"}})
    clock.now[0] += 100
    queue.enforce_task_timeouts()
    workers.assign_tasks()
    assert clock.jobs.empty() and not clock.frames and not host.attempts
    assert host.pending and "_terminalization_retry" not in host.pending[0]
    assert load_task_result(host.root, "pending")["status"] == "scheduled"


@pytest.mark.parametrize("owner", ["running", "worker", "direct"])
def test_a_deadline_never_terminalizes_a_still_live_owner(host, clock, monkeypatch, owner):  # noqa: F811
    task = add_task(host, clock, _is_direct_chat=owner == "direct")
    if owner == "running":
        host.running["pending"] = {"task": task}  # owned, without a started clock
    elif owner == "worker":
        workers.WORKERS[0] = SimpleNamespace(busy_task_id="pending")
    else:
        monkeypatch.setattr("supervisor.worker_assignment._direct_actor_still_registered", lambda _tid: True)
    clock.now[0] += 100
    queue.enforce_task_timeouts()
    clock.reap()
    assert host.pending and load_task_result(host.root, "pending")["status"] == "scheduled"
    assert not clock.frames and not host.attempts


def test_owner_stop_keeps_its_intent_and_custody_at_pending_expiry(host, clock):  # noqa: F811
    from ouroboros.cancel_intents import active_intents, request_cancel

    add_task(host, clock)
    request_cancel(host.root, "pending", source="test", reason="Owner requested Stop")
    before = active_intents(host.root, strict=True)
    clock.now[0] += 100
    queue.enforce_task_timeouts()
    clock.reap()
    assert active_intents(host.root, strict=True) == before
    assert load_task_result(host.root, "pending")["status"] == "scheduled"
    assert not clock.frames and not host.attempts
    assert queue.cancel_task_custody("pending") in {queue.CANCEL_CANCELLED, queue.CANCEL_ALREADY_SETTLED}
    assert load_task_result(host.root, "pending")["status"] == "cancelled"
    assert not host.pending


def test_a_saved_completion_wins_over_a_late_pending_expiry(host, clock):  # noqa: F811
    add_task(host, clock)
    write_task_result(host.root, "pending", "completed", result="Already saved answer")
    clock.now[0] += 100
    queue.enforce_task_timeouts()
    clock.reap()
    assert load_task_result(host.root, "pending")["result"] == "Already saved answer"
    assert _done(clock) == ["completed"]
    assert not host.pending and not host.attempts


@pytest.mark.parametrize("ingress", ["queue", "api"])
def test_a_deadline_crossed_during_assignment_preparation_never_hands_off(host, clock, monkeypatch, ingress):  # noqa: F811
    from supervisor import worker_assignment

    if ingress == "api":
        from starlette.requests import Request
        from ouroboros.gateway.tasks import api_tasks_create

        async def receive():
            return {"type": "http.request", "body": json.dumps({"description": "API work", "timeout_sec": 30}).encode()}

        app = SimpleNamespace(state=SimpleNamespace(drive_root=host.root, repo_dir=workers.REPO_DIR))
        request = Request({"type": "http", "method": "POST", "path": "/api/tasks", "headers": [], "app": app}, receive)
        workers.WORKERS[0] = SimpleNamespace(busy_task_id="occupied", reaping=False)
        try:
            response = host.run_async(api_tasks_create(request))
        finally:
            workers.WORKERS.clear()
        assert response.status_code == 200, response.body
        [task] = host.pending
        assert load_task_result(host.root, task["id"])["api_admission"]["status"] == "accepted"
    else:
        task = add_task(host, clock)
    assert task["admitted_dispatch"] == "none"
    dispatched = []
    mirrored = []
    worker = SimpleNamespace(in_q=SimpleNamespace(put=dispatched.append))
    original = worker_assignment._mirror_assigned_running_status

    def mirror_then_deadline(candidate):
        original(candidate)
        assert candidate["admitted_dispatch"] == "possible"
        mirrored.append(load_task_result(host.root, candidate["id"]))
        clock.now[0] += 100  # persistence and receipts took the remaining calendar time

    monkeypatch.setattr(worker_assignment, "_mirror_assigned_running_status", mirror_then_deadline)
    with queue._queue_lock:
        assert not worker_assignment._claim_worker_launch(queue, task, worker)
    assert len(mirrored) == 1  # the final guard, after the real receipt/mirror calls
    if ingress == "api":
        assert mirrored[0]["admitted_dispatch"] == "possible"
        assert mirrored[0]["status"] == "running"
        assert mirrored[0]["execution_owner"] == task["_execution_owner"]
    assert task["_terminalization_retry"]["trigger"] == "deadline"
    queue.enforce_task_timeouts()
    clock.reap()
    assert not dispatched and not host.pending and not host.attempts
    assert load_task_result(host.root, task["id"])["reason_code"] == "deadline"


@pytest.mark.parametrize("calendar", [False, True], ids=["duration-only", "deadline-passed"])
def test_a_restored_row_keeps_its_budget_and_carry_until_resume_and_assignment(host, clock, calendar):  # noqa: F811
    """A restart holds the queued row (saved work); only the owner's Resume releases it."""
    from ouroboros.model_wait import execution_elapsed_seconds
    from supervisor.events_budget import budget_hold_fact

    add_task(host, clock, deadline=calendar, _execution_handoff_sec=2)
    assert queue.persist_queue_snapshot()
    snapshot = json.loads(queue.QUEUE_SNAPSHOT_PATH.read_text())
    assert snapshot["pending"][0]["task"]["execution_timeout_sec"] == 5
    host.pending.clear()
    clock.now[0] += 500  # restart and queue time are not execution; any calendar deadline passes
    assert queue.restore_pending_from_snapshot() == 1
    assert budget_hold_fact(host.pending[0]) is not None
    queue.enforce_task_timeouts()
    if calendar:  # a positive never-started receipt expires even under the saved-work hold
        clock.reap()
        assert not host.pending and not host.running and not host.attempts
        assert load_task_result(host.root, "pending")["reason_code"] == "deadline"
        return
    assert clock.jobs.empty() and not clock.frames
    assert queue.resume_budget_paused_task("pending")["ok"]
    dispatched = []
    workers.WORKERS[0] = affirm_ready(SimpleNamespace(wid=0, busy_task_id=None, reaping=False,
                                         in_q=SimpleNamespace(put=dispatched.append)))
    workers.assign_tasks()
    clock.reap()
    assert not host.attempts
    assert len(dispatched) == 1 and dispatched[0]["execution_timeout_sec"] == 5
    running = host.running["pending"]
    assert running["task"]["_execution_consumed_sec"] == 2 and "_execution_handoff_sec" not in running["task"]
    assert execution_elapsed_seconds(running, clock.now[0]) == 2


def test_a_carried_handoff_never_regrants_spent_execution(host, clock, monkeypatch):  # noqa: F811
    """A successor carries its ended attempt's execution through the one stamp the crash
    requeue and planned successors use; dispatch never regrants it, and the ceiling it
    reaches queues a terminal-only job."""
    from ouroboros.model_wait import stamp_execution_handoff

    task = add_task(host, clock, deadline=False, execution_timeout_sec=500)
    host.pending.clear()
    write_task_result(host.root, "pending", "running", result="Some work was performed")
    meta = {"task": task, "attempt": 1, "started_at": clock.now[0] - 200}
    successor = {**task, "_attempt": 2}
    stamp_execution_handoff(successor, meta)
    assert not queue.enqueue_task(successor, front=True).get("_admission_blocked")
    assert host.pending[0]["_execution_handoff_sec"] == 200 and host.pending[0]["execution_timeout_sec"] == 500
    sent = []
    workers.WORKERS[0] = affirm_ready(SimpleNamespace(wid=0, busy_task_id=None, reaping=False,
                                         in_q=SimpleNamespace(put=sent.append)))
    monkeypatch.setattr(queue, "get_task_abs_ceiling_sec", lambda: None)
    monkeypatch.setattr(queue, "FINALIZATION_GRACE_SEC", 0)
    reaps = stdqueue.Queue()
    monkeypatch.setattr(queue, "_reap_queue", reaps)
    workers.assign_tasks()
    assert len(sent) == 1
    clock.now[0] += 299
    host.running["pending"]["last_progress_at"] = clock.now[0]
    queue.enforce_task_timeouts()
    assert reaps.empty()
    clock.now[0] += 1
    queue.enforce_task_timeouts()
    job = reaps.get_nowait()
    assert job["runtime_sec"] == 500 and job["terminal_reason"] == "absolute_ceiling"
    assert not {"will_retry", "retry_task_id"} & set(job) and not host.pending and not host.attempts


def test_pending_deadline_salvage_reads_history_outside_the_supervisor_queue_lock(host, clock, monkeypatch):  # noqa: F811
    """The salvage answer is owed before the terminal row, but its response-history read (a
    manifest scan and a full copy) runs before Q is taken: admission and control never wait on it."""
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from ouroboros import observability

    add_task(host, clock)
    clock.now[0] += 100
    queue.enforce_task_timeouts()
    entered, release = threading.Event(), threading.Event()
    actual = observability.latest_llm_response_text

    def paused_reader(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return actual(*args, **kwargs)

    monkeypatch.setattr(observability, "latest_llm_response_text", paused_reader)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(clock.reap)
        try:
            assert entered.wait(5)
            responsive = queue._queue_lock.acquire(blocking=False)
            if responsive:
                queue._queue_lock.release()
        finally:
            release.set()
        future.result(timeout=10)
    assert responsive, "the deadline salvage read holds Q and blocks supervisor admission and control"
    stored = load_task_result(host.root, "pending")
    assert stored["status"] == "failed"  # the settlement itself still completes
