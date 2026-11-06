"""#939/#1183: a crash retry's time carry and fence release through the real producers.

The worker-crash retry (``worker_health``) is driven by the real dead-worker job,
the real enqueue and the real ``assign_tasks``; the worker side builds its clock
from the row it was actually sent, through the real agent start seam. A preserved original start already spans the
dead attempt, so nothing is added again; a fresh start carries the ended total
once. The dead attempt's own acceptance fence never refuses its retry.
"""
from __future__ import annotations

import copy
import json
import time
from types import SimpleNamespace

from tests.worker_pool_support import affirm_ready

import pytest

from ouroboros.model_wait import execution_elapsed_seconds
from ouroboros.task_results import write_task_result
from tests._execution_clock_seams import agent_start as _agent_start
from tests.test_retained_work_recovery import _saved
from tests.test_worker_crash_retry import _isolate_worker_crash_state, _reserved_job  # noqa: F401



def _fake_clock(monkeypatch):
    clock = [time.time()]
    monkeypatch.setattr(time, "time", lambda: clock[0])
    monkeypatch.setattr(time, "monotonic", lambda: clock[0] - 1_000_000)
    return clock


def _assign_one(workers, state, monkeypatch, wid=7):
    sent = []
    workers.WORKERS[wid] = affirm_ready(SimpleNamespace(wid=wid, busy_task_id=None, reaping=False,
                                           in_q=SimpleNamespace(put=lambda row: sent.append(copy.deepcopy(row)))))
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_k: 5.0)
    monkeypatch.setattr(workers, "repo_writer_task_allowed", lambda task: True)  # no repository writer here
    import supervisor.queue as q

    monkeypatch.setattr(q, "persist_queue_snapshot", lambda **_k: True)  # the claim needs a confirmed snapshot
    workers.assign_tasks()
    return sent


@pytest.mark.parametrize("bounded", [True, False], ids=["task-budget", "legacy-global"])
@pytest.mark.parametrize("queue_delay", [0, 600])
@pytest.mark.parametrize("saved_work", [True, False], ids=["preserved-origin", "fresh-start"])
def test_crash_retry_carries_execution_once_through_assignment(tmp_path, monkeypatch, saved_work, queue_delay, bounded):
    import supervisor.queue as q
    import supervisor.state as state
    import supervisor.workers as W
    from supervisor.worker_health import recover_confirmed_dead_worker

    clock = _fake_clock(monkeypatch)
    job, _events = _reserved_job(tmp_path, monkeypatch, exitcode=1, attempt=1)
    pending = []  # one fresh queue list shared by both modules, as init_queue_refs binds it in production
    monkeypatch.setattr(W, "PENDING", pending)
    monkeypatch.setattr(q, "PENDING", pending)
    task_id, meta = job["task_id"], job["meta"]
    for row in (job["task"], meta["task"]):
        if bounded:
            row["execution_timeout_sec"] = 900
    write_task_result(tmp_path, task_id, "running")
    if saved_work:
        _saved(tmp_path, task_id, 1)  # the cognition checkpoint's original start is now - 100
        meta["started_at"] = clock[0] - 100
    else:
        meta["started_at"] = clock[0] - 300
    # A confirmed quota wait the supervisor saw AFTER the checkpoint; still open at its last observation.
    meta["model_wait_quota_clock"] = {"revision": 4, "elapsed_sec": 40.0, "observed_at": clock[0] - 10, "active": True}
    monkeypatch.setattr(W, "reconstruct_task_cost", lambda *_a, **_k: {"cost_accounting_status": "available"})
    recover_confirmed_dead_worker(job)
    [retry] = [row for row in q.PENDING if row["id"] == task_id]
    ended_total = (100 if saved_work else 300) - 40
    if bounded:
        assert retry["_execution_handoff_sec"] == pytest.approx(ended_total)  # closed at the observation, not now
    else:
        assert "_execution_handoff_sec" not in retry
    assert retry.get("_execution_consumed_sec", 0) == 0
    if saved_work:
        clock_fact = retry["_working_recovery"]["model_wait_quota_clock"]
        assert clock_fact["revision"] == 4 and clock_fact["active"] is False
    clock[0] += queue_delay  # the retry waits for a worker
    sent = _assign_one(W, state, monkeypatch)
    assert [row["id"] for row in sent] == [task_id]
    running = W.RUNNING[task_id]
    assert "_execution_handoff_sec" not in running["task"]
    if saved_work:
        assert running["task"].get("_execution_consumed_sec", 0) == 0  # the original start spans attempt 1
        assert running["started_at"] == pytest.approx(retry["_working_recovery"]["started_at"])
    else:
        assert running["task"].get("_execution_consumed_sec", 0) == pytest.approx(ended_total if bounded else 0)
        assert running["started_at"] == clock[0]
    waiter = _agent_start(sent[0], tmp_path, monkeypatch)
    clock[0] += 10
    assert waiter.executed_seconds() == pytest.approx(execution_elapsed_seconds(running, clock[0]), abs=1e-6)
    expected = ended_total + 10 if bounded else (ended_total + queue_delay + 10 if saved_work else 10)
    assert execution_elapsed_seconds(running, clock[0]) == pytest.approx(expected)
    assert waiter.executed_seconds() == pytest.approx(expected)
    if saved_work and bounded:
        from ouroboros.working_checkpoint import load_recovery
        from ouroboros.owner_wait import restore_continuation_state

        ctx = waiter.tool_context
        saved = load_recovery(ctx)
        restore_continuation_state(SimpleNamespace(_ctx=ctx), saved, [], {}, {}, set())
        # A subsequent pause reads this carrier from the restored loop context.
        assert ctx._budget_paused_sec == pytest.approx(queue_delay)
        assert waiter.executed_seconds() == pytest.approx(expected)


@pytest.mark.parametrize("fence", ["own_active", "legacy", "paused_root"])
def test_crash_retry_releases_only_the_dead_attempts_own_fence(tmp_path, monkeypatch, fence):
    import supervisor.queue as q
    import supervisor.workers as W
    from supervisor.worker_health import recover_confirmed_dead_worker

    job, _events = _reserved_job(tmp_path, monkeypatch, exitcode=1, attempt=1)
    task_id = job["task_id"]
    write_task_result(tmp_path, task_id, "running")
    owner = {"task_attempt": 1, "pid": job["worker"].proc.pid}
    row = {"token": "t", "root_task_id": task_id, "task_id": task_id, "status": "active",
           "owner_message_generation": 0, "pooled_owner": owner}
    if fence == "sealed":
        row["status"] = "sealed"
    elif fence == "other_attempt":
        row["pooled_owner"] = {"task_attempt": 5, "pid": owner["pid"]}
    elif fence == "legacy":
        row.pop("pooled_owner")
    elif fence == "paused_root":
        monkeypatch.setitem(q.BUDGET_ROOT_FENCES, task_id, {"status": "active", "fence_id": "f"})
    monkeypatch.setattr(q, "ACCEPTANCE_FENCES", {task_id: row})
    monkeypatch.setattr(W, "reconstruct_task_cost", lambda *_a, **_k: {"cost_accounting_status": "available"})
    recover_confirmed_dead_worker(job)
    retried = [r for r in q.PENDING if r["id"] == task_id]
    if fence == "own_active":
        assert q.ACCEPTANCE_FENCES == {}
        assert retried and not retried[0].get("_admission_blocked"), "the dead attempt's fence never blocks its retry"
    else:
        assert q.ACCEPTANCE_FENCES.get(task_id) == row  # live, sealed, foreign, legacy and paused controls stay


def test_a_boot_handoff_reads_raw_clock_facts_not_the_readers_derived_duration(tmp_path):
    """The SQL-era snapshot records raw clock facts; the boot reader derives ``execution_sec``
    at read time, extending a still-open quota wait across the downtime. The handoff closes
    that wait at its own last observation instead, so the downtime is never a regrant."""

    from ouroboros.model_wait import stamp_execution_handoff
    from ouroboros.queue_observation import load_queue_document

    now = time.time()
    (tmp_path / "state").mkdir(exist_ok=True)
    (tmp_path / "state" / "queue_snapshot.json").write_text(json.dumps({"running": [{
        "id": "bounded", "started_at": now - 1000, "task": {"id": "bounded", "execution_timeout_sec": 3600},
        "model_wait_quota_clock": {"active": True, "elapsed_sec": 100.0, "observed_at": now - 800}}]}))
    [row] = load_queue_document(tmp_path)["running"]
    assert row["execution_sec"] == pytest.approx(100, abs=2)  # the open wait read as lasting until boot
    successor = {"id": "bounded", "execution_timeout_sec": 3600}
    stamp_execution_handoff(successor, row, now=now)
    assert successor["_execution_handoff_sec"] == pytest.approx(900)  # closed at its observation, not regranted


@pytest.mark.parametrize("change", ["none", "damaged_source", "foreign_selected_grant"])
def test_saved_crash_retry_expires_in_queue_only_with_current_source(tmp_path, monkeypatch, change):
    import queue as stdqueue
    from supervisor import queue as q, workers as W, queue_timeouts
    from supervisor.worker_health import recover_confirmed_dead_worker
    from ouroboros.task_results import load_task_result

    job, _events = _reserved_job(tmp_path, monkeypatch, exitcode=1, attempt=1)
    task_id = job["task_id"]
    pending = []
    monkeypatch.setattr(W, "PENDING", pending)
    monkeypatch.setattr(q, "PENDING", pending)
    write_task_result(tmp_path, task_id, "running", task_attempt=1)
    _saved(tmp_path, task_id, 1)
    monkeypatch.setattr(W, "reconstruct_task_cost", lambda *_a, **_k: {"cost_accounting_status": "available"})
    monkeypatch.setattr(W, "respawn_worker", lambda wid, **_k: W.WORKERS.pop(wid, None))
    recover_confirmed_dead_worker(job)
    [task] = q.PENDING
    assert task["_working_recovery"]["from_attempt"] == 1
    task["deadline_at"] = "2000-01-01T00:00:00Z"
    jobs = stdqueue.Queue()
    monkeypatch.setattr(q, "_reap_queue", jobs)
    monkeypatch.setattr(queue_timeouts, "_pending_deadline_jobs", {})
    q.enforce_task_timeouts()
    assert task["_terminalization_retry"]["trigger"] == "deadline"
    if change == "damaged_source":
        task["_working_recovery"]["source_ref"]["sha256"] = "0" * 64
    elif change == "already_dispatched":
        write_task_result(tmp_path, task_id, "running", admitted_dispatch_attempt=2)
    elif change == "foreign_selected_grant":
        # The still-valid working source cannot bless an invalid selected Resume.
        task["_budget_pause_resume"] = {"grant_id": "foreign"}
    queue_timeouts._reap_pending_deadline(jobs.get_nowait())
    if change == "none":
        assert not q.PENDING and not q.RUNNING
        assert load_task_result(tmp_path, task_id)["reason_code"] == "deadline"
    else:
        assert q.PENDING == [task]
        assert load_task_result(tmp_path, task_id).get("reason_code") != "deadline"


@pytest.mark.parametrize(("fork_mirror", "change"), [
    pytest.param(False, "attempt", id='canonical-attempt'),
    pytest.param(False, "source", id='canonical-source'),
    pytest.param(False, "owner", id='canonical-owner'),
    pytest.param(False, "pause", id='canonical-pause'),
    pytest.param(False, "live", id='canonical-live'),
    pytest.param(True, "none", id='fork-mirror-none'),
    pytest.param(True, "snapshot_retry", id='fork-mirror-snapshot_retry'),
    pytest.param(True, "selected_source", id='fork-mirror-selected_source'),
    pytest.param(True, "claim_attempt", id='fork-mirror-claim_attempt'),
    pytest.param(True, "stop", id='fork-mirror-stop'),
    pytest.param(True, "completed", id='fork-mirror-completed'),
])
def test_deadline_crossing_after_recovery_claim_never_puts_the_worker_queue(tmp_path, monkeypatch, change, fork_mirror):
    import datetime
    from ouroboros.task_results import load_task_result
    from supervisor import queue as q, workers as W, state, worker_assignment, queue_timeouts
    from supervisor.worker_health import recover_confirmed_dead_worker

    clock = _fake_clock(monkeypatch)
    job, _ = _reserved_job(tmp_path, monkeypatch, exitcode=1, attempt=1)
    pending = []
    monkeypatch.setattr(W, "PENDING", pending)
    monkeypatch.setattr(q, "PENDING", pending)
    task_id = job["task_id"]
    write_task_result(tmp_path, task_id, "running", task_attempt=1)
    _saved(tmp_path, task_id, 1)
    monkeypatch.setattr(W, "respawn_worker", lambda wid, **_k: W.WORKERS.pop(wid, None))
    recover_confirmed_dead_worker(job)
    [task] = q.PENDING
    if fork_mirror:
        task["drive_root"] = str(tmp_path / "child-drive")
    monkeypatch.setattr(q, "reconstruct_task_cost", lambda *_a, **_k: {
        "cost_accounting_status": "unavailable", "cost_final": False})
    task["deadline_at"] = datetime.datetime.fromtimestamp(clock[0] + 30, datetime.timezone.utc).isoformat()
    mirror = worker_assignment._mirror_assigned_running_status

    def mirror_then_expire(row):
        mirror(row)
        clock[0] += 100

    monkeypatch.setattr(worker_assignment, "_mirror_assigned_running_status", mirror_then_expire)
    monkeypatch.setattr(queue_timeouts, "_pending_deadline_jobs", {})
    assert not _assign_one(W, state, monkeypatch)
    assert q.PENDING == [task] and not q.RUNNING
    assert W.WORKERS[7].busy_task_id is None
    claim = load_task_result(tmp_path, task_id)
    assert claim["admitted_dispatch"] == "possible" and claim["admitted_dispatch_attempt"] == 2
    # The caller owns a positive no-put fact; possible remains monotonic.
    assert task.get("_terminalization_retry", {}).get("trigger") == "deadline"
    if change != "none":
        from supervisor import queue_snapshot

        assert queue_snapshot.persist_queue_snapshot(reason="test_process_retry")
        q.PENDING.clear()
        assert queue_snapshot.restore_pending_from_snapshot() == 1
        [task] = q.PENDING
        # Lose process-local reservations; retry through the existing consumer.
        monkeypatch.setattr(queue_timeouts, "_pending_deadline_jobs", {})
        assert queue_timeouts.pending_deadline_expirable(task)
    if change == "attempt":
        task["_attempt"] += 1
    elif change == "source":
        task["_working_recovery"]["source_ref"]["sha256"] = "0" * 64
    elif change == "selected_source":
        task["_budget_pause_resume"] = {"grant_id": "foreign"}
    elif change == "owner":
        write_task_result(tmp_path, task_id, claim["status"], execution_owner={
            "kind": "direct", "task_id": task_id, "task_attempt": 2, "data_root": str(tmp_path)})
    elif change == "claim_attempt":
        write_task_result(tmp_path, task_id, claim["status"], admitted_dispatch_attempt=3)
    elif change == "pause":
        from supervisor.owner_pause_control import request_owner_pause

        assert request_owner_pause(task_id, request_id="late-pause")["ok"]
    elif change == "stop":
        from ouroboros.cancel_intents import request_cancel

        request_cancel(tmp_path, task_id, source="test", reason="Stop before reaper")
    elif change == "live":
        W.RUNNING[task_id] = {"task": task}
    elif change == "completed":
        write_task_result(tmp_path, task_id, "completed", result="Saved answer")
    W._settle_terminalization_task(task, reason="deadline", status="failed", trigger="deadline")
    while not q._reap_queue.empty():
        queue_timeouts._reap_pending_deadline(q._reap_queue.get_nowait())
    result = load_task_result(tmp_path, task_id)
    if change in {"none", "snapshot_retry", "completed"}:
        assert not q.PENDING and not q.RUNNING
        if change == "completed":
            assert result["status"] == "completed" and result["result"] == "Saved answer"
        else:
            assert result["status"] == "failed" and result["reason_code"] == "deadline"
            assert result["cost_accounting_status"] == "unavailable" and result["cost_final"] is False
    else:
        assert q.PENDING == [task] and result.get("reason_code") != "deadline"
    assert result["admitted_dispatch"] == "possible"
    assert result["admitted_dispatch_attempt"] == (3 if change == "claim_attempt" else 2)
