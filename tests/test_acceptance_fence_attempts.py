"""#1183: an ended attempt's acceptance fence never outlives it, through the real consumers.

A pooled begin is bound to its RUNNING attempt and pid. The handoff that creates a
proven-ended attempt's successor (crash retry, aborted-update return,
a boot-restored saved successor) drops exactly that attempt's active fence, before
admission or restore reads it. Live, sealed, foreign, direct/legacy and paused
fences stay; independent paid review evidence is untouched; no TTL, no purge.
"""
from __future__ import annotations

import copy
import json
import time
from types import SimpleNamespace

import pytest

from tests.test_worker_crash_retry import (
    _isolate_worker_crash_state,  # noqa: F401
    _reserved_job as _reserved_crash_job,
)

pytestmark = pytest.mark.serial


@pytest.fixture(autouse=True)
def isolated_fences(monkeypatch):
    from supervisor import queue, task_lifecycle, workers

    pending, budget_fences = [], {}
    monkeypatch.setattr(queue, "ACCEPTANCE_FENCES", {})
    monkeypatch.setattr(queue, "BUDGET_ROOT_FENCES", budget_fences)
    monkeypatch.setattr(task_lifecycle, "BUDGET_ROOT_FENCES", budget_fences)
    monkeypatch.setattr(queue, "PENDING", pending)
    monkeypatch.setattr(workers, "PENDING", pending)


def _reserved_job(tmp_path, monkeypatch):
    job, events = _reserved_crash_job(tmp_path, monkeypatch)
    job["task"]["root_task_id"] = job["task_id"]
    job["meta"]["task"]["root_task_id"] = job["task_id"]
    return job, events


def _open_reserved_fence(job):
    """The same worker was alive when it began acceptance, before health captured its death."""
    from supervisor import queue

    worker = job["worker"]
    worker.reaping = False
    worker.proc.is_alive.return_value = True
    owner = {"task_attempt": job["attempt"], "pid": worker.proc.pid}
    ack = queue.transition_acceptance_fence(action="begin", token="a" * 32, task_id=job["task_id"],
                                            root_task_id=job["task_id"], pooled_owner=owner)
    assert ack["status"] == "active"
    worker.reaping = True
    worker.proc.is_alive.return_value = False
    return copy.deepcopy(queue.ACCEPTANCE_FENCES[job["task_id"]])


def test_crash_recovery_releases_the_exact_fence_before_its_retry_and_refuses_a_late_begin(tmp_path, monkeypatch):
    from ouroboros.task_results import load_task_result, write_task_result
    from supervisor import queue, worker_health
    from supervisor.events_worker_reports import _handle_acceptance_fence

    job, events = _reserved_job(tmp_path, monkeypatch)
    row = _open_reserved_fence(job)
    paid_review = [{"review_id": "paid-operation", "status": "pending", "cost_usd": 0.25}]
    write_task_result(tmp_path, job["task_id"], "running", review_runs=paid_review)
    monkeypatch.setattr("ouroboros.headless.prepare_terminal_task_files",
                        lambda *_a, **_k: {"terminal_source_present": False})
    worker_health.recover_confirmed_dead_worker(job)

    assert not queue.ACCEPTANCE_FENCES
    assert [(task["id"], task["_attempt"]) for task in queue.PENDING] == [(job["task_id"], 2)]
    assert load_task_result(tmp_path, job["task_id"])["review_runs"] == paid_review  # review custody untouched
    assert not any(event.get("reason_code") == "worker_crash_retry_admission_blocked" for event in events.queue)

    # The dead attempt's begin, delivered late, owns nothing now.
    late = {"action": "begin", "token": row["token"], "req": "b" * 32, "task_id": job["task_id"],
            "root_task_id": job["task_id"], "pooled_owner": row["pooled_owner"]}
    _handle_acceptance_fence(late, SimpleNamespace(DRIVE_ROOT=tmp_path))
    ack = json.loads((tmp_path / "state" / "acceptance_fence_acks" / f"{late['token']}.{late['req']}.json").read_text())
    assert ack["status"] == "stale_attempt" and not ack["ok"]
    assert not queue.ACCEPTANCE_FENCES


@pytest.mark.parametrize("preserved", ["live", "replacement", "sealed", "foreign_task", "foreign_attempt",
                                       "foreign_pid", "missing_pid", "direct", "paused"])
def test_crash_recovery_preserves_other_acceptance_owners(tmp_path, monkeypatch, preserved):
    from supervisor import queue, workers, worker_health

    job, _events = _reserved_job(tmp_path, monkeypatch)
    _open_reserved_fence(job)
    row = queue.ACCEPTANCE_FENCES[job["task_id"]]
    if preserved == "live":
        job["worker"].proc.is_alive.return_value = True
    elif preserved == "replacement":
        workers.WORKERS[job["worker_id"]] = object()
    elif preserved == "sealed":
        row["status"] = "sealed"
    elif preserved == "foreign_task":
        row["task_id"] = "other-author"
    elif preserved == "foreign_attempt":
        row["pooled_owner"]["task_attempt"] += 1
    elif preserved == "foreign_pid":
        row["pooled_owner"]["pid"] += 1
    elif preserved == "missing_pid":
        del job["worker"].proc.pid
    elif preserved == "direct":
        row.pop("pooled_owner")
    elif preserved == "paused":
        queue.BUDGET_ROOT_FENCES[job["task_id"]] = {"root_task_id": job["task_id"], "status": "paused",
                                                    "cause": "owner_pause"}
    before = copy.deepcopy(row)
    monkeypatch.setattr("ouroboros.headless.prepare_terminal_task_files",
                        lambda *_a, **_k: {"terminal_source_present": False})
    worker_health.recover_confirmed_dead_worker(job)
    assert queue.ACCEPTANCE_FENCES[job["task_id"]] == before


def test_saved_pause_parking_keeps_its_acceptance_fence(tmp_path, monkeypatch):
    from supervisor import queue, worker_health

    job, _events = _reserved_job(tmp_path, monkeypatch)
    row = _open_reserved_fence(job)
    monkeypatch.setattr("ouroboros.headless.prepare_terminal_task_files",
                        lambda *_a, **_k: {"terminal_source_present": False})
    monkeypatch.setattr(worker_health, "_complete_exact_budget_pause_after_death", lambda *_a: (True, True))
    worker_health.recover_confirmed_dead_worker(job)
    assert queue.ACCEPTANCE_FENCES[job["task_id"]] == row


def test_live_silence_keeps_the_exact_active_fence_and_queues_no_reap(tmp_path, monkeypatch):
    """Quiet live work is attention, not an ended attempt: no successor exists to release the fence for.
    The crash tests above own the release of an ended attempt's exact fence."""
    from ouroboros.task_results import write_task_result
    from supervisor import queue, task_reaper, workers

    job, _events = _reserved_job(tmp_path, monkeypatch)
    monkeypatch.setattr("supervisor.worker_pool_lifecycle.kill_worker_tree", lambda *_a, **_kw: None)
    _open_reserved_fence(job)
    task_id = job["task_id"]
    before = copy.deepcopy(queue.ACCEPTANCE_FENCES[task_id])
    job["worker"].reaping = False
    job["meta"].update(started_at=time.time() - 1000, last_heartbeat_at=time.time() - 1000)
    write_task_result(tmp_path, task_id, "running")
    monkeypatch.setattr(queue, "get_task_idle_timeout_sec", lambda: 1)
    monkeypatch.setattr(queue, "get_per_call_timeout_ceiling_sec", lambda: 1)
    monkeypatch.setattr(queue, "FINALIZATION_GRACE_SEC", 0)
    monkeypatch.setattr(queue, "load_state", lambda: {})
    monkeypatch.setattr(task_reaper, "send_with_budget", lambda *_a, **_kw: None)
    monkeypatch.setattr(workers, "send_with_budget", lambda *_a, **_kw: None)
    monkeypatch.setattr("supervisor.cancel_publication._audit_delegated_runs_on_kill",
                        lambda *_a, **_kw: {"unreconciled": []})
    persisted = []
    monkeypatch.setattr(queue, "persist_queue_snapshot",
                        lambda **kw: persisted.append((kw.get("reason"), copy.deepcopy(queue.ACCEPTANCE_FENCES))))
    queue.enforce_task_timeouts()
    assert queue._reap_queue.empty()  # #1376: live silence preserves the fence and custody
    assert queue.ACCEPTANCE_FENCES[task_id] == before
    assert task_id in queue.RUNNING and not queue.PENDING and not job["worker"].reaping
    assert not any(reason == "acceptance_fence_attempt_ended" for reason, _fences in persisted)
