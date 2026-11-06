"""Existing automatic retries continue saved work (owner 2026-10-08, quiz 1b1a2d93, option 1).

Eligibility, count and cadence are unchanged: the same-ID worker-crash retry only
changes its STARTING POINT, from the original prompt to the dead attempt's frozen
working state. The idle-timeout retry the decision also covered is gone (#1376: silence
no longer kills, and a hard bound never retries); a stored new-ID successor still reads
its predecessor's source (``tests/test_working_retry_sources.py``).
Cases without an automatic retry stay manual; the two crash causes that retain a
source become Continue-eligible; a source-less pausing seed is completed.
"""
from __future__ import annotations

import pytest

from ouroboros import working_checkpoint as wc
from ouroboros.artifacts import read_actor_source_bytes
from ouroboros.task_results import load_task_result, write_task_result
from tests._budget_pause_exact_helpers import _loop_ctx
from tests.test_worker_crash_retry import _isolate_worker_crash_state, _reserved_job  # noqa: F401



def _saved(root, task_id, attempt, boundary="post_batch"):
    _ctx, limit_ctx = _loop_ctx(root, task_id, attempt=attempt)
    wc.save_round(limit_ctx, boundary)
    return wc.read_working(root, task_id, attempt)[0]  # the whole state a recovery freezes


@pytest.mark.parametrize("saved", [True, False])
def test_the_existing_crash_retry_continues_the_dead_attempts_saved_work(tmp_path, monkeypatch, saved):
    import supervisor.queue as q
    import supervisor.workers as W
    from supervisor.worker_health import recover_confirmed_dead_worker

    job, _events = _reserved_job(tmp_path, monkeypatch, exitcode=1, attempt=1)
    task_id = job["task_id"]
    write_task_result(tmp_path, task_id, "running")
    exact = _saved(tmp_path, task_id, 1) if saved else b""
    monkeypatch.setattr(W, "reconstruct_task_cost", lambda *_a, **_k: {"cost_accounting_status": "available"})
    enqueued = []
    monkeypatch.setattr(q, "enqueue_task", lambda task, front=False: enqueued.append(dict(task)) or task)
    recover_confirmed_dead_worker(job)
    assert len(enqueued) == 1 and enqueued[0]["_attempt"] == 2, "the same retry, the same count"
    handoff = enqueued[0].get("_working_recovery")
    if not saved:
        assert handoff is None, "no saved work: the original start, as before"
        return
    assert handoff["source_task_id"] == task_id and handoff["from_attempt"] == 1
    assert handoff["cause"] == "worker_crash"
    assert read_actor_source_bytes(tmp_path, task_id, handoff["source_ref"]) == exact
    # Kept until the retry's consumer loaded the frozen copy (a crash before the
    # enqueue's publication finds the same file again).
    assert wc.read_working(tmp_path, task_id, 1)[0] == exact


def test_a_signal_crash_continues_the_saved_work_once_and_the_bound_still_holds(tmp_path, monkeypatch):
    """A worker death (ARCHITECTURE §5): a signal death is no longer terminal on its own. Attempt 1 continues the dead
    attempt's saved work on the same id (the existing retry, its starting point unchanged); the retried
    attempt that dies again is terminal with the signal recorded, so saved work never widens the count."""
    import supervisor.queue as q
    import supervisor.workers as W
    from supervisor.worker_health import recover_confirmed_dead_worker

    job, _events = _reserved_job(tmp_path, monkeypatch, exitcode=-11, attempt=1)
    task_id = job["task_id"]
    write_task_result(tmp_path, task_id, "running")
    exact = _saved(tmp_path, task_id, 1)
    monkeypatch.setattr(W, "reconstruct_task_cost", lambda *_a, **_k: {"cost_accounting_status": "available"})
    monkeypatch.setattr(W, "send_with_budget", lambda *_a, **_k: None)
    enqueued = []
    monkeypatch.setattr(q, "enqueue_task", lambda task, front=False: enqueued.append(dict(task)) or task)
    recover_confirmed_dead_worker(job)
    assert len(enqueued) == 1 and enqueued[0]["_attempt"] == 2
    handoff = enqueued[0]["_working_recovery"]
    assert handoff["source_task_id"] == task_id and handoff["from_attempt"] == 1
    assert read_actor_source_bytes(tmp_path, task_id, handoff["source_ref"]) == exact
    assert load_task_result(tmp_path, task_id)["status"] == "interrupted"

    W.WORKERS.clear(); W.RUNNING.clear(); q.RUNNING = W.RUNNING
    job, _events = _reserved_job(tmp_path, monkeypatch, exitcode=-11, attempt=2)
    write_task_result(tmp_path, job["task_id"], "running")
    _saved(tmp_path, job["task_id"], 2)
    recover_confirmed_dead_worker(job)
    row = load_task_result(tmp_path, job["task_id"])
    assert len(enqueued) == 1 and row["reason_code"] == "worker_crash_retry_exhausted" and row["crash_signal"] == 11


@pytest.mark.parametrize("reason", ["worker_crash_owner_wait", "worker_crash_budget_pausing"])
def test_crashes_that_keep_their_source_are_continue_eligible(reason):
    from ouroboros.owner_continue import continuation_eligibility

    verdict = continuation_eligibility({"status": "failed", "reason_code": reason, "task_id": "r",
                                        "root_task_id": "r"}, "r")
    assert verdict == {"eligible": True, "cause": reason, "refusal": ""}


def test_actual_worker_crash_retry_relays_saved_work_when_receiver_died_before_save(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from supervisor import queue, task_admission, workers
    from supervisor.worker_health import recover_confirmed_dead_worker

    job, _events = _reserved_job(tmp_path, monkeypatch, exitcode=1, attempt=2)
    task_id = job["task_id"]
    write_task_result(tmp_path, task_id, "running", task_attempt=1)
    exact = _saved(tmp_path, task_id, 1)
    receiver = {"id": task_id, "_attempt": 2}
    assert wc.attach_recovery(tmp_path, receiver, source_task_id=task_id, from_attempt=1, cause="worker_crash")
    assert task_admission.record_project_dispatch_possible(receiver)
    write_task_result(tmp_path, task_id, "running", task_attempt=2)
    job["task"]["_working_recovery"] = receiver["_working_recovery"]
    job["meta"]["task"]["_working_recovery"] = receiver["_working_recovery"]
    wc.consume_recovery(SimpleNamespace(task_id=task_id, drive_root=tmp_path), receiver["_working_recovery"])
    monkeypatch.setattr(workers, "QUEUE_MAX_RETRIES", 2)  # existing configurable eligibility, unchanged
    monkeypatch.setattr(workers, "reconstruct_task_cost", lambda *_a, **_k: {"cost_accounting_status": "available"})
    enqueued = []
    monkeypatch.setattr(queue, "enqueue_task", lambda task, front=False: enqueued.append(dict(task)) or task)
    recover_confirmed_dead_worker(job)
    [retry] = enqueued
    assert retry["_attempt"] == 3 and retry["_working_recovery"]["target_attempt"] == 3
    assert retry["_working_recovery"]["from_attempt"] == 1
    assert read_actor_source_bytes(tmp_path, task_id, retry["_working_recovery"]["source_ref"]) == exact
    assert wc.recovery_source_for_task(tmp_path, retry)["messages"]
