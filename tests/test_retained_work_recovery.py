"""Existing automatic retries continue saved work (owner 2026-10-08, quiz 1b1a2d93, option 1).

Eligibility, count and cadence are unchanged: the same-ID worker-crash retry and
the idle-timeout retry (same or new execution id) only change their STARTING
POINT, from the original prompt to the dead attempt's frozen working state.
Cases without an automatic retry stay manual; the two crash causes that retain a
source become Continue-eligible; a source-less pausing seed is completed.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ouroboros import working_checkpoint as wc
from ouroboros.artifacts import read_actor_source_bytes
from ouroboros.task_results import load_task_result, write_task_result
from tests._budget_pause_exact_helpers import _loop_ctx
from tests.test_owner_wait_pool import pool  # noqa: F401 - the real pool fixture
from tests.test_worker_crash_retry import _isolate_worker_crash_state, _reserved_job  # noqa: F401

pytestmark = pytest.mark.serial


def _saved(root, task_id, attempt, boundary="post_batch"):
    _ctx, limit_ctx = _loop_ctx(root, task_id, attempt=attempt)
    wc.save_round(limit_ctx, boundary)
    return wc.checkpoint_path(root, task_id, attempt).read_bytes()


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
    assert wc.checkpoint_path(tmp_path, task_id, 1).read_bytes() == exact


def test_a_signal_crash_stays_terminal_saved_work_never_widens_the_retry(tmp_path, monkeypatch):
    import supervisor.queue as q
    import supervisor.workers as W
    from supervisor.worker_health import recover_confirmed_dead_worker

    job, _events = _reserved_job(tmp_path, monkeypatch, exitcode=-11, attempt=1)
    write_task_result(tmp_path, job["task_id"], "running")
    _saved(tmp_path, job["task_id"], 1)
    monkeypatch.setattr(W, "reconstruct_task_cost", lambda *_a, **_k: {"cost_accounting_status": "available"})
    monkeypatch.setattr(W, "send_with_budget", lambda *_a, **_k: None)
    monkeypatch.setattr(q, "enqueue_task", MagicMock())
    recover_confirmed_dead_worker(job)
    q.enqueue_task.assert_not_called()
    assert load_task_result(tmp_path, job["task_id"])["reason_code"] == "worker_crash_signal"


@pytest.mark.parametrize("reason", ["worker_crash_owner_wait", "worker_crash_budget_pausing"])
def test_crashes_that_keep_their_source_are_continue_eligible(reason):
    from ouroboros.owner_continue import continuation_eligibility

    verdict = continuation_eligibility({"status": "failed", "reason_code": reason, "task_id": "r",
                                        "root_task_id": "r"}, "r")
    assert verdict == {"eligible": True, "cause": reason, "refusal": ""}


def test_the_idle_retry_carries_the_killed_attempts_source_to_its_new_execution_id(pool, monkeypatch):  # noqa: F811
    from supervisor import queue, task_reaper, workers

    write_task_result(pool.root, "owner", "running")
    exact = _saved(pool.root, "owner", 3)
    task = {**pool.meta["task"], "_attempt": 3}
    requeued, new_attempt, _reason, _ = task_reaper._enqueue_retry(
        queue, task, task_id="owner", retry_task_id="owner-retry", attempt=3,
        terminal_reason="idle_timeout", recon_fields={})
    assert requeued and new_attempt == 4
    retry = workers.PENDING[-1]
    assert retry["id"] == "owner-retry" and retry["original_task_id"] == "owner"
    handoff = retry["_working_recovery"]
    assert handoff["source_task_id"] == "owner" and handoff["from_attempt"] == 3
    assert handoff["cause"] == "idle_timeout"
    assert read_actor_source_bytes(pool.root, "owner", handoff["source_ref"]) == exact


def test_retry_attachment_copy_failure_keeps_original_inputs_and_working_source(pool, monkeypatch):  # noqa: F811
    from pathlib import Path

    from ouroboros import artifacts
    from supervisor import queue, task_reaper, workers

    write_task_result(pool.root, "owner", "running")
    exact = _saved(pool.root, "owner", 3)
    source = pool.root / "source-input.txt"
    source.write_text("original attachment bytes")
    manifest = artifacts.stage_task_attachments(pool.root, "owner", [{"path": str(source)}])
    assert manifest[0]["status"] == "staged"
    staged = Path(manifest[0]["abs_path"])
    task = {**pool.meta["task"], "_attempt": 3, "task_contract": {"attachment_manifest": manifest}}

    def failed_copy(*args, **kwargs):
        raise OSError("retry attachment destination unwritable")

    monkeypatch.setattr(artifacts, "copy_artifact_file", failed_copy)
    requeued, _, reason, _ = task_reaper._enqueue_retry(
        queue, task, task_id="owner", retry_task_id="owner-retry", attempt=3,
        terminal_reason="idle_timeout", recon_fields={})

    assert not requeued and reason == "idle_timeout_retry_attachment_handoff_failed"
    assert not any(task["id"] == "owner-retry" for task in workers.PENDING)
    assert staged.read_text() == "original attachment bytes"
    assert wc.checkpoint_path(pool.root, "owner", 3).read_bytes() == exact
    handoff = wc.prepare_recovery(pool.root, "owner", from_attempt=3, cause="worker_crash")
    assert read_actor_source_bytes(pool.root, "owner", handoff["source_ref"]) == exact


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
