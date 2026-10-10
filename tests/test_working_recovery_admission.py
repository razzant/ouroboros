"""Saved work reaches the real assignment door without bypassing its scope proof."""
from __future__ import annotations

import copy
import json

import pytest

from ouroboros import working_checkpoint as wc
from ouroboros.task_results import load_task_result, write_task_result
from supervisor import queue, workers
from supervisor.events_budget import budget_hold_fact
from tests._budget_pause_exact_helpers import _loop_ctx
from tests.test_main_project_consistency import admit_main
from tests.test_project_hold_recovery import accepted, worker
from tests.test_restart_saved_work import _ack, _stop_and_boot
from tests.test_swarm_host_admission import host  # noqa: F401

pytestmark = pytest.mark.serial


def _running(host, tmp_path, monkeypatch, scope):  # noqa: F811
    task = accepted(host, tmp_path) if scope == "project" else admit_main(host, tid="held")
    original = copy.deepcopy(task)
    sent = worker(host, monkeypatch)
    workers.assign_tasks()
    assert [row["id"] for row in sent] == ["held"]
    write_task_result(host.root, "held", "running", task_attempt=1, started_at="2026-10-08T00:00:00Z")
    _, limit = _loop_ctx(host.root, "held", attempt=1)
    wc.save_round(limit, "post_batch")
    # Physical worker termination is doubled; queue handoff and its durable
    # metadata above, stop, boot, Resume and later assignment stay real.
    monkeypatch.setattr(workers, "WORKERS", {})
    return original


@pytest.mark.parametrize("scope", ["project", "main"])
@pytest.mark.parametrize("stop", ["quit", "restart"])
def test_saved_accepted_work_returns_through_real_assignment(host, tmp_path, monkeypatch, scope, stop):  # noqa: F811
    from supervisor.restart_retention import prepare_restart_returns

    original = _running(host, tmp_path, monkeypatch, scope)
    if stop == "restart":
        assert prepare_restart_returns(host.root, workers.RUNNING, workers.PENDING,
                                       transaction_id="working-return") == {"held"}
        _ack(host.root, "working-return")
    assert _stop_and_boot(queue, workers) == 1
    [restored] = host.pending
    assert restored["_attempt"] == 2
    sent = worker(host, monkeypatch)
    if stop == "quit":
        workers.assign_tasks()
        assert not sent and budget_hold_fact(restored)
        assert queue.resume_budget_paused_task("held")["ok"]
    workers.assign_tasks()
    workers.assign_tasks()
    assert [row["id"] for row in sent] == ["held"], restored
    assert sent[0]["_attempt"] == 2
    for key in ("project_id", "workspace_root", "drive_root", "_project_admission", "_project_scope_none"):
        assert sent[0].get(key) == original.get(key)
    assert load_task_result(host.root, "held")["admitted_dispatch_attempt"] == 2
    assert not host.attempts


@pytest.mark.parametrize("scope", ["project", "main"])
def test_old_saved_pending_snapshot_cannot_replay_after_handoff_when_running_mirror_fails(
    host, tmp_path, monkeypatch, scope,  # noqa: F811
):
    from supervisor import worker_assignment

    _running(host, tmp_path, monkeypatch, scope)
    assert _stop_and_boot(queue, workers) == 1
    assert queue.resume_budget_paused_task("held")["ok"]
    assert queue.persist_queue_snapshot()
    old_pending = queue.QUEUE_SNAPSHOT_PATH.read_bytes()
    sent = worker(host, monkeypatch)
    monkeypatch.setattr(worker_assignment, "_mirror_assigned_running_status", lambda task: None)
    workers.assign_tasks()
    assert len(sent) == 1 and sent[0]["_attempt"] == 2
    assert load_task_result(host.root, "held")["task_attempt"] == 1
    assert load_task_result(host.root, "held")["admitted_dispatch_attempt"] == 2

    workers.RUNNING.clear()
    workers.WORKERS[0].busy_task_id = None
    queue.QUEUE_SNAPSHOT_PATH.write_bytes(old_pending)
    assert queue.restore_pending_from_snapshot() == 1
    assert queue.resume_budget_paused_task("held")["ok"] is False
    workers.assign_tasks()
    assert len(sent) == 1
    assert budget_hold_fact(host.pending[0])
    assert not host.attempts


@pytest.mark.parametrize("marker", [None, True, "1", 0, -1, 4, 5])
def test_recovery_refuses_malformed_or_already_dispatched_attempt_marker(tmp_path, marker):
    from supervisor.task_admission import _working_resume_granted

    write_task_result(tmp_path, "task", "running", task_attempt=3, admitted_dispatch_attempt=marker)
    _, limit = _loop_ctx(tmp_path, "task", attempt=3)
    wc.save_round(limit, "ready")
    task = {"id": "task", "_attempt": 4}
    assert wc.attach_recovery(tmp_path, task, source_task_id="task", from_attempt=3, cause="app_stop")
    assert not _working_resume_granted(task, tmp_path)


@pytest.mark.parametrize("protocol,marker,allowed", [(True, None, True), (False, None, False),
                                                     (False, 3, True), (True, 3, True)])
def test_original_later_attempt_needs_exact_source_and_dispatch_protocol(tmp_path, protocol, marker, allowed):
    from supervisor.task_admission import _working_resume_granted

    write_task_result(tmp_path, "task", "running", task_attempt=3,
                      **({"admitted_dispatch_attempt": marker} if marker is not None else {}))
    _, limit = _loop_ctx(tmp_path, "task", attempt=3)
    wc.save_round(limit, "ready")
    path = wc.checkpoint_path(tmp_path, "task", 3)
    if not protocol:
        source = json.loads(path.read_bytes())
        source.pop("retained_work_dispatch_protocol")
        path.write_text(json.dumps(source))
    task = {"id": "task", "_attempt": 4}
    assert wc.attach_recovery(tmp_path, task, source_task_id="task", from_attempt=3, cause="app_stop")
    assert _working_resume_granted(task, tmp_path) is allowed
    task["_attempt"] = 5
    assert not _working_resume_granted(task, tmp_path), "a source authorizes only its exact successor"


@pytest.mark.parametrize("damage", ["source_digest", "wrong_attempt", "empty_locator", "malformed_ref"])
def test_real_resume_keeps_saved_work_held_when_its_source_is_not_exact(
    host, tmp_path, monkeypatch, damage,  # noqa: F811
):
    _running(host, tmp_path, monkeypatch, "project")
    assert _stop_and_boot(queue, workers) == 1
    [restored] = host.pending
    original_checkpoint = wc.checkpoint_path(host.root, "held", 1).read_bytes()
    if damage == "source_digest":
        restored["_working_recovery"]["source_ref"]["sha256"] = "0" * 64
    elif damage == "wrong_attempt":
        restored["_attempt"] += 1
    elif damage == "malformed_ref":
        restored["_working_recovery"]["source_ref"] = ["not a source handle"]
    else:
        restored["_working_recovery"] = {}
    assert queue.resume_budget_paused_task("held")["ok"] is False
    assert budget_hold_fact(restored)
    assert wc.checkpoint_path(host.root, "held", 1).read_bytes() == original_checkpoint
    sent = worker(host, monkeypatch)
    workers.assign_tasks()
    assert not sent


def test_scope_less_recovered_work_stamps_before_handoff_and_cannot_stamp_twice(tmp_path, monkeypatch):
    from supervisor import task_admission

    monkeypatch.setattr(queue, "DRIVE_ROOT", tmp_path)
    write_task_result(tmp_path, "direct", "running", task_attempt=3, admitted_dispatch="possible")
    _, limit = _loop_ctx(tmp_path, "direct", attempt=3)
    wc.save_round(limit, "ready")
    task = {"id": "direct", "_attempt": 4}
    assert wc.attach_recovery(tmp_path, task, source_task_id="direct", from_attempt=3, cause="app_stop")
    assert task_admission.record_project_dispatch_possible(task)
    stored = load_task_result(tmp_path, "direct")
    assert stored["admitted_dispatch_attempt"] == 4 and stored["task_attempt"] == 3
    assert not task_admission.record_project_dispatch_possible(task)
    assert not task_admission._working_resume_granted(task, tmp_path)


@pytest.mark.parametrize("scope", ["project", "main"])
def test_real_new_id_idle_retry_preserves_its_attempt_and_reaches_assignment(
    host, tmp_path, monkeypatch, scope,  # noqa: F811
):
    from supervisor import task_reaper

    original = _running(host, tmp_path, monkeypatch, scope)
    original["_attempt"] = 3
    write_task_result(host.root, "held", "running", task_attempt=3, admitted_dispatch_attempt=3)
    _, limit = _loop_ctx(host.root, "held", attempt=3)
    wc.save_round(limit, "post_batch")
    workers.RUNNING.clear()
    answer = task_reaper._enqueue_retry(queue, original, task_id="held", retry_task_id="held-retry",
                                       attempt=3, terminal_reason="idle_timeout", recon_fields={})
    assert answer[0] is True and answer[1] == 4, answer
    [retry] = host.pending
    assert retry["id"] == "held-retry" and retry["_attempt"] == 4
    stored = load_task_result(host.root, "held-retry")
    assert stored["status"] == "scheduled" and stored["timeout_retry_from"] == "held"
    assert "task_attempt" not in stored
    sent = worker(host, monkeypatch)
    workers.assign_tasks()
    workers.assign_tasks()
    assert [task["id"] for task in sent] == ["held-retry"], retry
    assert sent[0]["_attempt"] == 4
    assert load_task_result(host.root, "held-retry")["admitted_dispatch_attempt"] == 4
    assert not host.attempts
