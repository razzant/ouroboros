"""Saved work reaches the real assignment door without bypassing its scope proof;
a later continuation of that work is admitted by its own carrier."""
from __future__ import annotations

import copy
import json
import time
from types import SimpleNamespace

import pytest

from ouroboros import working_checkpoint as wc
from ouroboros.task_results import load_task_result, write_task_result
from supervisor import queue, workers
from supervisor.events_budget import budget_hold_fact
from tests._budget_pause_exact_helpers import _fast_hold, _loop_ctx, _mock_pause_observation
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
def test_a_stored_new_id_timeout_retry_keeps_its_attempt_and_reaches_assignment(
    host, tmp_path, monkeypatch, scope,  # noqa: F811
):
    """A pre-upgrade new-ID successor (no producer mints one now) restored with its saved work."""
    from tests._silence_reaper_support import historical_successor

    original = _running(host, tmp_path, monkeypatch, scope)
    original["_attempt"] = 3
    write_task_result(host.root, "held", "running", task_attempt=3, admitted_dispatch_attempt=3)
    _, limit = _loop_ctx(host.root, "held", attempt=3)
    wc.save_round(limit, "post_batch")
    workers.RUNNING.clear()
    historical_successor(original, 3, "held-retry")
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


# --- a later continuation of the recovered attempt governs its final dispatch ----------------

def _recovered_attempt_running(host, tmp_path, monkeypatch, scope):  # noqa: F811
    """Real return, Resume and first handoff; the attempt's own start consumes its source."""
    from ouroboros import loop
    from ouroboros.agent import OuroborosAgent

    _running(host, tmp_path, monkeypatch, scope)
    assert _stop_and_boot(queue, workers) == 1
    assert queue.resume_budget_paused_task("held")["ok"]
    sent = worker(host, monkeypatch)
    workers.assign_tasks()
    [first] = sent
    assert first["_attempt"] == 2 and first["_working_recovery"]["from_attempt"] == 1
    OuroborosAgent._persist_running_record(SimpleNamespace(
        env=SimpleNamespace(drive_root=host.root), _task_started_ts=time.time()), first)
    ctx, _limit = _loop_ctx(host.root, "held", attempt=2)
    ctx.working_recovery = first["_working_recovery"]
    assert loop._load_working_recovery(ctx, [])["task_attempt"] == 1
    stored = load_task_result(host.root, "held")
    assert stored["admitted_dispatch_attempt"] == stored["task_attempt"] == 2
    return sent


def _owner_pause_at(host, monkeypatch, round_idx):  # noqa: F811
    """The owner's Pause reaches the running attempt; the supervisor parks it exactly."""
    from ouroboros import budget_pause
    from supervisor.events_budget import install_exact_budget_pause
    from supervisor.owner_pause_control import request_owner_pause

    assert request_owner_pause("held", request_id=f"pause-{round_idx}")["ok"]
    _ctx, limit = _loop_ctx(host.root, "held", attempt=2)
    limit.round_idx = round_idx
    limit.messages = [{"role": "user", "content": "do it"},
                      {"role": "assistant", "content": f"state saved at round {round_idx}"}]
    _fast_hold(monkeypatch, budget_pause)
    _mock_pause_observation(monkeypatch, budget_pause, [])
    with pytest.raises(budget_pause.BudgetPauseRequested) as raised:
        budget_pause.enter_owner_pause(limit)
    budget_pause.end_dispatch_fence("held")
    install_exact_budget_pause(SimpleNamespace(
        DRIVE_ROOT=host.root, RUNNING=workers.RUNNING, PENDING=workers.PENDING, WORKERS=workers.WORKERS,
        sort_pending=queue.sort_pending, persist_queue_snapshot=queue.persist_queue_snapshot, bridge=None),
        "held", budget_pause.exact_pause_marker(raised.value.pause)["checkpoint"])
    [paused] = host.pending
    assert paused["_budget_pause"]["reason"] == "owner" and paused["_attempt"] == 2
    return paused


def _granted(host):  # noqa: F811
    granted = queue.resume_budget_paused_task("held")
    assert granted["ok"] and granted["exact_continuation"], granted
    [row] = host.pending
    assert row["_budget_pause_resume"]["grant_id"] == granted["grant_id"]
    assert row["_working_recovery"]["from_attempt"] == 1, "the earlier locator legitimately remains"
    return row


def _real_loop_resumes(host, monkeypatch, task, round_idx):  # noqa: F811
    """The actual loop selects the exact pause; the historical working source is never read."""
    from ouroboros import agent, budget_pause, loop
    from tests.test_loop_transport_wait import _loop_kwargs
    from tests.test_working_checkpoint import _registry

    registry = _registry(host.root, monkeypatch, "held", 2)
    registry._ctx.model_wait_context = None
    agent._restore_saved_clocks(registry._ctx, task)
    seen = []
    monkeypatch.setattr(loop, "_load_working_recovery", lambda *a, **k: pytest.fail("working source read"))
    monkeypatch.setattr(loop, "_dispatch_round_model", lambda call, *a, **k: (
        seen.extend(call.messages) or ({"role": "assistant", "content": "continued", "tool_calls": []}, 0.0)))
    kwargs = _loop_kwargs(host.root, registry, [], llm=SimpleNamespace(default_model=lambda: "m"))  # the pause checkpoint ran on Main
    kwargs["task_id"] = "held"
    result, _usage, _trace = loop.run_llm_loop(**kwargs)
    assert result == "continued"
    assert f"state saved at round {round_idx}" in [str(row.get("content")) for row in seen]
    grant = budget_pause.budget_pause_row(host.root, "held")["grant"]
    assert grant["grant_id"] == task["_budget_pause_resume"]["grant_id"] and grant["consumed_at"]


@pytest.mark.parametrize("scope", ["project", "main"])
def test_exact_resume_of_a_recovered_attempt_reaches_the_worker_and_the_loop(
    host, tmp_path, monkeypatch, scope,  # noqa: F811
):
    from supervisor import task_admission

    sent = _recovered_attempt_running(host, tmp_path, monkeypatch, scope)
    first = copy.deepcopy(sent[0])
    for cycle, round_idx in enumerate((29, 31), start=2):
        _owner_pause_at(host, monkeypatch, round_idx)
        row = _granted(host)
        workers.assign_tasks()
        workers.assign_tasks()
        assert [task["id"] for task in sent] == ["held"] * cycle and not host.pending
        assert sent[-1]["_budget_pause_resume"] == row["_budget_pause_resume"]
        stored = load_task_result(host.root, "held")
        assert stored["admitted_dispatch_attempt"] == stored["task_attempt"] == 2
        _real_loop_resumes(host, monkeypatch, sent[-1], round_idx)
    assert not task_admission.record_project_dispatch_possible(first), "the first handoff stays one-shot"
    assert not host.attempts


def _no_working_read(monkeypatch):
    from supervisor import task_admission

    monkeypatch.setattr(task_admission, "_working_source",
                        lambda *a, **k: pytest.fail("a valid newer continuation read the saved-work source"))


@pytest.mark.parametrize("scope", ["project", "main"])
def test_owner_wait_restart_of_a_recovered_attempt_governs_its_dispatch(
    host, tmp_path, monkeypatch, scope,  # noqa: F811
):
    from ouroboros import delegate_recovery, owner_wait
    from supervisor.restart_retention import prepare_restart_returns

    _recovered_attempt_running(host, tmp_path, monkeypatch, scope)
    ctx = SimpleNamespace(task_id="held", task_attempt=2, drive_root=host.root, budget_drive_root=str(host.root),
                          task_started_at=time.time() - 60, _owner_wait_requested="quiz-a",
                          _owner_directives=[{"text": "Continue the original work"}], active_model="fixture-model",
                          active_effort="high", active_use_local=False, active_context_mode="max")
    wait = owner_wait.checkpoint_owner_wait(ctx, [{"role": "user", "content": "asked the owner"}],
                                            {"tool_calls": []}, {"cost": 1.0}, 7, [], set())
    workers.RUNNING["held"]["owner_wait"] = owner_wait.set_owner_wait(host.root, "held", {**wait, "state": "waiting"})
    selected = owner_wait.prepare_owner_wait_handoffs(host.root, workers.RUNNING, "tx-wait")
    assert delegate_recovery.prepare_planned_restart_handoffs(
        host.root, workers.RUNNING, restart_transaction_id="tx-wait", additional_task_ids=selected) == {"held"}
    prepare_restart_returns(host.root, workers.RUNNING, list(workers.PENDING), transaction_id="tx-wait",
                            owner_wait_ids=selected)
    monkeypatch.setattr(workers, "WORKERS", {})
    assert workers.kill_workers(terminal_status="cancelled", result_reason="Planned self-restart",
                                preserve_pending=True, preserve_running_task_ids=selected,
                                reconcile_delegate_custody=False, archive_service_logs=False, retain_saved_work=True)
    _ack(host.root, "tx-wait")
    host.pending.clear()
    assert queue.restore_pending_from_snapshot() == 1
    [restored] = host.pending
    assert restored["_attempt"] == 2 and restored["_owner_wait_resume"]["wait_id"] == wait["wait_id"]
    assert restored["_working_recovery"]["from_attempt"] == 1
    _no_working_read(monkeypatch)
    sent = worker(host, monkeypatch)
    workers.assign_tasks()
    workers.assign_tasks()
    assert [task["id"] for task in sent] == ["held"] and sent[0]["_attempt"] == 2
    assert owner_wait.load_owner_wait(ctx, sent[0]["_owner_wait_resume"])["round_idx"] == 7
    assert load_task_result(host.root, "held")["admitted_dispatch_attempt"] == 2


@pytest.mark.parametrize("scope", ["project", "main"])
def test_valid_exact_resume_never_reads_a_broken_saved_work_source(host, tmp_path, monkeypatch, scope):  # noqa: F811
    from ouroboros.artifacts import task_artifact_dir_path

    sent = _recovered_attempt_running(host, tmp_path, monkeypatch, scope)
    _owner_pause_at(host, monkeypatch, 29)
    row = _granted(host)
    source = task_artifact_dir_path(host.root, "held") / row["_working_recovery"]["source_ref"]["path"]
    assert source.is_file()
    source.write_bytes(b"{torn")
    with pytest.raises((OSError, ValueError)):
        wc.recovery_source_for_task(host.root, row)
    _no_working_read(monkeypatch)
    workers.assign_tasks()
    assert [task["id"] for task in sent] == ["held", "held"]
    assert sent[-1]["_budget_pause_resume"]["grant_id"] == row["_budget_pause_resume"]["grant_id"]


# Malformed carriers (their loaders refuse them too), then supported races before the handoff.
@pytest.mark.parametrize("fault", ["wrong_grant", "spent_owner_wait",
                                   "revoked_race", "consumed_race", "newer_mark", "stop", "new_pause"])
def test_invalid_governing_resume_refuses_without_falling_back(host, tmp_path, monkeypatch, fault):  # noqa: F811
    from ouroboros import budget_pause
    from ouroboros.cancel_intents import request_cancel
    from supervisor import task_admission
    from supervisor.owner_pause_control import request_owner_pause

    sent = _recovered_attempt_running(host, tmp_path, monkeypatch, "project")
    _owner_pause_at(host, monkeypatch, 29)
    row = _granted(host)
    pause = budget_pause.budget_pause_row(host.root, "held")
    if fault == "wrong_grant":
        row["_budget_pause_resume"]["grant_id"] = "f" * 32
    elif fault in {"revoked_race", "consumed_race"}:
        stamp = {"revoked_at": "2026-10-11T00:00:00+00:00"} if fault == "revoked_race" else {"consumed_at": time.time()}
        budget_pause.set_budget_pause(host.root, "held", {**pause, "grant": {**pause["grant"], **stamp}},
                                      expected_pause_id=pause["pause_id"])
    elif fault == "newer_mark":
        write_task_result(host.root, "held", "scheduled", admitted_dispatch_attempt=3)
    elif fault == "spent_owner_wait":
        row["_owner_wait_resume"] = {"wait_id": "spent-wait", "restart_transaction_id": "tx-old"}
    elif fault == "stop":
        request_cancel(host.root, "held", reason="owner stop", source="owner", requested_by="owner")
    else:
        request_owner_pause("held", request_id="pause-again")
    _no_working_read(monkeypatch)
    if fault not in {"stop", "new_pause"}:
        assert not task_admission.record_project_dispatch_possible(row), "the final projector refuses"
    workers.assign_tasks()
    workers.assign_tasks()
    assert len(sent) == 1, "no handoff and no fallback to the earlier saved work"
    stored, pause = load_task_result(host.root, "held"), budget_pause.budget_pause_row(host.root, "held")
    assert stored["admitted_dispatch_attempt"] == (3 if fault == "newer_mark" else 2)
    if fault == "stop":
        assert stored["status"] == "cancelled" and not host.pending
    elif fault == "new_pause":  # the newer Pause returns the undispatched grant to its exact pause
        assert host.pending[0]["_budget_pause"] and pause["grant"]["revoked_at"]
    else:  # custody and the durable grant stay exactly as the refusal found them
        assert host.pending == [row] and pause["state"] == "resume_granted"
        assert bool(pause["grant"].get("consumed_at")) is (fault == "consumed_race")


@pytest.mark.parametrize("scope", ["project", "main"])
def test_dead_exact_handoff_never_falls_back_to_an_unstarted_saved_source(
    host, tmp_path, monkeypatch, scope,  # noqa: F811
):
    from supervisor import task_admission

    _running(host, tmp_path, monkeypatch, scope)
    assert _stop_and_boot(queue, workers) == 1
    assert queue.resume_budget_paused_task("held")["ok"]
    [restored] = host.pending
    restored["_budget_pause_resume"] = {"grant_id": "dead-grant", "pause_id": "dead-pause"}
    sent = worker(host, monkeypatch)
    workers.assign_tasks()
    workers.assign_tasks()
    assert not sent and load_task_result(host.root, "held").get("admitted_dispatch_attempt") != 2
    # Hold revalidation and the final projector select the same carrier; neither falls back.
    assert restored["_project_admission_restore_hold"]["reason"] == "project_dispatch_unconfirmed"
    assert not task_admission.record_project_dispatch_possible(restored)
    assert wc.recovery_source_for_task(host.root, restored), "the saved source stays intact for its own handoff"
    restored.pop("_budget_pause_resume")
    workers.assign_tasks()
    workers.assign_tasks()
    assert [task["id"] for task in sent] == ["held"] and sent[0]["_working_recovery"]
    assert load_task_result(host.root, "held")["admitted_dispatch_attempt"] == 2


def test_restart_revokes_the_undispatched_grant_and_a_fresh_resume_dispatches_once(
    host, tmp_path, monkeypatch,  # noqa: F811
):
    from ouroboros import budget_pause

    _recovered_attempt_running(host, tmp_path, monkeypatch, "project")
    _owner_pause_at(host, monkeypatch, 29)
    stale = _granted(host)["_budget_pause_resume"]["grant_id"]
    monkeypatch.setattr(workers, "WORKERS", {})
    assert _stop_and_boot(queue, workers) == 1
    [restored] = host.pending
    assert restored["_budget_pause"] and not restored.get("_budget_pause_resume")
    assert budget_pause.budget_pause_row(host.root, "held")["grant"]["revoked_at"]
    fresh = _granted(host)
    assert fresh["_budget_pause_resume"]["grant_id"] != stale
    sent = worker(host, monkeypatch)
    workers.assign_tasks()
    workers.assign_tasks()
    assert [task["id"] for task in sent] == ["held"]
    assert sent[0]["_budget_pause_resume"]["grant_id"] == fresh["_budget_pause_resume"]["grant_id"]


@pytest.mark.parametrize("scope", ["project", "main"])
def test_saved_work_paused_before_its_first_handoff_keeps_its_source(
    host, tmp_path, monkeypatch, scope,  # noqa: F811
):
    from supervisor.owner_pause_control import request_owner_pause

    _running(host, tmp_path, monkeypatch, scope)
    assert _stop_and_boot(queue, workers) == 1
    assert queue.resume_budget_paused_task("held")["ok"]
    assert request_owner_pause("held", request_id="queued-pause")["ok"]
    sent = worker(host, monkeypatch)
    workers.assign_tasks()
    assert not sent
    assert queue.resume_budget_paused_task("held")["ok"]
    workers.assign_tasks()
    workers.assign_tasks()
    assert [task["id"] for task in sent] == ["held"] and sent[0]["_attempt"] == 2
    assert wc.recovery_source_for_task(host.root, sent[0])["task_attempt"] == 1
    assert load_task_result(host.root, "held")["admitted_dispatch_attempt"] == 2
