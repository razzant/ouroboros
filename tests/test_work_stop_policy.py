"""Stops retain their decision; every Resume door separately proves continuation."""
import copy
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from ouroboros.work_stop import (
    continuation_refusal, current_stop, reason_policy, release_refusal, stop_record,
)
from supervisor import events_budget as eb
from tests._budget_pause_exact_helpers import _install_queue, _parked


@pytest.mark.parametrize("state", ["requested", "paused", "released", "absent"])
def test_owner_marker_after_root_release_requires_member_selection(state):
    task = {"id": "child", "root_task_id": "root", "_budget_pause": {"reason": "owner", "exact_continuation": True}}
    fence = {"state": state, "fence_id": "f"} if state != "absent" else {}
    stop = current_stop(task, {"owner": fence})
    assert stop["release"] == "resume"
    assert stop["cause"] == ("pause" if state in {"requested", "paused"} else "awaiting_selection")
    assert stop["scope"] == ("tree" if state in {"requested", "paused"} else "member")
    assert "member_stop" not in stop


@pytest.mark.parametrize("state", ["released", "absent"])
@pytest.mark.parametrize("actor", ["owner", "mind"])
def test_zero_dispatch_owner_marker_selects_against_the_current_fence(tmp_path, monkeypatch, state, actor):
    from ouroboros.control_actor import mind_control
    from ouroboros.task_results import write_task_result

    q, _state, workers = _install_queue(tmp_path, monkeypatch)
    root = {"id": "root", "root_task_id": "root", "type": "task", "_attempt": 1, "admitted_dispatch": "none"}
    write_task_result(tmp_path, "root", "scheduled", owner_pause=(
        {"state": "released", "fence_id": "ended", "generation": 1} if state == "released" else {}))
    workers.PENDING.append(root)
    eb.hold_budget_row(root, reason=eb.HOLD_SAVED_WORK)
    assert q.resume_budget_paused_task("root")["ok"]
    child = {"id": "child", "root_task_id": "root", "parent_task_id": "root", "type": "task", "_attempt": 1,
             "admitted_dispatch": "none", "_budget_pause": {
                 "reason": "owner", "scope": "root", "root_task_id": "root", "fence_id": "ended",
                 "status": "paused_before_dispatch", "replay_safe": True, "physical_calls": 0}}
    workers.PENDING.append(child)
    write_task_result(tmp_path, "child", "scheduled", root_task_id="root", parent_task_id="root")
    with mind_control("root", "select child") if actor == "mind" else nullcontext():
        outcome = q.resume_budget_paused_task("child", selected_by="root" if actor == "mind" else "")
    assert outcome["ok"], outcome
    assert "_budget_pause" not in child and eb.BUDGET_HOLD_KEY not in child
    assert child["_resume_selection"]["fence_id"] == ""
    assert eb.budget_resume_dispatch_allowed(q, child) is True


def test_scopes_keep_independent_member_authority_and_sleep_is_only_waiting():
    task = {"id": "child", "root_task_id": "root", "_budget_pause": {"reason": "sleep"}}
    assert not current_stop(task)
    task[eb.BUDGET_HOLD_KEY] = stop_record(eb.HOLD_PANIC, held_at="T")
    stop = current_stop(task, {"owner": {"state": "paused", "fence_id": "f"}})
    assert stop["cause"] == "pause" and stop["member_stop"]["cause"] == "panic"
    assert release_refusal(stop, {"actor": "ouroboros"})["error"] == "panic_requires_owner_resume"
    assert release_refusal(stop, {}) is None
    assert current_stop(task)["release"] == "owner"


@pytest.mark.parametrize("reason", ["unrecognized_future_name", "restore_refused:source_missing", "malformed_resume_identity"])
def test_unknown_or_technical_reason_never_creates_a_new_stop(reason):
    task = {"id": "root", "_budget_pause": {"reason": "sleep", "checkpoint": {"pause_id": "p"}},
            eb.BUDGET_HOLD_KEY: stop_record(eb.HOLD_PANIC, held_at="before")}
    prior = copy.deepcopy(task[eb.BUDGET_HOLD_KEY])
    eb.hold_budget_row(task, reason=reason, extra={"pause_id": "p", "grant_id": "g"})
    assert reason_policy(reason)["release"] == "resume"
    assert task[eb.BUDGET_HOLD_KEY] == prior and current_stop(task)["release"] == "owner"
    assert continuation_refusal(task)["reason"] == reason
    assert "release" not in task["_budget_pause"]["refusal"]


def test_revalidating_a_spent_grant_cannot_release_a_current_panic(tmp_path):
    from supervisor.budget_resume import resume_hold_requires_fence

    task = {"id": "root", eb.BUDGET_HOLD_KEY: stop_record(eb.HOLD_PANIC, held_at="T")}
    eb.hold_budget_row(task, reason=eb.HOLD_GRANT_CONSUMED_STALE_ROW, extra={"pause_id": "p", "grant_id": "g"})
    task["_working_recovery"] = {"source_task_id": "root", "from_attempt": 1}
    assert not resume_hold_requires_fence(task, tmp_path)
    assert not continuation_refusal(task) and current_stop(task)["release"] == "owner"
    assert "_budget_pause_resume" not in task


def test_unknown_legacy_name_does_not_inherit_the_old_none_release(tmp_path, monkeypatch):
    from ouroboros.gateway.state import _pending_resume_available
    from ouroboros.task_results import write_task_result

    q, _state, workers = _install_queue(tmp_path, monkeypatch)
    task = {"id": "root", "root_task_id": "root", "type": "task", "_attempt": 1, "admitted_dispatch": "none",
            eb.BUDGET_HOLD_KEY: {"reason": "unrecognized_text", "release": "none"}}
    workers.PENDING.append(task)
    write_task_result(tmp_path, "root", "scheduled")
    assert not current_stop(task) and _pending_resume_available(q, task, owner_paused=False)
    assert q.resume_budget_paused_task("root")["ok"]
    assert eb.BUDGET_HOLD_KEY not in task and not continuation_refusal(task)


@pytest.mark.parametrize("door", ["saved_sleep_recovery", eb.HOLD_SAVED_WORK, eb.HOLD_OWNER_RESTART, eb.HOLD_PANIC])
@pytest.mark.parametrize("prior", [None, "saved_sleep_recovery", eb.HOLD_CONTINUATION_WRITER,
                                   eb.HOLD_SAVED_WORK, eb.HOLD_OWNER_RESTART, eb.HOLD_PANIC])
def test_doors_preserve_the_existing_replacement_matrix(tmp_path, prior, door):
    from supervisor.restart_retention import hold_after_stop

    task = {"id": "root", "_budget_pause": {"reason": "sleep"}}
    if prior:
        task[eb.BUDGET_HOLD_KEY] = stop_record(prior, held_at="before")
    hold_after_stop(task, tmp_path, door)
    replaces = prior is None or door != "saved_sleep_recovery" and prior in {
        "saved_sleep_recovery", eb.HOLD_CONTINUATION_WRITER}
    hold = task[eb.BUDGET_HOLD_KEY]
    assert hold["reason"] == (door if replaces else prior)
    assert "prior_hold" not in hold
    first = copy.deepcopy(task)
    hold_after_stop(task, tmp_path, door)
    assert task == first


@pytest.mark.parametrize("source", ["unstarted", "exact", "missing"])
@pytest.mark.parametrize("reason", [eb.HOLD_SAVED_WORK, eb.HOLD_PANIC, "restore_refused:unavailable", "unknown_name"])
@pytest.mark.parametrize("door", ["owner", "mind"])
def test_resume_door_stop_and_source_matrix(tmp_path, monkeypatch, source, reason, door):
    from ouroboros.control_actor import mind_control
    from ouroboros.task_results import write_task_result

    q, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda *a, **k: 5.0)
    if source == "exact":
        task, _ = _parked(tmp_path, monkeypatch, task_id="root")
    else:
        task = {"id": "root", "root_task_id": "root", "type": "task", "_attempt": 1,
                "admitted_dispatch": "none" if source == "unstarted" else "possible"}
        workers.PENDING.append(task)
        write_task_result(tmp_path, "root", "scheduled")
    # Existing stops can coexist with independent exact custody (also legacy shapes).
    if reason_policy(reason)["cause"] != "refusal":
        task[eb.BUDGET_HOLD_KEY] = stop_record(reason, held_at="T")
    else:
        eb.hold_budget_row(task, reason=reason)
    before = copy.deepcopy(task)
    with mind_control("mind", "continue this work") if door == "mind" else nullcontext():
        result = q.resume_budget_paused_task("root")
    allowed = source != "missing" and not (door == "mind" and reason == eb.HOLD_PANIC)
    assert result["ok"] is allowed, result
    if allowed:
        assert not current_stop(task) and not continuation_refusal(task)
        assert eb.BUDGET_HOLD_KEY not in task
        assert bool(task.get("_budget_pause_resume")) == (source == "exact")
    else:
        assert task == before
        if door == "mind" and reason == eb.HOLD_PANIC:
            assert result["error"] == "panic_requires_owner_resume"


def test_pause_cannot_launder_panic_but_owner_can_replace_it(tmp_path, monkeypatch):
    from ouroboros.owner_pause import read_fence
    from ouroboros.task_results import write_task_result
    from supervisor.events_mind_controls import _apply
    from supervisor.owner_pause_control import request_owner_pause

    q, _state, workers = _install_queue(tmp_path, monkeypatch)
    task = {"id": "root", "root_task_id": "root", "type": "task", "_attempt": 1, "admitted_dispatch": "none"}
    workers.PENDING.append(task)
    write_task_result(tmp_path, "root", "scheduled")
    eb.hold_budget_row(task, reason=eb.HOLD_PANIC)
    prior = copy.deepcopy(task)
    for action in ("pause_task", "resume_task"):
        outcome = _apply({"action": action, "task_id": "root", "actor_task_id": "mind",
                          "action_nonce": action, "reason": "try this decision"})
        assert outcome["error"] == "panic_requires_owner_resume"
        assert task == prior and not read_fence(tmp_path, "root")
    assert request_owner_pause("root", request_id="owner-replaces")["ok"]
    assert eb.BUDGET_HOLD_KEY not in task
    assert current_stop(task, {"budget": q.BUDGET_ROOT_FENCES})["cause"] == "pause"
    assert q.resume_budget_paused_task("root")["ok"]


def test_pause_does_not_release_a_different_stop_arriving_during_its_fence_write(tmp_path, monkeypatch):
    from ouroboros import owner_pause
    from ouroboros.task_results import write_task_result
    from supervisor.owner_pause_control import request_owner_pause

    _q, _state, workers = _install_queue(tmp_path, monkeypatch)
    task = {"id": "root", "root_task_id": "root", "type": "task", "_attempt": 1, "admitted_dispatch": "none"}
    workers.PENDING.append(task)
    write_task_result(tmp_path, "root", "scheduled")
    install = owner_pause.install_fence

    def newer(*args, **kwargs):
        accepted = install(*args, **kwargs)
        eb.hold_budget_row(task, reason=eb.HOLD_PANIC)
        return accepted

    monkeypatch.setattr(owner_pause, "install_fence", newer)
    assert request_owner_pause("root", request_id="earlier")["error"] == "selection_authority_changed"
    assert current_stop(task)["release"] == "owner"


def test_panic_sleep_refuses_both_mind_doors_and_owner_releases(tmp_path, monkeypatch):
    from ouroboros import budget_pause as bp, model_sleep
    from supervisor.events_mind_controls import _apply
    from tests._budget_pause_exact_helpers import _fast_hold, _loop_ctx, _quiet_external
    from tests.test_restart_saved_work import _working_run

    q, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda *a, **k: 5.0)
    _working_run(tmp_path, workers, "root")
    _working_run(tmp_path, workers, "child", parent="root")
    ctx, limit = _loop_ctx(tmp_path, "child")
    ctx.root_task_id = "root"
    ctx.task_metadata = {"parent_task_id": "root", "root_task_id": "root"}
    _fast_hold(monkeypatch, bp)
    _quiet_external(monkeypatch, bp)
    monkeypatch.setattr(model_sleep, "cold_blockers", lambda *a, **k: [])
    ctx._model_sleep = {"sleep_id": "cold", "mode": "cold", **model_sleep.selectors(ctx, wake_after_sec=3600)}
    with pytest.raises(bp.BudgetPauseRequested) as parked:
        bp.enter_cold_sleep(limit)
    bp.end_dispatch_fence("child")
    outcomes = []
    sup = SimpleNamespace(DRIVE_ROOT=tmp_path, RUNNING=workers.RUNNING, PENDING=workers.PENDING,
                          WORKERS=workers.WORKERS, sort_pending=q.sort_pending,
                          persist_queue_snapshot=q.persist_queue_snapshot,
                          bridge=SimpleNamespace(push_log=outcomes.append))
    eb.install_exact_budget_pause(sup, "child", bp.exact_pause_marker(parked.value.pause)["checkpoint"])
    task = next(row for row in q.PENDING if row["id"] == "child")
    from supervisor.restart_retention import hold_after_stop
    hold_after_stop(task, tmp_path, eb.HOLD_PANIC)
    assert _apply({"action": "resume_task", "task_id": "child", "actor_task_id": "root",
                   "action_nonce": "generic", "reason": "wake"})["error"] == "panic_requires_owner_resume"
    eb._handle_budget_resume_child({"task_id": "child", "requested_by": "root", "reason": "wake"}, sup)
    assert outcomes[-1]["error"] == "panic_requires_owner_resume"
    assert current_stop(task)["release"] == "owner"
    assert q.resume_budget_paused_task("child")["ok"]
    assert eb.BUDGET_HOLD_KEY not in task
