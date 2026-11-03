"""Released stops never regain authority through a Q9 selection.

S4's complete return path uses the real pause, grant, consumption, checkpoint,
restart and assignment owners. Cognition and physical workers use suite doubles.
"""
from contextlib import nullcontext
import copy
import json
from types import SimpleNamespace

import pytest

from ouroboros import budget_pause as bp, working_checkpoint as wc
from ouroboros.control_actor import mind_control
from ouroboros.work_stop import SELECTION_KEY, reason_policy, stop_record
from supervisor import events_budget as eb
from supervisor.owner_pause_control import request_owner_pause
from tests._budget_pause_exact_helpers import _install_queue, _loop_ctx
from tests.test_budget_pause_holds import _idle_worker
from tests.test_budget_pause_lineage_restore import _consume, _restart
from tests.test_model_sleep import _cold_park
from tests.test_restart_retention import _pool_events

pytestmark = pytest.mark.serial


def _released_panic_running(root, monkeypatch):
    q, state, workers = _install_queue(root, monkeypatch)
    _pool_events(workers, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda *a, **k: 5.0)
    _cold_park(root, monkeypatch, workers, wake_after_sec=3600)
    assert q.persist_queue_snapshot(reason="before_panic")
    workers.PENDING.clear()
    flag = root / "state" / "panic_stop.flag"
    flag.write_text("panic")
    assert q.restore_pending_from_snapshot() == 1
    [task] = workers.PENDING
    assert eb.budget_hold_fact(task)["reason"] == eb.HOLD_PANIC
    flag.unlink()  # the boot consumer removes it after publishing its holds
    with mind_control("mind", "resume"):
        assert q.resume_budget_paused_task("sleeper")["error"] == "panic_requires_owner_resume"
    resumed = q.resume_budget_paused_task("sleeper")
    assert resumed["ok"] and resumed["released_hold"] == eb.HOLD_PANIC
    assert eb.BUDGET_HOLD_KEY not in task
    sent = []
    _idle_worker(workers, sent)
    workers.assign_tasks()
    assert [row["id"] for row in sent] == ["sleeper"]
    meta = workers.RUNNING["sleeper"]
    ctx, limit = _loop_ctx(root, "sleeper", attempt=meta["attempt"])
    _consume(monkeypatch, ctx, limit, meta["task"]["_budget_pause_resume"])
    exact = bp.budget_pause_row(root, "sleeper")
    assert exact["state"] == bp.STATE_RESUMED and exact["grant"]["consumed_at"]
    wc.save_round(limit, "post_batch")
    return q, workers, exact


@pytest.mark.parametrize("actor", ["owner", "mind"])
def test_released_panic_cannot_return_after_pause_before_park_and_ack_restart(tmp_path, monkeypatch, actor):
    from tests.test_work_projection_pause_producers import _works

    q, workers, exact = _released_panic_running(tmp_path, monkeypatch)
    with mind_control("other-mind", ""):
        paused = request_owner_pause("sleeper", request_id="pause-before-park")
    assert paused["ok"] and paused["state"] == "requested"
    _restart(q, workers, monkeypatch)
    [back] = [row for row in workers.PENDING if row["id"] == "sleeper"]
    assert "_budget_pause" not in back and "_budget_pause_resume" not in back
    assert eb.BUDGET_HOLD_KEY not in back and "reason" not in back[SELECTION_KEY]
    assert eb.budget_resume_dispatch_allowed(q, back) is False
    for worker in workers.WORKERS.values():
        worker.busy_task_id = None
    workers.assign_tasks()
    assert SELECTION_KEY not in back and eb.budget_hold_fact(back) is None
    assert not workers.RUNNING, "a new Pause still vetoes dispatch"
    assert "Ouroboros paused it" in json.dumps(_works(tmp_path))
    with mind_control("other-mind", "continue after the update") if actor == "mind" else nullcontext():
        resumed = q.resume_budget_paused_task("sleeper")
    assert resumed["ok"], resumed
    assert eb.BUDGET_HOLD_KEY not in back
    assert bp.budget_pause_row(tmp_path, "sleeper")["grant"] == exact["grant"], "the spent grant is not reissued"
    workers.assign_tasks()
    assert "sleeper" in workers.RUNNING


@pytest.mark.parametrize("failed_publish", [False, True])
def test_two_root_resumes_invalidate_child_selection_without_copying_its_stop(tmp_path, monkeypatch, failed_publish):
    from tests.test_root_resume_selection_survival import _hold_root, _world

    q, workers, _sent = _world(tmp_path, monkeypatch, "saved")
    assert q.resume_budget_paused_task("first", selected_by="saved-root")["ok"]
    child = next(row for row in q.PENDING if row["id"] == "first")
    old = copy.deepcopy(child[SELECTION_KEY])
    assert eb.BUDGET_HOLD_KEY not in child
    root = _hold_root(tmp_path, q, workers)
    if failed_publish:
        before = copy.deepcopy(q.PENDING)
        persist = q.persist_queue_snapshot
        with monkeypatch.context() as failed:
            failed.setattr(q, "persist_queue_snapshot", lambda **kw: False
                           if kw.get("reason") == "budget_hold_selected" else persist(**kw))
            assert q.resume_budget_paused_task("saved-root")["error"] == "snapshot_not_persisted"
        assert q.PENDING == before, "a failed publication restores both the hold and child token"
    assert q.resume_budget_paused_task("saved-root")["ok"]
    current = root[SELECTION_KEY]
    assert current["root_grant_id"] != old["root_grant_id"]
    assert SELECTION_KEY not in child
    hold = eb.budget_hold_fact(child)
    assert hold["reason"] == eb.HOLD_ROOT_FENCE_LIFTED
    assert "cause" not in hold
    assert hold["root_grant_id"] == current["root_grant_id"]
    assert q.resume_budget_paused_task("first", selected_by="saved-root")["ok"]
    assert eb.budget_resume_dispatch_allowed(q, child) is True
    assert eb.budget_hold_fact(next(row for row in q.PENDING if row["id"] == "second")), "no fan-out"


def test_failed_exact_root_resume_preserves_the_owners_existing_child_choice(tmp_path, monkeypatch):
    from tests._budget_pause_exact_helpers import _parked
    from tests.test_budget_pause_holds import _fenced_member

    q, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda *a, **k: 5.0)
    _parked(tmp_path, monkeypatch, task_id="root", scope="root")
    child = _fenced_member(workers, "child", "root")
    assert q.resume_budget_paused_task("child")["ok"]
    selected = copy.deepcopy(child[SELECTION_KEY])
    assert eb.budget_resume_dispatch_allowed(q, child) is True
    persist = q.persist_queue_snapshot
    with monkeypatch.context() as failed:
        failed.setattr(q, "persist_queue_snapshot", lambda **kw: False
                       if kw.get("reason") == "budget_exact_resume_granted" else persist(**kw))
        assert q.resume_budget_paused_task("root")["error"] == "snapshot_not_persisted"
    assert child.get(SELECTION_KEY) == selected
    assert eb.budget_hold_fact(child) is None
    assert eb.budget_resume_dispatch_allowed(q, child) is True
    assert q.resume_budget_paused_task("root")["ok"]
    assert eb.budget_resume_dispatch_allowed(q, child) is False
    sent = []
    _idle_worker(workers, sent)
    workers.assign_tasks()
    assert [row["id"] for row in sent] == ["root"]
    assert SELECTION_KEY not in child
    assert eb.budget_hold_fact(child)["reason"] == eb.HOLD_ROOT_FENCE_LIFTED
    assert q.resume_budget_paused_task("child", selected_by="root")["ok"]


def test_panic_sleep_child_after_two_root_resumes_keeps_its_own_pause(tmp_path, monkeypatch):
    from ouroboros import model_sleep
    from tests._budget_pause_exact_helpers import _fast_hold, _quiet_external
    from tests.test_restart_saved_work import _stop_and_boot, _working_run

    q, state, workers = _install_queue(tmp_path, monkeypatch)
    _pool_events(workers, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda *a, **k: 5.0)
    _working_run(tmp_path, workers, "root")
    _working_run(tmp_path, workers, "child", parent="root")
    ctx, limit = _loop_ctx(tmp_path, "child")
    ctx.root_task_id = "root"
    ctx.task_metadata = {"parent_task_id": "root", "root_task_id": "root"}
    _fast_hold(monkeypatch, bp)
    _quiet_external(monkeypatch, bp)
    monkeypatch.setattr(model_sleep, "cold_blockers", lambda *a, **k: [])
    ctx._model_sleep = {"sleep_id": "cold-child", "mode": "cold", **model_sleep.selectors(ctx, wake_after_sec=3600)}
    with pytest.raises(bp.BudgetPauseRequested) as parked:
        bp.enter_cold_sleep(limit)
    bp.end_dispatch_fence("child")
    sup = SimpleNamespace(DRIVE_ROOT=tmp_path, RUNNING=workers.RUNNING, PENDING=workers.PENDING,
                          WORKERS=workers.WORKERS, sort_pending=q.sort_pending,
                          persist_queue_snapshot=q.persist_queue_snapshot, bridge=None)
    eb.install_exact_budget_pause(sup, "child", bp.exact_pause_marker(parked.value.pause)["checkpoint"])
    assert q.persist_queue_snapshot(reason="before-panic")
    flag = tmp_path / "state" / "panic_stop.flag"
    flag.write_text("panic")
    assert _stop_and_boot(q, workers) == 2
    flag.unlink()
    child = next(row for row in q.PENDING if row["id"] == "child")
    assert eb.budget_hold_fact(child)["reason"] == eb.HOLD_PANIC
    assert q.resume_budget_paused_task("child")["ok"]
    assert eb.BUDGET_HOLD_KEY not in child
    # Quit before the child's dispatch revokes that unused exact grant.
    assert _stop_and_boot(q, workers) == 2
    assert q.resume_budget_paused_task("root")["ok"]
    sent = []
    _idle_worker(workers, sent)
    workers.assign_tasks()
    assert [row["id"] for row in sent] == ["root"]
    ctx, limit = _loop_ctx(tmp_path, "root", attempt=workers.RUNNING["root"]["attempt"])
    wc.save_round(limit, "post_batch")
    workers.WORKERS.clear()  # physical workers are doubles; preserve their RUNNING records for the stop owner
    assert _stop_and_boot(q, workers) == 2
    assert q.resume_budget_paused_task("root")["ok"]
    child = next(row for row in q.PENDING if row["id"] == "child")
    assert child["_budget_pause"]["reason"] == "sleep"
    assert (eb.budget_hold_fact(child) or {}).get("reason") != eb.HOLD_PANIC
    with mind_control("root", "choose the child again"):
        resumed = q.resume_budget_paused_task("child", selected_by="root")
    assert resumed["ok"], resumed
    assert bp.budget_pause_row(tmp_path, "child")["grant"]["by"]["text"] == "choose the child again"


def test_mind_resume_text_never_becomes_the_next_stop_reason(tmp_path, monkeypatch):
    from tests.test_saved_root_resume_grant import _saved_tree

    q, workers, rows = _saved_tree(tmp_path, monkeypatch)
    prose = "Continue the earlier request after the update"
    with mind_control("another-root", prose):
        assert q.resume_budget_paused_task("saved-root")["ok"]
    selected = rows["saved-root"][SELECTION_KEY]
    assert selected["selected_by"] == "owner", "the existing Q9 door is independent of the actor"
    assert selected["by"]["text"] == prose and selected["by"]["actor"] == "ouroboros"
    assert "reason" not in selected
    assert q.resume_budget_paused_task("first", selected_by="saved-root")["ok"]
    sent = []
    _idle_worker(workers, sent)
    workers.assign_tasks()
    assert sent[0]["id"] == "saved-root"
    with mind_control("another-root", "pause before update"):
        assert request_owner_pause("saved-root", request_id="new-pause")["ok"]
    workers.assign_tasks()
    assert eb.budget_hold_fact(rows["first"])["reason"] == eb.HOLD_ROOT_FENCE_LIFTED
    assert "cause" not in eb.budget_hold_fact(rows["first"])
    assert SELECTION_KEY not in rows["first"]


@pytest.mark.parametrize("kind", ["exact", "warm"])
def test_mind_attribution_on_a_grant_preserves_its_q9_door(tmp_path, monkeypatch, kind):
    from ouroboros.owner_pause import read_fence
    from tests.test_root_resume_selection_survival import _world, _exact_root, _warm_root

    q, workers, _sent = _world(tmp_path, monkeypatch, "saved")
    with mind_control("another-mind", "resume the work"):
        (_exact_root if kind == "exact" else _warm_root)(tmp_path, monkeypatch, q, workers)
    grant = (bp.budget_pause_row(tmp_path, "saved-root")["grant"] if kind == "exact"
             else read_fence(tmp_path, "saved-root")["resume_grant"])
    assert grant["by"]["actor"] == "ouroboros" and grant["by"]["text"] == "resume the work"
    assert grant["selected_by"] == "owner" and "reason" not in grant
    assert q.resume_budget_paused_task("first", selected_by="saved-root")["ok"]


def test_money_gate_reads_selection_from_dispatch_on_every_send(tmp_path, monkeypatch):
    from tests.test_budget_pause_holds import _fenced_member
    from ouroboros.task_results import write_task_result
    from ouroboros.usage_accounting import AttemptRequest, BudgetExceeded, UsageScope, reserve_attempt, usage_scope

    q, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda *a, **k: 5.0)
    root = _fenced_member(workers, "root", "root")
    root.pop("parent_task_id")
    write_task_result(tmp_path, "root", "scheduled", root_task_id="root")
    fence = eb._set_root_budget_pause_locked("root", {"scope": "root", "root_task_id": "root"})
    chosen = _fenced_member(workers, "chosen", "root")
    _fenced_member(workers, "other", "root")
    assert q.resume_budget_paused_task("chosen")["ok"]
    dispatch = json.loads((tmp_path / "state" / "queue_dispatch.json").read_text())
    assert dispatch["members"]["chosen"][SELECTION_KEY] == {"fence_id": fence["fence_id"]}
    assert eb.BUDGET_HOLD_KEY not in dispatch["members"]["chosen"]
    with usage_scope(UsageScope(drive_root=tmp_path, task_id="chosen", root_task_id="root", global_limit_usd=100.0)):
        for _ in range(2):
            assert reserve_attempt(AttemptRequest(model="fixture", provider="openai", reservation_usd=1.0)).attempt_id
        eb._set_root_budget_pause_locked("root", {"fence_id": "next", "scope": "root"})
        assert q.persist_queue_snapshot(reason="next-pause")
        with pytest.raises(BudgetExceeded):
            reserve_attempt(AttemptRequest(model="fixture", provider="openai", reservation_usd=1.0))
    assert SELECTION_KEY not in chosen


@pytest.mark.parametrize("reason,release", [
    ("saved_work_hold", "resume"), ("owner_restart_hold", "resume"), ("panic_hold", "owner"),
    ("saved_sleep_recovery", "resume"), ("unsaved_pause_intent", "resume"),
    ("root_fence_lifted_pending_selection", "resume"), ("root_fence_member_pending_selection", "resume"),
    ("continuation_writer_unsettled", "none"), ("restore_refused:source_missing", "resume"),
    ("malformed_resume_identity", "resume"), ("resume_grant_revocation_unwritten", "resume"),
    ("stale_resume_grant_superseded", "resume"), ("dispatch_outcome_unknown", "resume"),
])
def test_hold_builder_keeps_release_and_attribution_separate(reason, release):
    by = {"actor": "ouroboros", "task_id": "mind", "text": "explanation", "at": "T"}
    held = stop_record(reason, held_at="T", by=by, binding={"root_task_id": "root"})
    assert held["reason"] == reason and held["release"] == release
    assert held["by"] == by and "text" not in held and "selected" not in held
    assert reason_policy(reason)["ui_code"]
