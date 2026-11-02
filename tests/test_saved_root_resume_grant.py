"""A real saved-root Resume admits explicit child choices, never its whole tree.

Uses the existing stop/restore/queue consumers; physical workers are doubles.
"""
import copy

import pytest

from tests._budget_pause_exact_helpers import _install_queue, _supervisor_ctx
from tests.test_budget_pause_holds import _idle_worker
from tests.test_restart_saved_work import _pool_events, _stop_and_boot, _working_run

pytestmark = pytest.mark.serial


def _saved_tree(tmp_path, monkeypatch):
    from supervisor import events_budget as eb

    q, state, workers = _install_queue(tmp_path, monkeypatch)
    _pool_events(workers, monkeypatch)
    _working_run(tmp_path, workers, "saved-root")
    _working_run(tmp_path, workers, "first", parent="saved-root")
    _working_run(tmp_path, workers, "second", parent="saved-root")
    assert _stop_and_boot(q, workers) == 3
    rows = {t["id"]: t for t in workers.PENDING}
    assert all(eb.budget_hold_fact(t)["reason"] == eb.HOLD_SAVED_WORK for t in rows.values())
    monkeypatch.setattr(state, "budget_remaining", lambda *a, **k: 5.0)
    return q, workers, rows


def test_saved_root_resume_then_model_selects_one_child_through_final_dispatch(tmp_path, monkeypatch):
    from supervisor import events_budget as eb

    q, workers, rows = _saved_tree(tmp_path, monkeypatch)
    pushed = []
    ctx = _supervisor_ctx(tmp_path, workers, q, [], pushed)
    request = {"task_id": "first", "requested_by": "saved-root", "reason": "continue its saved work"}
    eb._handle_budget_resume_child(request, ctx)
    assert not pushed[-1]["ok"] and pushed[-1]["error"] == "root_resume_grant_missing"
    assert q.resume_budget_paused_task("saved-root")["ok"]
    grant = eb.live_root_resume_grant(q, "saved-root", tmp_path)
    assert grant["grant_id"] == rows["saved-root"][eb.SELECTION_KEY]["root_grant_id"]
    sent = []
    _idle_worker(workers, sent)
    workers.assign_tasks()
    assert [t["id"] for t in sent] == ["saved-root"]
    assert eb.live_root_resume_grant(q, "saved-root", tmp_path) == grant
    eb._handle_budget_resume_child({**request, "requested_by": "foreign"}, ctx)
    assert pushed[-1]["error"] == "not_a_budget_paused_descendant"
    eb._handle_budget_resume_child(request, ctx)
    assert pushed[-1]["ok"]
    assert rows["first"][eb.SELECTION_KEY]["root_grant_id"] == grant["grant_id"]
    workers.WORKERS[0].busy_task_id = None
    workers.assign_tasks()
    assert [t["id"] for t in sent] == ["saved-root", "first"]
    assert eb.budget_hold_fact(rows["second"]), "Resume never fans out the other child"
    assert sent[-1]["_working_recovery"]["source_ref"], "dispatch carries saved cognition, not a replay"


def test_explicit_owner_child_selection_survives_later_root_grant_visibility(tmp_path, monkeypatch):
    from supervisor import events_budget as eb

    q, workers, rows = _saved_tree(tmp_path, monkeypatch)
    assert q.resume_budget_paused_task("first")["ok"]
    assert rows["first"][eb.SELECTION_KEY]["selected_by"] == "owner"
    assert not rows["first"][eb.SELECTION_KEY].get("root_grant_id")
    assert q.resume_budget_paused_task("saved-root")["ok"]
    grant = eb.live_root_resume_grant(q, "saved-root", tmp_path)
    assert rows["first"][eb.SELECTION_KEY]["root_grant_id"] == grant["grant_id"]
    assert eb.budget_resume_dispatch_allowed(q, rows["first"])
    sent = []
    _idle_worker(workers, sent)
    for _ in range(3):
        workers.WORKERS[0].busy_task_id = None
        workers.assign_tasks()
    assert set(t["id"] for t in sent) == {"saved-root", "first"}
    assert eb.budget_hold_fact(rows["second"])


@pytest.mark.parametrize("change", ["new_fence", "generation", "unknown_pause", "owner_pause", "cancel"])
def test_child_selection_is_revalidated_before_dispatch(tmp_path, monkeypatch, change):
    from ouroboros import budget_pause, owner_pause
    from ouroboros.cancel_intents import request_cancel
    from supervisor import events_budget as eb

    q, workers, rows = _saved_tree(tmp_path, monkeypatch)
    assert q.resume_budget_paused_task("saved-root")["ok"]
    assert q.resume_budget_paused_task("first", selected_by="saved-root")["ok"]
    root, child = rows["saved-root"], rows["first"]
    before = copy.deepcopy(child)
    if change == "new_fence":
        eb._set_root_budget_pause_locked("saved-root", {"fence_id": "new-fence"})
    elif change == "generation":
        root[eb.SELECTION_KEY]["root_grant_id"] = "later-explicit-selection"
    elif change == "unknown_pause":
        monkeypatch.setattr(budget_pause, "budget_pause_row", lambda *a: (_ for _ in ()).throw(OSError("unreadable")))
    elif change == "owner_pause":
        monkeypatch.setattr(owner_pause, "read_fence", lambda *a: {
            "state": "paused", "fence_id": "owner-new", "root_task_id": "saved-root"})
    else:
        # Public cancellation intent is checked by assignment, not guessed from a grant.
        request_cancel(tmp_path, "first", reason="owner stopped child")
    sent = []
    _idle_worker(workers, sent)
    workers.assign_tasks()
    assert "first" not in [t["id"] for t in sent]
    assert before["_working_recovery"]["source_ref"], "a refusal does not authorize replay"


@pytest.mark.parametrize("reason", ["saved_work_hold", "owner_restart_hold", "panic_hold", "continuation_writer_unsettled"])
def test_the_manual_root_selection_grant_covers_each_existing_selectable_hold(tmp_path, monkeypatch, reason):
    from supervisor import events_budget as eb

    q, _workers, rows = _saved_tree(tmp_path, monkeypatch)
    # The recovery source and fresh authority checks are real. Only the reason
    # for holding this same saved source varies; Panic's stop flag is absent,
    # as after a manual application start, and this is an explicit owner Resume.
    rows["saved-root"][eb.BUDGET_HOLD_KEY]["reason"] = reason
    assert q.resume_budget_paused_task("saved-root")["ok"]
    assert q.resume_budget_paused_task("first", selected_by="saved-root")["ok"]
    assert eb.budget_hold_fact(rows["second"])


def test_failed_root_snapshot_restores_existing_owner_selection(tmp_path, monkeypatch):
    from supervisor import events_budget as eb

    q, _workers, rows = _saved_tree(tmp_path, monkeypatch)
    assert q.resume_budget_paused_task("first")["ok"]
    before = copy.deepcopy(rows)
    real_snapshot = q.persist_queue_snapshot
    observed = []

    def snapshot(*args, **kwargs):
        if kwargs.get("reason") == "budget_hold_selected":
            observed.append(rows["first"][eb.SELECTION_KEY].get("root_grant_id"))
            return False
        return real_snapshot(*args, **kwargs)

    monkeypatch.setattr(q, "persist_queue_snapshot", snapshot)
    assert q.resume_budget_paused_task("saved-root")["error"] == "snapshot_not_persisted"
    assert observed and observed[0], "child binding belongs to the root's own transaction"
    assert rows == before, "no child is rebound under an unpublished root grant"


def test_a_new_generation_refuses_the_rebound_owner_selection(tmp_path, monkeypatch):
    from supervisor import events_budget as eb

    q, workers, rows = _saved_tree(tmp_path, monkeypatch)
    assert q.resume_budget_paused_task("first")["ok"]
    assert q.resume_budget_paused_task("saved-root")["ok"]
    selected = copy.deepcopy(rows["first"][eb.SELECTION_KEY])
    rows["saved-root"][eb.SELECTION_KEY]["root_grant_id"] = "new-root-resume"
    assert not eb.budget_resume_dispatch_allowed(q, rows["first"])
    assert rows["first"][eb.SELECTION_KEY] == selected, "reader never repairs authority"
    sent = []
    _idle_worker(workers, sent)
    workers.assign_tasks()
    assert "first" not in [row["id"] for row in sent]


def test_a_consumed_exact_grant_does_not_override_a_new_saved_work_resume(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from ouroboros import budget_pause as bp
    from ouroboros import owner_wait
    from supervisor import events_budget as eb
    from tests._budget_pause_exact_helpers import _loop_ctx, _parked

    q, state, _workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda *a, **k: 5.0)
    task, _row = _parked(tmp_path, monkeypatch, task_id="saved-root")
    assert q.resume_budget_paused_task("saved-root")["ok"]
    ctx, _limit = _loop_ctx(tmp_path, "saved-root")
    ctx.budget_pause_resume = task["_budget_pause_resume"]
    saved = bp.load_budget_pause(ctx)
    monkeypatch.setattr(owner_wait, "rebind_restored_route", lambda *a, **k: (None, "max"))
    bp.resume_paused_loop(SimpleNamespace(_ctx=ctx), saved, [], {}, {}, set(), budget_remaining_usd=5.0)
    old = bp.budget_pause_row(tmp_path, "saved-root")
    assert old["grant"]["consumed_at"] and not old["grant"].get("revoked_at")

    q, workers, rows = _saved_tree(tmp_path, monkeypatch)
    assert bp.budget_pause_row(tmp_path, "saved-root")["grant"] == old["grant"]
    assert q.resume_budget_paused_task("saved-root")["ok"]
    fresh = eb.live_root_resume_grant(q, "saved-root", tmp_path)
    assert fresh["grant_id"] == rows["saved-root"][eb.SELECTION_KEY]["root_grant_id"]
    assert fresh["grant_id"] != old["grant"]["grant_id"]
    assert q.resume_budget_paused_task("first", selected_by="saved-root")["ok"]
    assert rows["first"][eb.SELECTION_KEY]["root_grant_id"] == fresh["grant_id"]
    assert q.persist_queue_snapshot(reason="selected_saved_tree")
    # Inspect the durable snapshot itself: no read-side mutation supplies these ids.
    import json
    snapshot = json.loads((tmp_path / "state" / "queue_snapshot.json").read_text(encoding="utf-8"))
    assert fresh["grant_id"] in json.dumps(snapshot)


@pytest.mark.parametrize("prior_kind", ["saved", "exact"])
def test_selected_saved_hold_is_superseded_by_a_real_warm_resume(tmp_path, monkeypatch, prior_kind):
    from ouroboros import budget_pause as bp
    from ouroboros import owner_pause
    from ouroboros.task_results import write_task_result
    from supervisor import continuation_admission
    from supervisor import events_budget as eb
    from supervisor.budget_resume import resume_warm_owner_pause_root

    q, workers, rows = _saved_tree(tmp_path, monkeypatch)
    assert q.resume_budget_paused_task("saved-root")["ok"]
    assert q.resume_budget_paused_task("first", selected_by="saved-root")["ok"]
    sent = []
    _idle_worker(workers, sent)
    workers.assign_tasks()
    assert [item["id"] for item in sent] == ["saved-root"]
    live = workers.RUNNING["saved-root"]["task"]
    stale = copy.deepcopy(live)
    if prior_kind == "exact":
        # A consumed exact handoff remains on the running row. The actual warm
        # producer must supersede it too, without rewriting durable history.
        live["_budget_pause_resume"] = {"grant_id": "prior-exact", "pause_id": "prior",
                                        "grant_generation": 1, "authority": "explicit_resume"}
        write_task_result(tmp_path, "saved-root", "running", budget_pause={
            "state": bp.STATE_RESUMED, "pause_id": "prior", "resume_generation": 1,
            "grant": {"grant_id": "prior-exact", "generation": 1, "consumed_at": 1}})
    fence, _ = owner_pause.install_fence(tmp_path, "saved-root", request_id="warm-next")
    eb._set_root_budget_pause_locked("saved-root", {**fence, "cause": "owner_pause"})
    # The actual Resume/grant/child-rebinding consumer runs. Physical parking
    # and whole-tree quiescence are fixture inputs, not live-process evidence.
    workers.RUNNING["saved-root"]["owner_wait"] = {"state": "waiting", "reason": "owner_pause"}
    monkeypatch.setattr(continuation_admission, "action_writers", lambda *a, **k: [])
    resumed = resume_warm_owner_pause_root("saved-root")
    assert resumed["ok"], resumed
    assert "_budget_pause_resume" not in live
    assert live[eb.SELECTION_KEY]['root_grant_id'] == resumed['grant_id']
    assert eb.live_root_resume_grant(q, "saved-root", tmp_path)["grant_id"] == resumed["grant_id"]
    before = copy.deepcopy(workers.RUNNING["saved-root"])
    assert not eb.fold_consumed_resume(workers.RUNNING["saved-root"], tmp_path, {
        "task_attempt": before.get("attempt"), "pause_id": "prior", "grant_id": "prior-exact"})
    assert workers.RUNNING["saved-root"] == before, "a delayed old consumption cannot revive its grant"
    # A stale queue projection must not override the currently running root.
    q.PENDING.append(stale)
    assert eb.live_root_resume_grant(q, "saved-root", tmp_path)["grant_id"] == resumed["grant_id"]
    q.PENDING.remove(stale)
    assert q.resume_budget_paused_task("first", selected_by="saved-root")["ok"]
    assert not rows["first"][eb.SELECTION_KEY].get("fence_id")
    assert eb.budget_resume_dispatch_allowed(q, rows["first"])
    workers.WORKERS[0].busy_task_id = None
    workers.assign_tasks()
    assert [item["id"] for item in sent] == ["saved-root", "first"]


@pytest.mark.parametrize("old_kind", ["exact", "warm"])
def test_a_stop_hold_masks_historical_grants_before_root_resume(tmp_path, monkeypatch, old_kind):
    from ouroboros import budget_pause as bp
    from ouroboros.task_results import write_task_result
    from supervisor import events_budget as eb

    q, _workers, rows = _saved_tree(tmp_path, monkeypatch)
    # Valid old authority is retained as history by stop; the live queue row
    # explicitly holds this root. This fixture tests the public selection seam,
    # not the earlier production of this historical grant.
    if old_kind == "exact":
        write_task_result(tmp_path, "saved-root", "running", budget_pause={
            "state": bp.STATE_RESUMED, "pause_id": "old", "resume_generation": 1,
            "grant": {"grant_id": "old-exact", "generation": 1, "consumed_at": 1}})
    else:
        write_task_result(tmp_path, "saved-root", "running", owner_pause={
            "state": "released", "fence_id": "old", "root_task_id": "saved-root",
            "resume_grant": {"grant_id": "old-warm", "generation": 1}})
    assert eb.live_root_resume_grant(q, "saved-root", tmp_path) == {}
    assert q.resume_budget_paused_task("first", selected_by="saved-root")["error"] == "root_resume_grant_missing"
    assert q.resume_budget_paused_task("first")["ok"], "the human's own selection is still allowed"
    assert not rows["first"][eb.SELECTION_KEY].get("root_grant_id")
    assert q.resume_budget_paused_task("saved-root")["ok"]
    assert rows["first"][eb.SELECTION_KEY]["root_grant_id"] == eb.live_root_resume_grant(
        q, "saved-root", tmp_path)["grant_id"]


# Zero-call application stops preserve explicit selection, not sibling fanout.

def _zero_call_app_stop_tree(tmp_path, monkeypatch):
    from ouroboros.task_results import STATUS_SCHEDULED, write_task_result
    from supervisor import events_budget as eb
    from tests.test_restart_retention import _queued

    q, state, workers = _install_queue(tmp_path, monkeypatch)
    _pool_events(workers, monkeypatch)
    for tid in ("zero-root", "selected", "sibling"):
        lineage = {"root_task_id": "zero-root"}
        if tid != "zero-root":
            lineage["parent_task_id"] = "zero-root"
        write_task_result(tmp_path, tid, STATUS_SCHEDULED,
                          chat_id=0, **lineage)
        q.PENDING.append(_queued(tid, **lineage))
    assert _stop_and_boot(q, workers) == 3
    rows = {row["id"]: row for row in q.PENDING}
    assert all(eb.budget_hold_fact(row)["reason"] == eb.HOLD_SAVED_WORK
               for row in rows.values())
    money = [5.0]
    monkeypatch.setattr(state, "budget_remaining", lambda *a, **k: money[0])
    return q, workers, rows, money


@pytest.mark.parametrize("legacy", [False, True], ids=["basis", "legacy"])
def test_global_zero_dispatch_pause_blocks_existing_bound_child(
        tmp_path, monkeypatch, legacy):
    from ouroboros import budget_pause as bp, owner_pause
    from ouroboros.agent_dispatch import _physical_calls_after_budget_rail
    from supervisor import events_budget as eb

    q, workers, rows, money = _zero_call_app_stop_tree(tmp_path, monkeypatch)
    assert q.resume_budget_paused_task("zero-root")["ok"]
    sent = []
    _idle_worker(workers, sent)
    workers.assign_tasks()
    assert [row["id"] for row in sent] == ["zero-root"]

    # The owner's child choice is bound to the currently running root grant.
    assert q.resume_budget_paused_task("selected")["ok"]
    child = rows["selected"]
    selection = child[eb.SELECTION_KEY]
    assert selection["selected_by"] == "owner"
    assert selection["root_grant_id"]
    assert selection["root_resume_basis"]
    if legacy:
        selection.pop("root_resume_basis")
    before_child = copy.deepcopy(child)
    before_pause = copy.deepcopy(bp.budget_pause_row(tmp_path, "zero-root"))
    before_owner = copy.deepcopy(owner_pause.read_fence(tmp_path, "zero-root"))
    assert before_pause == before_owner == {}
    assert q.BUDGET_ROOT_FENCES == {}

    # Real zero-dispatch park producer; no hand-written queue pause or grant.
    calls = _physical_calls_after_budget_rail(tmp_path, "zero-root")
    assert calls == 0
    money[0] = 0.0
    ctx = _supervisor_ctx(tmp_path, workers, q, [], [])
    ctx.persist_queue_snapshot = q.persist_queue_snapshot
    eb._handle_budget_pause({
        "type": "budget_pause", "task_id": "zero-root", "worker_id": 0,
        "root_task_id": "zero-root",
        "resource_limit": {
            "status": "paused_before_dispatch", "scope": "global",
            "root_task_id": "zero-root", "physical_calls": calls,
            "replay_safe": True, "auto_resume": False,
            "resume_policy": "manual_same_generation",
        },
    }, ctx)
    parked = next(row for row in q.PENDING if row["id"] == "zero-root")
    assert parked["_budget_pause"]["scope"] == "global"
    assert "zero-root" not in workers.RUNNING
    assert workers.WORKERS[0].busy_task_id is None
    assert bp.budget_pause_row(tmp_path, "zero-root") == before_pause
    assert owner_pause.read_fence(tmp_path, "zero-root") == before_owner
    assert q.BUDGET_ROOT_FENCES == {}
    assert child == before_child
    assert eb.live_root_resume_grant(q, "zero-root", tmp_path) == {}

    # Replenishment alone must not release the root-bound child.
    money[0] = 5.0
    assert eb.budget_resume_dispatch_allowed(q, child) is False
    workers.assign_tasks()
    assert [row["id"] for row in sent] == ["zero-root"]
    assert child in q.PENDING
    assert eb.budget_hold_fact(child) and eb.SELECTION_KEY not in child
    assert eb.budget_hold_fact(rows["sibling"])
    assert parked["_budget_pause"], "no root Resume occurred"


@pytest.mark.parametrize("legacy", [False, True], ids=["basis", "legacy"])
def test_terminal_root_still_allows_explicit_owner_unbound_child(
        tmp_path, monkeypatch, legacy):
    from ouroboros.task_results import STATUS_COMPLETED, write_task_result
    from supervisor import events_budget as eb

    q, workers, rows, _money = _zero_call_app_stop_tree(tmp_path, monkeypatch)
    assert q.resume_budget_paused_task("zero-root")["ok"]
    sent = []
    _idle_worker(workers, sent)
    workers.assign_tasks()
    assert [row["id"] for row in sent] == ["zero-root"]
    workers.RUNNING.pop("zero-root")
    workers.WORKERS[0].busy_task_id = None
    write_task_result(tmp_path, "zero-root", STATUS_COMPLETED,
                      chat_id=0, root_task_id="zero-root", result="done")

    assert q.resume_budget_paused_task("selected")["ok"]
    child = rows["selected"]
    selection = child[eb.SELECTION_KEY]
    assert selection["selected_by"] == "owner"
    assert not selection.get("root_grant_id")
    if legacy:
        selection.pop("root_resume_basis")
    assert eb.budget_resume_dispatch_allowed(q, child) is True
    workers.assign_tasks()
    assert [row["id"] for row in sent] == ["zero-root", "selected"]
    assert eb.budget_hold_fact(rows["sibling"])
