"""A real saved-root Resume admits explicit child choices, never its whole tree.

Uses the existing stop/restore/queue consumers; physical workers are doubles.
"""
import copy

import pytest

from tests._budget_pause_exact_helpers import _install_queue, _supervisor_ctx
from tests.test_budget_pause_holds import _idle_worker
from tests.test_restart_saved_work import _pool_events, _stop_and_boot, _working_run



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


@pytest.mark.parametrize("change", ["new_fence", "unknown_pause", "cancel"])
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


@pytest.mark.parametrize("reason", ["saved_work_hold", "owner_restart_hold", "continuation_writer_unsettled"])
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


# The owner's Pause of a queued saved root, then its Resume: the root lends one grant on its fence.

ROOT, CHOSEN, OTHER = "saved-root", "first", "second"


def _root_ran_children_queued_tree(tmp_path, monkeypatch):
    """The production shape: the root was dispatched by the real assignment and saved a round;
    while it ran it queued two children (the real ``enqueue_task``) that had not started when the
    application stopped."""
    from ouroboros import working_checkpoint as wc
    from ouroboros.task_results import STATUS_RUNNING, STATUS_SCHEDULED, write_task_result
    from supervisor import events_budget as eb
    from tests._budget_pause_exact_helpers import _loop_ctx

    q, state, workers = _install_queue(tmp_path, monkeypatch)
    _pool_events(workers, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda *a, **k: 5.0)
    write_task_result(tmp_path, ROOT, STATUS_SCHEDULED, chat_id=0, root_task_id=ROOT)
    q.PENDING.append({"id": ROOT, "type": "task", "chat_id": 0, "_attempt": 1, "text": ROOT,
                      "root_task_id": ROOT, "admitted_dispatch": "none"})
    sent = []
    _idle_worker(workers, sent)
    workers.assign_tasks()
    assert [t["id"] for t in sent] == [ROOT]
    for tid in (CHOSEN, OTHER):
        lineage = {"root_task_id": ROOT, "parent_task_id": ROOT}
        write_task_result(tmp_path, tid, STATUS_SCHEDULED, chat_id=0, delegation_role="subagent", **lineage)
        queued = q.enqueue_task({"id": tid, "type": "task", "chat_id": 0, "_attempt": 1, "text": tid,
                                 "delegation_role": "subagent", **lineage})
        assert not queued.get("_admission_blocked"), queued
    write_task_result(tmp_path, ROOT, STATUS_RUNNING, chat_id=0, root_task_id=ROOT)
    _ctx, limit = _loop_ctx(tmp_path, ROOT, attempt=1)
    wc.save_round(limit, "pre_effect")
    workers.WORKERS.clear()  # its worker process ends with the application
    assert _stop_and_boot(q, workers) == 3
    rows = {t["id"]: t for t in workers.PENDING}
    assert all(eb.budget_hold_fact(t)["reason"] == eb.HOLD_SAVED_WORK for t in rows.values())
    assert rows[ROOT]["admitted_dispatch"] == "possible" and rows[ROOT]["_working_recovery"]
    assert {rows[tid]["admitted_dispatch"] for tid in (CHOSEN, OTHER)} == {"none"}
    return q, workers, rows


def _owner_paused_tree(shape, tmp_path, monkeypatch):
    if shape == "root_ran":
        return (*_root_ran_children_queued_tree(tmp_path, monkeypatch), ROOT, CHOSEN, OTHER)
    if shape == "legacy_rows":  # saved rows without ``admitted_dispatch``: the non-admitted branch
        return (*_saved_tree(tmp_path, monkeypatch), ROOT, CHOSEN, OTHER)
    q, workers, rows, _money = _zero_call_app_stop_tree(tmp_path, monkeypatch)
    return q, workers, rows, "zero-root", "selected", "sibling"


def _pause(q, tmp_path, root, request_id="owner-press"):
    from ouroboros import owner_pause
    from supervisor import events_budget as eb
    from supervisor.owner_pause_control import request_owner_pause

    assert request_owner_pause(root, request_id=request_id)["ok"]
    return owner_pause.read_fence(tmp_path, root), eb._root_resume_state(q, root, tmp_path, actor_id=root)["basis"]


def _resume(q, tmp_path, root, paused, basis):
    """The owner's Resume; the one grant its released fence carries, bound to what the Resume observed."""
    from ouroboros import owner_pause

    resumed = q.resume_budget_paused_task(root)
    assert resumed["ok"] is True, resumed
    fence = owner_pause.read_fence(tmp_path, root)
    grant = fence.get("resume_grant") or {}
    assert grant.get("grant_id") and grant["grant_id"] == resumed.get("grant_id"), (fence, resumed)
    assert (fence["fence_id"], fence["state"], fence["release_reason"]) == (
        paused["fence_id"], "released", "owner_resume_unstarted_root")
    assert (grant["generation"], grant["root_resume_basis"]) == (paused["generation"], basis)
    assert (grant["authority"], grant["selected_by"]) == ("explicit_resume", "owner")
    assert not {"warm", "tree_only", "single_use", "consumed_at", "revoked_at"} & set(grant)
    return resumed, grant


def _model_selects(tmp_path, workers, q, child, root):
    from supervisor import events_budget as eb

    pushed = []
    eb._handle_budget_resume_child({"task_id": child, "requested_by": root, "reason": "continue its saved work"},
                                   _supervisor_ctx(tmp_path, workers, q, [], pushed))
    return pushed[-1]


def _own_resume_visible(q, tmp_path, rows, *ids):
    from types import SimpleNamespace

    from ouroboros.gateway.state import _pending_resume_available

    registries = SimpleNamespace(PENDING=q.PENDING, BUDGET_ROOT_FENCES=q.BUDGET_ROOT_FENCES,
                                 RUNNING=q.RUNNING, DRIVE_ROOT=tmp_path)
    return {tid: _pending_resume_available(registries, rows[tid], owner_paused=False) for tid in ids}


@pytest.mark.parametrize("shape", ["root_ran", "legacy_rows", "zero_call"])
def test_owner_resume_of_a_paused_queued_root_lends_one_grant_and_its_model_selects_one_child(
        tmp_path, monkeypatch, shape):
    """``root_ran`` is the production shape (the admitted branch). ``legacy_rows`` and ``zero_call``
    are fixture or legacy shapes reaching the never-started branch with children: the grant does not
    depend on members."""
    from supervisor import events_budget as eb

    q, workers, rows, root, chosen, other = _owner_paused_tree(shape, tmp_path, monkeypatch)
    paused, basis = _pause(q, tmp_path, root)
    children = copy.deepcopy({tid: rows[tid] for tid in (chosen, other)})
    resumed, grant = _resume(q, tmp_path, root, paused, basis)
    assert resumed["never_started"] is (shape != "root_ran")
    assert q.BUDGET_ROOT_FENCES == {} and eb.budget_hold_fact(rows[root]) is None
    assert {tid: rows[tid] for tid in (chosen, other)} == children
    # As on every other root Resume path, the root selects while it lends: no own Resume on its children.
    assert _own_resume_visible(q, tmp_path, rows, chosen, other) == {chosen: False, other: False}
    sent = []
    _idle_worker(workers, sent)
    workers.assign_tasks()
    assert [t["id"] for t in sent] == [root]
    live = eb.live_root_resume_grant(q, root, tmp_path)
    assert (live["grant_id"], live["generation"]) == (grant["grant_id"], grant["generation"])
    assert _model_selects(tmp_path, workers, q, chosen, root)["ok"] is True
    selection = rows[chosen][eb.SELECTION_KEY]
    assert (selection["selected_by"], selection["root_grant_id"], selection["root_resume_generation"]) == (
        root, grant["grant_id"], grant["generation"])
    assert eb.budget_hold_fact(rows[chosen]) is None
    assert rows[other] == children[other], "Resume never fans out the other child"
    workers.WORKERS[0].busy_task_id = None
    workers.assign_tasks()
    assert [t["id"] for t in sent] == [root, chosen]
    assert rows[other] in q.PENDING and eb.budget_hold_fact(rows[other])


def test_a_member_hold_bound_to_an_older_root_grant_is_rebound_to_the_unstarted_roots_grant(tmp_path, monkeypatch):
    """A warm Resume binds a member's hold to its grant; an application stop keeps the hold; the root,
    queued again, is paused and resumed. The same new grant reaches that hold and the fence."""
    from supervisor import events_budget as eb
    from supervisor.owner_pause_control import request_owner_pause

    q, workers, rows = _root_ran_children_queued_tree(tmp_path, monkeypatch)
    assert q.resume_budget_paused_task(ROOT)["ok"]
    sent = []
    _idle_worker(workers, sent)
    workers.assign_tasks()
    assert _model_selects(tmp_path, workers, q, CHOSEN, ROOT)["ok"] is True
    assert request_owner_pause(ROOT, request_id="press-1")["ok"]
    workers.RUNNING[ROOT]["owner_wait"] = {"state": "waiting", "reason": "owner_pause"}  # FIXTURE INPUT: warm park
    warm = q.resume_budget_paused_task(ROOT)
    assert warm["ok"] is True and warm["warm"] is True
    workers.WORKERS.clear()
    _stop_and_boot(q, workers)
    rows = {t["id"]: t for t in workers.PENDING}
    assert eb.budget_hold_fact(rows[CHOSEN])["root_grant_id"] == warm["grant_id"]
    paused, basis = _pause(q, tmp_path, ROOT, request_id="press-2")
    _resumed, grant = _resume(q, tmp_path, ROOT, paused, basis)
    hold = eb.budget_hold_fact(rows[CHOSEN])
    assert (hold["root_grant_id"], hold["root_resume_generation"]) == (grant["grant_id"], grant["generation"])
    assert grant["grant_id"] != warm["grant_id"] and "fence_id" not in hold
    sent = []
    _idle_worker(workers, sent)
    workers.assign_tasks()
    assert [t["id"] for t in sent] == [ROOT]
    assert _model_selects(tmp_path, workers, q, CHOSEN, ROOT)["ok"] is True
    assert eb.budget_hold_fact(rows[OTHER])


def _consumed_exact_generation_two(tmp_path, monkeypatch):
    """One exact pause of the root granted twice through the real doors: generation 1 revoked when
    money vanished before its dispatch, generation 2 consumed by the resumed loop."""
    from types import SimpleNamespace

    from ouroboros import budget_pause as bp
    from ouroboros import owner_wait
    from supervisor.budget_resume import revoke_exact_budget_resume
    from tests._budget_pause_exact_helpers import _loop_ctx, _parked

    q, state, workers = _install_queue(tmp_path, monkeypatch)
    money = [5.0]
    monkeypatch.setattr(state, "budget_remaining", lambda *a, **k: money[0])
    task, _row = _parked(tmp_path, monkeypatch, task_id=ROOT)
    first = q.resume_budget_paused_task(ROOT)
    assert first["ok"] and first["grant_generation"] == 1
    money[0] = 0.0
    _idle_worker(workers, [])
    workers.assign_tasks()
    if "_budget_pause_resume" in task:  # the money gate's own revoker, when the assignment kept the grant
        with q._queue_lock:
            revoke_exact_budget_resume(task, "budget_exhausted_before_dispatch")
    money[0] = 5.0
    assert q.resume_budget_paused_task(ROOT)["grant_generation"] == 2
    ctx, _limit = _loop_ctx(tmp_path, ROOT)
    ctx.budget_pause_resume = task["_budget_pause_resume"]
    monkeypatch.setattr(owner_wait, "rebind_restored_route", lambda *a, **k: (None, "max"))
    bp.resume_paused_loop(SimpleNamespace(_ctx=ctx), bp.load_budget_pause(ctx), [], {}, {}, set(),
                          budget_remaining_usd=5.0)
    row = bp.budget_pause_row(tmp_path, ROOT)
    assert row["resume_generation"] == 2 and row["grant"]["consumed_at"] and not row["grant"].get("revoked_at")
    return first, row


def test_consumed_exact_history_neither_lends_nor_masks_the_unstarted_roots_grant(tmp_path, monkeypatch):
    from ouroboros import budget_pause as bp
    from ouroboros import owner_pause
    from supervisor import events_budget as eb

    first, history = _consumed_exact_generation_two(tmp_path, monkeypatch)
    q, workers, _rows = _root_ran_children_queued_tree(tmp_path, monkeypatch)
    assert bp.budget_pause_row(tmp_path, ROOT)["grant"] == history["grant"]
    paused, basis = _pause(q, tmp_path, ROOT)
    _resumed, grant = _resume(q, tmp_path, ROOT, paused, basis)
    budget = grant["root_resume_basis"]["budget"]
    assert (budget["grant_id"], budget["resume_generation"], budget["revoked"]) == (
        history["grant"]["grant_id"], 2, False)
    sent = []
    _idle_worker(workers, sent)
    workers.assign_tasks()
    assert [t["id"] for t in sent] == [ROOT]
    assert eb.live_root_resume_grant(q, ROOT, tmp_path)["grant_id"] == grant["grant_id"] != history["grant"]["grant_id"]
    # Negative control, a FIXTURE INPUT: the same grant recorded over the revoked first generation's
    # budget identity lends nothing, and the model is refused as without a grant; the observed basis lends.
    older = {**budget, "resume_generation": 1, "grant_id": first["grant_id"], "grant_generation": 1, "revoked": True}
    owner_pause.set_fence_state(tmp_path, ROOT, fence_id=paused["fence_id"], state="released",
                                resume_grant={**grant, "root_resume_basis": {**basis, "budget": older}})
    assert eb.live_root_resume_grant(q, ROOT, tmp_path) == {}
    assert _model_selects(tmp_path, workers, q, CHOSEN, ROOT)["error"] == "root_resume_grant_missing"
    owner_pause.set_fence_state(tmp_path, ROOT, fence_id=paused["fence_id"], state="released", resume_grant=grant)
    assert eb.live_root_resume_grant(q, ROOT, tmp_path)["grant_id"] == grant["grant_id"]
    assert _model_selects(tmp_path, workers, q, CHOSEN, ROOT)["ok"] is True


@pytest.mark.parametrize("change", ["new_pause", "unknown_fence", "stop", "panic"])
def test_the_unstarted_roots_grant_yields_to_a_newer_pause_stop_panic_or_unknown_fence(tmp_path, monkeypatch, change):
    from ouroboros import owner_pause
    from ouroboros.cancel_intents import request_cancel
    from supervisor import events_budget as eb
    from supervisor.owner_pause_control import request_owner_pause

    q, workers, rows = _root_ran_children_queued_tree(tmp_path, monkeypatch)
    _resumed, grant = _resume(q, tmp_path, ROOT, *_pause(q, tmp_path, ROOT))
    sent = []
    _idle_worker(workers, sent)
    workers.assign_tasks()
    assert [t["id"] for t in sent] == [ROOT]
    # The owner's own choice binds to the live grant, as on every other path, and is revalidated too.
    assert q.resume_budget_paused_task(OTHER)["ok"]
    assert rows[OTHER][eb.SELECTION_KEY]["root_grant_id"] == grant["grant_id"]
    if change == "new_pause":
        assert request_owner_pause(ROOT, request_id="owner-press-2")["ok"]
    elif change == "unknown_fence":
        monkeypatch.setattr(owner_pause, "read_fence", lambda *a, **k: (_ for _ in ()).throw(OSError("unreadable")))
    elif change == "stop":
        request_cancel(tmp_path, ROOT, reason="owner stopped the tree")
    else:
        (tmp_path / "state" / "panic_stop.flag").write_text("panic", encoding="utf-8")
        workers.WORKERS.clear()
        _stop_and_boot(q, workers)
        rows = {t["id"]: t for t in workers.PENDING}
        assert eb.budget_hold_fact(rows[ROOT])["door"] == "panic", "only the owner resumes the root again"
    held = copy.deepcopy(rows[CHOSEN])
    outcome = _model_selects(tmp_path, workers, q, CHOSEN, ROOT)
    assert outcome["ok"] is False and outcome["error"] == {
        "new_pause": "selection_owner_held", "unknown_fence": "selection_observation_unavailable",
        "stop": "selection_owner_held", "panic": "root_resume_grant_missing"}[change], outcome
    assert rows[CHOSEN] == held
    sent.clear()
    _idle_worker(workers, sent)
    workers.assign_tasks()
    if change == "stop":
        # A single Stop of the root refuses every later selection while it settles; the root still
        # lends, and the choice the owner made before it stays the owner's.
        assert eb.live_root_resume_grant(q, ROOT, tmp_path)["grant_id"] == grant["grant_id"]
        assert [t["id"] for t in sent] == [OTHER]
    else:
        assert eb.live_root_resume_grant(q, ROOT, tmp_path) == {}
        assert sent == []


def test_a_spent_budget_parks_children_selected_under_the_unstarted_roots_grant(tmp_path, monkeypatch):
    from supervisor import events_budget as eb
    from supervisor import state

    q, workers, rows = _root_ran_children_queued_tree(tmp_path, monkeypatch)
    _resume(q, tmp_path, ROOT, *_pause(q, tmp_path, ROOT))
    sent = []
    _idle_worker(workers, sent)
    workers.assign_tasks()
    assert q.resume_budget_paused_task(OTHER)["ok"]
    assert _model_selects(tmp_path, workers, q, CHOSEN, ROOT)["ok"] is True
    money = [0.0]
    monkeypatch.setattr(state, "budget_remaining", lambda *a, **k: money[0])
    for _ in range(2):  # spent, then replenished: neither releases a parked choice by itself
        sent.clear()
        _idle_worker(workers, sent)
        workers.assign_tasks()
        assert sent == []
        money[0] = 5.0
    assert {(rows[tid]["_budget_pause"]["scope"], rows[tid]["_budget_pause"]["status"])
            for tid in (CHOSEN, OTHER)} == {("global", "paused_before_dispatch")}
    assert eb.SELECTION_KEY in rows[CHOSEN] and eb.SELECTION_KEY in rows[OTHER]


@pytest.mark.parametrize("failure", ["snapshot", "fence"])
def test_a_failed_root_resume_write_lends_no_grant_and_a_retry_mints_its_own(tmp_path, monkeypatch, failure):
    from ouroboros import owner_pause
    from supervisor import events_budget as eb

    q, _workers, rows = _root_ran_children_queued_tree(tmp_path, monkeypatch)
    paused, basis = _pause(q, tmp_path, ROOT)
    before = copy.deepcopy(rows)
    real_snapshot, real_fence = q.persist_queue_snapshot, owner_pause.set_fence_state

    def snapshot(*args, **kwargs):
        return False if kwargs.get("reason") == "owner_pause_root_resumed" else real_snapshot(*args, **kwargs)

    def fence(*args, **kwargs):
        if kwargs.get("state") == owner_pause.FENCE_RELEASED:
            raise OSError("fence write failed")
        return real_fence(*args, **kwargs)

    with monkeypatch.context() as fault:
        if failure == "snapshot":
            fault.setattr(q, "persist_queue_snapshot", snapshot)
        else:
            fault.setattr(owner_pause, "set_fence_state", fence)
        refused = q.resume_budget_paused_task(ROOT)
    assert refused["ok"] is False and refused["error"] == {
        "snapshot": "snapshot_not_persisted", "fence": "owner_pause_fence_unwritable"}[failure], refused
    assert owner_pause.read_fence(tmp_path, ROOT) == paused, "the fence stays closed, with no grant"
    assert rows == before and q.BUDGET_ROOT_FENCES[ROOT]["fence_id"] == paused["fence_id"]
    assert eb.live_root_resume_grant(q, ROOT, tmp_path) == {}
    _resumed, grant = _resume(q, tmp_path, ROOT, paused, basis)
    assert eb.budget_hold_fact(rows[CHOSEN]) and eb.budget_hold_fact(rows[OTHER])


# A saved source the aborted update could not freeze: the owner's Resume keeps the refusal it already observed.

def _update_aborted_tree(tmp_path, monkeypatch, *, freeze_fault):
    """Real doors: the root is dispatched by the real assignment and saves a round, queues two children
    (the real ``enqueue_task``), then an update stops the pool and aborts (``recover_aborted_update``);
    with ``freeze_fault`` its freeze of the root's saved work fails with ENOSPC. Processes are doubles."""
    import errno
    import os

    from ouroboros import artifacts
    from ouroboros import working_checkpoint as wc
    from ouroboros.task_results import STATUS_RUNNING, STATUS_SCHEDULED, write_task_result
    from tests._budget_pause_exact_helpers import _loop_ctx
    from tests.test_restart_retention import _queued
    from tests.test_update_abort_retention import _system

    system = _system(tmp_path, monkeypatch)
    q, workers = system.queue, system.workers
    monkeypatch.setattr(system.state, "budget_remaining", lambda *a, **k: 5.0)
    assert workers.ensure_worker_pool_started()
    write_task_result(tmp_path, ROOT, STATUS_SCHEDULED, chat_id=0, root_task_id=ROOT)
    assert not q.enqueue_task(_queued(ROOT, root_task_id=ROOT)).get("_admission_blocked")
    workers.assign_tasks()
    assert _sent(system) == [ROOT]
    for tid in (CHOSEN, OTHER):
        lineage = {"root_task_id": ROOT, "parent_task_id": ROOT}
        write_task_result(tmp_path, tid, STATUS_SCHEDULED, chat_id=0, delegation_role="subagent", **lineage)
        assert not q.enqueue_task(_queued(tid, delegation_role="subagent", **lineage)).get("_admission_blocked")
    write_task_result(tmp_path, ROOT, STATUS_RUNNING, chat_id=0, root_task_id=ROOT)
    _ctx, limit = _loop_ctx(tmp_path, ROOT, attempt=1)
    wc.save_round(limit, "pre_effect")
    assert system.control._quiesce_repo_writers("auto_merge") == []
    with monkeypatch.context() as fault:
        if freeze_fault:
            def full(*_a, **_k):
                raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))
            fault.setattr(artifacts, "store_actor_source_bytes", full)
        system.control._respawn_workers_after_failed_update()
    rows = {row["id"]: row for row in workers.PENDING}
    assert set(rows) == {ROOT, CHOSEN, OTHER} and ROOT not in workers.RUNNING
    assert (rows[ROOT]["_attempt"], rows[ROOT]["admitted_dispatch"]) == (2, "possible")
    # The producer's own output: a frozen source returns with the update; a failed freeze keeps the bare locator held.
    from supervisor.events_budget import budget_hold_fact

    assert ("source_ref" in rows[ROOT]["_working_recovery"], (budget_hold_fact(rows[ROOT]) or {}).get("reason")) == (
        (False, "saved_work_hold") if freeze_fault else (True, None))
    return system, rows


def _sent(system):
    return [commands.get_nowait()["id"] for commands in system.launches for _ in range(commands.qsize())]


def _offered(tmp_path, *ids):
    """What the owner is offered, read by the real census as Main reads it (children included): each row's own
    Resume and Main's attention rule."""
    from ouroboros.gateway.main_decisions import owner_actions
    from ouroboros.gateway.state import _chat_activities_snapshot_safe

    availability, owners = {}, []
    _chat_activities_snapshot_safe(tmp_path, direct_turns=[], availability=availability, decision_owners=owners)
    assert availability.get("complete") is not False, availability
    census = {row["activity_id"]: row for row in owners}
    return {tid: (census[tid].get("can_resume") is True, owner_actions(census[tid])[1]) for tid in ids}


@pytest.mark.parametrize("freeze", ["frozen", "enospc"])
def test_an_unusable_saved_source_keeps_its_refusal_through_the_owners_resume(tmp_path, monkeypatch, freeze):
    """The Resume decides on its one observation of the root's saved source: usable, the root lends one grant
    and its model selects one child; unusable (the update's ENOSPC), the same transition keeps that typed
    refusal, so the root's own Resume and Main ask instead of children hidden behind a root that never runs."""
    from typing import get_type_hints

    from ouroboros import working_checkpoint as wc
    from ouroboros.gateway.client_contracts import TaskResumeResponse
    from ouroboros.work_stop import continuation_refusal
    from supervisor import events_budget as eb
    from tests.test_resume_attempt_history import _events

    system, rows = _update_aborted_tree(tmp_path, monkeypatch, freeze_fault=freeze == "enospc")
    q, workers = system.queue, system.workers
    paused, basis = _pause(q, tmp_path, ROOT)
    reads, real_reader = [], wc.recovery_source_for_task
    with monkeypatch.context() as count:
        count.setattr(wc, "recovery_source_for_task", lambda root, task: reads.append(task.get("id")) or real_reader(
            root, task))
        resumed, grant = _resume(q, tmp_path, ROOT, paused, basis)
    assert reads == [ROOT], "the one source observation; the refusal reads nothing more"
    # ok=True is the released Pause, not a dispatch. The answer keeps its wire shape: only this path's two older
    # undeclared keys, no new one; the refusal is the row's, and the append-only event keeps that it was kept here.
    assert set(resumed) - set(get_type_hints(TaskResumeResponse)) == {"same_generation", "never_started"}, resumed
    [event] = _events(tmp_path, "owner_pause_resumed")
    children = copy.deepcopy({tid: rows[tid] for tid in (CHOSEN, OTHER)})
    for _ in range(3):
        workers.assign_tasks()
    if freeze == "frozen":
        assert "continuation_refused" not in event and not continuation_refusal(rows[ROOT])
        assert _sent(system) == [ROOT]
        assert eb.live_root_resume_grant(q, ROOT, tmp_path)["grant_id"] == grant["grant_id"]
        assert _offered(tmp_path, CHOSEN, OTHER) == {CHOSEN: (False, False), OTHER: (False, False)}
        assert _model_selects(tmp_path, workers, q, CHOSEN, ROOT)["ok"] is True
        assert rows[CHOSEN][eb.SELECTION_KEY]["root_grant_id"] == grant["grant_id"]
        workers.WORKERS[workers.RUNNING[ROOT]["worker_id"]].busy_task_id = None
        workers.assign_tasks()
        assert _sent(system) == [CHOSEN]
        assert rows[OTHER] == children[OTHER] and eb.budget_hold_fact(rows[OTHER])
        return
    # (the Resume's event, the root's refusal, its lending, (own Resume, Main attention) per row, dispatched)
    offered = (event.get("continuation_refused"), continuation_refusal(rows[ROOT]).get("reason"),
               bool(eb.live_root_resume_grant(q, ROOT, tmp_path)), _offered(tmp_path, ROOT, CHOSEN, OTHER),
               _sent(system))
    assert offered == ("dispatch_outcome_unknown", "dispatch_outcome_unknown", False,
                       {ROOT: (True, True), CHOSEN: (False, False), OTHER: (False, False)}, []), offered
    assert continuation_refusal(rows[ROOT])["dispatch_unknown"] is True
    assert _model_selects(tmp_path, workers, q, CHOSEN, ROOT)["error"] == "root_resume_grant_missing"
    held = copy.deepcopy(rows[ROOT])
    again = q.resume_budget_paused_task(ROOT)
    assert (again["ok"], again["error"], again["action"]) == (False, "dispatch_outcome_unknown", "cancel_or_new_run")
    assert rows[ROOT] == held, "this refusal authorizes no replay of the unusable source"
    # The owner's own choice of a child is still dispatched; the other child stays held.
    assert q.resume_budget_paused_task(CHOSEN)["ok"] is True
    workers.assign_tasks()
    assert _sent(system) == [CHOSEN]
    assert rows[OTHER] == children[OTHER] and eb.budget_hold_fact(rows[OTHER]) and rows[ROOT] in q.PENDING


@pytest.mark.parametrize("failure", ["snapshot", "fence"])
def test_a_failed_resume_write_keeps_no_refusal_and_a_retry_records_it(tmp_path, monkeypatch, failure):
    from ouroboros import owner_pause
    from ouroboros.work_stop import continuation_refusal
    from tests.test_resume_attempt_history import _events

    system, rows = _update_aborted_tree(tmp_path, monkeypatch, freeze_fault=True)
    q = system.queue
    paused, basis = _pause(q, tmp_path, ROOT)
    before = copy.deepcopy(rows)
    real_snapshot, real_fence = q.persist_queue_snapshot, owner_pause.set_fence_state

    def snapshot(*args, **kwargs):
        return False if kwargs.get("reason") == "owner_pause_root_resumed" else real_snapshot(*args, **kwargs)

    def fence(*args, **kwargs):
        if kwargs.get("state") == owner_pause.FENCE_RELEASED:
            raise OSError("fence write failed")
        return real_fence(*args, **kwargs)

    with monkeypatch.context() as fault:
        fault.setattr(*((q, "persist_queue_snapshot", snapshot) if failure == "snapshot"
                        else (owner_pause, "set_fence_state", fence)))
        refused = q.resume_budget_paused_task(ROOT)
    assert refused["ok"] is False and refused["error"] == {
        "snapshot": "snapshot_not_persisted", "fence": "owner_pause_fence_unwritable"}[failure], refused
    assert owner_pause.read_fence(tmp_path, ROOT) == paused and rows == before
    assert not continuation_refusal(rows[ROOT]), "the refusal stands only with its published Resume"
    assert not _events(tmp_path, "owner_pause_resumed"), "nor does its record"
    resumed, _grant = _resume(q, tmp_path, ROOT, paused, basis)
    assert ("continuation_refused" in resumed, continuation_refusal(rows[ROOT]).get("reason"),
            [e.get("continuation_refused") for e in _events(tmp_path, "owner_pause_resumed")]) == (
        False, "dispatch_outcome_unknown", ["dispatch_outcome_unknown"]), resumed


def test_a_refused_continuation_is_never_returned_to_a_writer_wait(tmp_path, monkeypatch):
    """At the function seam, on the real observation; its predecessor's running writer is a FIXTURE INPUT. A root
    the queue can never hand over keeps its refusal, not a wait promising a start, as a held row's own Resume
    orders them (``select_held_budget_row``)."""
    from ouroboros.work_stop import continuation_refusal, writer_wait
    from supervisor import events_budget as eb
    from supervisor.budget_resume_unstarted import _resume_unstarted_owner_paused_root
    from tests.test_continue_blocker_diagnostics import RUNNING

    system, rows = _update_aborted_tree(tmp_path, monkeypatch, freeze_fault=True)
    q = system.queue
    _pause(q, tmp_path, ROOT)
    observation = eb.observe_held_budget_selection(q, ROOT)
    assert (observation["safe"], observation["unsafe_error"]) == (False, "dispatch_outcome_unknown")
    rows[ROOT]["metadata"] = {**(rows[ROOT].get("metadata") or {}), "continuation": {"predecessor_task_id": "earlier"}}
    with q._queue_lock:
        outcome = _resume_unstarted_owner_paused_root(q, rows[ROOT], q.BUDGET_ROOT_FENCES[ROOT], {
            **observation, "blockers": [{**RUNNING, "task_id": "earlier"}], "owner_pause_blockers": []})
    assert (outcome["ok"], "continuation_refused" in outcome, outcome.get("held"),
            continuation_refusal(rows[ROOT]).get("reason")) == (True, False, None, "dispatch_outcome_unknown"), outcome
    assert not writer_wait(rows[ROOT])
