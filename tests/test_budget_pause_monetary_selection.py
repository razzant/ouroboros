"""An explicitly Resumed child of a monetary root latch reaches its sends; nobody else does.

The owner's Resume of ONE budget-paused child (or the model's, under a live
root grant) is a selection recorded against the root's CURRENT latch
generation. Assignment already admits it (``budget_resume_dispatch_allowed``);
the money gate in ``reserve_attempt`` must read the same selection through the
one queue predicate (``budget_pause.budget_fence_selected``) — for the first
send and every later one of the resumed worker — while unselected siblings and
the unselected root stay refused, the latch stays up, and the real global and
root caps still bind. Every scenario uses real ``ToolContext`` lineage, the
real queue, park, grant, assignment, consumption and reservation seams; only
cognition restore, pricing and the loop-side custody observation are stubbed.

Generations, characterized here because they decide what a selection is worth:

- the unselected root parking under its child's latch F (its own next send is
  refused by F) MATERIALIZES F: the latch keeps its id and the owner's child
  selection keeps working, pending or running;
- a NEW root pause after the root's own Resume lifted F raises a new latch:
  a pending child grant is revoked and a running child's older selection no
  longer admits a send.

A selected child that cold-sleeps beneath F and is Resumed by the owner starts
and keeps its selection when consumption (event, repair tick or a warm park)
retires only the sleep timing; its own readiness never starts or spends under F.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from tests.test_budget_pause_lineage_restore import (
    CHILD,
    ROOT,
    SIBLING,
    _complete_root,
    _consume,
    _install,
    _member_ctx,
    _pending,
    _planning_park,
    _record,
    _sup,
    _tree,
    _workers,
)

pytestmark = pytest.mark.serial


def _reserve(tmp_path, task_id, *, request=None, **limits):
    """One real reservation under the task's tree: its attempt id, else the refusal's scope and text."""
    from ouroboros.usage_accounting import AttemptRequest, BudgetExceeded, UsageScope, reserve_attempt, usage_scope

    try:
        with usage_scope(UsageScope(drive_root=tmp_path, task_id=task_id, root_task_id=ROOT, **limits)):
            return reserve_attempt(AttemptRequest(model="m", provider="p", task_id=task_id,
                                                  **(request or {}))).attempt_id
    except BudgetExceeded as exc:
        return (exc.limit_scope, str(exc))


def _settle_known(tmp_path, task_id, cost):
    """A send of the task's tree settled at a final price: KNOWN spend (#1487)."""
    from ouroboros import usage_accounting as ua

    with ua.usage_scope(ua.UsageScope(drive_root=tmp_path, task_id=task_id, root_task_id=ROOT)):
        held = ua.reserve_attempt(ua.AttemptRequest(model="m", provider="p", task_id=task_id, reservation_usd=cost))
        ua.mark_dispatched(held)
        ua.settle_attempt(held, {}, cost_usd=cost, cost_final=True)


def _fixed_price(monkeypatch):
    monkeypatch.setattr("ouroboros.usage_accounting._reservation_cost", lambda _r: 0.01)


FENCED = ("root", f"root model dispatch paused pending explicit resume for {ROOT}")


def _refused_dispatch_park(tmp_path, limit_ctx, task_id):
    """The task's next send meets the latch and parks on the dispatch rail exactly as
    ``loop_budget`` calls it: the real refusal's scope and root name the pause."""
    from ouroboros import budget_pause
    from ouroboros.usage_accounting import BudgetExceeded

    refused = _reserve(tmp_path, task_id)
    assert refused == FENCED, refused
    exc = BudgetExceeded(refused[1], limit_scope=refused[0], root_task_id=ROOT)
    with pytest.raises(budget_pause.BudgetPauseRequested) as raised:
        budget_pause.request_pause(limit_ctx, rail=budget_pause.RAIL_DISPATCH_REFUSED, scope=exc.limit_scope,
                                   reason_text=str(exc), root_task_id=exc.root_task_id)
    budget_pause.end_dispatch_fence(str(task_id))
    return raised.value.pause


def _snapshot_task(tmp_path, bucket, task_id):
    snap = json.loads((tmp_path / "state" / "queue_snapshot.json").read_text(encoding="utf-8"))
    return next((entry["task"] for entry in snap.get(bucket) or [] if entry["task"]["id"] == task_id), None)


def _child_under_latch(tmp_path, monkeypatch, *, sibling=False):
    """A child's planning pause latches the running root (monetary F, no owner cause)."""
    queue, _state, workers = _tree(tmp_path, monkeypatch, sibling=sibling)
    sup = _sup(tmp_path, queue, workers)
    child_ctx, child_limit = _member_ctx(tmp_path)
    _install(sup, CHILD, _planning_park(child_limit))
    fence_f = queue.BUDGET_ROOT_FENCES[ROOT]["fence_id"]
    assert "cause" not in queue.BUDGET_ROOT_FENCES[ROOT], "a monetary latch"
    _fixed_price(monkeypatch)
    return queue, workers, sup, (child_ctx, child_limit), fence_f


def _run_child(monkeypatch, queue, workers, sent, child_ctx, child_limit):
    workers.assign_tasks()
    assert [task["id"] for task in sent] == [CHILD]
    _consume(monkeypatch, child_ctx, child_limit, sent[0]["_budget_pause_resume"])
    sent.clear()


# --- the first and every later send of the selected child ---------------------------------

@pytest.mark.parametrize("root_state", ["running", "terminal"])
def test_an_owner_resumed_child_reserves_under_the_latch_while_the_rest_of_the_tree_stays_refused(
        tmp_path, monkeypatch, root_state):
    from ouroboros import budget_pause

    queue, workers, _sup_ctx, (child_ctx, child_limit), fence_f = _child_under_latch(
        tmp_path, monkeypatch, sibling=True)
    if root_state == "terminal":
        _complete_root(tmp_path, workers)
    granted = queue.resume_budget_paused_task(CHILD)
    assert granted["ok"] is True, granted
    handoff = _pending(workers, CHILD)["_budget_pause_resume"]
    assert (handoff["selected_by"], handoff["root_grant_id"], handoff["root_fence_id"]) == ("owner", "", fence_f)
    # Publication lag: the worker may send before the assignment snapshot names it RUNNING.
    lagging = _reserve(tmp_path, CHILD)
    sent = _workers(workers)
    _run_child(monkeypatch, queue, workers, sent, child_ctx, child_limit)
    assert budget_pause.budget_pause_row(tmp_path, CHILD)["state"] == budget_pause.STATE_RESUMED
    assert queue.persist_queue_snapshot(reason="test_after_consumption")
    running = _snapshot_task(tmp_path, "running", CHILD)
    outcome = {"lagging": lagging, "first": _reserve(tmp_path, CHILD), "later": _reserve(tmp_path, CHILD),
               "sibling": _reserve(tmp_path, SIBLING),
               "running_projection": {key: (running.get("_budget_pause_resume") or {}).get(key)
                                      for key in ("root_fence_id", "selected_by", "authority")},
               "running_marker": running.get("_budget_pause"), "latch": queue.BUDGET_ROOT_FENCES.get(ROOT)}
    if root_state == "running":
        outcome["root"] = _reserve(tmp_path, ROOT)
    _record(f"selected_child_sends[{root_state}]", outcome)
    for key in ("lagging", "first", "later"):
        assert isinstance(outcome[key], str) and outcome[key], outcome
    assert outcome["running_projection"] == {"root_fence_id": fence_f, "selected_by": "owner",
                                             "authority": "explicit_resume"}, outcome
    assert outcome["running_marker"] is None, outcome
    assert outcome["sibling"] == FENCED and outcome["latch"]["fence_id"] == fence_f, outcome
    if root_state == "running":
        assert outcome["root"] == FENCED, ("the unselected root stays fenced", outcome)
    assert [task["id"] for task in sent] == [], "the latch still holds the unselected sibling"
    workers.assign_tasks()
    assert [task["id"] for task in sent] == []


def test_the_real_caps_still_bind_a_selected_child_and_its_clocks_carry_the_pause(tmp_path, monkeypatch):
    """Selection opens the latch for one member, never money: the root and global caps
    refuse it at the same gate. The resumed row keeps its original start and carries
    the paused interval separately."""
    from ouroboros import budget_pause

    queue, workers, _sup_ctx, (child_ctx, child_limit), _fence_f = _child_under_latch(tmp_path, monkeypatch)
    started = budget_pause.budget_pause_row(tmp_path, CHILD)["started_at"]
    assert queue.resume_budget_paused_task(CHILD)["ok"]
    sent = _workers(workers)
    workers.assign_tasks()
    meta = workers.RUNNING[CHILD]
    _consume(monkeypatch, child_ctx, child_limit, sent[0]["_budget_pause_resume"])
    grant = budget_pause.budget_pause_row(tmp_path, CHILD)["grant"]
    _settle_known(tmp_path, CHILD, 0.01)  # the caps below decide on this known spend, not on holds
    root_cap = _reserve(tmp_path, CHILD, root_limit_usd=0.005)
    global_cap = _reserve(tmp_path, CHILD, request={"global_limit_usd": 0.005})
    _record("selected_child_caps", {"root_cap": root_cap, "global_cap": global_cap,
                                    "started_at": [started, meta["started_at"]],
                                    "budget_paused_sec": [grant["paused_duration_sec"], meta["budget_paused_sec"]]})
    assert root_cap[0] == "root" and "root model budget exhausted" in root_cap[1], root_cap
    assert global_cap[0] == "global" and "global model budget exhausted" in global_cap[1], global_cap
    assert meta["started_at"] == pytest.approx(started)
    assert meta["budget_paused_sec"] == pytest.approx(grant["paused_duration_sec"]) and meta["budget_paused_sec"] > 0


def test_stop_and_unpublished_or_unknown_selection_authority_admit_nothing(tmp_path, monkeypatch):
    """A Stop refuses the Resume itself; a grant whose snapshot is not published rolls
    back and selects nobody; an unreadable latch authority refuses typed."""
    from ouroboros.cancel_intents import request_cancel
    from ouroboros.usage_accounting import UsageAccountingError

    queue, workers, _sup_ctx, _child, _fence_f = _child_under_latch(tmp_path, monkeypatch)
    real_persist = queue.persist_queue_snapshot
    monkeypatch.setattr(queue, "persist_queue_snapshot",
                        lambda reason="": False if reason == "budget_exact_resume_granted" else real_persist(reason))
    unpublished = queue.resume_budget_paused_task(CHILD)
    monkeypatch.setattr(queue, "persist_queue_snapshot", real_persist)
    assert unpublished["ok"] is False and unpublished["error"] == "snapshot_not_persisted", unpublished
    child = _pending(workers, CHILD)
    assert "_budget_pause_resume" not in child and child["_budget_pause"]["exact_continuation"]
    assert _reserve(tmp_path, CHILD) == FENCED

    request_cancel(tmp_path, CHILD, reason="owner_stop", source="test")
    stopped = queue.resume_budget_paused_task(CHILD)
    assert stopped == {"ok": False, "error": "cancel_intent_active"}, stopped
    assert _reserve(tmp_path, CHILD) == FENCED

    (tmp_path / "state" / "queue_snapshot.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(UsageAccountingError, match="root budget fence authority unavailable"):
        _reserve(tmp_path, CHILD)


# --- which root pause is a new generation -----------------------------------------------

@pytest.mark.parametrize("child_state", ["pending_dispatch", "running"])
def test_the_unselected_root_parking_under_the_childs_latch_keeps_the_owner_selection(
        tmp_path, monkeypatch, child_state):
    """The ordinary next event: F refuses the running root's next send and the root parks
    on the dispatch rail. That materializes F (same id, nothing revoked): the selected
    child continues within its limits, whether it was still queued or already running."""
    from ouroboros import budget_pause
    from supervisor.events_budget import budget_resume_dispatch_allowed

    queue, workers, sup, (child_ctx, child_limit), fence_f = _child_under_latch(tmp_path, monkeypatch)
    assert queue.resume_budget_paused_task(CHILD)["ok"]
    sent = _workers(workers)
    evidence = {"fence_f": fence_f}
    if child_state == "running":
        _run_child(monkeypatch, queue, workers, sent, child_ctx, child_limit)
        evidence["before_root_park"] = _reserve(tmp_path, CHILD)
    root_ctx, root_limit = _member_ctx(tmp_path, ROOT, ROOT)
    _install(sup, ROOT, _refused_dispatch_park(tmp_path, root_limit, ROOT))
    evidence["latch_after_root_park"] = dict(queue.BUDGET_ROOT_FENCES[ROOT])
    evidence["root_marker_fence"] = _pending(workers, ROOT)["_budget_pause"].get("fence_id")
    if child_state == "pending_dispatch":
        child = _pending(workers, CHILD)
        evidence["dispatch_allowed"] = budget_resume_dispatch_allowed(queue, child)
        _run_child(monkeypatch, queue, workers, sent, child_ctx, child_limit)
        evidence["child_state"] = budget_pause.budget_pause_row(tmp_path, CHILD)["state"]
    evidence["after_root_park"] = [_reserve(tmp_path, CHILD), _reserve(tmp_path, CHILD)]
    _record(f"root_parks_under_child_latch[{child_state}]", evidence)
    assert evidence["latch_after_root_park"]["fence_id"] == fence_f == evidence["root_marker_fence"], evidence
    for attempt in [evidence.get("before_root_park"), *evidence["after_root_park"]]:
        assert attempt is None or isinstance(attempt, str), evidence
    if child_state == "pending_dispatch":
        assert evidence["dispatch_allowed"] is True and evidence["child_state"] == budget_pause.STATE_RESUMED, evidence


@pytest.mark.parametrize("child_state", ["pending_dispatch", "running"])
def test_a_new_root_pause_after_its_own_resume_spends_the_childs_older_selection(
        tmp_path, monkeypatch, child_state):
    """The root's own Resume lifts F; when the root later pauses again it raises a NEW
    latch G, and a selection made before G admits neither a dispatch nor a send."""
    from ouroboros import budget_pause

    queue, workers, sup, (child_ctx, child_limit), fence_f = _child_under_latch(tmp_path, monkeypatch)
    sent = _workers(workers)
    root_ctx, root_limit = _member_ctx(tmp_path, ROOT, ROOT)
    if child_state == "running":
        assert queue.resume_budget_paused_task(CHILD)["ok"]
        _run_child(monkeypatch, queue, workers, sent, child_ctx, child_limit)
    _install(sup, ROOT, _refused_dispatch_park(tmp_path, root_limit, ROOT))
    assert queue.resume_budget_paused_task(ROOT)["ok"]
    assert ROOT not in queue.BUDGET_ROOT_FENCES, "only the root's own Resume lifts F"
    workers.assign_tasks()
    assert [task["id"] for task in sent] == [ROOT]
    _consume(monkeypatch, root_ctx, root_limit, sent[0]["_budget_pause_resume"])
    sent.clear()
    if child_state == "pending_dispatch":
        granted = queue.resume_budget_paused_task(CHILD)
        assert granted["ok"] is True, granted
    evidence = {"fence_f": fence_f, "before_new_pause": _reserve(tmp_path, CHILD)}
    _install(sup, ROOT, _planning_park(root_limit))
    evidence["latch_g"] = dict(queue.BUDGET_ROOT_FENCES[ROOT])
    if child_state == "pending_dispatch":
        workers.assign_tasks()
        evidence["dispatched"] = [task["id"] for task in sent]
        row = budget_pause.budget_pause_row(tmp_path, CHILD)
        evidence["child_row"] = {"state": row["state"], "revoke_reason": row["grant"].get("revoke_reason")}
    evidence["after_new_pause"] = _reserve(tmp_path, CHILD)
    _record(f"new_root_pause[{child_state}]", evidence)
    assert evidence["latch_g"]["fence_id"] != fence_f, evidence
    assert isinstance(evidence["before_new_pause"], str), evidence
    assert evidence["after_new_pause"] == FENCED, evidence
    if child_state == "pending_dispatch":
        assert evidence["dispatched"] == [], evidence
        assert evidence["child_row"] == {"state": budget_pause.STATE_PAUSED,
                                         "revoke_reason": "new_root_budget_fence"}, evidence


@pytest.mark.parametrize("child_state", ["pending_dispatch", "running"])
def test_a_model_selection_under_a_legacy_root_selection_reserves_until_the_root_pauses_again(
        tmp_path, monkeypatch, child_state):
    """A legacy zero-dispatch root Resume keeps F and selects only the root; the root's
    model then selects one exact child under that root grant. The child's sends are
    admitted against F; once the selected root pauses again (a new generation) they are not."""
    from ouroboros import budget_pause
    from supervisor.events_budget import budget_resume_dispatch_allowed

    queue, workers, sup, (child_ctx, child_limit), fence_f = _child_under_latch(
        tmp_path, monkeypatch, sibling=True)
    root_task = workers.RUNNING.pop(ROOT)["task"]
    workers.PENDING.append({**root_task, "_attempt": 1, "admitted_dispatch": "none"})
    assert queue.resume_budget_paused_task(ROOT)["ok"]  # the legacy selection of the root alone
    assert queue.BUDGET_ROOT_FENCES[ROOT]["fence_id"] == fence_f
    sent = _workers(workers)
    workers.assign_tasks()
    assert [task["id"] for task in sent] == [ROOT]
    sent.clear()
    evidence = {"root_selected_send": _reserve(tmp_path, ROOT), "fence_f": fence_f}
    selected = queue.resume_budget_paused_task(CHILD, selected_by=ROOT)
    assert selected["ok"] is True, selected
    handoff = _pending(workers, CHILD)["_budget_pause_resume"]
    assert handoff["root_grant_id"] and handoff["root_resume_generation"] == 1
    assert handoff["root_fence_id"] == fence_f and budget_resume_dispatch_allowed(queue, _pending(workers, CHILD))
    if child_state == "running":
        _run_child(monkeypatch, queue, workers, sent, child_ctx, child_limit)
    # Also check the common money gate for a lateral review reservation. This is
    # consumer evidence with a different category/source, not a provider transport.
    evidence.update(first=_reserve(tmp_path, CHILD), later=_reserve(tmp_path, CHILD),
                    lateral=_reserve(tmp_path, CHILD, request={"category": "review", "source": "review_synthesis"}),
                    sibling_before=_reserve(tmp_path, SIBLING))
    root_ctx, root_limit = _member_ctx(tmp_path, ROOT, ROOT)
    _install(sup, ROOT, _planning_park(root_limit))
    evidence.update(latch_g=dict(queue.BUDGET_ROOT_FENCES[ROOT]),
                    root_marker_fence=_pending(workers, ROOT)["_budget_pause"]["fence_id"],
                    refused=_reserve(tmp_path, CHILD), sibling_after=_reserve(tmp_path, SIBLING))
    workers.assign_tasks()
    evidence["dispatched"] = [task["id"] for task in sent]
    row = budget_pause.budget_pause_row(tmp_path, CHILD)
    evidence["child_row"] = {"state": row["state"], "revoke_reason": row["grant"].get("revoke_reason")}
    _record(f"legacy_root_selection[{child_state}]", evidence)
    for key in ("root_selected_send", "first", "later", "lateral"):
        assert isinstance(evidence[key], str) and evidence[key], evidence
    assert evidence["root_marker_fence"] == evidence["latch_g"]["fence_id"] != fence_f, evidence
    assert evidence["refused"] == evidence["sibling_before"] == evidence["sibling_after"] == FENCED, evidence
    assert evidence["dispatched"] == [], evidence
    if child_state == "pending_dispatch":
        assert evidence["child_row"] == {"state": budget_pause.STATE_PAUSED,
                                         "revoke_reason": "new_root_budget_fence"}, evidence
    else:
        assert evidence["child_row"]["state"] == budget_pause.STATE_RESUMED, evidence


def test_a_selected_child_that_pauses_again_needs_a_new_resume(tmp_path, monkeypatch):
    """The resumed child parks again under the SAME latch F (its next send refused by the
    root cap). Its queue row keeps the spent handoff beside the new pause marker: the
    marker decides, so nothing is admitted until an explicit new Resume."""
    from ouroboros import budget_pause

    queue, workers, sup, (child_ctx, child_limit), fence_f = _child_under_latch(tmp_path, monkeypatch)
    assert queue.resume_budget_paused_task(CHILD)["ok"]
    sent = _workers(workers)
    _run_child(monkeypatch, queue, workers, sent, child_ctx, child_limit)
    assert isinstance(_reserve(tmp_path, CHILD), str)
    _install(sup, CHILD, _planning_park(child_limit))
    child = _pending(workers, CHILD)
    evidence = {"latch": queue.BUDGET_ROOT_FENCES[ROOT]["fence_id"], "fence_f": fence_f,
                "kept_spent_handoff": child.get("_budget_pause_resume", {}).get("root_fence_id"),
                "refused": _reserve(tmp_path, CHILD)}
    regrant = queue.resume_budget_paused_task(CHILD)
    _run_child(monkeypatch, queue, workers, sent, child_ctx, child_limit)
    evidence.update(regrant=regrant.get("grant_generation"), readmitted=_reserve(tmp_path, CHILD),
                    pause_generation=budget_pause.budget_pause_row(tmp_path, CHILD)["pause_generation"])
    _record("selected_child_pauses_again", evidence)
    assert evidence["latch"] == fence_f == evidence["kept_spent_handoff"], evidence
    assert evidence["refused"] == FENCED, evidence
    assert evidence["regrant"] == 1 and evidence["pause_generation"] == 2, evidence
    assert isinstance(evidence["readmitted"], str), evidence


# --- a cold sleep beneath the latch: consumption retires the sleep clock, not the selection --

def _cold_sleep_park(monkeypatch, limit_ctx, **selected):
    """The member's own cold sleep at its round boundary (``enter_cold_sleep``, scope task)."""
    from ouroboros import budget_pause, model_sleep

    ctx = limit_ctx.tools._ctx
    monkeypatch.setattr(model_sleep, "cold_blockers", lambda _ctx, **_kw: [])
    ctx._model_sleep = {"sleep_id": "s1", "mode": "cold", **model_sleep.selectors(ctx, **selected)}
    with pytest.raises(budget_pause.BudgetPauseRequested) as raised:
        budget_pause.enter_cold_sleep(limit_ctx)
    budget_pause.end_dispatch_fence(str(ctx.task_id))
    return raised.value.pause


def _sleeper_under_latch(tmp_path, monkeypatch, **selected):
    """The owner-selected child runs under F, then cold-sleeps beneath the same latch."""
    queue, workers, sup, (child_ctx, child_limit), fence_f = _child_under_latch(tmp_path, monkeypatch, sibling=True)
    assert queue.resume_budget_paused_task(CHILD)["ok"]
    sent = _workers(workers)
    _run_child(monkeypatch, queue, workers, sent, child_ctx, child_limit)
    _install(sup, CHILD, _cold_sleep_park(monkeypatch, child_limit, **selected))
    assert _pending(workers, CHILD)["_budget_pause"]["reason"] == "sleep"
    assert queue.BUDGET_ROOT_FENCES[ROOT]["fence_id"] == fence_f
    return queue, workers, sup, (child_ctx, child_limit), fence_f, sent


def _wake_and_consume(monkeypatch, workers, sent, child_ctx, child_limit):
    """Assign the granted sleeper and let its worker consume; return the consumption event."""
    workers.assign_tasks()
    assert [task["id"] for task in sent] == [CHILD]
    events = []
    child_ctx.event_queue = SimpleNamespace(put=events.append)
    _consume(monkeypatch, child_ctx, child_limit, sent[0]["_budget_pause_resume"])
    sent.clear()
    return next(event for event in events if event.get("phase") == "consumed")


@pytest.mark.parametrize("door", ["consumed_event", "repair_tick"])
def test_an_owner_resumed_cold_sleeper_keeps_its_selection_once_consumption_retires_the_sleep_clock(
        tmp_path, monkeypatch, door):
    """The owner's Resume of the sleeping child is a selection against F. Consumption — its
    event, or the assignment tick repairing a lost one — folds the sleep interval and spends
    only the sleep timing: after the snapshot publishes, the child's later sends still reserve
    under F, the rest of the tree stays refused, and stale or repeated notifications fold nothing."""
    from ouroboros import budget_pause
    from supervisor.events_budget import _handle_budget_pause
    from supervisor.worker_assignment import _tick_parked_work

    queue, workers, sup, (child_ctx, child_limit), fence_f, sent = _sleeper_under_latch(
        tmp_path, monkeypatch, wake_after_sec=3600)
    granted = queue.resume_budget_paused_task(CHILD)
    assert granted["ok"] is True, granted
    handoff = _pending(workers, CHILD)["_budget_pause_resume"]
    assert (handoff["authority"], handoff["selected_by"], handoff["root_fence_id"]) == (
        "explicit_resume", "owner", fence_f) and handoff["sleep_exclusion_since"], handoff
    consumed = _wake_and_consume(monkeypatch, workers, sent, child_ctx, child_limit)
    meta = workers.RUNNING[CHILD]
    evidence = {"fence_f": fence_f, "before_consumption": _reserve(tmp_path, CHILD)}
    for wrong in ({"pause_id": "other"}, {"grant_id": "other"}, {"task_attempt": 99}):
        _handle_budget_pause({**consumed, **wrong}, sup)
    evidence["stale_events_kept_clock"] = bool(
        meta.get("sleep_parked_at") and meta["task"]["_budget_pause_resume"].get("sleep_exclusion_since"))
    if door == "consumed_event":
        _handle_budget_pause(consumed, sup)
    else:
        _tick_parked_work(queue)
    folded = meta.get("budget_paused_sec")
    _handle_budget_pause(consumed, sup)
    _tick_parked_work(queue)
    assert queue.persist_queue_snapshot(reason="test_after_sleep_consumption")
    carrier = _snapshot_task(tmp_path, "running", CHILD).get("_budget_pause_resume") or {}
    row = budget_pause.budget_pause_row(tmp_path, CHILD)
    evidence.update(
        state=row["state"], folded=[folded, meta.get("budget_paused_sec"), row["paused_duration_sec"]],
        sleep_parked_at=meta.get("sleep_parked_at"),
        running_projection={key: carrier.get(key) for key in (
            "root_fence_id", "selected_by", "authority", "grant_id", "sleep_exclusion_since")},
        first=_reserve(tmp_path, CHILD), later=_reserve(tmp_path, CHILD),
        sibling=_reserve(tmp_path, SIBLING), root=_reserve(tmp_path, ROOT),
        latch=queue.BUDGET_ROOT_FENCES.get(ROOT))
    workers.assign_tasks()
    evidence["dispatched"] = [task["id"] for task in sent]
    _record(f"owner_resumed_cold_sleeper[{door}]", evidence)
    assert isinstance(evidence["before_consumption"], str), evidence
    assert evidence["stale_events_kept_clock"] is True, evidence
    assert evidence["state"] == budget_pause.STATE_RESUMED and evidence["sleep_parked_at"] is None, evidence
    assert evidence["folded"][0] == evidence["folded"][1] == pytest.approx(evidence["folded"][2]), evidence
    assert evidence["running_projection"] == {
        "root_fence_id": fence_f, "selected_by": "owner", "authority": "explicit_resume",
        "grant_id": granted["grant_id"], "sleep_exclusion_since": None}, evidence
    for key in ("first", "later"):
        assert isinstance(evidence[key], str) and evidence[key], evidence
    assert evidence["sibling"] == evidence["root"] == FENCED, evidence
    assert evidence["latch"]["fence_id"] == fence_f and evidence["dispatched"] == [], evidence


def test_a_warm_park_before_the_cold_notification_keeps_the_selection(tmp_path, monkeypatch):
    """The resumed sleeper warm-parks before its cold consumption event arrives: the park
    folds the cold interval first, retiring only that pending timing, then owns its own
    interval; the published snapshot admits the child's sends under F, parked and woken."""
    from supervisor.events_budget import _handle_budget_pause
    from supervisor.worker_assignment import _tick_parked_work
    from supervisor.worker_owner_wait import _grant_resume, handle_owner_wait

    queue, workers, sup, (child_ctx, child_limit), fence_f, sent = _sleeper_under_latch(
        tmp_path, monkeypatch, wake_after_sec=3600)
    assert queue.resume_budget_paused_task(CHILD)["ok"]
    consumed = _wake_and_consume(monkeypatch, workers, sent, child_ctx, child_limit)
    meta = workers.RUNNING[CHILD]
    worker = workers.WORKERS[meta["worker_id"]]
    worker.proc = SimpleNamespace(pid=77, is_alive=lambda: True)
    cold_started = meta["sleep_parked_at"]
    checkpoint = {"wait_id": "warm-after-cold", "task_attempt": meta["attempt"],
                  "source_ref": {"x": 1}, "reason": "sleep", "sleep": {"senders": [ROOT]}}
    handle_owner_wait({"task_id": CHILD, "wait_id": checkpoint["wait_id"], "task_attempt": meta["attempt"],
                       "worker_id": worker.wid, "pid": 77, "phase": "park", "checkpoint": checkpoint}, workers)
    warm = {"command": [(c.get("phase"), c.get("reason")) for c in sent if c.get("type") == "owner_wait"],
            "state": (meta.get("owner_wait") or {}).get("state"), "parked_at": meta.get("sleep_parked_at"),
            "cold_folded": meta.get("budget_paused_sec")}
    assert warm["command"] == [("parked", None)] and warm["state"] == "waiting", warm
    _tick_parked_work(queue)
    _handle_budget_pause(consumed, sup)
    assert queue.persist_queue_snapshot(reason="test_warm_parked")
    carrier = _snapshot_task(tmp_path, "running", CHILD).get("_budget_pause_resume") or {}
    evidence = {"fence_f": fence_f, "warm": warm, "cold_started": cold_started,
                "after_stale_repairs": [meta.get("sleep_parked_at"), meta.get("budget_paused_sec")],
                "carrier": {key: carrier.get(key) for key in ("authority", "root_fence_id", "sleep_exclusion_since")},
                "parked": _reserve(tmp_path, CHILD)}
    assert _grant_resume(CHILD, meta, worker)
    assert queue.persist_queue_snapshot(reason="test_warm_resumed")
    evidence.update(woken=_reserve(tmp_path, CHILD), sleep_parked_at=meta.get("sleep_parked_at"),
                    sibling=_reserve(tmp_path, SIBLING))
    _record("cold_then_warm_under_latch", evidence)
    assert warm["parked_at"] and warm["parked_at"] != cold_started, evidence
    assert evidence["after_stale_repairs"] == [warm["parked_at"], warm["cold_folded"]], evidence
    assert evidence["carrier"] == {"authority": "explicit_resume", "root_fence_id": fence_f,
                                   "sleep_exclusion_since": None}, evidence
    assert isinstance(evidence["parked"], str) and isinstance(evidence["woken"], str), evidence
    assert evidence["sleep_parked_at"] is None and evidence["sibling"] == FENCED, evidence


def test_a_readiness_wake_beneath_the_latch_neither_starts_nor_spends(tmp_path, monkeypatch):
    """No owner Resume: the sleeper's own readiness is granted under F, but that grant is
    never a selection — the launch is refused, nothing is consumed and no send reserves."""
    from ouroboros import budget_pause
    from supervisor.sleep_wake import wake_ready_sleepers
    from tests.test_model_sleep import _mail

    queue, workers, _sup_ctx, _child, fence_f, sent = _sleeper_under_latch(tmp_path, monkeypatch, senders=[SIBLING])
    _mail(tmp_path, CHILD, "done", sender=SIBLING)
    woke = wake_ready_sleepers(queue)
    assert [outcome.get("ok") for outcome in woke] == [True], woke
    handoff = _pending(workers, CHILD)["_budget_pause_resume"]
    assert (handoff["authority"], handoff["root_fence_id"]) == ("sleep_readiness", fence_f), handoff
    workers.assign_tasks()
    row = budget_pause.budget_pause_row(tmp_path, CHILD)
    evidence = {"fence_f": fence_f, "dispatched": [task["id"] for task in sent], "reserve": _reserve(tmp_path, CHILD),
                "grant": {key: row["grant"].get(key) for key in ("authority", "consumed_at")}, "state": row["state"],
                "latch": queue.BUDGET_ROOT_FENCES.get(ROOT)}
    _record("readiness_wake_under_latch", evidence)
    assert evidence["dispatched"] == [] and evidence["reserve"] == FENCED, evidence
    assert evidence["grant"] == {"authority": "sleep_readiness", "consumed_at": None}, evidence
    assert evidence["state"] == budget_pause.STATE_RESUME_GRANTED and evidence["latch"]["fence_id"] == fence_f, evidence


@pytest.mark.parametrize("wake", ["readiness", "owner"])
def test_a_latch_raised_after_the_wake_refuses_the_consumed_sleeper(tmp_path, monkeypatch, wake):
    """Woken and started before any latch, the sleeper consumes after the root's own pause
    raised F: a readiness carrier leaves, an owner's (fenceless) selection stays but names
    no fence, so neither admits a send under F."""
    from supervisor.events_budget import _handle_budget_pause
    from supervisor.sleep_wake import wake_ready_sleepers
    from tests.test_model_sleep import _mail

    queue, _state, workers = _tree(tmp_path, monkeypatch)
    sup = _sup(tmp_path, queue, workers)
    child_ctx, child_limit = _member_ctx(tmp_path)
    _fixed_price(monkeypatch)
    _install(sup, CHILD, _cold_sleep_park(monkeypatch, child_limit, senders=[ROOT]))
    if wake == "readiness":
        _mail(tmp_path, CHILD, "done", sender=ROOT)
        assert [outcome.get("ok") for outcome in wake_ready_sleepers(queue)] == [True]
    else:
        assert queue.resume_budget_paused_task(CHILD)["ok"]
    sent = _workers(workers)
    workers.assign_tasks()
    assert [task["id"] for task in sent] == [CHILD]
    evidence = {"before_latch": _reserve(tmp_path, CHILD)}
    _root_ctx, root_limit = _member_ctx(tmp_path, ROOT, ROOT)
    _install(sup, ROOT, _planning_park(root_limit))
    events = []
    child_ctx.event_queue = SimpleNamespace(put=events.append)
    _consume(monkeypatch, child_ctx, child_limit, sent[0]["_budget_pause_resume"])
    _handle_budget_pause(next(event for event in events if event.get("phase") == "consumed"), sup)
    assert queue.persist_queue_snapshot(reason="test_after_sleep_consumption")
    carrier = _snapshot_task(tmp_path, "running", CHILD).get("_budget_pause_resume")
    evidence.update(latch=queue.BUDGET_ROOT_FENCES.get(ROOT), after_latch=_reserve(tmp_path, CHILD),
                    carrier={key: carrier.get(key) for key in ("authority", "root_fence_id")} if carrier else None,
                    sleep_parked_at=workers.RUNNING[CHILD].get("sleep_parked_at"))
    _record(f"latch_after_wake[{wake}]", evidence)
    assert isinstance(evidence["before_latch"], str) and evidence["latch"]["fence_id"], evidence
    assert evidence["after_latch"] == FENCED and evidence["sleep_parked_at"] is None, evidence
    assert evidence["carrier"] == (None if wake == "readiness"
                                   else {"authority": "explicit_resume", "root_fence_id": ""}), evidence


def test_a_newer_root_latch_refuses_the_consumed_sleepers_older_selection(tmp_path, monkeypatch):
    """The selection kept past consumption names F only: once the root's own Resume lifts F
    and the root pauses again (latch G), the resumed sleeper's next send is refused."""
    from supervisor.events_budget import _handle_budget_pause

    queue, workers, sup, (child_ctx, child_limit), fence_f, sent = _sleeper_under_latch(
        tmp_path, monkeypatch, wake_after_sec=3600)
    assert queue.resume_budget_paused_task(CHILD)["ok"]
    _handle_budget_pause(_wake_and_consume(monkeypatch, workers, sent, child_ctx, child_limit), sup)
    evidence = {"fence_f": fence_f, "consumed_under_f": _reserve(tmp_path, CHILD)}
    root_ctx, root_limit = _member_ctx(tmp_path, ROOT, ROOT)
    _install(sup, ROOT, _refused_dispatch_park(tmp_path, root_limit, ROOT))
    assert queue.resume_budget_paused_task(ROOT)["ok"]
    workers.assign_tasks()
    assert [task["id"] for task in sent] == [ROOT]
    _consume(monkeypatch, root_ctx, root_limit, sent[0]["_budget_pause_resume"])
    sent.clear()
    _install(sup, ROOT, _planning_park(root_limit))
    evidence.update(latch_g=dict(queue.BUDGET_ROOT_FENCES[ROOT]), after_g=_reserve(tmp_path, CHILD))
    _record("consumed_sleeper_newer_latch", evidence)
    assert isinstance(evidence["consumed_under_f"], str), evidence
    assert evidence["latch_g"]["fence_id"] != fence_f and evidence["after_g"] == FENCED, evidence


def _handoff(**overrides):
    return {"pause_id": "p1", "grant_id": "g1", "authority": "explicit_resume", "root_fence_id": "F",
            "selected_by": "owner", "root_grant_id": "", "root_resume_generation": 0, **overrides}


@pytest.mark.parametrize("row,selected", [
    ({"_budget_pause_resume": _handoff()}, True),
    ({"_budget_pause_resume": _handoff(selected_by="rl-root", root_grant_id="R", root_resume_generation=1)}, True),
    ({"_budget_pause_resume": _handoff(selected_by="rl-root", root_grant_id="R", root_resume_generation=0)}, False),
    ({"_budget_pause_resume": _handoff(selected_by="rl-root")}, False),
    ({"_budget_pause_resume": _handoff(root_fence_id="E")}, False),
    ({"_budget_pause_resume": _handoff(root_fence_id="")}, False),
    ({"_budget_pause_resume": _handoff(pause_id=" ")}, False),
    ({"_budget_pause_resume": _handoff(grant_id="")}, False),
    ({"_budget_pause_resume": _handoff(authority="sleep_readiness", selected_by="sleep_wake")}, False),
    ({"_budget_pause_resume": _handoff(root_fence_id="E"),
      "_budget_pause_hold": {"selected": True, "fence_id": "F"}}, False),
    ({"_budget_pause_resume": _handoff(), "_budget_pause": {"exact_continuation": True}}, False),
    ({"_budget_pause_resume": _handoff(), "_budget_pause_consumed": {"grant_id": "g1"}}, False),
    ({"_budget_pause_resume": _handoff(), "_budget_pause_hold": {"selected": False, "reason": "x"}}, False),
    ({"_budget_pause_resume": _handoff(), "_continuation_prepared": True}, False),
    ({"_budget_pause_hold": {"selected": True, "fence_id": "F"}}, True),
    ({"_budget_pause_hold": {"selected": True, "fence_id": "E"}}, False),
    ({"_budget_pause_hold": {"selected": False, "fence_id": "F"}}, False),
    ({}, False),
], ids=["owner", "root_grant", "root_grant_generation_zero", "model_without_root_grant", "older_fence",
        "no_fence_at_grant", "blank_pause", "blank_grant", "sleep_readiness", "exact_decides_over_old_hold",
        "paused_again", "stale_consumed", "active_hold", "continuation_unconfirmed", "selected_hold",
        "hold_older_fence", "unselected_hold", "nothing"])
def test_the_selection_predicate_reads_one_current_carrier(row, selected):
    from ouroboros.budget_pause import budget_fence_selected

    assert budget_fence_selected({"id": CHILD, **row}, {"fence_id": "F"}) is selected
    assert budget_fence_selected({"id": CHILD, **row}, {"fence_id": ""}) is False


@pytest.mark.parametrize("door", ["park_event", "snapshot_restore", "kill_workers"])
def test_selected_legacy_root_interrupted_new_pause_closes_old_sibling_money_selection(tmp_path, monkeypatch, door):
    from types import SimpleNamespace

    from supervisor.events_budget import budget_resume_dispatch_allowed
    from tests.test_budget_pause_lineage_restore import _restart

    queue, workers, sup, _child, fence_f = _child_under_latch(tmp_path, monkeypatch, sibling=True)
    root_task = workers.RUNNING.pop(ROOT)["task"]
    workers.PENDING.append({**root_task, "_attempt": 1, "admitted_dispatch": "none"})
    assert queue.resume_budget_paused_task(ROOT)["ok"]
    sent = _workers(workers)
    workers.assign_tasks()
    assert [task["id"] for task in sent] == [ROOT]
    assert queue.resume_budget_paused_task(SIBLING, selected_by=ROOT)["ok"]
    assert isinstance(_reserve(tmp_path, SIBLING), str)
    _root_ctx, root_limit = _member_ctx(tmp_path, ROOT, ROOT)
    row = _planning_park(root_limit)
    assert queue.persist_queue_snapshot(reason="probe_root_new_pause_selected_sibling")
    if door == "park_event":
        _install(sup, ROOT, row)
    elif door == "snapshot_restore":
        _restart(queue, workers, monkeypatch)
    else:
        workers.WORKERS.clear()
        monkeypatch.setattr(workers, "_EVENT_Q_SHUTDOWN", False, raising=False)
        monkeypatch.setattr(workers, "get_event_q", lambda: SimpleNamespace(put=lambda _event: None), raising=False)
        workers.kill_workers(force=True, archive_service_logs=False, reconcile_delegate_custody=False)
    sibling = _pending(workers, SIBLING)
    reserved = _reserve(tmp_path, SIBLING)
    facts = {"door": door, "fence_f": fence_f, "latch": queue.BUDGET_ROOT_FENCES.get(ROOT),
             "hold": sibling.get("_budget_pause_hold"), "reserve": reserved,
             "dispatch_allowed": budget_resume_dispatch_allowed(queue, sibling)}
    sent = _workers(workers)
    workers.assign_tasks()
    facts["dispatched"] = [task["id"] for task in sent]
    _record("legacy_root_prepark_sibling_money", facts)
    assert facts["latch"]["fence_id"] != fence_f, facts
    assert facts["dispatched"] == [] and reserved == FENCED, facts
