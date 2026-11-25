"""#4123 repair: an expired park's exact Close keeps custody whatever interrupts it.

The real queue, cancel-intent store, custody and task results run here (the fixture of
``test_ended_work_helpers``). Each case names the class it defends: a failure before the
cancelled write returns the row, a replay after it finishes runs and settlement, a helper
admitted after the frozen membership refuses the root, a parent's request is durable before its
acknowledgement, one parked observation decides reaper, reminder and offer, and a Close is never
offered over work it would not take.
"""
from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest

from ouroboros import cancel_intents as ci
from ouroboros.store import StoreBusy
from ouroboros.task_results import load_task_result, write_task_result
from ouroboros.work_stop import stop_record
from tests._budget_pause_exact_helpers import _install_queue, _parked
from tests.test_ended_work_helpers import CHILD, ROOT, _paused_helper, _pending

DUE = {"deadline_at": "2026-01-01T00:00:00Z"}


@pytest.fixture
def env(tmp_path, monkeypatch):
    queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    events = []
    monkeypatch.setattr(workers, "get_event_q", lambda *a, **k: SimpleNamespace(put=events.append))
    return SimpleNamespace(q=queue, workers=workers, root=tmp_path, events=events, monkeypatch=monkeypatch)


def _offer(env, task, *, late=False):
    """The offer /api/state publishes: the door's own observation (``expired_park_close.observe``)."""
    from supervisor.expired_park_close import observe

    return observe(env.q, task["id"], late=late).get("offer") or {}


def _helper(env):
    write_task_result(env.root, ROOT, "running", root_task_id=ROOT)
    return _paused_helper(env.root, env.monkeypatch, extra=DUE)


def _tree(env):
    target, _ = _parked(env.root, env.monkeypatch, task_id=ROOT, extra={**DUE, "chat_id": 7})
    write_task_result(env.root, ROOT, "scheduled", root_task_id=ROOT, chat_id=7, **DUE)
    _paused_helper(env.root, env.monkeypatch)
    return target


def _closed_by(env, task_id, request_id):
    row = load_task_result(env.root, task_id)
    assert row["status"] == "cancelled" and row["pause_at_close"]["deadline_close"] == request_id, row
    return row


def test_the_row_stays_queued_until_the_cancelled_write_and_a_failure_before_it_keeps_everything(env, monkeypatch):
    """R2 detached the row before capture: a snapshot taken then lost it. Now every snapshot keeps it until commit."""
    import supervisor.ended_work_helpers as helpers
    from supervisor import expired_park_close as epc

    task = _helper(env)
    asked = epc.request_close(env.q, CHILD, _offer(env, task)["close_expired"])
    real_capture, seen, raised = helpers._capture_workspace, [], []

    def capture_then_fail_once(q, row, task_id):
        seen.append(_pending(env) is task)  # what a snapshot taken during capture would save
        if not raised:
            raised.append(True)
            raise StoreBusy("strict read", reason="contention")
        return real_capture(q, row, task_id)

    monkeypatch.setattr(helpers, "_capture_workspace", capture_then_fail_once)
    assert env.q.drive_cancel_intent_scope(CHILD) == "failed"
    assert raised and seen == [True] and _pending(env) is task
    assert load_task_result(env.root, CHILD)["status"] == "scheduled"
    assert ci.active_intent(env.root, CHILD, include_close=True)["request_id"] == asked["request_id"]
    assert env.q.drive_cancel_intent_scope(CHILD) == "cancelled"
    assert seen == [True, True] and _pending(env) is None and not ci.active_intent(env.root, CHILD, include_close=True)
    _closed_by(env, CHILD, asked["request_id"])


def test_a_replay_after_the_cancelled_write_finishes_runs_and_settlement_and_never_revives_the_row(env, monkeypatch):
    import supervisor.ended_work_helpers as helpers
    import supervisor.task_lifecycle as lifecycle
    from ouroboros.gateway import task_close
    from supervisor import expired_park_close as epc

    task = _helper(env)
    _offer_token = [_offer(env, task)["close_expired"]]
    asked = epc.request_close(env.q, CHILD, _offer_token[0])
    real_settle = lifecycle._settle_intent

    def ended(*_args, **_kwargs):
        raise RuntimeError("the process ended after the cancelled write")

    monkeypatch.setattr(lifecycle, "_settle_intent", ended)
    token = _offer_token[0]
    assert env.q.drive_cancel_intent_scope(CHILD) == "failed"
    _closed_by(env, CHILD, asked["request_id"])
    assert _pending(env) is None  # the written verdict is never undone by a restore
    assert ci.active_intent(env.root, CHILD, include_close=True)["request_id"] == asked["request_id"]
    pending = task_close.close_expired_response(CHILD, token)  # the owner's exact retry while the debt stands
    assert (pending.status_code, json.loads(pending.body)) == (
        202, {"ok": True, "task_id": CHILD, "request_id": asked["request_id"], "cancel_state": "pending"})
    env.workers.PENDING.append(dict(task))  # a restored snapshot copy of that same attempt
    monkeypatch.setattr(lifecycle, "_settle_intent", real_settle)
    stopped = []
    monkeypatch.setattr(helpers, "stop_closed_runs", lambda _q, task_id: stopped.append(task_id))
    env.events.clear()
    done = task_close.close_expired_response(CHILD, token)  # the same offer joins and drives that request
    assert (done.status_code, json.loads(done.body)) == (200, {"ok": True, "task_id": CHILD, "request_id": asked["request_id"]})
    assert stopped == [CHILD] and _pending(env) is None and not ci.active_intent(env.root, CHILD, include_close=True)
    assert [e["status"] for e in env.events if e.get("type") == "task_done" and e["task_id"] == CHILD] == ["cancelled"]


@pytest.mark.parametrize("arrival", ["queued", "reserved"])
def test_a_helper_admitted_after_the_frozen_membership_refuses_the_roots_write(env, monkeypatch, arrival):
    from supervisor import expired_park_close as epc

    root = _tree(env)
    asked = epc.request_close(env.q, ROOT, _offer(env, root)["close_expired"])
    assert asked["ok"]
    real, late = epc._close_one, {"id": "late-kid", "type": "task", "root_task_id": ROOT, "parent_task_id": ROOT,
                                 "_attempt": 1}

    def admitted_after_the_member(q, binding, claim, group, **kwargs):
        outcome = real(q, binding, claim, group, **kwargs)
        if binding["task_id"] == CHILD:
            write_task_result(env.root, "late-kid", "requested" if arrival == "reserved" else "scheduled",
                              root_task_id=ROOT, parent_task_id=ROOT)
            if arrival == "queued":
                env.workers.PENDING.append(late)
            else:
                monkeypatch.setitem(env.q.ADMISSION_RESERVATIONS, "late-kid", "admission-token")
        return outcome

    monkeypatch.setattr(epc, "_close_one", admitted_after_the_member)
    assert env.q.drive_cancel_intent_scope(ROOT) == epc.PARTIAL
    _closed_by(env, CHILD, asked["request_id"])
    assert load_task_result(env.root, ROOT)["status"] == "scheduled" and _pending(env, ROOT) is root
    assert not ci.active_intent(env.root, ROOT, include_close=True) and not ci.active_intent(env.root, "late-kid", include_close=True)
    assert load_task_result(env.root, "late-kid")["status"] in {"requested", "scheduled"}
    if arrival == "queued":
        assert _pending(env, "late-kid") is late


def test_a_root_spending_fence_binds_its_unstarted_helper_and_a_live_grant_still_refuses(env):
    """The fence binds a never-started helper (a root's Close takes it), but its own due deadline is the #939
    reaper's: no Close is offered on it alone."""
    from ouroboros.parked_deadline import parked_binding
    from ouroboros.work_stop import queue_control
    from supervisor import queue_timeouts

    _parked(env.root, env.monkeypatch, task_id=ROOT, scope="root")
    write_task_result(env.root, ROOT, "scheduled", root_task_id=ROOT)
    kid = {"id": "kid", "type": "task", "root_task_id": ROOT, "parent_task_id": ROOT, "_attempt": 1,
           "admitted_dispatch": "none", **DUE}
    write_task_result(env.root, "kid", "scheduled", root_task_id=ROOT, parent_task_id=ROOT)
    env.workers.PENDING.append(kid)
    assert env.q.BUDGET_ROOT_FENCES[ROOT]["status"] == "paused" and "state" not in env.q.BUDGET_ROOT_FENCES[ROOT]
    assert parked_binding(env.root, kid, queue_control(env.root, kid, dict(env.q.BUDGET_ROOT_FENCES))["view"])
    with env.q._queue_lock:
        assert queue_timeouts.pending_deadline_expirable(kid)
    assert _offer(env, kid) == {}

    stop = {"scope": "tree", "status": "paused", "fence_id": "f-1", "cause": "budget"}
    row = {"id": "other", "_attempt": 1, "parent_task_id": ROOT, "root_task_id": ROOT, "admitted_dispatch": "none"}
    assert parked_binding(env.root, row, {"stop": stop})
    assert not parked_binding(env.root, row, {"stop": {**stop, "status": "active"}})  # not settled yet
    assert not parked_binding(env.root, {**row, "_budget_pause_resume": {"grant_id": "g"}}, {"stop": stop})


def test_a_spent_grant_under_a_newer_park_binds_that_park(env):
    from ouroboros.parked_deadline import accepted_handoff

    task = _helper(env)
    assert _offer(env, task)
    spent = {**task, "_budget_pause_resume": {"grant_id": "consumed-earlier"}}  # copied into the newer park
    assert not accepted_handoff(spent) and _offer(env, spent)
    assert accepted_handoff({k: v for k, v in spent.items() if k != "_budget_pause"})


def test_a_parents_close_is_durable_before_its_ack_and_the_watchdog_finishes_a_lost_event(env, monkeypatch):
    from ouroboros.owner_mailbox import drain_owner_entries
    from ouroboros.tools import control, join_ledger
    from supervisor.task_lifecycle import sweep_cancel_intents
    from tests.test_model_sleep import _ctx

    task = _helper(env)
    env.workers.RUNNING[ROOT] = {"task": {"id": ROOT, "root_task_id": ROOT}, "attempt": 1}
    token = _offer(env, task)["close_expired"]
    monkeypatch.setattr(control, "_emit_control_event", lambda _ctx, _event: "serialization_failed")  # lost
    ctx = _ctx(env.root, ROOT, root=ROOT)
    assert "recorded." in join_ledger._cancel_task(ctx, CHILD, close_expired=token)
    assert "already recorded" in join_ledger._cancel_task(ctx, CHILD, close_expired=token)
    assert "empty" in join_ledger._cancel_task(ctx, CHILD, close_expired="")
    assert sweep_cancel_intents(now=time.time() + 3600) == {CHILD: "cancelled"}
    [outcome] = drain_owner_entries(env.root, ROOT)
    assert outcome["text"] == f"Close of {CHILD}: closed; its materials are kept."
    assert not ci.active_intent(env.root, CHILD, include_close=True)


def test_a_parents_close_of_a_park_that_changed_is_withdrawn_and_the_parent_is_told(env, monkeypatch):
    from ouroboros.owner_mailbox import drain_owner_entries
    from ouroboros.tools import control, join_ledger
    from supervisor.task_lifecycle import sweep_cancel_intents
    from tests.test_model_sleep import _ctx

    task = _helper(env)
    env.workers.RUNNING[ROOT] = {"task": {"id": ROOT, "root_task_id": ROOT}, "attempt": 1}
    token = _offer(env, task)["close_expired"]
    monkeypatch.setattr(control, "_emit_control_event", lambda _ctx, _event: "live")
    assert "recorded." in join_ledger._cancel_task(_ctx(env.root, ROOT, root=ROOT), CHILD, close_expired=token)
    env.workers.PENDING.remove(task)
    env.workers.RUNNING[CHILD] = {"task": task, "attempt": 2}  # resumed before the supervisor bound it
    assert sweep_cancel_intents(now=time.time() + 3600) == {CHILD: "expired_park_changed"}
    assert env.workers.RUNNING[CHILD]["attempt"] == 2 and not ci.active_intent(env.root, CHILD, include_close=True)
    [outcome] = drain_owner_entries(env.root, ROOT)
    assert "not closed" in outcome["text"] and "ordinary Stop" in outcome["text"]


def test_an_open_ordinary_stop_refuses_a_parents_close_and_keeps_its_own_custody(env, monkeypatch):
    from ouroboros.tools import control, join_ledger
    from tests.test_model_sleep import _ctx

    task = _helper(env)
    token = _offer(env, task)["close_expired"]
    stop = ci.request_cancel(env.root, CHILD, reason="owner stop", source="agent_tool")
    monkeypatch.setattr(control, "_emit_control_event", lambda _ctx, _event: pytest.fail("no Close was recorded"))
    assert "still open" in join_ledger._cancel_task(_ctx(env.root, ROOT, root=ROOT), CHILD, close_expired=token)
    kept = ci.active_intent(env.root, CHILD, include_close=True)
    assert kept["request_id"] == stop["request_id"] and not ci.conditional_park_close(kept)


def test_an_ordinary_stop_beside_a_reserved_close_wins_and_unrelated_intents_stay_readable(env):
    from supervisor import expired_park_close as epc

    task = _helper(env)
    asked = epc.request_close(env.q, CHILD, _offer(env, task)["close_expired"])
    write_task_result(env.root, "other-task", "running")
    other = ci.request_cancel(env.root, "other-task", reason="unrelated", source="agent_tool")
    path = env.root / "state" / "cancel_intents.json"
    document = json.loads(path.read_text())
    assert set(document["intents"]) == {f"park-close:{CHILD}", "other-task"}
    old_stop = {**document["intents"]["other-task"], "task_id": CHILD, "request_id": "ci_olderbody", "reason": "old Stop"}
    document["intents"][CHILD] = old_stop  # what a downgraded body writes beside the reserved Close
    path.write_text(json.dumps(document))
    assert ci.active_intent(env.root, CHILD, include_close=True)["request_id"] == "ci_olderbody"
    assert ci.active_intent(env.root, "other-task", include_close=True)["request_id"] == other["request_id"]
    assert ci.cancel_pending(env.root, CHILD)
    assert ci.mark_intent_scope(env.root, "other-task", ci.SCOPE_CASCADE)  # any next write drops the Close row
    assert set(json.loads(path.read_text())["intents"]) == {CHILD, "other-task"}
    assert asked["request_id"] not in path.read_text()


def test_the_reaper_alone_expires_never_started_holds_and_a_park_that_ran_tells_its_parent(env, monkeypatch):
    """#939 stays the one never-started decision (a lead-selection hold included); a park that ran is kept for
    Close and only its direct parent is told."""
    from supervisor import queue_timeouts, workspace_preparation

    write_task_result(env.root, ROOT, "running", root_task_id=ROOT)
    selection = {"id": "held", "type": "task", "chat_id": 7, "root_task_id": ROOT, "parent_task_id": ROOT,
                 "_attempt": 1, "delegation_role": "subagent", "admitted_dispatch": "none", **DUE,
                 "_budget_pause_hold": stop_record("root_fence_member_pending_selection", held_at="2026-11-10T00:00:00+00:00")}
    write_task_result(env.root, "held", "scheduled", root_task_id=ROOT, parent_task_id=ROOT, delegation_role="subagent")
    plain = {"id": "plain", "type": "task", "root_task_id": "plain", "_attempt": 1, "admitted_dispatch": "none", **DUE}
    write_task_result(env.root, "plain", "scheduled", root_task_id="plain")
    ran = _paused_helper(env.root, monkeypatch, extra=DUE)  # the incident: paused mid-run, never selected again
    env.workers.PENDING.extend([selection, plain])
    with env.q._queue_lock:
        assert queue_timeouts.pending_deadline_expirable(selection) and queue_timeouts.pending_deadline_expirable(plain)
        assert not queue_timeouts.pending_deadline_expirable(ran)
    assert _offer(env, selection) == {} and _offer(env, ran)
    jobs = []
    generation = SimpleNamespace(launches_closed=lambda: False, submit=lambda job, lane: jobs.append(job) or True)
    monkeypatch.setattr(workspace_preparation, "current_generation", lambda: generation)
    monkeypatch.setattr(queue_timeouts, "_queue_pending_deadline", lambda *_a, **_k: None)
    with env.q._queue_lock:
        queue_timeouts._expire_pending_deadlines_locked(env.workers, time.time())
    for row in (selection, plain):
        assert row["_terminalization_retry"]["trigger"] == queue_timeouts.REASON_DEADLINE
    assert "_terminalization_retry" not in ran and len(jobs) == 1  # only the park that ran tells its parent


def test_a_reminder_to_a_finished_parent_is_settled_once_and_not_looked_up_again(env, monkeypatch):
    from supervisor import queue_timeouts, workspace_preparation

    write_task_result(env.root, ROOT, "completed", root_task_id=ROOT, result="done")
    task = _paused_helper(env.root, monkeypatch, extra=DUE)
    jobs = []
    generation = SimpleNamespace(launches_closed=lambda: False, submit=lambda job, lane: jobs.append(job) or True)
    monkeypatch.setattr(workspace_preparation, "current_generation", lambda: generation)
    for _ in range(3):
        with env.q._queue_lock:
            queue_timeouts._expire_pending_deadlines_locked(env.workers, time.time())
        while jobs:
            jobs.pop()()
    assert task.get("_expired_park_notice")  # one lookup proved nobody can receive it


def test_a_helper_with_its_own_live_helper_is_no_close_target(env):
    from supervisor import expired_park_close as epc

    task = _helper(env)
    token = _offer(env, task)["close_expired"]
    nested = {"id": "grandkid", "type": "task", "root_task_id": ROOT, "parent_task_id": CHILD, "_attempt": 1}
    env.workers.PENDING.append(nested)
    assert epc.request_close(env.q, CHILD, token)["error"] == epc.CHANGED
    assert not ci.active_intent(env.root, CHILD, include_close=True) and _pending(env) is task and _pending(env, "grandkid") is nested
    env.workers.PENDING.remove(nested)
    assert epc.request_close(env.q, CHILD, token)["ok"]


def _census(env, task_id, *, late=False, cancelable=True):
    """What /api/state's census publishes for one park past its deadline (``state._expired_parks``)."""
    from ouroboros.gateway.state import _expired_parks

    shown, facts = [{"activity_id": task_id, "can_resume": True, "cancelable": cancelable}], {}
    _expired_parks(env.root, shown, {task_id: "late" if late else "pending"}, (time.time(), []), facts)
    return shown[0], facts


def test_the_census_publishes_only_the_doors_offer_and_otherwise_keeps_stop_without_resume(env):
    from supervisor import expired_park_close as epc

    task = _helper(env)
    shown, facts = _census(env, CHILD, cancelable=False)
    assert shown["close_expired"] == _offer(env, task) and not facts and shown["can_resume"] is False
    grandkid = {"id": "grandkid", "type": "task", "root_task_id": ROOT, "parent_task_id": CHILD, "_attempt": 1}
    env.workers.PENDING.append(grandkid)
    shown, facts = _census(env, CHILD, cancelable=False)
    assert "close_expired" not in shown and facts == {CHILD: {"deadline_expired": True}}
    assert (shown["can_resume"], shown["cancelable"]) == (False, False)  # the census's own Stop grant, unchanged
    env.workers.PENDING.remove(grandkid)
    env.workers.PENDING.remove(task)

    _tree(env)
    shown, _ = _census(env, ROOT)
    offer = shown["close_expired"]
    assert offer["scope"] == "cascade" and shown["cancelable"] is True  # Stop stays beside Close
    env.workers.RUNNING["warm"] = {"task": {"id": "warm", "root_task_id": ROOT, "parent_task_id": ROOT}, "attempt": 1}
    shown, facts = _census(env, ROOT)
    assert "close_expired" not in shown and shown["cancelable"] is True and facts[ROOT]["deadline_expired"]
    env.workers.RUNNING.pop("warm")
    unbound = {"id": "queued-kid", "type": "task", "root_task_id": ROOT, "parent_task_id": ROOT, "_attempt": 1}
    write_task_result(env.root, "queued-kid", "scheduled", root_task_id=ROOT, parent_task_id=ROOT)
    env.workers.PENDING.append(unbound)
    assert "close_expired" not in _census(env, ROOT)[0]
    env.workers.PENDING.remove(unbound)
    assert _census(env, ROOT)[0]["close_expired"] == offer
    assert epc.request_close(env.q, ROOT, offer["close_expired"])["ok"]  # the published token is the door's


def test_a_restored_childs_ancestor_with_only_a_close_is_not_interrupted(env):
    from supervisor import expired_park_close as epc
    from supervisor.queue_snapshot import _interrupted_ancestors

    root = _tree(env)
    assert epc.request_close(env.q, ROOT, _offer(env, root)["close_expired"])["ok"]
    child = _pending(env)
    assert _interrupted_ancestors([], [child], unknown=set()) == set()
    ci.request_cancel(env.root, ROOT, reason="owner stop", source="agent_tool")  # an explicit Stop converts it
    assert _interrupted_ancestors([], [child], unknown=set()) == {ROOT}


def test_close_answers_name_terminal_and_pending_outcomes_separately(env, monkeypatch):
    """Each answer rests on this request's own driver (``drive_request``: ``None`` drove nothing) or its own record
    ``(closed, kept, status)``; a terminal status alone proves nothing for this request."""
    from ouroboros.gateway import task_close
    from supervisor import expired_park_close as epc

    token = "a" * 64
    monkeypatch.setattr(epc, "request_close", lambda *_a, **_k: {"ok": True, "request_id": "ci_x"})
    answers = {}
    close_row = {"request_id": "ci_x", "source": ci.PARKED_CLOSE_SOURCE, "task_id": "t"}
    stop_row = {**close_row, "source": "http_single"}  # ``_close_replaced`` keeps the request id on the Stop
    for case, driven, record, intent in (
            ("kept-alone", "already_settled", (False, False, "completed"), None),
            ("closed", "cancelled", (True, False, "cancelled"), None),
            ("owed", "failed", (False, False, "scheduled"), close_row),
            ("late-kept-read", None, (False, True, "completed"), None),  # a retry reads this request's released fence
            ("closed-read", None, (True, False, "cancelled"), None),
            ("cancelled-elsewhere", "already_settled", (False, False, "cancelled"), None),
            ("cancelled-unproven", "cancelled", (False, False, "cancelled"), None),
            ("stop-replaced", None, (False, False, "cancelled"), stop_row),
            ("stop-mid-flight", "failed", (False, False, "scheduled"), stop_row),
            ("status-unread", "already_settled", (False, False, ""), None)):
        monkeypatch.setattr(epc, "drive_request", lambda *_a, driven=driven: driven)
        monkeypatch.setattr(task_close, "_own_record", lambda *_a, record=record: record)
        monkeypatch.setattr(ci, "active_intent", lambda *_a, intent=intent, **_k: intent)
        response = task_close.close_expired_response("t", token)
        answers[case] = (response.status_code, json.loads(response.body))
    kept = (200, {"ok": True, "task_id": "t", "request_id": "ci_x", "reason_code": "result_kept"})
    assert answers["kept-alone"] == kept and answers["late-kept-read"] == kept
    assert answers["closed"] == answers["closed-read"] == (200, {"ok": True, "task_id": "t", "request_id": "ci_x"})
    assert answers["owed"] == (202, {"ok": True, "task_id": "t", "request_id": "ci_x", "cancel_state": "pending"})
    for case in ("cancelled-unproven", "stop-replaced", "stop-mid-flight"):  # no proof of this request's disposition
        status, body = answers[case]
        assert status == 503 and body["request_id"] == "ci_x" and "reason_code" not in body, (case, body)
        assert "made no change" not in body["error"]
    for case in ("cancelled-elsewhere", "status-unread"):  # this driver found it ended by another hand
        assert answers[case][0] == 409 and answers[case][1]["reason_code"] == epc.CHANGED, answers[case]


def test_an_unread_close_outcome_answers_retry_never_closed_or_nothing_closed(env, monkeypatch):
    """A failed readback is unknown: 503 with the request id and the same Close to retry, whatever the driver said.
    Known corruption is its own 503: before the request was recorded nothing was requested; after it, unknown."""
    from ouroboros.gateway import task_close
    from supervisor import expired_park_close as epc

    monkeypatch.setattr(epc, "request_close", lambda *_a, **_k: {"ok": True, "request_id": "ci_x"})

    def unreadable(*_a, **_k):
        raise OSError("store unavailable")

    def corrupt(*_a, **_k):
        raise ci.CancelIntentProjectionCorrupt("malformed")

    for driven, record, intent in (("cancelled", None, None), ("failed", (False, False, "scheduled"), unreadable)):
        monkeypatch.setattr(epc, "drive_request", lambda *_a, driven=driven: driven)
        monkeypatch.setattr(task_close, "_own_record", lambda *_a, record=record: record)
        monkeypatch.setattr(ci, "active_intent", intent or (lambda *_a, **_k: None))
        response = task_close.close_expired_response("t", "a" * 64)
        body = json.loads(response.body)
        assert response.status_code == 503 and body["request_id"] == "ci_x" and "reason_code" not in body, body
    monkeypatch.setattr(epc, "drive_request", lambda *_a: None)
    monkeypatch.setattr(ci, "active_intent", corrupt)
    after = task_close.close_expired_response("t", "a" * 64)
    body = json.loads(after.body)
    assert after.status_code == 503 and body["reason_code"] == "cancel_intent_projection_corrupt", body
    assert body["request_id"] == "ci_x" and "was recorded" in body["error"] and "nothing was requested" not in body["error"]
    monkeypatch.setattr(epc, "drive_request", lambda *_a: epc.PARTIAL)  # known partial, record unread
    monkeypatch.setattr(task_close, "_own_record", lambda *_a: None)
    partial = task_close.close_expired_response("t", "a" * 64)
    assert partial.status_code == 409 and json.loads(partial.body)["reason_code"] == epc.PARTIAL
    monkeypatch.setattr(epc, "request_close", unreadable)
    assert task_close.close_expired_response("t", "a" * 64).status_code == 503
    monkeypatch.setattr(epc, "request_close", corrupt)
    before = json.loads(task_close.close_expired_response("t", "a" * 64).body)
    assert before["reason_code"] == "cancel_intent_projection_corrupt" and "nothing was requested" in before["error"]
    assert "request_id" not in before and "Retry the same" not in before["error"]
    monkeypatch.setattr(epc, "request_close", lambda *_a, **_k: {"ok": False, "error": epc.CHANGED})
    changed = json.loads(task_close.close_expired_response("t", "a" * 64).body)
    assert "This Close made no change" in changed["error"]  # nothing claimed about an earlier request


def test_a_clock_moved_back_before_the_write_withdraws_and_closes_nothing(env, monkeypatch):
    from supervisor import expired_park_close as epc

    task = _helper(env)
    asked = epc.request_close(env.q, CHILD, _offer(env, task)["close_expired"])
    assert asked["ok"]
    monkeypatch.setattr(epc.time, "time", lambda: 1.0)  # back before 2026-01-01: Resume is the answer again
    assert env.q.drive_cancel_intent_scope(CHILD) == epc.CHANGED
    assert load_task_result(env.root, CHILD)["status"] == "scheduled" and _pending(env) is task
    assert not ci.active_intent(env.root, CHILD, include_close=True)


def test_a_reservation_below_a_helper_withholds_its_close_and_a_siblings_does_not(env, monkeypatch):
    from supervisor import expired_park_close as epc

    task = _helper(env)
    write_task_result(env.root, "sibling", "requested", root_task_id=ROOT, parent_task_id=ROOT)
    monkeypatch.setitem(env.q.ADMISSION_RESERVATIONS, "sibling", "token-a")
    token = _offer(env, task)["close_expired"]
    assert token
    write_task_result(env.root, "grandkid", "requested", root_task_id=ROOT, parent_task_id=CHILD)
    monkeypatch.setitem(env.q.ADMISSION_RESERVATIONS, "grandkid", "token-b")
    assert epc.observe(env.q, CHILD).get("withheld") and not _offer(env, task)
    assert epc.request_close(env.q, CHILD, token) == {"ok": False, "error": epc.CHANGED}
    monkeypatch.setitem(env.q.ADMISSION_RESERVATIONS, "lost", "token-c")  # an unreadable reservation withholds
    monkeypatch.delitem(env.q.ADMISSION_RESERVATIONS, "grandkid")
    real = epc.load_task_result
    monkeypatch.setattr(epc, "load_task_result", lambda root, tid, **k: (_ for _ in ()).throw(OSError("x"))
                        if tid == "lost" else real(root, tid, **k))
    assert epc.observe(env.q, CHILD).get("withheld")


def test_an_answered_roots_open_synthesis_withholds_its_close_before_any_member(env, monkeypatch):
    from supervisor import expired_park_close as epc
    from supervisor import owner_pause_control

    root = _tree(env)
    seen, late_open = epc.observe(env.q, ROOT), [True]
    real = epc._late_settled
    # What an open synthesis checkpoint reads (``owner_pause_control._late_work_settled``), here on the bound park.
    monkeypatch.setattr(epc, "_late_settled", lambda q, binding, row: not late_open[0] and real(q, binding, row))
    assert epc.observe(env.q, ROOT) == {**{k: seen[k] for k in ("row", "binding", "stored", "limit")}, "withheld": True}
    assert epc.request_close(env.q, ROOT, seen["offer"]["close_expired"]) == {"ok": False, "error": epc.CHANGED}
    late_open[0] = False
    asked = epc.request_close(env.q, ROOT, _offer(env, root)["close_expired"])
    late_open[0] = True  # opened after the request: refused before any member is taken
    assert env.q.drive_cancel_intent_scope(ROOT) == epc.CHANGED
    assert load_task_result(env.root, CHILD)["status"] == "scheduled" and _pending(env) is not None
    assert asked["ok"] and owner_pause_control._late_work_settled  # the one predicate _late_settled asks


def test_a_parent_joining_the_owners_close_is_told_no_outcome_message_follows(env):
    from ouroboros.parked_deadline import record_parent_close
    from supervisor import expired_park_close as epc

    task = _helper(env)
    token = _offer(env, task)["close_expired"]
    assert epc.request_close(env.q, CHILD, token)["ok"]  # the owner's press
    words, recorded = record_parent_close(env.root, CHILD, token, requester=ROOT)
    assert not recorded and "already requested by the owner" in words and "no separate outcome" in words
    assert "requested_by" not in (ci.active_intent(env.root, CHILD, include_close=True)["parked_close"])


def test_a_replay_that_finds_this_groups_members_closed_reports_partial_never_nothing_closed(env, monkeypatch):
    from supervisor import expired_park_close as epc

    root = _tree(env)
    asked = epc.request_close(env.q, ROOT, _offer(env, root)["close_expired"])
    real, crashed = epc._close_one, []

    def crash_after_the_member(q, binding, claim, group, **kwargs):
        outcome = real(q, binding, claim, group, **kwargs)
        if binding["task_id"] == CHILD and not crashed:
            crashed.append(True)
            raise RuntimeError("the process ended after the member's write")
        return outcome

    monkeypatch.setattr(epc, "_close_one", crash_after_the_member)
    assert env.q.drive_cancel_intent_scope(ROOT) == "failed" and crashed
    _closed_by(env, CHILD, asked["request_id"])
    late = {"id": "late-kid", "type": "task", "root_task_id": ROOT, "parent_task_id": ROOT, "_attempt": 1}
    write_task_result(env.root, "late-kid", "scheduled", root_task_id=ROOT, parent_task_id=ROOT)
    env.workers.PENDING.append(late)  # admitted between the crash and the replay
    assert env.q.drive_cancel_intent_scope(ROOT) == epc.PARTIAL
    assert _pending(env, ROOT) is root and _pending(env, "late-kid") is late and not ci.active_intent(env.root, ROOT, include_close=True)


def test_a_stop_taking_the_roots_custody_mid_flight_is_never_reported_as_nothing_closed(env, monkeypatch):
    """The residual race: the frozen member is durably closed, then a separate ordinary Stop owns the root's request.
    The driver's ``failed`` is no disposition, so the acknowledgement is the existing 503 (the same Close may be
    retried) — never ``expired_park_changed``, which the closed member's own record contradicts."""
    from ouroboros.gateway import task_close
    from supervisor import expired_park_close as epc

    root = _tree(env)
    token = _offer(env, root)["close_expired"]
    asked = epc.request_close(env.q, ROOT, token)
    assert asked["ok"]
    real = epc._close_one

    def stop_takes_the_root_after_the_member(q, binding, claim, group, **kwargs):
        outcome = real(q, binding, claim, group, **kwargs)
        if binding["task_id"] == CHILD:  # the member is closed; an ordinary Stop then owns the root's own request
            ci.request_cancel(env.root, ROOT, reason="owner stop", source="http_single")
        return outcome

    monkeypatch.setattr(epc, "_close_one", stop_takes_the_root_after_the_member)
    unconfirmed = task_close.close_expired_response(ROOT, token)  # the owner's press joins and drives its request
    monkeypatch.setattr(epc, "_close_one", real)
    body = json.loads(unconfirmed.body)
    assert unconfirmed.status_code == 503 and body["request_id"] == asked["request_id"], body
    assert "reason_code" not in body and "made no change" not in body["error"]
    _closed_by(env, CHILD, asked["request_id"])  # this Close did close a frozen member
    assert load_task_result(env.root, ROOT)["status"] == "scheduled" and _pending(env, CHILD) is None
    stale = task_close.close_expired_response(ROOT, token)  # the same token again: a stale offer, no new action
    assert stale.status_code == 409 and json.loads(stale.body)["reason_code"] == epc.CHANGED
    assert load_task_result(env.root, ROOT)["status"] == "scheduled" and _pending(env, CHILD) is None


def _run_of(env, owner, run_id):
    """A run ``owner`` started in the tree, through custody's own writer: bound, unsettled, no end observed."""
    from ouroboros import delegate_custody as custody

    assert custody.record_started(env.root, custody.RunCustody(
        run_id=run_id, task_id=owner, root_task_id=ROOT, route_id="codex", model="probe", access="readonly"))


def _ended(env, owner, run_id):
    """Its engine's end observed: settlement is still owed, nothing runs (``delegate_continuation``'s test)."""
    from ouroboros import delegate_custody as custody

    assert custody.emit(env.root, custody.TERMINAL_OBSERVED, {"run_id": run_id, "task_id": owner, "state": "succeeded",
                                                               "observation_seq": 1, "source_ref": {"sha256": "0" * 64}})


def test_a_helpers_own_live_run_withholds_offer_reminder_and_close_until_its_end_is_observed(env, monkeypatch):
    from ouroboros.owner_mailbox import drain_owner_entries
    from ouroboros.tools import control, join_ledger
    from supervisor import expired_park_close as epc
    from supervisor import queue_timeouts, workspace_preparation
    from supervisor.task_lifecycle import sweep_cancel_intents
    from tests.test_model_sleep import _ctx

    task = _helper(env)
    env.workers.RUNNING[ROOT] = {"task": {"id": ROOT, "root_task_id": ROOT}, "attempt": 1}
    token = _offer(env, task)["close_expired"]
    monkeypatch.setattr(control, "_emit_control_event", lambda _ctx, _event: "live")
    ctx = _ctx(env.root, ROOT, root=ROOT)
    assert "recorded." in join_ledger._cancel_task(ctx, CHILD, close_expired=token)  # before its run is seen
    _run_of(env, CHILD, "run-own")  # a cold park may keep a body-owned run (a saved recovery hold)
    def census():
        return _census(env, CHILD, cancelable=False)

    shown, facts = census()
    assert "close_expired" not in shown and shown["cancelable"] is False and facts == {CHILD: {"deadline_expired": True}}
    # The durable request is validated against the same custody and withdrawn: nothing closed, the run untouched.
    assert sweep_cancel_intents(now=time.time() + 3600) == {CHILD: epc.CHANGED}
    assert _pending(env) is task and load_task_result(env.root, CHILD)["status"] == "scheduled"
    [outcome] = drain_owner_entries(env.root, ROOT)
    assert "not closed" in outcome["text"] and not ci.active_intent(env.root, CHILD, include_close=True)
    jobs = []
    generation = SimpleNamespace(launches_closed=lambda: False, submit=lambda job, lane: jobs.append(job) or True)
    monkeypatch.setattr(workspace_preparation, "current_generation", lambda: generation)
    with env.q._queue_lock:
        queue_timeouts._expire_pending_deadlines_locked(env.workers, time.time())
    while jobs:
        jobs.pop()()
    assert not [m for m in drain_owner_entries(env.root, ROOT) if m["msg_id"].startswith("expired-park:")]
    assert task["_expired_park_notice"]["quiet"]  # no offer now: the scan asks again after the recheck interval
    assert epc.request_close(env.q, CHILD, token, parent_id=ROOT) == {"ok": False, "error": epc.CHANGED}

    _ended(env, CHILD, "run-own")  # only its settlement is owed now: the same offer stands and closes
    shown, facts = census()
    assert shown["close_expired"]["close_expired"] == token and not facts
    asked = epc.request_close(env.q, CHILD, token, parent_id=ROOT)
    assert asked["ok"] and env.q.drive_cancel_intent_scope(CHILD) == "cancelled"
    _closed_by(env, CHILD, asked["request_id"])


def test_a_roots_frozen_member_with_a_live_run_withholds_the_whole_close_and_its_end_releases_it(env):
    from supervisor import expired_park_close as epc

    target = _tree(env)
    token = _offer(env, target)["close_expired"]
    asked = epc.request_close(env.q, ROOT, token)
    assert asked["ok"]
    _run_of(env, CHILD, "run-member")  # bound to a captured member, yet it may still be running
    child = _pending(env)
    shown, facts = _census(env, ROOT)
    assert "close_expired" not in shown and shown["cancelable"] is True and facts[ROOT]["deadline_expired"]
    assert epc.request_close(env.q, ROOT, token) == asked  # the exact retry joins its open request, mints none
    assert ci.active_intent(env.root, ROOT, include_close=True)["request_id"] == asked["request_id"]
    assert env.q.drive_cancel_intent_scope(ROOT) == epc.CHANGED  # validated before any member: nothing closed
    assert _pending(env) is child and _pending(env, ROOT) is target
    assert load_task_result(env.root, CHILD)["status"] == "scheduled" and not ci.active_intent(env.root, ROOT, include_close=True)
    _ended(env, CHILD, "run-member")
    again = epc.request_close(env.q, ROOT, token)
    assert again["ok"] and env.q.drive_cancel_intent_scope(ROOT) == "cancelled"
    _closed_by(env, ROOT, again["request_id"])
    _closed_by(env, CHILD, again["request_id"])


def test_a_helpers_descendant_behind_an_ended_intermediate_still_withholds_its_close(env):
    """R2 walked live rows only: A's child B ended while B's child C runs, so the walk from C stopped at B."""
    from supervisor import expired_park_close as epc

    task = _helper(env)
    write_task_result(env.root, "ended-mid", "completed", root_task_id=ROOT, parent_task_id=CHILD, result="done")
    env.workers.RUNNING["deep"] = {"task": {"id": "deep", "root_task_id": ROOT, "parent_task_id": "ended-mid"},
                                   "attempt": 1}
    assert not epc.observe(env.q, CHILD)  # no park the door would take
    shown, facts = _census(env, CHILD, cancelable=False)
    assert "close_expired" not in shown and facts[CHILD]["deadline_expired"] and shown["can_resume"] is False
    env.workers.RUNNING.pop("deep")
    assert _offer(env, task)


def test_a_retry_after_a_partial_close_never_reaches_a_member_that_parked_later(env, monkeypatch):
    """The token binds the frozen member bindings: the same offer cannot widen to a newcomer after a partial."""
    from supervisor import expired_park_close as epc

    root = _tree(env)
    token = _offer(env, root)["close_expired"]
    real = epc._close_one

    def newcomer_parks_after_the_member(q, binding, claim, group, **kwargs):
        outcome = real(q, binding, claim, group, **kwargs)
        if binding["task_id"] == CHILD:
            _paused_helper(env.root, monkeypatch, task_id="newcomer")
        return outcome

    monkeypatch.setattr(epc, "_close_one", newcomer_parks_after_the_member)
    asked = epc.request_close(env.q, ROOT, token)
    assert env.q.drive_cancel_intent_scope(ROOT) == epc.PARTIAL
    monkeypatch.setattr(epc, "_close_one", real)
    assert epc.request_close(env.q, ROOT, token) == {"ok": False, "error": epc.CHANGED}
    newcomer = _pending(env, "newcomer")
    assert newcomer is not None and load_task_result(env.root, "newcomer")["status"] == "scheduled"
    assert _offer(env, root)["close_expired"] != token  # a fresh offer names the new membership
    _closed_by(env, CHILD, asked["request_id"])


def test_an_unconfirmed_parent_outcome_keeps_the_request_and_its_retry_tells_once(env, monkeypatch):
    """R2 told the parent after withdrawing: a lost message left nothing to retry. Now the mail comes first."""
    from ouroboros import owner_mailbox
    from ouroboros.owner_mailbox import drain_owner_entries
    from ouroboros.tools import control, join_ledger
    from supervisor import expired_park_close as epc
    from supervisor.task_lifecycle import sweep_cancel_intents
    from tests.test_model_sleep import _ctx

    task = _helper(env)
    env.workers.RUNNING[ROOT] = {"task": {"id": ROOT, "root_task_id": ROOT}, "attempt": 1}
    token = _offer(env, task)["close_expired"]
    monkeypatch.setattr(control, "_emit_control_event", lambda _ctx, _event: "live")
    assert "recorded." in join_ledger._cancel_task(_ctx(env.root, ROOT, root=ROOT), CHILD, close_expired=token)
    env.workers.PENDING.remove(task)
    env.workers.RUNNING[CHILD] = {"task": task, "attempt": 2}  # resumed before the supervisor bound it
    real_write = owner_mailbox.write_task_message
    monkeypatch.setattr(owner_mailbox, "write_task_message", lambda *_a, **_k: False)
    assert sweep_cancel_intents(now=time.time() + 3600) == {CHILD: "failed"}
    assert ci.conditional_park_close(ci.active_intent(env.root, CHILD, include_close=True))
    monkeypatch.setattr(owner_mailbox, "write_task_message", real_write)
    assert sweep_cancel_intents(now=time.time() + 3600) == {CHILD: epc.CHANGED}
    assert not ci.active_intent(env.root, CHILD, include_close=True)
    [outcome] = drain_owner_entries(env.root, ROOT)
    assert "not closed" in outcome["text"] and env.workers.RUNNING[CHILD]["attempt"] == 2


@pytest.mark.parametrize("own", ["none", "future", "due"])
def test_a_never_started_member_is_closed_by_its_roots_close_unless_its_own_due_deadline_is_the_reapers(env, own):
    """A binding is no standalone expiry: a frozen member without its own deadline, or before it, goes with the root's
    exact Close. One whose own deadline is already due belongs to the #939 reaper: the root's Close waits for it."""
    from supervisor import expired_park_close as epc
    from supervisor import queue_timeouts

    _parked(env.root, env.monkeypatch, task_id=ROOT, scope="root", extra=DUE)
    write_task_result(env.root, ROOT, "paused", root_task_id=ROOT, **DUE)
    deadline = {"none": {}, "future": {"deadline_at": "2999-01-01T00:00:00Z"}, "due": DUE}[own]
    kid = {"id": "kid", "type": "task", "root_task_id": ROOT, "parent_task_id": ROOT, "_attempt": 1,
           "delegation_role": "subagent", "admitted_dispatch": "none", **deadline}
    write_task_result(env.root, "kid", "scheduled", root_task_id=ROOT, parent_task_id=ROOT, delegation_role="subagent")
    env.workers.PENDING.append(kid)
    if own == "due":
        assert epc.observe(env.q, ROOT).get("withheld") and not _offer(env, {"id": ROOT})
    env.monkeypatch.setattr(queue_timeouts, "_queue_pending_deadline", lambda *_a, **_k: None)
    with env.q._queue_lock:
        queue_timeouts._expire_pending_deadlines_locked(env.workers, time.time())
    if own == "due":  # the reaper took it; the root's Close waits until that ordinary reaping settles the row
        assert kid["_terminalization_retry"]["trigger"] == queue_timeouts.REASON_DEADLINE
        assert epc.observe(env.q, ROOT).get("withheld")
        env.workers.PENDING.remove(kid)
        write_task_result(env.root, "kid", "failed", reason_code="deadline")
    else:
        assert "_terminalization_retry" not in kid  # no due deadline of its own
    seen = epc.observe(env.q, ROOT)
    assert [m["task_id"] for m in seen["members"]] == ([] if own == "due" else ["kid"]), seen
    asked = epc.request_close(env.q, ROOT, seen["offer"]["close_expired"])
    assert asked["ok"] and env.q.drive_cancel_intent_scope(ROOT) == "cancelled"
    _closed_by(env, ROOT, asked["request_id"])
    if own != "due":
        _closed_by(env, "kid", asked["request_id"])
    assert _pending(env, ROOT) is None and _pending(env, "kid") is None


def test_a_reservation_chain_proves_a_sibling_or_withholds_the_helpers_close(env, monkeypatch):
    """Only a durable chain through this tree's records to its root proves a sibling; a missing or foreign link, a
    cycle or a record-less reservation a live preparation in this tree holds withholds."""
    from supervisor import expired_park_close as epc
    from supervisor import workspace_preparation

    task = _helper(env)
    write_task_result(env.root, "aunt", "running", root_task_id=ROOT, parent_task_id=ROOT)
    write_task_result(env.root, "cousin", "requested", root_task_id=ROOT, parent_task_id="aunt")
    monkeypatch.setitem(env.q.ADMISSION_RESERVATIONS, "cousin", "t1")
    assert _offer(env, task)  # a proven sibling line does not block
    for name, records in (("orphan", {"orphan": "ghost"}),  # an intermediate with no record
                          ("foreigner", {"foreigner": "stranger"}),  # a link in another tree
                          ("looped", {"looped": "loop-b", "loop-b": "looped"}),  # a cycle
                          ("unrecorded", {})):  # no record yet: a live preparation below the helper holds it
        for tid, parent in records.items():
            write_task_result(env.root, tid, "requested", root_task_id=ROOT, parent_task_id=parent)
        if name == "foreigner":
            write_task_result(env.root, "stranger", "running", root_task_id="other-tree")
        if name == "unrecorded":
            monkeypatch.setitem(workspace_preparation._LIVE, "t2", SimpleNamespace(
                task_id=name, route={"root_task_id": ROOT, "parent_id": CHILD}))
        monkeypatch.setitem(env.q.ADMISSION_RESERVATIONS, name, "t2")
        assert epc.observe(env.q, CHILD).get("withheld") and not _offer(env, task), name
        monkeypatch.delitem(env.q.ADMISSION_RESERVATIONS, name)
    assert _offer(env, task)
    # A record-less reservation no preparation holds is ``reserve_task_admission``'s fresh root (API, continuation,
    # project routing): a new tree, no member of this one. The same held by a preparation of another tree is foreign.
    monkeypatch.setitem(env.q.ADMISSION_RESERVATIONS, "fresh-root", "t3")
    assert _offer(env, task)
    monkeypatch.setitem(workspace_preparation._LIVE, "t3", SimpleNamespace(
        task_id="fresh-root", route={"root_task_id": "other-tree", "parent_id": "other-tree"}))
    assert _offer(env, task)
    monkeypatch.setitem(workspace_preparation._LIVE, "t3", SimpleNamespace(task_id="fresh-root", route={}))
    assert epc.observe(env.q, CHILD).get("withheld")  # a preparation of unknown lineage withholds


def test_a_withheld_reminder_rests_on_its_own_row_and_a_new_park_reads_again(env, monkeypatch):
    from supervisor import expired_park_close as epc
    from supervisor import workspace_preparation

    write_task_result(env.root, ROOT, "running", root_task_id=ROOT)
    task = _paused_helper(env.root, monkeypatch, extra=DUE)
    jobs = []
    monkeypatch.setattr(workspace_preparation, "current_generation", lambda: SimpleNamespace(
        launches_closed=lambda: False, submit=lambda job, lane: jobs.append(job) or True))
    monkeypatch.setattr(epc, "observe", lambda *_a, **_k: {"withheld": True})  # e.g. its own run not yet settled
    epc.observe_due_child(env.q, task)
    jobs.pop()()
    assert task[epc.NOTICE]["quiet"] and (str(env.q.DRIVE_ROOT), CHILD) not in epc._reminders_in_flight
    epc.observe_due_child(env.q, task)
    assert not jobs  # rests until the recheck interval or a new park marker
    task[epc.NOTICE]["until"] = time.time() - 1
    epc.observe_due_child(env.q, task)
    assert len(jobs) == 1
    assert not hasattr(epc, "_withheld_until")  # no installation-wide map that could never forget a park


def test_one_publication_copies_the_queue_once_for_all_its_parks_and_a_close_reads_afresh(env, monkeypatch):
    from supervisor import expired_park_close as epc

    _helper(env)
    _paused_helper(env.root, monkeypatch, task_id="second", extra=DUE)
    write_task_result(env.root, "second", "scheduled", root_task_id=ROOT, parent_task_id=ROOT)
    copies, real = [], epc._live_rows
    monkeypatch.setattr(epc, "_live_rows", lambda q: copies.append(1) or real(q))
    memo: dict = {}
    assert epc.observe(env.q, CHILD, closures=memo).get("offer") and epc.observe(env.q, "second", closures=memo).get("offer")
    assert len(copies) == 1
    epc.observe(env.q, CHILD)
    assert len(copies) == 2  # without a publication's memo, every reading is fresh


# ---- #4123 repair, fifth pass: proofs of this request, claim custody, the reaper's clock, the parent's outcome ----

def test_the_http_door_never_drives_or_claims_an_ordinary_stop_that_replaced_its_close(env, monkeypatch):
    """``_close_replaced`` keeps the request id and generation on the Stop: the claim write itself must refuse it."""
    from ouroboros.gateway import task_close

    task = _helper(env)
    token = _offer(env, task)["close_expired"]
    real = ci.claim_intent

    def stop_wins_the_gap(drive_root, task_id, **kwargs):  # after the door read its Close, before the claim write
        ci.request_cancel(env.root, task_id, reason="owner Stop", source="http_single")
        return real(drive_root, task_id, **kwargs)

    monkeypatch.setattr(ci, "claim_intent", stop_wins_the_gap)
    response = task_close.close_expired_response(CHILD, token)
    body = json.loads(response.body)
    assert response.status_code == 503 and "reason_code" not in body, body
    stop = ci.active_intent(env.root, CHILD)
    assert stop["source"] == "http_single" and stop["request_id"] == body["request_id"]
    assert stop.get("state") != ci.INTENT_CLAIMED and not stop.get("generation")  # never claimed by the Close
    assert _pending(env) is task and load_task_result(env.root, CHILD)["status"] == "scheduled"


def test_a_members_close_replaced_by_a_stop_is_never_claimed_and_the_roots_claim_returns(env, monkeypatch):
    from supervisor import expired_park_close as epc

    root = _tree(env)
    asked = epc.request_close(env.q, ROOT, _offer(env, root)["close_expired"])
    real = ci.claim_intent

    def member_stopped_meanwhile(drive_root, task_id, **kwargs):  # the member's Close minted, then replaced
        if task_id == CHILD:
            ci.request_cancel(env.root, CHILD, reason="owner Stop", source="http_single")
        return real(drive_root, task_id, **kwargs)

    monkeypatch.setattr(ci, "claim_intent", member_stopped_meanwhile)
    assert env.q.drive_cancel_intent_scope(ROOT) == "failed"
    member = ci.active_intent(env.root, CHILD)
    assert member["source"] == "http_single" and member.get("state") != ci.INTENT_CLAIMED
    assert not member.get("generation")  # never claimed, so never handed back: its own driver finds it untouched
    close = ci.active_intent(env.root, ROOT, include_close=True)
    assert close["request_id"] == asked["request_id"] and close.get("state") != ci.INTENT_CLAIMED
    assert load_task_result(env.root, CHILD)["status"] == "scheduled" and _pending(env) is not None


def test_a_member_whose_own_deadline_passes_during_capture_is_left_to_the_reaper(env, monkeypatch):
    """#939 at the last write: the clock read inside the final transaction, with the reaper's own predicate."""
    from supervisor import expired_park_close as epc
    from tests.test_ended_work_helpers import _fire_before_write

    _parked(env.root, env.monkeypatch, task_id=ROOT, scope="root", extra=DUE)
    write_task_result(env.root, ROOT, "paused", root_task_id=ROOT, **DUE)
    kid = {"id": "kid", "type": "task", "root_task_id": ROOT, "parent_task_id": ROOT, "_attempt": 1,
           "delegation_role": "subagent", "admitted_dispatch": "none", "deadline_at": "2999-01-01T00:00:00Z"}
    write_task_result(env.root, "kid", "scheduled", root_task_id=ROOT, parent_task_id=ROOT, delegation_role="subagent")
    env.workers.PENDING.append(kid)
    seen = epc.observe(env.q, ROOT)
    assert [m["task_id"] for m in seen["members"]] == ["kid"]
    asked = epc.request_close(env.q, ROOT, seen["offer"]["close_expired"])
    clock = [time.time()]
    monkeypatch.setattr(epc, "time", SimpleNamespace(time=lambda: clock[0]))
    fired = _fire_before_write(env, monkeypatch, "inside_write", lambda: clock.__setitem__(0, 32503680000.0 + 60))
    assert env.q.drive_cancel_intent_scope(ROOT) == epc.CHANGED
    assert fired and load_task_result(env.root, "kid")["status"] == "scheduled" and _pending(env, "kid") is kid
    assert load_task_result(env.root, ROOT)["status"] == "paused" and not ci.active_intent(env.root, "kid", include_close=True)
    assert not ci.active_intent(env.root, ROOT, include_close=True) and asked["ok"]


@pytest.mark.parametrize("ending", ["closed", "kept", "kept_mid_write"])
def test_a_parents_close_tells_its_one_outcome_before_retiring_and_a_lost_message_replays_it(env, monkeypatch, ending):
    """Closed, or already ended by itself (before the capture, or between it and the write) with its own result kept:
    the parent hears it before the request retires; an unconfirmed message keeps the request and its replay (same
    message id) tells exactly once."""
    from ouroboros import owner_mailbox
    from ouroboros.owner_mailbox import drain_owner_entries
    from ouroboros.tools import control, join_ledger
    from supervisor import expired_park_close as epc
    from supervisor.task_lifecycle import sweep_cancel_intents
    from tests.test_model_sleep import _ctx

    task = _helper(env)
    env.workers.RUNNING[ROOT] = {"task": {"id": ROOT, "root_task_id": ROOT}, "attempt": 1}
    token = _offer(env, task)["close_expired"]
    monkeypatch.setattr(control, "_emit_control_event", lambda _ctx, _event: "live")
    assert "recorded." in join_ledger._cancel_task(_ctx(env.root, ROOT, root=ROOT), CHILD, close_expired=token)
    origin = ci.active_intent(env.root, CHILD, include_close=True)["observation"]["request_origin"]
    assert origin == {"kind": "agent_task", "task_id": ROOT}  # the typed actor its record will prove
    if ending == "kept":
        write_task_result(env.root, CHILD, "completed", result="its own answer")
    elif ending == "kept_mid_write":
        from tests.test_ended_work_helpers import _fire_before_write

        _fire_before_write(env, monkeypatch, "before_write",
                           lambda: write_task_result(env.root, CHILD, "completed", result="its own answer"))
    real_write = owner_mailbox.write_task_message
    monkeypatch.setattr(owner_mailbox, "write_task_message", lambda *_a, **_k: False)
    assert sweep_cancel_intents(now=time.time() + 3600) == {CHILD: "failed"}
    assert ci.conditional_park_close(ci.active_intent(env.root, CHILD, include_close=True))  # still owed
    stored = load_task_result(env.root, CHILD)
    assert stored["status"] == ("cancelled" if ending == "closed" else "completed") and _pending(env) is None
    monkeypatch.setattr(owner_mailbox, "write_task_message", real_write)
    expected = "cancelled" if ending == "closed" else "already_settled"
    assert sweep_cancel_intents(now=time.time() + 3600) == {CHILD: expected}
    assert not ci.active_intent(env.root, CHILD, include_close=True)
    [outcome] = drain_owner_entries(env.root, ROOT)
    assert ("closed; its materials are kept" if ending == "closed" else "already ended by itself") in outcome["text"]
    if ending != "closed":
        assert load_task_result(env.root, CHILD)["result"] == "its own answer"
    else:
        from supervisor.cancel_publication import cancel_cause_clauses

        record = load_task_result(env.root, CHILD)
        clauses = [c for c in cancel_cause_clauses(record["cancel_origin"], record, {}) if c]
        assert "Requested by its parent task" in clauses and "Requested by a task" not in clauses, clauses
        assert not any(c.startswith("Stopped with") for c in clauses), clauses
    assert epc._requester({"parked_close": {"requested_by": ROOT}}) == ROOT


def test_a_deadline_close_never_reads_as_stopped_with_its_parent_or_requested_by_an_ancestor():
    from supervisor.cancel_publication import cancel_cause_clauses

    record = {"task_id": "kid", "parent_task_id": "mid", "root_task_id": "top"}
    proven = {"source": "deadline_close", "requested_by": "mid", "request_origin": {"kind": "agent_task", "task_id": "mid"}}
    unproven = {"source": "deadline_close", "requested_by": "mid"}
    ancestor = {"source": "deadline_close", "requested_by": "top", "request_origin": {"kind": "agent_task", "task_id": "top"}}
    stop = {"source": "agent_tool", "requested_by": "mid", "request_origin": {"kind": "agent_task", "task_id": "mid"}}
    clauses = {name: [c for c in cancel_cause_clauses(origin, record, {}) if c]
               for name, origin in (("proven", proven), ("unproven", unproven), ("ancestor", ancestor), ("stop", stop))}
    assert clauses["proven"] == ["Closed after deadline expired", "Requested by its parent task"]
    assert clauses["unproven"] == ["Closed after deadline expired"]
    assert "Requested by an ancestor task" not in clauses["ancestor"] and not any(
        c.startswith("Stopped with") for c in clauses["ancestor"])
    assert clauses["stop"][-1] == "Requested by its parent task"  # the ordinary agent Stop is unchanged


def test_a_late_roots_release_names_its_request_on_the_fence_and_an_interrupted_close_replays_to_result_kept(
        env, monkeypatch):
    from ouroboros.gateway import task_close
    from ouroboros.owner_pause import FENCE_PAUSED, FENCE_REQUESTED, install_fence, read_fence, set_fence_state
    import supervisor.task_lifecycle as lifecycle
    from supervisor import expired_park_close as epc
    from tests.test_ended_work_helpers import _work_root

    _work_root(env.root, deadline_at="2026-01-01T00:00:00Z")
    fence, _ = install_fence(env.root, ROOT, request_id="expired-late-fixture", late_work=True)
    set_fence_state(env.root, ROOT, fence_id=fence["fence_id"], expected_state=FENCE_REQUESTED, state=FENCE_PAUSED)
    _paused_helper(env.root, env.monkeypatch)
    token = epc.observe(env.q, ROOT, late=True)["offer"]["close_expired"]
    real_settle, interrupted = lifecycle._settle_intent, []

    def ends_after_the_release(q, task_id, **kwargs):
        if task_id == ROOT and not interrupted:
            interrupted.append(True)
            raise RuntimeError("the process ended after the fence CAS")
        return real_settle(q, task_id, **kwargs)

    monkeypatch.setattr(lifecycle, "_settle_intent", ends_after_the_release)
    first = task_close.close_expired_response(ROOT, token)
    request_id = json.loads(first.body)["request_id"]
    assert (first.status_code, json.loads(first.body)["cancel_state"]) == (202, "pending")
    released = read_fence(env.root, ROOT)
    assert released["state"] == "released" and (released["deadline_close"], released["close_token"]) == (request_id, token)
    kept = {"ok": True, "task_id": ROOT, "request_id": request_id, "reason_code": "result_kept"}
    again = task_close.close_expired_response(ROOT, token)  # the replay reads its own release, never "changed"
    assert (again.status_code, json.loads(again.body)) == (200, kept)
    assert not ci.active_intent(env.root, ROOT, include_close=True)
    later = task_close.close_expired_response(ROOT, token)  # after the request retired: a read of that same record
    assert (later.status_code, json.loads(later.body)) == (200, kept)
    assert load_task_result(env.root, ROOT)["result"] == "the answer" and _closed_by(env, CHILD, request_id)
    assert not epc._kept_late({"status": "completed", "owner_pause": {"state": "released"}}, token, request_id)


def test_admission_reservations_move_the_published_queue_revision(env, monkeypatch):
    """A reservation withholds an expired park's Close: its arrival and release move the revision, no store read."""
    from ouroboros.gateway.state_publication import _queue_revision
    from supervisor import queue

    before = _queue_revision()
    monkeypatch.setitem(queue.ADMISSION_RESERVATIONS, "fresh", "token-1")
    held = _queue_revision()
    monkeypatch.setitem(queue.ADMISSION_RESERVATIONS, "fresh", "token-2")
    assert len({before, held, _queue_revision()}) == 3
    monkeypatch.delitem(queue.ADMISSION_RESERVATIONS, "fresh")
    assert _queue_revision() == before


# ---- #4123 repair, sixth pass: a competing group driver, an unread tree, an inline Presence parent, one copy ----

def _member_left_open(env, monkeypatch):
    """A root Close whose driver ended after taking its member's request: both requests stay open, nothing closed."""
    from supervisor import expired_park_close as epc

    root = _tree(env)
    asked = epc.request_close(env.q, ROOT, _offer(env, root)["close_expired"])
    real = epc._close_one

    def ended_before_the_member(q, binding, claim, group, **kwargs):
        if binding["task_id"] == CHILD:
            raise RuntimeError("the process ended before the member's capture")
        return real(q, binding, claim, group, **kwargs)

    monkeypatch.setattr(epc, "_close_one", ended_before_the_member)
    assert env.q.drive_cancel_intent_scope(ROOT) == "failed"
    monkeypatch.setattr(epc, "_close_one", real)
    member = ci.active_intent(env.root, CHILD, include_close=True)
    assert (member["parked_close"] or {}).get("group_request") == asked["request_id"]
    return root, asked


def test_a_member_a_competing_driver_closes_after_the_snapshot_counts_as_closed_never_as_no_change(env, monkeypatch):
    """The replay read its members, then the watchdog drove the member's own open request to its cancelled write."""
    from supervisor import expired_park_close as epc

    _root, asked = _member_left_open(env, monkeypatch)
    real, raced = epc._observation, []

    def watchdog_wins_first(q, task_id, **kwargs):
        if task_id == CHILD and kwargs.get("nested") and not raced:
            raced.append(True)
            assert env.q.drive_cancel_intent_scope(CHILD) == "cancelled"  # the member's own request, another driver
        return real(q, task_id, **kwargs)

    monkeypatch.setattr(epc, "_observation", watchdog_wins_first)
    assert env.q.drive_cancel_intent_scope(ROOT) == "cancelled" and raced
    _closed_by(env, CHILD, asked["request_id"])
    _closed_by(env, ROOT, asked["request_id"])


def test_a_member_still_committing_keeps_the_roots_request_owed_and_the_retry_settles_it(env, monkeypatch):
    from supervisor import expired_park_close as epc

    _root, asked = _member_left_open(env, monkeypatch)
    real, raced = epc._observation, []

    def member_busy_once(q, task_id, **kwargs):
        if task_id == CHILD and kwargs.get("nested") and not raced:
            raced.append(True)
            return {}, {}, {}  # its own driver holds it: the read no longer matches, its request is still open
        return real(q, task_id, **kwargs)

    monkeypatch.setattr(epc, "_observation", member_busy_once)
    assert env.q.drive_cancel_intent_scope(ROOT) == "failed" and raced
    assert ci.active_intent(env.root, ROOT, include_close=True)["request_id"] == asked["request_id"]
    assert load_task_result(env.root, ROOT)["status"] == "scheduled"
    monkeypatch.setattr(epc, "_observation", real)
    assert env.q.drive_cancel_intent_scope(CHILD) == "cancelled"
    assert env.q.drive_cancel_intent_scope(ROOT) == "cancelled"
    _closed_by(env, CHILD, asked["request_id"])
    _closed_by(env, ROOT, asked["request_id"])


def test_an_unread_tree_keeps_a_parents_request_owed_and_never_answers_changed(env, monkeypatch):
    from ouroboros import delegate_custody_current
    from ouroboros.gateway import task_close
    from ouroboros.owner_mailbox import drain_owner_entries
    from ouroboros.parked_deadline import record_parent_close
    from supervisor import expired_park_close as epc

    task = _helper(env)
    env.workers.RUNNING[ROOT] = {"task": {"id": ROOT, "root_task_id": ROOT}, "attempt": 1}
    token = _offer(env, task)["close_expired"]
    real = delegate_custody_current.tree_closure

    def contended(*_args, **_kwargs):
        raise StoreBusy("strict read", reason="contention")

    monkeypatch.setattr(delegate_custody_current, "tree_closure", contended)
    seen = epc.observe(env.q, CHILD)
    assert seen["withheld"] and seen["unread"] and "offer" not in seen
    unconfirmed = task_close.close_expired_response(CHILD, token)  # the owner's press: retry, never "changed"
    assert unconfirmed.status_code == 503 and "reason_code" not in json.loads(unconfirmed.body)
    assert record_parent_close(env.root, CHILD, token, requester=ROOT)[1]
    assert env.q.drive_cancel_intent_scope(CHILD) == "failed"
    assert ci.conditional_park_close(ci.active_intent(env.root, CHILD, include_close=True))  # still owed
    assert drain_owner_entries(env.root, ROOT) == [] and _pending(env) is task
    monkeypatch.setattr(delegate_custody_current, "tree_closure", real)
    assert env.q.drive_cancel_intent_scope(CHILD) == "cancelled"
    [outcome] = drain_owner_entries(env.root, ROOT)
    assert "closed; its materials are kept" in outcome["text"]


def test_an_inline_presence_parent_hears_its_reminder_and_outcome_through_its_own_mailbox(env, monkeypatch):
    from ouroboros.owner_mailbox import drain_owner_entries
    from ouroboros.parked_deadline import record_parent_close
    from supervisor import expired_park_close as epc

    turn = "presence-turn"
    write_task_result(env.root, turn, "running", source="presence", _is_direct_chat=True, root_task_id=turn,
                      metadata={"presence": {"binding_id": "1" * 32}, "presence_event_identity": "event-identity"})
    task = _paused_helper(env.root, monkeypatch, extra=DUE, work=turn)
    assert epc._parent_actor(env.q, turn)["id"] == turn  # no queue map holds it; its exact mailbox does
    token = _offer(env, task)["close_expired"]
    marker = ["m"]
    epc._remind(env.q, CHILD, task, turn, (str(env.root), CHILD), marker)
    assert task.get("_expired_park_notice") == marker
    [reminder] = drain_owner_entries(env.root, turn)
    assert token in reminder["text"]
    assert record_parent_close(env.root, CHILD, token, requester=turn)[1]  # its own cancel_task(close_expired=...)
    assert env.q.drive_cancel_intent_scope(CHILD) == "cancelled"
    texts = [entry["text"] for entry in drain_owner_entries(env.root, turn)]
    assert len(texts) == 2 and token in texts[0] and "closed; its materials are kept" in texts[1], texts
    write_task_result(env.root, "plain-turn", "running", root_task_id="plain-turn")  # no Presence mailbox: no target
    assert epc._parent_actor(env.q, "plain-turn") is None


def test_a_roots_reading_without_a_memo_copies_the_queue_once_for_all_its_members(env, monkeypatch):
    from supervisor import expired_park_close as epc

    root = _tree(env)
    copies, real = [], epc._live_rows
    monkeypatch.setattr(epc, "_live_rows", lambda q: copies.append(1) or real(q))
    assert epc.observe(env.q, ROOT).get("offer") and len(copies) == 1
    assert epc.observe(env.q, ROOT)["offer"] == _offer(env, root) and len(copies) == 3  # each reading is fresh


@pytest.mark.parametrize("shape", ["bound", "held", "warm"])
def test_every_paused_row_past_its_own_deadline_drops_resume_and_only_a_bound_queued_park_offers_close(env, shape):
    """Resume refuses past the row's own hard deadline whatever binds it (``resume_clock_refusal``): a held row Close
    never binds and a warm park whose worker owns settlement say Deadline expired too, with Stop and without Close."""
    from ouroboros.gateway import work_status as ws
    from ouroboros.gateway.state import _chat_activities_snapshot
    from ouroboros.work_stop import task_current_stop

    root = _tree(env)
    if shape == "held":
        root["_continuation_prepared"] = True  # a Resume-controlled hold: parked_binding binds no Close
    pending, running = list(env.workers.PENDING), []
    if shape == "warm":  # its stack parked warm under the owner's settled Pause (``queue_control``'s running read)
        import ouroboros.work_stop as work_stop

        env.monkeypatch.setattr(work_stop, "owner_fence_state", lambda *_a: "paused")
        pending, running = [_pending(env)], [(ROOT, {**root, "_owner_pause_warm": True}, time.time())]
    facts = {}
    activities = _chat_activities_snapshot(env.root, {}, [], pending, dict(env.q.BUDGET_ROOT_FENCES), running, [], {},
                                           [], False, work_facts=facts, park_clock=(time.time(), []))
    [activity] = [item for item in activities if item["activity_id"] == ROOT]
    assert activity["phase"] == "budget_paused" and activity["can_resume"] is False
    assert bool(activity.get("close_expired")) is (shape == "bound")
    assert bool(facts[ROOT].get("deadline_expired")) is (shape != "bound")
    member = ws.classify_member(activity, facts[ROOT])
    assert (member["activity"], member["release"], member["resumable"]) == ("paused", "none", False)
    assert member["reasons"][0]["code"] == "deadline_expired"
    assert {entry["action"] for entry in ws.controls_for(member, activity)} == {"stop"}  # Close rides its own offer
    lane = "running" if shape == "warm" else "pending"
    snapshot = {lane: [{"id": ROOT, "task": root, "attempt": 1}]}
    seen = task_current_stop(env.root, snapshot, ROOT)
    assert seen.get("deadline_expired") is True and seen["release"] == "none", seen
