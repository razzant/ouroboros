"""Owner 2A and 4A (04.11): a reached GLOBAL limit refuses paid admission only, and queued work that
already dispatched is held with its saved work instead of failing.

2A: an attempt is refused at the reached global limit iff its reservation bound is known and above
zero, or its price is unknown on an API provider; the subscription engine's unknown-priced attempt is
admitted (refused again once that route's own receipts carried cash), a known $0 (the local lane) is
admitted, and root/group caps keep refusing everything. 4A: a non-replay-safe queued row is held under
the budget-pause marker when it carries saved work, and the existing Resume door releases it; a row
with nothing saved keeps the honest terminal (holding is no permission to replay).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ouroboros import usage_accounting as ua
from tests._usage_store_testing import ledger_rows
from tests.worker_pool_support import affirm_ready

SUBSCRIPTION = "claudexor::codex=gpt-6-astra"


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    root = tmp_path / "data"
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(root))
    monkeypatch.setenv("OUROBOROS_SETTINGS_PATH", str(root / "settings.json"))
    monkeypatch.setenv("TOTAL_BUDGET", "100")
    (root / "state").mkdir(parents=True)
    ua._reset_task_cache_splits()
    return root


def _paid(data_root, **overrides):
    values = {"model": "openai/gpt-5.2", "provider": "openai", "reservation_usd": 0.5, "drive_root": data_root,
              "task_id": "child", "root_task_id": "root", "source": "test", "global_limit_usd": 1.0}
    values.update(overrides)
    return ua.AttemptRequest(**values)


def _subscription(data_root, model=SUBSCRIPTION, **overrides):
    """``llm_claudexor``'s own reservation shape: provider claudexor, an opaque unknown bound."""
    return _paid(data_root, model=model, provider="claudexor", reservation_usd=None,
                 force_unknown_reservation=True, **overrides)


def _reach_global_limit(data_root):
    spent = ua.reserve_attempt(_paid(data_root, reservation_usd=1.0, task_id="spender"))
    ua.mark_dispatched(spent)
    ua.settle_attempt(spent, {}, cost_usd=1.0, cost_final=True)


def _row(data_root, reservation):
    return next(row for row in ledger_rows(data_root) if row["attempt_id"] == reservation.attempt_id)


@pytest.mark.parametrize("provider, bound, billed, paid", [
    ("claudexor", None, False, False),  # the subscription engine, price unknown before its receipt
    ("claudexor", None, True, True),    # ... whose own receipts already carried cash
    ("local", None, False, False),      # the local lane
    ("openai", 0.0, False, False),      # a known $0 price
    ("claudexor", 0.2, False, True),    # a known price decides, whatever the route
])
def test_paid_admission_is_the_one_rule(provider, bound, billed, paid):
    assert ua.paid_admission(provider, bound, billed=billed) is paid


def test_reached_limit_admits_a_subscription_attempt_that_settles_from_its_receipt(data_root):
    _reach_global_limit(data_root)
    reservation = ua.reserve_attempt(_subscription(data_root))
    ua.mark_dispatched(reservation)  # the dispatch recheck reads the same predicate
    ua.settle_attempt(reservation, {"prompt_tokens": 10, "completion_tokens": 2})  # no cash disclosed
    row = _row(data_root, reservation)
    # Unknown stays unknown: no $0 is invented on the reservation or the settlement.
    assert row["state"] == "settled" and row["reservation_upper_bound_usd"] is None
    assert row["pricing_known"] is False and row.get("cost_usd") is None
    assert ua.usage_projection(data_root, global_limit_usd=1.0)["settled_usd"] == 1.0


@pytest.mark.parametrize("request_kind", ["priced", "unknown_on_api_provider"])
def test_reached_limit_still_refuses_paid_admission(data_root, request_kind):
    _reach_global_limit(data_root)
    request = (_paid(data_root, reservation_usd=0.01) if request_kind == "priced" else
               _paid(data_root, model="vendor/unpriced", provider="openrouter", reservation_usd=None,
                     force_unknown_reservation=True))
    with pytest.raises(ua.BudgetExceeded) as refused:
        ua.reserve_attempt(request)
    assert refused.value.limit_scope == "global"
    assert [row["task_id"] for row in ledger_rows(data_root)] == ["spender"]  # nothing reserved


def test_a_subscription_route_that_billed_is_refused_from_the_next_call(data_root):
    """The owner's 2A cost: a call counted free that turns out paid is known only from its receipt,
    and the limit then refuses that route from the next call; an unbilled route keeps running."""
    first = ua.reserve_attempt(_subscription(data_root))
    ua.mark_dispatched(first)
    _reach_global_limit(data_root)
    ua.settle_attempt(first, {}, cost_usd=0.3, cost_final=True)  # the engine billed a key after all
    with pytest.raises(ua.BudgetExceeded):
        ua.reserve_attempt(_subscription(data_root))
    other = ua.reserve_attempt(_subscription(data_root, model="claudexor::codex=gpt-6-sol"))
    assert _row(data_root, other)["state"] == "reserved"


def test_the_dispatch_recheck_follows_the_same_predicate(data_root):
    subscription = ua.reserve_attempt(_subscription(data_root))
    paid = ua.reserve_attempt(_paid(data_root, reservation_usd=0.2))
    _reach_global_limit(data_root)  # known spend reaches the limit between reservation and dispatch
    ua.mark_dispatched(subscription)
    with pytest.raises(ua.BudgetExceeded, match="changed before dispatch"):
        ua.mark_dispatched(paid)
    assert _row(data_root, subscription)["state"] == "dispatched"
    assert _row(data_root, paid)["state"] == "reserved"


def test_a_known_zero_price_is_admitted_and_root_caps_still_refuse_everything(data_root):
    _reach_global_limit(data_root)
    local = ua.reserve_attempt(ua.AttemptRequest(model="local-model", provider="local", drive_root=data_root,
                                                 task_id="local", root_task_id="local", global_limit_usd=1.0))
    assert _row(data_root, local)["reservation_upper_bound_usd"] == 0.0
    with pytest.raises(ua.BudgetExceeded) as refused:  # 2A is the global limit's rule, not a root cap's
        ua.reserve_attempt(_subscription(data_root, task_id="capped", root_task_id="capped", root_limit_usd=0))
    assert refused.value.limit_scope == "root"


# ---- the queue (pre-worker hold, owner 2A) and saved work at the limit (owner 4A) ----------------------

def _install_queue(tmp_path, monkeypatch):
    from supervisor import queue, state, workers

    state.init(tmp_path, total_budget_limit=10.0)
    queue.init(tmp_path)
    workers.DRIVE_ROOT = tmp_path
    queue.DRIVE_ROOT = tmp_path
    workers.PENDING[:] = []
    workers.RUNNING.clear()
    workers.WORKERS.clear()
    queue.BUDGET_ROOT_FENCES.clear()
    queue.init_queue_refs(workers.PENDING, workers.RUNNING, workers.QUEUE_SEQ_COUNTER_REF)
    monkeypatch.setattr(workers, "load_state", lambda: {"owner_chat_id": 0})
    monkeypatch.setattr(state, "budget_remaining", lambda _st, **_kwargs: 0.0)
    sent = []
    workers.WORKERS[0] = affirm_ready(SimpleNamespace(
        wid=0, busy_task_id=None, reaping=False, in_q=SimpleNamespace(put=lambda task: sent.append(dict(task)))))
    return queue, state, workers, sent


@pytest.mark.parametrize("main, local, dispatched", [
    (SUBSCRIPTION, "", True),
    ("some-local-model", "true", True),
    ("openai/gpt-5.2", "", False),  # paid admission: paused before dispatch, as before
])
def test_at_the_limit_the_queue_pauses_only_work_that_needs_paid_admission(
        tmp_path, monkeypatch, main, local, dispatched):
    monkeypatch.setenv("OUROBOROS_MODEL", main)
    monkeypatch.setenv("USE_LOCAL_MAIN", local)
    queue, _state, workers, sent = _install_queue(tmp_path, monkeypatch)
    workers.PENDING.append({"id": "root", "admitted_dispatch": "none", "type": "task", "chat_id": 0, "priority": 1})
    workers.assign_tasks()
    if dispatched:
        assert [task["id"] for task in sent] == ["root"] and workers.PENDING == []
    else:
        assert sent == [] and workers.PENDING[0]["_budget_pause"]["status"] == "paused_before_dispatch"


def _scheduled_child(kind, target, parent_route, tid="child", root="root"):
    """A configured child exactly as the scheduler queues it (``build_scheduled_task_payload``): its frozen row and
    the parent's frozen cognition, no resolved model (dispatch resolves it later)."""
    import json
    from ouroboros.subagent_runtime import select_subagent_snapshot
    from ouroboros.supervisor_route import validate_cognitive_route
    from supervisor.task_dispatch import build_scheduled_task_payload

    row = {"subagent_id": "builder", "name": "Builder", "recommended_use": "Exact actor.", "effort": "high",
           "route": {"kind": kind, "target_id": target, **({"credential_profile_id": "p"} if kind == "agent_session"
                                                           else {})}}
    snapshot = select_subagent_snapshot({"OUROBOROS_SUBAGENTS": json.dumps({"enabled": True, "items": [row]})},
                                        subagent_id="builder")[0]
    model, chosen_by = parent_route
    cognition = validate_cognitive_route({
        "schema": 2, "model": model, "effort": "high", "use_local_model": False, "source_role": "main",
        "credential_profile_id": "", "processing_preference": "", "chosen_by": chosen_by,
        "follows_role": "main" if chosen_by == "inherited" else ""})
    return build_scheduled_task_payload({"tid": tid, "parent_id": root, "root_task_id": root, "depth": 1,
                                         "delegation_role": "subagent", "configured_subagent": snapshot,
                                         "parent_cognitive_route": cognition})


@pytest.mark.parametrize("main, kind, target, parent_route, paid", [
    ("openai/gpt-5.2", "api_model", SUBSCRIPTION, ("openai/gpt-5.2", "inherited"), False),
    # A nanny supervises its run on the parent's cognition: a parent that chose its own route hands that route down.
    ("openai/gpt-5.2", "agent_session", "codex=gpt-6-astra", (SUBSCRIPTION, "parent_switch"), False),
    # ... a parent following Main hands Main down, live.
    ("openai/gpt-5.2", "agent_session", "codex=gpt-6-astra", ("openai/gpt-5.2", "inherited"), True),
])
def test_a_queued_childs_paid_admission_is_its_own_routes_not_mains(monkeypatch, main, kind, target, parent_route,
                                                                    paid):
    """The pre-worker gate reads the route the child's dispatch will resolve from its frozen row, so a free child of
    a paid Main runs at the reached limit and a paid child of a free Main is held before dispatch."""
    from ouroboros.provider_models import provider_for_model
    from ouroboros.subagent_runtime import resolve_configured_actor_dispatch
    from supervisor.worker_assignment import needs_paid_admission

    monkeypatch.setenv("OUROBOROS_MODEL", main)
    monkeypatch.setenv("USE_LOCAL_MAIN", "")
    child = _scheduled_child(kind, target, parent_route)
    assert "model" not in child["metadata"] and "model" not in child  # the queue holds intent only
    assert needs_paid_admission(child) is paid
    if kind == "api_model":  # the same route the worker's own dispatch resolves
        lane = resolve_configured_actor_dispatch(child, task_type="task").lane
        assert ua.paid_admission("local" if lane.use_local_model else provider_for_model(lane.model), None) is paid


def test_at_the_limit_the_queue_runs_a_free_child_of_a_paid_main(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_MODEL", "openai/gpt-5.2")
    monkeypatch.setenv("USE_LOCAL_MAIN", "")
    _queue, _state, workers, sent = _install_queue(tmp_path, monkeypatch)
    free = _scheduled_child("api_model", SUBSCRIPTION, ("openai/gpt-5.2", "inherited"))
    workers.PENDING.append({**free, "admitted_dispatch": "none", "priority": 1})
    workers.assign_tasks()
    assert [task["id"] for task in sent] == ["child"] and workers.PENDING == []


def _prior_dispatch(tmp_path, task_id):
    held = ua.reserve_attempt(ua.AttemptRequest(model="openai/gpt-5.2", provider="openai", reservation_usd=0.1,
                                                drive_root=tmp_path, task_id=task_id, root_task_id=task_id,
                                                global_limit_usd=100.0))
    ua.mark_dispatched(held)
    ua.settle_attempt(held, {}, cost_usd=0.1, cost_final=True)


def test_a_dispatched_row_with_saved_work_is_held_and_resumes_not_failed(tmp_path, monkeypatch):
    from ouroboros.task_results import load_task_result
    from supervisor import task_admission

    queue, _state, workers, sent = _install_queue(tmp_path, monkeypatch)
    _prior_dispatch(tmp_path, "saved")
    workers.PENDING.append({"id": "saved", "type": "task", "chat_id": 0, "priority": 1, "_attempt": 2,
                            "_working_recovery": {"source_task_id": "saved", "from_attempt": 1}})
    workers.assign_tasks()
    [row] = workers.PENDING
    pause = row["_budget_pause"]
    assert sent == [] and pause["status"] == "paused_before_dispatch" and pause["continuation"] == "working"
    assert pause["replay_safe"] is False and pause["physical_calls"] == 1 and pause["auto_resume"] is False
    assert load_task_result(tmp_path, "saved")["status"] == "scheduled"  # nonterminal, never FAILED

    # The existing Resume door revalidates the saved source: unreadable keeps the hold, valid releases it.
    monkeypatch.setattr(task_admission, "_working_resume_granted", lambda *_a, **_k: False)
    refused = queue.resume_budget_paused_task("saved")
    assert refused["ok"] is False and refused["error"] == "dispatch_outcome_unknown"
    assert "_budget_pause" in workers.PENDING[0]
    monkeypatch.setattr(task_admission, "_working_resume_granted", lambda *_a, **_k: True)
    assert queue.resume_budget_paused_task("saved")["ok"] is True
    assert "_budget_pause" not in workers.PENDING[0] and workers.PENDING[0]["_working_recovery"]


def test_a_dispatched_row_with_nothing_saved_keeps_the_honest_terminal(tmp_path, monkeypatch):
    from ouroboros.task_results import load_task_result

    _queue, _state, workers, sent = _install_queue(tmp_path, monkeypatch)
    _prior_dispatch(tmp_path, "retry")
    workers.PENDING.append({"id": "retry", "type": "task", "chat_id": 0, "priority": 1, "_attempt": 2,
                            "timeout_retry_from": "retry"})
    workers.assign_tasks()
    stored = load_task_result(tmp_path, "retry")
    assert sent == [] and workers.PENDING == []
    assert stored["status"] == "failed" and "after prior dispatch" in stored["result"]


# ---- doors that start several actors at the reached limit (owner 2A): the assisted update, an answered root's Resume

_SESSION_ROW = {"kind": "agent_session", "target_id": "codex=gpt-6-astra"}


@pytest.mark.parametrize("main, pool, admitted", [
    (SUBSCRIPTION, ({"kind": "api_model", "target_id": SUBSCRIPTION}, _SESSION_ROW), True),  # every actor is free
    (SUBSCRIPTION, ({"kind": "api_model", "target_id": "openai/gpt-test"}, _SESSION_ROW), False),  # a paid seat
    ("openai/gpt-test", (_SESSION_ROW,), False),  # its resolver opens on a paid Main
])
def test_at_the_limit_an_assisted_update_starts_only_when_every_actor_is_free(data_root, monkeypatch, main, pool,
                                                                              admitted):
    """The update's resolver and every seat of its commit-review wave are the actors it starts: all free, it goes
    on at the reached limit (the wave answered by the reservation's own ``review_wave_admission``); a paid one keeps
    the refusal before anything is changed."""
    import json
    import ouroboros.pricing as pricing
    import supervisor.git_ops as git_ops
    import supervisor.state as state
    from ouroboros.gateway import control

    class Admitted(Exception):
        pass

    _reach_global_limit(data_root)  # $1 known against the $1 limit
    monkeypatch.setattr(state, "DRIVE_ROOT", data_root)
    monkeypatch.setattr(state, "TOTAL_BUDGET_LIMIT", 1.0)
    monkeypatch.setattr(state, "load_state", lambda: {})
    assert state.budget_remaining({}, strict=True) == 0.0
    monkeypatch.setenv("OUROBOROS_MODEL", main)
    monkeypatch.setenv("USE_LOCAL_MAIN", "")
    monkeypatch.setenv("OUROBOROS_SUBAGENTS", json.dumps({"enabled": True, "items": [
        {"subagent_id": f"critic-{index}", "recommended_use": "Reviews diffs.", "review_eligible": True,
         "route": route} for index, route in enumerate(pool)]}))
    for name in ("_cached_pricing", "_pricing_fetched_at", "_pricing_retry_after"):
        monkeypatch.setattr(pricing, name, {})
    monkeypatch.setattr(pricing, "_fetch_live_rows", lambda *_a, **_k: {"openai/gpt-test": (1.0, 0.1, None, 4.0)})
    monkeypatch.setattr(git_ops, "DRIVE_ROOT", data_root)
    (data_root / "logs").mkdir(exist_ok=True)
    monkeypatch.setattr(control, "_respawn_workers_after_failed_update", lambda: None)
    monkeypatch.setattr(git_ops, "_create_rescue_snapshot", lambda *_a, **_k: (_ for _ in ()).throw(Admitted()))
    plan = {"base_sha": "a" * 40, "target_sha": "b" * 40, "local_snapshot": "a" * 40, "code_conflict_paths": ["x.py"]}
    tx = {"phase": "stashing_local_work", "stash_sha": "", "local_work_carrier": "none"}
    if admitted:
        with pytest.raises(Admitted):  # past the money door: the update goes on to its rescue snapshot
            control._start_assisted_merge_fenced(plan, tx)
    else:
        response = control._start_assisted_merge_fenced(plan, tx)
        assert response.status_code == 409 and "needs model budget" in json.loads(response.body)["error"]


@pytest.mark.parametrize("member_routes, refused", [
    ((), False),  # review-only late work: the answered root runs nothing again, its unsent review is never sent
    ((SUBSCRIPTION, "small-model (local)"), False),  # every member it lets start is free
    ((SUBSCRIPTION, "openai/gpt-5.2"), True),  # a member that pays: the mixed Resume stays refused
])
def test_at_the_limit_an_answered_roots_resume_refuses_only_work_that_needs_paid_admission(
        tmp_path, monkeypatch, member_routes, refused):
    """The Resume of an answered root's Pause (its late phase or its tree) asks the global limit about what it
    starts: the tree's queued and warm members, each by its own route; it stays an explicit Resume per root."""
    from supervisor import budget_resume, continuation_admission

    queue, _state, workers, _sent = _install_queue(tmp_path, monkeypatch)  # remaining $0
    monkeypatch.setenv("OUROBOROS_MODEL", SUBSCRIPTION)
    monkeypatch.setenv("USE_LOCAL_MAIN", "")
    monkeypatch.setattr(budget_resume, "_restart_or_update_refusal", lambda *_a: None)
    monkeypatch.setattr("ouroboros.cancel_intents.has_active_intent", lambda *_a, **_k: False)
    monkeypatch.setattr(continuation_admission, "action_writers", lambda *_a, **_k: [])
    for index, target in enumerate(member_routes):
        workers.PENDING.append({**_scheduled_child("api_model", target, (SUBSCRIPTION, "inherited"),
                                                   tid=f"member-{index}", root="answered"), "priority": 1})
    refusal = budget_resume._late_phase_refusal(queue, tmp_path, "answered", {"status": "completed"},
                                                {"fence_id": "fence"})
    assert (refusal or {}).get("error") == ("budget_still_exhausted" if refused else None)


def test_at_the_limit_the_owners_resume_of_an_answered_roots_unsent_late_review_proceeds(tmp_path, monkeypatch):
    """Through the real door (``resume_budget_paused_task`` -> ``resume_late_phase``): the Pause of an answered root
    whose only late work is a review never sent (a preparation pointer: it is never sent again) starts nothing that
    pays, so at the reached global limit its explicit Resume releases the Pause."""
    from ouroboros import owner_pause
    from ouroboros.review_operation import _update_operations
    from ouroboros.task_results import STATUS_COMPLETED, write_task_result
    from ouroboros.utils import utc_now_iso
    from supervisor.owner_pause_control import refresh_owner_pause_tree

    queue, _state, _workers, sent = _install_queue(tmp_path, monkeypatch)  # remaining $0
    write_task_result(tmp_path, "answered", STATUS_COMPLETED, result="The answer.")
    _update_operations(tmp_path, "answered", lambda rows: {**rows, "late-review": {
        "surface": "commit_review", "retry_key": "k", "task_attempt": 1, "controller": {}, "operations": {"s": "op"},
        "recorded_at": utc_now_iso(), "state": "preparing"}}, create=True)
    fence, created = owner_pause.install_fence(tmp_path, "answered", request_id="pause-answered", late_work=True)
    assert created and owner_pause.fence_closed(fence)
    assert refresh_owner_pause_tree("answered") == owner_pause.FENCE_PAUSED  # the tick's census settles the Pause
    outcome = queue.resume_budget_paused_task("answered")
    assert outcome.get("ok") is True and outcome.get("late_phase") == "resumed", outcome
    assert not owner_pause.fence_closed(owner_pause.read_fence(tmp_path, "answered")) and sent == []
