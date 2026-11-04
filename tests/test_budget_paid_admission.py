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
    ("openrouter", None, False, True),  # unknown price on an API provider
    ("test", None, False, True),
    ("openai", 0.0, False, False),      # a known $0 price
    ("openai", 0.01, False, True),
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
