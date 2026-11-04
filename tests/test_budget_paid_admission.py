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

import pytest

from ouroboros import usage_accounting as ua
from tests._usage_store_testing import ledger_rows

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
