"""The rolling-24h consciousness allowance (P3).

``consciousness_allowance.allowance_window`` reads the money consciousness
(its wakes plus every root it started) accounted in the last 24 h straight off
the usage store: roots are named by the category of their current rows younger
than the 48 h root horizon, the window sums every money row kind under those
roots, and both selections are addressed queries filtered by the clock on
every call (owner decisions В11/В18; PLAN 5.5, 5.13 п.11, 5.14 п.1). Every
attempt keeps its own transition time (``ts``) for good: nothing folds.
"""

from __future__ import annotations

import datetime as _dt

import pytest

from ouroboros import consciousness_allowance as allowance
from ouroboros import usage_accounting as ua
from ouroboros import usage_store
from tests import fixtures_usage_store as _fixtures
from tests._usage_store_testing import write_compacted_journal
from tests.fixtures_usage_store import _ledger_rows, _request, _settle

data_root = _fixtures.data_root

T0 = _dt.datetime(2026, 9, 16, 12, 0, tzinfo=_dt.timezone.utc).timestamp()
HOUR = 3600.0


def _iso(ts: float) -> str:
    return _dt.datetime.fromtimestamp(ts, tz=_dt.timezone.utc).isoformat()


def _at(monkeypatch, ts: float) -> None:
    """Every row written from here on carries this transition stamp."""
    monkeypatch.setattr(usage_store, "utc_now_iso", lambda: _iso(ts))


def _wake(root, monkeypatch, ts, cost, task_id="wake-1"):
    _at(monkeypatch, ts)
    return _settle(root, cost=cost, cost_final=True, task_id=task_id, root_task_id=task_id,
                   category="consciousness")


def _started(root, monkeypatch, ts, cost, task_id="c-root", category="consciousness_task"):
    _at(monkeypatch, ts)
    return _settle(root, cost=cost, cost_final=True, task_id=task_id, root_task_id="c-root",
                   category=category)


# --- the window ----------------------------------------------------------------


def test_window_sums_the_whole_consciousness_tree_and_nothing_else(data_root, monkeypatch):
    monkeypatch.setenv("OUROBOROS_CONSCIOUSNESS_DAILY_USD", "20")
    _wake(data_root, monkeypatch, T0 - 2 * HOUR, 1.0)
    _started(data_root, monkeypatch, T0 - 3 * HOUR, 2.0)
    # A consolidation/review row of the SAME started tree keeps its own category yet counts.
    _started(data_root, monkeypatch, T0 - 1 * HOUR, 0.5, task_id="c-root-consolidation", category="consolidation")
    # The owner's own work never enters the window.
    _at(monkeypatch, T0 - 1 * HOUR)
    _settle(data_root, cost=50.0, cost_final=True, task_id="owner", root_task_id="owner", category="task")
    window = allowance.allowance_window(data_root, now=T0)
    assert window["status"] == "available"
    assert window["accounted_usd"] == pytest.approx(3.5)
    assert window["remaining_usd"] == pytest.approx(16.5)
    assert window["roots"] == ["c-root", "wake-1"]
    assert window["window_rows"] == 3 and window["unknown_unmetered"] == 0
    assert window["resets_at"] == _iso(T0 - 3 * HOUR + 24 * HOUR)  # the oldest counted spend leaves first


def test_every_money_row_kind_counts_but_imported_aggregates_never_do(data_root, monkeypatch):
    monkeypatch.setenv("OUROBOROS_CONSCIOUSNESS_DAILY_USD", "20")
    # An aggregate the retired compactor folded, arriving with the journal import.
    write_compacted_journal(data_root, [{
        "attempt_id": "baseline-x-g0001", "ts": _iso(T0 - HOUR), "cost_usd": "99", "cost_final": True,
        "model": "m", "provider": "p", "category": "consciousness_task", "source": "test",
        "task_id": "c-root", "root_task_id": "c-root", "parent_task_id": "", "folded_attempt_count": 3,
    }])
    _started(data_root, monkeypatch, T0 - 5 * HOUR, 1.0)
    _at(monkeypatch, T0 - 4 * HOUR)
    ua.record_subscription_session("sess-c", drive_root=data_root, route="claudexor:claude", model="fable",
                                   task_id="c-child", root_task_id="c-root", spend_usd=0.75,
                                   reset_at="2026-09-17T00:00:00Z", category="subagent")
    ua.record_unmetered_external_dispatch("ext-c", drive_root=data_root, model="ext-model", task_id="c-child",
                                          root_task_id="c-root", prompt_tokens=7, completion_tokens=3)
    assert any(row.get("kind") == "usage_baseline_group" for row in _ledger_rows(data_root))
    window = allowance.allowance_window(data_root, now=T0)
    assert window["accounted_usd"] == pytest.approx(1.75)
    assert window["unknown_unmetered"] == 1  # the external dispatch: cost unknown, counted as "at least"
    assert window["window_rows"] == 3


def test_the_time_filter_runs_on_every_call(data_root, monkeypatch):
    monkeypatch.setenv("OUROBOROS_CONSCIOUSNESS_DAILY_USD", "3")
    _wake(data_root, monkeypatch, T0 - 20 * HOUR, 2.0)
    _wake(data_root, monkeypatch, T0 - 2 * HOUR, 1.5, task_id="wake-2")
    exhausted = allowance.allowance_window(data_root, now=T0)
    assert exhausted["status"] == "exhausted" and exhausted["accounted_usd"] == pytest.approx(3.5)
    assert exhausted["resets_at"] == _iso(T0 + 4 * HOUR)
    # No new rows: the same addressed selection, re-filtered by the clock.
    later = allowance.allowance_window(data_root, now=T0 + 4 * HOUR + 1)
    assert later["status"] == "available" and later["accounted_usd"] == pytest.approx(1.5)
    empty = allowance.allowance_window(data_root, now=T0 + 24 * HOUR + 1)
    assert empty["status"] == "available" and empty["accounted_usd"] == 0.0
    assert empty["resets_at"] == "" and empty["window_rows"] == 0
    # The roots are still attributable inside the 48 h horizon; past it they drop out too.
    assert empty["roots"] == ["wake-1", "wake-2"]
    assert allowance.allowance_window(data_root, now=T0 + 48 * HOUR + 1)["roots"] == []


def test_a_root_whose_consciousness_rows_aged_past_the_horizon_drops_out(data_root, monkeypatch):
    monkeypatch.setenv("OUROBOROS_CONSCIOUSNESS_DAILY_USD", "20")
    # The started root's own (category-bearing) rows are 50 h old; a later consolidation
    # row of the same tree is fresh. Without a root index the tree is no longer attributable.
    _started(data_root, monkeypatch, T0 - 49 * HOUR, 5.0)
    _started(data_root, monkeypatch, T0 - 4 * HOUR, 4.0, task_id="c-root-consolidation", category="consolidation")
    window = allowance.allowance_window(data_root, now=T0)
    assert window["roots"] == [] and window["accounted_usd"] == 0.0
    # Two hours earlier the tree was still inside the horizon and its consolidation counted.
    inside = allowance.allowance_window(data_root, now=T0 - 2 * HOUR)
    assert inside["roots"] == ["c-root"] and inside["accounted_usd"] == pytest.approx(4.0)


def test_zero_allowance_means_consciousness_may_not_spend(data_root, monkeypatch):
    monkeypatch.setenv("OUROBOROS_CONSCIOUSNESS_DAILY_USD", "0")
    window = allowance.allowance_window(data_root, now=T0)
    assert window["status"] == "exhausted" and window["limit_usd"] == 0.0 and window["remaining_usd"] == 0.0


def test_an_unreadable_ledger_is_the_typed_unknown_outcome(data_root, monkeypatch):
    monkeypatch.setenv("OUROBOROS_CONSCIOUSNESS_DAILY_USD", "20")

    def boom(root, **_display_read):
        raise OSError("ledger locked")

    monkeypatch.setattr(usage_store, "read", boom)
    window = allowance.allowance_window(data_root, now=T0)
    assert window["status"] == "allowance_unknown" and "ledger locked" in window["error"]
    assert window["limit_usd"] == 20.0 and window["remaining_usd"] is None


def test_row_ts_epoch_reads_the_appender_stamp_and_refuses_to_guess():
    from ouroboros._usage_rows import row_ts_epoch

    assert row_ts_epoch({"ts": "2026-09-16T12:00:00+00:00"}) == T0
    assert row_ts_epoch({"ts": "2026-09-16T12:00:00Z"}) == T0
    assert row_ts_epoch({"ts": "2026-09-16T12:00:00"}) == T0  # naive = UTC
    assert row_ts_epoch({"ts": "not a time"}) is None
    assert row_ts_epoch({}) is None and row_ts_epoch(None) is None


# --- transition times ---------------------------------------------------------------


def test_every_attempt_keeps_its_own_transition_time(data_root, monkeypatch):
    """Nothing folds: an old chain and a fresh one both keep their own ``ts``."""
    _at(monkeypatch, T0 - 3 * 24 * HOUR)
    old = _settle(data_root, cost=1.25, cost_final=True, task_id="old", root_task_id="old")
    _at(monkeypatch, T0 - 2 * HOUR)
    fresh = _settle(data_root, cost=0.75, cost_final=True, task_id="fresh", root_task_id="fresh",
                    category="consciousness")
    live = {str(row.get("attempt_id")): row for row in _ledger_rows(data_root)}
    assert live[old.attempt_id]["ts"] == _iso(T0 - 3 * 24 * HOUR)
    assert live[fresh.attempt_id]["ts"] == _iso(T0 - 2 * HOUR)
    assert ua.usage_projection(data_root)["settled_usd"] == pytest.approx(2.0)
    assert allowance.allowance_window(data_root, now=T0)["accounted_usd"] == pytest.approx(0.75)


def test_the_root_horizon_is_twice_the_window():
    assert allowance.ROOT_HORIZON_SEC == 2 * allowance.WINDOW_SEC == 48 * 3600


# --- what a route's price actually costs the allowance ----------------------------
#
# The wake prompt asks this mind to tell incremental CASH from subscription quota
# and time. These pin what the EXISTING money reducer already answers, route by
# route, so the distinction the prompt draws is the one the allowance enforces.
# There is no quota budget here and none is wanted: quota and time are separate
# observed axes, and only cash is metered against the owner's daily dollars.


def _subscription(root, monkeypatch, ts, *, session, spend, estimated=False, task="c-child",
                  root_task="c-root", category="consciousness_task"):
    _at(monkeypatch, ts)
    return ua.record_subscription_session(
        session, drive_root=root, route="claudexor:claude", model="fable", task_id=task,
        root_task_id=root_task, spend_usd=spend, spend_estimated=estimated,
        reset_at="2026-09-17T00:00:00Z", category=category)


def test_a_confirmed_free_route_spends_no_cash_and_leaves_the_window_final(data_root, monkeypatch):
    """A settled zero the engine called FINAL is a fact, not a missing number."""
    monkeypatch.setenv("OUROBOROS_CONSCIOUSNESS_DAILY_USD", "20")
    _wake(data_root, monkeypatch, T0 - 2 * HOUR, 0.0)
    _subscription(data_root, monkeypatch, T0 - HOUR, session="sess-free", spend=0.0)
    window = allowance.allowance_window(data_root, now=T0)
    assert window["status"] == "available"
    assert window["accounted_usd"] == pytest.approx(0.0)
    assert window["remaining_usd"] == pytest.approx(20.0)
    assert window["window_rows"] == 2
    # Free is not unknown and not open: nothing here makes the window "at least".
    assert window["unknown_unmetered"] == 0 and window["non_final_rows"] == 0


def test_an_estimated_zero_costs_nothing_yet_but_the_window_is_not_final(data_root, monkeypatch):
    """The engine reported $0 it has not settled; the honest answer says so."""
    monkeypatch.setenv("OUROBOROS_CONSCIOUSNESS_DAILY_USD", "20")
    _wake(data_root, monkeypatch, T0 - 2 * HOUR, 1.0)
    _subscription(data_root, monkeypatch, T0 - HOUR, session="sess-est", spend=0.0, estimated=True)
    window = allowance.allowance_window(data_root, now=T0)
    assert window["accounted_usd"] == pytest.approx(1.0)  # the estimate adds no dollars
    assert window["non_final_rows"] == 1                  # but the window is still open
    assert window["unknown_unmetered"] == 0               # and the price is not unknown


def test_an_undisclosed_price_is_unknown_and_reads_as_at_least(data_root, monkeypatch):
    """No number at all: counted as unknown rather than silently as zero."""
    monkeypatch.setenv("OUROBOROS_CONSCIOUSNESS_DAILY_USD", "20")
    _wake(data_root, monkeypatch, T0 - 2 * HOUR, 1.0)
    _subscription(data_root, monkeypatch, T0 - HOUR, session="sess-unknown", spend=None)
    window = allowance.allowance_window(data_root, now=T0)
    assert window["accounted_usd"] == pytest.approx(1.0)
    assert window["unknown_unmetered"] == 1 and window["non_final_rows"] == 1


def test_a_known_reservation_bound_stays_accounted_while_the_price_is_unknown(data_root, monkeypatch):
    """An unresolved attempt holds its upper bound: disclosed, not forgotten."""
    monkeypatch.setenv("OUROBOROS_CONSCIOUSNESS_DAILY_USD", "20")
    _wake(data_root, monkeypatch, T0 - 3 * HOUR, 1.0)
    _at(monkeypatch, T0 - 2 * HOUR)
    reservation = ua.reserve_attempt(_request(data_root, reservation_usd=4.0, task_id="c-child",
                                              root_task_id="c-root", category="consciousness_task"))
    ua.mark_dispatched(reservation)
    ua.mark_unresolved(reservation, "the provider went dark")
    window = allowance.allowance_window(data_root, now=T0)
    assert window["accounted_usd"] == pytest.approx(5.0)  # 1.0 settled + the 4.0 bound, disclosed
    assert window["settled_usd"] == pytest.approx(1.0)
    assert window["remaining_usd"] == pytest.approx(19.0)  # known spend decides (#1487)
    assert window["non_final_rows"] == 1


def test_a_refused_oversized_request_never_closes_the_allowance_for_a_day(data_root, monkeypatch):
    """The 2026-10-05 incident (#1487 comment): $31.68 known of a $50 allowance, then a
    provider-refused fallback recorded `unresolved` at its $41.27 bound. Its worst case is
    exposure beside the known spend, never spending: the next wake is admitted with the
    known $18.32 left, instead of 21 hours of `allowance_exhausted`."""
    monkeypatch.setenv("OUROBOROS_CONSCIOUSNESS_DAILY_USD", "50")
    _wake(data_root, monkeypatch, T0 - 2 * HOUR, 31.68)
    _at(monkeypatch, T0 - HOUR)
    refused = ua.reserve_attempt(_request(data_root, reservation_usd=41.27, task_id="wake-1",
                                          root_task_id="wake-1", category="consciousness"))
    ua.mark_dispatched(refused)
    ua.mark_unresolved(refused, "HTTP 400 total text input size exceeds 8 MB")
    window = allowance.allowance_window(data_root, now=T0)
    assert window["status"] == allowance.STATUS_AVAILABLE
    assert window["settled_usd"] == pytest.approx(31.68)
    assert window["remaining_usd"] == pytest.approx(18.32)
    assert window["accounted_usd"] == pytest.approx(72.95)  # the open bound stays disclosed


def test_a_positive_charge_and_a_descendants_charge_both_land_on_the_one_allowance(data_root, monkeypatch):
    """The wake, the root it started and that root's descendant share one window."""
    monkeypatch.setenv("OUROBOROS_CONSCIOUSNESS_DAILY_USD", "20")
    _wake(data_root, monkeypatch, T0 - 4 * HOUR, 1.0)
    _started(data_root, monkeypatch, T0 - 3 * HOUR, 2.0)
    # A grandchild keeps its own category; the ROOT is what puts it in the window.
    _subscription(data_root, monkeypatch, T0 - HOUR, session="sess-grandchild", spend=0.5,
                  task="c-grandchild", category="subagent")
    window = allowance.allowance_window(data_root, now=T0)
    assert window["accounted_usd"] == pytest.approx(3.5)
    assert window["roots"] == ["c-root", "wake-1"] and window["window_rows"] == 3
    # The same money on a root that is NOT consciousness's stays out of the window.
    _at(monkeypatch, T0 - HOUR)
    ua.record_subscription_session("sess-owner", drive_root=data_root, route="claudexor:claude",
                                   model="fable", task_id="owner-child", root_task_id="owner",
                                   spend_usd=9.0, reset_at="2026-09-17T00:00:00Z", category="subagent")
    assert allowance.allowance_window(data_root, now=T0)["accounted_usd"] == pytest.approx(3.5)


def test_a_free_route_gets_no_exception_from_an_exhausted_or_zero_allowance(data_root, monkeypatch):
    """Cheap is not free of the owner's decision: the door is the same door."""
    import supervisor.queue as queue

    monkeypatch.setattr(queue, "DRIVE_ROOT", data_root)
    monkeypatch.setattr(queue, "live_consciousness_root_count", lambda: 0)
    monkeypatch.setenv("OUROBOROS_CONSCIOUSNESS_MAX_TASKS", "2")
    wake_task = {"id": "c-2", "root_task_id": "c-2",
                 "metadata": {"initiator": "consciousness", "usage_category": "consciousness_task"}}

    # Exhausted: the spend is real, and a zero-cost route does not reopen the door.
    monkeypatch.setenv("OUROBOROS_CONSCIOUSNESS_DAILY_USD", "3")
    _wake(data_root, monkeypatch, T0 - HOUR, 3.5)
    _subscription(data_root, monkeypatch, T0 - HOUR, session="sess-free-2", spend=0.0,
                  task="c-2", root_task="c-2")
    monkeypatch.setattr(allowance.time, "time", lambda: T0)
    reason, detail = queue._consciousness_admission_block(wake_task)
    assert reason == "consciousness_allowance_exhausted" and "3.50" in detail

    # Zero allowance is a real owner choice, stated as one.
    monkeypatch.setenv("OUROBOROS_CONSCIOUSNESS_DAILY_USD", "0")
    reason, detail = queue._consciousness_admission_block(wake_task)
    assert reason == "consciousness_allowance_exhausted" and "DAILY_USD=0" in detail


def test_an_unreadable_ledger_refuses_the_start_instead_of_assuming_free(data_root, monkeypatch):
    monkeypatch.setenv("OUROBOROS_CONSCIOUSNESS_DAILY_USD", "20")

    def _boom(*_args, **_kwargs):
        raise OSError("ledger unreadable")

    monkeypatch.setattr(usage_store, "read", _boom)
    window = allowance.allowance_window(data_root, now=T0)
    assert window["status"] == allowance.STATUS_UNKNOWN
    assert window["accounted_usd"] is None and window["remaining_usd"] is None


def test_the_wake_prompt_money_framing_matches_what_the_ledger_actually_records():
    """The prompt's three money words are the reducer's, not a separate vocabulary."""
    import pathlib

    prompt = (pathlib.Path(__file__).resolve().parent.parent / "prompts" / "CONSCIOUSNESS.md").read_text(
        encoding="utf-8")
    assert "Distinguish incremental cash cost from subscription quota and time." in prompt
    assert "A confirmed zero-cost call spends no cash; an undisclosed cost is unknown." in prompt
    assert "Existing limits still apply." in prompt
    # A confirmed zero and an undisclosed price are DIFFERENT ledger rows, which is
    # exactly the distinction the sentence above asks this mind to make.
    free = {"kind": "subscription_session", "state": "settled", "cost_usd": 0.0,
            "cost_final": True, "pricing_known": True}
    undisclosed = {"kind": "subscription_session", "state": "settled", "cost_usd": None,
                   "cost_final": False, "pricing_known": False}
    from ouroboros._usage_rows import _summary

    assert _summary([free])["accounted_usd"] == 0.0 and _summary([free])["unknown_unmetered"] == 0
    assert _summary([undisclosed])["unknown_unmetered"] == 1
    # The allowance is the only budget the prompt promises; there is no quota one.
    assert set(allowance.allowance_window.__doc__.split()) >= {"allowance"}
    assert not hasattr(allowance, "quota_window")
    # Time and quota stay separate OBSERVED axes of the same reducer, never budgets.
    session = _summary([{**free, "subscription_route": "r", "subscription_reset_at": "2026-09-17T00:00:00Z"}])
    assert session["subscription_sessions"] == 1
    assert session["subscription_windows"] == {"r": "2026-09-17T00:00:00Z"}
    assert "limit_usd" not in session


@pytest.mark.parametrize(("settled", "unresolved", "status", "remaining"), [
    ((2.0, True), 20.0, allowance.STATUS_AVAILABLE, 8.0),    # $2 known beside a $20 unknown
    ((9.0, False), None, allowance.STATUS_AVAILABLE, 1.0),   # a $9 estimate is known spend
    ((10.0, False), None, allowance.STATUS_EXHAUSTED, 0.0),  # equality: the allowance is reached
])
def test_the_daily_allowance_decides_on_known_spend(data_root, monkeypatch, settled, unresolved, status, remaining):
    """The same rule as every money limit (#1487): known spend (estimates included)
    decides; an unresolved bound is disclosed exposure."""
    monkeypatch.setenv("OUROBOROS_CONSCIOUSNESS_DAILY_USD", "10")
    cost, final = settled
    _at(monkeypatch, T0 - 2 * HOUR)
    _settle(data_root, cost=cost, cost_final=final, task_id="wake-1", root_task_id="wake-1", category="consciousness")
    if unresolved is not None:
        _at(monkeypatch, T0 - HOUR)
        held = ua.reserve_attempt(_request(data_root, reservation_usd=unresolved, task_id="wake-1",
                                           root_task_id="wake-1", category="consciousness"))
        ua.mark_dispatched(held)
        ua.mark_unresolved(held, "provider outcome unknown")
    window = allowance.allowance_window(data_root, now=T0)
    assert window["status"] == status and window["remaining_usd"] == pytest.approx(remaining)
    assert window["settled_usd"] == pytest.approx(cost)
    assert window["accounted_usd"] == pytest.approx(cost + (unresolved or 0.0))
