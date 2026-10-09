"""Known spend decides every money limit (owner Q4-A 2026-10-03, #1487).

Known spend is the settled bucket: confirmed prices AND disclosed estimates. A
reservation or the upper bound of an unresolved call is exposure, shown beside
it and never counted as spending. The global, root and original-group axes use
ONE predicate at reservation and again at dispatch: refuse once known spend has
reached the limit (equality refuses). Concurrent, in-flight and late charges can
still overshoot; nothing here promises a maximum. Every case runs the real
``reserve_attempt`` / ``mark_dispatched`` over the real usage store.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from ouroboros import usage_accounting as ua
from tests import fixtures_usage_store as _fixtures
from tests._usage_store_testing import ledger_rows

data_root = _fixtures.data_root

AXES = ("global", "root", "group")


def _scope(root, axis, task_id="t", root_task_id="t"):
    """A scope whose ONLY finite $10 limit is ``axis``."""
    common = dict(drive_root=root, task_id=task_id, root_task_id=root_task_id, source="test")
    if axis == "global":
        return ua.UsageScope(global_limit_usd=10.0, **common)
    if axis == "root":
        return ua.UsageScope(global_limit_usd=1000.0, root_limit_usd=10.0, **common)
    return ua.UsageScope(global_limit_usd=1000.0, billing_group_id="G", billing_group_limit_usd=10.0,
                         billing_group_limit_source="fixture", **common)


def _neighbour(root, axis):
    """Another worker on the same axis: another root (global), another member (root), a Continue (group)."""
    if axis == "global":
        return _scope(root, axis, "o", "o")
    if axis == "root":
        return _scope(root, axis, "t2", "t")
    return _scope(root, axis, "s", "s")


def _request(amount=1.0):
    return ua.AttemptRequest(model="openai/gpt-5.2", provider="openai", reservation_usd=amount)


def _settled(scope, cost, *, final=True):
    with ua.usage_scope(scope):
        held = ua.reserve_attempt(_request(cost))
        ua.mark_dispatched(held)
        ua.settle_attempt(held, {"prompt_tokens": 1, "completion_tokens": 1}, cost_usd=cost, cost_final=final)


def _unresolved(scope, bound):
    with ua.usage_scope(scope):
        held = ua.reserve_attempt(_request(bound))
        ua.mark_dispatched(held)
        ua.mark_unresolved(held, "provider outcome unknown")


def _state(root, attempt_id):
    return {row["attempt_id"]: row for row in ledger_rows(root)}[attempt_id]["state"]


@pytest.mark.parametrize("axis", AXES)
def test_two_dollars_known_beside_a_twenty_dollar_unknown_admits_under_ten(data_root, axis):
    scope = _scope(data_root, axis)
    _settled(scope, 2.0)
    _unresolved(scope, 20.0)
    with ua.usage_scope(scope):
        held = ua.reserve_attempt(_request(5.0))
        ua.mark_dispatched(held)  # the dispatch seam admits on the same known $2
    assert _state(data_root, held.attempt_id) == "dispatched"


@pytest.mark.parametrize("axis", AXES)
def test_nine_estimated_and_two_confirmed_refuse_ten(data_root, axis):
    """Estimates are known spend: known-only is not confirmed-only."""
    scope = _scope(data_root, axis)
    _settled(scope, 9.0, final=False)
    _settled(scope, 2.0, final=True)
    with ua.usage_scope(scope), pytest.raises(ua.BudgetExceeded, match=r"known=\$11\.000000") as refused:
        ua.reserve_attempt(_request(0.01))
    assert refused.value.limit_scope == ("global" if axis == "global" else "root")


@pytest.mark.parametrize("axis", AXES)
def test_known_spend_exactly_at_the_limit_refuses_both_seams(data_root, axis):
    scope = _scope(data_root, axis)
    with ua.usage_scope(scope):
        held = ua.reserve_attempt(_request(0.5))  # reserved while known is $0
    _settled(_neighbour(data_root, axis), 10.0)  # known spend reaches exactly $10
    with ua.usage_scope(scope):
        with pytest.raises(ua.BudgetExceeded):
            ua.reserve_attempt(_request(0.01))
        with pytest.raises(ua.BudgetExceeded, match="changed before dispatch|exhausted"):
            ua.mark_dispatched(held)
    # The crossing between reserve and dispatch never sent: the row is still only reserved.
    assert _state(data_root, held.attempt_id) == "reserved"
    ua.release_attempt(held, "before_dispatch_failed:limit")


@pytest.mark.parametrize("axis", AXES)
def test_a_crossing_between_reserve_and_dispatch_never_sends(data_root, axis, monkeypatch):
    sends, crossed = [], []
    scope = _scope(data_root, axis)
    _settled(scope, 9.99)
    original_dispatch = ua.mark_dispatched

    def cross_then_dispatch(reservation, **kwargs):
        if not crossed:  # once: the neighbour's own settlement dispatches normally
            crossed.append(True)
            _settled(_neighbour(data_root, axis), 0.01)  # lands after the reservation, before the send
        return original_dispatch(reservation, **kwargs)

    monkeypatch.setattr(ua, "mark_dispatched", cross_then_dispatch)
    with ua.usage_scope(scope), pytest.raises(ua.BudgetExceeded):
        ua.execute_physical_attempt(_request(0.5), lambda: sends.append("sent"))
    assert crossed and sends == []


@pytest.mark.parametrize("axis", AXES)
def test_a_late_charge_past_the_limit_is_recorded_not_prevented(data_root, axis):
    """Owner Q4-A accepts overshoot: two in-flight calls admitted at $9 settle to $13.
    The overrun is recorded, and the next admission is refused on it."""
    scope = _scope(data_root, axis)
    _settled(scope, 9.0)
    with ua.usage_scope(scope):
        first, second = ua.reserve_attempt(_request(2.0)), ua.reserve_attempt(_request(2.0))
        ua.mark_dispatched(first)
        ua.mark_dispatched(second)  # both in flight at known $9
        for held in (first, second):
            ua.settle_attempt(held, {"prompt_tokens": 1, "completion_tokens": 1}, cost_usd=2.0, cost_final=True)
        with pytest.raises(ua.BudgetExceeded):
            ua.reserve_attempt(_request(0.01))
    if axis == "group":
        projection = ua.usage_projection(data_root, billing_group_id="G")
    elif axis == "root":
        projection = ua.usage_projection(data_root, root_task_id="t")
    else:
        projection = ua.usage_projection(data_root, global_limit_usd=10.0)
    assert projection["settled_usd"] == pytest.approx(13.0)


def test_a_zero_limit_admits_nothing_and_no_limit_admits_on_any_exposure(data_root, monkeypatch):
    with ua.usage_scope(replace(_scope(data_root, "global"), global_limit_usd=0.0)):
        with pytest.raises(ua.BudgetExceeded):
            ua.reserve_attempt(_request(0.01))
    with ua.usage_scope(replace(_scope(data_root, "root"), root_limit_usd=0.0)):
        with pytest.raises(ua.BudgetExceeded):
            ua.reserve_attempt(_request(0.01))
    monkeypatch.setenv("TOTAL_BUDGET", "0")  # no finite global budget
    unlimited = ua.UsageScope(drive_root=data_root, task_id="u", root_task_id="u", source="test")
    _unresolved(unlimited, 500.0)
    _settled(unlimited, 400.0)
    with ua.usage_scope(unlimited):
        ua.release_attempt(ua.reserve_attempt(_request(1.0)))


def test_unavailable_group_authority_is_refused_never_a_healthy_empty_store(data_root):
    """Unknown price is not a zero, and missing group authority is not an empty group."""
    scope = ua.UsageScope(drive_root=data_root, task_id="t", root_task_id="t", global_limit_usd=1000.0,
                          billing_group_id="unavailable:t", billing_group_limit_usd=0.0)
    with ua.usage_scope(scope), pytest.raises(ua.BudgetExceeded, match="billing group authority unavailable"):
        ua.reserve_attempt(_request(0.01))


def test_remaining_known_usd_is_one_number_at_every_producer(data_root, monkeypatch):
    """The field keeps its name and gains its meaning: limit minus KNOWN spend, at
    the ledger projection, the root projection and the Costs page alike."""
    import asyncio
    import json

    from ouroboros.gateway.history import make_cost_breakdown_endpoint
    from supervisor import state as supervisor_state

    scope = ua.UsageScope(drive_root=data_root, task_id="t", root_task_id="t", global_limit_usd=10.0,
                          root_limit_usd=10.0)
    _settled(scope, 2.0)
    _unresolved(scope, 20.0)
    monkeypatch.setattr(supervisor_state, "TOTAL_BUDGET_LIMIT", 10.0)
    global_view = ua.usage_projection(data_root, global_limit_usd=10.0)
    root_view = ua.usage_projection(data_root, root_task_id="t")
    costs = json.loads(asyncio.run(make_cost_breakdown_endpoint(data_root)(None)).body)["accounting"]
    for view in (global_view, root_view, costs):
        assert view["remaining_known_usd"] == 8.0
        assert view["settled_usd"] == 2.0 and view["accounted_usd"] == 22.0  # exposure kept beside it


def test_an_ineligible_actor_is_stopped_by_the_fence_with_no_funded_recap(data_root, monkeypatch, tmp_path):
    """An actor without an exact continuation keeps the existing priced wrap-up rail
    (not redesigned here), but that rail cannot cross the physical money fence: with
    known spend at the cap, the forced send's own reservation is refused, nothing is
    sent, and the refusal ends the task in the typed terminal projection without any
    model-written recap being bought."""
    from types import SimpleNamespace

    from ouroboros import loop, task_pacing
    from ouroboros.contracts.task_contract import normalize_budget_profile
    from tests._budget_limits_helpers import _make_args

    sends = []
    scope = ua.UsageScope(drive_root=data_root, task_id="ineligible", root_task_id="ineligible",
                          global_limit_usd=1000.0, root_limit_usd=10.0)
    _settled(scope, 10.0)  # the tree's known spend is at its $10 cap
    before = ledger_rows(data_root)

    def funded_send(*_args, **_kwargs):
        ua.execute_physical_attempt(_request(0.5), lambda: sends.append("recap"))
        return {"role": "assistant", "content": "recap"}, 0.5

    monkeypatch.setattr(loop, "call_llm_with_retry", funded_send)
    monkeypatch.setattr(loop, "_loop_tree_accounting", lambda **_k: {"settled_usd": 10.0, "accounted_usd": 10.0})
    ceiling = task_pacing.resolve_cost_ceiling(1000.0, normalize_budget_profile({"cost_hard_stop_pct": 50}),
                                               root_cap_usd=10.0)
    args = _make_args(task_id="ineligible", accumulated_usage={"cost": 10.0}, cost_ceiling=ceiling,
                      budget_remaining_usd=1000.0, drive_logs=tmp_path)
    tool_ctx = SimpleNamespace(task_id="ineligible", drive_root=data_root, budget_drive_root=data_root,
                               owner_wait_callback=None, task_metadata={}, is_direct_chat=False,
                               task_contract={"budget_profile": {"cost_hard_stop_pct": 50}})  # the profile it stops on
    args["ctx"].tools = SimpleNamespace(_ctx=tool_ctx)
    with ua.usage_scope(scope):
        with pytest.raises(ua.BudgetExceeded):
            loop._check_budget_limits(**args)  # graceful stop -> ineligible -> priced rail -> fence
    usage = args["ctx"].accumulated_usage
    assert usage["exact_pause_unavailable"] == "no_continuation_owner"
    assert sends == [] and ledger_rows(data_root) == before  # no recap reserved, none sent

    exit_ctx = loop._LoopExitContext(
        tools=SimpleNamespace(_ctx=tool_ctx), drive_root=data_root, task_id="ineligible", event_queue=None,
        drive_logs=tmp_path, accumulated_usage=usage, llm_trace={})
    monkeypatch.setattr(loop, "_finalize_task_services", lambda _ctx: None)
    text, terminal, trace = loop._handle_budget_exceeded(
        ua.BudgetExceeded("root model budget exhausted", limit_scope="root", root_task_id="ineligible"),
        exit_ctx)
    assert terminal["execution_status"] == "failed" and terminal["reason_code"] == "budget_exhausted"
    assert trace["resource_limit"]["status"] == "resource_limited"
    assert trace["resource_limit"]["auto_resume"] is False
    assert "Resource limit reached before another model dispatch" in text
    assert sends == [] and ledger_rows(data_root) == before


def test_an_ordinary_ineligible_actor_meets_only_the_real_cap_with_no_funded_recap(data_root, monkeypatch, tmp_path):
    """The same actor without an explicit profile (owner 2026-10-07): no default early
    stop fires before the cap, so the budget gate buys nothing; the REAL root cap
    refuses its next send at reservation once known spend reached it, and the typed
    terminal follows with no model-written recap bought on the way out."""
    from types import SimpleNamespace

    from ouroboros import loop, task_pacing
    from ouroboros.contracts.task_contract import normalize_budget_profile
    from tests._budget_limits_helpers import _make_args

    sends = []
    scope = ua.UsageScope(drive_root=data_root, task_id="ordinary", root_task_id="ordinary",
                          global_limit_usd=1000.0, root_limit_usd=10.0)
    _settled(scope, 10.0)  # known spend is at the $10 cap
    before = ledger_rows(data_root)
    monkeypatch.setattr(loop, "call_llm_with_retry", lambda *_a, **_k: pytest.fail("no recap may be bought"))
    ceiling = task_pacing.resolve_cost_ceiling(1000.0, normalize_budget_profile(None), root_cap_usd=10.0)
    assert ceiling.state == task_pacing.COST_CEILING_DISABLED and ceiling.basis == task_pacing.COST_BASIS_NO_DEFAULT_STOP
    args = _make_args(task_id="ordinary", accumulated_usage={"cost": 10.0}, cost_ceiling=ceiling,
                      budget_remaining_usd=1000.0, drive_logs=tmp_path)
    tool_ctx = SimpleNamespace(task_id="ordinary", drive_root=data_root, budget_drive_root=data_root,
                               owner_wait_callback=None, task_metadata={}, is_direct_chat=False)
    args["ctx"].tools = SimpleNamespace(_ctx=tool_ctx)
    with ua.usage_scope(scope):
        assert loop._check_budget_limits(**args) is None  # no host stop of its own
        with pytest.raises(ua.BudgetExceeded) as refused:  # the next ordinary send meets the real cap
            ua.execute_physical_attempt(_request(0.5), lambda: sends.append("work"))
    assert refused.value.limit_scope == "root" and "known=$10.000000" in str(refused.value)
    assert sends == [] and ledger_rows(data_root) == before

    exit_ctx = loop._LoopExitContext(
        tools=SimpleNamespace(_ctx=tool_ctx), drive_root=data_root, task_id="ordinary", event_queue=None,
        drive_logs=tmp_path, accumulated_usage=args["ctx"].accumulated_usage, llm_trace={})
    monkeypatch.setattr(loop, "_finalize_task_services", lambda _ctx: None)
    text, terminal, trace = loop._handle_budget_exceeded(refused.value, exit_ctx)
    assert terminal["execution_status"] == "failed" and terminal["reason_code"] == "budget_exhausted"
    assert trace["resource_limit"]["scope"] == "root" and trace["resource_limit"]["auto_resume"] is False
    assert "Resource limit reached before another model dispatch" in text
    assert sends == [] and ledger_rows(data_root) == before


@pytest.mark.parametrize(("author", "known", "stops"), [
    ("producer", 8.99, False), ("producer", 9.0, True), ("producer", 9.5, True),
    ("explicit", 9.0, False), ("explicit", 9.01, True),
    ("unverified", 9.0, False), ("unverified", 9.5, True),
], ids=["producer-below", "producer-equal", "producer-above", "explicit-equal", "explicit-above",
        "unverified-equal", "unverified-above"])
def test_the_budget_gate_stops_a_producer_allowance_once_known_spend_reaches_it(
        data_root, monkeypatch, tmp_path, author, known, stops):
    """A wake's $9 daily remainder is a real limit on known spend, so, like the ledger
    caps, reaching it stops the task (owner 2026-10-07). An explicit profile's $9 point
    keeps its authored edge: only spend over it stops. Real ledger rows, the real
    task-start resolver, the real budget gate and the real forced rail; the model send
    is funded through the real ledger fence.

    Only an explicit profile buys the final answer it authored room for. The fence
    checks the $50 tree cap and the $1000 wallet, never the $9 allowance, so a recap
    after the allowance (or an inherited number whose author is unreadable, kept and
    never widened) would be paid past it: that actor ends on the host's own notice."""
    from types import SimpleNamespace

    from ouroboros import loop, task_pacing
    from tests._budget_limits_helpers import _make_args

    sends = []

    def funded_send(*_args, **_kwargs):
        ua.execute_physical_attempt(_request(0.5), lambda: sends.append("recap"))
        return {"role": "assistant", "content": "recap"}, 0.5

    monkeypatch.setattr(loop, "call_llm_with_retry", funded_send)
    task_id = f"gate-{author}-{known}"
    profile = {"cost_hard_stop_pct": 50} if author == "explicit" else None
    root_id = task_id if author != "unverified" else f"lost-root-{known}"  # a root with no readable row
    scope = ua.UsageScope(drive_root=data_root, task_id=task_id, root_task_id=root_id, global_limit_usd=1000.0, **(
        {} if author == "explicit" else {"root_limit_usd": 50.0, "root_cost_ceiling_usd": 9.0}),
        **({"parent_task_id": root_id} if author == "unverified" else {}))
    # The tree's known spend: the member's root spent it before the member's check.
    _settled(scope if author != "unverified" else ua.UsageScope(
        drive_root=data_root, task_id=root_id, root_task_id=root_id, global_limit_usd=1000.0,
        root_limit_usd=50.0), known)
    before = ledger_rows(data_root)
    tool_ctx = SimpleNamespace(task_id=task_id, drive_root=data_root, budget_drive_root=data_root,
                               owner_wait_callback=None, task_metadata={}, is_direct_chat=False,
                               task_contract={"budget_profile": profile} if profile else None)
    with ua.usage_scope(scope):
        ceiling = task_pacing.resolve_task_cost_ceiling(tool_ctx, 18.0)
        assert ceiling.state == task_pacing.COST_CEILING_ACTIVE and ceiling.ceiling_usd == 9.0
        assert task_pacing.cost_stop_authority(tool_ctx) == {
            "explicit": task_pacing.COST_STOP_EXPLICIT, "producer": task_pacing.COST_STOP_PRODUCER,
            "unverified": task_pacing.COST_STOP_UNKNOWN}[author]
        args = _make_args(task_id=task_id, accumulated_usage={"cost": known}, cost_ceiling=ceiling,
                          budget_remaining_usd=1000.0, drive_logs=tmp_path)
        args["ctx"].tools = SimpleNamespace(_ctx=tool_ctx)
        result = loop._check_budget_limits(**args)
    usage = args["ctx"].accumulated_usage
    if not stops:
        assert result is None and sends == [] and "exact_pause_unavailable" not in usage
        return
    assert result is not None
    assert usage["cost_stop_spend_basis"] == task_pacing.SPEND_BASIS_TREE
    assert usage["exact_pause_unavailable"] == "no_continuation_owner"  # the graceful pause rail ran
    assert usage["execution_status"] == "failed" and usage["reason_code"] == "budget_exhausted"
    if author == "explicit":
        assert sends == ["recap"] and result[0] == "recap"  # its authored final, fenced by the hard limits
        return
    assert sends == [] and ledger_rows(data_root) == before  # nothing reserved, nothing sent
    assert f"spent ${known:.3f}" in result[0] and "$9.00" in result[0]


def test_limit_texts_promise_a_saved_pause_only_where_an_exact_continuation_exists(data_root, monkeypatch):
    """The owner setting, the runtime fact and the ceiling disclosure describe the SAME
    two endings the ineligible-actor tests above establish: an exact continuation pauses
    with its work saved; an actor without one ends budget-exhausted. None promises the
    saved pause to every actor, and none grants an actor a new capability."""
    from types import SimpleNamespace

    from ouroboros import task_pacing
    from ouroboros.context_runtime_facts import _runtime_budget_info
    from ouroboros.settings_setup_contract import build_setup_contract

    monkeypatch.setenv("OUROBOROS_PER_TASK_COST_USD", "10")
    fields = {str(field.get("settingKey")): field for field in build_setup_contract()["budgetFields"]}
    texts = {
        "setting": str(fields["OUROBOROS_PER_TASK_COST_USD"]["note"]),
        "runtime": _runtime_budget_info(SimpleNamespace(drive_root=data_root), {})["per_task_tree_cap_rule"],
        "disclosure": task_pacing.cost_ceiling_disclosure(task_pacing.resolve_cost_ceiling(
            1000.0, {"cost_hard_stop_pct": None}, root_cap_usd=10.0))["rule"],
    }
    for surface, text in texts.items():
        assert "pauses with its work saved" in text, surface
        assert "the task pauses" not in text and "the task then pauses" not in text, surface
        assert "budget-exhausted" in text or "budget_exhausted" in text, surface
    assert "no paid recap" in texts["runtime"]  # at the hard cap the fence refuses the recap too
