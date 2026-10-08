"""Owner Batch4 (2A): whole-work money across Continue, through the REAL ledger.

The original root P, its Continue successor S and S's helper T share ONE
billing group under P's original cap. Every reservation — P's own late review
included — is checked against the same locked snapshot; legacy rows of P join
by P's root id without a rewrite; compaction, the tree-accounting cache the
pacing reads, the review wave and the Resume grant's strict read see the
group; a late charge establishes an overrun it did not prevent.
"""

from __future__ import annotations


import pytest

from ouroboros import usage_accounting as ua
from tests._usage_store_testing import ledger_rows, write_compacted_journal, write_journal


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    root = tmp_path / "data"
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(root))
    monkeypatch.setenv("OUROBOROS_SETTINGS_PATH", str(root / "settings.json"))
    monkeypatch.setenv("TOTAL_BUDGET", "1000")
    (root / "state").mkdir(parents=True)
    ua._reset_task_cache_splits()
    return root


def _scope(root, task_id, root_task_id, *, group="", group_limit=None, root_limit=None, **extra):
    return ua.UsageScope(drive_root=root, task_id=task_id, root_task_id=root_task_id, source="test",
                         root_limit_usd=root_limit, billing_group_id=group, billing_group_limit_usd=group_limit,
                         **extra)


def _spend(root, scope, cost, *, bound=None, final=True):
    with ua.usage_scope(scope):
        reservation = ua.reserve_attempt(ua.AttemptRequest(
            model="openai/gpt-5.2", provider="openai", reservation_usd=cost if bound is None else bound))
        ua.mark_dispatched(reservation)
        if cost is not None:
            ua.settle_attempt(reservation, {"prompt_tokens": 1, "completion_tokens": 1},
                              cost_usd=cost, cost_final=final)
        return reservation


P = dict(group="P", group_limit=20.0, root_limit=20.0)
S = dict(group="P", group_limit=20.0)          # a Continue's successor: no fresh cap of its own
T = dict(group="P", group_limit=20.0)          # the successor's helper


def test_every_member_sees_every_other_including_the_original_roots_late_review(data_root):
    _spend(data_root, _scope(data_root, "P", "P", **P), 8.0)
    _spend(data_root, _scope(data_root, "S", "S", **S), 10.0)
    # Known $18 of $20: P's late review is admitted although its own $4 bound would
    # pass the cap (#1487: a bound is exposure, not spending). It settles at $2.
    _spend(data_root, _scope(data_root, "P-review", "P", category="review", **P), 2.0, bound=4.0)
    # Known spend reached the cap: P's own later work, S and S's helper T are all refused.
    with pytest.raises(ua.BudgetExceeded, match="whole-work budget exhausted for group P"):
        _spend(data_root, _scope(data_root, "P-review", "P", category="review", **P), 0.5)
    with pytest.raises(ua.BudgetExceeded, match=r"known=\$20\.000000"):
        _spend(data_root, _scope(data_root, "T", "S", parent_task_id="S", **T), 0.5)
    with pytest.raises(ua.BudgetExceeded):
        _spend(data_root, _scope(data_root, "S", "S", **S), 1.0)
    projection = ua.usage_projection(data_root, billing_group_id="P")
    assert projection["settled_usd"] == pytest.approx(20.0) and projection["limit_usd"] == 20.0
    assert projection["remaining_known_usd"] == 0.0
    # Genuine ids stay genuine: each row names its own task and root.
    rows = ledger_rows(data_root)  # one current row per admitted attempt; refusals left none
    attributed = {(r["task_id"], r["root_task_id"], r["billing_group_id"]) for r in rows}
    assert attributed == {("P", "P", "P"), ("S", "S", "P"), ("P-review", "P", "P")}
    assert ua.usage_projection(data_root, root_task_id="S")["accounted_usd"] == pytest.approx(10.0)


def test_legacy_rows_of_the_original_root_join_the_group_without_a_rewrite(data_root):
    legacy = {"kind": "attempt", "attempt_id": "legacy-1", "model": "m", "provider": "openai",
              "task_id": "P", "root_task_id": "P", "parent_task_id": "", "category": "task", "source": "old",
              "reservation_upper_bound_usd": 15.0, "pricing_known": True, "root_limit_usd": 20.0}
    # Journaled before the group field existed: no ``billing_group_id`` anywhere;
    # the store's one-time import carries the row over as it is.
    write_journal(data_root, [{**legacy, "state": "reserved"}, {**legacy, "state": "dispatched"},
                              {**legacy, "state": "settled", "cost_usd": 20.0, "cost_final": True}])
    with pytest.raises(ua.BudgetExceeded):  # P's known $20 legacy spend binds its successor
        _spend(data_root, _scope(data_root, "S", "S", **S), 0.5)
    [imported] = [row for row in ledger_rows(data_root) if row["attempt_id"] == "legacy-1"]
    assert "billing_group_id" not in imported, "history is never rewritten"
    from ouroboros.usage_admission import original_group_limit

    assert original_group_limit(data_root, "P") == {"limit_usd": 20.0, "source": "ledger_first_row"}


def test_a_late_charge_establishes_an_overrun_it_did_not_prevent(data_root):
    """An unknown-priced reservation is admitted while the known accounting fits
    (the existing policy, unchanged). Its late settlement past the cap is
    recorded as an overrun and bars the next admission."""
    _spend(data_root, _scope(data_root, "P", "P", **P), 18.0)
    with ua.usage_scope(_scope(data_root, "S", "S", **S)):
        unknown = ua.reserve_attempt(ua.AttemptRequest(model="m", provider="openai", force_unknown_reservation=True))
        ua.mark_dispatched(unknown)
        ua.settle_attempt(unknown, {"prompt_tokens": 1, "completion_tokens": 1}, cost_usd=5.0, cost_final=True)
    projection = ua.usage_projection(data_root, billing_group_id="P")
    assert projection["accounted_usd"] == pytest.approx(23.0)
    assert projection["accounted_usd"] > projection["limit_usd"] == 20.0
    with pytest.raises(ua.BudgetExceeded):
        _spend(data_root, _scope(data_root, "P-post", "P", **P), 0.01)


def test_an_imported_aggregate_keeps_the_group_and_the_pacing_cache_reads_it(data_root):
    from ouroboros.loop_budget import _loop_tree_accounting
    from ouroboros.usage_admission import original_group_limit

    common = dict(model="openai/gpt-5.2", provider="openai", category="task", source="test",
                  billing_group_id="P", billing_group_limit_usd=20.0, pricing_known=True, cost_final=True)
    # What the retired compaction folded 16 of P's and 10 of S's settled attempts into.
    write_compacted_journal(data_root, [
        {**common, "attempt_id": "fold-P", "task_id": "P", "root_task_id": "P", "root_limit_usd": 20.0,
         "folded_attempt_count": 16, "cost_usd": 12.0, "reservation_upper_bound_usd": 12.0},
        {**common, "attempt_id": "fold-S", "task_id": "S", "root_task_id": "S",
         "folded_attempt_count": 10, "cost_usd": 8.0, "reservation_upper_bound_usd": 8.0},
    ])
    assert original_group_limit(data_root, "P")["limit_usd"] == 20.0
    aggregates = [row for row in ledger_rows(data_root) if row.get("kind") == "usage_baseline_group"]
    assert [(row["billing_group_id"], row["billing_group_limit_usd"]) for row in aggregates] == [("P", 20.0)] * 2
    projection = ua.usage_projection(data_root, billing_group_id="P")
    assert projection["accounted_usd"] == pytest.approx(20.0)
    assert projection["attempt_counts"] == {"settled": 26}  # counts weighted by the folded attempts
    with ua.usage_scope(_scope(data_root, "S", "S", **S)):
        tree = _loop_tree_accounting(refresh=True, max_age_sec=0.0, strict=True)
    assert tree["settled_usd"] == pytest.approx(20.0) and tree["root_limit_usd"] == 20.0
    with pytest.raises(ua.BudgetExceeded):
        _spend(data_root, _scope(data_root, "S", "S", **S), 0.5)


def test_the_review_wave_binds_on_the_group_when_it_is_tighter(data_root):
    _spend(data_root, _scope(data_root, "P", "P", **P), 17.0)
    with ua.usage_scope(_scope(data_root, "S", "S", **S)):
        wave = ua.review_wave_admission(data_root, root_task_id="S", models=["openai/gpt-5.2"],
                                        prompt_chars=4000, max_completion_tokens=1000)
    assert wave["binding_axis"] == "group" and wave["remaining_usd"] == pytest.approx(3.0)
    assert wave["billing_group_id"] == "P"


def test_task_start_fields_for_the_successor_its_helper_and_an_unreadable_root(data_root):
    from ouroboros.task_results import write_task_result
    from ouroboros.usage_admission import UNAVAILABLE_GROUP_PREFIX, task_billing_fields

    successor = {"id": "S", "metadata": {"continuation": {"billing_group_id": "P", "billing_group_limit_usd": 20.0}}}
    assert task_billing_fields(successor, "S", 5.0, data_root) == {
        "root_limit_usd": 5.0, "billing_group_id": "P", "billing_group_limit_usd": 20.0,
        "billing_group_limit_source": None, "billing_group_limit_revision": None}
    write_task_result(data_root, "S", "running", metadata=successor["metadata"])
    helper = {"id": "T", "root_task_id": "S", "parent_task_id": "S"}
    assert task_billing_fields(helper, "S", 5.0, data_root)["billing_group_id"] == "P"
    plain = {"id": "R"}
    assert task_billing_fields(plain, "R", 5.0, data_root) == {
        "root_limit_usd": 5.0, "billing_group_id": "R", "billing_group_limit_usd": 5.0}
    (data_root / "task_results" / "S.json").write_text("{not json", encoding="utf-8")
    fields = task_billing_fields(helper, "S", 5.0, data_root)
    assert fields["billing_group_id"].startswith(UNAVAILABLE_GROUP_PREFIX)
    with pytest.raises(ua.BudgetExceeded, match="billing group authority unavailable"):
        _spend(data_root, _scope(data_root, "T", "S", group=fields["billing_group_id"],
                                 group_limit=fields["billing_group_limit_usd"]), 0.01)


def test_the_continue_admission_carries_the_original_cap_and_the_resume_grant_reads_the_group(tmp_path, monkeypatch):
    from tests._budget_pause_exact_helpers import _install_queue
    from tests.test_owner_continue import NONCE, _interrupted
    from ouroboros.usage_admission import task_accounting_key

    _queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setenv("TOTAL_BUDGET", "1000")
    _spend(tmp_path, _scope(tmp_path, "pred-1", "pred-1", group="pred-1", group_limit=20.0, root_limit=20.0), 3.0)
    _interrupted(tmp_path)
    from supervisor.continuation_admission import admit_continuation

    ack = admit_continuation("pred-1", action_nonce=NONCE)
    row = next(task for task in workers.PENDING if task["id"] == ack["successor_task_id"])
    carried = row["metadata"]["continuation"]
    assert carried["billing_group_id"] == "pred-1" and carried["billing_group_limit_usd"] == 20.0
    assert carried["billing_group_limit_source"] == "ledger_first_row"
    assert task_accounting_key(tmp_path, row, row["id"]) == "group:pred-1"


def test_concurrent_siblings_are_admitted_on_known_spend_and_the_exposure_is_disclosed(data_root):
    """Owner Q4-A (#1487): S and its helper T race for the group's last $8 with $6
    bounds each. Known spend ($12) is below the cap, so BOTH are admitted: concurrent
    overshoot is accepted, never prevented by worst-case holds, and the $24 exposure
    is disclosed beside the known $12."""
    import threading

    _spend(data_root, _scope(data_root, "P", "P", **P), 12.0)
    barrier = threading.Barrier(2)
    outcomes: dict = {}

    def race(task_id, root_task_id, extra):
        scope = _scope(data_root, task_id, root_task_id, **extra, **S)
        barrier.wait()
        try:
            with ua.usage_scope(scope):
                ua.reserve_attempt(ua.AttemptRequest(model="openai/gpt-5.2", provider="openai",
                                                     reservation_usd=6.0))
            outcomes[task_id] = "admitted"
        except ua.BudgetExceeded:
            outcomes[task_id] = "refused"

    threads = [threading.Thread(target=race, args=("S", "S", {})),
               threading.Thread(target=race, args=("T", "S", {"parent_task_id": "S"}))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    assert sorted(outcomes.values()) == ["admitted", "admitted"], outcomes
    projection = ua.usage_projection(data_root, billing_group_id="P")
    assert projection["settled_usd"] == pytest.approx(12.0)
    assert projection["accounted_usd"] == pytest.approx(24.0)
    assert projection["accounted_usd"] > projection["limit_usd"] == 20.0


def test_late_session_uses_start_custody_after_root_loss_not_callers_group(data_root):
    from ouroboros.task_results import write_task_result, task_result_path
    from ouroboros.delegate_custody import record_start_requested

    binding = {"billing_group_id": "P", "billing_group_limit_usd": 20.0,
               "billing_group_limit_source": "initial_task_admission", "billing_group_limit_revision": "cap-1"}
    write_task_result(data_root, "S", "running", billing_group=binding)
    assert record_start_requested(data_root, task_id="S", root_task_id="S", invocation_id="late-start",
                                  request={"prompt": "do work"}, idempotency_key="late-start")
    task_result_path(data_root, "S").unlink()
    with ua.usage_scope(_scope(data_root, "unrelated", "unrelated", group="unrelated", group_limit=999.0)):
        ua.record_subscription_session("late-session", drive_root=data_root, route="test", model="m",
                                       task_id="S", root_task_id="S", spend_usd=20.0)
    assert ua.usage_projection(data_root, billing_group_id="P")["accounted_usd"] == pytest.approx(20.0)
    assert ua.usage_projection(data_root, root_task_id="S")["accounted_usd"] == pytest.approx(20.0)
    assert ua.usage_projection(data_root, billing_group_id="unrelated")["accounted_usd"] == 0.0
    with pytest.raises(ua.BudgetExceeded):
        _spend(data_root, _scope(data_root, "P", "P", **P), 1.0)


def test_sibling_reservations_share_one_atomic_known_spend(data_root):
    """Known spend exactly at the cap: the one locked check refuses BOTH racers."""
    import threading
    from concurrent.futures import ThreadPoolExecutor

    _spend(data_root, _scope(data_root, "P", "P", **P), 20.0)
    ready = threading.Barrier(2)
    def claim(tid):
        with ua.usage_scope(_scope(data_root, tid, tid, **S)):
            ready.wait(timeout=3)
            try:
                ua.reserve_attempt(ua.AttemptRequest(model="m", provider="test", reservation_usd=0.75))
                return "reserved"
            except ua.BudgetExceeded:
                return "refused"
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(claim, ["S", "T"])) == ["refused", "refused"]
    assert ua.usage_projection(data_root, billing_group_id="P")["accounted_usd"] == pytest.approx(20.0)


def test_root_cap_and_group_cap_are_both_required(data_root):
    tight = _scope(data_root, "S", "S", group="P", group_limit=20.0, root_limit=2.0)
    _spend(data_root, tight, 2.0)  # S's own known spend reaches its root cap
    with pytest.raises(ua.BudgetExceeded, match="root model budget exhausted for S"):
        _spend(data_root, tight, 0.01)  # although the group still has $18
    _spend(data_root, _scope(data_root, "P", "P", **P), 18.0)  # the group's known spend reaches $20
    with pytest.raises(ua.BudgetExceeded, match="whole-work budget exhausted for group P"):
        _spend(data_root, _scope(data_root, "S", "S", group="P", group_limit=20.0, root_limit=5.0), 0.01)


def test_explicit_other_root_cannot_inherit_the_callers_billing_group(data_root):
    from ouroboros.task_results import write_task_result

    write_task_result(data_root, "S", "running", billing_group={
        "billing_group_id": "P", "billing_group_limit_usd": 20.0,
        "billing_group_limit_source": "initial_task_admission", "billing_group_limit_revision": "r1"})
    with ua.usage_scope(_scope(data_root, "foreign", "foreign", group="foreign", group_limit=0.01)):
        reservation = ua.reserve_attempt(ua.AttemptRequest(task_id="helper", root_task_id="S",
                                          model="m", provider="test", reservation_usd=1.0))
    with ua._locked(data_root):
        row = {row['attempt_id']: row for row in ledger_rows(data_root)}[reservation.attempt_id]
    assert row["root_task_id"] == "S" and row["task_id"] == "helper"
    assert row["billing_group_id"] == "P"
    assert ua.usage_projection(data_root, billing_group_id="P")["accounted_usd"] == pytest.approx(1.0)
    assert ua.usage_projection(data_root, billing_group_id="foreign")["accounted_usd"] == 0.0
