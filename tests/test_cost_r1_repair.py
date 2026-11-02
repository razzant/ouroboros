"""Consumer controls for exact subtotals and bounded custody publication."""
from __future__ import annotations

import json
from decimal import Decimal
from typing import get_type_hints

import pytest

from ouroboros import delegate_custody as custody
from ouroboros import delegate_custody_current as current
from ouroboros import delegate_terminal, store, task_results, usage_store
from ouroboros import usage_accounting as usage
from supervisor.events_task_done import _refresh_terminal_task_cost
from supervisor.state import reconstruct_task_cost


def _display(root, **kwargs):
    # Exercise both R1's implicit coverage API and the repaired explicit path.
    from inspect import signature
    if "include_custody" in signature(usage.usage_breakdown).parameters:
        kwargs["include_custody"] = True
    return usage.usage_breakdown(root, **kwargs)


def _start(root, name="repair-run", task="repair-task"):
    entry = custody.RunCustody(run_id=name, task_id=task, root_task_id=task, route_id="test")
    assert custody.record_started(root, entry)
    assert not entry.settled and not entry.ledger_recorded
    return entry


def _cash(knowledge="exact", partial=True):
    return [{"usageCost": {"cashUsd": 2, "cashKnowledge": knowledge}}, *(
        [{"usageCost": {"cashUsd": None, "cashKnowledge": "unknown"}}] if partial else [])]


@pytest.mark.parametrize("knowledge,partial,confirmed,estimated", [
    ("exact", True, 2, 0), ("estimated", True, 0, 2),
    ("exact", False, 2, 0), ("estimated", False, 0, 2),
])
def test_partial_cash_summary_retains_precision_and_one_shot_identity(tmp_path, knowledge, partial, confirmed, estimated):
    kwargs = dict(drive_root=tmp_path, route="test", task_id="repair-task", root_task_id="repair-task")
    attempt = usage.record_subscription_session("repair-cash", attempt_execution=_cash(knowledge, partial), **kwargs)
    before = usage.read_usage_records(tmp_path)
    # A repeat with new model/price disclosure never reprices an existing receipt.
    assert usage.record_subscription_session("repair-cash", spend_usd=999, model="later", **kwargs) == attempt
    assert usage.read_usage_records(tmp_path) == before
    view = usage.usage_breakdown(tmp_path, task_id="repair-task")
    assert (view["settled_usd"], view["confirmed_usd"], view["estimated_usd"]) == (2, confirmed, estimated)
    assert view["cost_final"] is (knowledge == "exact" and not partial)
    assert view["unknown_unmetered"] == int(partial)
    if partial:
        assert before[0]["reservation_upper_bound_usd"] is None
    for bucket in (*view["by_task"].values(), *view["by_root"].values()):
        assert (bucket["confirmed_usd"], bucket["estimated_usd"]) == (confirmed, estimated)


@pytest.mark.parametrize("row,confirmed,estimated", [
    ({"cost_usd": True, "cost_final": True}, 1, 0),
    ({"cost_usd": "2", "cost_final": False}, 0, 2),
    ({"cost_usd": "2", "cost_final": False, "cost_estimated": False}, 0, 2),
    ({"cost_usd": "2", "cost_final": False, "cost_partial": True}, 0, 2),
    ({"cost_usd": "2", "cost_final": False, "cost_partial": True, "cost_estimated": False}, 2, 0),
])
def test_legacy_cash_grammar_is_not_provider_revalidation(row, confirmed, estimated):
    from ouroboros._usage_money import cash_contribution

    cash = cash_contribution({"kind": "subscription_session", "state": "settled", **row})
    assert cash[1:3] == (Decimal(confirmed), Decimal(estimated))


def test_amount_only_readers_never_wait_for_custody(tmp_path, monkeypatch):
    from ouroboros import delegate_custody_usage

    _start(tmp_path)
    usage.record_subscription_session("repair-paid", drive_root=tmp_path, route="test",
                                      task_id="repair-task", spend_usd=2)
    monkeypatch.setattr(delegate_custody_usage, "cost_snapshot", lambda *a, **kw: pytest.fail("amount reader entered custody"))
    assert usage.usage_breakdown(tmp_path, task_id="repair-task")["settled_usd"] == 2
    assert usage.usage_projection(tmp_path, root_task_id="repair-task")["settled_usd"] == 2
    assert usage.refresh_root_accounting(tmp_path, "repair-task", strict=True)["settled_usd"] == 2


def test_display_joins_each_receipt_once_for_own_and_tree(tmp_path, monkeypatch):
    from ouroboros.delegate_collection import record_terminal_money

    entry = _start(tmp_path)
    record_terminal_money(tmp_path, entry, {"summary": {"state": "succeeded", "spendUsd": 2}})
    seen, original = [], usage_store.Txn.attempt
    def read(txn, identity):
        seen.append(identity)
        return original(txn, identity)
    monkeypatch.setattr(usage_store.Txn, "attempt", read)
    result = _display(tmp_path, root_task_id=entry.task_id)
    assert result["cost_final"] and result["settled_usd"] == 2
    assert len(seen) == 1
    assert set(result["cost_custody"]) == {"status", "open_run_ids", "unrecorded_run_ids", "pending_invocation_ids"}


def test_no_debt_kill_refresh_does_not_read_money_write_or_emit(tmp_path, monkeypatch):
    # A legacy terminal without the optional field, and no delegated run.
    fields = reconstruct_task_cost("repair-idle", fields=True, drive_root=tmp_path)
    fields.pop("cost_custody", None)
    task_results.write_task_result(tmp_path, "repair-idle", "completed", result="answer", **fields)
    before = task_results.load_task_result(tmp_path, "repair-idle")
    events = tmp_path / "logs/events.jsonl"
    prior = events.read_bytes() if events.exists() else b""
    monkeypatch.setattr(usage_store, "read", lambda *a, **kw: pytest.fail("no cost debt"))
    assert not delegate_terminal.refresh_terminal_reconciliation(tmp_path, "repair-idle", trigger="kill_path_clear")
    assert task_results.load_task_result(tmp_path, "repair-idle") == before
    assert (events.read_bytes() if events.exists() else b"") == prior


def test_new_optional_coverage_alone_is_not_a_monetary_change(tmp_path):
    fields = reconstruct_task_cost("repair-legacy", fields=True, drive_root=tmp_path)
    fields.update(accounted_upper_bound_usd_with_children=0, cost_with_children_partial=False)
    fields["cost_presentation"]["scope"] = "root_tree"
    fields.pop("cost_custody", None)
    task_results.write_task_result(tmp_path, "repair-legacy", "completed", result="answer", **fields)
    before = task_results.load_task_result(tmp_path, "repair-legacy")
    assert _refresh_terminal_task_cost(tmp_path, "repair-legacy")
    assert task_results.load_task_result(tmp_path, "repair-legacy") == before
    events = (tmp_path / "logs/events.jsonl").read_text() if (tmp_path / "logs/events.jsonl").exists() else ""
    assert not any(json.loads(line).get("type") == "task_cost_finalized" for line in events.splitlines())


def test_refresh_uses_actual_root_and_retires_only_published_revision(tmp_path, monkeypatch):
    from supervisor import queue

    task_results.write_task_result(tmp_path, "repair-task", "completed", result="answer")
    _start(tmp_path)
    foreign = tmp_path / "unrelated-queue"
    (foreign / "state").mkdir(parents=True)
    (foreign / "state/current.sqlite").write_bytes(b"unreadable unrelated store")
    monkeypatch.setattr(queue, "DRIVE_ROOT", foreign)
    delegate_terminal._refresh_custody_cost(tmp_path, "repair-task")
    result = task_results.load_task_result(tmp_path, "repair-task")
    assert result["cost_final"] is False
    assert result["cost_custody"]["open_run_ids"] == ["repair-run"]
    with store.open_store(tmp_path, write=False) as txn:
        assert not (txn.get_row("delegated_runs", "repair-task") or {}).get("cost_revision")


@pytest.mark.parametrize("history", [5, 250])
def test_fold_import_rebuild_do_not_owe_closed_history(tmp_path, history):
    from ouroboros import store_import

    rows = []
    for i in range(history):
        for kind in (custody.STARTED, custody.LEDGER_RECORDED, custody.SETTLED):
            rows.append({"event_id": f"closed-{i}-{kind}", "type": kind, "run_id": f"closed-{i}",
                         "task_id": f"old-{i}", "root_task_id": f"old-{i}", "state": "succeeded"})
    rows.extend([
        {"event_id": "live", "type": custody.STARTED, "run_id": "open-run", "task_id": "child", "root_task_id": "root"},
        {"event_id": "pending", "type": custody.START_REQUESTED, "invocation_id": "pending", "task_id": "pending-task"},
    ])
    path = custody.event_log_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    expected = {"child", "root", "pending-task"}
    assert set(current.fold(rows)["cost_revisions"]) == expected
    _, imported = store_import._custody_rows(tmp_path, [])
    assert set(imported["delegated_runs"]) == expected
    current.rebuild(tmp_path, replace=True)
    with store.open_store(tmp_path, write=False) as txn:
        debts = txn.set_rows("delegated_runs")
    assert set(debts) == expected
    assert all(not debt.get("closed_runs") for debt in debts.values())
    # Missing owners keep the exact debt but cause no refresh attempt or events.
    assert delegate_terminal.refresh_recently_settled_terminals(tmp_path, live_task_ids=set()) == 0
    events = (tmp_path / "logs/events.jsonl").read_text() if (tmp_path / "logs/events.jsonl").exists() else ""
    assert not any(json.loads(line).get("type") == "task_cost_finalized" for line in events.splitlines())


@pytest.mark.parametrize("busy", [False, True])
def test_display_uses_display_wait_and_contention_stays_unknown_without_warning(tmp_path, monkeypatch, caplog, busy):
    from ouroboros.runtime_limits import USAGE_DISPLAY_LOCK_TIMEOUT_SEC

    usage.record_subscription_session("display-paid", drive_root=tmp_path, route="test", task_id="display", spend_usd=2)
    original, waits = current.task_snapshot, []
    def observe(*args, **kwargs):
        waits.append(kwargs["wait_sec"])
        if busy:
            raise store.StoreBusy("held for this request", reason="contention")
        return original(*args, **kwargs)
    monkeypatch.setattr(current, "task_snapshot", observe)
    fields = _display(tmp_path, task_id="display", allow_stale=True)
    assert all(0 < wait <= USAGE_DISPLAY_LOCK_TIMEOUT_SEC for wait in waits)
    assert fields["cost_final"] is (not busy) and fields["settled_usd"] == 2
    assert fields["cost_custody"]["status"] == ("unavailable" if busy else "available")
    assert not [record for record in caplog.records if record.levelname in {"WARNING", "ERROR"}]


def test_evidence_and_status_keep_partial_exact_subtotal(tmp_path):
    from ouroboros.delegate_evidence import task_execution_evidence
    from ouroboros.task_status import observe_cancellation_target

    entry = _start(tmp_path)
    custody.emit(tmp_path, custody.SETTLED, {"run_id": entry.run_id, "task_id": entry.task_id,
        "state": "succeeded", "cost_usd": 2, "spend_disclosed": True, "spend_estimated": False, "cost_partial": True})
    task_results.write_task_result(tmp_path, entry.task_id, "completed", result="answer")
    evidence = task_execution_evidence(tmp_path, entry.task_id)
    assert evidence["subscription_cost_usd"] == 2
    assert evidence["subscription_cost_partial"] and not evidence["subscription_cost_estimated"]
    observed = observe_cancellation_target(tmp_path, entry.task_id, include_execution=True)["delegated_execution"]
    assert observed["subscription_cost_usd"] == 2 and observed["subscription_cost_partial"]


def test_cost_custody_public_type_is_closed_and_nullable():
    from typing import get_args, get_origin

    from typing_extensions import NotRequired

    from ouroboros.delegate_custody_usage import CostCustody
    from ouroboros.gateway.contracts import ChatOutbound, TaskCostBreakdown

    assert set(get_type_hints(CostCustody)) == {"status", "open_run_ids", "unrecorded_run_ids", "pending_invocation_ids"}
    for shape in (ChatOutbound, TaskCostBreakdown):
        annotation = get_type_hints(shape, include_extras=True)["cost_custody"]
        # Key optionality is not nullability; Python 3.10 keeps typing_extensions' wrapper in the hints.
        if get_origin(annotation) is NotRequired:
            (annotation,) = get_args(annotation)
        assert set(get_args(annotation)) == {CostCustody, type(None)}


def test_real_name_lock_contention_is_a_display_gap_then_recovers(tmp_path, caplog):
    from tests.test_store import publish

    publish(tmp_path, tier=store.TIER_NAME)
    usage.record_subscription_session("lock-paid", drive_root=tmp_path, route="test", task_id="lock-task", spend_usd=2)
    with store.name_lock(tmp_path, store.LOCK_NAME, timeout_sec=0):
        held = _display(tmp_path, task_id="lock-task", allow_stale=True)
    assert held["cost_custody"]["status"] == "unavailable"
    assert held["settled_usd"] == 2 and not held["cost_final"]
    released = _display(tmp_path, task_id="lock-task", allow_stale=True)
    assert released["cost_custody"]["status"] == "available"
    assert released["settled_usd"] == 2 and released["cost_final"]
    assert not [r for r in caplog.records if r.levelname in {"WARNING", "ERROR"}]


def test_legacy_evidence_flags_are_quiet_until_cost_knowledge_changes():
    prior = {"subscription_cost_usd": 2}
    stored = {"subagent_envelope": {"execution_evidence": prior}}
    assert not delegate_terminal._stored_evidence_stale(stored, {
        **prior, "subscription_cost_estimated": False, "subscription_cost_partial": False})
    assert delegate_terminal._stored_evidence_stale(stored, {
        **prior, "subscription_cost_estimated": False, "subscription_cost_partial": True})


def test_legacy_null_ledger_root_joins_the_same_receipt(tmp_path):
    from ouroboros.delegate_collection import record_terminal_money

    entry = custody.RunCustody(run_id="legacy-null", task_id="legacy-task", root_task_id="legacy-task",
                                route_id="test", ledger_root=None)
    assert custody.record_started(tmp_path, entry)
    record_terminal_money(tmp_path, entry, {"summary": {"state": "succeeded", "spendUsd": 2}})
    result = _display(tmp_path, task_id=entry.task_id)
    assert result["cost_final"] and result["cost_custody"]["unrecorded_run_ids"] == []
