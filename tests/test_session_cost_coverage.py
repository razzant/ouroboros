"""Task money includes execution coverage, without manufacturing ledger rows."""
from __future__ import annotations

import contextlib
from types import SimpleNamespace

import pytest

from ouroboros import delegate_custody as custody
from ouroboros import usage_accounting as usage
from ouroboros.cost_projection import cost_display
from supervisor.state import reconstruct_task_cost
from tests._review_session_route_shared import _owned_gateway_uses_each_test_transport, fake_route  # noqa: F401


def start(root, run="run-one", task="task-one", tree="task-one", **fields):
    entry = custody.RunCustody(run_id=run, task_id=task, root_task_id=tree,
                               route_id="test", **fields)
    assert custody.record_started(root, entry)
    return entry


def attempts():
    return [{"usageCost": {"cashUsd": 2, "cashKnowledge": "exact"}},
            {"usageCost": {"cashUsd": None, "cashKnowledge": "unknown"}}]


def card_consumers(frames):
    """The task card's own JS words for real producer frames: label and detail, frame as context."""
    import json
    import subprocess
    from pathlib import Path

    script = """
        import fs from 'node:fs';
        import {taskCostMeta, costPresentationDetail} from './web/modules/chat_activity.js';
        const frames = JSON.parse(fs.readFileSync(0, 'utf8'));
        console.log(JSON.stringify(frames.map(frame => ({label: taskCostMeta(frame),
            detail: costPresentationDetail(frame.cost_presentation, frame)}))));
    """
    rendered = subprocess.run(['node', '--input-type=module', '-e', script], cwd=Path(__file__).parents[1],
                              input=json.dumps(frames), capture_output=True, text=True, check=True)
    return json.loads(rendered.stdout)


@pytest.mark.parametrize("ambiguous_failure", [False, True])
def test_accepted_pending_start_is_not_free_and_final(tmp_path, ambiguous_failure):
    assert custody.record_start_requested(tmp_path, invocation_id="pending", task_id="task-one",
                                          root_task_id="task-one", request={"prompt": "work"}, route="test")
    if ambiguous_failure:
        assert custody.emit(tmp_path, custody.START_FAILED, {"invocation_id": "pending", "task_id": "task-one",
                                                           "definite": False})
    fields = reconstruct_task_cost("task-one", fields=True, drive_root=tmp_path)
    assert fields["cost_final"] is False
    assert fields["cost_presentation"]["accounting_open"] is True
    assert fields["cost_presentation"]["has_rows"] is False
    assert fields["accounted_upper_bound_usd"] is None
    assert usage.read_usage_records(tmp_path) == []


def test_live_run_keeps_root_subtotal_open_without_charging_it(tmp_path):
    start(tmp_path, task="child", tree="root")
    usage.record_subscription_session("paid", drive_root=tmp_path, route="test", task_id="root",
                                       root_task_id="root", spend_usd=2)
    summary = usage.usage_breakdown(tmp_path, include_custody=True, root_task_id="root")
    assert summary["cost_final"] is False
    assert summary["settled_usd"] == 2
    assert summary["attempt_counts"] == {"settled": 1}
    assert summary["non_final_rows"] == 0


def test_terminal_custody_claim_without_receipt_is_not_money(tmp_path):
    entry = start(tmp_path)
    assert custody.emit(tmp_path, custody.LEDGER_RECORDED, {"run_id": entry.run_id, "task_id": entry.task_id})
    assert custody.emit(tmp_path, custody.SETTLED, {"run_id": entry.run_id, "task_id": entry.task_id,
                                                  "state": "succeeded", "cost_usd": 0, "cost_final": True})
    assert reconstruct_task_cost(entry.task_id, fields=True, drive_root=tmp_path)["cost_final"] is False


def test_another_tasks_receipt_cannot_close_the_held_run(tmp_path):
    entry = start(tmp_path)
    usage.record_subscription_session(entry.run_id, drive_root=tmp_path, route="test", task_id="foreign",
                                       root_task_id="foreign", spend_usd=0)
    custody.emit(tmp_path, custody.SETTLED, {"run_id": entry.run_id, "task_id": entry.task_id, "state": "succeeded"})
    fields = reconstruct_task_cost(entry.task_id, fields=True, drive_root=tmp_path)
    assert fields["cost_final"] is False
    assert fields["cost_custody"]["unrecorded_run_ids"] == [entry.run_id]
    assert fields["cost_presentation"]["has_rows"] is False


def test_collector_keeps_exact_debit_and_unpriced_remainder(tmp_path):
    from ouroboros.delegate_collection import record_terminal_money
    from ouroboros.tools.delegate_terminal_evidence import _reported_cost

    entry = start(tmp_path)
    detail = {"summary": {"state": "succeeded", "spendUsd": 2, "spendEstimated": True},
              "attemptExecution": attempts()}
    assert record_terminal_money(tmp_path, entry, detail)["ledger_recorded"]
    row, = usage.read_usage_records(tmp_path)
    assert row["cost_usd"] == 2
    assert row["reservation_upper_bound_usd"] is None
    summary = usage.usage_breakdown(tmp_path, include_custody=True, task_id=entry.task_id)
    assert summary["unknown_unmetered"] == 1
    fields = reconstruct_task_cost(entry.task_id, fields=True, drive_root=tmp_path)
    assert fields["cost_presentation"]["has_unpriced"] is True
    assert fields["cost_presentation"]["tracked_final"] is True
    assert "upper bound" not in cost_display(fields)
    assert "Tracked" in cost_display(fields)
    nanny = _reported_cost(detail["summary"], attempt_execution=attempts())
    assert nanny["cost_usd"] == 2 and nanny["cost_final"] is False
    assert "ESTIMATED" not in nanny["note"]


def test_receipt_and_terminal_observation_close_money_despite_cleanup_hold(tmp_path):
    from ouroboros import delegate_custody_current as current
    from ouroboros.delegate_collection import record_terminal_money

    entry = start(tmp_path, snapshot_id="held", project_owned=True)
    record_terminal_money(tmp_path, entry, {"summary": {"state": "succeeded", "spendUsd": 0}})
    assert entry.run_id in current.runs(tmp_path, "state=?", (current.HELD,))
    fields = reconstruct_task_cost(entry.task_id, fields=True, drive_root=tmp_path)
    assert fields["accounted_upper_bound_usd"] == 0
    assert fields["cost_final"] is True
    assert fields["cost_presentation"]["accounting_open"] is False


def test_closed_absent_is_unknown_execution_cost_without_ledger_row(tmp_path):
    entry = start(tmp_path)
    custody.emit(tmp_path, custody.CLOSED_ABSENT, {"run_id": entry.run_id, "task_id": entry.task_id})
    fields = reconstruct_task_cost(entry.task_id, fields=True, drive_root=tmp_path)
    assert fields["cost_final"] is False
    assert fields["cost_custody"]["unrecorded_run_ids"] == [entry.run_id]
    assert fields["cost_presentation"]["has_rows"] is False


def test_collector_write_failure_and_ledger_to_custody_handoff(tmp_path, monkeypatch):
    from ouroboros.delegate_collection import record_terminal_money

    entry = start(tmp_path)
    record = usage.record_subscription_session
    def fail(*a, **kw):
        raise OSError("synthetic receipt write failure")
    monkeypatch.setattr(usage, "record_subscription_session", fail)
    detail = {"summary": {"state": "succeeded", "spendUsd": 2}}
    assert record_terminal_money(tmp_path, entry, detail)["ledger_recorded"] is False
    assert reconstruct_task_cost(entry.task_id, fields=True, drive_root=tmp_path)["cost_final"] is False
    observations = []
    def write_then_read(*a, **kw):
        result = record(*a, **kw)
        observations.append(reconstruct_task_cost(entry.task_id, fields=True, drive_root=tmp_path))
        return result
    monkeypatch.setattr(usage, "record_subscription_session", write_then_read)
    assert record_terminal_money(tmp_path, entry, detail)["ledger_recorded"] is True
    assert observations[0]["accounted_upper_bound_usd"] == 2
    assert observations[0]["cost_final"] is False  # custody read preceded terminal publication
    assert reconstruct_task_cost(entry.task_id, fields=True, drive_root=tmp_path)["cost_final"] is True
    assert len(usage.read_usage_records(tmp_path)) == 1


def test_settlement_between_custody_and_money_read_is_conservative(tmp_path, monkeypatch):
    from ouroboros import delegate_custody_current as current
    from ouroboros.delegate_collection import record_terminal_money

    entry = start(tmp_path)
    observe = current.task_snapshot
    def finish_after_snapshot(*a, **kw):
        snapshot = observe(*a, **kw)
        record_terminal_money(tmp_path, entry, {"summary": {"state": "succeeded", "spendUsd": 3}})
        return snapshot
    monkeypatch.setattr(current, "task_snapshot", finish_after_snapshot)
    fields = reconstruct_task_cost(entry.task_id, fields=True, drive_root=tmp_path)
    assert fields["accounted_upper_bound_usd"] == 3
    assert fields["cost_final"] is False
    assert fields["cost_custody"]["unrecorded_run_ids"] == []
    monkeypatch.setattr(current, "task_snapshot", observe)
    assert reconstruct_task_cost(entry.task_id, fields=True, drive_root=tmp_path)["cost_final"] is True


def test_receipts_and_amounts_share_one_ledger_transaction(tmp_path, monkeypatch):
    from ouroboros import usage_store
    from ouroboros.delegate_collection import record_terminal_money

    entry = start(tmp_path)
    record_terminal_money(tmp_path, entry, {"summary": {"state": "succeeded", "spendUsd": 2}})
    reads = []
    original = usage_store.read
    @contextlib.contextmanager
    def counted(*a, **kw):
        with original(*a, **kw) as txn:
            seen = []
            class View:
                def __getattr__(self, key): return getattr(txn, key)
                def bucket(self, *a, **kw):
                    seen.append("amount")
                    return txn.bucket(*a, **kw)
                def summary_rows(self, *a, **kw):  # the amounts captured in one step (store_capture)
                    seen.append("amount")
                    return txn.summary_rows(*a, **kw)
                def attempt(self, *a, **kw):
                    seen.append("receipt")
                    return txn.attempt(*a, **kw)
            yield View()
            reads.append(seen)
    monkeypatch.setattr(usage_store, "read", counted)
    assert reconstruct_task_cost(entry.task_id, fields=True, drive_root=tmp_path)["cost_final"] is True
    assert len(reads) == 1 and "amount" in reads[0] and "receipt" in reads[0]


def test_own_tree_sibling_and_review_scopes(tmp_path):
    from supervisor.events_task_done import _authoritative_terminal_cost

    start(tmp_path, task="child", tree="root")
    start(tmp_path, run="review", task="root", tree="root", category="review", source="independent_review",
          review_slot_id="seat")
    start(tmp_path, run="elsewhere", task="foreign", tree="foreign")
    sibling = reconstruct_task_cost("sibling", fields=True, drive_root=tmp_path)
    assert sibling["cost_final"] is True and sibling["cost_presentation"]["accounting_open"] is False
    own = reconstruct_task_cost("child", fields=True, drive_root=tmp_path)
    assert own["cost_custody"]["open_run_ids"] == ["run-one"]
    root = _authoritative_terminal_cost("root", {}, {}, {}, tmp_path)
    assert root["cost_custody"]["open_run_ids"] == ["review", "run-one"]
    assert root["cost_presentation"]["scope"] == "root_tree"
    assert root["cost_final"] is False and root["cost_with_children_partial"] is True


def test_grouped_root_buckets_keep_their_addressed_coverage(tmp_path):
    from ouroboros.task_results import write_task_result

    for tree in ("root-a", "root-b"):
        write_task_result(tmp_path, tree, "running", root_task_id=tree)
        usage.record_subscription_session(tree, drive_root=tmp_path, route="test", task_id="same-task",
                                           root_task_id=tree, spend_usd=1)
    start(tmp_path, task="same-task", tree="root-a")
    summary = usage.usage_breakdown(tmp_path, include_custody=True, task_id="same-task")
    assert summary["cost_final"] is False
    assert summary["by_root"]["root-a"]["cost_final"] is False
    assert summary["by_root"]["root-b"]["cost_final"] is True
    assert summary["by_task"]["same-task"]["cost_final"] is False


def test_split_execution_custody_and_canonical_ledger(tmp_path):
    from ouroboros.delegate_collection import record_terminal_money
    from supervisor.events_task_done import _authoritative_terminal_cost, _terminal_cost_probe

    canonical, execution = tmp_path / "money", tmp_path / "execution"
    entry = start(execution, ledger_root=str(canonical))
    row = {"task_id": entry.task_id, "root_task_id": entry.task_id, "status": "completed",
           "budget_drive_root": str(canonical)}
    fields = _authoritative_terminal_cost(entry.task_id, row, row, {}, execution)
    assert fields["cost_final"] is False and fields["accounted_upper_bound_usd"] is None
    record_terminal_money(execution, entry, {"summary": {"state": "succeeded", "spendUsd": 2}})
    _, fields, _ = _terminal_cost_probe(execution, entry.task_id, row)
    assert fields["cost_final"] is True
    assert fields["accounted_upper_bound_usd_with_children"] == 2
    assert len(usage.read_usage_records(canonical)) == 1
    assert usage.read_usage_records(execution) == []


def test_unreadable_custody_keeps_known_money_but_not_finality(tmp_path, monkeypatch):
    from ouroboros import delegate_custody_current as current
    usage.record_subscription_session("paid", drive_root=tmp_path, route="test", task_id="task-one", spend_usd=2)
    def unreadable(*a, **kw): raise OSError("synthetic unreadable coverage")
    monkeypatch.setattr(current, "task_snapshot", unreadable)
    fields = reconstruct_task_cost("task-one", fields=True, drive_root=tmp_path)
    assert fields["accounted_upper_bound_usd"] == 2
    assert fields["cost_final"] is False
    assert fields["cost_custody"]["status"] == "unavailable"
    assert fields["cost_presentation"]["has_unpriced"] is True
    assert fields["non_final_rows"] == 0
    # The card keeps the known $2 as a qualified subtotal; only a missing amount reads unknown.
    [card] = card_consumers([fields])
    assert card["label"] == ["Tracked: $2.00"]
    assert "only the priced steps" in card["detail"]


@pytest.mark.parametrize("spend,final,tracked", [(0, True, 0), (None, False, None)])
def test_exact_zero_and_unknown_terminal_receipts(tmp_path, spend, final, tracked):
    from ouroboros.delegate_collection import record_terminal_money
    entry = start(tmp_path)
    record_terminal_money(tmp_path, entry, {"summary": {"state": "succeeded", "spendUsd": spend}})
    fields = reconstruct_task_cost(entry.task_id, fields=True, drive_root=tmp_path)
    assert fields["cost_final"] is final
    assert fields["cost_presentation"]["tracked_amount"] == tracked
    assert fields["cost_presentation"]["has_rows"] is True


def test_continued_review_retains_exact_cash_and_unknown_total(tmp_path):
    from ouroboros.delegate_custody_usage import observe_session_usage
    from ouroboros.review_execution import _combined_session_usage
    from ouroboros.review_session_usage import session_usage_facts

    stopped = {"summary": {"state": "failed"}, "attemptExecution": attempts()}
    final = {"summary": {"state": "succeeded", "spendUsd": 0}}
    spend, estimated, combined = _combined_session_usage(custody, [stopped], final)
    assert spend == 2 and estimated is False
    assert combined["_cost_facts"]["cost_partial"] is True
    assert combined["_cost_facts"]["cost_final"] is False
    rows = [session_usage_facts("continued-1", summary=stopped["summary"], attempt_execution=attempts()),
            session_usage_facts("continued-2", summary=final["summary"])]
    seen = []
    observe_session_usage(seen.append, {"session_usage_events": rows})
    assert len(seen) == 2
    assert seen[0]["cost_disclosed_usd"] == 2 and seen[0]["cost_partial"] is True
    assert seen[0]["cost_estimated"] is False and seen[0]["cost"] is None
    assert seen[1]["cost"] == 0 and seen[1]["cost_final"] is True


def test_cost_refresh_after_author_final_needs_no_new_money(tmp_path, monkeypatch):
    from ouroboros import delegate_terminal, store, task_results
    from supervisor import queue
    from supervisor.events_task_done import _refresh_terminal_task_cost

    monkeypatch.setattr(queue, "task_has_live_ownership", lambda *a, **kw: False)
    for tid in ("root", "child"):
        task_results.write_task_result(tmp_path, tid, "completed", result="kept answer",
            root_task_id="root", parent_task_id="root" if tid == "child" else "")
        assert _refresh_terminal_task_cost(tmp_path, tid)
    entry = start(tmp_path, task="child", tree="root")
    delegate_terminal.refresh_recently_settled_terminals(tmp_path, live_task_ids=set())
    for tid in ("root", "child"):
        row = task_results.load_task_result(tmp_path, tid)
        assert row["cost_final"] is False and row["cost_presentation"]["accounting_open"] is True
    # A definite failure closes a pending invocation without a money row too.
    custody.emit(tmp_path, custody.CLOSED_ABSENT, {"run_id": entry.run_id, "task_id": entry.task_id})
    delegate_terminal.refresh_recently_settled_terminals(tmp_path, live_task_ids=set())
    with store.open_store(tmp_path, write=False) as txn:
        assert not any(facts.get("cost_revision") for facts in txn.set_rows("delegated_runs").values())
    assert usage.read_usage_records(tmp_path) == []


def test_cli_partial_grace_returns_unsettled_result(tmp_path, monkeypatch):
    from ouroboros import cli
    start(tmp_path)
    row = {"status": "completed", "cost_with_children_partial": False,
           **reconstruct_task_cost("task-one", fields=True, drive_root=tmp_path)}
    clock = [0.0]
    monkeypatch.setattr(cli, "time", SimpleNamespace(time=lambda: clock[0], sleep=lambda seconds: clock.__setitem__(0, clock[0] + seconds)))
    calls = []
    def request(*a):
        calls.append(a)
        return dict(row)
    answer = cli._await_cost_finality(SimpleNamespace(request=request), "task-one", row, grace_sec=5)
    assert clock[0] == 5 and len(calls) == 3
    assert answer["cost_final"] is False and answer["cost_presentation"]["accounting_open"] is True


@pytest.mark.parametrize("cost_recorded", [False, True])
def test_actual_continued_review_collects_partial_cost_once(tmp_path, monkeypatch, fake_route, cost_recorded):  # noqa: F811
    """Joining a legacy successor records the stopped run's partial cost once, whether an older host left it
    unrecorded or already collected it. The current host creates no new continuation."""
    from tests._review_session_route_shared import FakeGateway
    from tests.test_review_session_continuation import _engine, _run, legacy_successor

    posts = _engine(monkeypatch)
    get = FakeGateway.get_run
    def detail(self, rid, **kw):
        result = get(self, rid, **kw)
        result["summary"]["spendUsd"] = 2 if rid == "run-1" else 0
        result["summary"]["spendEstimated"] = rid == "run-1"
        if rid == "run-1":
            result["attemptExecution"] = attempts()
        return result
    monkeypatch.setattr(FakeGateway, "get_run", detail)
    checkpoints = []
    with monkeypatch.context() as legacy_host:
        if not cost_recorded:
            # The retired automatic loop continued before settling the stopped run.
            legacy_host.setattr(custody, "settle_run", lambda *args, **kwargs: None)
        token = legacy_successor(tmp_path, checkpoints)
    stopped = custody.lookup(tmp_path, "t-review", "run-1")[1]
    assert stopped.settled is cost_recorded
    assert len(usage.read_usage_records(tmp_path)) == int(cost_recorded)
    facts = _run(tmp_path, {"pending_invocation_id": checkpoints[0]}, checkpoints, [], reconcile=True)
    assert [(request, key) for request, key in posts if "continueFrom" in request] == [({"continueFrom": "run-1"}, token)]
    assert facts["run_id"] == "run-2"
    assert facts["spend"] == 2 and facts["spend_estimated"] is False
    assert facts["usage"]["cost_partial"] is True and facts["usage"]["cost_final"] is False
    rows = usage.read_usage_records(tmp_path)
    assert len(rows) == 2 and sum(row["cost_usd"] for row in rows) == 2
    fields = reconstruct_task_cost("t-review", fields=True, drive_root=tmp_path)
    assert fields["cost_presentation"] == {"scope": "own", "tracked_amount": 2, "has_unpriced": True,
                                           "tracked_final": True, "accounting_open": True, "has_rows": True}
    assert fields["unknown_unmetered"] == 1
    assert fields["cost_custody"]["unrecorded_run_ids"] == []


def test_unreadable_coverage_is_published_but_does_not_retire_refresh_debt(tmp_path, monkeypatch):
    from ouroboros import delegate_custody_current as current
    from ouroboros import delegate_terminal, store, task_results
    from supervisor import queue

    monkeypatch.setattr(queue, "task_has_live_ownership", lambda *a, **kw: False)
    task_results.write_task_result(tmp_path, "task-one", "completed", result="kept", cost_final=True,
                                    accounted_upper_bound_usd=0)
    entry = start(tmp_path)
    observe = current.task_snapshot
    def unavailable(*a, **kw): raise OSError("temporarily unavailable")
    monkeypatch.setattr(current, "task_snapshot", unavailable)
    delegate_terminal.refresh_recently_settled_terminals(tmp_path, live_task_ids=set())
    assert task_results.load_task_result(tmp_path, entry.task_id)["cost_final"] is False
    with store.open_store(tmp_path, write=False) as txn:
        assert txn.get_row("delegated_runs", entry.task_id)["cost_revision"]
    monkeypatch.setattr(current, "task_snapshot", observe)
    delegate_terminal.refresh_recently_settled_terminals(tmp_path, live_task_ids=set())
    with store.open_store(tmp_path, write=False) as txn:
        assert not (txn.get_row("delegated_runs", entry.task_id) or {}).get("cost_revision")


def test_custody_change_during_cost_publication_keeps_new_revision(tmp_path, monkeypatch):
    from ouroboros import delegate_terminal, store, task_results
    from supervisor import events_task_done as done
    from supervisor import queue

    monkeypatch.setattr(queue, "task_has_live_ownership", lambda *a, **kw: False)
    task_results.write_task_result(tmp_path, "task-one", "completed", result="kept")
    start(tmp_path)
    refresh = done._refresh_terminal_task_cost
    changed = []
    def racing(*a, **kw):
        result = refresh(*a, **kw)
        if not changed:
            changed.append(True)
            start(tmp_path, run="late-review", source="independent_review", category="review")
        return result
    monkeypatch.setattr(done, "_refresh_terminal_task_cost", racing)
    delegate_terminal.refresh_recently_settled_terminals(tmp_path, live_task_ids=set())
    with store.open_store(tmp_path, write=False) as txn:
        assert txn.get_row("delegated_runs", "task-one")["cost_revision"]
    delegate_terminal.refresh_recently_settled_terminals(tmp_path, live_task_ids=set())
    fields = task_results.load_task_result(tmp_path, "task-one")
    assert fields["cost_custody"]["open_run_ids"] == ["late-review", "run-one"]


def test_definite_refusal_updates_terminal_cost_without_a_ledger_write(tmp_path, monkeypatch):
    from ouroboros import delegate_terminal, task_results
    from supervisor import queue

    monkeypatch.setattr(queue, "task_has_live_ownership", lambda *a, **kw: False)
    custody.record_start_requested(tmp_path, invocation_id="request", task_id="task-one",
                                   root_task_id="task-one", request={"prompt": "work"}, route="test")
    task_results.write_task_result(tmp_path, "task-one", "completed", result="kept",
                                   **reconstruct_task_cost("task-one", fields=True, drive_root=tmp_path))
    assert task_results.load_task_result(tmp_path, "task-one")["cost_final"] is False
    custody.emit(tmp_path, custody.START_FAILED, {"invocation_id": "request", "task_id": "task-one", "definite": True})
    delegate_terminal.refresh_recently_settled_terminals(tmp_path, live_task_ids=set())
    fields = task_results.load_task_result(tmp_path, "task-one")
    assert fields["cost_final"] is True and fields["cost_presentation"]["accounting_open"] is False
    assert usage.read_usage_records(tmp_path) == []


def test_projection_reads_addressed_custody_and_never_history(tmp_path, monkeypatch):
    from ouroboros import store
    start(tmp_path)
    select = store.Txn.custody_raw_rows
    def addressed(self, where="", params=()):
        assert where and params
        return select(self, where, params)
    monkeypatch.setattr(store.Txn, "custody_raw_rows", addressed)
    monkeypatch.setattr(custody, "_iter_rows", lambda *a, **kw: pytest.fail("custody history replay"))
    monkeypatch.setattr(usage, "read_usage_records", lambda *a, **kw: pytest.fail("monetary history scan"))
    assert reconstruct_task_cost("task-one", fields=True, drive_root=tmp_path)["cost_final"] is False
    assert usage.usage_breakdown(tmp_path, root_task_id="task-one", include_custody=True)["cost_final"] is False


def test_public_result_queue_gateway_and_heartbeat_carry_pending_cost(tmp_path, monkeypatch):
    import json

    from starlette.requests import Request

    from ouroboros import task_results
    from ouroboros.cost_projection import live_root_cost_projection
    from ouroboros.gateway.tasks import _task_get_response
    from ouroboros.task_status import load_effective_task_result
    from supervisor.events_task_done import _authoritative_terminal_cost

    # The real result/queue reader and HTTP detail receive one frozen cost observation.
    start(tmp_path, task="child", tree="root")
    task = {"id": "root", "root_task_id": "root"}
    fields = _authoritative_terminal_cost("root", task, {}, {}, tmp_path)
    task_results.write_task_result(tmp_path, "root", "completed", root_task_id="root", result="done", **fields)
    effective = load_effective_task_result(tmp_path, "root")
    assert effective["cost_presentation"]["accounting_open"] is True
    request = Request({"type": "http", "method": "GET", "path": "/api/tasks/root", "query_string": b"",
                       "path_params": {"task_id": "root"},
                       "app": SimpleNamespace(state=SimpleNamespace(drive_root=tmp_path))})
    response = _task_get_response(request)
    payload = json.loads(response.body)
    assert payload["cost_final"] is False and payload["cost_presentation"]["has_rows"] is False
    heartbeat = live_root_cost_projection("root", task, {}, tmp_path)
    assert heartbeat["cost_presentation"] == fields["cost_presentation"]


def test_actual_cost_frames_use_existing_card_labels(tmp_path):
    import json

    from ouroboros.delegate_collection import record_terminal_money

    custody.record_start_requested(tmp_path, invocation_id="pending", task_id="pending", root_task_id="pending",
                                   request={"prompt": "work"}, route="test")
    partial = start(tmp_path, task="partial", tree="partial")
    record_terminal_money(tmp_path, partial, {"summary": {"state": "succeeded", "spendUsd": 2},
                                            "attemptExecution": attempts()})
    zero = start(tmp_path, run="zero", task="zero", tree="zero")
    record_terminal_money(tmp_path, zero, {"summary": {"state": "succeeded", "spendUsd": 0}})
    frames = [reconstruct_task_cost(tid, fields=True, drive_root=tmp_path) for tid in ("pending", "partial", "zero")]
    cards = card_consumers(frames)
    assert [card["label"] for card in cards] == [["Cost pending"], ["Tracked: $2.00"], ["$0.00"]]
    assert "only the priced steps" in cards[1]["detail"]
    print(json.dumps({"cost_observations": frames, "card_consumers": cards}, sort_keys=True))


# ---- #2828: closed-empty, awaiting collection and closed without a receipt ----

GATEWAY = SimpleNamespace(store_identity="", remove_project=lambda project_id: None)
SOURCE = {"sha256": "0" * 64, "path": "synthetic"}


def card_sticky(steps):
    """The card's sticky money after each real frame, in order: ``taskCostProjection`` then ``mergeStickyCostMeta``."""
    import json
    import subprocess
    from pathlib import Path

    script = """
        import fs from 'node:fs';
        import {taskCostProjection, mergeStickyCostMeta} from './web/modules/chat_activity.js';
        let sticky = null;
        const out = [];
        for (const [frame, ts] of JSON.parse(fs.readFileSync(0, 'utf8'))) {
            sticky = mergeStickyCostMeta(sticky, taskCostProjection(frame, ts));
            out.push(sticky ? {meta: sticky.meta, detail: sticky.detail || '', cleared: Boolean(sticky.cleared)} : null);
        }
        console.log(JSON.stringify(out));
    """
    rendered = subprocess.run(['node', '--input-type=module', '-e', script], cwd=Path(__file__).parents[1],
                              input=json.dumps(steps), capture_output=True, text=True, check=True)
    return json.loads(rendered.stdout)


def _observe_terminal(root, entry):
    assert custody.emit(root, custody.TERMINAL_OBSERVED, {"run_id": entry.run_id, "task_id": entry.task_id,
                                                         "state": "succeeded", "observation_seq": 1, "source_ref": SOURCE})


def _close_absent(root, entry):
    from ouroboros import delegate_custody_current as current
    from tests._shared import run_absent_in
    custody.close_absent_run(root, GATEWAY, current.run(root, entry.run_id), "synthetic: the engine has no such run",
                            absence=run_absent_in(str(getattr(GATEWAY, "store_identity", "") or "")))


def _ledger_claim(root, entry):
    assert custody.emit(root, custody.LEDGER_RECORDED, {"run_id": entry.run_id, "task_id": entry.task_id})


def _settled_claim(root, entry):
    assert custody.emit(root, custody.SETTLED, {"run_id": entry.run_id, "task_id": entry.task_id, "state": "succeeded"})


def _collect(root, entry, spend=0.25):
    from ouroboros.delegate_collection import record_terminal_money
    assert record_terminal_money(root, entry, {"summary": {"state": "succeeded", "spendUsd": spend}})["ledger_recorded"]


@pytest.mark.parametrize("name,steps,expected", [
    # Custody closed, no matching receipt: no collector follows, so the cost is unknown, never pending.
    ("production close_absent_run", [_close_absent], "closed"),
    ("unmatched LEDGER_RECORDED claim, then production close", [_ledger_claim, _close_absent], "closed"),
    ("SETTLED claim without a receipt (integrity gap)", [_settled_claim], "closed"),
    # Custody still open: money may still arrive.
    ("live run", [], "open"),
    ("terminal run awaiting collection", [_observe_terminal], "awaiting"),
    # A real receipt closes money: a known zero and a known amount.
    ("collected zero", [_observe_terminal, lambda r, e: _collect(r, e, 0)], "zero"),
    ("collected amount", [_observe_terminal, _collect], "amount"),
])
def test_custody_settlement_and_receipt_decide_pending_unknown_or_known(tmp_path, name, steps, expected):
    import json

    entry = start(tmp_path)
    for step in steps:
        step(tmp_path, entry)
    if expected == "awaiting":  # production refuses to call a terminal-observed run absent
        with pytest.raises(Exception, match="absence is unproven"):
            _close_absent(tmp_path, entry)
    fields = reconstruct_task_cost(entry.task_id, fields=True, drive_root=tmp_path)
    coverage, carrier = fields["cost_custody"], fields["cost_presentation"]
    [card] = card_consumers([fields])
    observed = {"coverage": coverage, "carrier": carrier, "card": card, "text": cost_display(fields)}
    print(json.dumps({name: observed}, sort_keys=True))
    run = [entry.run_id]
    if expected == "closed":
        assert coverage["unrecorded_run_ids"] == run and coverage["closed_unrecorded_run_ids"] == run
        assert coverage["open_run_ids"] == []
        assert (carrier["has_rows"], carrier["accounting_open"], carrier["has_unpriced"]) == (False, False, True)
        assert fields["cost_final"] is False and fields["accounted_upper_bound_usd"] is None
        assert card["label"] == ["Cost unknown"]
        assert observed["text"] == "unknown (a delegated run closed without a cost record)"
    elif expected in {"open", "awaiting"}:
        assert coverage["unrecorded_run_ids"] == run and coverage["closed_unrecorded_run_ids"] == []
        assert coverage["open_run_ids"] == (run if expected == "open" else [])
        assert (carrier["has_rows"], carrier["accounting_open"], carrier["has_unpriced"]) == (False, True, True)
        assert card["label"] == ["Cost pending"]
        assert observed["text"] == "unknown (not settled yet)"
    else:
        assert coverage["unrecorded_run_ids"] == [] and coverage["closed_unrecorded_run_ids"] == []
        assert fields["cost_final"] is True and carrier["accounting_open"] is False
        assert card["label"] == (["$0.00"] if expected == "zero" else ["$0.25"])
        assert observed["text"] == ("$0.00" if expected == "zero" else "$0.25")


def test_closed_unrecorded_run_beside_priced_money_keeps_the_known_subtotal(tmp_path):
    """A settled run without a receipt leaves the scope's priced money known but not total, and settled."""
    entry = start(tmp_path)
    usage.record_subscription_session("priced", drive_root=tmp_path, route="test", task_id=entry.task_id,
                                       root_task_id=entry.task_id, spend_usd=1.5)
    _close_absent(tmp_path, entry)
    fields = reconstruct_task_cost(entry.task_id, fields=True, drive_root=tmp_path)
    carrier = fields["cost_presentation"]
    assert (carrier["tracked_amount"], carrier["accounting_open"], carrier["has_unpriced"]) == (1.5, False, True)
    assert fields["cost_final"] is False
    [card] = card_consumers([fields])
    assert card["label"] == ["Tracked: $1.50"]
    assert cost_display(fields) == "Tracked: $1.50 (total unknown)"


def _terminal_frames(tmp_path, monkeypatch, close, *, task="task-one", parent=""):
    """The terminal record written while custody is open, then the refresh after ``close``: real frames."""
    import json

    from ouroboros import delegate_terminal, task_results
    from ouroboros.cost_projection import carry_cost_meta
    from supervisor import queue
    from supervisor.events_task_done import _authoritative_terminal_cost

    monkeypatch.setattr(queue, "task_has_live_ownership", lambda *a, **kw: False)
    lineage = {"root_task_id": parent or task, **({"parent_task_id": parent, "delegation_role": "subagent"} if parent else {})}
    row = {"id": task, **lineage}
    task_results.write_task_result(tmp_path, task, "completed", result="kept", **lineage,
                                   **_authoritative_terminal_cost(task, row, {}, {}, tmp_path))
    first = carry_cost_meta(task_results.load_task_result(tmp_path, task))
    assert first["cost_presentation"]["scope"] == ("own" if parent else "root_tree")
    close()
    delegate_terminal.refresh_recently_settled_terminals(tmp_path, live_task_ids=set())
    events = [json.loads(line) for line in (tmp_path / "logs" / "events.jsonl").read_text().splitlines()]
    finalized = [row for row in events if row.get("type") == "task_cost_finalized" and row.get("task_id") == task]
    assert finalized, "the custody change owes a cost refresh frame"
    return first, finalized[-1], task_results.load_task_result(tmp_path, task)


def test_healthy_closed_empty_clears_the_earlier_pending_and_stale_replay_cannot_revive_it(tmp_path, monkeypatch):
    custody.record_start_requested(tmp_path, invocation_id="request", task_id="task-one",
                                   root_task_id="task-one", request={"prompt": "work"}, route="test")
    pending, cleared, stored = _terminal_frames(tmp_path, monkeypatch, lambda: custody.emit(
        tmp_path, custody.START_FAILED, {"invocation_id": "request", "task_id": "task-one", "definite": True}))
    assert pending["cost_presentation"]["accounting_open"] is True
    assert (cleared["cost_presentation"]["has_rows"], cleared["cost_presentation"]["accounting_open"],
            cleared["cost_presentation"]["has_unpriced"]) == (False, False, False)
    # Python surfaces read the stored closed-empty record as no money, never a manufactured $0.00.
    assert stored["accounted_upper_bound_usd"] == 0.0 and cost_display(stored) == "unknown (no usage recorded)"
    evidence_free = {"cost_usd": 0.03}  # an llm_round_finished delta: no task-scope evidence
    t0, t1, t2, t3 = "2026-11-09T00:00:00Z", cleared["ts"], "2099-01-01T00:00:00Z", "2099-01-01T00:01:00Z"
    from ouroboros.cost_projection import live_root_cost_projection
    from supervisor.events_task_done import _authoritative_terminal_cost

    row = {"id": "task-one", "root_task_id": "task-one"}
    custody.record_start_requested(tmp_path, invocation_id="retry", task_id="task-one",
                                   root_task_id="task-one", request={"prompt": "again"}, route="test")
    retry = live_root_cost_projection("task-one", row, {}, tmp_path)  # the running root's heartbeat figure
    assert retry["cost_presentation"]["scope"] == "root_tree" and retry["cost_presentation"]["accounting_open"] is True
    entry = start(tmp_path, run="retry-run")
    _collect(tmp_path, entry, 0.4)
    custody.emit(tmp_path, custody.START_FAILED, {"invocation_id": "retry", "task_id": "task-one", "definite": True})
    final = _authoritative_terminal_cost("task-one", row, {}, {}, tmp_path)
    assert final["cost_final"] is True and final["cost_presentation"]["scope"] == "root_tree"
    cards = card_sticky([[pending, t0], [cleared, t1], [pending, t0], [evidence_free, t2],
                         [retry, t2], [cleared, t1], [final, t3], [cleared, t3]])
    assert [card["meta"] for card in cards] == [
        ["Cost pending"],
        [],               # the clear settles the earlier pending: no money line
        [],               # an older replayed pending loses to the clear's timestamp
        [],               # an evidence-free frame keeps the clear
        ["Cost pending"], # a newer retry is pending again
        ["Cost pending"], # the older clear cannot hide it
        ["$0.40"],        # a final amount wins
        ["$0.40"],        # and a later clear does not erase a final amount
    ]
    assert cards[1]["cleared"] is True


def test_closed_absent_after_the_terminal_record_turns_pending_into_unknown(tmp_path, monkeypatch):
    entry = start(tmp_path)
    _ledger_claim(tmp_path, entry)  # an unmatched claim must not hide the closure
    pending, closed, stored = _terminal_frames(tmp_path, monkeypatch, lambda: _close_absent(tmp_path, entry))
    assert closed["cost_custody"]["closed_unrecorded_run_ids"] == [entry.run_id]
    assert cost_display(stored) == "unknown (a delegated run closed without a cost record)"
    cards = card_sticky([[pending, "2026-11-09T00:00:00Z"], [closed, closed["ts"]], [pending, "2026-11-09T00:00:00Z"]])
    assert [card["meta"] for card in cards] == [["Cost pending"], ["Cost unknown"], ["Cost unknown"]]


def test_terminal_run_awaiting_collection_stays_pending_then_settles_on_collection(tmp_path, monkeypatch):
    entry = start(tmp_path)
    _observe_terminal(tmp_path, entry)
    pending, collected, stored = _terminal_frames(tmp_path, monkeypatch, lambda: _collect(tmp_path, entry, 0.25))
    assert pending["cost_custody"]["closed_unrecorded_run_ids"] == []
    assert pending["cost_presentation"]["accounting_open"] is True
    assert stored["cost_final"] is True and cost_display(stored) == "$0.25"
    cards = card_sticky([[pending, "2026-11-09T00:00:00Z"], [collected, collected["ts"]],
                         [pending, "2026-11-09T00:00:00Z"]])
    assert [card["meta"] for card in cards] == [["Cost pending"], ["$0.25"], ["$0.25"]]


@pytest.mark.ui_browser
@pytest.mark.parametrize("engine", ["chromium", "webkit"])
@pytest.mark.parametrize("ending,expected", [("empty", ""), ("absent", "Cost unknown"), ("zero", "$0.00")])
@pytest.mark.parametrize("relation", ["root", "child"])
def test_settled_money_reaches_the_rendered_card_and_survives_reload(tmp_path, monkeypatch, engine, ending, expected, relation):
    """Real custody/ledger writers feed the actual chat module, through its log and history consumers."""
    import json
    from tests.test_task_card_b_browser import card_room, _done, _evidence, _say, _settle, _skip_without_smoke

    playwright = _skip_without_smoke()
    task_id, old = "task-one", "2000-01-01T00:00:00Z"
    parent = "w-root" if relation == "child" else ""
    if parent:
        from ouroboros import task_results
        task_results.write_task_result(tmp_path, parent, "running", root_task_id=parent)
    if ending == "empty":
        custody.record_start_requested(tmp_path, invocation_id="request", task_id=task_id,
                                       root_task_id=parent or task_id, request={"prompt": "work"}, route="test")
        close = lambda: custody.emit(tmp_path, custody.START_FAILED, {
            "invocation_id": "request", "task_id": task_id, "definite": True})
    else:
        entry = start(tmp_path, tree=parent or task_id)
        close = (lambda: _close_absent(tmp_path, entry)) if ending == "absent" else (lambda: _collect(tmp_path, entry, 0))
    pending, settled, stored = _terminal_frames(tmp_path, monkeypatch, close, parent=parent)
    assert pending["cost_presentation"]["accounting_open"]
    with playwright.sync_playwright() as pw, card_room(pw, engine, "light", 1100, 750) as room:
        page = room["page"]
        anchors = ([_say(parent, "Parent work", old)] if parent else []) + [_say(task_id, "Cost settlement", old, **({
            "parent_task_id": parent, "root_task_id": parent, "delegation_role": "subagent",
            "subagent_task_id": task_id, "subagent_role": "Cost checker", "subagent_event": "running",
        } if parent else {}))]
        room["emit"](anchors + [_done(task_id, old, **pending)])
        card = page.locator(f'.chat-live-card[data-task-id="{task_id}"]')
        meta = card.locator(':scope > .chat-live-foot > .chat-live-meta')
        page.wait_for_function("() => [...document.querySelectorAll('.chat-live-meta')].some(e => e.textContent.includes('Cost pending'))")
        room["emit"]([("log", {"chat_id": 1, "data": settled})])
        page.wait_for_function("([id, expected]) => document.querySelector(`.chat-live-card[data-task-id=\"${id}\"] > .chat-live-foot > .chat-live-meta`)?.textContent.trim() === expected", arg=[task_id, expected], timeout=5000)
        _settle(page)
        assert meta.text_content().strip() == expected
        assert card.get_attribute("data-finished") == "1"
        live = room["facts"](task_id)
        shot = _evidence(f"cost-settled-{relation}-{engine}-{ending}.png")
        if shot:
            page.screenshot(path=shot)
        # A stale pending event cannot resurrect the old line.
        room["emit"]([("log", {"chat_id": 1, "data": {
            "type": "task_cost_finalized", "task_id": task_id, "ts": old, **pending}})])
        _settle(page)
        assert meta.text_content().strip() == expected
        room["details"][task_id] = stored
        # History retains the original card anchor as well as the later summary;
        # a summary is metadata, not a substitute for the task's own visible row.
        room["rows"][:] = [{**row, "history_id": f"cost-start-{index}"} for index, (_, row) in enumerate(anchors)] + [
            {**stored, "chat_id": 1, "task_id": task_id, "role": "system",
             "system_type": "task_summary", "task_terminal_status": "completed", "text": "",
             "ts": settled["ts"], "history_id": "settled-money"},
        ]
        room["load"]()
        meta.wait_for(state="attached")
        _settle(page)
        assert meta.text_content().strip() == expected
        if ending != "zero":
            assert "$0" not in meta.text_content()
        assert card.get_attribute("data-finished") == "1"
        replay_shot = _evidence(f"cost-reloaded-{relation}-{engine}-{ending}.png")
        if replay_shot:
            page.screenshot(path=replay_shot)
        print(json.dumps({"ending": ending, "engine": engine, "relation": relation, "live": live,
                          "reloaded": room["facts"](task_id)}, ensure_ascii=False))
