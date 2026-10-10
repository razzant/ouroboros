"""Published late review/merge receipts composed with whole-work custody."""
import threading

import pytest

from ouroboros import usage_accounting as ua
from ouroboros.task_results import load_task_result, write_task_result
from tests.test_acceptance_history import _caller, _request, _source
from tests.test_acceptance_late_consumers import delivered, late  # noqa: F401
from tests.test_review_operation_collection import fresh_sends  # noqa: F401
from tests.test_review_operation_lifetime import until
from tests.test_billing_group import _scope, _spend
from tests._usage_store_testing import ledger_rows

pytestmark = pytest.mark.serial


def bind_group(f, group=None):
    fields = {"billing_group_id": group or f.accounting, "billing_group_limit_usd": 4.,
              "billing_group_limit_source": "initial_task_admission", "billing_group_limit_revision": "initial"}
    row = load_task_result(f.root, f.accounting)
    write_task_result(f.root, f.accounting, row["status"], billing_group=fields)
    return fields


def test_owner_cap_is_total_group_allowance_with_successor_and_late_charge(late, tmp_path, monkeypatch):  # noqa: F811
    f = delivered(tmp_path, monkeypatch)
    binding = bind_group(f)
    _spend(f.root, _scope(f.root, f.accounting, f.accounting, group=f.accounting, group_limit=4), 1)
    _spend(f.root, _scope(f.root, "successor", "successor", group=f.accounting, group_limit=4), 2)
    ctx = _caller(f)
    before = ledger_rows(f.root)
    out = _request(f, ctx, _source(ctx), action="amend_cap", new_original_root_cap_usd=9)
    assert out["status"] == "amended" and not out["dispatched"] and not late.calls
    assert ledger_rows(f.root) == before
    assert load_task_result(f.root, f.accounting)["billing_group"] == binding
    with ua.usage_scope(_scope(f.root, "successor", "successor", group=f.accounting, group_limit=4)):
        paid = ua.reserve_attempt(ua.AttemptRequest(model="m", provider="test", reservation_usd=6.0))
        assert paid.scope.billing_group_limit_usd == 9
        assert paid.scope.billing_group_limit_source == "owner_amendment"
        ua.mark_dispatched(paid)
        ua.settle_attempt(paid, {}, cost_usd=6.0, cost_final=True)  # known spend reaches the amended $9
    with pytest.raises(ua.BudgetExceeded):
        _spend(f.root, _scope(f.root, f.accounting, f.accounting, group=f.accounting, group_limit=4), .3)
    projection = ua.usage_projection(f.root, billing_group_id=f.accounting)
    assert projection["accounted_usd"] == pytest.approx(9.0) and projection["limit_usd"] == 9
    from ouroboros.usage_admission import task_money_snapshot
    snapshot = task_money_snapshot(f.root, {"id": f.accounting}, f.accounting)
    assert snapshot["remaining_known_usd"] == pytest.approx(0.0)
    assert snapshot["group_axis"]["source"] == "owner_amendment"
    with ua.usage_scope(_scope(f.root, f.accounting, f.accounting, group=f.accounting, group_limit=4)):
        from ouroboros.loop_budget import _loop_tree_accounting
        assert _loop_tree_accounting(refresh=True, strict=True)["settled_usd"] == pytest.approx(9.0)
    monkeypatch.setattr(ua, "_reservation_cost", lambda _request: .1)
    requested = _request(f, ctx, _source(ctx, text="Now review the historical answer"), action="review")
    assert requested["reason"] == "review_wave_budget_insufficient", requested
    assert not late.calls


def test_successor_cap_amendment_cannot_widen_original_group(late, tmp_path, monkeypatch):  # noqa: F811
    f = delivered(tmp_path, monkeypatch)
    bind_group(f, "original")
    _spend(f.root, _scope(f.root, "original", "original", group="original", group_limit=4), 4.0)
    ctx = _caller(f)
    assert _request(f, ctx, _source(ctx), action="amend_cap", new_original_root_cap_usd=9)["status"] == "amended"
    with pytest.raises(ua.BudgetExceeded):
        _spend(f.root, _scope(f.root, f.accounting, f.accounting, group="original", group_limit=4), .3)
    assert not late.calls


def test_historical_handed_review_survives_pause_and_keeps_original_group(late, tmp_path, monkeypatch):  # noqa: F811
    from ouroboros import review_operation
    f = delivered(tmp_path, monkeypatch)
    bind_group(f)
    gate = threading.Event()
    late.gates.append(gate)
    ctx = _caller(f)
    try:
        out = _request(f, ctx, _source(ctx, text="Review the historical answer"), action="review")
        assert out["status"] in {"pending", "announced", "published", "settled"}, out
        until(lambda: len(late.calls) == 3)
        row = load_task_result(f.root, f.accounting)
        write_task_result(f.root, f.accounting, row["status"], budget_pause={"state": "pausing", "reason": "owner"},
                          owner_pause={"state": "requested", "fence_id": "owner-pause"})
        for operation in list(review_operation._LIVE.values()):
            assert not operation.control()
        assert all(scope.billing_group_id == f.accounting for scope, _ in late.calls)
    finally:
        gate.set()
    until(lambda: not review_operation._LIVE)
    assert len(late.calls) == 3
    rows = ua.read_usage_records(f.root, final_only=True)
    assert len(rows) == 3 and all(row["state"] == "settled" for row in rows)
    assert load_task_result(f.root, f.accounting)["owner_pause"]["state"] == "requested"


@pytest.mark.parametrize("case,held", [("arguments", False), ("pre_effect", False), ("timeout_open", True), ("queued", True), ("ok", False)])
def test_real_pr_merge_registry_preserves_known_and_unknown_custody(tmp_path, monkeypatch, case, held):
    from ouroboros.tools.registry import ToolRegistry
    from ouroboros.tools import github
    from tests.test_pr_merge_receipts import FakeGh, HEAD
    from ouroboros.tool_custody import retained_tool_custody
    write_task_result(tmp_path, "merge-task", "running", root_task_id="merge-task")
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx.task_id = registry._ctx.root_task_id = "merge-task"
    fake = FakeGh(tmp_path, merge=case)
    monkeypatch.setattr(github, "_gh_run", fake)
    monkeypatch.setattr(github, "github_cli_configured", lambda: True)
    result = registry.execute_result("pr_merge", {"number": 7, "expected_head_sha": HEAD,
        "method": "merge", "review_scope": "invalid" if case == "arguments" else "full"})
    row = load_task_result(tmp_path, "merge-task")
    assert not row.get("launch_handoffs"), "the local merge invocation returned"
    custody = retained_tool_custody(tmp_path, "merge-task", row)
    assert any(item["kind"] == "merge_operation" for item in custody) is held, result
    if held:
        assert result.meta["operation_outcome"] == "unknown"
        before = len([c for c in fake.calls if c[:2] == ["pr", "merge"]])
        registry.execute_result("pr_merge", {"number": 7, "expected_head_sha": HEAD, "method": "merge"})
        assert len([c for c in fake.calls if c[:2] == ["pr", "merge"]]) == before
        after = retained_tool_custody(tmp_path, "merge-task", load_task_result(tmp_path, "merge-task"))
        assert {item['receipt_id'] for item in custody} <= {item['receipt_id'] for item in after}


@pytest.mark.parametrize("initial", ["timeout_open", "queued"])
def test_merge_readback_retires_only_exact_prior_claims(tmp_path, monkeypatch, initial):
    from ouroboros.tools import github
    from tests.test_pr_merge_receipts import FakeGh, HEAD, MERGE
    from tests.test_batch4_producer_custody import _registry, _consumers
    from ouroboros.tool_custody import retained_tool_custody
    registry, queue, workers = _registry(tmp_path, monkeypatch)
    fake = FakeGh(tmp_path, merge=initial)
    monkeypatch.setattr(github, "github_cli_configured", lambda: True)
    monkeypatch.setattr(github, "_gh_run", fake)
    args = {"number": 7, "expected_head_sha": HEAD, "method": "merge"}
    result = registry.execute_result("pr_merge", args)
    assert result.meta["operation_outcome"] == "unknown"
    row = load_task_result(tmp_path, "root")
    assert not row["launch_handoffs"]
    assert any(item['kind'] == 'merge_operation' for item in retained_tool_custody(tmp_path, 'root', row))
    from ouroboros.model_sleep import cold_blockers
    from supervisor.continuation_admission import conflicting_writers
    assert cold_blockers(registry._ctx) and conflicting_writers(queue, "root")
    write_task_result(tmp_path, "root", "running", launch_handoffs={
        "independent": {"tool": "pr_merge", "task_id": "root", "state": "claimed"}})
    fake.pr.update(state="MERGED", mergeCommit={"oid": MERGE})
    assert registry.execute_result("pr_merge", args).meta["operation_outcome"] == "completed"
    assert set(load_task_result(tmp_path, "root")["launch_handoffs"]) == {"independent"}
    assert [item['kind'] for item in retained_tool_custody(
        tmp_path, 'root', load_task_result(tmp_path, 'root'))] == ['tool_handoff']
    assert sum(c[:2] == ["pr", "merge"] for c in fake.calls) == 1
    assert cold_blockers(registry._ctx) and conflicting_writers(queue, "root")
    # Remove only the synthetic independent fixture claim to inspect all readers.
    write_task_result(tmp_path, "root", "running", launch_handoffs={})
    _consumers(tmp_path, registry, queue, workers, False)



def test_composite_headroom_keeps_successor_own_cap_independent(tmp_path, monkeypatch):
    from ouroboros.usage_admission import task_money_snapshot
    from ouroboros.loop_budget import _loop_tree_accounting
    monkeypatch.setenv("TOTAL_BUDGET", "1000")
    binding = {"billing_group_id": "original", "billing_group_limit_usd": 9.,
               "billing_group_limit_source": "initial_task_admission", "billing_group_limit_revision": "initial"}
    for tid in ("original", "successor"):
        write_task_result(tmp_path, tid, "running", root_task_id=tid, billing_group=binding)
    _spend(tmp_path, _scope(tmp_path, "original", "original", group="original", group_limit=9), 4)
    scope = _scope(tmp_path, "successor", "successor", group="original", group_limit=9, root_limit=2)
    _spend(tmp_path, scope, 1)
    snapshot = task_money_snapshot(tmp_path, {"id": "successor"}, "successor")
    assert snapshot["remaining_known_usd"] == 1 and snapshot["binding_axis"] == "root"
    assert snapshot["root_axis"]["limit_usd"] == 2 and snapshot["group_axis"]["limit_usd"] == 9
    with ua.usage_scope(scope):
        paced = _loop_tree_accounting(refresh=True, strict=True)
        assert paced["accounted_usd"] == 5 and paced["root_limit_usd"] == 6
