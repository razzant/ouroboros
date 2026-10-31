"""Published merge receipts and whole-work headroom composed with custody."""
import pytest

from ouroboros import usage_accounting as ua
from ouroboros.task_results import load_task_result, write_task_result
from tests.test_billing_group import _scope, _spend

pytestmark = pytest.mark.serial


@pytest.mark.parametrize("case,held", [("arguments", False), ("pre_effect", False), ("timeout_open", True), ("queued", True), ("ok", False)])
def test_real_pr_merge_registry_preserves_known_and_unknown_custody(tmp_path, monkeypatch, case, held):
    from ouroboros.tools.registry import ToolRegistry
    from ouroboros.tools import github
    from tests.test_pr_merge_receipts import FakeGh, HEAD
    from ouroboros.tool_custody import retained_tool_custody
    write_task_result(tmp_path, "merge-task", "running", root_task_id="merge-task")
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx.task_id = registry._ctx.root_task_id = "merge-task"
    registry._ctx.task_metadata = {"work_kind": "owner"}  # an owner task's minted kind
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
    assert not row.get("launch_handoffs", {})
    assert any(item['kind'] == 'merge_operation' for item in retained_tool_custody(tmp_path, 'root', row))
    from ouroboros.model_sleep import cold_blockers
    from supervisor.continuation_admission import conflicting_writers
    assert cold_blockers(registry._ctx) and conflicting_writers(queue, "root")
    write_task_result(tmp_path, "root", "running", launch_handoffs={
        "independent": {"tool": "pr_merge", "task_id": "root", "state": "claimed"}})
    fake.pr.update(state="MERGED", mergeCommit={"oid": MERGE})
    assert registry.execute_result("pr_merge", args).meta["operation_outcome"] == "completed"
    assert set(load_task_result(tmp_path, "root").get("launch_handoffs", {})) == {"independent"}
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
