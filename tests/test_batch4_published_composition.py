"""Published merge receipts and whole-work headroom composed with custody."""
import pytest

from ouroboros.task_results import load_task_result, write_task_result



@pytest.mark.parametrize("case,held", [("arguments", False), ("refused", False), ("timeout_open", True), ("queued", True), ("ok", False)])
def test_real_pr_merge_registry_preserves_known_and_unknown_custody(tmp_path, monkeypatch, case, held):
    from ouroboros.tools.registry import ToolRegistry
    from ouroboros.tools import github
    from tests.test_pr_merge_receipts import FakeGh, HEAD, sent
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
        before = len([c for c in fake.calls if sent(c)])
        registry.execute_result("pr_merge", {"number": 7, "expected_head_sha": HEAD, "method": "merge"})
        assert len([c for c in fake.calls if sent(c)]) == before
        after = retained_tool_custody(tmp_path, "merge-task", load_task_result(tmp_path, "merge-task"))
        assert {item['receipt_id'] for item in custody} <= {item['receipt_id'] for item in after}
