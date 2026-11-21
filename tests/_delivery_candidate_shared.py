from __future__ import annotations

import json


def write_child(
    tmp_path,
    child_id: str = "child1",
    status: str = "completed",
    **fields,
):
    """Write the canonical child-result fixture shared by delivery tests."""
    from ouroboros.task_results import write_task_result

    return write_task_result(
        tmp_path,
        child_id,
        status,
        parent_task_id="parent1",
        root_task_id="parent1",
        delegation_role="subagent",
        result="full child result",
        trace_summary="trace",
        artifact_status="ready",
        artifacts=[{"kind": "report", "name": "report.md", "sha256": "a" * 64}],
        **fields,
    )


def write_confirmed_disposition_fixture(
    tmp_path,
    *,
    child_id: str = "child1",
    disposition: str,
    rationale: str,
):
    """Append the authoritative task-tree row used by loop/outcome unit tests."""

    from ouroboros.task_status import load_effective_task_result
    from ouroboros.task_tree_ledger import tree_ledger_append
    from ouroboros.tools.join_ledger import _child_result_sha256

    child = load_effective_task_result(tmp_path, child_id)
    result_sha256 = _child_result_sha256(child)
    root_task_id = str(child.get("root_task_id") or "parent1")
    parent_task_id = str(child.get("parent_task_id") or "parent1")
    written = tree_ledger_append(
        root_task_id,
        "decision",
        rationale,
        task_id=parent_task_id,
        role="orchestrator",
        payload={
            "type": "child_result_disposition",
            "child_task_id": child_id,
            "disposition": disposition,
            "child_result_sha256": result_sha256,
        },
        allow_child_result_disposition=True,
        data_root=tmp_path,
    )
    assert written.startswith("OK:")
    return load_effective_task_result(tmp_path, child_id)


def holds(root):
    """The completion_held rows of the events log under ``root`` (absent file: none)."""
    path = root / "logs" / "events.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
    return [row for row in rows if row.get("type") == "task_checkpoint" and row.get("checkpoint_kind") == "completion_held"]
