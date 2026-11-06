"""D#7 soft-join: explicit child-decision tools (discard_child_result, peek_task).

These are the structured (P5, not parsed-from-prose) signals that let a parent finalize
without orphaning a child: discard appends an exact-hash task-tree decision; peek lets
the parent inspect a child WITHOUT absorbing it. A parent
decision may only touch the caller's OWN children (lineage-gated, fail-closed).
"""

from __future__ import annotations

from types import SimpleNamespace


def _ctx(tmp_path, task_id="parent1"):
    return SimpleNamespace(
        drive_root=str(tmp_path),
        budget_drive_root=str(tmp_path),
        task_metadata={},
        task_id=task_id,
        role="orchestrator",
    )


def _write_child(tmp_path, child_id, parent_id, **fields):
    from ouroboros.task_results import STATUS_RUNNING, write_task_result

    # Real subagent children carry delegation_role="subagent"; find_child_tasks (and thus
    # the lineage gate) only matches those.
    write_task_result(
        tmp_path, child_id, STATUS_RUNNING,
        parent_task_id=parent_id, root_task_id=parent_id, delegation_role="subagent", **fields,
    )


def test_discard_own_child_appends_typed_decision_without_mutating_result(tmp_path):
    from ouroboros.task_results import load_task_result
    from ouroboros.task_status import load_effective_task_result
    from ouroboros.tools.join_ledger import _discard_child_result

    _write_child(tmp_path, "child1", "parent1", result="partial work")
    raw_before = load_task_result(tmp_path, "child1")
    out = _discard_child_result(_ctx(tmp_path), "child1", "superseded by a different approach")
    assert "Discarded" in out
    assert load_task_result(tmp_path, "child1") == raw_before
    effective = load_effective_task_result(tmp_path, "child1")
    assert effective["child_result_disposition"] == "irrelevant"
    assert "superseded" in effective["child_result_disposition_reason"]


def test_discard_requires_reason(tmp_path):
    from ouroboros.tools.join_ledger import _discard_child_result

    _write_child(tmp_path, "child1", "parent1", result="x")
    out = _discard_child_result(_ctx(tmp_path), "child1", "   ")
    assert "reason is required" in out


def test_delegated_peek_is_lineage_scoped(tmp_path):
    from ouroboros.contracts.task_constraint import TaskConstraint
    from ouroboros.tools.join_ledger import _peek_task

    _write_child(tmp_path, "mine", "parent1", result="owned")
    _write_child(tmp_path, "sibling", "other_parent", result="not owned")
    ctx = _ctx(tmp_path)
    ctx.task_constraint = TaskConstraint(
        mode="local_readonly_subagent", allow_enable=False,
    )

    assert "owned" in _peek_task(ctx, "mine")
    assert "may inspect only its own children" in _peek_task(ctx, "sibling")


def test_peek_preserves_full_review_request_binding(tmp_path, monkeypatch):
    import ouroboros.task_tree_ledger as ledger
    from ouroboros.tools.join_ledger import _peek_task

    monkeypatch.setattr(ledger, "DATA_DIR", str(tmp_path))
    _write_child(tmp_path, "mine", "parent1", result="owned")
    ctx = _ctx(tmp_path)
    assert ledger.tree_ledger_append(
        "parent1", "review_requested", "Check the interface evidence.",
        task_id="mine",
        payload={"evidence_ref": "artifact:interface", "evidence_sha256": "b" * 64},
    ).startswith("OK:")

    out = _peek_task(ctx, "mine")
    assert "artifact:interface" in out
    assert "b" * 64 in out
