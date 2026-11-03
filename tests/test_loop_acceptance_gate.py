"""A shared root-task fixture for loop tests.

This module once held the task-acceptance gate's tests (the host acceptance
panel, its decision writer and the nomination tool). That machinery is gone: a
task's result is reviewed as ``review_change(subject="result")``, covered in
``tests/test_result_review.py``. What survives is the canonical RUNNING root
admission that delivery, the admission seal and the result review read their
root lineage from; many loop and delivery tests import it from here.
"""
from __future__ import annotations

from types import SimpleNamespace


def _seed_acceptance_root(tmp_path, task_id: str, ctx: SimpleNamespace):
    """Mirror the production pre-loop canonical RUNNING admission."""
    from ouroboros.contracts.task_contract import build_task_contract
    from ouroboros.task_results import STATUS_RUNNING, write_task_result

    contract = build_task_contract({
        "id": task_id,
        "root_task_id": task_id,
        "delegation_role": "root",
    })
    write_task_result(
        tmp_path,
        task_id,
        STATUS_RUNNING,
        root_task_id=task_id,
        delegation_role="root",
        task_contract=contract,
        result="Task is running.",
    )
    metadata = getattr(ctx, "task_metadata", {})
    metadata = dict(metadata) if isinstance(metadata, dict) else {}
    metadata.update({"root_task_id": task_id, "budget_drive_root": str(tmp_path)})
    ctx.task_id = task_id
    ctx.root_task_id = task_id
    ctx.delegation_role = "root"
    ctx.task_metadata = metadata
    ctx.task_contract = contract
    return contract
