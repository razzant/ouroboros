"""The PLANNING evidence horizon: one compact manifest for a plan review.

Its own module rather than more of `review_evidence`, because it answers a different
question. `review_evidence` builds the evidence a review of WORK ALREADY DONE reads;
this builds the evidence a review of WORK NOT YET DONE reads — a bounded picture of
the subject tree as it stands, which on a remote placement is the mirror rather than
anything on Home. One production caller (`tools/plan_review`), one purpose.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any


def planning_evidence_horizon(
    ctx: Any,
    *,
    governance_repo: pathlib.Path,
    subject_repo: pathlib.Path,
    scope: dict | None = None,
    remote_snapshot: Any = None,
) -> str:
    """One compact planning-evidence manifest; no second context pipeline. Contributes the
    durable task contract, lineage aliases, raw forensic refs and disclosed omissions exactly
    once to the shared reviewer prompt; plan and goal stay the canonical inline intent."""
    from ouroboros.observability import redact_projection
    # Read from its OWN module, not round-tripped through `plan_review`'s private alias:
    # that alias existed only because this function used to live there, and importing it
    # back made a dead-looking re-export load-bearing from another module.
    from ouroboros.tools.plan_review import _planning_handoff_path
    from ouroboros.tools.review_synthesis import normalize_plan_scope

    meta = getattr(ctx, "task_metadata", {})
    meta = meta if isinstance(meta, dict) else {}
    contract = getattr(ctx, "task_contract", {})
    contract = contract if isinstance(contract, dict) else {}
    task_id = str(getattr(ctx, "task_id", "") or meta.get("task_id") or "")
    root_id = str(meta.get("root_task_id") or task_id)
    refs: list[dict] = []
    if task_id:
        candidates = (
            pathlib.Path(ctx.drive_root) / "task_results" / f"{task_id}.json",
            _planning_handoff_path(ctx),
        )
        for candidate in candidates:
            if candidate.is_file():
                refs.append({"kind": candidate.stem, "path": str(candidate)})
    from ouroboros.remote_plan_review import remote_snapshot_evidence, snapshot_omission_rows

    omissions: list[dict] = []
    if not contract:
        omissions.append({
            "section": "task_contract",
            "reason": "not_available_in_tool_context",
        })
    # D7: a policy-filtered mirror is REVIEWED, with the omission named. A reviewer who
    # is not told about an exclusion reasons as if the tree were whole, which is the one
    # failure mode a "partial but honest" snapshot must not produce.
    omissions.extend(snapshot_omission_rows(remote_snapshot))
    payload = {
        "schema_version": 1,
        "canonical_intent": {
            "goal_ref": "Implementation Plan Under Review.Goal",
            "plan_ref": "Implementation Plan Under Review.Proposed Plan",
            "scope": normalize_plan_scope(scope),
            "task_contract": redact_projection(contract).value if contract else {},
        },
        "aliases": {
            "task_id": task_id,
            "root_task_id": root_id,
            "parent_task_id": str(meta.get("parent_task_id") or ""),
            "project_id": str(getattr(ctx, "project_id", "") or meta.get("project_id") or ""),
        },
        "roots": {
            "governance": str(governance_repo),
            "subject": str(subject_repo),
        },
        "forensic_refs": refs,
        "omissions_manifest": omissions,
        **remote_snapshot_evidence(remote_snapshot),
    }
    return (
        "## Planning Evidence Horizon\n\n```json\n"
        + json.dumps(payload, ensure_ascii=False, indent=2, default=str)
        + "\n```"
    )


__all__ = ["planning_evidence_horizon"]
