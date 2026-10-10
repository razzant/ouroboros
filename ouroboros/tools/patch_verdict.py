"""The one writer of patch apply/reject verdicts (artifact + custody row).

Extracted from ``tools/subagent_integration.py`` at its module-size ceiling:
one coherent WRITE concern shared by both patch pipelines
(``integrate_subagent_patch`` and ``integrate_delegated_patch``, which mints
``run_<rid>`` subjects). The subagent-integration module re-exports it, so
every historical import and monkeypatch target keeps working.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from ouroboros.artifacts import task_artifact_dir_path, task_id_for_artifacts
from ouroboros.tools.registry import ToolContext
from ouroboros.utils import atomic_write_json, utc_now_iso

log = logging.getLogger(__name__)


def write_patch_verdict(
    ctx: ToolContext,
    child_task_id: str,
    *,
    outcome: str,
    reason: str,
    files: List[str],
    manifest: Dict[str, Any],
    applied: bool,
    conflicts: List[str],
    protected: List[str],
    target: str = "",
) -> str:
    parent_task_id = task_id_for_artifacts(ctx)
    art_dir = task_artifact_dir_path(getattr(ctx, "drive_root", "."), parent_task_id, create=True)
    verdict = {
        "schema_version": 1,
        "created_at": utc_now_iso(),
        "tool": "integrate_subagent_patch",
        "parent_task_id": parent_task_id,
        "child_task_id": child_task_id,
        "outcome": outcome,
        "applied": bool(applied),
        "reason": str(reason or ""),
        "target_root": str(target or ""),
        "files": list(files or []),
        "protected_matches": list(protected or []),
        "conflicts": list(conflicts or []),
        "patch_sha256": str((manifest or {}).get("sha256") or ""),
        "diffstat": str((manifest or {}).get("diffstat") or ""),
    }
    path = art_dir / f"subagent_patch_verdict_{child_task_id}.json"
    artifact_write_failed = False
    try:
        atomic_write_json(path, verdict, trailing_newline=True)
    except Exception:
        artifact_write_failed = True
    # D-trace: the apply/reject decision also lands as a typed row in the
    # custody event log, so the acceptance packet builds the disposition
    # section from ONE replayable store for both patch pipelines (the
    # delegated pipeline's run-keyed PATCH_DISPOSED row rides beside it; the
    # scanner splits by `tool`, so nothing double-counts). A failed artifact
    # write is disclosed on the row rather than dying as a silent "".
    custody_row_landed = False
    try:
        from ouroboros import delegate_custody as custody

        # The canonical (budget) custody root, like every other custody write —
        # a row on a child's drive_root cannot outlive a pruned child drive, and
        # the acceptance-packet reader replays the canonical log.
        custody_row_landed = custody.emit(custody.custody_root(ctx), "delegate_run_patch_verdict", {
            "run_id": "",
            "task_id": parent_task_id,
            "child_task_id": child_task_id,
            # The delegated pipeline mints its verdict subject as run_<rid>
            # (every _integrate_delegated_patch call site); classifying at
            # the ONE writer keeps that convention local instead of leaking
            # prefix-matching into readers.
            "pipeline": "delegated" if child_task_id.startswith("run_") else "subagent",
            "disposition": outcome,
            "applied": bool(applied),
            "reason": str(reason or "")[:600],
            "patch_sha256": str(verdict.get("patch_sha256") or ""),
            "verdict_artifact_write_failed": artifact_write_failed,
            # Where the verdict was judged (a shared child folder need not be the
            # parent's); absent when no target is known, as on legacy rows.
            **({"target_root": verdict["target_root"]} if verdict["target_root"] else {}),
        })
    except Exception:
        log.warning("subagent patch verdict custody row failed", exc_info=True)
    if not custody_row_landed:
        # `custody.emit` reports durable-append failure as a FALSE RETURN, not
        # an exception (delegate_custody's own contract) — and the custody row
        # IS the acceptance-packet source: without this, an empty dispositions
        # section reads as "nothing recorded" when an attestation was made and
        # silently lost. Disclose the miss on the artifact so the
        # attested-not-gated contract stays honest (absence ≠ clean).
        try:
            verdict["custody_row_write_failed"] = True
            atomic_write_json(path, verdict, trailing_newline=True)
        except Exception:
            log.debug("subagent patch verdict custody-miss disclosure failed", exc_info=True)
    return "" if artifact_write_failed else str(path)
