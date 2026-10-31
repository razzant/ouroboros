"""The one writer of patch apply/reject verdicts (artifact + custody row).

Extracted from ``tools/subagent_integration.py`` at its module-size ceiling:
one coherent WRITE concern shared by both patch pipelines
(``integrate_subagent_patch`` and ``integrate_delegated_patch``, which mints
``run_<rid>`` subjects). The subagent-integration module re-exports it, so
every historical import and monkeypatch target keeps working.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pathlib
import stat
from typing import Any, Dict, Iterator, List

from ouroboros.artifact_storage import temporary_sibling
from ouroboros.artifacts import task_artifact_dir_path, task_id_for_artifacts
from ouroboros.tools.registry import ToolContext
from ouroboros.utils import atomic_write_json, utc_now_iso

log = logging.getLogger(__name__)


def retained_verdicts(canonical_root: Any, task_id: str, child_task_id: str) -> Iterator[Dict[str, Any]]:
    """Read TASK's retained verdict on CHILD from its host-layout artifact stores.

    Include collision copies named by their bytes at the publication owner.
    Missing is none; unreadable, linked or mismatched material raises. The existing
    custody lock keeps drive promotion/GC from moving a file between the reads.
    This reader only refuses a second decision; it never attests that a row landed.
    """
    from ouroboros.task_custody import _collision, task_artifact_stores, task_custody_lock

    with task_custody_lock(canonical_root, task_id) as locked:
        if not locked:
            raise OSError(f"{task_id}'s retained decision stores are busy")
        for store in task_artifact_stores(canonical_root, task_id):
            path = store / f"subagent_patch_verdict_{child_task_id}.json"
            if store.resolve(strict=False) != store:
                raise ValueError(f"{store} is not an own artifact store")
            # Publication retains changed bytes at _collision(path, sha256(bytes)).
            # A dotted child's own verdict also shares this prefix, but is not a
            # collision copy. Do not read that unrelated decision as this child's.
            variants = [path, *store.glob(f"{path.stem}.*{path.suffix}")]
            for variant in variants:
                if variant.is_symlink():
                    raise ValueError(f"{variant} is not an own artifact-store file")
                try:
                    data = variant.read_bytes()
                except FileNotFoundError:
                    continue
                if variant != path and _collision(path, hashlib.sha256(data).hexdigest()) != variant:
                    continue
                verdict = json.loads(data.decode("utf-8"))
                if not (isinstance(verdict, dict) and verdict.get("parent_task_id") == task_id
                        and verdict.get("child_task_id") == child_task_id):
                    raise ValueError(f"{variant} is not {task_id}'s verdict on {child_task_id}")
                yield verdict


def _retain_replaced(path: pathlib.Path) -> None:
    """Keep the verdict bytes a new write replaces, beside it, as publication keeps changed bytes.

    The name is ``task_custody._collision`` of those bytes, which ``retained_verdicts`` and
    promotion already resolve. It is created once, never replaced: a name holding other bytes
    raises, and the caller then keeps the earlier file instead of overwriting it. A path that
    is not a regular file holds no bytes of this writer's to keep.
    """
    from ouroboros.task_custody import _collision

    try:
        if not stat.S_ISREG(path.lstat().st_mode):
            return
    except FileNotFoundError:
        return
    data = path.read_bytes()
    beside = _collision(path, hashlib.sha256(data).hexdigest())
    temporary = temporary_sibling(beside)  # the atomic signature the temp sweep reaps after a hard kill
    try:
        temporary.write_bytes(data)
        os.link(temporary, beside)  # create-only, like publication's placement
    except FileExistsError:
        if beside.is_symlink() or beside.read_bytes() != data:
            raise OSError(f"{beside.name} already holds other bytes") from None
    finally:
        temporary.unlink(missing_ok=True)


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
    lineage: Dict[str, str] | None = None,
    observation: Dict[str, Any] | None = None,
    failures: List[str] | None = None,
) -> str:
    """``parent_task_id`` names the deciding task. An owner-Continue successor deciding
    its predecessor's child adds ``lineage``: the authority, the child's recorded parent
    and the exact result hash it judged. The child's own records are never rewritten.
    ``observation`` keeps what the check saw beside the decision (``observed_outcome``,
    ``path_findings``, ``path_findings_note``); only ``observed_outcome`` also rides the
    custody row. An earlier verdict on the same child stays readable beside this one.
    ``failures`` receives this write's own failure facts, the names its records use
    (``verdict_artifact_write_failed``, ``custody_row_write_failed``), for the caller to report."""
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
        **(observation or {}),
        **(lineage or {}),
    }
    observed = str(verdict.get("observed_outcome") or "")
    path = art_dir / f"subagent_patch_verdict_{child_task_id}.json"
    artifact_write_failed = False
    try:
        _retain_replaced(path)
        atomic_write_json(path, verdict, trailing_newline=True)
    except Exception:
        artifact_write_failed = True
    # D-trace: the apply/reject decision also lands as a typed row in the
    # custody event log, ONE replayable store for both patch pipelines (the
    # delegated pipeline's run-keyed PATCH_DISPOSED row rides beside it); the
    # decision gate replays it to refuse a second decision on the same result.
    # A failed artifact write is disclosed on the row rather than dying as a
    # silent "".
    custody_row_landed = False
    try:
        from ouroboros import delegate_custody as custody

        # The canonical (budget) custody root, like every other custody write —
        # a row on a child's drive_root cannot outlive a pruned child drive, and
        # the decision reader replays the canonical log.
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
            # An acceptance names what the check saw; the decision stays the disposition.
            **({"observed_outcome": observed} if observed else {}),
            **(lineage or {}),
        })
    except Exception:
        log.warning("subagent patch verdict custody row failed", exc_info=True)
    if not custody_row_landed and not artifact_write_failed:
        # `custody.emit` reports durable-append failure as a FALSE RETURN, not
        # an exception (delegate_custody's own contract) — and the custody row
        # is what the decision reader replays: without this, a lost row reads
        # as "nothing recorded" when an attestation was made. Disclose the miss
        # on the artifact (absence ≠ clean); an unwritten artifact has no place
        # for it, and the earlier file it kept must not be replaced here.
        try:
            verdict["custody_row_write_failed"] = True
            atomic_write_json(path, verdict, trailing_newline=True)
        except Exception:
            log.debug("subagent patch verdict custody-miss disclosure failed", exc_info=True)
    if failures is not None:
        failures.extend(fact for fact, failed in (("verdict_artifact_write_failed", artifact_write_failed),
                                                  ("custody_row_write_failed", not custody_row_landed)) if failed)
    return "" if artifact_write_failed else str(path)
