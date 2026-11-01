"""Temporary working environments beside a copy (``<copy>.env``), whole folders only.

The one producer is ``body_candidate.process_environment``: it declares the folder disposable process
state and names its owning task and copy in the declaration: a purpose and an owner, never a promise that
its contents come back byte for byte. Nothing else enrolls: no folder name, no age alone,
no ``.venv``, ``/tmp`` or project environment. Such a folder goes once its owner is proven over, its copy
is gone, no producer retained anything in it and the owner has not kept it. The next process in a recreated
copy gets a fresh folder from the same producer; installed packages come back only if that work installs
them, and a refused preparation is that work's own ``CANDIDATE_ENVIRONMENT_UNAVAILABLE``.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
from typing import Any, Callable, Dict, List, Optional

log = logging.getLogger(__name__)

UNDECLARED = "undeclared_environment"
RETAINED = "retained_evidence"
ENV_SUFFIX = ".env"
_SEEN: Dict[str, Dict[str, Any]] = {}


def scratch_owner(entry: pathlib.Path) -> Optional[Dict[str, Any]]:
    """The owner a declaration recorded; ``None`` when absent or unreadable: unknown never enrolls."""
    from ouroboros.test_environment import SCRATCH_MARKER

    try:
        with (entry / SCRATCH_MARKER).open(encoding="utf-8") as stream:
            lines = stream.read(4096).splitlines()
        owner = json.loads(lines[1]) if len(lines) > 1 else None
    except (OSError, ValueError):
        return None
    return owner if isinstance(owner, dict) else None


def environment_verdict(entry: pathlib.Path, data_root: Any, live: Callable[[str], Any],
                        cutoff: Optional[float] = None) -> str:
    """Why this environment folder is not ordinary temporary space, or ``""``. Read-only.
    ``cutoff`` (the background pass) also requires the last use to be older than it."""
    from ouroboros.owner_keep import OWNER_KEEP, is_kept
    from ouroboros.subagent_worktrees import _task_proven_over
    from ouroboros.test_environment import SCRATCH_MARKER, scratch_retention

    if entry.is_symlink() or not entry.is_dir():
        return UNDECLARED
    owner = scratch_owner(entry)
    task_id = str((owner or {}).get("task_id") or "")
    copy = str((owner or {}).get("copy") or "")
    if not task_id or pathlib.Path(copy).resolve(strict=False) != entry.with_name(entry.name[:-len(ENV_SUFFIX)]).resolve(strict=False):
        return UNDECLARED
    if os.path.lexists(copy) or not _task_proven_over(task_id, data_root, live):
        return "in_use"
    if is_kept(data_root, task_id):
        return OWNER_KEEP
    if cutoff is not None and (entry / SCRATCH_MARKER).stat().st_mtime >= cutoff:
        return "in_use"
    # One walk per quiescent tree, in this process: a later look reads only the recorded witness.
    _SEEN[str(entry)] = scratch_retention(entry, previous=_SEEN.get(str(entry)), classify=True)
    return RETAINED if _SEEN[str(entry)]["retained"] else ""


def _declaration(entry: pathlib.Path) -> Any:
    from ouroboros.test_environment import SCRATCH_MARKER

    try:
        seen = (entry / SCRATCH_MARKER).lstat()
        return (seen.st_dev, seen.st_ino, seen.st_size, seen.st_mtime_ns)
    except OSError:
        return None


def retire_environment(entry: pathlib.Path, data_root: Any, live: Callable[[str], Any], copies_root: pathlib.Path,
                       cutoff: Optional[float] = None) -> str:
    """Remove one environment; why it stayed, or ``""``. The verdict, liveness probe included, is read
    before the owner's keep gate (``owner_keep``: no probe under the custody lock); inside it only recorded
    facts are read again: the declaration unchanged, the copy still gone, the owner still terminal, the
    retention witness unchanged. A stale positive deletes nothing."""
    from ouroboros.body_candidate import _recorded_terminal
    from ouroboros.owner_keep import gate
    from ouroboros.subagent_worktrees import _deletable, _force_rmtree
    from ouroboros.test_environment import scratch_retention

    owner = scratch_owner(entry) or {}
    declared = _declaration(entry)
    why = environment_verdict(entry, data_root, live, cutoff)
    if why:
        return why
    with gate(data_root, owner.get("task_id"), timeout_sec=0) as why:
        if why:
            return why
        if (_declaration(entry) != declared or scratch_owner(entry) != owner or os.path.lexists(str(owner.get("copy")))
                or not _recorded_terminal(str(owner.get("task_id")), data_root)
                or scratch_retention(entry, previous=_SEEN.get(str(entry)))["retained"]):
            return "changed"
        if not _deletable(entry, copies_root):
            return "other"
        _force_rmtree(entry)
        return "other" if os.path.lexists(entry) else ""


def environment_folders(copies_root: pathlib.Path) -> List[pathlib.Path]:
    """The ``*.env`` folders of the copies' root and its unindexed folder: one listing each, no walk."""
    from ouroboros.unindexed import COPIES

    found: List[pathlib.Path] = []
    for folder in (copies_root, copies_root / COPIES):
        if folder.is_dir() and not folder.is_symlink():
            found += [entry for entry in sorted(folder.iterdir()) if entry.name.endswith(ENV_SUFFIX)]
    return found


def prune_environments(data_root: Any, live: Callable[[str], Any], *, retention_days: Optional[int] = None,
                       stop: Optional[Callable[[], bool]] = None) -> Dict[str, Any]:
    """The custody pass's step: the shared retention term for finished work's declared environments."""
    from ouroboros.retention import age_cutoff, get_gc_retention_days
    from ouroboros.subagent_worktrees import _resolve_root

    cutoff = age_cutoff(get_gc_retention_days() if retention_days is None else retention_days)
    root = _resolve_root()
    report: Dict[str, Any] = {"removed": [], "kept": 0, "errors": 0}
    for entry in environment_folders(root):
        if stop is not None and stop():
            break
        try:
            why = retire_environment(entry, data_root, live, root, cutoff)
        except OSError:
            log.warning("Environment %s was not retired", entry, exc_info=True)
            report["errors"] += 1
            continue
        if why:
            report["kept"] += 1
        else:
            report["removed"].append(entry.name)
    return report
