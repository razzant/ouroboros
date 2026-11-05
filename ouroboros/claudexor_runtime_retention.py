"""Retire the managed engine program copies no daemon can still run (owner Q6 = A, 2026-10-17).

Every engine update unpacks a new immutable runtime tree into ``DATA_DIR/state/cx``
(``claudexor_runtime``) and nothing removed the old ones. ``prune_unpinned_runtimes`` runs on the
600 s custody pass. It keeps the pinned install (the next-spawn selection), every tree of the
version the owned daemon last proved serving (``claudexor_daemon.owned_engine_version``: a
lazily loading daemon keeps reading its own tree), the Node trees and the pinned archive; it
removes the other version trees and archives, and what an interrupted install left (staging,
displaced trees, download and seed temporaries) once older than an hour. While no handshake of
this process proved the serving version, or no pin is published, it removes nothing. Under the
install lock a doomed tree is only renamed aside, so an install never meets half of one; the
deletion follows outside the lock. Going back to an older Ouroboros downloads its engine again;
engine state under ``DATA_DIR/claudexor`` is never touched here.
"""
from __future__ import annotations

import os
import pathlib
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

_INSTALLED = re.compile(r"^\d+\.\d+\.\d+-[0-9a-f]{12}$")  # ``ClaudexorRuntimePin.install_name``
_LEFTOVER_AGE_SEC = 3600.0  # an install still running keeps its staging (a physical bound, not judgment)


def _leftover(name: str) -> bool:
    """What an install, a download or a seed copy of ``claudexor_runtime`` names while in flight."""
    return name.startswith((".tmp-", ".old-")) or name.startswith(".") and (".tmp." in name or ".seed." in name)


def _doomed(entry: pathlib.Path, root: pathlib.Path, pin: Any, serving: str, cutoff: float) -> bool:
    if entry.is_symlink():
        return False  # never followed: nothing this module writes
    if _leftover(entry.name):
        info = entry.lstat()
        return max(info.st_mtime, info.st_ctime) < cutoff
    if entry.parent == root:
        return (entry.is_dir() and bool(_INSTALLED.fullmatch(entry.name)) and entry.name != pin.install_name
                and not entry.name.startswith(f"{serving}-"))
    return (entry.parent == root / "cache" and entry.is_file() and entry.name.endswith(".tar.gz")
            and entry.name != pin.archive_name)


def prune_unpinned_runtimes(*, now: Optional[float] = None) -> Dict[str, Any]:
    """One pass over ``state/cx``, its ``node/`` and ``cache/`` (module docstring); never raises. Reports the
    shared retirer shape (``maintenance_jobs.pass_report``): ``removed`` names, ``released`` their count,
    ``holds`` the doomed entries this pass could not remove (``removal_failed``: in use or survived their
    removal; the next pass retries them), ``errors`` nothing of its own; a pass that judged nothing says why
    in ``skipped``."""
    from ouroboros import claudexor_runtime as runtime
    from ouroboros.claudexor_daemon import owned_engine_version
    from ouroboros.maintenance_jobs import Holds, pass_report
    from ouroboros.platform_layer import acquire_exclusive_file_lock, release_exclusive_file_lock
    from ouroboros.force_remove import force_rmtree as _force_rmtree

    removed: List[str] = []
    holds = Holds()

    def shaped(**extra: Any) -> Dict[str, Any]:
        return {**pass_report(removed, holds), "removed": removed, **extra}

    try:
        serving, pin, root = owned_engine_version(), runtime.get_runtime_manager().pin, runtime.managed_runtime_root()
    except Exception as exc:  # a pin that cannot be read keeps everything
        return shaped(skipped=f"pin_unreadable: {type(exc).__name__}", rested=True)
    if not serving:
        return shaped(skipped="serving_version_unproven", rested=True)
    if pin is None or not root.is_dir():
        return shaped(skipped="nothing_installed", rested=True)
    lock = root / runtime._INSTALL_LOCK_FILENAME
    fd = acquire_exclusive_file_lock(lock, timeout_sec=0, stale_sec=runtime._INSTALL_LOCK_STALE_SEC,
                                     metadata=f"pid={os.getpid()} retiring unpinned runtimes\n")
    if fd is None:
        return shaped(skipped="install_in_progress", rested=True)
    aside: List[Tuple[str, pathlib.Path]] = []
    try:
        cutoff = (time.time() if now is None else now) - _LEFTOVER_AGE_SEC
        for folder in (root, root / "node", root / "cache"):
            for entry in sorted(folder.iterdir()) if folder.is_dir() else ():
                try:
                    if not _doomed(entry, root, pin, serving, cutoff):
                        continue
                    if not entry.is_dir():
                        entry.unlink()
                        removed.append(entry.name)
                        continue
                    moved = entry if entry.name.startswith(".") else entry.with_name(f".old-{uuid.uuid4().hex[:8]}")
                    if moved != entry:
                        os.replace(entry, moved)
                    aside.append((entry.name, moved))
                except OSError as exc:  # in use (Windows) or vanished: kept, the next pass retries
                    holds.note("removal_failed", f"{entry.name}: {type(exc).__name__}")
    finally:
        release_exclusive_file_lock(lock, fd)
    for name, moved in aside:
        _force_rmtree(moved)
        if moved.exists():  # renamed aside: a later pass retries it as a leftover
            holds.note("removal_failed", f"{name}: survived its removal")
        else:
            removed.append(name)
    return shaped()
