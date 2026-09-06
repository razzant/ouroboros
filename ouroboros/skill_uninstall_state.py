"""Uninstall tombstones for per-skill owner state (CPL4-C11, owner batch 3A).

Uninstalling a skill removes its payload but its ``state/skills/<name>/``
directory used to outlive it forever (only ``deps.json`` was cleared). The
hub uninstall paths now write an ``uninstalled.json`` tombstone, and the
startup sweep clears the dead state BY that mark — keeping ``grants.json``
(granted keys are OWNER authority, preserved across reinstall) and the
tombstone itself. A reinstall self-heals: the sweep sees a live payload and
retires the tombstone instead of sweeping.

The gateway's local delete path is untouched — it already removes the whole
state directory under the owner's explicit delete.
"""

from __future__ import annotations

import logging
import pathlib
import shutil
from typing import Any, Dict

from ouroboros.contracts.schema_versions import with_schema_version
from ouroboros.utils import atomic_write_json, utc_now_iso

log = logging.getLogger(__name__)

UNINSTALL_TOMBSTONE_FILENAME = "uninstalled.json"
# Owner authority survives the sweep (batch №8 3A): grants are the owner's
# durable key/permission decisions, not skill payload state.
_SWEEP_KEEP = frozenset({UNINSTALL_TOMBSTONE_FILENAME, "grants.json"})


def write_uninstall_tombstone(drive_root: pathlib.Path, name: str, *, source: str) -> None:
    """Durably mark a skill's payload as uninstalled. Never raises."""
    from ouroboros.skill_loader import SKILL_OWNER_STATE_SCHEMA_VERSION, skill_state_dir

    try:
        atomic_write_json(
            skill_state_dir(pathlib.Path(drive_root), name) / UNINSTALL_TOMBSTONE_FILENAME,
            with_schema_version(
                {"uninstalled_at": utc_now_iso(), "source": str(source or "")},
                SKILL_OWNER_STATE_SCHEMA_VERSION,
            ),
        )
    except Exception:
        log.debug("uninstall tombstone write failed for %s", name, exc_info=True)


def sweep_uninstalled_skill_state(drive_root: pathlib.Path) -> Dict[str, Any]:
    """Clear owner state of tombstoned skills; self-heal reinstalled ones.

    Fail-closed per entry: anything that cannot be removed is kept and
    reported, never half-guessed. A state dir WITHOUT a tombstone is never
    touched — the mark is the only authority to sweep by.
    """
    from ouroboros.skill_loader import find_skill

    report: Dict[str, Any] = {"swept": [], "restored": [], "errors": []}
    state_root = pathlib.Path(drive_root) / "state" / "skills"
    try:
        state_dirs = sorted(p for p in state_root.iterdir() if p.is_dir())
    except OSError:
        return report
    for state_dir in state_dirs:
        tombstone = state_dir / UNINSTALL_TOMBSTONE_FILENAME
        if not tombstone.exists():
            continue
        name = state_dir.name
        try:
            live = find_skill(pathlib.Path(drive_root), name) is not None
        except Exception:
            report["errors"].append({"skill": name, "error": "payload_probe_failed"})
            continue  # fail-closed: cannot prove the payload is gone
        if live:
            # Reinstalled since the tombstone landed: the mark is stale.
            try:
                tombstone.unlink()
                report["restored"].append(name)
            except OSError:
                report["errors"].append({"skill": name, "error": "tombstone_unlink_failed"})
            continue
        removed_any = False
        for entry in sorted(state_dir.iterdir()):
            if entry.name in _SWEEP_KEEP:
                continue
            try:
                if entry.is_dir() and not entry.is_symlink():
                    shutil.rmtree(entry)
                else:
                    entry.unlink()
                removed_any = True
            except OSError:
                report["errors"].append({"skill": name, "entry": entry.name,
                                         "error": "remove_failed"})
        if removed_any:
            report["swept"].append(name)
    return report


__all__ = [
    "UNINSTALL_TOMBSTONE_FILENAME",
    "sweep_uninstalled_skill_state",
    "write_uninstall_tombstone",
]
