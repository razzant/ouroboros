"""Staged import trees for in-process extensions, and their reclamation.

An extension is never imported from its payload directory: it is copied to a
per-load staging root under the skill state dir, so a rapid edit cannot be
served from stale bytecode and an unload can drop the whole tree. Staging leaves
carry their owner PID, which is what lets a concurrent worker tell a peer's
still-loading tree from a real orphan.
"""

from __future__ import annotations

import hashlib
import logging
import os
import pathlib
import shutil
import time
import uuid
from typing import Optional, Sequence

from ouroboros.extension_isolated_deps import is_skill_cache_path
from ouroboros.extension_registry_state import _extensions, _lock
from ouroboros.skill_loader import _SKILL_DIR_CACHE_NAMES, LoadedSkill, skill_state_dir

log = logging.getLogger(__name__)


def _plugin_entry_path(skill: LoadedSkill) -> Optional[pathlib.Path]:
    """Resolve manifest.entry inside the skill directory."""
    entry = str(skill.manifest.entry or "").strip()
    if not entry:
        return None
    candidate = (skill.skill_dir / entry).resolve()
    try:
        candidate.relative_to(skill.skill_dir.resolve())
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


def _module_key(skill_name: str) -> str:
    digest = hashlib.sha1(str(skill_name or "").encode("utf-8", errors="replace")).hexdigest()[:16]
    return f"ouroboros._extensions.m_{digest}"


def _purge_extension_bytecode(skill_dir: pathlib.Path) -> None:
    """Drop bytecode so rapid edits reload fresh source."""
    for pycache in skill_dir.rglob("__pycache__"):
        if pycache.is_dir():
            shutil.rmtree(pycache, ignore_errors=True)


def _stage_extension_import_tree(
    skill: LoadedSkill,
    *,
    state_dir: pathlib.Path,
    entry_path: pathlib.Path,
) -> tuple[pathlib.Path, pathlib.Path]:
    """Stage an extension under a fresh import root to avoid stale module reuse."""
    resolved_root = skill.skill_dir.resolve()
    relative_entry = entry_path.relative_to(resolved_root)
    for path in sorted(skill.skill_dir.rglob("*")):
        if is_skill_cache_path(path, resolved_root):
            continue
        if not path.is_symlink():
            continue
        try:
            resolved = path.resolve()
            resolved.relative_to(resolved_root)
        except Exception as exc:
            raise RuntimeError(
                f"extension {skill.name!r} contains a symlink that resolves outside the skill tree: {path}"
            ) from exc
    child_import_base = os.environ.get("OUROBOROS_EXTENSION_IMPORT_ROOT_BASE", "")
    if os.environ.get("OUROBOROS_EXTENSION_PROCESS_CHILD") == "1" and child_import_base:
        import_root = pathlib.Path(child_import_base) / uuid.uuid4().hex
    else:
        # Tag the staged-tree leaf with the OWNER PID. Under MAX_WORKERS>1 every
        # worker stages concurrently into this SHARED dir; the per-PID prefix lets
        # _sweep_stale_extension_imports tell a peer's still-loading tree (owner
        # alive / fresh) from a real orphan (owner dead + past grace) instead of
        # rmtree-ing a sibling mid-load (which would FileNotFoundError its
        # exec_module and silently drop the skill in that worker).
        import_root = state_dir / "__extension_imports" / f"{os.getpid()}-{uuid.uuid4().hex}"
    staged_skill_dir = import_root / "skill"
    import_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        skill.skill_dir,
        staged_skill_dir,
        ignore=shutil.ignore_patterns(*_SKILL_DIR_CACHE_NAMES),
    )
    _purge_extension_bytecode(staged_skill_dir)
    staged_entry = (staged_skill_dir / relative_entry).resolve()
    staged_entry.relative_to(staged_skill_dir.resolve())
    return import_root, staged_entry


# Grace window before a per-PID staged tree whose owner process is already gone is
# reaped: a just-spawned peer worker can still be mid-copytree of a fresh tree.
# Value-mirrored from supervisor/workers.py:_SPAWN_GRACE_SEC (do NOT import supervisor
# into ouroboros/ — layering inversion); it only affects reclaim latency, not safety.
_IMPORT_SWEEP_GRACE_SEC = 90.0


def _sweep_stale_extension_imports(
    drive_root: pathlib.Path,
    skill_name: str,
    *,
    keep: Sequence[pathlib.Path] = (),
) -> None:
    """Remove orphan staged import trees without touching skill state/payload.

    Per-PID safe: staged-tree leaves are named ``<owner_pid>-<uuid>`` (see
    _stage_extension_import_tree), so under MAX_WORKERS>1 — where every worker stages
    into this SHARED dir concurrently — a leaf is reaped ONLY when its owner process is
    dead AND its mtime is past the spawn grace. A peer's still-loading tree (owner
    alive, or fresh within grace) is left alone, so its exec_module never hits a
    FileNotFoundError from a sibling's sweep. Legacy bare-uuid leaves (no parseable
    owner) keep the prior keep-set-only behaviour."""
    root = skill_state_dir(drive_root, skill_name) / "__extension_imports"
    if not root.exists() or not root.is_dir():
        return
    keep_resolved = set()
    for path in keep or ():
        try:
            keep_resolved.add(path.resolve(strict=False))
        except OSError:
            pass
    with _lock:
        bundle = _extensions.get(skill_name)
        if bundle and bundle.import_root:
            try:
                keep_resolved.add(pathlib.Path(bundle.import_root).resolve(strict=False))
            except OSError:
                pass
    try:
        from ouroboros.platform_layer import pid_is_alive as _pid_is_alive
    except Exception:
        _pid_is_alive = None
    now = time.time()
    for child in list(root.iterdir()):
        try:
            resolved = child.resolve(strict=False)
        except OSError:
            resolved = child
        if resolved in keep_resolved:
            continue
        if not child.is_dir():
            continue
        # Cross-process safety (the MAX_WORKERS>1 staging race): only reap a per-PID
        # tree whose OWNER process is DEAD *and* whose mtime is past the spawn grace.
        # Never delete a tree a live (or just-spawned) peer worker is mid-loading.
        owner_pid = None
        try:
            parsed = int(child.name.split("-", 1)[0])
            # A real per-PID leaf is "<pid>-<uuid>" with a plausible PID. A legacy
            # bare-uuid that happens to be all digits would int-parse to a huge number
            # (and OverflowError os.kill); out-of-range -> treat as legacy (fall through
            # to the keep-set reap), never feed an implausible value to pid_is_alive.
            owner_pid = parsed if 0 < parsed < 2_147_483_648 else None
        except (ValueError, IndexError):
            owner_pid = None
        if owner_pid is not None:
            if _pid_is_alive is None:
                continue  # cannot verify liveness -> conservatively keep (never reap unverified)
            try:
                if _pid_is_alive(owner_pid):
                    continue  # owner still running -> tree may be mid-load
                if (now - child.stat().st_mtime) < _IMPORT_SWEEP_GRACE_SEC:
                    continue  # within spawn grace -> a just-spawned peer may be staging
            except Exception:
                continue  # cannot verify liveness/age -> conservative skip (never reap unverified)
        shutil.rmtree(child, ignore_errors=True)
