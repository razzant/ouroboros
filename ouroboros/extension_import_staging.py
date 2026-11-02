"""Staged import trees for in-process extensions, and their reclamation.

Completed trees are addressed by reviewed content and runtime/dependency
identity. Processes share the bytes, keeping PID references until unload;
reclamation requires both an obsolete identity and dead owners. Publication,
reference acquisition and sweeping use the same cross-process lock.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pathlib
import shutil
import sys
import time
import uuid
from contextlib import contextmanager
from typing import Optional, Sequence

from ouroboros.config import WORKER_SPAWN_GRACE_SEC
from ouroboros.extension_isolated_deps import is_skill_cache_path
from ouroboros.extension_registry_state import _extensions, _lock
from ouroboros.skill_loader import _SKILL_DIR_CACHE_NAMES, LoadedSkill, skill_state_dir
from ouroboros.utils import atomic_write_json, read_json_dict

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
    """Acquire a complete immutable import tree, copying only on a cache miss."""
    resolved_root = skill.skill_dir.resolve()
    relative_entry = entry_path.relative_to(resolved_root)
    identity = extension_import_identity(skill)
    import_root = state_dir / "__extension_imports" / identity
    with extension_cache_lock(state_dir, "imports"):
        if not skill.skill_dir.is_dir():
            raise RuntimeError(f"extension {skill.name!r} payload was removed before import acquisition")
        if not (import_root / "complete.json").is_file():
            _copy_extension_tree(skill, import_root)
        owners = import_root / "owners"
        owners.mkdir(exist_ok=True)
        (owners / str(os.getpid())).touch()
    staged_entry = (import_root / "skill" / relative_entry).resolve()
    staged_entry.relative_to((import_root / "skill").resolve())
    return import_root, staged_entry


def extension_import_identity(skill: LoadedSkill) -> str:
    """The installer's fingerprint owns dependency changes, not the review hash."""
    from ouroboros.marketplace.isolated_deps import FINGERPRINT_FILENAME

    executable = pathlib.Path(sys.executable).resolve()
    st = executable.stat()
    fingerprint = skill.skill_dir / ".ouroboros_env" / FINGERPRINT_FILENAME
    try:
        deps = fingerprint.read_bytes()
    except FileNotFoundError:
        deps = b""
    runtime = hashlib.sha256(repr((
        str(executable), st.st_size, st.st_mtime_ns, sys.version,
        sys.prefix, sys.platform, str(skill.skill_dir.resolve()),
    )).encode("utf-8") + deps).hexdigest()
    return f"sha256-{skill.content_hash}-{runtime}"


@contextmanager
def extension_cache_lock(state_dir: pathlib.Path, purpose: str):
    from ouroboros.platform_layer import acquire_exclusive_file_lock, release_exclusive_file_lock

    # Keep the cache lock beside state so deleting a skill cannot replace its inode.
    path = (state_dir.parent / f".{state_dir.name}-extension-imports.lock"
            if purpose == "imports" else state_dir / f".extension-{purpose}.lock")
    fd = acquire_exclusive_file_lock(path, timeout_sec=30, owner_aware_stale=True)
    if fd is None:
        raise RuntimeError(f"extension {purpose} cache lock unavailable")
    try:
        yield
    finally:
        release_exclusive_file_lock(path, fd)


def _copy_extension_tree(skill: LoadedSkill, import_root: pathlib.Path) -> None:
    resolved_root = skill.skill_dir.resolve()
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
    import_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = import_root.parent / f"{os.getpid()}-{uuid.uuid4().hex}"
    try:
        shutil.copytree(skill.skill_dir, temporary / "skill",
                        ignore=shutil.ignore_patterns(*_SKILL_DIR_CACHE_NAMES))
        atomic_write_json(temporary / "complete.json", {
            "identity": import_root.name, "content_hash": skill.content_hash,
            "owner_pid": os.getpid(),
        })
        # An incomplete named tree is never published by this writer.
        if import_root.exists():
            shutil.rmtree(import_root)
        temporary.rename(import_root)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def release_extension_import_tree(import_root: pathlib.Path) -> None:
    """Unload releases this process's reference, retaining reusable bytes."""
    if import_root.name.startswith("sha256-"):
        with extension_cache_lock(import_root.parent.parent, "imports"):
            (import_root / "owners" / str(os.getpid())).unlink(missing_ok=True)
    else:
        shutil.rmtree(import_root, ignore_errors=True)


def declared_catalog_inputs(skill: LoadedSkill) -> dict | None:
    """An opt-in declaration is exhaustive; unenumerated inputs are not cached.

    Catalog settings are a fixed snapshot of the declared keys (ordinary dispatch reads
    current settings); registration that depends on files, time, network or side effects
    must omit the declaration.
    """
    declaration = skill.manifest.raw_extra.get("catalog_inputs")
    if not isinstance(declaration, dict) or set(declaration) != {"settings", "environment"}:
        return None
    if any(not isinstance(keys, list) or any(not isinstance(k, str) or not k for k in keys)
           for keys in declaration.values()):
        return None
    return declaration


def catalog_identity(skill: LoadedSkill, settings: dict, environ: dict, grants: dict) -> str | None:
    from ouroboros.extension_health import code_stamp

    declaration = declared_catalog_inputs(skill)
    if declaration is None:
        return None
    inputs = {
        "runtime": extension_import_identity(skill),
        "host": code_stamp(),
        "settings": {k: settings[k] for k in declaration["settings"] if k in settings},
        "environment": {k: environ[k] for k in declaration["environment"] if k in environ},
        "grants": grants,
    }
    # Values can be credentials: only their digest is durable, never the inputs.
    return hashlib.sha256(json.dumps(inputs, sort_keys=True).encode("utf-8")).hexdigest()


# Grace window before a per-PID staged tree whose owner process is already gone is
# reaped: a just-spawned peer worker can still be mid-copytree of a fresh tree. The
# pool's spawn grace from the config SSOT (supervisor/ must not be imported into
# ouroboros/ — layering inversion); it only affects reclaim latency, not safety.
_IMPORT_SWEEP_GRACE_SEC = WORKER_SPAWN_GRACE_SEC


def _sweep_stale_extension_imports(
    drive_root: pathlib.Path,
    skill_name: str,
    *,
    keep: Sequence[pathlib.Path] = (),
    keep_hashes: Sequence[str] = (),
) -> None:
    """Remove orphan staged import trees without touching skill state/payload.

    Content trees (``sha256-<identity>``, see _stage_extension_import_tree) are shared by
    boots and workers: one is reaped only when no live extension keeps its content hash
    AND every owner PID in its ``complete.json`` and ``owners/`` is dead. Per-PID leaves
    of older bodies (``<owner_pid>-<uuid>``) are reaped only when their owner is dead AND
    their mtime is past the spawn grace, so a peer still loading one is left alone;
    legacy bare-uuid leaves keep the keep-set-only behaviour."""
    state_dir = skill_state_dir(drive_root, skill_name)
    root = state_dir / "__extension_imports"
    if not root.exists() or not root.is_dir():
        return
    with extension_cache_lock(state_dir, "imports"):
        _sweep_import_trees(root, skill_name, keep, keep_hashes)


def _sweep_import_trees(root, skill_name, keep, keep_hashes):
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
        if child.name.startswith("sha256-"):
            complete = read_json_dict(child / "complete.json")
            if not complete or complete.get("content_hash") in keep_hashes:
                continue
            owners = {int(complete["owner_pid"])}
            owners.update(int(p.name) for p in (child / "owners").glob("*") if p.name.isdigit())
            if _pid_is_alive is None or any(_pid_is_alive(pid) for pid in owners):
                continue
            shutil.rmtree(child, ignore_errors=True)
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
