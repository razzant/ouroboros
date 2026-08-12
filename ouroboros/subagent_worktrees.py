"""Lifecycle for acting-subagent ``self_worktree`` checkouts.

Acting (mutative) subagents that modify the Ouroboros body itself run inside an
isolated ``git worktree`` checked out from the parent's base commit, under a root
that lives OUTSIDE ``repo/`` and ``data/``. The child writes only there and
returns a ``workspace.patch``; the parent integrates and is the sole committer.

git has no automatic worktree garbage collection, so we keep a durable JSON
registry (``data/state/subagent_worktrees.json``) and prune orphans on startup.
All worktree mutations are serialized by a portable cross-process lock because
``git worktree add/remove/prune`` mutate shared ``.git/worktrees`` metadata and
the existing repo git lock is drive-root scoped, not ``.git`` scoped.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import stat
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from ouroboros.platform_layer import acquire_exclusive_file_lock, release_exclusive_file_lock
from ouroboros.utils import atomic_write_json
from ouroboros.config import DATA_DIR, get_subagent_projects_root, get_subagent_worktree_root
from ouroboros.retention import age_cutoff, get_gc_retention_days

_REGISTRY_NAME = "subagent_worktrees.json"
_LOCK_NAME = ".worktree_ops.lock"
_LOCK_TIMEOUT_SEC = 120.0
_LOCK_STALE_SEC = 600.0
_BRANCH_PREFIX = "subagent/"

# Serializes worktree mutations within this process; the on-disk lock serializes
# across processes (parent worker, supervisor startup prune, etc.). REENTRANT:
# the startup prune holds the registry-root lock while taking each affected
# REPO's lock in turn. The nesting order is always root -> repo and nothing ever
# takes them the other way round, so no deadlock cycle exists.
_inproc_lock = threading.RLock()


# --------------------------------------------------------------------------- #
# Paths and registry
# --------------------------------------------------------------------------- #
def _data_dir(data_dir: Optional[Any] = None) -> Path:
    if data_dir:
        return Path(data_dir)
    env = os.environ.get("OUROBOROS_DATA_DIR")
    if env:
        return Path(env)
    return Path(DATA_DIR)


def _registry_path(data_dir: Optional[Any] = None) -> Path:
    return _data_dir(data_dir) / "state" / _REGISTRY_NAME


def _resolve_root(worktree_root: Optional[Any] = None) -> Path:
    root = Path(worktree_root) if worktree_root else Path(get_subagent_worktree_root())
    return root.expanduser().resolve()


def _is_within(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except (ValueError, OSError):
        return False


def _assert_root_isolated(root: Path, repo_dir: Path, data_dir: Path) -> None:
    """Refuse a worktree root that overlaps the live repo or runtime data."""
    if _is_within(root, repo_dir) or _is_within(repo_dir, root):
        raise ValueError(f"subagent worktree root {root} overlaps the Ouroboros repo {repo_dir}")
    if _is_within(root, data_dir) or _is_within(data_dir, root):
        raise ValueError(f"subagent worktree root {root} overlaps runtime data {data_dir}")


def _safe_name(task_id: Any) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", str(task_id or "").strip())
    safe = safe or f"wt_{int(time.time())}"
    # Bound the path component so an arbitrary-length input (e.g. a project display name,
    # which is not length-validated upstream) never hits ENAMETOOLONG on mkdir. On
    # truncation keep a short hash of the full slug so two long names with the same prefix
    # do not silently collide.
    if len(safe) > 64:
        import hashlib
        digest = hashlib.sha256(safe.encode("utf-8")).hexdigest()[:8]
        safe = f"{safe[:55]}_{digest}"
    return safe


def _load_registry(data_dir: Optional[Any] = None) -> List[Dict[str, Any]]:
    path = _registry_path(data_dir)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return []
    entries = raw.get("worktrees") if isinstance(raw, dict) else raw
    if isinstance(entries, list):
        return [e for e in entries if isinstance(e, dict)]
    return []


def _save_registry(entries: List[Dict[str, Any]], data_dir: Optional[Any] = None) -> None:
    path = _registry_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, {"worktrees": entries}, trailing_newline=True)


# --------------------------------------------------------------------------- #
# Locking
# --------------------------------------------------------------------------- #
def _ops_lock_path(target: Any) -> Path:
    """Where the cross-process worktree-ops lockfile for ``target`` lives.

    A git REPO locks on its own git common directory
    (``<repo>/.git/.worktree_ops.lock``), because that is what the mutations
    actually contend for: ``git worktree add|remove|prune`` all rewrite the
    SAME ``.git/worktrees`` metadata, no matter which registry owns the
    checkout. Keying the lockfile on the worktree ROOT instead gave the
    subagent registry and the durable thread registry two DIFFERENT lockfiles
    over one ``.git`` — the "every worktree owner serializes on the same lock"
    promise below was not actually kept, and two owners could rewrite that
    metadata concurrently.

    A non-repo directory (the worktree root itself, the genesis projects root)
    keeps a lockfile of its own: those operations contend for a NAME under that
    root, not for git metadata.
    """
    path = Path(target)
    git_path = path / ".git"
    if git_path.is_dir():
        return git_path / _LOCK_NAME
    if git_path.is_file():
        # A linked worktree: ``.git`` is a file pointing at
        # ``<main>/.git/worktrees/<name>``. Lock the COMMON dir so a caller that
        # handed us a checkout still meets the main repo's owners.
        try:
            text = git_path.read_text(encoding="utf-8").strip()
            if text.startswith("gitdir:"):
                linked = Path(text.split(":", 1)[1].strip())
                if not linked.is_absolute():
                    linked = (path / linked).resolve()
                for parent in linked.parents:
                    if parent.name == ".git":
                        return parent / _LOCK_NAME
        except OSError:
            pass
    return path / _LOCK_NAME


@contextlib.contextmanager
def _ops_lock(target: Any, mkdir_root: Optional[Any] = None):
    """Serialize worktree mutations in-process (an RLock) and across processes
    via the shared portable file-lock SSOT (platform_layer).

    ``target`` is the REPO whose ``.git/worktrees`` metadata the operation
    mutates (see :func:`_ops_lock_path`); pass a plain directory only when the
    operation contends for a name under it rather than for git metadata.
    ``mkdir_root`` is created if given — the worktree root must exist before a
    checkout is placed in it.
    """
    if mkdir_root is not None:
        Path(mkdir_root).mkdir(parents=True, exist_ok=True)
    lock_path = _ops_lock_path(target)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with _inproc_lock:
        fd = acquire_exclusive_file_lock(
            lock_path,
            timeout_sec=_LOCK_TIMEOUT_SEC,
            stale_sec=_LOCK_STALE_SEC,
            metadata=str(os.getpid()),
        )
        if fd is None:
            raise TimeoutError(f"worktree ops lock timeout: {lock_path}")
        try:
            yield
        finally:
            release_exclusive_file_lock(lock_path, fd)


# --------------------------------------------------------------------------- #
# git helpers
# --------------------------------------------------------------------------- #
def _force_rmtree(path: Path) -> None:
    """Best-effort recursive delete that also removes read-only entries.

    On Windows git pack/object files under ``.git`` are read-only, and
    ``shutil.rmtree(ignore_errors=True)`` silently FAILS to delete them, leaving
    the directory behind. The onerror hook restores enough permission to retry
    so genesis-project / worktree teardown actually removes the tree.

    The permission repair is ADDITIVE (``current | wanted``) and grants a
    DIRECTORY its ``+x`` back. The previous ``os.chmod(p, stat.S_IWRITE)``
    replaced the mode outright with ``0o200``: on a directory that is
    write-only-no-execute, so the very next ``os.listdir``/``os.unlink`` inside
    it fails with ``EACCES`` — the hook turned a recoverable failure into a
    permanent one, and a retry (or the owner's own ``rm -rf``) hit the same
    wall. Directories are the common case here, because a worktree is removed
    directory-first.

    The failing entry's PARENT is repaired too: ``unlink``/``rmdir`` are checked
    against the containing directory's write bit, not the child's, so clearing
    the child alone leaves the exact case this hook exists for unfixed.

    The repair is PERMANENT, not scoped to the retry: nothing restores the previous
    mode, so a parent directory that survives this call (the containment check
    stops the delete on a path outside the allowed root, and the retry itself can
    fail) keeps the relaxed permissions — an observed ``0o555`` worktree root came
    back ``0o755``. That is deliberate here, since the tree is being deleted and a
    restore would have to happen on a path that may no longer exist, but it is a
    side effect on the owner's filesystem and is stated rather than implied (I19).
    """
    def _relax(target: Path, *, directory: bool) -> None:
        try:
            mode = stat.S_IMODE(target.lstat().st_mode)
        except OSError:
            return
        want = mode | stat.S_IWRITE | stat.S_IREAD
        if directory:
            want |= stat.S_IEXEC
        if want != mode:
            try:
                os.chmod(target, want)
            except OSError:
                pass

    def _on_error(func, p, _exc):
        try:
            target = Path(p)
            _relax(target, directory=target.is_dir() and not target.is_symlink())
            _relax(target.parent, directory=True)
            func(p)
        except Exception:
            pass

    try:
        shutil.rmtree(path, onerror=_on_error)
    except Exception:
        pass


def _git(
    repo_dir: Path,
    *args: str,
    check: bool = True,
    timeout: Optional[float] = None,
) -> subprocess.CompletedProcess:
    """One git call. ``timeout`` is a pass-through to ``subprocess.run``.

    The default stays ``None`` — UNBOUNDED — on purpose: this helper's own callers
    are background provisioning and the startup orphan sweep, no request waits on
    them, and giving them a ceiling here would be a behaviour change nobody asked
    for. The parameter exists because ``thread_worktrees`` reuses this primitive on
    OWNER-FACING request paths (``GET`` / ``POST`` on a thread's worktree, and the
    project delete), where a wedged git otherwise holds an HTTP request and a
    thread-pool thread forever; that module passes
    ``config.get_thread_git_timeout_sec()`` at every call site.
    """
    return subprocess.run(
        ["git", *args],
        cwd=str(repo_dir),
        capture_output=True,
        text=True,
        check=check,
        timeout=timeout,
    )


def _remove_paths(repo_dir: Path, wt_path: Path, branch: str, *, allowed_root: Optional[Any] = None) -> None:
    """Best-effort teardown: drop the worktree checkout, dir, and branch.

    When ``allowed_root`` is given, refuse to touch any path that is empty or not
    strictly inside it. The registry is durable runtime state; a corrupt/malformed
    entry must never cause deletion of an arbitrary filesystem path.
    """
    wt_path = Path(wt_path)
    wt_text = str(wt_path).strip()
    if allowed_root is not None and (
        not wt_text or wt_text in (".", "/", "//") or not _is_within(wt_path, Path(allowed_root))
    ):
        return
    try:
        _git(repo_dir, "worktree", "remove", "--force", str(wt_path), check=False)
    except Exception:
        pass
    if wt_path.exists():
        _force_rmtree(wt_path)
    try:
        _git(repo_dir, "worktree", "prune", check=False)
    except Exception:
        pass
    if branch:
        try:
            _git(repo_dir, "branch", "-D", branch, check=False)
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class WorktreeHandle:
    task_id: str
    path: str
    branch: str
    base_sha: str
    repo_dir: str
    created_at: float
    parent_task_id: str = ""


def provision_worktree(
    *,
    repo_dir: Any,
    task_id: Any,
    base_sha: str = "",
    parent_task_id: str = "",
    worktree_root: Optional[Any] = None,
    data_dir: Optional[Any] = None,
) -> WorktreeHandle:
    """Create an isolated worktree branched from ``base_sha`` (default HEAD).

    The returned branch is a delta base for the child; the child's patch is a
    diff against ``base_sha`` so the parent can integrate it deliberately.
    """
    repo_dir = Path(repo_dir).resolve()
    root = _resolve_root(worktree_root)
    _assert_root_isolated(root, repo_dir, _data_dir(data_dir))
    safe_task = _safe_name(task_id)
    # Lock the REPO: this rewrites <repo>/.git/worktrees, which the durable
    # thread-worktree registry mutates too.
    with _ops_lock(repo_dir, mkdir_root=root):
        if base_sha:
            _git(repo_dir, "rev-parse", "--verify", f"{base_sha}^{{commit}}")
            base_sha = _git(repo_dir, "rev-parse", base_sha).stdout.strip()
        else:
            base_sha = _git(repo_dir, "rev-parse", "HEAD").stdout.strip()
        wt_path = (root / safe_task).resolve()
        branch = f"{_BRANCH_PREFIX}{safe_task}"
        # Clear any stale checkout/branch left by a crashed run.
        _remove_paths(repo_dir, wt_path, branch, allowed_root=root)
        wt_path.parent.mkdir(parents=True, exist_ok=True)
        _git(repo_dir, "worktree", "add", "--force", "-b", branch, str(wt_path), base_sha)
        handle = WorktreeHandle(
            task_id=str(task_id),
            path=str(wt_path),
            branch=branch,
            base_sha=base_sha,
            repo_dir=str(repo_dir),
            created_at=time.time(),
            parent_task_id=str(parent_task_id or ""),
        )
        entries = [e for e in _load_registry(data_dir) if e.get("path") != str(wt_path)]
        entries.append(asdict(handle))
        _save_registry(entries, data_dir)
        return handle


def provision_genesis_project(
    *,
    repo_dir: Any,
    task_id: Any,
    parent_task_id: str = "",
    projects_root: Optional[Any] = None,
    data_dir: Optional[Any] = None,
    dir_name: str = "",
) -> WorktreeHandle:
    """Provision a durable, isolated, EMPTY git project for a genesis acting child.

    Unlike a worktree this is a standalone repo (not a checkout of the live body)
    under the durable projects root. It is the deliverable itself and is NEVER
    GC-pruned, so it is intentionally not added to the worktree registry. The
    child builds the whole project here and returns a ``workspace.patch`` that is
    a diff from the empty initial commit (``base_sha``).

    ``dir_name`` names the genesis directory meaningfully (e.g. the project name)
    instead of the raw task id, so sibling builders share a recognizable project
    root; the handle's binding identity stays ``task_id`` (I, v6.39).
    """
    repo_dir = Path(repo_dir).resolve()
    root = Path(projects_root) if projects_root else Path(get_subagent_projects_root())
    root = root.expanduser().resolve()
    _assert_root_isolated(root, repo_dir, _data_dir(data_dir))
    safe_task = _safe_name(dir_name or task_id)
    # A genesis project is a STANDALONE repo created under `root`; it touches no
    # existing .git/worktrees. What it contends for is a free NAME under the
    # projects root, so that directory is the right lock.
    with _ops_lock(root):
        proj = (root / safe_task).resolve()
        # Genesis projects are durable: never clobber an existing one -> unique name. Since
        # dir_name can repeat across projects (a shared display name), count up under the
        # ops lock until a free path is found — a single timestamp suffix could still
        # collide on a same-name re-provision within the same second (FileExistsError).
        _suffix = 0
        while proj.exists():
            _suffix += 1
            proj = (root / f"{safe_task}_{_suffix}").resolve()
        proj.mkdir(parents=True, exist_ok=False)
        try:
            _git(proj, "init")
            # A fresh repo may have no commit identity; set a local one for the seed
            # commit only (does not touch the user's global git config).
            _git(
                proj,
                "-c", "user.email=ouroboros@localhost",
                "-c", "user.name=Ouroboros",
                "commit", "--allow-empty", "-m", "genesis: empty project",
            )
            base_sha = _git(proj, "rev-parse", "HEAD").stdout.strip()
        except Exception:
            # Do not leak a partial/uninitialized project dir on git failure.
            _force_rmtree(proj)
            raise
        return WorktreeHandle(
            task_id=str(task_id),
            path=str(proj),
            branch="",
            base_sha=base_sha,
            repo_dir=str(proj),
            created_at=time.time(),
            parent_task_id=str(parent_task_id or ""),
        )


def remove_genesis_project(path: str, *, projects_root: Optional[Any] = None) -> bool:
    """Best-effort removal of a provisioned-but-unused genesis project.

    Only removes a path strictly INSIDE the configured projects root (never an
    arbitrary caller path). Used to clean up a genesis project whose schedule was
    rejected before the child ran; genesis projects are otherwise durable.
    """
    if not str(path or "").strip():
        return False
    root = Path(projects_root) if projects_root else Path(get_subagent_projects_root())
    root = root.expanduser().resolve()
    target = Path(path).resolve()
    if target == root or not _is_within(target, root):
        return False
    if target.exists():
        _force_rmtree(target)
    return True


def remove_worktree(
    *,
    task_id: str = "",
    path: str = "",
    worktree_root: Optional[Any] = None,
    data_dir: Optional[Any] = None,
) -> bool:
    """Tear down a worktree by task_id or path; unregister it. Returns success."""
    want_path = str(Path(path).resolve()) if path else ""
    entries = _load_registry(data_dir)
    match: Optional[Dict[str, Any]] = None
    for entry in entries:
        if task_id and entry.get("task_id") == str(task_id):
            match = entry
            break
        if want_path and entry.get("path") == want_path:
            match = entry
            break
    root = _resolve_root(worktree_root)
    # Lock the matched entry's REPO (its .git/worktrees is what shrinks here);
    # an unregistered path has no repo, so the root lock guards that fallback.
    lock_target = str(match.get("repo_dir") or "") if match is not None else ""
    with _ops_lock(lock_target or root, mkdir_root=root):
        if match is not None:
            _remove_paths(Path(match.get("repo_dir") or "."), Path(match.get("path") or ""), match.get("branch") or "", allowed_root=root)
            survivors = [e for e in _load_registry(data_dir) if e.get("path") != match.get("path")]
            _save_registry(survivors, data_dir)
            return True
        # Unregistered path: best-effort directory removal, but ONLY inside the
        # configured worktree root (never an arbitrary path supplied by a caller).
        if want_path and Path(want_path).exists() and _is_within(Path(want_path), root):
            _force_rmtree(Path(want_path))
            return True
    return False


def prune_orphans(
    *,
    worktree_root: Optional[Any] = None,
    data_dir: Optional[Any] = None,
    retention_days: Optional[int] = None,
) -> Dict[str, Any]:
    """Startup reconciliation: drop worktrees past retention or with a missing
    checkout, then reconcile git's own worktree metadata. Patch artifacts live in
    the task drive, independent of the worktree, so removal never loses results.
    """
    retention = retention_days if retention_days is not None else get_gc_retention_days()
    cutoff = age_cutoff(retention)
    root = _resolve_root(worktree_root)
    removed: List[Dict[str, Any]] = []
    kept: List[Dict[str, Any]] = []
    repos: set[str] = set()
    # Outer ROOT lock: registry consistency across the whole sweep. Each repo's
    # git metadata is then mutated under that REPO's own lock, so a thread
    # worktree owner working on the same .git is never racing this sweep.
    # Order is always root -> repo (see _inproc_lock); nothing takes them the
    # other way round.
    with _ops_lock(root):
        by_repo: Dict[str, List[Dict[str, Any]]] = {}
        for entry in _load_registry(data_dir):
            by_repo.setdefault(str(entry.get("repo_dir") or ""), []).append(entry)
        for repo_dir, group in by_repo.items():
            if repo_dir:
                repos.add(repo_dir)
            # A row with NO repo_dir touches no git metadata, so it needs no
            # repo lock — and falling back to the root here would re-acquire the
            # lockfile this sweep already holds. The in-process RLock is
            # re-entrant and hides that; the cross-process O_EXCL file lock is
            # not, so the sweep would block for the whole timeout and then throw
            # its work away.
            ctx = _ops_lock(repo_dir) if repo_dir else contextlib.nullcontext()
            with ctx:
                for entry in group:
                    wt_path = str(entry.get("path") or "")
                    created = float(entry.get("created_at") or 0)
                    path_exists = Path(wt_path).exists() if wt_path else False
                    if created < cutoff or not path_exists:
                        if repo_dir or wt_path:
                            _remove_paths(
                                Path(repo_dir or "."), Path(wt_path),
                                entry.get("branch") or "", allowed_root=root,
                            )
                        removed.append(entry)
                    else:
                        kept.append(entry)
                if repo_dir:
                    try:
                        _git(Path(repo_dir), "worktree", "prune", check=False)
                    except Exception:
                        pass
        _save_registry(kept, data_dir)
    return {"removed": len(removed), "kept": len(kept)}


def list_worktrees(data_dir: Optional[Any] = None) -> List[Dict[str, Any]]:
    """Return registered worktree records (for UI / inspection)."""
    return _load_registry(data_dir)


# --------------------------------------------------------------------------- #
# Shared primitives (durable thread worktrees reuse ONLY these)
# --------------------------------------------------------------------------- #
# `git worktree add/remove/prune` mutate shared `.git/worktrees` metadata, so
# every worktree owner must serialize on the SAME lock — which is why the
# lockfile is keyed on the REPO (`<repo>/.git/.worktree_ops.lock`), not on each
# registry's own worktree root — and use the SAME containment guards. Exported
# as public names so the durable thread-worktree registry can reuse them
# WITHOUT reaching into privates —
# and without inheriting this module's force-reset provisioning, force removal
# or age-based orphan sweep, none of which may ever touch a thread's worktree.
worktree_ops_lock = _ops_lock
assert_worktree_root_isolated = _assert_root_isolated
path_is_within = _is_within
run_git = _git
safe_path_component = _safe_name
force_rmtree = _force_rmtree
