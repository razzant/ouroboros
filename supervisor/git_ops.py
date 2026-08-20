"""Supervisor git/reset/rescue/dependency operations."""

from __future__ import annotations

import datetime
import json
import logging
import os
import pathlib
import re
import subprocess
import sys
import uuid
from typing import Any, Dict, List, Optional, Tuple

# The state helpers are parent bindings the G1 leaves read through the
# call-time handle _go() and tests monkeypatch on this module; utc_now_iso is
# read as a git_ops attribute by supervisor/update_recovery.py. They must stay
# importable here even when no body below uses them directly.
from supervisor.state import (  # noqa: F401
    append_jsonl, atomic_write_text, load_state, save_state,
)
from ouroboros import config as _config
from ouroboros.utils import utc_now_iso  # noqa: F401

log = logging.getLogger(__name__)


# Pre-``init`` defaults follow the same environment-aware roots as the rest of the
# runtime (``ouroboros.config``), so a process that never calls ``init`` — an
# isolated test or smoke — cannot write supervisor rows into the live data drive.
REPO_DIR: pathlib.Path = pathlib.Path(_config.REPO_DIR)
DRIVE_ROOT: pathlib.Path = pathlib.Path(_config.DATA_DIR)
REMOTE_URL: str = ""
BRANCH_DEV: str = "ouroboros"
BRANCH_STABLE: str = "ouroboros-stable"
MANAGED_REPO_META_NAME = "ouroboros-managed.json"
BOOTSTRAP_PIN_MARKER_NAME = "ouroboros-bootstrap-pending"
UPDATE_INTENT_MARKER_NAME = "ouroboros-update-intent.json"
OFFICIAL_UPDATE_REMOTE_URL = "https://github.com/razzant/ouroboros"


def _guard_live_repo_destructive_git(cmd: List[str]) -> None:
    if os.environ.get("OUROBOROS_ALLOW_LIVE_REPO_TESTS") == "1":
        return
    try:
        live_repo = REPO_DIR.resolve(strict=False) == (
            pathlib.Path.home() / "Ouroboros" / "repo"
        ).resolve(strict=False)
    except OSError:
        live_repo = False
    if not (("PYTEST_CURRENT_TEST" in os.environ or "pytest" in sys.modules) and live_repo):
        return
    normalized = [str(part) for part in cmd]
    is_reset_hard = normalized[:3] == ["git", "reset", "--hard"]
    is_clean = normalized[:2] == ["git", "clean"]
    if is_reset_hard or is_clean:
        raise RuntimeError(
            "Refusing to run destructive git reset/clean on the live Ouroboros repo from pytest. "
            "Use an isolated repo fixture, or OUROBOROS_ALLOW_LIVE_REPO_TESTS=1 for an explicit live-repo test."
        )


def init(repo_dir: pathlib.Path, drive_root: pathlib.Path, remote_url: str,
         branch_dev: str = "ouroboros", branch_stable: str = "ouroboros-stable") -> None:
    global REPO_DIR, DRIVE_ROOT, REMOTE_URL, BRANCH_DEV, BRANCH_STABLE
    REPO_DIR = repo_dir
    DRIVE_ROOT = drive_root
    REMOTE_URL = remote_url
    BRANCH_DEV = branch_dev
    BRANCH_STABLE = branch_stable


def _git_dir() -> pathlib.Path:
    return REPO_DIR / ".git"


def _managed_repo_meta_path() -> pathlib.Path:
    return _git_dir() / MANAGED_REPO_META_NAME


def _bootstrap_pin_marker_path() -> pathlib.Path:
    return _git_dir() / BOOTSTRAP_PIN_MARKER_NAME


def _update_intent_marker_path() -> pathlib.Path:
    return _git_dir() / UPDATE_INTENT_MARKER_NAME


def _read_managed_repo_meta() -> Dict[str, Any]:
    path = _managed_repo_meta_path()
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return raw if isinstance(raw, dict) else {}


def managed_branch_defaults(repo_dir: Optional[pathlib.Path] = None) -> Tuple[str, str]:
    repo = repo_dir or REPO_DIR
    meta_path = repo / ".git" / MANAGED_REPO_META_NAME
    if not meta_path.is_file():
        return BRANCH_DEV, BRANCH_STABLE
    try:
        raw = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return BRANCH_DEV, BRANCH_STABLE
    if not isinstance(raw, dict):
        return BRANCH_DEV, BRANCH_STABLE
    branch_dev = str(raw.get("managed_local_branch") or BRANCH_DEV).strip() or BRANCH_DEV
    branch_stable = str(raw.get("managed_local_stable_branch") or BRANCH_STABLE).strip() or BRANCH_STABLE
    return branch_dev, branch_stable


def _is_launcher_managed_repo() -> bool:
    if str(os.environ.get("OUROBOROS_MANAGED_BY_LAUNCHER", "") or "").strip() == "1":
        return True
    return bool(_read_managed_repo_meta())


def _list_remotes(*, capture=None, warnings: Optional[List[str]] = None) -> List[str]:
    capture_fn = capture or git_capture
    rc, remotes, error = capture_fn(["git", "remote"])
    if rc != 0:
        if warnings is not None:
            warnings.append(
                f"remotes_error:{error or f'git remote exited {rc} without stderr'}"
            )
        return []
    return [line.strip() for line in remotes.splitlines() if line.strip()]


def _has_remote(name: Optional[str] = None) -> bool:
    remotes = _list_remotes()
    if name is None:
        return bool(remotes)
    return name in remotes


def _managed_remote_name(meta: Optional[Dict[str, Any]] = None) -> str:
    info = meta if meta is not None else _read_managed_repo_meta()
    return str(info.get("managed_remote_name") or "managed").strip() or "managed"


def _managed_remote_branch_for(branch: str, meta: Optional[Dict[str, Any]] = None) -> str:
    info = meta if meta is not None else _read_managed_repo_meta()
    if branch == BRANCH_DEV:
        return str(info.get("managed_remote_branch") or branch).strip()
    if branch == BRANCH_STABLE:
        return str(info.get("managed_remote_stable_branch") or branch).strip()
    return branch


def _pin_to_bundle_sha_on_bootstrap(reason: str, managed_meta: Optional[Dict[str, Any]] = None) -> bool:
    if str(reason or "").strip().lower() != "bootstrap":
        return False
    if not _bootstrap_pin_marker_path().exists():
        return False
    info = managed_meta if managed_meta is not None else _read_managed_repo_meta()
    source_sha = str(info.get("source_sha") or "").strip()
    if not source_sha:
        return False
    rc, head_sha, _ = git_capture(["git", "rev-parse", "HEAD"])
    if rc != 0 or str(head_sha or "").strip() != source_sha:
        return False
    return True


def _clear_bootstrap_pin_marker() -> None:
    try:
        _bootstrap_pin_marker_path().unlink()
    except FileNotFoundError:
        return
    except Exception:
        log.warning("Failed to clear bootstrap pin marker", exc_info=True)


def _read_update_intent() -> Dict[str, Any]:
    path = _update_intent_marker_path()
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return raw if isinstance(raw, dict) else {}


def _write_update_intent(payload: Dict[str, Any]) -> None:
    # Atomic: a torn marker would make the next restart silently skip the
    # prepared managed update (reader fails closed on parse errors).
    from ouroboros.utils import atomic_write_json

    path = _update_intent_marker_path()
    atomic_write_json(path, payload, trailing_newline=True)


def _clear_update_intent() -> bool:
    try:
        _update_intent_marker_path().unlink()
    except FileNotFoundError:
        return True
    except Exception:
        log.warning("Failed to clear update intent marker", exc_info=True)
        return False
    return True


def _run_git_process_bounded(
    cmd: List[str], *, timeout: float, cwd: Optional[pathlib.Path] = None,
    env: Optional[Dict[str, str]] = None, text: bool = True,
) -> Tuple[int, Any, Any]:
    """Run one short-lived Git process and terminate its tree on timeout."""
    from ouroboros.platform_layer import kill_process_tree, subprocess_new_group_kwargs
    from ouroboros.tools.shell import _active_subprocesses, _subprocess_lock

    limit = float(timeout)
    empty = "" if text else b""
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd or REPO_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=text,
            env=env,
            **subprocess_new_group_kwargs(),
        )
    except OSError as exc:
        detail = str(exc)
        return 127, empty, detail if text else detail.encode("utf-8", "replace")

    with _subprocess_lock:
        _active_subprocesses.add(proc)
    try:
        try:
            stdout, stderr = proc.communicate(timeout=limit)
        except subprocess.TimeoutExpired as exc:
            partial_error = getattr(exc, "stderr", None)
            kill_process_tree(proc)
            try:
                _tail_out, tail_error = proc.communicate(timeout=10)
            except Exception:
                tail_error = None
            raw_detail = tail_error or partial_error or empty
            if isinstance(raw_detail, bytes):
                detail = raw_detail.decode("utf-8", "replace").strip()
            else:
                detail = str(raw_detail or "").strip()
            message = (
                f"git process timed out after {limit:g}s and was terminated: "
                f"{' '.join(cmd)}"
            )
            if detail:
                message += f" ({detail})"
            return (
                FETCH_TIMEOUT_RC,
                empty,
                message if text else message.encode("utf-8", "replace"),
            )
        return (
            int(proc.returncode if proc.returncode is not None else 1),
            stdout if stdout is not None else empty,
            stderr if stderr is not None else empty,
        )
    finally:
        with _subprocess_lock:
            _active_subprocesses.discard(proc)


def git_capture(cmd: List[str], *, timeout: Optional[float] = None) -> Tuple[int, str, str]:
    # Same reason as utils.run_cmd: this stderr is PARSED (`_maybe_repair_git_index`
    # matches English git diagnostics), so the operator's locale must not decide
    # whether a repairable index error is recognised.
    # ``timeout`` is None for existing non-rescue call sites: a bound here is a
    # behavior change, so it stays opt-in rather than silently retiming them.
    env = {**os.environ, "LC_ALL": "C", "LANG": "C"}
    for _attempt in range(2):
        if timeout is None:
            result = subprocess.run(
                cmd, cwd=str(REPO_DIR), capture_output=True, text=True, env=env,
            )
            returncode, stdout, raw_stderr = (
                result.returncode, result.stdout or "", result.stderr or "",
            )
        else:
            returncode, stdout, raw_stderr = _run_git_process_bounded(
                cmd, timeout=timeout, cwd=REPO_DIR, env=env, text=True,
            )
        stderr = str(raw_stderr or "").strip()
        if returncode == 0:
            return returncode, str(stdout or "").strip(), stderr
        if _maybe_repair_git_index(stderr, timeout=timeout):
            continue
        return returncode, str(stdout or "").strip(), stderr
    return returncode, str(stdout or "").strip(), stderr


from supervisor import update_source as _update_source

# One name per assignment: the module-handle pins (tests/test_module_handle_extraction.py)
# and the G1 leaves resolve these as parent-owned bindings, and a tuple target would
# hide them from the lexical binding scan.
FETCH_TIMEOUT_RC = _update_source.FETCH_TIMEOUT_RC
_git_network_bounded = _update_source._git_network_bounded
_managed_update_target = _update_source._managed_update_target
git_fetch_bounded = _update_source.git_fetch_bounded

def rescue_git_capture(cmd: List[str]) -> Tuple[int, str, str]:
    """Run ``git_capture`` under the configured rescue-only wall-clock bound.

    A timeout returns through the normal nonzero-rc shape, which every caller
    in the rescue graph already treats as a warning rather than a hard stop:
    fail-open, never a stall.
    """
    from ouroboros.update_channels import get_rescue_git_timeout_sec

    return git_capture(cmd, timeout=get_rescue_git_timeout_sec())


def _resolve_managed_update_target(
    remote_name: str, remote_branch: str, branch_ref: str, update_channel: str
) -> Tuple[str, str, str]:
    return _update_source.resolve_managed_update_target(
        remote_name,
        remote_branch,
        branch_ref,
        update_channel=update_channel,
        capture=git_capture,
    )


def _stale_git_lock_paths(max_age_sec: float = 15.0) -> List[pathlib.Path]:
    git_dir = REPO_DIR / ".git"
    if not git_dir.exists():
        return []
    candidates = [git_dir / "index.lock"]
    stale_paths: List[pathlib.Path] = []
    now = datetime.datetime.now(datetime.timezone.utc).timestamp()
    for path in candidates:
        try:
            age = now - path.stat().st_mtime
        except FileNotFoundError:
            continue
        except Exception:
            continue
        if age >= max_age_sec:
            stale_paths.append(path)
    return stale_paths


def _maybe_repair_git_index(stderr: str, *, timeout: Optional[float] = None) -> bool:
    error_text = str(stderr or "")
    error_lower = error_text.lower()
    repaired = False

    if "index.lock" in error_lower:
        for lock_path in _stale_git_lock_paths():
            try:
                lock_path.unlink()
                repaired = True
                log.warning("Removed stale git lock: %s", lock_path)
            except Exception:
                log.warning("Failed to remove stale git lock: %s", lock_path, exc_info=True)

    corrupt_markers = (
        "index file smaller than expected",
        "index file corrupt",
        "fatal: .git/index:",
    )
    if not any(marker in error_lower for marker in corrupt_markers):
        return repaired

    git_dir = REPO_DIR / ".git"
    if not git_dir.exists():
        return repaired

    index_path = git_dir / "index"
    if index_path.exists():
        backup_path = git_dir / f"index.corrupt.{uuid.uuid4().hex[:8]}.bak"
        try:
            index_path.replace(backup_path)
            repaired = True
            log.warning("Backed up corrupt git index to %s", backup_path)
        except Exception:
            log.warning("Failed to back up corrupt git index %s", index_path, exc_info=True)
            return repaired

    rebuild_cmd = ["git", "reset", "--mixed", "HEAD"]
    rebuild_env = {**os.environ, "LC_ALL": "C", "LANG": "C"}
    if timeout is None:
        rebuild = subprocess.run(
            rebuild_cmd,
            cwd=str(REPO_DIR),
            capture_output=True,
            text=True,
            env=rebuild_env,
        )
        rebuild_rc = rebuild.returncode
        rebuild_error = (rebuild.stderr or "").strip() or (rebuild.stdout or "").strip()
    else:
        rebuild_rc, rebuild_stdout, rebuild_stderr = _run_git_process_bounded(
            rebuild_cmd, timeout=timeout, cwd=REPO_DIR, env=rebuild_env, text=True,
        )
        rebuild_error = str(rebuild_stderr or "").strip() or str(rebuild_stdout or "").strip()
    if rebuild_rc == 0:
        log.warning("Rebuilt git index after corruption in %s", REPO_DIR)
        return True

    log.warning(
        "Failed to rebuild git index after corruption: %s",
        rebuild_error,
    )
    return repaired


_REPO_GITIGNORE = """\
# Secrets
.env
.env.*
*.key
*.pem

# IDE
.cursor/
.vscode/
.idea/

# Python bytecode
__pycache__/
*.pyc
*.pyo
*.egg-info/

# Build artifacts
dist/
build/
.pytest_cache/
.mypy_cache/

# Native / binary artifacts (PyInstaller, compiled extensions)
*.so
*.dylib
*.dll
*.dist-info/
base_library.zip

# OS
.DS_Store
Thumbs.db

# Release artifacts
.create_release.py
.release_notes.md
repo.bundle
repo_bundle_manifest.json
python-standalone/
"""


def _ensure_repo_gitignore(repo_dir: pathlib.Path = None) -> None:
    """Write .gitignore if missing before any git add -A."""
    target = repo_dir or REPO_DIR
    gi = target / ".gitignore"
    if not gi.exists():
        gi.write_text(_REPO_GITIGNORE, encoding="utf-8")


def _ensure_git_identity() -> None:
    """Ensure repo-local git identity exists for local commits/tags."""
    git_capture(["git", "config", "user.name", "Ouroboros"])
    git_capture(["git", "config", "user.email", "ouroboros@local.mac"])


def _ensure_local_version_tag() -> None:
    """Create the current VERSION tag locally when a local-only repo has none."""
    version_path = REPO_DIR / "VERSION"
    if not version_path.exists():
        return

    version = version_path.read_text(encoding="utf-8").strip().lstrip("v")
    if not re.match(r"^\d+\.\d+\.\d+(?:-?(?:rc|alpha|beta|a|b)\.?\d+)?$", version, re.IGNORECASE):
        return

    tag_name = f"v{version}"
    rc, tag_match, err = git_capture(["git", "tag", "-l", tag_name])
    if rc != 0:
        log.warning("Failed to check local tag %s: %s", tag_name, err)
        return
    if tag_match.strip():
        return

    rc, all_tags, err = git_capture(["git", "tag", "-l"])
    if rc != 0:
        log.warning("Failed to list local tags: %s", err)
        return
    if any(t.strip() for t in all_tags.splitlines()):
        return

    rc, head_sha, err = git_capture(["git", "rev-parse", "HEAD"])
    if rc != 0 or not head_sha:
        log.warning("Cannot create local version tag %s without HEAD: %s", tag_name, err)
        return

    _ensure_git_identity()
    rc, _, err = git_capture(["git", "tag", "-a", tag_name, "-m", f"Release {tag_name}"])
    if rc != 0:
        log.warning("Failed to create local version tag %s: %s", tag_name, err)
        return

    log.info("Created local-only version tag %s at %s", tag_name, head_sha[:8])


def ensure_repo_present() -> None:
    if not (REPO_DIR / ".git").exists():
        if _is_launcher_managed_repo():
            raise RuntimeError(
                "Launcher-managed repo is missing .git metadata. "
                "The launcher bootstrap must recreate REPO_DIR from the embedded repo bundle."
            )
        # REPO_DIR is live code: initialize in place, never remove it.
        REPO_DIR.mkdir(parents=True, exist_ok=True)
        _ensure_repo_gitignore()
        import dulwich.repo
        dulwich.repo.Repo.init(str(REPO_DIR))

        _ensure_git_identity()

        rc, _, _ = git_capture(["git", "status", "--porcelain"])
        if rc == 0:
            subprocess.run(["git", "add", "-A"], cwd=str(REPO_DIR), check=True)
            subprocess.run(["git", "commit", "-m", "Initial commit from bundle"], cwd=str(REPO_DIR), check=False)

        subprocess.run(["git", "branch", "-M", BRANCH_DEV], cwd=str(REPO_DIR), check=False)
        subprocess.run(["git", "branch", BRANCH_STABLE], cwd=str(REPO_DIR), check=False)

    if not _is_launcher_managed_repo():
        _ensure_local_version_tag()

# The rescue/snapshot machinery lives in supervisor/git_ops_rescue.py (G1
# split); re-exported because callers/tests address it through the git_ops
# facade (cycle-free: the leaf imports git_ops only at call time through its
# _go() handle).
from supervisor.git_ops_rescue import (  # noqa: E402,F401
    _atomic_write_bytes,
    _collect_repo_sync_state,
    _copy_untracked_for_rescue,
    _create_rescue_snapshot,
    _link_rescue_to_evolution_transaction,
    _rescue_untracked_incomplete,
    rescue_before_destructive_rollback,
    rescue_into_tx,
)


# The checkout/reset admission, dependency-sync and safe-restart surface lives
# in supervisor/git_ops_reset.py (G1 split); re-exported because callers/tests
# address it through the git_ops facade (cycle-free: the leaf imports git_ops
# only at call time through its _go() handle).
from supervisor.git_ops_reset import (  # noqa: E402,F401
    _admission_gate_for_unsynced_tree,
    _compute_ref_ahead_count,
    _preserve_branch_for_official_reset,
    _ref_points_at_ref,
    _run_git_resilient,
    checkout_and_reset,
    import_test,
    preserve_local_ref_branch,
    safe_restart,
    sync_runtime_dependencies,
)


# The managed-update status/preparation surface lives in
# supervisor/git_ops_updates.py (G1 split); re-exported because callers/tests
# address it through the git_ops facade (cycle-free: the leaf imports git_ops
# only at call time through its _go() handle).
from supervisor.git_ops_updates import (  # noqa: E402,F401
    compute_managed_update_status,
    ensure_official_update_remote,
    list_commits,
    list_official_update_tags,
    list_versions,
    prepare_managed_update,
)


# Owner recovery surface lives in supervisor/update_recovery.py; re-exported because
# callers/tests address it through the git_ops facade (cycle-free: update_recovery
# imports git_ops only inside functions).
from supervisor.update_recovery import promote_branch_exact, rollback_to_version  # noqa: E402,F401


# The personal persistence remote (`origin`) surface lives in
# supervisor/git_ops_remotes.py (G1 split); re-exported because callers/tests
# address it through the git_ops facade (cycle-free: the leaf imports git_ops
# only at call time through its _go() handle).
from supervisor.git_ops_remotes import (  # noqa: E402,F401
    _configure_credential_helper,
    configure_personal_remote,
    configure_remote,
    push_to_remote,
)
