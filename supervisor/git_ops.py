"""Supervisor git/reset/rescue/dependency operations."""

from __future__ import annotations

import datetime
import json
import logging
import os
import pathlib
import re
import shutil
import subprocess
import sys
import uuid
from typing import Any, Dict, List, Optional, Tuple

from supervisor.state import (
    load_state, save_state, append_jsonl, atomic_write_text,
)
from ouroboros.utils import utc_now_iso

log = logging.getLogger(__name__)


REPO_DIR: pathlib.Path = pathlib.Path.home() / "Ouroboros" / "repo"
DRIVE_ROOT: pathlib.Path = pathlib.Path.home() / "Ouroboros" / "data"
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


def _list_remotes() -> List[str]:
    rc, remotes, _ = git_capture(["git", "remote"])
    if rc != 0:
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


def _clear_update_intent() -> None:
    try:
        _update_intent_marker_path().unlink()
    except FileNotFoundError:
        return
    except Exception:
        log.warning("Failed to clear update intent marker", exc_info=True)

def git_capture(cmd: List[str]) -> Tuple[int, str, str]:
    for _attempt in range(2):
        r = subprocess.run(cmd, cwd=str(REPO_DIR), capture_output=True, text=True)
        stderr = (r.stderr or "").strip()
        if r.returncode == 0:
            return r.returncode, (r.stdout or "").strip(), stderr
        if _maybe_repair_git_index(stderr):
            continue
        return r.returncode, (r.stdout or "").strip(), stderr
    return r.returncode, (r.stdout or "").strip(), (r.stderr or "").strip()


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


def _maybe_repair_git_index(stderr: str) -> bool:
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

    rebuild = subprocess.run(
        ["git", "reset", "--mixed", "HEAD"],
        cwd=str(REPO_DIR),
        capture_output=True,
        text=True,
    )
    if rebuild.returncode == 0:
        log.warning("Rebuilt git index after corruption in %s", REPO_DIR)
        return True

    log.warning(
        "Failed to rebuild git index after corruption: %s",
        (rebuild.stderr or "").strip() or (rebuild.stdout or "").strip(),
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

def _collect_repo_sync_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "current_branch": "unknown",
        "dirty_lines": [],
        "unpushed_lines": [],
        "warnings": [],
    }

    rc, branch, err = git_capture(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    if rc == 0 and branch:
        state["current_branch"] = branch
    elif err:
        state["warnings"].append(f"branch_error:{err}")

    rc, dirty, err = git_capture(["git", "status", "--porcelain"])
    if rc == 0 and dirty:
        state["dirty_lines"] = [ln for ln in dirty.splitlines() if ln.strip()]
    elif rc != 0 and err:
        state["warnings"].append(f"status_error:{err}")

    upstream = ""
    current_branch = str(state.get("current_branch") or "")
    managed_meta = _read_managed_repo_meta()
    if managed_meta and current_branch not in ("", "HEAD", "unknown"):
        managed_remote = _managed_remote_name(managed_meta)
        managed_branch = _managed_remote_branch_for(current_branch, managed_meta)
        if managed_branch and _has_remote(managed_remote):
            upstream = f"{managed_remote}/{managed_branch}"

    if not upstream and _has_remote("origin"):
        rc, up, err = git_capture(["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"])
        if rc == 0 and up:
            upstream = up
        else:
            if current_branch not in ("", "HEAD", "unknown"):
                upstream = f"origin/{current_branch}"
            elif err:
                state["warnings"].append(f"upstream_error:{err}")

    if upstream:
        rc, unpushed, err = git_capture(["git", "log", "--oneline", f"{upstream}..HEAD"])
        if rc == 0 and unpushed:
            state["unpushed_lines"] = [ln for ln in unpushed.splitlines() if ln.strip()]
        elif rc != 0 and err:
            state["warnings"].append(f"unpushed_error:{err}")

    return state


def _copy_untracked_for_rescue(dst_root: pathlib.Path, max_files: int = 200,
                                max_total_bytes: int = 12_000_000) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "copied_files": 0, "skipped_files": 0, "copied_bytes": 0, "truncated": False,
    }
    rc, txt, err = git_capture(["git", "ls-files", "--others", "--exclude-standard"])
    if rc != 0:
        out["error"] = err or "git ls-files failed"
        return out

    lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
    if not lines:
        return out

    dst_root.mkdir(parents=True, exist_ok=True)
    for rel in lines:
        if out["copied_files"] >= max_files:
            out["truncated"] = True
            break
        src = (REPO_DIR / rel).resolve()
        try:
            src.relative_to(REPO_DIR.resolve())
        except Exception:
            out["skipped_files"] += 1
            continue
        if not src.exists() or not src.is_file():
            out["skipped_files"] += 1
            continue
        try:
            size = int(src.stat().st_size)
        except Exception:
            out["skipped_files"] += 1
            continue
        if (out["copied_bytes"] + size) > max_total_bytes:
            out["truncated"] = True
            break
        dst = dst_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(src, dst)
            out["copied_files"] += 1
            out["copied_bytes"] += size
        except Exception:
            out["skipped_files"] += 1
    return out


def _create_rescue_snapshot(branch: str, reason: str,
                             repo_state: Dict[str, Any]) -> Dict[str, Any]:
    now = datetime.datetime.now(datetime.timezone.utc)
    ts = now.strftime("%Y%m%d_%H%M%S")
    rescue_dir = DRIVE_ROOT / "archive" / "rescue" / f"{ts}_{uuid.uuid4().hex[:8]}"
    rescue_dir.mkdir(parents=True, exist_ok=True)

    info: Dict[str, Any] = {
        "ts": now.isoformat(),
        "target_branch": branch,
        "reason": reason,
        "current_branch": repo_state.get("current_branch"),
        "dirty_count": len(repo_state.get("dirty_lines") or []),
        "unpushed_count": len(repo_state.get("unpushed_lines") or []),
        "warnings": list(repo_state.get("warnings") or []),
        "path": str(rescue_dir),
    }

    rc_status, status_txt, _ = git_capture(["git", "status", "--porcelain"])
    if rc_status == 0:
        atomic_write_text(rescue_dir / "status.porcelain.txt",
                          status_txt + ("\n" if status_txt else ""))

    rc_diff, diff_txt, diff_err = git_capture(["git", "diff", "--binary", "HEAD"])
    if rc_diff == 0:
        atomic_write_text(rescue_dir / "changes.diff",
                          diff_txt + ("\n" if diff_txt else ""))
    else:
        info["diff_error"] = diff_err or "git diff failed"

    # Also capture tracked changes as a real, recoverable git object so recovery
    # is `git stash apply <sha>` / `git checkout <ref> -- .` rather than only a
    # loose diff file. `git stash create` snapshots staged+unstaged tracked
    # changes (it omits untracked files, which the copy below preserves). Purely
    # additive: failure here never blocks the reset and the diff/untracked copy
    # remain the primary recovery artifacts.
    rc_stash, stash_sha, _ = git_capture(["git", "stash", "create", f"rescue:{reason}"])
    stash_sha = stash_sha.strip()
    if rc_stash == 0 and stash_sha:
        ref_name = f"refs/rescue/{rescue_dir.name}"
        rc_ref, _, ref_err = git_capture(["git", "update-ref", ref_name, stash_sha])
        if rc_ref == 0:
            info["rescue_ref"] = ref_name
            info["rescue_commit"] = stash_sha
        else:
            info["rescue_ref_error"] = ref_err or "git update-ref failed"

    untracked_meta = _copy_untracked_for_rescue(rescue_dir / "untracked")
    info["untracked"] = untracked_meta

    unpushed_lines = [ln for ln in (repo_state.get("unpushed_lines") or []) if str(ln).strip()]
    if unpushed_lines:
        atomic_write_text(rescue_dir / "unpushed_commits.txt",
                          "\n".join(unpushed_lines) + "\n")

    atomic_write_text(rescue_dir / "rescue_meta.json",
                      json.dumps(info, ensure_ascii=False, indent=2))
    _link_rescue_to_evolution_transaction(info, reason)
    return info


def _link_rescue_to_evolution_transaction(rescue_info: Dict[str, Any], reason: str) -> None:
    """Attach rescue recovery pointers to the active evolution transaction."""
    try:
        path = pathlib.Path(DRIVE_ROOT) / "state" / "evolution_campaign.json"
        if not path.is_file():
            return
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or raw.get("status") not in {"active", "paused"}:
            return
        tx = raw.get("active_transaction")
        if not isinstance(tx, dict):
            return
        tx["rescue_ref"] = str(rescue_info.get("rescue_ref") or "")
        tx["dirty_snapshot_ref"] = str(rescue_info.get("rescue_ref") or "")
        tx["rescue_path"] = str(rescue_info.get("path") or "")
        tx["restart_decision"] = "rescue_snapshot_created"
        tx["recovery_hint"] = (
            f"Recover with {tx['rescue_ref']} or inspect {tx['rescue_path']}"
            if tx.get("rescue_ref") or tx.get("rescue_path")
            else "Rescue attempted; inspect supervisor logs."
        )
        tx["updated_at"] = utc_now_iso()
        raw["active_transaction"] = tx
        raw["updated_at"] = utc_now_iso()
        atomic_write_text(path, json.dumps(raw, ensure_ascii=False, indent=2) + "\n")
        append_jsonl(
            pathlib.Path(DRIVE_ROOT) / "logs" / "supervisor.jsonl",
            {
                "ts": utc_now_iso(),
                "type": "evolution_transaction_rescue_linked",
                "reason": reason,
                "transaction_id": tx.get("transaction_id"),
                "task_id": tx.get("task_id"),
                "rescue_ref": tx.get("rescue_ref"),
                "rescue_path": tx.get("rescue_path"),
            },
        )
    except Exception:
        log.debug("Failed to link rescue snapshot to evolution transaction", exc_info=True)


def _rescue_untracked_incomplete(rescue_info: Dict[str, Any]) -> str:
    """Return a human-readable reason when untracked rescue capture is incomplete."""
    meta = rescue_info.get("untracked")
    if not isinstance(meta, dict):
        return ""
    if meta.get("error"):
        return str(meta.get("error"))
    if meta.get("truncated"):
        return "untracked rescue copy was truncated"
    if int(meta.get("skipped_files") or 0) > 0:
        return f"{int(meta.get('skipped_files') or 0)} untracked file(s) were skipped"
    return ""


def _compute_ref_ahead_count(ref: str, target_ref: str) -> Tuple[bool, int, str]:
    """Return whether *ref* is ahead of *target_ref*, failing closed on errors."""
    if not ref or not target_ref:
        return False, 0, "missing ref for ahead comparison"
    rc, counts, err = git_capture([
        "git", "rev-list", "--left-right", "--count", f"{ref}...{target_ref}",
    ])
    if rc != 0:
        return False, 0, err or f"git rev-list failed for {ref}...{target_ref}"
    try:
        ahead, _behind = (int(part) for part in counts.split())
    except Exception:
        return False, 0, f"could not parse ahead/behind counts: {counts!r}"
    return True, ahead, ""


def _ref_points_at_ref(left_ref: str, right_ref: str) -> bool:
    left_ref = str(left_ref or "").strip()
    right_ref = str(right_ref or "").strip()
    if not left_ref or not right_ref:
        return False
    rc_left, left_sha, _ = git_capture(["git", "rev-parse", "--verify", left_ref])
    if rc_left != 0 or not left_sha:
        return False
    rc_right, right_sha, _ = git_capture(["git", "rev-parse", "--verify", right_ref])
    return rc_right == 0 and bool(right_sha) and left_sha.strip() == right_sha.strip()


def preserve_local_ref_branch(ref: str = "HEAD", prefix: str = "local-keep") -> Tuple[bool, str]:
    """Create a local branch pointing at *ref* before replacing it."""
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    branch_name = f"{prefix}-{now}-{uuid.uuid4().hex[:6]}"
    rc, _out, err = git_capture(["git", "branch", branch_name, ref])
    if rc != 0:
        return False, err or f"failed to create {branch_name}"
    return True, branch_name


def _preserve_branch_for_official_reset(
    branch: str,
    target_ref: str,
    update_intent: Dict[str, Any],
) -> Tuple[bool, str]:
    """Ensure local commits survive an explicit official update reset."""
    count_ok, ahead, count_error = _compute_ref_ahead_count(branch, target_ref)
    if not count_ok:
        return False, f"Could not compare {branch} with update target {target_ref}: {count_error}"
    if ahead <= 0:
        return True, ""
    existing = str(update_intent.get("keep_branch") or "").strip()
    if existing and _ref_points_at_ref(existing, branch):
        return True, existing
    ok, branch_or_error = preserve_local_ref_branch(branch)
    if not ok:
        return False, branch_or_error
    return True, branch_or_error

def checkout_and_reset(branch: str, reason: str = "unspecified",
                       unsynced_policy: str = "ignore") -> Tuple[bool, str]:
    managed_meta = _read_managed_repo_meta()
    fetch_remote = ""
    target_ref = ""
    pin_bundle_sha = _pin_to_bundle_sha_on_bootstrap(reason, managed_meta)
    update_intent = _read_update_intent()
    update_intent_target = ""
    intent_keep_branch = ""
    if managed_meta and not pin_bundle_sha and update_intent:
        intent_branch = str(update_intent.get("branch") or BRANCH_DEV)
        intent_sha = str(update_intent.get("target_sha") or "").strip()
        if intent_branch == branch and intent_sha:
            rc_intent, _sha_out, _sha_err = git_capture(["git", "rev-parse", "--verify", intent_sha])
            if rc_intent == 0:
                update_intent_target = intent_sha
                target_ref = intent_sha
                intent_keep_branch = str(update_intent.get("keep_branch") or "").strip()
            else:
                log.warning("Ignoring update intent with missing target sha %s", intent_sha)
    if not managed_meta and not pin_bundle_sha and _has_remote("origin"):
        fetch_remote = "origin"

    if fetch_remote:
        rc, _, err = git_capture(["git", "fetch", fetch_remote])
        if rc != 0:
            msg = f"git fetch {fetch_remote} failed: {err or 'unknown error'}"
            append_jsonl(
                DRIVE_ROOT / "logs" / "supervisor.jsonl",
                {
                    "ts": utc_now_iso(),
                    "type": "reset_fetch_failed",
                    "target_branch": branch, "reason": reason, "error": msg,
                    "remote": fetch_remote,
                    "continuing_local_reset": True,
                },
            )
            log.warning("%s; continuing with local reset for branch %s", msg, branch)

    policy = str(unsynced_policy or "ignore").strip().lower()
    if policy not in {"ignore", "block", "rescue_and_block", "rescue_and_reset"}:
        policy = "ignore"

    if policy != "ignore":
        repo_state = _collect_repo_sync_state()
        dirty_lines = list(repo_state.get("dirty_lines") or [])
        unpushed_lines = list(repo_state.get("unpushed_lines") or [])
        unpushed_needs_rescue = bool(update_intent_target and unpushed_lines)
        if dirty_lines or unpushed_needs_rescue:
            bits: List[str] = []
            if unpushed_lines and (dirty_lines or unpushed_needs_rescue):
                bits.append(f"unpushed={len(unpushed_lines)}")
            if dirty_lines:
                bits.append(f"dirty={len(dirty_lines)}")
            detail = ", ".join(bits) if bits else "unsynced"
            rescue_info: Dict[str, Any] = {}
            if policy in {"rescue_and_block", "rescue_and_reset"}:
                try:
                    rescue_info = _create_rescue_snapshot(
                        branch=branch, reason=reason, repo_state=repo_state)
                except Exception as e:
                    rescue_info = {"error": repr(e)}
                if policy == "rescue_and_reset" and rescue_info.get("error"):
                    msg = (
                        f"Reset blocked ({detail}) because rescue snapshot failed: "
                        f"{rescue_info.get('error')}. Local changes were left untouched."
                    )
                    append_jsonl(
                        DRIVE_ROOT / "logs" / "supervisor.jsonl",
                        {
                            "ts": utc_now_iso(),
                            "type": "reset_blocked_rescue_failed",
                            "target_branch": branch, "reason": reason, "policy": policy,
                            "current_branch": repo_state.get("current_branch"),
                            "dirty_count": len(dirty_lines),
                            "unpushed_count": len(unpushed_lines),
                            "dirty_preview": dirty_lines[:20],
                            "unpushed_preview": unpushed_lines[:20],
                            "warnings": list(repo_state.get("warnings") or []),
                            "rescue": rescue_info,
                            "incomplete_reason": "snapshot_error",
                        },
                    )
                    return False, msg
                if policy == "rescue_and_reset" and rescue_info.get("diff_error"):
                    msg = (
                        f"Reset blocked ({detail}) because rescue diff capture failed: "
                        f"{rescue_info.get('diff_error')}. Local changes were left untouched."
                    )
                    append_jsonl(
                        DRIVE_ROOT / "logs" / "supervisor.jsonl",
                        {
                            "ts": utc_now_iso(),
                            "type": "reset_blocked_rescue_incomplete",
                            "target_branch": branch, "reason": reason, "policy": policy,
                            "current_branch": repo_state.get("current_branch"),
                            "dirty_count": len(dirty_lines),
                            "unpushed_count": len(unpushed_lines),
                            "dirty_preview": dirty_lines[:20],
                            "unpushed_preview": unpushed_lines[:20],
                            "warnings": list(repo_state.get("warnings") or []),
                            "rescue": rescue_info,
                            "incomplete_reason": "diff_error",
                        },
                    )
                    return False, msg
                untracked_rescue_error = _rescue_untracked_incomplete(rescue_info)
                if policy == "rescue_and_reset" and untracked_rescue_error:
                    msg = (
                        f"Reset blocked ({detail}) because untracked-file rescue was incomplete: "
                        f"{untracked_rescue_error}. Local changes were left untouched."
                    )
                    append_jsonl(
                        DRIVE_ROOT / "logs" / "supervisor.jsonl",
                        {
                            "ts": utc_now_iso(),
                            "type": "reset_blocked_rescue_incomplete",
                            "target_branch": branch, "reason": reason, "policy": policy,
                            "current_branch": repo_state.get("current_branch"),
                            "dirty_count": len(dirty_lines),
                            "unpushed_count": len(unpushed_lines),
                            "dirty_preview": dirty_lines[:20],
                            "unpushed_preview": unpushed_lines[:20],
                            "warnings": list(repo_state.get("warnings") or []),
                            "rescue": rescue_info,
                            "incomplete_reason": "untracked_rescue",
                            "incomplete_detail": untracked_rescue_error,
                        },
                    )
                    return False, msg
            rescue_suffix = ""
            rescue_path = str(rescue_info.get("path") or "").strip()
            if rescue_path:
                rescue_suffix = f" Rescue saved to {rescue_path}."
            elif policy in {"rescue_and_block", "rescue_and_reset"} and rescue_info.get("error"):
                rescue_suffix = f" Rescue failed: {rescue_info.get('error')}."

            if policy in {"block", "rescue_and_block"}:
                msg = f"Reset blocked ({detail}) to protect local changes.{rescue_suffix}"
                append_jsonl(
                    DRIVE_ROOT / "logs" / "supervisor.jsonl",
                    {
                        "ts": utc_now_iso(),
                        "type": "reset_blocked_unsynced_state",
                        "target_branch": branch, "reason": reason, "policy": policy,
                        "current_branch": repo_state.get("current_branch"),
                        "dirty_count": len(dirty_lines),
                        "unpushed_count": len(unpushed_lines),
                        "dirty_preview": dirty_lines[:20],
                        "unpushed_preview": unpushed_lines[:20],
                        "warnings": list(repo_state.get("warnings") or []),
                        "rescue": rescue_info,
                    },
                )
                return False, msg

            append_jsonl(
                DRIVE_ROOT / "logs" / "supervisor.jsonl",
                {
                    "ts": utc_now_iso(),
                    "type": "reset_unsynced_rescued_then_reset",
                    "target_branch": branch, "reason": reason, "policy": policy,
                    "current_branch": repo_state.get("current_branch"),
                    "dirty_count": len(dirty_lines),
                    "unpushed_count": len(unpushed_lines),
                    "dirty_preview": dirty_lines[:20],
                    "unpushed_preview": unpushed_lines[:20],
                    "warnings": list(repo_state.get("warnings") or []),
                    "rescue": rescue_info,
                },
            )

    def _run_git_resilient(cmd, **kwargs):
        import time
        check = bool(kwargs.pop("check", False))
        _guard_live_repo_destructive_git(list(cmd))
        for attempt in range(5):
            run_kwargs = dict(kwargs)
            run_kwargs.setdefault("capture_output", True)
            run_kwargs.setdefault("text", True)
            result = subprocess.run(cmd, **run_kwargs)
            if result.returncode == 0:
                return result
            if _maybe_repair_git_index(result.stderr):
                time.sleep(0.2)
                continue
            if not check:
                return result
            if attempt == 4:
                raise subprocess.CalledProcessError(
                    result.returncode,
                    cmd,
                    output=result.stdout,
                    stderr=result.stderr,
                )
            time.sleep(1)
        return subprocess.run(cmd, check=check, **kwargs)

    remote_ref_exists = False
    if target_ref:
        remote_ref_exists = subprocess.run(
            ["git", "rev-parse", "--verify", target_ref],
            cwd=str(REPO_DIR),
            capture_output=True,
        ).returncode == 0

    if remote_ref_exists:
        if update_intent_target:
            preserve_ok, preserve_msg = _preserve_branch_for_official_reset(
                branch, target_ref, update_intent,
            )
            if not preserve_ok:
                return False, f"Could not preserve local branch before official update: {preserve_msg}"
            if preserve_msg and preserve_msg != intent_keep_branch:
                append_jsonl(
                    DRIVE_ROOT / "logs" / "supervisor.jsonl",
                    {
                        "ts": utc_now_iso(),
                        "type": "ui_update_preserved_late_head",
                        "target_branch": branch,
                        "reason": reason,
                        "target_ref": target_ref,
                        "keep_branch": preserve_msg,
                    },
                )
            _run_git_resilient(["git", "reset", "--hard", "HEAD"], cwd=str(REPO_DIR), check=True)
            _run_git_resilient(["git", "clean", "-fd"], cwd=str(REPO_DIR), check=True)
        _run_git_resilient(["git", "checkout", "-B", branch, target_ref], cwd=str(REPO_DIR), check=True)
        if update_intent_target:
            _run_git_resilient(["git", "reset", "--hard", target_ref], cwd=str(REPO_DIR), check=True)
        _run_git_resilient(["git", "clean", "-fd"], cwd=str(REPO_DIR), check=True)
    else:
        rc_local = subprocess.run(
            ["git", "rev-parse", "--verify", branch],
            cwd=str(REPO_DIR), capture_output=True,
        ).returncode

        if rc_local != 0:
            _run_git_resilient(["git", "reset", "--hard", "HEAD"], cwd=str(REPO_DIR), check=True)
            _run_git_resilient(["git", "clean", "-fd"], cwd=str(REPO_DIR), check=True)
            # §6 (same detached-HEAD class as BUG1): `-b` with check=False silently swallowed a
            # "branch already exists" error and proceeded with HEAD possibly detached/wrong;
            # `-B` force-creates the branch at HEAD and check=True raises a real failure.
            _run_git_resilient(["git", "checkout", "-B", branch], cwd=str(REPO_DIR), check=True)
        else:
            if policy == "rescue_and_reset":
                _run_git_resilient(["git", "reset", "--hard", "HEAD"], cwd=str(REPO_DIR), check=True)
                _run_git_resilient(["git", "clean", "-fd"], cwd=str(REPO_DIR), check=True)
            _run_git_resilient(["git", "checkout", branch], cwd=str(REPO_DIR), check=True)
            _run_git_resilient(["git", "reset", "--hard", "HEAD"], cwd=str(REPO_DIR), check=True)
            if policy == "rescue_and_reset":
                _run_git_resilient(["git", "clean", "-fd"], cwd=str(REPO_DIR), check=True)

    # Checkout may not update mtimes; remove stale bytecode.
    for p in REPO_DIR.rglob("__pycache__"):
        shutil.rmtree(p, ignore_errors=True)
    st = load_state()
    st["current_branch"] = branch
    st["current_sha"] = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=str(REPO_DIR),
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    save_state(st)
    if update_intent_target and st["current_sha"] != update_intent_target:
        return False, f"Update intent checkout landed on {st['current_sha']} but expected {update_intent_target}"
    if pin_bundle_sha:
        _clear_bootstrap_pin_marker()
    if update_intent_target and str(reason or "") != "ui_update_apply":
        _clear_update_intent()
    return True, "ok"

def sync_runtime_dependencies(reason: str) -> Tuple[bool, str]:
    if getattr(sys, 'frozen', False):
        log.info("Skipping pip install in frozen (PyInstaller) mode — deps are bundled.")
        return True, "frozen:bundled"

    req_path = REPO_DIR / "requirements.txt"
    cmd: List[str] = [sys.executable, "-m", "pip", "install", "-q"]
    source = ""
    if req_path.exists():
        cmd += ["-r", str(req_path)]
        source = f"requirements:{req_path}"
    else:
        cmd += ["openai>=1.0.0", "requests"]
        source = "fallback:minimal"
    try:
        subprocess.run(cmd, cwd=str(REPO_DIR), check=True, timeout=120)
        append_jsonl(
            DRIVE_ROOT / "logs" / "supervisor.jsonl",
            {
                "ts": utc_now_iso(),
                "type": "deps_sync_ok", "reason": reason, "source": source,
            },
        )
        return True, source
    except Exception as e:
        msg = repr(e)
        append_jsonl(
            DRIVE_ROOT / "logs" / "supervisor.jsonl",
            {
                "ts": utc_now_iso(),
                "type": "deps_sync_error", "reason": reason, "source": source, "error": msg,
            },
        )
        return False, msg


def import_test() -> Dict[str, Any]:
    if getattr(sys, 'frozen', False):
        log.info("Skipping import_test in frozen (PyInstaller) mode — modules are bundled.")
        return {"ok": True, "skipped": "frozen"}

    r = subprocess.run(
        [sys.executable, "-c", "import ouroboros, ouroboros.agent; print('import_ok')"],
        cwd=str(REPO_DIR),
        capture_output=True, text=True,
    )
    return {"ok": (r.returncode == 0), "stdout": r.stdout, "stderr": r.stderr,
            "returncode": r.returncode}

def safe_restart(
    reason: str,
    unsynced_policy: str = "rescue_and_reset",
) -> Tuple[bool, str]:
    """Checkout dev, sync deps, import-test, then fall back to stable if needed."""
    ok, err = checkout_and_reset(BRANCH_DEV, reason=reason, unsynced_policy=unsynced_policy)
    if not ok:
        return False, f"Failed checkout {BRANCH_DEV}: {err}"

    deps_ok, deps_msg = sync_runtime_dependencies(reason=reason)
    if not deps_ok:
        return False, f"Failed deps for {BRANCH_DEV}: {deps_msg}"

    t = import_test()
    if t["ok"]:
        return True, f"OK: {BRANCH_DEV}"

    append_jsonl(
        DRIVE_ROOT / "logs" / "supervisor.jsonl",
        {
            "ts": utc_now_iso(),
            "type": "safe_restart_dev_import_failed",
            "reason": reason,
            "branch": BRANCH_DEV,
            "stdout": t.get("stdout", ""),
            "stderr": t.get("stderr", ""),
            "returncode": t.get("returncode", -1),
        },
    )

    ok_s, err_s = checkout_and_reset(
        BRANCH_STABLE,
        reason=f"{reason}_fallback_stable",
        unsynced_policy="rescue_and_reset",
    )
    if not ok_s:
        return False, f"Failed checkout {BRANCH_STABLE}: {err_s}"

    deps_ok_s, deps_msg_s = sync_runtime_dependencies(reason=f"{reason}_fallback_stable")
    if not deps_ok_s:
        return False, f"Failed deps for {BRANCH_STABLE}: {deps_msg_s}"

    t2 = import_test()
    if t2["ok"]:
        return True, f"OK: fell back to {BRANCH_STABLE}"

    return False, "Both branches failed import (dev and stable)"


def list_versions(max_count: int = 50) -> List[Dict[str, Any]]:
    """Return list of annotated git tags sorted newest-first."""
    rc, raw, _ = git_capture([
        "git", "tag", "-l", "--sort=-creatordate",
        "--format=%(refname:short)\t%(creatordate:iso-strict)\t%(subject)",
    ])
    if rc != 0 or not raw.strip():
        return []
    versions: List[Dict[str, Any]] = []
    for line in raw.splitlines()[:max_count]:
        parts = line.split("\t", 2)
        if len(parts) >= 1:
            versions.append({
                "tag": parts[0],
                "date": parts[1] if len(parts) > 1 else "",
                "message": parts[2] if len(parts) > 2 else "",
            })
    return versions


def list_commits(max_count: int = 30) -> List[Dict[str, Any]]:
    """Return recent commits on current branch."""
    rc, raw, _ = git_capture([
        "git", "log", f"--max-count={max_count}",
        "--format=%H\t%h\t%ai\t%s",
    ])
    if rc != 0 or not raw.strip():
        return []
    commits: List[Dict[str, Any]] = []
    for line in raw.splitlines():
        parts = line.split("\t", 3)
        if len(parts) >= 4:
            commits.append({
                "sha": parts[0], "short_sha": parts[1],
                "date": parts[2], "message": parts[3],
            })
    return commits


def _managed_update_target(branch: Optional[str] = None) -> Tuple[str, str, str]:
    """Return (remote_name, remote_branch, target_ref) for launcher-managed updates."""
    target_branch = branch or BRANCH_DEV
    managed_meta = _read_managed_repo_meta()
    if not managed_meta:
        return "", "", ""
    remote_name = _managed_remote_name(managed_meta)
    remote_branch = _managed_remote_branch_for(target_branch, managed_meta)
    target_ref = f"{remote_name}/{remote_branch}" if remote_name and remote_branch else ""
    return remote_name, remote_branch, target_ref


def ensure_official_update_remote() -> Tuple[bool, str]:
    """Ensure the managed update remote points at the official Ouroboros repository."""
    # Honor the manifest-selected managed remote name (default "managed") so the
    # repaired/added remote matches the one _managed_update_target fetches from.
    remote_name = _managed_remote_name()
    remotes = _list_remotes()
    if remote_name in remotes:
        rc, _out, err = git_capture(["git", "remote", "set-url", remote_name, OFFICIAL_UPDATE_REMOTE_URL])
    else:
        rc, _out, err = git_capture(["git", "remote", "add", remote_name, OFFICIAL_UPDATE_REMOTE_URL])
    return rc == 0, err


def list_official_update_tags(max_count: int = 30) -> List[Dict[str, Any]]:
    """Return official tags from the official managed remote, separate from local/user tags."""
    remote_name = _managed_remote_name()
    if not _has_remote(remote_name):
        return []
    rc, raw, _err = git_capture([
        "git", "ls-remote", "--tags", "--refs", "--sort=-version:refname",
        remote_name, "refs/tags/v*",
    ])
    if rc != 0:
        return []
    tags: List[Dict[str, Any]] = []
    for line in raw.splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        tags.append({
            "tag": parts[1].rsplit("/", 1)[-1],
            "sha": parts[0],
            "source": "official",
        })
        if len(tags) >= max_count:
            break
    return tags


def compute_managed_update_status(fetch: bool = False) -> Dict[str, Any]:
    """Return current managed-remote divergence for the UI Update panel."""
    branch_dev, _branch_stable = managed_branch_defaults()
    remote_name, remote_branch, target_ref = _managed_update_target(branch_dev)
    official_remote_ok = True
    official_remote_err = ""
    if fetch and remote_name:
        official_remote_ok, official_remote_err = ensure_official_update_remote()
    state: Dict[str, Any] = {
        "managed": bool(_read_managed_repo_meta()),
        "remote": remote_name,
        "remote_branch": remote_branch,
        "target_ref": target_ref,
        "current_branch": "unknown",
        "current_sha": "",
        "current_short_sha": "",
        "latest_sha": "",
        "latest_short_sha": "",
        "latest_message": "",
        "ahead": 0,
        "behind": 0,
        "dirty": False,
        "dirty_count": 0,
        "dirty_preview": [],
        "warnings": [],
        "available": False,
        "safe_to_apply": False,
    }
    if not official_remote_ok:
        state["warnings"].append(f"remote_config_error:{official_remote_err or 'unknown error'}")
        state["managed"] = False
        state["available"] = False
        state["safe_to_apply"] = False
        return state

    rc, branch, err = git_capture(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    if rc == 0:
        state["current_branch"] = branch
    elif err:
        state["warnings"].append(f"branch_error:{err}")

    rc, sha, err = git_capture(["git", "rev-parse", "HEAD"])
    if rc == 0:
        state["current_sha"] = sha
        state["current_short_sha"] = sha[:8]
    elif err:
        state["warnings"].append(f"head_error:{err}")

    rc, dirty, err = git_capture(["git", "status", "--porcelain"])
    if rc == 0:
        dirty_lines = [line for line in dirty.splitlines() if line.strip()]
        state["dirty"] = bool(dirty_lines)
        state["dirty_count"] = len(dirty_lines)
        state["dirty_preview"] = dirty_lines[:20]
    elif err:
        state["warnings"].append(f"status_error:{err}")

    if fetch and remote_name:
        rc, _out, err = git_capture(["git", "fetch", remote_name])
        if rc != 0:
            state["warnings"].append(f"fetch_error:{err or 'unknown error'}")

    if not target_ref:
        state["warnings"].append("managed_updates_unavailable")
        return state
    if not fetch:
        state["warnings"].append("official_status_requires_check")
        return state
    if not _has_remote(remote_name):
        state["warnings"].append(f"missing_remote:{remote_name}")
        return state

    rc, latest_sha, err = git_capture(["git", "rev-parse", "--verify", target_ref])
    if rc != 0:
        state["warnings"].append(f"target_ref_error:{err or target_ref}")
        return state
    state["latest_sha"] = latest_sha
    state["latest_short_sha"] = latest_sha[:8]

    rc, latest_msg, _err = git_capture(["git", "log", "-1", "--format=%s", target_ref])
    if rc == 0:
        state["latest_message"] = latest_msg

    rc, counts, err = git_capture(["git", "rev-list", "--left-right", "--count", f"HEAD...{target_ref}"])
    if rc == 0:
        try:
            ahead, behind = (int(part) for part in counts.split())
        except Exception:
            ahead, behind = 0, 0
            state["warnings"].append(f"divergence_parse_error:{counts}")
        state["ahead"] = ahead
        state["behind"] = behind
        state["available"] = behind > 0
        state["safe_to_apply"] = behind > 0 and ahead == 0 and not state["dirty"]
    elif err:
        state["warnings"].append(f"divergence_error:{err}")
    return state


def prepare_managed_update(strategy: str = "replace") -> Tuple[bool, Dict[str, Any]]:
    """Prepare a user-requested managed update before the process restarts."""
    status = compute_managed_update_status(fetch=True)
    if not status.get("managed"):
        return False, {"error": "Managed updates are unavailable for this checkout.", "status": status}
    if not status.get("available"):
        return False, {"error": "No managed update is available.", "status": status}

    strategy = str(strategy or "replace").strip().lower()
    if strategy not in {"replace", "stash", "force"}:
        strategy = "replace"

    repo_state = _collect_repo_sync_state()
    rescue_info: Dict[str, Any] = {}
    try:
        rescue_info = _create_rescue_snapshot(
            branch=str(repo_state.get("current_branch") or BRANCH_DEV),
            reason=f"ui_update_{strategy}",
            repo_state=repo_state,
        )
    except Exception as exc:
        return False, {"error": f"Rescue snapshot failed: {exc!r}", "status": status}
    if rescue_info.get("diff_error"):
        return False, {"error": f"Rescue diff capture failed: {rescue_info.get('diff_error')}", "status": status}
    incomplete = _rescue_untracked_incomplete(rescue_info)
    if incomplete:
        return False, {"error": f"Untracked-file rescue incomplete: {incomplete}", "status": status}

    target_sha = str(status.get("latest_sha") or "").strip()
    if not target_sha:
        return False, {"error": "Managed update target SHA is missing.", "status": status}
    keep_branch = ""
    count_ok, ahead, count_error = _compute_ref_ahead_count(BRANCH_DEV, target_sha)
    if not count_ok:
        return False, {
            "error": f"Could not compare local branch with managed update target: {count_error}",
            "status": status,
        }
    if ahead > 0:
        ok, keep_branch_or_error = preserve_local_ref_branch(BRANCH_DEV)
        if not ok:
            return False, {"error": f"Could not preserve local branch: {keep_branch_or_error}", "status": status}
        keep_branch = keep_branch_or_error
    _write_update_intent({
        "schema_version": 1,
        "branch": BRANCH_DEV,
        "target_sha": target_sha,
        "target_ref": status.get("target_ref") or "",
        "strategy": strategy,
        "keep_branch": keep_branch,
        "requested_at": utc_now_iso(),
    })

    append_jsonl(
        DRIVE_ROOT / "logs" / "supervisor.jsonl",
        {
            "ts": utc_now_iso(),
            "type": "ui_update_requested",
            "strategy": strategy,
            "status": status,
            "rescue": rescue_info,
            "keep_branch": keep_branch,
        },
    )
    return True, {"status": status, "rescue": rescue_info, "keep_branch": keep_branch}


def rollback_to_version(tag_or_sha: str, reason: str = "manual_rollback") -> Tuple[bool, str]:
    """Rollback to a specific tag or commit SHA with rescue snapshot."""
    repo_state = _collect_repo_sync_state()
    try:
        _create_rescue_snapshot(
            branch=repo_state.get("current_branch", "unknown"),
            reason=reason,
            repo_state=repo_state,
        )
    except Exception as e:
        log.warning("Rescue snapshot failed before rollback: %s", e)

    # Resolve the target to a SHA first
    rc_rev, target_sha, err_rev = git_capture(["git", "rev-parse", tag_or_sha])
    if rc_rev != 0:
        return False, f"Cannot resolve {tag_or_sha}: {err_rev}"

    # Reset current branch to the target (avoids detached HEAD)
    _guard_live_repo_destructive_git(["git", "reset", "--hard", target_sha])
    rc, _, err = git_capture(["git", "reset", "--hard", target_sha])
    if rc != 0:
        return False, f"git reset failed: {err}"

    st = load_state()
    st["current_sha"] = target_sha.strip()
    save_state(st)

    warning = ""
    branch = repo_state.get("current_branch") or BRANCH_DEV
    if _has_remote("origin") and branch and branch not in {"HEAD", "unknown"}:
        should_sync = True
        rc_div, div_out, _ = git_capture(["git", "rev-list", "--left-right", "--count", f"HEAD...origin/{branch}"])
        if rc_div == 0:
            try:
                ahead, behind = (int(part) for part in div_out.split())
                should_sync = (ahead > 0) or (behind > 0)
            except Exception:
                should_sync = True
        if should_sync:
            rc_push, _, err_push = git_capture([
                "git", "push", "--force-with-lease", "origin", branch,
            ])
            if rc_push != 0:
                warning = f" ⚠️ Remote not synced: {err_push}"
                log.warning("Rollback remote sync failed for %s: %s", branch, err_push)

    append_jsonl(
        DRIVE_ROOT / "logs" / "supervisor.jsonl",
        {
            "ts": utc_now_iso(),
            "type": "manual_rollback",
            "target": tag_or_sha,
            "reason": reason,
            "new_sha": st["current_sha"],
            "remote_synced": not bool(warning),
            "branch": branch,
        },
    )
    return True, f"Rolled back to {tag_or_sha} ({st['current_sha'][:8]}){warning}"

def configure_remote(repo_slug: str, token: str) -> Tuple[bool, str]:
    """Configure origin while storing the token in git credential helper."""
    if not repo_slug or not token:
        return False, "Missing repo slug or token"

    clean_url = f"https://github.com/{repo_slug}.git"

    if _has_remote("origin"):
        rc, _, err = git_capture(["git", "remote", "set-url", "origin", clean_url])
    else:
        rc, _, err = git_capture(["git", "remote", "add", "origin", clean_url])
    if rc != 0:
        return False, f"Failed to configure remote: {err}"

    _configure_credential_helper(repo_slug, token)
    return True, "ok"


def configure_personal_remote(
    repo_slug: str,
    token: str,
    *,
    auto_fork: bool = True,
    confirm_replace_origin: bool = False,
) -> Tuple[bool, str, str]:
    """Configure the personal persistence remote (`origin`), ensuring `managed` exists."""
    if not token:
        return False, "Missing GitHub token", ""
    # Ensure the official update path lives on `managed` BEFORE (re)pointing
    # `origin` at the personal repo, so replacing a clone-default `origin` that
    # still points at the official upstream never orphans the official update
    # remote. Shared by every caller (startup + Settings save). Best-effort:
    # personal-origin configuration proceeds even if this step fails.
    try:
        ensure_official_update_remote()
    except Exception:
        log.warning("Official update remote setup failed during personal remote config", exc_info=True)
    resolved_slug = str(repo_slug or "").strip()
    warnings: List[str] = []
    # Always validate a configured slug (rejects the official repo and origin
    # conflicts); only empty-slug fork resolution is gated on auto_fork.
    if resolved_slug or auto_fork:
        try:
            from ouroboros.repo_remotes import ensure_personal_origin_target

            result = ensure_personal_origin_target(
                REPO_DIR,
                token,
                configured_repo=resolved_slug,
                confirm_replace_origin=confirm_replace_origin,
            )
        except Exception as exc:
            return False, f"Personal remote provisioning failed: {exc}", ""
        if not result.ok:
            return False, result.message or result.action or "personal remote provisioning failed", ""
        resolved_slug = result.repo_slug
        warnings = list(result.warnings or [])
    if not resolved_slug:
        return False, "Missing repo slug", ""
    ok, msg = configure_remote(resolved_slug, token)
    if not ok:
        return ok, msg, resolved_slug
    if warnings:
        msg = msg + " (" + "; ".join(warnings[:5]) + ")"
    return True, msg, resolved_slug


def _configure_credential_helper(repo_slug: str, token: str) -> None:
    """Store credentials in repo-local .git/credentials, not global state."""
    cred_path = REPO_DIR / ".git" / "credentials"
    git_capture([
        "git", "config", "--local", "credential.helper",
        f"store --file={cred_path}",
    ])
    cred_line = f"https://x-access-token:{token}@github.com"
    try:
        cred_path.write_text(cred_line + "\n", encoding="utf-8")
        cred_path.chmod(0o600)
    except Exception as e:
        log.warning("Failed to write repo credentials file: %s", e)


def push_to_remote(branch: Optional[str] = None, push_tags: bool = True) -> Tuple[bool, str]:
    """Push current branch (and optionally tags) to origin."""
    if not _has_remote("origin"):
        return False, "No remote configured"

    target = branch or BRANCH_DEV
    rc, out, err = git_capture(["git", "push", "-u", "origin", target])
    if rc != 0:
        return False, f"git push failed: {err}"

    result = f"Pushed {target} to origin"
    if push_tags:
        rc_t, _, err_t = git_capture(["git", "push", "origin", "--tags"])
        if rc_t != 0:
            result += f" (tags push failed: {err_t})"
        else:
            result += " + tags"
    return True, result
