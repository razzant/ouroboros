"""Supervisor git/reset/rescue/dependency operations."""

from __future__ import annotations

import json
import logging
import os
import pathlib
import re
import shutil  # noqa: F401
import subprocess
import sys
import uuid
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

from supervisor.state import (
    append_jsonl, atomic_write_text, load_state, save_state, update_state,  # noqa: F401
)
from ouroboros import config as _config
from ouroboros.git_observation import host_git_env
from ouroboros.utils import utc_now_iso  # noqa: F401

log = logging.getLogger(__name__)


# REPO_DIR / DRIVE_ROOT are UNBOUND until init() (server bootstrap), a worker
# rebind or a test monkeypatch pins them as real module attributes. Until then
# attribute access resolves PER CALL from the OUROBOROS_* env via the config
# path SSOT (module __getattr__ below), so a process that imported this module
# before its path isolation applied still reads and writes its OWN roots —
# not a root baked at import time (issue #455 hermeticity class; the old
# hardcoded ~/Ouroboros default never read the env at all).
REPO_DIR: pathlib.Path
DRIVE_ROOT: pathlib.Path
REMOTE_URL: str = ""
BRANCH_DEV: str = "ouroboros"
BRANCH_STABLE: str = "ouroboros-stable"
MANAGED_REPO_META_NAME = "ouroboros-managed.json"
BOOTSTRAP_PIN_MARKER_NAME = "ouroboros-bootstrap-pending"
UPDATE_INTENT_MARKER_NAME = "ouroboros-update-intent.json"
OFFICIAL_UPDATE_REMOTE_URL = "https://github.com/razzant/ouroboros"


def __getattr__(name: str) -> Any:
    if name == "REPO_DIR":
        return _config.resolve_repo_dir()
    if name == "DRIVE_ROOT":
        return _config.resolve_data_dir()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def current_repo_dir() -> pathlib.Path:
    """The pinned REPO_DIR when init()/a rebind set it, else the env-resolved one."""
    pinned = globals().get("REPO_DIR")
    return pinned if pinned is not None else _config.resolve_repo_dir()


def current_drive_root() -> pathlib.Path:
    """The pinned DRIVE_ROOT when init()/a rebind set it, else the env-resolved one."""
    pinned = globals().get("DRIVE_ROOT")
    return pinned if pinned is not None else _config.resolve_data_dir()


class BootstrapTransitionBusy(RuntimeError):
    """The optional bootstrap transition could not acquire its lock."""


@contextmanager
def bootstrap_transition_lock(timeout_sec: float = 30.0, *, data_dir: Optional[pathlib.Path] = None):
    """Serialize source bootstrap with writers that create transition markers.

    Git's ref transaction protects refs, HEAD and their native lockfiles, but
    adoption/update/evolution ownership is represented in the data tree.  The
    shared lock keeps those ownership records from appearing between the
    repair's final guard and its ref commit.  A lock refusal is a safe refusal
    of the optional repair, never a reason to reset or attach another way.
    """
    from ouroboros.platform_layer import acquire_exclusive_file_lock, release_exclusive_file_lock

    lock_path = pathlib.Path(data_dir) if data_dir is not None else current_drive_root()
    lock_path = lock_path / "locks" / "git-bootstrap.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = acquire_exclusive_file_lock(
        lock_path,
        timeout_sec=float(timeout_sec),
        stale_sec=600.0,
        owner_aware_stale=True,
        # Keep the creator identity in the on-disk stamp.  The platform lock
        # helper uses it to distinguish a dead owner from a live holder when
        # owner-aware stale recovery is enabled; supplying custom metadata
        # otherwise replaces its default ``pid=...`` stamp.
        metadata=f"pid={os.getpid()} locked_at={utc_now_iso()}\n",
        poll_sec=0.2,
    )
    if fd is None:
        raise BootstrapTransitionBusy(lock_path)
    try:
        yield
    finally:
        release_exclusive_file_lock(lock_path, fd)


def _guard_live_repo_destructive_git(cmd: List[str]) -> None:
    if os.environ.get("OUROBOROS_ALLOW_LIVE_REPO_TESTS") == "1":
        return
    try:
        live_repo = current_repo_dir().resolve(strict=False) == (
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
    return current_repo_dir() / ".git"


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
    repo = repo_dir or current_repo_dir()
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


def launcher_managed_repo_matches(
    repo_dir: Optional[pathlib.Path] = None,
    *,
    managed_by_launcher: Optional[bool] = None,
    managed_repo_dir: Optional[str] = None,
) -> bool:
    """Return the single launcher/managed identity predicate used at boot.

    A copied marker is not enough: when the launcher supplies a repo path it
    must name this checkout; when it does not, the managed metadata file is the
    identity witness.  Callers may pass captured server values so import-time
    environment snapshots and bootstrap probes use the same rule.
    """
    repo = pathlib.Path(repo_dir or current_repo_dir())
    marked = (
        str(os.environ.get("OUROBOROS_MANAGED_BY_LAUNCHER", "") or "").strip() == "1"
        if managed_by_launcher is None else bool(managed_by_launcher)
    )
    if not marked:
        return False
    target = (
        str(os.environ.get("OUROBOROS_MANAGED_REPO_DIR", "") or "").strip()
        if managed_repo_dir is None else str(managed_repo_dir or "").strip()
    )
    try:
        target_matches = bool(target) and pathlib.Path(target).resolve(strict=False) == repo.resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        target_matches = False
    if target_matches:
        return True
    # A stale launcher-path environment value must not demote a valid managed
    # checkout. The metadata witness is path-bound: an empty JSON file, or one
    # naming another checkout, is not managed authority. A launcher older than
    # v7.4.1 passes no path and rewrites its pathless record on every start;
    # under it that non-empty record stays the witness (older launchers start
    # newer checkouts, docs/architecture/01), so a copy of it is not told apart.
    try:
        raw = json.loads((repo / ".git" / MANAGED_REPO_META_NAME).read_text(encoding="utf-8"))
        recorded = str(raw.get("managed_repo_dir") or "").strip() if isinstance(raw, dict) else ""
        if not recorded:
            return not target and isinstance(raw, dict) and bool(raw)
        return pathlib.Path(recorded).resolve(strict=False) == repo.resolve(strict=False)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError):
        return False


def _is_launcher_managed_repo() -> bool:
    return launcher_managed_repo_matches()


def _restart_keeps_checkout(deployment_managed: bool) -> Dict[str, Any]:
    """How a Restart stays on what is on disk; ``{}`` when ``checkout_and_reset`` owns it.

    The boot predicate decides: a checkout it does not call launcher-managed keeps its branch, HEAD and files
    unless Update armed an intent (gated as ``checkout_and_reset`` gates it, so a stray marker moves nothing).
    """
    disabled = "with managed checkout disabled"
    live = {"current_branch": None, "current_sha": None}  # a managed past's stamp must not outrank the live read
    if deployment_managed:
        return {"event": "managed_by_deployment", "where": disabled, "receipt": {},
                "ok": "OK: the deployed program, unchanged"}
    if str(os.environ.get("OUROBOROS_DISABLE_MANAGED_UPDATES", "") or "").strip() == "1":
        return {"event": "managed_checkout_disabled", "where": disabled, "receipt": {}, "facts": live,
                "ok": "OK: managed checkout disabled — staying on the current checkout"}
    if _is_launcher_managed_repo():
        return {}
    intent = _read_update_intent()
    if intent and _read_managed_repo_meta() and str(intent.get("branch") or BRANCH_DEV) == BRANCH_DEV:
        return {}
    head = git_capture(["git", "rev-parse", "HEAD"])[1].strip()
    return {"event": "unmanaged_checkout_kept", "where": "on this unmanaged checkout", "receipt": {"head": head},
            "facts": live, "ok": f"OK: unmanaged checkout kept at {head[:12] or 'unknown HEAD'}"}


def _import_failure(result: Dict[str, Any]) -> str:
    """The exit code and stderr's last line, the part of a failed import an owner can act on."""
    last = str(result.get("stderr") or "").strip().splitlines()[-1:]
    return f" (rc={result.get('returncode', -1)})" + "".join(f": {line}" for line in last)


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
    cmd: List[str], *, timeout: Optional[float], cwd: Optional[pathlib.Path] = None,
    env: Optional[Dict[str, str]] = None, text: bool = True,
    input_data: Optional[Any] = None,
) -> Tuple[int, Any, Any]:
    """Run one Git process through ``shell_process.tracked_start`` and terminate its tree on timeout.

    ``timeout=None`` is ``git_capture``'s unbounded wait: no bound, the same
    admission and publication, so Emergency Stop (and a begun stop outside
    ``exit_finalization``) refuses the start as rc 127 or ends the child.
    ``input_data`` (str when ``text=True``, bytes otherwise) is fed to the
    process's stdin and the pipe is closed — the batch-stdin git plumbing
    (e.g. ``git cat-file --batch``) rides the same timeout/kill semantics."""
    from ouroboros.platform_layer import kill_process_tree, subprocess_new_group_kwargs
    from ouroboros.tools.shell_process import (
        CommandAdmissionClosed,
        contained_popen,
        retire_tracked,
        tracked_start,
    )

    limit = None if timeout is None else float(timeout)
    cleanup_timeout = 10  # the existing post-kill wait, also used after communication failures
    empty = "" if text else b""
    try:
        with tracked_start() as publish:
            proc = publish(contained_popen(
                subprocess.Popen, cmd,
                cwd=str(cwd or current_repo_dir()),
                stdin=subprocess.PIPE if input_data is not None else None,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=text,
                env=env,
                **subprocess_new_group_kwargs(),
            ))
    except (OSError, CommandAdmissionClosed) as exc:  # no Git process started
        detail = str(exc)
        return 127, empty, detail if text else detail.encode("utf-8", "replace")

    try:
        try:
            stdout, stderr = proc.communicate(input=input_data, timeout=limit)
        except subprocess.TimeoutExpired as exc:
            partial_error = getattr(exc, "stderr", None)
            kill_process_tree(proc)
            try:
                _tail_out, tail_error = proc.communicate(timeout=cleanup_timeout)
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
        except BaseException:
            # subprocess.run also cleans up on interruption/pipe errors. Keep that contract
            # when using the tracked runner, without replacing the caller's original error.
            try:
                kill_process_tree(proc)
                proc.wait(timeout=cleanup_timeout)
            except BaseException:
                log.warning("Git exception cleanup is unconfirmed; its child stays tracked", exc_info=True)
            raise
        return (
            int(proc.returncode if proc.returncode is not None else 1),
            stdout if stdout is not None else empty,
            stderr if stderr is not None else empty,
        )
    finally:
        retire_tracked(proc)  # a running root, or a dead root's live or unknown tree, stays named
        if proc.returncode is not None:
            for pipe in (proc.stdin, proc.stdout, proc.stderr):
                if pipe is not None:
                    try:
                        pipe.close()
                    except Exception:
                        log.warning("Could not close an exited Git child's pipe", exc_info=True)


def git_capture(
    cmd: List[str], *, timeout: Optional[float] = None, repair_index: bool = False,
) -> Tuple[int, str, str]:
    # Repair is a writer's explicit opt-in, never a side effect of observation.
    # The final stderr alone cannot describe an earlier attempt or its index rebuild.
    # Same reason as utils.run_cmd: this stderr is PARSED (`_maybe_repair_git_index`
    # matches English git diagnostics), so the operator's locale must not decide
    # whether a repairable index error is recognised.
    # ``timeout`` is None for existing non-rescue call sites: a bound here is a
    # behavior change, so it stays opt-in rather than silently retiming them.
    # Bounded or not, the process starts through the tracked runner: Panic is never bypassed.
    env = host_git_env()
    repo = current_repo_dir()
    for _attempt in range(2):
        returncode, stdout, raw_stderr = _run_git_process_bounded(
            cmd, timeout=timeout, cwd=repo, env=env, text=True,
        )
        stderr = str(raw_stderr or "").strip()
        if returncode == 0:
            return returncode, str(stdout or "").strip(), stderr
        if repair_index and _maybe_repair_git_index(stderr, timeout=timeout):
            continue
        return returncode, str(stdout or "").strip(), stderr
    return returncode, str(stdout or "").strip(), stderr


from supervisor import update_source as _update_source

FETCH_TIMEOUT_RC, _git_network_bounded = _update_source.FETCH_TIMEOUT_RC, _update_source._git_network_bounded
_managed_update_target, git_fetch_bounded = _update_source._managed_update_target, _update_source.git_fetch_bounded

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


def _maybe_repair_git_index(stderr: str, *, timeout: Optional[float] = None) -> bool:
    """Back up and rebuild an index whose content Git reports corrupt; True only once rebuilt.

    A held ``index.lock`` (its age proves no owner gone) or an index Git cannot
    open, stat or map is returned to the caller as Git's own refusal; so is a
    rebuild that fails, after the original index is put back beside its backup.
    While the lock exists the index is not touched at all: the rebuild needs that
    lock, and every read would otherwise leave one more backup behind.
    """
    error_lower = str(stderr or "").lower()
    if not any(marker in error_lower for marker in ("index file smaller than expected", "index file corrupt")):
        return False

    repo = current_repo_dir()
    git_dir = repo / ".git"
    if not git_dir.exists() or (git_dir / "index.lock").exists():
        return False

    backup_path = None
    index_path = git_dir / "index"
    if index_path.exists():
        backup_path = git_dir / f"index.corrupt.{uuid.uuid4().hex[:8]}.bak"
        try:
            index_path.replace(backup_path)
            log.warning("Backed up corrupt git index to %s", backup_path)
        except Exception:
            log.warning("Failed to back up corrupt git index %s", index_path, exc_info=True)
            return False

    rebuild_cmd = ["git", "reset", "--mixed", "HEAD"]
    rebuild_env = {**os.environ, "LC_ALL": "C", "LANG": "C"}
    rebuild_rc, rebuild_stdout, rebuild_stderr = _run_git_process_bounded(
        rebuild_cmd, timeout=timeout, cwd=repo, env=rebuild_env, text=True,
    )
    rebuild_error = str(rebuild_stderr or "").strip() or str(rebuild_stdout or "").strip()
    if rebuild_rc == 0:
        log.warning("Rebuilt git index after corruption in %s", repo)
        return True

    log.warning(
        "Failed to rebuild git index after corruption: %s",
        rebuild_error,
    )
    if backup_path is not None:
        # A missing index reads as every tracked file deleted, so the caller must not retry
        # against it: the original bytes return under their own identity (a hard link keeps
        # the backup and never replaces an index another writer published meanwhile).
        try:
            os.link(backup_path, index_path)
        except FileExistsError:
            pass
        except OSError:
            log.warning("Corrupt git index remains only at %s", backup_path, exc_info=True)
    return False


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
    target = repo_dir or current_repo_dir()
    gi = target / ".gitignore"
    if not gi.exists():
        gi.write_text(_REPO_GITIGNORE, encoding="utf-8")


def _ensure_git_identity() -> None:
    """Ensure repo-local git identity exists for local commits/tags."""
    git_capture(["git", "config", "user.name", "Ouroboros"])
    git_capture(["git", "config", "user.email", "ouroboros@local.mac"])


def _ensure_local_version_tag() -> None:
    """Create the current VERSION tag locally when a local-only repo has none."""
    version_path = current_repo_dir() / "VERSION"
    if not version_path.exists():
        return

    version = version_path.read_text(encoding="utf-8").strip().lstrip("v")
    if not re.match(r"^\d+\.\d+\.\d+(?:-?(?:rc|alpha|beta|a|b)\.?\d+)?$", version, re.IGNORECASE):
        return
    from ouroboros.tools.release_sync import is_development_marker

    if is_development_marker(version):  # a cycle's marker is never tagged (BIBLE P9)
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


def _bootstrap_git_markers() -> tuple:
    """Git-operation markers shared by the bootstrap receipt and source repair."""
    from supervisor.update_merge import UPDATE_TX_MARKER_NAME

    return (BOOTSTRAP_PIN_MARKER_NAME, UPDATE_INTENT_MARKER_NAME, UPDATE_TX_MARKER_NAME,
            "MERGE_HEAD", "CHERRY_PICK_HEAD", "REVERT_HEAD", "rebase-apply", "rebase-merge",
            "sequencer", "BISECT_START")


def _detached_repair_blocked() -> bool:
    from ouroboros.body_adoption import helper_dir, holds_checkout, read
    from ouroboros.body_switch import HANDOFF_NAME

    if _is_launcher_managed_repo() or holds_checkout(DRIVE_ROOT, current_repo_dir()):
        return True
    if (helper_dir(DRIVE_ROOT) / HANDOFF_NAME).exists() and not read(DRIVE_ROOT):
        raise RuntimeError("adoption handoff exists but could not be read")
    rc, path, error = rescue_git_capture(["git", "rev-parse", "--absolute-git-dir"])
    if rc or not path:
        raise RuntimeError(error or "Git directory could not be read")
    if any((pathlib.Path(path) / name).exists() for name in _bootstrap_git_markers()):
        return True
    campaign = DRIVE_ROOT / "state" / "evolution_campaign.json"
    if campaign.exists():
        row = json.loads(campaign.read_text(encoding="utf-8"))
        tx = row.get("active_transaction")
        if row.get("status") in {"active", "paused"} and isinstance(tx, dict) and not tx.get("commit_sha"):
            return True
    return False


def _attach_detached_head(branch: str, head: str) -> Tuple[bool, str]:
    state_fd = None
    try:
        # Evolution campaign writes take state.lock before the bootstrap lock;
        # take the same short admission lock here so the campaign cannot appear
        # between the guard and Git's prepared ref commit.
        from supervisor import state

        state_fd = state.acquire_file_lock(state.STATE_LOCK_PATH, timeout_sec=0.0)
        if state_fd is None:
            return False, "state transition lock unavailable"
        with bootstrap_transition_lock():
            return _attach_detached_head_locked(branch, head)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    finally:
        if state_fd is not None:
            try:
                from supervisor import state
                state.release_file_lock(state.STATE_LOCK_PATH, state_fd)
            except Exception:
                log.warning("Detached-HEAD state-lock cleanup is unconfirmed", exc_info=True)


def _attach_detached_head_locked(branch: str, head: str) -> Tuple[bool, str]:
    """Create the branch and attach HEAD in one prepared native Git transaction.

    An oid condition alone also accepts a *symbolic* HEAD at that oid. Read its
    form after prepare, while Git holds HEAD and branch locks, before committing.
    Old Git without symref-update refuses before commit; there is no unsafe fallback.
    No checkout, index refresh, reset or worktree write belongs to this repair.
    """
    import queue
    import threading
    from ouroboros.platform_layer import kill_process_tree, subprocess_new_group_kwargs
    from ouroboros.tools.shell_process import _active_subprocesses, _subprocess_lock, tracked_start
    from ouroboros.update_channels import get_rescue_git_timeout_sec

    timeout = get_rescue_git_timeout_sec()
    proc, reader = None, None
    try:
        with tracked_start() as publish:
            proc = publish(subprocess.Popen(
                ["git", "update-ref", "--no-deref", "--stdin"], cwd=str(current_repo_dir()),
                env=host_git_env(), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, **subprocess_new_group_kwargs()))
        # Bytes avoid CRLF translation in Git's line protocol on Windows.
        proc.stdin.write((f"start\ncreate refs/heads/{branch} {head}\n"
                          f"symref-update HEAD refs/heads/{branch} oid {head}\nprepare\n").encode("utf-8"))
        proc.stdin.flush()
        replies = queue.Queue()

        def prepared():
            replies.put((proc.stdout.readline(), proc.stdout.readline()))

        reader = threading.Thread(target=prepared, daemon=True)
        reader.start()
        reply = replies.get(timeout=timeout)
        reader.join()
        if reply != (b"start: ok\n", b"prepare: ok\n"):
            raise RuntimeError(b"".join(reply).decode("utf-8", "replace").strip())
        # No successful Git writer can move these refs until commit/abort. In
        # particular a concurrently chosen named HEAD at the SAME oid is kept.
        rc, _named, _err = rescue_git_capture(["git", "symbolic-ref", "-q", "HEAD"])
        if rc != 1 or _detached_repair_blocked():
            raise RuntimeError("HEAD is no longer detached, or a checkout transition owns it")
        rc, dirty, _err = rescue_git_capture(["git", "--no-optional-locks", "status", "--porcelain"])
        if rc or dirty:
            raise RuntimeError("the tree is no longer known clean")
        output, _ = proc.communicate(input=b"commit\n", timeout=timeout)
        if proc.returncode != 0 or output != b"commit: ok\n":
            raise RuntimeError(output.decode("utf-8", "replace").strip() or "commit outcome unconfirmed")
        return True, ""
    except Exception as exc:
        # An ordinary refusal releases Git's prepared locks through its protocol,
        # rather than killing it and leaving lockfiles behind.
        if proc is not None and proc.poll() is None and (reader is None or not reader.is_alive()):
            try:
                proc.communicate(input=b"abort\n", timeout=timeout)
            except Exception:
                log.warning("Could not confirm transaction abort", exc_info=True)
        return False, f"{type(exc).__name__}: {exc}"
    finally:
        if proc is not None:
            if proc.poll() is None:
                kill_process_tree(proc)  # an interrupted Git transaction is never retried
            if reader is not None:
                reader.join(timeout=10)
            try:
                if reader is None or not reader.is_alive():
                    proc.communicate(timeout=10)
            except Exception:
                log.warning("Detached-HEAD transaction cleanup is unconfirmed", exc_info=True)
            if proc.poll() is not None and (reader is None or not reader.is_alive()):
                with _subprocess_lock:
                    _active_subprocesses.discard(proc)
                for pipe in (proc.stdin, proc.stdout):
                    if pipe is not None:
                        pipe.close()
            else:
                log.warning("Detached-HEAD Git process/pipe cleanup is unconfirmed; its handle stays tracked")


def _repair_detached_head_branch() -> None:
    """Repair only a clean, idle source checkout on a known remote dev line.

    Managed reset and adoption retain their owners. Unknown probes refuse this
    optional repair; branch creation and HEAD attachment commit together, or an
    uncommitted transaction aborts without rewriting the owner's refs or files.
    """
    import shlex

    branch = managed_branch_defaults()[0]
    hint = f"git switch -c <new-branch> HEAD (unused name), or git switch {shlex.quote(branch)} (existing branch)"

    rc, current, _err = rescue_git_capture(["git", "symbolic-ref", "-q", "--short", "HEAD"])
    if rc == 0 and current.strip():
        return  # on a named branch; the common boot pays one cheap git call
    if rc != 1:
        log.warning(
            "Could not tell whether HEAD is detached (git exited %s); leaving the checkout "
            "alone. Run `%s` yourself if the branch is missing.", rc, hint)
        return
    rc, head, _err = rescue_git_capture(["git", "rev-parse", "--verify", "HEAD^{commit}"])
    if rc != 0 or not head.strip():
        log.warning(
            "HEAD is not a readable commit; not attaching branch '%s'. Run `%s` at the "
            "commit you mean.", branch, hint)
        return
    head = head.strip()
    rc, _out, _err = rescue_git_capture(["git", "check-ref-format", f"refs/heads/{branch}"])
    if rc:
        log.warning("Invalid dev branch name %r; choose a valid new branch with git switch -c.", branch)
        return
    try:
        if _detached_repair_blocked():
            log.warning("A checkout transition owns this tree; no detached-HEAD repair. After it settles, use %s.", hint)
            return
    except Exception as exc:
        log.warning("Could not check checkout transitions: %s; no detached-HEAD repair. After inspection, use %s.", exc, hint)
        return

    # With --quiet, rc 1 is exactly "no such ref"; any other nonzero is a Git
    # failure and must never read as "the branch is absent".
    rc, _out, _err = rescue_git_capture(
        ["git", "rev-parse", "--verify", "--quiet", f"refs/heads/{branch}"])
    if rc == 0:
        log.warning(
            "HEAD is detached while local branch '%s' exists elsewhere; leaving both alone. "
            "Run `%s` yourself only if that is the branch you mean.", branch, hint)
        return
    if rc != 1:
        log.warning(
            "Could not tell whether local branch '%s' exists (git exited %s); leaving the "
            "detached HEAD at %s alone. Run `%s` yourself if this is the branch you mean.",
            branch, rc, head[:12], hint)
        return

    on_line = ""
    for remote in _list_remotes(capture=rescue_git_capture):
        remote_ref = f"refs/remotes/{remote}/{branch}"
        rc, _out, _err = rescue_git_capture(
            ["git", "rev-parse", "--verify", "--quiet", remote_ref])
        if rc != 0:
            continue
        rc, _out, _err = rescue_git_capture(
            ["git", "merge-base", "--is-ancestor", head, remote_ref])
        if rc == 0:
            on_line = remote_ref
            break
    if not on_line:
        log.warning(
            "HEAD is detached at %s and no remote release line (*/%s) contains it; not creating "
            "branch '%s'. Run `%s` at the HEAD you mean if this is deliberate.",
            head[:12], branch, branch, hint)
        return

    rc, status, _err = rescue_git_capture(["git", "status", "--porcelain"])
    if rc != 0:
        log.warning(
            "Could not tell whether the tree at detached HEAD %s is clean; not creating "
            "branch '%s'. Run `%s` once the tree is settled.", head[:12], branch, hint)
        return
    if status.strip():
        log.warning(
            "HEAD is detached at %s (on %s) but the tree is not clean; not creating branch '%s'. "
            "Commit or stash your work, then run `%s`.", head[:12], on_line, branch, hint)
        return

    attached_ok, error = _attach_detached_head(branch, head)
    if not attached_ok:
        log.warning(
            "Detached-HEAD attachment was not confirmed: %s. Safe automatic repair requires "
            "Git update-ref --stdin with symref-update and prepared transactions. No retry or "
            "rollback was attempted; inspect HEAD/refs, then use %s.", error, hint)
        return
    rc_now, attached, _err = rescue_git_capture(["git", "symbolic-ref", "-q", "--short", "HEAD"])
    rc_at, at, _err = rescue_git_capture(["git", "rev-parse", "HEAD^{commit}"])
    if rc_now != 0 or attached.strip() != branch or rc_at != 0 or at.strip() != head:
        log.warning(
            "Attaching branch '%s' did not land on the expected end state (HEAD %s, branch "
            "%r); leaving the tree as Git left it. Run `%s` only if that is what you mean.",
            branch, (at.strip() or "?")[:12], attached.strip(), hint)
        return
    try:
        append_jsonl(DRIVE_ROOT / "logs" / "supervisor.jsonl", {
            "ts": utc_now_iso(),
            "type": "detached_head_branch_repaired",
            "branch": branch,
            "head": head,
            "on_line": on_line,
        })
    except Exception:
        log.warning("Could not record the detached-HEAD repair row; the repair stands.",
                    exc_info=True)
    log.info("Attached detached HEAD at %s to local branch '%s' (%s)", head[:12], branch, on_line)


def ensure_repo_present() -> None:
    repo = current_repo_dir()
    if not (repo / ".git").exists():
        if _is_launcher_managed_repo():
            raise RuntimeError(
                "Launcher-managed repo is missing .git metadata. "
                "The launcher bootstrap must recreate REPO_DIR from the embedded repo bundle."
            )
        # REPO_DIR is live code: initialize in place, never remove it.
        repo.mkdir(parents=True, exist_ok=True)
        _ensure_repo_gitignore()
        import dulwich.repo
        dulwich.repo.Repo.init(str(repo))

        _ensure_git_identity()

        rc, _, _ = git_capture(["git", "status", "--porcelain"])
        if rc == 0:
            subprocess.run(["git", "add", "-A"], cwd=str(repo), check=True)
            subprocess.run(["git", "commit", "-m", "Initial commit from bundle"], cwd=str(repo), check=False)

        subprocess.run(["git", "branch", "-M", BRANCH_DEV], cwd=str(repo), check=False)
        subprocess.run(["git", "branch", BRANCH_STABLE], cwd=str(repo), check=False)

    if not _is_launcher_managed_repo():
        try:
            _repair_detached_head_branch()
        except Exception:
            log.warning("Detached-HEAD branch repair failed; boot continues without it.", exc_info=True)
        _ensure_local_version_tag()


# Owner recovery surface lives in supervisor/update_recovery.py; re-exported because
# callers/tests address it through the git_ops facade (cycle-free: update_recovery
# imports git_ops only inside functions).
from supervisor.update_recovery import promote_branch_exact, rollback_to_version  # noqa: E402,F401
# --- G1 split re-exports (v7 module-size discipline) -----------------------
# The rescue/sync-state surface lives in supervisor/git_ops_rescue.py (G1
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


# The checkout/reset admission, dependency-sync and import-test surface lives
# in supervisor/git_ops_reset.py (G1 split); re-exported because callers/tests
# address it through the git_ops facade (cycle-free: the leaf imports git_ops
# only at call time through its _go() handle).
from supervisor.git_ops_reset import (  # noqa: E402,F401
    _admission_gate_for_unsynced_tree,
    _compute_ref_ahead_count,
    _preserve_branch_for_official_reset,
    _ref_points_at_ref,
    _run_git_resilient,
    advance_recovery_point,
    checkout_and_reset,
    import_test,
    preserve_local_ref_branch,
    safe_restart,
    sync_runtime_dependencies,
)


# The managed-update status surface lives in supervisor/git_ops_updates.py
# (G1 split); re-exported because callers/tests address it through the git_ops
# facade (cycle-free: the leaf imports git_ops only at call time through its
# _go() handle).
from supervisor.git_ops_updates import (  # noqa: E402,F401
    _public_repo_url,
    compute_managed_update_status,
    ensure_official_update_remote,
    list_commits,
    list_official_update_tags,
    list_versions,
    managed_update_remote_url,
    prepare_managed_update,
)


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
