"""Checkout/reset admission, dependency sync and safe restart, split out of
``supervisor/git_ops.py`` (module-size discipline, v7 G1 split).

Owns the destructive tree movement and its admission gate: the resilient
checkout/reset runner, the unsynced-policy block/rescue rules, the runtime
dependency sync, the import smoke and the dev-then-stable safe restart. The
parent keeps the rebindable module state (``init`` REBINDS REPO_DIR/BRANCH_*
and friends), the capture plumbing and the marker/meta probes, and re-exports
every name here, so ``supervisor.git_ops`` stays the one public surface.
Parent members and rebindable globals are read through the call-time handle
``_go()`` — never a from-import, which would freeze the binding this module
saw at import time.
"""

from __future__ import annotations

import datetime
import functools
import logging
import os
import pathlib
import re
import shutil
import subprocess
import sys
import uuid
from typing import Any, Dict, List, Optional, Tuple



def _go():
    """The parent module, read at call time.

    ``supervisor.git_ops`` owns the rebindable module state (``init`` REBINDS
    REPO_DIR, DRIVE_ROOT and BRANCH_*) and the helpers tests monkeypatch on
    the parent (``git_capture``, ``_guard_live_repo_destructive_git``, the
    sibling re-exports). Reading them through the module keeps one binding: a
    from-import here would freeze the value this module saw at import time.
    """
    from supervisor import git_ops

    return git_ops


# The parent's logger name is pinned so moved log records keep their `%(name)s`
# in server.log/stdout — the same logger object the parent binds.
log = logging.getLogger("supervisor.git_ops")


def _compute_ref_ahead_count(ref: str, target_ref: str) -> Tuple[bool, int, str]:
    """Return whether *ref* is ahead of *target_ref*, failing closed on errors."""
    if not ref or not target_ref:
        return False, 0, "missing ref for ahead comparison"
    rc, counts, err = _go().git_capture([
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
    rc_left, left_sha, _ = _go().git_capture(["git", "rev-parse", "--verify", left_ref])
    if rc_left != 0 or not left_sha:
        return False
    rc_right, right_sha, _ = _go().git_capture(["git", "rev-parse", "--verify", right_ref])
    return rc_right == 0 and bool(right_sha) and left_sha.strip() == right_sha.strip()


def preserve_local_ref_branch(ref: str = "HEAD", prefix: str = "local-keep") -> Tuple[bool, str]:
    """Create a local branch pointing at *ref* before replacing it."""
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    branch_name = f"{prefix}-{now}-{uuid.uuid4().hex[:6]}"
    rc, _out, err = _go().git_capture(["git", "branch", branch_name, ref])
    if rc != 0:
        return False, err or f"failed to create {branch_name}"
    return True, branch_name


def _preserve_branch_for_official_reset(
    branch: str,
    target_ref: str,
    update_intent: Dict[str, Any],
) -> Tuple[bool, str]:
    """Ensure local commits survive an explicit official update reset."""
    count_ok, ahead, count_error = _go()._compute_ref_ahead_count(branch, target_ref)
    if not count_ok:
        return False, f"Could not compare {branch} with update target {target_ref}: {count_error}"
    if ahead <= 0:
        return True, ""
    existing = str(update_intent.get("keep_branch") or "").strip()
    if existing and _go()._ref_points_at_ref(existing, branch):
        return True, existing
    ok, branch_or_error = _go().preserve_local_ref_branch(branch)
    if not ok:
        return False, branch_or_error
    return True, branch_or_error


class IndexLockRefusal(subprocess.CalledProcessError):
    """Every attempt of a ``git reset``/``git checkout`` died taking this checkout's ``index.lock``.

    Both take that lock before writing the index, the tree or a ref, so the command
    itself changed nothing. ``before_tree_effects`` is its raiser's fact that no earlier
    tree step of the same call ran (``checkout_and_reset``; Restore's single reset).
    """
    before_tree_effects = False


_LOCK_REFUSAL = re.compile(r"fatal: Unable to create '(.+)': File exists\.")


def effective_index_path() -> pathlib.Path:
    """Ask Git, including linked worktrees and an inherited GIT_INDEX_FILE."""
    rc, path, error = _go().git_capture(["git", "rev-parse", "--path-format=absolute", "--git-path", "index"])
    if rc != 0:
        raise RuntimeError(error or "Git could not locate its index")
    if len(path.splitlines()) == 1 and pathlib.Path(path).is_absolute():
        return pathlib.Path(path)
    # Git < 2.31 echoes the unknown --path-format flag and still exits zero.
    # Resolve the per-worktree Git directory, retaining alternate-index semantics.
    alternate = os.environ.get("GIT_INDEX_FILE")
    if alternate:
        index = pathlib.Path(alternate)
    else:
        rc, git_dir, error = _go().git_capture(["git", "rev-parse", "--git-dir"])
        if rc != 0 or len(git_dir.splitlines()) != 1:
            raise RuntimeError(error or "Git could not locate its directory")
        index = pathlib.Path(git_dir) / "index"
    return (_go().current_repo_dir() / index).absolute()


def observe_index_lock() -> dict:
    """An observation only: existence never establishes ownership or removal authority."""
    path = pathlib.Path(str(effective_index_path()) + ".lock")
    try:
        stat = path.lstat()
    except FileNotFoundError:
        return {"path": str(path), "exists": False}
    return {"path": str(path), "exists": True, "inode": stat.st_ino, "size": stat.st_size}


def _index_lock_refused(cmd, result) -> bool:
    """This attempt died acquiring this checkout's index lock (C-locale Git, first stderr line)."""
    if result.returncode != 128 or [str(part) for part in cmd[:2]] not in (["git", "reset"], ["git", "checkout"]):
        return False
    stderr = result.stderr.decode("utf-8", "replace") if isinstance(result.stderr, bytes) else str(result.stderr or "")
    match = _LOCK_REFUSAL.fullmatch((stderr.splitlines() or [""])[0].strip())
    try:
        return bool(match) and pathlib.Path(match.group(1)).resolve() == pathlib.Path(str(effective_index_path()) + ".lock").resolve()
    except (OSError, RuntimeError):
        return False


def _run_git_resilient(cmd, **kwargs):
    """Run a destructive-checkout git command with index-repair retries.

    The raise is an ``IndexLockRefusal`` only when every attempt died acquiring the
    index lock and no repair ran between them; anything else is a plain refusal.
    """
    import time
    check = bool(kwargs.pop("check", False))
    kwargs.setdefault("env", {**os.environ, "LC_ALL": "C", "LANG": "C"})  # stderr is parsed below
    _go()._guard_live_repo_destructive_git(list(cmd))
    lock_refused = True
    for attempt in range(5):
        run_kwargs = dict(kwargs)
        run_kwargs.setdefault("capture_output", True)
        run_kwargs.setdefault("text", True)
        result = subprocess.run(cmd, **run_kwargs)
        if result.returncode == 0:
            return result
        lock_refused = lock_refused and _index_lock_refused(cmd, result)
        if _go()._maybe_repair_git_index(result.stderr):
            lock_refused = False
            time.sleep(0.2)
            continue
        if not check:
            return result
        if attempt == 4:
            refusal = IndexLockRefusal if lock_refused else subprocess.CalledProcessError
            raise refusal(result.returncode, cmd, output=result.stdout, stderr=result.stderr)
        time.sleep(1)
    return subprocess.run(cmd, check=check, **kwargs)


def _tree_step(done: List[str], cmd: List[str]) -> None:
    """One tree-mutating ``checkout_and_reset`` command; ``done`` lists those already applied."""
    try:
        _go()._run_git_resilient(cmd, cwd=str(_go().REPO_DIR), check=True)
    except IndexLockRefusal as exc:
        exc.before_tree_effects = not done
        raise
    done.append(" ".join(cmd))


def _admission_gate_for_unsynced_tree(
    branch: str, reason: str, policy: str, update_intent_target: str,
) -> Optional[Tuple[bool, str]]:
    """Apply unsynced_policy's block/rescue rules for checkout_and_reset.

    Returns ``(False, msg)`` when the reset must stop here, or ``None`` to proceed.
    """
    repo_state = _go()._collect_repo_sync_state()
    dirty_lines = list(repo_state.get("dirty_lines") or [])
    unpushed_lines = list(repo_state.get("unpushed_lines") or [])
    unpushed_needs_rescue = bool(update_intent_target and unpushed_lines)

    # A failed status read or an unconsulted MERGE_HEAD used to read as a clean
    # tree; force the same rescue/block branch a dirty tree takes, matching the
    # fail-closed read already used for the managed-update rollback path.
    status_unreadable = any(
        str(w).startswith("status_error:") for w in (repo_state.get("warnings") or [])
    )
    merge_in_progress = False
    merge_head_unreadable = False
    # Keep the process-free path for normal clones.  Linked worktrees use a
    # .git pointer file, so ask Git for the worktree-specific admin path there.
    git_dir = _go()._git_dir()
    merge_head_path = git_dir / "MERGE_HEAD"
    if git_dir.is_file():
        rc_path, merge_head_rel, _path_err = _go().rescue_git_capture(
            ["git", "rev-parse", "--git-path", "MERGE_HEAD"]
        )
        if rc_path == 0 and merge_head_rel:
            merge_head_path = _go().REPO_DIR / merge_head_rel
        else:
            merge_head_unreadable = True

    # A present file whose content is not a SHA is unreadable, not absent, per
    # the issue's fix direction.
    if merge_head_path.is_file():
        try:
            merge_head_content = merge_head_path.read_text(encoding="utf-8").strip()
        except Exception:
            merge_head_content = ""
        if re.fullmatch(r"[0-9a-fA-F]{7,64}", merge_head_content):
            merge_in_progress = True
        else:
            merge_head_unreadable = True

    if dirty_lines or unpushed_needs_rescue or status_unreadable or merge_in_progress \
            or merge_head_unreadable:
        bits: List[str] = []
        if unpushed_lines and (dirty_lines or unpushed_needs_rescue):
            bits.append(f"unpushed={len(unpushed_lines)}")
        if dirty_lines:
            bits.append(f"dirty={len(dirty_lines)}")
        if status_unreadable:
            bits.append("status_unreadable")
        if merge_in_progress:
            bits.append("merge_in_progress")
        if merge_head_unreadable:
            bits.append("merge_head_unreadable")
        detail = ", ".join(bits) if bits else "unsynced"
        rescue_info: Dict[str, Any] = {}
        if policy in {"rescue_and_block", "rescue_and_reset"}:
            try:
                rescue_info = _go()._create_rescue_snapshot(
                    branch=branch, reason=reason, repo_state=repo_state)
            except Exception as e:
                rescue_info = {"error": repr(e)}
            if policy == "rescue_and_reset" and rescue_info.get("error"):
                msg = (
                    f"Reset blocked ({detail}) because rescue snapshot failed: "
                    f"{rescue_info.get('error')}. Local changes were left untouched."
                )
                _go().append_jsonl(
                    _go().DRIVE_ROOT / "logs" / "supervisor.jsonl",
                    {
                        "ts": _go().utc_now_iso(),
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
                _go().append_jsonl(
                    _go().DRIVE_ROOT / "logs" / "supervisor.jsonl",
                    {
                        "ts": _go().utc_now_iso(),
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
            untracked_rescue_error = _go()._rescue_untracked_incomplete(rescue_info)
            if policy == "rescue_and_reset" and untracked_rescue_error:
                msg = (
                    f"Reset blocked ({detail}) because untracked-file rescue was incomplete: "
                    f"{untracked_rescue_error}. Local changes were left untouched."
                )
                _go().append_jsonl(
                    _go().DRIVE_ROOT / "logs" / "supervisor.jsonl",
                    {
                        "ts": _go().utc_now_iso(),
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
            _go().append_jsonl(
                _go().DRIVE_ROOT / "logs" / "supervisor.jsonl",
                {
                    "ts": _go().utc_now_iso(),
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

        _go().append_jsonl(
            _go().DRIVE_ROOT / "logs" / "supervisor.jsonl",
            {
                "ts": _go().utc_now_iso(),
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
    return None


def _target_lacks_store(sha: str) -> bool:
    """Whether ``sha`` is a tree the store switch must reach (``store_switch.switch_needs``)."""
    from ouroboros.store_switch import switch_needs

    return switch_needs(_go().REPO_DIR, sha)["facts"]


def checkout_and_reset(branch: str, reason: str = "unspecified",
                       unsynced_policy: str = "ignore") -> Tuple[bool, str]:
    receipt = _bootstrap_receipt()
    if reason == "bootstrap" and receipt and receipt.get("branch") == branch:
        if receipt == _bootstrap_snapshot():
            _record_checkout_facts({"current_branch": branch, "current_sha": receipt["head"]},
                                   invalidate_bootstrap=False)
            return True, "unchanged bootstrap"
    _forget_bootstrap_receipt()
    before_head = _go().git_capture(["git", "rev-parse", "HEAD"])[1].strip()
    managed_meta = _go()._read_managed_repo_meta()
    fetch_remote = ""
    target_ref = ""
    pin_bundle_sha = _go()._pin_to_bundle_sha_on_bootstrap(reason, managed_meta)
    update_intent = _go()._read_update_intent()
    update_intent_target = ""
    intent_keep_branch = ""
    if managed_meta and not pin_bundle_sha and update_intent:
        intent_branch = str(update_intent.get("branch") or _go().BRANCH_DEV)
        intent_sha = str(update_intent.get("target_sha") or "").strip()
        if intent_branch == branch:
            from supervisor.update_merge import read_update_tx_strict

            tx_status, update_tx = read_update_tx_strict()
            tx_phase = str(update_tx.get("phase") or "")
            tx_matches = bool(
                tx_status == "valid"
                and tx_phase in {"applying_replace", "pending_boot_smoke"}
                and str(update_tx.get("target_sha") or "").strip() == intent_sha
                and str(update_tx.get("pre_update_branch") or _go().BRANCH_DEV) == branch
            )
            rc_intent = -1
            if intent_sha:
                rc_intent, _sha_out, _sha_err = _go().git_capture(
                    ["git", "rev-parse", "--verify", f"{intent_sha}^{{commit}}"]
                )
            constitution_ok = bool(
                tx_matches
                and intent_sha
                and rc_intent == 0
                and _go()._update_source.official_ref_has_constitution(
                    intent_sha, repo_dir=_go().REPO_DIR
                )
            )
            if constitution_ok:
                # Only the authority switch's own door checkout reaches a store-less target, after its
                # export; a bootstrap or restart replay leaves the armed intent to the boot finalizer.
                if str(reason or "") == "ui_update_apply" or not _target_lacks_store(intent_sha):
                    update_intent_target = intent_sha
                    target_ref = intent_sha
                    intent_keep_branch = str(update_intent.get("keep_branch") or "").strip()
            else:
                cleared = _go()._clear_update_intent()
                _go().append_jsonl(
                    _go().DRIVE_ROOT / "logs" / "supervisor.jsonl",
                    {
                        "ts": _go().utc_now_iso(),
                        "type": "managed_update_intent_invalid",
                        "target_branch": branch,
                        "target_sha": intent_sha,
                        "tx_status": tx_status,
                        "tx_phase": tx_phase,
                        "tx_target_sha": str(update_tx.get("target_sha") or ""),
                        "cleared": cleared,
                    },
                )
                detail = intent_sha[:12] if intent_sha else "missing SHA"
                return False, (
                    f"Managed update intent is invalid ({detail}); checkout was left unchanged. "
                    + ("The marker was cleared." if cleared else "The marker could not be cleared.")
                )
    if not managed_meta and not pin_bundle_sha and _go()._has_remote("origin"):
        fetch_remote = "origin"

    if fetch_remote:
        rc, _, err = _go()._git_network_bounded(["fetch", fetch_remote])
        if rc != 0:
            msg = f"git fetch {fetch_remote} failed: {err or 'unknown error'}"
            _go().append_jsonl(
                _go().DRIVE_ROOT / "logs" / "supervisor.jsonl",
                {
                    "ts": _go().utc_now_iso(),
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
        admission_result = _go()._admission_gate_for_unsynced_tree(
            branch, reason, policy, update_intent_target)
        if admission_result is not None:
            return admission_result

    remote_ref_exists = False
    if target_ref:
        remote_ref_exists = subprocess.run(
            ["git", "rev-parse", "--verify", target_ref],
            cwd=str(_go().REPO_DIR),
            capture_output=True,
        ).returncode == 0

    done: List[str] = []  # tree-mutating commands completed by this call (`_tree_step`)
    if remote_ref_exists:
        if update_intent_target:
            preserve_ok, preserve_msg = _go()._preserve_branch_for_official_reset(
                branch, target_ref, update_intent,
            )
            if not preserve_ok:
                return False, f"Could not preserve local branch before official update: {preserve_msg}"
            if preserve_msg and preserve_msg != intent_keep_branch:
                _go().append_jsonl(
                    _go().DRIVE_ROOT / "logs" / "supervisor.jsonl",
                    {
                        "ts": _go().utc_now_iso(),
                        "type": "ui_update_preserved_late_head",
                        "target_branch": branch,
                        "reason": reason,
                        "target_ref": target_ref,
                        "keep_branch": preserve_msg,
                    },
                )
            _tree_step(done, ["git", "reset", "--hard", "HEAD"])
            _tree_step(done, ["git", "clean", "-fd"])
        _tree_step(done, ["git", "checkout", "-B", branch, target_ref])
        if update_intent_target:
            _tree_step(done, ["git", "reset", "--hard", target_ref])
        _tree_step(done, ["git", "clean", "-fd"])
    else:
        rc_local = subprocess.run(
            ["git", "rev-parse", "--verify", branch],
            cwd=str(_go().REPO_DIR), capture_output=True,
        ).returncode

        if rc_local != 0:
            _tree_step(done, ["git", "reset", "--hard", "HEAD"])
            _tree_step(done, ["git", "clean", "-fd"])
            # §6 (same detached-HEAD class as BUG1): `-b` with check=False silently swallowed a
            # "branch already exists" error and proceeded with HEAD possibly detached/wrong;
            # `-B` force-creates the branch at HEAD and check=True raises a real failure.
            _tree_step(done, ["git", "checkout", "-B", branch])
        else:
            if policy == "rescue_and_reset":
                _tree_step(done, ["git", "reset", "--hard", "HEAD"])
                _tree_step(done, ["git", "clean", "-fd"])
            _tree_step(done, ["git", "checkout", branch])
            _tree_step(done, ["git", "reset", "--hard", "HEAD"])
            if policy == "rescue_and_reset":
                _tree_step(done, ["git", "clean", "-fd"])

    st = {"current_branch": branch, "current_sha": subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=str(_go().REPO_DIR),
        capture_output=True, text=True, check=True,
    ).stdout.strip()}
    if before_head and st["current_sha"] != before_head:
        for p in _go().REPO_DIR.rglob("__pycache__"):
            shutil.rmtree(p, ignore_errors=True)
    _record_checkout_facts(st)
    if update_intent_target and st["current_sha"] != update_intent_target:
        return False, f"Update intent checkout landed on {st['current_sha']} but expected {update_intent_target}"
    if pin_bundle_sha:
        _go()._clear_bootstrap_pin_marker()
    if update_intent_target and str(reason or "") != "ui_update_apply":
        _go()._clear_update_intent()
    return True, "ok"


def _dependency_install_plan(repo_dir: pathlib.Path) -> Tuple[List[str], str, pathlib.Path]:
    """``(command, source, lock path)`` of the one runtime-dependency install for ``repo_dir``."""
    from ouroboros.platform_layer import pip_install_target_args

    req_path = repo_dir / "requirements-runtime.lock"
    if not req_path.exists():
        # Preserve upgrades from managed repositories created before uv locks.
        req_path = repo_dir / "requirements.txt"
    # The sixth and last pip call site. On a packaged install `sys.executable` IS the
    # bundled interpreter, so an unflagged install wrote into the signed bundle.
    cmd: List[str] = [sys.executable, "-m", "pip", "install", "-q",
                      *pip_install_target_args(sys.executable)]
    if req_path.exists():
        return cmd + ["-r", str(req_path)], f"requirements:{req_path}", req_path
    return cmd + ["openai>=1.0.0", "requests"], "fallback:minimal", req_path


def runtime_dependency_command(repo_dir: pathlib.Path) -> Optional[List[str]]:
    """The same install command for a body switch that runs with no launcher; ``None`` where
    this chokepoint installs nothing (a frozen bundle, an active test boundary)."""
    if getattr(sys, 'frozen', False) or os.environ.get("OUROBOROS_PYTEST_ACTIVE") == "1":
        return None
    return _dependency_install_plan(pathlib.Path(repo_dir))[0]


def sync_runtime_dependencies(reason: str) -> Tuple[bool, str]:
    if getattr(sys, 'frozen', False):
        log.info("Skipping pip install in frozen (PyInstaller) mode — deps are bundled.")
        return True, "frozen:bundled"

    # The one dependency-install chokepoint every caller reaches — bootstrap,
    # managed update pre-restart and the reset fallbacks all route here — so the
    # test boundary belongs on THIS line, not on one caller's local-dev branch.
    # A verification run must never mutate the interpreter it is verifying (the
    # operator's, or a fixture's dependency-only venv): the marker is set by
    # tests/conftest.py and re-injected into every scrubbed child environment
    # (ouroboros/test_environment.py), so a server started by a test inherits it.
    # Production sets no such marker and keeps its install policy unchanged.
    if os.environ.get("OUROBOROS_PYTEST_ACTIVE") == "1":
        log.info("Skipping dependency sync under an active test boundary (%s).", reason)
        return True, "pytest:suppressed"

    cmd, source, req_path = _dependency_install_plan(_go().REPO_DIR)
    try:
        from ouroboros.startup_migrations import watermarks, stamp
        fingerprint = _dependency_fingerprint(req_path, cmd)
        try:
            synced = watermarks(_go().DRIVE_ROOT).get("dependencies")
        except ValueError:
            synced = None  # an unreadable watermark file means "not known synced": sync
        if synced == fingerprint:
            return True, "unchanged:" + source
        from ouroboros.platform_layer import kill_process_tree, subprocess_new_group_kwargs
        from ouroboros.tools.shell_process import _active_subprocesses, _subprocess_lock, tracked_start

        with tracked_start() as publish:
            proc = publish(subprocess.Popen(
                cmd, cwd=str(_go().REPO_DIR), **subprocess_new_group_kwargs()
            ))
        try:
            returncode = proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            kill_process_tree(proc)
            proc.wait(timeout=10)
            raise
        finally:
            with _subprocess_lock:
                _active_subprocesses.discard(proc)
        if returncode != 0:
            raise subprocess.CalledProcessError(returncode, cmd)
        try:
            stamp(_go().DRIVE_ROOT, dependencies=fingerprint)
        except Exception:
            log.warning("Dependency fingerprint not recorded; the next sync runs pip again", exc_info=True)
        _go().append_jsonl(
            _go().DRIVE_ROOT / "logs" / "supervisor.jsonl",
            {
                "ts": _go().utc_now_iso(),
                "type": "deps_sync_ok", "reason": reason, "source": source,
            },
        )
        return True, source
    except Exception as e:
        msg = repr(e)
        _go().append_jsonl(
            _go().DRIVE_ROOT / "logs" / "supervisor.jsonl",
            {
                "ts": _go().utc_now_iso(),
                "type": "deps_sync_error", "reason": reason, "source": source, "error": msg,
            },
        )
        return False, msg


def import_test() -> Dict[str, Any]:
    if getattr(sys, 'frozen', False):
        log.info("Skipping import_test in frozen (PyInstaller) mode — modules are bundled.")
        return {"ok": True, "skipped": "frozen"}

    before = _bootstrap_snapshot()
    if before and before == _bootstrap_receipt():  # this exact clean snapshot passed an earlier boot's smoke: record it
        _SMOKED["head"] = before["head"] if before["branch"] == _go().BRANCH_DEV and _go()._is_launcher_managed_repo() else ""
        return {"ok": True, "skipped": "unchanged_bootstrap"}
    _forget_bootstrap_receipt()
    _SMOKED.clear()
    head = _clean_dev_head()  # read before the smoke: a commit landing during it is not what passed
    r = subprocess.run(
        [sys.executable, "-c", "import ouroboros, ouroboros.agent; print('import_ok')"],
        cwd=str(_go().REPO_DIR),
        capture_output=True, text=True,
    )
    if r.returncode == 0 and head == _clean_dev_head():
        _SMOKED["head"] = head
    if r.returncode != 0:
        _forget_dependency_fingerprint()  # a broken environment is re-synced next time, never skipped
    elif before and before == _bootstrap_snapshot():
        from ouroboros.utils import atomic_write_json
        try:
            atomic_write_json(_bootstrap_receipt_path(), before)
        except OSError:
            log.warning("Bootstrap receipt not recorded; the next boot verifies again", exc_info=True)
    return {"ok": (r.returncode == 0), "stdout": r.stdout, "stderr": r.stderr,
            "returncode": r.returncode}


_SMOKED: Dict[str, str] = {}  # this process's passing smoke of a clean, launcher-managed dev checkout


def _clean_dev_head() -> str:
    """HEAD of a clean, launcher-managed dev checkout, else ``""``: what a passing smoke proves for
    ``advance_recovery_point`` (an update being finalized, its receipt kept off, counts)."""
    out = [_go().git_capture(["git", *args]) for args in (["rev-parse", "--abbrev-ref", "HEAD"], ["status", "--porcelain"],
                                                         ["rev-parse", "HEAD"])]
    clean = not any(rc for rc, _o, _e in out) and out[0][1].strip() == _go().BRANCH_DEV and not out[1][1].strip()
    return out[2][1].strip() if clean and _go()._is_launcher_managed_repo() else ""


def advance_recovery_point(ready: bool) -> str:
    """After a ready boot, ``update-ref`` the local recovery branch (never pushed; Save recovery point moves it at
    once) to the commit this process's smoke proved, never live HEAD; returns it, ``""`` when nothing moved."""
    head, ref = _SMOKED.get("head", ""), f"refs/heads/{_go().BRANCH_STABLE}"
    if not ready or not head or _go().git_capture(["git", "rev-parse", "--verify", "--quiet", ref])[1].strip() == head:
        return ""
    rc, _out, err = _go().git_capture(["git", "update-ref", ref, head])
    if rc != 0:
        log.warning("Recovery point not advanced to %s: %s", head[:12], err)
        return ""
    _go().append_jsonl(_go().current_drive_root() / "logs" / "supervisor.jsonl",
                       {"ts": _go().utc_now_iso(), "type": "recovery_point_advanced", "ref": ref, "to": head})
    return head


def _recovery_checkout(reason: str) -> Tuple[bool, str]:
    """The recovery point, checked out through the store-switch owner: a target that cannot read this
    installation's stores is never started automatically (a typed failure; the owner restores); one that can
    gets the stores handed off, then the branch is checked out as before (the dev branch is never rewritten)."""
    from ouroboros import store_switch

    repo, stable, why = _go().REPO_DIR, _go().BRANCH_STABLE, f"{reason}_fallback_stable"
    target = store_switch.rev_parse(repo, stable)
    if target and any(store_switch.switch_needs(repo, target).values()):
        _rc, dev_head, _ = _go().git_capture(["git", "rev-parse", "HEAD"])
        return False, (f"recovery_point_predates_stores: {_go().BRANCH_DEV} {dev_head.strip()[:12] or '?'} failed its "
                       f"import smoke and the recovery point {stable} ({target[:12]}) cannot read this "
                       "installation's stores, so it was not started; use Restore on the Updates page")
    if not target:  # nothing to examine: the checkout itself reports it
        return _checkout_outcome(stable, why, "rescue_and_reset")
    ok, message, _deferred = store_switch.switch_or_reset(_go().current_drive_root(), repo, target_sha=target, target_ref=stable,
                                                          reason=why, door="safe_restart_fallback",
                                                          reset=lambda: _checkout_outcome(stable, why, "rescue_and_reset"))
    return ok, message


def _forget_dependency_fingerprint() -> None:
    try:
        from ouroboros.startup_migrations import stamp
        stamp(_go().DRIVE_ROOT, dependencies=None)
    except Exception:
        log.warning("Dependency fingerprint not cleared after a failed import test", exc_info=True)


def _checkout_outcome(branch: str, reason: str, unsynced_policy: str) -> Tuple[bool, str]:
    """``checkout_and_reset`` whose Git refusal is a failed outcome, never a raise.

    Git explains the refusal (e.g. an ``index.lock`` another process holds or a
    crash left); it is recorded once and returned in Git's own words (``refusal_text``).
    """
    try:
        return _go().checkout_and_reset(branch, reason=reason, unsynced_policy=unsynced_policy)
    except subprocess.CalledProcessError as exc:
        cmd, detail = refusal_parts(exc)
        _go().append_jsonl(
            _go().current_drive_root() / "logs" / "supervisor.jsonl",
            {"ts": _go().utc_now_iso(), "type": "checkout_step_failed", "reason": reason,
             "target_branch": branch, "cmd": cmd, "returncode": exc.returncode, "stderr": detail},
        )
        return False, refusal_text(exc)


def _git_said(exc: subprocess.CalledProcessError) -> str:
    output = exc.stderr or exc.output or ""
    return output.decode("utf-8", "replace") if isinstance(output, bytes) else str(output)


def refusal_parts(exc: subprocess.CalledProcessError) -> Tuple[str, str]:
    """The refused command and Git's own stderr, each on one line."""
    cmd = " ".join(map(str, exc.cmd)) if isinstance(exc.cmd, (list, tuple)) else str(exc.cmd)
    return cmd, " ".join(_git_said(exc).split())


def refusal_text(exc: BaseException) -> str:
    """A checkout refusal: a headline line (the step, Git's own first line, what is known of the
    tree), then Git's whole output on the next line. Owner notices show the headline only."""
    if not isinstance(exc, subprocess.CalledProcessError):
        return str(exc)
    cmd, detail = refusal_parts(exc)
    first = next((" ".join(line.split()) for line in _git_said(exc).splitlines() if line.strip()), "no output")
    scope = ("no checkout step had changed the working tree" if getattr(exc, "before_tree_effects", False)
             else "earlier checkout steps may already have changed the working tree")
    return f"`{cmd}` failed (exit {exc.returncode}): {first}; {scope}.\nGit: {detail or 'no output'}"


def safe_restart(
    reason: str,
    unsynced_policy: str = "rescue_and_reset",
) -> Tuple[bool, str]:
    """Checkout dev, sync deps, import-test, then fall back to stable if needed.

    ``OUROBOROS_DISABLE_MANAGED_UPDATES=1`` is the stand lever: it keeps the deps
    sync and the import test but skips the checkout, so a stand pinned to one sha
    stays on it. This is the choke point EVERY unrequested tree move goes through
    (bootstrap, owner restart, agent restart) — the local-dev bootstrap branch in
    server.py only covered the first of the three. An explicit owner version
    change (Update / Rollback) calls ``checkout_and_reset`` directly and is
    deliberately still honoured: that one the operator asked for.
    """
    if str(os.environ.get("OUROBOROS_DISABLE_MANAGED_UPDATES", "") or "").strip() == "1":
        _go().append_jsonl(
            _go().current_drive_root() / "logs" / "supervisor.jsonl",
            {"ts": _go().utc_now_iso(), "type": "managed_checkout_disabled",
             "reason": reason, "target_branch": _go().BRANCH_DEV},
        )
        deps_ok, deps_msg = _go().sync_runtime_dependencies(reason=reason)
        if not deps_ok:
            return False, f"Failed deps with managed checkout disabled: {deps_msg}"
        t = _go().import_test()
        if t["ok"]:
            return True, "OK: managed checkout disabled — staying on the current checkout"
        return False, f"Import test failed with managed checkout disabled (rc={t.get('returncode', -1)})"

    ok, err = _checkout_outcome(_go().BRANCH_DEV, reason, unsynced_policy)
    if not ok:
        return False, f"Failed checkout {_go().BRANCH_DEV}: {err}"

    deps_ok, deps_msg = _go().sync_runtime_dependencies(reason=reason)
    if not deps_ok:
        return False, f"Failed deps for {_go().BRANCH_DEV}: {deps_msg}"

    t = _go().import_test()
    if t["ok"]:
        return True, f"OK: {_go().BRANCH_DEV}"

    _go().append_jsonl(
        _go().current_drive_root() / "logs" / "supervisor.jsonl",
        {
            "ts": _go().utc_now_iso(),
            "type": "safe_restart_dev_import_failed",
            "reason": reason,
            "branch": _go().BRANCH_DEV,
            "stdout": t.get("stdout", ""),
            "stderr": t.get("stderr", ""),
            "returncode": t.get("returncode", -1),
        },
    )

    ok_s, err_s = _recovery_checkout(reason)
    if not ok_s:
        return False, (err_s if err_s.startswith("recovery_point_predates_stores")
                       else f"Failed checkout {_go().BRANCH_STABLE}: {err_s}")

    deps_ok_s, deps_msg_s = _go().sync_runtime_dependencies(reason=f"{reason}_fallback_stable")
    if not deps_ok_s:
        return False, f"Failed deps for {_go().BRANCH_STABLE}: {deps_msg_s}"

    t2 = _go().import_test()
    if t2["ok"]:
        return True, f"OK: fell back to {_go().BRANCH_STABLE}"

    return False, "Both branches failed import (dev and stable)"


def _record_checkout_facts(facts: dict, *, invalidate_bootstrap: bool = True) -> None:
    """Field-update the checkout projection (#1307): never a stale whole-state write."""
    from supervisor.state import StateUnavailable

    if invalidate_bootstrap:
        # Restore and managed rollback also reach this seam, including a reset
        # back to the same HEAD. They must never inherit an earlier smoke receipt.
        _forget_bootstrap_receipt()
    try:
        _go().update_state(lambda live: live.update(facts))
    except StateUnavailable:
        log.warning("Checkout facts not recorded in state: runtime state unavailable", exc_info=True)
        return
    try:
        from ouroboros.startup_migrations import stamp
        stamp(_go().DRIVE_ROOT, observed_state_sha=facts.get("current_sha"))
    except Exception:
        # Bookkeeping never fails a checkout: without the stamp the next start re-imports.
        log.warning("Observed checkout SHA not stamped; the next start re-imports the obligation sets", exc_info=True)


def _dependency_fingerprint(req_path, cmd):
    """Actual requirement bytes, install target/environment and interpreter. A packaged host
    (``OUROBOROS_BUNDLE_DIR`` with its bundle manifest) counts by content: paths inside the bundle are relative
    to it in every field and its identity (``_packaged_host``, app version, ``sys.version``) is hashed, so the
    same app moved, remounted or re-extracted keeps its receipt while another app or Python re-syncs; paths
    outside it (the userbase, a container's venv), flags and the requirements digest stay as they are."""
    import hashlib
    import json
    import site
    import sysconfig
    source = req_path.read_bytes() if req_path.exists() else b"openai>=1.0.0\nrequests"
    purelib = sysconfig.get_path("purelib")
    bundle = os.environ.get("OUROBOROS_BUNDLE_DIR", "").strip()
    host = _packaged_host(bundle, purelib) if bundle else {}

    def at(value):
        relative = _inside_bundle(value, bundle) if host else None
        return value if relative is None else "<bundle>/" + relative

    facts = {"requirements_sha256": hashlib.sha256(source).hexdigest(), "command": [at(part) for part in cmd],
             "interpreter": [at(sys.executable), sys.version, at(sys.prefix), at(sys.base_prefix)],
             "target": [at(site.getuserbase()), at(purelib)],
             "environment": {key: at(os.environ.get(key, "")) for key in
                             ("PYTHONUSERBASE", "PIP_TARGET", "PIP_PREFIX", "PIP_USER", "PIP_CONFIG_FILE", "VIRTUAL_ENV")}}
    if host:
        facts["host"] = {**host, "app_version": os.environ.get("OUROBOROS_APP_VERSION", "")}
    return hashlib.sha256(json.dumps(facts, sort_keys=True).encode("utf-8")).hexdigest()


def _inside_bundle(value, bundle):
    """``value``'s path relative to the bundle root when it is an absolute path inside it (lexically or
    resolved), else ``None``: flags, external destinations and other volumes stay as they are."""
    if not isinstance(value, str) or not os.path.isabs(value):
        return None
    for root, text in ((bundle, value), (os.path.realpath(bundle), os.path.realpath(value))):
        try:
            relative = os.path.relpath(os.path.abspath(text), os.path.abspath(root))
        except ValueError:  # another Windows drive
            continue
        if relative != os.pardir and not relative.startswith(os.pardir + os.sep):
            return relative.replace(os.sep, "/")
    return None


@functools.lru_cache(maxsize=None)
def _packaged_host(bundle, purelib):
    """The packaged host's identity, read once per process (``{}`` without a bundle manifest); distributions come from
    the bundle's own purelib folder names (``name-version.dist-info``), never a venv the sync itself writes."""
    import platform
    from ouroboros.utils import read_json_dict
    manifest = read_json_dict(pathlib.Path(bundle) / "repo_bundle_manifest.json") or {}
    if not manifest:
        return {}
    try:
        names = ([entry.rsplit(".", 1)[0] for entry in os.listdir(purelib) if entry.endswith((".dist-info", ".egg-info"))]
                 if _inside_bundle(purelib, bundle) is not None else [])
    except OSError:
        names = []
    return {"cache_tag": sys.implementation.cache_tag, "machine": platform.machine(),
            "source_sha": manifest.get("source_sha", ""), "bundle_sha256": manifest.get("bundle_sha256", ""),
            "distributions": sorted("==".join(name.rsplit("-", 1)) for name in names)}


def _bootstrap_receipt_path() -> pathlib.Path:
    return _go().DRIVE_ROOT / "state" / "git_bootstrap.json"


def _bootstrap_receipt() -> dict:
    from ouroboros.utils import read_json_dict
    return read_json_dict(_bootstrap_receipt_path()) or {}


def _forget_bootstrap_receipt() -> None:
    # Invalidation must finish before a tree mutation can reuse the same HEAD.
    _bootstrap_receipt_path().unlink(missing_ok=True)


def _bootstrap_snapshot() -> Optional[dict]:
    """Current clean checkout and dependency identity; unknown/active work is no proof.

    Transaction markers (including corrupt ones) force the ordinary bootstrap.
    Reads are addressed; no history or remote fetch participates in the proof.
    The receipt this snapshot is compared with is written only by a successful
    import smoke; a failed or interrupted smoke certifies no later boot.
    """
    import json
    from supervisor.update_merge import UPDATE_TX_MARKER_NAME
    from ouroboros.body_adoption import holds_checkout

    try:
        repo, drive = _go().REPO_DIR, _go().DRIVE_ROOT
        if (_go()._bootstrap_pin_marker_path().exists() or _go()._update_intent_marker_path().exists()
                or holds_checkout(drive, repo)):
            return None
        git_dir = _go()._git_dir()
        if git_dir.is_file():
            rc, path, _ = _go().git_capture(["git", "rev-parse", "--absolute-git-dir"])
            if rc or not path:
                return None
            git_dir = pathlib.Path(path.strip())
        if any((git_dir / name).exists() for name in
               (_go().BOOTSTRAP_PIN_MARKER_NAME, _go().UPDATE_INTENT_MARKER_NAME,
                UPDATE_TX_MARKER_NAME, "MERGE_HEAD", "CHERRY_PICK_HEAD", "REVERT_HEAD",
                "rebase-apply", "rebase-merge", "sequencer")):
            return None
        campaign_path = drive / "state" / "evolution_campaign.json"
        if campaign_path.exists():
            campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
            tx = campaign.get("active_transaction")
            if campaign.get("status") in {"active", "paused"} and isinstance(tx, dict) and not tx.get("commit_sha"):
                return None
        rc, dirty, _ = _go().git_capture(["git", "status", "--porcelain"])
        if rc or dirty:
            return None
        rc, head, _ = _go().git_capture(["git", "rev-parse", "HEAD"])
        if rc or not head:
            return None
        rc, branch, _ = _go().git_capture(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        if rc or not branch:
            return None
        cmd, _, requirements = _dependency_install_plan(repo)
        return {"schema_version": 1, "repo": str(repo.resolve()), "head": head.strip(),
                "branch": branch.strip(), "dependencies": _dependency_fingerprint(requirements, cmd)}
    except Exception:
        log.debug("Bootstrap identity unavailable; running the ordinary checks", exc_info=True)
        return None
