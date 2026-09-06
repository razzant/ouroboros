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
import logging
import os
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


def _run_git_resilient(cmd, **kwargs):
    """Run a destructive-checkout git command with index-repair retries."""
    import time
    check = bool(kwargs.pop("check", False))
    _go()._guard_live_repo_destructive_git(list(cmd))
    for attempt in range(5):
        run_kwargs = dict(kwargs)
        run_kwargs.setdefault("capture_output", True)
        run_kwargs.setdefault("text", True)
        result = subprocess.run(cmd, **run_kwargs)
        if result.returncode == 0:
            return result
        if _go()._maybe_repair_git_index(result.stderr):
            time.sleep(0.2)
            continue
        if not check:
            return result
        if attempt == 4:
            raise subprocess.CalledProcessError(
                result.returncode, cmd, output=result.stdout, stderr=result.stderr,
            )
        time.sleep(1)
    return subprocess.run(cmd, check=check, **kwargs)


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


def checkout_and_reset(branch: str, reason: str = "unspecified",
                       unsynced_policy: str = "ignore") -> Tuple[bool, str]:
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
        rc, _, err = _go().git_capture(["git", "fetch", fetch_remote])
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
            _go()._run_git_resilient(["git", "reset", "--hard", "HEAD"], cwd=str(_go().REPO_DIR), check=True)
            _go()._run_git_resilient(["git", "clean", "-fd"], cwd=str(_go().REPO_DIR), check=True)
        _go()._run_git_resilient(["git", "checkout", "-B", branch, target_ref], cwd=str(_go().REPO_DIR), check=True)
        if update_intent_target:
            _go()._run_git_resilient(["git", "reset", "--hard", target_ref], cwd=str(_go().REPO_DIR), check=True)
        _go()._run_git_resilient(["git", "clean", "-fd"], cwd=str(_go().REPO_DIR), check=True)
    else:
        rc_local = subprocess.run(
            ["git", "rev-parse", "--verify", branch],
            cwd=str(_go().REPO_DIR), capture_output=True,
        ).returncode

        if rc_local != 0:
            _go()._run_git_resilient(["git", "reset", "--hard", "HEAD"], cwd=str(_go().REPO_DIR), check=True)
            _go()._run_git_resilient(["git", "clean", "-fd"], cwd=str(_go().REPO_DIR), check=True)
            # §6 (same detached-HEAD class as BUG1): `-b` with check=False silently swallowed a
            # "branch already exists" error and proceeded with HEAD possibly detached/wrong;
            # `-B` force-creates the branch at HEAD and check=True raises a real failure.
            _go()._run_git_resilient(["git", "checkout", "-B", branch], cwd=str(_go().REPO_DIR), check=True)
        else:
            if policy == "rescue_and_reset":
                _go()._run_git_resilient(["git", "reset", "--hard", "HEAD"], cwd=str(_go().REPO_DIR), check=True)
                _go()._run_git_resilient(["git", "clean", "-fd"], cwd=str(_go().REPO_DIR), check=True)
            _go()._run_git_resilient(["git", "checkout", branch], cwd=str(_go().REPO_DIR), check=True)
            _go()._run_git_resilient(["git", "reset", "--hard", "HEAD"], cwd=str(_go().REPO_DIR), check=True)
            if policy == "rescue_and_reset":
                _go()._run_git_resilient(["git", "clean", "-fd"], cwd=str(_go().REPO_DIR), check=True)

    # Checkout may not update mtimes; remove stale bytecode.
    for p in _go().REPO_DIR.rglob("__pycache__"):
        shutil.rmtree(p, ignore_errors=True)
    st = _go().load_state()
    st["current_branch"] = branch
    st["current_sha"] = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=str(_go().REPO_DIR),
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    _go().save_state(st)
    if update_intent_target and st["current_sha"] != update_intent_target:
        return False, f"Update intent checkout landed on {st['current_sha']} but expected {update_intent_target}"
    if pin_bundle_sha:
        _go()._clear_bootstrap_pin_marker()
    if update_intent_target and str(reason or "") != "ui_update_apply":
        _go()._clear_update_intent()
    return True, "ok"


def sync_runtime_dependencies(reason: str) -> Tuple[bool, str]:
    if getattr(sys, 'frozen', False):
        log.info("Skipping pip install in frozen (PyInstaller) mode — deps are bundled.")
        return True, "frozen:bundled"

    from ouroboros.platform_layer import pip_install_target_args

    req_path = _go().REPO_DIR / "requirements-runtime.lock"
    if not req_path.exists():
        # Preserve upgrades from managed repositories created before uv locks.
        req_path = _go().REPO_DIR / "requirements.txt"
    # The sixth and last pip call site. On a packaged install `sys.executable` IS the
    # bundled interpreter, so an unflagged install wrote into the signed bundle.
    cmd: List[str] = [sys.executable, "-m", "pip", "install", "-q",
                      *pip_install_target_args(sys.executable)]
    source = ""
    if req_path.exists():
        cmd += ["-r", str(req_path)]
        source = f"requirements:{req_path}"
    else:
        cmd += ["openai>=1.0.0", "requests"]
        source = "fallback:minimal"
    try:
        from ouroboros.platform_layer import kill_process_tree, subprocess_new_group_kwargs
        from ouroboros.tools.shell import _active_subprocesses, _subprocess_lock

        proc = subprocess.Popen(
            cmd, cwd=str(_go().REPO_DIR), **subprocess_new_group_kwargs()
        )
        with _subprocess_lock:
            _active_subprocesses.add(proc)
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

    r = subprocess.run(
        [sys.executable, "-c", "import ouroboros, ouroboros.agent; print('import_ok')"],
        cwd=str(_go().REPO_DIR),
        capture_output=True, text=True,
    )
    return {"ok": (r.returncode == 0), "stdout": r.stdout, "stderr": r.stderr,
            "returncode": r.returncode}


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

    ok, err = _go().checkout_and_reset(_go().BRANCH_DEV, reason=reason, unsynced_policy=unsynced_policy)
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

    ok_s, err_s = _go().checkout_and_reset(
        _go().BRANCH_STABLE,
        reason=f"{reason}_fallback_stable",
        unsynced_policy="rescue_and_reset",
    )
    if not ok_s:
        return False, f"Failed checkout {_go().BRANCH_STABLE}: {err_s}"

    deps_ok_s, deps_msg_s = _go().sync_runtime_dependencies(reason=f"{reason}_fallback_stable")
    if not deps_ok_s:
        return False, f"Failed deps for {_go().BRANCH_STABLE}: {deps_msg_s}"

    t2 = _go().import_test()
    if t2["ok"]:
        return True, f"OK: fell back to {_go().BRANCH_STABLE}"

    return False, "Both branches failed import (dev and stable)"
