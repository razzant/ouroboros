"""Exact owner Restore and promotion operations."""

from __future__ import annotations

import os
import subprocess
from typing import Any, Dict, Optional, Tuple


def rollback_to_version(tag_or_sha: str, reason: str = "manual_rollback") -> Tuple[bool, str]:
    """Restore one exact commit after pinning committed and dirty local state.

    Raises ``IndexLockRefusal`` (``before_tree_effects``) when Git refused its single reset
    while taking this checkout's index lock: the keep branch and rescue made before it stay.
    """
    from supervisor import git_ops as _g

    rc, target_sha, error = _g.git_capture(
        ["git", "rev-parse", "--verify", f"{tag_or_sha}^{{commit}}"]
    )
    if rc != 0:
        return False, f"Cannot resolve {tag_or_sha}: {error}"
    keep_ok, keep_branch = _g.preserve_local_ref_branch("HEAD", prefix="rollback-keep")
    if not keep_ok:
        return False, f"Could not preserve current HEAD before rollback: {keep_branch}"

    repo_state = _g._collect_repo_sync_state()
    try:
        rescue = _g._create_rescue_snapshot(
            branch=repo_state.get("current_branch", "unknown"),
            reason=reason,
            repo_state=repo_state,
        )
    except Exception as exc:
        _g.log.warning("Rescue snapshot failed before rollback: %s", exc)
        return False, f"Rescue snapshot failed; checkout was left unchanged: {exc}"
    if rescue.get("diff_error"):
        return False, (
            "Rescue snapshot could not capture tracked changes; checkout was left unchanged: "
            f"{rescue['diff_error']}"
        )
    incomplete = _g._rescue_untracked_incomplete(rescue)
    if incomplete:
        return False, (
            "Untracked-file rescue was incomplete; checkout was left unchanged: "
            f"{incomplete}"
        )

    cmd = ["git", "reset", "--hard", target_sha]
    _g._guard_live_repo_destructive_git(cmd)
    rc, out, error = _g.git_capture(cmd, repair_index=False)
    if rc != 0:
        from supervisor.git_ops_reset import IndexLockRefusal, _index_lock_refused

        if not _index_lock_refused(cmd, subprocess.CompletedProcess(cmd, rc, out, error)):
            return False, f"git reset failed: {error}"  # it may have moved the tree: the caller stays fenced
        refusal = IndexLockRefusal(rc, cmd, output=out, stderr=error)
        refusal.before_tree_effects = True  # this reset is the only tree command; Git died taking its lock
        _g.append_jsonl(_g.DRIVE_ROOT / "logs" / "supervisor.jsonl", {
            "ts": _g.utc_now_iso(), "type": "checkout_step_failed", "reason": reason, "target": tag_or_sha,
            "cmd": " ".join(cmd), "returncode": rc, "stderr": error, "keep_branch": keep_branch,
            "rescue_path": rescue.get("path")})
        raise refusal
    from supervisor.git_ops_reset import _record_checkout_facts

    state = {"current_sha": target_sha.strip()}
    _record_checkout_facts(state)

    _g.append_jsonl(
        _g.DRIVE_ROOT / "logs" / "supervisor.jsonl",
        {
            "ts": _g.utc_now_iso(),
            "type": "manual_rollback",
            "target": tag_or_sha,
            "reason": reason,
            "new_sha": state["current_sha"],
            "keep_branch": keep_branch,
            "remote_pushed": False,
            "branch": repo_state.get("current_branch") or _g.BRANCH_DEV,
        },
    )
    return True, (
        f"Rolled back to {tag_or_sha} ({state['current_sha'][:8]}). "
        f"Previous HEAD is preserved on {keep_branch}. Nothing was pushed."
    )


def promote_branch_exact(
    source_branch: str,
    stable_branch: str,
    *,
    push_remote: bool = False,
    remote_name: str = "origin",
    repo_dir: Optional[str] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """Move local/optional remote stable refs to one captured source SHA.

    ``repo_dir`` scopes the ref operations to the caller's repository (the
    events handler passes its ctx.REPO_DIR); by default the supervisor's own
    checkout is used. The remote push always targets the supervisor checkout's
    remotes, so it is honestly skipped for a non-default ``repo_dir``."""
    from supervisor import git_ops as _g

    cwd = str(repo_dir) if repo_dir else str(_g.REPO_DIR)

    def _capture(cmd: list) -> Tuple[int, str, str]:
        env = {**os.environ, "LC_ALL": "C", "LANG": "C"}
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, env=env)
        return proc.returncode, (proc.stdout or "").strip(), (proc.stderr or "").strip()

    rc, source_sha, error = _capture(
        ["git", "rev-parse", "--verify", f"{source_branch}^{{commit}}"]
    )
    if rc != 0 or not source_sha:
        return False, {"error": error or f"cannot resolve {source_branch}"}
    rc, _out, error = _capture(
        ["git", "branch", "-f", stable_branch, source_sha]
    )
    if rc != 0:
        return False, {"error": error or f"cannot update {stable_branch}"}
    rc, landed_sha, error = _capture(
        ["git", "rev-parse", "--verify", f"{stable_branch}^{{commit}}"]
    )
    if rc != 0 or landed_sha != source_sha:
        return False, {"error": f"{stable_branch} did not land on captured SHA {source_sha}"}

    result: Dict[str, Any] = {
        "sha": source_sha,
        "source_branch": source_branch,
        "stable_branch": stable_branch,
        "remote_pushed": False,
    }
    scoped_elsewhere = bool(repo_dir) and os.path.realpath(cwd) != os.path.realpath(str(_g.REPO_DIR))
    if push_remote and not scoped_elsewhere and _g._has_remote(remote_name):
        rc, _out, error = _g._git_network_bounded(
            ["push", remote_name, f"{source_sha}:refs/heads/{stable_branch}"]
        )
        if rc == 0:
            result["remote_pushed"] = True
        else:
            result["remote_error"] = error or f"push to {remote_name} failed"
    return True, result
