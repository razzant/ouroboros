"""Managed-update merge engine (P2): a REAL git 3-way merge in an isolated temp worktree,
the apply / rollback / smoke / finalize primitives, and a FAIL-CLOSED update lock.

Kept OUT of ``git_ops`` (module-size discipline) but depends on it for the live-repo git
helpers — referenced via the ``git_ops`` module object (``_g.X``) so a test that
monkeypatches ``git_ops.REPO_DIR`` / ``_managed_update_target`` / ``_git_dir`` /
``DRIVE_ROOT`` is followed by these primitives. Control plane: ``ouroboros.gateway.control``
orchestrates lock → kill workers → re-plan → rescue → tx marker → apply → smoke → restart.
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

from ouroboros.utils import append_jsonl, utc_now_iso
from supervisor import git_ops as _g

UPDATE_TX_MARKER_NAME = "ouroboros-update-tx.json"


def _git_run(
    cmd: List[str], *, cwd: Optional[str] = None, extra_env: Optional[Dict[str, str]] = None
) -> Tuple[int, str, str]:
    """Run a git command with an optional cwd / extra env (e.g. GIT_INDEX_FILE), WITHOUT
    the REPO_DIR pin and index-repair retry of ``git_capture``. For merge-planning in a
    temp index / temp worktree only — never the live-repo control path."""
    env = dict(os.environ)
    if extra_env:
        env.update(extra_env)
    r = subprocess.run(cmd, cwd=str(cwd or _g.REPO_DIR), capture_output=True, text=True, env=env)
    return r.returncode, (r.stdout or "").strip(), (r.stderr or "").strip()


def plan_managed_update_merge(
    fetch: bool = False, branch: Optional[str] = None, build: bool = False
) -> Dict[str, Any]:
    """Dry-run the managed update as a REAL 3-way merge in an ISOLATED temp worktree and
    classify the result (P2). NEVER touches the live worktree or index. Returns a
    ``merge_plan`` dict: available/kind/auto_mergeable, the doc/code/protected conflict
    split, target_sha/base_sha, local_dirty_count, recommended_strategy. Best-effort:
    always cleans up the temp index + worktree; classification uses update_merge_policy.

    When ``build=True`` AND the merge is clean, the merged tree is committed as a real
    merge commit (parents = [local_snapshot, target]) whose sha is returned as
    ``merge_commit`` — a durable object in the shared DB that survives temp-worktree
    removal, ready for ``apply_managed_merge_update`` to land on the live repo."""
    import shutil
    import tempfile

    from supervisor.update_merge_policy import classify_conflicts

    branch_dev = branch or _g.BRANCH_DEV
    remote_name, _remote_branch, target_ref = _g._managed_update_target(branch_dev)
    if not target_ref:
        return {"available": False, "kind": "unavailable", "error": "no managed update remote"}
    if fetch and remote_name:
        _g.git_capture(["git", "fetch", "--quiet", remote_name])

    rc_t, target_sha, _te = _g.git_capture(["git", "rev-parse", "--verify", f"{target_ref}^{{commit}}"])
    rc_h, base_sha, _he = _g.git_capture(["git", "rev-parse", "--verify", "HEAD"])
    if rc_t != 0 or rc_h != 0 or not target_sha or not base_sha:
        return {"available": False, "kind": "unavailable", "error": "could not resolve target/HEAD"}
    if target_sha == base_sha:
        return {"available": False, "kind": "current", "target_sha": target_sha, "base_sha": base_sha}

    _rc, dirty_out, _de = _g.git_capture(["git", "status", "--porcelain"])
    local_dirty_count = len([ln for ln in dirty_out.splitlines() if ln.strip()])

    tmp_index_path = None
    tmp_wt = None
    try:
        # 1. Local snapshot commit = HEAD + tracked-dirty + untracked. Built in a TEMP
        #    index (GIT_INDEX_FILE), so the live index is untouched. ``git add -A`` honors
        #    .gitignore, so ignored build/secret junk is excluded from durable history.
        fd, tmp_index_path = tempfile.mkstemp(prefix="ouro-update-index-")
        os.close(fd)
        # `git read-tree` wants a NON-existent index path — an existing zero-byte file errors
        # ("index file smaller than expected") on some git versions. Unlink so git creates it
        # fresh; the finally block's unlink is OSError-guarded if it's already gone.
        os.unlink(tmp_index_path)
        env = {"GIT_INDEX_FILE": tmp_index_path}
        if _git_run(["git", "read-tree", "HEAD"], extra_env=env)[0] != 0:
            return {"available": True, "kind": "unknown", "target_sha": target_sha,
                    "base_sha": base_sha, "error": "read-tree failed"}
        _git_run(["git", "add", "-A"], extra_env=env)
        rc_wt, local_tree, _we = _git_run(["git", "write-tree"], extra_env=env)
        if rc_wt != 0 or not local_tree:
            return {"available": True, "kind": "unknown", "target_sha": target_sha,
                    "base_sha": base_sha, "error": "write-tree failed"}
        rc_ct, local_snapshot, _ce = _git_run(
            ["git", "commit-tree", local_tree, "-p", base_sha,
             "-m", "ouroboros local snapshot (update merge plan)"],
            extra_env=env,
        )
        if rc_ct != 0 or not local_snapshot:
            return {"available": True, "kind": "unknown", "target_sha": target_sha,
                    "base_sha": base_sha, "error": "commit-tree failed"}

        # 2. Isolated temp worktree at the snapshot; merge the target THERE (never live).
        #    Use a NON-existent child path (git worktree add refuses an existing dir).
        tmp_wt = os.path.join(tempfile.mkdtemp(prefix="ouro-update-wt-"), "wt")
        rc_add, _ao, add_err = _g.git_capture(["git", "worktree", "add", "--detach", tmp_wt, local_snapshot])
        if rc_add != 0:
            return {"available": True, "kind": "unknown", "target_sha": target_sha,
                    "base_sha": base_sha, "error": f"worktree add failed: {add_err}"}
        # --no-commit --no-ff: leave the merged/conflicted index in place to inspect.
        _git_run(["git", "-C", tmp_wt, "merge", "--no-commit", "--no-ff", target_sha])
        rc_u, unmerged_out, _ue = _git_run(["git", "-C", tmp_wt, "diff", "--name-only", "--diff-filter=U"])
        unmerged = [ln.strip() for ln in unmerged_out.splitlines() if ln.strip()] if rc_u == 0 else []

        plan = classify_conflicts(unmerged)
        kind = str(plan["kind"])
        merge_commit = ""
        if build and kind == "clean":
            # Commit the (clean) merged tree as a real merge commit in the shared object
            # DB so it survives temp-worktree removal and can be landed on the live repo.
            rc_mt, merged_tree, _mte = _git_run(["git", "-C", tmp_wt, "write-tree"])
            if rc_mt == 0 and merged_tree:
                rc_mc, built, _mce = _git_run([
                    "git", "commit-tree", merged_tree,
                    "-p", local_snapshot, "-p", target_sha,
                    "-m", f"Merge official Ouroboros update {target_sha[:12]} (auto)",
                ])
                if rc_mc == 0 and built:
                    merge_commit = built
        return {
            "available": True,
            "kind": kind,
            "auto_mergeable": kind == "clean",
            "doc_conflict_paths": plan["doc_conflict_paths"],
            "code_conflict_paths": plan["code_conflict_paths"],
            "protected_conflict_paths": plan["protected_conflict_paths"],
            "hot_code_paths": plan["hot_code_paths"],
            "target_sha": target_sha,
            "base_sha": base_sha,
            "local_dirty_count": local_dirty_count,
            "local_snapshot": local_snapshot,
            "merge_commit": merge_commit,
            "recommended_strategy": "auto_merge" if kind == "clean" else "assisted",
        }
    except Exception as exc:  # pragma: no cover — planning is best-effort
        _g.log.warning("plan_managed_update_merge failed", exc_info=True)
        return {"available": True, "kind": "unknown", "target_sha": target_sha,
                "base_sha": base_sha, "error": f"{type(exc).__name__}: {exc}"}
    finally:
        if tmp_wt:
            _g.git_capture(["git", "worktree", "remove", "--force", tmp_wt])
            shutil.rmtree(os.path.dirname(tmp_wt), ignore_errors=True)
            _g.git_capture(["git", "worktree", "prune"])
        if tmp_index_path:
            try:
                os.unlink(tmp_index_path)
            except OSError:
                pass


def _update_tx_marker_path():
    return _g._git_dir() / UPDATE_TX_MARKER_NAME


def read_update_tx() -> Dict[str, Any]:
    import json

    path = _update_tx_marker_path()
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except Exception:
        return {}


def write_update_tx(payload: Dict[str, Any]) -> None:
    from ouroboros.utils import atomic_write_json

    atomic_write_json(_update_tx_marker_path(), payload, trailing_newline=True)


def clear_update_tx() -> None:
    try:
        _update_tx_marker_path().unlink()
    except FileNotFoundError:
        return
    except Exception:
        _g.log.warning("Failed to clear update tx marker", exc_info=True)


_ASSISTED_PHASES = ("materializing_assisted", "assisted_resolution", "committing_assisted")


def read_update_tx_strict() -> Tuple[str, Dict[str, Any]]:
    """Strict tx read for safety-critical gates (commit authorization, tx-active rejection):
    return ``(status, tx)`` where status is ``"absent"`` / ``"valid"`` / ``"corrupt"``. A
    marker that exists but is unreadable/invalid is ``corrupt`` — callers MUST fail closed
    (block mutative update/commit ops) rather than treat it as ``absent``."""
    import json

    path = _update_tx_marker_path()
    if not path.is_file():
        return "absent", {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return "corrupt", {}
    if not isinstance(raw, dict) or not raw:
        return "corrupt", {}
    return "valid", raw


def active_update_tx() -> Dict[str, Any]:
    """Return the active tx dict if a (valid or corrupt) marker is present, else ``{}``. A
    corrupt marker counts as ACTIVE (fail-closed) so a second apply cannot proceed over it."""
    status, tx = read_update_tx_strict()
    if status == "absent":
        return {}
    return tx or {"phase": "corrupt"}


def authorized_assisted_task(task_id: str) -> Dict[str, Any]:
    """Return the active assisted tx iff ``task_id`` is its authorized resolver, else ``{}``.
    The tx marker — never an LLM-supplied value — is the trust root for the managed merge."""
    status, tx = read_update_tx_strict()
    if status != "valid":
        return {}
    if str(tx.get("phase") or "") not in _ASSISTED_PHASES:
        return {}
    if str(tx.get("task_id") or "") != str(task_id or ""):
        return {}
    return tx


def _rev_parse(ref: str) -> str:
    rc, out, _e = _g.git_capture(["git", "rev-parse", "--verify", f"{ref}^{{commit}}"])
    return out if rc == 0 else ""


def _merge_head_sha() -> str:
    rc, out, _e = _g.git_capture(["git", "rev-parse", "--verify", "-q", "MERGE_HEAD"])
    return out if rc == 0 else ""


def create_rescue_local_ref(local_snapshot: str) -> str:
    """Pin the local snapshot (the ONLY home of the owner's uncommitted+untracked work) to a
    durable branch so a later rollback / git-gc can never lose it. Returns the branch name."""
    short = (local_snapshot or "")[:12]
    name = f"rescue-local-{short}"
    if local_snapshot:
        _g.git_capture(["git", "branch", "-f", name, local_snapshot])
    return name


def materialize_assisted_merge_live(
    branch: str, local_snapshot: str, target_sha: str, pre_update_sha: str
) -> Tuple[bool, str]:
    """Stage a REAL ``git merge --no-commit --no-ff target`` into the LIVE worktree (MERGE_HEAD +
    a conflicted index + markers) for the agent to resolve and the unmodified ``commit_reviewed``
    to finalize as a reviewed 2-parent commit. Caller MUST hold the update lock with workers
    stopped. Conflicts make ``git merge`` exit nonzero — that is EXPECTED, not failure: success is
    judged by MERGE_HEAD == target_sha. Returns (ok, message).

    P3 immune integrity: the merge is computed FROM ``local_snapshot`` (which captures the owner's
    committed + dirty + untracked work, so nothing is lost), but the first parent is then re-based
    to ``pre_update_sha`` (the last REVIEWED committed state) via a soft reset, so the reviewed
    ``git diff --cached`` (pre_update_sha → resolved) INCLUDES the owner's uncommitted/untracked
    work — none of it reaches history as an unreviewed parent."""
    if not local_snapshot or not target_sha or not pre_update_sha:
        return False, "missing local_snapshot/target_sha/pre_update_sha"
    # Clean the worktree first (dirty + untracked are all captured in local_snapshot + the rescue
    # snapshot + the rescue-local ref) so `checkout -B` cannot fail on "untracked file would be
    # overwritten"; checkout restores them from local_snapshot as tracked content. A real 3-way
    # merge needs a clean tree to run.
    _g.git_capture(["git", "reset", "--hard", "HEAD"])
    _g.git_capture(["git", "clean", "-fd"])
    rc_c, _o, e_c = _g.git_capture(["git", "checkout", "-B", branch, local_snapshot])
    if rc_c != 0:
        return False, f"checkout -B {branch} {local_snapshot[:12]} failed: {e_c}"
    # Ignore the merge return code; conflicts are expected. Judge by MERGE_HEAD.
    _g.git_capture(["git", "merge", "--no-commit", "--no-ff", target_sha])
    mh = _merge_head_sha()
    if not mh:
        return False, "merge produced no MERGE_HEAD (nothing to merge or fatal error)"
    if mh != target_sha:
        return False, f"MERGE_HEAD {mh[:12]} != target {target_sha[:12]}"
    # Re-base the first parent to the reviewed pre-update state WITHOUT disturbing the merge
    # result: `git reset --soft` is refused mid-merge, so move the branch ref directly with
    # update-ref (HEAD follows the symbolic ref) — the index (conflicted/merged entries), the
    # worktree, and MERGE_HEAD are all untouched, so commit_reviewed still makes a 2-parent
    # commit [pre_update_sha, target] whose reviewed diff (pre_update_sha → resolved) includes
    # the owner's dirty/untracked work.
    rc_r, _ro, e_r = _g.git_capture(["git", "update-ref", f"refs/heads/{branch}", pre_update_sha])
    if rc_r != 0:
        return False, f"update-ref {branch} -> {pre_update_sha[:12]} failed: {e_r}"
    if _merge_head_sha() != target_sha:
        return False, "MERGE_HEAD lost after re-parenting the branch"
    return True, f"materialized merge of {target_sha[:12]} (parent={pre_update_sha[:12]}, MERGE_HEAD set)"


def _assisted_head_state(tx: Dict[str, Any]) -> str:
    """Classify the live HEAD vs the assisted tx for boot recovery — keyed on MERGE STATE. During
    resolution HEAD == pre_update_sha (the merge result is staged but uncommitted); the reviewed
    merge commit has pre_update_sha as its FIRST parent and target_sha as its second:
      - ``committed``  : HEAD is a 2-parent commit whose 2nd parent is target_sha (descends from
                         pre_update_sha), or tx.merge_commit is in HEAD.
      - ``in_progress``: HEAD == pre_update_sha (no commit yet — re-materialize/resume).
      - ``diverged``   : HEAD descends from pre_update_sha but is NOT the target merge (a real
                         reviewed commit landed on top — keep it, never reset over it).
      - ``unknown``    : cannot resolve (fail safe: keep)."""
    pre = str(tx.get("pre_update_sha") or "")
    target_sha = str(tx.get("target_sha") or "")
    merge_commit = str(tx.get("merge_commit") or "")
    rc_h, head, _he = _g.git_capture(["git", "rev-parse", "--verify", "HEAD"])
    if rc_h != 0 or not head:
        return "unknown"
    if merge_commit and (
        head == merge_commit
        or _g.git_capture(["git", "merge-base", "--is-ancestor", merge_commit, "HEAD"])[0] == 0
    ):
        return "committed"
    if pre and head == pre:
        return "in_progress"
    # A merge commit whose 2nd parent is the target and which descends from pre_update_sha.
    if pre and target_sha:
        rc_p, parents, _pe = _g.git_capture(["git", "rev-list", "--parents", "-n", "1", "HEAD"])
        descends = _g.git_capture(["git", "merge-base", "--is-ancestor", pre, "HEAD"])[0] == 0
        if rc_p == 0 and target_sha in parents.split()[1:] and descends:
            return "committed"
        if descends:
            return "diverged"
    return "unknown"


def acquire_update_lock():
    """Acquire the FAIL-CLOSED managed-update lock; return an open file handle that keeps
    the lock held. Raise RuntimeError if another update operation holds it — the update
    MUST NOT proceed unlocked (a self-mod write or owner-restart racing the reset has
    corrupted trees before). Release with ``release_update_lock(fh)``."""
    from ouroboros.platform_layer import file_lock_exclusive_nb

    lock_dir = _g.DRIVE_ROOT / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    fh = (lock_dir / "managed_update.lock").open("a+")
    try:
        file_lock_exclusive_nb(fh.fileno())  # raises OSError if already held
    except OSError as exc:
        fh.close()
        raise RuntimeError("managed_update.lock is held by another update operation") from exc
    return fh


def release_update_lock(fh) -> None:
    from ouroboros.platform_layer import file_unlock

    try:
        file_unlock(fh.fileno())
    except Exception:
        pass
    try:
        fh.close()
    except Exception:
        pass


def apply_managed_merge_update(branch: str, merge_commit: str) -> Tuple[bool, str]:
    """Land a prepared merge commit on the LIVE repo. Caller MUST already hold the update
    lock, have stopped workers, and written the rescue + tx markers. The live dirty state
    is preserved in the merge's local-snapshot parent (and the rescue), so it is reset
    away here. Returns (ok, message)."""
    if not merge_commit:
        return False, "no merge_commit to apply"
    _g.git_capture(["git", "reset", "--hard", "HEAD"])
    _g.git_capture(["git", "clean", "-fd"])
    rc1, _o1, e1 = _g.git_capture(["git", "checkout", "-B", branch, merge_commit])
    if rc1 != 0:
        return False, f"checkout -B {branch} {merge_commit[:12]} failed: {e1}"
    rc2, _o2, e2 = _g.git_capture(["git", "reset", "--hard", merge_commit])
    if rc2 != 0:
        return False, f"reset --hard {merge_commit[:12]} failed: {e2}"
    _g.git_capture(["git", "clean", "-fd"])
    return True, f"applied merge {merge_commit[:12]} to {branch}"


def rollback_managed_update(reason: str = "update_rollback") -> Tuple[bool, str]:
    """Roll a failed managed update back to the pre-update SHA in the tx marker. Tags the
    bad candidate as ``failed-update-<sha>`` for forensics, hard-resets the branch to
    pre_update_sha, cleans, clears the update markers, and logs. Does NOT push (unlike
    rollback_to_version, which can push origin — wrong for an internal recovery)."""
    tx = read_update_tx()
    pre = str(tx.get("pre_update_sha") or "")
    branch = str(tx.get("pre_update_branch") or _g.BRANCH_DEV)
    if not pre:
        return False, "no pre_update_sha in update tx marker"
    rc_h, cur_head, _he = _g.git_capture(["git", "rev-parse", "--short", "HEAD"])
    if rc_h == 0 and cur_head:
        _g.git_capture(["git", "branch", "-f", f"failed-update-{cur_head}", "HEAD"])
    _g.git_capture(["git", "reset", "--hard", "HEAD"])
    _g.git_capture(["git", "clean", "-fd"])
    rc1, _o1, e1 = _g.git_capture(["git", "checkout", "-B", branch, pre])
    if rc1 != 0:
        return False, f"rollback checkout -B {branch} {pre[:12]} failed: {e1}"
    _g.git_capture(["git", "reset", "--hard", pre])
    _g.git_capture(["git", "clean", "-fd"])
    clear_update_tx()
    _g._clear_update_intent()
    append_jsonl(
        _g.DRIVE_ROOT / "logs" / "supervisor.jsonl",
        {"ts": utc_now_iso(), "type": "managed_update_rolled_back", "reason": reason,
         "pre_update_sha": pre, "branch": branch},
    )
    return True, f"rolled back to {pre[:12]}"


def _assisted_objective(tx: Dict[str, Any]) -> str:
    target = str(tx.get("target_sha") or "")[:12]
    conflicts = list(tx.get("conflict_paths") or [])
    files = ", ".join(conflicts) if conflicts else "see `git status` for unmerged paths"
    return (
        f"A managed Ouroboros update (target {target}) has been merged into your working tree by the "
        "supervisor: MERGE_HEAD is set and conflicts are marked in the files. Do NOT run any git command "
        "(fetch/merge/commit/checkout are blocked) — the merge is already staged for you. For each "
        f"conflicting file ({files}) reconcile OUR local changes with the official version, preserving both "
        "intents where possible and removing every conflict marker (<<<<<<<, =======, >>>>>>>). Do NOT "
        "weaken BIBLE.md, docs/CHECKLISTS.md, or prompts/SAFETY.md. When every conflict is resolved, run "
        "`advisory_review` with the commit message, then `commit_reviewed` (it will create the reviewed "
        "2-parent merge commit), then `request_restart` to finish landing the update."
    )


def enqueue_assisted_resolution_task(tx: Dict[str, Any]) -> str:
    """Enqueue (front) the single authorized resolution task for an assisted merge and start a
    worker for it. Used by both the apply orchestration and boot recovery so the objective +
    structured metadata stay in one place. Returns the task id."""
    from supervisor.queue import enqueue_task
    from supervisor.workers import spawn_workers

    task_id = str(tx.get("task_id") or "")
    task = {
        "id": task_id,
        "text": _assisted_objective(tx),
        "type": "task",
        "chat_id": int(tx.get("owner_chat_id") or 0),
        "metadata": {
            "managed_update": {
                "target_sha": str(tx.get("target_sha") or ""),
                "conflict_paths": list(tx.get("conflict_paths") or []),
                "local_snapshot": str(tx.get("local_snapshot") or ""),
            }
        },
    }
    enqueue_task(task, front=True)
    try:
        spawn_workers()
    except Exception:
        _g.log.warning("enqueue_assisted_resolution_task: spawn_workers failed", exc_info=True)
    return task_id


def update_restart_smoke() -> Dict[str, Any]:
    """Stronger pre-restart smoke than ``import_test`` for gating an update apply: no
    unmerged index, ``py_compile server.py``, and an import of the core boot surface.
    pytest is intentionally NOT in this blocking gate (bloat/risk in a live self-updater)."""
    if getattr(sys, "frozen", False):
        return {"ok": True, "skipped": "frozen"}
    rc_u, unmerged, _ue = _g.git_capture(["git", "diff", "--name-only", "--diff-filter=U"])
    if rc_u == 0 and unmerged.strip():
        return {"ok": False, "stderr": f"unmerged paths remain: {unmerged}", "returncode": 1}
    compiled = subprocess.run(
        [sys.executable, "-m", "py_compile", "server.py"],
        cwd=str(_g.REPO_DIR), capture_output=True, text=True,
    )
    if compiled.returncode != 0:
        return {"ok": False, "stderr": compiled.stderr, "returncode": compiled.returncode}
    imported = subprocess.run(
        [sys.executable, "-c",
         "import server, ouroboros.gateway.router, supervisor.queue, "
         "supervisor.events, ouroboros.tools.registry; print('smoke_ok')"],
        cwd=str(_g.REPO_DIR), capture_output=True, text=True,
    )
    return {"ok": (imported.returncode == 0), "stdout": imported.stdout,
            "stderr": imported.stderr, "returncode": imported.returncode}


_ASSISTED_BOOT_ATTEMPT_CAP = 3


def _log_supervisor(payload: Dict[str, Any]) -> None:
    append_jsonl(_g.DRIVE_ROOT / "logs" / "supervisor.jsonl", {"ts": utc_now_iso(), **payload})


def _finalize_pending_boot_smoke(tx: Dict[str, Any], supervisor_ready: bool) -> Dict[str, Any]:
    """Health-check a committed-and-restarted update (auto_merge OR a committed assisted
    merge). Pre-restart smoke already ran inline; this is the post-boot backstop + boot-loop
    guard: clear on healthy boot, roll back to pre_update_sha on a genuine miss / brick-loop."""
    attempts = int(tx.get("boot_attempts") or 0) + 1
    merge_commit = str(tx.get("merge_commit") or "")
    rc_h, head, _he = _g.git_capture(["git", "rev-parse", "HEAD"])
    head_resolved = rc_h == 0 and bool(merge_commit)
    merge_in_head = head_resolved and (
        head == merge_commit
        or _g.git_capture(["git", "merge-base", "--is-ancestor", merge_commit, "HEAD"])[0] == 0
    )
    if bool(supervisor_ready) and merge_in_head:
        clear_update_tx()
        _log_supervisor({"type": "managed_update_finalized", "head": head})
        return {"finalized": True}
    if (bool(supervisor_ready) and head_resolved and not merge_in_head) or attempts >= 2:
        ok, msg = rollback_managed_update("post_boot_smoke_failed")
        _log_supervisor({"type": "managed_update_rollback_after_failed_boot",
                         "ok": ok, "msg": msg, "boot_attempts": attempts})
        return {"finalized": False, "rolled_back": ok, "msg": msg}
    tx["boot_attempts"] = attempts
    write_update_tx(tx)
    return {"finalized": False, "boot_attempts": attempts}


def _recover_assisted_on_boot(tx: Dict[str, Any], supervisor_ready: bool) -> Dict[str, Any]:
    """Recover an in-flight assisted merge after a restart/rescue — re-keyed on MERGE STATE
    (during resolution HEAD == pre_update_sha, the reviewed base) and strictly non-destructive:
    a real reviewed commit that landed on top is NEVER reset away."""
    state = _assisted_head_state(tx)
    if state == "committed":
        # Crash after commit before the phase flipped: recover by transitioning to the
        # committed-and-verify path (set merge_commit from HEAD if missing).
        if not str(tx.get("merge_commit") or ""):
            rc_h, head, _he = _g.git_capture(["git", "rev-parse", "HEAD"])
            if rc_h == 0:
                tx["merge_commit"] = head
        tx["phase"] = "pending_boot_smoke"
        write_update_tx(tx)
        _log_supervisor({"type": "managed_update_assisted_committed_recovered",
                         "merge_commit": str(tx.get("merge_commit") or "")[:12]})
        return _finalize_pending_boot_smoke(tx, supervisor_ready)
    if state == "diverged":
        # A real reviewed commit landed; keep it (never reset over reviewed work), abandon
        # this update — it is re-planned fresh later.
        clear_update_tx()
        _log_supervisor({"type": "managed_update_assisted_abandoned_diverged"})
        return {"finalized": False, "abandoned": True, "reason": "head_diverged"}
    if state == "in_progress":
        attempts = int(tx.get("resolution_attempts") or 0) + 1
        if attempts > _ASSISTED_BOOT_ATTEMPT_CAP:
            ok, msg = rollback_managed_update("assisted_resolution_expired")
            _log_supervisor({"type": "managed_update_assisted_expired", "ok": ok, "msg": msg})
            return {"finalized": False, "rolled_back": ok, "msg": msg}
        # Re-establish the merge state if the restart/rescue wiped it; preserve partial
        # progress when MERGE_HEAD + a dirty tree already survived.
        rc_d, dirty, _de = _g.git_capture(["git", "status", "--porcelain"])
        has_progress = bool(_merge_head_sha()) and rc_d == 0 and bool(dirty.strip())
        if not has_progress:
            ok, msg = materialize_assisted_merge_live(
                str(tx.get("pre_update_branch") or _g.BRANCH_DEV),
                str(tx.get("local_snapshot") or ""),
                str(tx.get("target_sha") or ""),
                str(tx.get("pre_update_sha") or ""),
            )
            if not ok:
                # Could not re-stage the merge — fail closed to a clean pre-update state.
                rb_ok, rb_msg = rollback_managed_update("assisted_rematerialize_failed")
                _log_supervisor({"type": "managed_update_assisted_rematerialize_failed",
                                 "materialize_msg": msg, "rollback": rb_msg})
                return {"finalized": False, "rolled_back": rb_ok, "msg": msg}
        tx["phase"] = "assisted_resolution"
        tx["resolution_attempts"] = attempts
        write_update_tx(tx)
        enqueue_assisted_resolution_task(tx)
        _log_supervisor({"type": "managed_update_assisted_resumed",
                         "resolution_attempts": attempts, "preserved_progress": has_progress})
        return {"finalized": False, "resumed": True, "resolution_attempts": attempts}
    # unknown: do not touch the tree; leave the tx for the owner / a later boot.
    _log_supervisor({"type": "managed_update_assisted_unknown_state"})
    return {"finalized": False, "reason": "unknown_assisted_state"}


def managed_assisted_tx_for(task_id: str) -> Tuple[Dict[str, Any], str]:
    """For ``commit_reviewed``: return ``(managed_tx, block_message)``. While a managed assisted
    tx is active, ONLY its authorized resolution task may commit. A CORRUPT marker blocks too
    (fail-closed). Returns ``(tx, "")`` for the authorized task, ``({}, msg)`` to block another
    task, ``({}, "")`` when no managed tx is active. The tx marker — never an LLM value — is the
    trust root for the managed merge."""
    status, tx = read_update_tx_strict()
    if status == "absent":
        return {}, ""
    if status == "valid" and str(tx.get("phase") or "") in _ASSISTED_PHASES:
        if str(tx.get("task_id") or "") == str(task_id or ""):
            return tx, ""
    elif status == "valid":
        return {}, ""  # a non-assisted tx (pending_boot_smoke) does not gate commits
    return {}, (
        "⚠️ MANAGED_UPDATE_IN_PROGRESS: a managed update merge is being resolved by another "
        "task (or the update tx is unreadable); commits are blocked until it completes or is "
        "rolled back."
    )


def managed_assisted_precommit_verify(tx: Dict[str, Any]) -> Tuple[bool, str]:
    """Verify the live merge state matches the tx before the reviewed commit: on the expected
    branch, MERGE_HEAD == tx.target_sha, HEAD == tx.pre_update_sha (the reviewed first parent)."""
    branch = str(tx.get("pre_update_branch") or _g.BRANCH_DEV)
    target = str(tx.get("target_sha") or "")
    pre = str(tx.get("pre_update_sha") or "")
    rc_b, cur, _e = _g.git_capture(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    if rc_b != 0 or cur != branch:
        return False, f"⚠️ MANAGED_UPDATE_ERROR: on branch {cur!r}, expected {branch!r}"
    mh = _merge_head_sha()
    if mh != target:
        return False, f"⚠️ MANAGED_UPDATE_ERROR: MERGE_HEAD {(mh[:12] or 'absent')} != target {target[:12]}"
    rc_h, head, _he = _g.git_capture(["git", "rev-parse", "--verify", "HEAD"])
    if rc_h != 0 or head != pre:
        return False, f"⚠️ MANAGED_UPDATE_ERROR: HEAD {head[:12]} != reviewed base {pre[:12]}"
    return True, ""


def managed_assisted_marker_check() -> Tuple[bool, str]:
    """Reject leftover conflict markers in the STAGED tree — the PRIMARY leakage gate: once the
    agent `git add`-s a marked file it is a 'resolved' (stage-0) entry, so `--diff-filter=U`
    no longer catches it. Scan the raw staged blob (no diff '+' prefix); flag a file only when
    BOTH a `<<<<<<<` and a `>>>>>>>` marker line are present (avoids false-positives on a lone
    markdown `=======` underline)."""
    import re

    start_re = re.compile(r"^<{7}")
    end_re = re.compile(r"^>{7}")
    rc_n, names, _e = _g.git_capture(["git", "diff", "--cached", "--name-only"])
    bad: List[str] = []
    if rc_n == 0:
        for path in [p for p in names.splitlines() if p.strip()]:
            rc_s, blob, _se = _g.git_capture(["git", "show", f":{path}"])
            if rc_s != 0:
                continue
            lines = blob.splitlines()
            if any(start_re.match(ln) for ln in lines) and any(end_re.match(ln) for ln in lines):
                bad.append(path)
    if bad:
        return False, (
            "⚠️ MANAGED_UPDATE_ERROR: unresolved conflict markers remain in: "
            + ", ".join(bad[:20])
            + " — remove every <<<<<<< / ======= / >>>>>>> before committing."
        )
    return True, ""


def reestablish_merge_head(target_sha: str) -> None:
    """Re-write ``.git/MERGE_HEAD`` so a BLOCKED managed-merge review can be fixed and re-committed
    — the review's index reset (``git reset HEAD``) clears the in-progress merge state, after which
    ``managed_assisted_precommit_verify`` would fail on the agent's retry. Best-effort."""
    if not target_sha:
        return
    try:
        (_g._git_dir() / "MERGE_HEAD").write_text(target_sha + "\n", encoding="utf-8")
    except Exception:
        _g.log.warning("reestablish_merge_head failed", exc_info=True)


def managed_assisted_postcommit(tx: Dict[str, Any], commit_sha: str) -> Tuple[bool, str]:
    """After the reviewed 2-parent merge commit lands: record merge_commit + transition to
    ``pending_boot_smoke``, then run the pre-restart smoke INLINE (auto_merge parity). On smoke
    FAIL roll back to pre_update_sha (the agent's resolution survives on the failed-update tag +
    the rescue-local ref). On PASS the agent calls ``request_restart`` and boot finalize verifies
    the healthy boot. Returns (ok, message)."""
    tx = dict(tx)
    tx["phase"] = "pending_boot_smoke"
    tx["merge_commit"] = commit_sha
    write_update_tx(tx)
    smoke = update_restart_smoke()
    if smoke.get("ok"):
        return True, (
            "✅ Managed update committed as a reviewed 2-parent merge and passed the pre-restart "
            "smoke. Call `request_restart` now to finish landing the update."
        )
    ok, msg = rollback_managed_update("assisted_pre_restart_smoke_failed")
    # Preserve the FULL smoke trace durably (it explains why a self-modifying update rolled
    # back — never silently sliced); the chat message shows a head with an explicit omission note.
    _log_supervisor({
        "type": "managed_update_assisted_smoke_failed", "returncode": smoke.get("returncode"),
        "stdout": str(smoke.get("stdout") or ""), "stderr": str(smoke.get("stderr") or ""),
    })
    stderr = str(smoke.get("stderr") or "")
    shown = stderr if len(stderr) <= 400 else (
        stderr[:400] + f"… (+{len(stderr) - 400} more chars — full trace in data/logs/supervisor.jsonl)"
    )
    return False, (
        "⚠️ MANAGED_UPDATE_SMOKE_FAILED: the merged code failed the pre-restart smoke "
        f"({shown}). Rolled back to the prior version ({msg}). "
        "The resolved merge is preserved on a failed-update-* tag for inspection."
    )


def abort_orphaned_assisted_tx(task_id: str) -> Dict[str, Any]:
    """Watchdog called when a task ENDS: if it was the authorized assisted-resolution task and
    the tx is still mid-resolution (the merge never committed — failed / cancelled / gave up),
    roll back to pre_update_sha so the live worktree AND the commit-exclusivity guard are freed
    immediately (no starvation until a restart). A COMMITTED merge (phase pending_boot_smoke) or
    an in-flight commit (committing_assisted) is left for the restart / boot finalize."""
    status, tx = read_update_tx_strict()
    if status != "valid" or str(tx.get("phase") or "") not in ("materializing_assisted", "assisted_resolution"):
        return {"acted": False}
    if str(tx.get("task_id") or "") != str(task_id or ""):
        return {"acted": False}
    lock_fh = None
    try:
        try:
            lock_fh = acquire_update_lock()
        except RuntimeError:
            return {"acted": False, "reason": "lock held by an active apply"}
        s2, tx2 = read_update_tx_strict()  # re-read under the lock (it may have just committed)
        if s2 != "valid" or str(tx2.get("phase") or "") not in ("materializing_assisted", "assisted_resolution"):
            return {"acted": False}
        if str(tx2.get("task_id") or "") != str(task_id or ""):
            return {"acted": False}
        ok, msg = rollback_managed_update("assisted_resolution_orphaned")
        _log_supervisor({"type": "managed_update_assisted_orphaned_rollback", "ok": ok, "msg": msg})
        try:
            from supervisor.workers import spawn_workers

            spawn_workers()
        except Exception:
            _g.log.warning("abort_orphaned_assisted_tx: spawn_workers failed", exc_info=True)
        return {"acted": True, "rolled_back": ok, "msg": msg}
    finally:
        if lock_fh is not None:
            release_update_lock(lock_fh)


def finalize_managed_update_on_boot(supervisor_ready: bool = True) -> Dict[str, Any]:
    """Post-boot finalization of a managed update (P2). Called ONCE after the new process
    boots and the supervisor is ready. Acquires the update lock (skips if an apply holds it),
    strict-reads the tx, and dispatches by phase: ``pending_boot_smoke`` (committed +
    restarted) → health-check + boot-loop guard; an assisted phase → non-destructive
    merge-state recovery (resume / abandon-on-divergence / rollback-on-expiry). A CORRUPT
    marker fails closed (left for the owner). Best-effort; never raises."""
    lock_fh = None
    try:
        try:
            lock_fh = acquire_update_lock()
        except RuntimeError:
            return {"finalized": False, "reason": "update lock held by an active apply"}
        status, tx = read_update_tx_strict()
        if status == "absent":
            return {"finalized": False, "reason": "no pending update"}
        if status == "corrupt":
            _log_supervisor({"type": "managed_update_tx_corrupt_on_boot"})
            return {"finalized": False, "reason": "corrupt tx marker — left for owner"}
        phase = str(tx.get("phase") or "")
        if phase == "pending_boot_smoke":
            return _finalize_pending_boot_smoke(tx, supervisor_ready)
        if phase in _ASSISTED_PHASES:
            return _recover_assisted_on_boot(tx, supervisor_ready)
        return {"finalized": False, "reason": f"unhandled phase {phase}"}
    except Exception:
        _g.log.warning("finalize_managed_update_on_boot failed", exc_info=True)
        return {"finalized": False, "error": "exception"}
    finally:
        if lock_fh is not None:
            release_update_lock(lock_fh)
