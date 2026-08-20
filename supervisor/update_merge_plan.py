"""Managed-update merge planning and live materialization (P2), split out of
``supervisor/update_merge.py`` (module-size discipline).

Owns the isolated temp-worktree dry-run planner, the durable clean-plan merge
commit builder, and the live assisted-merge materializer. The parent keeps the
tx marker, lock, stash, rollback and boot-recovery primitives and re-exports
every name here, so ``supervisor.update_merge`` stays the one public surface.
Parent members that tests rebind on the parent module are read through the
call-time handle ``_um()`` — never a from-import, which would freeze the
binding this module saw at import time.
"""

from __future__ import annotations

import os
import subprocess
from typing import Any, Dict, List, Optional, Tuple

from supervisor import git_ops as _g
from supervisor.update_carriers import resolve_carrier_conflicts


def _um():
    """The parent module, read at call time.

    ``supervisor.update_merge`` owns ``managed_update_constitution_present`` and
    ``_merge_head_sha``, and tests monkeypatch them on the parent. Reading them
    through the module keeps one binding; a from-import here would freeze the
    value this module saw at import time.
    """
    from supervisor import update_merge

    return update_merge


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


def _build_clean_merge_commit(
    tmp_wt: str,
    base_sha: str,
    target_sha: str,
    *,
    fast_forwardable: bool,
    local_dirty_count: int,
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Build the durable merge commit for a CLEAN plan inside the temp worktree.

    Owner decision (2026-08, Q1=C): local dirty work NEVER enters committed
    history on a clean auto-update. The commit merges the reviewed HEAD (base)
    and the official target only; the apply path stashes dirty work and
    restores it as uncommitted content after the update. clean(snapshot,
    target) implies clean(base, target) for ordinary hunk overlaps, but
    file/directory-type collisions CAN break the implication — a conflicting
    base re-merge is returned as ``{"base_conflicts": [...]}`` so the caller
    routes it to the assisted lane. Returns (merge_commit, failure|None)."""
    if local_dirty_count:
        if fast_forwardable:
            # Base is an ancestor of the target: pure official history,
            # no merge commit needed at all.
            return target_sha, None
        rc_r, _ro, reset_error = _git_run(["git", "-C", tmp_wt, "reset", "--hard", base_sha])
        if rc_r != 0:
            return "", {"error": reset_error or "could not reset plan worktree to base"}
        rc_bm, _bo, base_merge_error = _git_run(
            ["git", "-C", tmp_wt, "merge", "--no-commit", "--no-ff", target_sha]
        )
        if rc_bm == 1:
            rc_u, unmerged_out, _ue = _git_run(
                ["git", "-C", tmp_wt, "diff", "--name-only", "--diff-filter=U"]
            )
            base_conflicts = (
                [ln.strip() for ln in unmerged_out.splitlines() if ln.strip()]
                if rc_u == 0 else []
            )
            if not base_conflicts:
                return "", {"error": base_merge_error or "base merge failed without an inventory"}
            # Carrier engine insertion point 2 of 3 (spec §1.9-10, owner batch №8
            # answer 6=A): the base re-merge, applied BEFORE write-tree. A base
            # conflict confined to declared version-carrier spans adopts the
            # official side of the span and stays on the clean path; anything
            # else routes to the assisted lane exactly as before.
            resolution = resolve_carrier_conflicts(tmp_wt, base_conflicts, prefer="theirs")
            carrier_resolved = set(resolution["resolved"])
            remaining = [path for path in base_conflicts if path not in carrier_resolved]
            if remaining:
                return "", {"base_conflicts": remaining}
        elif rc_bm != 0:
            return "", {"error": base_merge_error or "clean base merge unexpectedly failed"}
    rc_mt, merged_tree, _mte = _git_run(["git", "-C", tmp_wt, "write-tree"])
    if rc_mt != 0 or not merged_tree:
        return "", {"error": "could not build merged tree"}
    rc_mc, built, commit_error = _git_run([
        "git", "commit-tree", merged_tree,
        "-p", base_sha, "-p", target_sha,
        "-m", f"Merge official Ouroboros update {target_sha[:12]} (auto)",
    ])
    if rc_mc != 0 or not built:
        return "", {"error": commit_error or "could not build merge commit"}
    return built, None


def plan_managed_update_merge(
    fetch: bool = False, branch: Optional[str] = None, build: bool = False
) -> Dict[str, Any]:
    """Dry-run the managed update as a REAL 3-way merge in an ISOLATED temp worktree and
    classify the result (P2). NEVER touches the live worktree or index. Returns a
    ``merge_plan`` dict: available/kind/auto_mergeable, the doc/code conflict labels,
    target_sha/base_sha, local_dirty_count, recommended_strategy. Best-effort:
    always cleans up the temp index + worktree; classification uses update_merge_policy.

    When ``build=True`` AND the merge is clean, the merged tree is committed as a real
    merge commit (parents = [reviewed HEAD, target]; a fast-forwardable base lands the
    official target itself) whose sha is returned as ``merge_commit`` — a durable
    object in the shared DB that survives temp-worktree removal, ready for
    ``apply_managed_merge_update`` to land on the live repo. Dirty local work is used
    only to CLASSIFY conflicts (via the synthetic snapshot); it never enters the built
    commit — the apply path stashes and restores it (owner decision Q1=C)."""
    import shutil
    import tempfile

    from ouroboros.update_channels import get_update_channel
    from supervisor.update_merge_policy import classify_conflicts
    branch_dev = branch or _g.BRANCH_DEV
    remote_name, remote_branch, branch_ref = _g._managed_update_target()
    update_channel = get_update_channel()
    identity = {
        "remote": remote_name,
        "remote_branch": remote_branch,
        "target_ref": branch_ref,
        "update_channel": update_channel,
    }
    rc_b, current_branch, branch_error = _g.git_capture(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"]
    )
    if rc_b != 0 or current_branch != branch_dev:
        return {
            "available": False,
            "kind": "unavailable",
            "error": branch_error or f"managed update requires local branch {branch_dev}",
            "current_branch": current_branch if rc_b == 0 else "unknown",
            **identity,
        }
    if not branch_ref:
        return {
            "available": False,
            "kind": "unavailable",
            "error": "no managed update remote",
            **identity,
        }
    if fetch and remote_name:
        remote_ok, remote_error = _g.ensure_official_update_remote()
        if not remote_ok:
            return {
                "available": False,
                "kind": "unavailable",
                "error": remote_error or "could not configure official update remote",
                **identity,
            }
        fetch_rc, _fetch_out, fetch_error = _g.git_fetch_bounded(remote_name)
        if fetch_rc != 0:
            return {
                "available": False,
                "kind": "unavailable",
                "error": fetch_error or f"git fetch {remote_name} failed",
                **identity,
            }

    target_ref, target_sha, target_error = _g._resolve_managed_update_target(
        remote_name, remote_branch, branch_ref, update_channel
    )
    identity["target_ref"] = target_ref or branch_ref
    if not target_ref or not target_sha:
        return {
            "available": False,
            "kind": "unavailable",
            "error": target_error or "could not resolve managed update target",
            **identity,
        }

    rc_h, base_sha, head_error = _g.git_capture(
        ["git", "rev-parse", "--verify", "HEAD"]
    )
    pins = {"target_sha": target_sha, "base_sha": base_sha, **identity}
    if rc_h != 0 or not base_sha:
        return {
            "available": False,
            "kind": "unavailable",
            "error": target_error or head_error or "could not resolve target/HEAD",
            **pins,
        }
    if not _um().managed_update_constitution_present(target_sha):
        return {
            "available": False,
            "kind": "unavailable",
            "error": "official update target does not preserve BIBLE.md",
            **pins,
        }
    status_rc, dirty_out, status_error = _g.git_capture(["git", "status", "--porcelain"])
    if status_rc != 0:
        return {
            "available": target_sha != base_sha,
            "kind": "unknown",
            "error": status_error or "git status failed",
            **pins,
        }
    local_dirty_count = len([ln for ln in dirty_out.splitlines() if ln.strip()])
    pins["local_dirty_count"] = local_dirty_count
    if target_sha == base_sha:
        return {"available": False, "kind": "current", **pins}
    ancestor_rc, _ancestor_out, ancestor_error = _g.git_capture(
        ["git", "merge-base", "--is-ancestor", target_sha, base_sha]
    )
    if ancestor_rc == 0:
        return {"available": False, "kind": "current", **pins}
    if ancestor_rc not in (0, 1):
        return {
            "available": False,
            "kind": "unknown",
            "error": ancestor_error or "could not compare target with HEAD",
            **pins,
        }

    fast_forward_rc, _ff_out, fast_forward_error = _g.git_capture(
        ["git", "merge-base", "--is-ancestor", base_sha, target_sha]
    )
    if fast_forward_rc not in (0, 1):
        return {
            "available": True,
            "kind": "unknown",
            "error": fast_forward_error or "could not compare HEAD with target",
            **pins,
        }
    if fast_forward_rc == 0 and local_dirty_count == 0:
        return {
            "available": True,
            "kind": "clean",
            "auto_mergeable": True,
            "doc_conflict_paths": [],
            "code_conflict_paths": [],
            "hot_code_paths": [],
            "local_snapshot": base_sha,
            "merge_commit": target_sha if build else "",
            "carrier_resolved_paths": [],
            "recommended_strategy": "auto_merge",
            **pins,
        }

    tmp_index_path = None
    tmp_wt = None
    try:
        # A clean, diverged branch can merge directly from HEAD. A synthetic
        # snapshot commit is needed only when it is the sole durable carrier of
        # dirty/untracked local work.
        local_snapshot = base_sha
        if local_dirty_count:
            fd, tmp_index_path = tempfile.mkstemp(prefix="ouro-update-index-")
            os.close(fd)
            # `git read-tree` wants a NON-existent index path.
            os.unlink(tmp_index_path)
            env = {"GIT_INDEX_FILE": tmp_index_path}
            if _git_run(["git", "read-tree", "HEAD"], extra_env=env)[0] != 0:
                return {"available": True, "kind": "unknown", "error": "read-tree failed", **pins}
            add_rc, _add_out, add_error = _git_run(["git", "add", "-A"], extra_env=env)
            if add_rc != 0:
                return {
                    "available": True,
                    "kind": "unknown",
                    "error": add_error or "git add -A failed",
                    **pins,
                }
            rc_wt, local_tree, _we = _git_run(["git", "write-tree"], extra_env=env)
            if rc_wt != 0 or not local_tree:
                return {"available": True, "kind": "unknown", "error": "write-tree failed", **pins}
            rc_ct, local_snapshot, _ce = _git_run(
                ["git", "commit-tree", local_tree, "-p", base_sha,
                 "-m", "ouroboros local snapshot (update merge plan)"],
                extra_env=env,
            )
            if rc_ct != 0 or not local_snapshot:
                return {"available": True, "kind": "unknown", "error": "commit-tree failed", **pins}

        # 2. Isolated temp worktree at the snapshot; merge the target THERE (never live).
        #    Use a NON-existent child path (git worktree add refuses an existing dir).
        tmp_wt = os.path.join(tempfile.mkdtemp(prefix="ouro-update-wt-"), "wt")
        rc_add, _ao, add_err = _g.git_capture(["git", "worktree", "add", "--detach", tmp_wt, local_snapshot])
        if rc_add != 0:
            return {
                "available": True,
                "kind": "unknown",
                "error": f"worktree add failed: {add_err}",
                **pins,
            }
        # --no-commit --no-ff: leave the merged/conflicted index in place to inspect.
        merge_rc, _merge_out, merge_error = _git_run(
            ["git", "-C", tmp_wt, "merge", "--no-commit", "--no-ff", target_sha]
        )
        if merge_rc not in (0, 1):
            return {
                "available": True,
                "kind": "unknown",
                "error": merge_error or f"git merge failed with exit {merge_rc}",
                **pins,
            }
        rc_u, unmerged_out, unmerged_error = _git_run(
            ["git", "-C", tmp_wt, "diff", "--name-only", "--diff-filter=U"]
        )
        if rc_u != 0:
            return {
                "available": True,
                "kind": "unknown",
                "error": unmerged_error or "could not inspect merge conflicts",
                **pins,
            }
        unmerged = [ln.strip() for ln in unmerged_out.splitlines() if ln.strip()]
        if (merge_rc == 0 and unmerged) or (merge_rc == 1 and not unmerged):
            return {
                "available": True,
                "kind": "unknown",
                "error": "git merge result and conflict inventory disagree",
                **pins,
            }

        # Carrier engine insertion point 1 of 3 (spec §1.9-10, owner batch №8
        # answer 6=A): the planner merge, applied BEFORE write-tree. Conflicts
        # confined to declared version-carrier spans adopt the official side of
        # the span (staged in the ISOLATED temp worktree) and leave the plan's
        # conflict inventory; every other conflict classifies exactly as before.
        carrier_resolved: List[str] = []
        if unmerged:
            resolution = resolve_carrier_conflicts(tmp_wt, unmerged, prefer="theirs")
            carrier_resolved = list(resolution["resolved"])
            if carrier_resolved:
                unmerged = [path for path in unmerged if path not in set(carrier_resolved)]

        plan = classify_conflicts(unmerged)
        kind = str(plan["kind"])
        merge_commit = ""
        if build and kind == "clean":
            built, failure = _build_clean_merge_commit(
                tmp_wt, base_sha, target_sha,
                fast_forwardable=(fast_forward_rc == 0),
                local_dirty_count=local_dirty_count,
            )
            if failure is not None:
                if failure.get("base_conflicts"):
                    # Exotic but real (e.g. file/directory collisions): the local
                    # snapshot merged cleanly while the committed base does not.
                    # Route to the assisted lane with the BASE conflict inventory
                    # instead of refusing forever with kind=unknown.
                    base_plan = classify_conflicts(failure["base_conflicts"])
                    return {
                        "available": True,
                        "kind": base_plan["kind"] if base_plan["kind"] != "clean" else "unknown",
                        "auto_mergeable": False,
                        "doc_conflict_paths": base_plan["doc_conflict_paths"],
                        "code_conflict_paths": base_plan["code_conflict_paths"],
                        "hot_code_paths": base_plan["hot_code_paths"],
                        "local_dirty_count": local_dirty_count,
                        "local_snapshot": local_snapshot,
                        "merge_commit": "",
                        "carrier_resolved_paths": carrier_resolved,
                        "recommended_strategy": "assisted",
                        **pins,
                    }
                return {"available": True, "kind": "unknown", **pins,
                        "error": failure.get("error") or "could not build merge commit"}
            merge_commit = built
        return {
            "available": True,
            "kind": kind,
            "auto_mergeable": kind == "clean",
            "doc_conflict_paths": plan["doc_conflict_paths"],
            "code_conflict_paths": plan["code_conflict_paths"],
            "hot_code_paths": plan["hot_code_paths"],
            "local_dirty_count": local_dirty_count,
            "local_snapshot": local_snapshot,
            "merge_commit": merge_commit,
            "carrier_resolved_paths": carrier_resolved,
            # Git owns clean merges. Ouroboros is needed only for a real conflict.
            "recommended_strategy": "auto_merge" if kind == "clean" else "assisted",
            **pins,
        }
    except Exception as exc:  # pragma: no cover — planning is best-effort
        _g.log.warning("plan_managed_update_merge failed", exc_info=True)
        return {
            "available": True,
            "kind": "unknown",
            "error": f"{type(exc).__name__}: {exc}",
            **pins,
        }
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
    rc_reset, _ro, reset_error = _g.git_capture(["git", "reset", "--hard", "HEAD"])
    if rc_reset != 0:
        return False, f"could not clean tracked files before assisted merge: {reset_error}"
    rc_clean, _co, clean_error = _g.git_capture(["git", "clean", "-fd"])
    if rc_clean != 0:
        return False, f"could not clean untracked files before assisted merge: {clean_error}"
    rc_c, _o, e_c = _g.git_capture(["git", "checkout", "-B", branch, local_snapshot])
    if rc_c != 0:
        return False, f"checkout -B {branch} {local_snapshot[:12]} failed: {e_c}"
    # Ignore the merge return code; conflicts are expected. Judge by MERGE_HEAD.
    rc_m, _mo, merge_error = _g.git_capture(
        ["git", "merge", "--no-commit", "--no-ff", target_sha]
    )
    if rc_m not in (0, 1):
        return False, f"merge failed before conflict resolution: {merge_error or rc_m}"
    mh = _um()._merge_head_sha()
    if not mh:
        return False, "merge produced no MERGE_HEAD (nothing to merge or fatal error)"
    if mh != target_sha:
        return False, f"MERGE_HEAD {mh[:12]} != target {target_sha[:12]}"
    # Carrier engine insertion point 3 of 3 (spec §1.9-10, owner batch №8
    # answer 6=A): the live materializer. Version-carrier spans in the staged
    # merge adopt the official side so the assisted resolver only faces real
    # conflicts; best-effort — whatever stays unresolved remains for the
    # assisted lane exactly as before.
    rc_cu, carrier_unmerged_out, _cue = _g.git_capture(
        ["git", "diff", "--name-only", "--diff-filter=U"]
    )
    if rc_cu == 0:
        carrier_conflicted = [
            ln.strip() for ln in carrier_unmerged_out.splitlines() if ln.strip()
        ]
        if carrier_conflicted:
            resolve_carrier_conflicts(
                str(_g.REPO_DIR), carrier_conflicted, prefer="theirs"
            )
    # Re-base the first parent to the reviewed pre-update state WITHOUT disturbing the merge
    # result: `git reset --soft` is refused mid-merge, so move the branch ref directly with
    # update-ref (HEAD follows the symbolic ref) — the index (conflicted/merged entries), the
    # worktree, and MERGE_HEAD are all untouched, so commit_reviewed still makes a 2-parent
    # commit [pre_update_sha, target] whose reviewed diff (pre_update_sha → resolved) includes
    # the owner's dirty/untracked work.
    rc_r, _ro, e_r = _g.git_capture(["git", "update-ref", f"refs/heads/{branch}", pre_update_sha])
    if rc_r != 0:
        return False, f"update-ref {branch} -> {pre_update_sha[:12]} failed: {e_r}"
    if _um()._merge_head_sha() != target_sha:
        return False, "MERGE_HEAD lost after re-parenting the branch"
    return True, f"materialized merge of {target_sha[:12]} (parent={pre_update_sha[:12]}, MERGE_HEAD set)"
