"""Managed-update merge planning and live materialization (P2), split out of
``supervisor/update_merge.py`` (module-size discipline; the split is re-cut
from the update-flow redesign's two-module form, not the pre-redesign shape).

Owns the isolated temp-worktree dry-run planner, the durable clean-plan merge
commit builder, and the live assisted-merge materializer — the three insertion
points of the carrier-aware span resolver (``supervisor/update_carriers.py``,
owner-ratified spec §1.9-10 / v7next answers 5.12-5.14=A). The parent keeps the
tx marker, lock, rollback and boot-recovery primitives and re-exports every
name here, so ``supervisor.update_merge`` stays the one public surface.

Binding discipline (the module-handle rule the git_ops split pinned): candidate
primitives are read through the ``supervisor.update_candidate`` module object
(``_uc.X``) and the parent's own members through the call-time ``_um()`` handle
— never from-imports, which would freeze the binding this module saw at import
time and silently kill the test surface that monkeypatches those names on their
owner modules (e.g. ``update_merge.managed_update_constitution_present``,
``update_candidate.worktree_snapshot_tree``)."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from supervisor import git_ops as _g
from supervisor import update_candidate as _uc
from supervisor.update_carriers import resolve_carrier_conflicts


def _um():
    """The parent module, read at call time.

    ``supervisor.update_merge`` owns ``managed_update_constitution_present``
    and tests monkeypatch it on the parent. Reading it through the module keeps
    one binding; a from-import here would freeze the value this module saw at
    import time."""
    from supervisor import update_merge

    return update_merge


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
        rc_r, _ro, reset_error = _uc._git_run(["git", "-C", tmp_wt, "reset", "--hard", base_sha])
        if rc_r != 0:
            return "", {"error": reset_error or "could not reset plan worktree to base"}
        rc_bm, _bo, base_merge_error = _uc._git_run(
            ["git", *_uc._MERGE_NEUTRAL_FLAGS, "-C", tmp_wt, "merge", "--no-commit", "--no-ff", target_sha]
        )
        if rc_bm == 1:
            rc_u, unmerged_out, _ue = _uc._git_run(
                ["git", "-C", tmp_wt, "diff", "--name-only", "--diff-filter=U"]
            )
            base_conflicts = (
                [ln.strip() for ln in unmerged_out.splitlines() if ln.strip()]
                if rc_u == 0 else []
            )
            if not base_conflicts:
                return "", {"error": base_merge_error or "base merge failed without an inventory"}
            # Carrier engine insertion point 2 of 3 (spec §1.9-10, owner batch
            # №8 answer 6=A): the base re-merge, applied BEFORE write-tree. A
            # base conflict confined to declared version-carrier spans adopts
            # the official side of the span and stays on the clean path;
            # anything else routes to the assisted lane exactly as before.
            resolution = resolve_carrier_conflicts(tmp_wt, base_conflicts, prefer="theirs")
            carrier_resolved = set(resolution["resolved"])
            remaining = [path for path in base_conflicts if path not in carrier_resolved]
            if remaining:
                return "", {"base_conflicts": remaining}
        elif rc_bm != 0:
            return "", {"error": base_merge_error or "clean base merge unexpectedly failed"}
    # Q8 is unconditional on BOTH lanes: project VERSION + mechanical carrier
    # tokens inside the temp worktree before serializing the merge commit, so a
    # clean divergence can never ship the fork's version token. Typed failure —
    # never a silently half-projected commit.
    ok_p, _p_note, p_error = _uc.project_version_carriers(target_sha, cwd=tmp_wt)
    if not ok_p:
        return "", {"error": f"carrier projection failed: {p_error}"}
    rc_mt, merged_tree, _mte = _uc._git_run(["git", "-C", tmp_wt, "write-tree"])
    if rc_mt != 0 or not merged_tree:
        return "", {"error": "could not build merged tree"}
    rc_mc, built, commit_error = _uc._git_run([
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

    tmp_wt = None
    try:
        # A clean, diverged branch can merge directly from HEAD. A synthetic
        # snapshot commit is needed only when it is the sole durable carrier of
        # dirty/untracked local work (the informational/preview path; the apply
        # path stashes first and re-plans from a clean tree).
        local_snapshot = base_sha
        if local_dirty_count:
            local_tree, snapshot_error = _uc.worktree_snapshot_tree("HEAD")
            if not local_tree:
                return {
                    "available": True,
                    "kind": "unknown",
                    "error": snapshot_error or "worktree snapshot failed",
                    **pins,
                }
            rc_ct, local_snapshot, _ce = _uc._git_run(
                ["git", "commit-tree", local_tree, "-p", base_sha,
                 "-m", "ouroboros local snapshot (update merge plan)"],
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
        merge_rc, _merge_out, merge_error = _uc._git_run(
            ["git", *_uc._MERGE_NEUTRAL_FLAGS, "-C", tmp_wt, "merge", "--no-commit", "--no-ff", target_sha]
        )
        if merge_rc not in (0, 1):
            return {
                "available": True,
                "kind": "unknown",
                "error": merge_error or f"git merge failed with exit {merge_rc}",
                **pins,
            }
        rc_u, unmerged_out, unmerged_error = _uc._git_run(
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
        # answer 6=A): the planner merge, applied BEFORE classification and
        # write-tree — the same body serves the preview plan AND the stash-first
        # authoritative build=True replan. Conflicts confined to declared
        # version-carrier spans adopt the official side of the span (staged in
        # the ISOLATED temp worktree) and leave the plan's conflict inventory;
        # every other conflict classifies exactly as before.
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


def materialize_assisted_merge_live(
    branch: str, local_snapshot: str, target_sha: str, pre_update_sha: str
) -> Tuple[bool, str, str]:
    """Stage a REAL ``git merge --no-commit --no-ff target`` into the LIVE worktree (MERGE_HEAD +
    a conflicted index + markers) for the agent to resolve and the unmodified ``commit_reviewed``
    to finalize as a reviewed 2-parent commit. Caller MUST hold the update lock with workers
    stopped. Conflicts make ``git merge`` exit nonzero — that is EXPECTED, not failure: success is
    judged by MERGE_HEAD == target_sha. Returns ``(ok, message, m0_tree)`` where ``m0_tree`` is
    the pinned MECHANICAL MERGE BASELINE — the just-materialized worktree's tree (conflict
    markers as content) captured ONCE, before any resolver edit; reviewers diff m0_tree →
    candidate, and a later re-merge (rerere, config drift) is never authority. The carrier-span
    resolution below is applied BEFORE the M0 pin (owner decision Ф-2=A): span policy is part
    of the mechanical baseline, so the resolver and reviewers only face real conflicts.

    Since the stash-first apply order (Q9) ``local_snapshot`` normally equals ``pre_update_sha``
    (uncommitted work rides a stash). Legacy transactions whose snapshot still carries dirty
    work keep working: the merge is computed FROM ``local_snapshot``, then the first parent is
    re-based to ``pre_update_sha`` (the last REVIEWED committed state), so the reviewed diff
    includes that work — none of it reaches history as an unreviewed parent."""
    if not local_snapshot or not target_sha or not pre_update_sha:
        return False, "missing local_snapshot/target_sha/pre_update_sha", ""
    # Clean the worktree first (dirty + untracked are all captured in the stash — legacy: in
    # local_snapshot — plus the rescue snapshot) so `checkout -B` cannot fail on "untracked file
    # would be overwritten". A real 3-way merge needs a clean tree to run.
    rc_reset, _ro, reset_error = _g.git_capture(["git", "reset", "--hard", "HEAD"])
    if rc_reset != 0:
        return False, f"could not clean tracked files before assisted merge: {reset_error}", ""
    rc_clean, _co, clean_error = _g.git_capture(["git", "clean", "-fd"])
    if rc_clean != 0:
        return False, f"could not clean untracked files before assisted merge: {clean_error}", ""
    rc_c, _o, e_c = _g.git_capture(["git", "checkout", "-B", branch, local_snapshot])
    if rc_c != 0:
        return False, f"checkout -B {branch} {local_snapshot[:12]} failed: {e_c}", ""
    # Ignore the merge return code; conflicts are expected. Judge by MERGE_HEAD.
    rc_m, _mo, merge_error = _g.git_capture(
        ["git", *_uc._MERGE_NEUTRAL_FLAGS, "merge", "--no-commit", "--no-ff", target_sha]
    )
    if rc_m not in (0, 1):
        return False, f"merge failed before conflict resolution: {merge_error or rc_m}", ""
    mh = _uc._merge_head_sha()
    if not mh:
        return False, "merge produced no MERGE_HEAD (nothing to merge or fatal error)", ""
    if mh != target_sha:
        return False, f"MERGE_HEAD {mh[:12]} != target {target_sha[:12]}", ""
    # Carrier engine insertion point 3 of 3 (spec §1.9-10, owner batch №8
    # answer 6=A): the live materializer, applied BEFORE the Q8 projection and
    # the M0 pin (Ф-2=A). Version-carrier spans in the staged merge adopt the
    # official side so the assisted resolver only faces real conflicts;
    # best-effort — whatever stays unresolved remains for the assisted lane
    # exactly as before.
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
    # commit [pre_update_sha, target].
    # P9 projection (Q8, shared typed helper): VERSION := target, mechanical
    # carrier tokens synced for non-conflicted files. MANDATORY: a failed
    # projection aborts materialization — a half-projected tree must never be
    # frozen as the M0 baseline (the caller rolls back and the owner retries).
    ok_p, projected_note, projection_error = _uc.project_version_carriers(target_sha)
    if not ok_p:
        return False, f"carrier projection failed: {projection_error}", ""
    # CAS: expected-old = the snapshot we just checked out; a concurrently moved
    # ref (late human commit) must fail the re-parent instead of being clobbered.
    rc_r, _ro, e_r = _g.git_capture(
        ["git", "update-ref", f"refs/heads/{branch}", pre_update_sha, local_snapshot]
    )
    if rc_r != 0:
        return False, f"update-ref {branch} -> {pre_update_sha[:12]} failed: {e_r}", ""
    if _uc._merge_head_sha() != target_sha:
        return False, "MERGE_HEAD lost after re-parenting the branch", ""
    m0_tree, m0_error = _uc.worktree_snapshot_tree(pre_update_sha)
    if not m0_tree:
        return False, f"could not pin the mechanical merge baseline (M0): {m0_error}", ""
    return (
        True,
        f"materialized merge of {target_sha[:12]} (parent={pre_update_sha[:12]}, "
        f"MERGE_HEAD set, M0 {m0_tree[:12]}{projected_note})",
        m0_tree,
    )
