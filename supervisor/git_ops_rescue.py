"""Rescue/snapshot machinery for destructive tree movement, split out of
``supervisor/git_ops.py`` (module-size discipline, v7 G1 split).

Owns the repo sync-state probe and the rescue snapshot: porcelain status, the
binary diff, the stash-created rescue ref, copied untracked files with
completeness metadata, merge topology, the evolution-transaction link, and the
pre-destructive rescue hooks the managed-update rollback path calls. The
parent keeps the rebindable module state (``init`` REBINDS
REPO_DIR/DRIVE_ROOT/BRANCH_* and friends), the capture plumbing and the
marker/meta probes, and re-exports every name here, so ``supervisor.git_ops``
stays the one public surface. Parent members and rebindable globals are read
through the call-time handle ``_go()`` — never a from-import, which would
freeze the binding this module saw at import time.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import pathlib
import shutil
import uuid
from typing import Any, Dict



def _go():
    """The parent module, read at call time.

    ``supervisor.git_ops`` owns the rebindable module state (``init`` REBINDS
    REPO_DIR, DRIVE_ROOT and BRANCH_*) and the helpers tests monkeypatch on
    the parent (``rescue_git_capture``, ``append_jsonl``, the sibling
    re-exports). Reading them through the module keeps one binding: a
    from-import here would freeze the value this module saw at import time.
    """
    from supervisor import git_ops

    return git_ops


# The parent's logger name is pinned so moved log records keep their `%(name)s`
# in server.log/stdout — the same logger object the parent binds.
log = logging.getLogger("supervisor.git_ops")


def _collect_repo_sync_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "current_branch": "unknown",
        "dirty_lines": [],
        "unpushed_lines": [],
        "warnings": [],
    }

    rc, branch, err = _go().rescue_git_capture(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    if rc == 0 and branch:
        state["current_branch"] = branch
    elif err:
        state["warnings"].append(f"branch_error:{err}")

    rc, dirty, err = _go().rescue_git_capture(["git", "status", "--porcelain"])
    if rc == 0 and dirty:
        state["dirty_lines"] = [ln for ln in dirty.splitlines() if ln.strip()]
    elif rc != 0:
        detail = err or f"git status exited {rc} without stderr"
        state["warnings"].append(f"status_error:{detail}")

    remotes = set(_go()._list_remotes(
        capture=_go().rescue_git_capture,
        warnings=state["warnings"],
    ))
    upstream = ""
    current_branch = str(state.get("current_branch") or "")
    managed_meta = _go()._read_managed_repo_meta()
    if managed_meta and current_branch not in ("", "HEAD", "unknown"):
        managed_remote = _go()._managed_remote_name(managed_meta)
        managed_branch = _go()._managed_remote_branch_for(current_branch, managed_meta)
        if managed_branch and managed_remote in remotes:
            upstream = f"{managed_remote}/{managed_branch}"

    if not upstream and "origin" in remotes:
        rc, up, err = _go().rescue_git_capture(["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"])
        if rc == 0 and up:
            upstream = up
        else:
            if current_branch not in ("", "HEAD", "unknown"):
                upstream = f"origin/{current_branch}"
            elif err:
                state["warnings"].append(f"upstream_error:{err}")

    if upstream:
        rc, unpushed, err = _go().rescue_git_capture(["git", "log", "--oneline", f"{upstream}..HEAD"])
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
    rc, txt, err = _go().rescue_git_capture(["git", "ls-files", "--others", "--exclude-standard"])
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
        src = (_go().REPO_DIR / rel).resolve()
        try:
            src.relative_to(_go().REPO_DIR.resolve())
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


def _atomic_write_bytes(path: pathlib.Path, data: bytes) -> None:
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_bytes(data)
    tmp.replace(path)


def _create_rescue_snapshot(branch: str, reason: str,
                             repo_state: Dict[str, Any], *,
                             link_evolution: bool = True) -> Dict[str, Any]:
    now = datetime.datetime.now(datetime.timezone.utc)
    ts = now.strftime("%Y%m%d_%H%M%S")
    rescue_dir = _go().DRIVE_ROOT / "archive" / "rescue" / f"{ts}_{uuid.uuid4().hex[:8]}"
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

    rc_status, status_txt, status_error = _go().rescue_git_capture(
        ["git", "status", "--porcelain"]
    )
    if rc_status == 0:
        _go().atomic_write_text(rescue_dir / "status.porcelain.txt",
                          status_txt + ("\n" if status_txt else ""))
    else:
        info["warnings"].append(
            f"snapshot_status_error:{status_error or f'git status exited {rc_status} without stderr'}"
        )

    # changes.diff must survive BYTES end-to-end: on an unmerged index it is the
    # ONLY carrier of in-progress resolutions, and text-mode capture would corrupt
    # non-UTF-8 content into U+FFFD. The flag tail pins away operator config that
    # reshapes diff output into something `git apply` cannot re-apply: external
    # diff drivers (--no-ext-diff), textconv filters (--no-textconv), colour
    # escapes (--no-color) and prefix rewrites (--src-prefix/--dst-prefix beat
    # diff.noprefix). GIT_DIFF_OPTS is dropped from the environment because it
    # can carry a context-width override that beats the flags.
    try:
        from ouroboros.update_channels import get_rescue_git_timeout_sec

        capture_env = {k: v for k, v in os.environ.items() if k != "GIT_DIFF_OPTS"}
        capture_env.update({"LC_ALL": "C", "LANG": "C"})
        diff_rc, diff_stdout, diff_stderr = _go()._run_git_process_bounded(
            ["git", "diff", "--binary", "--no-ext-diff", "--no-textconv", "--no-color",
             "--src-prefix=a/", "--dst-prefix=b/", "HEAD"],
            cwd=_go().REPO_DIR,
            env=capture_env,
            text=False,
            timeout=get_rescue_git_timeout_sec(),
        )
        if diff_rc == 0:
            _go()._atomic_write_bytes(rescue_dir / "changes.diff", diff_stdout or b"")
        else:
            raw_error = diff_stderr or b""
            info["diff_error"] = (
                raw_error.decode("utf-8", "replace").strip()
                if isinstance(raw_error, bytes)
                else str(raw_error).strip()
            ) or "git diff failed"
    except Exception as diff_exc:
        log.warning("Rescue diff capture failed", exc_info=True)
        info["diff_error"] = repr(diff_exc)

    # Also capture tracked changes as a real, recoverable git object so recovery
    # is `git stash apply <sha>` / `git checkout <ref> -- .` rather than only a
    # loose diff file. `git stash create` snapshots staged+unstaged tracked
    # changes (it omits untracked files, which the copy below preserves). Purely
    # additive: failure here never blocks the reset and the diff/untracked copy
    # remain the primary recovery artifacts.
    rc_stash, stash_sha, stash_err = _go().rescue_git_capture(["git", "stash", "create", f"rescue:{reason}"])
    stash_sha = stash_sha.strip()
    if rc_stash != 0:
        # rc==0 with an empty sha is LEGITIMATE (nothing to stash / untracked-only
        # dirt); a nonzero rc — e.g. "needs merge" on an unmerged index — is
        # disclosed instead of silently omitting rescue_ref.
        info["rescue_stash_error"] = stash_err or "git stash create failed"
    elif stash_sha:
        ref_name = f"refs/rescue/{rescue_dir.name}"
        rc_ref, _, ref_err = _go().rescue_git_capture(["git", "update-ref", ref_name, stash_sha])
        if rc_ref == 0:
            info["rescue_ref"] = ref_name
            info["rescue_commit"] = stash_sha
        else:
            info["rescue_ref_error"] = ref_err or "git update-ref failed"

    # Merge topology (best-effort): an in-progress merge cannot be stash-captured,
    # so record MERGE_HEAD, the unmerged index entries, and the merge message —
    # together with changes.diff (a plain worktree-vs-HEAD diff that DOES carry
    # in-progress resolutions) they make the merge state operator-recoverable.
    try:
        rc_mh, merge_head, mh_error = _go().rescue_git_capture(
            ["git", "rev-parse", "-q", "--verify", "MERGE_HEAD"]
        )
        if rc_mh == 0 and merge_head.strip():
            info["merge_head"] = merge_head.strip()
            rc_u, unmerged_txt, unmerged_error = _go().rescue_git_capture(
                ["git", "ls-files", "-u"]
            )
            if rc_u == 0 and unmerged_txt:
                _go().atomic_write_text(rescue_dir / "unmerged.txt", unmerged_txt + "\n")
                # Unique conflicted PATHS (stage 1/2/3 rows collapse to one path).
                info["unmerged_count"] = len({
                    ln.split("\t", 1)[-1] for ln in unmerged_txt.splitlines() if ln.strip()
                })
            elif rc_u != 0:
                info["warnings"].append(
                    f"unmerged_index_error:{unmerged_error or f'git ls-files exited {rc_u} without stderr'}"
                )
            # --git-path: in a linked worktree .git is a FILE, so a naive
            # .git/MERGE_MSG probe would silently drop the message.
            rc_p, msg_rel, msg_path_error = _go().rescue_git_capture(
                ["git", "rev-parse", "--git-path", "MERGE_MSG"]
            )
            if rc_p != 0:
                info["warnings"].append(
                    f"merge_msg_path_error:{msg_path_error or f'git rev-parse exited {rc_p} without stderr'}"
                )
            merge_msg_path = (_go().REPO_DIR / msg_rel) if rc_p == 0 and msg_rel else (
                _go()._git_dir() / "MERGE_MSG"
            )
            if merge_msg_path.is_file():
                _go().atomic_write_text(rescue_dir / "merge_msg.txt",
                                  merge_msg_path.read_text(encoding="utf-8", errors="replace"))
        elif rc_mh != 1 or bool(mh_error.strip()):
            info["warnings"].append(
                f"merge_head_error:{mh_error or f'git rev-parse exited {rc_mh} without stderr'}"
            )
    except Exception as exc:
        log.warning("Failed to capture merge topology into rescue snapshot", exc_info=True)
        info["warnings"].append(f"merge_topology_error:{exc!r}")

    untracked_meta = _go()._copy_untracked_for_rescue(rescue_dir / "untracked")
    info["untracked"] = untracked_meta

    unpushed_lines = [ln for ln in (repo_state.get("unpushed_lines") or []) if str(ln).strip()]
    if unpushed_lines:
        _go().atomic_write_text(rescue_dir / "unpushed_commits.txt",
                          "\n".join(unpushed_lines) + "\n")

    _go().atomic_write_text(rescue_dir / "rescue_meta.json",
                      json.dumps(info, ensure_ascii=False, indent=2))
    if link_evolution:
        _go()._link_rescue_to_evolution_transaction(info, reason)
    return info


def _link_rescue_to_evolution_transaction(rescue_info: Dict[str, Any], reason: str) -> None:
    """Attach rescue recovery pointers to the active evolution transaction."""
    try:
        from supervisor.evolution_lifecycle import link_evolution_rescue

        linked = link_evolution_rescue(pathlib.Path(_go().DRIVE_ROOT), rescue_info)
        if not linked:
            return
        _go().append_jsonl(
            pathlib.Path(_go().DRIVE_ROOT) / "logs" / "supervisor.jsonl",
            {
                "ts": _go().utc_now_iso(),
                "type": "evolution_transaction_rescue_linked",
                "reason": reason,
                "transaction_id": linked.get("transaction_id"),
                "task_id": linked.get("task_id"),
                "rescue_ref": linked.get("rescue_ref"),
                "rescue_path": linked.get("rescue_path"),
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


def rescue_before_destructive_rollback(reason: str, *, context: str = "rollback") -> Dict[str, Any]:
    """Best-effort rescue snapshot before a destructive managed-update step.

    Returns a pointer ``{path, ref, ts}`` on capture, ``{}`` when the tree is
    clean and no merge is in progress — nothing to rescue, so a replayed
    ``rolling_back`` boot stays idempotent — and ``{"error": ...}`` on failure.
    A git-status failure counts as a DIRTY tree: an unreadable tree is rescued,
    not skipped. ``context`` only labels the durable reason (``rollback`` →
    ``managed_update_rollback:*``, anything else → ``managed_update_rescue:*``,
    e.g. the boot re-materialization path). FAIL-OPEN by owner decision
    (2026-08-10, 4=A): failures never block the rollback — they are logged and
    returned as the typed ``error`` marker. One durable supervisor.jsonl line
    records the capture (or its failure) before the destructive step; that
    write itself never branches the flow. The snapshot is NOT linked to the
    active evolution transaction — it documents a managed-update rollback, and
    the link would flip a live evolution cycle to "abandoned". Transaction
    bookkeeping stays with the caller (update_merge); this helper only talks to
    git and the supervisor log."""
    try:
        rc_status, dirty, status_error = _go().rescue_git_capture(
            ["git", "status", "--porcelain"]
        )
        rc_mh, merge_head, merge_head_error = _go().rescue_git_capture(
            ["git", "rev-parse", "-q", "--verify", "MERGE_HEAD"]
        )
        merge_in_progress = rc_mh == 0 and bool(merge_head.strip())
        merge_absent = rc_mh == 1 and not merge_head_error.strip()
        if rc_status == 0 and not dirty.strip() and merge_absent:
            return {}
        repo_state = _go()._collect_repo_sync_state()
        warnings = repo_state.setdefault("warnings", [])
        if rc_status != 0:
            warnings.append(
                f"rollback_status_error:{status_error or f'git status exited {rc_status} without stderr'}"
            )
        if not merge_in_progress and not merge_absent:
            warnings.append(
                f"merge_head_error:{merge_head_error or f'git rev-parse exited {rc_mh} without a merge head'}"
            )
        branch = str(repo_state.get("current_branch") or _go().BRANCH_DEV)
        prefix = "managed_update_rollback" if context == "rollback" else "managed_update_rescue"
        info = _go()._create_rescue_snapshot(
            branch, f"{prefix}:{reason}", repo_state, link_evolution=False,
        )
        result: Dict[str, Any] = {
            "path": str(info.get("path") or ""),
            "ref": str(info.get("rescue_ref") or ""),
            "ts": str(info.get("ts") or ""),
        }
        event = {
            "ts": _go().utc_now_iso(), "type": "managed_update_rescue_captured",
            "reason": reason, "rescue_path": result["path"],
            **({"rescue_ref": result["ref"]} if result["ref"] else {}),
            **({"warnings": list(info.get("warnings") or [])}
               if info.get("warnings") else {}),
        }
    except Exception as exc:
        log.warning(
            "rescue before destructive rollback failed (rollback continues)", exc_info=True
        )
        result = {"error": repr(exc)}
        event = {"ts": _go().utc_now_iso(), "type": "managed_update_rescue_failed",
                 "reason": reason, "error": repr(exc)}
    try:
        if not _go().append_jsonl(_go().DRIVE_ROOT / "logs" / "supervisor.jsonl", event):
            log.warning(
                "rescue disclosure could not be written to supervisor.jsonl "
                "(rescue itself is at %s)", result.get("path") or "<none>",
            )
    except Exception:
        log.warning("rescue disclosure raised (continuing)", exc_info=True)
    return result


def rescue_into_tx(tx: Dict[str, Any], *, key: str, reason: str, context: str,
                   writer) -> Dict[str, Any]:
    """Take a pre-destructive rescue and record its outcome in the update tx.

    A captured pointer lands under *key* as ``{path, ref?, ts, reason, count}``
    and is persisted via *writer* (``update_merge.write_update_tx``) BEFORE the
    caller's destructive step — the persisted pointer doubles as the replay
    guard against duplicate rescues. ``count`` increments when a previous
    pointer is overwritten (each re-materialization takes a fresh rescue), so
    the objective renderer can honestly say "latest of N". A capture failure is
    recorded in-memory under ``<key>_error`` for the caller's terminal event and
    is NOT persisted, so a retried rollback re-attempts the rescue. Fail-open
    throughout: a failed tx write is logged and never blocks the caller."""
    rescue_info = _go().rescue_before_destructive_rollback(reason, context=context)
    if rescue_info.get("path"):
        prior = tx.get(key)
        count = (int(prior.get("count") or 1) + 1) if isinstance(prior, dict) else 1
        pointer = {"path": rescue_info["path"], "ts": rescue_info.get("ts") or "",
                   "reason": reason, "count": count}
        if rescue_info.get("ref"):
            pointer["ref"] = rescue_info["ref"]
        tx[key] = pointer
        try:
            writer(tx)
        except Exception:
            log.warning("could not persist the %s rescue pointer into the update tx",
                        key, exc_info=True)
    elif rescue_info.get("error"):
        tx[f"{key}_error"] = str(rescue_info["error"])
    return rescue_info
