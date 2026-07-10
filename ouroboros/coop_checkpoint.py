"""Coop/genesis checkpoint-commit (v6.58.0, 2.4B).

When the ROOT of a task tree finalizes, any DIRTY host-minted genesis/coop tree its
children built in gets a local checkpoint commit, so cooperative work becomes durable
git history instead of an uncommitted pile a later crash/cleanup could lose.

Boundaries (BIBLE "Leaking secrets: nowhere" + owner-folder ownership):
- ONLY trees under the subagent-projects root (host-minted); an owner-attached folder
  is NEVER auto-committed.
- Credential-shaped files (the same `_sensitive_untracked_reason` patterns the
  workspace patch excludes) are unstaged before the commit, disclosed in the receipt.
- Skipped while the tree still has live tasks; fail-soft per root; never raises.
"""
from __future__ import annotations

import pathlib
import subprocess
from typing import Any, Dict, List, Sequence

from ouroboros.headless import _sensitive_untracked_reason

def _run_git(cmd: Sequence[str], cwd: pathlib.Path) -> "subprocess.CompletedProcess[str]":
    """Bounded git call returning the full CompletedProcess (checkpoint-commit path).
    A timeout/spawn failure returns a synthetic rc=124 result — callers treat any
    non-zero rc as a fail-soft skip, so this never raises."""
    try:
        return subprocess.run(list(cmd), cwd=str(cwd), capture_output=True, text=True, timeout=60)
    except Exception as exc:  # noqa: BLE001 — includes TimeoutExpired
        return subprocess.CompletedProcess(list(cmd), 124, stdout="", stderr=f"{type(exc).__name__}: {exc}")


def _task_tree_coop_roots(drive_root: pathlib.Path, root_task_id: str) -> List[pathlib.Path]:
    """Unique host-minted genesis/coop tree roots this task tree's children wrote to —
    read from the children's durable results (write_root/workspace_root under the
    subagent-projects root). NEVER an owner-attached folder: only paths inside
    ``get_subagent_projects_root()`` qualify (the checkpoint-commit boundary)."""
    from ouroboros.config import get_subagent_projects_root
    from ouroboros.task_status import find_child_tasks
    from ouroboros.tool_access import path_is_relative_to

    projects_root = pathlib.Path(get_subagent_projects_root()).expanduser().resolve(strict=False)
    out: List[pathlib.Path] = []
    try:
        children = find_child_tasks(
            pathlib.Path(drive_root), parent_task_id=str(root_task_id), root_task_id=str(root_task_id)
        )
    except Exception:
        return out
    for child in children:
        constraint = child.get("task_constraint") if isinstance(child.get("task_constraint"), dict) else {}
        for value in (constraint.get("write_root"), child.get("workspace_root")):
            text = str(value or "").strip()
            if not text:
                continue
            try:
                candidate = pathlib.Path(text).expanduser().resolve(strict=False)
            except (OSError, ValueError):
                continue
            if not path_is_relative_to(candidate, projects_root):
                continue
            if candidate not in out and (candidate / ".git").exists():
                out.append(candidate)
    return out


def checkpoint_commit_coop_roots(
    drive_root: pathlib.Path,
    root_task_id: str,
    *,
    title: str = "",
    has_live_tree_tasks: bool = False,
) -> List[Dict[str, Any]]:
    """v6.58.0 (2.4B) — host checkpoint-commit of DIRTY genesis/coop roots when the
    ROOT task finalizes, so cooperative work is never left as an uncommitted pile a
    later crash/cleanup could lose. Boundaries:

    - ONLY host-minted genesis/coop trees (under the subagent-projects root); an
      owner-attached folder is NEVER auto-committed (the owner owns its history).
    - Skipped entirely while the tree still has live tasks (a racing child could be
      mid-write); children are terminal by root finalization in the normal flow.
    - Credential-shaped files (the SAME `_sensitive_untracked_reason` patterns the
      workspace patch excludes) are NOT staged — BIBLE "Leaking secrets: nowhere":
      this is a refusal to bake secrets into git history, disclosed in the receipt.
    - Fail-soft per root (index.lock, git errors → logged skip; never raises).

    Returns a list of per-root receipts {root, committed, sha?, skipped_sensitive[], error?}.
    """
    receipts: List[Dict[str, Any]] = []
    if has_live_tree_tasks:
        return receipts
    for root in _task_tree_coop_roots(drive_root, root_task_id):
        receipt: Dict[str, Any] = {"root": str(root), "committed": False, "skipped_sensitive": []}
        try:
            status = _run_git(["git", "status", "--porcelain"], root)
            if status.returncode != 0:
                receipt["error"] = (status.stderr or status.stdout or "git status failed").strip()[:300]
                receipts.append(receipt)
                continue
            if not (status.stdout or "").strip():
                receipts.append(receipt)  # clean tree — nothing to checkpoint
                continue
            # Stage everything EXCEPT credential-shaped files (disclosed skip).
            add = _run_git(["git", "add", "-A"], root)
            if add.returncode != 0:
                receipt["error"] = (add.stderr or add.stdout or "git add failed").strip()[:300]
                receipts.append(receipt)
                continue
            staged = _run_git(["git", "diff", "--cached", "--name-only"], root)
            for rel in (staged.stdout or "").splitlines():
                rel = rel.strip()
                if not rel:
                    continue
                reason = _sensitive_untracked_reason(rel)
                if reason:
                    _run_git(["git", "reset", "-q", "HEAD", "--", rel], root)
                    receipt["skipped_sensitive"].append({"path": rel, "reason": reason})
            label = f" — {title.strip()}" if str(title or "").strip() else ""
            commit = _run_git(
                [
                    "git", "-c", "user.name=Ouroboros", "-c", "user.email=ouroboros@local",
                    "commit", "-m", f"ouroboros: checkpoint after task {root_task_id}{label}",
                ],
                root,
            )
            if commit.returncode == 0:
                head = _run_git(["git", "rev-parse", "HEAD"], root)
                receipt["committed"] = True
                receipt["sha"] = (head.stdout or "").strip()[:40]
            else:
                # "nothing to commit" after sensitive-only unstage is a benign no-op.
                detail = (commit.stderr or commit.stdout or "").strip()
                if "nothing to commit" not in detail.lower():
                    receipt["error"] = detail[:300]
        except Exception as exc:  # noqa: BLE001 — checkpoint is best-effort by design
            receipt["error"] = f"{type(exc).__name__}: {exc}"
        receipts.append(receipt)
    return receipts


