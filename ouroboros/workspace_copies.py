"""Source identity for task-owned isolated copies, independent of isolation.

The supervisor records this binding after provisioning. Legacy self_worktree
rows without it retain their original meaning: a copy of Ouroboros's body.
The working folder is a target address, not evidence that a foreign project is
Ouroboros source merely because both use the same isolated-copy mechanism.
"""
from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Any


def copy_binding(value: Any) -> dict:
    """Read the host's existing task metadata carrier, preserving legacy absence."""
    metadata = (value.get("metadata", value) if isinstance(value, dict)
                else getattr(value, "task_metadata", {}))
    binding = metadata.get("workspace_copy") if isinstance(metadata, dict) else None
    return binding if isinstance(binding, dict) else {}


def same_directory(left: Any, right: Any) -> bool:
    """Physical directory identity, including case aliases on macOS/Windows."""
    if not left or not right:
        return False
    left, right = Path(left).resolve(), Path(right).resolve()
    if left == right:
        return True
    try:
        return left.samefile(right)
    except OSError:
        return False


def source_is_system_repo(source: Any, system_repo: Any) -> bool:
    """Admission-time Git identity, including nested/linked copies of the body. A work-domain folder never is:
    the mappings may not reach the checkout, and a Git directory its repository names is the work's spelling,
    never one the service may resolve into its own."""
    from ouroboros import work_domain

    if work_domain.is_work(source):
        return False
    source, system = Path(source).resolve(), Path(system_repo).resolve()
    if same_directory(source, system):
        return True
    def common(root):
        try:
            from ouroboros import work_domain  # a work repository's Git runs as the work domain

            proc = work_domain.run(["git", "rev-parse", "--git-common-dir"], cwd=str(root),
                                   capture_output=True, text=True, encoding="utf-8", timeout=5)
        except (OSError, subprocess.SubprocessError):
            return None
        if proc.returncode:
            return None
        path = Path(proc.stdout.strip())
        return (path if path.is_absolute() else root / path).resolve()
    source_common = common(source)
    return source_common is not None and same_directory(source_common, common(system))


def workspace_copy_source_is_system(ctx: Any, selected_root: str = "") -> bool:
    """The selected copy source before scheduling; absence selects the own body."""
    from ouroboros.tools.tool_resolution import system_repo_dir_for

    if not selected_root:
        return True
    binding = copy_binding(ctx)
    if binding and same_directory(selected_root, binding.get("execution_root")):
        return binding.get("source_is_system_repo") is not False
    return source_is_system_repo(selected_root, system_repo_dir_for(ctx))


def is_system_copy(ctx: Any) -> bool:
    """Own-body policy of an already admitted isolated child; old rows stay old."""
    from ouroboros.contracts.task_constraint import normalize_task_constraint

    constraint = normalize_task_constraint(getattr(ctx, "task_constraint", None))
    isolated = ((constraint is not None and constraint.surface == "self_worktree")
                or getattr(ctx, "workspace_mode", "") == "self_worktree")
    if not isolated:
        return False
    binding = copy_binding(ctx)
    if not binding:
        return True
    selected = (constraint.write_root if constraint is not None else "") or getattr(ctx, "workspace_root", "")
    if not selected or not same_directory(selected, binding.get("execution_root")):
        return True
    return binding.get("source_is_system_repo") is not False


def admitted_copy_metadata(write_root: str, data_dir: Any = None) -> dict:
    """Materialize the newly provisioned host registry fact in durable task state."""
    from ouroboros.subagent_worktrees import list_worktrees

    for row in list_worktrees(data_dir):
        if row.get("path") != write_root or row.get("kind") == "delegated_exec":
            continue
        return {
            "source_root": row["repo_dir"], "execution_root": row["path"],
            "source_is_system_repo": row.get("source_is_system_repo", True),
            "baseline_sha": row["base_sha"], "target_head": row.get("target_head", ""),
            "source_index_clean": row.get("source_index_clean", False),
            "file_baseline": row.get("file_baseline", {}),
        }
    raise ValueError("provisioned isolated child has no task-owned registry binding")


def copy_apply_refusal(ctx: Any, target: Path, touched: list[str]) -> str:
    """Recheck the parent's existing write rights; selecting a copy grants none.

    Use the same physical-root and per-file resolution as file/process tools,
    without rebinding the task's workspace. Presence remains an intersection.
    """
    from ouroboros.tool_access import build_resolved_resource_binding
    from ouroboros.presence_authority import presence_ceiling_from_context, presence_ceiling_allows_binding
    from ouroboros.tools.tool_resolution import system_repo_dir_for
    from ouroboros.tools.subagent_integration import _integration_runtime_mode, _capped_self_repo_refusal
    from ouroboros.runtime_mode_policy import protected_paths_in
    from ouroboros.contracts.task_constraint import normalize_task_constraint

    try:
        selected = build_resolved_resource_binding(ctx, operation="write", process_cwd=str(target))
        ceiling = presence_ceiling_from_context(ctx)
        for relative in touched:
            binding = build_resolved_resource_binding(
                ctx, root=selected.root, operation="write", path=str(target / relative))
            if ceiling is not None and not presence_ceiling_allows_binding(ceiling, binding):
                return "PRESENCE_RESOURCE_BLOCKED: copy result is outside the parent's current write ceiling."
        if source_is_system_repo(target, system_repo_dir_for(ctx)):
            mode = _integration_runtime_mode(ctx)
            if mode == "light" or _capped_self_repo_refusal(ctx, "copy"):
                return "LIGHT_MODE_BLOCKED: the parent's current mode forbids writing the Ouroboros body."
            constraint = normalize_task_constraint(getattr(ctx, "task_constraint", None))
            acting = bool(constraint and constraint.mode == "acting_subagent")
            grant = bool(constraint and constraint.protected_paths_grant)
            if (protected := protected_paths_in(touched)) and not protected_write_admitted(mode, acting=acting, grant=grant):
                # The patch doors' own words: the mode that refuses, or the applying task's missing grant.
                return protected_apply_refusal(path=protected[0].path, runtime_mode=mode, grant_ok=not acting or grant,
                                               action="apply an isolated copy's result to").removeprefix("⚠️ ")
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        return f"INTEGRATE_TARGET_FORBIDDEN: {exc}"
    return ""


def target_write_refusal(target: Path, refusal: str) -> str:
    """One marked integration refusal at either current-authority recheck."""
    detail = refusal.removeprefix("⚠️ ").removeprefix("INTEGRATE_TARGET_FORBIDDEN: ")
    return (f"⚠️ INTEGRATE_TARGET_FORBIDDEN: current write authority at the recorded target "
            f"{target} does not admit this apply: {detail}")


def protected_write_admitted(runtime_mode: str, *, acting: bool, grant: bool) -> bool:
    """The one rule for a protected body path: the dispatcher's direct write and all three integration doors.

    Pro and Cyber Pro admit protected paths; below Pro nothing does, a consciousness Act/Observe tree
    included (``runtime_mode`` is the effective mode, its ``light`` cap applied). An acting child also
    needs its own ``protected_paths_grant``, the grant of the task applying the change, never of its
    author, except in Cyber Pro: there no internal grant vetoes Ouroboros (BIBLE P0), so the child may
    apply a nested child's finished patch to a protected file as it may already write that file itself
    (owner decision 6A, 21.11.2026). The doors' grant check came with the patch door (06.06,
    5d4f75110), the delegated door (31.08, 782b5fe35) and the copy target (30.09, 9b79876de); the
    dispatcher's Cyber Pro exception (11.09, 5b07a9a7b) never reached them, and 2adb54751 (10.11, an
    Ouroboros commit, not an owner decision) only said whose grant it is. What the refusal protected,
    and the cost the owner accepted: a nested child's changes reach protected files without the
    delegator's own look at them.
    """
    from ouroboros.runtime_mode_policy import mode_allows_protected_write, mode_has_unrestricted_agency

    return mode_allows_protected_write(runtime_mode) and (
        not acting or grant or mode_has_unrestricted_agency(runtime_mode))


def protected_apply_refusal(*, path: str, runtime_mode: str, action: str, grant_ok: bool) -> str:
    """The protected-path refusal of both integration doors, naming the cause that holds.

    ``grant_ok`` is False for an acting caller without its own ``protected_paths_grant``. A mode refusal
    keeps the policy's own words; a missing grant is named, since no switch to Pro supplies it, and the
    captured result stays for the caller's parent to decide.
    """
    from ouroboros.runtime_mode_policy import (
        mode_allows_protected_write, normalize_repo_path, protected_path_category, protected_write_block_message,
    )

    if grant_ok:
        return protected_write_block_message(path=path, runtime_mode=runtime_mode, action=action)
    norm = normalize_repo_path(path)
    grant = ("its own protected_paths_grant (the grant of the task applying a patch; its author's does not "
             "carry over)")
    cause = (f"runtime_mode={runtime_mode!r} admits an acting subagent's protected writes only with {grant}, "
             "which this task lacks" if mode_allows_protected_write(runtime_mode) else
             f"runtime_mode={runtime_mode!r} refuses protected writes, and this acting subagent also lacks {grant}, "
             "so 'pro' alone would not admit it")
    return (f"⚠️ CORE_PROTECTION_BLOCKED: this task may not {action} protected "
            f"{protected_path_category(norm) or 'core'} path: {norm}. {cause}. The captured "
            "result is kept unapplied: report it to your parent, which decides with its own authority and may "
            "schedule the work with protected_paths_grant.")
