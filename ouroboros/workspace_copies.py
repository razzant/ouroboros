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
    """Admission-time Git identity, including nested/linked copies of the body."""
    source, system = Path(source).resolve(), Path(system_repo).resolve()
    if same_directory(source, system):
        return True
    def common(root):
        try:
            proc = subprocess.run(["git", "rev-parse", "--git-common-dir"], cwd=str(root),
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
    from ouroboros.runtime_mode_policy import mode_allows_protected_write, protected_paths_in
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
            grant = not constraint or constraint.mode != "acting_subagent" or constraint.protected_paths_grant
            if protected_paths_in(touched) and not (mode_allows_protected_write(mode) and grant):
                return "CORE_PROTECTION_BLOCKED: the parent's current authority excludes these protected body paths."
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        return f"INTEGRATE_TARGET_FORBIDDEN: {exc}"
    return ""


def target_write_refusal(target: Path, refusal: str) -> str:
    """One marked integration refusal at either current-authority recheck."""
    detail = refusal.removeprefix("⚠️ ").removeprefix("INTEGRATE_TARGET_FORBIDDEN: ")
    return (f"⚠️ INTEGRATE_TARGET_FORBIDDEN: current write authority at the recorded target "
            f"{target} does not admit this apply: {detail}")
