"""Deliverable-export eligibility policy for declared process outputs.

Extracted from ``ouroboros/tools/shell.py`` (capinv-447, D4) when that module
crossed its size gate. These are the PURE per-path rules deciding which files a
``run_command``/``run_script`` ``outputs=[...]`` declaration may copy into the
task artifact store. The credential-shape judgment reuses the workspace-patch
SSOT (``workspace_patch_rules._sensitive_untracked_reason``), so exports and
patches can no longer disagree about names like ``.env.example`` or
``.gitignore``. ``shell`` re-exports every name here (same objects).
"""

from __future__ import annotations

import pathlib
from typing import Any

from ouroboros.runtime_mode_policy import is_protected_runtime_path
from ouroboros.tool_access import (
    ResolvedResourceBinding,
    path_is_relative_to,
    resource_root_path,
)
from ouroboros.workspace_patch_rules import _sensitive_untracked_reason

_OUTPUT_DIR_MAX_FILES = 1000
_OUTPUT_DIR_MAX_BYTES = 50 * 1024 * 1024


def _sensitive_output_component_reason(parts: tuple[str, ...]) -> str:
    """Credential-shape check for every relative path component of an export.

    D4 (capinv-447): reuses the workspace-patch SSOT (`_sensitive_untracked_reason`)
    per component — a plain dotfile like `.gitignore` or `.github/...` is an
    ordinary deliverable, while `.env`, `id_rsa`, or `secrets.json` anywhere in
    the path still refuses with the exact rule that fired.
    """
    for part in parts:
        text = str(part or "")
        if not text:
            continue
        reason = _sensitive_untracked_reason(text)
        if reason:
            return f"credential-like output path component {text} is not a deliverable artifact ({reason})"
    return ""


def _changed_path_covers(rel: str, changed_paths: set[str]) -> bool:
    clean = str(rel or "").strip().strip("/")
    if not clean:
        return False
    for item in changed_paths or set():
        path = str(item or "").strip().strip("/")
        if path == clean or path.startswith(clean + "/") or clean.startswith(path + "/"):
            return True
    return False


def _protected_output_source_reason(
    ctx: Any,
    source: pathlib.Path,
    label: str,
    changed_paths: set[str],
    binding: ResolvedResourceBinding | None = None,
) -> str:
    """Return a block reason for protected/control-plane output sources."""

    try:
        from ouroboros.protected_artifacts import block_reason_for_path

        protected_artifact_reason = block_reason_for_path(
            ctx, source, "copy", binding,
        )
        if protected_artifact_reason:
            return protected_artifact_reason
    except Exception:
        pass

    # D4 (capinv-447): the credential-shape judgment is the workspace-patch SSOT
    # (permits .env.example and ordinary dotfiles like .gitignore); the exporter
    # previously blanket-blocked every dotfile and disagreed with patch policy.
    sensitive_reason = _sensitive_untracked_reason(source.name)
    if sensitive_reason:
        return f"credential-like output {source.name} is not a deliverable artifact ({sensitive_reason})"

    try:
        system_repo = pathlib.Path(getattr(ctx, "system_repo_dir", None) or getattr(ctx, "repo_dir")).resolve(strict=False)
    except Exception:
        system_repo = pathlib.Path(getattr(ctx, "repo_dir")).resolve(strict=False)
    if path_is_relative_to(source, system_repo):
        try:
            rel = source.relative_to(system_repo).as_posix()
        except ValueError:
            rel = source.name
        if is_protected_runtime_path(rel):
            return f"protected repo output {rel} is not a deliverable artifact"
        if label in {"active_workspace", "system_repo"} and not _changed_path_covers(rel, changed_paths):
            return f"unchanged repo output {rel} is not a generated deliverable"

    try:
        drive = pathlib.Path(getattr(ctx, "drive_root")).resolve(strict=False)
        if path_is_relative_to(source, drive):
            if (
                binding is not None
                and binding.root == "skill_payload"
                and path_is_relative_to(source, binding.base_path)
            ):
                return ""
            task_drive = resource_root_path(ctx, "task_drive")
            artifact_store = resource_root_path(ctx, "artifact_store")
            if not (path_is_relative_to(source, task_drive) or path_is_relative_to(source, artifact_store)):
                return "runtime data output is not a user deliverable; use task_drive or artifact_store"
    except Exception:
        pass

    return ""


def _scan_directory_output_members(
    ctx: Any,
    source: pathlib.Path,
    *,
    label: str,
    changed_paths: set[str],
    binding: ResolvedResourceBinding | None = None,
) -> tuple[list[pathlib.Path], int, str, list[str]]:
    """Return ``(members, dir_size, whole_dir_block_reason, skipped_receipts)``.

    D4 (capinv-447): a policy-rejected MEMBER is skipped with a receipt instead
    of failing the whole declared directory (one ``.env`` no longer discards an
    otherwise-successful export). Structural failures — size/count caps and an
    unreadable tree — still refuse the directory as a whole.
    """
    root = pathlib.Path(source).resolve(strict=False)
    members: list[pathlib.Path] = []
    skipped: list[str] = []
    dir_size = 0
    try:
        for child in root.rglob("*"):
            if child.is_symlink():
                skipped.append(f"{child}: symlink members are not followed")
                continue
            if not child.is_file():
                continue
            try:
                rel_parts = child.resolve(strict=False).relative_to(root).parts
            except ValueError:
                rel_parts = child.parts
            component_reason = _sensitive_output_component_reason(rel_parts)
            if component_reason:
                skipped.append(f"{child}: {component_reason}")
                continue
            reason = _protected_output_source_reason(
                ctx, child.resolve(strict=False), label, changed_paths, binding,
            )
            if reason:
                skipped.append(f"{child}: {reason}")
                continue
            members.append(child)
            try:
                dir_size += child.stat().st_size
            except OSError:
                pass
            if len(members) > _OUTPUT_DIR_MAX_FILES:
                return [], dir_size, f"{source}: directory output has more than {_OUTPUT_DIR_MAX_FILES} files", skipped
            if dir_size > _OUTPUT_DIR_MAX_BYTES:
                return [], dir_size, f"{source}: directory output exceeds {_OUTPUT_DIR_MAX_BYTES} bytes", skipped
    except OSError as exc:
        return [], dir_size, f"{source}: {type(exc).__name__}: {exc}", skipped
    return sorted(members, key=lambda item: item.as_posix()), dir_size, "", skipped
