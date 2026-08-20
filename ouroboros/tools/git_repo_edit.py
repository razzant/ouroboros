"""Uncommitted repository write and exact-match edit implementations.

Owns write_file/edit_text behaviour: multi-file write batching, the pre-write
syntax and shrink guards, skill-payload redirection and repair CAS binding,
and the advisory invalidation that follows every mutation. The tool
descriptors stay with their catalog owners.
"""

from __future__ import annotations

import pathlib
import subprocess
from typing import Dict, List, Optional

from ouroboros.runtime_mode_policy import (
    core_patch_notice,
    is_protected_runtime_path,
    mode_allows_protected_write,
    normalize_repo_path,
    protected_paths_in,
    protected_write_block_message,
)
from ouroboros.tools.registry import (
    ToolContext,
    _authorized_managed_update_resolver,
)
from ouroboros.tool_access import (
    ResolvedResourceBinding,
    build_resolved_resource_binding,
)
from ouroboros.tools.commit_gate import _invalidate_advisory
from ouroboros.utils import write_text, safe_relpath
from ouroboros.tools.core import _data_skill_path, _str_match_replace, is_skill_control_plane_path
from ouroboros.contracts.task_constraint import normalize_task_constraint, resolve_payload_path
from ouroboros.contracts.skill_payload_policy import (
    cross_skill_redirect_error,
    decide_payload_short_form,
)
from ouroboros.tools.git_plumbing import (
    _binding_repo_rel,
    _binding_targets_system_repo,
    _current_runtime_mode,
)
_CONTENT_OMITTED_PREFIX = "<<CONTENT_OMITTED"


def _check_shrink_guard(
    binding: ResolvedResourceBinding,
    new_content: str,
    force: bool = False,
) -> Optional[str]:
    """Block likely accidental tracked-file truncation unless force=True."""
    if force:
        return None
    try:
        target = binding.target_path
        file_path = _binding_repo_rel(binding)
        if not target.exists():
            return None
        result = subprocess.run(
            ["git", "ls-files", "--error-unmatch", safe_relpath(file_path)],
            cwd=str(binding.base_path), capture_output=True, text=True,
        )
        if result.returncode != 0:
            return None
        old_content = target.read_text(encoding="utf-8")
        old_len = len(old_content)
        new_len = len(new_content)
        if old_len > 0 and new_len < old_len * 0.7:
            pct = round(new_len / old_len * 100)
            return (
                f"⚠️ WRITE_BLOCKED: new content for '{file_path}' is {pct}% of original "
                f"({old_len} -> {new_len} chars). This looks like accidental truncation. "
                f"Use edit_text for surgical edits, or pass force=true to confirm "
                f"intentional rewrite."
            )
    except Exception:
        pass
    return None


def _repo_write(ctx: ToolContext, path: str = "", content: str = "",
                files: Optional[List[Dict[str, str]]] = None,
                force: bool = False,
                display_root: str = "active_workspace",
                _resolved_binding: (
                    ResolvedResourceBinding | tuple[ResolvedResourceBinding, ...] | None
                ) = None) -> str:
    """Write file(s) to the repo working directory without committing."""
    write_list: List[Dict[str, str]] = []
    if files:
        for entry in files:
            if not isinstance(entry, dict):
                return "⚠️ WRITE_ERROR: each item in files must be {path, content}."
            p = entry.get("path", "").strip()
            c = entry.get("content", "")
            if not p:
                return "⚠️ WRITE_ERROR: every file entry must have a non-empty 'path'."
            write_list.append({"path": p, "content": c})
    elif path and content is not None:
        write_list.append({"path": path.strip(), "content": content})
    else:
        return "⚠️ WRITE_ERROR: provide either (path + content) or files array."

    if not write_list:
        return "⚠️ WRITE_ERROR: nothing to write."

    try:
        if _resolved_binding is None:
            binding_items = tuple(
                build_resolved_resource_binding(
                    ctx, root=display_root, operation="write", path=e["path"],
                )
                for e in write_list
            )
        elif isinstance(_resolved_binding, tuple):
            binding_items = _resolved_binding
        else:
            binding_items = (_resolved_binding,)
        if len(binding_items) != len(write_list):
            return "⚠️ WRITE_ERROR: resolved target count does not match files."
    except Exception as exc:
        return f"⚠️ WRITE_ERROR: could not resolve target: {type(exc).__name__}: {exc}"

    for e, binding in zip(write_list, binding_items):
        norm = normalize_repo_path(_binding_repo_rel(binding))
        if (
            _binding_targets_system_repo(ctx, binding)
            and is_protected_runtime_path(norm)
            and not mode_allows_protected_write(_current_runtime_mode())
            and not _authorized_managed_update_resolver(ctx)
        ):
            return protected_write_block_message(
                path=norm,
                runtime_mode=_current_runtime_mode(),
                action="write",
            )
        if isinstance(e["content"], str) and e["content"].strip().startswith(_CONTENT_OMITTED_PREFIX):
            return (
                f"⚠️ WRITE_ERROR: content for '{e['path']}' looks like a compaction marker. "
                "Re-read the file and provide the actual content."
            )

    # Pre-write syntax guard for known formats (from edit_sketch's verification
    # rails, editbench v2): a full-file overwrite that doesn't even parse is
    # never intentional — block BEFORE any write, force bypasses (deliberately
    # invalid fixtures). Runs before the write loop so the batch stays atomic.
    # P3: the force bypass is never silent — a forced write of invalid content
    # still discloses what the guard found in the success message.
    syntax_bypass_notes: List[str] = []
    from ouroboros.tools.edit_ops import _syntax_check

    for e, binding in zip(write_list, binding_items):
        rel_path = _binding_repo_rel(binding)
        syntax_err = _syntax_check(rel_path, e["content"])
        if not syntax_err:
            continue
        if force:
            syntax_bypass_notes.append(f"{rel_path}: {syntax_err}")
            continue
        return (
            f"⚠️ WRITE_BLOCKED_SYNTAX: {syntax_err} for '{e['path']}'. "
            "Nothing was written. Fix the content, or pass force=true for an "
            "intentionally invalid file."
        )

    written = []
    written_paths: List[str] = []
    overwrite_diffs: List[str] = []
    for e, binding in zip(write_list, binding_items):
        rel_path = _binding_repo_rel(binding)
        shrink_warning = _check_shrink_guard(binding, e["content"], force=force)
        if shrink_warning:
            if written:
                _invalidate_advisory(
                    ctx,
                    changed_paths=written_paths,
                    mutation_root=binding_items[0].base_path,
                    source_tool="write_file",
                )
            return shrink_warning
        try:
            target = binding.target_path
            old_content: Optional[str] = None
            if target.exists():
                try:
                    old_content = target.read_text(encoding="utf-8")
                except Exception:
                    old_content = None
            target.parent.mkdir(parents=True, exist_ok=True)
            write_text(target, e["content"])
            written.append(f"{display_root}:{rel_path} ({len(e['content'])} chars)")
            written_paths.append(rel_path)
            if old_content is not None and old_content != e["content"]:
                from ouroboros.tools.edit_ops import _unified_diff

                overwrite_diffs.append(_unified_diff(rel_path, old_content, e["content"], cap=120))
        except Exception as exc:
            if written:
                _invalidate_advisory(
                    ctx,
                    changed_paths=written_paths,
                    mutation_root=binding_items[0].base_path,
                    source_tool="write_file",
                )
            already = ", ".join(written) if written else "(none)"
            return (
                f"⚠️ FILE_WRITE_ERROR on '{e['path']}': {exc}\n"
                f"Successfully written before error: {already}"
            )

    _invalidate_advisory(
        ctx,
        changed_paths=written_paths,
        mutation_root=binding_items[0].base_path,
        source_tool="write_file",
    )
    summary = ", ".join(written)
    system_target = _binding_targets_system_repo(ctx, binding_items[0])
    if ctx.is_workspace_mode() and not system_target:
        result = (
            f"✅ Written {len(written)} file(s): {summary}\n"
            "Files are on disk in the active workspace. Do not commit; the headless runner will emit a patch artifact."
        )
    else:
        result = (
            f"✅ Written {len(written)} file(s): {summary}\n"
            "Files are on disk but NOT committed. Run commit_reviewed when ready.\n"
            "⚠️ Advisory pre-review is now stale — run advisory_review before commit_reviewed."
        )
    result += f"\nResolved root: {binding_items[0].base_path}"
    if syntax_bypass_notes:
        result += (
            "\n⚠️ SYNTAX_GUARD_BYPASSED (force=true): "
            + "; ".join(syntax_bypass_notes)
        )
    if overwrite_diffs:
        result += (
            "\nDiff vs the previous version (verify it matches your intent):\n"
            + "\n".join(overwrite_diffs)
        )
    if system_target and any(pathlib.PurePosixPath(item).parts[:1] == ("skills",) for item in written_paths):
        result += (
            "\nℹ️ Native seed boundary: system_repo/skills changed; the installed "
            "data/skills/native copy remains unchanged until launcher reseed."
        )
    protected_written = protected_paths_in(written_paths) if system_target else []
    if protected_written and mode_allows_protected_write(_current_runtime_mode()):
        result += "\n\n" + core_patch_notice(protected_written)
    return result


def _str_replace_editor(
    ctx: ToolContext,
    path: str,
    old_str: str,
    new_str: str,
    bucket: str = "",
    skill_name: str = "",
    display_root: str = "active_workspace",
    force: bool = False,
    _resolved_binding: ResolvedResourceBinding | None = None,
) -> str:
    """Replace exactly one occurrence of old_str with new_str in a file."""
    if not path or not path.strip():
        return "⚠️ STR_REPLACE_ERROR: path is required."
    if not old_str:
        return "⚠️ STR_REPLACE_ERROR: old_str is required (cannot be empty)."

    existing_tc = normalize_task_constraint(getattr(ctx, "task_constraint", None))
    data_skill_target = None
    task_constraint = existing_tc
    short_form = None
    binding = _resolved_binding
    if binding is not None:
        target = binding.target_path
        invalidation_root = binding.base_path
    elif not ctx.is_workspace_mode():
        short_form = decide_payload_short_form(
            bucket=bucket,
            skill_name=skill_name,
            path_text=path,
            repo_dir=pathlib.Path(ctx.repo_dir),
            drive_root=pathlib.Path(ctx.drive_root),
        )
        if short_form.error:
            return f"⚠️ STR_REPLACE_ERROR: {short_form.error}"
        synth = short_form.constraint
        redirect_err = cross_skill_redirect_error(existing_tc, synth)
        if redirect_err:
            return f"⚠️ SKILL_REDIRECT_BLOCKED: {redirect_err}"
        task_constraint = existing_tc if existing_tc and existing_tc.mode == "skill_repair" else synth or existing_tc

    if binding is None and not ctx.is_workspace_mode() and task_constraint and task_constraint.mode == "skill_repair" and task_constraint.payload_root:
        try:
            target = resolve_payload_path(pathlib.Path(ctx.drive_root), task_constraint, path)
            data_skill_target = target
        except ValueError as e:
            return f"⚠️ STR_REPLACE_ERROR: {e}"
        if is_skill_control_plane_path(target, pathlib.Path(ctx.drive_root).resolve(strict=False)):
            return (
                "⚠️ STR_REPLACE_BLOCKED: skill provenance, launcher seed, "
                "marketplace, dependency, and self-authored markers are "
                "control-plane state. Edit user-authored payload files instead."
            )
        invalidation_root = pathlib.Path(ctx.drive_root)
    elif binding is None and not ctx.is_workspace_mode():
        data_skill_target = _data_skill_path(path, pathlib.Path(ctx.drive_root))
        if data_skill_target is not None:
            if is_skill_control_plane_path(data_skill_target, pathlib.Path(ctx.drive_root).resolve(strict=False)):
                return (
                    "⚠️ STR_REPLACE_BLOCKED: skill provenance, launcher seed, "
                    "marketplace, dependency, and self-authored markers are "
                    "control-plane state. Edit user-authored payload files instead."
                )
            target = data_skill_target
            invalidation_root = pathlib.Path(ctx.drive_root)
    if binding is None and data_skill_target is None:
        try:
            binding = build_resolved_resource_binding(
                ctx, root=display_root, operation="edit", path=path,
            )
        except Exception as exc:
            return f"⚠️ PATH_ERROR: {exc}"
        target = binding.target_path
        invalidation_root = binding.base_path

    rel_path = _binding_repo_rel(binding) if binding is not None else safe_relpath(path)
    system_target = bool(binding and _binding_targets_system_repo(ctx, binding))
    norm = normalize_repo_path(rel_path)
    if (
        system_target
        and is_protected_runtime_path(norm)
        and not mode_allows_protected_write(_current_runtime_mode())
        and not _authorized_managed_update_resolver(ctx)
    ):
        return protected_write_block_message(
            path=norm,
            runtime_mode=_current_runtime_mode(),
            action="edit",
        )

    if not target.exists():
        return f"⚠️ STR_REPLACE_ERROR: file not found: {path}"

    try:
        content = target.read_text(encoding="utf-8")
    except Exception as e:
        return f"⚠️ STR_REPLACE_ERROR: cannot read {path}: {e}"

    # Shared exact-match single-replacement (deferral 4): identical count==0/count>1
    # feedback for the repo and data-plane editors.
    new_content, _match_err = _str_match_replace(content, old_str, new_str, path, "STR_REPLACE_ERROR")
    if _match_err:
        return _match_err
    if data_skill_target is not None:
        # Deferral 5: a data-plane skill payload edited via the active_workspace route gets
        # the SAME shrink guard as the root=skill_payload editor — no silent >30% truncation
        # of a payload file. (Intentional large rewrites go through root=skill_payload, which
        # carries the force escape hatch.)
        from ouroboros.tools.core import _check_data_shrink_guard

        _shrink_block = _check_data_shrink_guard(target, new_content, force)
        if _shrink_block:
            return _shrink_block
    elif binding is not None:
        _shrink_block = _check_shrink_guard(binding, new_content, force)
        if _shrink_block:
            return _shrink_block
    # X3 hash-bind: the ADMITTED repair task's payload edits CAS-check the
    # repair's own hash chain; drift outside the repair is a typed stale
    # terminalization, never a silent write over foreign changes.
    _repair_cas_constraint = (
        task_constraint
        if task_constraint and task_constraint.mode == "skill_repair"
        and str(getattr(task_constraint, "skill_name", "") or "")
        else None
    )
    if _repair_cas_constraint is not None:
        from ouroboros.skill_repair_admission import repair_write_cas_error

        _cas = repair_write_cas_error(
            pathlib.Path(ctx.drive_root), _repair_cas_constraint,
            task_id=str(getattr(ctx, "task_id", "") or ""),
            # Mandatory only for a real repair TASK; a synthesized short-form
            # selector on an ordinary edit lane is not an admitted repair.
            repair_task=bool(existing_tc and existing_tc.mode == "skill_repair"))
        if _cas:
            return _cas
    try:
        write_text(target, new_content)
    except Exception as e:
        return f"⚠️ STR_REPLACE_ERROR: write failed for {path}: {e}"
    if _repair_cas_constraint is not None:
        from ouroboros.skill_repair_admission import advance_repair_expected_hash

        advance_repair_expected_hash(
            pathlib.Path(ctx.drive_root), _repair_cas_constraint,
            task_id=str(getattr(ctx, "task_id", "") or ""))

    replacement_line = new_content[:new_content.index(new_str)].count('\n') + 1
    context_start = max(0, replacement_line - 3)
    context_lines = new_content.splitlines()[context_start:replacement_line + len(new_str.splitlines()) + 2]
    context_preview = "\n".join(
        f"{context_start + i + 1:>4}| {line}" for i, line in enumerate(context_lines)
    )

    _invalidate_advisory(
        ctx,
        changed_paths=[rel_path],
        mutation_root=invalidation_root,
        source_tool="edit_text",
    )
    result = (
        f"✅ Replaced in {display_root}:{rel_path} (line {replacement_line}).\n"
        f"Context:\n{context_preview}\n\n"
        "File is on disk but NOT committed."
    )
    if binding is not None:
        result += f"\nResolved root: {binding.base_path}"
    if short_form is not None and short_form.ignored_reason:
        result += f"\n⚠️ SKILL_SHORT_FORM_IGNORED: {short_form.ignored_reason}."
    if data_skill_target is None and ctx.is_workspace_mode() and not system_target:
        result += "\nDo not commit; the headless runner will emit a patch artifact."
    elif data_skill_target is None:
        result += "\nRun commit_reviewed when ready.\n⚠️ Advisory pre-review is now stale — run advisory_review before commit_reviewed."
    else:
        result += "\nRun skill_review for this skill before enabling or declaring it ready."
    if system_target and pathlib.PurePosixPath(rel_path).parts[:1] == ("skills",):
        result += (
            "\nℹ️ Native seed boundary: system_repo/skills changed; the installed "
            "data/skills/native copy remains unchanged until launcher reseed."
        )
    if system_target and is_protected_runtime_path(norm) and mode_allows_protected_write(_current_runtime_mode()):
        result += "\n\n" + core_patch_notice([norm])
    return result
