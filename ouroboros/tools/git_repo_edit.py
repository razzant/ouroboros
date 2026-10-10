"""Uncommitted repo write and exact-match edit surface.

The ``ouroboros/tools/git.py`` facade re-exports these definitions. Shared
helpers are read through the call-time handle ``_git()`` so the facade binding
stays the one tests monkeypatch.
"""

from __future__ import annotations

import pathlib
import subprocess
from typing import Dict, List, Optional

from ouroboros.reference_books import book_balance_note
from ouroboros.tools.registry import ToolContext
from ouroboros.tool_access import ResolvedResourceBinding, canonical_data_root
from ouroboros.tools.tool_result import publish_no_effect


def _git():
    """The parent module, read at call time.

    The parent owns the rebindable module state and the members tests
    monkeypatch there; reading them through the module at each call keeps
    one binding, where a from-import would freeze the value this leaf saw
    at import time.
    """
    from ouroboros.tools import git

    return git


_CONTENT_OMITTED_PREFIX = "<<CONTENT_OMITTED"


def _check_shrink_guard(
    binding: ResolvedResourceBinding,
    new_content: str,
    force: bool = False,
) -> Optional[str]:
    """Block likely accidental tracked-file truncation unless force=True."""
    from ouroboros.runtime_mode_policy import mode_has_unrestricted_agency

    if force or mode_has_unrestricted_agency(_git()._current_runtime_mode()):
        return None
    try:
        target = binding.target_path
        file_path = _git()._binding_repo_rel(binding)
        if not target.exists():
            return None
        result = subprocess.run(
            ["git", "ls-files", "--error-unmatch", _git().safe_relpath(file_path)],
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
                mode: str = "overwrite",
                force: bool = False,
                display_root: str = "active_workspace",
                _resolved_binding: (
                    ResolvedResourceBinding | tuple[ResolvedResourceBinding, ...] | None
                ) = None) -> str:
    """Write file(s) to the repo working directory without committing.

    ``mode="append"`` appends instead of overwriting (#447 D2: write_file declares
    the parameter for every root — dropping it here turned a chunked large-file
    write into an overwrite that destroyed every prior chunk while reporting
    success). An append chunk is not a full file, so the full-file syntax guard,
    the shrink guard, and the overwrite diff do not apply to it."""
    write_list: List[Dict[str, str]] = []
    if files:
        for entry in files:
            if not isinstance(entry, dict):
                return publish_no_effect(ctx, "⚠️ WRITE_ERROR: each item in files must be {path, content}.", tool_name="write_file")
            p = entry.get("path", "").strip()
            c = entry.get("content", "")
            if not p:
                return publish_no_effect(ctx, "⚠️ WRITE_ERROR: every file entry must have a non-empty 'path'.", tool_name="write_file")
            write_list.append({"path": p, "content": c})
    elif path and content is not None:
        write_list.append({"path": path.strip(), "content": content})
    else:
        return publish_no_effect(ctx, "⚠️ WRITE_ERROR: provide either (path + content) or files array.", tool_name="write_file")

    if not write_list:
        return publish_no_effect(ctx, "⚠️ WRITE_ERROR: nothing to write.", tool_name="write_file")

    try:
        if _resolved_binding is None:
            binding_items = tuple(
                _git().build_resolved_resource_binding(
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
        norm = _git().normalize_repo_path(_git()._binding_repo_rel(binding))
        if (
            _git()._binding_targets_system_repo(ctx, binding)
            and _git().is_protected_runtime_path(norm)
            and not _git().mode_allows_protected_write(_git()._current_runtime_mode())
            and not _git()._authorized_managed_update_resolver(ctx)
        ):
            return _git().protected_write_block_message(
                path=norm,
                runtime_mode=_git()._current_runtime_mode(),
                action="write",
            )
        if isinstance(e["content"], str) and e["content"].strip().startswith(_git()._CONTENT_OMITTED_PREFIX):
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
    from ouroboros.tools.edit_ops import _syntax_check, workspace_edit_note

    if mode != "append":  # an append chunk is not a full file — the guard would block every chunk
        for e, binding in zip(write_list, binding_items):
            rel_path = _git()._binding_repo_rel(binding)
            syntax_err = _syntax_check(rel_path, e["content"])
            if not syntax_err:
                continue
            if force:
                syntax_bypass_notes.append(f"{rel_path}: {syntax_err}")
                continue
            return publish_no_effect(ctx, (
                f"⚠️ WRITE_BLOCKED_SYNTAX: {syntax_err} for '{e['path']}'. "
                "Nothing was written. Fix the content, or pass force=true for an "
                "intentionally invalid file."
            ), tool_name="write_file")

    written = []
    written_paths: List[str] = []
    overwrite_diffs: List[str] = []
    capture_notes: List[str] = []
    from ouroboros.workspace_file_outputs import capture_known_workspace_outputs

    for e, binding in zip(write_list, binding_items):
        rel_path = _git()._binding_repo_rel(binding)
        # Append can only grow a file, so the truncation shrink-guard does not apply.
        shrink_warning = None if mode == "append" else _git()._check_shrink_guard(
            binding, e["content"], force=force,
        )
        if shrink_warning:
            if written:
                _git()._invalidate_advisory(
                    ctx,
                    changed_paths=written_paths,
                    mutation_root=binding_items[0].base_path,
                    source_tool="write_file",
                )
            return "\n".join([shrink_warning, *([f"Successfully written: {', '.join(written)}"] if written else []), *capture_notes])
        try:
            target = binding.target_path
            target.parent.mkdir(parents=True, exist_ok=True)
            if mode == "append":
                with target.open("a", encoding="utf-8", newline="") as fh:
                    fh.write(e["content"])  # append is intentionally NOT atomized
                written.append(f"{display_root}:{rel_path} (+{len(e['content'])} chars appended)")
                written_paths.append(rel_path)
                note = capture_known_workspace_outputs(
                    ctx, binding.base_path, [rel_path], source_tool="write_file", include_preamble=not capture_notes,
                )
                if note:
                    capture_notes.append(note)
                continue
            old_content: Optional[str] = None
            if target.exists():
                try:
                    old_content = target.read_text(encoding="utf-8")
                except Exception:
                    old_content = None
            _git().write_text(target, e["content"])
            written.append(f"{display_root}:{rel_path} ({len(e['content'])} chars)")
            written_paths.append(rel_path)
            note = capture_known_workspace_outputs(
                ctx, binding.base_path, [rel_path], source_tool="write_file", include_preamble=not capture_notes,
            )
            if note:
                capture_notes.append(note)
            if old_content is not None and old_content != e["content"]:
                from ouroboros.tools.edit_ops import _unified_diff

                overwrite_diffs.append(_unified_diff(rel_path, old_content, e["content"], cap=120))
        except Exception as exc:
            if written:
                _git()._invalidate_advisory(
                    ctx,
                    changed_paths=written_paths,
                    mutation_root=binding_items[0].base_path,
                    source_tool="write_file",
                )
            already = ", ".join(written) if written else "(none)"
            return (
                f"⚠️ FILE_WRITE_ERROR on '{e['path']}': {exc}\n"
                f"Successfully written before error: {already}"
                + ("\n" + "\n".join(capture_notes) if capture_notes else "")
            )

    _git()._invalidate_advisory(
        ctx,
        changed_paths=written_paths,
        mutation_root=binding_items[0].base_path,
        source_tool="write_file",
    )
    summary = ", ".join(written)
    system_target = _git()._binding_targets_system_repo(ctx, binding_items[0])
    if capture_notes:
        result = f"✅ Written {len(written)} file(s): {summary}\n" + "\n".join(capture_notes)
    elif ctx.is_workspace_mode() and not system_target:
        result = (
            f"✅ Written {len(written)} file(s): {summary}\n"
            "Files are on disk in the active workspace. " + workspace_edit_note(ctx)
        )
    elif not system_target:
        result = (
            f"✅ Written {len(written)} file(s): {summary}\n"
            "Files are on disk in the active workspace."
        )
    else:
        result = (
            f"✅ Written {len(written)} file(s): {summary}\n"
            "Files are on disk but NOT committed. Run commit_reviewed when ready."
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
    book_note = book_balance_note(binding_items[0].base_path, written_paths) if system_target else ""
    if book_note:
        result += "\n" + book_note
    protected_written = _git().protected_paths_in(written_paths) if system_target else []
    if protected_written and _git().mode_allows_protected_write(_git()._current_runtime_mode()):
        result += "\n\n" + _git().core_patch_notice(protected_written)
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
        return publish_no_effect(ctx, "⚠️ STR_REPLACE_ERROR: path is required.", tool_name="edit_text")
    if not old_str:
        return publish_no_effect(ctx, "⚠️ STR_REPLACE_ERROR: old_str is required (cannot be empty).", tool_name="edit_text")

    existing_tc = _git().normalize_task_constraint(getattr(ctx, "task_constraint", None))
    data_skill_target = None
    task_constraint = existing_tc
    short_form = None
    binding = _resolved_binding
    if binding is not None:
        target = binding.target_path
        invalidation_root = binding.base_path
    elif not ctx.is_workspace_mode():
        short_form = _git().decide_payload_short_form(
            bucket=bucket,
            skill_name=skill_name,
            path_text=path,
            repo_dir=pathlib.Path(ctx.repo_dir),
            drive_root=pathlib.Path(ctx.drive_root),
        )
        if short_form.error:
            return f"⚠️ STR_REPLACE_ERROR: {short_form.error}"
        synth = short_form.constraint
        redirect_err = _git().cross_skill_redirect_error(existing_tc, synth)
        if redirect_err:
            return f"⚠️ SKILL_REDIRECT_BLOCKED: {redirect_err}"
        task_constraint = existing_tc if existing_tc and existing_tc.has_selected_skill else synth or existing_tc

    if binding is None and not ctx.is_workspace_mode() and task_constraint and task_constraint.has_selected_skill and task_constraint.payload_root:
        try:
            target = _git().resolve_payload_path(pathlib.Path(ctx.drive_root), task_constraint, path)
            data_skill_target = target
        except ValueError as e:
            return f"⚠️ STR_REPLACE_ERROR: {e}"
        if _git().is_skill_control_plane_path(target, pathlib.Path(ctx.drive_root).resolve(strict=False)):
            return (
                "⚠️ STR_REPLACE_BLOCKED: skill provenance, launcher seed, "
                "marketplace, dependency, and self-authored markers are "
                "control-plane state. Edit user-authored payload files instead."
            )
        invalidation_root = pathlib.Path(ctx.drive_root)
    elif binding is None and not ctx.is_workspace_mode():
        data_skill_target = _git()._data_skill_path(path, pathlib.Path(ctx.drive_root))
        if data_skill_target is not None:
            if _git().is_skill_control_plane_path(data_skill_target, pathlib.Path(ctx.drive_root).resolve(strict=False)):
                return (
                    "⚠️ STR_REPLACE_BLOCKED: skill provenance, launcher seed, "
                    "marketplace, dependency, and self-authored markers are "
                    "control-plane state. Edit user-authored payload files instead."
                )
            target = data_skill_target
            invalidation_root = pathlib.Path(ctx.drive_root)
    if binding is None and data_skill_target is None:
        try:
            binding = _git().build_resolved_resource_binding(
                ctx, root=display_root, operation="edit", path=path,
            )
        except Exception as exc:
            return f"⚠️ PATH_ERROR: {exc}"
        target = binding.target_path
        invalidation_root = binding.base_path

    rel_path = _git()._binding_repo_rel(binding) if binding is not None else _git().safe_relpath(path)
    system_target = bool(binding and _git()._binding_targets_system_repo(ctx, binding))
    norm = _git().normalize_repo_path(rel_path)
    if (
        system_target
        and _git().is_protected_runtime_path(norm)
        and not _git().mode_allows_protected_write(_git()._current_runtime_mode())
        and not _git()._authorized_managed_update_resolver(ctx)
    ):
        return _git().protected_write_block_message(
            path=norm,
            runtime_mode=_git()._current_runtime_mode(),
            action="edit",
        )

    if not target.exists():
        return publish_no_effect(ctx, f"⚠️ STR_REPLACE_ERROR: file not found: {path}", tool_name="edit_text")

    try:
        content = target.read_text(encoding="utf-8")
    except Exception as e:
        return publish_no_effect(ctx, f"⚠️ STR_REPLACE_ERROR: cannot read {path}: {e}", tool_name="edit_text")

    # Shared exact-match single-replacement (deferral 4): identical count==0/count>1
    # feedback for the repo and data-plane editors.
    new_content, _match_err = _git()._str_match_replace(content, old_str, new_str, path, "STR_REPLACE_ERROR")
    if _match_err:
        return publish_no_effect(ctx, _match_err, tool_name="edit_text")
    if data_skill_target is not None:
        # Deferral 5: a data-plane skill payload edited via the active_workspace route gets
        # the SAME shrink guard as the root=skill_payload editor — no silent >30% truncation
        # of a payload file. (Intentional large rewrites go through root=skill_payload, which
        # carries the force escape hatch.)
        from ouroboros.tools.core import _check_data_shrink_guard

        _shrink_block = _check_data_shrink_guard(target, new_content, force)
        if _shrink_block:
            return publish_no_effect(ctx, _shrink_block, tool_name="edit_text")
    elif binding is not None:
        _shrink_block = _git()._check_shrink_guard(binding, new_content, force)
        if _shrink_block:
            return publish_no_effect(ctx, _shrink_block, tool_name="edit_text")
    # X3 hash-bind: the ADMITTED repair task's payload edits CAS-check the
    # repair's own hash chain; drift outside the repair is a typed stale
    # terminalization, never a silent write over foreign changes.
    _repair_cas_constraint = (
        task_constraint
        if task_constraint and task_constraint.has_selected_skill
        and (data_skill_target is not None or (binding is not None and binding.skill_name))
        and str(getattr(task_constraint, "skill_name", "") or "")
        else None
    )
    if _repair_cas_constraint is not None:
        from ouroboros.skill_repair_admission import repair_write_cas_error

        _cas = repair_write_cas_error(
            canonical_data_root(ctx), _repair_cas_constraint,
            task_id=str(getattr(ctx, "task_id", "") or ""),
            # Mandatory only for a real repair TASK; a synthesized short-form
            # selector on an ordinary edit lane is not an admitted repair.
            repair_task=bool(existing_tc and existing_tc.has_selected_skill))
        if _cas:
            return _cas
    try:
        _git().write_text(target, new_content)
    except Exception as e:
        return f"⚠️ STR_REPLACE_ERROR: write failed for {path}: {e}"
    from ouroboros.workspace_file_outputs import capture_known_workspace_outputs

    capture_note = capture_known_workspace_outputs(
        ctx, binding.base_path, [rel_path], source_tool="edit_text",
    ) if binding is not None else ""
    if _repair_cas_constraint is not None:
        from ouroboros.skill_repair_admission import advance_repair_expected_hash

        advance_repair_expected_hash(
            canonical_data_root(ctx), _repair_cas_constraint,
            task_id=str(getattr(ctx, "task_id", "") or ""))

    replacement_line = new_content[:new_content.index(new_str)].count('\n') + 1
    context_start = max(0, replacement_line - 3)
    context_lines = new_content.splitlines()[context_start:replacement_line + len(new_str.splitlines()) + 2]
    context_preview = "\n".join(
        f"{context_start + i + 1:>4}| {line}" for i, line in enumerate(context_lines)
    )

    _git()._invalidate_advisory(
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
    if capture_note:
        result += "\n" + capture_note
    elif data_skill_target is None and ctx.is_workspace_mode() and not system_target:
        from ouroboros.tools.edit_ops import workspace_edit_note
        result += "\n" + workspace_edit_note(ctx)
    elif system_target:
        result += "\nRun commit_reviewed when ready."
    elif data_skill_target is not None:
        result += "\nRun skill_review for this skill before enabling or declaring it ready."
    if system_target and pathlib.PurePosixPath(rel_path).parts[:1] == ("skills",):
        result += (
            "\nℹ️ Native seed boundary: system_repo/skills changed; the installed "
            "data/skills/native copy remains unchanged until launcher reseed."
        )
    book_note = book_balance_note(binding.base_path, [rel_path]) if system_target and binding is not None else ""
    if book_note:
        result += "\n" + book_note
    if system_target and _git().is_protected_runtime_path(norm) and _git().mode_allows_protected_write(_git()._current_runtime_mode()):
        result += "\n\n" + _git().core_patch_notice([norm])
    return result
