"""Advanced repo editing tools: apply_patch and edit_batch.

Two editing primitives beyond exact-match ``edit_text`` and full-file
``write_file``:

- ``apply_patch``  — context-anchored multi-file diff (V4A-style, no line
  numbers): hunks locate themselves by surrounding lines plus optional ``@@``
  anchors. Atomic VALIDATION across all files/hunks: any unmatched hunk aborts
  the whole patch before a single byte is written, with per-hunk diagnostics.
- ``edit_batch``   — batch of COUNTED exact replacements, validated as a whole.
  Each edit declares how many occurrences it expects (default 1); a count
  mismatch aborts the whole batch before anything is written. This is the safe
  form of "replace all".

Atomicity is over VALIDATION, not the write phase: nothing is written until every
file, hunk and count resolves, but the writes themselves are a per-file sequence,
so a mid-write I/O fault can leave earlier files applied. That case discloses
itself (``EDIT_OPS_PARTIAL_WRITE_FAILED``), names the written files and marks the
advisory snapshot stale.

Both target the repo lanes only (active_workspace / system_repo) and reuse the
same guard chain as ``edit_text``: path canonicalization FIRST (see
``_resolve_edit_target`` — a guard that judges a different spelling than the
write uses is not a guard), then root access, protected artifact paths,
project-room write guard, protected runtime paths. Because their paths ride
inside the payload rather than a ``path`` arg, the dispatch gates in
``registry.py`` read them back out through ``_payload_write_paths`` so the
acting-subagent and protected-write fences apply identically.

(An ``edit_sketch`` fast-apply tool — strong-model sketch merged by the cheap
LIGHT model — lived here through the editbench evaluation and was removed: the
sketch/apply split never beat the direct tools on either cost or robustness;
see devtools/benchmarks/editbench/README.md. Its useful rails — unified diff
in the result and a pre-write syntax check — moved into write_file.)

``_syntax_check`` and ``_unified_diff`` are shared helpers, also used by the
repo write path (git._repo_write).

Newlines follow the existing repo-write lane rather than diverging from it: the
lane reads with universal newlines and writes ``\n``, so a CRLF file is rewritten
LF-only — by ``edit_text`` and ``write_file`` today, and by these tools for the
same reason. Stated here because a patch tool implies surgical byte fidelity; a
lane-wide newline contract is not this module's to change.
"""

from __future__ import annotations

import logging
import pathlib
from typing import Any, Dict, List, Optional, Tuple

from ouroboros.config import get_runtime_mode
from ouroboros.runtime_mode_policy import (
    core_patch_notice,
    is_protected_runtime_path,
    mode_allows_protected_write,
    normalize_repo_path,
    protected_paths_in,
    protected_write_block_message,
)
from ouroboros.tool_access import (
    ResolvedResourceBinding,
    binding_targets_system_repo,
    build_resolved_resource_binding,
)
from ouroboros.patch_core import (  # noqa: F401 -- names re-exported for callers and tests
    _apply_hunks_to_text,
    _find_sequence,
    _FileOp,
    _Hunk,
    _line_positions,
    _parse_patch,
    plan_patch,
    _strip_directive_tail,
    _syntax_check,
    _unified_diff,
    patch_target_paths,
)
from ouroboros.tools.registry import ToolContext, ToolEntry, active_repo_dir_for
from ouroboros.utils import safe_relpath, write_text

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared target resolution (mirrors the edit_text guard chain)
# ---------------------------------------------------------------------------

def _resolve_edit_target(
    ctx: ToolContext,
    path: str,
    root: str,
    *,
    error_tag: str,
    _resolved_binding: ResolvedResourceBinding | None = None,
) -> Tuple[Optional[pathlib.Path], str, Optional[ResolvedResourceBinding], str]:
    """Resolve ``path`` under ``root`` with the same guards as edit_text.

    Returns ``(target, canonical_rel, binding, "")`` on success or an empty
    target/identity/binding plus the typed error on refusal.

    The canonical rel is returned, not just used internally: it is the file's
    IDENTITY. Callers plan, dedup, diagnose and invalidate by it, so two
    spellings of one file inside a single call collapse to one entry instead of
    two writes where the last silently discards the first.
    """
    from ouroboros.tools.core import (
        _access_or_block,
        project_room_lens_dir,
    )

    if not path or not str(path).strip():
        return None, "", None, f"⚠️ {error_tag}: path is required."
    normalized, block = _access_or_block(ctx, root, "edit")
    if block:
        return None, "", None, block
    if normalized not in {"active_workspace", "system_repo"}:
        return None, "", None, (
            f"⚠️ {error_tag}: root={normalized!r} is not supported; "
            "these tools edit repo lanes only (active_workspace / system_repo). "
            "Use write_file/edit_text for data-plane roots."
        )
    try:
        binding = _resolved_binding or build_resolved_resource_binding(
            ctx, root=normalized, operation="edit", path=path,
        )
    except Exception as exc:  # noqa: BLE001 - target selection must fail closed
        return None, "", None, f"⚠️ {error_tag}: {type(exc).__name__}: {exc}"
    if binding.root != normalized:
        return None, "", None, (
            f"⚠️ {error_tag}: internal target binding root mismatch "
            f"({binding.root!r} != {normalized!r})."
        )
    target = pathlib.Path(binding.target_path)
    try:
        rel = target.relative_to(binding.base_path).as_posix()
    except ValueError:
        return None, "", None, f"⚠️ {error_tag}: selected target escapes its repository root."
    from ouroboros.protected_artifacts import block_reason_for_path

    if reason := block_reason_for_path(ctx, target, "write", binding):
        return None, "", None, (
            f"⚠️ {error_tag}: protected artifact path blocked: {reason}"
        )
    if normalized == "active_workspace" and project_room_lens_dir(ctx) is not None:
        return None, "", None, (
            "⚠️ ROOM_WRITE_VIA_TASK: this room's files are edited by PROMOTED tasks — "
            "call promote_chat_to_task for real work there. For a deliberate edit of "
            'the Ouroboros system repo, pass root="system_repo" explicitly.'
        )
    norm = normalize_repo_path(rel)
    if (
        binding_targets_system_repo(ctx, binding)
        and is_protected_runtime_path(norm)
        and not mode_allows_protected_write(_runtime_mode())
        # The assisted managed-update resolver edits whatever official file the
        # merge conflicts on; git._repo_write and _str_replace_editor both carry
        # this exemption, so withholding it here would make these tools the one
        # lane that cannot finish a conflict resolution.
        and not _authorized_resolver(ctx)
    ):
        return None, "", None, protected_write_block_message(
            path=norm, runtime_mode=_runtime_mode(), action="edit"
        )
    return target, safe_relpath(rel), binding, ""


def _authorized_resolver(ctx: ToolContext) -> bool:
    try:
        from ouroboros.tools.registry import _authorized_managed_update_resolver

        return bool(_authorized_managed_update_resolver(ctx))
    except Exception:
        return False


def _runtime_mode() -> str:
    try:
        return get_runtime_mode()
    except Exception:
        return "advanced"


def _finish_mutation(
    ctx: ToolContext,
    changed_paths: List[str],
    source_tool: str,
    binding: ResolvedResourceBinding | None = None,
) -> str:
    """Advisory invalidation + the standard commit/patch-artifact footer."""
    from ouroboros.tools.commit_gate import _invalidate_advisory

    try:
        _invalidate_advisory(
            ctx,
            changed_paths=changed_paths,
            mutation_root=(binding.base_path if binding is not None else active_repo_dir_for(ctx)),
            source_tool=source_tool,
        )
    except Exception:
        log.debug("%s: advisory invalidation failed (non-critical)", source_tool, exc_info=True)
    targets_system = binding_targets_system_repo(ctx, binding) if binding is not None else False
    if ctx.is_workspace_mode() and not targets_system:
        return "Files are on disk but NOT committed. Do not commit; the headless runner will emit a patch artifact."
    footer = (
        "Files are on disk but NOT committed. Run commit_reviewed when ready.\n"
        "⚠️ Advisory pre-review is now stale — run advisory_review before commit_reviewed."
    )
    # A pro-mode edit of a protected surface announces itself here exactly as it
    # does from git._repo_write / _str_replace_editor (SYSTEM.md's protected-write
    # contract): the mode ALLOWS the write, and the notice is what keeps it visible.
    protected = protected_paths_in(changed_paths) if targets_system or not ctx.is_workspace_mode() else []
    if protected and mode_allows_protected_write(_runtime_mode()):
        footer += "\n\n" + core_patch_notice(protected)
    return footer


def _partial_write_failure(
    ctx: ToolContext,
    changed_paths: List[str],
    source_tool: str,
    tag: str,
    detail: str,
    binding: ResolvedResourceBinding | None = None,
) -> str:
    """Report an I/O failure that landed AFTER some files were already written.

    Validation is atomic — nothing is written until every file and hunk resolves
    — but the write phase itself is a sequence of per-file writes, so a disk
    error mid-sequence leaves the earlier files applied. Those files are real
    worktree mutations, so the advisory snapshot must go stale here exactly as it
    would on success; otherwise `commit_reviewed` would accept them against a
    pre-review taken before they existed. The residual is disclosed, not hidden.
    """

    if changed_paths:
        _finish_mutation(ctx, changed_paths, source_tool, binding)
        # NOT the tools' own *_ERROR prefix: those read as validation refusals
        # (a counted/context miss) and are classified as policy denials. This is a
        # genuine partial mutation from an I/O fault and must stay an execution
        # failure, so it carries its own prefix and lands in the generic `error`.
        return (
            f"⚠️ EDIT_OPS_PARTIAL_WRITE_FAILED ({tag}): {detail}\n"
            f"PARTIALLY APPLIED — these files WERE written: {', '.join(changed_paths)}. "
            "Re-read them before retrying; advisory pre-review is now stale."
        )
    return f"⚠️ {tag}: {detail}\nNothing was written."




# ---------------------------------------------------------------------------
# apply_patch
# ---------------------------------------------------------------------------

















def _apply_patch(
    ctx: ToolContext,
    patch: str,
    root: str = "active_workspace",
    _resolved_binding: ResolvedResourceBinding | tuple[ResolvedResourceBinding, ...] | None = None,
) -> str:
    if not patch or not patch.strip():
        return "⚠️ APPLY_PATCH_ERROR: patch is required."
    ops, err = _parse_patch(patch)
    if err:
        return err

    # Phase 1 lives in `patch_core.plan_patch` — the SAME planner the target-native
    # route runs. Atomicity is a property of the resolve-everything-then-write ORDER,
    # so it is decided in one place and this route supplies only its own doors: the
    # binding-aware target resolver, and Home's filesystem.
    supplied_bindings = (
        tuple(_resolved_binding)
        if isinstance(_resolved_binding, tuple)
        else ((_resolved_binding,) if _resolved_binding is not None else ())
    )
    if supplied_bindings and len(supplied_bindings) != len(ops):
        return "⚠️ APPLY_PATCH_ERROR: internal target binding count mismatch."
    binding_iter = iter(supplied_bindings)
    targets: Dict[str, pathlib.Path] = {}
    state: Dict[str, Any] = {"binding": None}

    def _resolve(raw_path: str) -> Tuple[str, str]:
        target, rel, item_binding, terr = _resolve_edit_target(
            ctx, raw_path, root,
            error_tag="APPLY_PATCH_BLOCKED",
            _resolved_binding=next(binding_iter, None),
        )
        if terr:
            return "", terr
        targets[rel] = target
        state["binding"] = state["binding"] or item_binding
        return rel, ""

    def _read(rel: str) -> Tuple[str, str]:
        try:
            return targets[rel].read_text(encoding="utf-8"), ""
        except Exception as e:  # noqa: BLE001 - report unreadable target
            return "", f"⚠️ APPLY_PATCH_ERROR: cannot read {rel}: {e}"

    plan = plan_patch(
        ops,
        resolve=_resolve,
        exists=lambda rel: targets[rel].exists(),
        read_text=_read,
    )
    if plan.error:
        return plan.error
    mutation_binding = state["binding"]

    # Phase 2: write.
    changed_paths: List[str] = []
    for rel, content in plan.writes:
        try:
            write_text(targets[rel], content)
        except Exception as e:  # noqa: BLE001 - surface the failed path
            return _partial_write_failure(
                ctx, changed_paths, "apply_patch", "APPLY_PATCH_ERROR",
                f"write failed for {rel}: {e}",
                mutation_binding,
            )
        changed_paths.append(rel)
    for rel in plan.deletes:
        try:
            targets[rel].unlink()
        except Exception as e:  # noqa: BLE001 - surface the failed path
            return _partial_write_failure(
                ctx, changed_paths, "apply_patch", "APPLY_PATCH_ERROR",
                f"delete failed for {rel}: {e}",
                mutation_binding,
            )
        changed_paths.append(rel)

    footer = _finish_mutation(ctx, changed_paths, "apply_patch", mutation_binding)
    body = "\n".join(plan.summaries)
    if plan.notes:
        body += "\nNotes:\n" + "\n".join("  " + n for n in plan.notes)
    return f"{body}\n{footer}"


# ---------------------------------------------------------------------------
# edit_batch
# ---------------------------------------------------------------------------

def _edit_batch(
    ctx: ToolContext,
    edits: List[Dict[str, Any]],
    root: str = "active_workspace",
    _resolved_binding: ResolvedResourceBinding | tuple[ResolvedResourceBinding, ...] | None = None,
) -> str:
    if not edits or not isinstance(edits, list):
        return "⚠️ EDIT_BATCH_ERROR: edits must be a non-empty array."
    contents: Dict[str, str] = {}
    targets: Dict[str, pathlib.Path] = {}
    applied: List[str] = []
    errors: List[str] = []
    supplied_bindings = (
        tuple(_resolved_binding)
        if isinstance(_resolved_binding, tuple)
        else ((_resolved_binding,) if _resolved_binding is not None else ())
    )
    binding_iter = iter(supplied_bindings)
    mutation_binding: ResolvedResourceBinding | None = None
    for idx, edit in enumerate(edits, 1):
        if not isinstance(edit, dict):
            errors.append(f"edit {idx}: must be an object")
            continue
        item_binding = next(binding_iter, None)
        path = str(edit.get("path", "") or "")
        old_str = edit.get("old_str", "")
        new_str = edit.get("new_str", "")
        if not isinstance(old_str, str) or not old_str:
            errors.append(f"edit {idx} ({path or '?'}): old_str is required (non-empty string)")
            continue
        if not isinstance(new_str, str):
            errors.append(f"edit {idx} ({path or '?'}): new_str must be a string")
            continue
        try:
            count = int(edit.get("count", 1))
        except (TypeError, ValueError):
            errors.append(f"edit {idx} ({path or '?'}): count must be an integer")
            continue
        if count < 1:
            errors.append(f"edit {idx} ({path or '?'}): count must be >= 1")
            continue
        # Resolve BEFORE keying: the canonical rel is the file's identity, so two
        # spellings of one file in a single batch share one buffer instead of two
        # that overwrite each other.
        target, rel, item_binding, terr = _resolve_edit_target(
            ctx,
            path,
            root,
            error_tag="EDIT_BATCH_BLOCKED",
            _resolved_binding=item_binding,
        )
        if terr:
            errors.append(f"edit {idx}: {terr.lstrip('⚠️ ')}")
            continue
        mutation_binding = mutation_binding or item_binding
        if rel not in contents:
            if not target.exists():
                errors.append(f"edit {idx} ({rel}): file not found")
                continue
            try:
                contents[rel] = target.read_text(encoding="utf-8")
            except Exception as e:  # noqa: BLE001 - report unreadable target
                errors.append(f"edit {idx} ({rel}): cannot read: {e}")
                continue
            targets[rel] = target
        text = contents[rel]
        occurrences = text.count(old_str)
        if occurrences != count:
            positions = _line_positions(text, old_str)
            where = f" (at: {', '.join(positions)})" if positions else ""
            errors.append(
                f"edit {idx} ({rel}): old_str occurs {occurrences} time(s), expected {count}{where}. "
                "Re-read the file and set count to the exact number of occurrences you intend to replace."
            )
            continue
        contents[rel] = text.replace(old_str, new_str)
        applied.append(f"edit {idx} ({rel}): replaced {count} occurrence(s)")
    if errors:
        return (
            "⚠️ EDIT_BATCH_ERROR: batch aborted, NOTHING was written (atomic). Problems:\n"
            + "\n".join("  - " + e for e in errors)
        )
    changed: List[str] = []
    for rel, text in contents.items():
        try:
            write_text(targets[rel], text)
        except Exception as e:  # noqa: BLE001 - surface the failed path
            return _partial_write_failure(
                ctx, changed, "edit_batch", "EDIT_BATCH_ERROR",
                f"write failed for {rel}: {e}",
                mutation_binding,
            )
        changed.append(rel)
    footer = _finish_mutation(ctx, changed, "edit_batch", mutation_binding)
    return (
        f"✅ edit_batch applied {len(applied)} edit(s) across {len(changed)} file(s):\n"
        + "\n".join("  " + a for a in applied)
        + f"\n{footer}"
    )


# ---------------------------------------------------------------------------
# shared verification helpers (also used by git._repo_write)
# ---------------------------------------------------------------------------





# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def get_tools() -> List[ToolEntry]:
    return [
        ToolEntry("apply_patch", {
            "name": "apply_patch",
            "description": (
                "Apply a context-anchored multi-file patch (no line numbers). Validation is "
                "atomic: every file and hunk must resolve before ANYTHING is written, so "
                "an unmatched hunk aborts the whole patch untouched (a mid-write disk "
                "error is the one case that can leave earlier files applied, and it says "
                "so). Format:\n"
                "*** Begin Patch\n"
                "*** Update File: relative/path.py\n"
                "@@ def nearest_function\n"
                " context line (starts with a space)\n"
                "-removed line\n"
                "+added line\n"
                "*** Add File: new/file.py\n"
                "+each line of the new file prefixed with +\n"
                "*** Delete File: old/file.py\n"
                "*** End Patch\n"
                "Hunks locate themselves by their exact context lines (copy them from "
                "read_file); the optional @@ anchor disambiguates repeated contexts. "
                "Prefer this over many edit_text calls for scattered multi-file changes. "
                "NOT for rewrites touching most of a file — there the patch grows as "
                "large as the file itself; use write_file instead."
            ),
            "parameters": {"type": "object", "properties": {
                "patch": {"type": "string", "description": "The full patch text (envelope lines optional)."},
                "root": {"type": "string", "enum": ["active_workspace", "system_repo"], "default": "active_workspace"},
            }, "required": ["patch"]},
        }, _apply_patch, is_code_tool=True, mutates_worktree=True),
        ToolEntry("edit_batch", {
            "name": "edit_batch",
            "description": (
                "Batch of COUNTED exact replacements across one or more files. "
                "Each edit replaces ALL occurrences of old_str in its file and declares "
                "the exact number it expects via count (default 1). Any count mismatch "
                "aborts the WHOLE batch before anything is written, with per-edit "
                "diagnostics (a mid-write disk error is the one case that can leave "
                "earlier files applied, and it says so) — read the file(s) "
                "first and state counts you verified. This is the safe 'replace all': "
                "use count>1 for identical repeated edits instead of many edit_text calls."
            ),
            "parameters": {"type": "object", "properties": {
                "edits": {"type": "array", "items": {"type": "object", "properties": {
                    "path": {"type": "string"},
                    "old_str": {"type": "string"},
                    "new_str": {"type": "string"},
                    "count": {"type": "integer", "default": 1,
                              "description": "Exact number of occurrences expected AND replaced."},
                }, "required": ["path", "old_str", "new_str"]}},
                "root": {"type": "string", "enum": ["active_workspace", "system_repo"], "default": "active_workspace"},
            }, "required": ["edits"]},
        }, _edit_batch, is_code_tool=True, mutates_worktree=True),
    ]
