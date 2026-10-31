"""Advanced editing tools: apply_patch and edit_batch.

Two editing primitives beyond exact-match ``edit_text`` and full-file
``write_file``:

- ``apply_patch``  — context-anchored multi-file diff (V4A-style, no line
  numbers): every hunk is placed against the whole original file by its exact
  context (else ignoring trailing whitespace only) at or after an optional ``@@``
  anchor's first hit (a lower bound, not a scope); the placement must be unique, an
  ambiguous one is reported with every placement counted. Indentation is never adapted.
- ``edit_batch``   — batch of COUNTED exact replacements, validated as a whole.
  Each edit declares how many occurrences it expects (default 1); a count
  mismatch aborts the whole batch before anything is written. This is the safe
  form of "replace all". ``edit_text`` with ``edits`` reuses it for one file.

Atomicity is over VALIDATION, not the write phase: nothing is written until every
file, hunk and count resolves, but the writes themselves are a per-file sequence,
so a mid-write I/O fault can leave earlier files applied. That case discloses
itself (``EDIT_OPS_PARTIAL_WRITE_FAILED``) with the written, failed and
not-attempted files, and marks the advisory snapshot stale. An explicit
``Delete File`` first stores the file's exact current bytes as an immutable
task artifact; a failed capture refuses the patch before any effect.

Both target the seven writable roots of ``edit_text`` and reuse the same physical
target binding and write policy (``_edit_mutation_block``): a guard that judges
a different spelling than the write uses is not a guard. Because their paths
ride inside the payload rather than a ``path`` arg, the dispatch gates in
``registry.py`` read them back out through ``_payload_write_paths`` so the
acting-subagent and protected-write fences apply identically. A final Python or
JSON parse failure on a repo lane is reported after the write, never a refusal.

``locate_edit_miss`` is the one bounded diagnosis all three exact editors attach
when their needle is not in the file (edit_text through ``core._str_match_replace``):
the closest region, the first differing line and the ``read_file`` window to copy from.
Matching splits on LF; every line number shown is ``read_file``'s.

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

import bisect
import difflib
import json
import logging
import pathlib
import re
import textwrap
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ouroboros.tools.arg_feedback import payload_item_feedback, with_argument_notes
from ouroboros.config import get_runtime_mode
from ouroboros.runtime_mode_policy import (
    core_patch_notice,
    mode_allows_protected_write,
    protected_paths_in,
)
from ouroboros.tool_access import (
    ResolvedResourceBinding,
    binding_targets_system_repo,
    build_resolved_resource_binding,
)
from ouroboros.tools.registry import ToolContext, ToolEntry, active_repo_dir_for
from ouroboros.utils import safe_relpath, write_text

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Payload item vocabulary (shared with core._write_file)
# ---------------------------------------------------------------------------

# The ONE declaration of the `edits` item shape: the published schema in
# get_tools() and the pre-edit guard both DERIVE from it, so the declared shape
# cannot drift from what the tool reads. `count` is optional; the rest are required.
_EDIT_BATCH_ITEM_PROPERTIES: Dict[str, Dict[str, Any]] = {
    "path": {"type": "string"},
    "old_str": {"type": "string"},
    "new_str": {"type": "string"},
    "count": {"type": "integer", "default": 1,
              "description": "Exact number of occurrences expected AND replaced."},
}
_EDIT_BATCH_ITEM_KEYS: Tuple[str, ...] = tuple(_EDIT_BATCH_ITEM_PROPERTIES)
_EDIT_BATCH_ITEM_REQUIRED: Tuple[str, ...] = ("path", "old_str", "new_str")
_FILE_ROOT_DESCRIPTION = (
    "When omitted, relative paths use active_workspace; absolute paths select the permitted "
    "physical root (including user_files for home paths). Explicit roots remain explicit."
)


# ---------------------------------------------------------------------------
# Shared target resolution (mirrors the edit_text guard chain)
# ---------------------------------------------------------------------------

def _edit_mutation_block(ctx: ToolContext, binding: ResolvedResourceBinding,
                         path: str, *, operation: str = "write") -> str:
    """Physical write policy shared by single, batch and patch editors."""
    from ouroboros.config import SETTINGS_PATH, get_runtime_mode
    from ouroboros.tools.core import (
        _native_payload_mutation_block_reason, _binding_skill_control_plane_path,
        is_skill_control_plane_path, _is_skill_owner_state_target,
        is_skill_owner_state_alias, _is_workspace_executor_control_state_path,
        _project_store_access_block, _normalize_data_read_path,
        artifact_store_path_block_reason, is_skill_create_typo,
    )
    from ouroboros.protected_artifacts import block_reason_for_path
    from ouroboros.runtime_mode_policy import mode_has_unrestricted_agency

    target = binding.target_path
    root = binding.root
    if reason := block_reason_for_path(ctx, target, operation, binding):
        return f"protected artifact path blocked: {reason}"
    from ouroboros.runtime_mode_policy import (
        is_protected_runtime_path, mode_allows_protected_write,
        normalize_repo_path, protected_write_block_message,
    )
    from ouroboros.tools.tool_resolution import body_relative_path
    from ouroboros.workspace_copies import is_system_copy

    if (body_path := body_relative_path(ctx, binding, own_body_copy=is_system_copy(ctx))) is not None:
        relative = normalize_repo_path(body_path)
        from ouroboros.tools.registry import _authorized_managed_update_resolver

        if (is_protected_runtime_path(relative)
                and not mode_allows_protected_write(get_runtime_mode())
                and not _authorized_managed_update_resolver(ctx)):
            return protected_write_block_message(path=relative, runtime_mode=get_runtime_mode(), action="edit")
    if root in {"active_workspace", "system_repo"} and binding.source in {
        "active_workspace", "system_repo", "project_room"
    }:
        return ""
    # The shared reservation applies its own mode rule, exactly as write_file reads it.
    if root == "artifact_store":
        if reason := artifact_store_path_block_reason(target, base_path=binding.base_path):
            return f"artifact_store path blocked: {reason}"
    if mode_has_unrestricted_agency(get_runtime_mode()):
        return ""
    data_root = binding.state_drive_root
    if reason := _native_payload_mutation_block_reason(target, data_root):
        return reason
    if (_binding_skill_control_plane_path(binding)
            or is_skill_control_plane_path(target, data_root)):
        return "skill provenance, launcher seed and marketplace control-plane state"
    if (_is_skill_owner_state_target(target, data_root)
            or is_skill_owner_state_alias(target, data_root)):
        return "skill review, enablement, grants and marketplace owner state"
    if (_is_workspace_executor_control_state_path(target, binding.base_path)
            or _is_workspace_executor_control_state_path(target, data_root)):
        return "workspace executor process control-plane state"
    try:
        settings = pathlib.Path(SETTINGS_PATH)
        if (target.exists() and settings.exists() and target.samefile(settings)) or (
            target.parent.resolve() == settings.parent.resolve()
            and target.name.lower() == settings.name.lower()
        ):
            return "settings.json is owner-edited control-plane state"
    except OSError:
        pass
    if root == "runtime_data":
        if block := _project_store_access_block(_normalize_data_read_path(ctx, path)):
            return block
    if (binding.source in {"external", "clawhub", "ouroboroshub"}
            and not binding.base_path.exists()
            and (binding.source != "external" or is_skill_create_typo(
                payload_root=binding.base_path, bucket="external",
                rel_within_payload=target.relative_to(binding.base_path).as_posix()))):
        return "skill payload not found; create an external skill through its manifest first"
    return ""


def _resolve_edit_target(
    ctx: ToolContext,
    path: str,
    root: str,
    *,
    error_tag: str,
    _resolved_binding: ResolvedResourceBinding | None = None,
    operation: str = "edit",
    bucket: str = "",
    skill_name: str = "",
) -> Tuple[Optional[pathlib.Path], str, Optional[ResolvedResourceBinding], str]:
    """Resolve ``path`` under ``root`` with the same guards as edit_text.

    Returns ``(target, canonical_rel, binding, "")`` on success or an empty
    target/identity/binding plus the typed error on refusal.

    The canonical rel is returned, not just used internally: it is the file's
    IDENTITY. Callers plan, dedup, diagnose and invalidate by it, so two
    spellings of one file inside a single call collapse to one entry instead of
    two writes where the last silently discards the first.
    """
    from ouroboros.tools.core import _access_or_block

    if not path or not str(path).strip():
        return None, "", None, f"⚠️ {error_tag}: path is required."
    normalized, block = _access_or_block(ctx, root, operation)
    if block:
        return None, "", None, block
    try:
        binding = _resolved_binding or build_resolved_resource_binding(
            ctx, root=normalized, operation=operation, path=path,
            bucket=bucket, skill_name=skill_name,
        )
    except Exception as exc:  # noqa: BLE001 - target selection must fail closed
        return None, "", None, f"⚠️ {error_tag}: {type(exc).__name__}: {exc}"
    if binding.root != normalized or binding.operation != operation:
        return None, "", None, (
            f"⚠️ {error_tag}: internal target binding root mismatch "
            f"({binding.root!r} != {normalized!r})."
        )
    target = pathlib.Path(binding.target_path)
    try:
        rel = target.relative_to(binding.base_path).as_posix()
    except ValueError:
        return None, "", None, f"⚠️ {error_tag}: selected target escapes its repository root."
    if reason := _edit_mutation_block(ctx, binding, path):
        return None, "", None, f"⚠️ {error_tag}: {reason}"
    return target, safe_relpath(rel), binding, ""


def _runtime_mode() -> str:
    try:
        return get_runtime_mode()
    except Exception:
        return "advanced"


def workspace_edit_note(ctx: ToolContext) -> str:
    """Explain capture without inventing broader Git authority."""
    from ouroboros.contracts.task_constraint import normalize_task_constraint

    constraint = normalize_task_constraint(getattr(ctx, "task_constraint", None))
    if constraint is not None and constraint.surface == "self_worktree":
        return "Do not commit this self-worktree; return the captured patch for parent integration."
    return "The headless runner captures a workspace patch; it is not proof of Git commit or publication."


def _finish_mutation(
    ctx: ToolContext,
    changed_paths: List[str],
    source_tool: str,
    binding: ResolvedResourceBinding | None = None,
) -> str:
    """Advisory invalidation + the standard commit/patch-artifact footer."""
    from ouroboros.review import diff_pulse
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
    if not targets_system:
        from ouroboros.workspace_file_outputs import capture_known_workspace_outputs

        capture_note = capture_known_workspace_outputs(
            ctx, binding.base_path if binding is not None else active_repo_dir_for(ctx),
            changed_paths, source_tool=source_tool,
        )
        if capture_note:
            return capture_note
        footer = "Files are on disk but NOT committed."
        if ctx.is_workspace_mode():
            footer += " " + workspace_edit_note(ctx)
        return footer
    footer = "Files are on disk but NOT committed. Run commit_reviewed when ready."
    # A pro-mode edit of a protected surface announces itself here exactly as it
    # does from git._repo_write / _str_replace_editor (the protected-write contract
    # in ARCHITECTURE "Safety and runtime mode" and SYSTEM.md "Safety-critical
    # files"): the mode ALLOWS the write, and the notice is what keeps it visible.
    pulse = diff_pulse(binding.base_path, changed_paths)
    footer += f"\n{pulse}" if pulse else ""
    protected = protected_paths_in(changed_paths)
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
    *,
    pending: List[str] = (),
    recovery: List[str] = (),
) -> str:
    """Report an I/O failure that landed AFTER some files were already written.

    Validation is atomic — nothing is written until every file and hunk resolves
    — but the write phase itself is a sequence of per-file writes, so a disk
    error mid-sequence leaves the earlier files applied. Those files are real
    worktree mutations, so the advisory snapshot must go stale here exactly as it
    would on success; otherwise `commit_reviewed` would accept them against a
    pre-review taken before they existed. The residual is disclosed, not hidden:
    changed and not-attempted files are named, and a caller is never told to
    repeat edits that already landed.
    """
    skipped = f"\nNot attempted: {', '.join(pending)}." if pending else ""
    copies = ("\nRecovery copies:\n" + "\n".join(recovery)) if recovery else ""

    if changed_paths:
        if binding is not None and binding.skill_name:
            from ouroboros.contracts.task_constraint import normalize_task_constraint
            from ouroboros.skill_repair_admission import advance_repair_expected_hash

            constraint = normalize_task_constraint(getattr(ctx, "task_constraint", None))
            if constraint and constraint.has_selected_skill:
                advance_repair_expected_hash(binding.state_drive_root, constraint, task_id=ctx.task_id)
        footer = _finish_mutation(ctx, changed_paths, source_tool, binding)
        # NOT the tools' own *_ERROR prefix: those read as validation refusals
        # (a counted/context miss) and are classified as policy denials. This is a
        # genuine partial mutation from an I/O fault and must stay an execution
        # failure, so it carries its own prefix and lands in the generic `error`.
        return (
            f"⚠️ EDIT_OPS_PARTIAL_WRITE_FAILED ({tag}): {detail}\n"
            f"PARTIALLY APPLIED — these files WERE written: {', '.join(changed_paths)}.{skipped}\n"
            "Do not repeat this call: read the written files and continue from their current content."
            f"{copies}\n" + footer
        )
    return f"⚠️ {tag}: {detail}\nNothing was written.{skipped}{copies}"


def _line_positions(text: str, needle: str, limit: int = 5) -> List[str]:
    positions: List[str] = []
    start = 0
    for _ in range(limit):
        idx = text.find(needle, start)
        if idx < 0:
            break
        positions.append(f"line {edit_source_line(text, idx)}")
        start = idx + 1
    return positions


def edit_source_line(text: str, offset: int) -> int:
    """Locate a source position using read_file's splitlines boundary rule."""
    offset = max(0, min(len(text), offset))
    end = 0
    number = 1
    for number, line in enumerate(text.splitlines(keepends=True), 1):
        end += len(line)
        if offset < end:
            return number
    return number


# Line boundaries str.splitlines (and so read_file) honours besides LF.
_OTHER_LINE_BOUNDARIES = frozenset("\r\x0b\x0c\x1c\x1d\x1e\x85\u2028\u2029")


class _ReaderLines:
    """read_file's 1-based line numbers for LF-split line indexes.

    Patch grammar and locator matching split on LF; every number shown to the
    model is the reader's, so a file holding another Unicode line boundary is
    cited exactly where read_file shows it. Plain LF text needs no index.
    """

    def __init__(self, text: str) -> None:
        self._ends: Optional[List[int]] = None
        if _OTHER_LINE_BOUNDARIES.isdisjoint(text):
            return
        self._length, self._ends, self._starts, offset = len(text), [], [], 0
        for line in text.splitlines(keepends=True):
            offset += len(line)
            self._ends.append(offset)
        offset = 0
        for line in text.split("\n"):
            self._starts.append(offset)
            offset += len(line) + 1

    def line(self, index: int) -> int:
        """The reader line where LF line ``index`` (0-based) begins."""
        if self._ends is None:
            return index + 1
        offset = self._starts[index] if index < len(self._starts) else self._length
        return bisect.bisect_right(self._ends, offset) + 1

    def count(self, index: int, lines: int) -> int:
        """Reader lines spanned by ``lines`` LF lines from ``index``."""
        return max(1, self.line(index + lines) - self.line(index)) if self._ends is not None else lines


_EDIT_PREVIEW_LINE_CHARS = 240   # per numbered post-edit row
_EDIT_PREVIEW_MAX_CHARS = 4_000  # whole post-edit preview, site notes included


def numbered_edit_preview(text: str, offsets: List[int], *, max_sites: int = 8) -> str:
    """Bounded numbered post-edit context at actual edited source positions.

    Rows carry read_file's own number-and-tab prefix, so every surface shows
    one line-number form and one line definition. A long row and the whole
    preview are cut through the shared disclosed bound; the site notes survive it.
    """
    from ouroboros.tools.core_file_tools import _display_prefix
    from ouroboros.utils import truncate_within_limit

    if not text:
        return "(empty file after edit; no surviving source lines)"
    lines = text.splitlines()
    chosen: set[int] = set()
    for offset in offsets[:max_sites]:
        line = edit_source_line(text, offset)
        chosen.update(range(max(1, line - 2), min(len(lines), line + 2) + 1))
    rows = "\n".join(_display_prefix(number) + truncate_within_limit(lines[number - 1], _EDIT_PREVIEW_LINE_CHARS)
                     for number in sorted(chosen))
    notes = []
    if any(offset >= len(text) for offset in offsets[:max_sites]):
        notes.append("Edit site is at end of file; nearest surviving source lines shown.")
    if len(offsets) > max_sites:
        notes.append(f"… {len(offsets) - max_sites} more edit site(s)")
    tail = "\n".join(notes)
    room = _EDIT_PREVIEW_MAX_CHARS - len(tail) - bool(tail)
    return "\n".join(part for part in (truncate_within_limit(rows, room), tail) if part)


def newline_normalization_note(target: pathlib.Path) -> str:
    """Disclose the existing universal-newline read and LF write behavior."""
    try:
        raw = target.read_bytes()
    except OSError:
        return ""
    return "⚠️ Existing CRLF/CR line endings were normalized to LF." if b"\r" in raw else ""


def _repo_edit_binding(binding: ResolvedResourceBinding) -> bool:
    """Git recovery applies to repo lanes, never a payload reached by a repo alias."""
    return binding.root in {"active_workspace", "system_repo"} and binding.source in {
        "active_workspace", "system_repo", "project_room"
    }


def _edit_content_block(binding: ResolvedResourceBinding, content: str, force: bool) -> str:
    """The existing shrink policy of the target's lane, before any edit write."""
    from ouroboros.tools.core import _check_data_shrink_guard
    from ouroboros.tools.git_repo_edit import _check_shrink_guard

    if _repo_edit_binding(binding):
        return _check_shrink_guard(binding, content, force) or ""
    return _check_data_shrink_guard(binding.target_path, content, force) or ""


def edit_syntax_note(binding: ResolvedResourceBinding | None, content: str) -> str:
    """Report, never refuse, a final Python/JSON parse failure of an exact edit on a repo lane.

    Moving code can leave a file invalid between steps; full overwrites and
    commits keep their own blocking syntax checks.
    """
    if binding is None or not _repo_edit_binding(binding):
        return ""
    rel = binding.target_path.relative_to(binding.base_path).as_posix()
    syntax = _syntax_check(rel, content)
    return f"⚠️ SYNTAX_WARNING: {syntax} in {rel}; the edit WAS applied." if syntax else ""


# ---------------------------------------------------------------------------
# Edit-miss locator (shared by edit_text / edit_batch / apply_patch)
# ---------------------------------------------------------------------------

_LOCATE_MAX_FILE_LINES = 20_000     # file lines scanned; the rest is disclosed, not read
_LOCATE_MAX_OLD_LINES = 200         # needle lines scored per candidate
_LOCATE_MAX_CANDIDATES = 3          # regions named in one diagnosis
_LOCATE_MAX_ANCHOR_HITS = 50        # anchor-line hits scored before the best is chosen
_LOCATE_MAX_DISTINCT_LINES = 8_000  # distinct lines offered to the similarity fallback
_LOCATE_EXCERPT_LINES = 12          # numbered file lines shown for the chosen region
_LOCATE_LINE_CHARS = 160            # per rendered line
_LOCATE_MAX_CHARS = 2_400           # whole rendered block
_LOCATE_BATCH_MAX = 3               # edit_batch misses located per call
_WHOLE_FILE_PREVIEW_CHARS = 2_000   # a file this small is shown whole (as the old head preview did)


def _norm_ws(line: str) -> str:
    """One whitespace-insensitive spelling of a line (indent, trailing, tabs, runs)."""
    return " ".join(line.split())


def _cut(line: str, limit: int = _LOCATE_LINE_CHARS) -> str:
    return line if len(line) <= limit else line[: limit - 1] + "…"


def _positions(n: int) -> List[str]:
    """Where each needle line sits in its file line: a needle may start or end
    mid-line, so its first line is a SUFFIX of the file line, its last line a
    PREFIX, a lone line a SUBSTRING and an inner line the whole line."""
    return ["single"] if n == 1 else ["first", *["inner"] * (n - 2), "last"]


def _at(file_line: str, needle_line: str, position: str) -> bool:
    if position == "single":
        return needle_line in file_line
    if position == "first":
        return file_line.endswith(needle_line)
    return file_line.startswith(needle_line) if position == "last" else file_line == needle_line


def _excerpt(file_lines: List[str], start: int, count: int, lines: _ReaderLines) -> str:
    from ouroboros.tools.core_file_tools import _display_prefix

    end = min(len(file_lines), start + count)
    shown = min(end - start, _LOCATE_EXCERPT_LINES)
    rows = [_display_prefix(lines.line(start + i)) + _cut(file_lines[start + i]) for i in range(shown)]
    if end - start > shown:
        rows.append(f"… {end - start - shown} more line(s) of the region not shown")
    return "\n".join(rows)


def _first_difference(file_lines: List[str], start: int, old_lines: List[str],
                      *, whole_lines: bool = False) -> Optional[Tuple[int, str, str]]:
    """The first ``(lf_index, file_bytes, needle_bytes)`` where the raw lines disagree."""
    positions = ["inner"] * len(old_lines) if whole_lines else _positions(len(old_lines))
    for k, (needle_line, position) in enumerate(zip(old_lines, positions)):
        if start + k >= len(file_lines):
            return start + k, "<end of file>", needle_line
        if not _at(file_lines[start + k], needle_line, position):
            return start + k, file_lines[start + k], needle_line
    return None


def _whitespace_reason(file_line: str, needle_line: str) -> str:
    if file_line.rstrip() == needle_line.rstrip():
        return "trailing whitespace differs"
    if file_line.strip() == needle_line.strip():
        lead = lambda s: len(s) - len(s.lstrip())  # noqa: E731 - local one-liner
        return f"indentation differs: file {lead(file_line)} leading chars, needle {lead(needle_line)}"
    return "tabs and spaces differ" if file_line.expandtabs() == needle_line.expandtabs() else "whitespace differs inside the line"


_READ_NUMBER = re.compile(r" *(\d+)\t")


def _without_read_numbers(needle: str) -> str:
    """``needle`` without consecutive read_file row prefixes, or ``""`` when it has none."""
    rows = needle.splitlines(keepends=True)
    found = [_READ_NUMBER.match(row) for row in rows]
    if not rows or not all(found):
        return ""
    numbers = [int(match.group(1)) for match in found]
    if numbers != list(range(numbers[0], numbers[0] + len(numbers))):
        return ""
    return "".join(row[match.end():] for row, match in zip(rows, found))


def whole_file_preview(text: str, limit: int = _WHOLE_FILE_PREVIEW_CHARS) -> str:
    """The whole file when small enough; a head cut of a large file previews nothing about a miss."""
    return "" if len(text) > limit else f"\nFile preview (whole file, {len(text)} chars):\n{text}"


def _bounded(block: str, notes: List[str]) -> str:
    out = "\n".join([block, *notes])
    return out if len(out) <= _LOCATE_MAX_CHARS else out[: _LOCATE_MAX_CHARS - 1] + "…"


def locate_edit_miss(
    text: str, old_str: str, *, cursor_line: int = 1, needle_name: str = "old_str",
    max_candidates: int = _LOCATE_MAX_CANDIDATES,
    whole_lines: bool = False,
) -> str:
    """Bounded, actionable diagnosis of why ``old_str`` is not in ``text``: the
    closest region, the FIRST line where the file's bytes differ from the needle
    and the ``read_file`` window to copy from. Tiers, cheapest first: line endings
    (CR/CRLF) → a whitespace-relaxed line match (indentation, trailing, tabs; a
    needle may start or end mid-line) → the nearest region by the needle's most
    distinctive line and difflib similarity. Every tier is bounded (lines scanned,
    candidates scored, excerpt and block size). Hunk diagnostics match whole
    lines; exact replacement diagnostics retain their substring vocabulary."""
    if not old_str:
        return ""
    if not text:
        return f"The file is empty: nothing can match {needle_name}. Use write_file to create its content."
    all_lines = text.split("\n")
    file_lines = all_lines[:_LOCATE_MAX_FILE_LINES]
    notes = ([f"(only the first {_LOCATE_MAX_FILE_LINES} of {len(all_lines)} lines were scanned)"]
             if len(all_lines) > _LOCATE_MAX_FILE_LINES else [])
    if "\r" in old_str or "\r" in text:  # tier 0: universal-newline reads leave the file LF-only
        n_text, n_old = (s.replace("\r\n", "\n").replace("\r", "\n") for s in (text, old_str))
        if n_old.strip() and n_old in n_text:
            index, n_lines = n_text[: n_text.index(n_old)].count("\n"), _ReaderLines(n_text)
            line = n_lines.line(index)
            side = (f"{needle_name} carries CR (\\r) characters the file does not" if "\r" not in text
                    else "the file carries CR (\\r) characters")
            return _bounded(
                f"{needle_name} matches at line {line} once line endings are normalized: {side}. Use LF (\\n) only.\n"
                f"Re-read that region (read_file start_line={line} "
                f"max_lines={n_lines.count(index, n_old.count(chr(10)) + 1)}) "
                f"and copy the exact bytes into {needle_name}.", notes)
    if (unnumbered := _without_read_numbers(old_str)).strip() and unnumbered in text:
        line = edit_source_line(text, text.index(unnumbered))
        return _bounded(
            f"{needle_name} carries read_file line numbers (the number and tab before each line are "
            f"display, not file text). Without them it matches at line {line}: copy only the text "
            "after each line's tab.", notes)
    old_lines = old_str.split("\n")
    while old_lines and not old_lines[0].strip():
        old_lines.pop(0)
    while old_lines and not old_lines[-1].strip():
        old_lines.pop()
    if not old_lines:
        return _bounded(f"{needle_name} is whitespace only; include at least one non-blank line.", notes)
    total = len(old_lines)  # the whole needle; only the window below is compared, and that is said
    old_lines = old_lines[:_LOCATE_MAX_OLD_LINES]
    n = len(old_lines)
    positions = ["inner"] * n if whole_lines else _positions(n)
    if total > n:
        notes.append(f"(only the first {n} of {total} {needle_name} lines were compared; the miss may be after them)")
    needle = [_norm_ws(line) for line in old_lines]
    hay = [_norm_ws(line) for line in file_lines]
    reader = _ReaderLines(text)

    def fits(start: int) -> bool:  # every needle line at its position, whitespace aside (short-circuits)
        return start + n <= len(hay) and all(_at(hay[start + k], needle[k], positions[k]) for k in range(n))

    def matched(start: int) -> int:
        return sum(1 for k in range(n) if start + k < len(hay) and _at(hay[start + k], needle[k], positions[k]))

    def span(start: int) -> str:
        first = reader.line(start)
        last = first + reader.count(start, min(len(file_lines), start + n) - start) - 1
        return f"line {first}" if last == first else f"lines {first}–{last}"

    def render(head: str, start: int, extra: List[str]) -> str:
        diff = _first_difference(file_lines, start, old_lines, whole_lines=whole_lines)
        body = [head, _excerpt(file_lines, start, n, reader)]
        if diff:
            body.append(f"first difference at line {reader.line(diff[0])}:\n"
                        f"  file   : {_cut(diff[1])!r}\n  {needle_name:<7}: {_cut(diff[2])!r}")
        body += extra + [f"Re-read that region (read_file start_line={reader.line(start)} "
                         f"max_lines={reader.count(start, total)}) "
                         f"and copy the exact bytes into {needle_name}."]
        return _bounded("\n".join(body), notes)

    relaxed = [i for i in range(len(hay) - n + 1) if fits(i)][:5]  # tier 1: the same lines, whitespace aside
    if relaxed:
        start = relaxed[0]
        diff = _first_difference(file_lines, start, old_lines, whole_lines=whole_lines)
        reason = (_whitespace_reason(diff[1], diff[2]) if diff
                  else (f"its first {n} lines match exactly; the difference is in the {total - n} lines after them, "
                        "which were not compared") if total > n
                  else "only leading/trailing blank lines differ")
        also = f"; also at lines {', '.join(str(reader.line(i)) for i in relaxed[1:])}" if len(relaxed) > 1 else ""
        return render(f"{needle_name} matches {span(start)} ignoring whitespace ({reason}){also}. "
                      "The file's exact bytes are:", start, [])
    anchor_k = max(range(n), key=lambda k: len(needle[k]))  # tier 2: the needle's most distinctive line
    anchor = needle[anchor_k]
    hits = [i for i, line in enumerate(hay) if anchor and (anchor == line if whole_lines else anchor in line)][:_LOCATE_MAX_ANCHOR_HITS]
    if not hits and len(anchor) >= 3:
        distinct: Dict[str, int] = {}
        for i, line in enumerate(hay):
            if len(line) >= 3 and line not in distinct:
                distinct[line] = i
                if len(distinct) >= _LOCATE_MAX_DISTINCT_LINES:
                    break
        hits = [distinct[c] for c in difflib.get_close_matches(anchor, list(distinct), n=max_candidates, cutoff=0.6)]
    if not hits:
        return _bounded(
            f"No line similar to {needle_name} was found ({len(file_lines)} lines scanned). The text may live in "
            "another file or have changed since you read it: search_code for a distinctive fragment, then re-read.",
            notes)
    scored = sorted((-matched(s), -difflib.SequenceMatcher(None, anchor, hay[min(s + anchor_k, len(hay)-1)]).ratio(),
                     abs(s + 1 - cursor_line), s) for s in {max(0, i - anchor_k) for i in hits})
    score, _similarity, _distance, start = scored[0]
    others = [str(reader.line(s)) for _sc, _sim, _d, s in scored[1:max_candidates]]
    return render(
        f"Nearest region: {span(start)} ({-score} of {n} {needle_name} line(s) match ignoring whitespace):",
        start, [f"Other candidate region(s) start at line(s): {', '.join(others)}."] if others else [])

# ---------------------------------------------------------------------------
# apply_patch
# ---------------------------------------------------------------------------

_PATCH_BEGIN = "*** Begin Patch"
_PATCH_END = "*** End Patch"
_UPDATE_HDR = "*** Update File:"
_ADD_HDR = "*** Add File:"
_DELETE_HDR = "*** Delete File:"
_PATCH_HEADERS = {_UPDATE_HDR: "update", _ADD_HDR: "add", _DELETE_HDR: "delete"}


@dataclass
class _Hunk:
    anchor: str = ""
    lines: List[Tuple[str, str]] = field(default_factory=list)  # (prefix, text)


@dataclass
class _FileOp:
    kind: str  # update | add | delete
    path: str
    hunks: List[_Hunk] = field(default_factory=list)
    add_lines: List[str] = field(default_factory=list)


def _strip_directive_tail(text: str) -> str:
    """Drop decorative trailing asterisks models like to add: '... ***'."""
    return text.strip().rstrip("*").strip()


def _parse_patch(patch: str) -> Tuple[List[_FileOp], str]:
    """Parse the V4A-style patch envelope. Returns (ops, error)."""
    lines = patch.splitlines()
    ops: List[_FileOp] = []
    current: Optional[_FileOp] = None
    seen_end = False
    for lineno, raw in enumerate(lines, 1):
        if seen_end:
            if raw.strip():
                return [], f"⚠️ APPLY_PATCH_ERROR: content after '{_PATCH_END}' (line {lineno})."
            continue
        # Envelope/headers tolerate decorative trailing '***' ("*** Begin Patch ***").
        directive = _strip_directive_tail(raw) if raw.lstrip().startswith("***") else raw.strip()
        if directive == _strip_directive_tail(_PATCH_BEGIN) and raw.lstrip().startswith("***"):
            continue
        if directive == _strip_directive_tail(_PATCH_END) and raw.lstrip().startswith("***"):
            seen_end = True
            continue
        header = next((h for h in _PATCH_HEADERS if raw.startswith(h)), None)
        if header is not None:
            current = _FileOp(_PATCH_HEADERS[header], _strip_directive_tail(raw[len(header):]))
            ops.append(current)
            continue
        if raw.startswith("***"):
            return [], f"⚠️ APPLY_PATCH_ERROR: unrecognized directive at line {lineno}: {raw.strip()!r}."
        if current is None:
            if raw.strip():
                return [], (
                    f"⚠️ APPLY_PATCH_ERROR: content before the first file header "
                    f"(line {lineno}). Start with '{_UPDATE_HDR} <path>'."
                )
            continue
        if current.kind == "add":
            if raw.startswith("+"):
                current.add_lines.append(raw[1:])
            elif not raw.strip():
                current.add_lines.append("")
            else:
                return [], (
                    f"⚠️ APPLY_PATCH_ERROR: Add File body lines must start with '+' "
                    f"(line {lineno}: {raw[:60]!r})."
                )
            continue
        if current.kind == "delete":
            if raw.strip():
                return [], f"⚠️ APPLY_PATCH_ERROR: Delete File takes no body (line {lineno})."
            continue
        # update
        if raw.startswith("@@"):
            current.hunks.append(_Hunk(anchor=raw[2:].strip()))
            continue
        if raw.startswith(("+", "-", " ")) or raw == "":
            if not current.hunks:
                current.hunks.append(_Hunk())
            prefix = raw[:1] if raw else " "
            current.hunks[-1].lines.append((prefix, raw[1:] if raw else ""))
            continue
        return [], (
            f"⚠️ APPLY_PATCH_ERROR: unrecognized hunk line at {lineno}: {raw[:60]!r}. "
            "Hunk lines must start with ' ', '-', '+' or '@@'."
        )
    if not ops:
        return [], (
            "⚠️ APPLY_PATCH_ERROR: no file operations found. Expected headers like "
            f"'{_UPDATE_HDR} <path>' with hunks of ' '/'-'/'+' lines."
        )
    for op in ops:
        if not op.path:
            return [], f"⚠️ APPLY_PATCH_ERROR: {op.kind} header is missing a file path."
        if op.kind == "update" and not any(h.lines for h in op.hunks):
            return [], f"⚠️ APPLY_PATCH_ERROR: Update File {op.path}: no hunk lines."
    return ops, ""


def patch_target_paths(patch: str) -> List[str]:
    """Every file path a patch addresses, derived from the REAL parser.

    The dispatch protected-path gate needs the same targets the handler will
    write. Deriving them from ``_parse_patch`` (rather than a second header
    scanner) is what keeps the gate from drifting: a parse failure returns no
    paths, and the handler refuses that patch before any write.
    """

    ops, err = _parse_patch(patch or "")
    if err:
        return []
    return [op.path for op in ops if op.path]


def normalize_patch_paths(patch: str, normalize) -> str:
    """Rewrite valid file directives only; guards and handler consume the same payload."""
    ops, error = _parse_patch(patch)
    if error:
        return patch
    paths = iter(normalize(op.path) for op in ops)
    lines = patch.splitlines(keepends=True)
    for index, raw in enumerate(lines):
        header = next((h for h in _PATCH_HEADERS if raw.startswith(h)), None)
        if header is not None:
            ending = "\r\n" if raw.endswith("\r\n") else "\n" if raw.endswith("\n") else ""
            lines[index] = f"{header} {next(paths)}{ending}"
    return "".join(lines)


def _context_matches(hay: List[str], needle: List[str], anchors: List[int]) -> List[int]:
    """Every start where ``needle`` matches ``hay`` at or after the FIRST ``@@`` anchor hit (a lower bound, not a scope)."""
    width = len(needle)
    first = min(anchors) if anchors else 0
    return [i for i in range(first, len(hay) - width + 1) if hay[i:i + width] == needle]


_AMBIGUITY_SHOWN, _AMBIGUITY_NAMED = 5, 20  # placements shown with neighbours; further ones named by line


def _ambiguity_detail(file_lines: List[str], starts: List[int], width: int, reader: _ReaderLines) -> str:
    """Every placement counted; the first few with their read_file line range and the
    line just outside each side (the line to extend the context with); the rest named."""
    rows = [f"{len(starts)} placements:"]
    for start in starts[:_AMBIGUITY_SHOWN]:
        first, last = reader.line(start), reader.line(start) + reader.count(start, width) - 1
        before = repr(_cut(file_lines[start - 1].strip(), 80)) if start else "<start of file>"
        after = repr(_cut(file_lines[start + width].strip(), 80)) if start + width < len(file_lines) else "<end of file>"
        rows.append(f"    {f'line {first}' if first == last else f'lines {first}–{last}'}: before {before} | after {after}")
    if rest := starts[_AMBIGUITY_SHOWN:]:
        rows.append(f"    … {len(rest)} more not shown, starting at line(s) "
                    + ", ".join(str(reader.line(start)) for start in rest[:_AMBIGUITY_NAMED])
                    + (f" and {len(rest) - _AMBIGUITY_NAMED} more" if len(rest) > _AMBIGUITY_NAMED else ""))
    return "\n".join(rows)


def _apply_hunks_to_text(
    content: str, hunks: List[_Hunk], path: str, *, _sites: Optional[List[int]] = None,
) -> Tuple[Optional[str], List[str], str]:
    """Validate every hunk against the original file, then apply disjoint spans.

    A repeated Update File directive is the explicit way to chain against a
    preceding result; hunks inside one directive never gain placement from a
    previous hunk. Context matches exactly, else ignoring trailing whitespace
    only, and the placement must be unique. Unchanged context keeps the file's
    bytes and added lines keep the patch's bytes; indentation is never adapted,
    so a miss shows the region and its exact bytes instead.
    """
    file_lines = content.split("\n")
    reader = _ReaderLines(content)
    notes: List[str] = []
    errors: List[str] = []
    changes: List[Tuple[int, int, List[str], int]] = []
    hunk_sites: Dict[int, List[int]] = {}
    old_line_sites: Dict[int, List[int]] = {}
    trimmed: Optional[List[str]] = None
    for hi, hunk in enumerate(hunks, 1):
        old = [t for p, t in hunk.lines if p in (" ", "-")]
        anchors = [i for i, line in enumerate(file_lines) if hunk.anchor in line] if hunk.anchor else []
        if hunk.anchor and not anchors:
            errors.append(f"hunk {hi}: @@ anchor {hunk.anchor!r} not found in {path}")
            continue
        if not old and not hunk.anchor:
            errors.append(f"hunk {hi}: pure insertion needs an @@ anchor or context lines")
            continue
        if not old:
            if len(anchors) != 1:
                errors.append(f"hunk {hi}: @@ anchor {hunk.anchor!r} is ambiguous in {path} — "
                              + _ambiguity_detail(file_lines, anchors, 1, reader)
                              + "\nA pure insertion goes after one anchor line: make the anchor unique or add context lines.")
                continue
            pos = anchors[0] + 1
            changes.append((pos, pos, [t for p, t in hunk.lines if p == "+"], hi))
            hunk_sites[hi] = [0]
            continue
        matches = _context_matches(file_lines, old, anchors)
        trailing = not matches
        if trailing:
            trimmed = trimmed if trimmed is not None else [line.rstrip() for line in file_lines]
            matches = _context_matches(trimmed, [line.rstrip() for line in old], anchors)
        if not matches:
            preview = "\n".join("    " + line for line in old[:6])
            errors.append(
                f"hunk {hi}: context not found in {path}. Hunk expects these consecutive lines:\n"
                f"{preview}\nCopy exact lines from read_file.\n"
                + locate_edit_miss(content, "\n".join(old), needle_name="the hunk context", whole_lines=True)
            )
            continue
        if len(matches) != 1:
            bound = f" at or after the @@ anchor's first hit (line {reader.line(anchors[0])})" if anchors else ""
            errors.append(f"hunk {hi}: context is ambiguous in {path}{bound} — "
                          + _ambiguity_detail(file_lines, matches, len(old), reader)
                          + "\nExtend the context with a line that differs between placements. An @@ anchor is only "
                          "a lower bound (its FIRST hit): it singles out a placement only when none follows it.")
            continue
        pos = matches[0]
        actual = file_lines[pos:pos + len(old)]
        replacement: List[str] = []
        local_sites: List[int] = []
        old_sites: List[int] = []
        old_index = 0
        for prefix, line in hunk.lines:
            if prefix in {" ", "-"}:
                old_sites.append(len(replacement))
            if prefix == " ":
                replacement.append(actual[old_index])
                old_index += 1
            elif prefix == "-":
                local_sites.append(len(replacement))
                old_index += 1
            else:
                local_sites.append(len(replacement))
                replacement.append(line)
        changes.append((pos, pos + len(old), replacement, hi))
        hunk_sites[hi] = local_sites
        old_line_sites[hi] = old_sites
        if trailing:
            notes.append(f"hunk {hi}: context matched ignoring trailing whitespace; "
                         "unchanged context keeps the file's bytes")
    ordered = sorted(changes, key=lambda item: (item[0], item[1]))
    for left, right in zip(ordered, ordered[1:]):
        if right[0] < left[1] or right[0] == left[0]:
            errors.append(f"hunks {left[3]} and {right[3]} overlap in {path}")
    if errors:
        return None, notes, "\n".join(errors)
    # These are placement coordinates, not a second diff alignment. Keep earlier
    # directive sites attached to their surviving lines across explicit chaining.
    points: List[int] = []
    if _sites is not None:
        for site in _sites:
            line, shift = content.count("\n", 0, site), 0
            for start, end, replacement, hi in ordered:
                if start <= line < end:
                    line = start + shift + old_line_sites[hi][line - start]
                    break
                if end <= line:
                    shift += len(replacement) - (end - start)
            else:
                line += shift
            points.append(line)
        shift = 0
        for start, end, replacement, hi in ordered:
            points.extend(start + shift + local for local in hunk_sites[hi])
            shift += len(replacement) - (end - start)
    for start, end, replacement, _ in reversed(ordered):
        file_lines[start:end] = replacement
    if _sites is not None:
        starts, offset = [], 0
        for line in file_lines:
            starts.append(offset)
            offset += len(line) + 1
        _sites[:] = [starts[min(point, len(starts) - 1)] if starts else 0 for point in points]
    return "\n".join(file_lines), notes, ""


def _apply_patch(
    ctx: ToolContext,
    patch: str,
    root: str | None = None,
    bucket: str = "",
    skill_name: str = "",
    force: bool = False,
    _resolved_binding: ResolvedResourceBinding | tuple[ResolvedResourceBinding, ...] | None = None,
) -> str:
    def no_effect(text: str) -> str:
        # Only Phase 1 owns this proof: no planned mutation has run yet.
        from ouroboros.tools.tool_result import LegacyTextResultAdapter, _publish_tool_result, _replace_tool_result

        result = LegacyTextResultAdapter.from_text("apply_patch", text)
        return _publish_tool_result(ctx, _replace_tool_result(
            result, meta_updates={"operation_outcome": "completed_no_effect"}))

    if not patch or not patch.strip():
        return no_effect("⚠️ APPLY_PATCH_ERROR: patch is required.")
    ops, err = _parse_patch(patch)
    if err:
        return no_effect(err)
    from ouroboros.tools.tool_resolution import inferred_file_root

    root, root_error = inferred_file_root(ctx, "apply_patch", root, [op.path for op in ops])
    if root_error:
        return no_effect(root_error)

    # Phase 1: resolve + validate everything BEFORE any write (atomicity).
    planned_writes: List[Tuple[pathlib.Path, str, str]] = []  # (target, rel_path, content)
    planned_deletes: List[Tuple[pathlib.Path, str]] = []
    summaries: List[str] = []
    all_notes: List[str] = []
    errors: List[str] = []
    seen: Dict[str, str] = {}  # rel path -> pending content (chained updates)
    deleted: set[str] = set()
    sites: Dict[str, List[int]] = {}
    bindings: Dict[str, ResolvedResourceBinding] = {}
    supplied_bindings = (
        tuple(_resolved_binding)
        if isinstance(_resolved_binding, tuple)
        else ((_resolved_binding,) if _resolved_binding is not None else ())
    )
    if supplied_bindings and len(supplied_bindings) != len(ops):
        return no_effect("⚠️ APPLY_PATCH_ERROR: internal target binding count mismatch.")
    binding_iter = iter(supplied_bindings)
    mutation_binding: ResolvedResourceBinding | None = None
    for op in ops:
        target, rel, item_binding, terr = _resolve_edit_target(
            ctx,
            op.path,
            root,
            error_tag="APPLY_PATCH_BLOCKED",
            _resolved_binding=next(binding_iter, None),
            operation="edit" if op.kind == "update" else "write",
            bucket=bucket, skill_name=skill_name,
        )
        if terr:
            errors.append(f"{op.path}: {terr}")
            continue
        if mutation_binding is not None and item_binding.base_path != mutation_binding.base_path:
            errors.append(f"{op.path}: mixed physical bases require separate calls")
            continue
        mutation_binding = mutation_binding or item_binding
        bindings[rel] = item_binding
        if op.kind == "add":
            if rel in seen or target.exists():
                errors.append(
                    f"⚠️ APPLY_PATCH_ERROR: Add File {op.path}: file already exists. "
                    "Use '*** Update File:' to modify it."
                )
                continue
            content = "\n".join(op.add_lines)
            if content and not content.endswith("\n"):
                content += "\n"
            planned_writes.append((target, rel, content))
            seen[rel] = content
            sites[rel] = [0]
            summaries.append(f"✅ Added {rel} ({len(op.add_lines)} lines)")
            continue
        if op.kind == "delete":
            if reason := _edit_mutation_block(ctx, item_binding, op.path, operation="delete"):
                errors.append(f"Delete File {op.path}: {reason}")
                continue
            from ouroboros.protected_artifacts import block_reason_for_path

            # Every explicit deletion first preserves the exact bytes, so reading
            # and copying them must be as authorized as the delete itself.
            recovery_denial = next((reason for operation in ("read_bytes", "copy")
                                   if (reason := block_reason_for_path(
                                       ctx, target, operation, item_binding))), "")
            if recovery_denial:
                errors.append(f"Delete File {op.path}: recovery capture blocked: {recovery_denial}")
                continue
            if rel in seen or rel in deleted:
                errors.append(f"Delete File {op.path}: conflicts with another operation on this file")
                continue
            if not target.exists():
                errors.append(f"⚠️ APPLY_PATCH_ERROR: Delete File {op.path}: file not found.")
                continue
            if not _repo_edit_binding(item_binding) and not force:
                errors.append(f"⚠️ APPLY_PATCH_ERROR: Delete File {op.path}: data-root deletion requires force=true.")
                continue
            planned_deletes.append((target, rel))
            deleted.add(rel)
            summaries.append(f"✅ Deleted {rel}")
            continue
        # update
        if rel in deleted:
            errors.append(f"Update File {op.path}: conflicts with Delete File on this file")
            continue
        if rel in seen:
            content = seen[rel]
        else:
            if not target.exists():
                errors.append(f"⚠️ APPLY_PATCH_ERROR: Update File {op.path}: file not found.")
                continue
            try:
                content = target.read_text(encoding="utf-8")
            except Exception as e:  # noqa: BLE001 - report unreadable target
                errors.append(f"⚠️ APPLY_PATCH_ERROR: cannot read {op.path}: {e}")
                continue
        new_content, notes, herr = _apply_hunks_to_text(content, op.hunks, rel, _sites=sites.setdefault(rel, []))
        if herr:
            errors.append(f"{op.path}: {herr}")
            continue
        seen[rel] = new_content
        planned_writes.append((target, rel, new_content))
        added = sum(1 for h in op.hunks for p, _ in h.lines if p == "+")
        removed = sum(1 for h in op.hunks for p, _ in h.lines if p == "-")
        summaries.append(f"✅ Updated {rel} ({len(op.hunks)} hunk(s), +{added}/-{removed} lines)")
        all_notes.extend(f"{rel}: {n}" for n in notes)

    if errors:
        return no_effect("⚠️ APPLY_PATCH_ERROR: validation failed; NOTHING was written. Problems:\n"
                         + "\n".join("  - " + problem for problem in errors))

    # Phase 2: write. Dedup chained updates so each file is written once (final content).
    final_content: Dict[str, Tuple[pathlib.Path, str]] = {}
    for target, rel, content in planned_writes:
        final_content[rel] = (target, content)
    syntax_notes: List[str] = []
    for rel, (target, content) in final_content.items():
        if refusal := _edit_content_block(bindings[rel], content, force):
            return no_effect(f"⚠️ APPLY_PATCH_ERROR: {refusal}\nNothing was written.")
        if note := newline_normalization_note(target):
            all_notes.append(f"{rel}: {note}")
        if note := edit_syntax_note(bindings[rel], content):
            syntax_notes.append(note)
    from ouroboros.contracts.task_constraint import normalize_task_constraint
    from ouroboros.skill_repair_admission import repair_write_cas_error, advance_repair_expected_hash

    constraint = normalize_task_constraint(getattr(ctx, "task_constraint", None))
    repair_binding = next((b for b in bindings.values() if b.skill_name), None)
    if repair_binding and constraint and constraint.has_selected_skill:
        if refusal := repair_write_cas_error(repair_binding.state_drive_root, constraint,
                                             task_id=ctx.task_id, repair_task=True):
            return no_effect(refusal)
    # Every explicit deletion first stores the file's exact current bytes as an
    # immutable task artifact. A failed capture refuses the whole patch.
    recovery: List[str] = []
    if planned_deletes:
        from ouroboros.artifacts import copy_file_to_task_artifacts

        try:
            for target, rel in planned_deletes:
                record = copy_file_to_task_artifacts(ctx, target, kind="delete_recovery", immutable=True)
                if not record:
                    raise OSError(f"{rel} is not a regular file")
                if pathlib.Path(record["path"]).resolve() == target.resolve():
                    raise OSError(f"{rel} is itself the registered copy; deleting it would leave none")
                recovery.append(f"{rel}: artifact_store:{record['name']} (sha256 {record['sha256'][:12]})")
        except Exception as exc:
            return no_effect(f"⚠️ APPLY_PATCH_ERROR: recovery capture failed: {exc}. Nothing was deleted or edited.")
    changed_paths: List[str] = []
    registered_outputs: List[str] = []
    pending = [*final_content, *(rel for _target, rel in planned_deletes)]

    def partial(detail: str, *, landed: bool = False) -> str:
        return _partial_write_failure(ctx, changed_paths, "apply_patch", "APPLY_PATCH_ERROR", detail,
                                      mutation_binding, recovery=recovery,
                                      pending=pending[len(changed_paths) + (0 if landed else 1):])

    for rel, (target, content) in final_content.items():
        try:
            write_text(target, content)
        except Exception as e:  # noqa: BLE001 - surface the failed path
            return partial(f"write failed for {rel}: {e}")
        changed_paths.append(rel)
        if bindings[rel].root == "user_files":
            from ouroboros.artifacts import copy_file_to_task_artifacts

            try:
                record = copy_file_to_task_artifacts(ctx, target, kind="user_file")
                if not record:
                    raise OSError("user file recovery copy was not registered")
                registered_outputs.append(f"ARTIFACT_OUTPUTS: registered user file -> artifact_store:{record['name']}")
            except Exception as exc:
                return partial(f"{rel} WAS written but its user artifact registration failed: {exc}", landed=True)
    for target, rel in planned_deletes:
        try:
            target.unlink()
        except Exception as e:  # noqa: BLE001 - surface the failed path
            return partial(f"delete failed for {rel}: {e}")
        changed_paths.append(rel)

    if repair_binding and constraint and constraint.has_selected_skill:
        advance_repair_expected_hash(repair_binding.state_drive_root, constraint, task_id=ctx.task_id)

    footer = _finish_mutation(ctx, changed_paths, "apply_patch", mutation_binding)
    body = "\n".join([*summaries, *syntax_notes])
    if all_notes:
        body += "\nNotes:\n" + "\n".join("  " + n for n in all_notes)
    if recovery:
        body += "\nRecovery copies:\n" + "\n".join(recovery)
    if registered_outputs:
        body += "\n" + "\n".join(registered_outputs)
    previews = []
    for rel, (_, content) in final_content.items():
        previews.append(f"Context after {rel}:\n{numbered_edit_preview(content, sites[rel] or [0])}")
    if previews:
        body += "\n" + "\n".join(previews)
    return f"{body}\n{footer}"


# ---------------------------------------------------------------------------
# edit_batch
# ---------------------------------------------------------------------------

def _edit_batch(
    ctx: ToolContext,
    edits: List[Dict[str, Any]],
    root: str | None = None,
    bucket: str = "",
    skill_name: str = "",
    force: bool = False,
    _resolved_binding: ResolvedResourceBinding | tuple[ResolvedResourceBinding, ...] | None = None,
    source_tool: str = "edit_batch",
) -> str:
    if not edits or not isinstance(edits, list):
        return "⚠️ EDIT_BATCH_ERROR: edits must be a non-empty array."
    from ouroboros.tools.tool_resolution import inferred_file_root

    root, root_error = inferred_file_root(ctx, "edit_batch", root,
                                          [str(e.get("path") or "") for e in edits if isinstance(e, dict)])
    if root_error:
        return root_error
    item_refusal, notes = payload_item_feedback(
        ctx, edits, _EDIT_BATCH_ITEM_PROPERTIES, item_label="edit", options={"root": root},
    )
    if item_refusal:
        return item_refusal
    contents: Dict[str, str] = {}
    targets: Dict[str, pathlib.Path] = {}
    bindings: Dict[str, ResolvedResourceBinding] = {}
    sites: Dict[str, List[int]] = {}
    applied: List[str] = []
    errors: List[str] = []
    supplied_bindings = (
        tuple(_resolved_binding)
        if isinstance(_resolved_binding, tuple)
        else ((_resolved_binding,) if _resolved_binding is not None else ())
    )
    if supplied_bindings and len(supplied_bindings) != len(edits):
        return "⚠️ EDIT_BATCH_ERROR: internal target binding count mismatch."
    binding_iter = iter(supplied_bindings)
    mutation_binding: ResolvedResourceBinding | None = None
    located = 0  # misses diagnosed so far (bounded per call)
    for idx, edit in enumerate(edits, 1):
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
            bucket=bucket, skill_name=skill_name,
        )
        if terr:
            errors.append(f"edit {idx}: {terr.lstrip('⚠️ ')}")
            continue
        if mutation_binding is not None and item_binding.base_path != mutation_binding.base_path:
            errors.append(f"edit {idx}: mixed physical bases require separate calls")
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
            bindings[rel] = item_binding
            sites[rel] = []
        text = contents[rel]
        occurrences = text.count(old_str)
        if occurrences != count:
            positions = _line_positions(text, old_str)
            where = f" (at: {', '.join(positions)})" if positions else ""
            errors.append(
                f"edit {idx} ({rel}): old_str occurs {occurrences} time(s), expected {count}{where}. "
                "Re-read the file and set count to the exact number of occurrences you intend to replace."
            )
            if occurrences == 0 and located < _LOCATE_BATCH_MAX:
                located += 1
                errors[-1] += "\n" + textwrap.indent(locate_edit_miss(text, old_str), "      ")
            continue
        positions: List[int] = []
        search = 0
        for _ in range(count):
            position = text.find(old_str, search)
            positions.append(position)
            search = position + len(old_str)
        delta = len(new_str) - len(old_str)
        def moved_site(site: int) -> int:
            earlier = 0
            for position in positions:
                if position <= site < position + len(old_str):
                    return position + earlier
                if position < site:
                    earlier += delta
            return site + earlier

        sites[rel] = [moved_site(site) for site in sites[rel]]
        sites[rel].extend(position + index * delta for index, position in enumerate(positions))
        contents[rel] = text.replace(old_str, new_str)
        applied.append(f"edit {idx} ({rel}): replaced {count} occurrence(s)")
    if errors:
        return (
            "⚠️ EDIT_BATCH_ERROR: batch aborted, NOTHING was written (atomic). Problems:\n"
            + "\n".join("  - " + e for e in errors)
        )
    for rel, text in contents.items():
        if refusal := _edit_content_block(bindings[rel], text, force):
            return f"⚠️ EDIT_BATCH_ERROR: {refusal}\nNothing was written."
        if note := newline_normalization_note(targets[rel]):
            notes.append(f"{rel}: {note}")
        if note := edit_syntax_note(bindings[rel], text):
            notes.append(note)
    from ouroboros.contracts.task_constraint import normalize_task_constraint
    from ouroboros.skill_repair_admission import repair_write_cas_error, advance_repair_expected_hash

    constraint = normalize_task_constraint(getattr(ctx, "task_constraint", None))
    repair_binding = next((b for b in bindings.values() if b.skill_name), None)
    if repair_binding and constraint and constraint.has_selected_skill:
        if refusal := repair_write_cas_error(repair_binding.state_drive_root, constraint,
                                             task_id=ctx.task_id, repair_task=True):
            return refusal
    changed: List[str] = []
    registered_outputs: List[str] = []
    order = list(contents)
    for rel, text in contents.items():
        try:
            write_text(targets[rel], text)
        except Exception as e:  # noqa: BLE001 - surface the failed path
            return _partial_write_failure(
                ctx, changed, source_tool, "EDIT_BATCH_ERROR",
                f"write failed for {rel}: {e}",
                mutation_binding, pending=order[len(changed) + 1:],
            )
        changed.append(rel)
        if bindings[rel].root == "user_files":
            from ouroboros.artifacts import copy_file_to_task_artifacts

            try:
                record = copy_file_to_task_artifacts(ctx, targets[rel], kind="user_file")
                if not record:
                    raise OSError("user file recovery copy was not registered")
                registered_outputs.append(f"ARTIFACT_OUTPUTS: registered user file -> artifact_store:{record['name']}")
            except Exception as exc:
                return _partial_write_failure(ctx, changed, source_tool, "EDIT_BATCH_ERROR",
                                              f"{rel} WAS written but its user artifact registration failed: {exc}",
                                              mutation_binding, pending=order[len(changed):])
    if repair_binding and constraint and constraint.has_selected_skill:
        advance_repair_expected_hash(repair_binding.state_drive_root, constraint, task_id=ctx.task_id)
    footer = _finish_mutation(ctx, changed, source_tool, mutation_binding)
    previews = "\n".join(f"Context after {rel}:\n{numbered_edit_preview(contents[rel], sites[rel])}"
                         for rel in changed)
    if registered_outputs:
        previews += "\n" + "\n".join(registered_outputs)
    return with_argument_notes(ctx, (
        f"✅ {source_tool} applied {len(applied)} edit(s) across {len(changed)} file(s):\n"
        + "\n".join("  " + a for a in applied)
        + f"\n{previews}\n{footer}"
    ), notes)


# ---------------------------------------------------------------------------
# shared verification helpers (also used by git._repo_write)
# ---------------------------------------------------------------------------

def _syntax_check(rel: str, content: str) -> str:
    """Cheap validity check for known formats. Returns error text or ''."""
    try:
        if rel.endswith(".py"):
            compile(content, rel, "exec")
        elif rel.endswith(".json"):
            json.loads(content)
    except SyntaxError as e:
        return f"content has a Python syntax error at line {e.lineno}: {e.msg}"
    except ValueError as e:
        # compile() raises a bare ValueError for content Python cannot even scan
        # (a NUL byte, for one). Report it against the format actually checked —
        # "not valid JSON" for a .py file sends the fix in the wrong direction.
        kind = "Python source" if rel.endswith(".py") else "JSON"
        return f"content is not valid {kind}: {e}"
    except Exception:
        return ""
    return ""


def _unified_diff(rel: str, before: str, after: str, cap: int = 400) -> str:
    diff_lines = list(
        difflib.unified_diff(
            before.splitlines(), after.splitlines(),
            fromfile=f"a/{rel}", tofile=f"b/{rel}", lineterm="",
        )
    )
    # splitlines() drops the final terminator, so adding or removing the trailing
    # newline is invisible to the line diff. This rail exists to let the agent
    # VERIFY an overwrite; reporting "no textual changes" for a file whose bytes
    # did change is the one thing it must never do.
    trailing_note = ""
    if before.endswith("\n") != after.endswith("\n"):
        trailing_note = (
            "\\ No newline at end of file (the previous version had one)"
            if before.endswith("\n")
            else "\\ Newline added at end of file"
        )
    if not diff_lines:
        return trailing_note or "(no textual changes)"
    clipped = diff_lines[:cap]
    if len(diff_lines) > cap:
        clipped.append(f"... diff truncated ({len(diff_lines) - cap} more lines)")
    if trailing_note:
        clipped.append(trailing_note)
    return "\n".join(clipped)


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
                "@@ def some_function\n"
                " context line (starts with a space)\n"
                "-removed line\n"
                "+added line\n"
                "*** Add File: new/file.py\n"
                "+each line of the new file prefixed with +\n"
                "*** Delete File: old/file.py\n"
                "*** End Patch\n"
                "The optional @@ line is a lower bound: the hunk is searched at or after its first hit; "
                "context must still match exactly once. Repeat '*** Update File:' to chain "
                "edits on an already patched file. Delete File stores the file's exact bytes "
                "as a task artifact first. On a repo root, a written file whose final Python/JSON "
                "does not parse (Add File included) is reported as SYNTAX_WARNING after it is "
                "applied, never refused; write_file keeps its own full-write check. "
                "Prefer this over many edit_text calls for related multi-file changes. "
                "NOT for rewrites touching most of a file — there the patch grows as "
                "large as the file itself; use write_file instead."
            ),
            "parameters": {"type": "object", "properties": {
                "patch": {"type": "string", "description": "The full patch text (envelope lines optional)."},
                "root": {"type": "string", "enum": ["active_workspace", "system_repo", "runtime_data", "task_drive", "artifact_store", "user_files", "skill_payload"], "description": _FILE_ROOT_DESCRIPTION},
                "bucket": {"type": "string", "description": "Skill payload bucket for root=skill_payload."},
                "skill_name": {"type": "string", "description": "Skill slug for root=skill_payload."},
                "force": {"type": "boolean", "default": False, "description": "Confirm an intentional shrink (>30% smaller); required to delete a file outside the repo roots."},
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
                "edits": {"type": "array", "items": {"type": "object",
                    "properties": {k: dict(v) for k, v in _EDIT_BATCH_ITEM_PROPERTIES.items()},
                    "required": list(_EDIT_BATCH_ITEM_REQUIRED)}},
                "root": {"type": "string", "enum": ["active_workspace", "system_repo", "runtime_data", "task_drive", "artifact_store", "user_files", "skill_payload"], "description": _FILE_ROOT_DESCRIPTION},
                "bucket": {"type": "string", "description": "Skill payload bucket for root=skill_payload."},
                "skill_name": {"type": "string", "description": "Skill slug for root=skill_payload."},
                "force": {"type": "boolean", "default": False, "description": "Confirm an intentional shrink (>30% smaller)."},
            }, "required": ["edits"]},
        }, _edit_batch, is_code_tool=True, mutates_worktree=True),
    ]
