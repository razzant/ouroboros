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

``locate_edit_miss`` is the one bounded diagnosis all three exact editors attach
when their needle is not in the file (edit_text through ``core._str_match_replace``):
the closest region, the first differing line and the ``read_file`` window to copy from.

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

import difflib
import json
import logging
import pathlib
import textwrap
from dataclasses import dataclass, field
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
    from ouroboros.tools.core import _access_or_block

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
    footer = (
        "Files are on disk but NOT committed. Run commit_reviewed when ready.\n"
        "⚠️ Advisory pre-review is now stale — run preflight_review before commit_reviewed."
    )
    # A pro-mode edit of a protected surface announces itself here exactly as it
    # does from git._repo_write / _str_replace_editor (the protected-write contract
    # in ARCHITECTURE "Safety and runtime mode" and SYSTEM.md "Safety-critical
    # files"): the mode ALLOWS the write, and the notice is what keeps it visible.
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
        footer = _finish_mutation(ctx, changed_paths, source_tool, binding)
        # NOT the tools' own *_ERROR prefix: those read as validation refusals
        # (a counted/context miss) and are classified as policy denials. This is a
        # genuine partial mutation from an I/O fault and must stay an execution
        # failure, so it carries its own prefix and lands in the generic `error`.
        return (
            f"⚠️ EDIT_OPS_PARTIAL_WRITE_FAILED ({tag}): {detail}\n"
            f"PARTIALLY APPLIED — these files WERE written: {', '.join(changed_paths)}. "
            "Re-read them before retrying; advisory pre-review is now stale.\n"
            + footer
        )
    return f"⚠️ {tag}: {detail}\nNothing was written."


def _line_positions(text: str, needle: str, limit: int = 5) -> List[str]:
    positions: List[str] = []
    start = 0
    for _ in range(limit):
        idx = text.find(needle, start)
        if idx < 0:
            break
        positions.append(f"line {text[:idx].count(chr(10)) + 1}")
        start = idx + 1
    return positions


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


def _excerpt(file_lines: List[str], start: int, count: int) -> str:
    end = min(len(file_lines), start + count)
    shown = min(end - start, _LOCATE_EXCERPT_LINES)
    width = len(str(start + shown)) + 1
    rows = [f"{start + i + 1:>{width}}| {_cut(file_lines[start + i])}" for i in range(shown)]
    if end - start > shown:
        rows.append(f"{'':>{width}}| … {end - start - shown} more line(s) of the region not shown")
    return "\n".join(rows)


def _first_difference(file_lines: List[str], start: int, old_lines: List[str]) -> Optional[Tuple[int, str, str]]:
    """The first ``(line_no, file_bytes, needle_bytes)`` where the raw lines disagree."""
    for k, (needle_line, position) in enumerate(zip(old_lines, _positions(len(old_lines)))):
        if start + k >= len(file_lines):
            return start + k + 1, "<end of file>", needle_line
        if not _at(file_lines[start + k], needle_line, position):
            return start + k + 1, file_lines[start + k], needle_line
    return None


def _whitespace_reason(file_line: str, needle_line: str) -> str:
    if file_line.rstrip() == needle_line.rstrip():
        return "trailing whitespace differs"
    if file_line.strip() == needle_line.strip():
        lead = lambda s: len(s) - len(s.lstrip())  # noqa: E731 - local one-liner
        return f"indentation differs: file {lead(file_line)} leading chars, needle {lead(needle_line)}"
    return "tabs and spaces differ" if file_line.expandtabs() == needle_line.expandtabs() else "whitespace differs inside the line"


def whole_file_preview(text: str, limit: int = _WHOLE_FILE_PREVIEW_CHARS) -> str:
    """The whole file when small enough; a head cut of a large file previews nothing about a miss."""
    return "" if len(text) > limit else f"\nFile preview (whole file, {len(text)} chars):\n{text}"


def _bounded(block: str, notes: List[str]) -> str:
    out = "\n".join([block, *notes])
    return out if len(out) <= _LOCATE_MAX_CHARS else out[: _LOCATE_MAX_CHARS - 1] + "…"


def locate_edit_miss(
    text: str, old_str: str, *, cursor_line: int = 1, needle_name: str = "old_str",
    max_candidates: int = _LOCATE_MAX_CANDIDATES,
) -> str:
    """Bounded, actionable diagnosis of why ``old_str`` is not in ``text``: the
    closest region, the FIRST line where the file's bytes differ from the needle
    and the ``read_file`` window to copy from. Tiers, cheapest first: line endings
    (CR/CRLF) → a whitespace-relaxed line match (indentation, trailing, tabs; a
    needle may start or end mid-line) → the nearest region by the needle's most
    distinctive line and difflib similarity. Every tier is bounded (lines scanned,
    candidates scored, excerpt and block size). ``cursor_line`` (1-based) is where
    the caller's search began — apply_patch hunks apply in file order, so a region
    before it is reported as such: the fix is reordering, not retyping."""
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
            line = n_text[: n_text.index(n_old)].count("\n") + 1
            side = (f"{needle_name} carries CR (\\r) characters the file does not" if "\r" not in text
                    else "the file carries CR (\\r) characters")
            return _bounded(
                f"{needle_name} matches at line {line} once line endings are normalized: {side}. Use LF (\\n) only.\n"
                f"Re-read that region (read_file start_line={line} max_lines={n_old.count(chr(10)) + 1}) "
                f"and copy the exact bytes into {needle_name}.", notes)
    old_lines = old_str.split("\n")
    while old_lines and not old_lines[0].strip():
        old_lines.pop(0)
    while old_lines and not old_lines[-1].strip():
        old_lines.pop()
    if not old_lines:
        return _bounded(f"{needle_name} is whitespace only; include at least one non-blank line.", notes)
    total = len(old_lines)  # the whole needle; only the window below is compared, and that is said
    old_lines = old_lines[:_LOCATE_MAX_OLD_LINES]
    n, positions = len(old_lines), _positions(len(old_lines))
    if total > n:
        notes.append(f"(only the first {n} of {total} {needle_name} lines were compared; the miss may be after them)")
    needle = [_norm_ws(line) for line in old_lines]
    hay = [_norm_ws(line) for line in file_lines]

    def fits(start: int) -> bool:  # every needle line at its position, whitespace aside (short-circuits)
        return start + n <= len(hay) and all(_at(hay[start + k], needle[k], positions[k]) for k in range(n))

    def matched(start: int) -> int:
        return sum(1 for k in range(n) if start + k < len(hay) and _at(hay[start + k], needle[k], positions[k]))

    def span(start: int) -> str:
        end = min(len(file_lines), start + n)
        return f"line {start + 1}" if end == start + 1 else f"lines {start + 1}–{end}"

    def render(head: str, start: int, extra: List[str]) -> str:
        diff = _first_difference(file_lines, start, old_lines)
        body = [head, _excerpt(file_lines, start, n)]
        if diff:
            body.append(f"first difference at line {diff[0]}:\n  file   : {_cut(diff[1])!r}\n  {needle_name:<7}: {_cut(diff[2])!r}")
        if start + 1 < cursor_line:
            body.append(f"Note: that region is BEFORE line {cursor_line}, where this search started: hunks apply "
                        "in file order — move this hunk earlier or add an @@ anchor above the region.")
        body += extra + [f"Re-read that region (read_file start_line={start + 1} max_lines={total}) "
                         f"and copy the exact bytes into {needle_name}."]
        return _bounded("\n".join(body), notes)

    relaxed = [i for i in range(len(hay) - n + 1) if fits(i)][:5]  # tier 1: the same lines, whitespace aside
    if relaxed:
        start = relaxed[0]
        diff = _first_difference(file_lines, start, old_lines)
        reason = (_whitespace_reason(diff[1], diff[2]) if diff
                  else (f"its first {n} lines match exactly; the difference is in the {total - n} lines after them, "
                        "which were not compared") if total > n
                  else "the bytes match" if start + 1 < cursor_line else "only leading/trailing blank lines differ")
        also = f"; also at lines {', '.join(str(i + 1) for i in relaxed[1:])}" if len(relaxed) > 1 else ""
        return render(f"{needle_name} matches {span(start)} ignoring whitespace ({reason}){also}. "
                      "The file's exact bytes are:", start, [])
    anchor_k = max(range(n), key=lambda k: len(needle[k]))  # tier 2: the needle's most distinctive line
    anchor = needle[anchor_k]
    hits = [i for i, line in enumerate(hay) if anchor and anchor in line][:_LOCATE_MAX_ANCHOR_HITS]
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
    scored = sorted((-matched(s), abs(s + 1 - cursor_line), s) for s in {max(0, i - anchor_k) for i in hits})
    score, _distance, start = scored[0]
    others = [str(s + 1) for _sc, _d, s in scored[1:max_candidates]]
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
        if raw.startswith(_UPDATE_HDR):
            current = _FileOp("update", _strip_directive_tail(raw[len(_UPDATE_HDR):]))
            ops.append(current)
            continue
        if raw.startswith(_ADD_HDR):
            current = _FileOp("add", _strip_directive_tail(raw[len(_ADD_HDR):]))
            ops.append(current)
            continue
        if raw.startswith(_DELETE_HDR):
            current = _FileOp("delete", _strip_directive_tail(raw[len(_DELETE_HDR):]))
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


def _find_sequence(
    file_lines: List[str], seq: List[str], start: int, *, fuzzy: bool
) -> List[int]:
    """Indices >= start where ``seq`` matches ``file_lines`` (cap 5)."""
    if not seq:
        return []
    matches: List[int] = []
    if fuzzy:
        hay = [l.rstrip() for l in file_lines]
        needle = [l.rstrip() for l in seq]
    else:
        hay = file_lines
        needle = seq
    n = len(needle)
    for i in range(start, len(hay) - n + 1):
        if hay[i:i + n] == needle:
            matches.append(i)
            if len(matches) >= 5:
                break
    return matches


def _apply_hunks_to_text(
    content: str, hunks: List[_Hunk], path: str
) -> Tuple[Optional[str], List[str], str]:
    """Apply hunks in order. Returns (new_content, notes, error)."""
    file_lines = content.split("\n")
    notes: List[str] = []
    cursor = 0
    for hi, hunk in enumerate(hunks, 1):
        old = [t for p, t in hunk.lines if p in (" ", "-")]
        new = [t for p, t in hunk.lines if p in (" ", "+")]
        start = cursor
        if hunk.anchor:
            anchor_hits = [
                i for i in range(start, len(file_lines)) if hunk.anchor in file_lines[i]
            ]
            if not anchor_hits:
                return None, notes, (
                    f"hunk {hi}: @@ anchor {hunk.anchor!r} not found in {path} "
                    f"after line {start + 1}"
                )
            start = anchor_hits[0]
        if not old:
            if not hunk.anchor:
                return None, notes, (
                    f"hunk {hi}: pure insertion needs an @@ anchor or context lines"
                )
            pos = start + 1
            file_lines[pos:pos] = new
            cursor = pos + len(new)
            continue
        matches = _find_sequence(file_lines, old, start, fuzzy=False)
        fuzzy_used = False
        if not matches:
            matches = _find_sequence(file_lines, old, start, fuzzy=True)
            fuzzy_used = bool(matches)
        if not matches:
            preview = "\n".join("    " + l for l in old[:6])
            return None, notes, (
                f"hunk {hi}: context not found in {path} (searched from line {start + 1}). "
                f"Hunk expects these consecutive lines:\n{preview}\n"
                "Copy the exact lines from the file (read_file) into the hunk context.\n"
                + locate_edit_miss(
                    "\n".join(file_lines), "\n".join(old),
                    cursor_line=start + 1, needle_name="the hunk context",
                )
            )
        if len(matches) > 1:
            where = ", ".join(f"line {m + 1}" for m in matches)
            return None, notes, (
                f"hunk {hi}: context is ambiguous in {path} — matches at {where}. "
                "Add an @@ anchor (e.g. '@@ def name') or more context lines."
            )
        pos = matches[0]
        file_lines[pos:pos + len(old)] = new
        cursor = pos + len(new)
        if fuzzy_used:
            notes.append(
                f"hunk {hi}: matched ignoring trailing whitespace — the replaced lines, "
                "INCLUDING context lines, now carry the patch's trailing whitespace"
            )
    return "\n".join(file_lines), notes, ""


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

    # Phase 1: resolve + validate everything BEFORE any write (atomicity).
    planned_writes: List[Tuple[pathlib.Path, str, str]] = []  # (target, rel_path, content)
    planned_deletes: List[Tuple[pathlib.Path, str]] = []
    summaries: List[str] = []
    all_notes: List[str] = []
    seen: Dict[str, str] = {}  # rel path -> pending content (chained updates)
    supplied_bindings = (
        tuple(_resolved_binding)
        if isinstance(_resolved_binding, tuple)
        else ((_resolved_binding,) if _resolved_binding is not None else ())
    )
    if supplied_bindings and len(supplied_bindings) != len(ops):
        return "⚠️ APPLY_PATCH_ERROR: internal target binding count mismatch."
    binding_iter = iter(supplied_bindings)
    mutation_binding: ResolvedResourceBinding | None = None
    for op in ops:
        target, rel, item_binding, terr = _resolve_edit_target(
            ctx,
            op.path,
            root,
            error_tag="APPLY_PATCH_BLOCKED",
            _resolved_binding=next(binding_iter, None),
        )
        if terr:
            return terr
        mutation_binding = mutation_binding or item_binding
        if op.kind == "add":
            if rel in seen or target.exists():
                return (
                    f"⚠️ APPLY_PATCH_ERROR: Add File {op.path}: file already exists. "
                    "Use '*** Update File:' to modify it."
                )
            content = "\n".join(op.add_lines)
            if content and not content.endswith("\n"):
                content += "\n"
            planned_writes.append((target, rel, content))
            seen[rel] = content
            summaries.append(f"✅ Added {rel} ({len(op.add_lines)} lines)")
            continue
        if op.kind == "delete":
            if not target.exists():
                return f"⚠️ APPLY_PATCH_ERROR: Delete File {op.path}: file not found."
            planned_deletes.append((target, rel))
            summaries.append(f"✅ Deleted {rel}")
            continue
        # update
        if rel in seen:
            content = seen[rel]
        else:
            if not target.exists():
                return f"⚠️ APPLY_PATCH_ERROR: Update File {op.path}: file not found."
            try:
                content = target.read_text(encoding="utf-8")
            except Exception as e:  # noqa: BLE001 - report unreadable target
                return f"⚠️ APPLY_PATCH_ERROR: cannot read {op.path}: {e}"
        new_content, notes, herr = _apply_hunks_to_text(content, op.hunks, rel)
        if herr:
            return f"⚠️ APPLY_PATCH_ERROR: {herr}\nNothing was applied (the patch is atomic)."
        seen[rel] = new_content
        planned_writes.append((target, rel, new_content))
        added = sum(1 for h in op.hunks for p, _ in h.lines if p == "+")
        removed = sum(1 for h in op.hunks for p, _ in h.lines if p == "-")
        summaries.append(f"✅ Updated {rel} ({len(op.hunks)} hunk(s), +{added}/-{removed} lines)")
        all_notes.extend(f"{rel}: {n}" for n in notes)

    # Phase 2: write. Dedup chained updates so each file is written once (final content).
    final_content: Dict[str, Tuple[pathlib.Path, str]] = {}
    for target, rel, content in planned_writes:
        final_content[rel] = (target, content)
    changed_paths: List[str] = []
    for rel, (target, content) in final_content.items():
        try:
            write_text(target, content)
        except Exception as e:  # noqa: BLE001 - surface the failed path
            return _partial_write_failure(
                ctx, changed_paths, "apply_patch", "APPLY_PATCH_ERROR",
                f"write failed for {rel}: {e}",
                mutation_binding,
            )
        changed_paths.append(rel)
    for target, rel in planned_deletes:
        try:
            target.unlink()
        except Exception as e:  # noqa: BLE001 - surface the failed path
            return _partial_write_failure(
                ctx, changed_paths, "apply_patch", "APPLY_PATCH_ERROR",
                f"delete failed for {rel}: {e}",
                mutation_binding,
            )
        changed_paths.append(rel)

    footer = _finish_mutation(ctx, changed_paths, "apply_patch", mutation_binding)
    body = "\n".join(summaries)
    if all_notes:
        body += "\nNotes:\n" + "\n".join("  " + n for n in all_notes)
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
    located = 0  # misses diagnosed so far (bounded per call)
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
            if occurrences == 0 and located < _LOCATE_BATCH_MAX:
                located += 1
                errors[-1] += "\n" + textwrap.indent(locate_edit_miss(text, old_str), "      ")
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
        if rel.endswith(".py"):
            return f"content is not valid Python source: {e}"
        return f"content is not valid JSON: {e}"
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
