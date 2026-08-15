"""Patch and batch-edit CORE: parsing, hunk location, text application.

Split out of ``tools/edit_ops`` so BOTH routes can share one authority. The Home
handler and the target-native operation must locate a hunk, count an occurrence and
apply a replacement identically — a second copy of a patch parser is the "one policy
× N doors" shape this branch exists to remove, and here the two answers would differ
in the worst possible way: silently, on the bytes of the owner's files.

Everything here is PURE TEXT. No filesystem, no policy, no context object, and in
particular no import of the tool registry — the target's kernel could not load it if
there were one. The route-specific halves (which path is writable, what the refusal
says, what the footer discloses) stay with their respective callers.
"""

from __future__ import annotations

import difflib
import json
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

_PATCH_BEGIN = "*** Begin Patch"
_PATCH_END = "*** End Patch"
_UPDATE_HDR = "*** Update File:"
_ADD_HDR = "*** Add File:"
_DELETE_HDR = "*** Delete File:"


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
                "Copy the exact lines from the file (read_file) into the hunk context."
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


__all__ = [
    "PatchPlan",
    "plan_patch",
    "_line_positions",
    "_Hunk",
    "_FileOp",
    "_strip_directive_tail",
    "_parse_patch",
    "patch_target_paths",
    "_find_sequence",
    "_apply_hunks_to_text",
    "_syntax_check",
    "_unified_diff",
]


@dataclass
class PatchPlan:
    """Everything the patch WOULD do, decided before anything is written."""

    writes: List[Tuple[str, str]] = field(default_factory=list)   # (rel, final content)
    deletes: List[str] = field(default_factory=list)              # rel
    summaries: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    error: str = ""


def plan_patch(ops, *, resolve, exists, read_text) -> PatchPlan:
    """Validate a parsed patch against a tree and return what it would do.

    ATOMICITY LIVES HERE, once. `apply_patch`'s headline guarantee is that an
    unmatched hunk aborts the whole patch untouched, and that guarantee is a
    property of this ORDER: resolve and validate every operation, then write. A
    second copy of this function on the target route would be free to drift on
    exactly that order — and the divergence would be invisible until a patch half
    applied on one host and not the other.

    The three callbacks are the route's own doors, and they are the ONLY difference
    between Home and a target:

    * ``resolve(raw_path) -> (rel, error)`` — the policy door. It decides whether
      this path may be written at all and what the refusal says; an error aborts
      the plan before any file is touched.
    * ``exists(rel) -> bool``
    * ``read_text(rel) -> (content, error)``

    Chained updates are deduplicated to the FINAL content per file, so a patch that
    updates one file twice writes it once.
    """
    plan = PatchPlan()
    pending: dict[str, str] = {}   # rel -> content as the patch has built it so far
    ordered: List[str] = []

    for op in ops:
        rel, err = resolve(op.path)
        if err:
            plan.error = err
            return plan
        if op.kind == "add":
            if rel in pending or exists(rel):
                plan.error = (
                    f"⚠️ APPLY_PATCH_ERROR: Add File {op.path}: file already exists. "
                    "Use '*** Update File:' to modify it."
                )
                return plan
            content = "\n".join(op.add_lines)
            if content and not content.endswith("\n"):
                content += "\n"
            pending[rel] = content
            if rel not in ordered:
                ordered.append(rel)
            plan.summaries.append(f"✅ Added {rel} ({len(op.add_lines)} lines)")
            continue
        if op.kind == "delete":
            if not exists(rel):
                plan.error = f"⚠️ APPLY_PATCH_ERROR: Delete File {op.path}: file not found."
                return plan
            plan.deletes.append(rel)
            plan.summaries.append(f"✅ Deleted {rel}")
            continue
        if rel in pending:
            content = pending[rel]
        else:
            if not exists(rel):
                plan.error = f"⚠️ APPLY_PATCH_ERROR: Update File {op.path}: file not found."
                return plan
            content, read_err = read_text(rel)
            if read_err:
                plan.error = read_err
                return plan
        new_content, notes, herr = _apply_hunks_to_text(content, op.hunks, rel)
        if herr:
            plan.error = (
                f"⚠️ APPLY_PATCH_ERROR: {herr}\nNothing was applied (the patch is atomic)."
            )
            return plan
        pending[rel] = new_content
        if rel not in ordered:
            ordered.append(rel)
        added = sum(1 for h in op.hunks for p, _ in h.lines if p == "+")
        removed = sum(1 for h in op.hunks for p, _ in h.lines if p == "-")
        plan.summaries.append(
            f"✅ Updated {rel} ({len(op.hunks)} hunk(s), +{added}/-{removed} lines)"
        )
        plan.notes.extend(f"{rel}: {n}" for n in notes)

    plan.writes = [(rel, pending[rel]) for rel in ordered]
    return plan
