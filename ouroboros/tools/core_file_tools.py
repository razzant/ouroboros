"""Read/list file tools and shared resource-access helpers."""

from __future__ import annotations

import bisect
import hashlib
import json
import logging
import os
import pathlib
import re
from collections import deque
from typing import Any, Callable, Dict, List, Optional, Tuple

from ouroboros.contracts.skill_payload_policy import (
    SKILL_OWNER_STATE_FILENAMES,
    is_skill_owner_state_target as _policy_is_skill_owner_state_target,
)
from ouroboros.contracts.task_constraint import normalize_task_constraint, resolve_payload_path
from ouroboros.project_facts import filter_out_project_store as _filter_out_project_store
from ouroboros.project_facts import project_store_access_block as _project_store_access_block
from ouroboros.protected_artifacts import block_reason_for_path
from ouroboros import store
from ouroboros.tool_access import (
    ResolvedResourceBinding,
    UserFilesPathBlockedError,
    active_tool_profile,
    build_resolved_resource_binding,
    decide_tool_access,
    normalize_root,
    normalize_runtime_data_path,
    user_files_path_block_reason,
)
from ouroboros.tools.registry import ToolContext, active_repo_dir_for
from ouroboros.tools.tool_result import ToolResult, _publish_tool_result, completed_local_read
from ouroboros.utils import safe_relpath

log = logging.getLogger(__name__)


_SKILL_OWNER_STATE_FILENAMES = SKILL_OWNER_STATE_FILENAMES


def _skill_owner_state_read_allowed(ctx: ToolContext) -> bool:
    """Skill owner-state reads follow the parent runtime mode for every actor."""
    try:
        from ouroboros.config import get_runtime_mode
        from ouroboros.runtime_mode_policy import runtime_mode_at_least

        return runtime_mode_at_least(get_runtime_mode(), "cyber_pro")
    except Exception:
        return False


def _runtime_data_read_check(ctx: ToolContext, *, listing: bool = False, root: str = "") -> Callable[[pathlib.Path], str]:
    """Prepare the parent's runtime rules once for this read/list/search call."""
    from ouroboros.tools.core_secret_paths import runtime_data_roots

    # Explicit user_files uses its own per-mode runtime-overlap policy. Cyber
    # parents already read those files there; child parity must not narrow it.
    roots = [] if root == "user_files" else runtime_data_roots(ctx)
    owner_state_blocked = not listing and not _skill_owner_state_read_allowed(ctx)

    def check(target: pathlib.Path) -> str:
        target = pathlib.Path(target).resolve(strict=False)
        for data_root in roots:
            try:
                relative = target.relative_to(data_root).as_posix()
            except ValueError:
                continue
            if reason := _project_store_access_block(relative):
                return str(reason)
            if (owner_state_blocked and _is_skill_owner_state_target(target, data_root)
                    and target.name.lower() != "review.json"):
                return "⚠️ DATA_READ_BLOCKED: skill owner state is not readable through generic data tools."
        return ""

    return check


def _runtime_data_read_block(ctx: ToolContext, target: pathlib.Path, *, listing: bool = False, root: str = "") -> str:
    return _runtime_data_read_check(ctx, listing=listing, root=root)(target)


def _direct_resource_binding(
    ctx: ToolContext,
    supplied: Any,
    *,
    root: str,
    operation: str,
    path: str,
    bucket: str = "",
    skill_name: str = "",
) -> ResolvedResourceBinding:
    if supplied is not None:
        return supplied
    return build_resolved_resource_binding(
        ctx,
        root=root,
        operation=operation,  # type: ignore[arg-type]
        path=path or ".",
        bucket=bucket,
        skill_name=skill_name,
    )


def _render_line_slice(path: str, content: str, max_lines: int = 2000, start_line: int = 1,
                       start_char: int = 0, extent: Optional[Dict[str, Any]] = None,
                       max_chars: Optional[int] = None) -> str:
    """Return a line-ranged file view with the shared read-tool header.

    ``extent`` (when a dict is passed) receives the DELIVERED window as FACTS,
    stamped AFTER the ``start_char`` cut: ``first_line`` (the first COMPLETE line
    in the body — a cursor that skips whole lines advances it, and a cursor that
    lands mid-line makes that line partial, so it is not counted), ``end_line``,
    ``total_lines``, ``body_start`` (where the numbered display begins),
    ``body_chars`` (source characters), ``partial_head`` (the body opens with
    a partial line), ``line_ends`` (display offsets of COMPLETE lines — partial
    head and tail rows excluded — on the very line
    definition the renderer cut by, so a consumer that cuts the body further
    counts complete lines from them and never recounts newlines), ``display_map``
    (the compact descriptor ``display_source_segments`` rebuilds the numbered
    rows from), plus the requested ``start_line``/``start_char``.
    An empty delivery (cursor at or past the window's end, or a start past EOF)
    is an EMPTY range (``end_line < first_line``), never an inverted claim. A
    consumer that must know what a read delivered (the native review episode's
    host-observed receipts) gets it from this arithmetic, never by parsing the
    header text back.

    ``start_char`` is a SUB-LINE cursor: it skips that many characters of the selected
    window's body before rendering. Delivery is char-bounded (the loop shows a
    result the measured frame cannot hold as head+tail with its exact source), so
    a single line longer than the room can never be delivered whole by any line
    window: the reader advances WITHIN it by re-reading the same window with a
    growing ``start_char``. ``max_chars`` bounds THIS view's body (after the cursor)
    to an exact character count, so the piece is chosen rather than cut from
    outside; the header then names the next ``start_char``. Both cursors count
    Unicode code points of the universal-newline source text, never display
    numbers. ``end_line`` is the last
    COMPLETE line delivered and ``partial_tail`` says the body ends mid-line.
    Disclosed in the header, so the view never silently masquerades as the whole line.
    """
    start_raw, max_raw = _coerce_line_window(start_line, max_lines)
    max_raw = max(1, max_raw)
    lines = content.splitlines(keepends=True)
    total = len(lines)
    start = max(1, min(start_raw, total + 1))
    end = min(start + max_raw - 1, total)
    window_lines = lines[start - 1:end]
    window = "".join(window_lines)
    # ONE line-boundary definition — the `splitlines` lines above (U+2028 and
    # U+2029 are line ends too; CR/CRLF never reach this renderer from
    # `read_file`, whose `read_text` opens with universal newlines and so
    # delivers LF) — serves the rendering, the cursor arithmetic AND the
    # consumer's bound-cut arithmetic: the end offset of every window line, in
    # window chars.
    ends: List[int] = []
    for line in window_lines:
        ends.append((ends[-1] if ends else 0) + len(line))
    offset = _coerce_start_char(start_char)
    budget = _coerce_max_chars(max_chars)
    body = window[offset:] if offset else window
    cut = budget is not None and len(body) > budget
    if cut:
        body = body[:budget]
    delivered_end = offset + len(body)  # window offset right after the body
    span = f"# {path} — lines {start}\u2013{end} of {total}"
    if cut:
        header = f"{span} (from char {offset} of this window, {len(body)} chars; next start_char={delivered_end})\n"
    elif offset:
        header = f"{span} (from char {offset} of this window)\n"
    else:
        header = f"{span}\n"
    whole = bisect.bisect_right(ends, offset)  # lines ending at or before the cursor: skipped whole
    partial_head = bool(body) and bool(offset) and not (whole and ends[whole - 1] == offset)  # landed mid-line
    first_line = start + whole + (1 if partial_head else 0)
    complete_through = bisect.bisect_right(ends, delivered_end)  # lines whose end the body reaches
    partial_tail = bool(body) and delivered_end < len(window) and not (complete_through and ends[complete_through - 1] == delivered_end)
    end_line = start + complete_through - 1 if cut else end  # a cut view ends at its last COMPLETE line
    if not body:
        first_line = end + 1  # nothing complete was delivered: an EMPTY range, never an inverted one
    # Display offsets are deliberately separate from source offsets: a number
    # prefix can be delivered without delivering any character of the file.
    display_parts: List[str] = []
    display_line_ends: List[int] = []
    display_pos = 0
    visible_line = start + whole  # a partial head row keeps its own source line number
    rows = body.splitlines(keepends=True)
    last_row = visible_line + len(rows) - 1
    for number, line in enumerate(rows, visible_line):
        prefix = _display_prefix(number)
        display_parts.extend((prefix, line))
        display_pos += len(prefix) + len(line)
        if not (partial_head and number == visible_line) and not (partial_tail and number == last_row):
            display_line_ends.append(display_pos)
    displayed = "".join(display_parts)
    if extent is not None:
        source_start = sum(len(line) for line in lines[:start - 1]) + min(offset, len(window))
        extent.update({"start_line": start, "end_line": end_line, "total_lines": total, "start_char": offset,
                       "first_line": first_line, "body_start": len(header), "body_chars": len(body),
                       "partial_head": partial_head, "partial_tail": partial_tail,
                       "line_ends": tuple(display_line_ends),
                       "display_map": {"scheme": _DISPLAY_SCHEME, "first_number": visible_line, "rows": len(rows)},
                       "complete_chars": len(content),
                       "complete_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                       "source_start_char": source_start, "source_end_char": source_start + len(body),
                       "range_basis": "unicode_text_universal_newlines", "source_masked": False})
    rendered = header + displayed
    return rendered


# The numbered display is deterministic: row k is ``_display_prefix(first_number + k)``
# followed by one source line on read_file's ``splitlines`` definition. A view
# carries only this descriptor (a 2,000-line read stays a few bytes of facts);
# consumers rebuild row coordinates from the rendered text and validate them.
_DISPLAY_SCHEME = "numbered_rows_v1"
_DISPLAY_BOUNDARY = re.compile("[\n\r\x0b\x0c\x1c\x1d\x1e\x85\u2028\u2029]")


def _display_prefix(number: int) -> str:
    return f"{number:>6}\t"


def display_source_segments(view: Dict[str, Any], rendered: str) -> Optional[List[Tuple[int, int, int]]]:
    """Rows of a numbered read as ``(display_begin, display_end, source_at)``.

    Display offsets are absolute in ``rendered``; ``source_at`` counts source
    characters from the window start. Rows are rebuilt from the deterministic
    display and checked against the view's descriptor; a rendering cut short
    yields the rows it still holds. ``None`` when the text is not this display.
    """
    body_start, chars, shape = view.get("body_start"), view.get("body_chars"), view.get("display_map")
    if (type(body_start) is not int or type(chars) is not int or body_start < 0 or chars < 0
            or not isinstance(shape, dict) or shape.get("scheme") != _DISPLAY_SCHEME
            or type(shape.get("first_number")) is not int or type(shape.get("rows")) is not int):
        return None
    segments: List[Tuple[int, int, int]] = []
    pos, source_at = body_start, 0
    for number in range(shape["first_number"], shape["first_number"] + shape["rows"]):
        prefix = _display_prefix(number)
        visible = rendered[pos:pos + len(prefix)]
        if visible != prefix:  # a cut inside or before this prefix ends the delivered rows
            return segments if pos + len(visible) == len(rendered) and prefix.startswith(visible) else None
        begin = pos + len(prefix)
        cap = begin + chars - source_at
        found = _DISPLAY_BOUNDARY.search(rendered, begin, cap)
        stop = found.end() if found else min(cap, len(rendered))
        if found and found.group() == "\r" and stop < cap and rendered.startswith("\n", stop):
            stop += 1
        segments.append((begin, stop, source_at))
        source_at += stop - begin
        pos = stop
    return segments if source_at == chars or pos == len(rendered) else None


def display_intervals_to_source(view: Dict[str, Any], rendered: str,
                                shown: List[Tuple[int, int]]) -> Optional[List[Tuple[int, int]]]:
    """File character intervals actually delivered by ``shown`` display intervals.

    ``shown`` are half-open offsets in ``rendered`` that reached the reader (one
    prefix, or a head and a tail around a hidden gap). Only source characters
    inside them count: number prefixes and hidden spans credit nothing. Results
    use the view's ``range_basis`` and start at its ``source_start_char``.
    """
    segments = display_source_segments(view, rendered)
    base = view.get("source_start_char")
    if segments is None or type(base) is not int:
        return None
    credited: List[Tuple[int, int]] = []
    for low, high in sorted(shown):
        for begin, stop, source_at in segments:
            first, last = max(begin, low), min(stop, high)
            if first >= last:
                continue
            interval = (base + source_at + first - begin, base + source_at + last - begin)
            if credited and interval[0] <= credited[-1][1]:
                credited[-1] = (credited[-1][0], max(credited[-1][1], interval[1]))
            else:
                credited.append(interval)
    return credited


def delivered_source_prefix(view: Dict[str, Any], rendered: str, shown: int) -> str:
    """Source characters in a delivered read_file display prefix.

    The renderer owns this mapping; consumers must not strip number prefixes
    with independent parsers. Rows are rebuilt from the delivered prefix alone,
    so a marker appended after the cut is never read as source; a cut within a
    prefix contributes zero chars.
    """
    delivered = rendered[:max(0, shown)]
    segments = display_source_segments(view, delivered) or []
    return "".join(delivered[begin:stop] for begin, stop, _ in segments)


def _read_source_text(target: pathlib.Path, extent: Optional[Dict[str, Any]]) -> str:
    """Bind the reader's text projection to the bytes from the same open.

    Keep the existing universal-newline text ABI. Character ranges address that
    text; source_revision names the actual file bytes, including CRLF. Reopening
    only to hash could bind a delivered view to a different concurrent revision.
    """
    try:
        raw = store.read_file_bytes(target)
    except FileNotFoundError:
        from ouroboros.evidence_packs import materialize_path
        if materialize_path(target) is None:  # a packed checkpoint handle is written back first
            raise
        raw = store.read_file_bytes(target)
    content = raw.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
    if extent is not None:
        extent.update(source_revision=hashlib.sha256(raw).hexdigest(), source_bytes=len(raw))
    return content


def _coerce_start_char(start_char: Any = 0) -> int:
    try:
        return max(0, int(start_char))
    except (TypeError, ValueError):
        return 0


def _coerce_max_chars(max_chars: Any = None) -> Optional[int]:
    """A positive character budget for one view; anything else means unbounded."""
    try:
        value = int(max_chars) if max_chars is not None else None
    except (TypeError, ValueError):
        return None
    return value if value is not None and value > 0 else None


def _coerce_line_window(start_line: Any = 1, max_lines: Any = 2000) -> tuple[int, int]:
    try:
        start_raw = int(start_line)
    except (TypeError, ValueError):
        start_raw = 1
    try:
        max_raw = int(max_lines)
    except (TypeError, ValueError):
        max_raw = 2000
    return start_raw, max(1, max_raw)


def _is_cognitive_data_path(norm: str) -> bool:
    text = str(norm or "").replace("\\", "/").lstrip("./")
    return text.startswith("memory/") or text in _MEMORY_AT_DRIVE_MEMORY


def _is_skill_owner_state_target(target: pathlib.Path, data_root: pathlib.Path) -> bool:
    return _policy_is_skill_owner_state_target(target, data_root)


class _ListingFailure(Exception):
    """A CONFINEMENT refusal: the resolved target escapes its root.

    v6.54.3 (review round 4): a refusal to list used to return a warning string
    INSIDE an ok-shaped JSON list — the exact error-inside-success shape the
    TB2.1 post-mortem showed silently poisoning reasoning. _list_files renders
    this as a leading ⚠️ LIST_FILES_ERROR, a first-class tool error. Discovery
    misses are the ``_ListingMiss`` subclass below and are NOT errors."""


class _ListingMiss(_ListingFailure):
    """The read-only DISCOVERY case: the named directory is simply not there.

    A confinement refusal and a miss are different outcomes. Looking for a
    directory that does not exist (or naming a file where a directory was
    expected) is what discovery IS, and colouring the whole task's execution
    axis for it made a later success on a differently-spelled path unable to
    credit the recovery. _list_files renders this as its own result with a
    leading ⚠️ LIST_FILES_NOT_FOUND and warning severity — still an explicit,
    marked refusal to list, never an error string inside an ok-shaped listing."""


def _list_dir(root: pathlib.Path, rel: str, max_entries: int = 500) -> List[str]:
    target = (root / safe_relpath(rel)).resolve()
    # CONFINE to the root before any iterdir: a resolved target that escapes (e.g. an
    # in-tree symlink pointing outside — common in untrusted child-created project /
    # deliverable trees behind the new read-only roots) is rejected, never listed.
    try:
        target.relative_to(root.resolve())
    except ValueError:
        raise _ListingFailure(f"Path escapes root: {rel}") from None
    if not target.exists():
        raise _ListingMiss(f"Directory not found: {rel}")
    if not target.is_dir():
        raise _ListingMiss(f"Not a directory: {rel}")
    items = []
    # A hard iterdir/permission/race failure PROPAGATES: _list_files renders it
    # as a first-class "⚠️ LIST_FILES_ERROR" tool error, never an ok-shaped JSON
    # listing carrying an error string inside (v6.54.3, review round 3).
    from ouroboros.evidence_packs import packed_handle_names_at
    entries = list(target.iterdir())  # a task's packed checkpoint handles list beside the loose ones
    present = {entry.name for entry in entries}
    entries += [target / name for name in packed_handle_names_at(target) if name not in present]
    for entry in sorted(entries):
        if len(items) >= max_entries:
            items.append(f"...(truncated at {max_entries})")
            break
        suffix = "/" if entry.is_dir() else ""
        items.append(entry.relative_to(root).as_posix() + suffix)
    return items


def _list_user_files_dir(ctx: ToolContext, root: pathlib.Path, target: pathlib.Path, max_entries: int = 500) -> List[str]:
    if not target.exists():
        raise _ListingMiss(f"Directory not found: {target}")
    if not target.is_dir():
        raise _ListingMiss(f"Not a directory: {target}")
    items: List[str] = []
    hidden = 0
    # A hard iterdir/permission/race failure PROPAGATES to the first-class
    # "⚠️ LIST_FILES_ERROR" path in _list_files (v6.54.3, review round 3).
    for entry in sorted(target.iterdir()):
        # operation="list" (capinv-447 / В23=A): the root principal's listing no
        # longer hides credential-SHAPED names — only control-plane/outside-home
        # entries stay omitted (and counted in the marker below).
        if user_files_path_block_reason(ctx, entry, operation="list"):
            hidden += 1
            continue
        if len(items) >= max_entries:
            items.append(f"...(truncated at {max_entries})")
            break
        suffix = "/" if entry.is_dir() else ""
        # An external-workspace listing outside the user_files home has no
        # home-relative form — render the absolute path instead of crashing
        # the whole listing on relative_to (v6.54.3: the TB2.1
        # "'/app/…' is not in the subpath of '/root'" class).
        try:
            rendered = entry.relative_to(root).as_posix()
        except ValueError:
            rendered = str(entry)
        items.append(rendered + suffix)
    if hidden:
        items.append(f"⚠️ {hidden} hidden/control entr{'y' if hidden == 1 else 'ies'} omitted from user_files listing.")
    return items


_TREE_SCAN_DEFAULT, _TREE_SCAN_MAX, _TREE_DEPTH_MAX, _TREE_GAP_NAMES = 2_000, 20_000, 20, 20
_TREE_GAP_HINTS = {
    "depth_limit": "directories at the depth limit, not opened; raise depth or list one directly",
    "generated_not_descended": "generated/vendor directories (search's skip set), not opened; list one directly",
    "protected_not_descended": "directories whose listing the resource policy refuses",
    "symlink_not_followed": "symlinks are named, never followed", "unreadable": "directories that could not be opened",
    "packed_index_not_scanned": "packed checkpoint names were not enumerated; list this exact directory with depth=1 and no include",
}
_TREE_ORDER = ("breadth-first live walk, not a snapshot; each directory's examined entries are sorted by name, "
               "but a directory cut by max_scan holds whichever entries the filesystem yielded first")


def _tree_list(ctx: ToolContext, binding: ResolvedResourceBinding, depth: int, include: str,
               max_entries: int, max_scan: int) -> str:
    """Bounded BFS: max_scan examined + one look-ahead, sorted within the bound; ordinary per-entry policy and named gaps."""
    from ouroboros.code_intelligence import SKIP_DIRS
    from ouroboros.code_search_rg import matches_include
    from ouroboros.evidence_packs import _checkpoint_path

    base, target, user_files = binding.base_path, binding.target_path, binding.root == "user_files"

    def shown(entry: pathlib.Path) -> str:  # an external user_files workspace has no home-relative form
        return entry.relative_to(base).as_posix() if entry.is_relative_to(base) else str(entry)

    if not user_files and not target.resolve().is_relative_to(base.resolve()):
        raise _ListingFailure(f"Path escapes root: {shown(target)}")
    if not target.is_dir():
        raise _ListingMiss(f"{'Not a directory' if target.exists() else 'Directory not found'}: {shown(target)}")
    hidden_by_policy = _runtime_data_read_check(ctx, listing=True, root=binding.root)
    gaps: Dict[str, List[str]] = {kind: [] for kind in _TREE_GAP_HINTS}
    entries: List[str] = []
    scanned, hidden, stop, cut, queue, folder = 0, 0, "", False, deque([(target, 1)]), target
    while queue and not stop:
        if scanned >= max_scan:
            stop = "scan_cap"  # the queued directories stay unopened
            break
        folder, level = queue.popleft()
        # Recheck queued paths after earlier iterations; directory snapshots are not atomic.
        if folder != target and (folder.is_symlink() or not folder.resolve().is_relative_to(target.resolve())):
            gaps["symlink_not_followed"].append(shown(folder))
            continue
        batch: List[pathlib.Path] = []
        try:
            with os.scandir(folder) as stream:
                for item in stream:
                    if scanned + len(batch) >= max_scan:
                        cut = True  # the one look-ahead entry: the directory goes on; it is not examined
                        break
                    batch.append(pathlib.Path(item.path))
        except OSError as exc:
            if folder == target:
                raise  # the requested directory itself: the first-class LIST_FILES_ERROR
            gaps["unreadable"].append(f"{shown(folder)}/ ({type(exc).__name__})")
            continue
        checkpoint = _checkpoint_path(folder)
        if checkpoint and len(checkpoint[2]) == 2:
            gaps["packed_index_not_scanned"].append(shown(folder))
        scanned += len(batch)
        for child in sorted(batch):
            if hidden_by_policy(child) or (user_files and user_files_path_block_reason(ctx, child, operation="list")):
                hidden += 1
                continue
            rel, link = shown(child), child.is_symlink()
            is_dir = not link and child.is_dir()
            gap = ("symlink_not_followed" if link else "" if not is_dir else "depth_limit" if level >= depth
                   else "generated_not_descended" if binding.root in {"active_workspace", "system_repo", "skill_payload"} and child.name in SKIP_DIRS
                   else "protected_not_descended" if block_reason_for_path(ctx, child, "static_introspection", binding) else "")
            if gap:
                gaps[gap].append(rel + ("/" if is_dir else ""))
            elif is_dir:
                queue.append((child, level + 1))
            if (include and is_dir) or not matches_include(child.name, include):
                continue
            if len(entries) >= max_entries:
                stop = "entry_cap"
                break
            entries.append(rel + ("/" if is_dir else ""))
        stop = stop or ("scan_cap" if cut else "")
    rows: List[Dict[str, Any]] = [{"kind": stop, "detail": (
        f"stopped at {'max_scan' if stop == 'scan_cap' else 'max_entries'} after examining {scanned} entries; "
        + (f"the rest of {shown(folder)}/ was not {'examined' if stop == 'entry_cap' else 'read'}; " if cut or stop == "entry_cap" else "")
        + f"{len(queue)} queued director(ies) were not opened")}] if stop else []
    rows += [{"kind": kind, "count": len(names), "paths": names[:_TREE_GAP_NAMES], "paths_omitted": max(0, len(names) - _TREE_GAP_NAMES), "detail": _TREE_GAP_HINTS[kind]}
             for kind, names in gaps.items() if names]
    rows += [{"kind": "policy_hidden", "count": hidden, "detail": "entries the list policy omits"}] if hidden else []
    complete = not rows
    return json.dumps({"path": f"{binding.root}:{shown(target)}", "depth": depth, **({"include": include} if include else {}),
                       "scanned": scanned, "max_scan": max_scan, "complete": complete, "order": _TREE_ORDER,
                       "entries": entries, "gaps": rows}, ensure_ascii=False, indent=2)


_MEMORY_AT_DRIVE_MEMORY = frozenset({
    "identity.md", "scratchpad.md", "dialogue_summary.md",
    "dialogue_blocks.json", "registry.md", "deep_review.md",
    "WORLD.md",
})


def _repo_read(
    ctx: ToolContext,
    path: str,
    max_lines: int = 2000,
    start_line: int = 1,
    start_char: int = 0,
    display_path: str | None = None,
    _resolved_binding: ResolvedResourceBinding | None = None,
    extent: Optional[Dict[str, Any]] = None,
    max_chars: Optional[int] = None,
) -> str:
    """Read a repo file; root-level memory names return a runtime_data read hint."""
    target = _resolved_binding.target_path if _resolved_binding is not None else ctx.repo_path(path)
    try:
        content = _read_source_text(target, extent)
    except FileNotFoundError:
        norm = path.strip().lstrip("./").replace("\\", "/")
        base = norm.rsplit("/", 1)[-1]
        if "/" not in norm and base in _MEMORY_AT_DRIVE_MEMORY:
            title = base.split('.')[0].title()
            return _publish_tool_result(ctx, ToolResult(
                status="ok",
                code="LEGACY_WARNING",
                text=(
                    f"⚠️ NOT_FOUND: '{path}' is not at the repo root.\n\n"
                    f"This file lives at `data_root/memory/{base}`, not in the "
                    f"git repo. Some memory artifacts are already summarized in "
                    f"context as `## {title}`, but raw memory state must be read "
                    f"from the data root. If you need the raw file, call "
                    f"`read_file(root='runtime_data', path='memory/{base}')`."
                ),
            ))
        return _publish_tool_result(ctx, ToolResult(
            status="ok",
            code="LEGACY_WARNING",
            text=f"⚠️ NOT_FOUND: file does not exist: {target}",
        ))
    return _render_line_slice(display_path or path, content, max_lines=max_lines, start_line=start_line,
                              start_char=start_char, extent=extent, max_chars=max_chars)


def _repo_list(
    ctx: ToolContext,
    dir: str = ".",
    max_entries: int = 500,
    _resolved_binding: ResolvedResourceBinding | None = None,
) -> str:
    repo_root = (
        _resolved_binding.base_path
        if _resolved_binding is not None
        else active_repo_dir_for(ctx)
    )
    target = _resolved_binding.target_path if _resolved_binding is not None else ctx.repo_path(dir)
    # ctx.repo_path already normalized absolute/redundant-prefix dirs; pass the
    # resulting root-relative form so _list_dir doesn't re-nest the raw input.
    try:
        listed_rel = target.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        listed_rel = dir
    items = _list_dir(repo_root, listed_rel, max_entries)
    runtime_check = _runtime_data_read_check(ctx, listing=True)
    items = [item for item in items if not runtime_check(repo_root / item.rstrip("/"))]
    return json.dumps(items, ensure_ascii=False, indent=2)


def _normalize_data_read_path(ctx: ToolContext, path: str) -> str:
    """Normalize paths that redundantly include the drive root."""
    return normalize_runtime_data_path(pathlib.Path(ctx.drive_root), path)


def _data_read(
    ctx: ToolContext,
    path: str,
    max_lines: int = 2000,
    start_line: int = 1,
    start_char: int = 0,
    display_path: str | None = None,
    _resolved_binding: ResolvedResourceBinding | None = None,
    extent: Optional[Dict[str, Any]] = None,
    max_chars: Optional[int] = None,
) -> str:
    """Read a drive text file; duplicate drive_root prefixes are stripped."""
    task_constraint = normalize_task_constraint(getattr(ctx, "task_constraint", None))
    norm = (_resolved_binding.target_path.relative_to(_resolved_binding.base_path).as_posix()
            if _resolved_binding is not None else _normalize_data_read_path(ctx, path))
    if (b := _project_store_access_block(norm)):
        return b
    if _resolved_binding is not None:
        target = _resolved_binding.target_path
    elif task_constraint and task_constraint.has_selected_skill and task_constraint.payload_root:
        try:
            target = resolve_payload_path(pathlib.Path(ctx.drive_root), task_constraint, norm)
        except ValueError as e:
            return _publish_tool_result(ctx, ToolResult(
                status="blocked", code="DATA_BLOCKED", text=f"⚠️ DATA_READ_BLOCKED: {e}",
            ))
    else:
        target = ctx.drive_path(norm)
    state_root = (
        _resolved_binding.base_path
        if _resolved_binding is not None
        else pathlib.Path(ctx.drive_root)
    )
    if (not _skill_owner_state_read_allowed(ctx)
            and _is_skill_owner_state_target(target, state_root) and target.name.lower() != "review.json"):
        # Owner item A.20: this refusal was the one in the family that shipped WITHOUT
        # the warning marker, so the adapter read a policy denial as a successful read
        # and the model was handed the refusal as if it were file content. The marker
        # is the approved text change; the code is the one the marker already implies.
        return _publish_tool_result(ctx, ToolResult(
            status="blocked",
            code="DATA_BLOCKED",
            text="⚠️ DATA_READ_BLOCKED: skill owner state is not readable through generic data tools.",
        ))
    try:
        content = _read_source_text(target, extent)
        start_raw, max_raw = _coerce_line_window(start_line, max_lines)
        # The cognitive full-read shortcut only applies to a DEFAULT read: an explicit
        # start_char or max_chars is a bounded-view request and must be honored, not swallowed.
        if (_is_cognitive_data_path(norm) and start_raw == 1 and max_raw == 2000
                and not _coerce_start_char(start_char) and _coerce_max_chars(max_chars) is None):
            if display_path is None:
                return content
            full_line_count = max(1, len(content.splitlines()))
            return _render_line_slice(display_path, content, max_lines=full_line_count, start_line=1, extent=extent)
        return _render_line_slice(display_path or norm, content, max_lines=max_raw, start_line=start_raw,
                                  start_char=start_char, extent=extent, max_chars=max_chars)
    except FileNotFoundError:
        if norm.replace("\\", "/").startswith("memory/"):
            explanation = (
                "Memory artifacts under memory/ are created lazily on first "
                "write. Treat this as an empty/absent state and proceed with "
                "initialization if that is the task."
            )
        else:
            explanation = (
                "This path does not exist yet. Treat it as an empty/absent "
                "state. Lazy-creation is not guaranteed for paths outside "
                "memory/; if this path was expected to exist, verify it was "
                "written correctly."
            )
        return _publish_tool_result(ctx, ToolResult(
            status="ok",
            code="LEGACY_WARNING",
            text=(
                f"⚠️ DATA_NOT_YET_CREATED: {path}\n\n"
                f"{explanation} Use list_files with root=runtime_data to confirm what currently exists."
            ),
        ))


def _data_list(
    ctx: ToolContext,
    dir: str = ".",
    max_entries: int = 500,
    _resolved_binding: ResolvedResourceBinding | None = None,
) -> str:
    task_constraint = normalize_task_constraint(getattr(ctx, "task_constraint", None))
    norm_dir = (_resolved_binding.target_path.relative_to(_resolved_binding.base_path).as_posix()
                if _resolved_binding is not None else _normalize_data_read_path(ctx, dir))
    # Whole-call block states are FIRST-CLASS tool errors, never ok-shaped
    # one-element JSON listings (v6.54.3, review round 5).
    if (b := _project_store_access_block(norm_dir)):
        return str(b)
    if _resolved_binding is not None:
        root = _resolved_binding.base_path
        try:
            rel = _resolved_binding.target_path.relative_to(root).as_posix() or "."
        except ValueError:
            return _publish_tool_result(ctx, ToolResult(
                status="blocked",
                code="DATA_BLOCKED",
                text="⚠️ DATA_LIST_BLOCKED: resolved target escapes runtime_data root.",
            ))
        items = _filter_out_project_store(norm_dir, _list_dir(root, rel, max_entries))
        return json.dumps(items, ensure_ascii=False, indent=2)
    if task_constraint and task_constraint.has_selected_skill and task_constraint.payload_root:
        try:
            root = resolve_payload_path(pathlib.Path(ctx.drive_root), task_constraint, dir)
        except ValueError as e:
            return _publish_tool_result(ctx, ToolResult(
                status="blocked", code="DATA_BLOCKED", text=f"⚠️ DATA_LIST_BLOCKED: {e}",
            ))
        items = _list_dir(root, ".", max_entries)
        return json.dumps(items, ensure_ascii=False, indent=2)
    # Drop any projects/<id> entry so a generic root listing never exposes the store.
    items = _filter_out_project_store(_normalize_data_read_path(ctx, dir), _list_dir(ctx.drive_root, dir, max_entries))
    return json.dumps(items, ensure_ascii=False, indent=2)


def _profile_roots_hint(ctx: ToolContext, operation: str) -> str:
    """Name the roots THIS profile can actually use for ``operation``.

    The host already knows the answer (the Tool API v2 matrix); telling the
    model turns a dead-end error into a self-correcting retry instead of a
    probe loop over blocked roots (v6.70.0)."""
    try:
        from ouroboros.tool_access import operation_roots

        return f" Roots your profile can {operation}: {operation_roots(active_tool_profile(ctx), operation)}."
    except Exception:
        return ""


def _access_or_block(ctx: ToolContext, root: str, operation: str) -> tuple[str, str]:
    try:
        normalized = normalize_root(root)
    except ValueError as exc:
        return "", _publish_tool_result(ctx, ToolResult(
            status="error",
            code="TOOL_ARG_ERROR",
            text=f"⚠️ TOOL_ARG_ERROR: {exc}{_profile_roots_hint(ctx, operation)}",
        ))
    profile = active_tool_profile(ctx)
    decision = decide_tool_access(profile=profile, root=normalized, operation=operation)  # type: ignore[arg-type]
    if not decision.allow:
        return "", _publish_tool_result(ctx, ToolResult(
            status="blocked",
            code="ACCESS_BLOCKED",
            text=f"⚠️ TOOL_ACCESS_BLOCKED: {str(decision.reason).rstrip('.')}.",
        ))
    return normalized, ""


def _root_display_path(root: str, path: str) -> str:
    return f"{root}:{safe_relpath(str(path or '.')).removeprefix('./') or '.'}"


def _annotate_reread(ctx: ToolContext, target: Any, start_line: int, max_lines: int, result: str,
                     start_char: int = 0, max_chars: Optional[int] = None) -> str:
    """Nudge against unchanged rereads, keyed per task by resolved path and slice.
    Size/mtime changes or different slices suppress the hint. Advisory only:
    it asks the mind to act on already-read text, never blocks."""
    try:
        resolved = pathlib.Path(target).resolve(strict=False)
        st = resolved.stat()
    except (OSError, TypeError, ValueError):
        return result
    if not isinstance(result, str) or result.startswith("⚠️"):
        return result
    key = f"{resolved}|{int(start_line)}|{int(max_lines)}|{_coerce_start_char(start_char)}|{_coerce_max_chars(max_chars)}"
    sig = (st.st_size, st.st_mtime_ns)
    seen = getattr(ctx, "_read_file_seen", None)
    if not isinstance(seen, dict):
        ctx._read_file_seen = seen = {}
    unchanged = seen.get(key) == sig
    seen[key] = sig
    return result + (
        "\n\nℹ️ This exact view is unchanged since you already read it this task — "
        "re-reading is usually wasted budget; act on what you have." if unchanged else "")


def _stamp_read_view(ctx: ToolContext, target: Any, opened: str, opened_root: str,
                     extent: Dict[str, Any], rendered: str) -> str:
    """Stamp this read's resolved target, root-relative opened_path, normalized
    opened_root and renderer window as ctx.last_read_view for native review receipts.
    The registry normalizes absolute, padded and redundant-root spellings first.
    Per-call binding: _read_file resets on entry; the episode clears before dispatch.
    These sole writers are statically pinned. Like reread hints, this is per-context
    disclosure, never a gate or a change to the actual read."""
    identity_note = ""
    if opened_root == "runtime_data" and opened == "memory/identity.md":
        from ouroboros.tool_access import canonical_data_root
        from ouroboros.tool_access_reads import canonical_runtime_reader

        canonical = canonical_data_root(ctx)
        physical = pathlib.Path(target).resolve(strict=False)
        source = ("canonical" if physical == (canonical / opened).resolve(strict=False) else
                  "execution_drive" if physical == (pathlib.Path(ctx.drive_root) / opened).resolve(strict=False)
                  else "inherited_runtime")
        identity_note = (f"identity_source={source}; source_path={physical}\n"
                         "Any revision identifies only the bytes returned here. For shared identity publication, read:\n"
                         f"{canonical_runtime_reader(ctx, canonical, opened)}\n")
    if "body_start" in extent:
        if opened_root == "runtime_data" and opened in ("memory/identity.md", "memory/WORLD.md") and extent.get("source_revision"):
            revision_line = f"revision={extent['source_revision']}\n" + identity_note
            rendered = revision_line + rendered
            extent["body_start"] += len(revision_line)
        view = {"target": str(target), "opened_path": str(opened), "opened_root": str(opened_root), **extent}
        ctx.last_read_view = view
        from ouroboros.delegate_output import acknowledge_staged_output_read
        from ouroboros.tools.tool_result import _published_tool_result, _replace_tool_result

        # display_map (a few bytes) lets the delivery credit decode numbered rows;
        # per-line facts such as line_ends stay off the bounded ToolResult meta.
        compact = {key: view[key] for key in (
            "target", "opened_root", "body_start", "body_chars", "source_start_char", "source_end_char",
            "complete_chars", "complete_sha256", "source_revision", "range_basis", "source_masked",
            "display_map") if key in view}
        prior = _published_tool_result(ctx, None)
        result = (_replace_tool_result(prior, meta_updates={"read_view": compact})
                  if isinstance(prior, ToolResult) and prior.text == rendered else
                  ToolResult(status="ok", code="OK", text=rendered, meta={"read_view": compact}))
        rendered = _publish_tool_result(ctx, result)
        acknowledge_staged_output_read(ctx, compact, rendered)
    elif identity_note:
        from ouroboros.tools.tool_result import append_published_text

        rendered = append_published_text(ctx, rendered, "\n" + identity_note)
    return rendered


@completed_local_read
def _read_file(
    ctx: ToolContext,
    path: str,
    root: str | None = None,
    max_lines: int = 2000,
    start_line: int = 1,
    start_char: int = 0,
    bucket: str = "",
    skill_name: str = "",
    _resolved_binding: ResolvedResourceBinding | None = None,
    max_chars: Optional[int] = None,
) -> str:
    # Reset first: a blocked or missing read must never inherit the previous
    # read's extent (the renderer fills `extent` only when it rendered).
    ctx.last_read_view = None
    extent: Dict[str, Any] = {}
    if not root:
        from ouroboros.tools.tool_resolution import _root_containing_absolute_path

        root = _root_containing_absolute_path(ctx, "read_file", path) or "active_workspace"
    normalized, block = _access_or_block(ctx, root, "read")
    if block:
        return block
    try:
        binding = _direct_resource_binding(
            ctx, _resolved_binding, root=normalized, operation="read", path=path,
            bucket=bucket, skill_name=skill_name,
        )
    except UserFilesPathBlockedError as exc:
        return _publish_tool_result(ctx, ToolResult(
            status="blocked", code="USER_FILES_PATH_BLOCKED", text=f"⚠️ USER_FILES_PATH_BLOCKED: {exc}"))
    except Exception as exc:
        return _publish_tool_result(ctx, ToolResult(
            status="error", code="LEGACY_TOOL_ERROR", text=f"⚠️ READ_FILE_ERROR: {type(exc).__name__}: {exc}"))
    target = binding.target_path
    try:
        opened = target.relative_to(binding.base_path).as_posix()  # what is read, relative to its root
    except ValueError:
        opened = str(target)
    opened_root = str(binding.root)  # the NORMALIZED root the binding used, not the model's spelling
    if runtime_block := _runtime_data_read_block(ctx, target, root=binding.root):
        return _publish_tool_result(ctx, ToolResult(status="blocked", code="DATA_BLOCKED", text=runtime_block))
    protected_block = block_reason_for_path(ctx, target, "read_bytes", binding)
    if protected_block:
        return protected_block
    if target.is_dir():  # admitted, but no file here: the read becomes the listing call to make
        call = f"list_files(root={json.dumps(normalized)}, path={json.dumps(str(target))}" + (  # preserve canonical/physical identity
            f", bucket={json.dumps(bucket or binding.source)}, skill_name={json.dumps(skill_name or binding.skill_name)}"
            if normalized == "skill_payload" else "")
        return _publish_tool_result(ctx, ToolResult(status="ok", code="LEGACY_WARNING", text=(
            f"⚠️ NOT_FOUND: {_root_display_path(normalized, opened)} is a directory, not a file. List it:\n"
            f"  {call})")))
    if normalized in {"active_workspace", "system_repo", "runtime_data"}:
        reader, display_path = ((_data_read, _root_display_path(normalized, path)) if normalized == "runtime_data" else
                                (_repo_read, f"{target} (project room)" if binding.source == "project_room"
                                 else _root_display_path(normalized, opened)))
        return _stamp_read_view(ctx, target, opened, opened_root, extent, _annotate_reread(ctx, target, start_line, max_lines, reader(
            ctx, path, max_lines=max_lines, start_line=start_line, start_char=start_char, display_path=display_path,
            _resolved_binding=binding, extent=extent, max_chars=max_chars,
        ), start_char=start_char, max_chars=max_chars))
    try:
        content = _read_source_text(target, extent)
        rendered = _render_line_slice(_root_display_path(normalized, path), content,
                                      max_lines=max_lines, start_line=start_line,
                                      start_char=start_char, extent=extent, max_chars=max_chars)
        return _stamp_read_view(ctx, target, opened, opened_root, extent,
                                _annotate_reread(ctx, target, start_line, max_lines, rendered,
                                                 start_char=start_char, max_chars=max_chars))
    except FileNotFoundError:
        return _publish_tool_result(ctx, ToolResult(
            status="ok", code="LEGACY_WARNING", text=f"⚠️ NOT_FOUND: {_root_display_path(normalized, path)} (resolved: {target})"))
    except UserFilesPathBlockedError as exc:
        # Typed POLICY refusal, not an executor failure: the runtime said "no"
        # to this read. The distinct prefix routes it into the v6.57.0
        # policy-denial partition instead of a generic error that falsely
        # degrades a shipped task to tool_failure.
        return _publish_tool_result(ctx, ToolResult(
            status="blocked", code="USER_FILES_PATH_BLOCKED", text=f"⚠️ USER_FILES_PATH_BLOCKED: {exc}"))
    except Exception as exc:
        return _publish_tool_result(ctx, ToolResult(
            status="error", code="LEGACY_TOOL_ERROR", text=f"⚠️ READ_FILE_ERROR: {type(exc).__name__}: {exc}"))


@completed_local_read
def _list_files(
    ctx: ToolContext,
    path: str = ".",
    root: str = "active_workspace",
    max_entries: int = 500,
    bucket: str = "",
    skill_name: str = "",
    _resolved_binding: ResolvedResourceBinding | None = None,
    depth: int = 1,
    include: str = "",
    max_scan: int = _TREE_SCAN_DEFAULT,
) -> str:
    try:  # the default (depth 1, no include) is the unchanged single-level listing
        depth = min(max(1, int(depth or 1)), _TREE_DEPTH_MAX)
        max_scan = min(max(1, int(max_scan or _TREE_SCAN_DEFAULT)), _TREE_SCAN_MAX)
    except (TypeError, ValueError):
        return _publish_tool_result(ctx, ToolResult(
            status="error", code="TOOL_ARG_ERROR", text="⚠️ TOOL_ARG_ERROR: depth and max_scan must be integers."))
    normalized, block = _access_or_block(ctx, root, "list")
    if block:
        return block
    try:
        binding = _direct_resource_binding(
            ctx, _resolved_binding, root=normalized, operation="list", path=path,
            bucket=bucket, skill_name=skill_name,
        )
    except UserFilesPathBlockedError as exc:
        return _publish_tool_result(ctx, ToolResult(
            status="blocked", code="USER_FILES_PATH_BLOCKED", text=f"⚠️ USER_FILES_PATH_BLOCKED: {exc}"))
    except Exception as exc:
        return _publish_tool_result(ctx, ToolResult(
            status="error", code="LEGACY_TOOL_ERROR", text=f"⚠️ LIST_FILES_ERROR ({type(exc).__name__}): {exc}"))
    if runtime_block := _runtime_data_read_block(ctx, binding.target_path, listing=True, root=binding.root):
        return _publish_tool_result(ctx, ToolResult(status="blocked", code="DATA_BLOCKED", text=runtime_block))
    protected_list_block = block_reason_for_path(ctx, binding.target_path, "static_introspection", binding)
    if protected_list_block:
        return protected_list_block
    try:
        # Every listing branch runs inside this try: a hard iterdir/permission/
        # race failure from any helper becomes the first-class LIST_FILES_ERROR
        # below (v6.54.3, review round 3 — helpers no longer swallow it into an
        # ok-shaped listing).
        if depth > 1 or include:
            return _tree_list(ctx, binding, depth, str(include), max_entries, max_scan)
        if normalized in {"active_workspace", "system_repo", "runtime_data"}:
            lister = _data_list if normalized == "runtime_data" else _repo_list
            return lister(ctx, dir=path, max_entries=max_entries, _resolved_binding=binding)
        if normalized == "skill_payload":
            rel = binding.target_path.relative_to(binding.base_path).as_posix() or "."
            items = _list_dir(binding.base_path, rel, max_entries)
            return json.dumps(items, ensure_ascii=False, indent=2)
        if normalized == "user_files":
            items = _list_user_files_dir(ctx, binding.base_path, binding.target_path, max_entries)
            return json.dumps(items, ensure_ascii=False, indent=2)
        rel = binding.target_path.relative_to(binding.base_path).as_posix() or "."
        items = _list_dir(binding.base_path, rel, max_entries)
        return json.dumps(items, ensure_ascii=False, indent=2)
    except _ListingMiss as exc:
        # A miss is discovery, not a failed tool: the same warning severity the
        # absent-memory-file read already uses (DATA_NOT_YET_CREATED below).
        return _publish_tool_result(ctx, ToolResult(
            status="ok", code="LEGACY_WARNING", text=f"⚠️ LIST_FILES_NOT_FOUND: {exc}",
        ))
    except _ListingFailure as exc:
        return _publish_tool_result(ctx, ToolResult(
            status="error", code="LEGACY_TOOL_ERROR", text=f"⚠️ LIST_FILES_ERROR: {exc}",
        ))
    except UserFilesPathBlockedError as exc:
        # Typed POLICY refusal (see _read_file): policy denial, not tool_failure.
        return _publish_tool_result(ctx, ToolResult(
            status="blocked", code="USER_FILES_PATH_BLOCKED", text=f"⚠️ USER_FILES_PATH_BLOCKED: {exc}"))
    except Exception as exc:
        # A hard failure is a first-class tool error, never a JSON "listing" that
        # reads as success with an error string inside (v6.54.3: that shape
        # silently poisoned reasoning in 63% of TB2.1 trials).
        return _publish_tool_result(ctx, ToolResult(
            status="error", code="LEGACY_TOOL_ERROR", text=f"⚠️ LIST_FILES_ERROR ({type(exc).__name__}): {exc}"))
