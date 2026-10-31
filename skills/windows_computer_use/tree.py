"""windows_computer_use.tree - walk results and their honest rendering.

Ported from the platform-free half of Praxis/Hélène ``uia.rs``: the Node
shape (missing state keys mean unknown, never false), the Walk counters
(discovered_unread, subtrees_unread, truncated_by...), tree/flat shaping
with the keep-the-path filter rule, and ``render`` which names every limit
and every cut. Runs on any OS.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from typing import Any, Optional

from .core import ReadPlan, clip_utf16

COORDINATES_NOTE = (
    "screen pixels on the virtual desktop; click points are the rect center, "
    "for use with coordinate input tools if you have them"
)
STATE_SEMANTICS = (
    "a missing state key means the element does not expose it; it does not mean false. "
    "A field named in a node's unread list failed to read - that is not absence either"
)
IDS_NOTE = (
    "ids index this answer only and are not stable across reads; "
    "use hwnd or automation_id to re-find an element"
)

WINDOW_READ_UIA_DETAIL = (
    "IUIAutomation control view, properties and patterns read from one cache "
    "request, walked breadth-first with a per-node deadline in this call's "
    "worker thread (isolated-dependency child process)"
)
WINDOW_READ_WIN32_DETAIL = "plain Win32 child-window walk with timed WM_GETTEXT; no COM involved"


def _clip(text: Optional[str], limit: int) -> tuple[Optional[str], bool]:
    if text is None or text == "":
        return (None if not text else text), False
    clipped, cut = clip_utf16(text, limit)
    return clipped, cut


@dataclass
class Rect:
    left: int
    top: int
    right: int
    bottom: int

    def is_empty(self) -> bool:
        return self.right <= self.left or self.bottom <= self.top

    def json(self) -> dict[str, Any]:
        return {
            "left": self.left,
            "top": self.top,
            "right": self.right,
            "bottom": self.bottom,
            "width": self.right - self.left,
            "height": self.bottom - self.top,
            # A point a mouse can actually click: the geometric center.
            "center": {
                "x": self.left + (self.right - self.left) // 2,
                "y": self.top + (self.bottom - self.top) // 2,
            },
        }


@dataclass
class WindowInfo:
    hwnd: int
    title: str
    cls: str
    pid: int
    rect: Optional[Rect]
    foreground: bool
    resolved_from: str
    dpi: int
    unread: tuple[str, ...] = ()

    def json(self) -> dict[str, Any]:
        out = {
            "hwnd": f"0x{self.hwnd:X}",
            "title": self.title,
            "class": self.cls,
            "pid": self.pid,
            "rect": self.rect.json() if self.rect else None,
            "foreground": self.foreground,
            "resolved_from": self.resolved_from,
            "dpi": self.dpi,
            "scale": round(self.dpi / 96.0, 3),
        }
        if self.unread:
            out["unread"] = list(self.unread)
        return out


@dataclass
class Node:
    parent: Optional[int] = None
    depth: int = 0
    role: str = ""
    localized_role: Optional[str] = None
    name: Optional[str] = None
    value: Optional[str] = None
    automation_id: Optional[str] = None
    cls: Optional[str] = None
    hwnd: Optional[int] = None
    rect: Optional[Rect] = None
    enabled: Optional[bool] = None
    focused: Optional[bool] = None
    offscreen: Optional[bool] = None
    checked: Optional[str] = None
    selected: Optional[bool] = None
    expanded: Optional[str] = None
    text_truncated: bool = False
    children_unread: Optional[str] = None
    unread: tuple[str, ...] = ()

    def haystack(self) -> str:
        parts = [self.role]
        for part in (self.name, self.value, self.automation_id, self.cls, self.localized_role):
            if part:
                parts.append(part)
        return "\x01".join(parts).lower()

    def json(self, id_: int, children: Optional[list[Any]] = None) -> dict[str, Any]:
        out: dict[str, Any] = {"id": id_, "role": self.role, "depth": self.depth}
        if self.parent is not None:
            out["parent"] = self.parent
        for key in ("name", "value", "automation_id", "class", "localized_role"):
            value = getattr(self, "cls" if key == "class" else key.replace("class", "cls"))
            if value is not None:
                out[key] = value
        if self.hwnd:
            out["hwnd"] = f"0x{self.hwnd:X}"
        if self.rect and not self.rect.is_empty():
            out["rect"] = self.rect.json()
        state: dict[str, Any] = {}
        for key in ("enabled", "focused", "offscreen", "selected"):
            value = getattr(self, key)
            if value is not None:
                state[key] = value
        for key in ("checked", "expanded"):
            value = getattr(self, key)
            if value is not None:
                state[key] = value
        if state:
            out["state"] = state
        if self.text_truncated:
            out["text_truncated"] = True
        if self.unread:
            out["unread"] = list(self.unread)
        if self.children_unread:
            out["children_unread"] = self.children_unread
        if children:
            out["children"] = children
        return out


@dataclass
class Walk:
    nodes: list[Node] = field(default_factory=list)
    discovered_unread: int = 0
    subtrees_unread: int = 0
    skipped_offscreen: int = 0
    depth_reached: int = 0
    truncated_by: list[str] = field(default_factory=list)
    walk_ms: int = 0
    result_budget: int = 0

    def mark(self, reason: str) -> None:
        if reason not in self.truncated_by:
            self.truncated_by.append(reason)


def children_of(nodes: list[Node]) -> list[list[int]]:
    children: list[list[int]] = [[] for _ in nodes]
    for index, node in enumerate(nodes):
        if node.parent is not None and node.parent < len(children):
            children[node.parent].append(index)
    return children


def keep_flags(plan: ReadPlan, nodes: list[Node], children: list[list[int]]) -> list[bool]:
    if not plan.needle:
        return [True] * len(nodes)
    keep = [plan.needle in node.haystack() for node in nodes]
    if plan.shape == "flat":
        return keep
    # A matched node without its ancestors can be neither shown nor re-found -
    # so the path to it is kept whole.
    for index in range(len(nodes) - 1, -1, -1):
        if any(keep[child] for child in children[index]):
            keep[index] = True
    return keep


def tree_json(index: int, nodes: list[Node], children: list[list[int]], keep: list[bool]) -> Any:
    kids = [
        tree_json(child, nodes, children, keep) for child in children[index] if keep[child]
    ]
    return nodes[index].json(index, kids)


# Reserve space within the host's 512 KiB protocol for its envelope. Measure
# the actual JSON-string encoding of our JSON answer: quotes and backslashes
# can nearly double its size. Cut here, where the lost nodes can be named.
RESULT_BYTES = 400_000


def render(
    plan: ReadPlan,
    window: WindowInfo,
    walk: Walk,
    backend: str,
    backend_detail: str,
    fallback_reason: Optional[str],
    elapsed_ms: int,
    extra_notes: Optional[list[str]] = None,
    workers_live: int = 0,
    max_bytes: int = RESULT_BYTES,
) -> dict[str, Any]:
    """The answer; a reading too large for ``max_bytes`` keeps its longest
    breadth-first prefix that fits and says so (``truncated_by: result_size``)."""
    args = (plan, window, backend, backend_detail, fallback_reason, elapsed_ms, extra_notes, workers_live)
    out = _render(walk, *args)
    if _size(out) <= max_bytes or len(walk.nodes) <= 1:
        return out
    low, high = 1, len(walk.nodes) - 1
    while low < high:  # parents precede children, so every prefix is a valid tree
        middle = (low + high + 1) // 2
        if _size(_render(_prefix(walk, middle, max_bytes), *args)) <= max_bytes:
            low = middle
        else:
            high = middle - 1
    return _render(_prefix(walk, low, max_bytes), *args)


def json_text(value: Any) -> str:
    """Readable Unicode JSON, escaping lone UTF-16 surrogates for the UTF-8 relay."""
    return json.dumps(value, ensure_ascii=False).encode("utf-8", errors="backslashreplace").decode("utf-8")


def _size(out: dict[str, Any]) -> int:
    answer = json_text(out)
    return len(json.dumps(answer, ensure_ascii=False).encode("utf-8"))


_DISPLAY_TEXT = {
    "role", "name", "value", "automation_id", "class", "localized_role", "title",
    "name_contains", "value_contains", "text_contains", "error", "fallback_reason",
}
_DISPLAY_LISTS = {
    "elements": ("matched", "shown", "omitted"),
    "candidates": ("matched", "shown", "omitted"),
    "windows": ("count", "shown", "omitted"),
    "unreadable_candidates": ("unreadable", "unreadable_shown", "unreadable_omitted"),
}


def _display_text(value: Any, limit: int) -> Any:
    """Copy for display ONLY: never changes the values used by selector matching."""
    if isinstance(value, list):
        return [_display_text(item, limit) for item in value]
    if not isinstance(value, dict):
        return value
    out = {}
    clipped = {}
    for key, item in value.items():
        if key in _DISPLAY_TEXT and isinstance(item, str):
            out[key], cut = clip_utf16(item, limit)
            if cut:
                clipped[key] = {
                    "utf16_units": sum(2 if ord(char) > 0xFFFF else 1 for char in item),
                    "shown_utf16_units": sum(2 if ord(char) > 0xFFFF else 1 for char in out[key]),
                }
        else:
            out[key] = _display_text(item, limit)
    if clipped:
        out["clipped_fields"] = clipped
    return out


def _display_counts(out: dict[str, Any], limit: Optional[int] = None) -> dict[str, Any]:
    out = dict(out)
    for key, (total, shown, omitted) in _DISPLAY_LISTS.items():
        if key in out:
            items = out[key] if limit is None else out[key][:limit]
            out[key] = items
            out[shown] = len(items)
            out[omitted] = max(0, out.get(total, len(items)) - len(items))
            if out[omitted]:
                out["truncated"] = True
    return out


def fit_result(out: dict[str, Any], max_bytes: int = RESULT_BYTES) -> dict[str, Any]:
    """Fit all four tools into the actual JSON-string relay, not just read trees.

    Ordinary answers keep their full strings. On overflow, free-form display
    fields use window_read's existing 4000 UTF-16-unit ceiling, then lists keep
    the longest fitting prefix. Counts, action outcomes and refusal reasons
    survive; omitted text is never represented as an unread provider field.
    """
    out = _display_counts(out)
    if _size(out) <= max_bytes:
        return out
    from .core import READ_CEILINGS

    out = _display_text(out, READ_CEILINGS["max_text_chars"])
    out["truncated"] = True
    out["truncated_by"] = list(dict.fromkeys([*out.get("truncated_by", []), "result_size"]))
    out["result_budget_bytes"] = max_bytes
    out["display_hint"] = (
        "display-only cut; see shown/omitted and clipped_fields (UTF-16 units). "
        "Narrow the selector, read an inner hwnd or filter the window to see the rest; "
        "matching used the original values, not these display prefixes"
    )
    if _size(out) <= max_bytes:
        return out
    low, high = 0, max((len(out.get(key, [])) for key in _DISPLAY_LISTS), default=0)
    while low < high:
        middle = (low + high + 1) // 2
        if _size(_display_counts(out, middle)) <= max_bytes:
            low = middle
        else:
            high = middle - 1
    return _display_counts(out, low)


def _prefix(walk: Walk, count: int, max_bytes: int) -> Walk:
    cut = dataclasses.replace(walk, nodes=walk.nodes[:count], truncated_by=list(walk.truncated_by),
                              discovered_unread=walk.discovered_unread + len(walk.nodes) - count)
    cut.mark("result_size")
    cut.result_budget = max_bytes
    return cut


def _render(
    walk: Walk,
    plan: ReadPlan,
    window: WindowInfo,
    backend: str,
    backend_detail: str,
    fallback_reason: Optional[str],
    elapsed_ms: int,
    extra_notes: Optional[list[str]],
    workers_live: int,
) -> dict[str, Any]:
    children = children_of(walk.nodes)
    keep = keep_flags(plan, walk.nodes, children)
    kept = sum(1 for value in keep if value)

    out: dict[str, Any] = {
        "ok": True,
        "capability": "window_read",
        "backend": backend,
        "backend_detail": backend_detail,
        "fallback_reason": fallback_reason,
        "window": window.json(),
        "shape": plan.shape,
    }

    if plan.shape == "tree":
        out["root"] = (
            tree_json(0, walk.nodes, children, keep) if walk.nodes and keep[0] else None
        )
    else:
        out["items"] = [
            node.json(index, None) for index, node in enumerate(walk.nodes) if keep[index]
        ]

    out["nodes_read"] = len(walk.nodes)
    out["nodes_returned"] = kept
    out["discovered_unread"] = walk.discovered_unread
    out["subtrees_unread"] = walk.subtrees_unread
    out["skipped_offscreen"] = walk.skipped_offscreen
    out["nodes_with_unread_fields"] = sum(1 for node in walk.nodes if node.unread)
    out["depth_reached"] = walk.depth_reached
    out["truncated"] = bool(walk.truncated_by)
    out["truncated_by"] = list(walk.truncated_by)
    # The full element count is known ONLY when the walk reached the end.
    out["total_known"] = not walk.truncated_by
    out["limits"] = plan.limits.json()
    out["elapsed_ms"] = elapsed_ms
    out["walk_ms"] = walk.walk_ms
    out["uia_workers_live"] = workers_live
    out["coordinates"] = COORDINATES_NOTE
    out["state_semantics"] = STATE_SEMANTICS
    out["ids"] = IDS_NOTE

    notes = list(plan.notes)
    notes.extend(extra_notes or [])
    if plan.needle:
        out["filter"] = {
            "text_contains": plan.needle,
            "matched_or_ancestor": kept,
            "of_nodes_read": len(walk.nodes),
            "rule": (
                "tree keeps matching elements and their ancestors as the path"
                if plan.shape == "tree"
                else "flat keeps only matching elements"
            ),
        }
    if "max_nodes" in walk.truncated_by:
        notes.append(
            f"stopped at max_nodes={plan.limits.max_nodes}: {walk.discovered_unread} more "
            "elements were already discovered and left unread, and there may be more behind them"
        )
    if "timeout" in walk.truncated_by:
        notes.append(
            f"stopped at timeout_ms={plan.limits.timeout_ms}: this reading is partial, "
            "not the whole window"
        )
    if "max_depth" in walk.truncated_by:
        notes.append(
            f"{walk.subtrees_unread} subtrees were left unopened at "
            f"max_depth={plan.limits.max_depth}"
        )
    if "max_children_per_node" in walk.truncated_by:
        notes.append(
            f"at least one element has more than max_children_per_node="
            f"{plan.limits.max_children_per_node} children; the rest of that list was not read"
        )
    if "result_size" in walk.truncated_by:
        notes.append(
            f"the answer was cut to {len(walk.nodes)} elements to stay within {walk.result_budget} "
            "encoded-result bytes; read an inner window (hwnd), filter with text_contains or lower max_depth "
            "to see the rest"
        )
    if walk.skipped_offscreen > 0:
        notes.append(
            f"visible_only=true skipped {walk.skipped_offscreen} offscreen elements "
            "together with everything under them"
        )
    if elapsed_ms > 1000 and backend == "uia":
        notes.append(
            "this read took over a second; one common cause is a first read of a "
            "Chromium/Electron window, which wakes its accessibility engine - later "
            "reads of the same window are usually much faster"
        )
    out["notes"] = notes
    return out
