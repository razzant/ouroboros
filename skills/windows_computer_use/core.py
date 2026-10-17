"""windows_computer_use.core - the platform-free contract layer.

Ported 1:1 in spirit and contract from Yegor Kosyrev's Praxis/Hélène
``element.rs`` (+ the plan/limit/render logic of ``uia.rs``): selector
matching, the choose() refusal-on-ambiguity, plan parsing with honest limit
clamping, the three different "not found" answers, receipts that never claim
success, and UTF-16-safe text clipping. Everything here runs on any OS and is
covered by tests/test_core.py (ported from the Rust unit tests).

The Windows-specific layers (UIA via comtypes, the Win32 fallback) live in
``win_uia`` and ``win32fb`` and import nothing from this module's tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

TOOL_WINDOW_READ = "window_read"
TOOL_ELEMENT_ACT = "element_act"
TOOL_ELEMENT_FIND = "element_find"

ACTS = (
    "invoke",
    "set_value",
    "toggle",
    "expand",
    "collapse",
    "select",
    "scroll_into_view",
    "focus",
)
NON_MUTATING_ACTS = {"focus", "scroll_into_view"}


# ─── limits (ported from uia.rs Limits; "no silent limits" is the law) ──────────

READ_DEFAULTS = {
    "max_nodes": 400,
    "max_depth": 24,
    "max_children_per_node": 128,
    "max_text_chars": 240,
    "timeout_ms": 1800,
}
READ_CEILINGS = {
    "max_nodes": 5000,
    "max_depth": 64,
    "max_children_per_node": 2000,
    "max_text_chars": 4000,
    "timeout_ms": 20000,
}
READ_FLOORS = {
    "max_nodes": 1,
    "max_depth": 0,
    "max_children_per_node": 1,
    "max_text_chars": 1,
    "timeout_ms": 100,
}
WORKER_GRACE_MS = 750
SWEEP_MS = 2000
WIN32_TEXT_TIMEOUT_MS = 150

# act/find walk limits (element.rs has its own, deeper/narrower)
ACT_MAX_NODES_DEFAULT = 4000
ACT_MAX_NODES_CAP = 40000
ACT_MAX_DEPTH_DEFAULT = 40
ACT_MAX_DEPTH_CAP = 120
ACT_TIMEOUT_DEFAULT_MS = 3000
ACT_TIMEOUT_CAP_MS = 60000
FIND_LIMIT_DEFAULT = 20
FIND_LIMIT_CAP = 200

READ_ARGS = (
    "hwnd",
    "backend",
    "shape",
    "text_contains",
    "visible_only",
    "max_nodes",
    "max_depth",
    "max_children_per_node",
    "max_text_chars",
    "timeout_ms",
)
SELECT_ARGS = (
    "automation_id",
    "role",
    "name",
    "name_contains",
    "value_contains",
    "nth",
)
ACT_ARGS = ("hwnd", "select", "do", "text", "timeout_ms", "max_nodes", "max_depth")
FIND_ARGS = ("hwnd", "select", "limit", "timeout_ms", "max_nodes", "max_depth")


def clamp_into_notes(
    name: str,
    requested: Any,
    current: int,
    floor: int,
    ceiling: int,
    notes: list[str],
) -> int:
    """Pull an out-of-range request to the border and say it out loud."""
    if requested is None:
        return current
    try:
        requested = int(requested)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be an integer, not {requested!r}") from None
    effective = max(floor, min(ceiling, requested))
    if effective != requested:
        notes.append(
            f"{name}={requested} is outside [{floor}, {ceiling}]; this read used {effective}"
        )
    return effective


@dataclass
class ReadLimits:
    max_nodes: int = READ_DEFAULTS["max_nodes"]
    max_depth: int = READ_DEFAULTS["max_depth"]
    max_children_per_node: int = READ_DEFAULTS["max_children_per_node"]
    max_text_chars: int = READ_DEFAULTS["max_text_chars"]
    timeout_ms: int = READ_DEFAULTS["timeout_ms"]

    @classmethod
    def resolve(cls, args: dict[str, Any], notes: list[str]) -> "ReadLimits":
        limits = cls()
        for name in READ_DEFAULTS:
            setattr(
                limits,
                name,
                clamp_into_notes(
                    name,
                    args.get(name),
                    getattr(limits, name),
                    READ_FLOORS[name],
                    READ_CEILINGS[name],
                    notes,
                ),
            )
        return limits

    def json(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name in READ_DEFAULTS:
            out[name] = getattr(self, name)
            out[f"{name}_ceiling"] = READ_CEILINGS[name]
        out["worker_grace_ms"] = WORKER_GRACE_MS
        out["win32_text_timeout_ms"] = WIN32_TEXT_TIMEOUT_MS
        return out


@dataclass
class Selector:
    automation_id: Optional[str] = None
    role: Optional[str] = None
    name: Optional[str] = None
    name_contains: Optional[str] = None
    value_contains: Optional[str] = None
    nth: Optional[int] = None

    def is_empty(self) -> bool:
        return not any(
            (
                self.automation_id,
                self.role,
                self.name,
                self.name_contains,
                self.value_contains,
            )
        )

    def matches(self, node: "NodeView") -> bool:
        if self.automation_id is not None and node.automation_id != self.automation_id:
            return False
        if self.role is not None and not same_fold(node.role, self.role):
            return False
        if self.name is not None:
            if node.name is None or not same_fold(node.name, self.name):
                return False
        if self.name_contains is not None:
            want = self.name_contains.lower()
            if node.name is None or want not in node.name.lower():
                return False
        if self.value_contains is not None:
            want = self.value_contains.lower()
            if node.value is None or want not in node.value.lower():
                return False
        return True

    def json(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key in ("automation_id", "role", "name", "name_contains", "value_contains"):
            value = getattr(self, key)
            if value is not None:
                out[key] = value
        if self.nth is not None:
            out["nth"] = self.nth
        return out


def same_fold(left: Optional[str], right: str) -> bool:
    """Case-folded equality that actually works for Cyrillic."""
    if left is None:
        return False
    return left == right or left.lower() == right.lower()


@dataclass
class NodeView:
    role: str = ""
    name: Optional[str] = None
    value: Optional[str] = None
    automation_id: Optional[str] = None


@dataclass
class Choice:
    which: str  # "one" | "none" | "ambiguous"
    index: Optional[int] = None
    total: int = 0

    def one(self) -> bool:
        return self.which == "one"


def choose(matched: list[int], nth: Optional[int]) -> Choice:
    if not matched:
        return Choice("none")
    if nth is not None:
        if 0 <= nth < len(matched):
            return Choice("one", index=matched[nth], total=len(matched))
        return Choice("none")
    if len(matched) == 1:
        return Choice("one", index=matched[0], total=1)
    return Choice("ambiguous", total=len(matched))


def reject_unknown(args: dict[str, Any], allowed: tuple[str, ...], tool: str) -> None:
    unknown = [key for key in args if key not in allowed]
    if unknown:
        raise ValueError(
            f"{tool} arguments: unknown field(s) {', '.join(sorted(unknown))}; "
            f"allowed: {', '.join(allowed)}"
        )


def parse_hwnd(value: Any) -> int:
    """Accept what window_list prints: 500, "500", "0x1F4"."""
    parsed: Optional[int] = None
    if isinstance(value, bool):
        parsed = None
    elif isinstance(value, int):
        parsed = value if value > 0 else None
    elif isinstance(value, str):
        text = value.strip()
        if text[:2].lower() == "0x":
            try:
                parsed = int(text[2:], 16)
            except ValueError:
                parsed = None
        else:
            try:
                parsed = int(text, 10)
            except ValueError:
                parsed = None
        if parsed is not None and parsed <= 0:
            parsed = None
    if parsed is None or parsed == 0 or parsed > (2**64 - 1):
        raise ValueError("invalid hwnd")
    return parsed


def _trim(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    return value if value.strip() else None


def reject_blank_selector(select: dict[str, Any]) -> None:
    for name in ("automation_id", "role", "name", "name_contains", "value_contains"):
        value = select.get(name)
        if isinstance(value, str) and not value.strip():
            raise ValueError(
                f"selector field {name} is present but blank; omitting it would widen the selection"
            )
    if "nth" in select and select["nth"] is not None:
        nth = select["nth"]
        if not isinstance(nth, int) or isinstance(nth, bool) or nth < 0:
            raise ValueError(f"select.nth must be a non-negative integer, not {nth!r}")


def selector_from(select: dict[str, Any]) -> Selector:
    select = select or {}
    unknown = [key for key in select if key not in SELECT_ARGS]
    if unknown:
        raise ValueError(
            f"select: unknown field(s) {', '.join(sorted(unknown))}; allowed: {', '.join(SELECT_ARGS)}"
        )
    reject_blank_selector(select)
    return Selector(
        automation_id=_trim(select.get("automation_id")),
        role=_trim(select.get("role")),
        name=_trim(select.get("name")),
        name_contains=_trim(select.get("name_contains")),
        value_contains=_trim(select.get("value_contains")),
        nth=select.get("nth"),
    )


def act_from(said: Any) -> str:
    if not isinstance(said, str) or said.strip() not in ACTS:
        raise ValueError(
            "do must be one of invoke, set_value, toggle, expand, collapse, select, "
            f"scroll_into_view, focus - not {said!r}"
        )
    return said.strip()


def act_mutating(act: str) -> bool:
    return act not in NON_MUTATING_ACTS


@dataclass
class ReadPlan:
    hwnd: Optional[int]
    force_win32: bool
    shape: str  # "tree" | "flat"
    needle: str
    visible_only: bool
    limits: ReadLimits
    notes: list[str] = field(default_factory=list)


def read_plan(args: dict[str, Any]) -> ReadPlan:
    args = args or {}
    reject_unknown(args, READ_ARGS, TOOL_WINDOW_READ)
    notes: list[str] = []
    limits = ReadLimits.resolve(args, notes)
    shape = (args.get("shape") or "tree").strip() if isinstance(args.get("shape"), str) else "tree"
    if shape not in ("tree", "flat"):
        raise ValueError(f"shape must be tree or flat, not {shape!r}")
    backend = (args.get("backend") or "auto").strip() if isinstance(args.get("backend"), str) else "auto"
    if backend not in ("auto", "win32"):
        raise ValueError(f"backend must be auto or win32, not {backend!r}")
    needle = args.get("text_contains")
    if needle is not None and not isinstance(needle, str):
        raise ValueError("text_contains must be a string")
    visible_only = args.get("visible_only", False)
    if not isinstance(visible_only, bool):
        raise ValueError("visible_only must be a boolean")
    return ReadPlan(
        hwnd=parse_hwnd(args["hwnd"]) if "hwnd" in args and args["hwnd"] is not None else None,
        force_win32=(backend == "win32"),
        shape=shape,
        needle=(needle or "").strip().lower(),
        visible_only=visible_only,
        limits=limits,
        notes=notes,
    )


@dataclass
class ActPlan:
    hwnd: Optional[int]
    select: Selector
    act: str
    text: Optional[str]
    timeout_ms: int
    max_nodes: int
    max_depth: int


def act_plan(args: dict[str, Any]) -> ActPlan:
    args = args or {}
    reject_unknown(args, ACT_ARGS, TOOL_ELEMENT_ACT)
    act = act_from(args.get("do"))
    select = selector_from(args.get("select") or {})
    if select.is_empty():
        raise ValueError(
            "select is empty: name the element by automation_id, role, name, name_contains "
            'or value_contains. Acting on "any element of the window" is a miss by design'
        )
    if act == "set_value" and "text" not in args:
        raise ValueError(
            "set_value needs text: pass an empty string to clear the field, not no key at all"
        )
    if act != "set_value" and args.get("text") is not None:
        raise ValueError(f"text belongs to set_value only: {act} ignores it")
    text = args.get("text")
    if text is not None and not isinstance(text, str):
        raise ValueError("text must be a string")
    return ActPlan(
        hwnd=parse_hwnd(args["hwnd"]) if "hwnd" in args and args["hwnd"] is not None else None,
        select=select,
        act=act,
        text=text,
        timeout_ms=_clamp_opt(args.get("timeout_ms"), ACT_TIMEOUT_DEFAULT_MS, ACT_TIMEOUT_CAP_MS, floor=0),
        max_nodes=_clamp_opt(args.get("max_nodes"), ACT_MAX_NODES_DEFAULT, ACT_MAX_NODES_CAP),
        max_depth=_clamp_opt(args.get("max_depth"), ACT_MAX_DEPTH_DEFAULT, ACT_MAX_DEPTH_CAP),
    )


@dataclass
class FindPlan:
    hwnd: Optional[int]
    select: Selector
    limit: int
    timeout_ms: int
    max_nodes: int
    max_depth: int


def find_plan(args: dict[str, Any]) -> FindPlan:
    args = args or {}
    reject_unknown(args, FIND_ARGS, TOOL_ELEMENT_FIND)
    select = selector_from(args.get("select") or {})
    if select.nth is not None:
        raise ValueError(
            "nth belongs to element_act: find returns every match, and picking one from "
            "them is what nth is for when you act"
        )
    if select.is_empty():
        raise ValueError(
            "select is empty: name what to look for by automation_id, role, name, "
            "name_contains or value_contains. To see the whole window there is window_read"
        )
    return FindPlan(
        hwnd=parse_hwnd(args["hwnd"]) if "hwnd" in args and args["hwnd"] is not None else None,
        select=select,
        limit=_clamp_opt(args.get("limit"), FIND_LIMIT_DEFAULT, FIND_LIMIT_CAP),
        timeout_ms=_clamp_opt(args.get("timeout_ms"), 0, ACT_TIMEOUT_CAP_MS, floor=0),
        max_nodes=_clamp_opt(args.get("max_nodes"), ACT_MAX_NODES_DEFAULT, ACT_MAX_NODES_CAP),
        max_depth=_clamp_opt(args.get("max_depth"), ACT_MAX_DEPTH_DEFAULT, ACT_MAX_DEPTH_CAP),
    )


def _clamp_opt(value: Any, default: int, cap: int, *, floor: int = 1) -> int:
    """floor=0 is legitimate for timeout_ms ('0 - do not wait at all', the
    source's contract); every other limit floors at 1 ('zero nodes is not a
    walk, it is silence')."""
    if value is None:
        return default
    try:
        value = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"limit argument must be an integer, not {value!r}") from None
    return max(floor, min(cap, value))


def not_found(limit: Optional[str], nth_given: bool, found_any: bool) -> tuple[str, str]:
    """Why nothing matched - and these are THREE different answers, not one."""
    if limit == "max_nodes":
        return (
            "max_nodes",
            "the walk stopped at max_nodes before the window ended: raise max_nodes, "
            "or point hwnd at a smaller window",
        )
    if limit == "timeout":
        return (
            "timeout",
            "the walk ran out of timeout_ms before the window ended: raise timeout_ms - "
            "big Electron/Chromium windows carry thousands of elements",
        )
    if limit == "max_depth":
        return ("max_depth", "the walk stopped at max_depth; raise max_depth")
    if limit:
        return ("incomplete", "the walk stopped early for an unnamed reason")
    if nth_given and found_any:
        return (
            "nth_out_of_range",
            "the selector matched, but fewer elements than the nth you asked for",
        )
    return (
        "not_found",
        "read the window (window_read) to see what is actually there; "
        "the selector may name something this window does not have",
    )


def ambiguous_error(total: int, candidates: list[Any]) -> dict[str, Any]:
    return {
        "ok": False,
        "reason": "ambiguous",
        "matched": total,
        "candidates": candidates,
        "hint": "narrow the selector (automation_id is exact) or say select.nth to pick one",
    }


def receipt(plan: ActPlan, chosen: Any, after: Any, waited_ms: int, polls: int) -> dict[str, Any]:
    return {
        "ok": True,
        "did": plan.act,
        "mutating": act_mutating(plan.act),
        "select": plan.select.json(),
        "element": chosen,
        "element_after": after,
        "waited_ms": waited_ms,
        "polls": polls,
        "note": "the pattern was invoked and the element re-read afterwards; "
        "whether that achieved the goal is yours to judge",
    }


def find_receipt(
    plan: FindPlan,
    matched: int,
    elements: list[Any],
    scanned: int,
    whole: bool,
    waited_ms: int,
    polls: int,
) -> dict[str, Any]:
    shown = len(elements)
    return {
        "ok": True,
        "select": plan.select.json(),
        "matched": matched,
        "shown": shown,
        "truncated": matched > shown,
        "elements": elements,
        "nodes_scanned": scanned,
        "searched_whole_window": whole,
        "waited_ms": waited_ms,
        "polls": polls,
        "note": "ids are not part of this answer on purpose: name the element the same "
        "way when you act on it (element_act)",
    }


def clip_utf16(text: str, limit: int) -> tuple[str, bool]:
    """Clip by UTF-16 units (as Windows sees them), not by characters."""
    result: list[str] = []
    units = 0
    for value in text:
        width = 2 if ord(value) > 0xFFFF else 1
        if units + width > limit:
            return "".join(result), True
        result.append(value)
        units += width
    return "".join(result), False


def role_name(control_type: int) -> str:
    known = _CONTROL_TYPE_NAMES.get(control_type, "")
    if known:
        return known
    # An unknown code is shown as its number, not passed off as "custom": better an
    # incomprehensible truth than a comprehensible lie.
    return f"control_type_{control_type}"


_CONTROL_TYPE_NAMES = {
    0: "unknown",
    50000: "button",
    50001: "calendar",
    50002: "check_box",
    50003: "combo_box",
    50004: "edit",
    50005: "hyperlink",
    50006: "image",
    50007: "list_item",
    50008: "list",
    50009: "menu",
    50010: "menu_bar",
    50011: "menu_item",
    50012: "progress_bar",
    50013: "radio_button",
    50014: "scroll_bar",
    50015: "slider",
    50016: "spinner",
    50017: "status_bar",
    50018: "tab",
    50019: "tab_item",
    50020: "text",
    50021: "tool_bar",
    50022: "tool_tip",
    50023: "tree",
    50024: "tree_item",
    50025: "custom",
    50026: "group",
    50027: "thumb",
    50028: "data_grid",
    50029: "data_item",
    50030: "document",
    50031: "split_button",
    50032: "window",
    50033: "pane",
    50034: "header",
    50035: "header_item",
    50036: "table",
    50037: "title_bar",
    50038: "separator",
    50039: "semantic_zoom",
    50040: "app_bar",
}


def win32_role(cls: str) -> str:
    cls = (cls or "").lower()
    if "richedit" in cls or cls == "edit":
        return "edit"
    if "button" in cls:
        return "button"
    if "static" in cls:
        return "text"
    if "combobox" in cls:
        return "combo_box"
    if "listbox" in cls or "syslistview" in cls:
        return "list"
    if "systreeview" in cls:
        return "tree"
    if "systabcontrol" in cls:
        return "tab"
    if "progress" in cls:
        return "progress_bar"
    if "scrollbar" in cls:
        return "scroll_bar"
    if "toolbarwindow" in cls:
        return "tool_bar"
    if "msctls_statusbar" in cls:
        return "status_bar"
    if cls == "#32770":
        return "dialog"
    return "window"
