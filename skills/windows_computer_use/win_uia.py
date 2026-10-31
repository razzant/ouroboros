"""windows_computer_use.win_uia - the UI Automation reader/actor via comtypes.

Ported from the Windows half of Praxis/Hélène ``uia.rs``:

- one cache request per walk (properties, pattern state properties included,
  read in a batch) instead of five inter-process calls per element - the
  live-probe lesson of 10.09;
- the control-view tree walker, breadth-first with a deadline on every node;
- act/find share ONE sweep so selector fixes land in both;
- ambiguity is a refusal with candidates; "not found" is never confused with
  "didn't finish reading", and a field the selector names that the provider
  failed to read is neither: the candidate stays a candidate, so act refuses;
- a hung provider degrades honestly: the worker thread is abandoned (Python
  cannot kill threads) and, for reads, the plain Win32 fallback answers with a
  named ``fallback_reason``.

comtypes facts this module relies on: a successful NULL interface out-param is
a falsy pointer object, never ``None``; ``GetCurrentPattern`` hands back
``IUnknown``, so the pattern interface needs ``QueryInterface``;
``GetCachedPropertyValue`` returns plain Python values (``BoundingRectangle``
as a left/top/width/height tuple) and a non-matching type for an unsupported
property. A raised call is a failed query, named as one.

The apartment: per worker thread, MTA requested (comtypes' import may have
set STA; RPC_E_CHANGED_MODE is honored - the apartment stays as the importer
set it, same honesty as the Rust original's fallback branch). The process
boundary (isolated dependency, host-owned kill) is the hang guard.
"""

from __future__ import annotations

import dataclasses
import threading
import time
from typing import Any, Optional

from .core import (
    SWEEP_MS,
    WORKER_GRACE_MS,
    choose,
    clip_utf16,
    not_found,
    role_name,
)
from .tree import Node, Rect, Walk, WindowInfo
from .tree import render as render_tree

# Property and pattern ids (UIAutomationClient.h).
UIA_NamePropertyId = 30005
UIA_ControlTypePropertyId = 30003
UIA_LocalizedControlTypePropertyId = 30004
UIA_BoundingRectanglePropertyId = 30001
UIA_IsEnabledPropertyId = 30010
UIA_IsOffscreenPropertyId = 30022
UIA_HasKeyboardFocusPropertyId = 30008
UIA_AutomationIdPropertyId = 30011
UIA_ClassNamePropertyId = 30012
UIA_NativeWindowHandlePropertyId = 30020
UIA_ProcessIdPropertyId = 30002
UIA_IsExpandCollapsePatternAvailablePropertyId = 30028
UIA_IsSelectionItemPatternAvailablePropertyId = 30036
UIA_IsTogglePatternAvailablePropertyId = 30041
UIA_IsValuePatternAvailablePropertyId = 30043
UIA_ValueValuePropertyId = 30045
UIA_ExpandCollapseExpandCollapseStatePropertyId = 30070
UIA_SelectionItemIsSelectedPropertyId = 30079
UIA_ToggleToggleStatePropertyId = 30086
UIA_ValuePatternId = 10002
UIA_TogglePatternId = 10015
UIA_SelectionItemPatternId = 10010
UIA_ExpandCollapsePatternId = 10005
UIA_InvokePatternId = 10000
UIA_ScrollItemPatternId = 10017
UIA_TextPatternId = 10014
UIA_EditControlTypeId = 50004
UIA_DocumentControlTypeId = 50030

CACHED_PROPERTIES = (
    UIA_NamePropertyId,
    UIA_ControlTypePropertyId,
    UIA_LocalizedControlTypePropertyId,
    UIA_BoundingRectanglePropertyId,
    UIA_IsEnabledPropertyId,
    UIA_IsOffscreenPropertyId,
    UIA_HasKeyboardFocusPropertyId,
    UIA_AutomationIdPropertyId,
    UIA_ClassNamePropertyId,
    UIA_NativeWindowHandlePropertyId,
    UIA_ProcessIdPropertyId,
    # Pattern state rides the same batch as plain properties: no pattern object
    # is needed to read a value, a toggle state, a selection or an expansion.
    UIA_IsValuePatternAvailablePropertyId,
    UIA_ValueValuePropertyId,
    UIA_IsTogglePatternAvailablePropertyId,
    UIA_ToggleToggleStatePropertyId,
    UIA_IsSelectionItemPatternAvailablePropertyId,
    UIA_SelectionItemIsSelectedPropertyId,
    UIA_IsExpandCollapsePatternAvailablePropertyId,
    UIA_ExpandCollapseExpandCollapseStatePropertyId,
)

TOGGLE_STATES = {0: "off", 1: "on", 2: "mixed"}
EXPAND_STATES = {0: "collapsed", 1: "expanded", 2: "partially_expanded", 3: "leaf"}
# How much of a document's text a receipt or candidate shows; a longer one is
# marked text_truncated. A selector's value_contains always sees the whole text.
DESCRIBE_TEXT_CHARS = 4000


class UIAError(RuntimeError):
    pass


class ActionCallError(UIAError):
    """The action method was entered; an exception cannot establish its effects."""


def client_module():
    """Import the UIAutomationClient wrapper, generating it on first use.

    Only the registry and the type library are read, never a target window, so
    callers run this before the hang deadline: comtypes writes the generated
    module in place, and an abandoned worker must not leave it half written.
    Importing comtypes here also keeps its own STA initialisation on the
    calling thread, so the worker below gets the MTA it asks for.
    """
    import comtypes
    import comtypes.client
    from comtypes import GUID

    # By registered typelib GUID, not by dll path: LoadTypeLibEx on the dll
    # fails on this install, the registry route always works.
    tlib = comtypes.typeinfo.LoadRegTypeLib(
        GUID("{944DE083-8FB8-45CF-BCB7-C477ACB2F897}"), 1, 0, 0
    )
    return comtypes.client.GetModule(tlib)


def _automation():
    import comtypes
    import comtypes.client

    client_module()
    from comtypes.gen.UIAutomationClient import CUIAutomation, IUIAutomation

    return comtypes.client.CreateObject(
        CUIAutomation, interface=IUIAutomation, clsctx=comtypes.CLSCTX_INPROC_SERVER
    )


_local = threading.local()


def thread_automation():
    """Per-thread automation object: COM interface pointers are apartment-bound,
    so the worker thread that runs every UIA call creates its own instance (and
    its own apartment) - the Rust original's dedicated thread, in Python form."""
    existing = getattr(_local, "automation", None)
    if existing is not None:
        return existing
    import comtypes

    # MTA, not STA: an STA thread must pump messages or incoming calls never
    # arrive, and we run no message loop (the Rust original's reason, and
    # Microsoft's recommendation for UIA clients). RPC_E_CHANGED_MODE (comtypes
    # was first imported on this thread and put it in STA) is honored, not fought.
    try:
        comtypes.CoInitializeEx(comtypes.COINIT_MULTITHREADED)
    except OSError as error:
        if getattr(error, "winerror", None) != -2147417850:  # RPC_E_CHANGED_MODE
            raise
    made = _automation()
    _local.automation = made
    return made


def _build_cache(automation):
    cache = automation.CreateCacheRequest()
    view = automation.ControlViewCondition  # property, not a method
    from comtypes.gen.UIAutomationClient import AutomationElementMode_Full, TreeScope_Element

    cache.TreeScope = TreeScope_Element
    cache.TreeFilter = view
    cache.AutomationElementMode = AutomationElementMode_Full
    for prop in CACHED_PROPERTIES:
        cache.AddProperty(prop)
    return cache


# ─── reads: a failed read is a fact of its own, never "absent" ─────────────────

_UNREAD = object()


def _cached(element, prop_id: int) -> Any:
    try:
        return element.GetCachedPropertyValue(prop_id)
    except Exception:
        return _UNREAD


def _text(value: Any) -> Optional[str]:
    return value if isinstance(value, str) and value.strip() else None


def _flag(value: Any) -> Optional[bool]:
    return value if isinstance(value, bool) else None


def _number(value: Any) -> Optional[int]:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _rect_of(raw: Any) -> Optional[Rect]:
    """BoundingRectangle as a property read returns it (left, top, width, height)
    or as a RECT structure (left, top, right, bottom)."""
    try:
        if hasattr(raw, "left"):
            rect = Rect(int(raw.left), int(raw.top), int(raw.right), int(raw.bottom))
        else:
            left, top, width, height = (float(part) for part in raw)
            rect = Rect(round(left), round(top), round(left + width), round(top + height))
    except (TypeError, ValueError, AttributeError):
        return None
    return None if rect.is_empty() else rect


def _clip(text: Optional[str], limit: int) -> tuple[Optional[str], bool]:
    from .core import clip_utf16

    if not text:
        return text, False
    clipped, cut = clip_utf16(text, limit)
    return (clipped if clipped else None), cut


def _pattern_present(unknown: Any) -> bool:
    """A NULL pointer (falsy, not None) means the element has no such pattern."""
    return unknown is not None and bool(unknown)


def _live_value(element, limit: Optional[int] = None) -> tuple[Optional[str], bool]:
    """Live ValuePattern value, else the TextPattern document text (WinForms
    multiline, RichEdit documents), and whether ``limit`` cut the document.
    ``None`` when the element exposes neither or both are empty; a failed query
    RAISES - it is not absence.

    GetText truncates at its maxLength without saying so: a limit asks for one
    UTF-16 unit more, which tells a longer document from one that fits. Without
    a limit it is GetText(-1), the whole document - what matching needs."""
    from comtypes.gen.UIAutomationClient import IUIAutomationTextPattern, IUIAutomationValuePattern

    unknown = element.GetCurrentPattern(UIA_ValuePatternId)
    if _pattern_present(unknown):
        text = _text(unknown.QueryInterface(IUIAutomationValuePattern).CurrentValue)
        if text:
            return text, False
    unknown = element.GetCurrentPattern(UIA_TextPatternId)
    if _pattern_present(unknown):
        document = unknown.QueryInterface(IUIAutomationTextPattern).DocumentRange
        text, cut = document.GetText(-1 if limit is None else limit + 1), False
        if limit is not None and isinstance(text, str):
            text, cut = clip_utf16(text, limit)
        text = _text(text)
        if text or cut:
            return text, cut
    return None, False


def _cached_value(element) -> Any:
    """The cached ValuePattern value, ``None`` without the pattern, or ``_UNREAD``."""
    available = _cached(element, UIA_IsValuePatternAvailablePropertyId)
    if available is _UNREAD:
        return _UNREAD
    if available is not True:
        return None
    value = _cached(element, UIA_ValueValuePropertyId)
    return value if value is _UNREAD else _text(value)


def _cached_state(element, available_id: int, state_id: int, coerce) -> Any:
    available = _cached(element, available_id)
    if available is _UNREAD:
        return _UNREAD
    if available is not True:
        return None
    state = _cached(element, state_id)
    return state if state is _UNREAD else coerce(state)


def node_from(element, depth: int, parent: Optional[int], limits) -> Node:
    """Element -> Node from the CACHE (one batch read per subtree). Fields the
    provider failed to read are listed in ``unread`` instead of looking absent."""
    unread: list[str] = []

    def read(field: str, raw: Any, coerce=lambda value: value) -> Any:
        if raw is _UNREAD:
            unread.append(field)
            return None
        return coerce(raw)

    truncated = False
    control_type = read("role", _cached(element, UIA_ControlTypePropertyId), _number) or 0
    texts = {}
    for field, prop_id in (
        ("name", UIA_NamePropertyId),
        ("localized_role", UIA_LocalizedControlTypePropertyId),
        ("automation_id", UIA_AutomationIdPropertyId),
        ("class", UIA_ClassNamePropertyId),
    ):
        texts[field], cut = _clip(read(field, _cached(element, prop_id), _text), limits.max_text_chars)
        truncated |= cut

    value = read("value", _cached_value(element))
    if value is None and "value" not in unread and control_type in (UIA_EditControlTypeId, UIA_DocumentControlTypeId):
        # Proxy controls (WinForms/Win32 EDIT via the MSAA bridge) expose an EMPTY
        # cached value while a live query returns the text (live probe 08.09).
        # Live call only for edit/document, or every node would cost one.
        try:
            value, cut = _live_value(element, limits.max_text_chars)
            truncated |= cut
        except Exception:
            unread.append("value")
    value, cut = _clip(value, limits.max_text_chars)
    truncated |= cut

    checked = read("checked", _cached_state(
        element, UIA_IsTogglePatternAvailablePropertyId, UIA_ToggleToggleStatePropertyId,
        lambda state: TOGGLE_STATES.get(_number(state))))
    selected = read("selected", _cached_state(
        element, UIA_IsSelectionItemPatternAvailablePropertyId, UIA_SelectionItemIsSelectedPropertyId, _flag))
    expanded = read("expanded", _cached_state(
        element, UIA_IsExpandCollapsePatternAvailablePropertyId, UIA_ExpandCollapseExpandCollapseStatePropertyId,
        lambda state: EXPAND_STATES.get(_number(state))))
    hwnd = read("hwnd", _cached(element, UIA_NativeWindowHandlePropertyId), _number) or 0
    return Node(
        parent=parent,
        depth=depth,
        role=role_name(int(control_type)),
        localized_role=texts["localized_role"],
        name=texts["name"],
        value=value,
        automation_id=texts["automation_id"],
        cls=texts["class"],
        hwnd=hwnd or None,
        rect=read("rect", _cached(element, UIA_BoundingRectanglePropertyId), _rect_of),
        enabled=read("enabled", _cached(element, UIA_IsEnabledPropertyId), _flag),
        focused=read("focused", _cached(element, UIA_HasKeyboardFocusPropertyId), _flag),
        offscreen=read("offscreen", _cached(element, UIA_IsOffscreenPropertyId), _flag),
        checked=checked,
        selected=selected,
        expanded=expanded,
        text_truncated=truncated,
        children_unread=None,
        unread=tuple(unread),
    )


def _root_element(automation, hwnd_value: int, cache):
    try:
        root = automation.ElementFromHandleBuildCache(hwnd_value, cache)
    except Exception as error:
        raise UIAError(f"ElementFromHandle: {error}") from error
    if not root:
        raise UIAError("ElementFromHandle returned no element for this window")
    return root


def uia_walk(automation, cache, walker, hwnd_value: int, plan) -> Walk:
    """Breadth-first over the control view with a deadline on every node."""
    from collections import deque

    started = time.monotonic()
    deadline = started + plan.limits.timeout_ms / 1000.0
    walk = Walk()
    root = _root_element(automation, hwnd_value, cache)
    queue: deque = deque()
    queue.append((root, 0, None))

    while queue:
        if len(walk.nodes) >= plan.limits.max_nodes:
            walk.mark("max_nodes")
            break
        if time.monotonic() >= deadline:
            walk.mark("timeout")
            break
        element, depth, parent = queue.popleft()
        node = node_from(element, depth, parent, plan.limits)
        if plan.visible_only and node.offscreen is True:
            walk.skipped_offscreen += 1
            continue
        walk.depth_reached = max(walk.depth_reached, depth)
        index = len(walk.nodes)

        if depth + 1 > plan.limits.max_depth:
            node.children_unread = "max_depth"
            walk.subtrees_unread += 1
            walk.mark("max_depth")
        else:
            # A provider refusal is NOT "no children" - it is a limit like any other.
            child = _walk_first_child(walker, element, cache)
            if child is _PROVIDER_REFUSED:
                node.children_unread = "provider_refused"
                walk.subtrees_unread += 1
                walk.mark("provider_refused")
            if child is _END_OF_LIST or child is _PROVIDER_REFUSED:
                child = None
            seen = 0
            while child is not None:
                if seen >= plan.limits.max_children_per_node:
                    node.children_unread = "max_children_per_node"
                    walk.subtrees_unread += 1
                    walk.mark("max_children_per_node")
                    break
                if time.monotonic() >= deadline:
                    node.children_unread = "timeout"
                    walk.subtrees_unread += 1
                    walk.mark("timeout")
                    break
                nxt = _walk_next_sibling(walker, child, cache)
                if nxt is _PROVIDER_REFUSED:
                    node.children_unread = "provider_refused"
                    walk.subtrees_unread += 1
                    walk.mark("provider_refused")
                if nxt is _END_OF_LIST or nxt is _PROVIDER_REFUSED:
                    nxt = None
                queue.append((child, depth + 1, index))
                child = nxt
                seen += 1
        walk.nodes.append(node)

    walk.discovered_unread = len(queue)
    walk.walk_ms = int((time.monotonic() - started) * 1000)
    return walk


_PROVIDER_REFUSED = object()
_END_OF_LIST = object()


def _walk_first_child(walker, element, cache):
    """A successful NULL (a falsy pointer in comtypes) is the end of the list.
    Any raised error - E_POINTER included, which in comtypes is a real failure,
    not a collapsed NULL as in the Rust windows crate - is a provider refusal."""
    try:
        child = walker.GetFirstChildElementBuildCache(element, cache)
    except Exception:
        return _PROVIDER_REFUSED
    return child if child else _END_OF_LIST


def _walk_next_sibling(walker, element, cache):
    try:
        sibling = walker.GetNextSiblingElementBuildCache(element, cache)
    except Exception:
        return _PROVIDER_REFUSED
    return sibling if sibling else _END_OF_LIST


# ─── act/find: ONE sweep for both ───────────────────────────────────────────────

class Match:
    __slots__ = ("element", "node", "describe")

    def __init__(self, element, node: Node, describe_json):
        self.element = element
        self.node = node
        self.describe = describe_json


_SELECTOR_FIELDS = {
    "role": ("role",),
    "name": ("name", "name_contains"),
    "automation_id": ("automation_id",),
    "value": ("value_contains",),
}


def _selector_view(element, select):
    """The fields ``select`` constrains, plus those the provider failed to read."""
    from .core import NodeView

    failed: list[str] = []
    raw_type = _cached(element, UIA_ControlTypePropertyId)
    control_type = None if raw_type is _UNREAD else (_number(raw_type) or 0)
    if control_type is None and select.role is not None:
        failed.append("role")
    texts: dict[str, Optional[str]] = {"name": None, "automation_id": None}
    for field, prop_id, wanted in (
        ("name", UIA_NamePropertyId, select.name is not None or select.name_contains is not None),
        ("automation_id", UIA_AutomationIdPropertyId, select.automation_id is not None),
    ):
        if wanted:
            raw = _cached(element, prop_id)
            if raw is _UNREAD:
                failed.append(field)
            else:
                texts[field] = _text(raw)
    value = None
    if select.value_contains is not None:
        try:
            value = _value_for_select(element, control_type)[0]
        except Exception:
            failed.append("value")
    view = NodeView(role=role_name(control_type or 0), name=texts["name"], value=value,
                    automation_id=texts["automation_id"])
    return view, failed


def sweep(automation, cache, walker, hwnd_value: int, select, max_nodes: int, max_depth: int, deadline: float):
    """Walk the control view collecting matches; returns (found, scanned, hit_limit, unread).

    hit_limit is None when the whole window was read - "not found" and "didn't
    finish" must stay different answers. ``unread`` holds elements that match
    every field the provider could read but failed a field the selector names:
    they may be competitors, so they are never silently dropped.
    """
    from collections import deque

    root = _root_element(automation, hwnd_value, cache)
    found: list[Match] = []
    unread: list[dict[str, Any]] = []
    queue: deque = deque()
    queue.append((root, 0))
    scanned = 0
    hit_limit: Optional[str] = None
    while queue:
        if scanned >= max_nodes:
            hit_limit = "max_nodes"
            break
        if time.monotonic() >= deadline:
            hit_limit = "timeout"
            break
        element, depth = queue.popleft()
        scanned += 1
        view, failed = _selector_view(element, select)
        role = view.role
        if failed:
            readable = dataclasses.replace(
                select, **{key: None for field in failed for key in _SELECTOR_FIELDS[field]})
            if readable.matches(view):
                description = _describe(element, role, cached=True)
                description["unread"] = list(dict.fromkeys([*description.get("unread", []), *failed]))
                unread.append(description)
        elif select.matches(view):
            found.append(Match(element, None, None))
            found[-1].describe = lambda el=element, r=role: _describe(el, r, cached=True)
        child = _walk_first_child(walker, element, cache)
        if child is _END_OF_LIST:
            child = None
        elif child is _PROVIDER_REFUSED:
            # A walker refusal is the verb's refusal (Rust act_walk?): the window
            # is NOT declared fully searched on top of an unread subtree.
            raise UIAError("UIA action traversal incomplete: provider refused the first child")
        if depth >= max_depth and child is not None:
            hit_limit = "max_depth"
            break
        while child is not None:
            if time.monotonic() >= deadline:
                hit_limit = "timeout"
                break
            if scanned + len(queue) >= max_nodes:
                hit_limit = "max_nodes"
                break
            nxt = _walk_next_sibling(walker, child, cache)
            if nxt is _END_OF_LIST:
                nxt = None
            elif nxt is _PROVIDER_REFUSED:
                raise UIAError("UIA action traversal incomplete: provider refused a sibling")
            queue.append((child, depth + 1))
            child = nxt
        if hit_limit:
            break
    return found, scanned, hit_limit, unread


def _value_for_select(element, control_type: Optional[int], limit: Optional[int] = None) -> tuple[Optional[str], bool]:
    """Value as SELECTION sees it: cached first, live for empty-cache proxy edits
    and documents (ValuePattern, then TextPattern - where RichEdit documents keep
    their text); with whether ``limit`` cut a document. Matching passes no limit,
    so it sees the whole text: a prefix would turn a later match into a false
    "not found", or a competitor into a false unique choice. Raises when a read
    fails, including an unreadable control type that decides whether the live
    read applies."""
    cached = _cached_value(element)
    if cached is _UNREAD:
        raise UIAError("the cached value could not be read")
    if cached:
        return cached, False
    if control_type is None:
        raise UIAError("the control type could not be read")
    if control_type not in (UIA_EditControlTypeId, UIA_DocumentControlTypeId):
        return None, False
    return _live_value(element, limit)


def _describe(element, role: str, cached: bool = True) -> dict[str, Any]:
    """Element in words - the same field set window_read uses, so receipts and
    readings read the same way. Cached=True is 'as the walk saw it'; False is the
    live re-read after an action (showing the stale cache as 'after' would show
    'before' twice). Unsupported fields are absent; failed fields are unread."""

    out: dict[str, Any] = {"role": role}
    unread: list[str] = []

    def prop(field: str, prop_id: int):
        try:
            return element.GetCachedPropertyValue(prop_id) if cached else element.GetCurrentPropertyValue(prop_id)
        except Exception:
            unread.append(field)
            return _UNREAD

    raw_type = prop("role", UIA_ControlTypePropertyId)
    out["role"] = role_name(_number(raw_type) or 0)

    for key, prop_id in (
        ("name", UIA_NamePropertyId),
        ("automation_id", UIA_AutomationIdPropertyId),
        ("class", UIA_ClassNamePropertyId),
    ):
        text = _text(prop(key, prop_id))
        if text:
            out[key] = text
    try:
        if cached:
            value, cut = _value_for_select(
                element, None if raw_type is _UNREAD else (_number(raw_type) or 0), DESCRIBE_TEXT_CHARS)
        else:
            value, cut = _live_value(element, DESCRIBE_TEXT_CHARS)
        out["value"] = value
        if cut:
            out["text_truncated"] = True
    except Exception:
        unread.append("value")
    state: dict[str, Any] = {}
    for key, prop_id in (
        ("enabled", UIA_IsEnabledPropertyId),
        ("focused", UIA_HasKeyboardFocusPropertyId),
        ("offscreen", UIA_IsOffscreenPropertyId),
    ):
        value = _flag(prop(key, prop_id))
        if value is not None:
            state[key] = value
    if state:
        out["state"] = state
    if unread:
        out["unread"] = unread
    return out


def patterns_of(element) -> list[str]:
    """Which patterns the element supports - the refusal needs this to be useful.
    A failed query is named as unknown, not counted as support or its absence."""
    said: list[str] = []
    unknown: list[str] = []
    for pattern_id, name in (
        (UIA_InvokePatternId, "invoke"),
        (UIA_ValuePatternId, "set_value"),
        (UIA_TogglePatternId, "toggle"),
        (UIA_ExpandCollapsePatternId, "expand/collapse"),
        (UIA_SelectionItemPatternId, "select"),
        (UIA_ScrollItemPatternId, "scroll_into_view"),
    ):
        try:
            present = _pattern_present(element.GetCurrentPattern(pattern_id))
        except Exception:
            unknown.append(name)
            continue
        if present:
            said.append(name)
    if not said:
        said.append("none (focus still works)")
    if unknown:
        said.append(f"could not query {', '.join(unknown)}")
    return said


def act_pattern(element, pattern_id: int, want: str):
    """Pattern for the ACTION, outcomes separated: a successful NULL means the
    element really has no such pattern; a failed query is a provider error and
    is named as one (the 11.09 review hole - never report a failed instrument
    as a fact about the world)."""
    try:
        unknown = element.GetCurrentPattern(pattern_id)
    except Exception as error:
        raise UIAError(
            f"{want}: the pattern query itself failed - this is a provider error, "
            f"not proof that the element lacks {want}: {error}"
        ) from error
    if not _pattern_present(unknown):
        return None
    return unknown


def _cast_pattern(unknown, wanted_iid, want: str):
    if unknown is None:
        return None
    try:
        return unknown.QueryInterface(wanted_iid)
    except Exception as error:
        raise UIAError(f"{want}: pattern cast failed: {error}") from error


def perform(element, act: str, text: Optional[str], guard) -> None:
    """Invoke the pattern. No coordinate fallback - that unreliability is what
    this tool exists to avoid."""
    from comtypes.gen.UIAutomationClient import (
        IUIAutomationExpandCollapsePattern,
        IUIAutomationInvokePattern,
        IUIAutomationScrollItemPattern,
        IUIAutomationSelectionItemPattern,
        IUIAutomationTogglePattern,
        IUIAutomationValuePattern,
    )

    def missing(want: str) -> UIAError:
        return UIAError(
            f"this element does not support {want}; patterns it does support: "
            f"{', '.join(patterns_of(element))}. The verb refuses to fall back to a "
            "click at coordinates - that is the unreliability it exists to avoid"
        )

    def call(method, *args):
        guard()  # a refusal here is still before the action method
        try:
            method(*args)
        except Exception as error:
            raise ActionCallError(f"{act}: action call raised: {error}") from error

    if act == "invoke":
        pattern = _cast_pattern(
            act_pattern(element, UIA_InvokePatternId, "InvokePattern"),
            IUIAutomationInvokePattern, "InvokePattern",
        )
        if pattern is None:
            raise missing("InvokePattern")
        call(pattern.Invoke)
    elif act == "set_value":
        pattern = _cast_pattern(
            act_pattern(element, UIA_ValuePatternId, "ValuePattern"),
            IUIAutomationValuePattern, "ValuePattern",
        )
        if pattern is None:
            raise missing("ValuePattern")
        call(pattern.SetValue, text)  # act_plan admits only a string; "" is the explicit clear
    elif act == "toggle":
        pattern = _cast_pattern(
            act_pattern(element, UIA_TogglePatternId, "TogglePattern"),
            IUIAutomationTogglePattern, "TogglePattern",
        )
        if pattern is None:
            raise missing("TogglePattern")
        call(pattern.Toggle)
    elif act in ("expand", "collapse"):
        pattern = _cast_pattern(
            act_pattern(element, UIA_ExpandCollapsePatternId, "ExpandCollapsePattern"),
            IUIAutomationExpandCollapsePattern, "ExpandCollapsePattern",
        )
        if pattern is None:
            raise missing("ExpandCollapsePattern")
        if act == "expand":
            call(pattern.Expand)
        else:
            call(pattern.Collapse)
    elif act == "select":
        pattern = _cast_pattern(
            act_pattern(element, UIA_SelectionItemPatternId, "SelectionItemPattern"),
            IUIAutomationSelectionItemPattern, "SelectionItemPattern",
        )
        if pattern is None:
            raise missing("SelectionItemPattern")
        call(pattern.Select)
    elif act == "scroll_into_view":
        pattern = _cast_pattern(
            act_pattern(element, UIA_ScrollItemPatternId, "ScrollItemPattern"),
            IUIAutomationScrollItemPattern, "ScrollItemPattern",
        )
        if pattern is None:
            raise missing("ScrollItemPattern")
        call(pattern.ScrollIntoView)
    elif act == "focus":
        call(element.SetFocus)
    else:  # pragma: no cover - act_plan validated the verb
        raise UIAError(f"unknown act {act!r}")


# Pause between element_act polls. Its deadline is strict, so a re-poll that
# cannot finish before it (at the last walk's pace, after this pause, or after an
# overslept pause) is not started: the whole walk just done is the final look,
# while a walk that did begin and was cut still answers with its own timeout.
POLL_PAUSE_S = 0.12


def act_in_session(automation, plan, window: WindowInfo, resolve_pid):
    """Find-and-act inside one apartment visit: the element never leaves COM
    between selection and pattern invocation."""
    from .core import ambiguous_error, receipt, unreadable_error

    started = time.monotonic()
    deadline = started + max(plan.timeout_ms, WORKER_GRACE_MS) / 1000.0
    cache = _build_cache(automation)
    walker = automation.ControlViewWalker
    target_pid = resolve_pid(window.hwnd)
    # The window an action call goes to, as window_list prints it; the guard below
    # re-checks this pid just before the call. An unread title is named, not blank.
    delivered_to = {"hwnd": f"0x{window.hwnd:X}", "title": window.title, "pid": target_pid}
    if window.unread:
        delivered_to["unread"] = list(window.unread)

    def keep_waiting(walk_s: float) -> bool:
        """True after a pause that leaves room for one more whole walk."""
        if plan.timeout_ms == 0 or time.monotonic() + POLL_PAUSE_S + walk_s >= deadline:
            return False
        time.sleep(POLL_PAUSE_S)
        return time.monotonic() + walk_s < deadline

    def disclose_wait_stop(answer, walk_s: float):
        # Against the caller's timeout_ms, not the grace-floored deadline; an ordinary
        # end leaves up to one pause unused and is not an early stop.
        remaining_s = started + plan.timeout_ms / 1000.0 - time.monotonic()
        if plan.timeout_ms > 0 and remaining_s > POLL_PAUSE_S:
            answer["stopped_before_timeout_ms"] = int(remaining_s * 1000)
            answer["last_walk_ms"] = int(walk_s * 1000)
            answer["hint"] += (
                ". Wait stopped early because another walk at the last pace would not have "
                "finished before the deadline; call element_act again to look again"
            )
        return answer

    polls = 0
    while True:
        polls += 1
        walk_started = time.monotonic()
        found, scanned, hit_limit, unread = sweep(
            automation, cache, walker, window.hwnd, plan.select, plan.max_nodes, plan.max_depth, deadline
        )
        walk_s = time.monotonic() - walk_started
        if hit_limit is not None:
            reason, hint = not_found(hit_limit, False, False)
            return {
                "ok": False, "reason": reason, "hint": hint,
                "matched": len(found), "unreadable": len(unread), "nodes_scanned": scanned,
                "searched_whole_window": False,
                "waited_ms": int((time.monotonic() - started) * 1000), "polls": polls,
            }
        if unread:
            # A candidate the provider could not read may be the real target or a
            # competitor: choosing among the readable rest would be a guess.
            if keep_waiting(walk_s):
                continue
            waited_ms = int((time.monotonic() - started) * 1000)
            return disclose_wait_stop(unreadable_error(len(found), unread, scanned, waited_ms, polls), walk_s)
        indexes = list(range(len(found)))
        choice = choose(indexes, plan.select.nth)
        if choice.which == "one":
            element = found[choice.index].element
            _guard_live_match(element, plan.select)
            def guard():
                if time.monotonic() >= deadline:
                    raise UIAError("element action deadline expired before mutation")
                if resolve_pid(window.hwnd) != target_pid:
                    raise UIAError("target window changed before element action")
                if plan.hwnd is None:
                    from . import win32fb as _w32

                    if _w32._hwnd_int(_w32._user32.GetForegroundWindow()) != window.hwnd:
                        raise UIAError("foreground changed before element action")
            role = role_name(_number(_cached(element, UIA_ControlTypePropertyId)) or 0)
            before = _describe(element, role, cached=True)
            try:
                perform(element, plan.act, plan.text, guard)
            except ActionCallError as error:
                from .core import act_mutating

                return {
                    "ok": False, "error": str(error), "requested": plan.act,
                    "mutating": act_mutating(plan.act), "action_call_made": True,
                    "action_outcome": "unknown", "select": plan.select.json(), "element": before,
                    "delivered_to": delivered_to,
                    "hint": "the action call was made but its outcome is unknown. Do not retry blindly; "
                    "check the window state first",
                }
            after = _describe(element, role, cached=False)
            return receipt(plan, before, after, int((time.monotonic() - started) * 1000), polls, delivered_to)
        if choice.which == "ambiguous":
            candidates = [m.describe() for m in found]
            return ambiguous_error(choice.total, candidates)
        # none: wait and retry until deadline
        if keep_waiting(walk_s):
            continue
        reason, hint = not_found(hit_limit, plan.select.nth is not None, bool(found))
        return disclose_wait_stop({
            "ok": False, "reason": reason, "hint": hint,
            "matched": len(found), "nodes_scanned": scanned,
            "searched_whole_window": hit_limit is None,
            "waited_ms": int((time.monotonic() - started) * 1000), "polls": polls,
        }, walk_s)


def _guard_live_match(element, select) -> None:
    """Fresh look before acting: what the walk remembered may have redrawn.
    A provider failure on a field the selector ASKED for aborts the action (a
    failed instrument is never a fact about the world - the 11.09 lesson);
    the value is re-read with the sweep's own semantics (ValuePattern, then
    TextPattern), so a document matched by its text is not refused here."""
    from .core import NodeView

    def live(read):
        try:
            return read()
        except Exception as error:
            raise UIAError(f"re-reading the chosen element before the action: {error}") from error

    role = role_name(_number(live(lambda: element.CurrentControlType)) or 0) if select.role is not None else ""
    name = None
    if select.name is not None or select.name_contains is not None:
        name = _text(live(lambda: element.CurrentName))
    automation_id = _text(live(lambda: element.CurrentAutomationId)) if select.automation_id is not None else None
    value = live(lambda: _live_value(element)[0]) if select.value_contains is not None else None
    if not select.matches(NodeView(role=role, name=name, value=value, automation_id=automation_id)):
        raise UIAError("selected element changed before action")


def find_in_session(automation, plan, hwnd_value: int):
    from .core import find_receipt, unreadable_error

    started = time.monotonic()
    deadline = started + plan.timeout_ms / 1000.0
    cache = _build_cache(automation)
    walker = automation.ControlViewWalker
    polls = 0
    while True:
        polls += 1
        # One sweep gets its own ceiling even with timeout_ms=0: "look now" must
        # still read something, not nothing.
        sweep_by = time.monotonic() + SWEEP_MS / 1000.0
        by = max(deadline, sweep_by)
        found, scanned, hit_limit, unread = sweep(
            automation, cache, walker, hwnd_value, plan.select, plan.max_nodes, plan.max_depth, by
        )
        # With no matches, find polls until timeout_ms. A last look gets at least
        # SWEEP_MS, but reads the whole window only if one walk fits that budget.
        # A longer timeout_ms cannot ensure that final walk finishes.
        if found or time.monotonic() >= deadline:
            matched = len(found)
            waited_ms = int((time.monotonic() - started) * 1000)
            shown = [m.describe() for m in found[: plan.limit]]
            if matched == 0 and unread and hit_limit is None:
                return unreadable_error(0, unread, scanned, waited_ms, polls)
            if matched == 0:
                reason, hint = not_found(hit_limit, False, False)
                return {
                    "ok": False, "reason": reason, "hint": hint,
                    "matched": 0, "unreadable": len(unread), "nodes_scanned": scanned,
                    "searched_whole_window": hit_limit is None,
                    "waited_ms": waited_ms, "polls": polls,
                }
            return find_receipt(
                plan, matched, shown, scanned, hit_limit is None and not unread, waited_ms, polls,
                unreadable=unread,
            )
        time.sleep(POLL_PAUSE_S)


def read_in_session(automation, plan, window: WindowInfo):
    cache = _build_cache(automation)
    walker = automation.ControlViewWalker
    started = time.monotonic()
    walk = uia_walk(automation, cache, walker, window.hwnd, plan)
    from .tree import WINDOW_READ_UIA_DETAIL

    return render_tree(
        plan, window, walk,
        backend="uia", backend_detail=WINDOW_READ_UIA_DETAIL,
        fallback_reason=None,
        elapsed_ms=int((time.monotonic() - started) * 1000),
        extra_notes=[],
        workers_live=0,
    )
