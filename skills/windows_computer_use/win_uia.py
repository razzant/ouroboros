"""windows_computer_use.win_uia - the UI Automation reader/actor via comtypes.

Ported from the Windows half of Praxis/Hélène ``uia.rs``:

- one cache request per walk (properties + patterns read in a batch) instead of
  five inter-process calls per element - the live-probe lesson of 10.09;
- the control-view tree walker, breadth-first with a deadline on every node;
- act/find share ONE sweep so selector fixes land in both;
- ambiguity is a refusal with candidates; "not found" is never confused with
  "didn't finish reading";
- a hung provider degrades honestly: the worker thread is abandoned (Python
  cannot kill threads) and, for reads, the plain Win32 fallback answers with a
  named ``fallback_reason``.

The apartment: per worker thread, MTA requested (comtypes' import may have
set STA; RPC_E_CHANGED_MODE is honored - the apartment stays as the importer
set it, same honesty as the Rust original's fallback branch). The process
boundary (isolated dependency, host-owned kill) is the hang guard.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Optional

from .core import (
    SWEEP_MS,
    WORKER_GRACE_MS,
    choose,
    not_found,
    role_name,
)
from .tree import Node, Rect, Walk, WindowInfo
from .tree import render as render_tree

UIA_NamePropertyId = 30005
UIA_ControlTypePropertyId = 30003
UIA_LocalizedControlTypePropertyId = 30004
UIA_BoundingRectanglePropertyId = 30001
UIA_IsEnabledPropertyId = 30010
UIA_IsOffscreenPropertyId = 30013
UIA_HasKeyboardFocusPropertyId = 30008
UIA_AutomationIdPropertyId = 30011
UIA_ClassNamePropertyId = 30012
UIA_NativeWindowHandlePropertyId = 30020
UIA_ProcessIdPropertyId = 30002
UIA_ValuePatternId = 10002
UIA_TogglePatternId = 10015
UIA_SelectionItemPatternId = 10010
UIA_ExpandCollapsePatternId = 10005
UIA_InvokePatternId = 10000
UIA_ScrollItemPatternId = 10017
UIA_TextPatternId = 10020
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
)
CACHED_PATTERNS = (
    UIA_ValuePatternId,
    UIA_TogglePatternId,
    UIA_SelectionItemPatternId,
    UIA_ExpandCollapsePatternId,
)

TOGGLE_STATES = {0: "off", 1: "on", 2: "mixed"}
EXPAND_STATES = {0: "collapsed", 1: "expanded", 2: "partially_expanded", 3: "leaf"}


class UIAError(RuntimeError):
    pass


def _automation():
    import comtypes
    import comtypes.client
    from comtypes import GUID

    # By registered typelib GUID, not by dll path: LoadTypeLibEx on the dll
    # fails on this install, the registry route always works.
    tlib = comtypes.typeinfo.LoadRegTypeLib(
        GUID("{944DE083-8FB8-45CF-BCB7-C477ACB2F897}"), 1, 0, 0
    )
    comtypes.client.GetModule(tlib)
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
    # already put this thread in STA on import) is honored, not fought.
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
    for pattern in CACHED_PATTERNS:
        cache.AddPattern(pattern)
    return cache


def _rect_of(tag_rect) -> Optional[Rect]:
    try:
        rect = Rect(
            int(tag_rect.left), int(tag_rect.top), int(tag_rect.right), int(tag_rect.bottom)
        )
    except (TypeError, ValueError, AttributeError):
        return None
    return None if rect.is_empty() else rect


def _get_cached(element, prop_id: int, default=None):
    try:
        return element.GetCachedPropertyValue(prop_id)
    except Exception:
        return default


def _cached_str(element, prop_id: int) -> Optional[str]:
    try:
        value = element.GetCachedPropertyValue(prop_id)
    except Exception:
        return None
    if value is None:
        return None
    text = str(value)
    return text if text.strip() else None


def _clip(text: Optional[str], limit: int) -> tuple[Optional[str], bool]:
    from .core import clip_utf16

    if not text:
        return text, False
    clipped, cut = clip_utf16(text, limit)
    return (clipped if clipped else None), cut


def node_from(element, depth: int, parent: Optional[int], limits) -> Node:
    """Element -> Node from the CACHE (one batch read per subtree)."""
    truncated = False
    control_type = _get_cached(element, UIA_ControlTypePropertyId, 0) or 0
    name, cut = _clip(_cached_str(element, UIA_NamePropertyId), limits.max_text_chars)
    truncated |= cut
    localized, cut = _clip(_cached_str(element, UIA_LocalizedControlTypePropertyId), limits.max_text_chars)
    truncated |= cut
    automation_id, cut = _clip(_cached_str(element, UIA_AutomationIdPropertyId), limits.max_text_chars)
    truncated |= cut
    cls, cut = _clip(_cached_str(element, UIA_ClassNamePropertyId), limits.max_text_chars)
    truncated |= cut

    value = None
    try:
        pattern = element.GetCachedPatternAs(UIA_ValuePatternId)
        if pattern is not None:
            raw = pattern.CachedValue
            value, cut = _clip(str(raw) if raw is not None and str(raw).strip() else None, limits.max_text_chars)
            truncated |= cut
    except Exception:
        pass
    if value is None and control_type in (UIA_EditControlTypeId, UIA_DocumentControlTypeId):
        # Proxy controls (WinForms/Win32 EDIT via the MSAA bridge) expose an EMPTY
        # cached ValuePattern while a live query returns the text (live probe 08.09).
        # Live call only for edit/document, or every node would cost one.
        value = _live_value(element, limits)

    checked = None
    try:
        pattern = element.GetCachedPatternAs(UIA_TogglePatternId)
        if pattern is not None:
            checked = TOGGLE_STATES.get(int(pattern.CachedToggleState))
    except Exception:
        pass
    selected = None
    try:
        pattern = element.GetCachedPatternAs(UIA_SelectionItemPatternId)
        if pattern is not None:
            selected = bool(pattern.CachedIsSelected)
    except Exception:
        pass
    expanded = None
    try:
        pattern = element.GetCachedPatternAs(UIA_ExpandCollapsePatternId)
        if pattern is not None:
            expanded = EXPAND_STATES.get(int(pattern.CachedExpandCollapseState))
    except Exception:
        pass

    def _bool_of(prop_id: int) -> Optional[bool]:
        try:
            value = element.GetCachedPropertyValue(prop_id)
            return bool(value) if value is not None else None
        except Exception:
            return None

    hwnd = _get_cached(element, UIA_NativeWindowHandlePropertyId, 0) or 0
    return Node(
        parent=parent,
        depth=depth,
        role=role_name(int(control_type)),
        localized_role=localized,
        name=name,
        value=value,
        automation_id=automation_id,
        cls=cls,
        hwnd=int(hwnd) or None,
        rect=_rect_of(_get_cached(element, UIA_BoundingRectanglePropertyId)),
        enabled=_bool_of(UIA_IsEnabledPropertyId),
        focused=_bool_of(UIA_HasKeyboardFocusPropertyId),
        offscreen=_bool_of(UIA_IsOffscreenPropertyId),
        checked=checked,
        selected=selected,
        expanded=expanded,
        text_truncated=truncated,
        children_unread=None,
    )


def _live_value(element, limits) -> Optional[str]:
    """Live ValuePattern query for proxy edit/document controls; falls back to
    TextPattern's document range (WinForms multiline)."""
    try:
        pattern = element.GetCurrentPattern(UIA_ValuePatternId)
        if pattern is not None:
            raw = pattern.CurrentValue
            text = str(raw) if raw is not None and str(raw).strip() else None
            if text:
                clipped, _ = _clip(text, limits.max_text_chars)
                return clipped
    except Exception:
        pass
    try:
        pattern = element.GetCurrentPattern(UIA_TextPatternId)
        if pattern is not None:
            document = pattern.DocumentRange
            raw = document.GetText(limits.max_text_chars)
            text = str(raw) if raw and str(raw).strip() else None
            if text:
                clipped, _ = _clip(text, limits.max_text_chars)
                return clipped
    except Exception:
        pass
    return None


def uia_walk(automation, cache, walker, hwnd_value: int, plan) -> Walk:
    """Breadth-first over the control view with a deadline on every node."""
    from collections import deque

    started = time.monotonic()
    deadline = started + plan.limits.timeout_ms / 1000.0
    walk = Walk()
    try:
        root = automation.ElementFromHandleBuildCache(hwnd_value, cache)
    except Exception as error:
        raise UIAError(f"ElementFromHandle: {error}") from error
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
_E_POINTER = -2147467261  # 0x80004003


def _com_error_code(error: Exception) -> Optional[int]:
    return getattr(error, "hresult", None) if isinstance(error, Exception) else None


def _walk_first_child(walker, element, cache):
    """Successful NULL means 'no children'; E_POINTER from these walkers is the
    generated wrapper collapsing a successful null (the windows-crate lesson:
    call the vtable to keep both pieces of evidence). Provider refusals are
    named refusals - matched by HRESULT only, never by message text (an
    unrelated error mentioning a pointer must stay a refusal)."""
    try:
        return walker.GetFirstChildElementBuildCache(element, cache)
    except Exception as error:
        if _com_error_code(error) == _E_POINTER:
            return _END_OF_LIST
        return _PROVIDER_REFUSED


def _walk_next_sibling(walker, element, cache):
    try:
        return walker.GetNextSiblingElementBuildCache(element, cache)
    except Exception as error:
        if _com_error_code(error) == _E_POINTER:
            return _END_OF_LIST
        return _PROVIDER_REFUSED


# ─── act/find: ONE sweep for both ───────────────────────────────────────────────

class Match:
    __slots__ = ("element", "node", "describe")

    def __init__(self, element, node: Node, describe_json):
        self.element = element
        self.node = node
        self.describe = describe_json


def sweep(automation, cache, walker, hwnd_value: int, select, max_nodes: int, max_depth: int, deadline: float):
    """Walk the control view collecting matches; returns (found, scanned, hit_limit).

    hit_limit is None when the whole window was read - "not found" and "didn't
    finish" must stay different answers.
    """
    from collections import deque

    from .core import NodeView

    try:
        root = automation.ElementFromHandleBuildCache(hwnd_value, cache)
    except Exception as error:
        raise UIAError(f"ElementFromHandle: {error}") from error
    found: list[Match] = []
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
        control_type = _get_cached(element, UIA_ControlTypePropertyId, 0) or 0
        role = role_name(int(control_type))
        # Read only what the selector asks of display fields; value live-reads
        # only for edit/document proxy controls (the 11.09 acceptance hole).
        name = None
        if select.name is not None or select.name_contains is not None:
            name = _cached_str(element, UIA_NamePropertyId)
        automation_id = None
        if select.automation_id is not None:
            automation_id = _cached_str(element, UIA_AutomationIdPropertyId)
        value = None
        if select.value_contains is not None:
            value = _value_for_select(element)
        view = NodeView(role=role, name=name, value=value, automation_id=automation_id)
        if select.matches(view):
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
    return found, scanned, hit_limit


def _value_for_select(element) -> Optional[str]:
    """Value for SELECTION: cached first, live for empty-cache proxy edits.
    RichEditD2DPT (modern Notepad) exposes document value through TextPattern,
    not ValuePattern - the live probe taught this on 12.10."""
    cached = None
    try:
        pattern = element.GetCachedPatternAs(UIA_ValuePatternId)
        if pattern is not None:
            raw = pattern.CachedValue
            cached = str(raw) if raw is not None and str(raw).strip() else None
    except Exception:
        cached = None
    if cached:
        return cached
    control_type = _get_cached(element, UIA_ControlTypePropertyId, 0) or 0
    if int(control_type) not in (UIA_EditControlTypeId, UIA_DocumentControlTypeId):
        return cached
    return _live_value_brief(element) or _live_text_value(element) or cached


def _live_text_value(element) -> Optional[str]:
    """TextPattern document range - where RichEdit documents keep their text."""
    try:
        pattern = element.GetCurrentPattern(UIA_TextPatternId)
        if pattern is None:
            return None
        concrete = pattern.QueryInterface(_text_pattern_iid())
        document = concrete.DocumentRange
        raw = document.GetText(4000)
        return str(raw) if raw and str(raw).strip() else None
    except Exception:
        return None


def _text_pattern_iid():
    from comtypes.gen.UIAutomationClient import IUIAutomationTextPattern

    return IUIAutomationTextPattern


def _live_value_brief(element) -> Optional[str]:
    try:
        unknown = element.GetCurrentPattern(UIA_ValuePatternId)
        if unknown is None or not getattr(unknown, "value", None):
            return None
        from comtypes.gen.UIAutomationClient import IUIAutomationValuePattern

        pattern = unknown.QueryInterface(IUIAutomationValuePattern)
        raw = pattern.CurrentValue
        return str(raw) if raw is not None and str(raw).strip() else None
    except Exception:
        return None


def _pattern_unknown_ok(unknown) -> bool:
    """comtypes returns a POINTER(IUnknown) even for a successful null (ptr=0):
    both the None case and the zero-pointer case mean 'no such pattern'."""
    if unknown is None:
        return False
    value = getattr(unknown, "value", None)
    if value is None:
        return True  # a comtypes pointer object without .value attr: assume real
    try:
        return int(value) != 0
    except (TypeError, ValueError):
        return True


def _describe(element, role: str, cached: bool = True) -> dict[str, Any]:
    """Element in words - the same field set window_read uses, so receipts and
    readings read the same way. Cached=True is 'as the walk saw it'; False is the
    live re-read after an action (showing the stale cache as 'after' would show
    'before' twice)."""

    out: dict[str, Any] = {"role": role}

    def prop(prop_id: int):
        try:
            return element.GetCachedPropertyValue(prop_id) if cached else element.GetCurrentPropertyValue(prop_id)
        except Exception:
            return None

    for key, prop_id in (
        ("name", UIA_NamePropertyId),
        ("automation_id", UIA_AutomationIdPropertyId),
        ("class", UIA_ClassNamePropertyId),
    ):
        value = prop(prop_id)
        text = str(value) if value is not None else None
        if text and text.strip():
            out[key] = text
    try:
        if cached:
            out["value"] = _value_for_select(element)
        else:
            live = _live_value_brief(element)
            if live:
                out["value"] = live
    except Exception:
        pass
    state: dict[str, Any] = {}
    for key, prop_id in (
        ("enabled", UIA_IsEnabledPropertyId),
        ("focused", UIA_HasKeyboardFocusPropertyId),
        ("offscreen", UIA_IsOffscreenPropertyId),
    ):
        value = prop(prop_id)
        if value is not None:
            state[key] = bool(value)
    if state:
        out["state"] = state
    return out


def patterns_of(element) -> list[str]:
    """Which patterns the element supports - the refusal needs this to be useful."""
    said: list[str] = []
    for pattern_id, name in (
        (UIA_InvokePatternId, "invoke"),
        (UIA_ValuePatternId, "set_value"),
        (UIA_TogglePatternId, "toggle"),
        (UIA_ExpandCollapsePatternId, "expand/collapse"),
        (UIA_SelectionItemPatternId, "select"),
        (UIA_ScrollItemPatternId, "scroll_into_view"),
    ):
        try:
            if _pattern_unknown_ok(element.GetCurrentPattern(pattern_id)):
                said.append(name)
        except Exception:
            continue
    if not said:
        said.append("none (focus still works)")
    return said


def act_pattern(element, pattern_id: int, want: str):
    """Pattern for the ACTION, outcomes separated: a successful null (None or a
    zero pointer - comtypes returns POINTER(IUnknown) with ptr=0) means the
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
    if not _pattern_unknown_ok(unknown):
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

    if act == "invoke":
        pattern = _cast_pattern(
            act_pattern(element, UIA_InvokePatternId, "InvokePattern"),
            IUIAutomationInvokePattern, "InvokePattern",
        )
        if pattern is None:
            raise missing("InvokePattern")
        guard()  # fresh check right before the mutation, not before the pattern query
        pattern.Invoke()
    elif act == "set_value":
        pattern = _cast_pattern(
            act_pattern(element, UIA_ValuePatternId, "ValuePattern"),
            IUIAutomationValuePattern, "ValuePattern",
        )
        if pattern is None:
            raise missing("ValuePattern")
        guard()
        pattern.SetValue(text or "")
    elif act == "toggle":
        pattern = _cast_pattern(
            act_pattern(element, UIA_TogglePatternId, "TogglePattern"),
            IUIAutomationTogglePattern, "TogglePattern",
        )
        if pattern is None:
            raise missing("TogglePattern")
        guard()
        pattern.Toggle()
    elif act in ("expand", "collapse"):
        pattern = _cast_pattern(
            act_pattern(element, UIA_ExpandCollapsePatternId, "ExpandCollapsePattern"),
            IUIAutomationExpandCollapsePattern, "ExpandCollapsePattern",
        )
        if pattern is None:
            raise missing("ExpandCollapsePattern")
        guard()
        if act == "expand":
            pattern.Expand()
        else:
            pattern.Collapse()
    elif act == "select":
        pattern = _cast_pattern(
            act_pattern(element, UIA_SelectionItemPatternId, "SelectionItemPattern"),
            IUIAutomationSelectionItemPattern, "SelectionItemPattern",
        )
        if pattern is None:
            raise missing("SelectionItemPattern")
        guard()
        pattern.Select()
    elif act == "scroll_into_view":
        pattern = _cast_pattern(
            act_pattern(element, UIA_ScrollItemPatternId, "ScrollItemPattern"),
            IUIAutomationScrollItemPattern, "ScrollItemPattern",
        )
        if pattern is None:
            raise missing("ScrollItemPattern")
        guard()
        pattern.ScrollIntoView()
    elif act == "focus":
        guard()
        element.SetFocus()
    else:  # pragma: no cover - act_plan validated the verb
        raise UIAError(f"unknown act {act!r}")


def act_in_session(automation, plan, window: WindowInfo, resolve_pid):
    """Find-and-act inside one apartment visit: the element never leaves COM
    between selection and pattern invocation."""
    started = time.monotonic()
    deadline = started + max(plan.timeout_ms, WORKER_GRACE_MS) / 1000.0
    cache = _build_cache(automation)
    walker = automation.ControlViewWalker
    target_pid = resolve_pid(window.hwnd)

    polls = 0
    while True:
        polls += 1
        found, scanned, hit_limit = sweep(
            automation, cache, walker, window.hwnd, plan.select, plan.max_nodes, plan.max_depth, deadline
        )
        if hit_limit is not None:
            reason, hint = not_found(hit_limit, False, False)
            return {
                "ok": False, "reason": reason, "hint": hint,
                "matched": len(found), "nodes_scanned": scanned,
                "searched_whole_window": False,
                "waited_ms": int((time.monotonic() - started) * 1000), "polls": polls,
            }
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
            role = role_name(int(_get_cached(element, UIA_ControlTypePropertyId, 0) or 0))
            before = _describe(element, role, cached=True)
            perform(element, plan.act, plan.text, guard)
            after_role = role_name(int(_live_control_type(element) or 0))
            after = _describe(element, after_role, cached=False)
            from .core import receipt

            return receipt(plan, before, after, int((time.monotonic() - started) * 1000), polls)
        if choice.which == "ambiguous":
            from .core import ambiguous_error

            candidates = [m.describe() for m in found]
            return ambiguous_error(choice.total, candidates)
        # none: wait and retry until deadline
        if plan.timeout_ms == 0 or time.monotonic() >= deadline:
            reason, hint = not_found(hit_limit, plan.select.nth is not None, bool(found))
            return {
                "ok": False, "reason": reason, "hint": hint,
                "matched": len(found), "nodes_scanned": scanned,
                "searched_whole_window": hit_limit is None,
                "waited_ms": int((time.monotonic() - started) * 1000), "polls": polls,
            }
        time.sleep(0.12)


def _guard_live_match(element, select) -> None:
    """Fresh look before acting: what the walk remembered may have redrawn.
    A provider failure on a field the selector ASKED for aborts the action (a
    failed instrument is never a fact about the world - the 11.09 lesson);
    display-only fields may degrade to absence."""
    from .core import NodeView

    role = ""
    if select.role is not None:
        try:
            control_type = int(element.CurrentControlType or 0)
        except Exception as error:
            raise UIAError(f"re-reading the chosen element before the action: {error}") from error
        role = role_name(control_type)
    name = None
    if select.name is not None or select.name_contains is not None:
        try:
            name = element.CurrentName or None
        except Exception as error:
            raise UIAError(f"re-reading the chosen element before the action: {error}") from error
    automation_id = None
    if select.automation_id is not None:
        try:
            automation_id = element.CurrentAutomationId or None
        except Exception as error:
            raise UIAError(f"re-reading the chosen element before the action: {error}") from error
    value = None
    if select.value_contains is not None:
        try:
            unknown = element.GetCurrentPattern(UIA_ValuePatternId)
            if _pattern_unknown_ok(unknown):
                from comtypes.gen.UIAutomationClient import IUIAutomationValuePattern

                pattern = unknown.QueryInterface(IUIAutomationValuePattern)
                raw = pattern.CurrentValue
                value = str(raw) if raw is not None and str(raw).strip() else None
        except Exception as error:
            raise UIAError(f"re-reading the chosen element before the action: {error}") from error
    if not select.matches(NodeView(role=role, name=name, value=value, automation_id=automation_id)):
        raise UIAError("selected element changed before action")


def _live_control_type(element) -> int:
    try:
        return int(element.CurrentControlType or 0)
    except Exception:
        try:
            return int(element.GetCurrentPropertyValue(UIA_ControlTypePropertyId) or 0)
        except Exception:
            return 0


def find_in_session(automation, plan, hwnd_value: int):
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
        found, scanned, hit_limit = sweep(
            automation, cache, walker, hwnd_value, plan.select, plan.max_nodes, plan.max_depth, by
        )
        if found or time.monotonic() >= deadline:
            matched = len(found)
            shown = [m.describe() for m in found[: plan.limit]]
            if matched == 0:
                reason, hint = not_found(hit_limit, False, False)
                return {
                    "ok": False, "reason": reason, "hint": hint,
                    "matched": 0, "nodes_scanned": scanned,
                    "searched_whole_window": hit_limit is None,
                    "waited_ms": int((time.monotonic() - started) * 1000), "polls": polls,
                }
            from .core import find_receipt

            return find_receipt(
                plan, matched, shown, scanned, hit_limit is None,
                int((time.monotonic() - started) * 1000), polls,
            )
        time.sleep(0.12)


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
