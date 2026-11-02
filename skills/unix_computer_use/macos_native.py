"""macOS native facts for unix_computer_use, run as a short-lived helper process.

``python -I macos_native.py <op> [args...]`` prints one JSON object. The plugin starts it
through its ordinary process boundary (``_run``) so that a stuck or crashing native call
can be killed and can never take the host server down, and so that every answer is fresh
(AppKit caches such as ``NSWorkspace.frontmostApplication`` refresh only on a main run loop,
which the long-lived server does not turn). Pure ``ctypes`` against ApplicationServices and
CoreGraphics: no pyobjc wrapper is required (the packaged interpreter carries AppKit and
Quartz but not the Accessibility API, and source installs carry none).

Operations (all read-only except ``activate`` and ``act``):

- ``apps [limit]``                running regular applications with their windows, ``frontmost`` and ``scriptable``
- ``resolve <query>``             which running application a name, bundle id or ``pid:<n>`` means (0, 1 or many)
- ``activate <query> [wait_ms]``  resolve, bring forward if needed, re-read who is in front (never launches)
- ``tree <pid> [max_nodes] [timeout_ms] [root] [include_web]``  bounded accessibility walk of one window
- ``perms``                       prompt-free permission preflights (Accessibility, Screen Recording, input events)
- ``displays``                    active displays: bounds in global points, pixel sizes, ``screencapture -D`` index
- ``act <pid> <press|set_value> <selector json> [max_nodes] [timeout_ms]``  ONE element named by identity (window,
  role, label, AXIdentifier, nth), one of its own actions, a readback; no pointer, no activation

Honesty rules: ``ok`` answers the question asked; ``unknown`` is spelled out, never rendered as an
empty list or ``false``; a window the walk did not finish says so in ``truncated_by`` (the only
bounds are ``max_nodes`` and ``timeout_ms``: a node's children are read up to the remaining node
budget, so a wide node is cut by ``max_nodes`` and the node says how many children were not read: one
``AXUIElementGetAttributeValueCount`` call per node pays for that count, and a wide node near the start can hold up to
``max_nodes`` child references at once, all released on every exit);
``-25204`` (the app did not answer in time) is busy, not denied. A switch is ``activated`` only when
the front was read as another application before and as the target afterwards. ``act`` finds the element and acts
inside one process (an element reference never outlives the helper, so a mark or node id of an earlier read is not
an identity), refuses whenever the choice is not proved (several matches, an unfinished walk, an unread element),
and says ``completed``, ``unknown`` (the application did not answer the action call in time and may still act) or
``failed``; who is in front is read before and after and reported, never changed.
"""

from __future__ import annotations

import ctypes
import json
import math
import plistlib
import re
import subprocess
import sys
import time
from collections import deque
from ctypes import POINTER, Structure, byref, c_char_p, c_double, c_float, c_int32, c_long, c_size_t, c_ubyte, c_uint32, c_ulong, c_void_p, create_string_buffer
from typing import Any, Callable

_AS_PATH = "/System/Library/Frameworks/ApplicationServices.framework/ApplicationServices"
_CF_PATH = "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation"
_CG_PATH = "/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics"
_UTF8 = 0x08000100
_LSAPPINFO = "/usr/bin/lsappinfo"
AX_ERR = {0: "ok", -25200: "Failure", -25201: "IllegalArgument", -25202: "InvalidUIElement", -25204: "CannotComplete",
          -25205: "AttributeUnsupported", -25206: "ActionUnsupported", -25208: "NotImplemented", -25211: "APIDisabled",
          -25212: "NoValue", -25213: "ParameterizedAttributeUnsupported", -25214: "NotEnoughPrecision"}
# Roles that get a numbered mark: elements a model usually wants to address. A mark is a position, not
# proof that the element reacts to a click (static text and images are listed because they are targets too).
MARKED = {"AXButton", "AXTextField", "AXTextArea", "AXLink", "AXCheckBox", "AXRadioButton", "AXPopUpButton",
          "AXComboBox", "AXMenuButton", "AXTab", "AXMenuItem", "AXRow", "AXCell", "AXStaticText", "AXImage"}
_WALK_ATTRS = ["AXRole", "AXSubrole", "AXTitle", "AXDescription", "AXValue", "AXPosition", "AXSize", "AXIdentifier"]
TREE_CEILINGS = {"max_nodes": 2000, "timeout_ms": 15000}


class CGPoint(Structure):
    _fields_ = [("x", c_double), ("y", c_double)]


class CGSize(Structure):
    _fields_ = [("w", c_double), ("h", c_double)]


class CGRect(Structure):
    _fields_ = [("origin", CGPoint), ("size", CGSize)]


# ----------------------------------------------------------------------------------------------
# lsappinfo: fresh application facts without AppKit (format is Apple's CLI output, parsed loosely)
# ----------------------------------------------------------------------------------------------

def _sh(cmd: list[str], timeout: float = 5.0) -> tuple[int, str]:
    """(exit code, stdout); a process that could not run answers (-1, '') so the caller renders an unknown."""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
        return int(proc.returncode), proc.stdout or ""
    except Exception:  # noqa: BLE001 - the caller renders an unknown
        return -1, ""


def parse_info(text: str) -> dict[str, Any]:
    """``lsappinfo info -only pid -only bundleid -only name <asn>`` -> {pid, app_id, app} (newlines or spaces between fields)."""
    out: dict[str, Any] = {}
    m = re.search(r'"pid"\s*=\s*(\d+)', text)
    if m:
        out["pid"] = int(m.group(1))
    m = re.search(r'"CFBundleIdentifier"\s*=\s*"([^"]*)"', text)
    if m:
        out["app_id"] = m.group(1)
    m = re.search(r'"LSDisplayName"\s*=\s*"([^"]*)"', text)
    if m:
        out["app"] = m.group(1)
    return out


def front_app() -> dict[str, Any] | None:
    _rc, asn_text = _sh([_LSAPPINFO, "front"])
    asn = asn_text.strip()
    if not asn:
        return None
    _rc, text = _sh([_LSAPPINFO, "info", "-only", "pid", "-only", "bundleid", "-only", "name", asn])
    info = parse_info(text)
    return info if info.get("pid") else None


def parse_list(text: str) -> list[dict[str, Any]]:
    """``lsappinfo list`` -> regular (``type="Foreground"``) applications: {app, pid, app_id, path}."""
    apps: list[dict[str, Any]] = []
    for block in re.split(r"\n(?= *\d+\) )", text):
        head = re.match(r'\s*\d+\) "(.*)" ASN:', block)
        if not head or 'type="Foreground"' not in block:
            continue
        pid = re.search(r"\bpid\s*=\s*(\d+)", block)
        if not pid:
            continue
        bundle = re.search(r'bundleID="([^"]*)"', block)
        path = re.search(r'bundle path="([^"]*)"', block)
        apps.append({"app": head.group(1), "pid": int(pid.group(1)),
                     "app_id": bundle.group(1) if bundle else "", "path": path.group(1) if path else ""})
    return apps


def running_apps() -> tuple[list[dict[str, Any]], str]:
    """(applications, error): an inventory that could not be read is an error, never an empty desktop."""
    rc, text = _sh([_LSAPPINFO, "list"], timeout=8.0)
    apps = parse_list(text)
    if rc != 0 or not text.strip():
        return apps, f"lsappinfo list gave no inventory (exit {rc})"
    return apps, ""


def scriptable(bundle_path: str) -> bool | None:
    """Whether the bundle declares an AppleScript dictionary (Info.plist); None when unreadable."""
    if not bundle_path:
        return None
    try:
        with open(bundle_path.rstrip("/") + "/Contents/Info.plist", "rb") as fh:
            info = plistlib.load(fh)
    except Exception:  # noqa: BLE001
        return None
    enabled = info.get("NSAppleScriptEnabled")
    return bool(info.get("OSAScriptingDefinition")) or enabled is True or str(enabled).upper() == "YES"


def resolve(query: str, apps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Launch-free target resolution: ``pid:<n>``, exact bundle id, or case-insensitive display/executable name.

    No prefix or substring guessing, and no bare numbers: ``pid:`` is the documented form."""
    q = str(query or "").strip()
    if not q:
        return []
    if q.lower().startswith("pid:"):
        digits = q[4:].strip()
        return [a for a in apps if digits.isdigit() and a["pid"] == int(digits)]
    exact_id = [a for a in apps if a["app_id"] and a["app_id"].lower() == q.lower()]
    if exact_id:
        return exact_id
    low = q.lower()
    return [a for a in apps if a["app"].lower() == low or a["path"].rstrip("/").rsplit("/", 1)[-1].lower() in (low, low + ".app")]


# ----------------------------------------------------------------------------------------------
# Accessibility API through ctypes (pattern measured on 27.10.2026: 37 nodes in 1.6 s cold / 50 ms warm)
# ----------------------------------------------------------------------------------------------

class _AX:
    """Lazily bound ApplicationServices/CoreFoundation symbols; one instance per helper process."""

    def __init__(self) -> None:
        self.AS = ctypes.CDLL(_AS_PATH)
        self.CF = ctypes.CDLL(_CF_PATH)
        sig = self._sig
        self.CFRelease = sig(self.CF, "CFRelease", None, [c_void_p])
        self.CFRetain = sig(self.CF, "CFRetain", c_void_p, [c_void_p])
        self.CFGetTypeID = sig(self.CF, "CFGetTypeID", c_ulong, [c_void_p])
        self.T_STR = sig(self.CF, "CFStringGetTypeID", c_ulong, [])()
        self.T_ARR = sig(self.CF, "CFArrayGetTypeID", c_ulong, [])()
        self.T_NUM = sig(self.CF, "CFNumberGetTypeID", c_ulong, [])()
        self.T_BOOL = sig(self.CF, "CFBooleanGetTypeID", c_ulong, [])()
        self.CFStringCreateWithCString = sig(self.CF, "CFStringCreateWithCString", c_void_p, [c_void_p, c_char_p, c_uint32])
        self.CFStringGetLength = sig(self.CF, "CFStringGetLength", c_long, [c_void_p])
        self.CFStringGetMaximumSizeForEncoding = sig(self.CF, "CFStringGetMaximumSizeForEncoding", c_long, [c_long, c_uint32])
        self.CFStringGetCString = sig(self.CF, "CFStringGetCString", c_ubyte, [c_void_p, c_void_p, c_long, c_uint32])
        self.CFArrayGetCount = sig(self.CF, "CFArrayGetCount", c_long, [c_void_p])
        self.CFArrayGetValueAtIndex = sig(self.CF, "CFArrayGetValueAtIndex", c_void_p, [c_void_p, c_long])
        self.CFArrayCreate = sig(self.CF, "CFArrayCreate", c_void_p, [c_void_p, c_void_p, c_long, c_void_p])
        self.CFBooleanGetValue = sig(self.CF, "CFBooleanGetValue", c_ubyte, [c_void_p])
        self.CFNumberGetValue = sig(self.CF, "CFNumberGetValue", c_ubyte, [c_void_p, c_long, c_void_p])
        self.ARRAY_CB = ctypes.addressof(ctypes.c_byte.in_dll(self.CF, "kCFTypeArrayCallBacks"))
        self.kCFBooleanTrue = c_void_p.in_dll(self.CF, "kCFBooleanTrue").value
        self.AXIsProcessTrusted = sig(self.AS, "AXIsProcessTrusted", c_ubyte, [])
        self.AXUIElementCreateApplication = sig(self.AS, "AXUIElementCreateApplication", c_void_p, [c_int32])
        self.AXUIElementCreateSystemWide = sig(self.AS, "AXUIElementCreateSystemWide", c_void_p, [])
        self.AXUIElementCopyAttributeValue = sig(self.AS, "AXUIElementCopyAttributeValue", c_int32, [c_void_p, c_void_p, POINTER(c_void_p)])
        self.AXUIElementSetAttributeValue = sig(self.AS, "AXUIElementSetAttributeValue", c_int32, [c_void_p, c_void_p, c_void_p])
        self.AXUIElementIsAttributeSettable = sig(self.AS, "AXUIElementIsAttributeSettable", c_int32, [c_void_p, c_void_p, POINTER(c_ubyte)])
        self.AXUIElementCopyActionNames = sig(self.AS, "AXUIElementCopyActionNames", c_int32, [c_void_p, POINTER(c_void_p)])
        self.AXUIElementPerformAction = sig(self.AS, "AXUIElementPerformAction", c_int32, [c_void_p, c_void_p])
        self.AXUIElementCopyMultipleAttributeValues = sig(self.AS, "AXUIElementCopyMultipleAttributeValues", c_int32,
                                                          [c_void_p, c_void_p, c_uint32, POINTER(c_void_p)])
        self.AXUIElementCopyAttributeValues = sig(self.AS, "AXUIElementCopyAttributeValues", c_int32,
                                                  [c_void_p, c_void_p, c_long, c_long, POINTER(c_void_p)])
        self.AXUIElementGetAttributeValueCount = sig(self.AS, "AXUIElementGetAttributeValueCount", c_int32,
                                                     [c_void_p, c_void_p, POINTER(c_long)])
        self.AXUIElementSetMessagingTimeout = sig(self.AS, "AXUIElementSetMessagingTimeout", c_int32, [c_void_p, c_float])
        self.AXValueGetType = sig(self.AS, "AXValueGetType", c_uint32, [c_void_p])
        self.AXValueGetValue = sig(self.AS, "AXValueGetValue", c_ubyte, [c_void_p, c_uint32, c_void_p])
        self.T_AXVALUE = sig(self.AS, "AXValueGetTypeID", c_ulong, [])()
        try:  # private but long-stable: AX window element -> CGWindowID
            self.AXUIElementGetWindow = sig(self.AS, "_AXUIElementGetWindow", c_int32, [c_void_p, POINTER(c_uint32)])
        except AttributeError:
            self.AXUIElementGetWindow = None
        self._strings: dict[str, int] = {}

    @staticmethod
    def _sig(lib, name, restype, argtypes):
        f = getattr(lib, name)
        f.restype, f.argtypes = restype, argtypes
        return f

    def cfs(self, name: str) -> int:
        ref = self._strings.get(name)
        if ref is None:
            ref = self.CFStringCreateWithCString(None, name.encode(), _UTF8)
            self._strings[name] = ref
        return ref

    def set_timeout(self, elem, seconds: float) -> None:
        """Messaging timeout for ``elem`` and, through the system-wide element, the default for every other
        element this process touches (a timeout set on one element does not reach the elements read from it)."""
        self.AXUIElementSetMessagingTimeout(elem, seconds)
        system_wide = self.AXUIElementCreateSystemWide()
        if system_wide:
            self.AXUIElementSetMessagingTimeout(system_wide, seconds)
            self.CFRelease(system_wide)

    def is_array(self, ref) -> bool:
        return bool(ref) and self.CFGetTypeID(ref) == self.T_ARR

    def to_py(self, ref) -> Any:
        if not ref:
            return None
        tid = self.CFGetTypeID(ref)
        if tid == self.T_STR:
            n = self.CFStringGetLength(ref)
            size = self.CFStringGetMaximumSizeForEncoding(n, _UTF8) + 1
            buf = create_string_buffer(size)
            return buf.value.decode("utf-8", "replace") if self.CFStringGetCString(ref, buf, size, _UTF8) else None
        if tid == self.T_BOOL:
            return bool(self.CFBooleanGetValue(ref))
        if tid == self.T_NUM:
            d = c_double()
            self.CFNumberGetValue(ref, 13, byref(d))  # kCFNumberDoubleType
            return d.value
        if tid == self.T_AXVALUE:
            kind = self.AXValueGetType(ref)
            if kind == 1:
                p = CGPoint(); self.AXValueGetValue(ref, kind, byref(p)); return (p.x, p.y)
            if kind == 2:
                s = CGSize(); self.AXValueGetValue(ref, kind, byref(s)); return (s.w, s.h)
            if kind == 5:  # kAXValueTypeAXError: a multi-attribute read answers a failed attribute with its error
                e = c_int32(); self.AXValueGetValue(ref, kind, byref(e)); return ("axerror", int(e.value))
            return None
        return ("ref", ref)

    def attr(self, elem, name: str) -> tuple[Any, int]:
        """One attribute as a python value; a CF array/element answers ('ref', ptr) RETAINED for the caller."""
        out = c_void_p()
        err = self.AXUIElementCopyAttributeValue(elem, self.cfs(name), byref(out))
        if err != 0 or not out.value:
            if out.value:
                self.CFRelease(out.value)
            return None, err
        try:
            val = self.to_py(out.value)
        except Exception:
            self.CFRelease(out.value)
            raise
        if isinstance(val, tuple) and val and val[0] == "ref":
            return val, 0
        self.CFRelease(out.value)
        if isinstance(val, tuple) and val and val[0] == "axerror":
            return None, val[1]
        return val, 0

    def multi(self, elem, names: list[str]) -> tuple[int, list[Any]]:
        arr = self.CFArrayCreate(None, (c_void_p * len(names))(*[self.cfs(n) for n in names]), len(names), self.ARRAY_CB)
        out = c_void_p()
        try:
            err = self.AXUIElementCopyMultipleAttributeValues(elem, arr, 0, byref(out))
        finally:
            self.CFRelease(arr)
        if err != 0 or not out.value:
            return err, []
        if not self.is_array(out.value):
            self.CFRelease(out.value)
            return -25200, []
        vals = []
        try:
            for i in range(len(names)):
                v = self.to_py(self.CFArrayGetValueAtIndex(out.value, i))
                vals.append(None if isinstance(v, tuple) and v and v[0] == "ref" else v)
        finally:
            self.CFRelease(out.value)
        return 0, vals

    def children(self, elem, limit: int) -> tuple[int, list[int], int | None]:
        """Up to ``limit`` children (ranged read; plain read when an app rejects the range form) as RETAINED
        refs, plus the child count when the app reports it (so a cut can be sized)."""
        total: int | None = None
        count = c_long(0)
        if self.AXUIElementGetAttributeValueCount(elem, self.cfs("AXChildren"), byref(count)) == 0:
            total = int(count.value)
        out = c_void_p()
        err = self.AXUIElementCopyAttributeValues(elem, self.cfs("AXChildren"), 0, limit, byref(out))
        if err == -25201:
            err = self.AXUIElementCopyAttributeValue(elem, self.cfs("AXChildren"), byref(out))
        if err != 0 or not out.value:
            return err, [], total
        if not self.is_array(out.value):
            self.CFRelease(out.value)
            return -25200, [], total
        try:
            available = self.CFArrayGetCount(out.value)
            if total is None:
                total = int(available)
            n = min(available, limit)
            return 0, [self.CFRetain(self.CFArrayGetValueAtIndex(out.value, i)) for i in range(n)], total
        finally:
            self.CFRelease(out.value)

    def windows(self, app_elem) -> tuple[list[dict[str, Any]], str | None]:
        """(window rows, unread reason): a list that could not be read is said so, never an empty list."""
        val, err = self.attr(app_elem, "AXWindows")
        if err != 0:
            return [], AX_ERR.get(err, str(err))
        if not val:
            return [], None
        arr = val[1]
        if not self.is_array(arr):
            self.CFRelease(arr)
            return [], "not_an_array"
        rows = []
        try:
            for i in range(self.CFArrayGetCount(arr)):
                win = self.CFArrayGetValueAtIndex(arr, i)
                e, vals = self.multi(win, ["AXTitle", "AXMinimized", "AXPosition", "AXSize", "AXMain"])
                row: dict[str, Any] = {}
                if e != 0:
                    row["unread"] = {"window": AX_ERR.get(e, str(e))}  # nothing of this window could be read
                else:
                    vals, failed = split_attr_errors(["AXTitle", "AXMinimized", "AXPosition", "AXSize", "AXMain"], vals)
                    title, minimized, pos, size, main = (vals + [None] * 5)[:5]
                    row["title"] = title
                    for key, value in (("minimized", minimized), ("main", main)):
                        if isinstance(value, bool):
                            row[key] = value
                    if failed:
                        row["unread"] = failed  # an attribute the app failed to answer, never rendered as false
                    bounds = _bounds(pos, size)
                    if bounds:
                        row["bounds"] = bounds
                wid = self.window_id(win)
                if wid is not None:
                    row["window_id"] = wid
                rows.append(row)
        finally:
            self.CFRelease(arr)
        return rows, None

    def window_id(self, win) -> int | None:
        """The CGWindowID of an AX window element (what ``apps`` prints as ``window_id``); None when the private
        symbol is absent or the element is not a window."""
        if self.AXUIElementGetWindow is None:
            return None
        wid = c_uint32()
        return int(wid.value) if self.AXUIElementGetWindow(win, byref(wid)) == 0 else None

    def window_refs(self, app_elem) -> tuple[list[Any], int]:
        """The application's ``AXWindows`` as RETAINED refs (the caller releases each); (refs, err)."""
        val, err = self.attr(app_elem, "AXWindows")
        if err != 0:
            return [], err
        if not val:
            return [], 0
        arr = val[1]
        if not self.is_array(arr):
            self.CFRelease(arr)
            return [], -25200
        try:
            return [self.CFRetain(self.CFArrayGetValueAtIndex(arr, i)) for i in range(self.CFArrayGetCount(arr))], 0
        finally:
            self.CFRelease(arr)

    def action_names(self, elem) -> tuple[list[str], int]:
        """The actions an element says it performs (``AXPress``, ``AXShowMenu``...); (names, err)."""
        out = c_void_p()
        err = self.AXUIElementCopyActionNames(elem, byref(out))
        if err != 0 or not out.value:
            if out.value:
                self.CFRelease(out.value)
            return [], err
        try:
            if not self.is_array(out.value):
                return [], -25200
            names = [self.to_py(self.CFArrayGetValueAtIndex(out.value, i)) for i in range(self.CFArrayGetCount(out.value))]
            return [n for n in names if isinstance(n, str)], 0
        finally:
            self.CFRelease(out.value)

    def settable(self, elem, name: str) -> tuple[bool | None, int]:
        """Whether the application lets ``name`` be set on the element; (None, err) when it did not answer."""
        flag = c_ubyte(0)
        err = self.AXUIElementIsAttributeSettable(elem, self.cfs(name), byref(flag))
        return (bool(flag.value) if err == 0 else None), err

    def perform(self, elem, action: str) -> int:
        """Ask the element to perform one of its own actions; the AXError (0 = the application accepted it)."""
        return int(self.AXUIElementPerformAction(elem, self.cfs(action)))

    def set_string(self, elem, name: str, value: str) -> int:
        """Set a string attribute (the ``AXValue`` of a field) from a fresh, released CFString; the AXError."""
        ref = self.CFStringCreateWithCString(None, value.encode("utf-8"), _UTF8)
        if not ref:
            return -25201
        try:
            return int(self.AXUIElementSetAttributeValue(elem, self.cfs(name), ref))
        finally:
            self.CFRelease(ref)


_ABSENT = (-25212, -25205)  # NoValue / AttributeUnsupported: ordinary absence, not a failed read


def split_attr_errors(names: list[str], vals: list[Any]) -> tuple[list[Any], dict[str, str]]:
    """Separate the values of a multi-attribute read from the attributes the app FAILED to answer (an AXError
    other than ordinary absence): (values with errors as None, {attribute: error name})."""
    clean: list[Any] = []
    failed: dict[str, str] = {}
    for name, v in zip(names, list(vals) + [None] * (len(names) - len(vals))):
        if isinstance(v, tuple) and v and v[0] == "axerror":
            if v[1] not in _ABSENT:
                failed[name] = AX_ERR.get(v[1], str(v[1]))
            clean.append(None)
        else:
            clean.append(v)
    return clean, failed


def _bounds(pos: Any, size: Any) -> list[int] | None:
    """[x, y, w, h] from AX position/size tuples; None when either is missing or not a finite number."""
    if not (isinstance(pos, tuple) and isinstance(size, tuple)):
        return None
    try:
        values = [float(pos[0]), float(pos[1]), float(size[0]), float(size[1])]
    except (TypeError, ValueError, IndexError):
        return None
    if not all(math.isfinite(v) for v in values):
        return None
    return [int(v) for v in values]


def walk(root: Any, read_node: Callable[[Any], tuple[int, list[Any]]], children_of: Callable[[Any, int], tuple],
         *, max_nodes: int, deadline_s: float, skip_children: tuple[str, ...] = ("AXWebArea",),
         release: Callable[[Any], None] | None = None, now: Callable[[], float] = time.perf_counter) -> dict[str, Any]:
    """Bounded breadth-first walk over an element graph (injectable, so the bound is testable without a desktop).

    ``read_node(el)`` -> (err, [role, subrole, title, description, value, position, size, identifier]); ``children_of(el, n)`` ->
    (err, [children]) or (err, [children], total). Stops at ``max_nodes`` visited nodes or at the deadline and
    SAYS which (``truncated_by``); a node or a children list the application failed to answer is a cut too
    (``read_errors``: that subtree is never requested, and ``errors`` counts the failures by attribute). A node's
    children are requested up to the remaining node budget plus one, so a
    node wider than the budget is cut by ``max_nodes`` and records ``children_unread``; children of
    ``skip_children`` roles (web content) are not fetched unless the caller clears the tuple. Owns ``root`` and
    every child it fetched: each is released exactly once, on every exit.
    """
    t0 = now()
    deadline = t0 + deadline_s
    queue: deque[tuple[Any, int]] = deque([(root, 0)])
    nodes: list[dict[str, Any]] = []
    visited = 0
    errors: dict[str, int] = {}
    truncated_by: list[str] = []
    skipped_web = 0

    def cut(reason: str) -> None:
        if reason not in truncated_by:
            truncated_by.append(reason)

    def count(key: str) -> None:
        errors[key] = errors.get(key, 0) + 1

    try:
        while queue:
            if visited >= max_nodes:
                cut("max_nodes")
                break
            if now() > deadline:
                cut("timeout")
                break
            el, depth = queue.popleft()
            try:
                err, vals = read_node(el)
                if err != 0:
                    count(AX_ERR.get(err, str(err)))
                    cut("read_errors")  # its subtree is never requested: the tree is incomplete, and says so
                    continue
                vals, failed = split_attr_errors(_WALK_ATTRS, list(vals))
                for name, reason in failed.items():
                    count(f"{name}:{reason}")
                if failed and all(v is None for v in vals):
                    cut("read_errors")
                    continue  # not one attribute of this element could be read: a failed read, not an empty node
                visited += 1
                role, subrole, title, desc, value, pos, size, ident = vals
                node: dict[str, Any] = {"id": visited, "depth": depth, "role": (role or "")[2:] if isinstance(role, str) and role.startswith("AX") else role}
                if subrole:
                    node["subrole"] = subrole
                label = title or desc or ""
                if label:
                    node["label"] = str(label)[:200]
                if isinstance(ident, str) and ident:
                    node["identifier"] = ident
                if isinstance(value, (str, float, bool)) and value != "":
                    text = str(value)
                    node["value"] = text[:200]
                    if len(text) > 200:
                        node["value_len"] = len(text)
                bounds = _bounds(pos, size)
                if bounds:
                    node["x"], node["y"], node["w"], node["h"] = bounds
                nodes.append(node)
                if role in skip_children:
                    skipped_web += 1
                    continue
                requested = max_nodes - visited + 1
                answer = children_of(el, requested)
                cerr, kids = answer[0], list(answer[1])
                total = answer[2] if len(answer) > 2 else None
                if cerr not in (0, -25212, -25205):
                    count("children:" + AX_ERR.get(cerr, str(cerr)))
                    cut("read_errors")
                if total is not None and total > len(kids):
                    node["children_unread"] = total - len(kids)
                    # The budget capped the request: a budget cut (whether or not the visited count reaches it). Fewer
                    # children than the count for any other reason (a failed read, no value, a count the read disagreed
                    # with) is a read error, never laundered into a budget cut.
                    cut("max_nodes" if cerr == 0 and len(kids) >= requested else "read_errors")
                for kid in kids:
                    queue.append((kid, depth + 1))
            finally:
                if release:
                    release(el)
    finally:
        if release:
            for el, _depth in queue:
                release(el)
    if skipped_web:
        cut("web_content_skipped")
    return {"nodes": nodes, "visited": visited, "truncated": bool(truncated_by), "truncated_by": truncated_by,
            "errors": errors, "elapsed_ms": int((now() - t0) * 1000)}


def marks_from(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Set-of-marks view: numbered elements of the marked roles with centre points in global logical points.

    A mark says where an element is, not that it reacts to a click."""
    marks = []
    for node in nodes:
        if "AX" + str(node.get("role")) in MARKED and "x" in node and node.get("w", 0) > 0 and node.get("h", 0) > 0:
            marks.append({"id": len(marks) + 1, "role": node["role"], "title": node.get("label", ""),
                          "center_x": node["x"] + node["w"] // 2, "center_y": node["y"] + node["h"] // 2,
                          "x": node["x"], "y": node["y"], "w": node["w"], "h": node["h"]})
    return marks


# ----------------------------------------------------------------------------------------------
# element actions: an element named by identity, one of its own AX actions, a readback (no pointer)
# ----------------------------------------------------------------------------------------------

ACTIONS = {"press": "AXPress", "set_value": "AXValue"}
ACT_DEFAULTS = {"max_nodes": 600, "timeout_ms": 3000}
_SELECT_KEYS = ("window", "role", "label", "identifier", "nth", "text", "include_web")
_ELEMENT_ATTRS = _WALK_ATTRS[:-1] + ["AXEnabled", "AXFocused", "AXIdentifier"]  # keep the revalidation read's order
_INCOMPLETE = {"max_nodes", "timeout", "read_errors"}  # cuts after which another match may exist unseen
_SETTLE_S = 0.15  # between the action and the readback: a field shows its new value, the call stays one call
_SHOWN = 10  # candidates listed in a refusal


def _role_name(role: Any) -> Any:
    return role[2:] if isinstance(role, str) and role.startswith("AX") else role


def parse_selector(raw: Any) -> tuple[dict[str, Any], str]:
    """The identity an action names, from the ``act`` op's JSON argument -> (selector, error).

    Source-grounded identity only: ``window`` (a ``window_id`` as ``apps`` prints it, or an exact title; absent =
    the focused window), ``role`` (``Button`` or ``AXButton``), ``label`` (the exact ``AXTitle`` or
    ``AXDescription``), ``identifier`` (the exact ``AXIdentifier``) and ``nth`` (1-based, among the matches in walk
    order); at least one of role, label, identifier. A mark or node id of an earlier read numbers that answer only
    and is not an identity. ``text`` is what ``set_value`` writes (``""`` clears); ``include_web`` walks web content."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw) if raw.strip() else {}
        except ValueError as exc:
            return {}, f"selector is not JSON: {exc}"
    if not isinstance(raw, dict):
        return {}, "selector must be a JSON object"
    unknown = sorted(str(k) for k in raw if k not in _SELECT_KEYS)
    if unknown:
        return {}, f"selector: unknown field(s) {', '.join(unknown)}; allowed: {', '.join(_SELECT_KEYS)}"
    sel: dict[str, Any] = {}
    for key in ("role", "label", "identifier"):
        value = raw.get(key)
        if value is None or value == "":
            continue
        if not isinstance(value, str) or not value.strip():
            return {}, f"selector.{key} must be a non-empty string, not {value!r}"
        sel[key] = value
    if not sel:
        return {}, ("selector is empty: name the element by role, label and/or identifier "
                    "(a mark or node id of an earlier read is not an identity)")
    nth = raw.get("nth")
    if nth is not None:
        if isinstance(nth, bool) or not isinstance(nth, int) or nth < 1:
            return {}, f"selector.nth is 1-based (1 = the first match in walk order), not {nth!r}"
        sel["nth"] = nth
    window = raw.get("window")
    if window is not None and window != "":
        if isinstance(window, bool) or not isinstance(window, (int, str)):
            return {}, f"selector.window must be a window_id or an exact title, not {window!r}"
        text = str(window).strip()
        sel["window"] = {"window_id": int(text)} if text.isdigit() else {"title": str(window)}
    if raw.get("text") is not None:
        if not isinstance(raw["text"], str):
            return {}, f"text must be a string (an empty string clears the field), not {raw['text']!r}"
        if "\x00" in raw["text"]:
            return {}, "text may not contain NUL"
        sel["text"] = raw["text"]
    sel["include_web"] = bool(raw.get("include_web"))
    return sel, ""


def matches(sel: dict[str, Any], role: Any, title: Any, desc: Any, identifier: Any) -> bool:
    """Whether the read attributes match: role folds case; label and identifier are full, exact strings."""
    want = sel.get("role")
    if want is not None and str(_role_name(want)).lower() != str(_role_name(role) or "").lower():
        return False
    label = sel.get("label")
    if label is not None and label not in (title, desc):
        return False
    ident = sel.get("identifier")
    return ident is None or (isinstance(identifier, str) and identifier == ident)


_NAMED = {"role": ("AXRole",), "label": ("AXTitle", "AXDescription"), "identifier": ("AXIdentifier",)}


def classify(sel: dict[str, Any], role: Any, title: Any, desc: Any, identifier: Any, failed: dict[str, str]) -> str:
    """One walked element against the selector: ``match``, ``miss``, or ``unread`` (the element did not answer an
    attribute the selector names, so whether it is the target is unknown). A definite mismatch excludes an element
    before later attributes matter; otherwise an unread candidate forbids picking among the readable rest."""
    for key, attrs in _NAMED.items():
        if key not in sel:
            continue
        if any(attr in failed for attr in attrs):
            return "unread"
        if not matches({key: sel[key]}, role, title, desc, identifier):
            return "miss"
    return "match"


def choose(found: int, nth: int | None, complete: bool) -> str:
    """Which match an action goes to: ``one`` (exactly the match asked for), ``none``, ``ambiguous`` (several, no
    ``nth``) or ``incomplete``. An unread subtree can hide earlier breadth-first matches, so even ``nth`` requires
    a complete walk: the readable nodes need not be a prefix of the whole."""
    if not complete:
        return "incomplete"
    if nth is not None:
        return "one" if found >= nth else "none"
    if found == 0:
        return "none"
    if found > 1:
        return "ambiguous"
    return "one"


def read_element(ax: Any, elem: Any) -> dict[str, Any]:
    """One fresh read of an element: role, subrole, label and value (both whole: a receipt is the record of what the
    action replaced and what the application kept), bounds, enabled, focused, identifier; ``unread`` names what the
    application failed to answer, and a read that failed whole is only ``unread``."""
    return _read(ax, elem)[0]


def _read(ax: Any, elem: Any) -> tuple[dict[str, Any], tuple[Any, Any, Any]]:
    """``read_element`` plus the raw (role, title, description) the read answered, for revalidating an identity."""
    err, vals = ax.multi(elem, _ELEMENT_ATTRS)
    if err != 0:
        return {"unread": {"element": AX_ERR.get(err, str(err))}}, (None, None, None)
    clean, failed = split_attr_errors(_ELEMENT_ATTRS, list(vals))
    role, subrole, title, desc, value, pos, size, enabled, focused, ident = clean
    out: dict[str, Any] = {"role": _role_name(role)}
    if subrole:
        out["subrole"] = subrole
    label = title or desc or ""
    if label:
        out["label"] = str(label)
    if isinstance(value, (str, float, bool)):
        out["value"] = value
    bounds = _bounds(pos, size)
    if bounds:
        out["x"], out["y"], out["w"], out["h"] = bounds
    for key, flag in (("enabled", enabled), ("focused", focused)):
        if isinstance(flag, bool):
            out[key] = flag
    if isinstance(ident, str) and ident:
        out["identifier"] = ident
    if failed:
        out["unread"] = failed
    return out, (role, title, desc)


def _identity(sel: dict[str, Any]) -> str:
    parts = [f"{key}={sel[key]!r}" for key in ("role", "label", "identifier") if key in sel]
    return " ".join(parts)


def _window_facts(ax: Any, win: Any, source: str) -> dict[str, Any]:
    facts: dict[str, Any] = {"from": source}
    title, err = ax.attr(win, "AXTitle")
    if isinstance(title, tuple) and title and title[0] == "ref":
        ax.CFRelease(title[1])
        title, err = None, err or -25200
    if err != 0 and err not in _ABSENT:
        facts["unread"] = {"AXTitle": AX_ERR.get(err, str(err))}
    elif isinstance(title, str):
        facts["title"] = title
    wid = ax.window_id(win)
    if wid is not None:
        facts["window_id"] = wid
    else:
        facts.setdefault("unread", {})["window_id"] = "unavailable"
    return facts


def pick_window(ax: Any, app: Any, want: dict[str, Any] | None) -> tuple[Any, dict[str, Any], dict[str, Any] | None]:
    """The window an action is scoped to -> (RETAINED window element or None, window facts, refusal).

    ``want`` None: the focused window, else the main one, else the first (as ``tree`` roots); a ``window_id`` or
    ``title`` names one of the application's windows exactly. Unread selector attributes or several matching
    windows forbid a pick; unavailable window ids do not prevent selection by a fully read exact title."""
    if not want:
        for name in ("AXFocusedWindow", "AXMainWindow"):
            val, err = ax.attr(app, name)
            if err in _ABSENT:
                continue
            if err != 0:
                return None, {}, {"reason": "windows_unread",
                                  "error": f"{name} could not be read ({AX_ERR.get(err, str(err))})"}
            if not (isinstance(val, tuple) and len(val) == 2 and val[0] == "ref" and val[1]):
                return None, {}, {"reason": "windows_unread", "error": f"{name} returned a malformed window reference"}
            return val[1], _window_facts(ax, val[1], name), None
    refs, err = ax.window_refs(app)
    if err != 0:
        return None, {}, {"reason": "windows_unread",
                          "error": f"the application's windows could not be read ({AX_ERR.get(err, str(err))})"}
    rows = [_window_facts(ax, ref, "AXWindows") for ref in refs]
    key = "window_id" if want and "window_id" in want else "title"
    attr = "window_id" if key == "window_id" else "AXTitle"
    unread = bool(want) and any(attr in row.get("unread", {}) for row in rows)
    if not want:
        picked = [0] if refs else []
    else:
        picked = [i for i, row in enumerate(rows) if row.get(key) == want[key]]
    chosen = picked[0] if len(picked) == 1 and not unread else None
    refusal: dict[str, Any] | None = None
    if chosen is None:
        if not refs:
            refusal = {"reason": "no_window", "error": "the application has no window"}
        elif unread:
            hint = "; name the window by its exact title" if key == "window_id" else "; retry the read or use a window_id"
            refusal = {"reason": "windows_unread", "windows": rows,
                       "error": f"not every window's {key} could be read, so the named window cannot be selected{hint}"}
        elif len(picked) > 1:
            refusal = {"reason": "window_ambiguous", "windows": rows,
                       "error": f"{len(picked)} windows have {key} {want[key]!r}; name one by a unique window_id or exact title"}
        else:
            refusal = {"reason": "window_not_found", "windows": rows,
                       "error": f"no window of the application has {key} {want[key]!r}"}
    for i, ref in enumerate(refs):
        if i != chosen:
            ax.CFRelease(ref)
    return (refs[chosen] if chosen is not None else None), (rows[chosen] if chosen is not None else {}), refusal


def _precondition(ax: Any, elem: Any, before: dict[str, Any], raw: tuple[Any, Any, Any], sel: dict[str, Any],
                  action: str) -> dict[str, Any] | None:
    """A fact that forbids the action, read fresh from the element; None when it may proceed. The fresh read must
    still be the identity the walk matched (a redrawn view can put another element where the match was)."""
    unread = before.get("unread") or {}
    if "element" in unread:
        return {"reason": "element_unread", "error": f"the matched element could not be read again before acting ({unread['element']})"}
    named = [attr for key in sel if key in _NAMED for attr in _NAMED[key] if attr in unread]
    if named:
        return {"reason": "element_unread",
                "error": f"re-read before acting, the element did not answer {', '.join(named)} ({', '.join(unread[a] for a in named)}), "
                         "so it cannot be confirmed as the match"}
    if not matches(sel, raw[0], raw[1], raw[2], before.get("identifier")):
        return {"reason": "element_changed",
                "error": f"re-read before acting, the element no longer matches {_identity(sel)} (the view was redrawn); read the window again"}
    if before.get("enabled") is False:
        return {"reason": "element_disabled", "error": "the element is disabled; the application would ignore the action"}
    if action == "press":
        names, err = ax.action_names(elem)
        if err != 0:
            return {"reason": "actions_unread", "error": f"the element did not answer which actions it performs ({AX_ERR.get(err, str(err))})"}
        if "AXPress" not in names:
            return {"reason": "action_unsupported", "actions": names,
                    "error": f"the element performs no AXPress (its actions: {', '.join(names) or 'none'}); "
                             "this tool acts only through the element's own actions, never through the pointer"}
        return None
    can, err = ax.settable(elem, "AXValue")
    if err != 0 or can is None:
        return {"reason": "value_settable_unread", "error": f"the element did not answer whether its value can be set ({AX_ERR.get(err, str(err))})"}
    if not can:
        return {"reason": "value_not_settable",
                "error": "the application does not let this element's value be set; type into it (type_text after it has focus) or use the app's own commands"}
    value = before.get("value")
    if value is not None and not isinstance(value, str):
        return {"reason": "unsupported_value_type",
                "error": f"the element's value is a {type(value).__name__}, not text; press is the action for a checkbox or radio button"}
    return None


def _note(action: str, outcome: str, after: dict[str, Any], verified: bool | None) -> str:
    if outcome == "unknown":
        return ("the action call was made but the application did not answer it within the messaging timeout: it may still act; "
                "do not retry blindly, read the window (ax_tree) first")
    if outcome == "failed":
        return "the application refused the action; nothing is known to have changed"
    if action == "press":
        text = "the element performed AXPress"
    elif verified:
        text = "the element took the new value and reads it back"
    elif verified is False:
        text = "the element took the new value, but reads back a different one (the application may have normalized or rejected it)"
    else:
        text = "the element took the new value; it could not be read back"
    gone = (after.get("unread") or {}).get("element")
    if gone:
        return text + f"; afterwards the element could not be read ({gone}: it may be gone, a closed dialog or a rebuilt view)"
    return text + "; this tool observed no other UI change: verify with ax_tree, a screenshot or the app's own commands"


def act_in_window(ax: Any, win: Any, sel: dict[str, Any], action: str, *, max_nodes: int, deadline_s: float,
                  settle: Callable[[float], None] = time.sleep) -> dict[str, Any]:
    """Find the ONE element the selector names under ``win`` (owned here: released on every exit) and perform the
    action through the element itself. ``ax`` is any object with the ``_AX`` surface, so the choice, refusal,
    outcome and readback rules are testable without a desktop.

    A refusal does nothing (``dispatched: false``): no match, several without ``nth``, a walk that did not finish
    (match order and uniqueness unproved, even with ``nth``), an element that could not be read again,
    a disabled element, an element without the action, a value the application does not let be set. An action call
    that was made is ``dispatched: true`` with ``action_outcome`` ``completed`` (the application accepted it; the
    receipt's effect mark), ``unknown`` (no answer within the messaging timeout; it may still act) or ``failed`` (it
    refused: nothing is known to have changed). A completed action stays completed when the readback after it fails.
    """
    found: list[Any] = []
    unread: list[Any] = []

    def read_node(el):
        err, vals = ax.multi(el, _WALK_ATTRS)
        if err == 0:
            clean, failed = split_attr_errors(_WALK_ATTRS, list(vals))
            role, _subrole, title, desc = clean[:4]
            verdict = classify(sel, role, title, desc, clean[-1], failed)
            if verdict == "match":
                found.append(ax.CFRetain(el))
            elif verdict == "unread":
                unread.append(ax.CFRetain(el))
        return err, vals

    try:
        walked = walk(win, read_node, ax.children, max_nodes=max_nodes, deadline_s=deadline_s,
                      skip_children=() if sel.get("include_web") else ("AXWebArea",), release=ax.CFRelease)
        complete = not (_INCOMPLETE & set(walked["truncated_by"]))
        facts = {"matched": len(found), "unreadable": len(unread), "searched_whole_window": complete, "visited": walked["visited"],
                 "truncated_by": walked["truncated_by"], "errors": walked["errors"], "elapsed_ms": walked["elapsed_ms"]}
        identity = _identity(sel)
        if unread:
            shown = [{"index": i + 1, **read_element(ax, el)} for i, el in enumerate(unread[:_SHOWN])]
            return {"ok": False, "reason": "candidates_unread", "dispatched": False, "unreadable_candidates": shown,
                    "error": f"{len(unread)} element(s) did not answer an attribute the selector names ({identity}), so the choice "
                             "cannot be made (one of them may be the target); retry, or name the element by another attribute",
                    "note": "no action was made", **facts}
        which = choose(len(found), sel.get("nth"), complete)
        if which != "one":
            candidates = [{"index": i + 1, **read_element(ax, el)} for i, el in enumerate(found[:_SHOWN])]
            if which == "none":
                error = (f"only {len(found)} element(s) match {identity}, nth={sel['nth']} was asked" if sel.get("nth")
                         else f"no element of the window matches {identity}")
                reason = "element_not_found"
            elif which == "ambiguous":
                reason = "element_ambiguous"
                error = f"{len(found)} elements match {identity}; pass nth (1-based, in this order) or add label/identifier"
            else:
                reason = "window_not_fully_read"
                error = (f"{len(found)} element(s) match {identity} but the walk did not finish (" + ", ".join(walked["truncated_by"]) +
                         "), so match order and uniqueness are unproved: retry the read or raise max_nodes/timeout_ms")
            return {"ok": False, "reason": reason, "error": error, "dispatched": False, "candidates": candidates,
                    "note": "no action was made", **facts}
        el = found[(sel.get("nth") or 1) - 1]
        before, raw = _read(ax, el)
        refusal = _precondition(ax, el, before, raw, sel, action)
        if refusal:
            return {"ok": False, **refusal, "dispatched": False, "element": before, "note": "no action was made", **facts}
        code = ax.perform(el, "AXPress") if action == "press" else ax.set_string(el, "AXValue", sel["text"])
        outcome = "completed" if code == 0 else ("unknown" if code == -25204 else "failed")
        payload: dict[str, Any] = {"ok": outcome == "completed", "dispatched": True, "action_outcome": outcome,
                                   "element": before, **facts}
        if code != 0:
            payload["ax_error"] = AX_ERR.get(code, str(code))
            payload["reason"] = "outcome_unknown" if outcome == "unknown" else "action_failed"
        if outcome == "completed":
            payload["observed_effect"] = True  # the application accepted the action: dispatch strength, not the goal
        try:
            settle(_SETTLE_S)
            after = read_element(ax, el)
        except Exception as exc:  # noqa: BLE001 - the action stands; only the look afterwards failed
            after = {"unread": {"element": f"{type(exc).__name__}: {exc}"}}
        payload["element_after"] = after
        if action == "set_value" and outcome != "failed":
            payload["verified"] = after["value"] == sel["text"] if "value" in after else None
        payload["note"] = _note(action, outcome, after, payload.get("verified"))
        return payload
    finally:
        for el in found + unread:
            ax.CFRelease(el)


def op_act(pid: int, action: str, selector: Any, max_nodes: int = ACT_DEFAULTS["max_nodes"],
           timeout_ms: int = ACT_DEFAULTS["timeout_ms"], *, ax: Any = None,
           front: Callable[[], dict[str, Any] | None] = front_app, settle: Callable[[float], None] = time.sleep) -> dict[str, Any]:
    """``act``: one element of application ``pid``, named by identity, one of its own actions, a readback. Who is in
    front is read before and after and REPORTED (``front_before``, ``front_after``, ``target_in_front``), never
    changed: the action goes to the element through its process, not to whatever is in front."""
    out: dict[str, Any] = {"pid": int(pid), "do": action}
    if action not in ACTIONS:
        return {**out, "ok": False, "reason": "unknown_action", "dispatched": False, "error": f"do must be press or set_value, not {action!r}"}
    sel, err = parse_selector(selector)
    if not err and action == "set_value" and "text" not in sel:
        err = "set_value needs text as a string: pass an empty string to clear the field"
    if not err and action == "press" and "text" in sel:
        err = "text belongs to set_value only: press ignores it"
    if err:
        return {**out, "ok": False, "reason": "bad_selector", "dispatched": False, "error": err}
    out["select"] = {k: v for k, v in sel.items() if k not in ("text", "include_web")}
    ax = ax or _AX()
    if not ax.AXIsProcessTrusted():
        return {**out, "ok": False, "reason": "accessibility_untrusted", "dispatched": False}
    max_nodes = max(1, min(int(max_nodes), TREE_CEILINGS["max_nodes"]))
    timeout_ms = max(100, min(int(timeout_ms), TREE_CEILINGS["timeout_ms"]))
    out["front_before"] = front() or None
    app = ax.AXUIElementCreateApplication(int(pid))
    ax.set_timeout(app, min(1.0, timeout_ms / 1000.0))
    try:
        win, window, refusal = pick_window(ax, app, sel.get("window"))
        if refusal:
            out.update({"ok": False, "dispatched": False, "note": "no action was made", **refusal})
        else:
            out.update(act_in_window(ax, win, sel, action, max_nodes=max_nodes, deadline_s=timeout_ms / 1000.0, settle=settle))
            out["window"] = window
    finally:
        ax.CFRelease(app)
    out["front_after"] = front() or None
    out["target_in_front"] = (out["front_after"].get("pid") == int(pid)) if isinstance(out["front_after"], dict) else None
    out["limits"] = {"max_nodes": max_nodes, "timeout_ms": timeout_ms, "ceilings": TREE_CEILINGS}
    return out


# ----------------------------------------------------------------------------------------------
# operations
# ----------------------------------------------------------------------------------------------

def op_apps(limit: int = 50, ax: _AX | None = None) -> dict[str, Any]:
    apps, error = running_apps()
    if error and not apps:
        return {"ok": False, "reason": "apps_unread", "error": error}
    front = front_app() or {}
    ax = ax or _AX()
    trusted = bool(ax.AXIsProcessTrusted())
    rows = []
    for app in apps[: max(1, limit)]:
        row: dict[str, Any] = {"app": app["app"], "pid": app["pid"], "app_id": app["app_id"]}
        if front.get("pid"):
            row["frontmost"] = app["pid"] == front.get("pid")
        s = scriptable(app["path"])
        if s is not None:
            row["scriptable"] = s
        if trusted:
            elem = ax.AXUIElementCreateApplication(int(app["pid"]))
            ax.set_timeout(elem, 0.25)
            try:
                windows, unread = ax.windows(elem)
            finally:
                ax.CFRelease(elem)
            if unread:
                row["windows_unread"] = unread
            else:
                row["windows"] = windows
        rows.append(row)
    out: dict[str, Any] = {"ok": True, "frontmost": front or None, "apps": rows, "titles_available": trusted,
                           "truncated": len(apps) > len(rows), "limit": limit, "total": len(apps)}
    if error:
        out["warning"] = error
    return out


def op_resolve(query: str) -> dict[str, Any]:
    apps, error = running_apps()
    if error and not apps:
        return {"ok": False, "reason": "apps_unread", "error": error, "query": query, "candidates": []}
    matches = resolve(query, apps)
    out = {"ok": True, "query": query, "candidates": [{"app": a["app"], "pid": a["pid"], "app_id": a["app_id"]} for a in matches]}
    if not matches:
        out["ok"] = False
        out["reason"] = "not_running"
        out["running"] = [a["app"] for a in apps][:40]
        if len(apps) > 40:
            out["running_more"] = len(apps) - 40
    elif len(matches) > 1:
        out["ok"] = False
        out["reason"] = "ambiguous"
    return out


def _wait_front(pid: int, wait_ms: int) -> tuple[dict[str, Any] | None, int]:
    """Poll who is in front until it is ``pid`` or ``wait_ms`` passed; (front, waited_ms)."""
    t0 = time.monotonic()
    front = front_app()
    while (not front or front.get("pid") != pid) and (time.monotonic() - t0) * 1000 < max(200, int(wait_ms)):
        time.sleep(0.05)
        front = front_app()
    return front or None, int((time.monotonic() - t0) * 1000)


def _activate_pid(ax: _AX, pid: int, wait_ms: int) -> tuple[str, dict[str, Any] | None, int]:
    """Bring ``pid`` forward: each mechanism is accepted only after a fresh read shows the target in front;
    an accepted-but-ignored request (macOS 14+ may ignore activation from a background process) falls through
    to the next one. Returns (mechanism that worked or the last one tried, front read, waited_ms)."""
    waited_total = 0
    front: dict[str, Any] | None = None
    tried = "none"
    if ax.AXIsProcessTrusted():
        app = ax.AXUIElementCreateApplication(pid)
        try:
            ax.set_timeout(app, 0.5)
            accepted = ax.AXUIElementSetAttributeValue(app, ax.cfs("AXFrontmost"), ax.kCFBooleanTrue) == 0
        finally:
            ax.CFRelease(app)
        if accepted:
            tried = "ax_frontmost"
            front, waited = _wait_front(pid, wait_ms)
            waited_total += waited
            if front and front.get("pid") == pid:
                return tried, front, waited_total
    try:
        from AppKit import NSApplicationActivateIgnoringOtherApps, NSRunningApplication  # type: ignore

        running = NSRunningApplication.runningApplicationWithProcessIdentifier_(pid)
        accepted = running is not None and bool(running.activateWithOptions_(NSApplicationActivateIgnoringOtherApps))
    except Exception:  # noqa: BLE001 - pyobjc absent or refused; the next mechanism follows
        accepted = False
    if accepted:
        tried = "nsrunningapplication"
        front, waited = _wait_front(pid, wait_ms)
        waited_total += waited
        if front and front.get("pid") == pid:
            return tried, front, waited_total
    # System Events needs the Automation grant for this helper's parent; macOS may show a one-time consent prompt.
    _sh(["/usr/bin/osascript", "-e",
         f'tell application "System Events" to set frontmost of (first process whose unix id is {int(pid)}) to true'], timeout=5)
    tried = "system_events"
    front, waited = _wait_front(pid, wait_ms)
    return tried, front, waited_total + waited


def op_activate(query: str, wait_ms: int = 2000) -> dict[str, Any]:
    resolved = op_resolve(query)
    if not resolved["ok"]:
        return resolved
    target = resolved["candidates"][0]
    before = front_app() or None
    out: dict[str, Any] = {"ok": True, "target": target, "front_before": before}
    if before and before.get("pid") == target["pid"]:
        out.update({"activated": False, "already_front": True, "verified": True, "front": before, "waited_ms": 0})
        return out
    method, front, waited = _activate_pid(_AX(), int(target["pid"]), int(wait_ms))
    verified = bool(front and front.get("pid") == target["pid"])
    # A switch is observed only when the front was read as another application before: with the before
    # read unknown the target is verified in front, but whether anything changed is unknown.
    activated: bool | None = (verified and before.get("pid") != target["pid"]) if before else (None if verified else False)
    out.update({"activated": activated, "verified": verified, "method": method, "front": front or None, "waited_ms": waited})
    if not verified:
        out["ok"] = False
        if front is None:
            out.update({"reason": "front_unread", "activated": None, "error": "lsappinfo named no frontmost application after the switch"})
        else:
            out["reason"] = "not_frontmost_after_activate"
    return out


def op_tree(pid: int, max_nodes: int = 150, timeout_ms: int = 3000, root: str = "window", include_web: bool = False) -> dict[str, Any]:
    ax = _AX()
    if not ax.AXIsProcessTrusted():
        return {"ok": False, "reason": "accessibility_untrusted", "pid": pid}
    max_nodes = max(1, min(int(max_nodes), TREE_CEILINGS["max_nodes"]))
    timeout_ms = max(100, min(int(timeout_ms), TREE_CEILINGS["timeout_ms"]))
    app = ax.AXUIElementCreateApplication(int(pid))
    ax.set_timeout(app, min(1.0, timeout_ms / 1000.0))
    root_ref = None
    root_from = None
    try:
        for name in (("AXFocusedUIElement",) if root == "focused" else ()) + ("AXFocusedWindow", "AXMainWindow"):
            val, err = ax.attr(app, name)
            if err == 0 and val and isinstance(val, tuple):
                root_ref, root_from = val[1], name
                break
        if root_ref is None:
            val, err = ax.attr(app, "AXWindows")
            if err == 0 and val and isinstance(val, tuple):
                arr = val[1]
                if ax.is_array(arr) and ax.CFArrayGetCount(arr) > 0:
                    root_ref, root_from = ax.CFRetain(ax.CFArrayGetValueAtIndex(arr, 0)), "AXWindows[0]"
                ax.CFRelease(arr)
            if root_ref is None:
                return {"ok": False, "reason": "no_window" if err in (0, -25212) else AX_ERR.get(err, str(err)), "pid": pid}
        title, _e = ax.attr(root_ref, "AXTitle")
        ref, root_ref = root_ref, None  # the walk owns the root from here and releases it on every exit
        result = walk(ref, lambda el: ax.multi(el, _WALK_ATTRS), ax.children, max_nodes=max_nodes,
                      deadline_s=timeout_ms / 1000.0, skip_children=() if include_web else ("AXWebArea",),
                      release=ax.CFRelease)
    finally:
        if root_ref is not None:
            ax.CFRelease(root_ref)
        ax.CFRelease(app)
    result.update({"pid": pid, "root": root_from, "window": title if isinstance(title, str) else None,
                   "limits": {"max_nodes": max_nodes, "timeout_ms": timeout_ms, "ceilings": TREE_CEILINGS}})
    if result["visited"] == 0:
        # Not one node could be read: a failed observation, not an empty window.
        result.update({"ok": False, "reason": "accessibility_read_failed",
                       "error": "no element of the window could be read: " + json.dumps(result["errors"])})
        return result
    result.update({"ok": True, "marks": marks_from(result["nodes"])})
    return result


def op_perms() -> dict[str, Any]:
    ax = _AX()
    cg = ctypes.CDLL(_CG_PATH)
    out: dict[str, Any] = {"ok": True, "accessibility": "trusted" if ax.AXIsProcessTrusted() else "untrusted"}
    for key, name in (("screen_recording", "CGPreflightScreenCaptureAccess"), ("post_events", "CGPreflightPostEventAccess"),
                      ("listen_events", "CGPreflightListenEventAccess")):
        try:
            fn = getattr(cg, name)
            fn.restype = c_ubyte
            out[key] = "granted" if fn() else "denied"
        except Exception:  # noqa: BLE001
            out[key] = "unknown"
    return out


def op_displays() -> dict[str, Any]:
    cg = ctypes.CDLL(_CG_PATH)
    cg.CGMainDisplayID.restype = c_uint32
    cg.CGDisplayBounds.restype = CGRect
    cg.CGDisplayBounds.argtypes = [c_uint32]
    cg.CGDisplayPixelsWide.restype = c_size_t
    cg.CGDisplayPixelsWide.argtypes = [c_uint32]
    cg.CGDisplayPixelsHigh.restype = c_size_t
    cg.CGDisplayPixelsHigh.argtypes = [c_uint32]
    cg.CGGetActiveDisplayList.argtypes = [c_uint32, POINTER(c_uint32), POINTER(c_uint32)]
    count = c_uint32(0)
    if cg.CGGetActiveDisplayList(0, None, byref(count)) != 0:
        return {"ok": False, "error": "CGGetActiveDisplayList failed"}
    total = max(1, int(count.value))
    ids = (c_uint32 * total)()
    if cg.CGGetActiveDisplayList(total, ids, byref(count)) != 0:
        return {"ok": False, "error": "CGGetActiveDisplayList failed"}
    main = cg.CGMainDisplayID()
    rows = []
    secondary = 1
    for i in range(count.value):
        did = ids[i]
        r = cg.CGDisplayBounds(did)
        # screencapture -D numbers the main display 1 and the others from 2 in the system's order.
        if did == main:
            capture_index = 1
        else:
            secondary += 1
            capture_index = secondary
        rows.append({"id": int(did), "main": did == main, "x": int(r.origin.x), "y": int(r.origin.y),
                     "width": int(r.size.w), "height": int(r.size.h),
                     "pixels_wide": int(cg.CGDisplayPixelsWide(did)), "pixels_high": int(cg.CGDisplayPixelsHigh(did)),
                     "index": capture_index})
    return {"ok": True, "displays": rows}


def main(argv: list[str]) -> int:
    if not argv:
        print(json.dumps({"ok": False, "error": "usage: macos_native.py <apps|resolve|activate|tree|act|perms|displays> [args]"}))
        return 2
    op, args = argv[0], argv[1:]
    try:
        if op == "apps":
            out = op_apps(int(args[0]) if args else 50)
        elif op == "resolve":
            out = op_resolve(args[0] if args else "")
        elif op == "activate":
            out = op_activate(args[0] if args else "", int(args[1]) if len(args) > 1 else 2000)
        elif op == "tree":
            out = op_tree(int(args[0]), int(args[1]) if len(args) > 1 else 150, int(args[2]) if len(args) > 2 else 3000,
                          args[3] if len(args) > 3 else "window", (args[4].lower() == "true") if len(args) > 4 else False)
        elif op == "act":
            out = op_act(int(args[0]), args[1] if len(args) > 1 else "", args[2] if len(args) > 2 else "",
                         int(args[3]) if len(args) > 3 else ACT_DEFAULTS["max_nodes"],
                         int(args[4]) if len(args) > 4 else ACT_DEFAULTS["timeout_ms"])
        elif op == "perms":
            out = op_perms()
        elif op == "displays":
            out = op_displays()
        else:
            out = {"ok": False, "error": f"unknown op {op!r}"}
    except Exception as exc:  # noqa: BLE001 - the plugin renders the failure; nothing here may crash silently
        out = {"ok": False, "error": f"{type(exc).__name__}: {exc}", "op": op}
    # ASCII-only output: the parent decodes it the same way under any locale.
    print(json.dumps(out, ensure_ascii=True))
    return 0 if out.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
