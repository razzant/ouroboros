"""windows_computer_use.win32fb - the plain Win32 fallback reader.

Ported from the fallback half of Praxis/Hélène ``uia.rs`` (win32_walk,
window_title/class/rect/dpi): EnumWindows top-level listing + GetWindow/GW_CHILD
walk with timed ``WM_GETTEXT`` (a hung window must not hang us), roles inferred
from window classes. Pure ctypes, no COM. Windows-only imports at call time.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
from typing import Any, Optional

from .core import WIN32_TEXT_TIMEOUT_MS, WORKER_GRACE_MS, ReadLimits, clip_utf16, win32_role
from .tree import Node, Rect, Walk, WindowInfo

WM_GETTEXT = 0x000D
WM_GETTEXTLENGTH = 0x000E
SMTO_ABORTIFHUNG = 0x0002
GW_CHILD = 5
GW_HWNDNEXT = 2

_user32 = ctypes.WinDLL("user32", use_last_error=True)
_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

_user32.IsWindow.argtypes = [wintypes.HWND]
_user32.IsWindow.restype = wintypes.BOOL
_user32.IsWindowVisible.argtypes = [wintypes.HWND]
_user32.IsWindowVisible.restype = wintypes.BOOL
_user32.IsWindowEnabled.argtypes = [wintypes.HWND]
_user32.IsWindowEnabled.restype = wintypes.BOOL
_user32.GetForegroundWindow.argtypes = []
_user32.GetForegroundWindow.restype = wintypes.HWND
_user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
_user32.GetWindowThreadProcessId.restype = wintypes.DWORD
_user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
_user32.GetWindowRect.restype = wintypes.BOOL
_user32.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
_user32.GetClassNameW.restype = ctypes.c_int
_user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
_user32.GetWindowTextLengthW.restype = ctypes.c_int
_user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
_user32.GetWindowTextW.restype = ctypes.c_int
_user32.GetWindow.argtypes = [wintypes.HWND, wintypes.UINT]
_user32.GetWindow.restype = wintypes.HWND
_user32.SendMessageTimeoutW.argtypes = [
    wintypes.HWND, wintypes.UINT, ctypes.c_size_t, ctypes.c_size_t,
    wintypes.UINT, wintypes.UINT, ctypes.POINTER(ctypes.c_size_t),
]
_user32.SendMessageTimeoutW.restype = ctypes.c_ssize_t
_user32.EnumWindows.argtypes = [ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM), wintypes.LPARAM]
_user32.EnumWindows.restype = wintypes.BOOL
# GetDpiForWindow needs Win10 1607+; absent on older systems.
try:
    _user32.GetDpiForWindow.argtypes = [wintypes.HWND]
    _user32.GetDpiForWindow.restype = wintypes.UINT
except AttributeError:
    _user32.GetDpiForWindow = None  # type: ignore[attr-defined]


def _hwnd_int(hwnd) -> int:
    if isinstance(hwnd, wintypes.HWND):
        return int(hwnd.value or 0)
    return int(hwnd or 0)


def is_window(hwnd) -> bool:
    return bool(_user32.IsWindow(hwnd))


def window_dpi(hwnd) -> int:
    if _user32.GetDpiForWindow is not None:
        got = int(_user32.GetDpiForWindow(hwnd))
        if got:
            return got
    return 96


def window_rect(hwnd) -> Optional[Rect]:
    rect = wintypes.RECT()
    if _user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        made = Rect(rect.left, rect.top, rect.right, rect.bottom)
        if not made.is_empty():
            return made
    return None


def window_class(hwnd) -> str:
    buffer = ctypes.create_unicode_buffer(512)
    copied = _user32.GetClassNameW(hwnd, buffer, 512)
    return buffer.value if copied > 0 else ""


def window_title(hwnd, unread: Optional[list[str]] = None) -> str:
    """GetWindowTextW in a foreign process only returns the caption; the label on
    a button lives behind WM_GETTEXT, which must never be sent without a timeout."""
    length = _user32.GetWindowTextLengthW(hwnd)
    if length > 0:
        buffer = ctypes.create_unicode_buffer(length + 1)
        copied = _user32.GetWindowTextW(hwnd, buffer, length + 1)
        if copied > 0:
            return buffer.value
    result = ctypes.c_size_t(0)
    ok = _user32.SendMessageTimeoutW(
        hwnd, WM_GETTEXTLENGTH, 0, 0, SMTO_ABORTIFHUNG, WIN32_TEXT_TIMEOUT_MS,
        ctypes.byref(result),
    )
    if not ok:
        if unread is not None:
            unread.append("name")
        return ""
    length = int(result.value or 0)
    if length == 0 or length > 8192:
        if length > 8192 and unread is not None:
            unread.append("name")  # known length, but the text was not read
        return ""
    buffer = ctypes.create_unicode_buffer(length + 1)
    result = ctypes.c_size_t(0)
    ok = _user32.SendMessageTimeoutW(
        hwnd, WM_GETTEXT, (length + 1), ctypes.addressof(buffer), SMTO_ABORTIFHUNG,
        WIN32_TEXT_TIMEOUT_MS, ctypes.byref(result),
    )
    if not ok:
        if unread is not None:
            unread.append("name")
        return ""
    copied = int(result.value or 0)
    return buffer.value[:copied] if copied else ""


def window_pid(hwnd) -> int:
    pid = wintypes.DWORD(0)
    _user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    return int(pid.value)


def resolve_window(hwnd_arg: Optional[int]) -> WindowInfo:
    """Pick the window: the named one or the foreground one. One place for all
    tools, so "foreground" cannot drift between them."""
    resolved_from = "argument"
    if hwnd_arg is not None:
        hwnd = wintypes.HWND(int(hwnd_arg))
    else:
        hwnd = wintypes.HWND(_hwnd_int(_user32.GetForegroundWindow()))
        resolved_from = "foreground"
    if not hwnd or not _hwnd_int(hwnd):
        raise ValueError("there is no foreground window in the interactive desktop")
    if not is_window(hwnd):
        raise ValueError(f"window 0x{_hwnd_int(hwnd):X} no longer exists")
    unread: list[str] = []
    title = window_title(hwnd, unread)
    return WindowInfo(
        hwnd=_hwnd_int(hwnd),
        title=title,
        cls=window_class(hwnd),
        pid=window_pid(hwnd),
        rect=window_rect(hwnd),
        foreground=_hwnd_int(_user32.GetForegroundWindow()) == _hwnd_int(hwnd),
        resolved_from=resolved_from,
        dpi=window_dpi(hwnd),
        unread=("title",) if unread else (),
    )


def list_windows(limit: int = 100) -> list[dict[str, Any]]:
    """Visible top-level windows with titles, newest-ish first (Z-order)."""
    found: list[dict[str, Any]] = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def on_window(hwnd, _lparam):
        if len(found) >= limit:
            return False  # stop enumeration
        if not _user32.IsWindowVisible(hwnd):
            return True
        unread: list[str] = []
        title = window_title(hwnd, unread)
        if not title and not unread:
            return True
        found.append(
            WindowInfo(
                hwnd=_hwnd_int(hwnd),
                title=title,
                cls=window_class(hwnd),
                pid=window_pid(hwnd),
                rect=window_rect(hwnd),
                foreground=_hwnd_int(_user32.GetForegroundWindow()) == _hwnd_int(hwnd),
                resolved_from="enumeration",
                dpi=window_dpi(hwnd),
                unread=("title",) if unread else (),
            ).json()
        )
        return True

    _user32.EnumWindows(on_window, 0)
    return found


def win32_walk(window: WindowInfo, limits: ReadLimits, visible_only: bool = False) -> Walk:
    """Classic child-window walk; Chromium/Electron/WPF/UWP look empty here."""
    import time

    started = time.monotonic()
    deadline = started + (limits.timeout_ms + WORKER_GRACE_MS) / 1000.0
    walk = Walk()
    root = wintypes.HWND(window.hwnd)
    queue: list[tuple[wintypes.HWND, int, Optional[int]]] = [(root, 0, None)]

    while queue:
        if len(walk.nodes) >= limits.max_nodes:
            walk.mark("max_nodes")
            break
        if time.monotonic() >= deadline:
            walk.mark("timeout")
            break
        hwnd, depth, parent = queue.pop(0)
        visible = bool(_user32.IsWindowVisible(hwnd))
        if visible_only and not visible:
            walk.skipped_offscreen += 1
            continue
        walk.depth_reached = max(walk.depth_reached, depth)
        index = len(walk.nodes)
        cls = window_class(hwnd)
        unread: list[str] = []
        name, name_cut = clip_utf16(window_title(hwnd, unread) or "", limits.max_text_chars)
        class_text, class_cut = clip_utf16(cls, limits.max_text_chars)
        node = Node(
            parent=parent,
            depth=depth,
            role=win32_role(cls),
            name=name or None,
            cls=class_text or None,
            hwnd=_hwnd_int(hwnd),
            rect=window_rect(hwnd),
            enabled=bool(_user32.IsWindowEnabled(hwnd)),
            offscreen=not visible,
            text_truncated=name_cut or class_cut,
            unread=tuple(unread),
        )
        if depth + 1 > limits.max_depth:
            node.children_unread = "max_depth"
            walk.subtrees_unread += 1
            walk.mark("max_depth")
        else:
            child = _user32.GetWindow(hwnd, GW_CHILD)
            seen = 0
            while child and _hwnd_int(child):
                if seen >= limits.max_children_per_node:
                    node.children_unread = "max_children_per_node"
                    walk.subtrees_unread += 1
                    walk.mark("max_children_per_node")
                    break
                queue.append((child, depth + 1, index))
                child = _user32.GetWindow(child, GW_HWNDNEXT)
                seen += 1
        walk.nodes.append(node)

    walk.discovered_unread = len(queue)
    walk.walk_ms = int((time.monotonic() - started) * 1000)
    return walk
