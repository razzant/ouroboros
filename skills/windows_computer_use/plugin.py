"""windows_computer_use plugin: 4 tools over the ported UIA core.

Tool surface (kept close to the Praxis/Hélène verbs so the mental model is
one family):

- window_list: visible top-level windows (pick an hwnd here)
- window_read: a window as a text tree (roles/labels/states/rects)
- element_find: search by selector, return every match
- element_act: invoke/set_value/toggle/expand/collapse/select/scroll_into_view/focus

COM hang isolation: this extension declares an isolated dependency (comtypes),
so the host runs every handler in a short-lived child process it can kill. The
handler itself keeps the Rust module's honest-degradation shape: a worker
thread with a deadline; on silence the thread is abandoned (threads cannot be
killed in Python - the process boundary does that job) and window_read falls
back to the plain Win32 reader naming ``fallback_reason``; act/find report a
typed timeout instead of pretending.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from . import core
from .core import SWEEP_MS, WORKER_GRACE_MS


class ToolError(ValueError):
    """A contract refusal (bad arguments, empty selector, ...)."""


def _fail(message: str) -> dict[str, Any]:
    return {"ok": False, "error": message}


def _ensure_windows() -> None:
    import sys

    if sys.platform != "win32":
        raise ToolError("windows_computer_use requires an interactive Windows session")


class _Worker:
    """Run fn in a thread; bounded wait; on silence return None (abandoned)."""

    def __init__(self, fn, wait_sec: float):
        self.fn = fn
        self.wait_sec = wait_sec
        self.result: Any = None
        self.error: BaseException | None = None

    def run(self):
        done = threading.Event()

        def inner():
            try:
                self.result = self.fn()
            except BaseException as error:  # surfaced below
                self.error = error
            finally:
                done.set()

        threading.Thread(target=inner, name="wcu-worker", daemon=True).start()
        if not done.wait(self.wait_sec):
            return None  # abandoned; the process exit / next call does not reuse it
        if self.error is not None:
            raise self.error
        return self.result


class WindowsComputerUse:
    def __init__(self, api: Any) -> None:
        self.api = api

    # ── window_list ────────────────────────────────────────────────────────
    def window_list(self, limit: int = 100) -> dict[str, Any]:
        _ensure_windows()
        from . import win32fb

        try:
            limit = max(1, min(int(limit), 500))
        except (TypeError, ValueError):
            raise ToolError(f"limit must be an integer, not {limit!r}") from None
        windows = win32fb.list_windows(limit)
        return {"ok": True, "windows": windows, "count": len(windows), "shown": len(windows)}

    # ── window_read ────────────────────────────────────────────────────────
    def window_read(self, **kwargs: Any) -> dict[str, Any]:
        _ensure_windows()
        plan = core.read_plan(kwargs)
        from . import win32fb

        window = win32fb.resolve_window(plan.hwnd)
        started = time.monotonic()
        if plan.force_win32:
            return self._render_win32(plan, window, "backend=win32 was asked for, so UI Automation was not consulted at all", started)

        wait_sec = (plan.limits.timeout_ms + WORKER_GRACE_MS) / 1000.0

        def do_uia():
            # COM objects must be created INSIDE the thread that uses them:
            # an STA apartment's interface pointers do not marshal across
            # threads on their own, and the worker is where all UIA calls run.
            from . import win_uia

            automation = win_uia.thread_automation()
            return win_uia.read_in_session(automation, plan, window)

        worker = _Worker(do_uia, wait_sec)
        try:
            result = worker.run()
        except Exception as error:
            return self._render_win32(plan, window, f"UI Automation did not come up: {error}", started)
        if result is None:
            return self._render_win32(
                plan, window,
                f"UI Automation did not answer within timeout_ms={plan.limits.timeout_ms} plus "
                f"worker_grace_ms={WORKER_GRACE_MS}; the target window is probably busy and its "
                "reader thread is still running",
                started,
            )
        return result

    def _render_win32(self, plan, window, reason: str, started: float) -> dict[str, Any]:
        from . import win32fb
        from .tree import WINDOW_READ_WIN32_DETAIL
        from .tree import render as render_tree

        walk = win32fb.win32_walk(window, plan.limits, visible_only=plan.visible_only)
        return render_tree(
            plan, window, walk,
            backend="win32", backend_detail=WINDOW_READ_WIN32_DETAIL,
            fallback_reason=reason,
            elapsed_ms=int((time.monotonic() - started) * 1000),
            extra_notes=[
                "this is the plain Win32 reading: it sees classic child windows only, so "
                "Chromium, Electron, WPF and UWP surfaces look empty here even when they are full",
                "checked/selected/expanded are absent on this path because Win32 does not report "
                "them; absent means unknown, not false",
            ],
        )

    # ── element_find ───────────────────────────────────────────────────────
    def element_find(self, **kwargs: Any) -> dict[str, Any]:
        _ensure_windows()
        plan = core.find_plan(kwargs)
        from . import win32fb

        window = win32fb.resolve_window(plan.hwnd)
        wait_ms = plan.timeout_ms + SWEEP_MS + WORKER_GRACE_MS * 2
        worker = _Worker(lambda: self._find_uia(plan, window.hwnd), wait_ms / 1000.0)
        try:
            result = worker.run()
        except Exception as error:
            return _fail(f"UI Automation failed: {error}")
        if result is None:
            return _fail(
                f"UI Automation did not answer within timeout_ms={plan.timeout_ms} plus "
                "sweep/grace; the target window is probably busy and its worker thread is still running"
            )
        return result

    def _find_uia(self, plan, hwnd_value: int):
        from . import win_uia

        automation = win_uia.thread_automation()
        return win_uia.find_in_session(automation, plan, hwnd_value)

    # ── element_act ───────────────────────────────────────────────────────
    def element_act(self, **kwargs: Any) -> dict[str, Any]:
        _ensure_windows()
        plan = core.act_plan(kwargs)
        from . import win32fb

        window = win32fb.resolve_window(plan.hwnd)
        wait_ms = plan.timeout_ms + WORKER_GRACE_MS * 2
        worker = _Worker(
            lambda: self._act_uia(plan, window),
            wait_ms / 1000.0,
        )
        try:
            result = worker.run()
        except Exception as error:
            # A contract refusal (missing pattern, guard, changed element) says
            # exactly what happened - it is NOT a "did not answer" frame claiming
            # an in-flight mutation may have acted.
            return _fail(str(error))
        if result is None:
            return _fail(
                f"UI Automation did not answer within timeout_ms={plan.timeout_ms} plus grace; "
                "the worker may still be running; an in-flight action may have acted. Do not "
                "retry blindly - check the window state first"
            )
        return result

    def _act_uia(self, plan, window):
        from . import win32fb, win_uia

        automation = win_uia.thread_automation()
        return win_uia.act_in_session(
            automation, plan, window, resolve_pid=lambda hwnd: win32fb.window_pid(hwnd),
        )


_SELECT_SCHEMA = {
    "type": "object",
    "properties": {
        "automation_id": {"type": "string"},
        "role": {"type": "string"},
        "name": {"type": "string"},
        "name_contains": {"type": "string"},
        "value_contains": {"type": "string"},
        "nth": {"type": "integer", "minimum": 0},
    },
    "additionalProperties": False,
}


def register(api: Any) -> None:
    impl = WindowsComputerUse(api)

    api.register_tool(
        "window_list",
        impl.window_list,
        description=(
            "List visible top-level Windows windows (hwnd, title, class, pid, rect). "
            "Pick an hwnd here for window_read/element_find/element_act."
        ),
        schema={"type": "object", "properties": {"limit": {"type": "integer", "default": 100}}},
        timeout_sec=15,
    )
    api.register_tool(
        "window_read",
        impl.window_read,
        description=(
            "Read a Windows window AS TEXT (roles, labels, values, states, rects) through UI "
            "Automation - much cheaper and more precise than a screenshot. Default: foreground "
            "window, tree shape, 400 nodes. Limits come back in the answer with ceilings; every "
            "cut is named in truncated_by/notes. backend=win32 forces the classic reader "
            "(classic windows only)."
        ),
        schema={
            "type": "object",
            "properties": {
                "hwnd": {"type": ["integer", "string"], "description": "window handle: 500, \"500\" or \"0x1F4\"; default foreground"},
                "backend": {"type": "string", "enum": ["auto", "win32"], "default": "auto"},
                "shape": {"type": "string", "enum": ["tree", "flat"], "default": "tree"},
                "text_contains": {"type": "string", "description": "keep only matching elements (and their path in tree shape)"},
                "visible_only": {"type": "boolean", "default": False},
                "max_nodes": {"type": "integer", "minimum": 1, "maximum": 5000},
                "max_depth": {"type": "integer", "minimum": 0},
                "max_children_per_node": {"type": "integer", "minimum": 1},
                "max_text_chars": {"type": "integer", "minimum": 1},
                "timeout_ms": {"type": "integer", "minimum": 100, "maximum": 20000},
            },
            "additionalProperties": False,
        },
        timeout_sec=30,
    )
    api.register_tool(
        "element_find",
        impl.element_find,
        description=(
            "Find elements in a window by selector (automation_id exact; role/name "
            "case-folded, Cyrillic included; all given fields must match). Returns every "
            "match up to limit. timeout_ms=0 (default) looks once; a positive value waits "
            "for the first match."
        ),
        schema={
            "type": "object",
            "properties": {
                "hwnd": {"type": ["integer", "string"]},
                "select": _SELECT_SCHEMA,
                "limit": {"type": "integer", "minimum": 1, "maximum": 200, "default": 20},
                "timeout_ms": {"type": "integer", "minimum": 0, "maximum": 60000},
                "max_nodes": {"type": "integer", "minimum": 1},
                "max_depth": {"type": "integer", "minimum": 0},
            },
            "required": ["select"],
            "additionalProperties": False,
        },
        timeout_sec=70,
    )
    api.register_tool(
        "element_act",
        impl.element_act,
        description=(
            "Act on the element a selector names, through its UIA pattern (no coordinate "
            "fallback): invoke | set_value | toggle | expand | collapse | select | "
            "scroll_into_view | focus. Two matches without select.nth is a refusal listing "
            "the candidates. The receipt shows the element before/after; whether the goal "
            "was achieved is yours to judge."
        ),
        schema={
            "type": "object",
            "properties": {
                "hwnd": {"type": ["integer", "string"]},
                "select": _SELECT_SCHEMA,
                "do": {
                    "type": "string",
                    "enum": ["invoke", "set_value", "toggle", "expand", "collapse", "select", "scroll_into_view", "focus"],
                },
                "text": {"type": "string", "description": "text for set_value (empty string clears; required for set_value)"},
                "timeout_ms": {"type": "integer", "minimum": 0, "maximum": 60000},
                "max_nodes": {"type": "integer", "minimum": 1},
                "max_depth": {"type": "integer", "minimum": 0},
            },
            "required": ["do", "select"],
            "additionalProperties": False,
        },
        timeout_sec=70,
    )
