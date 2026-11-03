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
typed timeout instead of pretending. Every tool answers a JSON string, the
PluginAPI tool contract the host relays from the child unchanged.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from . import core
from .core import SWEEP_MS, WORKER_GRACE_MS


class ToolError(ValueError):
    """A contract refusal (bad arguments, empty selector, ...)."""


WINDOW_LIST_SELECTION = (
    "visible top-level windows that have a title, front to back (Z-order); untitled windows "
    "are left out, and one whose title could not be read is listed with unread"
)


def _fail(message: str) -> dict[str, Any]:
    return {"ok": False, "error": message}


def _ensure_windows() -> None:
    import sys

    if sys.platform != "win32":
        raise ToolError("windows_computer_use requires an interactive Windows session")


def _client_error() -> str:
    """Ready the comtypes wrapper before any deadline starts (``win_uia.client_module``)."""
    from . import win_uia

    try:
        win_uia.client_module()
    except Exception as error:
        return str(error) or type(error).__name__
    return ""


_WORKER_THREAD = "wcu-worker"


def _workers_live() -> int:
    """UIA worker threads of this process still running now: an abandoned one
    stays alive (threads cannot be killed) until its provider call returns."""
    return sum(1 for thread in threading.enumerate() if thread.name == _WORKER_THREAD)


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

        thread = threading.Thread(target=inner, name=_WORKER_THREAD, daemon=True)
        thread.start()
        if not done.wait(self.wait_sec):
            return None  # abandoned; the process exit / next call does not reuse it
        thread.join()  # already returning: gone before any live-worker count
        if self.error is not None:
            raise self.error
        return self.result


class WindowsComputerUse:
    def __init__(self, api: Any) -> None:
        self.api = api

    # ── window_list ────────────────────────────────────────────────────────
    def window_list(self, limit: int = core.WINDOW_LIST_LIMIT_DEFAULT) -> dict[str, Any]:
        _ensure_windows()
        from . import win32fb

        cap = core.WINDOW_LIST_LIMIT_CAP
        try:
            requested = int(limit)
        except (TypeError, ValueError):
            raise ToolError(f"limit must be an integer, not {limit!r}") from None
        limit = max(1, min(requested, cap))
        notes = [] if limit == requested else [f"limit={requested} is outside [1, {cap}]; this listing used {limit}"]
        # The enumeration stops at its limit: one window more than listed tells a
        # whole listing from a cut one, whose total then stays unknown.
        found = win32fb.list_windows(limit + 1)
        windows = found[:limit]
        cut = len(found) > limit
        if cut:
            notes.append(f"stopped at limit={limit}: more windows exist and how many is unknown "
                         "(omitted counts display cuts only); "
                         + ("raise limit" if limit < cap else f"{cap} is the most one listing shows"))
        return {
            "ok": True, "windows": windows, "count": len(windows), "shown": len(windows), "limit": limit,
            "truncated": cut, "truncated_by": ["limit"] if cut else [], "total_known": not cut,
            "selection": WINDOW_LIST_SELECTION, "notes": notes,
        }

    # ── window_read ────────────────────────────────────────────────────────
    def window_read(self, **kwargs: Any) -> dict[str, Any]:
        _ensure_windows()
        plan = core.read_plan(kwargs)
        from . import win32fb

        window = win32fb.resolve_window(plan.hwnd)
        started = time.monotonic()
        if plan.force_win32:
            return self._render_win32(plan, window, "backend=win32 was asked for, so UI Automation was not consulted at all", started)

        error = _client_error()
        if error:
            return self._render_win32(plan, window, f"UI Automation did not come up: {error}", started)
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
                "reader thread was left running (uia_workers_live)",
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
            workers_live=_workers_live(),
        )

    # ── element_find ───────────────────────────────────────────────────────
    def element_find(self, **kwargs: Any) -> dict[str, Any]:
        _ensure_windows()
        plan = core.find_plan(kwargs)
        from . import win32fb

        window = win32fb.resolve_window(plan.hwnd)
        error = _client_error()
        if error:
            return _fail(f"UI Automation failed: {error}")
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
        error = _client_error()
        if error:
            return _fail(f"UI Automation failed: {error}")
        wait_ms = plan.timeout_ms + WORKER_GRACE_MS * 2
        worker = _Worker(
            lambda: self._act_uia(plan, window),
            wait_ms / 1000.0,
        )
        try:
            result = worker.run()
        except Exception as error:
            # Missing pattern, guard, traversal or changed element: no action call.
            return _fail(str(error))
        if result is None:
            # Abandoned anywhere from the sweep to the after-read: whether the call
            # was made is as unknown as its outcome, so neither is claimed.
            return {
                "ok": False,
                "error": f"UI Automation did not answer within timeout_ms={plan.timeout_ms} plus grace; "
                "the worker may still be running",
                "requested": plan.act, "mutating": core.act_mutating(plan.act),
                "action_call_made": "unknown", "action_outcome": "unknown", "select": plan.select.json(),
                "hint": "an in-flight action may have acted, or may still act. Do not retry blindly - "
                "check the window state first",
            }
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
    },
    "additionalProperties": False,
}

_ACT_SELECT_SCHEMA = {
    **_SELECT_SCHEMA,
    "properties": {**_SELECT_SCHEMA["properties"], "nth": {"type": "integer", "minimum": 0}},
}


def _json_tool(method):
    def run(**kwargs: Any) -> str:
        from .tree import fit_result, json_text

        return json_text(fit_result(method(**kwargs)))

    return run


def register(api: Any) -> None:
    impl = WindowsComputerUse(api)

    api.register_tool(
        "window_list",
        _json_tool(impl.window_list),
        description=(
            "List visible top-level Windows windows that have a title (hwnd, title, class, pid, "
            "rect), front to back. Pick an hwnd here for window_read/element_find/element_act. "
            "The listing stops at limit: truncated_by names that cut and total_known says "
            "whether the list is whole."
        ),
        schema={"type": "object", "properties": {"limit": {
            "type": "integer", "minimum": 1, "maximum": core.WINDOW_LIST_LIMIT_CAP,
            "default": core.WINDOW_LIST_LIMIT_DEFAULT}}},
        timeout_sec=15,
    )
    api.register_tool(
        "window_read",
        _json_tool(impl.window_read),
        description=(
            "Read a Windows window AS TEXT (roles, labels, values, states, rects) through UI "
            "Automation instead of an image; the answer grows with the nodes read. Default: "
            "foreground window, tree shape, 400 nodes. Limits come back in the answer with "
            "ceilings; every cut is named in truncated_by/notes. backend=win32 forces the "
            "classic reader (classic windows only)."
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
        # UIA and then Win32 each get up to 20 s plus 750 ms grace.
        # Leave headroom for starting the child and preparing comtypes as well.
        timeout_sec=50,
    )
    api.register_tool(
        "element_find",
        _json_tool(impl.element_find),
        description=(
            "Find elements in a window by selector (automation_id exact; role/name "
            "case-folded, Cyrillic included; all given fields must match). Returns every "
            "match up to limit; oversized displays name omissions and retain match counts. "
            "timeout_ms=0 (default) looks once; a positive value waits "
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
        _json_tool(impl.element_act),
        description=(
            "Act on the element a selector names, through its UIA pattern (no coordinate "
            "fallback): invoke | set_value | toggle | expand | collapse | select | "
            "scroll_into_view | focus. Two matches without select.nth, or a candidate whose "
            "selector field could not be read, is a refusal listing them. The receipt shows "
            "the element before/after (failed reads named); an in-call exception or a timed-out "
            "worker has unknown outcome, not a pre-call refusal. Whether the goal was achieved is "
            "yours to judge."
        ),
        schema={
            "type": "object",
            "properties": {
                "hwnd": {"type": ["integer", "string"]},
                "select": _ACT_SELECT_SCHEMA,
                "do": {
                    "type": "string",
                    "enum": ["invoke", "set_value", "toggle", "expand", "collapse", "select", "scroll_into_view", "focus"],
                },
                "text": {"type": "string", "description": "text for set_value, required as a string (empty string clears; null is refused)"},
                "timeout_ms": {"type": "integer", "minimum": 0, "maximum": 60000},
                "max_nodes": {"type": "integer", "minimum": 1},
                "max_depth": {"type": "integer", "minimum": 0},
            },
            "required": ["do", "select"],
            "additionalProperties": False,
        },
        timeout_sec=70,
    )
