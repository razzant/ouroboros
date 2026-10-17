"""The GLib main loop the Zoom Meeting SDK needs, without requiring PyGObject.

The Linux Meeting SDK dispatches its callbacks through GLib's default main context,
and that loop has to run on the process's main thread: Hope found that running it
in a thread gives empty audio buffers. Ouroboros installs often have no ``gi``
(a venv without system site packages), so this module drives libglib directly
through ``ctypes`` and uses PyGObject only when it happens to be importable. Both
loops attach to the default context, the same one the SDK uses.

Every scheduled Python callable stays referenced until GLib is done with it. The
ctypes loop owns one C trampoline for its whole lifetime and hands GLib a key, not a
fresh C function per callback, so no C function pointer can be collected while GLib
may still call it.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import itertools
import logging
import os
import threading
from typing import Callable, Dict, List, Optional

log = logging.getLogger("zoom_bot.glib")

Callback = Callable[[], bool]
_GSourceFunc = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p)
_LIBRARY_CANDIDATES = (
    "libglib-2.0.so.0",
    "/usr/lib/x86_64-linux-gnu/libglib-2.0.so.0",
    "/usr/lib64/libglib-2.0.so.0",
    "/opt/homebrew/lib/libglib-2.0.0.dylib",
    "/usr/local/lib/libglib-2.0.0.dylib",
    "/opt/local/lib/libglib-2.0.0.dylib",
)


class GLibUnavailable(RuntimeError):
    """Neither PyGObject nor a loadable libglib-2.0 is present."""


def find_libglib(candidates: Optional[List[str]] = None) -> Optional[str]:
    """The first libglib-2.0 that loads: the linker's answer, then the usual install paths."""
    names: List[str] = []
    found = ctypes.util.find_library("glib-2.0")
    if found:
        names.append(found)
    names.extend(candidates if candidates is not None else _LIBRARY_CANDIDATES)
    for name in names:
        if os.path.isabs(name) and not os.path.exists(name):
            continue
        try:
            ctypes.CDLL(name)
        except OSError:
            continue
        return name
    return None


def _call_safely(fn: Callback) -> bool:
    try:
        return bool(fn())
    except Exception:
        log.exception("GLib callback failed; it is removed")
        return False


class CtypesGLibLoop:
    """``g_main_loop_*``, ``g_idle_add`` and ``g_timeout_add`` through ctypes.

    ``ctypes.CDLL`` releases the GIL for the duration of ``g_main_loop_run``; GLib
    re-enters Python only through the trampoline, which takes the GIL back.
    ``idle_add``/``timeout_add``/``quit`` are safe from any thread, as in GLib.
    """

    backend = "ctypes"

    def __init__(self, library: str) -> None:
        lib = ctypes.CDLL(library)
        lib.g_main_loop_new.restype = ctypes.c_void_p
        lib.g_main_loop_new.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.g_main_loop_run.restype = None
        lib.g_main_loop_run.argtypes = [ctypes.c_void_p]
        lib.g_main_loop_quit.restype = None
        lib.g_main_loop_quit.argtypes = [ctypes.c_void_p]
        lib.g_main_loop_is_running.restype = ctypes.c_int
        lib.g_main_loop_is_running.argtypes = [ctypes.c_void_p]
        lib.g_idle_add.restype = ctypes.c_uint
        lib.g_idle_add.argtypes = [_GSourceFunc, ctypes.c_void_p]
        lib.g_timeout_add.restype = ctypes.c_uint
        lib.g_timeout_add.argtypes = [ctypes.c_uint, _GSourceFunc, ctypes.c_void_p]
        self.library = library
        self._lib = lib
        self._loop = lib.g_main_loop_new(None, 0)
        if not self._loop:
            raise GLibUnavailable(f"g_main_loop_new failed in {library}")
        self._lock = threading.Lock()
        self._callbacks: Dict[int, Callback] = {}
        self._keys = itertools.count(1)
        self._trampoline = _GSourceFunc(self._dispatch)

    def _dispatch(self, user_data: Optional[int]) -> int:
        key = int(user_data or 0)
        with self._lock:
            fn = self._callbacks.get(key)
        if fn is None:
            return 0
        keep = _call_safely(fn)
        if not keep:
            with self._lock:
                self._callbacks.pop(key, None)
        return 1 if keep else 0

    def _register(self, fn: Callback) -> int:
        key = next(self._keys)
        with self._lock:
            self._callbacks[key] = fn
        return key

    def idle_add(self, fn: Callback) -> int:
        key = self._register(fn)
        self._lib.g_idle_add(self._trampoline, ctypes.c_void_p(key))
        return key

    def timeout_add(self, interval_ms: int, fn: Callback) -> int:
        key = self._register(fn)
        self._lib.g_timeout_add(max(0, int(interval_ms)), self._trampoline, ctypes.c_void_p(key))
        return key

    def pending(self) -> int:
        with self._lock:
            return len(self._callbacks)

    def run(self) -> None:
        self._lib.g_main_loop_run(self._loop)

    def quit(self) -> None:
        self._lib.g_main_loop_quit(self._loop)

    def is_running(self) -> bool:
        return bool(self._lib.g_main_loop_is_running(self._loop))


class GiGLibLoop:
    """The same interface over PyGObject's ``GLib``."""

    backend = "gi"

    def __init__(self, glib_module) -> None:
        self._glib = glib_module
        self._loop = glib_module.MainLoop()
        self.library = "gi.repository.GLib"

    def idle_add(self, fn: Callback) -> int:
        return int(self._glib.idle_add(lambda: _call_safely(fn)))

    def timeout_add(self, interval_ms: int, fn: Callback) -> int:
        return int(self._glib.timeout_add(max(0, int(interval_ms)), lambda: _call_safely(fn)))

    def run(self) -> None:
        self._loop.run()

    def quit(self) -> None:
        self._loop.quit()

    def is_running(self) -> bool:
        return bool(self._loop.is_running())


def make_loop(*, prefer_gi: bool = True, library: Optional[str] = None):
    """PyGObject's loop when importable, else libglib through ctypes."""
    if prefer_gi and library is None:
        try:
            import gi  # type: ignore[import-not-found]

            gi.require_version("GLib", "2.0")
            from gi.repository import GLib  # type: ignore[import-not-found]

            return GiGLibLoop(GLib)
        except (ImportError, ValueError):
            pass
    path = library or find_libglib()
    if not path:
        raise GLibUnavailable("libglib-2.0 was not found (the Zoom Meeting SDK needs GLib)")
    try:
        return CtypesGLibLoop(path)
    except (OSError, AttributeError) as exc:
        raise GLibUnavailable(f"libglib-2.0 at {path} is unusable ({type(exc).__name__})") from exc
