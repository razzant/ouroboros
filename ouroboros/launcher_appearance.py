"""The desktop window's native caption follows the page's painted palette (Windows).

The page owns Light / Dark / System (``web/theme.js``): it resolves the palette
and tells its OWN window's bridge (``set_native_appearance``) at load, when the
bridge appears and whenever the painted palette changes. This leaf turns that
fact into ``DWMWA_USE_IMMERSIVE_DARK_MODE`` on that window's own handle, so the
ordinary system caption, its buttons, Snap and resizing stay native and only
their light or dark tint follows the app — the system tint, never the page's
exact colour. There is no setting and no native state: the page's stored choice
is the one authority, and each window keeps its own requests.

A window-lifecycle leaf, not a platform-layer wrapper (DEVELOPMENT "Platform
Abstraction Rule"): the handle exists from pywebview's ``before_show``, which
runs on the form's UI thread before the window is shown, so its write needs no
repaint; every later write is marshalled back onto that thread with
``BeginInvoke`` and repaints the shown frame; and the window's close releases
the one system-preference subscription. Bridge calls arrive on pywebview's
worker threads, possibly out of order, even across a reload, so each carries
when its document became the window's page and its count there: only a request
newer by that pair is kept, never the one that merely arrived last.

Until the page reports, the frame matches the window's own background colour,
the colour pywebview paints before the page loads (dark for the main window,
the default light for setup); a page whose palette differs corrects it once its
bridge is ready. High Contrast owns the caption colours: while it is on the
attribute is cleared explicitly, and a preference change, which the page does
not announce, re-applies the page's palette.

The intended caption is the app's palette whatever Windows' own mode: the
attribute is TRUE exactly when the page paints dark (High Contrast aside).
Microsoft's sources disagree on whether Windows honours that under a light
system mode: the attribute reference words TRUE as honoring dark mode "when the
dark mode system setting is enabled" (Windows 11 build 22000 on), while on
Microsoft Q&A (question 966330, 16 Aug 2022) a Microsoft engineer reproduced the
dark caption under a light system and left bug-or-documentation open. No native
Windows run has checked it here: dark-on-light, Windows 10 and the pixels are
unproven, not excluded. A refusal's HRESULT is logged once; an accepted call is
not proof of the pixels.
"""

from __future__ import annotations

import logging
import math
import sys
import threading
from typing import Any, Dict, Optional

log = logging.getLogger("launcher.appearance")

_WINDOWS = sys.platform == "win32"
THEMES = ("light", "dark")
_DWMWA_USE_IMMERSIVE_DARK_MODE = 20
_SPI_GETHIGHCONTRAST = 0x0042
_HCF_HIGHCONTRASTON = 0x00000001
# SetWindowPos: keep size, position, z-order and activation; recompute the frame
# (SWP_NOSIZE | SWP_NOMOVE | SWP_NOZORDER | SWP_NOACTIVATE | SWP_FRAMECHANGED).
_SWP_REFRESH_FRAME = 0x0001 | 0x0002 | 0x0004 | 0x0010 | 0x0020


def palette_of_background(color: Any) -> str:
    """``dark`` or ``light`` for a ``#rgb``/``#rrggbb`` window background; ``""`` when unreadable."""
    text = str(color or "").strip().lstrip("#")
    if len(text) == 3:
        text = "".join(digit * 2 for digit in text)
    if len(text) != 6:
        return ""
    try:
        red, green, blue = (int(text[index:index + 2], 16) for index in (0, 2, 4))
    except ValueError:
        return ""
    return "dark" if 0.2126 * red + 0.7152 * green + 0.0722 * blue < 128 else "light"


def _set_dark_frame(hwnd: int, dark: bool) -> int:
    """``DwmSetWindowAttribute(hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE, &BOOL, sizeof(BOOL))``: its HRESULT."""
    import ctypes
    from ctypes import wintypes

    # A private handle: pywebview's own helper sets argtypes on the shared windll function.
    call = ctypes.WinDLL("dwmapi").DwmSetWindowAttribute
    call.argtypes = (wintypes.HWND, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD)
    call.restype = ctypes.c_long  # HRESULT: negative is a refusal
    value = wintypes.BOOL(1 if dark else 0)
    return int(call(wintypes.HWND(hwnd), _DWMWA_USE_IMMERSIVE_DARK_MODE, ctypes.byref(value), ctypes.sizeof(value)))


def _high_contrast() -> bool:
    """Whether a Windows contrast theme is on (``SPI_GETHIGHCONTRAST``); an unreadable answer reads as off."""
    import ctypes
    from ctypes import wintypes

    class HighContrast(ctypes.Structure):
        _fields_ = [("cbSize", wintypes.UINT), ("dwFlags", wintypes.DWORD), ("lpszDefaultScheme", wintypes.LPWSTR)]

    call = ctypes.WinDLL("user32").SystemParametersInfoW
    call.argtypes = (wintypes.UINT, wintypes.UINT, ctypes.c_void_p, wintypes.UINT)
    call.restype = wintypes.BOOL
    info = HighContrast(cbSize=ctypes.sizeof(HighContrast))
    if not call(_SPI_GETHIGHCONTRAST, ctypes.sizeof(info), ctypes.byref(info), 0):
        log.debug("High Contrast state unreadable; the page's palette applies.")
        return False
    return bool(info.dwFlags & _HCF_HIGHCONTRASTON)


def _refresh_frame(hwnd: int) -> None:
    """Ask Windows to recompute the frame so a change on a visible window repaints its caption."""
    import ctypes
    from ctypes import wintypes

    call = ctypes.WinDLL("user32").SetWindowPos
    call.argtypes = (wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                     wintypes.UINT)
    call.restype = wintypes.BOOL
    call(wintypes.HWND(hwnd), None, 0, 0, 0, 0, _SWP_REFRESH_FRAME)


class NativeAppearance:
    """One desktop window's caption tint: the page's newest palette, written on its UI thread."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._window: Any = None
        self._form: Any = None
        self._hwnd = 0
        self._theme = ""  # the newest palette asked for ("" = none yet)
        self._order: tuple[float, int] = (-math.inf, 0)  # (its page's start, its count there)
        self._preferences: Any = None  # the subscribed .NET handler, released on close
        self._closed = False
        self._applied: Optional[bool] = None  # UI thread only; None = never written (the light default)
        self._refused = False

    def attach(self, window: Any) -> None:
        """Register on ``window`` before ``webview.start``; inert off Windows."""
        if not _WINDOWS or window is None:
            return
        self._window = window
        self._theme = palette_of_background(getattr(window, "background_color", ""))
        window.events.before_show += self._before_show
        window.events.closed += self._close

    def request(self, theme: Any, page: Any = None, sequence: Any = None) -> Dict[str, Any]:
        """The bridge call: keep the newest palette of the newest page, then post one write.

        ``page`` is when the calling document became the window's page (``theme.js``: its
        time origin, renewed when the back-forward cache restores it) and ``sequence`` its
        count of calls, so the order is the pages' own, not the order calls arrived in."""
        theme = str(theme or "")
        if theme not in THEMES:
            return {"ok": False, "state": "invalid_theme"}
        if not _WINDOWS:
            return {"ok": False, "state": "unsupported"}
        try:
            order = (float(page), int(sequence))
        except (TypeError, ValueError):
            return {"ok": False, "state": "invalid_order"}
        if not math.isfinite(order[0]):
            return {"ok": False, "state": "invalid_order"}
        with self._lock:
            if self._closed:
                return {"ok": False, "state": "closed"}
            if order <= self._order:
                return {"ok": True, "state": "stale"}
            self._order, self._theme, form = order, theme, self._form
        if form is None:
            return {"ok": True, "state": "pending"}  # before_show writes it
        return {"ok": True, "state": "scheduled" if self._post(form) else "unavailable"}

    def _before_show(self) -> None:
        """pywebview ``before_show`` (synchronous, the form's UI thread): take the handle, write the first tint."""
        form = getattr(self._window, "native", None)
        try:
            hwnd = int(form.Handle.ToInt64())
        except Exception:
            log.warning("No native window handle; the caption keeps the system tint.", exc_info=True)
            return
        with self._lock:
            if self._closed:
                return
            self._form, self._hwnd = form, hwnd
        self._subscribe_preferences()
        self._apply(shown=False)

    def _post(self, form: Any) -> bool:
        try:
            from System import Action

            form.BeginInvoke(Action(self._apply))
            return True
        except Exception:
            log.debug("Caption tint not scheduled: the window is closing or gone.", exc_info=True)
            return False

    def _apply(self, shown: bool = True) -> None:
        """The UI-thread write: the newest palette, cleared while High Contrast is on.

        Only ``before_show`` writes before the window is shown; every posted write runs
        after it, so a change there asks Windows to repaint the frame already on screen."""
        with self._lock:
            if self._closed or not self._hwnd or not self._theme:
                return
            hwnd, dark = self._hwnd, self._theme == "dark"
        dark = dark and not _high_contrast()
        if self._applied == dark or (self._applied is None and not dark):
            return  # unchanged, or the never-written default is already the light frame
        result = _set_dark_frame(hwnd, dark)
        if result < 0:
            if not self._refused:
                self._refused = True
                log.warning("DwmSetWindowAttribute refused the caption tint (HRESULT 0x%08X).", result & 0xFFFFFFFF)
            return
        self._applied = dark
        if shown:
            _refresh_frame(hwnd)

    def _subscribe_preferences(self) -> None:
        try:
            import clr

            clr.AddReference("System")
            from Microsoft.Win32 import SystemEvents, UserPreferenceChangedEventHandler

            handler = UserPreferenceChangedEventHandler(self._preference_changed)
            SystemEvents.UserPreferenceChanged += handler
        except Exception:
            log.info("High Contrast changes are not observed; the caption follows the page only.", exc_info=True)
            return
        with self._lock:
            self._preferences = handler

    def _preference_changed(self, _sender: Any = None, _args: Any = None) -> None:
        with self._lock:
            form = None if self._closed else self._form
        if form is not None:
            self._post(form)

    def _close(self) -> None:
        """pywebview ``closed``: no further writes, and the static system event lets go of this window."""
        with self._lock:
            self._closed = True
            self._form = None
            handler, self._preferences = self._preferences, None
        if handler is None:
            return
        try:
            from Microsoft.Win32 import SystemEvents

            SystemEvents.UserPreferenceChanged -= handler
        except Exception:
            log.debug("System preference subscription not released.", exc_info=True)
