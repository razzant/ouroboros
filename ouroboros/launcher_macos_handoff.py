"""Leave native fullscreen before replacing the desktop shell's process image.

A hidden window still owns its Space. AppKit clears the fullscreen style bit
before the exit finishes; wait for completion, then suppress re-entry until exec
or failure recovery. This does not certify retirement of WindowServer's private
Space record: a transient record may outlive DidExit. No sleeps or private Space
APIs decide production readiness. The lifecycle worker waits, never the UI or
Panic thread. Refusal keeps the shell alive; an already-issued exit may finish
later and is not undone. Destroying the last pywebview window would quit its loop.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import logging
import threading
import time
from typing import Any, Callable

from ouroboros import config
from ouroboros.platform_layer import IS_MACOS

log = logging.getLogger("launcher.handoff")


@lru_cache(maxsize=1)
def _failure_delegate_type():
    """AppKit failure callbacks are delegate methods, not notifications.

    Forward every other selector to pywebview's original delegate rather than
    copying its implementation or modifying its class for unrelated windows.
    """
    from AppKit import NSObject
    from objc import super as objc_super

    class OuroborosHandoffDelegate(NSObject):
        def respondsToSelector_(self, selector):
            return (objc_super(OuroborosHandoffDelegate, self).respondsToSelector_(selector)
                    or self.previous.respondsToSelector_(selector))

        def forwardingTargetForSelector_(self, selector):
            return self.previous

        def windowDidFailToEnterFullScreen_(self, window):
            self.owner._failed("windowed")
            if self.previous.respondsToSelector_("windowDidFailToEnterFullScreen:"):
                self.previous.windowDidFailToEnterFullScreen_(window)

        def windowDidFailToExitFullScreen_(self, window):
            self.owner._failed("fullscreen")
            if self.previous.respondsToSelector_("windowDidFailToExitFullScreen:"):
                self.previous.windowDidFailToExitFullScreen_(window)

    return OuroborosHandoffDelegate


@dataclass
class _ExitRequest:
    cancelled: Callable[[], bool]
    deadline: float
    done: threading.Event = field(default_factory=threading.Event)
    ok: bool = False
    requested: bool = False
    was_hidden: bool = False


class MacOSWindowHandoff:
    """One window's native transition and temporary handoff barrier."""

    def __init__(self) -> None:
        self._window: Any = None
        self._native: Any = None
        self._center: Any = None
        self._delegate: Any = None
        self._observers: list = []
        self._phase = "unbound"
        self._pending: _ExitRequest | None = None
        self._barrier: tuple[_ExitRequest, int] | None = None
        self._lock = threading.Lock()  # request identity only; Panic never acquires it

    def attach(self, window: Any) -> None:
        if IS_MACOS and window is not None:
            self._window = window
            window.events.before_show += self._bind
            window.events.closed += self._close

    def _bind(self) -> None:
        """before_show is synchronous on AppKit's main thread, independent of chrome."""
        try:
            import AppKit

            self._native = self._window.native
            if self._native is None:
                return
            delegate = _failure_delegate_type().alloc().init()
            delegate.previous, delegate.owner = self._native.delegate(), self
            self._delegate = delegate
            self._native.setDelegate_(delegate)
            self._center = AppKit.NSNotificationCenter.defaultCenter()
            for name, phase in (("WillEnter", "entering"), ("DidEnter", "fullscreen"),
                                ("WillExit", "exiting"), ("DidExit", "windowed")):
                self._observers.append(self._center.addObserverForName_object_queue_usingBlock_(
                    f"NSWindow{name}FullScreenNotification", self._native, None,
                    lambda _note, phase=phase: self._transition(phase)))
            self._phase = "fullscreen" if self._is_fullscreen() else "windowed"
        except Exception:
            self._phase = "unbound"
            log.warning("Native window observation unavailable; source handoff deferred until a cold start.", exc_info=True)

    def _is_fullscreen(self) -> bool:
        from AppKit import NSWindowStyleMaskFullScreen

        return bool(self._native.styleMask() & NSWindowStyleMaskFullScreen)

    def _transition(self, phase: str) -> None:
        self._phase = phase
        self._advance()

    def _failed(self, phase: str) -> None:
        """An explicit AppKit failure ends this attempt, but not future handoffs."""
        self._phase = phase
        with self._lock:
            if self._pending is not None:
                self._pending.done.set()

    def _advance(self) -> None:
        """Main-thread state recheck, action, and re-entry barrier are indivisible to UI input."""
        with self._lock:
            request = self._pending
            if request is None or request.done.is_set():
                return
            if request.cancelled() or time.monotonic() >= request.deadline or self._phase in ("unbound", "closed"):
                request.done.set()
                return
            if self._phase in ("entering", "exiting"):
                return
        try:
            fullscreen = self._is_fullscreen()
            with self._lock:
                if self._pending is not request or request.done.is_set() or request.cancelled() \
                        or time.monotonic() >= request.deadline:
                    return
                if not fullscreen:
                    from AppKit import (NSWindowCollectionBehaviorFullScreenNone,
                                        NSWindowCollectionBehaviorFullScreenPrimary,
                                        NSWindowCollectionBehaviorFullScreenAuxiliary)

                    if request.was_hidden:
                        self._native.orderOut_(None)  # AppKit may show an ordered-out window while exiting.
                    previous = int(self._native.collectionBehavior())
                    self._native.setCollectionBehavior_((previous & ~(
                        NSWindowCollectionBehaviorFullScreenPrimary | NSWindowCollectionBehaviorFullScreenAuxiliary))
                        | NSWindowCollectionBehaviorFullScreenNone)
                    self._barrier = (request, previous)
                    request.ok = True
                    request.done.set()
                    return
                if request.requested:
                    return
                request.requested = True  # an issued exit cannot be undone by cancellation
                request.was_hidden = not self._native.isVisible()
            self._native.toggleFullScreen_(None)
        except Exception:
            log.warning("Native fullscreen preparation failed; keeping the current shell.", exc_info=True)
            request.done.set()

    def _restore(self, request: _ExitRequest) -> None:
        with self._lock:
            if self._barrier is None or self._barrier[0] is not request:
                return
            _, previous = self._barrier
            self._native.setCollectionBehavior_(previous)
            self._barrier = None

    def release(self) -> None:
        """Restore normal controls after a refused/failed handoff, on the UI thread."""
        with self._lock:
            request = self._barrier[0] if self._barrier else None
        if request is not None:
            from PyObjCTools import AppHelper

            AppHelper.callAfter(self._restore, request)

    def prepare(self, cancelled: Callable[[], bool]) -> bool:
        """Called only by the lifecycle worker. No window means cold/headless startup."""
        if not IS_MACOS or self._window is None:
            return not cancelled()
        if threading.current_thread() is threading.main_thread():
            log.warning("Source handoff cannot wait for fullscreen on the UI thread; keeping this shell.")
            return False
        request = _ExitRequest(cancelled, time.monotonic() + config.MACOS_HANDOFF_FULLSCREEN_WAIT_SEC)
        with self._lock:
            if self._pending is not None or self._barrier is not None:
                return False
            self._pending = request
        ok = False
        try:
            from PyObjCTools import AppHelper

            AppHelper.callAfter(self._advance)
            while not request.done.is_set() and not cancelled():
                remaining = request.deadline - time.monotonic()
                if remaining <= 0:
                    break
                request.done.wait(min(remaining, 0.05))
            with self._lock:
                ok = request.done.is_set() and request.ok and not cancelled() and time.monotonic() < request.deadline
            if not ok:
                log.warning("Native fullscreen exit not confirmed; source handoff deferred in the current shell.")
            return ok
        except Exception:
            log.warning("Native handoff preparation unavailable; keeping this shell.", exc_info=True)
            return False
        finally:
            with self._lock:
                request.done.set()
                if self._pending is request:
                    self._pending = None
            if not ok:
                self.release()

    def before_release(self, cancelled: Callable[[], bool], cleanup: Callable[[], Any]) -> bool:
        if not self.prepare(cancelled):
            return False
        try:
            cleanup()
            if cancelled():
                self.release()
                return False
            return True
        except Exception:
            self.release()
            raise

    def _close(self) -> None:
        self._phase = "closed"
        with self._lock:
            if self._pending is not None:
                self._pending.done.set()
        from PyObjCTools import AppHelper

        AppHelper.callAfter(self._detach)

    def _detach(self) -> None:
        if self._native is not None and self._delegate is not None and self._native.delegate() == self._delegate:
            self._native.setDelegate_(self._delegate.previous)
        observers, self._observers = self._observers, []
        if self._center is not None:
            for token in observers:
                self._center.removeObserver_(token)
        self._delegate = None
