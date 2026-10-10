"""Desktop background mode: what closing the window does, the one consent question, the way back.

The owner's choice is ``OUROBOROS_DESKTOP_KEEP_RUNNING`` (``desktop_autostart``), read
when the window closes. A close hides the window only while a platform indicator is
live (``launcher_tray`` on Windows, ``launcher_tray_macos`` on macOS); otherwise it quits
as before, and a window hidden on purpose is shown again if its indicator goes away.
Until the owner decides, the first close asks once and stores the answer; dismissing
the question quits. A quit request (the indicator's Quit, Cmd+Q, sign-out, shutdown)
and Panic never reach the question and are never cancelled. An alert never raises a
window hidden on purpose. Linux and other hosts have no indicator: closing quits.

Second launch: a manual one shows the running window (``activate_running_instance`` -> ``Background.listen``),
an automatic one only logs. Windows signals a named auto-reset kernel event derived from the PID-lock
path; elsewhere the loser sends SIGURG to the PID in the lock file (an older launcher without a handler
ignores it). A lock loser never truncates the lock file. The listener starts right after the lock is
taken; a request that arrives before pywebview has shown the window is kept until then and cancels a
quiet start.

``DesktopApi`` is the page's alert half of the window bridge (``launcher.MainApi`` inherits it): the
attention cue, the shell's own facts, and the system notifications of ``desktop_notifications``,
whose click opens the window and hands the page its token; it also carries the painted palette to
the window's native caption (``launcher_appearance``).
"""
from __future__ import annotations

import json
import logging
import os
import pathlib
import signal
import sys
import threading
import urllib.request

from ouroboros.config import read_version
from ouroboros.desktop_autostart import BACKGROUND_ENV, keep_running_choice, set_keep_running
from ouroboros.desktop_notifications import UNAVAILABLE, capability, native_notifier, platform_name, refused
from ouroboros.launcher_appearance import NativeAppearance
from ouroboros.platform_layer import request_native_attention, signal_pid

log = logging.getLogger("launcher.background")

CONSENT_TITLE = "Keep Ouroboros running?"
CONSENT_MESSAGE = ("Ouroboros will keep working in the background: tasks, schedules, Telegram. "
                   "Keep it running in the background, or quit?")
INDICATOR_WAIT_SEC = 3.0
# Every first-party ``webview.start`` passes ``private_mode=False`` (launcher.py, launcher_onboarding.py):
# pywebview's private mode erases the WebView's website data each time a window opens (ARCHITECTURE §3).
PERSISTENT_WEBVIEW_STORAGE = True
STATUS_POLL_SEC = 15.0
_active = None  # the started indicator, for the exit and Panic paths
_notifier = None  # the system-notification adapter, for the same paths (Windows' icons outlive a process)


def indicator_class():
    """This platform's indicator, or None: closing the window quits, as before."""
    if sys.platform == "win32":
        from ouroboros.launcher_tray import WindowsTray
        return WindowsTray
    if sys.platform == "darwin":
        from ouroboros.launcher_tray_macos import MacStatusItem
        return MacStatusItem
    return None


def background_env(presentation: str) -> dict:
    """The server's ``BACKGROUND_ENV``: "1" where this launcher can hide its window and still be reached."""
    return {BACKGROUND_ENV: "1" if presentation == "desktop_window" and sys.platform in ("win32", "darwin") else ""}


WORKING_PHASES = ("thinking", "working", "finalizing")  # the page's own set (web/modules/project_activity.js)


def status_line(port: int) -> str:
    """The indicator's state line from the server's own ``/api/state``: work only from confirmed phases,
    and an unknown state said as unknown (not ready, an error, an incomplete census, an unknown phase)."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{int(port)}/api/state", timeout=5) as response:
            state = json.loads(response.read().decode("utf-8"))
        rows = state["active_chat_activities"]
        phases = [str(row.get("phase") or "") for row in rows]
    except Exception:
        return "Ouroboros: not responding"
    if state.get("supervisor_error"):
        return "Ouroboros: error, tasks are not running"
    if state.get("supervisor_ready") is not True:
        return "Ouroboros: starting"
    if state.get("active_chat_activities_complete") is not True or not set(phases) <= {
            *WORKING_PHASES, "queued", "budget_pausing", "budget_paused"}:
        return "Ouroboros: activity unconfirmed"
    working = sum(phase in WORKING_PHASES for phase in phases)
    if working:
        return f"Ouroboros: working on {working} task{'' if working == 1 else 's'}"
    if "budget_pausing" in phases:  # settling its Pause: neither working nor paused yet, as on the page
        return "Ouroboros: pausing"
    return "Ouroboros: paused" if "budget_paused" in phases else "Ouroboros: waiting"


def signin_startup_on(port: int) -> bool:
    """Sign-in startup as the OS reports it (``desktop_autostart.autostart_status``), asked of the server:
    the packaged-launcher facts it needs live in the server's environment, not in this process."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{int(port)}/api/desktop/autostart", timeout=10) as response:
            return json.loads(response.read().decode("utf-8")).get("state") == "on"
    except Exception:
        log.warning("Sign-in startup state unreadable; the window is shown.", exc_info=True)
        return False


def activate_running_instance(lock_path) -> bool:
    """A manual second launch asks the running launcher to show its window; False: nobody answered."""
    if sys.platform == "win32":
        from ouroboros.launcher_tray import activate_existing_tray
        return activate_existing_tray(lock_path)
    try:
        pid = int(pathlib.Path(lock_path).read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return False
    # SIGURG is ignored by default, so an older launcher without a handler survives it.
    return pid > 0 and pid != os.getpid() and signal_pid(pid, signal.SIGURG)


def request_tray_cleanup() -> None:
    """Panic: start removing the icons without waiting for them."""
    for owner in (_active, _notifier):
        if owner is not None:
            try:
                owner.stop(wait=0)
            except Exception:
                log.warning("Icon cleanup request failed; Panic continues.", exc_info=True)


def stop_tray_before_exit(release_lock, *, wait: float = 0.5) -> None:
    """Remove the icons before an ordinary exit (bounded each); Panic passes ``wait=0``."""
    for owner in (_active, _notifier):
        if owner is not None:
            try:
                owner.stop(wait=wait)
            except Exception:
                log.warning("Icon cleanup failed before process exit.", exc_info=True)
    release_lock()


class Indicator:
    """Common half of a platform indicator: ``start → bool``, ``hide_on_close``,
    ``show_if_unavailable``, ``stop(wait)``. The window hides only while the icon is
    live, and a window hidden on purpose comes back if the icon goes away."""

    decides_natively = False  # Windows: the form's FormClosing handler sees the close reason

    def __init__(self, background: "Background") -> None:
        self.background = background
        self.ready = threading.Event()  # the icon is visible
        self._stop = threading.Event()
        self._idle = threading.Event()  # no icon and no pump: start() may create one
        self._idle.set()
        self._hidden = False
        self._recovering = False
        self._state_lock = threading.Lock()

    @property
    def hidden(self) -> bool:
        return self._hidden

    def begin_hidden(self) -> None:
        self._hidden = True  # quiet start: the window is created hidden

    def start(self) -> bool:
        """Create the icon once; ``ready`` is set when it is visible."""
        global _active
        with self._state_lock:
            if not self._idle.is_set():
                return not self._stop.is_set()
            self._idle.clear()
            self._stop.clear()
        try:
            started = bool(self._launch())
        except Exception:
            log.warning("Background indicator could not start; closing the window quits.", exc_info=True)
            started = False
        if started:
            _active = self
        else:
            self._idle.set()
        return started

    def hide_on_close(self, window) -> bool:
        """Hide for a close; False (the close quits) when there is no live icon to come back through.

        ``hide()`` runs outside the state lock: on Windows it is a synchronous Invoke onto the UI
        thread, which may itself be waiting for this lock in a second close. The hidden state is
        committed under the lock only if the icon is still live; otherwise the close quits."""
        if not self.ready.is_set() or self.background.shutdown.is_set():
            return False
        try:
            window.hide()
        except Exception:
            log.warning("Could not hide the window; closing it quits.", exc_info=True)
            return False
        with self._state_lock:
            if not self.ready.is_set() or self.background.shutdown.is_set():
                return False  # the icon went away while hiding: never a hidden window without it
            self._hidden = True
            return True

    def show_if_unavailable(self, window) -> None:
        """Quiet start: the window stays hidden only if the icon actually came up."""
        if not self.ready.wait(INDICATOR_WAIT_SEC) or self._stop.is_set() or self.background.shutdown.is_set():
            self._show_or_exit(window)

    def restore(self, window) -> None:
        try:
            window.show()
        except Exception:
            log.warning("Could not show the window; the request can be repeated.", exc_info=True)
            return
        with self._state_lock:
            self._hidden = False
        self._shown()

    def stop(self, *, wait: float = 0.0) -> None:
        self.ready.clear()
        self._stop.set()
        self._dispose(wait)

    def _show_or_exit(self, window) -> None:
        if self.background.shutdown.is_set():
            return
        with self._state_lock:
            if not self._hidden or self._recovering:
                return  # another recovery already made the window visible
            self._recovering = True
        log.warning("Background indicator unavailable; showing the window.")
        try:
            for attempt in range(2):
                try:
                    window.show()
                except Exception:
                    log.error("Could not show the window.", exc_info=True)
                    if attempt == 0 and not self.background.shutdown.wait(0.1):
                        continue
                    if not self.background.shutdown.is_set():
                        self.background.exit_launcher()  # no usable UI: never a hidden process without a way back
                else:
                    with self._state_lock:
                        self._hidden = False
                break
        finally:
            with self._state_lock:
                self._recovering = False

    def _stopped(self) -> None:
        """The icon is gone; a window hidden on purpose must come back unless this is a stop."""
        with self._state_lock:
            self.ready.clear()
            must_restore = self._hidden and not self._stop.is_set() and not self.background.shutdown.is_set()
            self._idle.set()
        if must_restore:
            if self.background.window is None:
                self.background.exit_launcher()
            else:
                self._show_or_exit(self.background.window)

    # Platform half.
    def attach_native(self, window) -> None:
        raise NotImplementedError

    def _launch(self) -> bool:
        raise NotImplementedError

    def _dispose(self, wait: float) -> None:
        pass

    def _shown(self) -> None:
        pass

    def notify(self, title: str, body: str, sound: bool = True) -> bool:
        """Show a native banner for an alert; False when there is certainly none (the caller plays the sound)."""
        return False

    def set_status(self, text: str) -> None:
        pass

    def confirm(self, window, title: str, message: str) -> bool:
        return bool(window.create_confirmation_dialog(title, message))


class Background:
    """The desktop window's close policy, its indicator (None where there is none) and the way back."""

    def __init__(self, exit_launcher, read_port, shutdown_event) -> None:
        cls = indicator_class()
        self.exit_launcher = exit_launcher  # stop the agent, clean up, os._exit
        self.read_port = read_port
        self.shutdown = shutdown_event
        self.quit = threading.Event()  # a quit request: never asked, never cancelled
        self.window = None
        self.native_ready = False
        self.indicator = cls(self) if cls is not None else None
        self.appearance = NativeAppearance()  # the window's caption tint follows its page (Windows)
        global _notifier
        self.notifications = _notifier = native_notifier(self.open_notification)
        self._asking = threading.Lock()
        self._poller = None
        # An open request (a second launch) can arrive during the boot, before the window exists: it is
        # kept until pywebview's ``shown`` and cancels a quiet start. Re-entrant: a POSIX signal handler
        # runs on the thread that may hold it.
        self._opening = threading.RLock()
        self._open_pending = False
        self._can_show = False

    def listen(self, lock_path) -> "Background":
        """A manual second launch shows this window: a kernel event on Windows, SIGURG elsewhere.

        Installed right after the single-instance lock, before the boot; a request that comes
        before the window can be shown is kept (``show_window``)."""
        try:
            if sys.platform == "win32":
                from ouroboros.launcher_tray import listen_for_activation
                listen_for_activation(lock_path, self.show_window, self.shutdown)
            elif sys.platform == "darwin":
                from PyObjCTools import MachSignals  # wakes the run loop; a Python handler waits for bytecode
                MachSignals.signal(signal.SIGURG, lambda *_: self.show_window())
            else:
                signal.signal(signal.SIGURG, lambda *_: self.show_window())
        except Exception:
            log.warning("A second launch cannot show this window on this host.", exc_info=True)
        return self

    def start_hidden(self, intent: str) -> bool:
        """Quiet start (D3): an automatic launch starts hidden only with both checkboxes on, sign-in startup as
        the OS reports it and background; ``run`` shows it if no icon comes up. Any other state, a failed
        read, or a manual launch that asked for the window during the boot shows it."""
        with self._opening:
            if self._open_pending:
                return False
        hidden = (self.indicator is not None and intent == "automatic" and keep_running_choice() == "true"
                  and signin_startup_on(self.read_port()))
        if hidden:
            self.indicator.begin_hidden()
        return hidden

    def attach(self, window):
        self.window = window
        self.appearance.attach(window)  # every window, indicator or not: its own before_show and close
        window.events.closing += self.closing
        window.events.shown += self._window_shown  # set for a hidden window too (pywebview 5.4)
        if self.indicator is not None:
            window.events.before_show += self._attach_native
        return window

    def run(self) -> None:
        """``webview.start(func=...)``: bring the icon up when background is on."""
        if self.indicator is None or (keep_running_choice() != "true" and not self.indicator.hidden):
            return
        self._indicator_up()  # the state poller starts only once the icon is visible
        if self.indicator.hidden:
            self.indicator.show_if_unavailable(self.window)
        # Readiness can arrive during that second wait, or before _indicator_up.
        # Every live indicator needs the poller, whichever observation saw it.
        if self.indicator.ready.is_set():
            self._watch()

    def closing(self):
        """pywebview ``closing``; False keeps the process, every other outcome quits."""
        if self.indicator is None or not self.native_ready or self.quit.is_set():
            return self.exit_launcher()
        if self.indicator.decides_natively:
            return None  # Windows: its FormClosing handler holds the close reason
        return self.window_closed()

    def window_closed(self):
        """The owner closed the window (never a quit request). False cancels the close."""
        choice = keep_running_choice()
        if choice == "true" and self._indicator_up() and self.indicator.hide_on_close(self.window):
            return False
        if choice == "":
            if self._asking.acquire(blocking=False):
                threading.Thread(target=self._ask, name="ouroboros-background-consent", daemon=True).start()
            return False
        return self.exit_launcher()

    def request_quit(self) -> None:
        """The indicator's Quit: a plain exit; no Panic flag, sign-in startup untouched."""
        self.quit.set()
        self.exit_launcher()

    def request_panic(self) -> None:
        """The indicator's Panic: the server's own emergency stop over HTTP, like Android's notification."""
        def post() -> None:
            try:
                request = urllib.request.Request(
                    f"http://127.0.0.1:{self.read_port()}/api/command", method="POST",
                    data=json.dumps({"cmd": "/panic"}).encode("utf-8"), headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(request, timeout=10):
                    pass
            except Exception:
                log.error("Panic could not reach the server; showing the window.", exc_info=True)
                self.show_window()

        threading.Thread(target=post, name="ouroboros-indicator-panic", daemon=True).start()

    def show_window(self) -> None:
        """Open the window: the icon or its Open, a banner, the Dock, a second launch.

        Before pywebview has shown the window (hidden or not) the request is kept, never dropped."""
        with self._opening:
            if not self._can_show:
                self._open_pending = True
                return
        if self.indicator is not None:
            self.indicator.restore(self.window)
            return
        try:
            self.window.show()
        except Exception:
            log.warning("Could not show the window.", exc_info=True)

    def attention(self, sound: bool = True, title: str = "", body: str = "", cue_when_visible: bool = True) -> dict:
        """The bridge's alert cue (D7): never raises a window the owner hid on purpose. A page that
        will show its own browser banner asks first with ``cue_when_visible=False``: a visible
        window then gets nothing from here (that banner owns the sound) and answers "visible"."""
        if self.indicator is not None and self.indicator.hidden:
            banner = self.indicator.notify(title or "Ouroboros", body or "Something needs your attention.",
                                           bool(sound))
            cue = request_native_attention(None, sound=bool(sound) and not banner)
            return {"ok": bool(banner or cue.get("ok")), "status": "background", "banner": bool(banner),
                    "sound_played": bool(cue.get("sound_played"))}
        if not cue_when_visible:
            return {"ok": False, "status": "visible"}
        return request_native_attention(self.window.show if self.window is not None else None, sound=bool(sound))

    def open_notification(self, token: str) -> None:
        """A system notification was clicked: open the window as it was left, then hand the page the
        token it chose; the page maps it to the source (an unknown token after a reload only opens)."""
        self.show_window()
        if not token or self.window is None:
            return
        try:
            self.window.evaluate_js(f"window.ouroNotifications && window.ouroNotifications.activate({json.dumps(token)})")
        except Exception:
            log.warning("The page could not be told which notification was clicked; the window is open.", exc_info=True)

    def _window_shown(self) -> None:
        """pywebview ``shown``: from now on a request opens the window; one kept since the boot does so now."""
        with self._opening:
            self._can_show = True
            pending, self._open_pending = self._open_pending, False
        if pending:
            self.show_window()

    def _attach_native(self) -> None:
        try:
            self.indicator.attach_native(self.window)
            self.native_ready = True
        except Exception:
            log.warning("Window close hooks unavailable; closing the window quits.", exc_info=True)

    def _indicator_up(self) -> bool:
        if not self.indicator.ready.is_set():
            if not self.indicator.start() or not self.indicator.ready.wait(INDICATOR_WAIT_SEC):
                log.warning("Background indicator did not appear; closing the window quits.")
                return False
            self._watch()
        return True

    def _ask(self) -> None:
        try:
            try:
                keep = bool(self.indicator.confirm(self.window, CONSENT_TITLE, CONSENT_MESSAGE))
            except Exception:
                log.error("The background question could not be shown; quitting as before.", exc_info=True)
                self.exit_launcher()
                return
            try:
                set_keep_running(keep)
            except Exception:
                log.warning("Could not save the background choice; it applies to this close only.", exc_info=True)
            if self.quit.is_set():
                return
            if keep and self._indicator_up() and self.indicator.hide_on_close(self.window):
                return
            self.exit_launcher()
        finally:
            self._asking.release()

    def _watch(self) -> None:
        if self._poller is None or not self._poller.is_alive():
            self._poller = threading.Thread(target=self._poll, name="ouroboros-indicator-status", daemon=True)
            self._poller.start()

    def _poll(self) -> None:
        while self.indicator.ready.is_set() and not self.shutdown.is_set():
            if keep_running_choice() != "true" and not self.indicator.hidden:
                self.indicator.stop()  # the owner turned background off: no icon while the window is visible
                return
            self.indicator.set_status(status_line(self.read_port()))
            if self.shutdown.wait(STATUS_POLL_SEC):
                return


class DesktopApi:
    """The page's alert half of the window bridge. A page feature-detects each method per call: a launcher
    built before one of them simply lacks it (the page then falls back and says so, DESIGN §9)."""

    _background = None  # set by the inheriting MainApi; a leading underscore keeps it off the bridge

    def request_attention(self, sound: bool = True, title: str = "", body: str = "", cue_when_visible: bool = True) -> dict:
        return self._background.attention(bool(sound), str(title or ""), str(body or ""), bool(cue_when_visible))

    notify_owner = request_attention  # newer pages send the alert text; older launchers lack this name

    def set_native_appearance(self, theme: str = "", page: float | None = None, sequence: int | None = None) -> dict:
        """The page's painted palette for this window's native caption; the newest request wins."""
        return self._background.appearance.request(theme, page, sequence)

    def shell_info(self) -> dict:
        """What this desktop app is: its own version (not the core's), its storage and its notifications."""
        return {"shell_version": read_version(), "persistent_storage": PERSISTENT_WEBVIEW_STORAGE,
                "native_notifications": self._notifier_answer("status")}

    def request_native_notifications(self) -> dict:
        """The system's one permission question. The page asks it only from the owner's own gesture
        (switching notifications on, the Test button); a notification never does."""
        return self._notifier_answer("ask")

    def show_native_notification(self, title: str = "", body: str = "", sound: bool = True, token: str = "") -> dict:
        """One system notification: ``submitted`` (the system took it and owns its sound: no page tone),
        ``unknown`` (handed over, unanswered: it may still appear, so the page adds nothing) or a typed
        refusal on which the page falls back."""
        notifier = getattr(self._background, "notifications", None)
        if notifier is None:
            return refused(platform_name(), UNAVAILABLE, "no_platform_adapter")
        return notifier.notify(str(title or ""), str(body or ""), bool(sound), str(token or ""))

    def _notifier_answer(self, method: str) -> dict:
        notifier = getattr(self._background, "notifications", None)
        if notifier is None:
            return capability(platform_name(), UNAVAILABLE, "no_platform_adapter")
        return getattr(notifier, method)()
