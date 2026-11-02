"""Desktop background mode: what closing the window does, the one consent question, the way back.

The owner's choice is ``OUROBOROS_DESKTOP_KEEP_RUNNING`` (``desktop_autostart``), read
when the window closes. A close hides the window only while a platform indicator is
live (``launcher_tray`` on Windows, ``launcher_tray_macos`` on macOS); otherwise it quits
as before, and a window hidden on purpose is shown again if its indicator goes away.
Windows keeps its one icon for the whole desktop run, because system notifications are
balloons on it (owner decision 1A); macOS shows its item only while background is on.
Until the owner decides, the first close asks once and stores the answer; dismissing
the question quits. A quit request (the indicator's Quit, Cmd+Q, sign-out, shutdown)
and Panic never reach the question and are never cancelled. A quit latches shutdown at
once and runs the launcher's close body once, off the UI/STA thread that received it; the
window, its event loop and the indicator with its Panic stay until that body's terminal
exit, which an accepted Panic overrides (``exit_launcher``, ``leave``). An alert never raises a
window hidden on purpose. Linux and other hosts have no indicator: closing quits.

Second launch: a manual one shows the running window (``activate_running_instance`` -> ``Background.listen``),
an automatic one only logs. Windows signals a named auto-reset kernel event derived from the PID-lock
path; elsewhere the loser sends SIGURG to the PID in the lock file (an older launcher without a handler
ignores it). A lock loser never truncates the lock file. The listener starts right after the lock is
taken; a request that arrives before pywebview has shown the window is kept until then and cancels a
quiet start.

``DesktopApi`` is inherited by ``launcher.MainApi``: the independent Panic stop,
attention cue, shell facts, and system notifications of ``desktop_notifications``,
whose click opens the window and hands the page its token; it also carries the painted palette to
the window's native caption (``launcher_appearance``). On macOS the window's title bar and appearance
follow each document it shows (``launcher_window_chrome``). The owner's Interface size is the window's own page zoom
(``launcher_page_zoom``).
"""
from __future__ import annotations

import json
import logging
import os
import pathlib
import re
import signal
import sys
import threading
import urllib.request

from ouroboros.config import PANIC_EXIT_CODE, read_version
from ouroboros.desktop_autostart import BACKGROUND_ENV, keep_running_choice, set_keep_running
from ouroboros.desktop_notifications import UNAVAILABLE, capability, native_notifier, platform_name, refused
from ouroboros.launcher_appearance import NativeAppearance
from ouroboros.launcher_page_zoom import PageZoom
from ouroboros.launcher_window_chrome import WindowChrome
from ouroboros.platform_layer import request_native_attention, signal_pid

log = logging.getLogger("launcher.background")

CONSENT_TITLE = "Keep Ouroboros running?"
CONSENT_MESSAGE = ("Ouroboros will keep working in the background: tasks, schedules, Telegram. "
                   "Keep it running in the background, or quit?")
INDICATOR_WAIT_SEC = 3.0
# Every first-party ``webview.start`` passes ``private_mode=False`` (launcher.py, launcher_onboarding.py):
# pywebview's private mode erases the WebView's website data each time a window opens (ARCHITECTURE §3).
PERSISTENT_WEBVIEW_STORAGE = True
_SHELL_VERSION = read_version()  # the loaded code; a later in-place Update is not a new shell
STATUS_POLL_SEC = 15.0
_active = None  # the started indicator, for the exit and Panic paths (a Windows icon outlives its process)


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


TRAY_LINE_MAX = 63  # the Windows notify-icon tooltip limit (launcher_tray.py)


def status_line(port: int) -> str:
    """The indicator's state line from the server's own ``/api/state`` work projection: a work is working,
    pausing, paused or waiting as the server classified it, an incomplete projection is said as unknown, and
    what waits for the owner is counted (gateway/work_status.py)."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{int(port)}/api/state", timeout=5) as response:
            state = json.loads(response.read().decode("utf-8"))
        works = [work for work in state["works"] if isinstance(work, dict)]
        summary = state["work_summary"]
    except Exception:
        return "Ouroboros: not responding"
    if state.get("supervisor_error"):
        return "Ouroboros: error, tasks are not running"
    if state.get("supervisor_ready") is not True:
        return "Ouroboros: starting"
    if state.get("works_complete") is not True or not isinstance(summary, dict) or summary.get("confirmed") is not True \
            or any(work.get("activity") == "unknown" and work.get("confirmed") is not True for work in works):
        return "Ouroboros: activity unconfirmed"
    # A confirmed work whose only remainder is a run nobody can confirm is said beside the rest, never as all of it.
    unconfirmed = sum(work.get("activity") == "unknown" for work in works)
    known = [work for work in works if work.get("activity") != "unknown"]
    # The server's order (working, queued, waiting, paused), with the room's own transitions.
    activities = {work.get("activity") for work in known}
    working = [work for work in known if work.get("activity") == "working"]
    transitions = {work.get("transition") or "none" for work in working}
    moving = [work for work in working if work.get("transition") != "pausing"]
    if working and transitions == {"finalizing"}:
        line = "Ouroboros: finalizing"  # only answers' upkeep runs, as the page says
    elif working and "none" not in transitions and "pausing" in transitions:
        line = "Ouroboros: pausing"  # settling its Pause: neither working nor paused yet, as on the page
    elif moving:
        line = f"Ouroboros: working on {len(moving)} task{'' if len(moving) == 1 else 's'}"
    elif working:
        line = "Ouroboros: pausing"
    elif "queued" in activities:
        line = "Ouroboros: queued"
    elif "paused" in activities and "waiting" not in activities:
        line = "Ouroboros: paused"
    elif unconfirmed and not known:
        line = f"Ouroboros: {unconfirmed} task{'' if unconfirmed == 1 else 's'} unconfirmed"
    else:
        line = "Ouroboros: waiting"
    waits = sum(bool(work.get("turn")) for work in works)
    if waits:
        line += f" · {waits} {'waits' if waits == 1 else 'wait'} for you"
    if unconfirmed and known:  # after what waits for the owner, which a cut line must keep
        line += f" · {unconfirmed} unconfirmed"
    return line if len(line) <= TRAY_LINE_MAX else line[:TRAY_LINE_MAX - 1] + "…"


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


def other_copy_line(lock_path, port_file, bundle_dir) -> str:
    """The one line a manual second launch of a packaged copy shows first when the running instance is another
    copy, from that instance's own report: its command line names its copy (the text after its last
    ``--seed-bundle``, which the source handoff appends, else its executable) and its ``/api/health`` its
    ``app_version``. ``""`` when this copy's path occurs in that command line as a whole path (at its start or
    after a space or a quote, so ``~/Applications/Ouroboros.app`` is not ``/Applications/Ouroboros.app``), for an
    unreadable instance, a source run (no bundle manifest here), or this app's own AppImage, whose per-launch
    mount names no copy. ``ps`` joins argv without quoting, so the command line is never split into arguments."""
    from ouroboros.desktop_autostart import app_copy, own_appimage
    from ouroboros.platform_layer import process_command

    bundle = pathlib.Path(bundle_dir)
    if own_appimage(bundle) or not (bundle / "repo_bundle_manifest.json").is_file():
        return ""
    try:
        command = process_command(int(pathlib.Path(lock_path).read_text(encoding="utf-8").strip()))
    except (OSError, ValueError):
        return ""
    mine = app_copy(bundle)
    if not command or any(re.search(r'(?:^|[\s"])' + re.escape(os.path.normcase(path) + os.sep), os.path.normcase(command))
                          for path in (str(mine), os.path.realpath(mine))):
        return ""
    _, seeded, tail = command.rpartition(" --seed-bundle ")
    theirs = app_copy(pathlib.Path((tail if seeded else command.split(" --", 1)[0]).strip().strip('"')))
    try:
        port = int(pathlib.Path(port_file).read_text(encoding="utf-8").strip())
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=2) as response:
            version = str(json.loads(response.read().decode("utf-8")).get("app_version") or "") + " "
    except Exception:
        version = ""
    return f"Ouroboros {version}is already running from {theirs}. Quit it, then open this copy."


def request_tray_cleanup() -> None:
    """Panic: start removing the icon without waiting for it."""
    if _active is not None:
        try:
            _active.stop(wait=0)
        except Exception:
            log.warning("Icon cleanup request failed; Panic continues.", exc_info=True)


def stop_tray_before_exit(release_lock, *, wait: float = 0.5) -> None:
    """Remove the icon before an ordinary exit (bounded); Panic passes ``wait=0``."""
    if _active is not None:
        try:
            _active.stop(wait=wait)
        except Exception:
            log.warning("Icon cleanup failed before process exit.", exc_info=True)
    release_lock()


def restore_tray_after_handoff() -> None:
    """A definite spawn refusal, with the PID lock retaken: restore the same shell's indicator.

    Removal can precede pump exit; never start a second pump while the stopped one still owns it.
    If restoration fails, show a hidden window (or let cold startup create its visible window).
    """
    if _active is None or _active.background.shutdown.is_set():
        return
    indicator, background = _active, _active.background
    if indicator._idle.wait(INDICATOR_WAIT_SEC) and background._indicator_up():
        return
    log.warning("The launcher continues without its indicator; showing the window.")
    if background.window is not None:
        indicator._show_or_exit(background.window)


class Indicator:
    """Common half of a platform indicator: ``start → bool``, ``hide_on_close``,
    ``show_if_unavailable``, ``stop(wait)``. The window hides only while the icon is
    live, and a window hidden on purpose comes back if the icon goes away."""

    decides_natively = False  # Windows: the form's FormClosing handler sees the close reason
    carries_notifications = False  # Windows: notifications are balloons on this icon, up for the whole run

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

    def __init__(self, exit_launcher, read_port, shutdown_event, *, panic_child=lambda: (None, None)) -> None:
        cls = indicator_class()
        self._close = exit_launcher  # the launcher's close body: stop the agent, clean up, ``leave``
        self._closing = False  # that body runs now, on the one closer thread
        self._closer_lock = threading.Lock()
        self._terminal = threading.Lock()  # orders the ordinary exit against an accepted Panic; held for no I/O
        self.read_port = read_port
        self.panic_child = panic_child  # snapshot of this launcher's Popen and optional Job, not a PID scan
        self.shutdown = shutdown_event
        self.quit = threading.Event()  # a quit request: never asked, never cancelled
        self.panic = threading.Event()
        self.window = None
        self.native_ready = False
        self.indicator = cls(self) if cls is not None else None
        self.appearance = NativeAppearance()  # the window's caption tint follows its page (Windows)
        self.chrome = WindowChrome()  # the main window's title bar follows its document (macOS)
        self.zoom = PageZoom(on_changed=lambda: self.chrome.zoom_changed())  # the owner's Interface size here
        self.notifications = native_notifier(self.open_notification, self.indicator)
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
        if self.indicator is not None and self.indicator.carries_notifications:
            self._indicator_up()  # Windows: after the PID lock/runtime preparation, before long bootstrap
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
        self.chrome.attach(window)
        self.zoom.attach(window)
        window.events.closing += self.closing
        window.events.shown += self._window_shown  # set for a hidden window too (pywebview 5.4)
        if self.indicator is not None:
            window.events.before_show += self._attach_native
        return window

    def run(self) -> None:
        """``webview.start(func=...)``: keep Windows' early icon; start other indicators for background."""
        if self.indicator is None or (not self.indicator.carries_notifications and keep_running_choice() != "true"
                                      and not self.indicator.hidden):
            return
        self._indicator_up()  # the state poller starts only once the icon is visible
        if self.indicator.hidden:
            self.indicator.show_if_unavailable(self.window)

    def closing(self):
        """pywebview ``closing``; False keeps the window (hidden, asking, or quitting through the closer)."""
        if self.indicator is None or not self.native_ready or self.quit.is_set():
            return self.exit_launcher()
        if self.indicator.decides_natively:
            return None  # Windows: its FormClosing handler holds the close reason
        return self.window_closed()

    def exit_launcher(self):
        """Quit: latch shutdown now and run the close body, off the calling UI/STA thread and never twice
        at once. False: the caller's window and event loop stay, so Panic keeps its ingress until the
        body's exit. A body that returned without exiting (a startup close) lets a later quit run it again."""
        self.shutdown.set()
        with self._closer_lock:
            if self._closing:
                return False
            self._closing = True
        try:  # not a daemon: a main() that returns meanwhile still lets the body finish its stop
            threading.Thread(target=self._run_close, name="ouroboros-quit", daemon=False).start()
        except RuntimeError:
            self._run_close()  # thread admission failure cannot turn the owner's quit into a no-op
        return False

    def _run_close(self) -> None:
        try:
            self._close()
        finally:
            with self._closer_lock:
                self._closing = False

    def leave(self, release_lock, code: int = 0) -> None:
        """The close body's terminal exit: remove the icon, release the lock, then exit, unless an accepted
        Panic owns persistence and the exit (it is decided under the lock Panic is accepted under)."""
        stop_tray_before_exit(release_lock)
        with self._terminal:
            if not self.panic.is_set():
                os._exit(code)

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
        """The indicator's Quit: a plain exit; no Panic flag, sign-in startup untouched. Returns at once."""
        self.shutdown.set()
        self.quit.set()
        self.exit_launcher()

    def request_panic(self, ingress="launcher_tray") -> None:
        """Latch this launcher closed, request captured stops, then persist and exit.

        No callback registration, server port, HTTP request or reconnect is needed.
        The upstream emergency coordinator owns installation custody and hard exit.
        Accepted under the terminal lock: an ordinary exit either already committed or never will.
        """
        from ouroboros.launcher_bootstrap import source_handoff

        with self._terminal:
            if self.panic.is_set():
                return
            source_handoff.claim_stop()  # serialize with normal handoff exit, never with Popen
            self.panic.set()
        self.shutdown.set()
        self.quit.set()
        request_tray_cleanup()
        try:
            proc, job = self.panic_child()
            if proc is not None and proc.poll() is None:
                from ouroboros.platform_layer import IS_WINDOWS, kill_process_tree, request_process_tree_kill

                if IS_WINDOWS:
                    log.error("Panic captured server stop: %s", request_process_tree_kill(proc, job_handle=job))
                else:
                    # Capture descendants before server exit orphans workers' separate-session children.
                    kill_process_tree(proc)
        except Exception:
            log.exception("Captured server stop request failed; installation Panic continues.")
        try:
            from ouroboros.launcher_bootstrap import stop_bootstrap_commands

            log.error("Bootstrap stop requests: %s", stop_bootstrap_commands())
        except Exception:
            log.exception("Bootstrap stop request failed; installation Panic continues.")

        def stop() -> None:
            from ouroboros.config import DATA_DIR
            from ouroboros.server_control import execute_panic_stop

            try:
                from ouroboros.panic_source import record_source

                record_source(DATA_DIR, ingress=ingress, owner_surface="owner", receiver_is_server=False,
                              peer={"pid": os.getpid(), "ppid": os.getppid(),
                                    "executable": sys.executable[:2048], "command": str(sys.argv)[:2048]})
            except Exception:
                log.exception("Launcher Panic source evidence unavailable")
            execute_panic_stop(None, None, data_dir=DATA_DIR, panic_exit_code=PANIC_EXIT_CODE, log=log)

        # A main() returning at the shutdown latch must not lose the stop marker.
        try:
            threading.Thread(target=stop, name="ouroboros-indicator-panic", daemon=False).start()
        except RuntimeError:
            stop()  # thread admission failure cannot turn the owner gesture into a no-op


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
            if not self.indicator.carries_notifications and keep_running_choice() != "true" and not self.indicator.hidden:
                self.indicator.stop()  # the owner turned background off: no icon while the window is visible
                return
            self.indicator.set_status("Ouroboros: starting" if self.window is None else status_line(self.read_port()))
            if self.shutdown.wait(STATUS_POLL_SEC):
                return


class DesktopApi:
    """The page's alerts, shell facts and emergency stop. A page feature-detects each method per call: a launcher
    built before one of them simply lacks it (the page then falls back and says so, DESIGN §9)."""

    _background = None  # set by the inheriting MainApi; a leading underscore keeps it off the bridge
    _shell_facts = {}  # installed seed and loaded source, supplied by launcher.MainApi

    def panic_stop(self) -> dict:
        """Independent desktop emergency door; no server or reconnect dependency."""
        self._background.request_panic("desktop_bridge")
        return {"status": "stopping"}

    def request_attention(self, sound: bool = True, title: str = "", body: str = "", cue_when_visible: bool = True) -> dict:
        return self._background.attention(bool(sound), str(title or ""), str(body or ""), bool(cue_when_visible))

    notify_owner = request_attention  # newer pages send the alert text; older launchers lack this name

    def set_native_appearance(self, theme: str = "", page: float | None = None, sequence: int | None = None) -> dict:
        """The page's painted palette for this window's native caption; the newest request wins."""
        return self._background.appearance.request(theme, page, sequence)

    def set_interface_size(self, percent: int = 100, page: float | None = None, sequence: int | None = None) -> dict:
        """The owner's Interface size for this window, in percent (Settings → Appearance, ⌘/Ctrl + − 0); the newest
        request wins, and the next window opens at it."""
        return self._background.zoom.request(percent, page, sequence)

    def shell_info(self) -> dict:
        """Installed bundle and loaded shell are separate facts; capabilities are read from this process."""
        return {"shell_version": _SHELL_VERSION, **self._shell_facts, "persistent_storage": PERSISTENT_WEBVIEW_STORAGE,
                "native_notifications": self._notifier_answer("status"), "interface_size": self._zoom_info()}

    def _zoom_info(self) -> dict:
        zoom = getattr(self._background, "zoom", None)
        return zoom.info() if zoom is not None else {"supported": False}

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
