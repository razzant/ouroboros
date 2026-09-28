"""Windows desktop tray and same-install window activation.

The named auto-reset kernel event exists only while its owning launcher holds a
handle. Its name is derived from the installation's PID-lock path; it is not a
persistent file, and a crashed owner cannot leave a stale activation request.
WinForms and Win32 are imported only on the Windows desktop path.
"""

import hashlib
import logging
import os
import sys
import threading
import time

log = logging.getLogger("launcher.tray")
_active_tray = None


def _event_name(lock_path):
    import pathlib

    canonical = os.path.normcase(str(pathlib.Path(lock_path).resolve()))
    return "Local\\OuroborosTray-" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]


def _kernel():
    """Bind Win32 signatures locally; no platform DLL loads on non-Windows hosts."""
    if sys.platform != "win32":
        return None
    import ctypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateEventW.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_wchar_p)
    kernel.CreateEventW.restype = ctypes.c_void_p
    kernel.OpenEventW.argtypes = (ctypes.c_uint32, ctypes.c_int, ctypes.c_wchar_p)
    kernel.OpenEventW.restype = ctypes.c_void_p
    kernel.SetEvent.argtypes = (ctypes.c_void_p,)
    kernel.SetEvent.restype = ctypes.c_int
    kernel.WaitForSingleObject.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
    kernel.WaitForSingleObject.restype = ctypes.c_uint32
    kernel.CloseHandle.argtypes = (ctypes.c_void_p,)
    kernel.CloseHandle.restype = ctypes.c_int
    return kernel


def activate_existing_tray(lock_path, *, timeout=3.0):
    """Ask this installation's live tray to show its window; no second UI.

    Lock losers during bootstrap may race event creation. A failed/unsupported
    signal falls through to the existing already-running notice, never success.
    """
    kernel = _kernel()
    if kernel is None:
        return False
    deadline = time.monotonic() + timeout
    while True:
        handle = kernel.OpenEventW(0x0002, False, _event_name(lock_path))
        if handle:
            try:
                if kernel.SetEvent(handle):
                    return True  # accepted by kernel, not an attestation that UI painted
            finally:
                kernel.CloseHandle(handle)
        if time.monotonic() >= deadline:
            log.warning("Running desktop tray did not accept activation request.")
            return False
        time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))


def request_tray_cleanup():
    """Begin icon disposal while the launcher still has other cleanup to do."""
    if _active_tray is not None:
        try:
            _active_tray.stop(wait=0)
        except Exception:
            log.warning("Tray cleanup request failed; Panic continues.", exc_info=True)


def stop_tray_before_exit(release_lock, *, wait=0.5):
    """Remove the icon before normal Exit; Panic never waits on the tray."""
    tray = _active_tray
    if tray is not None:
        try:
            tray.stop(wait=wait)
        except Exception:
            log.warning("Tray cleanup failed before process exit.", exc_info=True)
    release_lock()


class WindowsTray:
    def __init__(self, get_window, exit_launcher, lock_path, shutdown_event):
        self.get_window = get_window
        self.exit_launcher = exit_launcher
        self.lock_path = lock_path
        self.shutdown_event = shutdown_event
        self.ready = threading.Event()
        self._stop = threading.Event()
        self._disposed = threading.Event()
        self._hidden = False
        self._recovering = False
        self._state_lock = threading.Lock()
        self._sta_id = None
        self._icon = None
        self._event_handle = None
        self._kernel_api = None

    def attach(self, window, *, initially_hidden=False):
        """Return the pywebview closing handler; failed setup keeps normal exit."""
        global _active_tray
        self._hidden = initially_hidden
        if self.start():
            _active_tray = self

        def closing():
            if self.hide_on_close(window):
                return False  # pywebview closing cancellation contract
            return self.exit_launcher()

        return closing

    def start(self):
        try:
            kernel = _kernel()
            if kernel is None:
                return False
            import clr
            clr.AddReference("System.Windows.Forms")
            clr.AddReference("System.Drawing")
            clr.AddReference("System.Threading")
            from System.Drawing import Icon, SystemIcons
            from System.Threading import ApartmentState, Thread, ThreadStart
            from System.Windows.Forms import (Application, ApplicationContext, ContextMenuStrip,
                                              MouseButtons, NotifyIcon, Timer, ToolStripMenuItem)
            handle = kernel.CreateEventW(None, False, False, _event_name(self.lock_path))
            if not handle:
                raise OSError("Cannot create desktop activation event")
            self._event_handle = handle
            self._kernel_api = kernel
        except Exception:
            log.warning("Tray unavailable; window close will exit normally.", exc_info=True)
            return False

        def restore(sender=None, args=None):
            window = self.get_window()
            if window is not None:
                try:
                    window.show()
                except Exception:
                    log.warning("Tray could not show window; activation may be retried.", exc_info=True)
                else:
                    with self._state_lock:
                        self._hidden = False

        def dispose_icon():
            icon = self._icon
            if icon is not None:
                icon.Visible = False
                icon.Dispose()
                self._icon = None
            self.ready.clear()
            self._disposed.set()

        def exit_clicked(sender, args):
            self._stop.set()
            try:
                dispose_icon()  # STA, before launcher os._exit (finally may never run)
            except Exception:
                log.warning("Tray icon removal failed before Exit.", exc_info=True)
            self.exit_launcher()

        def pump():
            timer = None
            owned_icon = None
            self._sta_id = threading.get_ident()
            try:
                menu = ContextMenuStrip()
                open_item = ToolStripMenuItem("Open Ouroboros")
                open_item.Click += restore
                exit_item = ToolStripMenuItem("Exit")
                exit_item.Click += exit_clicked
                menu.Items.Add(open_item)
                menu.Items.Add(exit_item)
                icon = NotifyIcon()
                self._icon = icon
                icon.Icon = SystemIcons.Application
                from ouroboros.platform_layer import bundled_resource_bases
                for base in bundled_resource_bases():
                    candidate = base / "assets" / "icon.ico"
                    if candidate.is_file():
                        try:
                            owned_icon = Icon(str(candidate))
                            icon.Icon = owned_icon
                            break
                        except Exception:
                            log.warning("Tray icon asset unreadable: %s", candidate, exc_info=True)
                icon.Text = "Ouroboros — running"
                icon.ContextMenuStrip = menu
                def mouse_click(sender, args):
                    if args.Button == MouseButtons.Left:
                        restore()
                icon.MouseClick += mouse_click
                icon.MouseDoubleClick += restore
                context = ApplicationContext()
                timer = Timer()
                timer.Interval = 200

                def tick(sender, args):
                    if self._stop.is_set() or self.shutdown_event.is_set():
                        dispose_icon()
                        Application.ExitThread()
                    else:
                        if self._kernel_api.WaitForSingleObject(self._event_handle, 0) == 0:
                            restore()
                        if icon.Visible:
                            self.ready.set()

                timer.Tick += tick
                timer.Start()
                icon.Visible = True
                Application.Run(context)
            except Exception:
                log.warning("Tray pump failed; window close will exit normally.", exc_info=True)
            finally:
                if timer is not None:
                    for operation in (timer.Stop, timer.Dispose):
                        try:
                            operation()
                        except Exception:
                            log.warning("Tray timer cleanup failed.", exc_info=True)
                try:
                    dispose_icon()
                except Exception:
                    log.warning("Tray icon cleanup failed.", exc_info=True)
                if owned_icon is not None:
                    try:
                        owned_icon.Dispose()
                    except Exception:
                        log.warning("Tray icon asset cleanup failed.", exc_info=True)
                self._kernel_api.CloseHandle(self._event_handle)
                self._pump_stopped()

        try:
            thread = Thread(ThreadStart(pump))
            thread.SetApartmentState(ApartmentState.STA)
            thread.Start()
        except Exception:
            kernel.CloseHandle(handle)
            log.warning("Tray STA thread could not start.", exc_info=True)
            return False
        return True

    def stop(self, *, wait=0.0):
        self.ready.clear()
        self._stop.set()
        if wait and self._sta_id != threading.get_ident():
            if not self._disposed.wait(wait):
                log.warning("Tray icon removal unconfirmed before process exit.")

    def show_if_unavailable(self, window):
        """On automatic launch, expose the hidden window if no live icon appears."""
        if not self.ready.wait(3.0) or self._stop.is_set() or self.shutdown_event.is_set():
            self._show_or_exit(window)

    def _show_or_exit(self, window):
        if self.shutdown_event.is_set():
            return
        with self._state_lock:
            if not self._hidden or self._recovering:
                return  # another recovery already made the window visible
            self._recovering = True
        log.warning("Desktop tray unavailable; showing the window.")
        try:
            for attempt in range(2):
                try:
                    window.show()
                except Exception:
                    log.error("Desktop tray could not show its window.", exc_info=True)
                    if attempt == 0 and not self.shutdown_event.wait(0.1):
                        continue
                    if not self.shutdown_event.is_set():
                        self.exit_launcher()  # no usable UI: do not leave a hidden owner running
                else:
                    with self._state_lock:
                        self._hidden = False
                break
        finally:
            with self._state_lock:
                self._recovering = False

    def _pump_stopped(self):
        with self._state_lock:
            self.ready.clear()
            must_restore = self._hidden and not self._stop.is_set() and not self.shutdown_event.is_set()
        if must_restore:
            window = self.get_window()
            if window is None:
                self.exit_launcher()
            else:
                self._show_or_exit(window)

    def hide_on_close(self, window):
        with self._state_lock:
            if not self.ready.is_set() or self.shutdown_event.is_set():
                return False
            window.hide()
            self._hidden = True
            return True
