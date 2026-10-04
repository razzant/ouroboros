"""Windows background indicator: a notification-area icon on its own STA thread.

Created only while background mode is on (``launcher_background``). A window close is
decided in the form's own FormClosing handler, the one place that sees its CloseReason
(pywebview's ``closing`` event does not): only the owner's close may hide the window or
ask, while sign-out, shutdown and Task Manager closes always quit. The named auto-reset
kernel event behind a manual second launch exists only while its owning launcher holds
a handle; its name derives from the installation's PID-lock path, so it is not a
persistent file and a crashed owner cannot leave a stale request. WinForms and Win32
load only on Windows.
"""

import hashlib
import logging
import os
import queue
import sys
import threading
import time

from ouroboros.launcher_background import Indicator

log = logging.getLogger("launcher.tray")
# pywebview's WinForms confirmation is a fixed OK/Cancel box, so the text names the buttons.
_CONSENT_BUTTONS = "\n\nOK keeps it running in the background. Cancel quits."


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
    """Ask this installation's running launcher to show its window; no second UI.

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
            log.warning("Running desktop launcher did not accept the activation request.")
            return False
        time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))


def listen_for_activation(lock_path, on_request, shutdown_event) -> bool:
    """Own the installation's activation event for as long as this launcher runs."""
    kernel = _kernel()
    handle = kernel.CreateEventW(None, False, False, _event_name(lock_path)) if kernel else None
    if not handle:
        log.warning("Second-launch activation is unavailable; a second launch shows the old notice.")
        return False

    def wait() -> None:
        try:
            while not shutdown_event.is_set():
                if kernel.WaitForSingleObject(handle, 500) == 0:
                    on_request()
        finally:
            kernel.CloseHandle(handle)

    threading.Thread(target=wait, name="ouroboros-activation", daemon=True).start()
    return True


class WindowsTray(Indicator):
    decides_natively = True

    def __init__(self, background):
        super().__init__(background)
        self._sta_id = None
        self._icon = None
        self._disposed = threading.Event()
        self._balloons = queue.SimpleQueue()
        self._status = "Ouroboros"

    def attach_native(self, window):
        from System.Windows.Forms import CloseReason  # pywebview's WinForms backend loaded the assembly

        background = self.background

        def form_closing(sender, args):
            if args.CloseReason != CloseReason.UserClosing:
                background.quit.set()  # sign-out, shutdown, Task Manager: never cancelled, never asked
                background.exit_launcher()
            elif background.window_closed() is False:
                args.Cancel = True

        window.native.FormClosing += form_closing

    def notify(self, title, body):
        if not self.ready.is_set():
            return False
        self._balloons.put((title, body))  # shown by the STA tick; the OS plays the banner's sound
        return True

    def set_status(self, text):
        self._status = text

    def confirm(self, window, title, message):
        return bool(window.create_confirmation_dialog(title, message + _CONSENT_BUTTONS))

    def _dispose(self, wait):
        if wait and self._sta_id != threading.get_ident() and not self._disposed.wait(wait):
            log.warning("Tray icon removal unconfirmed before process exit.")

    def _launch(self):
        if sys.platform != "win32":
            return False
        import clr

        clr.AddReference("System.Windows.Forms")
        clr.AddReference("System.Drawing")
        clr.AddReference("System.Threading")
        from System.Drawing import Icon, SystemIcons
        from System.Threading import ApartmentState, Thread, ThreadStart
        from System.Windows.Forms import (Application, ApplicationContext, ContextMenuStrip, MouseButtons,
                                          NotifyIcon, Timer, ToolStripMenuItem, ToolStripSeparator, ToolTipIcon)

        background = self.background
        self._disposed.clear()

        def restore(sender=None, args=None):
            background.show_window()

        def dispose_icon():
            icon = self._icon
            if icon is not None:
                icon.Visible = False
                icon.Dispose()
                self._icon = None
            self.ready.clear()
            self._disposed.set()

        def quit_clicked(sender, args):
            self._stop.set()
            try:
                dispose_icon()  # STA, before the launcher's os._exit (finally may never run)
            except Exception:
                log.warning("Tray icon removal failed before Quit.", exc_info=True)
            background.request_quit()

        def pump():
            timer = None
            owned_icon = None
            self._sta_id = threading.get_ident()
            try:
                menu = ContextMenuStrip()
                state_item = ToolStripMenuItem(self._status)
                state_item.Enabled = False
                menu.Items.Add(state_item)
                menu.Items.Add(ToolStripSeparator())
                for text, handler in (("Open Ouroboros", restore),
                                      ("Panic", lambda sender, args: background.request_panic()),
                                      ("Quit Ouroboros", quit_clicked)):
                    item = ToolStripMenuItem(text)
                    item.Click += handler
                    menu.Items.Add(item)
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
                icon.Text = self._status[:63]
                icon.ContextMenuStrip = menu

                def mouse_click(sender, args):
                    if args.Button == MouseButtons.Left:
                        restore()

                icon.MouseClick += mouse_click
                icon.BalloonTipClicked += restore
                context = ApplicationContext()
                timer = Timer()
                timer.Interval = 200

                def tick(sender, args):
                    if self._stop.is_set() or background.shutdown.is_set():
                        dispose_icon()
                        Application.ExitThread()
                        return
                    if state_item.Text != self._status:
                        state_item.Text = self._status
                        icon.Text = self._status[:63]  # the notification-area tooltip limit
                    while not self._balloons.empty():
                        title, body = self._balloons.get_nowait()
                        icon.ShowBalloonTip(5000, title, body, ToolTipIcon.Info)
                    if icon.Visible:
                        self.ready.set()

                timer.Tick += tick
                timer.Start()
                icon.Visible = True
                Application.Run(context)
            except Exception:
                log.warning("Tray pump failed; window close will quit.", exc_info=True)
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
                self._stopped()

        thread = Thread(ThreadStart(pump))
        thread.SetApartmentState(ApartmentState.STA)
        thread.Start()
        return True
