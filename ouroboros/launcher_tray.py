"""Windows background indicator: a notification-area icon on its own STA thread.

Created only while background mode is on (``launcher_background``). A window close is
decided in the form's own FormClosing handler, the one place that sees its CloseReason
(pywebview's ``closing`` event does not): only the owner's close may hide the window or
ask, while sign-out, shutdown and Task Manager closes always quit. The named auto-reset
kernel event behind a manual second launch exists only while its owning launcher holds
a handle; its name derives from the installation's PID-lock path, so it is not a
persistent file and a crashed owner cannot leave a stale request. ``NotificationIcon`` shows
system notifications (``desktop_notifications``) as balloons, each on an icon of its own
until its click. Every balloon is handed to the shell directly (``show_balloon``) and its caller
waits for the STA pump's ``Submission``: it learns the shell's own answer, not that a queue took
it, and Sound off can ask for a silent balloon; WinForms' own call, the audible fallback,
answers nothing, so its balloon stays unconfirmed. WinForms and Win32 load only on Windows.
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
SUBMIT_WAIT = 5.0  # seconds a caller waits for an STA pump to hand its balloon to the shell
# Notification icons alive at once. Each stays after its balloon timed out or was dismissed, so the
# notification stays in Windows' list, and leaves with its click; a newer one retires the oldest.
_BALLOON_ICONS = 3
_NIM_MODIFY, _NIF_INFO, _NIIF_INFO, _NIIF_NOSOUND = 0x1, 0x10, 0x1, 0x10


class Submission:
    """One balloon handed to an STA pump, and what became of it.

    ``wait`` answers the sound fact of a balloon the shell accepted (``show_balloon``: accepted is
    not proof that Windows showed it), False (never handed over, and now never will be: the pump
    skips a request its caller gave up on) or None (the pump took it and has not answered, or
    handed it over through a call that answers nothing)."""

    def __init__(self, *fields):
        self.fields = fields
        self._lock = threading.Lock()
        self._done = threading.Event()
        self._state, self._result = "queued", False

    def take(self) -> bool:
        """The pump starts on it; False when its caller already gave up."""
        with self._lock:
            if self._state != "queued":
                return False
            self._state = "taken"
            return True

    def finish(self, result) -> None:
        with self._lock:
            self._state, self._result = "finished", result
        self._done.set()

    def wait(self):
        self._done.wait(SUBMIT_WAIT)
        with self._lock:
            if self._state == "queued":
                self._state = "abandoned"
            return None if self._state == "taken" else self._result


def _drain(requests, submit) -> None:
    """STA tick: hand each queued balloon to ``submit`` and record the shell's answer."""
    while not requests.empty():
        request = requests.get_nowait()
        if request.take():
            try:
                request.finish(submit(*request.fields))
            except Exception:
                log.warning("A notification balloon was not handed to the shell.", exc_info=True)
                request.finish(False)


def _icon_identity(icon):
    """The window and id WinForms registered ``icon`` under: a balloon sent with them still reaches
    the icon's own click and close events. Private fields (.NET Framework's, then .NET's names)."""
    from System.Reflection import BindingFlags

    hidden, kind = BindingFlags.Instance | BindingFlags.NonPublic, icon.GetType()

    def field(*names):
        for name in names:
            found = kind.GetField(name, hidden)
            if found is not None:
                return found.GetValue(icon)
        raise LookupError(f"NotifyIcon keeps no {names[0]} field here")

    return int(field("window", "_window").Handle.ToInt64()), int(field("id", "_id"))


def notify_icon_data():
    """``NOTIFYICONDATAW`` (shellapi.h), built on demand: its WCHAR is two bytes only on Windows."""
    import ctypes
    from ctypes import wintypes

    class Guid(ctypes.Structure):
        _fields_ = [("Data1", wintypes.DWORD), ("Data2", wintypes.WORD), ("Data3", wintypes.WORD),
                    ("Data4", ctypes.c_ubyte * 8)]

    class NotifyIconData(ctypes.Structure):
        _fields_ = [("cbSize", wintypes.DWORD), ("hWnd", wintypes.HWND), ("uID", wintypes.UINT),
                    ("uFlags", wintypes.UINT), ("uCallbackMessage", wintypes.UINT), ("hIcon", wintypes.HICON),
                    ("szTip", wintypes.WCHAR * 128), ("dwState", wintypes.DWORD), ("dwStateMask", wintypes.DWORD),
                    ("szInfo", wintypes.WCHAR * 256), ("uVersion", wintypes.UINT),
                    ("szInfoTitle", wintypes.WCHAR * 64), ("dwInfoFlags", wintypes.DWORD), ("guidItem", Guid),
                    ("hBalloonIcon", wintypes.HICON)]

    return NotifyIconData


def _fit(text, units):
    """``text`` cut to ``units`` UTF-16 code units: a WCHAR array keeps one more for its terminator."""
    text = text[:units]
    while len(text.encode("utf-16-le")) > 2 * units:
        text = text[:-1]
    return text


def _shell_balloon(hwnd, uid, title, body, silent) -> bool:
    """``Shell_NotifyIconW(NIM_MODIFY, NIF_INFO)``: the shell's own answer, and NIIF_NOSOUND for silence."""
    import ctypes
    from ctypes import wintypes

    data_type = notify_icon_data()
    data = data_type(cbSize=ctypes.sizeof(data_type), hWnd=hwnd, uID=uid, uFlags=_NIF_INFO,
                     szInfo=_fit(body, 255), szInfoTitle=_fit(title, 63),
                     dwInfoFlags=_NIIF_INFO | (_NIIF_NOSOUND if silent else 0))
    shell = ctypes.WinDLL("shell32")
    shell.Shell_NotifyIconW.argtypes = (wintypes.DWORD, ctypes.POINTER(data_type))
    shell.Shell_NotifyIconW.restype = wintypes.BOOL
    return bool(shell.Shell_NotifyIconW(_NIM_MODIFY, ctypes.byref(data)))


def show_balloon(icon, title, body, sound):
    """On the STA thread: ``icon``'s balloon, handed to the shell. Returns its sound: ``os`` (Windows'
    notification sound) or ``off`` (sent silent); None after WinForms' fallback, whose void call
    discards the shell's answer. That fallback has no silent form, so a silent request is refused
    before submission if direct shell access is unavailable."""
    if not icon.Visible:
        raise RuntimeError("the icon is not in the notification area")
    try:
        hwnd, uid = _icon_identity(icon)
    except Exception:
        if not sound:
            raise RuntimeError("silent_balloon_unavailable")
        log.info("Direct balloon call unavailable; using WinForms for an audible request.", exc_info=True)
        from System.Windows.Forms import ToolTipIcon

        icon.ShowBalloonTip(5000, title, body, ToolTipIcon.Info)
        return None  # handed over, unconfirmed: the caller adds nothing, and the icon keeps its click
    if not _shell_balloon(hwnd, uid, title, body, not sound):
        raise OSError("the shell refused the balloon")
    return "os" if sound else "off"


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

    def notify(self, title, body, sound=True):
        if not self.ready.is_set():
            return False
        request = Submission(title, body, sound)
        self._balloons.put(request)  # handed to the shell by the STA tick; the OS plays the banner's sound
        return request.wait() is not False  # unanswered: it may still show, so the caller adds no sound

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
                                          NotifyIcon, Timer, ToolStripMenuItem, ToolStripSeparator)

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
                    _drain(self._balloons, lambda title, body, sound: show_balloon(icon, title, body, sound))
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


class NotificationIcon:
    """System notifications as notification-area balloons, each on an icon of its own.

    Windows 10/11 present a balloon as a system notification. A balloon's click event names no
    balloon, so one icon per notification is what ties a click to its own source. The icon stays in the
    notification area from its balloon until that balloon is clicked: a balloon that timed out or was
    dismissed keeps its notification in Windows' list, which removing the icon would take with it.
    At most ``_BALLOON_ICONS`` are; a newer one retires the oldest. Nothing is shown again. Its own STA
    pump, like ``WindowsTray``; a click hands that balloon's token to ``on_click``. ``stop`` (the
    launcher's exit, Panic) removes every icon and ends the pump: an icon left behind outlives the process."""

    def __init__(self, on_click):
        self._on_click = on_click
        self._queue = queue.SimpleQueue()
        self._lock = threading.Lock()
        self._tried = False
        self.alive = threading.Event()  # the pump runs: a queued balloon will be handed to the shell
        self._stop = threading.Event()
        self._disposed = threading.Event()  # set once the pump removed its icons and its timer
        self._sta_id = None

    def show(self, title, body, sound, token):
        """``Submission.wait``: the balloon's sound fact once the shell accepted it, False or None."""
        with self._lock:
            if not self._tried and not self._stop.is_set():
                self._tried = True
                self._start()
        if self._stop.is_set() or not self.alive.is_set():
            return False
        request = Submission(title, body, sound, token)
        self._queue.put(request)
        return request.wait()

    def stop(self, wait: float = 0.0) -> None:
        """Remove every icon and end the pump; ``wait`` bounds waiting for its STA tick to do so."""
        self._stop.set()
        if wait and self.alive.is_set() and self._sta_id != threading.get_ident() \
                and not self._disposed.wait(wait):
            log.warning("Notification icon removal unconfirmed before process exit.")

    def _start(self) -> None:
        if sys.platform != "win32":
            return
        try:
            import clr

            clr.AddReference("System.Windows.Forms")
            clr.AddReference("System.Drawing")
            from System.Drawing import Icon, SystemIcons
            from System.Threading import ApartmentState, Thread, ThreadStart
            from System.Windows.Forms import Application, ApplicationContext, NotifyIcon, Timer

            from ouroboros.platform_layer import bundled_resource_bases
        except Exception:
            log.warning("Notification-area balloons are unavailable here.", exc_info=True)
            return
        settled = threading.Event()
        live, gone, timers = [], [], []  # icons with a balloon, oldest first; retired ones; the pump's timer

        def remove_all():
            """On the STA thread: every icon, the timer, and any balloon still queued (never handed over)."""
            for icon in [*live, *gone]:
                try:
                    icon.Visible = False
                    icon.Dispose()
                except Exception:
                    log.warning("A notification icon could not be removed.", exc_info=True)
            live.clear()
            gone.clear()
            while timers:
                timer = timers.pop()
                for operation in (timer.Stop, timer.Dispose):
                    try:
                        operation()
                    except Exception:
                        log.warning("Notification timer cleanup failed.", exc_info=True)
            while not self._queue.empty():
                request = self._queue.get_nowait()
                if request.take():
                    request.finish(False)
            self._disposed.set()

        def pump():
            self._sta_id = threading.get_ident()
            try:
                image = SystemIcons.Application
                for base in bundled_resource_bases():
                    candidate = base / "assets" / "icon.ico"
                    if candidate.is_file():
                        try:
                            image = Icon(str(candidate))
                            break
                        except Exception:
                            log.warning("Notification icon asset unreadable: %s", candidate, exc_info=True)

                def retire(icon):
                    if any(other is icon for other in live):
                        live[:] = [other for other in live if other is not icon]
                        icon.Visible = False  # out of the notification area, and its balloon out of Windows' list
                        gone.append(icon)

                def submit(title, body, sound, token):
                    icon = NotifyIcon()
                    icon.Icon, icon.Text = image, "Ouroboros"
                    icon.BalloonTipClicked += lambda sender, args: (retire(icon), self._on_click(token))
                    live.append(icon)
                    try:
                        icon.Visible = True
                        result = show_balloon(icon, title, body, sound)
                    except Exception:
                        retire(icon)
                        raise
                    while len(live) > _BALLOON_ICONS:
                        retire(live[0])
                    return result

                def tick(sender, args):
                    while gone:
                        gone.pop().Dispose()  # never inside the icon's own event handler
                    if self._stop.is_set():
                        remove_all()
                        Application.ExitThread()
                        return
                    _drain(self._queue, submit)

                timer = Timer()
                timers.append(timer)
                timer.Interval = 200
                timer.Tick += tick
                timer.Start()
                self.alive.set()
                settled.set()
                Application.Run(ApplicationContext())
            except Exception:
                log.warning("Notification-area balloon pump failed.", exc_info=True)
            finally:
                self.alive.clear()  # later notifications answer unavailable: the page falls back
                remove_all()
                settled.set()

        thread = Thread(ThreadStart(pump))
        thread.SetApartmentState(ApartmentState.STA)
        thread.IsBackground = True  # never keeps the launcher alive
        thread.Start()
        settled.wait(3.0)
