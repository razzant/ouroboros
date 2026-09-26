"""Windows close-to-tray support for the desktop launcher.

Owns the WinForms ``NotifyIcon`` pump and the shared ready/exit events so the
root ``launcher.py`` stays inside the size-ratchet module budget. The closing
handler in the launcher hides the window only while :data:`tray_ready` is set
(clearing on pump failure), so a broken pythonnet can never trap the window in
an un-closable state. Other platforms never import this module.
"""

import logging
import threading

log = logging.getLogger("launcher.tray")

# Set once the tray pump is running; the closing handler hides to tray only
# while a live tray icon exists (cleared if the pump dies).
tray_ready = threading.Event()
# Set by the tray Exit item: the tray loop should stop after full shutdown ran.
tray_exit_needed = threading.Event()
# Shuttle for the graceful-exit callback (launcher wires its own _stop_agent_
# and_children / shutdown sequence; the tray never imports launcher internals).
shutdown_event: threading.Event | None = None
get_window = None  # callable() -> pywebview window | None


def _graceful_exit(port: int) -> None:
    """Full graceful shutdown from a non-window surface (tray Exit)."""
    log.info("Tray exit - graceful shutdown.")
    if shutdown_event is not None:
        shutdown_event.set()
    window = get_window() if get_window is not None else None
    if window is not None:
        window.show()
    tray_exit_needed.set()
    import os

    os._exit(0)  # the launcher's stop sequence is unreachable from a tray callback


def run_tray_icon(port: int, *, exit_handler) -> None:
    """Windows tray icon so closing the window hides to tray by default.

    Runs its own STA thread with a WinForms ApplicationContext (NotifyIcon
    needs a message pump; Application.Run(ApplicationContext) provides one
    without creating a form). Menu: Open (restore + activate), Exit (full
    graceful shutdown via ``exit_handler(port)``, which never returns). Left
    click restores. No new dependency: pythonnet (WinForms) is already the
    pywebview backend on Windows; the frozen build reuses the bundled
    assets/icon.ico and source/dev falls back to the system application
    icon. Other platforms never call this - their close behavior is
    unchanged.
    """
    try:
        import clr  # noqa: F401  (loads pythonnet's WinForms assemblies)

        clr.AddReference("System.Windows.Forms")
        clr.AddReference("System.Drawing")
        clr.AddReference("System.Threading")
        from System.Drawing import Icon, SystemIcons
        from System.Threading import ApartmentState, Thread, ThreadStart
        from System.Windows.Forms import (
            ApplicationContext,
            Application,
            ContextMenuStrip,
            NotifyIcon,
            Timer,
            ToolStripMenuItem,
            ToolTipIcon,
        )
    except Exception:
        # Tray unavailable (broken pythonnet etc.): never signal tray_ready,
        # so window-close keeps the ordinary full-shutdown behavior.
        log.warning("Tray icon unavailable; window close will exit.", exc_info=True)
        return

    def _pump() -> None:
        # Poll for shutdown from foreign threads: Application.ExitThread only
        # stops the pump when called ON the STA thread (verified live - a
        # foreign-thread ExitThread call is silently ignored), so a WinForms
        # Timer inside the pump watches the python events instead.
        timer = Timer()
        timer.Interval = 200

        def _tick(sender, args):
            if tray_exit_needed.is_set() or (shutdown_event is not None and shutdown_event.is_set()):
                timer.Stop()
                Application.ExitThread()

        timer.Tick += _tick
        timer.Start()
        context = ApplicationContext()
        tray_ready.set()
        try:
            Application.Run(context)
        except Exception:
            # If the message pump itself dies the tray is gone: clear the
            # ready latch so window close falls back to the plain exit path
            # instead of hiding into a tray that no longer exists.
            tray_ready.clear()
            raise
        finally:
            try:
                notify_icon.Visible = False
                notify_icon.Dispose()
            except Exception:
                pass

    def _icon():
        from ouroboros.platform_layer import bundled_resource_bases

        for base in bundled_resource_bases():
            candidate = base / "assets" / "icon.ico"
            if candidate.is_file():
                try:
                    return Icon(str(candidate))
                except Exception:
                    pass
        try:
            return SystemIcons.Application
        except Exception:
            return None

    def _open_click(sender, args):
        window = get_window() if get_window is not None else None
        if window is not None:
            window.show()

    def _exit_click(sender, args):
        exit_handler(port)

    menu = ContextMenuStrip()
    open_item = ToolStripMenuItem("Open Ouroboros")
    open_item.Click += _open_click
    exit_item = ToolStripMenuItem("Exit")
    exit_item.Click += _exit_click
    menu.Items.Add(open_item)
    menu.Items.Add(exit_item)

    notify_icon = NotifyIcon()
    notify_icon.Icon = _icon()
    notify_icon.Text = "Ouroboros - running (close hides to tray)"
    notify_icon.ContextMenuStrip = menu
    notify_icon.Visible = True
    notify_icon.BalloonTipTitle = "Ouroboros"
    notify_icon.BalloonTipText = "Still running in the tray. Click to reopen."
    notify_icon.BalloonTipIcon = ToolTipIcon.Info
    try:
        notify_icon.ShowBalloonTip(3000)
    except Exception:
        pass

    def _tray_click(sender, args):
        window = get_window() if get_window is not None else None
        if window is not None:
            window.show()

    notify_icon.MouseClick += _tray_click

    def _tray_doubleclick(sender, args):
        window = get_window() if get_window is not None else None
        if window is not None:
            window.show()

    notify_icon.MouseDoubleClick += _tray_doubleclick

    # NotifyIcon needs an STA message pump. pythonnet does not propagate COM
    # apartment state to plain threading.Thread (MTA), so the pump rides a
    # real System.Threading.Thread marked STA - the same construction
    # pywebview's own WinForms backend uses for its UI thread.
    pump = Thread(ThreadStart(_pump))
    pump.SetApartmentState(ApartmentState.STA)
    pump.Start()
