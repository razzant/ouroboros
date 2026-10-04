"""macOS background indicator: a menu-bar item, Dock reopen, and quit requests that are never cancelled.

pywebview 5.4 has no Dock-click hook, and Cmd+Q, Dock Quit and logout run the same ``closing``
event as the red button. The D2 probe (macOS 26.4, pywebview 5.4, PyObjC 11.1) showed that a
subclass of pywebview's own NSApplication delegate can answer
``applicationShouldHandleReopen:hasVisibleWindows:`` (a Dock click on a hidden window) and mark
``applicationShouldTerminate:`` as a quit request before pywebview runs that event, and that an
NSStatusItem made on the main thread renders with its menu. While the window is hidden an alert
badges and bounces the Dock icon instead of raising the window. AppKit loads only on macOS.
"""

import logging

from ouroboros.launcher_background import Indicator

log = logging.getLogger("launcher.tray")
_shared: dict = {}  # Objective-C class names are process-wide: each class is defined once


def _classes(base):
    """The delegate subclass and the menu target, bound to the one ``Background`` of this launcher."""
    if "delegate" not in _shared:
        import AppKit
        import objc

        class OuroborosAppDelegate(base):
            def applicationShouldHandleReopen_hasVisibleWindows_(self, sender, flag):
                _shared["background"].show_window()
                return False

            def applicationShouldTerminate_(self, sender):
                _shared["background"].quit.set()  # Cmd+Q, Dock Quit, logout: never cancelled, never asked
                return objc.super(OuroborosAppDelegate, self).applicationShouldTerminate_(sender)

        class OuroborosMenuTarget(AppKit.NSObject):
            def openWindow_(self, sender):
                _shared["background"].show_window()

            def panic_(self, sender):
                _shared["background"].request_panic()

            def quitApp_(self, sender):
                _shared["background"].request_quit()

        _shared.update(delegate=OuroborosAppDelegate, target=OuroborosMenuTarget)
    return _shared["delegate"], _shared["target"]


def _on_main(fn) -> None:
    import AppKit
    from PyObjCTools import AppHelper

    if AppKit.NSThread.isMainThread():
        fn()
    else:
        AppHelper.callAfter(fn)


class MacStatusItem(Indicator):
    def __init__(self, background):
        super().__init__(background)
        self._item = None
        self._state_item = None
        self._keep = []  # strong references: NSApplication and NSMenuItem hold these weakly
        self._status = "Ouroboros"

    def attach_native(self, window):
        """``before_show`` runs on the main thread after pywebview set its own delegate."""
        import AppKit

        app = AppKit.NSApplication.sharedApplication()
        _shared["background"] = self.background
        delegate_class, _target = _classes(type(app.delegate()))
        delegate = delegate_class.alloc().init()
        self._keep.append(delegate)
        app.setDelegate_(delegate)

    def notify(self, title, body):
        def cue():
            import AppKit

            app = AppKit.NSApplication.sharedApplication()
            app.dockTile().setBadgeLabel_("•")
            app.requestUserAttention_(AppKit.NSInformationalRequest)

        _on_main(cue)
        return False  # not a banner: the caller plays the system sound

    def set_status(self, text):
        self._status = text

        def apply():
            if self._state_item is not None:
                self._state_item.setTitle_(text)
                self._item.button().setToolTip_(text)

        _on_main(apply)

    def confirm(self, window, title, message):
        labels = window.localization
        saved = {key: labels.get(key) for key in ("global.ok", "global.cancel")}
        labels.update({"global.ok": "Keep running", "global.cancel": "Quit"})
        try:
            return bool(window.create_confirmation_dialog(title, message))
        finally:
            labels.update(saved)

    def _shown(self):
        def clear():
            import AppKit

            AppKit.NSApplication.sharedApplication().dockTile().setBadgeLabel_(None)

        _on_main(clear)

    def _launch(self):
        _on_main(self._create)  # synchronous on the main thread: a close may be waiting for it
        return True

    def _create(self):
        import AppKit

        try:
            if self._stop.is_set():
                raise RuntimeError("stopped before the menu-bar item was created")
            _delegate, target_class = _classes(type(AppKit.NSApplication.sharedApplication().delegate()))
            item = AppKit.NSStatusBar.systemStatusBar().statusItemWithLength_(AppKit.NSVariableStatusItemLength)
            image = _menu_bar_image()
            if image is not None:
                item.button().setImage_(image)
            else:
                item.button().setTitle_("Ouroboros")
            item.button().setToolTip_(self._status)
            target = target_class.alloc().init()
            menu = AppKit.NSMenu.alloc().init()
            menu.setAutoenablesItems_(False)
            state = menu.addItemWithTitle_action_keyEquivalent_(self._status, None, "")
            state.setEnabled_(False)
            menu.addItem_(AppKit.NSMenuItem.separatorItem())
            for title, action in (("Open Ouroboros", "openWindow:"), ("Panic", "panic:"), ("Quit Ouroboros", "quitApp:")):
                menu.addItemWithTitle_action_keyEquivalent_(title, action, "").setTarget_(target)
            item.setMenu_(menu)
            self._keep.extend([target, menu])
            self._item, self._state_item = item, state
            self.ready.set()
        except Exception:
            log.warning("Menu-bar item unavailable; closing the window quits.", exc_info=True)
            self._stopped()

    def _dispose(self, wait):
        def remove():
            import AppKit

            item, self._item, self._state_item = self._item, None, None
            if item is not None:
                AppKit.NSStatusBar.systemStatusBar().removeStatusItem_(item)
            self._stopped()

        _on_main(remove)  # Panic does not wait: the item leaves with the process


def _menu_bar_image():
    """The app icon at menu-bar size, or None (the item then shows its name)."""
    import AppKit

    from ouroboros.platform_layer import bundled_resource_bases

    for base in bundled_resource_bases():
        candidate = base / "assets" / "icon.icns"
        if candidate.is_file():
            image = AppKit.NSImage.alloc().initWithContentsOfFile_(str(candidate))
            if image is not None:
                image.setSize_((18, 18))
                return image
    return None
