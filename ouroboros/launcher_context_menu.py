"""Each desktop window's right-click menu: the engine's own menu, without the items that cannot work in this shell.

pywebview 5.4 takes the menu away outside debug, on every backend: WKWebView's ``willOpenMenu:withEvent:`` empties it
(macOS adds Services back when there is a selection, nothing else), WebView2 has ``AreDefaultContextMenusEnabled`` off,
GTK answers ``context-menu`` with "handled", Qt's view has ``NoContextMenu``. The page leaves links, fields and selected
words to this menu (``web/modules/chat_row_menu.js``, DESIGN "Replies"), so without it a link cannot be copied.

Each window gets the menu back in its ``before_show`` (the GUI thread, before the first load), whichever window it is:
``attach_every_window`` wraps ``webview.create_window``, so no window of this launcher needs its own line. A debug start
keeps pywebview's own menu. A refusal leaves that window as pywebview made it, and a menu that cannot be filtered opens
empty or stays closed: never unfiltered, which could take the window away.

What goes is what failed when chosen under pywebview 5.4, matched by the engine's own unlocalized name, never by a
title. Every other item stays, unknown future ones included, and separators left with nothing to separate go too. Each
list but WebView2's was decided by real right-clicks in a disposable guest:

- macOS 26.4, WebKit 21624.1.16.11.4: Open Link loads the link in the app's own window. Open Link/Image/Frame/Video in
  New Window do nothing: the delegate opens only what WebKit calls link-activated. Download Linked File/Image/Video
  fetch the target and save nothing, no panel. Stop, which has no identifier of its own, leaves a stalled document
  loading. Inspect Element appears only in debug. Everything else tried works and stays: Copy Link, Copy Image, Copy
  Video Address, Copy, Copy Link with Highlight, Cut, Paste, spelling, Share, Look Up, Translate, Search With (the
  system browser), Services, and Back, Forward, Reload (the window stays on its own pages).
- WebKitGTK 2.52.6, Ubuntu 24.04: the same Open Link, New Window items and Stop fail. Its Download items stay: they
  save without a prompt (the guest saved in its home directory). Open Audio in New Window goes with Video's, unseen: no page
  offered it.
- QtWebEngine 6.4 with PyQt6, the same Ubuntu: Save link/image/media/page save nothing, and View page source hands the
  page's own address to the system browser. Open link in new tab/window stay: they reach the system browser.
  ``OpenLinkInThisWindow``, which 6.4's menu does not offer, goes for what it says.
- WebView2 is unobserved, there is no Windows guest: only what pywebview's source refuses goes, every ``save…As`` of the
  documented lower-camel names, because ``on_download_starting`` cancels each download. Open link in new window stays,
  ``on_new_window_request`` opens the system browser.
"""

from __future__ import annotations

import functools
import logging
from typing import Any, Sequence

log = logging.getLogger("launcher.context_menu")

_COCOA_REMOVED = frozenset({
    "WKMenuItemIdentifierOpenLink",
    "WKMenuItemIdentifierOpenLinkInNewWindow",
    "WKMenuItemIdentifierOpenImageInNewWindow",
    "WKMenuItemIdentifierOpenFrameInNewWindow",
    "WKMenuItemIdentifierOpenMediaInNewWindow",
    "WKMenuItemIdentifierDownloadLinkedFile",
    "WKMenuItemIdentifierDownloadImage",
    "WKMenuItemIdentifierDownloadMedia",
})
_COCOA_STOP_TAG = 11  # WebMenuItemTagStop: Stop's identifier is only its action's name, shared with Cut and others
_GTK_REMOVED = (  # WebKit2.ContextMenuAction
    "OPEN_LINK", "OPEN_LINK_IN_NEW_WINDOW", "OPEN_IMAGE_IN_NEW_WINDOW", "OPEN_FRAME_IN_NEW_WINDOW",
    "OPEN_VIDEO_IN_NEW_WINDOW", "OPEN_AUDIO_IN_NEW_WINDOW", "STOP",
)
_QT_REMOVED = (  # QWebEnginePage.WebAction
    "OpenLinkInThisWindow", "DownloadLinkToDisk", "DownloadImageToDisk", "DownloadMediaToDisk", "SavePage", "ViewSource",
)

_patched: set = set()  # backends whose class was patched: a class is process-wide, its windows many


def attach_every_window(webview_module: Any) -> None:
    """Make ``webview.create_window`` attach each window it creates: one line in the launcher covers them all."""
    create = getattr(webview_module, "create_window", None)
    if create is None or getattr(create, "_attaches_context_menu", False):
        return

    @functools.wraps(create)
    def create_window(*args: Any, **kwargs: Any) -> Any:
        window = create(*args, **kwargs)
        attach(window)
        return window

    create_window._attaches_context_menu = True  # type: ignore[attr-defined]
    webview_module.create_window = create_window


def attach(window: Any) -> None:
    """Give each launcher window selectable content and its menu before ``webview.start``."""
    if getattr(getattr(window, "events", None), "before_show", None) is not None:
        window.text_select = True  # pywebview defaults to False, including setup and startup notices
        window.events.before_show += lambda: _restore(window)


def _restore(window: Any) -> None:
    """pywebview ``before_show``: the backend pywebview chose gives this window its menu back; a refusal is only a log."""
    try:
        import webview

        guilib = webview.guilib
        restore = _RESTORERS.get(str(getattr(guilib, "__name__", "")).rpartition(".")[2])
        if restore is not None and not _debug():
            restore(guilib, window)
    except Exception:
        log.warning("This window keeps pywebview's answer to a right-click.", exc_info=True)


def _debug() -> bool:
    import webview

    return bool(webview._state["debug"])


def separators_to_drop(is_separator: Sequence[bool]) -> list[int]:
    """Indexes, ascending, of the separators that separate nothing: leading, trailing and repeated ones."""
    drop: list[int] = []
    after_item = False
    pending = None
    for index, separator in enumerate(is_separator):
        if not separator:
            after_item, pending = True, None
        elif after_item:
            after_item, pending = False, index
        else:
            drop.append(index)
    if pending is not None:
        drop.append(pending)
    return sorted(drop)


# -- macOS ---------------------------------------------------------------------------------------------------------
def filter_cocoa_menu(menu: Any) -> None:
    """The NSMenu WebKit is about to show, without what fails in this shell."""
    for item in list(menu.itemArray()):
        if str(item.identifier()) in _COCOA_REMOVED or item.tag() == _COCOA_STOP_TAG:
            menu.removeItem_(item)
    flags = [bool(item.isSeparatorItem()) for item in menu.itemArray()]
    for index in reversed(separators_to_drop(flags)):
        menu.removeItemAtIndex_(index)


def _restore_cocoa(guilib: Any, _window: Any) -> None:
    if "cocoa" in _patched:
        return
    from objc import classAddMethods

    def willOpenMenu_withEvent_(_view: Any, menu: Any, _event: Any) -> None:  # noqa: N802 (Objective-C selector)
        if _debug():
            return
        try:
            filter_cocoa_menu(menu)
        except Exception:
            log.warning("This menu stays empty: it could not be filtered.", exc_info=True)
            menu.removeAllItems()

    classAddMethods(guilib.BrowserView.WebKitHost, [willOpenMenu_withEvent_])
    _patched.add("cocoa")


# -- Windows -------------------------------------------------------------------------------------------------------
def filter_webview2_menu(items: Any) -> None:
    """``CoreWebView2ContextMenuRequestedEventArgs.MenuItems`` without what fails in this shell."""
    kept = list(items)
    for index in reversed(range(len(kept))):
        name = str(kept[index].Name)
        if name.startswith("save") and name.endswith("As"):
            items.RemoveAt(index)
            del kept[index]
    for index in reversed(separators_to_drop([str(item.Kind) == "Separator" for item in kept])):
        items.RemoveAt(index)


def _on_webview2_menu(_sender: Any, args: Any) -> None:
    try:
        filter_webview2_menu(args.MenuItems)
    except Exception:
        log.warning("This menu stays closed: it could not be filtered.", exc_info=True)
        args.Handled = True  # the host took it: WebView2 shows no default menu


def _open_webview2(core: Any) -> None:
    core.ContextMenuRequested += _on_webview2_menu
    core.Settings.AreDefaultContextMenusEnabled = True  # the event is raised only while this is on


def _restore_winforms(guilib: Any, window: Any) -> None:
    if getattr(guilib, "renderer", "") != "edgechromium":
        return  # MSHTML and CEF keep pywebview's answer
    view = window.native.webview

    def ready(sender: Any, args: Any) -> None:  # after pywebview's own handler, added first, turned the menus off
        try:
            if args.IsSuccess:
                _open_webview2(sender.CoreWebView2)
        except Exception:
            log.warning("This window keeps WebView2's menus off.", exc_info=True)

    if view.CoreWebView2 is None:
        view.CoreWebView2InitializationCompleted += ready
    else:
        _open_webview2(view.CoreWebView2)


# -- Linux ---------------------------------------------------------------------------------------------------------
def filter_gtk_menu(menu: Any, removed: Sequence[Any]) -> None:
    """The WebKitContextMenu about to show, without the ``removed`` stock actions."""
    for item in list(menu.get_items()):
        if item.get_stock_action() in removed:
            menu.remove(item)
    left = list(menu.get_items())
    for index in reversed(separators_to_drop([bool(item.is_separator()) for item in left])):
        menu.remove(left[index])


def _restore_gtk(guilib: Any, window: Any) -> None:
    from gi.repository import GObject

    view = guilib.BrowserView.instances[window.uid].webview
    actions = guilib.webkit.ContextMenuAction
    removed = [getattr(actions, name) for name in _GTK_REMOVED if hasattr(actions, name)]

    def on_menu(_view: Any, menu: Any, *_rest: Any) -> bool:
        try:
            filter_gtk_menu(menu, removed)
        except Exception:
            log.warning("This menu stays closed: it could not be filtered.", exc_info=True)
            return True
        return False

    # pywebview's own handler is an anonymous lambda answering "handled", which ends the signal: it must go.
    signal = GObject.signal_lookup("context-menu", type(view))
    handler = GObject.signal_handler_find(view, GObject.SignalMatchType.ID, signal, 0, None, None, None)
    # Keep the old blocker until a filter is connected; a failed connection must not expose an unfiltered menu.
    view.connect("context-menu", on_menu)
    if handler:
        view.disconnect(handler)


def filter_qt_menu(menu: Any, page: Any, web_page_class: Any) -> None:
    """The QMenu of a QWebEngineView, without what fails in this shell (by web action: their texts are translated)."""
    actions = getattr(web_page_class, "WebAction", web_page_class)  # Qt6 scopes the enum; older bindings need not
    for name in _QT_REMOVED:
        action = getattr(actions, name, getattr(web_page_class, name, None))
        if action is not None:
            menu.removeAction(page.action(action))
    left = list(menu.actions())
    for index in reversed(separators_to_drop([bool(action.isSeparator()) for action in left])):
        menu.removeAction(left[index])


def _restore_qt(guilib: Any, window: Any) -> None:
    if not getattr(guilib, "is_webengine", False):
        return  # QtWebKit has no standard-menu factory here: it keeps pywebview's answer
    host = guilib.BrowserView.WebView
    if "qt" not in _patched:
        stock = host.contextMenuEvent  # the debug menu, with its Inspect Element

        def contextMenuEvent(view: Any, event: Any) -> None:  # noqa: N802 (Qt virtual)
            if _debug():
                stock(view, event)
                return
            try:  # an exception leaving a Qt virtual can end the process
                page = view.page()
                menu = view.createStandardContextMenu() if guilib._qt6 else page.createStandardContextMenu()
                filter_qt_menu(menu, page, guilib.QWebPage)
                menu.exec_(event.globalPos())
            except Exception:
                log.warning("This menu stays closed: it could not be filtered.", exc_info=True)

        host.contextMenuEvent = contextMenuEvent
        _patched.add("qt")
    guilib.BrowserView.instances[window.uid].webview.setContextMenuPolicy(guilib.QtCore.Qt.DefaultContextMenu)


_RESTORERS = {"cocoa": _restore_cocoa, "winforms": _restore_winforms, "gtk": _restore_gtk, "qt": _restore_qt}
