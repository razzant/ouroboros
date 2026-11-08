"""Each desktop window's right-click menu (``launcher_context_menu``): the engine's own, without what fails in this shell.

Stand-ins replace the engines' menus and pywebview's backends, so this runs on every CI OS. The menus below are the ones
a real right-click showed under pywebview 5.4, logged before any removal: macOS 26.4 (WebKit 21624.1.16.11.4), and in an
Ubuntu 24.04 guest WebKitGTK 2.52.6 and QtWebEngine 6.4. What each item does when chosen belongs to those native runs, not
to this file. WebView2's names are the documented ones: no Windows run stands behind them.
"""
from __future__ import annotations

import ast
import logging
import sys
import types
from pathlib import Path

import pytest

from ouroboros import launcher_context_menu as lcm

ROOT = Path(__file__).resolve().parents[1]
SEPARATOR = None
FORWARD = "forwardContextMenuAction:"  # AppKit reports the action's name for an item WebKit gave no identifier
ID = "WKMenuItemIdentifier"
OPEN_LINK, NEW_WINDOW, DOWNLOAD_LINK = (f"{ID}OpenLink", 33), (f"{ID}OpenLinkInNewWindow", 1), (f"{ID}DownloadLinkedFile", 2)
COPY_LINK, SHARE = (f"{ID}CopyLink", 3), (f"{ID}ShareMenu", 0)
NEW_IMAGE, DOWNLOAD_IMAGE, COPY_IMAGE = (f"{ID}OpenImageInNewWindow", 4), (f"{ID}DownloadImage", 5), (f"{ID}CopyImage", 6)
SUBJECT, REVEAL = (f"{ID}CopySubject", 97), (f"{ID}RevealImage", 94)
NEW_FRAME = (f"{ID}OpenFrameInNewWindow", 7)
LOOK_UP, TRANSLATE, SEARCH = (f"{ID}LookUp", 22), (f"{ID}Translate", 95), (f"{ID}SearchWeb", 21)
COPY, HIGHLIGHT, SPEECH = (f"{ID}Copy", 8), (f"{ID}CopyLinkWithHighlight", 102), (f"{ID}SpeechMenu", 0)
CUT, PASTE, SPELLING = (FORWARD, 13), (f"{ID}Paste", 14), (f"{ID}SpellingMenu", 0)
GUESS, IGNORE, SUBSTITUTIONS = (FORWARD, 15), (FORWARD, 17), (None, 0)
BACK, FORWARD_ITEM, RELOAD, STOP = (f"{ID}GoBack", 9), (f"{ID}GoForward", 10), (f"{ID}Reload", 12), (FORWARD, 11)
PLAY, CONTROLS, COPY_MEDIA = (FORWARD, 0), (f"{ID}ShowHideMediaControls", 0), (f"{ID}CopyMediaLink", 0)
NEW_MEDIA, DOWNLOAD_MEDIA = (f"{ID}OpenMediaInNewWindow", 0), (f"{ID}DownloadMedia", 0)
class Item:
    def __init__(self, entry):
        self.entry = entry

    def identifier(self):
        return None if self.entry is SEPARATOR else self.entry[0]

    def tag(self):
        return 0 if self.entry is SEPARATOR else self.entry[1]

    def isSeparatorItem(self):
        return self.entry is SEPARATOR


class Menu:
    def __init__(self, entries):
        self.items = [Item(entry) for entry in entries]

    def itemArray(self):
        return list(self.items)

    def removeItem_(self, item):
        self.items.remove(item)

    def removeItemAtIndex_(self, index):
        del self.items[index]

    def removeAllItems(self):
        self.items.clear()

    def entries(self):
        return [item.entry for item in self.items]


@pytest.mark.parametrize("name, stock, shown", [
    ("link", [OPEN_LINK, NEW_WINDOW, DOWNLOAD_LINK, COPY_LINK, SEPARATOR, SHARE], [COPY_LINK, SEPARATOR, SHARE]),
    ("selected text",
     [LOOK_UP, TRANSLATE, SEPARATOR, SEARCH, SEPARATOR, COPY, HIGHLIGHT, SEPARATOR, SHARE, SEPARATOR, SEPARATOR, SEPARATOR, SPEECH],
     [LOOK_UP, TRANSLATE, SEPARATOR, SEARCH, SEPARATOR, COPY, HIGHLIGHT, SEPARATOR, SHARE, SEPARATOR, SPEECH]),
    ("document still loading", [BACK, STOP, SEPARATOR], [BACK]),
    ("frame: nothing of use is left", [NEW_FRAME, SEPARATOR], []),
])
def test_cocoa_menus_keep_what_works(name, stock, shown):
    menu = Menu(stock)
    lcm.filter_cocoa_menu(menu)
    assert menu.entries() == shown, name


def test_cut_and_stop_share_an_identifier_so_only_the_tag_tells_them():
    menu = Menu([CUT, STOP, GUESS])
    lcm.filter_cocoa_menu(menu)
    assert menu.entries() == [CUT, GUESS]


@pytest.mark.parametrize("flags, drop", [
    ([], []),
    ([False, False], []),
    ([True], [0]),
    ([False, True, False], []),
    ([True, False, True, True, False, True], [0, 3, 5]),
])
def test_separators_that_separate_nothing(flags, drop):
    assert lcm.separators_to_drop(flags) == drop


# -- the launcher's one line: every window, whichever ---------------------------------------------------------------
class Hook:
    def __init__(self):
        self.handlers = []

    def __iadd__(self, handler):
        self.handlers.append(handler)
        return self


class Window:
    uid = "master"

    def __init__(self):
        self.text_select = False
        self.events = types.SimpleNamespace(before_show=Hook())


def test_every_created_window_is_attached_once():
    created = []
    fake = types.SimpleNamespace(create_window=lambda *a, **k: created.append((a, k)) or Window())
    lcm.attach_every_window(fake)
    lcm.attach_every_window(fake)
    window = fake.create_window("Title", url="http://127.0.0.1:1", width=2)
    assert created == [(("Title",), {"url": "http://127.0.0.1:1", "width": 2})]
    assert len(window.events.before_show.handlers) == 1
    assert window.text_select is True


def test_a_stand_in_webview_without_windows_events_or_create_window_is_left_alone():
    lcm.attach(None)
    lcm.attach(types.SimpleNamespace())
    lcm.attach_every_window(types.SimpleNamespace())
    fake = types.SimpleNamespace(create_window=lambda *a, **k: None)
    lcm.attach_every_window(fake)
    assert fake.create_window("Title") is None


@pytest.fixture
def backend(monkeypatch):
    """A stand-in ``webview`` module with the chosen backend's name; ``state`` says whether this start is debug."""
    monkeypatch.setattr(lcm, "_patched", set())
    fake = types.ModuleType("webview")
    fake._state = {"debug": False}
    monkeypatch.setitem(sys.modules, "webview", fake)

    def choose(name, **members):
        guilib = types.SimpleNamespace(__name__=f"webview.platforms.{name}", **members)
        fake.guilib = guilib
        return guilib

    choose.state = fake._state
    return choose


def test_each_backend_is_reached_by_pywebviews_choice(backend, monkeypatch):
    calls = []
    monkeypatch.setattr(lcm, "_RESTORERS", {"gtk": lambda guilib, window: calls.append((guilib, window))})
    window = Window()
    lcm.attach(window)
    guilib = backend("gtk")
    window.events.before_show.handlers[0]()
    backend("android")
    window.events.before_show.handlers[0]()
    assert calls == [(guilib, window)]


def test_a_debug_start_keeps_pywebviews_menu_and_a_refusal_is_only_a_log(backend, monkeypatch, caplog):
    def refuse(_guilib, _window):
        raise RuntimeError("refused")

    monkeypatch.setattr(lcm, "_RESTORERS", {"gtk": refuse})
    window = Window()
    lcm.attach(window)
    backend("gtk")
    backend.state["debug"] = True
    window.events.before_show.handlers[0]()
    assert not caplog.records
    backend.state["debug"] = False
    with caplog.at_level(logging.WARNING):
        window.events.before_show.handlers[0]()
    assert "keeps pywebview" in caplog.text


def test_the_launcher_attaches_before_it_can_open_any_window():
    tree = ast.parse((ROOT / "launcher.py").read_text(encoding="utf-8"))
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    lines = {}
    for node in ast.walk(main):
        if isinstance(node, ast.Call):
            name = getattr(node.func, "attr", getattr(node.func, "id", ""))
            lines.setdefault(name, node.lineno)
    assert lines["attach_every_window"] < min(lines[name] for name in ("create_window", "_already_running_window"))


# -- macOS: the method on pywebview's class -------------------------------------------------------------------------
@pytest.fixture
def cocoa(backend, monkeypatch):
    host = types.SimpleNamespace()
    fake_objc = types.ModuleType("objc")
    fake_objc.classAddMethods = lambda cls, methods: [setattr(cls, method.__name__, method) for method in methods]
    monkeypatch.setitem(sys.modules, "objc", fake_objc)
    guilib = backend("cocoa", BrowserView=types.SimpleNamespace(WebKitHost=host))
    lcm._restore_cocoa(guilib, None)
    return host, backend, guilib


def test_cocoa_method_filters_outside_debug(cocoa):
    host, _backend, _guilib = cocoa
    menu = Menu([OPEN_LINK, COPY_LINK])
    host.willOpenMenu_withEvent_(None, menu, None)
    assert menu.entries() == [COPY_LINK]


def test_cocoa_debug_menu_is_untouched(cocoa):
    host, backend_, _guilib = cocoa
    backend_.state["debug"] = True
    menu = Menu([OPEN_LINK, SEPARATOR, COPY_LINK, SEPARATOR])
    host.willOpenMenu_withEvent_(None, menu, None)
    assert menu.entries() == [OPEN_LINK, SEPARATOR, COPY_LINK, SEPARATOR]


def test_cocoa_method_is_installed_once_per_process(cocoa):
    host, _backend, guilib = cocoa
    method = host.willOpenMenu_withEvent_
    lcm._restore_cocoa(guilib, None)
    assert host.willOpenMenu_withEvent_ is method


def test_a_menu_that_cannot_be_filtered_opens_empty_never_unfiltered(cocoa, monkeypatch, caplog):
    host, _backend, _guilib = cocoa

    def boom(_menu):
        raise RuntimeError("boom")

    monkeypatch.setattr(lcm, "filter_cocoa_menu", boom)
    menu = Menu([OPEN_LINK, COPY_LINK])
    with caplog.at_level(logging.WARNING):
        host.willOpenMenu_withEvent_(None, menu, None)
    assert menu.entries() == []
    assert "could not be filtered" in caplog.text


# Importing the real Cocoa backend creates NSApplication and changes its activation
# policy. That one-off probe belongs to retained guest evidence, not the host suite.


def test_early_windows_allow_selection_as_the_main_window_does():
    window = Window()
    lcm.attach(window)
    assert window.text_select is True
    assert len(window.events.before_show.handlers) == 1


# -- Windows: WebView2 ----------------------------------------------------------------------------------------------
class EdgeItem:
    def __init__(self, name, kind="Command"):
        self.Name, self.Kind = name, kind


class EdgeItems(list):
    def RemoveAt(self, index):
        del self[index]

    def Clear(self):
        del self[:]


def edge_menu(*names):
    return EdgeItems(EdgeItem("separator", "Separator") if name == "-" else EdgeItem(name) for name in names)


def test_webview2_drops_each_save_and_only_them():
    items = edge_menu("openLinkInNewWindow", "saveLinkAs", "copyLinkLocation", "-", "saveImageAs", "-", "copy", "saveAs",
                      "extension", "custom", "spellCheck", "-")
    lcm.filter_webview2_menu(items)
    assert [item.Name for item in items] == ["openLinkInNewWindow", "copyLinkLocation", "separator", "copy", "extension",
                                             "custom", "spellCheck"]


class Settings:
    AreDefaultContextMenusEnabled = False


class Core:
    def __init__(self):
        self.ContextMenuRequested = Hook()
        self.Settings = Settings()


class WebView2:
    def __init__(self, core=None):
        self.CoreWebView2 = core
        self.CoreWebView2InitializationCompleted = Hook()


def edge_window(view):
    window = Window()
    window.native = types.SimpleNamespace(webview=view)
    return window


def test_webview2_opens_after_pywebviews_own_ready_handler_turned_the_menus_off(backend):
    guilib = backend("winforms", renderer="edgechromium")
    view = WebView2()
    lcm._restore_winforms(guilib, edge_window(view))
    assert len(view.CoreWebView2InitializationCompleted.handlers) == 1
    core = Core()
    ready = view.CoreWebView2InitializationCompleted.handlers[0]
    ready(types.SimpleNamespace(CoreWebView2=core), types.SimpleNamespace(IsSuccess=True))
    assert core.Settings.AreDefaultContextMenusEnabled is True
    assert core.ContextMenuRequested.handlers == [lcm._on_webview2_menu]


def test_webview2_failed_start_stays_as_pywebview_left_it(backend, caplog):
    guilib = backend("winforms", renderer="edgechromium")
    view = WebView2()
    lcm._restore_winforms(guilib, edge_window(view))
    core = Core()
    ready = view.CoreWebView2InitializationCompleted.handlers[0]
    ready(types.SimpleNamespace(CoreWebView2=core), types.SimpleNamespace(IsSuccess=False))
    assert core.Settings.AreDefaultContextMenusEnabled is False
    assert not core.ContextMenuRequested.handlers
    with caplog.at_level(logging.WARNING):
        ready(types.SimpleNamespace(), types.SimpleNamespace(IsSuccess=True))  # no CoreWebView2: refused, not raised
    assert "keeps WebView2's menus off" in caplog.text


def test_webview2_already_ready_opens_at_once(backend):
    core = Core()
    lcm._restore_winforms(backend("winforms", renderer="edgechromium"), edge_window(WebView2(core)))
    assert core.Settings.AreDefaultContextMenusEnabled is True


@pytest.mark.parametrize("renderer", ["mshtml", "cef", ""])
def test_other_windows_renderers_keep_pywebviews_answer(backend, renderer):
    view = WebView2()
    lcm._restore_winforms(backend("winforms", renderer=renderer), edge_window(view))
    assert not view.CoreWebView2InitializationCompleted.handlers


def test_webview2_menu_that_cannot_be_filtered_stays_closed(caplog):
    class Refusing(EdgeItems):
        def RemoveAt(self, index):
            raise RuntimeError("refused")

    args = types.SimpleNamespace(MenuItems=Refusing([EdgeItem("saveAs"), EdgeItem("copy")]), Handled=False)
    with caplog.at_level(logging.WARNING):
        lcm._on_webview2_menu(None, args)
    assert args.Handled is True
    assert "could not be filtered" in caplog.text


# -- Linux: WebKitGTK and Qt ----------------------------------------------------------------------------------------
class GtkItem:
    def __init__(self, action, separator=False):
        self.action, self.separator = action, separator

    def get_stock_action(self):
        return self.action

    def is_separator(self):
        return self.separator


class GtkMenu:
    def __init__(self, items):
        self.items = items

    def get_items(self):
        return list(self.items)

    def remove(self, item):
        self.items.remove(item)


def test_gtk_filter_by_stock_action_tidies_separators():
    open_link, copy_link, download, share = GtkItem(1), GtkItem(2), GtkItem(3), GtkItem(4)
    first, second, last = GtkItem(0, True), GtkItem(0, True), GtkItem(0, True)
    menu = GtkMenu([open_link, first, copy_link, second, download, last, share])
    lcm.filter_gtk_menu(menu, [1, 3])
    assert menu.items == [copy_link, second, share]


GTK_SEPARATOR = "-"
GTK_ACTIONS = ("OPEN_LINK", "OPEN_LINK_IN_NEW_WINDOW", "DOWNLOAD_LINK_TO_DISK", "COPY_LINK_TO_CLIPBOARD",
               "OPEN_IMAGE_IN_NEW_WINDOW", "DOWNLOAD_IMAGE_TO_DISK", "COPY_IMAGE_TO_CLIPBOARD", "COPY_IMAGE_URL_TO_CLIPBOARD",
               "OPEN_FRAME_IN_NEW_WINDOW", "GO_BACK", "GO_FORWARD", "STOP", "RELOAD", "COPY", "CUT", "PASTE", "DELETE",
               "SELECT_ALL", "OPEN_VIDEO_IN_NEW_WINDOW", "OPEN_AUDIO_IN_NEW_WINDOW", "DOWNLOAD_VIDEO_TO_DISK",
               "COPY_VIDEO_LINK_TO_CLIPBOARD", "MEDIA_PLAY", "CUSTOM")


@pytest.mark.parametrize("stock, shown", [
    (["OPEN_LINK", "OPEN_LINK_IN_NEW_WINDOW", "DOWNLOAD_LINK_TO_DISK", "COPY_LINK_TO_CLIPBOARD"],
     ["DOWNLOAD_LINK_TO_DISK", "COPY_LINK_TO_CLIPBOARD"]),
    (["CUT", "COPY", "PASTE", "DELETE", "-", "SELECT_ALL"], ["CUT", "COPY", "PASTE", "DELETE", "-", "SELECT_ALL"]),
    (["GO_BACK", "RELOAD", "OPEN_FRAME_IN_NEW_WINDOW"], ["GO_BACK", "RELOAD"]),
])
def test_gtk_menus_keep_what_works(stock, shown):
    menu = GtkMenu([GtkItem(GTK_SEPARATOR, True) if name == GTK_SEPARATOR else GtkItem(name) for name in stock])
    lcm.filter_gtk_menu(menu, lcm._GTK_REMOVED)
    assert [item.action for item in menu.items] == shown


class GtkView:
    def __init__(self):
        self.disconnected, self.connected = [], []

    def disconnect(self, handler):
        self.disconnected.append(handler)

    def connect(self, signal, handler):
        self.connected.append((signal, handler))


@pytest.fixture
def gtk(backend, monkeypatch):
    view = GtkView()
    gobject = types.SimpleNamespace(
        SignalMatchType=types.SimpleNamespace(ID="id"),
        signal_lookup=lambda name, cls: 7 if name == "context-menu" and cls is GtkView else 0,
        signal_handler_find=lambda instance, mask, signal, detail, closure, func, data: 41 if (
            instance is view and mask == "id" and signal == 7) else 0,
    )
    repository = types.ModuleType("gi.repository")
    repository.GObject = gobject
    monkeypatch.setitem(sys.modules, "gi", types.ModuleType("gi"))
    monkeypatch.setitem(sys.modules, "gi.repository", repository)
    actions = types.SimpleNamespace(**{name: name for name in GTK_ACTIONS})
    guilib = backend("gtk", webkit=types.SimpleNamespace(ContextMenuAction=actions),
                     BrowserView=types.SimpleNamespace(instances={"master": types.SimpleNamespace(webview=view)}))
    return guilib, view


def test_gtk_replaces_pywebviews_handled_answer_with_the_filter(gtk):
    guilib, view = gtk
    lcm._restore_gtk(guilib, Window())
    assert view.disconnected == [41]
    [(signal, handler)] = view.connected
    assert signal == "context-menu"
    kept = GtkItem("COPY_LINK_TO_CLIPBOARD")
    menu = GtkMenu([GtkItem("OPEN_LINK"), kept])
    assert handler(view, menu, object(), object()) is False  # False: the engine shows it
    assert menu.items == [kept]


def test_gtk_menu_that_cannot_be_filtered_stays_closed(gtk, caplog):
    guilib, view = gtk
    lcm._restore_gtk(guilib, Window())
    handler = view.connected[0][1]
    with caplog.at_level(logging.WARNING):
        assert handler(view, object()) is True
    assert "could not be filtered" in caplog.text


def test_gtk_connection_failure_keeps_the_original_blocker(gtk, monkeypatch):
    guilib, view = gtk

    def refuse(*_args):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(view, "connect", refuse)
    with pytest.raises(RuntimeError, match="connection refused"):
        lcm._restore_gtk(guilib, Window())
    assert view.disconnected == []


class QtAction:
    def __init__(self, separator=False):
        self.separator = separator

    def isSeparator(self):
        return self.separator


class QtMenu:
    def __init__(self, actions):
        self.items = list(actions)
        self.shown_at = None

    def actions(self):
        return list(self.items)

    def removeAction(self, action):
        if action in self.items:
            self.items.remove(action)

    def exec_(self, position):
        self.shown_at = position


QT_NAMES = ("Back", "Forward", "Reload", "SavePage", "ViewSource", "Copy", "Cut", "Paste", "SelectAll", "OpenLinkInThisWindow",
            "OpenLinkInNewTab", "OpenLinkInNewWindow", "DownloadLinkToDisk", "CopyLinkToClipboard", "DownloadImageToDisk",
            "CopyImageToClipboard", "CopyImageUrlToClipboard", "DownloadMediaToDisk", "CopyMediaUrlToClipboard", "ToggleMediaLoop")


class QtPage:
    def __init__(self):
        self.by_web_action = {name: QtAction() for name in QT_NAMES}
        self.menu = None

    def action(self, web_action):
        return self.by_web_action[web_action]

    def menu_of(self, names):
        return QtMenu(QtAction(True) if name == "-" else self.by_web_action[name] for name in names)

    def standard_menu(self):
        self.menu = self.menu_of(["OpenLinkInThisWindow", "CopyLinkToClipboard", "-", "ViewSource"])
        return self.menu

    def names(self, menu):
        reverse = {action: name for name, action in self.by_web_action.items()}
        return [reverse.get(action, "-") for action in menu.items]


class QtView:
    def __init__(self, page):
        self._page = page
        self.policy = "none"

    def page(self):
        return self._page

    def createStandardContextMenu(self):
        return self._page.standard_menu()

    def setContextMenuPolicy(self, policy):
        self.policy = policy


class QtHost:
    def contextMenuEvent(self, event):
        event.stock = True


def qt_backend(backend, page):
    view = QtView(page)
    guilib = backend(
        "qt", is_webengine=True, _qt6=True, QWebPage=types.SimpleNamespace(**{name: name for name in QT_NAMES}),
        QtCore=types.SimpleNamespace(Qt=types.SimpleNamespace(DefaultContextMenu="default")),
        BrowserView=types.SimpleNamespace(WebView=QtHost, instances={"master": types.SimpleNamespace(webview=view)}))
    return guilib, view


@pytest.fixture
def qt_host_restored(monkeypatch):
    monkeypatch.setattr(QtHost, "contextMenuEvent", QtHost.__dict__["contextMenuEvent"])


def test_qt_turns_the_menu_on_and_filters_by_web_action(backend, qt_host_restored):
    page = QtPage()
    guilib, view = qt_backend(backend, page)
    lcm._restore_qt(guilib, Window())
    assert view.policy == "default"
    event = types.SimpleNamespace(globalPos=lambda: (3, 4))
    QtHost.contextMenuEvent(view, event)
    assert page.names(page.menu) == ["CopyLinkToClipboard"]
    assert page.menu.shown_at == (3, 4)
    assert not hasattr(event, "stock")


@pytest.mark.parametrize(("stock", "shown", "enum_of"), [
    pytest.param(["Back", "Forward", "Reload", "-", "SavePage", "OpenLinkInNewTab", "OpenLinkInNewWindow", "-", "DownloadLinkToDisk",
      "CopyLinkToClipboard"], ["Back", "Forward", "Reload", "-", "OpenLinkInNewTab", "OpenLinkInNewWindow", "-", "CopyLinkToClipboard"], lambda names: types.SimpleNamespace(**names), id='stock0-shown0-flat'),  # bindings that keep the enum's names on the class
    pytest.param(["Back", "Forward", "Reload", "-", "SavePage", "-", "ViewSource"], ["Back", "Forward", "Reload"], lambda names: types.SimpleNamespace(WebAction=types.SimpleNamespace(**names)), id='stock1-shown1-scoped'),  # Qt6's scoped enum
    pytest.param(["Copy"], ["Copy"], lambda names: types.SimpleNamespace(**names), id='stock2-shown2-flat'),  # bindings that keep the enum's names on the class
    pytest.param(["Cut", "Copy", "Paste", "SelectAll"], ["Cut", "Copy", "Paste", "SelectAll"], lambda names: types.SimpleNamespace(WebAction=types.SimpleNamespace(**names)), id='stock3-shown3-scoped'),  # Qt6's scoped enum
    pytest.param(["Back", "Reload", "-", "SavePage", "-", "DownloadImageToDisk", "CopyImageToClipboard", "CopyImageUrlToClipboard"], ["Back", "Reload", "-", "CopyImageToClipboard", "CopyImageUrlToClipboard"], lambda names: types.SimpleNamespace(**names), id='stock4-shown4-flat'),  # bindings that keep the enum's names on the class
    pytest.param(["Back", "Reload", "-", "SavePage", "-", "ToggleMediaLoop", "DownloadMediaToDisk", "CopyMediaUrlToClipboard"], ["Back", "Reload", "-", "ToggleMediaLoop", "CopyMediaUrlToClipboard"], lambda names: types.SimpleNamespace(WebAction=types.SimpleNamespace(**names)), id='stock5-shown5-scoped'),  # Qt6's scoped enum
    pytest.param(["OpenLinkInThisWindow", "CopyLinkToClipboard"], ["CopyLinkToClipboard"], lambda names: types.SimpleNamespace(**names), id='stock6-shown6-flat'),  # bindings that keep the enum's names on the class
])
def test_qt_menus_keep_what_works(enum_of, stock, shown):
    page = QtPage()
    menu = page.menu_of(stock)
    lcm.filter_qt_menu(menu, page, enum_of({name: name for name in QT_NAMES}))
    assert page.names(menu) == shown


def test_qt_debug_keeps_pywebviews_own_menu(backend, qt_host_restored):
    page = QtPage()
    guilib, view = qt_backend(backend, page)
    lcm._restore_qt(guilib, Window())
    backend.state["debug"] = True
    event = types.SimpleNamespace(globalPos=lambda: (3, 4))
    QtHost.contextMenuEvent(view, event)
    assert event.stock is True
    assert page.menu is None


def test_qt_menu_that_cannot_be_built_stays_closed_and_leaves_the_virtual_quietly(backend, qt_host_restored, caplog):
    page = QtPage()
    guilib, view = qt_backend(backend, page)
    lcm._restore_qt(guilib, Window())

    def refuse():
        raise RuntimeError("refused")

    view.createStandardContextMenu = refuse
    with caplog.at_level(logging.WARNING):
        QtHost.contextMenuEvent(view, types.SimpleNamespace(globalPos=lambda: (3, 4)))
    assert "could not be filtered" in caplog.text


def test_qtwebkit_keeps_pywebviews_answer(backend):
    view = QtView(QtPage())
    guilib = backend("qt", is_webengine=False,
                     BrowserView=types.SimpleNamespace(WebView=QtHost, instances={"master": types.SimpleNamespace(webview=view)}))
    lcm._restore_qt(guilib, Window())
    assert view.policy == "none"
