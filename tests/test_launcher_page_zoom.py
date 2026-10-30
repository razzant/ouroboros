"""The desktop window's page zoom (``launcher_page_zoom``): the owner's Interface size for this window.

Fakes stand in for the engines' views (no window is opened): the leaf is checked for what it asks each engine to do,
on which thread's queue, in which order, and what it keeps for the next window. The engine's own behaviour, a real
WKWebView re-laying out at ``pageZoom`` 1.25 (innerWidth 880 of 1100, devicePixelRatio 2.5), was measured in an
off-screen probe for the design, not here.
"""
from __future__ import annotations

import json
import sys
import threading

import pytest

from ouroboros import launcher_page_zoom as pz


class _Events:
    def __init__(self):
        self.handlers = []

    def __iadd__(self, handler):
        self.handlers.append(handler)
        return self


class _Window:
    uid = "main"

    def __init__(self):
        self.events = type("E", (), {})()
        self.events.before_show = _Events()
        self.events.closed = _Events()


class _CocoaView:
    def __init__(self):
        self.zooms = []

    def setPageZoom_(self, factor):
        self.zooms.append(factor)


@pytest.fixture
def mac(monkeypatch, tmp_path):
    view = _CocoaView()
    queued = []
    monkeypatch.setattr(pz, "_backend", lambda: "cocoa")
    monkeypatch.setattr(pz, "_find_view", lambda kind, window: (view, None))
    changed = []
    zoom = pz.PageZoom(data_dir=lambda: tmp_path, on_changed=lambda: changed.append(True))
    monkeypatch.setattr(zoom, "_post", lambda kind, v, form: queued.append((kind, zoom.info()["percent"] / 100)) or zoom._apply_now(kind, v))
    return zoom, view, queued, changed, tmp_path


def test_clamp_keeps_whole_steps_inside_the_window_range():
    assert pz.clamp_percent(133) == 135
    assert pz.clamp_percent(10) == pz.PERCENT_MIN
    assert pz.clamp_percent(999) == pz.PERCENT_MAX
    assert pz.clamp_percent("80") == 80
    for bad in (None, "x", float("nan"), float("inf"), True):
        assert pz.clamp_percent(bad) is None


def test_a_stored_size_is_the_first_paint_and_a_request_is_kept_for_the_next_window(mac):
    zoom, view, _queued, changed, root = mac
    (root / "state").mkdir()
    (root / "state" / pz.STATE_NAME).write_text(json.dumps({"percent": 125}), encoding="utf-8")
    window = _Window()
    zoom.attach(window)
    window.events.before_show.handlers[0]()
    assert view.zooms == [1.25], "before_show gives the stored size before the first load"
    assert zoom.info() == {"supported": True, "min": 50, "max": 200, "step": 5, "percent": 125}
    answer = zoom.request(80, page=1.0, sequence=1)
    assert answer == {"ok": True, "state": "scheduled", "percent": 80}
    assert view.zooms[-1] == 0.8 and changed, "the title band is measured again after a change"
    assert json.loads((root / "state" / pz.STATE_NAME).read_text(encoding="utf-8")) == {"percent": 80}


def test_only_the_newest_request_of_the_newest_page_is_kept(mac):
    zoom, view, queued, _changed, _root = mac
    window = _Window()
    zoom.attach(window)
    window.events.before_show.handlers[0]()
    assert view.zooms == [], "100 % needs no write"
    assert zoom.request(150, page=2.0, sequence=3)["percent"] == 150
    stale = zoom.request(90, page=2.0, sequence=2)  # a reordered older call of the same page
    assert stale == {"ok": True, "state": "stale", "percent": 150}
    older_page = zoom.request(90, page=1.0, sequence=9)
    assert older_page["state"] == "stale"
    assert zoom.request(110, page=3.0, sequence=1)["percent"] == 110, "a newer page wins from its first call"
    assert [factor for _kind, factor in queued] == [1.5, 1.1]


def test_two_requests_that_overlap_on_their_threads_leave_the_newest_size(mac, monkeypatch):
    """Bridge calls run on pywebview's worker threads: the older one may finish its write after the newer one."""
    zoom, view, _queued, _changed, root = mac
    window = _Window()
    zoom.attach(window)
    window.events.before_show.handlers[0]()
    store, held, release = zoom._store, threading.Event(), threading.Event()

    def held_store():
        if threading.current_thread().name == "older":
            held.set()
            release.wait(5)
        store()

    monkeypatch.setattr(zoom, "_store", held_store)
    older = threading.Thread(target=zoom.request, args=(110, 1.0, 1), name="older")
    older.start()
    assert held.wait(5), "the older call passed its order check and is about to write"
    assert zoom.request(120, page=1.0, sequence=2)["percent"] == 120
    release.set()
    older.join(5)
    assert view.zooms[-1] == 1.2, "the older call's late write applies the newest size"
    assert json.loads((root / "state" / pz.STATE_NAME).read_text(encoding="utf-8")) == {"percent": 120}
    assert zoom.info()["percent"] == 120


def test_invalid_requests_and_a_closed_window_change_nothing(mac):
    zoom, view, queued, _changed, root = mac
    window = _Window()
    zoom.attach(window)
    window.events.before_show.handlers[0]()
    assert zoom.request("big", page=1.0, sequence=1)["state"] == "invalid_size"
    assert zoom.request(120, page="soon", sequence=1)["state"] == "invalid_order"
    assert zoom.request(120, page=float("inf"), sequence=1)["state"] == "invalid_order"
    window.events.closed.handlers[0]()
    assert zoom.request(120, page=1.0, sequence=1)["state"] == "closed"
    assert queued == [] and view.zooms == [] and not (root / "state" / pz.STATE_NAME).exists()


def test_a_window_whose_view_has_no_page_zoom_reports_no_capability(monkeypatch, tmp_path):
    monkeypatch.setattr(pz, "_backend", lambda: "edgechromium")
    monkeypatch.setattr(pz, "_find_view", lambda kind, window: (None, None))  # the legacy MSHTML control
    zoom = pz.PageZoom(data_dir=lambda: tmp_path)
    window = _Window()
    zoom.attach(window)
    window.events.before_show.handlers[0]()
    assert zoom.info()["supported"] is False
    assert zoom.request(120, page=1.0, sequence=1) == {"ok": False, "state": "unsupported", "percent": 100}


def test_each_engine_gets_its_own_page_zoom_call(monkeypatch, tmp_path):
    zoom = pz.PageZoom(data_dir=lambda: tmp_path)

    class WebView2:
        ZoomFactor = 1.0

    class Gtk:
        level = None

        def set_zoom_level(self, factor):
            self.level = factor

    edge, gtk = WebView2(), Gtk()
    zoom._percent = 125
    assert zoom._apply_now("edgechromium", edge) and edge.ZoomFactor == 1.25
    zoom._percent = 75
    assert zoom._apply_now("gtk", gtk) and gtk.level == 0.75


def test_an_unreadable_store_reads_as_100(tmp_path):
    (tmp_path / "state").mkdir()
    (tmp_path / "state" / pz.STATE_NAME).write_text("not json", encoding="utf-8")
    assert pz.PageZoom(data_dir=lambda: tmp_path)._load() == 100
    (tmp_path / "state" / pz.STATE_NAME).write_text(json.dumps({"percent": "huge"}), encoding="utf-8")
    assert pz.PageZoom(data_dir=lambda: tmp_path)._load() == 100


@pytest.mark.skipif(sys.platform != "darwin", reason="the View menu is macOS's")
def test_the_view_menu_gets_the_size_items_once_and_each_runs_the_pages_own_step(tmp_path):
    AppKit = pytest.importorskip("AppKit")
    main = AppKit.NSMenu.alloc().init()
    view = AppKit.NSMenu.alloc().initWithTitle_("View")
    view.addItemWithTitle_action_keyEquivalent_("Enter Fullscreen", "toggleFullScreen:", "f")
    holder = AppKit.NSMenuItem.alloc().init()
    holder.setSubmenu_(view)
    main.addItem_(holder)

    class Page:
        scripts = []

        def evaluateJavaScript_completionHandler_(self, script, handler):
            self.scripts.append(script)

    zoom = pz.PageZoom(data_dir=lambda: tmp_path)
    zoom._kind, zoom._view = "cocoa", Page()
    assert pz.view_menu(main) == view
    zoom._add_view_items(main)
    zoom._add_view_items(main)  # a second window of the same app finds them there
    titles = [view.itemAtIndex_(i).title() for i in range(view.numberOfItems())]
    assert titles == ["Actual Size", "Zoom In", "Zoom Out", "", "Enter Fullscreen"]
    assert [view.itemAtIndex_(i).keyEquivalent() for i in range(3)] == ["0", "+", "-"]
    for action in zoom._menu_actions:
        action._call()
    assert Page.scripts == [f"window.ouroInterfaceSize && window.ouroInterfaceSize.{name}()"
                            for name in ("reset", "zoomIn", "zoomOut")]
