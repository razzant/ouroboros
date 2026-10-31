"""In-chat image preview through the real SPA: photo tiles, image file cards, the owner's upload.

The mock model calls the real ``send_photo`` and ``send_file`` tools (the document reader test's
``_send_files_mock``), so every image reaches Chat through the supervisor's own handlers: inline
bytes live, the immutable task-artifact route after a reload. A Project room adds the owner's own
upload. ``window.open`` is recorded (a blocked popup raises no page event), so a tile is shown to
open nothing outside the app while the explicit Open actions still go out. Chromium and WebKit, the
desktop client size and a phone, light and dark. The 503 is a Playwright route stub; every other
answer is the real server's."""
from __future__ import annotations

import base64
import json
import os
import struct
import zlib
from pathlib import Path

import pytest

from devtools.benchmarks.common.server_runner import _api
from tests.test_ui_document_reader_browser import (
    LEFT_BEHIND, NARROW, OPEN_ROOM, WIDE, _check_reader_place, _close_with_escape,
    _enter_room, _feed_scroll, _open as _open_document, _reader_place, _send_files_mock, _wait_reader_footer,
)
from tests.test_ui_smoke_playwright import direct_server_with_data  # noqa: F401 - pytest fixture import
from tests.ui_chat_viewport_smoke import _CAPTURE_TEST_SOCKET, set_viewport

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]


def _png(width, height, cell=200):
    """A checkerboard PNG of ``cell`` px squares, stdlib only."""
    def chunk(kind, value):
        return struct.pack("!I", len(value)) + kind + value + struct.pack("!I", zlib.crc32(kind + value) & 0xffffffff)
    colours = (b"\x1f\x5f\x9f", b"\xf2\xa2\x3c")
    rows = [b"\x00" + b"".join(colours[(x // cell + band) % 2] for x in range(width)) for band in (0, 1)]
    data = b"".join(rows[(y // cell) % 2] for y in range(height))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack("!2I5B", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(data)) + chunk(b"IEND", b""))


PHOTOS = {"wide.png": _png(2400, 1600), "tiny.png": _png(40, 24, cell=8)}
FILES = {"chart.png": _png(900, 500, cell=100),
         "vector.svg": b'<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64"><rect width="64" height="64"/></svg>'}
UPLOAD = _png(640, 360, cell=40)

# Every window.open the page makes, without opening anything.
OPEN_RECORDER = """() => {
    window.__opened = [];
    window.open = (...args) => { window.__opened.push(String(args[0] || '')); return null; };
}"""

PREVIEW_STATE = """() => {
    const dialog = document.querySelector('dialog.document-reader[open]');
    if (!dialog) return null;
    const body = dialog.querySelector('.document-reader-body');
    const image = body.querySelector('.document-reader-image');
    const views = dialog.querySelector('.document-reader-views');
    const box = image?.getBoundingClientRect(), area = body.getBoundingClientRect(), rect = dialog.getBoundingClientRect();
    return {
        title: dialog.querySelector('.document-reader-title').textContent,
        meta: dialog.querySelector('.document-reader-meta').textContent,
        label: body.getAttribute('aria-label'), busy: body.getAttribute('aria-busy'),
        status: body.querySelector('.document-reader-status')?.textContent || '',
        retry: Boolean(body.querySelector('[data-reader-action="retry"]')),
        views: views.hidden ? [] : [...views.querySelectorAll('[data-reader-view]')].map((b) => [b.textContent, b.getAttribute('aria-pressed')]),
        src: image?.getAttribute('src') ?? null,
        natural: image ? [image.naturalWidth, image.naturalHeight] : null,
        shown: box ? [box.width, box.height] : null,
        inside: box ? box.left >= area.left - 0.5 && box.right <= area.right + 0.5
            && box.top >= area.top - 0.5 && box.bottom <= area.bottom + 0.5 : null,
        overflow: [body.scrollWidth - body.clientWidth, body.scrollHeight - body.clientHeight],
        scroll: [body.scrollLeft, body.scrollTop],
        active: document.activeElement === body,
        rect: [rect.left, rect.top, rect.width, rect.height],
        opened: window.__opened.slice(),
    };
}"""

# The focused opener by the name the owner sees: a tile's caption or its image's name, a card's file name.
FOCUSED_TILE = """() => {
    const node = document.activeElement;
    if (!node?.matches('.chat-photo-open, .chat-file-card')) return null;
    const named = node.closest('.chat-gallery-item, .chat-file-item')?.querySelector('figcaption, .chat-file-name');
    return (named?.textContent || node.querySelector('img')?.alt || '').trim();
}"""


def _state(page):
    return page.evaluate(PREVIEW_STATE)


def _open(page, opener, feed="#chat-messages", ready=".document-reader-image", at=None):
    """Press ``opener`` as the owner does (at the point ``at`` of a half-hidden one, which
    Playwright's own click would first scroll into view), recording the feed's place at the press."""
    opener.evaluate("""(node, feed) => node.addEventListener('click', () => {
        node.__previewOpeningScroll = document.querySelector(feed).scrollTop;
    }, {capture: true, once: true})""", feed)
    if at:
        page.mouse.click(*at)
    else:
        opener.click()
    page.locator("dialog.document-reader[open]").wait_for(state="visible")
    if ready == ".document-reader-image":
        page.wait_for_function("""() => {
            const image = document.querySelector('dialog.document-reader[open] .document-reader-image');
            return image?.isConnected && image.complete && image.naturalWidth > 0;
        }""")
    else:
        page.locator(f"dialog.document-reader[open] {ready}").wait_for(state="attached")
    state = _state(page)
    state["opening_scroll"] = opener.evaluate("node => node.__previewOpeningScroll")
    return state


def _escape(page):
    page.keyboard.press("Escape")
    page.locator("dialog.document-reader").wait_for(state="detached")


def _tab_to_region(page, limit=12):
    for _ in range(limit):
        page.keyboard.press("Tab")
        if page.evaluate("() => document.activeElement?.classList.contains('document-reader-body')"):
            return
    pytest.fail("Tab did not reach the image region")


def _check_fit(state, natural):
    assert state["natural"] == list(natural) and state["label"] == "Image" and state["busy"] == "false", state
    assert state["views"] == [["Fit", "true"], ["Actual size", "false"]], state["views"]
    assert state["inside"] and state["overflow"][0] <= 1 and state["overflow"][1] <= 1, state
    width, height = state["shown"]
    assert width <= natural[0] + 0.5 and height <= natural[1] + 0.5, "Fit never enlarges"
    assert abs(width / height - natural[0] / natural[1]) < 0.02, "Fit keeps the image's own ratio"


def _check_actual_and_keys(page, natural, *, arrows=True):
    """Actual size, then the keyboard scrolls the region. Playwright's WebKit moves a focused
    scroller only with PageDown/Space, never an arrow key (a plain overflow div alike), so
    ``arrows=False`` there proves the vertical page key alone."""
    page.locator('dialog.document-reader [data-reader-view="actual"]').click()
    state = _state(page)
    assert [round(value) for value in state["shown"]] == list(natural), state
    assert state["views"] == [["Fit", "false"], ["Actual size", "true"]]
    assert state["overflow"][0] > 0 and state["overflow"][1] > 0, "Actual size scrolls the region both ways"
    _tab_to_region(page)
    for key in ("PageDown", *(("ArrowRight",) * 3 if arrows else ())):
        page.keyboard.press(key)
    page.wait_for_function("arrows => { const b = document.querySelector('.document-reader-body');"
                           " return b.scrollTop > 0 && (!arrows || b.scrollLeft > 0); }", arg=arrows)
    return _state(page)


def _deliver(page, monkeypatch, paths, calls, *, tool, captions=None, replies):
    from tests import fixtures_mock_llm

    monkeypatch.setattr(fixtures_mock_llm._Handler, "do_POST", _send_files_mock(paths, calls, tool=tool, captions=captions))
    page.locator("#chat-input").fill(f"Send me the {'photos' if tool == 'send_photo' else 'image files'}.")
    page.locator("#chat-send").click()
    page.wait_for_function("count => [...document.querySelectorAll('#chat-messages .chat-bubble.assistant')]"
                           ".filter((node) => node.textContent.includes('The files are delivered.')).length >= count",
                           arg=replies, timeout=60_000)


@pytest.mark.ui_browser
@pytest.mark.serial
def test_image_preview_in_main_live_and_after_reload(direct_server_with_data, monkeypatch, tmp_path):  # noqa: F811
    from playwright.sync_api import sync_playwright

    root = direct_server_with_data["data_dir"]
    url = direct_server_with_data["url"]
    evidence = Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", str(tmp_path / "evidence")))
    evidence.mkdir(parents=True, exist_ok=True)
    source = root / "preview-src"
    source.mkdir()
    paths = {}
    for name, data in {**PHOTOS, **FILES}.items():
        (source / name).write_bytes(data)
        paths[name] = source / name
    calls, frames = [], {}
    with sync_playwright() as playwright:
        chromium = playwright.chromium.launch()
        try:
            page = chromium.new_page(viewport=WIDE, color_scheme="dark")
            page.add_init_script(f"({_CAPTURE_TEST_SOCKET})()")
            page.add_init_script(f"({OPEN_RECORDER})()")

            def capture(payload):
                value = json.loads(payload) if isinstance(payload, str) and '"download_url"' in payload else {}
                if value.get("type") in ("photo", "document") and value.get("download_url"):
                    frames.setdefault(value.get("filename") or f"photo-{len(frames)}", value)

            page.on("websocket", lambda socket: socket.on("framereceived", capture))
            page.goto(url, wait_until="domcontentloaded")
            page.wait_for_function("() => window.__testSockets?.some(socket => socket.readyState === WebSocket.OPEN)")
            _deliver(page, monkeypatch, [paths["wide.png"], paths["tiny.png"]], calls, tool="send_photo",
                     captions={"wide.png": "Wide chart"}, replies=1)
            _deliver(page, monkeypatch, [paths["chart.png"], paths["vector.svg"]], calls, tool="send_file", replies=2)
            assert sorted(calls) == sorted([*PHOTOS, *FILES]), calls
            page.locator('#chat-messages .chat-live-card [data-phase="done"]').last.wait_for()
            wide = page.locator("#chat-messages .chat-gallery-item").filter(has_text="Wide chart").locator(".chat-photo-open")
            tiny = page.locator("#chat-messages .chat-gallery-item").filter(has_not_text="Wide chart").locator(".chat-photo-open").last
            for tile in (wide, tiny):
                tile.locator("img.chat-photo").evaluate("img => img.decode()")
            assert wide.locator("img").get_attribute("src").startswith("data:image/png;base64,"), "live: the inline bytes"
            wide.scroll_into_view_if_needed()
            page.screenshot(path=str(evidence / "chromium-wide-dark-feed-tiles.png"))

            # A photo tile: the image itself is the button; it previews in the app, Fit first.
            state = _open(page, wide)
            assert state["opened"] == [], "the tile opened nothing outside the app"
            assert (state["title"], state["meta"]) == ("Wide chart", "PNG"), state
            assert state["src"] == wide.locator("img").get_attribute("src"), "the very source the tile shows"
            _check_fit(state, (2400, 1600))
            assert state["active"], "the image region takes focus so keys scroll it at once"
            page.screenshot(path=str(evidence / "chromium-wide-dark-photo-fit.png"))
            actual = _check_actual_and_keys(page, (2400, 1600))
            page.screenshot(path=str(evidence / "chromium-wide-dark-photo-actual.png"))
            assert actual["scroll"][0] > 0 and actual["scroll"][1] > 0
            _escape(page)
            assert page.evaluate(FOCUSED_TILE) == "Wide chart", "focus is back on the tile"
            assert abs(_feed_scroll(page) - state["opening_scroll"]) <= 1, "the chat keeps its place"

            # A tiny photo is never enlarged; Close closes.
            state = _open(page, tiny)
            _check_fit(state, (40, 24))
            assert [round(value) for value in state["shown"]] == [40, 24], state["shown"]
            page.screenshot(path=str(evidence / "chromium-wide-dark-tiny-fit.png"))
            page.locator('dialog.document-reader [data-reader-action="close"]').click()
            page.locator("dialog.document-reader").wait_for(state="detached")

            # A delivered raster file card says View and previews; the backdrop closes it.
            chart = page.locator("#chat-messages .chat-file-card").filter(has_text="chart.png")
            assert chart.locator(".chat-file-more").inner_text() == "View"
            state = _open(page, chart)
            assert state["title"] == "chart.png" and state["meta"].startswith("PNG · ") and state["opened"] == [], state
            _check_fit(state, (900, 500))
            page.mouse.click(4, 4)
            page.locator("dialog.document-reader").wait_for(state="detached")
            # SVG stays a file: the card's Open/Download dialog, no preview.
            page.locator("#chat-messages .chat-file-card").filter(has_text="vector.svg").click()
            page.locator(".chat-file-dialog[open]").wait_for(state="visible")
            assert page.locator("dialog.document-reader").count() == 0
            page.locator('.chat-file-dialog[open] [data-file-action="close"]').click()

            # The explicit actions still go out on purpose: the reader's Open and the tile's menu.
            state = _open(page, wide)
            page.locator('dialog.document-reader [data-reader-action="open"]').click()
            page.wait_for_function("() => window.__opened.length === 1")
            _escape(page)
            wide.locator("xpath=..").locator(".chat-photo-actions summary").click()
            page.locator('.chat-photo-menu:not([hidden]) [data-photo-action="open"]').click()
            page.wait_for_function("() => window.__opened.length === 2")
            opened = page.evaluate("() => window.__opened")
            assert all("/api/tasks/" in entry and "/artifacts/" in entry for entry in opened), opened

            # Replay: the same images from the artifact route; a refused one says so and Retry
            # asks the same address again (the 503 is a route stub).
            artifact = url + frames["chart.png"]["download_url"]
            page.route(artifact, lambda route: route.fulfill(status=503, body=b"unavailable"))
            page.reload(wait_until="domcontentloaded")
            chart = page.locator("#chat-messages .chat-file-card").filter(has_text="chart.png")
            chart.wait_for(state="visible", timeout=30_000)
            state = _open(page, chart, ready=".document-reader-status.is-error")
            assert "Could not load the image." in state["status"] and state["retry"] and state["src"] is None, state
            page.screenshot(path=str(evidence / "chromium-wide-dark-chart-refused.png"))
            page.unroute(artifact)
            requests = []
            page.on("request", lambda request: requests.append(request.url))
            page.locator('dialog.document-reader [data-reader-action="retry"]').click()
            page.wait_for_function("() => document.querySelector('dialog.document-reader .document-reader-image')?.naturalWidth === 900")
            assert artifact in requests and not [entry for entry in requests if "/api/files/" in entry], requests
            _escape(page)
            wide = page.locator("#chat-messages .chat-gallery-item").filter(has_text="Wide chart").locator(".chat-photo-open")
            state = _open(page, wide)
            assert "/artifacts/" in state["src"] and state["src"] == wide.locator("img").get_attribute("src"), state
            _check_fit(state, (2400, 1600))
            _escape(page)

            # Phone, light: a full sheet. The opener is half above the fold and stays where it was.
            set_viewport(page, NARROW["width"], NARROW["height"])
            page.emulate_media(color_scheme="light")
            state = _open(page, wide, at=_half_visible(page, wide))
            assert [round(value) for value in state["rect"]] == [0, 0, NARROW["width"], NARROW["height"]], state["rect"]
            _check_fit(state, (2400, 1600))
            page.screenshot(path=str(evidence / "chromium-narrow-light-photo-fit.png"))
            _escape(page)
            assert abs(_feed_scroll(page) - state["opening_scroll"]) <= 1, "a half-hidden opener is not scrolled into view"
        finally:
            chromium.close()

        webkit = playwright.webkit.launch()
        try:
            page = webkit.new_page(viewport=WIDE, color_scheme="light")
            page.add_init_script(f"({OPEN_RECORDER})()")
            page.goto(url, wait_until="domcontentloaded")
            wide = page.locator("#chat-messages .chat-gallery-item").filter(has_text="Wide chart").locator(".chat-photo-open")
            wide.wait_for(state="visible", timeout=30_000)
            # The room lands at its newest message once its history is read; the press is made on the landed feed.
            page.wait_for_selector('#chat-messages[data-history-hydrated="true"]', state="attached")
            _feed_scroll(page)
            state = _open(page, wide)
            assert state["opened"] == [] and state["title"] == "Wide chart"
            _check_fit(state, (2400, 1600))
            page.screenshot(path=str(evidence / "webkit-wide-light-photo-fit.png"))
            _check_actual_and_keys(page, (2400, 1600), arrows=False)
            page.screenshot(path=str(evidence / "webkit-wide-light-photo-actual.png"))
            _escape(page)
            assert page.evaluate(FOCUSED_TILE) == "Wide chart"
            assert abs(_feed_scroll(page) - state["opening_scroll"]) <= 1
            set_viewport(page, NARROW["width"], NARROW["height"])
            page.emulate_media(color_scheme="dark")
            # The file card lies lower in the conversation, so it can be left half past an edge here.
            chart = page.locator("#chat-messages .chat-file-card").filter(has_text="chart.png")
            state = _open(page, chart, at=_half_visible(page, chart))
            assert [round(value) for value in state["rect"]] == [0, 0, NARROW["width"], NARROW["height"]], state["rect"]
            _check_fit(state, (900, 500))
            page.screenshot(path=str(evidence / "webkit-narrow-dark-chart-fit.png"))
            _escape(page)
            assert abs(_feed_scroll(page) - state["opening_scroll"]) <= 1, "WebKit returns focus without scrolling too"
            assert page.evaluate(FOCUSED_TILE) == "chart.png"
            state = _open(page, wide)
            _check_fit(state, (2400, 1600))
            page.screenshot(path=str(evidence / "webkit-narrow-dark-photo-fit.png"))
            _escape(page)
            assert page.evaluate(FOCUSED_TILE) == "Wide chart"
        finally:
            webkit.close()


def _half_visible(page, opener, feed="#chat-messages"):
    """Scroll the feed so half the opener lies past an edge of the feed (the top one, or the bottom
    one when nothing below can scroll that far) and, once the scroll has settled, return a point on
    its visible part that hit-tests to it (floating chrome such as the composer may cover an edge)."""
    for edge in ("top", "bottom"):
        opener.evaluate("""(node, [feed, edge]) => {
            const root = document.querySelector(feed);
            const box = node.getBoundingClientRect(), view = root.getBoundingClientRect();
            const by = edge === 'top' ? box.top - view.top + box.height / 2 : box.bottom - view.bottom - box.height / 2;
            // The owner's own gesture moves the feed: a bare scroll keeps the room following its newest row, and the
            // phone composer folding as the owner reads back (composer_fold.js) would bring the feed back there.
            root.dispatchEvent(new WheelEvent('wheel', {deltaY: Math.sign(by) || -1}));
            root.scrollTop += by;
        }""", [feed, edge])
        _feed_scroll(page, feed)
        point = opener.evaluate("""(node, [feed, edge]) => {
            const box = node.getBoundingClientRect(), view = document.querySelector(feed).getBoundingClientRect();
            if (edge === 'top' ? !(box.top < view.top) : !(box.bottom > view.bottom)) return null;
            const x = box.left + box.width / 2;
            for (let y = Math.max(box.top, view.top) + 4; y < Math.min(box.bottom, view.bottom) - 4; y += 8) {
                if (node.contains(document.elementFromPoint(x, y))) return [x, y];
            }
            return null;
        }""", [feed, edge])
        if point:
            return point
    pytest.fail("the fixture must leave the opener partly past an edge of the feed")


@pytest.mark.ui_browser
@pytest.mark.serial
def test_owner_upload_preview_in_a_project_room_closes_with_back_and_navigation(direct_server_with_data, tmp_path):  # noqa: F811
    """The owner's uploaded screenshot in a Project room previews from its upload route; Back closes
    the preview and keeps the room; leaving the room closes it and leaves no modal behind."""
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    evidence = Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", str(tmp_path / "evidence")))
    evidence.mkdir(parents=True, exist_ok=True)
    shot = tmp_path / "room-shot.png"
    shot.write_bytes(UPLOAD)
    room = _api(url, "POST", "/api/projects", {"name": "Preview room"})["project"]
    other = _api(url, "POST", "/api/projects", {"name": "Other room"})["project"]
    with sync_playwright() as playwright:
        for engine, viewport, scheme in (("chromium", WIDE, "light"), ("webkit", NARROW, "dark")):
            browser = getattr(playwright, engine).launch()
            try:
                page = browser.new_page(viewport=viewport, color_scheme=scheme)
                page.add_init_script(f"({OPEN_RECORDER})()")
                page.goto(url, wait_until="domcontentloaded")
                feed = _enter_room(page, room)
                tiles = page.locator(f"{feed} .chat-bubble.user.has-attachments .chat-photo-open")
                if engine == "chromium":
                    panel = page.locator("#project-panel")
                    panel.locator("input[type=file]").set_input_files([str(shot)])
                    panel.locator("textarea").fill("Room screenshot")
                    panel.locator(".chat-send-inline").click()
                tile = tiles.last
                tile.wait_for(state="visible", timeout=30_000)
                tile.locator("img").evaluate("img => img.decode()")
                state = _open(page, tile, feed=feed)
                assert state["opened"] == [] and state["title"] == "room-shot.png" and state["meta"].startswith("PNG · "), state
                assert state["src"] == tile.locator("img").get_attribute("src") and "/api/files/download?upload=" in state["src"]
                _check_fit(state, (640, 360))
                page.screenshot(path=str(evidence / f"{engine}-{scheme}-room-upload-fit.png"))
                # Back is the dialog's Escape: the preview closes and the room stays.
                page.evaluate("() => history.back()")
                page.locator("dialog.document-reader").wait_for(state="detached")
                page.locator(feed).wait_for(state="visible")
                assert page.evaluate(FOCUSED_TILE) == "room-shot.png", page.evaluate(
                    "() => document.activeElement?.outerHTML.slice(0, 200)")
                # Leaving the room with the preview open takes the preview along.
                _open(page, tile, feed=feed)
                page.evaluate(OPEN_ROOM, other)
                page.locator(f'#pchat-{other["id"]}-messages').wait_for(state="visible", timeout=30_000)
                assert page.evaluate(LEFT_BEHIND, feed) == {
                    "readers": 0, "open_dialogs": 0, "room": False, "focus_connected": True}
                page.locator(f'[id="pchat-{other["id"]}-input"]').click(timeout=5_000)
            finally:
                browser.close()


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_reader_keeps_visible_place_when_a_late_row_grows_the_feed(direct_server_with_data, monkeypatch, tmp_path, engine):  # noqa: F811
    """A late socket row above an open reader legitimately changes scrollTop.
    Assert the visible card's position instead of mistaking that compensation
    for a close/focus regression; unchanged-layout checks still require both.
    The frame is synthetic, delivered through the real SPA socket decoder.
    """
    from playwright.sync_api import sync_playwright

    source = direct_server_with_data["data_dir"] / "anchor.md"
    source.write_text("# Anchored reader\n\nThe conversation stays in place.\n", encoding="utf-8")
    calls = []
    evidence = Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", str(tmp_path / "evidence")))
    evidence.mkdir(parents=True, exist_ok=True)
    records = {}
    with sync_playwright() as playwright:
        browser = getattr(playwright, engine).launch()
        try:
            page = browser.new_page(viewport=WIDE)
            page.add_init_script(f"({_CAPTURE_TEST_SOCKET})()")
            page.goto(direct_server_with_data["url"], wait_until="domcontentloaded")
            page.wait_for_function("() => window.__testSockets?.some(s => s.readyState === WebSocket.OPEN)")
            _deliver(page, monkeypatch, [source], calls, tool="send_file", replies=1)
            _wait_reader_footer(page)
            records["footer"] = page.locator('#chat-messages .chat-live-card [data-live-meta]').last.inner_text()
            emit = """frames => {
                const socket = window.__testSockets.find(s => s.readyState === WebSocket.OPEN);
                for (const frame of frames) socket.dispatchEvent(new MessageEvent('message', {data: JSON.stringify(frame)}));
            }"""
            for label, year in (("Earlier", 1999), ("Following", 2099)):
                page.evaluate(emit, [{"type": "chat", "role": "assistant", "chat_id": 1,
                    "content": f"{label} conversation {i}", "ts": f"{year}-01-01T00:00:{i:02d}Z"} for i in range(24)])
            card = page.locator("#chat-messages .chat-file-card").filter(has_text="anchor.md")
            card.scroll_into_view_if_needed()
            # A reader away from the live edge: no native anchoring can hide a
            # missing application-owned viewport compensation.
            page.add_style_tag(content="#chat-messages { overflow-anchor: none; }")
            box = card.bounding_box()
            page.mouse.move(box["x"] + 20, box["y"] + 20)
            page.mouse.wheel(0, -40)
            _feed_scroll(page)
            state = _open_document(page, "anchor.md", ready=".document-reader-markdown")
            before = state["opening_place"]
            records["before"] = before
            assert before["scroll"] > 0
            _close_with_escape(page)
            records["unchanged"] = _reader_place(page, "anchor.md")
            assert records["unchanged"]["height"] == before["height"], records
            _check_reader_place(page, "anchor.md", state, current=records["unchanged"])
            state = _open_document(page, "anchor.md", ready=".document-reader-markdown")
            before = records["before_growth"] = state["opening_place"]
            page.evaluate(emit, [{"type": "chat", "role": "assistant", "chat_id": 1,
                "content": "Late history above the visible window. " * 24, "ts": "1998-01-01T00:00:00Z"}])
            page.wait_for_function("h => document.querySelector('#chat-messages').scrollHeight > h", arg=before["height"])
            _close_with_escape(page)
            _check_reader_place(page, "anchor.md", state)
            records["after_growth"] = _reader_place(page, "anchor.md")
            page.screenshot(path=str(evidence / f"{engine}-late-row-preserved.png"))
            assert abs(_feed_scroll(page) - before["scroll"]) > 1, "the old raw-offset oracle would falsely fail"
            assert page.evaluate("() => document.activeElement?.classList.contains('chat-file-card')")
            # A real 16 px reveal must still fail even though the feed grew:
            # this oracle does not excuse visible jumps on changed geometry.
            shifted = _reader_place(page, "anchor.md", scroll_by=16)
            records["shifted"] = shifted
            assert abs(shifted["scroll"] - shifted["previousScroll"] - 16) <= 1
            with pytest.raises(AssertionError):
                _check_reader_place(page, "anchor.md", state, current=shifted)
        finally:
            (evidence / f"{engine}-late-row.json").write_text(json.dumps(records, indent=2) + "\n")
            browser.close()


INTERMEDIATE = {"width": 621, "height": 844}
LONG_MD = "quarterly-delivery-notes-for-the-owner-with-a-long-descriptive-name-2026-10-19.md"
LONG_TXT = "plain-text-checklist-from-the-owner.txt"
PORTRAIT = "owner-portrait-screenshot-of-the-settings-page-with-a-long-descriptive-name.png"
TINY = "tiny-owner-icon.png"
OWNER_MD = "# Owner notes\n\nThe first line\ncontinues here.\n"

# The reader's head, as the owner sees it: every control hit-tests to itself inside the window,
# none covers the name, and the whole name is laid out (wrapped), never cut to an ellipsis.
HEAD_STATE = """() => {
    const dialog = document.querySelector('dialog.document-reader[open]');
    const rect = (node) => { const b = node.getBoundingClientRect(); return [b.left, b.top, b.right, b.bottom]; };
    const title = dialog.querySelector('.document-reader-title');
    const buttons = [...dialog.querySelectorAll('.document-reader-head button')].filter((node) => node.getClientRects().length);
    return {
        viewport: [innerWidth, innerHeight], dialog: rect(dialog), head: rect(dialog.querySelector('.document-reader-head')),
        identity: rect(dialog.querySelector('.document-reader-identity')), title: rect(title),
        body: rect(dialog.querySelector('.document-reader-body')), name: title.textContent,
        cut: title.scrollWidth > title.clientWidth + 1 || title.scrollHeight > title.clientHeight + 1,
        wrap: getComputedStyle(title).whiteSpace,
        buttons: buttons.map((node) => { const [l, t, r, b] = rect(node);
            return { label: node.textContent.trim(), box: [l, t, r, b],
                hit: node.contains(document.elementFromPoint((l + r) / 2, (t + b) / 2)) }; }),
    };
}"""


def _overlap(a, b):
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def _check_head(state, name, labels, where):
    width, height = state["viewport"]
    assert state["name"] == name, where
    assert not state["cut"] and state["wrap"] != "nowrap", f"{where}: the whole name is laid out, not cut: {state}"
    assert [button["label"] for button in state["buttons"]] == labels, (where, state["buttons"])
    for button in state["buttons"]:
        left, top, right, bottom = button["box"]
        assert button["hit"] and left >= 0 and top >= 0 and right <= width + 0.5 and bottom <= height + 0.5, (where, button)
        assert _overlap(button["box"], state["identity"]) == 0, f"{where}: {button['label']} covers the name: {state}"
    identity = state["identity"][2] - state["identity"][0]
    assert identity >= 200, f"{where}: the actions wrap before the name is squeezed ({identity:.0f}px): {state}"
    body = state["body"][3] - state["body"][1]
    assert state["body"][1] >= state["head"][3] - 0.5 and body >= 0.4 * (state["dialog"][3] - state["dialog"][1]), \
        f"{where}: the reading region stays reachable under a long name: {state}"


def _upload(page, files, count):
    page.locator("#chat-file-input").set_input_files(files)
    page.wait_for_function("n => document.querySelectorAll('#chat-attachment-preview .attach-badge').length === n", arg=len(files))
    page.locator("#chat-send").click()
    page.wait_for_function("n => document.querySelectorAll('#chat-messages .chat-bubble.user.has-attachments').length === n",
                           arg=count, timeout=30_000)


def _hidden_above(page, opener, offset, feed="#chat-messages"):
    """Scroll the feed so ``offset`` px of the opener lie above the feed's top edge, and return a
    point on its visible part that hit-tests to it (the press the owner makes)."""
    opener.evaluate("""(node, [feed, offset]) => {
        const root = document.querySelector(feed);
        root.scrollTop += node.getBoundingClientRect().top - root.getBoundingClientRect().top + offset;
    }""", [feed, offset])
    _feed_scroll(page, feed)
    facts = opener.evaluate("""(node, feed) => {
        const box = node.getBoundingClientRect(), view = document.querySelector(feed).getBoundingClientRect();
        const x = box.left + box.width / 2, top = Math.max(box.top, view.top), bottom = Math.min(box.bottom, view.bottom);
        const ys = [(top + bottom) / 2];
        for (let y = top + 4; y < bottom - 4; y += 8) ys.push(y);
        const y = ys.find((value) => node.contains(document.elementFromPoint(x, value)));
        return { hidden: view.top - box.top, height: box.height, top: box.top, point: y === undefined ? null : [x, y] };
    }""", feed)
    assert abs(facts["hidden"] - offset) <= 1 and facts["point"], f"the fixture must hide {offset}px of the opener: {facts}"
    return facts


FOCUSED_TRIGGER = """() => {
    const node = document.activeElement;
    return node?.matches('.chat-photo-actions summary') ? node.closest('.chat-gallery-item').querySelector('img').alt : null;
}"""


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_owner_files_and_photos_read_in_place_with_a_head_that_wraps(direct_server_with_data, tmp_path, engine):  # noqa: F811
    """The owner's own Markdown and text read in the reader like a delivered file; the head's
    actions take their own line before the long name is squeezed, and the whole name is shown,
    at the desktop size, just above the phone breakpoint and on a phone, in both themes. The
    owner's 320px photo partly above the feed's top edge opens from a press on its visible part
    and Escape and Close leave the conversation where it was. A tiny photo, whose `•••`
    covers its centre, views from that menu by pointer, keyboard and touch."""
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    evidence = Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", str(tmp_path / "evidence")))
    evidence.mkdir(parents=True, exist_ok=True)
    record = {}
    with sync_playwright() as playwright:
        browser = getattr(playwright, engine).launch()
        try:
            page = browser.new_page(viewport=WIDE, color_scheme="light")
            page.add_init_script(f"({_CAPTURE_TEST_SOCKET})()")
            page.add_init_script(f"({OPEN_RECORDER})()")
            page.goto(url, wait_until="domcontentloaded")
            page.wait_for_function("() => window.__testSockets?.some(s => s.readyState === WebSocket.OPEN)")
            _upload(page, [{"name": PORTRAIT, "mimeType": "image/png", "buffer": _png(240, 320, cell=40)}], 1)
            _upload(page, [{"name": LONG_MD, "mimeType": "text/markdown", "buffer": OWNER_MD.encode()},
                           {"name": LONG_TXT, "mimeType": "text/plain", "buffer": b"first\tline\nsecond line\n"}], 2)
            _upload(page, [{"name": TINY, "mimeType": "image/png", "buffer": _png(8, 8, cell=2)}], 3)
            page.evaluate("""frames => {
                const socket = window.__testSockets.find(s => s.readyState === WebSocket.OPEN);
                for (const frame of frames) socket.dispatchEvent(new MessageEvent('message', {data: JSON.stringify(frame)}));
            }""", [{"type": "chat", "role": "assistant", "chat_id": 1, "content": f"Following conversation {i}",
                    "ts": f"2099-01-01T00:00:{i:02d}Z"} for i in range(24)])
            feed = "#chat-messages"
            cards = page.locator(f"{feed} .chat-bubble.user.has-attachments .chat-file-card")
            md, txt = cards.filter(has_text=LONG_MD), cards.filter(has_text=LONG_TXT)
            assert [md.locator(".chat-file-more").inner_text(), txt.locator(".chat-file-more").inner_text()] == ["Read", "Read"]

            # The owner's Markdown reads as a document from its own upload; its text file monospaced.
            requests = []
            page.on("request", lambda request: requests.append(request.url))
            md.click()
            page.locator("dialog.document-reader[open] .document-reader-markdown").wait_for()
            assert page.locator("dialog.document-reader .document-reader-markdown h1").inner_text() == "Owner notes"
            assert page.locator(".chat-file-dialog[open]").count() == 0 and page.evaluate("() => window.__opened") == []
            assert [entry for entry in requests if "/api/files/download?upload=" in entry], requests
            for viewport in (WIDE, INTERMEDIATE, NARROW):
                for scheme in ("light", "dark"):
                    page.set_viewport_size(viewport)
                    page.emulate_media(color_scheme=scheme)
                    where = f"{engine}-{viewport['width']}-{scheme}-owner-md"
                    state = page.evaluate(HEAD_STATE)
                    record[where] = state
                    _check_head(state, LONG_MD, ["Formatted", "Source", "Open", "Download", "Close"], where)
                    page.screenshot(path=str(evidence / f"{where}-head.png"))
            _escape(page)
            assert page.evaluate("name => document.activeElement?.closest('.chat-file-card')?.textContent.includes(name)", LONG_MD)
            page.set_viewport_size(WIDE)
            txt.click()
            page.locator("dialog.document-reader[open] .document-reader-source").wait_for()
            assert page.locator("dialog.document-reader .document-reader-source").inner_text().startswith("first\tline")
            _escape(page)

            # The photo's head carries the image views and its Copy too.
            portrait = page.locator(f'{feed} .chat-gallery-item:has(img[alt="{PORTRAIT}"]) .chat-photo-open')
            portrait.locator("img").evaluate("img => img.decode()")
            state = _open(page, portrait)
            _check_fit(state, (240, 320))
            for viewport in (WIDE, INTERMEDIATE, NARROW):
                for scheme in ("dark", "light"):
                    page.set_viewport_size(viewport)
                    page.emulate_media(color_scheme=scheme)
                    where = f"{engine}-{viewport['width']}-{scheme}-owner-photo"
                    state = page.evaluate(HEAD_STATE)
                    record[where] = state
                    _check_head(state, PORTRAIT, ["Fit", "Actual size", "Copy", "Open", "Download", "Close"], where)
                    page.screenshot(path=str(evidence / f"{where}-head.png"))
            _escape(page)

            # The exact case: on a phone the owner's 320px photo lies partly above the feed's top.
            page.set_viewport_size(NARROW)
            page.emulate_media(color_scheme="dark")
            assert abs(portrait.bounding_box()["height"] - 320) <= 1, portrait.bounding_box()
            for offset, close in ((40, "escape"), (160, "escape"), (240, "escape"), (160, "close"), (240, "close")):
                facts = _hidden_above(page, portrait, offset)
                state = _open(page, portrait, at=facts["point"])
                assert state["opened"] == [] and state["title"] == PORTRAIT, state
                if (offset, close) == (160, "escape"):
                    page.screenshot(path=str(evidence / f"{engine}-390-dark-owner-photo-hidden-160-open.png"))
                if close == "escape":
                    _escape(page)
                else:
                    page.locator('dialog.document-reader [data-reader-action="close"]').click()
                    page.locator("dialog.document-reader").wait_for(state="detached")
                after = _feed_scroll(page)
                now = portrait.evaluate("node => node.getBoundingClientRect().top")
                record[f"{engine}-hidden-{offset}-{close}"] = {**facts, "opening_scroll": state["opening_scroll"],
                                                              "after_scroll": after, "after_top": now}
                assert abs(after - state["opening_scroll"]) <= 1 and abs(now - facts["top"]) <= 1, \
                    f"{offset}px hidden, {close}: the chat moved: {record[f'{engine}-hidden-{offset}-{close}']}"
                assert page.evaluate(FOCUSED_TILE) == PORTRAIT, (offset, close)
            page.screenshot(path=str(evidence / f"{engine}-390-dark-owner-photo-after-close.png"))

            # A tiny photo: its always-present ••• covers the centre, so its menu views it.
            tiny = page.locator(f'{feed} .chat-gallery-item:has(img[alt="{TINY}"])')
            tiny.evaluate("el => el.scrollIntoView({ block: 'center' })")
            tiny.locator("img").evaluate("img => img.decode()")
            centre = tiny.evaluate("""item => { const b = item.querySelector('.chat-photo-open').getBoundingClientRect();
                const hit = document.elementFromPoint(b.left + b.width / 2, b.top + b.height / 2);
                return { box: [b.width, b.height], covered: item.querySelector('.chat-photo-actions summary').contains(hit) }; }""")
            assert centre["covered"], f"the genuine case: ••• covers the tiny photo's centre: {centre}"
            record[f"{engine}-tiny-centre"] = centre
            summary = tiny.locator(".chat-photo-actions summary")
            menu = page.locator(".chat-photo-menu:not([hidden])")
            for how in ("pointer", "keyboard"):
                before = _feed_scroll(page)
                if how == "pointer":
                    summary.click()
                    menu.locator('[data-photo-action="view"]').click()
                else:
                    summary.focus()
                    page.keyboard.press("Enter")
                    menu.wait_for(state="visible")
                    assert page.evaluate("() => document.activeElement?.dataset.photoAction") == "view", "View comes first"
                    page.keyboard.press("Enter")
                page.locator("dialog.document-reader[open]").wait_for(state="visible")
                page.wait_for_function("() => document.querySelector('dialog.document-reader[open] .document-reader-image')?.naturalWidth > 0")
                state = _state(page)
                _check_fit(state, (8, 8))
                assert state["title"] == TINY and [round(value) for value in state["shown"]] == [8, 8], state
                quiet = page.evaluate("() => document.querySelector('.document-reader-body').classList.contains('is-quiet-focus')")
                assert quiet == (how == "pointer"), "a keyboard opening shows its focus ring at once"
                if how == "pointer":
                    page.screenshot(path=str(evidence / f"{engine}-390-dark-tiny-view-from-menu.png"))
                _escape(page)
                assert page.evaluate(FOCUSED_TRIGGER) == TINY, f"{how}: focus is back on the ••• that opened it"
                assert abs(_feed_scroll(page) - before) <= 1

            # Touch: the same menu, tapped, on a phone.
            phone = browser.new_context(viewport=NARROW, has_touch=True, is_mobile=engine == "chromium", color_scheme="light")
            try:
                touch = phone.new_page()
                touch.add_init_script(f"({OPEN_RECORDER})()")
                touch.goto(url, wait_until="domcontentloaded")
                item = touch.locator(f'{feed} .chat-gallery-item:has(img[alt="{TINY}"])')
                item.wait_for(state="visible", timeout=30_000)
                # The room lands at its newest message once its history is read, and lands there again as each of its
                # photos loads (chat.js reflows on an image's load while the room follows its newest message): the
                # place is taken after that, or a photo that loads late moves the ••• under its open menu, which closes.
                touch.wait_for_selector(f'{feed}[data-history-hydrated="true"]', state="attached")
                touch.wait_for_function("feed => [...document.querySelectorAll(feed + ' img')].every((img) => img.complete)", arg=feed)
                _feed_scroll(touch)
                # The owner reads back to the photo with their own gesture, as a swipe does: following ends, so a reflow
                # while the preview opens or closes no longer lays the room out at its newest message again.
                touch.locator(feed).evaluate("n => n.dispatchEvent(new WheelEvent('wheel', {deltaY: -1}))")
                item.evaluate("el => el.scrollIntoView({ block: 'center' })")
                item.locator("img").evaluate("img => img.decode()")
                before = _feed_scroll(touch)
                item.locator(".chat-photo-actions summary").tap()
                touch.locator('.chat-photo-menu:not([hidden]) [data-photo-action="view"]').tap()
                touch.wait_for_function("() => document.querySelector('dialog.document-reader[open] .document-reader-image')?.naturalWidth > 0")
                assert touch.evaluate(PREVIEW_STATE)["title"] == TINY
                touch.screenshot(path=str(evidence / f"{engine}-390-light-touch-tiny-view.png"))
                touch.locator('dialog.document-reader [data-reader-action="close"]').tap()
                touch.locator("dialog.document-reader").wait_for(state="detached")
                assert abs(_feed_scroll(touch) - before) <= 1
            finally:
                phone.close()

            # The reader's Copy is the menu's: the image itself on the clipboard (Chromium grants it).
            if engine == "chromium":
                page.context.grant_permissions(["clipboard-read", "clipboard-write"], origin=url)
                state = _open(page, portrait)
                page.locator('dialog.document-reader [data-reader-action="copy"]').click()
                page.locator(".toast", has_text="Image copied.").wait_for(timeout=10_000)
                types = page.evaluate("async () => (await navigator.clipboard.read()).flatMap((item) => item.types)")
                assert "image/png" in types, types
                _escape(page)
                # A photo with only its inline bytes keeps the explicit Open of its menu.
                page.evaluate("""frame => window.__testSockets.find(s => s.readyState === WebSocket.OPEN)
                    .dispatchEvent(new MessageEvent('message', { data: JSON.stringify(frame) }))""", {
                    "type": "photo", "role": "assistant", "chat_id": 1, "mime": "image/png", "caption": "Ephemeral photo",
                    "image_base64": base64.b64encode(_png(64, 48, cell=8)).decode("ascii"),
                    "ts": "2099-02-01T00:00:00Z"})
                ephemeral = page.locator(f"{feed} .chat-gallery-item").filter(has_text="Ephemeral photo").locator(".chat-photo-open")
                ephemeral.scroll_into_view_if_needed()
                state = _open(page, ephemeral)
                assert state["src"].startswith("data:image/png;base64,"), state["src"]
                page.locator('dialog.document-reader [data-reader-action="open"]').click()
                page.wait_for_function("() => window.__opened.length === 1")
                assert page.evaluate("() => window.__opened[0]").startswith("data:image/png;base64,")
                _escape(page)
        finally:
            (evidence / f"{engine}-owner-reader-geometry.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
            browser.close()
