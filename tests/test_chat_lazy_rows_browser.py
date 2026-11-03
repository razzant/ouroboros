"""Lazy rows in the real app over real retained history (owner 18.10, 5A; rows asleep at rest, owner 20.10, 3A).

While a feed's width keeps moving (the sidebar's or a Project panel's handle dragged frame by frame), the rows off
the screen keep the size they were laid out at and the browser lays out only the rows near the screen (style.css,
chat_reading_position.js). Opening a room, a scrollbar appearing and a single window jump are laid out at once, as
before. Once the width holds, every row is laid out again. At rest a row sleeps (skipped while far from the screen,
chat_row_sleep.js) only once it was laid out at the feed's width, so every row asleep keeps its own size: the passage
being read is where it was, scrolling up through the loaded history moves every row by exactly the scroll, an older
page lands above the passage without moving it, and a reader at the newest message stays there. A row lazy at rest
before it was laid out fails this in both engines: in WebKit a row taking its real size only as it nears the screen
moves the passage being read, and Chromium ends short of the newest message.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from tests.test_chat_history_paging_browser import _FRAMES, MAIN, _append, _assert_at_newest, _idle, _open, _step
from tests.test_ui_smoke_playwright import direct_server_with_data as direct_server_with_data

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]

_WORDS = ("history keeps every passage where the reader left it while the column narrows and widens again "
          "and the rows far above wait with the size they last had").split()

# Every time a feed is marked lazy or unmarked, from the first parse on.
_RECORD_MARKS = """() => {
    window.__lazyMarks = [];
    new MutationObserver(records => { for (const record of records) window.__lazyMarks.push(
        {id: record.target.id, on: record.target.hasAttribute('data-resizing')}); })
        .observe(document, {subtree: true, attributes: true, attributeFilter: ['data-resizing']});
}"""
# What the feed's rows are doing: marked for a moving width, rows under the lazy rule, and rows whose contents the
# browser actually skips (a first child that is shown, yet not rendered because its row is lazy), of all rows.
_LAZY = """feed => {
    const root = document.querySelector(feed), rows = [...root.children];
    const skipped = row => { const child = row.firstElementChild;
        return Boolean(child?.checkVisibility() && !child.checkVisibility({contentVisibilityAuto: true})); };
    // Every row asleep keeps its own size: woken for a moment, none takes another (the marks are put back as they were).
    const asleep = rows.filter(row => row.hasAttribute('data-laid'));
    const kept = asleep.map(row => row.getBoundingClientRect().height);
    asleep.forEach(row => row.removeAttribute('data-laid'));
    const own = asleep.map(row => row.getBoundingClientRect().height);
    asleep.forEach(row => row.setAttribute('data-laid', ''));
    return {marked: root.hasAttribute('data-resizing'), rows: rows.length,
        lazy: rows.filter(row => getComputedStyle(row).contentVisibility === 'auto').length,
        skipped: rows.filter(skipped).length, asleep: asleep.length,
        resized: kept.filter((height, index) => Math.abs(height - own[index]) > 0.5).length,
        marks: window.__lazyMarks.filter(mark => mark.id === root.id).length};
}"""
# The real resize handle dragged one step per frame there and back (app.js writes the width once per frame), with
# the feed's state sampled every frame.
_DRAG = """async ([feed, handleSelector, steps]) => {
    const root = document.querySelector(feed), handle = document.querySelector(handleSelector);
    const frame = () => new Promise(resolve => requestAnimationFrame(resolve));
    const skipped = row => { const child = row.firstElementChild;
        return Boolean(child?.checkVisibility() && !child.checkVisibility({contentVisibilityAuto: true})); };
    const box = handle.getBoundingClientRect(), y = box.top + Math.min(200, box.height / 2);
    let x = box.left + box.width / 2;
    const send = type => handle.dispatchEvent(new PointerEvent(type, {bubbles: true, cancelable: true, clientX: x,
        clientY: y, pointerId: 7, pointerType: 'mouse', isPrimary: true, button: 0, buttons: type === 'pointerup' ? 0 : 1}));
    const seen = [];
    send('pointerdown');
    for (let i = 0; i < steps; i += 1) {
        x += i < steps / 2 ? 5 : -5;
        send('pointermove');
        await frame();
        seen.push({marked: root.hasAttribute('data-resizing'), skipped: [...root.children].filter(skipped).length,
            width: root.offsetWidth});
    }
    send('pointerup');
    return seen;
}"""
# The row the reading owner anchors on: the first one whose bottom is inside the feed's box.
_PASSAGE = """feed => {
    const root = document.querySelector(feed), top = root.getBoundingClientRect().top;
    const node = [...root.querySelectorAll(':scope > [data-history-id]')].find(n => n.getBoundingClientRect().bottom > top);
    return {id: node.dataset.historyId, offset: node.getBoundingClientRect().top - top, rows: root.children.length};
}"""
_OFFSET = """([feed, id]) => {
    const root = document.querySelector(feed);
    const node = [...root.querySelectorAll(':scope > [data-history-id]')].find(n => n.dataset.historyId === id);
    return node ? node.getBoundingClientRect().top - root.getBoundingClientRect().top : null;
}"""
# Up through what is loaded, a step at a time, without a wheel (a gesture at the top would read an older page):
# the first row starting inside the feed must move by exactly the distance scrolled, two frames later.
_CLIMB = """async ([feed, step]) => {
    const root = document.querySelector(feed), frame = () => new Promise(r => requestAnimationFrame(r));
    const misses = [];
    let steps = 0;
    while (root.scrollTop > 0 && steps < 400) {
        const top = root.getBoundingClientRect().top;
        const row = [...root.children].find(n => n.getBoundingClientRect().top >= top);
        const before = row.getBoundingClientRect().top, from = root.scrollTop;
        root.scrollTop = Math.max(0, from - step);
        const moved = from - root.scrollTop;
        await frame(); await frame();
        const miss = row.getBoundingClientRect().top - before - moved;
        if (Math.abs(miss) > 1) misses.push({row: row.dataset.historyId || row.className, miss, scrollTop: root.scrollTop});
        steps += 1;
    }
    return {steps, misses, top: root.scrollTop};
}"""


def _seed(root, chat_id=1, count=420):
    """Owner messages from one line to several paragraphs: rows far from the 160px a never-laid-out row takes."""
    start = datetime(2026, 9, 1, tzinfo=timezone.utc)
    rows = []
    for index in range(count):
        words = " ".join(_WORDS[(index + k) % len(_WORDS)] for k in range(4 + (index * 37) % 170))
        text = f"Lazy row {index:04d}. {words}"
        if index % 7 == 0:
            text += "\n\n" + " ".join(_WORDS) + "\n\n" + " ".join(reversed(_WORDS))
        rows.append({"direction": "in", "chat_id": chat_id, "ts": (start + timedelta(minutes=index)).isoformat(),
                     "client_message_id": f"lazy-row-{chat_id}-{index}", "text": text})
    _append(root, "chat", rows)


def _read_at(page, feed, fraction):
    """The reader turns back into history and stops with a passage `fraction` of the way down the loaded rows."""
    page.locator(feed).evaluate("""(root, fraction) => {
        root.dispatchEvent(new WheelEvent('wheel', {deltaY: -1}));
        root.scrollTop = Math.round((root.scrollHeight - root.clientHeight) * fraction);
    }""", fraction)
    page.evaluate(_FRAMES)
    page.evaluate(_FRAMES)
    return page.evaluate(_PASSAGE, feed)


def _held(page, feed):
    """At rest: the width held, the mark is gone, every row was laid out again and the far rows sleep again."""
    page.wait_for_function("feed => !document.querySelector(feed).hasAttribute('data-resizing')", arg=feed,
                           timeout=10_000)
    page.wait_for_function("""feed => { const rows = [...document.querySelector(feed).children];
        return rows.length > 0 && rows.filter(row => row.hasAttribute('data-laid')).length > rows.length / 2; }""",
                           arg=feed, timeout=10_000)
    page.evaluate(_FRAMES)
    state = page.evaluate(_LAZY, feed)
    assert state["resized"] == 0, ("every row asleep keeps its own size", state)
    return state


def _jump(page, feed, passage, width):
    """One window jump is laid out at once: the feed is never marked, the passage keeps its place."""
    marks = page.evaluate(_LAZY, feed)["marks"]
    page.set_viewport_size({"width": width, "height": 850})
    page.wait_for_function("""([feed, id, offset]) => {
        const root = document.querySelector(feed);
        const node = [...root.querySelectorAll(':scope > [data-history-id]')].find(n => n.dataset.historyId === id);
        return node && Math.abs(node.getBoundingClientRect().top - root.getBoundingClientRect().top - offset) <= 2;
    }""", arg=[feed, passage["id"], passage["offset"]], timeout=5_000)
    page.evaluate(_FRAMES)
    page.evaluate(_FRAMES)
    state = _held(page, feed)
    assert state["marks"] == marks, ("a single jump is laid out at once, never as a moving width", state)


def _drag(page, feed, handle, passage, rows):
    """The handle dragged frame by frame: the rows far off the screen are skipped meanwhile; once the width holds no
    row is lazy and the passage is where it was read."""
    seen = page.evaluate(_DRAG, [feed, handle, 40])
    assert len({frame["width"] for frame in seen}) > 10, ("the feed's width followed the handle", seen)
    assert any(frame["marked"] for frame in seen), seen
    assert max(frame["skipped"] for frame in seen) > rows // 2, ("the browser skipped the rows off the screen", seen)
    after = _held(page, feed)
    assert after["lazy"] == after["asleep"] and not after["marked"], ("at rest only rows asleep are lazy", after)
    placed = page.evaluate(_OFFSET, [feed, passage["id"]])
    assert abs(placed - passage["offset"]) <= 1, (passage, placed)


@pytest.mark.parametrize("browser_engine", ["chromium", "webkit"])
def test_lazy_rows_keep_the_passage_history_scroll_and_older_pages_exact(direct_server_with_data, browser_engine):
    from playwright.sync_api import sync_playwright

    _seed(direct_server_with_data["data_dir"])
    with sync_playwright() as pw:
        browser = getattr(pw, browser_engine).launch(headless=True)
        try:
            page = browser.new_page(viewport={"width": 1280, "height": 850})
            page.add_init_script(f"({_RECORD_MARKS})()")
            _open(page, direct_server_with_data["url"])
            rest = _held(page, MAIN)
            assert rest["rows"] > 100 and rest["marks"] == 0, (
                "opening the room (its history filling the feed, a scrollbar appearing) is no moving width", rest)
            assert rest["lazy"] == rest["asleep"] and rest["skipped"] > rest["rows"] // 2, (
                "the rows far from the screen sleep, each at the size it was laid out at", rest)

            passage = _read_at(page, MAIN, 0.55)
            _jump(page, MAIN, passage, 1180)
            _drag(page, MAIN, "#primary-sidebar > .resize-handle", passage, rest["rows"])

            # Every row was laid out at the held width: nothing above the passage changes size as it nears the
            # screen, so scrolling up moves each row by exactly the scroll.
            climb = page.evaluate(_CLIMB, [MAIN, 240])
            assert climb["steps"] > 10 and climb["top"] == 0 and not climb["misses"], climb

            # An older page lands above the passage being read and moves nothing on the screen; its own rows
            # scroll exactly too.
            page.locator(MAIN).evaluate("root => { root.scrollTop = 600; }")
            page.evaluate(_FRAMES)
            reading = page.evaluate(_PASSAGE, MAIN)
            _step(page, MAIN)
            page.evaluate(_FRAMES)
            grown = page.evaluate(_PASSAGE, MAIN)
            assert grown["rows"] > reading["rows"] + 100, (reading, grown)
            assert abs(page.evaluate(_OFFSET, [MAIN, reading["id"]]) - reading["offset"]) <= 1, (reading, grown)
            climb = page.evaluate(_CLIMB, [MAIN, 240])
            assert climb["top"] == 0 and not climb["misses"], climb

            # Following the newest message, a drag keeps the reader there once the width holds.
            page.locator("#chat-scroll-bottom").evaluate("node => node.click()")
            _assert_at_newest(page, MAIN, "Lazy row 0419")
            page.evaluate(_DRAG, [MAIN, "#primary-sidebar > .resize-handle", 30])
            _held(page, MAIN)
            _assert_at_newest(page, MAIN, "Lazy row 0419")
            _idle(page, MAIN)
        finally:
            browser.close()


@pytest.mark.parametrize("browser_engine", ["chromium", "webkit"])
def test_a_project_panel_feed_is_lazy_only_while_its_width_keeps_moving(direct_server_with_data, browser_engine):
    """The Project panel's own handle: the same lazy rows while it is dragged, the same exact passage and history
    once the width holds."""
    from playwright.sync_api import sync_playwright

    from ouroboros.projects_registry import create_project
    from tests.test_chat_history_paging_browser import _open_project

    root = direct_server_with_data["data_dir"]
    project = create_project(root, "lazy-rows-room", name="Lazy rows room")
    _seed(root, project["chat_id"], 200)
    with sync_playwright() as pw:
        browser = getattr(pw, browser_engine).launch(headless=True)
        try:
            page = browser.new_page(viewport={"width": 1280, "height": 850})
            page.add_init_script(f"({_RECORD_MARKS})()")
            _open(page, direct_server_with_data["url"])
            feed = _open_project(page, project)
            rest = _held(page, feed)
            assert rest["rows"] > 100 and rest["marks"] == 0 and rest["lazy"] == rest["asleep"], rest

            passage = _read_at(page, feed, 0.5)
            _drag(page, feed, "#project-panel > .resize-handle", passage, rest["rows"])
            climb = page.evaluate(_CLIMB, [feed, 240])
            assert climb["steps"] > 5 and climb["top"] == 0 and not climb["misses"], climb
        finally:
            browser.close()
