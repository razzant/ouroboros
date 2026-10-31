"""A kept page shows the version of the core it reconnects to (#2249): the real SPA, Chromium and WebKit.

Controlled transport (``tests/test_reconnect_feed_browser.py`` over ``tests/test_room_work_browser.py``). The static
fixture serves the shipped ``web/`` unmarked and ``/api/state`` reports ``served_ui``, so ws.js keeps the page across a
socket loss even when the served SHA changes: the path of an interface hosted apart from its core. ``/api/health``
answers from ``Transport.core`` when each read arrives, and the test changes it only after every read the page started
has landed. The kept page must keep its draft, caret, focus, reading place, document and nodes while both version
labels follow the new core; the core's own marked page still reloads onto the changed SHA, and a page opened after the
change reads the new version at load. A real server restart: ``tests/test_version_labels_restart_browser.py``.
"""
from __future__ import annotations

import time
from urllib.parse import urlparse

import pytest

from ouroboros.server_web import UI_OWNER_MARKER
from tests import test_reconnect_feed_browser as feed
from tests import test_room_work_browser as rw

room_ui = rw.room_ui
subscription_ui = rw.subscription_ui
pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]

MAIN = feed.MAIN
# A desktop app keeps its own version while the core inside it updates and restarts.
OLD = {"app": "7.8.0", "runtime": "7.8.0", "sha": "aaaa1111"}
NEW = {"app": "7.8.0", "runtime": "7.8.1", "sha": "bbbb2222"}
DRAFT = "Before I send: keep the citation order of section 6 and check the appendix numbers."
CARET = 31

# The page's /api/health reads: started, and settled once the body a reader applies is parsed.
HEALTH_READS = """() => {
    window.__healthReads = { started: 0, settled: 0 };
    const nativeFetch = window.fetch.bind(window);
    const settle = () => { window.__healthReads.settled += 1; };
    window.fetch = (input, init) => {
        const raw = typeof input === 'string' ? input : input?.url || '';
        if (new URL(raw, location.href).pathname !== '/api/health') return nativeFetch(input, init);
        window.__healthReads.started += 1;
        return nativeFetch(input, init).then((response) => {
            const nativeJson = response.json.bind(response);
            response.json = () => nativeJson().finally(settle);
            return response;
        }, (error) => { settle(); throw error; });
    };
}"""
_HEALTH_QUIET = "() => window.__healthReads.started > 0 && window.__healthReads.started === window.__healthReads.settled"

# The reader's place (a row 180 px below the feed's top), the draft's caret and focus, and the nodes a kept page keeps.
PIN = """([feedSelector, caret]) => {
    const root = document.querySelector(feedSelector), input = document.getElementById('chat-input');
    const rows = [...root.querySelectorAll('[data-history-id]')];
    const row = rows[Math.floor(rows.length / 2)];
    root.dispatchEvent(new WheelEvent('wheel', { deltaY: -1 }));  // the reader turns back: reading, not following
    root.scrollTop += row.getBoundingClientRect().top - root.getBoundingClientRect().top - 180;
    input.focus({ preventScroll: true });
    input.setSelectionRange(caret, caret);
    window.__kept = { row, input, top: row.getBoundingClientRect().top - root.getBoundingClientRect().top,
        nav: document.getElementById('nav-version'), about: document.getElementById('about-version') };
    return { rows: rows.length, below: root.scrollHeight - root.scrollTop - root.clientHeight };
}"""
FACTS = """(feedSelector) => {
    const kept = window.__kept, root = document.querySelector(feedSelector);
    const input = document.getElementById('chat-input');
    const label = (id) => document.getElementById(id)?.textContent ?? null;
    return {
        nav: label('nav-version'), about: label('about-version'), url: location.href, keptDocument: Boolean(kept),
        draft: input?.value ?? null, caret: input ? [input.selectionStart, input.selectionEnd] : null,
        focused: Boolean(input) && document.activeElement === input,
        sameNodes: Boolean(kept) && kept.input === input && kept.nav === document.getElementById('nav-version')
            && kept.about === document.getElementById('about-version'),
        rowKept: Boolean(kept) && Boolean(root?.contains(kept.row)),
        drift: kept && root ? Math.abs(kept.row.getBoundingClientRect().top - root.getBoundingClientRect().top - kept.top) : null,
        healthReads: window.__healthReads,
    };
}"""


def settle_labels(page, timeout=20_000):
    """Every /api/health read the page started has landed and painted (needs HEALTH_READS)."""
    page.wait_for_function(_HEALTH_QUIET, timeout=timeout)
    rw.settle_frames(page)


def facts(page, feed_selector=MAIN):
    return page.evaluate(FACTS, feed_selector)


def labels(core):
    return f"v{core['runtime']}", f"app {core['app']} | rt {core['runtime']}"


class Transport(feed.Transport):
    """The reconnect transport, with the served SHA and interface on ``/api/state`` and the versions on
    ``/api/health``, each read from ``self.core`` when its request arrives."""

    def __init__(self, page, *, core, **kwargs):
        self.core = dict(core)
        self.health_answers: list = []
        super().__init__(page, **kwargs)
        page.route("**/api/health*", self._health)

    def body(self, *, ready=True):
        return {**super().body(ready=ready), "sha": self.core["sha"], "served_ui": {"source": "builtin"}}

    def _health(self, route):
        runtime = self.core["runtime"]
        self.health_answers.append(runtime)
        route.fulfill(json={"status": "ok", "version": runtime, "runtime_version": runtime,
                            "app_version": self.core["app"]})


def conversation(count):
    """Questions and answers a minute apart, so the feed shows them in reading order."""
    return [row for index in range(count) for row in (
        rw.owner(f"Question {index}: what changed in section {index}?", f"version-q-{index}", index),
        rw.assistant(f"Answer {index}. " + "The reading notes continue with ordinary context. " * 5, index, 30))]


def open_room(ui, transport, page):
    page = rw.open_app({**ui, "page": page}, transport, viewport={"width": 1100, "height": 722})
    page.wait_for_function("(feed) => document.querySelectorAll(`${feed} [data-history-id]`).length >= 24", arg=MAIN)
    settle_labels(page)
    feed.settle(page)
    return page


def test_a_kept_page_takes_the_new_cores_versions_on_reconnect(room_ui):
    page = room_ui["page"]
    page.add_init_script(f"({HEALTH_READS})()")
    history = {1: conversation(12)}
    transport = Transport(page, core=OLD, history=history)
    page = open_room(room_ui, transport, page)
    page.fill("#chat-input", DRAFT)
    pinned = page.evaluate(PIN, [MAIN, CARET])
    assert pinned["below"] > 200, pinned  # the reader is in older history, not following the newest row
    feed.settle(page)
    before = facts(page)
    assert (before["nav"], before["about"]) == labels(OLD), before
    assert before["focused"] and before["caret"] == [CARET, CARET], before
    rw.capture(page, "version-labels-kept-before")

    # Every read started so far has landed: only reads the reconnect starts can see the new core.
    assert before["healthReads"]["started"] == before["healthReads"]["settled"], before
    transport.core = dict(NEW)
    saved = rw.assistant("Reply saved while the core restarted.", 50)
    history[1].append(saved)
    count = transport.disconnect()
    page.wait_for_function("() => document.querySelector('#chat-status')?.textContent.includes('Reconnecting')")
    transport.await_reconnect(count)
    page.locator(MAIN).get_by_text(saved["text"]).wait_for(state="attached")
    settle_labels(page)
    feed.settle(page)
    after = facts(page)
    notices = page.locator(feed.NOTICE).count()
    rw.capture(page, "version-labels-kept-after")
    page.locator('[data-nav-page="settings"]').click()
    page.locator('[data-settings-tab="about"]').click()
    page.locator("#about-version").wait_for(state="visible")
    rw.settle_frames(page)
    rw.capture(page, "version-labels-kept-about")
    # Control: a page opened after the change reads the new versions at load, with or without the repair.
    fresh = room_ui["new_page"]()
    fresh.add_init_script(f"({HEALTH_READS})()")
    fresh = open_room(room_ui, Transport(fresh, core=NEW, history=history), fresh)
    cold = facts(fresh)
    rw.evidence("version-labels-kept", {"before": before, "after": after, "cold": cold,
                                         "health_answers": transport.health_answers, "sockets": len(transport.sockets)})

    # The reload decision kept this page (unmarked, served_ui: keep on any SHA), with all the owner had in it.
    assert after["keptDocument"] and "_ouro_reason" not in after["url"] and notices == 0, after
    assert after["draft"] == DRAFT and after["caret"] == [CARET, CARET] and after["focused"], after
    assert after["sameNodes"] and after["rowKept"] and after["drift"] <= 6, after
    assert (cold["nav"], cold["about"]) == labels(NEW), cold
    # Both labels of the kept page now name the core it talks to.
    assert (after["nav"], after["about"]) == labels(NEW), after


def test_the_cores_own_page_still_reloads_onto_a_changed_sha(room_ui):
    page = room_ui["page"]
    marked = (rw.setup_browser.WEB / "index.html").read_text(encoding="utf-8").replace(
        "<head>", "<head>" + UI_OWNER_MARKER, 1)
    # The core's builtin index response, marked as its own (server_web.make_index_page), on the first load and the reload.
    page.route(lambda url: urlparse(url).path == "/",
               lambda route: route.fulfill(status=200, content_type="text/html; charset=utf-8", body=marked))
    page.add_init_script(f"({HEALTH_READS})()")
    history = {1: conversation(12)}
    transport = Transport(page, core=OLD, history=history)
    page = open_room(room_ui, transport, page)
    assert page.evaluate("() => document.querySelector('meta[name=\"ouroboros-ui-owner\"]')?.content") == "core"
    page.fill("#chat-input", DRAFT)
    page.evaluate(PIN, [MAIN, CARET])
    before = facts(page)
    assert (before["nav"], before["about"]) == labels(OLD), before
    rw.capture(page, "version-labels-reload-before")

    documents = []
    room_ui["errors"].watch(page)  # the reload below may cut off this page's own reads (see the last check)
    page.on("framenavigated", lambda frame: documents.append(frame.url) if frame == page.main_frame else None)
    transport.core = dict(NEW)
    count = transport.disconnect()
    # The kept socket reconnects, reads the changed SHA and reloads; the reloaded page opens its own socket.
    deadline = time.monotonic() + 30
    while (len(transport.sockets) < count + 2 or not documents) and time.monotonic() < deadline:
        page.wait_for_timeout(100)
    assert documents and "_ouro_reason=sha-change" in documents[0], documents
    page.wait_for_function("(feed) => document.querySelectorAll(`${feed} [data-history-id]`).length >= 24", arg=MAIN)
    page.locator(f"{MAIN} {feed.NOTICE}").wait_for(state="attached")
    settle_labels(page)
    feed.settle(page)
    after = facts(page)
    rw.evidence("version-labels-reload", {"before": before, "after": after, "documents": documents,
                                           "health_answers": transport.health_answers})
    rw.capture(page, "version-labels-reload-after")
    assert not after["keptDocument"], "the marked page reloaded onto the changed code"
    assert page.locator(f"{MAIN} {feed.NOTICE} .message").inner_text().strip() == "Restart complete"
    assert (after["nav"], after["about"]) == labels(NEW), after
    # WebKit reports the own-origin reads the reload cut off as page errors (ui_console_errors.PageErrors): only
    # those of this page, watched as navigating, are set aside; every other error of the fixture's pages counts.
    assert not room_ui["errors"].unexpected(), room_ui["errors"].report()
