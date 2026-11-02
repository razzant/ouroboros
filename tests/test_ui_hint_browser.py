"""The one hint stays inside the window (docs/DESIGN.md "Hints", web/modules/ui_hint.js).

In Chromium and WebKit on the real composer: a translated hint (the page's real translation overlay, which rewrites
every write in place) stays inside a 1000 px window, where the docked sidebar leaves Effort's capsule far enough to
the right for its hint to pass the window's edge, also after the page's next state read renders the capsule again
with the same words; and in a 320 x 280 phone window (a phone in landscape with its keyboard up leaves about that much)
Subagents' longest hint never leaves the window, with a long draft before it opens or a draft that grows while it is
open: it stands below the capsule or scrolls inside itself where neither side has the room.
"""
from __future__ import annotations

import json

import pytest

from tests.test_composer_capsules_browser import _install
from tests.test_composer_subagents_browser import STATE_READS, SUBAGENTS, _hold, _settle
from tests.test_effort_range_browser import _evidence_dir, _launch, _open_chat
from tests.test_ui_smoke_playwright import direct_server_with_data  # noqa: F401 - pytest fixture import

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]

EFFORT = "#page-chat .chat-effort-range"
# The hint of the capsule `sel` against the window that shows it (the visual viewport where there is one).
WHERE = """(sel) => {
    const hint = document.querySelector(sel).querySelector('.ui-hint');
    const r = hint.getBoundingClientRect(), v = window.visualViewport;
    const top = v ? v.offsetTop : 0, height = v ? v.height : innerHeight;
    return { left: r.left, right: r.right, top: r.top, bottom: r.bottom, width: document.documentElement.clientWidth,
        windowTop: top, windowBottom: top + height, side: hint.dataset.side || '', cramped: hint.dataset.cramped || '',
        hidden: hint.hidden, scrolls: hint.scrollHeight > hint.clientHeight + 1 };
}"""
# The shared rule line in a translation longer than the English.
LONGER = "Gilt für neue Schritte in diesem Chat; laufende Arbeit wird ohne Unterbrechung zu Ende geführt."
GERMAN = {"language": "de", "english": False, "revision": 1, "entries": {
    "Applies to new steps; running work finishes.": {"text": LONGER, "provenance": "imported"}}}


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_words_that_grow_after_the_hint_stood_keep_it_inside_the_window(direct_server_with_data, engine, tmp_path):  # noqa: F811
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    seen = {}
    with sync_playwright() as pw:
        browser, page = _launch(pw, engine, viewport={"width": 1000, "height": 820})
        try:
            _install(page, GERMAN)
            _open_chat(page, url, width=1000, height=820)
            page.click(f"{EFFORT} .chat-capsule-head")
            page.wait_for_function("sel => document.querySelector(sel)?.dataset.open === 'true'", arg=EFFORT)
            page.wait_for_function("([sel, words]) => document.querySelector(sel).querySelector('.ui-hint-rule')"
                                   ".textContent === words", arg=[EFFORT, LONGER], timeout=10_000)
            _settle(page)
            seen["translated"] = page.evaluate(WHERE, EFFORT)
            # The page reads its state every 3 s and the capsule renders its hint again, with the same words: the read's
            # body is in, and two frames later the page has applied it.
            with page.expect_response(lambda r: STATE_READS.search(r.url) is not None, timeout=20_000) as state:
                pass
            state.value.body()
            page.evaluate("() => new Promise((done) => requestAnimationFrame(() => requestAnimationFrame(done)))")
            _settle(page)
            seen["echoed"] = page.evaluate(WHERE, EFFORT)
            for step in ("translated", "echoed"):
                where = seen[step]
                assert not where["hidden"] and where["left"] >= 7 and where["right"] <= where["width"] - 7, \
                    (f"{step}: the translated hint stands inside the window", seen)
            assert page.locator(f"{EFFORT} .ui-hint-rule").inner_text() == LONGER, "the translation stays"
        finally:
            (_evidence_dir(tmp_path) / f"ui-hint-grown-words-{engine}.json").write_text(json.dumps(seen, indent=1),
                                                                                      encoding="utf-8")
            page.unroute_all(behavior="ignoreErrors")  # a state read still in flight ends with the browser
            browser.close()


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_a_short_phone_window_never_puts_the_hint_out_of_it(direct_server_with_data, engine, tmp_path):  # noqa: F811
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    seen = {}
    with sync_playwright() as pw:
        browser, page = _launch(pw, engine, viewport={"width": 320, "height": 280}, is_mobile=True, has_touch=True)
        try:
            _hold(page, {"subagents_mode": "off", "subagents_availability": "available"})
            _open_chat(page, url, width=320, height=280)
            page.fill("#chat-input", "\n".join(f"A line of the draft, number {n}" for n in range(1, 15)))
            page.tap(f"{SUBAGENTS} .chat-capsule-head")
            page.wait_for_function("sel => document.querySelector(sel)?.dataset.open === 'true'", arg=SUBAGENTS)
            _settle(page)
            seen["hint"] = where = page.evaluate(WHERE, SUBAGENTS)
            assert not where["hidden"], seen
            assert where["top"] >= where["windowTop"] - 1 and where["bottom"] <= where["windowBottom"] + 1, \
                ("the hint stands inside the window, above, below or scrolling", seen)
            assert where["side"] in ("above", "below") and (where["cramped"] == "true") == where["scrolls"], seen
            # A short draft, the hint open above the row; then the keyboard goes to the field and the draft grows to
            # the field's cap: the row rises with the capsule, and the hint finds its place again.
            _open_chat(page, url, width=320, height=280)
            page.fill("#chat-input", "A short draft")
            page.tap(f"{SUBAGENTS} .chat-capsule-head")
            page.wait_for_function("sel => document.querySelector(sel)?.dataset.open === 'true'", arg=SUBAGENTS)
            _settle(page)
            seen["short"] = page.evaluate(WHERE, SUBAGENTS)
            page.focus("#chat-input")
            page.keyboard.press("End")
            for n in range(2, 15):
                page.keyboard.press("Shift+Enter")
                page.keyboard.type(f"Line {n}")
            _settle(page)
            seen["grown"] = where = page.evaluate(WHERE, SUBAGENTS)
            assert not where["hidden"], ("the hint stays while the keyboard writes", seen)
            assert where["top"] >= where["windowTop"] - 1 and where["bottom"] <= where["windowBottom"] + 1, \
                ("a draft that grows under the open hint keeps it inside the window", seen)
        finally:
            (_evidence_dir(tmp_path) / f"ui-hint-short-window-{engine}.json").write_text(json.dumps(seen, indent=1),
                                                                                       encoding="utf-8")
            page.unroute_all(behavior="ignoreErrors")  # a state read still in flight ends with the browser
            browser.close()
