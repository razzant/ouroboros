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

pytestmark = [pytest.mark.ui_browser]

EFFORT = "#page-chat .chat-effort-range"
# The hint of the capsule `sel` against the window that shows it (the visual viewport where there is one).
WHERE = """(sel) => {
    const hint = document.querySelector(sel).querySelector('.ui-hint');
    const body = hint.querySelector('.ui-hint-body');
    const r = hint.getBoundingClientRect(), v = window.visualViewport;
    const top = v ? v.offsetTop : 0, height = v ? v.height : innerHeight;
    const mask = (style) => (style.maskImage && style.maskImage !== 'none') ? style.maskImage : (style.webkitMaskImage || 'none');
    return { left: r.left, right: r.right, top: r.top, bottom: r.bottom, width: document.documentElement.clientWidth,
        windowTop: top, windowBottom: top + height, side: hint.dataset.side || '', cramped: hint.dataset.cramped || '',
        hidden: hint.hidden, scrolls: body.scrollHeight > body.clientHeight + 1,
        bodyMask: mask(getComputedStyle(body)), shellMask: mask(getComputedStyle(hint)),
        shellBackground: getComputedStyle(hint).backgroundColor };
}"""
# The one cramped hint, built by the product's own module on the product's own page: its words wait past the edge and
# fade there; the glass they stand on does not.
CRAMPED = """(async () => {
    let module = null;
    for (const path of ['/static/modules/ui_hint.js', './modules/ui_hint.js', '/modules/ui_hint.js']) {
        try { module = await import(path); break; } catch (error) { /* the next place the page keeps it */ }
    }
    if (!module) throw new Error('the hint module is not served on this page');
    const host = document.createElement('div');
    host.id = 'probe-hint-host';
    host.style.cssText = 'position:absolute;left:16px;top:78%;width:2px;height:2px;z-index:80;';
    document.body.append(host);
    const hint = module.createHint(host, { doc: document });
    const words = Array.from({ length: 18 }, (_v, n) =>
        `Line ${n + 1} of a hint that has more words than the room it stands in.`).join(' ');
    hint.say({ word: 'Cramped', text: words, action: 'Learn more' });
    hint.show();
    const el = hint.el, body = el.querySelector('.ui-hint-body');
    const frames = [];
    for (let n = 0; n < 6; n++) {
        await new Promise(done => requestAnimationFrame(() => requestAnimationFrame(done)));
        frames.push(el.dataset.cramped || '');
    }
    const hs = getComputedStyle(el), bs = getComputedStyle(body);
    const mask = (style) => (style.maskImage && style.maskImage !== 'none') ? style.maskImage : (style.webkitMaskImage || 'none');
    return { cramped: el.dataset.cramped || '', side: el.dataset.side || '', frames,
        bodyMask: mask(bs), shellMask: mask(hs), shellBackground: hs.backgroundColor,
        shellBackdrop: hs.backdropFilter || hs.webkitBackdropFilter || 'none',
        bodyScrolls: body.scrollHeight > body.clientHeight + 1,
        bodyHeight: body.clientHeight, bodyScrollHeight: body.scrollHeight };
})()"""
# The shared rule line in a translation longer than the English.
LONGER = "Gilt für neue Schritte in diesem Chat; laufende Arbeit wird ohne Unterbrechung zu Ende geführt."
GERMAN = {"language": "de", "english": False, "revision": 1, "entries": {
    "Applies to new steps; running work finishes.": {"text": LONGER, "provenance": "imported"}}}


def _alpha(color: str) -> float:
    """The alpha of a computed colour: 0 for a colour with none, 1 for an opaque one."""
    if color.startswith("rgba"):
        parts = color[color.find("(") + 1:color.find(")")].split(",")
        return float(parts[3]) if len(parts) > 3 else 1.0
    if color.startswith("rgb"):
        return 1.0
    return 0.0


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


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_a_cramped_hint_fades_only_its_words_and_keeps_its_glass(direct_server_with_data, engine, tmp_path):  # noqa: F811
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    seen = {}
    with sync_playwright() as pw:
        browser, page = _launch(pw, engine, viewport={"width": 420, "height": 300}, color_scheme="dark")
        try:
            _open_chat(page, url, width=420, height=300, theme="dark")
            seen["hint"] = where = page.evaluate(CRAMPED)
            assert where["cramped"] == "true", ("the hint does not fit the room it was given", seen)
            assert where["bodyScrolls"], ("the words wait past the edge", seen)
            assert "linear-gradient" in where["bodyMask"], ("the words fade where they wait", seen)
            assert where["shellMask"] == "none", ("the glass the words stand on is not masked", seen)
            assert _alpha(where["shellBackground"]) > 0, ("the glass is painted", seen)
            assert set(where["frames"]) == {"true"}, ("the hint keeps its room instead of breathing", seen)
            page.locator("#probe-hint-host .ui-hint").screenshot(
                path=str(_evidence_dir(tmp_path) / f"ui-hint-cramped-{engine}.png"))
        finally:
            (_evidence_dir(tmp_path) / f"ui-hint-cramped-{engine}.json").write_text(json.dumps(seen, indent=1),
                                                                                   encoding="utf-8")
            page.unroute_all(behavior="ignoreErrors")
            browser.close()


# Effort's hint and where the keyboard stands in it: its words have no action, so a cramped body is the one way in.
READING = """(sel) => {
    const el = document.querySelector(sel), hint = el.querySelector('.ui-hint'), body = hint.querySelector('.ui-hint-body');
    const mask = (style) => (style.maskImage && style.maskImage !== 'none') ? style.maskImage : (style.webkitMaskImage || 'none');
    const pad = (name) => parseFloat(getComputedStyle(hint).getPropertyValue(name)) || 0;
    const a = document.activeElement, box = body.getBoundingClientRect();
    const seen = document.elementFromPoint(box.left + box.width / 2, box.top + Math.min(box.height / 2, 12));
    return { open: el.dataset.open, hidden: hint.hidden, cramped: hint.dataset.cramped || '', side: hint.dataset.side || '',
        actions: [...hint.querySelectorAll('button')].filter((b) => !b.hidden).length,
        tabindex: body.getAttribute('tabindex'), role: body.getAttribute('role'), label: body.getAttribute('aria-label'),
        clientHeight: body.clientHeight, scrollHeight: body.scrollHeight, scrollTop: body.scrollTop,
        natural: body.scrollHeight + pad('padding-top') + pad('padding-bottom') + 2,
        focused: a === body, inHint: hint.contains(a), inCapsule: el.contains(a), onHead: a === el.querySelector('.chat-capsule-head'),
        active: a ? `${a.tagName.toLowerCase()}.${a.className}` : '', keyboardRing: body.matches(':focus-visible'),
        bodyMask: mask(getComputedStyle(body)), bodyOutline: getComputedStyle(body).outlineStyle,
        shellOutline: getComputedStyle(hint).outlineStyle, shellMask: mask(getComputedStyle(hint)),
        onTop: Boolean(seen) && hint.contains(seen), capsuleTop: el.getBoundingClientRect().top,
        capsuleBottom: el.getBoundingClientRect().bottom, hintTop: hint.getBoundingClientRect().top, innerHeight };
}"""


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_a_cramped_hint_without_an_action_is_read_with_the_keyboard(direct_server_with_data, engine, tmp_path):  # noqa: F811
    """WebKit's Tab and Option+Tab reach no box that only scrolls: Effort's words (the translated rule makes them long),
    cramped in a short phone-wide window, are a named region the keyboard enters after the strip, scrolls and leaves by
    Escape, while the hint stays open; with room the same words are no stop at all."""
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    tab = "Alt+Tab" if engine == "webkit" else "Tab"  # macOS WebKit tabs to buttons with Option
    head = f"{EFFORT} .chat-capsule-head"
    seen = {}

    def open_by_keyboard(page):
        page.focus(head)
        page.keyboard.press("Enter")
        page.wait_for_function("sel => document.querySelector(sel)?.dataset.open === 'true'", arg=EFFORT)
        _settle(page)
        return page.evaluate(READING, EFFORT)

    with sync_playwright() as pw:
        browser, page = _launch(pw, engine, viewport={"width": 320, "height": 760})
        try:
            _install(page, GERMAN)
            _open_chat(page, url, width=320, height=760)
            roomy = seen["roomy"] = open_by_keyboard(page)
            assert not roomy["hidden"] and roomy["cramped"] == "false" and roomy["actions"] == 0, seen
            assert roomy["tabindex"] is None and roomy["role"] is None, ("words that fit are no stop", seen)
            walk = seen["roomy_walk"] = []
            for _ in range(6):
                page.keyboard.press(tab)
                walk.append(page.evaluate(READING, EFFORT))
                if not walk[-1]["inCapsule"]:
                    break
            assert not any(step["inHint"] for step in walk), ("with room, the keyboard passes the hint as before", seen)
            page.focus(head)
            page.keyboard.press("Escape")
            page.wait_for_function("sel => document.querySelector(sel)?.dataset.open === 'false'", arg=EFFORT)
            # The composer keeps its distance from the window's foot, its field under the capsules: a window that leaves
            # above the capsule more room than the field's, and less than the words need, makes the hint scroll inside
            # itself where it stands, above.
            foot = roomy["innerHeight"] - roomy["capsuleTop"]
            under = roomy["innerHeight"] - roomy["capsuleBottom"] - 2 * 8
            assert roomy["natural"] > under + 8, ("the words need more than the field's height", seen)
            short = int(foot + 2 * 8 + (under + roomy["natural"]) / 2)
            page.set_viewport_size({"width": 320, "height": short})
            page.wait_for_timeout(300)
            cramped = seen["cramped"] = open_by_keyboard(page)
            assert cramped["cramped"] == "true" and cramped["side"] == "above", ("less room than its words, above", seen)
            page.wait_for_function("sel => document.querySelector(sel).querySelector('.ui-hint-body').tabIndex === 0",
                                   arg=EFFORT)
            chain = seen["chain"] = []
            for _ in range(6):
                page.keyboard.press(tab)
                chain.append(page.evaluate(READING, EFFORT))
                if chain[-1]["focused"]:
                    break
            reading = chain[-1]
            assert reading["focused"] and reading["keyboardRing"], ("the keyboard reaches the words after the strip", seen)
            assert reading["role"] == "region" and reading["label"] == "Hint" and reading["tabindex"] == "0", seen
            assert reading["scrollHeight"] > reading["clientHeight"] + 1, ("the words wait past the edge", seen)
            assert reading["onTop"], ("the words stand over the page, nothing on them", seen)
            assert "linear-gradient" in reading["bodyMask"], ("the words keep fading while the keyboard reads them", seen)
            assert reading["shellOutline"] == "solid" and reading["bodyOutline"] == "none", (
                "the ring stands on the shell, outside the words' fade", seen)
            page.locator(f"{EFFORT} .ui-hint").screenshot(
                path=str(_evidence_dir(tmp_path) / f"ui-hint-keyboard-reading-{engine}.png"))
            page.keyboard.press("PageDown")
            page.wait_for_function("sel => document.querySelector(sel).querySelector('.ui-hint-body').scrollTop > 0",
                                   arg=EFFORT)
            page.wait_for_timeout(500)  # past the strip's leave delay: reading holds it
            seen["scrolled"] = scrolled = page.evaluate(READING, EFFORT)
            assert scrolled["open"] == "true" and not scrolled["hidden"] and scrolled["focused"], (
                "the hint stays open while its words are read", seen)
            page.locator(f"{EFFORT} .ui-hint").screenshot(
                path=str(_evidence_dir(tmp_path) / f"ui-hint-keyboard-scrolled-{engine}.png"))
            page.keyboard.press("Escape")
            page.wait_for_function("sel => document.querySelector(sel)?.dataset.open === 'false'", arg=EFFORT)
            seen["closed"] = closed = page.evaluate(READING, EFFORT)
            assert closed["hidden"] and closed["onHead"], ("Escape closes it and gives the keyboard back to Effort", seen)
            page.wait_for_function("sel => !document.querySelector(sel).querySelector('.ui-hint-body').hasAttribute('tabindex')",
                                   arg=EFFORT)
        finally:
            (_evidence_dir(tmp_path) / f"ui-hint-keyboard-{engine}.json").write_text(json.dumps(seen, indent=1),
                                                                                    encoding="utf-8")
            page.unroute_all(behavior="ignoreErrors")
            browser.close()
