"""The effort range control in the real composer (DECISIONS v3 §6; DESIGN "Composer owner controls").

On a real server in Chromium and WebKit: the round button after the pills opens inline into the
strip — by hover (an accelerator) and by press (which pins) on a desktop with a mouse, by press
alone on a phone; the mouse may drift within 32 px and the strip closes a second after it leaves
that zone; Esc closes and returns focus, an outside press closes. The open strip never takes a
second line: beside the pills on a wide screen, alone while the pills step aside on a 390 px
phone and in a ~440 px Project pane, scrolling inside itself on a 320 px phone; the pill covers
its word and the band meets the brackets. A drag and the keyboard save the full triple through
the owner endpoint, the pill pushes a bracket and the bracket stays pushed, and every composer
reads the one value back from /api/state; a change made just before Send lands before the
message; Settings → Behavior has no effort section and ends with Startup & background, whose
host fact reads as one plain note; Settings → Agents says Auto or the level in the model name.
"""
from __future__ import annotations

import json
import os
import pathlib
import time

import pytest

from tests import test_subscription_role_routes_browser as roles
from tests.test_ui_smoke_playwright import direct_server_with_data  # noqa: F401 - pytest fixture import

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]
subscription_ui = roles.subscription_ui
role_ui = roles.role_ui

CONTROL = "#page-chat .chat-effort-range"
HEAD = f"{CONTROL} .chat-effort-head"
HOVER_MEDIA = "(hover: hover) and (pointer: fine)"
FACTS = """sel => {
    const el = document.querySelector(sel);
    const rect = (node) => node.getBoundingClientRect();
    const r = rect(el);
    const head = el.querySelector('.chat-effort-head');
    const row = el.parentElement;
    const pills = row.querySelector('.chat-composer-pills');
    const page = el.closest('#page-chat, .chat-instance-panel');
    const area = page.querySelector('#chat-input-area, .chat-input-area');
    const wrap = area.querySelector('.chat-input-wrap');
    const input = page.querySelector('#chat-input, .chat-input');
    const text = (which) => el.querySelector('.chat-effort-' + which).getAttribute('aria-valuetext');
    const box = (node) => { const b = node ? rect(node) : null; return b && { left: b.left, right: b.right, top: b.top, bottom: b.bottom }; };
    const recSeg = el.querySelector('.chat-effort-seg[data-rec="true"]');
    const strip = el.querySelector('.chat-effort-strip');
    return {
        open: el.dataset.open, known: el.dataset.known, saving: el.dataset.saving || 'false',
        expanded: head.getAttribute('aria-expanded'), title: head.title,
        top: r.top, left: r.left, right: r.right, width: r.width, height: r.height,
        headWidth: rect(head).width, headHeight: rect(head).height,
        pillsTop: rect(pills).top, pillsBottom: rect(pills).bottom, rowRight: rect(row).right,
        swarmHeight: rect(row.querySelector('.chat-swarm')).height,
        sendHeight: rect(page.querySelector('.chat-send-inline')).height,
        inputTop: rect(input).top,
        docScroll: document.documentElement.scrollWidth > window.innerWidth + 1,
        areaScroll: area.scrollWidth > area.clientWidth + 1 || wrap.scrollWidth > wrap.clientWidth + 1,
        areaHeight: area.offsetHeight,
        reserve: parseFloat(getComputedStyle(page).getPropertyValue('--chat-input-reserve')) || 0,
        headFocused: document.activeElement === head,
        recFocused: document.activeElement === el.querySelector('.chat-effort-rec'),
        recTabIndex: el.querySelector('.chat-effort-rec').tabIndex,
        min: text('min'), rec: text('rec'), max: text('max'),
        hoverMedia: matchMedia('%s').matches,
        fit: el.dataset.fit || '', solo: row.dataset.effortSolo || '', pillsShown: getComputedStyle(pills).display !== 'none',
        rowHeight: rect(row).height, rowTop: rect(row).top,
        extras: el.querySelectorAll('.chat-effort-label, .chat-effort-sep, .chat-effort-reset').length,
        pill: box(el.querySelector('.chat-effort-rec')), seg: box(recSeg), word: box(recSeg?.querySelector('.chat-effort-seg-text')),
        band: box(el.querySelector('.chat-effort-band')), minCap: box(el.querySelector('.chat-effort-min')),
        maxCap: box(el.querySelector('.chat-effort-max')),
        stripScrolls: strip.scrollWidth > strip.clientWidth + 1, stripScroll: strip.dataset.scroll || '',
    };
}""" % HOVER_MEDIA


def _evidence_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    target = pathlib.Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", str(tmp_path)))
    target.mkdir(parents=True, exist_ok=True)
    return target


def _shot(page, tmp_path, name: str) -> None:
    page.screenshot(path=str(_evidence_dir(tmp_path) / f"{name}.png"), animations="disabled")


def _open_chat(page, url: str, *, width: int, height: int = 900, theme: str = "dark") -> None:
    page.add_init_script(f"try {{ localStorage.setItem('ouroboros.theme', '{theme}'); }} catch (e) {{}}")
    page.set_viewport_size({"width": width, "height": height})
    page.goto(url, wait_until="domcontentloaded", timeout=30_000)
    page.wait_for_selector("#chat-input", timeout=30_000)
    # The control paints the one global value once /api/state answered.
    page.wait_for_function(f"() => document.querySelector('{CONTROL}')?.dataset.known === 'true'", timeout=30_000)


def _facts(page, selector: str = CONTROL) -> dict:
    return page.evaluate(FACTS, selector)


def _wait(page, predicate: str, selector: str = CONTROL, timeout: int = 5_000) -> None:
    page.wait_for_function(f"sel => {{ const el = document.querySelector(sel); return el && ({predicate}); }}",
                           arg=selector, timeout=timeout)


def _segment_center(page, level: str, selector: str = CONTROL) -> tuple[float, float]:
    box = page.locator(f'{selector} [data-effort-seg="{level}"]').bounding_box()
    return box["x"] + box["width"] / 2, box["y"] + box["height"] / 2


def _assert_strip_geometry(facts: dict) -> None:
    """The pill covers its whole segment and its word with air on both sides; the band, the
    pill and the brackets share one vertical box, and the band's ends are the brackets."""
    pill, seg, word, band = facts["pill"], facts["seg"], facts["word"], facts["band"]
    assert abs(pill["left"] - seg["left"]) <= 0.6 and abs(pill["right"] - seg["right"]) <= 0.6, facts
    assert pill["left"] <= word["left"] - 1.5 and pill["right"] >= word["right"] + 1.5, facts
    for cap in (facts["minCap"], facts["maxCap"], pill):
        assert abs(cap["top"] - band["top"]) <= 0.6 and abs(cap["bottom"] - band["bottom"]) <= 0.6, facts
    assert abs(facts["minCap"]["left"] - band["left"]) <= 0.6 and abs(facts["maxCap"]["right"] - band["right"]) <= 0.6, facts


def _launch(pw, engine: str, **context):
    from playwright.sync_api import Error as PlaywrightError

    try:
        browser = getattr(pw, engine).launch(headless=True)
    except PlaywrightError as exc:
        pytest.skip(f"{engine} is not installed: {exc}")
    return browser, browser.new_page(**context)


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_desktop_hover_opens_press_pins_escape_and_outside_close(direct_server_with_data, engine, tmp_path):  # noqa: F811
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    with sync_playwright() as pw:
        browser, page = _launch(pw, engine, viewport={"width": 1440, "height": 900})
        try:
            _open_chat(page, url, width=1440)
            closed = _facts(page)
            assert closed["hoverMedia"] is True, closed
            assert closed["open"] == "false" and closed["expanded"] == "false", closed
            # A round 32 px button on the pills' row, after them.
            assert abs(closed["headWidth"] - 32) <= 2 and abs(closed["headHeight"] - 32) <= 2, closed
            assert abs(closed["top"] - closed["pillsTop"]) <= 2, closed
            assert closed["recTabIndex"] == -1, "closed handles are out of the tab order"
            assert closed["title"].startswith("Effort: ") and "Applies to new work." in closed["title"], closed
            assert closed["extras"] == 0, "no word Effort, no separator, no reset"
            _shot(page, tmp_path, f"composer-closed-{engine}-dark")
            # Hover opens after ~160 ms; at this width the strip stays beside the pills.
            page.hover(HEAD)
            _wait(page, "el.dataset.open === 'true'")
            page.wait_for_timeout(500)  # the open animation
            opened = _facts(page)
            assert opened["fit"] == "inline" and opened["solo"] == "" and opened["pillsShown"], opened
            assert abs(opened["top"] - opened["pillsTop"]) <= 2, opened
            assert opened["right"] <= opened["rowRight"] + 1 and not opened["docScroll"], opened
            assert opened["width"] <= 400, "the open control is compact: " + str(opened["width"])
            assert opened["recTabIndex"] == 0, opened
            _assert_strip_geometry(opened)
            _shot(page, tmp_path, f"composer-open-{engine}-dark")
            # The mouse may drift within the 32 px zone: the strip stays open.
            page.mouse.move(opened["right"] + 20, opened["top"] + opened["height"] / 2)
            page.wait_for_timeout(1300)
            assert _facts(page)["open"] == "true", "the zone keeps the strip open"
            # Leaving the zone closes it about a second later, not at once.
            page.mouse.move(20, 200)
            page.wait_for_timeout(500)
            assert _facts(page)["open"] == "true", "not before the delay"
            _wait(page, "el.dataset.open === 'false'", timeout=3_000)
            # A mouse press on a level of a hover-opened strip does not hold it open afterwards.
            page.hover(HEAD)
            _wait(page, "el.dataset.open === 'true'")
            page.wait_for_timeout(450)
            x, y = _segment_center(page, "medium")
            page.mouse.click(x, y)
            page.mouse.move(20, 200)
            _wait(page, "el.dataset.open === 'false'", timeout=3_000)
            # Hover again, then a press pins it: leaving no longer closes.
            page.hover(HEAD)
            _wait(page, "el.dataset.open === 'true'")
            page.click(HEAD)
            page.mouse.move(20, 200)
            page.wait_for_timeout(1300)  # longer than the 1 s close an unpinned strip would get
            assert _facts(page)["open"] == "true", "pinned: leaving does not close"
            # Esc closes and returns focus to the button.
            page.keyboard.press("Escape")
            _wait(page, "el.dataset.open === 'false'")
            assert _facts(page)["headFocused"] is True
            # Enter opens and focuses the recommended handle; an outside press closes.
            page.keyboard.press("Enter")
            _wait(page, "el.dataset.open === 'true'")
            assert _facts(page)["recFocused"] is True
            page.mouse.click(400, 300)
            _wait(page, "el.dataset.open === 'false'")
            # Light theme, for the eye.
            page.add_init_script("try { localStorage.setItem('ouroboros.theme', 'light'); } catch (e) {}")
            page.reload(wait_until="domcontentloaded")
            page.wait_for_function(f"() => document.querySelector('{CONTROL}')?.dataset.known === 'true'", timeout=30_000)
            _shot(page, tmp_path, f"composer-closed-{engine}-light")
            page.click(HEAD)
            _wait(page, "el.dataset.open === 'true'")
            page.wait_for_timeout(450)
            _shot(page, tmp_path, f"composer-open-{engine}-light")
        finally:
            browser.close()


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_a_tight_row_keeps_one_line_while_the_strip_closes(direct_server_with_data, engine, tmp_path):  # noqa: F811
    """A row just wide enough for the strip beside the pills at a tighter padding (a narrow
    window, a Project pane): closing animates the strip away without ever wrapping the row."""
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    with sync_playwright() as pw:
        browser, page = _launch(pw, engine, viewport={"width": 1440, "height": 900})
        try:
            _open_chat(page, url, width=1440)
            # Narrow the composer row until the strip only fits beside the pills below full padding.
            for width in range(1120, 820, -8):
                page.set_viewport_size({"width": width, "height": 900})
                page.click(HEAD)
                _wait(page, "el.dataset.open === 'true'")
                pad = page.evaluate(f"() => parseFloat(document.querySelector('{CONTROL}').style.getPropertyValue('--effort-seg-pad'))")
                fit = _facts(page)["fit"]
                if fit == "inline" and pad < 8:
                    break
                page.keyboard.press("Escape")
                _wait(page, "el.dataset.open === 'false'")
            else:
                pytest.fail("no viewport between 1120 and 828 px gave an inline strip below full padding")
            closed_height = page.evaluate(f"() => document.querySelector('{CONTROL}').parentElement.getBoundingClientRect().height")
            heights = page.evaluate(f"""() => new Promise((resolve) => {{
                const row = document.querySelector('{CONTROL}').parentElement;
                const seen = [];
                const sample = () => {{ seen.push(row.getBoundingClientRect().height); if (seen.length < 40) requestAnimationFrame(sample); else resolve(seen); }};
                document.dispatchEvent(new KeyboardEvent('keydown', {{ key: 'Escape' }}));
                requestAnimationFrame(sample);
            }})""")
            assert max(heights) <= closed_height + 1, (closed_height, heights)
        finally:
            browser.close()


@pytest.mark.parametrize("engine,width", [("chromium", 390), ("webkit", 390), ("chromium", 360)])
def test_phone_press_opens_alone_on_the_line_while_the_pills_step_aside(direct_server_with_data, engine, width, tmp_path):  # noqa: F811
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    with sync_playwright() as pw:
        browser, page = _launch(pw, engine, viewport={"width": width, "height": 800}, is_mobile=True, has_touch=True)
        try:
            _open_chat(page, url, width=width, height=800)
            closed = _facts(page)
            assert closed["hoverMedia"] is False, closed
            assert abs(closed["swarmHeight"] - closed["sendHeight"]) <= 1, closed
            assert abs(closed["height"] - closed["sendHeight"]) <= 1, "the round button keeps the pills' height"
            assert closed["top"] < closed["inputTop"] and not closed["docScroll"] and not closed["areaScroll"], closed
            _shot(page, tmp_path, f"composer-phone-{width}-closed-{engine}")
            page.tap(HEAD)
            _wait(page, "el.dataset.open === 'true'")
            # The opening frame already has every mark on its word: nothing slides into place.
            _assert_strip_geometry(_facts(page))
            page.wait_for_timeout(300)
            opened = _facts(page)
            # Alone on the same line: the pills step aside, the row keeps one line, nothing scrolls.
            assert opened["fit"] == "solo" and opened["solo"] == "true" and not opened["pillsShown"], opened
            assert abs(opened["rowHeight"] - closed["rowHeight"]) <= 1, (opened, closed)
            assert abs(opened["areaHeight"] - closed["areaHeight"]) <= 1, (opened, closed)
            assert opened["right"] <= opened["rowRight"] + 1, opened
            assert not opened["docScroll"] and not opened["areaScroll"] and not opened["stripScrolls"], opened
            assert opened["top"] < opened["inputTop"], opened
            _assert_strip_geometry(opened)
            _shot(page, tmp_path, f"composer-phone-{width}-open-{engine}")
            # A tap outside closes and brings the pills back at once.
            page.tap("#chat-messages")
            _wait(page, "el.dataset.open === 'false'")
            back = _facts(page)
            assert back["pillsShown"] and back["solo"] == "" and back["fit"] == "", back
            assert abs(back["rowHeight"] - closed["rowHeight"]) <= 1, (back, closed)
            page.tap(HEAD)
            _wait(page, "el.dataset.open === 'true'")
            page.tap(HEAD)
            _wait(page, "el.dataset.open === 'false'")
        finally:
            browser.close()


def test_project_pane_strip_stands_alone_and_shows_the_same_value(direct_server_with_data, tmp_path):  # noqa: F811
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import sync_playwright

    from ouroboros.projects_registry import create_project

    url = direct_server_with_data["url"]
    create_project(direct_server_with_data["data_dir"], "pane", name="Pane")
    panel = "#panel-pchat-pane .chat-effort-range"
    with sync_playwright() as pw:
        browser, page = _launch(pw, "chromium", viewport={"width": 1440, "height": 900})
        try:
            _open_chat(page, url, width=1440)
            page.click('.nav-project-row[data-project-id="pane"]')
            page.wait_for_selector("#project-panel:not([hidden])", timeout=30_000)
            page.wait_for_function(f"() => document.querySelector('{panel}')?.dataset.known === 'true'", timeout=30_000)
            main, pane = _facts(page), _facts(page, panel)
            assert (main["min"], main["rec"], main["max"]) == (pane["min"], pane["rec"], pane["max"]), (main, pane)
            width = page.locator("#project-panel").bounding_box()["width"]
            assert 340 <= width <= 440.5, width
            closed = _facts(page, panel)
            # The strip would stand alone here, so hovering does not open it under a still mouse:
            # the press opens it (pinned), and that press saves nothing.
            saves = []
            page.on("request", lambda r: saves.append(r.url) if r.url.endswith("/api/owner/effort-range") else None)
            page.hover(f"{panel} .chat-effort-head")
            page.wait_for_timeout(450)
            assert _facts(page, panel)["open"] == "false", "no hover-open where the pills would step aside"
            page.mouse.down()
            page.mouse.up()
            _wait(page, "el.dataset.open === 'true'", panel)
            page.wait_for_timeout(300)
            assert saves == [], saves
            page.wait_for_timeout(300)
            opened = _facts(page, panel)
            assert opened["fit"] == "solo" and not opened["pillsShown"], opened
            assert abs(opened["rowHeight"] - closed["rowHeight"]) <= 1, (opened, closed)
            assert opened["right"] <= opened["rowRight"] + 1 and not opened["areaScroll"] and not opened["docScroll"], opened
            assert opened["top"] < opened["inputTop"], opened
            _assert_strip_geometry(opened)
            _shot(page, tmp_path, "composer-project-pane-open-chromium")
            # A change in the pane reaches Main through /api/state.
            page.focus(f"{panel} .chat-effort-max")
            with page.expect_response(lambda r: r.url.endswith("/api/owner/effort-range") and r.status == 200, timeout=30_000):
                page.keyboard.press("ArrowRight")
            _wait(page, "el.getAttribute('aria-valuetext') !== 'High'", f"{panel} .chat-effort-max")
            page.wait_for_function(f"() => document.querySelector('{CONTROL} .chat-effort-max').getAttribute('aria-valuetext') === 'X-High'",
                                   timeout=30_000)
        finally:
            browser.close()


def test_drag_and_keyboard_save_the_full_triple_and_reload_reads_it_back(direct_server_with_data, tmp_path):  # noqa: F811
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    with sync_playwright() as pw:
        browser, page = _launch(pw, "chromium", viewport={"width": 1440, "height": 900})
        try:
            _open_chat(page, url, width=1440)
            page.click(HEAD)
            _wait(page, "el.dataset.open === 'true'")
            page.wait_for_timeout(450)
            before = _facts(page)
            assert (before["min"], before["rec"], before["max"]) == ("Low", "Medium", "High"), before
            # Drag the max bracket up to Ultra: the full triple is posted once the pointer lifts.
            cap = page.locator(f"{CONTROL} .chat-effort-max").bounding_box()
            x, y = _segment_center(page, "ultra")
            page.mouse.move(cap["x"] + cap["width"] / 2, cap["y"] + cap["height"] / 2)
            page.mouse.down()
            page.mouse.move(x, y, steps=8)
            saved = []
            page.on("response", lambda r: saved.append(r) if r.url.endswith("/api/owner/effort-range") else None)
            with page.expect_response(lambda r: r.url.endswith("/api/owner/effort-range"), timeout=30_000) as waited:
                page.mouse.up()
            assert waited.value.status == 200, waited.value.text()
            assert json.loads(waited.value.request.post_data) == {"min": "low", "recommended": "medium", "max": "ultra"}
            _wait(page, "(el.dataset.saving || 'false') === 'false'")
            after = _facts(page)
            assert (after["min"], after["rec"], after["max"]) == ("Low", "Medium", "Ultra"), after
            # A tap inside the range moves the recommended level; outside, the nearest bracket.
            x, y = _segment_center(page, "xhigh")
            with page.expect_response(lambda r: r.url.endswith("/api/owner/effort-range"), timeout=30_000) as waited:
                page.mouse.click(x, y)
            assert json.loads(waited.value.request.post_data) == {"min": "low", "recommended": "xhigh", "max": "ultra"}
            x, y = _segment_center(page, "none")
            with page.expect_response(lambda r: r.url.endswith("/api/owner/effort-range"), timeout=30_000) as waited:
                page.mouse.click(x, y)
            assert json.loads(waited.value.request.post_data) == {"min": "none", "recommended": "xhigh", "max": "ultra"}
            # Keyboard: the recommended handle steps, min and max push it.
            page.focus(f"{CONTROL} .chat-effort-rec")
            with page.expect_response(lambda r: r.url.endswith("/api/owner/effort-range"), timeout=30_000) as waited:
                page.keyboard.press("ArrowLeft")
            assert json.loads(waited.value.request.post_data) == {"min": "none", "recommended": "high", "max": "ultra"}
            page.focus(f"{CONTROL} .chat-effort-min")
            with page.expect_response(lambda r: r.url.endswith("/api/owner/effort-range"), timeout=30_000) as waited:
                page.keyboard.press("End")
            assert json.loads(waited.value.request.post_data) == {"min": "ultra", "recommended": "ultra", "max": "ultra"}
            _shot(page, tmp_path, "composer-open-changed-chromium")
            # The pill pushes the bracket in its way, and a pushed bracket stays when the pointer
            # comes back within the same drag: Ultra · Ultra · Ultra → drag the pill to None and
            # back to Medium → None · Medium · Ultra.
            pill = page.locator(f"{CONTROL} .chat-effort-rec").bounding_box()
            page.mouse.move(pill["x"] + pill["width"] / 2, pill["y"] + pill["height"] / 2)
            page.mouse.down()
            x, y = _segment_center(page, "none")
            page.mouse.move(x, y, steps=10)
            _wait(page, "el.querySelector('.chat-effort-min').getAttribute('aria-valuetext') === 'None'")
            x, y = _segment_center(page, "medium")
            page.mouse.move(x, y, steps=6)
            with page.expect_response(lambda r: r.url.endswith("/api/owner/effort-range"), timeout=30_000) as waited:
                page.mouse.up()
            assert json.loads(waited.value.request.post_data) == {"min": "none", "recommended": "medium", "max": "ultra"}
            _wait(page, "(el.dataset.saving || 'false') === 'false'")
            _assert_strip_geometry(_facts(page))
            # A reload reads the saved value back from /api/state.
            page.reload(wait_until="domcontentloaded")
            page.wait_for_function(f"() => document.querySelector('{CONTROL}')?.dataset.known === 'true'", timeout=30_000)
            reloaded = _facts(page)
            assert (reloaded["min"], reloaded["rec"], reloaded["max"]) == ("None", "Medium", "Ultra"), reloaded
            page.click(HEAD)
            _wait(page, "el.dataset.open === 'true'")
            # A refused save: the server's sentence, the control back at the server's value.
            page.route("**/api/owner/effort-range", lambda route: route.fulfill(
                status=400, content_type="application/json",
                body=json.dumps({"error": "Effort range refused by the test.", "saved": False, "code": "effort_range_invalid"})))
            page.focus(f"{CONTROL} .chat-effort-max")
            page.keyboard.press("ArrowLeft")
            page.wait_for_selector(".toast", timeout=10_000)
            assert "Effort range refused by the test." in page.locator(".toast").first.inner_text()
            _wait(page, "el.querySelector('.chat-effort-max').getAttribute('aria-valuetext') === 'Ultra'")
        finally:
            browser.close()


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_a_320_px_phone_scrolls_the_strip_inside_the_control(direct_server_with_data, engine, tmp_path):  # noqa: F811
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    with sync_playwright() as pw:
        browser, page = _launch(pw, engine, viewport={"width": 320, "height": 640}, is_mobile=True, has_touch=True)
        try:
            _open_chat(page, url, width=320, height=640)
            closed = _facts(page)
            page.tap(HEAD)
            _wait(page, "el.dataset.open === 'true'")
            _assert_strip_geometry(_facts(page))
            page.wait_for_timeout(300)
            opened = _facts(page)
            assert opened["fit"] == "overflow" and not opened["pillsShown"], opened
            assert opened["stripScrolls"] and opened["stripScroll"] in ("start", "middle"), opened
            assert not opened["docScroll"] and not opened["areaScroll"], opened
            assert abs(opened["rowHeight"] - closed["rowHeight"]) <= 1, (opened, closed)
            assert opened["headWidth"] > 20, "the round button stays"
            _assert_strip_geometry(opened)
            _shot(page, tmp_path, f"composer-phone-320-open-{engine}")
            # The hidden end is reached by scrolling the strip; a tap there moves the bracket.
            page.evaluate(f"() => {{ const s = document.querySelector('{CONTROL} .chat-effort-strip'); s.scrollLeft = s.scrollWidth; }}")
            _wait(page, "el.querySelector('.chat-effort-strip').dataset.scroll === 'end'")
            x, y = _segment_center(page, "ultra")
            with page.expect_response(lambda r: r.url.endswith("/api/owner/effort-range"), timeout=30_000) as waited:
                page.touchscreen.tap(x, y)
            assert json.loads(waited.value.request.post_data)["max"] == "ultra"
        finally:
            browser.close()


def test_a_change_made_before_send_lands_before_the_message(direct_server_with_data):  # noqa: F811
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    with sync_playwright() as pw:
        browser, page = _launch(pw, "chromium", viewport={"width": 1440, "height": 900})
        try:
            page.add_init_script("""(() => {
                window.__sends = [];
                const send = WebSocket.prototype.send;
                WebSocket.prototype.send = function (data) {
                    try { const frame = JSON.parse(data); if (frame.type === 'chat') window.__sends.push({ at: Date.now(), content: frame.content }); } catch (e) {}
                    return send.call(this, data);
                };
                const fetch_ = window.fetch;
                window.fetch = async (...args) => {
                    const response = await fetch_(...args);
                    if (String(args[0]).endsWith('/api/owner/effort-range')) window.__effortSaved = Date.now();
                    return response;
                };
            })();""")
            _open_chat(page, url, width=1440)

            # The save is held open by the test until Send has been asked to wait for it.
            held = []
            page.route("**/api/owner/effort-range", lambda route: held.append(route))
            page.click(HEAD)
            _wait(page, "el.dataset.open === 'true'")
            page.focus(f"{CONTROL} .chat-effort-rec")
            page.keyboard.press("ArrowRight")
            page.wait_for_function("() => true")  # let the request reach the route
            deadline = time.time() + 10
            while not held and time.time() < deadline:
                page.wait_for_timeout(50)
            assert held, "the gesture end posted the triple"
            page.fill("#chat-input", "use the new level")
            page.keyboard.press("Enter")
            # Send waits for the save: the button says so, the field keeps the draft meanwhile.
            page.wait_for_function("() => document.querySelector('#chat-send').textContent === 'Saving'", timeout=5_000)
            assert page.locator("#chat-input").input_value() == "use the new level"
            assert page.evaluate("() => window.__sends.length") == 0
            held[0].continue_()
            page.wait_for_function("() => window.__sends.length === 1", timeout=10_000)
            order = page.evaluate("() => ({ sent: window.__sends[0].at, saved: window.__effortSaved })")
            assert order["saved"] and order["sent"] >= order["saved"], order
            assert page.locator("#chat-input").input_value() == ""
        finally:
            browser.close()


def test_behavior_has_no_effort_section_and_ends_with_one_plain_startup_note(direct_server_with_data, tmp_path):  # noqa: F811
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    with sync_playwright() as pw:
        browser, page = _launch(pw, "chromium", viewport={"width": 1440, "height": 900})
        try:
            _open_chat(page, url, width=1440)
            page.locator('[data-nav-page="settings"]').click()
            page.wait_for_selector(".settings-shell", timeout=20_000)
            page.locator('[data-settings-tab="behavior"]').click()
            panel = page.locator('[data-settings-panel="behavior"]')
            # The range lives in the composer alone.
            assert panel.locator("h3", has_text="Reasoning Effort").count() == 0
            assert page.locator("[data-effort-range-summary]").count() == 0
            assert page.locator('[data-effort-target^="s-effort-"]').count() == 0
            # This test host is not the packaged desktop app: both switches are off and disabled,
            # and the one reason is said once, as a note, at the label text's edge.
            page.wait_for_selector("[data-autostart-settings]:not([hidden])", timeout=20_000)
            page.wait_for_selector("[data-autostart-shared-note]:not([hidden])", timeout=20_000)
            facts = page.evaluate("""() => {
                const panel = document.querySelector('[data-settings-panel="behavior"]');
                const sections = [...panel.querySelectorAll(':scope > .form-section')];
                const card = panel.querySelector('[data-autostart-settings]');
                const note = card.querySelector('[data-autostart-shared-note]');
                const label = card.querySelector('[data-autostart-toggle]').closest('label');
                const text = [...label.childNodes].find((node) => node.nodeType === 3 && node.textContent.trim());
                const range = document.createRange();
                range.selectNodeContents(text);
                const textLeft = range.getBoundingClientRect().left;
                const noteStyle = getComputedStyle(note);
                return {
                    last: sections.at(-1) === card,
                    note: note.textContent,
                    noteTone: note.dataset.tone || '',
                    noteDot: getComputedStyle(note, '::before').content,
                    noteColor: noteStyle.color,
                    metaColor: getComputedStyle(card.querySelector('.settings-section-copy')).color,
                    noteTextLeft: note.getBoundingClientRect().left + parseFloat(noteStyle.paddingLeft),
                    textLeft,
                    rowsHidden: [...card.querySelectorAll('[data-autostart-status], [data-background-status]')].every((el) => el.hidden),
                    describedBy: [...card.querySelectorAll('input[type="checkbox"]')].map((box) => box.getAttribute('aria-describedby')),
                    disabled: [...card.querySelectorAll('input[type="checkbox"]')].map((box) => box.disabled),
                };
            }""")
            assert facts["last"], "Startup & background is the last card of Behavior"
            assert facts["note"] == "Available only when the host runs the packaged desktop app.", facts
            assert facts["noteTone"] == "" and facts["noteDot"] in ("none", "normal", ""), facts
            assert facts["noteColor"] == facts["metaColor"], "a fact in meta ink, not a warning colour"
            assert abs(facts["noteTextLeft"] - facts["textLeft"]) <= 1, facts
            assert facts["rowsHidden"] and facts["describedBy"] == ["settings-autostart-note"] * 2, facts
            assert facts["disabled"] == [True, True], facts
            page.locator("[data-autostart-settings]").scroll_into_view_if_needed()
            _shot(page, tmp_path, "settings-behavior-startup-chromium")
        finally:
            browser.close()


@pytest.mark.parametrize("width", [1360, 390])
def test_agents_rows_say_auto_or_the_level_in_the_model_name(role_ui, tmp_path, width):
    ui = role_ui
    roles.configure_mixed(ui)
    items = ui["settings"]["OUROBOROS_SUBAGENTS"]["items"]
    items[1]["route"] = {"kind": "api_model", "target_id": "claudexor::agy=gemini-3.7-flash-xhigh"}
    items[2]["route"] = {"kind": "agent_session", "target_id": "cursor=gpt-5.6-sol-high-fast", "credential_profile_id": "personal"}
    ui["page"].set_viewport_size({"width": width, "height": 900})
    page = roles.open_agents(ui)
    rows = page.locator("[data-subagent-row]")
    plain, api_named, session_named = rows.nth(0), rows.nth(1), rows.nth(2)
    assert plain.locator('[data-subagent-field="effort"] option[value=""]').inner_text() == "Auto (reviews at the top of the chat range)"
    plain.locator('[data-subagent-field="review_eligible"]').uncheck()
    assert plain.locator('[data-subagent-field="effort"] option[value=""]').inner_text() == "Auto (chat range)"
    for row, expected in ((api_named, "X-High · in the model name"), (session_named, "High · in the model name")):
        assert row.locator("[data-subagent-effort-named]").inner_text() == expected
        assert row.locator('[data-subagent-field="effort"]').count() == 0, "a named row offers no select"
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    _shot(page, tmp_path, f"settings-agents-effort-{width}")
    session_named.locator("[data-subagent-effort-named]").scroll_into_view_if_needed()
    _shot(page, tmp_path, f"settings-agents-effort-named-{width}")
