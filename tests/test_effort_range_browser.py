"""The effort range control in the real composer (DECISIONS v3 §6; DESIGN "Composer owner controls").

On a real server in Chromium and WebKit: the round button after the pills opens inline into the
strip — by hover (an accelerator) and by press (which pins) on a desktop with a mouse, by press
alone on a phone; Esc closes and returns focus, an outside press closes; the open strip takes its
own line in a narrow composer column (a 390 px phone, a ~440 px Project pane on a 1440 px screen)
without a horizontal page scroll, and the composer reserve follows; a drag and the keyboard save
the full triple through the owner endpoint and every composer reads the one value back from
/api/state; a change made just before Send lands before the message; Settings → Behavior reads the
range only, and Settings → Agents says Auto or the level in the model name.
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
        resetAtDefault: el.querySelector('.chat-effort-reset').dataset.atDefault,
        hoverMedia: matchMedia('%s').matches,
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
            _shot(page, tmp_path, f"composer-closed-{engine}-dark")
            # Hover opens after ~160 ms; the strip stays on the same line at this width.
            page.hover(HEAD)
            _wait(page, "el.dataset.open === 'true'")
            opened = _facts(page)
            assert abs(opened["top"] - opened["pillsTop"]) <= 2, opened
            assert opened["right"] <= opened["rowRight"] + 1 and not opened["docScroll"], opened
            assert opened["recTabIndex"] == 0, opened
            page.wait_for_timeout(450)  # the open animation
            _shot(page, tmp_path, f"composer-open-{engine}-dark")
            # Leaving closes after ~450 ms when nothing pinned it.
            page.mouse.move(20, 200)
            _wait(page, "el.dataset.open === 'false'")
            # Hover again, then a press pins it: leaving no longer closes.
            page.hover(HEAD)
            _wait(page, "el.dataset.open === 'true'")
            page.click(HEAD)
            page.mouse.move(20, 200)
            page.wait_for_timeout(700)
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
def test_phone_press_opens_on_its_own_line_without_horizontal_scroll(direct_server_with_data, engine, tmp_path):  # noqa: F811
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    with sync_playwright() as pw:
        browser, page = _launch(pw, engine, viewport={"width": 390, "height": 844}, is_mobile=True, has_touch=True)
        try:
            _open_chat(page, url, width=390, height=844)
            closed = _facts(page)
            assert closed["hoverMedia"] is False, closed
            assert abs(closed["swarmHeight"] - closed["sendHeight"]) <= 1, closed
            assert abs(closed["height"] - closed["sendHeight"]) <= 1, "the round button keeps the pills' height"
            assert closed["top"] < closed["inputTop"] and not closed["docScroll"] and not closed["areaScroll"], closed
            _shot(page, tmp_path, f"composer-phone-closed-{engine}")
            page.tap(HEAD)
            _wait(page, "el.dataset.open === 'true'")
            page.wait_for_timeout(500)  # the open animation and the ResizeObserver
            opened = _facts(page)
            # Its own line under the pills, no wider than the row, no page or composer scroll.
            assert opened["top"] >= opened["pillsBottom"] - 1, opened
            assert opened["right"] <= opened["rowRight"] + 1, opened
            assert not opened["docScroll"] and not opened["areaScroll"], opened
            assert opened["top"] < opened["inputTop"], opened
            # The reserve follows the taller composer.
            assert opened["reserve"] >= opened["areaHeight"] + 16 - 1, opened
            assert opened["areaHeight"] > closed["areaHeight"], (opened, closed)
            _shot(page, tmp_path, f"composer-phone-open-{engine}")
            # A tap outside closes; a second tap on the button toggles.
            page.tap("#chat-messages")
            _wait(page, "el.dataset.open === 'false'")
            page.tap(HEAD)
            _wait(page, "el.dataset.open === 'true'")
            page.tap(HEAD)
            _wait(page, "el.dataset.open === 'false'")
        finally:
            browser.close()


def test_project_pane_strip_takes_its_own_line_and_shows_the_same_value(direct_server_with_data, tmp_path):  # noqa: F811
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
            page.click(f"{panel} .chat-effort-head")
            _wait(page, "el.dataset.open === 'true'", panel)
            page.wait_for_timeout(500)
            opened = _facts(page, panel)
            assert opened["top"] >= opened["pillsBottom"] - 1, opened
            assert opened["right"] <= opened["rowRight"] + 1 and not opened["areaScroll"] and not opened["docScroll"], opened
            assert opened["top"] < opened["inputTop"], opened
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
            assert after["resetAtDefault"] == "false", after
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
            # A reload reads the saved value back from /api/state, then the reset returns to the default.
            page.reload(wait_until="domcontentloaded")
            page.wait_for_function(f"() => document.querySelector('{CONTROL}')?.dataset.known === 'true'", timeout=30_000)
            reloaded = _facts(page)
            assert (reloaded["min"], reloaded["rec"], reloaded["max"]) == ("Ultra", "Ultra", "Ultra"), reloaded
            page.click(HEAD)
            _wait(page, "el.dataset.open === 'true'")
            with page.expect_response(lambda r: r.url.endswith("/api/owner/effort-range"), timeout=30_000) as waited:
                page.click(f"{CONTROL} .chat-effort-reset")
            assert json.loads(waited.value.request.post_data) == {"min": "low", "recommended": "medium", "max": "high"}
            _wait(page, "el.querySelector('.chat-effort-reset').dataset.atDefault === 'true'")
            # A refused save: the server's sentence, the control back at the server's value.
            page.route("**/api/owner/effort-range", lambda route: route.fulfill(
                status=400, content_type="application/json",
                body=json.dumps({"error": "Effort range refused by the test.", "saved": False, "code": "effort_range_invalid"})))
            page.focus(f"{CONTROL} .chat-effort-max")
            page.keyboard.press("ArrowRight")
            page.wait_for_selector(".toast", timeout=10_000)
            assert "Effort range refused by the test." in page.locator(".toast").first.inner_text()
            _wait(page, "el.querySelector('.chat-effort-max').getAttribute('aria-valuetext') === 'High'")
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


def test_behavior_reads_the_range_only(direct_server_with_data, tmp_path):  # noqa: F811
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
            line = page.locator("[data-effort-range-summary]")
            line.wait_for(timeout=20_000)
            page.wait_for_function("() => document.querySelector('[data-effort-range-summary]').dataset.i18nFmt", timeout=20_000)
            assert line.inner_text() == ("Effort range: Low · Medium · High (minimum · recommended · maximum). "
                                         "Change it with the round Effort button next to Swarm in the chat.")
            assert page.locator('[data-effort-target^="s-effort-"]').count() == 0
            line.scroll_into_view_if_needed()
            _shot(page, tmp_path, "settings-behavior-effort-chromium")
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
