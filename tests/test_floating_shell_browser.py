"""Approved floating shell on the real SPA, with isolated HTTP/WS fixture data."""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

from ouroboros.consciousness_view import describe_consciousness_state
from tests.test_room_work_browser import PROJECT, Transport, activity, progress, reading
from tests.test_subscription_setup_browser import capture, subscription_ui as subscription_ui
from tests.test_window_chrome_browser import record

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]


@pytest.fixture(params=["chromium", "webkit"])
def shell_ui(request, monkeypatch):
    monkeypatch.setenv("OUROBOROS_UI_BROWSER_ENGINE", request.param)
    ui = request.getfixturevalue("subscription_ui")
    ui["engine"] = request.param
    ui["engine_version"] = ui["page"].context.browser.version
    return ui


def _gap_history(page, transport):
    reads = []

    def history(route):
        chat_id = int(parse_qs(urlparse(route.request.url).query).get("chat_id", ["1"])[0])
        reads.append(chat_id)
        route.fulfill(json={"messages": transport.history[chat_id], "progress": [],
                            "has_more": True, "page_cursor": "recent", "next_cursor": "older"})

    page.route("**/api/chat/history*", history)
    return reads


def test_hidden_main_defers_gap_notice_until_shown_without_reading_again(shell_ui):
    page = shell_ui["page"]
    transport = Transport(page, history={1: reading(14)})
    reads = _gap_history(page, transport)
    page.goto(shell_ui["url"] + "/#settings")
    page.wait_for_selector("#page-settings.active")
    page.wait_for_function("document.querySelector('#page-chat .chat-load-older-note')?.textContent.includes('gaps')")
    hidden_notices = page.locator(".toast-stack--room .toast").count()
    before = len(reads)
    page.locator('[data-nav-page="chat"]').first.click()
    page.wait_for_selector("#page-chat.active")
    shown = page.locator("#page-chat .toast").count()
    capture(page, f"hidden-main-shown-{shell_ui['engine']}")
    record(f"hidden-main-{shell_ui['engine']}", {"hidden_notices": hidden_notices, "shown_notices": shown,
                                               "reads_before": before, "reads_after": len(reads)})
    assert hidden_notices == 0, "a hidden room must not spend the document's notice"
    page.wait_for_selector("#page-chat .toast-dismiss")
    assert len(reads) == before, "presenting the pending notice must not request history"
    page.click("#page-chat .toast-dismiss")
    page.wait_for_selector("#page-chat .toast", state="detached")
    page.locator('[data-nav-page="settings"]').first.click()
    page.locator('[data-nav-page="chat"]').first.click()
    assert page.locator(".toast-stack--room .toast").count() == 0
    assert "gaps" in page.locator("#page-chat .chat-load-older-note").inner_text()


@pytest.mark.parametrize("chat_id", [1, 42])
def test_delayed_gap_waits_for_drawer_close_without_another_history_read(shell_ui, chat_id):
    page = shell_ui["new_page"](viewport={"width": 390, "height": 844}, has_touch=True)
    transport = Transport(page, projects=[PROJECT], history={1: reading(14), 42: reading(14, room="Project")})
    pending, reads = [], []

    def history(route):
        room = int(parse_qs(urlparse(route.request.url).query).get("chat_id", ["1"])[0])
        reads.append(room)
        if room == chat_id:
            pending.append(route)
        else:
            route.fulfill(json={"messages": transport.history[room], "progress": [], "has_more": False})

    page.route("**/api/chat/history*", history)
    page.goto(shell_ui["url"])
    page.wait_for_selector("#chat-input")
    host = "#page-chat" if chat_id == 1 else "#project-panel"
    if chat_id == 42:
        page.evaluate("p => window.dispatchEvent(new CustomEvent('ouro:open-project', {detail: {project: p}}))", PROJECT)
        page.wait_for_selector("#project-panel.open .chat-input")
    before_drawer = page.locator('.toast-stack--room .toast').all_text_contents()
    page.locator(f"{host} [data-mobile-nav-toggle]").click()
    page.wait_for_function("document.body.classList.contains('nav-drawer-open') && window.__ouroLayers.top() === 'drawer'")
    assert pending, "the response must still be held when the drawer opens"
    for route in pending:
        route.fulfill(json={"messages": transport.history[chat_id], "progress": [],
                            "has_more": True, "page_cursor": "recent", "next_cursor": "older"})
    page.wait_for_function("host => document.querySelector(host + ' .chat-load-older-note')?.textContent.includes('gaps')", arg=host)
    hidden = page.locator(".toast-stack--room .toast").count()
    facts = page.evaluate("""async host => ({bodyClass: document.body.className,
        inert: Boolean(document.querySelector(host).closest('[inert]')), top: window.__ouroLayers.top(),
        guardServed: (await (await fetch('/static/modules/chat.js')).text()).includes("!messagesDiv.closest('[inert]')")})""", host)
    record(f"drawer-gap-hidden-{chat_id}-{shell_ui['engine']}", {"notices": hidden, "reads": reads, "before_drawer": before_drawer,
        "after_text": page.locator('.toast-stack--room .toast').all_text_contents(), **facts})
    assert hidden == 0, "a drawer-covered room must not spend the document's notice"
    page.wait_for_timeout(6200)  # longer than a mistakenly hidden notice's actual lifetime
    before = len(reads)
    page.keyboard.press("Escape")
    page.wait_for_selector(f"{host} .toast-dismiss")
    assert len(reads) == before, "exposing the notice must not fetch or move the history"
    capture(page, f"drawer-gap-shown-{chat_id}-{shell_ui['engine']}")
    page.locator(f"{host} .toast-dismiss").click()
    page.wait_for_selector(f"{host} .toast", state="detached")
    page.locator(f"{host} [data-mobile-nav-toggle]").click()
    page.keyboard.press("Escape")
    page.wait_for_function("!document.body.classList.contains('nav-drawer-open')")
    assert page.locator(".toast-stack--room .toast").count() == 0
    assert len(reads) == before


def test_project_header_reserve_is_spent_once_in_each_flow_state(shell_ui):
    page = shell_ui["new_page"](viewport={"width": 390, "height": 844}, has_touch=True)
    transport = Transport(page, projects=[PROJECT], history={1: reading(14), 42: reading(14, room="Project")})
    _gap_history(page, transport)
    page.goto(shell_ui["url"])
    page.wait_for_selector("#chat-input")
    page.evaluate("p => window.dispatchEvent(new CustomEvent('ouro:open-project', {detail: {project: p}}))", PROJECT)
    page.wait_for_selector("#project-panel.open .chat-messages .chat-bubble")
    page.wait_for_selector("#project-panel .rw-trigger")
    page.wait_for_timeout(400)
    states = page.evaluate("""() => {
        const panel = document.querySelector('#project-panel'), bar = panel.querySelector('.project-panel-bar');
        const feed = panel.querySelector('.chat-messages'), status = panel.querySelector('.chat-panel-statusbar');
        const note = document.createElement('span'); note.className = 'chat-load-older-note chat-history-status';
        note.textContent = 'Some saved history could not be loaded.';
        const facts = () => ({barBottom: bar.getBoundingClientRect().bottom,
            feedTop: feed.getBoundingClientRect().top, feedPadding: parseFloat(getComputedStyle(feed).paddingTop),
            statusTop: status.getBoundingClientRect().top, statusBottom: status.getBoundingClientRect().bottom,
            statusPadding: parseFloat(getComputedStyle(status).paddingTop), statusChildren: status.innerHTML});
        const normal = facts(); status.append(note); const failed = facts();
        note.textContent = 'Saved position could not be restored exactly.'; const approximate = facts();
        document.body.classList.add('keyboard-open'); const keyboardNote = facts();
        note.remove(); const keyboard = facts(); document.body.classList.remove('keyboard-open');
        return {normal, failed, approximate, keyboardNote, keyboard};
    }""")
    record(f"project-flow-{shell_ui['engine']}", states)
    assert states["normal"]["feedPadding"] >= 50
    for name in ("failed", "approximate"):
        assert states[name]["feedPadding"] == 8, (name, states)
        assert states[name]["statusPadding"] >= 50
    assert states["keyboard"]["feedPadding"] == 0, states
    assert abs(states["keyboard"]["feedTop"] - states["keyboard"]["barBottom"]) <= 1, states
    assert states["keyboardNote"]["feedPadding"] == 0 and states["keyboardNote"]["statusPadding"] == 8, states


def test_hidden_main_does_not_take_visible_projects_notice(shell_ui):
    page = shell_ui["new_page"](viewport={"width": 390, "height": 844}, has_touch=True)
    transport = Transport(page, projects=[PROJECT], history={1: reading(14), 42: reading(14, room="Project")})
    _gap_history(page, transport)
    page.goto(shell_ui["url"] + "/#settings")
    page.wait_for_function("document.querySelector('#page-chat .chat-load-older-note')?.textContent.includes('gaps')")
    page.evaluate("p => window.dispatchEvent(new CustomEvent('ouro:open-project', {detail: {project: p}}))", PROJECT)
    page.wait_for_selector("#project-panel.open .toast-dismiss")
    assert page.locator("#page-chat .toast").count() == 0
    page.click("#project-panel .toast-dismiss")
    page.wait_for_selector("#project-panel .toast", state="detached")
    page.click("#project-panel-close")
    page.wait_for_function("!document.body.classList.contains('project-panel-open')")
    assert page.locator(".toast-stack--room .toast").count() == 0


@pytest.mark.parametrize("chat_id,keyboard", [(1, False), (1, True), (42, False), (42, True)])
def test_work_navigation_clears_persistent_history_error(shell_ui, chat_id, keyboard):
    page = shell_ui["new_page"](viewport={"width": 390, "height": 844}, has_touch=True)
    rows = [*reading(14), progress("gap-work", "Comparing sources.", 30, suggested_name="Compare sources"),
            *reading(14, minute=35)]
    Transport(page, projects=[PROJECT], history={1: rows, 42: rows},
              activities=[activity("gap-work", chat_id=chat_id, **({"project_id": PROJECT["id"]} if chat_id == 42 else {}))])

    def history(route):
        query = parse_qs(urlparse(route.request.url).query)
        if query.get("cursor"):
            route.fulfill(status=503, json={"error": "History read failed"})
        else:
            route.fulfill(json={"messages": rows, "progress": [], "has_more": True,
                                "page_cursor": "recent", "next_cursor": "older"})

    page.route("**/api/chat/history*", history)
    page.goto(shell_ui["url"])
    page.wait_for_selector("#chat-input")
    host = "#page-chat" if chat_id == 1 else "#project-panel"
    if chat_id == 42:
        page.evaluate("p => window.dispatchEvent(new CustomEvent('ouro:open-project', {detail: {project: p}}))", PROJECT)
        page.wait_for_selector("#project-panel.open .chat-input")
    if keyboard:
        page.evaluate("document.body.classList.add('keyboard-open')")
    page.locator(f"{host} .chat-load-older-btn").click()
    page.locator(f"{host} .chat-history-status").wait_for(state="visible")
    page.locator(f"{host} .rw-trigger").click()
    page.locator(f'{host} .rw-row[data-id="gap-work"] .rw-main').click()
    page.wait_for_function("host => document.querySelector(host + ' .rw-shade:not(.cx-shade)').hidden", arg=host)
    facts = page.evaluate("""host => {
        const root = document.querySelector(host);
        return {card: root.querySelector('.chat-live-card[data-task-id="gap-work"]').getBoundingClientRect().top,
                error: root.querySelector('.chat-history-status').getBoundingClientRect().bottom};
    }""", host)
    record(f"work-navigation-error-{chat_id}-{keyboard}-{shell_ui['engine']}", facts)
    capture(page, f"work-navigation-error-{chat_id}-{keyboard}-{shell_ui['engine']}")
    assert facts["card"] >= facts["error"], facts


def test_gap_translation_reaches_both_in_feed_copies(shell_ui):
    page = shell_ui["page"]
    transport = Transport(page, history={1: reading(14)})
    _gap_history(page, transport)
    translated = "Часть сохранённой истории не загружена. В показанных сообщениях могут быть пропуски."
    payload = {"language": "ru", "english": False, "entries": {"code:history.gap": {"text": translated}}}
    page.route("**/api/ui/i18n*", lambda route: route.fulfill(json=payload))
    page.goto(shell_ui["url"])
    page.wait_for_selector("#page-chat .chat-load-older-note")
    page.wait_for_function("text => document.querySelector('#page-chat .chat-load-older-note').textContent === text",
                           arg=translated, timeout=3000)
    # Exercise the simultaneous read-error/approximation copy through the real producer in the skipped feed.
    result = page.evaluate("""async () => {
        const {createHistoryControls} = await import('/static/modules/chat_render_batch.js');
        const feed = document.createElement('div'); feed.className = 'chat-messages';
        document.body.append(feed); const status = document.createElement('div');
        const controls = createHistoryControls(feed, status);
        controls.render({initialized: true, error: new Error('read failed')}, {gaps: true}, true);
        const text = feed.querySelector('.chat-history-gap').textContent;
        controls.destroy(); feed.remove(); return text;
    }""")
    assert result == translated


@pytest.mark.parametrize("width,theme", [(375, "dark"), (375, "light"), (1100, "dark"), (1100, "light"),
                                       (390, "dark"), (390, "light"), (1280, "dark"), (1280, "light")])
def test_floating_headers_gap_notice_and_immediate_selection(shell_ui, tmp_path, width, theme):
    ui = shell_ui
    page = ui["new_page"](viewport={"width": width, "height": 844}, has_touch=width < 720,
                          device_scale_factor=2 if width == 390 else 1)
    page.add_init_script(f"localStorage.setItem('ouroboros.theme', '{theme}');")
    project = {**PROJECT, "name": "Floating shell — a long project title that must yield to its status"}
    transport = Transport(page, projects=[project], activities=[
        activity("main-work", "thinking"),
        activity("project-work", "thinking", chat_id=42, project_id=project["id"]),
    ], history={1: reading(14), 42: reading(14, room="Project")})
    # Consciousness sleeps until its next wake: the bar's chip says it and opens its Shade.
    census = transport.body
    next_wake = datetime.now(timezone.utc) + timedelta(hours=1)
    transport.body = lambda **kwargs: {**census(**kwargs), "bg_consciousness_enabled": True,
                                       "bg_consciousness_state": describe_consciousness_state(
                                           {"next_wake_at": next_wake.isoformat(), "spent_24h_usd": 1.5, "daily_usd": 20}, True)}

    def history(route):
        chat_id = int(parse_qs(urlparse(route.request.url).query).get("chat_id", ["1"])[0])
        route.fulfill(json={"messages": transport.history[chat_id], "progress": [],
                            "has_more": True, "page_cursor": "recent", "next_cursor": "older"})

    page.route("**/api/chat/history*", history)
    page.route("**/api/consciousness", lambda route: route.fulfill(json={"enabled": True, "recent_wakes": []}))
    try:
        page.goto(ui["url"] + "/")
        page.wait_for_selector("#page-chat .rw-trigger")
        page.wait_for_selector(".toast-stack--room .toast-dismiss")
        page.wait_for_function("getComputedStyle(document.querySelector('.toast-stack--room .toast')).opacity === '1'")
        facts = page.evaluate("""() => {
            const rect = s => document.querySelector(s).getBoundingClientRect();
            const header = document.querySelector('#page-chat .chat-page-header');
            return {header: rect('#page-chat .chat-page-header'), toast: rect('.toast-stack--room .toast'),
                composer: rect('#page-chat #chat-input-area'), background: getComputedStyle(header).backgroundColor,
                before: getComputedStyle(header, '::before').content};
        }""")
        assert facts["background"] == "rgba(0, 0, 0, 0)"
        assert facts["before"] in ("none", "normal")
        assert facts["toast"]["top"] > facts["header"]["bottom"]
        assert facts["toast"]["bottom"] <= facts["composer"]["top"]
        assert not page.locator("#page-chat .chat-page-note").is_visible()
        assert "Shown messages may have gaps" in page.locator("#page-chat .chat-load-older-note").inner_text()
        # The consciousness chip opens its list, hit-testable beside the toast, without activating a command.
        page.click("#page-chat .cx-chip:not([hidden])")
        page.locator('#page-chat .cx-shade .cx-toggle').click(trial=True)
        if theme == 'dark':
            # The chip's list is opaque like the work list: its base layer at full alpha (DESIGN principle 2).
            backing = page.locator('#page-chat .cx-shade').evaluate("el => getComputedStyle(el).backgroundColor")
            assert backing.startswith('rgb('), backing
        page.screenshot(path=str(tmp_path / f"main-shade-{width}-{theme}.png"))
        capture(page, f"main-shade-{width}-{theme}-{ui['engine']}")
        if width == 390:
            evidence = Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", tmp_path))
            page.locator('#page-chat .chat-page-header').screenshot(
                path=str(evidence / f"header-2x-{theme}-{ui['engine']}.png"))
            # Readable local backing must survive a browser that does not blur the transcript.
            no_blur = page.add_style_tag(content=".chat-page-header :is(.app-page-title-row, .ui-chrome-control) { backdrop-filter: none !important; -webkit-backdrop-filter: none !important; }")
            capture(page, f"main-no-blur-{width}-{theme}-{ui['engine']}")
            facts["capsuleBackingWithoutBlur"] = page.locator('#page-chat .app-page-title-row').evaluate(
                "el => ({color: getComputedStyle(el).backgroundColor, image: getComputedStyle(el).backgroundImage})")
            no_blur.evaluate("el => el.remove()")
        page.click("#page-chat .cx-chip")
        page.wait_for_selector("#page-chat .cx-shade", state="hidden")
        page.click(".toast-stack--room .toast-dismiss")
        page.wait_for_selector(".toast-stack--room .toast", state="detached")
        if width < 720:
            page.click('#page-chat [data-mobile-nav-toggle]')
            page.wait_for_function("window.__ouroLayers.top() === 'drawer'")
        page.locator(f'[data-project-id="{project["id"]}"]').first.click()
        page.wait_for_selector("#project-panel:not([hidden]) .rw-trigger")
        page.wait_for_function("!document.body.classList.contains('nav-drawer-open')")
        assert page.locator(".shell-pill").count() == 0
        assert page.locator(".toast-stack--room .toast").count() == 0, "room switch does not repeat the notice"
        page.wait_for_function("""() => [...document.getAnimations()].every(a =>
            a.effect?.getTiming().iterations === Infinity || a.playState !== 'running')""")
        geometry = page.locator("#project-panel").evaluate("""panel => {
            const rect = s => panel.querySelector(s).getBoundingClientRect();
            return {capsule: rect('.app-page-title-row'), caret: rect('.rw-caret'),
                mark: rect('.status-mark'), trigger: rect('.rw-trigger'), close: rect('#project-panel-close')};
        }""")
        assert geometry["capsule"]["right"] - geometry["caret"]["right"] >= 10
        assert geometry["caret"]["left"] - geometry["mark"]["right"] >= 4
        if width < 720:
            assert geometry["trigger"]["height"] >= 44
        title = page.locator('#project-panel .app-page-title')
        title.click()
        page.wait_for_selector('#project-panel .rw-shade:not(.cx-shade):not([hidden])')
        title.click()
        page.wait_for_selector('#project-panel .rw-shade:not(.cx-shade)', state='hidden', timeout=2000)
        page.locator("#project-panel .rw-trigger").focus()
        page.keyboard.press("Enter")
        page.wait_for_selector("#project-panel .rw-shade:not(.cx-shade):not([hidden])")
        page.screenshot(path=str(tmp_path / f"project-shade-{width}-{theme}.png"))
        capture(page, f"project-shade-{width}-{theme}-{ui['engine']}")
        record(f"floating-{width}-{theme}-{ui['engine']}", {"main": facts, "project": geometry,
                                                         "engine_version": ui["engine_version"]})
        page.keyboard.press("Escape")
        assert page.locator("#project-panel .rw-trigger").evaluate("el => el === document.activeElement")
    finally:
        page.close()


def test_toast_focus_timeout_and_disposal(shell_ui):
    page = shell_ui["page"]
    Transport(page)
    page.goto(shell_ui["url"] + "/")
    page.wait_for_selector("#chat-input")
    page.evaluate("""async () => {
        const {showToast} = await import('/static/modules/toast.js');
        window.testNotice = showToast('Focused notice', 'info', {ttl: 100,
            dismissLabel: 'Close test notice', host: document.querySelector('#page-chat')});
        window.testNotice.querySelector('button').focus();
    }""")
    page.wait_for_timeout(250)
    assert page.get_by_label("Close test notice").count() == 1
    assert "shell-icon-control" in page.get_by_label("Close test notice").get_attribute("class")
    page.locator("#chat-input").focus()
    page.wait_for_selector("[aria-label='Close test notice']", state="detached")
    page.evaluate("""async () => {
        const {showToast} = await import('/static/modules/toast.js');
        const notice = showToast('Dispose without animation', 'info', {ttl: 6000});
        notice.dispose();
        if (notice.isConnected) throw new Error('dispose left the toast mounted');
    }""")


def test_status_fitting_skips_unchanged_paints_and_tracks_capsule_width(shell_ui):
    page = shell_ui["new_page"](viewport={"width": 390, "height": 844})
    transport = Transport(page, activities=[activity("main-work", "thinking")], history={1: reading(14)})
    page.goto(shell_ui["url"])
    page.wait_for_selector("#page-chat .rw-trigger:not(:disabled)")
    page.wait_for_timeout(400)
    page.evaluate("""() => {
        const badge = document.querySelector('#page-chat .rw-trigger .status-badge');
        const width = Object.getOwnPropertyDescriptor(Element.prototype, 'scrollWidth').get;
        window.__wordReads = 0;
        Object.defineProperty(badge, 'scrollWidth', {get() { window.__wordReads++; return width.call(this); }});
    }""")
    for _ in range(3):
        transport.refresh()
    unchanged = page.evaluate("window.__wordReads")
    assert unchanged == 0, "unchanged census paints must not measure and rewrite the status word"
    title = page.locator("#page-chat .app-page-title")
    title.evaluate("el => { el.textContent = 'A long title that leaves room only for the mark and caret'; }")
    page.wait_for_function("document.querySelector('#page-chat .rw-trigger').dataset.word === 'off'")
    page.wait_for_timeout(250)
    narrow = page.evaluate("window.__wordReads")
    page.wait_for_timeout(250)
    assert page.evaluate("window.__wordReads") == narrow, "observing the fitted capsule must settle"
    title.evaluate("el => { el.textContent = 'Main'; }")
    page.wait_for_function("document.querySelector('#page-chat .rw-trigger').dataset.word !== 'off'")
    capsule = page.locator("#page-chat .app-page-title-row")
    rest = capsule.evaluate("el => getComputedStyle(el).backgroundImage")
    title.hover()
    page.wait_for_timeout(160)
    hover = capsule.evaluate("el => getComputedStyle(el).backgroundImage")
    assert hover != rest, "hovering the title answers across the whole capsule"
    title.click()
    page.wait_for_selector("#page-chat .rw-shade:not([hidden])")
    title.click()
    page.wait_for_selector("#page-chat .rw-shade", state="hidden", timeout=2000)
    title.click()
    page.wait_for_selector("#page-chat .rw-shade:not([hidden])")
    record(f"status-fit-{shell_ui['engine']}", {"unchangedReads": unchanged, "narrowReads": narrow,
                                               "rest": rest, "hover": hover})
    page.keyboard.press("Escape")
    assert page.locator("#page-chat .rw-trigger").evaluate("el => el === document.activeElement")


def test_separate_ui_history_oracle_sees_in_feed_gaps(shell_ui):
    from tests.test_separate_ui_browser import _history_notice

    page = shell_ui["page"]
    page.set_content('''<div id="page-chat"><span class="chat-history-status" hidden>Read failed</span>
        <div class="chat-load-older has-gaps"><span class="chat-load-older-note">History has gaps</span>
        <span class="chat-history-gap" hidden>Separate gap</span></div></div>''')
    assert _history_notice(page) == "History has gaps"
    page.locator('.chat-load-older-note').evaluate("el => { el.hidden = true; }")
    page.locator('.chat-history-gap').evaluate("el => { el.hidden = false; }")
    assert _history_notice(page) == "Separate gap"
    page.locator('.chat-load-older').evaluate("el => { el.classList.remove('has-gaps'); }")
    assert _history_notice(page) == ""
    page.locator('.chat-history-status').evaluate("el => { el.hidden = false; }")
    assert _history_notice(page) == "Read failed"
