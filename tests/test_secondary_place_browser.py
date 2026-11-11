"""A place left and shown again keeps what the owner did there, and a hidden Logs panel costs no drawing (owner
2026-11-03; DESIGN §10 "Moving between places", "History edges"; ARCHITECTURE §3 "Dashboard").

Real server and real page. Main left reading an older passage returns to it, left at its end follows it. Settings keeps
its page, a value being typed and its place across other pages. Logs takes a burst while hidden: no node, no history read; shown,
it draws the newest entries at once and the rest in chunks, keeps its filters, Raw is written when opened, and a live
event for the card holding the selection and the focus leaves them where they were. A live event for the open card the
owner reads inside, already the newest entry, keeps them on the row they were reading.
"""
from __future__ import annotations

import json
import re

import pytest

from ouroboros import dialogue_store
from tests._dialogue import advance_room_revision
from tests._room_browser_shared import _nav
from tests.settings_browser_helpers import open_settings, settings_page
from tests.test_ui_smoke_playwright import direct_server_with_data as direct_server_with_data
from tests.ui_chat_viewport_smoke import _CAPTURE_TEST_SOCKET, _SETTLE_RESTORE_FRAMES

pytestmark = [pytest.mark.ui_browser]
ENGINES = ["chromium", "webkit"]
_MAIN = "#chat-messages"
_ROW_AT_TOP = """selector => {
    const feed = document.querySelector(selector);
    const top = feed.getBoundingClientRect().top;
    const node = [...feed.querySelectorAll('[data-history-id]')].find(n => n.getBoundingClientRect().bottom > top + 1);
    return node && {id: node.dataset.historyId, offset: Math.round(node.getBoundingClientRect().top - top)};
}"""
# A burst of log frames from many works, delivered as the socket delivers them.
_BURST = """([count, prefix]) => {
    const socket = window.__testSockets[0];
    for (let index = 0; index < count; index += 1) {
        const data = index % 3 === 0
            ? {type: 'task_heartbeat', phase: 'running', task_id: `${prefix}-task-${index % 40}`, ts: new Date(Date.UTC(2026, 10, 4, 1, 0, 0, index)).toISOString()}
            : {type: `burst_note_${index % 7}`, note: `${prefix} ${index}`, ts: new Date(Date.UTC(2026, 10, 4, 1, 0, 1, index)).toISOString()};
        socket.dispatchEvent(new MessageEvent('message', {data: JSON.stringify({type: 'log', data})}));
    }
}"""
# Live tool frames of one task, each its own Timeline row, delivered as the socket delivers them.
_TASK_STEPS = """([group, first, count]) => {
    const socket = window.__testSockets[0];
    for (let index = first; index < first + count; index += 1) {
        const data = {type: 'tool_call_finished', tool: `step_${index}`, is_error: false, task_id: group,
            ts: new Date(Date.now() + index).toISOString()};
        socket.dispatchEvent(new MessageEvent('message', {data: JSON.stringify({type: 'log', data})}));
    }
}"""
# Where the Logs reader is: the scroller's top and, for the card and the row read at the top, their offsets from it.
_READER_PLACE = """(body, group) => {
    const top = body.getBoundingClientRect().top, list = document.getElementById('log-entries');
    const card = list.querySelector(`[data-task-group="${group}"]`);
    const rows = [...card.querySelectorAll('.log-task-event')];
    const row = rows.find(node => node.getBoundingClientRect().bottom > top + 1);
    return {scrollTop: body.scrollTop, fromEnd: body.scrollHeight - body.scrollTop - body.clientHeight,
        card: card.getBoundingClientRect().top - top, cardBottom: card.getBoundingClientRect().bottom - body.getBoundingClientRect().bottom,
        row: rows.indexOf(row), rowOffset: row.getBoundingClientRect().top - top, rows: rows.length,
        entries: list.children.length, newest: list.lastElementChild === card,
        last: list.lastElementChild.textContent.replace(/\\s+/g, ' ').trim().slice(0, 160)};
}"""
# How far up the reader scrolls from the end into the card: to its last row with more than a screen of the card below.
_INTO_THE_CARD = """(body, group) => {
    const card = document.querySelector(`#log-entries [data-task-group="${group}"]`);
    const bottom = card.getBoundingClientRect().bottom;
    const row = [...card.querySelectorAll('.log-task-event')]
        .findLast(node => bottom - node.getBoundingClientRect().top > body.clientHeight + 120);
    return row.getBoundingClientRect().top - body.getBoundingClientRect().top;
}"""


def _page(browser, url, viewport=None):
    page = browser.new_page(viewport=viewport or {"width": 1187, "height": 734})
    page.add_init_script(f"({_CAPTURE_TEST_SOCKET})()")
    logs = []
    page.on("request", lambda request: logs.append(request.url) if "/api/logs/" in request.url else None)
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.goto(url, wait_until="domcontentloaded")
    page.wait_for_function("() => window.__testSockets?.some(s => s.readyState === 1)")
    return page, logs, errors


@pytest.mark.parametrize("engine", ENGINES)
def test_main_returns_to_the_passage_it_was_left_at_and_follows_when_left_at_its_end(direct_server_with_data, engine):
    from playwright.sync_api import sync_playwright

    data = direct_server_with_data["data_dir"]
    for index in range(50):
        dialogue_store.append(data, "chat", {"ts": f"2026-09-28T10:{index:02d}:00Z", "direction": "out", "chat_id": 1,
                                             "text": f"Main row {index:02d} " + "words " * 12})
    advance_room_revision(data, chat_id=1)
    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch()
        try:
            page, _logs, errors = _page(browser, direct_server_with_data["url"])
            page.locator(f"{_MAIN} [data-history-id]").first.wait_for(state="attached")
            page.evaluate(_SETTLE_RESTORE_FRAMES)
            box = page.locator(_MAIN).bounding_box()
            page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
            for _ in range(4):
                page.mouse.wheel(0, -500)
                page.wait_for_timeout(120)
            page.evaluate(_SETTLE_RESTORE_FRAMES)
            page.wait_for_timeout(300)
            place = page.evaluate(_ROW_AT_TOP, _MAIN)
            _nav(page, "settings")
            _nav(page, "chat")
            page.wait_for_timeout(200)
            back = page.evaluate(_ROW_AT_TOP, _MAIN)
            assert back["id"] == place["id"] and abs(back["offset"] - place["offset"]) <= 2, (place, back)

            page.locator(f"{_MAIN} ~ * .chat-scroll-bottom-btn, .chat-scroll-bottom-btn").first.click()
            page.wait_for_function("s => { const f = document.querySelector(s); return f.scrollHeight - f.scrollTop - f.clientHeight <= 8; }", arg=_MAIN)
            _nav(page, "settings")
            _nav(page, "chat")
            page.wait_for_function("s => { const f = document.querySelector(s); return f.scrollHeight - f.scrollTop - f.clientHeight <= 8; }", arg=_MAIN)
            assert not errors, errors
        finally:
            browser.close()


@pytest.mark.parametrize("engine", ENGINES)
def test_settings_keeps_its_page_a_typed_value_and_its_place_across_other_pages(direct_server_with_data, engine):
    from playwright.sync_api import expect, sync_playwright

    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch()
        try:
            page, _logs, errors = _page(browser, direct_server_with_data["url"])
            _nav(page, "settings")
            # The home: scrolled down its list, left, shown again at the same place.
            home = page.locator("#page-settings [data-settings-home]")
            expect(home.locator("[data-settings-card]")).to_be_visible()
            home.evaluate("box => { box.scrollTop = Math.min(320, box.scrollHeight - box.clientHeight); }")
            page.wait_for_timeout(200)
            place = home.evaluate("box => box.scrollTop")
            assert place > 0
            _nav(page, "dashboard")
            _nav(page, "settings")
            assert abs(home.evaluate("box => box.scrollTop") - place) <= 1, "the home's place"

            # A page with a value being typed: leaving asks nothing, and the same page with the same field comes back.
            page.locator('#page-settings [data-settings-open="computer"]').scroll_into_view_if_needed()
            assert page.evaluate("() => window.__ouroSettings.open('skill-sources')") is not False
            level = page.locator('#page-settings [data-settings-page="skill-sources"]')
            field = level.locator("#setting-OUROBOROS_CLAWHUB_REGISTRY_URL")
            expect(field).to_be_visible()
            stored = field.input_value()
            field.fill("https://clawhub.ai/api/kept-while-away")
            field.evaluate("input => { window.__typedNode = input; }")
            _nav(page, "dashboard")
            assert page.get_by_role("button", name="Stay", exact=True).count() == 0, "leaving asks nothing"
            _nav(page, "chat")
            _nav(page, "settings")
            expect(level).to_be_visible()
            expect(field).to_have_value("https://clawhub.ai/api/kept-while-away")
            assert field.evaluate("input => input === window.__typedNode"), "the same field"
            # The read made when Settings is shown again does not rewrite a field that holds typed text.
            page.wait_for_timeout(400)
            expect(field).to_have_value("https://clawhub.ai/api/kept-while-away")
            field.fill(stored)
            assert not errors, errors
        finally:
            browser.close()


@pytest.mark.parametrize("engine", ENGINES)
def test_hidden_logs_take_a_burst_without_drawing_and_shown_keep_filters_raw_selection_and_focus(direct_server_with_data, engine):
    from playwright.sync_api import expect, sync_playwright

    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch()
        try:
            page, logs, errors = _page(browser, direct_server_with_data["url"])
            page.wait_for_timeout(1500)
            entries = page.locator("#log-entries")
            assert logs == [], "a hidden panel reads no history at startup"
            page.evaluate(_BURST, [900, "away"])
            assert entries.evaluate("list => list.children.length") == 0, "hidden, the burst is data"
            assert logs == []

            shown = page.evaluate("""() => new Promise(resolve => {
                const shown = event => {
                    if (event.detail?.page !== 'dashboard') return;
                    window.removeEventListener('ouro:page-shown', shown);
                    resolve(document.getElementById('log-entries').children.length);
                };
                window.addEventListener('ouro:page-shown', shown);
                document.querySelector('[data-nav-page="dashboard"]').click();
            })""")
            assert 0 < shown <= 40, shown  # the newest chunk at once, never one rebuild of everything
            page.wait_for_function("() => document.getElementById('log-entries').children.length >= 500")
            # The four history reads are sequential; the first live chunk can paint before all start.
            page.wait_for_function("() => !document.querySelector('.logs-history-status')?.textContent.includes('Loading')")
            assert len(logs) >= 4, "the owed history is read once shown"
            assert entries.evaluate("list => list.children.length") == 500, "the bound holds"
            page.evaluate(_SETTLE_RESTORE_FRAMES)
            body = page.locator("#page-logs")  # the tab body is the Logs scroller (style.css .dashboard-body)
            assert body.evaluate("box => box.scrollHeight - box.scrollTop - box.clientHeight") <= 48, "following the newest"

            # Raw is written when opened and released when closed.
            entry = entries.locator(".log-entry:not(.log-task-card)").filter(has_text="burst_note_").last
            raw = entry.locator(".log-raw")
            assert raw.evaluate("pre => pre.textContent") == ""
            entry.get_by_role("button", name="Raw", exact=True).click()
            expect(entry.locator(".log-raw-status")).to_contain_text("bytes")
            assert json.loads(raw.inner_text())["type"].startswith("burst_note_")
            entry.get_by_role("button", name="Hide raw", exact=True).click()
            assert raw.evaluate("pre => pre.textContent") == ""

            # The owner's filter stays across another page; so does a card's open Timeline.
            chip = page.locator(".filter-chip", has_text="System")
            chip.click()
            expect(chip).not_to_have_class(re.compile(r"\bactive\b"))
            expect(chip).to_have_attribute("aria-pressed", "false")  # off (quiet chips: none is toned while all are on)
            card = entries.locator(".log-task-card").last
            card.locator("summary").click()
            rows = card.locator(".log-task-event")
            assert rows.count() >= 1
            _nav(page, "settings")
            _nav(page, "dashboard")
            expect(chip).not_to_have_class(re.compile(r"\bactive\b"))
            expect(chip).to_have_attribute("aria-pressed", "false")
            assert card.locator("details").evaluate("d => d.open")
            chip.click()

            # A live event for the card holding the selection and the focus moves the card, not them.
            group = card.get_attribute("data-task-group")
            card.locator(".log-raw-toggle").first.focus()
            page.evaluate("""group => {
                const row = document.querySelector(`[data-task-group="${group}"] .log-task-event .log-headline`);
                const range = document.createRange();
                range.setStart(row.firstChild, 0);
                range.setEnd(row.firstChild, Math.min(5, row.firstChild.length));
                const selection = getSelection();
                selection.removeAllRanges();
                selection.addRange(range);
            }""", group)
            page.evaluate(_SETTLE_RESTORE_FRAMES)
            chosen = page.evaluate("() => getSelection().toString()")
            assert chosen
            count = rows.count()
            page.evaluate("""group => window.__testSockets[0].dispatchEvent(new MessageEvent('message', {data: JSON.stringify({type: 'log', data: {
                type: 'tool_call_finished', tool: 'read_file', is_error: false, task_id: group, ts: '2026-11-04T02:00:00.000Z'}})}))""", group)
            page.evaluate(_SETTLE_RESTORE_FRAMES)
            assert card.locator(".log-task-event").count() in (count + 1, 30), "the open Timeline grew in place"
            assert page.evaluate("() => getSelection().toString()") == chosen, "the selection"
            assert page.evaluate("group => document.querySelector(`[data-task-group=\"${group}\"]`).contains(document.activeElement)", group), "the focus"
            page.evaluate("() => getSelection().removeAllRanges()")
            box = entries.bounding_box()
            page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
            page.mouse.wheel(0, -1200)
            page.wait_for_timeout(300)
            body.evaluate("""box => {
                const top=box.getBoundingClientRect().top, list=document.getElementById('log-entries');
                const node=[...list.children].find(n=>!n.hidden && n.getBoundingClientRect().bottom>top+1);
                window.logPlace={node, offset:node.getBoundingClientRect().top-top};
            }""")
            _nav(page, "settings")
            page.evaluate(_BURST, [6, "later"])
            _nav(page, "dashboard")
            page.evaluate(_SETTLE_RESTORE_FRAMES)
            assert body.evaluate("""box => logPlace.node.isConnected && Math.abs(
                logPlace.node.getBoundingClientRect().top-box.getBoundingClientRect().top-logPlace.offset)<=2"""), "old Logs reading survives a hidden burst"
            assert not errors, errors
        finally:
            browser.close()


@pytest.mark.parametrize("engine", ENGINES)
def test_logs_reader_inside_the_newest_open_card_keeps_their_place_when_that_card_updates(direct_server_with_data, engine):
    """The owner reads the middle of an open task card that is the newest entry. A live event for that task redraws the
    card where it stands, so the reader stays on the row they were reading; nothing sends them to the new bottom."""
    from playwright.sync_api import expect, sync_playwright

    group = "reader-task"
    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch()
        try:
            page, _logs, errors = _page(browser, direct_server_with_data["url"])
            _nav(page, "dashboard")
            page.wait_for_function("() => !document.querySelector('.logs-history-status')?.textContent.includes('Loading')")
            page.evaluate(_BURST, [24, "earlier"])
            steps = 20
            page.evaluate(_TASK_STEPS, [group, 0, steps])
            card = page.locator(f'#log-entries [data-task-group="{group}"]')
            card.locator("summary").click()
            expect(card.locator(".log-task-event")).to_have_count(steps)
            body = page.locator("#page-logs")  # the tab body is the Logs scroller (style.css .dashboard-body)
            box = body.bounding_box()
            page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
            # The test server's own start-up frames (worker_boot, worker_ready, worker_sha_verify) can land after the card
            # at any moment: the case is taken again until no other entry arrives during it. A take adds two rows, five
            # stay within MAX_TASK_EVENTS (30), so no row ever leaves the top of the Timeline.
            for _take in range(5):
                page.evaluate(_TASK_STEPS, [group, steps, 1])  # the card is the newest entry
                body.evaluate("box => { box.scrollTop = box.scrollHeight; }")
                page.evaluate(_SETTLE_RESTORE_FRAMES)
                page.mouse.wheel(0, body.evaluate(_INTO_THE_CARD, group))  # the owner scrolls up into the card
                page.wait_for_timeout(300)
                page.evaluate(_SETTLE_RESTORE_FRAMES)
                before = body.evaluate(_READER_PLACE, group)
                page.evaluate(_TASK_STEPS, [group, steps + 1, 1])  # one live event for the task being read
                steps += 2
                page.evaluate(_SETTLE_RESTORE_FRAMES)
                after = body.evaluate(_READER_PLACE, group)
                if before["newest"] and after["entries"] == before["entries"]:
                    break
            else:
                pytest.fail("other entries kept landing after the card: " + json.dumps([before, after]))
            assert before["card"] < 0 and before["cardBottom"] > 0 and before["row"] > 0, json.dumps(before)  # inside it
            assert before["fromEnd"] > 48, json.dumps(before)  # reading, not following the newest
            assert after["newest"] and after["rows"] == before["rows"] + 1, json.dumps(after)  # drawn where it stands
            assert abs(after["scrollTop"] - before["scrollTop"]) <= 2, json.dumps([before, after])
            assert after["row"] == before["row"] and abs(after["rowOffset"] - before["rowOffset"]) <= 2, json.dumps([before, after])
            assert abs(after["card"] - before["card"]) <= 2, json.dumps([before, after])
            assert not errors, errors
        finally:
            browser.close()


@pytest.mark.parametrize("engine", ENGINES)
def test_safety_counter_defers_its_tail_until_permissions_is_shown(direct_server_with_data, engine):
    from playwright.sync_api import expect, sync_playwright

    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch()
        try:
            page, logs, errors = _page(browser, direct_server_with_data["url"])
            _nav(page, "settings")
            settings_page(page, "").locator("[data-settings-card]").wait_for(state="visible")
            page.wait_for_timeout(400)
            tails = lambda: [url for url in logs if '/events?limit=2000' in url]
            assert tails() == [], 'the home reads no whole event tail'
            counter = open_settings(page, "permissions").locator("[data-safety-skips]")
            expect(counter).to_contain_text('let through')
            assert len(tails()) == 1
            _nav(page, 'chat')
            page.wait_for_timeout(400)
            assert len(tails()) == 1, 'a hidden counter reads no additional tail'
            _nav(page, 'settings')
            # Settings comes back on the page it showed, and the counter of that page is filled again.
            expect(settings_page(page, "permissions").locator("[data-safety-skips]")).to_contain_text('let through')
            assert not errors, errors
        finally:
            browser.close()


@pytest.mark.parametrize("engine", ENGINES)
def test_received_raw_shown_again_keeps_its_page_scroll_and_selection_and_a_page_move_starts_at_its_top(direct_server_with_data, engine):
    """R1: back on Logs, a received event's Raw rebuilds only its download; its page, scroll and selection stay."""
    import os
    from pathlib import Path

    from playwright.sync_api import expect, sync_playwright

    evidence = Path(os.environ["OUROBOROS_UI_EVIDENCE_DIR"])
    evidence.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch()
        try:
            page, _logs, errors = _page(browser, direct_server_with_data["url"])
            page.evaluate("""() => window.__testSockets[0].dispatchEvent(new MessageEvent('message', {data: JSON.stringify({type: 'log',
                data: {type: 'raw_return_probe', ts: '2026-11-04T02:00:00.000Z', payload: 'x'.repeat(200000), sentinel: 'RAW-END'}})}))""")
            _nav(page, "dashboard")
            entry = page.locator("#log-entries .log-entry:not(.log-task-card)").filter(has_text="raw_return_probe").last
            entry.get_by_role("button", name="Raw", exact=True).click()
            status = entry.locator(".log-raw-status")
            expect(status).to_contain_text("Received event")
            download = entry.get_by_role("link", name="Download full")
            expect(download).to_be_visible()
            raw = entry.locator(".log-raw")
            before = raw.evaluate("""pre => {
                pre.scrollTop = 200;
                const text = pre.firstChild, range = document.createRange();
                range.setStart(text, 40); range.setEnd(text, 120);
                getSelection().removeAllRanges(); getSelection().addRange(range);
                window.rawText = text;
                return {text: pre.textContent, top: pre.scrollTop, selected: getSelection().toString(), status: pre.parentElement.querySelector('.log-raw-status').textContent};
            }""")
            assert before["top"] == 200 and len(before["selected"]) == 80
            page.wait_for_timeout(200)
            _nav(page, "settings")
            _nav(page, "dashboard")
            expect(download).to_be_visible()  # its download built again
            after = raw.evaluate("""pre => ({text: pre.textContent, top: pre.scrollTop, sameNode: pre.firstChild === window.rawText,
                selected: getSelection().toString(), status: pre.parentElement.querySelector('.log-raw-status').textContent})""")
            assert after["text"] == before["text"] and after["sameNode"], "the page on screen is not written again"
            assert abs(after["top"] - before["top"]) <= 2, (before["top"], after["top"])
            assert after["selected"] == before["selected"], "the selection in it stays"
            assert after["status"] == before["status"]
            page.screenshot(path=str(evidence / f"raw-return-{engine}.png"))
            entry.get_by_role("button", name="Next page", exact=True).click()
            expect(status).to_contain_text("65536")
            moved = raw.evaluate("pre => ({text: pre.textContent.slice(0, 40), top: pre.scrollTop})")
            assert moved["top"] == 0 and moved["text"] != before["text"][:40], "a page moved to starts at its top"
            assert not errors, errors
        finally:
            browser.close()
