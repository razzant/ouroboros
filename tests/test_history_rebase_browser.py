"""#1347: a read of the present rebases the window without losing readiness or the line being read."""
import json

import pytest

from tests.test_history_bookmark_browser import _GAP, _bookmark_reconnect, _retained_room, _stage_file
from tests.test_chat_history_paging_browser import (
    _append, _assert_at_newest, _open, _open_project, _step, _idle, _screenshot, _FRAMES, _result,
)
from tests.test_chat_history_recovery_browser import _click_project
from tests.test_history_continuity_browser import _FAULT, _bookmark_place
from tests.test_ui_smoke_playwright import direct_server_with_data as direct_server_with_data

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]

# Holds one room's history reads until released; the reader acts meanwhile.
_HOLD_ROOM = """() => {
    const fetch = window.fetch.bind(window);
    window.fetch = async (input, init) => {
        const url = new URL(typeof input === 'string' ? input : input.url, location.href);
        const hold = window.__holdRoom;
        if (hold && url.pathname === '/api/chat/history' && Number(url.searchParams.get('chat_id')) === hold.chatId) await hold.released;
        return fetch(input, init);
    };
}"""
# The chat frame stays in the page: the reading place, not the server's reply, is under test.
_KEEP_CHAT_FRAMES = """() => window.__testSockets.forEach(socket => {
    const send = socket.send.bind(socket);
    socket.send = data => { if (JSON.parse(data).type === 'chat') window.__sentChat = JSON.parse(data); else send(data); };
})"""


@pytest.mark.parametrize('browser_engine', ['chromium', 'webkit'])
def test_unavailable_recent_then_latest_keeps_a_retained_room_at_the_present(direct_server_with_data, browser_engine, tmp_path):
    """A recent read that could not be loaded, then a clean ↓ read: the retained room is at the present,
    and reopened it repaints its newest page once and follows the present."""
    from playwright.sync_api import sync_playwright
    from tests.ui_chat_viewport_smoke import _emit_ws_frame

    root = direct_server_with_data['data_dir']
    project = _retained_room(root, 'recent-latest', 1200)
    cid = project['chat_id']
    since = 'n => window.__historyReads.slice(n[0]).filter(r => r.chatId === n[1])'
    status = """feed => ({button: document.querySelector(`${feed} .chat-load-older button`)?.textContent || '',
        note: document.querySelector(feed).parentElement.innerText})"""
    with sync_playwright() as pw:
        browser = getattr(pw, browser_engine).launch(headless=True)
        try:
            page = browser.new_page(viewport={'width': 1280, 'height': 850})
            page.add_init_script(f'({_FAULT})()')
            _open(page, direct_server_with_data['url'])
            feed = _open_project(page, project)
            for _ in range(3):
                _step(page, feed, automatic=True)
            # The history route never reads an archive file: the recent window the reconnect
            # reads fails as a whole (its transport), where an unreadable archive once left a
            # 200 window with a source missing.
            page.evaluate("id => { window.__continuityFault = {chatId: id, fail: 'recent'}; }", cid)
            before = page.evaluate('() => window.__historyReads.length')
            _bookmark_reconnect(page)
            page.wait_for_function(f'n => ({since})(n).some(r => !r.cursor && r.done)', arg=[before, cid])
            _idle(page, feed)
            recent = page.evaluate(f'n => ({since})(n).find(r => !r.cursor)', [before, cid])
            assert recent.get('error') and not recent.get('body'), recent
            assert page.evaluate(status, feed)['button'] == 'Retry loading messages'
            _screenshot(page, tmp_path, f'unavailable-recent-{browser_engine}')

            page.evaluate('() => { window.__continuityFault = null; }')
            _stage_file(page)
            before = page.evaluate('() => window.__historyReads.length')
            page.locator('#project-panel-body .chat-scroll-bottom-btn').click()
            page.wait_for_function('n => window.__historyReads.length > n', arg=before)
            _idle(page, feed)
            page.evaluate(_FRAMES)
            latest = page.evaluate(since, [before, cid])
            assert [read['cursor'] for read in latest] == [None] and latest[0]['body']['page_cursor']
            assert page.locator(feed).evaluate(_GAP) <= 8
            after = page.evaluate(status, feed)
            assert after['button'] != 'Retry loading messages' and 'could not be loaded' not in after['note'], after
            _screenshot(page, tmp_path, f'unavailable-recent-latest-{browser_engine}')

            page.locator('#project-panel-close').click()
            page.locator('.chat-instance-panel[data-pending-work="1"]').wait_for(state='attached')
            assert page.locator('.chat-instance-panel[data-pending-work="1"]').count() == 1
            before = page.evaluate('() => window.__historyReads.length')
            _click_project(page, project)
            page.evaluate(_FRAMES)
            _emit_ws_frame(page, {'type': 'chat', 'chat_id': cid, 'role': 'assistant',
                                  'content': 'LIVE_REPLY_AFTER_CLEAN_LATEST', 'ts': '2026-09-27T22:00:00Z'})
            page.evaluate(_FRAMES)
            _screenshot(page, tmp_path, f'unavailable-recent-latest-reopen-{browser_engine}')
            # The room's last recent read failed, so the reopen repaints its newest page once
            # (app.js forces the paint the instance has not confirmed), and reads nothing older.
            page.wait_for_function(f'n => ({since})(n).some(r => r.done)', arg=[before, cid])
            reopened = page.evaluate(since, [before, cid])
            assert [(read['cursor'], read.get('status')) for read in reopened] == [(None, 200)], reopened
            assert page.locator(feed).get_by_text('LIVE_REPLY_AFTER_CLEAN_LATEST', exact=True).count() == 1
            assert page.locator(feed).evaluate(_GAP) <= 8, 'the reopened room keeps following the present'
        finally:
            browser.close()


@pytest.mark.parametrize('browser_engine', ['chromium', 'webkit'])
def test_a_visible_receipt_line_keeps_its_node_when_the_newest_window_moves_past_it(
        direct_server_with_data, browser_engine, tmp_path):
    """A receipt line read mid-timeline, its card header and Reviews offscreen: a reconnect's newest window
    no longer holds the card's rows, and the reader in older history is not re-anchored, so the line keeps
    its node, its newest revision and its place (owner decision 2026-10-05)."""
    from ouroboros.merge_receipts import card_row_text
    from ouroboros.projects_registry import create_project
    from tests.ui_chat_viewport_smoke import _emit_ws_frame
    from playwright.sync_api import sync_playwright

    root = direct_server_with_data['data_dir']
    project = create_project(root, 'receipt-rebase', name='Receipt rebase')
    cid, task_id = project['chat_id'], 'receipt-rebase-owner'
    _append(root, 'chat', [
        {'direction': 'in', 'chat_id': cid, 'ts': f'2026-09-01T10:{index:02d}:00Z',
         'client_message_id': f'receipt-rebase-{index}', 'text': f'Receipt rebase context {index:02d}'} for index in range(60)])
    receipts = [{'receipt_id': f'r{index}', 'number': 100 + index, 'revision': 3,
                 'outcome': {'status': 'merged', 'merge_sha': f'{index:x}' * 40},
                 'coverage': {'status': 'covers_head', 'gaps': ['Retained receipt detail ' * 14]}} for index in range(6)]
    rows = [{'chat_id': cid, 'task_id': task_id, 'ts': f'2026-09-01T10:30:{index + 1:02d}Z',
             'system_type': 'pr_merge_receipt', 'card_row': 'reviews',
             'card_row_id': f"merge-receipt:{receipt['receipt_id']}", 'card_row_revision': 3,
             'narration': False, 'text': card_row_text(receipt)} for index, receipt in enumerate(receipts)]
    _append(root, 'progress', [{'chat_id': cid, 'task_id': task_id, 'ts': '2026-09-01T10:30:00Z',
                                'content': 'Reading the retained pull requests.'}, *rows])
    _result(root, task_id, chat_id=cid, project_id=project['id'], result='Completed.', merge_receipts=receipts)
    _result(root, 'receipt-rebase-filler', chat_id=cid, project_id=project['id'], result='Filler completed.')
    live = lambda row, **fields: {**row, 'type': 'chat', 'role': 'system', 'is_progress': True, 'content': row['text'], **fields}
    geometry = """line => {
        const card = line.closest('.chat-live-card'), feed = card.closest('.chat-messages');
        const box = feed.getBoundingClientRect(), rect = node => node.getBoundingClientRect();
        const shown = node => node.getClientRects().length && rect(node).bottom > box.top && rect(node).top < box.bottom;
        return {header: rect(card.querySelector(':scope > [data-live-summary-button]')).bottom - box.top,
            reviews: rect(card.querySelector(':scope > [data-live-reviews-host]')).bottom - box.top,
            line: rect(line).top - box.top, height: box.height,
            others: [...feed.querySelectorAll('[data-history-id]')].filter(n => shown(n) && !n.matches('.chat-live-line.result')).length,
            focused: feed.contains(document.activeElement)};
    }"""
    with sync_playwright() as pw:
        browser = getattr(pw, browser_engine).launch(headless=True)
        try:
            # A feed shorter than the bounded timeline lets it overhang both edges.
            page = browser.new_page(viewport={'width': 1280, 'height': 360})
            _open(page, direct_server_with_data['url'])
            feed = _open_project(page, project)
            card = page.locator(f'{feed} .chat-live-card[data-task-id="{task_id}"]')
            if card.get_attribute('data-expanded') != '1':
                card.locator(':scope > [data-live-summary-button] [data-live-disclosure]').click()
            # Reading moves on from the header: focus there would itself protect the page.
            page.evaluate('() => document.activeElement?.blur()')
            lines = card.locator(':scope > [data-live-timeline] > .chat-live-line.result')
            assert lines.count() == 6
            line = lines.nth(2)
            # The current revision arrives live; the line keeps its archived revision's locator.
            current = rows[2]['text'] + ' Current review evidence.'
            _emit_ws_frame(page, live(rows[2], card_row_revision=4, ts='2026-09-02T00:00:00Z', text=current))
            assert 'Current review evidence.' in line.inner_text()
            line.evaluate('n => { window.__readingReceipt = n; }')
            line.evaluate("""line => {
                const timeline = line.closest('[data-live-timeline]'), feed = line.closest('.chat-messages');
                feed.dispatchEvent(new WheelEvent('wheel', {deltaY: -1}));
                const box = () => feed.getBoundingClientRect();
                feed.scrollTop += timeline.getBoundingClientRect().top - box().top + (timeline.clientHeight - feed.clientHeight) / 2;
                timeline.scrollTop += line.getBoundingClientRect().top - (box().top + box().height / 2) + line.offsetHeight / 2;
            }""")
            page.evaluate(_FRAMES)
            placed = line.evaluate(geometry)
            # Reviews sit under the card's header since #2278: both are above the feed, the line inside it.
            assert placed['header'] < 0 and placed['reviews'] < 0 and 0 <= placed['line'] < placed['height'], placed
            assert placed['others'] == 0 and not placed['focused'], placed
            _screenshot(page, tmp_path, f'receipt-rebase-before-{browser_engine}')

            _append(root, 'progress', [{'chat_id': cid, 'task_id': 'receipt-rebase-filler',
                'ts': f'2026-09-01T11:{index // 60:02d}:{index % 60:02d}Z', 'content': f'Filler activity {index:03d}'}
                for index in range(80)])
            before = page.evaluate('() => window.__historyReads.length')
            _bookmark_reconnect(page)
            # The newest read drops the owner's rows; the reader in older history keeps the chain.
            page.wait_for_function('n => window.__historyReads.slice(n[0]).some(r => r.chatId === n[1] && !r.cursor && r.done)',
                                   arg=[before, cid])
            _idle(page, feed)
            page.evaluate(_FRAMES)
            reads = page.evaluate('n => window.__historyReads.slice(n[0]).filter(r => r.chatId === n[1])', [before, cid])
            (tmp_path / f'receipt-rebase-reads-{browser_engine}.json').write_text(json.dumps(reads, indent=2))
            assert [read['cursor'] for read in reads] == [None], 'one newest read: no re-anchor under a reader in history'
            assert not any(row.get('task_id') == task_id for row in reads[-1]['body']['messages']), 'the source is absent elsewhere'
            assert lines.count() == 6, 'no line read under the viewport may leave with its page'
            assert line.evaluate('n => n === window.__readingReceipt'), 'the visible line keeps its node'
            assert 'Current review evidence.' in line.inner_text()
            assert abs(line.evaluate(geometry)['line'] - placed['line']) <= 8
            _screenshot(page, tmp_path, f'receipt-rebase-after-{browser_engine}')
            # A stale replay neither rolls back nor duplicates the retained receipt;
            # a source-only locator never doubles a physical row stamp.
            _emit_ws_frame(page, live(rows[2], ts='2026-09-02T00:00:01Z'))
            assert 'Current review evidence.' in line.inner_text() and lines.count() == 6
            stamps = page.locator(f'{feed} [data-history-id]').evaluate_all('ns => ns.map(n => n.dataset.historyId)')
            assert len(stamps) == len(set(stamps)), stamps
        finally:
            browser.close()


@pytest.mark.parametrize('browser_engine', ['chromium', 'webkit'])
def test_send_while_a_reopened_room_loads_keeps_a_failed_draft_and_stays_at_the_newest(direct_server_with_data, browser_engine, tmp_path):
    """Read pages back, a room opened again while its newest page is still loading: a failed Send keeps
    the draft and the file, and the room lands at its newest message; an accepted Send's echo is the
    newest row and the late read cannot pull the reader away from it (owner decision 2026-10-05)."""
    from playwright.sync_api import sync_playwright

    project = _retained_room(direct_server_with_data['data_dir'], 'send-place', 1200)
    cid, composer = project['chat_id'], f'#pchat-{project["id"]}-input'
    held = 'id => window.__historyReads.some(r => r.chatId === id && !r.done)'
    with sync_playwright() as pw:
        browser = getattr(pw, browser_engine).launch(headless=True)
        try:
            page = browser.new_page(viewport={'width': 1280, 'height': 850})
            page.add_init_script(f'({_HOLD_ROOM})()')
            _open(page, direct_server_with_data['url'])
            feed = _open_project(page, project)
            target = page.locator(f'{feed} [data-client-message-id="send-place-650"]')

            def reopen_held(read_back):
                if read_back:
                    for _ in range(3):
                        _step(page, feed, automatic=True)
                    _bookmark_place(page, target, 80)
                page.locator('#project-panel-close').click()
                page.locator(feed).wait_for(state='detached')  # a new instance owes the held newest read
                page.evaluate('id => { window.__holdRoom = {chatId: id, released: new Promise(r => { window.__releaseRoom = r; })}; }', cid)
                _click_project(page, project)
                page.wait_for_function(held, arg=cid)

            def release():
                assert page.evaluate(held, cid), 'the room was still loading when Send was pressed'
                page.evaluate('() => { window.__holdRoom = null; window.__releaseRoom(); }')
                _idle(page, feed)
                page.evaluate(_FRAMES)

            reopen_held(read_back=True)
            page.route('**/api/chat/upload', lambda route: route.fulfill(
                status=500, content_type='application/json', body='{"ok": false, "error": "controlled upload failure"}'))
            _stage_file(page)
            page.locator(composer).fill('KEPT_DRAFT')
            page.locator(composer).press('Enter')
            page.locator('.toast').filter(has_text='controlled upload failure').wait_for()
            release()
            _assert_at_newest(page, feed, 'send-place message 1199')
            assert page.locator(composer).input_value() == 'KEPT_DRAFT'
            assert page.locator('#project-panel .attach-name').filter(has_text='kept.txt').count() == 1
            _screenshot(page, tmp_path, f'send-failed-keeps-draft-{browser_engine}')

            page.unroute('**/api/chat/upload')
            page.locator('#project-panel .attach-remove').click()
            page.locator(composer).fill('')
            reopen_held(read_back=False)
            page.evaluate(_KEEP_CHAT_FRAMES)
            page.locator(composer).fill('SENT_WHILE_THE_ROOM_LOADS')
            page.locator(composer).press('Enter')
            page.wait_for_function("() => window.__sentChat?.content === 'SENT_WHILE_THE_ROOM_LOADS'")
            release()
            _screenshot(page, tmp_path, f'send-accepted-present-{browser_engine}')
            echo = page.locator(f'{feed} .chat-bubble.user').filter(has_text='SENT_WHILE_THE_ROOM_LOADS')
            assert echo.count() == 1
            assert page.locator(feed).evaluate(_GAP) <= 8, 'the late read cannot pull the reader away from the echo'
            assert page.locator(composer).input_value() == ''
        finally:
            browser.close()


@pytest.mark.parametrize('browser_engine', ['chromium', 'webkit'])
def test_unavailable_recent_in_a_shallow_room_reopens_following_the_present(direct_server_with_data, browser_engine, tmp_path):
    """A recent read that could not be loaded, ↓ with the present loaded: the retained room reopens live."""
    from playwright.sync_api import sync_playwright
    from tests.ui_chat_viewport_smoke import _emit_ws_frame

    root = direct_server_with_data['data_dir']
    project = _retained_room(root, 'recent-shallow', 1200)
    cid = project['chat_id']
    since = 'n => window.__historyReads.slice(n[0]).filter(r => r.chatId === n[1])'
    with sync_playwright() as pw:
        browser = getattr(pw, browser_engine).launch(headless=True)
        try:
            page = browser.new_page(viewport={'width': 1280, 'height': 850})
            page.add_init_script(f'({_FAULT})()')
            _open(page, direct_server_with_data['url'])
            feed = _open_project(page, project)
            # The reconnect's recent read fails as a whole (see the case above); it stays failed.
            page.evaluate("id => { window.__continuityFault = {chatId: id, fail: 'recent'}; }", cid)
            before = page.evaluate('() => window.__historyReads.length')
            _bookmark_reconnect(page)
            page.wait_for_function(f'n => ({since})(n).some(r => !r.cursor && r.done)', arg=[before, cid])
            _idle(page, feed)
            recent = page.evaluate(f'n => ({since})(n).find(r => !r.cursor)', [before, cid])
            assert recent.get('error') and not recent.get('body'), recent
            assert page.locator(f'{feed} .chat-load-older button').text_content() == 'Retry loading messages'
            _stage_file(page)
            page.locator(feed).evaluate("n => { n.dispatchEvent(new WheelEvent('wheel', {deltaY: -1})); n.scrollTop -= 600; }")
            page.evaluate(_FRAMES)
            before = page.evaluate('() => window.__historyReads.length')
            page.locator('#project-panel-body .chat-scroll-bottom-btn').click()
            page.evaluate(_FRAMES)
            page.evaluate(_FRAMES)
            assert page.evaluate(since, [before, cid]) == [], 'the loaded present needs no read'
            assert page.locator(feed).evaluate(_GAP) <= 8

            page.locator('#project-panel-close').click()
            page.locator('.chat-instance-panel[data-pending-work="1"]').wait_for(state='attached')
            assert page.locator('.chat-instance-panel[data-pending-work="1"]').count() == 1
            _click_project(page, project)
            page.evaluate(_FRAMES)
            _emit_ws_frame(page, {'type': 'chat', 'chat_id': cid, 'role': 'assistant',
                                  'content': 'LIVE_REPLY_AFTER_SHALLOW_REOPEN', 'ts': '2026-09-27T22:00:00Z'})
            page.evaluate(_FRAMES)
            _screenshot(page, tmp_path, f'unavailable-recent-shallow-reopen-{browser_engine}')
            page.wait_for_function(f'n => ({since})(n).some(r => r.done)', arg=[before, cid])
            reopened = page.evaluate(since, [before, cid])
            assert [(read['cursor'], bool(read.get('error'))) for read in reopened] == [(None, True)], \
                ('the reopen repaints the newest page once; the read still fails', reopened)
            assert page.locator(feed).get_by_text('LIVE_REPLY_AFTER_SHALLOW_REOPEN', exact=True).count() == 1
            assert page.locator(feed).evaluate(_GAP) <= 8, 'the reopened room keeps following the present'
            assert page.locator(f'{feed} .chat-load-older button').text_content() == 'Retry loading messages'
        finally:
            browser.close()


@pytest.mark.parametrize('browser_engine', ['chromium', 'webkit'])
def test_first_growth_of_an_empty_source_past_one_window_is_a_disclosed_reachable_gap(direct_server_with_data, browser_engine, tmp_path):
    """A room first read empty grows past the recent quota before the next read: the older half is a gap, not EOF."""
    from ouroboros.projects_registry import create_project
    from playwright.sync_api import sync_playwright

    root = direct_server_with_data['data_dir']
    project = create_project(root, 'empty-growth', name='Empty source growth')
    cid = project['chat_id']
    note = """feed => ({text: document.querySelector(feed).parentElement.querySelector('.chat-load-older-note')?.textContent || '',
        button: document.querySelector(`${feed} .chat-load-older button`)?.hidden === false})"""
    with sync_playwright() as pw:
        browser = getattr(pw, browser_engine).launch(headless=True)
        try:
            page = browser.new_page(viewport={'width': 1280, 'height': 850})
            _open(page, direct_server_with_data['url'])
            feed = _open_project(page, project)
            first = page.evaluate('id => window.__historyReads.find(r => r.chatId === id).body', cid)
            # The chain witness is the store's one constant: its coordinates are durable.
            assert first['coverage']['spans']['chat'] == {'from': 0, 'to': 0, 'chain': 'dialog', 'gaps': []}, first['coverage']
            _append(root, 'chat', [{'direction': 'in', 'chat_id': cid, 'ts': f'2026-09-01T{10 + index // 60:02d}:{index % 60:02d}:00Z',
                                    'client_message_id': f'empty-growth-{index}', 'text': f'Empty growth message {index:03d}'}
                                   for index in range(200)])
            before = page.evaluate('() => window.__historyReads.length')
            _bookmark_reconnect(page)
            page.wait_for_function('n => window.__historyReads.slice(n[0]).some(r => r.chatId === n[1] && !r.cursor && r.done)',
                                   arg=[before, cid])
            _idle(page, feed)
            recent = page.evaluate('n => window.__historyReads.slice(n[0]).find(r => r.chatId === n[1] && !r.cursor).body', [before, cid])
            span = recent['coverage']['spans']['chat']
            assert span['from'] > 0 and span['chain'] == 'dialog', span  # the same chain, rows past the first read's end
            first_row = page.locator(f'{feed} [data-client-message-id="empty-growth-0"]')
            assert first_row.count() == 0 and page.locator(f'{feed} [data-client-message-id="empty-growth-199"]').count() == 1
            status = page.evaluate(note, feed)
            _screenshot(page, tmp_path, f'empty-growth-gap-{browser_engine}')
            assert 'Shown messages may have gaps' in status['text'] and status['button'], status
            for _ in range(4):
                if first_row.count():
                    break
                _step(page, feed)
            _screenshot(page, tmp_path, f'empty-growth-filled-{browser_engine}')
            assert first_row.count() == 1, 'the disclosed gap is reachable'
            assert page.evaluate(note, feed)['text'] == 'Beginning of saved history'
            ids = page.locator(f'{feed} [data-client-message-id^="empty-growth-"]').evaluate_all('ns => ns.map(n => n.dataset.clientMessageId)')
            assert ids == [f'empty-growth-{index}' for index in range(200)]
        finally:
            browser.close()
