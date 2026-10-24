"""Card lines, receipts and Reviews keep their nodes while a room is read; a room
opened again, recreated or kept, is at its newest message (DESIGN "History edges",
owner decisions 2026-07-10 and 2026-10-05)."""
import json
import pytest

from ouroboros import dialogue_store
from tests._shared import producer_evidence_dir

from tests.test_history_continuity_browser import (
    _bookmark_place, _deep_gesture_history, _nested_full_output_history, _open_nested_full_output, _OFFSET,
)
from tests.test_chat_history_paging_browser import (
    _append, _assert_at_newest, _open, _open_project, _step, _idle, _screenshot, _FRAMES, _result,
)
from tests.test_chat_history_recovery_browser import _click_project
from tests.test_ui_smoke_playwright import direct_server_with_data as direct_server_with_data

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]


@pytest.mark.parametrize('browser_engine', ['chromium', 'webkit'])
def test_a_deep_place_with_an_open_review_and_line_reopens_at_the_newest_message(direct_server_with_data, browser_engine, tmp_path):
    """Read pages back, with a Review attempt and a long line of its card expanded, a Project closed and
    opened again is at its newest message: the recreated room reads its newest page only, no older page
    again, and claims no approximate place (owner decisions 2026-07-10, 2026-10-05)."""
    from playwright.sync_api import sync_playwright

    root = direct_server_with_data['data_dir']
    project, _ = _deep_gesture_history(root, narration=' Full retained long narration.' * 6)
    with sync_playwright() as pw:
        browser = getattr(pw, browser_engine).launch(headless=True)
        try:
            page = browser.new_page(viewport={'width': 1280, 'height': 850})
            _open(page, direct_server_with_data['url'])
            feed = _open_project(page, project)
            card = page.locator(f'{feed} .chat-live-card[data-task-id="deep-review"]')
            for _ in range(8):
                if card.count():
                    break
                _step(page, feed, automatic=True)
            card.wait_for(state='attached')
            _step(page, feed, automatic=True)  # read on past the card's page while it stays mounted
            if card.get_attribute('data-expanded') != '1':
                card.locator(':scope > [data-live-summary-button] [data-live-disclosure]').click()
            card.locator('[data-review-section-toggle]').click()
            card.locator('[data-review-group-toggle]').click()
            card.locator('[data-review-attempt-toggle]').first.click()
            detail = card.locator('[data-review-attempt-detail]').first
            detail.wait_for(state='visible')
            line = card.locator('.chat-live-line.expandable').last
            line.locator('[data-live-line-toggle]').click()
            assert line.get_attribute('data-expanded') == '1' and len(line.inner_text()) > 150
            _bookmark_place(page, detail)
            _screenshot(page, tmp_path, f'reopen-deep-before-{browser_engine}')
            page.locator('#project-panel-close').click()
            before = page.evaluate('() => window.__historyReads.length')
            _click_project(page, project)
            _idle(page, feed)
            reads = page.evaluate('([n, id]) => window.__historyReads.slice(n).filter(r => r.chatId === id).map(r => r.cursor)',
                                  [before, project['chat_id']])
            assert reads and not any(reads), ('the recreated room reads its newest page only', reads)
            _assert_at_newest(page, feed, 'Gesture message 0899')
            assert card.count() == 0, 'the card read pages back is not read again'
            assert 'could not be restored exactly' not in page.locator(feed).locator('..').inner_text()
            _screenshot(page, tmp_path, f'reopen-deep-after-{browser_engine}')
        finally:
            browser.close()


@pytest.mark.parametrize('browser_engine', ['chromium', 'webkit'])
@pytest.mark.parametrize('theme', ['light', 'dark'])
def test_main_coverage_status_remains_readable_away_from_history_control(direct_server_with_data, browser_engine, theme, tmp_path):
    from playwright.sync_api import sync_playwright

    # Long enough that an owner's bubble (right-aligned since #1998) reaches under the note on the left.
    words = ' '.join(['and the history keeps every message the owner sent across the whole column'] * 3)
    _append(direct_server_with_data['data_dir'], 'chat', [
        {'direction': 'in', 'chat_id': 1, 'ts': '2026-09-01T10:00:00Z',
         'client_message_id': f'main-{index}', 'text': f'Main saved message {index:04d} {words}'} for index in range(1400)])
    with sync_playwright() as pw:
        browser = getattr(pw, browser_engine).launch(headless=True)
        try:
            page = browser.new_page(viewport={'width':1280, 'height':850})
            page.add_init_script(f"localStorage.setItem('ouroboros.theme', '{theme}')")
            _open(page, direct_server_with_data['url'])
            for _ in range(5):
                _step(page, '#chat-messages', automatic=True)
            for width in (1280, 390):
                page.set_viewport_size({'width':width, 'height':850})
                page.locator('#chat-messages').evaluate('n => { n.scrollTop = 0; }')
                page.evaluate(_FRAMES)
                assert not page.locator('#page-chat .chat-page-note').is_visible()
                note = page.locator('#page-chat #chat-messages .chat-load-older .chat-load-older-note')
                assert note.is_visible()
                assert 'Shown messages may have gaps' in note.inner_text()
                box = note.bounding_box()
                assert box['x'] >= 0 and box['x'] + box['width'] <= width + 1
                assert box['y'] >= 0 and box['y'] + box['height'] < 850
                assert page.locator('#page-chat .chat-load-older-note').count() == 1
                _screenshot(page, tmp_path, f'main-history-status-{theme}-{width}-{browser_engine}')
                (tmp_path / f'main-history-status-{theme}-{width}-{browser_engine}.json').write_text(
                    json.dumps({'note': box}, indent=2))
        finally:
            browser.close()


@pytest.mark.parametrize('browser_engine', ['chromium', 'webkit'])
@pytest.mark.parametrize('theme', ['light', 'dark'])
def test_more_menu_stays_above_history_failure(direct_server_with_data, browser_engine, theme, tmp_path):
    """The real history-error chip must neither cover nor intercept More's controls."""
    import io
    from PIL import Image, ImageChops
    from playwright.sync_api import sync_playwright

    with sync_playwright() as pw:
        browser = getattr(pw, browser_engine).launch(headless=True)
        try:
            page = browser.new_page(viewport={'width': 1280, 'height': 850})
            page.add_init_script(f"localStorage.setItem('ouroboros.theme', '{theme}')")
            page.route('**/api/chat/history*', lambda route: route.fulfill(
                status=503, json={'error': 'Synthetic history read failure'}))
            _open(page, direct_server_with_data['url'])
            note = page.locator('#page-chat .chat-page-note .chat-history-status')
            note.wait_for(state='visible')
            assert 'Some saved history could not be loaded' in note.inner_text()
            more = page.locator('#page-chat .chat-header-more')
            summary = more.locator(':scope > summary')
            menu = more.locator('.chat-header-menu')
            for width in (1280, 390):
                page.set_viewport_size({'width': width, 'height': 850})
                page.evaluate(_FRAMES)
                summary.click()
                menu.wait_for(state='visible')
                page.evaluate(_FRAMES)
                _screenshot(page, tmp_path, f'more-failure-{theme}-{width}-{browser_engine}')
                if width == 390:
                    a, b = note.bounding_box(), menu.bounding_box()
                    assert min(a['x'] + a['width'], b['x'] + b['width']) > max(a['x'], b['x'])
                    assert min(a['y'] + a['height'], b['y'] + b['height']) > max(a['y'], b['y'])
                # Inspect every actual control, including the budget row moved into More on phones.
                for control in menu.locator('button, summary, input').all():
                    if control.is_visible():
                        assert control.evaluate('''el => {
                            const r = el.getBoundingClientRect();
                            return [0.2, 0.5, 0.8].every(y => {
                                const hit = document.elementFromPoint(r.left + r.width / 2, r.top + r.height * y);
                                return hit === el || el.contains(hit);
                            });
                        }'''), 'History chip must not intercept a More control'
                # A pointer-transparent overlay can still obscure text. The light menu is opaque;
                # hiding a lower chip must not change its pixels. Dark glass legitimately changes.
                if theme == 'light':
                    painted = Image.open(io.BytesIO(menu.screenshot(animations='disabled'))).convert('RGB')
                    note.evaluate("el => { el.style.visibility = 'hidden'; }")
                    try:
                        clear = Image.open(io.BytesIO(menu.screenshot(animations='disabled'))).convert('RGB')
                    finally:
                        note.evaluate("el => { el.style.removeProperty('visibility'); }")
                    delta = ImageChops.difference(painted, clear).tobytes()
                    changed = sum(sum(delta[i:i + 3]) > 24 for i in range(0, len(delta), 3))
                    assert changed / (len(delta) // 3) <= 0.002, 'History chip must not paint over More'
                first = more.locator('.chat-header-menu-item').first
                first.click(trial=True)  # Do not change consciousness or other runtime controls.
                first.focus()
                page.keyboard.press('Escape')
                assert more.get_attribute('open') is None
                summary.focus()
                summary.press('Enter')
                menu.wait_for(state='visible')
                summary.click()
                assert more.get_attribute('open') is None and note.is_visible()
            assert page.locator('#chat-messages .chat-load-older-btn').inner_text() == 'Retry loading messages'
        finally:
            browser.close()


@pytest.mark.parametrize('browser_engine', ['chromium', 'webkit'])
def test_late_latest_cannot_erase_newer_physical_coverage(direct_server_with_data, browser_engine, tmp_path):
    """A late newest read cannot erase a newer one's coverage. A room reads its newest page one read at
    a time (#1954): a reconnect that comes while that read is unanswered waits for it, so the late answer
    lands first and the newer page then moves the window past it. The reader follows the newest message,
    so the window re-anchors on that page with nothing between two chains to disclose; Load more history
    goes on from the newer page through the rows the late read never covered, only older, to the
    beginning (owner decisions 2026-09-14, 2026-10-05)."""
    from datetime import datetime, timedelta, timezone
    from ouroboros.projects_registry import create_project
    from playwright.sync_api import sync_playwright

    root = direct_server_with_data['data_dir']
    project = create_project(root, 'latest-race', name='Latest response race')
    start = datetime(2026, 9, 1, tzinfo=timezone.utc)

    def append(first, last):
        _append(root, 'chat', [{'direction': 'in', 'chat_id': project['chat_id'],
            'ts': (start + timedelta(minutes=index)).isoformat(),
            'client_message_id': f'race-{index}', 'text': f'Race retained message {index:04d}'} for index in range(first, last)])

    append(0, 150)
    with sync_playwright() as pw:
        browser = getattr(pw, browser_engine).launch(headless=True)
        try:
            page = browser.new_page(viewport={'width':1280, 'height':850})
            # The watched room's newest reads, each with when it was asked and when its answer reached the room;
            # the one armed is held until released. Preference reads are counted: a reconnect reads them first.
            page.add_init_script('''(() => {
                const fetch = window.fetch.bind(window);
                window.__prefs = {asked: 0, answered: 0};
                window.__roomLatest = [];
                window.fetch = async (input, init) => {
                    const url = new URL(typeof input === 'string' ? input : input.url, location.href);
                    if (url.pathname === '/api/ui/preferences') {
                        window.__prefs.asked += 1;
                        try { return await fetch(input, init); } finally { window.__prefs.answered += 1; }
                    }
                    const watched = url.pathname === '/api/chat/history' && !url.searchParams.has('cursor')
                        && Number(url.searchParams.get('chat_id')) === window.__watchRoom;
                    const read = watched ? {asked: performance.now(), answered: null} : null;
                    if (read) window.__roomLatest.push(read);
                    const hold = watched && window.__holdLatest;
                    if (hold) window.__holdLatest = false;
                    try {
                        const response = await fetch(input, init);
                        if (hold) {
                            window.__heldLatestBody = await response.clone().json();
                            await new Promise(resolve => { window.__releaseLatest = resolve; });
                        }
                        return response;
                    } finally { if (read) read.answered = performance.now(); }
                };
            })()''')
            _open(page, direct_server_with_data['url'])
            feed = _open_project(page, project)
            append(150, 151)
            page.evaluate('chatId => { window.__watchRoom = chatId; window.__holdLatest = true; }', project['chat_id'])

            def reconnect():
                page.evaluate('''() => {
                    const socket = window.__testSockets.find(s => s.readyState === WebSocket.OPEN);
                    if (!socket) throw new Error('No open socket to reconnect');
                    socket.close();
                }''')

            reconnect()
            page.wait_for_function('() => Boolean(window.__heldLatestBody && window.__releaseLatest)')
            held = page.evaluate('() => window.__heldLatestBody')
            assert held['messages'][-1]['client_message_id'] == 'race-150'
            append(151, 451)
            sockets, prefs = page.evaluate('() => [window.__testSockets.length, window.__prefs.asked]')
            rooms = page.evaluate("() => document.querySelectorAll('#chat-messages, .chat-messages').length")
            reconnect()
            # Each room reads its preferences on the open, then its newest page (chat.js, the socket's open): once
            # every room's preference read is answered, a room that read anew at once has asked by now.
            page.wait_for_function('''([n, p, rooms]) => window.__testSockets.length > n
                && window.__testSockets.at(-1).readyState === 1 && window.__prefs.asked >= p + rooms
                && window.__prefs.answered === window.__prefs.asked''', arg=[sockets, prefs, rooms])
            page.evaluate(_FRAMES)
            asked_while_held = page.evaluate('() => window.__roomLatest.length')
            page.evaluate('() => window.__releaseLatest()')
            newest = page.locator(f'{feed} [data-client-message-id="race-450"]')
            newest.wait_for(state='attached')
            _idle(page, feed)
            latest_reads = page.evaluate('() => window.__roomLatest')
            reads = page.evaluate('id => window.__historyReads.filter(r => r.chatId === id)', project['chat_id'])
            assert asked_while_held == 1, ('the newer reconnect asked nothing while the late read was unanswered',
                                           latest_reads)
            late, *later = latest_reads
            assert later and all(read['asked'] >= late['answered'] for read in later), \
                ('the newer read was asked only after the late one was answered', latest_reads)
            latest = [read for read in reads if not read['cursor']][-1]
            assert latest['body']['messages'][-1]['client_message_id'] == 'race-450', 'the newer read landed last'
            assert newest.count() == 1
            note = page.locator(feed).locator('..').locator('.chat-load-older-note')
            assert 'Shown messages may have gaps' not in note.inner_text() and 'Beginning' not in note.inner_text()
            _screenshot(page, tmp_path, f'latest-race-newest-{browser_engine}')
            button = page.locator(f'{feed} .chat-load-older button')
            assert not button.evaluate('node => node.hidden'), 'the rows behind the newest read are older history'
            walked = page.evaluate('() => window.__historyReads.length')
            for _ in range(6):  # reading on at the top edge goes only older
                if note.inner_text() == 'Beginning of saved history':
                    break
                _step(page, feed, automatic=True)
            assert note.inner_text() == 'Beginning of saved history'
            assert newest.count() == 1 and button.evaluate('node => node.hidden')
            walk = page.evaluate('([n, id]) => window.__historyReads.slice(n).filter(r => r.chatId === id)',
                                 [walked, project['chat_id']])
            assert walk and walk[0]['cursor'] == latest['body']['next_cursor'], ('reading on starts at the newer page',
                                                                               [read['cursor'] for read in walk])
            assert all(read['cursor'] == prior['body']['next_cursor'] for prior, read in zip(walk, walk[1:])), \
                ('every later read is the next older page', [read['cursor'] for read in walk])
            delivered = {row.get('client_message_id') for read in walk for row in read['body']['messages']}
            assert {'race-151', 'race-300'} <= delivered, 'the rows the late read never covered are read'
            _screenshot(page, tmp_path, f'latest-race-filled-{browser_engine}')
            (tmp_path / f'latest-race-reads-{browser_engine}.json').write_text(json.dumps(
                {'held': held, 'reads': page.evaluate('() => window.__historyReads')}, indent=2))
        finally:
            browser.close()


def _bookmark_reconnect(page):
    page.evaluate("""() => {
        const socket = window.__testSockets.find(s => s.readyState === WebSocket.OPEN);
        if (!socket) throw new Error('No open socket to reconnect');
        socket.close();
    }""")


@pytest.mark.parametrize('browser_engine', ['chromium', 'webkit'])
@pytest.mark.parametrize('line_kind', ['receipt', 'narration'])
def test_a_card_line_arrives_with_its_source_page_and_a_reopen_reads_only_the_newest(
        direct_server_with_data, browser_engine, line_kind, tmp_path):
    """A card's older line comes with the older page of the conversation that holds it, not with
    the card's recent progress; a live receipt of the same revision keeps its node and selection
    when the saved row adopts it. Closed and opened again, the room reads only its newest page and
    is at its newest message (owner decisions 2026-10-05)."""
    from datetime import datetime, timedelta, timezone
    from ouroboros.merge_receipts import card_row_text, review_coverage
    from ouroboros.projects_registry import create_project
    from tests.ui_chat_viewport_smoke import _emit_ws_frame
    from playwright.sync_api import sync_playwright

    root = direct_server_with_data['data_dir']
    project = create_project(root, 'source-page', name='Nested reading position')
    cid = project['chat_id']
    start = datetime(2026, 9, 1, 10, tzinfo=timezone.utc)
    stamp = lambda minute: (start + timedelta(minutes=minute)).isoformat()
    # A declared review with no host-observed record: the receipt honestly warns, never reads green.
    declared = {'reviewed_head_sha': 'b' * 40, 'reviewed_base_sha': 'c' * 40, 'scope': 'full', 'verdict': 'PASS'}
    receipt = {'receipt_id': 'source', 'number': 1347, 'revision': 3,
               'outcome': {'status': 'merged', 'merge_sha': 'a' * 40},
               'review': {'declared': declared, 'record': None, 'declared_only': True},
               'coverage': review_coverage(declared, [], 'b' * 40)}
    marker = 'PR #1347 merge:' if line_kind == 'receipt' else 'SOURCE_PAGE_NARRATION'
    text = card_row_text(receipt) if line_kind == 'receipt' else (
        'SOURCE_PAGE_NARRATION. ' + 'The original expanded activity remains readable. ' * 12)
    old = {'chat_id': cid, 'task_id': 'source-owner', 'ts': stamp(1), 'content': text}
    if line_kind == 'receipt':
        old.update(role='system', system_type='host_progress', narration=False,
                   card_row='reviews', card_row_id='merge-receipt:source', card_row_revision=3)
    _append(root, 'progress', [old, *[
        {'chat_id': cid, 'task_id': 'source-filler', 'ts': stamp(index + 2),
         'content': f'Archive activity {index:03d}'} for index in range(80)],
        {'chat_id': cid, 'task_id': 'source-owner', 'ts': stamp(83), 'content': 'RECENT_OWNER_PROGRESS'}])
    _append(root, 'chat', [
        {'chat_id': cid, 'direction': 'in', 'ts': stamp(index),
         'client_message_id': f'source-{index}', 'text': f'Reading context {index:03d}'} for index in range(300)])
    for task in ('source-owner', 'source-filler'):
        _result(root, task, chat_id=cid, project_id=project['id'], result='Completed retained task.',
                merge_receipts=[receipt] if task == 'source-owner' and line_kind == 'receipt' else [])
    with sync_playwright() as pw:
        browser = getattr(pw, browser_engine).launch(headless=True)
        try:
            page = browser.new_page(viewport={'width': 1280, 'height': 850})
            _open(page, direct_server_with_data['url'])
            feed = _open_project(page, project)
            recent = page.evaluate('id => window.__historyReads.find(r => r.chatId === id).body', cid)
            assert any(row.get('text') == 'RECENT_OWNER_PROGRESS' for row in recent['messages'])
            assert not any(marker in row.get('text', '') for row in recent['messages'])
            card = page.locator(f'{feed} .chat-live-card[data-task-id="source-owner"]')
            if card.get_attribute('data-expanded') != '1':
                card.locator(':scope > [data-live-summary-button] [data-live-disclosure]').click()
            line = card.locator(':scope > [data-live-timeline] > .chat-live-line').filter(has_text=marker)
            if line_kind == 'receipt':
                # A late live receipt already has the same revision as its archive row.
                _emit_ws_frame(page, {**old, 'type': 'chat', 'is_progress': True})
                line.wait_for(state='visible')
                line.evaluate('''n => {
                    window.__sourceLine = n;
                    const range = document.createRange();
                    range.selectNodeContents(n.querySelector('.chat-live-line-title'));
                    getSelection().removeAllRanges(); getSelection().addRange(range);
                    window.__sourceSelection = getSelection().toString();
                }''')
                live_key = line.get_attribute('data-live-line-key')
            _step(page, feed)
            line.wait_for(state='attached')
            supplying = page.evaluate('''({id, marker}) => window.__historyReads.filter(r => r.chatId === id)
                .find(r => r.cursor && r.body?.messages.some(m => m.text?.includes(marker))).body''', {'id': cid, 'marker': marker})
            assert supplying['page_cursor'] != recent['page_cursor']
            source_row = next(row for row in supplying['messages'] if marker in row.get('text', ''))
            source = source_row['history_id']
            if line_kind == 'receipt':
                assert source_row['card_row_revision'] == 3 and source_row['text'] == text
            if line_kind == 'receipt':
                assert line.evaluate('n => n === window.__sourceLine'), 'equal revision adoption keeps the mounted node'
                assert line.get_attribute('data-live-line-key') == live_key
                assert page.evaluate('() => getSelection().toString() === window.__sourceSelection')
                page.evaluate('() => getSelection().removeAllRanges()')
                # A late, lower revision cannot roll back the visible receipt.
                _emit_ws_frame(page, {**old, 'type': 'chat', 'is_progress': True,
                                     'card_row_revision': 2, 'ts': stamp(99), 'content': 'Stale queued receipt'})
                # The retained source above is exact; the known host prefix is display-only.
                assert text.startswith('⚠️ PR #1347 merge:') and 'no_host_observed_review_record' in text
                assert all(part in line.inner_text() for part in text.removeprefix('⚠️ ').splitlines())
            else:
                line.locator('[data-live-line-toggle]').click()
                assert line.get_attribute('data-expanded') == '1'
            _bookmark_place(page, line, 1)
            _screenshot(page, tmp_path, f'source-page-{line_kind}-before-{browser_engine}')
            page.locator('#project-panel-close').click()
            before = page.evaluate('() => window.__historyReads.length')
            _click_project(page, project)
            _idle(page, feed)
            reads = page.evaluate('n => window.__historyReads.slice(n)', before)
            (tmp_path / f'source-page-{line_kind}-{browser_engine}.json').write_text(json.dumps({
                'source': source, 'recent': recent, 'supplying': supplying, 'reopen': reads}, indent=2))
            assert not [read['cursor'] for read in reads if read['cursor']], 'the reopen reads only the newest page'
            _assert_at_newest(page, feed, 'Reading context 299')
            assert 'could not be restored exactly' not in page.locator(feed).locator('..').inner_text()
            _screenshot(page, tmp_path, f'source-page-{line_kind}-after-{browser_engine}')
        finally:
            browser.close()


@pytest.mark.parametrize('browser_engine', ['chromium', 'webkit'])
def test_a_moved_newest_window_leaves_a_reader_in_history_in_place_and_reopen_lands_at_the_newest(
        direct_server_with_data, browser_engine, tmp_path):
    """300 messages arrive while the reader is on the first page: the reconnect's newest read moves past
    their chain, but a reader in older history is not re-anchored, keeps the passage and keeps the rows
    the newest window let go. Closed and opened again, the room reads its newest page only and is at its
    newest message, not at that passage (owner decisions 2026-10-05)."""
    from datetime import datetime, timedelta, timezone
    from ouroboros.projects_registry import create_project
    from playwright.sync_api import sync_playwright

    root = direct_server_with_data['data_dir']
    project = create_project(root, 'retained-rebase', name='Retained reading page')
    start = datetime(2026, 9, 1, tzinfo=timezone.utc)

    def append(first, last):
        _append(root, 'chat', [{'direction': 'in', 'chat_id': project['chat_id'],
            'ts': (start + timedelta(minutes=index)).isoformat(),
            'client_message_id': f'rebase-{index}', 'text': f'Rebase retained message {index:04d}'} for index in range(first, last)])

    append(0, 150)
    with sync_playwright() as pw:
        browser = getattr(pw, browser_engine).launch(headless=True)
        try:
            page = browser.new_page(viewport={'width': 1280, 'height': 850})
            _open(page, direct_server_with_data['url'])
            feed = _open_project(page, project)
            target = page.locator(f'{feed} [data-client-message-id="rebase-50"]')
            offset = _bookmark_place(page, target, 50)
            append(150, 450)
            before = page.evaluate('() => window.__historyReads.length')
            _bookmark_reconnect(page)
            page.wait_for_function('([n, id]) => window.__historyReads.slice(n).some(r => r.chatId === id && r.done)',
                                   arg=[before, project['chat_id']])
            _idle(page, feed)
            page.evaluate(_FRAMES)
            moved = page.evaluate('([n, id]) => window.__historyReads.slice(n).filter(r => r.chatId === id)',
                                  [before, project['chat_id']])
            assert [read['cursor'] for read in moved] == [None], 'one newest read: no re-anchor under a reader in history'
            assert moved[0]['body']['messages'][0]['client_message_id'] == 'rebase-300', 'the newest window moved on'
            assert abs(target.evaluate(_OFFSET) - offset) <= 8
            for kept in ('rebase-0', 'rebase-149', 'rebase-449'):
                assert page.locator(f'{feed} [data-client-message-id="{kept}"]').count() == 1, kept
            _screenshot(page, tmp_path, f'composed-rebase-before-{browser_engine}')
            page.locator('#project-panel-close').click()
            before = page.evaluate('() => window.__historyReads.length')
            _click_project(page, project)
            _idle(page, feed)
            cursors = page.evaluate('n => window.__historyReads.slice(n).filter(r => r.cursor).map(r => r.cursor)', before)
            assert cursors == [], cursors
            _assert_at_newest(page, feed, 'Rebase retained message 0449')
            assert 'could not be restored exactly' not in page.locator(feed).locator('..').inner_text()
            _screenshot(page, tmp_path, f'composed-rebase-after-{browser_engine}')
            (tmp_path / f'composed-rebase-reads-{browser_engine}.json').write_text(json.dumps(page.evaluate('() => window.__historyReads'), indent=2))
        finally:
            browser.close()


@pytest.mark.parametrize('browser_engine', ['chromium', 'webkit'])
@pytest.mark.parametrize('line_kind', ['narration', 'full-result'])
def test_a_live_line_keeps_its_node_through_history_adoption_and_reopens_from_its_saved_row(
        direct_server_with_data, browser_engine, line_kind, tmp_path):
    """A live line the reader expanded keeps its node, key and disclosure when its saved row adopts it on
    a reconnect. Closed and opened again, the room is at its newest message and replays the line from
    its saved source (owner decisions 2026-10-05)."""
    from ouroboros.projects_registry import create_project
    from tests.ui_chat_viewport_smoke import _emit_ws_frame
    from playwright.sync_api import sync_playwright

    root = direct_server_with_data['data_dir']
    project = create_project(root, 'live-line-room', name='Live line adoption')
    cid = project['chat_id']
    full = 'Retained full result. ' * 320 + 'COMPOSED_FULL_RESULT_TAIL'
    lineage = {'subagent_task_id': 'adopt-child', 'parent_task_id': 'adopt-parent',
               'root_task_id': 'adopt-parent', 'delegation_role': 'subagent', 'subagent_role': 'History reader'}
    _append(root, 'chat', [
        {'direction': 'in', 'chat_id': cid, 'ts': f'2026-09-01T10:{index:02d}:00Z',
         'client_message_id': f'adopt-{index}', 'text': f'Live line context {index:02d}'} for index in range(60)])
    _append(root, 'progress', [{'chat_id': cid, 'ts': '2026-09-01T10:30:01Z', 'task_id': 'adopt-parent', 'content': 'Parent narration'}]
            if line_kind == 'full-result' else [])
    row = {'chat_id': cid, 'ts': '2026-09-01T10:30:02Z', 'task_id': 'adopt-child', **lineage,
           'subagent_event': 'completed', 'status': 'completed', 'content': 'Child finished',
           'result': full[:4000], 'result_truncated': True} if line_kind == 'full-result' else {
           'chat_id': cid, 'ts': '2026-09-01T10:30:02Z', 'task_id': 'adopt-parent',
           'content': 'Live narration. ' + 'This is retained expanded activity. ' * 10}
    _result(root, 'adopt-parent', chat_id=cid, project_id=project['id'], result='Parent finished.')
    _result(root, 'adopt-child', chat_id=cid, project_id=project['id'], result=full, **lineage)
    with sync_playwright() as pw:
        browser = getattr(pw, browser_engine).launch(headless=True)
        try:
            page = browser.new_page(viewport={'width': 1280, 'height': 850})
            _open(page, direct_server_with_data['url'])
            feed = _open_project(page, project)
            _emit_ws_frame(page, {**row, 'type': 'chat', 'role': 'assistant', 'is_progress': True})
            parent = page.locator(f'{feed} .chat-live-card[data-task-id="adopt-parent"]')
            if parent.get_attribute('data-expanded') != '1':
                parent.locator(':scope > [data-live-summary-button] [data-live-disclosure]').click()
            owner = page.locator(f'{feed} .chat-live-card[data-task-id="{row["task_id"]}"]')
            if owner.get_attribute('data-expanded') != '1':
                owner.locator(':scope > [data-live-summary-button] [data-live-disclosure]').click()
            line = owner.locator(':scope > [data-live-timeline] > .chat-live-line.expandable').last
            line.locator('[data-live-line-toggle]').click()
            if line_kind == 'full-result':
                line.get_by_text('COMPOSED_FULL_RESULT_TAIL', exact=False).wait_for(state='visible')
            live_key = line.get_attribute('data-live-line-key')
            assert not live_key.startswith(('history-', 'terminal-')), live_key
            line.evaluate('n => { window.__adoptedLine = n; }')
            _bookmark_place(page, line)
            dialogue_store.append(root, 'progress', row)
            before = page.evaluate('() => window.__historyReads.length')
            _bookmark_reconnect(page)
            page.wait_for_function('n => window.__historyReads.length > n && window.__historyReads.slice(n).every(r => r.done)', arg=before)
            _idle(page, feed)
            assert line.get_attribute('data-live-line-key') == live_key
            assert line.evaluate('n => n === window.__adoptedLine'), 'adoption preserves mounted DOM identity'
            assert line.get_attribute('data-expanded') == '1'
            _bookmark_place(page, line, 1)
            anchor = page.evaluate('''async feed => {
                const {createTimelineAnchors} = await import('/static/modules/chat_render_batch.js');
                return createTimelineAnchors({messagesDiv:document.querySelector(feed), liveCardRecords:new Map()}).serializeTimelineAnchor();
            }''', feed)
            assert anchor['lineKey'] == live_key and anchor['lineExpanded'] is True, anchor
            _screenshot(page, tmp_path, f'composed-{line_kind}-before-{browser_engine}')
            page.locator('#project-panel-close').click()
            # A reopen during the exit keeps the same room. Exercise cold replay after it leaves.
            page.locator(feed).wait_for(state='detached')
            _click_project(page, project)
            _idle(page, feed)
            _assert_at_newest(page, feed, 'Live line context 59')
            assert 'could not be restored exactly' not in page.locator(feed).locator('..').inner_text()
            for card in (parent, owner):
                if card.get_attribute('data-expanded') != '1':
                    card.locator(':scope > [data-live-summary-button] [data-live-disclosure]').click()
            assert line.get_attribute('data-live-line-key') != live_key, 'cold replay has its source-owned key'
            line.locator('[data-live-line-toggle]').click()
            assert line.get_attribute('data-expanded') == '1'
            if line_kind == 'full-result':
                line.get_by_text('COMPOSED_FULL_RESULT_TAIL', exact=False).wait_for(state='visible')
                assert full in line.inner_text()
            else:
                assert row['content'].strip() in line.inner_text()
            _screenshot(page, tmp_path, f'composed-{line_kind}-after-{browser_engine}')
        finally:
            browser.close()


@pytest.mark.parametrize('browser_engine', ['chromium', 'webkit'])
def test_a_live_receipt_survives_equal_replay_and_a_reopened_one_takes_only_newer_revisions(
        direct_server_with_data, browser_engine, tmp_path):
    """Live receipts of one revision keep their nodes and keys through a saved replay of the same
    revision. Closed and opened again, the room is at its newest message; its receipts come from
    the saved rows, and live frames update them in place only with a newer revision."""
    from ouroboros.merge_receipts import card_row_text
    from ouroboros.projects_registry import create_project
    from tests.ui_chat_viewport_smoke import _emit_ws_frame
    from playwright.sync_api import sync_playwright

    root = direct_server_with_data['data_dir']
    project = create_project(root, 'receipt-bookmark', name='Receipt reading place')
    cid, task_id = project['chat_id'], 'receipt-owner'
    _append(root, 'chat', [
        {'direction': 'in', 'chat_id': cid, 'ts': f'2026-09-01T10:{index:02d}:00Z',
         'client_message_id': f'receipt-context-{index}', 'text': f'Receipt context {index:02d}'} for index in range(60)])
    _append(root, 'progress', [{'chat_id': cid, 'task_id': task_id, 'ts': '2026-09-01T10:30:00Z',
                                'content': 'Reading the retained pull request.'}])
    receipts = [{'receipt_id': name, 'number': 7, 'revision': 3,
                 'outcome': {'status': 'merged', 'merge_sha': 'c' * 40},
                 'coverage': {'status': 'covers_head', 'gaps': ['Retained receipt detail ' * 14]}}
                for name in ('neighbour', 'reading')]
    _result(root, task_id, chat_id=cid, project_id=project['id'], result='Completed.', merge_receipts=receipts)
    rows = [{'chat_id': cid, 'task_id': task_id, 'ts': '2026-09-01T10:30:01Z',
             'system_type': 'pr_merge_receipt', 'card_row': 'reviews',
             'card_row_id': f"merge-receipt:{receipt['receipt_id']}", 'card_row_revision': 3,
             'narration': False, 'text': card_row_text(receipt)} for receipt in receipts]
    with sync_playwright() as pw:
        browser = getattr(pw, browser_engine).launch(headless=True)
        try:
            page = browser.new_page(viewport={'width': 1280, 'height': 850})
            _open(page, direct_server_with_data['url'])
            feed = _open_project(page, project)
            for row in rows:
                _emit_ws_frame(page, {**row, 'type': 'chat', 'role': 'system', 'is_progress': True, 'content': row['text']})
            card = page.locator(f'{feed} .chat-live-card[data-task-id="{task_id}"]')
            if card.get_attribute('data-expanded') != '1':
                card.locator(':scope > [data-live-summary-button] [data-live-disclosure]').click()
            lines = card.locator(':scope > [data-live-timeline] > .chat-live-line.result')
            assert lines.count() == 2, 'same text and time must not merge different receipts'
            line = lines.nth(1)
            # Receipt rows have no individual disclosure; their owning card does.
            assert line.locator('[data-live-line-toggle]').count() == 0
            live_key = line.get_attribute('data-live-line-key')
            assert not live_key.startswith('history-')
            line.evaluate('n => { window.__readingReceipt = n; }')
            _bookmark_place(page, line, 1)
            _append(root, 'progress', rows)
            before = page.evaluate('() => window.__historyReads.length')
            _bookmark_reconnect(page)
            page.wait_for_function('n => window.__historyReads.length > n && window.__historyReads.slice(n).every(r => r.done)', arg=before)
            _idle(page, feed)
            replay = page.evaluate('id => window.__historyReads.filter(r => r.chatId === id).at(-1).body', cid)
            assert [r['card_row_revision'] for r in replay['messages'] if r.get('card_row_id')] == [3, 3]
            assert line.evaluate('n => n === window.__readingReceipt')
            assert line.get_attribute('data-live-line-key') == live_key
            assert card.get_attribute('data-expanded') == '1'
            _bookmark_place(page, line, 1)
            _screenshot(page, tmp_path, f'receipt-bookmark-before-{browser_engine}')
            page.locator('#project-panel-close').click()
            page.locator(feed).wait_for(state='detached')  # cold replay, after the room's exit
            _click_project(page, project)
            _idle(page, feed)
            _assert_at_newest(page, feed, 'Receipt context 59')
            assert 'could not be restored exactly' not in page.locator(feed).locator('..').inner_text()
            if card.get_attribute('data-expanded') != '1':
                card.locator(':scope > [data-live-summary-button] [data-live-disclosure]').click()
            assert lines.count() == 2
            assert line.get_attribute('data-live-line-key') != live_key
            _screenshot(page, tmp_path, f'receipt-bookmark-after-{browser_engine}')
            evidence = producer_evidence_dir(tmp_path, "history-bookmark", env='HISTORY_UI_EVIDENCE_DIR')
            (evidence / f'receipt-bookmark-{browser_engine}.json').write_text(json.dumps({
                'live_key': live_key, 'cold_key': line.get_attribute('data-live-line-key'),
                'card_expanded': card.get_attribute('data-expanded'),
                'receipt_rows': [r for r in replay['messages'] if r.get('card_row_id')],
            }, indent=2))
            text = line.inner_text()
            line.evaluate('n => { window.__coldReceipt = n; }')
            for revision in (2, 3):
                _emit_ws_frame(page, {**rows[1], 'type': 'chat', 'role': 'system', 'is_progress': True,
                    'card_row_revision': revision, 'ts': '2026-09-02T00:00:00Z', 'text': 'Stale queued receipt',
                    'content': 'Stale queued receipt'})
                assert line.inner_text() == text
            _emit_ws_frame(page, {**rows[1], 'type': 'chat', 'role': 'system', 'is_progress': True,
                'card_row_revision': 4, 'ts': '2026-08-01T00:00:00Z',
                'text': rows[1]['text'] + ' Current review evidence.', 'content': rows[1]['text'] + ' Current review evidence.'})
            assert 'Current review evidence.' in line.inner_text()
            assert line.evaluate('n => n === window.__coldReceipt')
            assert card.get_attribute('data-expanded') == '1'
            assert lines.count() == 2
            _screenshot(page, tmp_path, f'receipt-bookmark-newer-{browser_engine}')
        finally:
            browser.close()


@pytest.mark.parametrize('browser_engine', ['chromium', 'webkit'])
def test_latest_waiting_for_history_yields_to_reading_inside_a_box(direct_server_with_data, browser_engine, tmp_path):
    """#1347 A: ↓ waits for an older read in flight; reading on inside a full output meanwhile keeps the place."""
    from playwright.sync_api import sync_playwright

    project = _nested_full_output_history(direct_server_with_data['data_dir'])
    with sync_playwright() as pw:
        browser = getattr(pw, browser_engine).launch(headless=True)
        try:
            page = browser.new_page(viewport={'width': 1280, 'height': 850})
            feed, _child, line, body = _open_nested_full_output(page, direct_server_with_data['url'], project)
            root = page.locator(feed)
            root.evaluate('feed => { feed.scrollTop = 0; }')
            body.evaluate('n => { n.scrollTop = Math.round((n.scrollHeight - n.clientHeight) / 2); }')
            page.evaluate(_FRAMES)
            # Turning back at the top edge reads an older page; that read is held.
            page.evaluate("() => { window.__historyFault = 'hold'; }")
            root.evaluate("feed => feed.dispatchEvent(new WheelEvent('wheel', {deltaY: -1}))")
            page.wait_for_function('() => Boolean(window.__heldHistory)')
            page.locator('#project-panel-body .chat-scroll-bottom-btn').click()
            rect, frame = body.bounding_box(), root.bounding_box()
            page.mouse.move(rect['x'] + rect['width'] / 2, max(rect['y'], frame['y']) + 40)
            page.wait_for_timeout(300)  # a new wheel burst, not the tail of the edge gesture
            before = {'body': body.evaluate('n => n.scrollTop'), 'line': line.evaluate(_OFFSET)}
            page.mouse.wheel(0, 240)
            page.wait_for_timeout(300)
            page.evaluate(_FRAMES)
            read = {'body': body.evaluate('n => n.scrollTop'), 'line': line.evaluate(_OFFSET)}
            assert read['body'] > before['body'] and abs(read['line'] - before['line']) <= 1, (before, read)
            page.evaluate('() => window.__releaseHistory()')
            _idle(page, feed)
            page.wait_for_timeout(200)
            page.evaluate(_FRAMES)
            after = {'body': body.evaluate('n => n.scrollTop'), 'line': line.evaluate(_OFFSET),
                     'gap': root.evaluate('n => n.scrollHeight - n.scrollTop - n.clientHeight')}
            (tmp_path / f'latest-yields-{browser_engine}.json').write_text(
                json.dumps({'before': before, 'read': read, 'after': after}, indent=2))
            _screenshot(page, tmp_path, f'latest-yields-to-box-{browser_engine}')
            assert after['gap'] > 200, ('the superseded ↓ must not pull the reader to the present', after)
            assert abs(after['line'] - read['line']) <= 8 and abs(after['body'] - read['body']) <= 2, (read, after)
        finally:
            browser.close()


def _retained_room(root, slug, count, **quiz):
    """A deep Project whose composer will hold a staged file, so hiding keeps its instance."""
    from datetime import datetime, timedelta, timezone
    from ouroboros.projects_registry import bind_task_to_project, create_project
    from ouroboros.task_results import write_task_result

    project = create_project(root, slug, name=slug.replace('-', ' ').title())
    start = datetime(2026, 9, 1, tzinfo=timezone.utc)
    _append(root, 'chat', [{'direction': 'in', 'chat_id': project['chat_id'], 'client_message_id': f'{slug}-{index}',
        'ts': (start + timedelta(minutes=index)).isoformat(), 'text': f'{slug} message {index:04d}. Reading an older discussion.'}
        for index in range(count)])
    if quiz:
        bind_task_to_project(root, 'continuity-question', project['id'], project['chat_id'], origin={'absent': 'system'})
        write_task_result(root, 'continuity-question', 'completed', project_id=project['id'], chat_id=project['chat_id'], owner_quiz=quiz)
    return project


def _stage_file(page):
    panel = page.locator('#project-panel')
    panel.locator('.chat-file-input-hidden').set_input_files([{'name': 'kept.txt', 'mimeType': 'text/plain', 'buffer': b'kept'}])
    panel.locator('.attach-name').filter(has_text='kept.txt').wait_for()


_GAP = 'n => n.scrollHeight - n.scrollTop - n.clientHeight'


@pytest.mark.parametrize('browser_engine', ['chromium', 'webkit'])
def test_a_room_kept_for_a_staged_file_reopens_at_the_newest_message_and_follows_replies(direct_server_with_data, browser_engine, tmp_path):
    """Read pages back, a room whose composer holds a staged file is hidden, not destroyed; opened again
    it is at its newest message, reads no older page again, keeps the file, and follows a new reply
    (owner decision 2026-10-05)."""
    from playwright.sync_api import sync_playwright
    from tests.ui_chat_viewport_smoke import _emit_ws_frame

    project = _retained_room(direct_server_with_data['data_dir'], 'kept-room', 1200)
    with sync_playwright() as pw:
        browser = getattr(pw, browser_engine).launch(headless=True)
        try:
            page = browser.new_page(viewport={'width': 1280, 'height': 850})
            _open(page, direct_server_with_data['url'])
            feed = _open_project(page, project)
            for _ in range(3):
                _step(page, feed, automatic=True)
            _bookmark_place(page, page.locator(f'{feed} [data-client-message-id="kept-room-650"]'), 80)
            _stage_file(page)
            page.locator('#project-panel-close').click()
            page.locator('.chat-instance-panel[data-pending-work="1"]').wait_for(state='attached')
            assert page.locator('.chat-instance-panel[data-pending-work="1"]').count() == 1
            before = page.evaluate('() => window.__historyReads.length')
            _click_project(page, project)
            _idle(page, feed)
            cursors = page.evaluate('n => window.__historyReads.slice(n).filter(r => r.cursor).map(r => r.cursor)', before)
            assert cursors == [], 'no older page is read again'
            _assert_at_newest(page, feed, 'kept-room message 1199')
            assert page.locator('#project-panel .attach-name').filter(has_text='kept.txt').count() == 1
            _emit_ws_frame(page, {'type': 'chat', 'chat_id': project['chat_id'], 'role': 'assistant',
                                  'content': 'LIVE_REPLY_AFTER_REOPEN', 'ts': '2026-09-27T22:00:00Z'})
            page.evaluate(_FRAMES)
            _screenshot(page, tmp_path, f'kept-room-reopen-{browser_engine}')
            assert page.locator(feed).get_by_text('LIVE_REPLY_AFTER_REOPEN', exact=True).count() == 1
            assert page.locator(feed).evaluate(_GAP) <= 8, 'the reopened room keeps following the present'
        finally:
            browser.close()


@pytest.mark.parametrize('browser_engine', ['chromium', 'webkit'])
def test_plain_reopen_voids_a_hidden_rooms_held_question_reveal(direct_server_with_data, browser_engine, tmp_path):
    """A reveal still awaiting detail when its retained room hid cannot move or focus a later plain reopen."""
    from playwright.sync_api import sync_playwright
    from tests.test_history_continuity_browser import _FAULT

    project = _retained_room(direct_server_with_data['data_dir'], 'held-reveal', 400, addressed={
        'quiz_id': 'addressed', 'state': 'expired_terminal', 'question': 'ADDRESSED_QUESTION',
        'options': ['Yes', 'No'], 'option_details': ['Proceed', 'Wait'], 'asked_at': '2026-09-01T02:00:00Z'})
    ask = """project => window.dispatchEvent(new CustomEvent('ouro:open-project', {
        detail: {project, task_id: 'continuity-question', quiz_id: 'addressed'}}))"""
    in_quiz = "() => Boolean(document.activeElement?.closest('[data-quiz-id=\"addressed\"]'))"
    with sync_playwright() as pw:
        browser = getattr(pw, browser_engine).launch(headless=True)
        try:
            page = browser.new_page(viewport={'width': 1280, 'height': 850})
            page.add_init_script(f'({_FAULT})()')
            _open(page, direct_server_with_data['url'])
            feed = _open_project(page, project)
            _stage_file(page)
            ids = page.locator(f'{feed} [data-client-message-id]').evaluate_all('ns => ns.map(n => n.dataset.clientMessageId)')
            target = page.locator(f'{feed} [data-client-message-id="{ids[len(ids) // 2]}"]')
            _bookmark_place(page, target, 80)
            page.locator('#project-panel-close').click()
            page.evaluate('() => { window.__holdQuestion = true; }')
            page.evaluate(ask, project)
            page.wait_for_function('() => Boolean(window.__releaseQuestion)')
            page.locator('#project-panel-close').click()
            page.locator('.chat-instance-panel[data-pending-work="1"]').wait_for(state='attached')
            assert page.locator('.chat-instance-panel[data-pending-work="1"]').count() == 1
            _click_project(page, project)
            _idle(page, feed)
            page.evaluate(_FRAMES)
            before = target.evaluate(_OFFSET)
            with page.expect_response(lambda response: '/api/tasks/continuity-question' in response.url):
                page.evaluate('() => { window.__holdQuestion = false; window.__releaseQuestion(); }')
            page.wait_for_timeout(150)
            page.evaluate(_FRAMES)
            _screenshot(page, tmp_path, f'held-reveal-after-plain-reopen-{browser_engine}')
            assert page.locator(f'{feed} [data-quiz-id="addressed"]').count() == 0
            assert abs(target.evaluate(_OFFSET) - before) <= 8 and not page.evaluate(in_quiz)
            page.evaluate(ask, project)  # a new addressed navigation still owns the viewport
            page.locator(f'{feed} [data-quiz-id="addressed"]').wait_for(state='visible')
            page.wait_for_function(in_quiz)
            _screenshot(page, tmp_path, f'held-reveal-new-reveal-{browser_engine}')
        finally:
            browser.close()
