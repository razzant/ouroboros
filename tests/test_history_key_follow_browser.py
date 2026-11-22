"""#1347 A: keys the engine animates to the live edge lead to the present, even when a relayout lands while they
still scroll. A key read without a relayout is a recorded feed session (tests/fixtures/feed_sessions/main_keys.json,
reading_back_and_following.json)."""
import json

import pytest

from tests.test_chat_history_paging_browser import _append, _open, _open_project, _screenshot, _FRAMES
from tests.test_history_continuity_browser import _FIRST_VISIBLE, _OFFSET, _bookmark_place
from tests.test_ui_smoke_playwright import direct_server_with_data as direct_server_with_data

pytestmark = [pytest.mark.ui_browser]

_AT_REST = """feed => new Promise((done, fail) => {
    const root = document.querySelector(feed), limit = 30000, deadline = performance.now() + limit, seen = [];
    let last = root.scrollTop, still = 0;
    const tick = () => {
        still = root.scrollTop === last ? still + 1 : 0;
        last = root.scrollTop;
        seen.push(last);
        if (still >= 3) done();
        else if (performance.now() > deadline) fail(new Error(`${feed} never rested for 3 frames in ${limit}ms `
            + `over ${seen.length} frames; last scrollTop ${seen.slice(-8).join(', ')} of `
            + `${root.scrollHeight - root.clientHeight}`));
        else requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
})"""

# While End or PageDown still animates: a late image load (the feed reflows in its next
# frame) and a typing receipt's state snapshot (app.js fans /api/state to every open room,
# which rewrites the feed through its reading anchor). Neither moves the feed's content.
_RELAYOUT = """([feed, cid, delay]) => document.addEventListener('keydown', ({key}) => {
    if (key !== 'End' && key !== 'PageDown') return;
    setTimeout(() => {
        const image = new Image();
        image.style.cssText = 'position:absolute;width:1px;height:1px';
        image.src = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jWQAAAABJRU5ErkJggg==';
        document.querySelector(`${feed} [data-client-message-id="key-30"] .message`).append(image);
        window.__testSockets[0].dispatchEvent(new MessageEvent('message', {data: JSON.stringify(
            {type: 'typing', chat_id: cid, client_message_id: `relayout-${key}-${performance.now()}`})}));
    }, delay);
}, true)"""


def _read_by_keys(direct_server_with_data, browser_engine, tmp_path, relayout_ms=None):
    from playwright.sync_api import sync_playwright

    from ouroboros.projects_registry import create_project
    from tests.ui_chat_viewport_smoke import _emit_ws_frame

    root = direct_server_with_data['data_dir']
    project = create_project(root, 'keyboard-room', name='Keyboard reading')
    cid = project['chat_id']
    _append(root, 'chat', [
        {'direction': 'in', 'chat_id': cid, 'ts': f'2026-09-01T10:{index:02d}:00Z',
         'client_message_id': f'key-{index}', 'text': f'Keyboard context {index:02d}'} for index in range(60)])
    name = browser_engine if relayout_ms is None else f'{browser_engine}-relayout-{relayout_ms}'
    with sync_playwright() as pw:
        browser = getattr(pw, browser_engine).launch(headless=True)
        try:
            page = browser.new_page(viewport={'width': 1280, 'height': 850})
            _open(page, direct_server_with_data['url'])
            feed = _open_project(page, project)
            root_feed = page.locator(feed)
            remaining = 'n => n.scrollHeight - n.scrollTop - n.clientHeight'
            steps = {}
            if relayout_ms is not None:
                page.evaluate(_RELAYOUT, [feed, cid, relayout_ms])

            def press(key):
                before = root_feed.evaluate('n => n.scrollTop')
                page.keyboard.press(key)
                page.wait_for_function('([feed, before]) => document.querySelector(feed).scrollTop !== before',
                                       arg=[feed, before])
                page.evaluate(_AT_REST, feed)
                return root_feed.evaluate(remaining)

            def reply(key):
                first = root_feed.evaluate(_FIRST_VISIBLE)
                _emit_ws_frame(page, {'type': 'chat', 'role': 'user', 'chat_id': cid, 'content': f'A new reply {key}',
                                      'ts': '2026-09-01T11:30:00Z', 'client_message_id': key, 'sender_session_id': 'other-tab'})
                page.locator(f'{feed} [data-client-message-id="{key}"]').wait_for(state='attached')
                page.evaluate(_FRAMES)
                kept = page.locator(f'{feed} [data-history-id="{first["id"]}"]').evaluate(_OFFSET)
                return {'gap': root_feed.evaluate(remaining), 'moved': kept - first['offset']}

            # Click the text being read (focus stays on the document): keys then scroll this feed.
            _bookmark_place(page, page.locator(f'{feed} [data-client-message-id="key-20"]'), 40)
            page.locator(f'{feed} [data-client-message-id="key-20"] .message').click()
            steps['reading'] = reply('key-reading')
            assert steps['reading']['gap'] > 20 and abs(steps['reading']['moved']) <= 2, steps
            steps['end'] = press('End')
            steps['end-reply'] = reply('key-end')
            assert steps['end'] <= 2 and steps['end-reply']['gap'] <= 2, ('End to the bottom follows', steps)
            steps['page-up'] = press('PageUp')
            steps['page-down'] = press('PageDown')
            assert steps['page-up'] > 48 >= steps['page-down'], steps
            steps['page-down-reply'] = reply('key-page-down')
            _screenshot(page, tmp_path, f'keys-follow-{name}')
            assert steps['page-down-reply']['gap'] <= 2, ('an animated PageDown to the bottom follows', steps)
            steps['page-up-again'] = press('PageUp')
            steps['away-reply'] = reply('key-away')
            _screenshot(page, tmp_path, f'keys-away-{name}')
            (tmp_path / f'keys-follow-{name}.json').write_text(json.dumps(steps))
            assert steps['away-reply']['gap'] > 20 and abs(steps['away-reply']['moved']) <= 2, (
                'an upward key stops following', steps)
            assert page.evaluate('() => window.__historyReads.every(read => !read.cursor)'), 'keys at the edge page nothing'
        finally:
            browser.close()


@pytest.mark.parametrize('relayout_ms', [20, 60])
@pytest.mark.parametrize('browser_engine', ['chromium', 'webkit'])
def test_a_relayout_while_a_key_still_scrolls_neither_stops_it_nor_ends_its_follow(
        direct_server_with_data, browser_engine, relayout_ms, tmp_path):
    """A re-layout landing while End or PageDown still animates is no reader gesture: it neither
    writes the feed under the animation (WebKit stops it at any write, even of the same scrollTop)
    nor ends what the key's own scrolls decide, so the key still reaches the live edge and follows."""
    _read_by_keys(direct_server_with_data, browser_engine, tmp_path, relayout_ms)


@pytest.mark.parametrize('browser_engine', ['chromium', 'webkit'])
def test_removing_a_visible_temporary_card_while_end_scrolls_neither_stops_it_nor_ends_its_follow(
        direct_server_with_data, browser_engine, tmp_path):
    """While End animates from the middle of a long history, the first visible temporary card
    receiving its task_summary and leaving the feed (withRemoteActivity -> reading.mutate) must
    not write scrollTop under the running animation or stop it short of the newest messages."""
    from playwright.sync_api import sync_playwright

    from ouroboros.projects_registry import create_project
    from tests.ui_chat_viewport_smoke import _emit_ws_frame

    root = direct_server_with_data['data_dir']
    project = create_project(root, 'keyboard-temp-card-room', name='Keyboard temp card removal')
    cid = project['chat_id']
    _append(root, 'chat', [
        {'direction': 'in', 'chat_id': cid, 'ts': f'2026-09-01T10:{index:02d}:00Z',
         'client_message_id': f'key-{index}',
         'text': f'Keyboard context {index:02d}: ' + ('long line of conversation history to scroll past. ' * 4)}
        for index in range(60)])
    temp_ts = '2026-09-01T10:19:30Z'
    with sync_playwright() as pw:
        browser = getattr(pw, browser_engine).launch(headless=True)
        try:
            page = browser.new_page(viewport={'width': 1280, 'height': 850})
            _open(page, direct_server_with_data['url'])
            feed = _open_project(page, project)
            root_feed = page.locator(feed)
            remaining = 'n => n.scrollHeight - n.scrollTop - n.clientHeight'

            _emit_ws_frame(page, {
                'type': 'chat', 'role': 'assistant', 'chat_id': cid, 'task_id': 'temp-end-1',
                'is_progress': True, 'cancelable': True, 'rounds': 1, 'tool_calls': 1,
                'completion_tool_calls': 1, 'content': '', 'ts': temp_ts,
            })
            temp_card = page.locator(f'{feed} .chat-live-card[data-task-id="temp-end-1"]')
            temp_card.wait_for(state='visible')
            _bookmark_place(page, temp_card, 40)
            assert root_feed.evaluate(remaining) > 1000
            page.locator(f'{feed} [data-client-message-id="key-20"] .message').click()

            # As soon as End starts moving the feed, deliver task_summary so the visible temporary card is removed mid-scroll.
            page.evaluate("""([feed, cid, ts]) => {
                const el = document.querySelector(feed);
                el.addEventListener('scroll', () => {
                    window.__testSockets[0].dispatchEvent(new MessageEvent('message', {
                        data: JSON.stringify({
                            type: 'chat', role: 'system', system_type: 'task_summary',
                            chat_id: cid, task_id: 'temp-end-1', status: 'completed',
                            task_terminal_status: 'completed', content: 'Done', ts,
                        }),
                    }));
                }, { once: true });
            }""", [feed, cid, temp_ts])

            before = root_feed.evaluate('n => n.scrollTop')
            page.keyboard.press('End')
            page.wait_for_function('([feed, before]) => document.querySelector(feed).scrollTop !== before',
                                   arg=[feed, before])
            page.evaluate(_AT_REST, feed)
            page.evaluate(_FRAMES)

            assert temp_card.count() == 0, 'temporary card was removed by task_summary'
            gap = root_feed.evaluate(remaining)
            assert gap <= 2, f'{browser_engine}: End stopped {gap}px above the newest messages'

            _emit_ws_frame(page, {
                'type': 'chat', 'role': 'user', 'chat_id': cid, 'content': 'A new reply after End',
                'ts': '2026-09-01T11:30:00Z', 'client_message_id': 'key-after-temp-end',
                'sender_session_id': 'other-tab',
            })
            page.locator(f'{feed} [data-client-message-id="key-after-temp-end"]').wait_for(state='attached')
            page.evaluate(_FRAMES)
            assert root_feed.evaluate(remaining) <= 2, f'{browser_engine}: End to the bottom keeps following new replies'
        finally:
            browser.close()
