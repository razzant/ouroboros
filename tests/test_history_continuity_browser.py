"""#1347: real retained bytes → gateway → Chat, with only transport faulted."""
import base64
from datetime import datetime, timedelta, timezone
import hashlib
import json
import re

import pytest

from tests.test_chat_history_paging_browser import (
    _assert_at_newest, _open, _open_project, _step, _idle, _write, _result, _screenshot, _FRAMES,
)
from tests.test_chat_history_recovery_browser import _click_project
from tests.test_ui_smoke_playwright import direct_server_with_data as direct_server_with_data

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]

_FAULT = """() => {
    const original = window.fetch.bind(window);
    window.__continuityFault = null;
    window.__continuityScroll = [];
    const scroll = Object.getOwnPropertyDescriptor(Element.prototype, 'scrollTop');
    Object.defineProperty(Element.prototype, 'scrollTop', {...scroll, set(value) {
        if (this.classList?.contains('chat-messages')) {
            window.__continuityScroll.push({mode: window.__continuityFault, id: this.id,
                before: scroll.get.call(this), value, stack: new Error().stack});
            if (window.__continuityScroll.length > 100) window.__continuityScroll.shift();
        }
        scroll.set.call(this, value);
    }});
    window.fetch = async (input, init) => {
        const url = new URL(typeof input === 'string' ? input : input.url, location.href);
        if (window.__holdQuestion && url.pathname === '/api/tasks/continuity-question') {
            await new Promise(resolve => { window.__releaseQuestion = resolve; });
        }
        const fault = window.__continuityFault;
        if (fault && url.pathname === '/api/chat/history' && Number(url.searchParams.get('chat_id')) === fault.chatId) {
            const kind = url.searchParams.has('cursor') ? 'saved' : 'recent';
            if (fault.fail === kind) throw new TypeError('controlled history read failure');
            if (fault.delay && (!fault.delayKind || fault.delayKind === kind))
                await new Promise(resolve => setTimeout(resolve, fault.delay));
        }
        return original(input, init);
    };
}"""


@pytest.mark.parametrize("browser_engine", ["chromium", "webkit"])
def test_deep_reopen_delay_retry_and_live_arrival(direct_server_with_data, browser_engine, tmp_path):
    """Read pages back, a Project closed and opened again is at its newest message whether its read is
    fast, slow or failed and retried, and no older page is read again; a switch away while that read is
    slow paints nothing into the other room. In the room a live answer stays below the passage being
    read, and a narrower window keeps that passage (owner decisions 2026-07-10, 2026-10-05)."""
    from ouroboros.projects_registry import create_project
    from playwright.sync_api import sync_playwright
    from tests.ui_chat_viewport_smoke import _emit_ws_frame

    root = direct_server_with_data["data_dir"]
    project = create_project(root, "continuity-room", name="History continuity")
    other = create_project(root, "other-room", name="Other room")
    start = datetime(2026, 9, 1, tzinfo=timezone.utc)
    _write(root / "logs" / "chat.jsonl", [{
        "direction": "in", "chat_id": project["chat_id"],
        "ts": (start + timedelta(minutes=index)).isoformat(),
        "client_message_id": f"continuity-{index}",
        "text": f"Saved message {index:04d}. Reading the original discussion across archive pages.",
    } for index in range(1200)])
    offset = "node => node.getBoundingClientRect().top - node.closest('.chat-messages').getBoundingClientRect().top"
    with sync_playwright() as pw:
        browser = getattr(pw, browser_engine).launch(headless=True)
        try:
            page = browser.new_page(viewport={"width": 1280, "height": 850})
            page.add_init_script(f"({_FAULT})()")
            _open(page, direct_server_with_data["url"])
            feed = _open_project(page, project)
            target = page.locator(f'{feed} [data-client-message-id="continuity-650"]')

            def read_back():
                for _ in range(3):
                    _step(page, feed, automatic=True)
                target.wait_for(state="attached")
                target.evaluate("""node => {
                    const feed = node.closest('.chat-messages');
                    feed.dispatchEvent(new WheelEvent('wheel', {deltaY: -1}));
                    feed.scrollTop += node.getBoundingClientRect().top - feed.getBoundingClientRect().top - 80;
                }""")
                page.evaluate(_FRAMES)
                return target.evaluate(offset)

            for mode in ("fast", "delayed", "recent", "saved"):
                read_back()
                _screenshot(page, tmp_path, f"continuity-before-{mode}-{browser_engine}")
                page.locator('#project-panel-close').click()
                page.evaluate("fault => { window.__continuityFault = fault; }", {
                    "chatId": project["chat_id"], "delay": 800 if mode == "delayed" else 0,
                    "fail": mode if mode in {"recent", "saved"} else None,
                })
                before = page.evaluate("() => window.__historyReads.length")
                _click_project(page, project)
                if mode == "recent":
                    retry = page.locator(f'{feed} .chat-load-older button')
                    page.wait_for_function("feed => document.querySelector(`${feed} .chat-load-older button`)?.textContent === 'Retry loading messages'", arg=feed)
                    assert 'could not be loaded' in page.locator(feed).locator('..').locator('.chat-load-older-note').inner_text()
                    retry.evaluate("node => node.addEventListener('click', () => { window.__continuityFault = null; }, {once:true, capture:true})")
                    retry.click()
                _idle(page, feed)
                page.evaluate("() => { window.__continuityFault = null; }")
                cursors = page.evaluate("n => window.__historyReads.slice(n).filter(r => r.cursor).map(r => r.cursor)", before)
                if cursors:
                    (tmp_path / f"continuity-failed-{mode}-{browser_engine}.json").write_text(json.dumps(
                        page.evaluate("() => ({scroll:window.__continuityScroll, reads:window.__historyReads})"), indent=2))
                assert cursors == [], (mode, "the reopened room reads only its newest page")
                _assert_at_newest(page, feed, "Saved message 1199")
                _screenshot(page, tmp_path, f"continuity-{mode}-{browser_engine}")

            # Switching away while the reopen read is slow paints nothing into the
            # other room; opened again, the room is at its newest message.
            read_back()
            page.locator('#project-panel-close').click()
            page.evaluate("fault => { window.__continuityFault = fault; }", {
                "chatId": project["chat_id"], "delay": 800})
            _click_project(page, project)
            page.locator(feed).wait_for(state="visible")
            page.locator('#project-panel-close').click()
            other_feed = _open_project(page, other)
            assert not page.locator(other_feed).get_by_text('Saved message', exact=False).count()
            page.locator('#project-panel-close').click()
            _click_project(page, project)
            _idle(page, feed)
            page.evaluate("() => { window.__continuityFault = null; }")
            _assert_at_newest(page, feed, "Saved message 1199")
            _screenshot(page, tmp_path, f"continuity-room-switch-{browser_engine}")

            # A live answer stays below the passage being read, without moving it.
            placed = read_back()
            _emit_ws_frame(page, {"type": "chat", "chat_id": project["chat_id"], "role": "assistant",
                                  "content": "LIVE_REPLY_AT_PRESENT", "ts": "2026-09-27T22:00:00Z"})
            page.evaluate(_FRAMES)
            assert page.locator(feed).get_by_text('LIVE_REPLY_AT_PRESENT', exact=True).count() == 1
            assert abs(target.evaluate(offset) - placed) <= 8
            reads = page.evaluate("() => window.__historyReads.length")
            resize_anchor = page.locator(feed).evaluate("""root => {
                const top = root.getBoundingClientRect().top;
                const node = [...root.querySelectorAll('.chat-bubble[data-history-id]')].find(n => n.getBoundingClientRect().bottom > top);
                return {id: node.dataset.historyId, offset: node.getBoundingClientRect().top - top};
            }""")
            page.set_viewport_size({"width": 760, "height": 850})
            page.evaluate(_FRAMES)
            assert page.evaluate("() => window.__historyReads.length") == reads, "layout starts no archive read"
            narrow = page.locator(f'{feed} [data-history-id="{resize_anchor["id"]}"]').evaluate(offset)
            assert abs(narrow - resize_anchor["offset"]) <= 8, (resize_anchor, narrow)
            _screenshot(page, tmp_path, f"continuity-live-narrow-{browser_engine}")
            (tmp_path / f"continuity-reads-{browser_engine}.json").write_text(json.dumps(
                page.evaluate("() => window.__historyReads"), indent=2), encoding="utf-8")
        finally:
            browser.close()


@pytest.mark.parametrize("browser_engine", ["chromium", "webkit"])
def test_terminal_publication_timing_and_answer_copy(direct_server_with_data, browser_engine, tmp_path, monkeypatch):
    from tests.test_terminal_occurrence import published_terminal_fixture
    from playwright.sync_api import sync_playwright

    root = direct_server_with_data['data_dir']
    direct_server_with_data['stop_server']()
    published_terminal_fixture(root, monkeypatch)
    direct_server_with_data['start_server']()
    with sync_playwright() as pw:
        browser = getattr(pw, browser_engine).launch(headless=True)
        try:
            page = browser.new_page(viewport={'width':1280, 'height':850}, timezone_id='UTC', locale='en-US')
            _open(page, direct_server_with_data['url'])
            def check():
                rows = page.locator('#chat-messages .project-answer')
                assert rows.count() == 3
                assert rows.locator('.message').all_inner_texts() == ['NEW_TASK_ANSWER', 'OLD_TASK_ANSWER', 'LEGACY_TASK_ANSWER']
                notes = rows.locator('.msg-provenance').all_inner_texts()
                assert 'Task ended' in notes[0] and 'Sep 25, 2026' in notes[0]
                assert 'Sep 24, 2026' in notes[1] and 'Notification added Sep 26, 2026' in notes[1]
                assert notes[2].startswith('Task end time not recorded')
                assert 'Notification added Sep 26, 2026' in notes[2]
                assert rows.nth(1).evaluate("node => !node.querySelector('.message').contains(node.querySelector('.msg-provenance'))")
                return notes
            before = check()
            _screenshot(page, tmp_path, f'terminal-times-{browser_engine}')
            page.reload(wait_until='domcontentloaded')
            _idle(page, '#chat-messages')
            assert check() == before
            page.set_viewport_size({'width':390, 'height':844})
            if page.locator('#nav-drawer-backdrop').is_visible():
                page.locator('#nav-drawer-backdrop').click(position={'x':380, 'y':400})
            page.evaluate(_FRAMES)
            assert check() == before
            _screenshot(page, tmp_path, f'terminal-times-narrow-{browser_engine}')
        finally:
            browser.close()


@pytest.mark.parametrize("browser_engine", ["chromium", "webkit"])
def test_project_room_terminal_lines_separate_end_and_notification(direct_server_with_data, browser_engine, tmp_path, monkeypatch):
    """Inside the Project room the task's own saved end row keeps both times (#1347 A)."""
    from playwright.sync_api import sync_playwright

    from ouroboros.project_dialogue import append_terminal_task_projection
    from ouroboros.projects_registry import bind_task_to_project, create_project
    from ouroboros.task_results import write_task_result

    root = direct_server_with_data['data_dir']
    direct_server_with_data['stop_server']()
    project = create_project(root, 'room-terminal-times', name='Room terminal times')
    ended = {'late': '2026-09-24T18:18:27+00:00', 'prompt': '2026-09-25T15:19:01+00:00', 'unknown': None}
    added = {'late': '2026-09-26T19:21:43+00:00', 'prompt': '2026-09-25T15:19:09+00:00', 'unknown': '2026-09-26T19:25:00+00:00'}
    for task_id in ('prompt', 'late', 'unknown'):
        bind_task_to_project(root, task_id, project['id'], project['chat_id'], origin={'absent': 'system'})
        write_task_result(root, task_id, 'running', task_attempt=0, started_at='2026-09-24T17:00:00+00:00',
                          project_id=project['id'], chat_id=project['chat_id'])
        monkeypatch.setattr('ouroboros.task_results.utc_now_iso', lambda: ended[task_id] or added[task_id])
        result = write_task_result(root, task_id, 'failed', _terminal_observed=bool(ended[task_id]),
                                   result=f'{task_id.upper()}_ROOM_RESULT')
        monkeypatch.setattr('ouroboros.terminal_projection.utc_now_iso', lambda: added[task_id])
        assert append_terminal_task_projection(root, task_id, {'id': task_id}, result, {})
    direct_server_with_data['start_server']()
    with sync_playwright() as pw:
        browser = getattr(pw, browser_engine).launch(headless=True)
        try:
            page = browser.new_page(viewport={'width': 1280, 'height': 850}, timezone_id='UTC', locale='en-US')
            _open(page, direct_server_with_data['url'])
            feed = _open_project(page, project)

            def notes():
                found = {}
                for task_id in ended:
                    card = page.locator(f'{feed} .chat-live-card[data-task-id="{task_id}"]')
                    card.wait_for(state='attached')
                    if card.get_attribute('data-expanded') != '1':
                        card.locator(':scope > [data-live-summary-button]').click()
                    found[task_id] = card.locator(':scope > [data-live-timeline]').inner_text()
                return found

            found = notes()
            # ICU spells the date/time joint differently per engine; the facts are fixed.
            assert re.search(r'Task ended Sep 24, 2026\D+06:18 PM · Notification added Sep 26, 2026\D+07:21 PM', found['late']), found
            assert re.search(r'Task end time not recorded · Notification added Sep 26, 2026\D+07:25 PM', found['unknown']), found
            assert 'Notification added' not in found['prompt'] and 'Task end' not in found['prompt'], found
            _screenshot(page, tmp_path, f'room-terminal-times-{browser_engine}')
            page.reload(wait_until='domcontentloaded')
            _idle(page, '#chat-messages')
            feed = _open_project(page, project)
            assert notes() == found
            page.set_viewport_size({'width': 390, 'height': 844})
            page.evaluate(_FRAMES)
            late = page.locator(f'{feed} .chat-live-card[data-task-id="late"]')
            late.scroll_into_view_if_needed()
            box = late.bounding_box()
            assert box['x'] >= 0 and box['x'] + box['width'] <= 391, box
            _screenshot(page, tmp_path, f'room-terminal-times-narrow-{browser_engine}')
        finally:
            browser.close()


@pytest.mark.parametrize("browser_engine", ["chromium", "webkit"])
def test_latest_supersedes_a_held_question_reveal_and_a_later_reveal_lands(direct_server_with_data, browser_engine, tmp_path):
    """A question opened from Main whose detail is still held yields to the reader's ↓; a later
    reveal lands on the question, focuses it and keeps it through a reflow (DESIGN "History edges":
    only an explicit navigation lands anywhere but the newest message)."""
    from ouroboros.projects_registry import create_project, bind_task_to_project
    from ouroboros.task_results import write_task_result
    from playwright.sync_api import sync_playwright

    root = direct_server_with_data['data_dir']
    project = create_project(root, 'navigation-room', name='Addressed history')
    _write(root / 'logs/chat.jsonl', [{
        'direction': 'in', 'chat_id': project['chat_id'],
        'ts': '2026-09-03T10:00:00Z' if index >= 1180 else '2026-09-01T10:00:00Z',
        'client_message_id': f'navigation-{index}', 'text': f'NAVIGATION_MESSAGE_{index:04d}',
    } for index in range(1200)])
    bind_task_to_project(root, 'continuity-question', project['id'], project['chat_id'], origin={'absent':'system'})
    write_task_result(root, 'continuity-question', 'completed', project_id=project['id'],
        chat_id=project['chat_id'], owner_quiz={'addressed': {
            'quiz_id':'addressed', 'state':'expired_terminal', 'question':'ADDRESSED_QUESTION',
            'options':['Yes', 'No'], 'option_details':['Proceed', 'Wait'], 'asked_at':'2026-09-02T10:00:00Z',
        }})
    with sync_playwright() as pw:
        browser = getattr(pw, browser_engine).launch(headless=True)
        try:
            page = browser.new_page(viewport={'width':1280, 'height':850})
            page.add_init_script(f'({_FAULT})()')
            _open(page, direct_server_with_data['url'])
            feed = _open_project(page, project)
            target = page.locator(f'{feed} [data-client-message-id="navigation-650"]')
            button = page.locator('#project-panel-body .chat-scroll-bottom-btn')

            # Begin a targeted navigation through the application's public event.
            # The detail response is held; a newer latest click must win even
            # when the real canonical question later arrives.
            for _ in range(3):
                _step(page, feed, automatic=True)
            target.wait_for(state='attached')
            target.evaluate("""node => {
                const feed = node.closest('.chat-messages');
                feed.dispatchEvent(new WheelEvent('wheel', {deltaY:-1}));
                feed.scrollTop += node.getBoundingClientRect().top - feed.getBoundingClientRect().top - 80;
            }""")
            page.evaluate(_FRAMES)
            page.locator('#project-panel-close').click()
            page.evaluate("""project => {
                window.__holdQuestion = true;
                window.dispatchEvent(new CustomEvent('ouro:open-project', {
                    detail:{project, task_id:'continuity-question', quiz_id:'addressed'}}));
            }""", project)
            page.wait_for_function('() => Boolean(window.__releaseQuestion)')
            _idle(page, feed)
            assert target.count() == 0, 'the passage read before closing is not restored'
            # Scroll up by a user gesture so the visible button can be activated.
            page.locator(feed).evaluate("n => { n.dispatchEvent(new WheelEvent('wheel', {deltaY:-1})); n.scrollTop -= 350; }")
            page.evaluate(_FRAMES)
            button.click()
            page.evaluate('() => { window.__holdQuestion = false; window.__releaseQuestion(); }')
            _idle(page, feed)
            page.wait_for_timeout(100)
            page.evaluate(_FRAMES)
            assert page.locator(f'{feed} [data-quiz-id="addressed"]').count() == 0
            assert page.locator(feed).evaluate('n => n.scrollHeight - n.scrollTop - n.clientHeight') <= 8

            # A subsequent addressed read lands and remains the reading target
            # after reflow, even though the preceding intent followed latest.
            page.evaluate("""project => window.dispatchEvent(new CustomEvent('ouro:open-project', {
                detail:{project, task_id:'continuity-question', quiz_id:'addressed'}}))""", project)
            quiz = page.locator(f'{feed} [data-quiz-id="addressed"]')
            quiz.wait_for(state='visible')
            _idle(page, feed)
            page.wait_for_function("selector => document.querySelector(selector)?.contains(document.activeElement)",
                                   arg=f'{feed} [data-quiz-id="addressed"]')
            anchor = page.locator(feed).evaluate("""root => {
                const top = root.getBoundingClientRect().top;
                const node = [...root.querySelectorAll('.chat-bubble[data-history-id]')].find(n => n.getBoundingClientRect().bottom > top);
                return {id:node.dataset.historyId, offset:node.getBoundingClientRect().top - top};
            }""")
            page.set_viewport_size({'width':900, 'height':850})
            page.evaluate(_FRAMES)
            page.wait_for_timeout(120)  # allow ResizeObserver → RAF → header-reserve layout to settle
            after = page.locator(f'{feed} [data-history-id="{anchor["id"]}"]').evaluate(
                'n => n.getBoundingClientRect().top - n.closest(".chat-messages").getBoundingClientRect().top')
            if abs(after - anchor['offset']) > 8:
                (tmp_path / f'question-reflow-{browser_engine}.json').write_text(json.dumps(page.locator(feed).evaluate("""root => ({
                    scrollTop:root.scrollTop, rect:root.getBoundingClientRect().toJSON(),
                    visible:[...root.children].filter(n => n.getBoundingClientRect().bottom > root.getBoundingClientRect().top
                        && n.getBoundingClientRect().top < root.getBoundingClientRect().bottom).map(n => ({
                        class:n.className, id:n.dataset.historyId, task:n.dataset.taskId,
                        top:n.getBoundingClientRect().top, bottom:n.getBoundingClientRect().bottom})),
                    scroll:window.__continuityScroll,
                })"""), indent=2))
                _screenshot(page, tmp_path, f'question-reflow-failed-{browser_engine}')
            assert abs(after - anchor['offset']) <= 8
            assert quiz.evaluate('n => n.getBoundingClientRect().bottom > n.closest(".chat-messages").getBoundingClientRect().top')
            _screenshot(page, tmp_path, f'question-after-load-{browser_engine}')
        finally:
            browser.close()


_OFFSET = "n => n.getBoundingClientRect().top - n.closest('.chat-messages').getBoundingClientRect().top"
_FIRST_VISIBLE = """root => {
    const top = root.getBoundingClientRect().top;
    const node = [...root.querySelectorAll('.chat-bubble[data-history-id]')].find(n => n.getBoundingClientRect().bottom > top + 1);
    return node && {id:node.dataset.historyId, offset:node.getBoundingClientRect().top - top, scrollTop:root.scrollTop};
}"""


def _deep_gesture_history(root):
    """A Review card, a photo and the reading target live on the third saved page."""
    from ouroboros.artifacts import store_task_artifact_bytes
    from ouroboros.projects_registry import create_project
    from tests.test_chat_history_paging_browser import _result

    project = create_project(root, 'gesture-room', name='Gesture continuity')
    cid, start = project['chat_id'], datetime(2026, 9, 1, tzinfo=timezone.utc)
    at = lambda index, seconds=0: (start + timedelta(minutes=index, seconds=seconds)).isoformat()
    image = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jWQAAAABJRU5ErkJggg==')
    image_name = f'chat-media-{hashlib.sha256(image).hexdigest()}.png'
    store_task_artifact_bytes(root, 'gesture-media', image_name, image)
    rows = [{'direction': 'in', 'chat_id': cid, 'ts': at(index), 'client_message_id': f'gesture-{index}',
             'text': f'Gesture message {index:04d}. Reading the original discussion across archive pages.'}
            for index in range(900)]
    rows[425] = {'direction': 'out', 'chat_id': cid, 'ts': at(425), 'type': 'photo', 'task_id': 'gesture-media',
                 'text': 'Held photo', 'caption': 'Held photo', 'mime': 'image/png',
                 'download_url': f'/api/tasks/gesture-media/artifacts/{image_name}'}
    _write(root / 'logs' / 'chat.jsonl', rows)
    review = {'panels': [{'panel_id': 'gesture-panel', 'surface': 'task_acceptance', 'aggregate_signal': 'PASS',
                          'transport_status': 'success', 'parse_status': 'valid',
                          'reason': 'DEEP_REVIEW_DETAIL ' + 'Reviewed evidence stays readable. ' * 12, 'actors': []}]}
    _write(root / 'logs' / 'progress.jsonl', [
        *[{'ts': at(420, index), 'chat_id': cid, 'task_id': 'deep-review',
           'content': f'Deep review narration {index}. ' + 'Inspecting retained history. ' * 6} for index in range(3)],
        *[{'ts': at(700 + index // 2, index % 2), 'chat_id': cid, 'task_id': 'later-work',
           'content': f'Later work {index:03d}'} for index in range(200)],
    ])
    for task_id, extra in [('deep-review', {'review_projection': review, 'suggested_name': 'Deep review card'}),
                           ('later-work', {}), ('gesture-media', {})]:
        _result(root, task_id, chat_id=cid, project_id=project['id'], **extra)
    return project, image_name


def _bookmark_place(page, node, offset=30):
    node.evaluate("""(node, offset) => {
        const feed = node.closest('.chat-messages');
        feed.dispatchEvent(new WheelEvent('wheel', {deltaY:-1}));
        feed.scrollTop += node.getBoundingClientRect().top - feed.getBoundingClientRect().top - offset;
    }""", offset)
    page.evaluate(_FRAMES)
    return node.evaluate(_OFFSET)


def _nested_full_output_history(root, card_after=49):
    """A Project card whose child line holds a long fetched full output, between saved talk."""
    from ouroboros.projects_registry import create_project

    project = create_project(root, 'nested-scroll-room', name='Nested scroll history')
    cid = project['chat_id']
    full = ''.join(f'Retained paragraph {index:03d}. ' + 'The full output stays readable in place. ' * 3 + '\n\n'
                   for index in range(120)) + 'UNIQUE_NESTED_TAIL'
    lineage = {'subagent_task_id': 'nested-child', 'parent_task_id': 'nested-parent',
               'root_task_id': 'nested-parent', 'delegation_role': 'subagent', 'subagent_role': 'Archive reader'}
    # 150 recent human rows leave 50 on an older page; by default the card
    # opens the recent window, right below its Load-older boundary.
    def minute(index):
        return f'2026-09-01T{10 + index // 60:02d}:{index % 60:02d}'
    at = minute(card_after)
    _write(root / 'logs/chat.jsonl', [
        {'direction': 'in', 'chat_id': cid, 'ts': f'{minute(index)}:00Z',
         'client_message_id': f'nested-{index}', 'text': f'Nested context {index:03d}'} for index in range(200)])
    _write(root / 'logs/progress.jsonl', [
        {'chat_id': cid, 'ts': f'{at}:30Z', 'task_id': 'nested-parent', 'content': 'Parent narration'},
        {'chat_id': cid, 'ts': f'{at}:31Z', 'task_id': 'nested-child', **lineage,
         'subagent_event': 'completed', 'status': 'completed', 'content': 'Child finished',
         'result': full[:4000], 'result_truncated': True}])
    _result(root, 'nested-parent', chat_id=cid, project_id=project['id'], result='Parent finished.')
    _result(root, 'nested-child', chat_id=cid, project_id=project['id'], result=full, **lineage)
    return project


def _open_nested_full_output(page, url, project):
    _open(page, url)
    feed = _open_project(page, project)
    parent = page.locator(f'{feed} .chat-live-card[data-task-id="nested-parent"]')
    child = page.locator(f'{feed} .chat-live-card[data-task-id="nested-child"]')
    for card in (parent, child):
        card.wait_for(state='attached')
        if card.get_attribute('data-expanded') != '1':
            card.locator(':scope > [data-live-summary-button]').click()
    line = child.locator(':scope > [data-live-timeline] > .chat-live-line.expandable')
    line.locator('[data-live-line-toggle]').click()
    body = line.locator(':scope > .chat-live-line-body-full')
    body.get_by_text('UNIQUE_NESTED_TAIL', exact=False).wait_for(state='attached')
    page.evaluate(_FRAMES)
    return feed, child, line, body


@pytest.mark.parametrize('browser_engine', ['chromium', 'webkit'])
def test_wheel_inside_full_output_neither_pages_nor_follows_until_its_edge(direct_server_with_data, browser_engine, tmp_path):
    """#1347 A: a real wheel scrolls the bounded output it is over; only at its edge does it move the feed."""
    from playwright.sync_api import sync_playwright

    project = _nested_full_output_history(direct_server_with_data['data_dir'])
    with sync_playwright() as pw:
        browser = getattr(pw, browser_engine).launch(headless=True)
        try:
            page = browser.new_page(viewport={'width': 1280, 'height': 850})
            feed, _child, _line, body = _open_nested_full_output(page, direct_server_with_data['url'], project)
            root = page.locator(feed)
            # Clicking the disclosures scrolled the card into view; settle at the
            # feed top without a gesture, so no archive read has been asked for.
            root.evaluate('feed => { feed.scrollTop = 0; }')
            page.evaluate(_FRAMES)
            body.evaluate('n => { n.scrollTop = Math.round((n.scrollHeight - n.clientHeight) / 2); }')
            page.evaluate(_FRAMES)
            rect = body.bounding_box()
            frame = root.bounding_box()
            assert rect['y'] + 40 < frame['y'] + frame['height'], (rect, frame)
            assert root.evaluate('n => n.scrollTop') < 80
            before = {'reads': page.evaluate('() => window.__historyReads.length'),
                      'feed': root.evaluate('n => n.scrollTop'), 'body': body.evaluate('n => n.scrollTop')}
            page.mouse.move(rect['x'] + rect['width'] / 2, max(rect['y'], frame['y']) + 40)
            page.mouse.wheel(0, -240)
            page.wait_for_timeout(400)
            page.evaluate(_FRAMES)
            after = {'reads': page.evaluate('() => window.__historyReads.length'),
                     'feed': root.evaluate('n => n.scrollTop'), 'body': body.evaluate('n => n.scrollTop')}
            (tmp_path / f'nested-wheel-{browser_engine}.json').write_text(json.dumps({'before': before, 'after': after}, indent=2))
            _screenshot(page, tmp_path, f'nested-wheel-{browser_engine}')
            assert after['body'] < before['body'], (before, after)
            assert abs(after['feed'] - before['feed']) <= 1, (before, after)
            assert after['reads'] == before['reads'], 'a wheel the full output consumed must not page the archive'

            # At the output's top edge the wheel chains to its card timeline, the
            # next bounded box that can still move: it scrolls; the feed still does not.
            timeline = body.locator('xpath=ancestor::*[@data-live-timeline][1]')
            timeline.evaluate('n => { n.scrollTop = n.scrollHeight; }')
            body.evaluate('n => { n.scrollTop = 0; }')
            page.evaluate(_FRAMES)
            chained = {'timeline': timeline.evaluate('n => n.scrollTop'), 'feed': root.evaluate('n => n.scrollTop')}
            assert chained['timeline'] > 0, chained
            inner = body.bounding_box()
            top, bottom = max(inner['y'], frame['y']), min(inner['y'] + inner['height'], frame['y'] + frame['height'])
            assert bottom - top > 20, (inner, frame)
            page.mouse.move(inner['x'] + inner['width'] / 2, (top + bottom) / 2)
            page.mouse.wheel(0, -240)
            page.wait_for_timeout(400)
            page.evaluate(_FRAMES)
            assert timeline.evaluate('n => n.scrollTop') < chained['timeline']
            assert abs(root.evaluate('n => n.scrollTop') - chained['feed']) <= 1
            assert page.evaluate('() => window.__historyReads.length') == before['reads'], 'the timeline absorbed it'

            # With the output and its card timeline at their top edges, the same
            # real wheel reaches the feed top.
            body.evaluate('n => { n.scrollTop = 0; n.closest("[data-live-timeline]").scrollTop = 0; }')
            page.evaluate(_FRAMES)
            page.mouse.wheel(0, -240)
            page.wait_for_function('n => window.__historyReads.length > n', arg=before['reads'], timeout=30_000)
            _idle(page, feed)
            assert page.locator(f'{feed} [data-client-message-id="nested-10"]').count() == 1
            _screenshot(page, tmp_path, f'nested-wheel-edge-{browser_engine}')
        finally:
            browser.close()


@pytest.mark.parametrize('browser_engine', ['chromium', 'webkit'])
def test_feed_scrollbar_drag_to_the_live_edge_follows_new_replies(direct_server_with_data, browser_engine, tmp_path):
    """#1347 A: dragging the feed's own scrollbar to the bottom leads to the present;
    a drag released away from it stays there even when a resize later reaches the end."""
    from playwright.sync_api import sync_playwright

    from ouroboros.projects_registry import create_project
    from tests.ui_chat_viewport_smoke import _emit_ws_frame

    root = direct_server_with_data['data_dir']
    project = create_project(root, 'scrollbar-room', name='Scrollbar drag')
    cid = project['chat_id']
    _write(root / 'logs/chat.jsonl', [
        {'direction': 'in', 'chat_id': cid, 'ts': f'2026-09-01T10:{index:02d}:00Z',
         'client_message_id': f'drag-{index}', 'text': f'Scrollbar context {index:02d}'} for index in range(60)])
    with sync_playwright() as pw:
        # Classic scrollbars: headless Chromium hides them by default.
        options = {'ignore_default_args': ['--hide-scrollbars']} if browser_engine == 'chromium' else {}
        browser = getattr(pw, browser_engine).launch(headless=True, **options)
        try:
            page = browser.new_page(viewport={'width': 1280, 'height': 850})
            _open(page, direct_server_with_data['url'])
            feed = _open_project(page, project)
            root_feed = page.locator(feed)
            bar = root_feed.evaluate('n => n.offsetWidth - n.clientWidth')
            assert bar > 0, 'a classic feed scrollbar is present'
            _bookmark_place(page, page.locator(f'{feed} [data-client-message-id="drag-10"]'), 40)
            box = root_feed.bounding_box()
            x = box['x'] + box['width'] - bar / 2
            thumb_js = """n => n.getBoundingClientRect().top
                + (n.scrollTop + n.clientHeight / 2) / n.scrollHeight * n.clientHeight"""

            def drag(to_y):
                page.mouse.move(x, root_feed.evaluate(thumb_js))
                page.mouse.down()
                page.mouse.move(x, to_y, steps=12)
                page.mouse.up()
                page.evaluate(_FRAMES)

            def reply(key):
                _emit_ws_frame(page, {'type': 'chat', 'role': 'user', 'chat_id': cid, 'content': f'A new reply {key}',
                                      'ts': '2026-09-01T11:30:00Z', 'client_message_id': key, 'sender_session_id': 'other-tab'})
                page.locator(f'{feed} [data-client-message-id="{key}"]').wait_for(state='attached')
                page.evaluate(_FRAMES)

            drag(box['y'] + box['height'] - 2)
            dragged = root_feed.evaluate('n => n.scrollHeight - n.scrollTop - n.clientHeight')
            assert dragged <= 2, dragged
            reply('drag-new')
            gap = root_feed.evaluate('n => n.scrollHeight - n.scrollTop - n.clientHeight')
            _screenshot(page, tmp_path, f'scrollbar-drag-follows-{browser_engine}')
            assert gap <= 2, f'the dragged-to-bottom feed follows the new reply ({gap}px short)'

            # Released a little above the end; a taller window then clamps the feed
            # to its end. Only the release decided following: a new reply stays below.
            ratio = root_feed.evaluate('n => n.scrollHeight / n.clientHeight')
            drag(root_feed.evaluate(thumb_js) - 90 / ratio)
            released = root_feed.evaluate('n => n.scrollHeight - n.scrollTop - n.clientHeight')
            assert 48 < released < 150, released
            page.set_viewport_size({'width': 1280, 'height': 1000})
            page.wait_for_timeout(150)
            page.evaluate(_FRAMES)
            resized = root_feed.evaluate('n => n.scrollHeight - n.scrollTop - n.clientHeight')
            first = root_feed.evaluate(_FIRST_VISIBLE)
            reply('drag-away')
            kept = page.locator(f'{feed} [data-history-id="{first["id"]}"]').evaluate(_OFFSET)
            away = root_feed.evaluate('n => n.scrollHeight - n.scrollTop - n.clientHeight')
            _screenshot(page, tmp_path, f'scrollbar-drag-released-away-{browser_engine}')
            (tmp_path / f'scrollbar-drag-{browser_engine}.json').write_text(json.dumps({
                'bar': bar, 'dragged': dragged, 'gap': gap, 'released': released, 'resized': resized,
                'first': first, 'kept': kept, 'away': away}))
            assert abs(kept - first['offset']) <= 2 and away > 20, ('a drag released away does not follow', first, kept, away)
            assert page.evaluate('() => window.__historyReads.every(read => !read.cursor)'), 'a scrollbar drag pages nothing'
        finally:
            browser.close()
