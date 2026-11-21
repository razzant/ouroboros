"""A room reopened on a phone paints this browser's saved text before its first read and gives way to the canonical
history (web/modules/chat_preview.js; DESIGN "History edges"), on the production SPA in WebKit and Chromium.

A fresh document of the same origin keeps localStorage, not sessionStorage; a recreated browser context stands in for
that. Canonical reads, never the preview, own history, actions and read receipts. Uses the static-server/API fixture;
never starts the runtime. Touch and visibility are emulated signals into the shell's own owners (room_retire.js,
selection_lifecycle.js), not an iPhone or Telegram client.
"""
from __future__ import annotations

from time import monotonic
from urllib.parse import parse_qs, urlsplit

import pytest

from tests._phone_room_shared import facts, history, shot
from tests.test_design_system_gate_browser import PROJECTS, rest, seed_page, settle
from tests.test_shell_interactions_browser import open_project, shell_ui
from tests.test_subscription_setup_browser import subscription_ui

pytestmark = [pytest.mark.ui_browser]
__all__ = ['shell_ui', 'subscription_ui']
PREVIEW_KEY = 'ouroboros.chat-preview.v1'
SAVED = "key => localStorage.getItem(key) || ''"


def wait_for_reads(page, pending, count=1, seconds=10):
    """Until `count` history reads are held: the read itself, not a guessed duration."""
    until = monotonic() + seconds
    while len(pending) < count and monotonic() < until:
        page.wait_for_timeout(20)
    assert len(pending) >= count, f'{len(pending)} of {count} history reads arrived'


def main_reads(page, pending, project=None):
    """Main's history reads wait in `pending`; a Project's are answered at once with `project` rows."""
    def answer(route):
        if 'chat_id=' in route.request.url:
            route.fulfill(json=history(project or ['A Project line']))
        else:
            pending.append(route)
    page.route('**/api/chat/history**', answer)


def select_preview_text(page, index=0):
    return page.evaluate("""index => {
        const text = document.querySelectorAll('#chat-messages [data-saved-preview] .message')[index];
        const range = document.createRange(); range.selectNodeContents(text);
        getSelection().removeAllRanges(); getSelection().addRange(range);
        return getSelection().toString();
    }""", index)


LOADING_STATEMENTS = """() => [...document.querySelectorAll('#chat-messages *, .chat-page-note *')]
    .filter(e => e.children.length === 0 && /Loading saved history|Saved text preview · updating/.test(e.textContent)
        && e.getClientRects().length && getComputedStyle(e).visibility !== 'hidden')
    .map(e => e.textContent.trim())"""


@pytest.mark.parametrize('mode', ['newest', 'reference', 'selected-empty', 'authorship', 'delta'])
def test_a_new_document_shows_saved_text_before_history_and_reconciles(shell_ui, mode):
    ui = shell_ui
    page = seed_page(ui, 390, 844, telegram=True)
    initial = history(['Saved answer', 'Obsolete preview row'])
    if mode == 'authorship':
        initial = history(['Saved answer\nSecond line', 'Another browser', 'An autonomous thought'])
        initial['messages'][0].update(role='user', source='telegram', sender_label='Mira')
        initial['messages'][1].update(role='user', source='web', sender_session_id='another-window-private')
        initial['messages'][2].update(initiator='consciousness')
    if mode == 'delta':
        initial = history([])
    deltas, delta_ready = [], [False]

    def first_history(route):
        cursor = parse_qs(urlsplit(route.request.url).query).get('cursor')
        if mode == 'delta' and delta_ready[0] and cursor == ['after:recent']:
            deltas.append(route.request.url)
            route.fulfill(json={**history(['Saved answer']), 'after_cursor': 'after:delta'})
        else:
            route.fulfill(json=initial)
    page.route('**/api/chat/history**', first_history)
    page.goto(ui['url'])
    if mode == 'delta':
        page.locator('#chat-messages[data-history-hydrated="true"]').wait_for()
        # The production chat instance's own arrivals read (`after_cursor`), into the same room and storage: a quiet
        # second Main instance, so the delta is the accepted canonical answer, not a newest page.
        page.evaluate("""async () => {
            const {createChatInstance} = await import('/static/modules/chat.js');
            const mount = document.createElement('section'); document.body.append(mount);
            let generation = 0;
            window.__deltaChat = createChatInstance({mountEl: mount, chatId: 1, idPrefix: 'delta',
                ws: {on: () => () => {}, isConnected: () => true, send() {}},
                state: {activePage: 'chat', projectChatIds: new Set(), unreadCount: 0},
                stateSnapshots: {begin: () => ({generation: ++generation, requestedAt: Date.now()}),
                    gate() { return Promise.resolve(this.begin()); }, isCurrent: () => true, apply() {}}});
            await window.__deltaChat.refreshHistory({revision: 1});
        }""")
        delta_ready[0] = True
        page.evaluate('window.__deltaChat.refreshHistory({revision: 2})')
        page.locator('#delta-messages .chat-bubble').filter(has_text='Saved answer').wait_for()
        assert deltas, 'the arrivals read, not a newest page, brought the row'
        page.evaluate('window.__deltaChat.destroy()')
    else:
        page.locator('#chat-messages .chat-bubble').filter(has_text='Saved answer').wait_for()
    page.wait_for_function(f"localStorage.getItem('{PREVIEW_KEY}')?.includes('Saved answer')")
    stored = page.evaluate(SAVED, PREVIEW_KEY)
    assert 'another-window-private' not in stored and 'sender_session_id' not in stored

    context = page.context.browser.new_context(storage_state=page.context.storage_state(),
                                               viewport={'width': 390, 'height': 844}, has_touch=True)
    other = context.new_page()
    other.route_web_socket('**/ws', lambda ws: ws.send('{"type":"heartbeat"}'))
    other.route('**/api/**', lambda route: route.fulfill(json={}))
    other.route('**/api/onboarding', lambda route: route.fulfill(status=204))
    other.route('**/api/state', lambda route: route.fulfill(json={'supervisor_ready': True, 'projects': [],
                                                                     'active_chat_activities': []}))
    other.route('**/api/projects', lambda route: route.fulfill(json={'projects': []}))
    pending = []
    other.route('**/api/chat/history**', lambda route: pending.append(route))
    ui['errors'].watch(other, navigates=True)
    other.goto(ui['url'])
    preview = other.locator('[data-saved-preview]')
    preview.wait_for(state='visible')
    assert 'Saved answer' in preview.inner_text()
    assert preview.locator('button, a, [data-history-id]').count() == 0, 'no action, link or history identity'
    assert other.locator('#chat-messages').get_attribute('data-history-hydrated') is None
    other.locator('#chat-input').fill('Unsent draft')
    wait_for_reads(other, pending)
    assert other.evaluate(LOADING_STATEMENTS) == ['Saved text preview · updating'], 'one loading statement'
    shot(other, f'reopen-preview-{mode}-{ui["engine"]}')
    try:
        if mode == 'authorship':
            expected = ['Mira', 'You', 'Ouroboros · Consciousness']
            assert preview.locator('.sender').all_text_contents() == expected
            text = preview.locator('.message').first
            assert text.text_content() == 'Saved answer\nSecond line'
            assert text.evaluate('e => getComputedStyle(e).whiteSpace') == 'pre-wrap'
            pending.pop(0).fulfill(status=500, json={'error': 'Controlled history failure'})
            other.get_by_role('button', name='Retry loading messages').wait_for()
            assert preview.locator('.chat-load-older-note').text_content() == 'Saved text preview · not current'
            assert preview.locator('.sender').all_text_contents() == expected
            assert other.locator('#chat-input').input_value() == 'Unsent draft'
            shot(other, f'reopen-failed-keeps-preview-{ui["engine"]}')
            other.route('**/api/chat/history**', lambda route: route.fulfill(json=initial))
            other.get_by_role('button', name='Retry loading messages').click()
            other.locator('#chat-messages[data-history-hydrated="true"]').wait_for()
            assert preview.count() == 0
            assert other.locator('#chat-messages .chat-bubble .sender').all_text_contents() == expected
            shot(other, f'reopen-authors-reconciled-{ui["engine"]}')
            return
        if mode == 'reference':
            pending.pop(0).fulfill(status=500, json={'error': 'Controlled history failure'})
            other.get_by_role('button', name='Retry loading messages').wait_for()
            target = history(['**Canonical formatted**'])
            target['anchor'] = {'state': 'found', 'task_id': 'preview-target', 'history_id': 'chat:0', 'chat_id': 1}
            other.route('**/api/chat/history**around=**', lambda route: route.fulfill(json=target))
            other.evaluate("window.dispatchEvent(new CustomEvent('ouro:open-task', {detail: {task_id: 'preview-target', chat_id: 1}}))")
            other.locator('#chat-messages strong').filter(has_text='Canonical formatted').wait_for()
            assert preview.count() == 0, 'the page around the task is canonical: it replaces the preview'
            assert 'Obsolete preview row' not in other.locator('#chat-messages').inner_text()
            assert other.locator('#chat-messages .chat-bubble button').count() > 0
            assert 'Canonical formatted' not in other.evaluate(SAVED, PREVIEW_KEY), 'a page around a task is no newest window'
            # ↓ from a navigation's chain reads the present as a `latest` page: that window refreshes the saved text.
            other.route('**/api/chat/history', lambda route: route.fulfill(json=history(['Latest after the reference'])))
            other.evaluate("document.getElementById('chat-scroll-bottom').click()")
            other.wait_for_function(f"localStorage.getItem('{PREVIEW_KEY}')?.includes('Latest after the reference')")
            return
        if mode == 'selected-empty':
            selected = select_preview_text(other)
            pending.pop(0).fulfill(json=history([]))
            other.locator('#chat-messages[data-history-hydrated="true"]').wait_for()
            assert preview.count() == 1
            assert preview.locator('.chat-load-older-note').text_content() == \
                'Saved text preview · retained until selection is released'
            assert other.evaluate('getSelection().toString()') == selected == 'Saved answer'
            shot(other, f'reopen-selected-retained-{ui["engine"]}')
            other.evaluate("getSelection().removeAllRanges()")
            preview.wait_for(state='detached')
            assert 'Saved answer' not in other.evaluate(SAVED, PREVIEW_KEY), 'the empty read replaced the saved text'
            return
        pending.pop(0).fulfill(json=history(['Saved answer', 'Fresh answer']))
        other.locator('#chat-messages[data-history-hydrated="true"]').wait_for()
        assert preview.count() == 0
        for text in ['Saved answer', 'Fresh answer']:
            assert other.locator('#chat-messages .chat-bubble').filter(has_text=text).count() == 1
        assert other.locator('#chat-input').input_value() == 'Unsent draft'
        shot(other, f'reopen-reconciled-{mode}-{ui["engine"]}')
    finally:
        context.close()
        page.close()


@pytest.mark.parametrize('read_older', [False, True], ids=['follow-latest', 'keep-older'])
def test_a_long_warm_preview_lands_before_the_canonical_handover_keeping_draft_and_files(shell_ui, read_older):
    """Padding and layout must not turn an opening preview into older-reading intent; a deliberate wheel must."""
    page = seed_page(shell_ui, 390, 844, telegram=True)
    saved = history([f'Line {i} ' + 'Long conversation text wraps. ' * 18 for i in range(50)])
    page.route('**/api/chat/history**', lambda route: route.fulfill(json=saved))
    page.goto(shell_ui['url'])
    page.locator('#chat-messages[data-history-hydrated="true"]').wait_for()
    page.wait_for_function(f"localStorage.getItem('{PREVIEW_KEY}')?.includes('Line 49')")
    pending = []
    page.route('**/api/chat/history**', lambda route: pending.append(route))
    page.reload()
    page.locator('[data-saved-preview]').wait_for()

    def gap():
        return page.locator('#chat-messages').evaluate('f => f.scrollHeight - f.clientHeight - f.scrollTop')
    page.wait_for_function("() => {const f = document.querySelector('#chat-messages'); "
                           "return f.scrollHeight - f.clientHeight - f.scrollTop <= 1;}")
    page.locator('#chat-input').fill('A draft written over the saved text')
    page.set_input_files('#chat-file-input', files=[{'name': 'notes.txt', 'mimeType': 'text/plain', 'buffer': b'kept'}])
    page.locator('.attach-badge').filter(has_text='notes.txt').wait_for()
    if read_older:
        page.locator('[data-saved-preview] .message').last.dispatch_event('wheel', {'deltaY': -200, 'bubbles': True})
        page.locator('#chat-messages').evaluate('f => {f.scrollTop = 1000; f.dispatchEvent(new Event("scroll"));}')
    wait_for_reads(page, pending)
    reading = page.locator('#chat-messages').evaluate("""f => {
        const top = f.getBoundingClientRect().top;
        const row = [...f.querySelectorAll('[data-preview-history-id]')].find(e => e.getBoundingClientRect().bottom > top + 80);
        return row ? {id: row.dataset.previewHistoryId, offset: row.getBoundingClientRect().top - top} : null;
    }""")
    while pending:
        pending.pop(0).fulfill(json=saved)
    page.locator('#chat-messages[data-history-hydrated="true"]').wait_for()
    settle(page)
    settle(page)
    landed = page.locator('#chat-messages').evaluate("""(f, id) => {
        const node = f.querySelector(`[data-history-id="${id}"]`);
        return node ? node.getBoundingClientRect().top - f.getBoundingClientRect().top : null;
    }""", reading['id'])
    facts(f'warm-handover-{"older" if read_older else "latest"}-{shell_ui["engine"]}',
          {'gap': gap(), 'read_before': reading, 'canonical_offset': landed})
    if read_older:
        assert gap() > 1000
        assert landed is not None and abs(landed - reading['offset']) <= 40, 'the passage read stays where it was read'
    else:
        assert gap() <= 1
    assert page.locator('[data-saved-preview]').count() == 0
    assert page.locator('#chat-input').input_value() == 'A draft written over the saved text'
    assert page.locator('.attach-badge').filter(has_text='notes.txt').count() == 1
    shot(page, f'warm-handover-{"older" if read_older else "latest"}-{shell_ui["engine"]}')
    page.close()


BACK = 'Telegram.WebApp.BackButton.callback()'
SELECTION = 'getSelection().toString()'


@pytest.mark.parametrize('cover', ['drawer', 'room', 'hidden'])
def test_a_selected_preview_passage_outlasts_a_covered_room_and_an_ordinary_click_releases_it(shell_ui, cover):
    """The selection room_retire.js keeps while a layer covers the room (WebKit drops a range in an inert room) or the
    room is away holds the preview after the canonical read; nothing but a release lets it go."""
    page = seed_page(shell_ui, 390, 844, telegram=True)
    page.route('**/api/chat/history**', lambda route: route.fulfill(json=history(['A passage the owner selects', 'Second line'])))
    page.goto(shell_ui['url'])
    page.wait_for_function(f"localStorage.getItem('{PREVIEW_KEY}')?.includes('A passage the owner selects')")
    pending = []
    main_reads(page, pending)
    page.reload()
    page.locator('[data-saved-preview]').wait_for()
    wait_for_reads(page, pending)
    assert select_preview_text(page) == 'A passage the owner selects'
    pending.pop(0).fulfill(json=history(['A passage the owner selects', 'Second line']))
    page.locator('#chat-messages[data-history-hydrated="true"]').wait_for()
    preview = page.locator('[data-saved-preview]')
    assert preview.count() == 1 and page.evaluate(SELECTION) == 'A passage the owner selects'
    observed = {'hydrated': page.evaluate(SELECTION)}
    if cover in ('drawer', 'room'):
        page.locator('[data-mobile-nav-toggle]:visible').first.tap()
        page.wait_for_function("() => window.__ouroLayers.top() === 'drawer'")
        rest(page)
        observed['covered'] = {'selection': page.evaluate(SELECTION), 'preview': preview.count()}
        assert preview.count() == 1, 'the drawer over the room keeps the held passage'
        if cover == 'room':
            page.locator('#nav-projects-list [data-project-id="gate-0"]').tap()
            page.wait_for_function("() => window.__ouroLayers.top() === 'project'")
            rest(page)
            observed['other room'] = {'selection': page.evaluate(SELECTION), 'preview': preview.count()}
            page.evaluate(BACK)
        else:
            page.locator('.nav-drawer-backdrop').tap()
        page.wait_for_function("() => window.__ouroLayers.size() === 0")
        rest(page)
        rest(page)
    else:
        page.evaluate("""hidden => {
            for (const [name, value] of [['hidden', true], ['visibilityState', 'hidden']])
                Object.defineProperty(document, name, {configurable: true, get: () => value});
            document.dispatchEvent(new Event('visibilitychange'));
        }""", True)
        settle(page)
        observed['hidden'] = {'selection': page.evaluate(SELECTION), 'preview': preview.count()}
        page.evaluate("""() => {
            for (const name of ['hidden', 'visibilityState']) delete document[name];
            document.dispatchEvent(new Event('visibilitychange'));
        }""")
        rest(page)
    observed['back'] = {'selection': page.evaluate(SELECTION), 'preview': preview.count(),
                        'note': preview.locator('.chat-load-older-note').text_content() if preview.count() else None}
    facts(f'selected-preview-{cover}-{shell_ui["engine"]}', observed)
    shot(page, f'selected-preview-after-{cover}-{shell_ui["engine"]}')
    assert preview.count() == 1, observed
    assert page.evaluate(SELECTION) == 'A passage the owner selects', 'the owner gets the selection back'
    # A deliberate primary click elsewhere ends the selection (selection_lifecycle.js): the preview goes with it.
    box = page.locator('#chat-messages [data-history-id] .message').last.bounding_box()
    page.mouse.click(box['x'] + box['width'] - 8, box['y'] + box['height'] / 2)
    preview.wait_for(state='detached')
    assert page.locator('#chat-messages [data-history-id]').count() >= 2
    page.close()


@pytest.mark.parametrize('read,status', [('newest', 401), ('older', 401), ('older', 403), ('older', 500)],
                         ids=['newest-401', 'older-401', 'older-403', 'older-500'])
def test_a_refused_session_takes_the_saved_text_at_once_even_while_selected(shell_ui, read, status):
    """Any history read the room makes, refused, takes the saved text at once, a held passage too: its newest read, or a
    page (Load more history after the canonical read, the passage still selected). A page that merely failed takes
    nothing: the held passage keeps its label and its storage, and Retry is the room's."""
    page = seed_page(shell_ui, 390, 844, telegram=True)
    recent = history(['Private saved words', 'Canonical newest row'], start=50)
    recent.update(has_more=True, next_cursor='before:older')
    recent['window'].update(complete=False, truncated_by=['quota'])
    recent['coverage']['spans']['chat']['from'] = 50
    page.route('**/api/chat/history**', lambda route: route.fulfill(json=recent))
    page.goto(shell_ui['url'])
    page.wait_for_function(f"localStorage.getItem('{PREVIEW_KEY}')?.includes('Private saved words')")
    pending = []
    main_reads(page, pending)
    page.reload()
    page.locator('[data-saved-preview]').wait_for()
    wait_for_reads(page, pending)
    page.locator('#chat-input').fill('Draft kept by the composer')  # first: focusing the field collapses a selection
    assert select_preview_text(page) == 'Private saved words'
    preview = page.locator('[data-saved-preview]')
    if read == 'older':
        pending.pop(0).fulfill(json=recent)
        page.locator('#chat-messages[data-history-hydrated="true"]').wait_for()
        assert preview.count() == 1, 'the held passage outlasts the canonical read'
        page.locator('#chat-messages .chat-load-older-btn').tap()
        wait_for_reads(page, pending)
        assert 'cursor=before%3Aolder' in pending[0].request.url, pending[0].request.url
    assert page.evaluate(SELECTION) == 'Private saved words', 'the passage is held selected when the answer lands'
    pending.pop(0).fulfill(status=status, json={'error': 'Telegram owner session required.'})
    name = f'refused-{read}-{status}-{shell_ui["engine"]}'
    if status == 500:
        page.get_by_role('button', name='Retry loading messages').wait_for()
        observed = {'preview': preview.count(), 'note': preview.locator('.chat-load-older-note').text_content(),
                    'selection': page.evaluate(SELECTION), 'stored': 'Private saved words' in page.evaluate(SAVED, PREVIEW_KEY)}
        facts(name, observed)
        shot(page, name)
        assert observed == {'preview': 1, 'note': 'Saved text preview · retained until selection is released',
                            'selection': 'Private saved words', 'stored': True}, observed
        page.evaluate('getSelection().removeAllRanges()')
        preview.wait_for(state='detached')
        page.close()
        return
    preview.wait_for(state='detached')
    assert page.evaluate(f"localStorage.getItem('{PREVIEW_KEY}')") is None
    if read == 'newest':  # no canonical read ever drew these words
        assert 'Private saved words' not in page.locator('#chat-messages').inner_text()
    assert page.locator('#chat-input').input_value() == 'Draft kept by the composer'
    facts(name, {'preview': preview.count(), 'stored': page.evaluate(f"localStorage.getItem('{PREVIEW_KEY}')")})
    shot(page, name)
    page.close()


def test_main_says_welcome_after_an_authoritative_empty_read_never_after_a_failed_one(shell_ui):
    page = seed_page(shell_ui, 390, 844, telegram=True)
    page.route('**/api/ui/preferences', lambda route: route.fulfill(json={'welcome': {'mode': 'default'}}))
    page.route('**/api/chat/history**', lambda route: route.fulfill(json=history(['Stale saved text'])))
    page.goto(shell_ui['url'])
    page.wait_for_function(f"localStorage.getItem('{PREVIEW_KEY}')?.includes('Stale saved text')")
    pending = []
    main_reads(page, pending)
    page.reload()
    page.locator('[data-saved-preview]').wait_for()
    wait_for_reads(page, pending)
    pending.pop(0).fulfill(status=503, json={'error': 'history_source_unavailable', 'reason_code': 'history_source_unavailable',
                                             'messages': [], 'has_more': True, 'next_cursor': None})
    page.get_by_role('button', name='Retry loading messages').wait_for()
    assert page.locator('.chat-empty-welcome').count() == 0, 'a failed read confirms no emptiness'
    assert page.locator('[data-saved-preview] .chat-load-older-note').text_content() == 'Saved text preview · not current'
    shot(page, f'failed-read-keeps-preview-no-welcome-{shell_ui["engine"]}')
    page.route('**/api/chat/history**', lambda route: route.fulfill(json=history([])))
    page.get_by_role('button', name='Retry loading messages').click()
    page.locator('.chat-empty-welcome').wait_for()
    assert page.locator('[data-saved-preview]').count() == 0
    assert 'Stale saved text' not in page.evaluate(SAVED, PREVIEW_KEY)
    shot(page, f'empty-read-replaces-preview-with-welcome-{shell_ui["engine"]}')
    page.close()


def test_a_project_read_answered_after_its_room_was_left_is_neither_drawn_nor_saved(shell_ui):
    page = seed_page(shell_ui, 390, 844, telegram=True)
    held = []

    def answer(route):
        if 'chat_id=100' in route.request.url:
            held.append(route)
        else:
            route.fulfill(json=history(['Main line']))
    page.route('**/api/chat/history**', answer)
    page.goto(shell_ui['url'])
    page.locator('#chat-messages[data-history-hydrated="true"]').wait_for()
    open_project(page)
    wait_for_reads(page, held)
    page.evaluate(BACK)
    page.wait_for_function("() => window.__ouroLayers.size() === 0")
    rest(page)
    try:
        held.pop(0).fulfill(json=history(['Late project answer'], start=40))
    except Exception as error:  # noqa: BLE001 - an aborted read may refuse its late answer; either way it is late
        print('late fulfil refused:', error)
    page.wait_for_timeout(400)
    assert 'Late project answer' not in page.evaluate(SAVED, PREVIEW_KEY)
    assert page.locator('text=Late project answer').count() == 0
    open_project(page)
    wait_for_reads(page, held)
    held.pop(0).fulfill(json=history(['Fresh project answer'], start=50))
    page.locator('.chat-instance-panel .chat-bubble').filter(has_text='Fresh project answer').wait_for()
    page.wait_for_function(f"localStorage.getItem('{PREVIEW_KEY}')?.includes('Fresh project answer')")
    assert 'Late project answer' not in page.evaluate(SAVED, PREVIEW_KEY)
    page.close()


@pytest.mark.parametrize('case', ['registry-deleting', 'answered-then-gone', 'refused', 'absent-only'])
def test_a_projects_saved_text_leaves_only_on_the_word_that_it_is_going_away(shell_ui, case):
    """Astra R2: a deleted Project's room kept its saved text (up to 80 rows) until eight newer rooms pushed it out. Two
    words take it, and only its rooms: the registry's `deleting` (app.js, where its room is let go too) and the core's
    answer to this browser's delete, which fenced it `deleting` before replying, so a tombstone may land before any
    read shows `deleting` (answered-then-gone). A refused delete keeps it, and so does absence alone from the bounded
    list: no deletion is inferred. Moving between rooms takes nothing."""
    page = seed_page(shell_ui, 1280, 800, telegram=False, touch=False)
    words = {'chat_id=100': 'Words of the deleted Project', 'chat_id=101': 'Another Project keeps these'}
    page.route('**/api/chat/history**', lambda route: route.fulfill(json=history([next(
        (text for query, text in words.items() if query in route.request.url), 'Main keeps these')])))
    page.goto(shell_ui['url'])
    page.locator('#chat-messages[data-history-hydrated="true"]').wait_for()
    for index, text in enumerate(words.values()):
        open_project(page, index)
        page.wait_for_function(f"localStorage.getItem('{PREVIEW_KEY}')?.includes('{text}')")
        page.keyboard.press('Escape')
        page.wait_for_function("() => window.__ouroLayers.size() === 0")
        rest(page)
    before = page.evaluate(SAVED, PREVIEW_KEY)
    assert all(text in before for text in ['Main keeps these', *words.values()]), 'navigation takes nothing'
    listed = ([project for project in PROJECTS if project['id'] != 'gate-0'] if case in ('answered-then-gone', 'absent-only')
              else [dict(project, lifecycle='deleting') if project['id'] == 'gate-0' else project for project in PROJECTS]
              if case == 'registry-deleting' else PROJECTS)
    reads = []

    def state(route):
        reads.append(route.request.url)
        route.fulfill(json={'supervisor_ready': True, 'active_chat_activities': [], 'projects': listed,
                            'project_chat_ids': [project['chat_id'] for project in listed]})
    page.route('**/api/state', state)
    if case in ('answered-then-gone', 'refused'):
        page.route('**/api/projects/gate-0/delete', lambda route: route.fulfill(
            json={'ok': True, 'project_id': 'gate-0', 'folder_untouched': True}) if case == 'answered-then-gone'
            else route.fulfill(status=500, json={'error': 'the core could not delete it now'}))
        page.locator('.nav-project-item[data-nav-project-id="gate-0"] .nav-project-kebab').click()
        page.locator('body > .project-row-menu [data-prm="delete"]').click()
        page.locator('.confirm-dialog [data-confirm-ok]').click()
        if case == 'refused':  # "Delete did not finish": its one button closes it
            page.locator('.confirm-dialog [data-confirm-ok]').click()
    gone = case in ('registry-deleting', 'answered-then-gone')
    if gone:
        page.wait_for_function(f"!(localStorage.getItem('{PREVIEW_KEY}') || '').includes('Words of the deleted Project')",
                               timeout=10_000)
    else:  # the shell's own next read (its poll, at most 3 s) has landed and been drawn
        reads.clear()
        for _ in range(50):
            if reads:
                break
            page.wait_for_timeout(200)
        assert reads, 'a state read came after the change'
        if case == 'absent-only':
            page.wait_for_function('() => !document.querySelector(\'[data-nav-project-id="gate-0"]\')')
        rest(page)
    after = page.evaluate(SAVED, PREVIEW_KEY)
    facts(f'project-preview-{case}-{shell_ui["engine"]}', {'before': len(before), 'after': len(after),
                                                           'kept': 'Words of the deleted Project' in after})
    assert ('Words of the deleted Project' in after) is not gone, after
    assert 'Main keeps these' in after and 'Another Project keeps these' in after, after
    page.close()


def test_the_sign_in_page_every_sign_out_reaches_releases_the_saved_text_and_nothing_else(shell_ui):
    """/auth/logout (GET or POST) and every ended session land on the core's own sign-in page before the interface runs
    again (ouroboros/server_auth.py; tests/test_server_auth.py pins the routes). Its markup, run here by the engine,
    removes this origin's saved conversation text; the appearance choice and the rest of the browser's keys stay."""
    from ouroboros import server_auth

    page = seed_page(shell_ui, 390, 844, telegram=False)
    page.route('**/api/chat/history**', lambda route: route.fulfill(json=history(['Words that leave with the session'])))
    page.goto(shell_ui['url'])
    page.wait_for_function(f"localStorage.getItem('{PREVIEW_KEY}')?.includes('Words that leave with the session')")
    page.evaluate("localStorage.setItem('ouroboros.theme', 'light')")
    page.route('**/auth/login**', lambda route: route.fulfill(status=401, content_type='text/html',
                                                              body=server_auth._login_page('/')))
    page.goto(shell_ui['url'] + '/auth/login?next=%2F')
    page.locator('input[name="password"]').wait_for()
    observed = page.evaluate("key => ({saved: localStorage.getItem(key), theme: localStorage.getItem('ouroboros.theme'),"
                             " appearance: document.documentElement.dataset.theme || null})", PREVIEW_KEY)
    facts(f'sign-in-page-releases-preview-{shell_ui["engine"]}', observed)
    shot(page, f'sign-in-page-releases-preview-{shell_ui["engine"]}')
    assert observed == {'saved': None, 'theme': 'light', 'appearance': 'light'}, observed
    page.close()


@pytest.mark.parametrize('answer', ['refused-401', 'refused-403', 'failed-500', 'superseded-401'])
def test_a_link_read_outside_the_room_takes_the_saved_text_only_when_currently_refused(shell_ui, answer):
    """Opening a task reads the page around it before switching (app.js revealQuotedSource), outside the room's own reads.
    A current refusal takes the saved text and every painted copy at once, a held passage too; a failed read keeps the
    navigation, the held passage and the draft; a refusal answered after a newer navigation decides nothing."""
    page = seed_page(shell_ui, 390, 844, telegram=True)
    page.route('**/api/chat/history**', lambda route: route.fulfill(json=history(['A passage the owner selects', 'Second line'])))
    page.goto(shell_ui['url'])
    page.wait_for_function(f"localStorage.getItem('{PREVIEW_KEY}')?.includes('A passage the owner selects')")
    pending = []
    main_reads(page, pending)
    page.reload()
    page.locator('[data-saved-preview]').wait_for()
    wait_for_reads(page, pending)
    page.locator('#chat-input').fill('Draft kept by the composer')  # first: focusing the field collapses a selection
    assert select_preview_text(page) == 'A passage the owner selects'
    pending.pop(0).fulfill(json=history(['A passage the owner selects', 'Second line']))
    page.locator('#chat-messages[data-history-hydrated="true"]').wait_for()
    preview = page.locator('[data-saved-preview]')
    assert preview.count() == 1 and page.evaluate(SELECTION) == 'A passage the owner selects'
    around = []
    page.route('**/api/chat/history**around=**', lambda route: around.append(route))
    page.evaluate("window.dispatchEvent(new CustomEvent('ouro:open-task', {detail: {task_id: 'link-target', chat_id: 1}}))")
    wait_for_reads(page, around)
    assert 'around=task%3Alink-target' in around[0].request.url, around[0].request.url
    if answer.startswith('superseded'):
        open_project(page)  # a newer navigation: the link's read is no longer current
    status = int(answer.split('-')[1])
    around.pop(0).fulfill(status=status, json={'error': 'Authentication required.'})
    stored = f"() => localStorage.getItem('{PREVIEW_KEY}')"
    if answer.startswith('refused'):
        preview.wait_for(state='detached')
    else:
        page.wait_for_timeout(600)
    observed = {'preview': preview.count(), 'stored': page.evaluate(stored) is not None,
                'selection': page.evaluate(SELECTION), 'draft': page.locator('#chat-input').input_value(),
                'top': page.evaluate('window.__ouroLayers.top() || null')}
    name = f'link-read-{answer}-{shell_ui["engine"]}'
    facts(name, observed)
    shot(page, name)
    if answer.startswith('refused'):
        assert observed['preview'] == 0 and not observed['stored'], observed
        assert observed['top'] is None, 'the room the owner is in stays'
    elif answer == 'failed-500':
        assert observed['preview'] == 1 and observed['stored'] and observed['top'] is None, observed
        assert observed['selection'] == 'A passage the owner selects', observed
    else:
        assert observed['preview'] == 1 and observed['stored'] and observed['top'] == 'project', observed
    assert observed['draft'] == 'Draft kept by the composer', observed
    page.close()
