"""Toast motion (DESIGN "Motion"), the shared edge on cut words alone (principle 9) and what a press does: the words
are only words, every toast's × only hides it, and a notification's Open leads to its source (owner 26.11, A).
Uses the real UI on a static-server fixture; never starts the runtime or reads live data."""
from __future__ import annotations

import json
import re

import pytest

from tests import test_subscription_setup_browser as setup_browser
from tests.settings_browser_helpers import open_settings
from tests.test_design_system_gate_browser import PROJECTS, rest, save_evidence, seed_page, settle
from tests.ui_clipboard_sink import INSTALL, recorded

subscription_ui = setup_browser.subscription_ui
pytestmark = [pytest.mark.ui_browser]


@pytest.fixture(params=['webkit', 'chromium'])
def toast_ui(request, monkeypatch):
    monkeypatch.setenv('OUROBOROS_UI_BROWSER_ENGINE', request.param)
    ui = request.getfixturevalue('subscription_ui')
    ui['engine'] = request.param
    return ui


def _open(ui, width, height, *, touch=False):
    page = seed_page(ui, width, height, touch=touch)
    page.goto(ui['url'] + '/')
    page.wait_for_selector('#chat-input', state='attached')
    settle(page)
    # The page's own notices (a room's history read, once its answer or failure has landed) are not the stack under
    # study: the room's first read settles, then they leave, so every count below is the test's own.
    page.wait_for_function("""() => document.querySelector('#chat-messages')?.dataset.historyHydrated === 'true'
        || document.querySelector('#chat-messages .chat-load-older-btn')?.textContent === 'Retry loading messages'""")
    page.evaluate("() => document.querySelectorAll('#toast-stack .toast').forEach((toast) => toast.dispose())")
    return page


# Three toasts, then the first dismissed: the top of the one that was second, frame by frame, until it rests.
GLIDE = """async () => {
    const { showToast } = await import('/static/modules/toast.js');
    const toasts = ['First notice', 'Second notice', 'Third notice'].map((text) => showToast(text, 'info', { ttl: 0 }));
    await new Promise((resolve) => setTimeout(resolve, 400));  // past their own arrival
    const second = toasts[1];
    const tops = [second.getBoundingClientRect().top];
    toasts[0].querySelector('.toast-dismiss').click();
    for (let i = 0; i < 40; i += 1) {
        await new Promise((resolve) => requestAnimationFrame(resolve));
        tops.push(second.getBoundingClientRect().top);
    }
    return { tops, gone: !toasts[0].isConnected, stack: document.querySelectorAll('#toast-stack .toast').length };
}"""


@pytest.mark.parametrize('reduce', [False, True], ids=['motion', 'reduced'])
def test_a_dismissed_toast_leaves_and_the_one_below_glides_into_its_place(toast_ui, tmp_path, reduce):
    page = _open(toast_ui, 1280, 800)
    if reduce:
        page.emulate_media(reduced_motion='reduce')
    facts = {'engine': toast_ui['engine'], 'reduce': reduce}
    try:
        facts.update(page.evaluate(GLIDE))
    finally:
        save_evidence(tmp_path, f'toast-glide-{toast_ui["engine"]}-{"reduced" if reduce else "motion"}', facts)
        print(json.dumps(facts))
        page.close()
    tops = facts['tops']
    steps = [round(b - a, 2) for a, b in zip(tops, tops[1:])]
    travel = tops[0] - tops[-1]
    assert facts['gone'] and facts['stack'] == 2, facts
    assert travel > 30, ('the one below took the place of the one that left', facts)
    assert all(step <= 0.5 for step in steps), ('it only ever moves up into the place', steps)
    if reduce:
        assert sum(1 for step in steps if step < -0.5) <= 1, ('at once with reduced motion', steps)
    else:
        assert max(-step for step in steps) < travel * 0.6, ('no one-frame jump into the place', steps)
        assert sum(1 for step in steps if step < -0.5) >= 3, ('it glides over several frames', steps)


STACK = '#toast-stack'
CLOSE = f'{STACK} .toast-dismiss'
PHONE = (320, 568)
WIDE = (1280, 1000)
CLEAR, ABOVE, BELOW, BOTH = (False, False), (True, False), (False, True), (True, True)


# Inspect actual boxes/paint together. The text may take a mask; its window, glass and controls must not.
STACK_VIEW = """stack => {
    const maskAxisY = (s, prop, fallback) => {
        const raw = (s.getPropertyValue(prop) || s.getPropertyValue(fallback) || '').trim();
        if (!raw) return '';
        const parts = raw.split(/\\s+/);
        return parts[1] ?? parts[0] ?? '';
    };
    const box = (el) => {
        if (!el) return null;
        const { top, bottom, left, right, height } = el.getBoundingClientRect();
        return { top, bottom, left, right, height, mask: getComputedStyle(el).maskImage };
    };
    const stackBox = box(stack);
    const stackCut = stackBox.top + stack.clientTop + stack.clientHeight;
    const toasts = [...stack.querySelectorAll('.toast')].map(toast => {
        const words = toast.querySelector('.toast-words');
        const wordsBox = box(words);
        if (wordsBox) {
            const ws = getComputedStyle(words);
            const posY = maskAxisY(ws, 'mask-position', '-webkit-mask-position')
                || (ws.getPropertyValue('mask-position-y') || ws.getPropertyValue('-webkit-mask-position-y') || '').trim();
            const sizeY = maskAxisY(ws, 'mask-size', '-webkit-mask-size');
            wordsBox.mask_position_y = posY;
            wordsBox.mask_size_y = sizeY;
            wordsBox.mask_end = (posY.endsWith('px') || posY === '0') && sizeY.endsWith('px')
                ? parseFloat(posY) + parseFloat(sizeY) : null;
            wordsBox.measured_cut = stackCut - wordsBox.top;
        }
        return { ...box(toast), words: wordsBox, close: box(toast.querySelector('.toast-dismiss')),
            action: box(toast.querySelector('.toast-action')),
            above: words?.hasAttribute('data-scroll-above'), below: words?.hasAttribute('data-scroll-below') };
    });
    return { ...stackBox, client_height: stack.clientHeight,
        scroll_top: stack.scrollTop, scroll_end: stack.scrollHeight - stack.clientHeight,
        limit: parseFloat(getComputedStyle(stack).maxHeight), role: stack.getAttribute('role'),
        label: stack.getAttribute('aria-label'), view_width: innerWidth, view_height: innerHeight, toasts,
        focused_close: [...stack.querySelectorAll('.toast-dismiss')].indexOf(document.activeElement) };
}"""


def _read(page):
    return page.locator(STACK).evaluate(STACK_VIEW)


def _assert_edges(view, *edges):
    toasts = view['toasts']
    assert [(t['above'], t['below']) for t in toasts] == list(edges), view
    assert view['mask'] == 'none', view
    for toast, edge in zip(toasts, edges):
        assert (toast['words']['mask'] != 'none') == any(edge), view
        assert all(b['mask'] == 'none' for b in (toast, toast['close'], toast['action']) if b), view


def _shown(box, view):
    return (max(0, view['top']) - 0.5 <= box['top'] and box['bottom'] <= min(view['view_height'], view['bottom']) + 0.5
            and -0.5 <= box['left'] and box['right'] <= view['view_width'] + 0.5)


def _scroll(page, top):
    page.locator(STACK).evaluate("(s, top) => s.scrollTo({ top, behavior: 'instant' })", top)
    settle(page)


def _wait_edge(page, end):
    page.wait_for_function("""end => {
        const s = document.getElementById('toast-stack');
        return end ? s.scrollTop >= s.scrollHeight - s.clientHeight - 1 : s.scrollTop === 0;
    }""", arg=end)
    settle(page)


def _two_kept_warnings(page):
    # Two actual Resume results; only their server answers are supplied. The transient "Resuming" toasts expire.
    page.evaluate("""async () => {
        const { resumeTaskAction } = await import('/static/modules/task_control_menu.js');
        for (const id of ['gap-a', 'gap-b']) {
            const rows = Array.from({ length: 20 }, (_, n) => ({ task_id: `${id}-${n}-${'f'.repeat(24)}`,
                reason: 'body_unreadable' }));
            await resumeTaskAction(id, { resume: async () => ({ ok: true, membership_gaps: { count: 23, rows } }) });
        }
    }""")
    page.wait_for_function("""() => {
        const toasts = [...document.querySelectorAll('#toast-stack .toast')];
        return toasts.length === 2 && toasts.every(el => el.classList.contains('toast-warn'));
    }""", timeout=15_000)
    rest(page)


@pytest.mark.parametrize('width,height', [PHONE, WIDE], ids=['phone', 'wide'])
def test_two_kept_warnings_are_reached_whole_by_wheel_and_keyboard(toast_ui, tmp_path, width, height):
    page = _open(toast_ui, width, height, touch=True)
    name = f'toast-reach-{toast_ui["engine"]}-{width}x{height}'
    facts = {}
    try:
        _two_kept_warnings(page)
        wide = (width, height) == WIDE
        for theme in ('dark', 'light'):
            page.emulate_media(color_scheme=theme)
            page.wait_for_function('(t) => document.documentElement.dataset.theme === t', arg=theme)
            rest(page)
            facts[theme] = start = _read(page)
            _assert_edges(start, CLEAR if wide else BELOW, CLEAR)
            setup_browser.capture(page, f'{name}-{theme}')
        first = start['toasts'][0]
        assert abs(first['top'] - 76) < 0.5, start
        if wide:
            assert start['scroll_end'] == 0 and start['role'] is None, start
            assert all(abs(width - 18 - t['right']) < 0.5 and _shown(t, start) and _shown(t['close'], start)
                       for t in start['toasts']), start
            return
        assert start['bottom'] <= height - 18 + 0.5 and start['scroll_end'] > 0, start
        assert (start['role'], start['label']) == ('region', 'Scrollable notices') and _shown(first['close'], start), start
        page.mouse.move((first['left'] + first['right']) / 2, (first['top'] + start['bottom']) / 2)
        page.mouse.wheel(0, start['scroll_end'] + height)
        _wait_edge(page, True)
        facts['wheel-end'] = end = _read(page)
        _assert_edges(end, CLEAR, ABOVE)  # both notices remain; the last word and its close are in sight
        assert end['toasts'][1]['words']['bottom'] <= end['bottom'] + 0.5 and _shown(end['toasts'][1]['close'], end), end
        setup_browser.capture(page, f'{name}-wheel-end')
        page.mouse.move(1, height / 2)
        page.locator(STACK).focus()
        for key, at_end in (('Home', False), ('End', True), ('Home', False)):
            page.keyboard.press(key)
            _wait_edge(page, at_end)
        words = first['words']
        both = (words['top'] + words['bottom'] - start['top'] - start['bottom']) / 2
        for state, y, edges in (('middle', start['scroll_end'] / 2, (ABOVE, BELOW)), ('both', both, (BOTH, CLEAR))):
            _scroll(page, y)
            facts[state] = _read(page)
            _assert_edges(facts[state], *edges)
        _scroll(page, 0)
        page.locator(CLOSE).first.focus()
        # Safari moves between buttons with Option-Tab.
        page.keyboard.press('Alt+Tab' if toast_ui['engine'] == 'webkit' else 'Tab')
        rest(page)
        facts['focus'] = focused = _read(page)
        assert focused['focused_close'] == 1 and _shown(focused['toasts'][1]['close'], focused), focused
        assert focused['toasts'][1]['close']['mask'] == 'none', focused
        page.keyboard.press('Enter')
        page.locator(CLOSE).last.wait_for(state='visible')
        page.wait_for_function("() => document.querySelectorAll('#toast-stack .toast').length === 1")
        _scroll(page, 0)
        facts['last'] = last = _read(page)
        close = last['toasts'][0]['close']
        assert _shown(close, last), last
        page.mouse.click((close['left'] + close['right']) / 2, (close['top'] + close['bottom']) / 2)
        page.locator(CLOSE).wait_for(state='detached')
    finally:
        save_evidence(tmp_path, name, facts)
        page.close()


def test_the_kept_column_follows_the_window_and_scrolls_from_its_margin(toast_ui, tmp_path):
    page = _open(toast_ui, *PHONE, touch=True)
    facts = {}
    try:
        _two_kept_warnings(page)
        for size, cut in ((PHONE, True), (WIDE, False), (PHONE, True)):
            page.set_viewport_size(dict(zip(('width', 'height'), size)))
            page.locator(f'{STACK}[role="region"]').wait_for(state='attached' if cut else 'detached')
            settle(page)
            view = _read(page)
            facts[f'{size}-{len(facts)}'] = view
            _assert_edges(view, BELOW if cut else CLEAR, CLEAR)
            x = (view['left'] + view['toasts'][0]['left']) / 2
            y = (view['top'] + view['bottom']) / 2
            assert page.locator(STACK).evaluate('(s, [x, y]) => s.contains(document.elementFromPoint(x, y))', [x, y]) == cut
            if cut and len(facts) == 1:
                page.mouse.move(x, y)
                page.mouse.wheel(0, 300)
                page.wait_for_function("() => document.getElementById('toast-stack').scrollTop > 0")
    finally:
        save_evidence(tmp_path, f'toast-follow-{toast_ui["engine"]}', facts)
        page.close()


def test_padding_the_column_scrolls_into_cuts_no_word_and_shows_no_edge(toast_ui, tmp_path):
    page = _open(toast_ui, *PHONE)
    facts = {}
    try:
        page.evaluate("""async () => {
            const { showToast } = await import('/static/modules/toast.js');
            showToast('The record is still available. '.repeat(36), 'warn', { ttl: 0 });
        }""")
        # Measure once, then resize the window around this same notice. A word-count search confounds font metrics
        # with the real distinction: the shadow reserve can overflow even though every word and control fits.
        rest(page)
        facts['initial'] = initial = _read(page)
        content = initial['bottom'] - initial['top'] + initial['scroll_end']
        padded_height = round(PHONE[1] + content - initial['limit'] - 6)
        page.set_viewport_size({'width': PHONE[0], 'height': padded_height})
        page.locator(f'{STACK}[role="region"]').wait_for(state='attached')
        for edge, y in (('start', 0), ('end', 1e5)):
            _scroll(page, y)
            facts[edge] = padded = _read(page)
            assert 0 < padded['scroll_end'] <= 11 and _shown(padded['toasts'][0]['words'], padded), padded
            _assert_edges(padded, CLEAR)
        # Shrink the window just past the bottom text line, without changing or replacing the warning.
        word_bottom = initial['toasts'][0]['words']['bottom'] - initial['top']
        page.set_viewport_size({'width': PHONE[0], 'height': round(PHONE[1] + word_bottom - initial['limit'] - 20)})
        _scroll(page, 0)
        page.wait_for_selector(f'{STACK} .toast-words[data-scroll-below]')
        facts['cut'] = _read(page)
        _assert_edges(facts['cut'], BELOW)
        _scroll(page, 1e5)
        facts['cut-end'] = _read(page)
        _assert_edges(facts['cut-end'], ABOVE)
        assert facts['cut-end']['toasts'][0]['words']['bottom'] <= facts['cut-end']['bottom'], facts
    finally:
        save_evidence(tmp_path, f'toast-padding-{toast_ui["engine"]}', facts)
        page.close()


def test_a_cut_notice_keeps_its_action_and_close_clear_and_operable(toast_ui, tmp_path):
    page = _open(toast_ui, *PHONE)
    name = f'toast-action-{toast_ui["engine"]}'
    facts = {}
    try:
        page.evaluate("""async () => {
            const { showToast } = await import('/static/modules/toast.js');
            window.started = 0;
            const text = 'The saved work can be started again. '.repeat(46);
            showToast(text, 'warn', { ttl: 0, action: { label: 'Start it', run() { window.started += 1; } } });
        }""")
        rest(page)
        for state, y, edge in (('start', 0, BELOW), ('end', 1e5, ABOVE)):
            _scroll(page, y)
            facts[state] = view = _read(page)
            _assert_edges(view, edge)
            assert _shown(view['toasts'][0]['close'], view), view
        assert _shown(view['toasts'][0]['action'], view), view
        setup_browser.capture(page, f'{name}-end')
        page.get_by_role('button', name='Start it').click()
        page.locator(f'{STACK} .toast').wait_for(state='detached')
        assert page.evaluate('window.started') == 1
    finally:
        save_evidence(tmp_path, name, facts)
        page.close()


def _buttons(toast):
    return toast.get_by_role('button').evaluate_all('nodes => nodes.map(n => n.getAttribute("aria-label") || n.textContent)')


def test_a_timed_hurry_refusal_fades_only_where_its_words_are_cut(toast_ui, tmp_path):
    """A real task-control refusal fills the page's notice stack; a timed notice has its own close too (owner 26.11, A),
    the shared edge fades only its words, and a press on them neither hides it nor leads anywhere."""
    page = _open(toast_ui, *PHONE)
    name = f'toast-timed-{toast_ui["engine"]}'
    facts = {}
    reason = 'The task has not confirmed its current action. ' * 36 + 'Try again after confirmation.'
    try:
        page.route('**/api/tasks/timed-hurry/hurry', lambda route: route.fulfill(json={'ok': False, 'error': reason}))
        assert page.evaluate("""async () => {
            const { hurryTaskAction } = await import('/static/modules/task_control_menu.js');
            return hurryTaskAction('timed-hurry');
        }""") is False
        toast = page.locator(f'{STACK} .toast')
        rest(page)
        assert toast.inner_text() == 'Hurry up: refused — ' + reason
        assert toast.get_attribute('role') == 'alert' and _buttons(toast) == ['Dismiss notice']
        facts['cut'] = cut = _read(page)
        setup_browser.capture(page, f'{name}-cut')
        _assert_edges(cut, BELOW)
        assert cut['role'] == 'region' and cut['toasts'][0]['words']['bottom'] > cut['bottom'], cut
        assert _shown(cut['toasts'][0]['close'], cut), cut
        toast.wait_for(state='detached', timeout=8000)
        assert not page.locator(STACK).get_attribute('role'), 'expiry releases the scroll region'
        page.evaluate("""async () => (await import('/static/modules/toast.js')).showToast('Saved.', 'ok') && true""")
        rest(page)
        facts['short'] = short = _read(page)
        _assert_edges(short, CLEAR)
        assert short['role'] is None and short['scroll_end'] == 0 and _shown(short['toasts'][0]['words'], short), short
        assert _buttons(toast) == ['Dismiss notice'] and _shown(short['toasts'][0]['close'], short), short
        toast.locator('.toast-words').click()
        page.wait_for_timeout(400)  # past the 180 ms a dismissed notice takes to leave
        facts['after_words_press'] = page.evaluate("""() => [...document.querySelectorAll('#toast-stack .toast')]
            .map(el => ({ hiding: el.classList.contains('is-hiding'), text: el.textContent }))""")
        assert facts['after_words_press'] == [{'hiding': False, 'text': 'Saved.'}], 'a press on the words leaves the notice'
        toast.get_by_role('button', name='Dismiss notice').click()
        toast.wait_for(state='detached')
    finally:
        save_evidence(tmp_path, name, facts)
        page.close()


# The in-app notifications through the real notifier and the real stack: a finished task's notice (Main, an addressed
# destination) and Settings' test (no destination). Navigation is recorded, never performed.
NOTIFY = """async (taskId) => {
    const { createNotifier, DEFAULT_NOTIFY_PREFS, NOTIFY_PREFS_KEY } = await import('/static/modules/notifications.js');
    const { showToast } = await import('/static/modules/toast.js');
    const prefs = JSON.stringify({ ...DEFAULT_NOTIFY_PREFS, enabled: true, sound: false });
    window.opened = window.opened || [];
    const notifier = createNotifier({
        storage: { getItem: (key) => (key === NOTIFY_PREFS_KEY ? prefs : null), setItem() {} },
        notificationCtor: undefined, audioContextCtor: null, documentRef: null, hostApi: {}, showToast,
        onActivate: (target) => { window.opened.push(target); },
    });
    const finished = notifier.handleFrame({ role: 'system', system_type: 'task_summary', task_id: taskId },
        { kind: 'chat', isMain: true });
    const test = notifier.test();
    return [finished?.surface, test?.surface];
}"""


WORD_SPAN = """(toast, word) => {
    const text = toast.querySelector('.toast-words').firstChild;
    const range = document.createRange();
    range.setStart(text, text.data.indexOf(word));
    range.setEnd(text, text.data.indexOf(word) + word.length);
    const r = range.getBoundingClientRect();
    return { left: r.left, right: r.right, y: (r.top + r.bottom) / 2 };
}"""


def test_a_notification_opens_its_source_only_from_open_and_its_close_only_hides_it(toast_ui, tmp_path):
    page = _open(toast_ui, *WIDE)
    name = f'toast-notification-open-{toast_ui["engine"]}'
    facts = {}
    finished = page.locator(f'{STACK} .toast', has_text='Task finished')
    tested = page.locator(f'{STACK} .toast', has_text='This is a test notification.')
    opened = lambda: page.evaluate('window.opened')  # noqa: E731
    try:
        assert page.evaluate(NOTIFY, 'toast-open-a') == ['in_app', 'in_app']
        rest(page)
        facts['buttons'] = {'finished': _buttons(finished), 'test': _buttons(tested)}
        assert facts['buttons'] == {'finished': ['Open', 'Dismiss notice'], 'test': ['Dismiss notice']}, facts
        facts['view'] = _read(page)
        setup_browser.capture(page, f'{name}-shown')
        page.evaluate("document.querySelector('#chat-input').value = 'half-typed draft'")
        place = page.evaluate("document.querySelector('[id^=page-].active')?.id")
        span = finished.evaluate(WORD_SPAN, 'finished')
        page.mouse.move(span['left'] + 0.5, span['y'])
        page.mouse.down()
        page.mouse.move(span['right'] - 0.5, span['y'], steps=8)
        page.mouse.up()
        facts['selected'] = page.evaluate('window.getSelection().toString()')
        assert facts['selected'] == 'finished', ('a real mouse drag selects the words', facts)
        finished.locator('.toast-words').click()
        tested.locator('.toast-words').click()
        page.wait_for_timeout(400)
        assert finished.count() == 1 and tested.count() == 1 and opened() == [], 'a press on the words does nothing'
        finished.get_by_role('button', name='Dismiss notice').click()
        tested.get_by_role('button', name='Dismiss notice').click()
        finished.wait_for(state='detached')
        tested.wait_for(state='detached')
        assert opened() == [], 'the close only hides the notice: no navigation'
        assert page.evaluate("document.querySelector('#chat-input').value") == 'half-typed draft', 'nor a draft touched'
        assert page.evaluate("document.querySelector('[id^=page-].active')?.id") == place
        assert page.evaluate(NOTIFY, 'toast-open-b') == ['in_app', 'in_app']
        finished.get_by_role('button', name='Open').click()
        finished.wait_for(state='detached')
        facts['opened'] = opened()
        assert [target.get('isMain') for target in facts['opened']] == [True], facts
        assert tested.count() == 1, 'opening one notice leaves the others'
    finally:
        save_evidence(tmp_path, name, facts)
        page.close()


ROOM_AND_PAGE = """() => {
    const look = (el) => { const s = getComputedStyle(el); return { background: s.backgroundColor, shadow: s.boxShadow,
        border: [s.borderTopWidth, s.borderTopStyle], radius: s.borderTopLeftRadius,
        glass: s.backdropFilter || s.webkitBackdropFilter || '' }; };
    const box = (el) => { const r = el.getBoundingClientRect(); return { top: r.top, bottom: r.bottom, left: r.left, right: r.right }; };
    const room = document.querySelector('#page-chat > .toast-stack--room .toast');
    return { room: look(room), page: look(document.querySelector('#toast-stack .toast')), box: box(room),
        chat: box(document.querySelector('#page-chat')), header: box(document.querySelector('#page-chat .chat-page-header')),
        composer: box(document.querySelector('#page-chat #chat-input-area')),
        buttons: [...room.querySelectorAll('button')].map((b) => b.getAttribute('aria-label') || b.textContent) };
}"""


def test_a_room_notice_keeps_its_place_and_wears_the_same_material_and_close(toast_ui, tmp_path):
    """A room's notice (toast.js `host`) stays above its composer, never in the page's corner, in the same float
    material with the same close as a page notice."""
    page = _open(toast_ui, *WIDE)
    name = f'toast-room-{toast_ui["engine"]}'
    facts = {}
    try:
        page.evaluate("""async () => {
            const { showToast } = await import('/static/modules/toast.js');
            showToast('A room notice keeps to the reading area.', 'info',
                { ttl: 0, haptic: false, host: document.querySelector('#page-chat') });
            showToast('A page notice.', 'info', { ttl: 0, haptic: false });
        }""")
        rest(page)
        facts = page.evaluate(ROOM_AND_PAGE)
        setup_browser.capture(page, name)
        assert facts['room'] == facts['page'] and facts['room']['shadow'] != 'none', facts
        assert facts['room']['border'][0] == '0px' or facts['room']['border'][1] == 'none', facts
        assert facts['buttons'] == ['Dismiss notice'], facts
        assert facts['header']['bottom'] < facts['box']['top'] and facts['box']['bottom'] <= facts['composer']['top'], facts
        middle = lambda b: (b['left'] + b['right']) / 2  # noqa: E731
        assert abs(middle(facts['box']) - middle(facts['chat'])) < 2, ('centred in its room, not in the corner', facts)
        room = page.locator('#page-chat > .toast-stack--room .toast')
        room.locator('.toast-words').click()
        page.wait_for_timeout(400)
        assert room.count() == 1, 'a press on the words leaves the notice'
        room.get_by_role('button', name='Dismiss notice').click()
        room.wait_for(state='detached')
        assert page.locator(f'{STACK} .toast').count() == 1, 'closing the room notice leaves the page notice'
    finally:
        save_evidence(tmp_path, name, facts)
        page.close()


def test_a_toast_close_is_quiet_and_a_finger_reaches_it(toast_ui, tmp_path):
    """Every toast's × is the quiet icon control (no body, no edge on the notice); under a finger its reach is the
    44 px touch minimum and stays inside the notice, so the page column never clips it."""
    page = _open(toast_ui, 390, 844, touch=True)
    name = f'toast-close-reach-{toast_ui["engine"]}'
    facts = {}
    try:
        page.evaluate("""async () => {
            const { showToast } = await import('/static/modules/toast.js');
            showToast('Copied.', 'ok', { ttl: 0 });
        }""")
        rest(page)
        facts = page.locator(f'{STACK} .toast').evaluate("""toast => {
            const close = toast.querySelector('.toast-dismiss');
            const box = (el) => { const r = el.getBoundingClientRect(); return { top: r.top, bottom: r.bottom, left: r.left, right: r.right }; };
            const reach = getComputedStyle(close, '::after');
            const c = box(close), t = box(toast);
            const s = getComputedStyle(toast);
            const hit = (x, y) => document.elementFromPoint(x, y)?.closest('.toast-dismiss') === close;
            const half = parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--touch-target-min')) / 2;
            const cx = (c.left + c.right) / 2, cy = (c.top + c.bottom) / 2;
            return { coarse: matchMedia('(pointer: coarse)').matches, close: c, toast: t, half,
                reach: { top: reach.top, left: reach.left, content: reach.content },
                hits: [hit(cx, cy - half + 1), hit(cx, cy + half - 1), hit(cx - half + 1, cy), hit(cx + half - 1, cy)],
                border: [s.borderTopWidth, s.borderTopStyle], shadow: s.boxShadow, closeBackground: getComputedStyle(close).backgroundColor,
                closeShadow: getComputedStyle(close).boxShadow };
        }""")
        setup_browser.capture(page, f'{name}-phone')
        assert facts['coarse'] and facts['half'] == 22, facts
        assert facts['hits'] == [True, True, True, True], ('a finger reaches the × 22 px from its centre each way', facts)
        assert facts['close']['top'] - 8 >= facts['toast']['top'] - 0.5, facts
        assert facts['close']['right'] + 8 <= facts['toast']['right'] + 0.5, facts
        assert facts['border'][0] == '0px' or facts['border'][1] == 'none', facts
        assert facts['shadow'] != 'none', facts
        assert facts['closeBackground'] == 'rgba(0, 0, 0, 0)' and facts['closeShadow'] == 'none', facts
        page.get_by_role('button', name='Dismiss notice').tap()
        page.locator(f'{STACK} .toast').wait_for(state='detached')
        # The live flow: a finger copies the address in Settings, and a tap on the notice's last glyph is a press on its
        # words: the × reach (8 px past its 28 px) and the engine's snap of a near tap onto a button both end in the gap.
        facts['live'] = _tap_last_glyph_then_the_reach(page, toast_ui['settings'])
    finally:
        save_evidence(tmp_path, name, facts)
        page.close()


# What happened to the one global notice, in order: a click on a close, then the first moment it carries `is-hiding`.
TRACE_CLOSE = """() => {
    window.__closeTrace = [];
    const now = () => performance.now();
    document.addEventListener('click', (event) => {
        if (event.target.closest?.('.toast-dismiss')) window.__closeTrace.push(['click', now()]);
    }, true);
    new MutationObserver(() => {
        if (document.querySelector('#toast-stack .toast.is-hiding') && !window.__closeTrace.some(([kind]) => kind === 'hiding')) {
            window.__closeTrace.push(['hiding', now()]);
        }
    }).observe(document.querySelector('#toast-stack'), { subtree: true, attributes: true, attributeFilter: ['class'] });
}"""

LAST_GLYPH = """toast => {
    const words = toast.querySelector('.toast-words');
    const text = words.firstChild;
    const range = document.createRange();
    range.setStart(text, text.length - 1);
    range.setEnd(text, text.length);
    const g = range.getBoundingClientRect();
    const c = toast.querySelector('.toast-dismiss').getBoundingClientRect();
    const x = (g.left + g.right) / 2, y = (g.top + g.bottom) / 2;
    const at = document.elementFromPoint(x, y);
    return { text: text.data, glyph: { left: g.left, right: g.right, top: g.top, bottom: g.bottom }, x, y,
        under: at?.closest('.toast-dismiss') ? 'close' : at?.closest('.toast-words') ? 'words' : at?.className,
        close: { left: c.left, right: c.right, top: c.top, bottom: c.bottom }, gap: c.left - g.right };
}"""


def _tap_last_glyph_then_the_reach(page, settings):
    settings['_meta'].update(reachability='lan_reachable', recommended_url='http://192.0.2.44:8765', warning='')
    page.evaluate(INSTALL)
    open_settings(page, 'computer')
    page.locator('#page-settings [data-copy-lan-url]').tap()
    toast = page.locator(f'{STACK} .toast', has_text='LAN URL copied.')
    toast.wait_for(state='visible')
    rest(page)
    facts = toast.evaluate(LAST_GLYPH)
    assert recorded(page)['writes'] == ['http://192.0.2.44:8765'], facts
    assert facts['text'] == 'LAN URL copied.' and facts['under'] == 'words', ('the last glyph is words, not the ×', facts)
    page.touchscreen.tap(facts['x'], facts['y'])
    page.wait_for_timeout(400)  # past the 180 ms a dismissed notice takes to leave
    facts['after_glyph'] = page.evaluate("""() => [...document.querySelectorAll('#toast-stack .toast')]
        .map(el => ({ hiding: el.classList.contains('is-hiding'), text: el.textContent }))""")
    assert facts['after_glyph'] == [{'hiding': False, 'text': 'LAN URL copied.'}], ('a tap on the last glyph keeps it', facts)
    c = facts['close']
    # The × is reached from its own edge (22 px from its centre, the near side included), and that hides the notice.
    # The notice is timed (6 s), so detaching alone could be its natural expiry: the page records the × click and the
    # moment the notice starts leaving, and only a click that came first, in the same turn, is the press's doing.
    page.evaluate(TRACE_CLOSE)
    page.touchscreen.tap((c['left'] + c['right']) / 2 - 21, (c['top'] + c['bottom']) / 2)
    toast.wait_for(state='detached')
    facts['trace'] = trace = page.evaluate('window.__closeTrace')
    assert [kind for kind, _ in trace] == ['click', 'hiding'], ('the × press hid it, not the timer', facts)
    assert trace[1][1] - trace[0][1] < 50, ('hidden by that very click', facts)
    return facts


def test_resizing_a_kept_notice_moves_the_fade_with_the_cut_words_below(toast_ui, tmp_path):
    """When the first of two kept global notices grows in a capped column, the second notice's
    cut words shift while stack height stays fixed; their painted mask end must follow the real cut."""
    page = _open(toast_ui, *PHONE)
    name = f'toast-sibling-resize-{toast_ui["engine"]}'
    facts = {}
    short_first = 'First kept notice.'
    longer_first = (
        'First kept notice translated into longer prose that wraps onto several extra lines '
        'and pushes the notice below it further past the edge.'
    )
    long_second = 'Second kept notice continues past the bottom of the column. ' * 14
    try:
        page.evaluate("""async ([firstText, secondText]) => {
            const { showToast } = await import('/static/modules/toast.js');
            showToast(firstText, 'warn', { ttl: 0 });
            showToast(secondText, 'warn', { ttl: 0 });
        }""", [short_first, long_second])
        rest(page)
        facts['before'] = before = _read(page)
        setup_browser.capture(page, f'{name}-before')
        assert before['scroll_end'] > 0, before
        _assert_edges(before, CLEAR, BELOW)
        before_second = before['toasts'][1]['words']
        assert before_second['mask_end'] is not None, before
        assert abs(before_second['mask_end'] - before_second['measured_cut']) <= 1.0, before

        page.evaluate("""text => {
            document.querySelector('#toast-stack .toast .toast-words').textContent = text;
        }""", longer_first)
        rest(page)
        facts['after'] = after = _read(page)
        setup_browser.capture(page, f'{name}-after')
        assert abs(after['height'] - before['height']) < 0.5 and after['client_height'] == before['client_height'], facts
        after_second = after['toasts'][1]['words']
        shift = after_second['top'] - before_second['top']
        facts['shift'] = shift
        assert shift > 32, ('first notice growth must shift the second notice past the 32px fade', facts)
        _assert_edges(after, CLEAR, BELOW)
        assert after_second['mask_end'] is not None, facts
        assert abs(after_second['mask_end'] - after_second['measured_cut']) <= 1.0, (
            'relative painted mask end must match the measured cut after sibling resize', facts,
        )

        page.evaluate("""text => {
            document.querySelector('#toast-stack .toast .toast-words').textContent = text;
        }""", short_first)
        rest(page)
        facts['restored'] = restored = _read(page)
        assert abs(restored['height'] - before['height']) < 0.5, facts
        restored_second = restored['toasts'][1]['words']
        assert abs(restored_second['top'] - before_second['top']) <= 1.0, facts
        _assert_edges(restored, CLEAR, BELOW)
        assert restored_second['mask_end'] is not None, facts
        assert abs(restored_second['mask_end'] - restored_second['measured_cut']) <= 1.0, facts
    finally:
        save_evidence(tmp_path, name, facts)
        page.close()


# Under a finger the close's reach (the touch minimum around its small ×) and the action's never overlap: a press
# anywhere on the visible action, its corners included, runs the action, and the × still takes a press at its centre.
CORNERS = """() => {
    const toast = [...document.querySelectorAll('#toast-stack .toast')].at(-1);
    const action = toast.querySelector('.toast-action'), close = toast.querySelector('.toast-dismiss');
    const a = action.getBoundingClientRect(), c = close.getBoundingClientRect();
    const first = (x, y) => document.elementsFromPoint(x, y)[0];
    const points = [[a.left + 2, a.top + 2], [a.right - 2, a.top + 2], [a.left + 2, a.bottom - 2], [a.right - 2, a.bottom - 2]];
    return {corners: points.map(([x, y]) => action.contains(first(x, y))),
            close: close.contains(first(c.left + c.width / 2, c.top + c.height / 2)),
            coarse: matchMedia('(pointer: coarse)').matches};
}"""


@pytest.mark.parametrize('width', [320, 390])
@pytest.mark.parametrize('words', ['Module failed.', 'A provider refused the request, so the review stays open.'],
                         ids=['short', 'long'])
def test_a_touch_on_the_actions_corner_runs_the_action_never_the_close(toast_ui, width, words):
    page = _open(toast_ui, width, 844, touch=True)
    try:
        page.evaluate("""async (words) => {
            const { showToast } = await import('/static/modules/toast.js');
            showToast(words, 'danger', { ttl: 0, haptic: false, action: { label: 'Open Settings', run() {} } });
        }""", words)
        page.wait_for_timeout(300)
        facts = page.evaluate(CORNERS)
        assert facts['coarse'], facts
        assert all(facts['corners']) and facts['close'], facts
    finally:
        page.close()


# A tap on the notice's last wrapped line above its action (a notification's Open, a recovery's Start it) is a press on
# the words: the action's reach (8 px past its 28 px under a finger) and the engine's snap of a near tap onto a button
# must both end short of that line (owner 26.11, A: the words do nothing). The taps are the glyphs of that line which
# stand over the action, at their middle and just above their foot.
LAST_LINE = """toast => {
    const words = toast.querySelector('.toast-words');
    const text = words.firstChild;
    const glyphs = [...text.data].map((char, i) => {
        const range = document.createRange();
        range.setStart(text, i);
        range.setEnd(text, i + 1);
        return { char, rect: range.getBoundingClientRect() };
    }).filter(({ rect }) => rect.width > 0);
    const foot = Math.max(...glyphs.map(({ rect }) => rect.bottom));
    const line = glyphs.filter(({ rect }) => rect.bottom > foot - 4);
    const a = toast.querySelector('.toast-action').getBoundingClientRect();
    const over = line.filter(({ char, rect }) => char !== ' ' && rect.left + rect.width / 2 > a.left
        && rect.left + rect.width / 2 < a.right);
    const taps = [over[0], over[Math.floor(over.length / 2)], over.at(-1)].flatMap(({ char, rect }) => [
        { char, x: rect.left + rect.width / 2, y: (rect.top + rect.bottom) / 2 },
        { char, x: rect.left + rect.width / 2, y: rect.bottom - 2 }]);
    return { taps, line: line.map(({ char }) => char).join(''),
        onWords: taps.map(({ x, y }) => document.elementFromPoint(x, y)?.closest('.toast-words') === words),
        wrapped: words.getBoundingClientRect().height > 40, gap: a.top - foot };
}"""


@pytest.mark.parametrize('label', ['Open', 'Start it'])
def test_a_tap_on_the_last_line_above_an_action_never_runs_it(toast_ui, tmp_path, label):
    page = _open(toast_ui, 390, 844, touch=True)
    name = f'toast-last-line-{label.replace(" ", "-").lower()}-{toast_ui["engine"]}'
    facts = {}
    show = """async ([text, label]) => {
        const { showToast } = await import('/static/modules/toast.js');
        showToast(text, 'ok', { ttl: 0, action: { label, run() { window.ran += 1; } } });
    }"""
    state = """() => ({ ran: window.ran, toasts: [...document.querySelectorAll('#toast-stack .toast')]
        .map(el => el.classList.contains('is-hiding')) })"""
    toast = page.locator(f'{STACK} .toast')
    try:
        page.evaluate('window.ran = 0')
        page.evaluate(show, ['The report is ready. Should we publish today?', label])
        rest(page)
        facts = toast.evaluate(LAST_LINE)
        setup_browser.capture(page, name)
        assert facts['wrapped'] and all(facts['onWords']), ('every tap point is on the last line of the words', facts)
        for tap in facts['taps']:
            page.touchscreen.tap(tap['x'], tap['y'])
            page.wait_for_timeout(300)  # past the 180 ms a dismissed notice takes to leave
            assert page.evaluate(state) == {'ran': 0, 'toasts': [False]}, ('a tap on the last line did nothing', tap, facts)
        toast.get_by_role('button', name='Dismiss notice').tap()
        toast.wait_for(state='detached')
        assert page.evaluate('window.ran') == 0, 'the × runs nothing'
        page.evaluate(show, ['A second notice.', label])
        rest(page)
        toast.get_by_role('button', name=label, exact=True).tap()
        toast.wait_for(state='detached')
        assert page.evaluate('window.ran') == 1, 'the action itself still runs, and hides the notice'
    finally:
        save_evidence(tmp_path, name, facts)
        page.close()


# A modal makes the page behind it inert: what an action of the photo reader or the small file dialog answers is shown
# inside that modal (chat_media.js, document_reader.js), with the shared close. Nothing here reaches the machine: the
# clipboard is the inert sink (ui_clipboard_sink.py, plus the image `write`) and the desktop file bridge is a stand-in.
MODAL_SETUP = """async () => {
    window.__sink.exec = false;  // the selection copy says no too: only the stand-in answers
    window.clip = { mode: 'ok', link: false, links: [], release: null, error: 'Permission denied' };
    Object.defineProperty(navigator, 'clipboard', { configurable: true, get: () => ({
        write: () => {
            if (window.clip.mode === 'reject') return Promise.reject(new Error(window.clip.error));
            if (window.clip.mode === 'hold') return new Promise((resolve, reject) => { window.clip.release = { resolve, reject }; });
            return Promise.resolve();
        },
        writeText: async (text) => {
            if (!window.clip.link) throw new Error('refused');
            window.clip.links.push(text);
        },
    }) });
    window.bridge = { hold: false, release: null };
    const answer = () => (window.bridge.hold
        ? new Promise((resolve) => { window.bridge.release = resolve; })
        : Promise.resolve({ ok: false, error: 'The desktop refused' }));
    window.pywebview = { api: { open_file_with_default_app: answer, download_file_to_downloads: answer } };
    const { createChatMedia } = await import('/static/modules/chat_media.js');
    const { taskArtifactDownloadUrl } = await import('/static/modules/api_client.js');
    const media = createChatMedia({ chatSessionId: 'modal-notice', durableChatMediaUrl: (s) => s || '',
        formatMsgTime: () => null, senderLabel: () => 'Owner', stampNodeTimestamp: () => {} });
    const root = document.querySelector('#chat-messages');
    const photo = document.createElement('canvas');
    Object.assign(photo, { width: 640, height: 420 });
    const paint = photo.getContext('2d');
    ['#7aa7d9', '#e7b3bd', '#f2d58c', '#8fc7a1'].forEach((colour, i) => {
        paint.fillStyle = colour;
        paint.fillRect((i % 2) * 320, Math.floor(i / 2) * 210, 320, 210);
    });
    root.append(media.buildMediaBubble({ type: 'photo', role: 'assistant', task_id: 't', mime: 'image/png',
        caption: 'Modal photo', image_base64: photo.toDataURL('image/png').split(',')[1],
        download_url: taskArtifactDownloadUrl('modal-task', 'photo.png') }));
    root.append(media.buildDocumentBubble({ type: 'document', role: 'assistant', task_id: 'modal-task',
        filename: 'archive.zip', mime: 'application/zip',
        download_url: taskArtifactDownloadUrl('modal-task', 'archive.zip') }));
    root.lastElementChild.scrollIntoView({ block: 'end' });
    window.media = media;
}"""
MODAL_FACTS = """() => {
    const dialog = document.querySelector('dialog[open]');
    const toast = dialog?.querySelector(':scope > .toast-stack--room .toast');
    const close = toast?.querySelector('.toast-dismiss');
    const box = (el) => { const r = el.getBoundingClientRect(); return { top: r.top, bottom: r.bottom, left: r.left, right: r.right }; };
    const hit = (el) => { const r = el.getBoundingClientRect();
        return el.contains(document.elementFromPoint((r.left + r.right) / 2, (r.top + r.bottom) / 2)); };
    const buttons = [...(dialog?.querySelectorAll('[data-file-action], [data-reader-action]') || [])]
        .filter((el) => el.getClientRects().length);
    return { dialog: dialog?.className || null, text: toast?.querySelector('.toast-words')?.textContent ?? null,
        page: [...document.querySelectorAll('#toast-stack .toast .toast-words')].map((el) => el.textContent),
        closeHit: Boolean(close) && hit(close), covered: buttons.filter((el) => !hit(el)).map((el) => el.textContent.trim()),
        toast: toast ? box(toast) : null, view: [innerWidth, innerHeight] };
}"""
NOTICES = "[...document.querySelectorAll('.toast:not(.is-hiding)')].length"
# A function, never the bare name: Playwright calls an expression that evaluates to a function, which would itself
# answer the held desktop bridge.
HELD = "() => typeof window.bridge.release === 'function'"


def _modal_page(ui, phone, height=None):
    page = _open(ui, *((390, height or 844) if phone else WIDE), touch=phone)
    page.evaluate(INSTALL)
    page.evaluate(MODAL_SETUP)
    rest(page)
    return page


def _press_where_it_is_seen(page, phone):
    """A real press at the notice's visible ×: whatever is on top there takes it, as for the owner."""
    page.locator('.toast').wait_for()
    rest(page)
    box = page.locator('.toast .toast-dismiss').bounding_box()
    x, y = box['x'] + box['width'] / 2, box['y'] + box['height'] / 2
    page.touchscreen.tap(x, y) if phone else page.mouse.click(x, y)
    page.wait_for_timeout(400)


@pytest.mark.parametrize('phone', [False, True], ids=['desktop', 'phone'])
def test_the_photo_readers_copy_answers_inside_it_and_its_close_only_hides_the_notice(toast_ui, tmp_path, phone):
    page = _modal_page(toast_ui, phone)
    name = f'toast-reader-copy-{"phone" if phone else "desktop"}-{toast_ui["engine"]}'
    facts = {}
    press = (lambda loc: loc.tap()) if phone else (lambda loc: loc.click())
    tile = page.locator('#chat-messages .chat-photo-open')
    dialog = page.locator('dialog.document-reader[open]')
    notice = dialog.locator('.toast')
    copy = dialog.locator('[data-reader-action="copy"]')

    def open_reader():
        press(tile)
        dialog.wait_for()
        page.wait_for_function("() => document.querySelector('dialog.document-reader .document-reader-image')?.naturalWidth > 0")

    try:
        # The failure, and the owner's press on the × they see.
        open_reader()
        page.evaluate("window.clip.mode = 'reject'")
        press(copy)
        page.locator('.toast').wait_for()
        rest(page)
        facts['error'] = page.evaluate(MODAL_FACTS)
        setup_browser.capture(page, f'{name}-error')
        _press_where_it_is_seen(page, phone)
        facts['after_close'] = {'reader': dialog.count(), 'notices': page.evaluate(NOTICES)}
        assert facts['after_close'] == {'reader': 1, 'notices': 0}, ('the × hides the notice, the photo stays', facts)
        assert facts['error']['text'] == 'Could not copy image: Permission denied' and facts['error']['page'] == [], facts
        assert facts['error']['closeHit'] and facts['error']['covered'] == [], facts
        # The success, its × reached and pressed from the keyboard; Escape then closes and returns to the photo.
        page.evaluate("window.clip.mode = 'ok'")
        press(copy)
        notice.wait_for()
        facts['success'] = page.evaluate(MODAL_FACTS)
        assert facts['success']['text'] == 'Image copied.' and facts['success']['page'] == [], facts
        tab = 'Alt+Tab' if toast_ui['engine'] == 'webkit' else 'Tab'  # WebKit's plain Tab skips buttons
        copy.focus()
        for _ in range(12):
            if page.evaluate("document.activeElement?.classList.contains('toast-dismiss')"):
                break
            page.keyboard.press(tab)
        assert page.evaluate("document.activeElement?.classList.contains('toast-dismiss')"), 'the × is reached by Tab'
        page.keyboard.press('Enter')
        notice.wait_for(state='detached')
        assert dialog.count() == 1
        page.keyboard.press('Escape')
        dialog.wait_for(state='detached')
        assert page.evaluate("document.activeElement?.classList.contains('chat-photo-open')"), 'focus is back on the photo'
        # Open and Download in the desktop shell that refuses the file: the copy-link handoff is the reader's too.
        page.evaluate("window.clip.link = true")
        open_reader()
        for action in ('open', 'download'):
            press(dialog.locator(f'[data-reader-action="{action}"]'))
            notice.wait_for()
            assert page.evaluate(MODAL_FACTS)['text'] == 'Link copied — open it in your browser.'
            assert page.evaluate(MODAL_FACTS)['page'] == []
            _press_where_it_is_seen(page, phone)
            assert dialog.count() == 1 and page.evaluate(NOTICES) == 0, action
        # Closing takes the notice along.
        press(copy)
        notice.wait_for()
        page.keyboard.press('Escape')
        dialog.wait_for(state='detached')
        assert page.evaluate(NOTICES) == 0
        # A card's own menu is no modal: its answer stays the page's.
        page.evaluate("window.clip.mode = 'ok'")
        press(page.locator('#chat-messages .chat-photo-actions summary'))
        press(page.locator('.chat-photo-menu:not([hidden]) [data-photo-action="copy"]'))
        page.locator(f'{STACK} .toast', has_text='Image copied.').wait_for()
    finally:
        save_evidence(tmp_path, name, facts)
        page.close()


@pytest.mark.parametrize('action,phone', [('open', False), ('download', False), ('download', True)],
                         ids=['open-desktop', 'download-desktop', 'download-phone'])
def test_the_small_file_dialogs_failed_open_or_download_answers_inside_it(toast_ui, tmp_path, action, phone):
    page = _modal_page(toast_ui, phone)
    name = f'toast-file-dialog-{action}-{"phone" if phone else "desktop"}-{toast_ui["engine"]}'
    facts = {}
    press = (lambda loc: loc.tap()) if phone else (lambda loc: loc.click())
    card = page.locator('#chat-messages .chat-file-card')
    dialog = page.locator('dialog.chat-file-dialog[open]')
    notice = dialog.locator('.toast')
    act = dialog.locator(f'[data-file-action="{action}"]')
    close = dialog.locator('[data-file-action="close"]')
    try:
        press(card)
        dialog.wait_for()
        press(act)
        page.locator('.toast').wait_for()
        rest(page)
        facts['error'] = page.evaluate(MODAL_FACTS)
        setup_browser.capture(page, f'{name}-error')
        _press_where_it_is_seen(page, phone)
        assert dialog.count() == 1 and page.evaluate(NOTICES) == 0, ('a failure keeps the dialog; the × hides the notice', facts)
        assert facts['error']['text'] == f'Could not {action} file: The desktop refused', facts
        assert facts['error']['page'] == [] and facts['error']['closeHit'] and facts['error']['covered'] == [], facts
        # Close takes the notice along, and so does a native close (Back, layers.js), whose reopening starts clean.
        press(act)
        notice.wait_for()
        press(close)
        dialog.wait_for(state='detached')
        assert page.evaluate(NOTICES) == 0
        press(card)
        press(act)
        notice.wait_for()
        page.evaluate("document.querySelector('dialog.chat-file-dialog').close()")
        page.wait_for_function(f'() => {NOTICES} === 0')
        # The `close` event is queued, so the dialog is shut before anyone hears it. Reopened in that gap, it starts clean:
        press(card)
        press(act)
        notice.wait_for()
        page.evaluate("""() => { document.querySelector('dialog.chat-file-dialog').close();
            document.querySelector('#chat-messages .chat-file-card').click(); }""")
        page.wait_for_timeout(300)
        assert dialog.count() == 1 and page.locator('dialog .toast').count() == 0, 'the earlier opening\'s notice is gone'
        # and an answer landing in that gap is the page's, never a notice inside the shut dialog.
        page.evaluate('() => { window.bridge.hold = true; window.bridge.release = null; }')
        press(act)
        page.wait_for_function(HELD)
        page.evaluate("""() => { window.bridge.release({ ok: false, error: 'In the gap' });
            document.querySelector('dialog.chat-file-dialog').close(); }""")
        page.locator(f'{STACK} .toast', has_text=f'Could not {action} file: In the gap').wait_for()
        page.evaluate("document.querySelector('#toast-stack .toast').dispose()")
        page.evaluate('window.bridge.hold = false')
        # A held answer that outlives its opening: the same file opened again is a new dialog. A late success leaves it
        # open and a late failure is the page's (waiting there until the dialog closes).
        for ok in (True, False):
            press(card)
            dialog.wait_for()
            page.evaluate('() => { window.bridge.hold = true; window.bridge.release = null; }')
            press(act)
            page.wait_for_function(HELD)
            press(close)
            dialog.wait_for(state='detached')
            press(card)
            dialog.wait_for()
            page.evaluate("ok => window.bridge.release({ ok, error: 'Too late' })", ok)
            page.wait_for_timeout(500)
            facts[f'late-{ok}'] = late = {'dialog': dialog.count(), 'inside': page.locator('dialog .toast').count(),
                'page': page.locator(f'{STACK} .toast .toast-words').all_text_contents()}
            assert late == {'dialog': 1, 'inside': 0, 'page': [] if ok else [f'Could not {action} file: Too late']}, facts
            page.evaluate("document.querySelectorAll('#toast-stack .toast').forEach((el) => el.dispose())")
            press(close)
            dialog.wait_for(state='detached')
        # The copy-link handoff succeeds: the dialog closes and the page says so.
        page.evaluate('() => { window.bridge.hold = false; window.clip.link = true; }')
        press(card)
        press(act)
        dialog.wait_for(state='detached')
        page.locator(f'{STACK} .toast', has_text='Link copied').wait_for()
        assert len(page.evaluate('window.clip.links')) == 1
    finally:
        save_evidence(tmp_path, name, facts)
        page.close()


# Behind a modal the page is inert (toast.js): a page or room notice there is painted under the backdrop, where a press
# at its × lands on the backdrop and closes the modal, and its timer runs out unread. It waits for the modal instead.
OUTSIDE = """() => [...document.querySelectorAll('.toast')]
    .filter((el) => !el.closest('dialog') && !el.classList.contains('is-hiding')).map((el) => {
        const close = el.querySelector('.toast-dismiss');
        const r = close.getBoundingClientRect();
        const at = document.elementFromPoint((r.left + r.right) / 2, (r.top + r.bottom) / 2);
        return { text: el.querySelector('.toast-words').textContent, room: Boolean(el.closest('.toast-stack--room')),
            visible: getComputedStyle(el).visibility === 'visible', hit: close.contains(at),
            under: at?.closest('dialog')?.className || at?.id || String(at?.className) };
    })"""
SHOWN = """([text, room]) => import('/static/modules/toast.js').then(({ showToast }) => Boolean(showToast(text, 'warn',
    { ttl: 1500, haptic: false, host: room ? document.querySelector('#page-chat') : null })))"""
# A routed notification through the real notifier: its in-app notice carries Open, which is only recorded here.
ROUTED = """async (taskId) => {
    const { createNotifier, DEFAULT_NOTIFY_PREFS, NOTIFY_PREFS_KEY } = await import('/static/modules/notifications.js');
    const { showToast } = await import('/static/modules/toast.js');
    const prefs = JSON.stringify({ ...DEFAULT_NOTIFY_PREFS, enabled: true, sound: false });
    window.opened = window.opened || [];
    return createNotifier({ storage: { getItem: (key) => (key === NOTIFY_PREFS_KEY ? prefs : null), setItem() {} },
        notificationCtor: undefined, audioContextCtor: null, documentRef: null, hostApi: {}, showToast,
        onActivate: (target) => { window.opened.push(target); } })
        .handleFrame({ role: 'system', system_type: 'task_summary', task_id: taskId }, { kind: 'chat', isMain: true }).surface;
}"""
TTL_NOTICES = "() => [...document.querySelectorAll('.toast .toast-words')].filter((el) => el.textContent.startsWith('Timed')).length"
# The timers of SHOWN's 1.5 s notices that still count down (no other timer of the page has that delay).
TIMERS = """() => {
    const set = window.setTimeout.bind(window);
    const clear = window.clearTimeout.bind(window);
    window.__ttl = new Set();
    window.setTimeout = (fn, ms, ...rest) => {
        if (ms !== 1500) return set(fn, ms, ...rest);
        const id = set(() => { window.__ttl.delete(id); fn(...rest); }, ms);
        window.__ttl.add(id);
        return id;
    };
    window.clearTimeout = (id) => { window.__ttl.delete(id); return clear(id); };
}"""


def _close_reader(page, phone):
    dialog = page.locator('dialog.document-reader[open]')
    if phone:
        dialog.locator('[data-reader-action="close"]').tap()
    else:
        page.keyboard.press('Escape')
    dialog.wait_for(state='detached')


@pytest.mark.parametrize('phone', [False, True], ids=['desktop', 'phone'])
def test_page_and_room_notices_wait_out_a_modal_and_come_back_whole(toast_ui, tmp_path, phone):
    page = _modal_page(toast_ui, phone)
    name = f'toast-hold-{"phone" if phone else "desktop"}-{toast_ui["engine"]}'
    facts = {}
    press = (lambda loc: loc.tap()) if phone else (lambda loc: loc.click())
    dialog = page.locator('dialog.document-reader[open]')
    try:
        page.evaluate(TIMERS)
        page.evaluate(SHOWN, ['Timed page notice, before.', False])
        page.evaluate(SHOWN, ['Timed room notice, before.', True])
        press(page.locator('#chat-messages .chat-photo-open'))
        dialog.wait_for()
        page.evaluate(SHOWN, ['Timed page notice, during.', False])
        assert page.evaluate(ROUTED, 'held-task') == 'in_app'
        rest(page)
        facts['held'] = held = page.evaluate(OUTSIDE)
        setup_browser.capture(page, f'{name}-held')
        assert len(held) == 4 and not any(n['visible'] or n['hit'] for n in held), ('nothing to see or press', facts)
        assert dialog.locator('.toast').count() == 0, 'a page notice is never the modal\'s'
        page.wait_for_timeout(2500)  # past every 1.5 s notice's own time
        assert len(page.evaluate(OUTSIDE)) == 4, ('none ran out behind the modal', facts)
        assert page.evaluate('window.__ttl.size') == 0, ('and nothing counts down while they wait', facts)
        _close_reader(page, phone)
        rest(page)
        facts['back'] = back = page.evaluate(OUTSIDE)
        setup_browser.capture(page, f'{name}-back')
        assert len(back) == 4 and all(n['visible'] and n['hit'] for n in back), ('every notice is back, pressable', facts)
        page.wait_for_timeout(700)  # their time starts afresh when the modal goes
        assert page.evaluate(TTL_NOTICES) == 3 and page.evaluate('window.__ttl.size') == 3, facts
        page.wait_for_function(f'({TTL_NOTICES})() === 0', timeout=4000)
        routed = page.locator(f'{STACK} .toast', has_text='Task finished')
        assert routed.count() == 1 and page.evaluate('window.opened') == [], ('and nothing ran by itself', facts)
        press(routed.get_by_role('button', name='Dismiss notice'))
        routed.wait_for(state='detached')
        assert page.evaluate('window.opened') == []
    finally:
        save_evidence(tmp_path, name, facts)
        page.close()


def test_an_open_dialog_that_is_not_modal_holds_nothing(toast_ui, tmp_path):
    """`show()` and a bare `open` attribute leave the page live: notices stay in sight, pressable and run out."""
    page = _open(toast_ui, *WIDE)
    name = f'toast-plain-dialog-{toast_ui["engine"]}'
    facts = {}
    try:
        page.evaluate(SHOWN, ['Timed notice, before.', False])
        page.evaluate("""() => {
            for (const modal of ['show', 'attribute']) {
                const dialog = document.body.appendChild(document.createElement('dialog'));
                dialog.textContent = modal;
                if (modal === 'show') dialog.show(); else dialog.setAttribute('open', '');
            }
        }""")
        page.evaluate(SHOWN, ['Timed notice, after.', False])
        rest(page)
        facts['beside'] = beside = page.evaluate(OUTSIDE)
        assert page.locator('dialog[open]').count() == 2
        assert [(n['visible'], n['hit']) for n in beside] == [(True, True), (True, True)], facts
        page.wait_for_function(f'({TTL_NOTICES})() === 0', timeout=4000)
    finally:
        save_evidence(tmp_path, name, facts)
        page.close()


def test_of_two_modals_only_the_one_holding_focus_shows_its_notices(toast_ui, tmp_path):
    """No flow stacks two native modals today. If one does, the lower one's notices stay out of sight (their × would
    close the upper one), a guess is never made, and a held notice whose dialog leaves the page is still let go."""
    page = _open(toast_ui, *WIDE)
    name = f'toast-stacked-modals-{toast_ui["engine"]}'
    facts = {}
    try:
        facts = page.evaluate("""async () => {
            const { showToast } = await import('/static/modules/toast.js');
            const frames = () => new Promise((resolve) => setTimeout(resolve, 80));
            const make = (id) => {
                const dialog = document.body.appendChild(document.createElement('dialog'));
                dialog.innerHTML = `<button>${id}</button>`;
                return dialog;
            };
            const seen = (toast) => getComputedStyle(toast).visibility === 'visible';
            const lower = make('lower'), upper = make('upper');
            lower.showModal();
            const low = showToast('Lower.', 'warn', { ttl: 600, host: lower });
            upper.showModal();
            const high = showToast('Upper.', 'warn', { ttl: 0, host: upper });
            await frames();
            const both = [seen(low), seen(high)];
            await new Promise((resolve) => setTimeout(resolve, 900));  // past the lower one's own time
            const waited = low.isConnected;
            document.activeElement.blur();
            showToast('Nobody says which is on top.', 'warn', { ttl: 0 });
            await frames();
            const unknown = [seen(low), seen(high)];
            upper.close();
            await frames();
            const released = seen(low);
            upper.showModal();
            const gone = showToast('Goes with its dialog.', 'warn', { ttl: 300, host: lower });
            lower.remove();
            await new Promise((resolve) => setTimeout(resolve, 900));
            return { both, waited, unknown, released, letGo: gone.parentElement === null };
        }""")
        assert facts == {'both': [False, True], 'waited': True, 'unknown': [False, False], 'released': True,
                         'letGo': True}, facts
    finally:
        save_evidence(tmp_path, name, facts)
        page.close()


@pytest.mark.parametrize('phone', [False, True], ids=['desktop', 'phone'])
def test_a_late_answer_behind_the_next_reader_is_never_a_press_that_closes_it(toast_ui, tmp_path, phone):
    """Close A, open B: A's late Copy refusal and a routed notice are the page's. Neither is shown over B, so nothing of
    theirs can be pressed there (a press on that × would close B); both are shown once B is closed."""
    page = _modal_page(toast_ui, phone)
    name = f'toast-late-behind-next-{"phone" if phone else "desktop"}-{toast_ui["engine"]}'
    facts = {}
    press = (lambda loc: loc.tap()) if phone else (lambda loc: loc.click())
    tile = page.locator('#chat-messages .chat-photo-open')
    dialog = page.locator('dialog.document-reader[open]')
    try:
        press(tile)
        dialog.wait_for()
        page.evaluate("() => { window.clip.mode = 'hold'; window.clip.release = null; }")
        press(dialog.locator('[data-reader-action="copy"]'))
        page.wait_for_function('window.clip.release')
        _close_reader(page, phone)
        press(tile)
        dialog.wait_for()
        page.evaluate("window.clip.release.reject(new Error('Too late'))")
        assert page.evaluate(ROUTED, 'late-task') == 'in_app'
        page.wait_for_function(f"() => document.querySelectorAll('{STACK} .toast').length === 2")
        rest(page)
        facts['behind'] = behind = page.evaluate(OUTSIDE)
        setup_browser.capture(page, f'{name}-behind')
        assert [n['visible'] or n['hit'] for n in behind] == [False, False] and dialog.count() == 1, facts
        assert dialog.locator('.toast').count() == 0, 'never the next opening\'s'
        _close_reader(page, phone)
        rest(page)
        facts['after'] = after = page.evaluate(OUTSIDE)
        assert [n['visible'] and n['hit'] for n in after] == [True, True], facts
        press(page.locator(f'{STACK} .toast', has_text='Could not copy image: Too late').get_by_role('button', name='Dismiss notice'))
        page.wait_for_function(f"() => document.querySelectorAll('{STACK} .toast').length === 1")
        assert page.evaluate('window.opened') == [], facts
    finally:
        save_evidence(tmp_path, name, facts)
        page.close()


# The desktop shell as the app meets it: the bridge announced, so the app's link interceptor stands between the page and
# window.open or a download link. Inline-only media (no address) is saved as bytes through that interceptor.
SHELL = """() => {
    window.shell = { ok: true, external: false, hold: false, release: null };
    const answer = (value) => (window.shell.hold
        ? new Promise((resolve) => { window.shell.release = () => resolve(value()); })
        : Promise.resolve(value()));
    Object.assign(window.pywebview.api, {
        save_bytes_to_downloads: (name) => answer(() => (window.shell.ok
            ? { ok: true, path: `/Users/owner/Downloads/${name}` } : { ok: false, error: 'Disk full' })),
        open_external_url: () => answer(() => ({ ok: window.shell.external })),
    });
    window.dispatchEvent(new Event('pywebviewready'));
}"""
INLINE = """async () => {
    const { mountChatMarkdown } = await import('/static/modules/chat_markdown.js');
    const root = document.querySelector('#chat-messages');
    const photo = document.createElement('canvas');
    Object.assign(photo, { width: 320, height: 200 });
    photo.getContext('2d').fillRect(0, 0, 160, 100);
    root.append(window.media.buildMediaBubble({ type: 'photo', role: 'assistant', task_id: 'inline', mime: 'image/png',
        caption: 'Inline photo', image_base64: photo.toDataURL('image/png').split(',')[1] }));
    root.append(window.media.buildDocumentBubble({ type: 'document', role: 'assistant', task_id: 'inline',
        filename: 'bundle.zip', mime: 'application/zip', file_base64: btoa('inline bytes') }));
    root.append(window.media.buildDocumentBubble({ type: 'document', role: 'assistant', task_id: 'inline',
        filename: 'links.md', mime: 'text/markdown', file_base64: btoa(
            '[Example page](https://example.com/page) and [the file](/api/files/download?path=docs/a.txt)') }));
    const words = document.createElement('div');
    words.id = 'outside-words';
    mountChatMarkdown(words, '[Outside page](https://example.com/outside)');
    root.append(words);
    words.scrollIntoView({ block: 'end' });
}"""


def _inside(page):
    """The notice the open modal hosts, and the page's, as words."""
    page.wait_for_function("() => document.querySelector('.toast:not(.is-hiding)')")
    rest(page)
    return page.evaluate(MODAL_FACTS)


# Every watch of one dialog's `open` that still observes: what a press in a modal may leave behind.
WATCHES = """() => {
    const Native = window.MutationObserver;
    window.__watches = new Set();
    window.MutationObserver = class extends Native {
        observe(target, options) {
            if (target?.tagName === 'DIALOG' && String(options?.attributeFilter) === 'open') window.__watches.add(this);
            return super.observe(target, options);
        }
        disconnect() { window.__watches.delete(this); return super.disconnect(); }
    };
}"""


def _shell_page(ui):
    page = _modal_page(ui, False)
    # The fixture refuses other origins; WebKit's blob: fetch (a shell download reads its own bytes) is not one.
    page.route('**/*', lambda r: r.continue_() if r.request.url.startswith('blob:') else r.fallback())
    page.evaluate(SHELL)
    page.evaluate(INLINE)
    rest(page)
    return page


def _dismiss_inside(page, dialog):
    dialog.locator('.toast').get_by_role('button', name='Dismiss notice').click()
    page.wait_for_function(f'() => {NOTICES} === 0')
    assert dialog.count() == 1, 'the × leaves the dialog open'


def test_inline_only_open_and_download_answer_where_they_were_asked(toast_ui, tmp_path):
    page = _shell_page(toast_ui)
    name = f'toast-inline-only-{toast_ui["engine"]}'
    facts = {}
    inline = page.locator('#chat-messages .chat-gallery-item', has_text='Inline photo')
    dialog = page.locator('dialog.document-reader[open]')

    def open_reader():
        inline.locator('.chat-photo-open').click()
        dialog.wait_for()
        page.wait_for_function("() => document.querySelector('dialog.document-reader .document-reader-image')?.naturalWidth > 0")

    try:
        open_reader()
        for ok, action, words in ((True, 'open', 'Saved to Downloads: download.png'),
                                  (True, 'download', 'Saved to Downloads: image.png'),
                                  (False, 'open', 'Could not save file: Disk full'),
                                  (False, 'download', 'Could not save file: Disk full')):
            page.evaluate('ok => { window.shell.ok = ok; }', ok)
            dialog.locator(f'[data-reader-action="{action}"]').click()
            facts[f'{action}-{ok}'] = seen = _inside(page)
            assert (seen['text'], seen['page'], seen['closeHit']) == (words, [], True), facts
            _dismiss_inside(page, dialog)
        # The answer of an opening that has gone is the page's.
        page.evaluate('() => { window.shell.ok = true; window.shell.hold = true; window.shell.release = null; }')
        dialog.locator('[data-reader-action="download"]').click()
        page.wait_for_function("() => typeof window.shell.release === 'function'")
        page.keyboard.press('Escape')
        dialog.wait_for(state='detached')
        page.evaluate('() => { window.shell.hold = false; window.shell.release(); }')
        page.locator(f'{STACK} .toast', has_text='Saved to Downloads: image.png').wait_for()
        page.evaluate("document.querySelector('#toast-stack .toast').dispose()")
        # No modal: the photo's own menu answers on the page.
        inline.locator('.chat-photo-actions summary').click()
        page.locator('.chat-photo-menu:not([hidden]) [data-photo-action="open"]').click()
        page.locator(f'{STACK} .toast', has_text='Saved to Downloads: download.png').wait_for()
        page.evaluate("document.querySelector('#toast-stack .toast').dispose()")
        # The small file dialog's inline download waits for the desktop bytes owner: a refusal stays in this opening,
        # its × leaves the dialog open, and a retry of the same opening succeeds and closes only it.
        file_dialog = page.locator('dialog.chat-file-dialog[open]')
        zip_card = page.locator('#chat-messages .chat-file-card', has_text='bundle.zip')
        # The bytes owner without a save method: the refusal is this opening's own error, read and dismissed in the
        # dialog (not held on the inert page), and the dialog stays open. Restoring the method lets a later save work.
        page.evaluate('() => { window.__save = window.pywebview.api.save_bytes_to_downloads; '
                      'delete window.pywebview.api.save_bytes_to_downloads; }')
        zip_card.click()
        file_dialog.wait_for()
        file_dialog.locator('[data-file-action="download"]').click()
        file_dialog.locator('.toast').wait_for()
        facts['zip-unavailable'] = seen = _inside(page)
        assert (seen['text'], seen['page'], seen['closeHit'], file_dialog.count()) == (
            "Saving isn't available in the app — open in a browser.", [], True, 1), facts
        _dismiss_inside(page, file_dialog)
        page.evaluate('() => { window.pywebview.api.save_bytes_to_downloads = window.__save; }')
        file_dialog.locator('[data-file-action="close"]').click()
        file_dialog.wait_for(state='detached')
        page.evaluate('() => { window.shell.ok = false; window.shell.hold = true; window.shell.release = null; }')
        zip_card.click()
        file_dialog.wait_for()
        file_dialog.locator('[data-file-action="download"]').click()
        page.wait_for_function("() => typeof window.shell.release === 'function'")
        facts['zip-pending'] = {'dialog': file_dialog.count(), 'held': True}
        page.evaluate('() => { window.shell.hold = false; window.shell.release(); }')
        file_dialog.locator('.toast').wait_for()
        facts['zip-refusal'] = seen = _inside(page)
        assert (seen['text'], seen['page'], seen['closeHit'], file_dialog.count()) == (
            'Could not save file: Disk full', [], True, 1), facts
        _dismiss_inside(page, file_dialog)
        page.evaluate('() => { window.shell.ok = true; }')
        file_dialog.locator('[data-file-action="download"]').click()
        file_dialog.wait_for(state='detached')
        page.locator(f'{STACK} .toast', has_text='Saved to Downloads: bundle.zip').wait_for()
        page.evaluate("document.querySelectorAll('#toast-stack .toast').forEach((el) => el.dispose())")
        # A late success of a closed opening never closes the next opening of the same file.
        page.evaluate('() => { window.shell.hold = true; window.shell.release = null; }')
        zip_card.click()
        file_dialog.wait_for()
        file_dialog.locator('[data-file-action="download"]').click()
        page.wait_for_function("() => typeof window.shell.release === 'function'")
        page.keyboard.press('Escape')
        file_dialog.wait_for(state='detached')
        zip_card.click()
        file_dialog.wait_for()
        page.evaluate('() => { window.shell.hold = false; window.shell.release(); }')
        page.wait_for_timeout(300)
        facts['zip-late'] = {'dialog': file_dialog.count(), 'inside': file_dialog.locator('.toast').count()}
        assert facts['zip-late'] == {'dialog': 1, 'inside': 0}, facts
    finally:
        save_evidence(tmp_path, name, facts)
        page.close()


def test_a_link_pressed_in_a_modal_answers_in_that_modal(toast_ui, tmp_path):
    """A refused external page or file link copies its address and says so where it was pressed: in the reader, in the
    skill sheet (its own link opener too), or, for an opening that has gone, on the page."""
    page = _shell_page(toast_ui)
    name = f'toast-modal-links-{toast_ui["engine"]}'
    facts = {}
    page.evaluate('() => { window.clip.link = true; }')
    card = page.locator('#chat-messages .chat-file-card', has_text='links.md')
    dialog = page.locator('dialog[open]')
    copied = 'Link copied — open it in your browser.'

    def press_link(label):
        dialog.get_by_role('link', name=label).first.click()
        seen = _inside(page)
        facts[label] = seen
        assert (seen['text'], seen['page'], seen['closeHit']) == (copied, [], True), facts
        _dismiss_inside(page, dialog)

    try:
        page.evaluate(WATCHES)
        card.click()
        dialog.wait_for()
        for label in ('Example page', 'the file'):
            press_link(label)
        # Links that open say nothing; however many presses, one opening is watched once.
        page.evaluate('() => { window.shell.external = true; }')
        for _ in range(3):
            dialog.get_by_role('link', name='Example page').click()
        page.wait_for_timeout(200)
        facts['quiet'] = {'notices': page.evaluate(NOTICES), 'watches': page.evaluate('window.__watches.size')}
        assert facts['quiet'] == {'notices': 0, 'watches': 1}, facts
        page.evaluate('() => { window.shell.external = false; }')
        # Held until the reader was closed and opened again: the page's, never the next opening's.
        page.evaluate('() => { window.shell.hold = true; window.shell.release = null; }')
        dialog.get_by_role('link', name='Example page').click()
        page.wait_for_function("() => typeof window.shell.release === 'function'")
        page.keyboard.press('Escape')
        dialog.wait_for(state='detached')
        card.click()
        dialog.wait_for()
        assert page.evaluate('window.__watches.size') == 0, 'the closed opening is watched no longer'
        page.evaluate('() => { window.shell.hold = false; window.shell.release(); }')
        page.wait_for_function(f"() => document.querySelector('{STACK} .toast')")
        assert dialog.locator('.toast').count() == 0, 'never the next opening'
        page.keyboard.press('Escape')
        dialog.wait_for(state='detached')
        page.locator(f'{STACK} .toast', has_text=copied).wait_for(state='visible')
        page.evaluate("document.querySelector('#toast-stack .toast').dispose()")
        # The skill sheet: its own link opener (data-skill-external) and a link of its description.
        page.evaluate("""async () => {
            const { openSkillDetailSheet } = await import('/static/modules/skill_detail_sheet.js');
            openSkillDetailSheet({ item: { id: 'x/demo', name: 'demo', title: 'Demo skill', source_id: 'x',
                source_name: 'Example source', description: 'Read [its notes](https://example.com/notes).',
                author_repo: 'https://github.com/example/demo', category: [] } });
        }""")
        dialog.wait_for()
        press_link('example/demo ↗')
        press_link('its notes')
        assert page.evaluate('window.__watches.size') == 1
        # Closed with a notice still showing, it leaves with the sheet at once, not at its own time in a detached tree.
        dialog.get_by_role('link', name='its notes').click()
        _inside(page)
        page.evaluate("() => { window.__shown = document.querySelector('dialog[open] .toast'); }")
        dialog.get_by_role('button', name='Close').click()
        dialog.wait_for(state='detached')
        assert page.evaluate('window.__shown.parentNode === null'), 'the notice left with its sheet'
        assert page.evaluate('window.__watches.size') == 0
        # Outside a modal the page says it, as before.
        page.locator('#outside-words a').click()
        page.locator(f'{STACK} .toast', has_text=copied).wait_for(state='visible')
    finally:
        save_evidence(tmp_path, name, facts)
        page.close()


# --- What blocks the page without being a <dialog>: the confirmation and the phone's Model sheet (owner 26.11, A) ---------
# Their backdrop and scrim lie over the page's stack (computed z-index 1100 and 1199 against the stack's 120), so a press at
# a page notice's × lands on them: it cancels the question or closes the sheet, and the notice's time runs out unread.
# They hold the page's notices as a modal dialog does; what a press inside the sheet answers stands in the sheet.
CONFIRM = """async () => {
    const { openConfirmDialog } = await import('/static/modules/confirm_dialog.js');
    window.confirmed = null;
    openConfirmDialog({ title: 'Forget this local answer?', body: 'It leaves this browser only.', confirmLabel: 'Forget' })
        .then((value) => { window.confirmed = value; });
}"""
# Every notice there is, where it stands, whether it can be seen and whether a press at its × reaches it.
TOASTS = """() => [...document.querySelectorAll('.toast:not(.is-hiding)')].map((toast) => {
    const close = toast.querySelector('.toast-dismiss');
    const r = close.getBoundingClientRect();
    const at = document.elementFromPoint((r.left + r.right) / 2, (r.top + r.bottom) / 2);
    const sheet = toast.closest('.model-sheet');
    const box = sheet?.getBoundingClientRect();
    return { text: toast.querySelector('.toast-words').textContent,
        where: toast.closest('#toast-stack') ? 'page' : sheet ? 'sheet' : toast.closest('dialog') ? 'dialog' : 'room',
        visible: getComputedStyle(toast).visibility === 'visible', hit: close.contains(at),
        under: at ? (at.id || String(at.className).slice(0, 48)) : null, box: [r.left, r.top, r.right, r.bottom].map(Math.round),
        inSheet: box ? r.left >= box.left && r.right <= box.right && r.top >= box.top && r.bottom <= box.bottom : null };
})"""
# A notice's own six seconds are stretched where a case reads it across slow steps (a screenshot on a loaded machine): the
# case is about where a notice stands and what its × does, not how long it lasts.
STRETCH = "() => { const set = window.setTimeout.bind(window); window.setTimeout = (fn, ms, ...rest) => set(fn, ms === 6000 ? 120000 : ms, ...rest); }"
KEPT = """(text) => import('/static/modules/toast.js').then(({ showToast }) => Boolean(showToast(text, 'warn', { ttl: 0, haptic: false })))"""
SHEET = """() => ({ sheet: Boolean(document.querySelector('.ui-popup.model-sheet')), scrim: Boolean(document.querySelector('.model-sheet-scrim')),
    bottom: document.querySelector('.ui-popup.model-sheet')?.dataset.sheet || null })"""
SHEET_GONE = "() => !document.querySelector('.ui-popup.model-sheet')"
CONFIRM_OPEN = '.confirm-dialog'


def _press_at(page, box, phone):
    x, y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    page.touchscreen.tap(x, y) if phone else page.mouse.click(x, y)
    page.wait_for_timeout(300)


@pytest.mark.parametrize('phone', [False, True], ids=['desktop', 'phone'])
def test_a_page_notice_waits_out_a_confirmation_instead_of_cancelling_it(toast_ui, tmp_path, phone):
    page = _open(toast_ui, *((390, 844) if phone else WIDE), touch=phone)
    name = f'toast-confirm-{"phone" if phone else "desktop"}-{toast_ui["engine"]}'
    facts = {}
    press = (lambda loc: loc.tap()) if phone else (lambda loc: loc.click())
    try:
        page.evaluate(TIMERS)
        page.evaluate(SHOWN, ['Timed page notice, before.', False])
        page.evaluate(CONFIRM)
        page.locator(CONFIRM_OPEN).wait_for()
        rest(page)
        facts['held'] = held = page.evaluate(OUTSIDE)
        setup_browser.capture(page, f'{name}-held')
        assert len(held) == 1 and not (held[0]['visible'] or held[0]['hit']), ('no × to see or press over the question', facts)
        page.wait_for_timeout(2300)  # past its own 1.5 s
        assert len(page.evaluate(OUTSIDE)) == 1 and page.evaluate('window.__ttl.size') == 0, ('its time waits too', facts)
        assert page.evaluate('window.confirmed') is None, 'the question is still asked'
        press(page.locator(f'{CONFIRM_OPEN} [data-confirm-cancel]'))
        page.wait_for_function('() => window.confirmed === false')
        rest(page)
        facts['back'] = back = page.evaluate(OUTSIDE)
        counting = page.evaluate('window.__ttl.size')  # read before the screenshot: it has only 1.5 s
        setup_browser.capture(page, f'{name}-back')
        assert len(back) == 1 and back[0]['visible'] and back[0]['hit'], ('back, whole, pressable', facts)
        assert counting == 1, ('its time starts afresh when the question goes', facts)
        press(page.locator(f'{STACK} .toast').get_by_role('button', name='Dismiss notice'))
        page.wait_for_function(f'() => {NOTICES} === 0')
    finally:
        save_evidence(tmp_path, name, facts)
        page.close()


def test_a_hidden_peer_backdrop_keeps_holding_page_notices_through_its_confirmation(toast_ui, tmp_path):
    page = _open(toast_ui, *WIDE)
    name = f'toast-peer-hidden-{toast_ui["engine"]}'
    facts = {}
    try:
        page.evaluate(TIMERS)
        page.evaluate(SHOWN, ['Timed page notice, before.', False])
        page.evaluate("""async () => {
            const { openIntroduction } = await import('/static/modules/peer_intro.js');
            window.__revoked = 0;
            openIntroduction({ link: { linkId: 'l1', placeId: 'p1', code: '123456', name: 'Peer', seed: 'peer' },
                api: { revokeLink: async () => { window.__revoked += 1; return { revokedAt: 'now' }; },
                    readLink: async () => null }, onChanged: () => {} });
        }""")
        page.locator('.peer-intro-backdrop').wait_for()
        rest(page)
        facts['opened'] = page.evaluate(OUTSIDE)
        assert len(facts['opened']) == 1 and not (facts['opened'][0]['visible'] or facts['opened'][0]['hit']), facts
        page.locator('[data-intro-differ]').click()
        page.locator(CONFIRM_OPEN).wait_for()
        page.locator('[data-confirm-cancel]').click()
        page.locator(CONFIRM_OPEN).wait_for(state='detached')
        page.wait_for_timeout(100)
        facts['back'] = back = page.evaluate(OUTSIDE)
        assert len(back) == 1 and not (back[0]['visible'] or back[0]['hit']), ('still held after Back', facts)
        assert page.evaluate('window.__ttl.size') == 0 and page.evaluate('window.__revoked') == 0, facts
        assert page.locator('.peer-intro-backdrop').is_visible(), 'the backdrop returned'
        page.wait_for_timeout(1700)
        assert page.evaluate(TTL_NOTICES) == 1 and page.evaluate('window.__ttl.size') == 0, ('its time never ran out', facts)
        page.keyboard.press('Escape')
        page.locator('.peer-intro-backdrop').wait_for(state='detached')
        rest(page)
        facts['released'] = released = page.evaluate(OUTSIDE)
        assert len(released) == 1 and released[0]['visible'] and released[0]['hit'], facts
        assert page.evaluate('window.__ttl.size') == 1, ('its time starts only when both surfaces are gone', facts)
        # A portal that stays hidden is not a blocker: with nothing else over the page it must hold no notice.
        facts['stays-hidden'] = page.evaluate("""async () => {
            const { showToast } = await import('/static/modules/toast.js');
            const portal = document.createElement('div');
            portal.setAttribute('aria-modal', 'true');
            portal.hidden = true;
            document.body.appendChild(portal);
            const toast = showToast('Over a portal that stays hidden.', 'warn', { ttl: 0, haptic: false });
            const held = toast.closest('.toast-stack').hasAttribute('data-held');
            toast.dispose();
            portal.remove();
            return held;
        }""")
        assert facts['stays-hidden'] is False, ('a portal that stays hidden holds no page notice', facts)
    finally:
        save_evidence(tmp_path, name, facts)
        page.close()


# The real composer's Model capsule on the static page: its catalog and Main answered, its two writes the test's.
MODEL = {'model': 'claudexor::codex=gpt-6-astra', 'use_local': False}


def _model_catalog():
    codex = lambda model: {'provider_id': 'claudexor', 'provider': 'Codex', 'source_id': 'codex', 'id': model, 'name': model,  # noqa: E731
                           'value': f'claudexor::codex={model}', 'credential_profile_id': 'home', 'availability': 'available'}
    # Its transport carries Ultrafast, so the strip offers it (an attempt nothing confirms).
    return {'model_sources': [{'id': 'codex', 'label': 'Codex', 'credentialHarness': 'codex',
                               'processingPreferences': ['standard', 'fast', 'ultrafast']}], 'errors': [],
            'items': [codex('gpt-6-astra'), codex('gpt-6.1-sol')]}


class ModelWrites:
    """What the two writes of a Model sheet answer: refuse (the default), or wait until the test lets the answer go."""

    def __init__(self):
        self.tier = 'refuse'
        self.pin = 'refuse'
        self.held = []

    def tiers(self, route):
        if self.tier == 'hold':
            self.held.append(route)
        else:
            route.fulfill(status=500, json={'ok': False, 'error': 'The server could not save the tier.'})

    def pins(self, route):
        if route.request.method == 'GET':
            route.fulfill(json={'model_recents': [], 'model_pins': []})
        else:
            route.fulfill(status=500, json={'ok': False, 'error': 'The pins could not be saved.'})


def _model_page(ui, width, height, *, touch, writes):
    page = seed_page(ui, width, height, touch=touch)
    state = {'supervisor_ready': True, 'active_chat_activities': [], 'projects': PROJECTS, 'main_route': MODEL,
             'project_chat_ids': [p['chat_id'] for p in PROJECTS], 'processing_tiers': {'main': '', 'default': 'standard'}}
    page.route(re.compile(r'/api/state(\?.*)?$'), lambda route: route.fulfill(json=state))
    page.route('**/api/model-catalog*', lambda route: route.fulfill(json=_model_catalog()))
    page.route('**/api/ui/preferences', writes.pins)
    page.route('**/api/owner/chat-processing', writes.tiers)
    page.add_init_script(f'({STRETCH})()')
    page.goto(ui['url'] + '/')
    page.wait_for_selector('#chat-input', state='attached')
    settle(page)
    return page


def _open_sheet(page, phone):
    page.locator('#chat-model-mode .chat-model-head').tap() if phone else page.locator('#chat-model-mode .chat-model-head').click()
    page.locator('.ui-popup.model-sheet .model-tier-seg[data-tier="fast"]').wait_for()
    page.locator('.ui-popup.model-sheet .chat-model-pin').first.wait_for()
    rest(page)


def _of(facts, where):
    return [fact for fact in facts if fact['where'] == where]


def test_the_phones_model_sheet_answers_a_tier_and_a_pin_inside_it_and_holds_the_page_notice(toast_ui, tmp_path):
    writes = ModelWrites()
    page = _model_page(toast_ui, 390, 844, touch=True, writes=writes)
    name = f'toast-model-sheet-{toast_ui["engine"]}'
    facts = {}
    sheet = page.locator('.ui-popup.model-sheet')
    try:
        page.evaluate(KEPT, 'Earlier page notice.')
        _open_sheet(page, True)
        facts['opened'] = opened = page.evaluate(TOASTS)
        setup_browser.capture(page, f'{name}-held')
        assert page.evaluate(SHEET)['bottom'] == 'bottom' and page.evaluate(SHEET)['scrim'], 'the phone\'s bottom sheet over its scrim'
        assert [(n['where'], n['visible'], n['hit']) for n in opened] == [('page', False, False)], ('nothing to press under the scrim', facts)
        # A tier the server refuses: said in the sheet, where it is pressed and can be read, its × pressed by a finger.
        sheet.locator('.model-tier-seg[data-tier="fast"]').tap()
        sheet.locator('.toast').wait_for()
        rest(page)
        facts['tier'] = tier = page.evaluate(TOASTS)
        setup_browser.capture(page, f'{name}-tier')
        assert [(n['where'], n['text'], n['visible'], n['hit'], n['inSheet']) for n in _of(tier, 'sheet')] == [
            ('sheet', 'The server could not save the tier.', True, True, True)], ('said in the sheet, pressable', facts)
        assert [(n['visible'], n['hit']) for n in _of(tier, 'page')] == [(False, False)], facts
        _press_at(page, _of(tier, 'sheet')[0]['box'], True)
        assert page.evaluate(NOTICES) == 1 and page.evaluate(SHEET)['sheet'], ('the × hid the notice and nothing else', facts)
        # A Pin the server refuses answers the same way.
        sheet.locator('.chat-model-pin').first.tap()
        sheet.locator('.toast').wait_for()
        rest(page)
        facts['pin'] = pin = page.evaluate(TOASTS)
        assert [(n['text'], n['visible'], n['hit'], n['inSheet']) for n in _of(pin, 'sheet')] == [
            ('The pins could not be saved.', True, True, True)], facts
        _press_at(page, _of(pin, 'sheet')[0]['box'], True)
        assert _of(page.evaluate(TOASTS), 'sheet') == [] and page.evaluate(SHEET)['sheet'], facts
        # The keyboard: the × is reached by Tab and pressed; the sheet keeps the focus (and its trap) and takes the notice
        # along when it closes, its timer with it.
        sheet.locator('.chat-model-pin').first.tap()
        sheet.locator('.toast').wait_for()
        page.evaluate("() => { window.__said = document.querySelector('.model-sheet .toast'); }")
        page.evaluate("() => document.querySelector('.model-sheet .toast .toast-dismiss').focus()")
        page.keyboard.press('Enter')
        page.wait_for_function('() => !document.querySelector(".model-sheet .toast")')
        facts['kept'] = kept = page.evaluate("() => document.querySelector('.ui-popup.model-sheet').contains(document.activeElement)")
        assert kept, ('the focus stays in the sheet, so its Tab trap still holds', facts)
        sheet.locator('.chat-model-pin').first.tap()
        sheet.locator('.toast').wait_for()
        page.evaluate("() => { window.__said = document.querySelector('.model-sheet .toast'); }")
        # Closing gives the page its notice back, whole, and takes what the sheet hosted.
        page.keyboard.press('Escape')
        page.wait_for_function(SHEET_GONE)
        rest(page)
        assert page.evaluate('window.__said.parentNode === null'), 'the notice, its timer and its listener left with the sheet'
        facts['closed'] = closed = page.evaluate(TOASTS)
        assert [(n['where'], n['visible'], n['hit']) for n in closed] == [('page', True, True)], facts
    finally:
        save_evidence(tmp_path, name, facts)
        page.close()


def test_the_question_before_main_takes_ultrafast_answers_inside_the_sheet_it_leaves_open(toast_ui, tmp_path):
    """Main's Ultrafast reads its settings first, with the sheet still open: a refusal of that read is said in it."""
    writes = ModelWrites()
    page = _model_page(toast_ui, 390, 844, touch=True, writes=writes)
    page.route('**/api/settings', lambda route: route.fulfill(status=500, json={'error': 'The settings could not be read.'}))
    name = f'toast-model-sheet-ultrafast-{toast_ui["engine"]}'
    facts = {}
    sheet = page.locator('.ui-popup.model-sheet')
    try:
        page.evaluate(KEPT, 'Earlier page notice.')
        page.locator('#chat-model-mode .chat-model-head').tap()
        sheet.locator('.model-tier-seg[data-tier="ultrafast"]').wait_for()
        rest(page)
        sheet.locator('.model-tier-seg[data-tier="ultrafast"]').tap()
        sheet.locator('.toast').wait_for()
        rest(page)
        facts['ultrafast'] = seen = page.evaluate(TOASTS)
        assert [(n['text'], n['visible'], n['hit'], n['inSheet']) for n in _of(seen, 'sheet')] == [
            ('The settings could not be read.', True, True, True)], ('said where it was asked', facts)
        assert page.evaluate(SHEET)['sheet'] and [(n['visible'], n['hit']) for n in _of(seen, 'page')] == [(False, False)], facts
    finally:
        save_evidence(tmp_path, name, facts)
        page.close()


def test_a_tier_refused_after_the_sheet_went_is_the_pages_never_a_later_sheets(toast_ui, tmp_path):
    writes = ModelWrites()
    writes.tier = 'hold'
    page = _model_page(toast_ui, 390, 844, touch=True, writes=writes)
    name = f'toast-model-sheet-late-{toast_ui["engine"]}'
    facts = {}
    sheet = page.locator('.ui-popup.model-sheet')
    try:
        _open_sheet(page, True)
        sheet.locator('.model-tier-seg[data-tier="fast"]').tap()
        page.wait_for_timeout(300)
        assert len(writes.held) == 1, 'the save is on its way'
        page.keyboard.press('Escape')
        page.wait_for_function(SHEET_GONE)
        _open_sheet(page, True)  # the next opening, of the same capsule
        writes.held.pop().fulfill(status=500, json={'ok': False, 'error': 'Too late.'})
        page.wait_for_timeout(500)
        facts['late'] = late = page.evaluate(TOASTS)
        assert [(n['where'], n['text']) for n in late] == [('page', 'Too late.')], ('the page\'s, not the next sheet\'s', facts)
        assert [(n['visible'], n['hit']) for n in late] == [(False, False)], ('and it waits behind the next sheet\'s scrim', facts)
        page.keyboard.press('Escape')
        page.wait_for_function(SHEET_GONE)
        rest(page)
        facts['after'] = after = page.evaluate(TOASTS)
        assert [(n['where'], n['visible'], n['hit']) for n in after] == [('page', True, True)], facts
    finally:
        save_evidence(tmp_path, name, facts)
        page.close()


def test_the_desktops_model_popover_blocks_nothing_and_says_its_failure_on_the_page(toast_ui, tmp_path):
    writes = ModelWrites()
    page = _model_page(toast_ui, *WIDE, touch=False, writes=writes)
    name = f'toast-model-popover-{toast_ui["engine"]}'
    facts = {}
    try:
        page.evaluate(KEPT, 'Earlier page notice.')
        _open_sheet(page, False)
        assert page.evaluate(SHEET) == {'sheet': True, 'scrim': False, 'bottom': None}, 'a popover: no scrim, no modal'
        facts['opened'] = opened = page.evaluate(TOASTS)
        assert [(n['where'], n['visible'], n['hit']) for n in opened] == [('page', True, True)], ('still live beside it', facts)
        page.locator('.ui-popup.model-sheet .model-tier-seg[data-tier="fast"]').click()
        page.locator(f'{STACK} .toast', has_text='could not save the tier').wait_for()
        facts['tier'] = tier = page.evaluate(TOASTS)
        assert sorted((n['where'], n['visible'], n['hit']) for n in tier) == [('page', True, True)] * 2, facts
    finally:
        save_evidence(tmp_path, name, facts)
        page.close()


@pytest.mark.parametrize('origin', ['composer', 'card'], ids=['composer', 'card'])
def test_a_page_toast_close_leaves_the_desktop_model_sheet_open(toast_ui, tmp_path, origin):
    writes = ModelWrites()
    page = _model_page(toast_ui, *WIDE, touch=False, writes=writes)
    name = f'toast-model-popover-close-{origin}-{toast_ui["engine"]}'
    facts = {}
    sheet = page.locator('.ui-popup.model-sheet')

    def open_sheet():
        if origin == 'card':
            if not page.locator('.card-model-fact').count():
                page.evaluate(CARD)
            page.locator('.card-model-fact').first.click()
        else:
            _open_sheet(page, False)
        sheet.locator('.chat-model-pin').first.wait_for()

    try:
        open_sheet()
        for step, press in (('mouse', lambda close: close.click()),
                            ('keyboard', lambda close: (close.focus(), page.keyboard.press('Enter')))):
            sheet.locator('.chat-model-pin').first.click()
            notice = page.locator(f'{STACK} .toast', has_text='pins could not be saved')
            notice.wait_for()
            press(notice.get_by_role('button', name='Dismiss notice'))
            notice.wait_for(state='detached')
            facts[step] = {'sheet': sheet.count(), 'notices': page.evaluate(NOTICES)}
            assert facts[step]['sheet'] == 1, ('the × hid only the notice', facts)
        page.keyboard.press('Escape')
        page.wait_for_function(SHEET_GONE)
        open_sheet()
        page.mouse.click(10, 10)
        page.wait_for_function(SHEET_GONE)
        facts['outside'] = True
    finally:
        save_evidence(tmp_path, name, facts)
        page.close()


# A work's model fact on its card: the real card module on the static page, a work of this window's Main.
CARD = """async () => {
    const { paintCardModel } = await import('/static/modules/card_model.js');
    const card = document.createElement('div');
    card.className = 'chat-live-card';
    card.dataset.taskId = 'card-task';
    const meta = document.createElement('div');
    meta.dataset.liveMeta = '';
    card.append(meta);
    document.querySelector('#chat-messages').append(card);
    const main = { model: 'claudexor::codex=gpt-6-astra', use_local: false };
    paintCardModel({ metaEl: meta }, { lead: { ...main, basis: 'main' }, target: 'card-task', main, queued: false, since: '',
        processing: { choice: '', resolved: 'standard' }, tiers: { main: '', default: 'standard' } }, { rerender: () => {} });
    card.scrollIntoView({ block: 'center' });
}"""


def test_a_works_model_sheet_answers_its_tier_and_pin_inside_it(toast_ui, tmp_path):
    writes = ModelWrites()
    page = _model_page(toast_ui, 390, 844, touch=True, writes=writes)
    name = f'toast-card-model-sheet-{toast_ui["engine"]}'
    facts = {}
    sheet = page.locator('.ui-popup.model-sheet')
    answered = []

    def task_model(route):  # CLIENT_CONTRACT §5: one act on a work's model
        answered.append(route.request.post_data)
        route.fulfill(json={'ok': True, 'task_id': 'card-task', 'applied': 'next_step', 'processing': {'choice': 'fast', 'resolved': 'fast'}})

    try:
        page.route('**/api/tasks/*/model', task_model)
        page.evaluate(CARD)
        page.evaluate(KEPT, 'Earlier page notice.')
        page.locator('.card-model-fact').tap()
        sheet.locator('.model-tier-seg[data-tier="fast"]').wait_for()
        sheet.locator('.chat-model-pin').first.wait_for()
        rest(page)
        assert sheet.locator('.card-model-head').count() == 1 and page.evaluate(SHEET)['bottom'] == 'bottom'
        facts['opened'] = opened = page.evaluate(TOASTS)
        assert [(n['where'], n['visible'], n['hit']) for n in opened] == [('page', False, False)], ('held behind the scrim', facts)
        # A tier is the work's own act: its answer stays in the sheet, which stays open.
        sheet.locator('.model-tier-seg[data-tier="fast"]').tap()
        sheet.locator('.toast').wait_for()
        rest(page)
        facts['tier'] = tier = page.evaluate(TOASTS)
        setup_browser.capture(page, f'{name}-tier')
        assert [(n['where'], n['visible'], n['hit'], n['inSheet']) for n in _of(tier, 'sheet')] == [('sheet', True, True, True)], facts
        assert _of(tier, 'sheet')[0]['text'].startswith('Fast for this work.'), facts
        _press_at(page, _of(tier, 'sheet')[0]['box'], True)
        assert _of(page.evaluate(TOASTS), 'sheet') == [] and page.evaluate(SHEET)['sheet'], ('the × hides only the notice', facts)
        # A Pin the server refuses.
        sheet.locator('.chat-model-pin').first.tap()
        sheet.locator('.toast').wait_for()
        rest(page)
        facts['pin'] = pin = page.evaluate(TOASTS)
        assert [(n['text'], n['visible'], n['hit']) for n in _of(pin, 'sheet')] == [('The pins could not be saved.', True, True)], facts
        _press_at(page, _of(pin, 'sheet')[0]['box'], True)
        assert page.evaluate(SHEET)['sheet'], facts
        page.keyboard.press('Escape')
        page.wait_for_function(SHEET_GONE)
        rest(page)
        assert [(n['where'], n['visible'], n['hit']) for n in page.evaluate(TOASTS)] == [('page', True, True)], facts
    finally:
        save_evidence(tmp_path, name, facts)
        page.close()


def test_a_notifications_open_runs_once_however_often_enter_comes_while_it_fades(toast_ui, tmp_path):
    """The notice leaves over 180 ms, and its button is still there to take the next Enter: Open is one act per notice."""
    page = _open(toast_ui, *WIDE)
    name = f'toast-open-once-{toast_ui["engine"]}'
    facts = {}
    try:
        assert page.evaluate(ROUTED, 'once-task') == 'in_app'
        routed = page.locator(f'{STACK} .toast', has_text='Task finished')
        routed.get_by_role('button', name='Open').focus()
        page.keyboard.press('Enter')
        page.wait_for_timeout(70)  # inside the fade
        page.keyboard.press('Enter')
        routed.wait_for(state='detached')
        facts['opened'] = page.evaluate('window.opened')
        assert len(facts['opened']) == 1, ('one notice, one Open', facts)
    finally:
        save_evidence(tmp_path, name, facts)
        page.close()


def test_a_photos_open_in_a_shell_that_cannot_open_files_copies_its_link_inside_the_reader(toast_ui, tmp_path):
    """A launcher older than its file bridge (or one with neither method) degrades to the copy-link handoff, and the reader
    that asked hears it where it is pressed. Without a shell, Open is the browser's own new tab, and says nothing."""
    page = _modal_page(toast_ui, False)
    name = f'toast-open-no-bridge-{toast_ui["engine"]}'
    facts = {}
    dialog = page.locator('dialog.document-reader[open]')
    tile = page.locator('#chat-messages .chat-photo-open')
    try:
        page.evaluate("() => { window.__opened = []; window.open = (...args) => { window.__opened.push(args[0]); return null; }; }")
        page.evaluate("() => { delete window.pywebview; window.clip.link = true; }")
        tile.click()
        dialog.wait_for()
        dialog.locator('[data-reader-action="open"]').click()
        page.wait_for_timeout(300)
        facts['web'] = {'opened': len(page.evaluate('window.__opened')), 'notices': page.evaluate(NOTICES)}
        assert facts['web'] == {'opened': 1, 'notices': 0}, ('the browser opens its own tab, quietly', facts)
        page.keyboard.press('Escape')
        dialog.wait_for(state='detached')
        # The desktop shell with a bridge that has neither file method: its interceptor stands in front of window.open.
        page.evaluate("""() => { window.pywebview = { api: {} }; window.dispatchEvent(new Event('pywebviewready')); }""")
        tile.click()
        dialog.wait_for()
        dialog.locator('[data-reader-action="open"]').click()
        page.locator('.toast').wait_for()
        rest(page)
        facts['shell'] = seen = page.evaluate(MODAL_FACTS)
        setup_browser.capture(page, f'{name}-copied')
        assert (seen['text'], seen['page'], seen['closeHit']) == ('Link copied — open it in your browser.', [], True), facts
        assert page.evaluate("window.__opened.length") == 1, 'no second tab'
    finally:
        save_evidence(tmp_path, name, facts)
        page.close()


STACK_FACTS = """() => {
    const dialog = document.querySelector('dialog[open]');
    const stack = dialog.querySelector(':scope > .toast-stack--room');
    const box = (el) => { const r = el.getBoundingClientRect(); return { top: r.top, bottom: r.bottom, left: r.left, right: r.right }; };
    const hit = (el) => { const r = el.getBoundingClientRect();
        return el.contains(document.elementFromPoint((r.left + r.right) / 2, (r.top + r.bottom) / 2)); };
    const actions = [...dialog.querySelectorAll('[data-reader-action]')].filter((el) => el.getClientRects().length);
    const toasts = [...stack.querySelectorAll('.toast')];
    return { count: toasts.length, stack: box(stack), view: [innerWidth, innerHeight],
        scrolls: stack.scrollHeight > stack.clientHeight + 1, scrollTop: stack.scrollTop,
        actions: actions.map((el) => ({ name: el.dataset.readerAction, hit: hit(el) })),
        closes: toasts.map((toast) => { const close = toast.querySelector('.toast-dismiss'); return { box: box(close),
            shown: box(close).top >= 0 && box(close).bottom <= innerHeight }; }) };
}"""


def test_a_readers_notices_are_bounded_to_it_and_scroll_so_every_close_and_action_stays_reachable(toast_ui, tmp_path):
    page = _modal_page(toast_ui, True, height=640)
    name = f'toast-reader-bounded-{toast_ui["engine"]}'
    facts = {}
    dialog = page.locator('dialog.document-reader[open]')
    try:
        page.evaluate(STRETCH)
        page.locator('#chat-messages .chat-photo-open').tap()
        dialog.wait_for()
        page.evaluate("""() => { window.clip.mode = 'reject';
            window.clip.error = 'The browser refused clipboard access for this page; allow it in the site settings and try again.'; }""")
        for _ in range(8):  # eight refusals inside one notice's six seconds
            dialog.locator('[data-reader-action="copy"]').evaluate('el => el.click()')
            page.wait_for_timeout(40)
        page.wait_for_function(f'() => {NOTICES} === 8')
        rest(page)
        facts['eight'] = seen = page.evaluate(STACK_FACTS)
        setup_browser.capture(page, f'{name}-eight')
        assert seen['count'] == 8 and seen['stack']['top'] >= 0, ('inside the screen', facts)
        assert all(action['hit'] for action in seen['actions']) and seen['actions'], ('the reader\'s actions stay clear', facts)
        assert seen['scrolls'], ('bounded: the column scrolls', facts)
        # Every close is reached by scrolling the column, the oldest first.
        page.locator('dialog[open] > .toast-stack--room').evaluate('el => { el.scrollTop = 0; }')
        page.wait_for_timeout(100)
        top = page.evaluate(STACK_FACTS)
        assert top['closes'][0]['shown'] and top['actions'] == seen['actions'], ('the oldest × is reached', facts)
        page.locator('dialog[open] > .toast-stack--room').evaluate('el => { el.scrollTop = el.scrollHeight; }')
        page.wait_for_timeout(100)
        bottom = page.evaluate(STACK_FACTS)
        assert bottom['closes'][-1]['shown'], ('and the newest', facts)
    finally:
        save_evidence(tmp_path, name, facts)
        page.close()


def test_the_place_sheet_takes_the_notices_it_hosted_with_it(toast_ui, tmp_path):
    page = _open(toast_ui, *WIDE)
    name = f'toast-place-sheet-{toast_ui["engine"]}'
    facts = {}
    try:
        facts = page.evaluate("""async () => {
            const { sheetShell, showSheet } = await import('/static/modules/place_sheet.js');
            const { showToast } = await import('/static/modules/toast.js');
            const dialog = sheetShell('A place', 'meta');
            const sheet = showSheet(dialog, { returnFocus: document.body });
            const toast = showToast('Said inside the place sheet.', 'warn', { host: dialog, ttl: 6000, haptic: false });
            const before = toast.isConnected && toast.closest('dialog') === dialog;
            sheet.close();
            await new Promise((resolve) => setTimeout(resolve, 50));
            return { before, left: toast.parentNode === null };
        }""")
        assert facts == {'before': True, 'left': True}, ('the notice, its timer and its listener leave with the sheet', facts)
    finally:
        save_evidence(tmp_path, name, facts)
        page.close()
