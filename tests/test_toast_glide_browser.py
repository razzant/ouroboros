"""Toast motion (DESIGN "Motion") and the shared edge on cut words alone (principle 9).
Uses the real UI on a static-server fixture; never starts the runtime or reads live data."""
from __future__ import annotations

import json

import pytest

from tests import test_subscription_setup_browser as setup_browser
from tests.test_design_system_gate_browser import rest, save_evidence, seed_page, settle

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
    toasts[0].click();
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


def test_a_timed_hurry_refusal_fades_only_where_its_words_are_cut(toast_ui, tmp_path):
    """A real task-control refusal can fill the page's notice stack without adding a close button."""
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
        assert toast.get_attribute('role') == 'alert' and not toast.get_by_role('button').count()
        facts['cut'] = cut = _read(page)
        setup_browser.capture(page, f'{name}-cut')
        _assert_edges(cut, BELOW)
        assert cut['role'] == 'region' and cut['toasts'][0]['words']['bottom'] > cut['bottom'], cut
        toast.wait_for(state='detached', timeout=8000)
        assert not page.locator(STACK).get_attribute('role'), 'expiry releases the scroll region'
        page.evaluate("""async () => (await import('/static/modules/toast.js')).showToast('Saved.', 'ok') && true""")
        rest(page)
        facts['short'] = short = _read(page)
        _assert_edges(short, CLEAR)
        assert short['role'] is None and short['scroll_end'] == 0 and _shown(short['toasts'][0]['words'], short), short
        assert not toast.get_by_role('button').count()
        toast.click()
        toast.wait_for(state='detached')
    finally:
        save_evidence(tmp_path, name, facts)
        page.close()


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
