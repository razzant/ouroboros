"""A dismissed toast fades where it stood while the toasts that stay glide into its place (docs/DESIGN.md §10
"Motion"; the R2 board: the one below jumped 53 px in one frame). With reduced motion they take their place at once.
Notices kept until dismissed stay reachable whole, however many the page's stack holds.
Uses the static-server fixture; never starts the runtime or reads live data."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tests import test_subscription_setup_browser as setup_browser
from tests.test_design_system_gate_browser import save_evidence, seed_page, settle

subscription_ui = setup_browser.subscription_ui
pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]


@pytest.fixture(params=['webkit', 'chromium'])
def toast_ui(request, monkeypatch):
    monkeypatch.setenv('OUROBOROS_UI_BROWSER_ENGINE', request.param)
    ui = request.getfixturevalue('subscription_ui')
    ui['engine'] = request.param
    return ui


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
    page = seed_page(toast_ui, 1280, 800, touch=False)
    if reduce:
        page.emulate_media(reduced_motion='reduce')
    facts = {'engine': toast_ui['engine'], 'reduce': reduce}
    try:
        page.goto(toast_ui['url'] + '/')
        page.wait_for_selector('#chat-input', state='attached')
        settle(page)
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


# Two Resumes of different tasks, each answered with a gap: the owner's own Resume door (its toasts, the real
# stylesheet and theme); only the server's answers are given.
TWO_GAPS = """async () => {
    const { resumeTaskAction } = await import('/static/modules/task_control_menu.js');
    const gaps = (id) => ({ count: 23, rows: Array.from({ length: 20 }, (_, n) => (
        { task_id: `${id}-${String(n).padStart(2, '0')}-${'f'.repeat(24)}`, reason: 'body_unreadable' })) });
    for (const id of ['gap-a', 'gap-b']) {
        await resumeTaskAction(id, { resume: async () => ({ ok: true, membership_gaps: gaps(id) }) });
    }
}"""
WARNINGS_ONLY = """() => {
    const toasts = [...document.querySelectorAll('#toast-stack .toast')];
    return toasts.length === 2 && toasts.every((el) => el.classList.contains('toast-warn'));
}"""
# The page's stack and each notice in it, where they are laid out now.
REACH = """() => {
    const stack = document.getElementById('toast-stack');
    const rect = (el) => { const r = el.getBoundingClientRect(); return { top: r.top, bottom: r.bottom, left: r.left, right: r.right }; };
    return {
        stack: { ...rect(stack), scroll_top: stack.scrollTop, scroll_end: stack.scrollHeight - stack.clientHeight,
                 role: stack.getAttribute('role'), label: stack.getAttribute('aria-label') },
        toasts: [...stack.querySelectorAll('.toast')].map((el) => ({ ...rect(el), close: rect(el.querySelector('.toast-dismiss')) })),
        focused_close: [...stack.querySelectorAll('.toast-dismiss')].indexOf(document.activeElement),
        view: { width: innerWidth, height: innerHeight }, theme: document.documentElement.dataset.theme,
    };
}"""


def _shown(box, facts):
    """Wholly inside the window and inside the part of the page's stack it shows."""
    view, stack = facts['view'], facts['stack']
    return (max(0, stack['top']) - 0.5 <= box['top'] and box['bottom'] <= min(view['height'], stack['bottom']) + 0.5
            and -0.5 <= box['left'] and box['right'] <= view['width'] + 0.5)


SCROLL = """() => { const s = document.getElementById('toast-stack');
    return { top: s.scrollTop, end: s.scrollHeight - s.clientHeight, focused: document.activeElement === s }; }"""


def _key_scroll(page, key, to_end):
    """Press ``key`` on the focused stack and read its smooth scroll until it rests where the key sends it."""
    page.keyboard.press(key)
    for _ in range(50):
        page.wait_for_timeout(100)
        state = page.evaluate(SCROLL)
        if (state['top'] >= state['end'] - 1) if to_end else state['top'] == 0:
            return state
    raise AssertionError((key, 'the keyboard did not scroll the notices there', state))


@pytest.mark.parametrize('width,height', [(320, 568), (1280, 800)], ids=['phone', 'wide'])
def test_two_kept_warnings_are_reached_whole_by_wheel_and_keyboard(toast_ui, tmp_path, width, height):
    """Once both "Resuming" lines time out, the two gap warnings stay. On a phone the page's stack ends above the
    screen's bottom and scrolls: a wheel over a notice reaches the second close button and the last word, the keyboard
    reaches them too, and scrolling dismisses nothing. A wide window has room for both: nothing there scrolls."""
    page = seed_page(toast_ui, width, height)
    evidence = Path(os.environ.get('OUROBOROS_UI_EVIDENCE_DIR', str(tmp_path)))
    evidence.mkdir(parents=True, exist_ok=True)
    name = f'toast-reach-{toast_ui["engine"]}-{width}x{height}'
    facts = {'engine': toast_ui['engine']}
    try:
        page.goto(toast_ui['url'] + '/')
        page.wait_for_selector('#chat-input', state='attached')
        settle(page)
        page.evaluate(TWO_GAPS)
        page.wait_for_function(WARNINGS_ONLY, timeout=15_000)
        for theme in ('dark', 'light'):
            page.emulate_media(color_scheme=theme)
            page.wait_for_function('(t) => document.documentElement.dataset.theme === t', arg=theme)
            settle(page)
            facts[theme] = page.evaluate(REACH)
            page.screenshot(path=str(evidence / f'{name}-{theme}.png'))
        at_rest = facts['light']
        assert abs(at_rest['toasts'][0]['top'] - 76) < 0.5, ('the first notice keeps its place under the header', at_rest)
        if width > 720:  # room for both: nothing scrolls and both keep their place, 18 px from the right
            assert at_rest['stack']['scroll_end'] <= 0 and at_rest['stack']['role'] is None, at_rest
            assert all(abs(width - 18 - t['right']) < 0.5 for t in at_rest['toasts']), at_rest
            assert all(_shown(t, at_rest) and _shown(t['close'], at_rest) for t in at_rest['toasts']), at_rest
            return
        assert at_rest['stack']['bottom'] <= height - 18 + 0.5, ('the stack ends above the screen', at_rest)
        assert at_rest['stack']['scroll_end'] > 0, at_rest
        assert (at_rest['stack']['role'], at_rest['stack']['label']) == ('region', 'Scrollable notices'), at_rest
        assert _shown(at_rest['toasts'][0]['close'], at_rest), ('the first close button is on screen', at_rest)
        first = at_rest['toasts'][0]
        scrolled = "(end) => { const s = document.getElementById('toast-stack'); " \
                   "return end ? s.scrollTop >= s.scrollHeight - s.clientHeight - 1 : s.scrollTop === 0; }"
        page.mouse.move((first['left'] + first['right']) / 2, (first['top'] + at_rest['stack']['bottom']) / 2)
        for _ in range(40):
            page.mouse.wheel(0, 300)
            settle(page)
        page.wait_for_function(scrolled, arg=True)
        page.mouse.move(1, height / 2)  # the wheel at rest, off the notices: the keys below are the keyboard's alone
        page.wait_for_timeout(300)
        facts['wheel'] = wheeled = page.evaluate(REACH)
        assert len(wheeled['toasts']) == 2, ('a scroll dismisses nothing', wheeled)
        assert wheeled['stack']['scroll_top'] >= wheeled['stack']['scroll_end'] - 1, ('the wheel reached the end', wheeled)
        assert _shown(wheeled['toasts'][1]['close'], wheeled), wheeled
        assert wheeled['toasts'][1]['bottom'] <= wheeled['stack']['bottom'] + 0.5, ('the last word is shown', wheeled)
        page.screenshot(path=str(evidence / f'{name}-wheel-end.png'))
        page.locator('#toast-stack').focus()  # the region the keyboard reaches
        facts['keys'] = [_key_scroll(page, 'Home', False), _key_scroll(page, 'End', True)]  # End: the last word
        page.screenshot(path=str(evidence / f'{name}-region-end.png'))
        facts['keys'].append(_key_scroll(page, 'Home', False))
        page.locator('#toast-stack .toast-dismiss').first.focus()
        # From the first notice's close button to the second's: Safari moves between buttons with Option-Tab.
        page.keyboard.press('Alt+Tab' if toast_ui['engine'] == 'webkit' else 'Tab')
        page.wait_for_timeout(500)
        facts['focus'] = focused = page.evaluate(REACH)
        assert focused['focused_close'] == 1 and _shown(focused['toasts'][1]['close'], focused), focused
        page.screenshot(path=str(evidence / f'{name}-second-close-focused.png'))
        page.keyboard.press('Enter')
        page.wait_for_function("() => document.querySelectorAll('#toast-stack .toast').length === 1")
    finally:
        save_evidence(tmp_path, name, facts)
        page.close()


MARGIN_HIT = """([x, y]) => document.getElementById('toast-stack').contains(document.elementFromPoint(x, y))"""


def test_the_kept_column_follows_the_window_and_scrolls_from_its_margin(toast_ui, tmp_path):
    """On a phone the two kept warnings cut the column, and a wheel over its empty left margin scrolls it in every
    engine. A window that grows releases the column with no notice coming or going, and its margin lets the page
    through again; shrunk back, the column is cut again."""
    page = seed_page(toast_ui, 320, 568)
    facts = {'engine': toast_ui['engine']}
    try:
        page.goto(toast_ui['url'] + '/')
        page.wait_for_selector('#chat-input', state='attached')
        settle(page)
        page.evaluate(TWO_GAPS)
        page.wait_for_function(WARNINGS_ONLY, timeout=15_000)
        settle(page)
        facts['cut'] = cut = page.evaluate(REACH)
        assert cut['stack']['role'] == 'region' and cut['stack']['scroll_end'] > 0, cut
        first = cut['toasts'][0]
        margin = ((cut['stack']['left'] + first['left']) / 2, (cut['stack']['top'] + cut['stack']['bottom']) / 2)
        assert page.evaluate(MARGIN_HIT, list(margin)), ('the cut column takes the pointer in its margin', facts)
        page.mouse.move(*margin)
        for _ in range(10):
            page.mouse.wheel(0, 300)
            settle(page)
        facts['margin_wheel'] = page.evaluate(SCROLL)
        assert facts['margin_wheel']['top'] > 0, ('a wheel over the margin scrolls the cut column', facts)
        page.set_viewport_size({'width': 1280, 'height': 800})
        page.wait_for_function("() => !document.getElementById('toast-stack').hasAttribute('tabindex')")
        facts['wide'] = wide = page.evaluate(REACH)
        first = wide['toasts'][0]
        margin = ((wide['stack']['left'] + first['left']) / 2, (wide['stack']['top'] + wide['stack']['bottom']) / 2)
        assert not page.evaluate(MARGIN_HIT, list(margin)), ('the released column lets the page through', facts)
        page.set_viewport_size({'width': 320, 'height': 568})
        page.wait_for_function("() => document.getElementById('toast-stack').getAttribute('role') === 'region'")
    finally:
        save_evidence(tmp_path, f'toast-follow-{toast_ui["engine"]}', facts)
        page.close()
