"""A message's own gestures on a phone (owner 19.10, 4A; docs/DESIGN.md "Replies").

Swiped left past the reply point, a saved message replies as its Reply does: the row follows the finger, the reply mark
fills and takes the accent, the composer stages the quote, and the row settles back; short of the point nothing is
staged, and a thumb that drifts down once the swipe is sideways never scrolls the feed. Rightwards the finger stays the
drawer's. A long press sinks the row while the finger holds, then lifts it over the dimmed feed while its menu is
open (the header stays bright), and the row is back at rest once the menu closes. Saved rows come from the real
history endpoint (test_shell_interactions_browser.saved_history); synthetic CDP touch is Chromium's input pipeline, not
a claim about an iPhone.
"""
from __future__ import annotations

import io

import pytest
from PIL import Image

from tests import test_subscription_setup_browser as setup_browser
from tests.test_design_system_gate_browser import seed_page
from tests.test_shell_interactions_browser import DRAWER, boot, chromium_ui, evidence, saved_history

subscription_ui = setup_browser.subscription_ui
pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]
__all__ = ['chromium_ui', 'subscription_ui']

ROW = """sel => { const row = document.querySelector(sel); const r = row.getBoundingClientRect();
    const mark = document.querySelector('.chat-row-reply-mark');
    return { x: r.x, y: r.y, w: r.width, h: r.height, swipe: row.dataset.rowSwipe || '', press: row.dataset.rowPress || '',
        transform: getComputedStyle(row).transform, shadow: getComputedStyle(row).boxShadow,
        select: getComputedStyle(row).userSelect || getComputedStyle(row).webkitUserSelect,
        mark: mark ? { armed: mark.dataset.armed || '', k: Number(mark.style.getPropertyValue('--reply-k')), shown: mark.isConnected } : null,
        staged: document.getElementById('chat-input-reply')?.textContent || '', scrollTop: document.getElementById('chat-messages').scrollTop,
        menu: [...document.querySelectorAll('.chat-row-menu:not([data-leaving]) [data-row-action]')].map((item) => item.dataset.rowAction) }; }"""
# Playwright leaves the order of init scripts undefined: the recorder patches the stub's Telegram whether the stub is
# already there or assigns it later.
RECORD_HAPTICS = """(() => { const played = window.__haptics = [];
    const patch = (tg) => { if (tg && tg.WebApp) tg.WebApp.HapticFeedback = { impactOccurred: (style) => played.push(style),
        selectionChanged: () => played.push('tick'), notificationOccurred: (type) => played.push(type) }; return tg; };
    if (window.Telegram) { patch(window.Telegram); return; }
    let telegram;
    Object.defineProperty(window, 'Telegram', { configurable: true, get: () => telegram, set: (value) => { telegram = patch(value); } });
})()"""
ASSISTANT = '#chat-messages .chat-bubble.assistant.has-reply'
USER = '#chat-messages .chat-bubble.user.has-reply'


class Finger:
    """One CDP touch, moved by hand."""

    def __init__(self, page):
        self.page = page
        self.cdp = page.context.new_cdp_session(page)

    def send(self, kind, x=0, y=0):
        points = [] if kind == 'touchEnd' else [{'x': x, 'y': y, 'id': 1}]
        self.cdp.send('Input.dispatchTouchEvent', {'type': kind, 'touchPoints': points})
        self.page.wait_for_timeout(16)

    def path(self, start, end, steps=12):
        for i in range(1, steps + 1):
            self.send('touchMove', start[0] + (end[0] - start[0]) * i / steps, start[1] + (end[1] - start[1]) * i / steps)

    def close(self):
        self.cdp.detach()


def luminance(png, x, y):
    r, g, b = Image.open(io.BytesIO(png)).convert('RGB').getpixel((x, y))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def booted(chromium_ui, tmp_path, theme='light', telegram=False):
    page = seed_page(chromium_ui, 390, 844, telegram=telegram)
    page.add_init_script(f"try {{ localStorage.setItem('ouroboros.theme', '{theme}'); }} catch (e) {{}}")
    if telegram:  # the Telegram stub's HapticFeedback records what the page asks Telegram to play
        page.add_init_script(RECORD_HAPTICS)
    client, respond = saved_history(tmp_path)
    page.route('**/api/chat/history**', respond)
    boot(page, chromium_ui)
    page.wait_for_selector(ASSISTANT, timeout=20_000)
    page.wait_for_timeout(400)
    return page, client


def test_a_message_swiped_left_past_the_point_replies_and_settles_back(chromium_ui, tmp_path):
    page, client = booted(chromium_ui, tmp_path)
    finger = Finger(page)
    facts = {}
    try:
        # Short of the reply point: the row comes back and nothing is staged.
        user = page.evaluate(ROW, USER)
        start = (user['x'] + user['w'] - 24, user['y'] + user['h'] / 2)
        finger.send('touchStart', *start)
        finger.path(start, (start[0] - 40, start[1] + 2), steps=6)
        facts['short_drag'] = page.evaluate(ROW, USER)
        finger.send('touchEnd')
        page.wait_for_timeout(450)
        facts['short_rest'] = page.evaluate(ROW, USER)
        # Past it, with the thumb drifting down once the swipe is sideways.
        row = page.evaluate(ROW, ASSISTANT)
        start = (row['x'] + row['w'] - 24, row['y'] + row['h'] / 2)
        facts['before'] = row
        finger.send('touchStart', *start)
        finger.path(start, (start[0] - 30, start[1] + 2), steps=3)
        facts['early'] = page.evaluate(ROW, ASSISTANT)
        finger.path((start[0] - 30, start[1] + 2), (start[0] - 110, start[1] + 60), steps=8)
        facts['armed'] = page.evaluate(ROW, ASSISTANT)
        page.screenshot(path=str(tmp_path / 'row-swipe-armed.png'))
        finger.send('touchEnd')
        page.wait_for_timeout(60)
        facts['released'] = page.evaluate(ROW, ASSISTANT)
        page.wait_for_timeout(450)
        facts['rest'] = page.evaluate(ROW, ASSISTANT)
    finally:
        finger.close()
        evidence(tmp_path, 'row-swipe-reply', facts)
        page.close()
        client.close()
    assert facts['short_drag']['swipe'] == 'drag' and facts['short_drag']['mark']['armed'] in ('', 'false'), facts
    assert facts['short_rest']['staged'] == '' and facts['short_rest']['transform'] == 'none', ('short of the point', facts)
    assert facts['short_rest']['swipe'] == '' and facts['short_rest']['mark'] is None, facts
    early, armed = facts['early'], facts['armed']
    assert early['swipe'] == 'drag' and early['transform'].startswith('matrix(1, 0, 0, 1, -'), ('the row follows', facts)
    assert 0 < early['mark']['k'] < 1, facts
    assert armed['mark']['armed'] == 'true' and armed['mark']['k'] == 1, ('past the point the mark takes the accent', facts)
    assert armed['scrollTop'] == facts['before']['scrollTop'], ('a sideways swipe never scrolls the feed', facts)
    assert 'The build is green' in facts['released']['staged'], ('released past it, the row replies', facts)
    assert facts['rest']['transform'] == 'none' and facts['rest']['swipe'] == '' and facts['rest']['mark'] is None, facts


def test_a_rightward_swipe_on_a_message_stays_the_drawers(chromium_ui, tmp_path):
    page, client = booted(chromium_ui, tmp_path)
    finger = Finger(page)
    facts = {}
    try:
        row = page.evaluate(ROW, ASSISTANT)
        start = (row['x'] + 30, row['y'] + row['h'] / 2)
        finger.send('touchStart', *start)
        finger.path(start, (start[0] + 240, start[1] - 10), steps=12)
        facts['during'] = page.evaluate(ROW, ASSISTANT)
        finger.send('touchEnd')
        page.wait_for_timeout(450)
        facts['drawer'] = page.evaluate(DRAWER)
        facts['row'] = page.evaluate(ROW, ASSISTANT)
    finally:
        finger.close()
        evidence(tmp_path, 'row-swipe-rightward', facts)
        page.close()
        client.close()
    assert facts['during']['swipe'] == '' and facts['during']['mark'] is None, facts
    assert facts['drawer']['open'] is True, ('rightwards the drawer takes the finger', facts)
    assert facts['row']['staged'] == '', facts


@pytest.mark.parametrize('theme', ['light', 'dark'])
def test_a_long_press_lifts_the_row_over_the_dimmed_feed_until_its_menu_closes(chromium_ui, tmp_path, theme):
    page, client = booted(chromium_ui, tmp_path, theme)
    finger = Finger(page)
    facts = {'theme': page.evaluate('document.documentElement.dataset.theme')}
    try:
        row = page.evaluate(ROW, ASSISTANT)
        # A pixel of the feed beside the row, and one of the header.
        beside = (8, round(row['y'] + row['h'] / 2))
        header = (195, 12)
        rest = page.screenshot()
        finger.send('touchStart', row['x'] + row['w'] / 2, row['y'] + min(row['h'] / 2, 30))
        page.wait_for_timeout(250)
        facts['hold'] = page.evaluate(ROW, ASSISTANT)
        page.wait_for_timeout(500)
        facts['lift'] = page.evaluate(ROW, ASSISTANT)
        lifted = page.screenshot()
        page.screenshot(path=str(tmp_path / 'row-lift.png'))
        facts['beside'] = [luminance(rest, *beside), luminance(lifted, *beside)]
        facts['header'] = [luminance(rest, *header), luminance(lifted, *header)]
        finger.send('touchEnd')
        page.wait_for_timeout(200)
        facts['after_up'] = page.evaluate(ROW, ASSISTANT)
        # The feed scrolls under the lifted row (its own reflow, as a growing live card makes): the row has not moved,
        # so its menu stays (the lift's scale moved its corners, never its middle).
        page.evaluate("() => document.getElementById('chat-messages').dispatchEvent(new Event('scroll'))")
        page.wait_for_timeout(50)
        facts['reflowed'] = page.evaluate(ROW, ASSISTANT)
        finger.send('touchStart', *header)
        finger.send('touchEnd')
        page.wait_for_timeout(80)
        facts['closing'] = page.evaluate(ROW, ASSISTANT)
        page.wait_for_timeout(450)
        facts['rest'] = page.evaluate(ROW, ASSISTANT)
    finally:
        finger.close()
        evidence(tmp_path, f'row-long-press-lift-{theme}', facts)
        page.close()
        client.close()
    hold, lift = facts['hold'], facts['lift']
    assert hold['press'] == 'hold' and hold['menu'] == [], ('a still finger sinks the row first', facts)
    assert lift['press'] == 'lift' and 'reply' in lift['menu'], facts
    assert lift['transform'].startswith('matrix(1.01'), ('the row lifts', facts)
    assert facts['theme'] == theme, facts
    assert facts['beside'][1] < facts['beside'][0] * 0.85, ('the feed around it dims', facts)
    assert abs(facts['header'][1] - facts['header'][0]) < 3, ('the header stays bright', facts)
    assert facts['after_up']['press'] == 'lift' and facts['after_up']['menu'], ('lifted while the menu is open', facts)
    assert facts['reflowed']['menu'] == facts['after_up']['menu'], ('the feed scrolling under the lifted row keeps its menu', facts)
    # The press is the row's: the browser selects no word under it (iOS would also open its own edit menu over ours).
    assert hold['select'] == 'none' and lift['select'] == 'none', facts
    assert facts['closing']['menu'] == [] and facts['closing']['press'] == 'release', facts
    assert facts['rest']['press'] == '' and facts['rest']['transform'] == 'none', facts
    assert facts['rest']['select'] != 'none', ('at rest its words are selectable again', facts)


def test_inside_telegram_each_moment_plays_its_one_haptic(chromium_ui, tmp_path):
    """A Subagents value chosen: a tick each. A swipe past the reply point: one medium tap (none for pulling further). A long
    press that opens the menu: one medium tap. The drawer latched open by a finger: one light tap; ☰ is silent."""
    page, client = booted(chromium_ui, tmp_path, telegram=True)
    finger = Finger(page)
    played = lambda: page.evaluate('window.__haptics.slice()')  # noqa: E731
    clear = lambda: page.evaluate('window.__haptics.length = 0')  # noqa: E731
    facts = {}
    try:
        facts['host'] = page.evaluate('document.documentElement.dataset.ouroborosHost')
        facts['idle'] = played()
        page.route('**/api/owner/subagents-mode', lambda route: route.fulfill(json={
            'ok': True, 'project_id': '', 'subagents_mode': route.request.post_data_json['mode']}))
        page.locator('#page-chat .chat-subagents-head').click()
        page.locator('#page-chat .chat-subagents-strip .chat-seg[data-mode="swarm"]').click()
        page.locator('#page-chat .chat-subagents-strip .chat-seg[data-mode="off"]').click()
        facts['mode'] = played()
        page.keyboard.press('Escape')
        clear()
        row = page.evaluate(ROW, ASSISTANT)
        start = (row['x'] + row['w'] - 24, row['y'] + row['h'] / 2)
        finger.send('touchStart', *start)
        finger.path(start, (start[0] - 100, start[1] + 4), steps=8)
        finger.path((start[0] - 100, start[1] + 4), (start[0] - 140, start[1] + 6), steps=4)
        finger.send('touchEnd')
        page.wait_for_timeout(450)
        facts['swipe'] = played()
        page.keyboard.press('Escape')
        page.evaluate('() => document.activeElement?.blur?.()')
        clear()
        row = page.evaluate(ROW, ASSISTANT)
        finger.send('touchStart', row['x'] + row['w'] / 2, row['y'] + min(row['h'] / 2, 30))
        page.wait_for_timeout(700)
        finger.send('touchEnd')
        facts['press'] = played()
        finger.send('touchStart', 195, 12)
        finger.send('touchEnd')
        page.wait_for_timeout(450)
        clear()
        mid = row['y'] + row['h'] / 2
        finger.send('touchStart', 40, mid)
        finger.path((40, mid), (330, mid), steps=12)
        finger.send('touchEnd')
        page.wait_for_timeout(500)
        facts['drawer'] = page.evaluate(DRAWER)
        facts['latch'] = played()
        page.evaluate('window.__ouroLayers.pop()')
        page.wait_for_timeout(450)
        clear()
        page.locator('#page-chat .chat-toolbar-row > .composer-menu').click()
        page.wait_for_timeout(450)
        facts['menu_open'] = page.evaluate(DRAWER)
        facts['menu'] = played()
    finally:
        finger.close()
        evidence(tmp_path, 'row-gestures-haptics', facts)
        page.close()
        client.close()
    assert facts['host'] == 'telegram' and facts['idle'] == [], facts
    assert facts['mode'] == ['tick', 'tick'], ('Swarm, then Off: a tick each', facts)
    assert facts['swipe'] == ['medium'], ('one tap as the reply point is crossed', facts)
    assert facts['press'] == ['medium'], ('one tap as the long press opens the menu', facts)
    assert facts['drawer']['open'] is True and facts['latch'] == ['light'], ('the finger latched the drawer open', facts)
    assert facts['menu_open']['open'] is True and facts['menu'] == [], ('a press on ☰ is its own feedback', facts)
