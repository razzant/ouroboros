"""A message's own gestures on a phone (owner 19.10, 4A; docs/DESIGN.md "Replies").

Swiped left past the reply point, a saved message replies as its Reply does: the row follows the finger, the reply mark
fills and takes the accent, the composer stages the quote, and the row settles back; short of the point nothing is
staged, and a thumb that drifts down once the swipe is sideways never scrolls the feed. Rightwards the finger stays the
drawer's. A long press sinks the row while the finger holds, then lifts it over the dimmed feed while its menu is
open (the header stays bright), and the row is back at rest once the menu closes. Inside a work card the same finger
answers for the target the reader meant: a line with an address takes a colour step and offers its own Reply; a line
without one presses its card, which offers its own commands and no Reply, and its swipe does nothing. Saved rows come
from the real history endpoint (test_shell_interactions_browser.saved_history); synthetic CDP touch is Chromium's input
pipeline, not a claim about an iPhone.
"""
from __future__ import annotations

import io
import json

import pytest
from PIL import Image

from tests import test_subscription_setup_browser as setup_browser
from tests.test_design_system_gate_browser import seed_page
from tests.test_shell_interactions_browser import DRAWER, boot, chromium_ui, evidence, saved_history

subscription_ui = setup_browser.subscription_ui
pytestmark = [pytest.mark.ui_browser]
__all__ = ['chromium_ui', 'subscription_ui']

ROW = """sel => { const row = document.querySelector(sel); const r = row.getBoundingClientRect();
    const mark = document.querySelector('.chat-row-reply-mark');
    return { x: r.x, y: r.y, w: r.width, h: r.height, swipe: row.dataset.rowSwipe || '', press: row.dataset.rowPress || '',
        transform: getComputedStyle(row).transform, shadow: getComputedStyle(row).boxShadow,
        select: getComputedStyle(row).userSelect || getComputedStyle(row).webkitUserSelect,
        mark: mark ? { armed: mark.dataset.armed || '', k: Number(mark.style.getPropertyValue('--reply-k')), shown: mark.isConnected } : null,
        staged: document.getElementById('chat-input-reply')?.textContent || '', scrollTop: document.getElementById('chat-messages').scrollTop,
        menu: [...document.querySelectorAll('.chat-row-menu:not([data-leaving]) [data-row-action]')].map((item) => item.dataset.rowAction),
        label: document.querySelector('.chat-row-menu:not([data-leaving])')?.getAttribute('aria-label') || '' }; }"""
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


def scoped_history(tmp_path):
    """The shared saved history, answered as the serving core does: with the profile scope the page's own document
    proves (a scoped page refuses history that names none)."""
    client, _ = saved_history(tmp_path)

    def respond(route):
        response = client.get('/' + route.request.url.split('/', 3)[3])
        route.fulfill(status=response.status_code, content_type='application/json',
                      body=json.dumps({**response.json(), 'profile_scope': setup_browser.PREVIEW_SCOPE}))
    return client, respond


def booted(chromium_ui, tmp_path, theme='light', telegram=False):
    page = seed_page(chromium_ui, 390, 844, telegram=telegram)
    page.add_init_script(f"try {{ localStorage.setItem('ouroboros.theme', '{theme}'); }} catch (e) {{}}")
    if telegram:  # the Telegram stub's HapticFeedback records what the page asks Telegram to play
        page.add_init_script(RECORD_HAPTICS)
    client, respond = scoped_history(tmp_path)
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


# A running work with a nested helper: every row a finger can meet inside a work card. A system row the host stamped
# no address on (a placed card row without `row_address`) is a line nothing can quote.
LINEAGE = {'parent_task_id': 'root-1', 'root_task_id': 'root-1', 'delegation_role': 'subagent',
           'subagent_task_id': 'child-1', 'subagent_role': 'worker'}
CARD = '#chat-messages .chat-live-card[data-task-id="%s"]'
ROOT, CHILD = CARD % 'root-1', CARD % 'child-1'
LINES = {'addressed': CHILD + ' .chat-live-line[data-work-reply-row]',
         'bare_child': CHILD + ' .chat-live-line:not([data-work-reply-row])',
         'bare_root': ROOT + ' > .chat-live-timeline > .chat-live-line'}


def with_work_card(page):
    """The card and its helper as the live socket delivers them, both open."""
    def frame(**row):
        page.evaluate("row => window.__ouroWs.emit('chat', row)", {'type': 'chat', 'chat_id': 1, **row})
    frame(role='assistant', is_progress=True, narration=True, task_id='root-1', cancelable=True,
          content='Auditing the three archive shards.', ts='2026-11-24T10:00:01+00:00')
    frame(role='assistant', is_progress=True, narration=True, task_id='child-1', subagent_event='progress',
          content='Shard two matches its checksum; reading shard three next.', ts='2026-11-24T10:00:05+00:00',
          row_address='row:1@2026-11-24T10:00:05+00:00#child-n1', **LINEAGE)
    frame(role='system', system_type='host_progress', task_id='child-1', card_row='timeline', card_row_id='sys-1',
          card_row_revision=1, content='Worker paused for verification', ts='2026-11-24T10:00:09+00:00', **LINEAGE)
    page.wait_for_selector(ROOT)
    page.evaluate("""() => { for (const id of ['root-1', 'child-1']) {
        const card = document.querySelector(`.chat-live-card[data-task-id="${id}"]`);
        if (card && card.dataset.expanded !== '1') card.querySelector('[data-live-disclosure]').click(); } }""")
    for selector in LINES.values():
        page.wait_for_selector(selector, state='visible')
    page.wait_for_timeout(500)


ROWS = ("sels => { const read = " + ROW + "; return Object.fromEntries(Object.entries(sels)"
        ".map(([name, sel]) => [name, read(sel)])); }")


def rows(page, line):
    """The pressed line and the two cards around it, read in one beat: who sinks, who lifts."""
    return page.evaluate(ROWS, {'line': line, 'child': CHILD, 'root': ROOT})


def long_press(page, finger, line):
    """A finger held on `line` through the sink and the lift, then lifted off; the menu closed by a press on the feed.
    Returns each moment's rows and the frames at rest and while lifted."""
    box = page.evaluate(ROW, line)
    mid = box['y'] + box['h'] / 2
    rest = page.screenshot()
    finger.send('touchStart', box['x'] + 40, box['y'] + min(box['h'] / 2, 24))
    page.wait_for_timeout(250)
    held = rows(page, line)
    page.wait_for_timeout(500)
    lifted_rows, lifted = rows(page, line), page.screenshot()
    finger.send('touchEnd')
    page.wait_for_timeout(200)
    after_up = rows(page, line)
    finger.send('touchStart', 8, mid)
    finger.send('touchEnd')
    page.wait_for_timeout(450)
    sample = lambda x: [luminance(rest, x, mid), luminance(lifted, x, mid)]  # noqa: E731
    # Beside the line: its own halo, the card's margin just past it, the feed outside the card.
    return {'box': box, 'hold': held, 'lift': lifted_rows, 'open': after_up, 'closed': rows(page, line),
            'pixels': {'halo': sample(box['x'] - 2), 'card': sample(box['x'] - 10), 'feed': sample(8)}}, lifted


def swipe_left(page, finger, line):
    """A finger dragged 120 px to the left from the line's right end: the line mid-drag, and after release."""
    box = page.evaluate(ROW, line)
    start = (box['x'] + box['w'] - 30, box['y'] + min(box['h'] / 2, 24))
    finger.send('touchStart', *start)
    finger.path(start, (start[0] - 120, start[1] + 2), steps=10)
    during = page.evaluate(ROW, line)
    finger.send('touchEnd')
    page.wait_for_timeout(450)
    return {'during': during, 'after': page.evaluate(ROW, line)}


@pytest.mark.parametrize('theme', ['light', 'dark'])
def test_a_line_inside_a_work_card_answers_for_the_target_the_reader_meant(chromium_ui, tmp_path, theme):
    page, client = booted(chromium_ui, tmp_path, theme)
    finger = Finger(page)
    facts = {'theme': theme}
    try:
        with_work_card(page)
        for name, line in LINES.items():
            facts[name], frame = long_press(page, finger, line)
            (tmp_path / f'work-card-{name}-{theme}.png').write_bytes(frame)
        # A swipe replies only to a line that has an address; the same finger on a bare line moves nothing.
        facts['swipe'] = {name: swipe_left(page, finger, LINES[name]) for name in ('bare_child', 'bare_root', 'addressed')}
        # A right button on a bare line opens its card's menu, as a long press does.
        page.locator(LINES['bare_root']).click(button='right')
        facts['right_click'] = page.evaluate(ROW, LINES['bare_root'])
        page.keyboard.press('Escape')
    finally:
        finger.close()
        evidence(tmp_path, f'row-gestures-work-card-{theme}', facts)
        page.close()
        client.close()
    addressed, bare_root, bare_child = facts['addressed'], facts['bare_root'], facts['bare_child']
    # An addressed line: its own Reply alone, and a colour step for a press: nothing sinks or lifts, nothing dims.
    assert addressed['lift']['line']['menu'] == ['reply'] and addressed['lift']['line']['label'] == 'Message actions', facts
    assert [addressed[at]['line']['press'] for at in ('hold', 'lift', 'open', 'closed')] == ['hold', 'lift', 'lift', ''], facts
    for moment in ('hold', 'lift', 'open'):
        assert {name: addressed[moment][name]['transform'] for name in ('line', 'child', 'root')} == {
            'line': 'none', 'child': 'none', 'root': 'none'}, (moment, facts)
    assert addressed['lift']['child']['press'] == addressed['lift']['root']['press'] == '', facts
    halo, margin, feed = (addressed['pixels'][name] for name in ('halo', 'card', 'feed'))
    assert abs(halo[1] - halo[0]) >= 3, ('the pressed line takes the colour step', facts)
    assert abs(margin[1] - margin[0]) <= 2 and abs(feed[1] - feed[0]) <= 2, ('and nothing beside it dims', facts)
    # A bare line: its card's own doors and never a Reply to it; the card is what sinks and lifts.
    for fact, owner in ((bare_root, 'root'), (bare_child, 'child')):
        menu = fact['lift']['line']['menu']
        assert 'reply' not in menu and 'copy' in menu and fact['lift']['line']['label'] == 'Work actions', facts
        assert fact['hold'][owner]['press'] == 'hold' and fact['lift'][owner]['press'] == 'lift', facts
        assert fact['lift']['line']['press'] == '' and fact['closed'][owner]['press'] == '', facts
    assert bare_root['lift']['line']['menu'] == ['copy', 'pause', 'stop'], ("the root's commands, as its own card offers", facts)
    assert bare_root['lift']['child']['press'] == '' and bare_child['lift']['root']['press'] == '', facts
    assert bare_root['pixels']['feed'][1] < bare_root['pixels']['feed'][0] * 0.85, ('the lifted card dims the feed', facts)
    assert facts['right_click']['menu'] == ['copy', 'pause', 'stop'], ('a right button gives the same menu', facts)
    # The swipe: a bare line neither moves nor quotes its card; an addressed line follows the finger and quotes itself.
    for name in ('bare_child', 'bare_root'):
        moved = facts['swipe'][name]
        assert moved['during']['swipe'] == '' and moved['during']['mark'] is None, facts
        assert moved['during']['transform'] == 'none' and moved['after']['staged'] == '', facts
    drag = facts['swipe']['addressed']
    assert drag['during']['swipe'] == 'drag' and drag['during']['mark']['armed'] == 'true', facts
    assert 'Shard two matches its checksum' in drag['after']['staged'], ('released past the point, it quotes the line', facts)
