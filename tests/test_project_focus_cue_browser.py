"""An open Project's transcript holds the keys; its keyboard focus is two corners, never a ring (DESIGN §3 Focus).

The real SPA through the static-server/API fixture (test_subscription_setup_browser), driven by real Playwright mouse
and keyboard input in both engines at the owner's window (1165 x 851 CSS px, DPR 2, dark, the desktop's integrated
title bar played in as tests/test_window_chrome_browser.py does). Every check reads the rendered frame: the focus
colour at the four arms of the two corners, plain room between them, and no full-height ring at the panel's edge.
Synthetic rooms, no runtime. Headless Chromium hides scrollbars; WebKit draws the 6px lane.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest
from PIL import Image

from ouroboros.launcher_window_chrome import DOCUMENT_SCRIPT, HANDLER
from tests import test_subscription_setup_browser as setup_browser
from tests._work_projection_shared import work_fields
from tests.test_design_system_gate_browser import rest

subscription_ui = setup_browser.subscription_ui
pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]

W, H, DPR = 1165, 851, 2
PROJECTS = [dict(id=pid, name=name, chat_id=chat, lifecycle='active', visible_revision=0) for pid, name, chat in (
    ('nova', 'Nova', 501), ('atlas', 'Atlas: a second room', 502), ('empty', 'An empty room', 503),
    ('notice', 'A room whose history failed', 504))]


def _room(prefix, count):
    rows = []
    for i in range(count):
        ask = f'{prefix}: why such a small simulation? I asked for many large ones ({i}).'
        say = f'{prefix} answer {i}. The short pairs were only a slice; before them came long trajectories. ' * 3
        rows += [{'role': 'user', 'text': ask, 'content': ask, 'ts': f'2026-10-28T06:{i:02d}:00Z',
                  'client_message_id': f'{prefix}-{i}'},
                 {'role': 'assistant', 'text': say, 'content': say, 'ts': f'2026-10-28T06:{i:02d}:30Z'}]
    return rows


HISTORY = {1: _room('Main', 6), 501: _room('Nova', 8), 502: _room('Atlas', 8), 503: []}
PORT = """Object.defineProperty(window, 'webkit', {configurable: true, value: {messageHandlers: {%s: {
    postMessage: (message) => (window.__shellMessages = window.__shellMessages || []).push(message)}}}});""" % HANDLER

FACTS = """(pid) => {
    const panel = document.getElementById('project-panel'), feed = document.getElementById(`pchat-${pid}-messages`);
    const a = document.activeElement, probe = document.createElement('i');
    probe.style.color = 'var(--focus-accent-border)';
    document.body.append(probe);
    const color = getComputedStyle(probe).color.match(/[\\d.]+/g).slice(0, 3).map(Number);
    probe.remove();
    const facts = {active: a === document.body ? 'body' : a?.id || a?.dataset?.projectId || a?.tagName || null,
        activeIsFeed: Boolean(feed) && a === feed, activeVisible: a?.matches?.(':focus-visible') ?? null, color,
        quiet: Boolean(feed?.classList.contains('is-quiet-focus')), layers: window.__ouroLayers?.ids?.() ?? null,
        panelOpen: !panel.hidden && panel.classList.contains('open')};
    if (!feed || !feed.getClientRects().length) return facts;
    const r = feed.getBoundingClientRect(), s = getComputedStyle(feed);
    const rows = [...feed.querySelectorAll('.chat-bubble')].map((b) => b.getBoundingClientRect())
        .filter((b) => b.width && b.bottom > r.top && b.top < r.bottom);
    return {...facts, focusVisible: feed.matches(':focus-visible'), outline: s.outlineStyle,
        note: Boolean(feed.parentElement.querySelector('.chat-panel-statusbar .chat-history-status:not([hidden])')),
        feed: {x: r.x, y: r.y, width: r.width, height: r.height, clientWidth: feed.clientWidth,
               offsetWidth: feed.offsetWidth, padTop: parseFloat(s.paddingTop), padLeft: parseFloat(s.paddingLeft),
               padRight: parseFloat(s.paddingRight),
               scrollTop: feed.scrollTop, scrolls: feed.scrollHeight > feed.clientHeight + 1},
        rowsLeft: rows.length ? Math.min(...rows.map((b) => b.left)) : null,
        rowsRight: rows.length ? Math.max(...rows.map((b) => b.right)) : null};
}"""


@pytest.fixture(params=['webkit', 'chromium'])
def cue_ui(request, monkeypatch, tmp_path):
    monkeypatch.setenv('OUROBOROS_UI_BROWSER_ENGINE', request.param)
    ui = request.getfixturevalue('subscription_ui')
    ui['engine'] = request.param
    ui['out'] = Path(os.environ.get('OUROBOROS_UI_EVIDENCE_DIR') or tmp_path) / 'project-focus-cue' / request.param
    ui['out'].mkdir(parents=True, exist_ok=True)
    return ui


def room_page(ui, *, width=W, height=H, scheme='dark', panel_width=575, activities=()):
    page = ui['new_page'](viewport={'width': width, 'height': height}, device_scale_factor=DPR,
                          color_scheme=scheme, has_touch=False)
    page.set_default_timeout(15_000)
    origin = urlparse(ui['url']).netloc
    page.route('**/*', lambda r: r.fallback() if urlparse(r.request.url).netloc == origin else r.abort())
    state = {'sha': 'focus-cue', 'supervisor_ready': True, 'active_chat_activities': list(activities), 'projects': PROJECTS,
             'project_chat_ids': [p['chat_id'] for p in PROJECTS],
             **work_fields(activities, {}, results={}, fences={}, complete=True)}
    page.route('**/api/state*', lambda route: route.fulfill(json=state))
    page.route('**/api/projects*', lambda route: route.fulfill(json={'projects': PROJECTS}))

    def history(route):
        chat_id = int(parse_qs(urlparse(route.request.url).query).get('chat_id', ['1'])[0])
        if chat_id == 504:  # a failed read: its note takes the Project's status bar
            return route.fulfill(status=500, json={'error': 'history unavailable'})
        route.fulfill(json={'messages': HISTORY.get(chat_id, []), 'progress': [], 'has_more': False,
                            'next_cursor': None, 'page_cursor': 'recent',
                            'window': {'complete': True, 'truncated_by': []}})

    page.route('**/api/chat/history*', history)
    prefs = {'sidebar_width': 370, 'project_panel_width': panel_width}
    page.route('**/api/ui/preferences*', lambda route: route.fulfill(
        json=prefs if route.request.method == 'GET' else {'ok': True}))
    page.add_init_script(PORT)
    page.add_init_script(DOCUMENT_SCRIPT)
    page.goto(ui['url'] + '/')
    page.wait_for_selector('#nav-projects-list [data-project-id="nova"]', state='attached')
    page.evaluate("""() => {
        const doc = (window.__shellMessages || []).filter((m) => m.type === 'declare').at(-1)?.doc;
        if (doc !== undefined) window.__ouroWindowChrome({doc, top: 52, start: 73});
    }""")
    page.wait_for_timeout(300)
    rest(page)
    return page


def center(page, selector):
    box = page.locator(selector).first.bounding_box()
    assert box, f'{selector} has no box'
    return box['x'] + box['width'] / 2, box['y'] + box['height'] / 2


def wait_open(page, pid):
    page.wait_for_function("""(pid) => document.getElementById('project-panel').classList.contains('open')
        && window.__ouroLayers?.top?.() === 'project' && document.getElementById(`pchat-${pid}-messages`)
        && !document.getElementById(`panel-pchat-${pid}`).hidden""", arg=pid)
    page.wait_for_timeout(450)
    rest(page)


def mouse_open(page, pid):
    x, y = center(page, f'#nav-projects-list [data-project-id="{pid}"]')
    page.mouse.move(x, y, steps=4)
    page.mouse.click(x, y)
    wait_open(page, pid)


def tab_to(page, engine, predicate, back=False):
    """Walk the focus with the keyboard (WebKit reaches buttons with Option+Tab, as Safari does) until it holds."""
    key = ('Alt+' if engine == 'webkit' else '') + ('Shift+Tab' if back else 'Tab')
    for _ in range(120):
        if page.evaluate(predicate):
            return True
        page.keyboard.press(key)
    return page.evaluate(predicate)


def press_text(page, pid):
    point = page.evaluate("""(pid) => {
        const feed = document.getElementById(`pchat-${pid}-messages`), r = feed.getBoundingClientRect();
        for (let y = r.top + 120; y < r.bottom - 160; y += 12) for (let x = r.left + 60; x < r.right - 60; x += 24) {
            const el = document.elementFromPoint(x, y);
            if (el && feed.contains(el) && el.closest('.chat-bubble .message') && !el.closest('button, a')) return {x, y};
        }
        return null;
    }""", pid)
    assert point, 'no message text to press'
    page.mouse.click(point['x'], point['y'])


def arms(facts):
    """A point on each arm of the two corners (CSS px), from the laid-out feed: 11px outside the rows' box (its content
    box, which ends where the drawn scrollbar begins) and 4px above it. A horizontal arm is read beside its vertical
    one: the bar's soft shade over the feed's top darkens it toward the room's middle."""
    feed = facts['feed']
    top = feed['y'] + feed['padTop'] - 4
    left = feed['x'] + feed['padLeft'] - 11
    right = feed['x'] + feed['clientWidth'] - feed['padRight'] + 11
    return {'left-vertical': (left + 1, top + 7), 'left-horizontal': (left + 3, top + 1),
            'right-vertical': (right - 1, top + 7), 'right-horizontal': (right - 3, top + 1)}


def look(ui, page, pid, name, full=False):
    """The rendered frame's answer: which arms wear the focus colour, whether the room between them is plain, and
    whether a column at the panel's edge carries a ring (the engine's blue or the focus colour) down most of it."""
    settle_frames(page)
    facts = page.evaluate(FACTS, pid)
    path = ui['out'] / f'{name}.png'
    page.screenshot(path=str(path))
    image = Image.open(path).convert('RGB')
    near = lambda p: wears(p, facts['color'])  # noqa: E731
    pixel = lambda x, y: image.getpixel((int(x * DPR), int(y * DPR)))  # noqa: E731
    if facts.get('feed'):
        feed = facts['feed']
        facts['arms'] = {key: near(pixel(*xy)) for key, xy in arms(facts).items()}
        between = [(feed['x'] + feed['clientWidth'] / 2, feed['y'] + feed['padTop'] - 3),
                   (feed['x'] + 4, feed['y'] + feed['height'] / 2), (feed['x'] + feed['offsetWidth'] - 10,
                                                                      feed['y'] + feed['height'] / 2)]
        facts['plain_between'] = not any(near(pixel(x, y)) for x, y in between)
        blue = lambda p: p[2] - max(p[0], p[1]) > 25 and p[2] > 60  # noqa: E731
        rows = range(int(feed['y'] * DPR), int((feed['y'] + feed['height']) * DPR), 2)
        facts['edge_ring'] = any(sum(blue(image.getpixel((x, y))) or near(image.getpixel((x, y))) for y in rows)
                                 > 0.5 * len(rows) for x in range(int((feed['x'] - 3) * DPR), int((feed['x'] + 3) * DPR)))
        top = int(max(0, feed['y'] + feed['padTop'] - 30) * DPR)
        image.crop((int((feed['x'] - 8) * DPR), top, int((feed['x'] + feed['width']) * DPR), top + 100 * DPR)).save(
            ui['out'] / f'{name}-top.png')
    if not full:
        path.unlink()
    (ui['out'] / f'{name}.json').write_text(json.dumps(facts, indent=1))
    print(ui['engine'], name, json.dumps({k: facts.get(k) for k in (
        'active', 'focusVisible', 'quiet', 'arms', 'plain_between', 'edge_ring', 'outline', 'note')}))
    return facts


def wears(pixel, color):
    """The focus colour, or that colour under the shade the floating bar casts on the feed's top (a uniform darkening
    of up to a quarter): never the room, a row or a bubble."""
    k = pixel[0] / color[0] if color[0] else 0
    return 0.75 <= k <= 1.05 and all(abs(a - k * b) <= 12 for a, b in zip(pixel, color))


def settle_frames(page):
    page.evaluate('() => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)))')


def corners(facts):
    assert facts['activeIsFeed'] and facts['focusVisible'] and not facts['quiet'], facts
    assert all(facts['arms'].values()) and facts['plain_between'], ('two corners, nothing between them', facts)
    assert facts['outline'] == 'none' and not facts['edge_ring'], ('no ring round the room', facts)


def quiet(facts, focused=True):
    assert facts['activeIsFeed'] is focused, facts
    assert not any(facts['arms'].values()) and facts['outline'] == 'none' and not facts['edge_ring'], facts


def test_a_pointer_keeps_the_transcript_quiet_and_its_keys_show_the_corners(cue_ui):
    ui = cue_ui
    page = room_page(ui)
    mouse_open(page, 'nova')
    opened = look(ui, page, 'nova', '01-open-by-mouse', full=True)
    quiet(opened)
    page.keyboard.press('ArrowUp')  # the room lands at its newest message: up scrolls
    shown = look(ui, page, 'nova', '02-arrow-up', full=True)
    assert shown['feed']['scrollTop'] < opened['feed']['scrollTop'], 'the key scrolls the transcript'
    corners(shown)
    press_text(page, 'nova')
    pressed = look(ui, page, 'nova', '03-press-on-text', full=True)
    quiet(pressed)
    # A modifier alone, a screenshot's Cmd+Shift and a copy reach the page and act on nothing here.
    for key in ('Shift', 'Alt', 'Control', 'Meta', 'Meta+c', 'Control+c'):
        page.keyboard.press(key)
    page.keyboard.down('Meta')
    page.keyboard.down('Shift')
    page.keyboard.up('Shift')
    page.keyboard.up('Meta')
    quiet(look(ui, page, 'nova', '04-modifiers-and-shortcuts'))
    page.keyboard.press('PageUp')
    again = look(ui, page, 'nova', '05-page-up-after-press')
    assert again['feed']['scrollTop'] < pressed['feed']['scrollTop']
    corners(again)
    press_text(page, 'nova')
    page.keyboard.press('Meta+ArrowDown' if ui['engine'] == 'webkit' else 'Control+End')  # a scroll with a modifier
    corners(look(ui, page, 'nova', '06-modified-scroll-key'))
    # The panel's edge is a pointer's: a drag on it quiets the corners, and the next key finds them at the new width.
    x, y = center(page, '#project-panel > .resize-handle-left')
    page.mouse.move(x, y, steps=4)
    page.mouse.down()
    for step in range(1, 9):
        page.mouse.move(x - 10 * step, y, steps=2)
    page.mouse.up()
    page.mouse.move(x - 300, y - 100, steps=4)
    rest(page)
    dragged = look(ui, page, 'nova', '07-after-edge-drag')
    quiet(dragged)
    assert dragged['feed']['width'] > shown['feed']['width'] + 40, dragged
    page.keyboard.press('ArrowUp')
    corners(look(ui, page, 'nova', '08-key-after-drag', full=True))
    # The scrim is Main beside the docked panel: a press there closes the Project and its focus returns, quietly.
    main_x = page.evaluate("() => (document.getElementById('primary-sidebar').getBoundingClientRect().right"
                           " + document.getElementById('project-panel').getBoundingClientRect().left) / 2")
    page.mouse.click(main_x, H * 0.5)
    page.wait_for_function("() => document.getElementById('project-panel').hidden")
    closed = page.evaluate(FACTS, 'nova')
    assert closed['layers'] == [] and closed['active'] != 'pchat-nova-messages', closed
    assert page.evaluate("() => getComputedStyle(document.getElementById('chat-messages')).outlineStyle") == 'none'


def test_the_keyboard_opens_with_the_corners_and_moves_them_with_the_room(cue_ui):
    ui, engine = cue_ui, cue_ui['engine']
    page = room_page(ui)
    assert tab_to(page, engine, "() => document.activeElement?.dataset?.projectId === 'nova'")
    page.keyboard.press('Enter')
    wait_open(page, 'nova')
    corners(look(ui, page, 'nova', '11-open-by-enter', full=True))
    # The next control takes the focus with its own ring; the transcript's corners go with the focus.
    page.keyboard.press('Alt+Tab' if engine == 'webkit' else 'Tab')
    left = look(ui, page, 'nova', '12-tab-to-the-next-control')
    quiet(left, focused=False)
    assert left['activeVisible'] and left['active'] != 'body', left
    # Another Project opened from its row by Enter: its room takes the focus, the corners and the keys.
    assert tab_to(page, engine, "() => document.activeElement?.dataset?.projectId === 'atlas'", back=True)
    page.keyboard.press('Enter')
    wait_open(page, 'atlas')
    switched = look(ui, page, 'atlas', '13-enter-on-another-project', full=True)
    corners(switched)
    page.keyboard.press('PageUp')
    assert look(ui, page, 'atlas', '14-page-up-in-it')['feed']['scrollTop'] < switched['feed']['scrollTop']
    page.keyboard.press('Escape')
    page.wait_for_function("() => document.getElementById('project-panel').hidden")
    back = page.evaluate(FACTS, 'atlas')
    assert back['active'] == 'atlas' and back['activeVisible'] and back['layers'] == [], (
        'focus returns to the row that opened this room', back)
    # Focus taken out to Main's field beside the docked room stays there when Escape closes the room.
    page.keyboard.press('Enter')
    wait_open(page, 'atlas')
    assert tab_to(page, engine, "() => document.activeElement?.id === 'chat-input'", back=True)
    page.keyboard.press('Escape')
    page.wait_for_function("() => document.getElementById('project-panel').hidden")
    kept = page.evaluate(FACTS, 'atlas')
    assert kept['active'] == 'chat-input' and kept['layers'] == [], ('the field keeps its focus', kept)


GUTTER_CASES = ['empty', 'notice', 'narrow', 'phone', 'light', 'phone-light', 'phone-empty']


@pytest.mark.parametrize('case', GUTTER_CASES)
def test_the_corners_keep_to_the_gutters(cue_ui, case):
    ui = cue_ui
    pid = {'empty': 'empty', 'notice': 'notice', 'phone-empty': 'empty'}.get(case, 'nova')
    phone = case.startswith('phone')
    page = room_page(ui, width=390 if phone else W, height=844 if phone else H,
                     scheme='light' if 'light' in case else 'dark', panel_width=320 if case == 'narrow' else 575)
    if phone:
        page.evaluate("p => window.dispatchEvent(new CustomEvent('ouro:open-project', {detail: {project: p}}))",
                      next(project for project in PROJECTS if project['id'] == pid))
        wait_open(page, pid)
        page.keyboard.press('ArrowUp')
    else:
        mouse_open(page, pid)
        page.keyboard.press('ArrowUp' if pid == 'nova' else 'ArrowDown')
    facts = look(ui, page, pid, f'2{GUTTER_CASES.index(case) + 1}-{case}', full=True)
    corners(facts)
    feed = facts['feed']
    # Each corner is 8px wide; its outer edge keeps 3px from the feed's edge or the scrollbar it draws, its inner edge
    # 3px from the rows' box, and the rows that pass keep to that box: no arm is ever under a bubble.
    outer_left, outer_right = feed['x'] + feed['padLeft'] - 11, feed['x'] + feed['clientWidth'] - feed['padRight'] + 11
    assert outer_left >= feed['x'] + 3 and outer_right <= feed['x'] + feed['clientWidth'] - 3, ('clear of the edge', facts)
    if facts['rowsLeft'] is not None:
        assert outer_left + 8 + 3 <= facts['rowsLeft'] and facts['rowsRight'] <= outer_right - 8 - 3, (
            'rows pass beside the corners', facts)
    if case == 'notice':
        assert facts['note'] and feed['padTop'] == 8, ('the note above the feed, the corners under it', facts)
    if phone:
        assert page.evaluate("document.documentElement.dataset.layout") == 'stack'


def test_a_pointer_reopening_after_a_keyboard_close_stays_quiet(cue_ui):
    """The keyboard opens a room and Escape gives its row the focus back; a mouse press on that row opens it again
    quietly, though the engine may still count the row's focus as the keyboard's. A Project reference passes the
    same intent: pressed, its room waits for a key; a key's click sets no quiet state and leaves it to the engine."""
    ui, engine = cue_ui, cue_ui['engine']
    page = room_page(ui)
    assert tab_to(page, engine, "() => document.activeElement?.dataset?.projectId === 'nova'")
    page.keyboard.press('Enter')
    wait_open(page, 'nova')
    corners(look(ui, page, 'nova', '31-open-by-enter'))
    page.keyboard.press('Escape')
    page.wait_for_function("() => document.getElementById('project-panel').hidden")
    assert page.evaluate("() => document.activeElement?.dataset?.projectId") == 'nova'
    mouse_open(page, 'nova')
    reopened = look(ui, page, 'nova', '32-mouse-reopen-after-escape', full=True)
    quiet(reopened)
    page.keyboard.press('ArrowUp')
    corners(look(ui, page, 'nova', '33-key-after-the-mouse-reopen'))
    for pointer, name in ((True, '34-reference-pressed'), (False, '35-reference-by-key')):
        page.keyboard.press('Escape')
        page.wait_for_function("() => document.getElementById('project-panel').hidden")
        # Mount the real production control in Main, then use actual input: exercise click detail through the
        # Project-reference producer, its event, the app listener and the newly focused transcript together.
        page.evaluate("""async (project) => {
            document.getElementById('focus-test-reference')?.remove();
            const { projectReference } = await import('/static/modules/project_reference.js');
            const button = projectReference(project);
            button.id = 'focus-test-reference';
            document.getElementById('chat-input').parentElement.prepend(button);
        }""", PROJECTS[1])
        if pointer:
            page.locator('#focus-test-reference').click()
        else:
            assert tab_to(page, engine, "() => document.activeElement?.id === 'focus-test-reference'")
            page.keyboard.press('Enter')
        wait_open(page, 'atlas')
        facts = look(ui, page, 'atlas', name)
        assert facts['activeIsFeed'] and facts['quiet'] is pointer, facts
        if pointer:
            quiet(facts)
        else:
            corners(facts)


@pytest.mark.parametrize('forced', [False, True])
def test_keyboard_return_from_the_composer_shows_the_reading_focus(cue_ui, monkeypatch, forced):
    """Use the real work list and Back to reading pill, not a substitute focus() button."""
    ui, engine = cue_ui, cue_ui['engine']
    if forced and engine == 'webkit':
        pytest.skip('Playwright WebKit does not emulate forced colors')
    tid = 'focus-work'
    monkeypatch.setitem(HISTORY, 501, [*HISTORY[501], {
        'role': 'assistant', 'direction': 'out', 'is_progress': True, 'task_id': tid,
        'text': 'Checking the reading focus', 'ts': '2026-10-28T06:09:00Z',
        'suggested_name': 'Check reading focus', 'cancelable': True}])
    activity = {'activity_id': tid, 'kind': 'managed_task', 'phase': 'working', 'chat_id': 501,
                'task_attempt': 1, 'started_at': 1793167740, 'project_id': 'nova',
                'suggested_name': 'Check reading focus'}
    page = room_page(ui, activities=[activity])
    if forced:
        page.emulate_media(forced_colors='active')
    mouse_open(page, 'nova')
    feed = page.locator('#pchat-nova-messages')
    feed.evaluate('el => { el.scrollTop = el.scrollHeight * 0.3; }')
    rest(page)
    page.locator('#project-panel .rw-trigger').focus()
    page.keyboard.press('ArrowDown')
    page.wait_for_function("() => document.activeElement?.closest('.rw-row')?.dataset.id === 'focus-work'")
    page.keyboard.press('Enter')
    back = page.locator('#project-panel .rw-back')
    back.wait_for(state='visible')
    page.locator('#pchat-nova-input').click()
    page.keyboard.type('draft')
    assert tab_to(page, engine, "() => document.activeElement?.matches('#project-panel .rw-back')", back=True)
    page.keyboard.press('Enter')
    back.wait_for(state='hidden')
    facts = look(ui, page, 'nova', f'back-from-composer-{forced}', full=True)
    assert facts['activeIsFeed'] and facts['focusVisible'] and not facts['quiet'], facts
    if forced:
        assert feed.evaluate("el => getComputedStyle(el).outlineStyle") == 'solid'
    else:
        corners(facts)


def test_phone_project_switch_returns_to_the_surviving_main_menu(cue_ui):
    ui, engine = cue_ui, cue_ui['engine']
    page = room_page(ui, width=390, height=844)
    for scope, pid in (('#page-chat', 'nova'), ('#project-panel', 'atlas')):
        menu = f'{scope} .chat-toolbar-row [data-mobile-nav-toggle]'
        assert tab_to(page, engine, f"() => document.activeElement === document.querySelector('{menu}')")
        page.keyboard.press('Enter')
        page.wait_for_function("() => window.__ouroLayers?.top?.() === 'drawer'")
        assert tab_to(page, engine, f"() => document.activeElement?.dataset?.projectId === '{pid}'")
        page.keyboard.press('Enter')
        wait_open(page, pid)
        corners(look(ui, page, pid, f'phone-switch-{pid}', full=True))
    page.keyboard.press('Escape')
    page.wait_for_function("() => document.getElementById('project-panel').hidden")
    rest(page)
    assert page.evaluate("""() => document.activeElement === document.querySelector(
        '#page-chat .chat-toolbar-row [data-mobile-nav-toggle]')"""), page.evaluate(FACTS, 'atlas')
    page.screenshot(path=str(ui['out'] / 'phone-switch-closed.png'))
    # Returning the keys must restore an operable control, not just a stored reference.
    page.keyboard.press('Enter')
    page.wait_for_function("() => window.__ouroLayers?.top?.() === 'drawer'")


def test_forced_colors_preserves_a_visible_keyboard_focus(subscription_ui, tmp_path):
    ui = {**subscription_ui, 'engine': 'chromium', 'out': tmp_path}
    page = room_page(ui)
    page.emulate_media(forced_colors='active')
    mouse_open(page, 'nova')
    page.keyboard.press('ArrowUp')
    feed = page.locator('#pchat-nova-messages')
    style = feed.evaluate("""el => {
        const s = getComputedStyle(el);
        return {outline: s.outlineStyle, width: s.outlineWidth, color: s.outlineColor,
                background: s.backgroundColor, focused: el === document.activeElement};
    }""")
    out = Path(os.environ.get('OUROBOROS_UI_EVIDENCE_DIR') or tmp_path) / 'forced-colors'
    out.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(out / 'keyboard.png'))
    (out / 'keyboard.json').write_text(json.dumps(style))
    assert style['focused'] and style['outline'] == 'solid' and style['width'] == '2px', style
    assert style['color'] != style['background'], style
    press_text(page, 'nova')
    assert feed.evaluate("el => getComputedStyle(el).outlineStyle") == 'none'
    page.screenshot(path=str(out / 'pointer.png'))
