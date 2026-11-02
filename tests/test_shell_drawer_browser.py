"""The phone drawer and a room's menu (owner 19.10; docs/DESIGN.md "Shell").

The owner could pull the drawer only from the middle of the room, never from the left edge, and it slipped off the
finger. A sideways finger now opens it from the screen's very edge, from the header and from the composer's
controls, and it keeps following while the row under the finger is redrawn or a browser bar changes the window's
height. In a room (Main, an open Project) ☰ is the composer row's first control, left of Swarm (1A); the drawer's
× stays in its head (2A); with the keyboard up the room's menu stays above it. Uses the static-server/API fixture
(test_subscription_setup_browser); synthetic CDP touch is Chromium's input pipeline, not a claim about an iPhone.
"""
from __future__ import annotations

import io

import pytest
from PIL import Image

from tests import test_subscription_setup_browser as setup_browser
from tests.test_design_system_gate_browser import open_nav, seed_page
from tests.test_shell_interactions_browser import DRAWER, UNDER, boot, chromium_ui, evidence, open_project, shell_ui
from tests.test_ui_smoke_playwright import _install_controlled_visual_viewport

subscription_ui = setup_browser.subscription_ui
pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]
__all__ = ['chromium_ui', 'shell_ui', 'subscription_ui']


def swipe_with(page, start, end, steps=12, during=None):
    """A touch swipe from `start` to `end` with `during` run a third of the way, while the finger is still down."""
    cdp = page.context.new_cdp_session(page)
    try:
        for i in range(steps + 1):
            point = {'x': start[0] + (end[0] - start[0]) * i / steps,
                     'y': start[1] + (end[1] - start[1]) * i / steps, 'id': 1}
            cdp.send('Input.dispatchTouchEvent', {'type': 'touchStart' if i == 0 else 'touchMove', 'touchPoints': [point]})
            if during and i == steps // 3:
                during()
            page.wait_for_timeout(16)
        cdp.send('Input.dispatchTouchEvent', {'type': 'touchEnd', 'touchPoints': []})
        page.wait_for_timeout(450)
    finally:
        cdp.detach()


# A row of the room's feed the finger lands on (the stub's feed may be empty), remembered at pointerdown.
SEED_ROW = """feed => {
    const root = document.querySelector(feed);
    const row = document.createElement('div');
    row.className = 'test-drag-row';
    row.style.cssText = 'min-height:120px;padding:12px';
    row.textContent = 'A row the finger starts on';
    root.append(row);
    root.scrollTop = root.scrollHeight;
    const r = row.getBoundingClientRect();
    window.__downRow = null;
    document.addEventListener('pointerdown', (event) => { window.__downRow = event.target.closest('.test-drag-row'); },
        {capture: true, once: true});
    return {x: r.x + 30, y: r.y + r.height / 2};
}"""
REDRAW = "() => { const row = window.__downRow; if (!row) return false; row.replaceWith(row.cloneNode(true)); return !row.isConnected; }"


@pytest.mark.parametrize('room', ['main', 'project'])
def test_the_drawer_opens_from_the_edge_from_the_chrome_and_keeps_the_finger(chromium_ui, tmp_path, room):
    """The edge, the header and a composer capsule open the drawer (the page itself only pans up and down); it keeps
    following when a browser bar changes the height and when the row first under the finger is redrawn; a field
    keeps its finger."""
    page = seed_page(chromium_ui, 390, 844)
    under = [] if room == 'main' else ['project']
    facts = {'room': room}
    scope = '#page-chat' if room == 'main' else '#project-panel'
    feed_selector = '#chat-messages' if room == 'main' else '#project-panel .chat-messages'

    def reset():
        if page.evaluate(DRAWER)['open']:
            page.evaluate('window.__ouroLayers.pop()')
            page.wait_for_timeout(350)
    try:
        boot(page, chromium_ui)
        if room == 'project':
            open_project(page)
        header = page.locator(f'{scope} .app-page-header').first.bounding_box()
        feed = page.locator(feed_selector).first.bounding_box()
        field = page.locator(f'{scope} textarea').first.bounding_box()
        capsule = page.locator(f'{scope} .chat-toolbar-row .chat-subagents-mode').first.bounding_box()
        mid = feed['y'] + feed['height'] / 2
        starts = {
            'edge': (3, mid),
            'header': (header['x'] + header['width'] / 2, header['y'] + header['height'] / 2),
            'capsule': (capsule['x'] + capsule['width'] / 2, capsule['y'] + capsule['height'] / 2),
        }
        for name, (x, y) in starts.items():
            facts[f'{name}_under'] = page.evaluate(UNDER, [x, y])
            swipe_with(page, (x, y), (x + 240, y - 16))
            facts[name] = page.evaluate(DRAWER)
            reset()
        swipe_with(page, (field['x'] + 30, field['y'] + field['height'] / 2), (field['x'] + 270, field['y'] + field['height'] / 2))
        facts['field'] = page.evaluate(DRAWER)
        reset()
        # A browser bar (or the keyboard) changes only the height while the finger is down.
        swipe_with(page, (40, mid), (300, mid), during=lambda: page.set_viewport_size({'width': 390, 'height': 790}))
        facts['bar_moved'] = page.evaluate(DRAWER)
        page.set_viewport_size({'width': 390, 'height': 844})
        page.wait_for_timeout(300)
        reset()
        # The row the finger started on is redrawn mid-drag (a live card, a new message).
        start = page.evaluate(SEED_ROW, feed_selector)
        facts['redrawn'] = None
        swipe_with(page, (start['x'], start['y']), (start['x'] + 260, start['y']),
                   during=lambda: facts.__setitem__('redrawn', page.evaluate(REDRAW)))
        facts['row_redrawn'] = page.evaluate(DRAWER)
    finally:
        evidence(tmp_path, f'shell-drawer-edge-{room}', facts)
        page.close()
    opened = {'open': True, 'layers': under + ['drawer']}
    for name in ('edge', 'header', 'capsule', 'bar_moved', 'row_redrawn'):
        assert facts[name] == opened, (name, facts)
    assert facts['redrawn'] is True, ('the row under the finger really left the page mid-drag', facts)
    assert facts['field'] == {'open': False, 'layers': under}, ('a field keeps its finger', facts)


MENU = """scope => {
    const room = document.querySelector(scope);
    const shown = (el) => Boolean(el && el.getClientRects().length && getComputedStyle(el).visibility !== 'hidden');
    const menu = [...room.querySelectorAll('[data-mobile-nav-toggle]')].find(shown);
    const row = room.querySelector('.chat-toolbar-row');
    const first = row?.querySelector('.chat-subagents-mode');
    const box = (el) => el ? (({x, y, width, height}) => ({x, y, w: width, h: height}))(el.getBoundingClientRect()) : null;
    const b = menu?.getBoundingClientRect();
    const at = (dx, dy) => document.elementFromPoint(b.x + b.width / 2 + dx, b.y + b.height / 2 + dy)
        ?.closest('[data-mobile-nav-toggle]') === menu;
    const area = menu ? getComputedStyle(menu, '::after') : null;
    return {inRow: Boolean(menu && menu.parentElement === row && row.firstElementChild === menu),
            inHeader: [...room.querySelectorAll('.app-page-header [data-mobile-nav-toggle]')].some(shown),
            menu: box(menu), first: box(first),
            // The finger's target is the touch minimum, centred on the button: 21px from its centre is still the
            // button, 30px is not (not the composer around it either).
            inside: b ? [[0, -21], [0, 21], [-21, 0]].map(([dx, dy]) => at(dx, dy)) : [],
            outside: b ? [[0, -30], [0, 30], [-30, 0]].map(([dx, dy]) => at(dx, dy)) : [],
            area: area && {w: area.width, h: area.height}};
}"""


@pytest.mark.parametrize(('height', 'theme'), [(844, 'light'), (640, 'light'), (844, 'dark')])
@pytest.mark.parametrize('room', ['main', 'project'])
def test_a_rooms_menu_is_the_composers_first_control(shell_ui, tmp_path, room, height, theme):
    """In a room ☰ is the composer row's first control, left of Swarm, under the thumb: as tall as the capsules,
    the touch minimum centred on it as its finger target; the room's header has none. It opens the drawer, whose ×
    stays in its head. A page without a composer keeps ☰ in its header. The light theme is the owner's: its
    capsules have no glass, and a glass backdrop would itself hold the finger target on the button."""
    page = seed_page(shell_ui, 390, height)
    page.add_init_script(f"try {{ localStorage.setItem('ouroboros.theme', '{theme}'); }} catch (e) {{}}")
    facts = {'engine': shell_ui['engine'], 'room': room, 'height': height}
    scope = '#page-chat' if room == 'main' else '#project-panel'
    try:
        boot(page, shell_ui)
        facts['theme'] = page.evaluate('document.documentElement.dataset.theme')
        if room == 'project':
            open_project(page)
        page.wait_for_timeout(800)  # the composer's controls settle (composer_owner_controls.js) before the measure
        facts.update(page.evaluate(MENU, scope))
        page.locator(f'{scope} .chat-toolbar-row [data-mobile-nav-toggle]').tap()
        page.wait_for_function("() => window.__ouroLayers.top() === 'drawer'")
        page.wait_for_timeout(300)
        facts['close'] = page.locator('#nav-drawer-close').bounding_box()
        facts['menu_after_open'] = page.locator(f'{scope} .chat-toolbar-row [data-mobile-nav-toggle]').get_attribute('aria-expanded')
        page.evaluate('window.__ouroLayers.pop()')
        page.wait_for_timeout(300)
        if room == 'main':
            open_nav(page, 'settings')
            facts['settings_menu_in_header'] = page.evaluate(
                "() => [...document.querySelectorAll('#page-settings .app-page-header [data-mobile-nav-toggle]')].some(e => e.getClientRects().length)")
    finally:
        evidence(tmp_path, f'shell-room-menu-{room}-{height}-{theme}-{shell_ui["engine"]}', facts)
        page.close()
    assert facts['theme'] == theme, facts
    assert facts['inRow'] is True and facts['inHeader'] is False, facts
    assert facts['menu']['x'] + facts['menu']['w'] <= facts['first']['x'], ('left of the first capsule, Subagents', facts)
    assert facts['menu']['h'] == facts['first']['h'] == 32, ('as tall as the capsules', facts)
    assert facts['inside'] == [True, True, True] and facts['outside'] == [False, False, False], (
        'the touch minimum, centred on the button, is its finger target', facts)
    assert facts['area'] == {'w': '44px', 'h': '44px'}, facts
    assert facts['menu_after_open'] == 'true', facts
    assert facts['close']['y'] < 100, ('the drawer\'s × stays in its head', facts)
    if room == 'main':
        assert facts['settings_menu_in_header'] is True, ('a page without a composer keeps ☰ in its header', facts)


KEYBOARD = """scope => {
    const menu = document.querySelector(scope + ' .chat-toolbar-row [data-mobile-nav-toggle]').getBoundingClientRect();
    const field = document.querySelector(scope + ' textarea').getBoundingClientRect();
    return {open: document.body.classList.contains('keyboard-open'), menuBottom: menu.bottom, fieldBottom: field.bottom,
            vvh: getComputedStyle(document.documentElement).getPropertyValue('--vvh').trim()};
}"""


@pytest.mark.parametrize('room', ['main', 'project'])
def test_a_rooms_menu_stays_above_the_keyboard(shell_ui, tmp_path, room):
    """With the keyboard up (the visible viewport 500px of 844), the room's composer, and the ☰ in it, sits inside
    the visible part, a Project room's as well as Main's."""
    page = seed_page(shell_ui, 390, 844)
    _install_controlled_visual_viewport(page, 844)
    facts = {'engine': shell_ui['engine'], 'room': room}
    scope = '#page-chat' if room == 'main' else '#project-panel'
    try:
        boot(page, shell_ui)
        if room == 'project':
            open_project(page)
        page.locator(f'{scope} textarea').first.focus()
        page.evaluate('() => window.__setTestVisualViewportHeight(500)')
        page.wait_for_function("() => document.body.classList.contains('keyboard-open')", timeout=5_000)
        page.wait_for_timeout(300)
        facts.update(page.evaluate(KEYBOARD, scope))
    finally:
        evidence(tmp_path, f'shell-room-menu-keyboard-{room}-{shell_ui["engine"]}', facts)
        page.close()
    assert facts['open'] is True and facts['vvh'] == '500px', facts
    assert facts['menuBottom'] <= 500 and facts['fieldBottom'] <= 500, ('above the keyboard', facts)


# Every frame for `ms`, the scrim's opacity and hidden state, the drawer's x and the room card's x, from just before an
# action.
FRAMES = """ms => new Promise((resolve) => {
    const scrim = document.getElementById('nav-drawer-backdrop');
    const side = document.getElementById('primary-sidebar');
    const room = document.getElementById('content');
    const out = [];
    const start = performance.now();
    const step = () => {
        out.push({t: performance.now() - start, opacity: Number(getComputedStyle(scrim).opacity), hidden: scrim.hidden,
                  x: side.getBoundingClientRect().x, room: room.getBoundingClientRect().x,
                  settle: side.style.getPropertyValue('--drawer-settle')});
        if (performance.now() - start < ms) requestAnimationFrame(step); else resolve(out);
    };
    requestAnimationFrame(step);
})"""
FEED_TOP = "sel => document.querySelector(sel).scrollTop"
SCRIM_HELD = """() => { const s = document.getElementById('nav-drawer-backdrop');
    return {hidden: s.hidden, dragging: s.classList.contains('dragging'), opacity: Number(getComputedStyle(s).opacity)}; }"""


def record(page, ms):
    """Start sampling frames now; `collected` waits for them."""
    page.evaluate(f"ms => {{ window.__frames = ({FRAMES})(ms); }}", ms)


def collected(page):
    return page.evaluate("() => window.__frames")


def test_the_drawer_and_its_scrim_settle_smoothly_and_a_drifting_thumb_keeps_it(chromium_ui, tmp_path):
    """☰ pushes the room aside as a card on its curve while the drawer comes up from under it and its clear scrim
    comes with it; × runs it back, and the scrim leaves the page only after; a fling settles at the finger's speed; a
    thumb that drifts down halfway through a drag keeps the drawer, and the feed under it does not scroll (owner 19.10:
    it came off halfway, and nothing moved smoothly)."""
    page = seed_page(chromium_ui, 390, 844)
    facts = {}
    try:
        boot(page, chromium_ui)
        page.evaluate("() => { const f = document.querySelector('#chat-messages'); for (let i = 0; i < 40; i += 1) {"
                      " const r = document.createElement('div'); r.style.minHeight = '90px'; r.textContent = 'row ' + i;"
                      " f.append(r); } f.scrollTop = 600; }")
        record(page, 600)
        page.locator('#page-chat .chat-toolbar-row [data-mobile-nav-toggle]').click()
        opening = collected(page)
        facts['open'] = opening
        page.wait_for_timeout(200)
        record(page, 900)
        page.locator('#nav-drawer-close').click()
        closing = collected(page)
        facts['close'] = closing
        page.wait_for_timeout(300)
        # Closed with ×, then taken again by a finger at once and held past the close's own fade: the scrim stays.
        page.locator('#page-chat .chat-toolbar-row [data-mobile-nav-toggle]').click()
        page.wait_for_timeout(500)
        page.locator('#nav-drawer-close').click()
        page.wait_for_timeout(50)
        feed = page.locator('#chat-messages').bounding_box()
        y = feed['y'] + feed['height'] / 2
        cdp = page.context.new_cdp_session(page)
        try:
            for i, x in enumerate((40, 60, 120, 200, 260)):
                cdp.send('Input.dispatchTouchEvent', {'type': 'touchStart' if i == 0 else 'touchMove',
                                                      'touchPoints': [{'x': x, 'y': y, 'id': 1}]})
                page.wait_for_timeout(16)
            held = []
            for _ in range(12):  # 600 ms with the finger down, past the 420 ms the close's fade took
                page.wait_for_timeout(50)
                held.append(page.evaluate(SCRIM_HELD))
            facts['regrab_held'] = held
            cdp.send('Input.dispatchTouchEvent', {'type': 'touchEnd', 'touchPoints': []})
        finally:
            cdp.detach()
        page.wait_for_timeout(500)
        facts['regrab'] = page.evaluate(DRAWER)
        page.evaluate('window.__ouroLayers.pop()')
        page.wait_for_timeout(500)
        # A fling: the settle is set from the finger's speed and the drawer is still on its way a frame later.
        record(page, 500)
        swipe_with(page, (40, y), (240, y), steps=5)
        facts['fling'] = collected(page)
        facts['fling_state'] = page.evaluate(DRAWER)
        page.evaluate('window.__ouroLayers.pop()')
        page.wait_for_timeout(500)
        # A thumb that drifts down 120px over the second half of the drag. The drawer follows from where the axis
        # locked (Chromium drops the first move inside its touch slop), so the drag ends well past half open and the
        # release opens it as a slow one too: a CI runner sends these steps ~50 ms apart, too slow for a fling.
        top = page.evaluate(FEED_TOP, '#chat-messages')
        cdp = page.context.new_cdp_session(page)
        try:
            path = [(40 + 14 * i, y + (0 if i < 8 else 15 * (i - 8))) for i in range(17)]
            for i, (px, py) in enumerate(path):
                cdp.send('Input.dispatchTouchEvent', {'type': 'touchStart' if i == 0 else 'touchMove',
                                                      'touchPoints': [{'x': px, 'y': py, 'id': 1}]})
                page.wait_for_timeout(16)
            facts['drift_mid'] = page.evaluate(DRAWER)
            facts['drift_held'] = page.evaluate(PULLED)
            cdp.send('Input.dispatchTouchEvent', {'type': 'touchEnd', 'touchPoints': []})
        finally:
            cdp.detach()
        page.wait_for_timeout(500)
        facts['drift'] = page.evaluate(DRAWER)
        facts['feed_top'] = [top, page.evaluate(FEED_TOP, '#chat-messages')]
    finally:
        evidence(tmp_path, 'shell-drawer-motion', facts)
        page.close()
    between = [f for f in opening if 0.05 < f['opacity'] < 0.95]
    assert between, ('the scrim fades in, not at once', opening)
    assert any(-300 < f['x'] < -5 for f in opening), ('the drawer slides in', opening)
    assert opening[-1]['opacity'] == 1 and opening[-1]['x'] == 0, opening[-1]
    assert opening[0]['room'] == 0 and any(5 < f['room'] < 280 for f in opening), ('the room slides as the card', opening)
    assert abs(opening[-1]['room'] - 0.75 * 390) < 1, ('a quarter of the room stays', opening[-1])
    fading = [f for f in closing if not f['hidden'] and 0.05 < f['opacity'] < 0.95]
    assert fading, ('the scrim fades out while it is still on the page', closing)
    assert any(-330 < f['x'] < -5 for f in closing), ('the drawer slides out', closing)
    assert any(5 < f['room'] < 285 for f in closing) and closing[-1]['room'] == 0, ('the card comes back', closing)
    assert closing[-1]['hidden'] is True, ('and it leaves once it has faded', closing[-1])
    assert all(not f['hidden'] and f['dragging'] for f in facts['regrab_held']), (
        'a finger that takes the drawer during the close keeps its scrim, past the close\'s fade', facts['regrab_held'])
    assert facts['regrab']['open'] is True, facts
    settles = {f['settle'] for f in facts['fling'] if f['settle']}
    assert settles and all(s.endswith('ms') for s in settles), ('a fling settles at the finger\'s speed', facts['fling'])
    assert facts['fling_state']['open'] is True, facts
    assert facts['drift_held']['dragging'] and facts['drift_held']['p'] > 0.5, (
        'at the end of the drift the drawer is still under the finger, past half open', facts)
    assert facts['drift_mid']['open'] is False and facts['drift']['open'] is True, ('the drifting thumb keeps the drawer', facts)
    assert facts['feed_top'][0] == facts['feed_top'][1], ('the feed under the dragged drawer did not scroll', facts)


PULLED = """() => { const s = document.getElementById('primary-sidebar');
    return {x: s.getBoundingClientRect().x, p: Number(s.style.getPropertyValue('--drawer-p')), dragging: s.classList.contains('dragging'),
            surface: getComputedStyle(s).backgroundColor.match(/\\d+/g).slice(0, 3).map(Number)}; }"""


@pytest.mark.parametrize(('width', 'height'), [(390, 844), (719, 844), (667, 375)])
def test_a_pull_past_fully_open_shows_the_drawer_at_the_edge_not_the_room(chromium_ui, tmp_path, width, height):
    """Pulled past fully open the drawer gives like a rubber band, and its surface fills the gap at the screen's edge:
    the room never shows through, on a phone and on the widest stack layout (719 px, a landscape phone), where the
    drawer is 360 px and a pull reaches farthest. (The drawer is its own scroller and would clip a child or a
    pseudo-element there.)"""
    page = seed_page(chromium_ui, width, height)
    page.add_init_script("try { localStorage.setItem('ouroboros.theme', 'light'); } catch (e) {}")
    facts = {}
    try:
        boot(page, chromium_ui)
        page.locator('#page-chat .chat-toolbar-row [data-mobile-nav-toggle]').click()
        page.wait_for_function("() => document.getElementById('primary-sidebar').classList.contains('open')")
        page.wait_for_timeout(500)
        y = height // 2
        cdp = page.context.new_cdp_session(page)
        try:
            # Left first (the close direction locks the axis), then right to the screen's far edge, past fully open.
            for i, x in enumerate((100, 80, 160, 260, 380, width - 10)):
                cdp.send('Input.dispatchTouchEvent', {'type': 'touchStart' if i == 0 else 'touchMove',
                                                      'touchPoints': [{'x': x, 'y': y, 'id': 1}]})
                page.wait_for_timeout(16)
            facts['pulled'] = page.evaluate(PULLED)
            png = page.screenshot(clip={'x': 4, 'y': y + 20, 'width': 1, 'height': 1})
            facts['gap'] = list(Image.open(io.BytesIO(png)).convert('RGB').getpixel((0, 0)))
            # And back 100 px, still past fully open: the drawer follows the finger back (a rubber band, not a wall).
            cdp.send('Input.dispatchTouchEvent', {'type': 'touchMove', 'touchPoints': [{'x': width - 110, 'y': y, 'id': 1}]})
            page.wait_for_timeout(32)
            facts['back'] = page.evaluate(PULLED)
            # Then the finger rests and lets go: a release, not a fling back (drawer_drag.js reads no speed from a
            # finger that stopped longer than DRAG.sampleWindow, 100 ms, before it lifted). Lifted at once, the 100 px
            # stroke back is a fling whose speed depends on how long the 1x1 screenshot above took.
            page.wait_for_timeout(150)
            cdp.send('Input.dispatchTouchEvent', {'type': 'touchEnd', 'touchPoints': []})
        finally:
            cdp.detach()
        page.wait_for_timeout(600)
        facts['after'] = page.evaluate(DRAWER)
    finally:
        evidence(tmp_path, f'shell-drawer-rubber-band-{width}x{height}', facts)
        page.close()
    assert facts['pulled']['dragging'] and facts['pulled']['p'] > 1 and facts['pulled']['x'] > 8, (
        'the drawer gives past fully open', facts)
    assert max(abs(a - b) for a, b in zip(facts['gap'], facts['pulled']['surface'])) <= 4, (
        'the gap at the edge shows the drawer\'s surface, not the dimmed room', facts)
    assert facts['back']['p'] > 1 and facts['back']['x'] < facts['pulled']['x'] - 0.5, (
        'pulled back, the drawer gives back: every pixel of the finger still moves it', facts)
    assert facts['after']['open'] is True, ('released, it settles back to fully open', facts)


CARD = """sel => {
    const room = document.querySelector(sel);
    const veil = room.querySelector(':scope > .room-veil');
    const side = document.getElementById('primary-sidebar');
    const at = (x, y) => document.elementFromPoint(x, y);
    const cs = getComputedStyle(room);
    return {x: room.getBoundingClientRect().x, transform: cs.transform, radius: cs.borderTopLeftRadius,
            shadow: cs.boxShadow, veil: Number(getComputedStyle(veil).opacity), drawerUnder: side.contains(at(120, 400)),
            quarter: at(350, 400)?.id || '', open: side.classList.contains('open'),
            p: Number(side.style.getPropertyValue('--drawer-p') || 0),
            project: document.getElementById('project-panel').classList.contains('open'),
            projectScrim: getComputedStyle(document.getElementById('project-panel-backdrop')).display};
}"""


def held(page, sel, xs, y):
    """A finger put down at the first x and moved through the rest, still down: the room card where the finger has it.
    Then the finger lifts and the drawer settles."""
    cdp = page.context.new_cdp_session(page)
    try:
        for i, x in enumerate(xs):
            cdp.send('Input.dispatchTouchEvent', {'type': 'touchStart' if i == 0 else 'touchMove',
                                                  'touchPoints': [{'x': x, 'y': y, 'id': 1}]})
            page.wait_for_timeout(16)
        facts = page.evaluate(CARD, sel)
        cdp.send('Input.dispatchTouchEvent', {'type': 'touchEnd', 'touchPoints': []})
    finally:
        cdp.detach()
    page.wait_for_timeout(600)
    return facts


@pytest.mark.parametrize('room', ['main', 'project'])
def test_the_room_is_the_card_the_drawer_pushes_aside(chromium_ui, tmp_path, room):
    """Owner 19.10 (the iOS board, 1A): ☰ pushes the room (Main, or the Project open over it) right by three quarters of
    the screen as a card with rounded corners, a shadow and faded words; the drawer under it takes the finger, the
    quarter left showing is the way back, and a finger moves the card and its veil with it: opening the drawer,
    taking the open one back from the drawer or from the quarter, and pulling it past open, where the card gives like
    a rubber band. (While the drawer was open the open card's rule outranked the finger's, and both stood still.)"""
    page = seed_page(chromium_ui, 390, 844)
    sel = '#content' if room == 'main' else '#project-panel'
    menu = f'{"#page-chat" if room == "main" else "#project-panel"} .chat-toolbar-row [data-mobile-nav-toggle]'
    facts = {'room': room}

    def reopen():
        if not page.evaluate(CARD, sel)['open']:
            page.locator(menu).click()
            page.wait_for_timeout(700)
    try:
        boot(page, chromium_ui)
        if room == 'project':
            open_project(page)
        facts['closed'] = page.evaluate(CARD, sel)
        page.locator(menu).click()
        page.wait_for_timeout(700)
        facts['open'] = page.evaluate(CARD, sel)
        page.mouse.click(350, 400)  # the room's quarter
        page.wait_for_timeout(700)
        facts['back'] = page.evaluate(CARD, sel)
        # A finger halfway, still down: the card and the veil are where the finger is.
        feed = page.locator(f'{sel} .chat-messages, {sel} #chat-messages').first.bounding_box()
        y = feed['y'] + feed['height'] / 2
        facts['held'] = held(page, sel, (40, 60, 100, 140, 180), y)
        # The open drawer taken back 90 px, from the drawer and from the quarter, and pulled past open.
        reopen()
        facts['closing'] = held(page, sel, (200, 190, 170, 150, 130, 110), y)
        reopen()
        facts['closing_quarter'] = held(page, sel, (370, 360, 340, 320, 300, 280), y)
        reopen()
        facts['past'] = held(page, sel, (100, 80, 160, 260, 380), y)
        facts['after'] = page.evaluate(CARD, sel)
    finally:
        evidence(tmp_path, f'shell-room-card-{room}', facts)
        page.close()
    closed, opened, back, held_open = facts['closed'], facts['open'], facts['back'], facts['held']
    assert closed['x'] == 0 and closed['radius'] == '0px' and closed['veil'] == 0, ('closed, the room is the screen', closed)
    assert abs(opened['x'] - 0.75 * 390) < 1 and opened['radius'] == '44px' and opened['shadow'] != 'none', opened
    assert abs(opened['veil'] - 0.42) < 0.01, ('its words are faded', opened)
    assert opened['drawerUnder'] and opened['quarter'] == 'nav-drawer-backdrop', ('the drawer takes the finger, the quarter closes', opened)
    assert opened['projectScrim'] == 'none', opened
    assert back['open'] is False and back['x'] == 0 and back['project'] is (room == 'project'), ('the quarter brought it back', back)
    assert 40 < held_open['x'] < 0.75 * 390 - 40 and 0.02 < held_open['veil'] < 0.4, (
        'the card and its veil follow the finger', held_open)
    # The open drawer under a finger: the card moves 1:1 by the drawer's own number and its veil fades by it.
    for name in ('closing', 'closing_quarter'):
        taken = facts[name]
        assert taken['open'] is True and 0.6 < taken['p'] < 0.85, (f'setup: {name} holds the open drawer partway', taken)
        assert abs(taken['x'] - taken['p'] * 0.75 * 390) < 1, (f'{name}: the card follows the finger back', taken)
        assert abs(taken['veil'] - 0.42 * taken['p']) < 0.01, (f'{name}: the veil fades with it', taken)
    past = facts['past']
    assert past['p'] > 1 and abs(past['x'] - past['p'] * 0.75 * 390) < 1 and past['x'] > opened['x'] + 8, (
        'past open the card gives like a rubber band', past)
    assert abs(past['veil'] - 0.42) < 0.01, past
    assert facts['after']['open'] is True and abs(facts['after']['x'] - opened['x']) < 1, (
        'released past open, it settles back open', facts['after'])


PINCH_FACTS = """() => ({scale: visualViewport.scale, vvWidth: visualViewport.width, innerWidth, innerHeight,
    vvh: getComputedStyle(document.documentElement).getPropertyValue('--vvh').trim(),
    app: Math.round(document.getElementById('app').getBoundingClientRect().height)})"""


def pinch(page):
    """Chromium's synthetic two-finger pinch to x2 at the room's centre."""
    cdp = page.context.new_cdp_session(page)
    try:
        cdp.send('Input.synthesizePinchGesture', {'x': 195, 'y': 420, 'scaleFactor': 2, 'relativeSpeed': 800,
                                                  'gestureSourceType': 'touch'})
    finally:
        cdp.detach()
    page.wait_for_timeout(600)


def test_a_pinch_magnifies_the_room_and_never_shrinks_it_or_moves_the_drawer(chromium_ui, tmp_path):
    """A two-finger pinch only magnifies (docs/DESIGN.md "Touch and containment"): the room keeps its height, where
    the stack layout's height once followed the shrinking visual viewport and halved at ×2 with the composer adrift,
    and a sideways finger then pans the magnified page instead of opening the drawer. Chromium's synthetic pinch, not
    a claim about an iPhone."""
    page = seed_page(chromium_ui, 390, 844)
    facts = {}
    try:
        boot(page, chromium_ui)
        page.wait_for_timeout(300)
        facts['before'] = page.evaluate(PINCH_FACTS)
        pinch(page)
        facts['pinched'] = page.evaluate(PINCH_FACTS)
        swipe_with(page, (60, 420), (320, 428))
        facts['drawer'] = page.evaluate(DRAWER)
    finally:
        evidence(tmp_path, 'shell-pinch', facts)
        page.close()
    before, pinched = facts['before'], facts['pinched']
    if pinched['scale'] == 1:
        # Linux headless Chromium applied no synthetic touch pinch at all (CI 40682936207); a plain page tells an engine
        # that cannot pinch from a room that refuses to be magnified.
        probe = chromium_ui['new_page'](viewport={'width': 390, 'height': 844}, has_touch=True)
        try:
            probe.set_content('<meta name="viewport" content="width=device-width, initial-scale=1.0">'
                              '<div style="height:3000px"></div>')
            pinch(probe)
            plain = probe.evaluate('visualViewport.scale')
        finally:
            probe.close()
        if plain == 1:
            pytest.skip('this Chromium applies no synthetic touch pinch, not even to a plain page')
    assert pinched['scale'] > 1.5 and pinched['vvWidth'] < pinched['innerWidth'] - 1, facts  # the pinch took
    assert before['app'] == before['innerHeight'] and pinched['app'] == before['app'], facts
    assert pinched['vvh'] == before['vvh'], facts
    assert not facts['drawer']['open'], facts
