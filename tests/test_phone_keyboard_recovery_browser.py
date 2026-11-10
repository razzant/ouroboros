"""The phone composer and its keyboard on the production SPA, Main and a Project, in WebKit and Chromium.

The visual viewport, touches and pinch are emulated signals into the page's own owners (app.js updateVvh/focus
recovery, chat_reading_position.js gestures, chat_input_sizing.js); none of this is a native iPhone keyboard, IME,
autocorrect panel or Telegram client. Uses the static-server/API fixture; never starts the runtime.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests._phone_room_shared import VIEWPORT, facts, history, shot, telegram_sdk
from tests.test_design_system_gate_browser import rest, seed_page, settle
from tests.test_shell_interactions_browser import open_project, shell_ui
from tests.test_subscription_setup_browser import subscription_ui

pytestmark = [pytest.mark.ui_browser]
__all__ = ['shell_ui', 'subscription_ui']

LONG_LINE = 'A saved conversation line that wraps across the narrow phone column. '
WEB = Path(__file__).resolve().parents[1] / 'web'


def phone(ui, *, telegram=True):
    page = seed_page({**ui, 'new_page': lambda **options: ui['new_page'](is_mobile=True, **options)}, 390, 844,
                     telegram=telegram)
    page.add_init_script(VIEWPORT)
    return page


def telegram_phone(ui, *, fullscreen):
    """The phone inside Telegram as the sidecar serves it (the host stamp) with the SDK stand-in: fullscreen granted under
    the device's and Telegram's top (47 + 46) and its sides, or refused (expanded: Telegram's own bar is outside the
    page, no insets)."""
    page = phone(ui, telegram=False)
    page.add_init_script(telegram_sdk(fullscreen='grant' if fullscreen else 'refuse',
                                      insets={'top': 47, 'bottom': 34, 'left': 20, 'right': 12} if fullscreen else {},
                                      content={'top': 46} if fullscreen else {}))
    html = (WEB / 'index.html').read_text(encoding='utf-8').replace('<html', '<html data-ouroboros-host="telegram"', 1)
    page.route(ui['url'] + '/', lambda route: route.fulfill(content_type='text/html', body=html))
    return page


def room_field(page, project_room):
    if project_room:
        open_project(page)
    field = page.locator('textarea.chat-input:visible' if project_room else '#chat-input')
    field.wait_for(state='visible')
    return field, field.get_attribute('id')


GEOMETRY = """id => {
    const field = document.getElementById(id), area = document.getElementById(id.replace(/input$/, 'input-area'));
    const feed = document.getElementById(id.replace(/input$/, 'messages'));
    return {areaBottom: area.getBoundingClientRect().bottom, field: field.getBoundingClientRect().height,
        top: field.getBoundingClientRect().top, bottom: field.getBoundingClientRect().bottom,
        cap: parseFloat(getComputedStyle(field).maxHeight), feed: feed.clientHeight,
        sideways: feed.scrollWidth - feed.clientWidth, focused: document.activeElement === field,
        keyboard: document.body.classList.contains('keyboard-open'),
        folded: 'folded' in area.querySelector('.chat-input-wrap').dataset,
        body: document.body.getBoundingClientRect().height};
}"""


@pytest.mark.parametrize('theme', ['light', 'dark'])
@pytest.mark.parametrize('project_room', [False, True], ids=['main', 'project'])
def test_a_short_phone_viewport_keeps_the_composer_and_room_and_asks_for_writing_assistance(shell_ui, theme, project_room):
    page = phone(shell_ui)
    page.route('**/api/chat/history**', lambda route: route.fulfill(json=history([LONG_LINE * 2] * 12)))
    page.add_init_script(f"localStorage.setItem('ouroboros.theme', '{theme}')")
    page.goto(shell_ui['url'])
    page.locator('#chat-messages[data-history-hydrated="true"]').wait_for()
    field, field_id = room_field(page, project_room)
    assert (field.get_attribute('autocorrect'), field.get_attribute('autocapitalize'),
            field.get_attribute('spellcheck')) == ('on', 'sentences', 'true')
    # A short viewport with no focused field is no keyboard, in a WebView that shrinks for its own reasons.
    page.evaluate('window.__viewport(250)')
    settle(page)
    assert not page.evaluate("document.body.classList.contains('keyboard-open')")
    page.evaluate('window.__viewport(844)')
    settle(page)
    field.focus()
    page.evaluate('window.__viewport(250)')
    page.wait_for_function("document.body.classList.contains('keyboard-open')")
    measured = {}
    for label, text in [('short', 'A normal message'), ('long', '\n'.join(['A long draft with wrapped words'] * 12)),
                        ('short-again', 'Short again')]:
        field.fill(text)
        settle(page)
        settle(page)
        measured[label] = page.evaluate(GEOMETRY, field_id)
        geometry = measured[label]
        # A browser's keyboard room, with no top inset, is no short one: the capsule row stays.
        assert geometry['areaBottom'] <= 251 and geometry['feed'] > 0 and not geometry['folded'], geometry
        assert geometry['field'] <= geometry['cap'] + 1 and geometry['sideways'] <= 1, geometry
        if label == 'long':
            shot(page, f'keyboard-250-long-draft-{"project" if project_room else "main"}-{theme}-{shell_ui["engine"]}',
                 {'x': 0, 'y': 0, 'width': 390, 'height': 250})
    assert measured['long']['cap'] == 62, 'a quarter of the 250 px viewport caps the field while the keyboard is up'
    assert measured['long']['field'] > measured['short']['field']
    name = f'keyboard-250-{"project" if project_room else "main"}-{theme}-{shell_ui["engine"]}'
    facts(name, measured)
    shot(page, name, {'x': 0, 'y': 0, 'width': 390, 'height': 250})
    page.evaluate('window.__viewport(844)')
    field.blur()
    settle(page)
    rested = page.evaluate(GEOMETRY, field_id)
    assert not rested['keyboard'] and rested['cap'] == 140 and rested['body'] == 844, rested
    assert field.input_value() == 'Short again'
    shot(page, f'room-after-keyboard-{"project" if project_room else "main"}-{theme}-{shell_ui["engine"]}')


# The keyboard's room as a finger meets it: every control of the bar, the composer's row and the field reached at its
# centre inside the visible height (nothing covers it), the bar's controls below Telegram's band, the feed's height and
# whether its newest row shows, and whether the capsule row is folded, its capsules out of sight.
FIT = """id => {
    const field = document.getElementById(id), area = document.getElementById(id.replace(/input$/, 'input-area'));
    const feed = document.getElementById(id.replace(/input$/, 'messages'));
    const room = feed.closest('.project-panel') || document.getElementById('page-chat');
    const bar = room.querySelector('.chat-page-header, .project-panel-bar');
    const box = el => el.getBoundingClientRect();
    const shown = el => el.getClientRects().length > 0 && getComputedStyle(el).visibility !== 'hidden';
    const reached = el => { const r = box(el), x = (r.left + r.right) / 2, y = (r.top + r.bottom) / 2;
        const hit = y < visualViewport.height ? document.elementFromPoint(x, y) : null;
        return Boolean(hit && el.contains(hit)); };
    const name = el => el.getAttribute('aria-label') || el.title || el.textContent.trim().slice(0, 24) || el.className;
    const barControls = [...bar.querySelectorAll('button')].filter(shown);
    // A closed capsule's levels are not controls yet: its head is (composer_owner_controls.js).
    const controls = [...barControls, ...area.querySelectorAll(
        '.chat-toolbar-row button:not(.chat-capsule-body *), .chat-text-row button')].filter(shown);
    // The newest message read as words: its text's line boxes (Range client rects) whole inside the feed, where neither
    // the bar nor the composer lies in this column; and its foot, the time under it.
    const inFeed = r => r.width > 0 && r.height > 0 && r.top >= box(feed).top - 0.5 && r.bottom <= box(feed).bottom + 0.5;
    const message = [...feed.querySelectorAll('.chat-bubble .message')].filter(shown).at(-1);
    const lines = new Set();
    for (const walk = document.createTreeWalker(message || feed, NodeFilter.SHOW_TEXT); message && walk.nextNode();) {
        const range = document.createRange();
        range.selectNodeContents(walk.currentNode);
        for (const r of range.getClientRects()) if (inFeed(r)) lines.add(Math.round(r.top));
    }
    const time = message?.closest('.chat-bubble')?.querySelector('.msg-time');
    const wrap = area.querySelector('.chat-input-wrap');
    return {viewport: visualViewport.height, room: box(room).height, barTop: Math.min(...barControls.map(el => box(el).top)),
        feed: feed.clientHeight, words: lines.size, foot: Boolean(time) && inFeed(box(time)),
        field: box(field).height, cap: parseFloat(getComputedStyle(field).maxHeight), send: box(area.querySelector('.chat-send-inline')).bottom,
        composer: box(area).bottom, controls: controls.length, unreached: controls.filter(el => !reached(el)).map(name),
        barUnreached: barControls.filter(el => !reached(el)).map(name),
        folded: 'folded' in wrap.dataset, focused: document.activeElement === field,
        capsules: [...wrap.querySelectorAll('.chat-toolbar-row .chat-capsule-head')].filter(shown).length,
        keyboard: document.body.classList.contains('keyboard-open'), sideways: document.scrollingElement.scrollWidth - innerWidth};
}"""


def keyboard_room(ui, project_room, fullscreen):
    """Main or a Project inside Telegram with a long draft focused and the keyboard rising to 508 px: a keyboard rises
    over frames, so a cap read late must not lose the newest row on the way."""
    page = telegram_phone(ui, fullscreen=fullscreen)
    page.route('**/api/chat/history**', lambda route: route.fulfill(json=history([LONG_LINE * 2] * 12)))
    page.goto(ui['url'])
    page.wait_for_function('mode => document.documentElement.dataset.telegramFullscreen === mode',
                           arg='active' if fullscreen else 'expanded')
    page.locator('#chat-messages[data-history-hydrated="true"]').wait_for()
    field, field_id = room_field(page, project_room)
    draft = '\n'.join(['A long message draft to be written'] * 15)
    field.fill(draft)
    field.focus()
    page.evaluate('window.__viewport(508)')
    page.wait_for_function("document.body.classList.contains('keyboard-open')")
    for _ in range(3):
        settle(page)
    return page, field, field_id, draft


def still(page):
    """Settled frames, then the room at rest: the capsule row's fold has run its motion (style.css, 300 ms)."""
    for _ in range(3):
        settle(page)
    rest(page)
    settle(page)


# The composer's "+" menu opened by a finger with the keyboard up (composer.js: the pointer path keeps the field's focus):
# where its sheet stands against the screen's top inset and its "+", whether the field kept the focus, and its last
# item scrolled into the sheet's own view.
PLUS_MENU = """id => {
    const field = document.getElementById(id), area = document.getElementById(id.replace(/input$/, 'input-area'));
    const plus = area.querySelector('[data-composer-plus]').getBoundingClientRect();
    const menu = document.querySelector('body > .chat-plus-menu'), sheet = menu.getBoundingClientRect();
    const items = [...menu.querySelectorAll('[role=menuitem]')];
    items.at(-1).scrollIntoView({block: 'nearest'});
    const last = items.at(-1).getBoundingClientRect();
    return {top: sheet.top, bottom: sheet.bottom, plusTop: plus.top, items: items.length,
        lastShows: last.top >= sheet.top - 0.5 && last.bottom <= sheet.bottom + 0.5,
        focused: document.activeElement === field, keyboard: document.body.classList.contains('keyboard-open')};
}"""


@pytest.mark.parametrize('fullscreen', [False, True], ids=['expanded', 'fullscreen'])
@pytest.mark.parametrize('project_room', [False, True], ids=['main', 'project'])
def test_a_short_keyboard_under_telegrams_fullscreen_band_keeps_history_the_composer_and_every_control(
        shell_ui, project_room, fullscreen):
    """Astra R2's counterexample: a 250 px visual viewport under Telegram fullscreen's top (device 47 + Telegram 46) put
    a long draft's Send at 265 px, below the keyboard, and left the feed 0 px. With the keyboard up the room is a column
    (style.css "Keyboard-open layout"): the field takes what the room's measured chrome and 60 px of feed leave, never
    under one line, and a room too short for that drops its air and folds the capsule row (owner 9.11), never another
    control, so its feed keeps a line of the newest message's words and its foot. Expanded is the control case, and so
    is fullscreen's roomier keyboard. Typing, a dismissal and a return, the room growing and shrinking again, and a
    pinch keep the draft and the composition."""
    page, field, field_id, draft = keyboard_room(shell_ui, project_room, fullscreen)
    measured = {'rising': page.evaluate(FIT, field_id)}
    page.evaluate('window.__viewport(250)')
    still(page)
    name = f'keyboard-250-{"fullscreen" if fullscreen else "expanded"}-{"project" if project_room else "main"}-{shell_ui["engine"]}'
    measured['keyboard'] = page.evaluate(FIT, field_id)
    shot(page, name, {'x': 0, 'y': 0, 'width': 390, 'height': 250})
    # "+" opens above itself, below the screen's top inset (this PR's fullscreen puts Telegram's Close and ⋯ there),
    # and the field keeps its keyboard; Escape closes only the menu.
    page.locator(f'#{field_id}').locator('xpath=ancestor::*[contains(@class,"chat-input-wrap")]') \
        .locator('[data-composer-plus]').click()
    page.locator('body > .chat-plus-menu').wait_for()
    # Measured once it has grown in (ui.css ui-popup-in scales it for a moment).
    page.locator('body > .chat-plus-menu').evaluate('m => Promise.all(m.getAnimations().map(a => a.finished.catch(() => null)))')
    measured['plus'] = page.evaluate(PLUS_MENU, field_id)
    shot(page, f'{name}-plus', {'x': 0, 'y': 0, 'width': 390, 'height': 250})
    page.keyboard.press('Escape')
    page.locator('body > .chat-plus-menu').wait_for(state='detached')
    settle(page)
    # Typing at the draft's end: the words go in, the field scrolls inside its cap, nothing else moves.
    field.evaluate('e => e.setSelectionRange(e.value.length, e.value.length)')
    page.keyboard.type(' and more')
    settle(page)
    settle(page)
    measured['typed'] = page.evaluate(FIT, field_id)
    # The keyboard goes down and comes back: the resting cap, then the same composition again.
    page.evaluate('window.__viewport(844)')
    field.blur()
    still(page)
    measured['rested'] = page.evaluate(FIT, field_id)
    field.focus()
    page.evaluate('window.__viewport(250)')
    page.wait_for_function("document.body.classList.contains('keyboard-open')")
    still(page)
    measured['again'] = page.evaluate(FIT, field_id)
    # The room grows with the keyboard still up (a smaller keyboard), then shrinks again.
    page.evaluate('window.__viewport(508)')
    still(page)
    measured['grown'] = page.evaluate(FIT, field_id)
    shot(page, f'{name}-grown', {'x': 0, 'y': 0, 'width': 390, 'height': 508})
    page.evaluate('window.__viewport(250)')
    still(page)
    measured['shrunk'] = page.evaluate(FIT, field_id)
    # A pinch only magnifies: the room keeps the height and the composition it had.
    page.evaluate('window.__pinch()')
    settle(page)
    settle(page)
    measured['pinched'] = page.evaluate(FIT, field_id)
    facts(name, measured)
    band = 93 if fullscreen else 0
    for state in ('rising', 'grown'):
        fit = measured[state]
        # A roomier keyboard, under the band too, keeps the capsule row and a line of the newest message's words.
        assert fit['words'] >= 1 and fit['foot'] and fit['unreached'] == [] and fit['composer'] <= 508, (state, fit)
        assert fit['keyboard'] and fit['focused'] and not fit['folded'] and fit['capsules'] == 3, (state, fit)
    menu = measured['plus']
    assert menu['top'] >= band + 8 - 0.5 and menu['bottom'] <= menu['plusTop'] - 4 + 0.5, f'"+" clears the band: {menu}'
    assert menu['focused'] and menu['keyboard'] and menu['items'] >= 1 and menu['lastShows'], menu
    for state in ('keyboard', 'typed', 'again', 'shrunk', 'pinched'):
        fit = measured[state]
        assert fit['room'] == 250 and fit['composer'] <= 250 and fit['send'] <= 250, (state, fit)
        # Fullscreen's room is short: its capsule row folds, ☰ beside "+", Subagents, Context and Effort out of sight
        # (owner 9.11); expanded keeps it. Every control in sight (the bar's Panic and a Project's ×, ☰, the capsules
        # when they show, "+" and Send) is reached, none covered.
        assert fit['folded'] == fullscreen and fit['capsules'] == (0 if fullscreen else 3), (state, fit)
        assert fit['keyboard'] and fit['focused'] and fit['unreached'] == [], (state, fit)
        assert fit['controls'] >= ((6 if project_room else 5) if fullscreen else (7 if project_room else 6)), (state, fit)
        assert fit['barTop'] >= band, f'Telegram keeps its Close and ⋯ band: {state} {fit}'
        # A line of the newest message's words in view (its text's line boxes), and its foot.
        assert fit['foot'] and fit['words'] >= 1, (state, fit)
        assert 33 <= fit['field'] <= fit['cap'] + 1 and fit['sideways'] <= 0, (state, fit)
    assert measured['again'] == measured['typed'], 'the same room composes the same way again'
    assert measured['shrunk'] == measured['again'], 'and again once it has grown and shrunk'
    assert measured['pinched'] == measured['shrunk'], 'a pinch moves nothing'
    rested = measured['rested']
    assert not rested['keyboard'] and rested['cap'] == 140 and rested['field'] == 140, rested
    assert not rested['folded'] and rested['capsules'] == 3, f'the keyboard down brings the row back: {rested}'
    assert field.input_value() == draft + ' and more', 'the draft stays whole'
    if not fullscreen:
        assert measured['keyboard']['cap'] == 62 and measured['keyboard']['feed'] == 60, \
            'expanded keeps the quarter cap and the history it had'


# An open capsule as the room turns short: whether the row folded, the open capsule in sight, the field's focus and the
# keyboard.
CAPSULE = """id => {
    const wrap = document.getElementById(id.replace(/input$/, 'input-area')).querySelector('.chat-input-wrap');
    const open = wrap.querySelector('.chat-toolbar-row [data-open="true"]');
    return {folded: 'folded' in wrap.dataset,
        open: Boolean(open) && open.getClientRects().length > 0 && getComputedStyle(open).visibility !== 'hidden',
        focused: document.activeElement === document.getElementById(id),
        keyboard: document.body.classList.contains('keyboard-open')};
}"""


@pytest.mark.parametrize('height', [250, 340])
@pytest.mark.parametrize('project_room', [False, True], ids=['main', 'project'])
def test_an_open_capsule_in_a_low_keyboard_room_keeps_the_bar_in_reach_and_folds_the_row_once_closed(
        shell_ui, project_room, height):
    """The short room's fold (owner 9.11) never takes an open capsule away: Effort opened with the keyboard down stays
    open in its row when the field takes the focus without a press (a staged reply does that) and the keyboard comes up
    under Telegram fullscreen's band. Its hint passes under the room's bar, so the bar's controls stay reached, in the
    short room (250 px) and just above it (340). A press on the field closes it, the field keeping the focus: the short
    room then folds the row with no scroll, and in both a line of the newest message's words shows beside the long draft."""
    page, field, field_id, draft = keyboard_room(shell_ui, project_room, fullscreen=True)
    page.evaluate('window.__viewport(844)')
    field.blur()
    still(page)
    area = '#' + re.sub(r'input$', 'input-area', field_id)
    page.locator(f'{area} .chat-effort-range .chat-capsule-head').click()
    page.locator(f'{area} .chat-toolbar-row [data-open="true"]').wait_for()
    field.focus()
    page.evaluate(f'window.__viewport({height})')
    page.wait_for_function("document.body.classList.contains('keyboard-open')")
    still(page)
    name = f'keyboard-{height}-fullscreen-open-capsule-{"project" if project_room else "main"}-{shell_ui["engine"]}'
    measured = {'open': {**page.evaluate(CAPSULE, field_id), **page.evaluate(FIT, field_id)}}
    shot(page, name, {'x': 0, 'y': 0, 'width': 390, 'height': height})
    field.click()  # a press outside the capsule, on the field that holds the focus
    page.locator(f'{area} .chat-toolbar-row [data-open="true"]').wait_for(state='detached')
    still(page)
    measured['closed'] = {**page.evaluate(CAPSULE, field_id), **page.evaluate(FIT, field_id)}
    shot(page, f'{name}-closed', {'x': 0, 'y': 0, 'width': 390, 'height': height})
    facts(name, measured)
    opened, closed = measured['open'], measured['closed']
    assert (opened['folded'], opened['open'], opened['focused'], opened['keyboard']) == (False, True, True, True), opened
    # The capsule's hint never covers the room's bar: Panic (and a Project's ×) reached at their centres, Send in view.
    assert opened['barUnreached'] == [] and opened['barTop'] >= 93 and opened['send'] <= height, opened
    assert (closed['open'], closed['focused'], closed['keyboard']) == (False, True, True), closed
    # Closed in the short room, the row folds again with no scroll; just above it the row stays.
    assert closed['folded'] == (height == 250) and closed['unreached'] == [] and closed['send'] <= height, closed
    # Folded or not, the feed keeps a line of the newest message's words and its foot (60 px at least, style.css).
    assert closed['words'] >= 1 and closed['foot'] and closed['capsules'] == (0 if height == 250 else 3), closed
    assert field.input_value() == draft, 'the draft stays whole'


def test_a_desk_window_as_low_as_the_short_room_never_folds_the_row(shell_ui):
    """The desk's control: a window as low as the short room, its field focused, is no keyboard room."""
    page = seed_page(shell_ui, 1280, 250)
    page.route('**/api/chat/history**', lambda route: route.fulfill(json=history([LONG_LINE * 2] * 12)))
    page.goto(shell_ui['url'])
    page.locator('#chat-messages[data-history-hydrated="true"]').wait_for()
    page.locator('#chat-input').focus()
    still(page)
    state = page.evaluate("""() => ({keyboard: document.body.classList.contains('keyboard-open'),
        folded: 'folded' in document.querySelector('#chat-input-area .chat-input-wrap').dataset,
        layout: document.documentElement.dataset.layout})""")
    facts(f'desk-250-{shell_ui["engine"]}', state)
    assert not state['folded'] and not state['keyboard'], state


SWIPE = """({id, touches}) => {
    const feed = document.getElementById(id);
    const send = (type, points) => {
        const event = new Event(type, {bubbles: true});
        Object.defineProperty(event, 'touches', {value: points.map(([clientX, clientY]) => ({clientX, clientY}))});
        feed.dispatchEvent(event);
    };
    for (const [type, points] of touches) send(type, points);
}"""


@pytest.mark.parametrize('project_room', [False, True], ids=['main', 'project'])
def test_the_phone_keyboard_goes_away_only_for_deliberate_older_reading(shell_ui, project_room):
    page = phone(shell_ui)
    page.route('**/api/chat/history**', lambda route: route.fulfill(json=history([LONG_LINE * 4] * 30)))
    page.goto(shell_ui['url'])
    page.locator('#chat-messages[data-history-hydrated="true"]').wait_for()
    field, field_id = room_field(page, project_room)
    feed_id = field_id.replace('input', 'messages')
    field.fill('This unsent draft must stay.')
    field.focus()
    page.evaluate('window.__viewport(320)')
    page.wait_for_function("document.body.classList.contains('keyboard-open')")
    focused = lambda: field.evaluate('e => document.activeElement === e')  # noqa: E731

    def swipe(*touches):
        page.evaluate(SWIPE, {'id': feed_id, 'touches': [list(item) for item in touches]})
        settle(page)

    # Layout and the app's own scrolls are no reading gesture.
    page.evaluate("id => { const f = document.getElementById(id); f.scrollTop = 10; f.dispatchEvent(new Event('scroll')); }", feed_id)
    settle(page)
    assert focused()
    swipe(('touchstart', [[100, 150]]), ('touchmove', [[170, 152]]))  # sideways: the drawer's or a reply's
    assert focused()
    swipe(('touchstart', [[100, 150]]), ('touchmove', [[100, 80]]))   # toward newer messages
    assert focused()
    # Two fingers spreading, the first one moving down the feed: a pinch only magnifies.
    swipe(('touchstart', [[100, 150]]), ('touchstart', [[100, 150], [160, 260]]),
          ('touchmove', [[100, 190], [160, 300]]), ('touchmove', [[100, 240], [160, 330]]),
          ('touchmove', [[100, 300]]))
    assert focused(), 'a pinch never dismisses the keyboard'
    swipe(('touchstart', [[100, 150]]), ('touchmove', [[100, 220]]))  # toward older messages
    page.wait_for_function('id => document.activeElement !== document.getElementById(id)', arg=field_id)
    assert field.input_value() == 'This unsent draft must stay.'
    page.evaluate('window.__viewport(844)')  # the keyboard goes down
    settle(page)
    shot(page, f'keyboard-dismissed-by-older-reading-{"project" if project_room else "main"}-{shell_ui["engine"]}')


@pytest.mark.parametrize('dismiss', ['blur', 'host-collapse', 'closing'])
@pytest.mark.parametrize('project_room', [False, True], ids=['main', 'project'])
def test_a_dismissal_reconciles_the_restored_viewport_without_another_resize(shell_ui, project_room, dismiss):
    page = phone(shell_ui)
    page.route('**/api/chat/history**', lambda route: route.fulfill(json=history(['Long saved line. ' * 25] * 20)))
    page.goto(shell_ui['url'], wait_until='domcontentloaded')
    page.locator('#chat-messages[data-history-hydrated="true"]').wait_for()
    field, field_id = room_field(page, project_room)
    field.fill('Keep this draft.')
    field.focus()
    page.evaluate('window.__viewport(320)')
    page.wait_for_function("document.body.classList.contains('keyboard-open')")
    # A native dismissal and the viewport's restoration can be one task: the measurement at the focusout's frame is
    # final, with no resize after it. A collapse under 100 px is Telegram-iOS #2235, which the host adapter recovers.
    page.evaluate("""({id, mode}) => {
        if (mode === 'host-collapse') window.__viewport(60);
        else document.getElementById(id).blur();
        if (mode !== 'closing') window.__viewport(844, false);
    }""", {'id': field_id, 'mode': dismiss})
    settle(page)
    settle(page)
    if dismiss == 'closing':
        # Focus is gone while the keyboard is still going down: the composer stays above it.
        closing = page.evaluate(GEOMETRY, field_id)
        assert closing['bottom'] <= 321 and not closing['keyboard'], closing
        page.evaluate('window.__viewport(844)')
        settle(page)
    recovered = page.evaluate(GEOMETRY, field_id)
    name = f'keyboard-recovery-{"project" if project_room else "main"}-{dismiss}-{shell_ui["engine"]}'
    facts(name, recovered)
    shot(page, name)
    assert recovered['body'] == 844 and not recovered['keyboard'] and not recovered['focused'], recovered
    assert 0 <= recovered['top'] < recovered['bottom'] <= 844, recovered
    assert field.input_value() == 'Keep this draft.'
    field.focus()
    page.evaluate('window.__viewport(320)')
    settle(page)
    again = page.evaluate(GEOMETRY, field_id)
    assert again['keyboard'] and again['bottom'] <= 321, again


CAP = """id => { const f = document.getElementById(id); return {height: f.getBoundingClientRect().height,
    cap: parseFloat(getComputedStyle(f).maxHeight), sizing: f.dataset.sizing,
    keyboard: document.body.classList.contains('keyboard-open')}; }"""


@pytest.mark.parametrize('project_room', [False, True], ids=['main', 'project'])
def test_a_long_draft_gets_its_resting_cap_back_after_a_dismissal_without_resize(shell_ui, project_room):
    """WebKit measures the field (chat_input_sizing.js): the keyboard's quarter cap must not outlive a dismissal whose
    restored viewport sends no resize (Opus R1: 62 px stayed while the cap was back at 140)."""
    page = phone(shell_ui)
    page.route('**/api/chat/history**', lambda route: route.fulfill(json=history(['Saved line.'] * 5)))
    page.goto(shell_ui['url'])
    page.locator('#chat-messages[data-history-hydrated="true"]').wait_for()
    field, field_id = room_field(page, project_room)
    field.fill('\n'.join(['A long draft with wrapped words'] * 12))
    settle(page)
    measured = {'rest': page.evaluate(CAP, field_id)}
    field.focus()
    page.evaluate('window.__viewport(250)')
    page.wait_for_function("document.body.classList.contains('keyboard-open')")
    settle(page)
    settle(page)
    measured['keyboard'] = page.evaluate(CAP, field_id)
    page.evaluate("id => { document.getElementById(id).blur(); window.__viewport(844, false); }", field_id)
    for _ in range(3):
        settle(page)
    measured['dismissed'] = page.evaluate(CAP, field_id)
    name = f'long-draft-cap-after-dismissal-{"project" if project_room else "main"}-{shell_ui["engine"]}'
    facts(name, measured)
    shot(page, name)
    assert measured['keyboard']['height'] == measured['keyboard']['cap'] == 62, measured
    assert not measured['dismissed']['keyboard'] and measured['dismissed']['cap'] == 140, measured
    assert measured['dismissed']['height'] == measured['rest']['height'] == 140, measured
    assert field.input_value().count('\n') == 11, 'the draft stays whole'


def test_a_blur_during_a_pinch_releases_the_keyboard_state_without_resizing_the_shell(shell_ui):
    page = phone(shell_ui)
    page.goto(shell_ui['url'], wait_until='domcontentloaded')
    field = page.locator('#chat-input')
    field.wait_for(state='visible')
    field.focus()
    page.evaluate('window.__viewport(320)')
    page.wait_for_function("document.body.classList.contains('keyboard-open')")
    before = page.evaluate('document.body.getBoundingClientRect().height')
    page.evaluate('window.__pinch()')
    field.blur()
    settle(page)
    settle(page)
    assert not page.evaluate("document.body.classList.contains('keyboard-open')")
    assert page.evaluate('document.body.getBoundingClientRect().height') == before, 'a pinch only magnifies'
