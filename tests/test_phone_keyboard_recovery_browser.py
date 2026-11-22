"""The phone composer and its keyboard on the production SPA, Main and a Project, in WebKit and Chromium.

The visual viewport, touches and pinch are emulated signals into the page's own owners (keyboard_viewport.js focus/height
contract, chat_reading_position.js gestures, chat_input_sizing.js, gesture_tail.js); none of this is a native iPhone
keyboard, IME, autocorrect panel or Telegram client. Uses the static-server/API fixture; never starts the runtime.
"""
from __future__ import annotations

import json
import os
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


def phone(ui, *, telegram=True, width=390, height=844):
    page = seed_page({**ui, 'new_page': lambda **options: ui['new_page'](is_mobile=True, **options)}, width, height,
                     telegram=telegram)
    page.add_init_script(VIEWPORT)
    return page


def telegram_phone(ui, *, fullscreen, width=390, height=844):
    """The phone inside Telegram as the sidecar serves it (the host stamp) with the SDK stand-in: fullscreen granted under
    the device's and Telegram's top (47 + 46) and its sides, or refused (expanded: Telegram's own bar is outside the
    page, no insets)."""
    page = phone(ui, telegram=False, width=width, height=height)
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
    const room = feed.closest('.project-panel') || document.getElementById('page-chat');
    return {areaBottom: area.getBoundingClientRect().bottom, field: field.getBoundingClientRect().height,
        top: field.getBoundingClientRect().top, bottom: field.getBoundingClientRect().bottom,
        cap: parseFloat(getComputedStyle(field).maxHeight), feed: feed.clientHeight,
        sideways: feed.scrollWidth - feed.clientWidth, focused: document.activeElement === field,
        keyboard: document.body.classList.contains('keyboard-open'),
        folded: 'folded' in area.querySelector('.chat-input-wrap').dataset,
        room: Math.round(room.getBoundingClientRect().height),
        vvh: Math.ceil(visualViewport.height),
        scrollY: window.scrollY,
        body: document.body.getBoundingClientRect().height};
}"""

FIT = """id => {
    const field = document.getElementById(id), area = document.getElementById(id.replace(/input$/, 'input-area'));
    const feed = document.getElementById(id.replace(/input$/, 'messages'));
    const room = feed.closest('.project-panel') || document.getElementById('page-chat');
    const bar = room.querySelector('.chat-page-header, .project-panel-bar');
    const box = el => el.getBoundingClientRect();
    const shown = el => el.getClientRects().length > 0 && getComputedStyle(el).visibility !== 'hidden'
        && !el.matches('.chat-scroll-bottom-btn:not(.visible)');
    const reached = el => { const r = box(el), x = (r.left + r.right) / 2, y = (r.top + r.bottom) / 2;
        const hit = y < visualViewport.height ? document.elementFromPoint(x, y) : null;
        return Boolean(hit && el.contains(hit)); };
    const name = el => el.getAttribute('aria-label') || el.title || el.textContent.trim().slice(0, 24) || el.className;
    const barControls = [...bar.querySelectorAll('button')].filter(shown);
    const controls = [...barControls, ...area.querySelectorAll(
        '.chat-toolbar-row button:not(.chat-capsule-body *), .chat-text-row button')].filter(shown);
    const edge = Math.round(parseFloat(getComputedStyle(bar, '::before').height) - box(bar).height);
    const clearTop = Math.max(box(feed).top, box(bar).bottom + edge);
    const clearBottom = Math.min(box(feed).bottom, box(area).top, visualViewport.height);
    const clear = r => r && r.width > 0 && r.height > 0 && r.top >= clearTop - 0.5 && r.bottom <= clearBottom + 0.5;
    const message = [...feed.querySelectorAll('.chat-bubble .message')].filter(shown).at(-1);
    let lastLine;
    for (const walk = document.createTreeWalker(message || feed, NodeFilter.SHOW_TEXT); message && walk.nextNode();) {
        if (!walk.currentNode.textContent.trim()) continue;
        const range = document.createRange();
        range.selectNodeContents(walk.currentNode);
        for (const r of range.getClientRects()) if (r.width && r.height) lastLine = r;
    }
    const bubble = message?.closest('.chat-bubble'), time = bubble?.querySelector('.msg-time');
    const wrap = area.querySelector('.chat-input-wrap');
    return {viewport: visualViewport.height, room: box(room).height, scrollY: window.scrollY,
        barTop: Math.min(...barControls.map(el => box(el).top)),
        clearTop, clearBottom, clearHeight: clearBottom - clearTop, composerTop: box(area).top,
        lastLine: Boolean(clear(lastLine)), foot: Boolean(time?.textContent.trim()) && clear(box(time)),
        underBar: box(feed).top <= box(bar).top + 1, underComposer: box(feed).bottom >= box(area).bottom - 1,
        bottomSeam: bubble ? box(area).top - box(bubble).bottom : null, edge,
        field: box(field).height, fieldTop: box(field).top, cap: parseFloat(getComputedStyle(field).maxHeight),
        send: box(area.querySelector('.chat-send-inline')).bottom,
        composer: box(area).bottom, controls: controls.length, unreached: controls.filter(el => !reached(el)).map(name),
        barUnreached: barControls.filter(el => !reached(el)).map(name),
        folded: 'folded' in wrap.dataset, focused: document.activeElement === field,
        capsules: [...wrap.querySelectorAll('.chat-toolbar-row .chat-capsule-head')].filter(shown).length,
        keyboard: document.body.classList.contains('keyboard-open'), sideways: document.scrollingElement.scrollWidth - innerWidth};
}"""

ROOM_CONTRACT = """({id, rowIndex}) => {
    const field = document.getElementById(id), area = document.getElementById(id.replace(/input$/, 'input-area'));
    const feed = document.getElementById(id.replace(/input$/, 'messages'));
    const room = feed.closest('.project-panel') || document.getElementById('page-chat');
    const bar = room.querySelector('.chat-page-header, .project-panel-bar');
    const row = rowIndex == null ? null : feed.querySelector(`[data-history-id="chat:${rowIndex}"] .message`);
    return {
        room: Math.round(room.getBoundingClientRect().height),
        vvh: Math.ceil(visualViewport.height),
        scrollY: window.scrollY,
        barTop: Math.round(bar.getBoundingClientRect().top),
        areaTop: area.getBoundingClientRect().top,
        areaBottom: area.getBoundingClientRect().bottom,
        field: field.getBoundingClientRect().height,
        cap: parseFloat(getComputedStyle(field).maxHeight),
        edge: feed.scrollHeight - feed.clientHeight - feed.scrollTop,
        lineTop: row ? row.getBoundingClientRect().top : null,
        keyboard: document.body.classList.contains('keyboard-open'),
        rootKeyboard: document.documentElement.classList.contains('keyboard-open'),
        focused: document.activeElement === field,
    };
}"""


def assert_contract(state, *, vvh, reading_y=None, at_bottom=False, keyboard=None):
    assert state['room'] == vvh and state['vvh'] == vvh, state
    assert state['scrollY'] == 0 and state['barTop'] >= 0 and state['areaBottom'] <= vvh + 1, state
    assert state['keyboard'] == state['rootKeyboard'], state
    if keyboard is not None:
        assert state['keyboard'] is keyboard, state
    if reading_y is not None:
        assert abs(state['lineTop'] - reading_y) <= 2, (state, reading_y)
    if at_bottom:
        assert abs(state['edge']) <= 1.5, state


def loaded_telegram_room(ui, project_room, fullscreen, texts, *, width=390, height=844):
    page = telegram_phone(ui, fullscreen=fullscreen, width=width, height=height)
    page.route('**/api/chat/history**', lambda route: route.fulfill(json=history(texts)))
    page.goto(ui['url'])
    page.wait_for_function('mode => document.documentElement.dataset.telegramFullscreen === mode',
                           arg='active' if fullscreen else 'expanded')
    page.locator('#chat-messages[data-history-hydrated="true"]').wait_for()
    field, field_id = room_field(page, project_room)
    page.locator(f'#{field_id.replace("input", "messages")} [data-history-id="chat:{len(texts) - 1}"]').wait_for()
    page.evaluate('h => window.__viewport(h)', height)
    still(page)
    return page, field, field_id


def loaded_browser_room(ui, project_room, texts, *, width=390, height=844):
    page = phone(ui, telegram=False, width=width, height=height)
    page.route('**/api/chat/history**', lambda route: route.fulfill(json=history(texts)))
    page.goto(ui['url'])
    page.locator('#chat-messages[data-history-hydrated="true"]').wait_for()
    field, field_id = room_field(page, project_room)
    page.locator(f'#{field_id.replace("input", "messages")} [data-history-id="chat:{len(texts) - 1}"]').wait_for()
    page.evaluate('h => window.__viewport(h)', height)
    still(page)
    return page, field, field_id


def scroll_row_to(page, field_id, row_index, y=200, *, top_anchor=False):
    feed = page.locator('#' + field_id.replace('input', 'messages'))
    feed.dispatch_event('wheel', {'deltaY': -240})
    settle(page)
    selector = f'[data-history-id="chat:{row_index}"]' + ('' if top_anchor else ' .message')
    feed.locator(selector).evaluate(
        '(el, y) => { const f = el.closest("#chat-messages, .chat-messages"); f.scrollTop += el.getBoundingClientRect().top - y; }',
        y,
    )
    still(page)
    assert feed.evaluate('f => f.scrollHeight - f.clientHeight - f.scrollTop') > 48
    return feed.locator(f'[data-history-id="chat:{row_index}"] .message').evaluate('el => el.getBoundingClientRect().top')


def keyboard_room(ui, project_room, fullscreen):
    """A long draft as the keyboard rises to 508 px."""
    page, field, field_id = loaded_telegram_room(ui, project_room, fullscreen, [LONG_LINE * 2] * 12)
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


LATE_READ = """({id, signal}) => {
    const feed = document.getElementById(id);
    if (signal === 'scroll') { feed.scrollTop -= 40; feed.dispatchEvent(new Event('scroll')); return; }
    const row = feed.querySelector('.chat-bubble .message');
    row.dispatchEvent(new Event('pointerdown', {bubbles: true}));
    row.dispatchEvent(new Event('touchstart', {bubbles: true}));
}"""


# ---- Scenario 1: iOS visual-only open/close, document scroll reset, reader scroll winning, silent restore + tap ----

@pytest.mark.parametrize('telegram', [True, False], ids=['telegram', 'browser'])
@pytest.mark.parametrize('reading', [False, True], ids=['latest', 'reading'])
def test_ios_visual_keyboard_open_close_document_scroll_and_silent_restore_tap(shell_ui, telegram, reading):
    texts = [f'Saved line {i}. ' + LONG_LINE for i in range(28)]
    loader = loaded_telegram_room if telegram else loaded_browser_room
    page, field, field_id = loader(shell_ui, False, True, texts) if telegram else loader(shell_ui, False, texts)
    feed_id = field_id.replace('input', 'messages')
    row_idx = 10 if reading else None
    target_y = scroll_row_to(page, field_id, row_idx, 200) if reading else None
    read_state = lambda: page.evaluate(ROOM_CONTRACT, {'id': field_id, 'rowIndex': row_idx})  # noqa: E731

    draft = '\n'.join(['Keep a long draft.'] * 12)
    field.fill(draft)
    still(page)

    # Simulate WebKit scrolling the document when the visual viewport shrinks on keyboard open.
    page.evaluate("""() => {
        document.body.style.minHeight = '1400px';
        window.scrollTo(0, 240);
        window.__viewport(508);
        window.dispatchEvent(new Event('scroll'));
        document.body.style.minHeight = '';
    }""")
    still(page)
    opened = read_state()
    assert_contract(opened, vvh=508, reading_y=target_y, at_bottom=not reading, keyboard=True)

    # Reader's own scroll while typing wins over earlier anchors across subsequent viewport resizes.
    if reading:
        moved_y = page.evaluate("""({feedId, rowIdx}) => {
            const feed = document.getElementById(feedId);
            feed.scrollTop -= 50;
            feed.dispatchEvent(new Event('scroll'));
            return feed.querySelector(`[data-history-id="chat:${rowIdx}"] .message`).getBoundingClientRect().top;
        }""", {'feedId': feed_id, 'rowIdx': row_idx})
        still(page)
        page.evaluate('window.__viewport(480)')
        still(page)
        after_reader_move = read_state()
        assert_contract(after_reader_move, vvh=480, reading_y=moved_y, keyboard=True)
        target_y = moved_y

    # Silent restore within the settle window reconciles automatically with no tap needed.
    page.clock.install()
    field.blur()
    settle(page)
    page.evaluate('window.__viewport(844, false)')
    page.clock.run_for(300)
    still(page)
    settled_close = read_state()
    assert_contract(settled_close, vvh=844, reading_y=target_y, at_bottom=not reading, keyboard=False)

    # Late silent restore after the settle window (the black-strip bug): waits quietly without timers until a tap,
    # then reconciles on the first tap and stops listening.
    field.focus()
    page.evaluate('window.__viewport(508)')
    settle(page)
    page.clock.run_for(300)
    still(page)
    assert_contract(read_state(), vvh=508, reading_y=target_y, at_bottom=not reading, keyboard=True)
    field.blur()
    still(page)
    page.clock.run_for(400)
    still(page)
    reads_after_settle = page.evaluate('window.__viewportReads()')
    page.clock.run_for(400)
    assert page.evaluate('window.__viewportReads()') == reads_after_settle, 'no background polling after settle'
    page.evaluate('window.__viewport(844, false)')
    stale = read_state()
    assert stale['room'] == 508 and not stale['keyboard'] and not stale['focused']

    page.evaluate(LATE_READ, {'id': feed_id, 'signal': 'touch'})
    still(page)
    restored = read_state()
    reads_after_tap = page.evaluate('window.__viewportReads()')
    page.evaluate(LATE_READ, {'id': feed_id, 'signal': 'touch'})
    settle(page)
    assert page.evaluate('window.__viewportReads()') == reads_after_tap, 'one-shot listener disarms after first tap'
    assert_contract(restored, vvh=844, reading_y=target_y, at_bottom=not reading, keyboard=False)
    assert restored['cap'] == 140 and field.input_value() == draft


# ---- Scenario 2: Android-style paired keyboard open, Back (focus kept), and close ----

@pytest.mark.parametrize('reading', [False, True], ids=['latest', 'reading'])
def test_android_paired_keyboard_open_back_and_close(shell_ui, reading):
    texts = [f'Saved line {i}. ' + LONG_LINE for i in range(28)]
    page, field, field_id = loaded_telegram_room(shell_ui, False, True, texts)
    row_idx = 11 if reading else None
    target_y = scroll_row_to(page, field_id, row_idx, 210) if reading else None
    read_state = lambda: page.evaluate(ROOM_CONTRACT, {'id': field_id, 'rowIndex': row_idx})  # noqa: E731

    draft = '\n'.join(['Paired Android draft line.'] * 12)
    field.fill(draft)
    still(page)

    page.evaluate('window.__layoutViewport(508)')
    still(page)
    assert_contract(read_state(), vvh=508, reading_y=target_y, at_bottom=not reading, keyboard=True)

    # Android Back hides the soft keyboard while the field keeps focus.
    page.evaluate('window.__layoutViewport(844)')
    still(page)
    back = read_state()
    assert back['focused']
    assert_contract(back, vvh=844, reading_y=target_y, at_bottom=not reading, keyboard=False)

    # Tapping the focused field brings the paired keyboard up again, then blur + restore closes it cleanly.
    page.evaluate('window.__layoutViewport(508)')
    still(page)
    assert_contract(read_state(), vvh=508, reading_y=target_y, at_bottom=not reading, keyboard=True)

    field.blur()
    page.evaluate('window.__layoutViewport(844)')
    still(page)
    closed = read_state()
    assert_contract(closed, vvh=844, reading_y=target_y, at_bottom=not reading, keyboard=False)
    assert closed['field'] == closed['cap'] == 140 and field.input_value() == draft


# ---- Scenario 3: Telegram fullscreen <-> expanded while typing and right after blur ----

@pytest.mark.parametrize('kind', ['visual', 'paired'])
@pytest.mark.parametrize('reading', [False, True], ids=['latest', 'reading'])
def test_telegram_fullscreen_and_expanded_while_typing_and_after_blur(shell_ui, kind, reading):
    texts = [f'Saved line {i}. ' + LONG_LINE for i in range(28)]
    page, field, field_id = loaded_telegram_room(shell_ui, False, True, texts)
    feed_id = field_id.replace('input', 'messages')
    row_idx = 12 if reading else None
    target_y = scroll_row_to(page, field_id, row_idx, 200) if reading else None
    read_state = lambda: page.evaluate(ROOM_CONTRACT, {'id': field_id, 'rowIndex': row_idx})  # noqa: E731

    field.fill('Keep my draft across Telegram modes.')
    still(page)
    move = '__viewport' if kind == 'visual' else '__layoutViewport'
    page.evaluate(f'window.{move}(508)')
    still(page)
    assert_contract(read_state(), vvh=508, reading_y=target_y, at_bottom=not reading, keyboard=True)

    # Switch fullscreen -> expanded while typing (Telegram's top chrome takes 84 px of layout height).
    if kind == 'visual':
        page.evaluate("() => { window.__tg.exit(); window.__layoutViewport(760, false); window.__viewport(424, false);"
                      " window.dispatchEvent(new Event('resize')); visualViewport.dispatchEvent(new Event('resize')); }")
    else:
        page.evaluate("() => { window.__tg.exit(); window.__layoutViewport(424); }")
    still(page)
    assert page.evaluate('document.documentElement.dataset.telegramFullscreen') == 'expanded'
    assert_contract(read_state(), vvh=424, reading_y=target_y, at_bottom=not reading, keyboard=True)

    # Switch expanded -> fullscreen while typing.
    if kind == 'visual':
        page.evaluate("() => { window.__tg.enter(); window.__layoutViewport(844, false); window.__viewport(508, false);"
                      " window.dispatchEvent(new Event('resize')); visualViewport.dispatchEvent(new Event('resize')); }")
    else:
        page.evaluate("() => { window.__tg.enter(); window.__layoutViewport(508); }")
    still(page)
    assert page.evaluate('document.documentElement.dataset.telegramFullscreen') == 'active'
    assert_contract(read_state(), vvh=508, reading_y=target_y, at_bottom=not reading, keyboard=True)

    # Right after blur, Telegram leaves fullscreen and silently restores expanded's 760 px height before a tap.
    page.clock.install()
    field.blur()
    still(page)
    page.clock.run_for(40)
    page.evaluate("() => { window.__tg.exit(); }")
    still(page)
    page.clock.run_for(350)
    still(page)
    page.evaluate(f'window.{move}(760, false)')
    page.evaluate(LATE_READ, {'id': feed_id, 'signal': 'touch'})
    still(page)
    restored = read_state()
    air = page.evaluate("getComputedStyle(document.getElementById('chat-input-area')).paddingTop")
    assert_contract(restored, vvh=760, reading_y=target_y, at_bottom=not reading, keyboard=False)
    assert air == '24px' and field.input_value() == 'Keep my draft across Telegram modes.'


# ---- Scenario 4: Rotation while typing, and bfcache return after rotation ----

@pytest.mark.parametrize('first', ['layout', 'visual'])
def test_rotation_while_typing_and_bfcache_return_after_rotation(shell_ui, first):
    texts = [f'Saved line {i}. ' + LONG_LINE for i in range(28)]
    page, field, field_id = loaded_telegram_room(shell_ui, False, True, texts, width=375, height=667)
    row_idx = 12
    target_y = scroll_row_to(page, field_id, row_idx, 8, top_anchor=True)
    read_state = lambda: page.evaluate(ROOM_CONTRACT, {'id': field_id, 'rowIndex': row_idx})  # noqa: E731

    draft = 'This draft survives rotation and bfcache.'
    field.fill(draft)
    page.evaluate('window.__viewport(400)')
    still(page)
    assert_contract(read_state(), vvh=400, reading_y=target_y, keyboard=True)

    # Rotate portrait (375x667, vv=400) -> landscape (667x375, vv=200) with layout/visual arriving in either order.
    if first == 'visual':
        page.evaluate('window.__viewport(200)')
        settle(page)
    else:
        page.evaluate('window.__layoutViewport(375, false); window.__viewport(400, false);')
    page.set_viewport_size({'width': 667, 'height': 375})
    page.evaluate("""first => {
        const old = visualViewport.height;
        window.__layoutViewport(375, false);
        window.__viewport(old, false);
        window.dispatchEvent(new Event('orientationchange'));
        window.dispatchEvent(new Event('resize'));
        if (first === 'layout') window.__viewport(200);
    }""", first)
    still(page)
    assert_contract(read_state(), vvh=200, reading_y=target_y, keyboard=True)

    # Rotate back to portrait with the keyboard hidden while keeping focus, then open again.
    page.evaluate('window.__layoutViewport(667, false)')
    page.set_viewport_size({'width': 375, 'height': 667})
    page.evaluate("() => { window.__layoutViewport(667); window.dispatchEvent(new Event('orientationchange')); }")
    still(page)
    assert_contract(read_state(), vvh=667, reading_y=target_y, keyboard=False)
    page.evaluate('window.__viewport(400)')
    still(page)
    assert_contract(read_state(), vvh=400, reading_y=target_y, keyboard=True)

    # Cached in portrait while keyboard was closing, rotated silently to landscape (667x375), then shown from bfcache.
    field.blur()
    still(page)
    page.evaluate("window.dispatchEvent(new PageTransitionEvent('pagehide', {persisted: true}))")
    page.evaluate('window.__layoutViewport(375, false)')
    page.set_viewport_size({'width': 667, 'height': 375})
    still(page)
    page.evaluate("() => { window.__layoutViewport(375, false); window.dispatchEvent(new PageTransitionEvent('pageshow', {persisted: true})); }")
    still(page)
    cached_return = read_state()
    assert_contract(cached_return, vvh=375, reading_y=target_y, keyboard=False)
    assert field.input_value() == draft


def test_paired_keyboard_rotation_keeps_keyboard_open_and_last_line_visible(shell_ui):
    texts = [f'Message {i}. One short line.' for i in range(30)] + ['Last message.']
    page, field, field_id = loaded_telegram_room(shell_ui, False, False, texts, width=360, height=640)
    field.fill('One-line draft.')
    page.evaluate('window.__layoutViewport(380)')
    still(page)
    assert page.evaluate(FIT, field_id)['keyboard'] is True

    # Rotate 360x380 -> 640x180 with the paired keyboard open: composer folds and the newest line stays readable.
    page.evaluate('window.__layoutViewport(180, false)')
    page.set_viewport_size({'width': 640, 'height': 180})
    page.evaluate("""() => {
        window.__layoutViewport(180);
        window.dispatchEvent(new Event('orientationchange'));
        visualViewport.dispatchEvent(new Event('resize'));
    }""")
    still(page)
    rotated = page.evaluate(FIT, field_id)
    assert rotated['room'] == rotated['viewport'] == 180 and rotated['scrollY'] == 0, rotated
    assert rotated['keyboard'] and rotated['folded'] and rotated['lastLine'] and rotated['unreached'] == [], rotated

    # Android Back restores landscape height (640x360) with focus kept, clearing keyboard-open.
    page.evaluate('window.__layoutViewport(360, false)')
    page.set_viewport_size({'width': 640, 'height': 360})
    page.evaluate('window.__layoutViewport(360)')
    still(page)
    restored = page.evaluate(FIT, field_id)
    assert restored['room'] == 360 and restored['focused'] and not restored['keyboard'], restored


# ---- Scenario 5: Pinch zoom while the keyboard is up ----

def test_pinch_zoom_while_keyboard_is_up_and_silent_restore_after_unpinch(shell_ui):
    texts = [f'Saved line {i}. ' + LONG_LINE for i in range(24)]
    page, field, field_id = loaded_browser_room(shell_ui, False, texts)
    row_idx = 10
    target_y = scroll_row_to(page, field_id, row_idx, 170)
    read_state = lambda: page.evaluate(ROOM_CONTRACT, {'id': field_id, 'rowIndex': row_idx})  # noqa: E731

    field.fill('Zoomed draft.')
    field.focus()
    page.evaluate('window.__viewport(320)')
    page.wait_for_function("document.body.classList.contains('keyboard-open')")
    still(page)
    assert_contract(read_state(), vvh=320, reading_y=target_y, keyboard=True)

    page.evaluate('window.__pinch(true)')
    field.blur()
    settle(page)
    settle(page)
    pinched = read_state()
    assert not pinched['keyboard'] and pinched['room'] == 320, 'a pinch only magnifies without resizing the room'

    # Unpinch while keyboard height still stands, then silent restore + tap recovers full 844 px height and reading place.
    page.evaluate('window.__pinch(false)')
    settle(page)
    page.evaluate('window.__viewport(844, false)')
    page.locator('#chat-messages').dispatch_event('pointerdown')
    still(page)
    assert_contract(read_state(), vvh=844, reading_y=target_y, keyboard=False)


# ---- Scenario 6: Telegram's brief collapse below 100 px on first keyboard open, then silent restore + tap ----

@pytest.mark.parametrize('project_room', [False, True], ids=['main', 'project'])
def test_telegram_brief_collapse_below_100px_then_silent_restore_and_tap(shell_ui, project_room):
    texts = [f'Saved line {i}. ' + LONG_LINE for i in range(24)]
    page, field, field_id = loaded_telegram_room(shell_ui, project_room, True, texts)
    feed_id = field_id.replace('input', 'messages')
    row_idx = 10
    target_y = scroll_row_to(page, field_id, row_idx, 200)
    read_state = lambda: page.evaluate(ROOM_CONTRACT, {'id': field_id, 'rowIndex': row_idx})  # noqa: E731

    field.fill('Keep this draft across collapse.')
    field.focus()
    page.clock.install()
    # Telegram-iOS #2235: first keyboard open collapses visualViewport under 100 px; host_telegram.js blurs + hides keyboard.
    page.evaluate("() => { window.__viewport(60); window.__tg.listeners.get('viewportChanged')?.(); }")
    settle(page)
    page.clock.run_for(350)
    settle(page)
    collapsed = page.evaluate("""id => ({
        focused: document.activeElement === document.getElementById(id),
        keyboard: document.body.classList.contains('keyboard-open'),
        calls: window.__tg.calls.slice(),
    })""", field_id)
    assert not collapsed['focused'] and not collapsed['keyboard'] and 'hideKeyboard' in collapsed['calls'], collapsed

    # Silent restore after settle timer, reconciled by the next tap.
    page.evaluate('window.__viewport(844, false)')
    page.evaluate(LATE_READ, {'id': feed_id, 'signal': 'touch'})
    page.clock.run_for(50)
    settle(page)
    recovered = read_state()
    assert_contract(recovered, vvh=844, reading_y=target_y, keyboard=False)
    assert field.input_value() == 'Keep this draft across collapse.'

    # Subsequent keyboard open works normally.
    field.focus()
    page.evaluate('window.__viewport(508)')
    page.clock.run_for(300)
    assert_contract(read_state(), vvh=508, reading_y=target_y, keyboard=True)


# ---- Scenario 7: A partial restore that pauses and then completes ----

@pytest.mark.parametrize('completion', ['event', 'silent-tap'])
def test_partial_restore_that_pauses_and_then_completes(shell_ui, completion):
    texts = [f'Saved line {i}. ' + LONG_LINE for i in range(26)]
    page, field, field_id = loaded_telegram_room(shell_ui, False, True, texts)
    feed_id = field_id.replace('input', 'messages')
    row_idx = 10
    target_y = scroll_row_to(page, field_id, row_idx, 200)
    read_state = lambda: page.evaluate(ROOM_CONTRACT, {'id': field_id, 'rowIndex': row_idx})  # noqa: E731

    draft = '\n'.join(['Keep a long draft through a paused close.'] * 12)
    field.fill(draft)
    page.evaluate('window.__layoutViewport(508)')
    still(page)
    assert_contract(read_state(), vvh=508, reading_y=target_y, keyboard=True)

    page.clock.install()
    field.blur()
    settle(page)
    page.evaluate('window.__layoutViewport(620)')
    page.clock.run_for(500)  # pauses past the settle timer
    settle(page)
    paused = read_state()
    assert_contract(paused, vvh=620, reading_y=target_y, keyboard=False)
    assert paused['field'] == paused['cap'] == 140

    if completion == 'event':
        page.evaluate('window.__layoutViewport(844)')
    else:
        page.evaluate('window.__layoutViewport(844, false)')
        page.evaluate(LATE_READ, {'id': feed_id, 'signal': 'touch'})
    page.clock.run_for(100)
    settle(page)
    completed = read_state()
    assert_contract(completed, vvh=844, reading_y=target_y, keyboard=False)
    assert completed['field'] == completed['cap'] == 140 and field.input_value() == draft


# ---- Scenario 8: Desktop (fine pointer) and tablet (coarse pointer) window resize with focused field ----

@pytest.mark.parametrize('surface', ['desktop-fine', 'tablet-coarse'])
def test_desktop_and_tablet_window_resize_with_focused_field(shell_ui, surface):
    texts = [f'Saved line {i}. ' + LONG_LINE for i in range(28)]
    coarse = surface == 'tablet-coarse'
    wrapped = {**shell_ui, 'new_page': lambda **opts: shell_ui['new_page'](is_mobile=coarse, **opts)}
    page = seed_page(wrapped, 680 if coarse else 620, 820 if coarse else 844, touch=coarse)
    page.route('**/api/chat/history**', lambda route: route.fulfill(json=history(texts)))
    page.goto(shell_ui['url'])
    page.locator('#chat-messages[data-history-hydrated="true"]').wait_for()
    field, field_id = room_field(page, False)
    still(page)
    row_idx = 12
    target_y = scroll_row_to(page, field_id, row_idx, 8, top_anchor=True)
    read_state = lambda: page.evaluate(ROOM_CONTRACT, {'id': field_id, 'rowIndex': row_idx})  # noqa: E731

    field.fill('Window resize draft.')
    field.focus()
    still(page)

    if surface == 'desktop-fine':
        assert page.evaluate('document.documentElement.dataset.pointer') == 'fine'
        # Vertical-only shrink on a narrow fine-pointer stack window is NEVER keyboard-open.
        page.set_viewport_size({'width': 620, 'height': 480})
        still(page)
        assert_contract(read_state(), vvh=480, reading_y=target_y, keyboard=False)
        # Widening into docked layout uses 100dvh and stays keyboard=False.
        page.set_viewport_size({'width': 1024, 'height': 480})
        still(page)
        docked = read_state()
        assert docked['room'] == 480 and not docked['keyboard'] and docked['scrollY'] == 0
    else:
        assert page.evaluate('document.documentElement.dataset.pointer') == 'coarse'
        # Resizing a coarse-pointer tablet window to another width and height resets focus baseline and fits exact height.
        page.set_viewport_size({'width': 600, 'height': 500})
        still(page)
        assert_contract(read_state(), vvh=500, reading_y=target_y, keyboard=False)
        # Vertical-only shrink at the same width opens, and growing back closes.
        page.set_viewport_size({'width': 600, 'height': 320})
        still(page)
        assert_contract(read_state(), vvh=320, reading_y=target_y, keyboard=True)
        page.set_viewport_size({'width': 600, 'height': 500})
        still(page)
        assert_contract(read_state(), vvh=500, reading_y=target_y, keyboard=False)


# ---- Existing user-facing room, reading-place, gesture, short-room, and caret tests ----

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
        assert geometry['areaBottom'] <= 251 and geometry['feed'] > 0 and not geometry['folded'], geometry
        assert geometry['field'] <= geometry['cap'] + 1 and geometry['sideways'] <= 1 and geometry['scrollY'] == 0, geometry
        if label == 'long':
            shot(page, f'keyboard-250-long-draft-{"project" if project_room else "main"}-{theme}-{shell_ui["engine"]}',
                 {'x': 0, 'y': 0, 'width': 390, 'height': 250})
    assert measured['long']['cap'] == 62
    assert measured['long']['field'] > measured['short']['field']
    name = f'keyboard-250-{"project" if project_room else "main"}-{theme}-{shell_ui["engine"]}'
    facts(name, measured)
    shot(page, name, {'x': 0, 'y': 0, 'width': 390, 'height': 250})
    page.evaluate('window.__viewport(844)')
    field.blur()
    settle(page)
    rested = page.evaluate(GEOMETRY, field_id)
    assert not rested['keyboard'] and rested['cap'] == 140 and rested['body'] == 844 and rested['scrollY'] == 0, rested
    assert field.input_value() == 'Short again'
    shot(page, f'room-after-keyboard-{"project" if project_room else "main"}-{theme}-{shell_ui["engine"]}')


@pytest.mark.parametrize(('project_room', 'latest'), [
    pytest.param(False, False, id='main-reading'), pytest.param(False, True, id='main-latest'),
    pytest.param(True, False, id='project-reading'),
])
def test_settling_telegram_viewport_keeps_the_same_line_in_place(shell_ui, project_room, latest):
    texts = [f'Saved line {i}.' for i in range(24)]
    page, field, field_id = loaded_telegram_room(shell_ui, project_room, True, texts)
    feed = page.locator('#' + field_id.replace('input', 'messages'))
    line = feed.get_by_text(texts[-1] if latest else texts[12], exact=True)
    if not latest:
        feed.dispatch_event('wheel', {'deltaY': -200})
        settle(page)
        line.evaluate("el => el.scrollIntoView({block: 'center'})")
        still(page)
        assert feed.evaluate('f => f.scrollHeight - f.clientHeight - f.scrollTop') > 48
    field.fill('A draft')
    page.evaluate('window.__viewport(634)')
    still(page)

    samples = {}
    for height in (634, 633, 632):
        if height != 634:
            page.evaluate('height => window.__viewport(height)', height)
            still(page)
        rect = line.evaluate("""el => {
            const text = document.createTreeWalker(el, NodeFilter.SHOW_TEXT).nextNode();
            const range = document.createRange();
            range.selectNodeContents(text);
            return range.getBoundingClientRect().toJSON();
        }""")
        fit = page.evaluate(FIT, field_id)
        edge_gap = feed.evaluate('f => f.scrollHeight - f.clientHeight - f.scrollTop')
        assert fit['focused'] and fit['scrollY'] == 0 and rect['height'] > 0 and rect['width'] > 0, (height, rect, fit)
        samples[height] = edge_gap if latest else rect['top']
        assert fit['clearTop'] - 0.5 <= rect['top'] and rect['bottom'] <= fit['clearBottom'] + 0.5, (samples, rect, fit)
    if latest:
        assert all(abs(gap) <= 1.5 for gap in samples.values()), samples
    else:
        assert max(samples.values()) - min(samples.values()) <= 2, samples
    if not latest:
        page.evaluate('window.__viewport(508)')
        still(page)
        fit = page.evaluate(FIT, field_id)
        rows = feed.locator('.chat-bubble').evaluate_all('els => els.map(el => el.getBoundingClientRect().toJSON())')
        assert fit['underBar'] and any(r['top'] < fit['clearTop'] - fit['edge'] and r['bottom'] > 47 for r in rows), fit
        assert fit['underComposer'] and any(r['bottom'] > fit['composerTop'] and r['top'] < fit['composer'] for r in rows), fit


@pytest.mark.parametrize('project_room', [False, True], ids=['main', 'project'])
def test_a_line_read_near_the_feeds_end_is_back_in_place_when_the_keyboard_returns(shell_ui, project_room):
    texts = [f'Saved line {i}.' for i in range(24)]
    page, field, field_id = loaded_telegram_room(shell_ui, project_room, True, texts)
    feed = page.locator('#' + field_id.replace('input', 'messages'))
    line = feed.get_by_text(texts[-4], exact=True)
    field.fill('A draft')
    field.focus()
    page.evaluate('window.__viewport(250)')
    still(page)
    feed.dispatch_event('wheel', {'deltaY': -200})
    settle(page)
    feed.evaluate("(f, [text, top]) => { const el = [...f.querySelectorAll('.chat-bubble .message')].find(m => m.textContent.trim() === text);"
                  " f.scrollTop += el.getBoundingClientRect().top - top; }", [texts[-4], page.evaluate(FIT, field_id)['clearTop'] + 4])
    still(page)

    def place():
        return {'y': line.evaluate('el => el.getBoundingClientRect().top'),
                'toEnd': feed.evaluate('f => f.scrollHeight - f.clientHeight - f.scrollTop')}

    samples = {'read': place()}
    page.evaluate('window.__viewport(844)')
    still(page)
    samples['away'] = place()
    field.focus()
    page.evaluate('window.__viewport(250)')
    still(page)
    samples['back'] = place()
    fit = page.evaluate(FIT, field_id)
    facts(f'reading-place-keyboard-round-trip-{"project" if project_room else "main"}-{shell_ui["engine"]}', samples)
    assert samples['read']['toEnd'] > 48, samples
    assert samples['away']['y'] - samples['read']['y'] > 100, samples
    assert abs(samples['back']['y'] - samples['read']['y']) <= 2, samples
    assert fit['clearTop'] - 0.5 <= samples['back']['y'] < fit['clearBottom'], (samples, fit)


@pytest.mark.parametrize('latest', [False, True], ids=['reading', 'live-end'])
def test_a_long_telegram_feed_stays_placed_without_scanning_its_rows_on_keyboard_resize(shell_ui, latest):
    texts = [f'Saved sample {i}, blue and green.' for i in range(240)]
    scene = history(texts)
    scene['coverage']['upper']['chat'] = scene['coverage']['spans']['chat']['to'] = len(texts)
    page = telegram_phone(shell_ui, fullscreen=True)
    page.route('**/api/chat/history**', lambda route: route.fulfill(json=scene))
    held = None  # the state reads left unanswered while the steps run

    def state(route):
        if held is None: route.fulfill(json={'supervisor_ready': True, 'projects': []})
        else: held.append(route)

    page.route('**/api/state', state)
    page.goto(shell_ui['url'])
    feed = page.locator('#chat-messages')
    feed.locator('[data-history-id="chat:239"]').wait_for()
    page.evaluate('window.__viewport(844)')
    still(page)
    feed.evaluate("el => el.style.setProperty('overflow-anchor', 'none')")
    if not latest:
        feed.dispatch_event('wheel', {'deltaY': -300})
        settle(page)
        feed.locator('[data-history-id="chat:179"]').evaluate(
            'el => { el.parentElement.scrollTop += el.getBoundingClientRect().top - 200; }')
        still(page)
    page.locator('#chat-input').fill('A draft for testing.')
    still(page)
    # The room reads the state every 3 s on its own clock (app.js pollShownRoomState) and, on a slow runner, inside a
    # step: its write re-anchors the feed by every row (~1200 reads; a resize asks none). Its next read waits unanswered
    # across the steps, and none other is asked while it does; whatever a step itself reads still counts.
    held = []
    with page.expect_request('**/api/state', timeout=10_000):
        pass
    samples = feed.evaluate("""async (feed, latest) => {
        const rows = new Set(feed.querySelectorAll(':scope > [data-history-id]'));
        const row = feed.querySelector(`[data-history-id="chat:${latest ? 239 : 179}"]`);
        const input = document.querySelector('#chat-input'), area = document.querySelector('#chat-input-area');
        const rect = Element.prototype.getBoundingClientRect;
        const placement = () => ({top: rect.call(row).top, bottom: rect.call(row.querySelector('.msg-time')).bottom,
            composer: rect.call(area).top, edge: feed.scrollHeight - feed.clientHeight - feed.scrollTop,
            focused: document.activeElement === input, open: document.body.classList.contains('keyboard-open')});
        const rest = placement(), steps = [];
        let reads = 0;
        Element.prototype.getBoundingClientRect = function() {
            if (rows.has(this)) reads++;
            return rect.call(this);
        };
        try {
            for (const h of [700, 634, 633, 632, 560, 508, 508, 'blur', 560, 634, 700, 844]) {
                reads = 0;
                if (h === 'blur') input.blur(); else window.__viewport(h);
                const syncReads = reads, sync = placement();
                await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
                steps.push({h, syncReads, reads, sync, ...placement()});
            }
            return {rows: rows.size, rest, steps};
        } finally { Element.prototype.getBoundingClientRect = rect; }
    }""", latest)
    routes, held = held, None
    for route in routes:
        route.fulfill(json={'supervisor_ready': True, 'projects': []})
    facts(f'keyboard-long-feed-{"latest" if latest else "reading"}-{shell_ui["engine"]}', samples)
    assert samples['rows'] == 240
    assert [step['open'] for step in samples['steps'][:4]] == [False, False, False, True], samples
    assert all(step['focused'] for step in samples['steps'][:7]), samples
    assert not samples['steps'][-1]['focused'] and not samples['steps'][-1]['open'], samples
    for step in samples['steps']:
        assert step['syncReads'] <= 24 and step['reads'] <= 48, step
        for place in (step['sync'], step) if step['h'] != 'blur' else (step,):
            assert place['bottom'] <= place['composer'] + 1, (step['h'], place)
            if latest:
                assert abs(place['edge']) <= 1, (step['h'], place)
            else:
                assert abs(place['top'] - samples['rest']['top']) <= 1, (step['h'], place)


@pytest.mark.parametrize('reading', [False, True], ids=['latest', 'reading'])
def test_crossing_the_keyboard_threshold_by_1px_keeps_the_reading_place_and_draft_scroll(shell_ui, reading):
    """Crossing 25% of 667 px (501 -> 500) toggles cosmetic keyboard-open: reading history stays within 2 px (never
    jumping by the composer padding change), a reader at the bottom stays at the bottom, and draft scroll is kept."""
    texts = [f'Saved line {i}. ' + LONG_LINE for i in range(28)]
    page, field, field_id = loaded_telegram_room(shell_ui, False, True, texts, height=667)
    row_idx = 12 if reading else None
    target_y = scroll_row_to(page, field_id, row_idx, 190) if reading else None
    field.fill('\n'.join(['A long message draft to be written'] * 15))
    still(page)
    assert page.evaluate(FIT, field_id)['cap'] == 140
    samples = []
    for height in (501, 500):
        page.evaluate('h => window.__viewport(h)', height)
        still(page)
        if not samples:
            field.evaluate('f => { f.setSelectionRange(f.value.length, f.value.length); f.scrollTop = 38; }')
        fit = page.evaluate(FIT, field_id)
        contract = page.evaluate(ROOM_CONTRACT, {'id': field_id, 'rowIndex': row_idx})
        assert field.evaluate('f => f.scrollHeight - f.clientHeight > 100 && Math.abs(f.scrollTop - 38) <= 1')
        assert_contract(contract, vvh=height, reading_y=target_y, at_bottom=not reading)
        samples.append({**fit, 'lineTop': contract['lineTop'], 'edgeGap': contract['edge']})
    before, after = samples
    assert [before['keyboard'], after['keyboard']] == [False, True], samples
    assert all(s['unreached'] == [] and (reading or (s['lastLine'] and s['foot'])) for s in samples), samples
    if reading:
        assert abs(after['lineTop'] - before['lineTop']) <= 2, samples


@pytest.mark.parametrize('partial', ['opening', 'closing'])
def test_refocus_during_partial_keyboard_motion_keeps_the_room_and_draft(shell_ui, partial):
    page, field, field_id = loaded_telegram_room(shell_ui, False, True, [LONG_LINE] * 12)
    field.fill('Keep my draft.')
    if partial == 'opening':
        page.evaluate('window.__viewport(650)')
        assert not page.evaluate(GEOMETRY, field_id)['keyboard']
    else:
        page.evaluate('window.__layoutViewport(508)')
        assert page.evaluate(GEOMETRY, field_id)['keyboard']
    page.evaluate("""async ({id, partial}) => {
        const field = document.getElementById(id);
        field.blur();
        if (partial === 'closing') window.__layoutViewport(667);
        await new Promise(requestAnimationFrame);
        field.focus();
        if (partial === 'opening') window.__viewport(508);
        else window.__layoutViewport(480);
    }""", {'id': field_id, 'partial': partial})
    still(page)
    expected = 508 if partial == 'opening' else 480
    state = page.evaluate(FIT, field_id)
    assert state['focused'] and state['keyboard'] and state['room'] == expected and state['scrollY'] == 0, state
    assert state['unreached'] == [] and field.input_value() == 'Keep my draft.'


def test_refocus_while_the_telegram_viewport_is_still_short_keeps_the_draft_sendable(shell_ui):
    page, field, field_id, draft = keyboard_room(shell_ui, False, fullscreen=True)
    still(page)
    before = page.evaluate(FIT, field_id)
    field.blur()
    still(page)
    field.focus()
    still(page)
    after = page.evaluate(FIT, field_id)
    assert after['focused'] and after['keyboard'] and after['viewport'] == 508 and after['scrollY'] == 0, after
    assert after['clearHeight'] == pytest.approx(before['clearHeight'], abs=1) and after['unreached'] == [], after
    assert field.input_value() == draft
    page.locator('#chat-send').tap()
    sent = page.locator('#chat-messages .chat-bubble.user .message').last
    assert sent.inner_text() == draft
    assert field.input_value() == ''


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
    """At 250 px (and 210 px) a long draft keeps Send above the keyboard and leaves a readable message band."""
    page, field, field_id, draft = keyboard_room(shell_ui, project_room, fullscreen)
    measured = {'rising': page.evaluate(FIT, field_id)}
    page.evaluate('window.__viewport(250)')
    still(page)
    name = f'keyboard-250-{"fullscreen" if fullscreen else "expanded"}-{"project" if project_room else "main"}-{shell_ui["engine"]}'
    measured['keyboard'] = page.evaluate(FIT, field_id)
    shot(page, name, {'x': 0, 'y': 0, 'width': 390, 'height': 250})
    page.locator(f'#{field_id}').locator('xpath=ancestor::*[contains(@class,"chat-input-wrap")]') \
        .locator('[data-composer-plus]').click()
    page.locator('body > .chat-plus-menu').wait_for()
    page.locator('body > .chat-plus-menu').evaluate('m => Promise.all(m.getAnimations().map(a => a.finished.catch(() => null)))')
    measured['plus'] = page.evaluate(PLUS_MENU, field_id)
    shot(page, f'{name}-plus', {'x': 0, 'y': 0, 'width': 390, 'height': 250})
    page.keyboard.press('Escape')
    page.locator('body > .chat-plus-menu').wait_for(state='detached')
    settle(page)
    field.evaluate('e => e.setSelectionRange(e.value.length, e.value.length)')
    page.keyboard.type(' and more')
    settle(page)
    settle(page)
    measured['typed'] = page.evaluate(FIT, field_id)
    page.evaluate('window.__viewport(844)')
    field.blur()
    still(page)
    measured['rested'] = page.evaluate(FIT, field_id)
    field.focus()
    page.evaluate('window.__viewport(250)')
    page.wait_for_function("document.body.classList.contains('keyboard-open')")
    still(page)
    measured['again'] = page.evaluate(FIT, field_id)
    page.evaluate('window.__viewport(508)')
    still(page)
    measured['grown'] = page.evaluate(FIT, field_id)
    shot(page, f'{name}-grown', {'x': 0, 'y': 0, 'width': 390, 'height': 508})
    page.evaluate('window.__viewport(250)')
    still(page)
    measured['shrunk'] = page.evaluate(FIT, field_id)
    page.evaluate('window.__pinch()')
    settle(page)
    settle(page)
    measured['pinched'] = page.evaluate(FIT, field_id)
    facts(name, measured)
    band = 93 if fullscreen else 0
    for state in ('rising', 'grown'):
        fit = measured[state]
        assert fit['lastLine'] and fit['foot'] and fit['unreached'] == [] and fit['composer'] <= 508, (state, fit)
        assert fit['underBar'] and fit['underComposer'] and abs(fit['bottomSeam']) <= 2, (state, fit)
        assert fit['edge'] == 32 and fit['scrollY'] == 0, (state, fit)
        assert fit['keyboard'] and fit['focused'] and not fit['folded'] and fit['capsules'] == 4, (state, fit)
    menu = measured['plus']
    assert menu['top'] >= band + 8 - 0.5 and menu['bottom'] <= menu['plusTop'] - 4 + 0.5, f'"+" clears the band: {menu}'
    assert menu['focused'] and menu['keyboard'] and menu['items'] >= 1 and menu['lastShows'], menu
    for state in ('keyboard', 'typed', 'again', 'shrunk', 'pinched'):
        fit = measured[state]
        assert fit['room'] == 250 and fit['composer'] <= 250 and fit['send'] <= 250 and fit['scrollY'] == 0, (state, fit)
        assert fit['folded'] == fullscreen and fit['capsules'] == (0 if fullscreen else 4), (state, fit)
        assert fit['keyboard'] and fit['focused'] and fit['unreached'] == [], (state, fit)
        assert fit['controls'] >= (4 if fullscreen else (7 if project_room else 6)), (state, fit)
        assert fit['barTop'] >= (47 if fullscreen else 0), f'the bar stands under the status bar: {state} {fit}'
        assert fit['underBar'] and fit['underComposer'] and abs(fit['bottomSeam']) <= 2, (state, fit)
        assert fit['foot'] and fit['lastLine'] and fit['edge'] == 0 and fit['clearHeight'] >= 59.5, (state, fit)
        assert 33 <= fit['field'] <= fit['cap'] + 1 and fit['sideways'] <= 0, (state, fit)
    assert measured['again'] == measured['typed']
    assert measured['shrunk'] == measured['again']
    assert measured['pinched'] == measured['shrunk']
    rested = measured['rested']
    assert not rested['keyboard'] and rested['cap'] == 140 and rested['field'] == 140, rested
    assert not rested['folded'] and rested['capsules'] == 4 and rested['edge'] == 32, rested
    assert field.input_value() == draft + ' and more'
    if not fullscreen:
        assert measured['keyboard']['cap'] == 62 and measured['keyboard']['clearHeight'] == pytest.approx(60, abs=0.5)


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
    field.click()
    page.locator(f'{area} .chat-toolbar-row [data-open="true"]').wait_for(state='detached')
    still(page)
    measured['closed'] = {**page.evaluate(CAPSULE, field_id), **page.evaluate(FIT, field_id)}
    shot(page, f'{name}-closed', {'x': 0, 'y': 0, 'width': 390, 'height': height})
    facts(name, measured)
    opened, closed = measured['open'], measured['closed']
    assert (opened['folded'], opened['open'], opened['focused'], opened['keyboard']) == (False, True, True, True), opened
    assert opened['barUnreached'] == [] and opened['barTop'] >= 47 and opened['send'] <= height, opened
    assert (closed['open'], closed['focused'], closed['keyboard']) == (False, True, True), closed
    assert closed['folded'] == (height == 250) and closed['unreached'] == [] and closed['send'] <= height, closed
    assert closed['lastLine'] and closed['foot'] and closed['clearHeight'] >= 59.5, closed
    assert closed['capsules'] == (0 if height == 250 else 4), closed
    assert field.input_value() == draft


def test_a_desk_window_as_low_as_the_short_room_never_folds_the_row(shell_ui):
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

    page.evaluate("id => { const f = document.getElementById(id); f.scrollTop = 10; f.dispatchEvent(new Event('scroll')); }", feed_id)
    settle(page)
    assert focused()
    swipe(('touchstart', [[100, 150]]), ('touchmove', [[170, 152]]))
    assert focused()
    swipe(('touchstart', [[100, 150]]), ('touchmove', [[100, 80]]))
    assert focused()
    swipe(('touchstart', [[100, 150]]), ('touchstart', [[100, 150], [160, 260]]),
          ('touchmove', [[100, 190], [160, 300]]), ('touchmove', [[100, 240], [160, 330]]),
          ('touchmove', [[100, 300]]))
    assert focused(), 'a pinch never dismisses the keyboard'
    swipe(('touchstart', [[100, 150]]), ('touchmove', [[100, 220]]))
    page.wait_for_function('id => document.activeElement !== document.getElementById(id)', arg=field_id)
    assert field.input_value() == 'This unsent draft must stay.'
    page.evaluate('window.__viewport(844)')
    settle(page)
    shot(page, f'keyboard-dismissed-by-older-reading-{"project" if project_room else "main"}-{shell_ui["engine"]}')


def test_a_fresh_tap_after_a_swipe_reaches_its_button_and_a_pinch_never_swipes_to_reply(shell_ui):
    """Lifting after a swipe suppresses only that gesture's release click; an immediate fresh tap on a button lands
    without a 400 ms swallow window, and a magnified pinch does not trigger swipe-to-reply."""
    page, _, _ = loaded_browser_room(shell_ui, False, [f'Row {i}. ' + LONG_LINE for i in range(8)])
    # A magnified pinch sideways over a chat bubble does not stage a reply.
    page.evaluate('window.__pinch(true)')
    bubble = page.locator('#chat-messages .chat-bubble:not(.typing-bubble)').last
    box = bubble.bounding_box()
    y = box['y'] + box['height'] / 2
    page.evaluate("""({x, y}) => {
        const el = document.querySelector('#chat-messages .chat-bubble:not(.typing-bubble):last-of-type .message')
            || [...document.querySelectorAll('#chat-messages .chat-bubble:not(.typing-bubble) .message')].at(-1);
        const touch = (type, cx) => {
            const ev = new Event(type, {bubbles: true, cancelable: true});
            const pt = {identifier: 1, clientX: cx, clientY: y};
            Object.defineProperty(ev, 'touches', {value: type === 'touchend' ? [] : [pt]});
            Object.defineProperty(ev, 'changedTouches', {value: [pt]});
            el.dispatchEvent(ev);
        };
        touch('touchstart', x + 220);
        touch('touchmove', x + 120);
        touch('touchend', x + 120);
    }""", {'x': box['x'], 'y': y})
    settle(page)
    assert page.locator('#chat-input-area .chat-quote--composer').count() == 0, 'pinch never swipes to reply'
    page.evaluate('window.__pinch(false)')
    settle(page)

    # Swipe-to-reply on the bubble, release, and immediately (< 20 ms) tap "+" — the fresh tap must open the menu.
    opened = page.evaluate("""({x, y}) => {
        const el = [...document.querySelectorAll('#chat-messages .chat-bubble:not(.typing-bubble) .message')].at(-1);
        const touch = (type, cx) => {
            const ev = new Event(type, {bubbles: true, cancelable: true});
            const pt = {identifier: 2, clientX: cx, clientY: y};
            Object.defineProperty(ev, 'touches', {value: type === 'touchend' ? [] : [pt]});
            Object.defineProperty(ev, 'changedTouches', {value: [pt]});
            el.dispatchEvent(ev);
        };
        touch('touchstart', x + 220);
        touch('touchmove', x + 130);
        touch('touchend', x + 130);
        el.dispatchEvent(new MouseEvent('click', {bubbles: true, cancelable: true})); // release click swallowed
        const plus = document.querySelector('#chat-input-area [data-composer-plus]');
        plus.dispatchEvent(new Event('pointerdown', {bubbles: true})); // fresh tap starts immediately
        plus.click();
        return Boolean(document.querySelector('body > .chat-plus-menu'));
    }""", {'x': box['x'], 'y': y})
    assert opened, 'a fresh tap immediately after a swipe reaches its button'


CANCELABLE_SWIPE = """({target, ys}) => {
    const node = document.querySelector(target);
    const send = (type, y) => {
        const event = new Event(type, {bubbles: true, cancelable: true});
        Object.defineProperty(event, 'touches', {value: [{clientX: 150, clientY: y}]});
        node.dispatchEvent(event);
        return event.defaultPrevented;
    };
    send('touchstart', ys[0]);
    return ys.slice(1).map(y => send('touchmove', y));
}"""

BOX_PLACE = """({box, feed}) => {
    const b = document.getElementById('nested-box'), f = document.getElementById('chat-messages');
    b.scrollTop = box === 'end' ? b.scrollHeight : 0;
    f.scrollTop = feed === 'end' ? f.scrollHeight : feed;
    return {box: b.scrollTop, boxRoom: b.scrollHeight - b.clientHeight - b.scrollTop,
        feed: f.scrollTop, feedRoom: f.scrollHeight - f.clientHeight - f.scrollTop};
}"""


def test_a_swipe_from_a_box_at_its_end_moves_the_feed_until_the_page_edge(shell_ui):
    page, field, field_id = loaded_telegram_room(shell_ui, False, True, [LONG_LINE * 2] * 30)
    field.fill('Keep my draft.')
    page.evaluate('window.__viewport(508)')
    still(page)
    assert page.evaluate(GEOMETRY, field_id)['keyboard']
    middle = page.evaluate("""() => {
        const feed = document.getElementById('chat-messages'), box = document.createElement('div');
        box.id = 'nested-box';
        box.style.cssText = 'overflow-y: auto; height: 120px';
        box.innerHTML = '<p style="height: 600px; margin: 0">A bounded box</p>';
        feed.querySelector('[data-history-id="chat:15"]').after(box);
        feed.scrollTop += box.getBoundingClientRect().top - 200;
        return feed.scrollTop;
    }""")
    measured = {'newer': page.evaluate(BOX_PLACE, {'box': 'end', 'feed': middle})}
    measured['newerMoves'] = page.evaluate(CANCELABLE_SWIPE, {'target': '#nested-box p', 'ys': [300, 280, 240]})
    measured['edge'] = page.evaluate(BOX_PLACE, {'box': 'end', 'feed': 'end'})
    measured['edgeMoves'] = page.evaluate(CANCELABLE_SWIPE, {'target': '#nested-box p', 'ys': [300, 280, 240]})
    measured['older'] = page.evaluate(BOX_PLACE, {'box': 'top', 'feed': middle})
    page.wait_for_timeout(150)
    measured['olderMoves'] = page.evaluate(CANCELABLE_SWIPE, {'target': '#nested-box p', 'ys': [200, 220, 260]})
    facts(f'keyboard-swipe-through-box-{shell_ui["engine"]}', measured)
    assert measured['newer']['boxRoom'] <= 1 and measured['newer']['feedRoom'] > 100, measured
    assert measured['newerMoves'] == [False, False], measured
    assert measured['edge']['boxRoom'] <= 1 and measured['edge']['feedRoom'] <= 1, measured
    assert measured['edgeMoves'] == [True, True], measured
    assert measured['older']['box'] == 0 and measured['older']['feed'] > 100, measured
    assert measured['olderMoves'] == [False, False], measured
    page.wait_for_function('id => document.activeElement !== document.getElementById(id)', arg=field_id)
    assert field.input_value() == 'Keep my draft.'


FIND_GEOMETRY = """scope => {
    const room = document.querySelector(scope), sheet = room.querySelector('.chat-find');
    const bar = room.classList.contains('chat-instance-panel')
        ? room.closest('.project-panel').querySelector('.project-panel-bar')
        : room.querySelector('.chat-page-header');
    const box = el => el.getBoundingClientRect(), p = box(room), s = box(sheet), h = box(bar);
    return {barHeight: h.height, barFlow: getComputedStyle(bar).position, gap: s.top - p.top,
        fromBar: s.top - h.bottom, sheetHeight: s.height,
        focused: document.activeElement === sheet.querySelector('input'),
        keyboard: document.body.classList.contains('keyboard-open')};
}"""


def keep_find_sheet(page, name, geometry, clip, tmp_path):
    """The keyboard-raised Find sheet as published diagnostics when a test owner names a destination: the frame and exact
    geometry this test also keeps as local evidence (the frame bounded by its clip, at CSS size), in a producer folder that
    `diagnostic_directory` keys by `tmp_path / name`, so each engine, room and height has its own. Evidence only: a failed
    export keeps the environment to inspect, says only its exception's class in public output (a message can carry a local
    path) and never rewrites the verdict."""
    if not os.environ.get('OUROBOROS_TEST_EVIDENCE_ROOT'):
        return
    from ouroboros.test_evidence import diagnostic_directory, diagnostics_incomplete, report_diagnostic
    try:
        folder = diagnostic_directory(tmp_path / name, 'find-sheet-keyboard', publish=True)
        (folder / f'{name}.json').write_text(json.dumps(geometry, indent=1, sort_keys=True), encoding='utf-8')
        page.screenshot(path=str(folder / f'{name}.png'), clip=clip, scale='css')
    except Exception as exc:
        reason = type(exc).__name__
        report_diagnostic(f'TEST_DIAGNOSTICS_INCOMPLETE {name}: {reason}')
        diagnostics_incomplete(reason)


@pytest.mark.parametrize('height', [508, 250])
@pytest.mark.parametrize('room', ['main', 'project', 'telegram-project'])
def test_the_find_sheet_stays_a_step_under_the_rooms_bar_with_the_keyboard_up(shell_ui, room, height, tmp_path):
    fullscreen = room == 'telegram-project'
    page = telegram_phone(shell_ui, fullscreen=True) if fullscreen else phone(shell_ui, telegram=False)
    page.route('**/api/chat/history**', lambda route: route.fulfill(json=history([LONG_LINE] * 12)))
    page.goto(shell_ui['url'])
    if fullscreen:
        page.wait_for_function("document.documentElement.hasAttribute('data-telegram-fullscreen')")
    page.locator('#chat-messages[data-history-hydrated="true"]').wait_for()
    scope = '#page-chat'
    if room != 'main':
        open_project(page)
        scope = '.project-panel .chat-instance-panel'
    feed = page.locator('#chat-messages' if room == 'main' else f'{scope} .chat-messages')
    feed.click(position={'x': 40, 'y': 200})
    page.keyboard.press('ControlOrMeta+f')
    page.wait_for_function("s => document.querySelector(s + ' .chat-find input') === document.activeElement", arg=scope)
    still(page)
    rested = page.evaluate(FIND_GEOMETRY, scope)
    page.evaluate(f'window.__viewport({height})')  # the page's own keyboard detection, not a class the test adds
    page.wait_for_function("document.body.classList.contains('keyboard-open')")
    still(page)
    raised = page.evaluate(FIND_GEOMETRY, scope)
    name = f'find-sheet-keyboard-{height}-{room}-{shell_ui["engine"]}'
    facts(name, {'rested': rested, 'raised': raised})
    shot(page, name, {'x': 0, 'y': 0, 'width': 390, 'height': height})
    keep_find_sheet(page, name, {'rested': rested, 'raised': raised},
                    {'x': 0, 'y': 0, 'width': 390, 'height': height}, tmp_path)
    assert rested['focused'] and not rested['keyboard'] and rested['sheetHeight'] > 0, rested
    assert raised['focused'] and raised['keyboard'] and raised['sheetHeight'] > 0, raised
    assert 0 <= rested['fromBar'] <= 16, ('rested: the sheet floats a step under the floating bar', rested)
    if fullscreen:
        assert raised['barFlow'] == 'absolute', raised
        assert abs(raised['fromBar'] - rested['fromBar']) <= 1, (rested, raised)
    elif room == 'project':
        assert raised['barFlow'] == 'relative', raised
        assert 0 <= raised['gap'] <= 16 and abs(raised['fromBar']) <= 16, \
            ('keyboard up: a full bar of extra gap is the #3653 bug', raised)
    else:
        assert raised['barFlow'] == 'relative', raised
        # The short viewport can shrink the bar itself; keep Find's gap from its lower edge.
        assert abs(raised['fromBar'] - rested['fromBar']) <= 1, (rested, raised)


CAP = """id => { const f = document.getElementById(id); return {height: f.getBoundingClientRect().height,
    cap: parseFloat(getComputedStyle(f).maxHeight), sizing: f.dataset.sizing,
    keyboard: document.body.classList.contains('keyboard-open')}; }"""


@pytest.mark.parametrize('project_room', [False, True], ids=['main', 'project'])
def test_a_long_draft_gets_its_resting_cap_back_after_a_dismissal_without_resize(shell_ui, project_room):
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
    assert field.input_value().count('\n') == 11


# The caret's line `y` counted from the end (no line after the caret's wraps). `shown`: the reader sees it, the one
# meaning the field's sizing keeps it by (chat_input_sizing.js caretRowShown); `whole`: its full line box in view.
CARET = r"""async f => {
    const { caretRowShown } = await import('/static/modules/chat_input_sizing.js');
    const s = getComputedStyle(f), line = parseFloat(s.lineHeight);
    const start = parseFloat(s.paddingTop), end = parseFloat(s.paddingBottom);
    const rows = Math.round((f.scrollHeight - start - end) / line);
    const y = start + line * (rows - (f.value.slice(f.selectionEnd).split('\n').length - 1) - 1);
    return {top: f.scrollTop, height: f.clientHeight, width: f.getBoundingClientRect().width, extent: f.scrollHeight, y, line,
        shown: caretRowShown({top: f.scrollTop, height: f.clientHeight}, y, line),
        whole: f.scrollTop <= y + 1 && y + line <= f.scrollTop + f.clientHeight + 1,
        selection: [f.selectionStart, f.selectionEnd], focused: document.activeElement === f};
}"""

WRAPPED_DRAFT = '\n'.join([LONG_LINE.strip()] * 6 + [f'Line {n:02d}.' for n in range(20)])


@pytest.mark.parametrize('project_room', [False, True], ids=['main', 'project'])
def test_a_widening_telegram_composer_keeps_its_visible_middle_caret(shell_ui, project_room):
    page, field, _ = loaded_telegram_room(shell_ui, project_room, True, [LONG_LINE] * 12)
    field.focus()
    page.evaluate('window.__viewport(250)')
    page.wait_for_function("document.body.classList.contains('keyboard-open')")
    still(page)
    text = WRAPPED_DRAFT
    page.keyboard.insert_text(text)
    i = text.index('Line 09') + 3
    field.evaluate('(f, i) => f.setSelectionRange(i, i)', i)
    page.keyboard.insert_text('x')
    text = text[:i] + 'x' + text[i:]
    field.evaluate('(f, g) => { f.scrollTop = g.y - parseFloat(getComputedStyle(f).paddingTop); }', field.evaluate(CARET))
    still(page)
    narrow = field.evaluate(CARET)
    assert narrow['whole'] and narrow['top'] > 0, narrow
    page.evaluate('window.__viewport(508)')
    still(page)
    wide = field.evaluate(CARET)
    facts(f'composer-caret-widening-{"project" if project_room else "main"}-{shell_ui["engine"]}',
          {'narrow': narrow, 'wide': wide})
    assert wide['width'] > narrow['width'] and wide['y'] < narrow['y'], (narrow, wide)
    assert wide['whole'] and wide['focused'] and wide['selection'] == narrow['selection'], (narrow, wide)
    assert field.input_value() == text


@pytest.mark.parametrize('reveal', ['engine', 'caret-box'])
@pytest.mark.parametrize('project_room', [False, True], ids=['main', 'project'])
def test_a_caret_moved_without_an_edit_stays_in_view_when_its_row_narrows_the_field(shell_ui, project_room, reveal):
    """`caret-box`: the place Linux Chromium's own reveal leaves (its caret box, shorter than the line, at the top
    edge; the line box 1.3 px above it), made here in every engine: still a caret the reader sees and keeps."""
    page, field, field_id = loaded_telegram_room(shell_ui, project_room, True, [LONG_LINE] * 12)
    field.focus()
    page.evaluate('window.__viewport(250)')
    page.wait_for_function("document.body.classList.contains('keyboard-open')")
    still(page)
    page.keyboard.insert_text(WRAPPED_DRAFT)
    for _ in range(10):
        page.keyboard.press('ArrowUp')
    settle(page)
    if reveal == 'caret-box':
        field.evaluate('(f, g) => { f.scrollTop = Math.ceil(g.y + 2); }', field.evaluate(CARET))
        still(page)
    moved = field.evaluate(CARET)
    assert moved['shown'] and moved['selection'][1] < len(WRAPPED_DRAFT) - 60, moved
    assert reveal == 'engine' or not moved['whole'], moved  # the line box starts above the top edge
    area = '#' + re.sub(r'input$', 'input-area', field_id)
    page.locator(f'{area} .chat-send-inline').evaluate('''b => { b.closest('.chat-send-group').dataset.busy = '1';
        b.disabled = true; b.textContent = b.title = 'Sending'; b.style.minWidth = `${b.offsetWidth + 40}px`; }''')
    still(page)
    narrowed = field.evaluate(CARET)
    facts(f'composer-caret-unedited-{reveal}-{"project" if project_room else "main"}-{shell_ui["engine"]}',
          {'moved': moved, 'narrowed': narrowed})
    assert narrowed['width'] < moved['width'] and narrowed['y'] > moved['y'], (moved, narrowed)
    assert narrowed['shown'] and narrowed['focused'] and narrowed['selection'] == moved['selection'], (moved, narrowed)
    assert field.input_value() == WRAPPED_DRAFT


def test_the_typing_read_band_keeps_the_bars_edge_where_the_reach_reads_back_unresolved(shell_ui):
    page, field, _ = loaded_telegram_room(shell_ui, False, True, [LONG_LINE] * 12)
    field.focus()
    page.evaluate('window.__viewport(508)')
    page.wait_for_function("document.body.classList.contains('keyboard-open')")
    still(page)
    band = page.evaluate("""async () => {
        const { readBand } = await import('/modules/project_read_state.js');
        const bar = document.querySelector('#page-chat .chat-page-header'), feed = document.getElementById('chat-messages');
        const resolved = parseFloat(getComputedStyle(bar).getPropertyValue('--edge-reach'));
        const real = window.getComputedStyle;
        window.getComputedStyle = (el, pseudo) => {
            const style = real.call(window, el, pseudo);
            if (pseudo) return style;
            return new Proxy(style, { get: (target, key) => key === 'getPropertyValue'
                ? name => (name === '--edge-reach' ? 'clamp(0px, 508px - 47px - 250px, 32px)' : target.getPropertyValue(name))
                : (typeof target[key] === 'function' ? target[key].bind(target) : target[key]) });
        };
        try {
            return { resolved, unresolved: getComputedStyle(bar).getPropertyValue('--edge-reach'),
                top: readBand(feed, bar, null).top, bar: bar.getBoundingClientRect().bottom };
        } finally { window.getComputedStyle = real; }
    }""")
    facts(f'typing-read-band-unresolved-{shell_ui["engine"]}', band)
    assert band['resolved'] == 32 and band['unresolved'].startswith('clamp('), band
    assert abs(band['top'] - (band['bar'] + 32)) <= 0.5, band


@pytest.mark.parametrize('project_room', [False, True], ids=['main', 'project'])
def test_a_shrinking_telegram_composer_keeps_its_visible_caret(shell_ui, project_room):
    page, field, _ = loaded_telegram_room(shell_ui, project_room, True, [LONG_LINE] * 12)

    def height(h):
        page.evaluate('h => window.__viewport(h)', h)
        still(page)

    def draft_view(*, shrink=False):
        return field.evaluate(r'''(f, shrink) => {
            if (shrink) window.__viewport(250);
            const s = getComputedStyle(f), line = parseFloat(s.lineHeight);
            const start = parseFloat(s.paddingTop), end = parseFloat(s.paddingBottom);
            const rows = Math.round((f.scrollHeight - start - end) / line);
            const hardAfter = f.value.slice(f.selectionEnd).split('\n').length - 1;
            const y = start + line * (rows - hardAfter - 1);
            return { top: f.scrollTop, height: f.clientHeight, width: f.clientWidth, extent: f.scrollHeight, y, line,
                selection: [f.selectionStart, f.selectionEnd], focused: document.activeElement === f };
        }''', shrink)

    def visible(g):
        return g['top'] <= g['y'] + 1 and g['y'] + g['line'] <= g['top'] + g['height'] + 1

    field.focus()
    samples = {}
    for scenario in ('end', 'wrapped-end', 'middle', 'reading'):
        height(508)
        field.press('ControlOrMeta+A')
        text = '\n'.join(f'Line {n:02d}.' for n in range(4 if scenario == 'end' else 20))
        if scenario != 'end':
            line = field.evaluate('''f => {
                const s = getComputedStyle(f), ink = document.createElement('canvas').getContext('2d');
                ink.font = `${s.fontWeight} ${s.fontSize} ${s.fontFamily}`;
                let line = 'Check dates:';
                for (let n = 1; n <= 31 && ink.measureText(`${line} ${n}`).width < f.clientWidth - 20; n++) line += ` ${n}`;
                return line;
            }''')
            text = {'wrapped-end': f'Keep this note.\n{line}', 'middle': f'{line}\n{text}',
                    'reading': text + '\n' + '\n'.join([line] * 13)}[scenario]
        page.keyboard.insert_text(text)
        if scenario == 'middle':
            i = text.index('Line 09') + 3
            field.evaluate('(f, i) => f.setSelectionRange(i, i)', i)
            page.keyboard.insert_text('x')
            text = text[:i] + 'x' + text[i:]
            field.evaluate('(f, g) => { f.scrollTop = g.y + g.line + 5 - f.clientHeight; }', draft_view())
        if scenario == 'reading':
            field.evaluate('f => { f.scrollTop = 38; }')
        before = draft_view()
        assert visible(before) == (scenario != 'reading'), (scenario, before)
        capped = draft_view(shrink=True)
        still(page)
        after = draft_view()
        samples[scenario] = {'before': before, 'capped': capped, 'folded': after}
        if scenario != 'end':
            assert capped['height'] == after['height'] and capped['width'] > after['width'], (capped, after)
            assert after['extent'] > capped['extent'], (capped, after)
        assert after['height'] < before['height'] and after['focused'] and after['selection'] == before['selection']
        assert field.input_value() == text
        if scenario == 'reading':
            assert before['top'] == after['top'] == 38, (scenario, before, after)
            page.keyboard.insert_text('!')
            typed = samples[scenario]['typed'] = draft_view()
            assert field.input_value() == text + '!' and visible(typed), typed
        else:
            assert visible(after), (scenario, before, capped, after)
    facts(f'composer-caret-{"project" if project_room else "main"}-{shell_ui["engine"]}', samples)
