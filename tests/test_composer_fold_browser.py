"""The row of capsules folds while the owner reads back through a room's history (owner 19.10, 2A; docs/DESIGN.md
"Composer owner controls"; web/modules/composer_fold.js).

On a phone, reading back folds the row (☰, Swarm, Context, Effort) away, keeping its fit, and ☰ waits beside the
field on its last line, beside "+", a draft of several lines included, on the field's leading side in either direction; the way toward
the latest and the field's focus bring it back; a tap on the folded ☰ opens the drawer; reduced motion folds it at
once; a desk never folds it. Uses the static-server/API
fixture. The feed is moved as the owner moves it, by a finger (CDP touch, Chromium's input pipeline) or a wheel: a
scroll the page makes itself is not the owner's, and the room pins it back to the latest.
"""
from __future__ import annotations

import pytest

from tests import test_subscription_setup_browser as setup_browser
from tests.test_design_system_gate_browser import seed_page
from tests.test_shell_interactions_browser import boot, chromium_ui, evidence, shell_ui

subscription_ui = setup_browser.subscription_ui
pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]
__all__ = ['chromium_ui', 'shell_ui', 'subscription_ui']

SEED = """() => { const f = document.querySelector('#chat-messages');
    for (let i = 0; i < 40; i += 1) { const r = document.createElement('div'); r.style.minHeight = '90px';
        r.textContent = 'row ' + i; f.append(r); }
    f.scrollTop = f.scrollHeight; }"""


def drag(page, dy, strokes=3):
    """A finger moves the feed by about `dy` per stroke (positive reads back), on the feed's right side."""
    cdp = page.context.new_cdp_session(page)
    try:
        for _ in range(strokes):
            y0 = 300 if dy > 0 else 600
            for i in range(19):
                cdp.send('Input.dispatchTouchEvent', {'type': 'touchStart' if i == 0 else 'touchMove',
                                                      'touchPoints': [{'x': 380, 'y': y0 + dy * i / 18, 'id': 1}]})
                page.wait_for_timeout(12)
            cdp.send('Input.dispatchTouchEvent', {'type': 'touchEnd', 'touchPoints': []})
            page.wait_for_timeout(250)
    finally:
        cdp.detach()
    page.wait_for_timeout(450)
FACTS = """() => { const wrap = document.querySelector('#page-chat .chat-input-wrap');
    const row = wrap.querySelector('.chat-toolbar-row'), text = wrap.querySelector('.chat-field');
    const button = row.querySelector(':scope > .composer-menu');
    const menu = button.getBoundingClientRect(), field = text.getBoundingClientRect();
    const plus = wrap.querySelector('[data-composer-plus]').getBoundingClientRect();
    const capsule = row.querySelector('.chat-subagents-mode');
    return {folded: 'folded' in wrap.dataset, rowHeight: Math.round(row.getBoundingClientRect().height),
            menuMiddle: menu.top + menu.height / 2, fieldMiddle: field.top + field.height / 2,
            plusMiddle: plus.top + plus.height / 2,
            menuRight: menu.right, fieldLeft: field.left, menuLeft: menu.left, fieldRight: field.right,
            // What a finger reaches at ☰'s centre: ☰ itself, never "+" painted over it.
            menuHit: button.contains(document.elementFromPoint(menu.left + menu.width / 2, menu.top + menu.height / 2)),
            fieldHeight: Math.round(field.height),
            capsule: getComputedStyle(capsule).visibility, fit: row.dataset.ownerCompact || '',
            area: Math.round(document.querySelector('#chat-input-area').getBoundingClientRect().height)}; }"""
# Every frame for `ms` from the fold's start (the composer takes data-folded): how far ☰ has gone down, the room the
# field makes on its leading side, the opacity of the first capsule (Subagents), and the transitions running on the
# line, the field and each control on the line. No fold within 5 s gives no frames: the facts read next say so.
FOLD_FRAMES = """ms => new Promise((resolve) => {
    const wrap = document.querySelector('#page-chat .chat-input-wrap');
    const row = wrap.querySelector('.chat-toolbar-row'), text = wrap.querySelector('.chat-field');
    const menu = row.querySelector(':scope > .composer-menu'), capsule = row.querySelector(':scope > .chat-subagents-mode');
    const lead = getComputedStyle(text).direction === 'rtl' ? 'marginRight' : 'marginLeft';
    const down = (el) => { const t = getComputedStyle(el).transform; return t === 'none' ? 0 : new DOMMatrixReadOnly(t).m42; };
    const frames = [];
    const sample = () => frames.push({menu: down(menu), margin: parseFloat(getComputedStyle(text)[lead]),
        capsule: Number(getComputedStyle(capsule).opacity),
        moving: [row, text, ...row.children].flatMap((el) => el.getAnimations().map((a) => a.transitionProperty || '?'))});
    const observer = new MutationObserver(() => {
        if (!('folded' in wrap.dataset)) return;
        observer.disconnect();
        clearTimeout(giveUp);
        const start = performance.now();
        sample();
        const step = () => { sample(); if (performance.now() - start < ms) requestAnimationFrame(step); else resolve(frames); };
        requestAnimationFrame(step);
    });
    observer.observe(wrap, {attributes: true, attributeFilter: ['data-folded']});
    const giveUp = setTimeout(() => { observer.disconnect(); resolve(frames); }, 5000);
})"""


def read_back_by_wheel(page, notches=8):
    """The owner reads back with the wheel, 120 px a notch (WebKit has no synthetic finger here)."""
    page.mouse.move(200, 400)
    for _ in range(notches):
        page.mouse.wheel(0, -120)
        page.wait_for_timeout(60)
    page.wait_for_timeout(900)


def test_reading_back_folds_the_row_and_the_way_down_brings_it_back(chromium_ui, tmp_path):
    page = seed_page(chromium_ui, 390, 844)
    facts = {}
    try:
        boot(page, chromium_ui)
        page.evaluate(SEED)
        page.wait_for_timeout(300)
        facts['latest'] = page.evaluate(FACTS)
        drag(page, 220)
        facts['reading'] = page.evaluate(FACTS)
        drag(page, -150, strokes=1)
        facts['back'] = page.evaluate(FACTS)
        drag(page, 220)
        facts['again'] = page.evaluate(FACTS)
        page.locator('#chat-input').focus()
        page.wait_for_timeout(450)
        facts['writing'] = page.evaluate(FACTS)
    finally:
        evidence(tmp_path, 'composer-fold-phone', facts)
        page.close()
    latest, reading, back, again, writing = (facts[k] for k in ('latest', 'reading', 'back', 'again', 'writing'))
    assert latest['folded'] is False and latest['capsule'] == 'visible', ('at the latest the row is there', latest)
    assert reading['folded'] is True and reading['capsule'] == 'hidden' and reading['rowHeight'] == 0, reading
    assert abs(reading['menuMiddle'] - reading['plusMiddle']) <= 1.5, ('☰ waits on the field\'s last line, beside "+"', reading)
    assert reading['menuRight'] < reading['fieldLeft'], ('beside the field, not over it', reading)
    assert reading['fit'] == latest['fit'], ('folded, the row keeps its fit (☰ off its line misled the refit)', reading, latest)
    assert reading['area'] <= latest['area'] - 30, ('the composer gives its line back to the feed', reading, latest)
    assert back['folded'] is False and back['capsule'] == 'visible' and back['area'] == latest['area'], ('the way down', back)
    assert again['folded'] is True and writing['folded'] is False, ('the field\'s focus brings it back', writing)


def test_a_folded_draft_of_several_lines_keeps_the_menu_on_its_last_line(chromium_ui, tmp_path):
    """Folded, the field is narrower by ☰'s place, so a draft of several lines wraps into one more line and grows from
    its bottom; ☰ stays on its last line beside "+", never its middle (owner 20.10, 1C: the controls ride the last
    line)."""
    page = seed_page(chromium_ui, 390, 844)
    facts = {}
    try:
        boot(page, chromium_ui)
        page.evaluate(SEED)
        page.locator('#chat-input').fill('Первая строка черновика\nвторая\nтретья\nчетвёртая')
        page.locator('#chat-input').blur()
        page.wait_for_timeout(300)
        facts['open'] = page.evaluate(FACTS)
        drag(page, 220)
        facts['folded'] = page.evaluate(FACTS)
    finally:
        evidence(tmp_path, 'composer-fold-draft', facts)
        page.close()
    folded = facts['folded']
    assert folded['folded'] is True and folded['fieldHeight'] > facts['open']['fieldHeight'], (
        'setup: folded, the draft wraps into one more line', facts)
    assert abs(folded['menuMiddle'] - folded['plusMiddle']) <= 1.5, ('☰ waits on the draft\'s last line', folded)
    assert folded['plusMiddle'] - folded['fieldMiddle'] >= 20, ('setup: the last line is not the middle', folded)
    assert folded['menuRight'] < folded['fieldLeft'], ('beside the field, not over it', folded)


def test_a_thumbs_tap_on_the_folded_menu_opens_the_drawer(chromium_ui, tmp_path):
    """Folded, ☰ waits beside the field. A thumb's tap on it (held 90 ms, hit-tested where it ends) opens the drawer:
    the press leaves the row folded, so neither ☰ nor 📎 beside it slides under the finger first (the row used to come
    back on any press in the composer, and the tap landed on 📎 and opened the file picker)."""
    page = seed_page(chromium_ui, 390, 844)
    facts = {}
    try:
        boot(page, chromium_ui)
        page.evaluate(SEED)
        page.wait_for_timeout(300)
        drag(page, 220)
        facts['folded'] = page.evaluate(FACTS)
        menu = page.locator('#page-chat .chat-toolbar-row > .composer-menu').bounding_box()
        cdp = page.context.new_cdp_session(page)
        try:
            point = {'x': menu['x'] + menu['width'] / 2, 'y': menu['y'] + menu['height'] / 2, 'id': 1}
            cdp.send('Input.dispatchTouchEvent', {'type': 'touchStart', 'touchPoints': [point]})
            page.wait_for_timeout(90)
            cdp.send('Input.dispatchTouchEvent', {'type': 'touchEnd', 'touchPoints': []})
        finally:
            cdp.detach()
        page.wait_for_timeout(500)
        facts['drawer'] = page.evaluate("() => document.getElementById('primary-sidebar').classList.contains('open')")
        facts['after'] = page.evaluate(FACTS)
    finally:
        evidence(tmp_path, 'composer-fold-menu-tap', facts)
        page.close()
    assert facts['folded']['folded'] is True, ('reading back folded the row', facts)
    assert facts['drawer'] is True and facts['after']['folded'] is True, ('the tap opened the drawer, nothing moved', facts)


@pytest.mark.parametrize('motion', ['no-preference', 'reduce'])
@pytest.mark.parametrize('direction', ['ltr', 'rtl'])
def test_the_folded_menu_waits_on_the_fields_leading_side_and_moves_only_where_motion_is_welcome(
        shell_ui, tmp_path, direction, motion):
    """☰ leads the row from the line's start, so folded it waits beside the field on that side: the left, or the right
    in a right-to-left interface (i18n.js sets dir=rtl), where it used to drop onto "+" while the field made room on
    the left. Asked for reduced motion, the row folds at once, ☰ and the capsules with it (their own rules outranked
    the reduced-motion one, and ☰ still glided); otherwise all of it moves."""
    page = seed_page(shell_ui, 390, 844)
    page.emulate_media(reduced_motion=motion)
    facts = {'engine': shell_ui['engine'], 'direction': direction, 'motion': motion}
    try:
        boot(page, shell_ui)
        if direction == 'rtl':
            page.evaluate("() => { document.documentElement.dir = 'rtl'; }")  # what i18n.js does for an RTL language
        page.evaluate(SEED)
        page.wait_for_timeout(300)
        page.evaluate(f"ms => {{ window.__foldFrames = ({FOLD_FRAMES})(ms); }}", 420)
        read_back_by_wheel(page)
        facts['frames'] = page.evaluate('() => window.__foldFrames')
        facts['folded'] = page.evaluate(FACTS)
    finally:
        evidence(tmp_path, f'composer-fold-{direction}-{motion}-{shell_ui["engine"]}', facts)
        page.close()
    folded, frames = facts['folded'], facts['frames']
    assert folded['folded'] is True and abs(folded['menuMiddle'] - folded['plusMiddle']) <= 1.5, folded
    assert folded['menuHit'] is True, ('folded, a finger reaches ☰, not "+"', folded)
    if direction == 'ltr':
        assert folded['menuRight'] < folded['fieldLeft'], ('beside the field, on its left', folded)
    else:
        assert folded['menuLeft'] > folded['fieldRight'], ('beside the field, on its right', folded)
    assert frames, ('setup: the fold was sampled from its start', facts)
    end = frames[-1]
    assert end['menu'] > 4 and end['margin'] > 30 and end['capsule'] == 0 and end['moving'] == [], ('setup: settled folded', end)
    if motion == 'reduce':
        assert all(f['moving'] == [] for f in frames), ('nothing moves', frames)
        assert all((f['menu'], f['margin'], f['capsule']) == (end['menu'], end['margin'], 0) for f in frames), (
            'from its first frame the row is folded', frames)
    else:
        lead = 'margin-left' if direction == 'ltr' else 'margin-right'
        assert {'transform', 'opacity', 'max-height', lead} <= set(frames[0]['moving']), (
            "at the fold ☰, the capsules, the line and the field's leading side all start moving", frames[0])
        assert any(0 < f['menu'] < end['menu'] for f in frames), ('and ☰ is seen on its way down', frames)


def test_a_desk_never_folds_the_row(chromium_ui, tmp_path):
    page = seed_page(chromium_ui, 1280, 860)
    facts = {}
    try:
        boot(page, chromium_ui)
        page.evaluate(SEED)
        page.wait_for_timeout(300)
        page.mouse.move(700, 400)
        for _ in range(6):
            page.mouse.wheel(0, -200)
            page.wait_for_timeout(120)
        page.wait_for_timeout(500)
        facts['reading'] = page.evaluate(FACTS)
        facts['scrolled'] = page.evaluate("() => { const f = document.querySelector('#chat-messages'); return f.scrollHeight - f.clientHeight - f.scrollTop; }")
    finally:
        evidence(tmp_path, 'composer-fold-desk', facts)
        page.close()
    assert facts['scrolled'] > 300, ('the owner read back', facts)
    assert facts['reading']['folded'] is False and facts['reading']['capsule'] == 'visible', facts
