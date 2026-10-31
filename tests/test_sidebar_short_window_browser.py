"""The sidebar's foot in a short window (owner 23.10; docs/DESIGN.md "Shell").

A window too short for the sidebar's zones lets the whole column scroll with Main pinned. The foot, the version
and the update, used to be that flow's last row, below the fold until the column was scrolled to its end; a chosen
Interface size reaches that mode from about 130 % in a 680 px desktop window and from about 120 % on an iPhone. The
owner chose to keep the foot on the bottom edge there too (answer A). Uses the static-server/API fixture
(test_subscription_setup_browser) with the gate's 30 Projects; never starts the runtime or reads live data.
"""
from __future__ import annotations

import pytest

from tests import test_subscription_setup_browser as setup_browser
from tests.test_design_system_gate_browser import seed_page, settle, sidebar_metrics
from tests.test_shell_interactions_browser import boot, chromium_ui, evidence, open_drawer, shell_ui

subscription_ui = setup_browser.subscription_ui
pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]
__all__ = ['chromium_ui', 'shell_ui', 'subscription_ui']

SHORT = "matchMedia('(max-height: 540px), (pointer: coarse) and (max-height: 600px)').matches"
# Where the foot is: its box against the column's bottom edge (0 = on the edge), the version's distance from that edge,
# whether the foot is what a finger finds at its middle (the rows pass under it), and the tools around it.
FOOT = """() => {
    const side = document.getElementById('primary-sidebar');
    const meta = side.querySelector('.sidebar-foot-meta');
    const box = (e) => e.getBoundingClientRect();
    const edge = box(side).bottom, m = box(meta);
    const under = document.elementFromPoint(m.left + 4, m.top + m.height / 2);
    return {scrolls: side.scrollHeight > side.clientHeight + 1, top: Math.round(side.scrollTop),
            position: getComputedStyle(meta).position, metaBottom: Math.round((m.bottom - edge) * 10) / 10,
            version: Math.round((edge - box(document.getElementById('nav-version')).bottom) * 10) / 10,
            onTop: Boolean(under && meta.contains(under)),
            opaque: getComputedStyle(meta).backgroundColor === getComputedStyle(side).backgroundColor
                && getComputedStyle(meta).backgroundColor !== 'rgba(0, 0, 0, 0)',
            toolsClear: m.top >= box(side.querySelector('[data-nav-page="settings"]')).bottom - 0.5,
            gapToTools: Math.round(box(side.querySelector('[data-nav-page="files"]')).top
                - box(document.getElementById('nav-projects')).bottom)};
}"""
SCROLL = "f => { const s = document.getElementById('primary-sidebar'); s.scrollTop = f * (s.scrollHeight - s.clientHeight); }"


@pytest.mark.parametrize(('width', 'height', 'touch', 'tall'),
                         [(1100, 500, False, 760), (844, 390, True, 760), (390, 570, True, 844)],
                         ids=['desk', 'landscape-phone', 'phone-drawer'])
def test_a_short_window_keeps_the_version_and_the_update_on_the_bottom_edge(shell_ui, tmp_path, width, height, touch,
                                                                            tall):
    page = seed_page(shell_ui, width, height, touch=touch)
    facts = {'engine': shell_ui['engine'], 'size': [width, height]}
    try:
        boot(page, shell_ui)
        if width < 720:
            open_drawer(page)
        facts['short'] = page.evaluate(SHORT)
        facts['scrollers'] = [row['node'] for row in sidebar_metrics(page)]
        for at in (0, 0.5, 1):
            page.evaluate(SCROLL, at)
            settle(page)
            facts[f'at-{at}'] = page.evaluate(FOOT)
        # Collapsed, the list takes less of the column: the foot stays on the edge and the tools follow the list.
        page.evaluate(SCROLL, 0)
        page.locator('#nav-projects-toggle').click()
        settle(page)
        facts['collapsed'] = page.evaluate(FOOT)
        page.locator('#nav-projects-toggle').click()
        # The same page in a tall window: the zones fit, the foot is the tools' box's last row again.
        page.set_viewport_size({'width': width, 'height': tall})
        settle(page)
        facts['tall_short'] = page.evaluate(SHORT)
        facts['tall'] = page.evaluate(FOOT)
    finally:
        evidence(tmp_path, f'sidebar-short-foot-{width}x{height}-{shell_ui["engine"]}', facts)
        page.close()
    tall = facts['tall']
    assert facts['short'] is True and facts['scrollers'] == ['primary-sidebar'], facts
    assert facts['at-0']['scrolls'] and facts['at-0']['top'] == 0 < facts['at-0.5']['top'] < facts['at-1']['top'], facts
    for state in ('at-0', 'at-0.5', 'at-1', 'collapsed'):
        foot = facts[state]
        assert foot['position'] == 'sticky' and abs(foot['metaBottom']) <= 1, ('the foot is on the edge', state, facts)
        assert abs(foot['version'] - tall['version']) <= 1, ('its text sits where a tall window puts it', state, facts)
        assert foot['onTop'] and foot['opaque'], ('the rows pass under an opaque foot', state, facts)
    assert facts['at-1']['toolsClear'], ('at the end of the scroll the last tool is above the foot', facts)
    assert abs(facts['collapsed']['gapToTools'] - tall['gapToTools']) <= 1, ('the tools follow the list', facts)
    # Outside the short-window mode the foot is not pinned: it rests in the tools' box, on the column's padding.
    assert facts['tall_short'] is False and tall['position'] == 'static' and not tall['scrolls'], facts
    assert tall['toolsClear'] and tall['version'] >= 12, facts


# The focused element against the two pinned bars: under the head, under the foot, or clear of both.
FOCUS = """() => {
    const side = document.getElementById('primary-sidebar');
    const head = side.querySelector('.sidebar-head'), meta = side.querySelector('.sidebar-foot-meta');
    const f = document.activeElement;
    if (!f || !side.contains(f) || f === side) return null;
    const r = f.getBoundingClientRect(), h = head.getBoundingClientRect(), m = meta.getBoundingClientRect();
    return {name: (f.getAttribute('aria-label') || f.textContent || '').trim().slice(0, 40), top: r.top, bottom: r.bottom,
            underHead: !head.contains(f) && r.top < h.bottom - 0.5, underFoot: !meta.contains(f) && r.bottom > m.top + 0.5,
            scrollTop: side.scrollTop};
}"""


def walk(page, key, limit=120):
    """Press `key` until focus leaves the sidebar; every stop on the way."""
    stops = []
    for _ in range(limit):
        page.keyboard.press(key)
        stop = page.evaluate(FOCUS)
        if stop is None:
            break
        stops.append(stop)
    return stops


def test_a_row_reached_with_tab_is_never_left_under_a_pinned_bar(chromium_ui, tmp_path):
    """Tab walks the column down to its foot and back up to Main: the browser brings each row into view, and the two
    bars' heights are the rows' scroll margin, so no focused row stays under the pinned foot or the pinned head.
    (WebKit on macOS does not Tab to buttons unless the system's keyboard navigation is on, so Chromium walks it.)"""
    page = seed_page(chromium_ui, 1100, 500, touch=False)
    facts = {'engine': 'chromium'}
    try:
        boot(page, chromium_ui)
        facts['short'] = page.evaluate(SHORT)
        page.locator('#primary-sidebar [data-nav-page="chat"]').first.focus()
        facts['down'] = walk(page, 'Tab')
        page.locator('#primary-sidebar .sidebar-foot [data-nav-page="settings"]').focus()
        facts['up'] = walk(page, 'Shift+Tab')
    finally:
        evidence(tmp_path, 'sidebar-short-focus-chromium', facts)
        page.close()
    down, up = facts['down'], facts['up']
    assert facts['short'] is True and len(down) > 60 and len(up) > 60, facts
    assert down[-1]['scrollTop'] > 0, ('the walk scrolled the column', facts)
    # Back at the head, Main leaves the column where the row before it did: a bar's own control moves nothing.
    assert up[-1]['name'] == 'Main Chat' and up[-1]['scrollTop'] == up[-2]['scrollTop'] < down[-1]['scrollTop'], facts
    hidden = [(way, stop['name']) for way, stops in (('down', down), ('up', up)) for stop in stops
              if stop['underFoot'] or stop['underHead']]
    assert not hidden, ('a focused row under a pinned bar', hidden, facts)


# The two pinned bars: the heights the rows' scroll margin holds against their real border boxes.
BARS = """() => {
    const side = document.getElementById('primary-sidebar');
    const css = getComputedStyle(side);
    return {head: side.querySelector('.sidebar-head').getBoundingClientRect().height,
            foot: side.querySelector('.sidebar-foot-meta').getBoundingClientRect().height,
            headPad: parseFloat(css.getPropertyValue('--sidebar-head-h')),
            footPad: parseFloat(css.getPropertyValue('--sidebar-foot-h'))};
}"""
LATE = {
    # The macOS shell writes its title band once the window has measured it (launcher_window_chrome.py).
    'title-band': "() => { const r = document.documentElement; r.dataset.windowChrome = 'integrated';"
                  " r.style.setProperty('--window-chrome-top', '52px'); }",
    # Telegram reports its safe area after the app has started (host_telegram.js mirrorInsets).
    'safe-area': "() => document.documentElement.style.setProperty('--tg-safe-area-inset-bottom', '34px')",
}


@pytest.mark.parametrize(('width', 'height', 'touch', 'late', 'bar', 'grows'),
                         [(1100, 500, False, 'title-band', 'head', 52), (390, 570, True, 'safe-area', 'foot', 34)],
                         ids=['desk-title-band', 'drawer-safe-area'])
def test_the_bars_measured_heights_follow_a_band_or_a_safe_area_that_arrives_later(shell_ui, tmp_path, width, height,
                                                                                   touch, late, bar, grows):
    """The bars' measured heights stay their real heights when only their padding changes after the first paint, so
    the rows' scroll margin still keeps a focused row clear of both."""
    page = seed_page(shell_ui, width, height, touch=touch)
    facts = {'engine': shell_ui['engine'], 'late': late}
    try:
        boot(page, shell_ui)
        if width < 720:
            open_drawer(page)
        facts['short'] = page.evaluate(SHORT)
        facts['before'] = page.evaluate(BARS)
        page.evaluate(LATE[late])
        settle(page)
        page.wait_for_timeout(150)
        facts['after'] = page.evaluate(BARS)
    finally:
        evidence(tmp_path, f'sidebar-short-late-{late}-{shell_ui["engine"]}', facts)
        page.close()
    before, after = facts['before'], facts['after']
    assert facts['short'] is True, facts
    assert abs(after[bar] - before[bar] - grows) <= 1, ('the late padding reached the bar', facts)
    for state in (before, after):
        assert abs(state['headPad'] - state['head']) <= 0.5 and abs(state['footPad'] - state['foot']) <= 0.5, facts


# Focus without a keyboard walk, so WebKit (which Tabs only to fields by default) measures it too: a bar's own control
# focused while the column sits mid-scroll, and every row brought into view by focus() from the top of the scroll.
FOCUS_MOVES = """async () => {
    const side = document.getElementById('primary-sidebar');
    const frames = () => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
    const head = side.querySelector('.sidebar-head'), meta = side.querySelector('.sidebar-foot-meta');
    const version = document.getElementById('nav-version');
    version.tabIndex = -1;  // the foot's own text stands in for its update capsule, hidden in this fixture
    const bars = {};
    for (const [name, el] of [['main', side.querySelector('.sidebar-head [data-nav-page="chat"]')], ['foot', version]]) {
        document.activeElement?.blur?.();
        side.scrollTop = Math.round((side.scrollHeight - side.clientHeight) / 2);
        await frames();
        const before = side.scrollTop;
        el.focus();
        await frames();
        bars[name] = {before, after: side.scrollTop};
    }
    // Every row parked half under the foot, then half under the head, and brought into view from there: only the
    // row's own scroll margin can move it clear (a row inside the scrollport counts as visible without it).
    const hidden = [], parked = {foot: 0, head: 0};
    const rows = [...side.querySelectorAll('#nav-projects-list [data-project-id], .sidebar-foot > [data-nav-page]')];
    const max = side.scrollHeight - side.clientHeight;
    for (const row of rows) {
        for (const bar of ['foot', 'head']) {
            document.activeElement?.blur?.();
            side.scrollTop = 0;
            await frames();
            const r0 = row.getBoundingClientRect(), y = r0.top - side.getBoundingClientRect().top;
            const middle = y + r0.height / 2;
            const at = bar === 'foot' ? middle - side.clientHeight + meta.getBoundingClientRect().height
                : middle - head.getBoundingClientRect().height;
            if (at < 0 || at > max) continue;
            side.scrollTop = at;
            await frames();
            const r = row.getBoundingClientRect(), h = head.getBoundingClientRect(), m = meta.getBoundingClientRect();
            if (bar === 'foot' ? r.bottom <= m.top + 0.5 : r.top >= h.bottom - 0.5) continue;
            parked[bar] += 1;
            row.focus();
            await frames();
            const s = row.getBoundingClientRect(), h2 = head.getBoundingClientRect(), m2 = meta.getBoundingClientRect();
            if (s.bottom > m2.top + 0.5 || s.top < h2.bottom - 0.5) hidden.push(bar + ': ' + row.textContent.trim().slice(0, 40));
        }
    }
    return {bars, rows: rows.length, parked, hidden};
}"""


@pytest.mark.parametrize(('width', 'height', 'touch'), [(1100, 500, False), (390, 570, True)],
                         ids=['desk', 'phone-drawer'])
def test_a_pinned_bars_own_control_leaves_the_column_where_it_is(shell_ui, tmp_path, width, height, touch):
    """The rows carry the bars' heights as their scroll margin, not the column as its scroll padding: a row brought
    into view from half under either bar clears both, the tools included, and focusing Main or the foot does not
    count them as covered and scroll the column."""
    page = seed_page(shell_ui, width, height, touch=touch)
    facts = {'engine': shell_ui['engine'], 'size': [width, height]}
    try:
        boot(page, shell_ui)
        if width < 720:
            open_drawer(page)
        facts['short'] = page.evaluate(SHORT)
        facts.update(page.evaluate(FOCUS_MOVES))
    finally:
        evidence(tmp_path, f'sidebar-short-bar-focus-{width}x{height}-{shell_ui["engine"]}', facts)
        page.close()
    assert facts['short'] is True and facts['rows'] > 30, facts
    for name, move in facts['bars'].items():
        assert move['before'] > 0 and abs(move['after'] - move['before']) <= 1, ('the column stayed put', name, facts)
    assert facts['parked']['foot'] > 20 and facts['parked']['head'] > 20, ('rows were parked under both bars', facts)
    assert facts['hidden'] == [], ('a row brought into view under a pinned bar', facts)


def test_the_bars_measured_heights_follow_a_tall_window_that_turns_short(shell_ui, tmp_path):
    """A phone's drawer made short after the first paint (a browser bar or a keyboard takes height): finger rows keep
    their height, so the only change is the foot taking the column's bottom padding as its own, and the measured
    heights follow it."""
    page = seed_page(shell_ui, 390, 700, touch=True)
    facts = {'engine': shell_ui['engine']}
    try:
        boot(page, shell_ui)
        open_drawer(page)
        facts['tall_short'] = page.evaluate(SHORT)
        facts['tall'] = page.evaluate(BARS)
        page.set_viewport_size({'width': 390, 'height': 570})
        settle(page)
        page.wait_for_timeout(150)
        facts['short'] = page.evaluate(SHORT)
        facts['after'] = page.evaluate(BARS)
    finally:
        evidence(tmp_path, f'sidebar-short-flip-{shell_ui["engine"]}', facts)
        page.close()
    tall, after = facts['tall'], facts['after']
    assert facts['tall_short'] is False and facts['short'] is True, facts
    assert abs(after['head'] - tall['head']) <= 0.5 and after['foot'] > tall['foot'], ('only the foot changed', facts)
    for state in (tall, after):
        assert abs(state['headPad'] - state['head']) <= 0.5 and abs(state['footPad'] - state['foot']) <= 0.5, facts
