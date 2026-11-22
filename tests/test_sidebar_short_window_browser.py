"""The sidebar's foot: its two cards at every width, its version row, and a short window (owner 23.10, 21.11;
docs/DESIGN.md "Shell").

A window too short for the sidebar's zones lets the whole column scroll with Main pinned. The foot, the version
and the update, used to be that flow's last row, below the fold until the column was scrolled to its end; a chosen
Interface size reaches that mode from about 130 % in a 680 px desktop window and from about 120 % on an iPhone. The
owner chose to keep the foot on the bottom edge there too (answer A), and the book keeps the six sections there with
it (DESIGN "Navigation"): the pinned foot is the sections, the version and the update. A column too short to show a
Project between Main and that foot (a finger's two cards are about 300 px: a phone held sideways, a 500 px drawer) lets
both go with the list instead, Find and the rows in view at rest and the cards at its end. The foot is two cards (owner
21.11): their names stay whole on one line in Russian and English from the narrowest sidebar to the widest, both turn to
one line each at 475 px where the four Russian sections fit, the version row keeps one height whatever the check or the
update says, and a 1280 × 540 window keeps the zones. Uses the static-server/API fixture (test_subscription_setup_browser)
with the gate's 30 Projects, update status and check answered by route mocks; never starts the runtime or reads live
data.
"""
from __future__ import annotations

import pytest

from tests import test_subscription_setup_browser as setup_browser
from tests.test_design_system_gate_browser import seed_page, settle, sidebar_metrics
from tests.test_shell_interactions_browser import boot, chromium_ui, evidence, open_drawer, shell_ui

subscription_ui = setup_browser.subscription_ui
pytestmark = [pytest.mark.ui_browser]
__all__ = ['chromium_ui', 'shell_ui', 'subscription_ui']

SHORT = "matchMedia('(max-height: 499px), (pointer: coarse) and (max-height: 640px)').matches"
# Where the foot is: its box against the column's bottom edge (0 = on the edge), the version's distance from that edge,
# whether the foot is what a finger finds at its middle (the rows pass under it), and whether Settings is in view.
FOOT = """() => {
    const side = document.getElementById('primary-sidebar');
    const meta = side.querySelector('.sidebar-foot');
    const box = (e) => e.getBoundingClientRect();
    const edge = box(side).bottom, m = box(meta);
    const under = document.elementFromPoint(m.left + 4, m.top + m.height / 2);
    return {scrolls: side.scrollHeight > side.clientHeight + 1, top: Math.round(side.scrollTop),
            position: getComputedStyle(meta).position, metaBottom: Math.round((m.bottom - edge) * 10) / 10,
            version: Math.round((edge - box(document.getElementById('nav-version')).bottom) * 10) / 10,
            onTop: Boolean(under && meta.contains(under)),
            opaque: getComputedStyle(meta).backgroundColor === getComputedStyle(side).backgroundColor
                && getComputedStyle(meta).backgroundColor !== 'rgba(0, 0, 0, 0)',
            settingsInView: box(side.querySelector('[data-nav-page="settings"]')).bottom <= edge + 0.5
                && box(side.querySelector('[data-nav-page="settings"]')).top >= box(side).top - 0.5};
}"""
SCROLL = "f => { const s = document.getElementById('primary-sidebar'); s.scrollTop = f * (s.scrollHeight - s.clientHeight); }"


CROWD = "document.getElementById('primary-sidebar').hasAttribute('data-bars-crowd')"


# Short windows where Main, a Project row and the foot fit together: both bars pinned (a too short column: below).
@pytest.mark.parametrize(('width', 'height', 'touch', 'tall'), [(1100, 480, False, 760), (390, 570, True, 844)],
                         ids=['desk', 'phone-drawer'])
def test_a_short_window_keeps_the_version_and_the_update_on_the_bottom_edge(shell_ui, tmp_path, width, height, touch,
                                                                            tall):
    page = seed_page(shell_ui, width, height, touch=touch)
    facts = {'engine': shell_ui['engine'], 'size': [width, height]}
    try:
        boot(page, shell_ui)
        if width < 720:
            open_drawer(page)
        facts['short'] = page.evaluate(SHORT)
        facts['crowd'] = page.evaluate(CROWD)
        facts['scrollers'] = [row['node'] for row in sidebar_metrics(page)]
        for at in (0, 0.5, 1):
            page.evaluate(SCROLL, at)
            settle(page)
            facts[f'at-{at}'] = page.evaluate(FOOT)
            if at == 0.5:
                setup_browser.capture(page, f'sidebar-short-{width}x{height}-mid-{shell_ui["engine"]}')
        # The same page in a tall window: the zones fit, the foot is the tools' box's last row again.
        page.set_viewport_size({'width': width, 'height': tall})
        settle(page)
        facts['tall_short'] = page.evaluate(SHORT)
        facts['tall'] = page.evaluate(FOOT)
    finally:
        evidence(tmp_path, f'sidebar-short-foot-{width}x{height}-{shell_ui["engine"]}', facts)
        page.close()
    tall = facts['tall']
    assert facts['short'] is True and facts['crowd'] is False and facts['scrollers'] == ['primary-sidebar'], facts
    assert facts['at-0']['scrolls'] and facts['at-0']['top'] == 0 < facts['at-0.5']['top'] < facts['at-1']['top'], facts
    for state in ('at-0', 'at-0.5', 'at-1'):
        foot = facts[state]
        assert foot['position'] == 'sticky' and abs(foot['metaBottom']) <= 1, ('the foot is on the edge', state, facts)
        assert abs(foot['version'] - tall['version']) <= 1, ('its text sits where a tall window puts it', state, facts)
        assert foot['onTop'] and foot['opaque'], ('the rows pass under an opaque foot', state, facts)
        assert foot['settingsInView'], ('the sections stay on the edge however far the list scrolls', state, facts)
    # Outside the short-window mode the foot is not pinned: it rests in the tools' box, on the column's padding.
    assert facts['tall_short'] is False and tall['position'] == 'static' and not tall['scrolls'], facts
    assert tall['settingsInView'] and tall['version'] >= 12, facts


# A column at rest: what a finger finds at Find's middle, and the Project rows standing whole between Find and the
# column's bottom edge, each the target at its middle.
AT_REST = """() => {
    const side = document.getElementById('primary-sidebar'), s = side.getBoundingClientRect();
    const hits = (el) => { const r = el.getBoundingClientRect();
        const at = document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2); return Boolean(at && el.contains(at)); };
    const find = side.querySelector('.nav-projects-find'), f = find.getBoundingClientRect();
    const rows = [...side.querySelectorAll('#nav-projects-list .nav-project-row')].filter((row) => {
        const r = row.getBoundingClientRect(); return r.top >= f.bottom && r.bottom <= s.bottom + 0.5 && hits(row); });
    return {top: side.scrollTop, find: hits(find) && f.top >= s.top - 0.5 && f.bottom <= s.bottom + 0.5, rows: rows.length,
            head: getComputedStyle(side.querySelector('.sidebar-head')).position,
            foot: getComputedStyle(side.querySelector('.sidebar-foot')).position};
}"""
# The foot's controls at the column's end: each the target a finger finds at its middle.
REACH = """() => [...document.querySelectorAll('#primary-sidebar .sidebar-foot :is([data-nav-page], #nav-version-check, #nav-version)')]
    .filter((el) => el.getClientRects().length).map((el) => { const r = el.getBoundingClientRect();
        const at = document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2);
        return [el.dataset.navPage || el.id, Boolean(at && el.contains(at))]; })"""
# The end and the position in ONE look: a wheel's own scroll is animated (WebKit), so a turn already under way can
# finish between two reads — the wheel that follows would then find nothing left to move and never report a change.
AT_EDGE = """() => { const s = document.getElementById('primary-sidebar');
    return [s.scrollHeight - s.clientHeight - s.scrollTop <= 1, s.scrollTop]; }"""
# After a wheel: the column moved, or the wheel found it already standing at its end (nothing left to move).
MOVED_OR_END = """t => { const s = document.getElementById('primary-sidebar');
    return s.scrollTop !== t || s.scrollHeight - s.clientHeight - s.scrollTop <= 1; }"""
# A turn that moved nothing: the wheel's own point and the column's box, so a column that does not scroll and a wheel
# that never reached it are told apart.
STALLED = """(t) => { const s = document.getElementById('primary-sidebar'), r = s.getBoundingClientRect();
    const at = document.elementFromPoint(r.left + Math.min(r.width / 2, 120), r.top + r.height / 2);
    return {turn: t, top: s.scrollTop, remaining: s.scrollHeight - s.clientHeight - s.scrollTop,
            scrollHeight: s.scrollHeight, clientHeight: s.clientHeight, overflowY: getComputedStyle(s).overflowY,
            hit: at ? (at.id || at.className || at.tagName) : null, hitInSidebar: Boolean(at && s.contains(at)),
            rows: s.querySelectorAll('#nav-projects-list .nav-project-row').length}; }"""


def wheel_to_end(page):
    """Real wheel turns over the column, no scrollTop written, until it stands at its end; the turns it took."""
    box = page.locator('#primary-sidebar').bounding_box()
    page.mouse.move(box['x'] + min(box['width'] / 2, 120), box['y'] + box['height'] / 2)
    for turn in range(80):
        at_end, before = page.evaluate(AT_EDGE)
        if at_end:
            return turn
        page.mouse.wheel(0, 500)
        for _ in range(40):  # a real wheel moves the column within a frame or two
            if page.evaluate(MOVED_OR_END, before):
                break
            page.wait_for_timeout(50)
        else:
            # The column stood still away from its end: it does not scroll here, and the wheel's own target says why.
            raise AssertionError(f'the wheel moved nothing over the column: {page.evaluate(STALLED, turn)}')
    return None


@pytest.mark.parametrize(('width', 'height', 'touch', 'tall'),
                         [(844, 390, True, 760), (390, 500, True, 844), (1100, 380, False, 760)],
                         ids=['landscape-phone', 'short-drawer', 'desk'])
def test_a_column_too_short_for_both_bars_scrolls_them_with_the_list(shell_ui, tmp_path, width, height, touch, tall):
    """Pinned, a finger's two cards (about 300 px) covered every Project of a 500 px drawer and Find in a phone held
    sideways. Where Main, one Project and the foot no longer fit, the head and the foot go with the list: at rest Find
    and the rows are there to use, a real wheel brings the cards to the bottom edge at the column's end, and every one
    of their controls answers there, Settings opening its page."""
    page = seed_page(shell_ui, width, tall, touch=touch)
    facts = {'engine': shell_ui['engine'], 'size': [width, height]}
    try:
        boot(page, shell_ui)
        if width < 720:
            open_drawer(page)
        facts['tall'] = page.evaluate(FOOT)  # where a tall window puts the version
        page.set_viewport_size({'width': width, 'height': height})
        settle(page)
        facts['short'] = page.evaluate(SHORT)
        facts['scrollers'] = [row['node'] for row in sidebar_metrics(page)]
        facts['crowd'] = page.evaluate(CROWD)
        facts['rest'] = page.evaluate(AT_REST)
        setup_browser.capture(page, f'sidebar-crowd-{width}x{height}-rest-{shell_ui["engine"]}')
        facts['turns'] = wheel_to_end(page)
        settle(page)
        facts['end_crowd'] = page.evaluate(CROWD)
        facts['end'] = page.evaluate(FOOT)
        setup_browser.capture(page, f'sidebar-crowd-{width}x{height}-end-{shell_ui["engine"]}')
        facts['reach'] = page.evaluate(REACH)
        facts['bars'] = page.evaluate(BARS)
        page.locator('#primary-sidebar .sidebar-foot [data-nav-page="settings"]').click()
        page.locator('#page-settings').wait_for(state='visible')
        facts['settings_opened'] = True
    finally:
        evidence(tmp_path, f'sidebar-short-crowd-{width}x{height}-{shell_ui["engine"]}', facts)
        page.close()
    rest, end, bars = facts['rest'], facts['end'], facts['bars']
    assert facts['short'] is True and facts['crowd'] is True and facts['scrollers'] == ['primary-sidebar'], facts
    assert rest['top'] == 0 and rest['head'] == 'static' and rest['foot'] == 'static', ('both bars go with the list', facts)
    assert rest['find'] and rest['rows'] >= 1, ('Find and a whole Project to use at rest', facts)
    assert facts['turns'] and facts['end_crowd'] is True, ('a real wheel reached the end, the column holding still', facts)
    assert abs(end['metaBottom']) <= 1 and abs(end['version'] - facts['tall']['version']) <= 1, ('the cards on the edge', facts)
    names = [name for name, _ in facts['reach']]
    assert {'circles', 'files', 'skills', 'widgets', 'dashboard', 'settings'} <= set(names), facts
    assert all(hit for _, hit in facts['reach']), ('every control of the foot answers at the end', facts)
    assert abs(bars['headPad'] - bars['head']) <= 0.5 and abs(bars['footPad'] - bars['foot']) <= 0.5, facts
    assert facts['settings_opened'], facts


# The crowd's own inputs (layout.js measurePinnedBars): the bars' boxes, one Project row and the clearance at both edges.
NEED = """() => {
    const side = document.getElementById('primary-sidebar');
    const h = (sel) => side.querySelector(sel).getBoundingClientRect().height;
    const clearance = parseFloat(getComputedStyle(side).getPropertyValue('--space-4')) || 0;
    return h('.sidebar-head') + h('#nav-projects-list .nav-project-row') + h('.sidebar-foot') + 2 * clearance;
}"""
# The attribute over a dozen frames, with the foot's position in the last.
HOLDS = """async () => {
    const side = document.getElementById('primary-sidebar'), seen = [];
    for (let i = 0; i < 12; i += 1) {
        await new Promise((r) => requestAnimationFrame(r));
        seen.push(side.hasAttribute('data-bars-crowd'));
    }
    return {seen, room: side.clientHeight, foot: getComputedStyle(side.querySelector('.sidebar-foot')).position};
}"""


def test_the_crowd_holds_still_at_its_edge_and_when_a_safe_area_arrives(shell_ui, tmp_path):
    """The column turns crowded where the bars' measured heights and one row pass its height, and the bars keep their
    boxes when it does, so the measure that set it reads the same: at a pixel either side of the edge, and when the
    safe area arrives after the first paint and grows the foot past it, the attribute holds still frame after frame."""
    page = seed_page(shell_ui, 390, 600, touch=True)
    facts = {'engine': shell_ui['engine'], 'edge': {}}
    try:
        boot(page, shell_ui)
        open_drawer(page)
        facts['need'] = need = page.evaluate(NEED)
        for height in sorted({int(need) - 1, int(need), int(need) + 1, int(need) + 2}):
            page.set_viewport_size({'width': 390, 'height': height})
            settle(page)
            facts['edge'][height] = page.evaluate(HOLDS)
        page.set_viewport_size({'width': 390, 'height': int(need) + 20})
        settle(page)
        facts['before_safe_area'] = page.evaluate(HOLDS)
        page.evaluate(LATE['safe-area'])
        settle(page)
        facts['after_safe_area'] = page.evaluate(HOLDS)
        facts['need_after'] = page.evaluate(NEED)
    finally:
        evidence(tmp_path, f'sidebar-short-crowd-edge-{shell_ui["engine"]}', facts)
        page.close()
    for height, state in facts['edge'].items():
        crowd = facts['need'] > state['room']
        assert state['seen'] == [crowd] * 12, ('the attribute holds still', height, facts)
        assert state['foot'] == ('static' if crowd else 'sticky'), (height, facts)
    assert facts['before_safe_area']['seen'] == [False] * 12, facts
    assert facts['need_after'] - facts['need'] >= 33 and facts['after_safe_area']['seen'] == [True] * 12, facts
    assert facts['after_safe_area']['foot'] == 'static', facts


# The focused element against the two pinned bars: under the head, under the foot, or clear of both.
FOCUS = """() => {
    const side = document.getElementById('primary-sidebar');
    const head = side.querySelector('.sidebar-head'), meta = side.querySelector('.sidebar-foot');
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
    page = seed_page(chromium_ui, 1100, 480, touch=False)
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
    assert up[-1]['name'].endswith('Main Chat') and up[-1]['scrollTop'] == up[-2]['scrollTop'] < down[-1]['scrollTop'], facts
    hidden = [(way, stop['name']) for way, stops in (('down', down), ('up', up)) for stop in stops
              if stop['underFoot'] or stop['underHead']]
    assert not hidden, ('a focused row under a pinned bar', hidden, facts)


# The two pinned bars: the heights the rows' scroll margin holds against their real border boxes.
BARS = """() => {
    const side = document.getElementById('primary-sidebar');
    const css = getComputedStyle(side);
    return {head: side.querySelector('.sidebar-head').getBoundingClientRect().height,
            foot: side.querySelector('.sidebar-foot').getBoundingClientRect().height,
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
                         [(1100, 480, False, 'title-band', 'head', 52), (390, 570, True, 'safe-area', 'foot', 34)],
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
    const head = side.querySelector('.sidebar-head'), meta = side.querySelector('.sidebar-foot');
    const version = document.getElementById('nav-version-check');  // the version is the foot's check (no capsule here)
    const bars = {}, missed = [];
    const controls = [['main', side.querySelector('.sidebar-head [data-nav-page="chat"]')], ['foot', version],
        ...[...side.querySelectorAll('.nav-card [data-nav-page]')].map(el => [el.dataset.navPage, el])];
    for (const [name, el] of controls) {
        document.activeElement?.blur?.();
        side.scrollTop = Math.round((side.scrollHeight - side.clientHeight) / 2);
        await frames();
        const before = side.scrollTop;
        el.focus();
        // A control that cannot take focus (hidden, or not yet mounted) would leave the column still for nothing.
        if (document.activeElement !== el) missed.push(name);
        await frames();
        bars[name] = {before, after: side.scrollTop};
    }
    // Every row parked half under the foot, then half under the head, and brought into view from there: only the
    // row's own scroll margin can move it clear (a row inside the scrollport counts as visible without it).
    const hidden = [], parked = {foot: 0, head: 0};
    const rows = [...side.querySelectorAll('#nav-projects-list [data-project-id]')];
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
    return {bars, missed, rows: rows.length, parked, hidden};
}"""


@pytest.mark.parametrize(('width', 'height', 'touch'), [(1100, 480, False), (390, 570, True)],
                         ids=['desk', 'phone-drawer'])
def test_a_pinned_bars_own_control_leaves_the_column_where_it_is(shell_ui, tmp_path, width, height, touch):
    """The rows carry the bars' heights as their scroll margin, not the column as its scroll padding: a row brought
    into view from half under either bar clears both, and focusing Main or the foot (the sections' bar) does not
    count them as covered and scroll the column."""
    # A managed install whose status offers the check: the version is a focusable control, not plain text.
    page, _ = foot_page(shell_ui, width, height, lang='en', touch=touch)
    facts = {'engine': shell_ui['engine'], 'size': [width, height]}
    try:
        if width < 720:
            open_drawer(page)
        facts['short'] = page.evaluate(SHORT)
        facts['check'] = page.evaluate("() => { const c = document.getElementById('nav-version-check');"
                                       " return Boolean(c && !c.hidden && c.contains(document.getElementById('nav-version'))); }")
        facts.update(page.evaluate(FOCUS_MOVES))
    finally:
        evidence(tmp_path, f'sidebar-short-bar-focus-{width}x{height}-{shell_ui["engine"]}', facts)
        page.close()
    assert facts['short'] is True and facts['rows'] >= 30, facts
    assert facts['check'] and facts['missed'] == [], ('every bar control took focus', facts)
    assert set(facts['bars']) == {'main', 'foot', 'circles', 'files', 'skills', 'widgets', 'dashboard', 'settings'}, facts
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


# The foot's two cards measured where they stand: every row's label on one line and unclipped, the edges the rows and
# the version share, and the four sections' need on one line (each row as wide as its words, 8 px between them).
RU = {'Circles': 'Круги', 'Files': 'Файлы', 'Extensions': 'Расширения', 'Widgets': 'Виджеты', 'Dashboard': 'Панель управления',
      'Settings': 'Настройки', 'Update': 'Обновить', 'Up to date': 'Обновлений нет', 'Check failed': 'Не удалось проверить',
      'Updating…': 'Обновляю…'}
REST = {'managed': True, 'available': False, 'checked_at': '2026-11-21T10:00:00Z', 'local_changes': 'absent'}
# The rows' names in the registry (pages.js), in the cards' order.
NAMES = ('Circles', 'Files', 'Extensions', 'Widgets', 'Dashboard', 'Settings')


def shown(lang, labels=None):
    """The words the rows must show while they are measured: the mocked translation and any other name put on a row."""
    names = {**(RU if lang == 'ru' else {}), **(labels or {})}
    return [names.get(name, name) for name in NAMES]


CARDS = """() => {
    const side = document.getElementById('primary-sidebar');
    const box = (e) => e.getBoundingClientRect();
    const rows = (sel) => [...side.querySelectorAll(sel + ' .nav-row')].map((row) => {
        const label = row.querySelector('.nav-row-label'), size = parseFloat(getComputedStyle(label).fontSize);
        return {id: row.dataset.navPage, label: label.textContent, top: box(row).top, left: box(row).left,
                right: box(row).right, height: box(row).height, width: box(row).width, glyph: box(row.querySelector('svg')).left,
                oneLine: box(label).height < 2 * size && label.scrollWidth <= label.clientWidth + 0.5};
    });
    const card = side.querySelector('.nav-sections'), pad = parseFloat(getComputedStyle(card).paddingLeft);
    const version = document.getElementById('nav-version');
    return {width: box(side).width, sections: rows('.nav-sections'), system: rows('.nav-system-rows'),
            inner: [box(card).left + pad, box(card).right - pad], versionLeft: box(version).left,
            gap: parseFloat(getComputedStyle(side.querySelector('.nav-sections')).columnGap) || 0};
}"""


def foot_page(ui, width, height, *, lang='ru', touch=False, status=REST, check=None, labels=None):
    """The shell with a mocked update status and check (`check` answers each press: a status, or 500 when 'fail'; None
    holds the press, so the row stays checking). `labels` puts other names on rows (a longer translation)."""
    page = seed_page(ui, width, height, touch=touch)
    if lang == 'ru' or labels:
        names = {**(RU if lang == 'ru' else {}), **(labels or {})}
        entries = {key: {'text': value} for key, value in names.items()}
        page.route('**/api/ui/i18n*', lambda route: route.fulfill(json={'language': lang, 'english': False, 'entries': entries}))
    page.route('**/api/health*', lambda route: route.fulfill(json={'ok': True, 'version': '7.11.1-alpha.0',
                                                                  'runtime_version': '7.11.1-alpha.0'}))
    world = {'status': status, 'check': check, 'held': [], 'checks': 0}
    page.route('**/api/update/status*', lambda route: route.fulfill(json=world['status']))

    def answer(route):
        world['checks'] += 1
        if world['check'] is None:
            world['held'].append(route)
        elif world['check'] == 'fail':
            route.fulfill(status=500, json={'error': 'Network is unreachable'})
        else:
            route.fulfill(json=world['check'])
    page.route('**/api/update/check*', answer)
    boot(page, ui)
    if lang == 'ru' or labels:
        extensions = (labels or {}).get('Extensions') or (RU['Extensions'] if lang == 'ru' else 'Extensions')
        page.wait_for_function("s => document.querySelector('[data-nav-page=skills] .nav-row-label').textContent === s",
                               arg=extensions)
    page.wait_for_function("document.getElementById('nav-version').textContent === 'v7.11.1-alpha.0'")
    return page, world


@pytest.mark.parametrize('lang', ['ru', 'en'])
def test_the_two_cards_keep_every_name_whole_and_turn_wide_at_475(shell_ui, tmp_path, lang):
    page, _ = foot_page(shell_ui, 1440, 900, lang=lang)
    facts = {'engine': shell_ui['engine'], 'lang': lang}
    try:
        for width in (270, 280, 474, 475, 480, 560):
            page.evaluate(f"document.documentElement.style.setProperty('--sidebar-width', '{width}px')")
            settle(page)
            facts[width] = page.evaluate(CARDS)
    finally:
        evidence(tmp_path, f'sidebar-foot-cards-{lang}-{shell_ui["engine"]}', facts)
        page.close()
    for width in (270, 280, 474, 475, 480, 560):
        at = facts[width]
        rows = at['sections'] + at['system']
        # The names measured are the ones asked for (a translation reset under the probe would measure other words).
        assert [row['label'] for row in rows] == shown(lang), ('the measured names', width, rows)
        assert round(at['width']) == width, facts
        assert all(row['oneLine'] for row in rows), ('every name whole on one line', width, rows)
        assert all(row['right'] <= at['inner'][1] + 0.5 for row in rows), ('nothing past the card', width, rows)
        first = [at['sections'][0], at['system'][0]]
        assert all(abs(row['left'] - at['inner'][0]) <= 0.5 for row in first), ('the left edges on one line', width, facts)
        assert abs(at['versionLeft'] - at['sections'][0]['glyph']) <= 1, ('the version on the rows\' glyph line', width)
        tops = {round(row['top']) for row in at['sections']}
        if width < 475:  # compact: the sections two to a line, the system rows a line each
            assert len(tops) == 2 and at['system'][0]['top'] < at['system'][1]['top'], (width, facts)
            assert abs(at['sections'][2]['left'] - at['inner'][0]) <= 0.5, ('the second line starts at the edge', width)
            half = (at['inner'][1] - at['inner'][0]) / 2
            if max(at['sections'][0]['width'], at['sections'][2]['width']) <= half + 0.5:
                assert abs(at['sections'][1]['left'] - at['sections'][3]['left']) <= 0.5, ('the columns hold', width, facts)
        else:  # wide: each card one line, the last rows' right edges on the card's edge
            assert len(tops) == 1 and abs(at['system'][0]['top'] - at['system'][1]['top']) < 1, (width, facts)
            for row in (at['sections'][-1], at['system'][-1]):
                assert abs(row['right'] - at['inner'][1]) <= 0.5, ('the right edges on one line', width, facts)
            gaps = [b['left'] - a['right'] for a, b in zip(at['sections'], at['sections'][1:])]
            assert max(gaps) - min(gaps) <= 1.5 and min(gaps) >= at['gap'] - 0.5, ('the room shared evenly', width, gaps)
    if lang == 'ru' and shell_ui['engine'] == 'chromium':
        # The switch is where the four sections fit one line with the longest name so far, «Расширения», rows and their
        # 8 px gaps: four fit at the switch itself, not above it. The bound is not tied to one font's widths (they
        # differ between systems); the strict unfolding above (two lines at 474, one at 475) and the rows keeping
        # their widths while the gaps take the room carry the rest.
        wide = facts[475]
        need = sum(row['width'] for row in wide['sections']) + 3 * wide['gap']
        fits_from = 475 - (wide['inner'][1] - wide['inner'][0] - need)
        assert fits_from <= 475, ('the switch is where four fit', fits_from, facts)
        later = min(b['left'] - a['right'] for a, b in zip(facts[480]['sections'], facts[480]['sections'][1:]))
        assert abs(later - (wide['gap'] + (480 - fits_from) / 3)) <= 1.5, ('the gaps take the room', later, fits_from)


ROW_H = "() => document.querySelector('.sidebar-foot-meta').getBoundingClientRect().height"
STATE = """() => {
    const check = document.getElementById('nav-version-check'), note = document.querySelector('.nav-check-note');
    const pill = document.getElementById('update-pill');
    return {busy: check.getAttribute('aria-busy'), check: !check.hidden, note: note.hidden ? '' : note.textContent,
            pill: pill && !pill.hidden ? pill.textContent : '', version: document.getElementById('nav-version').textContent};
}"""


def test_the_version_row_keeps_one_height_whatever_the_check_or_the_update_says(shell_ui, tmp_path):
    """The version is the check (owner 21.11): rest, hover, checking, Up to date, a failure, the Update capsule and an
    attempt that fits the line all stand in one row height, and one press is one request."""
    page, world = foot_page(shell_ui, 1440, 900)
    page.evaluate("document.documentElement.style.setProperty('--sidebar-width', '280px')")
    facts = {'engine': shell_ui['engine'], 'heights': {}, 'states': {}}

    def note(name):
        settle(page)
        facts['heights'][name] = page.evaluate(ROW_H)
        facts['states'][name] = page.evaluate(STATE)

    try:
        note('rest')
        page.hover('#nav-version-check')
        note('hover')
        page.locator('#nav-version-check').click()
        page.locator('#nav-version-check').click()
        page.wait_for_function("document.getElementById('nav-version-check').getAttribute('aria-busy') === 'true'")
        note('checking')
        facts['checks'] = world['checks']
        # Each completed check answers with a fresh checked_at (the update contract); an equal stamp
        # is the same check and keeps the held result, so the update found later needs its own newer stamp.
        world['held'].pop().fulfill(json=dict(REST, check_ok=True, checked_at='2026-11-21T10:05:00Z'))
        page.wait_for_function("!document.querySelector('.nav-check-note').hidden")
        note('up to date')
        world['check'] = 'fail'
        page.locator('#nav-version-check').click()
        page.wait_for_function("document.querySelector('.nav-check-note').dataset.tone === 'error'")
        note('failed')
        world['check'] = dict(REST, available=True, check_ok=True, checked_at='2026-11-21T10:10:00Z',
                              current_version='7.11.1-alpha.0', latest_version='7.12.0')
        page.locator('#nav-version-check').click()
        page.wait_for_function("document.getElementById('update-pill')?.hidden === false")
        note('update')
        # The passive read while the update runs is that same check (an equal stamp), the attempt riding on it.
        world['status'] = dict(world['check'], update_progress={'operation_id': 'a', 'generation': 'g', 'active': True})
        page.evaluate("window.__ouroWs.emit('update_progress_changed', {})")
        page.wait_for_function("document.getElementById('update-pill').dataset.tone === 'neutral'")
        note('attempt')
    finally:
        evidence(tmp_path, f'sidebar-foot-version-row-{shell_ui["engine"]}', facts)
        page.close()
    states, heights = facts['states'], facts['heights']
    assert facts['checks'] == 1, ('a press while it checks starts nothing', facts)
    assert states['checking']['busy'] == 'true' and states['rest']['busy'] is None, facts
    assert states['up to date']['note'] == 'Обновлений нет' and states['failed']['note'] == 'Не удалось проверить', facts
    assert states['update']['pill'] == 'Обновить' and not states['update']['check'], ('one action per row', facts)
    assert states['attempt']['pill'] == 'Обновляю…' and not states['attempt']['check'], facts
    assert {state['version'] for state in states.values()} == {'v7.11.1-alpha.0'}, facts
    assert max(heights.values()) - min(heights.values()) <= 0.5, ('one height in every state', facts)


# The docked zones in a window: Main's row and both cards inside the column, the list's own box and offsets.
ZONES = """() => {
    const side = document.getElementById('primary-sidebar'), list = document.getElementById('nav-projects-list');
    const s = side.getBoundingClientRect(), main = side.querySelector('.sidebar-head [data-nav-page="chat"]').getBoundingClientRect();
    return {main: main.top >= s.top - 0.5 && main.bottom <= s.bottom + 0.5,
            cards: [...side.querySelectorAll('.nav-card')].map((c) => { const r = c.getBoundingClientRect();
                return {top: r.top - s.top, bottom: s.bottom - r.bottom}; }),
            list: {height: list.clientHeight, overflow: list.scrollHeight - list.clientHeight, top: list.scrollTop},
            sideTop: side.scrollTop};
}"""


def test_a_1280_by_540_window_keeps_both_cards_whole_and_the_list_scrolls_alone(shell_ui, tmp_path):
    """Desktop 540 is the docked layout, not the short fallback: Main and both cards stay in the column while the
    wheel over the Projects moves the list alone, which has a real height of its own."""
    page = seed_page(shell_ui, 1280, 540, touch=False)
    facts = {'engine': shell_ui['engine']}
    try:
        boot(page, shell_ui)
        facts['short'] = page.evaluate(SHORT)
        facts['scrollers'] = [row['node'] for row in sidebar_metrics(page)]
        facts['before'] = page.evaluate(ZONES)
        box = page.locator('#nav-projects-list').bounding_box()
        page.mouse.move(box['x'] + box['width'] / 2, box['y'] + box['height'] / 2)
        page.mouse.wheel(0, 300)
        page.wait_for_function("document.getElementById('nav-projects-list').scrollTop > 0")
        settle(page)
        facts['after'] = page.evaluate(ZONES)
    finally:
        evidence(tmp_path, f'sidebar-foot-1280x540-{shell_ui["engine"]}', facts)
        page.close()
    assert facts['short'] is False, facts
    assert facts['scrollers'] == ['nav-projects-list'], ('the column itself does not scroll; the list does', facts)
    for state in (facts['before'], facts['after']):
        assert state['main'] and state['sideTop'] == 0, ('Main stays in the docked head', facts)
        assert all(card['top'] >= 0 and card['bottom'] >= 0 for card in state['cards']), ('both cards whole', facts)
    assert facts['before']['list']['height'] >= 80 and facts['before']['list']['overflow'] > 0, ('a list of its own', facts)
    assert facts['after']['list']['top'] > 0, ('the wheel moved the list', facts)


# Names longer than any translation so far: a cell never grows past its card, a name wraps at a space inside it.
# Files ends a line and Extensions starts one, so both kinds of cell meet a name past the card.
LONG = {'Files': 'Файлы, документы и вложения проекта', 'Extensions': 'Управление расширениями и навыками',
        'Dashboard': 'Панель управления и состояния системы'}


@pytest.mark.parametrize('width', [270, 480])
def test_a_name_longer_than_the_card_wraps_inside_it(shell_ui, tmp_path, width):
    page, _ = foot_page(shell_ui, 1440, 900, lang='ru', labels=LONG)
    facts = {'engine': shell_ui['engine']}
    try:
        page.evaluate(f"document.documentElement.style.setProperty('--sidebar-width', '{width}px')")
        settle(page)
        facts.update(page.evaluate(CARDS))
        facts['cells'] = page.evaluate("""() => [...document.querySelectorAll('.nav-sections > .nav-cell')].map((c) => {
            const r = c.getBoundingClientRect(); return {left: r.left, right: r.right}; })""")
    finally:
        evidence(tmp_path, f'sidebar-foot-long-{width}-{shell_ui["engine"]}', facts)
        page.close()
    rows = facts['sections'] + facts['system']
    assert [row['label'] for row in rows] == shown('ru', LONG), ('the measured names', rows)
    assert all(row['right'] <= facts['inner'][1] + 0.5 and row['left'] >= facts['inner'][0] - 0.5 for row in rows), facts
    assert all(cell['right'] <= facts['inner'][1] + 0.5 for cell in facts['cells']), ('no cell past the card', facts)
    long_rows = [row for row in rows if row['id'] in ('files', 'skills', 'dashboard')]
    assert width >= 480 or not any(row['oneLine'] for row in long_rows), ('the long names wrap at 270', facts)
    short = [row for row in facts['sections'] if row['id'] not in ('files', 'skills')]
    assert all(row['oneLine'] for row in short), ('the other names stay whole', facts)


def test_the_drawer_keeps_the_same_block_with_finger_targets(shell_ui, tmp_path):
    page, _ = foot_page(shell_ui, 390, 844, touch=True)
    facts = {'engine': shell_ui['engine']}
    try:
        open_drawer(page)
        facts.update(page.evaluate(CARDS))
        facts['version'] = page.evaluate("""() => { const c = document.getElementById('nav-version-check');
            const r = c.getBoundingClientRect(), a = getComputedStyle(c, '::after');
            return {height: r.height, reach: r.height - 2 * parseFloat(a.top || '0')}; }""")
    finally:
        evidence(tmp_path, f'sidebar-foot-drawer-{shell_ui["engine"]}', facts)
        page.close()
    rows = facts['sections'] + facts['system']
    assert [row['label'] for row in rows] == shown('ru'), ('the measured names', rows)
    assert all(row['oneLine'] for row in rows), rows
    assert all(row['height'] >= 44 and row['width'] >= 44 for row in rows), ('a finger\'s target each way', rows)
    assert facts['version']['reach'] >= 44 - 0.5, ('the version\'s reach is a finger\'s', facts)
