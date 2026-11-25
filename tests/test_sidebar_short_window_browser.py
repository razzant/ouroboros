"""The sidebar's rail beside the column, its version row, and a short window (owner 23.10, 21.11, 24.11;
docs/DESIGN.md "Shell").

The places stand in the rail at the column's start edge (owner 24.11, «Рельс снаружи»): the sections at its top,
Dashboard, Settings and the version with the update at its foot, each place its glyph with its word its name. A window
too short for the zones lets the whole column scroll with Main pinned; the rail stays in view beside it, the version
where a tall window puts it, and where even the rail is taller than the window it scrolls with the column, its foot at
the end. A column too short to show a Project under Main lets Main go with the list. The rail keeps its width whatever
the column's and whatever the language, the version row keeps one height whatever the check says (an update's capsule
stands over the version, which never moves), and a 1280 × 540 window keeps the zones. Uses the static-server/API fixture
(test_subscription_setup_browser) with the gate's 30 Projects, update status and check answered by route mocks; never
starts the runtime or reads live data.
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
# Where the rail is: its top against the column's, the version's distance from the column's bottom edge, whether the
# rail is what a finger finds at Settings' middle, and whether its sections and Settings are in view.
RAIL = """() => {
    const side = document.getElementById('primary-sidebar'), rail = side.querySelector('.nav-rail');
    const box = (e) => e.getBoundingClientRect();
    const s = box(side), r = box(rail);
    const inView = (sel) => { const el = side.querySelector(sel), b = box(el);
        const at = document.elementFromPoint(b.left + b.width / 2, b.top + b.height / 2);
        return b.top >= s.top - 0.5 && b.bottom <= s.bottom + 0.5 && Boolean(at && el.contains(at)); };
    return {scrolls: side.scrollHeight > side.clientHeight + 1, top: Math.round(side.scrollTop),
            position: getComputedStyle(rail).position, railTop: Math.round((r.top - s.top) * 10) / 10,
            version: Math.round((s.bottom - box(document.getElementById('nav-version')).bottom) * 10) / 10,
            sectionsInView: inView('.nav-rail [data-nav-page="circles"]') && inView('.nav-rail [data-nav-page="widgets"]'),
            settingsInView: inView('.nav-rail [data-nav-page="settings"]')};
}"""
SCROLL = "f => { const s = document.getElementById('primary-sidebar'); s.scrollTop = f * (s.scrollHeight - s.clientHeight); }"
CROWD = "document.getElementById('primary-sidebar').hasAttribute('data-bars-crowd')"
RAIL_IDS = ['circles', 'files', 'widgets', 'dashboard', 'settings']


@pytest.mark.parametrize(('width', 'height', 'touch', 'tall'), [(1100, 480, False, 760), (390, 570, True, 844)],
                         ids=['desk', 'phone-drawer'])
def test_a_short_window_keeps_the_rail_and_the_version_in_view(shell_ui, tmp_path, width, height, touch, tall):
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
            facts[f'at-{at}'] = page.evaluate(RAIL)
            if at == 0.5:
                setup_browser.capture(page, f'sidebar-short-{width}x{height}-mid-{shell_ui["engine"]}')
        # The same page in a tall window: the zones fit and the column does not scroll.
        page.set_viewport_size({'width': width, 'height': tall})
        settle(page)
        facts['tall_short'] = page.evaluate(SHORT)
        facts['tall'] = page.evaluate(RAIL)
    finally:
        evidence(tmp_path, f'sidebar-short-rail-{width}x{height}-{shell_ui["engine"]}', facts)
        page.close()
    tall = facts['tall']
    assert facts['short'] is True and facts['crowd'] is False and facts['scrollers'] == ['primary-sidebar'], facts
    assert facts['at-0']['scrolls'] and facts['at-0']['top'] == 0 < facts['at-0.5']['top'] < facts['at-1']['top'], facts
    for state in ('at-0', 'at-0.5', 'at-1'):
        rail = facts[state]
        assert rail['position'] == 'sticky' and abs(rail['railTop']) <= 0.5, ('the rail stays at the top', state, facts)
        assert abs(rail['version'] - tall['version']) <= 1, ('the version sits where a tall window puts it', state, facts)
        assert rail['sectionsInView'] and rail['settingsInView'], ('every place in view however far the list scrolls', state, facts)
    assert facts['tall_short'] is False and not tall['scrolls'] and tall['settingsInView'], facts
    assert tall['version'] >= 12, ('the version above the bottom inset', facts)


# A column at rest: what a finger finds at Find's middle, and where the head is.
AT_REST = """() => {
    const side = document.getElementById('primary-sidebar'), s = side.getBoundingClientRect();
    const hits = (el) => { const r = el.getBoundingClientRect();
        const at = document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2); return Boolean(at && el.contains(at)); };
    const find = side.querySelector('.nav-projects-find'), f = find.getBoundingClientRect();
    return {top: side.scrollTop, find: hits(find) && f.top >= s.top - 0.5 && f.bottom <= s.bottom + 0.5,
            head: getComputedStyle(side.querySelector('.sidebar-head')).position};
}"""
# The rail's controls and its version: each the target a finger finds at its middle, where it is now.
REACH = """() => [...document.querySelectorAll('#primary-sidebar .nav-rail :is([data-nav-page], #nav-version-check, #nav-version)')]
    .filter((el) => el.getClientRects().length).map((el) => { const r = el.getBoundingClientRect();
        const s = document.getElementById('primary-sidebar').getBoundingClientRect();
        const at = document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2);
        return [el.dataset.navPage || el.id, r.top >= s.top - 0.5 && r.bottom <= s.bottom + 0.5 && Boolean(at && el.contains(at))]; })"""
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


@pytest.mark.parametrize(('width', 'touch', 'tall'), [(844, True, 760), (390, True, 844), (1100, False, 760)],
                         ids=['landscape-phone', 'short-drawer', 'desk'])
def test_a_column_too_short_for_the_head_and_a_row_scrolls_both_with_the_list(shell_ui, tmp_path, width, touch, tall):
    """Where Main and one Project no longer fit the column (a few pixels under what they need: a phone held sideways
    at a large Interface size, a window dragged very short), Main goes with the list (a pinned head would cover every
    row), and the rail, taller than such a window, goes with it too: at rest its sections answer at the top, a real wheel
    brings its foot to the bottom edge at the column's end, its last controls answering there, Settings opening its
    page."""
    page = seed_page(shell_ui, width, tall, touch=touch)
    facts = {'engine': shell_ui['engine']}
    try:
        boot(page, shell_ui)
        if width < 720:
            open_drawer(page)
        facts['tall'] = page.evaluate(RAIL)  # where a tall window puts the version
        facts['need'] = page.evaluate(NEED)
        height = int(facts['need']) - 8
        facts['size'] = [width, height]
        page.set_viewport_size({'width': width, 'height': height})
        settle(page)
        facts['short'] = page.evaluate(SHORT)
        facts['scrollers'] = [row['node'] for row in sidebar_metrics(page)]
        facts['crowd'] = page.evaluate(CROWD)
        facts['rest'] = page.evaluate(AT_REST)
        facts['reach_rest'] = page.evaluate(REACH)
        setup_browser.capture(page, f'sidebar-crowd-{width}-rest-{shell_ui["engine"]}')
        facts['turns'] = wheel_to_end(page)
        settle(page)
        facts['end_crowd'] = page.evaluate(CROWD)
        facts['end'] = page.evaluate(RAIL)
        setup_browser.capture(page, f'sidebar-crowd-{width}-end-{shell_ui["engine"]}')
        facts['reach_end'] = page.evaluate(REACH)
        page.locator('#primary-sidebar .nav-rail [data-nav-page="settings"]').click()
        page.locator('#page-settings').wait_for(state='visible')
        facts['settings_opened'] = True
    finally:
        evidence(tmp_path, f'sidebar-short-crowd-{width}-{shell_ui["engine"]}', facts)
        page.close()
    rest, end = facts['rest'], facts['end']
    assert facts['short'] is True and facts['crowd'] is True and facts['scrollers'] == ['primary-sidebar'], facts
    assert rest['top'] == 0 and rest['head'] == 'static' and rest['find'], ('Main goes with the list; Find to use', facts)
    assert dict(facts['reach_rest'])['circles'], ('the sections answer at the top', facts)
    assert facts['turns'] and facts['end_crowd'] is True, ('a real wheel reached the end, the column holding still', facts)
    assert abs(end['version'] - facts['tall']['version']) <= 1, ('the rail\'s foot on the bottom edge at the end', facts)
    at_end = dict(facts['reach_end'])
    # The version: the check where the install offers one, else its plain text (this fixture reads no update status).
    assert at_end['settings'] and (at_end.get('nav-version-check') or at_end.get('nav-version')), ('its foot at the end', facts)
    assert facts['settings_opened'], facts


# The crowd's own inputs (layout.js measurePinnedBars): the head's box, one Project row and the clearance at both edges.
NEED = """() => {
    const side = document.getElementById('primary-sidebar');
    const h = (sel) => side.querySelector(sel).getBoundingClientRect().height;
    const clearance = parseFloat(getComputedStyle(side).getPropertyValue('--space-4')) || 0;
    return h('.sidebar-head') + h('#nav-projects-list .nav-project-row') + 2 * clearance;
}"""
# The attribute over a dozen frames, with the head's position in the last.
HOLDS = """async () => {
    const side = document.getElementById('primary-sidebar'), seen = [];
    for (let i = 0; i < 12; i += 1) {
        await new Promise((r) => requestAnimationFrame(r));
        seen.push(side.hasAttribute('data-bars-crowd'));
    }
    return {seen, room: side.clientHeight, head: getComputedStyle(side.querySelector('.sidebar-head')).position};
}"""
LATE = {
    # The macOS shell writes its title band once the window has measured it (launcher_window_chrome.py).
    'title-band': "() => { const r = document.documentElement; r.dataset.windowChrome = 'integrated';"
                  " r.style.setProperty('--window-chrome-top', '52px'); }",
    # Telegram reports its safe area after the app has started (host_telegram.js mirrorInsets).
    'safe-area': "() => document.documentElement.style.setProperty('--tg-safe-area-inset-top', '47px')",
}


def test_the_crowd_holds_still_at_its_edge_and_when_a_safe_area_arrives(shell_ui, tmp_path):
    """The column turns crowded where the head's measured height and one row pass its height, and the head keeps its
    box when it does, so the measure that set it reads the same: at a pixel either side of the edge, and when the safe
    area arrives after the first paint and grows the head past it, the attribute holds still frame after frame."""
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
        assert state['head'] == ('static' if crowd else 'sticky'), (height, facts)
    assert facts['before_safe_area']['seen'] == [False] * 12, facts
    assert facts['need_after'] - facts['need'] >= 21 and facts['after_safe_area']['seen'] == [True] * 12, facts
    assert facts['after_safe_area']['head'] == 'static', facts


# The focused element against the pinned head: under it, or clear of it, and where it stands — the rail, the head (Main
# among it) or a Project row — so a walk is read by what it visited, whichever of the rail and the head comes first.
FOCUS = """() => {
    const side = document.getElementById('primary-sidebar'), head = side.querySelector('.sidebar-head');
    const f = document.activeElement;
    if (!f || !side.contains(f) || f === side) return null;
    const r = f.getBoundingClientRect(), h = head.getBoundingClientRect(), s = side.getBoundingClientRect();
    // A Project row and its own controls (Pin, Actions) stand in one item; the row itself is its button.
    const item = f.closest('.nav-project-item'), row = item?.querySelector('[data-project-id]');
    return {name: (f.getAttribute('aria-label') || f.textContent || '').trim().slice(0, 40), top: r.top, bottom: r.bottom,
            zone: f.closest('.nav-rail') ? 'rail' : head.contains(f) ? 'head' : item ? 'project' : 'list',
            place: f.closest('[data-nav-page]')?.dataset.navPage || '', project: row?.dataset.projectId || '',
            isRow: Boolean(row) && f === row,
            underHead: !head.contains(f) && !f.closest('.nav-rail') && r.top < h.bottom - 0.5,
            pastEdge: r.bottom > s.bottom + 0.5 || r.top < s.top - 0.5, scrollTop: side.scrollTop};
}"""
# Where a walk starts: the sidebar's first focusable control in document order (forward), or the last of the Projects
# list's own (backward); never a hard-coded place, which the other DOM order would leave outside the walk.
EDGE = """(last) => {
    const scope = last ? document.getElementById('nav-projects-list') : document.getElementById('primary-sidebar');
    const able = [...scope.querySelectorAll('a[href], button, input, select, textarea, [tabindex]')].filter((el) =>
        el.tabIndex >= 0 && !el.disabled && el.getClientRects().length > 0 && getComputedStyle(el).visibility !== 'hidden');
    const edge = last ? able.at(-1) : able[0];
    edge.focus();
    return document.activeElement === edge;
}"""
# The DOM order this run exercised: which of the rail and the head comes first in the sidebar.
ORDER = """() => { const side = document.getElementById('primary-sidebar');
    return [...side.children].map((el) => el.classList.contains('nav-rail') ? 'rail' : el.classList.contains('sidebar-head')
        ? 'head' : el.id === 'nav-projects' ? 'projects' : '').filter(Boolean); }"""


def walk(page, key, limit=160):
    """The stop focus starts on, then press `key` until focus leaves the sidebar; every stop on the way."""
    stops = [page.evaluate(FOCUS)]
    for _ in range(limit):
        page.keyboard.press(key)
        stop = page.evaluate(FOCUS)
        if stop is None:
            break
        stops.append(stop)
    return stops


def test_a_row_reached_with_tab_is_never_left_under_the_pinned_head(chromium_ui, tmp_path):
    """Tab walks the whole column from its first control and Shift+Tab walks back from the last Project row: each walk
    visits every Project row, Main and, apart from them, the rail; the browser brings each row into view, and the
    head's height is the rows' scroll margin, so no focused row stays under the pinned head or past the column's edge.
    The walks start from the column's real edges, so they hold whichever of the rail and the head the DOM puts first.
    (WebKit on macOS does not Tab to buttons unless the system's keyboard navigation is on, so Chromium walks it.)"""
    page = seed_page(chromium_ui, 1100, 480, touch=False)
    facts = {'engine': 'chromium'}
    try:
        boot(page, chromium_ui)
        facts['short'] = page.evaluate(SHORT)
        facts['order'] = page.evaluate(ORDER)
        facts['rows'] = page.evaluate("() => [...document.querySelectorAll('#nav-projects-list [data-project-id]')]"
                                      ".map((row) => row.dataset.projectId)")
        assert page.evaluate(EDGE, False), 'the first control took focus'
        facts['down'] = walk(page, 'Tab')
        assert page.evaluate(EDGE, True), 'the last Project control took focus'
        facts['up'] = walk(page, 'Shift+Tab')
    finally:
        evidence(tmp_path, 'sidebar-short-focus-chromium', facts)
        page.close()
    down, up = facts['down'], facts['up']
    assert facts['short'] is True and len(facts['rows']) >= 30, facts
    assert up[0]['zone'] == 'project' and up[0]['project'] == facts['rows'][-1], ('back from the last Project row', facts)
    for way, stops in (('down', down), ('up', up)):
        # Every Project row is visited as a Project, Main as the head's place, and the rail's places apart from both.
        assert {s['project'] for s in stops if s['isRow']} == set(facts['rows']), ('every Project row', way, facts)
        assert any(s['zone'] == 'head' and s['place'] == 'chat' for s in stops), ('Main', way, facts)
        assert all(s['zone'] != 'rail' or not s['project'] for s in stops), ('a rail control is no Project visit', way, facts)
    assert {s['place'] for s in down if s['zone'] == 'rail'} >= set(RAIL_IDS), ('the forward walk reaches the rail', facts)
    assert max(stop['scrollTop'] for stop in down) > 0, ('the walk scrolled the column', facts)
    # Back at the head, Main leaves the column where the stop before it did: the head's own control moves nothing.
    main = next(i for i, s in enumerate(up) if s['zone'] == 'head' and s['place'] == 'chat')
    assert main > 0 and up[main]['scrollTop'] == up[main - 1]['scrollTop'], facts
    hidden = [(way, stop['name']) for way, stops in (('down', down), ('up', up)) for stop in stops
              if stop['underHead'] or stop['pastEdge']]
    assert not hidden, ('a focused row under the pinned head or past the edge', hidden, facts)


# The pinned head: the height the rows' scroll margin holds against its real border box.
BARS = """() => {
    const side = document.getElementById('primary-sidebar');
    return {head: side.querySelector('.sidebar-head').getBoundingClientRect().height,
            headPad: parseFloat(getComputedStyle(side).getPropertyValue('--sidebar-head-h')),
            railTop: parseFloat(getComputedStyle(side.querySelector('.nav-rail')).paddingTop),
            headTop: parseFloat(getComputedStyle(side.querySelector('.sidebar-head')).paddingTop)};
}"""


@pytest.mark.parametrize(('width', 'height', 'touch', 'late', 'grows'),
                         [(1100, 480, False, 'title-band', 52), (390, 570, True, 'safe-area', 34)],
                         ids=['desk-title-band', 'drawer-safe-area'])
def test_the_heads_measured_height_follows_a_band_or_a_safe_area_that_arrives_later(shell_ui, tmp_path, width, height,
                                                                                   touch, late, grows):
    """The head's measured height stays its real height when only its padding changes after the first paint, so the
    rows' scroll margin still keeps a focused row clear of it; the rail's first line moves with the head's. A drawer's
    top line is the larger of the page row's 18px and the inset (5px past it), so a 47px inset adds 34."""
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
    assert abs(after['head'] - before['head'] - grows) <= 1, ('the late padding reached the head', facts)
    for state in (before, after):
        assert abs(state['headPad'] - state['head']) <= 0.5, facts
        assert abs(state['railTop'] - state['headTop']) <= 0.5, ('the rail and the head share their first line', facts)


# Focus without a keyboard walk, so WebKit (which Tabs only to fields by default) measures it too: the head's control
# and the rail's focused while the column sits mid-scroll, and every row brought into view by focus() from half under
# the head.
FOCUS_MOVES = """async () => {
    const side = document.getElementById('primary-sidebar');
    const frames = () => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
    const head = side.querySelector('.sidebar-head');
    const version = document.getElementById('nav-version-check');  // the version is the rail's check (no capsule here)
    const bars = {}, missed = [];
    const controls = [['main', side.querySelector('.sidebar-head [data-nav-page="chat"]')], ['version', version],
        ...[...side.querySelectorAll('.nav-rail [data-nav-page]')].map(el => [el.dataset.navPage, el])];
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
    // Every row parked half under the head and brought into view from there: only the row's own scroll margin can move
    // it clear (a row inside the scrollport counts as visible without it).
    const hidden = [];
    let parked = 0;
    const rows = [...side.querySelectorAll('#nav-projects-list [data-project-id]')];
    const max = side.scrollHeight - side.clientHeight;
    for (const row of rows) {
        document.activeElement?.blur?.();
        side.scrollTop = 0;
        await frames();
        const r0 = row.getBoundingClientRect(), y = r0.top - side.getBoundingClientRect().top;
        const at = y + r0.height / 2 - head.getBoundingClientRect().height;
        if (at < 0 || at > max) continue;
        side.scrollTop = at;
        await frames();
        if (row.getBoundingClientRect().top >= head.getBoundingClientRect().bottom - 0.5) continue;
        parked += 1;
        row.focus();
        await frames();
        if (row.getBoundingClientRect().top < head.getBoundingClientRect().bottom - 0.5) hidden.push(row.textContent.trim().slice(0, 40));
    }
    return {bars, missed, rows: rows.length, parked, hidden};
}"""


@pytest.mark.parametrize(('width', 'height', 'touch'), [(1100, 480, False), (390, 570, True)],
                         ids=['desk', 'phone-drawer'])
def test_the_heads_and_the_rails_own_controls_leave_the_column_where_it_is(shell_ui, tmp_path, width, height, touch):
    """The rows carry the head's height as their scroll margin, not the column as its scroll padding: a row brought
    into view from half under the head clears it, and focusing Main or a place in the rail (which stays in view beside
    the list) does not scroll the column."""
    # A managed install whose status offers the check: the version is a focusable control, not plain text.
    page, _ = rail_page(shell_ui, width, height, lang='en', touch=touch)
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
    assert facts['check'] and facts['missed'] == [], ('every control took focus', facts)
    assert set(facts['bars']) == {'main', 'version', *RAIL_IDS}, facts
    for name, move in facts['bars'].items():
        assert move['before'] > 0 and abs(move['after'] - move['before']) <= 1, ('the column stayed put', name, facts)
    assert facts['parked'] > 20, ('rows were parked under the head', facts)
    assert facts['hidden'] == [], ('a row brought into view under the pinned head', facts)


# The rail measured where it stands: its box, each place's target and the word that is its name.
RU = {'Circles': 'Круги', 'Files': 'Файлы', 'Extensions': 'Расширения', 'Widgets': 'Виджеты', 'Dashboard': 'Панель управления',
      'Settings': 'Настройки', 'Update': 'Обновить', 'Up to date': 'Обновлений нет', 'Check failed': 'Не удалось проверить',
      'Updating…': 'Обновляю…'}
REST = {'managed': True, 'available': False, 'checked_at': '2026-11-21T10:00:00Z', 'local_changes': 'absent'}
# The rail's places in the registry (pages.js), in its order: Extensions opens through Settings (owner 24.11).
NAMES = ('Circles', 'Files', 'Widgets', 'Dashboard', 'Settings')


def shown(lang, labels=None):
    """The words the rows must carry as their names: the mocked translation and any other name put on a row."""
    names = {**(RU if lang == 'ru' else {}), **(labels or {})}
    return [names.get(name, name) for name in NAMES]


GEOMETRY = """() => {
    const side = document.getElementById('primary-sidebar'), rail = side.querySelector('.nav-rail');
    const box = (e) => e.getBoundingClientRect();
    const r = box(rail);
    const rows = [...rail.querySelectorAll('[data-nav-page]')].map((row) => {
        const label = row.querySelector('.nav-row-label'), b = box(row);
        return {id: row.dataset.navPage, label: label.textContent, title: row.title, left: b.left - r.left,
                right: r.right - b.right, top: b.top, width: b.width, height: b.height, labelDrawn: box(label).width > 1};
    });
    return {box: box(side).width, rail: r.width, column: box(side.querySelector('.sidebar-head')).width,
            railOverflow: rail.scrollWidth - rail.clientWidth, rows,
            skills: side.querySelectorAll('[data-nav-page="skills"]').length};
}"""


def rail_page(ui, width, height, *, lang='ru', touch=False, status=REST, check=None, labels=None):
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
        settings = (labels or {}).get('Settings') or (RU['Settings'] if lang == 'ru' else 'Settings')
        page.wait_for_function("s => document.querySelector('[data-nav-page=settings] .nav-row-label').textContent === s",
                               arg=settings)
    page.wait_for_function("document.getElementById('nav-version').textContent === 'v7.11.1-alpha.0'")
    return page, world


@pytest.mark.parametrize('lang', ['ru', 'en'])
def test_the_rail_keeps_its_width_and_its_names_at_every_column_width(shell_ui, tmp_path, lang):
    """The rail is beside the column (owner 24.11): whatever width the handle gave the column, the box is that width and
    the rail's 40px past its inset, the rail's places stand whole inside it, each its glyph alone, its word its name
    and its tooltip; Extensions has no place of its own there."""
    page, _ = rail_page(shell_ui, 1440, 900, lang=lang)
    facts = {'engine': shell_ui['engine'], 'lang': lang}
    try:
        for width in (270, 280, 474, 480, 560):
            page.evaluate(f"document.documentElement.style.setProperty('--sidebar-width', '{width}px')")
            settle(page)
            facts[width] = page.evaluate(GEOMETRY)
    finally:
        evidence(tmp_path, f'sidebar-rail-{lang}-{shell_ui["engine"]}', facts)
        page.close()
    for width in (270, 280, 474, 480, 560):
        at = facts[width]
        # The names measured are the ones asked for (a translation reset under the probe would measure other words).
        assert [row['label'] for row in at['rows']] == shown(lang), ('the names', width, at['rows'])
        assert [row['title'] for row in at['rows']] == shown(lang), ('the tooltips', width, at['rows'])
        assert round(at['box']) == width + 40 and round(at['column']) == width - 25, ('the room pays for the rail', width, at)
        assert round(at['rail']) == 52 and at['railOverflow'] <= 0 and at['skills'] == 0, (width, at)
        for row in at['rows']:
            assert row['width'] == row['height'] == 36 and abs(row['left'] - row['right']) <= 0.5, ('centred', width, row)
            assert not row['labelDrawn'], ('a glyph alone', width, row)


# The version's own line (its text, not its box: the plain version keeps the check's box, its padding included).
ROW = """() => { const meta = document.querySelector('.sidebar-foot-meta').getBoundingClientRect();
    const range = document.createRange();
    range.selectNodeContents(document.getElementById('nav-version'));
    const v = range.getBoundingClientRect();
    return {height: meta.height, versionTop: v.top, versionBottom: v.bottom}; }"""
STATE = """() => {
    const check = document.getElementById('nav-version-check'), note = document.querySelector('.nav-check-note');
    const pill = document.getElementById('update-pill');
    const glyph = check.querySelector('.nav-version-glyph');
    // What is drawn: the answer's own glyph, the version's glyph box and the number's lines (its text, not its box).
    const rect = (el) => { const r = el.getBoundingClientRect(); return {top: r.top, bottom: r.bottom, x: (r.left + r.right) / 2, w: r.width}; };
    const range = document.createRange();
    range.selectNodeContents(document.getElementById('nav-version'));
    const drawn = note.hidden ? null : rect(note.querySelector('svg'));
    // The tooltip a pointer resting on the answer reaches: the nearest title from what it hits.
    const hit = drawn && document.elementFromPoint(drawn.x, (drawn.top + drawn.bottom) / 2);
    return {busy: check.getAttribute('aria-busy'), check: !check.hidden, note: note.hidden ? '' : note.textContent,
            noteGlyph: drawn && drawn.w > 0 ? 'drawn' : '', drawn, glyphBox: rect(glyph.querySelector('svg')),
            versionGlyph: getComputedStyle(glyph).visibility, lines: range.getClientRects().length,
            text: rect(range), title: check.title, said: document.getElementById('nav-check-said').textContent,
            pointed: hit ? (hit.closest('[title]')?.getAttribute('title') || '') : null,
            pill: pill && !pill.hidden ? pill.textContent : '', version: document.getElementById('nav-version').textContent};
}"""


def test_the_version_row_keeps_one_height_and_its_number_never_moves(shell_ui, tmp_path):
    """The version is the check (owner 21.11): rest, hover, checking, Up to date and a failure stand in one row height,
    the answer a glyph in the version's glyph's place with its words read; the Update capsule and an attempt stand over
    the version, which keeps its place; one press is one request."""
    page, world = rail_page(shell_ui, 1440, 900)
    facts = {'engine': shell_ui['engine'], 'rows': {}, 'states': {}}

    def note(name):
        settle(page)
        facts['rows'][name] = page.evaluate(ROW)
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
        evidence(tmp_path, f'sidebar-rail-version-row-{shell_ui["engine"]}', facts)
        page.close()
    states, rows = facts['states'], facts['rows']
    assert facts['checks'] == 1, ('a press while it checks starts nothing', facts)
    assert states['checking']['busy'] == 'true' and states['rest']['busy'] is None, facts
    assert states['up to date']['note'] == 'Обновлений нет' and states['failed']['note'] == 'Не удалось проверить', facts
    # The fixture's version breaks onto a second line in the rail: the answer stands in the glyph's own place under the
    # whole number, never on its lines, wherever the number ends.
    assert states['rest']['lines'] >= 2, ('a version long enough to wrap', facts)
    for name in ('up to date', 'failed'):
        state = states[name]
        assert state['noteGlyph'] == 'drawn' and state['versionGlyph'] == 'hidden', ('the answer in its place', facts)
        assert state['drawn']['top'] >= state['text']['bottom'] - 0.5, ('under the whole number', name, facts)
        assert abs(state['drawn']['top'] - state['glyphBox']['top']) <= 0.5, ('the glyph\'s own place', name, facts)
        assert abs(state['drawn']['x'] - state['glyphBox']['x']) <= 0.5, ('the glyph\'s own place', name, facts)
    # A pointer resting on the warning reaches the reason (the check's tooltip, as the readers' words say it); the
    # check's own name stands again once the answer is gone.
    failed = states['failed']
    assert failed['title'].startswith('Could not check the official update channel') and failed['pointed'] == failed['title'], facts
    assert failed['said'] == failed['title'], facts
    assert states['up to date']['pointed'] == states['up to date']['title'] == states['rest']['title'], facts
    assert states['rest']['versionGlyph'] == 'visible', facts
    assert states['update']['pill'] == 'Обновить' and not states['update']['check'], ('one action per row', facts)
    assert states['attempt']['pill'] == 'Обновляю…' and not states['attempt']['check'], facts
    assert {state['version'] for state in states.values()} == {'v7.11.1-alpha.0'}, facts
    checks = [rows[name] for name in ('rest', 'hover', 'checking', 'up to date', 'failed')]
    assert max(r['height'] for r in checks) - min(r['height'] for r in checks) <= 0.5, ('one height for the check', facts)
    assert max(r['versionBottom'] for r in rows.values()) - min(r['versionBottom'] for r in rows.values()) <= 0.5, (
        'the number never moves', facts)
    assert rows['update']['height'] > rows['rest']['height'], ('the capsule stands over the version', facts)


# The docked zones in a window: Main's row and the rail's places inside the column, the list's own box and offsets.
ZONES = """() => {
    const side = document.getElementById('primary-sidebar'), list = document.getElementById('nav-projects-list');
    const s = side.getBoundingClientRect(), main = side.querySelector('.sidebar-head [data-nav-page="chat"]').getBoundingClientRect();
    return {main: main.top >= s.top - 0.5 && main.bottom <= s.bottom + 0.5,
            places: [...side.querySelectorAll('.nav-rail [data-nav-page], #nav-version-check')].map((c) => {
                const r = c.getBoundingClientRect(); return {top: r.top - s.top, bottom: s.bottom - r.bottom}; }),
            list: {height: list.clientHeight, overflow: list.scrollHeight - list.clientHeight, top: list.scrollTop},
            sideTop: side.scrollTop};
}"""


def test_a_1280_by_540_window_keeps_the_rail_whole_and_the_list_scrolls_alone(shell_ui, tmp_path):
    """Desktop 540 is the docked layout, not the short fallback: Main and the rail's places stay in the column while
    the wheel over the Projects moves the list alone, which has a real height of its own."""
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
        evidence(tmp_path, f'sidebar-rail-1280x540-{shell_ui["engine"]}', facts)
        page.close()
    assert facts['short'] is False, facts
    assert facts['scrollers'] == ['nav-projects-list'], ('the column itself does not scroll; the list does', facts)
    for state in (facts['before'], facts['after']):
        assert state['main'] and state['sideTop'] == 0, ('Main stays in the docked head', facts)
        assert len(state['places']) == 6, facts
        assert all(place['top'] >= 0 and place['bottom'] >= 0 for place in state['places']), ('the rail whole', facts)
    assert facts['before']['list']['height'] >= 80 and facts['before']['list']['overflow'] > 0, ('a list of its own', facts)
    assert facts['after']['list']['top'] > 0, ('the wheel moved the list', facts)


# Names longer than any translation so far: the rail draws glyphs, so a name never widens it or its column.
LONG = {'Files': 'Файлы, документы и вложения проекта', 'Dashboard': 'Панель управления и состояния системы',
        'Settings': 'Настройки и подключения этого Ouroboros'}


@pytest.mark.parametrize('width', [270, 480])
def test_a_name_longer_than_any_translation_never_widens_the_rail(shell_ui, tmp_path, width):
    page, _ = rail_page(shell_ui, 1440, 900, lang='ru', labels=LONG)
    facts = {'engine': shell_ui['engine']}
    try:
        page.evaluate(f"document.documentElement.style.setProperty('--sidebar-width', '{width}px')")
        settle(page)
        facts.update(page.evaluate(GEOMETRY))
    finally:
        evidence(tmp_path, f'sidebar-rail-long-{width}-{shell_ui["engine"]}', facts)
        page.close()
    assert [row['label'] for row in facts['rows']] == shown('ru', LONG), ('the names', facts)
    assert round(facts['rail']) == 52 and facts['railOverflow'] <= 0 and round(facts['box']) == width + 40, facts
    assert all(row['width'] == 36 and not row['labelDrawn'] for row in facts['rows']), facts


def test_the_drawer_keeps_the_same_rail_with_finger_targets(shell_ui, tmp_path):
    page, _ = rail_page(shell_ui, 390, 844, touch=True)
    facts = {'engine': shell_ui['engine']}
    try:
        open_drawer(page)
        facts.update(page.evaluate(GEOMETRY))
        facts['close'] = page.evaluate("""() => { const r = document.getElementById('nav-drawer-close').getBoundingClientRect();
            const rail = document.querySelector('.nav-rail').getBoundingClientRect();
            return {width: r.width, height: r.height, inRail: r.left >= rail.left && r.right <= rail.right,
                    above: r.bottom <= document.querySelector('.nav-rail [data-nav-page="circles"]').getBoundingClientRect().top}; }""")
        facts['version'] = page.evaluate("""() => { const c = document.getElementById('nav-version-check');
            const r = c.getBoundingClientRect(), a = getComputedStyle(c, '::after');
            return {height: r.height, reach: r.height - 2 * parseFloat(a.top || '0')}; }""")
    finally:
        evidence(tmp_path, f'sidebar-rail-drawer-{shell_ui["engine"]}', facts)
        page.close()
    assert [row['label'] for row in facts['rows']] == shown('ru'), ('the names', facts)
    assert round(facts['rail']) == 56 and facts['skills'] == 0, facts
    assert all(row['height'] >= 44 and row['width'] >= 44 for row in facts['rows']), ('a finger\'s target each way', facts)
    assert facts['close']['width'] >= 44 and facts['close']['inRail'] and facts['close']['above'], ('the × caps the rail', facts)
    assert facts['version']['reach'] >= 44 - 0.5, ('the version\'s reach is a finger\'s', facts)


# Which place stands chosen in the rail, and what the page shown is.
CHOSEN = """() => ({page: document.querySelector('.page.active')?.id || '',
    chosen: [...document.querySelectorAll('#primary-sidebar [data-nav-page].active')].map((row) => [row.dataset.navPage,
        row.getAttribute('aria-current')]),
    rows: document.querySelectorAll('#primary-sidebar [data-nav-page="skills"]').length})"""


def test_extensions_keeps_its_addresses_and_stands_under_settings(shell_ui, tmp_path):
    """Extensions has no place in the rail (owner 24.11, «как задумано»), but every way in still opens it: its own
    address `#skills`, the app's own `ouro:open-skills`, and Settings › Connections' door, whose ordinary Back returns
    to Settings. While it shows, Settings stands chosen for it, as the current one of the set, not the page itself."""
    page = seed_page(shell_ui, 1280, 860, touch=False)
    facts = {'engine': shell_ui['engine']}
    try:
        page.goto(shell_ui['url'] + '/#skills')
        page.locator('#page-skills').wait_for(state='visible')
        settle(page)
        facts['address'] = page.evaluate(CHOSEN)
        page.locator('#primary-sidebar [data-nav-page="chat"]').click()
        page.locator('#page-chat').wait_for(state='visible')
        page.evaluate("window.dispatchEvent(new CustomEvent('ouro:open-skills', {detail: {}}))")
        page.locator('#page-skills').wait_for(state='visible')
        settle(page)
        facts['event'] = page.evaluate(CHOSEN)
        page.locator('#primary-sidebar [data-nav-page="settings"]').click()
        page.locator('#page-settings').wait_for(state='visible')
        settle(page)
        facts['settings'] = page.evaluate(CHOSEN)
        page.locator('#page-settings [data-settings-open="page:skills"]').click()
        page.locator('#page-skills').wait_for(state='visible')
        settle(page)
        facts['door'] = page.evaluate(CHOSEN)
        back = page.locator('#page-skills > .app-page-header [data-shell-back]')
        facts['back'] = back.get_attribute('aria-label')
        back.click()
        page.locator('#page-settings').wait_for(state='visible')
        settle(page)
        facts['returned'] = page.evaluate(CHOSEN)
    finally:
        evidence(tmp_path, f'sidebar-rail-extensions-door-{shell_ui["engine"]}', facts)
        page.close()
    under = {'page': 'page-skills', 'chosen': [['settings', 'true']], 'rows': 0}
    assert facts['address'] == facts['event'] == facts['door'] == under, facts
    assert facts['settings'] == facts['returned'] == {'page': 'page-settings', 'chosen': [['settings', 'page']], 'rows': 0}, facts
    assert facts['back'] == 'Back to Settings', facts


RAIL_CROWD_HOLDS = """async () => {
    const side = document.getElementById('primary-sidebar');
    const rail = [], bars = [];
    for (let i = 0; i < 12; i += 1) {
        await new Promise((r) => requestAnimationFrame(r));
        rail.push(side.hasAttribute('data-rail-crowd'));
        bars.push(side.hasAttribute('data-bars-crowd'));
    }
    return {
        rail, bars,
        railPosition: getComputedStyle(side.querySelector('.nav-rail')).position,
        headPosition: getComputedStyle(side.querySelector('.sidebar-head')).position,
    };
}"""

RAIL_FOCUS_STOP = """() => {
    const side = document.getElementById('primary-sidebar');
    const rail = side.querySelector('.nav-rail');
    const f = document.activeElement;
    if (!f || !side.contains(f) || f === side) return null;
    const s = side.getBoundingClientRect();
    const r = f.getBoundingClientRect();
    const at = document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2);
    const id = f.dataset.navPage || f.id
        || (f.classList.contains('update-notice-close') ? 'update-notice-close' : f.className);
    return {
        id,
        inRail: Boolean(rail && rail.contains(f)),
        top: Math.round(r.top * 10) / 10,
        bottom: Math.round(r.bottom * 10) / 10,
        left: Math.round(r.left * 10) / 10,
        right: Math.round(r.right * 10) / 10,
        sideTop: Math.round(s.top * 10) / 10,
        sideBottom: Math.round(s.bottom * 10) / 10,
        inBounds: r.width > 0 && r.height > 0
            && r.top >= s.top - 0.5 && r.bottom <= s.bottom + 0.5
            && r.left >= s.left - 0.5 && r.right <= s.right + 0.5,
        hitSelf: Boolean(at && (at === f || f.contains(at))),
        hitTag: at ? (at.id || at.className || at.tagName) : null,
        scrollTop: Math.round(side.scrollTop * 10) / 10,
    };
}"""


def _choose_rail_tab_keys(page):
    """Use Tab/Shift+Tab where Tab focuses buttons (Chromium, Linux WebKit), or Option+Tab/Option+Shift+Tab on macOS
    WebKit where Safari's default skips buttons on plain Tab."""
    for fwd, rev in (('Tab', 'Shift+Tab'), ('Alt+Tab', 'Alt+Shift+Tab')):
        page.evaluate("""() => {
            const side = document.getElementById('primary-sidebar');
            side.scrollTop = 0;
            document.getElementById('nav-drawer-close').focus({preventScroll: true});
        }""")
        page.keyboard.press(fwd)
        if page.evaluate("() => document.activeElement?.dataset?.navPage === 'circles'"):
            return fwd, rev
    raise AssertionError('neither Tab nor Alt+Tab moved keyboard focus from #nav-drawer-close to Circles')


def _walk_rail_keyboard(page, key, start_selector, *, include_start, stop_on=None, capture_on=None, limit=16):
    page.evaluate("""(sel) => {
        const side = document.getElementById('primary-sidebar');
        side.scrollTop = 0;
        const el = document.querySelector(sel);
        if (!el) throw new Error('missing focus start: ' + sel);
        el.focus({preventScroll: true});
        if (document.activeElement !== el) throw new Error('start control did not take focus: ' + sel);
    }""", start_selector)
    settle(page)
    stops = []
    if include_start:
        stops.append(page.evaluate(RAIL_FOCUS_STOP))
    for _ in range(limit):
        page.keyboard.press(key)
        settle(page)
        stop = page.evaluate(RAIL_FOCUS_STOP)
        if stop is None or not stop['inRail']:
            break
        stops.append(stop)
        if capture_on and stop['id'] == capture_on[0]:
            setup_browser.capture(page, capture_on[1])
        if stop_on and stop['id'] == stop_on:
            break
    return stops


def test_a_very_short_drawer_reveals_rail_version_and_update_controls_by_keyboard(shell_ui, tmp_path):
    """In a 568×320 landscape touch drawer (30+ room rows), the rail is taller than the sidebar's content box while
    the head and one row still fit (`data-rail-crowd` true and stable over 12 frames, `data-bars-crowd` false, single
    scroller). Unpinning the rail lets genuine forward and reverse keyboard focus walks scroll every rail control —
    ordinary `#nav-version-check` and active `#update-pill` + `.update-notice-close` — cleanly inside sidebar bounds
    with passing hit-tests, and resizing to 390×570 and 390×844 restores sticky/tall behavior."""
    page, world = rail_page(shell_ui, 568, 320, lang='en', touch=True)
    facts = {'engine': shell_ui['engine']}
    try:
        open_drawer(page)
        page.wait_for_function(
            "() => { const c = document.getElementById('nav-version-check');"
            " return Boolean(c && !c.hidden && c.contains(document.getElementById('nav-version'))); }"
        )
        settle(page)
        facts['rows'] = page.evaluate("document.querySelectorAll('#nav-projects-list .nav-project-row').length")
        facts['scrollers'] = [row['node'] for row in sidebar_metrics(page)]
        facts['holds_ordinary'] = page.evaluate(RAIL_CROWD_HOLDS)
        fwd, rev = _choose_rail_tab_keys(page)
        facts['keys'] = [fwd, rev]
        facts['ordinary_fwd'] = _walk_rail_keyboard(
            page, fwd, '#nav-drawer-close', include_start=True,
            capture_on=('nav-version-check', f'sidebar-short-568x320-version-{shell_ui["engine"]}'),
        )
        facts['ordinary_rev'] = _walk_rail_keyboard(
            page, rev, '#primary-sidebar .sidebar-head [data-nav-page="chat"]',
            include_start=False, stop_on='nav-drawer-close',
        )
        # Switch to a terminal failed update attempt so both #update-pill and .update-notice-close are rendered.
        world['status'] = dict(
            REST,
            update_progress={
                'operation_id': 'op-short-drawer-failed',
                'generation': 'g',
                'active': False,
                'outcome': 'failed',
                'error': 'Simulated update failure',
                'started_at': '2026-11-21T10:12:00Z',
                'outcome_at': '2026-11-21T10:13:00Z',
            },
        )
        page.evaluate("window.__ouroWs.emit('update_progress_changed', {})")
        page.wait_for_function(
            "() => { const p = document.getElementById('update-pill');"
            " const x = document.querySelector('.update-notice-close');"
            " return Boolean(p && !p.hidden && x && !x.hidden && document.getElementById('nav-version-check')?.hidden); }"
        )
        settle(page)
        facts['holds_update'] = page.evaluate(RAIL_CROWD_HOLDS)
        facts['scrollers_update'] = [row['node'] for row in sidebar_metrics(page)]
        facts['update_fwd'] = _walk_rail_keyboard(
            page, fwd, '#nav-drawer-close', include_start=True,
            capture_on=('update-notice-close', f'sidebar-short-568x320-update-{shell_ui["engine"]}'),
        )
        facts['update_rev'] = _walk_rail_keyboard(
            page, rev, '#primary-sidebar .sidebar-head [data-nav-page="chat"]',
            include_start=False, stop_on='nav-drawer-close',
        )
        # Recovery to normal short drawer (390×570, sticky rail) and tall drawer (390×844, list-only scroller).
        page.set_viewport_size({'width': 390, 'height': 570})
        settle(page)
        page.evaluate(SCROLL, 0.5)
        settle(page)
        facts['recover_570'] = {
            'short': page.evaluate(SHORT),
            'scrollers': [row['node'] for row in sidebar_metrics(page)],
            'holds': page.evaluate(RAIL_CROWD_HOLDS),
            'rail': page.evaluate(RAIL),
        }
        page.set_viewport_size({'width': 390, 'height': 844})
        settle(page)
        facts['recover_844'] = {
            'short': page.evaluate(SHORT),
            'scrollers': [row['node'] for row in sidebar_metrics(page)],
            'holds': page.evaluate(RAIL_CROWD_HOLDS),
            'rail': page.evaluate(RAIL),
        }
    finally:
        evidence(tmp_path, f'sidebar-short-568x320-rail-focus-{shell_ui["engine"]}', facts)
        page.close()
    assert facts['rows'] >= 30, facts
    assert facts['scrollers'] == ['primary-sidebar'] and facts['scrollers_update'] == ['primary-sidebar'], facts
    for label in ('holds_ordinary', 'holds_update'):
        holds = facts[label]
        assert holds['rail'] == [True] * 12 and holds['bars'] == [False] * 12, (label, facts)
        assert holds['railPosition'] == 'static' and holds['headPosition'] == 'sticky', (label, facts)
    ordinary_order = ['nav-drawer-close', 'circles', 'files', 'widgets', 'dashboard', 'settings', 'nav-version-check']
    update_order = ['nav-drawer-close', 'circles', 'files', 'widgets', 'dashboard', 'settings', 'update-pill', 'update-notice-close']
    assert [s['id'] for s in facts['ordinary_fwd']] == ordinary_order, facts
    assert [s['id'] for s in facts['ordinary_rev']] == list(reversed(ordinary_order)), facts
    assert [s['id'] for s in facts['update_fwd']] == update_order, facts
    assert [s['id'] for s in facts['update_rev']] == list(reversed(update_order)), facts
    for walk_name in ('ordinary_fwd', 'ordinary_rev', 'update_fwd', 'update_rev'):
        for stop in facts[walk_name]:
            assert stop['inBounds'] and stop['hitSelf'], (walk_name, stop, facts)
    assert facts['ordinary_fwd'][-1]['scrollTop'] > 0 and facts['ordinary_rev'][0]['scrollTop'] > 0, facts
    assert facts['ordinary_rev'][-1]['scrollTop'] < facts['ordinary_rev'][0]['scrollTop'], facts
    assert facts['update_fwd'][-2]['scrollTop'] > 0 and facts['update_fwd'][-1]['scrollTop'] > 0, facts
    assert facts['update_rev'][0]['scrollTop'] > 0 and facts['update_rev'][1]['scrollTop'] > 0, facts
    assert facts['update_rev'][-1]['scrollTop'] < facts['update_rev'][0]['scrollTop'], facts
    r570, r844 = facts['recover_570'], facts['recover_844']
    assert r570['short'] is True and r570['scrollers'] == ['primary-sidebar'], r570
    assert r570['holds']['rail'] == [False] * 12 and r570['holds']['bars'] == [False] * 12, r570
    assert r570['holds']['railPosition'] == 'sticky' and r570['rail']['sectionsInView'] and r570['rail']['settingsInView'], r570
    assert r844['short'] is False and r844['scrollers'] == ['nav-projects-list'], r844
    assert r844['holds']['rail'] == [False] * 12 and r844['holds']['bars'] == [False] * 12, r844
    assert r844['holds']['railPosition'] == 'sticky' and not r844['rail']['scrolls'] and r844['rail']['settingsInView'], r844
