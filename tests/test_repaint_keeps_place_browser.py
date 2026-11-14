"""A repaint keeps the reading place (docs/DESIGN.md §10; web/modules/page_places.js).

A page draws again what it shows on a poll, a reconnect or a saved row. In WebKit (the desktop app's WKWebView,
Safari, Telegram on iOS) a box that is or lies in a size container (`container-type`: Settings, every Dashboard tab,
Files, Widgets, the Skills cards) is held to what remains while a part with an <svg> is drawn again: it falls to its
top, or part of the way under what stands above the part, so Settings "teleported up" every few seconds while the
owner read it (owner, 2026-11-13).

For every place in the registry (pages.js), every Dashboard and Skills tab, and every Settings view a module
registered, each box that scrolls is moved most of the way down and then its content is drawn again in place, the way a
page's repaint does (the part that holds most of it gets fresh, identical children). The box must not end above where
it was, or above its new end. A new place, tab or Settings page is walked with no edit here. A room's feed and the Logs
list keep their own reading place (page_places.js OWN_PLACE) and are left to their own tests. The second test drives a
real repaint: an account provider's page, which the shared status store redraws on its poll. The third is Files, which
shows the same folder again and another folder in one box.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

from tests import test_design_system_gate_browser as gate_browser
from tests import test_subscription_setup_browser as setup_browser
from tests.settings_browser_helpers import open_settings
from tests.test_design_system_gate_browser import nav_pages, open_nav, rest, save_evidence, seed_page, walk_settings

# The gate's page fixture and the one it stands on, by name, so pytest finds both here.
gate_ui = gate_browser.gate_ui
subscription_ui = setup_browser.subscription_ui
pytestmark = [pytest.mark.ui_browser]
STATUS = json.loads((Path(__file__).resolve().parents[1] / 'web/tests/fixtures/subscription_setup.json')
                    .read_text(encoding='utf-8'))['status']

# Every box on the shown place that scrolls: moved most of the way down (given room when it holds too little), its
# content drawn again in place, and read two frames later. Answers one row per box.
REPAINT = """async () => {
    const { OWN_PLACE } = await import('/static/modules/page_places.js');
    const frames = () => new Promise((done) => requestAnimationFrame(() => requestAnimationFrame(done)));
    const name = (e) => e.tagName.toLowerCase() + (e.id ? '#' + e.id : '') + [...e.classList].slice(0, 2).map((c) => '.' + c).join('')
        + (e.dataset.settingsPage ? `[${e.dataset.settingsPage}]` : '');
    const boxes = [...document.querySelectorAll('#content *')].filter((e) => {
        const style = getComputedStyle(e);
        return /(auto|scroll)/.test(style.overflowY) && e.getClientRects().length && e.clientHeight > 80
            && !e.closest(OWN_PLACE) && e.tagName !== 'TEXTAREA';
    });
    const rows = [];
    for (const box of boxes) {
        let room = null;
        if (box.scrollHeight - box.clientHeight < 400) {
            room = document.createElement('div');
            room.style.height = `${box.clientHeight + 400}px`;
            box.append(room);
        }
        let host = box;
        for (;;) {
            const most = [...host.children].find((c) => c !== room && getComputedStyle(c).position !== 'sticky'
                && c.getBoundingClientRect().height >= 0.6 * (box.scrollHeight - (room ? room.offsetHeight : 0)));
            if (!most) break;
            host = most;
        }
        // The room given stands in the part drawn again, as a long page's own rows would: beside the part it would
        // hold the box open through the redraw and hide the drop.
        if (room) host.append(room);
        box.scrollTop = Math.round((box.scrollHeight - box.clientHeight) * 0.8);
        await frames();  // the scroll event: the owner left the box here
        const before = box.scrollTop;
        // The page's own nodes come back after the reading, so its handlers still answer the next step of the walk.
        const own = [...host.children];
        const fresh = own.map((c) => c.cloneNode(true));
        if (!host.querySelector('svg')) {  // the same repaint with a mark in it, as most repaints have
            const mark = document.createElement('span');
            mark.innerHTML = '<svg width="8" height="8" aria-hidden="true"></svg>';
            fresh.push(mark);
        }
        host.replaceChildren(...fresh);
        await frames();
        rows.push({ box: name(box), drew: name(host), before, end: box.scrollHeight - box.clientHeight, after: box.scrollTop });
        host.replaceChildren(...own);
        room?.remove();
    }
    return rows;
}"""


def check(page, where, facts, failures):
    rows = page.evaluate(REPAINT)
    facts[where] = rows
    # Up is the defect. A copy may differ a little in height from what it replaced (a field's typed value, a picture
    # not decoded yet): a box that ends sooner stands at its new end, and an engine that anchors the reading place
    # answers a taller copy by moving the box down with it.
    moved = [row for row in rows if row['after'] < min(row['before'], row['end']) - 2]
    if moved:
        failures[where] = moved


@pytest.mark.parametrize('width,height', [(1087, 744), (390, 844)], ids=['desk', 'phone'])
def test_a_repaint_keeps_every_scrolled_box_where_the_owner_left_it(gate_ui, tmp_path, width, height):
    ui = gate_ui
    page = seed_page(ui, width, height)
    facts, failures = {}, {}
    try:
        page.goto(ui['url'] + '/')
        page.wait_for_selector('#chat-input', state='attached')
        # An engine that keeps the place through a redraw is left alone (the rule's one probe).
        facts['engine drops'] = page.evaluate("async () => (await import('/static/modules/page_places.js')).dropsOnRedraw()")
        if ui['engine'] == 'chromium':
            assert facts['engine drops'] is False
        for name in nav_pages(page):
            open_nav(page, name)
            if name == 'settings':
                walk_settings(page, lambda view: check(page, f'settings/{view}', facts, failures),
                              skill='gate_skill', settle=lambda: rest(page))
                continue
            if name == 'skills':
                page.wait_for_selector('.skills-card')
            if name == 'widgets':
                page.wait_for_selector('.widgets-card')
            selector = '[data-dashboard-tab]' if name == 'dashboard' else '.app-tab-strip [role="tab"]'
            tabs = page.locator(f'#page-{name} {selector}').all()
            if name == 'dashboard':
                assert [tab.get_attribute('data-dashboard-tab') for tab in tabs] == [
                    'logs', 'activity', 'consciousness', 'costs', 'updates', 'storage', 'evolution']
            for index in range(max(1, len(tabs))):
                tab = 'page'
                if tabs:
                    if name == 'dashboard' and not tabs[index].is_visible():
                        page.locator('#page-dashboard [data-dashboard-list-back]').click()
                        rest(page)
                    tabs[index].click()
                    tab = tabs[index].get_attribute('data-dashboard-tab') if name == 'dashboard' else tabs[index].inner_text().strip()
                rest(page)
                check(page, f'{name}/{tab}', facts, failures)
        assert any(facts.values()), 'no place had a box that scrolls'
    finally:
        path = save_evidence(tmp_path, f'repaint-place-{ui["engine"]}-{width}x{height}', facts)
        page.close()
    assert not failures, f'{ui["engine"]} {width}x{height}, evidence {path}:\n{json.dumps(failures, indent=2)}'


def test_an_account_provider_page_keeps_its_place_through_the_status_poll(gate_ui):
    """The shared status store redraws a provider's accounts on its poll (claudexor_status_store.js, every 5 s while
    the page shows; harness_accounts.js renderRows): with as many accounts as an owner keeps, the place survives the
    next tick."""
    status = copy.deepcopy(STATUS)
    one = status['profiles']['profiles'][0]
    status['profiles']['profiles'] = [
        {**one, 'profile': {**one['profile'], 'profile_id': f'account-{n}'}, 'identity': {'label': f'Account {n}'}}
        for n in range(24)]
    page = seed_page(gate_ui, 1087, 744, touch=False)
    page.route('**/api/claudexor/status*', lambda route: route.fulfill(json=status))
    try:
        page.goto(gate_ui['url'] + '/')
        page.wait_for_selector('#chat-input', state='attached')
        level = open_settings(page, 'accounts/codex')
        level.locator('#harness-accounts-groups .harness-account-row').nth(20).wait_for(state='attached')
        assert level.evaluate('box => box.scrollHeight - box.clientHeight') > 600, 'the page holds a long list'
        level.evaluate('box => { box.scrollTop = 300; }')
        rest(page)
        drawn = level.evaluate_handle("box => box.querySelector('#harness-accounts-groups').firstElementChild")
        page.wait_for_function("""old => document.querySelector('#harness-accounts-groups')?.firstElementChild !== old""",
                               arg=drawn, timeout=12_000)
        rest(page)
        assert level.evaluate('box => box.scrollTop') == 300
    finally:
        page.close()


def test_files_draws_the_same_folder_in_place_and_reads_another_from_its_first_row(gate_ui):
    """Files draws its list again when a file is chosen (the same folder: the place stays) and when a folder is opened
    (something else in the same box: Files sends it to its top itself). Both folders hold about as much, so only the
    page's own move tells the two apart."""
    def files(route):
        url = urlparse(route.request.url)
        path = parse_qs(url.query).get('path', ['.'])[0]
        here = '' if path == '.' else f'{path}/'
        if url.path.endswith('/list'):
            entries = [{'name': f'{"note" if n % 2 else "folder"}-{n:02}', 'path': f'{here}{"note" if n % 2 else "folder"}-{n:02}',
                        'type': 'file' if n % 2 else 'dir', 'size': 31} for n in range(60)]
            route.fulfill(json={'path': path, 'parent_path': '.', 'root_path': '/workspace', 'display_path': '/workspace',
                                'breadcrumb': [{'name': 'Workspace', 'path': '.'}], 'entries': entries})
        elif url.path.endswith('/read'):
            route.fulfill(json={'path': path, 'name': path, 'size': 31, 'root_path': '/workspace',
                                'display_path': f'/workspace/{path}', 'is_text': True, 'content': 'A line\n'})
        else:
            route.fulfill(status=404, json={'error': 'Not part of this case.'})
    # A row of one kind that stands whole inside the list, so a click moves nothing.
    in_view = """(box, kind) => {
        const frame = box.getBoundingClientRect();
        return [...box.querySelectorAll('.files-entry[data-path]')].find((row) => {
            const at = row.getBoundingClientRect();
            return row.dataset.path.startsWith(kind) && at.top >= frame.top + 60 && at.bottom <= frame.bottom - 60;
        }).dataset.path;
    }"""
    page = seed_page(gate_ui, 1087, 744, touch=False)
    page.route('**/api/files/**', files)
    try:
        page.goto(gate_ui['url'] + '/')
        page.wait_for_selector('#chat-input', state='attached')
        open_nav(page, 'files')
        box = page.locator('#files-list')
        box.locator('.files-entry').nth(59).wait_for(state='attached')
        box.evaluate('box => { box.scrollTop = 300; }')
        rest(page)
        box.locator(f'.files-entry[data-path="{box.evaluate(in_view, "note")}"]').click()
        box.locator('.files-entry.selected').wait_for(state='attached')
        rest(page)
        assert box.evaluate('box => box.scrollTop') == 300, 'the same folder drawn again keeps the place'
        box.locator(f'.files-entry[data-path="{box.evaluate(in_view, "folder")}"]').click()
        box.locator('.files-entry.parent-link').wait_for(state='attached')
        rest(page)
        assert box.evaluate('box => box.scrollTop') == 0, 'another folder is read from its first row'
    finally:
        page.close()
