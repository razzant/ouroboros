"""A repaint keeps the reading place (docs/DESIGN.md §10; web/modules/page_places.js).

A page draws again what it shows on a poll, a reconnect or a saved row. In WebKit (the desktop app's WKWebView,
Safari, Telegram on iOS) a box that is or lies in a size container (`container-type`: Settings, every Dashboard tab,
Files, Widgets, the Skills cards) fell back to its top whenever the part drawn again held an <svg>, so Settings
"teleported up" every few seconds while the owner read it (owner, 2026-11-13).

For every place in the registry (pages.js), every Dashboard and Skills tab, and every Settings view a module
registered, each box that scrolls is moved down and then its content is drawn again in place, the way a page's
repaint does (the part that holds most of it gets fresh, identical children). The box must not move up. A new
place, tab or Settings page is walked with no edit here. A room's feed and the Logs list keep their own reading place
(page_places.js OWN_PLACE) and are left to their own tests. The second test drives a real repaint: an account
provider's page, which the shared status store redraws on its poll.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tests import test_design_system_gate_browser as gate_browser
from tests.settings_browser_helpers import open_settings, walk_settings
from tests.test_design_system_gate_browser import nav_pages, open_nav, rest, save_evidence, seed_page

gate_ui = gate_browser.gate_ui
pytestmark = [pytest.mark.ui_browser]
STATUS = json.loads((Path(__file__).resolve().parents[1] / 'web/tests/fixtures/subscription_setup.json')
                    .read_text(encoding='utf-8'))['status']

# Every box on the shown place that scrolls: moved down (given room when it holds too little), its content drawn
# again in place, and read two frames later. Answers one row per box.
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
        box.scrollTop = 300;
        await frames();  // the scroll event: the owner left the box here
        const before = box.scrollTop;
        let host = box;
        for (;;) {
            const most = [...host.children].find((c) => c !== room && getComputedStyle(c).position !== 'sticky'
                && c.getBoundingClientRect().height >= 0.6 * (box.scrollHeight - (room ? room.offsetHeight : 0)));
            if (!most) break;
            host = most;
        }
        // The page's own nodes come back after the reading, so its handlers still answer the next step of the walk.
        const own = [...host.children];
        const fresh = own.map((c) => (c === room ? c : c.cloneNode(true)));
        if (!host.querySelector('svg')) {  // the same repaint with a mark in it, as most repaints have
            const mark = document.createElement('span');
            mark.innerHTML = '<svg width="8" height="8" aria-hidden="true"></svg>';
            fresh.splice(room ? fresh.indexOf(room) : fresh.length, 0, mark);
        }
        host.replaceChildren(...fresh);
        await frames();
        rows.push({ box: name(box), drew: name(host), before, after: box.scrollTop });
        host.replaceChildren(...own);
        room?.remove();
    }
    return rows;
}"""


def check(page, where, facts, failures):
    rows = page.evaluate(REPAINT)
    facts[where] = rows
    # Up is the defect. A copy may differ a little in height from what it replaced (a field's typed value, a picture
    # not decoded yet), and an engine that anchors the reading place answers that by moving the box down with it.
    moved = [row for row in rows if row['after'] < row['before'] - 2]
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
            tabs = page.locator(f'#page-{name} .app-tab-strip [role="tab"]').all()
            for index in range(max(1, len(tabs))):
                tab = 'page'
                if tabs:
                    tabs[index].click()
                    tab = tabs[index].inner_text().strip()
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
