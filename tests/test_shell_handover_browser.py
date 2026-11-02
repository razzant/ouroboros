"""A place picked in the phone drawer: Main's words fade under a cover (owner 20.10, 4A; room-build lane R4, P4a).

Animating the opacity of Main's page restyles every row of a long history in WebKit (R4: ~190 ms leaving Main for a
Project, ~335 ms arriving from Settings, at 34 k elements), so shell_motion.js fades Main's words under a cover of the
room's own background inside it and Main's own opacity never moves. A room that lies over Main (the Project panel, a
page leaving over it) fades by its own opacity: the room under it must show through. No cover is left once the beat
is over, and under reduced motion (no beat, only the arriving room's short fade) the cover lifts and goes the same way.
Uses the static-server fixture; never starts the runtime or reads live data."""
from __future__ import annotations

import json

import pytest

from tests import test_subscription_setup_browser as setup_browser
from tests.test_design_system_gate_browser import rest, save_evidence, seed_page, settle

subscription_ui = setup_browser.subscription_ui
pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]

# Every frame of a place picked in the drawer: Main's own opacity, the cover over its words, and the other room's.
BEAT = """([other, ms]) => { window.__beat = new Promise((resolve) => {
    const samples = [], t0 = performance.now(), main = document.getElementById('page-chat');
    const read = () => {
        const cover = main.querySelector(':scope > .shell-cover'), room = document.querySelector(other);
        samples.push({ main: getComputedStyle(main).opacity, cover: cover ? Number(getComputedStyle(cover).opacity) : null,
            other: room ? Number(getComputedStyle(room).opacity) : null });
        if (performance.now() - t0 < ms) requestAnimationFrame(read); else resolve(samples);
    };
    requestAnimationFrame(read);
}); }"""
COVERS = "document.querySelectorAll('.shell-cover').length"


@pytest.fixture(params=['webkit', 'chromium'])
def handover_ui(request, monkeypatch):
    monkeypatch.setenv('OUROBOROS_UI_BROWSER_ENGINE', request.param)
    ui = request.getfixturevalue('subscription_ui')
    ui['engine'] = request.param
    return ui


def open_drawer(page, room):
    """The ☰ of the room on screen (a Project room covers Main's own)."""
    page.locator(f'{room} [data-mobile-nav-toggle]:visible').first.click()
    page.wait_for_function("() => window.__ouroLayers.top() === 'drawer'")
    page.wait_for_timeout(250)


def pick(page, row, other):
    """Tap a drawer row and record every frame of the step."""
    page.evaluate(BEAT, [other, 700])
    page.locator(row).first.click()
    samples = page.evaluate('window.__beat')
    rest(page)
    return samples


def test_main_s_words_fade_under_a_cover_and_the_rooms_over_it_fade_themselves(handover_ui, tmp_path):
    page = seed_page(handover_ui, 390, 844)
    facts = {'engine': handover_ui['engine']}
    try:
        page.goto(handover_ui['url'] + '/')
        page.wait_for_selector('#chat-input', state='attached')
        page.wait_for_selector('#nav-projects-list [data-project-id]', state='attached')
        settle(page)
        open_drawer(page, '#page-chat')
        facts['to_project'] = pick(page, '#nav-projects-list [data-project-id]', '#project-panel')
        facts['covers_after_project'] = page.evaluate(COVERS)
        open_drawer(page, '#project-panel')
        pick(page, '#primary-sidebar [data-nav-page="settings"]', '#page-settings')
        open_drawer(page, '#page-settings')
        facts['to_main'] = pick(page, '#primary-sidebar [data-nav-page="chat"]', '#page-settings')
        facts['covers_after_main'] = page.evaluate(COVERS)
    finally:
        save_evidence(tmp_path, f'shell-cover-{handover_ui["engine"]}', facts)
        print(json.dumps(facts))
        page.close()
    for step in ('to_project', 'to_main'):
        assert {s['main'] for s in facts[step]} == {'1'}, (f"{step}: Main's own opacity never moves", facts)
        assert any(s['other'] is not None and s['other'] < 0.95 for s in facts[step]), (f'{step}: the other room fades itself', facts)
    leaving = [s['cover'] for s in facts['to_project'] if s['cover'] is not None]
    assert leaving and min(leaving) < 0.9 and max(leaving) > 0.99, ("leaving, the cover rises over Main's words", facts)
    arriving = [s['cover'] for s in facts['to_main'] if s['cover'] is not None]
    assert arriving and max(arriving) > 0.9 and min(arriving) < 0.05, ("arriving, the cover lifts off Main's words", facts)
    assert facts['covers_after_project'] == 0 and facts['covers_after_main'] == 0, ('the beat takes its cover away', facts)


def test_with_reduced_motion_main_picked_in_the_drawer_is_uncovered_and_its_cover_goes(handover_ui, tmp_path):
    page = seed_page(handover_ui, 390, 844)
    page.emulate_media(reduced_motion='reduce')
    facts = {'engine': handover_ui['engine']}
    try:
        page.goto(handover_ui['url'] + '/')
        page.wait_for_selector('#chat-input', state='attached')
        settle(page)
        open_drawer(page, '#page-chat')
        pick(page, '#primary-sidebar [data-nav-page="settings"]', '#page-settings')
        open_drawer(page, '#page-settings')
        facts['to_main'] = pick(page, '#primary-sidebar [data-nav-page="chat"]', '#page-settings')
        page.wait_for_timeout(300)
        facts['covers'] = page.evaluate(COVERS)
    finally:
        save_evidence(tmp_path, f'shell-cover-reduced-{handover_ui["engine"]}', facts)
        print(json.dumps(facts))
        page.close()
    shown = [s['cover'] for s in facts['to_main'] if s['cover'] is not None]
    assert shown and max(shown) > 0.5, ('the short fade lifts a cover off Main', facts)
    assert facts['to_main'][-1]['cover'] is None and facts['covers'] == 0, ('then the cover goes: Main is seen', facts)
    assert {s['main'] for s in facts['to_main']} == {'1'}, ("Main's own opacity never moves", facts)
