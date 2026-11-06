"""A Project on a finger-first screen (owner 06.11; docs/DESIGN.md "Shell").

On an iPad the Project panel came in beside a dimmed Main that a tap closed, and GigaAgent's head asked for one window
there («то есть в режиме проектов открывалось в едином окне как раньше было»), while the owner keeps the drawer on his
desk («я не хочу в десктопной обычной версии уробороса от этой шторки отказываться, это важно»). web/theme.js paints
the primary input beside the layout before CSS (`html[data-pointer]`); docked under a finger the Project takes the
whole content column in Main's place with its ×, nothing dimmed, no width handle, the sidebar live and the covered Main
inert once it has arrived; with a mouse the drawer is today's. While a Project is open Main's row counts Main's new
messages. Uses the static-server/API fixture; touch is the engine's emulation, not a claim about a real iPad.
"""
from __future__ import annotations

import pytest

from tests import test_subscription_setup_browser as setup_browser
from tests.test_design_system_gate_browser import seed_page
from tests.test_shell_interactions_browser import boot, evidence, open_project, shell_ui
from tests.test_status_marks_nav_browser import open_app, snapshot

subscription_ui = setup_browser.subscription_ui
pytestmark = [pytest.mark.ui_browser]
__all__ = ['shell_ui', 'subscription_ui']

GEOMETRY = """() => {
    const box = (selector) => {
        const node = document.querySelector(selector);
        if (!node) return null;
        const r = node.getBoundingClientRect(), style = getComputedStyle(node);
        return {x: Math.round(r.x), w: Math.round(r.width), h: Math.round(r.height),
                shown: r.width > 0 && r.height > 0 && style.display !== 'none' && style.visibility !== 'hidden'};
    };
    return {layout: document.documentElement.dataset.layout, pointer: document.documentElement.dataset.pointer,
            top: window.__ouroLayers.top(), panel: box('#project-panel'), content: box('#content'),
            scrim: box('#project-panel-backdrop'), close: box('#project-panel-close'),
            panelHandle: box('#project-panel > .resize-handle'), sidebarHandle: box('#primary-sidebar > .resize-handle'),
            sidebarInert: Boolean(document.getElementById('primary-sidebar')?.closest('[inert]')),
            mainInert: Boolean(document.getElementById('content')?.closest('[inert]'))};
}"""
ARRIVED = "() => document.getElementById('project-panel')?.classList.contains('open') && document.getElementById('content')?.inert === true"
CLOSED = "() => window.__ouroLayers.top() !== 'project'"


@pytest.mark.parametrize('size', [(1180, 820), (820, 1180)], ids=['landscape', 'portrait'])
def test_a_finger_opens_a_project_in_mains_place_with_the_sidebar_live(shell_ui, tmp_path, size):
    page = seed_page(shell_ui, *size, touch=True)
    facts = {}
    try:
        boot(page, shell_ui)
        facts['before'] = page.evaluate(GEOMETRY)
        open_project(page)
        page.wait_for_function(ARRIVED, timeout=5_000)
        facts['opened'] = page.evaluate(GEOMETRY)
        page.keyboard.press('Escape')
        page.wait_for_function(CLOSED)
        page.wait_for_timeout(400)
        facts['escaped'] = page.evaluate(GEOMETRY)
        open_project(page)
        page.wait_for_function(ARRIVED, timeout=5_000)
        page.locator('#primary-sidebar [data-nav-page="chat"]').click()  # the sidebar stays live: Main's row goes home
        page.wait_for_function(CLOSED)
        page.wait_for_timeout(400)
        facts['main_row'] = page.evaluate(GEOMETRY)
    finally:
        evidence(tmp_path, f'project-open-{shell_ui["engine"]}-finger-{size[0]}', facts)
        page.close()
    before, opened = facts['before'], facts['opened']
    assert (before['layout'], before['pointer']) == ('docked', 'coarse'), before
    # The room is the content column whole, with today's bar: nothing dimmed, nothing to drag, its × to close.
    assert opened['top'] == 'project', opened
    assert opened['panel']['x'] == opened['content']['x'] and opened['panel']['w'] == opened['content']['w'], opened
    assert not opened['scrim']['shown'], opened
    assert not opened['panelHandle']['shown'] and not opened['sidebarHandle']['shown'], opened
    assert opened['close']['shown'], opened
    # Covered Main takes no focus; the sidebar beside it stays usable.
    assert opened['mainInert'] and not opened['sidebarInert'], opened
    for case in ('escaped', 'main_row'):
        assert facts[case]['top'] != 'project' and not facts[case]['mainInert'], (case, facts[case])


def test_a_mouse_keeps_the_project_beside_main(shell_ui, tmp_path):
    page = seed_page(shell_ui, 1280, 800, touch=False)
    try:
        boot(page, shell_ui)
        open_project(page)
        page.wait_for_function("() => document.getElementById('project-panel')?.classList.contains('open')")
        page.wait_for_timeout(500)
        facts = page.evaluate(GEOMETRY)
        evidence(tmp_path, f'project-open-{shell_ui["engine"]}-mouse', facts)
    finally:
        page.close()
    # Unchanged on a desk (owner 18.10, 4A): its own width beside the dimmed room, its ×, both handles, Main not inert.
    assert facts['pointer'] == 'fine', facts
    assert facts['panel']['w'] < facts['content']['w'], facts
    assert facts['scrim']['shown'] and facts['close']['shown'], facts
    assert facts['panelHandle']['shown'] and facts['sidebarHandle']['shown'], facts
    assert not facts['mainInert'] and not facts['sidebarInert'], facts


@pytest.mark.parametrize('touch', [True, False], ids=['finger', 'mouse'])
def test_mains_row_counts_its_new_messages_while_a_project_is_open(shell_ui, tmp_path, touch):
    page, sync = open_app(shell_ui, 1180, 820, snapshot(), touch=touch)
    badge = '[data-nav-page="chat"] .unread-badge'
    facts = {}
    try:
        facts['main_on_screen'] = page.locator(badge).count()
        open_project(page)
        sync(snapshot(main_revision=1))  # Main's new message arrives while the Project covers or dims it
        page.wait_for_timeout(800)  # a read offered now would have answered by here and taken the number away
        facts['project_open'] = page.locator(badge).count()
        page.locator('#primary-sidebar [data-nav-page="chat"]').click()
        page.wait_for_function(CLOSED)
        page.wait_for_function(f"() => !document.querySelector('{badge}')", timeout=5_000)
        facts['back_on_main'] = page.locator(badge).count()
    finally:
        evidence(tmp_path, f'project-open-{shell_ui["engine"]}-main-unread-{touch}', facts)
        page.close()
    # Main on screen shows no number of its own; a Project over it (covered or dimmed) leaves Main's new message unread
    # and counted on its row, and back on Main the number goes. The read's own predicate is pinned statically too
    # (tests/test_shell_layout_static.py).
    assert (facts['main_on_screen'], facts['project_open'], facts['back_on_main']) == (0, 1, 0), facts
