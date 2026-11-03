"""Menus grow a little from where they were opened and leave the same way (owner 20.10, A; docs/DESIGN.md §10
"Moving between places"): the composer's "+" menu fades in from its button's side and its picture fades back toward it
when it closes, while the menu itself is gone at once. Reduced motion: at once both ways. Uses the static-server
fixture; never starts the runtime or reads live data."""
from __future__ import annotations

import json

import pytest

from tests import test_subscription_setup_browser as setup_browser
from tests.test_design_system_gate_browser import save_evidence, seed_page, settle

subscription_ui = setup_browser.subscription_ui
pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]


@pytest.fixture(params=['webkit', 'chromium'])
def menu_ui(request, monkeypatch):
    monkeypatch.setenv('OUROBOROS_UI_BROWSER_ENGINE', request.param)
    ui = request.getfixturevalue('subscription_ui')
    ui['engine'] = request.param
    return ui


# Open the "+" menu, read its entrance, close it with Escape, read what is left of it on the next frames.
OPEN_AND_CLOSE = """async () => {
    const frame = () => new Promise((resolve) => requestAnimationFrame(resolve));
    document.getElementById('chat-plus').click();
    const menu = document.querySelector('body > .chat-plus-menu');
    const entrance = menu.getAnimations().map((a) => a.animationName);
    const origin = menu.style.getPropertyValue('--ui-popup-origin');
    const box = menu.getBoundingClientRect(), button = document.getElementById('chat-plus').getBoundingClientRect();
    const height = Math.round(menu.offsetHeight);
    await new Promise((resolve) => setTimeout(resolve, 400));
    (document.activeElement || document.body).dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
    const after = [];
    for (let i = 0; i < 3; i += 1) {
        const pictures = [...document.querySelectorAll('body > .chat-plus-menu')];
        after.push(pictures.map((node) => ({ leaving: node.hasAttribute('data-leaving'), hidden: node.getAttribute('aria-hidden'),
            inert: node.inert, opacity: Number(getComputedStyle(node).opacity) })));
        await frame();
    }
    await new Promise((resolve) => setTimeout(resolve, 400));
    return { entrance, origin, above: box.bottom <= button.top + 1, height, after,
             left: document.querySelectorAll('body > .chat-plus-menu').length,
             expanded: document.getElementById('chat-plus').getAttribute('aria-expanded') };
}"""


@pytest.mark.parametrize('reduce', [False, True], ids=['motion', 'reduced'])
def test_the_plus_menu_grows_from_its_button_and_leaves_the_same_way(menu_ui, tmp_path, reduce):
    page = seed_page(menu_ui, 390, 844)
    if reduce:
        page.emulate_media(reduced_motion='reduce')
    facts = {'engine': menu_ui['engine'], 'reduce': reduce}
    try:
        page.goto(menu_ui['url'] + '/')
        page.wait_for_selector('#chat-input', state='attached')
        settle(page)
        facts.update(page.evaluate(OPEN_AND_CLOSE))
    finally:
        save_evidence(tmp_path, f'menu-motion-{menu_ui["engine"]}-{"reduced" if reduce else "motion"}', facts)
        print(json.dumps(facts))
        page.close()
    assert facts['left'] == 0 and facts['expanded'] == 'false', ('the menu and its picture are gone', facts)
    if reduce:
        assert facts['entrance'] == [], facts
        assert facts['after'][0] == [], ('closed at once, no picture', facts)
        return
    assert facts['entrance'] == ['ui-popup-in'], facts
    # Opened above its button, it grows from its foot (the button's side), not from a fixed corner.
    assert facts['above'] and facts['origin'].endswith(f" {facts['height']}px"), facts
    first = facts['after'][0]
    assert len(first) == 1 and first[0]['leaving'] and first[0]['hidden'] == 'true' and first[0]['inert'], (
        'the closed menu leaves as an inert picture assistive tech does not see', facts)
    assert any(step and step[0]['opacity'] < 1 for step in facts['after'][1:]), ('and it fades', facts)
