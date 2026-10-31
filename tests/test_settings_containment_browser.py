"""P10: real Settings tabs contain their controls and scroll only their tab strip."""
from __future__ import annotations

import json

import pytest

from tests import test_subscription_setup_browser as setup_browser

subscription_ui = setup_browser.subscription_ui
pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]


@pytest.fixture(autouse=True)
def webkit(monkeypatch):
    monkeypatch.setenv("OUROBOROS_UI_BROWSER_ENGINE", "webkit")


@pytest.mark.parametrize("width", [360, 390])
def test_every_settings_tab_contains_long_copy_and_controls(subscription_ui, width):
    ui = subscription_ui
    page = ui["page"]
    page.set_viewport_size({"width": width, "height": 844})
    page.goto(ui["url"] + "/#settings")
    page.wait_for_selector("#btn-save-settings")
    page.wait_for_selector("#available-subagents-editor .available-subagent-row", state="attached")
    tabs = page.locator('[data-settings-tab]').evaluate_all(
        "tabs => tabs.map(tab => tab.dataset.settingsTab)")
    assert len(tabs) == 8, tabs
    results = {}
    for tab in tabs:
        page.locator(f'[data-settings-tab="{tab}"]').click()
        panel = page.locator(f'[data-settings-panel="{tab}"]')
        panel.wait_for(state="visible")
        page.evaluate("() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))")
        results[tab] = page.evaluate("""() => {
            const s = document.querySelector('.settings-scroll');
            const strip = document.querySelector('.settings-tabs');
            const box = strip.getBoundingClientRect(), parent = strip.parentElement.getBoundingClientRect();
            return {overflow: s.scrollWidth - s.clientWidth,
                stripLeft: box.left - parent.left, stripRight: box.right - parent.right};
        }""")
    assert all(r['overflow'] <= 1 and r['stripLeft'] >= -1 and r['stripRight'] <= 1
               for r in results.values()), json.dumps(results, indent=2)

    # Prove this guard observes the owner's original notification-label leak.
    page.locator('[data-settings-tab="appearance"]').click()
    legacy = page.add_style_tag(content='.local-toggle { white-space: nowrap; }')
    assert page.locator('.settings-scroll').evaluate('s => s.scrollWidth - s.clientWidth') > 1
    legacy.evaluate('el => el.remove()')

    # A component must also fit an embedded narrow column at a wide viewport,
    # where viewport-specific one-column overrides cannot rescue its tracks.
    page.set_viewport_size({"width": 1280, "height": 844})
    page.locator('[data-settings-tab="behavior"]').click()
    page.add_style_tag(content='.settings-shell { max-width: 250px; }')
    assert page.locator('.settings-scroll').evaluate('s => s.scrollWidth - s.clientWidth') <= 1


def test_settings_tab_fade_tracks_hidden_tabs(subscription_ui):
    page = subscription_ui['page']
    page.set_viewport_size({"width": 390, "height": 844})
    page.goto(subscription_ui['url'] + '/#settings')
    strip = page.locator('.settings-tabs')
    strip.wait_for()
    page.wait_for_function("document.querySelector('.settings-tabs').hasAttribute('data-scroll-end')")
    assert strip.get_attribute('data-scroll-start') is None
    page.locator('[data-settings-tab="about"]').click()
    page.wait_for_function("document.querySelector('.settings-tabs').hasAttribute('data-scroll-start')")
    strip.evaluate('el => el.scrollLeft = el.scrollWidth')
    page.wait_for_function("!document.querySelector('.settings-tabs').hasAttribute('data-scroll-end')")
    assert 'gradient' in strip.evaluate('el => getComputedStyle(el).maskImage')
