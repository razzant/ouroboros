"""The Settings shell in a browser (docs/DESIGN.md "Settings"): the home opens a page, and every way back (the page's
first row, Escape, the browser's Back) returns where the owner was, one level at a time, the home keeping its scroll
and the field a level held committing before it closes. On a phone the ☰ stays in the header of every page; on a
desk the sidebar does. A path no page serves shows its chrome with nothing inside."""
from __future__ import annotations

import pytest

from tests import test_subscription_setup_browser as setup_browser
from tests.settings_browser_helpers import HOME, go_back, open_settings, settings_page, top_path

subscription_ui = setup_browser.subscription_ui
pytestmark = [pytest.mark.ui_browser]

# A page family of the test's own, in the app's own registry: a field that commits on change, and a row that opens a
# second level over the first. The events the shell sends are kept.
TEST_PAGE = """async () => {
    const { registerSettingsPage } = await import('/static/modules/settings_pages.js');
    window.testChanges = [];
    window.testShown = [];
    addEventListener('ouro:settings-page-shown', (event) => window.testShown.push(event.detail.path));
    registerSettingsPage('test-shell', {
        title: (path) => (path === 'test-shell' ? 'Shell test' : 'Deeper'),
        mount(body, { path, ctx }) {
            body.innerHTML = path === 'test-shell'
                ? '<label>Note <input data-test-field></label> <button type="button" data-test-deeper>Deeper</button>'
                : '<p data-test-deep>Second level</p>';
            body.querySelector('[data-test-field]')?.addEventListener('change',
                (event) => window.testChanges.push(event.target.value));
            body.querySelector('[data-test-deeper]')?.addEventListener('click', () => ctx.openSettings('test-shell/deeper'));
        },
    });
}"""

SETTINGS_LAYERS = "() => window.__ouroLayers.ids().filter((id) => id.startsWith('settings-level-'))"


def _layout(page):
    return page.locator("html").get_attribute("data-layout")


def _menu_visible(page):
    """The way to the other places: the ☰ in Settings' header on a phone, the sidebar's Settings row on a desk."""
    if _layout(page) == "stack":
        return page.locator("#page-settings .app-page-header [data-mobile-nav-toggle]").is_visible()
    return page.locator('[data-nav-page="settings"]').is_visible()


def _focused(page, selector):
    return page.evaluate("selector => Boolean(document.activeElement?.matches(selector))", selector)


def _open_from_home(page, target):
    row = page.locator(f'{HOME} [data-settings-open="{target}"]')
    row.scroll_into_view_if_needed()
    kept = page.locator(HOME).evaluate("home => home.scrollTop")
    row.click()
    level = settings_page(page, target)
    level.wait_for(state="visible")
    return level, kept


@pytest.mark.parametrize("width,height", [(1087, 740), (320, 640)])
def test_home_opens_a_page_and_every_way_back_returns_one_level(subscription_ui, width, height):
    ui = subscription_ui
    page = ui["new_page"](viewport={"width": width, "height": height}, has_touch=width < 720)
    page.goto(ui["url"] + "/#settings")
    page.locator(HOME).wait_for(state="visible")
    page.evaluate(TEST_PAGE)
    assert _layout(page) == ("stack" if width < 720 else "docked")
    assert _menu_visible(page)

    # The home: six groups, one row per home row, each a door to its page or place.
    wanted = page.evaluate("async () => (await import('/static/modules/settings_rows.js')).HOME_ROWS.map((r) => r.to)")
    shown = page.locator(f"{HOME} [data-settings-open]").evaluate_all("rows => rows.map((r) => r.dataset.settingsOpen)")
    assert shown == wanted
    assert page.locator(f"{HOME} .settings-group").count() == 6
    setup_browser.capture(page, f"settings-home-{width}x{height}")

    # Open the last row's page: the home scrolled to reach it, and that place is kept.
    level, kept = _open_from_home(page, "computer")
    setup_browser.capture(page, f"settings-page-{width}x{height}")
    assert kept > 0, "the home scrolls at this size, so its place is worth keeping"
    assert not page.locator(HOME).is_visible(), "one column: the page alone"
    back = level.locator("[data-settings-back]")
    assert back.inner_text().strip() == "Settings" and back.get_attribute("aria-label") == "Back to Settings"
    assert level.locator(".settings-page-title").inner_text().strip()
    assert _focused(page, '[data-settings-page="computer"] [data-settings-back]')
    assert _menu_visible(page)
    assert len(page.evaluate(SETTINGS_LAYERS)) == 1, "a narrow level is one layer of the shell's stack"
    assert page.evaluate("() => window.testShown.at(-1)") == "computer"

    # Back by the first row: the home, at the place it was left, focus on the row that opened the page.
    assert go_back(page) == ""
    page.locator(HOME).wait_for(state="visible")
    assert page.locator(HOME).evaluate("home => home.scrollTop") == kept
    assert _focused(page, f'{HOME} [data-settings-open="computer"]')
    assert page.evaluate(SETTINGS_LAYERS) == []
    assert page.evaluate("() => window.testShown.at(-1)") == ""

    # Escape closes the level the same way.
    _open_from_home(page, "computer")
    page.keyboard.press("Escape")
    page.locator(HOME).wait_for(state="visible")
    assert top_path(page) == "" and page.locator(HOME).evaluate("home => home.scrollTop") == kept

    # The browser's Back too, and it stays in Settings.
    _open_from_home(page, "computer")
    page.go_back()
    page.locator(HOME).wait_for(state="visible")
    assert top_path(page) == "" and page.locator("#page-settings").is_visible()
    assert page.locator(HOME).evaluate("home => home.scrollTop") == kept

    # Two levels: the second names the first as where Back leads, and each way back closes one level.
    first = open_settings(page, "test-shell")
    assert first.locator("[data-settings-back]").inner_text().strip() == "Settings"
    first.locator("[data-test-deeper]").click()
    second = settings_page(page, "test-shell/deeper")
    second.wait_for(state="visible")
    assert second.locator("[data-settings-back]").inner_text().strip() == "Shell test"
    assert second.locator(".settings-page-title").inner_text().strip() == "Deeper"
    assert _menu_visible(page)
    assert len(page.evaluate(SETTINGS_LAYERS)) == 2
    page.keyboard.press("Escape")
    first.wait_for(state="visible")
    assert top_path(page) == "test-shell"
    first.locator("[data-test-deeper]").click()
    second.wait_for(state="visible")
    page.go_back()
    first.wait_for(state="visible")
    assert top_path(page) == "test-shell"

    # A field that holds focus is left before its level closes, so its row commits.
    field = first.locator("[data-test-field]")
    field.fill("draft")
    assert _focused(page, "[data-test-field]")
    page.keyboard.press("Escape")
    page.locator(HOME).wait_for(state="visible")
    assert page.evaluate("() => window.testChanges") == ["draft"]

    # A path no page serves yet: the chrome, with nothing inside.
    empty = open_settings(page, "test-unregistered")
    assert empty.locator(".settings-page-title").inner_text().strip() == "test-unregistered"
    assert empty.locator(".settings-page-body").inner_html() == ""
    assert go_back(page) == ""
    page.close()


def test_a_wide_page_box_shows_the_home_beside_the_page_with_no_layer(subscription_ui):
    ui = subscription_ui
    page = ui["new_page"](viewport={"width": 1440, "height": 900})
    page.goto(ui["url"] + "/#settings")
    page.locator(HOME).wait_for(state="visible")
    level, _kept = _open_from_home(page, "rules")
    setup_browser.capture(page, "settings-beside-1440x900")
    assert page.locator(HOME).is_visible(), "wide: the home stays beside the page"
    assert page.locator(f'{HOME} [data-settings-open="rules"]').get_attribute("aria-current") == "page"
    assert page.evaluate(SETTINGS_LAYERS) == [], "beside the home a level is no layer"
    # Another home row replaces the page; Back returns to the home alone.
    page.locator(f'{HOME} [data-settings-open="appearance"]').click()
    settings_page(page, "appearance").wait_for(state="visible")
    assert level.count() == 0
    assert page.locator(f'{HOME} [data-settings-open="rules"]').get_attribute("aria-current") is None
    assert go_back(page) == ""
    assert page.locator(f'{HOME} [aria-current="page"]').count() == 0
    page.close()
