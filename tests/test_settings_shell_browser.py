"""The Settings shell in a browser (docs/DESIGN.md "Settings"): the home opens a page, and every way back (the page's
first row, Escape, the browser's Back) returns where the owner was, one level at a time, the home keeping its scroll
and the field a level held committing before it closes. On a phone the ☰ stays in the header of every page; on a
desk the sidebar does. An address no page registers never shows an empty page: the home opens with its last word
searched."""
from __future__ import annotations

import pytest

from tests import test_subscription_setup_browser as setup_browser
from tests.settings_browser_helpers import HOME, go_back, open_settings, settings_page, top_path

subscription_ui = setup_browser.subscription_ui
pytestmark = [pytest.mark.ui_browser]

# A page family of the test's own, in the app's own registry: a field that commits on change, and a row that opens a
# second level over the first. The events the shell sends are kept. The three home rows the test presses get a
# stand-in page where their own lane has not registered one yet, so each opens a page on any tree.
TEST_PAGE = """async () => {
    const { registerSettingsPage, settingsPageFor } = await import('/static/modules/settings_pages.js');
    window.testChanges = [];
    window.testShown = [];
    addEventListener('ouro:settings-page-shown', (event) => window.testShown.push(event.detail.path));
    for (const [path, title] of [['computer', 'This computer'], ['rules', 'Your rules'], ['appearance', 'Appearance']]) {
        if (!settingsPageFor(path)) registerSettingsPage(path, { title, mount(body) { body.innerHTML = '<p>Stand-in</p>'; } });
    }
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
    # On a phone a sideways finger over a page's scroller is the drawer's (style.css, the stack layout's scrollers).
    touch = "pan-y pinch-zoom" if width < 720 else "auto"
    assert page.locator(HOME).evaluate("node => getComputedStyle(node).touchAction") == touch

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
    assert level.evaluate("node => getComputedStyle(node).touchAction") == touch
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

    # An address no page registers (a skill written for the tabs Settings had): never a page with nothing inside.
    # The home opens with the address's last word searched, and the results say where the thing lives now.
    open_settings(page, "test-shell")
    assert page.evaluate("() => window.__ouroSettings.open('secrets')") is True
    page.locator(f'{HOME} [data-settings-result="home:accounts"]').wait_for(state="visible")
    assert top_path(page) == "" and page.locator("#page-settings [data-settings-page]").count() == 0
    assert page.locator("#settings-search").input_value() == "secrets"
    assert page.locator(f"{HOME} [data-settings-ask]").is_visible()
    assert page.evaluate(SETTINGS_LAYERS) == []
    setup_browser.capture(page, f"settings-unknown-address-{width}x{height}")
    assert page.evaluate("() => window.__ouroSettings.open('old/advanced')") is True
    page.locator(f'{HOME} [data-settings-result="door:skill"]').wait_for(state="visible")
    assert page.locator("#settings-search").input_value() == "advanced"
    # The door among the results opens the list of skill forms. Words the shell searched are spent by that press:
    # Back returns to the home's own rows, never to results nobody asked for again.
    page.locator(f'{HOME} [data-settings-result="door:skill"]').click()
    forms = settings_page(page, "skill")
    forms.wait_for(state="visible")
    assert forms.locator(".settings-page-title").inner_text().strip() == "Skill settings"
    forms.locator("[data-skill-forms]").wait_for(state="visible")
    assert "No installed skill has settings." in forms.inner_text()
    assert go_back(page) == ""
    assert page.locator("#settings-search").input_value() == ""
    page.locator(f'{HOME} [data-settings-open="models"]').wait_for(state="visible")
    assert page.locator(f"{HOME} [data-settings-result]").count() == 0
    # Words the owner typed are the owner's: they stay through a press and the return.
    page.locator("#settings-search").fill("advanced")
    page.locator(f'{HOME} [data-settings-result="door:skill"]').click()
    settings_page(page, "skill").wait_for(state="visible")
    assert go_back(page) == ""
    assert page.locator("#settings-search").input_value() == "advanced"
    assert page.locator(f'{HOME} [data-settings-result="door:skill"]').is_visible()
    page.close()


def test_a_wide_page_box_shows_the_home_beside_the_page_with_no_layer(subscription_ui):
    ui = subscription_ui
    page = ui["new_page"](viewport={"width": 1440, "height": 900})
    page.goto(ui["url"] + "/#settings")
    page.locator(HOME).wait_for(state="visible")
    page.evaluate(TEST_PAGE)
    level, _kept = _open_from_home(page, "rules")
    setup_browser.capture(page, "settings-beside-1440x900")
    # Beside the home too the opened page takes the focus on its Back: a keyboard reaches the page at once.
    assert _focused(page, '[data-settings-page="rules"] [data-settings-back]')
    assert page.locator(HOME).is_visible(), "wide: the home stays beside the page"
    assert page.locator(f'{HOME} [data-settings-open="rules"]').get_attribute("aria-current") == "page"
    assert page.evaluate(SETTINGS_LAYERS) == [], "beside the home a level is no layer"
    # The home is a rail: its card keeps its list behind its count. The page keeps its first row, Back.
    back = level.locator("[data-settings-back]")
    assert back.is_visible() and back.inner_text().strip() == "Settings"
    assert level.locator(".settings-page-title").inner_text().strip() == "Your rules"
    assert not page.locator(f"{HOME} [data-being-needs]").is_visible()
    # Another home row replaces the page.
    page.locator(f'{HOME} [data-settings-open="appearance"]').click()
    settings_page(page, "appearance").wait_for(state="visible")
    assert level.count() == 0
    assert page.locator(f'{HOME} [data-settings-open="rules"]').get_attribute("aria-current") is None
    # A page opened from a page lies over it and names it: that Back is shown and closes one level.
    first = open_settings(page, "test-shell")
    first.locator("[data-test-deeper]").click()
    second = settings_page(page, "test-shell/deeper")
    second.wait_for(state="visible")
    back = second.locator("[data-settings-back]")
    assert back.is_visible() and back.inner_text().strip() == "Shell test"
    assert go_back(page) == "test-shell"
    # A page opened from outside Settings names where it came from, in the wide layout too.
    page.evaluate("() => window.__ouroSettings.open('test-shell', { back: { label: 'Skills', go() { window.testWentBack = true; } } })")
    outside = settings_page(page, "test-shell")
    assert outside.locator("[data-settings-back]").inner_text().strip() == "Skills"
    outside.locator("[data-settings-back]").click()
    page.wait_for_function("() => window.testWentBack === true")
    # Back beside the rail returns to the home alone; narrowed, a page is alone and one layer of the shell's stack.
    open_settings(page, "rules")
    assert go_back(page) == ""
    assert page.locator(f'{HOME} [aria-current="page"]').count() == 0
    open_settings(page, "rules")
    page.set_viewport_size({"width": 1087, "height": 740})
    page.wait_for_function(SETTINGS_LAYERS.replace("() =>", "() => 1 === (") + ").length")
    assert not page.locator(HOME).is_visible()
    assert go_back(page) == ""
    page.close()
