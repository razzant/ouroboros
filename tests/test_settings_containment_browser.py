"""Settings contains its controls (docs/DESIGN.md "Settings"): the home and every page fit their column on a phone,
and fit a narrow column inside a wide window too, where no viewport rule could rescue a track: the layout reads the
Settings box, never the window."""
from __future__ import annotations

import json

import pytest

from tests import test_subscription_setup_browser as setup_browser
from tests.settings_browser_helpers import HOME, open_settings, top_path, walk_settings

subscription_ui = setup_browser.subscription_ui
pytestmark = [pytest.mark.ui_browser]

OVERFLOW = """() => {
    const column = [...document.querySelectorAll('#page-settings .settings-home, #page-settings .settings-page')]
        .filter((box) => box.getClientRects().length);
    return Math.max(...column.map((box) => box.scrollWidth - box.clientWidth));
}"""
FORM = {"skills": [dict(name="gate_skill", enabled=True, loaded=True, type="extension")],
        "live": {"settings_sections": [dict(skill="gate_skill", section_id="config", key="gate_skill:config", title="Gate skill settings",
            render={"components": [{"type": "form", "id": "form", "route": "settings", "submit_label": "Save", "fields": [
                {"name": "address", "label": "A service address with a long descriptive label", "type": "text"},
                {"name": "mode", "label": "Mode", "type": "select", "default": "safe",
                 "options": [{"label": "Safe", "value": "safe"}, {"label": "Fast", "value": "fast"}]}]}]})]}}


@pytest.fixture(autouse=True)
def webkit(monkeypatch):
    monkeypatch.setenv("OUROBOROS_UI_BROWSER_ENGINE", "webkit")


def _walk(page):
    results = {}
    views = walk_settings(page, lambda view: results.__setitem__(view, page.evaluate(OVERFLOW)), skill="gate_skill")
    assert "skill-sources" in views and "skill/gate_skill" in views, views
    return results


@pytest.mark.parametrize("width", [320, 360, 390])
def test_the_home_and_every_page_contain_their_controls(subscription_ui, width):
    ui = subscription_ui
    ui["settings"]["_meta"]["restart_state"] = {"restart_required": True, "restart_keys": ["OUROBOROS_SKILLS_REPO_PATH"],
                                                "restart_source_unknown_keys": [], "local_model": {"pending_keys": ["LOCAL_MODEL_PORT"]}}
    page = ui["new_page"](viewport={"width": width, "height": 844}, has_touch=True)
    page.route("**/api/extensions", lambda route: route.fulfill(json=FORM))
    page.goto(ui["url"] + "/#settings")
    page.locator(HOME).wait_for(state="visible")
    results = _walk(page)
    assert all(extra <= 1 for extra in results.values()), json.dumps(results, indent=2)

    # The guard observes a leak: a search field wider than its column makes the home scroll sideways.
    assert top_path(page) == ""
    legacy = page.add_style_tag(content="#page-settings .settings-search { min-width: 130%; }")
    assert page.evaluate(OVERFLOW) > 1
    legacy.evaluate("el => el.remove()")
    assert page.evaluate(OVERFLOW) <= 1
    page.close()


def test_a_narrow_column_in_a_wide_window_is_laid_out_as_a_phone(subscription_ui):
    ui = subscription_ui
    page = ui["new_page"](viewport={"width": 1280, "height": 844})
    page.route("**/api/extensions", lambda route: route.fulfill(json=FORM))
    page.goto(ui["url"] + "/#settings")
    page.locator(HOME).wait_for(state="visible")
    # The window is wide; the Settings box is not. One column, a Back on the page, nothing wider than its column.
    page.add_style_tag(content="#page-settings { max-width: 250px; }")
    results = _walk(page)
    assert all(extra <= 1 for extra in results.values()), json.dumps(results, indent=2)
    level = open_settings(page, "skill-sources")
    assert level.locator("[data-settings-back]").is_visible() and not page.locator(HOME).is_visible()
    page.close()
