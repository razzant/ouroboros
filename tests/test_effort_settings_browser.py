"""Rendered Settings keeps a reviewer row's effort and describes effort as a preference."""
import pytest

from tests import test_subscription_role_routes_browser as roles

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]
subscription_ui = roles.subscription_ui
role_ui = roles.role_ui


@pytest.mark.parametrize("width", [1360, 390])
def test_reviewer_effort_override_caption_and_saved_preference(role_ui, tmp_path, width):
    ui = role_ui
    roles.configure_mixed(ui)
    ui["settings"]["OUROBOROS_SUBAGENTS"]["items"][0]["effort"] = "high"
    ui["page"].set_viewport_size({"width": width, "height": 900})
    page = roles.open_agents(ui)
    rows = page.locator("[data-subagent-row]")
    pinned, unset = rows.nth(0), rows.nth(1)
    # A reviewer without its own effort reviews at the review default, and says so.
    assert "reviews at" not in pinned.locator("[data-subagent-review-facts]").inner_text()
    facts = unset.locator("[data-subagent-review-facts]")
    assert "reviews at high effort" in facts.inner_text()
    unset.locator('[data-subagent-field="effort"]').select_option('medium')
    assert "reviews at" not in facts.inner_text()
    unset.scroll_into_view_if_needed()
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    page.screenshot(path=str(tmp_path / f"effort-reviewer-{width}.png"))
    with page.expect_response('**/api/settings'):
        page.locator('#btn-save-settings').click()
    saved = roles.saved_catalog(ui)["items"]
    assert [row.get("effort") for row in saved[:2]] == ["high", "medium"]
    with page.expect_response('**/api/review-pool'):
        page.locator('#btn-reload-settings').click()
    assert unset.locator('[data-subagent-field="effort"]').input_value() == 'medium'
    assert "reviews at" not in facts.inner_text()
    page.locator('[data-settings-tab="behavior"]').click()
    help_text = page.get_by_text('Preferred reasoning effort per task type.', exact=False)
    assert "Requested, sent and reported effort are recorded in Logs" in help_text.inner_text()
    help_text.scroll_into_view_if_needed()
    page.screenshot(path=str(tmp_path / f"effort-settings-{width}.png"))
