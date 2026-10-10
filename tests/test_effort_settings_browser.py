"""Rendered Settings: a reviewer row's effort select says Auto, and Behavior reads the range only."""
import pytest

from tests import test_subscription_role_routes_browser as roles

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]
subscription_ui = roles.subscription_ui
role_ui = roles.role_ui


@pytest.mark.parametrize("width", [1360, 390])
def test_reviewer_effort_auto_caption_and_saved_preference(role_ui, tmp_path, width):
    ui = role_ui
    roles.configure_mixed(ui)
    ui["settings"]["OUROBOROS_SUBAGENTS"]["items"][0]["effort"] = "high"
    ui["settings"]["OUROBOROS_EFFORT_TASK"] = "high"
    ui["page"].set_viewport_size({"width": width, "height": 900})
    page = roles.open_agents(ui)
    rows = page.locator("[data-subagent-row]")
    pinned, unset = rows.nth(0), rows.nth(1)
    # The empty option is Auto: the chat range; a marked row reviews at its top and says so.
    default = 'option[value=""]'
    assert pinned.locator(f'[data-subagent-field="effort"] {default}').inner_text() == "Auto (reviews at the top of the chat range)"
    assert pinned.locator('[data-subagent-field="effort"]').input_value() == "high"
    facts = unset.locator(f'[data-subagent-field="effort"] {default}')
    assert facts.inner_text() == "Auto (reviews at the top of the chat range)"
    unset.locator('[data-subagent-field="review_eligible"]').uncheck()
    assert facts.inner_text() == "Auto (chat range)", "an unmarked row runs inside the chat range"
    unset.locator('[data-subagent-field="review_eligible"]').check()
    unset.locator('[data-subagent-field="effort"]').select_option('medium')
    unset.scroll_into_view_if_needed()
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    page.screenshot(path=str(tmp_path / f"effort-reviewer-{width}.png"))
    with page.expect_response('**/api/settings'):
        page.locator('#btn-save-settings').click()
    saved = roles.saved_catalog(ui)["items"]
    assert [row.get("effort") for row in saved[:2]] == ["high", "medium"]
    body = [body for path, body in ui["posts"] if path == "/api/settings"][-1]
    assert "OUROBOROS_EFFORT_TASK" not in body, "the range never rides the generic Settings save"
    with page.expect_response('**/api/review-pool'):
        page.locator('#btn-reload-settings').click()
    assert unset.locator('[data-subagent-field="effort"]').input_value() == 'medium'
    page.locator('[data-settings-tab="behavior"]').click()
    line = page.locator('[data-effort-range-summary]')
    assert line.inner_text() == ("Effort range: Low · High · High (minimum · recommended · maximum). "
                                 "Change it with the round Effort button next to Swarm in the chat.")
    assert page.locator('[data-effort-target^="s-effort-"]').count() == 0, "Behavior has no effort controls left"
    line.scroll_into_view_if_needed()
    page.screenshot(path=str(tmp_path / f"effort-settings-{width}.png"))
