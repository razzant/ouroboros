"""Rendered Settings: reviewer effort round-trips on its own page; Permissions has no effort control."""
import pytest

from tests import test_subscription_role_routes_browser as roles
from tests.settings_browser_helpers import open_settings, press_editor_save
from ouroboros.test_evidence import diagnostic_directory

pytestmark = [pytest.mark.ui_browser]
subscription_ui = roles.subscription_ui
role_ui = roles.role_ui


@pytest.mark.parametrize("width", [1360, 390])
def test_reviewer_effort_auto_caption_and_saved_preference(role_ui, tmp_path, width):
    ui = role_ui
    roles.configure_mixed(ui)
    ui["settings"]["OUROBOROS_SUBAGENTS"]["items"][0]["effort"] = "high"
    ui["settings"]["OUROBOROS_EFFORT_TASK"] = "high"
    ui["page"].set_viewport_size({"width": width, "height": 900})
    pinned = roles.open_agent(ui, "native")
    page = ui["page"]
    # The empty option is Auto: the chat range; a marked row reviews at its top and says so.
    default = 'option[value=""]'
    assert pinned.locator(f'[data-subagent-field="effort"] {default}').inner_text() == "Auto (reviews at the top of the chat range)"
    assert pinned.locator('[data-subagent-field="effort"]').input_value() == "high"
    unset = roles.open_agent(ui, "direct", fresh=False)
    facts = unset.locator(f'[data-subagent-field="effort"] {default}')
    assert facts.inner_text() == "Auto (reviews at the top of the chat range)"
    with page.expect_response("**/api/owner/safety-model"):
        unset.locator('.agent-use:has([data-subagent-field="review_eligible"]) .ui-switch').click()
    page.wait_for_function("() => document.querySelector('[data-subagent-field=effort]')?.options[0].textContent === 'Auto (chat range)'")
    assert facts.inner_text() == "Auto (chat range)", "an unmarked row runs inside the chat range"
    with page.expect_response("**/api/owner/safety-model"):
        unset.locator('.agent-use:has([data-subagent-field="review_eligible"]) .ui-switch').click()
    page.wait_for_function("() => document.querySelector('[data-subagent-field=effort]')?.options[0].textContent.includes('reviews at the top')")
    unset.locator('[data-subagent-field="effort"]').select_option('medium')
    unset.scroll_into_view_if_needed()
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    page.screenshot(path=str(diagnostic_directory(tmp_path, "effort-settings", publish=True) / f"effort-reviewer-{width}.png"))
    assert press_editor_save(page).startswith("Saved")
    saved = roles.saved_catalog(ui)["items"]
    assert [row.get("effort") for row in saved[:2]] == ["high", "medium"]
    body = [body["settings"] for path, body in ui["posts"] if path == "/api/owner/safety-model"][-1]
    assert "OUROBOROS_EFFORT_TASK" not in body, "the range never rides the generic Settings save"
    unset = roles.open_agent(ui, "direct")
    assert unset.locator('[data-subagent-field="effort"]').input_value() == 'medium'
    behavior = open_settings(page, "permissions")
    # The range lives in the composer alone: Behavior has no effort section, line or control.
    assert behavior.locator('h3', has_text="Reasoning Effort").count() == 0
    assert page.locator('[data-effort-range-summary]').count() == 0
    assert page.locator('[data-effort-target^="s-effort-"]').count() == 0, "Behavior has no effort controls left"
    page.screenshot(path=str(diagnostic_directory(tmp_path, "effort-settings", publish=True) / f"effort-settings-{width}.png"))
