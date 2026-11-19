"""Rendered Settings: an agent's effort select says Auto on its own page, and no Settings page has an effort section."""
import pytest

from tests import test_subscription_role_routes_browser as roles
from tests.settings_browser_helpers import open_entry, open_settings, press_editor_save
from ouroboros.test_evidence import diagnostic_directory

pytestmark = [pytest.mark.ui_browser]
subscription_ui = roles.subscription_ui
role_ui = roles.role_ui
AUTO = 'option[value=""]'


@pytest.mark.parametrize("width", [1360, 390])
def test_reviewer_effort_auto_caption_and_saved_preference(role_ui, tmp_path, width):
    from playwright.sync_api import expect
    ui = role_ui
    roles.configure_mixed(ui)
    ui["settings"]["OUROBOROS_SUBAGENTS"]["items"][0]["effort"] = "high"
    ui["settings"]["OUROBOROS_EFFORT_TASK"] = "high"
    ui["page"].set_viewport_size({"width": width, "height": 900})
    page = ui["page"]
    # The empty option is Auto: the chat range; a marked agent reviews at its top and says so.
    pinned = roles.open_agent(ui, "native")
    assert pinned.locator(f'[data-subagent-field="effort"] {AUTO}').inner_text() == "Auto (reviews at the top of the chat range)"
    assert pinned.locator('[data-subagent-field="effort"]').input_value() == "high"
    unset = roles.open_agent(ui, "direct", fresh=False)
    facts = unset.locator(f'[data-subagent-field="effort"] {AUTO}')
    assert facts.inner_text() == "Auto (reviews at the top of the chat range)"
    reviewer = unset.locator('[data-subagent-field="review_eligible"]')
    # The Reviewer mark is a switch: a hand presses its track (the input lies under it).
    reviewer.locator("xpath=..").click()
    assert not reviewer.is_checked()
    expect(facts).to_have_text("Auto (chat range)")  # an unmarked agent runs inside the chat range
    reviewer.locator("xpath=..").click()
    assert reviewer.is_checked()
    expect(facts).to_have_text("Auto (reviews at the top of the chat range)")
    unset.locator('[data-subagent-field="effort"]').select_option('medium')
    unset.scroll_into_view_if_needed()
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    page.screenshot(path=str(diagnostic_directory(tmp_path, "effort-settings", publish=True) / f"effort-reviewer-{width}.png"))
    assert press_editor_save(page).startswith("Saved")
    saved = roles.saved_catalog(ui)["items"]
    assert [row.get("effort") for row in saved[:2]] == ["high", "medium"]
    assert all("OUROBOROS_EFFORT_TASK" not in body for body in roles.settings_saves(ui)), \
        "the range never rides an agent's Save"
    # A fresh page reads the saved roster back.
    page.reload()
    open_settings(page, "agents").locator("[data-entry-id]").first.wait_for(state="visible")
    reread = open_entry(page, "direct").locator("[data-subagent-row]")
    assert reread.locator('[data-subagent-field="effort"]').input_value() == 'medium'
    # The range lives in the composer alone: no Settings page has an effort section, line or control.
    for path in ("permissions", "rules", "models", "computer/advanced"):
        level = open_settings(page, path)
        assert level.locator('h3', has_text="Reasoning Effort").count() == 0, path
        assert level.locator('[data-effort-range-summary]').count() == 0, path
        assert level.locator('[data-effort-target^="s-effort-"]').count() == 0, f"{path} has an effort control"
    open_settings(page, "permissions")
    page.screenshot(path=str(diagnostic_directory(tmp_path, "effort-settings", publish=True) / f"effort-settings-{width}.png"))
