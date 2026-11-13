"""The home of Settings in a browser (docs/DESIGN.md "Settings"): every row says its fact or that it was not read,
Ouroboros's card lists what waits for the owner and never a fact nobody read, search finds a row by the words it had
and opens its page at that row, the Ask line leaves the words in Main as a draft, and a skill's form opened from the
Skills page leads back there."""
from __future__ import annotations

import pytest

from tests import test_subscription_setup_browser as setup_browser
from tests.settings_browser_helpers import HOME, home_answer, needs, open_settings, search_settings, settings_page, top_path

subscription_ui = setup_browser.subscription_ui
pytestmark = [pytest.mark.ui_browser]

STARTS = ("/api/model-catalog", "/api/review-pool", "/api/claudexor/wake", "/api/claudexor/status", "/api/mcp/refresh")
WAITING = {"restart_required": True, "restart_keys": ["OUROBOROS_SKILLS_REPO_PATH"], "restart_source_unknown_keys": [],
           "unknown_keys": [], "local_model": {"status": "ready", "pending_keys": [], "unknown_keys": [], "summary": ""}, "summary": ""}


def _paths(ui):
    return [url.split("?")[0].split("://", 1)[1].split("/", 1)[1] for url in ui["reads"]]


@pytest.mark.parametrize("width,height", [(1087, 740), (320, 640)])
def test_the_home_answers_from_passive_reads_and_the_card_lists_what_waits(subscription_ui, width, height):
    ui = subscription_ui
    ui["settings"]["OUROBOROS_SAFETY_OWNER_RULES"] = "Ask before any payment."
    ui["settings"]["_meta"]["restart_state"] = dict(WAITING)
    page = ui["new_page"](viewport={"width": width, "height": height}, has_touch=width < 720)
    ui["reads"].clear()
    page.goto(ui["url"] + "/#settings")
    page.locator(f"{HOME} [data-settings-card]").wait_for(state="visible")
    page.wait_for_function("() => document.querySelector('#page-settings [data-settings-open=rules] [data-settings-answer]').textContent")

    # One fact per row, from what was read; nothing the home shows started a daemon, a catalogue or a poll.
    assert home_answer(page, "models") == "gpt-test"
    assert home_answer(page, "rules") == "Written"
    assert home_answer(page, "agents") == "1 of 1 on"
    assert home_answer(page, "accounts").startswith("Accounts not checked")
    assert home_answer(page, "dashboard:storage") == "Not measured"
    seen = _paths(ui)
    assert not [path for path in seen if any(("/" + path).startswith(start) for start in STARTS)], seen

    # The card: who it is, its state, and what waits. The accounts were not read, so that row is neutral with Check.
    card = page.locator(f"{HOME} [data-settings-card]")
    assert card.locator(".settings-being-name").inner_text().strip() == "Ouroboros"
    assert card.locator("[data-being-state]").inner_text().strip() == "Online"
    assert needs(page) == ["unread:accounts", "restart:restart"]
    assert card.locator("[data-being-count]").inner_text().strip() == "One thing needs you"
    assert "nothing" not in card.inner_text().lower()
    setup_browser.capture(page, f"settings-home-card-{width}x{height}")

    # Check is the owner's press: one read of the accounts. They are all signed in, so the row leaves.
    card.locator('[data-needs="unread"] [data-need-action]').click()
    page.wait_for_function("() => !document.querySelector('#page-settings [data-needs=unread]')")
    assert "api/claudexor/status" in _paths(ui) and "api/claudexor/wake" not in _paths(ui)
    assert home_answer(page, "accounts").startswith("2 accounts")

    # Restart asks first, in the shared confirmation; the row stays until the server says nothing waits.
    restart = card.locator('[data-needs="restart"]')
    assert restart.locator(".ui-row-title").inner_text().strip() == "Restart to apply: Skills folder"
    restart.locator("[data-need-action]").click()
    dialog = page.locator(".confirm-dialog-backdrop")
    dialog.wait_for(state="visible")
    dialog.get_by_role("button", name="Cancel").click()
    dialog.wait_for(state="detached")
    assert "restart:restart" in needs(page)
    ui["settings"]["_meta"]["restart_state"] = {**WAITING, "restart_required": False, "restart_keys": []}
    page.evaluate("() => window.__ouroSettings.open('')")
    page.wait_for_function("() => !document.querySelector('#page-settings [data-needs=restart]')")
    assert not card.locator("[data-being-count]").is_visible(), "nothing waits: the card says who it is and its state"
    page.close()


@pytest.mark.parametrize("width,height", [(1087, 740), (320, 640)])
def test_search_opens_a_row_where_it_lives_and_the_ask_line_leaves_a_draft(subscription_ui, width, height):
    ui = subscription_ui
    page = ui["new_page"](viewport={"width": width, "height": height}, has_touch=width < 720)
    page.goto(ui["url"] + "/#settings")
    page.locator(HOME).wait_for(state="visible")

    # A tab Settings had still leads somewhere; an old label finds the row that replaced it.
    assert "home:permissions" in search_settings(page, "behavior")
    assert search_settings(page, "REGISTRY URL") == ["OUROBOROS_CLAWHUB_REGISTRY_URL"]
    setup_browser.capture(page, f"settings-search-{width}x{height}")
    page.locator(f'{HOME} [data-settings-result="OUROBOROS_CLAWHUB_REGISTRY_URL"]').click()
    level = settings_page(page, "skill-sources")
    level.wait_for(state="visible")
    page.wait_for_function("() => document.activeElement?.id === 'setting-OUROBOROS_CLAWHUB_REGISTRY_URL'")
    assert level.locator(".settings-page-title").inner_text().strip() == "Skill sources"

    # Words no setting has: the list says so, and the last line hands them to Ouroboros.
    open_settings(page, "")
    assert search_settings(page, "never email my clients") == []
    assert "No setting has these words." in page.locator(HOME).inner_text()
    posts = len(ui["posts"])
    page.locator(f"{HOME} [data-settings-ask]").click()
    page.locator("#page-chat").wait_for(state="visible")
    assert page.locator("#page-chat #chat-input").input_value() == "never email my clients"
    assert len(ui["posts"]) == posts, "a draft: nothing was sent and nothing was saved"
    # A draft that is already there stays.
    open_settings(page, "")
    search_settings(page, "quiet")
    page.locator(f"{HOME} [data-settings-ask]").click()
    page.locator("#page-chat").wait_for(state="visible")
    assert page.locator("#page-chat #chat-input").input_value() == "never email my clients"
    page.close()


SKILLS = {"skills": [dict(name=f"gate_skill_{i}", description="An installed skill.", enabled=True, loaded=True, live_loaded=True,
                          type="extension", review_status="pass", review_gate="pass", source="self_authored") for i in range(8)],
          "live": {"settings_sections": [dict(skill="gate_skill_7", section_id="config", key="gate_skill_7:config", title="Gate skill settings",
              render={"components": [{"type": "form", "id": "form", "route": "settings", "submit_label": "Save",
                                      "fields": [{"name": "address", "label": "Address", "type": "text"}]}]})]}}


def test_a_skill_form_opened_from_skills_leads_back_to_skills(subscription_ui):
    ui = subscription_ui
    page = ui["new_page"](viewport={"width": 320, "height": 640}, has_touch=True)
    page.route("**/api/extensions", lambda route: route.fulfill(json=SKILLS))
    page.goto(ui["url"] + "/#skills")
    card = page.locator('.skills-installed-card[data-skill="gate_skill_7"]')
    card.wait_for(state="visible")
    offered = page.locator('#page-skills [role="menuitem"][data-skill-action="open_settings"]').evaluate_all(
        "items => items.map((item) => item.dataset.skill)")
    assert offered == ["gate_skill_7"], "only the skill that registered a form offers its settings"

    def open_form():
        card.locator("[data-skill-menu-trigger]").click()
        # The open menu lies over the page, outside its card.
        page.locator('[role="menuitem"][data-skill-action="open_settings"][data-skill="gate_skill_7"]:visible').click()

    card.locator("[data-skill-menu-trigger]").scroll_into_view_if_needed()
    kept = page.evaluate("() => document.querySelector('#page-skills .skills-scroll').scrollTop")
    assert kept > 0, "the Skills list scrolls at this size, so its place is worth keeping"
    open_form()
    level = settings_page(page, "skill/gate_skill_7")
    level.wait_for(state="visible")
    level.locator("[data-extension-settings-form]").wait_for(state="visible")
    back = level.locator("[data-settings-back]")
    assert back.inner_text().strip() == "Skills" and back.get_attribute("aria-label") == "Back to Skills"
    assert level.locator(".settings-page-title").inner_text().strip() == "Gate skill settings"
    assert page.locator("#page-settings .app-page-header [data-mobile-nav-toggle]").is_visible()
    assert page.locator("#extension-settings-sections").count() == 1
    # Only the opened skill's route was read for values.
    assert [path for path in _paths(ui) if path.startswith("api/extensions/")] == ["api/extensions/gate_skill_7/settings"]
    setup_browser.capture(page, "settings-skill-form-320x640")
    page.keyboard.press("Escape")
    page.locator("#page-skills").wait_for(state="visible")
    assert top_path(page) == ""
    assert page.evaluate("() => document.querySelector('#page-skills .skills-scroll').scrollTop") == kept
    # The first row's Back does the same.
    open_form()
    level.wait_for(state="visible")
    level.locator("[data-settings-back]").click()
    page.locator("#page-skills").wait_for(state="visible")
    page.close()
