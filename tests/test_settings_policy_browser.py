"""Permissions and checks against a real server: Access and the result review save themselves, keep the truth about
the restart, and the Access levels say what each does to reviews."""

from __future__ import annotations

import json
import os
import pathlib

import pytest

from tests.settings_browser_helpers import commit_row, open_settings, setting_row
from tests.test_ui_smoke_playwright import direct_server_with_data  # noqa: F401

META = "() => document.querySelector('#page-settings [data-setting-row=\"{row}\"] [data-setting-meta]').textContent"


def _meta_says(page, row, words):
    page.wait_for_function(f"words => ({META.format(row=row)})().includes(words)", arg=words)
    return setting_row(page, row).locator("[data-setting-meta]").inner_text()


def _open_permissions(page, url):
    page.goto(url + "/#settings", wait_until="domcontentloaded")
    level = open_settings(page, "permissions")
    # The page reads the settings when it is shown: the chosen Access level is drawn from that read.
    level.locator('[data-setting-row="OUROBOROS_RUNTIME_MODE"] [role="radio"][aria-checked="true"]').wait_for(state="visible")
    return level


@pytest.mark.ui_browser
def test_access_and_review_round_trip_without_losing_restart_truth(direct_server_with_data):  # noqa: F811
    from playwright.sync_api import sync_playwright

    fixture = direct_server_with_data
    evidence = pathlib.Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", str(fixture["data_dir"].parent)))
    evidence.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1400, "height": 1000})
        try:
            _open_permissions(page, fixture["url"])
            access = setting_row(page, "OUROBOROS_RUNTIME_MODE")
            # Raising Access asks first, then writes at once through its owner endpoint: no page-wide Save.
            access.locator('[data-setting-value="cyber_pro"]').click()
            page.locator('[data-confirm-ok]').click()
            said = _meta_says(page, "OUROBOROS_RUNTIME_MODE", "after restart: Cyber Pro")
            assert said == "Running: Light · after restart: Cyber Pro", said  # the level in force is the server's own answer
            running = page.evaluate("async () => (await (await fetch('/api/settings')).json())._meta.restart_state.running")
            assert running["OUROBOROS_RUNTIME_MODE"] == "light", running
            # The server's receipt and timing, word for word: a restart row says no "was" (its meta says both values)
            # and Access offers no Undo (choosing again may ask); a delayed strip says what it replaced, with Undo.
            assert access.locator("[data-setting-result]").inner_text() == "Saved · applies after restart"
            assert commit_row(page, "OUROBOROS_TASK_REVIEW_MODE", "required") == "Saved · applies to new tasks · was Auto · Undo"
            stored = json.loads((fixture["data_dir"] / "settings.json").read_text(encoding="utf-8"))
            assert stored["OUROBOROS_RUNTIME_MODE"] == "cyber_pro"
            assert stored["OUROBOROS_TASK_REVIEW_MODE"] == "required"

            # A reload keeps both facts: the saved level is chosen, the level in force is named beside it.
            _open_permissions(page, fixture["url"])
            access = setting_row(page, "OUROBOROS_RUNTIME_MODE")
            assert access.locator('[role="radio"][aria-checked="true"]').get_attribute("data-setting-value") == "cyber_pro"
            assert _meta_says(page, "OUROBOROS_RUNTIME_MODE", "Running:") == "Running: Light · after restart: Cyber Pro"
            marks = dict(zip(access.locator(".settings-choice").evaluate_all("nodes => nodes.map((n) => n.dataset.settingValue)"),
                             access.locator("[data-setting-mark]").all_inner_texts()))
            assert marks == {"light": "Now", "advanced": "", "pro": "", "cyber_pro": "After restart"}
            review = setting_row(page, "OUROBOROS_TASK_REVIEW_MODE")
            assert review.locator('[data-setting-value="required"]').get_attribute("aria-pressed") == "true"
            assert page.locator('[data-setting-row="OUROBOROS_REVIEW_ENFORCEMENT"]').count() == 0  # the setting is retired
            access.scroll_into_view_if_needed()
            page.screenshot(path=str(evidence / "settings-policy-reload-desktop.png"))
        finally:
            browser.close()


@pytest.mark.ui_browser
def test_mobile_onboarding_keeps_all_four_access_choices_usable(direct_server_with_data):  # noqa: F811
    from playwright.sync_api import sync_playwright

    fixture = direct_server_with_data
    evidence = pathlib.Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", str(fixture["data_dir"].parent)))
    evidence.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 390, "height": 844})
        try:
            page.goto(fixture["url"] + "/onboarding", wait_until="domcontentloaded")
            page.locator("#next-btn").click()
            for field in page.locator("[data-model-role-model]").all():
                if not field.input_value():
                    field.fill("mock-model")
            page.locator("#next-btn").click()
            choices = page.locator("[data-runtime-mode]")
            choices.last.wait_for(state="visible")
            assert choices.count() == 4
            geometry = choices.evaluate_all("nodes => nodes.map(n => {const r=n.getBoundingClientRect();return {x:r.x,y:r.y,w:r.width};})")
            assert max(row["x"] for row in geometry) - min(row["x"] for row in geometry) < 2
            assert all(row["w"] > 250 for row in geometry), geometry
            page.locator('[data-runtime-mode="cyber_pro"]').click()
            assert page.locator('[data-runtime-mode="cyber_pro"]').get_attribute("aria-pressed") == "true"
            assert page.locator("[data-review-mode]").count() == 0  # the step chooses Access only
            page.screenshot(path=str(evidence / "onboarding-cyber-mobile.png"), full_page=True)
        finally:
            browser.close()


@pytest.mark.ui_browser
def test_cyber_pro_says_what_it_does_to_reviews_and_lowering_it_asks_nothing(direct_server_with_data):  # noqa: F811
    """A leftover Blocking value in the file draws nothing; the Cyber Pro level says what it does to reviews in its own
    line, and a lower Access saved for the next start is a distinct fact from the level in force."""
    from playwright.sync_api import sync_playwright

    fixture = direct_server_with_data
    evidence = pathlib.Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", str(fixture["data_dir"].parent)))
    evidence.mkdir(parents=True, exist_ok=True)
    settings_path = fixture["data_dir"] / "settings.json"
    fixture["stop_server"]()
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    settings.update(OUROBOROS_RUNTIME_MODE="cyber_pro", OUROBOROS_REVIEW_ENFORCEMENT="blocking")
    settings_path.write_text(json.dumps(settings), encoding="utf-8")
    fixture["start_server"]()  # this process boots under Cyber Pro
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            for theme, width in (("light", 1400), ("dark", 1400), ("dark", 390)):
                page = browser.new_page(viewport={"width": width, "height": 1000})
                page.add_init_script(f"localStorage.setItem('ouroboros.theme', {json.dumps(theme)})")
                _open_permissions(page, fixture["url"])
                assert page.evaluate("document.documentElement.dataset.theme") == theme
                access = setting_row(page, "OUROBOROS_RUNTIME_MODE")
                assert page.locator('[data-setting-row="OUROBOROS_REVIEW_ENFORCEMENT"]').count() == 0
                # The level itself says what it does to reviews, in its own line.
                assert access.locator('[data-setting-value="cyber_pro"] .settings-choice-note').inner_text() == \
                    "Full say over itself and this computer. Asks for reviews when it judges."
                assert access.locator('[data-setting-value="pro"] .settings-choice-note').inner_text() == \
                    "May change its protected core. Reads a review before each change."
                assert "nothing blocks" not in page.locator('[data-settings-page="permissions"]').inner_text()
                access.scroll_into_view_if_needed()
                page.screenshot(path=str(evidence / f"settings-cyber-{theme}-{width}.png"))
                # A lower Access saved for the next start is a distinct fact from the level in force: lowering asks
                # nothing and writes at once.
                pro = access.locator('[data-setting-value="pro"]')
                pro.focus()
                page.keyboard.press("Enter")
                assert _meta_says(page, "OUROBOROS_RUNTIME_MODE", "after restart") == "Running: Cyber Pro · after restart: Pro"
                assert access.locator("[data-setting-result]").inner_text() == "Saved · applies after restart"
                assert not page.locator('[data-confirm-ok]').is_visible()  # lowering Access is never asked about
                page.wait_for_function("() => document.activeElement?.dataset.settingValue === 'pro'")  # focus stays put
                assert json.loads(settings_path.read_text(encoding="utf-8"))["OUROBOROS_RUNTIME_MODE"] == "pro"
                assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
                page.screenshot(path=str(evidence / f"settings-cyber-saved-pro-{theme}-{width}.png"))
                # Back to the state this loop starts from: raising Access asks first.
                access.locator('[data-setting-value="cyber_pro"]').click()
                page.locator('[data-confirm-ok]').click()
                _meta_says(page, "OUROBOROS_RUNTIME_MODE", "How far it reaches")  # nothing waits for a restart any more
                page.close()
        finally:
            browser.close()
    stored = json.loads(settings_path.read_text(encoding="utf-8"))
    assert stored["OUROBOROS_RUNTIME_MODE"] == "cyber_pro"
