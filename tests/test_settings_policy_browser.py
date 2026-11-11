"""Permissions and checks against a real server: Access and review enforcement save themselves, keep the truth about
the restart, and say what Cyber Pro does to a kept Blocking choice."""

from __future__ import annotations

import json
import os
import pathlib

import pytest

from tests.settings_browser_helpers import open_settings, setting_row
from tests.test_ui_smoke_playwright import direct_server_with_data  # noqa: F401

META = "() => document.querySelector('#page-settings [data-setting-row=\"{row}\"] [data-setting-meta]').textContent"


def _meta_says(page, row, words):
    page.wait_for_function(f"words => ({META.format(row=row)})().includes(words)", arg=words)
    return setting_row(page, row).locator("[data-setting-meta]").inner_text()


def _choose(page, row, value):
    """Press a strip's option and wait until the row holds it (a second save may say the same words as the first)."""
    setting_row(page, row).locator(f'[data-setting-value="{value}"]').click()
    page.wait_for_function("""([row, value]) => {
        const option = document.querySelector(`#page-settings [data-setting-row="${row}"] [data-setting-value="${value}"]`);
        return option.getAttribute('aria-pressed') === 'true' && !option.hasAttribute('aria-busy');
    }""", arg=[row, value])


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
            assert said == "Running: Light · after restart: Cyber Pro", said
            assert access.locator("[data-setting-result]").inner_text().startswith("Saved")
            _choose(page, "OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
            stored = json.loads((fixture["data_dir"] / "settings.json").read_text(encoding="utf-8"))
            assert stored["OUROBOROS_RUNTIME_MODE"] == "cyber_pro"
            assert stored["OUROBOROS_REVIEW_ENFORCEMENT"] == "blocking"

            # A reload keeps both facts: the saved level is chosen, the level in force is named beside it.
            _open_permissions(page, fixture["url"])
            access = setting_row(page, "OUROBOROS_RUNTIME_MODE")
            assert access.locator('[role="radio"][aria-checked="true"]').get_attribute("data-setting-value") == "cyber_pro"
            assert _meta_says(page, "OUROBOROS_RUNTIME_MODE", "Running:") == "Running: Light · after restart: Cyber Pro"
            marks = dict(zip(access.locator(".settings-choice").evaluate_all("nodes => nodes.map((n) => n.dataset.settingValue)"),
                             access.locator("[data-setting-mark]").all_inner_texts()))
            assert marks == {"light": "Now", "advanced": "", "pro": "", "cyber_pro": "After restart"}
            enforcement = setting_row(page, "OUROBOROS_REVIEW_ENFORCEMENT")
            assert enforcement.locator('[data-setting-value="blocking"]').get_attribute("aria-pressed") == "true"
            # Blocking is enforced until the restart that applies Cyber Pro: the row says both halves.
            assert _meta_says(page, "OUROBOROS_REVIEW_ENFORCEMENT", "Enforced now") == \
                "Enforced now. Not enforced after the restart that applies Cyber Pro."
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
            page.locator('[data-review-mode="blocking"]').click()
            assert page.locator('[data-runtime-mode="cyber_pro"]').get_attribute("aria-pressed") == "true"
            assert page.locator('[data-review-mode="blocking"]').get_attribute("aria-pressed") == "true"
            page.screenshot(path=str(evidence / "onboarding-cyber-mobile.png"), full_page=True)
        finally:
            browser.close()


@pytest.mark.ui_browser
def test_cyber_pro_names_a_kept_blocking_choice_beside_its_control(direct_server_with_data):  # noqa: F811
    """#1736: the Blocking choice is kept and its Cyber Pro effect is said, for now and after restart."""
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
                enforcement = setting_row(page, "OUROBOROS_REVIEW_ENFORCEMENT")
                blocking = enforcement.locator('[data-setting-value="blocking"]')
                assert blocking.get_attribute("aria-pressed") == "true"  # the owner's choice is kept
                assert _meta_says(page, "OUROBOROS_REVIEW_ENFORCEMENT", "Cyber Pro") == \
                    "Not enforced under Cyber Pro: reviews advise, it decides."
                # The level itself says what it does to reviews, in its own line.
                assert access.locator('[data-setting-value="cyber_pro"] .settings-choice-note').inner_text() == \
                    "Full say over itself and this computer. Reviews advise, it decides."
                assert "nothing blocks" not in page.locator('[data-settings-page="permissions"]').inner_text()
                enforcement.scroll_into_view_if_needed()
                page.screenshot(path=str(evidence / f"settings-cyber-blocking-{theme}-{width}.png"))
                # A lower Access saved for the next start is a distinct fact from the level in force: lowering asks
                # nothing, writes at once, and the Blocking row says both halves.
                pro = access.locator('[data-setting-value="pro"]')
                pro.focus()
                page.keyboard.press("Enter")
                assert _meta_says(page, "OUROBOROS_RUNTIME_MODE", "after restart") == "Running: Cyber Pro · after restart: Pro"
                assert not page.locator('[data-confirm-ok]').is_visible()  # lowering Access is never asked about
                assert _meta_says(page, "OUROBOROS_REVIEW_ENFORCEMENT", "Enforced after") == \
                    "Not enforced now, under Cyber Pro. Enforced after the restart that applies Pro."
                page.wait_for_function("() => document.activeElement?.dataset.settingValue === 'pro'")  # focus stays put
                assert json.loads(settings_path.read_text(encoding="utf-8"))["OUROBOROS_RUNTIME_MODE"] == "pro"
                assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
                enforcement.scroll_into_view_if_needed()
                page.screenshot(path=str(evidence / f"settings-cyber-blocking-saved-pro-{theme}-{width}.png"))
                # Advisory has nothing to qualify.
                _choose(page, "OUROBOROS_REVIEW_ENFORCEMENT", "advisory")
                assert _meta_says(page, "OUROBOROS_REVIEW_ENFORCEMENT", "decides") == "It reads the finding and decides."
                # Back to the state this loop starts from: raising Access asks first.
                _choose(page, "OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
                access.locator('[data-setting-value="cyber_pro"]').click()
                page.locator('[data-confirm-ok]').click()
                _meta_says(page, "OUROBOROS_RUNTIME_MODE", "How far it reaches")  # nothing waits for a restart any more
                page.close()
        finally:
            browser.close()
    stored = json.loads(settings_path.read_text(encoding="utf-8"))
    assert (stored["OUROBOROS_RUNTIME_MODE"], stored["OUROBOROS_REVIEW_ENFORCEMENT"]) == ("cyber_pro", "blocking")
