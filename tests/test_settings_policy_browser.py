"""Real Settings save/reload proof for the three independent policy controls."""

from __future__ import annotations

import json
import os
import pathlib

import pytest

from tests.test_ui_smoke_playwright import direct_server_with_data  # noqa: F401


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
            def open_behavior():
                page.goto(fixture["url"], wait_until="domcontentloaded")
                page.locator('[data-nav-page="settings"]').click()
                page.locator('[data-settings-tab="behavior"]').click()
                page.wait_for_function("() => document.querySelector('#btn-save-settings')?.disabled === false")

            open_behavior()
            page.locator('[data-runtime-mode-group] [data-effort-value="cyber_pro"]').click()
            page.locator('[data-enforcement-group] [data-effort-value="blocking"]').click()
            assert page.locator('#s-runtime-mode').input_value() == "cyber_pro"
            assert page.locator('#s-review-enforcement').input_value() == "blocking"
            assert page.locator('[data-enforcement-group] [data-effort-value="blocking"]').is_enabled()
            page.locator('#btn-save-settings').click()
            page.locator('[data-confirm-ok]').click()
            page.wait_for_function("() => !document.querySelector('#btn-save-settings').disabled && (document.querySelector('#settings-status').textContent || '').includes('restart required')")
            stored = json.loads((fixture["data_dir"] / "settings.json").read_text())
            assert stored["OUROBOROS_RUNTIME_MODE"] == "cyber_pro"
            assert stored["OUROBOROS_REVIEW_ENFORCEMENT"] == "blocking"

            open_behavior()
            assert page.locator('#s-runtime-mode').input_value() == "cyber_pro"
            assert page.locator('#s-review-enforcement').input_value() == "blocking"
            access = page.locator('[data-policy-state="access"]')
            assert "Saved: Cyber Pro" in access.inner_text()
            assert "Current process: Light" in access.inner_text()
            assert "After restart: Cyber Pro" in access.inner_text()
            assert "Next task" not in access.inner_text()
            assert "Restart required" in access.inner_text()
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
    settings = json.loads(settings_path.read_text())
    settings.update(OUROBOROS_RUNTIME_MODE="cyber_pro", OUROBOROS_REVIEW_ENFORCEMENT="blocking")
    settings_path.write_text(json.dumps(settings))
    fixture["start_server"]()  # this process boots under Cyber Pro
    saved = settings_path.read_bytes()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            for theme, width in (("light", 1400), ("dark", 1400), ("dark", 390)):
                page = browser.new_page(viewport={"width": width, "height": 1000})
                page.add_init_script(f"localStorage.setItem('ouroboros.theme', {json.dumps(theme)})")
                page.goto(fixture["url"] + "/#settings", wait_until="domcontentloaded")
                page.locator('[data-settings-tab="behavior"]').click()
                page.wait_for_function("() => document.querySelector('#btn-save-settings')?.disabled === false")
                assert page.evaluate("document.documentElement.dataset.theme") == theme
                note = page.locator('[data-policy-state="review"]')
                blocking = page.locator('[data-enforcement-group] [data-effort-value="blocking"]')
                assert blocking.get_attribute("aria-pressed") == "true"  # the owner's choice is kept
                assert "Blocking now and for new tasks" in note.inner_text()  # one value, said once
                assert "Cyber Pro: Blocking is not enforced; review findings do not block actions" in note.inner_text()
                access_copy = page.locator('[data-settings-panel="behavior"] .form-section', has=page.locator("#s-runtime-mode"))
                copy = access_copy.text_content()  # the section's copy beyond its lead is behind its Details
                assert "review findings do not block actions in Cyber Pro" in copy
                assert "nothing blocks" not in copy and "stays available" not in copy
                note.scroll_into_view_if_needed()
                page.screenshot(path=str(evidence / f"settings-cyber-blocking-{theme}-{width}.png"))
                # A drafted Access for the next start is a distinct fact from the current process.
                pro = page.locator('[data-runtime-mode-group] [data-effort-value="pro"]')
                pro.focus()
                page.keyboard.press("Enter")
                page.wait_for_function("() => document.querySelector('[data-policy-state=\"review\"]').textContent.includes('selected Pro')")
                assert ("Cyber Pro: Blocking is not enforced now; selected Blocking would be enforced only after a restart "
                        "under the selected Pro") in note.inner_text()
                assert page.evaluate("document.activeElement?.dataset.effortValue") == "pro"  # focus stays put
                assert page.locator('#s-runtime-mode').input_value() == "pro"
                assert not page.locator('[data-confirm-ok]').is_visible()  # nothing is confirmed or saved
                assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
                access_note = page.locator('[data-policy-state="access"]')
                assert "Cyber Pro now and after restart" in access_note.inner_text()
                note.scroll_into_view_if_needed()
                page.screenshot(path=str(evidence / f"settings-cyber-blocking-draft-pro-{theme}-{width}.png"))
                access_note.scroll_into_view_if_needed()
                page.screenshot(path=str(evidence / f"settings-cyber-access-draft-pro-{theme}-{width}.png"))
                page.locator('[data-enforcement-group] [data-effort-value="advisory"]').click()
                page.wait_for_function("() => !document.querySelector('[data-policy-state=\"review\"]').textContent.includes('selected Blocking')")
                assert note.inner_text().endswith("Cyber Pro: Blocking is not enforced now")  # the current fact stays
                page.close()
        finally:
            browser.close()
    assert settings_path.read_bytes() == saved, "the check saved nothing"
