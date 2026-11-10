"""The real SPA under a deployment policy, with the same server unmanaged as its positive control.

An employee sees what the organization fixes, still changes what it allows, finds no editor for what it fixes and
no door to replace the serving program, and keeps Panic. Frames go to ``OUROBOROS_UI_EVIDENCE_DIR`` for a look.
"""

from __future__ import annotations

import json
import os
import pathlib

import pytest

from tests.test_ui_smoke_playwright import direct_server_with_data  # noqa: F401

POLICY = {"version": 1, "access": {"max": "pro"},
          "safety": {"min_coverage": "full", "checker": {"model": "anthropic@safety::claude-opus-5-5"}},
          "rules": "Customer records stay inside the corporate network."}
CHECKER_ROW = {"kind": "anthropic", "name": "safety", "label": "Organization checker", "api_key": "sk-ant-fixture-organization"}


def _open(page, url, theme, tab):
    page.add_init_script(f"localStorage.setItem('ouroboros.theme', {json.dumps(theme)})")
    page.goto(url + "/#settings", wait_until="domcontentloaded")
    page.wait_for_function("() => document.querySelector('#btn-save-settings')?.disabled === false")
    page.locator(f'[data-settings-tab="{tab}"]').click()


def _shot(page, evidence, name, locator=None):
    target = locator or page
    target.screenshot(path=str(evidence / f"{name}.png"))


def _builtin_row(page, card_id):
    """Open one API key card as the owner would; its built-in ("Default") row."""
    card = page.locator(f'[data-provider-card="{card_id}"]')
    toggle = card.locator('[data-provider-toggle]')
    if toggle.get_attribute("aria-expanded") != "true":
        toggle.click()
    return card.locator('[data-connection-row][data-builtin="1"]')


@pytest.mark.ui_browser
def test_managed_settings_and_updates_say_what_the_organization_fixes(direct_server_with_data, tmp_path):  # noqa: F811
    from playwright.sync_api import sync_playwright

    fixture = direct_server_with_data
    evidence = pathlib.Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", str(tmp_path / "frames")))
    evidence.mkdir(parents=True, exist_ok=True)
    settings_path = fixture["data_dir"] / "settings.json"
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            # --- unmanaged: the positive control, today's behaviour, in both themes ---
            for theme, width in (("light", 1400), ("dark", 1400), ("dark", 390)):
                page = browser.new_page(viewport={"width": width, "height": 1000 if width > 500 else 844})
                _open(page, fixture["url"], theme, "behavior")
                assert page.evaluate("document.documentElement.dataset.theme") == theme
                assert page.locator('[data-runtime-mode-group] [data-effort-value]').evaluate_all(
                    "nodes => nodes.map(n => n.dataset.effortValue)") == ["light", "advanced", "pro", "cyber_pro"]
                assert page.locator('[data-effort-target="s-safety-mode"]').is_visible()
                assert page.locator('[data-effort-target="s-update-channel"]').is_visible()
                assert not page.locator('[data-update-channel-fixed]').is_visible()
                assert "Change in Models" in page.locator('[data-safety-checker]').inner_text()
                assert not page.locator('[data-safety-exits]').is_visible(), "personal OSS keeps both exits, unsaid"
                page.locator('[data-safety-checker]').scroll_into_view_if_needed()
                _shot(page, evidence, f"unmanaged-behavior-{theme}-{width}")
                page.locator('[data-settings-tab="models"]').click()
                assert page.locator('#s-model-safety').count() == 1
                page.locator('[data-settings-tab="providers"]').click()
                openai = _builtin_row(page, "openai")
                assert not page.locator('#s-openai').evaluate("node => node.readOnly")
                assert openai.locator('[data-connection-toggle]').is_visible()
                assert not openai.locator('[data-connection-fixed-fields]').is_visible()
                assert not page.locator('#s-openai-base-url').evaluate("node => node.readOnly")
                assert not page.locator('[data-legacy-base-url-fixed]').is_visible()
                openai.scroll_into_view_if_needed()
                _shot(page, evidence, f"unmanaged-accounts-{theme}-{width}")
                page.close()

            # --- the same server, started under the administrator's policy file ---
            fixture["stop_server"]()
            stored = json.loads(settings_path.read_text())
            stored["OUROBOROS_CONNECTIONS"] = [CHECKER_ROW]
            settings_path.write_text(json.dumps(stored))
            policy_path = tmp_path / "admin" / "deployment-policy.json"
            policy_path.parent.mkdir(exist_ok=True)
            policy_path.write_text(json.dumps(POLICY))
            fixture["env"]["OUROBOROS_DEPLOYMENT_POLICY"] = str(policy_path)
            fixture["start_server"]()

            for theme, width in (("light", 1400), ("dark", 1400), ("dark", 390)):
                page = browser.new_page(viewport={"width": width, "height": 1000 if width > 500 else 844})
                _open(page, fixture["url"], theme, "behavior")
                assert page.evaluate("document.documentElement.dataset.theme") == theme
                access = page.locator('[data-runtime-mode-group] [data-effort-value]')
                assert access.evaluate_all("nodes => nodes.map(n => n.dataset.effortValue)") == ["light", "advanced", "pro"]
                assert "Your organization allows up to Pro" in page.locator('[data-policy-state="access"]').inner_text()
                assert not page.locator('[data-effort-target="s-safety-mode"]').is_visible()
                assert not page.locator('[data-effort-target="s-consciousness-safety-mode"]').is_visible()
                assert page.locator('[data-policy-state="supervisor"]').inner_text() == "Set by your organization: Full"
                checker = page.locator('[data-safety-checker]').inner_text()
                assert "Set by your organization" in checker and "Change in Models" not in checker
                assert "decides alone" in page.locator('[data-safety-second-check]').inner_text()
                assert page.locator('[data-safety-exits]').inner_text() == (  # unnamed under the organization's checker
                    "Set by your organization: your approval cannot send a message the Supervisor has not checked.")
                assert page.locator('[data-safety-organization-rules]').is_visible()
                assert page.locator('#s-safety-rules').is_enabled(), "the employee's own rules stay editable"
                assert page.locator('[data-update-channel-fixed]').is_visible()
                assert not page.locator('[data-effort-target="s-update-channel"]').is_visible()
                assert page.locator('[data-chat-command="panic"]').count() >= 1 or width < 500
                assert page.evaluate("document.querySelector('[data-settings-panel=\"behavior\"]').scrollWidth <= innerWidth + 1")
                page.locator('[data-runtime-mode-group]').scroll_into_view_if_needed()
                _shot(page, evidence, f"managed-access-{theme}-{width}")
                page.locator('[data-safety-checker]').scroll_into_view_if_needed()
                _shot(page, evidence, f"managed-safety-{theme}-{width}")
                if width == 1400 and theme == "light":
                    # Keyboard: the capped row is the only stop set; no hidden Cyber Pro takes focus.
                    access.first.focus()
                    stops = [page.evaluate("document.activeElement.dataset.effortValue || ''")]
                    for _ in range(3):
                        page.keyboard.press("Tab")
                        stops.append(page.evaluate("document.activeElement.dataset.effortValue || ''"))
                    assert stops[:3] == ["light", "advanced", "pro"] and stops[3] != "cyber_pro", stops
                    # A change the policy allows still saves; the raw shadowed Access stays the employee's.
                    page.locator('[data-enforcement-group] [data-effort-value="blocking"]').click()
                    page.locator('#btn-save-settings').click()
                    page.wait_for_function("() => (document.querySelector('#settings-status')?.textContent || '').match(/[Ss]aved/)")
                    saved = json.loads(settings_path.read_text())
                    assert saved["OUROBOROS_REVIEW_ENFORCEMENT"] == "blocking"
                    assert saved["OUROBOROS_RUNTIME_MODE"] == "light"
                    assert "OUROBOROS_SAFETY_MODE" not in saved and "OUROBOROS_MODEL_SAFETY" not in saved
                    page.locator('[data-settings-tab="models"]').click()
                    fixed = page.locator('[data-model-role-group="safety"][data-model-role-fixed]')
                    assert fixed.is_visible() and "Set by your organization" in fixed.inner_text()
                    assert page.locator('#s-model-safety').count() == 0
                    fixed.scroll_into_view_if_needed()
                    _shot(page, evidence, "managed-models-light-1400")
                    page.locator('[data-settings-tab="providers"]').click()
                    row = page.locator('[data-connection-row][data-kind="anthropic"][data-name="safety"]')
                    row.scroll_into_view_if_needed()
                    assert row.locator('[data-connection-fixed]').is_visible()
                    assert not row.locator('[data-connection-toggle]').is_visible()
                    assert not row.locator('[data-connection-remove]').is_visible()
                    assert row.locator('[data-connection-label]').evaluate("node => node.readOnly")
                    _shot(page, evidence, "managed-accounts-light-1400", row)
                    page.locator('[data-nav-page="dashboard"]').click()
                    page.locator('[data-dashboard-tab="updates"]').click()
                    page.wait_for_function("() => (document.querySelector('#updates-summary')?.textContent || '') === 'Managed by your organization'")
                    assert not page.locator('.updates-recovery').is_visible()
                    assert not page.locator('#btn-update-replace').is_visible()
                    _shot(page, evidence, "managed-updates-light-1400")
                page.close()
            # The wizard draws the same contract: the Supervisor fixed in Models, Access capped with its bound.
            page = browser.new_page(viewport={"width": 390, "height": 844})
            page.add_init_script("localStorage.setItem('ouroboros.theme', 'dark')")
            page.goto(fixture["url"] + "/onboarding", wait_until="domcontentloaded")
            page.locator("#next-btn").click()
            fixed = page.locator('[data-model-role-group="safety"][data-model-role-fixed]')
            fixed.wait_for(state="visible")
            assert "Set by your organization" in fixed.inner_text() and page.locator("#safety-model").count() == 0
            fixed.scroll_into_view_if_needed()
            _shot(page, evidence, "managed-wizard-models-dark-390")
            for field in page.locator("[data-model-role-model]").all():
                if not field.input_value():
                    field.fill("mock-model")
            page.locator("#next-btn").click()
            choices = page.locator("[data-runtime-mode]")
            choices.last.wait_for(state="visible")
            assert choices.evaluate_all("nodes => nodes.map(n => n.dataset.runtimeMode)") == ["light", "advanced", "pro"]
            assert page.locator("[data-access-bound]").inner_text() == "Your organization allows up to Pro"
            page.locator(".runtime-mode-card").scroll_into_view_if_needed()
            _shot(page, evidence, "managed-wizard-access-dark-390")
            page.close()
            # Nothing the browser did wrote a fixed value or touched the organization's connection.
            final = json.loads(settings_path.read_text())
            assert final["OUROBOROS_CONNECTIONS"][0]["api_key"] == CHECKER_ROW["api_key"]

            # --- the legacy source branch: a built-in compatible checker reaching OPENAI_BASE_URL + OPENAI_API_KEY ---
            fixture["stop_server"]()
            stored = json.loads(settings_path.read_text())
            stored["OPENAI_BASE_URL"] = stored.pop("OPENAI_COMPATIBLE_BASE_URL")
            stored["OPENAI_API_KEY"] = stored.pop("OPENAI_COMPATIBLE_API_KEY")
            settings_path.write_text(json.dumps(stored))
            legacy = {key: stored[key] for key in ("OPENAI_BASE_URL", "OPENAI_API_KEY")}
            policy_path.write_text(json.dumps({"version": 1, "safety": {"checker": {"model": "openai-compatible::mock-model"},
                                                                        "allowed_exits": []}}))
            fixture["start_server"]()
            for theme in ("light", "dark"):
                page = browser.new_page(viewport={"width": 1400, "height": 1000})
                _open(page, fixture["url"], theme, "behavior")  # the strict file closes all three Safety exits
                exits = page.locator('[data-safety-exits]')
                assert exits.inner_text() == (
                    "Set by your organization: Ouroboros cannot override a DANGEROUS refusal on its own; "
                    "your approval cannot override a DANGEROUS refusal; "
                    "your approval cannot send a message the Supervisor has not checked.")
                exits.scroll_into_view_if_needed()
                _shot(page, evidence, f"managed-strict-safety-{theme}-1400")
                page.locator('[data-settings-tab="providers"]').click()
                openai = _builtin_row(page, "openai")
                assert page.locator('#s-openai').evaluate("node => node.readOnly")
                assert openai.locator('[data-connection-fixed-fields]').inner_text() == (
                    "API key: set by your organization, its Safety Supervisor uses it.")
                assert not openai.locator('[data-connection-fixed]').is_visible(), "the OpenAI key itself is not the checker's"
                assert openai.locator('[data-connection-toggle]').is_visible(), "its Disable stays the employee's"
                assert not openai.locator('[data-connection-remove]').is_visible()
                assert not openai.locator('.secret-clear').is_visible()
                openai.scroll_into_view_if_needed()
                _shot(page, evidence, f"managed-legacy-openai-{theme}-1400", openai)
                compatible = _builtin_row(page, "compatible")
                assert compatible.locator('[data-connection-fixed]').is_visible()
                assert not compatible.locator('[data-connection-toggle]').is_visible()
                assert "change the address" not in compatible.locator('[data-connection-meta]').inner_text()
                compatible.scroll_into_view_if_needed()
                _shot(page, evidence, f"managed-legacy-compatible-{theme}-1400", compatible)
                address = page.locator('#s-openai-base-url')
                assert address.evaluate("node => node.readOnly") and address.input_value() == legacy["OPENAI_BASE_URL"]
                note = page.locator('[data-legacy-base-url-fixed]')
                assert note.is_visible() and note.inner_text().startswith("Legacy OpenAI Base URL: set by your organization")
                address.scroll_into_view_if_needed()
                _shot(page, evidence, f"managed-legacy-address-{theme}-1400")
                if theme == "light":
                    # A permitted save: the employee turns their own OpenAI key off; the checker's pair stays as deployed.
                    openai.locator('[data-connection-toggle]').click()
                    page.locator('#btn-save-settings').click()
                    page.wait_for_function("() => (document.querySelector('#settings-status')?.textContent || '').match(/[Ss]aved/)")
                    saved = json.loads(settings_path.read_text())
                    assert {key: saved.get(key) for key in legacy} == legacy
                    assert not saved.get("OPENAI_COMPATIBLE_BASE_URL"), "the legacy pair is still the one dispatch reads"
                    assert {"kind": "openai", "name": "default", "enabled": False} in saved["OUROBOROS_CONNECTIONS"]
                    _shot(page, evidence, "managed-legacy-saved-light-1400", openai)
                page.close()
            page = browser.new_page(viewport={"width": 390, "height": 844})
            page.add_init_script("localStorage.setItem('ouroboros.theme', 'dark')")
            page.goto(fixture["url"] + "/onboarding", wait_until="domcontentloaded")
            key = page.locator('#openai-key')  # the wizard's own key field, the same flat setting
            key.wait_for(state="visible")
            assert key.evaluate("node => node.readOnly") and page.locator('[data-clear="openai-key"]').count() == 0
            assert page.locator('#openai-key-help').inner_text() == (
                "OpenAI API Key: set by your organization, its Safety Supervisor uses it.")
            assert not page.locator('#openrouter-key').evaluate("node => node.readOnly"), "a key it does not read stays free"
            compatible = page.locator('[data-settings-panel="providers"] [data-connection-row][data-builtin="1"]')
            assert compatible.locator('[data-connection-fixed]').is_visible()
            key.scroll_into_view_if_needed()
            _shot(page, evidence, "managed-legacy-wizard-keys-dark-390")
            compatible.scroll_into_view_if_needed()
            _shot(page, evidence, "managed-legacy-wizard-compatible-dark-390")
            page.close()
        finally:
            browser.close()
