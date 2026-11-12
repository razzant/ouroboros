"""The real SPA under a deployment policy, with the same server unmanaged as its positive control.

An employee sees what the organization fixes, still changes what it allows, finds no editor for what it fixes and
no door to replace the serving program, and keeps Panic. Frames go to ``OUROBOROS_UI_EVIDENCE_DIR`` for a look.

Permissions, model roles and provider keys are exercised on their own pages. The organization owns only
its fixed fields; the employee can still turn a different provider key off.
"""

from __future__ import annotations

import json
import os
import pathlib

import pytest

from tests.settings_browser_helpers import commit_row, open_settings, setting_row
from tests.test_ui_smoke_playwright import direct_server_with_data  # noqa: F401

POLICY = {"version": 1, "access": {"max": "pro"},
          "safety": {"min_coverage": "full", "checker": {"model": "anthropic@safety::claude-opus-5-5"}},
          "rules": "Customer records stay inside the corporate network."}
CHECKER_ROW = {"kind": "anthropic", "name": "safety", "label": "Organization checker", "api_key": "sk-ant-fixture-organization"}


def _shot(page, evidence, name, locator=None):
    target = locator or page
    target.screenshot(path=str(evidence / f"{name}.png"))


def _open_page(page, url, theme, path):
    page.add_init_script(f"localStorage.setItem('ouroboros.theme', {json.dumps(theme)})")
    page.goto(url + "/#settings", wait_until="domcontentloaded")
    level = open_settings(page, path)
    if path == "permissions":  # drawn from the read the page makes when it is shown
        level.locator('[data-setting-row="OUROBOROS_RUNTIME_MODE"] [role="radio"][aria-checked="true"]').wait_for(state="visible")
        page.wait_for_function("() => document.querySelector('#page-settings [data-setting-row=\"permissions:supervisor\"] "
                               "[data-setting-meta]').textContent.trim() !== ''")
    return level


def _values(row):
    return row.locator("[data-setting-value]").evaluate_all("nodes => nodes.map((node) => node.dataset.settingValue)")


@pytest.mark.ui_browser
def test_managed_permissions_say_what_the_organization_fixes(direct_server_with_data, tmp_path):  # noqa: F811
    """Permissions and checks and Your rules under the administrator's policy file: a bounded choice offers only what
    is allowed and its row says whose limit it is, a fixed one stays on the page with its one value, the Supervisor is
    said and not offered for change, and what the policy allows still saves. Unmanaged, nothing of it is said."""
    from playwright.sync_api import sync_playwright

    fixture = direct_server_with_data
    evidence = pathlib.Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", str(tmp_path / "frames")))
    evidence.mkdir(parents=True, exist_ok=True)
    settings_path = fixture["data_dir"] / "settings.json"
    meta = "[data-setting-meta]"
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            # --- unmanaged: the positive control ---
            page = browser.new_page(viewport={"width": 1400, "height": 1000})
            level = _open_page(page, fixture["url"], "dark", "permissions")
            assert _values(setting_row(page, "OUROBOROS_RUNTIME_MODE")) == ["light", "advanced", "pro", "cyber_pro"]
            assert _values(setting_row(page, "OUROBOROS_SAFETY_MODE")) == ["off", "light", "full"]
            assert "organization" not in level.inner_text().lower()
            supervisor = setting_row(page, "permissions:supervisor")
            assert supervisor.evaluate("row => row.tagName") == "BUTTON", "the Supervisor's model is a door to Models"
            assert not level.locator("[data-safety-exits]").is_visible(), "personal OSS keeps both exits, unsaid"
            assert not level.locator("[data-safety-organization-rules]").is_visible()
            _shot(page, evidence, "unmanaged-permissions-dark-1400")
            page.close()

            # --- the same server, started under the administrator's policy file ---
            fixture["stop_server"]()
            stored = json.loads(settings_path.read_text(encoding="utf-8"))
            stored["OUROBOROS_CONNECTIONS"] = [CHECKER_ROW]
            settings_path.write_text(json.dumps(stored), encoding="utf-8")
            policy_path = tmp_path / "admin" / "deployment-policy.json"
            policy_path.parent.mkdir(exist_ok=True)
            policy_path.write_text(json.dumps(POLICY), encoding="utf-8")
            fixture["env"]["OUROBOROS_DEPLOYMENT_POLICY"] = str(policy_path)
            fixture["start_server"]()

            for theme, width in (("light", 1400), ("dark", 1400), ("dark", 320)):
                page = browser.new_page(viewport={"width": width, "height": 1000 if width > 500 else 640})
                level = _open_page(page, fixture["url"], theme, "permissions")
                assert page.evaluate("document.documentElement.dataset.theme") == theme
                access = setting_row(page, "OUROBOROS_RUNTIME_MODE")
                assert _values(access) == ["light", "advanced", "pro"], "the ceiling leaves only what it allows"
                assert access.locator(meta).inner_text() == "Your organization allows up to Pro"
                for key in ("OUROBOROS_SAFETY_MODE", "OUROBOROS_CONSCIOUSNESS_SAFETY_MODE"):
                    row = setting_row(page, key)
                    assert row.is_visible() and _values(row) == ["full"], "a fixed level stays on the page as its one value"
                    assert row.locator(meta).inner_text().endswith("Set by your organization: + Commands.")
                supervisor = setting_row(page, "permissions:supervisor")
                assert supervisor.evaluate("row => row.tagName") == "DIV", "said, never offered for change"
                assert supervisor.locator(meta).inner_text() == "claude-opus-5-5 · Set by your organization"
                assert "decides alone" in setting_row(page, "permissions:second-check").locator(meta).inner_text()
                assert level.locator("[data-safety-exits]").inner_text() == (  # unnamed under the organization's checker
                    "Set by your organization: your approval cannot send a message the Supervisor has not checked.")
                assert level.locator("[data-safety-organization-rules]").is_visible()
                assert level.evaluate("level => level.scrollWidth <= level.clientWidth") and \
                    page.evaluate("document.documentElement.scrollWidth <= innerWidth")
                access.scroll_into_view_if_needed()
                _shot(page, evidence, f"managed-access-{theme}-{width}")
                supervisor.scroll_into_view_if_needed()
                _shot(page, evidence, f"managed-safety-{theme}-{width}")
                if width == 1400 and theme == "light":
                    # Keyboard: the capped list is the whole radio group; no hidden Cyber Pro takes the focus.
                    access.locator('[data-setting-value="pro"]').focus()
                    page.keyboard.press("ArrowDown")
                    assert page.evaluate("document.activeElement.dataset.settingValue") == "light"
                    page.keyboard.press("End")
                    assert page.evaluate("document.activeElement.dataset.settingValue") == "pro"
                    # A change the policy allows still saves at once; the raw shadowed Access stays the employee's.
                    assert commit_row(page, "OUROBOROS_REVIEW_ENFORCEMENT", "blocking") == \
                        "Saved · applies to new tasks · was Advisory · Undo"
                    saved = json.loads(settings_path.read_text(encoding="utf-8"))
                    assert saved["OUROBOROS_REVIEW_ENFORCEMENT"] == "blocking"
                    assert saved["OUROBOROS_RUNTIME_MODE"] == "light"
                    assert "OUROBOROS_SAFETY_MODE" not in saved and "OUROBOROS_MODEL_SAFETY" not in saved
                    # The employee's own rules stay editable, beside the sentence that the organization's apply too.
                    rules = _open_page(page, fixture["url"], theme, "rules")
                    assert rules.locator("#settings-rules").is_enabled()
                    rules.locator("[data-rules-organization]").wait_for(state="visible")
                    _shot(page, evidence, "managed-rules-light-1400")
                page.close()

            # --- a strict file: every Safety exit closed, said in one sentence ---
            fixture["stop_server"]()
            policy_path.write_text(json.dumps({"version": 1, "safety": {"checker": {"model": "anthropic@safety::claude-opus-5-5"},
                                                                        "allowed_exits": []}}), encoding="utf-8")
            fixture["start_server"]()
            page = browser.new_page(viewport={"width": 1400, "height": 1000})
            level = _open_page(page, fixture["url"], "dark", "permissions")
            exits = level.locator("[data-safety-exits]")
            assert exits.inner_text() == (
                "Set by your organization: Ouroboros cannot override a DANGEROUS refusal on its own; "
                "your approval cannot override a DANGEROUS refusal; "
                "your approval cannot send a message the Supervisor has not checked.")
            exits.scroll_into_view_if_needed()
            _shot(page, evidence, "managed-strict-safety-dark-1400")
            page.close()
            final = json.loads(settings_path.read_text(encoding="utf-8"))
            assert final["OUROBOROS_CONNECTIONS"][0]["api_key"] == CHECKER_ROW["api_key"]
        finally:
            browser.close()


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
            for theme, width in (("light", 1400), ("dark", 1400), ("dark", 390)):
                page = browser.new_page(viewport={"width": width, "height": 1000})
                _open_page(page, fixture["url"], theme, "models/role/safety")
                page.locator('[data-model-role-model]').wait_for(state='visible')
                assert page.locator('[data-model-role-model]').count() == 1
                key = open_settings(page, "accounts/key/openai")
                assert not key.locator('#s-openai').evaluate("node => node.readOnly")
                assert key.locator('[data-key-toggle]').is_visible()
                key.locator('[data-settings-fold] > summary').click()
                assert not key.locator('[data-setting-row="OPENAI_BASE_URL"] input').evaluate("node => node.readOnly")
                _shot(page, evidence, f"unmanaged-accounts-{theme}-{width}")
                page.close()

            fixture["stop_server"]()
            stored = json.loads(settings_path.read_text(encoding="utf-8"))
            stored["OUROBOROS_CONNECTIONS"] = [CHECKER_ROW]
            settings_path.write_text(json.dumps(stored), encoding="utf-8")
            policy_path = tmp_path / "admin" / "deployment-policy.json"
            policy_path.parent.mkdir(exist_ok=True)
            policy_path.write_text(json.dumps(POLICY), encoding="utf-8")
            fixture["env"]["OUROBOROS_DEPLOYMENT_POLICY"] = str(policy_path)
            fixture["start_server"]()
            for theme, width in (("light", 1400), ("dark", 1400), ("dark", 390)):
                page = browser.new_page(viewport={"width": width, "height": 1000})
                fixed = _open_page(page, fixture["url"], theme, "models/role/safety")
                fixed.locator('[data-model-role-fixed]').wait_for(state="visible")
                assert "Set by your organization" in fixed.inner_text()
                assert fixed.locator('[data-model-role-model]').count() == 0
                _shot(page, evidence, f"managed-models-{theme}-{width}")
                key = open_settings(page, "accounts/key/anthropic/safety")
                key.locator('[data-key-fixed]').wait_for(state="visible")
                assert not key.locator('[data-key-toggle], [data-editor-remove], [data-editor-save]').count()
                assert key.locator('[data-connection-label]').evaluate("node => node.readOnly")
                _shot(page, evidence, f"managed-accounts-{theme}-{width}")
                if width == 1400 and theme == "light":
                    open_settings(page).locator('[data-settings-open="dashboard:updates"]').click()
                    page.wait_for_function("() => document.querySelector('#updates-summary')?.textContent === 'Managed by your organization'")
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
            final = json.loads(settings_path.read_text(encoding="utf-8"))
            assert final["OUROBOROS_CONNECTIONS"][0]["api_key"] == CHECKER_ROW["api_key"]

            # --- the legacy source branch: a built-in compatible checker reaching OPENAI_BASE_URL + OPENAI_API_KEY ---
            fixture["stop_server"]()
            stored = json.loads(settings_path.read_text(encoding="utf-8"))
            stored["OPENAI_BASE_URL"] = stored.pop("OPENAI_COMPATIBLE_BASE_URL")
            stored["OPENAI_API_KEY"] = stored.pop("OPENAI_COMPATIBLE_API_KEY")
            settings_path.write_text(json.dumps(stored), encoding="utf-8")
            legacy = {key: stored[key] for key in ("OPENAI_BASE_URL", "OPENAI_API_KEY")}
            policy_path.write_text(json.dumps({"version": 1, "safety": {"checker": {"model": "openai-compatible::mock-model"},
                                                                        "allowed_exits": []}}), encoding="utf-8")
            fixture["start_server"]()
            for theme in ("light", "dark"):
                page = browser.new_page(viewport={"width": 1400, "height": 1000})
                _open_page(page, fixture["url"], theme, "accounts/key/openai")
                openai = page.locator('[data-connection-row]')
                assert page.locator('#s-openai').evaluate("node => node.readOnly")
                assert openai.locator('[data-connection-fixed-fields]').inner_text() == (
                    "API key: set by your organization, its Safety Supervisor uses it.")
                assert not openai.locator('[data-key-fixed]').is_visible()
                assert page.locator('[data-key-toggle]').is_visible()
                assert not page.locator('[data-editor-remove]').count()
                assert not openai.locator('.secret-clear').is_visible()
                fold = page.locator('[data-settings-page="accounts/key/openai"] [data-settings-fold]')
                fold.locator(':scope > summary').click()
                address = page.locator('[data-setting-row="OPENAI_BASE_URL"] input')
                assert address.evaluate("node => node.readOnly") and address.input_value() == legacy["OPENAI_BASE_URL"]
                assert "set by your organization" in page.locator('[data-setting-row="OPENAI_BASE_URL"]').inner_text().lower()
                _shot(page, evidence, f"managed-legacy-openai-{theme}-1400")
                if theme == "light":
                    with page.expect_response(lambda r: r.url.endswith('/api/settings') and r.request.method == 'POST'):
                        page.locator('[data-key-toggle]').click()
                    from playwright.sync_api import expect
                    expect(page.locator('[data-key-toggle]')).to_have_text('Turn on')
                    saved = json.loads(settings_path.read_text(encoding="utf-8"))
                    assert {key: saved.get(key) for key in legacy} == legacy
                    assert not saved.get("OPENAI_COMPATIBLE_BASE_URL")
                    assert {"kind": "openai", "name": "default", "enabled": False} in saved["OUROBOROS_CONNECTIONS"]
                compatible = open_settings(page, "accounts/key/openai-compatible")
                compatible.locator('[data-key-fixed]').wait_for(state='visible')
                assert not compatible.locator('[data-key-toggle], [data-editor-save]').count()
                _shot(page, evidence, f"managed-legacy-compatible-{theme}-1400")
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
