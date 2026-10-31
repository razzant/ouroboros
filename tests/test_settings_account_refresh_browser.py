"""Settings account discovery and legacy cadence through the actual browser UI."""
from __future__ import annotations

import copy
import json
import re

import pytest

from tests import test_subscription_setup_browser as setup_browser

subscription_ui = setup_browser.subscription_ui
capture = setup_browser.capture

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]


@pytest.mark.parametrize("transition", ["login", "reconnect"])
@pytest.mark.parametrize("settle_first", ["settings", "catalog"])
def test_confirmed_accounts_refresh_models_without_replacing_draft(subscription_ui, transition, settle_first):
    from playwright.sync_api import expect

    ui, page = subscription_ui, subscription_ui["page"]
    connected = copy.deepcopy(ui["fixture"]["status"])
    catalog = copy.deepcopy(ui["fixture"]["catalog"])
    ui["settings"].update(OUROBOROS_MODEL="openai::saved-model", OPENAI_API_KEY="configured...")
    ui["fixture"]["catalog"] = {"items": [], "model_sources": [], "errors": []}
    if transition == "login":
        ui["fixture"]["status"]["profiles"]["profiles"] = []
    held_settings, held_catalog = [], []
    hold_catalog = True

    def hold_settings_read(route):
        if len(held_settings) == 2:
            route.fallback()
            return
        ui["reads"].append(route.request.url)
        held_settings.append(route)
        page.evaluate("count => window.heldSettingsReads = count", len(held_settings))

    def hold_catalog_read(route):
        if not hold_catalog:
            route.fallback()
            return
        ui["reads"].append(route.request.url)
        held_catalog.append(route)
        page.evaluate("window.heldCatalogRead = true")

    page.route("**/api/settings", hold_settings_read)
    page.route("**/api/model-catalog", hold_catalog_read)
    page.goto(ui["url"] + "/#settings")
    page.wait_for_function("window.heldSettingsReads === 2")
    # Page-show supersedes the initial GET before either document is applied.
    held_settings[0].fulfill(json=ui["settings"])
    page.wait_for_function("window.heldCatalogRead === true")
    if settle_first == "settings":
        held_settings[1].fulfill(json=ui["settings"])
        expect(page.locator("#btn-save-settings")).to_be_enabled()
    hold_catalog = False
    held_catalog[0].fulfill(json=ui["fixture"]["catalog"])
    expect(page.locator("#settings-model-catalog-status")).to_contain_text("No provider catalogs")
    if settle_first == "catalog":
        expect(page.locator("#btn-save-settings")).to_be_disabled()
        held_settings[1].fulfill(json=ui["settings"])
    expect(page.locator("#settings-status")).to_have_text("Settings refreshed")
    page.locator('[data-settings-tab="models"]').click()
    row = page.locator('[data-model-role="main"]')
    source, model = row.locator('[data-model-role-source]'), row.locator('[data-model-role-model]')
    expect(source.locator('option[value="subscription:codex"]')).to_have_count(0)
    model.fill("owner-unsaved-model")
    model.evaluate("e => { window.ownerDraftNode = e; e.setSelectionRange(4, 9); }")
    settings_reads = sum(url.endswith("/api/settings") for url in ui["reads"])

    def refresh_accounts():
        # The same store refresh used when the Accounts login card settles.
        page.evaluate("""async () => {
            const {claudexorStatus} = await import('/static/modules/claudexor_status_store.js');
            await claudexorStatus.refresh();
        }""")

    if transition == "reconnect":
        ui["fixture"]["status"]["reads"]["accounts"] = "failed"
        refresh_accounts()
    ui["fixture"]["status"] = connected
    ui["fixture"]["catalog"] = catalog
    with page.expect_response(lambda response: response.url.endswith("/api/model-catalog"), timeout=5000):
        refresh_accounts()
    expect(source.locator('option[value="subscription:codex"]')).to_have_text("Codex (model)")
    expect(model).to_have_value("owner-unsaved-model")
    assert model.evaluate("e => e === window.ownerDraftNode && document.activeElement === e")
    assert model.evaluate("e => [e.selectionStart, e.selectionEnd]") == [4, 9]
    assert sum(url.endswith("/api/settings") for url in ui["reads"]) == settings_reads
    reads = sum(url.endswith("/api/model-catalog") for url in ui["reads"])
    refresh_accounts()
    assert sum(url.endswith("/api/model-catalog") for url in ui["reads"]) == reads
    assert not ui["posts"], "discovery cannot save or replace the owner's draft"
    capture(page, f"settings-account-{transition}-{settle_first}-first")


def test_legacy_30_second_cadence_loads_as_valid_settings_form(subscription_ui, tmp_path, monkeypatch):
    from playwright.sync_api import expect
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient
    from ouroboros import config
    from ouroboros.gateway import settings

    ui, page = subscription_ui, subscription_ui["page"]
    legacy = {**ui["settings"], "OUROBOROS_BG_WAKEUP_MIN": 30, "OUROBOROS_BG_WAKEUP_MAX": 7200}
    legacy["OUROBOROS_SUBAGENTS"] = json.dumps(legacy["OUROBOROS_SUBAGENTS"])
    legacy.pop("_meta")
    legacy["OUROBOROS_CONTEXT_MODE_AUTO_LOW"] = "false"
    # The independent one-time Safety upgrade has already run; cadence is still legacy.
    legacy["OUROBOROS_CONSCIOUSNESS_SAFETY_MODE"] = "full"
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(legacy), encoding="utf-8")
    before = path.read_bytes()
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "SETTINGS_PATH", path)
    monkeypatch.setattr(settings, "apply_runtime_provider_defaults", lambda value: (value, False, []))
    app = Starlette(routes=[Route("/api/settings", settings.api_settings_get)])
    app.state.drive_root = app.state.repo_dir = tmp_path
    with TestClient(app) as client:
        response = client.get("/api/settings")
    assert response.status_code == 200
    ui["settings"].clear()
    ui["settings"].update(response.json())
    assert path.read_bytes() == before, "GET must not rewrite the legacy file"
    def save_response(route):
        if route.request.method != "POST":
            route.fallback()
            return
        payload = route.request.post_data_json
        ui["posts"].append(("/api/settings", payload))
        ui["settings"].update(payload)
        route.fulfill(json={"status": "saved", "saved": True})

    page.route("**/api/settings", save_response)
    page.goto(ui["url"] + "/#settings")
    expect(page.locator("#settings-status")).to_have_text(re.compile("Settings (loaded|refreshed)"))
    expect(page.locator("#s-bg-wakeup-min")).to_have_value("60")
    expect(page.locator("#s-bg-wakeup-max")).to_have_value("7200")
    assert page.locator("#s-bg-wakeup-min").evaluate("e => e.checkValidity()")
    page.locator('[data-settings-tab="advanced"]').click()
    page.locator("#s-gh-repo").fill("owner/unrelated-edit")
    with page.expect_request(lambda request: request.url.endswith("/api/settings") and request.method == "POST", timeout=5000):
        page.locator("#btn-save-settings").click()
    saved = [body for url, body in ui["posts"] if url == "/api/settings"][-1]
    assert saved["OUROBOROS_BG_WAKEUP_MIN"] == 60
    assert saved["GITHUB_REPO"] == "owner/unrelated-edit"
    expect(page.locator("#settings-status")).to_have_text("Settings saved")
    page.locator('[data-settings-tab="behavior"]').click()
    page.locator("#s-bg-wakeup-min").scroll_into_view_if_needed()
    capture(page, "settings-legacy-cadence")
