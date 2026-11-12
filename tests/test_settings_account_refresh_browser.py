"""Settings account discovery and legacy cadence through the actual browser UI."""
from __future__ import annotations

import copy
import json

import pytest

from tests import test_subscription_setup_browser as setup_browser
from tests.settings_browser_helpers import open_entry, open_settings, commit_row

subscription_ui = setup_browser.subscription_ui
capture = setup_browser.capture

pytestmark = [pytest.mark.ui_browser]


@pytest.mark.parametrize("transition", ["login", "reconnect"])
def test_confirmed_accounts_refresh_models_without_replacing_draft(subscription_ui, transition):
    from playwright.sync_api import expect

    ui, page = subscription_ui, subscription_ui["page"]
    connected = copy.deepcopy(ui["fixture"]["status"])
    catalog = copy.deepcopy(ui["fixture"]["catalog"])
    ui["settings"].update(OUROBOROS_MODEL="openai::saved-model", OPENAI_API_KEY="configured...")
    ui["fixture"]["catalog"] = {"items": [], "model_sources": [], "errors": []}
    if transition == "login":
        ui["fixture"]["status"]["profiles"]["profiles"] = []
    page.goto(ui["url"] + "/#settings")
    open_settings(page, "models/role/main")
    row = page.locator('[data-model-role="main"]')
    source, model = row.locator('[data-model-role-source]'), row.locator('[data-model-role-model]')
    expect(model).to_have_value("saved-model")
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
    capture(page, f"settings-account-{transition}")


def test_an_agents_form_renews_its_model_list_when_a_sign_in_settles(subscription_ui):
    """An agent's form arms the watch a role's page arms: a changed account answer reads the
    catalogue once, the same answer again reads nothing, and the form's unsaved words stay."""
    from playwright.sync_api import expect

    ui, page = subscription_ui, subscription_ui["page"]
    connected = copy.deepcopy(ui["fixture"]["status"])
    ui["fixture"]["status"]["profiles"]["profiles"] = []
    page.goto(ui["url"] + "/#settings")
    open_settings(page, "agents").locator("[data-entry-id]").first.wait_for(state="visible")
    form = open_entry(page, "codex").locator("[data-subagent-row]")
    words = form.locator('[data-subagent-field="recommended_use"]')
    expect(words).to_be_visible()

    def refresh_accounts():
        # The same store refresh used when a sign-in card settles.
        page.evaluate("""async () => {
            const {claudexorStatus} = await import('/static/modules/claudexor_status_store.js');
            await claudexorStatus.refresh();
        }""")

    def catalogue_reads():
        return sum("/api/model-catalog" in url for url in ui["reads"])

    refresh_accounts()  # the answer the form opened with is confirmed and is not a change
    words.fill("An unsaved description.")
    words.evaluate("e => { window.ownerDraftNode = e; }")
    settings_reads = sum(url.endswith("/api/settings") for url in ui["reads"])
    quiet = catalogue_reads()
    refresh_accounts()
    assert catalogue_reads() == quiet, "the same account answer reads no catalogue"
    ui["fixture"]["status"] = connected
    with page.expect_response(lambda response: "/api/model-catalog" in response.url, timeout=5000):
        refresh_accounts()
    assert words.evaluate("e => e === window.ownerDraftNode && e.value") == "An unsaved description."
    assert sum(url.endswith("/api/settings") for url in ui["reads"]) == settings_reads
    after = catalogue_reads()
    assert after == quiet + 1, "one catalogue read for one changed answer"
    refresh_accounts()
    assert catalogue_reads() == after
    # Shown again (a return from another page, a reconnect), the form reads its roster and catches the catalogue up;
    # the unsaved words stay (the form is drawn again around them).
    with page.expect_response(lambda response: "/api/model-catalog" in response.url, timeout=5000):
        page.evaluate("window.dispatchEvent(new CustomEvent('ouro:page-shown', { detail: { page: 'settings' } }))")
    expect(words).to_have_value("An unsaved description.")
    assert not ui["posts"], "discovery cannot save or replace the owner's draft"


def test_legacy_30_second_cadence_loads_as_valid_settings_form(subscription_ui, tmp_path, monkeypatch):
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
    # Cadence lives on Consciousness. Its GET normalization still never rewrites the file;
    # an unrelated row now writes only its own key instead of authoring the whole form.
    assert ui["settings"]["OUROBOROS_BG_WAKEUP_MIN"] == 60
    assert ui["settings"]["OUROBOROS_BG_WAKEUP_MAX"] == 7200
    open_settings(page, "accounts")
    assert "Saved" in commit_row(page, "GITHUB_REPO", "owner/unrelated-edit")
    saved = [body for url, body in ui["posts"] if url == "/api/settings"][-1]
    assert saved == {"GITHUB_REPO": "owner/unrelated-edit"}
    assert path.read_bytes() == before
    capture(page, "settings-legacy-cadence")
