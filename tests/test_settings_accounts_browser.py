"""Accounts and keys pages: passive list, explicit provider activation, local key writes and phone layout."""
from __future__ import annotations

import copy
import os
from pathlib import Path

import pytest

from tests import test_subscription_setup_browser as setup_browser
from tests.test_harness_maintenance_browser import _inventory_row
from tests.settings_browser_helpers import open_settings, settings_page, open_entry, go_back, commit_row, press_editor_save, search_settings

subscription_ui = setup_browser.subscription_ui

pytestmark = [pytest.mark.ui_browser]


@pytest.fixture
def accounts_ui(subscription_ui):
    ui, page = subscription_ui, subscription_ui["page"]
    profiles = ui["fixture"]["status"]["profiles"]["profiles"]
    seed = copy.deepcopy(profiles[0])
    profiles[:] = []
    for number in range(18):
        profile = copy.deepcopy(seed)
        profile["profile"].update(harness_id="codex", profile_id=f"studio-{number + 1:02}",
                                  display_name=f"studio-{number + 1:02}@example.com", enabled=number != 4)
        profile["identity"] = {"email": f"studio-{number + 1:02}@example.com", "plan": "Pro"}
        profile["status"].update(verification="failed" if number == 2 else "passed",
                                  verification_source="vendor", availability="unavailable" if number == 2 else "available")
        profiles.append(profile)
    ui["fixture"]["status"]["quota"] = [{"subject": {"harness": "codex", "subject_id": p["profile"]["profile_id"]},
        "freshness": "fresh", "constraints": [{"id": "window", "used_ratio": number / 20,
        "resets_at": "2027-01-01T12:00:00Z"}]} for number, p in enumerate(profiles)]
    ui["settings"].update(OPENROUTER_API_KEY="synthetic-mask...", GITHUB_TOKEN="synthetic-mask...", GITHUB_REPO="studio/home",
        TEAM_HOOK="synthetic-mask...", OUROBOROS_CONNECTIONS=[{"kind": "openrouter", "name": "research",
        "label": "Research", "api_key": "synthetic-mask..."}])
    ui["settings"]["_meta"].update(custom_secret_keys=["TEAM_HOOK"], effects={"OUROBOROS_CONNECTIONS": "next_call",
        "GITHUB_TOKEN": "next_task", "GITHUB_REPO": "now"}, tokens={"OUROBOROS_CONNECTIONS": "list-1", "GITHUB_REPO": "repo-1"})
    facts = {"stale": False, "revision": 1, "program_reads": [], "probes": [], "reveals": [], "actions": []}
    program = _inventory_row("codex")
    program.update(newest={"status": "available", "version": "3.0.0", "source": "https://vendor.invalid/latest"},
                   updateAvailable="yes", releaseNotesUrl="https://vendor.invalid/releases")

    def settings(route):
        ui["settings"]["_meta"]["connections"] = [{"kind": "openrouter", "id": "openrouter", "name": "", "builtin": True,
            "enabled": True, "configured": True, "source": "settings"}, *[{"kind": row["kind"], "id": f"{row['kind']}@{row['name']}",
            "name": row["name"], "label": row.get("label", ""), "builtin": False, "enabled": row.get("enabled", True),
            "configured": True, "source": "settings"} for row in ui["settings"]["OUROBOROS_CONNECTIONS"]]]
        if route.request.method == "GET":
            route.fulfill(json=ui["settings"])
            return
        body = route.request.post_data_json
        ui["posts"].append(("/api/settings", body))
        if facts["stale"]:
            facts["stale"] = False
            route.fulfill(status=409, json={"saved": False, "code": "settings_changed_elsewhere", "error": "Changed elsewhere"})
            return
        changes = {}
        for key, value in body.items():
            if key == "if_unchanged":
                continue
            old = ui["settings"].get(key)
            ui["settings"][key] = value
            changes[key] = {"withheld": "structure" if key == "OUROBOROS_CONNECTIONS" else "secret"} if key != "GITHUB_REPO" else {"old": old, "new": value, "token": "repo-2"}
        facts["revision"] += 1
        ui["settings"]["_meta"]["tokens"]["OUROBOROS_CONNECTIONS"] = f"list-{facts['revision']}"
        route.fulfill(json={"status": "saved", "saved": True, "changes": changes})

    def inventory(route):
        facts["program_reads"].append(route.request.url)
        route.fulfill(json={"harnesses": [program]})

    def test_key(route):
        facts["probes"].append(route.request.post_data_json)
        route.fulfill(json={"ok": False, "error": "No credits"})

    def account_action(route):
        facts["actions"].append((route.request.method, route.request.url, route.request.post_data_json))
        route.fulfill(json={"ok": True})

    page.route("**/api/settings", settings)
    page.route("**/api/claudexor/maintenance/harnesses*", inventory)
    page.route("**/api/providers/test", test_key)
    page.route("**/api/claudexor/credential-profiles/**", account_action)
    page.route("**/api/extensions", lambda route: route.fulfill(json={"skills": [
        {"name": "weather", "grants": {"requested_keys": ["WEATHER_TOKEN"]}}], "live": {"settings_sections": []}}))
    return {**ui, "facts": facts}


def no_overflow(page):
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")
    assert page.locator("[data-settings-page]:visible").evaluate("e => e.scrollWidth <= e.clientWidth + 1")


def test_accounts_is_passive_and_provider_activation_is_explicit(accounts_ui):
    from playwright.sync_api import expect
    ui, page = accounts_ui, accounts_ui["page"]
    page.goto(ui["url"] + "/#settings")
    home = open_settings(page)
    home.locator('[data-settings-open="accounts"]').click()
    listing = settings_page(page, "accounts")
    expect(listing.locator('[data-entry-id="openrouter"]')).to_be_visible()
    starts = [url for url in ui["reads"] if any(part in url for part in ("/claudexor/status", "/model-catalog", "/review-pool"))]
    assert starts == [] and ui["facts"]["program_reads"] == [] and ui["posts"] == []
    provider = open_entry(page, "codex")
    expect(provider.locator(".harness-account-row")).to_have_count(18)
    expect(provider.locator(".harness-account-row").first).to_have_attribute("data-profile", "studio-03")
    expect(provider.locator("[data-family-maintenance]")).to_contain_text("3.0.0")
    assert len(ui["facts"]["program_reads"]) == 1
    assert len([url for url in ui["reads"] if "/claudexor/status" in url]) == 1
    go_back(page)
    expect(listing.locator('[data-entry-id="codex"]')).to_have_count(1)
    expect(listing.locator('[data-entry-id="codex"]')).to_contain_text("18")
    expect(listing.locator(".harness-account-row")).to_have_count(0)
    open_settings(page)
    assert "accounts:key:openrouter" in search_settings(page, "OpenRouter key")
    page.locator('[data-settings-result="accounts:key:openrouter"]').click()
    expect(settings_page(page, "accounts/key/openrouter").locator("[data-editor-save]")).to_be_visible()


def test_key_switch_save_test_and_confirmed_removal(accounts_ui):
    from playwright.sync_api import expect
    ui, page = accounts_ui, accounts_ui["page"]
    page.goto(ui["url"] + "/#settings")
    listing = open_settings(page, "accounts")
    head = listing.locator('[data-entry-id="openrouter@research"]')
    expect(head.locator("[data-entry-switch]")).to_be_checked()
    waiting = []

    def hold_read(route):
        waiting.append(route)
        page.evaluate("window.keyReadHeld = true")

    page.route("**/api/settings", hold_read)
    head.locator(".ui-switch").click()
    page.wait_for_function("window.keyReadHeld === true")
    expect(head.locator("[data-entry-switch]")).not_to_be_checked()
    assert not ui["posts"], "the optimistic switch changes before the fresh read and write"
    page.unroute("**/api/settings", hold_read)
    waiting.pop().fallback()
    expect(head.locator("[data-entry-switch]")).to_be_enabled()
    expect(head.locator("[data-entry-switch]")).not_to_be_checked()
    body = ui["posts"][-1][1]
    assert set(body) == {"OUROBOROS_CONNECTIONS", "if_unchanged"}
    assert body["OUROBOROS_CONNECTIONS"][0]["enabled"] is False
    assert body["if_unchanged"] == {"OUROBOROS_CONNECTIONS": "list-1"}
    ui["facts"]["stale"] = True
    before = len(ui["posts"])
    head.locator(".ui-switch").click()
    expect(listing.locator("[data-entry-result]")).to_contain_text("Changed on another device")
    expect(head.locator("[data-entry-switch]")).not_to_be_checked()
    assert len(ui["posts"]) == before + 1
    key = open_entry(page, "openrouter@research")
    key.locator("[data-connection-label]").fill("Night research")
    assert "Saved" in press_editor_save(page)
    assert ui["posts"][-1][1]["OUROBOROS_CONNECTIONS"][0]["label"] == "Night research"
    key.locator("[data-connection-test]").click()
    expect(key.locator("[data-editor-result]")).to_contain_text("No credits")
    assert ui["facts"]["probes"] == [{"provider_id": "openrouter", "connection": "research", "overrides": {}}]
    before = len(ui["posts"])
    key.locator("[data-editor-remove]").click()
    page.get_by_role("button", name="Cancel", exact=True).click()
    assert len(ui["posts"]) == before
    key.locator("[data-editor-remove]").click()
    page.get_by_role("button", name="Remove", exact=True).last.click()
    expect(settings_page(page, "accounts")).to_be_visible()
    assert ui["posts"][-1][1]["OUROBOROS_CONNECTIONS"] == []
    token = listing.locator('[data-setting-row="GITHUB_TOKEN"]')
    token.locator("[data-secret-open]").click()
    result = commit_row(page, "GITHUB_TOKEN", "synthetic-new-token")
    assert "Saved" in result and "was" not in result and "Undo" not in result
    expect(token.locator("[data-setting-control]")).to_have_value("")
    expect(token.locator("[data-setting-undo]")).to_have_count(0)
    assert "synthetic-new-token" not in token.inner_text()


@pytest.mark.parametrize("width,height", [(1087, 740), (320, 640)])
@pytest.mark.parametrize("theme", ["dark", "light"])
def test_accounts_provider_and_key_layout_and_evidence(accounts_ui, width, height, theme):
    from playwright.sync_api import expect
    ui, page = accounts_ui, accounts_ui["page"]
    page.set_viewport_size({"width": width, "height": height})
    page.add_init_script(f"localStorage.setItem('ouroboros.theme', '{theme}')")
    page.goto(ui["url"] + "/#settings")
    open_settings(page, "accounts/codex")
    expect(page.locator(".harness-account-row")).to_have_count(18)
    expect(page.locator("[data-family-maintenance]")).to_contain_text("3.0.0")
    evidence = os.environ.get("OUROBOROS_UI_EVIDENCE_DIR")

    def shot(name):
        page.evaluate("() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))")
        no_overflow(page)
        if evidence:
            Path(evidence).mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(Path(evidence) / f"e3-{name}-{width}x{height}-{theme}.png"), animations="disabled")

    shot("provider-top")
    if width == 320:
        page.locator("[data-family-maintenance]").evaluate("e => { const page = e.closest('[data-settings-page]'); page.scrollTop += e.getBoundingClientRect().top - page.getBoundingClientRect().top - 42; }")
    shot("provider")
    listing = open_settings(page, "accounts")
    expect(listing.locator('[data-entry-id="codex"]')).to_have_count(1)
    expect(listing.locator('[data-entry-id="codex"]')).to_contain_text("18")
    expect(listing.locator('[data-setting-row="TEAM_HOOK"]')).to_be_visible()
    page.locator("[data-settings-page]:visible").evaluate("e => e.scrollTop = 0")
    shot("accounts-top")
    listing.locator('[data-setting-row="TEAM_HOOK"]').scroll_into_view_if_needed()
    shot("accounts-end")
    key = open_settings(page, "accounts/key/openrouter/research")
    expect(key.locator("[data-connection-label]")).to_have_value("Research")
    shot("key")
