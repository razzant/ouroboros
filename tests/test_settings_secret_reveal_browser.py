"""Full saved-secret Show through the served Settings UI, with synthetic keys."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

from tests.ui_chat_viewport_smoke import wait_drawer_closed

pytest_plugins = ("tests.test_ui_smoke_playwright",)

ANTHROPIC = "sk-ant-api03-SYNTHETIC-" + "0123456789abcdef" * 7 + "-END"
CUSTOM = "synthetic-custom-" + "ABCDEF0123456789" * 6 + "-END"
MCP = "synthetic-mcp-" + "fedcba9876543210" * 6 + "-END"
ADDED = "sk-or-v1-SYNTHETIC-added-" + "13579bdf" * 8 + "-END"


@pytest.mark.serial
@pytest.mark.ui_browser
@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_saved_secret_show_preserves_settings_and_drafts(direct_server_with_data, engine):
    from playwright.sync_api import expect, sync_playwright

    server = direct_server_with_data
    server["stop_server"]()
    settings_path = server["data_dir"] / "settings.json"
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    settings.update(ANTHROPIC_API_KEY=ANTHROPIC, TEST_REVEAL_KEY=CUSTOM,
                    GITHUB_TOKEN="synthetic-github-token", MCP_ENABLED=False,
                    OUROBOROS_CONNECTIONS=[{"kind": "openrouter", "name": "claude-credits",
                                            "label": "Claude credits", "api_key": ADDED}],
                    MCP_SERVERS=[{"id": "saved-mcp", "name": "Reveal fixture",
                                  "transport": "streamable_http", "enabled": False,
                                  "url": "https://example.invalid/mcp", "auth_token": MCP}])
    settings_path.write_text(json.dumps(settings), encoding="utf-8")
    server["start_server"]()
    saves, reveals, probes, held = [], [], [], []
    behavior = {"reveal": "real"}
    evidence = Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", str(server["data_dir"].parent)))
    evidence.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch()
        try:
            page = browser.new_page(viewport={"width": 1280, "height": 900})

            def record(request):
                if request.method == "POST" and request.url.endswith("/api/settings"):
                    saves.append(request.post_data_json)

            def reveal_route(route):
                reveals.append(route.request.post_data_json)
                if behavior["reveal"] == "hold":
                    held.append(route)
                elif behavior["reveal"] == "fail":
                    route.fulfill(status=500, json={"error": "Synthetic read failure"})
                else:
                    route.continue_()

            def probe_route(route):
                probes.append(route.request.post_data_json)
                route.fulfill(json={"ok": True})

            page.on("request", record)
            page.route("**/api/settings/secret", reveal_route)
            page.route("**/api/providers/test", probe_route)
            # Only advertise a requested key; saved values and reveals use the real gateway.
            page.route("**/api/extensions", lambda route: route.fulfill(json={
                "skills": [{"name": "reveal-fixture", "grants": {"requested_keys": ["TEST_REVEAL_KEY"]}}],
                "live": {"settings_sections": []},
            }))
            page.goto(server["url"], wait_until="domcontentloaded")
            page.wait_for_selector("#page-chat", timeout=30_000)
            page.click('[data-nav-page="settings"]')
            expect(page.locator("#btn-save-settings")).to_be_enabled(timeout=30_000)
            # Save is available before the initial page-shown document finishes applying.
            expect(page.locator("#settings-status")).to_have_text("Settings refreshed", timeout=30_000)
            anthropic_card = page.locator('[data-provider-card="anthropic"]')
            card_toggle = anthropic_card.locator("[data-provider-toggle]")
            if card_toggle.get_attribute("aria-expanded") != "true":
                card_toggle.click()
            field = page.locator("#s-anthropic")
            toggle = page.locator('.secret-toggle[data-target="s-anthropic"]')
            preview = page.locator("#s-anthropic-reveal")
            mask = field.input_value()
            assert mask == ANTHROPIC[:8] + "..."
            page.screenshot(path=str(evidence / f"secret-before-{engine}.png"))
            masked_box = field.bounding_box()
            toggle.focus()
            toggle.press("Enter")
            expect(preview).to_have_text(ANTHROPIC)
            expect(toggle).to_have_attribute("aria-expanded", "true")
            # The value takes the masked field's own place; the field steps aside unchanged.
            expect(field).to_be_hidden()
            shown_box = preview.bounding_box()
            assert abs(shown_box["x"] - masked_box["x"]) <= 1 and abs(shown_box["y"] - masked_box["y"]) <= 1
            # Its column, not a full-width row: the value ends before Hide (whose label resizes the column).
            assert shown_box["x"] + shown_box["width"] <= toggle.bounding_box()["x"], (masked_box, shown_box)
            expect(field).to_have_value(mask)
            expect(field).to_have_attribute("type", "password")
            expect(page.locator("#settings-unsaved-indicator")).not_to_have_class(re.compile(r"\bis-visible\b"))
            assert saves == [] and reveals == [{"key": "ANTHROPIC_API_KEY"}]
            anthropic_card.locator('[data-provider-test="anthropic"]').click()
            expect(anthropic_card.locator("[data-provider-test-status]")).to_contain_text("Works")
            assert probes == [{"provider_id": "anthropic", "overrides": {}}]

            for width in (1280, 780, 390):
                page.set_viewport_size({"width": width, "height": 900})
                menu = page.locator('#page-settings [data-mobile-nav-toggle]')
                if menu.is_visible():
                    if menu.get_attribute("aria-expanded") == "true":
                        menu.click()
                    wait_drawer_closed(page)
                preview.scroll_into_view_if_needed()
                expect(preview).to_have_text(ANTHROPIC)
                assert page.locator(".settings-scroll").evaluate("e => e.scrollWidth <= e.clientWidth + 1")
                assert preview.evaluate("e => e.scrollWidth <= e.clientWidth + 1")
                assert preview.evaluate("e => { const s = getComputedStyle(e); return s.getPropertyValue('user-select') || s.getPropertyValue('-webkit-user-select'); }") == "text"
                page.screenshot(path=str(evidence / f"secret-shown-{engine}-{width}.png"))
            page.set_viewport_size({"width": 1280, "height": 900})
            toggle.click()
            expect(preview).to_be_hidden()
            expect(preview).to_have_text("")
            expect(field).to_be_visible()

            # A key added in a card is read by its connection and field, never by a top-level setting.
            added = page.locator('[data-connection-row][data-kind="openrouter"][data-name="claude-credits"]')
            added.locator(".secret-toggle").click()
            expect(added.locator(".settings-secret-value")).to_have_text(ADDED)
            assert reveals[-1] == {"connection": "openrouter@claude-credits", "field": "api_key"}
            expect(page.locator("#settings-unsaved-indicator")).not_to_have_class(re.compile(r"\bis-visible\b"))
            added.locator(".secret-toggle").click()
            expect(added.locator(".settings-secret-value")).to_be_hidden()

            # Every top-level consumer uses the same complete view, including dynamic rows. A provider
            # key has its one input in Accounts; Secrets draws no second one.
            page.click('[data-settings-tab="secrets"]')
            expect(page.locator('#page-settings [data-secret-setting="ANTHROPIC_API_KEY"]')).to_have_count(1)
            custom = page.locator('[data-custom-secret-row][data-original-key="TEST_REVEAL_KEY"]')
            custom.locator("[data-row-secret-toggle]").click()
            expect(custom.locator(".settings-secret-value")).to_have_text(CUSTOM)
            custom.locator("[data-custom-secret-key]").fill("RENAMED_REVEAL_KEY")
            custom.locator("[data-row-secret-toggle]").click()
            expect(custom.locator(".settings-secret-value")).to_have_text(CUSTOM)
            expect(custom.locator(".settings-secret-source")).to_have_text("Saved value for TEST_REVEAL_KEY")
            assert reveals[-1] == {"key": "TEST_REVEAL_KEY"}
            custom.locator("[data-custom-secret-key]").fill("TEST_REVEAL_KEY")
            requested = page.locator("#skill-requested-secrets")
            requested.locator("[data-row-secret-toggle]").click()
            expect(requested.locator(".settings-secret-value")).to_have_text(CUSTOM)
            expect(page.locator("#settings-unsaved-indicator")).not_to_have_class(re.compile(r"\bis-visible\b"))

            page.click('[data-settings-tab="advanced"]')
            mcp = page.locator("[data-mcp-card]").first
            mcp.locator("[data-mcp-header-toggle]").click()
            expect(mcp.locator(".settings-secret-value")).to_have_text(MCP)
            assert reveals[-1] == {"mcp_server_id": "saved_mcp"}
            mcp.locator('[data-mcp-field="id"]').fill("renamed-mcp")
            mcp.locator("[data-mcp-header-toggle]").click()
            expect(mcp.locator(".settings-secret-value")).to_have_text(MCP)
            expect(mcp.locator(".settings-secret-source")).to_have_text("Saved value for saved_mcp")
            assert reveals[-1] == {"mcp_server_id": "saved_mcp"}
            mcp.locator('[data-mcp-field="id"]').fill("saved_mcp")
            page.locator("#s-gh-repo").fill("owner/reveal-proof")
            page.click("#btn-save-settings")
            expect(page.locator("#settings-status")).to_contain_text("Settings saved", timeout=20_000)
            assert len(saves) == 1
            assert "ANTHROPIC_API_KEY" not in saves[0] and "TEST_REVEAL_KEY" not in saves[0]
            assert saves[0]["MCP_SERVERS"][0]["auth_token"] == MCP[:8] + "..."
            assert all(value not in json.dumps(saves[0]) for value in (ANTHROPIC, CUSTOM, MCP, ADDED))
            assert "OUROBOROS_CONNECTIONS" not in saves[0], "an untouched added key travels in no save"

            # A real edit in Accounts survives Show: a draft is shown as typed, never as the saved key.
            page.click('[data-settings-tab="providers"]')
            replacement = "synthetic-replacement-" + "1234567890" * 10
            field.fill(replacement)
            before = len(reveals)
            toggle.click()
            expect(preview).to_have_text(replacement)
            assert len(reveals) == before, "draft Show must not retrieve the old saved key"
            toggle.click()
            expect(field).to_have_value(replacement)
            page.click("#btn-save-settings")
            expect(page.locator("#settings-status")).to_contain_text("Settings saved", timeout=20_000)
            assert saves[-1]["ANTHROPIC_API_KEY"] == replacement
            assert json.loads(settings_path.read_text(encoding="utf-8"))["ANTHROPIC_API_KEY"] == replacement

            # Clear during a pending read remains an edit; the late old value never reappears.
            page.click('[data-settings-tab="secrets"]')
            behavior["reveal"] = "hold"
            with page.expect_request(lambda r: r.url.endswith("/api/settings/secret")):
                custom.locator("[data-row-secret-toggle]").click()
            expect(custom.locator("[data-row-secret-toggle]")).to_have_text("Hide")
            custom.locator("[data-row-secret-clear]").click()
            assert held
            held.pop().fulfill(json={"value": CUSTOM})
            expect(custom.locator(".settings-secret-value")).to_be_hidden()
            expect(custom.locator("[data-custom-secret-value]")).to_have_value("")
            behavior["reveal"] = "real"
            page.click("#btn-save-settings")
            expect(page.locator("#settings-status")).to_contain_text("Settings saved", timeout=20_000)
            assert saves[-1]["TEST_REVEAL_KEY"] == ""

            # Read failure stays local and retryable, without changing the key or dirty state.
            github = page.locator("#s-secret-github-token").locator("..")
            before_value = github.locator("input").input_value()
            behavior["reveal"] = "fail"
            github.locator(".secret-toggle").click()
            expect(github.locator("[data-secret-reveal-status]")).to_contain_text("Synthetic read failure")
            expect(github.locator("input")).to_have_value(before_value)
            behavior["reveal"] = "real"
            github.locator(".secret-toggle").click()
            expect(github.locator(".settings-secret-value")).to_have_text("synthetic-github-token")
            expect(page.locator("#settings-unsaved-indicator")).not_to_have_class(re.compile(r"\bis-visible\b"))
            page.click("#btn-reload-settings")
            expect(github.locator(".settings-secret-value")).to_be_hidden()
        finally:
            browser.close()


@pytest.mark.serial
@pytest.mark.ui_browser
@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_settings_paints_and_saves_while_the_installed_skill_list_is_held(direct_server_with_data, engine):
    """The settings document, Save and an owner edit never wait for the installed-skill
    list; a hidden page does not read it; the skill-requested rows are enrichment that
    lands after that read and never replaces a draft the owner changed meanwhile."""
    from playwright.sync_api import expect, sync_playwright

    server = direct_server_with_data
    evidence = Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", str(server["data_dir"].parent)))
    evidence.mkdir(parents=True, exist_ok=True)
    listing = {"skills": [{"name": "reveal-fixture", "grants": {"requested_keys": ["TEST_REVEAL_KEY"]}}],
               "live": {"settings_sections": []}}
    held, saves = [], []
    behavior = {"hold": True}

    def listing_route(route):
        if behavior["hold"]:
            held.append(route)
        else:
            route.fulfill(json=listing)

    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch()
        try:
            page = browser.new_page(viewport={"width": 1280, "height": 900})
            page.on("request", lambda request: saves.append(request.post_data_json)
                    if request.method == "POST" and request.url.endswith("/api/settings") else None)
            page.route("**/api/extensions", listing_route)
            page.goto(server["url"], wait_until="domcontentloaded")
            page.wait_for_selector("#page-chat", timeout=30_000)
            page.wait_for_function("document.querySelector('#s-gh-repo') !== null", timeout=30_000)
            assert held == [], "the boot-time Settings load reads no installed-skill list while the page is hidden"

            page.click('[data-nav-page="settings"]')
            expect(page.locator("#btn-save-settings")).to_be_enabled(timeout=30_000)
            assert len(held) == 1, "the visible page reads the list once, beside the document"
            rows = page.locator("#skill-requested-secrets [data-secret-setting]")
            expect(rows).to_have_count(0)
            page.screenshot(path=str(evidence / f"settings-list-held-{engine}.png"))
            held.pop().fulfill(json=listing)
            expect(rows).to_have_count(1, timeout=20_000)
            expect(page.locator("#settings-status")).to_have_text("Settings refreshed", timeout=20_000)

            # An edit made while the list is held survives its arrival: the draft stays
            # the owner's, dirty, and Save sends it.
            page.click("#btn-reload-settings")
            for _ in range(200):  # the route interception lands asynchronously
                if held:
                    break
                page.wait_for_timeout(50)
            assert len(held) == 1, "Reload reads the list once more, beside the document"
            expect(page.locator("#btn-save-settings")).to_be_enabled()
            page.click('[data-settings-tab="advanced"]')  # the repo field lives on Advanced
            page.locator("#s-gh-repo").fill("owner/held-proof")
            expect(page.locator("#settings-unsaved-indicator")).to_have_class(re.compile(r"\bis-visible\b"))
            held.pop().fulfill(json=listing)
            expect(page.locator("#s-gh-repo")).to_have_value("owner/held-proof")
            expect(page.locator("#settings-unsaved-indicator")).to_have_class(re.compile(r"\bis-visible\b"))
            # A confirmed Save waits for the settings document only: its reload's list read stays
            # held, yet the save completes, Save is usable again and the owner can leave the page.
            page.click("#btn-save-settings")
            expect(page.locator("#settings-status")).to_contain_text("Settings saved", timeout=20_000)
            assert saves[-1]["GITHUB_REPO"] == "owner/held-proof"
            for _ in range(200):
                if held:
                    break
                page.wait_for_timeout(50)
            assert len(held) == 1, "the post-save reload reads the list once more, still held"
            expect(page.locator("#btn-save-settings")).to_be_enabled()
            expect(page.locator("#settings-unsaved-indicator")).not_to_have_class(re.compile(r"\bis-visible\b"))
            page.click('[data-nav-page="chat"]')
            expect(page.locator("#page-chat")).to_be_visible(timeout=10_000)
            behavior["hold"] = False
            held.pop().fulfill(json=listing)
            page.click('[data-nav-page="settings"]')
            expect(rows).to_have_count(1, timeout=20_000)
        finally:
            for route in held:
                route.abort()
            browser.close()
