"""Full saved-secret Show through the served Settings UI, with synthetic keys."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tests.ui_chat_viewport_smoke import wait_drawer_closed
from tests.settings_browser_helpers import open_settings, commit_row, press_editor_save

pytest_plugins = ("tests.test_ui_smoke_playwright",)

ANTHROPIC = "sk-ant-api03-SYNTHETIC-" + "0123456789abcdef" * 7 + "-END"
CUSTOM = "synthetic-custom-" + "ABCDEF0123456789" * 6 + "-END"
OWN = "synthetic-own-" + "0A1B2C3D4E5F6789" * 6 + "-END"
MCP = "synthetic-mcp-" + "fedcba9876543210" * 6 + "-END"
ADDED = "sk-or-v1-SYNTHETIC-added-" + "13579bdf" * 8 + "-END"


@pytest.mark.ui_browser
@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_saved_secret_show_preserves_settings_and_drafts(direct_server_with_data, engine):
    from playwright.sync_api import expect, sync_playwright

    server = direct_server_with_data
    server["stop_server"]()
    settings_path = server["data_dir"] / "settings.json"
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    settings.update(ANTHROPIC_API_KEY=ANTHROPIC, TEST_REVEAL_KEY=CUSTOM, TEST_OWN_KEY=OWN,
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
            anthropic_card = open_settings(page, "accounts/key/anthropic")
            expect(anthropic_card.locator("[data-editor-save]")).to_be_visible(timeout=30_000)
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
            assert saves == [] and reveals == [{"key": "ANTHROPIC_API_KEY"}]
            anthropic_card.locator("[data-connection-test]").click()
            expect(anthropic_card.locator("[data-editor-result]")).to_contain_text("Works")
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
                assert page.locator("[data-settings-page]:visible").evaluate("e => e.scrollWidth <= e.clientWidth + 1")
                assert preview.evaluate("e => e.scrollWidth <= e.clientWidth + 1")
                assert preview.evaluate("e => { const s = getComputedStyle(e); return s.getPropertyValue('user-select') || s.getPropertyValue('-webkit-user-select'); }") == "text"
                page.screenshot(path=str(evidence / f"secret-shown-{engine}-{width}.png"))
            page.set_viewport_size({"width": 1280, "height": 900})
            toggle.click()
            expect(preview).to_be_hidden()
            expect(preview).to_have_text("")
            expect(field).to_be_visible()

            # A key added in a card is read by its connection and field, never by a top-level setting.
            open_settings(page, "accounts/key/openrouter/claude-credits")
            added = page.locator('[data-connection-row][data-kind="openrouter"][data-name="claude-credits"]')
            added.locator(".secret-toggle").click()
            expect(added.locator(".settings-secret-value")).to_have_text(ADDED)
            assert reveals[-1] == {"connection": "openrouter@claude-credits", "field": "api_key"}
            added.locator(".secret-toggle").click()
            expect(added.locator(".settings-secret-value")).to_be_hidden()

            # Every top-level consumer uses the same complete view, including dynamic rows. A provider
            # key has its one input in Accounts; Secrets draws no second one. A key a skill asked for is
            # drawn once, under its asker; an owner's own key is a custom row.
            listing = open_settings(page, "accounts")
            requested = listing.locator('[data-setting-row="TEST_REVEAL_KEY"]')
            expect(requested).to_have_count(1)
            custom = listing.locator('[data-setting-row="TEST_OWN_KEY"]')
            custom.locator("[data-secret-open]").click()
            custom.locator("[data-secret-show]").click()
            expect(custom.locator(".settings-secret-value")).to_have_text(OWN)
            assert reveals[-1] == {"key": "TEST_OWN_KEY"}
            requested.locator("[data-secret-open]").click()
            requested.locator("[data-secret-show]").click()
            expect(requested.locator(".settings-secret-value")).to_have_text(CUSTOM)
            assert reveals[-1] == {"key": "TEST_REVEAL_KEY"}
            expect(listing.locator('[data-setting-row="ANTHROPIC_API_KEY"]')).to_have_count(0)

            mcp_page = open_settings(page, "tools/saved_mcp")
            mcp = mcp_page.locator("[data-mcp-card]")
            expect(mcp.locator('[data-mcp-field="name"]')).to_have_value("Reveal fixture")
            mcp.locator('[data-mcp-field="name"]').fill("Reveal proof")
            mcp.locator("[data-mcp-more] > summary").click()
            mcp.locator("[data-mcp-header-toggle]").click()
            expect(mcp.locator(".settings-secret-value")).to_have_text(MCP)
            assert reveals[-1] == {"mcp_server_id": "saved_mcp"}
            mcp.locator('[data-mcp-field="id"]').fill("renamed-mcp")
            mcp.locator("[data-mcp-header-toggle]").click()
            expect(mcp.locator(".settings-secret-value")).to_have_text(MCP)
            expect(mcp.locator(".settings-secret-source")).to_have_text("Saved value for saved_mcp")
            assert reveals[-1] == {"mcp_server_id": "saved_mcp"}
            mcp.locator('[data-mcp-field="id"]').fill("saved_mcp")
            mcp.locator('[data-mcp-field="name"]').fill("Reveal proof")
            assert "Saved" in press_editor_save(page)
            assert saves[-1]["MCP_SERVERS"][0]["auth_token"] == MCP[:8] + "..."
            open_settings(page, "accounts")
            assert "Saved" in commit_row(page, "GITHUB_REPO", "owner/reveal-proof")
            assert saves[-1] == {"GITHUB_REPO": "owner/reveal-proof"}
            assert all(value not in json.dumps(saves) for value in (ANTHROPIC, CUSTOM, OWN, MCP, ADDED))

            # A real edit in Accounts survives Show: a draft is shown as typed, never as the saved key.
            open_settings(page, "accounts/key/anthropic")
            replacement = "synthetic-replacement-" + "1234567890" * 10
            field.fill(replacement)
            before = len(reveals)
            toggle.click()
            expect(preview).to_have_text(replacement)
            assert len(reveals) == before, "draft Show must not retrieve the old saved key"
            toggle.click()
            expect(field).to_have_value(replacement)
            assert "Saved" in press_editor_save(page)
            assert saves[-1]["ANTHROPIC_API_KEY"] == replacement
            assert json.loads(settings_path.read_text(encoding="utf-8"))["ANTHROPIC_API_KEY"] == replacement

            # Remove during a pending read remains a write; late old bytes never reappear.
            listing = open_settings(page, "accounts")
            custom = listing.locator('[data-setting-row="TEST_OWN_KEY"]')
            if custom.locator("[data-secret-show]").inner_text() == "Hide":
                custom.locator("[data-secret-show]").click()
            if custom.locator("[data-secret-open]").get_attribute("aria-expanded") != "true":
                custom.locator("[data-secret-open]").click()
            behavior["reveal"] = "hold"
            with page.expect_request(lambda r: r.url.endswith("/api/settings/secret")):
                custom.locator("[data-secret-show]").click()
            custom.locator("[data-secret-remove]").click()
            page.locator("[data-confirm-ok]").click()
            expect(custom).to_have_count(0)
            assert held
            held.pop().fulfill(json={"value": OWN})
            assert OWN not in listing.inner_text()
            assert saves[-1]["TEST_OWN_KEY"] == ""

            # Read failure stays local and retryable, without changing the key.
            github = listing.locator('[data-setting-row="GITHUB_TOKEN"]')
            github.locator("[data-secret-open]").click()
            behavior["reveal"] = "fail"
            github.locator("[data-secret-show]").click()
            expect(github.locator("[data-secret-reveal-status]")).to_contain_text("Synthetic read failure")
            expect(github.locator("[data-setting-control]")).to_have_value("")
            behavior["reveal"] = "real"
            github.locator("[data-secret-show]").click()
            expect(github.locator(".settings-secret-value")).to_have_text("synthetic-github-token")
            open_settings(page)
            listing = open_settings(page, "accounts")
            expect(listing.locator('[data-setting-row="GITHUB_TOKEN"] .settings-secret-value')).to_be_hidden()
        finally:
            browser.close()


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
            assert held == [], "chat does not read the Settings skill list"
            open_settings(page)
            listing_page = open_settings(page, "accounts")
            repo = listing_page.locator('[data-setting-row="GITHUB_REPO"] [data-setting-control]')
            expect(repo).to_be_visible(timeout=20_000)
            assert held, "Accounts renders while the passive skill reads are held"
            rows = listing_page.locator('[data-setting-row="TEST_REVEAL_KEY"]')
            expect(rows).to_have_count(0)
            repo.fill("owner/held-proof")
            repo.evaluate("e => { window.heldRepoNode = e; e.setSelectionRange(3, 8); }")
            while held:
                held.pop().fulfill(json=listing)
            expect(rows).to_have_count(1)
            expect(repo).to_have_value("owner/held-proof")
            assert repo.evaluate("e => e === window.heldRepoNode && document.activeElement === e")
            assert repo.evaluate("e => [e.selectionStart, e.selectionEnd]") == [3, 8]
            assert "Saved" in commit_row(page, "GITHUB_REPO", "owner/held-proof")
            assert saves[-1] == {"GITHUB_REPO": "owner/held-proof"}
            assert json.loads((server["data_dir"] / "settings.json").read_text(encoding="utf-8"))["GITHUB_REPO"] == "owner/held-proof"
            behavior["hold"] = False
            page.click('[data-nav-page="chat"]')
            expect(page.locator("#page-chat")).to_be_visible()
        finally:
            for route in held:
                route.abort()
            browser.close()
