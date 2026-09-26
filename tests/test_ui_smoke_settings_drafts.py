"""Settings draft and action receipts through the production browser UI."""

from __future__ import annotations

import re
import json
import os
from pathlib import Path

import pytest
from tests import test_subscription_setup_browser as setup_browser

subscription_ui = setup_browser.subscription_ui

pytest_plugins = ("tests.test_ui_smoke_playwright",)


@pytest.mark.serial
@pytest.mark.ui_browser
@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_settings_complete_draft_validation_and_local_stop(direct_server_with_data, engine):
    from playwright.sync_api import expect, sync_playwright

    saves = []
    owner_writes = []
    stop_requests = []
    delayed_reads = []
    hold_read = False
    receipt = {"status": 400, "body": {"saved": False, "error": "Synthetic validation refusal"}}

    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch()
        try:
            page = browser.new_page(viewport={"width": 1280, "height": 900})

            def settings_route(route):
                if route.request.method == "POST":
                    saves.append(route.request.post_data_json)
                    route.fulfill(status=receipt["status"], json=receipt["body"])
                    return
                response = route.fetch()
                data = response.json()
                data["TEST_ORIGINAL"] = "abcdefgh..."
                data.setdefault("_meta", {})["custom_secret_keys"] = ["TEST_ORIGINAL"]
                if hold_read:
                    delayed_reads.append((route, data))
                    page.evaluate('window.__heldSettingsReadReady = true')
                    return
                route.fulfill(response=response, json=data)

            def owner_route(route):
                if route.request.url.endswith("/api/owner/autostart"):
                    # Passive GET state read (Windows autostart): unavailable on
                    # this test host, and never an owner write.
                    route.fulfill(json={
                        "ok": True, "available": False, "enabled": False,
                        "launcher_exe": None, "platform_supported": False,
                    })
                    return
                if route.request.method != "POST":
                    route.fulfill(json={"ok": True})
                    return
                owner_writes.append(route.request.url)
                route.fulfill(json={"ok": True, "runtime_mode": "light", "restart_required": False})

            def stop_route(route):
                stop_requests.append(route.request.method)
                route.fulfill(status=500, json={"error": "Synthetic Stop refusal"})

            page.route("**/api/settings", settings_route)
            page.route("**/api/owner/**", owner_route)
            page.route("**/api/local-model/status", lambda route: route.fulfill(json={
                "status": "ready", "runtime_status": "available", "context_length": 8192,
            }))
            page.route("**/api/local-model/stop", stop_route)
            page.goto(direct_server_with_data["url"], wait_until="domcontentloaded")
            page.wait_for_selector("#page-chat", timeout=30_000)
            page.click('[data-nav-page="settings"]')
            expect(page.locator("#btn-save-settings")).to_be_enabled(timeout=30_000)
            page.click('[data-settings-tab="secrets"]')
            page.click("#btn-add-custom-secret")
            new_row = page.locator('[data-custom-secret-row]').last
            key = new_row.locator('[data-custom-secret-key]')
            value = new_row.locator('[data-custom-secret-value]')
            expect(key).to_be_focused()
            expect(page.locator("#settings-unsaved-indicator")).to_have_class(
                re.compile("is-visible"))
            page.evaluate("window.dispatchEvent(new CustomEvent('ouro:settings-updated', {detail: {source: 'test'}}))")
            expect(page.locator('[data-custom-secret-row]')).to_have_count(2)
            key.fill("bad!")
            value.fill("kept secret draft")
            expect(key).not_to_have_attribute("aria-invalid", "true")
            page.evaluate("""() => {
                const input = document.getElementById('s-workers');
                input.value = '0'; input.dispatchEvent(new Event('input', {bubbles: true}));
            }""")
            page.evaluate("window.__settingsDraftNode = document.querySelectorAll('[data-custom-secret-value]')[1]")
            page.click("#btn-save-settings")
            expect(key).to_have_attribute("aria-invalid", "true")
            expect(page.locator("#s-workers")).to_have_attribute("aria-invalid", "true")
            assert saves == [] and owner_writes == []
            assert key.get_attribute("aria-describedby") == key.get_attribute("id") + "-error"
            assert page.evaluate("window.__settingsDraftNode === document.querySelectorAll('[data-custom-secret-value]')[1]")
            key.fill("TEST_NEW")
            expect(key).not_to_have_attribute("aria-invalid", "true")
            expect(page.locator("#" + key.get_attribute("id") + "-error")).to_be_hidden()
            expect(value).to_have_value("kept secret draft")
            new_row.locator('[data-row-secret-toggle]').click()
            expect(value).to_have_attribute("type", "text")
            new_row.locator('[data-row-secret-clear]').click()
            expect(value).to_have_value("")
            expect(value).to_have_attribute("data-force-clear", "1")
            expect(value).to_have_attribute("type", "password")
            value.fill("kept secret draft")
            expect(value).not_to_have_attribute("data-force-clear", "1")
            page.click("#btn-save-settings")
            assert saves == [], "a corrected key cannot bypass another invalid field"
            page.evaluate("""() => {
                const input = document.getElementById('s-workers');
                input.value = '1'; input.dispatchEvent(new Event('input', {bubbles: true}));
            }""")
            expect(page.locator("#s-workers")).not_to_have_attribute("aria-invalid", "true")

            # A server refusal retains the entire editor and the new row.
            page.click("#btn-save-settings")
            expect(page.locator("#settings-status")).to_contain_text("Settings were not saved")
            assert len(saves) == 1 and saves[0]["TEST_NEW"] == "kept secret draft"
            assert "TEST_ORIGINAL" not in saves[0], "the unchanged mask is not a new credential"
            expect(value).to_have_value("kept secret draft")

            receipt.update(status=500, body={"saved": True, "error": "Synthetic post-commit failure"})
            page.click("#btn-save-settings")
            expect(page.locator("#settings-status")).to_contain_text("Settings were saved, but a later step failed")
            expect(value).to_have_value("kept secret draft")
            assert len(saves) == 2 and owner_writes == []
            page.click("#btn-reload-settings")
            page.get_by_role("button", name="Stay", exact=True).click()
            expect(value).to_have_value("kept secret draft")

            # A timed-out writer cannot invite a duplicate Save of an unknown outcome.
            receipt.update(status=503, body={"saved": None, "error": "Synthetic writer timeout"})
            page.click("#btn-save-settings")
            expect(page.locator("#settings-status")).to_contain_text("Save outcome unknown")
            expect(page.locator("#btn-save-settings")).to_be_disabled()
            expect(value).to_have_value("kept secret draft")
            assert len(saves) == 3 and owner_writes == []
            page.evaluate("document.getElementById('btn-save-settings').click()")
            assert len(saves) == 3

            evidence = os.environ.get("OUROBOROS_UI_EVIDENCE_OUT")
            if evidence:
                Path(evidence).mkdir(parents=True, exist_ok=True)
                page.screenshot(path=str(Path(evidence) / f"settings-draft-{engine}.png"))

            page.click("#btn-reload-settings")
            page.locator("[data-confirm-ok]").click()
            expect(page.locator("#btn-save-settings")).to_be_enabled()
            expect(page.locator('[data-custom-secret-row]')).to_have_count(1)
            original = page.locator('[data-custom-secret-row]')
            original.locator('[data-custom-secret-key]').fill("TEST_RENAMED")
            page.click("#btn-save-settings")
            expect(original.locator('[data-custom-secret-value]')).to_have_attribute("aria-invalid", "true")
            assert len(saves) == 3
            original.locator('[data-custom-secret-remove]').click()
            receipt.update(status=200, body={"status": "saved", "next_task_changed": True})
            page.click("#btn-save-settings")
            expect(page.locator("#settings-status")).to_contain_text("Settings saved")
            assert saves[-1]["TEST_ORIGINAL"] == ""
            assert "TEST_RENAMED" not in saves[-1]

            # A response begun while clean cannot replace a new draft typed during the read.
            page.click('[data-settings-tab="advanced"]')
            hold_read = True
            with page.expect_request(lambda request: request.url.endswith('/api/settings') and request.method == 'GET'):
                page.evaluate("window.dispatchEvent(new CustomEvent('ouro:settings-updated', {detail: {source: 'test'}}))")
            repo = page.locator("#s-gh-repo")
            repo.fill("owner/edited-during-refresh")
            page.wait_for_function('window.__heldSettingsReadReady === true')
            assert delayed_reads
            page.evaluate("window.__settingsRepoNode = document.getElementById('s-gh-repo')")
            hold_read = False
            for route, data in delayed_reads:
                route.fulfill(json=data)
            delayed_reads.clear()
            page.evaluate("() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))")
            expect(repo).to_have_value("owner/edited-during-refresh")
            assert page.evaluate("window.__settingsRepoNode === document.getElementById('s-gh-repo')")

            # Stop's HTTP failure stays beside the action, without rewriting confirmed state.
            expect(page.locator("#btn-local-stop")).to_be_enabled(timeout=10_000)
            page.click("#btn-local-stop")
            expect(page.locator("#local-model-action-status")).to_contain_text("Stop failed: Synthetic Stop refusal")
            assert page.evaluate("""() => {
                const result = document.getElementById('local-model-action-status').getBoundingClientRect();
                const viewport = document.querySelector('.settings-scroll').getBoundingClientRect();
                return result.top >= viewport.top && result.bottom <= viewport.bottom + 1;
            }""")
            expect(page.locator("#local-model-status")).to_contain_text("Ready")
            expect(page.locator("#btn-local-stop")).to_be_enabled()
            assert stop_requests == ["POST"]
            if evidence:
                page.screenshot(path=str(Path(evidence) / f"settings-stop-{engine}.png"))

            # Confirmed page leave also discards the reviewer editor, before any future GET.
            page.click('[data-settings-tab="agents"]')
            reviewer = page.locator('[data-slot-custom-api]').first
            saved_model = reviewer.input_value()
            reviewer.fill("test/unsaved-reviewer")
            page.click('[data-nav-page="chat"]')
            page.locator('[data-confirm-ok]').click()
            expect(reviewer).to_have_value(saved_model)
        finally:
            browser.close()


@pytest.mark.serial
@pytest.mark.ui_browser
def test_initial_settings_document_survives_early_edit_while_enrichment_waits(subscription_ui):
    """Hold actual optional reads until after an edit; no timing race or runtime."""
    from playwright.sync_api import expect

    ui, pending = subscription_ui, {"reviewers": [], "status": [], "catalog": []}
    page = ui["page"]
    ui["settings"]["OUROBOROS_MODEL"] = "claudexor::opaque-source=gpt-test"
    # An API lane is offered per provider whose credential is stored.
    ui["settings"]["OPENROUTER_API_KEY"] = "***set***"
    ui["fixture"]["catalog"]["model_sources"] = [
        {"id": "opaque-source", "label": "Managed models", "credentialHarness": "codex"},
    ]
    page.route("**/api/reviewer-slots", lambda route: pending["reviewers"].append(route))
    page.route("**/api/claudexor/status*", lambda route: pending["status"].append(route))
    page.route("**/api/model-catalog", lambda route: pending["catalog"].append(route))
    page.goto(ui["url"] + "/#settings")
    page.click('[data-settings-tab="models"]')
    main = page.locator('[data-model-role="main"]')
    main.wait_for(state="visible")
    assert pending["reviewers"], "the reviewer read must still be pending"
    assert pending["status"], "the status read must still be pending"
    assert page.locator('#btn-save-settings').is_enabled(), "the known document can be saved before enrichment"
    source = main.locator('[data-model-role-source]')
    source.select_option("api:openrouter")
    model = main.locator('[data-model-role-model]')
    model.fill("owner-kept-model")
    model.evaluate("element => { window.__earlyModel = element; element.setSelectionRange(4, 4); }")
    expect(page.locator('#settings-unsaved-indicator')).to_have_class(re.compile('is-visible'))

    page.unroute("**/api/reviewer-slots")
    page.unroute("**/api/claudexor/status*")
    for route in pending["reviewers"]:
        route.fulfill(json=ui["fixture"]["preview"]["reviewer_slots"])
    for route in pending["status"]:
        route.fulfill(json=ui["fixture"]["status"])
    expect(page.locator('#settings-status')).to_contain_text('Your edits are kept')
    assert pending["catalog"], "a kept draft must not skip global catalog enrichment"
    page.unroute("**/api/model-catalog")
    for route in pending["catalog"]:
        route.fulfill(content_type="application/json", body=json.dumps(ui["fixture"]["catalog"]))
    page.wait_for_function("""() => document.querySelector('[data-model-role="main"] [data-model-role-source]')
        .querySelector('option[value="subscription:opaque-source"]')""")
    expect(model).to_have_value("owner-kept-model")
    assert source.input_value() == "api:openrouter"
    assert model.evaluate("element => element === window.__earlyModel && element.selectionStart === 4")
    expect(page.locator('#settings-unsaved-indicator')).to_have_class(re.compile('is-visible'))
    expect(page.locator('#btn-save-settings')).to_be_enabled()
    expect(page.locator('#settings-status')).to_contain_text('Your edits are kept')
    assert not any(path == '/api/settings' for path, _ in ui['posts'])


@pytest.mark.serial
@pytest.mark.ui_browser
def test_mcp_local_json_errors_hold_whole_settings_save(subscription_ui):
    from playwright.sync_api import expect

    ui = subscription_ui
    page = ui['page']
    page.goto(ui['url'] + '/#settings')
    expect(page.locator('#btn-save-settings')).to_be_enabled()
    page.locator('[data-settings-tab="advanced"]').click()
    timeout = page.locator('#s-mcp-tool-timeout')
    timeout.fill('0')
    page.locator('#btn-save-settings').click()
    expect(page.locator('#settings-status')).to_have_text('Enter a positive tool timeout in seconds.')
    assert not [path for path, _ in ui['posts'] if path == '/api/settings']
    timeout.fill('60')
    expect(timeout).not_to_have_attribute('aria-invalid', 'true')
    page.locator('#btn-mcp-add-server').click()
    card = page.locator('[data-mcp-card]').last
    card.locator('[data-mcp-field="id"]').fill('synthetic-server')
    card.locator('[data-mcp-field="transport"]').select_option('stdio')
    card.locator('[data-mcp-field="command"]').fill('node')
    env = card.locator('[data-mcp-field="env"]')
    refs = card.locator('[data-mcp-field="env_from_settings"]')
    for invalid in ['{', '[]', '{"PORT": 42}']:
        env.fill(invalid)
        page.locator('#btn-save-settings').click()
        assert not [path for path, _ in ui['posts'] if path == '/api/settings']
        expect(env).to_have_value(invalid)
        expect(env).to_have_attribute('aria-invalid', 'true')
    env.fill('{"PORT":"9000"}')
    refs.fill('{')
    page.locator('#btn-save-settings').click()
    assert not [path for path, _ in ui['posts'] if path == '/api/settings']
    expect(refs).to_have_attribute('aria-invalid', 'true')
    refs.fill('null')
    expect(refs).not_to_have_attribute('aria-invalid', 'true')
    with page.expect_request(lambda request: request.url.endswith('/api/settings') and request.method == 'POST'):
        page.locator('#btn-save-settings').click()
    saved = [body for path, body in ui['posts'] if path == '/api/settings'][-1]
    servers = saved['MCP_SERVERS']
    if isinstance(servers, str): servers = json.loads(servers)
    assert servers[-1]['env'] == {'PORT': '9000'}
    assert servers[-1].get('env_from_settings') in ({}, None)


def test_older_failed_settings_reload_cannot_disable_newer_success(subscription_ui):
    """Release an old failure only after the newer real UI load has finished."""
    from playwright.sync_api import expect

    ui, page = subscription_ui, subscription_ui['page']
    with page.expect_response(lambda response: response.url.endswith('/api/model-catalog')):
        page.goto(ui['url'] + '/#settings')
    expect(page.locator('#btn-save-settings')).to_be_enabled()
    page.evaluate("""() => {
        const originalFetch = window.fetch;
        window.testSettingsReads = [];
        window.fetch = (input, init) => {
            if (String(input).endsWith('/api/settings') && (init?.method || 'GET') === 'GET') {
                return new Promise((resolve, reject) => testSettingsReads.push({resolve, reject}));
            }
            return originalFetch(input, init);
        };
    }""")
    page.locator('#btn-reload-settings').click()
    page.wait_for_function('testSettingsReads.length === 1')
    page.locator('#btn-reload-settings').click()
    page.wait_for_function('testSettingsReads.length === 2')
    newer = {**ui['settings'], 'GITHUB_REPO': 'owner/newer-document'}
    page.evaluate("""data => testSettingsReads[1].resolve(new Response(JSON.stringify(data), {
        status: 200, headers: {'Content-Type': 'application/json'},
    }))""", newer)
    expect(page.locator('#s-gh-repo')).to_have_value('owner/newer-document')
    expect(page.locator('#settings-status')).to_have_text('Settings loaded')
    page.evaluate("""async () => {
        testSettingsReads[0].reject(new Error('older read failed'));
        // Await rejection handlers before asserting; no elapsed-time race.
        await new Promise(resolve => setTimeout(resolve, 0));
    }""")
    expect(page.locator('#btn-save-settings')).to_be_enabled()
    expect(page.locator('#settings-status')).to_have_text('Settings loaded')
    expect(page.locator('#s-gh-repo')).to_have_value('owner/newer-document')

    # A failure of the current request still reports the real load failure.
    page.locator('#btn-reload-settings').click()
    page.wait_for_function('testSettingsReads.length === 3')
    page.evaluate("testSettingsReads[2].reject(new Error('current read failed'))")
    expect(page.locator('#btn-save-settings')).to_be_disabled()
    expect(page.locator('#settings-status')).to_contain_text('current read failed')
