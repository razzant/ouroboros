"""Rendered MCP import and literal headers on an isolated real app, saved and reloaded.

The app is the ordinary direct-mode server over a temporary data root; the MCP
servers are synthetic loopback FastMCP apps that refuse requests without the
exact headers. Screenshots land in ``OUROBOROS_UI_EVIDENCE_DIR`` when set.
"""

from __future__ import annotations

import json
import os
import pathlib

import pytest

from tests.test_mcp_headers_loopback import loopback  # noqa: F401
from tests.test_ui_smoke_playwright import direct_server_with_data  # noqa: F401

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]

BEARER = "Bearer synthetic-browser-0001"
BASIC = "Basic c3ludGhldGljOmJyb3dzZXI="
CUSTOM = "synthetic-browser-key-0002"


def _capture(page, name):
    root = os.environ.get("OUROBOROS_UI_EVIDENCE_DIR")
    if root:
        pathlib.Path(root).mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(pathlib.Path(root) / f"{getattr(page, '_evidence_engine', 'chromium')}-{name}.png"), animations="disabled", full_page=False)


def _open_mcp(page, url):
    page.goto(url, wait_until="domcontentloaded", timeout=30_000)
    page.click('[data-nav-page="settings"]')
    page.locator('[data-settings-tab="advanced"]').click()
    page.locator("#btn-mcp-import").wait_for(state="visible", timeout=15_000)


def _save(page):
    page.click("#btn-save-settings")
    page.wait_for_function(
        "() => (document.getElementById('settings-status')?.textContent || '').startsWith('Settings saved')",
        timeout=20_000)


@pytest.mark.parametrize('engine', ['chromium', 'webkit'])
def test_mcp_import_ui(direct_server_with_data, loopback, engine):  # noqa: F811
    """Import, edit, refuse a repeated name, save, reload, Show, Refresh and Test.

    The short name keeps the isolated server's socket paths under the macOS
    AF_UNIX limit when the test root is a short TMPDIR.
    """
    from playwright.sync_api import sync_playwright

    url, data_dir = direct_server_with_data["url"], direct_server_with_data["data_dir"]
    http_app, http_url = loopback("http", {"Authorization": BEARER, "X-Api-Key": CUSTOM})
    sse_app, sse_url = loopback("sse", {"Authorization": BASIC, "X-Api-Key": CUSTOM})
    before = json.loads((data_dir / "settings.json").read_text(encoding="utf-8"))
    document = {"mcpServers": {
        "Docs HTTP": {"type": "http", "url": http_url, "headers": {"Authorization": BEARER, "X-Api-Key": CUSTOM}},
        "Docs SSE": {"type": "sse", "url": sse_url, "headers": {"Authorization": BASIC, "X-Api-Key": CUSTOM}},
        "Local files": {"command": "python3", "args": ["-m", "synthetic_server"], "env": {"DEBUG": "1"},
                        "autoApprove": ["read"]},
    }}
    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page._evidence_engine = engine
        try:
            _open_mcp(page, url)
            page.check("#s-mcp-enabled")
            page.click("#btn-mcp-import")
            dialog = page.locator(".mcp-import-dialog")
            dialog.locator("#mcp-import-text").fill(json.dumps(document, indent=2))
            dialog.locator("[data-mcp-import-preview-btn]").click()
            dialog.locator(".mcp-import-entry").nth(2).wait_for(timeout=15_000)
            preview_text = dialog.inner_text()
            assert preview_text.count("Add (disabled)") == 3 and "Not imported: autoApprove" in preview_text
            assert "Headers: Authorization, X-Api-Key" in preview_text
            for secret in (BEARER, BASIC, CUSTOM):
                assert secret not in preview_text
            _capture(page, "mcp-import-preview")
            dialog.locator("[data-mcp-import-apply]").click()
            assert dialog.count() == 0
            cards = page.locator("[data-mcp-card]")
            assert cards.count() == 3
            first = cards.nth(0)
            assert first.locator('[data-mcp-header-field="name"]').evaluate_all("els => els.map(e => e.value)") == [
                "Authorization", "X-Api-Key"]
            # A repeated name is a field error at Save, never a silent object collapse.
            first.locator("[data-mcp-header-add]").click()
            first = page.locator("[data-mcp-card]").nth(0)
            first.locator('[data-mcp-header-row="2"] [data-mcp-header-field="name"]').fill("x-api-key")
            first.locator('[data-mcp-header-row="2"] [data-mcp-header-field="value"]').fill("other")
            page.click("#btn-save-settings")
            error = page.locator("#mcp-0-header-2-name-error")
            error.wait_for(state="visible", timeout=10_000)
            assert "repeats header X-Api-Key" in error.inner_text()
            first.locator('[data-mcp-header-row="2"]').scroll_into_view_if_needed()
            _capture(page, "mcp-header-duplicate-refused")
            first.locator('[data-mcp-header-row="2"] [data-mcp-header-remove]').click()
            page.locator("[data-mcp-card]").nth(0).locator('[data-mcp-field="enabled"]').check()
            page.locator("[data-mcp-card]").nth(1).locator('[data-mcp-field="enabled"]').check()
            page.locator("[data-mcp-card]").nth(1).locator('[data-mcp-field="allowed_tools"]').fill("echo")
            _save(page)

            saved = json.loads((data_dir / "settings.json").read_text(encoding="utf-8"))
            servers = {entry["id"]: entry for entry in saved["MCP_SERVERS"]}
            assert servers["docs_http"]["headers"] == {"Authorization": BEARER, "X-Api-Key": CUSTOM}
            assert servers["docs_sse"]["headers"] == {"Authorization": BASIC, "X-Api-Key": CUSTOM}
            assert servers["local_files"]["enabled"] is False and servers["local_files"]["env"] == {"DEBUG": "1"}
            assert all(not entry.get("auth_token") for entry in servers.values())
            assert saved["OUROBOROS_MODEL"] == before["OUROBOROS_MODEL"]

            page.reload(wait_until="domcontentloaded")
            _open_mcp(page, url)
            served = page.evaluate("async () => (await fetch('/api/settings', {cache: 'no-store'})).text()")
            for secret in (BEARER, BASIC, CUSTOM):
                assert secret not in served
            card = page.locator("[data-mcp-card]").nth(0)
            value = card.locator('[data-mcp-header-row="0"] [data-mcp-header-field="value"]')
            assert value.input_value() == "***set***"
            card.locator('[data-mcp-header-row="0"] [data-mcp-header-toggle]').click()
            shown = card.locator('[data-mcp-header-row="0"] .settings-secret-value')
            shown.wait_for(state="visible", timeout=10_000)
            assert shown.inner_text() == BEARER and value.input_value() == "***set***"
            _capture(page, "mcp-headers-reloaded-show")
            card.locator('[data-mcp-header-row="0"] [data-mcp-header-toggle]').click()

            card.locator("[data-mcp-refresh]").click()
            card.locator("[data-mcp-message]").filter(has_text="Saved catalog refreshed").wait_for(timeout=20_000)
            sse_card = page.locator("[data-mcp-card]").nth(1)
            sse_card.locator("[data-mcp-test]").click()
            sse_card.locator("[data-mcp-message]").filter(has_text="Draft connection OK").wait_for(timeout=20_000)
            assert "No tool was called" in sse_card.locator("[data-mcp-message]").inner_text()
            sse_card.locator("[data-mcp-message]").scroll_into_view_if_needed()
            _capture(page, "mcp-refresh-and-test-draft")
            page.set_viewport_size({"width": 390, "height": 844})
            sse_card.locator('[data-mcp-header-row="0"]').scroll_into_view_if_needed()
            row = sse_card.locator('[data-mcp-header-row="0"]').bounding_box()
            assert row["x"] >= 0 and row["x"] + row["width"] <= 390, row
            _capture(page, "mcp-headers-narrow")
            page.set_viewport_size({"width": 1440, "height": 1000})

            # A second import updates in place: absent fields stay, a transport change is refused.
            update = {"mcpServers": {
                "Docs SSE": {"type": "sse", "url": sse_url},
                "Local files": {"command": "python3", "args": ["-m", "synthetic_server_v2"]},
                "Docs HTTP": {"type": "sse", "url": http_url},
            }}
            page.click("#btn-mcp-import")
            dialog.locator("#mcp-import-text").fill(json.dumps(update))
            dialog.locator("[data-mcp-import-preview-btn]").click()
            dialog.locator(".mcp-import-entry").nth(2).wait_for(timeout=15_000)
            preview_text = dialog.inner_text()
            assert preview_text.count("Update") == 2 and "Not applied" in preview_text
            assert "remove that server first" in preview_text
            _capture(page, "mcp-import-update-preview")
            dialog.locator("[data-mcp-import-apply]").click()
            _save(page)
            updated = {entry["id"]: entry for entry in json.loads(
                (data_dir / "settings.json").read_text(encoding="utf-8"))["MCP_SERVERS"]}
            assert updated["docs_sse"]["headers"] == {"Authorization": BASIC, "X-Api-Key": CUSTOM}
            assert updated["docs_sse"]["enabled"] is True and updated["docs_sse"]["allowed_tools"] == ["echo"]
            assert updated["local_files"]["args"] == ["-m", "synthetic_server_v2"]
            assert updated["local_files"]["enabled"] is False and updated["local_files"]["env"] == {"DEBUG": "1"}
            assert updated["docs_http"] == servers["docs_http"]  # the refused entry changed nothing

            # The ordinary form is a separate input path, not just an import renderer.
            page.click('#btn-mcp-add-server')
            manual = page.locator('[data-mcp-card]').nth(3)
            manual.locator('[data-mcp-field="id"]').fill('manual_docs')
            manual.locator('[data-mcp-field="url"]').fill(http_url)
            for number, (name, literal) in enumerate({'Authorization': BEARER, 'X-Api-Key': CUSTOM}.items()):
                manual.locator('[data-mcp-header-add]').click()
                manual = page.locator('[data-mcp-card]').nth(3)
                manual.locator(f'[data-mcp-header-row="{number}"] [data-mcp-header-field="name"]').fill(name)
                manual.locator(f'[data-mcp-header-row="{number}"] [data-mcp-header-field="value"]').fill(literal)
            manual.locator('[data-mcp-field="enabled"]').check()
            _save(page)
            saved_from_form = json.loads((data_dir / 'settings.json').read_text(encoding='utf-8'))
            assert saved_from_form['MCP_SERVERS'][3]['headers'] == {'Authorization': BEARER, 'X-Api-Key': CUSTOM}
            from ouroboros import mcp_client
            manager = mcp_client.get_manager()
            manager.reconfigure(saved_from_form)
            for server_id in ('docs_http', 'docs_sse', 'manual_docs'):
                assert manager.refresh_server(server_id)['ok']
                result = manager._call_tool_result(f'mcp_{server_id}__echo', {'text': 'consumer-check'})
                assert result.status == 'ok' and 'echo:consumer-check' in result.text
            manual.locator('[data-mcp-header-row="0"]').scroll_into_view_if_needed()
            _capture(page, 'mcp-headers-manual-form')
        finally:
            browser.close()
    assert {headers["authorization"] for headers in http_app.seen} == {BEARER}
    assert {headers["x-api-key"] for headers in sse_app.seen} == {CUSTOM}


@pytest.mark.parametrize('engine', ['chromium', 'webkit'])
@pytest.mark.parametrize('saved_headers', [False, True])
def test_header_switch_stdio(direct_server_with_data, engine, saved_headers):  # noqa: F811
    """Retain draft headers on transport change until explicitly removed, then Save.

    Cover both a draft-only row and an edited previously saved map: the latter
    must not be resurrected from header row state after Remove unsupported.
    """
    from playwright.sync_api import sync_playwright

    url, data_dir = direct_server_with_data['url'], direct_server_with_data['data_dir']
    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch(headless=True)
        page = browser.new_page(viewport={'width': 1440, 'height': 1000})
        page._evidence_engine = engine
        try:
            _open_mcp(page, url)
            page.click('#btn-mcp-add-server')
            card = page.locator('[data-mcp-card]').last
            card.locator('[data-mcp-field="id"]').fill('switch_headers')
            card.locator('[data-mcp-field="url"]').fill('https://example.com/mcp')
            card.locator('[data-mcp-header-add]').click()
            card = page.locator('[data-mcp-card]').last
            card.locator('[data-mcp-header-field="name"]').fill('X-Api-Key')
            card.locator('[data-mcp-header-field="value"]').fill(CUSTOM)
            if saved_headers:
                _save(page)
                page.reload(wait_until='domcontentloaded')
                _open_mcp(page, url)
                card = page.locator('[data-mcp-card]').last
                assert card.locator('[data-mcp-header-field="value"]').input_value() == '***set***'
                card.locator('[data-mcp-header-field="value"]').fill('synthetic-edited-key')
            card.locator('[data-mcp-field="transport"]').select_option('stdio')
            card = page.locator('[data-mcp-card]').last
            card.locator('[data-mcp-field="command"]').fill('python3')
            remove = card.locator('[data-mcp-clear-unsupported]')
            assert remove.is_visible(), 'Draft-only header must have an explicit removal control'
            assert 'headers' in card.locator('.form-row:has([data-mcp-clear-unsupported]) .muted').inner_text()
            assert CUSTOM not in card.inner_text() and 'synthetic-edited-key' not in card.inner_text()
            _capture(page, f'mcp-stdio-retained-{saved_headers}')
            remove.click()
            card = page.locator('[data-mcp-card]').last
            assert card.locator('[data-mcp-clear-unsupported]').count() == 0
            _save(page)
            saved = json.loads((data_dir / 'settings.json').read_text(encoding='utf-8'))
            server = next(s for s in saved['MCP_SERVERS'] if s['id'] == 'switch_headers')
            assert server['transport'] == 'stdio' and server['command'] == 'python3'
            assert 'headers' not in server, 'Removed draft rows must not resurrect at collection'
            assert server['enabled'] is False
            card.locator('[data-mcp-field="transport"]').select_option('streamable_http')
            assert page.locator('[data-mcp-card]').last.locator('[data-mcp-header-row]').count() == 0
        finally:
            browser.close()
