"""Rendered MCP import and literal headers on an isolated real app, saved and reloaded.

The app is the ordinary direct-mode server over a temporary data root; the MCP
servers are synthetic loopback FastMCP apps that refuse requests without the
exact headers. Screenshots land in ``OUROBOROS_UI_EVIDENCE_DIR`` when set.
"""

from __future__ import annotations

import json
import os
import pathlib
import re

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


def _open_mcp(page, url, *, open_drawer=False):
    page.goto(url, wait_until="domcontentloaded", timeout=30_000)
    if open_drawer:
        page.locator('[data-mobile-nav-toggle]:visible').first.click()
    page.click('[data-nav-page="settings"]')
    page.locator('[data-settings-tab="advanced"]').click()
    page.locator("#btn-mcp-import").wait_for(state="visible", timeout=15_000)
    page.locator("#btn-save-settings").wait_for(state="visible", timeout=15_000)
    page.wait_for_function("() => !document.getElementById('btn-save-settings').disabled", timeout=20_000)


def _save(page):
    """Save a draft that changes something; an honest no-op is asserted where it happens."""
    expected_ids = page.locator('[data-mcp-field="id"]').evaluate_all('els => els.map(e => e.value)')
    with page.expect_response(lambda response: response.url.endswith('/api/settings')
                              and response.request.method == 'POST', timeout=20_000) as saved:
        page.click("#btn-save-settings")
    response = saved.value
    answer = response.json()
    assert response.ok and answer.get('status') == 'saved' and not answer.get('no_changes'), answer
    posted_ids = [s.get('id') for s in response.request.post_data_json['MCP_SERVERS']]
    assert posted_ids == expected_ids, ('save omitted visible draft rows', expected_ids, posted_ids)
    page.wait_for_function(
        "() => (document.getElementById('settings-status')?.textContent || '').startsWith('Settings saved')",
        timeout=20_000)
    loaded = page.evaluate("async () => (await fetch('/api/settings', {cache: 'no-store'})).json()")
    assert [s.get('id') for s in loaded['MCP_SERVERS']] == expected_ids


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
@pytest.mark.parametrize('reload_pending', [False, True])
def test_header_switch_stdio(direct_server_with_data, engine, saved_headers, reload_pending):  # noqa: F811
    """Retain draft headers on transport change until explicitly removed, then Save.

    Cover both a draft-only row and an edited previously saved map: the latter
    must not be resurrected from header row state after Remove unsupported.
    """
    from playwright.sync_api import expect, sync_playwright

    url, data_dir = direct_server_with_data['url'], direct_server_with_data['data_dir']
    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch(headless=True)
        page = browser.new_page(viewport={'width': 1440, 'height': 1000})
        page._evidence_engine = engine
        try:
            _open_mcp(page, url)
            pending = []
            def hold_reload(route):
                if route.request.method == 'GET' and not pending:
                    pending.append(route)
                else:
                    route.continue_()
            def start_reload():
                # The page's own load on showing Settings must settle first: held, it is not the
                # owner's reload, and landing later its status replaces the reload's own sentence.
                expect(page.locator('#settings-status')).to_have_text('Settings refreshed', timeout=30_000)
                page.route('**/api/settings', hold_reload)
                with page.expect_request(lambda request: request.url.endswith('/api/settings')
                                         and request.method == 'GET'):
                    page.click('#btn-reload-settings')
                assert len(pending) == 1
            if reload_pending and not saved_headers:
                start_reload()
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
                if reload_pending:
                    start_reload()
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
            if reload_pending:
                pending[0].fulfill(response=pending[0].fetch())
                page.unroute('**/api/settings', hold_reload)
                page.wait_for_function("() => (document.getElementById('settings-status')?.textContent || '').includes('Your edits are kept')")
                assert page.locator('[data-mcp-card]').last.locator('[data-mcp-field="id"]').input_value() == 'switch_headers'
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


@pytest.mark.parametrize('engine', ['chromium', 'webkit'])
def test_late_status_keeps_typed_field(direct_server_with_data, loopback, engine):  # noqa: F811
    """A status answer landing mid-typing repaints status only, never the field being typed.

    Playwright's fill (select, then insert) and a person typing both lose text
    when that answer rebuilds the card under the caret.
    """
    from playwright.sync_api import expect, sync_playwright

    url, data_dir = direct_server_with_data['url'], direct_server_with_data['data_dir']
    _app, http_url = loopback('http', {'X-Api-Key': CUSTOM})
    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch(headless=True)
        page = browser.new_page(viewport={'width': 1440, 'height': 1000})
        page._evidence_engine = engine
        try:
            held = []
            page.route('**/api/mcp/status', lambda route: held.append(route))
            _seed(page, url, [{'id': 'late_status', 'enabled': True, 'transport': 'streamable_http',
                               'url': http_url, 'headers': {'X-Api-Key': CUSTOM}}])
            saved = page.locator('[data-mcp-card]').first
            with page.expect_request(lambda request: request.url.endswith('/api/mcp/status'), timeout=20_000):
                saved.locator('[data-mcp-refresh]').click()
            page.click('#btn-mcp-add-server')
            card = page.locator('[data-mcp-card]').last
            card.locator('[data-mcp-field="id"]').fill('typed_draft')
            card.locator('[data-mcp-field="transport"]').select_option('stdio')
            command = page.locator('[data-mcp-card]').last.locator('[data-mcp-field="command"]')
            command.click()
            page.keyboard.type('pyth')
            page.evaluate('() => { window.__typedField = document.activeElement; }')
            for route in held:
                route.fulfill(response=route.fetch())
            held.clear()
            expect(saved.locator('.mcp-server-status')).to_have_text('1 tool discovered', timeout=20_000)
            expect(saved.locator('[data-mcp-message]')).to_contain_text('Saved catalog refreshed')
            assert 'echo' in saved.locator('.mcp-tools-disclosure').text_content()
            page.keyboard.insert_text('on3')
            assert page.evaluate('() => document.activeElement === window.__typedField'
                                 ' && window.__typedField.isConnected'), 'the status answer replaced the typed field'
            assert command.input_value() == 'python3'
            command.scroll_into_view_if_needed()
            _capture(page, 'mcp-late-status-typed')
            page.unroute('**/api/mcp/status')
            _save(page)
            assert _disk(data_dir)['typed_draft']['command'] == 'python3'
        finally:
            browser.close()


@pytest.mark.parametrize('engine', ['chromium', 'webkit'])
def test_late_status_keeps_tools_disclosure(direct_server_with_data, loopback, engine):  # noqa: F811
    """A status answer patches only changed tools: an open disclosure keeps its focus and selection.

    Unchanged tools keep every node; a changed list is patched inside the same
    open disclosure; an empty list removes it.
    """
    from playwright.sync_api import expect, sync_playwright

    url = direct_server_with_data['url']
    _app, http_url = loopback('http', {'X-Api-Key': CUSTOM})
    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch(headless=True)
        page = browser.new_page(viewport={'width': 1440, 'height': 1000})
        page._evidence_engine = engine
        try:
            held = []
            page.route('**/api/mcp/status', lambda route: held.append(route))
            _seed(page, url, [{'id': 'tools_focus', 'enabled': True, 'transport': 'streamable_http',
                               'url': http_url, 'headers': {'X-Api-Key': CUSTOM}}])
            card = page.locator('[data-mcp-card]').first
            message, badge = card.locator('[data-mcp-message]'), card.locator('.mcp-server-status')

            def answer(change=None, focused=False):
                """Refresh without moving focus, then answer its held status read, changed or not."""
                with page.expect_request(lambda request: request.url.endswith('/api/mcp/status'), timeout=20_000):
                    button = card.locator('[data-mcp-refresh]')
                    button.evaluate('button => button.click()') if focused else button.click()
                expect(message).to_contain_text('Refreshing tools')
                for route in held:
                    response = route.fetch()
                    data = response.json()
                    for entry in data['servers']:
                        if change and entry.get('id') == 'tools_focus':
                            change(entry)
                    route.fulfill(response=response, json=data)
                held.clear()
                expect(message).to_contain_text('Saved catalog refreshed', timeout=20_000)

            def kept():
                return page.evaluate("""() => {
                    const {details, summary, field, text} = window.__tools;
                    const card = document.querySelector('[data-mcp-card]');
                    return {same: card.querySelector('.mcp-tools-disclosure') === details && details.isConnected,
                            open: details.open, focused: document.activeElement === summary,
                            field: card.querySelector('[data-mcp-field="id"]') === field && field.isConnected,
                            selected: getSelection().toString() === text && text !== ''};
                }""")

            answer()
            expect(badge).to_have_text('1 tool discovered')
            summary = card.locator('.mcp-tools-disclosure summary')
            summary.focus()
            page.keyboard.press('Enter')
            page.evaluate("""() => {
                const details = document.querySelector('[data-mcp-card] .mcp-tools-disclosure');
                const name = details.querySelector('.mcp-tools-list strong');
                const range = document.createRange();
                range.selectNodeContents(name);
                getSelection().removeAllRanges();
                getSelection().addRange(range);
                window.__tools = {details, summary: document.activeElement, text: name.textContent,
                                  field: document.querySelector('[data-mcp-card] [data-mcp-field="id"]')};
            }""")
            ready = {'same': True, 'open': True, 'focused': True, 'field': True, 'selected': True}
            assert kept() == ready, 'test setup: the open disclosure must hold focus and a selection'

            answer(focused=True)
            assert kept() == ready, 'an unchanged status answer replaced the Discovered tools disclosure'
            card.locator('.mcp-tools-disclosure').scroll_into_view_if_needed()
            _capture(page, 'mcp-late-status-tools-kept')

            answer(lambda entry: (entry['tools'].append({'name': 'synthetic_lookup', 'description': 'Synthetic.'}),
                                  entry.update(tool_count=2)), focused=True)
            expect(badge).to_have_text('2 tools discovered')
            assert card.locator('.mcp-tools-list li').count() == 2
            assert 'synthetic_lookup' in card.locator('.mcp-tools-list').text_content()
            assert {key: value for key, value in kept().items() if key != 'selected'} == {
                'same': True, 'open': True, 'focused': True, 'field': True}, 'a changed tool list must patch in place'

            answer(lambda entry: entry.update(tools=[], tool_count=0), focused=True)
            expect(badge).to_have_text('0 tools discovered')
            assert card.locator('.mcp-tools-disclosure').count() == 0
            page.unroute('**/api/mcp/status')
        finally:
            browser.close()


@pytest.mark.parametrize('engine', ['chromium', 'webkit'])
@pytest.mark.parametrize('legacy_header', ['Authorization', 'X-Api-Key'])
def test_edited_legacy_token_stays_removed_on_stdio(direct_server_with_data, engine, legacy_header):  # noqa: F811
    """An edited saved legacy token, switched to stdio and removed, stays gone from Save, disk and reload.

    The edit makes the legacy row a header-map row and clears the legacy pair, so
    Remove unsupported fields must not rebuild the rows from the replaced saved token.
    """
    from playwright.sync_api import sync_playwright

    from ouroboros.mcp_client import normalize_server_config

    url, data_dir = direct_server_with_data['url'], direct_server_with_data['data_dir']
    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch(headless=True)
        page = browser.new_page(viewport={'width': 1440, 'height': 1000})
        page._evidence_engine = engine
        try:
            _seed(page, url, [{'id': 'legacy_switch', 'enabled': False, 'transport': 'streamable_http',
                               'url': 'https://legacy.example/mcp', 'auth_header': legacy_header, 'auth_token': BEARER}])
            card = page.locator('[data-mcp-card]').last
            assert _rows(card) == [(legacy_header, BEARER[:8] + '...')]
            card.locator('[data-mcp-header-field="value"]').fill('Bearer synthetic-edited-0006')
            card.locator('[data-mcp-field="transport"]').select_option('stdio')
            card = page.locator('[data-mcp-card]').last
            card.locator('[data-mcp-field="command"]').fill('python3')
            notice = card.locator('.form-row:has([data-mcp-clear-unsupported]) .muted').inner_text()
            card.locator('[data-mcp-clear-unsupported]').click()
            card = page.locator('[data-mcp-card]').last
            remaining = card.locator('.form-row:has([data-mcp-clear-unsupported]) .muted').all_inner_texts()
            _capture(page, f'mcp-legacy-stdio-removed-{legacy_header}')
            _save(page)
            entry = _disk(data_dir)['legacy_switch']
            errors = []
            assert normalize_server_config(entry, errors=errors) is not None, errors
            assert BEARER not in json.dumps(entry) and 'synthetic-edited' not in json.dumps(entry)
            assert (entry['transport'], entry['command'], entry.get('auth_token', ''),
                    entry.get('auth_header', 'Authorization'), 'headers' in entry) == (
                'stdio', 'python3', '', 'Authorization', False)
            assert notice == 'Fields retained but not applied: url, headers', notice
            assert remaining == [], remaining
            page.locator('[data-mcp-card]').last.locator('[data-mcp-field="transport"]').select_option('streamable_http')
            assert page.locator('[data-mcp-card]').last.locator('[data-mcp-header-row]').count() == 0
        finally:
            browser.close()


TENANT = "synthetic-browser-tenant-0003"
ALPHA = "synthetic-browser-alpha-0004"
BETA = "synthetic-browser-beta-0005"


def _seed(page, url, servers):
    """Write through the real Settings writer, then load the page from it and let its loads settle."""
    from playwright.sync_api import expect

    page.goto(url, wait_until="domcontentloaded", timeout=30_000)
    status = page.evaluate("""async (servers) => {
        const response = await fetch('/api/settings', {method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({MCP_ENABLED: true, MCP_SERVERS: servers})});
        return [response.status, await response.text()];
    }""", servers)
    assert status[0] == 200, status
    _open_mcp(page, url)
    # Showing Settings reloads it once more; that late load replaces every card and
    # drops a pending Refresh or Test message, so act only on the settled draft.
    expect(page.locator("#settings-status")).to_have_text("Settings refreshed", timeout=30_000)


def _rows(card):
    return [(row.locator('[data-mcp-header-field="name"]').input_value(),
             row.locator('[data-mcp-header-field="value"]').input_value())
            for row in card.locator('[data-mcp-header-row]').all()]


def _settings_posts(page):
    posts = []
    page.on("request", lambda request: posts.append(request.post_data_json)
            if request.method == "POST" and request.url.endswith("/api/settings") else None)
    return posts


def _disk(data_dir):
    servers = json.loads((data_dir / "settings.json").read_text(encoding="utf-8"))["MCP_SERVERS"]
    return {entry.get("id") or entry.get("name"): entry for entry in servers}


@pytest.mark.parametrize('engine', ['chromium', 'webkit'])
def test_mcp_header_editor_ui(direct_server_with_data, loopback, engine):  # noqa: F811
    """One header list over unchanged storage, and Show/Test/Save agreeing on identity.

    Saved: a legacy Authorization pair beside X-Tenant, two masked headers A/B,
    and a server identified by its name only. Every value is synthetic.
    """
    from playwright.sync_api import expect, sync_playwright

    url, data_dir = direct_server_with_data["url"], direct_server_with_data["data_dir"]
    http_app, http_url = loopback("http", {"Authorization": BEARER, "X-Tenant": TENANT})
    sse_app, sse_url = loopback("sse", {"X-Api-Key": CUSTOM})
    seed = [
        {"id": "legacy_docs", "enabled": True, "transport": "streamable_http", "url": http_url,
         "auth_header": "Authorization", "auth_token": BEARER, "headers": {"X-Tenant": TENANT}},
        {"id": "pair", "enabled": False, "transport": "streamable_http", "url": "https://pair.example/mcp",
         "headers": {"X-A": ALPHA, "X-B": BETA}},
        {"name": "Name Only", "enabled": True, "transport": "sse", "url": sse_url, "headers": {"X-Api-Key": CUSTOM}},
    ]
    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page._evidence_engine = engine
        try:
            _seed(page, url, seed)
            posts = _settings_posts(page)
            tests = []
            page.on("request", lambda request: tests.append(request.post_data_json)
                    if request.url.endswith("/api/mcp/test") else None)
            cards = page.locator("[data-mcp-card]")
            legacy, pair, named = cards.nth(0), cards.nth(1), cards.nth(2)
            # The legacy pair is one ordinary row with its own prefix mask; no second editor.
            assert _rows(legacy) == [("Authorization", BEARER[:8] + "..."), ("X-Tenant", "***set***")]
            assert page.locator('[data-mcp-field="auth_token"], [data-mcp-field="auth_header"]').count() == 0
            legacy.locator('[data-mcp-header-row="0"]').scroll_into_view_if_needed()
            _capture(page, "mcp-unified-legacy-row")
            legacy.locator('[data-mcp-header-row="0"] [data-mcp-header-toggle]').click()
            expect(legacy.locator('[data-mcp-header-row="0"] .settings-secret-value')).to_have_text(BEARER)
            legacy.locator('[data-mcp-header-row="0"] [data-mcp-header-toggle]').click()

            # An unrelated Save rewrites no credential: the legacy fields go back with their mask.
            page.fill("#s-mcp-tool-timeout", "45")
            _save(page)
            legacy_post = next(s for s in posts[-1]["MCP_SERVERS"] if s.get("id") == "legacy_docs")
            assert legacy_post["auth_header"] == "Authorization" and legacy_post["auth_token"] == BEARER[:8] + "..."
            disk = _disk(data_dir)
            assert {key: disk["legacy_docs"][key] for key in ("auth_header", "auth_token", "headers")} == {
                "auth_header": "Authorization", "auth_token": BEARER, "headers": {"X-Tenant": TENANT}}
            assert disk["pair"]["headers"] == {"X-A": ALPHA, "X-B": BETA}
            assert disk["Name Only"]["headers"] == {"X-Api-Key": CUSTOM}

            # A server known by its name alone: Test and Refresh use that saved identity.
            named.locator("[data-mcp-test]").click()
            expect(named.locator("[data-mcp-message]")).to_contain_text("Draft connection OK", timeout=20_000)
            assert tests[-1]["server_id"] == "Name Only" and tests[-1]["server"]["headers"] == {"X-Api-Key": "***set***"}
            named.locator("[data-mcp-refresh]").click()
            expect(named.locator("[data-mcp-message]")).to_contain_text("Saved catalog refreshed", timeout=20_000)

            # Another spelling of the same canonical server ID keeps every saved value for Test and Save.
            legacy.locator('[data-mcp-field="id"]').fill("Legacy Docs")
            legacy.locator("[data-mcp-test]").click()
            expect(legacy.locator("[data-mcp-message]")).to_contain_text("Draft connection OK", timeout=20_000)
            assert tests[-1]["server_id"] == "legacy_docs" and tests[-1]["server"]["id"] == "Legacy Docs"
            assert page.locator('[id^="mcp-0-header-"][id$="-error"]:visible').count() == 0
            saved_before = (data_dir / "settings.json").read_text(encoding="utf-8")
            with page.expect_response(lambda response: response.url.endswith('/api/settings')
                                      and response.request.method == 'POST', timeout=20_000) as renamed_save:
                page.click("#btn-save-settings")
            # Both spellings are one canonical ID, so this Save is an honest no-op that changes nothing on disk.
            answer = renamed_save.value.json()
            assert renamed_save.value.ok and answer["status"] == "saved" and answer.get("no_changes") is True, answer
            assert renamed_save.value.request.post_data_json["MCP_SERVERS"][0]["id"] == "Legacy Docs"
            page.wait_for_function(
                "() => (document.getElementById('settings-status')?.textContent || '').startsWith('No changes detected')",
                timeout=20_000)
            assert json.loads((data_dir / "settings.json").read_text(encoding="utf-8")) == json.loads(saved_before)
            disk = _disk(data_dir)
            assert disk["legacy_docs"]["auth_token"] == BEARER and disk["legacy_docs"]["headers"] == {"X-Tenant": TENANT}
            expect(legacy.locator('[data-mcp-field="id"]')).to_have_value("legacy_docs")
            assert _rows(legacy) == [("Authorization", BEARER[:8] + "..."), ("X-Tenant", "***set***")]

            # Saved A and B: remove B, rename masked A to B. Nothing substitutes B's value.
            pair.locator('[data-mcp-header-row="1"] [data-mcp-header-remove]').click()
            pair.locator('[data-mcp-header-row="0"] [data-mcp-header-field="name"]').fill("X-B")
            before = len(posts)
            page.click("#btn-save-settings")
            error = page.locator("#mcp-1-header-0-value-error")
            expect(error).to_contain_text("belongs to header X-A", timeout=10_000)
            assert len(posts) == before, "a refused draft must not be posted"
            tested = len(tests)
            pair.locator("[data-mcp-test]").click()
            expect(pair.locator("[data-mcp-message]")).to_contain_text("belongs to header X-A")
            assert len(tests) == tested, "Test must refuse the same row Save refuses"
            pair.locator('[data-mcp-header-row="0"] [data-mcp-header-toggle]').click()
            expect(pair.locator('[data-mcp-header-row="0"] .settings-secret-value')).to_have_text(ALPHA)
            expect(pair.locator('[data-mcp-header-row="0"] .settings-secret-source')).to_have_text("Saved value for X-A")
            pair.locator('[data-mcp-header-row="0"]').scroll_into_view_if_needed()
            _capture(page, "mcp-masked-rename-refused")
            pair.locator('[data-mcp-header-row="0"] [data-mcp-header-field="name"]').fill("x-a")  # case only
            _save(page)
            assert _disk(data_dir)["pair"]["headers"] == {"x-a": ALPHA}

            # Exact duplicate empty rows: the saved map would collapse them; the draft is still dirty.
            pair.locator("[data-mcp-header-add]").click()
            pair.locator('[data-mcp-header-row="1"] [data-mcp-header-field="name"]').fill("X-Dup")
            _save(page)
            assert _disk(data_dir)["pair"]["headers"] == {"x-a": ALPHA, "X-Dup": ""}
            indicator = page.locator("#settings-unsaved-indicator")
            expect(indicator).not_to_have_class(re.compile(r"\bis-visible\b"))
            pair.locator("[data-mcp-header-add]").click()
            pair.locator('[data-mcp-header-row="2"] [data-mcp-header-field="name"]').fill("X-Dup")
            expect(indicator).to_have_class(re.compile(r"\bis-visible\b"))
            page.click("#btn-save-settings")
            expect(page.locator("#mcp-1-header-2-name-error")).to_contain_text("repeats header X-Dup")
            pair.locator('[data-mcp-header-row="2"] [data-mcp-header-remove]').click()
            expect(indicator).not_to_have_class(re.compile(r"\bis-visible\b"))

            # Re-typing the mask is an edit: a preview taken before it is stale, the page is dirty,
            # leaving asks first, and Save refuses the reserved marker instead of keeping the old value.
            page.click("#btn-mcp-import")
            dialog = page.locator(".mcp-import-dialog")
            dialog.locator("#mcp-import-text").fill(json.dumps({"mcpServers": {"pair": {
                "type": "http", "url": "https://pair.example/mcp"}}}))
            dialog.locator("[data-mcp-import-preview-btn]").click()
            dialog.locator(".mcp-import-entry").first.wait_for(timeout=15_000)
            page.evaluate("""() => {
                const input = document.querySelector('[data-mcp-index="1"] [data-mcp-header-row="0"] [data-mcp-header-field="value"]');
                for (const value of ['synthetic-other', '***set***']) {
                    input.value = value;
                    input.dispatchEvent(new Event('input', {bubbles: true}));
                }
            }""")
            dialog.locator("[data-mcp-import-apply]").click()
            expect(dialog.locator("[data-mcp-import-status]")).to_contain_text("changed after this preview")
            dialog.locator("[data-mcp-import-cancel]").first.click()
            assert pair.locator('[data-mcp-header-row="0"] [data-mcp-header-field="value"]').input_value() == "***set***"
            expect(indicator).to_have_class(re.compile(r"\bis-visible\b"))
            page.click("#btn-reload-settings")
            confirm = page.get_by_role("button", name="Stay")
            confirm.wait_for(timeout=10_000)
            assert "You have unsaved settings changes" in page.locator("body").inner_text()
            confirm.click()
            expect(indicator).to_have_class(re.compile(r"\bis-visible\b"))
            before = len(posts)
            page.click("#btn-save-settings")
            expect(page.locator("#mcp-1-header-0-value-error")).to_contain_text("reserved for keeping a saved value")
            assert len(posts) == before
            pair.locator('[data-mcp-header-row="0"]').scroll_into_view_if_needed()
            _capture(page, "mcp-retyped-mask-refused")

            # A deliberate address change keeps disclosed values: origins only, never path or query.
            endpoint = legacy.locator('[data-mcp-field="url"]')
            endpoint.fill("https://moved.example:8443/v2/mcp?token=synthetic-query")
            notice = legacy.locator("[data-mcp-endpoint-notice]")
            port = http_url.split(":")[2].split("/")[0]
            expect(notice).to_have_text(
                f"Address changes from http://127.0.0.1:{port} to https://moved.example:8443. Saved values of "
                "Authorization, X-Tenant stay with this server and go to the new address when you Test or Save.")
            assert "/v2" not in notice.inner_text() and "synthetic-query" not in notice.inner_text()
            legacy.scroll_into_view_if_needed()
            _capture(page, "mcp-endpoint-change-notice")
            endpoint.fill(http_url)
            expect(notice).to_be_hidden()

            # A result never certifies a later draft, and a pending one never lands on another card.
            held = []
            page.route("**/api/mcp/test", lambda route: held.append(route))
            named.locator("[data-mcp-test]").click()
            page.wait_for_function("() => true")
            expect(named.locator("[data-mcp-message]")).to_contain_text("Testing connection")
            named.locator('[data-mcp-field="allowed_tools"]').fill("echo")
            page.wait_for_timeout(200)
            held[0].fulfill(response=held[0].fetch())
            expect(named.locator("[data-mcp-message]")).to_contain_text("changed while it was tested", timeout=20_000)
            page.unroute("**/api/mcp/test")
            held.clear()
            page.route("**/api/mcp/refresh", lambda route: held.append(route))
            legacy.locator("[data-mcp-refresh]").click()
            page.wait_for_timeout(200)
            assert len(held) == 1
            legacy.locator("[data-mcp-remove]").click()
            with page.expect_response(lambda response: response.url.endswith("/api/mcp/status")):
                held[0].fulfill(response=held[0].fetch())
            page.wait_for_timeout(300)
            assert page.locator("[data-mcp-card]").count() == 2
            for card in page.locator("[data-mcp-card]").all():
                assert "Saved catalog refreshed" not in card.locator("[data-mcp-message]").inner_text()
            page.unroute("**/api/mcp/refresh")
        finally:
            browser.close()
    assert {headers["authorization"] for headers in http_app.seen} == {BEARER}
    assert {headers["x-api-key"] for headers in sse_app.seen} == {CUSTOM}


@pytest.mark.parametrize('engine', ['chromium', 'webkit'])
def test_mcp_import_replaces_legacy_ui(direct_server_with_data, loopback, engine):  # noqa: F811
    """A file's headers replace the whole list, the saved single token included; Apply selects; Save writes."""
    from playwright.sync_api import expect, sync_playwright

    url, data_dir = direct_server_with_data["url"], direct_server_with_data["data_dir"]
    http_app, http_url = loopback("http", {"Authorization": BASIC, "X-Api-Key": CUSTOM})
    seed = [
        {"id": "legacy_docs", "enabled": True, "transport": "streamable_http", "url": http_url,
         "auth_header": "Authorization", "auth_token": BEARER, "headers": {"X-Tenant": TENANT}},
        {"id": "keep_me", "enabled": False, "transport": "streamable_http", "url": "https://keep.example/mcp",
         "headers": {"X-Keep": ALPHA}},
    ]
    document = {"servers": {
        "legacy_docs": {"type": "streamable-http", "url": http_url, "headers": {"Authorization": BASIC, "X-Api-Key": CUSTOM}},
        "keep_me": {"type": "http", "url": "https://keep.example/v2/mcp", "headers": {"X-Keep": "${KEEP_TOKEN}"}},
        "marker": {"type": "http", "url": "https://marker.example/mcp", "headers": {"X-Key": "***set***"}},
        "fresh": {"type": "http", "url": "https://fresh.example:9443/mcp?k=synthetic-query"},
    }, "inputs": []}
    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page._evidence_engine = engine
        try:
            _seed(page, url, seed)
            previews = []
            page.on("request", lambda request: previews.append(request.post_data_json)
                    if request.url.endswith("/api/mcp/import/preview") else None)
            page.click("#btn-mcp-import")
            dialog = page.locator(".mcp-import-dialog")
            dialog.locator("#mcp-import-text").fill(json.dumps(document, indent=2))
            dialog.locator("[data-mcp-import-preview-btn]").click()
            dialog.locator(".mcp-import-entry").nth(3).wait_for(timeout=15_000)
            text = dialog.inner_text()
            for fact in ("The file’s headers replace this server’s whole header list.", "Adds: X-Api-Key",
                         "Replaces: Authorization (saved single token)", "Removes: X-Tenant",
                         "X-Keep: ${...} is kept as written", "has the value ***set***",
                         "Address changes on https://keep.example", "https://fresh.example:9443",
                         "Other top-level keys are not imported: inputs"):
                assert fact in text, (fact, text)
            for secret in (BEARER, BASIC, CUSTOM, ALPHA, TENANT, "synthetic-query", "/v2"):
                assert secret not in text, secret
            sent = json.dumps(previews[-1]["servers"])
            assert previews[-1]["servers"][0] == {"id": "legacy_docs", "transport": "streamable_http",
                                                  "header_names": ["Authorization", "X-Tenant"],
                                                  "legacy_header": "Authorization"}
            assert all(value not in sent for value in (http_url, "keep.example", "***set***", BEARER[:8]))
            _capture(page, "mcp-import-replace-preview")
            dialog.locator('[data-mcp-import-select="1"]').uncheck()  # keep_me stays as saved
            expect(dialog.locator("[data-mcp-import-apply]")).to_have_text("Apply 2 to draft")
            dialog.locator("[data-mcp-import-apply]").click()
            assert dialog.count() == 0
            cards = page.locator("[data-mcp-card]")
            assert cards.count() == 3
            focused = page.evaluate("() => [document.activeElement?.dataset?.mcpField,"
                                    " document.activeElement?.closest('[data-mcp-card]')?.dataset?.mcpIndex]")
            assert focused == ["id", "2"], focused  # the new card takes the caret
            assert _rows(cards.nth(0)) == [("Authorization", BASIC), ("X-Api-Key", CUSTOM)]
            assert _rows(cards.nth(1)) == [("X-Keep", "***set***")]
            _save(page)
            disk = _disk(data_dir)
            assert disk["legacy_docs"]["headers"] == {"Authorization": BASIC, "X-Api-Key": CUSTOM}
            assert disk["legacy_docs"]["auth_token"] == "" and disk["legacy_docs"]["enabled"] is True
            # Save normalizes every server's browser opt-in, including unselected import rows.
            # HTTP entries remain ordinary MCP connections; the file cannot enable the bridge.
            assert disk["keep_me"] == {**seed[1], "id": "keep_me", "name": "", "command": "", "args": [],
                                       "auth_header": "Authorization", "auth_token": "", "allowed_tools": [],
                                       "browser_bridge": False, "browser_protocol": "playwright"}
            assert disk["fresh"]["enabled"] is False and disk["fresh"]["url"].endswith("?k=synthetic-query")
            seen_before = len(http_app.seen)
            cards.nth(0).locator("[data-mcp-refresh]").click()
            expect(cards.nth(0).locator("[data-mcp-message]")).to_contain_text("Saved catalog refreshed", timeout=20_000)
            for theme in ("light", "dark"):
                page.set_viewport_size({"width": 1440, "height": 1000})
                page.evaluate("(theme) => window.ouroTheme.set(theme)", theme)
                page.locator(f'html[data-theme="{theme}"]').wait_for(state="attached")
                cards.nth(0).scroll_into_view_if_needed()
                _capture(page, f"mcp-headers-{theme}")
                page.set_viewport_size({"width": 390, "height": 844})
                cards.nth(0).locator('[data-mcp-header-row="1"]').scroll_into_view_if_needed()
                row = cards.nth(0).locator('[data-mcp-header-row="1"]').bounding_box()
                assert row["x"] >= 0 and row["x"] + row["width"] <= 390, row
                _capture(page, f"mcp-headers-{theme}-narrow")
        finally:
            browser.close()
    # Before Save the old single token was still the saved credential; after it, only the file's values.
    assert http_app.seen[seen_before:] and {headers["authorization"] for headers in http_app.seen[seen_before:]} == {BASIC}


@pytest.mark.parametrize('engine', ['chromium', 'webkit'])
@pytest.mark.parametrize('imported', ['empty', 'map', 'absent'])
def test_removed_legacy_then_import(direct_server_with_data, loopback, engine, imported):  # noqa: F811
    """Remove the saved single token without saving, then import the same server: Apply, Test and Save keep it gone.

    The removed row lives only in the editor, while the server object still holds
    the saved token's mask; a file's ``headers`` clears that pair in its own patch.
    A file without ``headers`` keeps the draft list as edited, removal included.
    """
    from playwright.sync_api import expect, sync_playwright

    url, data_dir = direct_server_with_data["url"], direct_server_with_data["data_dir"]
    expected = {"empty": {}, "map": {"X-Api-Key": CUSTOM}, "absent": {"X-Tenant": TENANT}}[imported]
    http_app, http_url = loopback("http", expected)
    entry = {"type": "http", "url": http_url, **({} if imported == "absent" else {"headers": expected})}
    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page._evidence_engine = engine
        try:
            _seed(page, url, [{"id": "legacy_docs", "enabled": True, "transport": "streamable_http", "url": http_url,
                               "auth_header": "Authorization", "auth_token": BEARER, "headers": {"X-Tenant": TENANT}}])
            tests = []
            page.on("request", lambda request: tests.append(request.post_data_json)
                    if request.url.endswith("/api/mcp/test") else None)
            card = page.locator("[data-mcp-card]").first
            assert _rows(card) == [("Authorization", BEARER[:8] + "..."), ("X-Tenant", "***set***")]
            card.locator('[data-mcp-header-row="0"] [data-mcp-header-remove]').click()
            page.click("#btn-mcp-import")
            dialog = page.locator(".mcp-import-dialog")
            dialog.locator("#mcp-import-text").fill(json.dumps({"mcpServers": {"legacy_docs": entry}}))
            dialog.locator("[data-mcp-import-preview-btn]").click()
            dialog.locator(".mcp-import-entry").first.wait_for(timeout=15_000)
            assert "Update" in dialog.inner_text() and "Authorization" not in dialog.inner_text()
            dialog.locator("[data-mcp-import-apply]").click()
            assert dialog.count() == 0
            card = page.locator("[data-mcp-card]").first
            rows = [(name, "***set***" if imported == "absent" else value) for name, value in expected.items()]
            assert _rows(card) == rows, _rows(card)
            card.locator('[data-mcp-header-row="0"]' if rows else '[data-mcp-test]').scroll_into_view_if_needed()
            _capture(page, f"mcp-removed-legacy-import-{imported}")
            seen = len(http_app.seen)
            card.locator("[data-mcp-test]").click()
            expect(card.locator("[data-mcp-message]")).to_contain_text("Draft connection OK", timeout=20_000)
            assert tests[-1]["server"].get("auth_token", "") == ""
            assert http_app.seen[seen:] and all("authorization" not in headers for headers in http_app.seen[seen:])
            _save(page)
            disk = _disk(data_dir)["legacy_docs"]
            assert (disk.get("auth_token", ""), disk["headers"]) == ("", expected)
            assert BEARER not in json.dumps(disk)
            page.reload(wait_until="domcontentloaded")
            _open_mcp(page, url)
            assert _rows(page.locator("[data-mcp-card]").first) == [(name, "***set***") for name in expected]
        finally:
            browser.close()


@pytest.mark.parametrize('engine', ['chromium', 'webkit'])
def test_retyped_legacy_mask_cannot_replace_saved_token(direct_server_with_data, engine):  # noqa: F811
    """Edited-away-and-back legacy display masks must not become literal credentials."""
    from playwright.sync_api import expect, sync_playwright

    url, data_dir = direct_server_with_data['url'], direct_server_with_data['data_dir']
    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch(headless=True)
        page = browser.new_page(viewport={'width': 1280, 'height': 900})
        try:
            for token in (BEARER, 'short'):
                seed = [{'id': 'legacy_mask', 'enabled': False, 'transport': 'streamable_http',
                         'url': 'https://example.test/mcp', 'auth_header': 'X-Legacy', 'auth_token': token}]
                _seed(page, url, seed)
                field = page.locator('#mcp-0-header-0-value')
                mask = field.input_value()
                assert mask != token
                posts = []
                handler = lambda request: posts.append(request) if request.url.endswith('/api/settings') and request.method == 'POST' else None
                page.on('request', handler)
                field.fill('synthetic-other')
                field.fill(mask)
                page.click('#btn-save-settings')
                expect(page.locator('#mcp-0-header-0-value-error')).to_contain_text("saved token's mask")
                assert posts == []
                assert _disk(data_dir)['legacy_mask']['auth_token'] == token
                field.fill('synthetic-new-credential')
                _save(page)
                saved = _disk(data_dir)['legacy_mask']
                assert saved['auth_token'] == '' and saved['headers'] == {'X-Legacy': 'synthetic-new-credential'}
                page.remove_listener('request', handler)
        finally:
            browser.close()
