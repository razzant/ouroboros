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
import time
from urllib.parse import quote

import pytest

from tests.test_mcp_headers_loopback import loopback  # noqa: F401
from tests.test_ui_smoke_playwright import direct_server_with_data  # noqa: F401
from tests.settings_browser_helpers import open_settings, open_entry, press_editor_save, commit_row, setting_row, top_path

pytestmark = [pytest.mark.ui_browser]

BEARER = "Bearer synthetic-browser-0001"
BASIC = "Basic c3ludGhldGljOmJyb3dzZXI="
CUSTOM = "synthetic-browser-key-0002"


def _capture(page, name):
    root = os.environ.get("OUROBOROS_UI_EVIDENCE_DIR")
    if root:
        pathlib.Path(root).mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(pathlib.Path(root) / f"{getattr(page, '_evidence_engine', 'chromium')}-{name}.png"), animations="disabled", full_page=False)


def _settings_ready(page, action):
    """Wait for the mount and shown-event settings reads, never for a held MCP status reply."""
    pending = set()
    def started(request):
        if request.url.endswith('/api/settings') and request.method == 'GET':
            pending.add(request)
    def finished(request):
        pending.discard(request)
    page.on('request', started)
    page.on('requestfinished', finished)
    page.on('requestfailed', finished)
    try:
        result = action()
        deadline = time.monotonic() + 30
        while True:
            while pending:
                assert time.monotonic() < deadline, 'Settings reads did not finish'
                page.wait_for_timeout(20)
            page.evaluate('() => new Promise(done => requestAnimationFrame(() => requestAnimationFrame(done)))')
            if not pending:
                return result
    finally:
        page.remove_listener('request', started)
        page.remove_listener('requestfinished', finished)
        page.remove_listener('requestfailed', finished)


def _open_mcp(page, url):
    page.goto(url, wait_until="domcontentloaded", timeout=30_000)
    return _list(page)


def _list(page):
    return _settings_ready(page, lambda: open_settings(page, 'tools', 'tools:import', timeout=30_000))


def _card(page, identity):
    level = _settings_ready(page, lambda: open_settings(page, 'tools/' + quote(identity, safe=''), timeout=30_000))
    card = level.locator('[data-mcp-card]')
    card.wait_for(state='visible')
    return card


def _save(page):
    """Write this server's form; an Add returns to the list, a saved entry stays on its page."""
    from playwright.sync_api import expect
    adding = top_path(page) == 'tools/new'
    identity = page.locator('[data-mcp-field="id"]').input_value()
    if adding:
        with page.expect_response(lambda r: r.url.endswith('/api/settings') and r.request.method == 'POST') as saved:
            page.locator('[data-editor-save]').click()
        assert saved.value.ok, saved.value.text()
        expect(page.locator('[data-settings-page="tools"] [data-entry-result]')).to_contain_text('Added')
        _card(page, identity)
    else:
        assert press_editor_save(page).startswith('Saved')


def _edit(card):
    """The server's secondary fields are in its own More disclosure."""
    more = card.locator('[data-mcp-more]')
    if not more.evaluate('node => node.open'):
        more.locator(':scope > summary').click()


def _apply_import(dialog):
    from playwright.sync_api import expect
    page = dialog.page
    with page.expect_response(lambda r: r.url.endswith('/api/settings') and r.request.method == 'POST') as saved:
        dialog.locator('[data-mcp-import-apply]').click()
    assert saved.value.ok, saved.value.text()
    expect(dialog).to_have_count(0)
    result = setting_row(page, 'tools:import').locator('[data-setting-result]')
    if saved.value.json().get('no_changes'):
        expect(result).to_have_text('')
    else:
        expect(result).to_contain_text('Imported.')


def _enable(page, identity):
    from playwright.sync_api import expect
    card = _card(page, identity)
    control = card.locator('[data-entry-switch]')
    if not control.is_checked():
        with page.expect_response(lambda r: r.url.endswith('/api/settings') and r.request.method == 'POST'):
            control.locator('xpath=..').click()
        expect(card.locator('[data-entry-switch]')).to_be_checked()
        expect(card.locator('[data-entry-switch]')).to_be_enabled()
    return card


def _facts(card):
    """The exact stamps and the engine's words on a server's page: {name: value}."""
    return card.locator('[data-mcp-facts]').evaluate(
        "dl => Object.fromEntries([...dl.children].map((fact) => [fact.querySelector('dt').textContent,"
        " fact.querySelector('dd').textContent]))")


@pytest.mark.parametrize('engine', ['chromium', 'webkit'])
def test_mcp_import_ui(direct_server_with_data, loopback, engine):  # noqa: F811
    """Import, edit, refuse a repeated name, save, reload, Show, Refresh and Test.

    The short name keeps the isolated server's socket paths under the macOS
    AF_UNIX limit when the test root is a short TMPDIR.
    """
    from playwright.sync_api import expect, sync_playwright

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
            control = setting_row(page, 'MCP_ENABLED').locator('[data-setting-control]')
            if not control.is_checked():
                with page.expect_response(lambda r: r.url.endswith('/api/settings') and r.request.method == 'POST'):
                    control.locator('xpath=..').click()
            _list(page)
            page.locator('[data-tools-action="tools:import"]').click()
            dialog = page.locator(".mcp-import-dialog")
            dialog.locator("#mcp-import-text").fill(json.dumps(document, indent=2))
            dialog.locator("[data-mcp-import-preview-btn]").click()
            dialog.locator(".mcp-import-entry").nth(2).wait_for(timeout=15_000)
            preview_text = dialog.inner_text()
            assert preview_text.count("Add (off)") == 3 and "Not imported: autoApprove" in preview_text
            assert "Headers: Authorization, X-Api-Key" in preview_text
            for secret in (BEARER, BASIC, CUSTOM):
                assert secret not in preview_text
            _capture(page, "mcp-import-preview")
            _apply_import(dialog)
            assert page.locator('[data-settings-page="tools"] [data-entry-id]').count() == 3
            first = _card(page, 'docs_http')
            _edit(first)
            assert first.locator('[data-mcp-header-field="name"]').evaluate_all("els => els.map(e => e.value)") == [
                "Authorization", "X-Api-Key"]
            # A repeated name is a field error at Save, never a silent object collapse.
            first.locator("[data-mcp-header-add]").click()
            first = page.locator("[data-mcp-card]").nth(0)
            first.locator('[data-mcp-header-row="2"] [data-mcp-header-field="name"]').fill("x-api-key")
            first.locator('[data-mcp-header-row="2"] [data-mcp-header-field="value"]').fill("other")
            page.locator("[data-editor-save]").click()
            error = first.locator('[data-editor-result]')
            expect(error).to_contain_text('repeats header X-Api-Key', timeout=10_000)
            expect(first.locator('[data-mcp-header-row="2"] [data-mcp-header-field="name"]')).to_be_focused()
            assert "repeats header X-Api-Key" in error.inner_text()
            first.locator('[data-mcp-header-row="2"]').scroll_into_view_if_needed()
            _capture(page, "mcp-header-duplicate-refused")
            first.locator('[data-mcp-header-row="2"] [data-mcp-header-remove]').click()
            _enable(page, 'docs_http')
            sse_card = _enable(page, 'docs_sse')
            _edit(sse_card)
            sse_card.locator('[data-mcp-field="allowed_tools"]').fill('echo')
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
            # Showing Settings loads it once more and redraws every row: open the form on the settled one.
            card = _card(page, 'docs_http')
            _edit(card)
            value = card.locator('[data-mcp-header-row="0"] [data-mcp-header-field="value"]')
            assert value.input_value() == "***set***"
            card.locator('[data-mcp-header-row="0"] [data-mcp-header-toggle]').click()
            shown = card.locator('[data-mcp-header-row="0"] .settings-secret-value')
            shown.wait_for(state="visible", timeout=10_000)
            assert shown.inner_text() == BEARER and value.input_value() == "***set***"
            _capture(page, "mcp-headers-reloaded-show")
            card.locator('[data-mcp-header-row="0"] [data-mcp-header-toggle]').click()

            card.locator("[data-mcp-refresh]").click()
            card.locator("[data-mcp-message]").filter(has_text="Listed again.").wait_for(timeout=20_000)
            sse_card = _card(page, 'docs_sse')
            sse_card.locator("[data-mcp-test]").click()
            sse_card.locator("[data-mcp-message]").filter(has_text="The connection works.").wait_for(timeout=20_000)
            assert "No tool was called" in sse_card.locator("[data-mcp-message]").inner_text()
            sse_card.locator("[data-mcp-message]").scroll_into_view_if_needed()
            _capture(page, "mcp-refresh-and-test-draft")
            _edit(sse_card)
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
            _list(page)
            page.locator('[data-tools-action="tools:import"]').click()
            dialog.locator("#mcp-import-text").fill(json.dumps(update))
            dialog.locator("[data-mcp-import-preview-btn]").click()
            dialog.locator(".mcp-import-entry").nth(2).wait_for(timeout=15_000)
            preview_text = dialog.inner_text()
            assert preview_text.count("Update") == 2 and "Not applied" in preview_text
            assert "remove that server first" in preview_text
            _capture(page, "mcp-import-update-preview")
            _apply_import(dialog)
            updated = {entry["id"]: entry for entry in json.loads(
                (data_dir / "settings.json").read_text(encoding="utf-8"))["MCP_SERVERS"]}
            assert updated["docs_sse"]["headers"] == {"Authorization": BASIC, "X-Api-Key": CUSTOM}
            assert updated["docs_sse"]["enabled"] is True and updated["docs_sse"]["allowed_tools"] == ["echo"]
            assert updated["local_files"]["args"] == ["-m", "synthetic_server_v2"]
            assert updated["local_files"]["enabled"] is False and updated["local_files"]["env"] == {"DEBUG": "1"}
            assert updated["docs_http"] == servers["docs_http"]  # the refused entry changed nothing

            # The ordinary form is a separate input path, not just an import renderer.
            _list(page)
            page.click('[data-tools-add]')
            manual = page.locator('[data-mcp-card]')
            manual.locator('[data-mcp-field="id"]').fill('manual_docs')
            manual.locator('[data-mcp-field="url"]').fill(http_url)
            _edit(manual)
            for number, (name, literal) in enumerate({'Authorization': BEARER, 'X-Api-Key': CUSTOM}.items()):
                manual.locator('[data-mcp-header-add]').click()
                manual = page.locator('[data-mcp-card]')
                manual.locator(f'[data-mcp-header-row="{number}"] [data-mcp-header-field="name"]').fill(name)
                manual.locator(f'[data-mcp-header-row="{number}"] [data-mcp-header-field="value"]').fill(literal)
            manual.locator('[data-mcp-field="enabled"]').locator('xpath=..').click()
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
            _edit(manual)
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
            page.click('[data-tools-add]')
            card = page.locator('[data-mcp-card]').last
            card.locator('[data-mcp-field="id"]').fill('switch_headers')
            card.locator('[data-mcp-field="url"]').fill('https://example.com/mcp')
            _edit(card)
            card.locator('[data-mcp-header-add]').click()
            card = page.locator('[data-mcp-card]').last
            card.locator('[data-mcp-header-field="name"]').fill('X-Api-Key')
            card.locator('[data-mcp-header-field="value"]').fill(CUSTOM)
            if saved_headers:
                _save(page)
                page.reload(wait_until='domcontentloaded')
                _open_mcp(page, url)
                card = _card(page, 'switch_headers')
                _edit(card)
                assert card.locator('[data-mcp-header-field="value"]').input_value() == '***set***'
                card.locator('[data-mcp-header-field="value"]').fill('synthetic-edited-key')
            card.locator('[data-mcp-field="transport"]').select_option('stdio')
            card = page.locator('[data-mcp-card]').last
            card.locator('[data-mcp-field="command"]').fill('python3')
            remove = card.locator('[data-mcp-clear-unsupported]')
            assert remove.is_visible(), 'Draft-only header must have an explicit removal control'
            assert 'headers' in card.locator('[data-mcp-unsupported]').inner_text()
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
            _edit(card)
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
            open_settings(page, 'tools/new')
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
            page.keyboard.insert_text('on3')
            assert page.evaluate('() => document.activeElement === window.__typedField'
                                 ' && window.__typedField.isConnected'), 'the status answer replaced the typed field'
            assert command.input_value() == 'python3'
            command.scroll_into_view_if_needed()
            _capture(page, 'mcp-late-status-typed')
            page.unroute('**/api/mcp/status')
            _save(page)
            assert _disk(data_dir)['typed_draft']['command'] == 'python3'
            saved = _card(page, 'late_status')
            expect(saved.locator('[data-mcp-state]')).to_have_text(re.compile(r'^1 tool · listed .+$'))
            assert 'echo' in saved.locator('.mcp-tools-disclosure').text_content()
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
            message, badge = card.locator('[data-mcp-message]'), card.locator('[data-mcp-state]')

            def answer(change=None, focused=False, expected='Listed again.'):
                """Refresh without moving focus, then answer its held status read, changed or not."""
                with page.expect_request(lambda request: request.url.endswith('/api/mcp/status'), timeout=20_000):
                    button = card.locator('[data-mcp-refresh]')
                    button.evaluate('button => button.click()') if focused else button.click()
                expect(message).to_contain_text('Listing its tools again...')
                for route in held:
                    response = route.fetch()
                    data = response.json()
                    for entry in data['servers']:
                        if change and entry.get('id') == 'tools_focus':
                            change(entry)
                    route.fulfill(response=response, json=data)
                held.clear()
                expect(message).to_contain_text(expected, timeout=20_000)

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
            expect(badge).to_have_text(re.compile(r'^1 tool · listed [^·]+$'))
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
            expect(badge).to_have_text(re.compile(r'^2 tools · listed [^·]+$'))
            assert card.locator('.mcp-tools-list li').count() == 2
            assert 'synthetic_lookup' in card.locator('.mcp-tools-list').text_content()
            assert {key: value for key, value in kept().items() if key != 'selected'} == {
                'same': True, 'open': True, 'focused': True, 'field': True}, 'a changed tool list must patch in place'

            answer(lambda entry: entry.update(tools=[], tool_count=0), focused=True)
            expect(badge).to_have_text(re.compile(r'^0 tools · listed [^·]+$'))
            assert card.locator('.mcp-tools-disclosure').count() == 0

            # The chip says the state with a relative age; the server page has the exact stamps and engine's text.
            def stamp(iso):
                return page.evaluate("async (iso) => (await import('/static/modules/utils.js')).absoluteTime(iso)", iso)

            expect(card.locator('[data-mcp-facts]')).to_be_visible()
            answer(lambda entry: entry.update(last_error='offline', latest_attempt={
                'at': '2026-10-21T10:00:00+00:00', 'ok': False, 'error': 'offline', 'partial': False}))
            expect(badge).to_have_text(re.compile(r'^1 tool · listed [^·]+ · listing failed [^·]+$'))
            assert not re.search(r'2026-10-21|offline', badge.inner_text()), badge.inner_text()
            facts = _facts(card)
            assert facts['Listed'] and (facts['Last attempt'], facts['Error']) == (
                f"{stamp('2026-10-21T10:00:00+00:00')} · failed", 'offline'), facts
            badge.scroll_into_view_if_needed()
            _capture(page, 'mcp-catalog-failed-with-tools')
            answer(lambda entry: entry.update(tools=[], tool_count=0, listed_at=None, latest_attempt=None,
                                               last_refreshed='', last_attempted='', last_error=''))
            expect(badge).to_have_text('Not listed yet')
            answer(lambda entry: entry.update(tools=[], tool_count=0, listed_at=None, last_error='offline',
                latest_attempt={'at': '2026-10-21T10:00:00+00:00', 'ok': False, 'error': 'offline', 'partial': False}))
            expect(badge).to_have_text(re.compile(r'^Listing failed [^·]+$'))
            assert _facts(card) == {'Last attempt': f"{stamp('2026-10-21T10:00:00+00:00')} · failed",
                                    'Error': 'offline'}, _facts(card)  # never listed: no Listed stamp
            badge.scroll_into_view_if_needed()
            _capture(page, 'mcp-catalog-never-listed')
            page.set_viewport_size({'width': 390, 'height': 844})
            badge.scroll_into_view_if_needed()
            expect(badge).to_be_visible()
            bounds = badge.bounding_box()
            assert bounds and bounds['x'] >= 0 and bounds['x'] + bounds['width'] <= 390
            _capture(page, 'mcp-catalog-never-listed-narrow')
            page.route('**/api/mcp/refresh', lambda route: route.fulfill(json={
                'ok': True, 'published': False, 'tool_count': 1, 'publication_error': 'disk full'}))
            answer(lambda entry: entry.update(publication_error='disk full'), expected='Refresh failed: disk full')
            expect(badge).to_have_text(re.compile(r'^1 tool · listed [^·]+ · catalog not published$'))
            assert _facts(card)['Catalog publication'] == 'disk full', _facts(card)
            assert 'Listed again.' not in message.inner_text()
            page.unroute('**/api/mcp/refresh')
            page.route('**/api/mcp/refresh', lambda route: route.fulfill(json={'ok': False, 'error': 'offline'}))
            answer(lambda entry: entry.update(latest_attempt={'at': '2026-10-21T11:00:00+00:00',
                'ok': False, 'error': 'offline', 'partial': False}), expected='Refresh failed: offline')
            expect(badge).to_have_text(re.compile(r'^1 tool · listed [^·]+ · listing failed [^·]+$'))
            facts = _facts(card)
            assert (facts['Last attempt'], facts['Error']) == (f"{stamp('2026-10-21T11:00:00+00:00')} · failed",
                                                               'offline'), facts
            page.unroute('**/api/mcp/refresh')
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
            _edit(card)
            card.locator('[data-mcp-header-field="value"]').fill('Bearer synthetic-edited-0006')
            card.locator('[data-mcp-field="transport"]').select_option('stdio')
            card = page.locator('[data-mcp-card]').last
            card.locator('[data-mcp-field="command"]').fill('python3')
            notice = card.locator('[data-mcp-unsupported]').inner_text()
            card.locator('[data-mcp-clear-unsupported]').click()
            card = page.locator('[data-mcp-card]').last
            remaining = card.locator('[data-mcp-unsupported]').all_inner_texts()
            _capture(page, f'mcp-legacy-stdio-removed-{legacy_header}')
            _save(page)
            entry = _disk(data_dir)['legacy_switch']
            errors = []
            assert normalize_server_config(entry, errors=errors) is not None, errors
            assert BEARER not in json.dumps(entry) and 'synthetic-edited' not in json.dumps(entry)
            assert (entry['transport'], entry['command'], entry.get('auth_token', ''),
                    entry.get('auth_header', 'Authorization'), 'headers' in entry) == (
                'stdio', 'python3', '', 'Authorization', False)
            assert notice == 'Fields kept but not used: url, headers', notice
            assert remaining == [], remaining
            _edit(page.locator('[data-mcp-card]').last)
            page.locator('[data-mcp-card]').last.locator('[data-mcp-field="transport"]').select_option('streamable_http')
            assert page.locator('[data-mcp-card]').last.locator('[data-mcp-header-row]').count() == 0
        finally:
            browser.close()


TENANT = "synthetic-browser-tenant-0003"
ALPHA = "synthetic-browser-alpha-0004"
BETA = "synthetic-browser-beta-0005"


def _seed(page, url, servers):
    """Write through the real Settings writer, then load the page from it and let its loads settle."""

    page.goto(url, wait_until="domcontentloaded", timeout=30_000)
    status = page.evaluate("""async (servers) => {
        const response = await fetch('/api/settings', {method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({MCP_ENABLED: true, MCP_SERVERS: servers})});
        return [response.status, await response.text()];
    }""", servers)
    assert status[0] == 200, status
    _open_mcp(page, url)
    first = page.locator('[data-settings-page="tools"] [data-entry-id]').first
    _settings_ready(page, lambda: open_entry(page, first.get_attribute('data-entry-id')))


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


def test_stdio_task_session_choice(direct_server_with_data):  # noqa: F811
    """Opt an existing server in and out through Settings; preserve an imported browser policy."""
    from playwright.sync_api import expect, sync_playwright

    url, data_dir = direct_server_with_data['url'], direct_server_with_data['data_dir']
    servers = [
        {'id': 'local_browser', 'transport': 'stdio', 'command': 'python3', 'enabled': False},
        {'id': 'guarded_browser', 'transport': 'stdio', 'command': 'python3', 'enabled': False,
         'task_session': True, 'browser_policy': True},
        {'id': 'remote', 'transport': 'sse', 'url': 'http://127.0.0.1:9/mcp', 'enabled': False},
    ]  # Disabled fixtures never launch a helper or contact an endpoint.
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page(viewport={'width': 1440, 'height': 1000})
        page._evidence_engine = 'chromium'
        try:
            _seed(page, url, servers)
            card = _card(page, 'local_browser')
            choice = card.locator('[data-mcp-field="task_session"]')
            expect(choice).not_to_be_checked()
            _edit(card)
            choice.scroll_into_view_if_needed()
            _capture(page, 'mcp-task-session-off')
            choice.check()
            expect(choice).to_be_focused()
            expect(card.locator('[data-mcp-lifetime]')).to_contain_text('One server process serves every call in a task')
            _capture(page, 'mcp-task-session-on')
            _save(page)
            assert _disk(data_dir)['local_browser']['task_session'] is True
            _open_mcp(page, url)
            card = _card(page, 'local_browser')
            expect(choice).to_be_checked()
            _edit(card)
            choice.uncheck()
            _save(page)
            assert _disk(data_dir)['local_browser']['task_session'] is False
            card = _card(page, 'guarded_browser')
            _edit(card)
            guarded = card.locator('[data-mcp-field="task_session"]')
            expect(guarded).to_be_checked()
            expect(guarded).to_be_disabled()
            expect(card.locator('[data-mcp-lifetime]')).to_contain_text('requires a task session')
            assert _disk(data_dir)['guarded_browser']['browser_policy'] is True
            assert _card(page, 'remote').locator('[data-mcp-field="task_session"]').count() == 0
        finally:
            browser.close()


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
            legacy = _card(page, 'legacy_docs')
            # The legacy pair is one ordinary row with its own prefix mask; no second editor.
            assert _rows(legacy) == [("Authorization", BEARER[:8] + "..."), ("X-Tenant", "***set***")]
            assert page.locator('[data-mcp-field="auth_token"], [data-mcp-field="auth_header"]').count() == 0
            _edit(legacy)
            legacy.locator('[data-mcp-header-row="0"]').scroll_into_view_if_needed()
            _capture(page, "mcp-unified-legacy-row")
            legacy.locator('[data-mcp-header-row="0"] [data-mcp-header-toggle]').click()
            expect(legacy.locator('[data-mcp-header-row="0"] .settings-secret-value')).to_have_text(BEARER)
            legacy.locator('[data-mcp-header-row="0"] [data-mcp-header-toggle]').click()

            # A row's unrelated commit sends no servers or credentials.
            open_settings(page, 'tools', 'MCP_TOOL_TIMEOUT_SEC')
            assert commit_row(page, 'MCP_TOOL_TIMEOUT_SEC', 45).startswith('Saved')
            assert posts[-1] == {'MCP_TOOL_TIMEOUT_SEC': 45}
            disk = _disk(data_dir)
            assert {key: disk["legacy_docs"][key] for key in ("auth_header", "auth_token", "headers")} == {
                "auth_header": "Authorization", "auth_token": BEARER, "headers": {"X-Tenant": TENANT}}
            assert disk["pair"]["headers"] == {"X-A": ALPHA, "X-B": BETA}
            assert disk["Name Only"]["headers"] == {"X-Api-Key": CUSTOM}

            # A server known by its name alone: Test and Refresh use that saved identity.
            named = _card(page, "name_only")
            named.locator("[data-mcp-test]").click()
            expect(named.locator("[data-mcp-message]")).to_contain_text("The connection works.", timeout=20_000)
            assert tests[-1]["server_id"] == "Name Only" and tests[-1]["server"]["headers"] == {"X-Api-Key": "***set***"}
            named.locator("[data-mcp-refresh]").click()
            expect(named.locator("[data-mcp-message]")).to_contain_text("Listed again.", timeout=20_000)

            # Save this entry once to materialize its form defaults; unrelated row saves never do that.
            legacy = _card(page, "legacy_docs")
            _save(page)
            # Another spelling of the same canonical server ID keeps every saved value for Test and Save.
            _edit(legacy)
            legacy.locator('[data-mcp-field="id"]').fill("Legacy Docs")
            legacy.locator("[data-mcp-test]").click()
            expect(legacy.locator("[data-mcp-message]")).to_contain_text("The connection works.", timeout=20_000)
            assert tests[-1]["server_id"] == "legacy_docs" and tests[-1]["server"]["id"] == "Legacy Docs"
            assert legacy.locator('[data-editor-result]').get_attribute('data-tone') != 'danger'
            saved_before = (data_dir / "settings.json").read_text(encoding="utf-8")
            with page.expect_response(lambda response: response.url.endswith('/api/settings')
                                      and response.request.method == 'POST', timeout=20_000) as renamed_save:
                page.locator("[data-editor-save]").click()
            # Both spellings are one canonical ID, so this Save is an honest no-op that changes nothing on disk.
            answer = renamed_save.value.json()
            assert renamed_save.value.ok and answer["status"] == "saved" and answer.get("no_changes") is True, answer
            assert renamed_save.value.request.post_data_json["MCP_SERVERS"][0]["id"] == "Legacy Docs"
            expect(legacy.locator('[data-editor-save]')).to_be_enabled()
            expect(legacy.locator('[data-editor-result]')).to_have_text('')  # unchanged is quiet
            assert json.loads((data_dir / "settings.json").read_text(encoding="utf-8")) == json.loads(saved_before)
            disk = _disk(data_dir)
            assert disk["legacy_docs"]["auth_token"] == BEARER and disk["legacy_docs"]["headers"] == {"X-Tenant": TENANT}
            _open_mcp(page, url)
            legacy = _card(page, "legacy_docs")
            expect(legacy.locator('[data-mcp-field="id"]')).to_have_value("legacy_docs")
            assert _rows(legacy) == [("Authorization", BEARER[:8] + "..."), ("X-Tenant", "***set***")]

            # Saved A and B: remove B, rename masked A to B. Nothing substitutes B's value.
            pair = _card(page, "pair")
            _edit(pair)
            pair.locator('[data-mcp-header-row="1"] [data-mcp-header-remove]').click()
            pair.locator('[data-mcp-header-row="0"] [data-mcp-header-field="name"]').fill("X-B")
            before = len(posts)
            page.locator("[data-editor-save]").click()
            error = pair.locator('[data-editor-result]')
            expect(error).to_contain_text("belongs to header X-A", timeout=10_000)
            expect(pair.locator('[data-mcp-header-row="0"] [data-mcp-header-field="value"]')).to_be_focused()
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
            _edit(pair)
            pair.locator("[data-mcp-header-add]").click()
            pair.locator('[data-mcp-header-row="1"] [data-mcp-header-field="name"]').fill("X-Dup")
            _save(page)
            assert _disk(data_dir)["pair"]["headers"] == {"x-a": ALPHA, "X-Dup": ""}
            _edit(pair)
            pair.locator("[data-mcp-header-add]").click()
            pair.locator('[data-mcp-header-row="2"] [data-mcp-header-field="name"]').fill("X-Dup")
            page.locator("[data-editor-save]").click()
            expect(pair.locator('[data-editor-result]')).to_contain_text("repeats header X-Dup")
            pair.locator('[data-mcp-header-row="2"] [data-mcp-header-remove]').click()

            # Re-typing the mask is an edit. The server's form keeps it across navigation and refuses it.
            pair.locator('[data-mcp-header-row="0"] [data-mcp-header-field="value"]').fill('synthetic-other')
            pair.locator('[data-mcp-header-row="0"] [data-mcp-header-field="value"]').fill('***set***')
            _list(page)
            pair = _card(page, 'pair')
            _edit(pair)
            assert pair.locator('[data-mcp-header-row="0"] [data-mcp-header-field="value"]').input_value() == '***set***'
            before = len(posts)
            page.locator("[data-editor-save]").click()
            expect(pair.locator('[data-editor-result]')).to_contain_text("reserved for keeping a saved value")
            assert len(posts) == before
            pair.locator('[data-mcp-header-row="0"]').scroll_into_view_if_needed()
            _capture(page, "mcp-retyped-mask-refused")

            # A deliberate address change keeps disclosed values: origins only, never path or query.
            legacy = _card(page, "legacy_docs")
            _edit(legacy)
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
            named = _card(page, "name_only")
            held = []
            page.route("**/api/mcp/test", lambda route: held.append(route))
            named.locator("[data-mcp-test]").click()
            page.wait_for_function("() => true")
            expect(named.locator("[data-mcp-message]")).to_contain_text("Testing the connection...")
            _edit(named)
            named.locator('[data-mcp-field="allowed_tools"]').fill("echo")
            page.wait_for_timeout(200)
            held[0].fulfill(response=held[0].fetch())
            expect(named.locator("[data-mcp-message]")).to_contain_text("changed while it was tested", timeout=20_000)
            page.unroute("**/api/mcp/test")
            held.clear()
            page.route("**/api/mcp/refresh", lambda route: held.append(route))
            legacy = _card(page, "legacy_docs")
            legacy.locator("[data-mcp-refresh]").click()
            page.wait_for_timeout(200)
            assert len(held) == 1
            legacy.locator("[data-editor-remove]").click()
            page.locator(".confirm-dialog-backdrop").get_by_role("button", name="Remove", exact=True).click()
            with page.expect_response(lambda response: response.url.endswith("/api/mcp/status")):
                held[0].fulfill(response=held[0].fetch())
            page.wait_for_timeout(300)
            expect(page.locator('[data-settings-page="tools"] [data-entry-id]')).to_have_count(2)
            named = _card(page, 'name_only')
            assert 'Listed again.' not in named.locator('[data-mcp-message]').inner_text()
            page.unroute("**/api/mcp/refresh")
        finally:
            browser.close()
    assert {headers["authorization"] for headers in http_app.seen} == {BEARER}
    assert {headers["x-api-key"] for headers in sse_app.seen} == {CUSTOM}


@pytest.mark.parametrize('engine', ['chromium', 'webkit'])
def test_mcp_import_replaces_legacy_ui(direct_server_with_data, loopback, engine):  # noqa: F811
    """A file's headers replace the whole list, the saved single token included; Apply saves the selected entries."""
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
            unselected_before = _disk(data_dir)["keep_me"]
            previews = []
            page.on("request", lambda request: previews.append(request.post_data_json)
                    if request.url.endswith("/api/mcp/import/preview") else None)
            _list(page)
            page.locator('[data-tools-action="tools:import"]').click()
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
            expect(dialog.locator("[data-mcp-import-apply]")).to_have_text("Apply 2")
            _apply_import(dialog)
            assert page.locator('[data-settings-page="tools"] [data-entry-id]').count() == 3
            assert _rows(_card(page, 'legacy_docs')) == [('Authorization', '***set***'), ('X-Api-Key', '***set***')]
            assert _rows(_card(page, 'keep_me')) == [('X-Keep', '***set***')]
            disk = _disk(data_dir)
            assert disk["legacy_docs"]["headers"] == {"Authorization": BASIC, "X-Api-Key": CUSTOM}
            assert disk["legacy_docs"]["auth_token"] == "" and disk["legacy_docs"]["enabled"] is True
            # The settings reader materializes an absent legacy token as empty (secret_rows.py).
            assert disk['keep_me'] == {**unselected_before, 'auth_token': ''}
            assert disk["fresh"]["enabled"] is False and disk["fresh"]["url"].endswith("?k=synthetic-query")
            card = _card(page, 'legacy_docs')
            seen_before = len(http_app.seen)
            card.locator("[data-mcp-refresh]").click()
            expect(card.locator("[data-mcp-message]")).to_contain_text("Listed again.", timeout=20_000)
            _edit(card)
            for theme in ("light", "dark"):
                page.set_viewport_size({"width": 1440, "height": 1000})
                page.evaluate("(theme) => window.ouroTheme.set(theme)", theme)
                page.locator(f'html[data-theme="{theme}"]').wait_for(state="attached")
                card.scroll_into_view_if_needed()
                _capture(page, f"mcp-headers-{theme}")
                page.set_viewport_size({"width": 390, "height": 844})
                card.locator('[data-mcp-header-row="1"]').scroll_into_view_if_needed()
                row = card.locator('[data-mcp-header-row="1"]').bounding_box()
                assert row["x"] >= 0 and row["x"] + row["width"] <= 390, row
                _capture(page, f"mcp-headers-{theme}-narrow")
        finally:
            browser.close()
    # Before Apply the old single token was still the saved credential; after it, only the file's values.
    assert http_app.seen[seen_before:] and {headers["authorization"] for headers in http_app.seen[seen_before:]} == {BASIC}


@pytest.mark.parametrize('engine', ['chromium', 'webkit'])
@pytest.mark.parametrize('imported', ['empty', 'map', 'absent'])
def test_removed_legacy_then_import(direct_server_with_data, loopback, engine, imported):  # noqa: F811
    """Save removal of the single token, then import the same server: Import and Test keep it gone.

    Import writes from a fresh saved list. Each server form saves its own changes before that list action.
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
            _edit(card)
            card.locator('[data-mcp-header-row="0"] [data-mcp-header-remove]').click()
            _save(page)
            _list(page)
            page.locator('[data-tools-action="tools:import"]').click()
            dialog = page.locator(".mcp-import-dialog")
            dialog.locator("#mcp-import-text").fill(json.dumps({"mcpServers": {"legacy_docs": entry}}))
            dialog.locator("[data-mcp-import-preview-btn]").click()
            dialog.locator(".mcp-import-entry").first.wait_for(timeout=15_000)
            assert "Update" in dialog.inner_text() and "Authorization" not in dialog.inner_text()
            _apply_import(dialog)
            card = _card(page, 'legacy_docs')
            _edit(card)
            rows = [(name, "***set***") for name, value in expected.items()]
            assert _rows(card) == rows, _rows(card)
            card.locator('[data-mcp-header-row="0"]' if rows else '[data-mcp-test]').scroll_into_view_if_needed()
            _capture(page, f"mcp-removed-legacy-import-{imported}")
            seen = len(http_app.seen)
            card.locator("[data-mcp-test]").click()
            expect(card.locator("[data-mcp-message]")).to_contain_text("The connection works.", timeout=20_000)
            assert tests[-1]["server"].get("auth_token", "") == ""
            assert http_app.seen[seen:] and all("authorization" not in headers for headers in http_app.seen[seen:])
            disk = _disk(data_dir)["legacy_docs"]
            assert (disk.get("auth_token", ""), disk["headers"]) == ("", expected)
            assert BEARER not in json.dumps(disk)
            page.reload(wait_until="domcontentloaded")
            _open_mcp(page, url)
            assert _rows(_card(page, "legacy_docs")) == [(name, "***set***") for name in expected]
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
                _edit(page.locator('[data-mcp-card]').first)
                field = page.locator('#mcp-0-header-0-value')
                mask = field.input_value()
                assert mask != token
                posts = []
                handler = lambda request: posts.append(request) if request.url.endswith('/api/settings') and request.method == 'POST' else None
                page.on('request', handler)
                field.fill('synthetic-other')
                field.fill(mask)
                page.locator('[data-editor-save]').click()
                expect(page.locator('[data-editor-result]')).to_contain_text("saved token's mask")
                expect(field).to_be_focused()
                assert posts == []
                assert _disk(data_dir)['legacy_mask']['auth_token'] == token
                field.fill('synthetic-new-credential')
                _save(page)
                saved = _disk(data_dir)['legacy_mask']
                assert saved['auth_token'] == '' and saved['headers'] == {'X-Legacy': 'synthetic-new-credential'}
                page.remove_listener('request', handler)
        finally:
            browser.close()


@pytest.fixture(autouse=True)
def _private_mcp_catalog_root(tmp_path, monkeypatch):
    from ouroboros import config
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
