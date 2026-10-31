"""Static UI invariants for the MCP Servers widget.

These tests do not launch a browser — they read ``settings_ui.js``,
``settings.js``, ``mcp_settings.js``, and ``settings.css`` and assert
the structural facts the implementation relies on (element ids, CSS
class names, key API endpoints, secret masking, etc.).
"""

from __future__ import annotations

import pathlib
import re
import shutil
import subprocess

import pytest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
WEB = REPO_ROOT / "web" / "modules"


def _node_bin():
    bundled = pathlib.Path.home() / ".claudexor" / "node" / "bin" / "node"
    return str(bundled if bundled.exists() else pathlib.Path(shutil.which("node") or "node"))


@pytest.fixture(scope="module")
def settings_ui_source() -> str:
    return (WEB / "settings_ui.js").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def settings_source() -> str:
    return (WEB / "settings.js").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def mcp_source() -> str:
    return (WEB / "mcp_settings.js").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def settings_css() -> str:
    return (REPO_ROOT / "web" / "settings.css").read_text(encoding="utf-8")


def test_settings_ui_has_mcp_section(settings_ui_source: str) -> None:
    assert "<h3>MCP Servers</h3>" in settings_ui_source
    assert 'id="s-mcp-enabled"' in settings_ui_source
    assert 'id="s-mcp-tool-timeout"' in settings_ui_source
    assert 'id="btn-mcp-add-server"' in settings_ui_source
    assert 'id="btn-mcp-refresh-all"' in settings_ui_source
    assert 'id="mcp-servers-list"' in settings_ui_source
    assert 'id="mcp-global-status"' in settings_ui_source
    assert "local stdio processes" in settings_ui_source


def test_settings_ui_mcp_section_sits_in_advanced_panel(settings_ui_source: str) -> None:
    """MCP lives in Advanced; the old Integrations tab remains retired."""
    assert "Integrations" not in settings_ui_source
    advanced_index = settings_ui_source.find('data-settings-panel="advanced"')
    about_index = settings_ui_source.find('data-settings-panel="about"')
    mcp_index = settings_ui_source.find("<h3>MCP Servers</h3>")
    assert advanced_index >= 0 and about_index >= 0 and mcp_index >= 0
    assert advanced_index < mcp_index < about_index


def test_settings_js_imports_mcp_module(settings_source: str) -> None:
    assert "from './mcp_settings.js'" in settings_source
    assert "applyMcpSettings" in settings_source
    assert "collectMcpSettings" in settings_source
    assert "initMcpSettings" in settings_source


def test_settings_js_includes_mcp_in_collect_body(settings_source: str) -> None:
    assert "...collectMcpSettings()" in settings_source


def test_mcp_module_uses_mcp_endpoints(mcp_source: str) -> None:
    assert "/api/mcp/status" in mcp_source
    assert "/api/mcp/refresh" in mcp_source
    assert "/api/mcp/test" in mcp_source


def test_mcp_module_drops_masked_token_in_test_payload(mcp_source: str) -> None:
    """A masked token must not be sent as a Bearer credential when the user
    hits Test connection without re-typing it, unless paired with server_id
    so the backend can rehydrate the persisted token."""
    assert "looksMasked" in mcp_source
    assert "out.auth_token = ''" in mcp_source
    assert "{ server_id: saved.identity, server: candidate }" in mcp_source


def test_mcp_module_supports_http_sse_and_stdio(mcp_source: str) -> None:
    assert "streamable_http" in mcp_source
    assert "sse" in mcp_source
    assert "stdio" in mcp_source
    assert "data-mcp-field=\"command\"" in mcp_source
    assert "data-mcp-field=\"args\"" in mcp_source
    assert "does not use a shell" in mcp_source


def test_mcp_module_renders_status_classes(mcp_source: str) -> None:
    assert "mcp-server-status-ok" in mcp_source
    assert "mcp-server-status-warn" in mcp_source
    assert "mcp-server-status-danger" in mcp_source


def test_mcp_module_escapes_untrusted_strings(mcp_source: str) -> None:
    """Tool descriptions are server-supplied untrusted data; the renderer
    must HTML-escape them before injecting into the DOM."""
    assert "escapeHtmlAttr as escapeHtml" in mcp_source
    # The renderer must call escapeHtml on at least the tool name and
    # description fields it interpolates from server-provided data.
    assert "escapeHtml(t.name" in mcp_source
    assert "escapeHtml(String(t.description)" in mcp_source


def test_mcp_css_defines_required_classes(settings_css: str) -> None:
    assert ".mcp-servers-list" in settings_css
    assert ".mcp-server-card" in settings_css
    assert ".mcp-server-status-ok" in settings_css
    assert ".mcp-server-status-danger" in settings_css
    # DESIGN §2: a subsection heading is --type-body semibold, never the browser's bold h4.
    heading = re.search(r"\.mcp-headers-fields h4 \{([^}]*)\}", settings_css)
    assert heading and "font-size: var(--type-body);" in heading[1] and "font-weight: 600;" in heading[1]
    shared = (REPO_ROOT / "web" / "ui.css").read_text(encoding="utf-8")
    assert ".ui-control {" in shared and "textarea.ui-control" in shared
    source = (WEB / "mcp_settings.js").read_text(encoding="utf-8")
    assert '<textarea class="ui-control"' in source
    assert 'class="form-field ui-field"' in source


def test_mcp_ui_roundtrip_preserves_environment_and_unsupported_fields():
    """Execute the real JS projection; this is not a browser/visual receipt."""
    node = _node_bin()
    if not node:
        pytest.skip("Node is unavailable")
    script = r'''
import assert from 'node:assert/strict';
import { applyMcpSettings, collectMcpSettings } from './web/modules/mcp_settings.js';
const host = { innerHTML: '', querySelectorAll: () => [] };
globalThis.document = { getElementById: (id) => id === 'mcp-servers-list' ? host : null };
globalThis.fetch = async () => ({ ok: false });
const server = { id: 'selected', transport: 'stdio', command: 'python',
    args: ['exact argument', ''], enabled: true, cwd: '/project/"quoted"',
    env: { PORT: '8080' }, env_from_settings: { TOKEN: 'CUSTOM_MCP_KEY' }, future_field: { retained: true },
    url: 'https://unsupported-for-stdio.example/mcp' };
applyMcpSettings({ MCP_ENABLED: true, MCP_SERVERS: [{ ...server, auth_configured: false }] });
const saved = collectMcpSettings().MCP_SERVERS[0];
for (const key of Object.keys(server)) assert.deepEqual(saved[key], server[key]);
assert.match(host.innerHTML, /data-mcp-field="cwd"/);
assert.match(host.innerHTML, /data-mcp-field="env_from_settings"/);
assert.match(host.innerHTML, /data-mcp-field="env"/);
assert.match(host.innerHTML, /Fields retained but not applied:.*future_field/);
assert.ok(!host.innerHTML.includes('auth_configured'));
assert.ok(!Object.hasOwn(saved, 'auth_configured'));
assert.ok(!host.innerHTML.includes('value="/project/"quoted""'));
applyMcpSettings({ MCP_SERVERS: [{ ...server, env_from_settings: '{unfinished', args: 'unsupported' }] });
assert.equal(collectMcpSettings().MCP_SERVERS[0].env_from_settings, '{unfinished');
assert.equal(collectMcpSettings().MCP_SERVERS[0].args, 'unsupported');
'''
    result = subprocess.run([node, "--input-type=module", "-e", script], cwd=REPO_ROOT,
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr


def test_mcp_test_button_preserves_saved_url_only_credentials():
    """Exercise the registered click handler, not a copy of its payload expression."""
    node = _node_bin()
    if not node:
        pytest.skip("Node is unavailable")
    script = r'''
import assert from 'node:assert/strict';
import { applyMcpSettings } from './web/modules/mcp_settings.js';
let click, requests = [];
const message = { hidden: true, dataset: {} };
const button = { disabled: false, addEventListener: (_kind, handler) => { click = handler; } };
const card = { dataset: { mcpIndex: '0' }, querySelectorAll: () => [], querySelector: (selector) =>
    selector === '[data-mcp-test]' ? button : selector === '[data-mcp-message]' ? message : null };
const host = { innerHTML: '', querySelectorAll: (selector) => selector === '[data-mcp-card]' ? [card] : [] };
globalThis.document = { getElementById: (id) => id === 'mcp-servers-list' ? host : null,
    querySelector: (selector) => selector === '[data-mcp-index="0"]' ? card : null };
globalThis.fetch = async (url, options) => {
    if (url === '/api/mcp/status') return { ok: false };
    assert.equal(url, '/api/mcp/test');
    requests.push(JSON.parse(options.body));
    return { ok: true, json: async () => ({ ok: true, tool_count: 1 }) };
};
for (const [server, selected] of [
    [{ id: 'saved', url: 'https://***@host.test/mcp', auth_token: '' }, 'saved'],
    [{ id: 'saved', url: 'https://host.test/mcp', auth_token: '***' }, 'saved'],
    // A saved server identified by its name alone still names its saved identity.
    [{ name: 'Name Only', url: 'https://host.test/mcp', headers: { 'X-Key': '***set***' } }, 'Name Only'],
    [{ id: 'new', url: 'https://new:credential@host.test/mcp', auth_token: '' }, undefined],
    [{ id: '', url: 'https://host.test/mcp', auth_token: '' }, undefined],
]) {
    applyMcpSettings({ MCP_SERVERS: [server] });
    await click();
    const body = requests.at(-1);
    assert.equal(body.server.url, server.url);
    assert.equal(body.server_id, selected);
    assert.match(message.textContent, /Draft connection OK/);
    assert.equal(button.disabled, false);
}
'''
    result = subprocess.run([node, "--input-type=module", "-e", script], cwd=REPO_ROOT,
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr


def test_mcp_headers_ui_preserves_malformed_rows_and_literal_values():
    """Execute the actual draft projection, including entries hidden from the editor."""
    script = r'''
import assert from 'node:assert/strict';
import { applyMcpSettings, collectMcpSettings } from './web/modules/mcp_settings.js';
const host = {innerHTML:'', querySelectorAll:()=>[]};
globalThis.document = {getElementById:(id)=> id==='mcp-servers-list'?host:null};
globalThis.fetch = async()=>({ok:false});
const entries = [null, 'malformed legacy row', {id:'old', headers:'***set***'},
 {id:'multi', headers:{Authorization:'***set***','X-Key':'literal...','X-Empty':''}, future_field:{kept:true}}];
applyMcpSettings({MCP_SERVERS: entries});
const out = collectMcpSettings().MCP_SERVERS;
assert.deepEqual(out.slice(0,2), entries.slice(0,2));
assert.equal(out[2].headers, entries[2].headers);
assert.deepEqual(out[3].headers, entries[3].headers);
assert.deepEqual(out[3].future_field, entries[3].future_field);
assert.match(host.innerHTML, /malformed and retained unchanged/);
assert.match(host.innerHTML, /Saved headers are malformed/);
'''
    result = subprocess.run([_node_bin(), '--input-type=module', '-e', script], cwd=REPO_ROOT,
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr


def test_mcp_cards_show_the_schema_pin_and_what_each_server_costs_a_request():
    """#2332: the owner sees each server's weight and pins its call schemas from task start."""
    script = r'''
import assert from 'node:assert/strict';
import { applyMcpSettings, collectMcpSettings } from './web/modules/mcp_settings.js';
const host = { innerHTML: '', querySelectorAll: () => [] };
const status = { textContent: '', dataset: {} };
globalThis.document = { getElementById: (id) => id === 'mcp-servers-list' ? host : id === 'mcp-global-status' ? status : null,
    querySelector: () => null };
globalThis.fetch = async () => ({ ok: true, json: async () => ({ enabled: true, sdk_available: true, servers: [
    { id: 'gitlab', tool_count: 99, map_tokens: 2400, schema_tokens: 31000, schemas_at_start: false },
    { id: 'slack', tool_count: 11, map_tokens: 120, schema_tokens: 4600, schemas_at_start: true }],
    weights: { tools: 110, map_tokens: 2600, schema_tokens_from_task_start: 4600 } }) });
const servers = [{ id: 'gitlab', url: 'https://g.test/mcp', enabled: true },
                 { id: 'slack', url: 'https://s.test/mcp', enabled: true, schemas_at_start: 'true' }];
applyMcpSettings({ MCP_ENABLED: true, MCP_SERVERS: servers });
for (let i = 0; i < 5; i++) await new Promise((resolve) => setTimeout(resolve, 0));
applyMcpSettings({ MCP_ENABLED: true, MCP_SERVERS: servers });
assert.match(host.innerHTML, /data-mcp-field="schemas_at_start"/);
assert.match(host.innerHTML, /Connections map ≈2\.4k tokens in every Low\/Max request; call schemas ≈31\.0k tokens, loaded when Ouroboros takes up this server\./);
assert.match(host.innerHTML, /call schemas ≈4\.6k tokens in every Low\/Max request from task start/);
assert.match(status.textContent, /Connections map ≈2\.6k tokens in every Low\/Max request; call schemas from task start ≈4\.6k tokens\./);
const saved = collectMcpSettings().MCP_SERVERS;
assert.equal(saved[0].schemas_at_start, false);
assert.equal(saved[1].schemas_at_start, true);
'''
    result = subprocess.run([_node_bin(), '--input-type=module', '-e', script], cwd=REPO_ROOT,
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
