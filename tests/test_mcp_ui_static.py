"""Static UI invariants for the MCP Servers widget.

These tests do not launch a browser — they read ``settings_ui.js``,
``settings.js``, ``mcp_settings.js``, and ``settings.css`` and assert
the structural facts the implementation relies on (element ids, CSS
class names, key API endpoints, secret masking, etc.).
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess

import pytest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
WEB = REPO_ROOT / "web" / "modules"


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
    assert "server_id: sid, server: { ...server }" in mcp_source


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
    assert ".settings-shell .form-field textarea" in settings_css


def test_settings_ui_mcp_section_describes_hot_reload(settings_ui_source: str) -> None:
    """The UI copy must communicate that MCP changes are hot-reloadable
    (it's the ergonomic difference vs A2A which requires restart)."""
    assert "Hot-reloadable" in settings_ui_source or "hot-reloadable" in settings_ui_source
    assert "untrusted third-party data" in settings_ui_source


def test_mcp_ui_roundtrip_preserves_environment_and_unsupported_fields():
    """Execute the real JS projection; this is not a browser/visual receipt."""
    node = shutil.which("node")
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
    env_from_settings: { TOKEN: 'CUSTOM_MCP_KEY' }, future_field: { retained: true },
    url: 'https://unsupported-for-stdio.example/mcp' };
applyMcpSettings({ MCP_ENABLED: true, MCP_SERVERS: [server] });
const saved = collectMcpSettings().MCP_SERVERS[0];
for (const key of Object.keys(server)) assert.deepEqual(saved[key], server[key]);
assert.match(host.innerHTML, /data-mcp-field="cwd"/);
assert.match(host.innerHTML, /data-mcp-field="env_from_settings"/);
assert.match(host.innerHTML, /Unsupported fields retained:.*future_field/);
assert.ok(!host.innerHTML.includes('value="/project/"quoted""'));
applyMcpSettings({ MCP_SERVERS: [{ ...server, env_from_settings: '{unfinished', args: 'unsupported' }] });
assert.equal(collectMcpSettings().MCP_SERVERS[0].env_from_settings, '{unfinished');
assert.equal(collectMcpSettings().MCP_SERVERS[0].args, 'unsupported');
'''
    result = subprocess.run([node, "--input-type=module", "-e", script], cwd=REPO_ROOT,
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
