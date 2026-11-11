"""The tool servers of Settings › Tools without a browser (web/modules/mcp_settings.js).

Each test runs the real module in Node over the repo's DOM stand-in (web/tests/dashboard_dom_fixture.js): the page
is mounted the way the shell mounts it (``settingsPageFor(path).mount``), a server's form is edited through its own
handlers, and what its Save would send is read from the module. Layout and a real engine are the browser lane's
(tests/test_settings_tools_models_browser.py).
"""

from __future__ import annotations

import json
import pathlib
import shutil
import subprocess

import pytest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
WEB = REPO_ROOT / "web" / "modules"


def _node_bin():
    bundled = pathlib.Path.home() / ".claudexor" / "node" / "bin" / "node"
    return str(bundled if bundled.exists() else pathlib.Path(shutil.which("node") or "node"))


@pytest.fixture(scope="module")
def mcp_source() -> str:
    return (WEB / "mcp_settings.js").read_text(encoding="utf-8")


# The page on the DOM stand-in: `open(path, settings, status)` mounts a Tools page over a fake settings read and
# tool status; `page.type`, `page.tick` and `page.press` use the form's own handlers.
PRELUDE = r"""
import assert from 'node:assert/strict';
import { NodeStub, setup } from './web/tests/dashboard_dom_fixture.js';
import { applyMcpSettings, collectMcpSettings, validateMcpSettings } from './web/modules/mcp_settings.js';
import { forgetSettingRows, readSettings } from './web/modules/settings_controls.js';
import { settingsPageFor } from './web/modules/settings_pages.js';
const env = setup({ after() {} });
// The stand-in has no `after`: a revealed secret stands after its field (settings_secrets.js).
NodeStub.prototype.after = function after(node) { this.parentElement.insertBefore(node, this.parentElement.children[this.parentElement.children.indexOf(this) + 1] || null); };
const json = (data, status = 200) => ({ ok: status < 400, status, json: async () => structuredClone(data) });
const settle = async (rounds = 12) => { for (let i = 0; i < rounds; i += 1) await new Promise((resolve) => setImmediate(resolve)); };
const server = { settings: { MCP_SERVERS: [] }, status: null, posts: [], tests: [] };
env.routes.set('/api/settings', async (_url, init = {}) => {
    if (init.method !== 'POST') return json(server.settings);
    server.posts.push(JSON.parse(init.body));
    return json({ status: 'saved' });
});
env.routes.set('/api/mcp/status', async () => (server.status ? json(server.status) : { ok: false }));
env.routes.set('/api/mcp/test', async (_url, init) => { server.tests.push(JSON.parse(init.body)); return json({ ok: true, tool_count: 1 }); });
let mounted = null;
async function open(path, settings, status = null) {
    mounted?.unmount?.();
    forgetSettingRows();
    applyMcpSettings({ MCP_SERVERS: [] });  // a form's unsaved text is kept across visits: each scene starts without one
    Object.assign(server, { settings, status });
    const container = new NodeStub();
    mounted = await settingsPageFor(path).mount(container, { path, row: '', ctx: { readSettings, openSettings: () => true, ask: async () => true, state: {} } });
    await settle();
    const all = (selector) => container.querySelectorAll(selector);
    const field = (name) => all('[data-mcp-field]').find((node) => node.dataset.mcpField === name) || null;
    return { container, all, field, text: () => container.textContent, result: () => container.querySelector('[data-editor-result]').textContent,
        async type(name, value) { const input = field(name); input.value = value; await input.fire('input'); },
        async tick(name, on, event = 'input') { const input = field(name); input.checked = on; await input.fire(event); return input; },
        async press(selector) { const node = container.querySelector(selector); await node.fire('click', { target: node }); await settle(); } };
}
const saved = (index = 0) => collectMcpSettings().MCP_SERVERS[index];
"""


def _run(script: str) -> str:
    result = subprocess.run([_node_bin(), "--input-type=module", "-e", PRELUDE + script], cwd=REPO_ROOT,
                            capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_mcp_module_drops_masked_token_in_test_payload(mcp_source: str) -> None:
    """A masked token must not be sent as a Bearer credential when the user
    hits Test without re-typing it, unless paired with server_id
    so the backend can rehydrate the persisted token."""
    assert "looksMasked" in mcp_source
    assert "looksMasked(server.auth_token) ? { ...server, auth_token: '' }" in mcp_source
    assert "{ server_id: saved.identity, server: candidate }" in mcp_source


def test_mcp_ui_roundtrip_preserves_environment_and_unsupported_fields():
    """Execute the real JS projection; this is not a browser/visual receipt."""
    _run(r"""
const row = { id: 'selected', transport: 'stdio', command: 'python',
    args: ['exact argument', ''], enabled: true, cwd: '/project/"quoted"',
    env: { PORT: '8080' }, env_from_settings: { TOKEN: 'CUSTOM_MCP_KEY' }, future_field: { retained: true },
    url: 'https://unsupported-for-stdio.example/mcp' };
const page = await open('tools/selected', { MCP_ENABLED: true, MCP_SERVERS: [{ ...row, auth_configured: false }] });
for (const key of Object.keys(row)) assert.deepEqual(saved()[key], row[key]);
for (const name of ['cwd', 'env_from_settings', 'env', 'args', 'command']) assert.ok(page.field(name), name);
assert.match(page.text(), /Fields kept but not used:.*future_field/);
assert.ok(!page.text().includes('auth_configured'));
assert.ok(!Object.hasOwn(saved(), 'auth_configured'));
assert.equal(page.field('cwd').getAttribute('value'), '/project/"quoted"', 'a quote in a value never ends its attribute');
applyMcpSettings({ MCP_SERVERS: [{ ...row, env_from_settings: '{unfinished', args: 'unsupported' }] });
assert.equal(saved().env_from_settings, '{unfinished');
assert.equal(saved().args, 'unsupported');
""")


def test_mcp_ui_keeps_call_lifetime_flags_as_applied_fields():
    """``task_session``/``browser_policy`` round-trip through the real projection and are not 'kept but not used'."""
    _run(r"""
const row = { id: 'chrome', transport: 'stdio', command: 'node', args: ['server.mjs'], enabled: true,
    task_session: true, browser_policy: true };
const page = await open('tools/chrome', { MCP_ENABLED: true, MCP_SERVERS: [row, { ...row, id: 'plain', task_session: false,
    browser_policy: false }] });
assert.deepEqual(collectMcpSettings().MCP_SERVERS.map((each) => [each.task_session, each.browser_policy]), [[true, true], [false, false]]);
assert.ok(!page.text().includes('Fields kept but not used'), page.text());
""")


def test_mcp_ui_will_not_save_held_flags_on_a_remote_transport_until_they_are_removed():
    """Transport select → the form's notice → Save's refusal → Remove unsupported fields, through the real handlers.

    Import cannot reach this form (it changes no transport and refuses the flags off stdio:
    tests/test_mcp_import.py); false/false on a remote server stays equivalent to omission.
    On stdio the lifetime tick edits the form and its note in place, and an imported browser policy holds it on.
    The rendered run that switched back to stdio with its URL still typed (Save then wrote a
    row the loader refused, and its tools disappeared) is replayed last: each payload the real
    writer would send goes to the real loader, and Save refuses exactly the ones it drops."""
    out = _run(r"""
const HELD = { id: 'held', enabled: true, transport: 'stdio', command: 'node', args: ['server.mjs'],
    task_session: true, browser_policy: true };
const notice = (page) => (page.container.querySelector('[data-mcp-unsupported]')?.textContent || '').replace('Fields kept but not used: ', '');
let page = await open('tools/held', { MCP_SERVERS: [HELD] });
assert.equal(notice(page), '');
assert.equal(validateMcpSettings().length, 0);
await page.type('transport', 'streamable_http');
assert.match(notice(page), /task_session, browser_policy/);
let errors = validateMcpSettings();
assert.equal(errors.length, 1);
assert.equal(errors[0].input, page.field('transport'));
assert.match(errors[0].message, /task_session and browser_policy apply only to/);
// Save says the refusal beside itself and sends nothing.
await page.press('[data-editor-save]');
assert.match(page.result(), /task_session and browser_policy apply only to/);
assert.equal(server.posts.length, 0);
await page.type('transport', 'stdio');  // a display or a passing switch clears nothing
assert.deepEqual([saved().task_session, saved().browser_policy], [true, true]);
assert.equal(validateMcpSettings().length, 0);
await page.type('transport', 'sse');
await page.type('url', 'https://example.test/mcp');
assert.equal(validateMcpSettings().length, 1);
await page.press('[data-mcp-clear-unsupported]');
assert.ok(!Object.hasOwn(saved(), 'task_session') && !Object.hasOwn(saved(), 'browser_policy'), JSON.stringify(saved()));
assert.equal(saved().transport, 'sse');
assert.equal(validateMcpSettings().length, 0);
assert.equal(notice(page), '');
assert.equal(page.all('[data-mcp-lifetime]').length, 0);
page = await open('tools/held', { MCP_SERVERS: [HELD, { ...HELD, id: 'policy-off', browser_policy: false }] });
assert.equal(page.all('[data-mcp-lifetime]').length, 1);  // a held stdio form says what it does
assert.match(page.text(), /browser_policy:/);
assert.doesNotMatch((await open('tools/policy-off', server.settings)).text(), /browser_policy:/);
page = await open('tools/plain', { MCP_SERVERS: [{ id: 'plain', enabled: true, transport: 'stdio', command: 'node', args: ['server.mjs'],
    env_from_settings: { TOKEN: 'CUSTOM_KEY' }, allowed_tools: ['read'] }] });
let session = page.field('task_session');
assert.ok(!session.hasAttribute('checked') && !session.disabled);  // per-call stays the default
assert.equal(session.getAttribute('aria-describedby'), 'mcp-0-lifetime');
const note = page.container.querySelector('[data-mcp-lifetime]');
assert.match(note.textContent, /^Each tool call starts a new server process\./);
assert.ok(!Object.hasOwn(saved(), 'task_session'));
const sent = env.calls.length;
for (const [on, event] of [[true, 'input'], [false, 'change'], [true, 'change'], [false, 'input']]) {
    assert.equal(await page.tick('task_session', on, event), session, event);  // edited in place: the focused tick keeps its node
    assert.equal(saved().task_session, on, event);
    assert.deepEqual([saved().env_from_settings, saved().allowed_tools], [{ TOKEN: 'CUSTOM_KEY' }, ['read']], event);
    assert.match(note.textContent, on ? /^One server process serves every call in a task\./ : /^Each tool call starts/);
}
assert.equal(page.container.querySelector('[data-editor-discard]').hasAttribute('hidden'), false, 'the form says it holds unsaved text');
assert.equal(env.calls.length, sent, 'no edit writes or reads by itself');
page = await open('tools/held', { MCP_SERVERS: [{ ...HELD, task_session: 'TRUE', browser_policy: 'yes' }] });  // the loader's spellings
session = page.field('task_session');
assert.ok(session.hasAttribute('checked') && session.disabled);
assert.match(page.text(), /requires a task session; use Import to change that policy explicitly/);
for (const event of ['input', 'change']) {  // a stale or synthetic event cannot drop the imported policy's lifetime
    await page.tick('task_session', false, event);
    assert.equal(session.checked, true, event);
    assert.deepEqual([saved().task_session, saved().browser_policy], ['TRUE', 'yes'], event);
}
assert.equal(page.container.querySelector('[data-editor-discard]').hasAttribute('hidden'), true, 'nothing was edited');
for (const off of [false, 'false', '', 0, null]) {  // false/false on a remote server equals omission
    page = await open('tools/remote', { MCP_SERVERS: [{ id: 'remote', enabled: true, transport: 'streamable_http',
        url: 'https://example.test/mcp', task_session: off, browser_policy: off }] });
    assert.equal(notice(page), '', String(off));
    assert.equal(validateMcpSettings().length, 0, String(off));
    assert.equal(saved().task_session, off);
}
page = await open('tools/remote', { MCP_SERVERS: [{ id: 'remote', transport: 'sse', url: 'https://example.test/mcp', task_session: 'true' }] });
assert.match(notice(page), /task_session/);  // the loader reads 'true' as on
assert.equal(validateMcpSettings().length, 1);
const snaps = [];
const snap = (label) => snaps.push({ label, refused: validateMcpSettings().map((error) => error.message), server: saved() });
page = await open('tools/held', { MCP_SERVERS: [HELD] });  // the rendered run: stdio → HTTP with a URL, refused, back to stdio
await page.type('transport', 'streamable_http');
await page.type('url', 'http://127.0.0.1:9/mcp');
snap('http-with-held-flags');
await page.type('transport', 'stdio');
assert.equal(notice(page), 'url');
errors = validateMcpSettings();
assert.equal(errors.length, 1);
assert.equal(errors[0].input, page.field('transport'));
assert.match(errors[0].message, /^url is not used by Local process \(stdio\) servers, so the server would not load with it\. /);
assert.equal(saved().url, 'http://127.0.0.1:9/mcp');  // refused, never dropped
snap('back-on-stdio-with-url');
await page.press('[data-mcp-clear-unsupported]');  // only the URL goes: on stdio the held flags are applied fields
assert.deepEqual([saved().transport, saved().url, saved().task_session, saved().browser_policy], ['stdio', '', true, true]);
assert.equal(validateMcpSettings().length, 0);
snap('back-on-stdio-url-removed');
page = await open('tools/held', { MCP_SERVERS: [{ ...HELD, future_field: { retained: true } }] });
assert.equal(notice(page), 'future_field');  // shown as before; the loader keeps it with a warning
assert.equal(validateMcpSettings().length, 0);
snap('unknown-field-kept');
console.log(JSON.stringify(snaps));
""")
    from ouroboros import mcp_client

    loaded = {}
    for snap in json.loads(out.strip().splitlines()[-1]):  # what the real writer would send, as loaded
        errors = []
        loaded[snap["label"]] = (mcp_client.normalize_server_config(snap["server"], errors=errors), errors)
        assert (loaded[snap["label"]][0] is None) == bool(snap["refused"]), (snap, errors)  # Save refuses the drops
    assert list(loaded) == ["http-with-held-flags", "back-on-stdio-with-url", "back-on-stdio-url-removed",
                            "unknown-field-kept"]
    assert loaded["back-on-stdio-with-url"][1] == ["MCP_CONFIG_ERROR: fields unsupported by this transport: url"]
    removed = loaded["back-on-stdio-url-removed"][0]
    assert (removed.transport, removed.task_session, removed.browser_policy) == ("stdio", True, True)
    assert loaded["unknown-field-kept"][0].configuration_warnings  # kept with the loader's warning, never refused


def test_mcp_test_button_preserves_saved_url_only_credentials():
    """Exercise the registered click handler, not a copy of its payload expression."""
    _run(r"""
for (const [row, page_id, selected] of [
    [{ id: 'saved', url: 'https://***@host.test/mcp', auth_token: '' }, 'saved', 'saved'],
    [{ id: 'saved', url: 'https://host.test/mcp', auth_token: '***' }, 'saved', 'saved'],
    // A saved server identified by its name alone still names its saved identity.
    [{ name: 'Name Only', url: 'https://host.test/mcp', headers: { 'X-Key': '***set***' } }, 'name_only', 'Name Only'],
    // A plain saved server sends its fields alone: nothing of it is kept on the server's side.
    [{ id: 'plain', url: 'https://new:credential@host.test/mcp', auth_token: '' }, 'plain', undefined],
]) {
    const page = await open(`tools/${page_id}`, { MCP_SERVERS: [row] });
    await page.press('[data-mcp-test]');
    const body = server.tests.at(-1);
    assert.equal(body.server.url, row.url);
    assert.equal(body.server_id, selected);
    assert.match(page.result(), /The connection works\. 1 tool discovered\. No tool was called\./);
    assert.equal(page.container.querySelector('[data-mcp-test]').disabled, false);
}
// A server not saved yet has no saved identity to name.
const fresh = await open('tools/new', { MCP_SERVERS: [] });
await fresh.type('name', 'Fresh');
await fresh.type('url', 'https://host.test/mcp');
await fresh.press('[data-mcp-test]');
assert.equal(server.tests.at(-1).server_id, undefined);
assert.equal(server.tests.at(-1).server.url, 'https://host.test/mcp');
assert.equal(server.posts.length, 0, 'Test saves nothing');
""")


def test_mcp_headers_ui_preserves_malformed_rows_and_literal_values():
    """Execute the actual projection, including entries the list cannot draw as a server."""
    _run(r"""
const entries = [null, 'malformed legacy row', {id:'old', headers:'***set***'},
 {id:'multi', headers:{Authorization:'***set***','X-Key':'literal...','X-Empty':''}, future_field:{kept:true}}];
const list = await open('tools', { MCP_SERVERS: entries });
const out = collectMcpSettings().MCP_SERVERS;
assert.deepEqual(out.slice(0,2), entries.slice(0,2));
assert.equal(out[2].headers, entries[2].headers);
assert.deepEqual(out[3].headers, entries[3].headers);
assert.deepEqual(out[3].future_field, entries[3].future_field);
assert.equal(list.all('[data-mcp-invalid]').length, 2);
assert.match(list.text(), /Entry 1 cannot be read.*It is kept as it is until you remove it\./s);
assert.deepEqual(list.all('[data-entry-id]').map((node) => node.dataset.entryId), ['old', 'multi']);
assert.match((await open('tools/old', { MCP_SERVERS: entries })).text(), /Saved headers are malformed/);
""")


def test_mcp_cards_show_the_schema_pin_and_what_each_server_costs_a_request():
    """#2332: the owner sees each server's weight and pins its call schemas from task start."""
    _run(r"""
const status = { enabled: true, sdk_available: true, servers: [
    { id: 'gitlab', tool_count: 99, map_tokens: 2400, schema_tokens: 31000, schemas_at_start: false },
    { id: 'slack', tool_count: 11, map_tokens: 120, schema_tokens: 4600, schemas_at_start: true }],
    weights: { tools: 110, map_tokens: 2600, schema_tokens_from_task_start: 4600 } };
const settings = { MCP_ENABLED: true, MCP_SERVERS: [{ id: 'gitlab', url: 'https://g.test/mcp', enabled: true },
    { id: 'slack', url: 'https://s.test/mcp', enabled: true, schemas_at_start: 'true' }] };
const weight = (page) => page.container.querySelector('[data-mcp-weight]').textContent;
const gitlab = await open('tools/gitlab', settings, status);
assert.ok(gitlab.field('schemas_at_start'));
assert.equal(weight(gitlab), 'Connections map ≈2.4k tokens in every Low/Max request; call schemas ≈31.0k tokens, loaded when Ouroboros takes up this server.');
const slack = await open('tools/slack', settings, status);
assert.match(weight(slack), /call schemas ≈4\.6k tokens in every Low\/Max request from task start/);
assert.ok(slack.field('schemas_at_start').hasAttribute('checked'));
assert.equal(saved(0).schemas_at_start, false);
assert.equal(saved(1).schemas_at_start, true);
// Unpinning says at once what the request then carries, before any Save.
await slack.tick('schemas_at_start', false, 'change');
assert.match(weight(slack), /call schemas ≈4\.6k tokens, loaded when Ouroboros takes up this server\./);
assert.equal(saved(1).schemas_at_start, false);
const list = await open('tools', settings, status);
assert.equal(list.container.querySelector('[data-mcp-global]').textContent,
    '2 servers configured, 110 tools discovered. Connections map ≈2.6k tokens in every Low/Max request; call schemas from task start ≈4.6k tokens.');
""")


def test_mcp_cards_show_a_remote_servers_sign_in_with_one_fact_in_view():
    """A remote server that asked for sign-in (``mcp_oauth``) shows why and one button; the paste and the
    owner's own app registration stay under Details; a signed-in server offers Sign out; others show nothing."""
    _run(r"""
const status = { enabled: true, sdk_available: true, servers: [
    { id: 'superset', tool_count: 0, sign_in: { state: 'sign_in_required', reason: 'refresh_rejected', since: 'x', issuer: 'https://a.test/' } },
    { id: 'docs', tool_count: 3, sign_in: { state: 'signed_in', since: 'x', issuer: 'https://d.test/' } },
    { id: 'plain', tool_count: 2 }] };
const settings = { MCP_ENABLED: true, MCP_SERVERS: [{ id: 'superset', url: 'https://s.test/mcp', enabled: true },
    { id: 'docs', url: 'https://d.test/mcp', enabled: true }, { id: 'plain', url: 'https://p.test/mcp', enabled: true }] };
const strip = (page) => page.container.querySelector('[data-mcp-sign-in]');
const actions = (page) => strip(page).querySelectorAll('[data-mcp-sign-in-action]').map((node) => node.dataset.mcpSignInAction);
const asked = await open('tools/superset', settings, status);
assert.match(strip(asked).textContent, /Sign-in needed: the provider no longer accepts the sign-in/);
assert.deepEqual(actions(asked), ['begin', 'complete']);
const details = strip(asked).querySelector('.mcp-sign-in-details');
assert.ok(details.querySelector('[data-mcp-sign-in-returned]') && details.querySelector('[data-mcp-sign-in-client-id]'), 'the paste and the app registration are under Details');
assert.equal(details.querySelectorAll('[data-mcp-sign-in-action]')[0].dataset.mcpSignInAction, 'complete');
const signed = await open('tools/docs', settings, status);
assert.match(strip(signed).textContent, /Signed in/);
assert.deepEqual(actions(signed), ['sign_out']);
assert.equal(strip(signed).querySelector('[data-mcp-sign-in-returned]'), null);
const plain = await open('tools/plain', settings, status);
assert.ok(strip(plain).hidden && strip(plain).dataset.signInKey === '');
// The list offers Sign in on the head of the one server whose status asks for it.
const list = await open('tools', settings, status);
assert.deepEqual(list.all('[data-entry-id]').map((head) => [head.dataset.entryId, head.querySelector('[data-entry-sign-in]')?.textContent || '']),
    [['superset', 'Sign in'], ['docs', ''], ['plain', '']]);
""")


def test_mcp_sign_in_row_follows_the_saved_identity_and_offers_a_step_up():
    """MCP_RV1: a server saved without an explicit id (identity from its name) finds its status row, and a
    grant that needs a wider scope stays signed in while offering Sign in again."""
    _run(r"""
const status = { enabled: true, sdk_available: true, servers: [
    { id: 'superset', tool_count: 0, sign_in: { state: 'sign_in_required', reason: 'not_signed_in', since: 'x' } },
    { id: 'files', tool_count: 4, sign_in: { state: 'signed_in', reason: 'insufficient_scope', required_scope: 'files:read files:write', since: 'x' } }] };
const settings = { MCP_ENABLED: true, MCP_SERVERS: [
    { id: '', name: 'Superset', url: 'https://s.example/mcp', enabled: true },
    { id: 'files', url: 'https://f.example/mcp', enabled: true }] };
const strip = (page) => page.container.querySelector('[data-mcp-sign-in]');
const actions = (page) => strip(page).querySelectorAll('[data-mcp-sign-in-action]').map((node) => `${node.dataset.mcpSignInAction}:${node.textContent}`);
const named = await open('tools/superset', settings, status);  // its page is found by the identity its name gives
assert.match(strip(named).textContent, /Sign-in needed: not signed in yet/);
assert.equal(actions(named)[0], 'begin:Sign in');
const wider = await open('tools/files', settings, status);
assert.match(strip(wider).textContent, /Signed in; a call needs more access: files:read files:write/);
assert.deepEqual(actions(wider), ['begin:Sign in again', 'sign_out:Sign out', 'complete:Finish sign-in']);  // a step-up finishes by paste too
assert.ok(strip(wider).querySelector('[data-mcp-sign-in-returned]'));
const list = await open('tools', settings, status);
assert.deepEqual(list.all('[data-entry-sign-in]').map((button) => button.textContent), ['Sign in', 'Sign in again']);
""")
